// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`EinFoldExec`], the physical operator that computes a
//! [`FoldNode`](crate::FoldNode) or a [`PipelineNode`](crate::PipelineNode).

use std::fmt;
use std::future::Future;
use std::sync::{Arc, OnceLock};

use datafusion::arrow::array::{ArrayRef, BooleanArray, RecordBatch};
use datafusion::arrow::compute::nullif;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::error::ArrowError;
use datafusion::common::runtime::SpawnedTask;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{internal_err, DataFusionError, Result};
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::execution::{SendableRecordBatchStream, SessionStateBuilder, TaskContext};
use datafusion::logical_expr::execution_props::ExecutionProps;
use datafusion::logical_expr::LogicalPlan;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{
    self, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    self, DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, Partitioning, PlanProperties,
};
use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};
use einfold_ir::{ColumnIndex, Fold, Pipeline};
use einfold_kernels::{Algorithm, Budget, DomainRows, Folded, Options, Relation};
use futures::stream::BoxStream;
use futures::{future, stream, StreamExt};

use crate::{fold_node, pipeline_node};
use crate::{EinFoldPlanner, HOST};

/// What an [`EinFoldExec`] computes.
#[derive(Clone, Debug)]
enum Work {
    Fold(Fold),
    Pipeline(Pipeline),
}

/// Computes a fold over a join (see [`FoldNode`](crate::FoldNode)) without
/// forming the join's rows: each joined pair updates its output group
/// directly. Or computes a fold pipeline (see
/// [`PipelineNode`](crate::PipelineNode)) without forming its producer's
/// results: each tile of them is folded into the consumer as it is made.
///
/// Its children are a fold's operands, then its domain if it has one, or a
/// pipeline's producer operands, then its values. It reads every partition of
/// each, and emits one partition, in batches of the session's batch size. It
/// runs `einfold_kernels`' algorithms on as many threads as the session's
/// target partitions: for a fold, the dense algorithm when the keys and rows
/// allow it, and the discrimination algorithm otherwise; for a pipeline, the
/// fused algorithm. No result depends on the number of threads or on the
/// order rows arrive in.
///
/// Like DataFusion's own operators, it reserves the memory it takes from the
/// session's memory pool: its inputs as it reads them, what the kernels
/// allocate before they allocate it, and its output. Where the pool can't
/// give it that, it runs its fallback, if it has one, as the host's spilling
/// operators would have run, and counts it in `memory_fallbacks`; without
/// one, it fails, as an operator that can't spill does. It runs the kernels
/// on a blocking thread, off the async runtime; and its
/// metrics, shown by `EXPLAIN ANALYZE`, count which algorithm ran
/// (`dense_folds`, `discrimination_folds`, `fused_pipelines`), the pairs of
/// tiles a pipeline visited of all it has (`tiles_visited`, `tiles`), its
/// compute time and its output rows.
///
/// Given the plan it replaced (see [`EinFoldExec::with_fallback`]), it runs
/// that plan where the dense algorithm doesn't apply, and counts it in
/// `host_plans`, so it is never slower than the host's own plan by more than
/// reading its inputs once and planning that plan. The fallback is kept as a
/// logical plan, and planned only when it runs: most never do, and planning
/// each would plan its subtree, the operator's inputs included, a second
/// time, which in a deep plan grows with the square of its depth. It is
/// planned with DataFusion's default physical optimizer rules, under the
/// configuration of the task that runs it.
#[derive(Debug)]
pub struct EinFoldExec {
    work: Work,
    inputs: Vec<Arc<dyn ExecutionPlan>>,
    /// The host's plan for the same rows, run instead of the discrimination
    /// algorithm, or where memory runs short, and planned only then.
    fallback: Option<Arc<LogicalPlan>>,
    /// The planning session's properties, such as its variable providers,
    /// for planning the fallback as the session would.
    execution_props: Option<Arc<ExecutionProps>>,
    /// Whether the fallback runs wherever the dense algorithm declines, and
    /// not only where memory runs short.
    host_on_decline: bool,
    /// Whether to run the fallback unless no two domain rows reach one group.
    domain_rows: DomainRows,
    /// The domain's Boolean column that makes a row's value NULL where true.
    null_when: Option<ColumnIndex>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl EinFoldExec {
    /// The operator computing `fold` over `inputs` (operands, then domain),
    /// producing `schema`, which [`FoldNode`](crate::FoldNode) derives.
    #[must_use]
    pub fn new(fold: Fold, inputs: Vec<Arc<dyn ExecutionPlan>>, schema: SchemaRef) -> Self {
        EinFoldExec::computing(Work::Fold(fold), inputs, schema)
    }

    /// The operator computing `pipeline` over `inputs` (its producer's two
    /// operands, then its values), producing `schema`, which
    /// [`PipelineNode`](crate::PipelineNode) derives. The fused algorithm
    /// declines inputs it can't compute, so the operator needs a fallback
    /// (see [`EinFoldExec::with_fallback`]).
    #[must_use]
    pub fn for_pipeline(
        pipeline: Pipeline,
        inputs: Vec<Arc<dyn ExecutionPlan>>,
        schema: SchemaRef,
    ) -> Self {
        EinFoldExec::computing(Work::Pipeline(pipeline), inputs, schema)
    }

    fn computing(work: Work, inputs: Vec<Arc<dyn ExecutionPlan>>, schema: SchemaRef) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        EinFoldExec {
            work,
            inputs,
            fallback: None,
            execution_props: None,
            host_on_decline: false,
            domain_rows: DomainRows::Any,
            null_when: None,
            properties: Arc::new(properties),
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// This operator, running `plan` when the dense algorithm doesn't apply,
    /// instead of the discrimination algorithm. `plan` must produce the
    /// operator's columns, in order, under any names. It is planned only if
    /// it runs.
    #[must_use]
    pub fn with_fallback(mut self, plan: LogicalPlan) -> Self {
        self.fallback = Some(Arc::new(plan));
        self.host_on_decline = true;
        self
    }

    /// This operator, running `plan` only where the session's memory pool
    /// can't give it what it needs, and the discrimination algorithm where
    /// the dense one declines. `plan` must produce the operator's columns, in
    /// order, under any names. It is planned only if it runs.
    #[must_use]
    pub fn with_memory_fallback(mut self, plan: LogicalPlan) -> Self {
        self.fallback = Some(Arc::new(plan));
        self.host_on_decline = false;
        self
    }

    /// This operator, planning its fallback with `props`, the planning
    /// session's: its variable providers (`@name` in SQL), for one.
    #[must_use]
    pub fn with_execution_props(mut self, props: ExecutionProps) -> Self {
        self.execution_props = Some(Arc::new(props));
        self
    }

    /// This operator, computing its fold only when no two of its domain's
    /// rows reach one group, and running its fallback otherwise. It needs a
    /// fallback (see [`EinFoldExec::with_fallback`]).
    #[must_use]
    pub fn requiring_unique_domain(mut self) -> Self {
        self.domain_rows = DomainRows::UniqueKeys;
        self
    }

    /// This operator, giving a NULL value to each row of its domain where the
    /// domain's Boolean column `column` is true, as `CASE WHEN <column> THEN
    /// NULL ELSE <value> END` does. The host's plan, if it runs, computes its
    /// own values.
    #[must_use]
    pub fn with_null_when(mut self, column: ColumnIndex) -> Self {
        self.null_when = Some(column);
        self
    }

    /// The host's plan this operator runs where the dense algorithm doesn't
    /// apply, if it has one.
    #[must_use]
    pub fn fallback(&self) -> Option<&LogicalPlan> {
        self.fallback.as_deref()
    }

    /// This operator over `inputs`, with fresh metrics.
    fn copy(&self, inputs: Vec<Arc<dyn ExecutionPlan>>) -> EinFoldExec {
        let mut exec = EinFoldExec::computing(self.work.clone(), inputs, self.schema());
        exec.fallback = self.fallback.clone();
        exec.execution_props = self.execution_props.clone();
        exec.host_on_decline = self.host_on_decline;
        exec.domain_rows = self.domain_rows;
        exec.null_when = self.null_when;
        exec
    }

    /// The fold this operator computes, if it computes one.
    #[must_use]
    pub fn fold(&self) -> Option<&Fold> {
        match &self.work {
            Work::Fold(fold) => Some(fold),
            Work::Pipeline(_) => None,
        }
    }

    /// The pipeline this operator computes, if it computes one.
    #[must_use]
    pub fn pipeline(&self) -> Option<&Pipeline> {
        match &self.work {
            Work::Pipeline(pipeline) => Some(pipeline),
            Work::Fold(_) => None,
        }
    }
}

impl DisplayAs for EinFoldExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        match &self.work {
            Work::Fold(fold) => {
                write!(
                    f,
                    "EinFoldExec: {}",
                    fold_node::describe(fold, self.domain_rows)
                )
            }
            Work::Pipeline(p) => write!(f, "EinFoldExec: {}", pipeline_node::describe(p)),
        }
    }
}

/// `work`'s result. A plan's operators spawn tasks on a Tokio runtime, so
/// where none is current, as when a host in another library polls this
/// operator through DataFusion's FFI, `work` runs on one of this library's
/// own, made when first needed. Dropping the future, as a cancelled query
/// does, aborts that work.
async fn on_a_runtime<T: Send + 'static>(
    work: impl Future<Output = Result<T>> + Send + 'static,
) -> Result<T> {
    static RUNTIME: OnceLock<Option<tokio::runtime::Runtime>> = OnceLock::new();
    if tokio::runtime::Handle::try_current().is_ok() {
        return work.await;
    }
    let runtime = RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .ok()
    });
    let Some(runtime) = runtime else {
        return internal_err!("EinFoldExec could not start a Tokio runtime");
    };
    // Entered only to spawn: the task aborts when dropped.
    let task = {
        let _entered = runtime.enter();
        SpawnedTask::spawn(work)
    };
    task.join()
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?
}

/// The host's plan `plan`, planned for the task `ctx`: with its
/// configuration and runtime, the planning session's `props` if given, and
/// DataFusion's default physical optimizer rules. A fold node within it is
/// planned as einfold's operator, and only as that: as when einfold planned
/// the query, other extensions' nodes aren't planned.
async fn plan_host(
    plan: &LogicalPlan,
    ctx: &TaskContext,
    props: Option<&ExecutionProps>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let mut builder = SessionStateBuilder::new()
        .with_config(ctx.session_config().clone())
        .with_runtime_env(ctx.runtime_env())
        .with_default_features();
    if let Some(props) = props {
        builder = builder.with_execution_props(props.clone());
    }
    let state = builder.build();
    DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(EinFoldPlanner)])
        .create_physical_plan(plan, &state)
        .await
}

/// Every batch of every partition of the host's plan `plan`, planned for the
/// task `ctx` (see [`plan_host`]), in partition order, on a runtime (see
/// [`on_a_runtime`]).
async fn run_host_plan(
    plan: Arc<LogicalPlan>,
    ctx: Arc<TaskContext>,
    props: Option<Arc<ExecutionProps>>,
) -> Result<Vec<Vec<RecordBatch>>> {
    on_a_runtime(async move {
        let physical = plan_host(&plan, &ctx, props.as_deref()).await?;
        physical_plan::collect_partitioned(physical, ctx).await
    })
    .await
}

/// The rows of the host's plan `plan`, planned for the task `ctx` (see
/// [`plan_host`]), as one stream, its columns under this operator's names,
/// `out`, and counted in `output_rows`. They stream through, as they would
/// in the host's own plan, where a Tokio runtime is current; where none is,
/// they are collected on this library's runtime (see [`run_host_plan`]).
async fn host_stream(
    plan: Arc<LogicalPlan>,
    ctx: Arc<TaskContext>,
    props: Option<Arc<ExecutionProps>>,
    out: SchemaRef,
    output_rows: metrics::Count,
) -> Result<BoxStream<'static, Result<RecordBatch>>> {
    use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
    let rename = move |b: Result<RecordBatch>| -> Result<RecordBatch> {
        let b = RecordBatch::try_new(Arc::clone(&out), b?.columns().to_vec())?;
        output_rows.add(b.num_rows());
        Ok(b)
    };
    if tokio::runtime::Handle::try_current().is_err() {
        let batches = run_host_plan(plan, ctx, props).await?.concat();
        return Ok(stream::iter(batches.into_iter().map(Ok).map(rename)).boxed());
    }
    let physical = plan_host(&plan, &ctx, props.as_deref()).await?;
    let one: Arc<dyn ExecutionPlan> = match physical.properties().partitioning.partition_count() {
        1 => physical,
        _ => Arc::new(CoalescePartitionsExec::new(physical)),
    };
    Ok(one.execute(0, ctx)?.map(rename).boxed())
}

/// Whether `e` says that memory ran short: that the session's memory pool
/// refused a reservation, or that einfold's kernels found their budget too
/// small.
fn short_of_memory(e: &DataFusionError) -> bool {
    match e.find_root() {
        DataFusionError::ResourcesExhausted(_) => true,
        DataFusionError::ArrowError(e, _) => matches!(**e, ArrowError::MemoryError(_)),
        _ => false,
    }
}

/// The session's memory pool, as a budget for einfold's kernels: what they
/// ask for is reserved, as long as `0` is.
#[derive(Debug)]
struct Pool<'a>(&'a MemoryReservation);

impl Budget for Pool<'_> {
    fn grow(&self, bytes: usize) -> bool {
        self.0.try_grow(bytes).is_ok()
    }
}

/// `batches` as this operator's output stream: in batches of `batch_size`
/// rows at most, counted in `output_rows`. `reservation` counts them, and
/// gives back each batch's share as its last slice is emitted: the slices
/// share the batch's buffers, which this stream holds until then. From then
/// on, what reads it holds it, and counts it if it keeps it, as it would the
/// host's.
fn slices(
    batches: Vec<RecordBatch>,
    batch_size: usize,
    output_rows: &metrics::Count,
    reservation: Arc<MemoryReservation>,
) -> BoxStream<'static, Result<RecordBatch>> {
    let slices: Vec<(RecordBatch, usize)> = batches
        .iter()
        .inspect(|b| output_rows.add(b.num_rows()))
        .flat_map(|batch| {
            let (rows, bytes) = (batch.num_rows(), batch.get_array_memory_size());
            (0..rows).step_by(batch_size).map(move |start| {
                let n = batch_size.min(rows - start);
                let last = start + n == rows;
                (batch.slice(start, n), if last { bytes } else { 0 })
            })
        })
        .collect();
    stream::iter(slices)
        .map(move |(b, bytes)| {
            reservation.shrink(bytes.min(reservation.size()));
            Ok(b)
        })
        .boxed()
}

/// Every batch of every partition of `plan`, in partition order, each counted
/// against `reservation`. The partitions are read concurrently, since an
/// operator such as a hash repartition produces them together and would
/// otherwise buffer all but one. They run on a runtime (see
/// [`on_a_runtime`]), since an input's operators may spawn tasks too.
async fn collect(
    plan: Arc<dyn ExecutionPlan>,
    ctx: Arc<TaskContext>,
    reservation: &MemoryReservation,
) -> Result<Vec<RecordBatch>> {
    let batches = on_a_runtime(physical_plan::collect_partitioned(plan, ctx))
        .await?
        .concat();
    for b in &batches {
        reservation.try_grow(b.get_array_memory_size())?;
    }
    Ok(batches)
}

impl ExecutionPlan for EinFoldExec {
    fn name(&self) -> &'static str {
        "EinFoldExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.inputs.iter().collect()
    }

    /// None: the fold names its operands' columns by position, and holds no
    /// physical expressions.
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::UnspecifiedDistribution; self.children().len()]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false; self.children().len()]
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let expected = self.children().len();
        if children.len() != expected {
            return internal_err!(
                "EinFoldExec has {expected} children, given {}",
                children.len()
            );
        }
        Ok(Arc::new(self.copy(children)))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        use futures::TryStreamExt;
        if partition != 0 {
            return internal_err!("EinFoldExec has one partition, asked for {partition}");
        }
        let work = self.work.clone();
        let inputs = self.inputs.clone();
        let schema = self.schema();
        let expected = match &work {
            Work::Fold(fold) => 2 + usize::from(fold.domain().is_some()),
            Work::Pipeline(_) => 3,
        };
        if inputs.len() != expected {
            return internal_err!("EinFoldExec was given {} inputs", inputs.len());
        }
        if matches!(work, Work::Pipeline(_)) && self.fallback.is_none() {
            return internal_err!("EinFoldExec computes a pipeline only with a fallback");
        }
        let batch_size = context.session_config().batch_size().max(1);
        let options = Options {
            host: HOST,
            threads: context.session_config().target_partitions(),
            domain_rows: self.domain_rows,
            budget: None,
        };
        if self.domain_rows == DomainRows::UniqueKeys && self.fallback.is_none() {
            return internal_err!("EinFoldExec requires a unique domain, but has no fallback");
        }
        let builder = || MetricBuilder::new(&self.metrics);
        let dense_folds = builder().global_counter("dense_folds");
        let discrimination_folds = builder().global_counter("discrimination_folds");
        let fused_pipelines = builder().global_counter("fused_pipelines");
        let tiles_visited = builder().global_counter("tiles_visited");
        let tiles = builder().global_counter("tiles");
        let host_plans = builder().global_counter("host_plans");
        let memory_fallbacks = builder().global_counter("memory_fallbacks");
        let fallback = self.fallback.clone();
        let execution_props = self.execution_props.clone();
        let host_on_decline = self.host_on_decline;
        let null_when = self.null_when;
        let compute = builder().elapsed_compute(0);
        let output_rows = builder().output_rows(0);
        // Shared with the kernel's task, which holds the inputs it counts: a
        // query cancelled while the kernel runs drops this future, but the
        // task runs on, and its inputs stay counted until it ends.
        let reservation =
            Arc::new(MemoryConsumer::new("EinFoldExec").register(context.memory_pool()));
        let out = Arc::clone(&schema);
        let result = async move {
            // The host's plan, where memory ran short: whatever this operator
            // held is released first, and the plan's operators reserve their
            // own.
            let short = |e: DataFusionError| async {
                let Some(plan) = fallback.clone().filter(|_| short_of_memory(&e)) else {
                    return Err(e);
                };
                reservation.free();
                memory_fallbacks.add(1);
                host_plans.add(1);
                let (rows, props) = (output_rows.clone(), execution_props.clone());
                host_stream(plan, Arc::clone(&context), props, Arc::clone(&out), rows).await
            };
            // Every input is read whole, and stays in the batches it arrived
            // in: the kernels read them batch by batch.
            let schemas: Vec<SchemaRef> = inputs.iter().map(|i| i.schema()).collect();
            let reads =
                (inputs.into_iter()).map(|i| collect(i, Arc::clone(&context), &reservation));
            let read = match future::try_join_all(reads).await {
                Ok(read) => read,
                Err(e) => return short(e).await,
            };
            let mut read = read.into_iter();
            // Two operands, then a fold's domain or a pipeline's values, as
            // checked above.
            let (Some(a), Some(b), third) = (read.next(), read.next(), read.next()) else {
                return internal_err!("EinFoldExec read fewer than two inputs");
            };
            let domain = third.clone().filter(|_| matches!(work, Work::Fold(_)));
            // Each domain batch's NULL mask: the kernels give one output batch
            // per domain batch, row for row.
            let masks: Option<Vec<ArrayRef>> = null_when
                .zip(domain.as_ref())
                .map(|(c, d)| d.iter().map(|b| Arc::clone(b.column(c.0))).collect());
            let timer = compute.timer();
            let kernel_out = Arc::clone(&out);
            let held = Arc::clone(&reservation);
            let kernel = move || {
                let pool = Pool(&held);
                let options = Options {
                    budget: Some(&pool),
                    ..options
                };
                let (a, b) = (
                    Relation::new(&schemas[0], &a),
                    Relation::new(&schemas[1], &b),
                );
                let fold = match &work {
                    Work::Fold(fold) => fold,
                    Work::Pipeline(pipeline) => {
                        let values = third.as_deref().unwrap_or_default();
                        let v = Relation::new(&schemas[2], values);
                        let piped =
                            einfold_kernels::pipeline(pipeline, [a, b, v], kernel_out, options)?;
                        return Ok(piped.map(|p| {
                            let counts = (p.tiles_visited, p.tiles);
                            (
                                Folded {
                                    algorithm: Algorithm::Fused,
                                    batches: p.batches,
                                },
                                Some(counts),
                            )
                        }));
                    }
                };
                let domain = domain.as_deref().map(|d| Relation::new(&schemas[2], d));
                let tiles = None;
                match host_on_decline {
                    true => einfold_kernels::fold_dense(fold, a, b, domain, kernel_out, options)
                        .map(|r| {
                            r.map(|batches| {
                                let algorithm = Algorithm::Dense;
                                (Folded { algorithm, batches }, tiles)
                            })
                        }),
                    false => einfold_kernels::fold(fold, a, b, domain, kernel_out, options)
                        .map(|folded| Some((folded, tiles))),
                }
            };
            // On a blocking thread, off the async runtime; or, where there is
            // no Tokio runtime to spawn on, as when a host in another library
            // polls this plan through DataFusion's FFI, on this thread.
            let ran = match tokio::runtime::Handle::try_current() {
                Ok(_) => SpawnedTask::spawn_blocking(kernel)
                    .await
                    .map_err(|e| DataFusionError::External(Box::new(e)))?,
                Err(_) => kernel(),
            };
            timer.done();
            let ran = match ran {
                Ok(ran) => ran,
                Err(e) => return short(DataFusionError::from(e)).await,
            };
            // The kernel has dropped its inputs: release them, so that the
            // host's plan, if it runs, and the operators downstream have the
            // memory they would have had without einfold. The output, which
            // may share the domain's columns, is counted below.
            reservation.free();
            let masked = |batches: Vec<RecordBatch>| -> Result<Vec<RecordBatch>> {
                let Some(masks) = &masks else {
                    return Ok(batches);
                };
                (batches.iter().zip(masks))
                    .map(|(b, mask)| {
                        let mask = mask.as_any().downcast_ref::<BooleanArray>();
                        let Some(mask) = mask else {
                            return internal_err!("EinFoldExec's NULL mask isn't Boolean");
                        };
                        let mut columns = b.columns().to_vec();
                        let last = columns.len() - 1;
                        columns[last] = nullif(&columns[last], mask)?;
                        Ok(RecordBatch::try_new(b.schema(), columns)?)
                    })
                    .collect()
            };
            let batches = match (ran, fallback.clone()) {
                (Some((Folded { algorithm, batches }, counts)), _) => {
                    if let Some((visited, all)) = counts {
                        tiles_visited.add(visited);
                        tiles.add(all);
                    }
                    match algorithm {
                        Algorithm::Dense => dense_folds.add(1),
                        Algorithm::Discriminate => discrimination_folds.add(1),
                        Algorithm::Fused => fused_pipelines.add(1),
                        // An algorithm added later, which no metric counts yet.
                        _ => {}
                    }
                    masked(batches)?
                }
                (None, Some(plan)) => {
                    host_plans.add(1);
                    let rows = output_rows.clone();
                    let props = execution_props.clone();
                    return host_stream(plan, Arc::clone(&context), props, Arc::clone(&out), rows)
                        .await;
                }
                (None, None) => return internal_err!("EinFoldExec ran no algorithm"),
            };
            let size: usize = batches.iter().map(RecordBatch::get_array_memory_size).sum();
            if let Err(e) = reservation.try_grow(size) {
                drop(batches);
                return short(e).await;
            }
            Ok::<_, DataFusionError>(slices(batches, batch_size, &output_rows, reservation))
        };
        let stream = stream::once(result).try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream.boxed(),
        )))
    }
}
