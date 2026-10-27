// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`EinFoldExec`], the physical operator that computes a [`FoldNode`](crate::FoldNode).

use std::fmt;
use std::sync::{Arc, OnceLock};

use datafusion::arrow::array::{ArrayRef, BooleanArray, RecordBatch};
use datafusion::arrow::compute::nullif;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::runtime::SpawnedTask;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{internal_err, DataFusionError, Result};
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricBuilder, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    self, DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, Partitioning, PlanProperties,
};
use einfold_ir::{ColumnIndex, Fold};
use einfold_kernels::{Algorithm, DomainRows, Folded, Options, Relation};
use futures::{future, stream};

use crate::fold_node;
use crate::HOST;

/// Computes a fold over a join (see [`FoldNode`](crate::FoldNode)) without forming the join's
/// rows: each joined pair updates its output group directly.
///
/// Its children are the fold's operands, then its domain if it has one. It
/// reads every partition of each, and emits one partition, in batches of the
/// session's batch size. It runs `einfold_kernels`' dense algorithm, on as
/// many threads as the session's target partitions, when the keys and rows
/// allow it, and the discrimination algorithm otherwise. Neither result depends on the
/// number of threads or on the order rows arrive in.
///
/// Like DataFusion's own operators, it reserves the memory its inputs take
/// from the session's memory pool, and fails if the pool can't give it; it
/// runs the kernels on a blocking thread, off the async runtime; and its
/// metrics, shown by `EXPLAIN ANALYZE`, count which algorithm ran
/// (`dense_folds`, `discrimination_folds`), its compute time and its output rows.
///
/// Given the plan it replaced (see [`EinFoldExec::with_fallback`]), it runs
/// that plan where the dense algorithm doesn't apply, and counts it in
/// `host_plans`, so it is never slower than the host's own plan by more than
/// reading its inputs once. The fallback is planned and optimized when the
/// operator is, but isn't one of its children, so that the session's
/// physical optimizer doesn't optimize it a second time each time a query
/// is planned. Resetting the operator's state, as a recursive query does
/// before each iteration, resets the fallback's too.
#[derive(Debug)]
pub struct EinFoldExec {
    fold: Fold,
    inputs: Vec<Arc<dyn ExecutionPlan>>,
    /// The host's plan for the same rows, run instead of the discrimination algorithm.
    fallback: Option<Arc<dyn ExecutionPlan>>,
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
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        EinFoldExec {
            fold,
            inputs,
            fallback: None,
            domain_rows: DomainRows::Any,
            null_when: None,
            properties: Arc::new(properties),
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// This operator, running `plan` when the dense algorithm doesn't apply,
    /// instead of the discrimination algorithm. `plan` must produce the operator's
    /// columns, in order, under any names.
    pub fn with_fallback(mut self, plan: Arc<dyn ExecutionPlan>) -> Self {
        self.fallback = Some(plan);
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
    pub fn fallback(&self) -> Option<&Arc<dyn ExecutionPlan>> {
        self.fallback.as_ref()
    }

    /// This operator over `inputs`, with fresh metrics.
    fn copy(&self, inputs: Vec<Arc<dyn ExecutionPlan>>) -> EinFoldExec {
        let mut exec = EinFoldExec::new(self.fold.clone(), inputs, self.schema());
        exec.fallback = self.fallback.clone();
        exec.domain_rows = self.domain_rows;
        exec.null_when = self.null_when;
        exec
    }

    /// The fold this operator computes.
    #[must_use]
    pub fn fold(&self) -> &Fold {
        &self.fold
    }
}

impl DisplayAs for EinFoldExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "EinFoldExec: {}",
            fold_node::describe(&self.fold, self.domain_rows)
        )
    }
}

/// Every batch of every partition of the host's plan `plan`, in partition
/// order. Its operators spawn tasks on a Tokio runtime, so where none is
/// current, as when a host in another library polls this operator through
/// DataFusion's FFI, they run on one of this library's own, made when first
/// needed.
async fn run_host_plan(
    plan: Arc<dyn ExecutionPlan>,
    ctx: Arc<TaskContext>,
) -> Result<Vec<Vec<RecordBatch>>> {
    static RUNTIME: OnceLock<Option<tokio::runtime::Runtime>> = OnceLock::new();
    if tokio::runtime::Handle::try_current().is_ok() {
        return physical_plan::collect_partitioned(plan, ctx).await;
    }
    let runtime = RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .ok()
    });
    let Some(runtime) = runtime else {
        return internal_err!("EinFoldExec could not start a Tokio runtime for the host's plan");
    };
    runtime
        .spawn(physical_plan::collect_partitioned(plan, ctx))
        .await
        .map_err(|e| DataFusionError::External(Box::new(e)))?
}

/// Every batch of every partition of `plan`, in partition order, each counted
/// against `reservation`. The partitions are read concurrently, since an
/// operator such as a hash repartition produces them together and would
/// otherwise buffer all but one.
async fn collect(
    plan: Arc<dyn ExecutionPlan>,
    ctx: Arc<TaskContext>,
    reservation: &MemoryReservation,
) -> Result<Vec<RecordBatch>> {
    let batches = match tokio::runtime::Handle::try_current() {
        Ok(_) => physical_plan::collect_partitioned(plan, ctx)
            .await?
            .concat(),
        // No runtime to spawn on: the partitions concurrently, as futures on
        // this thread.
        Err(_) => {
            use futures::TryStreamExt;
            let partitions = plan.properties().partitioning.partition_count();
            let reads = (0..partitions).map(|p| {
                let stream = plan.execute(p, Arc::clone(&ctx));
                async move { stream?.try_collect::<Vec<_>>().await }
            });
            future::try_join_all(reads).await?.concat()
        }
    };
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

    /// This operator with its fallback's state reset too: the fallback isn't
    /// a child, so `reset_plan_states` doesn't reach it otherwise, and its
    /// operators, such as a hash repartition, would keep their first run's
    /// state.
    fn reset_state(self: Arc<Self>) -> Result<Arc<dyn ExecutionPlan>> {
        use datafusion::physical_plan::execution_plan::reset_plan_states;
        let mut exec = self.copy(self.inputs.clone());
        exec.fallback = exec.fallback.map(reset_plan_states).transpose()?;
        Ok(Arc::new(exec))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        use futures::{StreamExt, TryStreamExt};
        if partition != 0 {
            return internal_err!("EinFoldExec has one partition, asked for {partition}");
        }
        let fold = self.fold.clone();
        let inputs = self.inputs.clone();
        let schema = self.schema();
        if inputs.len() != 2 + usize::from(fold.domain().is_some()) {
            return internal_err!("EinFoldExec was given {} inputs", inputs.len());
        }
        let batch_size = context.session_config().batch_size().max(1);
        let options = Options {
            host: HOST,
            threads: context.session_config().target_partitions(),
            domain_rows: self.domain_rows,
        };
        if self.domain_rows == DomainRows::UniqueKeys && self.fallback.is_none() {
            return internal_err!("EinFoldExec requires a unique domain, but has no fallback");
        }
        let builder = || MetricBuilder::new(&self.metrics);
        let dense_folds = builder().global_counter("dense_folds");
        let discrimination_folds = builder().global_counter("discrimination_folds");
        let host_plans = builder().global_counter("host_plans");
        let fallback = self.fallback.clone();
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
            // Every input is read whole, and stays in the batches it arrived
            // in: the kernels read them batch by batch.
            let schemas: Vec<SchemaRef> = inputs.iter().map(|i| i.schema()).collect();
            let reads =
                (inputs.into_iter()).map(|i| collect(i, Arc::clone(&context), &reservation));
            let mut read = future::try_join_all(reads).await?.into_iter();
            // Two operands, then the domain if the fold has one, as checked
            // above.
            let (Some(a), Some(b), domain) = (read.next(), read.next(), read.next()) else {
                return internal_err!("EinFoldExec read fewer than two inputs");
            };
            // Each domain batch's NULL mask: the kernels give one output batch
            // per domain batch, row for row.
            let masks: Option<Vec<ArrayRef>> = null_when
                .zip(domain.as_ref())
                .map(|(c, d)| d.iter().map(|b| Arc::clone(b.column(c.0))).collect());
            let timer = compute.timer();
            let with_fallback = fallback.is_some();
            let kernel_out = Arc::clone(&out);
            let held = Arc::clone(&reservation);
            let kernel = move || {
                let _held = held;
                let (a, b) = (
                    Relation::new(&schemas[0], &a),
                    Relation::new(&schemas[1], &b),
                );
                let domain = domain.as_deref().map(|d| Relation::new(&schemas[2], d));
                match with_fallback {
                    true => einfold_kernels::fold_dense(&fold, a, b, domain, kernel_out, options)
                        .map(|r| {
                            r.map(|batches| Folded {
                                algorithm: Algorithm::Dense,
                                batches,
                            })
                        }),
                    false => {
                        einfold_kernels::fold(&fold, a, b, domain, kernel_out, options).map(Some)
                    }
                }
            };
            // On a blocking thread, off the async runtime; or, where there is
            // no Tokio runtime to spawn on, as when a host in another library
            // polls this plan through DataFusion's FFI, on this thread.
            let ran = match tokio::runtime::Handle::try_current() {
                Ok(_) => SpawnedTask::spawn_blocking(kernel)
                    .await
                    .map_err(|e| DataFusionError::External(Box::new(e)))??,
                Err(_) => kernel()?,
            };
            timer.done();
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
            let batches = match (ran, fallback) {
                (Some(Folded { algorithm, batches }), _) => {
                    match algorithm {
                        Algorithm::Dense => dense_folds.add(1),
                        Algorithm::Discriminate => discrimination_folds.add(1),
                        // An algorithm added later, which no metric counts yet.
                        _ => {}
                    }
                    masked(batches)?
                }
                (None, Some(plan)) => {
                    // The host's plan, its columns under this operator's names.
                    host_plans.add(1);
                    run_host_plan(plan, Arc::clone(&context))
                        .await?
                        .concat()
                        .into_iter()
                        .map(|b| RecordBatch::try_new(Arc::clone(&out), b.columns().to_vec()))
                        .collect::<std::result::Result<Vec<_>, _>>()?
                }
                (None, None) => return internal_err!("EinFoldExec ran no algorithm"),
            };
            for b in &batches {
                output_rows.add(b.num_rows());
                reservation.try_grow(b.get_array_memory_size())?;
            }
            let slices: Vec<Result<RecordBatch>> = batches
                .iter()
                .flat_map(|batch| {
                    (0..batch.num_rows()).step_by(batch_size).map(move |start| {
                        Ok(batch.slice(start, batch_size.min(batch.num_rows() - start)))
                    })
                })
                .collect();
            // The reservation lives as long as the output does.
            Ok::<_, DataFusionError>(stream::iter(slices).map(move |b| {
                let _ = &reservation;
                b
            }))
        };
        let stream = stream::once(result).try_flatten();
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            stream.boxed(),
        )))
    }
}
