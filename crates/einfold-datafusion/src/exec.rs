// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`EinFoldExec`], the physical operator that computes a [`FoldNode`], and
//! the planners that put it in DataFusion's physical plans.

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::compute::concat_batches;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::Session;
use datafusion::common::runtime::SpawnedTask;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::{internal_err, DataFusionError, Result};
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion::optimizer::{OptimizerContext, OptimizerRule};
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricBuilder, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    collect_partitioned, DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, Partitioning,
    PlanProperties,
};
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use einfold_ir::Fold;
use einfold_kernels::{Algorithm, Options};
use futures::future::try_join_all;
use futures::{stream, StreamExt, TryStreamExt};

use crate::node::{describe, FoldNode};
use crate::rule::DetectFolds;
use crate::HOST;

/// Computes a fold over a join (see [`FoldNode`]) without forming the join's
/// rows: each joined pair updates its output group directly.
///
/// Its children are the fold's operands, then its domain if it has one. It
/// reads every partition of each, and emits one partition, in batches of the
/// session's batch size. It runs `einfold_kernels`' dense algorithm, on as
/// many threads as the session's target partitions, when the keys and rows
/// allow it, and the hash algorithm otherwise. Neither result depends on the
/// number of threads or on the order rows arrive in.
///
/// Like DataFusion's own operators, it reserves the memory its inputs take
/// from the session's memory pool, and fails if the pool can't give it; it
/// runs the kernels on a blocking thread, off the async runtime; and its
/// metrics, shown by `EXPLAIN ANALYZE`, count which algorithm ran
/// (`dense_folds`, `hash_folds`), its compute time and its output rows.
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
    /// The host's plan for the same rows, run instead of the hash algorithm.
    fallback: Option<Arc<dyn ExecutionPlan>>,
    /// Whether to run the fallback unless no two domain rows reach one group.
    unique_domain: bool,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl EinFoldExec {
    /// The operator computing `fold` over `inputs` (operands, then domain),
    /// producing `schema`, which [`FoldNode`] derives.
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
            unique_domain: false,
            properties: Arc::new(properties),
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// This operator, running `plan` when the dense algorithm doesn't apply,
    /// instead of the hash algorithm. `plan` must produce the operator's
    /// columns, in order, under any names.
    pub fn with_fallback(mut self, plan: Arc<dyn ExecutionPlan>) -> Self {
        self.fallback = Some(plan);
        self
    }

    /// This operator, computing its fold only when no two of its domain's
    /// rows reach one group, and running its fallback otherwise. It needs a
    /// fallback (see [`EinFoldExec::with_fallback`]).
    pub fn requiring_unique_domain(mut self) -> Self {
        self.unique_domain = true;
        self
    }

    /// The host's plan this operator runs where the dense algorithm doesn't
    /// apply, if it has one.
    pub fn fallback(&self) -> Option<&Arc<dyn ExecutionPlan>> {
        self.fallback.as_ref()
    }

    /// This operator over `inputs`, with fresh metrics.
    fn copy(&self, inputs: Vec<Arc<dyn ExecutionPlan>>) -> EinFoldExec {
        let mut exec = EinFoldExec::new(self.fold.clone(), inputs, self.schema());
        exec.fallback = self.fallback.clone();
        exec.unique_domain = self.unique_domain;
        exec
    }

    /// The fold this operator computes.
    pub fn fold(&self) -> &Fold {
        &self.fold
    }
}

impl DisplayAs for EinFoldExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "EinFoldExec: {}",
            describe(&self.fold, self.unique_domain)
        )
    }
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
    let batches = collect_partitioned(plan, ctx).await?.concat();
    for b in &batches {
        reservation.try_grow(b.get_array_memory_size())?;
    }
    Ok(batches)
}

/// `batches` as one batch, its memory counted against `reservation`, and the
/// parts' released.
fn concat(
    schema: &SchemaRef,
    batches: Vec<RecordBatch>,
    reservation: &MemoryReservation,
) -> Result<RecordBatch> {
    let parts: usize = batches.iter().map(|b| b.get_array_memory_size()).sum();
    let whole = concat_batches(schema, &batches)?;
    reservation.try_grow(whole.get_array_memory_size())?;
    drop(batches);
    reservation.shrink(parts);
    Ok(whole)
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
            unique_domain: self.unique_domain,
        };
        if self.unique_domain && self.fallback.is_none() {
            return internal_err!("EinFoldExec requires a unique domain, but has no fallback");
        }
        let builder = || MetricBuilder::new(&self.metrics);
        let dense_folds = builder().global_counter("dense_folds");
        let hash_folds = builder().global_counter("hash_folds");
        let host_plans = builder().global_counter("host_plans");
        let fallback = self.fallback.clone();
        let compute = builder().elapsed_compute(0);
        let output_rows = builder().output_rows(0);
        // Shared with the kernel's task, which holds the inputs it counts: a
        // query cancelled while the kernel runs drops this future, but the
        // task runs on, and its inputs stay counted until it ends.
        let reservation =
            Arc::new(MemoryConsumer::new("EinFoldExec").register(context.memory_pool()));
        let out = Arc::clone(&schema);
        let result = async move {
            // The operands are read whole; the domain, whose rows the output
            // has one for one, stays in its batches.
            let schemas: Vec<SchemaRef> = inputs.iter().map(|i| i.schema()).collect();
            let mut batches = try_join_all(
                inputs
                    .into_iter()
                    .map(|i| collect(i, Arc::clone(&context), &reservation)),
            )
            .await?;
            let domain = (batches.len() == 3).then(|| batches.pop().unwrap());
            let b = concat(&schemas[1], batches.pop().unwrap(), &reservation)?;
            let a = concat(&schemas[0], batches.pop().unwrap(), &reservation)?;
            let timer = compute.timer();
            let with_fallback = fallback.is_some();
            let kernel_out = Arc::clone(&out);
            let held = Arc::clone(&reservation);
            let ran = SpawnedTask::spawn_blocking(move || {
                let _held = held;
                let domain = domain.as_deref();
                match with_fallback {
                    true => einfold_kernels::fold_dense(&fold, &a, &b, domain, kernel_out, options)
                        .map(|r| r.map(|batches| (Algorithm::Dense, batches))),
                    false => {
                        einfold_kernels::fold(&fold, &a, &b, domain, kernel_out, options).map(Some)
                    }
                }
            })
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))??;
            timer.done();
            // The kernel has dropped its inputs: release them, so that the
            // host's plan, if it runs, and the operators downstream have the
            // memory they would have had without einfold. The output, which
            // may share the domain's columns, is counted below.
            reservation.free();
            let batches = match (ran, fallback) {
                (Some((Algorithm::Dense, batches)), _) => {
                    dense_folds.add(1);
                    batches
                }
                (Some((Algorithm::Hash, batches)), _) => {
                    hash_folds.add(1);
                    batches
                }
                (None, Some(plan)) => {
                    // The host's plan, its columns under this operator's names.
                    host_plans.add(1);
                    collect_partitioned(plan, Arc::clone(&context))
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

/// Plans a [`FoldNode`] as an [`EinFoldExec`]. Give it to DataFusion's
/// physical planner with
/// [`DefaultPhysicalPlanner::with_extension_planners`], or use
/// [`EinFoldQueryPlanner`].
#[derive(Debug, Default)]
pub struct EinFoldPlanner;

#[async_trait]
impl ExtensionPlanner for EinFoldPlanner {
    async fn plan_extension(
        &self,
        planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session: &dyn Session,
        _planning_ctx: &PhysicalPlanningContext,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(node) = node.as_any().downcast_ref::<FoldNode>() else {
            return Ok(None);
        };
        let schema = Arc::new(node.schema().as_arrow().clone());
        let mut exec = EinFoldExec::new(node.fold().clone(), physical_inputs.to_vec(), schema);
        if let Some(plan) = node.fallback() {
            exec = exec.with_fallback(planner.create_physical_plan(plan, session).await?);
        }
        if node.unique_domain() {
            exec = exec.requiring_unique_domain();
        }
        Ok(Some(Arc::new(exec)))
    }
}

/// DataFusion's default query planner, with [`EinFoldPlanner`] added, and,
/// if made with [`EinFoldQueryPlanner::detecting`], [`DetectFolds`] applied to
/// each optimized logical plan just before it is planned. Install it with
/// `SessionStateBuilder::with_query_planner`, or use [`install`](crate::install).
///
/// Detecting here, rather than among the optimizer's rules, keeps `Fold`
/// nodes out of optimized logical plans, which other code reads: ddx, for
/// one, turns them into Substrait to differentiate them, and Substrait can't
/// encode a `Fold`. Only the plan that runs has them. It is also where a host
/// that can't add optimizer rules from outside, such as DataFusion's Python
/// bindings, can take a query planner.
#[derive(Debug, Default)]
pub struct EinFoldQueryPlanner {
    detect: Option<DetectFolds>,
}

impl EinFoldQueryPlanner {
    /// The planner, applying `rule` to each plan before planning it.
    pub fn detecting(rule: DetectFolds) -> Self {
        EinFoldQueryPlanner { detect: Some(rule) }
    }
}

#[async_trait]
impl QueryPlanner for EinFoldQueryPlanner {
    async fn create_physical_plan(
        &self,
        plan: &LogicalPlan,
        session: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let detected;
        let plan = match &self.detect {
            None => plan,
            Some(rule) => {
                let config = OptimizerContext::new();
                detected = plan
                    .clone()
                    .transform_up(|node| rule.rewrite(node, &config))?
                    .data;
                &detected
            }
        };
        DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(EinFoldPlanner)])
            .create_physical_plan(plan, session)
            .await
    }
}
