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
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{internal_err, Result};
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
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
use futures::future::try_join_all;
use futures::{stream, StreamExt, TryStreamExt};

use crate::node::{describe, FoldNode};

/// Computes a fold over a join (see [`FoldNode`]) without materializing the
/// join: each joined pair of rows updates its output group directly.
///
/// Its children are the fold's operands, then its fill, if it has one. It
/// reads every partition of each, and emits one partition, in batches of the
/// session's batch size.
///
/// It uses the dense algorithm, by positions and on as many threads as the
/// session's target partitions, when the keys and rows allow it, and the
/// hash algorithm otherwise. Both give results that don't depend on the
/// number of threads or on the order rows arrive in. Its metrics count each
/// (`dense_folds`, `hash_folds`).
#[derive(Debug)]
pub struct EinFoldExec {
    fold: Fold,
    inputs: Vec<Arc<dyn ExecutionPlan>>,
    properties: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl EinFoldExec {
    /// The operator computing `fold` over `inputs` (operands, then fill),
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
            properties: Arc::new(properties),
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }

    /// The fold this operator computes.
    pub fn fold(&self) -> &Fold {
        &self.fold
    }
}

impl DisplayAs for EinFoldExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "EinFoldExec: {}", describe(&self.fold))
    }
}

/// Every batch of every partition of `plan`, in partition order. The
/// partitions are read concurrently, since an operator such as a hash
/// repartition produces them together and would otherwise buffer all but one.
async fn collect(plan: Arc<dyn ExecutionPlan>, ctx: Arc<TaskContext>) -> Result<Vec<RecordBatch>> {
    Ok(collect_partitioned(plan, ctx).await?.concat())
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
        vec![Distribution::UnspecifiedDistribution; self.inputs.len()]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false; self.inputs.len()]
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != self.inputs.len() {
            return internal_err!(
                "EinFoldExec has {} children, given {}",
                self.inputs.len(),
                children.len()
            );
        }
        Ok(Arc::new(EinFoldExec::new(
            self.fold.clone(),
            children,
            self.schema(),
        )))
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
        if inputs.len() != 2 + usize::from(fold.fill().is_some()) {
            return internal_err!("EinFoldExec was given {} inputs", inputs.len());
        }
        let batch_size = context.session_config().batch_size().max(1);
        let threads = context.session_config().target_partitions();
        let dense_folds = MetricBuilder::new(&self.metrics).global_counter("dense_folds");
        let hash_folds = MetricBuilder::new(&self.metrics).global_counter("hash_folds");
        let out = Arc::clone(&schema);
        let result = async move {
            // The operands are read whole; the fill, whose rows the output
            // has one for one, is passed on batch by batch.
            let schemas: Vec<SchemaRef> = inputs.iter().map(|i| i.schema()).collect();
            let mut batches =
                try_join_all(inputs.into_iter().map(|i| collect(i, Arc::clone(&context)))).await?;
            let fill = (batches.len() == 3).then(|| batches.pop().unwrap());
            let fill = fill.as_deref();
            let a = concat_batches(&schemas[0], &batches[0])?;
            let b = concat_batches(&schemas[1], &batches[1])?;
            let dense = crate::dense::fold(&fold, &a, &b, fill, Arc::clone(&out), threads)?;
            let batches = match dense {
                Some(batches) => {
                    dense_folds.add(1);
                    batches
                }
                None => {
                    hash_folds.add(1);
                    crate::hash::fold(&fold, &a, &b, fill, out)?
                }
            };
            let slices: Vec<Result<RecordBatch>> = batches
                .iter()
                .flat_map(|batch| {
                    (0..batch.num_rows()).step_by(batch_size).map(move |start| {
                        Ok(batch.slice(start, batch_size.min(batch.num_rows() - start)))
                    })
                })
                .collect();
            Ok::<_, datafusion::error::DataFusionError>(stream::iter(slices))
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
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session: &dyn Session,
        _planning_ctx: &PhysicalPlanningContext,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(node) = node.as_any().downcast_ref::<FoldNode>() else {
            return Ok(None);
        };
        let schema = Arc::new(node.schema().as_arrow().clone());
        Ok(Some(Arc::new(EinFoldExec::new(
            node.fold().clone(),
            physical_inputs.to_vec(),
            schema,
        ))))
    }
}

/// DataFusion's default query planner, with [`EinFoldPlanner`] added. Install
/// it with `SessionStateBuilder::with_query_planner`.
#[derive(Debug, Default)]
pub struct EinFoldQueryPlanner;

#[async_trait]
impl QueryPlanner for EinFoldQueryPlanner {
    async fn create_physical_plan(
        &self,
        plan: &LogicalPlan,
        session: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(EinFoldPlanner)])
            .create_physical_plan(plan, session)
            .await
    }
}
