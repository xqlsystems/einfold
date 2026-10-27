// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The planners that put [`EinFoldExec`] in DataFusion's physical plans.

use std::sync::Arc;

use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::common::Result;
use datafusion::execution::context::QueryPlanner;
use datafusion::logical_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion::logical_expr::{LogicalPlan, UserDefinedLogicalNode};
use datafusion::optimizer::OptimizerContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use einfold_kernels::DomainRows;

use crate::detect_folds::DetectFolds;
use crate::ein_fold_exec::EinFoldExec;
use crate::fold_node::FoldNode;
use crate::pipeline_node::PipelineNode;

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
        if let Some(node) = node.as_any().downcast_ref::<PipelineNode>() {
            let schema = Arc::new(node.schema().as_arrow().clone());
            let pipeline = node.pipeline().clone();
            let exec = EinFoldExec::for_pipeline(pipeline, physical_inputs.to_vec(), schema);
            let fallback = planner
                .create_physical_plan(node.fallback(), session)
                .await?;
            return Ok(Some(Arc::new(exec.with_fallback(fallback))));
        }
        let Some(node) = node.as_any().downcast_ref::<FoldNode>() else {
            return Ok(None);
        };
        let schema = Arc::new(node.schema().as_arrow().clone());
        let mut exec = EinFoldExec::new(node.fold().clone(), physical_inputs.to_vec(), schema);
        if let Some(plan) = node.fallback() {
            exec = exec.with_fallback(planner.create_physical_plan(plan, session).await?);
        }
        if node.domain_rows() == DomainRows::UniqueKeys {
            exec = exec.requiring_unique_domain();
        }
        if let Some(c) = node.null_when() {
            exec = exec.with_null_when(c);
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
    #[must_use]
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
        use datafusion::common::tree_node::TreeNode;
        use datafusion::optimizer::OptimizerRule;
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
