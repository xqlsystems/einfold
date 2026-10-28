// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`Bridge`]: the query planner einfold installs in a datafusion-python
//! session, and what it needs to run plans there.

use std::any::Any;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::task::{Context, Poll};

use async_trait::async_trait;
use datafusion::arrow::array::RecordBatch;
use datafusion::catalog::{CatalogProviderList, Session};
use datafusion::common::config::{ConfigOptions, TableOptions};
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{DFSchema, Result};
use datafusion::execution::config::SessionConfig;
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::functions_aggregate::min_max::{max_udaf, Max};
use datafusion::functions_aggregate::sum::{sum_udaf, Sum};
use datafusion::logical_expr::execution_props::ExecutionProps;
use datafusion::logical_expr::registry::ExtensionTypeRegistryRef;
use datafusion::logical_expr::{
    AggregateUDF, Expr, HigherOrderUDF, LogicalPlan, ScalarUDF, WindowFunctionDefinition, WindowUDF,
};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_optimizer::optimizer::PhysicalOptimizer;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    self, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
};
use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};
use einfold_datafusion::{DetectFolds, EinFoldPlanner};

use crate::lean;
use futures::Stream;

/// The Tokio runtime einfold's operators run on. This library holds its own
/// copy of Tokio, so the session's runtime isn't visible to it.
pub(crate) fn runtime() -> tokio::runtime::Handle {
    static RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    let runtime = RUNTIME.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("a Tokio runtime")
    });
    runtime.handle().clone()
}

/// DataFusion's own aggregate for a foreign one of the same name, for the
/// aggregates detection matches by type.
fn builtin(name: &str) -> Option<Arc<AggregateUDF>> {
    match name {
        "sum" => Some(sum_udaf()),
        "max" => Some(max_udaf()),
        _ => None,
    }
}

/// Whether `f` is DataFusion's own aggregate, rather than a foreign one.
fn is_builtin(f: &AggregateUDF) -> bool {
    let inner: &dyn Any = f.inner().as_ref();
    inner.is::<Sum>() || inner.is::<Max>()
}

/// `plan` with every `SUM` and `MAX` that crossed the FFI as a foreign
/// function, in aggregates and in windows, replaced by DataFusion's own, so
/// that detection, which matches them by type, recognizes them. The
/// session's functions of those names are DataFusion's built-in ones unless
/// it registers others, which this would then replace.
fn relink(plan: LogicalPlan) -> Result<LogicalPlan> {
    let relinked = plan.transform_up(|node| {
        node.map_expressions(|e| {
            e.transform(|e| match e {
                Expr::AggregateFunction(mut f) if !is_builtin(&f.func) => {
                    match builtin(f.func.name()) {
                        Some(own) => {
                            f.func = own;
                            Ok(Transformed::yes(Expr::AggregateFunction(f)))
                        }
                        None => Ok(Transformed::no(Expr::AggregateFunction(f))),
                    }
                }
                Expr::WindowFunction(mut w) => match &w.fun {
                    WindowFunctionDefinition::AggregateUDF(f) if !is_builtin(f) => {
                        match builtin(f.name()) {
                            Some(own) => {
                                w.fun = WindowFunctionDefinition::AggregateUDF(own);
                                Ok(Transformed::yes(Expr::WindowFunction(w)))
                            }
                            None => Ok(Transformed::no(Expr::WindowFunction(w))),
                        }
                    }
                    _ => Ok(Transformed::no(Expr::WindowFunction(w))),
                },
                e => Ok(Transformed::no(e)),
            })
        })
    })?;
    // Expressions that changed type change their columns' names, which
    // DataFusion derives from the expression: recompute the schemas.
    relinked.data.recompute_schema()
}

/// What the planner has seen, for a session's reports.
#[derive(Debug, Default)]
pub(crate) struct Counts {
    pub(crate) plans: AtomicUsize,
    pub(crate) with_einfold: AtomicUsize,
    pub(crate) with_pipelines: AtomicUsize,
}

/// How many of einfold's fold nodes and pipeline nodes `plan` holds.
fn einfold_nodes(plan: &LogicalPlan) -> Result<(usize, usize)> {
    let (mut folds, mut pipelines) = (0, 0);
    plan.apply(|node| {
        if let LogicalPlan::Extension(e) = node {
            let any = e.node.as_any();
            folds += usize::from(any.is::<einfold_datafusion::FoldNode>());
            pipelines += usize::from(any.is::<einfold_datafusion::PipelineNode>());
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    Ok((folds, pipelines))
}

/// The session's planner, with einfold's detection in front of it. Every
/// plan is planned here, so that none crosses back to the session as
/// protobuf, which copies in-memory tables.
#[derive(Debug)]
pub(crate) struct Bridge {
    pub(crate) counts: Arc<Counts>,
}

#[async_trait]
impl QueryPlanner for Bridge {
    async fn create_physical_plan(
        &self,
        plan: &LogicalPlan,
        session: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.counts.plans.fetch_add(1, Ordering::Relaxed);
        let plan = lean::localize(relink(plan.clone())?, session).await?;
        let detected = DetectFolds::default().detect(plan)?;
        let (folds, pipelines) = einfold_nodes(&detected)?;
        if folds + pipelines > 0 {
            self.counts.with_einfold.fetch_add(1, Ordering::Relaxed);
        }
        if pipelines > 0 {
            self.counts.with_pipelines.fetch_add(1, Ordering::Relaxed);
        }
        let local = LocalRules {
            inner: session,
            rules: PhysicalOptimizer::new().rules,
            table_options: session.table_options().clone(),
        };
        let planner =
            DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(EinFoldPlanner)]);
        let physical = planner.create_physical_plan(&detected, &local).await?;
        Ok(Arc::new(Opaque(physical)))
    }
}

/// A planned plan, shown to the session as one node with no children, so
/// that returning it through the FFI encodes nothing but a reference (see
/// [`crate::codec::ByReference`]). Its nodes stay in this library, and run
/// on this library's runtime, entered whenever the session polls them.
#[derive(Debug)]
struct Opaque(Arc<dyn ExecutionPlan>);

impl DisplayAs for Opaque {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let inner = physical_plan::displayable(self.0.as_ref()).indent(false);
        write!(f, "einfold:\n{inner}")
    }
}

impl ExecutionPlan for Opaque {
    fn name(&self) -> &str {
        "einfold"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.0.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    /// The plan within, with its state reset: a session that runs a plan
    /// again, as a recursive query does, resets it first.
    fn reset_state(self: Arc<Self>) -> Result<Arc<dyn ExecutionPlan>> {
        let reset = physical_plan::execution_plan::reset_plan_states(Arc::clone(&self.0))?;
        Ok(Arc::new(Opaque(reset)))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let handle = runtime();
        let inner = {
            let _entered = handle.enter();
            self.0.execute(partition, context)?
        };
        let schema = inner.schema();
        let stream = Entered { inner, handle };
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }

    fn metrics(&self) -> Option<physical_plan::metrics::MetricsSet> {
        self.0.metrics()
    }
}

/// A stream polled inside this library's runtime.
struct Entered {
    inner: SendableRecordBatchStream,
    handle: tokio::runtime::Handle,
}

impl Stream for Entered {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let _entered = self.handle.enter();
        self.inner.as_mut().poll_next(cx)
    }
}

/// The session, but with this library's own physical optimizer rules: the
/// session's reach a plan built here only as FFI proxies of its nodes,
/// which they can't optimize, and each call would cross the FFI.
struct LocalRules<'a> {
    inner: &'a dyn Session,
    rules: Vec<Arc<dyn PhysicalOptimizerRule + Send + Sync>>,
    table_options: TableOptions,
}

#[async_trait]
impl Session for LocalRules<'_> {
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }
    fn config(&self) -> &SessionConfig {
        self.inner.config()
    }
    fn config_options(&self) -> &ConfigOptions {
        self.inner.config_options()
    }
    fn catalog_list(&self) -> Arc<dyn CatalogProviderList> {
        self.inner.catalog_list()
    }
    fn query_planner(&self) -> Arc<dyn QueryPlanner + Send + Sync> {
        self.inner.query_planner()
    }
    fn optimize(&self, plan: &LogicalPlan) -> Result<LogicalPlan> {
        self.inner.optimize(plan)
    }
    fn physical_optimizers(&self) -> &[Arc<dyn PhysicalOptimizerRule + Send + Sync>] {
        &self.rules
    }
    async fn create_physical_plan(&self, plan: &LogicalPlan) -> Result<Arc<dyn ExecutionPlan>> {
        self.inner.create_physical_plan(plan).await
    }
    fn create_physical_expr(&self, expr: Expr, schema: &DFSchema) -> Result<Arc<dyn PhysicalExpr>> {
        self.inner.create_physical_expr(expr, schema)
    }
    fn scalar_functions(&self) -> &HashMap<String, Arc<ScalarUDF>> {
        self.inner.scalar_functions()
    }
    fn higher_order_functions(&self) -> &HashMap<String, Arc<HigherOrderUDF>> {
        self.inner.higher_order_functions()
    }
    fn aggregate_functions(&self) -> &HashMap<String, Arc<AggregateUDF>> {
        self.inner.aggregate_functions()
    }
    fn window_functions(&self) -> &HashMap<String, Arc<WindowUDF>> {
        self.inner.window_functions()
    }
    fn extension_type_registry(&self) -> &ExtensionTypeRegistryRef {
        self.inner.extension_type_registry()
    }
    fn runtime_env(&self) -> &Arc<RuntimeEnv> {
        self.inner.runtime_env()
    }
    fn execution_props(&self) -> &ExecutionProps {
        self.inner.execution_props()
    }
    fn as_any(&self) -> &dyn Any {
        self.inner.as_any()
    }
    fn table_options(&self) -> &TableOptions {
        self.inner.table_options()
    }
    fn table_options_mut(&mut self) -> &mut TableOptions {
        &mut self.table_options
    }
    fn task_ctx(&self) -> Arc<TaskContext> {
        self.inner.task_ctx()
    }
}
