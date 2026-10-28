// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Scanning a session's tables through DataFusion's FFI at about the cost of
//! a table scan: [`Lean`] for any table, and [`localize`] for in-memory ones.
//!
//! Every table a plan reads reaches this library as a foreign table, even an
//! in-memory one, so each of its scans crosses the FFI. A foreign scan
//! converts the session it is planned in, at planning, and the task context
//! it runs in, once per partition. Each conversion copies the session's whole
//! configuration as strings and wraps each of its several hundred functions,
//! into the FFI here and back out on the session's side. A plan that reads
//! tables many times, as one that expands a view at each read does, pays
//! that at every scan, where the session itself reads an in-memory table at
//! the cost of handing over its batches.
//!
//! A scan needs only the session's configuration, not its functions or
//! catalogs, unless the table evaluates filters given to it, which may name
//! functions. So [`Lean`] scans with the configuration only, and remembers
//! the table's and each scan's statistics, which DataFusion's join planner
//! asks for at each join. [`localize`] goes further for small in-memory
//! tables: it reads each once per plan, and every scan reads the copy.

use std::any::Any;
use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::default_table_source::{provider_as_source, source_as_provider};
use datafusion::catalog::{CatalogProviderList, MemoryCatalogProviderList, Session, TableProvider};
use datafusion::common::config::{ConfigOptions, TableOptions};
use datafusion::common::stats::Precision;
use datafusion::common::tree_node::{Transformed, TreeNodeRecursion};
use datafusion::common::{
    Constraints, DFSchema, DataFusionError, Result, Statistics, TableReference,
};
use datafusion::datasource::MemTable;
use datafusion::datasource::TableType;
use datafusion::execution::config::SessionConfig;
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::runtime_env::RuntimeEnv;
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::dml::InsertOp;
use datafusion::logical_expr::execution_props::ExecutionProps;
use datafusion::logical_expr::registry::ExtensionTypeRegistryRef;
use datafusion::logical_expr::{
    AggregateUDF, Expr, HigherOrderUDF, LogicalPlan, ScalarUDF, TableProviderFilterPushDown,
    TableSource, WindowUDF,
};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::statistics::StatisticsArgs;
use datafusion::physical_plan::{
    self, DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties,
};

/// A session's table, scanned with the session's configuration only (see
/// the module's documentation).
#[derive(Debug)]
pub(crate) struct Lean {
    table: Arc<dyn TableProvider>,
    statistics: OnceLock<Option<Statistics>>,
}

impl Lean {
    pub(crate) fn new(table: Arc<dyn TableProvider>) -> Self {
        Lean {
            table,
            statistics: OnceLock::new(),
        }
    }
}

#[async_trait]
impl TableProvider for Lean {
    fn schema(&self) -> SchemaRef {
        self.table.schema()
    }

    fn constraints(&self) -> Option<&Constraints> {
        self.table.constraints()
    }

    fn table_type(&self) -> TableType {
        self.table.table_type()
    }

    fn get_table_definition(&self) -> Option<&str> {
        self.table.get_table_definition()
    }

    fn get_column_default(&self, column: &str) -> Option<&Expr> {
        self.table.get_column_default(column)
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        // The table evaluates the filters given to it, which may name the
        // session's functions.
        let plan = match filters.is_empty() {
            true => {
                let lean = LeanSession::new(state);
                self.table.scan(&lean, projection, filters, limit).await?
            }
            false => self.table.scan(state, projection, filters, limit).await?,
        };
        Ok(Arc::new(LeanExec::new(plan)))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        self.table.supports_filters_pushdown(filters)
    }

    fn statistics(&self) -> Option<Statistics> {
        (self.statistics)
            .get_or_init(|| self.table.statistics())
            .clone()
    }

    async fn insert_into(
        &self,
        state: &dyn Session,
        input: Arc<dyn ExecutionPlan>,
        insert_op: InsertOp,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.table.insert_into(state, input, insert_op).await
    }
}

/// A foreign scan, run in a task context with the session's configuration
/// only, and shown to the planner as a leaf: the planner can't change a
/// foreign plan's nodes anyway.
#[derive(Debug)]
struct LeanExec {
    plan: Arc<dyn ExecutionPlan>,
    statistics: OnceLock<Arc<Statistics>>,
}

impl LeanExec {
    fn new(plan: Arc<dyn ExecutionPlan>) -> Self {
        LeanExec {
            plan,
            statistics: OnceLock::new(),
        }
    }
}

impl DisplayAs for LeanExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.plan.fmt_as(t, f)
    }
}

impl ExecutionPlan for LeanExec {
    fn name(&self) -> &str {
        self.plan.name()
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        self.plan.properties()
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

    fn reset_state(self: Arc<Self>) -> Result<Arc<dyn ExecutionPlan>> {
        let plan = physical_plan::execution_plan::reset_plan_states(Arc::clone(&self.plan))?;
        Ok(Arc::new(LeanExec::new(plan)))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        self.plan.execute(partition, lean_task(&context))
    }

    /// The foreign scan's statistics, which it computes from nothing below
    /// it: remembered for the whole scan, which the join planner asks for at
    /// each join above.
    fn statistics_from_inputs(
        &self,
        input_stats: &[Arc<Statistics>],
        args: &StatisticsArgs,
    ) -> Result<Arc<Statistics>> {
        if args.partition().is_some() {
            return self.plan.statistics_from_inputs(input_stats, args);
        }
        if let Some(s) = self.statistics.get() {
            return Ok(Arc::clone(s));
        }
        let s = self.plan.statistics_from_inputs(input_stats, args)?;
        Ok(Arc::clone(self.statistics.get_or_init(|| s)))
    }

    fn metrics(&self) -> Option<physical_plan::metrics::MetricsSet> {
        self.plan.metrics()
    }
}

/// `context` without its functions.
fn lean_task(context: &TaskContext) -> Arc<TaskContext> {
    Arc::new(TaskContext::new(
        context.task_id(),
        context.session_id(),
        context.session_config().clone(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        HashMap::new(),
        context.runtime_env(),
    ))
}

/// A session with `inner`'s configuration, and no functions or catalogs.
struct LeanSession<'a> {
    inner: &'a dyn Session,
    catalogs: Arc<dyn CatalogProviderList>,
    table_options: TableOptions,
    none_scalar: HashMap<String, Arc<ScalarUDF>>,
    none_higher_order: HashMap<String, Arc<HigherOrderUDF>>,
    none_aggregate: HashMap<String, Arc<AggregateUDF>>,
    none_window: HashMap<String, Arc<WindowUDF>>,
}

impl<'a> LeanSession<'a> {
    fn new(inner: &'a dyn Session) -> Self {
        LeanSession {
            inner,
            catalogs: Arc::new(MemoryCatalogProviderList::new()),
            table_options: inner.table_options().clone(),
            none_scalar: HashMap::new(),
            none_higher_order: HashMap::new(),
            none_aggregate: HashMap::new(),
            none_window: HashMap::new(),
        }
    }
}

#[async_trait]
impl Session for LeanSession<'_> {
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
        Arc::clone(&self.catalogs)
    }
    fn query_planner(&self) -> Arc<dyn QueryPlanner + Send + Sync> {
        self.inner.query_planner()
    }
    fn optimize(&self, plan: &LogicalPlan) -> Result<LogicalPlan> {
        self.inner.optimize(plan)
    }
    fn physical_optimizers(&self) -> &[Arc<dyn PhysicalOptimizerRule + Send + Sync>] {
        self.inner.physical_optimizers()
    }
    async fn create_physical_plan(&self, plan: &LogicalPlan) -> Result<Arc<dyn ExecutionPlan>> {
        self.inner.create_physical_plan(plan).await
    }
    fn create_physical_expr(&self, expr: Expr, schema: &DFSchema) -> Result<Arc<dyn PhysicalExpr>> {
        self.inner.create_physical_expr(expr, schema)
    }
    fn scalar_functions(&self) -> &HashMap<String, Arc<ScalarUDF>> {
        &self.none_scalar
    }
    fn higher_order_functions(&self) -> &HashMap<String, Arc<HigherOrderUDF>> {
        &self.none_higher_order
    }
    fn aggregate_functions(&self) -> &HashMap<String, Arc<AggregateUDF>> {
        &self.none_aggregate
    }
    fn window_functions(&self) -> &HashMap<String, Arc<WindowUDF>> {
        &self.none_window
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
        &self.table_options
    }
    fn table_options_mut(&mut self) -> &mut TableOptions {
        &mut self.table_options
    }
    fn task_ctx(&self) -> Arc<TaskContext> {
        lean_task(&self.inner.task_ctx())
    }
}

/// The largest table [`localize`] copies, in bytes: xarray-sql's bound for
/// the datasets it keeps as in-memory tables.
const COPIED_BYTES: usize = 64 << 20;

/// `plan` with each in-memory table it reads replaced by this library's own
/// copy of it, which shares the table's buffers, so that no scan of it
/// crosses the FFI: one scan of each crosses it, here, once.
///
/// Within one plan, a table's name names one table, so every scan of a name
/// reads the copy. A table is copied only when DataFusion's own data source
/// scans it, as it does an in-memory table, and the scan's statistics say
/// it holds at most [`COPIED_BYTES`]. Another source, such as xarray-sql's
/// lazily loaded datasets, would be read whole here, rather than streamed as
/// the query runs; a small file, read here, is read once rather than at each
/// scan. A table given filters to evaluate in its scan keeps its own scans,
/// since the copy doesn't evaluate them.
pub(crate) async fn localize(plan: LogicalPlan, session: &dyn Session) -> Result<LogicalPlan> {
    // Each table by name, unless one of its scans has filters.
    let mut tables: HashMap<TableReference, Option<Arc<dyn TableProvider>>> = HashMap::new();
    plan.apply_with_subqueries(|node| {
        if let LogicalPlan::TableScan(scan) = node {
            let provider = source_as_provider(&scan.source)?;
            let lean = provider.downcast_ref::<Lean>().is_some() && scan.filters.is_empty();
            let entry = tables
                .entry(scan.table_name.clone())
                .or_insert(Some(provider));
            if !lean {
                *entry = None;
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    let mut copies: HashMap<TableReference, Arc<dyn TableSource>> = HashMap::new();
    let mut context = None;
    for (name, provider) in tables {
        let Some(lean) = provider.as_deref().and_then(|p| p.downcast_ref::<Lean>()) else {
            continue;
        };
        let scan = lean
            .table
            .scan(&LeanSession::new(session), None, &[], None)
            .await?;
        let statistics = scan.statistics_from_inputs(&[], &StatisticsArgs::new())?;
        let small = matches!(statistics.num_rows, Precision::Exact(_))
            && matches!(statistics.total_byte_size, Precision::Exact(b) if b <= COPIED_BYTES);
        if scan.name() != "DataSourceExec" || !small {
            continue;
        }
        let context = context.get_or_insert_with(|| lean_task(&session.task_ctx()));
        let read = crate::bridge::runtime().spawn(physical_plan::collect_partitioned(
            scan,
            Arc::clone(context),
        ));
        let partitions = read
            .await
            .map_err(|e| DataFusionError::External(Box::new(e)))??;
        let mut copy = MemTable::try_new(lean.schema(), partitions)?;
        if let Some(c) = lean.constraints() {
            copy = copy.with_constraints(c.clone());
        }
        copies.insert(name, provider_as_source(Arc::new(copy)));
    }
    if copies.is_empty() {
        return Ok(plan);
    }
    let copied = plan.transform_up_with_subqueries(|node| match node {
        LogicalPlan::TableScan(mut scan) => match copies.get(&scan.table_name) {
            Some(copy) => {
                scan.source = Arc::clone(copy);
                Ok(Transformed::yes(LogicalPlan::TableScan(scan)))
            }
            None => Ok(Transformed::no(LogicalPlan::TableScan(scan))),
        },
        node => Ok(Transformed::no(node)),
    })?;
    Ok(copied.data)
}
