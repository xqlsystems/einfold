// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Scanning a session's tables through the FFI (foreign function interface)
//! of DataFusion, the Rust query engine on Apache Arrow that the session
//! runs, at about the cost of a table scan: [`Lean`] for any table, and
//! [`localize`] for in-memory ones.
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
use std::fmt::{self, Formatter};
use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use datafusion::arrow::array::{ArrayData, RecordBatch};
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::default_table_source;
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
use datafusion::physical_plan::metrics::MetricsSet;
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
        // session's functions: such a scan is planned, and runs, with them.
        let (plan, functions) = match filters.is_empty() {
            true => {
                let lean = LeanSession::new(state);
                let plan = self.table.scan(&lean, projection, filters, limit).await?;
                (plan, Functions::Without)
            }
            false => {
                let plan = self.table.scan(state, projection, filters, limit).await?;
                (plan, Functions::Kept)
            }
        };
        Ok(Arc::new(LeanExec::new(plan, functions)))
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

/// Whether a scan runs with the session's functions.
#[derive(Clone, Copy, Debug)]
enum Functions {
    /// With the session's configuration only: a scan without filters.
    Without,
    /// With everything the session's task context has: a scan whose table
    /// evaluates filters, which may name functions.
    Kept,
}

/// A foreign scan, run in a task context with the session's configuration
/// only unless it evaluates filters, and shown to the planner as a leaf: the
/// planner can't change a foreign plan's nodes anyway.
#[derive(Debug)]
struct LeanExec {
    plan: Arc<dyn ExecutionPlan>,
    functions: Functions,
    statistics: OnceLock<Arc<Statistics>>,
}

impl LeanExec {
    fn new(plan: Arc<dyn ExecutionPlan>, functions: Functions) -> Self {
        LeanExec {
            plan,
            functions,
            statistics: OnceLock::new(),
        }
    }
}

impl DisplayAs for LeanExec {
    fn fmt_as(&self, format: DisplayFormatType, formatter: &mut Formatter) -> fmt::Result {
        self.plan.fmt_as(format, formatter)
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
        Ok(Arc::new(LeanExec::new(plan, self.functions)))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let context = match self.functions {
            Functions::Without => lean_task(&context),
            Functions::Kept => context,
        };
        self.plan.execute(partition, context)
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
        if let Some(statistics) = self.statistics.get() {
            return Ok(Arc::clone(statistics));
        }
        let statistics = self.plan.statistics_from_inputs(input_stats, args)?;
        Ok(Arc::clone(self.statistics.get_or_init(|| statistics)))
    }

    fn metrics(&self) -> Option<MetricsSet> {
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

/// The largest table [`localize`] copies, in bytes: the bound under which
/// xarray-sql, an XQL Systems project for querying xarray datasets with SQL,
/// keeps a dataset as an in-memory table.
const COPIED_BYTES: usize = 64 << 20;

/// What a scan of a small in-memory table reads, as far as this library can
/// tell. Each scan reaches this library as a foreign table of its own, which
/// says nothing of the table behind it, and a name doesn't say either: a
/// plan may join a DataFrame made before a name was registered again with
/// one made after, and every unnamed table, such as a DataFrame's cached
/// result (`DataFrame.cache()`), is called `?table?`. But a foreign scan of
/// an in-memory table hands over the table's own buffers, so two scans read
/// the same table when they have the same name, schema and number of rows,
/// and every column of their first batch is the same slice of the same
/// buffers. Columns alone can coincide: two cached results of one table
/// share the buffers of each column neither changed.
#[derive(Debug, PartialEq, Eq, Hash)]
struct Data {
    name: TableReference,
    schema: SchemaRef,
    rows: usize,
    /// Each column of the first batch with rows, empty if there's none.
    first: Vec<Slice>,
}

/// Where a column's values are: its buffers' addresses, its offset and
/// length in them, and the same of each child array a nested column has.
#[derive(Debug, PartialEq, Eq, Hash)]
struct Slice {
    buffers: Vec<usize>,
    offset: usize,
    len: usize,
    children: Vec<Slice>,
}

impl Slice {
    fn of(data: &ArrayData) -> Self {
        let buffers = (data.buffers().iter())
            .chain(data.nulls().map(|nulls| nulls.buffer()))
            .map(|buffer| buffer.as_ptr() as usize)
            .collect();
        Slice {
            buffers,
            offset: data.offset(),
            len: data.len(),
            children: data.child_data().iter().map(Slice::of).collect(),
        }
    }
}

/// Where each of `batch`'s columns is.
fn slices(batch: &RecordBatch) -> Vec<Slice> {
    (batch.columns().iter())
        .map(|column| Slice::of(&column.to_data()))
        .collect()
}

/// Where each column of `scan`'s first batch with rows is.
async fn first_of(scan: &Arc<dyn ExecutionPlan>, context: &Arc<TaskContext>) -> Result<Vec<Slice>> {
    use futures::StreamExt;

    let partitions = scan.properties().output_partitioning().partition_count();
    for partition in 0..partitions {
        let mut stream = scan.execute(partition, Arc::clone(context))?;
        while let Some(batch) = stream.next().await {
            let batch = batch?;
            if batch.num_rows() > 0 {
                return Ok(slices(&batch));
            }
        }
    }
    Ok(Vec::new())
}

/// A scan of a small in-memory table, and its number of rows.
struct SmallScan {
    rows: usize,
    scan: Arc<dyn ExecutionPlan>,
}

/// A scan of `table`, if it's a small in-memory table: one DataFusion's own
/// data source scans, as it does an in-memory table, whose statistics say it
/// holds at most [`COPIED_BYTES`].
async fn small_scan(table: &Lean, session: &dyn Session) -> Result<Option<SmallScan>> {
    let scan = (table.table)
        .scan(&LeanSession::new(session), None, &[], None)
        .await?;
    let statistics = scan.statistics_from_inputs(&[], &StatisticsArgs::new())?;
    let (Precision::Exact(rows), Precision::Exact(bytes)) =
        (statistics.num_rows, statistics.total_byte_size)
    else {
        return Ok(None);
    };
    match scan.name() == "DataSourceExec" && bytes <= COPIED_BYTES {
        true => Ok(Some(SmallScan { rows, scan })),
        false => Ok(None),
    }
}

/// `plan` with each small in-memory table it reads replaced by this
/// library's own copy of it, which shares the table's buffers, so that no
/// scan of it crosses the FFI as the query runs. Each table crosses it here,
/// once, and each scan once more, to tell which table it reads (see
/// [`Data`]).
///
/// A table is copied only when it's small and in memory (see
/// [`small_scan`]). Another source, such as xarray-sql's lazily loaded
/// datasets, would be read whole here, rather than streamed as the query
/// runs. A scan given filters to evaluate keeps its own table, since the
/// copy doesn't evaluate them.
pub(crate) async fn localize(plan: LogicalPlan, session: &dyn Session) -> Result<LogicalPlan> {
    /// A scan's table and name.
    struct Scanned {
        source: Arc<dyn TableSource>,
        name: TableReference,
    }
    // Each scan's, unless the scan has filters.
    let mut scans: Vec<Scanned> = Vec::new();
    plan.apply_with_subqueries(|node| {
        if let LogicalPlan::TableScan(scan) = node {
            if scan.filters.is_empty() {
                scans.push(Scanned {
                    source: Arc::clone(&scan.source),
                    name: scan.table_name.clone(),
                });
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    let address = |source: &Arc<dyn TableSource>| Arc::as_ptr(source) as *const () as usize;
    let context = lean_task(&session.task_ctx());
    let mut copies: HashMap<Data, Arc<dyn TableSource>> = HashMap::new();
    let mut chosen: HashMap<usize, Arc<dyn TableSource>> = HashMap::new();
    for Scanned { source, name } in scans {
        if chosen.contains_key(&address(&source)) {
            continue;
        }
        let provider = default_table_source::source_as_provider(&source)?;
        let Some(lean) = provider.downcast_ref::<Lean>() else {
            continue;
        };
        let Some(SmallScan { rows, scan }) = small_scan(lean, session).await? else {
            continue;
        };
        // Read on this library's runtime, where the foreign scan's streams
        // are polled.
        let task_context = Arc::clone(&context);
        let probe = Arc::clone(&scan);
        let first = crate::bridge::runtime()
            .spawn(async move { first_of(&probe, &task_context).await })
            .await
            .map_err(|error| DataFusionError::External(Box::new(error)))??;
        let data = Data {
            name,
            schema: lean.schema(),
            rows,
            first,
        };
        let copy = match copies.get(&data) {
            Some(copy) => Arc::clone(copy),
            None => {
                let read = crate::bridge::runtime().spawn(physical_plan::collect_partitioned(
                    scan,
                    Arc::clone(&context),
                ));
                let partitions = read
                    .await
                    .map_err(|error| DataFusionError::External(Box::new(error)))??;
                let mut table = MemTable::try_new(lean.schema(), partitions)?;
                if let Some(constraints) = lean.constraints() {
                    table = table.with_constraints(constraints.clone());
                }
                let copy = default_table_source::provider_as_source(Arc::new(table));
                copies.insert(data, Arc::clone(&copy));
                copy
            }
        };
        chosen.insert(address(&source), copy);
    }
    if chosen.is_empty() {
        return Ok(plan);
    }
    let copied = plan.transform_up_with_subqueries(|node| match node {
        LogicalPlan::TableScan(mut scan) if scan.filters.is_empty() => {
            match chosen.get(&address(&scan.source)) {
                Some(copy) => {
                    scan.source = Arc::clone(copy);
                    Ok(Transformed::yes(LogicalPlan::TableScan(scan)))
                }
                None => Ok(Transformed::no(LogicalPlan::TableScan(scan))),
            }
        }
        node => Ok(Transformed::no(node)),
    })?;
    Ok(copied.data)
}
