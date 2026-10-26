// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Spike S29: einfold's detection and operator inside datafusion-python, the
//! Python bindings of Apache DataFusion, through the FFI query planner its
//! `SessionContext.with_extensions` installs.
//!
//! `EinFold` is an extension bundle. Its planner hook receives the session's
//! planner so far, and returns one that applies `DetectFolds` to each plan:
//! when it finds a fold, the plan is planned with `EinFoldPlanner`; otherwise
//! the session's own planner plans it, unchanged.
//!
//! datafusion-python hands its plans to such a planner as protobuf, and
//! protobuf can't encode a table such as an in-memory one without a codec
//! for it. The bundle's components hook therefore adds `ByReference`: a
//! logical codec that encodes any table as an index into this process's
//! [`TABLES`], and decodes it back to the same table. The planned physical
//! plan comes back as protobuf too, and neither `EinFoldExec` nor the
//! session's own nodes, which reach the planner as FFI proxies, have a codec
//! there. A physical `ByReference` passes them the same way, through
//! [`PLANS`]. The planner runs in the same process as the session, so a
//! reference is enough.

use std::ptr::NonNull;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{not_impl_err, plan_err, Result, TableReference};
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::execution::TaskContext;
use datafusion::logical_expr::Extension;
use datafusion::execution::context::QueryPlanner;
use datafusion::logical_expr::LogicalPlan;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};
use datafusion_ffi::proto::logical_extension_codec::FFI_LogicalExtensionCodec;
use datafusion_ffi::proto::physical_extension_codec::FFI_PhysicalExtensionCodec;
use datafusion_ffi::execution::FFI_TaskContextProvider;
use datafusion_ffi::query_planner::FFI_QueryPlanner;
use datafusion_proto::logical_plan::LogicalExtensionCodec;
use datafusion_proto::physical_plan::{PhysicalExtensionCodec, PhysicalProtoConverterExtension};
use einfold_datafusion::{DetectFolds, EinFoldPlanner};
use pyo3::exceptions::PyImportError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};

/// `obj`'s capsule from its `getter`, or `obj` itself if it has no getter.
fn capsule<'py>(obj: &Bound<'py, PyAny>, getter: &str) -> PyResult<Bound<'py, PyCapsule>> {
    let obj = match obj.hasattr(getter)? {
        true => obj.getattr(getter)?.call0()?,
        false => obj.clone(),
    };
    Ok(obj.cast_into::<PyCapsule>()?)
}

/// Fails unless a foreign object was built against this DataFusion's FFI.
fn check_version(kind: &str, found: u64) -> PyResult<()> {
    let expected = datafusion_ffi::version();
    match found == expected {
        true => Ok(()),
        false => Err(PyImportError::new_err(format!(
            "{kind} has FFI version {found}; einfold expects {expected}"
        ))),
    }
}

/// The value of capsule `capsule`, named `name`.
///
/// # Safety
///
/// The capsule must hold a `T`, as datafusion-python's capsules of that name do.
unsafe fn capsule_value<T: Clone>(capsule: &Bound<'_, PyCapsule>, name: &std::ffi::CStr) -> PyResult<T> {
    let data: NonNull<T> = capsule.pointer_checked(Some(name))?.cast();
    // SAFETY: the caller guarantees the capsule holds a `T`.
    Ok(unsafe { data.as_ref() }.clone())
}

/// `plan` with every aggregate named `sum` that crossed the FFI as a foreign
/// function replaced by DataFusion's own `SUM`, so that `DetectFolds`, which
/// matches `SUM` by type, recognizes it. This assumes the host's `sum` is
/// DataFusion's built-in one, as datafusion-python's is unless a session
/// registers another.
fn relink_sum(plan: LogicalPlan) -> Result<datafusion::common::tree_node::Transformed<LogicalPlan>> {
    use datafusion::common::tree_node::{Transformed, TreeNode};
    use datafusion::functions_aggregate::sum::{sum_udaf, Sum};
    use datafusion::logical_expr::Expr;
    plan.transform_up(|node| {
        node.map_expressions(|e| {
            e.transform(|e| match e {
                Expr::AggregateFunction(mut f) if f.func.name() == "sum" => {
                    let inner: &dyn std::any::Any = f.func.inner().as_ref();
                    match inner.is::<Sum>() {
                        true => Ok(Transformed::no(Expr::AggregateFunction(f))),
                        false => {
                            f.func = sum_udaf();
                            Ok(Transformed::yes(Expr::AggregateFunction(f)))
                        }
                    }
                }
                e => Ok(Transformed::no(e)),
            })
        })
    })
}

/// The tables [`ByReference`] has encoded, by index. They are never removed:
/// the spike's process is short-lived.
static TABLES: Mutex<Vec<Arc<dyn TableProvider>>> = Mutex::new(Vec::new());

/// Marks a buffer [`ByReference`] wrote.
const MAGIC: &[u8] = b"einfold-s29:";

/// A logical codec that passes tables by reference within this process.
#[derive(Debug)]
struct ByReference;

impl LogicalExtensionCodec for ByReference {
    fn try_decode(&self, _buf: &[u8], _inputs: &[LogicalPlan], _ctx: &TaskContext) -> Result<Extension> {
        not_impl_err!("einfold's spike codec encodes only tables")
    }

    fn try_encode(&self, _node: &Extension, _buf: &mut Vec<u8>) -> Result<()> {
        not_impl_err!("einfold's spike codec encodes only tables")
    }

    fn try_decode_table_provider(
        &self,
        buf: &[u8],
        _table_ref: &TableReference,
        _schema: SchemaRef,
        _ctx: &TaskContext,
    ) -> Result<Arc<dyn TableProvider>> {
        let Some(index) = buf.strip_prefix(MAGIC).and_then(|b| b.try_into().ok()) else {
            return plan_err!("not a table einfold's spike codec encoded");
        };
        let tables = TABLES.lock().expect("no panic while holding the tables");
        match tables.get(u64::from_le_bytes(index) as usize) {
            Some(table) => Ok(Arc::clone(table)),
            None => plan_err!("einfold's spike codec holds no such table"),
        }
    }

    fn try_encode_table_provider(
        &self,
        _table_ref: &TableReference,
        node: Arc<dyn TableProvider>,
        buf: &mut Vec<u8>,
    ) -> Result<()> {
        let mut tables = TABLES.lock().expect("no panic while holding the tables");
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&(tables.len() as u64).to_le_bytes());
        tables.push(node);
        Ok(())
    }
}

/// [`ByReference`] as a codec datafusion-python can install.
#[pyclass(name = "ByReferenceCodec", module = "einfold_spike")]
struct ByReferenceCodec {
    codec: FFI_LogicalExtensionCodec,
}

#[pymethods]
impl ByReferenceCodec {
    #[getter]
    fn __datafusion_codec_id__(&self) -> &'static str {
        "einfold_spike.by_reference.v1"
    }

    #[pyo3(signature = (session=None))]
    fn __datafusion_logical_extension_codec__<'py>(
        &self,
        py: Python<'py>,
        session: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let _ = session;
        PyCapsule::new_with_value(py, self.codec.clone(), c"datafusion_logical_extension_codec")
    }
}

/// The physical plans [`ByReference`] has encoded, by index.
static PLANS: Mutex<Vec<Arc<dyn ExecutionPlan>>> = Mutex::new(Vec::new());

impl PhysicalExtensionCodec for ByReference {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        _ctx: &TaskContext,
        _proto_converter: &dyn PhysicalProtoConverterExtension,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let Some(index) = buf.strip_prefix(MAGIC).and_then(|b| b.try_into().ok()) else {
            return plan_err!("not a plan einfold's spike codec encoded");
        };
        let plan = {
            let plans = PLANS.lock().expect("no panic while holding the plans");
            match plans.get(u64::from_le_bytes(index) as usize) {
                Some(plan) => Arc::clone(plan),
                None => return plan_err!("einfold's spike codec holds no such plan"),
            }
        };
        // The node, over its children as the decoder rebuilt them.
        match inputs.is_empty() {
            true => Ok(plan),
            // with_new_children is deprecated for replace_children in DataFusion 55,
            // which a spike need not follow.
            #[allow(deprecated)]
            false => plan.with_new_children(inputs.to_vec()),
        }
    }

    fn try_encode(
        &self,
        node: Arc<dyn ExecutionPlan>,
        buf: &mut Vec<u8>,
        _proto_converter: &dyn PhysicalProtoConverterExtension,
    ) -> Result<()> {
        let mut plans = PLANS.lock().expect("no panic while holding the plans");
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&(plans.len() as u64).to_le_bytes());
        plans.push(node);
        Ok(())
    }
}

/// [`ByReference`] as a physical codec datafusion-python can install.
#[pyclass(name = "ByReferencePhysicalCodec", module = "einfold_spike")]
struct ByReferencePhysicalCodec {
    codec: FFI_PhysicalExtensionCodec,
}

#[pymethods]
impl ByReferencePhysicalCodec {
    #[getter]
    fn __datafusion_codec_id__(&self) -> &'static str {
        "einfold_spike.by_reference_physical.v1"
    }

    #[pyo3(signature = (session=None))]
    fn __datafusion_physical_extension_codec__<'py>(
        &self,
        py: Python<'py>,
        session: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let _ = session;
        PyCapsule::new_with_value(py, self.codec.clone(), c"datafusion_physical_extension_codec")
    }
}

/// The Tokio runtime this library's FFI wrappers run einfold's operators on.
/// The library holds its own copy of Tokio, so the host's runtime isn't
/// visible to it.
fn runtime() -> tokio::runtime::Handle {
    static RUNTIME: std::sync::OnceLock<tokio::runtime::Runtime> = std::sync::OnceLock::new();
    RUNTIME
        .get_or_init(|| tokio::runtime::Runtime::new().expect("a Tokio runtime"))
        .handle()
        .clone()
}

/// A planned plan, shown to the host as one node with no children, so that
/// returning it through the FFI serializes nothing but [`ByReference`]'s
/// index. DataFusion's protobuf encodes an in-memory scan with its data, so
/// a plan returned node by node copies every table it reads. Its nodes stay
/// in this library, and run on this library's runtime, entered whenever the
/// host polls it.
#[derive(Debug)]
struct Opaque(Arc<dyn ExecutionPlan>);

impl datafusion::physical_plan::DisplayAs for Opaque {
    fn fmt_as(
        &self,
        _t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let inner = datafusion::physical_plan::displayable(self.0.as_ref()).indent(false);
        write!(f, "einfold's plan:\n{inner}")
    }
}

impl ExecutionPlan for Opaque {
    fn name(&self) -> &str {
        "Opaque"
    }

    fn properties(&self) -> &Arc<datafusion::physical_plan::PlanProperties> {
        self.0.properties()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(
            &Arc<dyn datafusion::physical_expr::PhysicalExpr>,
        ) -> Result<datafusion::common::tree_node::TreeNodeRecursion>,
    ) -> Result<datafusion::common::tree_node::TreeNodeRecursion> {
        Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<datafusion::execution::SendableRecordBatchStream> {
        let handle = runtime();
        let inner = {
            let _entered = handle.enter();
            self.0.execute(partition, context)?
        };
        let schema = inner.schema();
        let stream = Entered { inner, handle };
        Ok(Box::pin(
            datafusion::physical_plan::stream::RecordBatchStreamAdapter::new(schema, stream),
        ))
    }
}

/// A stream polled inside a runtime.
struct Entered {
    inner: datafusion::execution::SendableRecordBatchStream,
    handle: tokio::runtime::Handle,
}

impl futures::Stream for Entered {
    type Item = Result<datafusion::arrow::array::RecordBatch>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let _entered = self.handle.enter();
        self.inner.as_mut().poll_next(cx)
    }
}

/// The host's session, but with this library's own physical optimizer
/// rules: the host's reach a plan built here only as FFI proxies of its
/// nodes, which they can't optimize, and each call crosses the FFI.
struct LocalRules<'a> {
    inner: &'a dyn Session,
    rules: Vec<Arc<dyn datafusion::physical_optimizer::PhysicalOptimizerRule + Send + Sync>>,
    table_options: datafusion::common::config::TableOptions,
}

#[async_trait]
impl Session for LocalRules<'_> {
    fn session_id(&self) -> &str {
        self.inner.session_id()
    }
    fn config(&self) -> &datafusion::execution::config::SessionConfig {
        self.inner.config()
    }
    fn config_options(&self) -> &datafusion::common::config::ConfigOptions {
        self.inner.config_options()
    }
    fn catalog_list(&self) -> Arc<dyn datafusion::catalog::CatalogProviderList> {
        self.inner.catalog_list()
    }
    fn query_planner(&self) -> Arc<dyn QueryPlanner + Send + Sync> {
        self.inner.query_planner()
    }
    fn optimize(&self, plan: &LogicalPlan) -> Result<LogicalPlan> {
        self.inner.optimize(plan)
    }
    fn physical_optimizers(
        &self,
    ) -> &[Arc<dyn datafusion::physical_optimizer::PhysicalOptimizerRule + Send + Sync>] {
        &self.rules
    }
    async fn create_physical_plan(&self, logical_plan: &LogicalPlan) -> Result<Arc<dyn ExecutionPlan>> {
        self.inner.create_physical_plan(logical_plan).await
    }
    fn create_physical_expr(
        &self,
        expr: datafusion::logical_expr::Expr,
        df_schema: &datafusion::common::DFSchema,
    ) -> Result<Arc<dyn datafusion::physical_expr::PhysicalExpr>> {
        self.inner.create_physical_expr(expr, df_schema)
    }
    fn scalar_functions(&self) -> &std::collections::HashMap<String, Arc<datafusion::logical_expr::ScalarUDF>> {
        self.inner.scalar_functions()
    }
    fn higher_order_functions(
        &self,
    ) -> &std::collections::HashMap<String, Arc<datafusion::logical_expr::HigherOrderUDF>> {
        self.inner.higher_order_functions()
    }
    fn aggregate_functions(
        &self,
    ) -> &std::collections::HashMap<String, Arc<datafusion::logical_expr::AggregateUDF>> {
        self.inner.aggregate_functions()
    }
    fn window_functions(&self) -> &std::collections::HashMap<String, Arc<datafusion::logical_expr::WindowUDF>> {
        self.inner.window_functions()
    }
    fn extension_type_registry(&self) -> &datafusion::logical_expr::registry::ExtensionTypeRegistryRef {
        self.inner.extension_type_registry()
    }
    fn runtime_env(&self) -> &Arc<datafusion::execution::runtime_env::RuntimeEnv> {
        self.inner.runtime_env()
    }
    fn execution_props(&self) -> &datafusion::logical_expr::execution_props::ExecutionProps {
        self.inner.execution_props()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self.inner.as_any()
    }
    fn table_options(&self) -> &datafusion::common::config::TableOptions {
        self.inner.table_options()
    }
    fn table_options_mut(&mut self) -> &mut datafusion::common::config::TableOptions {
        &mut self.table_options
    }
    fn task_ctx(&self) -> Arc<TaskContext> {
        self.inner.task_ctx()
    }
}

/// Counts what the planner saw, for the spike's report.
#[derive(Debug, Default)]
struct Counts {
    plans: AtomicUsize,
    with_folds: AtomicUsize,
}

/// The session's planner, with einfold's detection in front of it.
#[derive(Debug)]
struct Bridge {
    fallback: Arc<dyn QueryPlanner + Send + Sync>,
    counts: Arc<Counts>,
}

#[async_trait]
impl QueryPlanner for Bridge {
    async fn create_physical_plan(
        &self,
        plan: &LogicalPlan,
        session: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        use datafusion::common::tree_node::TreeNode;
        use datafusion::optimizer::{OptimizerContext, OptimizerRule};
        self.counts.plans.fetch_add(1, Ordering::SeqCst);
        let rule = DetectFolds::default();
        let config = OptimizerContext::new();
        let detected = relink_sum(plan.clone())?
            .data
            .transform_up(|node| rule.rewrite(node, &config))?;
        let started = std::time::Instant::now();
        if std::env::var("EINFOLD_SPIKE_DEBUG").is_ok() {
            eprintln!(
                "session: {} physical optimizer rules, target partitions {}",
                session.physical_optimizers().len(),
                session.config().target_partitions()
            );
        }
        let debug = std::env::var("EINFOLD_SPIKE_DEBUG").is_ok();
        let nodes = || {
            let mut n = 0;
            let _ = plan.apply(|_| {
                n += 1;
                Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
            });
            n
        };
        let found = detected.transformed;
        if found {
            self.counts.with_folds.fetch_add(1, Ordering::SeqCst);
        } else if std::env::var("EINFOLD_SPIKE_HOST_PLANNER").is_ok() {
            // The session's own planner, through the FFI.
            return self.fallback.create_physical_plan(plan, session).await;
        }
        // Every plan is planned here, so that none crosses back as protobuf.
        let local = LocalRules {
            inner: session,
            rules: datafusion::physical_optimizer::optimizer::PhysicalOptimizer::new().rules,
            table_options: session.table_options().clone(),
        };
        let physical = DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(EinFoldPlanner)])
            .create_physical_plan(&detected.data, &local)
            .await?;
        if debug {
            let shown = datafusion::physical_plan::displayable(physical.as_ref()).indent(false).to_string();
            let execs: Vec<&str> = shown.lines().filter(|l| l.contains("EinFoldExec")).map(|l| l.trim()).collect();
            eprintln!("plan of {} nodes: {:?}, planned in {:.2} ms", nodes(), execs, started.elapsed().as_secs_f64() * 1e3);
        }
        Ok(Arc::new(Opaque(physical)))
    }
}

/// einfold, as an extension bundle for `SessionContext.with_extensions`.
#[pyclass(name = "EinFold", module = "einfold_spike")]
#[derive(Default)]
struct EinFold {
    counts: Arc<Counts>,
}

#[pymethods]
impl EinFold {
    #[new]
    fn new() -> Self {
        EinFold::default()
    }

    /// How many plans the planner was given.
    fn plans(&self) -> usize {
        self.counts.plans.load(Ordering::SeqCst)
    }

    /// How many of them einfold found a fold in.
    fn plans_with_folds(&self) -> usize {
        self.counts.with_folds.load(Ordering::SeqCst)
    }

    /// How many tables the codec has passed by reference.
    fn tables_passed(&self) -> usize {
        TABLES.lock().map_or(0, |t| t.len())
    }

    fn __datafusion_session_components__<'py>(
        &self,
        py: Python<'py>,
        ctx: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let provider = capsule(&ctx, "__datafusion_task_context_provider__")?;
        // SAFETY: capsules of this name hold an FFI_TaskContextProvider.
        let provider: FFI_TaskContextProvider =
            unsafe { capsule_value(&provider, c"datafusion_task_context_provider")? };
        let logical = FFI_LogicalExtensionCodec::new(Arc::new(ByReference), Some(runtime()), provider.clone());
        let logical = Py::new(py, ByReferenceCodec { codec: logical })?;
        let physical = FFI_PhysicalExtensionCodec::new(Arc::new(ByReference), Some(runtime()), provider);
        let physical = Py::new(py, ByReferencePhysicalCodec { codec: physical })?;
        let components = py.import("datafusion")?.getattr("SessionExtensionComponents")?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("logical_extension_codecs", (logical,))?;
        kwargs.set_item("physical_extension_codecs", (physical,))?;
        components.call((), Some(&kwargs))
    }

    fn __datafusion_session_planner__<'py>(
        &self,
        py: Python<'py>,
        ctx: Bound<'py, PyAny>,
        fallback: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let planner = capsule(&fallback, "__datafusion_query_planner__")?;
        // SAFETY: capsules of this name hold an FFI_QueryPlanner.
        let fallback: FFI_QueryPlanner = unsafe { capsule_value(&planner, c"datafusion_query_planner")? };
        check_version("the session's query planner", unsafe { (fallback.version)() })?;
        let logical = capsule(&ctx, "__datafusion_logical_extension_codec__")?;
        // SAFETY: capsules of this name hold an FFI_LogicalExtensionCodec.
        let logical: FFI_LogicalExtensionCodec =
            unsafe { capsule_value(&logical, c"datafusion_logical_extension_codec")? };
        let physical = capsule(&ctx, "__datafusion_physical_extension_codec__")?;
        // SAFETY: capsules of this name hold an FFI_PhysicalExtensionCodec.
        let physical: FFI_PhysicalExtensionCodec =
            unsafe { capsule_value(&physical, c"datafusion_physical_extension_codec")? };
        let bridge = Bridge {
            fallback: (&fallback).into(),
            counts: Arc::clone(&self.counts),
        };
        let ffi = FFI_QueryPlanner::new_with_ffi_codecs(Arc::new(bridge), logical, physical);
        PyCapsule::new_with_value(py, ffi, c"datafusion_query_planner")
    }
}

#[pymodule]
fn einfold_spike(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EinFold>()?;
    m.add_class::<ByReferenceCodec>()?;
    m.add_class::<ByReferencePhysicalCodec>()
}
