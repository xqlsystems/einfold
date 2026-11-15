// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`EinFold`]: einfold as an extension datafusion-python installs in a
//! session.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use datafusion::execution::cache::cache_manager::CacheManagerConfig;
use datafusion::execution::runtime_env::{RuntimeEnv, RuntimeEnvBuilder};
use datafusion_ffi::execution::FFI_TaskContextProvider;
use datafusion_ffi::proto::logical_extension_codec::FFI_LogicalExtensionCodec;
use datafusion_ffi::proto::physical_extension_codec::FFI_PhysicalExtensionCodec;
use datafusion_ffi::query_planner::FFI_QueryPlanner;
use einfold_kernels::DenseBackend;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};

use crate::bridge::{self, Bridge, Counts};
use crate::by_reference::{self, ByReference};
use crate::capsule;
use crate::dense_backend::PythonBackend;
use crate::logical_codec::LogicalCodec;
use crate::physical_codec::PhysicalCodec;
use crate::stack;

/// einfold, as an extension for DataFusion's `SessionContext.with_extensions`.
/// It counts the plans it was given and those it ran folds or pipelines in.
///
/// Experimental and temporary: `dense_backend`, a function, computes the
/// dense work einfold has detected and checked, in place of einfold's own
/// arithmetic, for this extension's sessions only. Each kind of work is one
/// of einfold's nodes, offered by name as
/// `dense_backend(kind, arrays, shape, scale, ranges)` with `pyarrow`
/// `float64` arrays; the function returns one, or `None` to decline:
///
/// - `kind == "matmul"`, a matrix product: `arrays` is `[a, b]`, `shape` is
///   `(batches, rows, inner, columns)`, and the result is `a @ b` for each
///   batch;
/// - `kind == "attention"`, a softmax-weighted average: `arrays` is `[q, kt,
///   v]`, `shape` is `(batches, queries, features, keys, values)`, `scale` is
///   `(0, 1.0)` for none, `(1, c)` for `* c` or `(2, c)` for `/ c`, and
///   `ranges[x]` is query `x`'s half-open range of keys, in every batch. The
///   result is the softmax of each query's scaled scores over its keys,
///   shifted by its largest, times the values.
///
/// Later versions may offer other kinds, such as convolutions, so the
/// function returns `None` for a kind it doesn't know. All arrays are
/// row-major, batch first. Work of fewer than `dense_min_ops`
/// multiplications, or whose arrays take more than `dense_max_bytes`, isn't
/// offered. The function's results are the session's answers: einfold
/// doesn't check their precision, so one that computes in `float32` changes
/// them by as much.
#[pyclass(name = "EinFold", module = "einfold._native", frozen)]
#[derive(Default)]
pub(crate) struct EinFold {
    counts: Arc<Counts>,
    backend: Option<Arc<dyn DenseBackend>>,
}

#[pymethods]
impl EinFold {
    #[new]
    #[pyo3(signature = (*, dense_backend=None, dense_min_ops=1 << 20, dense_max_bytes=512 << 20))]
    fn new(dense_backend: Option<Py<PyAny>>, dense_min_ops: usize, dense_max_bytes: usize) -> Self {
        let backend = dense_backend.map(|callback| {
            let backend = PythonBackend {
                callback,
                min_ops: dense_min_ops,
                max_bytes: dense_max_bytes,
            };
            Arc::new(backend) as Arc<dyn DenseBackend>
        });
        EinFold {
            counts: Arc::default(),
            backend,
        }
    }

    /// How many plans einfold planned.
    fn plans(&self) -> usize {
        self.counts.plans.load(Ordering::Relaxed)
    }

    /// How many of them held a fold or a fold pipeline einfold ran.
    fn plans_with_einfold(&self) -> usize {
        self.counts.with_einfold.load(Ordering::Relaxed)
    }

    /// How many of them held a fold pipeline, such as attention.
    fn plans_with_pipelines(&self) -> usize {
        self.counts.with_pipelines.load(Ordering::Relaxed)
    }

    /// How many tables einfold's codec holds, encoded and not yet decoded:
    /// 0 between queries.
    #[staticmethod]
    fn tables_pending() -> usize {
        by_reference::tables_pending()
    }

    /// How many plans einfold's codec holds, encoded and not yet decoded:
    /// 0 between queries.
    #[staticmethod]
    fn plans_pending() -> usize {
        by_reference::plans_pending()
    }

    fn __datafusion_session_components__<'py>(
        &self,
        py: Python<'py>,
        ctx: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let provider = capsule::capsule(&ctx, "__datafusion_task_context_provider__")?;
        // SAFETY: capsules of this name hold an FFI_TaskContextProvider.
        let provider: &FFI_TaskContextProvider =
            unsafe { capsule::capsule_value(&provider, c"datafusion_task_context_provider")? };
        let provider = provider.clone();
        let runtime = Some(bridge::runtime());
        let logical = FFI_LogicalExtensionCodec::new(
            Arc::new(ByReference),
            runtime.clone(),
            provider.clone(),
        );
        let physical = FFI_PhysicalExtensionCodec::new(Arc::new(ByReference), runtime, provider);
        let components = py
            .import("datafusion")?
            .getattr("SessionExtensionComponents")?;
        let keyword_arguments = PyDict::new(py);
        keyword_arguments.set_item(
            "logical_extension_codecs",
            (Py::new(py, LogicalCodec { codec: logical })?,),
        )?;
        keyword_arguments.set_item(
            "physical_extension_codecs",
            (Py::new(py, PhysicalCodec { codec: physical })?,),
        )?;
        components.call((), Some(&keyword_arguments))
    }

    fn __datafusion_session_planner__<'py>(
        &self,
        py: Python<'py>,
        ctx: Bound<'py, PyAny>,
        fallback: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        // Plans reach einfold as protobuf, decoded on the thread that asks
        // for them: see `stack`.
        stack::deepen();
        let planner = capsule::capsule(&fallback, "__datafusion_query_planner__")?;
        // Each foreign struct's version is checked before it's cloned, since
        // cloning calls the struct's own functions, which a struct of
        // another FFI version may lay out elsewhere. Reading `version` where
        // this FFI version puts it is the best check this side can make.
        // SAFETY, for each `version` call: `version` is a plain function the
        // FFI struct carries.
        //
        // SAFETY: capsules of this name hold an FFI_QueryPlanner.
        let session_planner: &FFI_QueryPlanner =
            unsafe { capsule::capsule_value(&planner, c"datafusion_query_planner")? };
        capsule::check_version("the session's query planner", unsafe {
            (session_planner.version)()
        })?;
        let logical = capsule::capsule(&ctx, "__datafusion_logical_extension_codec__")?;
        // SAFETY: capsules of this name hold an FFI_LogicalExtensionCodec.
        let logical: &FFI_LogicalExtensionCodec =
            unsafe { capsule::capsule_value(&logical, c"datafusion_logical_extension_codec")? };
        capsule::check_version("the session's logical codec", unsafe {
            (logical.version)()
        })?;
        let physical = capsule::capsule(&ctx, "__datafusion_physical_extension_codec__")?;
        // SAFETY: capsules of this name hold an FFI_PhysicalExtensionCodec.
        let physical: &FFI_PhysicalExtensionCodec =
            unsafe { capsule::capsule_value(&physical, c"datafusion_physical_extension_codec")? };
        capsule::check_version("the session's physical codec", unsafe {
            (physical.version)()
        })?;
        let (logical, physical) = (logical.clone(), physical.clone());
        let bridge = Bridge {
            counts: Arc::clone(&self.counts),
            backend: self.backend.clone(),
            runtime_env: session_runtime_env()?,
        };
        let ffi_planner =
            FFI_QueryPlanner::new_with_ffi_codecs(Arc::new(bridge), logical, physical);
        PyCapsule::new_with_value(py, ffi_planner, c"datafusion_query_planner")
    }
}

/// The runtime environment einfold plans and runs a session's plans with,
/// from query to query (see `Bridge::runtime_env`), with DataFusion's caches
/// except its cache of each directory's files.
///
/// The caches of each file's Parquet metadata and statistics check an entry
/// against the file's size and time of change, which a listing of its
/// directory gives, so they can't serve a file that changed. The listing
/// cache is the source of those, and DataFusion clears it where the session
/// drops or replaces a table, or where the session's own runtime settings
/// (`SET datafusion.runtime.list_files_cache_limit`) say so, none of which
/// reaches einfold through the FFI. Without it, einfold lists a table's
/// files on each query, so it never reads them staler than the session
/// does.
fn session_runtime_env() -> PyResult<Arc<RuntimeEnv>> {
    let caches = CacheManagerConfig::default().with_list_files_cache_limit(0);
    let runtime_env = RuntimeEnvBuilder::new()
        .with_cache_manager(caches)
        .build_arc()
        .map_err(|error| pyo3::exceptions::PyRuntimeError::new_err(error.to_string()))?;
    Ok(runtime_env)
}
