// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! einfold for Python: an extension for datafusion-python sessions.
//!
//! datafusion-python, Apache DataFusion's Python bindings, installs an
//! extension in a session with `SessionContext.with_extensions(extension)`.
//! It asks the extension for two things through DataFusion's FFI, its
//! stable C interface between separately built libraries:
//!
//! - **components** (`__datafusion_session_components__`): here, codecs that
//!   pass tables and plans by reference (see [`codec`]);
//! - **a query planner** (`__datafusion_session_planner__`): here, [`Bridge`],
//!   which finds folds and fold pipelines in each plan, as einfold does in
//!   DataFusion, and plans them as `EinFoldExec`.
//!
//! A *fold* is a sum of products over a join, grouped by columns of its
//! inputs, such as a matrix product: `SELECT a.i, b.k, SUM(a.v * b.v) FROM a
//! JOIN b ON a.j = b.j GROUP BY a.i, b.k`. A *fold pipeline* is a fold whose
//! results weight a second sum, such as attention's scores and its weighted
//! average of the values, computed together tile by tile.
//!
//! The Python package exports [`EinFold`] as `einfold.datafusion.EinFold`.

use std::ffi::CStr;
use std::ptr::NonNull;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use datafusion_ffi::execution::FFI_TaskContextProvider;
use datafusion_ffi::proto::logical_extension_codec::FFI_LogicalExtensionCodec;
use datafusion_ffi::proto::physical_extension_codec::FFI_PhysicalExtensionCodec;
use datafusion_ffi::query_planner::FFI_QueryPlanner;
use pyo3::exceptions::PyImportError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};

mod bridge;
mod codec;
mod lean;
mod stack;

/// mimalloc, the allocator datafusion-python uses (see `Cargo.toml`).
#[global_allocator]
static ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

use bridge::{Bridge, Counts};
use codec::ByReference;

/// `obj`'s capsule from its `getter`, or `obj` itself if it has no getter.
fn capsule<'py>(obj: &Bound<'py, PyAny>, getter: &str) -> PyResult<Bound<'py, PyCapsule>> {
    let obj = match obj.hasattr(getter)? {
        true => obj.getattr(getter)?.call0()?,
        false => obj.clone(),
    };
    Ok(obj.cast_into::<PyCapsule>()?)
}

/// The value in capsule `capsule`, named `name`, borrowed for as long as the
/// capsule is.
///
/// # Safety
///
/// The capsule must hold a `T`, as datafusion-python's capsules of that name
/// do.
unsafe fn capsule_value<'a, T>(capsule: &'a Bound<'_, PyCapsule>, name: &CStr) -> PyResult<&'a T> {
    let data: NonNull<T> = capsule.pointer_checked(Some(name))?.cast();
    // SAFETY: the caller guarantees that the capsule holds a `T`, and
    // `pointer_checked` that it is the capsule of that name. The capsule
    // owns the value, and outlives the borrow.
    Ok(unsafe { data.as_ref() })
}

/// Fails unless a foreign object was built against this DataFusion's FFI.
fn check_version(kind: &str, found: u64) -> PyResult<()> {
    let expected = datafusion_ffi::version();
    match found == expected {
        true => Ok(()),
        false => Err(PyImportError::new_err(format!(
            "{kind} has DataFusion FFI version {found}; einfold is built for {expected}"
        ))),
    }
}

/// [`ByReference`] as a logical codec datafusion-python can install.
#[pyclass(name = "LogicalCodec", module = "einfold._native", frozen)]
struct LogicalCodec {
    codec: FFI_LogicalExtensionCodec,
}

#[pymethods]
impl LogicalCodec {
    #[getter]
    fn __datafusion_codec_id__(&self) -> &'static str {
        "einfold.by_reference.v1"
    }

    #[pyo3(signature = (session=None))]
    fn __datafusion_logical_extension_codec__<'py>(
        &self,
        py: Python<'py>,
        session: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let _ = session;
        PyCapsule::new_with_value(
            py,
            self.codec.clone(),
            c"datafusion_logical_extension_codec",
        )
    }
}

/// [`ByReference`] as a physical codec datafusion-python can install.
#[pyclass(name = "PhysicalCodec", module = "einfold._native", frozen)]
struct PhysicalCodec {
    codec: FFI_PhysicalExtensionCodec,
}

#[pymethods]
impl PhysicalCodec {
    #[getter]
    fn __datafusion_codec_id__(&self) -> &'static str {
        "einfold.by_reference_physical.v1"
    }

    #[pyo3(signature = (session=None))]
    fn __datafusion_physical_extension_codec__<'py>(
        &self,
        py: Python<'py>,
        session: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let _ = session;
        PyCapsule::new_with_value(
            py,
            self.codec.clone(),
            c"datafusion_physical_extension_codec",
        )
    }
}

/// einfold, as an extension for DataFusion's `SessionContext.with_extensions`.
/// It counts the plans it was given and those it ran folds or pipelines in.
#[pyclass(name = "EinFold", module = "einfold._native", frozen)]
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
        codec::tables_pending()
    }

    /// How many plans einfold's codec holds, encoded and not yet decoded:
    /// 0 between queries.
    #[staticmethod]
    fn plans_pending() -> usize {
        codec::plans_pending()
    }

    fn __datafusion_session_components__<'py>(
        &self,
        py: Python<'py>,
        ctx: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let provider = capsule(&ctx, "__datafusion_task_context_provider__")?;
        // SAFETY: capsules of this name hold an FFI_TaskContextProvider.
        let provider: &FFI_TaskContextProvider =
            unsafe { capsule_value(&provider, c"datafusion_task_context_provider")? };
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
        let kwargs = PyDict::new(py);
        kwargs.set_item(
            "logical_extension_codecs",
            (Py::new(py, LogicalCodec { codec: logical })?,),
        )?;
        kwargs.set_item(
            "physical_extension_codecs",
            (Py::new(py, PhysicalCodec { codec: physical })?,),
        )?;
        components.call((), Some(&kwargs))
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
        let planner = capsule(&fallback, "__datafusion_query_planner__")?;
        // Each foreign struct's version is checked before it's cloned, since
        // cloning calls the struct's own functions, which a struct of
        // another FFI version may lay out elsewhere. Reading `version` where
        // this FFI version puts it is the best check this side can make.
        // SAFETY, for each `version` call: `version` is a plain function the
        // FFI struct carries.
        //
        // SAFETY: capsules of this name hold an FFI_QueryPlanner.
        let session_planner: &FFI_QueryPlanner =
            unsafe { capsule_value(&planner, c"datafusion_query_planner")? };
        check_version("the session's query planner", unsafe {
            (session_planner.version)()
        })?;
        let logical = capsule(&ctx, "__datafusion_logical_extension_codec__")?;
        // SAFETY: capsules of this name hold an FFI_LogicalExtensionCodec.
        let logical: &FFI_LogicalExtensionCodec =
            unsafe { capsule_value(&logical, c"datafusion_logical_extension_codec")? };
        check_version("the session's logical codec", unsafe {
            (logical.version)()
        })?;
        let physical = capsule(&ctx, "__datafusion_physical_extension_codec__")?;
        // SAFETY: capsules of this name hold an FFI_PhysicalExtensionCodec.
        let physical: &FFI_PhysicalExtensionCodec =
            unsafe { capsule_value(&physical, c"datafusion_physical_extension_codec")? };
        check_version("the session's physical codec", unsafe {
            (physical.version)()
        })?;
        let (logical, physical) = (logical.clone(), physical.clone());
        let bridge = Bridge {
            counts: Arc::clone(&self.counts),
        };
        let ffi = FFI_QueryPlanner::new_with_ffi_codecs(Arc::new(bridge), logical, physical);
        PyCapsule::new_with_value(py, ffi, c"datafusion_query_planner")
    }
}

/// The native part of the `einfold` Python package.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<EinFold>()?;
    m.add_class::<LogicalCodec>()?;
    m.add_class::<PhysicalCodec>()?;
    m.add("DATAFUSION_VERSION", datafusion::DATAFUSION_VERSION)?;
    Ok(())
}
