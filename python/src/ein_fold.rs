// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`EinFold`]: einfold as an extension datafusion-python installs in a
//! session.

use std::sync::atomic::Ordering;
use std::sync::Arc;

use datafusion_ffi::execution::FFI_TaskContextProvider;
use datafusion_ffi::proto::logical_extension_codec::FFI_LogicalExtensionCodec;
use datafusion_ffi::proto::physical_extension_codec::FFI_PhysicalExtensionCodec;
use datafusion_ffi::query_planner::FFI_QueryPlanner;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict};

use crate::bridge::{self, Bridge, Counts};
use crate::by_reference::{self, ByReference};
use crate::capsule::{self, check_version};
use crate::logical_codec::{LogicalCodec, PhysicalCodec};
use crate::stack;

/// einfold, as an extension for DataFusion's `SessionContext.with_extensions`.
/// It counts the plans it was given and those it ran folds or pipelines in.
#[pyclass(name = "EinFold", module = "einfold._native", frozen)]
#[derive(Default)]
pub(crate) struct EinFold {
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
        check_version("the session's query planner", unsafe {
            (session_planner.version)()
        })?;
        let logical = capsule::capsule(&ctx, "__datafusion_logical_extension_codec__")?;
        // SAFETY: capsules of this name hold an FFI_LogicalExtensionCodec.
        let logical: &FFI_LogicalExtensionCodec =
            unsafe { capsule::capsule_value(&logical, c"datafusion_logical_extension_codec")? };
        check_version("the session's logical codec", unsafe {
            (logical.version)()
        })?;
        let physical = capsule::capsule(&ctx, "__datafusion_physical_extension_codec__")?;
        // SAFETY: capsules of this name hold an FFI_PhysicalExtensionCodec.
        let physical: &FFI_PhysicalExtensionCodec =
            unsafe { capsule::capsule_value(&physical, c"datafusion_physical_extension_codec")? };
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
