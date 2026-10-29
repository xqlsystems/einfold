// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`LogicalCodec`]: einfold's by-reference logical codec (see
//! [`ByReference`]) as the Python object datafusion-python installs.

use datafusion_ffi::proto::logical_extension_codec::FFI_LogicalExtensionCodec;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

#[cfg(doc)]
use crate::by_reference::ByReference;

/// [`ByReference`] as a logical codec datafusion-python can install.
#[pyclass(name = "LogicalCodec", module = "einfold._native", frozen)]
pub(crate) struct LogicalCodec {
    pub(crate) codec: FFI_LogicalExtensionCodec,
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
