// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`PhysicalCodec`]: einfold's by-reference physical codec (see
//! [`ByReference`]) as the Python object datafusion-python installs.

use datafusion_ffi::proto::physical_extension_codec::FFI_PhysicalExtensionCodec;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

#[cfg(doc)]
use crate::by_reference::ByReference;

/// [`ByReference`] as a physical codec datafusion-python can install.
#[pyclass(name = "PhysicalCodec", module = "einfold._native", frozen)]
pub(crate) struct PhysicalCodec {
    pub(crate) codec: FFI_PhysicalExtensionCodec,
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
