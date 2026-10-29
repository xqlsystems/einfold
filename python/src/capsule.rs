// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Reading the capsules datafusion-python hands an extension: Python
//! objects that each hold a pointer to one of DataFusion's FFI structs.

use std::ffi::CStr;
use std::ptr::NonNull;

use pyo3::exceptions::PyImportError;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

/// `obj`'s capsule from its `getter`, or `obj` itself if it has no getter.
pub(crate) fn capsule<'py>(
    obj: &Bound<'py, PyAny>,
    getter: &str,
) -> PyResult<Bound<'py, PyCapsule>> {
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
pub(crate) unsafe fn capsule_value<'a, T>(
    capsule: &'a Bound<'_, PyCapsule>,
    name: &CStr,
) -> PyResult<&'a T> {
    let data: NonNull<T> = capsule.pointer_checked(Some(name))?.cast();
    // SAFETY: the caller guarantees that the capsule holds a `T`, and
    // `pointer_checked` that it is the capsule of that name. The capsule
    // owns the value, and outlives the borrow.
    Ok(unsafe { data.as_ref() })
}

/// Fails unless a foreign object was built against this DataFusion's FFI.
pub(crate) fn check_version(kind: &str, found: u64) -> PyResult<()> {
    let expected = datafusion_ffi::version();
    match found == expected {
        true => Ok(()),
        false => Err(PyImportError::new_err(format!(
            "{kind} has DataFusion FFI version {found}; einfold is built for {expected}"
        ))),
    }
}
