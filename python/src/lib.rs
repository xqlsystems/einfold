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
//!   pass tables and plans by reference (see `by_reference`);
//! - **a query planner** (`__datafusion_session_planner__`): here, `Bridge`,
//!   which finds folds and fold pipelines in each plan, as einfold does in
//!   DataFusion, and plans them as `EinFoldExec`.
//!
//! A *fold* is a sum of products over a join, grouped by columns of its
//! inputs, such as a matrix product: `SELECT a.i, b.k, SUM(a.v * b.v) FROM a
//! JOIN b ON a.j = b.j GROUP BY a.i, b.k`. A *fold pipeline* is a fold whose
//! results weight a second sum, such as attention's scores and its weighted
//! average of the values, computed together tile by tile.
//!
//! The Python package exports `EinFold` as `einfold.datafusion.EinFold`.

use pyo3::prelude::*;

mod allocator;
mod bridge;
mod by_reference;
mod capsule;
mod ein_fold;
mod lean;
mod logical_codec;
mod physical_codec;
mod stack;

/// The native part of the `einfold` Python package.
#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ein_fold::EinFold>()?;
    m.add_class::<logical_codec::LogicalCodec>()?;
    m.add_class::<physical_codec::PhysicalCodec>()?;
    m.add("DATAFUSION_VERSION", datafusion::DATAFUSION_VERSION)?;
    Ok(())
}
