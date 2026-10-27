// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! EinFold's algorithms: computing a fold over a join (see [`einfold_ir::fold`])
//! from Arrow arrays to Arrow arrays, without forming the join's rows.
//!
//! In SQL terms, a fold of operands `a` and `b` is
//!
//! ```sql
//! SELECT <outputs>, SUM(a.<factor> * b.<factor>)
//! FROM a JOIN b ON <shared dimensions> GROUP BY <outputs>
//! ```
//!
//! and these functions compute exactly that, SQL's corners included: NULL
//! keys and factors, duplicate rows, groups reached only by NULLs, NaN and
//! infinities. A *host*, the engine whose plan the fold came from, decides a
//! few things SQL leaves open, such as whether `-0.0` and `0.0` are one key.
//! [`Host`] says which.
//!
//! Nothing here depends on a query engine, so any engine that holds its data
//! as Arrow arrays can run these, from DataFusion to an extension of DuckDB.

use arrow::error::ArrowError;

mod algorithm;
mod dense;
mod discriminate;
mod keys;
mod options;
mod relation;
mod schema;

pub use algorithm::{fold, fold_dense, Algorithm, Folded};
pub use options::{DomainRows, FloatKeys, Host, Options};
pub use relation::Relation;
pub use schema::schema;

/// The result of a kernel.
pub type Result<T> = std::result::Result<T, ArrowError>;

/// An error for a fold that the inputs don't fit.
fn bad<T>(why: String) -> Result<T> {
    Err(ArrowError::InvalidArgumentError(why))
}
