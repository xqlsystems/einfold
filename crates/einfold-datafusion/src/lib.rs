// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! einfold for Apache DataFusion.
//!
//! einfold speeds up tensor contractions written in SQL. A contraction, such
//! as a matrix product, is a join followed by `SUM(product)` grouped by the
//! output's coordinates: a *fold over a join*. SQL engines build the join's
//! rows before summing them; einfold sums each joined pair as it is found.
//!
//! A plan producer that knows it is writing a fold, such as ddx (an XQL
//! Systems project for automatic differentiation of SQL queries), puts a
//! [`FoldNode`] in its logical plan. [`EinFoldQueryPlanner`] plans the node as
//! [`EinFoldExec`]. This is the only einfold crate that depends on a query
//! engine.

mod exec;
mod hash;
mod node;

pub use exec::{EinFoldExec, EinFoldPlanner, EinFoldQueryPlanner};
pub use node::FoldNode;

/// The DataFusion version this crate is built against. ddx must link the same.
pub const DATAFUSION_VERSION: &str = datafusion::DATAFUSION_VERSION;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_datafusion_version() {
        assert_eq!(DATAFUSION_VERSION, "55.2.0");
    }
}
