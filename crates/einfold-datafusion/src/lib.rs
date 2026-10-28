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
//! A [`FoldNode`] in a logical plan stands for such a fold, and
//! [`EinFoldQueryPlanner`] plans it as [`EinFoldExec`], which runs
//! `einfold_kernels`. This is the only einfold crate that depends on a query
//! engine.

mod exec;
mod node;

pub use exec::{EinFoldExec, EinFoldPlanner, EinFoldQueryPlanner};
pub use node::FoldNode;

/// The DataFusion version this crate is built against.
pub const DATAFUSION_VERSION: &str = datafusion::DATAFUSION_VERSION;

/// What DataFusion decides that SQL leaves open, as `einfold_kernels` needs
/// it. `tests/datafusion_behaviors.rs` pins each choice.
pub const HOST: einfold_kernels::Host = einfold_kernels::Host {
    float_keys: einfold_kernels::FloatKeys::SignedZerosEqual,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_datafusion_version() {
        assert_eq!(DATAFUSION_VERSION, "55.2.0");
    }
}
