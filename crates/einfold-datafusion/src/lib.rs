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
//! [`EinFoldRule`] finds such folds in DataFusion's optimized plans and
//! replaces each with a [`FoldNode`]; [`EinFoldQueryPlanner`] applies it just
//! before physical planning, and plans each node as [`EinFoldExec`], which
//! runs `einfold_kernels`. [`install`] adds it to a session. This is the only einfold crate that depends on a query
//! engine.

mod exec;
mod node;
mod rule;

use std::sync::Arc;

use datafusion::execution::SessionStateBuilder;

pub use exec::{EinFoldExec, EinFoldPlanner, EinFoldQueryPlanner};
pub use node::FoldNode;
pub use rule::EinFoldRule;

/// `builder` with einfold installed: an [`EinFoldQueryPlanner`] that applies
/// [`EinFoldRule`] to each optimized plan, then plans the nodes it makes.
pub fn install(builder: SessionStateBuilder) -> SessionStateBuilder {
    builder.with_query_planner(Arc::new(EinFoldQueryPlanner::detecting(
        EinFoldRule::default(),
    )))
}

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
