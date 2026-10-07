// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! einfold for Apache DataFusion.
//!
//! einfold speeds up tensor contractions written in SQL. A contraction, such
//! as a matrix product, is a join followed by `SUM(product)` grouped by the
//! output's coordinates. This crate finds such queries in DataFusion's plans
//! and runs them with a fused join-and-sum operator. It is the only einfold
//! crate that depends on a query engine.
//!
//! More generally, it finds *folds over joins*: a join followed by `SUM`,
//! `COUNT` or `AVG` of a product, grouped by some of the join's columns. The
//! matrix product is the case of `SUM`. [`mod@detect`] finds them.

pub mod detect;

pub use detect::{detect, FoldMatch, OperandInput};

/// The DataFusion version this crate is built against. ddx must link the same.
pub const DATAFUSION_VERSION: &str = datafusion::DATAFUSION_VERSION;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_datafusion_version() {
        assert_eq!(DATAFUSION_VERSION, "54.1.0");
    }
}
