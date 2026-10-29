// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! einfold's engine-independent representations.
//!
//! Nothing in this crate depends on a query engine, so every engine
//! integration can share it.

pub mod expression;
pub mod fold;
pub mod pipeline;

pub use expression::{Binary, Constant, Expression, Unary};
pub use fold::{
    Aggregate, ColumnIndex, Comparison, Condition, Dimension, DimensionId, Domain, DomainKey,
    Finish, Fold, InvalidFold, Key, KeyEquality, Monoid, Operand, Output, RowValue,
};
pub use pipeline::{Normalizer, Pipeline, Scale, Shift};
