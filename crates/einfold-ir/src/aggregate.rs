// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Aggregates: how a fold combines the values of a group's rows.
//!
//! Most SQL aggregates fold the group's values with a single [`Op`]: `SUM`
//! with `+`, `MIN` with `min`, `MAX` with `max`, `BOOL_OR` with `OR`,
//! `BOOL_AND` with `AND`. [`Aggregate::Fold`] covers that whole class at once.
//! Two common aggregates are not a single operation, and are named
//! separately: `COUNT`, which sums a 1 per non-NULL value, and `AVG`, which is
//! the ratio of a `SUM` and a `COUNT`.
//!
//! All of them skip NULL inputs, and each carries SQL's rule for a group whose
//! every input was NULL. A group with *no* rows never exists in a `GROUP BY`
//! result, so that case doesn't arise.

use std::fmt;

use crate::algebra::Op;

/// An aggregate a fold can compute, with its SQL semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Aggregate {
    /// Fold the non-NULL values with one operation; NULL if there are none.
    /// `SUM` is `Fold(Op::Add)`, `MIN` is `Fold(Op::Min)`, `MAX` is
    /// `Fold(Op::Max)`, `BOOL_OR` is `Fold(Op::Or)`, and `BOOL_AND` is
    /// `Fold(Op::And)`.
    Fold(Op),
    /// `COUNT(x)`: the number of non-NULL values, so 0 if all are NULL.
    /// `COUNT(*)` is `COUNT` of a value that is never NULL. It is the sum of
    /// one 1 per non-NULL value.
    Count,
    /// `AVG(x)`: `SUM(x) / COUNT(x)`, or NULL if every value is NULL. It is
    /// not a single operation, but is computed exactly from two that are; see
    /// [`Aggregate::parts`].
    Avg,
}

impl Aggregate {
    /// SQL's `SUM`.
    pub const SUM: Aggregate = Aggregate::Fold(Op::Add);
    /// SQL's `MIN`.
    pub const MIN: Aggregate = Aggregate::Fold(Op::Min);
    /// SQL's `MAX`.
    pub const MAX: Aggregate = Aggregate::Fold(Op::Max);
    /// SQL's `BOOL_OR`.
    pub const BOOL_OR: Aggregate = Aggregate::Fold(Op::Or);
    /// SQL's `BOOL_AND`.
    pub const BOOL_AND: Aggregate = Aggregate::Fold(Op::And);

    /// The SQL name of the aggregate.
    pub fn sql_name(self) -> &'static str {
        match self {
            Aggregate::Fold(Op::Add) => "SUM",
            Aggregate::Fold(Op::Mul) => "PRODUCT",
            Aggregate::Fold(Op::Min) => "MIN",
            Aggregate::Fold(Op::Max) => "MAX",
            Aggregate::Fold(Op::Or) => "BOOL_OR",
            Aggregate::Fold(Op::And) => "BOOL_AND",
            Aggregate::Count => "COUNT",
            Aggregate::Avg => "AVG",
        }
    }

    /// For an aggregate computed from others, the aggregates it is computed
    /// from: `AVG` is `SUM / COUNT`. `None` for aggregates that are a single
    /// fold.
    pub fn parts(self) -> Option<[Aggregate; 2]> {
        match self {
            Aggregate::Avg => Some([Aggregate::SUM, Aggregate::Count]),
            _ => None,
        }
    }

    /// Whether the aggregate's result is exact whatever order its float
    /// inputs are combined in. Rewrites may change that order, so an inexact
    /// result may change in its last bits; an exact one may never change.
    /// `MIN`, `MAX`, the logical aggregates and `COUNT` are exact; `SUM`,
    /// a product, and `AVG` of floats are not.
    pub fn is_exact(self) -> bool {
        match self {
            Aggregate::Fold(op) => !matches!(op, Op::Add | Op::Mul),
            Aggregate::Count => true,
            Aggregate::Avg => false,
        }
    }
}

impl fmt::Display for Aggregate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.sql_name())
    }
}
