// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Aggregates: how a fold combines the values of a group's rows.
//!
//! An aggregate such as `SUM` reduces many values to one. einfold can only
//! split that work into pieces (per partition, per chunk, per input of a join)
//! and combine the pieces later if combining is associative and commutative,
//! with an identity: in algebra, a *commutative monoid*. Every aggregate here
//! is one. Each also carries SQL's rules for NULL inputs and empty groups,
//! which einfold must reproduce exactly.

use std::fmt;

/// An aggregate a fold can compute, with its SQL semantics.
///
/// All of them skip NULL inputs. They differ in what a group whose every input
/// was NULL returns, and in whether their result is exact. A group with *no*
/// rows never exists in a `GROUP BY` result, so that case doesn't arise.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Aggregate {
    /// `SUM(x)`: the sum of the non-NULL inputs, or NULL if there are none.
    /// Over floats it is not exact: the order of additions changes the last
    /// bits.
    Sum,
    /// `COUNT(x)`: the number of non-NULL inputs, so 0 if all are NULL.
    /// `COUNT(*)` is `COUNT` of a value that is never NULL. Always exact.
    Count,
    /// `AVG(x)`: the sum of the non-NULL inputs divided by their count, or NULL
    /// if there are none. Not exact over floats.
    Avg,
}

impl Aggregate {
    /// The SQL name of the aggregate.
    pub fn sql_name(self) -> &'static str {
        match self {
            Aggregate::Sum => "SUM",
            Aggregate::Count => "COUNT",
            Aggregate::Avg => "AVG",
        }
    }

    /// Whether the aggregate's result is exact, whatever order its inputs
    /// are combined in. Rewrites may change that order, so an inexact result
    /// may change in its last bits; an exact one may never change.
    pub fn is_exact(self) -> bool {
        matches!(self, Aggregate::Count)
    }
}

impl fmt::Display for Aggregate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.sql_name())
    }
}
