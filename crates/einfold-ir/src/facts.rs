// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Facts: what einfold knows about an operand, and how it knows it.
//!
//! A fact might be a table's row count, or that a column has no NULLs. Facts
//! differ in how far they can be trusted: a count read from metadata is
//! exact, while one estimated from statistics is a guess. einfold may use a
//! fact to decide that a rewrite is *correct* only when it is exact; guesses
//! may only influence which correct plan is *cheaper*. For now this module
//! defines only how a fact is known; the kinds of fact come later.

/// How a fact is known.
///
/// Ordered from weakest to strongest guarantee about correctness: only
/// [`Precision::Exact`] facts may affect whether a rewrite is correct.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Precision {
    /// A best guess from statistics. Usable for costs only.
    Estimate,
    /// A guaranteed upper limit. Usable for memory decisions.
    Bound,
    /// Observed while running. Replaces Estimates and Bounds for the rest of
    /// the query.
    Measured,
    /// Known from metadata or a reader's guarantee. Usable for correctness.
    Exact,
}

/// A value together with how it is known.
#[derive(Clone, Debug, PartialEq)]
pub struct Fact<T> {
    /// The value.
    pub value: T,
    /// How the value is known.
    pub precision: Precision,
}

impl<T> Fact<T> {
    /// A fact known exactly.
    pub fn exact(value: T) -> Self {
        Fact {
            value,
            precision: Precision::Exact,
        }
    }

    /// Whether this fact may be used to decide correctness.
    pub fn is_exact(&self) -> bool {
        self.precision == Precision::Exact
    }
}
