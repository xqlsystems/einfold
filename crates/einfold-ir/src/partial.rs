// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Partial aggregates: the state of one output group, computed over part of
//! the input and combined later.
//!
//! An engine rarely aggregates a group in one go. It aggregates pieces in
//! parallel, per partition, per chunk, or per input of a join, and combines
//! the pieces at the end. Every piece must carry enough state for the combined
//! result to equal what SQL would have returned over all the rows.
//!
//! That state has two independent parts:
//!
//! - **whether the group exists.** SQL creates a group exactly when some
//!   joined row reaches it, whatever that row's value. This is the same for
//!   every aggregate;
//! - **the aggregate's own state:** a running sum, a count, a sum and a count.
//!
//! Keeping them apart gives every aggregate SQL's subtle rule for free: a group
//! reached only by rows whose values are NULL still exists, and its `SUM` is
//! NULL rather than missing.
//!
//! Sums here use plain `f64` addition. Float addition is not associative
//! (`(a + b) + c` can differ from `a + (b + c)` in the last bits), so the
//! order of [`update`](PartialAggregate::update) and
//! [`merge`](PartialAggregate::merge) calls decides a float result's last
//! bits. Callers that promise repeatable results must use a fixed order.

use crate::aggregate::Aggregate;

/// The state of one [`Aggregate`] over part of a group's values.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum AggregateState {
    /// `SUM`: the sum of the non-NULL values so far, or `None` if there were
    /// none yet.
    Sum(Option<f64>),
    /// `COUNT`: the number of non-NULL values so far.
    Count(i64),
    /// `AVG`: the sum and the count of the non-NULL values so far.
    Avg {
        /// Sum of the non-NULL values.
        sum: f64,
        /// Number of non-NULL values.
        count: i64,
    },
}

/// A finished aggregate value, in the type SQL gives that aggregate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AggregateValue {
    /// `SUM` and `AVG` of floats: a nullable float.
    Float(Option<f64>),
    /// `COUNT`: a non-NULL integer.
    Int(i64),
}

impl AggregateState {
    /// The state of `aggregate` before any value.
    pub fn empty(aggregate: Aggregate) -> Self {
        match aggregate {
            Aggregate::Sum => AggregateState::Sum(None),
            Aggregate::Count => AggregateState::Count(0),
            Aggregate::Avg => AggregateState::Avg { sum: 0.0, count: 0 },
        }
    }

    /// Add one value; `None` is a NULL value, which every aggregate skips.
    pub fn update(&mut self, value: Option<f64>) {
        let Some(v) = value else { return };
        match self {
            AggregateState::Sum(s) => *s = Some(s.map_or(v, |s| s + v)),
            AggregateState::Count(c) => *c += 1,
            AggregateState::Avg { sum, count } => {
                *sum += v;
                *count += 1;
            }
        }
    }

    /// Combine another state of the same aggregate into this one.
    ///
    /// # Panics
    ///
    /// If the two states belong to different aggregates.
    pub fn merge(&mut self, other: &AggregateState) {
        match (self, other) {
            (AggregateState::Sum(a), AggregateState::Sum(b)) => {
                if let Some(b) = b {
                    *a = Some(a.map_or(*b, |a| a + b));
                }
            }
            (AggregateState::Count(a), AggregateState::Count(b)) => *a += b,
            (AggregateState::Avg { sum, count }, AggregateState::Avg { sum: s2, count: c2 }) => {
                *sum += s2;
                *count += c2;
            }
            (a, b) => panic!("cannot merge {b:?} into {a:?}: different aggregates"),
        }
    }

    /// The aggregate's value: NULL for `SUM` and `AVG` when every value was
    /// NULL, and 0 for `COUNT`.
    pub fn finish(&self) -> AggregateValue {
        match *self {
            AggregateState::Sum(s) => AggregateValue::Float(s),
            AggregateState::Count(c) => AggregateValue::Int(c),
            AggregateState::Avg { sum, count } => {
                AggregateValue::Float((count > 0).then(|| sum / count as f64))
            }
        }
    }
}

/// The partial aggregate of one output group: whether any row has reached
/// it, and the aggregate's state over the values of the rows that have.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartialAggregate {
    reached: bool,
    state: AggregateState,
}

impl PartialAggregate {
    /// The state of a group no row has reached yet.
    pub fn new(aggregate: Aggregate) -> Self {
        PartialAggregate {
            reached: false,
            state: AggregateState::empty(aggregate),
        }
    }

    /// Fold in one joined row's value; `None` is a NULL value.
    ///
    /// The row marks the group as reached even when its value is NULL.
    pub fn update(&mut self, value: Option<f64>) {
        self.reached = true;
        self.state.update(value);
    }

    /// Combine another partial aggregate of the same group into this one.
    pub fn merge(&mut self, other: &PartialAggregate) {
        self.reached |= other.reached;
        self.state.merge(&other.state);
    }

    /// Whether any joined row reached the group.
    pub fn reached(&self) -> bool {
        self.reached
    }

    /// The final result: `None` if the group does not exist, and otherwise
    /// the aggregate's value.
    pub fn finish(&self) -> Option<AggregateValue> {
        self.reached.then(|| self.state.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use AggregateValue::{Float, Int};

    fn fold(aggregate: Aggregate, values: &[Option<f64>]) -> PartialAggregate {
        let mut p = PartialAggregate::new(aggregate);
        for v in values {
            p.update(*v);
        }
        p
    }

    #[test]
    fn unreached_group_does_not_exist() {
        for a in [Aggregate::Sum, Aggregate::Count, Aggregate::Avg] {
            assert_eq!(PartialAggregate::new(a).finish(), None);
        }
    }

    #[test]
    fn group_reached_only_by_nulls() {
        let nulls = [None, None];
        assert_eq!(fold(Aggregate::Sum, &nulls).finish(), Some(Float(None)));
        assert_eq!(fold(Aggregate::Count, &nulls).finish(), Some(Int(0)));
        assert_eq!(fold(Aggregate::Avg, &nulls).finish(), Some(Float(None)));
    }

    #[test]
    fn nulls_are_skipped() {
        let vs = [None, Some(2.0), None, Some(4.0)];
        assert_eq!(fold(Aggregate::Sum, &vs).finish(), Some(Float(Some(6.0))));
        assert_eq!(fold(Aggregate::Count, &vs).finish(), Some(Int(2)));
        assert_eq!(fold(Aggregate::Avg, &vs).finish(), Some(Float(Some(3.0))));
    }

    #[test]
    fn nan_propagates_but_is_counted() {
        let vs = [Some(1.0), Some(f64::NAN)];
        let Some(Float(Some(s))) = fold(Aggregate::Sum, &vs).finish() else {
            panic!()
        };
        assert!(s.is_nan());
        assert_eq!(fold(Aggregate::Count, &vs).finish(), Some(Int(2)));
    }

    #[test]
    fn merge_matches_sequential_updates() {
        let parts: [&[Option<f64>]; 4] = [&[Some(1.0)], &[None], &[], &[Some(4.0), Some(3.0)]];
        for a in [Aggregate::Sum, Aggregate::Count, Aggregate::Avg] {
            let mut merged = PartialAggregate::new(a);
            for p in parts {
                merged.merge(&fold(a, p));
            }
            let all: Vec<Option<f64>> = parts.concat();
            assert_eq!(merged.finish(), fold(a, &all).finish(), "{a}");
        }
        // Merging only empty pieces leaves the group nonexistent.
        let mut empty = PartialAggregate::new(Aggregate::Sum);
        empty.merge(&PartialAggregate::new(Aggregate::Sum));
        assert_eq!(empty.finish(), None);
    }
}
