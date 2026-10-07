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

use std::cmp::Ordering;

use crate::aggregate::Aggregate;
use crate::algebra::Op;

/// The state of one [`Aggregate`] over part of a group's values.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum AggregateState {
    /// A numeric [`Aggregate::Fold`]: the non-NULL values so far, folded with
    /// `op`, or `None` if there were none yet.
    Fold {
        /// The folding operation: `+`, `*`, `min` or `max`.
        op: Op,
        /// The folded value so far.
        acc: Option<f64>,
    },
    /// A logical [`Aggregate::Fold`] (`BOOL_AND`, `BOOL_OR`): the non-NULL
    /// truth values so far, folded with `op`, or `None` if there were none.
    Logical {
        /// `AND` or `OR`.
        op: Op,
        /// The folded truth value so far.
        acc: Option<bool>,
    },
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
    /// Numeric folds and `AVG`: a nullable float.
    Float(Option<f64>),
    /// Logical folds: a nullable truth value.
    Bool(Option<bool>),
    /// `COUNT`: a non-NULL integer.
    Int(i64),
}

/// Combine two numbers with `op`. `min` and `max` order floats as Arrow and
/// DataFusion do: by IEEE 754's total order, where NaN is above every number
/// and `-0.0` is below `0.0`.
fn combine(op: Op, a: f64, b: f64) -> f64 {
    let pick = |want: Ordering| if a.total_cmp(&b) == want { a } else { b };
    match op {
        Op::Add => a + b,
        Op::Mul => a * b,
        Op::Min => pick(Ordering::Less),
        Op::Max => pick(Ordering::Greater),
        Op::And | Op::Or => unreachable!("logical operations fold truth values"),
    }
}

impl AggregateState {
    /// The state of `aggregate` before any value.
    pub fn empty(aggregate: Aggregate) -> Self {
        match aggregate {
            Aggregate::Fold(op) if op.is_logical() => AggregateState::Logical { op, acc: None },
            Aggregate::Fold(op) => AggregateState::Fold { op, acc: None },
            Aggregate::Count => AggregateState::Count(0),
            Aggregate::Avg => AggregateState::Avg { sum: 0.0, count: 0 },
        }
    }

    /// Add one value; `None` is a NULL value, which every aggregate skips.
    /// Logical aggregates read a non-zero value as `TRUE`.
    pub fn update(&mut self, value: Option<f64>) {
        let Some(v) = value else { return };
        match self {
            AggregateState::Fold { op, acc } => *acc = Some(acc.map_or(v, |a| combine(*op, a, v))),
            AggregateState::Logical { op, acc } => {
                let t = v != 0.0;
                *acc = Some(acc.map_or(t, |a| if *op == Op::And { a && t } else { a || t }));
            }
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
            (AggregateState::Fold { op, acc }, AggregateState::Fold { op: o2, acc: b })
                if op == o2 =>
            {
                if let Some(b) = b {
                    *acc = Some(acc.map_or(*b, |a| combine(*op, a, *b)));
                }
            }
            (AggregateState::Logical { op, acc }, AggregateState::Logical { op: o2, acc: b })
                if op == o2 =>
            {
                if let Some(b) = b {
                    *acc = Some(acc.map_or(*b, |a| if *op == Op::And { a && *b } else { a || *b }));
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

    /// The aggregate's value: NULL for folds and `AVG` when every value was
    /// NULL, and 0 for `COUNT`.
    pub fn finish(&self) -> AggregateValue {
        match *self {
            AggregateState::Fold { acc, .. } => AggregateValue::Float(acc),
            AggregateState::Logical { acc, .. } => AggregateValue::Bool(acc),
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
        for a in [
            Aggregate::SUM,
            Aggregate::Count,
            Aggregate::Avg,
            Aggregate::MIN,
            Aggregate::MAX,
        ] {
            assert_eq!(PartialAggregate::new(a).finish(), None);
        }
    }

    #[test]
    fn group_reached_only_by_nulls() {
        let nulls = [None, None];
        assert_eq!(fold(Aggregate::SUM, &nulls).finish(), Some(Float(None)));
        assert_eq!(fold(Aggregate::Count, &nulls).finish(), Some(Int(0)));
        assert_eq!(fold(Aggregate::Avg, &nulls).finish(), Some(Float(None)));
    }

    #[test]
    fn nulls_are_skipped() {
        let vs = [None, Some(2.0), None, Some(4.0)];
        assert_eq!(fold(Aggregate::SUM, &vs).finish(), Some(Float(Some(6.0))));
        assert_eq!(fold(Aggregate::Count, &vs).finish(), Some(Int(2)));
        assert_eq!(fold(Aggregate::Avg, &vs).finish(), Some(Float(Some(3.0))));
    }

    #[test]
    fn min_max_and_logical_folds() {
        let vs = [None, Some(2.0), Some(-1.0), None, Some(4.0)];
        assert_eq!(fold(Aggregate::MIN, &vs).finish(), Some(Float(Some(-1.0))));
        assert_eq!(fold(Aggregate::MAX, &vs).finish(), Some(Float(Some(4.0))));
        assert_eq!(fold(Aggregate::MIN, &[None]).finish(), Some(Float(None)));
        // NaN is above every number, as in Arrow's ordering; -0.0 is below 0.0.
        let Some(Float(Some(m))) = fold(Aggregate::MAX, &[Some(1.0), Some(f64::NAN)]).finish()
        else {
            panic!()
        };
        assert!(m.is_nan());
        assert_eq!(
            fold(Aggregate::MIN, &[Some(0.0), Some(-0.0)]).finish(),
            Some(Float(Some(-0.0)))
        );
        let tf = [Some(1.0), None, Some(0.0)];
        assert_eq!(
            fold(Aggregate::BOOL_OR, &tf).finish(),
            Some(AggregateValue::Bool(Some(true)))
        );
        assert_eq!(
            fold(Aggregate::BOOL_AND, &tf).finish(),
            Some(AggregateValue::Bool(Some(false)))
        );
        assert_eq!(
            fold(Aggregate::BOOL_OR, &[None]).finish(),
            Some(AggregateValue::Bool(None))
        );
    }

    #[test]
    fn nan_propagates_but_is_counted() {
        let vs = [Some(1.0), Some(f64::NAN)];
        let Some(Float(Some(s))) = fold(Aggregate::SUM, &vs).finish() else {
            panic!()
        };
        assert!(s.is_nan());
        assert_eq!(fold(Aggregate::Count, &vs).finish(), Some(Int(2)));
    }

    #[test]
    fn merge_matches_sequential_updates() {
        let parts: [&[Option<f64>]; 4] = [&[Some(1.0)], &[None], &[], &[Some(4.0), Some(3.0)]];
        for a in [
            Aggregate::SUM,
            Aggregate::Count,
            Aggregate::Avg,
            Aggregate::MIN,
            Aggregate::MAX,
        ] {
            let mut merged = PartialAggregate::new(a);
            for p in parts {
                merged.merge(&fold(a, p));
            }
            let all: Vec<Option<f64>> = parts.concat();
            assert_eq!(merged.finish(), fold(a, &all).finish(), "{a}");
        }
        // Merging only empty pieces leaves the group nonexistent.
        let mut empty = PartialAggregate::new(Aggregate::SUM);
        empty.merge(&PartialAggregate::new(Aggregate::SUM));
        assert_eq!(empty.finish(), None);
    }
}
