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
//! - **the aggregate's own state:** one partial result per distributive part
//!   of the aggregate (see [`Aggregate::parts`]). `SUM` has one part, and
//!   `AVG` has two, a sum and a count.
//!
//! Each piece is a monoid, and so is their product, so pieces combine in any
//! grouping and order. Keeping them apart gives every aggregate SQL's subtle
//! rule for free: a group reached only by rows whose values are NULL still
//! exists, and its `SUM` is NULL rather than missing.
//!
//! Sums here use plain `f64` addition. Float addition is not associative
//! (`(a + b) + c` can differ from `a + (b + c)` in the last bits), so the
//! order of [`update`](PartialAggregate::update) and
//! [`merge`](PartialAggregate::merge) calls decides a float result's last
//! bits. Callers that promise repeatable results must use a fixed order.

use crate::aggregate::Aggregate;
use crate::algebra::Value;

/// The most distributive parts an aggregate has.
const MAX_PARTS: usize = 2;

/// The state of one [`Aggregate`] over part of a group's values: one partial
/// result per part, `None` while the part has seen no value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AggregateState {
    aggregate: Aggregate,
    parts: [Option<Value>; MAX_PARTS],
}

impl AggregateState {
    /// The state of `aggregate` before any value.
    pub fn new(aggregate: Aggregate) -> Self {
        assert!(
            aggregate.parts().len() <= MAX_PARTS,
            "{aggregate} has too many parts"
        );
        AggregateState {
            aggregate,
            parts: [None; MAX_PARTS],
        }
    }

    /// The aggregate whose state this is.
    pub fn aggregate(&self) -> Aggregate {
        self.aggregate
    }

    /// Add one value; `None` is a NULL value, which every aggregate skips.
    ///
    /// # Panics
    ///
    /// If the aggregate does not act on the value's type, or the type
    /// differs from earlier values'.
    pub fn update(&mut self, value: Option<Value>) {
        let Some(v) = value else { return };
        for (part, acc) in self.aggregate.parts().iter().zip(&mut self.parts) {
            *acc = Some(part.update(*acc, v));
        }
    }

    /// Combine another state of the same aggregate into this one.
    ///
    /// # Panics
    ///
    /// If the two states belong to different aggregates.
    pub fn merge(&mut self, other: &AggregateState) {
        assert_eq!(
            self.aggregate, other.aggregate,
            "cannot merge states of different aggregates"
        );
        for ((part, acc), b) in self
            .aggregate
            .parts()
            .iter()
            .zip(&mut self.parts)
            .zip(other.parts)
        {
            *acc = part.merge(*acc, b);
        }
    }

    /// The aggregate's SQL result: NULL for most aggregates when every value
    /// was NULL, and 0 for `COUNT`.
    pub fn finish(&self) -> Option<Value> {
        self.aggregate
            .finish(&self.parts[..self.aggregate.parts().len()])
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
            state: AggregateState::new(aggregate),
        }
    }

    /// Fold in one joined row's value; `None` is a NULL value.
    ///
    /// The row marks the group as reached even when its value is NULL.
    pub fn update(&mut self, value: Option<Value>) {
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
    /// the aggregate's SQL result, which may be NULL (`Some(None)`).
    pub fn finish(&self) -> Option<Option<Value>> {
        self.reached.then(|| self.state.finish())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Value::{Bool, Float, Int};

    const ALL: [Aggregate; 7] = [
        Aggregate::SUM,
        Aggregate::COUNT,
        Aggregate::AVG,
        Aggregate::MIN,
        Aggregate::MAX,
        Aggregate::BOOL_OR,
        Aggregate::BOOL_AND,
    ];

    fn fold(aggregate: Aggregate, values: &[Option<Value>]) -> PartialAggregate {
        let mut p = PartialAggregate::new(aggregate);
        for v in values {
            p.update(*v);
        }
        p
    }

    fn floats(xs: &[Option<f64>]) -> Vec<Option<Value>> {
        xs.iter().map(|x| x.map(Float)).collect()
    }

    #[test]
    fn unreached_group_does_not_exist() {
        for a in ALL {
            assert_eq!(PartialAggregate::new(a).finish(), None);
        }
    }

    #[test]
    fn group_reached_only_by_nulls() {
        let nulls = [None, None];
        assert_eq!(fold(Aggregate::SUM, &nulls).finish(), Some(None));
        assert_eq!(fold(Aggregate::COUNT, &nulls).finish(), Some(Some(Int(0))));
        assert_eq!(fold(Aggregate::AVG, &nulls).finish(), Some(None));
        assert_eq!(fold(Aggregate::BOOL_OR, &nulls).finish(), Some(None));
    }

    #[test]
    fn nulls_are_skipped() {
        let vs = floats(&[None, Some(2.0), None, Some(4.0)]);
        assert_eq!(fold(Aggregate::SUM, &vs).finish(), Some(Some(Float(6.0))));
        assert_eq!(fold(Aggregate::COUNT, &vs).finish(), Some(Some(Int(2))));
        assert_eq!(fold(Aggregate::AVG, &vs).finish(), Some(Some(Float(3.0))));
    }

    #[test]
    fn integers_stay_integers_until_avg() {
        let vs = [Some(Int(3)), None, Some(Int(4))];
        assert_eq!(fold(Aggregate::SUM, &vs).finish(), Some(Some(Int(7))));
        assert_eq!(fold(Aggregate::MAX, &vs).finish(), Some(Some(Int(4))));
        assert_eq!(fold(Aggregate::AVG, &vs).finish(), Some(Some(Float(3.5))));
    }

    #[test]
    fn min_max_and_logical_folds() {
        let vs = floats(&[None, Some(2.0), Some(-1.0), None, Some(4.0)]);
        assert_eq!(fold(Aggregate::MIN, &vs).finish(), Some(Some(Float(-1.0))));
        assert_eq!(fold(Aggregate::MAX, &vs).finish(), Some(Some(Float(4.0))));
        // NaN is above every number, as in Arrow's ordering; -0.0 is below 0.0.
        let Some(Some(Float(m))) =
            fold(Aggregate::MAX, &floats(&[Some(1.0), Some(f64::NAN)])).finish()
        else {
            panic!()
        };
        assert!(m.is_nan());
        let zeros = floats(&[Some(0.0), Some(-0.0)]);
        let Some(Some(Float(z))) = fold(Aggregate::MIN, &zeros).finish() else {
            panic!()
        };
        assert!(z.is_sign_negative());
        let tf = [Some(Bool(true)), None, Some(Bool(false))];
        assert_eq!(
            fold(Aggregate::BOOL_OR, &tf).finish(),
            Some(Some(Bool(true)))
        );
        assert_eq!(
            fold(Aggregate::BOOL_AND, &tf).finish(),
            Some(Some(Bool(false)))
        );
    }

    #[test]
    fn nan_propagates_but_is_counted() {
        let vs = floats(&[Some(1.0), Some(f64::NAN)]);
        let Some(Some(Float(s))) = fold(Aggregate::SUM, &vs).finish() else {
            panic!()
        };
        assert!(s.is_nan());
        assert_eq!(fold(Aggregate::COUNT, &vs).finish(), Some(Some(Int(2))));
    }

    #[test]
    fn merge_matches_sequential_updates() {
        let parts: [Vec<Option<Value>>; 4] = [
            vec![Some(Int(1))],
            vec![None],
            vec![],
            vec![Some(Int(4)), Some(Int(3))],
        ];
        for a in [
            Aggregate::SUM,
            Aggregate::COUNT,
            Aggregate::AVG,
            Aggregate::MIN,
            Aggregate::MAX,
        ] {
            let mut merged = PartialAggregate::new(a);
            for p in &parts {
                merged.merge(&fold(a, p));
            }
            let all: Vec<Option<Value>> = parts.concat();
            assert_eq!(merged.finish(), fold(a, &all).finish(), "{a}");
        }
        // Merging only empty pieces leaves the group nonexistent.
        let mut empty = PartialAggregate::new(Aggregate::SUM);
        empty.merge(&PartialAggregate::new(Aggregate::SUM));
        assert_eq!(empty.finish(), None);
    }

    #[test]
    #[should_panic(expected = "cannot combine")]
    fn mixed_types_are_rejected() {
        fold(Aggregate::SUM, &[Some(Int(1)), Some(Float(1.0))]);
    }
}
