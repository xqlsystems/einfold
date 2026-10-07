// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Aggregates: how a fold combines the values of a group's rows.
//!
//! Aggregates fall into classes by the state a partial result must carry,
//! after Gray et al.'s data-cube paper ("Data Cube: A Relational Aggregation
//! Operator", 1997). The class decides whether a group can be aggregated in
//! pieces and the pieces combined, which every parallel or pre-aggregating
//! rewrite needs. DataFusion's aggregates sort into them as follows:
//!
//! | Class | State | DataFusion's aggregates | Here |
//! |---|---|---|---|
//! | distributive | one value, combined with an [`Op`] | `sum`, `count`, `min`, `max`, `bool_and`, `bool_or`, `bit_and`, `bit_or`, `bit_xor` | [`Distributive`] |
//! | algebraic | a fixed number of distributive parts, then a final function | `avg`, `var*`, `stddev*`, `covar*`, `corr`, `regr_*` | [`Algebraic`] |
//! | holistic | unbounded: all the values, or all the distinct ones | `median`, `percentile_cont`, `count(DISTINCT)`, `array_agg` | not yet |
//! | approximate | a bounded summary that merges | `approx_distinct`, `approx_median`, `approx_percentile_cont` | not yet |
//! | order-dependent | depends on the order of the rows | `first_value`, `last_value`, `nth_value`, `string_agg`, ordered `array_agg` | out of scope |
//!
//! - **Distributive** aggregates *lift* each value (`COUNT` lifts it to 1) and
//!   fold the lifted values with one commutative monoid ([`Op`]). Partial
//!   results combine with the same operation. The bitwise aggregates belong
//!   here, once [`Op`] gains the bitwise operations.
//! - **Algebraic** aggregates are a fixed tuple of distributive ones followed
//!   by a final function: `AVG` is `SUM / COUNT`. In algebra, the tuple's
//!   state is the *direct product* of the parts' monoids, itself a monoid.
//!   Variance and the regression aggregates are also sums of powers and
//!   products of the inputs, though engines often keep a numerically stabler
//!   state with its own merge (Chan, Golub and LeVeque, 1979).
//! - **Holistic** aggregates are monoids too (on bags or sets of values), but
//!   their state grows with the input, so aggregating early gains nothing.
//! - **Approximate** aggregates keep a *mergeable summary* (Agarwal et al.,
//!   "Mergeable Summaries", 2012), such as a HyperLogLog sketch or a t-digest:
//!   a bounded state with a merge, approximately a monoid.
//! - **Order-dependent** aggregates fold with an operation that is not
//!   commutative, such as concatenation, so their result depends on the row
//!   order that joins and parallel plans don't keep.
//!
//! The same three-step shape, lift, fold with a monoid, then finish, is the
//! aggregate interface of most engines: Algebird's `Aggregator` (`prepare`,
//! `monoid`, `present`), Beam's `CombineFn` (`addInput`, `mergeAccumulators`,
//! `extractOutput`), and DataFusion's `Accumulator` (`update_batch`,
//! `merge_batch`, `evaluate`).
//!
//! All of SQL's aggregates here skip NULL inputs. Over no values, a monoid's
//! fold would be its identity, but SQL returns NULL instead, except for
//! `COUNT`, which returns 0. A group with *no* rows never exists in a
//! `GROUP BY` result, so that case concerns only groups whose every value was
//! NULL.

use std::fmt;

use crate::algebra::{Op, Value, ValueType};

/// What a distributive aggregate folds for each non-NULL value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Lift {
    /// The value itself.
    Value,
    /// The integer 1, whatever the value: counting.
    One,
}

impl Lift {
    /// The lifted value.
    pub fn apply(self, v: Value) -> Value {
        match self {
            Lift::Value => v,
            Lift::One => Value::Int(1),
        }
    }

    /// The type of every lifted value, if the lift fixes it.
    fn fixed_type(self) -> Option<ValueType> {
        match self {
            Lift::Value => None,
            Lift::One => Some(ValueType::Int),
        }
    }
}

/// A distributive aggregate: lift each non-NULL value, and fold the lifted
/// values with one operation.
///
/// Over no values the result is NULL, as SQL has it, except when the lift
/// fixes the type of what is folded: then it is the operation's identity, so
/// `COUNT` of nothing is 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Distributive {
    lift: Lift,
    op: Op,
}

impl Distributive {
    /// SQL's `SUM`.
    pub const SUM: Distributive = Distributive::new(Lift::Value, Op::Add);
    /// SQL's `COUNT`. `COUNT(*)` is `COUNT` of a value that is never NULL.
    pub const COUNT: Distributive = Distributive::new(Lift::One, Op::Add);
    /// SQL's `MIN`.
    pub const MIN: Distributive = Distributive::new(Lift::Value, Op::Min);
    /// SQL's `MAX`.
    pub const MAX: Distributive = Distributive::new(Lift::Value, Op::Max);
    /// SQL's `BOOL_OR`.
    pub const BOOL_OR: Distributive = Distributive::new(Lift::Value, Op::Or);
    /// SQL's `BOOL_AND`.
    pub const BOOL_AND: Distributive = Distributive::new(Lift::Value, Op::And);

    /// The aggregate that lifts each value with `lift` and folds with `op`.
    pub const fn new(lift: Lift, op: Op) -> Self {
        Distributive { lift, op }
    }

    /// How each value is lifted.
    pub fn lift(self) -> Lift {
        self.lift
    }

    /// The operation that folds the lifted values, and combines partial
    /// results.
    pub fn op(self) -> Op {
        self.op
    }

    /// Fold one non-NULL value into the result so far (`None` before any).
    ///
    /// # Panics
    ///
    /// If the operation does not act on the value's type, or the type
    /// differs from earlier values'.
    pub fn update(self, acc: Option<Value>, v: Value) -> Value {
        self.merge(acc, Some(self.lift.apply(v)))
            .expect("a value was added")
    }

    /// Combine two partial results, either of which may have seen no values.
    ///
    /// # Panics
    ///
    /// As for [`update`](Distributive::update).
    pub fn merge(self, a: Option<Value>, b: Option<Value>) -> Option<Value> {
        match (a, b) {
            (Some(a), Some(b)) => Some(
                self.op
                    .combine(a, b)
                    .unwrap_or_else(|| panic!("{} cannot combine {a:?} and {b:?}", self.op)),
            ),
            (a, b) => a.or(b),
        }
    }

    /// The SQL result from the partial result: NULL, or the identity, if no
    /// value was folded.
    pub fn finish(self, acc: Option<Value>) -> Option<Value> {
        acc.or_else(|| self.op.identity(self.lift.fixed_type()?))
    }

    /// Whether the result is the same whatever order float inputs are folded
    /// in. Float `+` and `*` round, so their results can change in the last
    /// bits when a rewrite changes the order.
    pub fn is_exact(self) -> bool {
        self.lift == Lift::One || !matches!(self.op, Op::Add | Op::Mul)
    }
}

/// The final function of an [`Algebraic`] aggregate, from its parts'
/// results.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Finish {
    /// The first part divided by the second, as a float; NULL if either is
    /// NULL or the second is 0.
    Ratio,
}

impl Finish {
    /// Apply the function to the parts' results.
    pub fn apply(self, parts: &[Option<Value>]) -> Option<Value> {
        let float = |v: Value| match v {
            Value::Int(i) => Some(i as f64),
            Value::Float(f) => Some(f),
            Value::Bool(_) => None,
        };
        match self {
            Finish::Ratio => {
                let (n, d) = (float(parts[0]?)?, float(parts[1]?)?);
                (d != 0.0).then(|| Value::Float(n / d))
            }
        }
    }
}

/// An algebraic aggregate: a fixed tuple of distributive aggregates, then a
/// final function of their results.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Algebraic {
    parts: &'static [Distributive],
    finish: Finish,
}

impl Algebraic {
    /// SQL's `AVG`: `SUM / COUNT`.
    pub const AVG: Algebraic = Algebraic {
        parts: &[Distributive::SUM, Distributive::COUNT],
        finish: Finish::Ratio,
    };

    /// The distributive aggregates it is computed from.
    pub fn parts(self) -> &'static [Distributive] {
        self.parts
    }

    /// The final function.
    pub fn finish(self) -> Finish {
        self.finish
    }
}

/// An aggregate a fold can compute, with its SQL semantics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Aggregate {
    /// One value, folded with one operation.
    Distributive(Distributive),
    /// Several distributive parts, then a final function.
    Algebraic(Algebraic),
}

impl Aggregate {
    /// SQL's `SUM`.
    pub const SUM: Aggregate = Aggregate::Distributive(Distributive::SUM);
    /// SQL's `COUNT`.
    pub const COUNT: Aggregate = Aggregate::Distributive(Distributive::COUNT);
    /// SQL's `MIN`.
    pub const MIN: Aggregate = Aggregate::Distributive(Distributive::MIN);
    /// SQL's `MAX`.
    pub const MAX: Aggregate = Aggregate::Distributive(Distributive::MAX);
    /// SQL's `BOOL_OR`.
    pub const BOOL_OR: Aggregate = Aggregate::Distributive(Distributive::BOOL_OR);
    /// SQL's `BOOL_AND`.
    pub const BOOL_AND: Aggregate = Aggregate::Distributive(Distributive::BOOL_AND);
    /// SQL's `AVG`.
    pub const AVG: Aggregate = Aggregate::Algebraic(Algebraic::AVG);

    const NAMED: [(Aggregate, &'static str); 7] = [
        (Aggregate::SUM, "SUM"),
        (Aggregate::COUNT, "COUNT"),
        (Aggregate::MIN, "MIN"),
        (Aggregate::MAX, "MAX"),
        (Aggregate::BOOL_OR, "BOOL_OR"),
        (Aggregate::BOOL_AND, "BOOL_AND"),
        (Aggregate::AVG, "AVG"),
    ];

    /// The SQL name of the aggregate, if it is one of SQL's.
    pub fn sql_name(self) -> Option<&'static str> {
        Self::NAMED
            .iter()
            .find(|(a, _)| *a == self)
            .map(|(_, n)| *n)
    }

    /// The distributive aggregates its state is made of: itself, for a
    /// distributive aggregate.
    pub fn parts(&self) -> &[Distributive] {
        match self {
            Aggregate::Distributive(d) => std::slice::from_ref(d),
            Aggregate::Algebraic(a) => a.parts,
        }
    }

    /// The SQL result from its parts' partial results, one per
    /// [`part`](Aggregate::parts).
    pub fn finish(self, parts: &[Option<Value>]) -> Option<Value> {
        match self {
            Aggregate::Distributive(d) => d.finish(parts[0]),
            Aggregate::Algebraic(a) => {
                let results: Vec<Option<Value>> = a
                    .parts
                    .iter()
                    .zip(parts)
                    .map(|(d, &p)| d.finish(p))
                    .collect();
                a.finish.apply(&results)
            }
        }
    }

    /// Whether the result is the same whatever order float inputs are
    /// combined in. Rewrites may change that order, so an inexact result may
    /// change in its last bits; an exact one may never change. `MIN`, `MAX`,
    /// the logical aggregates and `COUNT` are exact; `SUM` and `AVG` of
    /// floats are not.
    pub fn is_exact(self) -> bool {
        self.parts().iter().all(|d| d.is_exact())
    }
}

impl fmt::Display for Aggregate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(name) = self.sql_name() {
            return f.write_str(name);
        }
        match self {
            Aggregate::Distributive(d) if d.lift == Lift::One => write!(f, "FOLD({} of 1)", d.op),
            Aggregate::Distributive(d) => write!(f, "FOLD({})", d.op),
            Aggregate::Algebraic(a) => write!(f, "{:?}", a.finish),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_results_follow_sql() {
        assert_eq!(Aggregate::SUM.finish(&[None]), None);
        assert_eq!(Aggregate::MIN.finish(&[None]), None);
        assert_eq!(Aggregate::COUNT.finish(&[None]), Some(Value::Int(0)));
        assert_eq!(Aggregate::AVG.finish(&[None, None]), None);
    }

    #[test]
    fn avg_is_the_ratio_of_its_parts() {
        let sum = Some(Value::Int(7));
        let count = Some(Value::Int(2));
        assert_eq!(
            Aggregate::AVG.finish(&[sum, count]),
            Some(Value::Float(3.5))
        );
    }

    #[test]
    fn names_and_exactness() {
        assert_eq!(Aggregate::AVG.to_string(), "AVG");
        let product = Aggregate::Distributive(Distributive::new(Lift::Value, Op::Mul));
        assert_eq!(product.sql_name(), None);
        assert_eq!(product.to_string(), "FOLD(*)");
        assert!(Aggregate::COUNT.is_exact() && Aggregate::MAX.is_exact());
        assert!(!Aggregate::SUM.is_exact() && !Aggregate::AVG.is_exact());
    }
}
