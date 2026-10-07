// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The algebra of folds: operations, and the laws that relate them.
//!
//! An aggregate such as `SUM` combines values with one binary operation (`+`).
//! A product of factors combines them with another (`*`). Which rewrites are
//! correct depends only on the *laws* these operations obey, not on which
//! operations they are. This module states those laws once, so that every
//! aggregate and every semiring is described by its operations rather than by
//! a list of special cases.
//!
//! - An [`Op`] is a binary operation that is associative and commutative and
//!   has an identity: in algebra, a *commutative monoid*. That is exactly
//!   what lets an engine combine partial results in any grouping and order.
//! - A [`Semiring`] is a pair of operations, "add" (⊕) and "multiply" (⊗),
//!   where ⊗ *distributes* over ⊕: `a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)`.
//!   Distributivity is what makes it correct to aggregate one input before
//!   joining it with another (`Σ_j a·b_j = a·Σ_j b_j`), and to reorder joins.
//!
//! These are the standard structures for the problem. Abo Khamis, Ngo and
//! Rudra's *FAQ* ("Functional Aggregate Queries", PODS 2016) solves
//! aggregates over joins in any commutative semiring, and its algorithm,
//! InsideOut, is the "aggregate early, reorder joins" rewrite above. Green,
//! Karvounarakis and Tannen's *provenance semirings* (PODS 2007) show that
//! one semiring-generic evaluation covers bag semantics, probabilities and
//! more. Lin's "Monoidify!" (2013) makes the case that an aggregate's state
//! must be a monoid for parallel, partial aggregation to be correct.

use std::fmt;

/// A binary operation that is associative and commutative and has an
/// identity, so that values can be combined in any grouping and any order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Op {
    /// `+` over numbers. Identity 0.
    Add,
    /// `*` over numbers. Identity 1.
    Mul,
    /// The smaller of two numbers. Identity +∞, or the largest integer.
    Min,
    /// The larger of two numbers. Identity −∞, or the smallest integer.
    Max,
    /// Logical `AND`. Identity `TRUE`.
    And,
    /// Logical `OR`. Identity `FALSE`.
    Or,
}

/// A non-NULL value that operations combine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Value {
    /// A 64-bit integer.
    Int(i64),
    /// A 64-bit float.
    Float(f64),
    /// A truth value.
    Bool(bool),
}

/// The type of a [`Value`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueType {
    /// [`Value::Int`].
    Int,
    /// [`Value::Float`].
    Float,
    /// [`Value::Bool`].
    Bool,
}

impl Value {
    /// The value's type.
    pub fn value_type(self) -> ValueType {
        match self {
            Value::Int(_) => ValueType::Int,
            Value::Float(_) => ValueType::Float,
            Value::Bool(_) => ValueType::Bool,
        }
    }
}

/// Whether one operation distributes over another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Distributivity {
    /// For all values.
    Always,
    /// Only when every value is non-negative. `*` distributes over `MAX`
    /// for non-negative numbers (`a·max(b, c) = max(a·b, a·c)` if `a ≥ 0`),
    /// but a negative `a` flips the order. Using it requires proving the
    /// values non-negative.
    IfNonNegative,
    /// Not in general.
    Never,
}

impl Op {
    /// The operation's identity among values of type `ty`: combining it
    /// with any value leaves the value unchanged. `None` if the operation
    /// does not act on that type.
    pub fn identity(self, ty: ValueType) -> Option<Value> {
        use ValueType as T;
        Some(match (self, ty) {
            (Op::Add, T::Int) => Value::Int(0),
            (Op::Add, T::Float) => Value::Float(0.0),
            (Op::Mul, T::Int) => Value::Int(1),
            (Op::Mul, T::Float) => Value::Float(1.0),
            (Op::Min, T::Int) => Value::Int(i64::MAX),
            (Op::Min, T::Float) => Value::Float(f64::INFINITY),
            (Op::Max, T::Int) => Value::Int(i64::MIN),
            (Op::Max, T::Float) => Value::Float(f64::NEG_INFINITY),
            (Op::And, T::Bool) => Value::Bool(true),
            (Op::Or, T::Bool) => Value::Bool(false),
            _ => return None,
        })
    }

    /// Combine two values of the same type. `None` if their types differ, or
    /// the operation does not act on their type.
    ///
    /// Integer `+` and `*` wrap on overflow, as DataFusion's do. `min` and
    /// `max` order floats as Arrow and DataFusion do: by IEEE 754's total
    /// order, where NaN is above every number and `-0.0` is below `0.0`.
    pub fn combine(self, a: Value, b: Value) -> Option<Value> {
        use Value::{Bool, Float, Int};
        Some(match (self, a, b) {
            (Op::Add, Int(a), Int(b)) => Int(a.wrapping_add(b)),
            (Op::Add, Float(a), Float(b)) => Float(a + b),
            (Op::Mul, Int(a), Int(b)) => Int(a.wrapping_mul(b)),
            (Op::Mul, Float(a), Float(b)) => Float(a * b),
            (Op::Min, Int(a), Int(b)) => Int(a.min(b)),
            (Op::Min, Float(a), Float(b)) => Float(if b.total_cmp(&a).is_lt() { b } else { a }),
            (Op::Max, Int(a), Int(b)) => Int(a.max(b)),
            (Op::Max, Float(a), Float(b)) => Float(if b.total_cmp(&a).is_gt() { b } else { a }),
            (Op::And, Bool(a), Bool(b)) => Bool(a && b),
            (Op::Or, Bool(a), Bool(b)) => Bool(a || b),
            _ => return None,
        })
    }

    /// Whether `self` distributes over `over`:
    /// `a self (b over c) = (a self b) over (a self c)`.
    pub fn distributes_over(self, over: Op) -> Distributivity {
        use Distributivity::*;
        use Op::*;
        match (self, over) {
            // Arithmetic: a·(b + c) = a·b + a·c.
            (Mul, Add) => Always,
            // Tropical: a + min(b, c) = min(a + b, a + c), and likewise max.
            (Add, Min) | (Add, Max) => Always,
            // a·min(b, c) = min(a·b, a·c) only when a ≥ 0; likewise max.
            (Mul, Min) | (Mul, Max) => IfNonNegative,
            // Min and max distribute over each other, as do AND and OR.
            (Min, Max) | (Max, Min) | (And, Or) | (Or, And) => Always,
            _ => Never,
        }
    }
}

impl fmt::Display for Op {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Op::Add => "+",
            Op::Mul => "*",
            Op::Min => "min",
            Op::Max => "max",
            Op::And => "and",
            Op::Or => "or",
        })
    }
}

/// A semiring: an "add" (⊕) and a "multiply" (⊗), where ⊗ distributes over
/// ⊕, possibly only under a condition on the values.
///
/// In every semiring here, ⊕'s identity, the semiring's "zero", also
/// *annihilates*: `0 ⊗ a = 0`. That is what lets a dense kernel pad absent
/// entries with the zero. In the conditional semirings it holds only for
/// positive `a`: `max` of `*` has zero −∞, and `−∞ · 0` is NaN.
///
/// Semirings are built only through [`Semiring::new`], which checks the law,
/// so holding one is evidence that it holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Semiring {
    add: Op,
    mul: Op,
    law: Distributivity,
}

impl Semiring {
    /// `SUM` of `*`: ordinary arithmetic, as in linear algebra.
    pub const SUM_PRODUCT: Semiring = Semiring {
        add: Op::Add,
        mul: Op::Mul,
        law: Distributivity::Always,
    };
    /// `MIN` of `+`: shortest paths.
    pub const MIN_PLUS: Semiring = Semiring {
        add: Op::Min,
        mul: Op::Add,
        law: Distributivity::Always,
    };
    /// `MAX` of `+`: longest or best paths.
    pub const MAX_PLUS: Semiring = Semiring {
        add: Op::Max,
        mul: Op::Add,
        law: Distributivity::Always,
    };
    /// `OR` of `AND`: reachability and existence.
    pub const BOOLEAN: Semiring = Semiring {
        add: Op::Or,
        mul: Op::And,
        law: Distributivity::Always,
    };

    /// The semiring with "add" `add` and "multiply" `mul`, if `mul`
    /// distributes over `add`, perhaps under a condition (see
    /// [`Semiring::requires_non_negative`]).
    pub fn new(add: Op, mul: Op) -> Option<Semiring> {
        match mul.distributes_over(add) {
            Distributivity::Never => None,
            law => Some(Semiring { add, mul, law }),
        }
    }

    /// The "add", ⊕: the aggregate.
    pub fn add(self) -> Op {
        self.add
    }

    /// The "multiply", ⊗: how factors combine.
    pub fn mul(self) -> Op {
        self.mul
    }

    /// Whether the law holds only for non-negative values, which must then be
    /// proven before any rewrite relies on it.
    pub fn requires_non_negative(self) -> bool {
        self.law == Distributivity::IfNonNegative
    }
}

impl fmt::Display for Semiring {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({}, {})", self.add, self.mul)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn named_semirings_are_what_new_builds() {
        assert_eq!(Semiring::new(Op::Add, Op::Mul), Some(Semiring::SUM_PRODUCT));
        assert_eq!(Semiring::new(Op::Min, Op::Add), Some(Semiring::MIN_PLUS));
        assert_eq!(Semiring::new(Op::Max, Op::Add), Some(Semiring::MAX_PLUS));
        assert_eq!(Semiring::new(Op::Or, Op::And), Some(Semiring::BOOLEAN));
    }

    #[test]
    fn non_distributive_pairs_are_rejected() {
        assert_eq!(Semiring::new(Op::Mul, Op::Add), None); // a + (b·c) ≠ (a+b)·(a+c)
        assert_eq!(Semiring::new(Op::Add, Op::Add), None);
    }

    #[test]
    fn conditional_laws_are_recorded() {
        let max_times = Semiring::new(Op::Max, Op::Mul).unwrap();
        assert!(max_times.requires_non_negative());
        assert!(!Semiring::SUM_PRODUCT.requires_non_negative());
    }

    fn floats(xs: &[f64]) -> Vec<Value> {
        xs.iter().map(|&x| Value::Float(x)).collect()
    }

    /// Sample values for an operation, with the condition on its law.
    fn samples(op: Op, law: Distributivity) -> Vec<Value> {
        if op == Op::And || op == Op::Or {
            return vec![Value::Bool(false), Value::Bool(true)];
        }
        let all = [-3.0, -0.5, 0.0, 0.5, 2.0, 7.0];
        let xs: Vec<f64> = match law {
            Distributivity::IfNonNegative => all.into_iter().filter(|&x| x >= 0.0).collect(),
            _ => all.to_vec(),
        };
        floats(&xs)
    }

    const OPS: [Op; 6] = [Op::Add, Op::Mul, Op::Min, Op::Max, Op::And, Op::Or];

    /// Check each claimed law on sample values, including the condition.
    #[test]
    fn claimed_laws_hold_on_samples() {
        for mul in OPS {
            for add in OPS {
                let law = mul.distributes_over(add);
                if law == Distributivity::Never {
                    continue;
                }
                let vs = samples(mul, law);
                for &a in &vs {
                    for &b in &vs {
                        for &c in &vs {
                            let lhs = mul.combine(a, add.combine(b, c).unwrap());
                            let rhs =
                                add.combine(mul.combine(a, b).unwrap(), mul.combine(a, c).unwrap());
                            assert_eq!(lhs, rhs, "{mul} over {add} at {a:?}, {b:?}, {c:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn identities_are_identities() {
        for op in OPS {
            for v in samples(op, Distributivity::Always) {
                let e = op.identity(v.value_type()).unwrap();
                assert_eq!(op.combine(e, v), Some(v), "{op} at {v:?}");
            }
            for v in [Value::Int(-4), Value::Int(9)] {
                if let Some(e) = op.identity(ValueType::Int) {
                    assert_eq!(op.combine(e, v), Some(v), "{op} at {v:?}");
                }
            }
        }
    }

    #[test]
    fn zero_annihilates() {
        for mul in OPS {
            for add in OPS {
                let Some(s) = Semiring::new(add, mul) else {
                    continue;
                };
                for a in samples(mul, Distributivity::Always) {
                    if s.requires_non_negative() && !matches!(a, Value::Float(x) if x > 0.0) {
                        continue;
                    }
                    let zero = add.identity(a.value_type()).unwrap();
                    assert_eq!(mul.combine(zero, a), Some(zero), "{s} at {a:?}");
                }
            }
        }
    }

    #[test]
    fn mismatched_types_do_not_combine() {
        assert_eq!(Op::Add.combine(Value::Int(1), Value::Float(1.0)), None);
        assert_eq!(Op::And.combine(Value::Float(1.0), Value::Float(1.0)), None);
        assert_eq!(Op::Or.identity(ValueType::Int), None);
    }
}
