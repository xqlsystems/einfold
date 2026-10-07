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
    /// The smaller of two numbers. Identity +∞.
    Min,
    /// The larger of two numbers. Identity −∞.
    Max,
    /// Logical `AND`. Identity `TRUE`.
    And,
    /// Logical `OR`. Identity `FALSE`.
    Or,
}

/// The identity of an [`Op`]: combining it with any value leaves the value
/// unchanged.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Identity {
    /// A number: 0 for `+`, 1 for `*`, ±∞ for `MIN` and `MAX`.
    Number(f64),
    /// A truth value: `TRUE` for `AND`, `FALSE` for `OR`.
    Bool(bool),
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
    /// The operation's identity.
    pub fn identity(self) -> Identity {
        match self {
            Op::Add => Identity::Number(0.0),
            Op::Mul => Identity::Number(1.0),
            Op::Min => Identity::Number(f64::INFINITY),
            Op::Max => Identity::Number(f64::NEG_INFINITY),
            Op::And => Identity::Bool(true),
            Op::Or => Identity::Bool(false),
        }
    }

    /// Whether the operation acts on truth values rather than numbers.
    pub fn is_logical(self) -> bool {
        matches!(self, Op::And | Op::Or)
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

    /// Check each claimed law on sample values, including the condition.
    #[test]
    fn claimed_laws_hold_on_samples() {
        let num = |op: Op, a: f64, b: f64| match op {
            Op::Add => a + b,
            Op::Mul => a * b,
            Op::Min => a.min(b),
            Op::Max => a.max(b),
            _ => unreachable!(),
        };
        let numeric = [Op::Add, Op::Mul, Op::Min, Op::Max];
        let samples = [-3.0, -0.5, 0.0, 0.5, 2.0, 7.0];
        for mul in numeric {
            for add in numeric {
                let law = mul.distributes_over(add);
                if law == Distributivity::Never {
                    continue;
                }
                for a in samples {
                    for b in samples {
                        for c in samples {
                            if law == Distributivity::IfNonNegative && a < 0.0 {
                                continue;
                            }
                            let lhs = num(mul, a, num(add, b, c));
                            let rhs = num(add, num(mul, a, b), num(mul, a, c));
                            assert_eq!(lhs, rhs, "{mul} over {add} at {a}, {b}, {c}");
                        }
                    }
                }
            }
        }
    }
}
