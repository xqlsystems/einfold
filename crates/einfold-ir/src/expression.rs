// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Row expressions: a fold's row value when it isn't a product of factors.
//!
//! A Gaussian kernel's weight reads two operands at once: `power(t.x − s.x,
//! 2) + power(t.y − s.y, 2)` is a squared distance between a target point
//! and a source point, and no product of one factor per operand. An
//! [`Expression`] is such a value: a tree over the joined row's columns, in a
//! small, closed set of `DOUBLE` operators. Each has SQL's NULL rule (NULL in
//! any argument gives NULL), and is defined as one IEEE operation or one call
//! of Rust's `f64` function of that name, which is also what DataFusion
//! computes, so an evaluator that follows [`Expression::evaluate`] gives the
//! host's bits.
//!
//! Which inputs a host raises an error on is the host's business: a detector
//! takes an operator only where it can't, such as `power` with a positive
//! constant exponent.

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::fold::ColumnIndex;

/// A `DOUBLE` constant, compared by its bits so that equality, hashing and
/// ordering are total (a NaN equals itself).
#[derive(Clone, Copy, Debug)]
pub struct Constant(pub f64);

impl PartialEq for Constant {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}

impl Eq for Constant {}

impl Hash for Constant {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state)
    }
}

impl PartialOrd for Constant {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Constant {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// An operator of one argument.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Unary {
    /// SQL `-x`.
    Negate,
    /// SQL `abs(x)`: [`f64::abs`].
    Abs,
    /// SQL `exp(x)`: [`f64::exp`].
    Exp,
    /// SQL `ln(x)`: [`f64::ln`], NaN below 0 and `-inf` at 0.
    Ln,
}

impl Unary {
    /// The operator applied to `x`.
    #[must_use]
    pub fn apply(self, x: f64) -> f64 {
        match self {
            Unary::Negate => -x,
            Unary::Abs => x.abs(),
            Unary::Exp => x.exp(),
            Unary::Ln => x.ln(),
        }
    }
}

/// An operator of two arguments.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Binary {
    /// SQL `a + b`.
    Add,
    /// SQL `a - b`.
    Subtract,
    /// SQL `a * b`.
    Multiply,
    /// SQL `a / b`: IEEE division, ±inf or NaN when `b` is 0.
    Divide,
    /// SQL `power(a, b)`: [`f64::powf`].
    Power,
}

impl Binary {
    /// The operator applied to `a` and `b`.
    #[must_use]
    pub fn apply(self, a: f64, b: f64) -> f64 {
        match self {
            Binary::Add => a + b,
            Binary::Subtract => a - b,
            Binary::Multiply => a * b,
            Binary::Divide => a / b,
            Binary::Power => a.powf(b),
        }
    }
}

/// A `DOUBLE` expression of a joined row: what a fold's
/// [`RowValue::Expression`](crate::RowValue::Expression) contributes for each
/// row (see the [module](self)'s documentation).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Expression {
    /// Column `column` of operand `operand`'s row.
    Column {
        /// The operand, by its place in the fold's operands.
        operand: usize,
        /// The column, by its place in the operand's input.
        column: ColumnIndex,
    },
    /// A constant.
    Literal(Constant),
    /// An operator of one argument.
    Unary(Unary, Box<Expression>),
    /// An operator of two arguments.
    Binary(Binary, Box<Expression>, Box<Expression>),
}

impl Expression {
    /// Column `column` of operand `operand`.
    #[must_use]
    pub fn column(operand: usize, column: usize) -> Self {
        Expression::Column {
            operand,
            column: ColumnIndex(column),
        }
    }

    /// The constant `x`.
    #[must_use]
    pub fn literal(x: f64) -> Self {
        Expression::Literal(Constant(x))
    }

    /// `op(self)`.
    #[must_use]
    pub fn unary(self, op: Unary) -> Self {
        Expression::Unary(op, Box::new(self))
    }

    /// `self <op> other`.
    #[must_use]
    pub fn binary(self, op: Binary, other: Expression) -> Self {
        Expression::Binary(op, Box::new(self), Box::new(other))
    }

    /// The expression's value, with `column(operand, column)` the value of
    /// each column it reads, `None` for NULL: NULL if any column it reads
    /// is.
    pub fn evaluate(&self, column: &impl Fn(usize, ColumnIndex) -> Option<f64>) -> Option<f64> {
        match self {
            Expression::Column { operand, column: c } => column(*operand, *c),
            Expression::Literal(Constant(x)) => Some(*x),
            Expression::Unary(op, x) => Some(op.apply(x.evaluate(column)?)),
            Expression::Binary(op, a, b) => {
                let (a, b) = (a.evaluate(column), b.evaluate(column));
                Some(op.apply(a?, b?))
            }
        }
    }

    /// Every column the expression reads, as `(operand, column)`, in the
    /// order it reads them, repeats included.
    #[must_use]
    pub fn columns(&self) -> Vec<(usize, ColumnIndex)> {
        let mut out = Vec::new();
        self.visit(&mut |e| {
            if let Expression::Column { operand, column } = e {
                out.push((*operand, *column));
            }
        });
        out
    }

    fn visit(&self, f: &mut impl FnMut(&Expression)) {
        f(self);
        match self {
            Expression::Column { .. } | Expression::Literal(_) => {}
            Expression::Unary(_, x) => x.visit(f),
            Expression::Binary(_, a, b) => {
                a.visit(f);
                b.visit(f);
            }
        }
    }
}

/// SQL-like, with each column as `#<operand>.<column>`: `power(#0.1 - #1.1,
/// 2)`.
impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Column { operand, column } => write!(f, "#{operand}.{}", column.0),
            Expression::Literal(Constant(x)) => write!(f, "{x}"),
            Expression::Unary(Unary::Negate, x) => write!(f, "-({x})"),
            Expression::Unary(op, x) => {
                let name = match op {
                    Unary::Abs => "abs",
                    Unary::Exp => "exp",
                    _ => "ln",
                };
                write!(f, "{name}({x})")
            }
            Expression::Binary(Binary::Power, a, b) => write!(f, "power({a}, {b})"),
            Expression::Binary(op, a, b) => {
                let sign = match op {
                    Binary::Add => "+",
                    Binary::Subtract => "-",
                    Binary::Multiply => "*",
                    _ => "/",
                };
                write!(f, "({a} {sign} {b})")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn distance() -> Expression {
        let d = |c| Expression::column(0, c).binary(Binary::Subtract, Expression::column(1, c));
        let square = |e: Expression| e.binary(Binary::Power, Expression::literal(2.0));
        square(d(1)).binary(Binary::Add, square(d(2)))
    }

    #[test]
    fn evaluates_as_sql_with_nulls() {
        let row = |o: usize, ColumnIndex(c): ColumnIndex| Some((10 * o + c) as f64);
        // (1 - 11)² + (2 - 12)²
        assert_eq!(distance().evaluate(&row), Some(200.0));
        let null = |o: usize, ColumnIndex(c): ColumnIndex| (o, c) != (1, 2);
        let with_null = |o, c| null(o, c).then(|| row(o, c)).flatten();
        assert_eq!(distance().evaluate(&with_null), None);
        let ln0 = Expression::literal(0.0).unary(Unary::Ln);
        assert_eq!(ln0.evaluate(&row), Some(f64::NEG_INFINITY));
        let over0 = Expression::literal(1.0).binary(Binary::Divide, Expression::literal(0.0));
        assert_eq!(over0.evaluate(&row), Some(f64::INFINITY));
    }

    #[test]
    fn lists_and_shows_its_columns() {
        let columns: Vec<(usize, usize)> = distance()
            .columns()
            .into_iter()
            .map(|(o, c)| (o, c.0))
            .collect();
        assert_eq!(columns, [(0, 1), (1, 1), (0, 2), (1, 2)]);
        assert_eq!(
            distance().to_string(),
            "(power((#0.1 - #1.1), 2) + power((#0.2 - #1.2), 2))"
        );
        assert_eq!(Expression::literal(f64::NAN), Expression::literal(f64::NAN));
    }
}
