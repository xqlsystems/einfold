// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A row expression evaluated over a tile of pairs at once: the fused
//! algorithm's producer block when the producer isn't a product.
//!
//! In a tile of `rows × width` pairs, a column of the first operand varies
//! only along the rows, and one of the second only along the columns. Each
//! value is kept in the smallest shape it varies in, a scalar, a row or a
//! column vector, or a block, so that `t.x * 2`, say, is computed once per
//! row, and only an operator of both operands' values works per pair. Every
//! operator is [`Unary::apply`] or [`Binary::apply`], element by element, so
//! each pair's value has the bits [`Expression::evaluate`] gives it.

use einfold_ir::{Binary, ColumnIndex, Constant, Expression, Unary};

/// Values over a tile of `rows × width` pairs, by what they vary along.
#[derive(Debug)]
pub(crate) enum Lanes {
    /// One value for every pair.
    Scalar(f64),
    /// One value per row, the same along it.
    Rows(Vec<f64>),
    /// One value per column, the same down it.
    Columns(Vec<f64>),
    /// One value per pair, row by row.
    Block(Vec<f64>),
}

impl Lanes {
    /// The value at row `r`, column `c`.
    fn at(&self, r: usize, c: usize, width: usize) -> f64 {
        match self {
            Lanes::Scalar(x) => *x,
            Lanes::Rows(v) => v[r],
            Lanes::Columns(v) => v[c],
            Lanes::Block(v) => v[r * width + c],
        }
    }

    /// Each pair's value, row by row, as `block`, added to it when `add`.
    pub(crate) fn write(&self, (rows, width): (usize, usize), block: &mut [f64], add: bool) {
        for r in 0..rows {
            for (c, b) in block[r * width..(r + 1) * width].iter_mut().enumerate() {
                let x = self.at(r, c, width);
                *b = if add { *b + x } else { x };
            }
        }
    }
}

/// `e` over a tile of `rows × width` pairs, with `column(operand, column)`
/// the lanes of each column it reads.
pub(crate) fn evaluate(
    e: &Expression,
    (rows, width): (usize, usize),
    column: &impl Fn(usize, ColumnIndex) -> Lanes,
) -> Lanes {
    match e {
        Expression::Column { operand, column: c } => column(*operand, *c),
        Expression::Literal(Constant(x)) => Lanes::Scalar(*x),
        Expression::Unary(op, x) => unary(*op, evaluate(x, (rows, width), column)),
        Expression::Binary(op, a, b) => {
            let a = evaluate(a, (rows, width), column);
            let b = evaluate(b, (rows, width), column);
            binary(*op, a, b, (rows, width))
        }
    }
}

fn unary(op: Unary, x: Lanes) -> Lanes {
    let map = |mut v: Vec<f64>| {
        v.iter_mut().for_each(|x| *x = op.apply(*x));
        v
    };
    match x {
        Lanes::Scalar(x) => Lanes::Scalar(op.apply(x)),
        Lanes::Rows(v) => Lanes::Rows(map(v)),
        Lanes::Columns(v) => Lanes::Columns(map(v)),
        Lanes::Block(v) => Lanes::Block(map(v)),
    }
}

fn binary(op: Binary, a: Lanes, b: Lanes, (rows, width): (usize, usize)) -> Lanes {
    let zip = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(&a, &b)| op.apply(a, b)).collect();
    let with = |v: &[f64], x: f64, left: bool| {
        (v.iter())
            .map(|&v| if left { op.apply(v, x) } else { op.apply(x, v) })
            .collect()
    };
    match (&a, &b) {
        (Lanes::Scalar(x), Lanes::Scalar(y)) => Lanes::Scalar(op.apply(*x, *y)),
        (Lanes::Rows(u), Lanes::Rows(v)) => Lanes::Rows(zip(u, v)),
        (Lanes::Columns(u), Lanes::Columns(v)) => Lanes::Columns(zip(u, v)),
        (Lanes::Rows(u), Lanes::Scalar(y)) => Lanes::Rows(with(u, *y, true)),
        (Lanes::Scalar(x), Lanes::Rows(v)) => Lanes::Rows(with(v, *x, false)),
        (Lanes::Columns(u), Lanes::Scalar(y)) => Lanes::Columns(with(u, *y, true)),
        (Lanes::Scalar(x), Lanes::Columns(v)) => Lanes::Columns(with(v, *x, false)),
        _ => {
            let mut out = Vec::with_capacity(rows * width);
            for r in 0..rows {
                out.extend((0..width).map(|c| op.apply(a.at(r, c, width), b.at(r, c, width))));
            }
            Lanes::Block(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every pair's value is `Expression::evaluate`'s, bit for bit, whatever
    /// shapes the operators combine.
    #[test]
    fn a_tile_has_each_pairs_value() {
        let (rows, width) = (3, 4);
        let a = |r: usize, c: usize| [0.5, -2.0, f64::INFINITY][r] * [1.0, 2.0][c - 1];
        let b = |j: usize, c: usize| [0.0, -0.0, 3.0, f64::NAN][j] + [1.0, 7.0][c - 1];
        let col = |o, c| Expression::column(o, c);
        let lit = Expression::literal;
        let d = |c| col(0, c).binary(Binary::Subtract, col(1, c));
        let exprs = [
            d(1).binary(Binary::Power, lit(2.0))
                .binary(Binary::Add, d(2).binary(Binary::Power, lit(2.0)))
                .unary(Unary::Negate)
                .binary(Binary::Divide, lit(0.25)),
            col(0, 1)
                .unary(Unary::Exp)
                .binary(Binary::Multiply, lit(3.0)),
            lit(2.0).binary(Binary::Divide, col(1, 2)).unary(Unary::Ln),
            lit(1.0).binary(Binary::Add, lit(-0.0)),
            col(0, 2)
                .binary(Binary::Multiply, col(0, 1))
                .binary(Binary::Subtract, col(1, 1).unary(Unary::Abs)),
        ];
        for e in &exprs {
            let lanes = |o: usize, ColumnIndex(c): ColumnIndex| match o {
                0 => Lanes::Rows((0..rows).map(|r| a(r, c)).collect()),
                _ => Lanes::Columns((0..width).map(|j| b(j, c)).collect()),
            };
            let mut block = vec![0.0; rows * width];
            evaluate(e, (rows, width), &lanes).write((rows, width), &mut block, false);
            for r in 0..rows {
                for j in 0..width {
                    let value = |o: usize, ColumnIndex(c): ColumnIndex| {
                        Some(if o == 0 { a(r, c) } else { b(j, c) })
                    };
                    let want = e.evaluate(&value).unwrap();
                    let got = block[r * width + j];
                    assert!(
                        got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()),
                        "{e} at ({r}, {j}): {got}, want {want}"
                    );
                }
            }
        }
    }
}
