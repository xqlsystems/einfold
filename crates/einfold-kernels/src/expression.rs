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
    /// The values, with the strides that place pair `(r, c)` at `r · row +
    /// c · column`, so that loops over pairs don't branch on the shape.
    fn strided(&self, width: usize) -> (&[f64], usize, usize) {
        match self {
            Lanes::Scalar(x) => (std::slice::from_ref(x), 0, 0),
            Lanes::Rows(v) => (v, 1, 0),
            Lanes::Columns(v) => (v, 0, 1),
            Lanes::Block(v) => (v, width, 1),
        }
    }

    /// Each pair's value, row by row, into `block`, as `write` says.
    pub(crate) fn write(&self, (rows, width): (usize, usize), block: &mut [f64], write: Write) {
        let (v, rs, cs) = self.strided(width);
        for (r, row) in block.chunks_exact_mut(width).take(rows).enumerate() {
            for (c, b) in row.iter_mut().enumerate() {
                let x = v[r * rs + c * cs];
                *b = match write {
                    Write::Replace => x,
                    Write::Add => *b + x,
                };
            }
        }
    }
}

/// How [`Lanes::write`] writes into a block.
#[derive(Clone, Copy)]
pub(crate) enum Write {
    /// Each value replaces the block's.
    Replace,
    /// Each value is added to the block's.
    Add,
}

/// Which side of a binary operator a scalar is on.
#[derive(Clone, Copy)]
enum Scalar {
    /// `v op x`.
    Right,
    /// `x op v`.
    Left,
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
    let with = |v: &[f64], x: f64, scalar: Scalar| {
        (v.iter())
            .map(|&v| match scalar {
                Scalar::Right => op.apply(v, x),
                Scalar::Left => op.apply(x, v),
            })
            .collect()
    };
    match (&a, &b) {
        (Lanes::Scalar(x), Lanes::Scalar(y)) => Lanes::Scalar(op.apply(*x, *y)),
        (Lanes::Rows(u), Lanes::Rows(v)) => Lanes::Rows(zip(u, v)),
        (Lanes::Columns(u), Lanes::Columns(v)) => Lanes::Columns(zip(u, v)),
        (Lanes::Rows(u), Lanes::Scalar(y)) => Lanes::Rows(with(u, *y, Scalar::Right)),
        (Lanes::Scalar(x), Lanes::Rows(v)) => Lanes::Rows(with(v, *x, Scalar::Left)),
        (Lanes::Columns(u), Lanes::Scalar(y)) => Lanes::Columns(with(u, *y, Scalar::Right)),
        (Lanes::Scalar(x), Lanes::Columns(v)) => Lanes::Columns(with(v, *x, Scalar::Left)),
        _ => {
            let (u, v) = (a.strided(width), b.strided(width));
            // One loop per operator, so that none branches on it per pair.
            Lanes::Block(match op {
                Binary::Add => block(|a, b| Binary::Add.apply(a, b), u, v, (rows, width)),
                Binary::Subtract => block(|a, b| Binary::Subtract.apply(a, b), u, v, (rows, width)),
                Binary::Multiply => block(|a, b| Binary::Multiply.apply(a, b), u, v, (rows, width)),
                Binary::Divide => block(|a, b| Binary::Divide.apply(a, b), u, v, (rows, width)),
                Binary::Power => block(|a, b| Binary::Power.apply(a, b), u, v, (rows, width)),
                _ => block(|a, b| op.apply(a, b), u, v, (rows, width)),
            })
        }
    }
}

/// `f` of `u` and `v`, given as [`Lanes::strided`] gives them, at every
/// pair of a tile, row by row.
fn block(
    f: impl Fn(f64, f64) -> f64,
    (u, urs, ucs): (&[f64], usize, usize),
    (v, vrs, vcs): (&[f64], usize, usize),
    (rows, width): (usize, usize),
) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows * width);
    for r in 0..rows {
        let (u, v) = (&u[r * urs..], &v[r * vrs..]);
        out.extend((0..width).map(|c| f(u[c * ucs], v[c * vcs])));
    }
    out
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
            evaluate(e, (rows, width), &lanes).write((rows, width), &mut block, Write::Replace);
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
