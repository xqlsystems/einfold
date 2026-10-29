// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Property: a row expression's value, as [`Expression::evaluate`] defines
//! it, is DataFusion's for the same SQL, bit for bit.
//!
//! Each case draws two tables `a(i, x, y)` and `b(j, x, y)` of `DOUBLE`
//! values, among them NULL, NaN, ±inf, ±0, negatives, and values large and
//! small enough that `exp` and `power` overflow and underflow, and a random
//! expression over both tables' columns of up to five levels, of every
//! operator: `+ - * /`, negation, `abs`, `exp`, `ln`, and `power` with a
//! positive constant exponent (DataFusion raises an error for zero to a
//! negative power, so a detector takes only positive ones). The invariant: for every pair of rows of
//! `SELECT a.i, b.j, <expression> FROM a CROSS JOIN b`, the value is NULL
//! exactly where `evaluate` gives `None`, and otherwise has the same bits,
//! up to NaN's payload.
//!
//! Reproduce a failure with `EINFOLD_SEED=<seed printed>`; search longer with
//! `EINFOLD_CASES=<n>`.

use std::sync::Arc;

use datafusion::arrow::array::{Array, Float64Array, Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use einfold_ir::{Binary, ColumnIndex, Expression, Unary};

struct Rng(u64);

impl Rng {
    /// A generator for `seed`, scrambled by SplitMix64 so that nearby seeds
    /// give unrelated streams, and never zero, which xorshift can't leave.
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        Rng((z ^ (z >> 31)).max(1))
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f64 {
        ((self.next() >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

/// A value: mostly ordinary, sometimes special.
fn value(rng: &mut Rng) -> Option<f64> {
    let special = [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        1.0,
        -1.0,
        800.0,
        -800.0,
        1e-300,
        1e300,
    ];
    match rng.below(10) {
        0 => None,
        1 | 2 => Some(special[rng.below(special.len())]),
        _ => Some((rng.unit() - 0.5) * 20.0),
    }
}

/// A random expression and its SQL, `depth` levels at most.
fn expression(rng: &mut Rng, depth: usize) -> (Expression, String) {
    let leaf = depth == 0 || rng.below(4) == 0;
    if leaf {
        return match rng.below(5) {
            0 => {
                let x = [0.0, 1.0, 2.0, -0.5, 0.25, 3.5][rng.below(6)];
                (Expression::literal(x), format!("CAST({x:?} AS DOUBLE)"))
            }
            n => {
                let (operand, table) = [(0, "a"), (1, "b")][n % 2];
                let (column, name) = [(1, "x"), (2, "y")][n / 3];
                (
                    Expression::column(operand, column),
                    format!("{table}.{name}"),
                )
            }
        };
    }
    match rng.below(9) {
        0..=3 => {
            let (op, sign) = [
                (Binary::Add, "+"),
                (Binary::Subtract, "-"),
                (Binary::Multiply, "*"),
                (Binary::Divide, "/"),
            ][rng.below(4)];
            let (a, sa) = expression(rng, depth - 1);
            let (b, sb) = expression(rng, depth - 1);
            (a.binary(op, b), format!("({sa} {sign} {sb})"))
        }
        4 => {
            let (a, sa) = expression(rng, depth - 1);
            let p = [1.0, 2.0, 3.0, 0.5, 1.5][rng.below(5)];
            (
                a.binary(Binary::Power, Expression::literal(p)),
                format!("power({sa}, CAST({p:?} AS DOUBLE))"),
            )
        }
        n => {
            let (op, name) = [
                (Unary::Negate, "-"),
                (Unary::Abs, "abs"),
                (Unary::Exp, "exp"),
                (Unary::Ln, "ln"),
            ][n - 5];
            let (a, sa) = expression(rng, depth - 1);
            (a.unary(op), format!("{name}({sa})"))
        }
    }
}

/// A table of `rows` rows, `(key, x, y)`, and its values.
fn table(ctx: &SessionContext, name: &str, rows: usize, rng: &mut Rng) -> Vec<[Option<f64>; 2]> {
    let values: Vec<[Option<f64>; 2]> = (0..rows).map(|_| [value(rng), value(rng)]).collect();
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", DataType::Int64, false),
        Field::new("x", DataType::Float64, true),
        Field::new("y", DataType::Float64, true),
    ]));
    let column = |c: usize| Float64Array::from(values.iter().map(|v| v[c]).collect::<Vec<_>>());
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values(0..rows as i64)),
            Arc::new(column(0)),
            Arc::new(column(1)),
        ],
    )
    .unwrap();
    let mem = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    ctx.register_table(name, Arc::new(mem)).unwrap();
    values
}

#[tokio::test]
async fn row_expressions_compute_what_datafusion_does() {
    let seed: u64 = std::env::var("EINFOLD_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xe4e5);
    let cases: u64 = match std::env::var("EINFOLD_CASES") {
        Ok(n) => n.parse().unwrap(),
        Err(_) if std::env::var("EINFOLD_SEED").is_ok() => 1,
        Err(_) => 300,
    };
    for i in 0..cases {
        let seed = seed.wrapping_add(i);
        let mut rng = Rng::new(seed);
        let ctx = SessionContext::new();
        let a = table(&ctx, "a", 1 + rng.below(8), &mut rng);
        let b = table(&ctx, "b", 1 + rng.below(8), &mut rng);
        let (e, sql) = expression(&mut rng, 5);
        let query = format!("SELECT a.k, b.k, {sql} AS v FROM a CROSS JOIN b");
        let fail =
            |what: String| -> ! { panic!("seed {seed}: {what}\nexpression: {e}\nSQL: {query}") };
        let batches = match ctx.sql(&query).await {
            Ok(frame) => frame.collect().await,
            Err(err) => Err(err),
        };
        let batches = batches.unwrap_or_else(|err| fail(format!("DataFusion failed: {err}")));
        let mut pairs = 0;
        for batch in &batches {
            let key = |c: usize| {
                let keys = batch.column(c).as_any().downcast_ref::<Int64Array>();
                keys.unwrap().clone()
            };
            let (ka, kb) = (key(0), key(1));
            let v = batch.column(2).as_any().downcast_ref::<Float64Array>();
            let v = v.unwrap_or_else(|| fail("the value isn't DOUBLE".into()));
            for r in 0..batch.num_rows() {
                pairs += 1;
                let rows = [&a[ka.value(r) as usize], &b[kb.value(r) as usize]];
                let column = |o: usize, ColumnIndex(c): ColumnIndex| rows[o][c - 1];
                let ours = e.evaluate(&column);
                let theirs = v.is_valid(r).then(|| v.value(r));
                let same = match (ours, theirs) {
                    (Some(x), Some(y)) if x.is_nan() || y.is_nan() => x.is_nan() && y.is_nan(),
                    (Some(x), Some(y)) => x.to_bits() == y.to_bits(),
                    (None, None) => true,
                    _ => false,
                };
                if !same {
                    fail(format!(
                        "rows {:?}: einfold gives {ours:?}, DataFusion {theirs:?}",
                        (ka.value(r), kb.value(r))
                    ));
                }
            }
        }
        if pairs != a.len() * b.len() {
            fail(format!("{pairs} pairs, of {}", a.len() * b.len()));
        }
    }
}
