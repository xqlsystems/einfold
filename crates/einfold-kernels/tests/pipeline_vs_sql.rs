// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Property: EinFold's fused pipeline algorithm computes what the SQL query a
//! pipeline stands for computes on DataFusion, the host, for random
//! attention-shaped pipelines over random tables.
//!
//! Each case draws tables `q(t, d, x)`, `k(t, d, x)` and `v(t, [c,] x)` with
//! random extents, and a pipeline: scores `SUM(q.x * k.x)`, or the sum of a
//! row expression of `q.x` and `k.x`, such as a Gaussian kernel's
//! `-power(q.x - k.x, 2)`, either over `d` or, without it, over a `CROSS
//! JOIN`'s one row per pair; scaled by a
//! random constant or not, under none, one or two random conditions between
//! `q.t` and `k.t`, `q.t <op> k.t + c` (either way round, any of `<`, `<=`,
//! `>`, `>=`, with a small offset `c` or none), two of which may make a band
//! such as a sliding window's;
//! weights `exp(s - MAX(s) OVER (PARTITION BY i))` or `exp(s)`; and an average
//! normalized by the kept groups' weights (a `norm` relation of `SUM(w) … GROUP BY i`) or by the
//! joined rows' (`SUM(w * v.x) / SUM(w)`). Values are drawn so that scores
//! spread widely, and some are NaN, ±inf, or large enough that `exp(s)`
//! overflows, and some rows' scores are all `-inf`, or `-inf` beside a NaN
//! in their first tile of keys only. Rows are shuffled and split into batches at random. The
//! invariants:
//!
//! 1. **Runs when it can, and declines when it can't.** With every table
//!    complete, a producer that reads both of its operands, and no special
//!    values, the algorithm runs; with special values it may decline. When a table misses a row, repeats one,
//!    or holds a NULL factor, it either returns `None`, so that the caller
//!    runs the host's plan, or (when the gap only shrinks a dimension's
//!    range) gives SQL's result, as below.
//! 2. **Same rows.** The same keys as SQL: no rows for a query position that
//!    the condition leaves no key.
//! 3. **Same values.** NaN where SQL gives NaN, the same infinity, and
//!    otherwise within `1e-9·max|v.x|` of SQL's value: an average of `v.x`
//!    with positive weights, so a scale on which any wrong weight, missed
//!    term or masked pair counted shows.
//! 4. **Deterministic.** Other batches, another order and another number of
//!    threads give the same rows, bit for bit.
//!
//! Reproduce a failure with `EINFOLD_SEED=<seed printed>`; search longer with
//! `EINFOLD_CASES=<n>`.

use std::sync::Arc;

use datafusion::arrow::array::{Array, Float64Array, Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::datasource::MemTable;
use datafusion::prelude::{SessionConfig, SessionContext};
use einfold_ir::{
    Aggregate, Binary, ColumnIndex, Comparison, Condition, Dimension, DimensionId, Expression,
    Fold, Key, KeyEquality, Normalizer, Operand, Output, Pipeline, RowValue, Scale, Shift, Unary,
    Weight,
};
use einfold_kernels::{FloatKeys, Host, Options, Relation};
use einfold_testkit::Rng;

/// A table of integer keys and a nullable `DOUBLE` value, as rows.
struct Table {
    schema: SchemaRef,
    rows: Vec<(Vec<i64>, Option<f64>)>,
}

impl Table {
    /// Every key tuple of `extents`, with values from `value`.
    fn complete(names: &[&str], extents: &[usize], mut value: impl FnMut() -> f64) -> Table {
        let mut fields: Vec<Field> = (names.iter())
            .map(|n| Field::new(*n, DataType::Int64, false))
            .collect();
        fields.push(Field::new("x", DataType::Float64, true));
        let cells: usize = extents.iter().product();
        let rows = (0..cells)
            .map(|mut c| {
                let mut keys = vec![0; extents.len()];
                for (k, &n) in keys.iter_mut().zip(extents).rev() {
                    *k = (c % n) as i64;
                    c /= n;
                }
                (keys, Some(value()))
            })
            .collect();
        Table {
            schema: Arc::new(Schema::new(fields)),
            rows,
        }
    }

    /// The rows shuffled, in partitions of batches of random sizes.
    fn partitions(&self, rng: &mut Rng) -> Vec<Vec<RecordBatch>> {
        let mut order: Vec<usize> = (0..self.rows.len()).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let mut parts = vec![Vec::new(); 1 + rng.below(3)];
        let mut start = 0;
        while start < order.len() {
            let end = (start + 1 + rng.below(300)).min(order.len());
            let rows = &order[start..end];
            let keys = self.schema.fields().len() - 1;
            let mut columns: Vec<Arc<dyn Array>> = (0..keys)
                .map(|c| {
                    Arc::new(Int64Array::from_iter_values(
                        rows.iter().map(|&r| self.rows[r].0[c]),
                    )) as _
                })
                .collect();
            columns.push(Arc::new(
                rows.iter()
                    .map(|&r| self.rows[r].1)
                    .collect::<Float64Array>(),
            ));
            let part = rng.below(parts.len());
            parts[part].push(RecordBatch::try_new(Arc::clone(&self.schema), columns).unwrap());
            start = end;
        }
        parts
    }
}

struct Case {
    q: Table,
    k: Table,
    v: Table,
    pipeline: Pipeline,
    sql: String,
    /// Whether every table has one non-NULL row per key tuple.
    complete: bool,
    /// The largest `|v.x|`, the scale of every average.
    v_scale: f64,
    /// For a sum that nothing normalizes, SQL for each group's `Σ |w · v.x|`,
    /// the scale of its rounding.
    magnitudes: Option<String>,
}

/// A value: mostly in [-1, 1] times `spread`, sometimes special.
fn value(rng: &mut Rng, spread: f64, specials: bool) -> f64 {
    if specials && rng.chance(0.004) {
        return [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e300, -1e300][rng.below(5)];
    }
    (2.0 * rng.unit() - 1.0) * spread
}

/// A producer's row value, as the IR and as SQL over `q.x` and `k.x`: a
/// product, or one of several row expressions, kernels of a distance among
/// them, with `x` operand `q`'s and `k`'s value column. Rarely, an
/// expression that reads only one operand, or neither: the fused algorithm
/// places only the operands a producer reads, so it can't tell which rows
/// the other has, and declines.
fn row_value(rng: &mut Rng, x: usize) -> (RowValue, &'static str) {
    let (q, k) = (Expression::column(0, x), Expression::column(1, x));
    let lit = Expression::literal;
    let d = || q.clone().binary(Binary::Subtract, k.clone());
    if rng.chance(0.1) {
        return match rng.below(3) {
            0 => (RowValue::Expression(q.unary(Unary::Exp)), "exp(q.x)"),
            1 => (
                RowValue::Expression(k.unary(Unary::Abs).unary(Unary::Negate)),
                "-abs(k.x)",
            ),
            _ => (RowValue::Expression(lit(0.5)), "CAST(0.5 AS DOUBLE)"),
        };
    }
    match rng.below(6) {
        0 | 1 => (RowValue::Product, "q.x * k.x"),
        2 => (
            RowValue::Expression(d().binary(Binary::Power, lit(2.0)).unary(Unary::Negate)),
            "-power(q.x - k.x, 2.0)",
        ),
        3 => (
            RowValue::Expression(d().unary(Unary::Abs).unary(Unary::Negate)),
            "-abs(q.x - k.x)",
        ),
        4 => (
            RowValue::Expression(q.clone().binary(Binary::Multiply, k.clone())),
            "q.x * k.x",
        ),
        _ => (
            RowValue::Expression(
                q.clone()
                    .binary(
                        Binary::Multiply,
                        k.clone().binary(Binary::Divide, lit(4.0)).unary(Unary::Exp),
                    )
                    .binary(Binary::Add, lit(0.5)),
            ),
            "q.x * exp(k.x / 4.0) + 0.5",
        ),
    }
}

fn case(rng: &mut Rng) -> Case {
    // A producer with a row expression and nothing to sum, the pair's one
    // row of a `CROSS JOIN`, as a Gaussian kernel's over points; or summed
    // over a feature dimension `d`, as attention's.
    let cross = rng.chance(0.25);
    let (nq, nk) = (1 + rng.below(150), 1 + rng.below(150));
    let nd = if cross { 1 } else { 1 + rng.below(6) };
    let channels = (rng.chance(0.6)).then(|| 1 + rng.below(4));
    let specials = rng.chance(0.3);
    let spread = [1.0, 4.0, 12.0][rng.below(3)];
    let (keys, extents): (&[&str], _) = match cross {
        true => (&["t"], ([nq].to_vec(), [nk].to_vec())),
        false => (&["t", "d"], ([nq, nd].to_vec(), [nk, nd].to_vec())),
    };
    let mut q = Table::complete(keys, &extents.0, || 0.0);
    let mut k = Table::complete(keys, &extents.1, || 0.0);
    for r in q.rows.iter_mut().chain(k.rows.iter_mut()) {
        r.1 = Some(value(rng, spread, specials));
    }
    let mut v = match channels {
        Some(c) => Table::complete(&["t", "c"], &[nk, c], || 0.0),
        None => Table::complete(&["t"], &[nk], || 0.0),
    };
    for r in v.rows.iter_mut() {
        r.1 = Some(value(rng, 1.0, specials));
    }
    // Rows of q whose scores are -inf: every one, which SQL turns into NaN;
    // or only those with the first 64 keys, beside a NaN score, with finite
    // ones after. Rows are (t, d) in order, so row t·nd + d is (t, d).
    let at = |t: usize, d: usize| t * nd + d;
    let target = rng.below(nq);
    let extremes = rng.below(6);
    match extremes {
        0 => {
            q.rows[at(target, 0)].1 = Some(f64::NEG_INFINITY);
            for j in 0..nk {
                let x = &mut k.rows[at(j, 0)].1;
                *x = x.map(|x| x.abs() + 0.1);
            }
        }
        1 if nk > 64 => {
            q.rows[at(target, 0)].1 = Some(-1e300);
            for j in 0..64 {
                k.rows[at(j, 0)].1 = Some(1e300);
            }
            k.rows[at(rng.below(64), nd - 1)].1 = Some(f64::NAN);
        }
        _ => {}
    }
    // Sometimes a table that isn't complete: a missing, repeated or NULL row.
    let complete = !rng.chance(0.15);
    if !complete {
        let t = match rng.below(3) {
            0 => &mut q,
            1 => &mut k,
            _ => &mut v,
        };
        let r = rng.below(t.rows.len());
        match rng.below(3) {
            0 => drop(t.rows.remove(r)),
            1 => t.rows.push(t.rows[r].clone()),
            _ => t.rows[r].1 = None,
        }
    }
    let v_scale = (v.rows.iter())
        .filter_map(|r| r.1)
        .filter(|x| x.is_finite())
        .fold(0.0f64, |m, x| m.max(x.abs()));

    // Dimensions: 0 = i (q.t), 1 = e (d), 2 = j (k.t), then 3 = c (v.c);
    // without d, 0 = i, 1 = j, 2 = c.
    let (di, dj, dc) = if cross { (0, 1, 2) } else { (0, 2, 3) };
    let x = keys.len();
    let (row, row_sql) = match cross {
        // A product over a CROSS JOIN is an einsum the kernel doesn't take.
        true => loop {
            let r = row_value(rng, x);
            if matches!(r.0, RowValue::Expression(_)) {
                break r;
            }
        },
        false => row_value(rng, x),
    };
    // A producer that reads one operand only, or neither, declines (see
    // `row_value`).
    let reads_both = match &row {
        RowValue::Expression(e) => {
            let read = e.columns();
            (0..2).all(|o| read.iter().any(|&(operand, _)| operand == o))
        }
        _ => true,
    };
    const EQ: Dimension = Dimension {
        equality: KeyEquality::Equal,
    };
    let key = |d, c| Key::new(DimensionId(d), ColumnIndex(c));
    let out = |d| Output {
        dimension: DimensionId(d),
        domain_key: None,
    };
    let ops = match cross {
        true => vec![
            Operand {
                keys: vec![key(di, 0)],
                factor: Some(ColumnIndex(1)),
            },
            Operand {
                keys: vec![key(dj, 0)],
                factor: Some(ColumnIndex(1)),
            },
        ],
        false => vec![
            Operand {
                keys: vec![key(di, 0), key(1, 1)],
                factor: Some(ColumnIndex(2)),
            },
            Operand {
                keys: vec![key(dj, 0), key(1, 1)],
                factor: Some(ColumnIndex(2)),
            },
        ],
    };
    let mut producer = Fold::try_new(
        vec![EQ; if cross { 2 } else { 3 }],
        ops,
        vec![out(di), out(dj)],
        row,
        Aggregate::sum(),
        None,
    )
    .unwrap();
    let comparisons = [
        (Comparison::Less, "<"),
        (Comparison::LessOrEqual, "<="),
        (Comparison::Greater, ">"),
        (Comparison::GreaterOrEqual, ">="),
    ];
    // None, one condition, or two: a band such as a sliding window's, when
    // they bound `k.t` from both sides. Offsets are small, against extents
    // of up to 150, so that bands are narrow and wide, and some empty.
    let mut condition = String::new();
    let mut conditions = Vec::new();
    let count = [0, 1, 1, 2][rng.below(4)];
    for _ in 0..count {
        let (comparison, op) = comparisons[rng.below(4)];
        let offset = match rng.chance(0.3) {
            true => 0,
            false => rng.below(81) as i64 - 40,
        };
        let (left, right, l, r) = match rng.chance(0.5) {
            true => (di, dj, "q.t", "k.t"),
            false => (dj, di, "k.t", "q.t"),
        };
        condition += &format!(" AND {l} {op} {r} + {offset}");
        conditions.push(Condition {
            left: DimensionId(left),
            comparison,
            right: DimensionId(right),
            offset,
        });
    }
    producer = producer.with_conditions(conditions).unwrap();
    let (scale, scale_sql) = match rng.below(3) {
        0 => (Scale::One, String::new()),
        1 => (Scale::Times(-0.7), " * -0.7".to_string()),
        _ => (Scale::Over(8.0), " / 8.0".to_string()),
    };
    // A softmax's weight, or another expression of the score: positive, so
    // that a normalized average stays an average, or, for a sum that
    // nothing normalizes, the score itself, a chain of two einsums.
    let normalizer = [
        Normalizer::KeptGroups,
        Normalizer::JoinedRows,
        Normalizer::None,
    ][rng.below(3)];
    let lit = Expression::literal;
    let p = || {
        let p = Expression::column(0, 0);
        match scale {
            Scale::Times(c) => p.binary(Binary::Multiply, lit(c)),
            Scale::Over(c) => p.binary(Binary::Divide, lit(c)),
            _ => p,
        }
    };
    let one_over = |e: Expression| lit(1.0).binary(Binary::Divide, lit(1.0).binary(Binary::Add, e));
    let (weight_ir, weight) = match rng.below(5) {
        0 => (
            Weight::Of(one_over(p().unary(Unary::Negate).unary(Unary::Exp))),
            "1.0 / (1.0 + exp(-s))",
        ),
        1 => (
            Weight::Of(one_over(p().binary(Binary::Multiply, p()))),
            "1.0 / (1.0 + s * s)",
        ),
        2 if normalizer == Normalizer::None => (Weight::Of(p()), "s"),
        2 | 3 => (
            Weight::Exp {
                scale,
                shift: Shift::KeptMaximum,
            },
            "exp(s - MAX(s) OVER (PARTITION BY i))",
        ),
        _ => (
            Weight::Exp {
                scale,
                shift: Shift::None,
            },
            "exp(s)",
        ),
    };
    let (values, outputs, c_col, c_group) = match channels {
        Some(_) => (
            Operand {
                keys: vec![key(dj, 0), key(dc, 1)],
                factor: Some(ColumnIndex(2)),
            },
            vec![DimensionId(di), DimensionId(dc)],
            ", v.c",
            ", v.c",
        ),
        None => (
            Operand {
                keys: vec![key(dj, 0)],
                factor: Some(ColumnIndex(1)),
            },
            vec![DimensionId(di)],
            "",
            "",
        ),
    };
    let own = if channels.is_some() { vec![EQ] } else { vec![] };
    let pipeline =
        Pipeline::try_new(producer, weight_ir, own, values, outputs, normalizer).unwrap();
    let producer = match cross {
        true => format!(
            "SELECT q.t AS i, k.t AS j, ({row_sql}){scale_sql} AS s \
             FROM q CROSS JOIN k WHERE TRUE{condition}"
        ),
        false => format!(
            "SELECT q.t AS i, k.t AS j, SUM({row_sql}){scale_sql} AS s \
             FROM q JOIN k ON q.d = k.d{condition} GROUP BY q.t, k.t"
        ),
    };
    let scores =
        format!("WITH scores AS ({producer}), weights AS (SELECT i, j, {weight} AS w FROM scores)");
    let sql = match normalizer {
        Normalizer::KeptGroups => format!(
            "{scores}, norm AS (SELECT i, SUM(w) AS z FROM weights GROUP BY i) \
             SELECT weights.i{c_col}, SUM(weights.w * v.x) / norm.z \
             FROM weights JOIN v ON weights.j = v.t JOIN norm ON weights.i = norm.i \
             GROUP BY weights.i{c_group}, norm.z"
        ),
        Normalizer::JoinedRows => format!(
            "{scores} SELECT weights.i{c_col}, SUM(weights.w * v.x) / SUM(weights.w) \
             FROM weights JOIN v ON weights.j = v.t GROUP BY weights.i{c_group}"
        ),
        _ => format!(
            "{scores} SELECT weights.i{c_col}, SUM(weights.w * v.x) \
             FROM weights JOIN v ON weights.j = v.t GROUP BY weights.i{c_group}"
        ),
    };
    let magnitudes = (normalizer == Normalizer::None).then(|| {
        format!(
            "{scores} SELECT weights.i{c_col}, SUM(abs(weights.w * v.x)) \
             FROM weights JOIN v ON weights.j = v.t GROUP BY weights.i{c_group}"
        )
    });
    Case {
        q,
        k,
        v,
        pipeline,
        sql,
        // Special values, or the extreme scores above, may make it decline
        // too: sums that overflow, or expression weights that aren't finite.
        complete: complete && reads_both && !specials && extremes > 1,
        v_scale,
        magnitudes,
    }
}

/// Rows as (keys, value), sorted by keys.
fn rows(batches: &[RecordBatch]) -> Vec<(Vec<i64>, Option<f64>)> {
    let mut out = Vec::new();
    for b in batches {
        let keys = b.num_columns() - 1;
        let value = b
            .column(keys)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for r in 0..b.num_rows() {
            let k: Vec<i64> = (0..keys)
                .map(|c| {
                    b.column(c)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap()
                        .value(r)
                })
                .collect();
            out.push((k, value.is_valid(r).then(|| value.value(r))));
        }
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

const HOST: Host = Host {
    float_keys: FloatKeys::SignedZerosEqual,
};

/// The kernel's rows, from the tables in a random order and batches, on a
/// random number of threads, or `None` if it declined.
fn run(case: &Case, rng: &mut Rng) -> Option<Vec<RecordBatch>> {
    let batches = |t: &Table, rng: &mut Rng| -> Vec<RecordBatch> { t.partitions(rng).concat() };
    let (q, k, v) = (
        batches(&case.q, rng),
        batches(&case.k, rng),
        batches(&case.v, rng),
    );
    let schemas = [
        case.q.schema.as_ref(),
        case.k.schema.as_ref(),
        case.v.schema.as_ref(),
    ];
    let schema = einfold_kernels::pipeline_schema(&case.pipeline, &schemas).unwrap();
    let options = Options {
        host: HOST,
        threads: 1 + rng.below(12),
        ..Options::default()
    };
    let inputs = [
        Relation::new(&case.q.schema, &q),
        Relation::new(&case.k.schema, &k),
        Relation::new(&case.v.schema, &v),
    ];
    let piped = einfold_kernels::pipeline(&case.pipeline, inputs, schema, options).unwrap();
    piped.map(|p| p.batches)
}

#[tokio::test]
async fn pipeline_matches_sql() {
    let seed: u64 = std::env::var("EINFOLD_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xa77e_0001);
    let cases: u64 = match std::env::var("EINFOLD_CASES") {
        Ok(n) => n.parse().unwrap(),
        Err(_) if std::env::var("EINFOLD_SEED").is_ok() => 1,
        Err(_) => 150,
    };
    for i in 0..cases {
        let seed = seed.wrapping_add(i);
        let mut rng = Rng::seeded(seed, 0);
        let case = case(&mut rng);
        let fail = |what: String| panic!("seed {seed}: {what}\nSQL: {}", case.sql);
        let ours = run(&case, &mut rng);
        if case.complete && ours.is_none() {
            fail("declined complete inputs".into());
        }
        let Some(ours) = ours else { continue };
        let again = run(&case, &mut rng).unwrap();
        let bits = |r: Vec<(Vec<i64>, Option<f64>)>| -> Vec<_> {
            r.into_iter()
                .map(|(k, v)| (k, v.map(f64::to_bits)))
                .collect()
        };
        let (ours, again) = (rows(&ours), rows(&again));
        if bits(ours.clone()) != bits(again) {
            fail("two arrival orders and thread counts gave different bits".into());
        }

        let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(4));
        for (name, t) in [("q", &case.q), ("k", &case.k), ("v", &case.v)] {
            let mem = MemTable::try_new(Arc::clone(&t.schema), t.partitions(&mut rng)).unwrap();
            ctx.register_table(name, Arc::new(mem)).unwrap();
        }
        let theirs = rows(&ctx.sql(&case.sql).await.unwrap().collect().await.unwrap());
        if ours.len() != theirs.len() {
            fail(format!("{} rows, SQL has {}", ours.len(), theirs.len()));
        }
        // An average's rounding is on the scale of the values; a sum's, of
        // the sum of its terms' magnitudes.
        let magnitudes = match &case.magnitudes {
            Some(sql) => rows(&ctx.sql(sql).await.unwrap().collect().await.unwrap()),
            None => vec![(Vec::new(), Some(case.v_scale)); theirs.len()],
        };
        for (((ok, ov), (tk, tv)), (_, scale)) in ours.iter().zip(&theirs).zip(&magnitudes) {
            let bound = 1e-9 * scale.unwrap_or(0.0);
            let same = match (ov, tv) {
                (Some(x), Some(y)) if x.is_nan() || y.is_nan() => x.is_nan() && y.is_nan(),
                (Some(x), Some(y)) if x.is_infinite() || y.is_infinite() => x == y,
                (Some(x), Some(y)) => (x - y).abs() <= bound,
                (None, None) => true,
                _ => false,
            };
            if ok != tk || !same {
                fail(format!("row {ok:?} = {ov:?}, SQL has {tk:?} = {tv:?}"));
            }
        }
    }
}

/// The pairs of tiles visited and the pairs in all, for attention over 1,024
/// positions (16 tiles of 64) under `conditions` between `q.t` and `k.t`,
/// checking that every average of `v.x = 1` is 1.
fn tiles_visited(conditions: &[(Comparison, i64)]) -> (usize, usize) {
    let mut rng = Rng::seeded(7, 0);
    let l = 64 * 16;
    let q = Table::complete(&["t", "d"], &[l, 8], || rng.unit());
    let k = Table::complete(&["t", "d"], &[l, 8], || 0.5);
    let v = Table::complete(&["t"], &[l], || 1.0);
    const EQ: Dimension = Dimension {
        equality: KeyEquality::Equal,
    };
    let key = |d, c| Key::new(DimensionId(d), ColumnIndex(c));
    let out = |d| Output {
        dimension: DimensionId(d),
        domain_key: None,
    };
    let ops = vec![
        Operand {
            keys: vec![key(0, 0), key(1, 1)],
            factor: Some(ColumnIndex(2)),
        },
        Operand {
            keys: vec![key(2, 0), key(1, 1)],
            factor: Some(ColumnIndex(2)),
        },
    ];
    let conditions = (conditions.iter())
        .map(|&(comparison, offset)| Condition {
            left: DimensionId(0),
            comparison,
            right: DimensionId(2),
            offset,
        })
        .collect();
    let producer = Fold::try_new(
        vec![EQ; 3],
        ops,
        vec![out(0), out(2)],
        RowValue::Product,
        Aggregate::sum(),
        None,
    )
    .unwrap()
    .with_conditions(conditions)
    .unwrap();
    let values = Operand {
        keys: vec![key(2, 0)],
        factor: Some(ColumnIndex(1)),
    };
    let pipeline = Pipeline::try_new(
        producer,
        Weight::Exp {
            scale: Scale::One,
            shift: Shift::KeptMaximum,
        },
        vec![],
        values,
        vec![DimensionId(0)],
        Normalizer::KeptGroups,
    )
    .unwrap();
    let batches = |t: &Table| t.partitions(&mut Rng::seeded(1, 0)).concat();
    let (bq, bk, bv) = (batches(&q), batches(&k), batches(&v));
    let schemas = [q.schema.as_ref(), k.schema.as_ref(), v.schema.as_ref()];
    let schema = einfold_kernels::pipeline_schema(&pipeline, &schemas).unwrap();
    let inputs = [
        Relation::new(&q.schema, &bq),
        Relation::new(&k.schema, &bk),
        Relation::new(&v.schema, &bv),
    ];
    let piped = einfold_kernels::pipeline(&pipeline, inputs, schema, Options::default())
        .unwrap()
        .unwrap();
    // Every average of v.x = 1 is 1.
    let rows = rows(&piped.batches);
    assert_eq!(rows.len(), l);
    assert!(rows.iter().all(|(_, x)| (x.unwrap() - 1.0).abs() < 1e-12));
    (piped.tiles_visited, piped.tiles)
}

/// The causal mask `q.t >= k.t` on a square leaves about half the pairs of
/// tiles unvisited: of 16 × 16, those on and below the diagonal are visited.
#[test]
fn causal_masks_skip_half_the_tiles() {
    let causal = [(Comparison::GreaterOrEqual, 0)];
    assert_eq!(tiles_visited(&causal), (16 * 17 / 2, 256));
}

/// A sliding window of 64 keys, `q.t >= k.t AND q.t - k.t < 64`, visits each
/// query tile's diagonal tile and the one before it: 16 + 15 of 256.
#[test]
fn sliding_windows_visit_a_band_of_tiles() {
    let window = [(Comparison::GreaterOrEqual, 0), (Comparison::Less, 64)];
    assert_eq!(tiles_visited(&window), (16 + 15, 256));
}

/// A pipeline may leave one of its own dimensions out of its outputs, to be
/// summed away: `SELECT i, SUM(w * v.x) / norm.z ... GROUP BY i` over values
/// keyed by `(t, c)`. The fused algorithm writes a row for each position of
/// its own dimensions, so it declines such a pipeline, and the caller runs
/// SQL's plan instead.
#[test]
fn declines_a_pipeline_that_sums_its_own_dimension_away() {
    let q = Table::complete(&["t", "d"], &[4, 2], || 0.5);
    let k = Table::complete(&["t", "d"], &[4, 2], || 0.25);
    let v = Table::complete(&["t", "c"], &[4, 3], || 1.0);
    const EQ: Dimension = Dimension {
        equality: KeyEquality::Equal,
    };
    let key = |d, c| Key::new(DimensionId(d), ColumnIndex(c));
    let out = |d| Output {
        dimension: DimensionId(d),
        domain_key: None,
    };
    let ops = vec![
        Operand {
            keys: vec![key(0, 0), key(1, 1)],
            factor: Some(ColumnIndex(2)),
        },
        Operand {
            keys: vec![key(2, 0), key(1, 1)],
            factor: Some(ColumnIndex(2)),
        },
    ];
    let producer = Fold::try_new(
        vec![EQ; 3],
        ops,
        vec![out(0), out(2)],
        RowValue::Product,
        Aggregate::sum(),
        None,
    )
    .unwrap();
    let values = Operand {
        keys: vec![key(2, 0), key(3, 1)],
        factor: Some(ColumnIndex(2)),
    };
    // Dimension 3, `v.c`, is the pipeline's own, and not an output.
    let pipeline = Pipeline::try_new(
        producer,
        Weight::Exp {
            scale: Scale::One,
            shift: Shift::KeptMaximum,
        },
        vec![EQ],
        values,
        vec![DimensionId(0)],
        Normalizer::KeptGroups,
    )
    .unwrap();
    let batches = |t: &Table| t.partitions(&mut Rng::new(1)).concat();
    let (bq, bk, bv) = (batches(&q), batches(&k), batches(&v));
    let schemas = [q.schema.as_ref(), k.schema.as_ref(), v.schema.as_ref()];
    let schema = einfold_kernels::pipeline_schema(&pipeline, &schemas).unwrap();
    let inputs = [
        Relation::new(&q.schema, &bq),
        Relation::new(&k.schema, &bk),
        Relation::new(&v.schema, &bv),
    ];
    let piped = einfold_kernels::pipeline(&pipeline, inputs, schema, Options::default()).unwrap();
    assert!(piped.is_none());
}

/// With no queries or keys, attention has no rows, whatever the values: an
/// empty `q` and `k` give no positions of the query or of the feature, so no
/// rows, as SQL's inner joins give none.
#[test]
fn gives_no_rows_without_queries_or_keys() {
    let q = Table::complete(&["t", "d"], &[0, 2], || 0.5);
    let k = Table::complete(&["t", "d"], &[0, 2], || 0.25);
    let v = Table::complete(&["t"], &[4], || 1.0);
    const EQ: Dimension = Dimension {
        equality: KeyEquality::Equal,
    };
    let key = |d, c| Key::new(DimensionId(d), ColumnIndex(c));
    let out = |d| Output {
        dimension: DimensionId(d),
        domain_key: None,
    };
    let ops = vec![
        Operand {
            keys: vec![key(0, 0), key(1, 1)],
            factor: Some(ColumnIndex(2)),
        },
        Operand {
            keys: vec![key(2, 0), key(1, 1)],
            factor: Some(ColumnIndex(2)),
        },
    ];
    let producer = Fold::try_new(
        vec![EQ; 3],
        ops,
        vec![out(0), out(2)],
        RowValue::Product,
        Aggregate::sum(),
        None,
    )
    .unwrap();
    let values = Operand {
        keys: vec![key(2, 0)],
        factor: Some(ColumnIndex(1)),
    };
    let pipeline = Pipeline::try_new(
        producer,
        Weight::Exp {
            scale: Scale::One,
            shift: Shift::KeptMaximum,
        },
        vec![],
        values,
        vec![DimensionId(0)],
        Normalizer::KeptGroups,
    )
    .unwrap();
    let batches = |t: &Table| t.partitions(&mut Rng::new(1)).concat();
    let (bq, bk, bv) = (batches(&q), batches(&k), batches(&v));
    let schemas = [q.schema.as_ref(), k.schema.as_ref(), v.schema.as_ref()];
    let schema = einfold_kernels::pipeline_schema(&pipeline, &schemas).unwrap();
    let inputs = [
        Relation::new(&q.schema, &bq),
        Relation::new(&k.schema, &bk),
        Relation::new(&v.schema, &bv),
    ];
    let piped = einfold_kernels::pipeline(&pipeline, inputs, schema, Options::default()).unwrap();
    // Declining is right too: the caller's SQL then gives no rows.
    assert!(piped.is_none_or(|p| rows(&p.batches).is_empty()));
}

/// A row's sums overflow in its first tile of keys, then a larger score in
/// a later tile shifts every weight before it to 0. SQL weighs each term by
/// the final maximum, `exp(0 - 800) · 1e308 = 0`, so the average is 0; the
/// fused algorithm's running sums would rescale infinity to NaN. It
/// declines instead, normalized or not, so that the caller's SQL runs.
#[test]
fn declines_where_a_rescale_meets_an_overflowed_sum() {
    for normalizer in [Normalizer::KeptGroups, Normalizer::None] {
        let q = Table::complete(&["t", "d"], &[1, 1], || 1.0);
        let mut i = 0;
        let k = Table::complete(&["t", "d"], &[65, 1], || {
            i += 1;
            if i == 65 {
                800.0
            } else {
                0.0
            }
        });
        let mut i = 0;
        let v = Table::complete(&["t"], &[65], || {
            i += 1;
            if i == 65 {
                0.0
            } else {
                1e308
            }
        });
        const EQ: Dimension = Dimension {
            equality: KeyEquality::Equal,
        };
        let key = |d, c| Key::new(DimensionId(d), ColumnIndex(c));
        let out = |d| Output {
            dimension: DimensionId(d),
            domain_key: None,
        };
        let ops = vec![
            Operand {
                keys: vec![key(0, 0), key(1, 1)],
                factor: Some(ColumnIndex(2)),
            },
            Operand {
                keys: vec![key(2, 0), key(1, 1)],
                factor: Some(ColumnIndex(2)),
            },
        ];
        let producer = Fold::try_new(
            vec![EQ; 3],
            ops,
            vec![out(0), out(2)],
            RowValue::Product,
            Aggregate::sum(),
            None,
        )
        .unwrap();
        let values = Operand {
            keys: vec![key(2, 0)],
            factor: Some(ColumnIndex(1)),
        };
        let pipeline = Pipeline::try_new(
            producer,
            Weight::Exp {
                scale: Scale::One,
                shift: Shift::KeptMaximum,
            },
            vec![],
            values,
            vec![DimensionId(0)],
            normalizer,
        )
        .unwrap();
        let batches = |t: &Table| t.partitions(&mut Rng::new(1)).concat();
        let (bq, bk, bv) = (batches(&q), batches(&k), batches(&v));
        let schemas = [q.schema.as_ref(), k.schema.as_ref(), v.schema.as_ref()];
        let schema = einfold_kernels::pipeline_schema(&pipeline, &schemas).unwrap();
        let inputs = [
            Relation::new(&q.schema, &bq),
            Relation::new(&k.schema, &bk),
            Relation::new(&v.schema, &bv),
        ];
        let piped =
            einfold_kernels::pipeline(&pipeline, inputs, schema, Options::default()).unwrap();
        // Declining is right; so would be SQL's 0.
        if let Some(p) = piped {
            let rows = rows(&p.batches);
            assert!(
                rows.iter().all(|(_, x)| *x == Some(0.0)),
                "{normalizer:?}: {rows:?}"
            );
        }
    }
}
