// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Property: with einfold installed, DataFusion returns what it returns
//! without for softmax-weighted averages written in SQL, and `DetectFolds`
//! finds a fold pipeline in each spelling it claims to, and in none of the
//! look-alikes.
//!
//! Each case draws tables `q(t, d, x)`, `k(t, d, x)` and `v(t, c, x)`, each
//! with one row for every `(t, d)` or `(t, c)`, and sometimes damages one: a
//! row missing, repeated, or with a NULL `x`; an `x` that is NaN or an
//! infinity; a NULL `t`; or no rows at all. Then it draws a query:
//!
//! - **causal attention**, the maximum as a window (`MAX(s) OVER
//!   (PARTITION BY i)`) or as `MAX(s) … GROUP BY i` joined back, the scores
//!   divided or multiplied by a constant or neither, with the causal mask
//!   either way round, a band such as a sliding window's (`q.t - k.t < 5`,
//!   `k.t BETWEEN q.t - 4 AND q.t`, `abs(q.t - k.t) < 4`), an offset alone,
//!   or none, normalized by `norm.z`, either the sum or, as ddx (an XQL
//!   Systems project for automatic differentiation of SQL queries) writes
//!   it, each term (`SUM(w / norm.z * v.x)`);
//! - **a weighted average over joined rows**, `SUM(exp(s) * v.x) / SUM(exp(s))`
//!   with no shift, as a recommender or kernel smoother writes it;
//! - **other weights**: a sigmoid of attention's scores, averaged; a Cauchy
//!   kernel of points, `1 / (1 + (q.x - k.x)²)`, averaged; and the scores
//!   themselves, summed with no normalizer, `SUM(scores.s * v.x)`, a chain
//!   of two einsums;
//! - **kernel smoothers**, the same average with weights of a distance
//!   between points (`exp(-power(q.x - k.x, 2) / 0.5)` or `exp(-abs(q.x -
//!   k.x) * 2)`), over a `CROSS JOIN` of one coordinate per point with
//!   nothing to sum, or `exp(SUM(-power(q.x - k.x, 2)))` over every
//!   coordinate `d`, the Nadaraya–Watson estimator's shape;
//! - **look-alikes** it must leave alone: a maximum partitioned by the summed
//!   key or ordered, a normalizer of an expression of the weights
//!   (`exp(2 * w)`), and a normalizer of weights over other scores (the same
//!   fold, scaled differently).
//!
//! Invariants: the same rows as DataFusion without einfold, values within
//! `1e-9` of its, and the same NaNs and infinities; a pipeline found exactly
//! where the query is one (where a key column is nullable, only that none is
//! found in a look-alike); and the fused algorithm run whenever every key
//! has one row and every value is finite.
//!
//! Reproduce a failure with `EINFOLD_SEED=<seed printed>`; search longer with
//! `EINFOLD_CASES=<n>`.

use std::sync::Arc;

use datafusion::arrow::array::{Array, Float64Array, Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::prelude::{SessionConfig, SessionContext};
use einfold_testkit::Rng;

#[derive(Clone, Copy, Debug, PartialEq)]
enum Expect {
    /// A pipeline the rule must find.
    Pipeline,
    /// A look-alike it must leave alone.
    Decline,
}

/// A query and whether it is a pipeline.
fn query(rng: &mut Rng) -> (String, Expect) {
    let scale = [" / 8.0", " * 0.5", ""][rng.below(3)];
    let mask = [
        " AND q.t >= k.t",
        " AND k.t <= q.t",
        " AND q.t > k.t",
        "",
        // Bands: a sliding window, spelled as a difference, a sum or a
        // `BETWEEN`, and an offset on its own.
        " AND q.t >= k.t AND q.t - k.t < 5",
        " AND k.t BETWEEN q.t - 4 AND q.t",
        " AND k.t >= q.t - 7 AND q.t + 2 >= k.t",
        " AND q.t < k.t + 3",
        " AND abs(q.t - k.t) < 4",
        " AND 3 >= abs(k.t - q.t)",
    ][rng.below(10)];
    let scores = format!(
        "scores AS (SELECT q.t AS i, k.t AS j, SUM(q.x * k.x){scale} AS s \
         FROM q JOIN k ON q.d = k.d{mask} GROUP BY q.t, k.t)"
    );
    let average = |weights: &str, norm: &str| {
        format!(
            "WITH {scores}, {weights}, norm AS ({norm}) \
             SELECT weights.i AS t, v.c, SUM(weights.w * v.x) / norm.z AS x \
             FROM weights JOIN v ON weights.j = v.t JOIN norm ON weights.i = norm.i \
             GROUP BY weights.i, v.c, norm.z"
        )
    };
    let window = "weights AS (SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS w FROM scores)";
    let group_by = "m AS (SELECT i, MAX(s) AS m FROM scores GROUP BY i), \
                    weights AS (SELECT scores.i, scores.j, exp(scores.s - m.m) AS w \
                    FROM scores JOIN m ON scores.i = m.i)";
    let norm = "SELECT i, SUM(w) AS z FROM weights GROUP BY i";
    // The normalizer inside the sum, as ddx writes it: no GROUP BY key
    // depends on the weights.
    let inside = |weights: &str| {
        format!(
            "WITH {scores}, {weights}, norm AS ({norm}) \
             SELECT weights.i AS t, v.c, SUM(weights.w / norm.z * v.x) AS x \
             FROM weights JOIN v ON weights.j = v.t JOIN norm ON weights.i = norm.i \
             GROUP BY weights.i, v.c"
        )
    };
    // Kernel smoothers: weights of a distance between points rather than of
    // a dot product. Over a `CROSS JOIN` of one coordinate per point, with
    // nothing to sum; or summing the squared distance over every coordinate
    // `d`.
    let kernel = |weights: &str| {
        format!(
            "WITH weights AS ({weights}) \
             SELECT weights.i AS t, v.c, SUM(weights.w * v.x) / SUM(weights.w) AS x \
             FROM weights JOIN v ON weights.j = v.t GROUP BY weights.i, v.c"
        )
    };
    let distance = ["-power(q.x - k.x, 2.0) / 0.5", "-abs(q.x - k.x) * 2.0"][rng.below(2)];
    let cross = format!(
        "SELECT q.t AS i, k.t AS j, exp({distance}) AS w \
         FROM q CROSS JOIN k WHERE q.d = 0 AND k.d = 0{mask}"
    );
    let long = format!(
        "SELECT i, j, exp(s) AS w FROM (SELECT q.t AS i, k.t AS j, \
         SUM(-power(q.x - k.x, 2.0)) AS s FROM q JOIN k ON q.d = k.d{mask} GROUP BY q.t, k.t)"
    );
    // Weights other than a softmax's: a sigmoid of the scores, averaged; a
    // Cauchy kernel of points, averaged; and the scores themselves, summed
    // with no normalizer, a chain of two einsums.
    let sigmoid = "weights AS (SELECT i, j, 1.0 / (1.0 + exp(-s)) AS w FROM scores)";
    let cauchy = format!(
        "SELECT q.t AS i, k.t AS j, 1.0 / (1.0 + power(q.x - k.x, 2.0)) AS w \
         FROM q CROSS JOIN k WHERE q.d = 0 AND k.d = 0{mask}"
    );
    let chain = format!(
        "WITH {scores} SELECT scores.i AS t, v.c, SUM(scores.s * v.x) AS x \
         FROM scores JOIN v ON scores.j = v.t GROUP BY scores.i, v.c"
    );
    match rng.below(16) {
        13 => (average(sigmoid, norm), Expect::Pipeline),
        14 => (kernel(&cauchy), Expect::Pipeline),
        15 => (chain, Expect::Pipeline),
        10 | 11 => (kernel(&cross), Expect::Pipeline),
        12 => (kernel(&long), Expect::Pipeline),
        8 => (inside(window), Expect::Pipeline),
        9 => (inside(group_by), Expect::Pipeline),
        0 | 1 => (average(window, norm), Expect::Pipeline),
        2 | 3 => (average(group_by, norm), Expect::Pipeline),
        4 => (
            format!(
                "WITH {scores}, weights AS (SELECT i, j, exp(s) AS w FROM scores) \
                 SELECT weights.i AS t, v.c, SUM(weights.w * v.x) / SUM(weights.w) AS x \
                 FROM weights JOIN v ON weights.j = v.t GROUP BY weights.i, v.c"
            ),
            Expect::Pipeline,
        ),
        5 => {
            let wrong = [
                "weights AS (SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY j)) AS w FROM scores)",
                "weights AS (SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i ORDER BY j)) AS w FROM scores)",
            ][rng.below(2)];
            (average(wrong, norm), Expect::Decline)
        }
        6 => match rng.chance(0.5) {
            true => {
                let other = "SELECT i, SUM(exp(2.0 * w)) AS z FROM weights GROUP BY i";
                (average(window, other), Expect::Decline)
            }
            // A key named `s`, beside the weights merged with their scores.
            false if rng.chance(0.5) => (
                format!(
                    "WITH {scores}, weights AS (SELECT i AS s, j, exp(s) AS w FROM scores) \
                     SELECT weights.s AS t, v.c, SUM(weights.w * v.x) / SUM(weights.w) AS x \
                     FROM weights JOIN v ON weights.j = v.t GROUP BY weights.s, v.c"
                ),
                Expect::Pipeline,
            ),
            // The scores and their weights side by side, the scores summed
            // with the values: DataFusion merges the projections, and the
            // raw score is no weight.
            false => (
                format!(
                    "WITH {scores}, weights AS (SELECT i, j, s, exp(s) AS w FROM scores) \
                     SELECT weights.i AS t, v.c, SUM(weights.s * v.x) / SUM(weights.w) AS x \
                     FROM weights JOIN v ON weights.j = v.t GROUP BY weights.i, v.c"
                ),
                Expect::Decline,
            ),
        },
        _ => {
            // The normalizer sums weights of other scores: the same fold,
            // scaled differently, so the shift doesn't cancel.
            let other = format!(
                "WITH {scores}, \
                 scores2 AS (SELECT q.t AS i, k.t AS j, SUM(q.x * k.x) / 4.0 AS s \
                   FROM q JOIN k ON q.d = k.d{mask} GROUP BY q.t, k.t), \
                 {window}, \
                 weights2 AS (SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS w FROM scores2), \
                 norm AS (SELECT i, SUM(w) AS z FROM weights2 GROUP BY i) \
                 SELECT weights.i AS t, v.c, SUM(weights.w * v.x) / norm.z AS x \
                 FROM weights JOIN v ON weights.j = v.t JOIN norm ON weights.i = norm.i \
                 GROUP BY weights.i, v.c, norm.z"
            );
            (other, Expect::Decline)
        }
    }
}

/// What's wrong with a table, if anything.
#[derive(Clone, Copy, Debug)]
enum Damage {
    None,
    /// Row `r` is missing.
    Missing(usize),
    /// Row `r` appears twice.
    Repeated(usize),
    /// Row `r`'s `x` is NULL.
    NullValue(usize),
    /// Row `r`'s `x` is NaN or an infinity.
    Special(usize, f64),
    /// Row `r`'s `t` is NULL, in a nullable column.
    NullKey(usize),
    /// The table has no rows.
    Empty,
}

impl Damage {
    /// Some damage to a table of `rows` rows.
    fn draw(rng: &mut Rng, rows: usize) -> Self {
        let r = rng.below(rows);
        match rng.below(6) {
            0 => Damage::Missing(r),
            1 => Damage::Repeated(r),
            2 => Damage::NullValue(r),
            3 => Damage::Special(
                r,
                [f64::NAN, f64::INFINITY, f64::NEG_INFINITY][rng.below(3)],
            ),
            4 => Damage::NullKey(r),
            _ => Damage::Empty,
        }
    }

    /// Whether the fused algorithm must run: every key has exactly one row,
    /// and every value is finite. With a NaN or an infinity it may decline,
    /// as it does where unnormalized or expression weights meet one (#116).
    fn complete(self) -> bool {
        matches!(self, Damage::None)
    }
}

/// Registers `name(t, second, x)` with every `(t, second)` in `extents`, as
/// `damage` leaves it, in two partitions, in each of `ctxs`.
fn register(
    ctxs: &[&SessionContext],
    name: &str,
    second: &str,
    extents: (usize, usize),
    damage: Damage,
    rng: &mut Rng,
) {
    let schema = Arc::new(Schema::new(vec![
        Field::new("t", DataType::Int64, matches!(damage, Damage::NullKey(_))),
        Field::new(second, DataType::Int64, false),
        Field::new("x", DataType::Float64, true),
    ]));
    let mut rows: Vec<usize> = (0..extents.0 * extents.1).collect();
    match damage {
        Damage::Missing(r) => drop(rows.remove(r)),
        Damage::Repeated(r) => rows.push(r),
        Damage::Empty => rows.clear(),
        _ => {}
    }
    let values: Vec<Option<f64>> = rows
        .iter()
        .map(|&r| match damage {
            Damage::NullValue(d) if d == r => None,
            Damage::Special(d, x) if d == r => Some(x),
            _ => Some((rng.below(2001) as f64 - 1000.0) / 500.0),
        })
        .collect();
    let t = rows.iter().map(|&r| match damage {
        Damage::NullKey(d) if d == r => None,
        _ => Some((r / extents.1) as i64),
    });
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter(t)),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|&r| (r % extents.1) as i64),
            )),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .unwrap();
    let split = batch.num_rows() / 2;
    let parts = vec![
        vec![batch.slice(0, split)],
        vec![batch.slice(split, batch.num_rows() - split)],
    ];
    for ctx in ctxs {
        let mem = MemTable::try_new(Arc::clone(&schema), parts.clone()).unwrap();
        ctx.register_table(name, Arc::new(mem)).unwrap();
    }
}

/// Rows sorted by keys, a NULL `t` first.
fn rows(batches: &[RecordBatch]) -> Vec<(Option<i64>, i64, Option<f64>)> {
    let mut out = Vec::new();
    for b in batches {
        let key = |c: usize| {
            b.column(c)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .clone()
        };
        let (t, c) = (key(0), key(1));
        let x = b.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        for r in 0..b.num_rows() {
            out.push((
                t.is_valid(r).then(|| t.value(r)),
                c.value(r),
                x.is_valid(r).then(|| x.value(r)),
            ));
        }
    }
    out.sort_by_key(|r| (r.0, r.1));
    out
}

#[tokio::test]
async fn einfold_finds_pipelines_and_changes_no_answer() {
    let seed: u64 = std::env::var("EINFOLD_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xf1a5_4a77);
    let cases: u64 = match std::env::var("EINFOLD_CASES") {
        Ok(n) => n.parse().unwrap(),
        Err(_) if std::env::var("EINFOLD_SEED").is_ok() => 1,
        Err(_) => 120,
    };
    let (mut found, mut fused) = (0, 0);
    for i in 0..cases {
        let seed = seed.wrapping_add(i);
        let mut rng = Rng::seeded(seed, 0);
        let threads = [1 + rng.below(6), 1 + rng.below(6)];
        let config = |t: usize| SessionConfig::new().with_target_partitions(t);
        let host = SessionContext::new_with_config(config(threads[0]));
        let builder = SessionStateBuilder::new()
            .with_default_features()
            .with_config(config(threads[1]));
        let ours = SessionContext::new_with_state(einfold_datafusion::install(builder).build());
        let (l, d, c) = (1 + rng.below(90), 1 + rng.below(6), 1 + rng.below(4));
        // Sometimes one table is damaged.
        let tables = [("q", "d", (l, d)), ("k", "d", (l, d)), ("v", "c", (l, c))];
        let damaged = rng.chance(0.3).then(|| rng.below(3));
        let mut damage = Damage::None;
        for (i, (name, second, extents)) in tables.into_iter().enumerate() {
            let this = match damaged == Some(i) {
                true => Damage::draw(&mut rng, extents.0 * extents.1),
                false => Damage::None,
            };
            damage = match damaged == Some(i) {
                true => this,
                false => damage,
            };
            register(&[&host, &ours], name, second, extents, this, &mut rng);
        }
        let complete = damage.complete();
        let (sql, expect) = query(&mut rng);
        let fail = |what: String| {
            panic!("seed {seed}: {what} ({damage:?} in table {damaged:?})\nSQL: {sql}")
        };

        let frame = ours.sql(&sql).await.unwrap();
        let task = Arc::new(frame.task_ctx());
        let plan = frame.create_physical_plan().await.unwrap();
        let got = rows(
            &datafusion::physical_plan::collect(Arc::clone(&plan), task)
                .await
                .unwrap(),
        );
        let shown = DisplayableExecutionPlan::with_metrics(plan.as_ref())
            .indent(true)
            .to_string();
        let is_pipeline = shown.contains("softmax(") || shown.contains("weights(");
        // A nullable key may keep detection from proving the plan a
        // pipeline; it must still not find one in a look-alike.
        let nullable = matches!(damage, Damage::NullKey(_));
        match (expect, is_pipeline) {
            (Expect::Pipeline, false) if !nullable => fail(format!("found no pipeline:\n{shown}")),
            (Expect::Decline, true) => fail(format!("found a pipeline in a look-alike:\n{shown}")),
            _ => {}
        }
        found += usize::from(is_pipeline);
        let ran = shown.contains("fused_pipelines=1");
        fused += usize::from(ran);
        if is_pipeline && complete && !ran {
            fail(format!(
                "the fused algorithm didn't run on complete tables:\n{shown}"
            ));
        }
        let want = rows(&host.sql(&sql).await.unwrap().collect().await.unwrap());
        if got.len() != want.len() {
            fail(format!("{} rows, DataFusion has {}", got.len(), want.len()));
        }
        for (o, t) in got.iter().zip(&want) {
            let same = (o.0, o.1) == (t.0, t.1)
                && match (o.2, t.2) {
                    (Some(x), Some(y)) if x.is_nan() || y.is_nan() => x.is_nan() && y.is_nan(),
                    (Some(x), Some(y)) if x.is_infinite() || y.is_infinite() => x == y,
                    (Some(x), Some(y)) => (x - y).abs() <= 1e-9 * (1.0 + y.abs()),
                    (None, None) => true,
                    _ => false,
                };
            if !same {
                fail(format!("row {o:?}, DataFusion has {t:?}"));
            }
        }
    }
    if cases >= 50 {
        assert!(found > 0 && fused > 0, "found {found}, fused {fused}");
    }
}

/// The Nadaraya–Watson estimator, a kernel smoother used in regridding, as
/// written for points in two coordinates `x` and `y`: a Gaussian kernel of
/// the distance between each target and each source point, averaging the
/// sources' `v`. It is
/// found as a pipeline, fused, and gives DataFusion's averages.
#[tokio::test]
async fn the_nadaraya_watson_control_fuses() {
    let sql = "WITH kernel AS ( \
                 SELECT t.p, s.q, exp(-(power(t.x - s.x, 2) + power(t.y - s.y, 2)) / 0.25) AS w \
                 FROM target t CROSS JOIN source s) \
               SELECT k.p, SUM(k.w * s.v) / SUM(k.w) AS v \
               FROM kernel k JOIN source s ON k.q = s.q GROUP BY k.p";
    let host = SessionContext::new();
    let builder = SessionStateBuilder::new().with_default_features();
    let ours = SessionContext::new_with_state(einfold_datafusion::install(builder).build());
    let mut rng = Rng::seeded(97, 0);
    let mut unit = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|_| ((rng.next_u64() >> 11) as f64) / ((1u64 << 53) as f64))
            .collect()
    };
    let (targets, sources) = (64, 256);
    let (tx, ty) = (unit(targets), unit(targets));
    let (sx, sy, sv) = (unit(sources), unit(sources), unit(sources));
    let table = |key: &str, n: usize, values: Vec<(&str, &Vec<f64>)>| {
        let mut fields = vec![Field::new(key, DataType::Int64, false)];
        let mut columns: Vec<Arc<dyn Array>> =
            vec![Arc::new(Int64Array::from_iter_values(0..n as i64))];
        for (name, v) in values {
            fields.push(Field::new(name, DataType::Float64, true));
            columns.push(Arc::new(Float64Array::from(v.clone())));
        }
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
        Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap())
    };
    for ctx in [&host, &ours] {
        let target = table("p", targets, vec![("x", &tx), ("y", &ty)]);
        let source = table("q", sources, vec![("x", &sx), ("y", &sy), ("v", &sv)]);
        ctx.register_table("target", target).unwrap();
        ctx.register_table("source", source).unwrap();
    }
    let frame = ours.sql(sql).await.unwrap();
    let task = Arc::new(frame.task_ctx());
    let plan = frame.create_physical_plan().await.unwrap();
    let batches = datafusion::physical_plan::collect(Arc::clone(&plan), task)
        .await
        .unwrap();
    let shown = DisplayableExecutionPlan::with_metrics(plan.as_ref())
        .indent(true)
        .to_string();
    assert!(
        shown.contains("softmax(") && shown.contains("fused_pipelines=1"),
        "not fused:\n{shown}"
    );
    let averages = |batches: &[RecordBatch]| -> Vec<(i64, f64)> {
        let mut out = Vec::new();
        for b in batches {
            let p = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let v = b.column(1).as_any().downcast_ref::<Float64Array>().unwrap();
            out.extend((0..b.num_rows()).map(|r| (p.value(r), v.value(r))));
        }
        out.sort_by_key(|r| r.0);
        out
    };
    let (got, want) = (
        averages(&batches),
        averages(&host.sql(sql).await.unwrap().collect().await.unwrap()),
    );
    assert_eq!(got.len(), targets);
    for ((p, x), (q, y)) in got.iter().zip(&want) {
        assert!(p == q && (x - y).abs() <= 1e-12, "{p}: {x}, DataFusion {y}");
    }
}
