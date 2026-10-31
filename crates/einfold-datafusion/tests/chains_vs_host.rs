// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Property: a `SUM` of the product of three factors, each from one of three
//! inner-joined tables, such as a chain of matrix products, answers with
//! einfold as without. einfold computes it as two folds of two operands, in
//! the order of contractions it chooses from the tables' extents, and runs
//! the host's plan wherever that doesn't apply.
//!
//! Each case draws one of five shapes over tables `a`, `b` and `c`:
//!
//! - a chain, `a(i, j)·b(j, k)·c(k, l)`, grouped by `i` and `l`;
//! - a matrix times a matrix times a vector, `a(i, j)·b(j, k)·c(k)`;
//! - a batch of chains, every table also keyed by `h`, which is grouped by;
//! - a star, `a(i, j)·b(j, k)·c(j, l)`, where all three share `j`;
//! - a chain grouped also by a joined key, `j`.
//!
//! Tables have `BIGINT` keys, and `DOUBLE` values that are small integers
//! (every sum is then exact in any order, so results compare bit for bit) or
//! of random magnitudes (results then agree within the bound below). Some
//! tables are complete, so that the dense algorithm runs; others have
//! missing rows, duplicate key tuples, NULL or negative keys, NULL values,
//! NaN, infinities, or values large enough that sums could overflow, which
//! the chain must leave to the host. Each query writes its joins in a random
//! order and its product in a random order and parenthesization.
//!
//! Invariants: the same groups as the host; the same NULLs, NaNs,
//! infinities and zeros, signs included; with small integers, the same bits;
//! otherwise values within `2·(n + 2)·u·Σ|terms|` of the host's, where `n` is
//! the group's number of terms and `u` the unit roundoff: each result is
//! within `(n + 2)·u·Σ|terms|` of the exact sum, from `n − 1` roundings of
//! additions and three of products. And coverage: the chain runs by the dense
//! algorithm in some cases, and the plan behind it in others: half the cases
//! keep the host's plan behind every fold, and half einfold's default, under
//! which a declined chain runs the product as a fold of two operands, one of
//! them a join, by einfold's kernels.
//!
//! Reproduce a failure with `EINFOLD_SEED=<seed printed>`; search longer with
//! `EINFOLD_CASES=<n>`.

use std::collections::BTreeMap;
use std::sync::Arc;

use datafusion::arrow::array::{Array, Float64Array, Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::prelude::{SessionConfig, SessionContext};
use einfold_datafusion::{DetectFolds, EinFoldQueryPlanner};
use einfold_testkit::Rng;

/// The unit roundoff of `DOUBLE`.
const U: f64 = f64::EPSILON / 2.0;

/// A table: its key columns' names, and its rows (keys, then the value).
struct Table {
    keys: Vec<&'static str>,
    rows: Vec<(Vec<Option<i64>>, Option<f64>)>,
}

/// How a case draws its values.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Values {
    /// Small integers: every sum is exact.
    Exact,
    /// Random magnitudes.
    Inexact,
}

/// One case: its tables, the query, and the query of each group's `n` and
/// `Σ|terms|`.
struct Case {
    tables: [Table; 3],
    sql: String,
    magnitudes: String,
    values: Values,
    threads: usize,
}

fn draw(seed: u64) -> Case {
    let mut rng = Rng::seeded(seed, 0xc4a1_5eed);
    let values = match rng.chance(0.5) {
        true => Values::Exact,
        false => Values::Inexact,
    };
    // Each shape: the tables' keys, the joins' equalities, and the groups.
    let (keys, on, groups): ([Vec<&'static str>; 3], Vec<&str>, Vec<&str>) = match rng.below(5) {
        0 => (
            [vec!["i", "j"], vec!["j", "k"], vec!["k", "l"]],
            vec!["a.j = b.j", "b.k = c.k"],
            vec!["a.i", "c.l"],
        ),
        1 => (
            [vec!["i", "j"], vec!["j", "k"], vec!["k"]],
            vec!["a.j = b.j", "b.k = c.k"],
            vec!["a.i"],
        ),
        2 => (
            [
                vec!["h", "i", "j"],
                vec!["h", "j", "k"],
                vec!["h", "k", "l"],
            ],
            vec!["a.j = b.j AND a.h = b.h", "b.k = c.k AND b.h = c.h"],
            vec!["a.h", "a.i", "c.l"],
        ),
        3 => (
            [vec!["i", "j"], vec!["j", "k"], vec!["j", "l"]],
            vec!["a.j = b.j", "b.j = c.j"],
            vec!["a.i", "b.k", "c.l"],
        ),
        _ => (
            [vec!["i", "j"], vec!["j", "k"], vec!["k", "l"]],
            vec!["a.j = b.j", "b.k = c.k"],
            vec!["a.i", "b.j", "c.l"],
        ),
    };
    let mut extent: BTreeMap<&str, i64> = BTreeMap::new();
    for k in ["h", "i", "j", "k", "l"] {
        extent.insert(k, 1 + rng.below(6) as i64);
    }
    let tables = keys.map(|keys| table(&mut rng, keys, &extent, values));
    // Sums of large values aren't exact, whatever the others are.
    let large = (tables.iter())
        .flat_map(|t| &t.rows)
        .any(|(_, v)| v.is_some_and(|v| v.is_finite() && v.abs() > 1e6));
    let values = if large { Values::Inexact } else { values };
    // The product, in a random order and parenthesization.
    let mut factors = ["a.v", "b.v", "c.v"];
    for i in (1..3).rev() {
        factors.swap(i, rng.below(i + 1));
    }
    let [x, y, z] = factors;
    let product = match rng.chance(0.5) {
        true => format!("({x} * {y}) * {z}"),
        false => format!("{x} * ({y} * {z})"),
    };
    // The joins, in a random order: `a` with `b` first, or `b` with `c`.
    let from = match rng.chance(0.5) {
        true => format!("a JOIN b ON {} JOIN c ON {}", on[0], on[1]),
        false => format!("b JOIN c ON {} JOIN a ON {}", on[1], on[0]),
    };
    let names: Vec<String> = (0..groups.len()).map(|g| format!("g{g}")).collect();
    let select: Vec<String> = (groups.iter().zip(&names))
        .map(|(g, n)| format!("{g} AS {n}"))
        .collect();
    let by = groups.join(", ");
    let sql = format!(
        "SELECT {}, SUM({product}) AS v FROM {from} GROUP BY {by}",
        select.join(", ")
    );
    let magnitudes = format!(
        "SELECT {}, COUNT({product}) AS n, SUM(abs({product})) AS m FROM {from} GROUP BY {by}",
        select.join(", ")
    );
    Case {
        tables,
        sql,
        magnitudes,
        values,
        threads: 1 + rng.below(4),
    }
}

/// A table over `keys`, each of its extent: complete, or with missing rows
/// and duplicate tuples, and with the occasional NULL or negative key and
/// special value.
fn table(
    rng: &mut Rng,
    keys: Vec<&'static str>,
    extent: &BTreeMap<&str, i64>,
    values: Values,
) -> Table {
    let complete = rng.chance(0.6);
    let special = rng.chance(0.15);
    let large = rng.chance(0.05);
    let mut tuples: Vec<Vec<i64>> = vec![vec![]];
    for k in &keys {
        tuples = (tuples.into_iter())
            .flat_map(|t| {
                (0..extent[k]).map(move |v| {
                    let mut t = t.clone();
                    t.push(v);
                    t
                })
            })
            .collect();
    }
    let mut rows = Vec::new();
    for t in tuples {
        if !complete && rng.chance(0.2) {
            continue;
        }
        let copies = if !complete && rng.chance(0.05) { 2 } else { 1 };
        for _ in 0..copies {
            let mut key: Vec<Option<i64>> = t.iter().map(|&k| Some(k)).collect();
            if !complete && rng.chance(0.03) {
                let c = rng.below(key.len());
                key[c] = match rng.chance(0.5) {
                    true => None,
                    false => Some(-1),
                };
            }
            let value = match () {
                _ if !complete && rng.chance(0.05) => None,
                _ if special && rng.chance(0.2) => {
                    Some([f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.0][rng.below(4)])
                }
                _ if large && rng.chance(0.3) => {
                    Some(1e200 * if rng.chance(0.5) { 1.0 } else { -1.0 })
                }
                _ => Some(match values {
                    Values::Exact => rng.below(9) as f64 - 4.0,
                    Values::Inexact => {
                        (rng.unit() * 2.0 - 1.0) * 10f64.powi(rng.below(7) as i32 - 3)
                    }
                }),
            };
            rows.push((key, value));
        }
    }
    // Shuffled, so that rows don't arrive in key order.
    for i in (1..rows.len()).rev() {
        rows.swap(i, rng.below(i + 1));
    }
    Table { keys, rows }
}

fn register(ctx: &SessionContext, name: &str, table: &Table, partitions: usize) {
    let mut fields: Vec<Field> = (table.keys.iter())
        .map(|k| Field::new(*k, DataType::Int64, true))
        .collect();
    fields.push(Field::new("v", DataType::Float64, true));
    let schema = Arc::new(Schema::new(fields));
    let chunk = table.rows.len().div_ceil(partitions).max(1);
    let mut parts: Vec<Vec<RecordBatch>> = (table.rows.chunks(chunk))
        .map(|rows| {
            let mut columns: Vec<Arc<dyn Array>> = (0..table.keys.len())
                .map(|c| {
                    let keys: Int64Array = rows.iter().map(|(k, _)| k[c]).collect();
                    Arc::new(keys) as Arc<dyn Array>
                })
                .collect();
            let values: Float64Array = rows.iter().map(|(_, v)| *v).collect();
            columns.push(Arc::new(values));
            vec![RecordBatch::try_new(Arc::clone(&schema), columns).unwrap()]
        })
        .collect();
    if parts.is_empty() {
        parts.push(vec![]);
    }
    let mem = MemTable::try_new(schema, parts).unwrap();
    ctx.register_table(name, Arc::new(mem)).unwrap();
}

/// DataFusion, without einfold, or with it under `rule`.
fn session(threads: usize, rule: Option<DetectFolds>) -> SessionContext {
    let config = SessionConfig::new().with_target_partitions(threads);
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(config);
    let state = match rule {
        Some(rule) => state.with_query_planner(Arc::new(EinFoldQueryPlanner::detecting(rule))),
        None => state,
    };
    SessionContext::new_with_state(state.build())
}

/// Each row's group, its first `keys` columns as text, and its column
/// `value`, a `DOUBLE` or a `BIGINT`.
fn groups(
    batches: &[RecordBatch],
    keys: usize,
    value: usize,
) -> BTreeMap<String, Vec<Option<f64>>> {
    let mut out: BTreeMap<String, Vec<Option<f64>>> = BTreeMap::new();
    for b in batches {
        for r in 0..b.num_rows() {
            let key: Vec<String> = (0..keys)
                .map(|c| {
                    let keys = b.column(c).as_any().downcast_ref::<Int64Array>().unwrap();
                    match keys.is_valid(r) {
                        true => keys.value(r).to_string(),
                        false => "NULL".into(),
                    }
                })
                .collect();
            let column = b.column(value);
            let v = match column.as_any().downcast_ref::<Int64Array>() {
                Some(n) => n.is_valid(r).then(|| n.value(r) as f64),
                None => {
                    let v = column.as_any().downcast_ref::<Float64Array>().unwrap();
                    v.is_valid(r).then(|| v.value(r))
                }
            };
            out.entry(key.join("|")).or_default().push(v);
        }
    }
    out
}

/// What ran: whether the chain ran by the dense algorithm, and whether it
/// ran the host's plan.
#[derive(Default)]
struct Ran {
    dense: bool,
    host: bool,
}

/// One case: `None` if it holds, or what failed.
async fn check(seed: u64) -> (Option<String>, Ran) {
    let case = draw(seed);
    // Half the cases with the host's plan behind every fold, and half with
    // einfold's default, where a declined chain runs the product as a fold of
    // two operands by einfold's kernels.
    let rule = match seed % 2 {
        0 => DetectFolds::with_host_fallback(),
        _ => DetectFolds::default(),
    };
    let (plain, ours) = (
        session(case.threads, None),
        session(case.threads, Some(rule)),
    );
    for ctx in [&plain, &ours] {
        for (name, t) in ["a", "b", "c"].into_iter().zip(&case.tables) {
            register(ctx, name, t, 1 + (seed as usize % 3));
        }
    }
    let frame = ours.sql(&case.sql).await.unwrap();
    let task = Arc::new(frame.task_ctx());
    let plan = frame.create_physical_plan().await.unwrap();
    let got = datafusion::physical_plan::collect(Arc::clone(&plan), task).await;
    let want = plain.sql(&case.sql).await.unwrap().collect().await;
    let shown = DisplayableExecutionPlan::with_metrics(plan.as_ref())
        .indent(true)
        .to_string();
    let chain = shown
        .lines()
        .find(|l| l.contains("EinFoldExec") && l.matches(" * ").count() == 2);
    let ran = Ran {
        dense: chain.is_some_and(|l| l.contains("dense_folds=1")),
        host: chain.is_some_and(|l| l.contains("host_plans=1")),
    };
    let fail = |what: String| {
        Some(format!(
            "seed {seed}: {what}\n  {:?} threads={}\n  SQL: {}\n{shown}",
            case.values, case.threads, case.sql
        ))
    };
    if chain.is_none() {
        return (fail("the chain wasn't found".into()), ran);
    }
    let (got, want) = match (got, want) {
        (Ok(g), Ok(w)) => (g, w),
        // Both fail alike, as on an overflowing cast; the query's error
        // isn't einfold's to change.
        (Err(_), Err(_)) => return (None, ran),
        (g, w) => {
            let (g, w) = (g.err(), w.err());
            return (fail(format!("einfold {g:?}, the host {w:?}")), ran);
        }
    };
    let value = got.first().map_or(0, |b| b.num_columns() - 1);
    let (got, want) = (groups(&got, value, value), groups(&want, value, value));
    if got.keys().ne(want.keys()) {
        return (fail("the groups differ".into()), ran);
    }
    let magnitudes = plain
        .sql(&case.magnitudes)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let (counts, sums) = (
        groups(&magnitudes, value, value),
        groups(&magnitudes, value, value + 1),
    );
    for (key, w) in &want {
        let g = &got[key];
        if g.len() != 1 || w.len() != 1 {
            return (fail(format!("group {key} isn't one row")), ran);
        }
        let (g, w) = (g[0], w[0]);
        let n = counts[key][0].unwrap_or(0.0);
        let bound = 2.0 * (n + 2.0) * U * sums[key][0].unwrap_or(0.0);
        let agree = match (g, w) {
            (None, None) => true,
            (Some(g), Some(w)) if g.is_nan() || w.is_nan() => g.is_nan() && w.is_nan(),
            (Some(g), Some(w)) if g == 0.0 && w == 0.0 => g.to_bits() == w.to_bits(),
            (Some(g), Some(w)) if g.is_infinite() || w.is_infinite() => g == w,
            (Some(g), Some(w)) => match case.values {
                Values::Exact => g.to_bits() == w.to_bits(),
                Values::Inexact => (g - w).abs() <= bound,
            },
            _ => false,
        };
        if !agree {
            return (
                fail(format!(
                    "group {key}: einfold {g:?}, the host {w:?}, bound {bound:e}"
                )),
                ran,
            );
        }
    }
    (None, ran)
}

#[tokio::test]
async fn chains_answer_as_the_host_does() {
    let start: u64 = std::env::var("EINFOLD_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let cases: u64 = std::env::var("EINFOLD_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if std::env::var("EINFOLD_SEED").is_ok() {
            1
        } else {
            300
        });
    let (mut failures, mut dense, mut host) = (Vec::new(), 0, 0);
    for seed in start..start + cases {
        let (failed, ran) = check(seed).await;
        failures.extend(failed);
        dense += usize::from(ran.dense);
        host += usize::from(ran.host);
    }
    eprintln!("{cases} cases: the chain ran dense in {dense}, the host's plan in {host}");
    for f in failures.iter().take(3) {
        eprintln!("{f}\n");
    }
    assert!(
        failures.is_empty(),
        "{} of {cases} cases failed",
        failures.len()
    );
    if cases >= 100 {
        assert!(
            dense * 5 >= cases as usize,
            "the dense algorithm ran in {dense} cases"
        );
        assert!(
            host * 20 >= cases as usize,
            "the host's plan ran in {host} cases"
        );
    }
}
