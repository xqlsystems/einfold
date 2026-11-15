// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Property: a convolution written in SQL, a `SUM` of products grouped by
//! keys that shift one table's key by the other's, answers with einfold as
//! without. einfold computes it by its stencil algorithm, one dense product
//! per tap of the kernel, and runs the host's plan wherever that doesn't
//! apply.
//!
//! Each case draws an image `x(c, i, j, v)`, sometimes with a batch key `n`,
//! and a kernel `w(o, c, p, q, v)`, and the query
//!
//! ```sql
//! SELECT * FROM (
//!   SELECT w.o, <shift of x.i by w.p> AS i, <x.j, or its shift by w.q> AS j,
//!          SUM(x.v * w.v) AS v
//!   FROM x JOIN w ON x.c = w.c GROUP BY ...) s
//! WHERE <bounds on s.i, or none>
//! ```
//!
//! where a shift is `a·x.i + b·w.p + k` with random coefficients and
//! constant, in one of several spellings such as `x.i - (w.p - 1) * 2`, and
//! the shift of `x.j` may wrap, `% m`, as longitude does on a globe. The
//! bounds, as zero padding gives, are pushed into the join by DataFusion.
//! The batch key, when there is one, is in the image only, or in both tables
//! as a join key.
//!
//! Tables have `BIGINT` keys and `DOUBLE` values that are small integers, so
//! every sum is exact in any order. Some tables are complete, so that the
//! stencil algorithm runs; others have missing rows, duplicate key tuples,
//! NULL or negative keys, NULL values, NaN or infinities, which it must leave
//! to the host, or, for holes in the image, take where the kernel is finite.
//!
//! Invariants: einfold finds the fold in every case, its groups are the
//! host's, and each group's value has the host's bits, NULL, NaN and the
//! sign of zero included. And coverage: the stencil algorithm runs in some
//! cases, and the host's plan in others.
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

/// A table: its key columns' names, and its rows (keys, then the value).
struct Table {
    keys: Vec<&'static str>,
    rows: Vec<(Vec<Option<i64>>, Option<f64>)>,
}

/// One case: its two tables, the query, and the number of threads.
struct Case {
    image: Table,
    kernel: Table,
    sql: String,
    threads: usize,
}

/// A shift of `x.<key>` by `w.<by>` in a random spelling, with the extents
/// of the two keys: `a·key + b·by + k`.
fn shift(rng: &mut Rng, key: &str, by: &str) -> String {
    match rng.below(4) {
        // A dilated convolution's, as Samudra's U-Net writes it.
        0 => format!("x.{key} - (w.{by} - 1) * {}", 1 + rng.below(3)),
        // A correlation's.
        1 => format!("x.{key} + w.{by}"),
        // A transposed or strided one's, with any coefficients.
        2 => {
            let a = [1, 2, -1][rng.below(3)];
            let b = [1, -1, 2, -3][rng.below(4)];
            let k = rng.below(7) as i64 - 3;
            format!("{a} * x.{key} + w.{by} * {b} + {k}")
        }
        _ => format!("x.{key} - w.{by}"),
    }
}

fn draw(seed: u64) -> Case {
    let mut rng = Rng::seeded(seed, 0x57e9_c11e);
    let mut extent: BTreeMap<&str, i64> = BTreeMap::new();
    for (k, most) in [
        ("n", 3),
        ("c", 4),
        ("o", 4),
        ("i", 7),
        ("j", 7),
        ("p", 4),
        ("q", 4),
    ] {
        extent.insert(k, 1 + rng.below(most) as i64);
    }
    // The batch key: none, in the image only, or in both as a join key.
    let batch = rng.below(4);
    let mut image_keys = vec!["c", "i", "j"];
    let mut kernel_keys = vec!["o", "c", "p", "q"];
    let mut on = vec!["x.c = w.c"];
    let mut select = vec!["w.o AS o".to_string()];
    match batch {
        1 => {
            image_keys.insert(0, "n");
            select.push("x.n AS n".into());
        }
        2 => {
            image_keys.insert(0, "n");
            kernel_keys.insert(0, "n");
            on.push("x.n = w.n");
            select.push("x.n AS n".into());
        }
        _ => {}
    }
    // Latitude-like `i` is always shifted; longitude-like `j` is shifted,
    // wrapping or not, or a plain key, with the kernel then one-dimensional.
    select.push(format!("{} AS i", shift(&mut rng, "i", "p")));
    let j = match rng.below(3) {
        0 => {
            kernel_keys.retain(|&k| k != "q");
            "x.j".to_string()
        }
        1 => shift(&mut rng, "j", "q"),
        _ => {
            let m = extent["j"];
            match rng.chance(0.5) {
                true => format!("({} + {m}) % {m}", shift(&mut rng, "j", "q")),
                false => format!("({}) % {m}", shift(&mut rng, "j", "q")),
            }
        }
    };
    select.push(format!("{j} AS j"));
    let groups: Vec<String> = (1..=select.len()).map(|g| g.to_string()).collect();
    let inner = format!(
        "SELECT {}, SUM(x.v * w.v) AS v FROM x JOIN w ON {} GROUP BY {}",
        select.join(", "),
        on.join(" AND "),
        groups.join(", ")
    );
    let top = extent["i"] - 1;
    let filter = match rng.below(4) {
        0 => String::new(),
        1 => format!(" WHERE s.i BETWEEN 0 AND {top}"),
        2 => format!(" WHERE s.i >= {}", rng.below(3) as i64 - 1),
        _ => format!(" WHERE s.i < {top} AND s.i > -2"),
    };
    let sql = format!("SELECT * FROM ({inner}) s{filter}");
    let (image, kernel) = (
        table(&mut rng, image_keys, &extent),
        table(&mut rng, kernel_keys, &extent),
    );
    Case {
        image,
        kernel,
        sql,
        threads: 1 + rng.below(4),
    }
}

/// A table over `keys`, each of its extent: complete, or with missing rows,
/// duplicate tuples, and the occasional NULL or negative key; and with the
/// occasional NULL value or special value.
fn table(rng: &mut Rng, keys: Vec<&'static str>, extent: &BTreeMap<&str, i64>) -> Table {
    let complete = rng.chance(0.6);
    let special = rng.chance(0.15);
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
                _ => Some(rng.below(9) as f64 - 4.0),
            };
            rows.push((key, value));
        }
    }
    rng.shuffle(&mut rows);
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

/// Each row's group, its key columns as text, and its value's bits, or
/// `None` for NULL, as a multiset. Every NaN is alike.
fn groups(batches: &[RecordBatch]) -> BTreeMap<String, Vec<Option<u64>>> {
    let mut out: BTreeMap<String, Vec<Option<u64>>> = BTreeMap::new();
    for b in batches {
        let value = b.num_columns() - 1;
        for r in 0..b.num_rows() {
            let key: Vec<String> = (0..value)
                .map(|c| {
                    let keys = b.column(c).as_any().downcast_ref::<Int64Array>().unwrap();
                    match keys.is_valid(r) {
                        true => keys.value(r).to_string(),
                        false => "NULL".into(),
                    }
                })
                .collect();
            let v = b
                .column(value)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            let bits = v.is_valid(r).then(|| match v.value(r) {
                x if x.is_nan() => f64::NAN.to_bits(),
                x => x.to_bits(),
            });
            out.entry(key.join("|")).or_default().push(bits);
        }
    }
    for v in out.values_mut() {
        v.sort_unstable();
    }
    out
}

/// What ran: whether the stencil algorithm did, and whether the host's plan.
#[derive(Default)]
struct Ran {
    stencil: bool,
    host: bool,
}

/// One case: `None` if it holds, or what failed.
async fn check(seed: u64) -> (Option<String>, Ran) {
    let case = draw(seed);
    let (plain, ours) = (
        session(case.threads, None),
        session(case.threads, Some(DetectFolds::default())),
    );
    for ctx in [&plain, &ours] {
        register(ctx, "x", &case.image, 1 + (seed as usize % 3));
        register(ctx, "w", &case.kernel, 1 + (seed as usize / 3 % 2));
    }
    let frame = ours.sql(&case.sql).await.unwrap();
    let task = Arc::new(frame.task_ctx());
    let plan = frame.create_physical_plan().await.unwrap();
    let got = datafusion::physical_plan::collect(Arc::clone(&plan), task).await;
    let want = plain.sql(&case.sql).await.unwrap().collect().await;
    let shown = DisplayableExecutionPlan::with_metrics(plan.as_ref())
        .indent(true)
        .to_string();
    // A shifted output shows as its key's sum, in parentheses.
    let node = (shown.lines()).find(|l| l.contains("EinFoldExec") && l.contains("->"));
    let ran = Ran {
        stencil: node.is_some_and(|l| l.contains("dense_folds=1")),
        host: node.is_some_and(|l| l.contains("host_plans=1")),
    };
    let fail = |what: String| {
        Some(format!(
            "seed {seed}: {what}\n  threads={}\n  SQL: {}\n{shown}",
            case.threads, case.sql
        ))
    };
    if !node.is_some_and(|l| l.contains('(')) {
        return (fail("the shifted fold wasn't found".into()), ran);
    }
    let (got, want) = match (got, want) {
        (Ok(g), Ok(w)) => (g, w),
        (g, w) => {
            let (g, w) = (g.err(), w.err());
            return (fail(format!("einfold {g:?}, the host {w:?}")), ran);
        }
    };
    let (got, want) = (groups(&got), groups(&want));
    if got != want {
        let differ = (want.iter())
            .filter(|(k, w)| got.get(*k) != Some(w))
            .map(|(k, w)| format!("{k}: host {w:?}, einfold {:?}", got.get(k)))
            .chain(
                (got.keys())
                    .filter(|k| !want.contains_key(*k))
                    .map(|k| format!("{k}: einfold only")),
            )
            .take(5)
            .collect::<Vec<_>>();
        return (fail(format!("the rows differ: {differ:?}")), ran);
    }
    (None, ran)
}

#[tokio::test]
async fn convolutions_answer_as_the_host_does() {
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
    let (mut failures, mut stencil, mut host) = (Vec::new(), 0, 0);
    for seed in start..start + cases {
        let (failed, ran) = check(seed).await;
        failures.extend(failed);
        stencil += usize::from(ran.stencil);
        host += usize::from(ran.host);
    }
    eprintln!("{cases} cases: the stencil algorithm ran in {stencil}, the host's plan in {host}");
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
            stencil * 3 >= cases as usize,
            "the stencil algorithm ran in {stencil} cases"
        );
        assert!(
            host * 10 >= cases as usize,
            "the host's plan ran in {host} cases"
        );
    }
}
