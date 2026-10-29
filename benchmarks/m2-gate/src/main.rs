// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! einfold's second milestone, measured: one head of causal self-attention,
//! written in plain SQL, on DataFusion with einfold off and on, and on
//! DuckDB. The milestone's demonstration is that einfold plans this query as
//! FlashAttention does, tile by tile with an online softmax, without being
//! told it is attention.
//!
//! The tables are `q`, `k` and `v`, each `(t, d, x)`: token, feature, value,
//! with 64 features. The query is the softmax of the scaled scores
//! `q · k / 8` under the causal mask `q.t >= k.t`, averaging `v`, in two
//! spellings: the maximum as a window, and as `GROUP BY` joined back, as ddx,
//! an XQL Systems project for automatic differentiation of SQL queries,
//! writes it.
//!
//! Subcommands, one measurement per process so that peak memory is the
//! process's own (`run.sh` runs them under GNU `time`):
//!
//! - `df <window|groupby> <L> <off|on>`: DataFusion, with a 6 GB memory pool;
//! - `duck-sql <window|groupby> <L>`: print a DuckDB CLI script;
//! - `accuracy <L> <scale>`: einfold and DataFusion against a compensated
//!   reference, with `q` scaled to spread the scores, and einfold's bits on
//!   1, 2, 5 and 12 threads;
//! - `edges`: NaN and infinities in every input, and a row whose scores are
//!   all `-inf`, einfold against DataFusion.

use std::env;
use std::sync::Arc;
use std::time::Instant;

use datafusion::arrow::array::{Array, Float64Array, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::prelude::{SessionConfig, SessionContext};

const D: usize = 64;

/// Values of table `name` at row `r = t·D + d`, as spike S27 drew them.
fn value(name: &str, r: usize) -> f64 {
    let phase = match name {
        "q" => 0.0,
        "k" => 0.5,
        _ => 1.0,
    };
    (r as f64 * 0.7 + phase).sin()
}

fn dense(name: &str, l: usize) -> Vec<f64> {
    (0..l * D).map(|r| value(name, r)).collect()
}

/// The attention query, in `form`'s spelling.
fn sql(form: &str) -> String {
    let scores = "scores AS (SELECT q.t AS i, k.t AS j, SUM(q.x * k.x) / 8.0 AS s \
                  FROM q JOIN k ON q.d = k.d WHERE q.t >= k.t GROUP BY q.t, k.t)";
    let weights = match form {
        "window" => {
            "weights AS (SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS w FROM scores)"
        }
        "groupby" => {
            "m AS (SELECT i, MAX(s) AS m FROM scores GROUP BY i), \
             weights AS (SELECT scores.i, scores.j, exp(scores.s - m.m) AS w \
             FROM scores JOIN m ON scores.i = m.i)"
        }
        f => panic!("unknown form {f}"),
    };
    format!(
        "WITH {scores}, {weights}, norm AS (SELECT i, SUM(w) AS z FROM weights GROUP BY i) \
         SELECT weights.i AS t, v.d, SUM(weights.w * v.x) / norm.z AS x \
         FROM weights JOIN v ON weights.j = v.t JOIN norm ON weights.i = norm.i \
         GROUP BY weights.i, v.d, norm.z"
    )
}

/// A DataFusion session, with einfold or not, on `threads` threads, with a
/// 6 GB memory pool, and the tables.
fn context(l: usize, tables: &[(&str, Vec<f64>)], einfold: bool, threads: usize) -> SessionContext {
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_limit(6 << 30, 1.0)
        .with_max_temp_directory_size(4 << 30)
        .build_arc()
        .unwrap();
    let builder = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(threads))
        .with_runtime_env(runtime);
    let builder = match einfold {
        true => einfold_datafusion::install(builder),
        false => builder,
    };
    let ctx = SessionContext::new_with_state(builder.build());
    let schema = Arc::new(Schema::new(vec![
        Field::new("t", DataType::Int64, false),
        Field::new("d", DataType::Int64, false),
        Field::new("x", DataType::Float64, true),
    ]));
    for (name, vals) in tables {
        let n = l * D;
        let mut parts = vec![Vec::new(); 12];
        for (i, start) in (0..n).step_by(8192).enumerate() {
            let end = (start + 8192).min(n);
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from_iter_values(
                        (start..end).map(|r| (r / D) as i64),
                    )),
                    Arc::new(Int64Array::from_iter_values(
                        (start..end).map(|r| (r % D) as i64),
                    )),
                    Arc::new(Float64Array::from(vals[start..end].to_vec())),
                ],
            )
            .unwrap();
            parts[i % 12].push(batch);
        }
        let table = MemTable::try_new(Arc::clone(&schema), parts).unwrap();
        ctx.register_table(*name, Arc::new(table)).unwrap();
    }
    ctx
}

/// The query's rows, `x[t·D + d]`, NaN where a row is missing or NULL, and
/// what `EXPLAIN ANALYZE` shows of the plan.
async fn run(ctx: &SessionContext, form: &str, l: usize) -> (Vec<f64>, usize, String) {
    let frame = ctx.sql(&sql(form)).await.unwrap();
    let task = Arc::new(frame.task_ctx());
    let plan = frame.create_physical_plan().await.unwrap();
    let batches = datafusion::physical_plan::collect(Arc::clone(&plan), task)
        .await
        .unwrap();
    let mut out = vec![f64::NAN; l * D];
    let mut rows = 0;
    for b in &batches {
        let t = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let d = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let x = b.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        for r in 0..b.num_rows() {
            rows += 1;
            if x.is_valid(r) {
                out[t.value(r) as usize * D + d.value(r) as usize] = x.value(r);
            }
        }
    }
    let shown = DisplayableExecutionPlan::with_metrics(plan.as_ref())
        .indent(true)
        .to_string();
    (out, rows, shown)
}

/// A metric's value from `EXPLAIN ANALYZE`'s text, such as `tiles_visited`,
/// as shown: DataFusion abbreviates large counts, as in `2.08 K`.
fn metric<'a>(shown: &'a str, name: &str) -> Option<&'a str> {
    let at = shown.find(&format!(" {name}="))? + name.len() + 2;
    shown[at..].split([',', ']']).next()
}

/// The pairs of 64 × 64 tiles that the causal mask leaves for `L` tokens:
/// each query tile visits the key tiles up to its own.
fn causal_tiles(l: usize) -> (usize, usize) {
    let tiles = l.div_ceil(64);
    (tiles * (tiles + 1) / 2, tiles * tiles)
}

/// Neumaier's compensated sum.
fn csum(xs: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for x in xs {
        let t = s + x;
        c += if s.abs() >= x.abs() {
            (s - t) + x
        } else {
            (x - t) + s
        };
        s = t;
    }
    s + c
}

/// The SQL's arithmetic with compensated sums: the row maximum, then the
/// weights, the normalizer and the weighted sums.
fn reference(q: &[f64], k: &[f64], v: &[f64], l: usize) -> Vec<f64> {
    let mut out = vec![0.0; l * D];
    for i in 0..l {
        let s: Vec<f64> = (0..=i)
            .map(|j| csum((0..D).map(|e| q[i * D + e] * k[j * D + e])) / 8.0)
            .collect();
        let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
        let z = csum(w.iter().copied());
        for e in 0..D {
            out[i * D + e] = csum(w.iter().enumerate().map(|(j, wj)| wj * v[j * D + e])) / z;
        }
    }
    out
}

fn arg(i: usize) -> String {
    env::args().nth(i).unwrap_or_default()
}

fn checksum(xs: &[f64]) -> f64 {
    xs.iter().map(|x| x.abs()).sum()
}

#[tokio::main]
async fn main() {
    match arg(1).as_str() {
        "df" => {
            let (form, l, on) = (arg(2), arg(3).parse::<usize>().unwrap(), arg(4) == "on");
            let tables: Vec<(&str, Vec<f64>)> =
                ["q", "k", "v"].iter().map(|n| (*n, dense(n, l))).collect();
            let ctx = context(l, &tables, on, 12);
            let t = Instant::now();
            let (out, rows, shown) = run(&ctx, &form, l).await;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let tiles = match (metric(&shown, "tiles_visited"), metric(&shown, "tiles")) {
                (Some(v), Some(a)) => {
                    let (want, all) = causal_tiles(l);
                    format!(" tiles_visited={v} of {a} (causal: {want} of {all})")
                }
                _ => String::new(),
            };
            let fused = shown.contains("fused_pipelines=1");
            println!(
                "ms={ms:.1} rows={rows} checksum={:.9e} fused={fused}{tiles}",
                checksum(&out)
            );
        }
        "duck-sql" => {
            let (form, l) = (arg(2), arg(3).parse::<usize>().unwrap());
            println!("SET threads = 12;\nSET memory_limit = '6GB';\nSET max_temp_directory_size = '4GB';");
            for (i, name) in ["q", "k", "v"].iter().enumerate() {
                let phase = [0.0, 0.5, 1.0][i];
                println!(
                    "CREATE TABLE {name} AS SELECT CAST(r // {D} AS BIGINT) AS t, CAST(r % {D} AS BIGINT) AS d, \
                     sin(CAST(r AS DOUBLE) * 0.7 + {phase}) AS x FROM range(0, {}) tbl(r);",
                    l * D
                );
            }
            println!(
                ".timer on\nCREATE TEMP TABLE out AS {};\n.timer off",
                sql(&form)
            );
            println!("SELECT count(*), sum(abs(x)) FROM out;");
        }
        "accuracy" => {
            let l = arg(2).parse::<usize>().unwrap();
            let qs = arg(3).parse::<f64>().unwrap_or(1.0);
            let q: Vec<f64> = dense("q", l).iter().map(|x| x * qs).collect();
            let (k, v) = (dense("k", l), dense("v", l));
            let truth = reference(&q, &k, &v, l);
            let scale = truth.iter().fold(0.0f64, |a, x| a.max(x.abs()));
            let tables = [("q", q.clone()), ("k", k.clone()), ("v", v.clone())];
            let err = |xs: &[f64]| {
                xs.iter()
                    .zip(&truth)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max)
                    / scale
            };
            let (host, _, _) = run(&context(l, &tables, false, 12), "window", l).await;
            let mut bits = Vec::new();
            let mut ours = Vec::new();
            for threads in [1, 2, 5, 12] {
                let (out, _, shown) = run(&context(l, &tables, true, threads), "window", l).await;
                assert!(shown.contains("fused_pipelines=1"), "{shown}");
                bits.push(out.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
                ours = out;
            }
            let same_bits = bits.windows(2).all(|w| w[0] == w[1]);
            println!(
                "L={l} q_scale={qs} max|out|={scale:.3e} einfold_err={:.2e} datafusion_err={:.2e} \
                 einfold_same_bits_1_2_5_12_threads={same_bits}",
                err(&ours),
                err(&host)
            );
        }
        "edges" => {
            let l = 128;
            let cases: [(&str, &str, usize, f64); 7] = [
                ("none", "q", 0, 0.0),
                ("NaN in q", "q", 70 * D + 3, f64::NAN),
                ("NaN in v", "v", 5 * D + 1, f64::NAN),
                ("+inf in q", "q", 90 * D + 2, f64::INFINITY),
                ("-inf in q", "q", 100 * D + 2, f64::NEG_INFINITY),
                ("1e300 in k", "k", 3 * D + 4, 1e300),
                ("row 100's scores all -inf", "q", 100 * D, f64::NEG_INFINITY),
            ];
            println!(
                "| case | rows | NaN in DataFusion | NaN in einfold | rows that differ | fused |"
            );
            println!("|---|---:|---:|---:|---:|---|");
            for (label, table, at, x) in cases {
                let mut t: Vec<(&str, Vec<f64>)> =
                    ["q", "k", "v"].iter().map(|n| (*n, dense(n, l))).collect();
                if label != "none" {
                    t.iter_mut().find(|(n, _)| *n == table).unwrap().1[at] = x;
                }
                if label.starts_with("row 100") {
                    // k(j, 0) = 1 for every j, so every score of row 100 is -inf.
                    for j in 0..l {
                        t[1].1[j * D] = 1.0;
                    }
                }
                let (host, rows, _) = run(&context(l, &t, false, 4), "window", l).await;
                let (ours, _, shown) = run(&context(l, &t, true, 4), "window", l).await;
                let same = |a: f64, b: f64| {
                    (a.is_nan() && b.is_nan()) || a == b || (a - b).abs() <= 1e-12 * (1.0 + b.abs())
                };
                let differ = ours
                    .iter()
                    .zip(&host)
                    .filter(|(a, b)| !same(**a, **b))
                    .count();
                let nan = |xs: &[f64]| xs.iter().filter(|x| x.is_nan()).count();
                println!(
                    "| {label} | {rows} | {} | {} | {differ} | {} |",
                    nan(&host),
                    nan(&ours),
                    shown.contains("fused_pipelines=1")
                );
            }
        }
        c => panic!("unknown subcommand {c:?}"),
    }
}
