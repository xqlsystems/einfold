// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! einfold's first milestone, timed: ddx's `matmul` gradient program, end to
//! end, with einfold on against off, on the same DataFusion. The milestone is
//! met when einfold makes it at least 4× faster at `n = 50,000`. ddx is an XQL
//! Systems project for automatic differentiation of SQL queries; the program
//! is the one its benchmark `crates/ddx-datafusion/tests/ad_perf.rs` builds,
//! for `SUM(tanh(a·w)²)` with `a` of `n × 16` and `w` of `16 × 8`.
//!
//! It runs on Tokio's multi-threaded runtime, as a host does: JAX on CPU uses
//! every core too. It also checks the gate's other two clauses:
//!
//! - einfold on and off give the same gradients, within how far two correct
//!   evaluations can differ when each adds its sums in its own order (see
//!   `bounds`);
//! - with einfold on, one thread and every thread give the same gradients,
//!   bit for bit.
//!
//! It prints each step's time with einfold on and off.
//!
//! Run with `cargo run --release [n …]`; it needs `protoc` for ddx.

use std::time::Instant;

use datafusion::arrow::array::{Array, Float64Array, Int64Array};
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use ddx_datafusion::ad::{self, ColumnRef};
use ddx_datafusion::engine;

/// The fastest of five runs of `run`, in milliseconds.
async fn best_ms(mut run: impl AsyncFnMut()) -> f64 {
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        run().await;
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    best
}

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

/// One gradient table's rows: its two keys and its value, in key order.
type Gradient = Vec<(i64, i64, Option<f64>)>;

/// `a` (`n × d`) and `w` (`d × h`) as dense row-major matrices, from their
/// complete `(key, key, val)` tables.
async fn dense(ctx: &SessionContext, table: &str, rows: usize, columns: usize) -> Vec<f64> {
    let sql = format!("SELECT * FROM {table}");
    let mut m = vec![0.0; rows * columns];
    for b in ctx.sql(&sql).await.unwrap().collect().await.unwrap() {
        let key = |c: usize| {
            b.column(c)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .clone()
        };
        let (i, j) = (key(0), key(1));
        let v = b.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        for r in 0..b.num_rows() {
            m[i.value(r) as usize * columns + j.value(r) as usize] = v.value(r);
        }
    }
    m
}

/// How far two correct evaluations of the program's gradients can differ,
/// for each gradient value, given that each evaluates every sum in some
/// order. With `u = 2⁻⁵³`, two orders of a sum of `n` terms differ by at most
/// `2·(n − 1)·u·Σ|terms|`. The program computes:
///
/// - `z[s, o] = Σ_k a[s, k]·w[k, o]`, a sum of `d` terms;
/// - the loss's cotangent `c = 2·tanh(z)·(1 − tanh(z)²)`, whose derivative
///   is at most 2 in magnitude, so `|Δc| ≤ 2·|Δz|`, plus the rounding of its
///   own few operations, which both runs perform on their own `z`
///   (`16·u` is generous);
/// - `W̄[k, o] = Σ_s a[s, k]·c[s, o]` and `Ā[s, k] = Σ_o c[s, o]·w[k, o]`,
///   sums whose terms each carry `c`'s difference and whose order may differ.
///
/// Returns the bounds for `W̄`, then `Ā`, in key order.
fn bounds(a: &[f64], w: &[f64], (n, d, h): (usize, usize, usize)) -> [Vec<f64>; 2] {
    let u = f64::EPSILON / 2.0;
    let order = |terms: usize, magnitude: f64| 2.0 * (terms as f64 - 1.0).max(0.0) * u * magnitude;
    let mut c = vec![0.0; n * h];
    let mut dc = vec![0.0; n * h];
    for s in 0..n {
        for o in 0..h {
            let (mut z, mut mag) = (0.0, 0.0);
            for k in 0..d {
                z += a[s * d + k] * w[k * h + o];
                mag += (a[s * d + k] * w[k * h + o]).abs();
            }
            let t = z.tanh();
            c[s * h + o] = 2.0 * t * (1.0 - t * t);
            dc[s * h + o] = 2.0 * order(d, mag) + 16.0 * u;
        }
    }
    let mut w_bar = vec![0.0; d * h];
    for k in 0..d {
        for o in 0..h {
            let (mut carried, mut mag) = (0.0, 0.0);
            for s in 0..n {
                carried += a[s * d + k].abs() * dc[s * h + o];
                mag += (a[s * d + k] * c[s * h + o]).abs();
            }
            w_bar[k * h + o] = carried + order(n, mag);
        }
    }
    let mut a_bar = vec![0.0; n * d];
    for s in 0..n {
        for k in 0..d {
            let (mut carried, mut mag) = (0.0, 0.0);
            for o in 0..h {
                carried += w[k * h + o].abs() * dc[s * h + o];
                mag += (c[s * h + o] * w[k * h + o]).abs();
            }
            a_bar[s * d + k] = carried + order(h, mag);
        }
    }
    [w_bar, a_bar]
}

/// What one run measured: the program's time end to end, each check's and
/// step's time, its two gradients (`W̄`, then `Ā`), and the tables `a` and
/// `w`, dense.
struct Measured {
    run: f64,
    steps: Vec<(String, f64)>,
    gradients: Vec<Gradient>,
    a: Vec<f64>,
    w: Vec<f64>,
}

/// ddx's `matmul` tables and gradient program, run and timed on `ctx`.
async fn measure(ctx: SessionContext, n: usize) -> Measured {
    let (d, h) = (16, 8);
    exec(&ctx, &format!("CREATE TABLE a AS SELECT CAST(r / {d} AS BIGINT) AS s, CAST(r % {d} AS BIGINT) AS k, sin(CAST(r AS DOUBLE)) AS val FROM (SELECT unnest(range(0, {})) AS r)", n * d)).await;
    exec(&ctx, &format!("CREATE TABLE w AS SELECT CAST(r / {h} AS BIGINT) AS k, CAST(r % {h} AS BIGINT) AS o, 0.1 * cos(CAST(r AS DOUBLE)) AS val FROM (SELECT unnest(range(0, {})) AS r)", d * h)).await;
    let loss = "WITH c AS (SELECT a.s, w.o, SUM(a.val * w.val) AS z FROM a JOIN w ON a.k = w.k \
                GROUP BY a.s, w.o) SELECT SUM(tanh(z) * tanh(z)) AS l FROM c";
    let wrt = [ColumnRef::new("w", "val"), ColumnRef::new("a", "val")];
    let program = ad::grad(&ctx, loss, &wrt).await.unwrap();
    engine::execute(&ctx, &program).await.unwrap();
    let run = best_ms(async || {
        engine::execute(&ctx, &program).await.unwrap();
    })
    .await;
    let mut steps = Vec::new();
    // ddx's checks run before the steps: plans that must return no rows,
    // such as one that each table's keys identify its rows.
    for (i, check) in program.checks.iter().enumerate() {
        let ms = best_ms(async || {
            engine::returns_rows(&ctx, &check.plan).await.unwrap();
        })
        .await;
        steps.push((format!("check {i}"), ms));
    }
    for step in program.steps() {
        let ms = best_ms(async || {
            engine::run_step(&ctx, step).await.unwrap();
        })
        .await;
        let short = step
            .name
            .splitn(4, '_')
            .last()
            .unwrap_or(&step.name)
            .to_string();
        steps.push((short, ms));
    }
    let mut gradients = Vec::new();
    for step in program.steps().filter(|s| s.name.contains("_grad_")) {
        let sql = format!("SELECT * FROM {} ORDER BY 1, 2", step.name);
        let batches = ctx.sql(&sql).await.unwrap().collect().await.unwrap();
        let mut rows = Vec::new();
        for b in &batches {
            let key = |c: usize| {
                b.column(c)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .clone()
            };
            let (i, j) = (key(0), key(1));
            let v = b.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
            rows.extend(
                (0..b.num_rows())
                    .map(|r| (i.value(r), j.value(r), v.is_valid(r).then(|| v.value(r)))),
            );
        }
        gradients.push(rows);
    }
    // W̄ has d·h rows, Ā has n·d.
    gradients.sort_by_key(|g| g.len());
    let (a, w) = (dense(&ctx, "a", n, d).await, dense(&ctx, "w", d, h).await);
    Measured {
        run,
        steps,
        gradients,
        a,
        w,
    }
}

#[tokio::main]
async fn main() {
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let sizes = if sizes.is_empty() {
        vec![10_000, 50_000]
    } else {
        sizes
    };
    println!(
        "Threads: {}. DataFusion {}.",
        std::thread::available_parallelism().unwrap(),
        datafusion::DATAFUSION_VERSION
    );
    for n in sizes {
        let off = measure(SessionContext::new(), n).await;
        let einfold = |config: SessionConfig| {
            let builder = SessionStateBuilder::new()
                .with_default_features()
                .with_config(config);
            SessionContext::new_with_state(einfold_datafusion::install(builder).build())
        };
        let on = measure(einfold(SessionConfig::new()), n).await;
        let one_thread = measure(einfold(SessionConfig::new().with_target_partitions(1)), n).await;
        // The same gradients, within their bounds.
        let bound = bounds(&on.a, &on.w, (n, 16, 8));
        for ((g_off, g_on), (bound, width)) in off
            .gradients
            .iter()
            .zip(&on.gradients)
            .zip(bound.iter().zip([8, 16]))
        {
            assert_eq!(g_off.len(), g_on.len());
            for (&(i, j, x), &(i2, j2, y)) in g_off.iter().zip(g_on) {
                assert_eq!((i, j), (i2, j2));
                let b = bound[i as usize * width + j as usize];
                match (x, y) {
                    (Some(x), Some(y)) => {
                        assert!((x - y).abs() <= b, "({i}, {j}): {x} vs {y}, bound {b:e}")
                    }
                    (None, None) => {}
                    _ => panic!("NULL mismatch at ({i}, {j})"),
                }
            }
        }
        // The same bits on one thread as on every thread.
        let bits = |g: &[Gradient]| -> Vec<Vec<(i64, i64, Option<u64>)>> {
            g.iter()
                .map(|g| {
                    g.iter()
                        .map(|&(i, j, v)| (i, j, v.map(f64::to_bits)))
                        .collect()
                })
                .collect()
        };
        assert!(
            bits(&on.gradients) == bits(&one_thread.gradients),
            "one thread changed the gradients' bits"
        );
        let (off, on) = ((off.run, off.steps), (on.run, on.steps));
        println!("\n### matmul n = {n}\n");
        println!("| step | einfold off, ms | einfold on, ms |\n|---|---:|---:|");
        for ((name, a), (_, b)) in off.1.iter().zip(&on.1) {
            println!("| `{name}` | {a:.1} | {b:.1} |");
        }
        println!(
            "| **whole program** | **{:.1}** | **{:.1}** |\n\nSpeedup: {:.1}×. Gradients agree within their bounds, \
             and einfold's are the same bits on one thread.",
            off.0,
            on.0,
            off.0 / on.0
        );
    }
}
