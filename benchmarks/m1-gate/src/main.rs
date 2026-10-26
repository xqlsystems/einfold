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
//! every core too. It also checks that both runs give the same gradients,
//! within float summation's error, and prints each step's time with einfold
//! on and off.
//!
//! Run with `cargo run --release [n …]`; it needs `protoc` for ddx.

use std::time::Instant;

use datafusion::arrow::array::{Array, Float64Array};
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;
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

/// ddx's `matmul` tables, the gradient program, its time end to end, each
/// check's and step's time, and its two gradients' values in key order.
async fn measure(
    ctx: SessionContext,
    n: usize,
) -> (f64, Vec<(String, f64)>, Vec<Vec<Option<f64>>>) {
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
        let mut values = Vec::new();
        for b in &batches {
            let v = b.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
            values.extend((0..b.num_rows()).map(|r| v.is_valid(r).then(|| v.value(r))));
        }
        gradients.push(values);
    }
    (run, steps, gradients)
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
        let builder = SessionStateBuilder::new().with_default_features();
        let on_ctx = SessionContext::new_with_state(einfold_datafusion::install(builder).build());
        let on = measure(on_ctx, n).await;
        // Same gradients, within a relative 1e-12 of the largest value.
        for (g_off, g_on) in off.2.iter().zip(&on.2) {
            assert_eq!(g_off.len(), g_on.len());
            let scale = g_off
                .iter()
                .flatten()
                .fold(0f64, |m, v| m.max(v.abs()))
                .max(1.0);
            for (x, y) in g_off.iter().zip(g_on) {
                match (x, y) {
                    (Some(x), Some(y)) => assert!((x - y).abs() <= 1e-12 * scale, "{x} vs {y}"),
                    (None, None) => {}
                    _ => panic!("NULL mismatch"),
                }
            }
        }
        println!("\n### matmul n = {n}\n");
        println!("| step | einfold off, ms | einfold on, ms |\n|---|---:|---:|");
        for ((name, a), (_, b)) in off.1.iter().zip(&on.1) {
            println!("| `{name}` | {a:.1} | {b:.1} |");
        }
        println!(
            "| **whole program** | **{:.1}** | **{:.1}** |\n\nSpeedup: {:.1}×. Gradients agree.",
            off.0,
            on.0,
            off.0 / on.0
        );
    }
}
