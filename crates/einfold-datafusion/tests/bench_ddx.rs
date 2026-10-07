// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A benchmark of einfold's rewrite: the contraction queries of the `matmul`
//! (matrix multiplication) and `attn` (attention, as in transformers) families
//! of ddx's performance test, forward and backward, with einfold off and on.
//! ddx (github.com/xqlsystems/ddx) is a sibling project that differentiates SQL
//! queries; its test is `crates/ddx-datafusion/tests/ad_perf.rs`, and einfold
//! does not depend on it.
//!
//! Each row of the output table times one query with plain DataFusion ("off")
//! and with einfold's optimizer rule installed ("on"). It also records whether
//! the plan with einfold contains the `EinFoldExec` operator, and whether both
//! runs returned the same result (the harness comparison of `einfold-testkit`:
//! rows as a multiset, floats to a relative 1e-9).
//!
//! The tables and forward queries are ddx's. The backward contractions are
//! written by hand. Each einsum is timed on its own, since the full loss also
//! has non-einsum steps. Run it, and read the markdown table on stdout:
//!
//! ```sh
//! cargo test -p einfold-datafusion --release --test bench_ddx -- --ignored --nocapture
//! ```

use std::sync::Arc;
use std::time::{Duration, Instant};

use datafusion::arrow::compute::concat_batches;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use einfold_datafusion::enable;
use einfold_testkit::compare;

/// Timed runs per query and mode, after one warm-up run.
const RUNS: usize = 5;

/// If the environment variable `BENCH_ONLY` is set, only queries whose label
/// (such as `matmul n=50000 | backward`) contains it are run.
fn selected(label: &str) -> bool {
    std::env::var("BENCH_ONLY").map_or(true, |f| label.contains(&f))
}

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

/// A session with einfold off, and one with it on, sharing every table.
struct Pair {
    off: SessionContext,
    on: SessionContext,
}

impl Pair {
    fn new() -> Self {
        let on_state = enable(SessionStateBuilder::new().with_default_features()).build();
        Pair {
            off: SessionContext::new(),
            on: SessionContext::new_with_state(on_state),
        }
    }

    /// Create table `name` from `select`, in both sessions (the same data).
    async fn table(&self, name: &str, select: &str) {
        exec(&self.off, &format!("CREATE TABLE {name} AS {select}")).await;
        let provider = self.off.table_provider(name).await.unwrap();
        self.on.register_table(name, provider).unwrap();
    }
}

/// `n` rows of a ddx-style table: coordinates `(r / cols, r % cols)` and a value.
fn grid(names: (&str, &str), rows: usize, cols: usize, val: &str) -> String {
    format!(
        "SELECT CAST(r / {cols} AS BIGINT) AS {}, CAST(r % {cols} AS BIGINT) AS {}, {val} AS val \
         FROM (SELECT unnest(range(0, {})) AS r)",
        names.0,
        names.1,
        rows * cols
    )
}

fn proj(w: &str) -> String {
    format!(
        "SELECT x.t, {w}.j, SUM(x.val * {w}.val) AS v FROM x JOIN {w} ON x.k = {w}.k \
         GROUP BY x.t, {w}.j"
    )
}

/// ddx's `contraction(n, d, h)` tables, and the queries to time on them.
async fn matmul(p: &Pair, n: usize, d: usize, h: usize) -> Vec<(String, String)> {
    p.table("a", &grid(("s", "k"), n, d, "sin(CAST(r AS DOUBLE))"))
        .await;
    p.table("w", &grid(("k", "o"), d, h, "0.1 * cos(CAST(r AS DOUBLE))"))
        .await;
    let z = "SELECT a.s, w.o, SUM(a.val * w.val) AS z FROM a JOIN w ON a.k = w.k GROUP BY a.s, w.o";
    p.table(
        "zbar",
        &format!("SELECT s, o, 2.0 * tanh(z) * (1.0 - tanh(z) * tanh(z)) AS val FROM ({z})"),
    )
    .await;
    vec![
        ("forward z = a.w".into(), z.into()),
        (
            "forward AVG(a.val * w.val), not an einsum".into(),
            "SELECT a.s, w.o, AVG(a.val * w.val) AS z FROM a JOIN w ON a.k = w.k \
             GROUP BY a.s, w.o"
                .into(),
        ),
        (
            "backward W-bar = a^T.zbar".into(),
            "SELECT a.k, zbar.o, SUM(a.val * zbar.val) AS v FROM a JOIN zbar ON a.s = zbar.s \
             GROUP BY a.k, zbar.o"
                .into(),
        ),
        (
            "backward A-bar = zbar.w^T".into(),
            "SELECT zbar.s, w.k, SUM(zbar.val * w.val) AS v FROM zbar JOIN w ON zbar.o = w.o \
             GROUP BY zbar.s, w.k"
                .into(),
        ),
    ]
}

/// ddx's `attention(l, d)` tables, and its einsum queries. The softmax steps
/// (`m`, `e`, `z`) are materialized as tables, so each einsum is timed alone.
async fn attention(p: &Pair, l: usize, d: usize) -> Vec<(String, String)> {
    p.table("x", &grid(("t", "k"), l, d, "sin(CAST(r AS DOUBLE))"))
        .await;
    for (name, phase) in [("wq", 0.1), ("wk", 0.2), ("wv", 0.3)] {
        let val = format!("0.1 * cos(CAST(r AS DOUBLE) + {phase})");
        p.table(name, &grid(("k", "j"), d, d, &val)).await;
    }
    for (name, w) in [("q", "wq"), ("k", "wk"), ("vv", "wv")] {
        p.table(name, &proj(w)).await;
    }
    p.table(
        "s",
        &format!(
            "SELECT q.t AS t, k.t AS u, SUM(q.v * k.v) / sqrt({d}.0) AS v FROM q JOIN k \
             ON q.j = k.j GROUP BY q.t, k.t"
        ),
    )
    .await;
    p.table("m", "SELECT t, MAX(v) AS m FROM s GROUP BY t")
        .await;
    p.table(
        "e",
        "SELECT s.t, s.u, exp(s.v - m.m) AS e FROM s JOIN m ON s.t = m.t",
    )
    .await;
    p.table("z", "SELECT t, SUM(e) AS z FROM e GROUP BY t")
        .await;
    p.table(
        "p",
        "SELECT e.t, e.u, e.e / z.z AS v FROM e JOIN z ON e.t = z.t",
    )
    .await;
    vec![
        ("projection q = x.wq".into(), proj("wq")),
        ("projection k = x.wk".into(), proj("wk")),
        ("projection v = x.wv".into(), proj("wv")),
        (
            "scores q.k".into(),
            format!(
                "SELECT q.t AS t, k.t AS u, SUM(q.v * k.v) / sqrt({d}.0) AS v FROM q JOIN k \
                 ON q.j = k.j GROUP BY q.t, k.t"
            ),
        ),
        (
            "scores AVG(q.v * k.v), not an einsum".into(),
            "SELECT q.t AS t, k.t AS u, AVG(q.v * k.v) AS v FROM q JOIN k ON q.j = k.j \
             GROUP BY q.t, k.t"
                .into(),
        ),
        (
            "output (e/z).vv, ddx's 3-way join".into(),
            "SELECT e.t, vv.j, SUM(e.e / z.z * vv.v) AS v FROM e JOIN z ON e.t = z.t \
             JOIN vv ON e.u = vv.t GROUP BY e.t, vv.j"
                .into(),
        ),
        (
            "output p.vv, p = e/z precomputed".into(),
            "SELECT p.t, vv.j, SUM(p.v * vv.v) AS v FROM p JOIN vv ON p.u = vv.t \
             GROUP BY p.t, vv.j"
                .into(),
        ),
    ]
}

/// Run `sql` once, returning its time and its result as one batch.
async fn run(ctx: &SessionContext, sql: &str) -> (Duration, RecordBatch) {
    let start = Instant::now();
    let df = ctx.sql(sql).await.unwrap();
    let schema = Arc::new(df.schema().as_arrow().clone());
    let batches = df.collect().await.unwrap();
    let elapsed = start.elapsed();
    (elapsed, concat_batches(&schema, &batches).unwrap())
}

/// The median time of [`RUNS`] runs after one warm-up, and the last result.
async fn median(ctx: &SessionContext, sql: &str) -> (Duration, RecordBatch) {
    run(ctx, sql).await;
    let mut times = Vec::new();
    let mut last = None;
    for _ in 0..RUNS {
        let (t, r) = run(ctx, sql).await;
        times.push(t);
        last = Some(r);
    }
    times.sort();
    (times[RUNS / 2], last.unwrap())
}

async fn fired(ctx: &SessionContext, sql: &str) -> bool {
    let plan = ctx
        .sql(&format!("EXPLAIN {sql}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let text = datafusion::arrow::util::pretty::pretty_format_batches(&plan).unwrap();
    text.to_string().contains("EinFoldExec")
}

/// Time every query off and on, print one markdown row each, and return the
/// rows whose results disagree.
async fn report(label: &str, p: &Pair, queries: Vec<(String, String)>) -> Vec<String> {
    let mut bad = Vec::new();
    for (name, sql) in queries {
        if !selected(&format!("{label} | {name}")) {
            continue;
        }
        // With einfold on, this query ran for over 500 s without finishing (plain
        // DataFusion takes about 0.2 s), so it is not run unless asked for.
        let known_slow = label == "matmul n=50000" && name.contains("A-bar");
        if known_slow && std::env::var("BENCH_ONLY").is_err() {
            println!("| {label} | {name} | - | over 500000 | - | yes | not run |");
            continue;
        }
        let (off, expected) = median(&p.off, &sql).await;
        let (on, actual) = median(&p.on, &sql).await;
        let fired = fired(&p.on, &sql).await;
        let agree = compare(&expected, &actual);
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        println!(
            "| {label} | {name} | {:.1} | {:.1} | {:.2}x | {} | {} |",
            ms(off),
            ms(on),
            ms(off) / ms(on),
            if fired { "yes" } else { "no" },
            if agree.is_ok() { "yes" } else { "NO" },
        );
        if let Err(diff) = agree {
            bad.push(format!("{label}: {name}\n{diff}"));
        }
    }
    bad
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "a measurement; minutes of runtime, best in --release"]
async fn ddx_matmul_and_attn() {
    println!("| size | query | off (ms) | on (ms) | speedup | EinFoldExec | same result |");
    println!("|---|---|---:|---:|---:|---|---|");
    let mut bad = Vec::new();
    for n in [1_000, 10_000, 50_000] {
        let p = Pair::new();
        let queries = matmul(&p, n, 16, 8).await;
        bad.extend(report(&format!("matmul n={n}"), &p, queries).await);
    }
    for l in [16, 64, 256] {
        let p = Pair::new();
        let queries = attention(&p, l, 16).await;
        bad.extend(report(&format!("attn L={l}"), &p, queries).await);
    }
    assert!(
        bad.is_empty(),
        "einfold changed results:\n{}",
        bad.join("\n")
    );
}
