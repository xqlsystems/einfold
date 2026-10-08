// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Spike S22: where does ddx's gradient time go, and what would einfold have
//! to match to speed it up?
//!
//! ddx (an XQL Systems project for automatic differentiation of SQL queries)
//! turns a query computing a loss into a *program*: a list of SQL steps, each
//! materialized as a table, that computes the loss and its gradient. Spike
//! S21 timed the matrix products of a gradient as plain two-table queries
//! (`SELECT x, y, SUM(a.val * b.val) FROM a JOIN b ON a.z = b.z GROUP BY x,
//! y`). This spike times the steps ddx actually emits, and asks three
//! questions:
//!
//! 1. **Profile.** Per step: time, output rows, and plan shape (scans, inner
//!    and left joins, aggregates, and `CASE WHEN … IS NULL` tests below an
//!    aggregate). Does each step match a detector for "a `SUM` of products
//!    over a join of two tables", which is what einfold's first milestone
//!    detects?
//! 2. **Guards.** ddx's gradient steps recompute the forward join only to test
//!    whether the forward product was NULL. Is that test redundant, so that
//!    each gradient equals a two-table contraction plus ddx's final left
//!    join? Checked on random small tables with NULLs, NaNs, missing rows and
//!    duplicate keys.
//! 3. **Filling groups.** Each gradient step ends with a left join that gives
//!    every input row a gradient (0 where no group was reached). What does it
//!    cost, next to a dense kernel that writes every position anyway?

use std::sync::Arc;
use std::thread;
use std::time::Instant;

use datafusion::arrow::array::{Array, Float64Array, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::logical_expr::{Expr, JoinType, LogicalPlan};
use datafusion::prelude::SessionContext;
use ddx_datafusion::ad::{self, BackwardProgram, ColumnRef};

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

/// Best of three wall times, in milliseconds, of an async closure.
macro_rules! best_ms {
    ($body:expr) => {{
        let mut best = f64::MAX;
        for _ in 0..3 {
            let t = Instant::now();
            $body;
            best = best.min(t.elapsed().as_secs_f64() * 1e3);
        }
        best
    }};
}

// --- workloads ---------------------------------------------------------------

/// `CREATE TABLE name AS` a complete coordinate table over `0..e0 × 0..e1`,
/// with `val = scale · f(r + phase)`.
fn table(
    name: &str,
    c0: &str,
    c1: &str,
    e0: usize,
    e1: usize,
    f: &str,
    scale: f64,
    phase: f64,
) -> String {
    format!(
        "CREATE TABLE {name} AS SELECT CAST(r / {e1} AS BIGINT) AS {c0}, CAST(r % {e1} AS BIGINT) AS {c1}, \
         {scale} * {f}(CAST(r AS DOUBLE) + {phase}) AS val FROM (SELECT unnest(range(0, {})) AS r)",
        e0 * e1
    )
}

struct Workload {
    name: String,
    tables: Vec<String>,
    loss: String,
    wrt: Vec<(&'static str, &'static str)>,
}

/// ddx's `matmul` benchmark (`ad_perf.rs`): `SUM(tanh(a·w)²)`.
fn matmul(n: usize) -> Workload {
    Workload {
        name: format!("matmul n={n} d=16 h=8"),
        tables: vec![
            table("a", "s", "k", n, 16, "sin", 1.0, 0.0),
            table("w", "k", "o", 16, 8, "cos", 0.1, 0.0),
        ],
        loss: "WITH c AS (SELECT a.s, w.o, SUM(a.val * w.val) AS z FROM a JOIN w ON a.k = w.k \
               GROUP BY a.s, w.o) SELECT SUM(tanh(z) * tanh(z)) AS l FROM c"
            .into(),
        wrt: vec![("w", "val"), ("a", "val")],
    }
}

/// Two layers: `SUM(tanh(tanh(x·w1)·w2)²)`, with respect to both weights.
fn mlp2(n: usize) -> Workload {
    Workload {
        name: format!("mlp2 n={n} 16→16→8"),
        tables: vec![
            table("x", "s", "k", n, 16, "sin", 1.0, 0.0),
            table("w1", "k", "j", 16, 16, "cos", 0.1, 0.0),
            table("w2", "j", "o", 16, 8, "cos", 0.1, 0.5),
        ],
        loss: "WITH h AS (SELECT x.s, w1.j, tanh(SUM(x.val * w1.val)) AS v FROM x JOIN w1 ON x.k = w1.k \
               GROUP BY x.s, w1.j), y AS (SELECT h.s, w2.o, SUM(h.v * w2.val) AS z FROM h JOIN w2 \
               ON h.j = w2.j GROUP BY h.s, w2.o) SELECT SUM(tanh(z) * tanh(z)) AS l FROM y"
            .into(),
        wrt: vec![("w1", "val"), ("w2", "val")],
    }
}

/// ddx's `attn` benchmark: single-head self-attention with a softmax over
/// keys, with respect to the three projection matrices.
fn attn(l: usize) -> Workload {
    let d = 16;
    let proj = |w: &str| {
        format!("SELECT x.t, {w}.j, SUM(x.val * {w}.val) AS v FROM x JOIN {w} ON x.k = {w}.k GROUP BY x.t, {w}.j")
    };
    Workload {
        name: format!("attn L={l} d={d}"),
        tables: vec![
            table("x", "t", "k", l, d, "sin", 1.0, 0.0),
            table("wq", "k", "j", d, d, "cos", 0.1, 0.1),
            table("wk", "k", "j", d, d, "cos", 0.1, 0.2),
            table("wv", "k", "j", d, d, "cos", 0.1, 0.3),
        ],
        loss: format!(
            "WITH q AS ({}), k AS ({}), vv AS ({}), \
             s AS (SELECT q.t AS t, k.t AS u, SUM(q.v * k.v) / sqrt({d}.0) AS v \
                   FROM q JOIN k ON q.j = k.j GROUP BY q.t, k.t), \
             m AS (SELECT t, MAX(v) AS m FROM s GROUP BY t), \
             e AS (SELECT s.t, s.u, exp(s.v - m.m) AS e FROM s JOIN m ON s.t = m.t), \
             z AS (SELECT t, SUM(e) AS z FROM e GROUP BY t), \
             o AS (SELECT e.t, vv.j, SUM(e.e / z.z * vv.v) AS v \
                   FROM e JOIN z ON e.t = z.t JOIN vv ON e.u = vv.t GROUP BY e.t, vv.j) \
             SELECT SUM(v * v) AS l FROM o",
            proj("wq"),
            proj("wk"),
            proj("wv")
        ),
        wrt: vec![("wq", "val"), ("wk", "val"), ("wv", "val")],
    }
}

// --- plan shape --------------------------------------------------------------

#[derive(Default, Debug, Clone, Copy)]
struct Shape {
    scans: usize,
    inner: usize,
    left: usize,
    aggregates: usize,
    /// `CASE WHEN … IS NULL` tests inside an aggregate's input.
    guards: usize,
}

fn null_tests(e: &Expr) -> usize {
    let mut n = 0;
    e.apply(|x| {
        if let Expr::Case(c) = x {
            n += c
                .when_then_expr
                .iter()
                .filter(|(w, _)| matches!(w.as_ref(), Expr::IsNull(_)))
                .count();
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    n
}

fn shape(p: &LogicalPlan) -> Shape {
    let mut s = Shape::default();
    p.apply(|node| {
        match node {
            LogicalPlan::TableScan(_) => s.scans += 1,
            LogicalPlan::Join(j) if j.join_type == JoinType::Inner => s.inner += 1,
            LogicalPlan::Join(j) if j.join_type == JoinType::Left => s.left += 1,
            LogicalPlan::Aggregate(a) => {
                s.aggregates += 1;
                a.input
                    .apply(|below| {
                        s.guards += below.expressions().iter().map(null_tests).sum::<usize>();
                        Ok(TreeNodeRecursion::Continue)
                    })
                    .unwrap();
            }
            _ => {}
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    s
}

/// The first milestone's detector: some aggregate computes only `SUM`s, over
/// exactly one inner join of exactly two table scans, with no NULL guard.
fn two_table_sum(p: &LogicalPlan) -> bool {
    let mut found = false;
    p.apply(|node| {
        if let LogicalPlan::Aggregate(a) = node {
            let only_sums = !a.aggr_expr.is_empty()
                && a.aggr_expr
                    .iter()
                    .all(|e| format!("{e}").to_lowercase().starts_with("sum("));
            let s = shape(&a.input);
            if only_sums
                && s.scans == 2
                && s.inner == 1
                && s.left == 0
                && s.aggregates == 0
                && s.guards == 0
            {
                found = true;
                return Ok(TreeNodeRecursion::Stop);
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    found
}

// --- part 1: profile -----------------------------------------------------------

async fn rows(ctx: &SessionContext, table: &str) -> usize {
    ctx.table(table).await.unwrap().count().await.unwrap()
}

async fn profile(w: &Workload) {
    let ctx = SessionContext::new();
    for t in &w.tables {
        exec(&ctx, t).await;
    }
    let wrt: Vec<ColumnRef> = w.wrt.iter().map(|(t, c)| ColumnRef::new(*t, *c)).collect();
    let forward = best_ms!(exec(&ctx, &w.loss).await);
    let build = best_ms!(ad::grad(&ctx, w.loss.as_str(), &wrt).await.unwrap());
    let program = ad::grad(&ctx, w.loss.as_str(), &wrt).await.unwrap();
    let run = best_ms!(ad::run(&ctx, &program).await.unwrap());
    println!("\n### {}\n", w.name);
    println!(
        "forward {forward:.1} ms · build {build:.1} ms · run {run:.1} ms ({:.1}× forward)\n",
        run / forward
    );
    println!("| step | ms | rows | scans | inner joins | left joins | aggregates | NULL guards | two-table `SUM` |");
    println!("|---|---:|---:|---:|---:|---:|---:|---:|---|");
    let mut total = 0.0;
    let mut matched = 0.0;
    for step in program.steps() {
        let ms = best_ms!(ad::run_step(&ctx, step).await.unwrap());
        let plan = ad::logical_plan(&ctx, &step.plan).await.unwrap();
        let s = shape(&plan);
        let m = two_table_sum(&plan);
        total += ms;
        if m {
            matched += ms;
        }
        let short = step.name.splitn(4, '_').last().unwrap_or(&step.name);
        println!(
            "| `{short}` | {ms:.1} | {} | {} | {} | {} | {} | {} | {} |",
            rows(&ctx, &step.name).await,
            s.scans,
            s.inner,
            s.left,
            s.aggregates,
            s.guards,
            if m { "yes" } else { "no" }
        );
    }
    println!("\nSteps total {total:.1} ms; steps the two-table detector matches: {matched:.1} ms ({:.0}%).", 100.0 * matched / total);
}

// --- part 2: are the guards redundant? -------------------------------------------

/// A small deterministic generator (xorshift64*).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
    fn chance(&mut self, p: f64) -> bool {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64 <= p
    }
}

/// Rows of `(c0, c1, val)` over `0..e0 × 0..e1`: each present with
/// probability 0.8, duplicated with probability `dup`; values NULL with
/// probability 0.15, NaN with 0.03, otherwise multiples of 1/8 in [-2, 2].
fn random_rows(
    rng: &mut Rng,
    e0: usize,
    e1: usize,
    dup: f64,
) -> (Vec<i64>, Vec<i64>, Vec<Option<f64>>) {
    let (mut c0, mut c1, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..e0 {
        for j in 0..e1 {
            if !rng.chance(0.8) {
                continue;
            }
            let copies = if rng.chance(dup) { 2 } else { 1 };
            for _ in 0..copies {
                c0.push(i as i64);
                c1.push(j as i64);
                v.push(if rng.chance(0.15) {
                    None
                } else if rng.chance(0.03) {
                    Some(f64::NAN)
                } else {
                    Some((rng.below(33) as f64 - 16.0) / 8.0)
                });
            }
        }
    }
    (c0, c1, v)
}

fn register(
    ctx: &SessionContext,
    name: &str,
    cols: (&str, &str),
    data: (Vec<i64>, Vec<i64>, Vec<Option<f64>>),
) {
    let schema = Arc::new(Schema::new(vec![
        Field::new(cols.0, DataType::Int64, false),
        Field::new(cols.1, DataType::Int64, false),
        Field::new("val", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(data.0)),
            Arc::new(Int64Array::from(data.1)),
            Arc::new(Float64Array::from(data.2)),
        ],
    )
    .unwrap();
    ctx.register_batch(name, batch).unwrap();
}

/// Run `program` with its checks, then run every step again so that the
/// intermediate tables `ad::run` releases (such as the cotangents) stay
/// registered for the comparisons below.
async fn run_keep(
    ctx: &SessionContext,
    program: &BackwardProgram,
) -> datafusion::error::Result<()> {
    ad::run(ctx, program).await?;
    for step in program.steps() {
        ad::run_step(ctx, step).await?;
    }
    Ok(())
}

/// The cotangent of the forward product: the backward step with three
/// columns, `(s, o, value)`.
async fn cotangent(ctx: &SessionContext, program: &BackwardProgram) -> (String, Vec<String>) {
    for step in &program.backward_steps {
        if step.name.contains("cotangent") {
            let df = ctx.table(step.name.as_str()).await.unwrap();
            let cols: Vec<String> = df
                .schema()
                .fields()
                .iter()
                .map(|f| format!("\"{}\"", f.name()))
                .collect();
            if cols.len() == 3 {
                return (format!("\"{}\"", step.name), cols);
            }
        }
    }
    panic!("no three-column cotangent step");
}

/// Two-table forms of the gradients, each followed by ddx's own left join.
fn plain_abar(cot: &str, c: &[String]) -> String {
    format!(
        "SELECT a.s, a.k, CASE WHEN a.val IS NULL THEN NULL WHEN r.v IS NULL THEN 0.0 ELSE r.v END AS val \
         FROM a LEFT JOIN (SELECT c.{c0} AS s, w.k, SUM(c.{c2} * w.val) AS v FROM {cot} c JOIN w \
         ON c.{c1} = w.o GROUP BY c.{c0}, w.k) r ON a.s = r.s AND a.k = r.k",
        c0 = c[0],
        c1 = c[1],
        c2 = c[2]
    )
}

fn plain_wbar(cot: &str, c: &[String]) -> String {
    format!(
        "SELECT w.k, w.o, CASE WHEN w.val IS NULL THEN NULL WHEN r.v IS NULL THEN 0.0 ELSE r.v END AS val \
         FROM w LEFT JOIN (SELECT a.k, c.{c1} AS o, SUM(c.{c2} * a.val) AS v FROM a JOIN {cot} c \
         ON a.s = c.{c0} GROUP BY a.k, c.{c1}) r ON w.k = r.k AND w.o = r.o",
        c0 = c[0],
        c1 = c[1],
        c2 = c[2]
    )
}

/// Every row of `ours` must equal a row of ddx's gradient table on the same
/// keys, and the other way round: same NULLs, NaN where NaN, values within
/// 1e-9 relative. Duplicate keys are compared as multisets of values.
async fn same(
    ctx: &SessionContext,
    ours: &str,
    theirs: &str,
    keys: (&str, &str),
) -> Result<[usize; 3], String> {
    let collect = |sql: String| async move {
        let mut out: Vec<(i64, i64, Option<f64>)> = Vec::new();
        for b in ctx.sql(&sql).await.unwrap().collect().await.unwrap() {
            let k0 = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
            let k1 = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
            let v = b.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
            for i in 0..b.num_rows() {
                out.push((
                    k0.value(i),
                    k1.value(i),
                    if v.is_null(i) { None } else { Some(v.value(i)) },
                ));
            }
        }
        out.sort_by(|x, y| {
            (x.0, x.1)
                .cmp(&(y.0, y.1))
                .then(x.2.map(f64::to_bits).cmp(&y.2.map(f64::to_bits)))
        });
        out
    };
    let (k0, k1) = keys;
    let a = collect(format!(
        "SELECT {k0}, {k1}, CAST(val AS DOUBLE) FROM {ours}"
    ))
    .await;
    let b = collect(format!(
        "SELECT {k0}, {k1}, CAST(val AS DOUBLE) FROM {theirs}"
    ))
    .await;
    if a.len() != b.len() {
        return Err(format!("{} rows vs ddx's {}", a.len(), b.len()));
    }
    for (x, y) in a.iter().zip(&b) {
        let ok = (x.0, x.1) == (y.0, y.1)
            && match (x.2, y.2) {
                (None, None) => true,
                (Some(p), Some(q)) => {
                    (p.is_nan() && q.is_nan()) || (p - q).abs() <= 1e-9 * (1.0 + q.abs())
                }
                _ => false,
            };
        if !ok {
            return Err(format!("{x:?} vs ddx's {y:?}"));
        }
    }
    // Rows compared: total, NULL gradients, NaN gradients.
    Ok([
        b.len(),
        b.iter().filter(|y| y.2.is_none()).count(),
        b.iter().filter(|y| y.2.is_some_and(f64::is_nan)).count(),
    ])
}

async fn guards(cases: usize) {
    let mut rng = Rng(0x5eed_2222);
    let (mut agreed, mut refused, mut failed) = (0, 0, Vec::new());
    let mut seen = [0usize; 3];
    for case in 0..cases {
        // Two cases in three differentiate both tables, whose keys ddx checks;
        // the third differentiates `w` only, and `a` may repeat keys.
        let both = case % 3 != 2;
        let (ns, nk, no) = (
            1 + rng.below(6) as usize,
            1 + rng.below(4) as usize,
            1 + rng.below(4) as usize,
        );
        let ctx = SessionContext::new();
        register(
            &ctx,
            "a",
            ("s", "k"),
            random_rows(&mut rng, ns, nk, if both { 0.0 } else { 0.3 }),
        );
        register(&ctx, "w", ("k", "o"), random_rows(&mut rng, nk, no, 0.0));
        let loss = matmul(1).loss;
        let wrt: Vec<ColumnRef> = if both {
            vec![ColumnRef::new("w", "val"), ColumnRef::new("a", "val")]
        } else {
            vec![ColumnRef::new("w", "val")]
        };
        let program = ad::grad(&ctx, loss.as_str(), &wrt).await.unwrap();
        if run_keep(&ctx, &program).await.is_err() {
            refused += 1;
            continue;
        }
        let (cot, c) = cotangent(&ctx, &program).await;
        let mut ok = true;
        for g in &program.gradients {
            let theirs = format!("\"{}\"", g.step);
            let (sql, keys) = if g.columns[0] == "s" {
                (plain_abar(&cot, &c), ("s", "k"))
            } else {
                (plain_wbar(&cot, &c), ("k", "o"))
            };
            exec(&ctx, &format!("CREATE TABLE ours AS {sql}")).await;
            let r = same(&ctx, "ours", &theirs, keys).await;
            if let Ok(c) = r {
                for (s, v) in seen.iter_mut().zip(c) {
                    *s += v;
                }
            }
            if let Err(e) = r {
                failed.push(format!(
                    "case {case} ({ns}×{nk}×{no}, wrt {}): {e}",
                    if both { "a, w" } else { "w" }
                ));
                ok = false;
            }
            exec(&ctx, "DROP TABLE ours").await;
        }
        if ok {
            agreed += 1;
        }
    }
    println!(
        "\n{cases} random cases: {agreed} agree, {refused} refused by ddx, {} disagree.",
        failed.len()
    );
    println!(
        "Gradient rows compared: {}, of which NULL {} and NaN {}.",
        seen[0], seen[1], seen[2]
    );
    for f in failed.iter().take(10) {
        println!("- {f}");
    }
}

// --- part 3: the left join that fills groups ---------------------------------------

/// Ā as a dense positional kernel: stream the cotangent's rows `(s, o, v)`,
/// adding `v · W[o, :]` into output row `s`, then write every `(s, k)`
/// position as a coordinate row. Every position is written, so no group is
/// missing and no fill is needed (both inputs complete).
fn dense_abar(
    cot: &[RecordBatch],
    w: &[RecordBatch],
    ns: usize,
    nk: usize,
    no: usize,
    threads: usize,
) -> usize {
    let mut wt = vec![0.0; no * nk];
    for b in w {
        let (k, o, v) = cols(b);
        for i in 0..b.num_rows() {
            wt[o.value(i) as usize * nk + k.value(i) as usize] = v.value(i);
        }
    }
    let mut out = vec![0.0; ns * nk];
    let per = ns.div_ceil(threads);
    // Bucket cotangent rows by output-row range, so threads own disjoint rows.
    let mut buckets: Vec<Vec<(usize, usize)>> = vec![Vec::new(); threads];
    for (bi, b) in cot.iter().enumerate() {
        let s = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        for i in 0..b.num_rows() {
            buckets[s.value(i) as usize / per].push((bi, i));
        }
    }
    thread::scope(|scope| {
        for (t, (chunk, bucket)) in out.chunks_mut(per * nk).zip(&buckets).enumerate() {
            let wt = &wt;
            scope.spawn(move || {
                for &(bi, i) in bucket {
                    let (s, o, v) = cols(&cot[bi]);
                    let row = (s.value(i) as usize - t * per) * nk;
                    let wrow = &wt[o.value(i) as usize * nk..(o.value(i) as usize + 1) * nk];
                    for (x, wv) in chunk[row..row + nk].iter_mut().zip(wrow) {
                        *x += v.value(i) * wv;
                    }
                }
            });
        }
    });
    let s: Int64Array = (0..ns * nk).map(|p| (p / nk) as i64).collect();
    let k: Int64Array = (0..ns * nk).map(|p| (p % nk) as i64).collect();
    let v = Float64Array::from(out);
    assert_eq!(s.len() + k.len(), 2 * v.len());
    v.len()
}

fn cols(b: &RecordBatch) -> (&Int64Array, &Int64Array, &Float64Array) {
    (
        b.column(0).as_any().downcast_ref::<Int64Array>().unwrap(),
        b.column(1).as_any().downcast_ref::<Int64Array>().unwrap(),
        b.column(2).as_any().downcast_ref::<Float64Array>().unwrap(),
    )
}

async fn fill(n: usize) {
    let w = matmul(n);
    let ctx = SessionContext::new();
    for t in &w.tables {
        exec(&ctx, t).await;
    }
    let wrt: Vec<ColumnRef> = w.wrt.iter().map(|(t, c)| ColumnRef::new(*t, *c)).collect();
    let program = ad::grad(&ctx, w.loss.as_str(), &wrt).await.unwrap();
    run_keep(&ctx, &program).await.unwrap();
    let ddx_step = program
        .gradients
        .iter()
        .find(|g| g.columns[0] == "s")
        .unwrap()
        .step
        .clone();
    let ddx = best_ms!(
        ad::run_step(&ctx, program.steps().find(|s| s.name == ddx_step).unwrap())
            .await
            .unwrap()
    );
    let (cot, c) = cotangent(&ctx, &program).await;
    let full = plain_abar(&cot, &c);
    let inner = format!(
        "SELECT c.{c0} AS s, w.k, SUM(c.{c2} * w.val) AS v FROM {cot} c JOIN w ON c.{c1} = w.o GROUP BY c.{c0}, w.k",
        c0 = c[0],
        c1 = c[1],
        c2 = c[2]
    );
    let t_inner = best_ms!(exec(&ctx, &format!("CREATE OR REPLACE TABLE r AS {inner}")).await);
    let t_full = best_ms!(exec(&ctx, &format!("CREATE OR REPLACE TABLE r AS {full}")).await);
    // ddx registers each step's result as one partition. The same cotangent,
    // spread over as many partitions as cores:
    let threads = thread::available_parallelism().unwrap().get();
    let one = ctx
        .sql(&format!("SELECT * FROM {cot}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let rows: Vec<RecordBatch> = one
        .iter()
        .flat_map(|b| {
            (0..b.num_rows())
                .step_by(8192)
                .map(move |o| b.slice(o, 8192.min(b.num_rows() - o)))
        })
        .collect();
    let mut parts = vec![Vec::new(); threads];
    for (i, b) in rows.into_iter().enumerate() {
        parts[i % threads].push(b);
    }
    let mem = datafusion::datasource::MemTable::try_new(one[0].schema(), parts).unwrap();
    ctx.register_table("cot_parts", Arc::new(mem)).unwrap();
    let t_full_parts = best_ms!(
        exec(
            &ctx,
            &format!(
                "CREATE OR REPLACE TABLE r AS {}",
                plain_abar("cot_parts", &c)
            )
        )
        .await
    );
    let cot_b = ctx
        .sql(&format!("SELECT {}, {}, {} FROM {cot}", c[0], c[1], c[2]))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let w_b = ctx
        .sql("SELECT k, o, val FROM w")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let t1 = best_ms!(assert_eq!(dense_abar(&cot_b, &w_b, n, 16, 8, 1), n * 16));
    let tn = best_ms!(assert_eq!(
        dense_abar(&cot_b, &w_b, n, 16, 8, threads),
        n * 16
    ));
    println!("\n### Ā at n = {n}: filling missing groups\n");
    println!("| form | ms |\n|---|---:|");
    println!("| ddx's step (guard, recomputed join, aggregate, left join) | {ddx:.1} |");
    println!("| two-table aggregate only | {t_inner:.1} |");
    println!("| two-table aggregate + ddx's left join and `CASE` | {t_full:.1} |");
    println!("| the same, cotangent in {threads} partitions instead of 1 | {t_full_parts:.1} |");
    println!("| dense fold writing every (s, k), 1 thread | {t1:.1} |");
    println!("| dense fold writing every (s, k), {threads} threads | {tn:.1} |");
}

#[tokio::main]
async fn main() {
    let parts = std::env::var("S22_PARTS").unwrap_or_else(|_| "123".into());
    if parts.contains('1') {
        println!("## Part 1: per-step profile");
        for w in [
            matmul(10_000),
            matmul(50_000),
            mlp2(10_000),
            attn(64),
            attn(256),
        ] {
            profile(&w).await;
        }
    }
    if parts.contains('2') {
        println!("\n## Part 2: are the NULL guards redundant?");
        guards(300).await;
    }
    if parts.contains('3') {
        println!("\n## Part 3: filling missing groups");
        fill(50_000).await;
    }
}
