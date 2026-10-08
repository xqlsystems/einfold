// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Spike S25: if ddx emitted its gradient contractions as an explicit fold
//! node, instead of SQL that einfold has to recognize, how much would that
//! take, and what would it give?
//!
//! ddx (an XQL Systems project for automatic differentiation of SQL queries)
//! turns a loss query into a program of SQL steps. Spike S22 found that each
//! gradient step of a matrix product recomputes the forward join for a NULL
//! test that turns out redundant, and that einfold's first-milestone
//! detector would not match those steps at all. ddx, though, knows at the
//! point it writes a gradient step exactly what the step is: the cotangent,
//! the other operand, which columns are which dimensions, and the table
//! whose rows the gradient must have. This spike plays ddx: from a ddx
//! program's own metadata it builds a `GradFold` node (`fold.rs`) for each
//! gradient, runs it inside DataFusion, and checks it against ddx's result.

mod fold;

use std::sync::Arc;
use std::time::Instant;

use datafusion::arrow::array::{Array, Float64Array, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::DFSchema;
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::logical_expr::{Extension, LogicalPlan};
use datafusion::prelude::SessionContext;
use ddx_datafusion::ad::{self, BackwardProgram, ColumnRef, OutputTable};
use fold::{Columns, GradFold, GradFoldQueryPlanner};

const LOSS: &str =
    "WITH c AS (SELECT a.s, w.o, SUM(a.val * w.val) AS z FROM a JOIN w ON a.k = w.k \
                    GROUP BY a.s, w.o) SELECT SUM(tanh(z) * tanh(z)) AS l FROM c";

fn context() -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_query_planner(Arc::new(GradFoldQueryPlanner))
        .build();
    SessionContext::new_with_state(state)
}

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

/// Run `program` with its checks, then every step again, so that the tables
/// `ad::run` releases (the cotangents) stay registered.
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

/// The cotangent of the forward product: the backward step whose table has
/// three columns, `(s, o, value)`.
async fn cotangent(ctx: &SessionContext, program: &BackwardProgram) -> String {
    for step in &program.backward_steps {
        if step.name.contains("cotangent")
            && ctx
                .table(step.name.as_str())
                .await
                .unwrap()
                .schema()
                .fields()
                .len()
                == 3
        {
            return step.name.clone();
        }
    }
    panic!("no three-column cotangent step");
}

/// What ddx would emit for gradient `g` of `SUM(a.val * w.val) … GROUP BY s, o`:
/// for `a`'s gradient, `Σ_o z̄[s, o] · w[k, o]` over `a`'s rows; for `w`'s,
/// `Σ_s a[s, k] · z̄[s, o]` over `w`'s rows. Every table is `(dim, dim, val)`.
async fn emit(ctx: &SessionContext, g: &OutputTable, cot: &str) -> LogicalPlan {
    let scan = |name: String| async move {
        ctx.table(name.as_str())
            .await
            .unwrap()
            .into_unoptimized_plan()
    };
    let (inputs, cols) = if g.columns[0] == "s" {
        (
            vec![
                scan(cot.into()).await,
                scan("w".into()).await,
                scan("a".into()).await,
            ],
            Columns {
                l: (0, 1, 2),
                r: (1, 0, 2),
                f: (0, 1, 2),
            },
        )
    } else {
        (
            vec![
                scan("a".into()).await,
                scan(cot.into()).await,
                scan("w".into()).await,
            ],
            Columns {
                l: (1, 0, 2),
                r: (0, 1, 2),
                f: (0, 1, 2),
            },
        )
    };
    let schema = Schema::new(vec![
        Field::new(&g.columns[0], DataType::Int64, false),
        Field::new(&g.columns[1], DataType::Int64, false),
        Field::new(&g.columns[2], DataType::Float64, true),
    ]);
    let schema = Arc::new(DFSchema::try_from(schema).unwrap());
    LogicalPlan::Extension(Extension {
        node: Arc::new(GradFold {
            cols,
            inputs,
            schema,
        }),
    })
}

/// Run `plan` and register its rows as `name`, as ddx materializes a step.
async fn materialize(ctx: &SessionContext, name: &str, plan: LogicalPlan) {
    let df = ctx.execute_logical_plan(plan).await.unwrap();
    let batches = df.collect().await.unwrap();
    let table = MemTable::try_new(batches[0].schema(), vec![batches]).unwrap();
    let _ = ctx.deregister_table(name);
    ctx.register_table(name, Arc::new(table)).unwrap();
}

// --- random tables, as in spike S22 --------------------------------------------------

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
            for _ in 0..if rng.chance(dup) { 2 } else { 1 } {
                c0.push(i as i64);
                c1.push(j as i64);
                v.push(if rng.chance(0.15) {
                    None
                } else if rng.chance(0.03) {
                    Some(f64::NAN)
                } else if rng.chance(0.02) {
                    Some(f64::INFINITY)
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

/// Rows of a `(k0, k1, val)` table, sorted, for comparison as multisets.
async fn rows_of(ctx: &SessionContext, table: &str) -> Vec<(i64, i64, Option<f64>)> {
    let mut out = Vec::new();
    for b in ctx
        .sql(&format!("SELECT * FROM \"{table}\""))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap()
    {
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
}

fn agree(a: &[(i64, i64, Option<f64>)], b: &[(i64, i64, Option<f64>)]) -> Result<(), String> {
    if a.len() != b.len() {
        return Err(format!("{} rows vs ddx's {}", a.len(), b.len()));
    }
    for (x, y) in a.iter().zip(b) {
        let ok = (x.0, x.1) == (y.0, y.1)
            && match (x.2, y.2) {
                (None, None) => true,
                (Some(p), Some(q)) => {
                    (p.is_nan() && q.is_nan()) || p == q || (p - q).abs() <= 1e-9 * (1.0 + q.abs())
                }
                _ => false,
            };
        if !ok {
            return Err(format!("{x:?} vs ddx's {y:?}"));
        }
    }
    Ok(())
}

async fn correctness(cases: usize) {
    let mut rng = Rng(0x5eed_2525);
    let (mut agreed, mut refused, mut failed, mut rows, mut nulls, mut nans) =
        (0, 0, Vec::new(), 0, 0, 0);
    for case in 0..cases {
        let both = case % 3 != 2;
        let (ns, nk, no) = (
            1 + rng.below(6) as usize,
            1 + rng.below(4) as usize,
            1 + rng.below(4) as usize,
        );
        let ctx = context();
        register(
            &ctx,
            "a",
            ("s", "k"),
            random_rows(&mut rng, ns, nk, if both { 0.0 } else { 0.3 }),
        );
        register(&ctx, "w", ("k", "o"), random_rows(&mut rng, nk, no, 0.0));
        let wrt = if both {
            vec![ColumnRef::new("w", "val"), ColumnRef::new("a", "val")]
        } else {
            vec![ColumnRef::new("w", "val")]
        };
        let program = ad::grad(&ctx, LOSS, &wrt).await.unwrap();
        if run_keep(&ctx, &program).await.is_err() {
            refused += 1;
            continue;
        }
        let cot = cotangent(&ctx, &program).await;
        let mut ok = true;
        for g in &program.gradients {
            materialize(&ctx, "ours", emit(&ctx, g, &cot).await).await;
            let theirs = rows_of(&ctx, &g.step).await;
            rows += theirs.len();
            nulls += theirs.iter().filter(|r| r.2.is_none()).count();
            nans += theirs
                .iter()
                .filter(|r| r.2.is_some_and(f64::is_nan))
                .count();
            if let Err(e) = agree(&rows_of(&ctx, "ours").await, &theirs) {
                failed.push(format!(
                    "case {case} ({ns}×{nk}×{no}, wrt {}): {e}",
                    if both { "a, w" } else { "w" }
                ));
                ok = false;
            }
        }
        agreed += ok as usize;
    }
    println!(
        "{cases} random cases: {agreed} agree, {refused} refused by ddx, {} disagree.",
        failed.len()
    );
    println!(
        "Paths taken: dense {}, hash {}.",
        fold::DENSE.load(std::sync::atomic::Ordering::Relaxed),
        fold::HASHED.load(std::sync::atomic::Ordering::Relaxed)
    );
    println!("Gradient rows compared: {rows}, of which NULL {nulls} and NaN {nans}.");
    for f in failed.iter().take(10) {
        println!("- {f}");
    }
}

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

async fn timing(n: usize) {
    let ctx = context();
    exec(&ctx, &format!("CREATE TABLE a AS SELECT CAST(r / 16 AS BIGINT) AS s, CAST(r % 16 AS BIGINT) AS k, sin(CAST(r AS DOUBLE)) AS val FROM (SELECT unnest(range(0, {})) AS r)", n * 16)).await;
    exec(&ctx, "CREATE TABLE w AS SELECT CAST(r / 8 AS BIGINT) AS k, CAST(r % 8 AS BIGINT) AS o, 0.1 * cos(CAST(r AS DOUBLE)) AS val FROM (SELECT unnest(range(0, 128)) AS r)").await;
    let wrt = [ColumnRef::new("w", "val"), ColumnRef::new("a", "val")];
    let program = ad::grad(&ctx, LOSS, &wrt).await.unwrap();
    run_keep(&ctx, &program).await.unwrap();
    let cot = cotangent(&ctx, &program).await;
    println!("| gradient | ddx's step, ms | `GradFold`, ms | speedup | same result |\n|---|---:|---:|---:|---|");
    let (mut ddx_total, mut ours_total) = (0.0, 0.0);
    for g in &program.gradients {
        let step = program.steps().find(|s| s.name == g.step).unwrap();
        let theirs_ms = best_ms!(ad::run_step(&ctx, step).await.unwrap());
        let ours_ms = best_ms!(materialize(&ctx, "ours", emit(&ctx, g, &cot).await).await);
        let same = agree(&rows_of(&ctx, "ours").await, &rows_of(&ctx, &g.step).await).is_ok();
        ddx_total += theirs_ms;
        ours_total += ours_ms;
        let label = if g.columns[0] == "s" {
            "Ā (a's)"
        } else {
            "W̄ (w's)"
        };
        println!(
            "| {label} | {theirs_ms:.1} | {ours_ms:.1} | {:.0}× | {} |",
            theirs_ms / ours_ms,
            if same { "yes" } else { "NO" }
        );
    }
    let all = best_ms!(ad::run(&ctx, &program).await.unwrap());
    println!("\nddx's whole run: {all:.1} ms; its two gradient steps {ddx_total:.1} ms, which `GradFold` does in {ours_total:.1} ms.");
}

#[tokio::main]
async fn main() {
    println!("## Correctness against ddx\n");
    correctness(300).await;
    println!("\n## Time at n = 50,000\n");
    timing(50_000).await;
}
