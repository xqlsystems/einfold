// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Spike S28: how much of ddx's gradient time could einfold find by detecting
//! folds in DataFusion's plans, with no help from ddx?
//!
//! ddx (an XQL Systems project for automatic differentiation of SQL queries)
//! turns a loss query into a program of SQL steps. Spike S22 found that its
//! steps rebuilt the forward join for a NULL test, so a detector for two-table
//! folds matched only 13–32% of step time, and design v15 made M1 an explicit
//! fold that ddx would emit. ddx#126 has since removed that join. This spike
//! asks again, at ddx `main`, on the optimized plans an optimizer rule would
//! see.
//!
//! The detector looks for three shapes:
//!
//! - **fold:** an `Aggregate` computing one `SUM(l * r)`, grouped by plain
//!   columns, over projections of an inner join with equality keys, where `l`
//!   reads only the join's left input and `r` only its right;
//! - **masked fold:** the same, but the join also has a condition that isn't
//!   an equality, such as causal attention's `q.t >= k.t` (milestone M2);
//! - **fill:** a left join from a relation onto an aggregate, on every one of
//!   the aggregate's group keys, which is how ddx gives every row of a table a
//!   gradient.
//!
//! For each step it reports its time, and the time of each matched subtree
//! run alone: what einfold could replace.

use std::time::Instant;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{Column, DFSchemaRef, NullEquality};
use datafusion::logical_expr::{BinaryExpr, Expr, JoinType, LogicalPlan, Operator};
use datafusion::prelude::SessionContext;
use ddx_datafusion::ad::{self, ColumnRef};
use ddx_datafusion::engine;

async fn exec(ctx: &SessionContext, sql: &str) {
    ctx.sql(sql).await.unwrap().collect().await.unwrap();
}

/// Best of three wall times, in milliseconds.
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

/// `CREATE TABLE name AS` a complete coordinate table over `0..e0 × 0..e1`.
fn table(name: &str, c0: &str, c1: &str, e0: usize, e1: usize, val: &str) -> String {
    format!(
        "CREATE TABLE {name} AS SELECT CAST(r / {e1} AS BIGINT) AS {c0}, CAST(r % {e1} AS BIGINT) AS {c1}, \
         {val} AS {} FROM (SELECT unnest(range(0, {})) AS r)",
        if c1 == "d" { "x" } else { "val" },
        e0 * e1
    )
}

struct Workload {
    name: String,
    tables: Vec<String>,
    loss: String,
    wrt: Vec<(&'static str, &'static str)>,
    /// A forward query to profile as well, as a demo runs it.
    forward: Option<String>,
}

/// ddx's `matmul` benchmark (`ad_perf.rs`): `SUM(tanh(a·w)²)`.
fn matmul(n: usize) -> Workload {
    Workload {
        name: format!("matmul n={n} d=16 h=8"),
        tables: vec![
            table("a", "s", "k", n, 16, "sin(CAST(r AS DOUBLE))"),
            table("w", "k", "o", 16, 8, "0.1 * cos(CAST(r AS DOUBLE))"),
        ],
        loss: "WITH c AS (SELECT a.s, w.o, SUM(a.val * w.val) AS z FROM a JOIN w ON a.k = w.k \
               GROUP BY a.s, w.o) SELECT SUM(tanh(z) * tanh(z)) AS l FROM c"
            .into(),
        wrt: vec![("w", "val"), ("a", "val")],
        forward: None,
    }
}

/// Two layers: `SUM(tanh(tanh(x·w1)·w2)²)`, with respect to both weights.
fn mlp2(n: usize) -> Workload {
    Workload {
        name: format!("mlp2 n={n} 16→16→8"),
        tables: vec![
            table("x", "s", "k", n, 16, "sin(CAST(r AS DOUBLE))"),
            table("w1", "k", "j", 16, 16, "0.1 * cos(CAST(r AS DOUBLE))"),
            table("w2", "j", "o", 16, 8, "0.1 * cos(CAST(r AS DOUBLE) + 0.5)"),
        ],
        loss: "WITH h AS (SELECT x.s, w1.j, tanh(SUM(x.val * w1.val)) AS v FROM x JOIN w1 ON x.k = w1.k \
               GROUP BY x.s, w1.j), y AS (SELECT h.s, w2.o, SUM(h.v * w2.val) AS z FROM h JOIN w2 \
               ON h.j = w2.j GROUP BY h.s, w2.o) SELECT SUM(tanh(z) * tanh(z)) AS l FROM y"
            .into(),
        wrt: vec![("w1", "val"), ("w2", "val")],
        forward: None,
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
            table("x", "t", "k", l, d, "sin(CAST(r AS DOUBLE))"),
            table("wq", "k", "j", d, d, "0.1 * cos(CAST(r AS DOUBLE) + 0.1)"),
            table("wk", "k", "j", d, d, "0.1 * cos(CAST(r AS DOUBLE) + 0.2)"),
            table("wv", "k", "j", d, d, "0.1 * cos(CAST(r AS DOUBLE) + 0.3)"),
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
        forward: None,
    }
}

/// Demo 1 (einfold's docs/demos.md, xarray-sql's benchmarks/ml/01_attention.py):
/// causal attention over `(t, d, x)` tables with 64 features, in its window or
/// `GROUP BY` spelling, and the gradient of half its squared norm with respect
/// to every value of Q, K and V.
fn demo1(l: usize, window: bool) -> Workload {
    let d = 64;
    let scores = "scores AS (\
          SELECT q.t AS i, k.t AS j, SUM(q.x * k.x) / 8.0 AS s \
          FROM q JOIN k ON q.d = k.d WHERE q.t >= k.t GROUP BY q.t, k.t)";
    let weights = if window {
        "weights AS (SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS w FROM scores)".to_string()
    } else {
        "row_max AS (SELECT i, MAX(s) AS m FROM scores GROUP BY i), \
         weights AS (SELECT s.i, s.j, exp(s.s - r.m) AS w FROM scores s JOIN row_max r ON s.i = r.i)"
            .to_string()
    };
    let attention = format!(
        "{scores}, {weights}, norm AS (SELECT i, SUM(w) AS z FROM weights GROUP BY i), \
         attention AS (SELECT w.i AS t, v.d, SUM(w.w / n.z * v.x) AS x FROM weights w \
                       JOIN v ON w.j = v.t JOIN norm n ON w.i = n.i GROUP BY w.i, v.d)"
    );
    let phase = |p: f64| format!("sin(0.7 * CAST(r AS DOUBLE) + {p})");
    Workload {
        name: format!(
            "demo 1 L={l} d={d}, {}",
            if window { "window" } else { "GROUP BY" }
        ),
        tables: vec![
            table("q", "t", "d", l, d, &phase(0.1)),
            table("k", "t", "d", l, d, &phase(0.2)),
            table("v", "t", "d", l, d, &phase(0.3)),
        ],
        loss: format!("WITH {attention} SELECT SUM(x * x) / 2 AS l FROM attention"),
        wrt: vec![("q", "x"), ("k", "x"), ("v", "x")],
        forward: Some(format!("WITH {attention} SELECT t, d, x FROM attention")),
    }
}

// --- the detector --------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Fold,
    MaskedFold,
    Fill { fold: bool, null_safe: bool },
}

struct Found {
    kind: Kind,
    plan: LogicalPlan,
}

/// `e` with each column produced by one of `projections` (outermost first)
/// replaced by the expression that produced it, aliases removed.
fn resolve(e: &Expr, projections: &[(DFSchemaRef, Vec<Expr>)]) -> Expr {
    let mut e = e.clone().unalias();
    for (schema, exprs) in projections {
        e = e
            .transform(|x| match &x {
                Expr::Column(c) => match schema.index_of_column(c) {
                    Ok(i) => Ok(Transformed::yes(exprs[i].clone().unalias())),
                    Err(_) => Ok(Transformed::no(x)),
                },
                _ => Ok(Transformed::no(x)),
            })
            .unwrap()
            .data;
    }
    e
}

fn columns(e: &Expr) -> Vec<Column> {
    e.column_refs().into_iter().cloned().collect()
}

/// Which input of a join every column of `e` comes from: `Some(true)` for the
/// left, `Some(false)` for the right, `None` for both or neither.
fn side(e: &Expr, left: &DFSchemaRef, right: &DFSchemaRef) -> Option<bool> {
    let cols = columns(e);
    if cols.is_empty() {
        return None;
    }
    if cols.iter().all(|c| left.has_column(c)) {
        Some(true)
    } else if cols.iter().all(|c| right.has_column(c)) {
        Some(false)
    } else {
        None
    }
}

/// Whether `node` is a fold, and whether it is masked.
fn fold_kind(node: &LogicalPlan) -> Option<Kind> {
    let LogicalPlan::Aggregate(agg) = node else {
        return None;
    };
    let aggr: Vec<Expr> = agg.aggr_expr.iter().map(|e| e.clone().unalias()).collect();
    let [Expr::AggregateFunction(f)] = aggr.as_slice() else {
        return None;
    };
    let p = &f.params;
    if f.func.name() != "sum" || p.distinct || p.filter.is_some() || p.args.len() != 1 {
        return None;
    }
    // Down through projections and filters to a join.
    let mut projections = Vec::new();
    let mut filters = Vec::new();
    let mut below = agg.input.as_ref();
    let join = loop {
        match below {
            LogicalPlan::Projection(pr) => {
                projections.push((pr.schema.clone(), pr.expr.clone()));
                below = pr.input.as_ref();
            }
            LogicalPlan::Filter(fi) => {
                filters.push(fi.predicate.clone());
                below = fi.input.as_ref();
            }
            LogicalPlan::Join(j) => break j,
            _ => return None,
        }
    };
    if join.join_type != JoinType::Inner || join.on.is_empty() {
        return None;
    }
    let (l, r) = (join.left.schema(), join.right.schema());
    let arg = resolve(&p.args[0], &projections);
    let Expr::BinaryExpr(BinaryExpr { left, op: Operator::Multiply, right }) = &arg else {
        return None;
    };
    match (side(left, l, r), side(right, l, r)) {
        (Some(a), Some(b)) if a != b => {}
        _ => return None,
    }
    for g in &agg.group_expr {
        let g = resolve(g, &projections);
        if !matches!(g, Expr::Column(_)) || side(&g, l, r).is_none() {
            return None;
        }
    }
    // A condition reading both inputs, other than the equality keys, masks.
    let masked = join.filter.is_some()
        || filters
            .iter()
            .any(|f| side(&resolve(f, &projections), l, r).is_none());
    Some(if masked { Kind::MaskedFold } else { Kind::Fold })
}

/// Whether `node` is a left join from a relation onto an aggregate, on all of
/// the aggregate's group keys.
fn fill_kind(node: &LogicalPlan) -> Option<Kind> {
    let LogicalPlan::Join(j) = node else {
        return None;
    };
    if j.join_type != JoinType::Left || j.filter.is_some() {
        return None;
    }
    let mut right = j.right.as_ref();
    loop {
        right = match right {
            LogicalPlan::SubqueryAlias(a) => a.input.as_ref(),
            LogicalPlan::Projection(p) => p.input.as_ref(),
            _ => break,
        };
    }
    let LogicalPlan::Aggregate(agg) = right else {
        return None;
    };
    if agg.group_expr.len() != j.on.len() {
        return None;
    }
    Some(Kind::Fill {
        fold: fold_kind(right) == Some(Kind::Fold),
        null_safe: j.null_equality == NullEquality::NullEqualsNull,
    })
}

/// Every outermost match in `plan`.
fn detect(plan: &LogicalPlan) -> Vec<Found> {
    let mut found = Vec::new();
    plan.apply(|node| {
        if let Some(kind) = fill_kind(node).or_else(|| fold_kind(node)) {
            found.push(Found {
                kind,
                plan: node.clone(),
            });
            return Ok(TreeNodeRecursion::Jump);
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    found
}

fn label(k: Kind) -> String {
    match k {
        Kind::Fold => "fold".into(),
        Kind::MaskedFold => "masked fold".into(),
        Kind::Fill { fold, null_safe } => format!(
            "fill onto {}{}",
            if fold { "a fold" } else { "an aggregate that isn't a fold" },
            if null_safe { ", NULL-safe" } else { ", `=`" }
        ),
    }
}

/// For a fill onto an aggregate that isn't a fold, the folds inside it.
fn inner_folds(plan: &LogicalPlan) -> Vec<Found> {
    let mut found = Vec::new();
    plan.apply(|node| {
        if let Some(kind) = fold_kind(node) {
            found.push(Found {
                kind,
                plan: node.clone(),
            });
            return Ok(TreeNodeRecursion::Jump);
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    found
}

async fn time_plan(ctx: &SessionContext, plan: &LogicalPlan) -> f64 {
    best_ms!(ctx
        .execute_logical_plan(plan.clone())
        .await
        .unwrap()
        .collect()
        .await
        .unwrap())
}

// --- the profile -----------------------------------------------------------------

async fn profile(w: &Workload) {
    let ctx = SessionContext::new();
    for t in &w.tables {
        exec(&ctx, t).await;
    }
    let wrt: Vec<ColumnRef> = w.wrt.iter().map(|(t, c)| ColumnRef::new(*t, *c)).collect();
    let program = ad::grad(&ctx, w.loss.as_str(), &wrt).await.unwrap();
    engine::execute(&ctx, &program).await.unwrap();
    let run = best_ms!(engine::execute(&ctx, &program).await.unwrap());
    println!("\n### {}\n", w.name);
    println!("Whole gradient program: {run:.1} ms.\n");
    println!("| step | ms | matched | matched alone, ms |");
    println!("|---|---:|---|---:|");
    let (mut total, mut covered, mut masked) = (0.0, 0.0, 0.0);
    let mut steps: Vec<(String, LogicalPlan)> = Vec::new();
    for step in program.steps() {
        let lp = engine::logical_plan(&ctx, &step.plan).await.unwrap();
        let ms = best_ms!(engine::run_step(&ctx, step).await.unwrap());
        total += ms;
        let short = step.name.splitn(4, '_').last().unwrap_or(&step.name).to_string();
        steps.push((format!("`{short}`"), lp));
        let lp = &steps.last().unwrap().1;
        report(&ctx, &steps.last().unwrap().0, ms, lp, &mut covered, &mut masked).await;
    }
    if let Some(q) = &w.forward {
        let lp = ctx.sql(q).await.unwrap().into_optimized_plan().unwrap();
        let ms = time_plan(&ctx, &lp).await;
        let (mut c, mut m) = (0.0, 0.0);
        report(&ctx, "forward query", ms, &lp, &mut c, &mut m).await;
    }
    println!(
        "\nSteps total {total:.1} ms. Matched folds and fills, each run alone: {covered:.1} ms ({:.0}%); masked folds: {masked:.1} ms ({:.0}%).",
        100.0 * covered / total,
        100.0 * masked / total
    );
}

async fn report(
    ctx: &SessionContext,
    name: &str,
    ms: f64,
    lp: &LogicalPlan,
    covered: &mut f64,
    masked: &mut f64,
) {
    let opt = ctx.state().optimize(lp).unwrap();
    let found = detect(&opt);
    if found.is_empty() {
        println!("| {name} | {ms:.1} | | |");
        // With S28_PLANS set, the plans of slow steps nothing matched.
        if std::env::var("S28_PLANS").is_ok() && ms > 15.0 {
            eprintln!("==== {name}, {ms:.1} ms\n{}", opt.display_indent());
        }
        return;
    }
    for (i, f) in found.iter().enumerate() {
        let alone = time_plan(ctx, &f.plan).await;
        let first = if i == 0 { format!("{name} | {ms:.1}") } else { " | ".into() };
        println!("| {first} | {} | {alone:.1} |", label(f.kind));
        match f.kind {
            Kind::MaskedFold => *masked += alone,
            Kind::Fill { fold: false, .. } => {
                // Only the folds inside it are einfold's without help.
                for g in inner_folds(&f.plan) {
                    let t = time_plan(ctx, &g.plan).await;
                    println!("| | | inside it: {} | {t:.1} |", label(g.kind));
                    match g.kind {
                        Kind::MaskedFold => *masked += t,
                        _ => *covered += t,
                    }
                }
            }
            _ => *covered += alone,
        }
    }
}

#[tokio::main]
async fn main() {
    println!(
        "Threads: {}. DataFusion {}.",
        std::thread::available_parallelism().unwrap(),
        datafusion::DATAFUSION_VERSION
    );
    for w in [
        matmul(50_000),
        mlp2(10_000),
        attn(256),
        demo1(256, true),
        demo1(256, false),
    ] {
        profile(&w).await;
    }
}
