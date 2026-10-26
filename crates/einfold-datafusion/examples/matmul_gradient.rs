// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Time the two gradient contractions of a matrix product, as a `Fold` node
//! and as the SQL it stands for, on DataFusion.
//!
//! For `Y = A·W`, with `A` of shape `n × 16` and `W` of `16 × 8`, the
//! gradients are `Ā = Ȳ·Wᵀ` and `W̄ = Aᵀ·Ȳ`, where `Ȳ` is the cotangent of `Y`.
//! ddx, an XQL Systems project for automatic differentiation of SQL queries,
//! writes each as a contraction left-joined from the table it is the gradient
//! of, so every row of that table gets a gradient: the fold's *domain*. These
//! are the shapes of ddx's `matmul` benchmark.
//!
//! Run with `cargo run --release --example matmul_gradient [n …]`.

use std::sync::Arc;
use std::time::Instant;

use datafusion::common::ScalarValue;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::SessionStateBuilder;
use datafusion::logical_expr::{col, lit, when, LogicalPlan, LogicalPlanBuilder};
use datafusion::prelude::SessionContext;
use einfold_datafusion::{EinFoldQueryPlanner, FoldNode};
use einfold_ir::{
    Aggregate, ColumnIndex, Dimension, DimensionId, Domain, Fold, KeyEquality, Operand, Output,
    RowValue,
};

/// The fastest of five runs of `f`, in milliseconds.
async fn best<F, Fut>(mut f: F) -> Result<f64>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = Result<()>>,
{
    let mut best = f64::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        f().await?;
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    Ok(best)
}

/// `out[x, y] = Σ_z l[x, z]·r[z, y]` for each row of `domain(x, y, v)`, as
/// ddx writes a gradient: NULL where the domain's `v` is, else 0 where no
/// group was reached. Every table is `(key, key, v)`.
fn gradient(
    l: LogicalPlan,
    r: LogicalPlan,
    domain: LogicalPlan,
    cols: [&str; 3],
) -> Result<LogicalPlan> {
    let eq = KeyEquality::NotDistinct;
    let fold = Fold::try_new(
        vec![Dimension { equality: eq }; 3],
        vec![
            Operand {
                keys: vec![
                    (DimensionId(0), ColumnIndex(0)),
                    (DimensionId(1), ColumnIndex(1)),
                ],
                factor: Some(ColumnIndex(2)),
            },
            Operand {
                keys: vec![
                    (DimensionId(1), ColumnIndex(0)),
                    (DimensionId(2), ColumnIndex(1)),
                ],
                factor: Some(ColumnIndex(2)),
            },
        ],
        vec![
            Output {
                dimension: DimensionId(0),
                name: cols[0].into(),
                domain_key: Some((ColumnIndex(0), eq)),
            },
            Output {
                dimension: DimensionId(2),
                name: cols[2].into(),
                domain_key: Some((ColumnIndex(1), eq)),
            },
        ],
        RowValue::Product,
        Aggregate::sum(),
        "g",
        Some(Domain {
            carry: vec![ColumnIndex(0), ColumnIndex(1), ColumnIndex(2)],
            missing: Some(0.0),
        }),
    )
    .map_err(|e| DataFusionError::Plan(e.to_string()))?;
    let node = FoldNode::try_new(fold, vec![l, r], Some(domain))?;
    let value = when(col("v").is_null(), lit(ScalarValue::Float64(None))).otherwise(col("g"))?;
    LogicalPlanBuilder::from(node.into_plan())
        .project(vec![col(cols[0]), col(cols[2]), value.alias("v")])?
        .build()
}

async fn run(n: usize) -> Result<()> {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_query_planner(Arc::new(EinFoldQueryPlanner))
        .build();
    let ctx = SessionContext::new_with_state(state);
    let table = |name: &str, k0: &str, k1: &str, rows: usize, width: usize, scale: f64| {
        format!(
            "CREATE TABLE {name} AS SELECT CAST(r / {width} AS BIGINT) AS {k0}, \
             CAST(r % {width} AS BIGINT) AS {k1}, {scale} * sin(CAST(r AS DOUBLE)) AS v \
             FROM (SELECT unnest(range(0, {rows})) AS r)"
        )
    };
    for sql in [
        table("a", "s", "k", n * 16, 16, 1.0),
        table("w", "k", "o", 16 * 8, 8, 0.1),
        table("cot", "s", "o", n * 8, 8, 0.5),
        // The same tables with their keys swapped, for the transposes.
        "CREATE TABLE wt AS SELECT o, k, v FROM w".into(),
        "CREATE TABLE at AS SELECT k, s, v FROM a".into(),
    ] {
        ctx.sql(&sql).await?.collect().await?;
    }
    let scan = |t: &'static str| {
        let ctx = ctx.clone();
        async move { Ok::<_, DataFusionError>(ctx.table(t).await?.into_unoptimized_plan()) }
    };
    // Ā[s, k] = Σ_o Ȳ[s, o]·W[k, o]; W̄[k, o] = Σ_s A[s, k]·Ȳ[s, o].
    let cases = [
        (
            "Ā = Ȳ·Wᵀ",
            "SELECT a.s, a.k, CASE WHEN a.v IS NULL THEN NULL WHEN g.v IS NULL THEN 0.0 ELSE g.v END AS v \
             FROM a LEFT JOIN (SELECT cot.s, wt.k, SUM(cot.v * wt.v) AS v FROM cot JOIN wt ON cot.o = wt.o \
             GROUP BY cot.s, wt.k) g ON (a.s IS NOT DISTINCT FROM g.s) AND (a.k IS NOT DISTINCT FROM g.k)",
            ("cot", "wt", "a", ["s", "o", "k"]),
        ),
        (
            "W̄ = Aᵀ·Ȳ",
            "SELECT w.k, w.o, CASE WHEN w.v IS NULL THEN NULL WHEN g.v IS NULL THEN 0.0 ELSE g.v END AS v \
             FROM w LEFT JOIN (SELECT at.k, cot.o, SUM(at.v * cot.v) AS v FROM at JOIN cot ON at.s = cot.s \
             GROUP BY at.k, cot.o) g ON (w.k IS NOT DISTINCT FROM g.k) AND (w.o IS NOT DISTINCT FROM g.o)",
            ("at", "cot", "w", ["k", "s", "o"]),
        ),
    ];
    for (name, sql, (l, r, f, cols)) in cases {
        let plan = gradient(scan(l).await?, scan(r).await?, scan(f).await?, cols)?;
        let sql_ms = best(|| async { ctx.sql(sql).await?.collect().await.map(|_| ()) }).await?;
        let fold_ms = best(|| async {
            ctx.execute_logical_plan(plan.clone())
                .await?
                .collect()
                .await
                .map(|_| ())
        })
        .await?;
        println!(
            "| {n} | {name} | {sql_ms:.1} | {fold_ms:.1} | {:.0}× |",
            sql_ms / fold_ms
        );
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    println!("| n | gradient | SQL, ms | Fold, ms | speedup |\n|---:|---|---:|---:|---:|");
    for n in if sizes.is_empty() {
        vec![10_000, 50_000]
    } else {
        sizes
    } {
        run(n).await?;
    }
    Ok(())
}
