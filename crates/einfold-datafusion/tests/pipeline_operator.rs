// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Hand tests of `EinFoldExec` computing a fold pipeline: causal attention,
//! as a `PipelineNode` built by hand over DataFusion tables, gives the rows
//! of the SQL it stands for, runs the fused algorithm when its inputs are
//! complete, and runs the host's plan when they aren't. The algorithm's
//! semantics are checked in `einfold-kernels`' `pipeline_vs_sql.rs`; these
//! check the operator around it.

use std::sync::Arc;

use datafusion::arrow::array::{Array, Float64Array, Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::{self, ExecutionPlan};
use datafusion::prelude::{SessionConfig, SessionContext};
use einfold_datafusion::{EinFoldQueryPlanner, PipelineNode};
use einfold_ir::{
    Aggregate, ColumnIndex, Comparison, Condition, Dimension, DimensionId, Fold, Key, KeyEquality,
    Normalizer, Operand, Output, Pipeline, RowValue, Scale, Shift,
};

const L: usize = 150;
const D: usize = 8;
const C: usize = 4;

/// The SQL the pipeline stands for: causal attention.
const SQL: &str = "WITH scores AS (\
      SELECT q.t AS i, k.t AS j, SUM(q.x * k.x) / 8.0 AS s \
      FROM q JOIN k ON q.d = k.d WHERE q.t >= k.t GROUP BY q.t, k.t), \
    weights AS (SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS w FROM scores), \
    norm AS (SELECT i, SUM(w) AS z FROM weights GROUP BY i) \
    SELECT weights.i AS t, v.c AS c, SUM(weights.w * v.x) / norm.z AS x \
    FROM weights JOIN v ON weights.j = v.t JOIN norm ON weights.i = norm.i \
    GROUP BY weights.i, v.c, norm.z";

/// A session whose planner plans `PipelineNode`s, with `q`, `k` and `v`:
/// every `(t, d)` or `(t, c)`, except `v`'s rows in `missing`.
fn session(missing: &[usize]) -> SessionContext {
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(SessionConfig::new().with_target_partitions(4))
        .with_query_planner(Arc::new(EinFoldQueryPlanner::default()))
        .build();
    let ctx = SessionContext::new_with_state(state);
    let table = |name: &str, second: &str, width: usize, skip: &[usize]| {
        let schema = Arc::new(Schema::new(vec![
            Field::new("t", DataType::Int64, false),
            Field::new(second, DataType::Int64, false),
            Field::new("x", DataType::Float64, true),
        ]));
        let rows: Vec<usize> = (0..L * width).filter(|r| !skip.contains(r)).collect();
        let phase = name.len() as f64 + name.as_bytes()[0] as f64;
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from_iter_values(
                    rows.iter().map(|&r| (r / width) as i64),
                )),
                Arc::new(Int64Array::from_iter_values(
                    rows.iter().map(|&r| (r % width) as i64),
                )),
                Arc::new(Float64Array::from_iter_values(
                    rows.iter().map(|&r| (r as f64 * 0.37 + phase).sin()),
                )),
            ],
        )
        .unwrap();
        let halves = vec![
            vec![batch.slice(0, rows.len() / 2)],
            vec![batch.slice(rows.len() / 2, rows.len() - rows.len() / 2)],
        ];
        ctx.register_table(name, Arc::new(MemTable::try_new(schema, halves).unwrap()))
            .unwrap();
    };
    table("q", "d", D, &[]);
    table("k", "d", D, &[]);
    table("v", "c", C, missing);
    ctx
}

/// Causal attention as a pipeline: dimensions i, e, j of the scores
/// `i,e * j,e -> i,j` where `i >= j`, then the average of `v(j, c)`.
fn attention() -> Pipeline {
    const EQ: Dimension = Dimension {
        equality: KeyEquality::Equal,
    };
    let key = |d, c| Key::new(DimensionId(d), ColumnIndex(c));
    let out = |d| Output {
        dimension: DimensionId(d),
        domain_key: None,
    };
    let operand = |keys| Operand {
        keys,
        factor: Some(ColumnIndex(2)),
    };
    let ops = vec![
        operand(vec![key(0, 0), key(1, 1)]),
        operand(vec![key(2, 0), key(1, 1)]),
    ];
    let causal = Condition {
        left: DimensionId(0),
        comparison: Comparison::GreaterOrEqual,
        right: DimensionId(2),
        offset: 0,
    };
    let scores = Fold::try_new(
        vec![EQ; 3],
        ops,
        vec![out(0), out(2)],
        RowValue::Product,
        Aggregate::sum(),
        None,
    )
    .unwrap()
    .with_conditions(vec![causal])
    .unwrap();
    let values = operand(vec![key(2, 0), key(3, 1)]);
    let outputs = vec![DimensionId(0), DimensionId(3)];
    Pipeline::try_new(
        scores,
        Scale::Over(8.0),
        Shift::KeptMaximum,
        vec![EQ],
        values,
        outputs,
        Normalizer::KeptGroups,
    )
    .unwrap()
}

/// The pipeline's physical plan on `ctx`, with the SQL's plan as fallback.
async fn plan(ctx: &SessionContext) -> Arc<dyn ExecutionPlan> {
    let fallback = ctx.sql(SQL).await.unwrap().into_optimized_plan().unwrap();
    let scan = |name: &'static str| async move {
        ctx.table(name)
            .await
            .unwrap()
            .into_optimized_plan()
            .unwrap()
    };
    let inputs = [scan("q").await, scan("k").await, scan("v").await];
    let names = vec!["t".to_string(), "c".to_string(), "x".to_string()];
    let node = PipelineNode::try_new(attention(), inputs, names, fallback).unwrap();
    let state = ctx.state();
    state.create_physical_plan(&node.into_plan()).await.unwrap()
}

/// Rows sorted by keys: `(t, c, x)`.
fn rows(batches: &[RecordBatch]) -> Vec<(i64, i64, f64)> {
    let mut out = Vec::new();
    for b in batches {
        let col = |c: usize| {
            b.column(c)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .clone()
        };
        let (t, c) = (col(0), col(1));
        let x = b.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        for r in 0..b.num_rows() {
            out.push((t.value(r), c.value(r), x.value(r)));
        }
    }
    out.sort_by_key(|r| (r.0, r.1));
    out
}

/// Runs the pipeline and the SQL, checks they agree, and returns what
/// `EXPLAIN ANALYZE` shows of the operator.
async fn check(missing: &[usize]) -> String {
    let ctx = session(missing);
    let plan = plan(&ctx).await;
    let ours = rows(
        &physical_plan::collect(Arc::clone(&plan), ctx.task_ctx())
            .await
            .unwrap(),
    );
    let theirs = rows(&ctx.sql(SQL).await.unwrap().collect().await.unwrap());
    assert_eq!(ours.len(), theirs.len());
    for (o, t) in ours.iter().zip(&theirs) {
        assert_eq!((o.0, o.1), (t.0, t.1));
        assert!((o.2 - t.2).abs() <= 1e-12, "{o:?} vs {t:?}");
    }
    let shown = DisplayableExecutionPlan::with_metrics(plan.as_ref())
        .indent(true)
        .to_string();
    shown
}

#[tokio::test]
async fn runs_the_fused_algorithm_on_complete_inputs() {
    let shown = check(&[]).await;
    assert!(shown.contains("softmax(/ 8, shifted)"), "{shown}");
    assert!(shown.contains("fused_pipelines=1"), "{shown}");
    // 3 × 3 pairs of tiles, of which the causal mask leaves 6.
    assert!(
        shown.contains("tiles_visited=6") && shown.contains("tiles=9"),
        "{shown}"
    );
}

#[tokio::test]
async fn runs_the_host_plan_where_an_input_has_a_hole() {
    let shown = check(&[17]).await;
    assert!(shown.contains("host_plans=1"), "{shown}");
    assert!(!shown.contains("fused_pipelines=1"), "{shown}");
}

/// With no values at all, the SQL's inner join with `v` has no rows, and
/// neither does the pipeline, whichever plan runs.
#[tokio::test]
async fn gives_no_rows_where_there_are_no_values() {
    let every: Vec<usize> = (0..L * C).collect();
    check(&every).await;
}
