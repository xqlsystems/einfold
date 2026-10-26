// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Hand tests of `EinFoldExec` as a DataFusion operator. What a fold computes
//! is tested against SQL in `einfold-kernels`' property test; these check
//! what the operator adds: reading partitioned inputs, emitting batches of the
//! session's size, its metrics, and the memory pool.
//!
//! The shape is ddx's gradient of a matrix product (ddx is an XQL Systems
//! project for automatic differentiation of SQL queries):
//!
//! ```sql
//! SELECT a.s, a.k, CASE WHEN a.v IS NULL THEN NULL ELSE COALESCE(g.v, 0.0) END
//! FROM a LEFT JOIN (SELECT c.s, w.k, SUM(c.v * w.v) AS v FROM c JOIN w ON c.o = w.o
//!                   GROUP BY c.s, w.k) g
//!   ON (a.s IS NOT DISTINCT FROM g.s) AND (a.k IS NOT DISTINCT FROM g.k)
//! ```
//!
//! with every value a small integer, so every sum is exact in any order and
//! results compare exactly.

use std::sync::Arc;

use datafusion::arrow::array::{Array, Float64Array, Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::ScalarValue;
use datafusion::datasource::MemTable;
use datafusion::execution::memory_pool::FairSpillPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::SessionStateBuilder;
use datafusion::logical_expr::{self, LogicalPlan, LogicalPlanBuilder};
use datafusion::prelude::{SessionConfig, SessionContext};
use einfold_datafusion::{EinFoldQueryPlanner, FoldNode};
use einfold_ir::{
    Aggregate, ColumnIndex, Dimension, DimensionId, Domain, DomainKey, Fold, Key, KeyEquality,
    Operand, Output, RowValue,
};

const N: i64 = 300;
const K: i64 = 4;
const O: i64 = 3;

fn context(threads: usize, batch_size: usize, pool: Option<usize>) -> SessionContext {
    let config = SessionConfig::new()
        .with_target_partitions(threads)
        .with_batch_size(batch_size);
    let mut runtime = RuntimeEnvBuilder::new();
    if let Some(bytes) = pool {
        runtime = runtime.with_memory_pool(Arc::new(FairSpillPool::new(bytes)));
    }
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(config)
        .with_runtime_env(runtime.build_arc().unwrap())
        .with_query_planner(Arc::new(EinFoldQueryPlanner::default()))
        .build();
    SessionContext::new_with_state(state)
}

/// Table `name (k0, k1, v)` over `0..e0 × 0..e1`, `v = f(row)` (NULL where
/// `f` says), dealt into `parts` partitions of 7-row batches.
fn register(
    ctx: &SessionContext,
    name: &str,
    keys: [&str; 2],
    [e0, e1]: [i64; 2],
    f: impl Fn(i64) -> Option<f64>,
    parts: usize,
) {
    let schema = Arc::new(Schema::new(vec![
        Field::new(keys[0], DataType::Int64, true),
        Field::new(keys[1], DataType::Int64, true),
        Field::new("v", DataType::Float64, true),
    ]));
    let rows = 0..e0 * e1;
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![
            Arc::new(Int64Array::from_iter_values(rows.clone().map(|r| r / e1))),
            Arc::new(Int64Array::from_iter_values(rows.clone().map(|r| r % e1))),
            Arc::new(rows.map(f).collect::<Float64Array>()),
        ],
    )
    .unwrap();
    let mut partitions = vec![Vec::new(); parts];
    for (i, start) in (0..batch.num_rows()).step_by(7).enumerate() {
        let len = 7.min(batch.num_rows() - start);
        partitions[i % parts].push(batch.slice(start, len));
    }
    let table = MemTable::try_new(schema, partitions).unwrap();
    ctx.register_table(name, Arc::new(table)).unwrap();
}

/// The tables: `a (s, k)`, the cotangent `c (s, o)`, and `w (k, o)`.
fn tables(ctx: &SessionContext, parts: usize) {
    // A few NULL values in the domain, which get NULL gradients.
    register(
        ctx,
        "a",
        ["s", "k"],
        [N, K],
        |r| (r % 11 != 0).then_some(1.0),
        parts,
    );
    register(
        ctx,
        "c",
        ["s", "o"],
        [N, O],
        |r| Some((r % 7 - 3) as f64),
        parts,
    );
    register(
        ctx,
        "w",
        ["k", "o"],
        [K, O],
        |r| Some((r % 5 - 2) as f64),
        parts,
    );
}

const SQL: &str =
    "SELECT a.s, a.k, CASE WHEN a.v IS NULL THEN NULL ELSE COALESCE(g.v, 0.0) END AS v \
     FROM a LEFT JOIN (SELECT c.s, w.k, SUM(c.v * w.v) AS v FROM c JOIN w ON c.o = w.o \
     GROUP BY c.s, w.k) g ON (a.s IS NOT DISTINCT FROM g.s) AND (a.k IS NOT DISTINCT FROM g.k)";

/// The same as a fold `s,o * k,o -> s,k` with domain `a`, and the `CASE` as
/// a projection above it.
async fn fold_plan(ctx: &SessionContext) -> LogicalPlan {
    let scan = |t: &'static str| async move { ctx.table(t).await.unwrap().into_unoptimized_plan() };
    let eq = KeyEquality::NotDistinct;
    let fold = Fold::try_new(
        vec![Dimension { equality: eq }; 3],
        vec![
            Operand {
                keys: vec![
                    Key::new(DimensionId(0), ColumnIndex(0)),
                    Key::new(DimensionId(1), ColumnIndex(1)),
                ],
                factor: Some(ColumnIndex(2)),
            },
            Operand {
                keys: vec![
                    Key::new(DimensionId(2), ColumnIndex(0)),
                    Key::new(DimensionId(1), ColumnIndex(1)),
                ],
                factor: Some(ColumnIndex(2)),
            },
        ],
        vec![
            Output {
                dimension: DimensionId(0),
                domain_key: Some(DomainKey::new(ColumnIndex(0), eq)),
            },
            Output {
                dimension: DimensionId(2),
                domain_key: Some(DomainKey::new(ColumnIndex(1), eq)),
            },
        ],
        RowValue::Product,
        Aggregate::sum(),
        Some(Domain {
            carry: vec![ColumnIndex(0), ColumnIndex(1), ColumnIndex(2)],
            missing: Some(0.0),
        }),
    )
    .unwrap();
    let node = FoldNode::try_new(
        fold,
        vec![scan("c").await, scan("w").await],
        Some(scan("a").await),
        ["s", "k", "v", "g"].map(String::from).to_vec(),
    )
    .unwrap();
    let value = logical_expr::when(
        logical_expr::col("v").is_null(),
        logical_expr::lit(ScalarValue::Float64(None)),
    )
    .otherwise(logical_expr::col("g"))
    .unwrap();
    LogicalPlanBuilder::from(node.into_plan())
        .project(vec![
            logical_expr::col("s"),
            logical_expr::col("k"),
            value.alias("v"),
        ])
        .unwrap()
        .build()
        .unwrap()
}

/// Rows as sorted `(s, k, v bits)`.
fn rows(batches: &[RecordBatch]) -> Vec<(i64, i64, Option<u64>)> {
    let mut out = Vec::new();
    for b in batches {
        let s = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let k = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let v = b.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        for r in 0..b.num_rows() {
            out.push((
                s.value(r),
                k.value(r),
                v.is_valid(r).then(|| v.value(r).to_bits()),
            ));
        }
    }
    out.sort();
    out
}

#[tokio::test]
async fn matches_sql_across_partitions_threads_and_batch_sizes() {
    let want = {
        let ctx = context(4, 8192, None);
        tables(&ctx, 1);
        rows(&ctx.sql(SQL).await.unwrap().collect().await.unwrap())
    };
    assert_eq!(want.len(), (N * K) as usize);
    for (parts, threads, batch_size) in [(1, 1, 8192), (3, 6, 5), (5, 2, 64)] {
        let ctx = context(threads, batch_size, None);
        tables(&ctx, parts);
        let plan = fold_plan(&ctx).await;
        let got = ctx
            .execute_logical_plan(plan)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap();
        assert!(got.iter().all(|b| b.num_rows() <= batch_size));
        assert_eq!(rows(&got), want, "{parts} partitions, {threads} threads");
    }
}

#[tokio::test]
async fn reports_which_algorithm_ran() {
    let ctx = context(4, 8192, None);
    tables(&ctx, 2);
    let plan = fold_plan(&ctx).await;
    let df = ctx.execute_logical_plan(plan).await.unwrap();
    let analyzed = df.explain(false, true).unwrap().collect().await.unwrap();
    let text = datafusion::arrow::util::pretty::pretty_format_batches(&analyzed)
        .unwrap()
        .to_string();
    assert!(
        text.contains("EinFoldExec: a,b * c,b -> a,c + domain"),
        "{text}"
    );
    assert!(text.contains("dense_folds=1"), "{text}");
}

#[tokio::test]
async fn fails_cleanly_when_the_memory_pool_is_too_small() {
    let ctx = context(4, 8192, Some(4096));
    tables(&ctx, 2);
    let plan = fold_plan(&ctx).await;
    let err = ctx
        .execute_logical_plan(plan)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap_err();
    assert!(err.to_string().contains("Resources exhausted"), "{err}");
}

#[tokio::test]
async fn holds_only_its_output_in_the_memory_pool() {
    use datafusion::physical_plan::ExecutionPlan;
    use futures::StreamExt;
    fn find(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
        match plan.name() == "EinFoldExec" {
            true => Some(Arc::clone(plan)),
            false => plan.children().into_iter().find_map(find),
        }
    }
    let ctx = context(4, 64, None);
    tables(&ctx, 2);
    let plan = fold_plan(&ctx).await;
    let physical = ctx.state().create_physical_plan(&plan).await.unwrap();
    let exec = find(&physical).unwrap();
    let mut stream = exec.execute(0, ctx.task_ctx()).unwrap();
    // While its output is read, the operator counts that output in the pool,
    // and not the inputs it has folded.
    // Arrow counts a sliced batch at its whole buffers' size, as the pool
    // does, so the output's batches can be summed to compare.
    let first = stream.next().await.unwrap().unwrap();
    let reserved = ctx.runtime_env().memory_pool.reserved();
    let mut output = first.get_array_memory_size();
    while let Some(batch) = stream.next().await {
        output += batch.unwrap().get_array_memory_size();
    }
    assert!(
        reserved <= output,
        "{reserved} bytes reserved for an output of {output}"
    );
}

#[test]
fn runs_where_there_is_no_tokio_runtime() {
    use datafusion::physical_plan::{self, ExecutionPlan};
    use futures::TryStreamExt;
    fn find(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
        match plan.name() == "EinFoldExec" {
            true => Some(Arc::clone(plan)),
            false => plan.children().into_iter().find_map(find),
        }
    }
    // A host in another library, such as DataFusion's Python bindings, polls
    // the operator through DataFusion's FFI on its own threads, where this
    // library's Tokio sees no runtime.
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let ctx = context(1, 8192, None);
    tables(&ctx, 3);
    // A plan's operators run once, so each run plans the query anew.
    let plan = || {
        runtime.block_on(async {
            let plan = fold_plan(&ctx).await;
            let physical = ctx.state().create_physical_plan(&plan).await.unwrap();
            find(&physical).unwrap()
        })
    };
    let (first, exec) = (plan(), plan());
    assert!(
        exec.children()
            .iter()
            .all(|c| c.properties().partitioning.partition_count() > 1),
        "the operator's inputs should have several partitions"
    );
    let with = runtime
        .block_on(physical_plan::collect(first, ctx.task_ctx()))
        .unwrap();
    drop(runtime);
    let stream = exec.execute(0, ctx.task_ctx()).unwrap();
    let without: Vec<RecordBatch> = futures::executor::block_on(stream.try_collect()).unwrap();
    assert_eq!(with, without);
}
