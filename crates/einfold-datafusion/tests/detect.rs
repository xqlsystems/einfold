// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Detection over plans of real SQL: which queries it reads as folds, and
//! which it must decline.

use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::datasource::MemTable;
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::SessionContext;
use einfold_datafusion::detect::{detect, FoldMatch, MOVABLE_FUNCTIONS};
use einfold_ir::{Aggregate, Dim, KeyEquality, Semiring};

fn table(cols: &[(&str, DataType)]) -> Arc<MemTable> {
    let fields: Vec<Field> = cols
        .iter()
        .map(|(n, t)| Field::new(*n, t.clone(), true))
        .collect();
    Arc::new(MemTable::try_new(Arc::new(Schema::new(fields)), vec![vec![]]).unwrap())
}

fn context() -> SessionContext {
    use DataType::{Float64 as F, Int64 as I};
    let ctx = SessionContext::new();
    let tables = [
        ("a", vec![("i", I), ("k", I), ("v", F)]),
        ("b", vec![("k", I), ("j", I), ("v", F)]),
        ("c", vec![("j", I), ("l", I), ("v", F)]),
        ("x", vec![("n", I), ("i", I), ("k", I), ("v", F)]),
        ("y", vec![("n", I), ("k", I), ("j", I), ("v", F)]),
        ("m", vec![("i", I), ("j", I), ("v", F)]),
        ("act", vec![("n", I), ("o", I), ("z", F), ("g", F)]),
        ("inp", vec![("n", I), ("p", I), ("x", F)]),
        ("ints", vec![("i", I), ("k", I), ("x", I)]),
        ("strs", vec![("k", I), ("s", DataType::Utf8)]),
    ];
    for (name, cols) in tables {
        ctx.register_table(name, table(&cols)).unwrap();
    }
    ctx
}

/// Plan `sql` and return its first `Aggregate` node. With `optimize`, the
/// plan goes through the analyzer and optimizer, as an optimizer rule sees it;
/// without, it is the SQL planner's output.
async fn aggregate(sql: &str, optimize: bool) -> LogicalPlan {
    let df = context().sql(sql).await.unwrap();
    let plan = if optimize {
        df.into_optimized_plan().unwrap()
    } else {
        df.logical_plan().clone()
    };
    let mut found = None;
    plan.apply(|p| {
        if matches!(p, LogicalPlan::Aggregate(_)) {
            found = Some(p.clone());
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    found.unwrap_or_else(|| panic!("no aggregate in\n{plan}"))
}

/// Detects the fold in the optimized plan, and checks that the planned
/// plan gives the same fold.
async fn detected(sql: &str) -> FoldMatch {
    let [planned, optimized] = [false, true].map(|optimize| async move {
        let agg = aggregate(sql, optimize).await;
        detect(&agg).unwrap_or_else(|| panic!("not detected:\n{}", agg.display_indent()))
    });
    let (planned, optimized) = (planned.await, optimized.await);
    assert_eq!(planned.fold, optimized.fold);
    optimized
}

/// Asserts that detection declines both the planned and the optimized plan.
async fn declined(sql: &str) {
    for optimize in [false, true] {
        let agg = aggregate(sql, optimize).await;
        if let Some(d) = detect(&agg) {
            panic!("detected {} in\n{}", d.fold, agg.display_indent());
        }
    }
}

/// Asserts that detection declines the planned plan. For queries the
/// optimizer rewrites into a shape that is a fold.
async fn declined_before_optimization(sql: &str) {
    let agg = aggregate(sql, false).await;
    if let Some(d) = detect(&agg) {
        panic!("detected {} in\n{}", d.fold, agg.display_indent());
    }
}

fn d(s: &str) -> Dim {
    Dim::new(s)
}

/// Each operand's dimension columns, as `qualifier.name` strings.
fn dim_columns(det: &FoldMatch) -> Vec<Vec<String>> {
    det.operands
        .iter()
        .map(|o| o.dim_columns.iter().map(|c| c.flat_name()).collect())
        .collect()
}

fn values(det: &FoldMatch) -> Vec<String> {
    det.operands.iter().map(|o| o.value.to_string()).collect()
}

#[tokio::test]
async fn matrix_product() {
    let det = detected(
        "SELECT a.i, b.j, SUM(a.v * b.v) AS s FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j",
    )
    .await;
    assert_eq!(det.fold.to_string(), "SUM(a[i,k] · b[k,j]) -> [i,j]");
    assert_eq!(det.fold.equality(&d("k")), Some(KeyEquality::Equal));
    // Group keys outside every join keep NULL as a coordinate, as GROUP BY does.
    assert_eq!(
        det.fold.equality(&d("i")),
        Some(KeyEquality::NotDistinctFrom)
    );
    assert_eq!(dim_columns(&det), [vec!["a.i", "a.k"], vec!["b.k", "b.j"]]);
    assert_eq!(values(&det), ["a.v", "b.v"]);
    let outs: Vec<(String, Dim)> = det
        .group_outputs
        .iter()
        .map(|(c, d)| (c.flat_name(), d.clone()))
        .collect();
    assert_eq!(outs, [("a.i".into(), d("i")), ("b.j".into(), d("j"))]);
    assert_eq!(det.value_output.name, "sum(a.v * b.v)");
}

#[tokio::test]
async fn where_clause_join() {
    let det =
        detected("SELECT a.i, b.j, SUM(a.v * b.v) FROM a, b WHERE a.k = b.k GROUP BY a.i, b.j")
            .await;
    assert_eq!(det.fold.to_string(), "SUM(a[i,k] · b[k,j]) -> [i,j]");
}

#[tokio::test]
async fn batched_matrix_product() {
    let det = detected(
        "SELECT x.n, x.i, y.j, SUM(x.v * y.v) FROM x JOIN y ON x.n = y.n AND x.k = y.k \
         GROUP BY x.n, x.i, y.j",
    )
    .await;
    assert_eq!(det.fold.to_string(), "SUM(x[n,i,k] · y[n,k,j]) -> [n,i,j]");
    assert_eq!(det.fold.summed(), [d("k")].into());
}

#[tokio::test]
async fn three_operand_chain() {
    let det = detected(
        "SELECT a.i, c.l, SUM(a.v * b.v * c.v) FROM a \
         JOIN b ON a.k = b.k JOIN c ON b.j = c.j GROUP BY a.i, c.l",
    )
    .await;
    assert_eq!(
        det.fold.to_string(),
        "SUM(a[i,k] · b[k,j] · c[j,l]) -> [i,l]"
    );
    assert_eq!(values(&det), ["a.v", "b.v", "c.v"]);
}

#[tokio::test]
async fn is_not_distinct_from_join() {
    let det = detected(
        "SELECT a.i, b.j, SUM(a.v * b.v) FROM a JOIN b ON a.k IS NOT DISTINCT FROM b.k \
         GROUP BY a.i, b.j",
    )
    .await;
    assert_eq!(det.fold.to_string(), "SUM(a[i,k] · b[k,j]) -> [i,j]");
    assert_eq!(
        det.fold.equality(&d("k")),
        Some(KeyEquality::NotDistinctFrom)
    );
}

#[tokio::test]
async fn transitive_class_spans_three_operands() {
    let det = detected(
        "SELECT a.i, SUM(a.v * b.v * c.v) FROM a JOIN b ON a.k = b.k JOIN c ON b.k = c.j \
         GROUP BY a.i",
    )
    .await;
    assert_eq!(det.fold.to_string(), "SUM(a[i,k] · b[k] · c[k]) -> [i]");
}

#[tokio::test]
async fn diagonal() {
    let det = detected("SELECT m.i, SUM(m.v) FROM m WHERE m.i = m.j GROUP BY m.i").await;
    assert_eq!(det.fold.to_string(), "SUM(m[i,i]) -> [i]");
    assert_eq!(det.fold.equality(&d("i")), Some(KeyEquality::Equal));
    assert_eq!(dim_columns(&det), [vec!["m.i", "m.j"]]);
}

#[tokio::test]
async fn group_key_that_is_not_joined() {
    // b.j is not a join key: it becomes a dimension of its own.
    let det = detected("SELECT b.j, SUM(a.v * b.v) FROM a JOIN b ON a.k = b.k GROUP BY b.j").await;
    assert_eq!(det.fold.to_string(), "SUM(a[k] · b[k,j]) -> [j]");
}

#[tokio::test]
async fn two_group_keys_in_one_class() {
    let det =
        detected("SELECT a.k, b.k, SUM(a.v * b.v) FROM a JOIN b ON a.k = b.k GROUP BY a.k, b.k")
            .await;
    assert_eq!(det.fold.to_string(), "SUM(a[k] · b[k]) -> [k]");
    let dims: Vec<&Dim> = det.group_outputs.iter().map(|(_, d)| d).collect();
    assert_eq!(dims, [&d("k"), &d("k")]);
}

#[tokio::test]
async fn self_join_keeps_operands_apart() {
    let det = detected(
        "SELECT p.i, q.k, SUM(p.v * q.v) FROM a p JOIN a q ON p.k = q.i GROUP BY p.i, q.k",
    )
    .await;
    assert_eq!(det.fold.to_string(), "SUM(p[i,k] · q[k,q.k]) -> [i,q.k]");
}

#[tokio::test]
async fn single_leaf_filters() {
    let det = detected(
        "SELECT a.i, b.j, SUM(a.v * b.v) FROM a JOIN b ON a.k = b.k \
         WHERE a.i < 10 AND b.v > 0.5 GROUP BY a.i, b.j",
    )
    .await;
    assert_eq!(det.fold.to_string(), "SUM(a[i,k] · b[k,j]) -> [i,j]");
    for (op, pred) in det
        .operands
        .iter()
        .zip(["a.i < Int64(10)", "b.v > Float64(0.5)"])
    {
        let LogicalPlan::Filter(f) = op.plan.as_ref() else {
            panic!("no filter on\n{}", op.plan.display_indent());
        };
        assert_eq!(f.predicate.to_string(), pred);
    }
}

#[tokio::test]
async fn factors_in_projection_below_aggregate() {
    // As ddx writes them: the product is a column of a subquery.
    let det = detected(
        "SELECT i, j, SUM(prod) FROM \
         (SELECT a.i AS i, b.j AS j, a.v * b.v * 2.0 AS prod FROM a JOIN b ON a.k = b.k) t \
         GROUP BY i, j",
    )
    .await;
    assert_eq!(det.fold.to_string(), "SUM(a[i,k] · b[k,j]) -> [i,j]");
    // The constant is folded into the first operand.
    assert_eq!(values(&det), ["a.v * Float64(2)", "b.v"]);
}

#[tokio::test]
async fn derived_factor_over_one_leaf() {
    // A tanh layer's weight gradient: Σₙ x · ȳ · (1 − z²), with ȳ and z saved.
    let det = detected(
        "SELECT inp.p, act.o, SUM(inp.x * act.g * (1 - act.z * act.z)) FROM inp \
         JOIN act ON inp.n IS NOT DISTINCT FROM act.n GROUP BY inp.p, act.o",
    )
    .await;
    assert_eq!(det.fold.to_string(), "SUM(inp[n,p] · act[n,o]) -> [p,o]");
    assert_eq!(
        values(&det),
        ["inp.x", "act.g * (Float64(1) - act.z * act.z)"]
    );
}

#[tokio::test]
async fn operand_without_factor() {
    let det = detected("SELECT a.i, SUM(a.v) FROM a JOIN b ON a.k = b.k GROUP BY a.i").await;
    assert_eq!(det.fold.to_string(), "SUM(a[i,k] · b[k]) -> [i]");
    assert_eq!(values(&det), ["a.v", "Float64(1)"]);
}

#[tokio::test]
async fn declines_left_join() {
    declined("SELECT a.i, SUM(a.v * b.v) FROM a LEFT JOIN b ON a.k = b.k GROUP BY a.i").await;
}

#[tokio::test]
async fn declines_cross_leaf_inequality() {
    declined("SELECT a.i, SUM(a.v * b.v) FROM a JOIN b ON a.k = b.k WHERE a.i >= b.j GROUP BY a.i")
        .await;
}

#[tokio::test]
async fn declines_sum_distinct() {
    // The optimizer rewrites SUM(DISTINCT v) into a SUM over a GROUP BY leaf,
    // which is a fold.
    declined_before_optimization("SELECT a.i, SUM(DISTINCT a.v) FROM a GROUP BY a.i").await;
}

#[tokio::test]
async fn declines_integer_sum() {
    declined("SELECT ints.i, SUM(ints.x) FROM ints GROUP BY ints.i").await;
}

#[tokio::test]
async fn declines_integer_product_cast_to_float() {
    declined("SELECT a.i, SUM(CAST(a.k * b.j AS DOUBLE)) FROM a JOIN b ON a.k = b.k GROUP BY a.i")
        .await;
}

#[tokio::test]
async fn declines_group_by_expression() {
    declined("SELECT a.i + 1, SUM(a.v) FROM a GROUP BY a.i + 1").await;
}

#[tokio::test]
async fn declines_group_by_expression_from_projection() {
    declined("SELECT i2, SUM(v) FROM (SELECT a.i + 1 AS i2, a.v FROM a) t GROUP BY i2").await;
}

#[tokio::test]
async fn declines_sum_of_cross_leaf_sum() {
    declined("SELECT a.i, SUM(a.v + b.v) FROM a JOIN b ON a.k = b.k GROUP BY a.i").await;
}

#[tokio::test]
async fn declines_nonlinear_cross_leaf_factor() {
    declined("SELECT a.i, SUM(exp(a.v * b.v)) FROM a JOIN b ON a.k = b.k GROUP BY a.i").await;
}

#[tokio::test]
async fn declines_mixed_key_equality() {
    // The optimizer turns the IS NOT DISTINCT FROM into `=`, since b.k cannot
    // be NULL after the first join; the planned plan still mixes them.
    declined_before_optimization(
        "SELECT a.i, SUM(a.v * b.v * c.v) FROM a JOIN b ON a.k = b.k \
         JOIN c ON b.k IS NOT DISTINCT FROM c.j GROUP BY a.i",
    )
    .await;
}

#[tokio::test]
async fn declines_two_aggregates() {
    declined("SELECT a.i, SUM(a.v), SUM(a.k) FROM a GROUP BY a.i").await;
}

#[tokio::test]
async fn declines_no_group_by() {
    // SQL returns one row even when nothing joined; a fold has none.
    declined("SELECT SUM(a.v * b.v) FROM a JOIN b ON a.k = b.k").await;
}

#[tokio::test]
async fn declines_volatile_factor() {
    declined("SELECT a.i, SUM(a.v * random()) FROM a GROUP BY a.i").await;
}

#[tokio::test]
async fn declines_other_aggregate() {
    declined("SELECT a.i, MAX(a.v) FROM a GROUP BY a.i").await;
}

#[tokio::test]
async fn allowlisted_function_in_factor() {
    let det =
        detected("SELECT a.i, SUM(tanh(a.v) * exp(b.v)) FROM a JOIN b ON a.k = b.k GROUP BY a.i")
            .await;
    assert_eq!(det.fold.to_string(), "SUM(a[i,k] · b[k]) -> [i]");
}

// A factor or filter moved onto its leaf runs on rows that may never join, so
// anything that could raise an error must stay where it is: the rewrite must not
// fail where the original query succeeds.

#[tokio::test]
async fn declines_fallible_cast() {
    declined(
        "SELECT a.i, SUM(CAST(strs.s AS DOUBLE) * a.v) FROM a JOIN strs ON a.k = strs.k \
         GROUP BY a.i",
    )
    .await;
}

#[tokio::test]
async fn declines_integer_division() {
    declined("SELECT a.i, SUM(a.v * b.v * (a.i / a.k)) FROM a JOIN b ON a.k = b.k GROUP BY a.i")
        .await;
}

#[tokio::test]
async fn declines_unlisted_function() {
    declined("SELECT a.i, SUM(cos(a.v) * b.v) FROM a JOIN b ON a.k = b.k GROUP BY a.i").await;
}

/// Every function detection may move evaluates without error on awkward
/// `Float64` inputs.
#[tokio::test]
async fn movable_functions_never_fail() {
    use datafusion::arrow::array::Float64Array;
    use datafusion::arrow::record_batch::RecordBatch;

    let values = Float64Array::from(vec![
        Some(f64::NAN),
        Some(f64::INFINITY),
        Some(f64::NEG_INFINITY),
        Some(0.0),
        Some(-0.0),
        Some(-2.5),
        Some(-3.0),
        Some(0.5),
        Some(2.0),
        Some(f64::MAX),
        Some(f64::MIN_POSITIVE),
        None,
    ]);
    let n = 12;
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Float64, true)]));
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(values)]).unwrap();
    let ctx = SessionContext::new();
    let vals = MemTable::try_new(schema, vec![vec![batch]]).unwrap();
    ctx.register_table("vals", Arc::new(vals)).unwrap();

    for f in MOVABLE_FUNCTIONS {
        let sql = format!("SELECT {f}(x) FROM vals");
        let batches = ctx
            .sql(&sql)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap_or_else(|e| panic!("{sql} failed: {e}"));
        let got: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(got, n, "{sql}");
    }
}

#[tokio::test]
async fn count_star() {
    let det = detected("SELECT a.i, COUNT(*) FROM a JOIN b ON a.k = b.k GROUP BY a.i").await;
    assert_eq!(det.fold.to_string(), "COUNT(a[i,k] · b[k]) -> [i]");
    // COUNT of a product is a sum of 0/1 indicators: a semiring fold.
    assert_eq!(det.fold.semiring(), Some(Semiring::SumProduct));
    assert_eq!(
        values(&det),
        [
            "CASE WHEN Int64(1) IS NOT NULL THEN Float64(1) END",
            "Float64(1)"
        ]
    );
}

#[tokio::test]
async fn count_of_an_integer_product_and_of_strings() {
    // COUNT is exact on any type, and only asks whether each factor is NULL.
    let det = detected(
        "SELECT ints.i, COUNT(ints.x * b.j) FROM ints JOIN b ON ints.k = b.k GROUP BY ints.i",
    )
    .await;
    assert_eq!(det.fold.to_string(), "COUNT(ints[i,k] · b[k]) -> [i]");
    assert_eq!(
        values(&det),
        [
            "CASE WHEN ints.x IS NOT NULL THEN Float64(1) END",
            "CASE WHEN b.j IS NOT NULL THEN Float64(1) END"
        ]
    );
    let det =
        detected("SELECT a.i, COUNT(strs.s) FROM a JOIN strs ON a.k = strs.k GROUP BY a.i").await;
    assert_eq!(det.fold.to_string(), "COUNT(a[i,k] · strs[k]) -> [i]");
}

#[tokio::test]
async fn avg_of_a_product() {
    let det =
        detected("SELECT a.i, AVG(a.v * b.v) AS m FROM a JOIN b ON a.k = b.k GROUP BY a.i").await;
    assert_eq!(det.fold.to_string(), "AVG(a[i,k] · b[k]) -> [i]");
    // AVG is SUM / COUNT, not itself a semiring aggregate.
    assert_eq!(det.fold.semiring(), None);
    assert_eq!(det.fold.aggregate(), Aggregate::Avg);
    assert_eq!(det.value_output.name, "avg(a.v * b.v)");
}

#[tokio::test]
async fn sum_is_a_semiring_fold() {
    let det = detected("SELECT a.i, SUM(a.v * b.v) FROM a JOIN b ON a.k = b.k GROUP BY a.i").await;
    assert_eq!(det.fold.semiring(), Some(Semiring::SumProduct));
}

#[tokio::test]
async fn integer_avg_is_a_float_avg() {
    // DataFusion averages integers as Float64, casting each one first, so the
    // plan itself is a float AVG; the cast to Float64 cannot fail.
    // Before type coercion adds the cast, the argument is an integer, which
    // detection declines.
    let sql = "SELECT ints.i, AVG(ints.x) FROM ints GROUP BY ints.i";
    declined_before_optimization(sql).await;
    let det = detect(&aggregate(sql, true).await).unwrap();
    assert_eq!(values(&det), ["CAST(ints.x AS Float64)"]);
}

#[tokio::test]
async fn declines_count_distinct() {
    // The optimizer turns COUNT(DISTINCT) into a COUNT over a grouped leaf.
    declined_before_optimization("SELECT a.i, COUNT(DISTINCT a.v) FROM a GROUP BY a.i").await;
}

#[tokio::test]
async fn declines_count_of_fallible_cast() {
    declined(
        "SELECT a.i, COUNT(CAST(strs.s AS DOUBLE)) FROM a JOIN strs ON a.k = strs.k GROUP BY a.i",
    )
    .await;
}

#[tokio::test]
async fn declines_count_of_cross_leaf_sum() {
    declined("SELECT a.i, COUNT(a.v + b.v) FROM a JOIN b ON a.k = b.k GROUP BY a.i").await;
}
