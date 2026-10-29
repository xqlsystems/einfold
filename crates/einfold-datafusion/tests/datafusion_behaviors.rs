// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Pins the behaviors of DataFusion, the host engine, that einfold's operator
//! must reproduce, or that einfold works around. A test here fails when
//! DataFusion changes the behavior, which is the signal to update einfold to
//! match, or to drop a workaround.
//!
//! **Float keys.** When a query joins or groups on a `DOUBLE` column,
//! DataFusion compares keys by their bits, except that `-0.0` and `0.0` are
//! one key. So `NaN` matches `NaN`, a NaN with its sign bit set is a different
//! key from one without, and a group of `-0.0` and `0.0` rows is emitted as
//! `0.0`. DataFusion 54 kept `-0.0` and `0.0` apart. A dictionary-encoded
//! `DOUBLE` still keeps them apart when grouped, which looks like an oversight.
//!
//! **NULL equality in join chains** (einfold issues #21 and #22). A join on
//! `=` and one on `IS NOT DISTINCT FROM` in the same chain can share one
//! setting for whether NULL matches NULL, so one of them answers wrongly. An
//! `=` in the WHERE clause over an `IS NOT DISTINCT FROM` join does too.
//!
//! **TopK over signed zeros** (einfold issue #63). `ORDER BY … LIMIT` over a
//! float column holding `-0.0` and `0.0` gives answers that depend on
//! partitions, batch sizes and timing.
//!
//! **Sort-merge joins over signed zeros** (einfold issue #67). A sort-merge
//! join on a float key and another key can drop a match when one side holds
//! both `-0.0` and `0.0`.
//!
//! **Sort-merge join filters** (apache/datafusion#24414, fixed after 55.2).
//! A sort-merge join's filter can read the wrong columns once a projection is
//! pushed into the join.

use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, AsArray, DictionaryArray, Float64Array, Int32Array, Int64Array, RecordBatch,
};
use datafusion::arrow::datatypes::{DataType, Field, Float64Type, Int64Type, Schema};
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;

/// A table `name (k, i)` of keys `k` and row numbers `i`, one row per batch,
/// dealt round-robin into `partitions` partitions, so that grouping merges
/// partial results.
fn register(ctx: &SessionContext, name: &str, k: ArrayRef, partitions: usize) {
    let i: ArrayRef = Arc::new(Int64Array::from_iter_values(0..k.len() as i64));
    let schema = Arc::new(Schema::new(vec![
        Field::new("k", k.data_type().clone(), true),
        Field::new("i", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![k, i]).unwrap();
    let mut parts = vec![Vec::new(); partitions];
    for r in 0..batch.num_rows() {
        parts[r % partitions].push(batch.slice(r, 1));
    }
    let table = MemTable::try_new(schema, parts).unwrap();
    ctx.register_table(name, Arc::new(table)).unwrap();
}

async fn run(ctx: &SessionContext, sql: &str) -> Vec<RecordBatch> {
    ctx.sql(sql).await.unwrap().collect().await.unwrap()
}

/// Column `c` of every row, as `f64` bits.
fn float_bits(batches: &[RecordBatch], c: usize) -> Vec<u64> {
    let mut out = Vec::new();
    for b in batches {
        let col = datafusion::arrow::compute::cast(b.column(c), &DataType::Float64).unwrap();
        out.extend(
            col.as_primitive::<Float64Type>()
                .values()
                .iter()
                .map(|v| v.to_bits()),
        );
    }
    out
}

/// Columns `c` and `c + 1` of every row, as integer pairs.
fn pairs(batches: &[RecordBatch], c: usize) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    for b in batches {
        let (x, y) = (b.column(c), b.column(c + 1));
        let (x, y) = (x.as_primitive::<Int64Type>(), y.as_primitive::<Int64Type>());
        out.extend((0..b.num_rows()).map(|r| (x.value(r), y.value(r))));
    }
    out
}

fn negative_nan() -> f64 {
    f64::from_bits(f64::NAN.to_bits() | (1 << 63))
}

#[tokio::test]
async fn grouping_a_double_merges_signed_zeros_into_positive_zero() {
    let ctx = SessionContext::new();
    let keys = vec![0.0, -0.0, -0.0, f64::NAN, negative_nan(), 1.0];
    register(&ctx, "t", Arc::new(Float64Array::from(keys)), 3);
    register(&ctx, "z", Arc::new(Float64Array::from(vec![-0.0, -0.0])), 2);

    // One key column, and two, which DataFusion groups with different code.
    for sql in [
        "SELECT k, COUNT(*) FROM t GROUP BY k",
        "SELECT k, COUNT(*) FROM t GROUP BY k, i % 1",
    ] {
        let out = run(&ctx, sql).await;
        let mut groups: Vec<(u64, i64)> = float_bits(&out, 0)
            .into_iter()
            .zip(
                out.iter()
                    .flat_map(|b| b.column(1).as_primitive::<Int64Type>().values().to_vec()),
            )
            .collect();
        groups.sort();
        let mut want = vec![
            (0.0f64.to_bits(), 3),
            (1.0f64.to_bits(), 1),
            (f64::NAN.to_bits(), 1),
            (negative_nan().to_bits(), 1),
        ];
        want.sort();
        assert_eq!(groups, want, "{sql}");
    }

    // A group of only -0.0 rows is still emitted as 0.0.
    let out = run(&ctx, "SELECT k FROM z GROUP BY k").await;
    assert_eq!(float_bits(&out, 0), vec![0.0f64.to_bits()]);

    let out = run(&ctx, "SELECT COUNT(DISTINCT k) FROM t").await;
    assert_eq!(out[0].column(0).as_primitive::<Int64Type>().value(0), 4);
}

/// Fails once DataFusion groups dictionary-encoded floats as it groups plain
/// ones.
#[tokio::test]
async fn grouping_a_dictionary_encoded_double_keeps_signed_zeros_apart() {
    let ctx = SessionContext::new();
    let keys: ArrayRef = Arc::new(DictionaryArray::new(
        Int32Array::from(vec![0, 1, 1]),
        Arc::new(Float64Array::from(vec![0.0, -0.0])),
    ));
    register(&ctx, "d", keys, 2);
    let out = run(&ctx, "SELECT k FROM d GROUP BY k").await;
    let mut keys = float_bits(&out, 0);
    keys.sort();
    assert_eq!(keys, vec![0.0f64.to_bits(), (-0.0f64).to_bits()]);
}

#[tokio::test]
async fn joining_doubles_matches_signed_zeros_and_equal_nans() {
    let ctx = SessionContext::new();
    let a = vec![-0.0, f64::NAN, negative_nan()];
    register(&ctx, "a", Arc::new(Float64Array::from(a)), 1);
    register(
        &ctx,
        "b",
        Arc::new(Float64Array::from(vec![0.0, f64::NAN])),
        1,
    );
    // A hash join on each kind of equality, and a nested-loop join's filter.
    for sql in [
        "SELECT a.i, b.i FROM a JOIN b ON a.k = b.k ORDER BY 1, 2",
        "SELECT a.i, b.i FROM a JOIN b ON a.k IS NOT DISTINCT FROM b.k ORDER BY 1, 2",
        "SELECT a.i, b.i FROM a JOIN b ON a.k = b.k OR a.i < 0 ORDER BY 1, 2",
    ] {
        // -0.0 matches 0.0, NaN matches NaN, and the negative NaN matches
        // nothing.
        assert_eq!(
            pairs(&run(&ctx, sql).await, 0),
            vec![(0, 0), (1, 1)],
            "{sql}"
        );
    }
}

async fn null_key_tables() -> SessionContext {
    let ctx = SessionContext::new();
    for sql in [
        "CREATE TABLE a AS VALUES (1, CAST(NULL AS BIGINT), 1.0)",
        "CREATE TABLE b AS VALUES (CAST(NULL AS BIGINT), 1.0)",
        "CREATE TABLE c AS VALUES (1, 1.0)",
    ] {
        run(&ctx, sql).await;
    }
    ctx
}

async fn count(ctx: &SessionContext, sql: &str) -> usize {
    run(ctx, sql).await.iter().map(|b| b.num_rows()).sum()
}

/// Issue #21. a.column2 is NULL, so the `=` join with b drops a's only row:
/// the answer is no rows. Changing only the other join to IS NOT DISTINCT
/// FROM makes the `=` join match NULL to NULL. Fails once fixed.
#[tokio::test]
async fn mixed_null_equality_in_a_join_chain_matches_null_with_equals() {
    let ctx = null_key_tables().await;
    let equals = "SELECT a.column1 FROM a JOIN b ON (b.column1 = a.column2) \
                  JOIN c ON (c.column1 = a.column1)";
    let mixed = "SELECT a.column1 FROM a JOIN b ON (b.column1 = a.column2) \
                 JOIN c ON (c.column1 IS NOT DISTINCT FROM a.column1)";
    assert_eq!(count(&ctx, equals).await, 0);
    assert_eq!(count(&ctx, mixed).await, 1, "wrong: should be 0");
}

/// Issue #21 again, with one join: the `=` in the WHERE clause becomes a
/// second key of the IS NOT DISTINCT FROM join, and matches NULL to NULL
/// with it. l.a and r.b are NULL, so `l.a = r.b` is NULL and the answer is
/// no rows, as it is with the same condition in ON. Fails once fixed.
#[tokio::test]
async fn where_equality_on_a_not_distinct_join_matches_null_with_equals() {
    let ctx = SessionContext::new();
    for sql in [
        "CREATE TABLE l (a BIGINT, b BIGINT) AS VALUES (NULL, 2)",
        "CREATE TABLE r (a BIGINT, b BIGINT) AS VALUES (2, NULL)",
    ] {
        run(&ctx, sql).await;
    }
    let on = "SELECT l.a FROM r JOIN l ON (l.b IS NOT DISTINCT FROM r.a) AND (l.a = r.b)";
    let filtered = "SELECT l.a FROM r JOIN l ON (l.b IS NOT DISTINCT FROM r.a) WHERE (l.a = r.b)";
    assert_eq!(count(&ctx, on).await, 0);
    assert_eq!(count(&ctx, filtered).await, 1, "wrong: should be 0");
}

/// Issue #22. NULL IS NOT DISTINCT FROM NULL, so the join keeps a's row, and
/// a cross join with one row should keep it too. Fails once fixed.
#[tokio::test]
async fn not_distinct_join_then_cross_join_drops_null_matches() {
    let ctx = null_key_tables().await;
    let join = "SELECT a.column1 FROM a JOIN b ON (b.column1 IS NOT DISTINCT FROM a.column2)";
    assert_eq!(count(&ctx, join).await, 1);
    assert_eq!(
        count(&ctx, &format!("{join} CROSS JOIN c")).await,
        0,
        "wrong: should be 1"
    );
}

/// The answers `SELECT k, i FROM t ORDER BY k, i LIMIT 5` gives over 40 rows
/// whose keys alternate between `first` and `second`, across partition
/// counts and batch sizes, each as its rows' `(k bits, i)`.
async fn top_five_answers(first: f64, second: f64) -> std::collections::BTreeSet<Vec<(u64, i64)>> {
    use datafusion::prelude::SessionConfig;
    let mut answers = std::collections::BTreeSet::new();
    for partitions in 1..=8 {
        for batch_size in [1, 2, 8192] {
            let config = SessionConfig::new()
                .with_target_partitions(partitions)
                .with_batch_size(batch_size);
            let ctx = SessionContext::new_with_config(config);
            let k: Vec<f64> = (0..40)
                .map(|i| if i % 2 == 0 { second } else { first })
                .collect();
            let i: Vec<i64> = (0..40).map(|i| (7 * i) % 13).collect();
            let schema = Arc::new(Schema::new(vec![
                Field::new("k", DataType::Float64, true),
                Field::new("i", DataType::Int64, false),
            ]));
            let columns: Vec<ArrayRef> = vec![
                Arc::new(Float64Array::from(k)),
                Arc::new(Int64Array::from(i)),
            ];
            let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
            let parts = (0..4).map(|p| vec![batch.slice(p * 10, 10)]).collect();
            let table = MemTable::try_new(schema, parts).unwrap();
            ctx.register_table("t", Arc::new(table)).unwrap();
            let out = run(&ctx, "SELECT k, i FROM t ORDER BY k, i LIMIT 5").await;
            let bits = float_bits(&out, 0);
            let rows: Vec<i64> = out
                .iter()
                .flat_map(|b| b.column(1).as_primitive::<Int64Type>().values().to_vec())
                .collect();
            answers.insert(bits.into_iter().zip(rows).collect());
        }
    }
    answers
}

/// Issue #63. `ORDER BY` sorts floats by IEEE 754's total order, `-0.0` before `0.0`,
/// but TopK's dynamic filter, which prunes
/// rows that can't make the top `LIMIT`, compares with SQL's `<` and `=`,
/// under which `-0.0 = 0.0`. So which rows survive depends on when the filter
/// tightens: on partitions, batch sizes and timing. The right answer here is
/// the `-0.0` rows with `i` 0, 0, 1, 2, 3. Fails once DataFusion gives one
/// answer.
#[tokio::test]
async fn top_k_over_signed_zeros_depends_on_timing() {
    let answers = top_five_answers(-0.0, 0.0).await;
    let right: Vec<(u64, i64)> = [0, 0, 1, 2, 3].map(|i| ((-0.0f64).to_bits(), i)).to_vec();
    assert!(answers.contains(&right));
    assert!(answers.len() > 1, "fixed: one answer, {answers:?}");
}

/// Issue #67. A sort-merge join on a `DOUBLE` key and a second key
/// drops a match when one side holds both `-0.0` and `0.0`: the sort orders
/// `-0.0` first, while the merge compares with `=`, under which they are
/// equal, so it decides by the second key and passes the match. `d`'s row
/// `(0.0, 0)` matches `e`'s; the hash join finds it. Fails once fixed.
#[tokio::test]
async fn sort_merge_join_over_signed_zeros_drops_a_match() {
    use datafusion::prelude::SessionConfig;
    let mut found = Vec::new();
    for hash in [true, false] {
        let config = SessionConfig::new()
            .with_target_partitions(4)
            .set_bool("datafusion.optimizer.prefer_hash_join", hash);
        let ctx = SessionContext::new_with_config(config);
        run(
            &ctx,
            "CREATE TABLE d (a DOUBLE, b BIGINT) AS VALUES (0.0, 0), (-0.0, 1)",
        )
        .await;
        run(
            &ctx,
            "CREATE TABLE e (a DOUBLE, b BIGINT) AS VALUES (0.0, 0)",
        )
        .await;
        let sql = "SELECT d.a, d.b FROM d JOIN e ON d.a = e.a AND d.b = e.b";
        found.push(count(&ctx, sql).await);
    }
    assert_eq!(
        found,
        [1, 0],
        "wrong: the sort-merge join should find 1 row"
    );
}

/// apache/datafusion#24414, fixed on DataFusion's main after 55.2. A
/// sort-merge join with a filter beyond its equalities keeps the filter's
/// column indices when a projection is pushed into it, so the filter reads
/// the wrong columns: here it drops the one row that joins. Upstream's own
/// reproduction. Fails once fixed.
#[tokio::test]
async fn sort_merge_join_filter_reads_stale_columns() {
    use datafusion::prelude::SessionConfig;
    let config = SessionConfig::new()
        .with_target_partitions(2)
        .set_bool("datafusion.optimizer.prefer_hash_join", false)
        .set_bool("datafusion.optimizer.repartition_joins", true);
    let ctx = SessionContext::new_with_config(config);
    let sql = "SELECT t1.column2 AS left_b1, t2.column2 AS right_b1 \
               FROM (VALUES (100, 1, 0)) t1 JOIN (VALUES (10, 1)) t2 \
               ON t1.column2 = t2.column2 AND t1.column1 > t2.column1";
    assert_eq!(
        count(&ctx, sql).await,
        0,
        "fixed: the right answer is 1 row"
    );
}
