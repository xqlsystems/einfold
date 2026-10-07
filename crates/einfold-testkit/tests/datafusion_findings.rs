// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! DataFusion behavior the harness found. These tests pin the current
//! behavior, so a DataFusion bump that fixes a bug fails here on purpose:
//! then lift the generator's workaround.

use datafusion::prelude::SessionContext;

async fn count(sql: &str) -> usize {
    let ctx = SessionContext::new();
    for q in [
        "CREATE TABLE a AS VALUES (1, CAST(NULL AS BIGINT), 1.0)",
        "CREATE TABLE b AS VALUES (CAST(NULL AS BIGINT), 1.0)",
        "CREATE TABLE c AS VALUES (1, 1.0)",
    ] {
        ctx.sql(q).await.unwrap().collect().await.unwrap();
    }
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    batches.iter().map(|b| b.num_rows()).sum()
}

/// Float keys compare bitwise in `GROUP BY` and in both join forms: `-0.0`
/// differs from `0.0`, and NaN equals NaN (not IEEE, and not SQL's `=`). This
/// matches arrow's `RowConverter`, so the harness's naive oracle does the same.
#[tokio::test]
async fn float_keys_compare_bitwise() {
    let ctx = SessionContext::new();
    let q = "CREATE TABLE f AS SELECT * FROM (VALUES (CAST(0.0 AS DOUBLE)), \
             (CAST(-0.0 AS DOUBLE)), (CAST('NaN' AS DOUBLE)), (CAST('NaN' AS DOUBLE))) t(k)";
    ctx.sql(q).await.unwrap().collect().await.unwrap();
    for sql in [
        "SELECT k FROM f GROUP BY k",
        "SELECT x.k FROM f x JOIN (SELECT k FROM f GROUP BY k) y ON x.k = y.k",
        "SELECT x.k FROM f x JOIN (SELECT k FROM f GROUP BY k) y ON x.k IS NOT DISTINCT FROM y.k",
    ] {
        let n: usize = ctx
            .sql(sql)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap()
            .iter()
            .map(|b| b.num_rows())
            .sum();
        // Groups: 0.0, -0.0, NaN. Join: each of the 4 rows meets exactly its own group.
        assert_eq!(n, if sql.contains("JOIN") { 4 } else { 3 }, "{sql}");
    }
}

/// Issue #21. In `a JOIN b ON a.k = b.k JOIN c ON c.j IS NOT DISTINCT FROM a.j`, the
/// lower `=` join must drop the NULL key, so the result is empty. DataFusion
/// 54.1.0 returns one row: the physical plan marks the `=` join
/// `NullsEqual: true`. The same query with `=` in both joins is correct.
#[tokio::test]
async fn mixed_null_equality_across_joins_is_wrong() {
    let both_eq = "SELECT a.column1 FROM a JOIN b ON (b.column1 = a.column2) \
                   JOIN c ON (c.column1 = a.column1)";
    let mixed = "SELECT a.column1 FROM a JOIN b ON (b.column1 = a.column2) \
                 JOIN c ON (c.column1 IS NOT DISTINCT FROM a.column1)";
    assert_eq!(count(both_eq).await, 0);
    assert_eq!(
        count(mixed).await,
        1,
        "expected 0 rows; if this is 0, DataFusion fixed the bug"
    );
}

/// Issue #22. A chain of joins that uses `IS NOT DISTINCT FROM` loses its NULL matches:
/// `a JOIN b ON a.k IS NOT DISTINCT FROM b.k CROSS JOIN c` should keep the
/// NULL-to-NULL pair, but DataFusion 54.1.0 returns no rows.
#[tokio::test]
async fn not_distinct_join_then_cross_join_loses_null_matches() {
    let one = "SELECT a.column1 FROM a JOIN b ON (b.column1 IS NOT DISTINCT FROM a.column2)";
    let chained = format!("{one} CROSS JOIN c");
    assert_eq!(count(one).await, 1);
    assert_eq!(
        count(&chained).await,
        0,
        "expected 1 row; if this is 1, DataFusion fixed the bug"
    );
}
