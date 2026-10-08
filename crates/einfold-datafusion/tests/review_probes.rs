// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Review probes: run a query with einfold off and on, and report whether the
//! operator fired, whether the schema and the rows agree, and whether the
//! float results agree *bit for bit* (the harness compares to 1e-9, which
//! cannot see the sign of zero).

use datafusion::arrow::array::{Array, AsArray};
use datafusion::arrow::compute::concat_batches;
use datafusion::arrow::datatypes::{DataType, Float64Type};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use einfold_datafusion::enable;

struct Outcome {
    fired: bool,
    err_off: Option<String>,
    err_on: Option<String>,
    schema_same: bool,
    rows_same: bool,
    float_bits_same: bool,
    off: Option<RecordBatch>,
    on: Option<RecordBatch>,
}

async fn probe(setup: &[&str], query: &str) -> Outcome {
    let off = SessionContext::new();
    let on = SessionContext::new_with_state(
        enable(SessionStateBuilder::new().with_default_features()).build(),
    );
    for ctx in [&off, &on] {
        for s in setup {
            ctx.sql(s).await.unwrap().collect().await.unwrap();
        }
    }
    let fired = match on.sql(&format!("EXPLAIN {query}")).await {
        Ok(df) => format!("{:?}", df.collect().await.unwrap()).contains("EinFoldExec"),
        Err(_) => false,
    };
    let run = |ctx: &SessionContext| {
        let q = query.to_string();
        let ctx = ctx.clone();
        async move {
            let df = ctx.sql(&q).await.map_err(|e| e.to_string())?;
            let schema = std::sync::Arc::new(df.schema().as_arrow().clone());
            let b = df.collect().await.map_err(|e| e.to_string())?;
            Ok::<_, String>(concat_batches(&schema, &b).unwrap())
        }
    };
    let (a, b) = (run(&off).await, run(&on).await);
    let (err_off, err_on) = (a.as_ref().err().cloned(), b.as_ref().err().cloned());
    let (a, b) = (a.ok(), b.ok());
    let (mut schema_same, mut rows_same, mut float_bits_same) = (false, false, false);
    if let (Some(a), Some(b)) = (&a, &b) {
        schema_same = a.schema() == b.schema();
        rows_same = einfold_testkit::compare(a, b).is_ok();
        // Sort both by the debug form of each row, then compare float bits.
        float_bits_same = rows_same && {
            let bits = |r: &RecordBatch| {
                let mut v: Vec<String> = (0..r.num_rows())
                    .map(|i| {
                        r.columns()
                            .iter()
                            .map(|c| {
                                if c.is_null(i) {
                                    "NULL".into()
                                } else if *c.data_type() == DataType::Float64 {
                                    format!("{:#x}", c.as_primitive::<Float64Type>().value(i).to_bits())
                                } else {
                                    datafusion::arrow::util::display::array_value_to_string(c, i).unwrap()
                                }
                            })
                            .collect::<Vec<_>>()
                            .join(",")
                    })
                    .collect();
                v.sort();
                v
            };
            bits(a) == bits(b)
        };
    }
    Outcome { fired, err_off, err_on, schema_same, rows_same, float_bits_same, off: a, on: b }
}

fn report(name: &str, o: &Outcome) {
    println!(
        "{name:44} fired={:5} schema={:5} rows={:5} bits={:5} err_off={:?} err_on={:?}",
        o.fired, o.schema_same, o.rows_same, o.float_bits_same, o.err_off, o.err_on
    );
    if o.fired && (!o.rows_same || !o.schema_same || !o.float_bits_same) {
        println!("   off: {:?}\n   on:  {:?}", o.off, o.on);
    }
}

const MM: &str = "SELECT a.i, b.j, SUM(a.v * b.v) AS s FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j";

#[tokio::test]
async fn probes() {
    let tab = |ty: &str, lit: &[&str]| -> Vec<String> {
        let vals = |f: &dyn Fn(&str) -> String| {
            lit.iter().map(|l| f(l)).collect::<Vec<_>>().join(", ")
        };
        vec![
            format!("CREATE TABLE a AS SELECT * FROM (VALUES {}) t(i, k, v)", vals(&|l| format!("(1, CAST({l} AS {ty}), 2.0)"))),
            format!("CREATE TABLE b AS SELECT * FROM (VALUES {}) t(k, j, v)", vals(&|l| format!("(CAST({l} AS {ty}), 7, 3.0)"))),
        ]
    };
    let cases: Vec<(&str, Vec<String>, String)> = vec![
        ("sign of zero: SUM(-0.0 * 1.0)", vec![
            "CREATE TABLE a AS VALUES (1, 1, CAST(-0.0 AS DOUBLE))".into(),
            "CREATE TABLE b AS VALUES (1, 1, CAST(1.0 AS DOUBLE))".into(),
        ], "SELECT a.column1 AS i, b.column2 AS j, SUM(a.column3 * b.column3) AS s FROM a JOIN b ON a.column2 = b.column1 GROUP BY a.column1, b.column2".into()),
        ("key Date32", tab("DATE", &["'2020-01-01'", "'2020-01-02'"]), MM.into()),
        ("key Boolean", tab("BOOLEAN", &["true", "false"]), MM.into()),
        ("key Decimal(10,2)", tab("DECIMAL(10,2)", &["1.10", "2.20"]), MM.into()),
        ("key Timestamp", tab("TIMESTAMP", &["'2020-01-01 00:00:00'"]), MM.into()),
        ("key Float32", tab("FLOAT", &["0.5", "-0.0", "0.0"]), MM.into()),
        ("key Int8 (a) vs Int64 (b)", vec![
            "CREATE TABLE a AS SELECT CAST(1 AS BIGINT) i, CAST(2 AS TINYINT) k, 2.0 v".into(),
            "CREATE TABLE b AS SELECT CAST(2 AS BIGINT) k, CAST(7 AS BIGINT) j, 3.0 v".into(),
        ], MM.into()),
        ("key Utf8View", vec![
            "CREATE TABLE a AS SELECT 1 i, arrow_cast('x','Utf8View') k, 2.0 v".into(),
            "CREATE TABLE b AS SELECT arrow_cast('x','Utf8View') k, 7 j, 3.0 v".into(),
        ], MM.into()),
        ("key Dictionary", vec![
            "CREATE TABLE a AS SELECT 1 i, arrow_cast('x','Dictionary(Int32, Utf8)') k, 2.0 v".into(),
            "CREATE TABLE b AS SELECT arrow_cast('x','Dictionary(Int32, Utf8)') k, 7 j, 3.0 v".into(),
        ], MM.into()),
        ("self join, aliased", vec![
            "CREATE TABLE a AS SELECT * FROM (VALUES (1,1,2.0),(2,1,3.0)) t(i,k,v)".into(),
            "CREATE TABLE b AS SELECT 1".into(),
        ], "SELECT x.i, y.i AS j, SUM(x.v * y.v) AS s FROM a x JOIN a y ON x.k = y.k GROUP BY x.i, y.i".into()),
        ("inf*0 -> NaN, overflow", vec![
            "CREATE TABLE a AS SELECT 1 i, 1 k, 1e308 v".into(),
            "CREATE TABLE b AS SELECT 1 k, 1 j, 10.0 v".into(),
        ], MM.into()),
        ("SUM across mixed magnitudes (order)", vec![
            "CREATE TABLE a AS SELECT 1 i, c.k, c.v FROM (VALUES (1,1e16),(2,1.0),(3,-1e16),(4,1.0)) c(k,v)".into(),
            "CREATE TABLE b AS SELECT 1 AS k, 1 AS j, 1.0 AS v UNION ALL SELECT 2,1,1.0 UNION ALL SELECT 3,1,1.0 UNION ALL SELECT 4,1,1.0".into(),
        ], MM.into()),
        ("filter on a.k = 1", vec![
            "CREATE TABLE a AS SELECT * FROM (VALUES (1,1,2.0),(1,2,3.0)) t(i,k,v)".into(),
            "CREATE TABLE b AS SELECT * FROM (VALUES (1,5,2.0),(2,5,3.0)) t(k,j,v)".into(),
        ], format!("{} ", MM.replace("GROUP BY", "WHERE a.k = 1 GROUP BY"))),
        ("HAVING + ORDER + LIMIT above", vec![
            "CREATE TABLE a AS SELECT * FROM (VALUES (1,1,2.0),(2,1,3.0)) t(i,k,v)".into(),
            "CREATE TABLE b AS SELECT * FROM (VALUES (1,5,2.0),(1,6,3.0)) t(k,j,v)".into(),
        ], format!("{MM} HAVING SUM(a.v * b.v) > 4 ORDER BY 3 DESC LIMIT 1")),
        ("int division in factor (error movement)", vec![
            "CREATE TABLE a AS SELECT * FROM (VALUES (1,1,10,0),(2,2,10,2)) t(i,k,x,y)".into(),
            "CREATE TABLE b AS SELECT * FROM (VALUES (1,5,2.0)) t(k,j,v)".into(),
        ], "SELECT a.i, b.j, SUM(CAST(a.x / a.y AS DOUBLE) * b.v) AS s FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j".into()),
        ("power() in factor (error movement)", vec![
            "CREATE TABLE a AS SELECT * FROM (VALUES (1,1,0.0),(2,2,2.0)) t(i,k,x)".into(),
            "CREATE TABLE b AS SELECT * FROM (VALUES (1,5,2.0)) t(k,j,v)".into(),
        ], "SELECT a.i, b.j, SUM(power(a.x, -1.0) * b.v) AS s FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j".into()),
        ("random() in factor", vec![
            "CREATE TABLE a AS SELECT * FROM (VALUES (1,1,1.0)) t(i,k,x)".into(),
            "CREATE TABLE b AS SELECT * FROM (VALUES (1,5,2.0),(1,6,2.0)) t(k,j,v)".into(),
        ], "SELECT a.i, SUM(a.x * random() * b.v) AS s FROM a JOIN b ON a.k = b.k GROUP BY a.i".into()),
        ("COUNT(decimal * decimal) overflow", vec![
            "CREATE TABLE a AS SELECT 1 i, 1 k, CAST(99999999999999999999999999999999999999 AS DECIMAL(38,0)) v".into(),
            "CREATE TABLE b AS SELECT 1 k, 1 j, CAST(99999999999999999999999999999999999999 AS DECIMAL(38,0)) v".into(),
        ], "SELECT a.i, b.j, COUNT(a.v * b.v) AS s FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j".into()),
        ("AVG over a join with no matching rows + group", vec![
            "CREATE TABLE a AS SELECT 1 i, 1 k, 1.0 v".into(),
            "CREATE TABLE b AS SELECT 2 k, 1 j, 1.0 v".into(),
        ], MM.replace("SUM", "AVG")),
    ];
    for (name, setup, q) in cases {
        let setup: Vec<&str> = setup.iter().map(String::as_str).collect();
        report(name, &probe(&setup, &q).await);
    }
}

/// `-0.0` summed alone: SQL (DataFusion) starts its accumulator at `+0.0`
/// and returns `+0.0`; `PartialAggregate` starts from the first value and
/// returns `-0.0`. The harness compares floats with `==`, so it cannot see it.
#[tokio::test]
async fn sign_of_zero_differs_from_datafusion() {
    let o = probe(
        &[
            "CREATE TABLE a AS VALUES (1, 1, CAST(-0.0 AS DOUBLE))",
            "CREATE TABLE b AS VALUES (1, 1, CAST(1.0 AS DOUBLE))",
        ],
        "SELECT a.column1 AS i, SUM(a.column3 * b.column3) AS s FROM a JOIN b ON a.column2 = b.column1 GROUP BY a.column1",
    )
    .await;
    assert!(o.fired && o.rows_same && !o.float_bits_same);
    let get = |r: &Option<RecordBatch>| r.as_ref().unwrap().column(1).as_primitive::<Float64Type>().value(0);
    println!("off = {:?}, on = {:?}", get(&o.off), get(&o.on));
}

/// DataFusion's own operators account their memory against the session's
/// memory pool and fail with "Resources exhausted" or spill; `EinFoldExec`
/// collects both inputs and every group's state without asking the pool.
#[tokio::test(flavor = "multi_thread")]
async fn memory_pool_is_ignored() {
    use datafusion::execution::memory_pool::GreedyMemoryPool;
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use datafusion::prelude::SessionConfig;
    use std::sync::Arc;
    let env = Arc::new(
        RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(GreedyMemoryPool::new(8 << 20)))
            .build()
            .unwrap(),
    );
    let mk = |einfold: bool| {
        let b = SessionStateBuilder::new()
            .with_default_features()
            .with_config(SessionConfig::new())
            .with_runtime_env(env.clone());
        SessionContext::new_with_state(if einfold { enable(b) } else { b }.build())
    };
    let (off, on) = (mk(false), mk(true));
    // Tables are built with plain DataFusion over an unlimited pool.
    let big = SessionContext::new();
    for (n, c, name, cols) in [(200_000i64, 16i64, "a", ("i", "k")), (16, 8, "b", ("k", "j"))] {
        big.sql(&format!(
            "CREATE TABLE {name} AS SELECT CAST(r / {c} AS BIGINT) AS {}, CAST(r % {c} AS BIGINT) AS {}, \
             sin(CAST(r AS DOUBLE)) AS v FROM (SELECT unnest(range(0, {})) AS r)",
            cols.0, cols.1, n * c / if name == "a" { 1 } else { 16 } / if name == "a" { 1 } else { 1 }
        ))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    }
    for ctx in [&off, &on] {
        for t in ["a", "b"] {
            ctx.register_table(t, big.table_provider(t).await.unwrap()).unwrap();
        }
    }
    let q = "SELECT a.i, b.j, SUM(a.v * b.v) AS s FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j";
    let r_off = off.sql(q).await.unwrap().collect().await.map(|b| b.len());
    let r_on = on.sql(q).await.unwrap().collect().await.map(|b| b.len());
    println!("8 MB pool: off = {:?}, on = {:?}", r_off.map_err(|e| e.to_string().chars().take(80).collect::<String>()), r_on.map_err(|e| e.to_string()));
}

/// `einfold_ir::PartialAggregate` models `AVG` over integers as an exact
/// wrapping `i64` sum divided by the count. DataFusion's `AVG` accumulates in
/// `f64`. At the extremes they disagree, so the IR's `AVG` is not the engine's.
#[tokio::test]
async fn datafusion_avg_of_large_integers_is_a_float_average() {
    let ctx = SessionContext::new();
    let b = ctx
        .sql("SELECT AVG(x) FROM (VALUES (CAST(9223372036854775807 AS BIGINT)), (9223372036854775807)) t(x)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let v = b[0].column(0).as_primitive::<Float64Type>().value(0);
    assert_eq!(v, 9.223372036854775807e18); // the IR's AVG says -1.0 (see review_findings in einfold-ir)
}
