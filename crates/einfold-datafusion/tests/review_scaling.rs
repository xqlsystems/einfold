// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Review: how does EinFoldExec scale on the one contraction that ddx's
//! backward pass needs and the benchmark gave up on? Run with
//! `cargo test --release -p einfold-datafusion --test review_scaling -- --ignored --nocapture`.

use std::time::Instant;

use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;
use einfold_datafusion::enable;

async fn exec(ctx: &SessionContext, sql: &str) -> usize {
    ctx.sql(sql).await.unwrap().collect().await.unwrap().iter().map(|b| b.num_rows()).sum()
}

fn grid(c0: &str, c1: &str, rows: usize, cols: usize, val: &str) -> String {
    format!(
        "SELECT CAST(r / {cols} AS BIGINT) AS {c0}, CAST(r % {cols} AS BIGINT) AS {c1}, {val} AS val \
         FROM (SELECT unnest(range(0, {})) AS r)",
        rows * cols
    )
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "a measurement"]
async fn a_bar_scaling() {
    let on = SessionContext::new_with_state(
        enable(SessionStateBuilder::new().with_default_features()).build(),
    );
    let off = SessionContext::new();
    let q = "SELECT zbar.s, w.k, SUM(zbar.val * w.val) AS v FROM zbar JOIN w ON zbar.o = w.o GROUP BY zbar.s, w.k";
    for n in std::env::var("N").unwrap().split(',').map(|s| s.parse::<usize>().unwrap()) {
        exec(&off, "DROP TABLE IF EXISTS zbar").await;
        exec(&off, "DROP TABLE IF EXISTS w").await;
        exec(&off, "DROP TABLE IF EXISTS a").await;
        exec(&off, &format!("CREATE TABLE a AS {}", grid("s", "k", n, 16, "sin(CAST(r AS DOUBLE))"))).await;
        exec(&off, &format!("CREATE TABLE w AS {}", grid("k", "o", 16, 8, "cos(CAST(r AS DOUBLE))"))).await;
        if std::env::var("DERIVED").is_ok() {
            let z = "SELECT a.s, w.o, SUM(a.val * w.val) AS z FROM a JOIN w ON a.k = w.k GROUP BY a.s, w.o";
            exec(&off, &format!("CREATE TABLE zbar AS SELECT s, o, 2.0 * tanh(z) * (1.0 - tanh(z) * tanh(z)) AS val FROM ({z})")).await;
        } else {
            exec(&off, &format!("CREATE TABLE zbar AS {}", grid("s", "o", n, 8, "sin(CAST(r AS DOUBLE))"))).await;
        }
        let _ = on.deregister_table("zbar");
        let _ = on.deregister_table("w");
        for t in ["zbar", "w"] {
            on.register_table(t, off.table_provider(t).await.unwrap()).unwrap();
        }
        let t = Instant::now();
        let rows = exec(&off, q).await;
        let t_off = t.elapsed();
        let t = Instant::now();
        exec(&on, q).await;
        let t_on = t.elapsed();
        println!("n={n:6} rows={rows:8} off={:8.1}ms on={:9.1}ms ratio={:.1}x", t_off.as_secs_f64()*1e3, t_on.as_secs_f64()*1e3, t_on.as_secs_f64()/t_off.as_secs_f64());
    }
}

/// The benchmark's note says A-bar "ran for over 500 s" with einfold on. The
/// query itself takes ~1 s (see `a_bar_scaling`). `einfold_testkit::compare`
/// matches each expected row by a linear scan of the unmatched actual rows
/// (`unused.iter().position(..)`), which is quadratic: comparing a result with
/// itself takes time proportional to rows².
#[test]
#[ignore = "a measurement"]
fn compare_is_quadratic() {
    use datafusion::arrow::array::{Float64Array, Int64Array};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::arrow::record_batch::RecordBatch;
    use std::sync::Arc;
    for n in [5_000i64, 10_000, 20_000, 40_000] {
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Int64, true),
            Field::new("v", DataType::Float64, true),
        ]));
        // Same rows in opposite order: the worst case for a front-to-back scan.
        let fwd = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from_iter_values(0..n)),
                Arc::new(Float64Array::from_iter_values((0..n).map(|x| x as f64))),
            ],
        )
        .unwrap();
        let rev = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from_iter_values((0..n).rev())),
                Arc::new(Float64Array::from_iter_values((0..n).rev().map(|x| x as f64))),
            ],
        )
        .unwrap();
        let t = Instant::now();
        einfold_testkit::compare(&fwd, &rev).unwrap();
        println!("rows={n:6} compare={:8.1}ms", t.elapsed().as_secs_f64() * 1e3);
    }
}
