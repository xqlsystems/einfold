// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! End-to-end: DataFusion with einfold enabled gives the same results, with
//! the same schemas, as DataFusion without it.
//!
//! The main invariant is checked by the equivalence harness
//! (`einfold-testkit`): on random two-operand folds (`SUM`, `COUNT` and `AVG`
//! over joins with NULL and duplicate keys, NULL and NaN values, and empty
//! tables), the result with einfold equals SQL's, and einfold's operator runs
//! whenever the query has a `GROUP BY`. The hand tests below check `EXPLAIN`,
//! plans einfold must leave alone, and a gradient as ddx writes it.

use std::collections::BTreeMap;
use std::sync::Arc;

use datafusion::arrow::array::{Float64Array, Int64Array};
use datafusion::arrow::compute::concat_batches;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::{collect, displayable};
use datafusion::prelude::SessionContext;
use einfold_testkit::{assert_same_result, check, sql_reference, Case};

fn context(einfold: bool) -> SessionContext {
    let builder = SessionStateBuilder::new().with_default_features();
    let builder = if einfold {
        einfold_datafusion::enable(builder)
    } else {
        builder
    };
    SessionContext::new_with_state(builder.build())
}

fn register(ctx: &SessionContext, name: &str, batch: &RecordBatch) {
    let table = MemTable::try_new(batch.schema(), vec![vec![batch.clone()]]).unwrap();
    ctx.register_table(name, Arc::new(table)).unwrap();
}

/// Run `sql`; returns the result and whether the plan used `EinFoldExec`.
async fn run(ctx: &SessionContext, sql: &str) -> (RecordBatch, bool) {
    let df = ctx.sql(sql).await.unwrap();
    let schema = Arc::new(df.schema().as_arrow().clone());
    let plan = df.create_physical_plan().await.unwrap();
    let shown = displayable(plan.as_ref()).indent(false).to_string();
    let batches = collect(plan, ctx.task_ctx()).await.unwrap();
    let fired = shown.contains("EinFoldExec");
    (concat_batches(&schema, &batches).unwrap(), fired)
}

fn block_on<T>(f: impl std::future::Future<Output = T>) -> T {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(f)
}

fn run_case(case: &Case) -> (RecordBatch, bool) {
    block_on(async {
        let ctx = context(true);
        for (op, t) in case.fold.operands().iter().zip(&case.tables) {
            register(&ctx, &op.name, t);
        }
        run(&ctx, &case.sql).await
    })
}

/// 600 random two-operand folds, einfold on versus SQL.
#[test]
fn harness_two_operands() {
    let fired = std::cell::RefCell::new(BTreeMap::<String, usize>::new());
    check(600, 0xE1F0_1D00, |case| {
        let (out, f) = run_case(case);
        // Only an aggregate without GROUP BY may decline: SQL gives it a row
        // even when nothing joined, which a fold can't express yet.
        assert_eq!(f, !case.fold.output().is_empty(), "{case}");
        if f {
            *fired
                .borrow_mut()
                .entry(case.fold.aggregate().to_string())
                .or_default() += 1;
        }
        out
    });
    let fired = fired.into_inner();
    for agg in ["SUM", "COUNT", "AVG"] {
        let n = fired.get(agg).copied().unwrap_or(0);
        assert!(
            n >= 100,
            "EinFoldExec ran on only {n} {agg} cases: {fired:?}"
        );
    }
}

/// Three operands are not rewritten in M1, and still give SQL's result.
#[test]
fn harness_three_operands_unchanged() {
    for i in 0..100 {
        let case = Case::generate_with(einfold_testkit::case_seed(7, i), 3);
        let (out, fired) = run_case(&case);
        assert!(!fired, "{case}");
        assert_same_result(&sql_reference(&case), &out);
    }
}

fn tables(ctx: &SessionContext) {
    let ints = |v: &[Option<i64>]| Arc::new(Int64Array::from(v.to_vec()));
    let floats = |v: &[Option<f64>]| Arc::new(Float64Array::from(v.to_vec()));
    let batch = |names: &[&str], cols: Vec<datafusion::arrow::array::ArrayRef>| {
        let fields: Vec<Field> = names
            .iter()
            .zip(&cols)
            .map(|(n, c)| Field::new(*n, c.data_type().clone(), true))
            .collect();
        RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).unwrap()
    };
    let n = [Some(0), Some(1), None, Some(1), Some(2)];
    register(
        ctx,
        "a",
        &batch(
            &["i", "k", "v", "n"],
            vec![
                ints(&[Some(0), Some(0), Some(1), None, Some(1), Some(2)]),
                ints(&[Some(0), Some(1), None, Some(1), Some(1), Some(0)]),
                floats(&[Some(1.0), Some(2.0), Some(3.0), None, Some(f64::NAN), None]),
                ints(&[Some(0), Some(1), None, Some(1), Some(2), Some(0)]),
            ],
        ),
    );
    register(
        ctx,
        "b",
        &batch(
            &["k", "j", "v"],
            vec![
                ints(&[Some(0), Some(1), Some(1), None, Some(5)]),
                ints(&[Some(0), Some(0), Some(1), Some(1), None]),
                floats(&[Some(0.5), Some(-1.0), None, Some(4.0), Some(2.0)]),
            ],
        ),
    );
    register(
        ctx,
        "inp",
        &batch(
            &["n", "p", "x"],
            vec![
                ints(&n),
                ints(&[Some(0), Some(1), Some(0), Some(1), None]),
                floats(&[Some(1.0), Some(-2.0), Some(0.5), None, Some(3.0)]),
            ],
        ),
    );
    register(
        ctx,
        "act",
        &batch(
            &["n", "o", "z", "g"],
            vec![
                ints(&[Some(0), None, Some(1), Some(1), Some(3)]),
                ints(&[Some(0), Some(0), Some(1), None, Some(1)]),
                floats(&[Some(0.1), Some(0.2), Some(-0.3), Some(0.4), None]),
                floats(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)]),
            ],
        ),
    );
    register(
        ctx,
        "w",
        &batch(
            &["p", "o", "v"],
            vec![
                ints(&[Some(0), Some(0), Some(1), Some(1), None, Some(2)]),
                ints(&[Some(0), Some(1), Some(0), Some(1), None, Some(2)]),
                floats(&[Some(0.0); 6]),
            ],
        ),
    );
}

/// Runs `sql` with einfold on and off, asserts the same result, and returns
/// whether einfold rewrote it.
async fn same_on_and_off(sql: &str) -> bool {
    let (on, off) = (context(true), context(false));
    tables(&on);
    tables(&off);
    let (want, fired_off) = run(&off, sql).await;
    let (got, fired) = run(&on, sql).await;
    assert!(!fired_off);
    assert_eq!(got.schema(), want.schema());
    assert_same_result(&want, &got);
    fired
}

#[tokio::test]
async fn explain_shows_the_fold() {
    let sql = "SELECT a.i, b.j, SUM(a.v * b.v) AS s FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j";
    assert!(same_on_and_off(sql).await);
    let ctx = context(true);
    tables(&ctx);
    let explain = ctx.sql(&format!("EXPLAIN {sql}")).await.unwrap();
    let shown =
        datafusion::arrow::util::pretty::pretty_format_batches(&explain.collect().await.unwrap())
            .unwrap()
            .to_string();
    assert!(
        shown.contains("FoldNode: SUM(a[i,k] · b[k,j]) -> [i,j]"),
        "{shown}"
    );
    assert!(
        shown.contains("EinFoldExec: SUM(a[i,k] · b[k,j]) -> [i,j]"),
        "{shown}"
    );
}

#[tokio::test]
async fn null_keys_and_values() {
    // `=` drops NULL keys, IS NOT DISTINCT FROM joins them; NULL and NaN values.
    for sql in [
        "SELECT a.i, b.j, SUM(a.v * b.v) FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j",
        "SELECT a.i, b.j, SUM(a.v * b.v) FROM a JOIN b ON a.k IS NOT DISTINCT FROM b.k \
         GROUP BY a.i, b.j",
        "SELECT b.j, SUM(a.v) FROM a JOIN b ON a.k = b.k WHERE b.v > 0 GROUP BY b.j",
        "SELECT a.i, SUM(a.v * b.v * 2.0) FROM a CROSS JOIN b GROUP BY a.i",
    ] {
        assert!(same_on_and_off(sql).await, "{sql}");
    }
}

#[tokio::test]
async fn two_group_columns_one_dimension() {
    let sql = "SELECT a.k, b.k, SUM(a.v * b.v) FROM a JOIN b ON a.k = b.k GROUP BY a.k, b.k";
    assert!(same_on_and_off(sql).await);
}

#[tokio::test]
async fn result_above_the_aggregate_is_kept() {
    let sql = "SELECT j, s * 2 FROM (SELECT b.j, SUM(a.v * b.v) AS s FROM a JOIN b \
               ON a.k = b.k GROUP BY b.j) t WHERE s IS NOT NULL ORDER BY j";
    assert!(same_on_and_off(sql).await);
}

/// As ddx writes a gradient: the factor is computed in a projection, joins use
/// IS NOT DISTINCT FROM, and the gradient is left-joined back onto the weight
/// table so that weights no row reached get 0.
#[tokio::test]
async fn ddx_shaped_gradient() {
    let sql = "SELECT w.p, w.o, COALESCE(gr.v, 0.0) AS grad FROM w LEFT JOIN ( \
                 SELECT p, o, SUM(prod) AS v FROM ( \
                   SELECT inp.p AS p, act.o AS o, inp.x * act.g * (1.0 - act.z * act.z) AS prod \
                   FROM inp JOIN act ON inp.n IS NOT DISTINCT FROM act.n) t \
                 GROUP BY p, o) gr \
               ON (w.p IS NOT DISTINCT FROM gr.p) AND (w.o IS NOT DISTINCT FROM gr.o)";
    assert!(same_on_and_off(sql).await);
}

#[tokio::test]
async fn non_matching_plans_are_unchanged() {
    for sql in [
        "SELECT a.i, SUM(a.v * b.v) FROM a LEFT JOIN b ON a.k = b.k GROUP BY a.i",
        "SELECT a.i, SUM(a.k) FROM a GROUP BY a.i",
        "SELECT a.i, SUM(a.v * b.v * inp.x) FROM a JOIN b ON a.k = b.k \
         JOIN inp ON b.j = inp.p GROUP BY a.i",
        "SELECT a.i, SUM(a.v) FROM a GROUP BY a.i",
        "SELECT SUM(a.v * b.v) FROM a JOIN b ON a.k = b.k",
        "SELECT a.i, SUM(a.v + b.v) FROM a JOIN b ON a.k = b.k GROUP BY a.i",
        "SELECT a.i, MAX(a.v * b.v) FROM a JOIN b ON a.k = b.k GROUP BY a.i",
        "SELECT a.i, COUNT(DISTINCT b.j) FROM a JOIN b ON a.k = b.k GROUP BY a.i",
        "SELECT COUNT(*) FROM a JOIN b ON a.k = b.k",
    ] {
        let mut plans = Vec::new();
        for einfold in [true, false] {
            let ctx = context(einfold);
            tables(&ctx);
            let df = ctx.sql(sql).await.unwrap();
            plans.push(
                df.into_optimized_plan()
                    .unwrap()
                    .display_indent()
                    .to_string(),
            );
        }
        assert_eq!(plans[0], plans[1], "{sql}");
        assert!(!same_on_and_off(sql).await, "{sql}");
    }
}

#[tokio::test]
async fn count_and_avg() {
    // Group `a.i = 2` joins, but its only `a.v` is NULL: its AVG is NULL and
    // its COUNT(a.v) is 0, while COUNT(*) counts its rows. NaN propagates.
    for sql in [
        "SELECT a.i, COUNT(*) FROM a JOIN b ON a.k = b.k GROUP BY a.i",
        "SELECT a.i, COUNT(*) AS c FROM a JOIN b ON a.k IS NOT DISTINCT FROM b.k GROUP BY a.i",
        "SELECT a.i, COUNT(a.v * b.v) FROM a JOIN b ON a.k = b.k GROUP BY a.i",
        "SELECT b.j, COUNT(a.v) FROM a JOIN b ON a.k IS NOT DISTINCT FROM b.k GROUP BY b.j",
        "SELECT a.i, AVG(a.v * b.v) FROM a JOIN b ON a.k = b.k GROUP BY a.i",
        "SELECT b.j, AVG(a.v) AS m FROM a JOIN b ON a.k IS NOT DISTINCT FROM b.k GROUP BY b.j",
        "SELECT inp.p, AVG(inp.x * act.g) FROM inp JOIN act ON inp.n = act.n GROUP BY inp.p",
    ] {
        assert!(same_on_and_off(sql).await, "{sql}");
    }
    // COUNT is a non-NULL Int64 with einfold too.
    let ctx = context(true);
    tables(&ctx);
    let (out, fired) = run(
        &ctx,
        "SELECT a.i, COUNT(*) FROM a JOIN b ON a.k = b.k GROUP BY a.i",
    )
    .await;
    assert!(fired);
    let f = out.schema().field(1).clone();
    assert_eq!((f.data_type(), f.is_nullable()), (&DataType::Int64, false));
}

/// Detection moves `+`, `-` and `*` on integers onto a table's rows, on the
/// grounds that they cannot fail: DataFusion wraps integer overflow instead of
/// raising an error. If a DataFusion upgrade changes that, this test fails,
/// and that assumption must be revisited.
#[tokio::test]
async fn integer_arithmetic_wraps_on_overflow() {
    let ctx = SessionContext::new();
    let x = Int64Array::from(vec![i64::MAX, i64::MIN]);
    let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
    let batch = RecordBatch::try_new(schema, vec![Arc::new(x)]).unwrap();
    register(&ctx, "t", &batch);
    let df = ctx
        .sql("SELECT x + 1, x - 1, x * 2, -x FROM t")
        .await
        .unwrap();
    let out = concat_batches(
        &Arc::new(df.schema().as_arrow().clone()),
        &df.collect()
            .await
            .expect("integer overflow raised an error"),
    )
    .unwrap();
    let col = |c: usize| {
        let a = out.column(c).as_any().downcast_ref::<Int64Array>().unwrap();
        (a.value(0), a.value(1))
    };
    assert_eq!(col(0), (i64::MIN, i64::MIN + 1));
    assert_eq!(col(1), (i64::MAX - 1, i64::MAX));
    assert_eq!(col(2), (-2, 0));
    assert_eq!(col(3), (i64::MIN + 1, i64::MIN));
}
