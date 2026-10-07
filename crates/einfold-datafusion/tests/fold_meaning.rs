// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The invariant behind detection: a detected fold means exactly what the
//! plan means.
//!
//! For each query and each of many seeded random tables, this test runs the
//! `Aggregate` node in DataFusion, and separately evaluates the detected
//! `FoldMatch` by brute force: run each operand's bound plan, try every
//! combination of one row per operand, keep the combinations whose dimension
//! columns match (under `=` or `IS NOT DISTINCT FROM`, as the fold records),
//! multiply the factors, and aggregate per group. The two must give the same
//! groups and the same values.
//!
//! The tables have NULL keys (which `=` drops and `IS NOT DISTINCT FROM`
//! joins), NULL and NaN values, duplicate key tuples, empty tables, and groups
//! whose values are all NULL (`SUM` and `AVG` give NULL, `COUNT` gives 0).

use std::collections::BTreeMap;
use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, Float64Array, Int64Array};
use datafusion::arrow::compute::concat_batches;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::common::ScalarValue;
use datafusion::datasource::MemTable;
use datafusion::logical_expr::{Expr, LogicalPlan, LogicalPlanBuilder};
use datafusion::prelude::SessionContext;
use einfold_datafusion::detect::{detect, FoldMatch};
use einfold_ir::{Aggregate, Dim, KeyEquality};

/// splitmix64, so the tables are stable across dependency versions.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Tables `a(i, k, v)`, `b(k, j, v, n)`, `c(j, l, v)` with random rows.
fn context(seed: u64) -> SessionContext {
    let mut rng = Rng(seed);
    let ctx = SessionContext::new();
    let tables: [(&str, &[&str]); 3] = [
        ("a", &["i", "k", "v"]),
        ("b", &["k", "j", "v", "n"]),
        ("c", &["j", "l", "v"]),
    ];
    // Per run: how often keys and values are NULL, so that some groups have
    // only NULL values.
    let key_null = [0, 20, 50][rng.below(3) as usize];
    let v_null = [0, 30, 90][rng.below(3) as usize];
    for (name, cols) in tables {
        let rows = if rng.below(10) == 0 { 0 } else { rng.below(8) };
        let mut fields = Vec::new();
        let mut arrays: Vec<ArrayRef> = Vec::new();
        for c in cols {
            if *c == "v" {
                let v: Float64Array = (0..rows)
                    .map(|_| match rng.below(100) {
                        p if p < v_null => None,
                        p if p >= 97 => Some(f64::NAN),
                        _ => Some((rng.below(9) as f64 - 4.0) / 2.0),
                    })
                    .collect();
                fields.push(Field::new(*c, DataType::Float64, true));
                arrays.push(Arc::new(v));
            } else {
                let k: Int64Array = (0..rows)
                    .map(|_| (rng.below(100) >= key_null).then(|| rng.below(3) as i64))
                    .collect();
                fields.push(Field::new(*c, DataType::Int64, true));
                arrays.push(Arc::new(k));
            }
        }
        let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), arrays).unwrap();
        let table = MemTable::try_new(batch.schema(), vec![vec![batch]]).unwrap();
        ctx.register_table(name, Arc::new(table)).unwrap();
    }
    ctx
}

const QUERIES: &[&str] = &[
    "SELECT a.i, b.j, SUM(a.v * b.v) FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j",
    "SELECT a.i, b.j, SUM(a.v * b.v) FROM a JOIN b ON a.k IS NOT DISTINCT FROM b.k \
     GROUP BY a.i, b.j",
    "SELECT a.i, COUNT(*) FROM a JOIN b ON a.k = b.k GROUP BY a.i",
    "SELECT b.j, COUNT(*) FROM a JOIN b ON a.k IS NOT DISTINCT FROM b.k GROUP BY b.j",
    "SELECT a.i, COUNT(a.v * b.v) FROM a JOIN b ON a.k = b.k GROUP BY a.i",
    "SELECT a.i, COUNT(b.n * a.k) FROM a JOIN b ON a.k = b.k GROUP BY a.i",
    "SELECT a.i, AVG(a.v * b.v) FROM a JOIN b ON a.k = b.k GROUP BY a.i",
    "SELECT b.j, AVG(a.v) FROM a JOIN b ON a.k IS NOT DISTINCT FROM b.k GROUP BY b.j",
    "SELECT a.i, c.l, SUM(a.v * b.v * c.v) FROM a JOIN b ON a.k = b.k \
     JOIN c ON b.j = c.j GROUP BY a.i, c.l",
    "SELECT a.i, SUM(a.v * 2.0 * b.v) FROM a JOIN b ON a.k = b.k \
     WHERE a.v > -1.5 AND b.j IS NOT NULL GROUP BY a.i",
    "SELECT a.i, SUM(a.v) FROM a WHERE a.i = a.k GROUP BY a.i",
    "SELECT a.k, b.k, AVG(a.v * b.v) FROM a JOIN b ON a.k = b.k GROUP BY a.k, b.k",
    "SELECT a.i, COUNT(*) FROM a CROSS JOIN c GROUP BY a.i",
    "SELECT i, j, SUM(p) FROM (SELECT a.i AS i, b.j AS j, \
     a.v * b.v * (1.0 - b.v * b.v) AS p FROM a JOIN b ON a.k = b.k) t GROUP BY i, j",
];

/// One group: its output coordinates, and the aggregate's value.
type Group = (Vec<ScalarValue>, ScalarValue);

async fn collect(ctx: &SessionContext, plan: LogicalPlan) -> RecordBatch {
    let df = ctx.execute_logical_plan(plan).await.unwrap();
    let schema = Arc::new(df.schema().as_arrow().clone());
    concat_batches(&schema, &df.collect().await.unwrap()).unwrap()
}

fn scalar(batch: &RecordBatch, c: usize, r: usize) -> ScalarValue {
    ScalarValue::try_from_array(batch.column(c), r).unwrap()
}

/// Whether the values of one dimension in a candidate joined row all match.
fn keys_match(keys: &[&ScalarValue], eq: KeyEquality) -> bool {
    let strict = eq == KeyEquality::Equal;
    keys.iter()
        .all(|k| *k == keys[0] && !(strict && k.is_null()))
}

/// Evaluate the fold by brute force over its operands' bound plans.
async fn brute_force(ctx: &SessionContext, m: &FoldMatch) -> Vec<Group> {
    // Each operand's rows: (dimension values, factor).
    let mut operands = Vec::new();
    for input in &m.operands {
        let mut exprs: Vec<Expr> = input
            .dim_columns
            .iter()
            .cloned()
            .map(Expr::Column)
            .collect();
        exprs.push(input.value.clone().alias("__factor"));
        let plan = LogicalPlanBuilder::from(input.plan.as_ref().clone())
            .project(exprs)
            .unwrap()
            .build()
            .unwrap();
        let batch = collect(ctx, plan).await;
        let n = input.dim_columns.len();
        let rows: Vec<(Vec<ScalarValue>, Option<f64>)> = (0..batch.num_rows())
            .map(|r| {
                let dims = (0..n).map(|c| scalar(&batch, c, r)).collect();
                let v = batch
                    .column(n)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .unwrap();
                (dims, v.is_valid(r).then(|| v.value(r)))
            })
            .collect();
        operands.push(rows);
    }
    let fold = &m.fold;
    // Per group: (sum of non-NULL products, how many there were).
    let mut groups: BTreeMap<Vec<String>, (Vec<ScalarValue>, Option<f64>, i64)> = BTreeMap::new();
    let mut pick = vec![0usize; operands.len()];
    if operands.iter().any(|o| o.is_empty()) {
        return Vec::new();
    }
    loop {
        let rows: Vec<&(Vec<ScalarValue>, Option<f64>)> =
            pick.iter().zip(&operands).map(|(&p, o)| &o[p]).collect();
        let values_of = |d: &Dim| -> Vec<&ScalarValue> {
            fold.operands()
                .iter()
                .zip(&rows)
                .flat_map(|(op, row)| {
                    op.dims
                        .iter()
                        .zip(&row.0)
                        .filter(move |(od, _)| *od == d)
                        .map(|(_, v)| v)
                })
                .collect()
        };
        let joins = fold
            .dims()
            .iter()
            .all(|d| keys_match(&values_of(d), fold.equality(d).unwrap()));
        if joins {
            let key: Vec<ScalarValue> = fold
                .output()
                .iter()
                .map(|d| values_of(d)[0].clone())
                .collect();
            let product = rows.iter().try_fold(1.0, |acc, r| r.1.map(|v| acc * v));
            let g = groups
                .entry(key.iter().map(|k| format!("{k:?}")).collect())
                .or_insert((key, None, 0));
            if let Some(p) = product {
                g.1 = Some(g.1.unwrap_or(0.0) + p);
                g.2 += 1;
            }
        }
        // Next combination.
        let mut i = 0;
        loop {
            if i == pick.len() {
                return finish(fold.aggregate(), groups);
            }
            pick[i] += 1;
            if pick[i] < operands[i].len() {
                break;
            }
            pick[i] = 0;
            i += 1;
        }
    }
}

fn finish(
    aggregate: Aggregate,
    groups: BTreeMap<Vec<String>, (Vec<ScalarValue>, Option<f64>, i64)>,
) -> Vec<Group> {
    groups
        .into_values()
        .map(|(key, sum, count)| {
            let value = match aggregate {
                Aggregate::SUM => ScalarValue::Float64(sum),
                Aggregate::COUNT => ScalarValue::Int64(Some(count)),
                Aggregate::AVG => ScalarValue::Float64(sum.map(|s| s / count as f64)),
                other => panic!("unexpected aggregate {other}"),
            };
            (key, value)
        })
        .collect()
}

/// The `Aggregate`'s own result, keyed by the fold's output dimensions.
async fn by_sql(ctx: &SessionContext, agg: &LogicalPlan, m: &FoldMatch) -> Vec<Group> {
    let batch = collect(ctx, agg.clone()).await;
    let n = m.group_outputs.len();
    (0..batch.num_rows())
        .map(|r| {
            let key = m
                .fold
                .output()
                .iter()
                .map(|d| {
                    let cols: Vec<usize> = (0..n).filter(|&c| m.group_outputs[c].1 == *d).collect();
                    // Group columns that hold one dimension are equal.
                    for &c in &cols {
                        assert_eq!(scalar(&batch, c, r), scalar(&batch, cols[0], r));
                    }
                    scalar(&batch, cols[0], r)
                })
                .collect();
            (key, scalar(&batch, n, r))
        })
        .collect()
}

fn same_value(a: &ScalarValue, b: &ScalarValue) -> bool {
    match (a, b) {
        (ScalarValue::Float64(Some(x)), ScalarValue::Float64(Some(y))) => {
            x == y || (x.is_nan() && y.is_nan()) || (x - y).abs() <= 1e-9 * x.abs().max(y.abs())
        }
        _ => a == b,
    }
}

fn find_aggregate(plan: &LogicalPlan) -> LogicalPlan {
    let mut found = None;
    plan.apply(|p| {
        if matches!(p, LogicalPlan::Aggregate(_)) {
            found = Some(p.clone());
            return Ok(TreeNodeRecursion::Stop);
        }
        Ok(TreeNodeRecursion::Continue)
    })
    .unwrap();
    found.unwrap()
}

#[tokio::test]
async fn detected_folds_mean_what_the_plan_means() {
    let seed = 0xF01D;
    let mut checked = 0;
    for run in 0..40 {
        let case_seed = seed + run;
        let ctx = context(case_seed);
        for sql in QUERIES {
            let plan = ctx.sql(sql).await.unwrap().into_optimized_plan().unwrap();
            let agg = find_aggregate(&plan);
            let m = detect(&agg).unwrap_or_else(|| panic!("not detected: {sql}"));
            let mut want = by_sql(&ctx, &agg, &m).await;
            let mut got = brute_force(&ctx, &m).await;
            let order = |g: &Group| format!("{:?}", g.0);
            want.sort_by_key(order);
            got.sort_by_key(order);
            let fail = |why: &str| {
                panic!(
                    "{why} for seed {case_seed}\n{sql}\nfold: {}\nsql: {want:?}\nbrute force: {got:?}",
                    m.fold
                )
            };
            if want.len() != got.len() {
                fail("group counts differ");
            }
            for (w, g) in want.iter().zip(&got) {
                if w.0 != g.0 || !same_value(&w.1, &g.1) {
                    fail("results differ");
                }
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 40 * QUERIES.len());
}
