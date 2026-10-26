// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Property: with einfold installed, DataFusion returns exactly what it
//! returns without, for random queries over random tables; and `EinFoldRule`
//! finds the folds it claims to and leaves the rest alone.
//!
//! Each case draws two operand tables `l` and `r`, and a domain table `d`,
//! with `BIGINT` or `VARCHAR` keys (NULL keys, duplicate tuples, missing rows,
//! keys too large or negative for the dense algorithm) and `DOUBLE` values that
//! are NULL, NaN, ±inf, `-0.0` or small integers, split across partitions.
//! Every sum is then exact in any order, so results compare exactly. The
//! queries are written in several spellings:
//!
//! - **folds** (`SUM` of a product over an inner equi-join, grouped by columns
//!   of either side, with `=` or `IS NOT DISTINCT FROM`, factors that are
//!   expressions, aliases), which the rule must replace;
//! - **fills** (ddx's left join from `d` onto a fold, with its `CASE`), which
//!   the rule must turn into a fold with a domain;
//! - **look-alikes** it must leave alone: a mask (`AND l.a >= r.b`), no
//!   `GROUP BY`, `AVG` rather than `SUM`, and a sum that isn't a product;
//!   and fills' look-alikes, in which it must find the fold but give it no
//!   domain: a projection that reads the fold's keys, and inner and right
//!   joins.
//!
//! Invariants: the same rows, as a multiset, bit for bit (every NaN alike), as
//! without einfold; the rule fires on every fold and fill, and never on a
//! look-alike. Half the cases run every fold the rule finds through einfold's
//! own kernels (`EinFoldRule::without_fallback`), so that the folds the rule
//! describes are checked even where the dense algorithm doesn't apply.
//!
//! Reproduce a failure with `EINFOLD_SEED=<seed printed>`; search longer with
//! `EINFOLD_CASES=<n>`.

use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::util::display::array_value_to_string;
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::{SessionConfig, SessionContext};
use einfold_datafusion::{EinFoldQueryPlanner, EinFoldRule, FoldNode};

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        Rng((z ^ (z >> 31)).max(1))
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn chance(&mut self, p: f64) -> bool {
        ((self.next() >> 11) as f64) / ((1u64 << 53) as f64) < p
    }
}

/// How keys look: dense non-negative integers, integers the dense algorithm
/// can't use, or text.
#[derive(Clone, Copy)]
enum Keys {
    Dense,
    Wild,
    Text,
}

/// A table `name (a, b, v)`: keys `a` over `0..ea` and `b` over `0..eb`, of
/// the given kind, with some rows missing, and, if `messy`, some duplicated
/// and some NULL keys.
fn table(
    rng: &mut Rng,
    ctx: &[&SessionContext],
    name: &str,
    [ea, eb]: [usize; 2],
    keys: Keys,
    messy: bool,
) {
    let present = if messy || rng.chance(0.4) { 0.85 } else { 1.0 };
    let (mut a, mut b, mut v) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..ea {
        for j in 0..eb {
            if !rng.chance(present) {
                continue;
            }
            for _ in 0..if messy && rng.chance(0.1) { 2 } else { 1 } {
                let key = |k: usize, rng: &mut Rng| -> Option<i64> {
                    if messy && rng.chance(0.06) {
                        return None;
                    }
                    Some(match keys {
                        Keys::Wild if k == 0 => -7,
                        _ => k as i64,
                    })
                };
                a.push(key(i, rng));
                b.push(key(j, rng));
                v.push(match rng.below(20) {
                    0 => None,
                    1 => Some(f64::NAN),
                    2 => Some(f64::INFINITY),
                    3 => Some(-0.0),
                    _ => Some((rng.below(7) as f64) - 3.0),
                });
            }
        }
    }
    let key_type = match keys {
        Keys::Text => DataType::Utf8,
        _ => DataType::Int64,
    };
    let column = |k: Vec<Option<i64>>| -> ArrayRef {
        match keys {
            Keys::Text => Arc::new(
                k.into_iter()
                    .map(|x| x.map(|x| format!("k{x}")))
                    .collect::<StringArray>(),
            ),
            _ => Arc::new(Int64Array::from(k)),
        }
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", key_type.clone(), true),
        Field::new("b", key_type, true),
        Field::new("v", DataType::Float64, true),
    ]));
    let batch = RecordBatch::try_new(
        Arc::clone(&schema),
        vec![column(a), column(b), Arc::new(Float64Array::from(v))],
    )
    .unwrap();
    let parts = 1 + rng.below(3);
    let mut partitions = vec![Vec::new(); parts];
    let mut start = 0;
    while start < batch.num_rows() {
        let len = (1 + rng.below(6)).min(batch.num_rows() - start);
        partitions[rng.below(parts)].push(batch.slice(start, len));
        start += len;
    }
    for c in ctx {
        let t = MemTable::try_new(Arc::clone(&schema), partitions.clone()).unwrap();
        c.register_table(name, Arc::new(t)).unwrap();
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum Expect {
    /// A fold, with no domain.
    Fold,
    /// A fold with a domain.
    Fill,
    /// A fill's look-alike: the fold is found, but it gets no domain.
    FoldOnly,
    /// No fold at all.
    Decline,
}

/// A random query over `l`, `r` and `d`, and whether the rule must fire.
fn query(rng: &mut Rng) -> (String, Expect) {
    let eq = |x: &str, y: &str, safe: bool| match safe {
        true => format!("({x} IS NOT DISTINCT FROM {y})"),
        false => format!("{x} = {y}"),
    };
    let safe = rng.chance(0.5);
    // Join on l.b = r.a, sometimes also l.a = r.b.
    let mut on = vec![eq("l.b", "r.a", safe)];
    let two_keys = rng.chance(0.3);
    if two_keys {
        on.push(eq("l.a", "r.b", safe));
    }
    let factor = |t: &str, rng: &mut Rng| match rng.below(3) {
        0 => format!("{t}.v"),
        1 => format!("({t}.v * 2.0)"),
        _ => format!("({t}.v + 1.0)"),
    };
    let (fl, fr) = (factor("l", rng), factor("r", rng));
    // Group keys: l.a and r.b, or one of them, or the join key.
    let groups: Vec<&str> = match rng.below(4) {
        0 => vec!["l.a", "r.b"],
        1 => vec!["l.a"],
        2 => vec!["r.b", "l.a"],
        _ => vec!["l.b"],
    };
    let groups = if two_keys { vec!["l.b"] } else { groups };
    let select: Vec<String> = groups
        .iter()
        .enumerate()
        .map(|(i, g)| format!("{g} AS g{i}"))
        .collect();
    let group = groups.join(", ");
    let fold = |aggregate: &str, mask: bool| {
        let mask = if mask { " AND l.a >= r.b" } else { "" };
        format!(
            "SELECT {}, {aggregate} AS v FROM l JOIN r ON {}{mask} GROUP BY {group}",
            select.join(", "),
            on.join(" AND ")
        )
    };
    let product = format!("SUM({fl} * {fr})");
    match rng.below(8) {
        0..=2 => (fold(&product, false), Expect::Fold),
        3..=4 if groups.len() == 2 => {
            // ddx's fill: every row of d gets its group's sum, 0 where none.
            // Its look-alikes, which keep the fold but must not become a
            // domain: a projection that reads the fold's keys, or a join
            // that isn't a left join from d.
            let safe = rng.chance(0.7);
            let keys = [eq("d.a", "s.g0", safe), eq("d.b", "s.g1", safe)].join(" AND ");
            let (reads, join) = match rng.below(6) {
                0 => ("s.g0", "LEFT JOIN"),
                1 => ("d.b", "JOIN"),
                2 => ("d.b", "RIGHT JOIN"),
                _ => ("d.b", "LEFT JOIN"),
            };
            let expect = match (reads, join) {
                ("d.b", "LEFT JOIN") => Expect::Fill,
                _ => Expect::FoldOnly,
            };
            (
                format!(
                    "SELECT d.a, {reads}, CASE WHEN d.v IS NULL THEN NULL WHEN s.v IS NULL THEN 0.0 \
                     ELSE s.v END AS v FROM d {join} ({}) s ON {keys}",
                    fold(&product, false)
                ),
                expect,
            )
        }
        3..=4 => (fold(&product, false), Expect::Fold),
        5 => (fold(&product, true), Expect::Decline),
        6 => (
            [
                format!(
                    "SELECT {product} AS v FROM l JOIN r ON {}",
                    on.join(" AND ")
                ),
                fold(&format!("AVG({fl} * {fr})"), false),
            ][rng.below(2)]
            .clone(),
            Expect::Decline,
        ),
        _ => (fold(&format!("SUM({fl} + {fr})"), false), Expect::Decline),
    }
}

fn host(threads: usize) -> SessionContext {
    SessionContext::new_with_config(SessionConfig::new().with_target_partitions(threads))
}

/// DataFusion with einfold. With `fallback` false, every fold the rule finds
/// runs through einfold's kernels, the hash algorithm included, rather than
/// through the host's plan where the dense algorithm doesn't apply.
fn einfold(threads: usize, fallback: bool) -> SessionContext {
    let config = SessionConfig::new().with_target_partitions(threads);
    let rule = match fallback {
        true => EinFoldRule::default(),
        false => EinFoldRule::without_fallback(),
    };
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(config)
        .with_optimizer_rule(Arc::new(rule))
        .with_query_planner(Arc::new(EinFoldQueryPlanner))
        .build();
    SessionContext::new_with_state(state)
}

/// How many `Fold` nodes a plan has, and how many of them have a domain.
fn folds(plan: &LogicalPlan) -> (usize, usize) {
    let (mut all, mut domains) = (0, 0);
    if let LogicalPlan::Extension(e) = plan {
        if let Some(node) = e.node.as_any().downcast_ref::<FoldNode>() {
            all += 1;
            domains += usize::from(node.fold().domain().is_some());
        }
    }
    for input in plan.inputs() {
        let (a, d) = folds(input);
        all += a;
        domains += d;
    }
    (all, domains)
}

/// Every row, printed with float bits so `-0.0` and each NaN differ, sorted.
fn rows(batches: &[RecordBatch]) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        for r in 0..b.num_rows() {
            let row: Vec<String> = b
                .columns()
                .iter()
                .map(|c| match c.as_any().downcast_ref::<Float64Array>() {
                    // NaN's sign and payload depend on the order of
                    // additions, even within the host, and SQL doesn't tell
                    // them apart.
                    Some(f) if f.is_valid(r) && f.value(r).is_nan() => "NaN".into(),
                    Some(f) if f.is_valid(r) => format!("{:x}", f.value(r).to_bits()),
                    _ => array_value_to_string(c, r).unwrap(),
                })
                .collect();
            out.push(row.join("|"));
        }
    }
    out.sort();
    out
}

#[tokio::test]
async fn einfold_changes_no_answer() {
    let seed: u64 = std::env::var("EINFOLD_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xde7e_c7ed);
    let cases: u64 = match std::env::var("EINFOLD_CASES") {
        Ok(n) => n.parse().unwrap(),
        Err(_) if std::env::var("EINFOLD_SEED").is_ok() => 1,
        Err(_) => 300,
    };
    let (mut fired, mut dense, mut host_plans) = (0, 0, 0);
    for i in 0..cases {
        let seed = seed.wrapping_add(i);
        let mut rng = Rng::new(seed);
        let keys = [Keys::Dense, Keys::Dense, Keys::Wild, Keys::Text][rng.below(4)];
        // Dense keys are sometimes clean enough for the dense algorithm.
        let messy = !matches!(keys, Keys::Dense) || rng.chance(0.4);
        let fallback = rng.chance(0.5);
        let (ours, theirs) = (einfold(1 + rng.below(6), fallback), host(1 + rng.below(6)));
        let ctxs = [&ours, &theirs];
        let ext = |rng: &mut Rng| 1 + rng.below(5);
        let (ea, eb, ec) = (ext(&mut rng), ext(&mut rng), ext(&mut rng));
        table(&mut rng, &ctxs, "l", [ea, eb], keys, messy);
        table(&mut rng, &ctxs, "r", [eb, ec], keys, messy);
        table(&mut rng, &ctxs, "d", [ea + 1, ec + 1], keys, messy);
        let (sql, expect) = query(&mut rng);
        let fail = |what: String| panic!("seed {seed}: {what}\nSQL: {sql}");

        let df = ours.sql(&sql).await.unwrap();
        let plan = df.clone().into_optimized_plan().unwrap();
        let (found, domains) = folds(&plan);
        let ok = match expect {
            Expect::Fold | Expect::FoldOnly => found == 1 && domains == 0,
            Expect::Fill => found == 1 && domains == 1,
            Expect::Decline => found == 0,
        };
        if !ok {
            fail(format!(
                "expected {expect:?}, found {found} folds, {domains} with a domain:\n{}",
                plan.display_indent()
            ));
        }
        fired += usize::from(found > 0);
        let physical = df.clone().create_physical_plan().await.unwrap();
        let got = datafusion::physical_plan::collect(Arc::clone(&physical), ours.task_ctx())
            .await
            .unwrap();
        let shown = datafusion::physical_plan::display::DisplayableExecutionPlan::with_metrics(
            physical.as_ref(),
        )
        .indent(true)
        .to_string();
        dense += usize::from(shown.contains("dense_folds=1"));
        host_plans += usize::from(shown.contains("host_plans=1"));
        let want = theirs.sql(&sql).await.unwrap().collect().await.unwrap();
        let (got, want) = (rows(&got), rows(&want));
        if got != want {
            fail(format!("einfold gave\n{got:#?}\nthe host gave\n{want:#?}"));
        }
    }
    eprintln!(
        "the rule fired in {fired} of {cases} cases; the dense path ran in {dense}, the host's plan in {host_plans}"
    );
    // Most spellings rule the dense path out (a group key that is a join key,
    // a summed key held by one side, keys that can't be positions), and the
    // kernels' own test covers it. Here both it and the host's plan must run.
    if cases >= 100 {
        let some = |n: usize| n * 50 >= cases as usize;
        assert!(some(dense), "the dense path ran in only {dense} cases");
        assert!(
            some(host_plans),
            "the host's plan ran in only {host_plans} cases"
        );
    }
}
