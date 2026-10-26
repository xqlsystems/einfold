// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Property: with einfold installed, DataFusion returns exactly what it
//! returns without, for random queries over random tables; and `DetectFolds`
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
//! - **folds from a domain** (ddx's left join from `d` onto a fold, with its
//!   `CASE`), which the rule must turn into a fold with `d` as its domain;
//! - **look-alikes** it must leave alone: a mask (`AND l.a >= r.b`), no
//!   `GROUP BY`, `AVG` rather than `SUM`, and a sum that isn't a product;
//!   and look-alikes of a fold from a domain, in which it must find the fold
//!   but give it no domain: a projection that reads the fold's keys, and
//!   inner and right joins;
//! - **folds joined back to their domain** (ddx's gradient step: the fold
//!   joined to `d`, grouped again by `d`'s keys, and left-joined from `d`),
//!   which the rule must turn into a fold from `d` that checks `d`'s keys are
//!   unique when it runs, and their look-alikes, which keep the fold alone.
//!
//! Invariants: the same rows, as a multiset, bit for bit (every NaN alike), as
//! without einfold; the rule fires on every fold, fold from a domain and fold
//! joined back, and never on a look-alike. A second test draws only folds
//! joined back, over operands the dense algorithm takes and a domain whose
//! keys are duplicated half the time, so that both outcomes of the
//! uniqueness check are exercised. Half the cases run every fold the rule finds through einfold's
//! own kernels (`DetectFolds::without_fallback`), so that the folds the rule
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
use einfold_datafusion::{DetectFolds, EinFoldExec, EinFoldQueryPlanner, FoldNode};

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
    FromDomain,
    /// A look-alike of a fold from a domain: the fold is found, but it gets no domain.
    FoldOnly,
    /// A fold joined back to its domain and grouped again, as ddx writes a
    /// gradient: with the host's plan as a fallback, a fold from a domain
    /// whose keys must be unique; without one, the fold alone.
    JoinedBack,
    /// No fold at all.
    Decline,
}

/// A random query over `l`, `r` and `d`, and whether the rule must fire;
/// with `joined_back`, always a fold joined back to `d` or a look-alike.
fn query(rng: &mut Rng, joined_back: bool) -> (String, Expect) {
    let eq = |x: &str, y: &str, safe: bool| match safe {
        true => format!("({x} IS NOT DISTINCT FROM {y})"),
        false => format!("{x} = {y}"),
    };
    let safe = rng.chance(0.5);
    // Join on l.b = r.a, sometimes also l.a = r.b.
    let mut on = vec![eq("l.b", "r.a", safe)];
    let two_keys = rng.chance(0.3) && !joined_back;
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
        0 | 1 | 3 if joined_back => vec!["l.a", "r.b"],
        0 => vec!["l.a", "r.b"],
        1 => vec!["l.a"],
        2 => vec!["r.b", "l.a"],
        _ => vec!["l.b"],
    };
    let groups = if two_keys { vec!["l.b"] } else { groups };
    // The select list in an order of its own, so that the result's columns
    // aren't in GROUP BY's order.
    let mut select: Vec<String> = groups
        .iter()
        .enumerate()
        .map(|(i, g)| format!("{g} AS g{i}"))
        .collect();
    if rng.chance(0.5) {
        select.reverse();
    }
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
    let shape = rng.below(10);
    match if joined_back { 8 } else { shape } {
        0..=2 => (fold(&product, false), Expect::Fold),
        8..=9 if groups.len() == 2 => {
            // ddx's gradient step: the fold joined back to d, grouped again by
            // d's keys, then left-joined from d. Its look-alikes keep the fold
            // alone: a second GROUP BY by other columns, a sum that isn't NULL where
            // the fold's value is, a filtered d, a projection reading the
            // second GROUP BY's keys, and `=` outside `IS NOT DISTINCT FROM`.
            let (inner, outer) = match rng.below(8) {
                0 => (true, false),
                _ => {
                    let s = rng.chance(0.7);
                    (s, s)
                }
            };
            let look = rng.below(11);
            // The second GROUP BY's second column: the join's, or d's value.
            let col = if look == 10 { "v" } else { "b" };
            let inner_keys = [eq("d2.a", "f.g0", inner), eq("d2.b", "f.g1", inner)].join(" AND ");
            let outer_keys = [
                eq("d.a", "s.a", outer),
                eq(&format!("d.{col}"), "s.b", outer),
            ];
            let outer_keys = outer_keys.join(" AND ");
            let (group, sum, d2, reads) = match look {
                0 => ("d2.a, d2.b", "SUM(f.v * 2.0)", "d", "d.b"),
                1 => ("d2.a, d2.b", "SUM(COALESCE(f.v, 1.0))", "d", "d.b"),
                2 => (
                    "d2.a, d2.b",
                    "SUM(f.v)",
                    "(SELECT * FROM d WHERE d.v IS NOT NULL)",
                    "d.b",
                ),
                3 => ("d2.a, d2.b", "SUM(f.v)", "d", "s.b"),
                4 => ("d2.b, d2.a", "SUM(f.v)", "d", "d.b"),
                5 => ("d2.a, d2.b", "SUM(f.v)", "(SELECT a, b, v FROM d)", "d.b"),
                10 => ("d2.a, d2.v", "SUM(f.v)", "d", "d.b"),
                _ => (
                    "d2.a, d2.b",
                    "SUM(CASE WHEN d2.v IS NULL THEN NULL ELSE f.v END)",
                    "d",
                    "d.b",
                ),
            };
            let expect = match (look, inner, outer) {
                (0..=3 | 10, _, _) | (_, true, false) => Expect::FoldOnly,
                _ => Expect::JoinedBack,
            };
            let regroup = format!(
                "SELECT d2.a AS a, d2.{col} AS b, {sum} AS v FROM {d2} d2 JOIN ({}) f ON {inner_keys} \
                 GROUP BY {group}",
                fold(&product, false)
            );
            (
                format!(
                    "SELECT d.a, {reads}, CASE WHEN d.v IS NULL THEN NULL WHEN s.v IS NULL THEN 0.0 \
                     ELSE s.v END AS v FROM d LEFT JOIN ({regroup}) s ON {outer_keys}"
                ),
                expect,
            )
        }
        3..=4 if groups.len() == 2 => {
            // A fold from a domain, as ddx writes it: every row of d gets its group's sum, 0 where none.
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
                ("d.b", "LEFT JOIN") => Expect::FromDomain,
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
        3..=4 | 8..=9 => (fold(&product, false), Expect::Fold),
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
        true => DetectFolds::default(),
        false => DetectFolds::without_fallback(),
    };
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(config)
        .with_query_planner(Arc::new(EinFoldQueryPlanner::detecting(rule)))
        .build();
    SessionContext::new_with_state(state)
}

/// How many `EinFoldExec`s a physical plan has, and how many of them have a
/// domain.
fn exec_folds(plan: &Arc<dyn datafusion::physical_plan::ExecutionPlan>) -> (usize, usize) {
    let (mut all, mut domains) = (0, 0);
    let any: &dyn std::any::Any = plan.as_ref();
    if let Some(e) = any.downcast_ref::<EinFoldExec>() {
        all += 1;
        domains += usize::from(e.fold().domain().is_some());
    }
    for child in plan.children() {
        let (a, d) = exec_folds(child);
        all += a;
        domains += d;
    }
    (all, domains)
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
    let (cases, dense, host_plans, joined) = sweep(0xde7e_c7ed, false).await;
    // Most spellings rule the dense path out (a group key that is a join key,
    // a summed key held by one side, keys that can't be positions), and the
    // kernels' own test covers it. Here both it and the host's plan must run.
    if cases >= 100 {
        let some = |n: usize| n * 50 >= cases;
        assert!(some(dense), "the dense path ran in only {dense} cases");
        assert!(
            some(host_plans),
            "the host's plan ran in only {host_plans} cases"
        );
        assert!(joined.0 + joined.1 > 0, "no fold joined back ran");
    }
}

#[tokio::test]
async fn folds_joined_back_check_their_domain() {
    let (cases, _, _, (dense, host)) = sweep(0x10_1ced, true).await;
    if cases >= 100 {
        let some = |n: usize| n * 10 >= cases;
        assert!(
            some(dense) && some(host),
            "folds joined back ran dense in {dense} cases and on the host in {host}"
        );
    }
}

/// Runs the cases from `default_seed`, or as `EINFOLD_SEED` and
/// `EINFOLD_CASES` say, panicking at the first that fails; with
/// `joined_back`, only folds joined back to their domain. Returns how many
/// cases ran, how many ran the dense path and the host's plan, and how many
/// folds joined back ran each.
async fn sweep(default_seed: u64, joined_back: bool) -> (usize, usize, usize, (usize, usize)) {
    let seed: u64 = std::env::var("EINFOLD_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default_seed);
    let cases: u64 = match std::env::var("EINFOLD_CASES") {
        Ok(n) => n.parse().unwrap(),
        Err(_) if std::env::var("EINFOLD_SEED").is_ok() => 1,
        Err(_) => 300,
    };
    let (mut fired, mut dense, mut host_plans) = (0, 0, 0);
    let (mut joined_dense, mut joined_host) = (0, 0);
    for i in 0..cases {
        let seed = seed.wrapping_add(i);
        let mut rng = Rng::new(seed);
        let keys = [Keys::Dense, Keys::Dense, Keys::Wild, Keys::Text][rng.below(4)];
        // Dense keys are sometimes clean enough for the dense algorithm.
        let messy = !matches!(keys, Keys::Dense) || rng.chance(0.4);
        let fallback = rng.chance(0.5);
        let (keys, messy, fallback) = match joined_back {
            true => (Keys::Dense, false, true),
            false => (keys, messy, fallback),
        };
        let (ours, theirs) = (einfold(1 + rng.below(6), fallback), host(1 + rng.below(6)));
        let ctxs = [&ours, &theirs];
        let ext = |rng: &mut Rng| 1 + rng.below(5);
        let (ea, eb, ec) = (ext(&mut rng), ext(&mut rng), ext(&mut rng));
        table(&mut rng, &ctxs, "l", [ea, eb], keys, messy);
        table(&mut rng, &ctxs, "r", [eb, ec], keys, messy);
        // d is messy on its own sometimes, so that a domain with duplicate
        // keys meets operands the dense algorithm can take.
        let d_messy = messy || rng.chance(if joined_back { 0.5 } else { 0.3 });
        table(&mut rng, &ctxs, "d", [ea + 1, ec + 1], keys, d_messy);
        let (sql, expect) = query(&mut rng, joined_back);
        let fail = |what: String| panic!("seed {seed}: {what}\nSQL: {sql}");

        let df = ours.sql(&sql).await.unwrap();
        // Optimized logical plans stay standard: other code reads them.
        let logical = df.clone().into_optimized_plan().unwrap();
        if folds(&logical) != (0, 0) {
            fail(format!(
                "a Fold in the optimized plan:\n{}",
                logical.display_indent()
            ));
        }
        let physical = df.clone().create_physical_plan().await.unwrap();
        let (found, domains) = exec_folds(&physical);
        let ok = match expect {
            Expect::Fold | Expect::FoldOnly => found == 1 && domains == 0,
            Expect::FromDomain => found == 1 && domains == 1,
            Expect::JoinedBack => found == 1 && domains == usize::from(fallback),
            Expect::Decline => found == 0,
        };
        if !ok {
            let shown = datafusion::physical_plan::displayable(physical.as_ref()).indent(true);
            fail(format!(
                "expected {expect:?}, found {found} folds, {domains} with a domain:\n{shown}"
            ));
        }
        fired += usize::from(found > 0);
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
        if expect == Expect::JoinedBack && fallback {
            joined_dense += usize::from(shown.contains("dense_folds=1"));
            joined_host += usize::from(shown.contains("host_plans=1"));
        }
        let want = theirs.sql(&sql).await.unwrap().collect().await.unwrap();
        let (got, want) = (rows(&got), rows(&want));
        if got != want {
            fail(format!("einfold gave\n{got:#?}\nthe host gave\n{want:#?}"));
        }
    }
    eprintln!(
        "the rule fired in {fired} of {cases} cases; the dense path ran in {dense}, the host's plan in {host_plans}"
    );
    eprintln!(
        "folds joined back to their domain: the dense path ran in {joined_dense}, the host's plan in {joined_host}"
    );
    (
        cases as usize,
        dense,
        host_plans,
        (joined_dense, joined_host),
    )
}

/// Runs `setup`'s statements, then `sql` with and without einfold, and
/// returns how many folds einfold found, after checking both give the same
/// rows.
async fn same_answer(setup: &[&str], sql: &str) -> usize {
    let (plain, with) = (host(4), einfold(4, true));
    for ctx in [&plain, &with] {
        for s in setup {
            ctx.sql(s).await.unwrap().collect().await.unwrap();
        }
    }
    let expected = rows(&plain.sql(sql).await.unwrap().collect().await.unwrap());
    let frame = with.sql(sql).await.unwrap();
    let state = Arc::new(frame.task_ctx());
    let plan = frame.create_physical_plan().await.unwrap();
    let found = exec_folds(&plan).0;
    let got = datafusion::physical_plan::collect(plan, state)
        .await
        .unwrap_or_else(|e| panic!("einfold failed where DataFusion answers: {e}\n{sql}"));
    assert_eq!(rows(&got), expected, "{sql}");
    found
}

/// A factor that can fail stays where the host computes it, on joined rows
/// only: `l`'s third row joins nothing, so DataFusion never casts `'oops'`
/// or divides by its zero.
#[tokio::test]
async fn fallible_factors_stay_above_the_join() {
    let setup = [
        "CREATE TABLE l (a BIGINT, b BIGINT, t VARCHAR, n BIGINT) AS VALUES \
         (0, 0, '1.5', 1), (1, 0, '2.5', 2), (1, 9, 'oops', 0)",
        "CREATE TABLE r (a BIGINT, b BIGINT, v DOUBLE) AS VALUES (0, 0, 1.0), (0, 1, 2.0)",
    ];
    for factor in [
        "CAST(l.t AS DOUBLE)",
        "CAST(l.a / l.n AS DOUBLE)",
        "CAST(l.a % l.n AS DOUBLE)",
    ] {
        let sql = format!(
            "SELECT l.a, r.b, SUM({factor} * r.v) AS v FROM l JOIN r ON l.b = r.a GROUP BY l.a, r.b"
        );
        assert_eq!(same_answer(&setup, &sql).await, 0, "{sql}");
    }
}

/// A volatile factor is drawn once per joined pair in the host's plan, and
/// would be drawn once per input row in a fold's operand.
#[tokio::test]
async fn volatile_factors_are_left_alone() {
    let setup = [
        "CREATE TABLE l (a BIGINT, b BIGINT, v DOUBLE) AS VALUES (0, 0, 1.0)",
        "CREATE TABLE r (a BIGINT, b BIGINT, v DOUBLE) AS VALUES (0, 0, 1.0)",
    ];
    let sql = "SELECT l.a, r.b, SUM((l.v * 0.0 + random() * 0.0) * r.v) AS v \
               FROM l JOIN r ON l.b = r.a GROUP BY l.a, r.b";
    assert_eq!(same_answer(&setup, sql).await, 0);
}

/// Factors built from infallible pieces are still found.
#[tokio::test]
async fn infallible_factors_are_found() {
    let setup = [
        "CREATE TABLE l (a BIGINT, b BIGINT, v DOUBLE, i INT) AS VALUES (0, 0, 1.0, 3), (1, 0, -2.0, 4)",
        "CREATE TABLE r (a BIGINT, b BIGINT, v DOUBLE) AS VALUES (0, 0, 1.0), (0, 1, 2.0)",
    ];
    let factors = [
        "exp(l.v)",
        "CAST(l.i * 2 AS DOUBLE)",
        "CASE WHEN l.v > 0 THEN ln(l.v) ELSE -l.v END",
        "COALESCE(abs(l.v), 0.0)",
    ];
    for factor in factors {
        let sql = format!(
            "SELECT l.a, r.b, SUM({factor} * r.v) AS v FROM l JOIN r ON l.b = r.a GROUP BY l.a, r.b"
        );
        assert_eq!(same_answer(&setup, &sql).await, 1, "{sql}");
    }
}

/// Every function a factor may apply evaluates without error on awkward
/// `Float64` inputs.
#[tokio::test]
async fn movable_functions_never_fail() {
    let ctx = host(1);
    let values = "(CAST('NaN' AS DOUBLE)), (CAST('inf' AS DOUBLE)), (CAST('-inf' AS DOUBLE)), \
                  (0.0), (-0.0), (-2.5), (-3.0), (0.5), (2.0), (1.7976931348623157e308), \
                  (2.2250738585072014e-308), (NULL)";
    ctx.sql(&format!("CREATE TABLE vals (x DOUBLE) AS VALUES {values}"))
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    for f in einfold_datafusion::MOVABLE_FUNCTIONS {
        let sql = format!("SELECT {f}(x) FROM vals");
        let batches = (ctx.sql(&sql).await.unwrap().collect().await)
            .unwrap_or_else(|e| panic!("{sql} failed: {e}"));
        assert_eq!(
            batches.iter().map(|b| b.num_rows()).sum::<usize>(),
            12,
            "{sql}"
        );
    }
}

/// Text keys, so the dense algorithm declines and the host's plan runs.
const TEXT_TABLES: [&str; 2] = [
    "CREATE TABLE l (a VARCHAR, b VARCHAR, v DOUBLE) AS VALUES ('k0', 'j0', 1.0), ('k1', 'j0', 2.0)",
    "CREATE TABLE r (a VARCHAR, b VARCHAR, v DOUBLE) AS VALUES ('j0', 'm0', 3.0), ('j0', 'm1', 4.0)",
];

/// A recursive query resets its recursive term's plan before each
/// iteration, so the fallback, which runs here, must be reset with it.
#[tokio::test]
async fn folds_whose_fallback_runs_recur() {
    let sql = "WITH RECURSIVE it(n, s) AS (\
                 SELECT 0, 0.0 \
                 UNION ALL \
                 SELECT it.n + 1, it.s + f.v \
                 FROM it JOIN (SELECT l.a AS g0, SUM(l.v * r.v) AS v \
                               FROM l JOIN r ON l.b = r.a GROUP BY l.a) f ON f.g0 = 'k0' \
                 WHERE it.n < 4) \
               SELECT * FROM it";
    assert!(same_answer(&TEXT_TABLES, sql).await > 0);
}

/// A plan whose fallback ran, reset and run again, gives the same rows.
#[tokio::test]
async fn reset_plans_whose_fallback_ran_rerun() {
    let ctx = einfold(4, true);
    for s in TEXT_TABLES {
        ctx.sql(s).await.unwrap().collect().await.unwrap();
    }
    let sql = "SELECT l.a, r.b, SUM(l.v * r.v) AS v FROM l JOIN r ON l.b = r.a GROUP BY l.a, r.b";
    let frame = ctx.sql(sql).await.unwrap();
    let task = Arc::new(frame.task_ctx());
    let plan = frame.create_physical_plan().await.unwrap();
    let first = rows(
        &datafusion::physical_plan::collect(Arc::clone(&plan), Arc::clone(&task))
            .await
            .unwrap(),
    );
    let reset = datafusion::physical_plan::execution_plan::reset_plan_states(plan).unwrap();
    let again = rows(
        &datafusion::physical_plan::collect(Arc::clone(&reset), task)
            .await
            .unwrap(),
    );
    assert_eq!(first.len(), 4);
    assert_eq!(first, again);
}

/// A fold from a domain falls back to the host's own plan, which holds no
/// `EinFoldExec`: one there would read the operands again.
#[tokio::test]
async fn fallbacks_hold_only_the_hosts_plan() {
    let ctx = einfold(4, true);
    for s in TEXT_TABLES {
        ctx.sql(s).await.unwrap().collect().await.unwrap();
    }
    ctx.sql("CREATE TABLE d (a VARCHAR, b VARCHAR, v DOUBLE) AS VALUES ('k0', 'm0', 1.0)")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let sql = "SELECT d.a, d.b, CASE WHEN d.v IS NULL THEN NULL WHEN s.v IS NULL THEN 0.0 \
               ELSE s.v END AS v FROM d LEFT JOIN (SELECT l.a AS g0, r.b AS g1, \
               SUM(l.v * r.v) AS v FROM l JOIN r ON l.b = r.a GROUP BY l.a, r.b) s \
               ON (d.a IS NOT DISTINCT FROM s.g0) AND (d.b IS NOT DISTINCT FROM s.g1)";
    let plan = ctx
        .sql(sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    fn execs(
        plan: &Arc<dyn datafusion::physical_plan::ExecutionPlan>,
        out: &mut Vec<Arc<dyn datafusion::physical_plan::ExecutionPlan>>,
    ) {
        let any: &dyn std::any::Any = plan.as_ref();
        if any.is::<EinFoldExec>() {
            out.push(Arc::clone(plan));
        }
        for c in plan.children() {
            execs(c, out);
        }
    }
    let mut found = Vec::new();
    execs(&plan, &mut found);
    assert_eq!(found.len(), 1, "one fold from a domain");
    let any: &dyn std::any::Any = found[0].as_ref();
    let fallback = any
        .downcast_ref::<EinFoldExec>()
        .unwrap()
        .fallback()
        .expect("a fallback");
    let mut inner = Vec::new();
    execs(fallback, &mut inner);
    assert!(inner.is_empty(), "the fallback runs einfold again");
}
