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
//! - **folds from a domain** (the left join from `d` onto a fold, with its
//!   `CASE`, that ddx, an XQL Systems project for automatic differentiation
//!   of SQL queries, writes for a gradient), which the rule must turn into a
//!   fold with `d` as its domain;
//! - **masked folds** (`AND l.a >= r.b`), a fold where both masked columns
//!   are group keys and a look-alike otherwise, alone, from a domain and
//!   joined back;
//! - **look-alikes** it must leave alone: no
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
//! own kernels (`DetectFolds::default`), so that the folds the rule
//! describes are checked even where the dense algorithm doesn't apply.
//!
//! Reproduce a failure with `EINFOLD_SEED=<seed printed>`; search longer with
//! `EINFOLD_CASES=<n>`.

use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::util::display;
use datafusion::datasource::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::logical_expr::LogicalPlan;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion::scalar::ScalarValue;
use datafusion::variable::{VarProvider, VarType};
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
            // The regroup's test: ddx's, or one of another column, which
            // duplicate rows of d may disagree on.
            let test = ["d2.v IS NULL", "d2.v < 0.0", "d2.v IS NULL OR d2.v > 2.0"][rng.below(3)];
            let masked = format!("SUM(CASE WHEN {test} THEN NULL ELSE f.v END)");
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
                _ => ("d2.a, d2.b", masked.as_str(), "d", "d.b"),
            };
            let expect = match (look, inner, outer) {
                (0..=3 | 10, _, _) | (_, true, false) => Expect::FoldOnly,
                _ => Expect::JoinedBack,
            };
            let regroup = format!(
                "SELECT d2.a AS a, d2.{col} AS b, {sum} AS v FROM {d2} d2 JOIN ({}) f ON {inner_keys} \
                 GROUP BY {group}",
                // Both of the fold's outputs are group keys, so a mask
                // between them keeps it a fold.
                fold(&product, rng.chance(0.3))
            );
            // The value as ddx reads it, masked by d's own row, or plainly;
            // d.v is read either way, so that the test's column is d's.
            let value = [
                "CASE WHEN d.v IS NULL THEN NULL WHEN s.v IS NULL THEN 0.0 ELSE s.v END",
                "s.v",
                "COALESCE(s.v, 0.0)",
            ][rng.below(3)];
            (
                format!(
                    "SELECT d.a, {reads}, d.v AS w, {value} AS v FROM d LEFT JOIN ({regroup}) s ON {outer_keys}"
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
                    fold(&product, rng.chance(0.3))
                ),
                expect,
            )
        }
        3..=4 | 8..=9 => (fold(&product, false), Expect::Fold),
        // A mask between two group keys is a masked fold; any other mask
        // declines.
        5 => {
            let both = groups.contains(&"l.a") && groups.contains(&"r.b") && !two_keys;
            let expect = if both { Expect::Fold } else { Expect::Decline };
            (fold(&product, true), expect)
        }
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

/// DataFusion with einfold. With `fallback`, a fold the rule finds runs the
/// host's plan where the dense algorithm doesn't apply; without it, as by
/// default, every fold runs through einfold's kernels.
fn einfold(threads: usize, fallback: bool) -> SessionContext {
    let config = SessionConfig::new().with_target_partitions(threads);
    let rule = match fallback {
        true => DetectFolds::with_host_fallback(),
        false => DetectFolds::default(),
    };
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(config)
        .with_query_planner(Arc::new(EinFoldQueryPlanner::detecting(rule)))
        .build();
    SessionContext::new_with_state(state)
}

/// DataFusion, with einfold or without, under a memory pool of `bytes`
/// shared fairly among the operators that can spill and given first come,
/// first served to those that can't, as DataFusion's `FairSpillPool` does.
fn pooled(threads: usize, with_einfold: Option<DetectFolds>, bytes: usize) -> SessionContext {
    use datafusion::execution::memory_pool::FairSpillPool;
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    let config = SessionConfig::new().with_target_partitions(threads);
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(FairSpillPool::new(bytes)))
        .build_arc()
        .unwrap();
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(config)
        .with_runtime_env(runtime);
    let state = match with_einfold {
        Some(rule) => state.with_query_planner(Arc::new(EinFoldQueryPlanner::detecting(rule))),
        None => state,
    };
    SessionContext::new_with_state(state.build())
}

/// How many `EinFoldExec`s a physical plan has, and how many of them have a
/// domain.
fn exec_folds(plan: &Arc<dyn datafusion::physical_plan::ExecutionPlan>) -> (usize, usize) {
    let (mut all, mut domains) = (0, 0);
    let any: &dyn Any = plan.as_ref();
    if let Some(e) = any.downcast_ref::<EinFoldExec>() {
        all += 1;
        domains += usize::from(e.fold().is_some_and(|f| f.domain().is_some()));
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
                    _ => display::array_value_to_string(c, r).unwrap(),
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
    let Ran {
        cases,
        dense,
        host_plans,
        joined_dense,
        joined_host,
    } = sweep(0xde7e_c7ed, false).await;
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
        assert!(joined_dense + joined_host > 0, "no fold joined back ran");
    }
}

#[tokio::test]
async fn folds_joined_back_check_their_domain() {
    let ran = sweep(0x10_1ced, true).await;
    let (dense, host) = (ran.joined_dense, ran.joined_host);
    if ran.cases >= 100 {
        let some = |n: usize| n * 10 >= ran.cases;
        assert!(
            some(dense) && some(host),
            "folds joined back ran dense in {dense} cases and on the host in {host}"
        );
    }
}

/// How many cases a sweep ran, and how many ran each path.
struct Ran {
    cases: usize,
    /// Cases that ran the dense algorithm.
    dense: usize,
    /// Cases that ran the host's own plan in place of einfold's.
    host_plans: usize,
    /// Folds joined back to their domain that ran the dense algorithm.
    joined_dense: usize,
    /// Folds joined back to their domain that ran the host's plan.
    joined_host: usize,
}

/// Runs the cases from `default_seed`, or as `EINFOLD_SEED` and
/// `EINFOLD_CASES` say, panicking at the first that fails; with
/// `joined_back`, only folds joined back to their domain.
async fn sweep(default_seed: u64, joined_back: bool) -> Ran {
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
            Expect::JoinedBack => found == 1 && domains == 1,
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
        let shown = DisplayableExecutionPlan::with_metrics(physical.as_ref())
            .indent(true)
            .to_string();
        dense += usize::from(shown.contains("dense_folds=1"));
        host_plans += usize::from(shown.contains("host_plans=1"));
        if expect == Expect::JoinedBack {
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
    Ran {
        cases: cases as usize,
        dense,
        host_plans,
        joined_dense,
        joined_host,
    }
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

/// A session variable, `@t`, a `DOUBLE`.
#[derive(Debug)]
struct Threshold;

impl VarProvider for Threshold {
    fn get_value(&self, _names: Vec<String>) -> datafusion::common::Result<ScalarValue> {
        Ok(ScalarValue::Float64(Some(1.5)))
    }

    fn get_type(&self, _names: &[String]) -> Option<DataType> {
        Some(DataType::Float64)
    }
}

/// A fold's fallback is planned only when it runs, with the planning
/// session's variables: one naming `@t` plans then as it did with the query.
#[tokio::test]
async fn fallbacks_plan_with_the_sessions_variables() {
    let sql = "SELECT l.a, r.b, SUM(l.v * r.v) AS v FROM l JOIN r ON l.b = r.a \
               WHERE l.v > @t GROUP BY l.a, r.b";
    let (plain, with) = (host(4), einfold(4, true));
    for ctx in [&plain, &with] {
        ctx.register_variable(VarType::UserDefined, Arc::new(Threshold));
        for s in TEXT_TABLES {
            ctx.sql(s).await.unwrap().collect().await.unwrap();
        }
    }
    let expected = rows(&plain.sql(sql).await.unwrap().collect().await.unwrap());
    let frame = with.sql(sql).await.unwrap();
    let task = Arc::new(frame.task_ctx());
    let plan = frame.create_physical_plan().await.unwrap();
    assert!(exec_folds(&plan).0 > 0, "einfold finds the fold");
    let got = rows(
        &datafusion::physical_plan::collect(Arc::clone(&plan), task)
            .await
            .unwrap(),
    );
    assert_eq!(got, expected);
    let shown = DisplayableExecutionPlan::with_metrics(plan.as_ref())
        .indent(true)
        .to_string();
    assert!(shown.contains("host_plans=1"), "{shown}");
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
    assert_eq!(folds(fallback), (0, 0), "the fallback runs einfold again");
}

/// A domain whose duplicate rows disagree on the regroup's test: the uniqueness
/// check declines, and the host's plan gives every row of a group the same
/// sum, which the node mustn't mask a second time (#60).
#[tokio::test]
async fn joined_back_tests_apply_once() {
    let setup = [
        "CREATE TABLE l (a BIGINT, b BIGINT, v DOUBLE) AS VALUES (0,0,1.0),(0,1,2.0),(1,0,3.0),(1,1,4.0)",
        "CREATE TABLE r (a BIGINT, b BIGINT, v DOUBLE) AS VALUES (0,0,1.0),(0,1,10.0),(1,0,100.0),(1,1,1000.0)",
        "CREATE TABLE d (a BIGINT, b BIGINT, v DOUBLE) AS VALUES (0,0,NULL),(0,0,5.0),(1,1,7.0)",
        "CREATE TABLE e (a BIGINT, b BIGINT, v DOUBLE) AS VALUES (0,0,NULL),(0,1,5.0),(1,1,-7.0)",
    ];
    for (table, test) in [
        ("d", "d2.v IS NULL"),
        ("d", "d2.v > 6.0"),
        ("e", "d2.v < 0.0"),
    ] {
        let sql = format!(
            "SELECT d.a, d.b, d.v AS w, COALESCE(s.v, 0.0) AS v FROM {table} d LEFT JOIN ( \
               SELECT d2.a AS a, d2.b AS b, SUM(CASE WHEN {test} THEN NULL ELSE f.v END) AS v \
               FROM {table} d2 JOIN (SELECT l.a AS g0, r.b AS g1, SUM(l.v * r.v) AS v \
                                     FROM l JOIN r ON l.b = r.a GROUP BY l.a, r.b) f \
                 ON d2.a = f.g0 AND d2.b = f.g1 \
               GROUP BY d2.a, d2.b) s \
             ON d.a = s.a AND d.b = s.b"
        );
        // Table e's keys are unique, so there the dense path runs.
        assert!(same_answer(&setup, &sql).await > 0, "{sql}");
    }
}

/// A fold joined back to its domain falls back to the host's own plan too.
#[tokio::test]
async fn joined_back_fallbacks_hold_only_the_hosts_plan() {
    let ctx = einfold(4, true);
    for s in [
        "CREATE TABLE l (a BIGINT, b BIGINT, v DOUBLE) AS VALUES (0,0,1.0),(0,1,2.0),(1,0,3.0),(1,1,4.0)",
        "CREATE TABLE r (a BIGINT, b BIGINT, v DOUBLE) AS VALUES (0,0,1.0),(0,1,10.0),(1,0,100.0),(1,1,1000.0)",
        "CREATE TABLE d (a BIGINT, b BIGINT, v DOUBLE) AS VALUES (0,0,NULL),(0,0,5.0),(1,1,7.0)",
    ] {
        ctx.sql(s).await.unwrap().collect().await.unwrap();
    }
    let sql = "SELECT d.a, d.b, d.v AS w, COALESCE(s.v, 0.0) AS v FROM d LEFT JOIN ( \
               SELECT d2.a AS a, d2.b AS b, SUM(CASE WHEN d2.v IS NULL THEN NULL ELSE f.v END) AS v \
               FROM d d2 JOIN (SELECT l.a AS g0, r.b AS g1, SUM(l.v * r.v) AS v \
                               FROM l JOIN r ON l.b = r.a GROUP BY l.a, r.b) f \
                 ON d2.a = f.g0 AND d2.b = f.g1 \
               GROUP BY d2.a, d2.b) s \
             ON d.a = s.a AND d.b = s.b";
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
    assert_eq!(found.len(), 1, "one fold joined back");
    let any: &dyn std::any::Any = found[0].as_ref();
    let fallback = any
        .downcast_ref::<EinFoldExec>()
        .unwrap()
        .fallback()
        .expect("a fallback");
    assert_eq!(folds(fallback), (0, 0), "the fallback runs einfold again");
}

#[test]
fn host_plan_runs_where_there_is_no_tokio_runtime() {
    use datafusion::physical_plan::{self, ExecutionPlan};
    use futures::TryStreamExt;
    fn find(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
        match plan.name() == "EinFoldExec" {
            true => Some(Arc::clone(plan)),
            false => plan.children().into_iter().find_map(find),
        }
    }
    // Text keys, which the dense algorithm declines, so the operator runs the
    // host's plan, whose operators spawn tasks. A host in another library
    // polls the operator through DataFusion's FFI, where this library's Tokio
    // sees no runtime.
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let mut rng = Rng::new(7);
    let ours = einfold(4, true);
    table(&mut rng, &[&ours], "l", [4, 3], Keys::Text, false);
    table(&mut rng, &[&ours], "r", [3, 5], Keys::Text, false);
    let sql = "SELECT l.a, r.b, SUM(l.v * r.v) AS v FROM l JOIN r ON l.b = r.a GROUP BY l.a, r.b";
    // A plan's operators run once, so each run plans the query anew.
    let plan = || {
        runtime.block_on(async {
            let df = ours.sql(sql).await.unwrap();
            let physical = df.create_physical_plan().await.unwrap();
            find(&physical).expect("einfold finds the fold")
        })
    };
    let (first, exec) = (plan(), plan());
    let with = runtime
        .block_on(physical_plan::collect(first, ours.task_ctx()))
        .unwrap();
    drop(runtime);
    let stream = exec.execute(0, ours.task_ctx()).unwrap();
    let without: Vec<RecordBatch> = futures::executor::block_on(stream.try_collect()).unwrap();
    assert_eq!(rows(&with), rows(&without));
    let shown = DisplayableExecutionPlan::with_metrics(exec.as_ref())
        .indent(true)
        .to_string();
    assert!(shown.contains("host_plans=1"), "{shown}");
}

/// A masked fold under ddx's left join from a domain, and joined back to its
/// domain: the fold the rule makes keeps the mask, so the pairs it disallows
/// add nothing, by einfold's kernels and with the host's plan as fallback.
#[tokio::test]
async fn masked_folds_keep_their_mask_under_a_domain() {
    // With BIGINT keys the dense algorithm runs; with text keys it declines,
    // and the host's plan, which must keep the mask too, runs where the
    // session allows it.
    for text in [false, true] {
        let key = |x: i64| match text {
            true => format!("'{x}'"),
            false => x.to_string(),
        };
        let table = |name: &str, rows: &[(i64, i64, &str)]| {
            let ty = if text { "VARCHAR" } else { "BIGINT" };
            let values: Vec<String> = (rows.iter())
                .map(|(a, b, v)| format!("({},{},{v})", key(*a), key(*b)))
                .collect();
            format!(
                "CREATE TABLE {name} (a {ty}, b {ty}, v DOUBLE) AS VALUES {}",
                values.join(",")
            )
        };
        let setup = [
            table(
                "l",
                &[(0, 0, "1.0"), (0, 1, "2.0"), (1, 0, "3.0"), (1, 1, "4.0")],
            ),
            table(
                "r",
                &[
                    (0, 0, "1.0"),
                    (0, 1, "10.0"),
                    (0, 2, "5.0"),
                    (1, 0, "100.0"),
                    (1, 1, "1000.0"),
                    (1, 2, "7.0"),
                ],
            ),
            // Groups (0, 2) and (1, 2) have joined rows, summing to 19 and 43
            // unmasked, but the mask `l.a >= r.b` excludes them, so they're 0
            // from the domain.
            table(
                "d",
                &[
                    (0, 0, "1.0"),
                    (0, 1, "NULL"),
                    (0, 2, "1.0"),
                    (1, 0, "1.0"),
                    (1, 1, "1.0"),
                    (1, 2, "1.0"),
                ],
            ),
        ];
        let masked = "SELECT l.a AS g0, r.b AS g1, SUM(l.v * r.v) AS v FROM l JOIN r \
                      ON l.b = r.a AND l.a >= r.b GROUP BY l.a, r.b";
        let from_domain = format!(
            "SELECT d.a, d.b, CASE WHEN d.v IS NULL THEN NULL WHEN s.v IS NULL THEN 0.0 \
             ELSE s.v END AS v FROM d LEFT JOIN ({masked}) s ON d.a = s.g0 AND d.b = s.g1"
        );
        let joined_back = format!(
            "SELECT d.a, d.b, d.v AS w, COALESCE(s.v, 0.0) AS v FROM d LEFT JOIN ( \
               SELECT d2.a AS a, d2.b AS b, SUM(f.v) AS v \
               FROM d d2 JOIN ({masked}) f ON d2.a = f.g0 AND d2.b = f.g1 \
               GROUP BY d2.a, d2.b) s \
             ON d.a = s.a AND d.b = s.b"
        );
        for sql in [&from_domain, &joined_back] {
            for fallback in [false, true] {
                let (plain, with) = (host(4), einfold(4, fallback));
                for ctx in [&plain, &with] {
                    for s in &setup {
                        ctx.sql(s).await.unwrap().collect().await.unwrap();
                    }
                }
                let expected = rows(&plain.sql(sql).await.unwrap().collect().await.unwrap());
                let frame = with.sql(sql).await.unwrap();
                let state = Arc::new(frame.task_ctx());
                let plan = frame.create_physical_plan().await.unwrap();
                let (found, domains) = exec_folds(&plan);
                assert!(
                    found > 0 && domains > 0,
                    "einfold finds the fold from d (text keys: {text})\n{sql}"
                );
                let got = datafusion::physical_plan::collect(Arc::clone(&plan), state)
                    .await
                    .unwrap();
                assert_eq!(
                    rows(&got),
                    expected,
                    "text keys: {text}, fallback {fallback}\n{sql}"
                );
                if text && fallback {
                    let shown = DisplayableExecutionPlan::with_metrics(plan.as_ref())
                        .indent(true)
                        .to_string();
                    assert!(shown.contains("host_plans=1"), "{shown}");
                }
            }
        }
    }
}

/// Causal attention's scores over two heads, `h`: a masked fold whose join
/// key `h` is also grouped by, a batch dimension, as the scores of a
/// multi-head model in SQL have. The dense algorithm computes it, so the
/// host's plan never runs.
#[tokio::test]
async fn masked_folds_over_heads_run_dense() {
    use datafusion::physical_plan;
    let values = |phase: f64| -> String {
        let mut rows = Vec::new();
        for h in 0..2 {
            for t in 0..5 {
                for d in 0..3 {
                    // Small integers, so that every order of the sums is exact.
                    let v = ((h * 15 + t * 3 + d) as f64 + phase) % 7.0 - 3.0;
                    rows.push(format!("({h}, {t}, {d}, {v:?})"));
                }
            }
        }
        rows.join(", ")
    };
    let setup = [
        format!(
            "CREATE TABLE q (h BIGINT, t BIGINT, d BIGINT, x DOUBLE) AS VALUES {}",
            values(0.0)
        ),
        format!(
            "CREATE TABLE k (h BIGINT, t BIGINT, d BIGINT, x DOUBLE) AS VALUES {}",
            values(2.0)
        ),
    ];
    let setup: Vec<&str> = setup.iter().map(String::as_str).collect();
    let sql = "SELECT q.h, q.t AS i, k.t AS j, SUM(q.x * k.x) AS s FROM q JOIN k \
               ON q.h = k.h AND q.d = k.d WHERE q.t >= k.t GROUP BY q.h, q.t, k.t";
    assert_eq!(same_answer(&setup, sql).await, 1);
    let ours = einfold(4, true);
    for s in &setup {
        ours.sql(s).await.unwrap().collect().await.unwrap();
    }
    let frame = ours.sql(sql).await.unwrap();
    let ctx = Arc::new(frame.task_ctx());
    let plan = frame.create_physical_plan().await.unwrap();
    physical_plan::collect(Arc::clone(&plan), ctx)
        .await
        .unwrap();
    let shown = DisplayableExecutionPlan::with_metrics(plan.as_ref())
        .indent(true)
        .to_string();
    assert!(shown.contains("dense_folds=1"), "{shown}");
    assert!(!shown.contains("host_plans=1"), "{shown}");
}

/// Property: under a small memory pool, einfold runs out of memory no more
/// often than the host does. Each case runs a random query of the sweep's
/// spellings with einfold, by its kernels or with the host's plan as
/// fallback, and without, each under its own pool of the same size. Where
/// both answer, they answer the same rows; neither fails but for memory.
/// einfold reserves what it reads and what its kernels allocate, and where
/// the pool refuses, runs the host's plan, whose operators can spill. The
/// operators above einfold's can still differ from the host's, as einfold's
/// estimates of its output differ from the host's, so a case can run short
/// of memory in one plan and not the other, either way.
#[tokio::test]
async fn einfold_runs_short_of_memory_only_where_the_host_does() {
    let seed: u64 = std::env::var("EINFOLD_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x5_9001);
    let cases: u64 = match std::env::var("EINFOLD_CASES") {
        Ok(n) => n.parse().unwrap(),
        Err(_) if std::env::var("EINFOLD_SEED").is_ok() => 1,
        Err(_) => 300,
    };
    let short = |e: &datafusion::error::DataFusionError| {
        e.to_string().contains("Resources exhausted") || e.to_string().contains("budget")
    };
    let (mut both_short, mut host_short, mut ours_short, mut fell_back) = (0, 0, 0, 0);
    for i in 0..cases {
        let seed = seed.wrapping_add(i);
        let mut rng = Rng::new(seed);
        let keys = [Keys::Dense, Keys::Dense, Keys::Wild, Keys::Text][rng.below(4)];
        let messy = !matches!(keys, Keys::Dense) || rng.chance(0.4);
        let rule = match rng.chance(0.5) {
            true => DetectFolds::with_host_fallback(),
            false => DetectFolds::default(),
        };
        let bytes = [2 << 10, 4 << 10, 8 << 10, 16 << 10, 64 << 10, 256 << 10][rng.below(6)];
        let threads = 1 + rng.below(4);
        let (ours, theirs) = (
            pooled(threads, Some(rule), bytes),
            pooled(threads, None, bytes),
        );
        let ctxs = [&ours, &theirs];
        let ext = |rng: &mut Rng| 1 + rng.below(6);
        let (ea, eb, ec) = (ext(&mut rng), ext(&mut rng), ext(&mut rng));
        table(&mut rng, &ctxs, "l", [ea, eb], keys, messy);
        table(&mut rng, &ctxs, "r", [eb, ec], keys, messy);
        let d_messy = messy || rng.chance(0.3);
        table(&mut rng, &ctxs, "d", [ea + 1, ec + 1], keys, d_messy);
        let joined_back = rng.chance(0.2);
        let (sql, _) = query(&mut rng, joined_back);
        let fail =
            |what: String| panic!("seed {seed}, a pool of {bytes} bytes: {what}\nSQL: {sql}");
        let want = match theirs.sql(&sql).await.unwrap().collect().await {
            Ok(want) => Some(rows(&want)),
            Err(e) if short(&e) => None,
            Err(e) => {
                fail(format!("the host failed: {e}"));
                None
            }
        };
        let df = ours.sql(&sql).await.unwrap();
        let physical = df.create_physical_plan().await.unwrap();
        let got = datafusion::physical_plan::collect(Arc::clone(&physical), ours.task_ctx()).await;
        let shown = DisplayableExecutionPlan::with_metrics(physical.as_ref())
            .indent(true)
            .to_string();
        fell_back += usize::from(shown.contains("memory_fallbacks=1"));
        match (got, want) {
            (Ok(got), Some(want)) if rows(&got) == want => {}
            (Ok(got), Some(want)) => {
                let got = rows(&got);
                fail(format!("einfold gave\n{got:#?}\nthe host gave\n{want:#?}"));
            }
            (Err(e), Some(_)) if short(&e) => ours_short += 1,
            (Err(e), Some(_)) => {
                fail(format!("einfold failed where the host answers: {e}"));
            }
            (Err(e), None) if short(&e) => both_short += 1,
            (Err(e), None) => {
                fail(format!("einfold failed: {e}"));
            }
            (Ok(_), None) => host_short += 1,
        }
    }
    eprintln!(
        "out of memory in both {both_short}, in einfold only {ours_short}, in the host only \
         {host_short}; einfold ran the host's plan for memory in {fell_back} of {cases} cases"
    );
    assert!(
        ours_short <= host_short,
        "einfold alone ran out of memory in {ours_short} cases, the host alone in {host_short}"
    );
    if cases >= 100 {
        assert!(fell_back > 0, "einfold never ran short of memory");
    }
}
