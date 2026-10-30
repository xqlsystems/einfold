// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Simulation tests: random queries, in many spellings, over random tables,
//! run on DataFusion with einfold installed and without, and held to
//! properties that need no hand-written expected answer.
//!
//! `detect_vs_host.rs` checks that detection fires on the shapes it should,
//! for a handful of spellings each. This file generates the spellings (see
//! `simulation/queries.rs`): joins written as `JOIN … ON` or as comma joins,
//! sides swapped, operands behind subqueries and CTEs, filters and `HAVING`,
//! factors that are expressions (some of which can fail on rows the join
//! drops), keys of eight types (see `simulation/data.rs`), domains with
//! duplicate keys, the regroup's `CASE` testing columns other than the keys,
//! two folds under `UNION ALL`, a fold of a fold, and masked folds.
//!
//! # Properties
//!
//! Each case runs the query on plain DataFusion (the *host*) and on
//! DataFusion with einfold, on different numbers of partitions and batch
//! sizes, with the same session settings, drawn per case: sort-merge rather
//! than hash joins, no repartitioning, no dynamic filters, and sometimes a
//! small memory pool. It checks:
//!
//! - **`[host-settings]`** DataFusion's own answer doesn't depend on those
//!   settings. Where it does, that's DataFusion's bug, and einfold can't
//!   match both answers.
//! - **`[error]`** einfold succeeds wherever the host does. A query the host
//!   can answer is one einfold must answer too: moving an expression below a
//!   join must not make it fail on rows the join drops.
//! - **`[masks-error]`** einfold fails wherever the host does, rather than
//!   answering a query whose answer is an error, and with the same error.
//!   Where an error happens can depend on DataFusion's own plan, so where
//!   only one of them fails, the host on einfold's partitions and batch size
//!   is the oracle instead, for these and every later check. A panic in
//!   einfold is always its failure, unless the host panicked the same way.
//! - **`[schema]`** the same columns, in order: names, types, nullability and
//!   metadata, both as planned and as the batches carry them.
//! - **`[rows]`** the same rows, as a multiset. With exact values (small
//!   multiples of 0.5, NULL, NaN, ±inf, `-0.0`), every sum is exact in any
//!   order, so cells compare bit for bit (every NaN alike). With inexact ones
//!   (ill-conditioned sums), each value is within float summation's bound
//!   of the host's, `2·(n − 1)·u·Σ|terms|`, where the host computes `n` and
//!   `Σ|terms|` by running the same query over twin tables holding 1.0 and
//!   `|v|`; this is why inexact cases use only factors whose magnitude is a
//!   fixed multiple of `|v|`.
//! - **`[rerun]`** running the same physical plan again gives the same rows,
//!   as ddx does when it runs a prepared program for each training step.
//! - **`[threads]`** where einfold computed every sum itself (no host plan ran
//!   and no other aggregate is in the plan), the same bits on another number
//!   of partitions and another batch size.
//! - **`[partition]`** a mask comparing two of a fold's outputs (`WHERE
//!   l.a <= r.b`, with both grouped by) decides each group whole, so the
//!   groups of the mask, of its negation and of its `IS NULL` case are,
//!   between them, exactly the unmasked fold's, with the same values. An
//!   exact check that needs no oracle.
//! - **`[kernels]`** a third session runs every fold detection finds through
//!   einfold's own kernels (`DetectFolds::default`), so the discrimination
//!   algorithm is checked on the plans detection finds, not only the dense
//!   one and the host's.
//!
//! - **`[stall]`** (soak only) no case runs longer than
//!   `EINFOLD_SOAK_STALL_SECS` (default 120): a watchdog reports a hang as a
//!   failure of its seed, then ends the soak.
//!
//! Under a small memory pool, either side may run out of memory where the
//! other doesn't, by its operators' peaks. Those cases are tallied rather
//! than failed: `EinFoldExec` holds its inputs and can't spill.
//!
//! Tallies of which shapes fired and which algorithm ran are asserted on
//! long enough runs, so a generator that drifted into shapes detection never
//! finds would fail loudly rather than pass vacuously.
//!
//! # Running
//!
//! `simulation` runs a bounded number of cases: `EINFOLD_CASES` of them
//! (default 150, or 60 in a debug build) from `EINFOLD_SEED`; with `EINFOLD_SEED` alone it replays
//! one case and prints its SQL and tables.
//!
//! `soak` is an `#[ignore]`d variant that runs fresh seeds for a wall-clock
//! budget, with the knobs and log lines of ddx's soaks, so that
//! `.github/scripts/report_fuzz_findings.sh` triages it:
//!
//! ```text
//! EINFOLD_SOAK_SECS=300 EINFOLD_SOAK_BASE=0 EINFOLD_SOAK_LOG=/path/to/soak.log \
//!   cargo test -p einfold-datafusion --test simulation --release \
//!   -- --ignored --nocapture soak
//! ```
//!
//! Failures known to an open issue are kept out of both by default, so that
//! they don't drown new ones (see [`KNOWN`]); `EINFOLD_SIM_KNOWN=1` brings
//! them back, to check whether a fix holds.

#[path = "simulation/check.rs"]
mod check;
#[path = "simulation/data.rs"]
mod data;
#[path = "simulation/queries.rs"]
mod queries;
#[path = "simulation/rng.rs"]
mod rng;

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::io::Write as _;
use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use datafusion::arrow::datatypes::DataType;
use datafusion::prelude::SessionContext;

use data::{KeyType, Rows, Table, Values};
use queries::{Options, Query, Shape};
use rng::Rng;

/// Bugs with an open issue, which the simulation steps around unless
/// `EINFOLD_SIM_KNOWN` is set, so that they don't drown new findings. Each
/// says what is left out. Remove an entry when its issue is fixed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Known {
    /// DataFusion's TopK prunes with a dynamic filter that compares `-0.0`
    /// and `0.0` as equal, while its sort orders `-0.0` first, so `ORDER BY
    /// … LIMIT` over a float column depends on timing. The sort's keys
    /// include the sum, a float, so: no `LIMIT` (einfold#63, upstream).
    TopKSignedZeros,
    /// `EinFoldExec`'s fallback is optimized when planned, then again as a
    /// child of the plan, and `JoinSelection` can't swap a join that already
    /// has dynamic filters: that planning error isn't reported (einfold#66).
    FallbackOptimizedTwice,
    /// DataFusion's sort-merge join on a float key and another key drops
    /// matches when a side holds both `-0.0` and `0.0`: no sort-merge joins
    /// over float keys (einfold#67, upstream).
    SortMergeSignedZeros,
    /// DataFusion 55.2's sort-merge join keeps its filter's column indices
    /// when a projection is pushed into it, so a join with a filter beyond
    /// its equalities can drop rows or fail on a column's type
    /// (apache/datafusion#24414, fixed after 55.2): no sort-merge joins with
    /// a join filter.
    SortMergeFilterProjection,
    /// DataFusion's plans can hang under a small memory pool when they
    /// repartition, on a first run as on a rerun: no repartitioning, and no
    /// `[rerun]`, with a pool (einfold#71, upstream).
    ResetRerunUnderPool,
    /// DataFusion's `EliminateCrossJoin` makes an `=` in the WHERE clause a
    /// key of the join below it, and gives it that join's NULL equality: over
    /// an `IS NOT DISTINCT FROM` join, NULL then equals NULL. DataFusion
    /// simplifies `NOT (a <> b)` to `a = b` first, so a mask's `<>` reaches
    /// it through the mask's negation. No `[partition]` where a mask's `=`
    /// or `<>` is over such a join (einfold#21, upstream).
    WhereEqualityNullsEqual,
    /// DataFusion compares dictionary-encoded floats by their bits: of them,
    /// `0.0` is greater than `-0.0` and the two are separate groups, while
    /// plain floats' `0.0` and `-0.0` are equal. A mask comparing such a key
    /// with another float key then splits a group of the unmasked fold
    /// between the mask and its negation: no `[partition]` where a mask
    /// compares a dictionary-encoded float (einfold#41, upstream).
    DictionaryFloatZeros,
}

const KNOWN: &[Known] = &[
    Known::TopKSignedZeros,
    Known::FallbackOptimizedTwice,
    Known::SortMergeSignedZeros,
    Known::SortMergeFilterProjection,
    Known::ResetRerunUnderPool,
    Known::WhereEqualityNullsEqual,
    Known::DictionaryFloatZeros,
];

impl check::Settings {
    /// These settings, without repartitioning where a memory pool would
    /// otherwise make DataFusion's plans hang (einfold#71).
    fn avoiding_pool_hang(mut self) -> Self {
        if self.pool.is_some() && avoids(Known::ResetRerunUnderPool) {
            self.no_repartition = true;
        }
        self
    }
}

/// Whether the simulation steps around `bug`.
fn avoids(bug: Known) -> bool {
    KNOWN.contains(&bug) && std::env::var("EINFOLD_SIM_KNOWN").is_err()
}

/// One generated case.
struct Case {
    seed: u64,
    values: Values,
    tables: Vec<Table>,
    /// The tables with every non-NULL value 1.0, then with `|v|`: running the
    /// query over them gives each group's number of terms and their
    /// magnitude, from which the summation bound follows.
    twins: Option<[Vec<Table>; 2]>,
    query: Query,
    /// Partitions and batch sizes for the host, einfold, and einfold again.
    threads: [usize; 3],
    batch_sizes: [usize; 3],
    fallback: bool,
    settings: check::Settings,
}

fn case(seed: u64) -> Case {
    let mut rng = Rng::seeded(seed, 0x51b_ca5e);
    let values = if rng.chance(0.75) {
        Values::Exact
    } else {
        Values::Inexact
    };
    // Dimensions I (l.a, d.a), J (l.b, r.a) and K (r.b, d.b).
    let dense = rng.chance(0.45);
    let ty = |rng: &mut Rng| {
        if dense {
            KeyType::Int64
        } else {
            KeyType::draw(rng)
        }
    };
    let (ti, tj, tk) = (ty(&mut rng), ty(&mut rng), ty(&mut rng));
    let float = |t: KeyType| matches!(t, KeyType::Float64 | KeyType::DictFloat64);
    let floats = [ti, tj, tk].into_iter().any(float);
    let options = Options {
        fallible: values == Values::Exact && rng.chance(0.25),
        exact: values == Values::Exact,
        // Not only over float keys: the sort's last key is the sum `v`, and
        // where keys repeat, as in a domain's duplicates or under `UNION
        // ALL`, `v`'s `-0.0` and `0.0` decide which rows are kept.
        // And only with exact values: at a tie on the keys, an inexact `v`
        // that differs in its last bits between the host and einfold, as
        // the bound allows, keeps other rows.
        limit: values == Values::Exact && !avoids(Known::TopKSignedZeros),
    };
    // Sometimes r's J column is narrower than l's, so the join casts it.
    let tj_r = match tj {
        KeyType::Int64 if rng.chance(0.1) => KeyType::Int32,
        t => t,
    };
    let ext = |rng: &mut Rng| 1 + rng.below(4);
    let (ei, ej, ek) = (ext(&mut rng), ext(&mut rng), ext(&mut rng));
    let rows = |rng: &mut Rng| {
        let mut r = Rows::draw(rng);
        if options.fallible {
            // Text that doesn't read as a number, on rows that may not join:
            // a NULL key joins nothing under `=`.
            r.bad_text = 0.15;
            r.null_key = r.null_key.max(0.2);
        }
        r
    };
    let (lr, rr) = (rows(&mut rng), rows(&mut rng));
    // d repeats keys more often: a domain with duplicate keys is where folds
    // joined back must fall back to the host's plan.
    let mut dr = Rows::draw(&mut rng);
    if rng.chance(0.5) {
        dr.duplicate = 0.25;
    }
    let tables = vec![
        data::table(&mut rng, "l", (ti, ei), (tj, ej), lr, values, "t"),
        data::table(&mut rng, "r", (tj_r, ej), (tk, ek), rr, values, "t"),
        data::table(&mut rng, "d", (ti, ei + 1), (tk, ek + 1), dr, values, "c"),
    ];
    let twins = (values == Values::Inexact).then(|| {
        [
            tables.iter().map(|t| twin(t, |_| 1.0)).collect(),
            tables.iter().map(|t| twin(t, f64::abs)).collect(),
        ]
    });
    let query = queries::query(&mut rng, options);
    let join_filter = query.join_filter;
    let threads = [1 + rng.below(6), 1 + rng.below(6), 1 + rng.below(6)];
    let batch_sizes = [*rng.pick(&[1, 3, 8192]), *rng.pick(&[1, 3, 8192]), 8192];
    Case {
        seed,
        values,
        tables,
        twins,
        query,
        threads,
        batch_sizes,
        fallback: rng.chance(0.8),
        settings: check::Settings {
            sort_merge: rng.chance(0.2)
                && !(floats && avoids(Known::SortMergeSignedZeros))
                && !(join_filter && avoids(Known::SortMergeFilterProjection)),
            no_repartition: rng.chance(0.15),
            no_dynamic_filters: rng.chance(0.15),
            pool: rng
                .chance(0.15)
                .then(|| *rng.pick(&[8 << 10, 32 << 10, 256 << 10])),
        }
        .avoiding_pool_hang(),
    }
}

/// `t` with each non-NULL value `v` replaced by `f(v)`, in one partition.
fn twin(t: &Table, f: impl Fn(f64) -> f64) -> Table {
    let v = t
        .all
        .column(2)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("v");
    let mapped: Float64Array = (0..v.len())
        .map(|r| v.is_valid(r).then(|| f(v.value(r))))
        .collect();
    let mut columns: Vec<ArrayRef> = t.all.columns().to_vec();
    columns[2] = Arc::new(mapped);
    let all = RecordBatch::try_new(Arc::clone(&t.schema), columns).expect("same schema");
    Table {
        name: t.name,
        schema: Arc::clone(&t.schema),
        partitions: vec![vec![all.clone()]],
        all,
    }
}

/// Whether an error says a memory pool was exhausted.
fn out_of_memory(e: &str) -> bool {
    e.contains("Resources exhausted") || e.contains("Not enough memory")
}

/// Which side alone ran out of memory under a small pool, if one did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Memory {
    #[default]
    Neither,
    EinfoldOnly,
    HostOnly,
}

/// What one case found.
#[derive(Default)]
struct Outcome {
    failures: Vec<String>,
    shape: Option<Shape>,
    routes: check::Routes,
    host_failed: bool,
    host_rerun_failed: bool,
    deterministic: bool,
    look_alike: bool,
    memory: Memory,
    pooled: bool,
}

/// The tallies of a run.
#[derive(Default)]
struct Tally {
    cases: usize,
    failures: usize,
    host_failed: usize,
    host_rerun_failed: usize,
    /// Cases built as something detection must leave alone at the top.
    look_alikes: usize,
    /// Cases with a small memory pool, and those where only einfold, or
    /// only the host, ran out of memory.
    pooled: usize,
    out_of_memory: [usize; 2],
    /// Per shape: cases, cases where einfold ran a fold, and the routes.
    by_shape: BTreeMap<String, [usize; 6]>,
    deterministic: usize,
}

impl Tally {
    fn add(&mut self, o: &Outcome) {
        self.cases += 1;
        self.failures += o.failures.len();
        self.host_failed += usize::from(o.host_failed);
        self.host_rerun_failed += usize::from(o.host_rerun_failed);
        self.look_alikes += usize::from(o.look_alike);
        self.pooled += usize::from(o.pooled);
        self.out_of_memory[0] += usize::from(o.memory == Memory::EinfoldOnly);
        self.out_of_memory[1] += usize::from(o.memory == Memory::HostOnly);
        self.deterministic += usize::from(o.deterministic);
        let s = self
            .by_shape
            .entry(format!("{:?}", o.shape.expect("a shape")))
            .or_default();
        let r = o.routes;
        for (slot, n) in s.iter_mut().zip([
            1,
            usize::from(r.folds > 0),
            r.dense,
            r.discrimination,
            r.host,
            r.domains,
        ]) {
            *slot += n;
        }
    }

    fn summary(&self) -> String {
        let mut s = format!(
            "cases={} failures={} host_failed={} host_rerun_failed={} deterministic={} look_alikes={}",
            self.cases, self.failures, self.host_failed, self.host_rerun_failed, self.deterministic, self.look_alikes
        );
        let _ = write!(
            s,
            "\n  memory pools: {} cases; out of memory in einfold only {}, in the host only {}",
            self.pooled, self.out_of_memory[0], self.out_of_memory[1]
        );
        for (shape, [n, fired, dense, discrimination, host, domains]) in &self.by_shape {
            let _ = write!(
                s,
                "\n  {shape}: {n} cases, fired in {fired}; dense {dense}, discrimination {discrimination}, \
                 host {host}; folds with a domain {domains}"
            );
        }
        s
    }

    fn get(&self, shape: Shape, slot: usize) -> usize {
        self.by_shape
            .get(&format!("{shape:?}"))
            .map_or(0, |s| s[slot])
    }
}

fn register(ctx: &SessionContext, tables: &[Table]) {
    for t in tables {
        t.register(ctx);
    }
}

/// The host's run on einfold's partitions, batch size and settings.
async fn control(case: &Case, sql: &str) -> check::Ran {
    trace("host, on einfold's partitions");
    let control = check::session(case.threads[1], None, case.batch_sizes[1], case.settings);
    register(&control, &case.tables);
    check::run(&control, sql).await
}

/// An error's message without the values it quotes, such as the text a cast
/// couldn't read, which depend on which row a plan reached first.
fn error_kind(e: &str) -> String {
    e.split('\'').step_by(2).collect::<Vec<_>>().join("'…'")
}

/// A failure report: the tag, what differed, and how to replay it.
fn report(case: &Case, tag: &str, what: &str, ours: Option<&check::Ran>) -> String {
    let mut s = format!(
        "[{tag}] {what}\n  shape={:?} values={:?} fallback={} threads={:?} batch_sizes={:?}\n  settings={:?}\n  SQL: {}\n",
        case.query.shape, case.values, case.fallback, case.threads, case.batch_sizes, case.settings, case.query.sql
    );
    for t in &case.tables {
        s += &t.dump();
    }
    if let Some(ran) = ours {
        let _ = writeln!(s, "  einfold's plan:");
        for line in ran.shown().lines() {
            let _ = writeln!(s, "    {line}");
        }
    }
    let _ = write!(
        s,
        "  replay: EINFOLD_SEED={} cargo test -p einfold-datafusion --test simulation -- --nocapture simulation",
        case.seed
    );
    s
}

/// Shows the first rows where two results differ.
fn diff(ours: &[String], theirs: &[String]) -> String {
    let mut s = format!(
        "einfold gave {} rows, the host {}",
        ours.len(),
        theirs.len()
    );
    let only = |a: &[String], b: &[String]| -> Vec<String> {
        let mut b = b.to_vec();
        a.iter()
            .filter(|x| match b.iter().position(|y| y == *x) {
                Some(i) => {
                    b.remove(i);
                    false
                }
                None => true,
            })
            .take(8)
            .cloned()
            .collect()
    };
    let _ = write!(
        s,
        "\n  only einfold's: {:?}\n  only the host's: {:?}",
        only(ours, theirs),
        only(theirs, ours)
    );
    s
}

async fn run_case(case: &Case) -> Outcome {
    let mut out = Outcome {
        shape: Some(case.query.shape),
        look_alike: case.query.look_alike,
        pooled: case.settings.pool.is_some(),
        ..Outcome::default()
    };
    let sql = &case.query.sql;
    let host = check::session(case.threads[0], None, case.batch_sizes[0], case.settings);
    let ours = check::session(
        case.threads[1],
        Some(case.fallback),
        case.batch_sizes[1],
        case.settings,
    );
    register(&host, &case.tables);
    register(&ours, &case.tables);
    trace(&format!(
        "host, seed {} threads {:?} batch sizes {:?} {:?}",
        case.seed, case.threads, case.batch_sizes, case.settings
    ));
    let theirs = check::run(&host, sql).await;
    trace("einfold");
    let mine = check::run(&ours, sql).await;
    out.routes = mine.routes();
    let mut fail = |tag: &str, what: String, ran: Option<&check::Ran>| {
        out.failures.push(report(case, tag, &what, ran));
    };
    // The host itself under default settings: where its answer depends on the
    // settings, that's DataFusion's bug, and einfold can't match both.
    if case.values == Values::Exact && case.settings.pool.is_none() {
        trace("host, default settings");
        let default = check::session(case.threads[0], None, 8192, check::Settings::default());
        register(&default, &case.tables);
        if let (Ok(a), Ok(b)) = (&theirs.result, &check::run(&default, sql).await.result) {
            let (a, b) = (check::exact(a), check::exact(b));
            if a != b {
                fail(
                    "host-settings",
                    format!(
                        "DataFusion's own answer depends on the settings: {}",
                        diff(&a, &b)
                    ),
                    None,
                );
                return out;
            }
        }
    }
    // An error can depend on the plan, not only on einfold: a cast of text
    // that doesn't read as a number fails only where a row reaches it, and
    // DataFusion's own plan on other partitions can reach rows the host's
    // plan didn't. So where only one of the host and einfold fails, the host
    // on einfold's partitions and batch size is the oracle instead: einfold
    // must succeed where it does, match its answer, and fail where it does
    // with the same error.
    let controlled = theirs.result.is_ok() != mine.result.is_ok();
    let theirs = match controlled {
        true => control(case, sql).await,
        false => theirs,
    };
    let (want, got) = match (&theirs.result, &mine.result) {
        (Ok(w), Ok(g)) => (w, g),
        // Under a small memory pool, either side may run out where the other
        // doesn't, by its operators' peaks: tallied, not failed.
        (Ok(_), Err(e)) if out_of_memory(e) => {
            out.memory = Memory::EinfoldOnly;
            return out;
        }
        (Err(e), Ok(_)) if out_of_memory(e) => {
            out.memory = Memory::HostOnly;
            return out;
        }
        (Ok(_), Err(e))
            if avoids(Known::FallbackOptimizedTwice)
                && e.contains("Cannot swap HashJoinExec inputs after dynamic filters") =>
        {
            return out;
        }
        (Ok(_), Err(e)) => {
            fail(
                "error",
                format!("einfold failed where the host succeeded: {e}"),
                Some(&mine),
            );
            return out;
        }
        (Err(e), Ok(_)) => {
            out.host_failed = true;
            fail(
                "masks-error",
                format!("einfold answered where the host failed: {e}"),
                Some(&mine),
            );
            return out;
        }
        // A panic in einfold is einfold's, unless the host panicked the
        // same way; and the control must fail as einfold did. Running out of
        // memory is tallied, as above, whatever the other side's error.
        (Err(w), Err(g))
            if error_kind(w) != error_kind(g)
                && (controlled || g.starts_with("panicked"))
                && !out_of_memory(w)
                && !out_of_memory(g) =>
        {
            out.host_failed = true;
            fail(
                "error",
                format!("einfold failed unlike the host: {g}\n  the host: {w}"),
                Some(&mine),
            );
            return out;
        }
        (Err(_), Err(_)) => {
            out.host_failed = true;
            return out;
        }
    };
    if let (Some(a), Some(b)) = (&mine.schema, &theirs.schema) {
        if let Err(e) = check::same_fields(a, b) {
            fail("schema", format!("planned schema: {e}"), Some(&mine));
        }
    }
    if let (Some(a), Some(b)) = (got.first(), want.first()) {
        if let Err(e) = check::same_fields(&a.schema(), &b.schema()) {
            fail("schema", format!("batches' schema: {e}"), Some(&mine));
        }
    }
    match case.values {
        Values::Exact => {
            let (g, w) = (check::exact(got), check::exact(want));
            if g != w {
                fail("rows", diff(&g, &w), Some(&mine));
            }
        }
        Values::Inexact => {
            if let Err(e) = within_bound(case, got, want).await {
                fail("rows", e, Some(&mine));
            }
        }
    }
    // Every sum einfold's own: then bits must not depend on the run.
    let r = out.routes;
    out.deterministic = r.folds > 0 && r.host == 0 && r.other_aggregates == 0;
    let bits_must_match = out.deterministic || case.values == Values::Exact;
    // [partition]: a mask comparing two of a fold's outputs decides each
    // group whole, so the groups of the mask, of its negation and of its NULL
    // case are, between them, exactly the unmasked fold's, with the same
    // values.
    if let (Some(m), Values::Exact) = (&case.query.mask, case.values) {
        let nulls_equal = (m.condition.contains(" = ") || m.condition.contains(" <> "))
            && case.query.sql.contains("IS NOT DISTINCT FROM")
            && avoids(Known::WhereEqualityNullsEqual);
        // The mask's columns, as `l.a`: the tables' aliases are their names.
        let dictionary_floats = (case.tables.iter())
            .flat_map(|t| t.schema.fields().iter().map(move |f| (t.name, f)))
            .any(|(table, f)| {
                matches!(f.data_type(), DataType::Dictionary(_, v) if **v == DataType::Float64)
                    && m.condition.contains(&format!("{table}.{}", f.name()))
            })
            && avoids(Known::DictionaryFloatZeros);
        if m.partitions && !nulls_equal && !dictionary_floats {
            trace("einfold, partition");
            let with = |c: &str| m.template.replace("{MASK}", c);
            let all = check::run(&host, &with("TRUE")).await.result;
            let mut parts = Vec::new();
            for c in [
                m.condition.clone(),
                format!("NOT {}", m.condition),
                format!("({}) IS NULL", m.condition),
            ] {
                parts.push(check::run(&ours, &with(&c)).await.result);
            }
            if let (Ok(all), [Ok(a), Ok(b), Ok(c)]) = (all, parts.as_slice()) {
                let mut union: Vec<String> =
                    [a, b, c].iter().flat_map(|p| check::exact(p)).collect();
                union.sort();
                let all = check::exact(&all);
                if union != all {
                    fail("partition", diff(&union, &all), Some(&mine));
                }
            }
        }
    }
    // [rerun], on sessions of their own without dynamic filters: DataFusion's
    // `reset_plan_states` doesn't support plans with them, and a hash join's
    // can hang a plan run again. The host's own plan, run again, is the
    // control: where it can't run twice, einfold's needn't either.
    let mut settings = case.settings;
    settings.no_dynamic_filters = true;
    if avoids(Known::ResetRerunUnderPool) {
        settings.pool = None;
    }
    let host2 = check::session(case.threads[0], None, case.batch_sizes[0], settings);
    let ours2 = check::session(
        case.threads[1],
        Some(case.fallback),
        case.batch_sizes[1],
        settings,
    );
    register(&host2, &case.tables);
    register(&ours2, &case.tables);
    trace(&format!("host, first run for rerun, {settings:?}"));
    let theirs2 = check::run(&host2, sql).await;
    trace(&format!(
        "host's first run: {:?}",
        theirs2.result.as_ref().map(|b| b.len())
    ));
    trace("einfold, first run for rerun");
    let mine2 = check::run(&ours2, sql).await;
    trace("host, rerun");
    let host_reruns = match (&theirs2.result, check::rerun(&host2, &theirs2).await) {
        (Ok(first), Some(Ok(again))) => {
            check::exact(&again) == check::exact(first) || case.values == Values::Inexact
        }
        _ => false,
    };
    out.host_rerun_failed = !host_reruns;
    if let (true, true, Ok(first)) = (bits_must_match, host_reruns, &mine2.result) {
        trace("einfold, rerun");
        match check::rerun(&ours2, &mine2).await {
            Some(Ok(again)) if check::exact(&again) != check::exact(first) => {
                fail(
                    "rerun",
                    diff(&check::exact(&again), &check::exact(first)),
                    Some(&mine2),
                );
            }
            Some(Err(e)) if !out_of_memory(&e) => fail(
                "rerun",
                format!("running the reset plan again failed: {e}"),
                Some(&mine2),
            ),
            _ => {}
        }
    }
    trace("einfold, other threads");
    if out.deterministic {
        let other = check::session(
            case.threads[2],
            Some(case.fallback),
            case.batch_sizes[2],
            case.settings,
        );
        register(&other, &case.tables);
        if let Ok(b) = check::run(&other, sql).await.result {
            if check::exact(&b) != check::exact(got) {
                fail(
                    "threads",
                    diff(&check::exact(&b), &check::exact(got)),
                    Some(&mine),
                );
            }
        }
    }
    // Every fold through einfold's kernels, the discrimination algorithm
    // included.
    if case.fallback && r.folds > 0 && case.seed.is_multiple_of(3) {
        let kernels = check::session(
            case.threads[2],
            Some(false),
            case.batch_sizes[2],
            case.settings,
        );
        register(&kernels, &case.tables);
        trace("einfold, kernels");
        let ran = check::run(&kernels, sql).await;
        match &ran.result {
            Err(e) if out_of_memory(e) => {}
            Err(e) => fail(
                "kernels",
                format!("failed where the host succeeded: {e}"),
                Some(&ran),
            ),
            Ok(b) if case.values == Values::Exact => {
                let (g, w) = (check::exact(b), check::exact(want));
                if g != w {
                    fail("kernels", diff(&g, &w), Some(&ran));
                }
            }
            Ok(b) => {
                if let Err(e) = within_bound(case, b, want).await {
                    fail("kernels", e, Some(&ran));
                }
            }
        }
    }
    out
}

/// Whether `got` is `want` within float summation's bound, row by row.
async fn within_bound(
    case: &Case,
    got: &[RecordBatch],
    want: &[RecordBatch],
) -> Result<(), String> {
    let [ones, mags] = case.twins.as_ref().expect("an inexact case has twins");
    let aside = |tables: &[Table]| {
        let ctx = check::session(1, None, 8192, check::Settings::default());
        register(&ctx, tables);
        ctx
    };
    let (n, m) = (aside(ones), aside(mags));
    let sql = &case.query.sql;
    let (n, m) = (
        check::run(&n, sql).await.result,
        check::run(&m, sql).await.result,
    );
    let (n, m) = match (n, m) {
        (Ok(n), Ok(m)) => (n, m),
        // The twins are partitioned otherwise, and whether DataFusion
        // evaluates an expression that can fail (a cast of a join key, say)
        // depends on its plan's execution: without the twins' answers there
        // is no bound to check against.
        _ => return Ok(()),
    };
    let (g, w, n, m) = (
        check::rows(got),
        check::rows(want),
        check::rows(&n),
        check::rows(&m),
    );
    if g.len() != w.len() || n.len() != w.len() || m.len() != w.len() {
        return Err(format!(
            "einfold gave {} rows, the host {} (twins {}, {})",
            g.len(),
            w.len(),
            n.len(),
            m.len()
        ));
    }
    // Rows can share keys and differ in value, so the twins' rows can't be
    // paired one by one: each key's bound uses the largest count and
    // magnitude among its rows.
    let mut most: std::collections::HashMap<(&[String], usize), (f64, f64)> =
        std::collections::HashMap::new();
    for (n, m) in n.iter().zip(&m) {
        for i in 0..n.values.len() {
            let e = most.entry((n.keys.as_slice(), i)).or_default();
            e.0 = e.0.max(n.values[i].unwrap_or(0.0).abs());
            let e = most.entry((m.keys.as_slice(), i)).or_default();
            e.1 = e.1.max(m.values[i].unwrap_or(0.0).abs());
        }
    }
    for (g, w) in g.iter().zip(&w) {
        if g.keys != w.keys {
            return Err(format!("row {g:?}, the host's {w:?}"));
        }
        for (i, (&x, &y)) in g.values.iter().zip(&w.values).enumerate() {
            let (count, mag) = most
                .get(&(g.keys.as_slice(), i))
                .copied()
                .unwrap_or_default();
            let bound = 2.0 * (count - 1.0).max(0.0) * (f64::EPSILON / 2.0) * mag;
            // With an infinite or NaN magnitude, a term is non-finite, and an
            // inexact sum it multiplies can be 0 in one order of additions
            // and not in another: 0·∞ is NaN, while x·∞ is ±∞. The bound
            // then promises only that both results are non-finite.
            let non_finite = |v: Option<f64>| v.is_some_and(|v| !v.is_finite());
            if !mag.is_finite() && non_finite(x) && non_finite(y) {
                continue;
            }
            if !check::agree(x, y, bound) {
                return Err(format!(
                    "row {:?}: einfold {x:?}, the host {y:?}, beyond {bound:e} ({count} terms of magnitude {mag:e})",
                    g.keys
                ));
            }
        }
    }
    Ok(())
}

/// With `EINFOLD_SIM_TRACE` set, says which stage of a case is starting, so
/// that a hang shows where it is.
fn trace(stage: &str) {
    if std::env::var("EINFOLD_SIM_TRACE").is_ok() {
        eprintln!("TRACE {stage}");
    }
}

fn env_u64(name: &str) -> Option<u64> {
    std::env::var(name).ok().and_then(|s| s.parse().ok())
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("a runtime")
}

#[test]
fn simulation() {
    let replay = env_u64("EINFOLD_SEED");
    let base = replay.unwrap_or(0x5eed_0001);
    // Fewer in a debug build, where DataFusion runs several times slower.
    let default = if cfg!(debug_assertions) { 60 } else { 150 };
    let cases = env_u64("EINFOLD_CASES").unwrap_or(if replay.is_some() { 1 } else { default });
    let rt = runtime();
    let mut tally = Tally::default();
    let mut failures = Vec::new();
    for i in 0..cases {
        let case = case(base.wrapping_add(i));
        if replay.is_some() && cases == 1 {
            eprintln!("SQL: {}", case.query.sql);
            for t in &case.tables {
                eprint!("{}", t.dump());
            }
        }
        let o = rt.block_on(run_case(&case));
        tally.add(&o);
        failures.extend(
            o.failures
                .into_iter()
                .map(|f| format!("seed {}: {f}", case.seed)),
        );
    }
    eprintln!("{}", tally.summary());
    assert!(
        failures.is_empty(),
        "{} failure(s):\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
    if cases >= 60 {
        check_coverage(&tally);
    }
}

/// Fails if the generator stopped reaching what the properties are about.
fn check_coverage(tally: &Tally) {
    let fired = |s: Shape| tally.get(s, 1);
    let ran = |slot: usize| -> usize {
        [
            Shape::Fold,
            Shape::FromDomain,
            Shape::JoinedBack,
            Shape::Union,
            Shape::Nested,
            Shape::Masked,
        ]
        .iter()
        .map(|&s| tally.get(s, slot))
        .sum()
    };
    let checks = [
        (
            "folds fired",
            fired(Shape::Fold) * 4 >= tally.get(Shape::Fold, 0),
        ),
        ("folds from a domain fired", fired(Shape::FromDomain) > 0),
        ("folds joined back fired", fired(Shape::JoinedBack) > 0),
        ("the dense algorithm ran", ran(2) > 0),
        ("the discrimination algorithm ran", ran(3) > 0),
        ("the host's plan ran", ran(4) > 0),
        ("some cases were deterministic", tally.deterministic > 0),
        (
            "most cases ran on the host",
            tally.host_failed * 3 <= tally.cases,
        ),
    ];
    for (what, ok) in checks {
        assert!(ok, "coverage: expected that {what}\n{}", tally.summary());
    }
}

#[test]
#[ignore]
fn soak() {
    use std::time::Instant;
    let budget = env_u64("EINFOLD_SOAK_SECS").unwrap_or(30);
    let base = env_u64("EINFOLD_SOAK_BASE").unwrap_or(0);
    let log_path = std::env::var("EINFOLD_SOAK_LOG").ok();
    let mut log = log_path.as_ref().map(|p| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .unwrap_or_else(|e| panic!("cannot open EINFOLD_SOAK_LOG `{p}`: {e}"))
    });
    let mut logline = |s: &str| {
        eprintln!("{s}");
        if let Some(f) = log.as_mut() {
            let _ = writeln!(f, "{s}");
            let _ = f.flush();
        }
    };
    let stop_on_fail = std::env::var("EINFOLD_SOAK_STOP_ON_FAIL").is_ok();
    let rt = runtime();
    let start = Instant::now();
    let mut tally = Tally::default();
    let (mut iters, mut last_beat) = (0u64, 0u64);
    logline(&format!(
        "SOAK start: budget={budget}s base={base} log={log_path:?}"
    ));
    logline(
        "REPRO EINFOLD_SEED={seed} cargo test -p einfold-datafusion --test simulation \
         --release -- --nocapture simulation",
    );
    // A watchdog: a case that runs for more than `EINFOLD_SOAK_STALL_SECS`
    // (default 120) is a hang, reported as a failure of its seed before the
    // soak exits, since the stuck case can't be abandoned.
    let stall = env_u64("EINFOLD_SOAK_STALL_SECS").unwrap_or(120);
    let running = Arc::new(std::sync::Mutex::new((u64::MAX, Instant::now())));
    {
        let running = Arc::clone(&running);
        let log_path = log_path.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            let (seed, since) = *running.lock().expect("the watchdog's lock");
            if seed != u64::MAX && since.elapsed().as_secs() > stall {
                let line = format!(
                    "FAILURE (seed={seed}, base={base}):\n[stall] the case ran for more than {stall} s; \
                     replay: EINFOLD_SEED={seed} cargo test -p einfold-datafusion --test simulation \
                     --release -- --nocapture simulation\n"
                );
                eprintln!("{line}");
                if let Some(p) = &log_path {
                    if let Ok(mut f) = std::fs::OpenOptions::new().append(true).open(p) {
                        let _ = writeln!(f, "{line}");
                    }
                }
                std::process::exit(1);
            }
        });
    }
    while start.elapsed().as_secs() < budget {
        let seed = base.wrapping_add(iters);
        *running.lock().expect("the watchdog's lock") = (seed, Instant::now());
        let o = rt.block_on(run_case(&case(seed)));
        tally.add(&o);
        for f in &o.failures {
            logline(&format!("FAILURE (seed={seed}, base={base}):\n{f}"));
        }
        iters += 1;
        if stop_on_fail && tally.failures > 0 {
            break;
        }
        let elapsed = start.elapsed().as_secs();
        if elapsed >= last_beat + 10 {
            last_beat = elapsed;
            logline(&format!(
                "HEARTBEAT elapsed={elapsed}s iters={iters} failures={}",
                tally.failures
            ));
        }
    }
    logline(&format!(
        "SOAK done: elapsed={}s iters={iters} failures={} base={base} next_base={}\n  {}",
        start.elapsed().as_secs(),
        tally.failures,
        base.wrapping_add(iters),
        tally.summary()
    ));
    assert_eq!(
        tally.failures, 0,
        "the soak found {} failure(s); see FAILURE lines",
        tally.failures
    );
}

/// Plain DataFusion's plan for `EINFOLD_SEED`'s query (default the seed that
/// found einfold#71), reset with `reset_plan_states` and run again,
/// on several partition counts and batch sizes, with and without a small
/// memory pool: prints each setting under which a rerun didn't finish in
/// 3 s. A reproduction for that issue, not a property.
#[test]
#[ignore]
fn host_rerun_hang() {
    let seed = env_u64("EINFOLD_SEED").unwrap_or(70019748);
    let rt = runtime();
    let case = case(seed);
    eprintln!(
        "threads={:?} batch_sizes={:?} settings={:?}",
        case.threads, case.batch_sizes, case.settings
    );
    for threads in [1usize, 2, 4, 6] {
        for batch in [1usize, 3, 8192] {
            for pool in [None, Some(8192usize)] {
                let mut hangs = 0;
                for _ in 0..8 {
                    let settings = check::Settings {
                        pool,
                        no_dynamic_filters: true,
                        ..Default::default()
                    };
                    let host = check::session(threads, None, batch, settings);
                    register(&host, &case.tables);
                    let ran = rt.block_on(check::run(&host, &case.query.sql));
                    if ran.result.is_err() {
                        continue;
                    }
                    let plan = ran.plan.clone().expect("a plan");
                    let reset = datafusion::physical_plan::execution_plan::reset_plan_states(plan)
                        .expect("reset");
                    let task = host.task_ctx();
                    let done = rt.block_on(async move {
                        let run = tokio::spawn(async move {
                            datafusion::physical_plan::collect(reset, task)
                                .await
                                .is_ok()
                        });
                        tokio::time::timeout(std::time::Duration::from_secs(3), run)
                            .await
                            .is_ok()
                    });
                    hangs += usize::from(!done);
                }
                if hangs > 0 {
                    eprintln!("HANG threads={threads} batch={batch} pool={pool:?}: {hangs} of 8");
                }
            }
        }
    }
}

/// Writes generated cases for other hosts to run, such as einfold's Python
/// package through datafusion-python: `EINFOLD_SIM_EXPORT` (a directory)
/// gets one directory per seed, from `EINFOLD_SEED` (default 0) for
/// `EINFOLD_CASES` cases (default 200), each with `query.sql`, `values`
/// (`exact` or `inexact`), `threads`, and each table's partitions as Arrow
/// IPC files `<table>.<partition>.arrow`.
#[test]
#[ignore]
fn export() {
    use datafusion::arrow::ipc::writer::FileWriter;
    let dir = std::env::var("EINFOLD_SIM_EXPORT").expect("EINFOLD_SIM_EXPORT, a directory");
    let base = env_u64("EINFOLD_SEED").unwrap_or(0);
    let cases = env_u64("EINFOLD_CASES").unwrap_or(200);
    for i in 0..cases {
        let seed = base.wrapping_add(i);
        let case = case(seed);
        let at = std::path::Path::new(&dir).join(seed.to_string());
        std::fs::create_dir_all(&at).expect("a case directory");
        std::fs::write(at.join("query.sql"), &case.query.sql).expect("the query");
        let values = match case.values {
            Values::Exact => "exact",
            Values::Inexact => "inexact",
        };
        std::fs::write(at.join("values"), values).expect("the values");
        std::fs::write(at.join("threads"), case.threads[1].to_string()).expect("the threads");
        for t in &case.tables {
            for (p, batches) in t.partitions.iter().enumerate() {
                let file = std::fs::File::create(at.join(format!("{}.{p}.arrow", t.name)))
                    .expect("a table file");
                let mut w = FileWriter::try_new(file, &t.schema).expect("a writer");
                for b in batches {
                    w.write(b).expect("a batch");
                }
                w.finish().expect("a finished file");
            }
        }
    }
}

/// Plain DataFusion's run of `EINFOLD_SEED`'s query, with the case's own
/// settings and with variations of them, 20 times each, reporting runs that
/// didn't finish in 5 s. A reproduction for einfold#71, a hang under a
/// memory pool, not a property.
#[test]
#[ignore]
fn host_hang() {
    let seed = env_u64("EINFOLD_SEED").unwrap_or(95007199);
    let rt = runtime();
    let case = case(seed);
    eprintln!(
        "threads={:?} batch_sizes={:?} settings={:?}",
        case.threads, case.batch_sizes, case.settings
    );
    // Repartitioning back on: the case's settings step around the hang.
    let base = check::Settings {
        no_repartition: false,
        ..case.settings
    };
    let variants = [
        ("the case's", base),
        ("no pool", check::Settings { pool: None, ..base }),
        (
            "no dynamic filters",
            check::Settings {
                no_dynamic_filters: true,
                ..base
            },
        ),
        (
            "no repartition",
            check::Settings {
                no_repartition: true,
                ..base
            },
        ),
    ];
    for (name, settings) in variants {
        for (threads, batch) in [(case.threads[0], case.batch_sizes[0]), (1, 8192), (4, 1)] {
            let mut hangs = 0;
            for _ in 0..20 {
                let host = check::session(threads, None, batch, settings);
                register(&host, &case.tables);
                let sql = case.query.sql.clone();
                let done = rt.block_on(async move {
                    let run =
                        tokio::spawn(async move { check::run(&host, &sql).await.result.is_ok() });
                    tokio::time::timeout(std::time::Duration::from_secs(5), run)
                        .await
                        .is_ok()
                });
                hangs += usize::from(!done);
            }
            eprintln!("{name} settings, threads={threads} batch={batch}: {hangs} of 20 hung");
        }
    }
}
