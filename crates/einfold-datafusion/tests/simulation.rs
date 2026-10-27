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
//! two folds under `UNION ALL`, and a fold of a fold.
//!
//! # Properties
//!
//! Each case runs the query on plain DataFusion (the *host*) and on
//! DataFusion with einfold, on different numbers of partitions and batch
//! sizes, and checks:
//!
//! - **`[error]`** einfold succeeds wherever the host does. A query the host
//!   can answer is one einfold must answer too: moving an expression below a
//!   join must not make it fail on rows the join drops.
//! - **`[masks-error]`** einfold fails wherever the host does, rather than
//!   answering a query whose answer is an error.
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
//! - **`[hash]`** a third session runs every fold detection finds through
//!   einfold's own kernels (`DetectFolds::without_fallback`), so the hash
//!   algorithm is checked on the plans detection finds, not only the dense
//!   one and the host's.
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
    /// … LIMIT` over float keys depends on timing: no `LIMIT` over them
    /// (einfold#63, upstream).
    TopKSignedZeros,
}

const KNOWN: &[Known] = &[Known::TopKSignedZeros];

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
    let options = Options {
        fallible: values == Values::Exact && rng.chance(0.25),
        exact: values == Values::Exact,
        limit: !([ti, tj, tk].into_iter().any(float) && avoids(Known::TopKSignedZeros)),
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
            r.hash,
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
        for (shape, [n, fired, dense, hash, host, domains]) in &self.by_shape {
            let _ = write!(
                s,
                "\n  {shape}: {n} cases, fired in {fired}; dense {dense}, hash {hash}, \
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

/// A failure report: the tag, what differed, and how to replay it.
fn report(case: &Case, tag: &str, what: &str, ours: Option<&check::Ran>) -> String {
    let mut s = format!(
        "[{tag}] {what}\n  shape={:?} values={:?} fallback={} threads={:?} batch_sizes={:?}\n  SQL: {}\n",
        case.query.shape, case.values, case.fallback, case.threads, case.batch_sizes, case.query.sql
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
        ..Outcome::default()
    };
    let sql = &case.query.sql;
    let host = check::session(case.threads[0], None, case.batch_sizes[0]);
    let ours = check::session(case.threads[1], Some(case.fallback), case.batch_sizes[1]);
    register(&host, &case.tables);
    register(&ours, &case.tables);
    let theirs = check::run(&host, sql).await;
    let mine = check::run(&ours, sql).await;
    out.routes = mine.routes();
    let mut fail = |tag: &str, what: String, ran: Option<&check::Ran>| {
        out.failures.push(report(case, tag, &what, ran));
    };
    let (want, got) = match (&theirs.result, &mine.result) {
        (Ok(w), Ok(g)) => (w, g),
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
    // The host's own plan, run again, is the control: where it can't run
    // twice, einfold's needn't either.
    let host_reruns = match check::rerun(&host, &theirs).await {
        Some(Ok(again)) => {
            check::exact(&again) == check::exact(want) || case.values == Values::Inexact
        }
        _ => false,
    };
    out.host_rerun_failed = !host_reruns;
    if bits_must_match && host_reruns {
        match check::rerun(&ours, &mine).await {
            Some(Ok(again)) if check::exact(&again) != check::exact(got) => {
                fail(
                    "rerun",
                    diff(&check::exact(&again), &check::exact(got)),
                    Some(&mine),
                );
            }
            Some(Err(e)) => fail(
                "rerun",
                format!("running the reset plan again failed: {e}"),
                Some(&mine),
            ),
            _ => {}
        }
    }
    if out.deterministic {
        let other = check::session(case.threads[2], Some(case.fallback), case.batch_sizes[2]);
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
    // Every fold through einfold's kernels, the hash algorithm included.
    if case.fallback && r.folds > 0 && case.seed.is_multiple_of(3) {
        let hash = check::session(case.threads[2], Some(false), case.batch_sizes[2]);
        register(&hash, &case.tables);
        let ran = check::run(&hash, sql).await;
        match &ran.result {
            Err(e) => fail(
                "hash",
                format!("failed where the host succeeded: {e}"),
                Some(&ran),
            ),
            Ok(b) if case.values == Values::Exact => {
                let (g, w) = (check::exact(b), check::exact(want));
                if g != w {
                    fail("hash", diff(&g, &w), Some(&ran));
                }
            }
            Ok(b) => {
                if let Err(e) = within_bound(case, b, want).await {
                    fail("hash", e, Some(&ran));
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
        let ctx = check::session(1, None, 8192);
        register(&ctx, tables);
        ctx
    };
    let (n, m) = (aside(ones), aside(mags));
    let sql = &case.query.sql;
    let (Ok(n), Ok(m)) = (
        check::run(&n, sql).await.result,
        check::run(&m, sql).await.result,
    ) else {
        return Err("the twin tables' query failed".into());
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
        ("the hash algorithm ran", ran(3) > 0),
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
    while start.elapsed().as_secs() < budget {
        let seed = base.wrapping_add(iters);
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
