// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Spike S20: tilings in egglog, checked against Cubed's matmul plans.
//!
//! The rules in `tiles.egg` build every tiled plan for C = A·B over candidate
//! tile sizes. Extraction uses Cubed's own projected-memory formulas, and a
//! plan whose largest task exceeds the budget gets infinite cost.
//!
//! Experiments:
//! 1. Reproduce Cubed's plans (cubed_reference.py), with Cubed's fan-in of 4.
//! 2. The same, with the fan-in free to choose.
//! 3. A budget sweep where the storage chunks are too big for the budget.
//! 4. Randomized check: extraction never exceeds the budget, and its cost
//!    equals the minimum found by brute-force enumeration of the same space.

use egglog::extract::{Cost, CostModel, Extractor};
use egglog::prelude::exprs;
use egglog::sort::S;
use egglog::{ArcSort, EGraph, Enode, Function, Term, TermDag, TermId, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

const RULES: &str = include_str!("../tiles.egg");
const MB: f64 = 1e6;
const BYTES: f64 = 8.0; // float64
const RESERVED: f64 = 100.0 * MB; // Cubed's reserved_mem in cubed_reference.py
const TASK_OVERHEAD: f64 = 10.0 * MB; // cost of one task, in bytes-equivalent of IO

#[derive(Clone)]
struct Case {
    name: String,
    n: i64,
    storage: BTreeMap<String, (i64, i64)>,
    budget: f64,
    splits: Vec<i64>,
    /// Propose tile sizes beyond the storage tiling (least common multiples, halvings).
    propose: bool,
}

fn ceil_div(a: i64, b: i64) -> i64 {
    (a + b - 1) / b
}

/// Candidate tile sizes per dimension. Without proposing, each dimension
/// gets only the storage tile sizes along it, which leaves Cubed's choices.
fn candidates_by_dim(c: &Case) -> BTreeMap<&'static str, BTreeSet<i64>> {
    let (a, b) = (c.storage["A"], c.storage["B"]);
    if !c.propose {
        return [("i", [a.0].into()), ("k", [a.1, b.0].into()), ("j", [b.1].into())].into_iter().collect();
    }
    let all = candidates(c);
    ["i", "k", "j"].into_iter().map(|d| (d, all.clone())).collect()
}

/// The tile-size proposer: storage tile sizes, their least common multiples,
/// and halvings of the extent (rechunker-style sizes that fit smaller budgets).
fn candidates(c: &Case) -> BTreeSet<i64> {
    let mut s: BTreeSet<i64> = c.storage.values().flat_map(|(r, k)| [*r, *k]).collect();
    let base: Vec<i64> = s.iter().cloned().collect();
    for a in &base {
        for b in &base {
            let l = a / gcd(*a, *b) * b;
            if l <= c.n {
                s.insert(l);
            }
        }
    }
    let mut t = c.n;
    while t >= c.n / 32 {
        s.insert(t);
        t /= 2;
    }
    s
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

// ---------------------------------------------------------------------------
// Cost model: Cubed's projected memory, IO bytes and task counts

#[derive(Clone, Debug, Default)]
struct Est {
    cost: f64,
    mem: f64,
    io: f64,
    tasks: f64,
    tile: (i64, i64),
    m: i64,
    num: Option<i64>,
    name: Option<String>,
}
impl PartialEq for Est {
    fn eq(&self, o: &Self) -> bool {
        self.cost.total_cmp(&o.cost).is_eq()
    }
}
impl Eq for Est {}
impl PartialOrd for Est {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Est {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.cost.total_cmp(&o.cost)
    }
}
impl Cost for Est {
    fn identity() -> Self {
        Est::default()
    }
    fn unit() -> Self {
        Est { cost: 1.0, ..Est::default() }
    }
    fn combine(self, o: &Self) -> Self {
        Est { cost: self.cost + o.cost, ..self }
    }
}

struct Model {
    c: Case,
}

impl Model {
    fn finish(&self, mut e: Est) -> Est {
        e.cost = if e.mem > self.c.budget { f64::INFINITY } else { e.io + e.tasks * TASK_OVERHEAD };
        e
    }
    fn stored(&self, name: &str) -> Est {
        let tile = self.c.storage[name];
        Est { tile, ..Est::default() }
    }
    /// Cubed's rechunk uses rechunker's multi-stage algorithm. Here: one stage,
    /// each task holding a source and a target tile, read and written once.
    fn retile(&self, x: &Est, r: i64, c: i64) -> Est {
        let n = self.c.n as f64;
        let area = |t: (i64, i64)| (t.0 * t.1) as f64 * BYTES;
        let mem = RESERVED + 2.0 * area(x.tile) + 2.0 * area((r, c));
        let tasks = (ceil_div(self.c.n, r) * ceil_div(self.c.n, c)) as f64;
        let own = Est { mem, io: 2.0 * n * n * BYTES, tasks, tile: (r, c), ..Est::default() };
        self.finish(Est { mem: own.mem.max(x.mem), io: own.io + x.io, tasks: own.tasks + x.tasks, ..own })
    }
    /// Cubed's blockwise: inputs × (1 + read copies) + output × (1 + write copies).
    fn block_mm(&self, x: &Est, y: &Est) -> Est {
        if x.tile.1 != y.tile.0 {
            return Est { cost: f64::INFINITY, ..Est::default() };
        }
        let (ti, tk, tj) = (x.tile.0, x.tile.1, y.tile.1);
        let n = self.c.n;
        let mem = RESERVED + 2.0 * BYTES * (ti * tk + tk * tj) as f64 + 2.0 * BYTES * (ti * tj) as f64;
        let tasks = (ceil_div(n, ti) * ceil_div(n, tk) * ceil_div(n, tj)) as f64;
        let m = ceil_div(n, tk);
        let io = tasks * BYTES * (ti * tk + tk * tj) as f64 + (n * n * m) as f64 * BYTES;
        let own = Est { mem, io, tasks, tile: (ti, tj), m, ..Est::default() };
        self.finish(Est {
            mem: mem.max(x.mem).max(y.mem),
            io: io + x.io + y.io,
            tasks: tasks + x.tasks + y.tasks,
            ..own
        })
    }
    /// Cubed's partial_reduce: one input chunk at a time (× 2), an extra chunk
    /// plus two output-sized buffers, and the output (× 2): reserved + 7 chunks.
    fn partial_sum(&self, x: &Est, s: i64) -> Est {
        let n = self.c.n;
        let chunk = BYTES * (x.tile.0 * x.tile.1) as f64;
        let mem = RESERVED + 7.0 * chunk;
        let m = ceil_div(x.m, s);
        let tasks = (ceil_div(n, x.tile.0) * ceil_div(n, x.tile.1) * m) as f64;
        let io = ((x.m + m) * n * n) as f64 * BYTES;
        self.finish(Est { mem: mem.max(x.mem), io: io + x.io, tasks: tasks + x.tasks, tile: x.tile, m, ..Est::default() })
    }
}

impl CostModel<Est> for Model {
    fn fold(&self, head: &str, ch: &[Est], _: Est) -> Est {
        let num = |k: usize| ch[k].num.expect("number");
        if ch.iter().any(|c| c.cost.is_infinite()) {
            return Est { cost: f64::INFINITY, ..Est::default() };
        }
        match head {
            "Stored" => self.stored(ch[0].name.as_deref().unwrap()),
            "Retile" => self.retile(&ch[0], num(1), num(2)),
            "BlockMM" => self.block_mm(&ch[0], &ch[1]),
            "PartialSum" => self.partial_sum(&ch[0], num(1)),
            // Names of e-classes, not operations: never chosen over a real node.
            _ => Est { cost: f64::INFINITY, ..Est::default() },
        }
    }
    fn enode_cost(&self, _: &EGraph, _: &Function, _: &Enode<'_>) -> Est {
        Est::default()
    }
    fn base_value_cost(&self, eg: &EGraph, sort: &ArcSort, v: Value) -> Est {
        match sort.name() {
            "String" => Est { name: Some(eg.value_to_base::<S>(v).0.clone()), ..Est::default() },
            "i64" => Est { num: Some(eg.value_to_base::<i64>(v)), ..Est::default() },
            _ => Est::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Plans as our own terms, for printing and for brute force

#[derive(Clone, Debug)]
enum Plan {
    Stored(String),
    Retile(Box<Plan>, i64, i64),
    BlockMM(Box<Plan>, Box<Plan>),
    PartialSum(Box<Plan>, i64),
}

fn from_term(dag: &TermDag, id: TermId) -> Plan {
    let num = |id: TermId| match dag.get(id) {
        Term::Lit(egglog::ast::Literal::Int(i)) => *i,
        t => panic!("expected int, got {t:?}"),
    };
    match dag.get(id) {
        Term::App(h, ch) => match h.as_str() {
            "Stored" => match dag.get(ch[0]) {
                Term::Lit(egglog::ast::Literal::String(s)) => Plan::Stored(s.clone()),
                t => panic!("{t:?}"),
            },
            "Retile" => Plan::Retile(Box::new(from_term(dag, ch[0])), num(ch[1]), num(ch[2])),
            "BlockMM" => Plan::BlockMM(Box::new(from_term(dag, ch[0])), Box::new(from_term(dag, ch[1]))),
            "PartialSum" => Plan::PartialSum(Box::new(from_term(dag, ch[0])), num(ch[1])),
            h => panic!("unexpected {h}"),
        },
        t => panic!("{t:?}"),
    }
}

fn eval(m: &Model, p: &Plan, ops: &mut Vec<String>) -> Est {
    let e = match p {
        Plan::Stored(a) => return m.stored(a),
        Plan::Retile(x, r, c) => {
            let x = eval(m, x, ops);
            m.retile(&x, *r, *c)
        }
        Plan::BlockMM(x, y) => {
            let (x, y) = (eval(m, x, ops), eval(m, y, ops));
            m.block_mm(&x, &y)
        }
        Plan::PartialSum(x, s) => {
            let x = eval(m, x, ops);
            m.partial_sum(&x, *s)
        }
    };
    let head = match p {
        Plan::Retile(_, r, c) => format!("rechunk to ({r}, {c})"),
        Plan::BlockMM(..) => format!("blockwise matmul, tiles ({}, {}), {} partials", e.tile.0, e.tile.1, e.m),
        Plan::PartialSum(_, s) => format!("partial_reduce, fan-in {s}, {} left", e.m),
        Plan::Stored(_) => unreachable!(),
    };
    // Per-op numbers, not cumulative: recompute this op alone.
    let own = match p {
        Plan::Retile(x, r, c) => m.retile(&eval(m, x, &mut vec![]), *r, *c).tasks - eval(m, x, &mut vec![]).tasks,
        Plan::BlockMM(x, y) => {
            let (x, y) = (eval(m, x, &mut vec![]), eval(m, y, &mut vec![]));
            m.block_mm(&x, &y).tasks - x.tasks - y.tasks
        }
        Plan::PartialSum(x, s) => {
            let x = eval(m, x, &mut vec![]);
            m.partial_sum(&x, *s).tasks - x.tasks
        }
        Plan::Stored(_) => 0.0,
    };
    let own_mem = match p {
        Plan::Retile(x, r, c) => {
            let x = eval(m, x, &mut vec![]);
            RESERVED + 2.0 * BYTES * (x.tile.0 * x.tile.1 + r * c) as f64
        }
        Plan::BlockMM(..) => {
            let (ti, tj) = e.tile;
            let tk = ceil_div(m.c.n, e.m);
            RESERVED + 2.0 * BYTES * (ti * tk + tk * tj + ti * tj) as f64
        }
        Plan::PartialSum(..) => RESERVED + 7.0 * BYTES * (e.tile.0 * e.tile.1) as f64,
        Plan::Stored(_) => 0.0,
    };
    ops.push(format!("{head}: tasks={own}, projected_mem={:.0} MB", own_mem / MB));
    e
}

/// Every plan in the same space, for the brute-force check.
fn brute_force(m: &Model, cands: &BTreeMap<&str, BTreeSet<i64>>) -> f64 {
    let view = |a: &str, r: i64, c: i64| -> Est {
        let s = m.stored(a);
        if s.tile == (r, c) { s } else { m.retile(&s, r, c) }
    };
    fn reduce(m: &Model, x: Est, best: &mut f64) {
        if x.m == 1 {
            *best = best.min(x.cost);
            return;
        }
        for &s in &m.c.splits {
            reduce(m, m.partial_sum(&x, s), best);
        }
    }
    let mut best = f64::INFINITY;
    for &ti in &cands["i"] {
        for &tk in &cands["k"] {
            for &tj in &cands["j"] {
                let p = m.block_mm(&view("A", ti, tk), &view("B", tk, tj));
                if p.m == 1 {
                    best = best.min(p.cost);
                } else {
                    reduce(m, p, &mut best);
                }
            }
        }
    }
    best
}

struct Out {
    cost: f64,
    mem: f64,
    ops: Vec<String>,
    nodes: usize,
    secs: f64,
}

fn run(c: &Case) -> Out {
    let t0 = Instant::now();
    let mut eg = EGraph::default();
    let mut prog = String::from(RULES);
    prog.push_str(&format!("(set (extent) {})\n", c.n));
    for (d, ts) in candidates_by_dim(c) {
        for t in ts {
            prog.push_str(&format!("(cand \"{d}\" {t})\n"));
        }
    }
    prog.push_str("(arrdims \"A\" \"i\" \"k\")\n(arrdims \"B\" \"k\" \"j\")\n");
    for s in &c.splits {
        prog.push_str(&format!("(split {s})\n"));
    }
    for (a, (r, k)) in &c.storage {
        prog.push_str(&format!("(storage \"{a}\" {r} {k})\n"));
    }
    prog.push_str("(let $root (MatMul \"A\" \"B\"))\n(run-schedule (saturate (run tiles)))\n");
    eg.parse_and_run_program(None, &prog).expect("program");
    let nodes = eg.num_tuples();
    let model = Model { c: c.clone() };
    let (sort, value) = eg.eval_expr(&exprs::var("$root")).expect("root");
    let ex = Extractor::compute_costs_from_rootsorts(Some(vec![sort]), &eg, Model { c: c.clone() });
    let mut dag = TermDag::default();
    let (cost, term) = ex.extract_best(&eg, &mut dag, value).expect("extract");
    let secs = t0.elapsed().as_secs_f64();
    if cost.cost.is_infinite() {
        return Out { cost: f64::INFINITY, mem: f64::NAN, ops: vec!["no plan fits the budget".into()], nodes, secs };
    }
    let plan = from_term(&dag, term);
    let mut ops = Vec::new();
    let e = eval(&model, &plan, &mut ops);
    Out { cost: cost.cost, mem: e.mem, ops, nodes, secs }
}

fn case(name: &str, ca: (i64, i64), cb: (i64, i64), budget_mb: f64, splits: &[i64], propose: bool) -> Case {
    Case {
        name: name.into(),
        n: 20000,
        storage: [("A".to_string(), ca), ("B".to_string(), cb)].into_iter().collect(),
        budget: budget_mb * MB,
        splits: splits.to_vec(),
        propose,
    }
}

fn main() {
    let cubed_cases = [
        ("aligned 5000x2000 / 2000x5000, 2 GB", (5000, 2000), (2000, 5000), 2000.0),
        ("aligned square 2000, 1 GB", (2000, 2000), (2000, 2000), 1000.0),
        ("misaligned k: 2000 vs 3000, 1 GB", (2000, 2000), (3000, 2000), 1000.0),
        ("large chunks 8000, 1 GB", (8000, 8000), (8000, 8000), 1000.0),
    ];
    for (title, splits, propose) in [
        ("Storage tilings only, fan-in 4: Cubed's choices", vec![4], false),
        ("Proposed tilings, fan-in 4", vec![4], true),
        ("Proposed tilings, fan-in free (2, 4, 8, 16)", vec![2, 4, 8, 16], true),
    ] {
        println!("## {title}\n");
        for (name, ca, cb, mb) in cubed_cases {
            let c = case(name, ca, cb, mb, &splits, propose);
            let o = run(&c);
            println!("### {name}\n");
            println!("e-graph tuples: {}; saturate + extract: {:.3} s; max projected memory: {:.0} MB; cost: {:.3e}\n", o.nodes, o.secs, o.mem / MB, o.cost);
            for op in &o.ops {
                println!("- {op}");
            }
            println!();
        }
    }

    println!("## Budget sweep: storage chunks 8000 × 8000, fan-in free\n");
    println!("| Budget (MB) | Max projected memory (MB) | Plan |");
    println!("|---|---|---|");
    for mb in [1000.0, 1200.0, 1300.0, 1500.0, 2000.0, 3000.0, 4000.0] {
        let c = case("sweep", (8000, 8000), (8000, 8000), mb, &[2, 4, 8, 16], true);
        let o = run(&c);
        println!("| {mb} | {:.0} | {} |", o.mem / MB, o.ops.join("; "));
    }

    println!("\n## Randomized check against brute force\n");
    let sizes = [1000, 2000, 2500, 3000, 4000, 5000, 8000, 10000];
    let budgets = [150.0, 300.0, 600.0, 1000.0, 2000.0, 4000.0];
    let mut seed: u64 = 42;
    let mut next = |k: usize| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 33) as usize) % k
    };
    let (mut trials, mut over, mut mismatched, mut infeasible) = (0, 0, 0, 0);
    for _ in 0..200 {
        let ca = (sizes[next(8)], sizes[next(8)]);
        let cb = (sizes[next(8)], sizes[next(8)]);
        let c = case("random", ca, cb, budgets[next(6)], &[2, 4, 8, 16], true);
        let o = run(&c);
        let m = Model { c: c.clone() };
        let bf = brute_force(&m, &candidates_by_dim(&c));
        trials += 1;
        if o.cost.is_infinite() {
            infeasible += 1;
        } else if o.mem > c.budget {
            over += 1;
        }
        let same = (o.cost.is_infinite() && bf.is_infinite()) || ((o.cost - bf).abs() <= 1e-9 * bf.abs());
        if !same {
            mismatched += 1;
            println!("mismatch: A {:?} B {:?} budget {} MB: extracted {:.4e}, brute force {:.4e}", ca, cb, c.budget / MB, o.cost, bf);
        }
    }
    println!("trials: {trials}; extracted plans over budget: {over}; no feasible plan: {infeasible}; cost differs from brute force: {mismatched}");
}
