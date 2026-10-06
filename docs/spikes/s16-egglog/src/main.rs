// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Spike S16: einfold's algebraic rewrites in egglog, with a hybrid contraction planner.
//!
//! Pipeline per experiment:
//! 1. Load `rules.egg`, declare each operand's dimensions, and add the query term.
//! 2. Saturate with egglog (algebra rules, optionally associativity), within limits.
//! 3. Extract with a custom cost model that estimates rows like einfold's planner.
//! 4. Hybrid only: reorder each sum-product region with a dynamic-programming planner.
//!
//! Costs use the uniform-density estimate from the design doc, section 9.3.

use egglog::ast::Literal;
use egglog::extract::{Cost, CostModel, Extractor};
use egglog::prelude::exprs;
use egglog::sort::S;
use egglog::{ArcSort, EGraph, Enode, Function, Term, TermDag, TermId, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

const RULES: &str = include_str!("../rules.egg");

// ---------------------------------------------------------------------------
// Problems

#[derive(Clone, Debug)]
struct Operand {
    dims: Vec<String>,
    nnz: f64,
}

#[derive(Clone, Debug)]
struct Problem {
    name: String,
    extents: BTreeMap<String, f64>,
    operands: BTreeMap<String, Operand>,
    root: String,
}

fn ext(pairs: &[(&str, f64)]) -> BTreeMap<String, f64> {
    pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

fn op(dims: &[&str], nnz: f64) -> Operand {
    Operand { dims: dims.iter().map(|d| d.to_string()).collect(), nnz }
}

fn dense(extents: &BTreeMap<String, f64>, dims: &[&str]) -> f64 {
    dims.iter().map(|d| extents[*d]).product()
}

// ---------------------------------------------------------------------------
// Cost model shared by egglog extraction and the planner

#[derive(Clone, Debug)]
struct Est {
    cost: f64,
    rows: f64,
    dims: BTreeSet<String>,
    tag: Option<String>,
}

impl Est {
    fn zero() -> Self {
        Est { cost: 0.0, rows: 0.0, dims: BTreeSet::new(), tag: None }
    }
}
impl PartialEq for Est {
    fn eq(&self, other: &Self) -> bool {
        self.cost.total_cmp(&other.cost).is_eq()
    }
}
impl Eq for Est {}
impl PartialOrd for Est {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Est {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cost.total_cmp(&other.cost)
    }
}
impl Cost for Est {
    fn identity() -> Self {
        Est::zero()
    }
    fn unit() -> Self {
        Est { cost: 1.0, ..Est::zero() }
    }
    fn combine(self, other: &Self) -> Self {
        Est { cost: self.cost + other.cost, ..self }
    }
}

#[derive(Clone)]
struct Model {
    extents: BTreeMap<String, f64>,
    operands: BTreeMap<String, Operand>,
}

impl Model {
    fn dense(&self, dims: &BTreeSet<String>) -> f64 {
        dims.iter().map(|d| self.extents[d]).product()
    }
    /// Scanning an operand costs its rows.
    fn leaf(&self, name: &str) -> Est {
        let o = &self.operands[name];
        Est { cost: o.nnz, rows: o.nnz, dims: o.dims.iter().cloned().collect(), tag: None }
    }
    /// A join on shared dimensions; rows estimated assuming uniform density.
    fn mul(&self, a: &Est, b: &Est) -> Est {
        let shared: f64 = a.dims.intersection(&b.dims).map(|d| self.extents[d]).product();
        let dims: BTreeSet<String> = a.dims.union(&b.dims).cloned().collect();
        let rows = (a.rows * b.rows / shared).min(self.dense(&dims));
        Est { cost: a.cost + b.cost + rows, rows, dims, tag: None }
    }
    /// A pointwise sum reads both inputs.
    fn add(&self, a: &Est, b: &Est) -> Est {
        let dims: BTreeSet<String> = a.dims.union(&b.dims).cloned().collect();
        let rows = (a.rows + b.rows).min(self.dense(&dims));
        Est { cost: a.cost + b.cost + a.rows + b.rows, rows, dims, tag: None }
    }
    /// An aggregation reads its input once.
    fn sum(&self, i: &str, a: &Est) -> Est {
        let mut dims = a.dims.clone();
        dims.remove(i);
        let rows = a.rows.min(self.dense(&dims));
        Est { cost: a.cost + a.rows, rows, dims, tag: None }
    }
    fn scale(&self, a: &Est) -> Est {
        Est { cost: a.cost + a.rows, ..a.clone() }
    }
    fn cost(&self, e: &E) -> Est {
        match e {
            E::T(n) => self.leaf(n),
            E::Mul(a, b) => self.mul(&self.cost(a), &self.cost(b)),
            E::Add(a, b) => self.add(&self.cost(a), &self.cost(b)),
            E::Sum(i, a) => self.sum(i, &self.cost(a)),
            E::Scale(_, a) => self.scale(&self.cost(a)),
        }
    }
}

impl CostModel<Est> for Model {
    fn fold(&self, head: &str, ch: &[Est], _head_cost: Est) -> Est {
        let tag = |k: usize| ch[k].tag.clone().expect("string child");
        match head {
            "T" => self.leaf(&tag(0)),
            "Mul" => self.mul(&ch[0], &ch[1]),
            "Add" => self.add(&ch[0], &ch[1]),
            "Sum" => self.sum(&tag(0), &ch[1]),
            "Scale" => self.scale(&ch[1]),
            _ => Est::zero(),
        }
    }
    fn enode_cost(&self, _: &EGraph, _: &Function, _: &Enode<'_>) -> Est {
        Est::zero()
    }
    fn base_value_cost(&self, egraph: &EGraph, sort: &ArcSort, value: Value) -> Est {
        if sort.name() == "String" {
            let s: S = egraph.value_to_base::<S>(value);
            Est { tag: Some(s.0.clone()), ..Est::zero() }
        } else {
            Est::zero()
        }
    }
}

// ---------------------------------------------------------------------------
// Our own copy of the term, for planning and printing

#[derive(Clone, Debug, PartialEq)]
enum E {
    T(String),
    Mul(Box<E>, Box<E>),
    Add(Box<E>, Box<E>),
    Sum(String, Box<E>),
    Scale(String, Box<E>),
}

fn lit(dag: &TermDag, id: TermId) -> String {
    match dag.get(id) {
        Term::Lit(Literal::String(s)) => s.clone(),
        t => panic!("expected a string literal, got {t:?}"),
    }
}

fn from_term(dag: &TermDag, id: TermId) -> E {
    match dag.get(id) {
        Term::App(h, ch) => match h.as_str() {
            "T" => E::T(lit(dag, ch[0])),
            "Mul" => E::Mul(Box::new(from_term(dag, ch[0])), Box::new(from_term(dag, ch[1]))),
            "Add" => E::Add(Box::new(from_term(dag, ch[0])), Box::new(from_term(dag, ch[1]))),
            "Sum" => E::Sum(lit(dag, ch[0]), Box::new(from_term(dag, ch[1]))),
            "Scale" => E::Scale(lit(dag, ch[0]), Box::new(from_term(dag, ch[1]))),
            other => panic!("unexpected head {other}"),
        },
        t => panic!("unexpected term {t:?}"),
    }
}

fn show(e: &E) -> String {
    match e {
        E::T(n) => n.clone(),
        E::Mul(a, b) => format!("({} * {})", show(a), show(b)),
        E::Add(a, b) => format!("({} + {})", show(a), show(b)),
        E::Sum(i, a) => format!("Σ{i} {}", show(a)),
        E::Scale(i, a) => format!("n_{i}·{}", show(a)),
    }
}

// ---------------------------------------------------------------------------
// Hybrid planner: reorder each sum-product region (a tree of Mul and Sum nodes)

struct Planner<'a> {
    m: &'a Model,
}

impl Planner<'_> {
    fn plan(&self, e: &E) -> E {
        match e {
            E::Mul(..) | E::Sum(..) => self.plan_region(e),
            E::Add(a, b) => E::Add(Box::new(self.plan(a)), Box::new(self.plan(b))),
            E::Scale(i, a) => E::Scale(i.clone(), Box::new(self.plan(a))),
            E::T(_) => e.clone(),
        }
    }

    fn collect(&self, e: &E, ops: &mut Vec<E>, sums: &mut Vec<(String, usize, usize)>) {
        match e {
            E::Mul(a, b) => {
                self.collect(a, ops, sums);
                self.collect(b, ops, sums);
            }
            E::Sum(i, a) => {
                let start = ops.len();
                self.collect(a, ops, sums);
                sums.push((i.clone(), start, ops.len()));
            }
            other => ops.push(self.plan(other)),
        }
    }

    fn plan_region(&self, e: &E) -> E {
        let (mut ops, mut sums) = (Vec::new(), Vec::new());
        self.collect(e, &mut ops, &mut sums);
        let ests: Vec<Est> = ops.iter().map(|o| self.m.cost(o)).collect();
        // Hoisting a sum out of its subtree is only valid if no operand outside
        // the subtree uses the same dimension name.
        let hoistable = sums.iter().all(|(i, s, t)| {
            (0..ops.len()).filter(|k| k < s || k >= t).all(|k| !ests[k].dims.contains(i))
        });
        if !hoistable || ops.len() > 16 {
            return e.clone();
        }
        let summed: BTreeSet<String> = sums.iter().map(|(i, _, _)| i.clone()).collect();
        let all: BTreeSet<String> = ests.iter().flat_map(|x| x.dims.iter().cloned()).collect();
        let out: BTreeSet<String> = all.difference(&summed).cloned().collect();

        let n = ops.len();
        let full = (1usize << n) - 1;
        let dims_of = |set: usize| -> BTreeSet<String> {
            (0..n).filter(|k| set >> k & 1 == 1).flat_map(|k| ests[k].dims.iter().cloned()).collect()
        };
        // Dimensions a subset must keep: the output, plus anything the rest still needs.
        let keep = |set: usize| -> BTreeSet<String> {
            let mut need = out.clone();
            need.extend(dims_of(full & !set));
            dims_of(set).intersection(&need).cloned().collect()
        };
        let finish = |mut est: Est, mut tree: E, set: usize| -> (Est, E) {
            let k = keep(set);
            for d in est.dims.clone().difference(&k) {
                est = self.m.sum(d, &est);
                tree = E::Sum(d.clone(), Box::new(tree));
            }
            (est, tree)
        };
        let mut best: Vec<Option<(Est, E)>> = vec![None; full + 1];
        for k in 0..n {
            best[1 << k] = Some(finish(ests[k].clone(), ops[k].clone(), 1 << k));
        }
        for set in 1..=full {
            if set.count_ones() < 2 {
                continue;
            }
            let mut a = (set - 1) & set;
            while a > 0 {
                let b = set & !a;
                if a < b {
                    let (ea, ta) = best[a].clone().unwrap();
                    let (eb, tb) = best[b].clone().unwrap();
                    let joined = self.m.mul(&ea, &eb);
                    let cand = finish(joined, E::Mul(Box::new(ta), Box::new(tb)), set);
                    if best[set].as_ref().map_or(true, |(c, _)| cand.0.cost < c.cost) {
                        best[set] = Some(cand);
                    }
                }
                a = (a - 1) & set;
            }
        }
        // A summed dimension that no operand has is broadcast factoring:
        // Σ_i B = n_i · B. Dropping it would change the result.
        let mut tree = best[full].clone().unwrap().1;
        for d in summed.difference(&all) {
            tree = E::Scale(d.clone(), Box::new(tree));
        }
        tree
    }
}

// ---------------------------------------------------------------------------
// Running egglog

struct Limits {
    max_iters: usize,
    max_tuples: usize,
    max_time: Duration,
}

struct Run {
    extracted: E,
    tuples: usize,
    iters: usize,
    saturated: bool,
    secs: f64,
}

fn run_egglog(p: &Problem, assoc: bool, threads: usize, lim: &Limits) -> Run {
    let mut eg = EGraph::default();
    eg.set_num_threads(threads);
    let mut prog = String::from(RULES);
    for (name, o) in &p.operands {
        let dims: Vec<String> = o.dims.iter().map(|d| format!("\"{d}\"")).collect();
        prog.push_str(&format!("(set (dims (T \"{name}\")) (set-of {}))\n", dims.join(" ")));
    }
    prog.push_str(&format!("(let $root {})\n", p.root));
    prog.push_str("(run-schedule (saturate (run analysis)))\n");
    eg.parse_and_run_program(None, &prog).expect("program");

    let step = if assoc {
        "(run-schedule (saturate (run analysis)) (run algebra) (run assoc))"
    } else {
        "(run-schedule (saturate (run analysis)) (run algebra))"
    };
    let start = Instant::now();
    let (mut iters, mut saturated) = (0, false);
    while iters < lim.max_iters {
        let before = eg.num_tuples();
        eg.parse_and_run_program(None, step).expect("step");
        iters += 1;
        let after = eg.num_tuples();
        if after == before {
            saturated = true;
            break;
        }
        if after > lim.max_tuples || start.elapsed() > lim.max_time {
            break;
        }
    }
    let secs = start.elapsed().as_secs_f64();
    let tuples = eg.num_tuples();

    let model = Model { extents: p.extents.clone(), operands: p.operands.clone() };
    let (sort, value) = eg.eval_expr(&exprs::var("$root")).expect("root");
    let extractor = Extractor::compute_costs_from_rootsorts(Some(vec![sort.clone()]), &eg, model);
    let mut dag = TermDag::default();
    let (_, term) = extractor.extract_best(&eg, &mut dag, value).expect("extract");
    Run { extracted: from_term(&dag, term), tuples, iters, saturated, secs }
}

fn parse_root(p: &Problem) -> E {
    // Parse the root by running it through egglog with no rewrites.
    let mut eg = EGraph::default();
    let mut prog = String::from(RULES);
    for (name, o) in &p.operands {
        let dims: Vec<String> = o.dims.iter().map(|d| format!("\"{d}\"")).collect();
        prog.push_str(&format!("(set (dims (T \"{name}\")) (set-of {}))\n", dims.join(" ")));
    }
    prog.push_str(&format!("(let $root {})\n", p.root));
    eg.parse_and_run_program(None, &prog).expect("program");
    let model = Model { extents: p.extents.clone(), operands: p.operands.clone() };
    let (sort, value) = eg.eval_expr(&exprs::var("$root")).expect("root");
    let extractor = Extractor::compute_costs_from_rootsorts(Some(vec![sort]), &eg, model);
    let mut dag = TermDag::default();
    let (_, term) = extractor.extract_best(&eg, &mut dag, value).expect("extract");
    from_term(&dag, term)
}

// ---------------------------------------------------------------------------
// Experiments

fn problems() -> Vec<Problem> {
    let mut ps = Vec::new();

    // E1: SPORES's PNMF example: sum(W·H) = Σ_k (Σ_i W_ik)(Σ_j H_kj).
    let e = ext(&[("i", 1e5), ("j", 1e5), ("k", 100.0)]);
    ps.push(Problem {
        name: "E1 sum(W·H) (SPORES PNMF)".into(),
        operands: [("W", op(&["i", "k"], dense(&e, &["i", "k"]))), ("H", op(&["k", "j"], dense(&e, &["k", "j"])))]
            .into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        extents: e,
        root: r#"(Sum "i" (Sum "j" (Sum "k" (Mul (T "W") (T "H")))))"#.into(),
    });

    // E2: distributivity. X ∘ (A + B), summed. Expanding wins only when X is sparse.
    for (label, x_nnz) in [("E2a X·(A+B), X sparse", 1e4), ("E2b X·(A+B), X dense", 1e8)] {
        let e = ext(&[("i", 1e4), ("j", 1e4)]);
        let d = dense(&e, &["i", "j"]);
        ps.push(Problem {
            name: label.into(),
            operands: [("X", op(&["i", "j"], x_nnz)), ("A", op(&["i", "j"], d)), ("B", op(&["i", "j"], d))]
                .into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
            extents: e,
            root: r#"(Sum "i" (Sum "j" (Mul (T "X") (Add (T "A") (T "B")))))"#.into(),
        });
    }

    // E3: broadcast factoring. A latitude-weighted mean, grouped by time,
    // and its normalizer Σ w over all three dimensions.
    let e = ext(&[("t", 1000.0), ("lat", 721.0), ("lon", 1440.0)]);
    let ops3: BTreeMap<String, Operand> = [("w", op(&["lat"], 721.0)), ("x", op(&["t", "lat", "lon"], dense(&e, &["t", "lat", "lon"])))]
        .into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    ps.push(Problem {
        name: "E3a Σ w(lat)·x(t,lat,lon) by t".into(),
        operands: ops3.clone(),
        extents: e.clone(),
        root: r#"(Sum "lat" (Sum "lon" (Mul (T "w") (T "x"))))"#.into(),
    });
    ps.push(Problem {
        name: "E3b Σ_{t,lat,lon} w(lat)".into(),
        operands: ops3,
        extents: e,
        root: r#"(Sum "t" (Sum "lat" (Sum "lon" (T "w"))))"#.into(),
    });

    // E4: ddx two-layer gradient, W1bar[d,h] = Σ_n Σ_o X[n,d] Ybar[n,o] W2[h,o],
    // written in the expensive order X·(Ybar·W2).
    let e = ext(&[("n", 5e4), ("d", 16.0), ("o", 8.0), ("h", 64.0)]);
    ps.push(Problem {
        name: "E4 ddx gradient X·(Ȳ·W2)".into(),
        operands: [
            ("X", op(&["n", "d"], dense(&e, &["n", "d"]))),
            ("Ybar", op(&["n", "o"], dense(&e, &["n", "o"]))),
            ("W2", op(&["h", "o"], dense(&e, &["h", "o"]))),
        ]
        .into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        extents: e,
        root: r#"(Sum "n" (Sum "o" (Mul (T "X") (Mul (T "Ybar") (T "W2")))))"#.into(),
    });
    ps
}

/// A dense matrix chain A1·A2·…·An, written right-nested, for the scaling test.
fn chain(n: usize) -> Problem {
    let sizes = [100.0, 10.0, 1000.0, 20.0, 500.0, 5.0, 2000.0, 50.0, 300.0, 30.0, 800.0, 15.0, 700.0, 40.0, 900.0, 25.0, 600.0];
    let dim = |k: usize| format!("d{k}");
    let extents: BTreeMap<String, f64> = (0..=n).map(|k| (dim(k), sizes[k])).collect();
    let mut operands = BTreeMap::new();
    for k in 1..=n {
        operands.insert(format!("A{k}"), Operand { dims: vec![dim(k - 1), dim(k)], nnz: sizes[k - 1] * sizes[k] });
    }
    let mut body = format!("(T \"A{n}\")");
    for k in (1..n).rev() {
        body = format!("(Mul (T \"A{k}\") {body})");
    }
    for k in (1..n).rev() {
        body = format!("(Sum \"{}\" {body})", dim(k));
    }
    Problem { name: format!("chain of {n}"), extents, operands, root: body }
}

fn main() {
    let lim = Limits { max_iters: 30, max_tuples: 2_000_000, max_time: Duration::from_secs(60) };

    println!("## Algebraic rewrites (E1–E4)\n");
    println!("| Problem | Input cost | Hybrid cost | Full-AC cost | Hybrid result | Hybrid: tuples / iters / saturated / s | Full-AC: tuples / iters / saturated / s |");
    println!("|---|---|---|---|---|---|---|");
    let mut det_inputs = Vec::new();
    for p in problems() {
        let model = Model { extents: p.extents.clone(), operands: p.operands.clone() };
        let planner = Planner { m: &model };
        let input = parse_root(&p);
        let h = run_egglog(&p, false, 1, &lim);
        let hplan = planner.plan(&h.extracted);
        let a = run_egglog(&p, true, 1, &lim);
        println!(
            "| {} | {:.3e} | {:.3e} | {:.3e} | `{}` | {} / {} / {} / {:.3} | {} / {} / {} / {:.3} |",
            p.name,
            model.cost(&input).cost,
            model.cost(&hplan).cost,
            model.cost(&a.extracted).cost,
            show(&hplan),
            h.tuples, h.iters, h.saturated, h.secs,
            a.tuples, a.iters, a.saturated, a.secs,
        );
        det_inputs.push(p);
    }

    println!("\n## Scaling: matrix chains (E5)\n");
    println!("| n | Hybrid: tuples / iters / saturated / s | Planner s | Hybrid cost | Full-AC: tuples / iters / saturated / s | Full-AC cost |");
    println!("|---|---|---|---|---|---|");
    for n in 3..=12 {
        let p = chain(n);
        let model = Model { extents: p.extents.clone(), operands: p.operands.clone() };
        let planner = Planner { m: &model };
        let h = run_egglog(&p, false, 1, &lim);
        let t0 = Instant::now();
        let hplan = planner.plan(&h.extracted);
        let plan_secs = t0.elapsed().as_secs_f64();
        let a = run_egglog(&p, true, 1, &lim);
        println!(
            "| {n} | {} / {} / {} / {:.3} | {:.4} | {:.3e} | {} / {} / {} / {:.3} | {:.3e} |",
            h.tuples, h.iters, h.saturated, h.secs, plan_secs, model.cost(&hplan).cost,
            a.tuples, a.iters, a.saturated, a.secs, model.cost(&a.extracted).cost,
        );
    }

    println!("\n## Determinism (E6)\n");
    println!("| Problem | Variant | Distinct extracted terms | Distinct e-graph sizes |");
    println!("|---|---|---|---|");
    for p in &det_inputs {
        for assoc in [false, true] {
            let mut terms = BTreeSet::new();
            let mut sizes = BTreeSet::new();
            for threads in [1, 1, 1, 4, 4, 4] {
                let r = run_egglog(p, assoc, threads, &lim);
                terms.insert(show(&r.extracted));
                sizes.insert(r.tuples);
            }
            let variant = if assoc { "full AC" } else { "hybrid" };
            println!("| {} | {variant}, 3 runs × threads 1 and 4 | {} | {} |", p.name, terms.len(), sizes.len());
        }
    }
}
