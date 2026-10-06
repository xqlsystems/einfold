// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Spike S17: n-ary sum-product nodes in egglog, on Einsum Benchmark instances.
//!
//! For each instance (from `make_instances.py`):
//! 1. Binary form: S16's rules over nested `Mul` and `Sum`, without associativity.
//! 2. N-ary form: `nary.egg`, one `SP` node over a multiset of operands.
//! 3. Extraction of the n-ary form, with a greedy contraction planner running
//!    inside egglog's cost model, so extraction sees each candidate's planned cost.
//! 4. Growth under distributivity: the same instance with k operands replaced by sums.
//!
//! Usage: s17-nary <instances.txt> [max binary operands]

use egglog::extract::{Cost, CostModel, Extractor};
use egglog::prelude::exprs;
use egglog::sort::S;
use egglog::{ArcSort, EGraph, Enode, Function, TermDag, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::time::{Duration, Instant};

const NARY: &str = include_str!("../nary.egg");
const BINARY: &str = include_str!("../../s16-egglog/rules.egg");

// ---------------------------------------------------------------------------
// Instances

struct Instance {
    name: String,
    operands: Vec<Vec<String>>,
    output: BTreeSet<String>,
    extents: BTreeMap<String, f64>,
}

fn parse(line: &str) -> Instance {
    let f: Vec<&str> = line.split('|').collect();
    Instance {
        name: f[0].to_string(),
        operands: f[1].split(',').map(|o| o.split_whitespace().map(String::from).collect()).collect(),
        output: f[2].split_whitespace().map(String::from).collect(),
        extents: f[3]
            .split_whitespace()
            .map(|kv| {
                let (k, v) = kv.split_once('=').unwrap();
                (k.to_string(), v.parse().unwrap())
            })
            .collect(),
    }
}

impl Instance {
    fn summed(&self) -> BTreeSet<String> {
        self.operands.iter().flatten().filter(|d| !self.output.contains(*d)).cloned().collect()
    }
    fn declare_dims(&self, prog: &mut String) {
        for (k, o) in self.operands.iter().enumerate() {
            let ds: BTreeSet<&String> = o.iter().collect();
            let ds: Vec<String> = ds.iter().map(|d| format!("\"{d}\"")).collect();
            prog.push_str(&format!("(set (dims (T \"t{k}\")) (set-of {}))\n", ds.join(" ")));
            // Copies used as the second term of a sum, for the distributivity test.
            prog.push_str(&format!("(set (dims (T \"u{k}\")) (set-of {}))\n", ds.join(" ")));
        }
    }
    /// Operand k, or (t_k + u_k) for the first `adds` operands.
    fn leaf(&self, k: usize, adds: usize) -> String {
        if k < adds {
            format!("(Add (T \"t{k}\") (T \"u{k}\"))")
        } else {
            format!("(T \"t{k}\")")
        }
    }
    fn binary_term(&self) -> String {
        let n = self.operands.len();
        let mut body = self.leaf(n - 1, 0);
        for k in (0..n - 1).rev() {
            body = format!("(Mul {} {body})", self.leaf(k, 0));
        }
        for d in self.summed() {
            body = format!("(Sum \"{d}\" {body})");
        }
        body
    }
    fn nary_term(&self, adds: usize) -> String {
        let ops: Vec<String> = (0..self.operands.len()).map(|k| self.leaf(k, adds)).collect();
        let sums: Vec<String> = self.summed().iter().map(|d| format!("\"{d}\"")).collect();
        format!("(SP (multiset-of {}) (set-of {}))", ops.join(" "), sums.join(" "))
    }
}

// ---------------------------------------------------------------------------
// Saturation

struct Limits {
    max_iters: usize,
    max_tuples: usize,
    max_time: Duration,
}

struct Run {
    eg: EGraph,
    tuples: usize,
    iters: usize,
    saturated: bool,
    secs: f64,
}

fn saturate(rules: &str, inst: &Instance, root: &str, threads: usize, lim: &Limits) -> Run {
    let mut eg = EGraph::default();
    eg.set_num_threads(threads);
    let mut prog = String::from(rules);
    inst.declare_dims(&mut prog);
    prog.push_str(&format!("(let $root {root})\n(run-schedule (saturate (run analysis)))\n"));
    let start = Instant::now();
    eg.parse_and_run_program(None, &prog).expect("program");
    let step = "(run-schedule (saturate (run analysis)) (run algebra))";
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
    Run { tuples: eg.num_tuples(), eg, iters, saturated, secs }
}

// ---------------------------------------------------------------------------
// Cost model with a greedy contraction planner inside it

#[derive(Clone, Debug, Default)]
struct Est {
    cost: f64,
    rows: f64,
    dims: BTreeSet<String>,
    tag: Option<String>,
    parts: Vec<Est>,
    tags: Vec<String>,
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
    fn combine(self, other: &Self) -> Self {
        Est { cost: self.cost + other.cost, ..self }
    }
}

struct Model {
    extents: BTreeMap<String, f64>,
    leaf_dims: BTreeMap<String, BTreeSet<String>>,
}

impl Model {
    fn dense(&self, dims: &BTreeSet<String>) -> f64 {
        dims.iter().map(|d| self.extents[d]).product()
    }
    fn leaf(&self, name: &str) -> Est {
        let dims = self.leaf_dims[name].clone();
        let rows = self.dense(&dims);
        Est { cost: rows, rows, dims, ..Est::default() }
    }
    /// Plan Σ_sums Π parts greedily: repeatedly join the pair of intermediates
    /// that share a dimension and whose result, after summing out dimensions
    /// nothing else needs, is smallest relative to its inputs (opt_einsum's
    /// greedy rule). Cost is rows produced by each join plus rows read by each
    /// aggregation, as in spike S16.
    fn plan(&self, parts: &[Est], sums: &BTreeSet<String>) -> Est {
        let mut live: Vec<Est> = parts.to_vec();
        let mut cost: f64 = parts.iter().map(|p| p.cost).sum();
        let all: BTreeSet<String> = parts.iter().flat_map(|p| p.dims.iter().cloned()).collect();
        let out: BTreeSet<String> = all.difference(sums).cloned().collect();
        // Sum out dimensions held by one operand only.
        let reduce = |e: &mut Est, others: &[&Est], cost: &mut f64| {
            let needed: BTreeSet<String> = others.iter().flat_map(|o| o.dims.iter().cloned()).collect();
            let drop: Vec<String> =
                e.dims.iter().filter(|d| !out.contains(*d) && !needed.contains(*d)).cloned().collect();
            if !drop.is_empty() {
                *cost += e.rows;
                for d in drop {
                    e.dims.remove(&d);
                }
                e.rows = e.rows.min(self.dense(&e.dims));
            }
        };
        for k in 0..live.len() {
            let others: Vec<&Est> = parts.iter().enumerate().filter(|(j, _)| *j != k).map(|(_, p)| p).collect();
            let mut e = live[k].clone();
            reduce(&mut e, &others, &mut cost);
            live[k] = e;
        }
        // How many live intermediates hold each dimension.
        let mut count: BTreeMap<String, usize> = BTreeMap::new();
        for e in &live {
            for d in &e.dims {
                *count.entry(d.clone()).or_default() += 1;
            }
        }
        while live.len() > 1 {
            let mut by_dim: BTreeMap<&String, Vec<usize>> = BTreeMap::new();
            for (k, e) in live.iter().enumerate() {
                for d in &e.dims {
                    by_dim.entry(d).or_default().push(k);
                }
            }
            let mut pairs: BTreeSet<(usize, usize)> = BTreeSet::new();
            for ks in by_dim.values() {
                for a in 0..ks.len() {
                    for b in a + 1..ks.len() {
                        pairs.insert((ks[a], ks[b]));
                    }
                }
            }
            if pairs.is_empty() {
                pairs.insert((0, 1)); // disconnected: an outer product
            }
            let mut best: Option<(f64, usize, usize, Est, f64)> = None;
            for &(a, b) in &pairs {
                let (ea, eb) = (&live[a], &live[b]);
                let shared: f64 = ea.dims.intersection(&eb.dims).map(|d| self.extents[d]).product();
                let dims: BTreeSet<String> = ea.dims.union(&eb.dims).cloned().collect();
                let rows = (ea.rows * eb.rows / shared).min(self.dense(&dims));
                // Keep a dimension if it is an output, or another intermediate still has it.
                let kept: BTreeSet<String> = dims
                    .iter()
                    .filter(|d| {
                        let here = ea.dims.contains(*d) as usize + eb.dims.contains(*d) as usize;
                        out.contains(*d) || count[*d] > here
                    })
                    .cloned()
                    .collect();
                let step = if kept.len() < dims.len() { 2.0 * rows } else { rows };
                let out_rows = rows.min(self.dense(&kept));
                let score = out_rows - ea.rows - eb.rows;
                if best.as_ref().map_or(true, |(s, ..)| score < *s) {
                    best = Some((score, a, b, Est { rows: out_rows, dims: kept, ..Est::default() }, step));
                }
            }
            let (_, a, b, e, step) = best.unwrap();
            cost += step;
            for d in live[a].dims.iter().chain(live[b].dims.iter()) {
                *count.get_mut(d).unwrap() -= 1;
            }
            for d in &e.dims {
                *count.get_mut(d).unwrap() += 1;
            }
            live.remove(b);
            live.remove(a);
            live.push(e);
        }
        let mut e = live.pop().unwrap();
        e.cost = cost;
        e
    }
}

impl CostModel<Est> for Model {
    fn fold(&self, head: &str, ch: &[Est], _head_cost: Est) -> Est {
        match head {
            "T" => self.leaf(ch[0].tag.as_deref().expect("name")),
            "SP" => {
                let sums: BTreeSet<String> = ch[1].tags.iter().cloned().collect();
                self.plan(&ch[0].parts, &sums)
            }
            "Add" => {
                let dims: BTreeSet<String> = ch[0].dims.union(&ch[1].dims).cloned().collect();
                let rows = (ch[0].rows + ch[1].rows).min(self.dense(&dims));
                Est { cost: ch[0].cost + ch[1].cost + ch[0].rows + ch[1].rows, rows, dims, ..Est::default() }
            }
            "Scale" => Est { cost: ch[1].cost + ch[1].rows, ..ch[1].clone() },
            _ => Est::default(),
        }
    }
    fn enode_cost(&self, _: &EGraph, _: &Function, _: &Enode<'_>) -> Est {
        Est::default()
    }
    fn container_cost(&self, _: &EGraph, _: &ArcSort, _: Value, elements: &[Est]) -> Est {
        // A multiset of operands keeps each element's estimate; a set of
        // dimension names keeps the names.
        Est {
            tags: elements.iter().filter_map(|e| e.tag.clone()).collect(),
            parts: elements.iter().filter(|e| e.tag.is_none()).cloned().collect(),
            ..Est::default()
        }
    }
    fn base_value_cost(&self, egraph: &EGraph, sort: &ArcSort, value: Value) -> Est {
        if sort.name() == "String" {
            Est { tag: Some(egraph.value_to_base::<S>(value).0.clone()), ..Est::default() }
        } else {
            Est::default()
        }
    }
}

fn model(inst: &Instance) -> Model {
    let mut leaf_dims = BTreeMap::new();
    for (k, o) in inst.operands.iter().enumerate() {
        let ds: BTreeSet<String> = o.iter().cloned().collect();
        leaf_dims.insert(format!("t{k}"), ds.clone());
        leaf_dims.insert(format!("u{k}"), ds);
    }
    Model { extents: inst.extents.clone(), leaf_dims }
}

/// Extract the root of an n-ary run. Returns (cost, seconds, root head, Add count).
fn extract(run: &mut Run, inst: &Instance) -> (f64, f64, String, usize) {
    let t0 = Instant::now();
    let (sort, value) = run.eg.eval_expr(&exprs::var("$root")).expect("root");
    let ex = Extractor::compute_costs_from_rootsorts(Some(vec![sort]), &run.eg, model(inst));
    let mut dag = TermDag::default();
    let (cost, term) = ex.extract_best(&run.eg, &mut dag, value).expect("extract");
    let text = dag.to_string(term);
    let head = text.trim_start_matches('(').split_whitespace().next().unwrap_or("").to_string();
    (cost.cost, t0.elapsed().as_secs_f64(), head, text.matches("(Add ").count())
}

fn main() {
    let path = std::env::args().nth(1).expect("instances file");
    let max_binary: usize = std::env::args().nth(2).map_or(40, |s| s.parse().unwrap());
    let insts: Vec<Instance> = std::fs::read_to_string(path).unwrap().lines().map(parse).collect();
    let lim = Limits { max_iters: 30, max_tuples: 2_000_000, max_time: Duration::from_secs(60) };

    println!("## Pure sum-product instances\n");
    println!("| Instance | Operands | Binary: tuples / iters / saturated / s | N-ary: tuples / iters / saturated / s | N-ary extraction s | Planned cost (log10) |");
    println!("|---|---|---|---|---|---|");
    for inst in &insts {
        let n = inst.operands.len();
        let binary = if n <= max_binary {
            let b = saturate(BINARY, inst, &inst.binary_term(), 1, &lim);
            format!("{} / {} / {} / {:.3}", b.tuples, b.iters, if b.saturated { "yes" } else { "**no**" }, b.secs)
        } else {
            "not run".into()
        };
        let mut r = saturate(NARY, inst, &inst.nary_term(0), 1, &lim);
        let (cost, xs, _, _) = extract(&mut r, inst);
        println!(
            "| {} | {n} | {binary} | {} / {} / {} / {:.3} | {:.3} | {:.2} |",
            inst.name, r.tuples, r.iters, if r.saturated { "yes" } else { "**no**" }, r.secs, xs, cost.log10()
        );
    }

    println!("\n## Distributivity: k operands replaced by sums (t_k + u_k)\n");
    println!("| Instance | k | N-ary: tuples / iters / saturated / s | Extraction s | Extracted root | Adds in extracted term |");
    println!("|---|---|---|---|---|---|");
    for inst in insts.iter().filter(|i| i.name == "matrix_chain n=50" || i.name == "random 3-regular n=100") {
        for k in [1, 2, 4, 6, 8] {
            let mut r = saturate(NARY, inst, &inst.nary_term(k), 1, &lim);
            let (_, xs, head, adds) = extract(&mut r, inst);
            println!(
                "| {} | {k} | {} / {} / {} / {:.3} | {:.3} | `{head}` | {adds} |",
                inst.name, r.tuples, r.iters, if r.saturated { "yes" } else { "**no**" }, r.secs, xs
            );
        }
    }

    println!("\n## Determinism: n-ary, with sums, threads 1 and 4, three runs each\n");
    println!("| Instance | Distinct e-graph sizes | Distinct extracted costs |");
    println!("|---|---|---|");
    for inst in insts.iter().filter(|i| i.name == "random 3-regular n=100" || i.name == "lattice 8x8") {
        let (mut sizes, mut costs) = (BTreeSet::new(), BTreeSet::new());
        for threads in [1, 1, 1, 4, 4, 4] {
            let mut r = saturate(NARY, inst, &inst.nary_term(4), threads, &lim);
            sizes.insert(r.tuples);
            costs.insert(extract(&mut r, inst).0.to_bits());
        }
        println!("| {} | {} | {} |", inst.name, sizes.len(), costs.len());
    }
}
