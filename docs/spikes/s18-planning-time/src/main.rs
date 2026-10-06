// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Spike S18: how long does egglog take to plan realistic sum-product regions?
//!
//! Uses S17's n-ary rules (`../s17-nary/nary.egg`) and its cost model with a
//! greedy planner inside extraction (copied here, with leaves sized by their
//! row counts). Each case is planned cold, as einfold would per query: a new
//! e-graph, the rules parsed, the facts and terms added, saturation, and
//! extraction of every region. Median of 25 runs.

use egglog::extract::{Cost, CostModel, Extractor};
use egglog::prelude::exprs;
use egglog::sort::S;
use egglog::{ArcSort, EGraph, Enode, Function, TermDag, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

const NARY: &str = include_str!("../../s17-nary/nary.egg");

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
    leaves: BTreeMap<String, (BTreeSet<String>, f64)>,
}

impl Model {
    fn dense(&self, dims: &BTreeSet<String>) -> f64 {
        dims.iter().map(|d| self.extents[d]).product()
    }
    fn leaf(&self, name: &str) -> Est {
        let (dims, rows) = self.leaves[name].clone();
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


/// A region: operands (name, dimensions, rows) and output dimensions.
struct Region {
    operands: Vec<(&'static str, Vec<&'static str>, f64)>,
    output: Vec<&'static str>,
}

struct Case {
    name: &'static str,
    extents: Vec<(&'static str, f64)>,
    regions: Vec<Region>,
}

fn r(operands: Vec<(&'static str, Vec<&'static str>, f64)>, output: Vec<&'static str>) -> Region {
    Region { operands, output }
}

fn cases() -> Vec<Case> {
    // TPC-H at scale factor 1. Each table is an operand over its key columns;
    // a query's join-and-SUM core is one sum-product region. Grouping columns
    // that are functions of a key (o_orderdate of o_orderkey) are represented
    // by that key.
    let tpch_ext = vec![
        ("custkey", 150e3), ("orderkey", 6e6), ("linenumber", 7.0), ("suppkey", 10e3),
        ("partkey", 200e3), ("nationkey", 25.0), ("nation2", 25.0), ("regionkey", 5.0), ("flag", 6.0),
    ];
    let (c, o, l, s, p, ps, n, rg) = (150e3, 1.5e6, 6e6, 10e3, 200e3, 800e3, 25.0, 5.0);
    let tpch = |name, regions| Case { name, extents: tpch_ext.clone(), regions };
    let mlp_ext = vec![("n", 5e4), ("d", 16.0), ("h", 64.0), ("o", 8.0)];
    let attn_ext = vec![("t", 1024.0), ("s", 1024.0), ("e", 64.0)];
    vec![
        tpch("TPC-H Q1 (lineitem only)", vec![r(vec![("L", vec!["orderkey", "linenumber", "flag"], l)], vec!["flag"])]),
        tpch("TPC-H Q3 (3 tables)", vec![r(vec![
            ("C", vec!["custkey"], c), ("O", vec!["orderkey", "custkey"], o), ("L", vec!["orderkey", "linenumber"], l),
        ], vec!["orderkey"])]),
        tpch("TPC-H Q5 (6 tables)", vec![r(vec![
            ("C", vec!["custkey", "nationkey"], c), ("O", vec!["orderkey", "custkey"], o),
            ("L", vec!["orderkey", "linenumber", "suppkey"], l), ("S", vec!["suppkey", "nationkey"], s),
            ("N", vec!["nationkey", "regionkey"], n), ("R", vec!["regionkey"], rg),
        ], vec!["nationkey"])]),
        tpch("TPC-H Q7 (6 tables)", vec![r(vec![
            ("S", vec!["suppkey", "nationkey"], s), ("L", vec!["orderkey", "linenumber", "suppkey"], l),
            ("O", vec!["orderkey", "custkey"], o), ("C", vec!["custkey", "nation2"], c),
            ("N1", vec!["nationkey"], n), ("N2", vec!["nation2"], n),
        ], vec!["nationkey", "nation2"])]),
        tpch("TPC-H Q8 (8 tables)", vec![r(vec![
            ("P", vec!["partkey"], p), ("S", vec!["suppkey", "nation2"], s),
            ("L", vec!["orderkey", "linenumber", "partkey", "suppkey"], l), ("O", vec!["orderkey", "custkey"], o),
            ("C", vec!["custkey", "nationkey"], c), ("N1", vec!["nationkey", "regionkey"], n),
            ("N2", vec!["nation2"], n), ("R", vec!["regionkey"], rg),
        ], vec!["nation2"])]),
        tpch("TPC-H Q9 (6 tables)", vec![r(vec![
            ("P", vec!["partkey"], p), ("S", vec!["suppkey", "nationkey"], s),
            ("L", vec!["orderkey", "linenumber", "partkey", "suppkey"], l), ("PS", vec!["partkey", "suppkey"], ps),
            ("O", vec!["orderkey"], o), ("N", vec!["nationkey"], n),
        ], vec!["nationkey"])]),
        tpch("TPC-H Q10 (4 tables)", vec![r(vec![
            ("C", vec!["custkey", "nationkey"], c), ("O", vec!["orderkey", "custkey"], o),
            ("L", vec!["orderkey", "linenumber"], l), ("N", vec!["nationkey"], n),
        ], vec!["custkey"])]),
        Case {
            name: "ddx: 2-layer MLP, forward and backward (5 regions)",
            extents: mlp_ext.clone(),
            regions: vec![
                r(vec![("X", vec!["n", "d"], 8e5), ("W1", vec!["d", "h"], 1024.0)], vec!["n", "h"]),
                r(vec![("A", vec!["n", "h"], 3.2e6), ("W2", vec!["h", "o"], 512.0)], vec!["n", "o"]),
                r(vec![("A", vec!["n", "h"], 3.2e6), ("Ybar", vec!["n", "o"], 4e5)], vec!["h", "o"]),
                r(vec![("Ybar", vec!["n", "o"], 4e5), ("W2", vec!["h", "o"], 512.0)], vec!["n", "h"]),
                r(vec![("X", vec!["n", "d"], 8e5), ("Hbar", vec!["n", "h"], 3.2e6)], vec!["d", "h"]),
            ],
        },
        Case {
            name: "ddx: attention, forward and backward (6 regions)",
            extents: attn_ext,
            regions: vec![
                r(vec![("Q", vec!["t", "e"], 65536.0), ("K", vec!["s", "e"], 65536.0)], vec!["t", "s"]),
                r(vec![("P", vec!["t", "s"], 1048576.0), ("V", vec!["s", "e"], 65536.0)], vec!["t", "e"]),
                r(vec![("P", vec!["t", "s"], 1048576.0), ("dO", vec!["t", "e"], 65536.0)], vec!["s", "e"]),
                r(vec![("dO", vec!["t", "e"], 65536.0), ("V", vec!["s", "e"], 65536.0)], vec!["t", "s"]),
                r(vec![("dS", vec!["t", "s"], 1048576.0), ("K", vec!["s", "e"], 65536.0)], vec!["t", "e"]),
                r(vec![("dS", vec!["t", "s"], 1048576.0), ("Q", vec!["t", "e"], 65536.0)], vec!["s", "e"]),
            ],
        },
        Case {
            name: "ddx: MLP gradient written as one 3-operand region (S16 E4)",
            extents: mlp_ext,
            regions: vec![r(vec![
                ("X", vec!["n", "d"], 8e5), ("Ybar", vec!["n", "o"], 4e5), ("W2", vec!["h", "o"], 512.0),
            ], vec!["d", "h"])],
        },
    ]
}

struct Timing {
    total_ms: f64,
    rules_ms: f64,
    tuples: usize,
}

/// Plan one case. Cold: a fresh e-graph that parses the rules. Warm: a clone
/// of an e-graph that already holds them (EGraph is Clone in egglog 3.0), as a
/// long-lived einfold session would keep.
fn plan_once(case: &Case, warm: Option<&EGraph>) -> Timing {
    let t0 = Instant::now();
    let mut eg = match warm {
        Some(base) => base.clone(),
        None => {
            let mut eg = EGraph::default();
            eg.parse_and_run_program(None, NARY).expect("rules");
            eg
        }
    };
    let rules_ms = t0.elapsed().as_secs_f64() * 1e3;
    let mut prog = String::new();
    let mut leaves = BTreeMap::new();
    for reg in &case.regions {
        for (name, dims, rows) in &reg.operands {
            let ds: BTreeSet<String> = dims.iter().map(|d| d.to_string()).collect();
            if leaves.insert(name.to_string(), (ds.clone(), *rows)).is_none() {
                let q: Vec<String> = ds.iter().map(|d| format!("\"{d}\"")).collect();
                prog.push_str(&format!("(set (dims (T \"{name}\")) (set-of {}))\n", q.join(" ")));
            }
        }
    }
    for (k, reg) in case.regions.iter().enumerate() {
        let all: BTreeSet<&str> = reg.operands.iter().flat_map(|(_, d, _)| d.iter().cloned()).collect();
        let summed: Vec<String> = all.iter().filter(|d| !reg.output.contains(d)).map(|d| format!("\"{d}\"")).collect();
        let ops: Vec<String> = reg.operands.iter().map(|(n, _, _)| format!("(T \"{n}\")")).collect();
        prog.push_str(&format!("(let $r{k} (SP (multiset-of {}) (set-of {})))\n", ops.join(" "), summed.join(" ")));
    }
    prog.push_str("(run-schedule (saturate (saturate (run analysis)) (run algebra)))\n");
    eg.parse_and_run_program(None, &prog).expect("program");
    let tuples = eg.num_tuples();
    let model = Model { extents: case.extents.iter().map(|(k, v)| (k.to_string(), *v)).collect(), leaves };
    let mut values = Vec::new();
    for k in 0..case.regions.len() {
        values.push(eg.eval_expr(&exprs::var(&format!("$r{k}"))).expect("root"));
    }
    // Every region is an Expr, and extract_best wants exactly one root sort.
    let ex = Extractor::compute_costs_from_rootsorts(Some(vec![values[0].0.clone()]), &eg, model);
    let mut dag = TermDag::default();
    for (_, v) in &values {
        ex.extract_best(&eg, &mut dag, *v).expect("extract");
    }
    Timing { total_ms: t0.elapsed().as_secs_f64() * 1e3, rules_ms, tuples }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn main() {
    let mut base = EGraph::default();
    base.parse_and_run_program(None, NARY).expect("rules");
    println!("| Case | Regions | Operands | E-graph tuples | Cold: rules loaded (ms) | Cold: total (ms) | Warm: clone (ms) | Warm: total (ms) |");
    println!("|---|---|---|---|---|---|---|---|");
    for case in cases() {
        plan_once(&case, None); // warm up the allocator and caches
        let cold: Vec<Timing> = (0..25).map(|_| plan_once(&case, None)).collect();
        let warm: Vec<Timing> = (0..25).map(|_| plan_once(&case, Some(&base))).collect();
        let ops: usize = case.regions.iter().map(|r| r.operands.len()).sum();
        let med = |ts: &[Timing], f: fn(&Timing) -> f64| median(ts.iter().map(f).collect());
        println!(
            "| {} | {} | {ops} | {} | {:.2} | {:.2} | {:.2} | {:.2} |",
            case.name,
            case.regions.len(),
            cold[0].tuples,
            med(&cold, |t| t.rules_ms),
            med(&cold, |t| t.total_ms),
            med(&warm, |t| t.rules_ms),
            med(&warm, |t| t.total_ms),
        );
    }
}
