// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Properties of the kernels at the dense algorithm's scale: shapes on its
//! block edges, against an independent oracle and against themselves.
//!
//! The dense algorithm flattens each role's dimensions into one position:
//! `nx` positions of the dimensions only `a` holds, `ny` of those only `b`
//! holds, and `nz` of those both hold and the fold sums. It changes how it
//! splits work at fixed sizes: once the output has 4,096 cells (`nx·ny`), it
//! works in blocks of 256 output rows and 2,048 output columns; below that, in
//! blocks of 4,096 positions of `z`, added in block order. Its bugs would live
//! at those edges, so each case draws `nx`, `ny` and `nz` themselves from
//! values at and around them (255–257, 2047–2049, 4095–4097, …), sometimes
//! puts `nx·ny` on 4,095–4,097 on purpose, and only then factors each into one
//! or two dimensions. Every table holds the row with each dimension's largest
//! key, so the extents the algorithm sees are the ones drawn.
//!
//! Factors include NULL, NaN, ∞, `-0.0` and ±1e16. Half the cases have a
//! domain, with keys one past each extent, repeated rows and sometimes NULL
//! keys, each matched with `=` or `IS NOT DISTINCT FROM`. A few cases are
//! instead *sparse*: a summed dimension of 10⁶ positions holding a few hundred
//! keys, where the dense algorithm must decline.
//!
//! The invariants:
//!
//! - **`[oracle]`** the result equals a nested-loop evaluation of SQL's
//!   semantics, written here with no code shared with the kernels: the same
//!   groups or domain rows, NULL where SQL gives NULL, NaN where it gives NaN,
//!   and otherwise within `2·(n − 1)·u·Σ|terms|` of the oracle's sum. A NULL
//!   domain key matches no group under either equality, since the operands
//!   have no NULL keys;
//! - **`[scale]`** doubling every factor of `a` doubles every result, and
//!   negating them negates it, bit for bit: exact checks that the summation
//!   bound can't give, as it absorbs a dropped small term beside a `1e16`;
//! - **`[relabel]`** permuting the keys of the output dimensions (and the
//!   domain's with them) permutes the result's keys and keeps every value's
//!   bits, since each cell's terms are still added in order of `z`. This
//!   aims at the positions, strides and domain lookup;
//! - **`[key-type]`** keys cast to `Int32`, `Int16` or `UInt32` give the same
//!   bits;
//! - **`[swap]`** swapping the operands gives the same values within the
//!   bound. The dense algorithm treats its operands differently (only `b` may
//!   have holes), so this crosses its conditions;
//! - **`[discriminate]`** one row of `b` that joins nothing (a negative key) moves the
//!   fold to the discrimination algorithm, which must give the oracle's rows;
//! - **`[duplicates]`** a repeated row of `a` or `b` moves the fold to the
//!   discrimination algorithm, which must sum it twice, as the oracle does;
//! - **`[sparse]`** a sparse case runs the discrimination algorithm and gives the
//!   oracle's rows;
//! - **`[domain]`** each domain row gets exactly its group's bits from the same
//!   fold without a domain, or the default;
//! - **`[unique]`** asked to check that the domain's keys are unique, the
//!   dense algorithm declines exactly when two domain rows with non-NULL keys
//!   inside the extents share a cell, and otherwise gives the unchecked bits;
//! - **`[threads]`** the same bits on any number of threads and batching,
//!   empty batches included.
//!
//! The test runs a number of cases (`EINFOLD_CASES`, default 60), drawing
//! seeds from `EINFOLD_SEED` and skipping shapes too large for the oracle,
//! and asserts that each regime and each edge was reached. Reproduce a
//! failure with `EINFOLD_SEED=<seed> EINFOLD_CASES=1`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use einfold_ir::{
    Aggregate, ColumnIndex, Dimension, DimensionId, Domain, DomainKey, Fold, Key, KeyEquality,
    Operand, Output, RowValue,
};
use einfold_kernels::{Algorithm, DomainRows, FloatKeys, Host, Options, Relation};
use einfold_testkit::Rng;

/// A NULL key, in a domain only.
const NULL_KEY: i64 = i64::MIN;

/// The dense algorithm's block sizes, as `dense.rs` defines them.
const XB: usize = 256;
const YB: usize = 2048;
const ZB: usize = 4096;
const FEW_CELLS: usize = 4096;

/// Flattened extents for each role: at and around the edges, or small.
const X_EDGES: [usize; 9] = [1, 2, 3, 64, XB - 1, XB, XB + 1, 2 * XB, 2 * XB + 1];
const Y_EDGES: [usize; 7] = [1, 2, 5, YB - 1, YB, YB + 1, 2 * YB + 1];
const Z_EDGES: [usize; 7] = [1, 2, 3, 16, ZB - 1, ZB, ZB + 1];

/// One relation: rows of integer keys (one per dimension it holds) and a
/// value.
#[derive(Clone)]
struct Rel {
    dims: Vec<usize>,
    keys: Vec<Vec<i64>>,
    values: Vec<Option<f64>>,
}

/// A fold's domain: its rows (keys in output order), default, and how each
/// output's key is matched.
#[derive(Clone)]
struct Dom {
    rel: Rel,
    missing: Option<f64>,
    equality: Vec<KeyEquality>,
}

#[derive(Clone)]
struct Case {
    xs: Vec<usize>,
    zs: Vec<usize>,
    ys: Vec<usize>,
    outputs: Vec<usize>,
    extent: Vec<usize>,
    a: Rel,
    b: Rel,
    domain: Option<Dom>,
    sparse: bool,
}

impl Case {
    fn cells(&self, dims: &[usize]) -> usize {
        dims.iter().map(|&d| self.extent[d]).product()
    }
}

fn value(rng: &mut Rng, specials: bool) -> Option<f64> {
    match rng.below(100) {
        0..=4 => None,
        5..=6 if specials => Some(f64::NAN),
        7 if specials => Some(f64::INFINITY),
        8..=10 => Some(-0.0),
        11..=16 => Some(if rng.chance(0.5) { 1e16 } else { -1e16 }),
        _ => Some(((rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 8.0),
    }
}

/// `t` as the extents of one or two dimensions whose product is `t`; none
/// when `t` is 1 and `empty` allows it.
fn factor(rng: &mut Rng, t: usize, empty: bool) -> Vec<usize> {
    if t == 1 && empty && rng.chance(0.5) {
        return vec![];
    }
    let divisors: Vec<usize> = (2..t).filter(|&d| t.is_multiple_of(d)).collect();
    if !divisors.is_empty() && rng.chance(0.5) {
        let d = *rng.pick(&divisors);
        return vec![d, t / d];
    }
    vec![t]
}

/// Every key tuple over `dims`, each kept with probability `present`,
/// except the last, whose keys are each dimension's largest, which is
/// always kept.
fn rel(rng: &mut Rng, dims: &[usize], extent: &[usize], present: f64, specials: bool) -> Rel {
    let total: usize = dims.iter().map(|&d| extent[d]).product();
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for t0 in 0..total {
        if t0 + 1 < total && !rng.chance(present) {
            continue;
        }
        let mut t = t0;
        let mut k = vec![0i64; dims.len()];
        for (slot, &d) in dims.iter().enumerate().rev() {
            k[slot] = (t % extent[d]) as i64;
            t /= extent[d];
        }
        keys.push(k);
        values.push(value(rng, specials));
    }
    Rel {
        dims: dims.to_vec(),
        keys,
        values,
    }
}

/// The cap on the oracle's work, `nx·nz·ny`, lower in a debug build.
fn work_cap() -> usize {
    if cfg!(debug_assertions) {
        3_000_000
    } else {
        30_000_000
    }
}

fn case(rng: &mut Rng) -> Option<Case> {
    if rng.chance(0.06) {
        return Some(sparse_case(rng));
    }
    // Flattened extents per role, each kind of case aimed at one edge: the
    // regime switch at nx·ny = 4,096; nx at a 256-row edge and ny at a
    // 2,048-column edge, both with the output blocked; nz at a 4,096 edge
    // with z blocked; or drawn freely from the edges.
    let near = |rng: &mut Rng, edge: usize| edge - 1 + rng.below(3);
    let small = |rng: &mut Rng| 1 + rng.below(6);
    let (nx, ny, mut nz) = match rng.below(10) {
        0..=2 => {
            let p = near(rng, FEW_CELLS);
            let divisors: Vec<usize> = (1..=p).filter(|&d| p.is_multiple_of(d)).collect();
            let nx = *rng.pick(&divisors);
            (nx, p / nx, *rng.pick(&Z_EDGES[..4]))
        }
        3 | 4 => (near(rng, XB), 17 + rng.below(24), small(rng)),
        5 | 6 => (2 + rng.below(5), near(rng, YB), small(rng)),
        7 => (small(rng), small(rng), near(rng, ZB)),
        _ => {
            let draw = |rng: &mut Rng, edges: &[usize]| {
                if rng.chance(0.25) {
                    1 + rng.below(6)
                } else {
                    *rng.pick(edges)
                }
            };
            (
                draw(rng, &X_EDGES),
                draw(rng, &Y_EDGES),
                draw(rng, &Z_EDGES),
            )
        }
    };
    let fits = |nz: usize| {
        nx * nz <= 300_000 && nz * ny <= 300_000 && nx * ny <= 600_000 && nx * nz * ny <= work_cap()
    };
    if !fits(nz) {
        nz = 1 + rng.below(3);
    }
    if !fits(nz) {
        return None;
    }
    let (fx, fz, fy) = (
        factor(rng, nx, true),
        factor(rng, nz, false),
        factor(rng, ny, true),
    );
    let mut extent = Vec::new();
    let mut role = |sizes: Vec<usize>| -> Vec<usize> {
        (sizes.into_iter())
            .map(|e| {
                extent.push(e);
                extent.len() - 1
            })
            .collect()
    };
    let (xs, zs, ys) = (role(fx), role(fz), role(fy));
    let mut outputs: Vec<usize> = xs.iter().chain(&ys).copied().collect();
    for i in (1..outputs.len()).rev() {
        outputs.swap(i, rng.below(i + 1));
    }
    let specials = rng.chance(0.2);
    let a_dims: Vec<usize> = xs.iter().chain(&zs).copied().collect();
    let b_dims: Vec<usize> = zs.iter().chain(&ys).copied().collect();
    let pa = if rng.chance(0.5) { 1.0 } else { 0.7 };
    let pb = if rng.chance(0.85) { 1.0 } else { 0.9 };
    let a = rel(rng, &a_dims, &extent, pa, specials);
    let mut b = rel(rng, &b_dims, &extent, pb, specials);
    // A NULL factor in b is a hole to the dense algorithm, like a missing
    // row, so most cases keep b's factors non-NULL.
    if rng.chance(0.7) {
        for v in b.values.iter_mut().filter(|v| v.is_none()) {
            *v = Some(0.25);
        }
    }
    let domain = rng.chance(0.5).then(|| domain(rng, &outputs, &extent));
    Some(Case {
        xs,
        zs,
        ys,
        outputs,
        extent,
        a,
        b,
        domain,
        sparse: false,
    })
}

/// A domain over `outputs`: keys one past each extent, some rows repeated
/// (sometimes none), sometimes NULL keys, a random default and equalities.
fn domain(rng: &mut Rng, outputs: &[usize], extent: &[usize]) -> Dom {
    let ext: Vec<usize> = extent.iter().map(|e| e + 1).collect();
    let mut d = rel(rng, outputs, &ext, 0.6, false);
    if rng.chance(0.6) {
        let extra: Vec<usize> = (0..d.keys.len() / 10 + 1)
            .map(|_| rng.below(d.keys.len()))
            .collect();
        for i in extra {
            d.keys.push(d.keys[i].clone());
            d.values.push(None);
        }
    }
    if rng.chance(0.3) {
        for k in d.keys.iter_mut().flatten() {
            if rng.chance(0.1) {
                *k = NULL_KEY;
            }
        }
    }
    let equality = (0..outputs.len())
        .map(|_| match rng.chance(0.5) {
            true => KeyEquality::Equal,
            false => KeyEquality::NotDistinct,
        })
        .collect();
    Dom {
        rel: d,
        missing: [None, Some(0.0), Some(2.5)][rng.below(3)],
        equality,
    }
}

/// One summed dimension of 10⁶ positions holding a few hundred keys, with
/// small outputs: too sparse for the dense algorithm.
fn sparse_case(rng: &mut Rng) -> Case {
    let extent = vec![1 + rng.below(5), 1_000_000, 1 + rng.below(5)];
    let (xs, zs, ys) = (vec![0], vec![1], vec![2]);
    let z_keys: Vec<i64> = (0..300).map(|_| rng.below(1_000_000) as i64).collect();
    let rel_over = |rng: &mut Rng, dims: Vec<usize>, other: usize| {
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for &z in &z_keys {
            if rng.chance(0.8) {
                let o = rng.below(extent[other]) as i64;
                keys.push(if dims[0] == 1 { vec![z, o] } else { vec![o, z] });
                values.push(value(rng, false));
            }
        }
        Rel { dims, keys, values }
    };
    let a = rel_over(rng, vec![0, 1], 0);
    let b = rel_over(rng, vec![1, 2], 2);
    let outputs = vec![0, 2];
    let domain = rng.chance(0.5).then(|| domain(rng, &outputs, &extent));
    Case {
        xs,
        zs,
        ys,
        outputs,
        extent,
        a,
        b,
        domain,
        sparse: true,
    }
}

/// A relation's keys as columns of `ty` (NULL for [`NULL_KEY`]), then its
/// values.
fn batch(rel: &Rel, ty: &DataType) -> (SchemaRef, RecordBatch) {
    let mut fields: Vec<Field> = (0..rel.dims.len())
        .map(|i| Field::new(format!("k{i}"), ty.clone(), true))
        .collect();
    fields.push(Field::new("v", DataType::Float64, true));
    let mut columns: Vec<ArrayRef> = (0..rel.dims.len())
        .map(|i| {
            let keys = rel.keys.iter().map(|k| (k[i] != NULL_KEY).then_some(k[i]));
            let col: ArrayRef = Arc::new(keys.collect::<Int64Array>());
            arrow::compute::cast(&col, ty).expect("keys fit")
        })
        .collect();
    columns.push(Arc::new(Float64Array::from(rel.values.clone())));
    let schema = Arc::new(Schema::new(fields));
    let b = RecordBatch::try_new(Arc::clone(&schema), columns).expect("a batch");
    (schema, b)
}

fn fold_of(case: &Case, swapped: bool, with_domain: bool) -> Fold {
    let n = case.extent.len();
    let op = |rel: &Rel| Operand {
        keys: (rel.dims.iter().enumerate())
            .map(|(c, &d)| Key::new(DimensionId(d), ColumnIndex(c)))
            .collect(),
        factor: Some(ColumnIndex(rel.dims.len())),
    };
    let operands = match swapped {
        false => vec![op(&case.a), op(&case.b)],
        true => vec![op(&case.b), op(&case.a)],
    };
    let domain = case.domain.as_ref().filter(|_| with_domain);
    let outputs = (case.outputs.iter().enumerate())
        .map(|(i, &d)| Output {
            dimension: DimensionId(d),
            domain_key: domain.map(|dom| DomainKey::new(ColumnIndex(i), dom.equality[i])),
        })
        .collect();
    Fold::try_new(
        vec![
            Dimension {
                equality: KeyEquality::Equal
            };
            n
        ],
        operands,
        outputs,
        RowValue::Product,
        Aggregate::sum(),
        domain.map(|d| Domain {
            carry: (0..d.rel.dims.len()).map(ColumnIndex).collect(),
            missing: d.missing,
        }),
    )
    .expect("a valid fold")
}

/// A batch's rows shuffled and split into random batches, empty ones
/// included.
fn split(rng: &mut Rng, b: &RecordBatch) -> Vec<RecordBatch> {
    let n = b.num_rows();
    let mut order: Vec<u32> = (0..n as u32).collect();
    for i in (1..n).rev() {
        order.swap(i, rng.below(i + 1));
    }
    let take = arrow::compute::take_record_batch(b, &UInt32Array::from(order)).expect("take");
    let mut out = Vec::new();
    let mut at = 0;
    while at < n {
        let len = rng.below(n / 3 + 2).min(n - at);
        out.push(take.slice(at, len));
        at += len;
    }
    out
}

/// How a run computes the fold: by the best algorithm, or by the dense one
/// alone, with or without checking the domain's keys.
#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Best,
    Dense(DomainRows),
}

struct Run {
    algorithm: Algorithm,
    batches: Vec<RecordBatch>,
}

/// The case's fold, on a random number of threads and batching; `None` if
/// `mode` asks for the dense algorithm and it declined.
fn run(
    case: &Case,
    rng: &mut Rng,
    ty: &DataType,
    swapped: bool,
    with_domain: bool,
    mode: Mode,
) -> Option<Run> {
    let threads = 1 + rng.below(8);
    let fold = fold_of(case, swapped, with_domain);
    let (first, second) = if swapped {
        (&case.b, &case.a)
    } else {
        (&case.a, &case.b)
    };
    let (sa, ba) = batch(first, ty);
    let (sb, bb) = batch(second, ty);
    let dom = case
        .domain
        .as_ref()
        .filter(|_| with_domain)
        .map(|d| batch(&d.rel, ty));
    let (ba, bb) = (split(rng, &ba), split(rng, &bb));
    let bd = dom.as_ref().map(|(_, b)| split(rng, b));
    let schema = einfold_kernels::schema(
        &fold,
        &[sa.as_ref(), sb.as_ref()],
        dom.as_ref().map(|(s, _)| s.as_ref()),
    )
    .expect("a schema");
    let domain = dom
        .as_ref()
        .zip(bd.as_deref())
        .map(|((s, _), b)| Relation::new(s, b));
    let host = Host {
        float_keys: FloatKeys::SignedZerosEqual,
    };
    let (a, b) = (Relation::new(&sa, &ba), Relation::new(&sb, &bb));
    match mode {
        Mode::Best => {
            let options = Options {
                host,
                threads,
                ..Options::default()
            };
            let f = einfold_kernels::fold(&fold, a, b, domain, schema, options).expect("runs");
            Some(Run {
                algorithm: f.algorithm,
                batches: f.batches,
            })
        }
        Mode::Dense(domain_rows) => {
            let options = Options {
                host,
                threads,
                domain_rows,
                budget: None,
            };
            let f =
                einfold_kernels::fold_dense(&fold, a, b, domain, schema, options).expect("runs");
            f.map(|batches| Run {
                algorithm: Algorithm::Dense,
                batches,
            })
        }
    }
}

/// A group's state in the oracle: whether any non-NULL product reached it,
/// its sum, and their count and magnitude.
#[derive(Clone, Copy, Default)]
struct State {
    valued: bool,
    sum: f64,
    terms: f64,
    magnitude: f64,
}

/// SQL's groups, by nested loops over a hash of `b` by its `z` keys, keyed by
/// the outputs' keys in output order.
fn oracle(case: &Case) -> BTreeMap<Vec<i64>, State> {
    let col = |rel: &Rel, d: usize| rel.dims.iter().position(|&e| e == d).expect("held");
    let z_of = |rel: &Rel, r: usize| -> Vec<i64> {
        case.zs.iter().map(|&d| rel.keys[r][col(rel, d)]).collect()
    };
    let mut by_z: HashMap<Vec<i64>, Vec<usize>> = HashMap::new();
    for r in 0..case.b.keys.len() {
        by_z.entry(z_of(&case.b, r)).or_default().push(r);
    }
    let mut groups: BTreeMap<Vec<i64>, State> = BTreeMap::new();
    for ra in 0..case.a.keys.len() {
        let Some(matches) = by_z.get(&z_of(&case.a, ra)) else {
            continue;
        };
        for &rb in matches {
            let key: Vec<i64> = (case.outputs.iter())
                .map(|&d| match case.a.dims.contains(&d) {
                    true => case.a.keys[ra][col(&case.a, d)],
                    false => case.b.keys[rb][col(&case.b, d)],
                })
                .collect();
            let g = groups.entry(key).or_default();
            if let (Some(x), Some(y)) = (case.a.values[ra], case.b.values[rb]) {
                g.valued = true;
                g.sum += x * y;
                g.terms += 1.0;
                g.magnitude += (x * y).abs();
            }
        }
    }
    groups
}

type Rows = Vec<(Vec<i64>, Option<f64>)>;

/// A result's rows: keys as `i64` ([`NULL_KEY`] for NULL), and value, sorted.
fn rows(batches: &[RecordBatch], keys: usize) -> Rows {
    let mut out = Vec::new();
    for b in batches {
        let cols: Vec<Int64Array> = (0..keys)
            .map(|c| {
                let k = arrow::compute::cast(b.column(c), &DataType::Int64).expect("integer keys");
                k.as_any()
                    .downcast_ref::<Int64Array>()
                    .expect("i64")
                    .clone()
            })
            .collect();
        let v = b
            .column(keys)
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("values");
        for r in 0..b.num_rows() {
            let key = cols
                .iter()
                .map(|c| if c.is_valid(r) { c.value(r) } else { NULL_KEY });
            out.push((key.collect(), v.is_valid(r).then(|| v.value(r))));
        }
    }
    sort(&mut out);
    out
}

fn sort(rows: &mut Rows) {
    rows.sort_by(|x, y| (&x.0, x.1.map(f64::to_bits)).cmp(&(&y.0, y.1.map(f64::to_bits))));
}

/// Rows as bits, every NaN alike, for exact comparison.
fn bits(rows: &Rows) -> Vec<(Vec<i64>, Option<u64>)> {
    let canon = |v: f64| {
        if v.is_nan() {
            f64::NAN.to_bits()
        } else {
            v.to_bits()
        }
    };
    rows.iter()
        .map(|(k, v)| (k.clone(), v.map(canon)))
        .collect()
}

fn agree(ours: Option<f64>, theirs: Option<f64>, bound: f64) -> bool {
    match (ours, theirs) {
        (None, None) => true,
        (Some(x), Some(y)) if x.is_nan() || y.is_nan() => x.is_nan() && y.is_nan(),
        (Some(x), Some(y)) if x == y => x.is_sign_negative() == y.is_sign_negative(),
        (Some(x), Some(y)) => x.is_finite() && y.is_finite() && (x - y).abs() <= bound,
        _ => false,
    }
}

fn bound(s: &State) -> f64 {
    2.0 * (s.terms - 1.0).max(0.0) * (f64::EPSILON / 2.0) * s.magnitude
}

/// The oracle's rows, as `(keys, value, bound)`, sorted: its groups, or with
/// a domain each domain row's group (the default where it has no value).
fn expected(case: &Case, with_domain: bool) -> Vec<(Vec<i64>, Option<f64>, f64)> {
    let groups = oracle(case);
    let mut out: Vec<(Vec<i64>, Option<f64>, f64)> =
        match case.domain.as_ref().filter(|_| with_domain) {
            None => (groups.iter())
                .map(|(k, s)| (k.clone(), s.valued.then_some(s.sum), bound(s)))
                .collect(),
            // The operands have no NULL keys, so a NULL domain key matches no
            // group under either equality.
            Some(d) => (d.rel.keys.iter())
                .map(|k| match groups.get(k) {
                    Some(s) if s.valued => (k.clone(), Some(s.sum), bound(s)),
                    _ => (k.clone(), d.missing, 0.0),
                })
                .collect(),
        };
    out.sort_by(|x, y| (&x.0, x.1.map(f64::to_bits)).cmp(&(&y.0, y.1.map(f64::to_bits))));
    out
}

/// Whether `ours` is the oracle's `want`: the same keys, and values within
/// the bound.
fn matches(tag: &str, ours: &Rows, want: &[(Vec<i64>, Option<f64>, f64)]) -> Result<(), String> {
    if ours.len() != want.len() {
        return Err(format!(
            "[{tag}] {} rows, the oracle has {}",
            ours.len(),
            want.len()
        ));
    }
    for ((k, v), (wk, wv, b)) in ours.iter().zip(want) {
        if k != wk || !agree(*v, *wv, *b) {
            return Err(format!(
                "[{tag}] row {k:?} = {v:?}, the oracle's {wk:?} = {wv:?} within {b:e}"
            ));
        }
    }
    Ok(())
}

/// The case with `a`'s factors mapped by `f`.
fn map_a(case: &Case, f: impl Fn(f64) -> f64) -> Case {
    let mut c = case.clone();
    for v in c.a.values.iter_mut().flatten() {
        *v = f(*v);
    }
    c
}

/// What the runs reached, for the coverage assertions.
#[derive(Default, Debug)]
struct Stats {
    seeds: usize,
    skipped: usize,
    cases: usize,
    dense: usize,
    /// Dense runs in each regime: blocks of the output, and blocks of `z`.
    blocked: usize,
    z_blocked: usize,
    /// Dense runs on each edge: `nx·ny` at 4,095–4,097; `nx` at 255–257 or
    /// `ny` at 2,047–2,049 with the output blocked; `nz` at 4,095–4,097 with
    /// `z` blocked.
    at_switch: usize,
    at_row_edge: usize,
    at_column_edge: usize,
    at_z_edge: usize,
    unique_declined: usize,
    unique_accepted: usize,
    null_domain_keys: usize,
    duplicates: usize,
    sparse: usize,
}

fn check(seed: u64, stats: &mut Stats) -> Result<(), String> {
    let mut rng = Rng::seeded(seed, 0);
    stats.seeds += 1;
    let Some(case) = case(&mut rng) else {
        stats.skipped += 1;
        return Ok(());
    };
    stats.cases += 1;
    let keys = case.outputs.len();
    let int64 = DataType::Int64;
    let first = run(&case, &mut rng, &int64, false, true, Mode::Best).expect("runs");
    let ours = rows(&first.batches, keys);
    let want = expected(&case, true);
    matches("oracle", &ours, &want)?;
    if case
        .domain
        .as_ref()
        .is_some_and(|d| d.rel.keys.iter().flatten().any(|&k| k == NULL_KEY))
    {
        stats.null_domain_keys += 1;
    }

    // [sparse]
    if case.sparse {
        stats.sparse += 1;
        if first.algorithm != Algorithm::Discriminate {
            return Err("[sparse] the dense algorithm ran on a sparse case".into());
        }
        return Ok(());
    }
    let dense = first.algorithm == Algorithm::Dense;
    if dense {
        stats.dense += 1;
        let (nx, nz, ny) = (
            case.cells(&case.xs),
            case.cells(&case.zs),
            case.cells(&case.ys),
        );
        let blocked = nx * ny >= FEW_CELLS;
        stats.blocked += usize::from(blocked);
        stats.z_blocked += usize::from(!blocked);
        let near = |n: usize, edge: usize| (edge - 1..=edge + 1).contains(&n);
        stats.at_switch += usize::from(near(nx * ny, FEW_CELLS));
        stats.at_row_edge += usize::from(blocked && near(nx, XB));
        stats.at_column_edge += usize::from(blocked && near(ny, YB));
        stats.at_z_edge += usize::from(!blocked && near(nz, ZB));
    }

    // [threads]
    let again = run(&case, &mut rng, &int64, false, true, Mode::Best).expect("runs");
    if bits(&rows(&again.batches, keys)) != bits(&ours) {
        return Err("[threads] other threads and batches changed the bits".into());
    }

    // [scale]: doubling and negating a's factors, without a domain, whose
    // default doesn't scale.
    let base = run(&case, &mut rng, &int64, false, false, Mode::Best).expect("runs");
    let base_rows = rows(&base.batches, keys);
    type Map = fn(f64) -> f64;
    let maps: [(&str, Map); 2] = [("doubling", |v| v * 2.0), ("negating", |v| -v)];
    for (tag, f) in maps {
        let mapped = map_a(&case, f);
        let r = run(&mapped, &mut rng, &int64, false, false, Mode::Best).expect("runs");
        if r.algorithm != base.algorithm {
            continue;
        }
        let mut want: Rows = base_rows
            .iter()
            .map(|(k, v)| (k.clone(), v.map(f)))
            .collect();
        sort(&mut want);
        // Negating can turn a sum of +0.0 into -0.0 or back, as SQL's sum
        // starts at +0.0, so zeros compare by value.
        let zero = |rows: &Rows| -> Rows {
            rows.iter()
                .map(|(k, v)| (k.clone(), v.map(|x| if x == 0.0 { 0.0 } else { x })))
                .collect()
        };
        let mut got = zero(&rows(&r.batches, keys));
        let mut want = zero(&want);
        sort(&mut got);
        sort(&mut want);
        if bits(&got) != bits(&want) {
            return Err(format!(
                "[scale] {tag} a's factors didn't {tag} the results exactly"
            ));
        }
    }

    // [relabel]: a permutation of each output dimension's keys.
    let mut perms: HashMap<usize, Vec<i64>> = HashMap::new();
    for &d in case.xs.iter().chain(&case.ys) {
        let mut p: Vec<i64> = (0..case.extent[d] as i64).collect();
        for i in (1..p.len()).rev() {
            p.swap(i, rng.below(i + 1));
        }
        perms.insert(d, p);
    }
    let apply = |d: usize, k: i64| -> i64 {
        match perms.get(&d) {
            Some(p) if k >= 0 && (k as usize) < p.len() => p[k as usize],
            _ => k,
        }
    };
    let relabel = |rel: &mut Rel| {
        for row in rel.keys.iter_mut() {
            for (slot, &d) in rel.dims.iter().enumerate() {
                row[slot] = apply(d, row[slot]);
            }
        }
    };
    let mut relabeled = case.clone();
    relabel(&mut relabeled.a);
    relabel(&mut relabeled.b);
    if let Some(d) = relabeled.domain.as_mut() {
        relabel(&mut d.rel);
    }
    let r = run(&relabeled, &mut rng, &int64, false, true, Mode::Best).expect("runs");
    if r.algorithm == first.algorithm {
        let mut want: Rows = ours.clone();
        for (k, _) in want.iter_mut() {
            for (slot, &d) in case.outputs.iter().enumerate() {
                k[slot] = apply(d, k[slot]);
            }
        }
        sort(&mut want);
        if bits(&rows(&r.batches, keys)) != bits(&want) {
            return Err("[relabel] permuting output keys changed a value's bits".into());
        }
    }

    // [key-type]
    let ty = [DataType::Int32, DataType::Int16, DataType::UInt32][rng.below(3)].clone();
    let fits =
        |r: &Rel| (r.keys.iter().flatten()).all(|&k| k == NULL_KEY || (0..32_000).contains(&k));
    if fits(&case.a) && fits(&case.b) && case.domain.as_ref().is_none_or(|d| fits(&d.rel)) {
        let narrow = run(&case, &mut rng, &ty, false, true, Mode::Best).expect("runs");
        if narrow.algorithm == first.algorithm && bits(&rows(&narrow.batches, keys)) != bits(&ours)
        {
            return Err(format!("[key-type] {ty} keys changed the bits"));
        }
    }

    // [domain]
    if let Some(dom) = &case.domain {
        if base.algorithm == first.algorithm {
            let groups: HashMap<Vec<i64>, Option<f64>> = base_rows.iter().cloned().collect();
            for (k, v) in &ours {
                let expect = groups.get(k).copied().flatten().or(dom.missing);
                if v.map(f64::to_bits) != expect.map(f64::to_bits) {
                    return Err(format!(
                        "[domain] row {k:?} = {v:?}, its group without a domain gives {expect:?}"
                    ));
                }
            }
        }
    }

    // [unique]
    if let (Some(dom), true) = (&case.domain, dense) {
        let in_range = |k: &[i64]| {
            (k.iter().zip(&case.outputs))
                .all(|(&k, &d)| k != NULL_KEY && k >= 0 && (k as usize) < case.extent[d])
        };
        let mut seen = HashSet::new();
        let repeated = (dom.rel.keys.iter())
            .filter(|k| in_range(k))
            .any(|k| !seen.insert(k.clone()));
        let any = Mode::Dense(DomainRows::Any);
        let unchecked = run(&case, &mut rng, &int64, false, true, any);
        let unique = Mode::Dense(DomainRows::UniqueKeys);
        let checked = run(&case, &mut rng, &int64, false, true, unique);
        match (unchecked, checked) {
            (Some(_), Some(_)) if repeated => {
                return Err("[unique] accepted two domain rows that share a cell".into())
            }
            (Some(_), None) if !repeated => {
                return Err("[unique] declined a domain whose rows have distinct cells".into())
            }
            (Some(u), Some(c)) => {
                stats.unique_accepted += 1;
                if bits(&rows(&u.batches, keys)) != bits(&rows(&c.batches, keys)) {
                    return Err("[unique] checking the domain's keys changed the bits".into());
                }
            }
            (Some(_), None) => stats.unique_declined += 1,
            (None, _) => return Err("[unique] the dense algorithm declined a fold it ran".into()),
        }
    }

    // [discriminate]: one row of b that joins nothing.
    let mut extra = case.clone();
    extra.b.keys.push(vec![-1; extra.b.dims.len()]);
    extra.b.values.push(Some(1.0));
    let discriminated = run(&extra, &mut rng, &int64, false, true, Mode::Best).expect("runs");
    if discriminated.algorithm != Algorithm::Discriminate {
        return Err(
            "[discriminate] a negative key didn't move the fold to the discrimination algorithm"
                .into(),
        );
    }
    matches("discriminate", &rows(&discriminated.batches, keys), &want)?;

    // [duplicates]: a repeated row of a or b.
    if dense {
        let mut dup = case.clone();
        let side = if rng.chance(0.5) {
            &mut dup.a
        } else {
            &mut dup.b
        };
        let r = rng.below(side.keys.len());
        side.keys.push(side.keys[r].clone());
        side.values.push(Some(0.5));
        let ran = run(&dup, &mut rng, &int64, false, true, Mode::Best).expect("runs");
        if ran.algorithm != Algorithm::Discriminate {
            return Err("[duplicates] a repeated key tuple stayed on the dense algorithm".into());
        }
        matches(
            "duplicates",
            &rows(&ran.batches, keys),
            &expected(&dup, true),
        )?;
        stats.duplicates += 1;
    }

    // [swap]
    if case.domain.is_none() {
        let swapped = run(&case, &mut rng, &int64, true, true, Mode::Best).expect("runs");
        matches("swap", &rows(&swapped.batches, keys), &want)?;
    }
    Ok(())
}

#[test]
fn dense_metamorphic() {
    let env = |n: &str| std::env::var(n).ok().and_then(|s| s.parse::<u64>().ok());
    let base = env("EINFOLD_SEED").unwrap_or(0x000b_10c5);
    let cases = env("EINFOLD_CASES").unwrap_or(60) as usize;
    let mut stats = Stats::default();
    let mut failures = Vec::new();
    let mut seed = base;
    // Seeds whose shapes are too large for the oracle are skipped; the target
    // is a number of cases run, within four times as many seeds.
    while stats.cases < cases && stats.seeds < 4 * cases.max(1) {
        if let Err(e) = check(seed, &mut stats) {
            failures.push(format!("seed {seed}: {e}"));
        }
        seed += 1;
    }
    eprintln!("{stats:?}");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    assert!(
        stats.cases >= cases,
        "only {} cases ran: {stats:?}",
        stats.cases
    );
    if cases >= 60 {
        let reached = [
            (
                "the dense algorithm ran in a quarter of cases",
                stats.dense * 4 >= stats.cases,
            ),
            ("the output was blocked", stats.blocked > 0),
            ("z was blocked", stats.z_blocked > 0),
            ("nx·ny was at the switch", stats.at_switch > 0),
            ("nx was at a row edge", stats.at_row_edge > 0),
            ("ny was at a column edge", stats.at_column_edge > 0),
            ("nz was at a z edge", stats.at_z_edge > 0),
            ("the unique check accepted", stats.unique_accepted > 0),
            ("the unique check declined", stats.unique_declined > 0),
            ("a domain had NULL keys", stats.null_domain_keys > 0),
            ("a duplicate moved to discrimination", stats.duplicates > 0),
            ("a sparse case ran", stats.sparse > 0),
        ];
        for (what, ok) in reached {
            assert!(ok, "coverage: expected that {what}: {stats:?}");
        }
    }
}
