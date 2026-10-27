// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Properties of the kernels at the dense algorithm's scale: shapes near its
//! block sizes, against an independent oracle and against themselves.
//!
//! `kernels_vs_sql.rs` checks the kernels against SQL on small tables and
//! three fixed large shapes. The dense algorithm changes how it splits work
//! at fixed sizes (blocks of 256 output rows and 2,048 output columns once
//! the output has 4,096 cells, and of 4,096 positions of the summed
//! dimension below that), so its bugs live at those edges. Here each case
//! draws a fold `out[x…, y…] = Σ_z a[x…, z…]·b[z…, y…]` with zero to two
//! dimensions per role and extents drawn near the edges, a domain half the
//! time, and checks:
//!
//! - **`[oracle]`** the result equals a nested-loop evaluation of SQL's
//!   semantics written here, with no code shared with the kernels: the same
//!   groups (or domain rows), NULL where SQL gives NULL, NaN where it gives
//!   NaN, and otherwise within `2·(n − 1)·u·Σ|terms|` of the oracle's sum;
//! - **`[key-type]`** the same fold over keys cast to another integer type
//!   gives the same bits: keys are positions whatever their width;
//! - **`[swap]`** swapping the operands (and so X and Y) gives the same
//!   values within the bound. The dense algorithm treats its operands
//!   differently (only `b` may have holes), so this crosses its conditions;
//! - **`[hash]`** a row of `b` that joins nothing (a negative key), which
//!   moves the fold to the hash algorithm, changes no group and no value
//!   beyond the bound;
//! - **`[domain]`** with a domain, each domain row gets exactly the bits of
//!   its group in the same fold without a domain, or the default;
//! - **`[threads]`** the same bits on any number of threads and batching.
//!
//! Reproduce with `EINFOLD_SEED=<seed>`; search longer with
//! `EINFOLD_CASES=<n>`.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use einfold_ir::{
    Aggregate, ColumnIndex, Dimension, DimensionId, Domain, DomainKey, Fold, Key, KeyEquality,
    Operand, Output, RowValue,
};
use einfold_kernels::{Algorithm, FloatKeys, Host, Options, Relation};

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x2545_f491_4f6c_dd1d) ^ 0x00de_75e5)
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
    fn chance(&mut self, p: f64) -> bool {
        ((self.next() >> 11) as f64) / ((1u64 << 53) as f64) < p
    }
}

/// An extent near one of the dense algorithm's edges, or a small one.
fn extent(rng: &mut Rng) -> usize {
    let edges = [
        1, 2, 3, 16, 63, 64, 65, 255, 256, 257, 2047, 2048, 2049, 4095, 4096, 4097,
    ];
    match rng.below(3) {
        0 => 1 + rng.below(6),
        _ => edges[rng.below(edges.len())],
    }
}

/// One operand: rows of integer keys (one per dimension it holds) and a
/// factor.
#[derive(Clone)]
struct Rel {
    dims: Vec<usize>,
    keys: Vec<Vec<i64>>,
    values: Vec<Option<f64>>,
}

/// A case: dimension roles, extents, the operands, an optional domain over
/// the outputs, and its default.
struct Case {
    xs: Vec<usize>,
    zs: Vec<usize>,
    ys: Vec<usize>,
    outputs: Vec<usize>,
    a: Rel,
    b: Rel,
    domain: Option<(Rel, Option<f64>)>,
}

fn value(rng: &mut Rng, specials: bool) -> Option<f64> {
    match rng.below(100) {
        0..=4 => None,
        5..=6 if specials => Some(f64::NAN),
        7 if specials => Some(f64::INFINITY),
        8..=10 => Some(-0.0),
        11..=16 => Some(if rng.chance(0.5) { 1e16 } else { -1e16 }),
        _ => Some(((rng.next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 8.0),
    }
}

/// Every key tuple over `dims` with these extents, each kept with
/// probability `present`.
fn rel(rng: &mut Rng, dims: &[usize], extent: &[usize], present: f64, specials: bool) -> Rel {
    let total: usize = dims.iter().map(|&d| extent[d]).product();
    let mut keys = Vec::new();
    let mut values = Vec::new();
    for mut t in 0..total {
        let mut k = vec![0i64; dims.len()];
        for (slot, &d) in dims.iter().enumerate().rev() {
            k[slot] = (t % extent[d]) as i64;
            t /= extent[d];
        }
        if rng.chance(present) {
            keys.push(k);
            values.push(value(rng, specials));
        }
    }
    Rel {
        dims: dims.to_vec(),
        keys,
        values,
    }
}

fn case(rng: &mut Rng) -> Option<Case> {
    // Dimensions 0.. in roles; zero to two per role, one to two summed.
    let (nx, nz, ny) = (rng.below(3), 1 + rng.below(2), rng.below(3));
    let n = nx + nz + ny;
    let xs: Vec<usize> = (0..nx).collect();
    let zs: Vec<usize> = (nx..nx + nz).collect();
    let ys: Vec<usize> = (nx + nz..n).collect();
    let extent: Vec<usize> = (0..n).map(|_| extent(rng)).collect();
    let cells = |dims: &[usize]| dims.iter().map(|&d| extent[d]).product::<usize>();
    // Keep the oracle's work and the tables small enough to be quick.
    let (cx, cz, cy) = (cells(&xs), cells(&zs), cells(&ys));
    if cx * cz > 300_000 || cz * cy > 300_000 || cx * cy > 600_000 || cx * cz * cy > 30_000_000 {
        return None;
    }
    let mut outputs: Vec<usize> = xs.iter().chain(&ys).copied().collect();
    for i in (1..outputs.len()).rev() {
        outputs.swap(i, rng.below(i + 1));
    }
    let specials = rng.chance(0.3);
    let a_dims: Vec<usize> = xs.iter().chain(&zs).copied().collect();
    let b_dims: Vec<usize> = zs.iter().chain(&ys).copied().collect();
    let (pa, pb) = (
        if rng.chance(0.5) { 1.0 } else { 0.7 },
        if rng.chance(0.7) { 1.0 } else { 0.9 },
    );
    let a = rel(rng, &a_dims, &extent, pa, specials);
    let b = rel(rng, &b_dims, &extent, pb, specials);
    let domain = rng.chance(0.5).then(|| {
        // Keys one past each extent, some rows repeated.
        let ext: Vec<usize> = extent.iter().map(|e| e + 1).collect();
        let mut d = rel(rng, &outputs, &ext, 0.6, false);
        let extra: Vec<usize> = (0..d.keys.len() / 10)
            .map(|_| rng.below(d.keys.len()))
            .collect();
        for i in extra {
            d.keys.push(d.keys[i].clone());
            d.values.push(None);
        }
        let missing = [None, Some(0.0), Some(2.5)][rng.below(3)];
        (d, missing)
    });
    Some(Case {
        xs,
        zs,
        ys,
        outputs,
        a,
        b,
        domain,
    })
}

/// A relation's keys as columns of `ty`, then its values.
fn batch(rel: &Rel, ty: &DataType, extra: Option<(i64, f64)>) -> (SchemaRef, RecordBatch) {
    let mut keys = rel.keys.clone();
    let mut values = rel.values.clone();
    if let Some((k, v)) = extra {
        keys.push(vec![k; rel.dims.len()]);
        values.push(Some(v));
    }
    let mut fields: Vec<Field> = (0..rel.dims.len())
        .map(|i| Field::new(format!("k{i}"), ty.clone(), true))
        .collect();
    fields.push(Field::new("v", DataType::Float64, true));
    let mut columns: Vec<ArrayRef> = (0..rel.dims.len())
        .map(|i| {
            let col: ArrayRef = Arc::new(Int64Array::from(
                keys.iter().map(|k| k[i]).collect::<Vec<_>>(),
            ));
            arrow::compute::cast(&col, ty).expect("keys fit")
        })
        .collect();
    columns.push(Arc::new(Float64Array::from(values)));
    let schema = Arc::new(Schema::new(fields));
    let b = RecordBatch::try_new(Arc::clone(&schema), columns).expect("a batch");
    (schema, b)
}

fn fold_of(case: &Case, swapped: bool, with_domain: bool) -> Fold {
    let n = case.xs.len() + case.zs.len() + case.ys.len();
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
            domain_key: domain.map(|_| DomainKey::new(ColumnIndex(i), KeyEquality::Equal)),
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
        domain.map(|(d, missing)| Domain {
            carry: (0..d.dims.len()).map(ColumnIndex).collect(),
            missing: *missing,
        }),
    )
    .expect("a valid fold")
}

/// Splits a batch into `parts` random batches, empty ones included.
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

struct Run {
    algorithm: Algorithm,
    batches: Vec<RecordBatch>,
}

fn run(
    case: &Case,
    rng: &mut Rng,
    ty: &DataType,
    swapped: bool,
    with_domain: bool,
    hash_row: bool,
) -> Run {
    let threads = 1 + rng.below(8);
    let fold = fold_of(case, swapped, with_domain);
    let (first, second) = if swapped {
        (&case.b, &case.a)
    } else {
        (&case.a, &case.b)
    };
    let (sa, ba) = batch(first, ty, None);
    let (sb, bb) = batch(second, ty, hash_row.then_some((-1, 1.0)));
    let dom = case
        .domain
        .as_ref()
        .filter(|_| with_domain)
        .map(|(d, _)| batch(d, ty, None));
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
    let options = Options {
        host: Host {
            float_keys: FloatKeys::SignedZerosEqual,
        },
        threads,
        ..Options::default()
    };
    let folded = einfold_kernels::fold(
        &fold,
        Relation::new(&sa, &ba),
        Relation::new(&sb, &bb),
        domain,
        schema,
        options,
    )
    .expect("the fold runs");
    Run {
        algorithm: folded.algorithm,
        batches: folded.batches,
    }
}

/// A group's state in the oracle: whether it exists, its sum of non-NULL
/// products, their count and magnitude.
#[derive(Clone, Copy, Default)]
struct State {
    valued: bool,
    sum: f64,
    terms: f64,
    magnitude: f64,
}

/// SQL's answer, by nested loops over a hash of b by its z keys: each
/// group's state, keyed by the outputs' keys in output order.
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

/// The result's rows: keys (as `i64`) and value.
fn result_rows(batches: &[RecordBatch], keys: usize) -> Vec<(Vec<i64>, Option<f64>)> {
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
            out.push((
                cols.iter().map(|c| c.value(r)).collect(),
                v.is_valid(r).then(|| v.value(r)),
            ));
        }
    }
    out.sort_by(|x, y| (&x.0, x.1.map(f64::to_bits)).cmp(&(&y.0, y.1.map(f64::to_bits))));
    out
}

fn bits(rows: &[(Vec<i64>, Option<f64>)]) -> Vec<(Vec<i64>, Option<u64>)> {
    rows.iter()
        .map(|(k, v)| (k.clone(), v.map(f64::to_bits)))
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

/// The oracle's rows for the case: its groups, or with a domain each domain
/// row's group (with the default where it has no value), as `(keys, value,
/// bound)`.
fn expected(case: &Case, with_domain: bool) -> Vec<(Vec<i64>, Option<f64>, f64)> {
    let groups = oracle(case);
    let mut out: Vec<(Vec<i64>, Option<f64>, f64)> =
        match case.domain.as_ref().filter(|_| with_domain) {
            None => (groups.iter())
                .map(|(k, s)| (k.clone(), s.valued.then_some(s.sum), bound(s)))
                .collect(),
            Some((d, missing)) => (d.keys.iter())
                .map(|k| match groups.get(k) {
                    Some(s) if s.valued => (k.clone(), Some(s.sum), bound(s)),
                    _ => (k.clone(), *missing, 0.0),
                })
                .collect(),
        };
    out.sort_by(|x, y| (&x.0, x.1.map(f64::to_bits)).cmp(&(&y.0, y.1.map(f64::to_bits))));
    out
}

fn check(seed: u64, stats: &mut [usize; 3]) -> Result<(), String> {
    let mut rng = Rng::new(seed);
    let Some(case) = case(&mut rng) else {
        return Ok(());
    };
    let keys = case.outputs.len();
    let domain = case.domain.is_some();
    let first = run(&case, &mut rng, &DataType::Int64, false, true, false);
    stats[0] += 1;
    stats[1] += usize::from(first.algorithm == Algorithm::Dense);
    let ours = result_rows(&first.batches, keys);

    // [oracle]
    let want = expected(&case, true);
    if ours.len() != want.len() {
        return Err(format!(
            "[oracle] {} rows, the oracle has {}",
            ours.len(),
            want.len()
        ));
    }
    for ((k, v), (wk, wv, b)) in ours.iter().zip(&want) {
        if k != wk || !agree(*v, *wv, *b) {
            return Err(format!(
                "[oracle] row {k:?} = {v:?}, the oracle's {wk:?} = {wv:?} within {b:e}"
            ));
        }
    }
    // [threads]
    let again = run(&case, &mut rng, &DataType::Int64, false, true, false);
    if bits(&result_rows(&again.batches, keys)) != bits(&ours) {
        return Err("[threads] other threads and batches changed the bits".into());
    }
    // [key-type]
    let ty = [DataType::Int32, DataType::Int16, DataType::UInt32][rng.below(3)].clone();
    let fits = |r: &Rel| r.keys.iter().flatten().all(|&k| k < 32_000);
    if fits(&case.a) && fits(&case.b) && case.domain.as_ref().is_none_or(|(d, _)| fits(d)) {
        let narrow = run(&case, &mut rng, &ty, false, true, false);
        if narrow.algorithm == first.algorithm
            && bits(&result_rows(&narrow.batches, keys)) != bits(&ours)
        {
            return Err(format!("[key-type] {ty} keys changed the bits"));
        }
    }
    // [domain]
    if domain {
        let bare = run(&case, &mut rng, &DataType::Int64, false, false, false);
        if bare.algorithm == first.algorithm {
            let groups: HashMap<Vec<i64>, Option<f64>> =
                result_rows(&bare.batches, keys).into_iter().collect();
            let missing = case.domain.as_ref().expect("a domain").1;
            for (k, v) in &ours {
                let expect = groups.get(k).copied().flatten().or(missing);
                if v.map(f64::to_bits) != expect.map(f64::to_bits) {
                    return Err(format!(
                        "[domain] row {k:?} = {v:?}, its group without a domain gives {expect:?}"
                    ));
                }
            }
        }
    }
    // [hash]
    let hashed = run(&case, &mut rng, &DataType::Int64, false, true, true);
    stats[2] +=
        usize::from(first.algorithm == Algorithm::Dense && hashed.algorithm == Algorithm::Hash);
    let theirs = result_rows(&hashed.batches, keys);
    for ((k, v), (_, w, b)) in theirs.iter().zip(&want) {
        if !agree(*v, *w, *b) {
            return Err(format!(
                "[hash] row {k:?} = {v:?}, the oracle's {w:?} within {b:e}"
            ));
        }
    }
    // [swap]
    if !domain {
        let swapped = run(&case, &mut rng, &DataType::Int64, true, true, false);
        let theirs = result_rows(&swapped.batches, keys);
        if theirs.len() != want.len() {
            return Err(format!(
                "[swap] {} rows, the oracle has {}",
                theirs.len(),
                want.len()
            ));
        }
        for ((k, v), (wk, w, b)) in theirs.iter().zip(&want) {
            if k != wk || !agree(*v, *w, *b) {
                return Err(format!(
                    "[swap] row {k:?} = {v:?}, the oracle's {w:?} within {b:e}"
                ));
            }
        }
    }
    Ok(())
}

#[test]
fn dense_metamorphic() {
    let base = std::env::var("EINFOLD_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x000b_10c5);
    let cases: u64 = std::env::var("EINFOLD_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);
    let mut stats = [0usize; 3];
    let mut failures = Vec::new();
    for i in 0..cases {
        let seed = base + i;
        if let Err(e) = check(seed, &mut stats) {
            failures.push(format!("seed {seed}: {e}"));
        }
    }
    eprintln!(
        "cases run {}, dense {}, dense moved to hash {}",
        stats[0], stats[1], stats[2]
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    if cases >= 50 {
        assert!(
            stats[1] * 4 >= stats[0],
            "the dense path ran in only {} of {}",
            stats[1],
            stats[0]
        );
    }
}
