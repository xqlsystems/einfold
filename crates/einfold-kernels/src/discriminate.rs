// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! EinFold's discrimination algorithm: a two-operand fold for keys of any
//! type and density, by sorting rather than hashing.
//!
//! Discrimination (Henglein, "Generic Discrimination: Sorting and
//! Partitioning Unshared Data in Linear Time", ICFP 2008) partitions keys by
//! their bytes, one radix pass per byte, with no comparisons. Here it gives
//! each distinct key a *rank*, `0..k`, in the order of its bytes:
//!
//! 1. **Keys as bytes.** Each operand's keys are put in Arrow's row format,
//!    which gives keys bytes that are equal exactly when the host calls the
//!    keys equal (see [`crate::FloatKeys`]). The join's keys are ranked over
//!    both operands at once, so that equal keys share a rank, and the
//!    outputs' over each operand, with the domain's keys if there is one.
//! 2. **Rows in order.** Each operand's rows are put in key order, `a`'s by
//!    its outputs then the join's keys, `b`'s by the join's keys then its
//!    outputs, with counting sorts over the ranks.
//! 3. **Gustavson's product** ("Two Fast Algorithms for Sparse Matrices",
//!    1978): for each of `a`'s output keys, each of its rows adds its factor
//!    times every row of `b` with its join key into an accumulator over
//!    `b`'s output keys.
//!
//! It follows SQL exactly, as the dense algorithm does where that applies:
//!
//! - **Bag semantics.** Every joined pair counts, duplicate key tuples
//!   included.
//! - **Group existence.** A group exists when any joined pair reaches it,
//!   whatever the factors. Its sum skips a product with a NULL factor, as
//!   `SUM` skips NULL, and is NULL when every product was NULL.
//! - **NULL keys.** A NULL in a dimension compared with `=` joins nothing; one
//!   compared with `IS NOT DISTINCT FROM` joins NULL. Grouping keeps NULL as a
//!   group of its own.
//! - **A domain is a left join:** each domain row gets the group its keys
//!   match, under each key's equality, or none.
//!
//! **Determinism.** Every addition happens in an order fixed by the rows'
//! contents: keys, and among rows with the same keys, their factors' bits.
//! So the bits depend on the inputs' rows, not on how they were batched or
//! ordered, nor on the number of threads.

use std::sync::Arc;
use std::thread;

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, UInt32Array};
use arrow::compute;
use arrow::datatypes::{DataType, SchemaRef};
use arrow::error::ArrowError;
use einfold_ir::{DimensionId, Fold, KeyEquality};

use crate::keys::{self, Keys};
use crate::options::Options;
use crate::{dense, Result};

/// Output keys of `a` per block of the product, each block on one thread.
const XB: usize = 256;
/// Below this many rows, a radix pass runs on one thread.
const CHUNK: usize = 65_536;
/// Buckets of the byte-by-byte sort this small are finished by comparison.
const SMALL: usize = 16;

/// Each row's rank among the distinct keys, and one row for each rank.
struct Ranks {
    rank: Vec<u32>,
    first: Vec<u32>,
}

/// The keys as `u64`s that order as their bytes do, when every key has the
/// same width and at most 8 of its bytes differ between keys: the bytes all
/// keys share are dropped. Integer and float keys fit.
fn words(keys: &Keys, n: usize) -> Option<Vec<u64>> {
    let first = keys.get(0);
    let width = first.len();
    let mut varies = vec![false; width];
    for i in 1..n {
        let key = keys.get(i);
        if key.len() != width {
            return None;
        }
        for (v, (a, b)) in varies.iter_mut().zip(key.iter().zip(first)) {
            *v |= a != b;
        }
    }
    let bytes: Vec<usize> = (0..width).filter(|&b| varies[b]).collect();
    if bytes.len() > 8 {
        return None;
    }
    let word = |i: usize| {
        let key = keys.get(i);
        bytes
            .iter()
            .fold(0u64, |w, &b| (w << 8) | u64::from(key[b]))
    };
    Some((0..n).map(word).collect())
}

/// `(word, row)` for every row, sorted by word, stably: a radix sort over
/// the pairs, least significant byte first, each pass counting and
/// scattering by chunks of rows on `threads` threads, and skipping bytes that
/// no word varies in.
fn radix(words: &[u64], threads: usize) -> Vec<(u64, u32)> {
    let n = words.len();
    let chunks = threads.min(n.div_ceil(CHUNK)).max(1);
    let bounds: Vec<(usize, usize)> = (0..chunks)
        .map(|c| (n * c / chunks, n * (c + 1) / chunks))
        .collect();
    let mut pairs: Vec<(u64, u32)> = (words.iter().enumerate())
        .map(|(i, &w)| (w, i as u32))
        .collect();
    let mut next = vec![(0u64, 0u32); n];
    let (any, all) = (words.iter()).fold((0u64, u64::MAX), |(o, a), &w| (o | w, a & w));
    for shift in (0..64).step_by(8) {
        if ((any ^ all) >> shift) & 0xff == 0 {
            continue;
        }
        let byte = |w: u64| ((w >> shift) & 0xff) as usize;
        let counts: Vec<[usize; 256]> = dense::par_tasks(chunks, threads, |c| {
            let (lo, hi) = bounds[c];
            let mut count = [0usize; 256];
            for &(w, _) in &pairs[lo..hi] {
                count[byte(w)] += 1;
            }
            count
        });
        // Each chunk's rows of each byte go to their own contiguous region,
        // byte-major then chunk, so the regions split `next` without
        // overlapping, and each chunk scatters into its own.
        let mut regions: Vec<Vec<&mut [(u64, u32)]>> =
            (0..chunks).map(|_| Vec::with_capacity(256)).collect();
        let mut rest = &mut next[..];
        for byte in 0..256 {
            for (count, region) in counts.iter().zip(regions.iter_mut()) {
                let (head, tail) = rest.split_at_mut(count[byte]);
                region.push(head);
                rest = tail;
            }
        }
        let pairs_ref = &pairs;
        thread::scope(|s| {
            for (&(lo, hi), mut region) in bounds.iter().zip(regions) {
                s.spawn(move || {
                    let mut at = [0usize; 256];
                    for &p in &pairs_ref[lo..hi] {
                        let b = byte(p.0);
                        region[b][at[b]] = p;
                        at[b] += 1;
                    }
                });
            }
        });
        std::mem::swap(&mut pairs, &mut next);
    }
    pairs
}

/// Sorts `index` by `keys.get(i)`, most significant byte first: a counting
/// pass per byte, then each bucket the same way at the next byte. A key that
/// ends sorts before one that continues, and equal keys keep their order.
///
/// The buckets still to sort are a list on the heap rather than calls on the
/// stack: keys can share a prefix thousands of bytes long, as URLs or paths
/// do, and a call per byte would overflow a thread's stack.
fn by_bytes(keys: &Keys, index: &mut [u32], scratch: &mut Vec<u32>) {
    // Each range of `index` to sort, and the byte it is sorted by.
    let mut work = vec![(0, index.len(), 0usize)];
    while let Some((lo, hi, depth)) = work.pop() {
        let index = &mut index[lo..hi];
        let rest = |i: u32| keys.get(i as usize).get(depth..).unwrap_or(&[]);
        if index.len() <= SMALL {
            index.sort_by(|&a, &b| rest(a).cmp(rest(b)));
            continue;
        }
        // Bucket 0 holds keys that end here, bucket 1 + b those whose byte
        // is b.
        let bucket = |i: u32| {
            keys.get(i as usize)
                .get(depth)
                .map_or(0, |&b| usize::from(b) + 1)
        };
        let mut start = [0usize; 258];
        for &i in index.iter() {
            start[bucket(i) + 1] += 1;
        }
        if start[1] == 0 && start.iter().filter(|&&c| c > 0).count() == 1 {
            // Every key has the same byte here.
            work.push((lo, hi, depth + 1));
            continue;
        }
        for b in 1..258 {
            start[b] += start[b - 1];
        }
        let mut at = start;
        scratch.clear();
        scratch.resize(index.len(), 0);
        for &i in index.iter() {
            let b = bucket(i);
            scratch[at[b]] = i;
            at[b] += 1;
        }
        index.copy_from_slice(scratch);
        for b in 1..257 {
            if start[b + 1] - start[b] > 1 {
                work.push((lo + start[b], lo + start[b + 1], depth + 1));
            }
        }
    }
}

/// The ranks of the `n` keys of `keys`, in the order of their bytes.
fn ranks(keys: &Keys, n: usize, threads: usize) -> Ranks {
    let mut rank = vec![0u32; n];
    let mut first = Vec::new();
    if n == 0 {
        return Ranks { rank, first };
    }
    match words(keys, n) {
        Some(words) => return ranks_of_words(&words, threads),
        None => {
            let mut index: Vec<u32> = (0..n as u32).collect();
            by_bytes(keys, &mut index, &mut Vec::new());
            let mut previous: Option<&[u8]> = None;
            for &i in &index {
                let key = keys.get(i as usize);
                if previous != Some(key) {
                    first.push(i);
                    previous = Some(key);
                }
                rank[i as usize] = first.len() as u32 - 1;
            }
        }
    }
    Ranks { rank, first }
}

/// The ranks of the keys in `columns`, `n` rows. A single integer column
/// with no NULLs is ranked by its values directly; other keys by the bytes
/// of Arrow's row format.
fn rank_columns(columns: &[ArrayRef], n: usize, threads: usize) -> Result<Ranks> {
    use arrow::array::AsArray;
    use arrow::datatypes::{Int32Type, Int64Type, UInt32Type, UInt64Type};
    // Flipping the sign bit orders signed integers as unsigned words.
    let flip = |k: i64| (k as u64) ^ (1 << 63);
    let integers: Option<Vec<u64>> = match columns {
        [c] if c.null_count() == 0 => match c.data_type() {
            DataType::Int64 => Some(
                c.as_primitive::<Int64Type>()
                    .values()
                    .iter()
                    .map(|&k| flip(k))
                    .collect(),
            ),
            DataType::Int32 => Some(
                c.as_primitive::<Int32Type>()
                    .values()
                    .iter()
                    .map(|&k| flip(i64::from(k)))
                    .collect(),
            ),
            DataType::UInt64 => Some(c.as_primitive::<UInt64Type>().values().to_vec()),
            DataType::UInt32 => Some(
                c.as_primitive::<UInt32Type>()
                    .values()
                    .iter()
                    .map(|&k| u64::from(k))
                    .collect(),
            ),
            _ => None,
        },
        _ => None,
    };
    Ok(match integers {
        Some(words) => ranks_of_words(&words, threads),
        None => ranks(&Keys::new(columns)?, n, threads),
    })
}

/// The ranks of `words`, in their order.
fn ranks_of_words(words: &[u64], threads: usize) -> Ranks {
    let mut rank = vec![0u32; words.len()];
    let mut first = Vec::new();
    let mut previous = None;
    for (w, i) in radix(words, threads) {
        if previous != Some(w) {
            first.push(i);
            previous = Some(w);
        }
        rank[i as usize] = first.len() as u32 - 1;
    }
    Ranks { rank, first }
}

/// `rows` in order of `key` (each below `k`), stably: a counting sort.
fn counting(rows: &[u32], key: impl Fn(u32) -> usize, k: usize) -> Vec<u32> {
    let mut start = vec![0usize; k + 1];
    for &r in rows {
        start[key(r) + 1] += 1;
    }
    for i in 0..k {
        start[i + 1] += start[i];
    }
    let mut out = vec![0u32; rows.len()];
    for &r in rows {
        let slot = &mut start[key(r)];
        out[*slot] = r;
        *slot += 1;
    }
    out
}

/// `rows`, each run of rows with equal keys (per `same`) put in order of
/// their factors' bits, NULL first, so that duplicates add in a fixed order.
fn break_ties(rows: &mut [u32], same: impl Fn(u32, u32) -> bool, factors: &Float64Array) {
    let bits = |r: u32| {
        let r = r as usize;
        factors.is_valid(r).then(|| factors.value(r).to_bits())
    };
    let mut start = 0;
    while start < rows.len() {
        let mut end = start + 1;
        while end < rows.len() && same(rows[start], rows[end]) {
            end += 1;
        }
        if end - start > 1 {
            rows[start..end].sort_by_key(|&r| bits(r));
        }
        start = end;
    }
}

/// `columns` of every batch, concatenated.
fn concat(columns: &[Vec<ArrayRef>]) -> Result<Vec<ArrayRef>> {
    let width = columns.first().map_or(0, Vec::len);
    (0..width)
        .map(|c| {
            let parts: Vec<&dyn Array> = columns.iter().map(|b| b[c].as_ref()).collect();
            compute::concat(&parts)
        })
        .collect()
}

/// Which rows of `columns` can join: those with no NULL in a column whose
/// dimension is compared with `=`. NULL here is logical: a dictionary's NULL
/// value counts, as it does in SQL.
fn joinable(columns: &[ArrayRef], strict: &[bool], n: usize) -> Vec<bool> {
    let mut ok = vec![true; n];
    for (c, _) in columns.iter().zip(strict).filter(|(_, &s)| s) {
        if let Some(nulls) = c.logical_nulls() {
            for (o, valid) in ok.iter_mut().zip(nulls.iter()) {
                *o &= valid;
            }
        }
    }
    ok
}

/// One output group, in `(x, y)` order: its ranks, its sum, and whether any
/// product was non-NULL (otherwise the sum is NULL).
#[derive(Clone, Copy)]
struct Group {
    x: u32,
    y: u32,
    sum: f64,
    valued: bool,
}

/// See [`crate::fold`]. `a` and `b` are the operands, each as one batch.
pub(crate) fn fold(
    fold: &Fold,
    a: &RecordBatch,
    b: &RecordBatch,
    domain: Option<&[RecordBatch]>,
    schema: SchemaRef,
    options: &Options,
) -> Result<Vec<RecordBatch>> {
    let (host, threads) = (options.host, options.threads.max(1));
    // Rows are ranked as u32s: the operands' together, and each operand's
    // with the domain's.
    let (na, nb) = (a.num_rows(), b.num_rows());
    let nd: usize = domain.map_or(0, |d| d.iter().map(RecordBatch::num_rows).sum());
    let ranked = (na.saturating_add(nb)).max(na.max(nb).saturating_add(nd));
    if u32::try_from(ranked).is_err() {
        return Err(ArrowError::ComputeError(
            "EinFold's discrimination algorithm ranks at most 2³² - 1 rows together".into(),
        ));
    }
    let ops = fold.operands();
    let shared: Vec<DimensionId> = (fold.dimension_ids())
        .filter(|&d| ops[0].column_of(d).is_some() && ops[1].column_of(d).is_some())
        .collect();
    // Each output is taken from the first operand that holds it: its index
    // in the fold's outputs, split by operand.
    let from_a = |i: &usize| ops[0].column_of(fold.outputs()[*i].dimension).is_some();
    let (a_out, b_out): (Vec<usize>, Vec<usize>) = (0..fold.outputs().len()).partition(from_a);
    let column = |op: usize, batch: &RecordBatch, d: DimensionId| -> ArrayRef {
        let c = ops[op].column_of(d).expect("a dimension the operand holds");
        Arc::clone(batch.column(c.0))
    };
    let outputs = |op: usize, batch: &RecordBatch, outs: &[usize]| -> Vec<ArrayRef> {
        let dims = outs.iter().map(|&i| fold.outputs()[i].dimension);
        keys::host_keys(
            &dims.map(|d| column(op, batch, d)).collect::<Vec<_>>(),
            host,
        )
    };
    let factors = |op: usize, batch: &RecordBatch| -> Result<Float64Array> {
        let f = ops[op]
            .factor
            .and_then(|c| (batch.column(c.0).as_any().downcast_ref::<Float64Array>()).cloned());
        f.ok_or_else(|| ArrowError::InvalidArgumentError("factors must be Float64".into()))
    };
    let (a_factors, b_factors) = (factors(0, a)?, factors(1, b)?);
    // What ranking and ordering the rows allocate, per row ranked: its keys
    // in the row format, which take about what the key columns do; the
    // radix sort's words and pairs, 24 bytes; its rank, 4; and the orders of
    // each operand's rows, with b's laid out by z, about 40 more. The join's
    // keys rank a's and b's rows, and the outputs' rank them again, with the
    // domain's.
    let nd: usize = domain.map_or(0, |d| d.iter().map(RecordBatch::num_rows).sum());
    let key_bytes = |batch: &RecordBatch, op: usize| -> usize {
        (ops[op].keys.iter())
            .map(|k| batch.column(k.column.0).get_array_memory_size())
            .sum()
    };
    let ranked = (na + nb)
        .saturating_mul(2)
        .saturating_add(nd.saturating_mul(2));
    let working = ranked
        .saturating_mul(28)
        .saturating_add((na + nb).saturating_mul(40))
        .saturating_add(
            key_bytes(a, 0)
                .saturating_add(key_bytes(b, 1))
                .saturating_mul(2),
        );
    options.reserve(working, "the discrimination algorithm's ranks")?;

    // The join's keys, over both operands at once; rows with a NULL in a key
    // compared with `=` join nothing.
    let strict: Vec<bool> = (shared.iter())
        .map(|&d| fold.dimension(d).equality == KeyEquality::Equal)
        .collect();
    let a_shared = keys::host_keys(
        &shared.iter().map(|&d| column(0, a, d)).collect::<Vec<_>>(),
        host,
    );
    let b_shared = keys::host_keys(
        &shared.iter().map(|&d| column(1, b, d)).collect::<Vec<_>>(),
        host,
    );
    let (a_joins, b_joins) = (
        joinable(&a_shared, &strict, na),
        joinable(&b_shared, &strict, nb),
    );
    let z = rank_columns(&concat(&[a_shared, b_shared])?, na + nb, threads)?;

    // The outputs' keys, each operand's with the domain's keys for them.
    let (a_outputs, b_outputs) = (outputs(0, a, &a_out), outputs(1, b, &b_out));
    let domain_key = |i: usize| fold.outputs()[i].domain_key.expect("a fold with a domain");
    let domain_columns = |outs: &[usize]| -> Result<Vec<ArrayRef>> {
        let Some(domain) = domain else {
            return Ok(Vec::new());
        };
        let per_batch: Vec<Vec<ArrayRef>> = (domain.iter())
            .map(|batch| {
                let cols: Vec<ArrayRef> = (outs.iter())
                    .map(|&i| Arc::clone(batch.column(domain_key(i).column.0)))
                    .collect();
                keys::host_keys(&cols, host)
            })
            .collect();
        concat(&per_batch)
    };
    let with_domain = |own: &[ArrayRef], outs: &[usize], n: usize| -> Result<Ranks> {
        match nd {
            0 => rank_columns(own, n, threads),
            _ => rank_columns(
                &concat(&[own.to_vec(), domain_columns(outs)?])?,
                n + nd,
                threads,
            ),
        }
    };
    let x = with_domain(&a_outputs, &a_out, na)?;
    let y = with_domain(&b_outputs, &b_out, nb)?;
    let (nx, nz, ny) = (
        x.first.len().max(1),
        z.first.len().max(1),
        y.first.len().max(1),
    );

    // a's joining rows in (x, z) order, b's in (z, y) order, duplicates by
    // their factors.
    let a_z = |r: u32| z.rank[r as usize] as usize;
    let b_z = |r: u32| z.rank[na + r as usize] as usize;
    let a_rows: Vec<u32> = (0..na as u32).filter(|&r| a_joins[r as usize]).collect();
    let a_rows = counting(&a_rows, a_z, nz);
    let mut a_rows = counting(&a_rows, |r| x.rank[r as usize] as usize, nx);
    let same_a = |p: u32, q: u32| x.rank[p as usize] == x.rank[q as usize] && a_z(p) == a_z(q);
    break_ties(&mut a_rows, same_a, &a_factors);
    let b_rows: Vec<u32> = (0..nb as u32).filter(|&r| b_joins[r as usize]).collect();
    let b_rows = counting(&b_rows, |r| y.rank[r as usize] as usize, ny);
    let mut b_rows = counting(&b_rows, b_z, nz);
    let same_b = |p: u32, q: u32| b_z(p) == b_z(q) && y.rank[p as usize] == y.rank[q as usize];
    break_ties(&mut b_rows, same_b, &b_factors);
    // Where each x's rows of a, and each z's rows of b, start.
    let starts = |rows: &[u32], key: &dyn Fn(u32) -> usize, k: usize| {
        let mut start = vec![0usize; k + 1];
        for &r in rows {
            start[key(r) + 1] += 1;
        }
        for i in 0..k {
            start[i + 1] += start[i];
        }
        start
    };
    let a_start = starts(&a_rows, &|r| x.rank[r as usize] as usize, nx);
    let b_start = starts(&b_rows, &b_z, nz);
    // b laid out by z: each row's y rank and factor, contiguous.
    let b_y: Vec<u32> = b_rows.iter().map(|&r| y.rank[r as usize]).collect();
    let b_f: Vec<Option<f64>> = (b_rows.iter())
        .map(|&r| {
            b_factors
                .is_valid(r as usize)
                .then(|| b_factors.value(r as usize))
        })
        .collect();

    // The groups the product makes: no more than the joined pairs, nor than
    // every (x, y). Each takes 24 bytes, and then, without a domain, a row of
    // the output, 8 bytes a column; with one, the output is a column of 8
    // bytes per domain row, beside the domain's own columns.
    let pairs: usize = (a_rows.iter())
        .map(|&r| b_start[a_z(r) + 1] - b_start[a_z(r)])
        .fold(0, usize::saturating_add);
    let groups = pairs.min(nx.saturating_mul(ny));
    let output = match fold.domain() {
        None => groups.saturating_mul(8 * (fold.outputs().len() + 1)),
        Some(_) => nd.saturating_mul(8),
    };
    options.reserve(
        groups.saturating_mul(24).saturating_add(output),
        "the discrimination algorithm's groups",
    )?;
    // Gustavson's product, blocks of x on the threads in turn, each block's
    // groups in (x, y) order.
    let blocks = dense::par_tasks(nx.div_ceil(XB), threads, |t| {
        let mut acc = vec![0.0f64; ny];
        let mut valued = vec![false; ny];
        let mut reached = vec![false; ny];
        let mut touched: Vec<u32> = Vec::new();
        let mut groups = Vec::new();
        for xr in t * XB..((t + 1) * XB).min(nx) {
            for &r in &a_rows[a_start[xr]..a_start[xr + 1]] {
                let zr = a_z(r);
                let av = (a_factors.is_valid(r as usize)).then(|| a_factors.value(r as usize));
                for j in b_start[zr]..b_start[zr + 1] {
                    let yr = b_y[j] as usize;
                    if !std::mem::replace(&mut reached[yr], true) {
                        touched.push(yr as u32);
                    }
                    if let (Some(p), Some(q)) = (av, b_f[j]) {
                        acc[yr] += p * q;
                        valued[yr] = true;
                    }
                }
            }
            touched.sort_unstable();
            for &yr in &touched {
                let yr = yr as usize;
                groups.push(Group {
                    x: xr as u32,
                    y: yr as u32,
                    sum: acc[yr],
                    valued: valued[yr],
                });
                (acc[yr], valued[yr], reached[yr]) = (0.0, false, false);
            }
            touched.clear();
        }
        groups
    });
    let groups: Vec<Group> = blocks.concat();

    let Some(spec) = fold.domain() else {
        // Each group's keys, from a row of a and a row of b that have them.
        let a_take = UInt32Array::from_iter_values(groups.iter().map(|g| x.first[g.x as usize]));
        let b_take = UInt32Array::from_iter_values(groups.iter().map(|g| y.first[g.y as usize]));
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(fold.outputs().len() + 1);
        for i in 0..fold.outputs().len() {
            let column = match a_out.iter().position(|&o| o == i) {
                Some(j) => compute::take(&a_outputs[j], &a_take, None)?,
                None => {
                    let j = b_out.iter().position(|&o| o == i).expect("from a or b");
                    compute::take(&b_outputs[j], &b_take, None)?
                }
            };
            columns.push(column);
        }
        let values: Float64Array = groups.iter().map(|g| g.valued.then_some(g.sum)).collect();
        columns.push(Arc::new(values));
        return Ok(vec![RecordBatch::try_new(schema, columns)?]);
    };
    let Some(domain) = domain else {
        return Err(ArrowError::InvalidArgumentError(
            "a fold with a domain was given no domain relation".into(),
        ));
    };
    // Where each x's groups start; a domain row's group is found among them
    // by its y rank. Its keys were ranked with the operands', so equal keys
    // have equal ranks.
    let mut group_start = vec![0usize; nx + 1];
    for g in &groups {
        group_start[g.x as usize + 1] += 1;
    }
    for i in 0..nx {
        group_start[i + 1] += group_start[i];
    }
    let strict: Vec<bool> = (0..fold.outputs().len())
        .map(|i| domain_key(i).equality == KeyEquality::Equal)
        .collect();
    let mut offset = 0;
    let mut out = Vec::with_capacity(domain.len());
    for batch in domain {
        let n = batch.num_rows();
        let keys: Vec<ArrayRef> = (0..fold.outputs().len())
            .map(|i| Arc::clone(batch.column(domain_key(i).column.0)))
            .collect();
        let matches = joinable(&keys, &strict, n);
        let values: Float64Array = (0..n)
            .map(|r| {
                let (xr, yr) = (x.rank[na + offset + r] as usize, y.rank[nb + offset + r]);
                let found = (matches[r] && xr < nx).then(|| {
                    let candidates = &groups[group_start[xr]..group_start[xr + 1]];
                    let i = candidates.binary_search_by_key(&yr, |g| g.y).ok()?;
                    candidates[i].valued.then_some(candidates[i].sum)
                });
                found.flatten().or(spec.missing)
            })
            .collect();
        offset += n;
        let mut columns: Vec<ArrayRef> = (spec.carry.iter())
            .map(|&c| Arc::clone(batch.column(c.0)))
            .collect();
        columns.push(Arc::new(values));
        out.push(RecordBatch::try_new(Arc::clone(&schema), columns)?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{ArrayRef, StringArray};

    use super::*;

    /// Keys sharing a prefix 100,000 bytes long sort on a thread with a
    /// 256 KB stack, which a call per byte would overflow, and sort as their
    /// bytes do.
    #[test]
    fn sorts_keys_with_long_shared_prefixes_in_a_small_stack() {
        let prefix = "p".repeat(100_000);
        let strings: Vec<String> = (0..200).rev().map(|i| format!("{prefix}{i:03}")).collect();
        let column: ArrayRef = Arc::new(StringArray::from(strings.clone()));
        let keys = Keys::new(&[column]).unwrap();
        let sorted = thread::Builder::new()
            .stack_size(256 << 10)
            .spawn(move || {
                let mut index: Vec<u32> = (0..200).collect();
                by_bytes(&keys, &mut index, &mut Vec::new());
                index
            })
            .unwrap()
            .join()
            .unwrap();
        let got: Vec<&String> = sorted.iter().map(|&i| &strings[i as usize]).collect();
        let mut want: Vec<&String> = strings.iter().collect();
        want.sort();
        assert_eq!(got, want);
    }
}
