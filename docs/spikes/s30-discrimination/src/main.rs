// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Spike S30: a fold over a join, `SELECT a.x, b.y, SUM(a.v * b.v) FROM a
//! JOIN b ON a.z = b.z GROUP BY a.x, b.y`, computed by discrimination, against
//! einfold's kernels and DataFusion.
//!
//! Discrimination (Henglein, "Generic Discrimination", ICFP 2008) partitions
//! keys by their bytes, one radix pass per byte, with no comparisons and no
//! hashing. Here it gives each distinct key a rank, `0..k`, in the order of
//! its bytes. Keys are first put in Arrow's row format, byte strings that
//! compare as the keys do, so one discriminator serves every key type.
//! With ranks for `x`, `z` and `y`, the fold is Gustavson's sparse product
//! over rows bucketed by rank with counting sorts: for each `x`, each of its
//! `(z, v)` adds `v` times `b`'s row `z` into a dense accumulator over `y`.
//! The order of every addition is fixed by the keys' values, so the result
//! doesn't depend on how rows arrive.
//!
//! Each method runs from Arrow batches to Arrow batches:
//! - `discriminate`: this spike, on 12 threads where it parallelizes;
//! - `einfold`: `einfold_kernels::fold`, which runs the dense algorithm when
//!   keys are small non-negative integers and the hash algorithm otherwise;
//! - `datafusion`: the SQL, on 12 partitions.
//!
//! Run with `cargo run --release`.

use std::sync::Arc;
use std::thread;
use std::time::Instant;

use datafusion::arrow::array::{
    Array, ArrayRef, Float64Array, Int64Array, RecordBatch, StringArray, UInt32Array,
};
use datafusion::arrow::compute;
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::arrow::row::{RowConverter, Rows, SortField};
use datafusion::datasource::MemTable;
use datafusion::prelude::{SessionConfig, SessionContext};
use einfold_ir::{
    Aggregate, ColumnIndex, Dimension, DimensionId, Fold, Key, KeyEquality, Operand, Output,
    RowValue,
};
use einfold_kernels::{Algorithm, Options, Relation};

const THREADS: usize = 12;

// --- discrimination -----------------------------------------------------------

/// Sorts `index` by `keys[i][depth..]`, most significant byte first: one
/// counting pass per byte, recursing into each bucket. A key that ends sorts
/// before any that continues. Equal keys keep their order. Buckets of a few
/// rows are finished by comparison, as radix sorts usually are.
fn discriminate(keys: &Rows, index: &mut [u32], depth: usize, scratch: &mut Vec<u32>) {
    if index.len() <= 16 {
        index.sort_by(|&a, &b| keys.row(a as usize).data()[depth..].cmp(&keys.row(b as usize).data()[depth..]));
        return;
    }
    // Bucket 0: keys that end here; bucket 1 + b: keys whose byte is b.
    let mut count = [0usize; 258];
    let bucket = |i: u32| keys.row(i as usize).data().get(depth).map_or(0, |&b| b as usize + 1);
    for &i in index.iter() {
        count[bucket(i) + 1] += 1;
    }
    if count.iter().filter(|&&c| c > 0).count() == 1 && count[1] == 0 {
        // Every key has the same byte here: go on to the next.
        return discriminate(keys, index, depth + 1, scratch);
    }
    for b in 1..258 {
        count[b] += count[b - 1];
    }
    let starts = count;
    scratch.clear();
    scratch.resize(index.len(), 0);
    for &i in index.iter() {
        let b = bucket(i);
        scratch[count[b]] = i;
        count[b] += 1;
    }
    index.copy_from_slice(scratch);
    for b in 1..257 {
        let range = starts[b]..starts[b + 1];
        if range.len() > 1 {
            discriminate(keys, &mut index[range], depth + 1, scratch);
        }
    }
}

/// The keys as `u64`s that order as their bytes do, when every row has the
/// same width and at most 8 bytes differ between rows: the bytes all rows
/// share are dropped. Integer and float keys fit, after Arrow's row format
/// adds its null flag.
fn words(keys: &Rows) -> Option<Vec<u64>> {
    let n = keys.num_rows();
    let first = keys.row(0).data();
    let width = first.len();
    let mut varies = vec![false; width];
    for i in 1..n {
        let row = keys.row(i).data();
        if row.len() != width {
            return None;
        }
        for (v, (a, b)) in varies.iter_mut().zip(row.iter().zip(first)) {
            *v |= a != b;
        }
    }
    let bytes: Vec<usize> = (0..width).filter(|&b| varies[b]).collect();
    if bytes.len() > 8 {
        return None;
    }
    Some(
        (0..n)
            .map(|i| {
                let row = keys.row(i).data();
                bytes.iter().fold(0u64, |w, &b| (w << 8) | row[b] as u64)
            })
            .collect(),
    )
}

/// `(word, row)` for every row, sorted by word, stably: a
/// least-significant-byte-first radix sort over the pairs themselves, so
/// that each pass reads them in order. Each pass counts by chunks of rows
/// in parallel, and a pass over a byte every word shares is skipped.
fn radix(words: &[u64]) -> Vec<(u64, u32)> {
    let n = words.len();
    let chunks = THREADS.min(n.div_ceil(65536)).max(1);
    let bounds: Vec<(usize, usize)> = (0..chunks).map(|c| (n * c / chunks, n * (c + 1) / chunks)).collect();
    let mut pairs: Vec<(u64, u32)> = words.iter().enumerate().map(|(i, &w)| (w, i as u32)).collect();
    let mut next = vec![(0u64, 0u32); n];
    // Which bytes vary at all.
    let (or, and) = words.iter().fold((0u64, u64::MAX), |(o, a), &w| (o | w, a & w));
    let varies = or ^ and;
    for pass in 0..8 {
        let shift = pass * 8;
        if (varies >> shift) & 0xff == 0 {
            continue;
        }
        let byte = |w: u64| ((w >> shift) & 0xff) as usize;
        let counts: Vec<[usize; 256]> = thread::scope(|s| {
            let handles: Vec<_> = (bounds.iter())
                .map(|&(lo, hi)| {
                    let pairs = &pairs;
                    s.spawn(move || {
                        let mut c = [0usize; 256];
                        for &(w, _) in &pairs[lo..hi] {
                            c[byte(w)] += 1;
                        }
                        c
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        // The slots, byte-major then chunk, are contiguous: split them so
        // that each chunk owns its 256 regions, and scatter in parallel.
        let mut regions: Vec<Vec<&mut [(u64, u32)]>> = (0..chunks).map(|_| Vec::with_capacity(256)).collect();
        let mut rest = &mut next[..];
        for b in 0..256 {
            for (c, region) in regions.iter_mut().enumerate() {
                let (head, tail) = rest.split_at_mut(counts[c][b]);
                region.push(head);
                rest = tail;
            }
        }
        thread::scope(|s| {
            for (&(lo, hi), mut region) in bounds.iter().zip(regions) {
                let pairs = &pairs;
                s.spawn(move || {
                    let mut at = [0usize; 256];
                    for &p in &pairs[lo..hi] {
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

/// Each row's rank among the distinct keys of `keys`, in byte order, and for
/// each rank, one row that has it.
fn ranks(keys: &Rows) -> (Vec<u32>, Vec<u32>) {
    if keys.num_rows() > 0 {
        if let Some(words) = words(keys) {
            return ranks_of_words(&words);
        }
    }
    let n = keys.num_rows();
    let mut index: Vec<u32> = (0..n as u32).collect();
    // The first byte, in parallel buckets; then each bucket on its own.
    let first = |i: u32| keys.row(i as usize).data().first().map_or(0, |&b| b as usize + 1);
    let mut count = [0usize; 258];
    for &i in &index {
        count[first(i) + 1] += 1;
    }
    for b in 1..258 {
        count[b] += count[b - 1];
    }
    let starts = count;
    let mut placed = vec![0u32; n];
    for &i in &index {
        let b = first(i);
        placed[count[b]] = i;
        count[b] += 1;
    }
    index = placed;
    let mut buckets: Vec<&mut [u32]> = Vec::new();
    let mut rest = &mut index[..];
    for b in 0..257 {
        let (head, tail) = rest.split_at_mut(starts[b + 1] - starts[b]);
        buckets.push(head);
        rest = tail;
    }
    // The largest buckets first, dealt to threads in turn.
    buckets.sort_by_key(|b| std::cmp::Reverse(b.len()));
    let mut per_thread: Vec<Vec<&mut [u32]>> = (0..THREADS).map(|_| Vec::new()).collect();
    for (i, b) in buckets.into_iter().enumerate() {
        per_thread[i % THREADS].push(b);
    }
    thread::scope(|s| {
        for work in per_thread {
            s.spawn(move || {
                let mut scratch = Vec::new();
                for b in work {
                    if b.len() > 1 {
                        discriminate(keys, b, 1, &mut scratch);
                    }
                }
            });
        }
    });
    let mut rank = vec![0u32; n];
    let mut representative = Vec::new();
    let mut previous: Option<&[u8]> = None;
    for &i in &index {
        let key = keys.row(i as usize).data();
        if previous != Some(key) {
            representative.push(i);
            previous = Some(key);
        }
        rank[i as usize] = representative.len() as u32 - 1;
    }
    (rank, representative)
}

/// An integer column's keys as `u64`s that order as the integers do, with no
/// NULLs: the row format isn't needed for them.
fn integer_words(column: &ArrayRef) -> Option<Vec<u64>> {
    let c = column.as_any().downcast_ref::<Int64Array>()?;
    (c.null_count() == 0).then(|| c.values().iter().map(|&k| (k as u64) ^ (1 << 63)).collect())
}

/// [`ranks`] of one column.
fn column_ranks(column: &ArrayRef) -> (Vec<u32>, Vec<u32>) {
    match integer_words(column) {
        Some(words) => ranks_of_words(&words),
        None => ranks(&rows_of(&[Arc::clone(column)])),
    }
}

/// Each word's rank among the distinct words, in order, and for each rank,
/// one row that has it.
fn ranks_of_words(words: &[u64]) -> (Vec<u32>, Vec<u32>) {
    let sorted = radix(words);
    let mut rank = vec![0u32; sorted.len()];
    let mut representative = Vec::new();
    let mut previous = None;
    for &(w, i) in &sorted {
        if previous != Some(w) {
            representative.push(i);
            previous = Some(w);
        }
        rank[i as usize] = representative.len() as u32 - 1;
    }
    (rank, representative)
}

fn rows_of(columns: &[ArrayRef]) -> Rows {
    let fields = columns.iter().map(|c| SortField::new(c.data_type().clone())).collect();
    let converter = RowConverter::new(fields).unwrap();
    converter.convert_columns(columns).unwrap()
}

/// The positions `0..n` ordered by `key` (all below `k`), stably: a counting
/// sort. Returns them with each key's start.
fn counting(key: &[u32], order: &[u32], k: usize) -> (Vec<u32>, Vec<usize>) {
    let mut start = vec![0usize; k + 1];
    for &i in order {
        start[key[i as usize] as usize + 1] += 1;
    }
    for r in 0..k {
        start[r + 1] += start[r];
    }
    let mut next = start.clone();
    let mut out = vec![0u32; order.len()];
    for &i in order {
        let r = key[i as usize] as usize;
        out[next[r]] = i;
        next[r] += 1;
    }
    (out, start)
}

/// A table as one batch: `(key, key, v)`.
struct Table {
    schema: SchemaRef,
    batch: RecordBatch,
}

/// The fold by discrimination: ranks for x, z (over both tables at once) and
/// y, then Gustavson's product over rows bucketed by rank.
fn by_discrimination(a: &Table, b: &Table) -> RecordBatch {
    let (ax, az, av) = (a.batch.column(0), a.batch.column(1), a.batch.column(2));
    let (bz, by, bv) = (b.batch.column(0), b.batch.column(1), b.batch.column(2));
    let (na, nb) = (ax.len(), bz.len());
    let clock = Instant::now();
    let lap = |what: &str| {
        if std::env::var("S30_PHASES").is_ok() {
            eprintln!("  {what}: {:.1} ms", clock.elapsed().as_secs_f64() * 1e3);
        }
    };
    // The three discriminations are independent.
    let z_all = compute::concat(&[az.as_ref(), bz.as_ref()]).unwrap();
    if std::env::var("S30_PHASES").is_ok() {
        let t = Instant::now();
        let rows = rows_of(&[Arc::clone(&z_all)]);
        let conv = t.elapsed().as_secs_f64() * 1e3;
        let _ = ranks(&rows);
        eprintln!("  z alone: rows {conv:.1} ms, then ranks {:.1} ms", t.elapsed().as_secs_f64() * 1e3 - conv);
    }
    let ((x_rank, x_rep), (z_rank, z_rep), (y_rank, y_rep)) = thread::scope(|s| {
        let x = s.spawn(|| column_ranks(ax));
        let y = s.spawn(|| column_ranks(by));
        let z = column_ranks(&z_all);
        (x.join().unwrap(), z, y.join().unwrap())
    });
    lap("ranks");
    let (nx, nz, ny) = (x_rep.len(), z_rep.len(), y_rep.len());
    let (a_z, b_z) = z_rank.split_at(na);
    // a's rows in (x, z) order, b's in (z, y) order: two stable counting
    // sorts each, least significant key first.
    let all = |n: usize| (0..n as u32).collect::<Vec<u32>>();
    let (by_z, _) = counting(a_z, &all(na), nz);
    let (a_order, a_start) = counting(&x_rank, &by_z, nx);
    let (by_y, _) = counting(&y_rank, &all(nb), ny);
    let (b_order, b_start) = counting(b_z, &by_y, nz);
    lap("counting sorts");
    let av = av.as_any().downcast_ref::<Float64Array>().unwrap();
    let bv = bv.as_any().downcast_ref::<Float64Array>().unwrap();
    // b's rows, laid out by z: their y ranks and values, contiguous.
    let b_y: Vec<u32> = b_order.iter().map(|&i| y_rank[i as usize]).collect();
    let b_val: Vec<Option<f64>> = b_order.iter().map(|&i| bv.is_valid(i as usize).then(|| bv.value(i as usize))).collect();
    // Gustavson's product, for blocks of x on the threads in turn; each
    // block's rows come back in x order.
    let blocks = nx.div_ceil(256).max(1);
    let work: Vec<usize> = (0..blocks).collect();
    let results: Vec<Vec<(u32, u32, Option<f64>)>> = {
        let mut slots: Vec<Option<Vec<(u32, u32, Option<f64>)>>> = (0..blocks).map(|_| None).collect();
        thread::scope(|s| {
            let handles: Vec<_> = (0..THREADS.min(blocks))
                .map(|t| {
                    let (a_order, a_start, b_start, b_y, b_val, a_z, work) =
                        (&a_order, &a_start, &b_start, &b_y, &b_val, a_z, &work);
                    s.spawn(move || {
                        let mut acc = vec![0.0f64; ny];
                        let mut valued = vec![false; ny];
                        let mut seen = vec![false; ny];
                        let mut touched: Vec<u32> = Vec::new();
                        let mut done = Vec::new();
                        for &blk in work.iter().skip(t).step_by(THREADS) {
                            let mut out = Vec::new();
                            for x in blk * 256..((blk + 1) * 256).min(nx) {
                                for &i in &a_order[a_start[x]..a_start[x + 1]] {
                                    let z = a_z[i as usize] as usize;
                                    let va = av.is_valid(i as usize).then(|| av.value(i as usize));
                                    for j in b_start[z]..b_start[z + 1] {
                                        let y = b_y[j] as usize;
                                        if !std::mem::replace(&mut seen[y], true) {
                                            touched.push(y as u32);
                                        }
                                        if let (Some(p), Some(q)) = (va, b_val[j]) {
                                            acc[y] += p * q;
                                            valued[y] = true;
                                        }
                                    }
                                }
                                touched.sort_unstable();
                                for &y in &touched {
                                    let y = y as usize;
                                    out.push((x as u32, y as u32, valued[y].then_some(acc[y])));
                                    (acc[y], valued[y], seen[y]) = (0.0, false, false);
                                }
                                touched.clear();
                            }
                            done.push((blk, out));
                        }
                        done
                    })
                })
                .collect();
            for h in handles {
                for (blk, out) in h.join().unwrap() {
                    slots[blk] = Some(out);
                }
            }
        });
        slots.into_iter().map(|s| s.unwrap()).collect()
    };
    lap("gustavson");
    let rows: Vec<(u32, u32, Option<f64>)> = results.concat();
    // The output's keys: each rank's representative row's key.
    let x_rows = UInt32Array::from_iter_values(rows.iter().map(|r| x_rep[r.0 as usize]));
    let y_rows = UInt32Array::from_iter_values(rows.iter().map(|r| y_rep[r.1 as usize]));
    let x_keys = compute::take(ax.as_ref(), &x_rows, None).unwrap();
    let y_keys = compute::take(by.as_ref(), &y_rows, None).unwrap();
    let value = match rows.iter().all(|r| r.2.is_some()) {
        true => Float64Array::from(rows.iter().map(|r| r.2.unwrap_or(0.0)).collect::<Vec<f64>>()),
        false => rows.iter().map(|r| r.2).collect(),
    };
    let schema = Arc::new(Schema::new(vec![
        Field::new("x", ax.data_type().clone(), true),
        Field::new("y", by.data_type().clone(), true),
        Field::new("v", DataType::Float64, true),
    ]));
    let _ = z_rep;
    lap("output");
    RecordBatch::try_new(schema, vec![x_keys, y_keys, Arc::new(value)]).unwrap()
}

// --- the other methods ------------------------------------------------------------

fn by_einfold(a: &Table, b: &Table) -> (RecordBatch, Algorithm) {
    const EQ: Dimension = Dimension { equality: KeyEquality::Equal };
    let key = |d, c| Key::new(DimensionId(d), ColumnIndex(c));
    let out = |d| Output { dimension: DimensionId(d), domain_key: None };
    let fold = Fold::try_new(
        vec![EQ; 3],
        vec![
            Operand { keys: vec![key(0, 0), key(1, 1)], factor: Some(ColumnIndex(2)) },
            Operand { keys: vec![key(1, 0), key(2, 1)], factor: Some(ColumnIndex(2)) },
        ],
        vec![out(0), out(2)],
        RowValue::Product,
        Aggregate::sum(),
        None,
    )
    .unwrap();
    let schema = einfold_kernels::schema(&fold, &[&a.schema, &b.schema], None).unwrap();
    let (ab, bb) = ([a.batch.clone()], [b.batch.clone()]);
    let options = Options { threads: THREADS, ..Options::default() };
    let folded = einfold_kernels::fold(
        &fold,
        Relation::new(&a.schema, &ab),
        Relation::new(&b.schema, &bb),
        None,
        schema.clone(),
        options,
    )
    .unwrap();
    let batch = compute::concat_batches(&schema, &folded.batches).unwrap();
    (batch, folded.algorithm)
}

async fn by_datafusion(ctx: &SessionContext) -> RecordBatch {
    let sql = "SELECT a.x, b.y, SUM(a.v * b.v) AS v FROM a JOIN b ON a.z = b.z GROUP BY a.x, b.y";
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    compute::concat_batches(&batches[0].schema(), &batches).unwrap()
}

// --- workloads ------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
enum Keys {
    /// 0..k: einfold's dense algorithm takes these.
    Dense,
    /// Integers spread over 0..2⁴⁰: the dense algorithm declines.
    Sparse,
    /// Text, `k…`.
    Text,
}

fn column(keys: Keys, values: &[u64]) -> ArrayRef {
    match keys {
        Keys::Dense => Arc::new(Int64Array::from_iter_values(values.iter().map(|&k| k as i64))),
        Keys::Sparse => Arc::new(Int64Array::from_iter_values(
            values.iter().map(|&k| (k.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 24) as i64),
        )),
        Keys::Text => Arc::new(StringArray::from_iter_values(values.iter().map(|k| format!("k{k:07}")))),
    }
}

fn table(keys: Keys, rows: &[(u64, u64)], salt: f64) -> Table {
    let first: Vec<u64> = rows.iter().map(|r| r.0).collect();
    let second: Vec<u64> = rows.iter().map(|r| r.1).collect();
    let v = Float64Array::from_iter_values((0..rows.len()).map(|i| ((i as f64) * 0.7 + salt).sin()));
    let (c1, c2) = (column(keys, &first), column(keys, &second));
    let schema = Arc::new(Schema::new(vec![
        Field::new("k1", c1.data_type().clone(), false),
        Field::new("k2", c2.data_type().clone(), false),
        Field::new("v", DataType::Float64, false),
    ]));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![c1, c2, Arc::new(v)]).unwrap();
    Table { schema, batch }
}

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: u64) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0 % n
    }
}

/// ddx's matrix product, `n × 16` by `16 × 8`; and a sparse product, `n × n`
/// by `n × n` with 8 entries per row of each.
fn workload(name: &str, n: usize, keys: Keys) -> (Table, Table) {
    match name {
        "matmul" => {
            let a: Vec<(u64, u64)> = (0..n as u64).flat_map(|s| (0..16).map(move |k| (s, k))).collect();
            let b: Vec<(u64, u64)> = (0..16).flat_map(|k| (0..8).map(move |o| (k, o))).collect();
            (table(keys, &a, 0.0), table(keys, &b, 0.5))
        }
        _ => {
            let mut rng = Rng(0x5eed_0000_0001 + n as u64);
            let entries = |rng: &mut Rng| -> Vec<(u64, u64)> {
                let mut e: Vec<(u64, u64)> = (0..n as u64)
                    .flat_map(|r| (0..8).map(move |_| r))
                    .map(|r| (r, rng.below(n as u64)))
                    .collect();
                e.sort_unstable();
                e.dedup();
                e
            };
            let (a, b) = (entries(&mut rng), entries(&mut rng));
            (table(keys, &a, 0.0), table(keys, &b, 0.5))
        }
    }
}

/// The fastest of five runs, in milliseconds, and the last result.
fn best<T>(mut run: impl FnMut() -> T) -> (f64, T) {
    let mut best = f64::MAX;
    let mut out = None;
    for _ in 0..5 {
        let t = Instant::now();
        out = Some(run());
        best = best.min(t.elapsed().as_secs_f64() * 1e3);
    }
    (best, out.unwrap())
}

/// Rows as sortable strings of keys, with their values.
fn rows(batch: &RecordBatch) -> Vec<(String, Option<f64>)> {
    use datafusion::arrow::util::display::array_value_to_string;
    let v = batch.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
    let mut out: Vec<(String, Option<f64>)> = (0..batch.num_rows())
        .map(|r| {
            let k = format!(
                "{}|{}",
                array_value_to_string(batch.column(0), r).unwrap(),
                array_value_to_string(batch.column(1), r).unwrap()
            );
            (k, v.is_valid(r).then(|| v.value(r)))
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn same(a: &RecordBatch, b: &RecordBatch) -> bool {
    let (a, b) = (rows(a), rows(b));
    a.len() == b.len()
        && a.iter().zip(&b).all(|((ka, va), (kb, vb))| {
            ka == kb
                && match (va, vb) {
                    (Some(x), Some(y)) => (x - y).abs() <= 1e-9 * (1.0 + y.abs()),
                    (None, None) => true,
                    _ => false,
                }
        })
}

#[tokio::main]
async fn main() {
    println!("| workload | keys | n | out rows | discriminate, ms | einfold, ms (algorithm) | DataFusion, ms |");
    println!("|---|---|---:|---:|---:|---:|---:|");
    for name in ["matmul", "sparse"] {
        for keys in [Keys::Dense, Keys::Sparse, Keys::Text] {
            for n in [10_000, 50_000] {
                if let Ok(only) = std::env::var("S30_ONLY") {
                    if only != format!("{name},{keys:?},{n}") {
                        continue;
                    }
                }
                let (a, b) = workload(name, n, keys);
                let ctx = SessionContext::new_with_config(SessionConfig::new().with_target_partitions(THREADS));
                for (t, name) in [(&a, "a"), (&b, "b")] {
                    let renamed = Arc::new(Schema::new(vec![
                        Field::new(if name == "a" { "x" } else { "z" }, t.schema.field(0).data_type().clone(), false),
                        Field::new(if name == "a" { "z" } else { "y" }, t.schema.field(1).data_type().clone(), false),
                        Field::new("v", DataType::Float64, false),
                    ]));
                    let batch = RecordBatch::try_new(Arc::clone(&renamed), t.batch.columns().to_vec()).unwrap();
                    let parts: Vec<Vec<RecordBatch>> = (0..THREADS)
                        .map(|p| {
                            let rows = batch.num_rows();
                            let (s, e) = (rows * p / THREADS, rows * (p + 1) / THREADS);
                            vec![batch.slice(s, e - s)]
                        })
                        .collect();
                    ctx.register_table(name, Arc::new(MemTable::try_new(renamed, parts).unwrap())).unwrap();
                }
                let (td, ours) = best(|| by_discrimination(&a, &b));
                let (te, (theirs, algorithm)) = best(|| by_einfold(&a, &b));
                let mut tf = f64::MAX;
                let mut host = None;
                for _ in 0..5 {
                    let t = Instant::now();
                    host = Some(by_datafusion(&ctx).await);
                    tf = tf.min(t.elapsed().as_secs_f64() * 1e3);
                }
                let host = host.unwrap();
                assert!(same(&ours, &host), "{name} {keys:?} {n}: discrimination differs from DataFusion");
                assert!(same(&theirs, &host), "{name} {keys:?} {n}: einfold differs from DataFusion");
                println!(
                    "| {name} | {keys:?} | {n} | {} | {td:.1} | {te:.1} ({algorithm:?}) | {tf:.1} |",
                    host.num_rows()
                );
            }
        }
    }
}
