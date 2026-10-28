// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! EinFold's dense algorithm: a two-operand fold computed by positions, with
//! no hashing, when the keys are small non-negative integers.
//!
//! Every dimension must have one of four roles:
//!
//! - **W**: an output held by both operands, such as attention's batch and
//!   head: in SQL, a join key that is also grouped by;
//! - **X**: an output held only by the first operand, `a`;
//! - **Z**: a summed dimension held by both operands (the join's other keys);
//! - **Y**: an output held only by the second operand, `b`.
//!
//! That is a batch of matrix products `out[w, x, y] = Σ_z a[w, x, z]·b[w, z, y]`,
//! with each role's dimensions flattened into one position. Below, a *row*
//! of `a` or of the output is a position of W and X together, `(w, x)`, W
//! slowest. Keys are used as positions directly: key `k` of a dimension is
//! position `k`, and a dimension's extent is its largest key plus one. Both operands are placed in dense matrices,
//! batch by batch, as they arrive, and each position of `a` adds its factor
//! times a row of `b` into a row of the output.
//!
//! **When SQL and dense arrays agree.** A dense array has a value at every
//! position, where SQL has rows only where they exist:
//!
//! - Each operand must have one row per key tuple, so that a position holds
//!   one row. Duplicate tuples, which SQL sums row by row, use the
//!   discrimination algorithm.
//! - Keys must be non-NULL: a dense array has no position for NULL.
//! - A missing or NULL row of `b` is held as 0. That is right for a sum,
//!   until a NaN or an infinity in `a` meets it: `0·∞` is NaN, but SQL forms no
//!   product for a missing row and skips a NULL one. So holes in `b` are
//!   allowed only when every non-NULL factor of `a` is finite.
//! - Which groups exist doesn't matter with a domain whose default is `0.0`:
//!   an unreached group and one whose products were all NULL both give 0, as
//!   `COALESCE(SUM(…), 0.0)` does. Otherwise `b` must be complete, with every
//!   position present and non-NULL. Then a group `(x, y)` exists exactly when
//!   `a` has a row at `x`, and its sum is NULL exactly when every such row's
//!   factor is NULL.
//!
//! Otherwise the caller falls back to the discrimination algorithm.
//!
//! **Masks.** A fold's conditions may compare two of its outputs, such as
//! causal attention's scores' `q.t >= k.t`. Each decides an output group as
//! a whole, so the groups a condition disallows are computed and dropped,
//! and never shown. A condition on a summed dimension declines.
//!
//! **Determinism.** Work is split into fixed blocks chosen from the shape
//! alone: blocks of output rows and columns when the output is large, or
//! blocks along `z` when it is small, whose partial outputs are added in
//! block order. Within a block, rows are visited in position order. So the
//! bits depend neither on the number of threads nor on the order in which
//! rows arrive.

use std::sync::atomic::{self, AtomicBool};
use std::sync::Arc;
use std::thread;

use arrow::array::{Array, ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch};
use arrow::compute;
use arrow::datatypes::{DataType, SchemaRef};
use einfold_ir::{ColumnIndex, DimensionId, Domain, Fold, Operand};

use crate::options::DomainRows;
use crate::Result;

/// Output rows per block, when the output has at least [`FEW_CELLS`] cells.
const XB: usize = 256;
/// Output columns per block, likewise, so that an output of few rows still
/// splits into blocks.
const YB: usize = 2048;
/// Positions of `z` per block, when the output is smaller.
const ZB: usize = 4096;
const FEW_CELLS: usize = 4096;
/// Input rows below which a fold runs on one thread: starting threads costs
/// tens of microseconds each, about what folding this many rows does.
const FEW_ROWS: usize = 1 << 16;
/// What a position of `a` holds: no row, a row with a NULL factor, or a row
/// with a value.
const ABSENT: u8 = 0;
const NULL: u8 = 1;
const VALUED: u8 = 2;

/// Whether a dense array of `cells` is worth building for `rows` rows: no
/// more than four cells per row, plus some slack for small inputs. Measured
/// on CPU, a dense product beats a hash-based one once about a fifth of its
/// cells hold rows, and one row in four cells is past that.
fn worth(cells: usize, rows: usize) -> bool {
    cells <= rows.saturating_mul(4).saturating_add(FEW_CELLS) && cells < u32::MAX as usize
}

/// `column` as non-negative `i64` keys, with their extent (the largest key
/// plus one), or `None` if it isn't an integer column, or holds a NULL, a
/// negative key, or one too large for `i64` (an unsigned key of 2⁶³ or more),
/// or whose extent doesn't fit (a key of `i64::MAX`).
pub(crate) fn keys(column: &ArrayRef) -> Option<(Int64Array, usize)> {
    if !column.data_type().is_integer() || column.null_count() > 0 {
        return None;
    }
    // The cast turns a key too large for i64 into NULL, which reads as 0.
    let keys = compute::cast(column, &DataType::Int64).ok()?;
    let keys = keys.as_any().downcast_ref::<Int64Array>()?.clone();
    // One pass for both bounds, which the compiler vectorizes.
    let (lo, hi) = (keys.values().iter()).fold((0, -1), |(lo, hi), &k| (k.min(lo), k.max(hi)));
    if keys.null_count() > 0 || lo < 0 {
        return None;
    }
    let extent = usize::try_from(hi.checked_add(1)?).ok()?;
    Some((keys, extent))
}

/// Each row's position over `dims`, flattened with the first dimension
/// slowest, from a batch's `(dimension, keys)` columns. It is computed a
/// column at a time, in 32 bits, which the compiler vectorizes: every
/// position fits, as the dense algorithm runs only on fewer cells than
/// `u32::MAX`.
pub(crate) fn positions(
    keys: &[(usize, Int64Array)],
    dims: &[usize],
    extent: &[usize],
    rows: usize,
) -> Vec<u32> {
    let mut position = vec![0u32; rows];
    for &d in dims {
        let (_, k) = keys
            .iter()
            .find(|(e, _)| *e == d)
            .expect("a held dimension");
        let n = extent[d] as u32;
        for (p, &k) in position.iter_mut().zip(k.values()) {
            *p = *p * n + k as u32;
        }
    }
    position
}

/// Run `f(i)` for every task `i` in `0..tasks` on up to `threads` threads.
/// Results come back in task order, so how tasks were assigned to threads
/// never shows in them. One thread's tasks run on the calling thread:
/// starting a thread costs more than a small fold.
pub(crate) fn par_tasks<T: Send>(
    tasks: usize,
    threads: usize,
    f: impl Fn(usize) -> T + Sync,
) -> Vec<T> {
    let threads = threads.clamp(1, tasks.max(1));
    if threads == 1 {
        return (0..tasks).map(f).collect();
    }
    let f = &f;
    let mut slots: Vec<Option<T>> = (0..tasks).map(|_| None).collect();
    thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| s.spawn(move || (t..tasks).step_by(threads).map(|i| (i, f(i))).collect()))
            .collect();
        for h in handles {
            let done: Vec<(usize, T)> = h.join().expect("a dense fold task panicked");
            for (i, r) in done {
                slots[i] = Some(r);
            }
        }
    });
    slots
        .into_iter()
        .map(|s| s.expect("every task ran"))
        .collect()
}

/// The fold of `a` and `b`'s batches (and `domain`'s, if the fold has one)
/// by the dense algorithm on `threads` threads, or `None` when its requirements
/// don't hold and the discrimination algorithm must be used instead. With a domain, there is
/// one output batch per domain batch. With [`DomainRows::UniqueKeys`], it is
/// also `None` when two of the domain's rows locate the same cell of the
/// output.
pub(crate) fn fold(
    fold: &Fold,
    a: &[RecordBatch],
    b: &[RecordBatch],
    domain: Option<&[RecordBatch]>,
    schema: SchemaRef,
    threads: usize,
    domain_rows: DomainRows,
) -> Result<Option<Vec<RecordBatch>>> {
    let [op_a, op_b] = fold.operands() else {
        return Ok(None);
    };
    let inputs: usize = (a.iter().chain(b).chain(domain.unwrap_or_default()))
        .map(RecordBatch::num_rows)
        .sum();
    let threads = if inputs < FEW_ROWS { 1 } else { threads };
    // The roles: W, X and Y in output order, Z in dimension order. Here a
    // dimension is its index, which indexes the extents too.
    let (mut ws, mut xs, mut ys, mut zs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let output_dimensions = fold.outputs().iter().map(|o| o.dimension);
    for d in output_dimensions {
        match (op_a.column_of(d), op_b.column_of(d)) {
            (Some(_), Some(_)) => ws.push(d.0),
            (Some(_), None) => xs.push(d.0),
            (None, Some(_)) => ys.push(d.0),
            (None, None) => return Ok(None),
        }
    }
    // From here on, `xs` are the dimensions of a row: W's, then X's. Each
    // operand's rows hold the batch's positions first, so that a batch's
    // matrices are contiguous.
    xs.splice(0..0, ws.iter().copied());
    for d in fold.dimension_ids() {
        if fold.is_aggregated(d) {
            if op_a.column_of(d).is_none() || op_b.column_of(d).is_none() {
                return Ok(None);
            }
            zs.push(d.0);
        }
    }
    // A condition between two outputs decides each output group as a whole:
    // every joined pair of a group has the group's keys. So the groups it
    // disallows are computed with the others, and dropped. A condition on a
    // summed dimension would mask pairs within a group, which this
    // algorithm doesn't do.
    let is_output = |d: DimensionId| xs.contains(&d.0) || ys.contains(&d.0);
    if !(fold.conditions().iter()).all(|c| is_output(c.left) && is_output(c.right)) {
        return Ok(None);
    }
    // Each batch's key columns as non-negative i64, cast once, with their
    // extents, batches in parallel; then each dimension's extent, its largest
    // key in either operand, plus one.
    let cast_keys = |op: &Operand, batches: &[RecordBatch]| {
        let batch_keys = par_tasks(batches.len(), threads, |i| {
            (op.keys.iter())
                .map(|key| {
                    let (DimensionId(d), ColumnIndex(c)) = (key.dimension, key.column);
                    let (k, n) = keys(batches[i].column(c))?;
                    Some((d, k, n))
                })
                .collect::<Option<Vec<_>>>()
        });
        batch_keys.into_iter().collect::<Option<Vec<_>>>()
    };
    let (Some(a_keys), Some(b_keys)) = (cast_keys(op_a, a), cast_keys(op_b, b)) else {
        return Ok(None);
    };
    let mut extent = vec![0usize; fold.dimensions().len()];
    for &(d, _, n) in a_keys.iter().chain(&b_keys).flatten() {
        extent[d] = extent[d].max(n);
    }
    let strip =
        |batch_keys: Vec<Vec<(usize, Int64Array, usize)>>| -> Vec<Vec<(usize, Int64Array)>> {
            (batch_keys.into_iter())
                .map(|keys| keys.into_iter().map(|(d, k, _)| (d, k)).collect())
                .collect()
        };
    let (a_keys, b_keys) = (strip(a_keys), strip(b_keys));
    let rows = |batches: &[RecordBatch]| batches.iter().map(|b| b.num_rows()).sum::<usize>();
    let size = |dims: &[usize]| {
        dims.iter()
            .try_fold(1usize, |n, &d| n.checked_mul(extent[d]))
    };
    // b's rows are positions of W and Z together, W slowest.
    let wzs: Vec<usize> = ws.iter().chain(&zs).copied().collect();
    let (Some(nw), Some(nx), Some(nz), Some(ny)) = (size(&ws), size(&xs), size(&zs), size(&ys))
    else {
        return Ok(None);
    };
    // Rows per batch: an output row `r` is in batch `r / per_batch`.
    let per_batch = nx.checked_div(nw).unwrap_or(0);
    let out_rows = domain.map_or(nx.saturating_mul(ny), |f| {
        f.iter().map(|b| b.num_rows()).sum()
    });
    if !worth(nx.saturating_mul(nz), rows(a))
        || !worth(nw.saturating_mul(nz).saturating_mul(ny), rows(b))
        || !worth(nx.saturating_mul(ny), out_rows)
    {
        return Ok(None);
    }
    let factors = |op: &Operand, batch: &'_ RecordBatch| {
        op.factor
            .and_then(|c| batch.column(c.0).as_any().downcast_ref::<Float64Array>())
            .cloned()
    };

    // Whether existence can be ignored: a domain whose default is +0.0.
    let zero_default = fold
        .domain()
        .is_some_and(|d| d.missing.map(f64::to_bits) == Some(0.0f64.to_bits()));
    // In one pass over a: a dense matrix [x][z] of its factors, and what each
    // position holds, where a repeated position means duplicate tuples; which
    // x positions it reaches; whether each has a non-NULL factor,
    // `valued[x]`, which existence needs only without the 0.0 default; and
    // whether every non-NULL factor is finite.
    let mut a_matrix = vec![0.0; nx * nz];
    let mut held = vec![ABSENT; nx * nz];
    let mut reached = vec![false; nx];
    let mut valued = vec![false; if zero_default { 0 } else { nx }];
    let mut finite = true;
    // Each row's x and z positions, batches in parallel; then the rows
    // placed in order.
    let a_positions = par_tasks(a.len(), threads, |i| {
        let rows = a[i].num_rows();
        let x = positions(&a_keys[i], &xs, &extent, rows);
        let z = positions(&a_keys[i], &zs, &extent, rows);
        (x, z)
    });
    for (batch, (a_x, a_z)) in a.iter().zip(&a_positions) {
        let Some(a_factors) = factors(op_a, batch) else {
            return Ok(None);
        };
        for r in 0..batch.num_rows() {
            let x = a_x[r] as usize;
            let p = x * nz + a_z[r] as usize;
            let valid = a_factors.is_valid(r);
            if std::mem::replace(&mut held[p], if valid { VALUED } else { NULL }) != ABSENT {
                return Ok(None);
            }
            reached[x] = true;
            if valid {
                a_matrix[p] = a_factors.value(r);
                finite &= a_matrix[p].is_finite();
            }
            if !zero_default {
                valued[x] |= valid;
            }
        }
    }
    // b as dense matrices [w][z][y], with holes (missing or NULL rows) as 0.
    let mut b_matrix = vec![0.0; nw * nz * ny];
    let mut present = vec![false; nw * nz * ny];
    let mut b_nulls = false;
    let b_positions = par_tasks(b.len(), threads, |i| {
        let rows = b[i].num_rows();
        let z = positions(&b_keys[i], &wzs, &extent, rows);
        let y = positions(&b_keys[i], &ys, &extent, rows);
        (z, y)
    });
    for (batch, (b_z, b_y)) in b.iter().zip(&b_positions) {
        let Some(b_factors) = factors(op_b, batch) else {
            return Ok(None);
        };
        b_nulls |= b_factors.null_count() > 0;
        for r in 0..batch.num_rows() {
            let p = b_z[r] as usize * ny + b_y[r] as usize;
            if std::mem::replace(&mut present[p], true) {
                return Ok(None);
            }
            if b_factors.is_valid(r) {
                b_matrix[p] = b_factors.value(r);
            }
        }
    }
    let holes = b_nulls || present.iter().any(|&p| !p);
    if holes && (!zero_default || !finite) {
        return Ok(None);
    }
    // Otherwise b is complete, so group (x, y) exists when a has a row at x,
    // and is NULL when all of that row's factors are: when `valued[x]` is
    // false.

    let out = accumulate(
        &a_matrix,
        &held,
        &b_matrix,
        (nx, nz, ny),
        per_batch,
        threads,
    );
    // A dimension's key at an output cell (x, y): its place in x's or y's
    // position, first dimension slowest. Keys are positions.
    let key_at = |d: usize, x: usize, y: usize| -> usize {
        let (role, p) = match xs.contains(&d) {
            true => (&xs, x),
            false => (&ys, y),
        };
        let slot = role.iter().position(|&e| e == d).expect("an output");
        let stride: usize = role[slot + 1..].iter().map(|&e| extent[e]).product();
        (p / stride) % extent[d]
    };
    // Whether the fold's conditions allow cell (x, y): a group they disallow
    // doesn't exist.
    let allowed = |x: usize, y: usize| -> bool {
        (fold.conditions().iter()).all(|c| {
            let (l, r) = (key_at(c.left.0, x, y), key_at(c.right.0, x, y));
            c.comparison.holds(l, r)
        })
    };
    let masked = !fold.conditions().is_empty();
    match (fold.domain(), domain) {
        (Some(spec), Some(domain)) => {
            // The dense output is laid out [x][y], X's and Y's dimensions each
            // first-slowest, so a key's stride is the product of the extents
            // after it in that order.
            let order: Vec<usize> = xs.iter().chain(&ys).copied().collect();
            let cell_keys: Vec<CellKey> = (fold.outputs().iter())
                .map(|o| {
                    let d = o.dimension.0;
                    let slot = order.iter().position(|&e| e == d).expect("an output");
                    CellKey {
                        column: o.domain_key.expect("a fold with a domain").column,
                        extent: extent[d],
                        stride: order[slot + 1..].iter().map(|&e| extent[e]).product(),
                    }
                })
                .collect();
            // A cell's value as the left join gives it: its sum where the
            // group exists with a value, else the default. Whether it has one
            // is spread from x to its cells, so that a lookup needs no
            // division.
            let cell_valued: Vec<bool> = match zero_default {
                true => Vec::new(),
                false => (valued.iter())
                    .flat_map(|&v| std::iter::repeat_n(v, ny))
                    .collect(),
            };
            let value = |c: usize| -> Option<f64> {
                if masked && !allowed(c / ny, c % ny) {
                    return spec.missing;
                }
                if zero_default || cell_valued[c] {
                    Some(out[c])
                } else {
                    spec.missing
                }
            };
            // Each batch's cells, then its rows, on the threads in turn: the
            // batches are independent, and each comes back in its place.
            let located = par_tasks(domain.len(), threads, |i| locate(&domain[i], &cell_keys))
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            if domain_rows == DomainRows::UniqueKeys {
                // Batches in parallel: whichever of two rows with one cell
                // marks it second sees the mark, in any order.
                let seen: Vec<AtomicBool> = (0..nx * ny).map(|_| AtomicBool::new(false)).collect();
                let repeated = par_tasks(located.len(), threads, |i| {
                    (located[i].iter())
                        .any(|&c| c != NO_GROUP && seen[c].swap(true, atomic::Ordering::Relaxed))
                });
                if repeated.contains(&true) {
                    return Ok(None);
                }
            }
            let batches = par_tasks(domain.len(), threads, |i| {
                domain_batch(&domain[i], &located[i], spec, &value, &schema)
            });
            Ok(Some(batches.into_iter().collect::<Result<_>>()?))
        }
        (None, None) => {
            // Without a domain, the groups are (x, y) for each x some row of a
            // reaches, in position order, and every y.
            let rows: Vec<usize> = (0..nx).filter(|&x| reached[x]).collect();
            let n = rows.len() * ny;
            let mut columns: Vec<ArrayRef> = Vec::with_capacity(fold.outputs().len() + 1);
            let output_dimensions = fold.outputs().iter().map(|o| o.dimension.0);
            for d in output_dimensions {
                // The key's place in x's or y's position, first dimension
                // slowest.
                let (in_x, role) = match xs.contains(&d) {
                    true => (true, &xs),
                    false => (false, &ys),
                };
                let slot = role.iter().position(|&e| e == d).unwrap();
                let stride: usize = role[slot + 1..].iter().map(|&e| extent[e]).product();
                let key_at = |p: usize| ((p / stride) % extent[d]) as i64;
                let mut key = Vec::with_capacity(n);
                if in_x {
                    for &x in &rows {
                        key.extend(std::iter::repeat_n(key_at(x), ny));
                    }
                } else {
                    let along: Vec<i64> = (0..ny).map(key_at).collect();
                    for _ in &rows {
                        key.extend_from_slice(&along);
                    }
                }
                let key = Int64Array::from(key);
                let ty = schema.field(columns.len()).data_type();
                columns.push(compute::cast(&(Arc::new(key) as ArrayRef), ty)?);
            }
            let result = if rows.iter().all(|&x| valued[x]) {
                // No NULLs: the output's rows as they are, copied only when
                // some x is missing.
                match rows.len() == nx {
                    true => Float64Array::from(out),
                    false => Float64Array::from(
                        (rows.iter())
                            .flat_map(|&x| out[x * ny..(x + 1) * ny].iter().copied())
                            .collect::<Vec<f64>>(),
                    ),
                }
            } else {
                (rows.iter())
                    .flat_map(|&x| (0..ny).map(move |y| (x, y)))
                    .map(|(x, y)| valued[x].then(|| out[x * ny + y]))
                    .collect()
            };
            columns.push(Arc::new(result));
            let batch = RecordBatch::try_new(schema, columns)?;
            if !masked {
                return Ok(Some(vec![batch]));
            }
            let keep: BooleanArray = (rows.iter())
                .flat_map(|&x| (0..ny).map(move |y| (x, y)))
                .map(|(x, y)| Some(allowed(x, y)))
                .collect();
            Ok(Some(vec![compute::filter_record_batch(&batch, &keep)?]))
        }
        _ => Ok(None),
    }
}

/// How one of a domain's key columns places a row in the dense output: an
/// output's key, the extent of its dimension, and its stride.
struct CellKey {
    /// The domain's column for the output.
    column: ColumnIndex,
    extent: usize,
    stride: usize,
}

/// A domain row whose keys locate no cell of the output.
const NO_GROUP: usize = usize::MAX;

/// The cell of the output that each row of a domain batch's keys locate, or
/// [`NO_GROUP`]. A NULL key, or one outside every operand's range, matches no
/// group (the operands have no NULL keys), under either key equality.
fn locate(batch: &RecordBatch, cell_keys: &[CellKey]) -> Result<Vec<usize>> {
    let mut cell = vec![0usize; batch.num_rows()];
    for key in cell_keys {
        // Out-of-range unsigned keys become NULL here, so match no group.
        let k = compute::cast(batch.column(key.column.0), &DataType::Int64)?;
        let k = k
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("cast to Int64");
        let range = 0..key.extent as i64;
        let rows = cell.iter_mut().zip(k.values()).enumerate();
        for (r, (c, &value)) in rows {
            let in_range = range.contains(&value) && k.is_valid(r);
            *c = match (*c, in_range) {
                (NO_GROUP, _) | (_, false) => NO_GROUP,
                (c, true) => c + value as usize * key.stride,
            };
        }
    }
    Ok(cell)
}

/// The output rows for one batch of the domain: its carried columns, and the
/// value of the cell each row locates (see [`locate`]), or the default where
/// it locates none.
fn domain_batch(
    batch: &RecordBatch,
    cell: &[usize],
    spec: &Domain,
    value: &impl Fn(usize) -> Option<f64>,
    schema: &SchemaRef,
) -> Result<RecordBatch> {
    let at = |c: usize| {
        if c == NO_GROUP {
            spec.missing
        } else {
            value(c)
        }
    };
    // With a default, no value is NULL, so the column needs no validity map.
    let values = match spec.missing {
        Some(m) => Float64Array::from(
            cell.iter()
                .map(|&c| at(c).unwrap_or(m))
                .collect::<Vec<f64>>(),
        ),
        None => cell.iter().map(|&c| at(c)).collect(),
    };
    let mut columns: Vec<ArrayRef> = spec
        .carry
        .iter()
        .map(|&c| Arc::clone(batch.column(c.0)))
        .collect();
    columns.push(Arc::new(values));
    RecordBatch::try_new(Arc::clone(schema), columns)
}

/// `out[x][y] = Σ_z a[x, z]·b[w][z][y]` over the positions of `a` that hold
/// a value (`held` says which), skipping missing rows and NULL factors, in
/// fixed blocks. Here `x` is a row, `(w, x)` in the module's terms, and its
/// batch `w` is `x / per_batch`.
fn accumulate(
    a_matrix: &[f64],
    held: &[u8],
    b_matrix: &[f64],
    (nx, nz, ny): (usize, usize, usize),
    per_batch: usize,
    threads: usize,
) -> Vec<f64> {
    // Adds a[x, z]·b[w][z][y0..y1] into `row`, the output's row x over
    // those columns.
    let add = |row: &mut [f64], x: usize, z: usize, y0: usize| {
        if held[x * nz + z] == VALUED {
            let v = a_matrix[x * nz + z];
            let start = ((x / per_batch) * nz + z) * ny + y0;
            let b_row = &b_matrix[start..start + row.len()];
            for (o, b) in row.iter_mut().zip(b_row) {
                *o += v * b;
            }
        }
    };
    if nx * ny >= FEW_CELLS {
        // Blocks of rows and columns of the output; each cell's terms are
        // still added in z order.
        let (row_blocks, column_blocks) = (nx.div_ceil(XB), ny.div_ceil(YB));
        let blocks = par_tasks(row_blocks * column_blocks, threads, |t| {
            let (x0, y0) = ((t / column_blocks) * XB, (t % column_blocks) * YB);
            let (x1, width) = ((x0 + XB).min(nx), YB.min(ny - y0));
            let mut block = vec![0.0; (x1 - x0) * width];
            for x in x0..x1 {
                let row = &mut block[(x - x0) * width..(x - x0 + 1) * width];
                for z in 0..nz {
                    add(row, x, z, y0);
                }
            }
            block
        });
        let mut out = vec![0.0; nx * ny];
        for (t, block) in blocks.into_iter().enumerate() {
            let (x0, y0) = ((t / column_blocks) * XB, (t % column_blocks) * YB);
            let width = YB.min(ny - y0);
            for (i, row) in block.chunks(width).enumerate() {
                let start = (x0 + i) * ny + y0;
                out[start..start + width].copy_from_slice(row);
            }
        }
        out
    } else {
        let partials = par_tasks(nz.div_ceil(ZB), threads, |t| {
            let mut part = vec![0.0; nx * ny];
            for z in t * ZB..((t + 1) * ZB).min(nz) {
                for x in 0..nx {
                    add(&mut part[x * ny..(x + 1) * ny], x, z, 0);
                }
            }
            part
        });
        let mut out = vec![0.0; nx * ny];
        for part in partials {
            for (o, v) in out.iter_mut().zip(part) {
                *o += v;
            }
        }
        out
    }
}
