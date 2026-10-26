// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! EinFold's dense algorithm: a two-operand fold computed by positions, with
//! no hashing, when the keys are small non-negative integers.
//!
//! Every dimension must have one of three roles:
//!
//! - **X**: an output held only by the first operand, `a`;
//! - **Z**: a summed dimension held by both operands (the join's keys);
//! - **Y**: an output held only by the second operand, `b`.
//!
//! That is a matrix product `out[x, y] = Σ_z a[x, z]·b[z, y]`, with each role's
//! dimensions flattened into one position. Keys are used as positions
//! directly: key `k` of a dimension is position `k`, and a dimension's extent
//! is its largest key plus one. Both operands are placed in dense matrices,
//! batch by batch, as they arrive, and each position of `a` adds its factor
//! times a row of `b` into a row of the output.
//!
//! **When SQL and dense arrays agree.** A dense array has a value at every
//! position, where SQL has rows only where they exist:
//!
//! - Each operand must have one row per key tuple, so that a position holds
//!   one row. Duplicate tuples, which SQL sums row by row, use the hash path.
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
//! Otherwise the caller falls back to the hash path.
//!
//! **Determinism.** Work is split into fixed blocks chosen from the shape
//! alone: blocks of output rows when the output is large, or blocks along `z`
//! when it is small, whose partial outputs are added in block order. Within a
//! block, rows are visited in position order. So the bits depend neither on
//! the number of threads nor on the order in which rows arrive.

use std::sync::Arc;
use std::thread;

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow::compute::cast;
use arrow::datatypes::{DataType, SchemaRef};
use einfold_ir::{ColumnIndex, DimensionId, Domain, Fold};

use crate::options::DomainRows;
use crate::Result;

/// Output rows per block, when the output has at least [`FEW_CELLS`] cells.
const XB: usize = 256;
/// Positions of `z` per block, when the output is smaller.
const ZB: usize = 4096;
const FEW_CELLS: usize = 4096;
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
fn keys(column: &ArrayRef) -> Option<(Int64Array, usize)> {
    if !column.data_type().is_integer() || column.null_count() > 0 {
        return None;
    }
    // The cast turns a key too large for i64 into NULL, which reads as 0.
    let keys = cast(column, &DataType::Int64).ok()?;
    let keys = keys.as_any().downcast_ref::<Int64Array>()?.clone();
    // One pass for both bounds, which the compiler vectorizes.
    let (lo, hi) = (keys.values().iter()).fold((0, -1), |(lo, hi), &k| (k.min(lo), k.max(hi)));
    if keys.null_count() > 0 || lo < 0 {
        return None;
    }
    let extent = usize::try_from(hi.checked_add(1)?).ok()?;
    Some((keys, extent))
}

/// A row's position over some dimensions, flattened with the first
/// dimension slowest: each dimension's keys, with its extent.
struct Position<'a>(Vec<(&'a [i64], usize)>);

impl<'a> Position<'a> {
    /// The position over `dims`, from an operand's `(dimension, keys)`
    /// columns.
    fn new(keys: &'a [(usize, Int64Array)], dims: &[usize], extent: &[usize]) -> Self {
        Position(
            dims.iter()
                .map(|&d| {
                    let (_, k) = keys
                        .iter()
                        .find(|(e, _)| *e == d)
                        .expect("a held dimension");
                    (&k.values()[..], extent[d])
                })
                .collect(),
        )
    }

    /// Row `r`'s position.
    fn of(&self, r: usize) -> usize {
        self.0.iter().fold(0, |p, (k, n)| p * n + k[r] as usize)
    }
}

/// Run `f(i)` for every task `i` in `0..tasks` on up to `threads` threads.
/// Results come back in task order, so how tasks were assigned to threads
/// never shows in them.
fn par_tasks<T: Send>(tasks: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let threads = threads.clamp(1, tasks.max(1));
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
/// by the dense algorithm on `threads` threads, or `None` when its conditions
/// don't hold and the hash path must be used instead. With a domain, there is
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
    // The roles: X and Y in output order, Z in dimension order. Here a
    // dimension is its index, which indexes the extents too.
    let (mut xs, mut ys, mut zs) = (Vec::new(), Vec::new(), Vec::new());
    for d in fold.outputs().iter().map(|o| o.dimension) {
        match (op_a.column_of(d), op_b.column_of(d)) {
            (Some(_), None) => xs.push(d.0),
            (None, Some(_)) => ys.push(d.0),
            _ => return Ok(None),
        }
    }
    for d in fold.dimension_ids() {
        if fold.is_aggregated(d) {
            if op_a.column_of(d).is_none() || op_b.column_of(d).is_none() {
                return Ok(None);
            }
            zs.push(d.0);
        }
    }
    // Each batch's key columns as non-negative i64, cast once; and each
    // dimension's extent, its largest key in either operand, plus one.
    let mut extent = vec![0usize; fold.dimensions().len()];
    let mut cast_keys = |op: &einfold_ir::Operand, batches: &[RecordBatch]| {
        (batches.iter())
            .map(|batch| {
                (op.keys.iter())
                    .map(|key| {
                        let (DimensionId(d), ColumnIndex(c)) = (key.dimension, key.column);
                        let (k, n) = keys(batch.column(c))?;
                        extent[d] = extent[d].max(n);
                        Some((d, k))
                    })
                    .collect::<Option<Vec<_>>>()
            })
            .collect::<Option<Vec<_>>>()
    };
    let (Some(a_keys), Some(b_keys)) = (cast_keys(op_a, a), cast_keys(op_b, b)) else {
        return Ok(None);
    };
    let rows = |batches: &[RecordBatch]| batches.iter().map(|b| b.num_rows()).sum::<usize>();
    let size = |dims: &[usize]| {
        dims.iter()
            .try_fold(1usize, |n, &d| n.checked_mul(extent[d]))
    };
    let (Some(nx), Some(nz), Some(ny)) = (size(&xs), size(&zs), size(&ys)) else {
        return Ok(None);
    };
    let out_rows = domain.map_or(nx.saturating_mul(ny), |f| {
        f.iter().map(|b| b.num_rows()).sum()
    });
    if !worth(nx.saturating_mul(nz), rows(a))
        || !worth(nz.saturating_mul(ny), rows(b))
        || !worth(nx.saturating_mul(ny), out_rows)
    {
        return Ok(None);
    }
    let factors = |op: &einfold_ir::Operand, batch: &'_ RecordBatch| {
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
    let mut am = vec![0.0; nx * nz];
    let mut held = vec![ABSENT; nx * nz];
    let mut reached = vec![false; nx];
    let mut valued = vec![false; if zero_default { 0 } else { nx }];
    let mut finite = true;
    for (batch, keys) in a.iter().zip(&a_keys) {
        let (ax, az) = (
            Position::new(keys, &xs, &extent),
            Position::new(keys, &zs, &extent),
        );
        let Some(av) = factors(op_a, batch) else {
            return Ok(None);
        };
        for r in 0..batch.num_rows() {
            let x = ax.of(r);
            let p = x * nz + az.of(r);
            let valid = av.is_valid(r);
            if std::mem::replace(&mut held[p], if valid { VALUED } else { NULL }) != ABSENT {
                return Ok(None);
            }
            reached[x] = true;
            if valid {
                am[p] = av.value(r);
                finite &= am[p].is_finite();
            }
            if !zero_default {
                valued[x] |= valid;
            }
        }
    }
    // b as a dense matrix [z][y], with holes (missing or NULL rows) as 0.
    let mut bm = vec![0.0; nz * ny];
    let mut present = vec![false; nz * ny];
    let mut b_nulls = false;
    for (batch, keys) in b.iter().zip(&b_keys) {
        let (bz, by) = (
            Position::new(keys, &zs, &extent),
            Position::new(keys, &ys, &extent),
        );
        let Some(bv) = factors(op_b, batch) else {
            return Ok(None);
        };
        b_nulls |= bv.null_count() > 0;
        for r in 0..batch.num_rows() {
            let p = bz.of(r) * ny + by.of(r);
            if std::mem::replace(&mut present[p], true) {
                return Ok(None);
            }
            if bv.is_valid(r) {
                bm[p] = bv.value(r);
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

    let out = accumulate(&am, &held, &bm, (nx, nz, ny), threads);
    match (fold.domain(), domain) {
        (Some(spec), Some(domain)) => {
            // The dense output is laid out [x][y], X's and Y's dimensions each
            // first-slowest, so a key's stride is the product of the extents
            // after it in that order.
            let order: Vec<usize> = xs.iter().chain(&ys).copied().collect();
            let dims: Vec<usize> = fold.outputs().iter().map(|o| o.dimension.0).collect();
            let keys: Vec<ColumnIndex> = fold
                .outputs()
                .iter()
                .map(|o| o.domain_key.expect("a fold with a domain").column)
                .collect();
            let strides: Vec<usize> = dims
                .iter()
                .map(|d| {
                    let slot = order.iter().position(|e| e == d).expect("an output");
                    order[slot + 1..].iter().map(|&e| extent[e]).product()
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
                if zero_default || cell_valued[c] {
                    Some(out[c])
                } else {
                    spec.missing
                }
            };
            let cells = Cells {
                dims: &dims,
                keys: &keys,
                strides: &strides,
                extent: &extent,
            };
            // Each batch's cells, then its rows, on the threads in turn: the
            // batches are independent, and each comes back in its place.
            let located = par_tasks(domain.len(), threads, |i| locate(&domain[i], &cells))
                .into_iter()
                .collect::<Result<Vec<_>>>()?;
            if domain_rows == DomainRows::UniqueKeys {
                let mut seen = vec![false; nx * ny];
                for &c in located.iter().flatten() {
                    if c != NO_GROUP && std::mem::replace(&mut seen[c], true) {
                        return Ok(None);
                    }
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
            for d in fold.outputs().iter().map(|o| o.dimension.0) {
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
                columns.push(cast(&(Arc::new(key) as ArrayRef), ty)?);
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
            Ok(Some(vec![RecordBatch::try_new(schema, columns)?]))
        }
        _ => Ok(None),
    }
}

/// Where the output's cells are: each output dimension, its stride in the
/// dense output, and every dimension's extent.
struct Cells<'a> {
    dims: &'a [usize],
    /// The domain's column for each output.
    keys: &'a [ColumnIndex],
    strides: &'a [usize],
    extent: &'a [usize],
}

/// A domain row whose keys locate no cell of the output.
const NO_GROUP: usize = usize::MAX;

/// The cell of the output that each row of a domain batch's keys locate, or
/// [`NO_GROUP`]. A NULL key, or one outside every operand's range, matches no
/// group (the operands have no NULL keys), under either key equality.
fn locate(batch: &RecordBatch, cells: &Cells) -> Result<Vec<usize>> {
    let mut cell = vec![0usize; batch.num_rows()];
    for ((&d, &stride), &ColumnIndex(col)) in cells.dims.iter().zip(cells.strides).zip(cells.keys) {
        // Out-of-range unsigned keys become NULL here, so match no group.
        let k = cast(batch.column(col), &DataType::Int64)?;
        let k = k
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("cast to Int64");
        let range = 0..cells.extent[d] as i64;
        for (r, (c, key)) in cell.iter_mut().zip(k.values()).enumerate() {
            let in_range = range.contains(key) && k.is_valid(r);
            *c = match (*c, in_range) {
                (NO_GROUP, _) | (_, false) => NO_GROUP,
                (c, true) => c + *key as usize * stride,
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

/// `out[x][y] = Σ_z a[x, z]·b[z][y]` over the positions of `a` that hold a
/// value (`held` says which), skipping missing rows and NULL factors, in
/// fixed blocks.
fn accumulate(
    am: &[f64],
    held: &[u8],
    bm: &[f64],
    (nx, nz, ny): (usize, usize, usize),
    threads: usize,
) -> Vec<f64> {
    let add = |row: &mut [f64], x: usize, z: usize| {
        if held[x * nz + z] == VALUED {
            let v = am[x * nz + z];
            for (o, b) in row.iter_mut().zip(&bm[z * ny..(z + 1) * ny]) {
                *o += v * b;
            }
        }
    };
    if nx * ny >= FEW_CELLS {
        par_tasks(nx.div_ceil(XB), threads, |t| {
            let (x0, x1) = (t * XB, ((t + 1) * XB).min(nx));
            let mut block = vec![0.0; (x1 - x0) * ny];
            for x in x0..x1 {
                let row = &mut block[(x - x0) * ny..(x - x0 + 1) * ny];
                for z in 0..nz {
                    add(row, x, z);
                }
            }
            block
        })
        .concat()
    } else {
        let partials = par_tasks(nz.div_ceil(ZB), threads, |t| {
            let mut part = vec![0.0; nx * ny];
            for z in t * ZB..((t + 1) * ZB).min(nz) {
                for x in 0..nx {
                    add(&mut part[x * ny..(x + 1) * ny], x, z);
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
