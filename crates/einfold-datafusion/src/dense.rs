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
//! is its largest key plus one. `b` is held as a dense matrix, and each row of
//! `a` adds its factor times a row of `b` into a row of the output. This is the
//! positional *fold* of the design: only `b` and the output are dense.
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
//! - Which groups exist matters only without a fill (with one, an unreached
//!   group and one whose products were all NULL both give 0). Without a fill,
//!   `b` must be complete, with every position present and non-NULL. Then a
//!   group `(x, y)` exists exactly when `a` has a row at `x`, and its sum is
//!   NULL exactly when every such row's factor is NULL.
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

use datafusion::arrow::array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{DataType, SchemaRef};
use datafusion::common::Result;
use einfold_ir::Fold;

/// Output rows per block, when the output has at least [`FEW_CELLS`] cells.
const XB: usize = 256;
/// Positions of `z` per block, when the output is smaller.
const ZB: usize = 4096;
const FEW_CELLS: usize = 4096;
/// A row of `a` that isn't there.
const NONE: u32 = u32::MAX;

/// Whether a dense array of `cells` is worth building for `rows` rows: no
/// more than four cells per row (S11 found dense beats sparse above about 20%
/// density), plus some slack for small inputs.
fn worth(cells: usize, rows: usize) -> bool {
    cells <= rows.saturating_mul(4).saturating_add(FEW_CELLS) && cells < NONE as usize
}

/// `column` as non-negative `i64` keys, or `None` if it isn't an integer
/// column, or holds a NULL or a negative key.
fn keys(column: &ArrayRef) -> Option<Int64Array> {
    if !column.data_type().is_integer() || column.null_count() > 0 {
        return None;
    }
    let keys = cast(column, &DataType::Int64).ok()?;
    let keys = keys.as_any().downcast_ref::<Int64Array>()?.clone();
    keys.values().iter().all(|&k| k >= 0).then_some(keys)
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

/// The fold of `a` and `b` (and the batches of `fill`, if the fold has one)
/// by the dense algorithm on `threads` threads, or `None` when its conditions
/// don't hold and the hash path must be used instead. With a fill, there is
/// one output batch per fill batch.
pub(crate) fn fold(
    fold: &Fold,
    a: &RecordBatch,
    b: &RecordBatch,
    fill: Option<&[RecordBatch]>,
    schema: SchemaRef,
    threads: usize,
) -> Result<Option<Vec<RecordBatch>>> {
    let [op_a, op_b] = fold.operands() else {
        return Ok(None);
    };
    // The roles: X and Y in output order, Z in dimension order.
    let (mut xs, mut ys, mut zs) = (Vec::new(), Vec::new(), Vec::new());
    for &d in fold.outputs() {
        match (op_a.column_of(d), op_b.column_of(d)) {
            (Some(_), None) => xs.push(d),
            (None, Some(_)) => ys.push(d),
            _ => return Ok(None),
        }
    }
    for d in 0..fold.dimensions().len() {
        if fold.is_summed(d) {
            if op_a.column_of(d).is_none() || op_b.column_of(d).is_none() {
                return Ok(None);
            }
            zs.push(d);
        }
    }
    // Each operand's key columns as non-negative i64, cast once; and each
    // dimension's extent, its largest key in either operand, plus one.
    let mut extent = vec![0usize; fold.dimensions().len()];
    let mut cast_keys = |op: &einfold_ir::Operand, batch: &RecordBatch| {
        op.keys
            .iter()
            .map(|&(d, c)| {
                let k = keys(batch.column(c))?;
                let max = k.values().iter().max().map_or(0, |&m| m as usize + 1);
                extent[d] = extent[d].max(max);
                Some((d, k))
            })
            .collect::<Option<Vec<_>>>()
    };
    let (Some(a_keys), Some(b_keys)) = (cast_keys(op_a, a), cast_keys(op_b, b)) else {
        return Ok(None);
    };
    let size = |dims: &[usize]| {
        dims.iter()
            .try_fold(1usize, |n, &d| n.checked_mul(extent[d]))
    };
    let (Some(nx), Some(nz), Some(ny)) = (size(&xs), size(&zs), size(&ys)) else {
        return Ok(None);
    };
    let out_rows = fill.map_or(nx.saturating_mul(ny), |f| {
        f.iter().map(|b| b.num_rows()).sum()
    });
    if !worth(nx.saturating_mul(nz), a.num_rows())
        || !worth(nz.saturating_mul(ny), b.num_rows())
        || !worth(nx.saturating_mul(ny), out_rows)
    {
        return Ok(None);
    }
    let (ax, az) = (
        Position::new(&a_keys, &xs, &extent),
        Position::new(&a_keys, &zs, &extent),
    );
    let (bz, by) = (
        Position::new(&b_keys, &zs, &extent),
        Position::new(&b_keys, &ys, &extent),
    );
    let Some(av) = a.column(op_a.value).as_any().downcast_ref::<Float64Array>() else {
        return Ok(None);
    };
    let Some(bv) = b.column(op_b.value).as_any().downcast_ref::<Float64Array>() else {
        return Ok(None);
    };

    // a's rows by position; a repeated position means duplicate tuples.
    let mut at = vec![NONE; nx * nz];
    for r in 0..a.num_rows() {
        if std::mem::replace(&mut at[ax.of(r) * nz + az.of(r)], r as u32) != NONE {
            return Ok(None);
        }
    }
    // b as a dense matrix [z][y], with holes (missing or NULL rows) as 0.
    let mut bm = vec![0.0; nz * ny];
    let mut present = vec![false; nz * ny];
    for r in 0..b.num_rows() {
        let p = bz.of(r) * ny + by.of(r);
        if std::mem::replace(&mut present[p], true) {
            return Ok(None);
        }
        if bv.is_valid(r) {
            bm[p] = bv.value(r);
        }
    }
    let holes = bv.null_count() > 0 || present.iter().any(|&p| !p);
    if holes {
        let finite = (0..av.len()).all(|r| av.is_null(r) || av.value(r).is_finite());
        if fill.is_none() || !finite {
            return Ok(None);
        }
    }

    let out = accumulate(&at, av, &bm, (nx, nz, ny), threads);
    match (fold.fill(), fill) {
        (Some(spec), Some(fill)) => {
            // The dense output is laid out [x][y], X's and Y's dimensions each
            // first-slowest, so a key's stride is the product of the extents
            // after it in that order.
            let order: Vec<usize> = xs.iter().chain(&ys).copied().collect();
            let strides: Vec<usize> = fold
                .outputs()
                .iter()
                .map(|d| {
                    let slot = order.iter().position(|e| e == d).expect("an output");
                    order[slot + 1..].iter().map(|&e| extent[e]).product()
                })
                .collect();
            let batches = fill
                .iter()
                .map(|f| fill_batch(f, spec, fold.outputs(), &strides, &extent, &out, &schema))
                .collect::<Result<_>>()?;
            Ok(Some(batches))
        }
        (None, None) => {
            // b is complete, so group (x, y) exists when a has a row at x, and
            // is NULL when all of that row's factors are.
            let (mut reached, mut valued) = (vec![false; nx], vec![false; nx]);
            for r in 0..a.num_rows() {
                let x = ax.of(r);
                reached[x] = true;
                valued[x] |= av.is_valid(r);
            }
            let groups: Vec<(usize, usize)> = (0..nx)
                .filter(|&x| reached[x])
                .flat_map(|x| (0..ny).map(move |y| (x, y)))
                .collect();
            let mut columns: Vec<ArrayRef> = Vec::with_capacity(fold.outputs().len() + 1);
            for &d in fold.outputs() {
                // The key's place in x's or y's position, first dimension
                // slowest.
                let (in_x, role) = match xs.contains(&d) {
                    true => (true, &xs),
                    false => (false, &ys),
                };
                let slot = role.iter().position(|&e| e == d).unwrap();
                let stride: usize = role[slot + 1..].iter().map(|&e| extent[e]).product();
                let key = Int64Array::from_iter_values(groups.iter().map(|&(x, y)| {
                    let p = if in_x { x } else { y };
                    ((p / stride) % extent[d]) as i64
                }));
                let ty = schema.field(columns.len()).data_type();
                columns.push(cast(&(Arc::new(key) as ArrayRef), ty)?);
            }
            let result: Float64Array = groups
                .iter()
                .map(|&(x, y)| valued[x].then(|| out[x * ny + y]))
                .collect();
            columns.push(Arc::new(result));
            Ok(Some(vec![RecordBatch::try_new(schema, columns)?]))
        }
        _ => Ok(None),
    }
}

/// The output rows for one batch of the fill: each fill row's key columns,
/// and the value of its group's cell of `out`, which `strides` locate. A
/// NULL key, or one outside every operand's range, belongs to no group, so
/// its row gets 0; a row whose own value is NULL gets NULL.
fn fill_batch(
    fill: &RecordBatch,
    spec: &einfold_ir::Fill,
    outputs: &[usize],
    strides: &[usize],
    extent: &[usize],
    out: &[f64],
    schema: &SchemaRef,
) -> Result<RecordBatch> {
    const NO_GROUP: usize = usize::MAX;
    let mut cell = vec![0usize; fill.num_rows()];
    for ((&d, &stride), &col) in outputs.iter().zip(strides).zip(&spec.keys) {
        let k = cast(fill.column(col), &DataType::Int64)?;
        let k = k
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("cast to Int64");
        let range = 0..extent[d] as i64;
        for (r, (c, key)) in cell.iter_mut().zip(k.values()).enumerate() {
            let in_range = range.contains(key) && k.is_valid(r);
            *c = match (*c, in_range) {
                (NO_GROUP, _) | (_, false) => NO_GROUP,
                (c, true) => c + *key as usize * stride,
            };
        }
    }
    let values: Vec<f64> = cell
        .iter()
        .map(|&c| if c == NO_GROUP { 0.0 } else { out[c] })
        .collect();
    // The fill's own NULLs, as they are.
    let nulls = fill.column(spec.value).logical_nulls();
    let mut columns: Vec<ArrayRef> = spec
        .keys
        .iter()
        .map(|&c| Arc::clone(fill.column(c)))
        .collect();
    columns.push(Arc::new(Float64Array::new(values.into(), nulls)));
    Ok(RecordBatch::try_new(Arc::clone(schema), columns)?)
}

/// `out[x][y] = Σ_z a[x, z]·b[z][y]` over the rows of `a` (`at` gives each
/// position's row), skipping NULL factors of `a`, in fixed blocks.
fn accumulate(
    at: &[u32],
    av: &Float64Array,
    bm: &[f64],
    (nx, nz, ny): (usize, usize, usize),
    threads: usize,
) -> Vec<f64> {
    let add = |row: &mut [f64], x: usize, z: usize| {
        let r = at[x * nz + z];
        if r != NONE && av.is_valid(r as usize) {
            let v = av.value(r as usize);
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
