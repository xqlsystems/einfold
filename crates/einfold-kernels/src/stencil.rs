// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! EinFold's stencil algorithm: a fold of two operands whose outputs are
//! shown shifted by a dimension of the other operand (see
//! [`ShiftedOutput`]), such as a convolution, computed by positions.
//!
//! A convolution of an image `x(c, lat, lon)` by a kernel `w(co, ci, ky, kx)`
//! is, in SQL,
//!
//! ```sql
//! SELECT w.co, x.lat - (w.ky - 1), x.lon - (w.kx - 1), SUM(x.v * w.v)
//! FROM x JOIN w ON x.c = w.ci GROUP BY 1, 2, 3
//! ```
//!
//! Each *tap* of the kernel, a position `(ky, kx)` of the dimensions the
//! outputs are shifted by, is a matrix product of the image's channels with
//! the tap's weights, `out[lat', lon', co] += Σ_ci x[ci, lat, lon]·w[co, ci,
//! ky, kx]`, whose rows land on output cells shifted by the tap. So the
//! algorithm computes one dense product per tap and adds its rows into the
//! output cells they land on.
//!
//! In general, every dimension has one of these roles: an output held by
//! both operands (W), by the first only (X) or by the second only (Y); a
//! dimension summed away and held by both (Z); or a dimension of a shift, an
//! output shown shifted or the dimension it is shifted by, each held by one
//! operand. The first operand is held as matrices `a[w][x, sa][z]`, rows a
//! position of X and of its shift dimensions `sa`, and the second as
//! `b[w][sb][z][y]`, one matrix per tap `sb`.
//!
//! **When SQL and dense arrays agree**, as for the dense algorithm: keys are
//! small non-negative integers and non-NULL, each operand has one row per key
//! tuple, and the second operand is complete, with a row and a non-NULL
//! factor at every position. Then a group exists exactly when a row of the
//! first operand reaches it through some tap, and its sum is NULL exactly
//! when every such row's factor is. The first operand may have holes, held as
//! 0, only when every factor of the second is finite: `0·∞` is NaN, where SQL
//! forms no product. Otherwise the caller runs something else, such as the
//! host's plan.
//!
//! A shift's bounds, a condition of the join on the shifted key, drop the
//! rows a tap lands outside them, as the join drops their joined rows.
//!
//! **Determinism.** Each tap's product is computed alone, and the taps' rows
//! are added into the output in tap order, so the bits depend neither on the
//! number of threads nor on the order in which rows arrive.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow::compute;
use arrow::datatypes::SchemaRef;
use einfold_ir::{ColumnIndex, DimensionId, Fold, Operand, ShiftedOutput};

use crate::dense::{keys, par_tasks, positions, worth, FEW_ROWS};
use crate::gemm::{self, Matrix};
use crate::options::Options;
use crate::Result;

/// What a position of the first operand holds: no row, a row with a NULL
/// factor, or a row with a value.
const ABSENT: u8 = 0;
const NULL: u8 = 1;
const VALUED: u8 = 2;
/// A row of a tap's product that lands on no output cell.
const NOWHERE: u32 = u32::MAX;

/// One shift, with the range of keys it shows: `low..low + extent`.
struct Range {
    shift: ShiftedOutput,
    low: i64,
    extent: usize,
}

/// The fold of `a` and `b`'s batches by the stencil algorithm on up to
/// `options.threads` threads, as one batch of `schema`, or `None` when its
/// requirements don't hold (see the module's documentation).
pub(crate) fn fold(
    fold: &Fold,
    a: &[RecordBatch],
    b: &[RecordBatch],
    schema: SchemaRef,
    options: &Options,
) -> Result<Option<Vec<RecordBatch>>> {
    let [op_a, op_b] = fold.operands() else {
        return Ok(None);
    };
    if fold.shifts().is_empty()
        || fold.domain().is_some()
        || !fold.conditions().is_empty()
        || !fold.is_einsum()
    {
        return Ok(None);
    }
    let rows = |batches: &[RecordBatch]| batches.iter().map(RecordBatch::num_rows).sum::<usize>();
    let threads = match rows(a) + rows(b) < FEW_ROWS {
        true => 1,
        false => options.threads.max(1),
    };
    // The roles, each a list of dimensions: W, X and Y in output order; Z,
    // and each operand's shift dimensions, in dimension order.
    let shifted = |d: DimensionId| fold.shift_of(d).is_some();
    let shifted_by = |d: DimensionId| fold.shifts().iter().any(|s| s.by == d);
    let (mut ws, mut xs, mut ys, mut zs) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for o in fold.outputs() {
        let d = o.dimension;
        if shifted(d) {
            continue;
        }
        match (op_a.column_of(d), op_b.column_of(d)) {
            (Some(_), Some(_)) => ws.push(d.0),
            (Some(_), None) => xs.push(d.0),
            (None, Some(_)) => ys.push(d.0),
            (None, None) => return Ok(None),
        }
    }
    let (mut sa, mut sb) = (Vec::new(), Vec::new());
    for d in fold.dimension_ids() {
        let (in_a, in_b) = (op_a.column_of(d).is_some(), op_b.column_of(d).is_some());
        if shifted(d) || shifted_by(d) {
            // `Fold::with_shifts` checked that one operand holds each.
            match in_a {
                true => sa.push(d.0),
                false => sb.push(d.0),
            }
        } else if fold.is_aggregated(d) {
            // A summed dimension that one operand holds alone would sum that
            // operand's rows before the product: not this algorithm's shape.
            if !(in_a && in_b) {
                return Ok(None);
            }
            zs.push(d.0);
        }
    }

    // Every key as a non-negative `i64`, batch by batch, with each
    // dimension's extent: its largest key in either operand, plus one.
    let integers = |op: &Operand, batches: &[RecordBatch]| {
        (batches.first()).is_none_or(|batch| {
            (op.keys.iter()).all(|key| batch.column(key.column.0).data_type().is_integer())
        })
    };
    if !integers(op_a, a) || !integers(op_b, b) {
        return Ok(None);
    }
    let cast = rows(a) * op_a.keys.len() + rows(b) * op_b.keys.len();
    options.reserve(cast.saturating_mul(8), "the stencil algorithm's keys")?;
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
    let size = |dims: &[usize]| {
        dims.iter()
            .try_fold(1usize, |n, &d| n.checked_mul(extent[d]))
    };

    // Each shift's range of keys, from every pair of keys its two dimensions
    // can hold.
    let mut ranges = Vec::new();
    for o in fold.outputs() {
        let Some(&shift) = fold.shift_of(o.dimension) else {
            continue;
        };
        let (n, m) = (extent[shift.output.0], extent[shift.by.0]);
        if n.saturating_mul(m) > 1 << 24 {
            return Ok(None);
        }
        let mut bounds: Option<(i64, i64)> = None;
        for (k, by) in (0..n).flat_map(|k| (0..m).map(move |by| (k, by))) {
            if let Some(key) = shift.key(k as i64, by as i64) {
                bounds = Some(bounds.map_or((key, key), |(lo, hi)| (lo.min(key), hi.max(key))));
            }
        }
        let (low, extent) = match bounds {
            Some((lo, hi)) => (
                lo,
                usize::try_from(hi - lo).ok().and_then(|e| e.checked_add(1)),
            ),
            None => (0, Some(0)),
        };
        let Some(extent) = extent else {
            return Ok(None);
        };
        ranges.push(Range { shift, low, extent });
    }
    let shown = (ranges.iter()).try_fold(1usize, |n, r| n.checked_mul(r.extent));
    let (Some(nw), Some(nx), Some(nz), Some(ny), Some(npa), Some(npb), Some(ns)) = (
        size(&ws),
        size(&xs),
        size(&zs),
        size(&ys),
        size(&sa),
        size(&sb),
        shown,
    ) else {
        return Ok(None);
    };
    // Cells of a, b, a tap's product, and the output.
    let cells = |dims: &[usize]| dims.iter().try_fold(1usize, |n, &d| n.checked_mul(d));
    let (Some(na), Some(nb), Some(nc), Some(nout)) = (
        cells(&[nw, nx, npa, nz]),
        cells(&[nw, npb, nz, ny]),
        cells(&[nx, npa, ny]),
        cells(&[nw, nx, ns, ny]),
    ) else {
        return Ok(None);
    };
    // b must be complete, so has a row per cell.
    if !worth(na, rows(a)) || nb != rows(b) || nout >= u32::MAX as usize {
        return Ok(None);
    }
    // a and its flags; b and its; one product per thread; the output, with
    // each row's flags; and the output's columns.
    let working = [
        na.saturating_mul(9),
        nb.saturating_mul(9),
        nc.saturating_mul(8).saturating_mul(threads),
        nout.saturating_mul(8),
        nw.saturating_mul(nx).saturating_mul(ns).saturating_mul(2),
        nout.saturating_mul(fold.outputs().len() + 1)
            .saturating_mul(8),
    ];
    let working = working.into_iter().fold(0usize, usize::saturating_add);
    options.reserve(working, "the stencil algorithm's matrices")?;

    let factors = |op: &Operand, batch: &RecordBatch| {
        op.factor
            .and_then(|c| batch.column(c.0).as_any().downcast_ref::<Float64Array>())
            .cloned()
    };
    // a as matrices [w][x, sa][z]: each row's cell.
    let mut a_matrix = vec![0.0; na];
    let mut held = vec![ABSENT; na];
    let a_cells = par_tasks(a.len(), threads, |i| {
        let n = a[i].num_rows();
        let (w, x) = (
            positions(&a_keys[i], &ws, &extent, n),
            positions(&a_keys[i], &xs, &extent, n),
        );
        let (s, z) = (
            positions(&a_keys[i], &sa, &extent, n),
            positions(&a_keys[i], &zs, &extent, n),
        );
        (0..n)
            .map(|r| {
                let row = (w[r] as usize * nx + x[r] as usize) * npa + s[r] as usize;
                row * nz + z[r] as usize
            })
            .collect::<Vec<usize>>()
    });
    for (batch, cells) in a.iter().zip(&a_cells) {
        let Some(values) = factors(op_a, batch) else {
            return Ok(None);
        };
        for (r, &p) in cells.iter().enumerate() {
            let valid = values.is_valid(r);
            if std::mem::replace(&mut held[p], if valid { VALUED } else { NULL }) != ABSENT {
                return Ok(None);
            }
            if valid {
                a_matrix[p] = values.value(r);
            }
        }
    }
    // b as matrices [w][sb][z][y], complete and non-NULL.
    let mut b_matrix = vec![0.0; nb];
    let mut present = vec![false; nb];
    let b_cells = par_tasks(b.len(), threads, |i| {
        let n = b[i].num_rows();
        let (w, s) = (
            positions(&b_keys[i], &ws, &extent, n),
            positions(&b_keys[i], &sb, &extent, n),
        );
        let (z, y) = (
            positions(&b_keys[i], &zs, &extent, n),
            positions(&b_keys[i], &ys, &extent, n),
        );
        (0..n)
            .map(|r| {
                let tap = w[r] as usize * npb + s[r] as usize;
                (tap * nz + z[r] as usize) * ny + y[r] as usize
            })
            .collect::<Vec<usize>>()
    });
    let mut finite = true;
    for (batch, cells) in b.iter().zip(&b_cells) {
        let Some(values) = factors(op_b, batch) else {
            return Ok(None);
        };
        if values.null_count() > 0 {
            return Ok(None);
        }
        for (r, &p) in cells.iter().enumerate() {
            if std::mem::replace(&mut present[p], true) {
                return Ok(None);
            }
            b_matrix[p] = values.value(r);
            finite &= b_matrix[p].is_finite();
        }
    }
    // As many rows as cells, and none twice: every cell is present.
    let holes = held.iter().any(|&h| h != VALUED);
    if holes && !finite {
        return Ok(None);
    }
    // Whether each row (w, x, sa) of a holds a row of the operand, and
    // whether one with a value: the groups its products reach exist, and
    // aren't NULL.
    let a_rows = nw * nx * npa;
    let row_held: Vec<u8> = (0..a_rows)
        .map(|r| {
            held[r * nz..(r + 1) * nz]
                .iter()
                .copied()
                .max()
                .unwrap_or(ABSENT)
        })
        .collect();
    drop(held);

    // The keys each position of a's and b's shift dimensions holds.
    let decode = |dims: &[usize], p: usize, d: usize| -> i64 {
        let slot = dims
            .iter()
            .position(|&e| e == d)
            .expect("a shift dimension");
        let stride: usize = dims[slot + 1..].iter().map(|&e| extent[e]).product();
        ((p / stride) % extent[d]) as i64
    };
    let key_of = |d: usize, pa: usize, pb: usize| match sa.contains(&d) {
        true => decode(&sa, pa, d),
        false => decode(&sb, pb, d),
    };
    // Where the rows of tap pb's product land: each position of a's shift
    // dimensions' shifted cell, or NOWHERE.
    let landing = |pb: usize| -> Vec<u32> {
        (0..npa)
            .map(|pa| {
                let mut cell = 0usize;
                for r in &ranges {
                    let (k, by) = (
                        key_of(r.shift.output.0, pa, pb),
                        key_of(r.shift.by.0, pa, pb),
                    );
                    let Some(key) = r.shift.key(k, by) else {
                        return NOWHERE;
                    };
                    cell = cell * r.extent + (key - r.low) as usize;
                }
                cell as u32
            })
            .collect()
    };

    // The output, [w][x][s][y], and each (w, x, s)'s flag: the most a row
    // reaching it held.
    let mut out = vec![0.0; nout];
    let mut reached = vec![ABSENT; nw * nx * ns];
    for w in 0..nw {
        let a_w = &a_matrix[w * nx * npa * nz..(w + 1) * nx * npa * nz];
        // Taps a batch at a time, a product per thread, each added in tap
        // order.
        for first in (0..npb).step_by(threads) {
            let taps = threads.min(npb - first);
            let products = par_tasks(taps, threads, |t| {
                let pb = first + t;
                let start = (w * npb + pb) * nz * ny;
                let mut c = vec![0.0; nc];
                gemm::multiply_add(
                    (nx * npa, nz, ny),
                    Matrix {
                        data: a_w,
                        stride: nz,
                    },
                    Matrix {
                        data: &b_matrix[start..start + nz * ny],
                        stride: ny,
                    },
                    &mut c,
                    ny,
                );
                (c, landing(pb))
            });
            for (c, lands) in products {
                for x in 0..nx {
                    for (pa, &s) in lands.iter().enumerate() {
                        if s == NOWHERE {
                            continue;
                        }
                        let row = (w * nx + x) * npa + pa;
                        let flag = row_held[row];
                        if flag == ABSENT {
                            continue;
                        }
                        let cell = (w * nx + x) * ns + s as usize;
                        reached[cell] = reached[cell].max(flag);
                        let product = &c[(x * npa + pa) * ny..(x * npa + pa + 1) * ny];
                        let into = &mut out[cell * ny..(cell + 1) * ny];
                        for (o, v) in into.iter_mut().zip(product) {
                            *o += v;
                        }
                    }
                }
            }
        }
    }
    drop(a_matrix);
    drop(b_matrix);

    // The groups: each (w, x, s) a row reached, with every y.
    let groups: Vec<usize> = (0..nw * nx * ns)
        .filter(|&g| reached[g] != ABSENT)
        .collect();
    let n = groups.len() * ny;
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(fold.outputs().len() + 1);
    // The key of plain dimension `d` at a position `p` of its role.
    let plain = |role: &[usize], d: usize, p: usize| -> i64 {
        let slot = role.iter().position(|&e| e == d).expect("an output");
        let stride: usize = role[slot + 1..].iter().map(|&e| extent[e]).product();
        ((p / stride) % extent[d]) as i64
    };
    for o in fold.outputs() {
        let d = o.dimension.0;
        let mut key = Vec::with_capacity(n);
        for &g in &groups {
            let (wx, s) = (g / ns, g % ns);
            let (w, x) = (wx / nx, wx % nx);
            let at = |y: usize| -> i64 {
                if let Some(i) = ranges.iter().position(|r| r.shift.output.0 == d) {
                    let stride: usize = ranges[i + 1..].iter().map(|r| r.extent).product();
                    ranges[i].low + ((s / stride) % ranges[i].extent) as i64
                } else if ws.contains(&d) {
                    plain(&ws, d, w)
                } else if xs.contains(&d) {
                    plain(&xs, d, x)
                } else {
                    plain(&ys, d, y)
                }
            };
            key.extend((0..ny).map(at));
        }
        let key = Arc::new(Int64Array::from(key)) as ArrayRef;
        let ty = schema.field(columns.len()).data_type();
        columns.push(compute::cast(&key, ty)?);
    }
    let values: Float64Array = (groups.iter())
        .flat_map(|&g| {
            let valued = reached[g] == VALUED;
            out[g * ny..(g + 1) * ny]
                .iter()
                .map(move |&v| valued.then_some(v))
        })
        .collect();
    columns.push(Arc::new(values));
    Ok(Some(vec![RecordBatch::try_new(schema, columns)?]))
}
