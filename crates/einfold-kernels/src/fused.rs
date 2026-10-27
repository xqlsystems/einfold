// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! EinFold's fused pipeline algorithm: a [`Pipeline`] computed tile by tile,
//! so that the producer's results never exist in full.
//!
//! Every dimension has one of four roles:
//!
//! - **X**: a producer output held by its first operand, `q`, kept by the
//!   pipeline (attention's queries `i`);
//! - **E**: summed by the producer, held by both its operands (features);
//! - **J**: a producer output held by its second operand, `k`, summed by the
//!   consumer and joined with the values operand `v` (keys `j`);
//! - **Y**: the pipeline's own dimensions, held by `v` (value features `d`).
//!
//! As in the dense fold, keys are positions, each role's dimensions are
//! flattened into one position, and every operand is placed in a dense
//! matrix. For each tile of X positions, on one thread, the J positions are
//! visited in tiles, in order. Each pair of tiles computes its block of
//! producer values with a matrix product over E, turns them into scores and
//! weights, and folds the weights into the tile's state, then the next pair
//! reuses the block. Memory beyond inputs and output is one block per thread.
//!
//! **The scaled accumulator.** A row's state is `(m, z, a[y])`: a shift `m`,
//! the sum of its weights `z = Σ exp(s − m)`, and the weighted sums `a[y] =
//! Σ exp(s − m)·v[y]`. A block whose largest score is above `m` raises the
//! shift to it, and rescales `z` and `a` by `exp(m_old − m_new)` first. This
//! is Milakov and Gimelshein's online softmax ("Online normalizer calculation
//! for softmax", 2018). With [`Shift::KeptMaximum`], it equals SQL's `exp(s −
//! MAX(s))`, since the shift cancels in `a / z`. No weight exceeds 1, and
//! each term goes through one rounding of `exp` and a product per later
//! raise of the shift. With [`Shift::None`], `m` stays 0, as SQL's `exp(s)`
//! does, overflow included.
//!
//! **Masks are absent rows.** A [`Condition`] between the X and J dimensions
//! allows, for each X position, an interval of J positions, since keys are
//! positions. A pair of tiles the condition disallows entirely is never
//! visited. In a pair it disallows in part, disallowed pairs are skipped, not
//! given weight 0: `0 · NaN` is NaN, but SQL forms no row for them. An X
//! position with no allowed J position has no rows in the output, as in SQL.
//!
//! **When SQL and dense arrays agree.** Every operand must be complete: one
//! row at every position of its dimensions, and no NULL factors. Then every
//! producer group exists unless masked, both [`Normalizer`]s agree, and NaN
//! and infinities propagate as SQL's arithmetic does. Otherwise the caller
//! runs something else, such as the host's plan.
//!
//! **Determinism.** Tiles are fixed by the shape, each X tile is computed by
//! one thread, J tiles are visited in order, and each tile's matrix products
//! are the same calls, on the same shapes, on every run. So the bits don't
//! depend on the number of threads nor on the order in which rows arrive.
//! They can differ between machines, as the matrix product picks its
//! kernel, with or without fused multiply-adds, by the CPU's features.
//!
//! [`Condition`]: einfold_ir::Condition

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow::compute;
use arrow::datatypes::SchemaRef;
use einfold_ir::{Comparison, DimensionId, Operand, Pipeline, Shift};

use crate::dense;
use crate::gemm::{self, Matrix};
use crate::Result;

/// X positions per tile, and J positions per tile.
const XT: usize = 64;
const JT: usize = 64;

/// A pipeline's result, and how many pairs of tiles its masks let it skip.
#[derive(Clone, Debug)]
pub struct Piped {
    /// The result's rows.
    pub batches: Vec<RecordBatch>,
    /// Pairs of an X tile and a J tile that were computed.
    pub tiles_visited: usize,
    /// Every pair of an X tile and a J tile.
    pub tiles: usize,
}

/// How one operand's rows are placed: its dimensions along the matrix's rows
/// and along its columns.
struct Placement<'a> {
    operand: &'a Operand,
    batches: &'a [RecordBatch],
    rows: &'a [usize],
    columns: &'a [usize],
}

/// `pipeline` over its operands `q`, `k` and values `v`, by the fused
/// algorithm on `threads` threads, or `None` when its requirements don't
/// hold.
pub(crate) fn pipeline(
    pipeline: &Pipeline,
    [q, k, v]: [&[RecordBatch]; 3],
    schema: SchemaRef,
    threads: usize,
) -> Result<Option<Piped>> {
    let producer = pipeline.producer();
    let [op_q, op_k] = producer.operands() else {
        return Ok(None);
    };
    let values = pipeline.values();
    let own = producer.dimensions().len();
    let n = own + pipeline.dimensions().len();
    let holds = |op: &Operand, d: usize| op.column_of(DimensionId(d)).is_some();
    let output = |d: usize| pipeline.outputs().contains(&DimensionId(d));
    let (mut xs, mut es, mut js) = (Vec::new(), Vec::new(), Vec::new());
    for d in 0..own {
        let produced = producer.output_of(DimensionId(d)).is_some();
        match (
            holds(op_q, d),
            holds(op_k, d),
            produced,
            output(d),
            holds(values, d),
        ) {
            (true, false, true, true, false) => xs.push(d),
            (true, true, false, false, false) => es.push(d),
            (false, true, true, false, true) => js.push(d),
            _ => return Ok(None),
        }
    }
    let ys: Vec<usize> = (own..n).collect();
    // A condition between the one X and the one J dimension, as `x <op> j`.
    let mask = match producer.conditions() {
        [] => None,
        [c] => match (&xs[..], &js[..]) {
            ([x], [j]) if (c.left.0, c.right.0) == (*x, *j) => Some(c.comparison),
            ([x], [j]) if (c.left.0, c.right.0) == (*j, *x) => match c.comparison {
                Comparison::Less => Some(Comparison::Greater),
                Comparison::LessOrEqual => Some(Comparison::GreaterOrEqual),
                Comparison::Greater => Some(Comparison::Less),
                Comparison::GreaterOrEqual => Some(Comparison::LessOrEqual),
                _ => return Ok(None),
            },
            _ => return Ok(None),
        },
        _ => return Ok(None),
    };

    // Each batch's keys as positions, and each dimension's extent.
    let placements = [
        Placement {
            operand: op_q,
            batches: q,
            rows: &xs,
            columns: &es,
        },
        // k is placed transposed, [e][j], so that the scores are q · k.
        Placement {
            operand: op_k,
            batches: k,
            rows: &es,
            columns: &js,
        },
        Placement {
            operand: values,
            batches: v,
            rows: &js,
            columns: &ys,
        },
    ];
    let mut extent = vec![0usize; n];
    let mut all_keys = Vec::new();
    for p in &placements {
        let batch_keys = dense::par_tasks(p.batches.len(), threads, |i| {
            (p.operand.keys.iter())
                .map(|key| {
                    let (k, n) = dense::keys(p.batches[i].column(key.column.0))?;
                    Some((key.dimension.0, k, n))
                })
                .collect::<Option<Vec<_>>>()
        });
        let Some(batch_keys) = batch_keys.into_iter().collect::<Option<Vec<_>>>() else {
            return Ok(None);
        };
        for &(d, _, n) in batch_keys.iter().flatten() {
            extent[d] = extent[d].max(n);
        }
        all_keys.push(batch_keys);
    }
    let size = |dims: &[usize]| {
        dims.iter()
            .try_fold(1usize, |n, &d| n.checked_mul(extent[d]))
    };
    let (Some(nx), Some(ne), Some(nj), Some(ny)) = (size(&xs), size(&es), size(&js), size(&ys))
    else {
        return Ok(None);
    };
    let mut matrices = Vec::new();
    for (p, batch_keys) in placements.iter().zip(all_keys) {
        let keys: Vec<Vec<(usize, Int64Array)>> = (batch_keys.into_iter())
            .map(|keys| keys.into_iter().map(|(d, k, _)| (d, k)).collect())
            .collect();
        let (Some(rows), Some(columns)) = (size(p.rows), size(p.columns)) else {
            return Ok(None);
        };
        let Some(matrix) = place(p, &keys, &extent, (rows, columns), threads) else {
            return Ok(None);
        };
        matrices.push(matrix);
    }
    let [q, kt, v] = &matrices[..] else {
        unreachable!("three placements");
    };
    if nx
        .checked_mul(ny)
        .is_none_or(|cells| cells >= u32::MAX as usize)
    {
        return Ok(None);
    }

    // Each X position's allowed J positions, an interval.
    let allowed = |x: usize| -> (usize, usize) {
        match mask {
            None => (0, nj),
            Some(Comparison::GreaterOrEqual) => (0, (x + 1).min(nj)),
            Some(Comparison::Greater) => (0, x.min(nj)),
            Some(Comparison::LessOrEqual) => (x.min(nj), nj),
            Some(_) => ((x + 1).min(nj), nj),
        }
    };
    let (x_tiles, j_tiles) = (nx.div_ceil(XT), nj.div_ceil(JT));
    let tiles = dense::par_tasks(x_tiles, threads, |t| {
        tile(pipeline, t, (nx, ne, nj, ny), [q, kt, v], &allowed)
    });
    let tiles_visited = tiles.iter().map(|t| t.visited).sum();

    // The rows: every y for each x with an allowed J position, in position
    // order.
    let rows: Vec<usize> = (0..nx)
        .filter(|&x| {
            nj > 0 && {
                let (lo, hi) = allowed(x);
                lo < hi
            }
        })
        .collect();
    let total = rows.len() * ny;
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(pipeline.outputs().len() + 1);
    for &DimensionId(d) in pipeline.outputs() {
        // The key's place in x's or y's position, first dimension slowest.
        let (in_x, role) = match xs.contains(&d) {
            true => (true, &xs),
            false => (false, &ys),
        };
        let slot = role.iter().position(|&e| e == d).expect("an output's role");
        let stride: usize = role[slot + 1..].iter().map(|&e| extent[e]).product();
        let key_at = |p: usize| ((p / stride) % extent[d]) as i64;
        let mut key = Vec::with_capacity(total);
        match in_x {
            true => rows
                .iter()
                .for_each(|&x| key.extend(std::iter::repeat_n(key_at(x), ny))),
            false => {
                let along: Vec<i64> = (0..ny).map(key_at).collect();
                rows.iter().for_each(|_| key.extend_from_slice(&along));
            }
        }
        let ty = schema.field(columns.len()).data_type();
        columns.push(compute::cast(
            &(Arc::new(Int64Array::from(key)) as ArrayRef),
            ty,
        )?);
    }
    let mut out = Vec::with_capacity(total);
    for &x in &rows {
        let state = &tiles[x / XT];
        out.extend_from_slice(&state.out[(x % XT) * ny..(x % XT + 1) * ny]);
    }
    columns.push(Arc::new(Float64Array::from(out)));
    Ok(Some(Piped {
        batches: vec![RecordBatch::try_new(schema, columns)?],
        tiles_visited,
        tiles: x_tiles * j_tiles,
    }))
}

/// `p`'s rows as a dense `rows × columns` matrix of its factors, or `None`
/// unless it has exactly one row, with a non-NULL factor, at every position.
fn place(
    p: &Placement,
    keys: &[Vec<(usize, Int64Array)>],
    extent: &[usize],
    (rows, columns): (usize, usize),
    threads: usize,
) -> Option<Vec<f64>> {
    let cells = rows.checked_mul(columns)?;
    let total: usize = p.batches.iter().map(|b| b.num_rows()).sum();
    if total != cells || cells >= u32::MAX as usize {
        return None;
    }
    let factor = p.operand.factor?;
    let positions = dense::par_tasks(p.batches.len(), threads, |i| {
        let n = p.batches[i].num_rows();
        let row = dense::positions(&keys[i], p.rows, extent, n);
        let column = dense::positions(&keys[i], p.columns, extent, n);
        (row, column)
    });
    let mut matrix = vec![0.0; cells];
    let mut held = vec![false; cells];
    for (batch, (row, column)) in p.batches.iter().zip(&positions) {
        let factors = batch
            .column(factor.0)
            .as_any()
            .downcast_ref::<Float64Array>()?;
        if factors.null_count() > 0 {
            return None;
        }
        for ((&r, &c), &x) in row.iter().zip(column).zip(factors.values()) {
            let cell = r as usize * columns + c as usize;
            if std::mem::replace(&mut held[cell], true) {
                return None;
            }
            matrix[cell] = x;
        }
    }
    // As many rows as cells, none repeated: every cell is held.
    Some(matrix)
}

/// One X tile's averages, `[x][y]` over its positions, and how many J tiles
/// it visited.
struct Tile {
    out: Vec<f64>,
    visited: usize,
}

/// X tile `t` of `pipeline`, from `q [x][e]`, `kt [e][j]` and `v [j][y]`,
/// with `allowed(x)` the interval of J positions that position `x` may join.
fn tile(
    pipeline: &Pipeline,
    t: usize,
    (nx, ne, nj, ny): (usize, usize, usize, usize),
    [q, kt, v]: [&[f64]; 3],
    allowed: &(impl Fn(usize) -> (usize, usize) + Sync),
) -> Tile {
    let (x0, x1) = (t * XT, ((t + 1) * XT).min(nx));
    let rows = x1 - x0;
    let scale = pipeline.scale();
    let shifted = pipeline.shift() == Shift::KeptMaximum;
    let mut m = vec![if shifted { f64::NEG_INFINITY } else { 0.0 }; rows];
    let mut z = vec![0.0; rows];
    let mut a = vec![0.0; rows * ny];
    let mut block = vec![0.0; XT * JT];
    let mut visited = 0;
    for j0 in (0..nj).step_by(JT) {
        let width = JT.min(nj - j0);
        // Each row's allowed positions within this tile, relative to j0.
        let spans: Vec<(usize, usize)> = (x0..x1)
            .map(|x| {
                let (lo, hi) = allowed(x);
                (lo.clamp(j0, j0 + width) - j0, hi.clamp(j0, j0 + width) - j0)
            })
            .collect();
        if spans.iter().all(|&(lo, hi)| lo >= hi) {
            continue;
        }
        visited += 1;
        let full = spans.iter().all(|&span| span == (0, width));
        let block = &mut block[..rows * width];
        block.fill(0.0);
        // The producer's values for the pair: q[x0..x1] · kt[.., j0..j0+width].
        let q_tile = Matrix {
            data: &q[x0 * ne..],
            stride: ne,
        };
        let k_tile = Matrix {
            data: &kt[j0..],
            stride: nj,
        };
        gemm::multiply_add((rows, ne, width), q_tile, k_tile, block, width);
        for (r, &(lo, hi)) in spans.iter().enumerate() {
            if lo >= hi {
                continue;
            }
            let row = &mut block[r * width..(r + 1) * width];
            let scores = &mut row[lo..hi];
            for s in scores.iter_mut() {
                *s = scale.apply(*s);
            }
            // The largest score so far, or NaN once any score is NaN: its
            // weight is NaN, and so is every average of its row, in SQL too.
            let shift = match shifted {
                true => (scores.iter()).fold(m[r], |m, &s| if s > m || s.is_nan() { s } else { m }),
                false => 0.0,
            };
            if shift == f64::NEG_INFINITY {
                // Every score so far is -inf, so every weight is exp(-inf -
                // shift) for the final shift, or NaN if that is -inf too:
                // 0 here, rescaled to NaN below if it stays so.
                scores.fill(0.0);
                continue;
            }
            let rescale = (m[r] - shift).exp();
            let mut sum = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - shift).exp();
                sum += *s;
            }
            if shifted && rescale != 1.0 {
                z[r] *= rescale;
                a[r * ny..(r + 1) * ny]
                    .iter_mut()
                    .for_each(|a| *a *= rescale);
            }
            z[r] += sum;
            m[r] = shift;
        }
        // The weighted values: a += w · v[j0..j0+width], skipping disallowed
        // pairs, which have no row.
        let v_tile = Matrix {
            data: &v[j0 * ny..],
            stride: ny,
        };
        if full {
            let w = Matrix {
                data: block,
                stride: width,
            };
            gemm::multiply_add((rows, width, ny), w, v_tile, &mut a, ny);
        } else {
            for (r, &(lo, hi)) in spans.iter().enumerate() {
                let w = Matrix {
                    data: &block[r * width + lo..],
                    stride: width,
                };
                let v_rows = Matrix {
                    data: &v[(j0 + lo) * ny..],
                    stride: ny,
                };
                let hi = hi.max(lo);
                gemm::multiply_add((1, hi - lo, ny), w, v_rows, &mut a[r * ny..], ny);
            }
        }
    }
    // A row whose every score was -inf has z = 0, and 0 / 0 is NaN, as
    // SQL's exp(-inf - -inf) is.
    for (r, z) in z.iter().enumerate() {
        a[r * ny..(r + 1) * ny].iter_mut().for_each(|a| *a /= z);
    }
    Tile { out: a, visited }
}
