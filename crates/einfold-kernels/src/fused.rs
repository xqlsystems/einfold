// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! EinFold's fused pipeline algorithm: a [`Pipeline`] computed tile by tile,
//! so that the producer's results never exist in full.
//!
//! It is loop fusion: the producer's loop over its groups and the consumer's
//! loop over the values it sums share one loop over pairs of tiles. That
//! applies to any pipeline whose producer groups pair the consumer's groups
//! with the dimensions it sums away, whatever the query means. Two such
//! queries: attention, in which each query position averages the values at
//! the key positions, weighted by a softmax of its dot products with the
//! keys (each exponentiated, then divided by their sum, so that the weights
//! sum to one); and a kernel smoother, which averages values at source
//! points, weighted by a function of their distance to each target point.
//! The names below follow attention, where they are familiar: the operands
//! `q`, `k` and `v`, and a query, a key and a feature.
//!
//! Every dimension has one of four roles:
//!
//! - **X**: a producer output held by its first operand, `q`, kept by the
//!   pipeline (in attention, the query `i`; in a kernel smoother, the target
//!   point);
//! - **E**: summed by the producer, held by both its operands (a feature, or
//!   a coordinate);
//! - **J**: a producer output held by its second operand, `k`, summed by the
//!   consumer and joined with the values operand `v` (a key, or a source
//!   point);
//! - **Y**: the pipeline's own dimensions, held by `v` (a value's feature).
//!
//! As in the dense fold, keys are positions, each role's dimensions are
//! flattened into one position, and every operand is placed in a dense
//! matrix. For each tile of X positions, on one thread, the J positions are
//! visited in tiles, in order. Each pair of tiles computes its block of
//! producer values, turns them into scores and weights, and folds the
//! weights into the tile's state, then the next pair reuses the block.
//! Memory beyond inputs and output is one block per thread.
//!
//! **The producer's block.** A `SUM` of products is a matrix product over E.
//! A `SUM` of a row expression, such as a Gaussian kernel's `-dist²` between
//! a target and a source point, is the expression evaluated over the block's
//! pairs at each E position, summed in position order; with no E dimension,
//! it is the expression itself, with each pair's bits as SQL computes them.
//! Its columns are placed as a product's factors are, so each pair has
//! exactly one row: two would be two weights `exp(s)` in SQL, not one
//! `exp(2s)`.
//!
//! **Weights.** A weight of an expression of the score (see [`Weight::Of`])
//! is evaluated over each row's scores of the block, and a row's state is
//! plain sums, `z = Σ w` and `a[y] = Σ w·v[y]`. A softmax's weight is
//! `exp(score)`, computed in a scaled accumulator, below. Either way, the
//! weighted sums of a block are a matrix product of its weights and `v`'s
//! rows, and the result is `a / z`, or `a` itself when nothing normalizes
//! it ([`Normalizer::None`]).
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
//! **Masks are absent rows.** A [`Condition`] between the X and J dimensions,
//! `x <op> j + offset`, allows for each X position an interval of J
//! positions, since keys are positions; several conditions, such as a
//! sliding window's two, allow the intervals' intersection. A pair of tiles
//! the conditions disallow entirely is never visited. In a pair they disallow
//! in part, disallowed pairs are skipped, not given weight 0: `0 · NaN` is
//! NaN, but SQL forms no row for them. An X position with no allowed J
//! position has no rows in the output, as in SQL.
//!
//! **When SQL and dense arrays agree.** Every operand must be complete: one
//! row at every position of its dimensions, and no NULL in a column read. Then every
//! producer group exists unless masked, both [`Normalizer`]s agree, and NaN
//! and infinities propagate as SQL's arithmetic does. Otherwise the caller
//! runs something else, such as the host's plan.
//!
//! **Determinism.** Tiles are fixed by the shape, each X tile is computed by
//! one thread, J tiles are visited in order, and each tile's matrix products
//! and expressions are the same calls, on the same shapes, on every run. So the bits don't
//! depend on the number of threads nor on the order in which rows arrive.
//! They can differ between machines, as the matrix product picks its
//! kernel, with or without fused multiply-adds, by the CPU's features.
//!
//! [`Condition`]: einfold_ir::Condition

use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int64Array, RecordBatch};
use arrow::compute;
use arrow::datatypes::SchemaRef;
use einfold_ir::{
    ColumnIndex, Comparison, DimensionId, Expression, Normalizer, Operand, Pipeline, RowValue,
    Scale, Shift, Weight,
};

use crate::dense;
use crate::expression::{self, Lanes};
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
    /// The column placed.
    column: ColumnIndex,
}

/// How a pair of tiles computes its block of producer values.
enum Scores<'a> {
    /// A product's: `q [x][e]` times `kt [e][j]`.
    Product { q: &'a [f64], kt: &'a [f64] },
    /// A row expression's, from the columns it reads, each placed as `q`'s
    /// or `kt`'s factor would be, by its operand, summed over E.
    Expression {
        e: &'a Expression,
        columns: Vec<((usize, ColumnIndex), &'a [f64])>,
    },
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
    if !matches!(pipeline.weight(), Weight::Exp { .. } | Weight::Of(_)) {
        return Ok(None);
    }
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
    // The output has a row for each position of the pipeline's own
    // dimensions, so one left out of the outputs, to be summed away, isn't
    // this algorithm's.
    if !ys.iter().all(|&d| output(d)) {
        return Ok(None);
    }
    // The conditions, each between the one X and the one J dimension, as
    // `x <op> j + offset`.
    let mut mask = Vec::new();
    for c in producer.conditions() {
        let condition = match (&xs[..], &js[..]) {
            // The four comparisons `allowed` knows; one added later declines.
            ([x], [j]) if (c.left.0, c.right.0) == (*x, *j) => match c.comparison {
                Comparison::Less
                | Comparison::LessOrEqual
                | Comparison::Greater
                | Comparison::GreaterOrEqual => *c,
                _ => return Ok(None),
            },
            ([x], [j]) if (c.left.0, c.right.0) == (*j, *x) => match c.mirrored() {
                Some(m) => m,
                None => return Ok(None),
            },
            _ => return Ok(None),
        };
        mask.push(condition);
    }

    // The producer's columns: each operand's factor for a product, or every
    // column its expression reads.
    let read: Vec<(usize, ColumnIndex)> = match producer.row() {
        RowValue::Product => match (op_q.factor, op_k.factor) {
            (Some(a), Some(b)) => vec![(0, a), (1, b)],
            _ => return Ok(None),
        },
        RowValue::Expression(e) => {
            let mut read = e.columns();
            read.sort();
            read.dedup();
            // Only the operands an expression reads are placed, and placing
            // an operand is what checks that it has every row. One that
            // reads neither operand, or one only, isn't this algorithm's.
            if !(0..2).all(|o| read.iter().any(|&(operand, _)| operand == o)) {
                return Ok(None);
            }
            read
        }
        _ => return Ok(None),
    };
    let Some(factor) = values.factor else {
        return Ok(None);
    };
    // Each batch's keys as positions, and each dimension's extent. q's
    // columns are placed [x][e], and k's transposed, [e][j], so that a
    // product's scores are q · k; the values [j][y], last.
    let mut placements: Vec<Placement> = (read.iter())
        .map(|&(o, column)| match o {
            0 => Placement {
                operand: op_q,
                batches: q,
                rows: &xs,
                columns: &es,
                column,
            },
            _ => Placement {
                operand: op_k,
                batches: k,
                rows: &es,
                columns: &js,
                column,
            },
        })
        .collect();
    placements.push(Placement {
        operand: values,
        batches: v,
        rows: &js,
        columns: &ys,
        column: factor,
    });
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
    let v = matrices.pop().expect("the values' placement");
    let scores = match producer.row() {
        RowValue::Expression(e) => Scores::Expression {
            e,
            columns: read
                .iter()
                .copied()
                .zip(matrices.iter().map(Vec::as_slice))
                .collect(),
        },
        _ => Scores::Product {
            q: &matrices[0],
            kt: &matrices[1],
        },
    };
    let v = &v;
    if nx
        .checked_mul(ny)
        .is_none_or(|cells| cells >= u32::MAX as usize)
    {
        return Ok(None);
    }

    // Each X position's allowed J positions, an interval: the intersection
    // of each condition's. `x >= j + c` allows `j <= x - c`, `x < j + c`
    // allows `j > x - c`, and so on, worked in i128 so that no offset
    // overflows, then clamped to the J positions.
    let allowed = |x: usize| -> (usize, usize) {
        let x = i128::try_from(x).expect("a position fits in i128");
        let n = i128::try_from(nj).expect("an extent fits in i128");
        let (mut lo, mut hi) = (0i128, n);
        for condition in &mask {
            let bound = x - i128::from(condition.offset);
            match condition.comparison {
                Comparison::GreaterOrEqual => hi = hi.min(bound + 1),
                Comparison::Greater => hi = hi.min(bound),
                Comparison::LessOrEqual => lo = lo.max(bound),
                _ => lo = lo.max(bound + 1),
            }
        }
        let clamp = |p: i128| usize::try_from(p.clamp(0, n)).expect("clamped to the extent");
        (clamp(lo), clamp(hi.max(lo)))
    };
    let (x_tiles, j_tiles) = (nx.div_ceil(XT), nj.div_ceil(JT));
    let tiles = dense::par_tasks(x_tiles, threads, |t| {
        tile(pipeline, t, (nx, ne, nj, ny), (&scores, v), &allowed)
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

/// `p`'s rows as a dense `rows × columns` matrix of its column, or `None`
/// unless it has exactly one row, with a non-NULL value, at every position.
fn place(
    p: &Placement,
    keys: &[Vec<(usize, Int64Array)>],
    extent: &[usize],
    (rows, columns): (usize, usize),
    threads: usize,
) -> Option<Vec<f64>> {
    use arrow::array::Array;

    let cells = rows.checked_mul(columns)?;
    let total: usize = p.batches.iter().map(|b| b.num_rows()).sum();
    if total != cells || cells >= u32::MAX as usize {
        return None;
    }
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
            .column(p.column.0)
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

/// X tile `t` of `pipeline`, from its `scores` and `v [j][y]`, with
/// `allowed(x)` the interval of J positions that position `x` may join.
fn tile(
    pipeline: &Pipeline,
    t: usize,
    (nx, ne, nj, ny): (usize, usize, usize, usize),
    (scores, v): (&Scores, &[f64]),
    allowed: &(impl Fn(usize) -> (usize, usize) + Sync),
) -> Tile {
    let (x0, x1) = (t * XT, ((t + 1) * XT).min(nx));
    let rows = x1 - x0;
    // A softmax's scale and whether it is shifted, or another weight's
    // expression of the score.
    let (scale, shifted, map) = match pipeline.weight() {
        Weight::Exp { scale, shift } => (*scale, *shift == Shift::KeptMaximum, None),
        Weight::Of(e) => (Scale::One, false, Some(e)),
        _ => unreachable!("pipeline() declines other weights"),
    };
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
        match scores {
            // The producer's values for the pair: q[x0..x1] · kt[.., j0..j0+width].
            Scores::Product { q, kt } => {
                block.fill(0.0);
                let q_tile = Matrix {
                    data: &q[x0 * ne..],
                    stride: ne,
                };
                let k_tile = Matrix {
                    data: &kt[j0..],
                    stride: nj,
                };
                gemm::multiply_add((rows, ne, width), q_tile, k_tile, block, width);
            }
            // The expression over the pair's rows at each E position, summed
            // in position order.
            Scores::Expression { e, columns } => {
                for at in 0..ne {
                    let lanes = |o: usize, c: ColumnIndex| {
                        let (_, m) = (columns.iter())
                            .find(|(read, _)| *read == (o, c))
                            .expect("every column read is placed");
                        match o {
                            0 => Lanes::Rows((x0..x1).map(|x| m[x * ne + at]).collect()),
                            _ => Lanes::Columns(m[at * nj + j0..at * nj + j0 + width].to_vec()),
                        }
                    };
                    // The first position replaces the block's, and the rest
                    // add to it.
                    let write = match at {
                        0 => expression::Write::Replace,
                        _ => expression::Write::Add,
                    };
                    expression::evaluate(e, (rows, width), &lanes).write(
                        (rows, width),
                        block,
                        write,
                    );
                }
            }
        }
        for (r, &(lo, hi)) in spans.iter().enumerate() {
            if lo >= hi {
                continue;
            }
            let row = &mut block[r * width..(r + 1) * width];
            let scores = &mut row[lo..hi];
            if let Some(e) = map {
                // Each weight is the expression of its score, and the row's
                // state is a plain sum, `z = Σ w`, `a[y] = Σ w·v[y]`.
                let shape = (1, hi - lo);
                let lanes = |_: usize, _: ColumnIndex| Lanes::Columns(scores.to_vec());
                let weights = expression::evaluate(e, shape, &lanes);
                weights.write(shape, scores, false);
                z[r] += scores.iter().sum::<f64>();
                continue;
            }
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
    // Unnormalized, there is no division to give it, so a shifted row
    // whose every score was -inf is NaN outright.
    match pipeline.normalizer() {
        Normalizer::None => {
            for (r, &m) in m.iter().enumerate() {
                if shifted && m == f64::NEG_INFINITY {
                    a[r * ny..(r + 1) * ny].fill(f64::NAN);
                }
            }
        }
        _ => {
            for (r, z) in z.iter().enumerate() {
                a[r * ny..(r + 1) * ny].iter_mut().for_each(|a| *a /= z);
            }
        }
    }
    Tile { out: a, visited }
}
