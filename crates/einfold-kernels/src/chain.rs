// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A fold of three operands, such as a chain of matrix products, computed as
//! two folds of two operands, in the cheaper order.
//!
//! In SQL, the chain `D[i, l] = Σ_{j,k} A[i, j]·B[j, k]·C[k, l]` is
//!
//! ```sql
//! SELECT a.i, c.l, SUM(a.v * b.v * c.v)
//! FROM a JOIN b ON a.j = b.j JOIN c ON b.k = c.k GROUP BY a.i, c.l
//! ```
//!
//! whose join has a row for every `(i, j, k, l)` that joins. Summing `j`
//! away first, as `T[i, k] = Σ_j A[i, j]·B[j, k]`, and then computing
//! `Σ_k T[i, k]·C[k, l]` gives the same groups and, with exact arithmetic, the
//! same sums, without that join: the order of contractions. Which pair to
//! contract first is chosen from the dimensions' extents, as the order whose
//! two products multiply the fewest numbers: `(A·B)·C` or `A·(B·C)`, or, where
//! `A` and `C` share a dimension, `(A·C)·B`.
//!
//! **The same rows as SQL.** A group `(i, k)` of `T` exists exactly when some
//! `j` joins, and its sum is NULL exactly when every such product is, as
//! `SUM` gives. A NULL `T[i, k]` then contributes no product, as each of its
//! rows' products would have been NULL. So the groups, and which are NULL,
//! are the query's.
//!
//! **Floats.** With floats, `T` changes which numbers are added: each sum of
//! `j` is rounded once, and then multiplied, where the query multiplies each
//! product and rounds the sum once. The result stays within the bound that
//! summing in any order has (`n − 1` roundings of the sum of the terms'
//! magnitudes), widened by one rounding of each term, as long as nothing
//! overflows: then the partial sums of both orders are finite. That is
//! checked before anything runs. Every factor must be finite, and the
//! product over the operands of their largest magnitude times their number
//! of rows, which bounds every partial sum's magnitude in either order, must
//! be far below the largest float. Otherwise the chain declines, and the
//! caller runs something else, such as the host's plan. So the chain never
//! turns a NaN into a number, or an overflow into a finite sum.
//!
//! Each of the two folds runs by the dense algorithm, and declines where it
//! does: keys that aren't small non-negative integers, duplicate key tuples
//! within an operand, or holes it can't take.

use arrow::array::{Array, Float64Array};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use einfold_ir::{Aggregate, ColumnIndex, DimensionId, Fold, Key, Operand, Output, RowValue};

use crate::options::Options;
use crate::relation::Relation;
use crate::{dense, Result};

/// The largest bound on a partial sum's magnitude that the chain takes:
/// `2¹⁰⁰⁰`, far enough below the largest float, about `2¹⁰²⁴`, that the
/// rounding of the bound itself can't matter.
const LARGEST_BOUND: f64 = 1.0715086071862673e301;

/// `fold`, a `SUM` of the product of its three operands' factors with no
/// domain and no conditions, over `operands`, as batches of `schema` (which
/// [`schema`](crate::schema) gives), or `None` where the chain declines (see
/// the module's documentation).
pub(crate) fn fold(
    fold: &Fold,
    operands: [Relation; 3],
    schema: SchemaRef,
    options: &Options,
) -> Result<Option<Vec<RecordBatch>>> {
    let chained = fold.operands().len() == 3
        && *fold.row() == RowValue::Product
        && fold.aggregate().is_sum()
        && fold.domain().is_none()
        && fold.conditions().is_empty()
        && fold.operands().iter().all(|op| op.factor.is_some());
    if !chained || !guarded(fold, &operands) {
        return Ok(None);
    }
    let Some(extent) = extents(fold, &operands) else {
        return Ok(None);
    };
    // The pair contracted first: the cheapest of those that share a
    // dimension, the first of equals.
    let cost = |[x, y]: [usize; 2], rest: usize| -> Option<u128> {
        let held = |o: usize| dimensions(&fold.operands()[o]);
        let pair = union(&held(x), &held(y));
        let kept = kept(fold, &pair, rest);
        let product = |dims: &[DimensionId]| {
            (dims.iter()).try_fold(1u128, |n, d| n.checked_mul(extent[d.0] as u128))
        };
        let second = union(&kept, &held(rest));
        product(&pair)?.checked_add(product(&second)?)
    };
    let shares = |x: usize, y: usize| {
        let theirs = dimensions(&fold.operands()[y]);
        dimensions(&fold.operands()[x])
            .iter()
            .any(|d| theirs.contains(d))
    };
    let best = [([0, 1], 2), ([1, 2], 0), ([0, 2], 1)]
        .into_iter()
        .filter(|&([x, y], _)| shares(x, y))
        .filter_map(|(pair, rest)| Some((cost(pair, rest)?, pair, rest)))
        .min_by_key(|&(cost, ..)| cost);
    let Some((_, [x, y], rest)) = best else {
        return Ok(None);
    };
    let Some((first, second)) = steps(fold, [x, y], rest) else {
        return Ok(None);
    };
    let first_schema = crate::schema(&first, &[operands[x].schema, operands[y].schema], None)?;
    let Some(t) = either(
        &first,
        operands[x],
        operands[y],
        first_schema.clone(),
        options,
    )?
    else {
        return Ok(None);
    };
    let t = Relation::new(&first_schema, &t);
    either(&second, t, operands[rest], schema, options)
}

/// `fold` of `a` and `b` by the dense algorithm, or, where it declines, of
/// `b` and `a`: the algorithm takes holes in its second operand only, and
/// either operand may be the one with holes.
fn either(
    fold: &Fold,
    a: Relation,
    b: Relation,
    schema: SchemaRef,
    options: &Options,
) -> Result<Option<Vec<RecordBatch>>> {
    if let Some(batches) = dense::fold(fold, a.batches, b.batches, None, schema.clone(), options)? {
        return Ok(Some(batches));
    }
    let [op_a, op_b] = fold.operands() else {
        return Ok(None);
    };
    let Ok(swapped) = Fold::try_new(
        fold.dimensions().to_vec(),
        vec![op_b.clone(), op_a.clone()],
        fold.outputs().to_vec(),
        RowValue::Product,
        Aggregate::sum(),
        None,
    ) else {
        return Ok(None);
    };
    dense::fold(&swapped, b.batches, a.batches, None, schema, options)
}

/// Whether no partial sum can overflow, in the query's order of additions or
/// the chain's: every non-NULL factor is finite, and the product over the
/// operands of their largest factor's magnitude times their number of rows
/// is at most [`LARGEST_BOUND`]. Every term's magnitude is at most the
/// product of the largest magnitudes, and a group has at most the product of
/// the numbers of rows as terms, so that bounds every sum of terms'
/// magnitudes, and with it every partial sum, of `T` or of the result.
fn guarded(fold: &Fold, operands: &[Relation; 3]) -> bool {
    let mut bound = 1.0f64;
    for (op, relation) in fold.operands().iter().zip(operands) {
        let Some(ColumnIndex(c)) = op.factor else {
            return false;
        };
        let (mut largest, mut rows) = (0.0f64, 0usize);
        for batch in relation.batches {
            let Some(factors) = batch.column(c).as_any().downcast_ref::<Float64Array>() else {
                return false;
            };
            rows += batch.num_rows();
            for v in factors.iter().flatten() {
                if !v.is_finite() {
                    return false;
                }
                largest = largest.max(v.abs());
            }
        }
        bound *= largest * rows as f64;
    }
    bound <= LARGEST_BOUND
}

/// Each dimension's extent, its largest key in any operand plus one, or
/// `None` where a key column isn't small non-negative integers, which the
/// dense algorithm would decline.
fn extents(fold: &Fold, operands: &[Relation; 3]) -> Option<Vec<usize>> {
    let mut extent = vec![0usize; fold.dimensions().len()];
    for (op, relation) in fold.operands().iter().zip(operands) {
        for batch in relation.batches {
            for key in &op.keys {
                let (_, n) = dense::keys(batch.column(key.column.0))?;
                extent[key.dimension.0] = extent[key.dimension.0].max(n);
            }
        }
    }
    Some(extent)
}

/// The dimensions `op` holds, in order.
fn dimensions(op: &Operand) -> Vec<DimensionId> {
    let mut dims: Vec<DimensionId> = op.keys.iter().map(|k| k.dimension).collect();
    dims.sort_unstable();
    dims
}

/// The dimensions in `a` or `b`, in order.
fn union(a: &[DimensionId], b: &[DimensionId]) -> Vec<DimensionId> {
    let mut all: Vec<DimensionId> = a.iter().chain(b).copied().collect();
    all.sort_unstable();
    all.dedup();
    all
}

/// Of the dimensions `pair` of the first two operands, those the first fold
/// keeps: the outputs, and those the operand `rest` joins on. The others
/// are summed away.
fn kept(fold: &Fold, pair: &[DimensionId], rest: usize) -> Vec<DimensionId> {
    let rest = &fold.operands()[rest];
    (pair.iter().copied())
        .filter(|&d| fold.output_of(d).is_some() || rest.column_of(d).is_some())
        .collect()
}

/// The two folds of two operands that compute `fold`: the first of operands
/// `x` and `y`, whose result has one key column per kept dimension (see
/// [`kept`]), in order, then the sum; and the second of that result and
/// operand `rest`, with `fold`'s outputs. Each numbers its own dimensions.
fn steps(fold: &Fold, [x, y]: [usize; 2], rest: usize) -> Option<(Fold, Fold)> {
    let ops = fold.operands();
    // `op` with its dimensions numbered by their place in `local`.
    let renumbered = |op: &Operand, local: &[DimensionId]| -> Option<Operand> {
        let keys = (op.keys.iter())
            .map(|k| {
                let d = local.iter().position(|&l| l == k.dimension)?;
                Some(Key::new(DimensionId(d), k.column))
            })
            .collect::<Option<Vec<_>>>()?;
        Some(Operand {
            keys,
            factor: op.factor,
        })
    };
    let of = |local: &[DimensionId]| local.iter().map(|&d| fold.dimension(d)).collect();
    let output = |local: &[DimensionId], d: DimensionId| -> Option<Output> {
        Some(Output {
            dimension: DimensionId(local.iter().position(|&l| l == d)?),
            domain_key: None,
        })
    };
    let pair = union(&dimensions(&ops[x]), &dimensions(&ops[y]));
    let kept = kept(fold, &pair, rest);
    let first = Fold::try_new(
        of(&pair),
        vec![renumbered(&ops[x], &pair)?, renumbered(&ops[y], &pair)?],
        (kept.iter())
            .map(|&d| output(&pair, d))
            .collect::<Option<_>>()?,
        RowValue::Product,
        Aggregate::sum(),
        None,
    )
    .ok()?;
    let local = union(&kept, &dimensions(&ops[rest]));
    let t = Operand {
        keys: (kept.iter().enumerate())
            .map(|(c, &d)| {
                let d = local.iter().position(|&l| l == d)?;
                Some(Key::new(DimensionId(d), ColumnIndex(c)))
            })
            .collect::<Option<_>>()?,
        factor: Some(ColumnIndex(kept.len())),
    };
    let second = Fold::try_new(
        of(&local),
        vec![t, renumbered(&ops[rest], &local)?],
        (fold.outputs().iter())
            .map(|o| output(&local, o.dimension))
            .collect::<Option<_>>()?,
        RowValue::Product,
        Aggregate::sum(),
        None,
    )
    .ok()?;
    Some((first, second))
}
