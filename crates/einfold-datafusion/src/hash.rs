// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! EinFold's hash algorithm: a two-operand fold computed with hash tables,
//! for keys of any type and any density.
//!
//! It is Gustavson's sparse matrix multiply ("Two Fast Algorithms for Sparse
//! Matrices", 1978) generalized to folds: the second operand is built into a
//! hash table keyed by the join's dimensions, and the first is streamed
//! against it. Each joined pair updates its output group's state directly, so
//! the join's rows never exist. It follows SQL exactly:
//!
//! - **Bag semantics.** Every joined pair counts, duplicate key tuples
//!   included.
//! - **Group existence.** A group exists when any joined pair reaches it,
//!   whatever the factors. Its sum skips a product with a NULL factor, as
//!   `SUM` skips NULL, and is NULL when every product was NULL.
//! - **NULL keys.** A NULL in a dimension compared with `=` joins nothing; one
//!   compared with `IS NOT DISTINCT FROM` joins NULL. Grouping keeps NULL as a
//!   group of its own.
//! - **Keys compare by their bits,** as DataFusion's do: `-0.0` and `0.0` are
//!   different keys, and a NaN key matches the same NaN.
//!
//! **Determinism.** A float sum's bits depend on the order of its additions,
//! and DataFusion doesn't fix the order in which rows arrive. So each operand
//! is visited in an order fixed by its rows' contents, keys then factor, and a
//! group's products are added in that order. The result's bits then depend on
//! the inputs' rows, not on how they were partitioned or batched.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, UInt32Array};
use datafusion::arrow::compute::take;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::row::{RowConverter, Rows, SortField};
use datafusion::common::{internal_err, Result};
use einfold_ir::{Fold, KeyEquality};

/// One operand's rows, as the hash algorithm reads them.
struct Side<'a> {
    batch: &'a RecordBatch,
    /// The operand's columns for the dimensions both operands hold.
    shared: Vec<ArrayRef>,
    /// The operand's columns for the outputs taken from it.
    outputs: Vec<ArrayRef>,
    values: &'a Float64Array,
    /// The rows that can join, in the order fixed by their contents.
    order: Vec<u32>,
}

impl<'a> Side<'a> {
    fn new(
        fold: &Fold,
        operand: usize,
        batch: &'a RecordBatch,
        shared: &[usize],
        outputs: &[usize],
    ) -> Result<Self> {
        let op = &fold.operands()[operand];
        let col = |d: usize| -> ArrayRef {
            Arc::clone(batch.column(op.column_of(d).expect("dimension held by this operand")))
        };
        let shared_cols: Vec<ArrayRef> = shared.iter().map(|&d| col(d)).collect();
        let output_cols: Vec<ArrayRef> = outputs.iter().map(|&d| col(d)).collect();
        let Some(values) = batch.column(op.value).as_any().downcast_ref() else {
            return internal_err!("EinFold factors must be Float64");
        };
        // A NULL key in a dimension compared with `=` joins no row.
        let joins = |row: usize| {
            shared.iter().zip(&shared_cols).all(|(&d, c)| {
                fold.dimensions()[d].equality == KeyEquality::NotDistinct || c.is_valid(row)
            })
        };
        let mut sort_cols = shared_cols.clone();
        sort_cols.extend(output_cols.iter().cloned());
        sort_cols.push(Arc::clone(batch.column(op.value)));
        let rows = convert(&sort_cols)?;
        let mut order: Vec<u32> = (0..batch.num_rows() as u32)
            .filter(|&r| joins(r as usize))
            .collect();
        order.sort_by(|&x, &y| rows.row(x as usize).cmp(&rows.row(y as usize)));
        Ok(Side {
            batch,
            shared: shared_cols,
            outputs: output_cols,
            values,
            order,
        })
    }
}

/// `columns` in Arrow's row format, which compares rows by their bytes: equal
/// keys (NULL equal to NULL, floats by their bits) give equal bytes.
fn convert(columns: &[ArrayRef]) -> Result<Rows> {
    let fields = columns
        .iter()
        .map(|c| SortField::new(c.data_type().clone()))
        .collect();
    Ok(RowConverter::new(fields)?.convert_columns(columns)?)
}

/// Each row's key bytes, or one empty key per row when there are no columns.
fn keys(columns: &[ArrayRef], len: usize) -> Result<Vec<Box<[u8]>>> {
    if columns.is_empty() {
        return Ok(vec![Box::default(); len]);
    }
    let rows = convert(columns)?;
    Ok((0..len).map(|r| rows.row(r).as_ref().into()).collect())
}

/// Small integer ids for distinct keys, in order of first appearance.
#[derive(Default)]
struct Interner(HashMap<Box<[u8]>, u32>);

impl Interner {
    fn id(&mut self, key: Box<[u8]>) -> u32 {
        let next = self.0.len() as u32;
        *self.0.entry(key).or_insert(next)
    }
}

/// One output group: where its keys come from, and its sum so far.
struct Group {
    a_row: u32,
    b_row: u32,
    sum: f64,
    /// Whether any product was non-NULL; otherwise the sum is NULL.
    valued: bool,
}

/// The fold of operands `a` and `b`, with `fill` its fill relation when it
/// has one, as one batch of `schema`.
pub(crate) fn fold(
    fold: &Fold,
    a: &RecordBatch,
    b: &RecordBatch,
    fill: Option<&RecordBatch>,
    schema: SchemaRef,
) -> Result<RecordBatch> {
    let ops = fold.operands();
    let shared: Vec<usize> = (0..fold.dimensions().len())
        .filter(|&d| ops[0].column_of(d).is_some() && ops[1].column_of(d).is_some())
        .collect();
    // Each output is taken from the first operand that holds it.
    let from_a = |d: &&usize| ops[0].column_of(**d).is_some();
    let a_outputs: Vec<usize> = fold.outputs().iter().filter(from_a).copied().collect();
    let b_outputs: Vec<usize> = fold
        .outputs()
        .iter()
        .filter(|d| !from_a(d))
        .copied()
        .collect();
    let a = Side::new(fold, 0, a, &shared, &a_outputs)?;
    let b = Side::new(fold, 1, b, &shared, &b_outputs)?;

    // Build: b's rows by their shared keys, each list in b's fixed order.
    let b_keys = keys(&b.shared, b.batch.num_rows())?;
    let mut table: HashMap<&[u8], Vec<u32>> = HashMap::new();
    for &r in &b.order {
        table.entry(&b_keys[r as usize]).or_default().push(r);
    }

    // Probe: stream a's rows in its fixed order; a group is the pair of a's
    // and b's output keys, each interned to an id.
    let a_keys = keys(&a.shared, a.batch.num_rows())?;
    let a_out = keys(&a.outputs, a.batch.num_rows())?;
    let b_out = keys(&b.outputs, b.batch.num_rows())?;
    let (mut a_ids, mut b_ids) = (Interner::default(), Interner::default());
    let b_id: Vec<u32> = b_out.into_iter().map(|k| b_ids.id(k)).collect();
    let mut index: HashMap<(u32, u32), u32> = HashMap::new();
    let mut groups: Vec<Group> = Vec::new();
    for &ar in &a.order {
        let Some(matches) = table.get(&*a_keys[ar as usize]) else {
            continue;
        };
        let aid = a_ids.id(a_out[ar as usize].clone());
        let av = (a.values.is_valid(ar as usize)).then(|| a.values.value(ar as usize));
        for &br in matches {
            let g = *index.entry((aid, b_id[br as usize])).or_insert_with(|| {
                groups.push(Group {
                    a_row: ar,
                    b_row: br,
                    sum: 0.0,
                    valued: false,
                });
                groups.len() as u32 - 1
            });
            if let (Some(av), true) = (av, b.values.is_valid(br as usize)) {
                let g = &mut groups[g as usize];
                g.sum += av * b.values.value(br as usize);
                g.valued = true;
            }
        }
    }

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(fold.outputs().len() + 1);
    match (fold.fill(), fill) {
        (None, _) => {
            let a_rows = UInt32Array::from_iter_values(groups.iter().map(|g| g.a_row));
            let b_rows = UInt32Array::from_iter_values(groups.iter().map(|g| g.b_row));
            for &d in fold.outputs() {
                let (side, rows) = if from_a(&&d) {
                    (&a, &a_rows)
                } else {
                    (&b, &b_rows)
                };
                let col = side
                    .batch
                    .column(ops[usize::from(!from_a(&&d))].column_of(d).unwrap());
                columns.push(take(col, rows, None)?);
            }
            let values: Float64Array = groups.iter().map(|g| g.valued.then_some(g.sum)).collect();
            columns.push(Arc::new(values));
        }
        (Some(spec), Some(fill)) => {
            // Each fill row's group, found by its keys split as the groups'
            // are: the outputs taken from a, then those taken from b.
            let key_col = |d: usize| {
                let i = fold.outputs().iter().position(|&o| o == d).unwrap();
                Arc::clone(fill.column(spec.keys[i]))
            };
            let n = fill.num_rows();
            let fa = keys(
                &a_outputs.iter().map(|&d| key_col(d)).collect::<Vec<_>>(),
                n,
            )?;
            let fb = keys(
                &b_outputs.iter().map(|&d| key_col(d)).collect::<Vec<_>>(),
                n,
            )?;
            let fill_values = fill.column(spec.value);
            let values: Float64Array = (0..n)
                .map(|r| {
                    if fill_values.is_null(r) {
                        return None;
                    }
                    let group = a_ids.0.get(&fa[r]).zip(b_ids.0.get(&fb[r]));
                    let g = group.and_then(|(&x, &y)| index.get(&(x, y)));
                    Some(match g.map(|&g| &groups[g as usize]) {
                        Some(g) if g.valued => g.sum,
                        _ => 0.0,
                    })
                })
                .collect();
            for &col in &spec.keys {
                columns.push(Arc::clone(fill.column(col)));
            }
            columns.push(Arc::new(values));
        }
        (Some(_), None) => return internal_err!("a fold with a fill was given no fill relation"),
    }
    Ok(RecordBatch::try_new(schema, columns)?)
}
