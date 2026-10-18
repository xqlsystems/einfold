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
use datafusion::common::{exec_err, internal_err, Result};
use einfold_ir::{Fold, KeyEquality};

/// Each row's key for some columns, as bytes in Arrow's row format, which
/// gives equal keys equal bytes: NULL equals NULL, floats compare by their
/// bits, and a dictionary-encoded value encodes as the value itself. With no
/// columns, every key is empty.
struct Keys(Option<Rows>);

impl Keys {
    fn new(columns: &[ArrayRef]) -> Result<Self> {
        if columns.is_empty() {
            return Ok(Keys(None));
        }
        let fields = columns
            .iter()
            .map(|c| SortField::new(c.data_type().clone()))
            .collect();
        Ok(Keys(Some(
            RowConverter::new(fields)?.convert_columns(columns)?,
        )))
    }

    fn get(&self, row: usize) -> &[u8] {
        self.0.as_ref().map_or(&[], |rows| rows.row(row).data())
    }
}

/// One operand's rows, as the hash algorithm reads them.
struct Side<'a> {
    batch: &'a RecordBatch,
    /// Each row's key for the dimensions both operands hold.
    shared: Keys,
    /// Each row's key for the outputs taken from this operand.
    outputs: Keys,
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
        let Ok(len) = u32::try_from(batch.num_rows()) else {
            return exec_err!("EinFold's hash path takes at most 2³² - 1 rows per operand");
        };
        let op = &fold.operands()[operand];
        let col = |d: usize| -> ArrayRef {
            Arc::clone(batch.column(op.column_of(d).expect("dimension held by this operand")))
        };
        let shared_cols: Vec<ArrayRef> = shared.iter().map(|&d| col(d)).collect();
        let output_cols: Vec<ArrayRef> = outputs.iter().map(|&d| col(d)).collect();
        let Some(values) = batch
            .column(op.value)
            .as_any()
            .downcast_ref::<Float64Array>()
        else {
            return internal_err!("EinFold factors must be Float64");
        };
        // A NULL key in a dimension compared with `=` joins no row. NULL here
        // is logical: a dictionary's NULL value counts, as it does in SQL.
        let strict: Vec<_> = shared
            .iter()
            .zip(&shared_cols)
            .filter(|(&d, _)| fold.dimensions()[d].equality == KeyEquality::Equal)
            .filter_map(|(_, c)| c.logical_nulls())
            .collect();
        let shared = Keys::new(&shared_cols)?;
        let outputs = Keys::new(&output_cols)?;
        // Keys, then factor: NULL first, then by the factor's bits.
        let value = |r: u32| {
            values
                .is_valid(r as usize)
                .then(|| values.value(r as usize).to_bits())
        };
        let mut order: Vec<u32> = (0..len)
            .filter(|&r| strict.iter().all(|n| n.is_valid(r as usize)))
            .collect();
        order.sort_by(|&x, &y| {
            let key = |r: u32| (shared.get(r as usize), outputs.get(r as usize), value(r));
            key(x).cmp(&key(y))
        });
        Ok(Side {
            batch,
            shared,
            outputs,
            values,
            order,
        })
    }
}

/// Small integer ids for distinct keys, in order of first appearance.
#[derive(Default)]
struct Interner<'a>(HashMap<&'a [u8], u32>);

impl<'a> Interner<'a> {
    fn id(&mut self, key: &'a [u8]) -> u32 {
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
    let mut table: HashMap<&[u8], Vec<u32>> = HashMap::new();
    for &r in &b.order {
        table.entry(b.shared.get(r as usize)).or_default().push(r);
    }

    // Probe: stream a's rows in its fixed order; a group is the pair of a's
    // and b's output keys, each interned to an id.
    let (mut a_ids, mut b_ids) = (Interner::default(), Interner::default());
    let b_id: Vec<u32> = (0..b.batch.num_rows())
        .map(|r| b_ids.id(b.outputs.get(r)))
        .collect();
    let mut index: HashMap<(u32, u32), usize> = HashMap::new();
    let mut groups: Vec<Group> = Vec::new();
    for &ar in &a.order {
        let Some(matches) = table.get(a.shared.get(ar as usize)) else {
            continue;
        };
        let aid = a_ids.id(a.outputs.get(ar as usize));
        let av = (a.values.is_valid(ar as usize)).then(|| a.values.value(ar as usize));
        for &br in matches {
            let g = *index.entry((aid, b_id[br as usize])).or_insert_with(|| {
                groups.push(Group {
                    a_row: ar,
                    b_row: br,
                    sum: 0.0,
                    valued: false,
                });
                groups.len() - 1
            });
            if let (Some(av), true) = (av, b.values.is_valid(br as usize)) {
                let g = &mut groups[g];
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
            let fa = Keys::new(&a_outputs.iter().map(|&d| key_col(d)).collect::<Vec<_>>())?;
            let fb = Keys::new(&b_outputs.iter().map(|&d| key_col(d)).collect::<Vec<_>>())?;
            // A fill row whose value is NULL gets NULL. NULL here is logical,
            // so a column of type Null counts.
            let fill_nulls = fill.column(spec.value).logical_nulls();
            let values: Float64Array = (0..fill.num_rows())
                .map(|r| {
                    if fill_nulls.as_ref().is_some_and(|n| n.is_null(r)) {
                        return None;
                    }
                    let group = a_ids.0.get(fa.get(r)).zip(b_ids.0.get(fb.get(r)));
                    let g = group.and_then(|(&x, &y)| index.get(&(x, y)));
                    Some(match g.map(|&g| &groups[g]) {
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
