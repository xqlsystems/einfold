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
//! - **Keys compare as the host's do** (see [`crate::FloatKeys`]).
//! - **A domain is a left join:** each domain row gets the group its keys
//!   match, under each key's equality, or none.
//!
//! **Determinism.** A float sum's bits depend on the order of its additions,
//! and hosts don't fix the order in which rows arrive. So each operand is
//! visited in an order fixed by its rows' contents, keys then factor, and a
//! group's products are added in that order. The result's bits then depend on
//! the inputs' rows, not on how they were batched.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, RecordBatch, UInt32Array};
use arrow::compute::take;
use arrow::datatypes::SchemaRef;
use arrow::error::ArrowError;
use einfold_ir::{DimensionId, Fold, KeyEquality};

use crate::keys::{host_keys, Keys};
use crate::{Host, Result};

/// One operand's rows, as the hash algorithm reads them.
struct Side<'a> {
    /// Each row's key for the dimensions both operands hold.
    shared: Keys,
    /// Each row's key for the outputs taken from this operand.
    outputs: Keys,
    /// The columns of those outputs, as the host compares and shows them.
    output_cols: Vec<ArrayRef>,
    factors: &'a Float64Array,
    /// The rows that can join, in the order fixed by their contents.
    order: Vec<u32>,
}

impl<'a> Side<'a> {
    fn new(
        fold: &Fold,
        operand: usize,
        batch: &'a RecordBatch,
        shared: &[DimensionId],
        outputs: &[DimensionId],
        host: Host,
    ) -> Result<Self> {
        let Ok(len) = u32::try_from(batch.num_rows()) else {
            return Err(ArrowError::ComputeError(
                "EinFold's hash path takes at most 2³² - 1 rows per operand".into(),
            ));
        };
        let op = &fold.operands()[operand];
        let col = |d: DimensionId| -> ArrayRef {
            let c = op.column_of(d).expect("dimension held by this operand");
            Arc::clone(batch.column(c.0))
        };
        let shared_cols = host_keys(&shared.iter().map(|&d| col(d)).collect::<Vec<_>>(), host);
        let output_cols = host_keys(&outputs.iter().map(|&d| col(d)).collect::<Vec<_>>(), host);
        let factors = op
            .factor
            .and_then(|c| batch.column(c.0).as_any().downcast_ref::<Float64Array>())
            .ok_or_else(|| ArrowError::InvalidArgumentError("factors must be Float64".into()))?;
        // A NULL key in a dimension compared with `=` joins no row. NULL here
        // is logical: a dictionary's NULL value counts, as it does in SQL.
        let strict: Vec<_> = shared
            .iter()
            .zip(&shared_cols)
            .filter(|(&d, _)| fold.dimension(d).equality == KeyEquality::Equal)
            .filter_map(|(_, c)| c.logical_nulls())
            .collect();
        let shared = Keys::new(&shared_cols)?;
        let outputs = Keys::new(&output_cols)?;
        // Keys, then factor: NULL first, then by the factor's bits.
        let factor = |r: u32| {
            factors
                .is_valid(r as usize)
                .then(|| factors.value(r as usize).to_bits())
        };
        let mut order: Vec<u32> = (0..len)
            .filter(|&r| strict.iter().all(|n| n.is_valid(r as usize)))
            .collect();
        order.sort_by(|&x, &y| {
            let key = |r: u32| (shared.get(r as usize), outputs.get(r as usize), factor(r));
            key(x).cmp(&key(y))
        });
        Ok(Side {
            shared,
            outputs,
            output_cols,
            factors,
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

/// See [`crate::fold`].
pub(crate) fn fold(
    fold: &Fold,
    a: &RecordBatch,
    b: &RecordBatch,
    domain: Option<&[RecordBatch]>,
    schema: SchemaRef,
    host: Host,
) -> Result<Vec<RecordBatch>> {
    let ops = fold.operands();
    let shared: Vec<DimensionId> = fold
        .dimension_ids()
        .filter(|&d| ops[0].column_of(d).is_some() && ops[1].column_of(d).is_some())
        .collect();
    // Each output is taken from the first operand that holds it: its index
    // in the fold's outputs, split by operand.
    let from_a = |i: &usize| ops[0].column_of(fold.outputs()[*i].dimension).is_some();
    let (a_out, b_out): (Vec<usize>, Vec<usize>) = (0..fold.outputs().len()).partition(from_a);
    let dims = |outs: &[usize]| -> Vec<DimensionId> {
        outs.iter().map(|&i| fold.outputs()[i].dimension).collect()
    };
    let a = Side::new(fold, 0, a, &shared, &dims(&a_out), host)?;
    let b = Side::new(fold, 1, b, &shared, &dims(&b_out), host)?;

    // Build: b's rows by their shared keys, each list in b's fixed order.
    let mut table: HashMap<&[u8], Vec<u32>> = HashMap::new();
    for &r in &b.order {
        table.entry(b.shared.get(r as usize)).or_default().push(r);
    }

    // Probe: stream a's rows in its fixed order; a group is the pair of a's
    // and b's output keys, each interned to an id.
    let (mut a_ids, mut b_ids) = (Interner::default(), Interner::default());
    let b_id: Vec<u32> = (0..b.factors.len())
        .map(|r| b_ids.id(b.outputs.get(r)))
        .collect();
    let mut index: HashMap<(u32, u32), usize> = HashMap::new();
    let mut groups: Vec<Group> = Vec::new();
    for &ar in &a.order {
        let Some(matches) = table.get(a.shared.get(ar as usize)) else {
            continue;
        };
        let aid = a_ids.id(a.outputs.get(ar as usize));
        let av = (a.factors.is_valid(ar as usize)).then(|| a.factors.value(ar as usize));
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
            if let (Some(av), true) = (av, b.factors.is_valid(br as usize)) {
                let g = &mut groups[g];
                g.sum += av * b.factors.value(br as usize);
                g.valued = true;
            }
        }
    }

    let Some(spec) = fold.domain() else {
        let a_rows = UInt32Array::from_iter_values(groups.iter().map(|g| g.a_row));
        let b_rows = UInt32Array::from_iter_values(groups.iter().map(|g| g.b_row));
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(fold.outputs().len() + 1);
        for i in 0..fold.outputs().len() {
            let column = match a_out.iter().position(|&o| o == i) {
                Some(j) => take(&a.output_cols[j], &a_rows, None)?,
                None => {
                    let j = b_out.iter().position(|&o| o == i).expect("from a or b");
                    take(&b.output_cols[j], &b_rows, None)?
                }
            };
            columns.push(column);
        }
        let values: Float64Array = groups.iter().map(|g| g.valued.then_some(g.sum)).collect();
        columns.push(Arc::new(values));
        return Ok(vec![RecordBatch::try_new(schema, columns)?]);
    };
    let Some(domain) = domain else {
        return Err(ArrowError::InvalidArgumentError(
            "a fold with a domain was given no domain relation".into(),
        ));
    };
    // One output batch per domain batch. Each domain row finds its group by
    // its keys, split as the groups' are: the outputs taken from a, then
    // those taken from b. Groups have distinct keys, so it finds at most one.
    domain
        .iter()
        .map(|batch| {
            // Each output's domain key: its column, and how it is matched.
            let key = |i: usize| fold.outputs()[i].domain_key.expect("a fold with a domain");
            let key_cols = |outs: &[usize]| -> Vec<ArrayRef> {
                outs.iter()
                    .map(|&i| Arc::clone(batch.column(key(i).column.0)))
                    .collect()
            };
            let (da, db) = (
                Keys::new(&host_keys(&key_cols(&a_out), host))?,
                Keys::new(&host_keys(&key_cols(&b_out), host))?,
            );
            // A NULL key compared with `=` matches no group.
            let strict: Vec<_> = (0..fold.outputs().len())
                .map(key)
                .filter(|k| k.equality == KeyEquality::Equal)
                .filter_map(|k| batch.column(k.column.0).logical_nulls())
                .collect();
            let values: Float64Array = (0..batch.num_rows())
                .map(|r| {
                    let matched = strict.iter().all(|n| n.is_valid(r));
                    let group = a_ids.0.get(da.get(r)).zip(b_ids.0.get(db.get(r)));
                    let g = group.and_then(|(&x, &y)| index.get(&(x, y)));
                    let value = match g {
                        Some(&g) if matched && groups[g].valued => Some(groups[g].sum),
                        _ => None,
                    };
                    value.or(spec.missing)
                })
                .collect();
            let mut columns: Vec<ArrayRef> = spec
                .carry
                .iter()
                .map(|&c| Arc::clone(batch.column(c.0)))
                .collect();
            columns.push(Arc::new(values));
            RecordBatch::try_new(Arc::clone(&schema), columns)
        })
        .collect()
}
