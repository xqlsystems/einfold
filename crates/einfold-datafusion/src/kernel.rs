// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The EinFold hash kernel over Arrow arrays.
//!
//! A fold over a join of two operands, computed without materializing the
//! join: build a hash table on `b`, probe it with `a` row by row, and fold
//! each pair's value straight into its output group's state. SQL would first
//! build every joined row and then aggregate them; here memory is proportional
//! to the inputs and the groups, not to the join.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, Float64Array, Int64Array, UInt64Array};
use datafusion::arrow::compute::take;
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::row::{RowConverter, Rows, SortField};
use datafusion::error::{DataFusionError, Result};
use einfold_ir::{
    Aggregate, Dim, Fold, KeyEquality, Op, Operand, PartialAggregate, RowValue, Value,
};

/// One operand's rows: one array per entry of the operand's dims, in order,
/// plus its value column (Float64, nullable).
pub struct OperandArrays {
    /// The dimension columns, in the order of the operand's `dims`.
    pub dims: Vec<ArrayRef>,
    /// The operand's value, one per row. NULL is a NULL factor.
    pub value: Float64Array,
}

/// The field of a fold's result holding its aggregate's value: named after the
/// aggregate in lower case (`sum`, `count`, `avg`). `SUM` and `AVG` give a
/// nullable Float64; `COUNT` gives a non-null Int64, since SQL's `COUNT` is 0
/// rather than NULL for a group of only NULL values.
pub fn aggregate_field(aggregate: Aggregate) -> Field {
    let name = aggregate.to_string().to_lowercase();
    if aggregate == Aggregate::COUNT {
        Field::new(name, DataType::Int64, false)
    } else {
        Field::new(name, DataType::Float64, true)
    }
}

fn plan_err<T>(msg: String) -> Result<T> {
    Err(DataFusionError::Plan(msg))
}

/// Columns of each dimension of `op`, in order of appearance.
fn columns_by_dim(op: &Operand) -> BTreeMap<&Dim, Vec<usize>> {
    let mut m: BTreeMap<&Dim, Vec<usize>> = BTreeMap::new();
    for (c, d) in op.dims.iter().enumerate() {
        m.entry(d).or_default().push(c);
    }
    m
}

fn rows_of(cols: &[&ArrayRef]) -> Result<Rows> {
    let fields = cols
        .iter()
        .map(|c| SortField::new(c.data_type().clone()))
        .collect();
    let owned: Vec<ArrayRef> = cols.iter().map(|c| Arc::clone(c)).collect();
    Ok(RowConverter::new(fields)?.convert_columns(&owned)?)
}

/// Which rows of an operand can take part in the join: those whose `=` keys
/// (shared dimensions and diagonals) are non-NULL, and whose repeated
/// dimensions agree under the dimension's equality.
fn live_rows(
    fold: &Fold,
    op: &Operand,
    arrays: &OperandArrays,
    shared: &BTreeSet<Dim>,
) -> Result<Vec<bool>> {
    let n = arrays.value.len();
    let mut live = vec![true; n];
    for (d, cols) in columns_by_dim(op) {
        let strict = fold.equality(d) == Some(KeyEquality::Equal);
        if strict && (shared.contains(d) || cols.len() > 1) {
            for &c in &cols {
                let col = &arrays.dims[c];
                for (r, l) in live.iter_mut().enumerate() {
                    *l &= col.is_valid(r);
                }
            }
        }
        // A diagonal: every repeat must agree with the first column.
        for &c in cols.iter().skip(1) {
            let (first, other) = (&arrays.dims[cols[0]], &arrays.dims[c]);
            if first.data_type() != other.data_type() {
                return plan_err(format!("dimension `{d}` has two types in `{}`", op.name));
            }
            let (x, y) = (rows_of(&[first])?, rows_of(&[other])?);
            for (r, l) in live.iter_mut().enumerate() {
                *l &= x.row(r) == y.row(r);
            }
        }
    }
    Ok(live)
}

/// Fold the join of `a` and `b` as `fold` describes (exactly two operands).
///
/// In SQL terms: `SELECT <out>, AGG(a.v * b.v) FROM a JOIN b ON <shared dims>
/// GROUP BY <out>`, where `AGG` is the fold's aggregate (`SUM`, `COUNT` or
/// `AVG`) and each dimension is compared under its [`KeyEquality`]. SQL's `=`
/// never matches NULL; `IS NOT DISTINCT FROM` does. To get `COUNT(*)`, pass
/// value columns that are all 1.0.
///
/// The output has one column per output dimension (typed as in the operand it
/// comes from, `a` first), then the aggregate's value (see [`aggregate_field`]).
/// It has one row per group that some pair of rows reached, as in `GROUP BY`.
/// A group reached only by NULL products still exists: `SUM` and `AVG` are NULL
/// for it, because they skip NULLs and have nothing left, and `COUNT` is 0.
///
/// Repeatable: `a` is probed in order, each match list is in `b`'s order, and
/// groups come out in first-seen order, so equal inputs give identical bits and
/// row order. Floating-point addition is not associative, so a fixed order of
/// additions is what makes the bits repeatable.
pub fn fold_join(fold: &Fold, a: &OperandArrays, b: &OperandArrays) -> Result<RecordBatch> {
    if fold.row_value() != RowValue::Product(Op::Mul) {
        return plan_err("the hash kernel folds products of the two operands' values".into());
    }
    let [opa, opb] = fold.operands() else {
        return plan_err(format!(
            "the hash kernel folds exactly two operands, got {}",
            fold.operands().len()
        ));
    };
    for (op, arrs) in [(opa, a), (opb, b)] {
        let n = arrs.value.len();
        if arrs.dims.len() != op.dims.len() || arrs.dims.iter().any(|c| c.len() != n) {
            return plan_err(format!("arrays of `{}` do not match its dims", op.name));
        }
    }
    let shared = fold.shared();
    let (live_a, live_b) = (
        live_rows(fold, opa, a, &shared)?,
        live_rows(fold, opb, b, &shared)?,
    );
    let (cols_a, cols_b) = (columns_by_dim(opa), columns_by_dim(opb));

    // Join keys: the first column of each shared dimension (BTreeSet order).
    let key_a: Vec<&ArrayRef> = shared.iter().map(|d| &a.dims[cols_a[d][0]]).collect();
    let key_b: Vec<&ArrayRef> = shared.iter().map(|d| &b.dims[cols_b[d][0]]).collect();
    // `RowConverter` encodes NULL as a value equal to NULL, which is right for
    // `IS NOT DISTINCT FROM` but not for `=`. So `live_rows` has already
    // dropped rows with a NULL in an `=` key.
    let (ka, kb) = if shared.is_empty() {
        (None, None)
    } else {
        let conv = RowConverter::new(
            key_a
                .iter()
                .map(|c| SortField::new(c.data_type().clone()))
                .collect(),
        )?;
        let own = |v: &[&ArrayRef]| v.iter().map(|c| Arc::clone(c)).collect::<Vec<_>>();
        (
            Some(conv.convert_columns(&own(&key_a))?),
            Some(conv.convert_columns(&own(&key_b))?),
        )
    };
    let key_bytes = |rows: &Option<Rows>, r: usize| -> Vec<u8> {
        rows.as_ref()
            .map_or_else(Vec::new, |x| x.row(r).as_ref().to_vec())
    };

    // Build: b's row indices per key, in insertion order.
    let mut table: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();
    for r in (0..b.value.len()).filter(|&r| live_b[r]) {
        table.entry(key_bytes(&kb, r)).or_default().push(r);
    }

    // Output dimensions come from `a` when it has them, else from `b`.
    let from_a: Vec<Option<usize>> = fold
        .output()
        .iter()
        .map(|d| cols_a.get(d).map(|c| c[0]))
        .collect();
    let out_a: Vec<&ArrayRef> = from_a.iter().flatten().map(|&c| &a.dims[c]).collect();
    let out_b: Vec<&ArrayRef> = fold
        .output()
        .iter()
        .zip(&from_a)
        .filter(|(_, f)| f.is_none())
        .map(|(d, _)| &b.dims[cols_b[d][0]])
        .collect();
    let group_a = (!out_a.is_empty()).then(|| rows_of(&out_a)).transpose()?;
    let group_b = (!out_b.is_empty()).then(|| rows_of(&out_b)).transpose()?;

    // Probe: `a` in order, matches in `b`'s order. Each group keeps a
    // `PartialAggregate`: the aggregate's running state, plus whether any row
    // reached the group (so a group of only NULL products exists). Groups
    // are numbered in first-seen order, which depends only on the input order
    // and so is repeatable, unlike hash-map iteration order. Each group also
    // remembers one representative (a row, b row) to read its key values from.
    // Extension point: the Gustavson variant of this algorithm handles input
    // that arrives grouped by the output key. It replaces `groups` with dense
    // state for one output row at a time; the hash table on `b` and the
    // `PartialAggregate` updates stay.
    let aggregate = fold.aggregate();
    let mut groups: HashMap<Vec<u8>, usize> = HashMap::new();
    let mut states: Vec<PartialAggregate> = Vec::new();
    let mut reps: Vec<(u64, u64)> = Vec::new();
    let mut key = Vec::new();
    for ra in (0..a.value.len()).filter(|&r| live_a[r]) {
        let Some(matches) = table.get(&key_bytes(&ka, ra)) else {
            continue;
        };
        let va = a.value.is_valid(ra).then(|| a.value.value(ra));
        key.clear();
        let part = key_bytes(&group_a, ra);
        key.extend_from_slice(&(part.len() as u64).to_le_bytes());
        key.extend_from_slice(&part);
        let prefix = key.len();
        for &rb in matches {
            key.truncate(prefix);
            if let Some(g) = &group_b {
                key.extend_from_slice(g.row(rb).as_ref());
            }
            let next = states.len();
            let g = *groups.entry(key.clone()).or_insert(next);
            if g == next {
                states.push(PartialAggregate::new(aggregate));
                reps.push((ra as u64, rb as u64));
            }
            let vb = b.value.is_valid(rb).then(|| b.value.value(rb));
            states[g].update(va.zip(vb).map(|(x, y)| Value::Float(x * y)));
        }
    }

    // Emit: gather each output column at its group's representative row.
    let idx_a = UInt64Array::from_iter_values(reps.iter().map(|r| r.0));
    let idx_b = UInt64Array::from_iter_values(reps.iter().map(|r| r.1));
    let mut fields = Vec::new();
    let mut columns: Vec<ArrayRef> = Vec::new();
    for (d, f) in fold.output().iter().zip(&from_a) {
        let (src, idx) = match f {
            Some(c) => (&a.dims[*c], &idx_a),
            None => (&b.dims[cols_b[d][0]], &idx_b),
        };
        fields.push(Field::new(d.0.clone(), src.data_type().clone(), true));
        columns.push(take(src.as_ref(), idx, None)?);
    }
    let values = states
        .iter()
        .map(|s| s.finish().expect("a group exists only if reached"));
    columns.push(if aggregate == Aggregate::COUNT {
        Arc::new(Int64Array::from_iter_values(values.map(|v| match v {
            Some(Value::Int(c)) => c,
            _ => unreachable!("COUNT finishes as a non-NULL integer"),
        })))
    } else {
        Arc::new(Float64Array::from_iter(values.map(|v| match v {
            None => None,
            Some(Value::Float(f)) => Some(f),
            Some(_) => unreachable!("SUM and AVG of floats finish as floats"),
        })))
    });
    fields.push(aggregate_field(aggregate));
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{AsArray, StringArray};
    use datafusion::arrow::datatypes::{Float64Type, Int64Type};

    /// A key value, compared by bits for floats (so NaN equals NaN).
    #[derive(Clone, Debug, PartialEq)]
    enum Cell {
        I(i64),
        S(String),
        F(u64),
    }
    type Cells = Vec<Option<Cell>>;

    struct Col {
        arr: ArrayRef,
        cells: Cells,
    }
    fn ints(v: &[Option<i64>]) -> Col {
        Col {
            arr: Arc::new(Int64Array::from(v.to_vec())),
            cells: v.iter().map(|x| x.map(Cell::I)).collect(),
        }
    }
    fn strs(v: &[Option<&str>]) -> Col {
        Col {
            arr: Arc::new(StringArray::from(v.to_vec())),
            cells: v.iter().map(|x| x.map(|s| Cell::S(s.into()))).collect(),
        }
    }
    fn floats(v: &[Option<f64>]) -> Col {
        Col {
            arr: Arc::new(Float64Array::from(v.to_vec())),
            cells: v.iter().map(|x| x.map(|f| Cell::F(f.to_bits()))).collect(),
        }
    }
    fn some(v: &[i64]) -> Col {
        ints(&v.iter().map(|&x| Some(x)).collect::<Vec<_>>())
    }
    fn vals(v: &[f64]) -> Vec<Option<f64>> {
        v.iter().map(|&x| Some(x)).collect()
    }

    fn fold_of(a: &[&str], b: &[&str], out: &[&str], nulls_match: &[&str]) -> Fold {
        let dims = |s: &[&str]| s.iter().map(|d| Dim::new(*d)).collect::<Vec<_>>();
        let eq = a
            .iter()
            .chain(b)
            .map(|d| {
                let k = if nulls_match.contains(d) {
                    KeyEquality::NotDistinctFrom
                } else {
                    KeyEquality::Equal
                };
                (Dim::new(*d), k)
            })
            .collect();
        Fold::einsum(
            vec![Operand::new("a", dims(a)), Operand::new("b", dims(b))],
            dims(out),
            eq,
        )
        .unwrap()
    }

    fn cells_of(arr: &ArrayRef) -> Cells {
        (0..arr.len())
            .map(|i| {
                if arr.is_null(i) {
                    None
                } else {
                    Some(match arr.data_type() {
                        DataType::Int64 => Cell::I(arr.as_primitive::<Int64Type>().value(i)),
                        DataType::Utf8 => Cell::S(arr.as_string::<i32>().value(i).into()),
                        DataType::Float64 => {
                            Cell::F(arr.as_primitive::<Float64Type>().value(i).to_bits())
                        }
                        t => panic!("unexpected {t}"),
                    })
                }
            })
            .collect()
    }

    /// A finished value as the tests compare it.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Value {
        Float(Option<f64>),
        Int(i64),
    }
    type Out = Vec<(Cells, Value)>;

    /// The aggregate of one group's products, in SQL terms and independent of
    /// the kernel's `PartialAggregate`: skip NULLs, add in order.
    fn aggregate_of(agg: Aggregate, products: &[Option<f64>]) -> Value {
        let present: Vec<f64> = products.iter().flatten().copied().collect();
        let sum = present.iter().copied().reduce(|x, y| x + y);
        match agg {
            Aggregate::SUM => Value::Float(sum),
            Aggregate::COUNT => Value::Int(present.len() as i64),
            Aggregate::AVG => Value::Float(sum.map(|s| s / present.len() as f64)),
            _ => unreachable!(),
        }
    }

    type Side<'a> = (&'a [Col], &'a [Option<f64>]);

    /// SQL's join + GROUP BY + aggregate as a nested loop, in the kernel's order.
    fn naive(e: &Fold, a: Side, b: Side) -> Out {
        let (opa, opb) = (&e.operands()[0], &e.operands()[1]);
        let mut groups: Vec<(Cells, Vec<Option<f64>>)> = Vec::new();
        for ra in 0..a.1.len() {
            for rb in 0..b.1.len() {
                // Every occurrence of a dimension must match the first.
                let occ = |d: &Dim| -> Vec<&Option<Cell>> {
                    let mut v = vec![];
                    for (op, cols, r) in [(opa, a.0, ra), (opb, b.0, rb)] {
                        for (c, dd) in op.dims.iter().enumerate() {
                            if dd == d {
                                v.push(&cols[c].cells[r]);
                            }
                        }
                    }
                    v
                };
                let ok = e.dims().iter().all(|d| {
                    let o = occ(d);
                    let strict = e.equality(d) == Some(KeyEquality::Equal);
                    o.iter()
                        .all(|x| **x == *o[0] && !(strict && o.len() > 1 && x.is_none()))
                });
                if !ok {
                    continue;
                }
                let key: Cells = e.output().iter().map(|d| occ(d)[0].clone()).collect();
                let at = groups
                    .iter()
                    .position(|(k, _)| *k == key)
                    .unwrap_or_else(|| {
                        groups.push((key, vec![]));
                        groups.len() - 1
                    });
                groups[at].1.push(a.1[ra].zip(b.1[rb]).map(|(x, y)| x * y));
            }
        }
        groups
            .into_iter()
            .map(|(k, products)| (k, aggregate_of(e.aggregate(), &products)))
            .collect()
    }

    fn run(e: &Fold, a: Side, b: Side) -> RecordBatch {
        let arrays = |t: Side| OperandArrays {
            dims: t.0.iter().map(|c| Arc::clone(&c.arr)).collect(),
            value: Float64Array::from(t.1.to_vec()),
        };
        fold_join(e, &arrays(a), &arrays(b)).unwrap()
    }

    fn to_out(batch: &RecordBatch) -> Out {
        let n = batch.num_columns() - 1;
        let last = batch.column(n);
        let keys: Vec<Cells> = (0..n).map(|c| cells_of(batch.column(c))).collect();
        (0..batch.num_rows())
            .map(|r| {
                let v = match last.data_type() {
                    DataType::Int64 => Value::Int(last.as_primitive::<Int64Type>().value(r)),
                    _ => {
                        let f = last.as_primitive::<Float64Type>();
                        Value::Float(f.is_valid(r).then(|| f.value(r)))
                    }
                };
                (keys.iter().map(|k| k[r].clone()).collect(), v)
            })
            .collect()
    }

    /// The same fold with another aggregate.
    fn with_aggregate(e: &Fold, aggregate: Aggregate) -> Fold {
        let equality = e
            .dims()
            .into_iter()
            .map(|d| {
                let k = e.equality(&d).unwrap();
                (d, k)
            })
            .collect();
        Fold::new(
            e.operands().to_vec(),
            e.output().to_vec(),
            equality,
            RowValue::Product(Op::Mul),
            aggregate,
        )
        .unwrap()
    }

    fn same(x: &Value, y: &Value) -> bool {
        match (x, y) {
            (Value::Float(p), Value::Float(q)) => p.map(f64::to_bits) == q.map(f64::to_bits),
            _ => x == y,
        }
    }

    /// For each of SUM, COUNT and AVG, the kernel's output must equal the
    /// reference, bit for bit and in the same row order. Returns the SUM
    /// output as plain floats, for tests that also assert values.
    fn check(e: &Fold, a: Side, b: Side) -> Vec<(Cells, Option<f64>)> {
        let mut sum = vec![];
        for agg in [Aggregate::SUM, Aggregate::COUNT, Aggregate::AVG] {
            let f = with_aggregate(e, agg);
            let batch = run(&f, a, b);
            let last = batch.num_columns() - 1;
            assert_eq!(batch.schema().field(last), &aggregate_field(agg));
            let got = to_out(&batch);
            let want = naive(&f, a, b);
            assert_eq!(got.len(), want.len(), "{agg}: {got:?} vs {want:?}");
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.0, w.0, "{agg}");
                assert!(same(&g.1, &w.1), "{agg}: {got:?} vs {want:?}");
            }
            if agg == Aggregate::SUM {
                sum = got
                    .into_iter()
                    .map(|(k, v)| match v {
                        Value::Float(f) => (k, f),
                        Value::Int(_) => unreachable!(),
                    })
                    .collect();
            }
        }
        sum
    }

    /// The fold's output for one aggregate, as the reference computes it.
    fn want(e: &Fold, agg: Aggregate, a: Side, b: Side) -> Out {
        naive(&with_aggregate(e, agg), a, b)
    }

    #[test]
    fn matrix_product() {
        let e = fold_of(&["i", "k"], &["k", "j"], &["i", "j"], &[]);
        let a = [some(&[0, 0, 1, 1]), some(&[0, 1, 0, 1])];
        let b = [some(&[0, 0, 1, 1]), some(&[0, 1, 0, 1])];
        let out = check(
            &e,
            (&a, &vals(&[1.0, 2.0, 3.0, 4.0])),
            (&b, &vals(&[5.0, 6.0, 7.0, 8.0])),
        );
        let sums: Vec<_> = out.iter().map(|o| o.1.unwrap()).collect();
        assert_eq!(sums, [19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn batched_product() {
        let e = fold_of(&["b", "i", "k"], &["b", "k", "j"], &["b", "i", "j"], &[]);
        let a = [
            some(&[0, 0, 1, 1, 1]),
            some(&[0, 1, 0, 0, 1]),
            some(&[0, 0, 0, 1, 1]),
        ];
        let b = [
            some(&[1, 0, 0, 1]),
            some(&[0, 0, 1, 1]),
            some(&[0, 1, 0, 1]),
        ];
        check(
            &e,
            (&a, &vals(&[1.0, 2.0, 3.0, 4.0, 5.0])),
            (&b, &vals(&[1.5, 2.5, 3.5, 4.5])),
        );
    }

    #[test]
    fn outer_product_and_one_sided_output() {
        let e = fold_of(&["i"], &["j"], &["i", "j"], &[]);
        let a = [some(&[1, 2])];
        let b = [some(&[7, 8, 9])];
        let out = check(&e, (&a, &vals(&[2.0, 3.0])), (&b, &vals(&[1.0, 2.0, 3.0])));
        assert_eq!(out.len(), 6);
        // Output dims from `a` only: j is private to b and summed.
        let e = fold_of(&["i", "k"], &["k", "j"], &["i"], &[]);
        let a = [some(&[1, 1, 2]), some(&[0, 1, 0])];
        let b = [some(&[0, 1, 1]), some(&[5, 5, 6])];
        check(
            &e,
            (&a, &vals(&[1.0, 2.0, 3.0])),
            (&b, &vals(&[1.0, 2.0, 3.0])),
        );
        // Output dims from `b` only, and an empty output (a scalar).
        let e = fold_of(&["k"], &["k", "j"], &["j"], &[]);
        let a = [some(&[0, 1])];
        check(&e, (&a, &vals(&[1.0, 2.0])), (&b, &vals(&[1.0, 2.0, 3.0])));
        let e = fold_of(&["k"], &["k", "j"], &[], &[]);
        let out = check(&e, (&a, &vals(&[1.0, 2.0])), (&b, &vals(&[1.0, 2.0, 3.0])));
        assert_eq!(out, vec![(vec![], Some(11.0))]);
    }

    #[test]
    fn diagonal() {
        let e = fold_of(&["i", "i", "k"], &["k"], &["i"], &[]);
        let a = [
            ints(&[Some(1), Some(1), Some(2), None, None]),
            ints(&[Some(1), Some(2), Some(2), None, Some(3)]),
            some(&[0, 0, 0, 0, 0]),
        ];
        let b = [some(&[0])];
        let out = check(&e, (&a, &vals(&[1.0; 5])), (&b, &vals(&[10.0])));
        // The NULL diagonal never agrees under `=` ...
        assert_eq!(out.len(), 2);
        // ... but does under IS NOT DISTINCT FROM.
        let e = fold_of(&["i", "i", "k"], &["k"], &["i"], &["i"]);
        assert_eq!(
            check(&e, (&a, &vals(&[1.0; 5])), (&b, &vals(&[10.0]))).len(),
            3
        );
    }

    #[test]
    fn null_values_and_all_null_groups() {
        let e = fold_of(&["i", "k"], &["k"], &["i"], &[]);
        let a = [some(&[1, 1, 2, 2, 3]), some(&[0, 0, 0, 0, 9])];
        let av = [Some(1.0), None, None, None, Some(1.0)];
        let b = [some(&[0])];
        let out = check(&e, (&a, &av), (&b, &[Some(2.0)]));
        // Group 1 skips the NULL; group 2 is NULL; group 3 matched nothing.
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].1, Some(2.0));
        assert_eq!(out[1].1, None);
    }

    #[test]
    fn null_keys() {
        let a = [ints(&[Some(1), None, Some(1)]), some(&[5, 6, 7])];
        let b = [ints(&[None, Some(1), Some(1)]), some(&[5, 6, 7])];
        let (av, bv) = (vals(&[1.0, 2.0, 4.0]), vals(&[8.0, 16.0, 32.0]));
        let e = fold_of(&["k", "x"], &["k", "y"], &["x", "y"], &[]);
        assert_eq!(check(&e, (&a, &av), (&b, &bv)).len(), 4);
        let e = fold_of(&["k", "x"], &["k", "y"], &["x", "y"], &["k"]);
        assert_eq!(check(&e, (&a, &av), (&b, &bv)).len(), 5);
        // A NULL in an output-only dimension is its own group, under `=`.
        let e = fold_of(&["k", "x"], &["k"], &["x"], &[]);
        let a = [some(&[1, 1, 1]), ints(&[None, Some(1), None])];
        let b = [some(&[1])];
        let out = check(&e, (&a, &av), (&b, &[Some(2.0)]));
        assert_eq!(
            out,
            vec![
                (vec![None], Some(10.0)),
                (vec![Some(Cell::I(1))], Some(4.0))
            ]
        );
    }

    #[test]
    fn empty_inputs() {
        let e = fold_of(&["i", "k"], &["k", "j"], &["i", "j"], &[]);
        let none = || [some(&[]), some(&[])];
        let some_rows = [some(&[1]), some(&[1])];
        assert!(check(&e, (&none(), &[]), (&some_rows, &[Some(1.0)])).is_empty());
        assert!(check(&e, (&some_rows, &[Some(1.0)]), (&none(), &[])).is_empty());
        let out = run(&e, (&none(), &[]), (&none(), &[]));
        assert_eq!(out.num_rows(), 0);
        assert_eq!(out.schema().field(2).name(), "sum");
    }

    #[test]
    fn duplicates_add() {
        let e = fold_of(&["i", "k"], &["k"], &["i"], &[]);
        let a = [some(&[1, 1, 1]), some(&[0, 0, 0])];
        let b = [some(&[0, 0])];
        let out = check(
            &e,
            (&a, &vals(&[1.0, 2.0, 3.0])),
            (&b, &vals(&[10.0, 100.0])),
        );
        assert_eq!(out, vec![(vec![Some(Cell::I(1))], Some(660.0))]);
    }

    #[test]
    fn nan_propagates() {
        let e = fold_of(&["i", "k"], &["k"], &["i"], &[]);
        let a = [some(&[1, 1, 2]), some(&[0, 0, 0])];
        let b = [some(&[0])];
        let out = check(&e, (&a, &vals(&[1.0, f64::NAN, 2.0])), (&b, &vals(&[1.0])));
        assert!(out[0].1.unwrap().is_nan());
        assert_eq!(out[1].1, Some(2.0));
        // NaN as a join and group key is an ordinary value.
        let e = fold_of(&["k"], &["k"], &["k"], &[]);
        let f = [floats(&[Some(f64::NAN), Some(1.5), None])];
        check(
            &e,
            (&f, &vals(&[1.0, 2.0, 3.0])),
            (&f, &vals(&[1.0, 2.0, 3.0])),
        );
    }

    #[test]
    fn key_types() {
        let e = fold_of(&["s", "f"], &["s", "f"], &["s", "f"], &["s"]);
        let a = [
            strs(&[Some("x"), Some("y"), None, Some("x")]),
            floats(&[Some(0.5), Some(1.5), Some(2.5), Some(0.5)]),
        ];
        let out = check(
            &e,
            (&a, &vals(&[1.0, 2.0, 3.0, 4.0])),
            (&a, &vals(&[1.0, 2.0, 3.0, 4.0])),
        );
        assert_eq!(out.len(), 3);
        let e = fold_of(&["k"], &["k"], &["k"], &[]);
        let s = [strs(&[Some("a"), Some("b")])];
        check(&e, (&s, &vals(&[1.0, 2.0])), (&s, &vals(&[3.0, 4.0])));
        // Mismatched key types are an error, not a wrong answer.
        let i = [some(&[1, 2])];
        let arrays = |c: &[Col]| OperandArrays {
            dims: c.iter().map(|c| Arc::clone(&c.arr)).collect(),
            value: Float64Array::from(vals(&[1.0, 2.0])),
        };
        assert!(fold_join(&e, &arrays(&s), &arrays(&i)).is_err());
    }

    #[test]
    fn count_and_avg_values() {
        let e = fold_of(&["i", "k"], &["k"], &["i"], &[]);
        let a = [some(&[1, 1, 1, 2, 2]), some(&[0, 0, 0, 0, 0])];
        let av = [Some(1.0), None, Some(5.0), None, None];
        let b = [some(&[0])];
        let (a, b) = ((&a[..], &av[..]), (&b[..], &[Some(2.0)][..]));
        check(&e, a, b);
        let key = |i| vec![Some(Cell::I(i))];
        assert_eq!(
            want(&e, Aggregate::COUNT, a, b),
            vec![(key(1), Value::Int(2)), (key(2), Value::Int(0))]
        );
        assert_eq!(
            want(&e, Aggregate::AVG, a, b),
            vec![
                (key(1), Value::Float(Some(6.0))),
                (key(2), Value::Float(None))
            ]
        );
        // COUNT is a non-null Int64 even for the group of only NULLs.
        let out = run(&with_aggregate(&e, Aggregate::COUNT), a, b);
        let counts = out.column(1).as_primitive::<Int64Type>();
        assert_eq!(counts.values().to_vec(), vec![2, 0]);
        assert_eq!(counts.null_count(), 0);
        assert!(!out.schema().field(1).is_nullable());
    }

    #[test]
    fn rejects_non_product_row_values() {
        let e = fold_of(&["k"], &["k"], &["k"], &[]);
        let e = Fold::new(
            e.operands().to_vec(),
            e.output().to_vec(),
            [(Dim::new("k"), KeyEquality::Equal)].into(),
            RowValue::Expr,
            Aggregate::SUM,
        )
        .unwrap();
        let x = OperandArrays {
            dims: vec![Arc::new(Int64Array::from(vec![1]))],
            value: Float64Array::from(vec![1.0]),
        };
        assert!(fold_join(&e, &x, &x).is_err());
    }

    #[test]
    fn deterministic() {
        // Values whose sum depends on the order of addition.
        let e = fold_of(&["i", "k"], &["k", "j"], &["i", "j"], &[]);
        let n = 200;
        let a = [
            some(&(0..n).map(|x| x % 7).collect::<Vec<_>>()),
            some(&(0..n).map(|x| x % 5).collect::<Vec<_>>()),
        ];
        let b = [
            some(&(0..n).map(|x| x % 5).collect::<Vec<_>>()),
            some(&(0..n).map(|x| x % 3).collect::<Vec<_>>()),
        ];
        let av = vals(&(0..n).map(|x| 0.1 * x as f64 + 1e-9).collect::<Vec<_>>());
        let bv = vals(&(0..n).map(|x| 1e8 / (x as f64 + 1.0)).collect::<Vec<_>>());
        let r1 = run(&e, (&a, &av), (&b, &bv));
        let r2 = run(&e, (&a, &av), (&b, &bv));
        assert_eq!(r1, r2);
        let bits = |r: &RecordBatch| {
            r.column(2)
                .as_primitive::<Float64Type>()
                .values()
                .iter()
                .map(|f| f.to_bits())
                .collect::<Vec<_>>()
        };
        assert_eq!(bits(&r1), bits(&r2));
        check(&e, (&a, &av), (&b, &bv));
    }

    #[test]
    fn rejects_wrong_operand_count() {
        let e = Fold::einsum(
            vec![Operand::new("a", [Dim::new("i")])],
            vec![],
            [(Dim::new("i"), KeyEquality::Equal)].into(),
        )
        .unwrap();
        let x = OperandArrays {
            dims: vec![],
            value: Float64Array::from(Vec::<f64>::new()),
        };
        assert!(fold_join(&e, &x, &x).is_err());
    }
}
