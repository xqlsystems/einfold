// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The two oracles: SQL itself, and nested loops.

use crate::compare::{rows, Key, Val};
use crate::generate::Case;
use datafusion::arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;
use einfold_ir::{Aggregate, Dim, KeyEquality, PartialAggregate, Value};
use std::collections::BTreeMap;
use std::sync::Arc;

/// Run `case.sql` in a fresh DataFusion `SessionContext` over `MemTable`s.
///
/// # Panics
///
/// If DataFusion rejects the SQL: that is a harness bug.
pub fn sql_reference(case: &Case) -> RecordBatch {
    let run = async {
        let ctx = SessionContext::new();
        for (op, t) in case.fold.operands().iter().zip(&case.tables) {
            let table = MemTable::try_new(t.schema(), vec![vec![t.clone()]]).expect("table");
            ctx.register_table(op.name.as_str(), Arc::new(table))
                .expect("register");
        }
        let df = ctx.sql(&case.sql).await.expect("plan");
        let schema = Arc::new(df.schema().as_arrow().clone());
        let batches = df.collect().await.expect("execute");
        datafusion::arrow::compute::concat_batches(&schema, &batches).expect("concat")
    };
    // A private thread and runtime, so callers inside a Tokio runtime work too.
    std::thread::scope(|s| {
        s.spawn(|| {
            tokio::runtime::Builder::new_current_thread()
                .build()
                .expect("runtime")
                .block_on(run)
        })
        .join()
        .expect("sql_reference thread")
    })
}

/// Compute the case's result by nested loops over the tables.
///
/// Every combination of one row per table is a candidate joined row. It
/// joins if, for each dimension its rows share, their coordinates match by the
/// dimension's [`KeyEquality`]: `=` never matches NULL, and `IS NOT DISTINCT
/// FROM` matches NULL with NULL. The row's value is the product of its `v`
/// values, NULL if any is NULL. Each joined row updates the
/// [`PartialAggregate`] of the group of its output coordinates (NULL
/// coordinates form one group, as in `GROUP BY`). A group exists only if some
/// joined row reached it. `SUM` and `AVG` of a group whose values are all NULL
/// are NULL, and `COUNT` is 0.
///
/// A fold with no output dimensions is a global aggregate, which SQL answers
/// with one row even when nothing joined (`SUM` and `AVG` NULL, `COUNT` 0).
pub fn naive_reference(case: &Case) -> RecordBatch {
    let e = &case.fold;
    let agg = e.aggregate();
    let tables: Vec<_> = case.tables.iter().map(rows).collect();
    let mut groups: BTreeMap<Vec<Key>, PartialAggregate> = BTreeMap::new();
    let mut stack = vec![(0, BTreeMap::<&Dim, Key>::new(), Some(1.0))];
    while let Some((i, bound, product)) = stack.pop() {
        if i == tables.len() {
            let key = e.output().iter().map(|d| bound[d].clone()).collect();
            groups
                .entry(key)
                .or_insert(PartialAggregate::new(agg))
                .update(product.map(Value::Float));
            continue;
        }
        let dims = &e.operands()[i].dims;
        'rows: for (ks, v) in &tables[i] {
            let mut next = bound.clone();
            for (d, k) in dims.iter().zip(ks) {
                if let Some(old) = next.get(d) {
                    let eq = match e.equality(d) {
                        Some(KeyEquality::Equal) => *old != Key::Null && old == k,
                        _ => old == k,
                    };
                    if !eq {
                        continue 'rows;
                    }
                } else {
                    next.insert(d, k.clone());
                }
            }
            stack.push((
                i + 1,
                next,
                match v {
                    Val::Float(v) => product.map(|p| p * v),
                    _ => None,
                },
            ));
        }
    }
    if e.output().is_empty() {
        groups
            .entry(vec![])
            .or_insert(PartialAggregate::new(agg))
            .update(None);
    }
    let result: Vec<(Vec<Key>, Option<Value>)> = groups
        .into_iter()
        .filter_map(|(k, s)| s.finish().map(|v| (k, v)))
        .collect();
    to_batch(case, &result)
}

fn to_batch(case: &Case, result: &[(Vec<Key>, Option<Value>)]) -> RecordBatch {
    let e = &case.fold;
    let mut fields = Vec::new();
    let mut cols: Vec<ArrayRef> = Vec::new();
    for (c, d) in e.output().iter().enumerate() {
        let ty = key_type(case, d);
        let col: ArrayRef = match ty {
            DataType::Utf8 => Arc::new(
                result
                    .iter()
                    .map(|(k, _)| match &k[c] {
                        Key::Str(s) => Some(s.clone()),
                        _ => None,
                    })
                    .collect::<StringArray>(),
            ),
            DataType::Float64 => Arc::new(
                result
                    .iter()
                    .map(|(k, _)| match &k[c] {
                        Key::Float(b) => Some(f64::from_bits(*b)),
                        _ => None,
                    })
                    .collect::<Float64Array>(),
            ),
            _ => Arc::new(
                result
                    .iter()
                    .map(|(k, _)| match &k[c] {
                        Key::Int(i) => Some(*i),
                        _ => None,
                    })
                    .collect::<Int64Array>(),
            ),
        };
        fields.push(Field::new(&d.0, ty, true));
        cols.push(col);
    }
    // `COUNT` is a non-NULL Int64, as in SQL; `SUM` and `AVG` are nullable floats.
    if e.aggregate() == Aggregate::COUNT {
        let counts = result.iter().map(|r| match r.1 {
            Some(Value::Int(i)) => i,
            _ => panic!("COUNT yields a non-NULL integer"),
        });
        fields.push(Field::new("v", DataType::Int64, false));
        cols.push(Arc::new(counts.collect::<Int64Array>()));
    } else {
        let floats = result.iter().map(|r| match r.1 {
            None => None,
            Some(Value::Float(f)) => Some(f),
            Some(_) => panic!("SUM and AVG of floats yield floats"),
        });
        fields.push(Field::new("v", DataType::Float64, true));
        cols.push(Arc::new(floats.collect::<Float64Array>()));
    }
    RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).expect("columns match")
}

fn key_type(case: &Case, d: &Dim) -> DataType {
    let (_, t) = case
        .fold
        .operands()
        .iter()
        .zip(&case.tables)
        .find(|(o, _)| o.dims.contains(d))
        .expect("dimension in an operand");
    t.schema()
        .field_with_name(&d.0)
        .expect("column")
        .data_type()
        .clone()
}
