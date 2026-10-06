// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Result comparison (design §11, "bits versus math").

use datafusion::arrow::array::{Array, ArrayRef, AsArray};
use datafusion::arrow::compute::cast;
use datafusion::arrow::datatypes::{DataType, Float64Type, Int64Type};
use datafusion::arrow::record_batch::RecordBatch;
use std::fmt::{self, Write};

/// One coordinate. `Null` sorts first and equals `Null`, as in `GROUP BY`.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Key {
    Null,
    Int(i64),
    /// A `Float64` key by its bits: DataFusion keys floats bitwise, so `-0.0`
    /// and `0.0` differ and NaN equals NaN (tests/datafusion_findings.rs).
    Float(u64),
    Str(String),
}

impl fmt::Display for Key {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Key::Null => f.write_str("NULL"),
            Key::Int(i) => write!(f, "{i}"),
            Key::Float(b) => write!(f, "{:?}", f64::from_bits(*b)),
            Key::Str(s) => write!(f, "{s:?}"),
        }
    }
}

/// A row: its key columns, and its value (`None` is NULL).
pub(crate) type Row = (Vec<Key>, Option<f64>);

fn keys(col: &ArrayRef) -> Vec<Key> {
    if col.data_type() == &DataType::Int64 {
        let a = col.as_primitive::<Int64Type>();
        (0..a.len())
            .map(|i| {
                a.is_valid(i)
                    .then(|| a.value(i))
                    .map_or(Key::Null, Key::Int)
            })
            .collect()
    } else if col.data_type() == &DataType::Float64 {
        let a = col.as_primitive::<Float64Type>();
        (0..a.len())
            .map(|i| {
                a.is_valid(i)
                    .then(|| a.value(i).to_bits())
                    .map_or(Key::Null, Key::Float)
            })
            .collect()
    } else {
        // Utf8, LargeUtf8, and Utf8View all compare as strings.
        let s = cast(col, &DataType::Utf8).expect("key columns are Int64 or a string type");
        let a = s.as_string::<i32>();
        (0..a.len())
            .map(|i| {
                a.is_valid(i)
                    .then(|| a.value(i).to_string())
                    .map_or(Key::Null, Key::Str)
            })
            .collect()
    }
}

/// Decode a batch whose last column is the `Float64` value `v` and whose other
/// columns are keys.
pub(crate) fn rows(batch: &RecordBatch) -> Vec<Row> {
    let n_keys = batch.num_columns() - 1;
    let key_cols: Vec<Vec<Key>> = batch.columns()[..n_keys].iter().map(keys).collect();
    let v = cast(batch.column(n_keys), &DataType::Float64).expect("v is numeric");
    let v = v.as_primitive::<Float64Type>();
    (0..batch.num_rows())
        .map(|r| {
            let k = key_cols.iter().map(|c| c[r].clone()).collect();
            (k, v.is_valid(r).then(|| v.value(r)))
        })
        .collect()
}

/// Equal values: NULL with NULL, NaN with NaN, otherwise within a relative
/// `1e-9`, since rewrites may change summation order.
fn close(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            a == b || (a.is_nan() && b.is_nan()) || (a - b).abs() <= 1e-9 * a.abs().max(b.abs())
        }
        _ => false,
    }
}

fn show(r: &Row) -> String {
    let ks: Vec<String> = r.0.iter().map(Key::to_string).collect();
    let v = r.1.map_or("NULL".to_string(), |v| format!("{v:?}"));
    format!("({}) -> {v}", ks.join(", "))
}

/// Compare two results as multisets of rows.
///
/// Both batches hold key columns and then a `Float64` value column. Keys must
/// match exactly (NULL equals NULL); values match as [`assert_same_result`]
/// says. Returns a readable diff on mismatch.
pub fn compare(expected: &RecordBatch, actual: &RecordBatch) -> Result<(), String> {
    if expected.num_columns() != actual.num_columns() {
        return Err(format!(
            "column count differs: expected {}, got {}",
            expected.num_columns(),
            actual.num_columns()
        ));
    }
    let mut unused = rows(actual);
    let mut missing = Vec::new();
    for e in rows(expected) {
        match unused.iter().position(|a| a.0 == e.0 && close(a.1, e.1)) {
            Some(i) => {
                unused.swap_remove(i);
            }
            None => missing.push(e),
        }
    }
    if missing.is_empty() && unused.is_empty() {
        return Ok(());
    }
    let mut out = format!(
        "results differ ({} expected rows, {} actual rows)\n",
        expected.num_rows(),
        actual.num_rows()
    );
    missing.sort_by(|a, b| a.0.cmp(&b.0));
    unused.sort_by(|a, b| a.0.cmp(&b.0));
    for r in &missing {
        let _ = writeln!(out, "  - expected, not found: {}", show(r));
    }
    for r in &unused {
        let _ = writeln!(out, "  + unexpected:          {}", show(r));
    }
    Err(out)
}

/// Assert that `actual` equals `expected` as SQL results (design §11).
///
/// Rows compare as a multiset, so order does not matter. Keys compare
/// exactly, with NULL equal to NULL. Values compare to a relative `1e-9`,
/// with NULL equal to NULL and NaN equal to NaN.
///
/// # Panics
///
/// With a diff of the unmatched rows on mismatch.
#[track_caller]
pub fn assert_same_result(expected: &RecordBatch, actual: &RecordBatch) {
    if let Err(diff) = compare(expected, actual) {
        panic!("{diff}");
    }
}
