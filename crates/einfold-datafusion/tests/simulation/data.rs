// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Random tables: `l(a, b, v, t)`, `r(a, b, v, t)` and `d(a, b, v, c)`.
//!
//! Their key columns hold three dimensions, `I`, `J` and `K`: `l.a` and `d.a`
//! hold `I`, `l.b` and `r.a` hold `J`, and `r.b` and `d.b` hold `K`. So
//! `l JOIN r ON l.b = r.a`, grouped by `l.a, r.b`, is a matrix product, and
//! `d` is a relation with the product's keys: a domain. `v` is a `DOUBLE`
//! value. `t` is text that usually reads as a number, so that a query can cast
//! it, and that sometimes doesn't, so that the cast fails on that row. `c` is
//! text for tests in `CASE`.
//!
//! Each dimension's keys have one type, drawn per case: integers of several
//! widths, unsigned keys too large for `BIGINT`, text, dictionary-encoded
//! text, and floats that include `-0.0` beside `0.0` and NaNs of both signs
//! and two payloads. Rows go missing, repeat their keys, have NULL keys, and
//! values that are NULL, NaN, ±inf and `-0.0`. A NULL slot holds a random
//! value underneath, as Arrow allows, so code that reads a slot without
//! checking its validity shows. Rows are shuffled and split into partitions
//! and batches, empty batches included.

use std::fmt::Write as _;
use std::ops::Range;
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, DictionaryArray, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringArray, UInt64Array, UInt8Array,
};
use datafusion::arrow::buffer::NullBuffer;
use datafusion::arrow::compute;
use datafusion::arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;

use einfold_testkit::Rng;

/// How a dimension's keys are stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeyType {
    Int64,
    Int32,
    UInt8,
    /// `UInt64`, whose key 0 is 2⁶³: beyond `BIGINT`.
    Wide,
    Utf8,
    DictUtf8,
    Float64,
    DictFloat64,
}

impl KeyType {
    pub fn draw(rng: &mut Rng) -> Self {
        match rng.below(16) {
            0..=6 => KeyType::Int64,
            7 => KeyType::Int32,
            8 => KeyType::UInt8,
            9 => KeyType::Wide,
            10 | 11 => KeyType::Utf8,
            12 => KeyType::DictUtf8,
            13 | 14 => KeyType::Float64,
            _ => KeyType::DictFloat64,
        }
    }
}

/// Float keys, in the order a dimension's positions take them: the
/// distinctions a host may or may not make between equal-looking keys.
const FLOAT_KEYS: [u64; 7] = [
    0x0000_0000_0000_0000, // 0.0
    0x8000_0000_0000_0000, // -0.0
    0x7ff8_0000_0000_0000, // NaN
    0xfff8_0000_0000_0000, // NaN with the sign bit set
    0x7ff8_0000_0000_0001, // NaN with another payload
    0x4004_0000_0000_0000, // 2.5
    0xbff0_0000_0000_0000, // -1.0
];

/// Key number `i` of a dimension of type `ty`; with `wild`, key 0 is one the
/// dense algorithm can't use as a position (negative, or too large).
fn key_column(rng: &mut Rng, ty: KeyType, keys: &[Option<usize>], wild: bool) -> ArrayRef {
    let nulls = NullBuffer::from(keys.iter().map(Option::is_some).collect::<Vec<_>>());
    let nulls = (nulls.null_count() > 0).then_some(nulls);
    // A value for every slot, a random one under a NULL.
    let int = |rng: &mut Rng, k: &Option<usize>| -> i64 {
        match k {
            Some(0) if wild => -7,
            Some(i) => *i as i64,
            None => rng.below(1000) as i64 - 500,
        }
    };
    match ty {
        KeyType::Int64 => {
            let v: Vec<i64> = keys.iter().map(|k| int(rng, k)).collect();
            Arc::new(Int64Array::new(v.into(), nulls))
        }
        KeyType::Int32 => {
            let v: Vec<i32> = keys.iter().map(|k| int(rng, k) as i32).collect();
            Arc::new(Int32Array::new(v.into(), nulls))
        }
        KeyType::UInt8 => {
            let v: Vec<u8> = (keys.iter())
                .map(|k| match k {
                    Some(0) if wild => 255,
                    _ => int(rng, k).rem_euclid(256) as u8,
                })
                .collect();
            Arc::new(UInt8Array::new(v.into(), nulls))
        }
        KeyType::Wide => {
            let v: Vec<u64> = (keys.iter())
                .map(|k| match k {
                    Some(0) => 1 << 63,
                    _ => int(rng, k) as u64,
                })
                .collect();
            Arc::new(UInt64Array::new(v.into(), nulls))
        }
        KeyType::Float64 | KeyType::DictFloat64 => {
            let v: Vec<f64> = (keys.iter())
                .map(|k| match k {
                    Some(i) => f64::from_bits(FLOAT_KEYS[i % FLOAT_KEYS.len()]),
                    None => rng.below(9) as f64,
                })
                .collect();
            let values = Float64Array::new(v.into(), nulls);
            if ty == KeyType::Float64 {
                return Arc::new(values);
            }
            // One dictionary entry per row, in a shuffled order: the
            // dictionary's layout must not matter.
            let mut order: Vec<i32> = (0..values.len() as i32).collect();
            rng.shuffle(&mut order);
            let dict = datafusion::arrow::compute::take(
                &values,
                &datafusion::arrow::array::UInt32Array::from(
                    order.iter().map(|&i| i as u32).collect::<Vec<_>>(),
                ),
                None,
            )
            .expect("take");
            let mut index = vec![0i32; order.len()];
            for (slot, &row) in order.iter().enumerate() {
                index[row as usize] = slot as i32;
            }
            Arc::new(DictionaryArray::new(Int32Array::from(index), dict))
        }
        KeyType::Utf8 | KeyType::DictUtf8 => {
            let text: StringArray = (keys.iter())
                .map(|k| {
                    k.map(|i| {
                        if wild && i == 0 {
                            "-k".to_string()
                        } else {
                            format!("k{i}")
                        }
                    })
                })
                .collect();
            if ty == KeyType::Utf8 {
                return Arc::new(text);
            }
            // Some NULLs as a NULL dictionary value rather than a NULL index:
            // NULL to SQL, though the index column has no NULL.
            let mut dict: Vec<Option<String>> = vec![None];
            let index: Int32Array = (0..text.len())
                .map(|r| {
                    let v = text.is_valid(r).then(|| text.value(r).to_string());
                    if v.is_none() && rng.chance(0.5) {
                        return None;
                    }
                    let i = dict.iter().position(|d| *d == v).unwrap_or_else(|| {
                        dict.push(v);
                        dict.len() - 1
                    });
                    Some(i as i32)
                })
                .collect();
            Arc::new(DictionaryArray::new(
                index,
                Arc::new(StringArray::from(dict)),
            ))
        }
    }
}

/// What a case's values are like.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Values {
    /// Small multiples of 0.5, with NULL, NaN, ±inf and `-0.0`: every sum of
    /// a few products is exact in any order, so results compare bit for bit.
    Exact,
    /// Ill-conditioned values beside ordinary ones, so that sums depend on
    /// the order of their additions.
    Inexact,
}

fn value(rng: &mut Rng, values: Values) -> Option<f64> {
    match rng.below(100) {
        0..=5 => None,
        6..=8 => Some(f64::NAN),
        9..=10 => Some(f64::INFINITY),
        11 => Some(f64::NEG_INFINITY),
        12..=15 => Some(-0.0),
        _ if values == Values::Exact => Some((rng.below(13) as f64 - 6.0) / 2.0),
        16..=30 => Some(if rng.chance(0.5) { 1e16 } else { -1e16 }),
        _ => {
            let m = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            Some(m * 10f64.powi(rng.below(7) as i32 - 3))
        }
    }
}

/// How a table's rows are drawn.
#[derive(Clone, Copy, Debug)]
pub struct Rows {
    /// The chance that each key tuple has a row.
    pub present: f64,
    /// The chance that a row is repeated, with its own value.
    pub duplicate: f64,
    /// The chance that a key is NULL.
    pub null_key: f64,
    /// Whether key 0 is one the dense algorithm can't use.
    pub wild: bool,
    /// The chance that `t` doesn't read as a number.
    pub bad_text: f64,
}

impl Rows {
    /// Every key tuple once, with no NULL keys: what the dense algorithm takes.
    pub const CLEAN: Rows = Rows {
        present: 1.0,
        duplicate: 0.0,
        null_key: 0.0,
        wild: false,
        bad_text: 0.0,
    };

    pub fn draw(rng: &mut Rng) -> Self {
        if rng.chance(0.4) {
            return Rows {
                present: if rng.chance(0.7) { 1.0 } else { 0.8 },
                ..Rows::CLEAN
            };
        }
        Rows {
            present: *rng.pick(&[1.0, 0.85, 0.6]),
            duplicate: *rng.pick(&[0.0, 0.1, 0.25]),
            null_key: *rng.pick(&[0.0, 0.0, 0.08]),
            wild: rng.chance(0.15),
            bad_text: 0.0,
        }
    }
}

/// One table, as batches split into partitions.
#[derive(Clone)]
pub struct Table {
    pub name: &'static str,
    pub schema: SchemaRef,
    /// All its rows, for reports.
    pub all: RecordBatch,
    /// Its batches, as ranges of `all`'s rows, each with its partition, in
    /// the order the partitions hold them. Some are empty.
    pub slices: Vec<Slice>,
    pub partitions: Vec<Vec<RecordBatch>>,
}

/// One batch of a table: a range of its rows, in a partition.
#[derive(Clone, Debug)]
pub struct Slice {
    pub partition: usize,
    pub rows: Range<usize>,
}

impl Table {
    pub fn register(&self, ctx: &SessionContext) {
        let t = MemTable::try_new(Arc::clone(&self.schema), self.partitions.clone())
            .expect("a valid table");
        ctx.register_table(self.name, Arc::new(t))
            .expect("a new table name");
    }

    /// The table of `all`'s rows, batched as `slices` say.
    pub fn sliced(name: &'static str, all: RecordBatch, slices: Vec<Slice>) -> Table {
        let parts = slices.iter().map(|s| s.partition + 1).max().unwrap_or(1);
        let mut partitions = vec![Vec::new(); parts];
        for s in &slices {
            partitions[s.partition].push(all.slice(s.rows.start, s.rows.len()));
        }
        Table {
            name,
            schema: all.schema(),
            all,
            slices,
            partitions,
        }
    }

    /// The table without the rows in `rows`, each batch keeping the rest of
    /// its own.
    pub fn without(&self, rows: Range<usize>) -> Table {
        let (cut, n) = (rows.len(), self.all.num_rows());
        let kept = [
            self.all.slice(0, rows.start),
            self.all.slice(rows.end, n - rows.end),
        ];
        let all = compute::concat_batches(&self.schema, &kept).expect("same schema");
        let at = |i: usize| match i {
            i if i < rows.start => i,
            i if i < rows.end => rows.start,
            i => i - cut,
        };
        let slices = (self.slices.iter())
            .map(|s| Slice {
                partition: s.partition,
                rows: at(s.rows.start)..at(s.rows.end),
            })
            .collect();
        Table::sliced(self.name, all, slices)
    }

    /// The table in one partition of one batch.
    pub fn in_one_batch(&self) -> Table {
        let rows = 0..self.all.num_rows();
        let slices = vec![Slice { partition: 0, rows }];
        Table::sliced(self.name, self.all.clone(), slices)
    }

    /// The table with its values `v` in `rows` set to 1.
    pub fn with_ones(&self, rows: Range<usize>) -> Table {
        let v = (self.all.column(2).as_any())
            .downcast_ref::<Float64Array>()
            .expect("v");
        let ones: Float64Array = (0..v.len())
            .map(|r| match rows.contains(&r) {
                true => Some(1.0),
                false => v.is_valid(r).then(|| v.value(r)),
            })
            .collect();
        let mut columns: Vec<ArrayRef> = self.all.columns().to_vec();
        columns[2] = Arc::new(ones);
        let all = RecordBatch::try_new(Arc::clone(&self.schema), columns).expect("same schema");
        Table::sliced(self.name, all, self.slices.clone())
    }

    pub fn dump(&self) -> String {
        let shown = pretty_format_batches(std::slice::from_ref(&self.all))
            .map(|t| t.to_string())
            .unwrap_or_else(|e| e.to_string());
        let mut s = String::new();
        let _ = writeln!(s, "{} ({} rows):", self.name, self.all.num_rows());
        for line in shown.lines() {
            let _ = writeln!(s, "  {line}");
        }
        s
    }
}

/// A table `name(a, b, v, <extra>)` with `a`'s keys of type `ta` over
/// `0..ea` and `b`'s of type `tb` over `0..eb`.
#[allow(clippy::too_many_arguments)]
pub fn table(
    rng: &mut Rng,
    name: &'static str,
    (ta, ea): (KeyType, usize),
    (tb, eb): (KeyType, usize),
    rows: Rows,
    values: Values,
    extra: &'static str,
) -> Table {
    let (mut a, mut b, mut v, mut x) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut tuples: Vec<(usize, usize)> = Vec::new();
    for i in 0..ea {
        for j in 0..eb {
            if rng.chance(rows.present) {
                tuples.push((i, j));
                if rng.chance(rows.duplicate) {
                    tuples.push((i, j));
                }
            }
        }
    }
    rng.shuffle(&mut tuples);
    for (i, j) in tuples {
        let key = |k: usize, rng: &mut Rng| (!rng.chance(rows.null_key)).then_some(k);
        a.push(key(i, rng));
        b.push(key(j, rng));
        let val = value(rng, values);
        v.push(val);
        x.push(match extra {
            // Text that reads as the value, or that doesn't read as a number.
            "t" if rng.chance(rows.bad_text) => Some("oops".to_string()),
            "t" => val.filter(|f| f.is_finite()).map(|f| format!("{f}")),
            _ => Some(rng.pick(&["x", "y"]).to_string()),
        });
    }
    let n = v.len();
    let v_nulls = NullBuffer::from(v.iter().map(Option::is_some).collect::<Vec<_>>());
    let v_values: Vec<f64> = (v.iter())
        .map(|x| x.unwrap_or_else(|| rng.below(7) as f64 - 3.0))
        .collect();
    let columns: Vec<ArrayRef> = vec![
        key_column(rng, ta, &a, rows.wild),
        key_column(rng, tb, &b, rows.wild),
        Arc::new(Float64Array::new(
            v_values.into(),
            (v_nulls.null_count() > 0).then_some(v_nulls),
        )),
        Arc::new(StringArray::from(x)),
    ];
    let fields: Vec<Field> = ["a", "b", "v", extra]
        .iter()
        .zip(&columns)
        .map(|(n, c)| Field::new(*n, c.data_type().clone(), true))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let all = RecordBatch::try_new(Arc::clone(&schema), columns).expect("a valid batch");
    let parts = 1 + rng.below(3);
    let mut slices = Vec::new();
    let mut start = 0;
    while start < n {
        let len = rng.below(7).min(n - start);
        slices.push(Slice {
            partition: rng.below(parts),
            rows: start..start + len,
        });
        start += len;
    }
    let mut t = Table::sliced(name, all, slices);
    // Partitions the slices left empty still count.
    t.partitions.resize(parts, Vec::new());
    t
}
