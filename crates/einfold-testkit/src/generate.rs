// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The seeded case generator.

use datafusion::arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use einfold_ir::{Aggregate, Dim, Fold, KeyEquality, Op, Operand, RowValue};
use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;

/// splitmix64: tiny, and stable across dependency versions.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in `0..n` (`n > 0`).
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn chance(&mut self, percent: usize) -> bool {
        self.below(100) < percent
    }
}

/// The coordinates of a `Float64` dimension, by index: both zeros and NaN.
const FLOATS: [f64; 6] = [0.0, -0.0, f64::NAN, 1.5, -1.5, 2.0];

/// One generated test: a fold over a join, the tables for its operands, and
/// the SQL query that means the same thing.
///
/// The SQL follows the standard translation of an einsum (Blacher et al.,
/// 2023): list the operands in `FROM`, equate each shared dimension in `JOIN
/// ... ON`, `GROUP BY` the output dimensions, and aggregate the product of the
/// operands' value columns. Here the aggregate is `SUM`, `COUNT`, or `AVG`.
///
/// Table `i` is named after operand `i`.
#[derive(Clone, Debug)]
pub struct Case {
    /// The fold under test: its row value is the product of the operands'
    /// `v` columns, and its aggregate is `SUM`, `COUNT`, or `AVG`.
    pub fold: Fold,
    /// One table per operand: its dimension columns, then `v` (`Float64`).
    pub tables: Vec<RecordBatch>,
    /// `JOIN ... ON` with `=` or `IS NOT DISTINCT FROM`, `GROUP BY` the output
    /// dimensions, and the aggregate of the product of the `v` columns.
    pub sql: String,
}

impl Case {
    /// The case for `seed`: two operands.
    pub fn generate(seed: u64) -> Case {
        Case::generate_with(seed, 2)
    }

    /// The case for `seed` with `n_operands` operands.
    ///
    /// Dimensions have 0 to 6 distinct coordinates and type `Int64`, `Utf8`, or
    /// `Float64` (whose coordinates include `0.0`, `-0.0`, and NaN).
    /// Keys are sometimes NULL; `v` has NULLs and NaN; tables may be empty and
    /// repeat coordinate tuples. With more than two operands, every dimension
    /// uses [`KeyEquality::Equal`], to avoid a DataFusion bug (issues #21 and #22).
    pub fn generate_with(seed: u64, n_operands: usize) -> Case {
        let mut rng = Rng(seed);
        let n_dims = 2 + rng.below(3);
        // Per dimension: its type (0 Int64, 1 Utf8, 2 Float64), extent, and equality.
        let mut pool: Vec<(usize, usize, KeyEquality)> = (0..n_dims)
            .map(|_| {
                let eq = if rng.chance(50) {
                    KeyEquality::Equal
                } else {
                    KeyEquality::NotDistinctFrom
                };
                (rng.below(3), rng.below(7), eq)
            })
            .collect();
        // DataFusion 54.1.0 mis-plans chains of joins that use
        // `IS NOT DISTINCT FROM` (see tests/datafusion_findings.rs). One join
        // is fine, so only up to two operands may use it.
        if n_operands > 2 {
            pool.iter_mut().for_each(|p| p.2 = KeyEquality::Equal);
        }
        let dim = |i: usize| Dim::new(format!("d{i}"));
        let mut operands = Vec::new();
        for o in 0..n_operands {
            let mut ids: Vec<usize> = (0..n_dims).collect();
            let mut dims = Vec::new();
            for _ in 0..1 + rng.below(3.min(n_dims)) {
                dims.push(dim(ids.remove(rng.below(ids.len()))));
            }
            operands.push(Operand::new(format!("t{o}"), dims));
        }
        let mut output: Vec<Dim> = operands
            .iter()
            .flat_map(|o| o.dims.iter().cloned())
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .filter(|_| rng.chance(50))
            .collect();
        for i in (1..output.len()).rev() {
            output.swap(i, rng.below(i + 1));
        }
        let equality: BTreeMap<Dim, KeyEquality> =
            (0..n_dims).map(|i| (dim(i), pool[i].2)).collect();
        let aggregate = [Aggregate::SUM, Aggregate::COUNT, Aggregate::AVG][rng.below(3)];
        let fold = Fold::new(
            operands,
            output,
            equality,
            RowValue::Product(Op::Mul),
            aggregate,
        )
        .expect("generated folds are well formed");
        let info = |d: &Dim| pool[d.0[1..].parse::<usize>().unwrap()];
        let tables = fold
            .operands()
            .iter()
            .map(|op| {
                let rows = if rng.chance(12) { 0 } else { rng.below(9) };
                let key_null = [0, 20, 20, 60][rng.below(4)];
                let v_null = [0, 20, 80][rng.below(3)];
                let mut fields = Vec::new();
                let mut cols: Vec<ArrayRef> = Vec::new();
                for d in &op.dims {
                    let (kind, extent, _) = info(d);
                    let ks: Vec<Option<usize>> = (0..rows)
                        .map(|_| (extent > 0 && !rng.chance(key_null)).then(|| rng.below(extent)))
                        .collect();
                    let (ty, col): (DataType, ArrayRef) = match kind {
                        0 => (
                            DataType::Int64,
                            Arc::new(
                                ks.iter()
                                    .map(|k| k.map(|k| k as i64))
                                    .collect::<Int64Array>(),
                            ),
                        ),
                        1 => (
                            DataType::Utf8,
                            Arc::new(
                                ks.iter()
                                    .map(|k| k.map(|k| format!("k{k}")))
                                    .collect::<StringArray>(),
                            ),
                        ),
                        _ => (
                            DataType::Float64,
                            Arc::new(
                                ks.iter()
                                    .map(|k| k.map(|k| FLOATS[k]))
                                    .collect::<Float64Array>(),
                            ),
                        ),
                    };
                    fields.push(Field::new(&d.0, ty, true));
                    cols.push(col);
                }
                let v: Float64Array = (0..rows)
                    .map(|_| {
                        if rng.chance(v_null) {
                            None
                        } else if rng.chance(5) {
                            Some(f64::NAN)
                        } else {
                            Some((rng.below(17) as f64 - 8.0) / 4.0)
                        }
                    })
                    .collect();
                fields.push(Field::new("v", DataType::Float64, true));
                cols.push(Arc::new(v));
                RecordBatch::try_new(Arc::new(Schema::new(fields)), cols).expect("columns match")
            })
            .collect();
        let sql = build_sql(&fold);
        Case { fold, tables, sql }
    }
}

/// The SQL for a fold whose operands each hold a dimension at most once.
fn build_sql(e: &Fold) -> String {
    let ops = e.operands();
    let first = |d: &Dim| {
        ops.iter()
            .find(|o| o.dims.contains(d))
            .expect("in an operand")
    };
    let select: Vec<String> = e
        .output()
        .iter()
        .map(|d| format!("{}.{d}", first(d).name))
        .collect();
    let product: Vec<String> = ops.iter().map(|o| format!("{}.v", o.name)).collect();
    let mut sql = String::from("SELECT ");
    for (s, d) in select.iter().zip(e.output()) {
        sql += &format!("{s} AS {d}, ");
    }
    sql += &format!(
        "{}({}) AS v FROM {}",
        e.aggregate(),
        product.join(" * "),
        ops[0].name
    );
    for (i, op) in ops.iter().enumerate().skip(1) {
        let on: Vec<String> = op
            .dims
            .iter()
            .filter_map(|d| {
                let prev = ops[..i].iter().find(|p| p.dims.contains(d))?;
                let cmp = match e.equality(d)? {
                    KeyEquality::Equal => "=",
                    KeyEquality::NotDistinctFrom => "IS NOT DISTINCT FROM",
                };
                Some(format!("({}.{d} {cmp} {}.{d})", op.name, prev.name))
            })
            .collect();
        if on.is_empty() {
            sql += &format!(" CROSS JOIN {}", op.name);
        } else {
            sql += &format!(" JOIN {} ON {}", op.name, on.join(" AND "));
        }
    }
    if !select.is_empty() {
        sql += &format!(" GROUP BY {}", select.join(", "));
    }
    sql
}

impl fmt::Display for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "fold: {} {}\nsql: {}",
            self.fold.aggregate(),
            self.fold,
            self.sql
        )?;
        for (op, t) in self.fold.operands().iter().zip(&self.tables) {
            let t = datafusion::arrow::util::pretty::pretty_format_batches(std::slice::from_ref(t));
            writeln!(f, "table {}:\n{}", op.name, t.map_err(|_| fmt::Error)?)?;
        }
        Ok(())
    }
}
