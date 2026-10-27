// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Property: EinFold's kernels compute what the SQL query a fold stands for
//! computes on DataFusion, the host, for random folds over random tables.
//!
//! Each case draws a fold of two operands over one to four dimensions (keys
//! `BIGINT`, `VARCHAR` or `DOUBLE`, compared with `=` or `IS NOT DISTINCT
//! FROM`), random outputs, and half the time a domain: a relation the result
//! is left-joined from, each key matched with `=` or `IS NOT DISTINCT FROM`,
//! with a random `COALESCE` default or none. Tables have NULL keys,
//! duplicate key tuples, missing rows, and factors that are NULL, NaN, ±inf,
//! `-0.0`, or ill-conditioned (±1e16 beside small values), split across
//! partitions and batches at random. The invariants:
//!
//! 1. **Same rows.** The same key tuples, as a multiset, with keys compared by
//!    their bits (`-0.0` is not `0.0`).
//! 2. **Same values, within float summation's bound.** NULL where SQL gives
//!    NULL; NaN where SQL gives NaN; the same infinity; and otherwise within
//!    `2·(n − 1)·u·Σ|terms|` of SQL's value, where `n` is the group's number
//!    of products and `u = 2⁻⁵³`. Any order of the additions is within half
//!    that of the exact sum, and so is the host's.
//! 3. **Deterministic.** The same rows, in another order and other batches,
//!    on another number of threads, give the same rows, bit for bit.
//! 4. **Unique domains.** Where the dense path runs with a domain, asking it
//!    to check that the domain's keys are unique declines whenever two domain
//!    rows have the keys of one group, and otherwise, when it accepts, gives
//!    the same rows, bit for bit.
//!
//! Half the cases are drawn from a *dense family*, shaped so that the dense
//! algorithm may apply: non-negative integer keys with no NULLs in the
//! operands, and every dimension either an output of one operand or summed
//! and held by both. Some of them still take the discrimination algorithm (duplicate key
//! tuples, or holes in an operand beside NaN or ±inf), and the test checks
//! that the dense path ran in a good share of them.
//!
//! Reproduce a failure with `EINFOLD_SEED=<seed printed>`; search longer with
//! `EINFOLD_CASES=<n>` (and a starting `EINFOLD_SEED`).

use std::collections::HashSet;
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, AsArray, DictionaryArray, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringArray, UInt64Array,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::ScalarValue;
use datafusion::datasource::MemTable;
use datafusion::prelude::{SessionConfig, SessionContext};
use einfold_ir::{
    Aggregate, ColumnIndex, Dimension, DimensionId, Domain, DomainKey, Fold, Key, KeyEquality,
    Operand, Output, RowValue,
};
use einfold_kernels::{Algorithm, DomainRows, FloatKeys, Host, Options, Relation};

struct Rng(u64);

impl Rng {
    /// A generator for `seed`, scrambled by SplitMix64 so that nearby seeds
    /// give unrelated streams, and never zero, which xorshift can't leave.
    fn new(seed: u64) -> Self {
        let mut z = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        Rng((z ^ (z >> 31)).max(1))
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn chance(&mut self, p: f64) -> bool {
        ((self.next() >> 11) as f64) / ((1u64 << 53) as f64) < p
    }
}

#[derive(Clone, Copy, Debug)]
enum KeyType {
    Int,
    /// Non-negative integers, usable as positions.
    Pos,
    /// Unsigned integers, usable as positions except key 0, which is 2⁶³:
    /// beyond `BIGINT`, so a dense array can't hold it.
    Wide,
    Text,
    Float,
    /// Dictionary-encoded text whose NULLs are a NULL dictionary value, not a
    /// NULL index: NULL to SQL, though the column's null buffer has none.
    Dict,
}

impl KeyType {
    fn data_type(self) -> DataType {
        match self {
            KeyType::Int | KeyType::Pos => DataType::Int64,
            KeyType::Wide => DataType::UInt64,
            KeyType::Text => DataType::Utf8,
            KeyType::Float => DataType::Float64,
            KeyType::Dict => {
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8))
            }
        }
    }
    /// The type of the keys `key` returns.
    fn value_type(self) -> DataType {
        match self {
            KeyType::Dict => DataType::Utf8,
            t => t.data_type(),
        }
    }
    /// `keys` as a column of this type.
    fn column(self, keys: Vec<ScalarValue>) -> ArrayRef {
        let values = if keys.is_empty() {
            datafusion::arrow::array::new_empty_array(&self.value_type())
        } else {
            ScalarValue::iter_to_array(keys).unwrap()
        };
        let KeyType::Dict = self else {
            return values;
        };
        let mut dict: Vec<Option<&str>> = vec![None];
        let indices: Int32Array = values
            .as_string::<i32>()
            .iter()
            .map(|v| {
                let i = dict.iter().position(|d| *d == v).unwrap_or_else(|| {
                    dict.push(v);
                    dict.len() - 1
                });
                Some(i as i32)
            })
            .collect();
        Arc::new(DictionaryArray::new(
            indices,
            Arc::new(StringArray::from(dict)),
        ))
    }
    /// A literal that stands for NULL in the oracle's NULL-safe keys.
    fn sentinel(self) -> &'static str {
        match self {
            KeyType::Int | KeyType::Pos => "CAST(0 AS BIGINT)",
            KeyType::Wide => "arrow_cast(0, 'UInt64')",
            KeyType::Text | KeyType::Dict => "''",
            KeyType::Float => "CAST(0.0 AS DOUBLE)",
        }
    }
    /// Key `i` of this type; floats include `-0.0` beside `0.0`, and NaN.
    fn key(self, i: usize) -> ScalarValue {
        match self {
            KeyType::Int => ScalarValue::Int64(Some(i as i64 - 1)),
            KeyType::Pos => ScalarValue::Int64(Some(i as i64)),
            // 2⁶³ doesn't fit BIGINT, and 2⁶³ - 1 is BIGINT's largest.
            KeyType::Wide => ScalarValue::UInt64(Some(match i {
                0 => 1 << 63,
                1 => (1 << 63) - 1,
                i => i as u64,
            })),
            KeyType::Text | KeyType::Dict => ScalarValue::Utf8(Some(format!("k{i}"))),
            KeyType::Float => ScalarValue::Float64(Some([0.0, -0.0, f64::NAN, 2.5, -1.0][i % 5])),
        }
    }
}

/// A random factor: NULL, special values, ill-conditioned or ordinary; with
/// `specials` false, never NaN or infinite.
fn factor(rng: &mut Rng, specials: bool) -> Option<f64> {
    let r = match rng.below(100) {
        10..=15 if !specials => 50,
        r => r,
    };
    match r {
        0..=9 => None,
        10..=12 => Some(f64::NAN),
        13..=14 => Some(f64::INFINITY),
        15 => Some(f64::NEG_INFINITY),
        16..=18 => Some(-0.0),
        19..=25 => Some(if rng.chance(0.5) { 1e16 } else { -1e16 }),
        _ => {
            let mantissa = (rng.next() >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
            Some(mantissa * 10f64.powi(rng.below(7) as i32 - 3))
        }
    }
}

/// A table's rows, column by column.
struct Table {
    schema: Arc<Schema>,
    columns: Vec<ArrayRef>,
}

impl Table {
    fn rows(&self) -> usize {
        self.columns.first().map_or(0, |c| c.len())
    }

    /// The rows in a random order, split into random partitions and batches.
    fn partitions(&self, rng: &mut Rng) -> Vec<Vec<RecordBatch>> {
        let n = self.rows();
        let mut order: Vec<u64> = (0..n as u64).collect();
        for i in (1..n).rev() {
            order.swap(i, rng.below(i + 1));
        }
        let shuffled: Vec<ArrayRef> = self
            .columns
            .iter()
            .map(|c| {
                datafusion::arrow::compute::take(c, &UInt64Array::from(order.clone()), None)
                    .unwrap()
            })
            .collect();
        let batch = RecordBatch::try_new(Arc::clone(&self.schema), shuffled).unwrap();
        let parts = 1 + rng.below(3);
        let mut out = vec![Vec::new(); parts];
        let mut start = 0;
        while start < n {
            let len = (1 + rng.below(5)).min(n - start);
            out[rng.below(parts)].push(batch.slice(start, len));
            start += len;
        }
        out
    }
}

/// How a table's rows are drawn.
#[derive(Clone, Copy)]
struct Rows {
    /// The chance that each key tuple has a row.
    present: f64,
    /// Whether key tuples may repeat.
    duplicates: bool,
    /// Whether keys may be NULL.
    null_keys: bool,
    /// Whether factors may be NaN or infinite.
    specials: bool,
}

const MESSY: Rows = Rows {
    present: 0.75,
    duplicates: true,
    null_keys: true,
    specials: true,
};

/// A table with `keys` (one column per dimension, of its type, named
/// `names[i]`) and a `DOUBLE` column `v`, over the key tuples of `extents`,
/// drawn as `how` says.
fn table(
    rng: &mut Rng,
    names: &[String],
    types: &[KeyType],
    extents: &[usize],
    how: Rows,
) -> Table {
    let mut fields: Vec<Field> = names
        .iter()
        .zip(types)
        .map(|(n, t)| Field::new(n, t.data_type(), true))
        .collect();
    fields.push(Field::new("v", DataType::Float64, true));
    let mut keys: Vec<Vec<ScalarValue>> = vec![Vec::new(); names.len()];
    let mut values = Vec::new();
    let total: usize = extents.iter().product();
    for mut t in 0..total {
        if !rng.chance(how.present) {
            continue;
        }
        let copies = if how.duplicates && rng.chance(0.15) {
            2
        } else {
            1
        };
        let tuple: Vec<ScalarValue> = extents
            .iter()
            .zip(types)
            .map(|(&e, ty)| {
                let i = t % e;
                t /= e;
                if how.null_keys && rng.chance(0.08) {
                    ScalarValue::try_from(ty.value_type()).unwrap()
                } else {
                    ty.key(i)
                }
            })
            .collect();
        for _ in 0..copies {
            for (k, v) in keys.iter_mut().zip(&tuple) {
                k.push(v.clone());
            }
            values.push(factor(rng, how.specials));
        }
    }
    let mut columns: Vec<ArrayRef> = keys
        .into_iter()
        .zip(types)
        .map(|(k, t)| t.column(k))
        .collect();
    columns.push(Arc::new(Float64Array::from(values)));
    Table {
        schema: Arc::new(Schema::new(fields)),
        columns,
    }
}

/// One random case: a fold, its tables, and the SQL it stands for.
struct Case {
    fold: Fold,
    a: Table,
    b: Table,
    domain: Option<Table>,
    sql: String,
}

/// Shapes `[x, z, y]` large enough to cross the dense algorithm's block
/// sizes: an output of at least 4,096 cells, split into blocks of rows; one
/// of few rows and more than 2,048 columns, split into blocks of columns;
/// and a small output whose sums run over more than 4,096 positions of `z`,
/// split into blocks along `z`.
const LARGE: [[usize; 3]; 3] = [[300, 7, 16], [3, 4, 5000], [3, 9000, 5]];

/// A random case; from the dense family when `dense`. With `large`, a dense
/// case of one dimension per role, `x` in `a`, `z` in both and `y` in `b`,
/// with those extents.
fn case(rng: &mut Rng, dense: bool, large: Option<[usize; 3]>) -> Case {
    use std::fmt::Write as _;
    let dense = dense || large.is_some();
    let n = if large.is_some() { 3 } else { 1 + rng.below(4) };
    let types: Vec<KeyType> = (0..n)
        .map(|_| match dense {
            true if large.is_none() && rng.chance(0.1) => KeyType::Wide,
            true => KeyType::Pos,
            false => [
                KeyType::Int,
                KeyType::Int,
                KeyType::Text,
                KeyType::Float,
                KeyType::Dict,
            ][rng.below(5)],
        })
        .collect();
    let extents: Vec<usize> = match large {
        Some(shape) => shape.to_vec(),
        None => (0..n)
            .map(|_| 1 + rng.below(if dense { 6 } else { 4 }))
            .collect(),
    };
    let dims: Vec<Dimension> = (0..n)
        .map(|_| Dimension {
            equality: equality(rng),
        })
        .collect();
    // Each dimension in a, in b, or in both; never in neither.
    let mut held = vec![[false; 2]; n];
    for (d, h) in held.iter_mut().enumerate() {
        // A large case's x is in a, z in both, and y in b.
        match large.map_or_else(|| rng.below(3), |_| [0, 2, 1][d]) {
            0 => h[0] = true,
            1 => h[1] = true,
            _ => *h = [true, true],
        }
    }
    // In the dense family, a dimension in one operand is an output, and one
    // in both is summed.
    let mut outputs: Vec<usize> = (0..n)
        .filter(|&d| match dense {
            true => held[d] != [true, true],
            false => rng.chance(0.5),
        })
        .collect();
    for i in (1..outputs.len()).rev() {
        outputs.swap(i, rng.below(i + 1));
    }
    let how = |rng: &mut Rng| match dense {
        true => Rows {
            present: if rng.chance(0.6) { 1.0 } else { 0.75 },
            duplicates: rng.chance(0.2),
            null_keys: false,
            specials: rng.chance(0.5),
        },
        false => MESSY,
    };
    let side = |s: usize, rng: &mut Rng| {
        let mine: Vec<usize> = (0..n).filter(|&d| held[d][s]).collect();
        let names: Vec<String> = mine.iter().map(|d| format!("k{d}")).collect();
        let ty: Vec<KeyType> = mine.iter().map(|&d| types[d]).collect();
        let ext: Vec<usize> = mine.iter().map(|&d| extents[d]).collect();
        let rows = how(rng);
        let t = table(rng, &names, &ty, &ext, rows);
        let op = Operand {
            keys: mine
                .iter()
                .enumerate()
                .map(|(c, &d)| Key::new(DimensionId(d), ColumnIndex(c)))
                .collect(),
            factor: Some(ColumnIndex(mine.len())),
        };
        (t, op, mine)
    };
    let (a, op_a, dims_a) = side(0, rng);
    let (b, op_b, dims_b) = side(1, rng);
    let with_domain = rng.chance(0.5);
    let domain_table = with_domain.then(|| {
        let names: Vec<String> = outputs.iter().map(|d| format!("k{d}")).collect();
        let ty: Vec<KeyType> = outputs.iter().map(|&d| types[d]).collect();
        let ext: Vec<usize> = outputs.iter().map(|&d| extents[d] + 1).collect();
        table(rng, &names, &ty, &ext, MESSY)
    });
    // The domain carries its key columns, so the result's key columns are
    // where they are without one.
    // How each output is matched with its domain column, which is column i.
    let domain_keys: Vec<KeyEquality> = match with_domain {
        true => (0..outputs.len()).map(|_| equality(rng)).collect(),
        false => vec![],
    };
    let domain = with_domain.then(|| Domain {
        carry: (0..outputs.len()).map(ColumnIndex).collect(),
        // The dense family leans to 0.0, the default ddx's gradients use, and
        // the one that lets the dense path ignore which groups exist.
        missing: match rng.below(if dense { 5 } else { 3 }) {
            0 => None,
            1 => Some(2.5),
            _ => Some(0.0),
        },
    });
    let fold = Fold::try_new(
        dims.clone(),
        vec![op_a, op_b],
        outputs
            .iter()
            .enumerate()
            .map(|(i, &d)| Output {
                dimension: DimensionId(d),
                domain_key: domain_keys
                    .get(i)
                    .map(|&eq| DomainKey::new(ColumnIndex(i), eq)),
            })
            .collect(),
        RowValue::Product,
        Aggregate::sum(),
        domain.clone(),
    )
    .unwrap();

    // The oracle's SQL. A key compared with IS NOT DISTINCT FROM is compared
    // as the pair (key IS NULL, COALESCE(key, sentinel)) with `=`, which is
    // the same relation; DataFusion's hash join has one NULL-equality setting
    // per join, so mixing both kinds in one join is avoided.
    let safe = |t: &str, d: usize| {
        format!(
            "{t}.k{d} IS NULL AS n{d}, COALESCE({t}.k{d}, {}) AS c{d}",
            types[d].sentinel()
        )
    };
    let view = |t: &str, mine: &[usize]| {
        let mut cols = vec![format!("{t}.v")];
        for &d in mine {
            cols.push(format!("{t}.k{d}"));
            cols.push(safe(t, d));
        }
        format!("(SELECT {} FROM {t}) {t}", cols.join(", "))
    };
    let shared: Vec<usize> = (0..n).filter(|&d| held[d] == [true, true]).collect();
    let mut on: Vec<String> = Vec::new();
    for &d in &shared {
        match dims[d].equality {
            KeyEquality::Equal => on.push(format!("a.k{d} = b.k{d}")),
            KeyEquality::NotDistinct => {
                on.push(format!("a.n{d} = b.n{d} AND a.c{d} = b.c{d}"));
            }
        }
    }
    let src = |d: usize| if held[d][0] { "a" } else { "b" };
    let group: Vec<String> = outputs
        .iter()
        .map(|&d| format!("{}.k{d}", src(d)))
        .collect();
    let mut folded = format!(
        "SELECT {} SUM(a.v * b.v) AS v, SUM(abs(a.v * b.v)) AS mag, COUNT(*) AS cnt FROM {} JOIN {} ON {}",
        outputs
            .iter()
            .map(|&d| format!("{}.k{d} AS d{d}, {} ", src(d), safe(src(d), d)) + ",")
            .collect::<String>(),
        view("a", &dims_a),
        view("b", &dims_b),
        if on.is_empty() { "true".into() } else { on.join(" AND ") },
    );
    if !group.is_empty() {
        let _ = write!(folded, " GROUP BY {}", group.join(", "));
    }
    // A fold over a join has a row only where something joined, as a grouped
    // query does; without GROUP BY, SQL would return one row regardless.
    folded.push_str(" HAVING COUNT(*) > 0");
    let sql =
        if let Some(domain) = &domain {
            let domain_view = format!(
                "(SELECT f.v, {} FROM f) f",
                outputs
                    .iter()
                    .map(|&d| format!("f.k{d}, {}", safe("f", d)))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            let on: Vec<String> = outputs
                .iter()
                .zip(&domain_keys)
                .map(|(&d, eq)| match eq {
                    KeyEquality::Equal => format!("f.k{d} = s.d{d}"),
                    KeyEquality::NotDistinct => format!("f.n{d} = s.n{d} AND f.c{d} = s.c{d}"),
                })
                .collect();
            let value = match domain.missing {
                Some(m) => format!("COALESCE(s.v, CAST({m:?} AS DOUBLE))"),
                None => "s.v".into(),
            };
            format!(
            "SELECT {} {value} AS v, s.mag, s.cnt FROM {domain_view} LEFT JOIN ({folded}) s ON {}",
            outputs.iter().map(|&d| format!("f.k{d} AS d{d}, ")).collect::<String>(),
            if on.is_empty() { "true".into() } else { on.join(" AND ") },
        )
        } else {
            format!(
                "SELECT {} v, mag, cnt FROM ({folded}) s",
                outputs
                    .iter()
                    .map(|&d| format!("d{d}, "))
                    .collect::<String>()
            )
        };
    Case {
        fold,
        a,
        b,
        domain: domain_table,
        sql,
    }
}

fn equality(rng: &mut Rng) -> KeyEquality {
    if rng.chance(0.5) {
        KeyEquality::Equal
    } else {
        KeyEquality::NotDistinct
    }
}

fn context(threads: usize) -> SessionContext {
    SessionContext::new_with_config(SessionConfig::new().with_target_partitions(threads))
}

fn register(ctx: &SessionContext, name: &str, t: &Table, rng: &mut Rng) {
    let mem = MemTable::try_new(Arc::clone(&t.schema), t.partitions(rng)).unwrap();
    ctx.register_table(name, Arc::new(mem)).unwrap();
}

/// A row: its keys, printed so that `-0.0` and `0.0` differ and a dictionary
/// key prints as its value, and its value.
type Row = (String, Option<f64>);

fn rows(batches: &[RecordBatch], keys: usize) -> Vec<Row> {
    use std::fmt::Write as _;
    let mut out = Vec::new();
    for b in batches {
        let v = b
            .column(keys)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        for r in 0..b.num_rows() {
            let mut k = String::new();
            for c in 0..keys {
                let col = match b.column(c).data_type() {
                    DataType::Dictionary(_, v) => {
                        datafusion::arrow::compute::cast(b.column(c), v).unwrap()
                    }
                    _ => Arc::clone(b.column(c)),
                };
                let s = ScalarValue::try_from_array(&col, r).unwrap();
                let _ = write!(k, "{s:?}|");
            }
            out.push((k, v.is_valid(r).then(|| v.value(r))));
        }
    }
    out
}

/// The oracle's rows, each with its bound on the difference of two sums.
fn oracle_rows(batches: &[RecordBatch], keys: usize) -> Vec<(Row, f64)> {
    let mut out = Vec::new();
    let mut base = rows(batches, keys).into_iter();
    for b in batches {
        let mag = b
            .column(keys + 1)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        let cnt = b
            .column(keys + 2)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for r in 0..b.num_rows() {
            let n = if cnt.is_valid(r) { cnt.value(r) } else { 0 } as f64;
            let m = if mag.is_valid(r) { mag.value(r) } else { 0.0 };
            out.push((
                base.next().unwrap(),
                2.0 * (n - 1.0).max(0.0) * f64::EPSILON / 2.0 * m,
            ));
        }
    }
    out
}

fn equivalent(ours: Option<f64>, theirs: Option<f64>, bound: f64) -> bool {
    match (ours, theirs) {
        (None, None) => true,
        (Some(x), Some(y)) if x.is_nan() || y.is_nan() => x.is_nan() && y.is_nan(),
        (Some(x), Some(y)) if x == y => x.is_sign_negative() == y.is_sign_negative(),
        (Some(x), Some(y)) => x.is_finite() && y.is_finite() && (x - y).abs() <= bound,
        _ => false,
    }
}

/// DataFusion 55's key equality (crates/einfold-datafusion pins it).
const HOST: Host = Host {
    float_keys: FloatKeys::SignedZerosEqual,
};

/// The kernels' output, from the tables' rows in a random order and random
/// batches, on a random number of threads, and which algorithm ran.
fn run_fold(case: &Case, rng: &mut Rng) -> (Vec<RecordBatch>, Algorithm) {
    let batches = |t: &Table, rng: &mut Rng| -> Vec<RecordBatch> {
        t.partitions(rng).into_iter().flatten().collect()
    };
    let (a, b) = (batches(&case.a, rng), batches(&case.b, rng));
    let domain = case.domain.as_ref().map(|t| batches(t, rng));

    let domain = (case.domain.as_ref())
        .zip(domain.as_deref())
        .map(|(t, d)| Relation::new(&t.schema, d));
    let schema = einfold_kernels::schema(
        &case.fold,
        &[&case.a.schema, &case.b.schema],
        case.domain.as_ref().map(|t| t.schema.as_ref()),
    )
    .unwrap();
    let options = Options {
        host: HOST,
        threads: 1 + rng.below(8),
        ..Options::default()
    };
    let folded = einfold_kernels::fold(
        &case.fold,
        Relation::new(&case.a.schema, &a),
        Relation::new(&case.b.schema, &b),
        domain,
        schema,
        options,
    )
    .unwrap();
    (folded.batches, folded.algorithm)
}

/// Checks invariant 4 for a case with a domain that the dense path computes,
/// and says whether the checked path accepted.
fn check_unique_domain(case: &Case, rng: &mut Rng, fail: impl Fn(String)) -> bool {
    let fold = &case.fold;
    let domain = case.domain.as_ref().expect("a case with a domain");
    let batches = |t: &Table, rng: &mut Rng| -> Vec<RecordBatch> {
        t.partitions(rng).into_iter().flatten().collect()
    };
    let (a, b) = (batches(&case.a, rng), batches(&case.b, rng));
    let (a, b) = (
        Relation::new(&case.a.schema, &a),
        Relation::new(&case.b.schema, &b),
    );
    let d: Vec<RecordBatch> = domain.partitions(rng).into_iter().flatten().collect();
    let operands = [case.a.schema.as_ref(), case.b.schema.as_ref()];
    let mut dense = |fold: &Fold, d: Option<&[RecordBatch]>, domain_rows| {
        let schema = einfold_kernels::schema(fold, &operands, d.map(|_| domain.schema.as_ref()));
        let options = Options {
            host: HOST,
            threads: 1 + rng.below(8),
            domain_rows,
        };
        let d = d.map(|batches| Relation::new(&domain.schema, batches));
        einfold_kernels::fold_dense(fold, a, b, d, schema.unwrap(), options).unwrap()
    };
    let keys = fold.outputs().len();
    let unchecked = dense(fold, Some(&d), DomainRows::Any).expect("the dense path applies");
    // The groups, from the same fold without its domain.
    let outputs = fold
        .outputs()
        .iter()
        .map(|o| Output {
            domain_key: None,
            ..*o
        })
        .collect();
    let bare = Fold::try_new(
        fold.dimensions().to_vec(),
        fold.operands().to_vec(),
        outputs,
        fold.row(),
        fold.aggregate().clone(),
        None,
    )
    .unwrap();
    let groups: HashSet<String> = match dense(&bare, None, DomainRows::Any) {
        Some(out) => rows(&out, keys).into_iter().map(|(k, _)| k).collect(),
        None => return false,
    };
    let mut seen = HashSet::new();
    let repeated = rows(&unchecked, keys)
        .into_iter()
        .any(|(k, _)| groups.contains(&k) && !seen.insert(k));
    match dense(fold, Some(&d), DomainRows::UniqueKeys) {
        Some(_) if repeated => {
            fail("the checked dense path accepted two domain rows with one group's keys".into());
            true
        }
        Some(checked) => {
            let bits = |out: &[RecordBatch]| {
                let mut r: Vec<_> = (rows(out, keys).into_iter())
                    .map(|(k, v)| (k, v.map(f64::to_bits)))
                    .collect();
                r.sort();
                r
            };
            if bits(&checked) != bits(&unchecked) {
                fail("the checked dense path changed the output".into());
            }
            true
        }
        None => false,
    }
}

#[tokio::test]
async fn fold_matches_sql() {
    let seed: u64 = std::env::var("EINFOLD_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x5eed_f01d);
    let cases: u64 = match std::env::var("EINFOLD_CASES") {
        Ok(n) => n.parse().unwrap(),
        Err(_) if std::env::var("EINFOLD_SEED").is_ok() => 1,
        Err(_) => 400,
    };
    let (mut dense_family, mut dense_ran) = (0, 0);
    let (mut unique_checked, mut unique_accepted) = (0, 0);
    for i in 0..cases {
        let seed = seed.wrapping_add(i);
        let mut rng = Rng::new(seed);
        // The family follows from the seed, so a seed alone reproduces a case.
        let dense = seed % 2 == 1;
        let large = (seed % 20 == 1).then(|| LARGE[(seed / 20) as usize % LARGE.len()]);
        let case = case(&mut rng, dense, large);
        let keys = case.fold.outputs().len();
        let fail = |what: String| panic!("seed {seed}: {what}\nSQL: {}", case.sql);

        let (ours, ran) = run_fold(&case, &mut rng);
        let (again, _) = run_fold(&case, &mut rng);
        dense_family += usize::from(dense);
        dense_ran += usize::from(ran == Algorithm::Dense);
        if ran == Algorithm::Dense && case.domain.is_some() {
            unique_checked += 1;
            unique_accepted += usize::from(check_unique_domain(&case, &mut rng, fail));
        }
        let (ours, again) = (rows(&ours, keys), rows(&again, keys));
        // SQL doesn't order rows, so the outputs are compared as multisets.
        let bits = |r: &[Row]| -> Vec<(String, Option<u64>)> {
            let mut b: Vec<_> = r
                .iter()
                .map(|(k, v)| (k.clone(), v.map(f64::to_bits)))
                .collect();
            b.sort();
            b
        };
        if bits(&ours) != bits(&again) {
            fail(format!(
                "two arrival orders and thread counts gave different outputs:\n{ours:?}\n{again:?}"
            ));
        }

        let ctx = context(4);
        register(&ctx, "a", &case.a, &mut rng);
        register(&ctx, "b", &case.b, &mut rng);
        if let Some(f) = &case.domain {
            register(&ctx, "f", f, &mut rng);
        }
        let theirs = ctx.sql(&case.sql).await.unwrap().collect().await.unwrap();
        let mut theirs = oracle_rows(&theirs, keys);
        let mut ours = ours;
        if ours.len() != theirs.len() {
            fail(format!("{} rows, SQL has {}", ours.len(), theirs.len()));
        }
        // Domain rows with the same keys differ only in whether they are NULL.
        ours.sort_by_key(|(k, v)| (k.clone(), v.is_some()));
        theirs.sort_by_key(|((k, v), _)| (k.clone(), v.is_some()));
        for (o, ((k, v), bound)) in ours.iter().zip(&theirs) {
            if o.0 != *k || !equivalent(o.1, *v, *bound) {
                fail(format!("row {o:?}, SQL's ({k}, {v:?}) within {bound:e}"));
            }
        }
    }
    eprintln!("dense path: {dense_ran} of {cases} cases; dense family: {dense_family}");
    eprintln!("unique-domain check: accepted {unique_accepted} of {unique_checked}");
    // About a third of the dense family qualifies (the rest has duplicate
    // tuples, keys too large for BIGINT, or holes where existence matters);
    // if far fewer did, the dense path would go untested.
    if cases >= 100 {
        assert!(
            dense_ran * 4 >= dense_family,
            "the dense path ran in only {dense_ran} of {dense_family} dense-family cases"
        );
    }
}
