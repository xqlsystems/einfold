// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! EinFold's algorithms: computing a fold over a join (see [`einfold_ir::fold`])
//! from Arrow arrays to Arrow arrays, without forming the join's rows.
//!
//! In SQL terms, a fold of operands `a` and `b` is
//!
//! ```sql
//! SELECT <outputs>, SUM(a.<factor> * b.<factor>)
//! FROM a JOIN b ON <shared dimensions> GROUP BY <outputs>
//! ```
//!
//! and these functions compute exactly that, SQL's corners included: NULL
//! keys and factors, duplicate rows, groups reached only by NULLs, NaN and
//! infinities. A *host*, the engine whose plan the fold came from, decides a
//! few things SQL leaves open, such as whether `-0.0` and `0.0` are one key.
//! [`Host`] says which.
//!
//! Nothing here depends on a query engine, so any engine that holds its data
//! as Arrow arrays can run these, from DataFusion to an extension of DuckDB.

use std::sync::Arc;

use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use einfold_ir::{ColumnIndex, Fold, Key};

mod dense;
mod hash;
mod keys;

/// The result of a kernel.
pub type Result<T> = std::result::Result<T, ArrowError>;

/// How the host compares floating-point keys, in joins and in `GROUP BY`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatKeys {
    /// By their bits: `-0.0` and `0.0` are different keys, and a NaN matches
    /// a NaN with the same bits.
    Bitwise,
    /// By their bits, except that `-0.0` and `0.0` are one key, and a group of
    /// them is shown as `0.0`. DataFusion 55 does this for `Float16`, `Float32`
    /// and `Float64` columns, but not for dictionary-encoded ones.
    SignedZerosEqual,
}

/// What the kernels need to know about the host.
#[derive(Clone, Copy, Debug)]
pub struct Host {
    /// How the host compares float keys.
    pub float_keys: FloatKeys,
}

/// How to run a fold.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// What the host decides.
    pub host: Host,
    /// The most threads to use. The result's bits don't depend on it.
    pub threads: usize,
    /// Decline unless no two of the domain's rows locate the same group. A
    /// plan that joins a fold back to its domain and groups again by the
    /// domain's keys is the fold from its domain exactly when they are
    /// unique. Only [`fold_dense`] checks it, as it can decline.
    pub unique_domain: bool,
}

/// Which algorithm computed a fold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Algorithm {
    /// By positions, with no hashing: keys are small non-negative integers,
    /// used as array indices. Work is split into fixed blocks chosen from the
    /// shape alone, so the bits don't depend on the number of threads.
    Dense,
    /// Gustavson's row-by-row sparse product, generalized to folds: hash
    /// tables keyed by the join's and the groups' keys. Any keys, any density.
    Hash,
}

/// An error for a fold that the inputs don't fit.
fn bad<T>(why: String) -> Result<T> {
    Err(ArrowError::InvalidArgumentError(why))
}

/// The schema of `fold`'s result over inputs of these schemas: its two
/// operands, then its domain if it has one.
///
/// Without a domain, the result has one key column per output, typed as the
/// dimension's columns are, then the value. With a domain, it has the
/// domain's carried columns, then the value. The value is a `Float64`,
/// nullable unless the domain gives NULLs a value.
///
/// A fold names no columns, so these are named by position: `"0"`, `"1"`, ….
/// The kernels read only the types of the schema they are given, so a caller
/// may rename its fields freely.
///
/// Fails unless every column the fold names exists, factors are `Float64`,
/// and every column of one dimension, in the operands and the domain, has one
/// type, so that keys compare as the SQL join would, without a cast.
pub fn schema(fold: &Fold, operands: &[&Schema], domain: Option<&Schema>) -> Result<SchemaRef> {
    if fold.operands().len() != 2 || operands.len() != 2 {
        return bad(format!(
            "EinFold computes folds of two operands, not {}",
            fold.operands().len()
        ));
    }
    if fold.domain().is_some() != domain.is_some() {
        return bad("a domain relation must be given exactly when the fold has a domain".into());
    }
    if !fold.is_einsum() {
        return bad("EinFold computes SUM of products only".into());
    }
    let field = |schema: &Schema, ColumnIndex(col): ColumnIndex| -> Result<Field> {
        match schema.fields().get(col) {
            Some(f) => Ok(f.as_ref().clone()),
            None => bad(format!(
                "column {col} of an input with {}",
                schema.fields().len()
            )),
        }
    };
    // Each dimension's type and nullability, from the first operand holding it.
    let mut types: Vec<Option<Field>> = vec![None; fold.dimensions().len()];
    for (op, schema) in fold.operands().iter().zip(operands) {
        let Some(factor) = op.factor else {
            return bad("EinFold doesn't compute masks yet".into());
        };
        let factor = field(schema, factor)?;
        if factor.data_type() != &DataType::Float64 {
            return bad(format!(
                "EinFold factors must be Float64, not {}",
                factor.data_type()
            ));
        }
        for &Key {
            dimension: d,
            column: col,
        } in &op.keys
        {
            let f = field(schema, col)?;
            match &types[d.0] {
                None => types[d.0] = Some(f),
                Some(t) if t.data_type() == f.data_type() => {}
                Some(t) => {
                    return bad(format!(
                        "dimension {}'s columns have types {} and {}",
                        d.0,
                        t.data_type(),
                        f.data_type()
                    ))
                }
            }
        }
    }
    let mut fields: Vec<Field> = Vec::new();
    match (fold.domain(), domain) {
        (Some(spec), Some(schema)) => {
            for o in fold.outputs() {
                let col = o.domain_key.expect("checked by Fold::try_new").column;
                let key = field(schema, col)?;
                let want = types[o.dimension.0].as_ref().expect("held by an operand");
                if key.data_type() != want.data_type() {
                    return bad(format!(
                        "the domain's column for dimension {} is {}, the operands' {}",
                        o.dimension.0,
                        key.data_type(),
                        want.data_type()
                    ));
                }
            }
            for &col in &spec.carry {
                let f = field(schema, col)?;
                fields.push(f.with_name(fields.len().to_string()));
            }
        }
        _ => {
            for o in fold.outputs() {
                let f = types[o.dimension.0].as_ref().expect("held by an operand");
                let name = fields.len().to_string();
                fields.push(Field::new(name, f.data_type().clone(), f.is_nullable()));
            }
        }
    }
    let nullable = fold.domain().is_none_or(|d| d.missing.is_none());
    fields.push(Field::new(
        fields.len().to_string(),
        DataType::Float64,
        nullable,
    ));
    Ok(Arc::new(Schema::new(fields)))
}

/// The rows of one of a fold's inputs, as batches of one schema, in the
/// order and batches they arrived in.
#[derive(Clone, Copy, Debug)]
pub struct Relation<'a> {
    /// The batches' schema.
    pub schema: &'a SchemaRef,
    /// The rows.
    pub batches: &'a [RecordBatch],
}

impl<'a> Relation<'a> {
    /// `batches`, all of `schema`.
    pub fn new(schema: &'a SchemaRef, batches: &'a [RecordBatch]) -> Self {
        Relation { schema, batches }
    }
}

/// [`fold`] by the dense algorithm alone, or `None` when its conditions don't
/// hold, for a caller that would rather run something else than the hash
/// algorithm.
pub fn fold_dense(
    fold: &Fold,
    a: Relation,
    b: Relation,
    domain: Option<Relation>,
    schema: SchemaRef,
    options: Options,
) -> Result<Option<Vec<RecordBatch>>> {
    let threads = options.threads.max(1);
    let domain = domain.map(|d| d.batches);
    dense::fold(
        fold,
        a.batches,
        b.batches,
        domain,
        schema,
        threads,
        options.unique_domain,
    )
}

/// `fold` over operands `a` and `b`, and its domain if it has one, as
/// batches of `schema` (which [`schema`] gives): one batch per domain batch,
/// or one in all without a domain. Also says which algorithm ran: the dense
/// one when its conditions hold, and the hash one otherwise.
///
/// The result's bits depend only on the inputs' rows, not on how they were
/// batched or ordered, nor on the number of threads.
pub fn fold(
    fold: &Fold,
    a: Relation,
    b: Relation,
    domain: Option<Relation>,
    schema: SchemaRef,
    options: Options,
) -> Result<(Algorithm, Vec<RecordBatch>)> {
    if options.unique_domain {
        return bad("only fold_dense checks that a domain's keys are unique".into());
    }
    let threads = options.threads.max(1);
    let domain = domain.map(|d| d.batches);
    let dense = dense::fold(
        fold,
        a.batches,
        b.batches,
        domain,
        schema.clone(),
        threads,
        false,
    )?;
    if let Some(batches) = dense {
        return Ok((Algorithm::Dense, batches));
    }
    // The hash algorithm reads each operand as one batch.
    let whole = |r: Relation| concat_batches(r.schema, r.batches);
    let (a, b) = (whole(a)?, whole(b)?);
    let batches = hash::fold(fold, &a, &b, domain, schema, options.host)?;
    Ok((Algorithm::Hash, batches))
}
