// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`schema`]: the schema of a fold's result.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use einfold_ir::{ColumnIndex, Fold, Key};

use crate::{bad, Result};

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
