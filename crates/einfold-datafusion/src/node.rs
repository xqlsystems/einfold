// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`FoldNode`]: a fold over a join as a node of DataFusion's logical plan.
//!
//! A plan producer that knows it is writing a fold, such as ddx writing a
//! gradient step, puts this node in its plan directly instead of the join and
//! aggregate it stands for. DataFusion's optimizer leaves the node alone, and
//! [`EinFoldPlanner`](crate::EinFoldPlanner) turns it into
//! [`EinFoldExec`](crate::EinFoldExec).

use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::common::{plan_err, DFSchema, DFSchemaRef, Result};
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, UserDefinedLogicalNodeCore};
use einfold_ir::Fold;

/// A fold over a join (see [`einfold_ir::fold`]) in a DataFusion logical
/// plan. Its inputs are the fold's operands, in order, then its fill, if it
/// has one.
///
/// The output has one key column per output dimension, named after the
/// dimension, then the value column, a nullable `Float64`. Each key column
/// takes its type from the fill's column for it, if there is a fill, and
/// otherwise from the first operand that holds the dimension.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FoldNode {
    fold: Fold,
    inputs: Vec<LogicalPlan>,
    schema: DFSchemaRef,
}

impl FoldNode {
    /// The node computing `fold` over `operands`, with `fill` as the fill
    /// relation when `fold` has one.
    ///
    /// Checks what the fold's description can't know: that every column it
    /// names exists, that factors are `Float64`, and that every column of
    /// one dimension, in the operands and the fill, has one type, so keys
    /// compare as the SQL join would without a cast.
    pub fn try_new(
        fold: Fold,
        operands: Vec<LogicalPlan>,
        fill: Option<LogicalPlan>,
    ) -> Result<Self> {
        if operands.len() != 2 {
            return plan_err!("EinFold supports two operands, not {}", operands.len());
        }
        if fold.operands().len() != operands.len() {
            return plan_err!(
                "the fold has {} operands, given {} inputs",
                fold.operands().len(),
                operands.len()
            );
        }
        if fold.fill().is_some() != fill.is_some() {
            return plan_err!("a fill relation must be given exactly when the fold has a fill");
        }
        let field = |plan: &LogicalPlan, col: usize| -> Result<Field> {
            let schema = plan.schema();
            if col >= schema.fields().len() {
                return plan_err!("column {col} of an input with {}", schema.fields().len());
            }
            Ok(schema.field(col).as_ref().clone())
        };
        // Each dimension's type, from the first operand holding it.
        let mut types: Vec<Option<DataType>> = vec![None; fold.dimensions().len()];
        for (op, plan) in fold.operands().iter().zip(&operands) {
            let value = field(plan, op.value)?;
            if value.data_type() != &DataType::Float64 {
                return plan_err!("EinFold factors must be Float64, not {}", value.data_type());
            }
            for &(d, col) in &op.keys {
                let ty = field(plan, col)?.data_type().clone();
                match &types[d] {
                    None => types[d] = Some(ty),
                    Some(t) if *t == ty => {}
                    Some(t) => return plan_err!("dimension {d}'s columns have types {t} and {ty}"),
                }
            }
        }
        let mut fields = Vec::with_capacity(fold.outputs().len() + 1);
        for (i, &d) in fold.outputs().iter().enumerate() {
            let ty = types[d]
                .clone()
                .expect("every dimension is held by an operand");
            let source = match (fold.fill(), &fill) {
                (Some(f), Some(plan)) => {
                    let source = field(plan, f.keys[i])?;
                    if source.data_type() != &ty {
                        return plan_err!(
                            "the fill's column for dimension {d} is {}, the operands' {ty}",
                            source.data_type()
                        );
                    }
                    source
                }
                _ => {
                    let (op, plan) = fold
                        .operands()
                        .iter()
                        .zip(&operands)
                        .find(|(op, _)| op.column_of(d).is_some())
                        .expect("every dimension is held by an operand");
                    field(plan, op.column_of(d).expect("found above"))?
                }
            };
            let name = &fold.dimensions()[d].name;
            fields.push(Field::new(name, ty, source.is_nullable()));
        }
        if let (Some(f), Some(plan)) = (fold.fill(), &fill) {
            field(plan, f.value)?;
        }
        fields.push(Field::new(fold.value_name(), DataType::Float64, true));
        let schema = Arc::new(DFSchema::try_from(Schema::new(fields))?);
        let mut inputs = operands;
        inputs.extend(fill);
        Ok(FoldNode {
            fold,
            inputs,
            schema,
        })
    }

    /// The fold this node computes.
    pub fn fold(&self) -> &Fold {
        &self.fold
    }

    /// This node as a logical plan.
    pub fn into_plan(self) -> LogicalPlan {
        LogicalPlan::Extension(Extension {
            node: Arc::new(self),
        })
    }
}

// The schema follows from the fold and the inputs, so ordering ignores it.
impl PartialOrd for FoldNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (&self.fold, &self.inputs).partial_cmp(&(&other.fold, &other.inputs))
    }
}

impl UserDefinedLogicalNodeCore for FoldNode {
    fn name(&self) -> &str {
        "Fold"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        self.inputs.iter().collect()
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Fold: {}", describe(&self.fold))
    }

    fn with_exprs_and_inputs(&self, exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> Result<Self> {
        if !exprs.is_empty() {
            return plan_err!("a Fold node has no expressions");
        }
        let mut operands = inputs;
        let fill = match (self.fold.fill(), operands.len()) {
            (None, 2) => None,
            (Some(_), 3) => operands.pop(),
            (_, n) => return plan_err!("a Fold node has {} inputs, given {n}", self.inputs.len()),
        };
        FoldNode::try_new(self.fold.clone(), operands, fill)
    }
}

/// The fold in einsum-like notation for `EXPLAIN`: each operand's
/// dimensions, then the outputs, such as `n,d * d,h -> n,h`, with `+ fill`
/// when there is a fill.
pub(crate) fn describe(fold: &Fold) -> String {
    let names = |dims: &mut dyn Iterator<Item = usize>| {
        dims.map(|d| fold.dimensions()[d].name.as_str())
            .collect::<Vec<_>>()
            .join(",")
    };
    let operands: Vec<String> = fold
        .operands()
        .iter()
        .map(|op| names(&mut op.keys.iter().map(|(d, _)| *d)))
        .collect();
    format!(
        "{} -> {}{}",
        operands.join(" * "),
        names(&mut fold.outputs().iter().copied()),
        if fold.fill().is_some() { " + fill" } else { "" }
    )
}
