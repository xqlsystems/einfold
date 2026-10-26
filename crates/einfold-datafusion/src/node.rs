// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`FoldNode`]: a fold over a join as a node of DataFusion's logical plan.
//!
//! The node stands for the join and aggregate it replaces (see
//! [`einfold_ir::fold`]). DataFusion's optimizer leaves it alone, and
//! [`EinFoldPlanner`](crate::EinFoldPlanner) turns it into
//! [`EinFoldExec`](crate::EinFoldExec).

use std::fmt;
use std::sync::Arc;

use datafusion::common::{plan_err, DFSchema, DFSchemaRef, Result};
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, UserDefinedLogicalNodeCore};
use einfold_ir::{DimensionId, Fold};

/// A fold over a join in a DataFusion logical plan. Its inputs are the fold's
/// two operands, then its domain if it has one; its schema is
/// [`einfold_kernels::schema`]'s.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FoldNode {
    fold: Fold,
    inputs: Vec<LogicalPlan>,
    schema: DFSchemaRef,
}

impl FoldNode {
    /// The node computing `fold` over `operands`, with `domain` as the domain
    /// relation when `fold` has one. Fails when the inputs don't fit the fold
    /// (see [`einfold_kernels::schema`]).
    pub fn try_new(
        fold: Fold,
        operands: Vec<LogicalPlan>,
        domain: Option<LogicalPlan>,
    ) -> Result<Self> {
        let arrow = |p: &LogicalPlan| p.schema().as_arrow().clone();
        let operand_schemas: Vec<_> = operands.iter().map(arrow).collect();
        let domain_schema = domain.as_ref().map(arrow);
        let schema = einfold_kernels::schema(
            &fold,
            &operand_schemas.iter().collect::<Vec<_>>(),
            domain_schema.as_ref(),
        )?;
        let schema = Arc::new(DFSchema::try_from(schema.as_ref().clone())?);
        let mut inputs = operands;
        inputs.extend(domain);
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
        let domain = match (self.fold.domain(), operands.len()) {
            (None, 2) => None,
            (Some(_), 3) => operands.pop(),
            (_, n) => return plan_err!("a Fold node has {} inputs, given {n}", self.inputs.len()),
        };
        FoldNode::try_new(self.fold.clone(), operands, domain)
    }
}

/// The fold in einsum-like notation for `EXPLAIN`: each operand's
/// dimensions, then the outputs, such as `n,d * d,h -> n,h`, with `+ domain`
/// when there is one. A dimension that isn't an output is shown by number.
pub(crate) fn describe(fold: &Fold) -> String {
    let name = |d: DimensionId| match fold.output_of(d) {
        Some(i) => fold.outputs()[i].name.clone(),
        None => format!("#{}", d.0),
    };
    let operands: Vec<String> = fold
        .operands()
        .iter()
        .map(|op| {
            op.keys
                .iter()
                .map(|(d, _)| name(*d))
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect();
    let outputs: Vec<String> = fold.outputs().iter().map(|o| o.name.clone()).collect();
    format!(
        "{} -> {}{}",
        operands.join(" * "),
        outputs.join(","),
        if fold.domain().is_some() {
            " + domain"
        } else {
            ""
        }
    )
}
