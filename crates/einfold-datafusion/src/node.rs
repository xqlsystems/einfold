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

use datafusion::arrow::datatypes::{Field, Schema};
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
    /// The plan this node replaced, if it replaced one: its columns are the
    /// node's, in order, though their names may differ.
    fallback: Option<LogicalPlan>,
    /// Whether [`DetectFolds`](crate::DetectFolds) made this node.
    found: bool,
}

impl FoldNode {
    /// The node computing `fold` over `operands`, with `domain` as the domain
    /// relation when `fold` has one, and `names` naming its columns: a fold's
    /// columns are positional (see [`einfold_kernels::schema`]), and the plan
    /// above refers to them by name. Fails when the inputs don't fit the fold,
    /// or when there isn't one name per column.
    pub fn try_new(
        fold: Fold,
        operands: Vec<LogicalPlan>,
        domain: Option<LogicalPlan>,
        names: Vec<String>,
    ) -> Result<Self> {
        let arrow = |p: &LogicalPlan| p.schema().as_arrow().clone();
        let operand_schemas: Vec<_> = operands.iter().map(arrow).collect();
        let domain_schema = domain.as_ref().map(arrow);
        let schema = einfold_kernels::schema(
            &fold,
            &operand_schemas.iter().collect::<Vec<_>>(),
            domain_schema.as_ref(),
        )?;
        if names.len() != schema.fields().len() {
            return plan_err!(
                "a fold with {} columns, given {} names",
                schema.fields().len(),
                names.len()
            );
        }
        let fields: Vec<Field> = (schema.fields().iter().zip(names))
            .map(|(f, name)| f.as_ref().clone().with_name(name))
            .collect();
        let schema = Arc::new(DFSchema::try_from(Schema::new(fields))?);
        let mut inputs = operands;
        inputs.extend(domain);
        Ok(FoldNode {
            fold,
            inputs,
            schema,
            fallback: None,
            found: false,
        })
    }

    /// This node, keeping `plan` as the plan it replaced. When the dense
    /// algorithm doesn't apply, [`EinFoldExec`](crate::EinFoldExec) runs
    /// `plan` instead of the hash algorithm, so the host's own plan is the
    /// slowest the fold can be.
    pub fn with_fallback(mut self, plan: LogicalPlan) -> Self {
        self.fallback = Some(plan);
        self.found = true;
        self
    }

    /// This node, marked as made by [`DetectFolds`](crate::DetectFolds)
    /// without a fallback.
    pub(crate) fn marked_found(mut self) -> Self {
        self.found = true;
        self
    }

    /// Whether [`DetectFolds`](crate::DetectFolds) made this node.
    pub(crate) fn found(&self) -> bool {
        self.found
    }

    /// The plan this node replaced, if any.
    pub fn fallback(&self) -> Option<&LogicalPlan> {
        self.fallback.as_ref()
    }

    /// The node's inputs: its operands, then its domain.
    pub fn inputs_owned(&self) -> Vec<LogicalPlan> {
        self.inputs.clone()
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
        (&self.fold, &self.inputs, &self.fallback, self.found).partial_cmp(&(
            &other.fold,
            &other.inputs,
            &other.fallback,
            other.found,
        ))
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
        let names = self.schema.fields().iter().map(|f| f.name().clone());
        let node = FoldNode::try_new(self.fold.clone(), operands, domain, names.collect())?;
        Ok(FoldNode {
            fallback: self.fallback.clone(),
            found: self.found,
            ..node
        })
    }
}

/// The fold in einsum notation for `EXPLAIN`: each operand's dimensions,
/// then the outputs, such as `a,b * b,c -> a,c` for a matrix product, with
/// `+ domain` when there is one. Dimensions are lettered by number, as the
/// fold names none; past `z`, they are shown as `#26`, `#27`, ….
pub(crate) fn describe(fold: &Fold) -> String {
    let name = |DimensionId(d): DimensionId| match u8::try_from(d) {
        Ok(d) if d < 26 => char::from(b'a' + d).to_string(),
        _ => format!("#{d}"),
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
    let outputs: Vec<String> = fold.outputs().iter().map(|o| name(o.dimension)).collect();
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
