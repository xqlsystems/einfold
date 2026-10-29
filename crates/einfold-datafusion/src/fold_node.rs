// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`FoldNode`]: a fold over a join as a node of DataFusion's logical plan.
//!
//! The node stands for the join and aggregate it replaces (see
//! [`einfold_ir::fold`]). DataFusion's optimizer leaves it alone, and
//! [`EinFoldPlanner`](crate::EinFoldPlanner) turns it into
//! [`EinFoldExec`](crate::EinFoldExec).

use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::{Field, Schema};
use datafusion::common::{plan_err, DFSchema, DFSchemaRef, Result};
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, UserDefinedLogicalNodeCore};
use einfold_ir::{ColumnIndex, Comparison, DimensionId, Fold};
use einfold_kernels::DomainRows;

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
    /// Whether the fallback runs wherever the dense algorithm declines, or
    /// only where memory runs short.
    host_on_decline: bool,
    /// Whether [`DetectFolds`](crate::DetectFolds) made this node.
    found: bool,
    /// Whether the node computes its fold only when no two domain rows reach
    /// one group, and runs its fallback otherwise.
    domain_rows: DomainRows,
    /// A Boolean column of the domain whose rows get a NULL value where it is
    /// true, whatever their group's.
    null_when: Option<ColumnIndex>,
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
            host_on_decline: false,
            found: false,
            domain_rows: DomainRows::Any,
            null_when: None,
        })
    }

    /// This node, keeping `plan` as the plan it replaced. When the dense
    /// algorithm doesn't apply, [`EinFoldExec`](crate::EinFoldExec) runs
    /// `plan` instead of the discrimination algorithm, so the host's own plan is the
    /// slowest the fold can be. It runs `plan` too where the session's
    /// memory pool can't give it what it needs.
    #[must_use]
    pub fn with_fallback(mut self, plan: LogicalPlan) -> Self {
        self.fallback = Some(plan);
        self.host_on_decline = true;
        self.found = true;
        self
    }

    /// This node, keeping `plan` as the plan it replaced, which
    /// [`EinFoldExec`](crate::EinFoldExec) runs only where the session's
    /// memory pool can't give it what it needs: the host's plan can spill to
    /// disk, and einfold's kernels can't.
    #[must_use]
    pub fn with_memory_fallback(mut self, plan: LogicalPlan) -> Self {
        self.fallback = Some(plan);
        self.host_on_decline = false;
        self.found = true;
        self
    }

    /// Whether the node runs its fallback wherever the dense algorithm
    /// declines (see [`FoldNode::with_fallback`]), and not only where memory
    /// runs short.
    #[must_use]
    pub fn host_on_decline(&self) -> bool {
        self.host_on_decline
    }

    /// This node, computing its fold only when no two of its domain's rows
    /// reach one group, and running its fallback otherwise: for a plan that
    /// joins a fold back to its domain and groups again by the domain's keys,
    /// which is the fold from its domain exactly then.
    pub(crate) fn requiring_unique_domain(mut self) -> Self {
        self.domain_rows = DomainRows::UniqueKeys;
        self
    }

    /// Whether the node computes its fold only when its domain's keys are
    /// unique ([`DomainRows::UniqueKeys`]), running its fallback otherwise.
    #[must_use]
    pub fn domain_rows(&self) -> DomainRows {
        self.domain_rows
    }

    /// This node, giving a NULL value to each row of its domain where the
    /// domain's Boolean column `column` is true, as `CASE WHEN <column> THEN
    /// NULL ELSE <value> END` does.
    pub(crate) fn with_null_when(mut self, column: ColumnIndex) -> Self {
        self.null_when = Some(column);
        self
    }

    /// The domain's column that makes a row's value NULL where it is true, if
    /// any: such a row gets a NULL value, as `CASE WHEN <column> THEN NULL
    /// ELSE <value> END` gives it.
    #[must_use]
    pub fn null_when(&self) -> Option<ColumnIndex> {
        self.null_when
    }

    /// Whether [`DetectFolds`](crate::DetectFolds) made this node.
    pub(crate) fn found(&self) -> bool {
        self.found
    }

    /// The plan this node replaced, if any.
    #[must_use]
    pub fn fallback(&self) -> Option<&LogicalPlan> {
        self.fallback.as_ref()
    }

    /// The node's inputs: its operands, then its domain.
    #[must_use]
    pub fn inputs_owned(&self) -> Vec<LogicalPlan> {
        self.inputs.clone()
    }

    /// The fold this node computes.
    #[must_use]
    pub fn fold(&self) -> &Fold {
        &self.fold
    }

    /// This node as a logical plan.
    #[must_use]
    pub fn into_plan(self) -> LogicalPlan {
        LogicalPlan::Extension(Extension {
            node: Arc::new(self),
        })
    }
}

// The schema follows from the fold and the inputs, so ordering ignores it.
impl PartialOrd for FoldNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        fn key(n: &FoldNode) -> impl PartialOrd + '_ {
            (
                &n.fold,
                &n.inputs,
                &n.fallback,
                n.host_on_decline,
                n.found,
                n.domain_rows,
                n.null_when,
            )
        }
        key(self).partial_cmp(&key(other))
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
        write!(f, "Fold: {}", describe(&self.fold, self.domain_rows))?;
        match self.null_when {
            Some(ColumnIndex(c)) => write!(f, ", NULL where domain column {c}"),
            None => Ok(()),
        }
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
            host_on_decline: self.host_on_decline,
            found: self.found,
            domain_rows: self.domain_rows,
            null_when: self.null_when,
            ..node
        })
    }
}

/// The fold in einsum notation for `EXPLAIN`: each operand's dimensions,
/// then the outputs, such as `a,b * b,c -> a,c` for a matrix product, with
/// `+ domain` when there is one (`+ unique domain` when the node requires
/// its keys to be unique), and its conditions after `where`. Dimensions are lettered by number, as the
/// fold names none; past `z`, they are shown as `#26`, `#27`, ….
pub(crate) fn describe(fold: &Fold, domain_rows: DomainRows) -> String {
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
                .map(|k| name(k.dimension))
                .collect::<Vec<_>>()
                .join(",")
        })
        .collect();
    let outputs: Vec<String> = fold.outputs().iter().map(|o| name(o.dimension)).collect();
    let conditions: Vec<String> = (fold.conditions().iter())
        .map(|c| {
            let op = match c.comparison {
                Comparison::Less => "<",
                Comparison::LessOrEqual => "<=",
                Comparison::Greater => ">",
                Comparison::GreaterOrEqual => ">=",
                _ => "?",
            };
            format!(" {} {op} {}", name(c.left), name(c.right))
        })
        .collect();
    format!(
        "{} -> {}{}{}",
        operands.join(" * "),
        outputs.join(","),
        match (fold.domain(), domain_rows) {
            (None, _) => "",
            (Some(_), DomainRows::Any) => " + domain",
            (Some(_), DomainRows::UniqueKeys) => " + unique domain",
        },
        match conditions.is_empty() {
            true => String::new(),
            false => format!(" where{}", conditions.join(" and")),
        }
    )
}
