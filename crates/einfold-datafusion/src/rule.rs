// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The rewrite that makes DataFusion run folds over joins with EinFold.
//!
//! A *fold over a join* is a join followed by `SUM`, `COUNT` or `AVG` of a
//! product, grouped by some of the join's columns (see [`mod@crate::detect`]).
//! DataFusion normally materializes every joined row and then aggregates.
//! [`EinFoldExec`] instead adds each joined row straight into its group, so
//! the joined rows never exist.
//!
//! Three pieces connect the two:
//!
//! - [`FoldRule`], a logical optimizer rule, replaces each `Aggregate` node
//!   that [`detect`] reads as a two-operand fold with a [`FoldNode`];
//! - [`EinFoldPlanner`], a DataFusion `ExtensionPlanner`, turns each
//!   [`FoldNode`] into an [`EinFoldExec`];
//! - [`enable`] installs both in a session.

use std::collections::BTreeMap;
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::tree_node::Transformed;
use datafusion::common::{Column, DFSchema, DFSchemaRef, ExprSchema, Result};
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::{SessionState, SessionStateBuilder};
use datafusion::logical_expr::{
    cast, Expr, Extension, LogicalPlan, Projection, UserDefinedLogicalNode,
    UserDefinedLogicalNodeCore,
};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use einfold_ir::{Dim, Fold};

use crate::detect::{detect, FoldMatch};
use crate::exec::EinFoldExec;
use crate::kernel::aggregate_field;

/// Turn einfold on in a DataFusion session: add [`FoldRule`] after
/// DataFusion's own optimizer rules, and plan with [`EinFoldQueryPlanner`].
///
/// ```no_run
/// use datafusion::execution::SessionStateBuilder;
/// use datafusion::prelude::SessionContext;
///
/// let state = einfold_datafusion::enable(SessionStateBuilder::new().with_default_features());
/// let ctx = SessionContext::new_with_state(state.build());
/// ```
///
/// This replaces the builder's query planner. A session that needs its own
/// query planner should instead add [`FoldRule`] with
/// `SessionStateBuilder::with_optimizer_rule`, and include [`EinFoldPlanner`]
/// in the extension planners its physical planner uses, for example through
/// `DefaultPhysicalPlanner::with_extension_planners`.
pub fn enable(builder: SessionStateBuilder) -> SessionStateBuilder {
    builder
        .with_optimizer_rule(Arc::new(FoldRule))
        .with_query_planner(Arc::new(EinFoldQueryPlanner))
}

/// A logical plan node that computes a two-operand fold over a join.
///
/// Each input produces one operand: its dimension columns in the order of the
/// operand's `dims`, then its `Float64` factor. The output has one column per
/// output dimension, named after it, then the aggregate's value,
/// [`VALUE_FIELD`]: a nullable `Float64` for `SUM` and `AVG`, and a non-NULL
/// `Int64` for `COUNT`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldNode {
    fold: Fold,
    inputs: Vec<LogicalPlan>,
    schema: DFSchemaRef,
}

/// The name of a [`FoldNode`]'s value column.
pub const VALUE_FIELD: &str = "__einfold_value";

impl FoldNode {
    /// The fold this node computes.
    pub fn fold(&self) -> &Fold {
        &self.fold
    }
}

// `Fold` has no `Hash` or ordering of its own; its debug form identifies it.
impl Hash for FoldNode {
    fn hash<H: Hasher>(&self, state: &mut H) {
        format!("{:?}", self.fold).hash(state);
        self.inputs.hash(state);
    }
}

impl PartialOrd for FoldNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let key = |n: &Self| (format!("{:?}", n.fold), n.inputs.clone());
        key(self).partial_cmp(&key(other))
    }
}

impl UserDefinedLogicalNodeCore for FoldNode {
    fn name(&self) -> &str {
        "FoldNode"
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
        write!(f, "FoldNode: {}", self.fold)
    }

    fn with_exprs_and_inputs(&self, _exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> Result<Self> {
        Ok(FoldNode {
            fold: self.fold.clone(),
            inputs,
            schema: self.schema.clone(),
        })
    }

    // The defaults for pushing filters and pruning columns treat every column
    // as needed, so later optimizer passes leave the node's inputs intact.
}

/// The optimizer rule: replace each `Aggregate` that [`detect`] reads as a
/// fold with exactly two operands by a [`FoldNode`], under a `Projection`
/// whose output schema is the `Aggregate`'s own: the same names, qualifiers,
/// types and nullability. Every other plan is left unchanged.
#[derive(Debug, Default)]
pub struct FoldRule;

impl OptimizerRule for FoldRule {
    fn name(&self) -> &str {
        "einfold_fold"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        // Declining is always safe: any doubt leaves the plan as it was.
        match detect(&plan).and_then(|m| rewrite(&m)) {
            Some(new) => Ok(Transformed::yes(new)),
            None => Ok(Transformed::no(plan)),
        }
    }
}

/// The replacement for a detected fold, or `None` to leave it alone.
fn rewrite(m: &FoldMatch) -> Option<LogicalPlan> {
    if m.operands.len() != 2 {
        return None;
    }
    // Every column of one dimension must have one type: the kernel compares
    // keys by value, and the output column takes that type.
    let mut types: BTreeMap<&Dim, &DataType> = BTreeMap::new();
    for (op, input) in m.fold.operands().iter().zip(&m.operands) {
        for (dim, col) in op.dims.iter().zip(&input.dim_columns) {
            let ty = input.plan.schema().field_from_column(col).ok()?.data_type();
            if *types.entry(dim).or_insert(ty) != ty {
                return None;
            }
        }
    }

    // Inputs: the dimension columns, then the factor as a `Float64`.
    let mut inputs = Vec::new();
    for input in &m.operands {
        let mut exprs: Vec<Expr> = input
            .dim_columns
            .iter()
            .enumerate()
            .map(|(i, c)| Expr::Column(c.clone()).alias(format!("__einfold_d{i}")))
            .collect();
        exprs.push(cast(input.value.clone(), DataType::Float64).alias("__einfold_v"));
        let proj = Projection::try_new(exprs, input.plan.clone()).ok()?;
        inputs.push(LogicalPlan::Projection(proj));
    }

    // The node's columns take the `Aggregate`'s types and nullability, which
    // must be what the kernel produces.
    let agg_field = |c: &Column| m.schema.field_from_column(c).ok();
    let mut fields = Vec::new();
    for dim in m.fold.output() {
        let (col, _) = m.group_outputs.iter().find(|(_, g)| g == dim)?;
        let f = agg_field(col)?;
        if Some(&f.data_type()) != types.get(dim) {
            return None;
        }
        fields.push(Field::new(
            dim.0.clone(),
            f.data_type().clone(),
            f.is_nullable(),
        ));
    }
    let value = agg_field(&m.value_output)?;
    let kernel_value = aggregate_field(m.fold.aggregate());
    if value.data_type() != kernel_value.data_type()
        || (kernel_value.is_nullable() && !value.is_nullable())
    {
        return None;
    }
    fields.push(Field::new(
        VALUE_FIELD,
        value.data_type().clone(),
        value.is_nullable(),
    ));
    let schema = DFSchema::from_unqualified_fields(fields.into(), Default::default()).ok()?;
    let node = LogicalPlan::Extension(Extension {
        node: Arc::new(FoldNode {
            fold: m.fold.clone(),
            inputs,
            schema: Arc::new(schema),
        }),
    });

    // Rename to the `Aggregate`'s columns. Two group columns may hold one
    // dimension (`GROUP BY a.k, b.k` where `a.k = b.k`); both read its column.
    let rename = |name: &str, i: usize| {
        let (q, f) = m.schema.qualified_field(i);
        Expr::Column(Column::new_unqualified(name)).alias_qualified(q.cloned(), f.name())
    };
    let mut exprs: Vec<Expr> = (m.group_outputs.iter().enumerate())
        .map(|(i, (_, dim))| rename(&dim.0, i))
        .collect();
    exprs.push(rename(VALUE_FIELD, m.group_outputs.len()));
    // Reusing the `Aggregate`'s schema object keeps the output identical,
    // including what DataFusion knows about which columns determine others.
    let proj = Projection::try_new_with_schema(exprs, Arc::new(node), m.schema.clone()).ok()?;
    Some(LogicalPlan::Projection(proj))
}

/// A DataFusion `ExtensionPlanner` that plans [`FoldNode`] as
/// [`EinFoldExec`], and leaves every other node to the next planner.
#[derive(Debug, Default)]
pub struct EinFoldPlanner;

#[async_trait]
impl ExtensionPlanner for EinFoldPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(node) = node.as_any().downcast_ref::<FoldNode>() else {
            return Ok(None);
        };
        let [a, b] = physical_inputs else {
            return Ok(None);
        };
        let schema = Arc::new(node.schema.as_arrow().clone());
        let exec = EinFoldExec::try_new(node.fold.clone(), a.clone(), b.clone(), schema)?;
        Ok(Some(Arc::new(exec)))
    }
}

/// DataFusion's default physical planner, with [`EinFoldPlanner`] added.
#[derive(Debug, Default)]
pub struct EinFoldQueryPlanner;

#[async_trait]
impl QueryPlanner for EinFoldQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(EinFoldPlanner)])
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}
