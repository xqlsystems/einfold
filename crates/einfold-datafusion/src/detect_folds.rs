// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`DetectFolds`]: an optimizer rule that finds folds over joins in
//! DataFusion's plans and replaces them with [`FoldNode`]s.
//!
//! It finds two shapes, and leaves everything else alone:
//!
//! 1. **A fold.** In SQL,
//!
//!    ```sql
//!    SELECT <columns of l or r>, SUM(<expr of l> * <expr of r>)
//!    FROM l JOIN r ON <equalities between l and r> GROUP BY <those columns>
//!    ```
//!
//!    with one aggregate, a `SUM` of `DOUBLE`s with no `DISTINCT`, `FILTER` or
//!    ordering, and at least one group key. (Without `GROUP BY`, SQL returns a
//!    row even when nothing joined, and a fold doesn't.) A condition of the
//!    join that isn't an equality, such as causal attention's `q.t >= k.t`,
//!    is a mask, which comes later; such joins are left alone.
//! 2. **A fold from a domain.** A projection over `d LEFT JOIN f ON <d's
//!    columns> = <f's group keys>`, where `f` is a fold found by shape 1, the
//!    join matches every one of `f`'s group keys, and the projection reads
//!    only `d`'s columns and `f`'s sum. The fold gets `d` as its domain, the
//!    relation whose rows its output has. This is how ddx, an XQL Systems
//!    project for automatic differentiation of SQL queries, gives every row
//!    of a table its gradient; ddx calls these rows its "fill".
//!
//! The rule runs after DataFusion's own rules, so filters are already pushed
//! below the join, and keeps every plan's schema exactly: names, qualifiers,
//! types and nullability. [`EinFoldExec`](crate::EinFoldExec) runs each
//! node by the dense algorithm where its keys and rows allow, and by the
//! discrimination algorithm otherwise. A node that can be computed only under
//! a condition known when it runs, such as a fold joined back to its domain,
//! keeps the plan it replaced and runs that where the condition fails. With
//! [`DetectFolds::with_host_fallback`], every node keeps it, and runs it
//! wherever the dense algorithm doesn't apply.

use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::datatypes::DataType;
use datafusion::common::tree_node::Transformed;
use datafusion::common::{Column, DFSchema, NullEquality, Result};
use datafusion::functions_aggregate::sum::Sum;
use datafusion::logical_expr::{
    Aggregate, BinaryExpr, Expr, Join, JoinType, LogicalPlan, Operator, Projection,
};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use einfold_ir::{
    Aggregate as FoldAggregate, ColumnIndex, Comparison, Condition, Dimension, DimensionId, Domain,
    DomainKey, Fold, Key, KeyEquality, Operand, Output, RowValue,
};

use crate::detect_pipelines;
use crate::fold_node::FoldNode;
use crate::join_back;

/// The prefix of the names this rule gives the columns of the nodes it makes,
/// chosen not to clash with a domain's own columns.
pub(crate) const PREFIX: &str = "__einfold_";

/// Finds folds over joins in DataFusion's optimized plans (see the module's
/// documentation). [`EinFoldQueryPlanner::detecting`](crate::EinFoldQueryPlanner::detecting)
/// applies it to each plan just before planning it, which [`install`](crate::install)
/// sets up. It can also be added after DataFusion's own optimizer rules, but
/// then optimized plans contain `Fold` nodes, which a Substrait producer, such
/// as ddx's, can't encode.
#[derive(Clone, Copy, Debug, Default)]
pub struct DetectFolds {
    fallback: Fallback,
}

/// What the nodes [`DetectFolds`] makes run where the dense algorithm
/// doesn't apply.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Fallback {
    /// The plan each node replaced, so a detected fold is never slower than
    /// the host's own plan.
    HostPlan,
    /// EinFold's discrimination algorithm, for any keys and density. The
    /// default: on the folds in einfold's demos and xarray-sql's benchmarks
    /// that the dense algorithm can't take (sparse weight tables, graph
    /// edges, embedding lookups), it beats the host's plan.
    #[default]
    Kernels,
}

impl DetectFolds {
    /// The rule, but with nodes that keep the plan they replaced and run it
    /// wherever the dense algorithm doesn't apply, rather than the
    /// discrimination algorithm: so that a detected fold is never slower
    /// than the host's own plan by more than reading its inputs once.
    #[must_use]
    pub fn with_host_fallback() -> Self {
        DetectFolds {
            fallback: Fallback::HostPlan,
        }
    }
}

impl DetectFolds {
    /// `plan` with every fold pipeline and fold this rule finds replaced by
    /// its node. Pipelines are found first, from the top down, since a
    /// pipeline's producer is itself a fold, which the rule's bottom-up pass
    /// would otherwise replace first. As an optimizer rule, [`DetectFolds`]
    /// finds folds only.
    pub fn detect(&self, plan: LogicalPlan) -> Result<LogicalPlan> {
        use datafusion::common::tree_node::TreeNode;
        use datafusion::optimizer::OptimizerContext;
        let plan = plan
            .transform_down(|node| {
                let found = match &node {
                    LogicalPlan::Projection(p) => detect_pipelines::pipeline(p)?,
                    _ => None,
                };
                Ok(match found {
                    Some(rewritten) => Transformed::yes(rewritten),
                    None => Transformed::no(node),
                })
            })?
            .data;
        let config = OptimizerContext::new();
        Ok(plan.transform_up(|node| self.rewrite(node, &config))?.data)
    }
}

impl OptimizerRule for DetectFolds {
    fn name(&self) -> &str {
        "detect_folds"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let found = match &plan {
            LogicalPlan::Aggregate(agg) => fold(agg, self.fallback)?,
            LogicalPlan::Projection(p) => match from_domain(p, self.fallback)? {
                // A fold joined back to its domain needs the host's plan
                // for when the domain's keys aren't unique.
                None => join_back::joined_back(p)?,
                found => found,
            },
            _ => None,
        };
        Ok(match found {
            Some(rewritten) => Transformed::yes(rewritten),
            None => Transformed::no(plan),
        })
    }
}

// --- shape 1: a fold ----------------------------------------------------------

/// `e` with each column a projection in `projections` (outermost first)
/// produced replaced by the expression that produced it, aliases removed.
pub(crate) fn resolve(e: &Expr, projections: &[&Projection]) -> Result<Expr> {
    use datafusion::common::tree_node::TreeNode;
    let mut e = e.clone().unalias();
    for p in projections {
        e = e
            .transform(|x| match &x {
                Expr::Column(c) => match p.schema.index_of_column(c) {
                    Ok(i) => Ok(Transformed::yes(p.expr[i].clone().unalias())),
                    Err(_) => Ok(Transformed::no(x)),
                },
                _ => Ok(Transformed::no(x)),
            })?
            .data;
    }
    Ok(e)
}

/// Which side of a join every column of `e` reads: `Some(0)` for the left,
/// `Some(1)` for the right, `None` for both, neither, or a column of neither.
fn side(e: &Expr, inputs: [&DFSchema; 2]) -> Option<usize> {
    let cols = e.column_refs();
    if cols.is_empty() {
        return None;
    }
    (0..2).find(|&s| cols.iter().all(|c| inputs[s].has_column(c)))
}

/// Scalar functions that a factor may apply, when every argument is
/// `Float64`. Each returns NaN, ±inf or NULL rather than an error for any
/// input, which `tests/detect_vs_host.rs` checks over NaN, ±inf, ±0,
/// negatives and NULL. `power` and `sqrt` are left out: `power(0, -1)` and
/// `sqrt(-1)` raise errors in DataFusion 55.
pub const MOVABLE_FUNCTIONS: &[&str] = &["abs", "exp", "ln", "tanh", "signum"];

/// Whether `e`, an expression over `schema`, may be computed on every row of
/// a join's input rather than on the joined rows only, as the host's plan
/// computes it. A fold's operands compute their keys and factors that way,
/// so a row that joins nothing must not make the query fail where the host
/// answers it: `CAST('oops' AS DOUBLE)` or `n / 0` on such a row would. So
/// only deterministic, row-wise expressions that can't raise an error move;
/// a volatile one, such as `random()`, would also be drawn once per input
/// row rather than once per joined pair.
pub(crate) fn movable(e: &Expr, schema: &DFSchema) -> bool {
    use datafusion::common::tree_node::TreeNode;
    use datafusion::logical_expr::ExprSchemable;
    let ty = |e: &Expr| e.get_type(schema).ok();
    let arithmetic = |e: &Expr| ty(e).is_some_and(|t| t.is_integer() || t.is_floating());
    let fails = |e: &Expr| {
        let ok = match e {
            Expr::Alias(_) | Expr::Column(_) | Expr::Literal(..) => true,
            // Integer arithmetic wraps in DataFusion, and float arithmetic
            // gives ±inf or NaN. `/` and `%` can fail on integers.
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => match op {
                Operator::Plus | Operator::Minus | Operator::Multiply => {
                    arithmetic(left) && arithmetic(right)
                }
                Operator::Eq
                | Operator::NotEq
                | Operator::Lt
                | Operator::LtEq
                | Operator::Gt
                | Operator::GtEq
                | Operator::And
                | Operator::Or
                | Operator::IsDistinctFrom
                | Operator::IsNotDistinctFrom => true,
                _ => false,
            },
            Expr::Negative(inner) => arithmetic(inner),
            Expr::Not(_)
            | Expr::IsNotNull(_)
            | Expr::IsNull(_)
            | Expr::IsTrue(_)
            | Expr::IsFalse(_)
            | Expr::IsUnknown(_)
            | Expr::IsNotTrue(_)
            | Expr::IsNotFalse(_)
            | Expr::IsNotUnknown(_)
            | Expr::Between(_)
            | Expr::Case(_)
            | Expr::TryCast(_) => true,
            Expr::InList(list) => list.list.iter().all(|e| matches!(e, Expr::Literal(..))),
            Expr::Cast(cast) => {
                ty(&cast.expr).is_some_and(|from| widens(&from, cast.field.data_type()))
            }
            Expr::ScalarFunction(f) => {
                MOVABLE_FUNCTIONS.contains(&f.name())
                    && f.args.iter().all(|a| ty(a) == Some(DataType::Float64))
            }
            _ => false,
        };
        Ok(!ok)
    };
    // Every node of `e` must be one of those: a CASE's branches are nodes too.
    !e.exists(fails).expect("the check doesn't fail")
}

/// Whether a cast from `from` to `to` can't fail: any integer or float to
/// `Float64`, rounding if it must, or an integer to an integer type that
/// holds all its values, as DataFusion's coercion inserts for `INT * 2`.
fn widens(from: &DataType, to: &DataType) -> bool {
    use DataType::{Int16, Int32, Int8, UInt16, UInt32, UInt8};
    let bits = |t: &DataType| match t {
        Int8 | UInt8 => 8,
        Int16 | UInt16 => 16,
        Int32 | UInt32 => 32,
        _ => 64,
    };
    match (from, to) {
        (f, DataType::Float64) => f.is_integer() || f.is_floating(),
        (f, t) if f.is_signed_integer() && t.is_signed_integer() => bits(f) <= bits(t),
        (f, t) if f.is_unsigned_integer() && t.is_unsigned_integer() => bits(f) <= bits(t),
        (f, t) if f.is_unsigned_integer() && t.is_signed_integer() => bits(f) < bits(t),
        _ => false,
    }
}

/// The sum's argument, if `agg` computes exactly one plain `SUM`.
pub(crate) fn sum_argument(agg: &Aggregate) -> Option<Expr> {
    match agg.aggr_expr.len() {
        1 => sum_argument_at(agg, 0),
        _ => None,
    }
}

/// The argument of `agg`'s aggregate `a`, if it is a plain `SUM`.
pub(crate) fn sum_argument_at(agg: &Aggregate, a: usize) -> Option<Expr> {
    let aggregate = agg.aggr_expr.get(a)?;
    let Expr::AggregateFunction(f) = aggregate.clone().unalias() else {
        return None;
    };
    let p = &f.params;
    let plain = !p.distinct && p.filter.is_none() && p.order_by.is_empty() && p.args.len() == 1;
    // By type, not by name, as DataFusion asks of rules: a session may
    // register another function under the name `sum`.
    let function: &dyn Any = f.func.inner().as_ref();
    (function.is::<Sum>() && plain).then(|| p.args[0].clone())
}

/// A fold found in an aggregate: its description, and its operands' plans.
pub(crate) struct Found {
    pub(crate) fold: Fold,
    pub(crate) operands: Vec<LogicalPlan>,
}

/// `agg` as a fold, if it is one: a [`FoldNode`] under a projection giving it
/// `agg`'s schema. A fold whose join has a condition, a mask, is left alone:
/// the fold kernels don't compute masks yet.
fn fold(agg: &Aggregate, fallback: Fallback) -> Result<Option<LogicalPlan>> {
    let Some(Found {
        fold: spec,
        operands: plans,
    }) = analyze(agg)?
    else {
        return Ok(None);
    };
    if !spec.conditions().is_empty() {
        return Ok(None);
    }
    let original = LogicalPlan::Aggregate(agg.clone());
    let names: Vec<String> = (0..agg.group_expr.len())
        .map(|i| format!("{PREFIX}o{i}"))
        .chain([format!("{PREFIX}v")])
        .collect();
    let Ok(node) = FoldNode::try_new(spec, plans, None, names.clone()) else {
        return Ok(None);
    };
    let node = match fallback {
        Fallback::HostPlan => node.with_fallback(original),
        Fallback::Kernels => node.marked_found(),
    }
    .into_plan();
    // Give the node the aggregate's schema: its names, qualifiers, types
    // and nullability.
    restore(node, &names, &agg.schema)
}

/// `agg` as a fold, if it is one: a `SUM` of a product of one factor from
/// each side of an inner equi-join, grouped by columns, through projections.
/// A condition of the join that compares two of the fold's dimensions, such
/// as `q.t >= k.t`, becomes one of the fold's conditions; any other declines.
pub(crate) fn analyze(agg: &Aggregate) -> Result<Option<Found>> {
    use datafusion::logical_expr::ExprSchemable;
    let Some(arg) = sum_argument(agg) else {
        return Ok(None);
    };
    if agg.group_expr.is_empty() {
        return Ok(None);
    }
    // Down through projections to the join; a filter between them would
    // read both sides, or DataFusion would have pushed it below.
    let mut projections = Vec::new();
    let mut below = agg.input.as_ref();
    let join = loop {
        match below {
            LogicalPlan::Projection(p) => {
                projections.push(p);
                below = p.input.as_ref();
            }
            LogicalPlan::Join(j) => break j,
            _ => return Ok(None),
        }
    };
    if join.join_type != JoinType::Inner || join.on.is_empty() {
        return Ok(None);
    }
    let inputs = [join.left.schema().as_ref(), join.right.schema().as_ref()];
    let Expr::BinaryExpr(BinaryExpr {
        left,
        op: Operator::Multiply,
        right,
    }) = resolve(&arg, &projections)?
    else {
        return Ok(None);
    };
    let mut factors: [Option<Expr>; 2] = [None, None];
    for f in [*left, *right] {
        let Some(s) = side(&f, inputs) else {
            return Ok(None);
        };
        if factors[s].is_some()
            || f.get_type(inputs[s])? != DataType::Float64
            || !movable(&f, inputs[s])
        {
            return Ok(None);
        }
        factors[s] = Some(f);
    }
    let [Some(lf), Some(rf)] = factors else {
        return Ok(None);
    };

    // Dimensions: one per pair of join keys, then one per group key that
    // isn't a join key. Each operand projects its keys, then its factor.
    let equality = match join.null_equality {
        NullEquality::NullEqualsNull => KeyEquality::NotDistinct,
        NullEquality::NullEqualsNothing => KeyEquality::Equal,
    };
    let mut dimensions = Vec::new();
    let mut columns: [Vec<Expr>; 2] = [Vec::new(), Vec::new()];
    let mut keys: [Vec<Key>; 2] = [Vec::new(), Vec::new()];
    let mut hold = |s: usize, d: usize, e: Expr, columns: &mut [Vec<Expr>; 2]| {
        if columns[s].contains(&e) {
            return false;
        }
        keys[s].push(Key::new(DimensionId(d), ColumnIndex(columns[s].len())));
        columns[s].push(e);
        true
    };
    for (l, r) in &join.on {
        if !movable(l, inputs[0]) || !movable(r, inputs[1]) {
            return Ok(None);
        }
        let d = dimensions.len();
        dimensions.push(Dimension { equality });
        if !hold(0, d, l.clone(), &mut columns) || !hold(1, d, r.clone(), &mut columns) {
            return Ok(None);
        }
    }
    let mut outputs = Vec::new();
    for g in &agg.group_expr {
        let g = resolve(g, &projections)?;
        let (Expr::Column(_), Some(s)) = (&g, side(&g, inputs)) else {
            return Ok(None);
        };
        let d = match join.on.iter().position(|pair| [&pair.0, &pair.1][s] == &g) {
            Some(d) => d,
            None => {
                dimensions.push(Dimension { equality });
                if !hold(s, dimensions.len() - 1, g, &mut columns) {
                    return Ok(None);
                }
                dimensions.len() - 1
            }
        };
        if outputs
            .iter()
            .any(|o: &Output| o.dimension == DimensionId(d))
        {
            return Ok(None);
        }
        outputs.push(Output {
            dimension: DimensionId(d),
            domain_key: None,
        });
    }
    // The join's condition: comparisons between two of the dimensions.
    let mut conditions = Vec::new();
    if let Some(filter) = &join.filter {
        for c in datafusion::logical_expr::utils::split_conjunction(filter) {
            let Some(condition) = comparison(c, &columns, &keys, inputs) else {
                return Ok(None);
            };
            conditions.push(condition);
        }
    }
    let mut operands = Vec::new();
    for (s, (input, factor)) in [(&join.left, lf), (&join.right, rf)]
        .into_iter()
        .enumerate()
    {
        let mut exprs: Vec<Expr> = columns[s]
            .iter()
            .enumerate()
            .map(|(c, e)| e.clone().alias(format!("k{c}")))
            .collect();
        exprs.push(factor.alias("f"));
        operands.push((
            Operand {
                keys: std::mem::take(&mut keys[s]),
                factor: Some(ColumnIndex(columns[s].len())),
            },
            LogicalPlan::Projection(Projection::try_new(exprs, Arc::clone(input))?),
        ));
    }
    let operand_specs = operands.iter().map(|(o, _)| o.clone()).collect();
    let Ok(spec) = Fold::try_new(
        dimensions,
        operand_specs,
        outputs,
        RowValue::Product,
        FoldAggregate::sum(),
        None,
    ) else {
        return Ok(None);
    };
    let Ok(spec) = spec.with_conditions(conditions) else {
        return Ok(None);
    };
    let operands = operands.into_iter().map(|(_, p)| p).collect();
    Ok(Some(Found {
        fold: spec,
        operands,
    }))
}

/// `e` as a condition between two of a fold's dimensions: `l <op> r` with
/// `l` and `r` columns that an operand holds as keys (`columns` and `keys`,
/// per side of the join), on different sides.
fn comparison(
    e: &Expr,
    columns: &[Vec<Expr>; 2],
    keys: &[Vec<Key>; 2],
    inputs: [&DFSchema; 2],
) -> Option<Condition> {
    let Expr::BinaryExpr(BinaryExpr { left, op, right }) = e else {
        return None;
    };
    let comparison = match op {
        Operator::Lt => Comparison::Less,
        Operator::LtEq => Comparison::LessOrEqual,
        Operator::Gt => Comparison::Greater,
        Operator::GtEq => Comparison::GreaterOrEqual,
        _ => return None,
    };
    let dimension = |e: &Expr| -> Option<(usize, DimensionId)> {
        let s = side(e, inputs)?;
        let c = columns[s].iter().position(|k| k == e)?;
        let key = keys[s].iter().find(|k| k.column == ColumnIndex(c))?;
        Some((s, key.dimension))
    };
    let ((ls, l), (rs, r)) = (dimension(left)?, dimension(right)?);
    (ls != rs && l != r).then_some(Condition {
        left: l,
        comparison,
        right: r,
    })
}

/// `plan` with each [`FoldNode`] that keeps the plan it replaced swapped
/// back for that plan, under the node's column names, so that whatever reads
/// the node still resolves. A node's fallback is the host's own plan, which
/// must not run einfold again: an operator that falls back to a plan
/// holding another of its kind would read the same operands once more.
pub(crate) fn host_plan(plan: LogicalPlan) -> Result<LogicalPlan> {
    use datafusion::common::tree_node::TreeNode;
    let swapped = plan.transform_up(|plan| {
        let LogicalPlan::Extension(e) = &plan else {
            return Ok(Transformed::no(plan));
        };
        let Some(node) = e.node.as_any().downcast_ref::<FoldNode>() else {
            return Ok(Transformed::no(plan));
        };
        let Some(fallback) = node.fallback() else {
            return Ok(Transformed::no(plan));
        };
        let names = (e.node.schema().fields().iter()).map(|f| f.name().clone());
        let exprs: Vec<Expr> = (fallback.schema().columns().into_iter())
            .zip(names)
            .map(|(c, name)| Expr::Column(c).alias(name))
            .collect();
        let projection = Projection::try_new(exprs, Arc::new(fallback.clone()))?;
        Ok(Transformed::yes(LogicalPlan::Projection(projection)))
    })?;
    Ok(swapped.data)
}

/// `node`, whose columns are `names`, renamed to `schema`'s columns, or
/// `None` if the result's schema wouldn't be exactly `schema`.
pub(crate) fn restore(
    node: LogicalPlan,
    names: &[String],
    schema: &DFSchema,
) -> Result<Option<LogicalPlan>> {
    let exprs: Vec<Expr> = names
        .iter()
        .zip(schema.iter())
        .map(|(n, (q, f))| {
            Expr::Column(Column::new_unqualified(n)).alias_qualified(q.cloned(), f.name())
        })
        .collect();
    let projection = Projection::try_new(exprs, Arc::new(node))?;
    Ok(same_schema(&projection.schema, schema).then_some(LogicalPlan::Projection(projection)))
}

/// Whether two schemas have the same columns, qualifiers, types and
/// nullability, which an optimizer rule must keep.
pub(crate) fn same_schema(a: &DFSchema, b: &DFSchema) -> bool {
    a.fields().len() == b.fields().len()
        && a.iter().zip(b.iter()).all(|((qa, fa), (qb, fb))| {
            qa == qb
                && fa.name() == fb.name()
                && fa.data_type() == fb.data_type()
                && fa.is_nullable() == fb.is_nullable()
        })
}

// --- shape 2: a fold from a domain --------------------------------------------

/// The fold this rule made under `plan`, through any aliases and projections
/// that only rename or reorder columns, with, for each column of `plan`, its
/// column in the fold's output: an output's index, or the number of outputs
/// for the sum.
pub(crate) fn found_fold(plan: &LogicalPlan) -> Option<(&FoldNode, Vec<usize>)> {
    match plan {
        LogicalPlan::Extension(e) => {
            let node = e.node.as_any().downcast_ref::<FoldNode>()?;
            let usable = node.fold().domain().is_none() && node.found();
            usable.then(|| (node, (0..=node.fold().outputs().len()).collect()))
        }
        LogicalPlan::SubqueryAlias(a) => found_fold(&a.input),
        LogicalPlan::Projection(p) => {
            let (node, below) = found_fold(&p.input)?;
            let columns = p
                .expr
                .iter()
                .map(|e| match e.clone().unalias() {
                    Expr::Column(c) => p.input.schema().index_of_column(&c).ok().map(|i| below[i]),
                    _ => None,
                })
                .collect::<Option<Vec<usize>>>()?;
            Some((node, columns))
        }
        _ => None,
    }
}

/// `p` over a left join from a relation onto a fold, with the relation as the
/// fold's domain, if that is what `p` is.
fn from_domain(p: &Projection, fallback: Fallback) -> Result<Option<LogicalPlan>> {
    let LogicalPlan::Join(join) = p.input.as_ref() else {
        return Ok(None);
    };
    let Join {
        left: domain,
        right,
        on,
        filter,
        join_type,
        null_equality,
        ..
    } = join;
    if *join_type != JoinType::Left || filter.is_some() {
        return Ok(None);
    }
    let Some((node, columns)) = found_fold(right) else {
        return Ok(None);
    };
    let fold = node.fold();
    let n = fold.outputs().len();
    // The right side's columns are the fold's outputs, then its sum.
    let right_schema = right.schema();
    let equality = match null_equality {
        NullEquality::NullEqualsNull => KeyEquality::NotDistinct,
        NullEquality::NullEqualsNothing => KeyEquality::Equal,
    };
    let mut keys: Vec<Option<DomainKey>> = vec![None; n];
    for (l, r) in on {
        let (Expr::Column(lc), Expr::Column(rc)) = (l, r) else {
            return Ok(None);
        };
        let (Ok(i), Ok(d)) = (
            right_schema.index_of_column(rc),
            domain.schema().index_of_column(lc),
        ) else {
            return Ok(None);
        };
        let i = columns[i];
        if i >= n
            || keys[i]
                .replace(DomainKey::new(ColumnIndex(d), equality))
                .is_some()
        {
            return Ok(None);
        }
    }
    let Some(keys) = keys.into_iter().collect::<Option<Vec<_>>>() else {
        return Ok(None);
    };
    // The projection may read the domain and the fold's sum, not its keys,
    // which are NULL where the domain matched no group.
    let Some(v) = columns.iter().position(|&c| c == n) else {
        return Ok(None);
    };
    let (value_q, value_field) = right_schema.qualified_field(v);
    let value = Column::new(value_q.cloned(), value_field.name());
    let reads_keys = p.expr.iter().any(|e| {
        e.column_refs()
            .iter()
            .any(|c| right_schema.has_column(c) && **c != value)
    });
    if reads_keys {
        return Ok(None);
    }
    let width = domain.schema().fields().len();
    let spec = Domain {
        carry: (0..width).map(ColumnIndex).collect(),
        missing: None,
    };
    // Each output, now matched with its domain column.
    let outputs = fold
        .outputs()
        .iter()
        .zip(keys)
        .map(|(o, key)| Output {
            domain_key: Some(key),
            ..*o
        })
        .collect();
    let Ok(with_domain) = Fold::try_new(
        fold.dimensions().to_vec(),
        fold.operands().to_vec(),
        outputs,
        fold.row(),
        fold.aggregate().clone(),
        Some(spec),
    ) else {
        return Ok(None);
    };
    // The plan it replaces: the join, as the domain's columns and the sum.
    let mut kept: Vec<Expr> = domain
        .schema()
        .columns()
        .into_iter()
        .map(Expr::Column)
        .collect();
    kept.push(Expr::Column(value.clone()));
    let input = host_plan(p.input.as_ref().clone())?;
    let original = LogicalPlan::Projection(Projection::try_new(kept, Arc::new(input))?);
    let inputs = node.inputs_owned();
    // The domain's columns keep their names, and the value takes one that
    // can't clash with theirs.
    let names: Vec<String> = (domain.schema().fields().iter())
        .map(|f| f.name().clone())
        .chain([format!("{PREFIX}v")])
        .collect();
    let Ok(new) = FoldNode::try_new(
        with_domain,
        inputs[..2].to_vec(),
        Some(domain.as_ref().clone()),
        names.clone(),
    ) else {
        return Ok(None);
    };
    let new = match fallback {
        Fallback::HostPlan => new.with_fallback(original.clone()),
        Fallback::Kernels => new.marked_found(),
    }
    .into_plan();
    // Rename the node's columns to the join's, so `p` reads them as before.
    let Some(renamed) = restore(new, &names, original.schema())? else {
        return Ok(None);
    };
    let projection = Projection::try_new(p.expr.clone(), Arc::new(renamed))?;
    Ok(same_schema(&projection.schema, &p.schema).then_some(LogicalPlan::Projection(projection)))
}
