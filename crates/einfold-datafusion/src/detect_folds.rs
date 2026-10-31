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
//!    of a table its gradient; ddx calls these rows its "fill". Where the
//!    projection reads the sum only through `COALESCE(sum, c)` or ddx's
//!    `CASE … WHEN sum IS NULL THEN c ELSE sum END`, the fold computes `c`
//!    for a group no row reached, as the projection would make of it (see
//!    [`fill_default`]), so the dense algorithm can take operands with
//!    missing rows, such as a causal mask's triangle.
//!
//! The rule runs after DataFusion's own rules, so filters are already pushed
//! below the join, and keeps every plan's schema exactly: names, qualifiers,
//! types and nullability. [`EinFoldExec`](crate::EinFoldExec) runs each
//! node by the dense algorithm where its keys and rows allow, and by the
//! discrimination algorithm otherwise. Every node keeps the plan it
//! replaced, the host's own, and runs it where the session's memory pool
//! can't give einfold what it needs: the host's operators can spill to disk.
//! A node that can be computed only under a condition known when it runs,
//! such as a fold joined back to its domain, also runs it where the
//! condition fails. With [`DetectFolds::with_host_fallback`], every node
//! runs it wherever the dense algorithm doesn't apply.

use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::datatypes::DataType;
use datafusion::common::tree_node::Transformed;
use datafusion::common::{Column, DFSchema, NullEquality, Result, ScalarValue};
use datafusion::functions_aggregate::sum::Sum;
use datafusion::logical_expr::{
    Aggregate, Between, BinaryExpr, Expr, Join, JoinType, LogicalPlan, Operator, Projection,
};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use einfold_ir::{
    Aggregate as FoldAggregate, Binary, ColumnIndex, Comparison, Condition, Dimension, DimensionId,
    Domain, DomainKey, Expression, Fold, Key, KeyEquality, Operand, Output, RowValue, Unary,
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
/// doesn't apply. Where memory runs short, they all run the host's plan.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Fallback {
    /// The plan each node replaced, so a detected fold is never slower than
    /// the host's own plan.
    HostPlan,
    /// EinFold's discrimination algorithm, for any keys and density. The
    /// default: on the folds the dense algorithm can't take, such as sums
    /// over sparse weight tables, graph edges and embedding lookups, it
    /// beats the host's plan.
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
/// `agg`'s schema. A fold whose join has a condition, a mask, always keeps
/// the plan it replaced: the dense algorithm computes masks between outputs,
/// and the discrimination algorithm none.
fn fold(agg: &Aggregate, fallback: Fallback) -> Result<Option<LogicalPlan>> {
    let Some(Found {
        fold: spec,
        operands: plans,
    }) = analyze(agg, false)?
    else {
        return Ok(None);
    };
    let masked = !spec.conditions().is_empty();
    let original = LogicalPlan::Aggregate(agg.clone());
    let names: Vec<String> = (0..agg.group_expr.len())
        .map(|i| format!("{PREFIX}o{i}"))
        .chain([format!("{PREFIX}v")])
        .collect();
    let Ok(node) = FoldNode::try_new(spec, plans, None, names.clone()) else {
        return Ok(None);
    };
    let node = match (fallback, masked) {
        (Fallback::HostPlan, _) | (_, true) => node.with_fallback(original),
        (Fallback::Kernels, false) => node.with_memory_fallback(original),
    }
    .into_plan();
    // Give the node the aggregate's schema: its names, qualifiers, types
    // and nullability.
    restore(node, &names, &agg.schema)
}

/// `agg` as a fold, if it is one: a `SUM` of a product of one factor from
/// each side of an inner equi-join, grouped by columns, through projections;
/// or, with `expressions`, a `SUM` of a row expression of both sides (see
/// [`fold_over`]). A condition of the join that compares two of the fold's
/// dimensions, such as `q.t >= k.t`, becomes one of the fold's conditions;
/// any other declines.
pub(crate) fn analyze(agg: &Aggregate, expressions: bool) -> Result<Option<Found>> {
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
    if join.on.is_empty() {
        return Ok(None);
    }
    let groups = (agg.group_expr.iter())
        .map(|g| resolve(g, &projections))
        .collect::<Result<Vec<_>>>()?;
    fold_over(join, &groups, &resolve(&arg, &projections)?, expressions)
}

/// The `SUM` of `value` over `join`, an inner join, grouped by `groups`, as
/// a fold, with `groups` and `value` expressions of the join's columns.
/// `value` is a product of one factor from each side, or, with
/// `expressions`, a row expression (see [`row_expression`]). Over a join with
/// no equalities, such as a `CROSS JOIN`, and a group key from each side,
/// each pair of rows is its own group, and its one row is its sum.
pub(crate) fn fold_over(
    join: &Join,
    groups: &[Expr],
    value: &Expr,
    expressions: bool,
) -> Result<Option<Found>> {
    use datafusion::logical_expr::ExprSchemable;

    if join.join_type != JoinType::Inner {
        return Ok(None);
    }
    let inputs = [join.left.schema().as_ref(), join.right.schema().as_ref()];
    // A product's factors, one from each side.
    let factor = |f: &Expr| -> Result<Option<usize>> {
        let Some(s) = side(f, inputs) else {
            return Ok(None);
        };
        let ok = f.get_type(inputs[s])? == DataType::Float64 && movable(f, inputs[s]);
        Ok(ok.then_some(s))
    };
    let mut factors: [Option<Expr>; 2] = [None, None];
    if let Expr::BinaryExpr(BinaryExpr {
        left,
        op: Operator::Multiply,
        right,
    }) = value
    {
        for f in [left.as_ref(), right.as_ref()] {
            if let Some(s) = factor(f)? {
                if factors[s].replace(f.clone()).is_some() {
                    factors = [None, None];
                    break;
                }
            }
        }
    }
    let product = factors[0].is_some() && factors[1].is_some();
    if !product && !expressions {
        return Ok(None);
    }

    // Dimensions: one per pair of join keys, then one per group key that
    // isn't a join key. Each operand projects its keys, then its factor or
    // the columns a row expression reads.
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
    for g in groups {
        let (Expr::Column(_), Some(s)) = (g, side(g, inputs)) else {
            return Ok(None);
        };
        let d = match join.on.iter().position(|pair| [&pair.0, &pair.1][s] == g) {
            Some(d) => d,
            None => {
                dimensions.push(Dimension { equality });
                if !hold(s, dimensions.len() - 1, g.clone(), &mut columns) {
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
            let Some(found) = comparisons(c, &columns, &keys, inputs) else {
                return Ok(None);
            };
            conditions.extend(found);
        }
    }
    // What each side projects after its keys, and the row value.
    let (row, values): (RowValue, [Vec<Expr>; 2]) = match factors {
        [Some(lf), Some(rf)] => (RowValue::Product, [vec![lf], vec![rf]]),
        _ => {
            let mut values = [Vec::new(), Vec::new()];
            let base = [columns[0].len(), columns[1].len()];
            match row_expression(value, inputs, base, &mut values)? {
                Some(e) => (RowValue::Expression(e), values),
                None => return Ok(None),
            }
        }
    };
    let mut operands = Vec::new();
    for (s, (input, values)) in [&join.left, &join.right]
        .into_iter()
        .zip(values)
        .enumerate()
    {
        let mut exprs: Vec<Expr> = columns[s]
            .iter()
            .enumerate()
            .map(|(c, e)| e.clone().alias(format!("k{c}")))
            .collect();
        let factor = (row == RowValue::Product).then_some(ColumnIndex(exprs.len()));
        exprs.extend((values.into_iter().enumerate()).map(|(i, e)| e.alias(format!("f{i}"))));
        operands.push((
            Operand {
                keys: std::mem::take(&mut keys[s]),
                factor,
            },
            LogicalPlan::Projection(Projection::try_new(exprs, Arc::clone(input))?),
        ));
    }
    let operand_specs = operands.iter().map(|(o, _)| o.clone()).collect();
    let Ok(spec) = Fold::try_new(
        dimensions,
        operand_specs,
        outputs,
        row,
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

/// `e`, a `DOUBLE` expression of both sides of a join, as a row expression.
/// Each largest part of `e` that reads one side only and may be computed
/// on every row of it (see [`movable`]) becomes a column of that side's
/// operand, appended to `values` after its `base` key columns. The rest must
/// be operators of the closed set (see [`closed_set`]).
fn row_expression(
    e: &Expr,
    inputs: [&DFSchema; 2],
    base: [usize; 2],
    values: &mut [Vec<Expr>; 2],
) -> Result<Option<Expression>> {
    use datafusion::logical_expr::ExprSchemable;

    let mut leaf = |e: &Expr| -> Result<Option<Option<Expression>>> {
        let Some(s) = side(e, inputs) else {
            return Ok(None);
        };
        if e.get_type(inputs[s])? != DataType::Float64 || !movable(e, inputs[s]) {
            return Ok(Some(None));
        }
        let i = match values[s].iter().position(|v| v == e) {
            Some(i) => i,
            None => {
                values[s].push(e.clone());
                values[s].len() - 1
            }
        };
        Ok(Some(Some(Expression::column(s, base[s] + i))))
    };
    closed_set(e, &mut leaf)
}

/// `e` as a row expression, with `leaf` deciding the expressions it treats
/// as columns: `Some(Some(column))` for one, `Some(None)` for one it can't
/// take, `None` for an expression to look into. The rest must be operators of
/// the closed set: `+ - * /`, negation, `abs`, `exp`, `ln`, and `power` with
/// a positive constant exponent, on `DOUBLE`s, and `DOUBLE` literals. None of
/// these raises an error in DataFusion on `DOUBLE`s, so computing them on
/// the joined pairs gives what the host gives: zero to a negative power,
/// which raises one, can't occur.
pub(crate) fn closed_set(
    e: &Expr,
    leaf: &mut impl FnMut(&Expr) -> Result<Option<Option<Expression>>>,
) -> Result<Option<Expression>> {
    let e = e.clone().unalias();
    if let Some(column) = leaf(&e)? {
        return Ok(column);
    }
    let mut part = |e: &Expr| closed_set(e, leaf);
    Ok(match &e {
        Expr::Literal(ScalarValue::Float64(Some(x)), _) => Some(Expression::literal(*x)),
        Expr::Negative(x) => part(x)?.map(|x| x.unary(Unary::Negate)),
        Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
            let op = match op {
                Operator::Plus => Binary::Add,
                Operator::Minus => Binary::Subtract,
                Operator::Multiply => Binary::Multiply,
                Operator::Divide => Binary::Divide,
                _ => return Ok(None),
            };
            match (part(left)?, part(right)?) {
                (Some(a), Some(b)) => Some(a.binary(op, b)),
                _ => None,
            }
        }
        Expr::ScalarFunction(f) => match (f.name(), f.args.as_slice()) {
            ("abs", [x]) => part(x)?.map(|x| x.unary(Unary::Abs)),
            ("exp", [x]) => part(x)?.map(|x| x.unary(Unary::Exp)),
            ("ln", [x]) => part(x)?.map(|x| x.unary(Unary::Ln)),
            ("power", [x, Expr::Literal(ScalarValue::Float64(Some(p)), _)]) if *p > 0.0 => {
                part(x)?.map(|x| x.binary(Binary::Power, Expression::literal(*p)))
            }
            _ => None,
        },
        _ => None,
    })
}

/// The largest offset a condition may add to a key, `2³²`. Keys the kernels
/// take are non-negative `Int64` positions in arrays that fit in memory, far
/// below `2⁶²`, so SQL's arithmetic on them with such offsets can't overflow,
/// and a condition computes exactly what SQL's comparison does.
const MAX_OFFSET: i128 = 1 << 32;

/// `e` as conditions between two of a fold's dimensions, each `l <op> r +
/// offset` with `l` and `r` columns that an operand holds as keys (`columns`
/// and `keys`, per side of the join), on different sides. `e` is a
/// comparison whose sides add and subtract keys and integer literals, such
/// as `q.t - k.t < 32` or `q.t < k.t + 32`; `abs(q.t - k.t) < 32`, which is
/// two of them; or `x BETWEEN lo AND hi`, which is `x >= lo AND x <= hi`. A
/// comparison with arithmetic needs `Int64` keys (see [`MAX_OFFSET`]).
fn comparisons(
    e: &Expr,
    columns: &[Vec<Expr>; 2],
    keys: &[Vec<Key>; 2],
    inputs: [&DFSchema; 2],
) -> Option<Vec<Condition>> {
    if let Expr::Between(Between {
        expr,
        negated: false,
        low,
        high,
    }) = e
    {
        let x = expr.as_ref().clone();
        let mut found = comparisons(
            &x.clone().gt_eq(low.as_ref().clone()),
            columns,
            keys,
            inputs,
        )?;
        found.extend(comparisons(
            &x.lt_eq(high.as_ref().clone()),
            columns,
            keys,
            inputs,
        )?);
        return Some(found);
    }
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
    // `abs(x) < w` is `x < w AND x > 0 - w`, and `abs(x) <= w` the same with
    // `<=` and `>=`: a band either side of the diagonal.
    let abs = |e: &Expr| match e {
        Expr::ScalarFunction(f) if f.name() == "abs" && f.args.len() == 1 => {
            Some(f.args[0].clone())
        }
        _ => None,
    };
    let band = match (abs(left), comparison, abs(right)) {
        (Some(x), Comparison::Less | Comparison::LessOrEqual, None) => Some((x, *op, right)),
        (None, Comparison::Greater | Comparison::GreaterOrEqual, Some(x)) => {
            Some((x, op.swap()?, left))
        }
        (None, _, None) => None,
        _ => return None,
    };
    if let Some((x, op, w)) = band {
        let w = w.as_ref().clone();
        let below = Expr::BinaryExpr(BinaryExpr::new(
            Box::new(x.clone()),
            op,
            Box::new(w.clone()),
        ));
        let negated = Expr::BinaryExpr(BinaryExpr::new(
            Box::new(Expr::Literal(ScalarValue::Int64(Some(0)), None)),
            Operator::Minus,
            Box::new(w),
        ));
        let above = Expr::BinaryExpr(BinaryExpr::new(Box::new(x), op.swap()?, Box::new(negated)));
        let mut found = comparisons(&below, columns, keys, inputs)?;
        found.extend(comparisons(&above, columns, keys, inputs)?);
        return Some(found);
    }
    let dimension = |e: &Expr| -> Option<(usize, DimensionId)> {
        let s = side(e, inputs)?;
        let c = columns[s].iter().position(|k| k == e)?;
        let key = keys[s].iter().find(|k| k.column == ColumnIndex(c))?;
        Some((s, key.dimension))
    };
    use datafusion::logical_expr::ExprSchemable;

    // `left - right <op> 0`, as signed terms and a constant.
    let mut sum = Linear::default();
    sum.add(left, 1, &dimension)?;
    sum.add(right, -1, &dimension)?;
    let Linear {
        terms,
        constant,
        arithmetic,
    } = sum;
    // `a - b + constant <op> 0` is `a <op> b - constant`.
    let (plus, minus) = match &terms[..] {
        [a, b] if (a.sign, b.sign) == (1, -1) => (a, b),
        [b, a] if (a.sign, b.sign) == (1, -1) => (a, b),
        _ => return None,
    };
    let int64 = |t: &Term| t.expression.get_type(inputs[t.side]).ok() == Some(DataType::Int64);
    let crosses = plus.side != minus.side && plus.dimension != minus.dimension;
    if !crosses || (arithmetic && !(int64(plus) && int64(minus))) {
        return None;
    }
    Some(vec![Condition {
        left: plus.dimension,
        comparison,
        right: minus.dimension,
        offset: i64::try_from(-constant).ok()?,
    }])
}

/// A key in a [`Linear`] sum.
struct Term<'a> {
    /// `1` or `-1`.
    sign: i8,
    /// Which side of the join the key is on.
    side: usize,
    dimension: DimensionId,
    expression: &'a Expr,
}

/// An integer expression of keys as a sum: signed key terms, plus a
/// constant; and whether it did any arithmetic.
#[derive(Default)]
struct Linear<'a> {
    terms: Vec<Term<'a>>,
    constant: i128,
    arithmetic: bool,
}

impl<'a> Linear<'a> {
    /// Adds `sign · e`, if `e` is a key, an integer literal of at most
    /// [`MAX_OFFSET`], or a sum or difference of them.
    fn add(
        &mut self,
        e: &'a Expr,
        sign: i8,
        dimension: &impl Fn(&Expr) -> Option<(usize, DimensionId)>,
    ) -> Option<()> {
        match e {
            Expr::BinaryExpr(BinaryExpr { left, op, right })
                if matches!(op, Operator::Plus | Operator::Minus) =>
            {
                self.arithmetic = true;
                self.add(left, sign, dimension)?;
                let sign = if *op == Operator::Plus { sign } else { -sign };
                self.add(right, sign, dimension)
            }
            Expr::Literal(value, _) => {
                let v = integer(value)?;
                if v.abs() > MAX_OFFSET {
                    return None;
                }
                self.arithmetic = true;
                self.constant += i128::from(sign) * v;
                Some(())
            }
            e => {
                let (side, dimension) = dimension(e)?;
                self.terms.push(Term {
                    sign,
                    side,
                    dimension,
                    expression: e,
                });
                Some(())
            }
        }
    }
}

/// A non-NULL integer literal's value.
fn integer(value: &ScalarValue) -> Option<i128> {
    Some(match value {
        ScalarValue::Int8(Some(v)) => i128::from(*v),
        ScalarValue::Int16(Some(v)) => i128::from(*v),
        ScalarValue::Int32(Some(v)) => i128::from(*v),
        ScalarValue::Int64(Some(v)) => i128::from(*v),
        ScalarValue::UInt8(Some(v)) => i128::from(*v),
        ScalarValue::UInt16(Some(v)) => i128::from(*v),
        ScalarValue::UInt32(Some(v)) => i128::from(*v),
        ScalarValue::UInt64(Some(v)) => i128::from(*v),
        _ => return None,
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

/// The value `exprs` give a fold's sum `value` where it is NULL, if each of
/// their reads of `value` is a *fill* that gives one `DOUBLE` constant `c`
/// there and `value` everywhere else:
///
/// - `COALESCE(value, c)`;
/// - `CASE … WHEN value IS NULL THEN c ELSE value END`, or `CASE … WHEN value
///   IS NOT NULL THEN value ELSE c END`, as DataFusion writes `COALESCE`,
///   where the test of `value` is the last and no earlier branch reads it.
///
/// ddx writes the first form of `CASE` for each gradient, to give every row
/// of a table its gradient, 0 where none reached it. A fold from a domain can
/// then compute `c` where no row reached a group, or where every product of
/// one was NULL, rather than NULL (the domain's `missing`): each fill gives
/// `c` for `c` as it did for NULL, and the dense algorithm can take operands
/// with missing rows. `None` if any read of `value` is anything else, or if
/// two fills differ.
pub(crate) fn fill_default(exprs: &[Expr], value: &Column) -> Option<f64> {
    use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion::functions::core::coalesce::CoalesceFunc;
    use datafusion::logical_expr::expr::{Case, ScalarFunction};
    let is_value = |e: &Expr| matches!(e, Expr::Column(c) if c == value);
    let constant = |e: &Expr| match e {
        Expr::Literal(ScalarValue::Float64(Some(c)), _) => Some(*c),
        _ => None,
    };
    let reads = |e: &Expr| e.column_refs().contains(value);
    // The constant `e` gives where `value` is NULL, if `e` is a fill.
    let fill = |e: &Expr| -> Option<f64> {
        match e {
            // DataFusion's own `COALESCE`, by type, not by name: a session
            // may register another function under the name.
            Expr::ScalarFunction(ScalarFunction { func, args })
                if (func.inner().as_ref() as &dyn Any).is::<CoalesceFunc>()
                    && args.len() == 2
                    && is_value(&args[0]) =>
            {
                constant(&args[1])
            }
            Expr::Case(Case {
                expr: None,
                when_then_expr,
                else_expr: Some(otherwise),
            }) => {
                let (last, earlier) = when_then_expr.split_last()?;
                if earlier.iter().any(|(w, t)| reads(w) || reads(t)) {
                    return None;
                }
                match (last.0.as_ref(), last.1.as_ref(), otherwise.as_ref()) {
                    (Expr::IsNull(v), then, other) if is_value(v) && is_value(other) => {
                        constant(then)
                    }
                    (Expr::IsNotNull(v), then, other) if is_value(v) && is_value(then) => {
                        constant(other)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    };
    let mut found: Option<f64> = None;
    let mut ok = true;
    for e in exprs {
        let visited = e.apply(|x| {
            if let Some(c) = fill(x) {
                // Fills of one value, to its last bit: `-0.0` isn't `0.0`.
                ok &= found.is_none_or(|f| f.to_bits() == c.to_bits());
                found = Some(c);
                return Ok(TreeNodeRecursion::Jump);
            }
            ok &= !is_value(x);
            Ok(if ok {
                TreeNodeRecursion::Continue
            } else {
                TreeNodeRecursion::Stop
            })
        });
        if visited.is_err() || !ok {
            return None;
        }
    }
    found
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
        // Where `p` fills a missing sum, the node computes the fill.
        missing: fill_default(&p.expr, &value),
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
    let with_domain = Fold::try_new(
        fold.dimensions().to_vec(),
        fold.operands().to_vec(),
        outputs,
        fold.row().clone(),
        fold.aggregate().clone(),
        Some(spec),
    )
    // The inner fold's conditions, such as a causal mask, hold here too.
    .and_then(|f| f.with_conditions(fold.conditions().to_vec()));
    let Ok(with_domain) = with_domain else {
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
    let Ok(new) = new.with_nullable_value() else {
        return Ok(None);
    };
    // A mask needs the host's plan where the dense algorithm declines, as
    // the inner fold did.
    let masked = !fold.conditions().is_empty();
    let new = match (fallback, masked) {
        (Fallback::HostPlan, _) | (_, true) => new.with_fallback(original.clone()),
        (Fallback::Kernels, false) => new.with_memory_fallback(original.clone()),
    }
    .into_plan();
    // Rename the node's columns to the join's, so `p` reads them as before.
    let Some(renamed) = restore(new, &names, original.schema())? else {
        return Ok(None);
    };
    let projection = Projection::try_new(p.expr.clone(), Arc::new(renamed))?;
    Ok(same_schema(&projection.schema, &p.schema).then_some(LogicalPlan::Projection(projection)))
}
