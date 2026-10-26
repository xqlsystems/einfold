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
//! types and nullability. Each node also keeps the plan it replaced, which
//! [`EinFoldExec`](crate::EinFoldExec) runs when its dense algorithm doesn't
//! apply, so a detected fold is never slower than the host's own plan.

use std::sync::Arc;

use datafusion::arrow::datatypes::DataType;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, DFSchemaRef, NullEquality, Result};
use datafusion::functions_aggregate::sum::Sum;
use datafusion::logical_expr::{
    Aggregate, BinaryExpr, Expr, ExprSchemable, Join, JoinType, LogicalPlan, Operator, Projection,
};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use einfold_ir::{
    Aggregate as FoldAggregate, ColumnIndex, Dimension, DimensionId, Domain, Fold, KeyEquality,
    Operand, Output, RowValue,
};

use crate::node::FoldNode;

/// The prefix of the names this rule gives the columns of the nodes it makes,
/// chosen not to clash with a domain's own columns.
const PREFIX: &str = "__einfold_";

/// Finds folds over joins in DataFusion's optimized plans (see the module's
/// documentation). [`EinFoldQueryPlanner::detecting`](crate::EinFoldQueryPlanner::detecting)
/// applies it to each plan just before planning it, which [`install`](crate::install)
/// sets up. It can also be added after DataFusion's own optimizer rules, but
/// then optimized plans contain `Fold` nodes, which a Substrait producer, such
/// as ddx's, can't encode.
#[derive(Debug)]
pub struct DetectFolds {
    fallback: bool,
}

impl Default for DetectFolds {
    fn default() -> Self {
        DetectFolds { fallback: true }
    }
}

impl DetectFolds {
    /// The rule, but with nodes that run EinFold's hash algorithm where the
    /// dense one doesn't apply, rather than the host's plan. Slower, and for
    /// tests: it makes every fold the rule describes run through einfold.
    pub fn without_fallback() -> Self {
        DetectFolds { fallback: false }
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
            LogicalPlan::Projection(p) => from_domain(p, self.fallback)?,
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
fn resolve(e: &Expr, projections: &[&Projection]) -> Result<Expr> {
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
fn side(e: &Expr, inputs: [&DFSchemaRef; 2]) -> Option<usize> {
    let cols = e.column_refs();
    if cols.is_empty() {
        return None;
    }
    (0..2).find(|&s| cols.iter().all(|c| inputs[s].has_column(c)))
}

/// The sum's argument, if `agg` computes exactly one plain `SUM`.
fn sum_argument(agg: &Aggregate) -> Option<Expr> {
    let [aggregate] = agg.aggr_expr.as_slice() else {
        return None;
    };
    let Expr::AggregateFunction(f) = aggregate.clone().unalias() else {
        return None;
    };
    let p = &f.params;
    let plain = !p.distinct && p.filter.is_none() && p.order_by.is_empty() && p.args.len() == 1;
    // By type, not by name, as DataFusion asks of rules: a session may
    // register another function under the name `sum`.
    let function: &dyn std::any::Any = f.func.inner().as_ref();
    (function.is::<Sum>() && plain).then(|| p.args[0].clone())
}

/// `agg` as a fold, if it is one: a [`FoldNode`] under a projection giving it
/// `agg`'s schema.
fn fold(agg: &Aggregate, fallback: bool) -> Result<Option<LogicalPlan>> {
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
    if join.join_type != JoinType::Inner || join.filter.is_some() || join.on.is_empty() {
        return Ok(None);
    }
    let inputs = [join.left.schema(), join.right.schema()];
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
        if factors[s].is_some() || f.get_type(inputs[s].as_ref())? != DataType::Float64 {
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
    let mut keys: [Vec<(DimensionId, ColumnIndex)>; 2] = [Vec::new(), Vec::new()];
    let mut hold = |s: usize, d: usize, e: Expr, columns: &mut [Vec<Expr>; 2]| {
        if columns[s].contains(&e) {
            return false;
        }
        keys[s].push((DimensionId(d), ColumnIndex(columns[s].len())));
        columns[s].push(e);
        true
    };
    for (l, r) in &join.on {
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
    let Ok(spec) = Fold::try_new(
        dimensions,
        operands.iter().map(|(o, _)| o.clone()).collect(),
        outputs,
        RowValue::Product,
        FoldAggregate::sum(),
        None,
    ) else {
        return Ok(None);
    };
    let plans = operands.into_iter().map(|(_, p)| p).collect();
    let original = LogicalPlan::Aggregate(agg.clone());
    let names: Vec<String> = (0..agg.group_expr.len())
        .map(|i| format!("{PREFIX}o{i}"))
        .chain([format!("{PREFIX}v")])
        .collect();
    let Ok(node) = FoldNode::try_new(spec, plans, None, names.clone()) else {
        return Ok(None);
    };
    let node = match fallback {
        true => node.with_fallback(original),
        false => node.marked_found(),
    }
    .into_plan();
    // Give the node the aggregate's schema: its names, qualifiers, types
    // and nullability.
    restore(node, &names, &agg.schema)
}

/// `node`, whose columns are `names`, renamed to `schema`'s columns, or
/// `None` if the result's schema wouldn't be exactly `schema`.
fn restore(
    node: LogicalPlan,
    names: &[String],
    schema: &DFSchemaRef,
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
fn same_schema(a: &DFSchemaRef, b: &DFSchemaRef) -> bool {
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
fn found_fold(plan: &LogicalPlan) -> Option<(&FoldNode, Vec<usize>)> {
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
fn from_domain(p: &Projection, fallback: bool) -> Result<Option<LogicalPlan>> {
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
    let mut keys: Vec<Option<(ColumnIndex, KeyEquality)>> = vec![None; n];
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
        if i >= n || keys[i].replace((ColumnIndex(d), equality)).is_some() {
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
    let original = LogicalPlan::Projection(Projection::try_new(kept, Arc::clone(&p.input))?);
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
        true => new.with_fallback(original.clone()),
        false => new.marked_found(),
    }
    .into_plan();
    // Rename the node's columns to the join's, so `p` reads them as before.
    let Some(renamed) = restore(new, &names, original.schema())? else {
        return Ok(None);
    };
    let projection = Projection::try_new(p.expr.clone(), Arc::new(renamed))?;
    Ok(same_schema(&projection.schema, &p.schema).then_some(LogicalPlan::Projection(projection)))
}
