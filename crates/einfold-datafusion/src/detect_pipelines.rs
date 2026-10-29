// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The fourth shape [`DetectFolds`](crate::DetectFolds) finds: a fold
//! pipeline, a softmax-weighted average of a fold's results (see
//! [`einfold_ir::pipeline`]). Causal attention, written in SQL, is one:
//!
//! ```sql
//! WITH scores AS (SELECT q.t AS i, k.t AS j, SUM(q.x * k.x) / 8.0 AS s
//!                 FROM q JOIN k ON q.d = k.d WHERE q.t >= k.t GROUP BY q.t, k.t),
//! weights AS (SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS w FROM scores),
//! norm AS (SELECT i, SUM(w) AS z FROM weights GROUP BY i)
//! SELECT weights.i, v.d, SUM(weights.w * v.x) / norm.z
//! FROM weights JOIN v ON weights.j = v.t JOIN norm ON weights.i = norm.i
//! GROUP BY weights.i, v.d, norm.z
//! ```
//!
//! It is found in DataFusion's optimized plan, where each CTE is copied into
//! every place that reads it, through these parts:
//!
//! - **the scores:** a fold, as the rule's first shape, whose join may have
//!   conditions comparing its group keys (the causal mask), under a
//!   projection giving its group keys and its sum, scaled by a constant
//!   (`* c` or `/ c`) or not. The sum may be of a row expression rather than
//!   a product, such as `SUM(power(t.x - s.x, 2))` over a coordinate `d`;
//!   and over a join with no equalities, such as a `CROSS JOIN` of points,
//!   there may be no sum at all, each pair's one row its score (see
//!   [`pair_weights`]);
//! - **the weights:** the scores' group keys and `exp(s - m)`, with `m` the
//!   largest score of the same kept key, as a window (`MAX(s) OVER
//!   (PARTITION BY i)`, over the whole partition) or as `MAX(s) … GROUP BY i`
//!   joined back on `i`; or `exp(s)`, with no shift;
//! - **the average:** an aggregate grouped by the weights' kept key and the
//!   values' other columns, of `SUM(w * v.x)`, over the weights joined with a
//!   values relation on the summed key. It is divided by `norm.z`, a
//!   `SUM(w) … GROUP BY i` of the same weights joined on `i`: either the sum
//!   (`SUM(w * v.x) / norm.z`, with `norm.z` grouped by too), or each term
//!   (`SUM(w / norm.z * v.x)`, the spelling that ddx, an XQL Systems project
//!   for automatic differentiation of SQL queries, can differentiate, since
//!   it groups by no key that depends on a differentiated column). Or it is
//!   divided by a second aggregate of the same `GROUP BY`, `SUM(w)`.
//!
//! Every copy of the scores and weights must be the same fold with the same
//! scale and shift; otherwise the shift wouldn't cancel, and the rule
//! declines. The node keeps the plan it replaced, which runs wherever the
//! fused algorithm declines.

use std::any::Any;
use std::sync::Arc;

use datafusion::common::{Column, NullEquality, Result, ScalarValue};
use datafusion::functions_aggregate::min_max::Max;
use datafusion::logical_expr::expr::{AggregateFunction, WindowFunction};
use datafusion::logical_expr::{
    BinaryExpr, Expr, JoinType, LogicalPlan, Operator, Projection, WindowFunctionDefinition,
};
use einfold_ir::{
    ColumnIndex, Dimension, DimensionId, Key, KeyEquality, Normalizer, Operand, Pipeline, Scale,
    Shift,
};

use crate::detect_folds::{self, Found, PREFIX};
use crate::pipeline_node::PipelineNode;

/// `plan` without the aliases over it.
fn strip(plan: &LogicalPlan) -> &LogicalPlan {
    match plan {
        LogicalPlan::SubqueryAlias(a) => strip(&a.input),
        other => other,
    }
}

/// `e`, an expression over `plan`'s columns, as an expression over the
/// columns of the relations under `plan`'s projections and inner joins: its
/// leaves.
fn trace(plan: &LogicalPlan, e: &Expr) -> Result<Expr> {
    use datafusion::common::tree_node::{Transformed, TreeNode};
    let e = e.clone().unalias();
    match plan {
        LogicalPlan::Projection(p) => {
            let e = detect_folds::resolve(&e, &[p])?;
            trace(&p.input, &e)
        }
        LogicalPlan::Join(j) if j.join_type == JoinType::Inner => Ok(e
            .transform(|x| match &x {
                Expr::Column(c) if j.left.schema().has_column(c) => {
                    trace(&j.left, &x).map(Transformed::yes)
                }
                Expr::Column(c) if j.right.schema().has_column(c) => {
                    trace(&j.right, &x).map(Transformed::yes)
                }
                _ => Ok(Transformed::no(x)),
            })?
            .data),
        _ => Ok(e),
    }
}

/// The leaves under `plan`'s projections and inner joins with no condition
/// but equalities, and those joins' equalities, traced to the leaves.
fn leaves<'a>(
    plan: &'a LogicalPlan,
    out: &mut Vec<&'a LogicalPlan>,
    on: &mut Vec<(Expr, Expr)>,
) -> Result<bool> {
    match plan {
        LogicalPlan::Projection(p) => leaves(&p.input, out, on),
        LogicalPlan::Join(j) => {
            let plain = j.join_type == JoinType::Inner
                && j.filter.is_none()
                && j.null_equality == NullEquality::NullEqualsNothing;
            if !plain {
                return Ok(false);
            }
            for (l, r) in &j.on {
                on.push((trace(&j.left, l)?, trace(&j.right, r)?));
            }
            Ok(leaves(&j.left, out, on)? && leaves(&j.right, out, on)?)
        }
        leaf => {
            out.push(leaf);
            Ok(true)
        }
    }
}

/// Which of `leaves` every column of `e` belongs to.
fn leaf_of(e: &Expr, leaves: &[&LogicalPlan]) -> Option<usize> {
    let cols = e.column_refs();
    let first = cols.iter().next()?;
    let l = leaves.iter().position(|p| p.schema().has_column(first))?;
    cols.iter()
        .all(|c| leaves[l].schema().has_column(c))
        .then_some(l)
}

/// The column `e` is, if it is one.
fn column(e: &Expr) -> Option<&Column> {
    match e {
        Expr::Column(c) => Some(c),
        _ => None,
    }
}

/// `plan`'s column at `index`.
fn column_at(plan: &LogicalPlan, index: usize) -> Column {
    Column::from(plan.schema().qualified_field(index))
}

/// Whether `function` is DataFusion's `MAX`, by type, as DataFusion asks of
/// rules.
fn is_max(function: &dyn Any) -> bool {
    function.is::<Max>()
}

/// A fold's scores: the fold, its scale, and, for each column of the
/// relation, which of the fold's outputs it is or `None` for the score.
struct Scores {
    found: Found,
    scale: Scale,
    columns: Vec<Option<usize>>,
}

/// `plan` as a fold's scores: a projection of the fold's group keys and its
/// scaled sum over the fold's aggregate.
fn scores(plan: &LogicalPlan) -> Result<Option<Scores>> {
    let LogicalPlan::Projection(p) = strip(plan) else {
        return Ok(None);
    };
    let LogicalPlan::Aggregate(agg) = p.input.as_ref() else {
        return Ok(None);
    };
    let Some(found) = detect_folds::analyze(agg, true)? else {
        return Ok(None);
    };
    let sum = column_at(&p.input, agg.group_expr.len());
    let group = |c: &Column| (0..agg.group_expr.len()).find(|&g| column_at(&p.input, g) == *c);
    let mut scale = None;
    let mut columns = Vec::new();
    for e in &p.expr {
        match e.clone().unalias() {
            // One score column, scaled or not: a second would leave which
            // is the score unclear.
            Expr::Column(c) if c == sum => {
                if scale.is_some() {
                    return Ok(None);
                }
                scale = Some(Scale::One);
                columns.push(None);
            }
            Expr::Column(c) => match group(&c) {
                Some(g) => columns.push(Some(g)),
                None => return Ok(None),
            },
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
                let (Expr::Column(c), Expr::Literal(ScalarValue::Float64(Some(k)), _)) =
                    (left.as_ref(), right.as_ref())
                else {
                    return Ok(None);
                };
                let s = match op {
                    Operator::Divide => Scale::Over(*k),
                    Operator::Multiply => Scale::Times(*k),
                    _ => return Ok(None),
                };
                if *c != sum || scale.is_some() {
                    return Ok(None);
                }
                scale = Some(s);
                columns.push(None);
            }
            _ => return Ok(None),
        }
    }
    Ok(scale.map(|scale| Scores {
        found,
        scale,
        columns,
    }))
}

/// Whether two scores are the same fold, scaled the same way.
fn same_scores(a: &Scores, b: &Scores) -> bool {
    a.found.fold == b.found.fold && a.found.operands == b.found.operands && a.scale == b.scale
}

/// A relation of weights: its scores, its shift, the fold output its
/// maximum is taken per (with [`Shift::KeptMaximum`]), and, for each of its
/// columns, the fold output it is, or `None` for the weight.
struct Weights {
    scores: Scores,
    shift: Shift,
    maximum_per: Option<usize>,
    columns: Vec<Option<usize>>,
}

/// The score column of `scores`, read through `plan`'s schema.
fn score_column(scores: &Scores, plan: &LogicalPlan) -> Option<Column> {
    let s = scores.columns.iter().position(Option::is_none)?;
    Some(column_at(plan, s))
}

/// `plan` as a relation of weights over scores (see the module's
/// documentation).
fn weights(plan: &LogicalPlan) -> Result<Option<Weights>> {
    let LogicalPlan::Projection(p) = strip(plan) else {
        return Ok(None);
    };
    if let Some(weights) = pair_weights(p)? {
        return Ok(Some(weights));
    }
    if let Some(weights) = merged_weights(p)? {
        return Ok(Some(weights));
    }
    // The relation the weights are computed over, its scores, the score
    // column, and the shift's column if there is one.
    let below = p.input.as_ref();
    let (s_plan, shift_col, maximum_per): (&LogicalPlan, Option<Column>, Option<usize>) =
        match below {
            LogicalPlan::Window(w) => {
                let [Expr::WindowFunction(f)] = w.window_expr.as_slice() else {
                    return Ok(None);
                };
                let WindowFunction { fun, params } = f.as_ref();
                let WindowFunctionDefinition::AggregateUDF(udf) = fun else {
                    return Ok(None);
                };
                let whole = params.window_frame.start_bound.is_unbounded()
                    && params.window_frame.end_bound.is_unbounded();
                let plain = whole
                    && params.order_by.is_empty()
                    && params.filter.is_none()
                    && !params.distinct
                    && params.args.len() == 1;
                if !plain || !is_max(udf.inner().as_ref()) {
                    return Ok(None);
                }
                let Some(scores) = scores(&w.input)? else {
                    return Ok(None);
                };
                // MAX of the score, partitioned by one of the fold's outputs.
                let score = score_column(&scores, &w.input);
                if column(&params.args[0].clone().unalias()) != score.as_ref() {
                    return Ok(None);
                }
                let Some(per) = partition(&params.partition_by, &scores, &w.input) else {
                    return Ok(None);
                };
                let window = column_at(below, w.input.schema().fields().len());
                (w.input.as_ref(), Some(window), Some(per))
            }
            LogicalPlan::Join(j) => {
                // The scores joined to their maximum by the kept output.
                let (Some(sc), Some(maximum)) = (scores(&j.left)?, maximum(&j.right)?) else {
                    return Ok(None);
                };
                let kept = maximum.kept;
                // An `=` join: one that matched NULL kept keys would join a
                // NULL group's scores to its maximum, which SQL's `=` doesn't.
                if !same_scores(&sc, &maximum.scores)
                    || j.join_type != JoinType::Inner
                    || j.null_equality != NullEquality::NullEqualsNothing
                    || j.filter.is_some()
                {
                    return Ok(None);
                }
                let kept_col = sc
                    .columns
                    .iter()
                    .position(|&c| c == Some(kept))
                    .map(|i| column_at(&j.left, i));
                let [(l, r)] = j.on.as_slice() else {
                    return Ok(None);
                };
                if column(l) != kept_col.as_ref() || column(r) != Some(&maximum.key_column) {
                    return Ok(None);
                }
                (j.left.as_ref(), Some(maximum.maximum_column), Some(kept))
            }
            other => (other, None, None),
        };
    let Some(scores) = scores(s_plan)? else {
        return Ok(None);
    };
    let Some(score) = score_column(&scores, s_plan) else {
        return Ok(None);
    };
    let mut columns = Vec::new();
    let mut weight = false;
    for e in &p.expr {
        let e = detect_folds::resolve(e, &[])?;
        match &e {
            Expr::Column(c) => match s_plan
                .schema()
                .index_of_column(c)
                .ok()
                .and_then(|i| scores.columns.get(i).copied().flatten())
            {
                Some(o) => columns.push(Some(o)),
                None => return Ok(None),
            },
            Expr::ScalarFunction(f) if f.name() == "exp" && f.args.len() == 1 && !weight => {
                let arg = f.args[0].clone().unalias();
                let shifted = match (&arg, &shift_col) {
                    (Expr::Column(c), None) => *c == score,
                    (
                        Expr::BinaryExpr(BinaryExpr {
                            left,
                            op: Operator::Minus,
                            right,
                        }),
                        Some(m),
                    ) => column(left) == Some(&score) && column(right) == Some(m),
                    _ => false,
                };
                if !shifted {
                    return Ok(None);
                }
                weight = true;
                columns.push(None);
            }
            _ => return Ok(None),
        }
    }
    let shift = match shift_col {
        Some(_) => Shift::KeptMaximum,
        None => Shift::None,
    };
    Ok(weight.then_some(Weights {
        scores,
        shift,
        maximum_per,
        columns,
    }))
}

/// `p` as unshifted weights merged with their scores, as DataFusion merges
/// `SELECT i, j, exp(s) FROM (SELECT q.t AS i, k.t AS j, SUM(…) AS s …)`
/// into one projection of `exp(SUM(…))` over the aggregate: the scores are
/// `p` with the weight's argument in place of the weight.
fn merged_weights(p: &Projection) -> Result<Option<Weights>> {
    if !matches!(p.input.as_ref(), LogicalPlan::Aggregate(_)) {
        return Ok(None);
    }
    let exp = |e: &Expr| match e.clone().unalias() {
        Expr::ScalarFunction(f) if f.name() == "exp" && f.args.len() == 1 => {
            Some(f.args[0].clone())
        }
        _ => None,
    };
    let [w] = (0..p.expr.len())
        .filter(|&i| exp(&p.expr[i]).is_some())
        .collect::<Vec<_>>()[..]
    else {
        return Ok(None);
    };
    let mut exprs = p.expr.clone();
    exprs[w] = exp(&p.expr[w]).expect("an exp").alias("s");
    let plan = LogicalPlan::Projection(Projection::try_new(exprs, Arc::clone(&p.input))?);
    let Some(scores) = scores(&plan)? else {
        return Ok(None);
    };
    // The weight's argument is the scores' one column that isn't a key; a
    // projection that also shows the raw score has two, and the raw one is
    // no weight.
    if scores.columns.iter().filter(|c| c.is_none()).count() != 1 {
        return Ok(None);
    }
    let columns = scores.columns.clone();
    Ok(Some(Weights {
        scores,
        shift: Shift::None,
        maximum_per: None,
        columns,
    }))
}

/// `p` as weights computed on each pair of a join with no equalities, such
/// as a `CROSS JOIN` of target and source points: a key column from each
/// side, and `exp(s)` of a row expression `s` of both, as a Gaussian kernel's
/// weights are,
///
/// ```sql
/// SELECT t.p, s.q, exp(-(power(t.x - s.x, 2) + power(t.y - s.y, 2)) / 0.25)
/// FROM target t CROSS JOIN source s
/// ```
///
/// Their scores are a fold with nothing to sum: each pair of keys is a
/// group, and its one row's `s` its value. Two rows for one pair would be
/// two weights here, which the fused algorithm, placing each pair once,
/// declines.
fn pair_weights(p: &Projection) -> Result<Option<Weights>> {
    let LogicalPlan::Join(join) = p.input.as_ref() else {
        return Ok(None);
    };
    if !join.on.is_empty() {
        return Ok(None);
    }
    let mut groups = Vec::new();
    let mut columns = Vec::new();
    let mut score = None;
    for e in &p.expr {
        match e.clone().unalias() {
            Expr::Column(c) => {
                columns.push(Some(groups.len()));
                groups.push(Expr::Column(c));
            }
            Expr::ScalarFunction(f)
                if f.name() == "exp" && f.args.len() == 1 && score.is_none() =>
            {
                score = Some(f.args[0].clone());
                columns.push(None);
            }
            _ => return Ok(None),
        }
    }
    let Some(score) = score else {
        return Ok(None);
    };
    let Some(found) = detect_folds::fold_over(join, &groups, &score, true)? else {
        return Ok(None);
    };
    Ok(Some(Weights {
        scores: Scores {
            found,
            scale: Scale::One,
            columns: columns.clone(),
        },
        shift: Shift::None,
        maximum_per: None,
        columns,
    }))
}

/// The fold output `partition_by` partitions by, if it is exactly one of
/// the scores' outputs.
fn partition(partition_by: &[Expr], scores: &Scores, plan: &LogicalPlan) -> Option<usize> {
    let [e] = partition_by else {
        return None;
    };
    let c = column(e)?;
    let i = plan.schema().index_of_column(c).ok()?;
    scores.columns.get(i).copied().flatten()
}

/// The largest score of each kept output, `MAX(s) … GROUP BY i`.
struct Maximum {
    /// The scores it's the largest of.
    scores: Scores,
    /// The kept output it's grouped by.
    kept: usize,
    /// The plan's column holding the maximum.
    maximum_column: Column,
    /// The plan's column holding the kept output's key.
    key_column: Column,
}

/// `plan` as the largest score of each kept output, or `None`.
fn maximum(plan: &LogicalPlan) -> Result<Option<Maximum>> {
    let LogicalPlan::Projection(p) = strip(plan) else {
        return Ok(None);
    };
    let LogicalPlan::Aggregate(agg) = p.input.as_ref() else {
        return Ok(None);
    };
    let ([g], [a]) = (agg.group_expr.as_slice(), agg.aggr_expr.as_slice()) else {
        return Ok(None);
    };
    let Expr::AggregateFunction(AggregateFunction { func, params }) = a.clone().unalias() else {
        return Ok(None);
    };
    let plain = !params.distinct && params.filter.is_none() && params.args.len() == 1;
    if !plain || !is_max(func.inner().as_ref()) {
        return Ok(None);
    }
    let Some(scores) = scores(&agg.input)? else {
        return Ok(None);
    };
    let (Some(score), Some(group)) = (score_column(&scores, &agg.input), column(g)) else {
        return Ok(None);
    };
    if params.args[0].clone().unalias() != Expr::Column(score) {
        return Ok(None);
    }
    let Some(kept) =
        (agg.input.schema().index_of_column(group).ok()).and_then(|i| scores.columns[i])
    else {
        return Ok(None);
    };
    // The projection's columns: the key, then the maximum, in some order.
    let (key, max) = (column_at(&p.input, 0), column_at(&p.input, 1));
    let mut key_column = None;
    let mut maximum_column = None;
    for (i, e) in p.expr.iter().enumerate() {
        match e.clone().unalias() {
            Expr::Column(c) if c == key => key_column = Some(column_at(plan, i)),
            Expr::Column(c) if c == max => maximum_column = Some(column_at(plan, i)),
            _ => return Ok(None),
        }
    }
    let (Some(key_column), Some(maximum_column)) = (key_column, maximum_column) else {
        return Ok(None);
    };
    Ok(Some(Maximum {
        scores,
        kept,
        maximum_column,
        key_column,
    }))
}

/// `p` as a fold pipeline (see the module's documentation): a
/// [`PipelineNode`] under a projection giving it `p`'s schema.
pub(crate) fn pipeline(p: &Projection) -> Result<Option<LogicalPlan>> {
    let LogicalPlan::Aggregate(agg) = p.input.as_ref() else {
        return Ok(None);
    };
    let sums: Vec<Expr> = (0..agg.aggr_expr.len())
        .filter_map(|a| detect_folds::sum_argument_at(agg, a))
        .collect();
    if sums.len() != agg.aggr_expr.len() || !(1..=2).contains(&sums.len()) {
        return Ok(None);
    }
    let groups = agg.group_expr.len();
    let sum_col = |a: usize| column_at(&p.input, groups + a);
    // The projection: group keys, then the average last: `sum / norm`, or
    // the sum itself when the normalizer divides inside it.
    let Some((value, keys)) = p.expr.split_last() else {
        return Ok(None);
    };
    let group_of = |c: &Column| (0..groups).find(|&g| column_at(&p.input, g) == *c);
    let (normalizer, weighted, z_group, inside) = match value.clone().unalias() {
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Divide,
            right,
        }) => {
            let (Some(numerator), Some(denominator)) = (column(&left), column(&right)) else {
                return Ok(None);
            };
            match sums.len() {
                1 if *numerator == sum_col(0) => match group_of(denominator) {
                    Some(g) => (Normalizer::KeptGroups, 0, Some(g), false),
                    None => return Ok(None),
                },
                2 if *numerator == sum_col(0) && *denominator == sum_col(1) => {
                    (Normalizer::JoinedRows, 0, None, false)
                }
                2 if *numerator == sum_col(1) && *denominator == sum_col(0) => {
                    (Normalizer::JoinedRows, 1, None, false)
                }
                _ => return Ok(None),
            }
        }
        Expr::Column(c) if sums.len() == 1 && c == sum_col(0) => {
            (Normalizer::KeptGroups, 0, None, true)
        }
        _ => return Ok(None),
    };

    // The relations under the aggregate.
    let (mut relations, mut on) = (Vec::new(), Vec::new());
    if !leaves(&agg.input, &mut relations, &mut on)? {
        return Ok(None);
    }
    let traced = trace(&agg.input, &sums[weighted])?;
    // With the normalizer inside the sum, `w / z * x` or `w * x / z`: the
    // product of the weight and the value, and `z`.
    let (product, z_inside) = match inside {
        false => (traced, None),
        true => match divided(traced) {
            Some((product, z)) => (product, Some(z)),
            None => return Ok(None),
        },
    };
    let Expr::BinaryExpr(BinaryExpr {
        left,
        op: Operator::Multiply,
        right,
    }) = &product
    else {
        return Ok(None);
    };
    // The weight is one factor, the values' column the other.
    let mut found = None;
    for (w, x) in [(left, right), (right, left)] {
        let (Some(wl), Some(xl)) = (leaf_of(w, &relations), leaf_of(x, &relations)) else {
            continue;
        };
        if wl == xl {
            continue;
        }
        if let (Some(wc), Some(xc)) = (column(w), column(x)) {
            if let Some(weights) = weights(relations[wl])? {
                found = Some((wl, xl, wc.clone(), xc.clone(), weights));
                break;
            }
        }
    }
    let Some((wl, vl, w_col, x_col, weights)) = found else {
        return Ok(None);
    };
    let (w_rel, v_rel) = (relations[wl], relations[vl]);
    let w_index = |c: &Column| w_rel.schema().index_of_column(c).ok();
    if w_index(&w_col).and_then(|i| weights.columns.get(i).copied()) != Some(None) {
        return Ok(None);
    }
    let fold = &weights.scores.found.fold;
    // The weights' columns as fold outputs.
    let output_of = |e: &Expr| -> Option<usize> {
        let c = column(e)?;
        weights.columns.get(w_index(c)?).copied().flatten()
    };
    // The kept output: the one the normalizer and the GROUP BY use; the
    // summed one: joined with the values.
    let n_outputs = fold.outputs().len();
    if n_outputs != 2 {
        return Ok(None);
    }
    let mut summed = None;
    let mut values_key = None;
    let mut kept_join = None;
    for (l, r) in &on {
        for (a, b) in [(l, r), (r, l)] {
            if leaf_of(a, &relations) == Some(wl) && leaf_of(b, &relations) == Some(vl) {
                summed = output_of(a);
                values_key = column(b).cloned();
            } else if leaf_of(a, &relations) == Some(wl)
                && leaf_of(b, &relations).is_some_and(|n| n != vl)
            {
                kept_join = Some((output_of(a), b.clone()));
            }
        }
    }
    let (Some(summed), Some(values_key)) = (summed, values_key) else {
        return Ok(None);
    };
    let kept = 1 - summed;
    // The maximum, if any, must be over the kept output's scores, or the
    // shift wouldn't cancel.
    if weights.shift == Shift::KeptMaximum && weights.maximum_per != Some(kept) {
        return Ok(None);
    }
    // The normalizer: the same weights, summed by the kept output.
    if normalizer == Normalizer::KeptGroups {
        let Some((Some(k), n_key)) = kept_join else {
            return Ok(None);
        };
        if k != kept || on.len() != 2 || relations.len() != 3 {
            return Ok(None);
        }
        let nl = leaf_of(&n_key, &relations).expect("a leaf");
        let z = match (z_group, z_inside) {
            (Some(g), _) => trace(&agg.input, &agg.group_expr[g])?,
            (None, Some(z)) => z,
            (None, None) => return Ok(None),
        };
        if !normalizes(relations[nl], &weights, kept, column(&n_key), column(&z))? {
            return Ok(None);
        }
    } else {
        // `SUM(w)` of the same weights, over the same rows.
        let other = trace(&agg.input, &sums[1 - weighted])?;
        if column(&other) != Some(&w_col) || on.len() != 1 || relations.len() != 2 {
            return Ok(None);
        }
    }

    // The outputs, in the projection's order: the kept output, and the
    // values' other columns, each its own dimension.
    let producer_dims = fold.dimensions().len();
    let mut own = Vec::new();
    let mut values_columns: Vec<Column> = vec![values_key.clone()];
    let mut outputs = Vec::new();
    let mut names = Vec::new();
    // Each GROUP BY key shown once: one shown twice would let the count
    // below pass with another key hidden.
    let mut shown = vec![false; groups];
    for (i, e) in keys.iter().enumerate() {
        let Some(c) = column(&e.clone().unalias()).cloned() else {
            return Ok(None);
        };
        let Some(g) = group_of(&c) else {
            return Ok(None);
        };
        if std::mem::replace(&mut shown[g], true) {
            return Ok(None);
        }
        let traced = trace(&agg.input, &agg.group_expr[g])?;
        match leaf_of(&traced, &relations) {
            Some(l) if l == wl && output_of(&traced) == Some(kept) => {
                outputs.push(fold.outputs()[kept].dimension);
            }
            Some(l) if l == vl => {
                let Some(vc) = column(&traced) else {
                    return Ok(None);
                };
                own.push(Dimension {
                    equality: KeyEquality::Equal,
                });
                values_columns.push(vc.clone());
                outputs.push(DimensionId(producer_dims + own.len() - 1));
            }
            _ => return Ok(None),
        }
        names.push(format!("{PREFIX}o{i}"));
    }
    names.push(format!("{PREFIX}v"));
    // Every GROUP BY key is shown, except the normalizer's.
    if outputs.len() + usize::from(z_group.is_some()) != groups {
        return Ok(None);
    }
    let summed_dim = fold.outputs()[summed].dimension;
    if fold.dimension(summed_dim).equality != KeyEquality::Equal {
        return Ok(None);
    }
    let mut keys_of_values = vec![Key::new(summed_dim, ColumnIndex(0))];
    for (i, _) in own.iter().enumerate() {
        keys_of_values.push(Key::new(DimensionId(producer_dims + i), ColumnIndex(i + 1)));
    }
    let values = Operand {
        keys: keys_of_values,
        factor: Some(ColumnIndex(values_columns.len())),
    };
    let mut exprs: Vec<Expr> = (values_columns.iter().enumerate())
        .map(|(i, c)| Expr::Column(c.clone()).alias(format!("k{i}")))
        .collect();
    exprs.push(Expr::Column(x_col).alias("f"));
    let values_plan = LogicalPlan::Projection(Projection::try_new(exprs, Arc::new(v_rel.clone()))?);
    let Ok(pipeline) = Pipeline::try_new(
        fold.clone(),
        weights.scores.scale,
        weights.shift,
        own,
        values,
        outputs,
        normalizer,
    ) else {
        return Ok(None);
    };
    let [q, k] =
        <[LogicalPlan; 2]>::try_from(weights.scores.found.operands.clone()).expect("two operands");
    let fallback = LogicalPlan::Projection(p.clone());
    let Ok(node) = PipelineNode::try_new(pipeline, [q, k, values_plan], names.clone(), fallback)
    else {
        return Ok(None);
    };
    detect_folds::restore(node.into_plan(), &names, &p.schema)
}

/// `e` as a product divided by a column: `a / z * b`, `b * (a / z)` or
/// `a * b / z`, as `(a * b, z)`. Dividing each term by `z` and dividing their
/// sum by it give the same average, up to one rounding per term, which the
/// summation bound covers.
fn divided(e: Expr) -> Option<(Expr, Expr)> {
    let Expr::BinaryExpr(BinaryExpr { left, op, right }) = e else {
        return None;
    };
    let quotient = |e: &Expr| -> Option<(Expr, Expr)> {
        match e {
            Expr::BinaryExpr(BinaryExpr {
                left,
                op: Operator::Divide,
                right,
            }) if column(right).is_some() => Some(((**left).clone(), (**right).clone())),
            _ => None,
        }
    };
    match op {
        // `a * b / z`.
        Operator::Divide if column(&right).is_some() => matches!(
            left.as_ref(),
            Expr::BinaryExpr(BinaryExpr {
                op: Operator::Multiply,
                ..
            })
        )
        .then(|| (*left, *right)),
        // `a / z * b` or `b * (a / z)`.
        Operator::Multiply => {
            if let Some((a, z)) = quotient(&left) {
                return Some((a * *right, z));
            }
            let (a, z) = quotient(&right)?;
            Some((*left * a, z))
        }
        _ => None,
    }
}

/// Whether `plan` is `SUM(w) … GROUP BY i` of weights the same as `weights`,
/// with `i` their `kept` output, `key` its key column and `z` its sum's.
fn normalizes(
    plan: &LogicalPlan,
    weights: &Weights,
    kept: usize,
    key: Option<&Column>,
    z: Option<&Column>,
) -> Result<bool> {
    let (Some(key), Some(z)) = (key, z) else {
        return Ok(false);
    };
    let LogicalPlan::Projection(p) = strip(plan) else {
        return Ok(false);
    };
    let LogicalPlan::Aggregate(agg) = p.input.as_ref() else {
        return Ok(false);
    };
    let ([g], [_], Some(arg)) = (
        agg.group_expr.as_slice(),
        agg.aggr_expr.as_slice(),
        detect_folds::sum_argument_at(agg, 0),
    ) else {
        return Ok(false);
    };
    let Some(theirs) = self::weights(&agg.input)? else {
        return Ok(false);
    };
    let same = same_scores(&theirs.scores, &weights.scores)
        && theirs.shift == weights.shift
        && theirs.maximum_per == weights.maximum_per;
    let role = |e: &Expr| {
        let i = agg.input.schema().index_of_column(column(e)?).ok()?;
        theirs.columns.get(i).copied()
    };
    // Grouped by the kept output, summing the weight.
    let grouped = role(g) == Some(Some(kept)) && role(&arg) == Some(None);
    // The projection shows the key and the sum as `key` and `z`.
    let shown = |c: Column| -> Option<Column> {
        let i = (p.expr.iter()).position(|e| e.clone().unalias() == Expr::Column(c.clone()))?;
        Some(column_at(plan, i))
    };
    let (g_col, s_col) = (column_at(&p.input, 0), column_at(&p.input, 1));
    Ok(same && grouped && shown(g_col).as_ref() == Some(key) && shown(s_col).as_ref() == Some(z))
}
