// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The third shape [`DetectFolds`](crate::DetectFolds) finds: a fold joined
//! back to its domain. In SQL,
//!
//! ```sql
//! SELECT <d's columns, and s>
//! FROM d LEFT JOIN (
//!     SELECT d2.<keys>, SUM(<e>) AS s
//!     FROM d2 JOIN f ON d2.<key> = f.<output> AND …
//!     GROUP BY d2.<keys>
//! ) r ON d.<key> = r.<key> AND …
//! ```
//!
//! where `f` is a fold the rule found, `d2` is `d` under other names, the
//! inner join matches every one of `f`'s outputs, the second `GROUP BY` is by
//! exactly those keys, and `e` is `f`'s value, or `NULL` where a test of
//! `d2`'s row holds. ddx, an XQL Systems project for automatic
//! differentiation of SQL queries, writes each gradient step this way.
//!
//! When no two rows of `d` have the keys of one of `f`'s groups, each group
//! of the second `GROUP BY` holds one row, and the query is the fold with `d`
//! as its domain: each row of `d` gets `0.0 + e` evaluated on itself and its
//! group's value, where `0.0 +` is what `SUM` of one value computes, and NULL
//! where no group matched. The node computes `e`'s tests as a column of its
//! domain, and makes the value NULL where one holds, so that the host's plan,
//! when it runs instead, isn't given them a second time. `e` is NULL where `f`'s value is, so a missing
//! group and a group with a NULL sum agree. Whether the keys are unique is
//! known only when the query runs, so the node checks it then, and runs the
//! host's own plan where they aren't.

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, NullEquality, Result};
use datafusion::logical_expr::expr::Case;
use datafusion::logical_expr::{lit, Expr, Join, JoinType, LogicalPlan, Projection};
use einfold_ir::{ColumnIndex, Domain, DomainKey, Fold, KeyEquality, Output};

use crate::detect_folds::{
    found_fold, host_plan, movable, resolve, restore, same_schema, sum_argument, PREFIX,
};
use crate::fold_node::FoldNode;

/// `plan` without the aliases and column-only projections over it, with, for
/// each of `plan`'s columns, its column in what is left. A table scan's
/// columns are given as the table's own, whichever of them it reads.
fn base(plan: &LogicalPlan) -> Option<(&LogicalPlan, Vec<usize>)> {
    match plan {
        LogicalPlan::SubqueryAlias(a) => base(&a.input),
        LogicalPlan::Projection(p) => {
            let (b, below) = base(&p.input)?;
            let columns = (p.expr.iter())
                .map(|e| match e.clone().unalias() {
                    Expr::Column(c) => p.input.schema().index_of_column(&c).ok().map(|i| below[i]),
                    _ => None,
                })
                .collect::<Option<Vec<usize>>>()?;
            Some((b, columns))
        }
        LogicalPlan::TableScan(scan) => match &scan.projection {
            Some(columns) => Some((plan, columns.clone())),
            None => Some((plan, (0..scan.source.schema().fields().len()).collect())),
        },
        other => Some((other, (0..other.schema().fields().len()).collect())),
    }
}

/// Whether two bases (see [`base`]) are one relation: the same plan, or
/// scans of one table with the same filters and limit, whichever columns
/// each reads. Within one plan, a table's name names one table, though
/// each scan of it may hold its own handle to it.
fn same_relation(a: &LogicalPlan, b: &LogicalPlan) -> bool {
    match (a, b) {
        (LogicalPlan::TableScan(a), LogicalPlan::TableScan(b)) => {
            a.table_name == b.table_name
                && a.source.schema() == b.source.schema()
                && a.filters == b.filters
                && a.fetch == b.fetch
        }
        (a, b) => a == b,
    }
}

/// Whether `e` is `value`, or `CASE WHEN <test> THEN NULL ELSE value END`
/// with `test` reading only `row`'s columns: NULL wherever `value` is.
fn null_with_value(e: &Expr, value: &Column, row: &LogicalPlan) -> bool {
    let is_value = |e: &Expr| matches!(e, Expr::Column(c) if c == value);
    match e {
        Expr::Case(Case {
            expr: None,
            when_then_expr,
            else_expr: Some(otherwise),
        }) => {
            is_value(otherwise)
                && when_then_expr.iter().all(|(test, then)| {
                    matches!(then.as_ref(), Expr::Literal(v, _) if v.is_null())
                        && (test.column_refs().iter()).all(|c| row.schema().has_column(c))
                })
        }
        e => is_value(e),
    }
}

/// `p` over a fold joined back to its domain (see the module's
/// documentation), as the fold from its domain, if that is what `p` is.
pub(crate) fn joined_back(p: &Projection) -> Result<Option<LogicalPlan>> {
    let LogicalPlan::Join(Join {
        left: d,
        right,
        on,
        filter: None,
        join_type: JoinType::Left,
        null_equality: outer,
        ..
    }) = p.input.as_ref()
    else {
        return Ok(None);
    };
    // The right side: an aggregate with one SUM, under aliases and renames;
    // each of its columns is one of the aggregate's.
    let Some((LogicalPlan::Aggregate(agg), right_map)) = base(right) else {
        return Ok(None);
    };
    let Some(arg) = sum_argument(agg) else {
        return Ok(None);
    };
    // Down through projections to the inner join of d2 and the fold.
    let mut projections = Vec::new();
    let mut below = agg.input.as_ref();
    let inner = loop {
        match below {
            LogicalPlan::Projection(q) => {
                projections.push(q);
                below = q.input.as_ref();
            }
            LogicalPlan::Join(j) => break j,
            _ => return Ok(None),
        }
    };
    if inner.join_type != JoinType::Inner || inner.filter.is_some() {
        return Ok(None);
    }
    let (d2, f, (node, f_columns), flip) = match (found_fold(&inner.left), found_fold(&inner.right))
    {
        (None, Some(found)) => (&inner.left, &inner.right, found, false),
        (Some(found), None) => (&inner.right, &inner.left, found, true),
        _ => return Ok(None),
    };
    let fold = node.fold();
    let n = fold.outputs().len();
    // d and d2 must be one relation under two sets of names.
    let (Some((d_base, d_map)), Some((d2_base, d2_map))) = (base(d), base(d2)) else {
        return Ok(None);
    };
    if !same_relation(d_base, d2_base) {
        return Ok(None);
    }
    // The inner join pairs a column of d2 with each of the fold's outputs.
    let mut output_key: Vec<Option<usize>> = vec![None; n];
    for (l, r) in &inner.on {
        let (dk, fk) = if flip { (r, l) } else { (l, r) };
        let (Expr::Column(dc), Expr::Column(fc)) = (dk, fk) else {
            return Ok(None);
        };
        let (Ok(di), Ok(fi)) = (
            d2.schema().index_of_column(dc),
            f.schema().index_of_column(fc),
        ) else {
            return Ok(None);
        };
        let o = f_columns[fi];
        if o >= n || output_key[o].replace(d2_map[di]).is_some() {
            return Ok(None);
        }
    }
    let Some(output_key) = output_key.into_iter().collect::<Option<Vec<usize>>>() else {
        return Ok(None);
    };
    // The second GROUP BY is by exactly those columns, each once.
    let mut grouped = Vec::new();
    for g in &agg.group_expr {
        let Expr::Column(c) = resolve(g, &projections)? else {
            return Ok(None);
        };
        let Ok(i) = d2.schema().index_of_column(&c) else {
            return Ok(None);
        };
        grouped.push(d2_map[i]);
    }
    let (mut a, mut b) = (grouped.clone(), output_key.clone());
    a.sort();
    b.sort();
    a.dedup();
    if a != b || grouped.len() != n {
        return Ok(None);
    }
    // The left join pairs each group key with its own column of d.
    let right_schema = right.schema();
    let mut joined = vec![false; n];
    for (l, r) in on {
        let (Expr::Column(lc), Expr::Column(rc)) = (l, r) else {
            return Ok(None);
        };
        let (Ok(di), Ok(j)) = (
            d.schema().index_of_column(lc),
            right_schema.index_of_column(rc),
        ) else {
            return Ok(None);
        };
        let j = right_map[j];
        if j >= n || grouped[j] != d_map[di] || std::mem::replace(&mut joined[j], true) {
            return Ok(None);
        }
    }
    if joined.contains(&false) {
        return Ok(None);
    }
    // A NULL key the inner join matched, and `=` in the left join didn't.
    if *outer == NullEquality::NullEqualsNothing
        && inner.null_equality == NullEquality::NullEqualsNull
    {
        return Ok(None);
    }
    // The sum's argument: the fold's value, NULL wherever it is.
    let Some(v) = f_columns.iter().position(|&c| c == n) else {
        return Ok(None);
    };
    let value = Column::from(f.schema().qualified_field(v));
    let e = resolve(&arg, &projections)?;
    if !null_with_value(&e, &value, d2) {
        return Ok(None);
    }
    // p may read d's columns and the sum, not the second GROUP BY's keys.
    let Some(s) = right_map.iter().position(|&c| c == n) else {
        return Ok(None);
    };
    let sum = Column::from(right_schema.qualified_field(s));
    let reads_keys = p.expr.iter().any(|e| {
        (e.column_refs().iter()).any(|c| {
            let i = right_schema.index_of_column(c).ok();
            i.is_some_and(|i| right_map[i] != n)
        })
    });
    if reads_keys {
        return Ok(None);
    }

    // The fold, with d as its domain: each output matched with its column
    // of d, under the inner join's equality.
    let d_column = |b: usize| d_map.iter().position(|&c| c == b);
    let equality = match inner.null_equality {
        NullEquality::NullEqualsNull => KeyEquality::NotDistinct,
        NullEquality::NullEqualsNothing => KeyEquality::Equal,
    };
    let mut outputs = Vec::new();
    for (o, &b) in fold.outputs().iter().zip(&output_key) {
        let Some(c) = d_column(b) else {
            return Ok(None);
        };
        outputs.push(Output {
            domain_key: Some(DomainKey::new(ColumnIndex(c), equality)),
            ..*o
        });
    }
    let width = d.schema().fields().len();
    // `e`'s tests, as a column of d: true where `e` is NULL whatever the
    // fold's value. The tests are then computed on every row of d, where the
    // host's plan computes them only on rows that joined the fold, so they
    // must be unable to fail.
    let d_expr = |c: &Column| -> Option<Expr> {
        let i = d2.schema().index_of_column(c).ok()?;
        let j = d_column(d2_map[i])?;
        Some(Expr::Column(Column::from(d.schema().qualified_field(j))))
    };
    let null_when = match &e {
        Expr::Case(Case { when_then_expr, .. }) => {
            let mut unmapped = false;
            let mut tests = Vec::new();
            for (test, _) in when_then_expr {
                let test = (test.as_ref().clone())
                    .transform(|x| match &x {
                        Expr::Column(c) => match d_expr(c) {
                            Some(e) => Ok(Transformed::yes(e)),
                            None => {
                                unmapped = true;
                                Ok(Transformed::no(x))
                            }
                        },
                        _ => Ok(Transformed::no(x)),
                    })?
                    .data;
                // A CASE branch is taken where its test is true, not NULL.
                tests.push(match test {
                    Expr::IsNull(_) | Expr::IsNotNull(_) => test,
                    test => Expr::IsTrue(Box::new(test)),
                });
            }
            let Some(mask) = tests.into_iter().reduce(Expr::or) else {
                return Ok(None);
            };
            if unmapped || !movable(&mask, d.schema()) {
                return Ok(None);
            }
            Some(mask)
        }
        _ => None,
    };
    let Ok(with_domain) = Fold::try_new(
        fold.dimensions().to_vec(),
        fold.operands().to_vec(),
        outputs,
        fold.row(),
        fold.aggregate().clone(),
        Some(Domain {
            carry: (0..width).map(ColumnIndex).collect(),
            missing: None,
        }),
    ) else {
        return Ok(None);
    };
    // The plan it replaces: the left join, as d's columns and the sum.
    let mut kept: Vec<Expr> = d.schema().columns().into_iter().map(Expr::Column).collect();
    kept.push(Expr::Column(sum.clone()));
    // The host's own plan, with the inner fold swapped back for the plan it
    // replaced, so that falling back doesn't run einfold again.
    let input = host_plan(p.input.as_ref().clone())?;
    let original = LogicalPlan::Projection(Projection::try_new(kept, Arc::new(input))?);
    let names: Vec<String> = (d.schema().fields().iter())
        .map(|f| f.name().clone())
        .chain([format!("{PREFIX}v")])
        .collect();
    let inputs = node.inputs_owned();
    // The domain: d, and the mask after its columns.
    let domain = match &null_when {
        None => d.as_ref().clone(),
        Some(mask) => {
            let mut columns: Vec<Expr> =
                d.schema().columns().into_iter().map(Expr::Column).collect();
            columns.push(mask.clone().alias(format!("{PREFIX}null")));
            LogicalPlan::Projection(Projection::try_new(columns, Arc::clone(d))?)
        }
    };
    let Ok(new) = FoldNode::try_new(
        with_domain,
        inputs[..2].to_vec(),
        Some(domain),
        names.clone(),
    ) else {
        return Ok(None);
    };
    let new = new
        .with_fallback(original.clone())
        .requiring_unique_domain();
    let new = match null_when {
        Some(_) => new.with_null_when(ColumnIndex(width)),
        None => new,
    }
    .into_plan();
    // Over it: d's columns, and the sum as SUM of one value gives it. The
    // node has applied `e`'s tests where it computed the fold, and the host's
    // plan has where it ran.
    let Some(columns) = restore(new, &names[..width], d.schema())? else {
        return Ok(None);
    };
    let LogicalPlan::Projection(columns) = columns else {
        return Ok(None);
    };
    let mut exprs = columns.expr.clone();
    let value = Expr::Column(Column::new_unqualified(&names[width]));
    exprs.push((lit(0.0) + value).alias_qualified(sum.relation.clone(), &sum.name));
    let projection = Projection::try_new(exprs, Arc::clone(&columns.input))?;
    if !same_schema(&projection.schema, original.schema()) {
        return Ok(None);
    }
    let projection = Projection::try_new(
        p.expr.clone(),
        Arc::new(LogicalPlan::Projection(projection)),
    )?;
    Ok(same_schema(&projection.schema, &p.schema).then_some(LogicalPlan::Projection(projection)))
}
