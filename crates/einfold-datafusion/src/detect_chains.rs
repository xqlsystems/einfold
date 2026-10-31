// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`chain`]: a `SUM` of a product of three factors, each from one of three
//! inner-joined relations, as a fold of three operands, such as a chain of
//! matrix products:
//!
//! ```sql
//! SELECT a.i, c.l, SUM(a.v * b.v * c.v)
//! FROM a JOIN b ON a.j = b.j JOIN c ON b.k = c.k GROUP BY a.i, c.l
//! ```
//!
//! The kernel computes it as two folds of two operands, contracting first
//! the pair that costs the least, `(a·b)·c` here, so that the join's rows,
//! one per `(i, j, k, l)`, never exist. That changes which floats are added,
//! so it runs only where no sum can overflow, and the node keeps the host's
//! plan for everywhere else.

use datafusion::arrow::datatypes::DataType;
use datafusion::common::{Column, Result};
use datafusion::logical_expr::{
    Aggregate, BinaryExpr, Expr, ExprSchemable, LogicalPlan, Operator, Projection,
};
use einfold_ir::{
    Aggregate as FoldAggregate, ColumnIndex, Dimension, DimensionId, Fold, Key, KeyEquality,
    Operand, Output, RowValue,
};

use crate::detect_folds::{self, Found};
use crate::detect_pipelines;

/// `agg` as a fold of three operands, if it is one: a plain `SUM` of
/// `DOUBLE`s, grouped by columns, over inner joins of three relations on
/// equalities alone, whose argument multiplies three factors, each reading
/// one relation and each relation read by one. Each join key and factor
/// must be safe to compute on every row of its relation (see
/// [`detect_folds::movable`]), as the operands compute them before the join.
pub(crate) fn chain(agg: &Aggregate) -> Result<Option<Found>> {
    let Some(arg) = detect_folds::sum_argument(agg) else {
        return Ok(None);
    };
    if agg.group_expr.is_empty() {
        return Ok(None);
    }
    // Down through projections to the joins.
    let mut projections = Vec::new();
    let mut below = agg.input.as_ref();
    loop {
        match below {
            LogicalPlan::Projection(p) => {
                projections.push(p);
                below = p.input.as_ref();
            }
            LogicalPlan::Join(_) => break,
            _ => return Ok(None),
        }
    }
    let (mut leaves, mut on) = (Vec::new(), Vec::new());
    if !detect_pipelines::leaves(below, &mut leaves, &mut on)? || leaves.len() != 3 {
        return Ok(None);
    }
    // Every column must name one relation, or a factor's would be ambiguous.
    let columns: Vec<Column> = leaves.iter().flat_map(|l| l.schema().columns()).collect();
    if (1..columns.len()).any(|i| columns[..i].contains(&columns[i])) {
        return Ok(None);
    }
    // The one relation every column of `e` belongs to.
    let leaf_of = |e: &Expr| -> Option<usize> {
        let mut read = e.column_refs().into_iter();
        let first = read.next()?;
        let l = leaves.iter().position(|p| p.schema().has_column(first))?;
        read.all(|c| leaves[l].schema().has_column(c)).then_some(l)
    };
    let traced = |e: &Expr| -> Result<Expr> {
        let e = detect_folds::resolve(e, &projections)?;
        detect_pipelines::trace(below, &e)
    };
    let value = traced(&arg)?;
    let groups = (agg.group_expr.iter())
        .map(traced)
        .collect::<Result<Vec<_>>>()?;

    // The factors, one per relation.
    let mut product = Vec::new();
    multiplied(&value, &leaf_of, &mut product);
    let mut factors: [Option<Expr>; 3] = [None, None, None];
    for f in product {
        let Some(l) = leaf_of(&f) else {
            return Ok(None);
        };
        let schema = leaves[l].schema();
        if f.get_type(schema)? != DataType::Float64 || !detect_folds::movable(&f, schema) {
            return Ok(None);
        }
        if factors[l].replace(f).is_some() {
            return Ok(None);
        }
    }
    let Some(factors) = factors.into_iter().collect::<Option<Vec<Expr>>>() else {
        return Ok(None);
    };

    // The dimensions: classes of key expressions that the joins equate, then
    // one for each group column no join equates. Each is held by at most one
    // expression of each relation.
    let mut keys: Vec<(usize, Expr)> = Vec::new();
    let mut class: Vec<usize> = Vec::new();
    let node = |l: usize, e: &Expr, keys: &mut Vec<(usize, Expr)>, class: &mut Vec<usize>| {
        if let Some(n) = keys.iter().position(|(m, k)| *m == l && k == e) {
            return n;
        }
        keys.push((l, e.clone()));
        class.push(class.len());
        class.len() - 1
    };
    fn root(class: &[usize], mut n: usize) -> usize {
        while class[n] != n {
            n = class[n];
        }
        n
    }
    for (left, right) in &on {
        let (Some(l), Some(r)) = (leaf_of(left), leaf_of(right)) else {
            return Ok(None);
        };
        let movable = |l: usize, e: &Expr| detect_folds::movable(e, leaves[l].schema());
        if l == r || !movable(l, left) || !movable(r, right) {
            return Ok(None);
        }
        let (a, b) = (
            node(l, left, &mut keys, &mut class),
            node(r, right, &mut keys, &mut class),
        );
        let (a, b) = (root(&class, a), root(&class, b));
        class[a.max(b)] = a.min(b);
    }
    let mut outputs_of = Vec::new();
    for g in &groups {
        let Expr::Column(_) = g else {
            return Ok(None);
        };
        let Some(l) = leaf_of(g) else {
            return Ok(None);
        };
        outputs_of.push(node(l, g, &mut keys, &mut class));
    }
    // The keys in the order of their relations and of their columns there,
    // and each class numbered in the order its first key comes: so that
    // `a(i, j)`, `b(j, k)` and `c(k, l)` hold dimensions 0 and 1, 1 and 2,
    // and 2 and 3, as `EXPLAIN` shows them.
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by_key(|&n| {
        let (l, e) = &keys[n];
        let column = match e {
            Expr::Column(c) => leaves[*l].schema().index_of_column(c).ok(),
            _ => None,
        };
        (*l, column.unwrap_or(usize::MAX), n)
    });
    let mut roots: Vec<usize> = Vec::new();
    let mut dimension_of = vec![0; keys.len()];
    for &n in &order {
        let r = root(&class, n);
        dimension_of[n] = match roots.iter().position(|&x| x == r) {
            Some(d) => d,
            None => {
                roots.push(r);
                roots.len() - 1
            }
        };
    }
    let mut outputs: Vec<Output> = Vec::new();
    for n in outputs_of {
        let dimension = DimensionId(dimension_of[n]);
        if outputs.iter().any(|o| o.dimension == dimension) {
            return Ok(None);
        }
        outputs.push(Output {
            dimension,
            domain_key: None,
        });
    }

    // Each relation's operand projects its keys, then its factor.
    let mut operands = Vec::new();
    let mut plans = Vec::new();
    for (l, factor) in factors.into_iter().enumerate() {
        let mut exprs = Vec::new();
        let mut held = Vec::new();
        for &n in &order {
            let (m, e) = &keys[n];
            if *m != l {
                continue;
            }
            let dimension = DimensionId(dimension_of[n]);
            if held.iter().any(|k: &Key| k.dimension == dimension) {
                return Ok(None);
            }
            held.push(Key::new(dimension, ColumnIndex(exprs.len())));
            exprs.push(e.clone().alias(format!("k{}", exprs.len())));
        }
        let factor_column = ColumnIndex(exprs.len());
        exprs.push(factor.alias("f0"));
        operands.push(Operand {
            keys: held,
            factor: Some(factor_column),
        });
        let leaf = std::sync::Arc::new(leaves[l].clone());
        plans.push(LogicalPlan::Projection(Projection::try_new(exprs, leaf)?));
    }
    let dimensions = vec![
        Dimension {
            equality: KeyEquality::Equal,
        };
        roots.len()
    ];
    let Ok(fold) = Fold::try_new(
        dimensions,
        operands,
        outputs,
        RowValue::Product,
        FoldAggregate::sum(),
        None,
    ) else {
        return Ok(None);
    };
    Ok(Some(Found {
        fold,
        operands: plans,
    }))
}

/// The factors of `e`, a product, in order, each the largest part that
/// reads one relation (see `leaf_of`): `a.v * b.v * c.v` is `[a.v, b.v,
/// c.v]` however it is parenthesized, and `(a.v * a.w) * b.v` is `[a.v *
/// a.w, b.v]`.
fn multiplied(e: &Expr, leaf_of: &dyn Fn(&Expr) -> Option<usize>, out: &mut Vec<Expr>) {
    match e {
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Multiply,
            right,
        }) if leaf_of(e).is_none() => {
            multiplied(left, leaf_of, out);
            multiplied(right, leaf_of, out);
        }
        other => out.push(other.clone()),
    }
}
