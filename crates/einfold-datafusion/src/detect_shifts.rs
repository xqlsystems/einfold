// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`shifted`]: a fold whose group keys shift a key of one side of the join
//! by a key of the other, such as a convolution.
//!
//! A 3 × 3 convolution with dilation `d`, of an image `x(c, lat, lon, v)` by
//! a kernel `w(co, ci, ky, kx, v)`, is, in SQL,
//!
//! ```sql
//! SELECT w.co, x.lat - (w.ky - 1) * d AS lat,
//!        (x.lon - (w.kx - 1) * d + 64) % 64 AS lon, SUM(x.v * w.v)
//! FROM x JOIN w ON x.c = w.ci
//! WHERE x.lat - (w.ky - 1) * d BETWEEN 0 AND 31
//! GROUP BY 1, 2, 3
//! ```
//!
//! Each input cell, through each tap of the kernel, reaches the output cell
//! the tap's offset points to. Latitude stops at the grid's edges, as zero
//! padding gives: DataFusion pushes the `WHERE` on the group key into the
//! join. Longitude wraps around, as on a globe.
//!
//! A group key of the form `p·l + q·r + c`, where `l` is a column of one side
//! and `r` one of the other, neither a join key, and `p`, `q` and `c` integer
//! literals, possibly taken modulo a positive literal, is an output shown
//! shifted ([`ShiftedOutput`]): the output `l`, shifted by `r`. A condition of
//! the join that bounds the same sum, `p·l + q·r` plus any constant, by a
//! literal, bounds the shifted key. Everything else is as for any fold.

use datafusion::arrow::datatypes::DataType;
use datafusion::common::{DFSchema, Result};
use datafusion::logical_expr::{Aggregate, Between, BinaryExpr, Expr, Join, LogicalPlan, Operator};
use einfold_ir::{Fold, Output, ShiftedOutput};

use crate::detect_folds::{self, Found, MAX_OFFSET};

/// The largest coefficient of a key in a shifted key, `2¹⁶`. Keys the
/// kernels take are below `2³²`, so with offsets of at most `2³²` (see
/// [`MAX_OFFSET`]) the host's `Int64` arithmetic on them can't overflow, and
/// the shifted key is exactly the host's.
const MAX_SCALE: i128 = 1 << 16;

/// A group key `scale · output + by_scale · by + offset`, possibly modulo
/// `period`.
struct Shifted {
    output: Expr,
    by: Expr,
    scale: i64,
    by_scale: i64,
    offset: i64,
    period: Option<i64>,
    low: Option<i64>,
    high: Option<i64>,
}

/// An integer expression of columns as a sum: each column's coefficient,
/// plus a constant.
#[derive(Default)]
struct Affine {
    terms: Vec<(Expr, i128)>,
    constant: i128,
}

impl Affine {
    /// Adds `scale · e`, if `e` is a column, an integer literal, or sums,
    /// differences, negations and products by integer literals of them.
    fn add(&mut self, e: &Expr, scale: i128) -> Option<()> {
        match e {
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => match op {
                Operator::Plus => {
                    self.add(left, scale)?;
                    self.add(right, scale)
                }
                Operator::Minus => {
                    self.add(left, scale)?;
                    self.add(right, -scale)
                }
                Operator::Multiply => match (literal(left), literal(right)) {
                    (Some(k), _) => self.add(right, scale.checked_mul(k)?),
                    (_, Some(k)) => self.add(left, scale.checked_mul(k)?),
                    _ => None,
                },
                _ => None,
            },
            Expr::Negative(inner) => self.add(inner, -scale),
            Expr::Literal(..) => {
                self.constant = self.constant.checked_add(scale.checked_mul(literal(e)?)?)?;
                Some(())
            }
            Expr::Column(_) => {
                match self.terms.iter_mut().find(|(c, _)| c == e) {
                    Some((_, k)) => *k = k.checked_add(scale)?,
                    None => self.terms.push((e.clone(), scale)),
                }
                self.terms.retain(|&(_, k)| k != 0);
                Some(())
            }
            _ => None,
        }
    }

    /// `e` as a sum, if it is one.
    fn of(e: &Expr) -> Option<Self> {
        let mut sum = Affine::default();
        sum.add(e, 1)?;
        Some(sum)
    }
}

/// An integer literal's value, if `e` is one, of at most [`MAX_OFFSET`].
fn literal(e: &Expr) -> Option<i128> {
    let Expr::Literal(value, _) = e else {
        return None;
    };
    detect_folds::integer(value).filter(|v| v.abs() <= MAX_OFFSET)
}

/// `g`, a group key over `join`'s inputs, as a shifted key, if it is one: a
/// column of each side, each with a nonzero coefficient, plus a constant,
/// possibly modulo a positive literal, all `Int64`.
fn shifted_key(g: &Expr, inputs: [&DFSchema; 2], join: &Join) -> Option<Shifted> {
    use datafusion::logical_expr::ExprSchemable;
    let (sum, period) = match g {
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Modulo,
            right,
        }) => (left.as_ref(), Some(literal(right).filter(|&p| p > 0)?)),
        _ => (g, None),
    };
    let Affine { terms, constant } = Affine::of(sum)?;
    let [(a, p), (b, q)] = terms.as_slice() else {
        return None;
    };
    let side = |e: &Expr| detect_folds::side(e, inputs);
    let (Some(sa), Some(sb)) = (side(a), side(b)) else {
        return None;
    };
    if sa == sb || p.abs() > MAX_SCALE || q.abs() > MAX_SCALE {
        return None;
    }
    // The shifted key's arithmetic, and its columns, are the host's `Int64`.
    let int64 = |e: &Expr, s: usize| e.get_type(inputs[s]).ok() == Some(DataType::Int64);
    let joined = DFSchema::join(inputs[0], inputs[1]).ok()?;
    if !int64(a, sa) || !int64(b, sb) || g.get_type(&joined).ok() != Some(DataType::Int64) {
        return None;
    }
    // A join key is held by both sides; a shifted key's columns by one each.
    let keyed = |e: &Expr| join.on.iter().any(|(l, r)| l == e || r == e);
    if keyed(a) || keyed(b) {
        return None;
    }
    // The output is the left side's column, shifted by the right's.
    let ((output, scale), (by, by_scale)) = match sa {
        0 => ((a, p), (b, q)),
        _ => ((b, q), (a, p)),
    };
    Some(Shifted {
        output: output.clone(),
        by: by.clone(),
        scale: i64::try_from(*scale).ok()?,
        by_scale: i64::try_from(*by_scale).ok()?,
        offset: i64::try_from(constant)
            .ok()
            .filter(|c| i128::from(*c).abs() <= MAX_OFFSET)?,
        period: period.map(i64::try_from).transpose().ok()?,
        low: None,
        high: None,
    })
}

/// Bounds `c`, a condition of the join, puts on one of `shifts`, if it bounds
/// one: `true` if it did, and `None` if it bounds a shifted key with a
/// period, which the stencil algorithm can't take.
fn bound(c: &Expr, shifts: &mut [Shifted]) -> Option<bool> {
    if let Expr::Between(Between {
        expr,
        negated: false,
        low,
        high,
    }) = c
    {
        let x = expr.as_ref().clone();
        let above = x.clone().gt_eq(low.as_ref().clone());
        let below = x.lt_eq(high.as_ref().clone());
        // Both halves or neither: one half taken as a bound would leave the
        // other unchecked.
        let before: Vec<_> = shifts.iter().map(|s| (s.low, s.high)).collect();
        let found = bound(&above, shifts)? && bound(&below, shifts)?;
        if !found {
            for (s, (low, high)) in shifts.iter_mut().zip(before) {
                (s.low, s.high) = (low, high);
            }
        }
        return Some(found);
    }
    let Expr::BinaryExpr(BinaryExpr { left, op, right }) = c else {
        return Some(false);
    };
    // `=` is both bounds, as DataFusion writes `BETWEEN k AND k`.
    if !matches!(
        op,
        Operator::Lt | Operator::LtEq | Operator::Gt | Operator::GtEq | Operator::Eq
    ) {
        return Some(false);
    }
    // `left - right <op> 0`.
    let mut sum = Affine::default();
    if sum.add(left, 1).and_then(|()| sum.add(right, -1)).is_none() {
        return Some(false);
    }
    for s in shifts.iter_mut() {
        let coefficient = |e: &Expr| sum.terms.iter().find(|(c, _)| c == e).map(|&(_, k)| k);
        let (p, q) = (i128::from(s.scale), i128::from(s.by_scale));
        if sum.terms.len() != 2 {
            continue;
        }
        // `left - right` is `sign · (key - offset) + d`, so the condition is
        // `key <op> offset - d`, or, with sign -1, `key <op mirrored> offset +
        // d`.
        let sign = match (coefficient(&s.output), coefficient(&s.by)) {
            (Some(a), Some(b)) if (a, b) == (p, q) => 1,
            (Some(a), Some(b)) if (a, b) == (-p, -q) => -1,
            _ => continue,
        };
        if s.period.is_some() {
            return None;
        }
        let d = sum.constant;
        let (value, op) = match sign {
            1 => (i128::from(s.offset) - d, *op),
            _ => (i128::from(s.offset) + d, op.swap()?),
        };
        // As an inclusive bound on the key.
        let (low, high) = match op {
            Operator::Gt => (Some(value + 1), None),
            Operator::GtEq => (Some(value), None),
            Operator::Lt => (None, Some(value - 1)),
            Operator::Eq => (Some(value), Some(value)),
            _ => (None, Some(value)),
        };
        let fits = |v: Option<i128>| {
            v.map(|v| i64::try_from(v).ok())
                .map_or(Some(None), |v| v.map(Some))
        };
        let (low, high) = (fits(low)?, fits(high)?);
        if let Some(low) = low {
            s.low = Some(s.low.map_or(low, |l| l.max(low)));
        }
        if let Some(high) = high {
            s.high = Some(s.high.map_or(high, |h| h.min(high)));
        }
        return Some(true);
    }
    Some(false)
}

/// `agg` as a fold with shifted outputs, if it is one (see the module's
/// documentation).
pub(crate) fn shifted(agg: &Aggregate) -> Result<Option<Found>> {
    let Some(arg) = detect_folds::sum_argument(agg) else {
        return Ok(None);
    };
    if agg.group_expr.is_empty() {
        return Ok(None);
    }
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
    let inputs = [join.left.schema().as_ref(), join.right.schema().as_ref()];
    let groups = (agg.group_expr.iter())
        .map(|g| detect_folds::resolve(g, &projections))
        .collect::<Result<Vec<_>>>()?;
    let value = detect_folds::resolve(&arg, &projections)?;
    // Each group key a column, or a shifted key.
    let mut shifts: Vec<(usize, Shifted)> = Vec::new();
    for (i, g) in groups.iter().enumerate() {
        if matches!(g, Expr::Column(_)) {
            continue;
        }
        match shifted_key(g, inputs, join) {
            Some(s) => shifts.push((i, s)),
            None => return Ok(None),
        }
    }
    if shifts.is_empty() {
        return Ok(None);
    }
    // The join's conditions on the shifted keys become their bounds; the
    // rest stay the join's.
    let mut kept = Vec::new();
    if let Some(filter) = &join.filter {
        let mut specs: Vec<Shifted> = shifts.drain(..).map(|(_, s)| s).collect();
        for c in datafusion::logical_expr::utils::split_conjunction(filter) {
            match bound(c, &mut specs) {
                Some(true) => {}
                Some(false) => kept.push(c.clone()),
                None => return Ok(None),
            }
        }
        let at = (groups.iter().enumerate())
            .filter(|(_, g)| !matches!(g, Expr::Column(_)))
            .map(|(i, _)| i);
        shifts = at.zip(specs).collect();
    }
    let mut unbounded = join.clone();
    unbounded.filter = datafusion::logical_expr::utils::conjunction(kept);
    // The fold of the join without the bounds, grouped by each shifted key's
    // output and by the dimension it is shifted by, which are then summed
    // away under the shift.
    let mut plain = groups.clone();
    for (i, s) in &shifts {
        plain[*i] = s.output.clone();
    }
    plain.extend(shifts.iter().map(|(_, s)| s.by.clone()));
    let Some(found) = detect_folds::fold_over(&unbounded, &plain, &value, false)? else {
        return Ok(None);
    };
    let fold = &found.fold;
    let outputs: Vec<Output> = fold.outputs()[..groups.len()].to_vec();
    let shown: Vec<ShiftedOutput> = (shifts.iter().enumerate())
        .map(|(j, (i, s))| ShiftedOutput {
            output: fold.outputs()[*i].dimension,
            by: fold.outputs()[groups.len() + j].dimension,
            scale: s.scale,
            by_scale: s.by_scale,
            offset: s.offset,
            period: s.period,
            low: s.low,
            high: s.high,
        })
        .collect();
    let rebuilt = Fold::try_new(
        fold.dimensions().to_vec(),
        fold.operands().to_vec(),
        outputs,
        fold.row().clone(),
        fold.aggregate().clone(),
        None,
    )
    .and_then(|f| f.with_conditions(fold.conditions().to_vec()))
    .and_then(|f| f.with_shifts(shown));
    let Ok(rebuilt) = rebuilt else {
        return Ok(None);
    };
    Ok(Some(Found {
        fold: rebuilt,
        operands: found.operands,
    }))
}
