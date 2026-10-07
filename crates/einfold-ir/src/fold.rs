// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A fold over a join: the query shape einfold makes fast.
//!
//! In SQL:
//!
//! ```sql
//! SELECT <output columns>, AGG(<row value>)
//! FROM a JOIN b ON <shared columns equal> ...
//! GROUP BY <output columns>
//! ```
//!
//! An engine usually materializes every row of the join, then aggregates.
//! einfold instead *folds* each joined row into its group's state as the
//! join produces it, so the join's rows never exist.
//!
//! The most important case is the *einsum* (Einstein summation), where the
//! row value is a product of one factor per input and the aggregate is
//! `SUM`. The matrix product `C[i,j] = Σ_k A[i,k]·B[k,j]` is one:
//!
//! ```sql
//! SELECT a.i, b.j, SUM(a.v * b.v) FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j
//! ```
//!
//! Folds with that product structure admit algebraic rewrites, such as
//! summing early and choosing the order of joins, that other folds don't; see
//! [`Fold::semiring`].

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::aggregate::Aggregate;

/// A dimension of a fold: a set of columns that the query equates.
///
/// Join conditions such as `a.k = b.k AND b.k = c.k` put `a.k`, `b.k` and
/// `c.k` in one dimension. Operands that share a dimension are joined on it.
/// A column that is only grouped by, and never joined, is a dimension of its
/// own. Data axes such as latitude or time become dimensions when a query
/// joins or groups on them.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Dim(pub String);

impl Dim {
    /// A dimension with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Dim(name.into())
    }
}

impl fmt::Display for Dim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// How the query compares a dimension's values when joining on it.
///
/// The two kinds differ only for NULL keys, and einfold preserves whichever
/// the plan used.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyEquality {
    /// SQL `=`: a NULL key matches nothing, so rows with a NULL key never join.
    Equal,
    /// SQL `IS NOT DISTINCT FROM`: NULL matches NULL, as in `GROUP BY`.
    NotDistinctFrom,
}

/// A pair of operations, "add" (⊕) and "multiply" (⊗), where ⊗ distributes
/// over ⊕: `a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)`.
///
/// A fold whose row value is a ⊗-product of one factor per operand, and whose
/// aggregate is ⊕, has this structure. It is what makes it correct to
/// aggregate one operand before joining it with the others (`Σ_j a·b_j =
/// a·Σ_j b_j`), and to choose the order of joins freely. Only `SUM` of `*`
/// exists today; others, such as `MIN` of `+` for shortest paths, may follow.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Semiring {
    /// `SUM` of `*`: ordinary arithmetic, as in linear algebra. `COUNT` of a
    /// product is also this semiring, over 0/1 indicators of non-NULL values.
    SumProduct,
}

/// What each joined row contributes to its group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum RowValue {
    /// A product of one *factor* per operand, where each factor is an
    /// expression over that operand's columns alone. An operand with no
    /// factor of its own contributes 1. This is the structure that the
    /// algebraic rewrites need.
    Product,
    /// Any other expression over the joined row, such as `exp(a.v * b.v)`.
    /// Such a fold can still be computed without materializing the join, but
    /// admits no algebraic rewrites.
    Expr,
}

/// One input of a fold: a table, a subquery, or any other relation, over
/// some dimensions.
///
/// `dims` lists the operand's dimension columns in order. A dimension may
/// repeat, which is a diagonal: the query equated two columns of one table,
/// as in `WHERE a.i = a.j`. The operand's factor, if any, is held by
/// whatever binds the fold to a query plan, not here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Operand {
    /// A name for display and error messages, such as the table's name.
    pub name: String,
    /// The operand's dimensions, one per dimension column.
    pub dims: Vec<Dim>,
}

impl Operand {
    /// An operand named `name` over `dims`.
    pub fn new(name: impl Into<String>, dims: impl IntoIterator<Item = Dim>) -> Self {
        Operand {
            name: name.into(),
            dims: dims.into_iter().collect(),
        }
    }

    /// The operand's distinct dimensions.
    pub fn dim_set(&self) -> BTreeSet<Dim> {
        self.dims.iter().cloned().collect()
    }
}

/// Why a [`Fold`] could not be built.
///
/// More reasons may be added as the representation grows.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FoldError {
    /// The fold has no operands.
    NoOperands,
    /// An output dimension does not appear in any operand.
    OutputDimNotInOperands(Dim),
    /// An output dimension is listed twice.
    DuplicateOutputDim(Dim),
    /// A dimension has no recorded [`KeyEquality`].
    MissingEquality(Dim),
}

impl fmt::Display for FoldError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FoldError::NoOperands => write!(f, "a fold needs at least one operand"),
            FoldError::OutputDimNotInOperands(d) => {
                write!(f, "output dimension `{d}` appears in no operand")
            }
            FoldError::DuplicateOutputDim(d) => {
                write!(f, "output dimension `{d}` is listed twice")
            }
            FoldError::MissingEquality(d) => {
                write!(
                    f,
                    "dimension `{d}` has no key equality (`=` or `IS NOT DISTINCT FROM`)"
                )
            }
        }
    }
}

impl std::error::Error for FoldError {}

/// A fold over a join: group the join of the operands by the output
/// dimensions, and fold each group's row values with an aggregate.
///
/// In SQL terms: `SELECT <output>, AGG(<row value>) FROM <operands> WHERE
/// <columns of each dimension equal> GROUP BY <output>`. Its result has one row
/// per output tuple that some joined row reached: a group no row reaches
/// does not exist, as with any SQL `GROUP BY`.
///
/// Each dimension has one [`KeyEquality`]. SQL attaches `=` or
/// `IS NOT DISTINCT FROM` to each join condition, not to a dimension, so this
/// representation is only valid for queries whose conditions within a
/// dimension all agree; whoever builds a `Fold` from a query must check that.
///
/// The output lists *dimensions*. A query may name one dimension in several
/// output columns (`GROUP BY a.k, b.k` where `a.k = b.k`); mapping dimensions
/// back to columns is the job of whatever binds the fold to a query.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fold {
    operands: Vec<Operand>,
    output: Vec<Dim>,
    equality: BTreeMap<Dim, KeyEquality>,
    row_value: RowValue,
    aggregate: Aggregate,
}

impl Fold {
    /// Build a fold, checking that it is well formed.
    ///
    /// `equality` must give the [`KeyEquality`] of every dimension that
    /// appears in an operand.
    pub fn new(
        operands: Vec<Operand>,
        output: Vec<Dim>,
        equality: BTreeMap<Dim, KeyEquality>,
        row_value: RowValue,
        aggregate: Aggregate,
    ) -> Result<Self, FoldError> {
        if operands.is_empty() {
            return Err(FoldError::NoOperands);
        }
        let all: BTreeSet<Dim> = operands
            .iter()
            .flat_map(|o| o.dims.iter().cloned())
            .collect();
        let mut seen = BTreeSet::new();
        for d in &output {
            if !all.contains(d) {
                return Err(FoldError::OutputDimNotInOperands(d.clone()));
            }
            if !seen.insert(d.clone()) {
                return Err(FoldError::DuplicateOutputDim(d.clone()));
            }
        }
        if let Some(d) = all.iter().find(|d| !equality.contains_key(*d)) {
            return Err(FoldError::MissingEquality(d.clone()));
        }
        Ok(Fold {
            operands,
            output,
            equality,
            row_value,
            aggregate,
        })
    }

    /// An einsum: `SUM` of the product of the operands' factors.
    pub fn einsum(
        operands: Vec<Operand>,
        output: Vec<Dim>,
        equality: BTreeMap<Dim, KeyEquality>,
    ) -> Result<Self, FoldError> {
        Fold::new(
            operands,
            output,
            equality,
            RowValue::Product,
            Aggregate::Sum,
        )
    }

    /// The operands, in plan order.
    pub fn operands(&self) -> &[Operand] {
        &self.operands
    }

    /// The output dimensions `O`, in the order of the result's columns.
    pub fn output(&self) -> &[Dim] {
        &self.output
    }

    /// What each joined row contributes.
    pub fn row_value(&self) -> RowValue {
        self.row_value
    }

    /// The aggregate folding each group's row values.
    pub fn aggregate(&self) -> Aggregate {
        self.aggregate
    }

    /// The semiring this fold computes in, if any.
    ///
    /// A fold is a semiring fold when its row value is a product of
    /// per-operand factors and its aggregate is the semiring's "add". Only
    /// then may an optimizer aggregate an operand before joining it, or
    /// reorder the joins. `AVG` is not a semiring aggregate, but is exactly
    /// `SUM / COUNT`, two semiring folds over the same join.
    pub fn semiring(&self) -> Option<Semiring> {
        match (self.row_value, self.aggregate) {
            (RowValue::Product, Aggregate::Sum | Aggregate::Count) => Some(Semiring::SumProduct),
            _ => None,
        }
    }

    /// How the plan compares coordinates of dimension `d`, if `d` is a dimension.
    pub fn equality(&self, d: &Dim) -> Option<KeyEquality> {
        self.equality.get(d).copied()
    }

    /// Every dimension of every operand.
    pub fn dims(&self) -> BTreeSet<Dim> {
        self.operands
            .iter()
            .flat_map(|o| o.dims.iter().cloned())
            .collect()
    }

    /// Dimensions aggregated away: those not in the output.
    pub fn summed(&self) -> BTreeSet<Dim> {
        let out: BTreeSet<&Dim> = self.output.iter().collect();
        self.dims()
            .into_iter()
            .filter(|d| !out.contains(d))
            .collect()
    }

    /// Dimensions that appear in two or more operands: the join keys.
    pub fn shared(&self) -> BTreeSet<Dim> {
        self.dims()
            .into_iter()
            .filter(|d| self.operands.iter().filter(|o| o.dims.contains(d)).count() >= 2)
            .collect()
    }

    /// Dimensions private to operand `i`: in that operand only, and summed.
    pub fn private(&self, i: usize) -> BTreeSet<Dim> {
        let shared = self.shared();
        let out: BTreeSet<&Dim> = self.output.iter().collect();
        self.operands[i]
            .dim_set()
            .into_iter()
            .filter(|d| !shared.contains(d) && !out.contains(d))
            .collect()
    }
}

impl fmt::Display for Fold {
    /// The aggregate over the operands, with named dimensions, such as
    /// `SUM(A[i,k] · B[k,j]) -> [i,j]` for a product, or `SUM(f(A[i,k], B[k,j]))
    /// -> [i,j]` for another row value.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let join = |ds: &[Dim]| {
            ds.iter()
                .map(|d| d.0.as_str())
                .collect::<Vec<_>>()
                .join(",")
        };
        let ops: Vec<String> = self
            .operands
            .iter()
            .map(|o| format!("{}[{}]", o.name, join(&o.dims)))
            .collect();
        let value = match self.row_value {
            RowValue::Product => ops.join(" · "),
            RowValue::Expr => format!("f({})", ops.join(", ")),
        };
        write!(
            f,
            "{}({}) -> [{}]",
            self.aggregate,
            value,
            join(&self.output)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> Dim {
        Dim::new(s)
    }

    fn eq_all(ds: &[&str]) -> BTreeMap<Dim, KeyEquality> {
        ds.iter().map(|s| (d(s), KeyEquality::Equal)).collect()
    }

    fn ab() -> Vec<Operand> {
        vec![
            Operand::new("A", [d("i"), d("k")]),
            Operand::new("B", [d("k"), d("j")]),
        ]
    }

    #[test]
    fn matmul_roles() {
        let f = Fold::einsum(ab(), vec![d("i"), d("j")], eq_all(&["i", "j", "k"])).unwrap();
        assert_eq!(f.summed(), [d("k")].into());
        assert_eq!(f.shared(), [d("k")].into());
        assert!(f.private(0).is_empty());
        assert_eq!(f.to_string(), "SUM(A[i,k] · B[k,j]) -> [i,j]");
    }

    #[test]
    fn semiring_is_derived_from_row_value_and_aggregate() {
        let make = |v, a| Fold::new(ab(), vec![d("i")], eq_all(&["i", "j", "k"]), v, a).unwrap();
        let p = RowValue::Product;
        assert_eq!(
            make(p, Aggregate::Sum).semiring(),
            Some(Semiring::SumProduct)
        );
        assert_eq!(
            make(p, Aggregate::Count).semiring(),
            Some(Semiring::SumProduct)
        );
        assert_eq!(make(p, Aggregate::Avg).semiring(), None);
        assert_eq!(make(RowValue::Expr, Aggregate::Sum).semiring(), None);
        assert_eq!(
            make(RowValue::Expr, Aggregate::Avg).to_string(),
            "AVG(f(A[i,k], B[k,j])) -> [i]"
        );
    }

    #[test]
    fn private_dimensions() {
        // A[i,j,x] · B[j] -> [i]: j is shared; x is aggregated and only in A, so private to A.
        let f = Fold::einsum(
            vec![
                Operand::new("A", [d("i"), d("j"), d("x")]),
                Operand::new("B", [d("j")]),
            ],
            vec![d("i")],
            eq_all(&["i", "j", "x"]),
        )
        .unwrap();
        assert_eq!(f.private(0), [d("x")].into());
        assert!(f.private(1).is_empty());
        assert_eq!(f.summed(), [d("j"), d("x")].into());
    }

    #[test]
    fn rejects_malformed() {
        let a = || vec![Operand::new("A", [d("i")])];
        assert_eq!(
            Fold::einsum(vec![], vec![], BTreeMap::new()),
            Err(FoldError::NoOperands)
        );
        assert_eq!(
            Fold::einsum(a(), vec![d("z")], eq_all(&["i"])),
            Err(FoldError::OutputDimNotInOperands(d("z")))
        );
        assert_eq!(
            Fold::einsum(a(), vec![d("i"), d("i")], eq_all(&["i"])),
            Err(FoldError::DuplicateOutputDim(d("i")))
        );
        assert_eq!(
            Fold::einsum(a(), vec![d("i")], BTreeMap::new()),
            Err(FoldError::MissingEquality(d("i")))
        );
    }

    #[test]
    fn diagonal_is_allowed() {
        let f = Fold::einsum(
            vec![Operand::new("A", [d("i"), d("i")])],
            vec![d("i")],
            eq_all(&["i"]),
        )
        .unwrap();
        assert!(f.summed().is_empty());
        assert!(f.shared().is_empty());
    }
}
