// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A fold over a join: an aggregate computed over the rows of a join,
//! described by its operands, dimensions and aggregate rather than by SQL
//! text.
//!
//! With operands `a` and `b`, the row value [`RowValue::Product`] and the
//! aggregate [`Aggregate::sum`], a fold means, in SQL:
//!
//! ```sql
//! SELECT <outputs>, SUM(a.<factor> * b.<factor>) AS <value>
//! FROM a JOIN b ON a.<dimension's column> = b.<its column> AND …
//! GROUP BY <outputs>
//! ```
//!
//! A **dimension** is one variable of the fold: the set of columns, at most
//! one per operand, that the join equates. A dimension in two operands is a
//! join key. A dimension that is not an output is aggregated away; when only
//! one operand holds it, its values are simply not grouped by. Matrix
//! multiplication `C[n, h] = Σ_d X[n, d]·W[d, h]` is the fold with dimensions
//! `n`, `d` and `h`, operands `X(n, d)` and `W(d, h)`, and outputs `n` and `h`.
//!
//! An **output** is a dimension shown as a column of the result. Each
//! dimension is shown at most once; a query that shows one in several columns
//! copies it in a projection above the fold.
//!
//! **A fold names no columns.** Its inputs' columns are given by position
//! ([`ColumnIndex`]), and so are its result's: without a domain, the
//! outputs' key columns in order, then the value; with one, the domain's
//! carried columns, then the value (see [`Domain`]). Naming them is the host's business, as it is for a
//! Substrait relation, so two folds that compute the same thing are equal
//! whatever the query called their columns.
//!
//! **Groups exist only where some row joined,** with or without outputs. So
//! a fold with no outputs has one row when anything joined and none
//! otherwise, unlike SQL's `SELECT SUM(…)` without `GROUP BY`, which always
//! returns one row.
//!
//! A fold may also have a **domain**: a relation that the output is
//! left-joined from, so that the output has one row per domain row (see
//! [`Domain`]).
//!
//! Indices into a fold's dimensions and into its inputs' columns have types
//! of their own, [`DimensionId`] and [`ColumnIndex`], so that one can't be
//! passed for the other.

use std::fmt;

/// A dimension of a fold: an index into [`Fold::dimensions`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DimensionId(pub usize);

/// A column of one of a fold's inputs, by position.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ColumnIndex(pub usize);

/// How a dimension's columns are compared in the join.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum KeyEquality {
    /// SQL `=`: a NULL key matches nothing, so its row joins no row.
    Equal,
    /// SQL `IS NOT DISTINCT FROM`: a NULL key matches a NULL key.
    NotDistinct,
}

/// One dimension of a fold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Dimension {
    /// How the join compares the dimension's columns. It matters only for a
    /// dimension in two operands. Grouping, as SQL's `GROUP BY` does, always
    /// treats NULL as one more value.
    pub equality: KeyEquality,
}

/// A column of an operand that holds one of the fold's dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Key {
    /// The dimension held.
    pub dimension: DimensionId,
    /// The operand's column that holds it.
    pub column: ColumnIndex,
}

impl Key {
    /// Dimension `dimension`, held in column `column`.
    #[must_use]
    pub fn new(dimension: DimensionId, column: ColumnIndex) -> Self {
        Key { dimension, column }
    }
}

/// One input of a fold: which of its columns hold which dimensions, and
/// which holds the factor it contributes to each joined row's value.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Operand {
    /// Each dimension this operand holds, and its column.
    pub keys: Vec<Key>,
    /// The column holding this operand's factor. `None` makes the operand a
    /// *mask*: its rows decide which rows join, and it contributes no factor,
    /// as if its factor were 1.
    pub factor: Option<ColumnIndex>,
}

impl Operand {
    /// The column of this operand that holds dimension `dim`, if any.
    #[must_use]
    pub fn column_of(&self, dim: DimensionId) -> Option<ColumnIndex> {
        let key = self.keys.iter().find(|k| k.dimension == dim);
        key.map(|k| k.column)
    }
}

/// The domain's column an output is matched with, and how.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DomainKey {
    /// The domain's column.
    pub column: ColumnIndex,
    /// How it is compared with the output's key.
    pub equality: KeyEquality,
}

impl DomainKey {
    /// The domain's column `column`, compared with `equality`.
    #[must_use]
    pub fn new(column: ColumnIndex, equality: KeyEquality) -> Self {
        DomainKey { column, equality }
    }
}

/// One key column of a fold's result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Output {
    /// The dimension shown.
    pub dimension: DimensionId,
    /// With a [`Domain`], the domain's column this output is matched with, and
    /// how; `None` without one.
    pub domain_key: Option<DomainKey>,
}

/// What each joined row contributes to its group.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum RowValue {
    /// The product of the operands' factors, NULL when any factor is.
    Product,
}

/// A commutative monoid that folds row values into one part of a group's
/// state: an operation with an identity, associative and commutative, so a
/// group's partial states merge in any grouping and order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Monoid {
    /// SQL `SUM`: `+`, skipping NULL; NULL when every value was NULL.
    Sum,
}

/// How a group's value is computed from its parts.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Finish {
    /// The value is part `i` itself.
    Part(usize),
}

/// How joined rows combine into each group's value: one or more monoid
/// parts, folded in one pass, then a final function of them.
///
/// This is the shape of an aggregate in Gray et al.'s classification and in
/// Haskell's `foldl` library: a *distributive* aggregate such as `SUM` is one
/// part; an *algebraic* one such as `AVG` is several (`SUM` and `COUNT`) with
/// a final function (division). The laws a kernel relies on, such as merging
/// partial states in any order, are each part's monoid laws.
///
/// SQL's semantics hold in every case: a group exists when some joined row
/// reaches it, whatever the row's value.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Aggregate {
    parts: Vec<Monoid>,
    finish: Finish,
}

impl Aggregate {
    /// SQL `SUM`.
    pub fn sum() -> Self {
        Aggregate {
            parts: vec![Monoid::Sum],
            finish: Finish::Part(0),
        }
    }

    /// The monoids folded in one pass.
    pub fn parts(&self) -> &[Monoid] {
        &self.parts
    }

    /// The final function of the parts.
    pub fn finish(&self) -> Finish {
        self.finish
    }

    /// Whether this is plain `SUM`, which with a [`RowValue::Product`] makes
    /// the fold an einsum: the sum-product semiring.
    pub fn is_sum(&self) -> bool {
        self.parts == [Monoid::Sum] && self.finish == Finish::Part(0)
    }
}

/// The relation a fold's output is left-joined from.
///
/// With a domain, the output has one row per domain row, and its columns are
/// the domain's carried columns, then the value. In SQL:
///
/// ```sql
/// SELECT <domain's carried columns>,
///        COALESCE(folded.value, <missing>) AS value
/// FROM domain LEFT JOIN folded
///   ON domain.<key of output 1> <equality> folded.<output 1> AND …
/// ```
///
/// where `folded` is the fold without its domain, and each output's
/// [`Output::domain_key`] names its domain column and equality. Every domain
/// row appears once for each group its keys match, and once with a NULL
/// value when none match. The `COALESCE` is left out when `missing` is
/// `None`.
///
/// ddx, an XQL Systems project for automatic differentiation of SQL queries,
/// writes each gradient this way: the table being differentiated is the
/// domain, so each of its rows gets a gradient, and 0 where none reached it.
#[derive(Clone, Debug)]
pub struct Domain {
    /// The domain's columns the output carries, in order, before the value.
    pub carry: Vec<ColumnIndex>,
    /// The value of a domain row whose group's value is NULL, or that matches
    /// no group: `COALESCE`'s second argument.
    pub missing: Option<f64>,
}

impl Domain {
    /// Everything that identifies the domain, with `missing` as its bits, so
    /// that equality, hashing and ordering are total (a NaN equals itself).
    fn identity(&self) -> (&[ColumnIndex], Option<u64>) {
        (&self.carry, self.missing.map(f64::to_bits))
    }
}

impl PartialEq for Domain {
    fn eq(&self, other: &Self) -> bool {
        self.identity() == other.identity()
    }
}

impl Eq for Domain {}

impl std::hash::Hash for Domain {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::hash::Hash::hash(&self.identity(), state)
    }
}

impl PartialOrd for Domain {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Domain {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.identity().cmp(&other.identity())
    }
}

/// A fold over a join of operands, grouped by output dimensions.
///
/// Build one with [`Fold::try_new`], which checks that the description is
/// consistent: holding a `Fold` is evidence that it is.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Fold {
    dimensions: Vec<Dimension>,
    operands: Vec<Operand>,
    outputs: Vec<Output>,
    row: RowValue,
    aggregate: Aggregate,
    domain: Option<Domain>,
}

/// Why a fold's description is inconsistent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InvalidFold(String);

impl fmt::Display for InvalidFold {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid fold: {}", self.0)
    }
}

impl std::error::Error for InvalidFold {}

impl Fold {
    /// A fold, if its description is consistent:
    ///
    /// - every dimension it names is in range;
    /// - an operand holds each dimension in at most one column (a dimension
    ///   repeated within an operand, a diagonal such as `ii->i`, is not
    ///   supported);
    /// - every dimension is held by some operand;
    /// - no dimension is output twice;
    /// - some operand has a factor;
    /// - the aggregate's final function names one of its parts;
    /// - with a domain, every output has a domain key, and without one, none
    ///   does.
    pub fn try_new(
        dimensions: Vec<Dimension>,
        operands: Vec<Operand>,
        outputs: Vec<Output>,
        row: RowValue,
        aggregate: Aggregate,
        domain: Option<Domain>,
    ) -> Result<Self, InvalidFold> {
        let bad = |why: String| Err(InvalidFold(why));
        let n = dimensions.len();
        for (i, op) in operands.iter().enumerate() {
            let mut seen = vec![false; n];
            for &Key {
                dimension: DimensionId(d),
                ..
            } in &op.keys
            {
                if d >= n {
                    return bad(format!("operand {i} names dimension {d}, of {n}"));
                }
                if std::mem::replace(&mut seen[d], true) {
                    return bad(format!("operand {i} holds dimension {d} twice"));
                }
            }
        }
        for d in 0..n {
            if !operands
                .iter()
                .any(|op| op.column_of(DimensionId(d)).is_some())
            {
                return bad(format!("no operand holds dimension {d}"));
            }
        }
        if operands.iter().all(|op| op.factor.is_none()) {
            return bad("no operand has a factor".into());
        }
        let mut seen = vec![false; n];
        for o in &outputs {
            let DimensionId(d) = o.dimension;
            if d >= n {
                return bad(format!("output dimension {d}, of {n}"));
            }
            if std::mem::replace(&mut seen[d], true) {
                return bad(format!("dimension {d} is output twice"));
            }
            if o.domain_key.is_some() != domain.is_some() {
                return bad(format!(
                    "output of dimension {d} has a domain key exactly when the fold has a domain"
                ));
            }
        }
        let Finish::Part(p) = aggregate.finish;
        if p >= aggregate.parts.len() {
            return bad(format!(
                "the aggregate finishes with part {p}, of {}",
                aggregate.parts.len()
            ));
        }
        Ok(Fold {
            dimensions,
            operands,
            outputs,
            row,
            aggregate,
            domain,
        })
    }

    /// The dimensions.
    pub fn dimensions(&self) -> &[Dimension] {
        &self.dimensions
    }

    /// Dimension `d`.
    pub fn dimension(&self, d: DimensionId) -> Dimension {
        self.dimensions[d.0]
    }

    /// Every dimension's id, in order.
    pub fn dimension_ids(&self) -> impl Iterator<Item = DimensionId> {
        (0..self.dimensions.len()).map(DimensionId)
    }

    /// The operands.
    pub fn operands(&self) -> &[Operand] {
        &self.operands
    }

    /// The outputs, in order: without a domain, the result's key columns;
    /// with one, the keys matched with the domain's columns, whose carried
    /// columns are the result's (see [`Domain`]).
    pub fn outputs(&self) -> &[Output] {
        &self.outputs
    }

    /// What each joined row contributes.
    pub fn row(&self) -> RowValue {
        self.row
    }

    /// How joined rows combine into each group's value.
    pub fn aggregate(&self) -> &Aggregate {
        &self.aggregate
    }

    /// The domain, if the output is left-joined from one (see [`Domain`]).
    pub fn domain(&self) -> Option<&Domain> {
        self.domain.as_ref()
    }

    /// The position in [`Fold::outputs`] of dimension `dim`, if it is an
    /// output.
    pub fn output_of(&self, dim: DimensionId) -> Option<usize> {
        self.outputs.iter().position(|o| o.dimension == dim)
    }

    /// Whether dimension `dim` is aggregated away: not an output.
    pub fn is_aggregated(&self, dim: DimensionId) -> bool {
        self.output_of(dim).is_none()
    }

    /// Whether this fold is an einsum: a `SUM` of products, the sum-product
    /// semiring. Derived from the row value and the aggregate, never declared.
    pub fn is_einsum(&self) -> bool {
        self.row == RowValue::Product && self.aggregate.is_sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EQ: Dimension = Dimension {
        equality: KeyEquality::Equal,
    };

    fn out(d: usize) -> Output {
        Output {
            dimension: DimensionId(d),
            domain_key: None,
        }
    }

    fn op(keys: &[(usize, usize)], factor: Option<usize>) -> Operand {
        Operand {
            keys: keys
                .iter()
                .map(|&(d, c)| Key::new(DimensionId(d), ColumnIndex(c)))
                .collect(),
            factor: factor.map(ColumnIndex),
        }
    }

    fn fold(
        dims: usize,
        operands: Vec<Operand>,
        outputs: Vec<Output>,
        domain: Option<Domain>,
    ) -> Result<Fold, InvalidFold> {
        Fold::try_new(
            vec![EQ; dims],
            operands,
            outputs,
            RowValue::Product,
            Aggregate::sum(),
            domain,
        )
    }

    /// `n,d * d,h -> outputs`.
    fn matmul(outputs: Vec<Output>, domain: Option<Domain>) -> Result<Fold, InvalidFold> {
        let ops = vec![
            op(&[(0, 0), (1, 1)], Some(2)),
            op(&[(1, 0), (2, 1)], Some(2)),
        ];
        fold(3, ops, outputs, domain)
    }

    #[test]
    fn matrix_product() {
        let f = matmul(vec![out(0), out(2)], None).unwrap();
        assert!(f.is_einsum());
        assert!(f.is_aggregated(DimensionId(1)));
        assert_eq!(f.output_of(DimensionId(2)), Some(1));
        assert_eq!(
            f.operands()[1].column_of(DimensionId(2)),
            Some(ColumnIndex(1))
        );
        assert_eq!(f.operands()[0].column_of(DimensionId(2)), None);
    }

    #[test]
    fn domains_compare_missing_by_bits() {
        let d = |missing| Domain {
            carry: vec![ColumnIndex(0), ColumnIndex(1)],
            missing,
        };
        assert_eq!(d(Some(f64::NAN)), d(Some(f64::NAN)));
        assert_ne!(d(Some(0.0)), d(Some(-0.0)));
        assert_ne!(d(Some(0.0)), d(None));
    }

    #[test]
    fn rejects_inconsistent_descriptions() {
        assert!(matmul(vec![out(0), out(0)], None).is_err());
        assert!(matmul(vec![out(3)], None).is_err());
        let domain = || Domain {
            carry: vec![ColumnIndex(0)],
            missing: Some(0.0),
        };
        // With a domain, every output needs a key; without one, none may.
        assert!(matmul(vec![out(0), out(2)], Some(domain())).is_err());
        let mut keyed = out(0);
        keyed.domain_key = Some(DomainKey::new(ColumnIndex(0), KeyEquality::NotDistinct));
        assert!(matmul(vec![keyed], None).is_err());
        assert!(matmul(vec![keyed], Some(domain())).is_ok());
        assert!(fold(1, vec![op(&[(0, 0), (0, 1)], Some(2))], vec![], None).is_err());
        assert!(fold(2, vec![op(&[(0, 0)], Some(1))], vec![], None).is_err());
        assert!(fold(1, vec![op(&[(0, 0)], None)], vec![], None).is_err());
        let mask_and_factor = vec![op(&[(0, 0)], None), op(&[(0, 0)], Some(1))];
        assert!(fold(1, mask_and_factor, vec![], None).is_ok());
    }
}
