// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A fold over a join: an aggregate computed over the rows of a join,
//! described by its operands, dimensions and aggregate rather than by SQL
//! text.
//!
//! With operands `a` and `b` and the aggregate [`Aggregate::SumOfProducts`],
//! a fold means, in SQL:
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
//! A dimension is not a column name: an **output** names the column a
//! dimension appears as in the result, since a query may show one dimension
//! under several names.
//!
//! A fold may also have a **domain**: a relation that the output is
//! left-joined from, so that the output has one row per domain row (see
//! [`Domain`]).

use std::fmt;

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

/// One input of a fold: which of its columns hold which dimensions, and
/// which holds the factor it contributes to each joined row's value.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Operand {
    /// `(dimension, column)` pairs: the dimension's index in
    /// [`Fold::dimensions`], and the column of this operand that holds it.
    pub keys: Vec<(usize, usize)>,
    /// The column holding this operand's factor. `None` makes the operand a
    /// *mask*: its rows decide which rows join, and it contributes no factor,
    /// as if its factor were 1.
    pub factor: Option<usize>,
}

impl Operand {
    /// The column of this operand that holds dimension `dim`, if any.
    pub fn column_of(&self, dim: usize) -> Option<usize> {
        self.keys.iter().find(|(d, _)| *d == dim).map(|(_, c)| *c)
    }
}

/// One key column of a fold's result.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Output {
    /// The dimension, as an index into [`Fold::dimensions`].
    pub dimension: usize,
    /// The column's name.
    pub name: String,
}

/// How a fold combines its joined rows into each group's value.
///
/// SQL's semantics hold in every case: a group exists when some joined row
/// reaches it, whatever the row's value, and its value skips NULL, so it is
/// NULL when every contribution was NULL.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Aggregate {
    /// `SUM` of the product of the operands' factors: an einsum, the
    /// sum-product semiring. A product is NULL when any factor is.
    SumOfProducts,
}

/// The relation a fold's output is left-joined from.
///
/// With a domain, the output has one row per domain row, in SQL:
///
/// ```sql
/// SELECT <domain's carried columns>,
///        COALESCE(folded.value, <missing>) AS value
/// FROM domain LEFT JOIN folded
///   ON domain.<key of output 1> <equality> folded.<output 1> AND …
/// ```
///
/// where `folded` is the fold without its domain. Every domain row appears
/// once for each group its keys match, and once with a NULL value when none
/// match. The `COALESCE` is left out when `missing` is `None`.
///
/// ddx, an XQL Systems project for automatic differentiation of SQL queries,
/// writes each gradient this way: the table being differentiated is the
/// domain, so each of its rows gets a gradient, and 0 where none reached it.
#[derive(Clone, Debug)]
pub struct Domain {
    /// For each of the fold's outputs, in order, the domain's column it is
    /// matched with, and how.
    pub keys: Vec<(usize, KeyEquality)>,
    /// The domain's columns the output carries, in order, before the value.
    pub carry: Vec<usize>,
    /// The value of a domain row whose group's value is NULL, or that matches
    /// no group: `COALESCE`'s second argument.
    pub missing: Option<f64>,
}

impl Domain {
    /// Everything that identifies the domain, with `missing` as its bits, so
    /// that equality, hashing and ordering are total (a NaN equals itself).
    fn identity(&self) -> (&[(usize, KeyEquality)], &[usize], Option<u64>) {
        (&self.keys, &self.carry, self.missing.map(f64::to_bits))
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
        self.identity().hash(state)
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
    aggregate: Aggregate,
    value: String,
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
    /// - every dimension index is in range;
    /// - an operand holds each dimension in at most one column (a dimension
    ///   repeated within an operand, a diagonal such as `ii->i`, is not
    ///   supported);
    /// - every dimension is held by some operand;
    /// - no dimension is output twice;
    /// - some operand has a factor;
    /// - a domain names one key column per output.
    ///
    /// `value` names the output's value column.
    pub fn try_new(
        dimensions: Vec<Dimension>,
        operands: Vec<Operand>,
        outputs: Vec<Output>,
        aggregate: Aggregate,
        value: impl Into<String>,
        domain: Option<Domain>,
    ) -> Result<Self, InvalidFold> {
        let bad = |why: String| Err(InvalidFold(why));
        let n = dimensions.len();
        for (i, op) in operands.iter().enumerate() {
            let mut seen = vec![false; n];
            for &(d, _) in &op.keys {
                if d >= n {
                    return bad(format!("operand {i} names dimension {d}, of {n}"));
                }
                if std::mem::replace(&mut seen[d], true) {
                    return bad(format!("operand {i} holds dimension {d} twice"));
                }
            }
        }
        for d in 0..n {
            if !operands.iter().any(|op| op.column_of(d).is_some()) {
                return bad(format!("no operand holds dimension {d}"));
            }
        }
        if operands.iter().all(|op| op.factor.is_none()) {
            return bad("no operand has a factor".into());
        }
        let mut seen = vec![false; n];
        for o in &outputs {
            if o.dimension >= n {
                return bad(format!("output dimension {}, of {n}", o.dimension));
            }
            if std::mem::replace(&mut seen[o.dimension], true) {
                return bad(format!("dimension {} is output twice", o.dimension));
            }
        }
        if let Some(domain) = &domain {
            if domain.keys.len() != outputs.len() {
                return bad(format!(
                    "the domain has {} key columns for {} outputs",
                    domain.keys.len(),
                    outputs.len()
                ));
            }
        }
        Ok(Fold {
            dimensions,
            operands,
            outputs,
            aggregate,
            value: value.into(),
            domain,
        })
    }

    /// The dimensions.
    pub fn dimensions(&self) -> &[Dimension] {
        &self.dimensions
    }

    /// The operands.
    pub fn operands(&self) -> &[Operand] {
        &self.operands
    }

    /// The output's key columns, in order.
    pub fn outputs(&self) -> &[Output] {
        &self.outputs
    }

    /// How joined rows combine into each group's value.
    pub fn aggregate(&self) -> Aggregate {
        self.aggregate
    }

    /// The name of the output's value column.
    pub fn value(&self) -> &str {
        &self.value
    }

    /// The domain, if the output is left-joined from one (see [`Domain`]).
    pub fn domain(&self) -> Option<&Domain> {
        self.domain.as_ref()
    }

    /// The position in [`Fold::outputs`] of dimension `dim`, if it is an
    /// output.
    pub fn output_of(&self, dim: usize) -> Option<usize> {
        self.outputs.iter().position(|o| o.dimension == dim)
    }

    /// Whether dimension `dim` is aggregated away: not an output.
    pub fn is_aggregated(&self, dim: usize) -> bool {
        self.output_of(dim).is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EQ: Dimension = Dimension {
        equality: KeyEquality::Equal,
    };

    fn out(dimension: usize, name: &str) -> Output {
        Output {
            dimension,
            name: name.into(),
        }
    }

    fn op(keys: Vec<(usize, usize)>, factor: Option<usize>) -> Operand {
        Operand { keys, factor }
    }

    /// `n,d * d,h -> outputs`.
    fn matmul(outputs: Vec<Output>, domain: Option<Domain>) -> Result<Fold, InvalidFold> {
        Fold::try_new(
            vec![EQ; 3],
            vec![
                op(vec![(0, 0), (1, 1)], Some(2)),
                op(vec![(1, 0), (2, 1)], Some(2)),
            ],
            outputs,
            Aggregate::SumOfProducts,
            "v",
            domain,
        )
    }

    #[test]
    fn matrix_product() {
        let f = matmul(vec![out(0, "n"), out(2, "h")], None).unwrap();
        assert!(f.is_aggregated(1));
        assert_eq!(f.output_of(2), Some(1));
        assert_eq!(f.operands()[1].column_of(2), Some(1));
        assert_eq!(f.operands()[0].column_of(2), None);
    }

    #[test]
    fn domains_compare_missing_by_bits() {
        let d = |missing| Domain {
            keys: vec![(0, KeyEquality::NotDistinct)],
            carry: vec![0, 1],
            missing,
        };
        assert_eq!(d(Some(f64::NAN)), d(Some(f64::NAN)));
        assert_ne!(d(Some(0.0)), d(Some(-0.0)));
        assert_ne!(d(Some(0.0)), d(None));
    }

    #[test]
    fn rejects_inconsistent_descriptions() {
        assert!(matmul(vec![out(0, "a"), out(0, "b")], None).is_err());
        assert!(matmul(vec![out(3, "x")], None).is_err());
        let domain = Domain {
            keys: vec![(0, KeyEquality::NotDistinct)],
            carry: vec![0],
            missing: Some(0.0),
        };
        assert!(matmul(vec![out(0, "n"), out(2, "h")], Some(domain)).is_err());
        let fold = |ops: Vec<Operand>, dims: usize| {
            Fold::try_new(
                vec![EQ; dims],
                ops,
                vec![],
                Aggregate::SumOfProducts,
                "v",
                None,
            )
        };
        assert!(fold(vec![op(vec![(0, 0), (0, 1)], Some(2))], 1).is_err());
        assert!(fold(vec![op(vec![(0, 0)], Some(1))], 2).is_err());
        assert!(fold(vec![op(vec![(0, 0)], None)], 1).is_err());
        assert!(fold(vec![op(vec![(0, 0)], None), op(vec![(0, 0)], Some(1))], 1).is_ok());
    }
}
