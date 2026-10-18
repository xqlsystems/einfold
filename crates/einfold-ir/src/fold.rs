// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A fold over a join: an aggregate computed over the rows of a join,
//! described by its operands and dimensions rather than by SQL text.
//!
//! The fold this module describes is a sum of products. With operands `a` and
//! `b`, it means, in SQL:
//!
//! ```sql
//! SELECT <output dimensions>, SUM(a.value * b.value) AS value
//! FROM a JOIN b ON <a dimension's column in a> = <its column in b> AND …
//! GROUP BY <output dimensions>
//! ```
//!
//! A **dimension** is one variable of the fold: the set of columns, at most
//! one per operand, that the join equates. A dimension in both operands is a
//! join key. A dimension that is not an output is summed away; when it is in
//! one operand only, its values are simply not grouped by. Matrix
//! multiplication `C[n, h] = Σ_d X[n, d]·W[d, h]` is the fold with dimensions
//! `n`, `d` and `h`, operands `X(n, d)` and `W(d, h)`, and outputs `n` and `h`.
//!
//! A fold may also carry a **fill**: a third relation whose rows the output
//! has, one per fill row, as a gradient has one row per row of the table it
//! is the gradient of. See [`Fill`].

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
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Dimension {
    /// The name of the dimension's column in the fold's output, when it is an
    /// output.
    pub name: String,
    /// How the join compares the dimension's columns. It matters only for a
    /// dimension in both operands. Grouping, as SQL's `GROUP BY` does, always
    /// treats NULL as one more value.
    pub equality: KeyEquality,
}

/// One input of a fold: which of its columns hold which dimensions, and
/// which holds the factor it contributes to each product.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Operand {
    /// `(dimension, column)` pairs: the dimension's index in
    /// [`Fold::dimensions`], and the column of this operand that holds it.
    pub keys: Vec<(usize, usize)>,
    /// The column holding this operand's factor.
    pub value: usize,
}

impl Operand {
    /// The column of this operand that holds dimension `dim`, if any.
    pub fn column_of(&self, dim: usize) -> Option<usize> {
        self.keys.iter().find(|(d, _)| *d == dim).map(|(_, c)| *c)
    }
}

/// The relation whose rows a fold's output has.
///
/// With a fill, the fold's output has exactly one row per fill row, in SQL:
///
/// ```sql
/// SELECT <fill's key columns>,
///        CASE WHEN fill.value IS NULL THEN NULL
///             WHEN folded.value IS NULL THEN 0.0
///             ELSE folded.value END AS value
/// FROM fill LEFT JOIN folded
///   ON fill.<key of output 1> IS NOT DISTINCT FROM folded.<output 1> AND …
/// ```
///
/// where `folded` is the fold without its fill. So a fill row gets its
/// group's sum, 0 where no row reached the group or every product was NULL,
/// and NULL where the fill row's own value is NULL. That is how ddx, an XQL
/// Systems project for automatic differentiation of SQL queries, gives every
/// row of a table a gradient. Keys are matched as `GROUP BY` matches them,
/// NULL to NULL, whatever the dimension's [`KeyEquality`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Fill {
    /// The fill's column for each of the fold's outputs, in the order of
    /// [`Fold::outputs`].
    pub keys: Vec<usize>,
    /// The fill's value column: a fill row whose value is NULL gets NULL.
    pub value: usize,
}

/// A sum of products over a join of operands, grouped by output dimensions.
///
/// Build one with [`Fold::try_new`], which checks that the description is
/// consistent; holding a `Fold` is evidence that it is.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Fold {
    dimensions: Vec<Dimension>,
    operands: Vec<Operand>,
    outputs: Vec<usize>,
    value_name: String,
    fill: Option<Fill>,
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
    /// - every dimension and output index is in range;
    /// - an operand holds each dimension in at most one column (a dimension
    ///   repeated within an operand, a diagonal such as `ii->i`, is not
    ///   supported);
    /// - every dimension is held by some operand;
    /// - no output is listed twice;
    /// - a fill names one key column per output.
    ///
    /// `value_name` names the output's value column.
    pub fn try_new(
        dimensions: Vec<Dimension>,
        operands: Vec<Operand>,
        outputs: Vec<usize>,
        value_name: impl Into<String>,
        fill: Option<Fill>,
    ) -> Result<Self, InvalidFold> {
        let bad = |why: String| Err(InvalidFold(why));
        if operands.is_empty() {
            return bad("a fold needs at least one operand".into());
        }
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
        let mut seen = vec![false; n];
        for &o in &outputs {
            if o >= n {
                return bad(format!("output dimension {o}, of {n}"));
            }
            if std::mem::replace(&mut seen[o], true) {
                return bad(format!("dimension {o} is output twice"));
            }
        }
        if let Some(fill) = &fill {
            if fill.keys.len() != outputs.len() {
                return bad(format!(
                    "the fill has {} key columns for {} outputs",
                    fill.keys.len(),
                    outputs.len()
                ));
            }
        }
        Ok(Fold {
            dimensions,
            operands,
            outputs,
            value_name: value_name.into(),
            fill,
        })
    }

    /// The dimensions.
    pub fn dimensions(&self) -> &[Dimension] {
        &self.dimensions
    }

    /// The operands, whose factors are multiplied.
    pub fn operands(&self) -> &[Operand] {
        &self.operands
    }

    /// The output dimensions, as indices into [`Fold::dimensions`], in the
    /// order of the output's key columns.
    pub fn outputs(&self) -> &[usize] {
        &self.outputs
    }

    /// The name of the output's value column.
    pub fn value_name(&self) -> &str {
        &self.value_name
    }

    /// The fill, if the output's rows are a fill's (see [`Fill`]).
    pub fn fill(&self) -> Option<&Fill> {
        self.fill.as_ref()
    }

    /// Whether dimension `dim` is summed away: not an output.
    pub fn is_summed(&self, dim: usize) -> bool {
        !self.outputs.contains(&dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dim(name: &str) -> Dimension {
        Dimension {
            name: name.into(),
            equality: KeyEquality::Equal,
        }
    }

    fn matmul(outputs: Vec<usize>, fill: Option<Fill>) -> Result<Fold, InvalidFold> {
        Fold::try_new(
            vec![dim("n"), dim("d"), dim("h")],
            vec![
                Operand {
                    keys: vec![(0, 0), (1, 1)],
                    value: 2,
                },
                Operand {
                    keys: vec![(1, 0), (2, 1)],
                    value: 2,
                },
            ],
            outputs,
            "v",
            fill,
        )
    }

    #[test]
    fn matrix_product() {
        let f = matmul(vec![0, 2], None).unwrap();
        assert!(f.is_summed(1));
        assert!(!f.is_summed(0));
        assert_eq!(f.operands()[1].column_of(2), Some(1));
        assert_eq!(f.operands()[0].column_of(2), None);
    }

    #[test]
    fn rejects_inconsistent_descriptions() {
        assert!(matmul(vec![0, 0], None).is_err());
        assert!(matmul(vec![3], None).is_err());
        let fill = Fill {
            keys: vec![0],
            value: 2,
        };
        assert!(matmul(vec![0, 2], Some(fill)).is_err());
        let repeated = Operand {
            keys: vec![(0, 0), (0, 1)],
            value: 2,
        };
        assert!(Fold::try_new(vec![dim("i")], vec![repeated], vec![0], "v", None).is_err());
        let unheld = Operand {
            keys: vec![(0, 0)],
            value: 1,
        };
        assert!(Fold::try_new(vec![dim("i"), dim("j")], vec![unheld], vec![], "v", None).is_err());
    }
}
