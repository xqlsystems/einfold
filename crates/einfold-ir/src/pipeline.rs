// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A fold pipeline: a fold (the *producer*) whose results another fold (the
//! *consumer*) reads through an expression.
//!
//! Hosts compute the producer in full before the consumer reads it. When
//! every producer group is a consumer group paired with values of
//! dimensions the consumer sums away, the two can instead be computed
//! together, one tile of producer groups at a time, each folded into the
//! consumer's state as soon as it is complete and then dropped, so the
//! producer's results never exist in full. That condition is all fusion
//! needs; nothing in it is specific to one query.
//!
//! The consumers this module describes are weighted averages whose weights
//! are exponentials of the producer's values, a softmax: each producer
//! group's value `p` becomes a *score* (`p`, `p * c` or `p / c`), the score
//! a weight `exp(score)`, optionally shifted, and the consumer averages a
//! third relation's values (the *values* operand) with those weights, over
//! the producer's outputs that it sums away:
//!
//! ```sql
//! WITH p AS (SELECT <kept>, <summed>, <fold> AS s FROM … GROUP BY <kept>, <summed>),
//! w AS (SELECT <kept>, <summed>, exp(s) AS w FROM p)
//! SELECT <kept>, <the values' own keys>, SUM(w.w * v.y) / SUM(w.w)
//! FROM w JOIN values v ON w.<summed> = v.<summed>
//! GROUP BY <kept>, <the values' own keys>
//! ```
//!
//! Many queries have this shape:
//!
//! - attention, where the producer is `SUM(q.x * k.x)` over a feature, the
//!   kept output a query, and the summed output a key;
//! - a kernel smoother, such as the Nadaraya–Watson estimator used in
//!   regridding (resampling values from one set of points onto another),
//!   where the score is `-dist²/h²` between a target and a
//!   source point, and the average is of the sources' values. Its producer
//!   computes `-dist²` as a row expression of the two points (see
//!   [`RowValue::Expression`]), over a `CROSS JOIN` with nothing to sum but
//!   the pair's one row;
//! - a softmax over a recommender's scores, averaging items' features.
//!
//! **Dimensions.** A pipeline's dimensions are its producer's, then its
//! own: dimensions that only the values operand holds, such as a feature of
//! the values. Each of the producer's outputs is either *kept*, an output of
//! the pipeline too, or *summed* by the consumer. The values operand is
//! joined with the producer's groups on the producer's outputs it holds,
//! compared as each dimension's [`KeyEquality`] says.
//!
//! [`KeyEquality`]: crate::KeyEquality

use std::cmp::Ordering;
use std::fmt;
use std::hash::{Hash, Hasher};

use crate::expression::Expression;
use crate::fold::{ColumnIndex, Dimension, DimensionId, Fold, InvalidFold, Operand, RowValue};

/// How a producer group's value `p` becomes its score.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Scale {
    /// The score is `p`.
    One,
    /// The score is `p * c`, rounded once.
    Times(f64),
    /// The score is `p / c`, rounded once.
    Over(f64),
}

impl Scale {
    /// The score of producer value `p`.
    #[must_use]
    pub fn apply(self, p: f64) -> f64 {
        match self {
            Scale::One => p,
            Scale::Times(c) => p * c,
            Scale::Over(c) => p / c,
        }
    }

    /// Everything that identifies the scale, with its constant as bits, so
    /// that equality, hashing and ordering are total (a NaN equals itself).
    fn identity(self) -> (u8, u64) {
        match self {
            Scale::One => (0, 0),
            Scale::Times(c) => (1, c.to_bits()),
            Scale::Over(c) => (2, c.to_bits()),
        }
    }
}

impl PartialEq for Scale {
    fn eq(&self, other: &Self) -> bool {
        self.identity() == other.identity()
    }
}

impl Eq for Scale {}

impl Hash for Scale {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.identity().hash(state)
    }
}

impl PartialOrd for Scale {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Scale {
    fn cmp(&self, other: &Self) -> Ordering {
        self.identity().cmp(&other.identity())
    }
}

/// What is subtracted from a score before `exp`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Shift {
    /// Nothing: the weight is `exp(score)`, which overflows to infinity for
    /// scores above about 709, as SQL's does.
    None,
    /// The largest score among the kept outputs' groups, in SQL `exp(s -
    /// MAX(s) OVER (PARTITION BY <kept>))`, or `MAX(s) … GROUP BY <kept>`
    /// joined back. It keeps every weight at most 1, and cancels between the
    /// average's numerator and its normalizer while the unshifted weights
    /// and their sums are finite and normal, changing the result only by
    /// rounding there. Outside that range they differ: where `exp(score)`
    /// overflows, the unshifted average is NaN (`inf / inf`) and the shifted
    /// one finite. So the two shifts aren't interchangeable.
    KeptMaximum,
}

/// What the consumer's weighted sum is divided by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Normalizer {
    /// The sum of the weights of every producer group with the output's kept
    /// keys, whether or not it joined a row of the values operand: a `norm`
    /// relation of `SUM(w) … GROUP BY <kept>`, joined on the kept outputs.
    KeptGroups,
    /// The sum of the weights of the joined rows of the output's own group:
    /// `SUM(w * y) / SUM(w)` in one aggregate.
    JoinedRows,
    /// Nothing: the consumer is the weighted sum `SUM(w * y)` itself, as a
    /// chain of two einsums such as `(q · k) · v` is.
    None,
}

/// How each producer group's value `p` becomes its weight.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum Weight {
    /// A softmax's: `exp(score - shift)`, with the score `p` scaled.
    Exp {
        /// How `p` becomes the score.
        scale: Scale,
        /// What is subtracted from the score before `exp`.
        shift: Shift,
    },
    /// Any row expression of `p`, which it reads as column 0 of operand 0
    /// (see [`Expression`]): `p` itself, for a chain of einsums;
    /// `1 / (1 + exp(-p))`, for sigmoid attention; or a Cauchy kernel's `1 /
    /// (1 + p)` of a squared distance `p`.
    Of(Expression),
}

/// A fold pipeline whose consumer is a weighted sum of the values, its
/// weights a function of the producer's values, divided by a normalizer or
/// not (see the [module](self)'s documentation). Its result's columns are
/// the outputs' keys in order, then the sum.
///
/// In SQL, with `kept` and `summed` the producer's outputs that are and aren't
/// outputs of the pipeline:
///
/// ```sql
/// WITH p AS (<producer>),
/// w AS (SELECT p.*, <weight>(p.value) AS w FROM p)
/// SELECT <outputs>, SUM(w.w * v.<factor>) [/ <normalizer>]
/// FROM w JOIN values v ON w.<summed and kept outputs v holds> = v.<their columns>
/// GROUP BY <outputs>
/// ```
///
/// An output group exists when some producer group joins a row of the values
/// operand. NULLs behave as in that SQL: the weighted sum skips a row whose
/// weight or value is NULL, as `SUM` does, while each [`Normalizer`] is the
/// sum of weights it names, so it still counts the weight of a joined row
/// whose value is NULL. Weights 1 and 1 of values 10 and NULL average to 5.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Pipeline {
    producer: Fold,
    weight: Weight,
    dimensions: Vec<Dimension>,
    values: Operand,
    outputs: Vec<DimensionId>,
    normalizer: Normalizer,
}

impl Pipeline {
    /// A pipeline, if its description is consistent:
    ///
    /// - the producer is a `SUM`, of products or of a row expression, with no
    ///   domain;
    /// - a weight of an expression reads only the producer's value;
    /// - the values operand has a factor, and its keys name the producer's
    ///   outputs or the pipeline's own `dimensions`, each at most once;
    /// - it holds every one of the pipeline's own dimensions, and every
    ///   producer output that the pipeline doesn't output;
    /// - the outputs are distinct, and each is a producer output or one of
    ///   the pipeline's own dimensions.
    pub fn try_new(
        producer: Fold,
        weight: Weight,
        dimensions: Vec<Dimension>,
        values: Operand,
        outputs: Vec<DimensionId>,
        normalizer: Normalizer,
    ) -> Result<Self, InvalidFold> {
        let bad = |why: String| Err(InvalidFold(why));
        let summable = matches!(producer.row(), RowValue::Product | RowValue::Expression(_));
        if !summable || !producer.aggregate().is_sum() || producer.domain().is_some() {
            return bad(
                "a pipeline's producer is a SUM of products or of a row expression, with no domain"
                    .into(),
            );
        }
        if let Weight::Of(e) = &weight {
            if e.columns().iter().any(|&c| c != (0, ColumnIndex(0))) {
                return bad("a pipeline's weight reads only the producer's value".into());
            }
        }
        if values.factor.is_none() {
            return bad("a pipeline's values operand has no factor".into());
        }
        let own = producer.dimensions().len();
        let n = own + dimensions.len();
        let produced = |d: DimensionId| producer.output_of(d).is_some();
        let mut held = vec![false; n];
        for key in &values.keys {
            let DimensionId(d) = key.dimension;
            if d >= n || (d < own && !produced(key.dimension)) {
                return bad(format!(
                    "the values operand holds dimension {d}, not an output"
                ));
            }
            if std::mem::replace(&mut held[d], true) {
                return bad(format!("the values operand holds dimension {d} twice"));
            }
        }
        let mut output = vec![false; n];
        for &DimensionId(d) in &outputs {
            if d >= n || (d < own && !produced(DimensionId(d))) {
                return bad(format!("output dimension {d} is not a producer output"));
            }
            if std::mem::replace(&mut output[d], true) {
                return bad(format!("dimension {d} is output twice"));
            }
        }
        let summed =
            (producer.outputs().iter()).find(|o| !output[o.dimension.0] && !held[o.dimension.0]);
        if let Some(o) = summed {
            let d = o.dimension.0;
            return bad(format!("producer output {d} is summed but not joined"));
        }
        if let Some(d) = (own..n).find(|&d| !held[d]) {
            return bad(format!("the values operand doesn't hold dimension {d}"));
        }
        Ok(Pipeline {
            producer,
            weight,
            dimensions,
            values,
            outputs,
            normalizer,
        })
    }

    /// The producer fold.
    #[must_use]
    pub fn producer(&self) -> &Fold {
        &self.producer
    }

    /// How a producer value becomes its weight.
    #[must_use]
    pub fn weight(&self) -> &Weight {
        &self.weight
    }

    /// The pipeline's own dimensions, numbered after the producer's.
    #[must_use]
    pub fn dimensions(&self) -> &[Dimension] {
        &self.dimensions
    }

    /// The values operand.
    #[must_use]
    pub fn values(&self) -> &Operand {
        &self.values
    }

    /// The dimensions whose keys are the result's columns, in order.
    #[must_use]
    pub fn outputs(&self) -> &[DimensionId] {
        &self.outputs
    }

    /// What the weighted sum is divided by, if anything.
    #[must_use]
    pub fn normalizer(&self) -> Normalizer {
        self.normalizer
    }

    /// Whether `d` is one of the producer's outputs that the consumer sums
    /// away.
    #[must_use]
    pub fn is_summed(&self, d: DimensionId) -> bool {
        self.producer.output_of(d).is_some() && !self.outputs.contains(&d)
    }
}

impl fmt::Display for Scale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scale::One => Ok(()),
            Scale::Times(c) => write!(f, " * {c}"),
            Scale::Over(c) => write!(f, " / {c}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fold::{Aggregate, ColumnIndex, Key, KeyEquality, Output, RowValue};

    const EQ: Dimension = Dimension {
        equality: KeyEquality::Equal,
    };

    fn op(keys: &[(usize, usize)], factor: Option<usize>) -> Operand {
        Operand {
            keys: (keys.iter())
                .map(|&(d, c)| Key::new(DimensionId(d), ColumnIndex(c)))
                .collect(),
            factor: factor.map(ColumnIndex),
        }
    }

    /// Attention's scores: `i,e * j,e -> i,j` (dimensions 0, 1, 2 = i, e, j).
    fn scores() -> Fold {
        let output = |d| Output {
            dimension: DimensionId(d),
            domain_key: None,
        };
        let ops = vec![
            op(&[(0, 0), (1, 1)], Some(2)),
            op(&[(2, 0), (1, 1)], Some(2)),
        ];
        let outputs = vec![output(0), output(2)];
        Fold::try_new(
            vec![EQ; 3],
            ops,
            outputs,
            RowValue::Product,
            Aggregate::sum(),
            None,
        )
        .unwrap()
    }

    fn attention(values: Operand, outputs: &[usize]) -> Result<Pipeline, InvalidFold> {
        Pipeline::try_new(
            scores(),
            Weight::Exp {
                scale: Scale::Over(8.0),
                shift: Shift::KeptMaximum,
            },
            vec![EQ],
            values,
            outputs.iter().map(|&d| DimensionId(d)).collect(),
            Normalizer::KeptGroups,
        )
    }

    #[test]
    fn attention_keeps_i_and_sums_j() {
        // v(j, d, x), with d the pipeline's own dimension 3.
        let p = attention(op(&[(2, 0), (3, 1)], Some(2)), &[0, 3]).unwrap();
        assert!(p.is_summed(DimensionId(2)));
        assert!(!p.is_summed(DimensionId(0)) && !p.is_summed(DimensionId(1)));
        let Weight::Exp { scale, .. } = p.weight() else {
            panic!("a softmax");
        };
        assert_eq!(scale.apply(4.0), 0.5);
        assert_eq!(Scale::Times(f64::NAN), Scale::Times(f64::NAN));
        assert_ne!(Scale::Times(2.0), Scale::Over(2.0));
    }

    #[test]
    fn rejects_inconsistent_descriptions() {
        let v = || op(&[(2, 0), (3, 1)], Some(2));
        // A values operand without a factor, or holding the producer's
        // summed dimension e.
        assert!(attention(op(&[(2, 0), (3, 1)], None), &[0, 3]).is_err());
        assert!(attention(op(&[(1, 0), (2, 1), (3, 2)], Some(3)), &[0, 3]).is_err());
        // j neither output nor joined; d not held; outputs repeated or out of
        // range.
        assert!(attention(op(&[(3, 1)], Some(2)), &[0, 3]).is_err());
        assert!(attention(op(&[(2, 0)], Some(2)), &[0, 3]).is_err());
        assert!(attention(v(), &[0, 0, 3]).is_err());
        assert!(attention(v(), &[0, 4]).is_err());
        assert!(attention(v(), &[1, 3]).is_err());
        assert!(attention(v(), &[0, 3]).is_ok());
    }

    #[test]
    fn a_weight_of_an_expression_reads_only_the_producers_value() {
        let with = |weight| {
            Pipeline::try_new(
                scores(),
                weight,
                vec![EQ],
                op(&[(2, 0), (3, 1)], Some(2)),
                vec![DimensionId(0), DimensionId(3)],
                Normalizer::None,
            )
        };
        let p = Expression::column(0, 0);
        let sigmoid = Expression::literal(1.0).binary(
            crate::Binary::Divide,
            Expression::literal(1.0).binary(
                crate::Binary::Add,
                p.unary(crate::Unary::Negate).unary(crate::Unary::Exp),
            ),
        );
        assert!(with(Weight::Of(sigmoid)).is_ok());
        assert!(with(Weight::Of(Expression::column(0, 1))).is_err());
        assert!(with(Weight::Of(Expression::column(1, 0))).is_err());
    }
}
