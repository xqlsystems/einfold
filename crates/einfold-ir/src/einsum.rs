// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! An einsum: a sum of products over named operands.
//!
//! "Einsum" is short for Einstein summation. Each input names its axes, and
//! every axis that does not appear in the output is summed over: the matrix
//! product `C[i,j] = Σ_k A[i,k]·B[k,j]` is written `ik,kj->ij`. In SQL, each
//! input is a table of `(dimension columns…, value)` rows, shared axes are join
//! keys, and output axes are `GROUP BY` keys. This module calls axes
//! *dimensions*.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

/// A dimension of an einsum.
///
/// In a query, a dimension is a group of columns that the query equates, by
/// join conditions such as `a.k = b.k`, possibly across several tables.
/// Operands that share a `Dim` are joined on it.
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

/// The pair of operations an einsum uses for "add" and "multiply".
///
/// Only the ordinary pair, `SUM` of `*`, exists today. Others (such as `MIN`
/// of `+`, for shortest paths) may be added later.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Semiring {
    /// `SUM` of products: ordinary contraction.
    SumProduct,
}

/// One input of an einsum: a table of values over some dimensions.
///
/// `dims` lists the operand's dimension columns in order. A dimension may
/// repeat, which is a diagonal: the plan equated two columns of one table
/// (`ii->i`). The operand's value is implicit: each operand contributes one
/// factor to every product.
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

/// Why an [`Einsum`] could not be built.
///
/// More reasons may be added as the representation grows.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EinsumError {
    /// The einsum has no operands.
    NoOperands,
    /// An output dimension does not appear in any operand.
    OutputDimNotInOperands(Dim),
    /// An output dimension is listed twice.
    DuplicateOutputDim(Dim),
    /// A dimension has no recorded [`KeyEquality`].
    MissingEquality(Dim),
}

impl fmt::Display for EinsumError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EinsumError::NoOperands => write!(f, "an einsum needs at least one operand"),
            EinsumError::OutputDimNotInOperands(d) => {
                write!(f, "output dimension `{d}` appears in no operand")
            }
            EinsumError::DuplicateOutputDim(d) => {
                write!(f, "output dimension `{d}` is listed twice")
            }
            EinsumError::MissingEquality(d) => {
                write!(
                    f,
                    "dimension `{d}` has no key equality (`=` or `IS NOT DISTINCT FROM`)"
                )
            }
        }
    }
}

impl std::error::Error for EinsumError {}

/// An einsum: `Σ` over the summed dimensions of the product of the operands,
/// grouped by the output dimensions.
///
/// In SQL terms: `SELECT <output>, SUM(<product of values>) FROM <operands>
/// WHERE <shared dimensions equal> GROUP BY <output>`. Its result has one
/// row per output coordinate tuple that some joined row reached: a group no
/// row reaches does not exist, as with any SQL `GROUP BY`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Einsum {
    operands: Vec<Operand>,
    output: Vec<Dim>,
    equality: BTreeMap<Dim, KeyEquality>,
    semiring: Semiring,
}

impl Einsum {
    /// Build an einsum, checking that it is well formed.
    ///
    /// `equality` must give the [`KeyEquality`] of every dimension that
    /// appears in an operand.
    pub fn new(
        operands: Vec<Operand>,
        output: Vec<Dim>,
        equality: BTreeMap<Dim, KeyEquality>,
        semiring: Semiring,
    ) -> Result<Self, EinsumError> {
        if operands.is_empty() {
            return Err(EinsumError::NoOperands);
        }
        let all: BTreeSet<Dim> = operands
            .iter()
            .flat_map(|o| o.dims.iter().cloned())
            .collect();
        let mut seen = BTreeSet::new();
        for d in &output {
            if !all.contains(d) {
                return Err(EinsumError::OutputDimNotInOperands(d.clone()));
            }
            if !seen.insert(d.clone()) {
                return Err(EinsumError::DuplicateOutputDim(d.clone()));
            }
        }
        if let Some(d) = all.iter().find(|d| !equality.contains_key(*d)) {
            return Err(EinsumError::MissingEquality(d.clone()));
        }
        Ok(Einsum {
            operands,
            output,
            equality,
            semiring,
        })
    }

    /// The operands, in plan order.
    pub fn operands(&self) -> &[Operand] {
        &self.operands
    }

    /// The output dimensions `O`, in the order of the result's columns.
    pub fn output(&self) -> &[Dim] {
        &self.output
    }

    /// The semiring.
    pub fn semiring(&self) -> Semiring {
        self.semiring
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

    /// Dimensions summed over: those not in the output.
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

impl fmt::Display for Einsum {
    /// Einsum notation with named dimensions, such as `A[i,k] · B[k,j] -> [i,j]`.
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
        write!(f, "{} -> [{}]", ops.join(" · "), join(&self.output))
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

    fn matmul() -> Einsum {
        Einsum::new(
            vec![
                Operand::new("A", [d("i"), d("k")]),
                Operand::new("B", [d("k"), d("j")]),
            ],
            vec![d("i"), d("j")],
            eq_all(&["i", "j", "k"]),
            Semiring::SumProduct,
        )
        .unwrap()
    }

    #[test]
    fn matmul_roles() {
        let e = matmul();
        assert_eq!(e.summed(), [d("k")].into());
        assert_eq!(e.shared(), [d("k")].into());
        assert!(e.private(0).is_empty());
        assert_eq!(e.to_string(), "A[i,k] · B[k,j] -> [i,j]");
    }

    #[test]
    fn private_dimensions() {
        // A[i,j,x] · B[j] -> [i]: j is shared; x is summed and only in A, so private to A.
        let e = Einsum::new(
            vec![
                Operand::new("A", [d("i"), d("j"), d("x")]),
                Operand::new("B", [d("j")]),
            ],
            vec![d("i")],
            eq_all(&["i", "j", "x"]),
            Semiring::SumProduct,
        )
        .unwrap();
        assert_eq!(e.private(0), [d("x")].into());
        assert!(e.private(1).is_empty());
        assert_eq!(e.summed(), [d("j"), d("x")].into());
    }

    #[test]
    fn rejects_malformed() {
        let a = || vec![Operand::new("A", [d("i")])];
        assert_eq!(
            Einsum::new(vec![], vec![], BTreeMap::new(), Semiring::SumProduct),
            Err(EinsumError::NoOperands)
        );
        assert_eq!(
            Einsum::new(a(), vec![d("z")], eq_all(&["i"]), Semiring::SumProduct),
            Err(EinsumError::OutputDimNotInOperands(d("z")))
        );
        assert_eq!(
            Einsum::new(
                a(),
                vec![d("i"), d("i")],
                eq_all(&["i"]),
                Semiring::SumProduct
            ),
            Err(EinsumError::DuplicateOutputDim(d("i")))
        );
        assert_eq!(
            Einsum::new(a(), vec![d("i")], BTreeMap::new(), Semiring::SumProduct),
            Err(EinsumError::MissingEquality(d("i")))
        );
    }

    #[test]
    fn diagonal_is_allowed() {
        let e = Einsum::new(
            vec![Operand::new("A", [d("i"), d("i")])],
            vec![d("i")],
            eq_all(&["i"]),
            Semiring::SumProduct,
        )
        .unwrap();
        assert!(e.summed().is_empty());
        assert!(e.shared().is_empty());
    }
}
