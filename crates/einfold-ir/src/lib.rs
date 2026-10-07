// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! einfold's engine-independent representations.
//!
//! einfold speeds up queries such as tensor contractions written in SQL. In
//! SQL, a tensor is a table with one row per coordinate tuple: key columns for
//! its dimensions (such as `i` and `k`) and a value column. A contraction such as the matrix
//! product `C[i,j] = Σ_k A[i,k]·B[k,j]` is then
//!
//! ```sql
//! SELECT a.i, b.j, SUM(a.v * b.v) FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j
//! ```
//!
//! More generally, einfold speeds up any *fold over a join*: a join followed
//! by an aggregate such as `SUM`, `COUNT` or `AVG`, grouped by some columns.
//! The matrix product, an *einsum*, is the most important case.
//!
//! This crate describes such queries without depending on any query engine,
//! so every engine integration can share it:
//!
//! - [`fold`]: the query itself, as a [`Fold`] over [`Operand`]s;
//! - [`aggregate`]: the [`Aggregate`]s a fold can compute, with their SQL
//!   rules;
//! - [`partial`]: [`PartialAggregate`], the state of one group computed in
//!   pieces and combined later;
//! - [`facts`]: what einfold knows about its inputs, and how surely.

pub mod aggregate;
pub mod facts;
pub mod fold;
pub mod partial;

pub use aggregate::Aggregate;
pub use facts::{Fact, Precision};
pub use fold::{Dim, Fold, FoldError, KeyEquality, Operand, RowValue, Semiring};
pub use partial::{AggregateState, AggregateValue, PartialAggregate};
