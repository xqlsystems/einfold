// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! einfold's engine-independent representations.
//!
//! einfold speeds up tensor contractions written in SQL. In SQL, a tensor is a
//! table with one row per coordinate tuple: key columns for its dimensions
//! (such as `i` and `k`) and a value column. A contraction such as the matrix
//! product `C[i,j] = Σ_k A[i,k]·B[k,j]` is then
//!
//! ```sql
//! SELECT a.i, b.j, SUM(a.v * b.v) FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j
//! ```
//!
//! This crate describes such queries without depending on any query engine,
//! so every engine integration can share it:
//!
//! - [`einsum`]: the contraction itself, as an [`Einsum`] over [`Operand`]s;
//! - [`partial`]: [`PartialSum`], the per-group state of a `SUM` that is
//!   computed in pieces and combined;
//! - [`facts`]: what einfold knows about its inputs, and how surely.

pub mod einsum;
pub mod facts;
pub mod partial;

pub use einsum::{Dim, Einsum, EinsumError, KeyEquality, Operand, Semiring};
pub use facts::{Fact, Precision};
pub use partial::PartialSum;
