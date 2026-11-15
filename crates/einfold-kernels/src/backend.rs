// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`DenseBackend`]: a host's own arithmetic for the dense work einfold's
//! kernels compute, offered one checked node at a time as [`DenseWork`].
//! Experimental and temporary: an interface that hands the runtime each
//! detected node, its checks and the plan it replaced, rather than its packed
//! operands, is planned to replace it.

use std::fmt;

use einfold_ir::Scale;

/// The shape of a batched matrix product, `A[w][x][z] · B[w][z][y]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProductShape {
    /// Independent products, `w`.
    pub batches: usize,
    /// Rows of each product's left operand and of its result, `x`.
    pub rows: usize,
    /// The summed dimension, `z`.
    pub inner: usize,
    /// Columns of each product's right operand and of its result, `y`.
    pub columns: usize,
}

/// The shape of batched attention: for each batch `w`, queries `q[w][x][e]`,
/// keys `k[w][e][j]` (transposed) and values `v[w][j][y]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionShape {
    /// Independent attention problems, such as a transformer's batches and
    /// heads, `w`.
    pub batches: usize,
    /// Query positions of each batch, `x`.
    pub queries: usize,
    /// Features each score sums over, `e`.
    pub features: usize,
    /// Key positions of each batch, `j`.
    pub keys: usize,
    /// Columns of the values and of the result, `y`.
    pub values: usize,
}

/// One node of dense work einfold has detected and checked, with its
/// operands packed row-major, batch first. Each kind is one of einfold's
/// nodes in a shape a host may have a kernel for. More kinds may come, such
/// as convolutions, so a backend declines a kind it doesn't know.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum DenseWork<'a> {
    /// `a · b` for each batch: a fold over a join whose row value is the
    /// product of its operands' values, summed. `a` is `[w][x][z]`, `b` is
    /// `[w][z][y]`, and the result is `[w][x][y]`.
    Product {
        /// The left operands.
        a: &'a [f64],
        /// The right operands.
        b: &'a [f64],
        /// Their shape.
        shape: ProductShape,
    },
    /// Softmax attention for each batch, a fold pipeline: row `x` of the
    /// result is `Σ_j softmax_j(scale(q[x] · k[j])) · v[j]` over the keys `j`
    /// in `allowed[x]`, a half-open range, the same in every batch. The
    /// softmax is shifted by each row's largest score, as SQL's `exp(s -
    /// MAX(s) OVER (PARTITION BY …))` is. `q` is `[w][x][e]`, `kt` is
    /// `[w][e][j]`, `v` is `[w][j][y]`, and the result is `[w][x][y]`, with
    /// every row computed, even one whose range is empty, which einfold
    /// drops.
    Attention {
        /// The queries.
        q: &'a [f64],
        /// The keys, transposed.
        kt: &'a [f64],
        /// The values.
        v: &'a [f64],
        /// Their shape.
        shape: AttentionShape,
        /// The constant each score is scaled by.
        scale: Scale,
        /// Each query's keys.
        allowed: &'a [(usize, usize)],
    },
}

/// A host's arithmetic for dense work einfold has already checked: every key
/// in range, no duplicate rows, no NULLs, and every factor finite. Which rows
/// exist, and which are NULL, stays einfold's.
///
/// Its results are the host's: einfold doesn't check their precision or
/// determinism, so a host that computes in `float32`, or adds in an order
/// that depends on timing, changes the query's answer by as much.
pub trait DenseBackend: Send + Sync + fmt::Debug {
    /// The result of `work`, row-major and batch first, or `None` to decline
    /// it, in which case einfold computes it itself. A backend declines a
    /// kind of work it doesn't know.
    ///
    /// # Errors
    ///
    /// Whatever error the host's arithmetic gives, which fails the query.
    fn run(&self, work: DenseWork<'_>) -> crate::Result<Option<Vec<f64>>>;
}
