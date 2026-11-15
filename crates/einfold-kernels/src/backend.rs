// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`DenseBackend`]: a host's own arithmetic for the dense matrix products
//! and attention that einfold's kernels compute. Experimental and
//! temporary: an interface that hands the runtime each detected node, its
//! checks and the plan it replaced, rather than its packed operands, is
//! planned to replace it.

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

/// A host's arithmetic for dense work einfold has already checked: every key
/// in range, no duplicate rows, no NULLs, and every factor finite. Each
/// method returns the result, row-major, or `None` to decline, in which case
/// einfold computes it itself. An error fails the query.
///
/// Its results are the host's: einfold doesn't check their precision or
/// determinism, so a host that computes in `float32`, or adds in an order
/// that depends on timing, changes the query's answer by as much.
pub trait DenseBackend: Send + Sync + fmt::Debug {
    /// `a · b` for each batch: `a` is `[w][x][z]` and `b` is `[w][z][y]`,
    /// and the result is `[w][x][y]`.
    ///
    /// # Errors
    ///
    /// Whatever error the host's arithmetic gives.
    fn matmul(&self, a: &[f64], b: &[f64], shape: ProductShape) -> crate::Result<Option<Vec<f64>>>;

    /// Softmax attention for each batch: row `x` of the result is
    /// `Σ_j softmax_j(scale(q[x] · k[j])) · v[j]` over the keys `j` in
    /// `allowed[x]`, a half-open range, the same in every batch. The softmax
    /// is shifted by each row's largest score, as SQL's `exp(s - MAX(s) OVER
    /// (PARTITION BY …))` is. `q` is `[w][x][e]`, `kt` is `[w][e][j]`, `v`
    /// is `[w][j][y]`, and the result is `[w][x][y]`, with every row computed,
    /// even one whose range is empty, which einfold drops.
    ///
    /// # Errors
    ///
    /// Whatever error the host's arithmetic gives.
    fn attention(
        &self,
        q: &[f64],
        kt: &[f64],
        v: &[f64],
        shape: AttentionShape,
        scale: Scale,
        allowed: &[(usize, usize)],
    ) -> crate::Result<Option<Vec<f64>>>;
}
