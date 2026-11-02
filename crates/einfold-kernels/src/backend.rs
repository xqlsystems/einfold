// SPDX-FileCopyrightText: 2026 xarray-sql contributors
// SPDX-License-Identifier: Apache-2.0

//! Optional session-local execution of validated dense matrices.

use einfold_ir::Scale;
use std::fmt;

/// An executor of dense arithmetic, after SQL keys, duplicates and NULLs
/// have been checked. Returning None keeps the original CPU algorithm.
pub trait DenseBackend: Send + Sync + fmt::Debug {
    /// Batched A[w,x,z] times B[w,z,y], flattened in row-major order.
    fn matmul(
        &self,
        a: &[f64],
        b: &[f64],
        shape: (usize, usize, usize, usize),
    ) -> crate::Result<Option<Vec<f64>>>;

    /// Shifted, normalized attention on complete, finite matrices. Each
    /// query sees only the key interval in ranges; other pairs are absent.
    fn attention(
        &self,
        q: &[f64],
        kt: &[f64],
        v: &[f64],
        shape: (usize, usize, usize, usize),
        scale: Scale,
        ranges: &[(usize, usize)],
    ) -> crate::Result<Option<Vec<f64>>>;
}
