// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`multiply_add`]: matrix products for tiles that fit in cache, by
//! [`matrixmultiply`](https://docs.rs/matrixmultiply), a Rust crate of dense
//! matrix products, through the safe interface of
//! [`ndarray`](https://docs.rs/ndarray), Rust's n-dimensional arrays. It
//! picks kernels using AVX2 and FMA, x86's 256-bit vector and fused
//! multiply-add instructions, at run time when the CPU has them.

/// A matrix in a slice, row-major, with `stride` elements from one row's
/// start to the next.
#[derive(Clone, Copy)]
pub(crate) struct Matrix<'a> {
    pub(crate) data: &'a [f64],
    pub(crate) stride: usize,
}

/// `c[m×n] += a[m×k] · b[k×n]`, with `c` row-major at `c_stride`.
///
/// Each cell is within float summation's bound of the exact sum. The order
/// of its additions is the library's, fixed for a given shape on a given
/// machine, so the same call gives the same bits.
pub(crate) fn multiply_add(
    (m, k, n): (usize, usize, usize),
    a: Matrix,
    b: Matrix,
    c: &mut [f64],
    c_stride: usize,
) {
    use ndarray::{linalg, ArrayView2, ArrayViewMut2, ShapeBuilder};
    // With no terms to add, or no cells to add them to, nothing changes.
    if m == 0 || n == 0 || k == 0 {
        return;
    }
    fn view(data: &[f64], rows: usize, cols: usize, stride: usize) -> ArrayView2<'_, f64> {
        let len = (rows - 1) * stride + cols;
        ArrayView2::from_shape((rows, cols).strides((stride, 1)), &data[..len]).expect("a matrix")
    }
    let (a, b) = (view(a.data, m, k, a.stride), view(b.data, k, n, b.stride));
    let len = (m - 1) * c_stride + n;
    let mut c =
        ArrayViewMut2::from_shape((m, n).strides((c_stride, 1)), &mut c[..len]).expect("a matrix");
    linalg::general_mat_mul(1.0, &a, &b, 1.0, &mut c);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn products_within_the_summation_bound_and_repeatable() {
        // Sizes on both sides of any register block, inside larger matrices.
        let (m, k, n) = (9, 5, 19);
        let value = |i: usize| ((i as f64) * 0.37).sin() * 1e3;
        let a: Vec<f64> = (0..m * 7).map(value).collect();
        let b: Vec<f64> = (0..k * 23).map(|i| value(i + 101)).collect();
        let c0: Vec<f64> = (0..m * 21).map(|i| value(i + 7)).collect();
        let mut c = c0.clone();
        let (am, bm) = (
            Matrix {
                data: &a,
                stride: 7,
            },
            Matrix {
                data: &b,
                stride: 23,
            },
        );
        multiply_add((m, k, n), am, bm, &mut c, 21);
        let u = f64::EPSILON / 2.0;
        for cell in 0..m * 21 {
            let (i, j) = (cell / 21, cell % 21);
            if j >= n {
                assert_eq!(c[cell].to_bits(), c0[cell].to_bits(), "outside the product");
                continue;
            }
            let terms: Vec<f64> = (0..k).map(|p| a[i * 7 + p] * b[p * 23 + j]).collect();
            let exact = c0[cell] + terms.iter().sum::<f64>();
            let magnitude = c0[cell].abs() + terms.iter().map(|t| t.abs()).sum::<f64>();
            assert!((c[cell] - exact).abs() <= 2.0 * (k as f64 + 1.0) * u * magnitude);
        }
        // The same call gives the same bits.
        let mut again = c0.clone();
        multiply_add((m, k, n), am, bm, &mut again, 21);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&c), bits(&again));
    }
}
