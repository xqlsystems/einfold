// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`PythonBackend`]: a session's Python function as einfold's
//! [`DenseBackend`], the host's arithmetic for dense matrix products and
//! attention. Experimental and temporary (see [`DenseBackend`]).

use datafusion::arrow::array::{make_array, Array, ArrayData, Float64Array};
use datafusion::arrow::error::ArrowError;
use datafusion::arrow::pyarrow::{FromPyArrow, ToPyArrow};
use einfold_ir::Scale;
use einfold_kernels::{AttentionShape, DenseBackend, ProductShape};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// A Python function `backend(kind, arrays, shape, scale, ranges)`, called
/// with einfold's checked operands as `pyarrow` `float64` arrays, row-major,
/// and returning the result as one, or `None` to decline:
///
/// - `kind == "matmul"`: `arrays` is `[a, b]`, `shape` is `(batches, rows,
///   inner, columns)`, and the result is `a · b` per batch;
/// - `kind == "attention"`: `arrays` is `[q, kt, v]`, `shape` is `(batches,
///   queries, features, keys, values)`, `scale` is `(0, 1.0)` for none,
///   `(1, c)` for `· c` or `(2, c)` for `/ c`, and `ranges` holds each
///   query's half-open range of keys, the same in every batch.
///
/// Work smaller than `min_ops` multiplications, or whose operands and result
/// take more than `max_bytes`, isn't offered.
#[derive(Debug)]
pub(crate) struct PythonBackend {
    pub(crate) callback: Py<PyAny>,
    pub(crate) min_ops: usize,
    pub(crate) max_bytes: usize,
}

impl PythonBackend {
    /// Calls the function with `arrays`, if the work is worth offering, and
    /// returns its result, checked to be `float64` with no NULLs.
    fn call(
        &self,
        kind: &str,
        arrays: &[&[f64]],
        shape: Vec<usize>,
        (ops, output): (usize, usize),
        scale: (u8, f64),
        ranges: &[(usize, usize)],
    ) -> Result<Option<Vec<f64>>, ArrowError> {
        let cells = arrays.iter().map(|a| a.len()).sum::<usize>();
        let bytes = cells
            .saturating_add(output)
            .saturating_mul(std::mem::size_of::<f64>());
        if ops < self.min_ops || bytes > self.max_bytes {
            return Ok(None);
        }
        Python::attach(|py| -> PyResult<Option<Vec<f64>>> {
            let arrays: Vec<_> = (arrays.iter())
                .map(|a| Float64Array::from(a.to_vec()).to_data().to_pyarrow(py))
                .collect::<PyResult<_>>()?;
            let result = self
                .callback
                .call1(py, (kind, arrays, shape, scale, ranges.to_vec()))?;
            if result.is_none(py) {
                return Ok(None);
            }
            let result = make_array(ArrayData::from_pyarrow_bound(result.bind(py))?);
            let result = (result.as_any().downcast_ref::<Float64Array>())
                .ok_or_else(|| PyValueError::new_err("a dense backend must return float64"))?;
            if result.null_count() > 0 {
                return Err(PyValueError::new_err("a dense backend returned NULLs"));
            }
            Ok(Some(result.values().to_vec()))
        })
        .map_err(|e| ArrowError::ExternalError(Box::new(e)))
    }
}

impl DenseBackend for PythonBackend {
    fn matmul(
        &self,
        a: &[f64],
        b: &[f64],
        shape: ProductShape,
    ) -> Result<Option<Vec<f64>>, ArrowError> {
        let ProductShape {
            batches,
            rows,
            inner,
            columns,
        } = shape;
        let output = batches.saturating_mul(rows).saturating_mul(columns);
        let ops = output.saturating_mul(inner);
        let shape = vec![batches, rows, inner, columns];
        self.call("matmul", &[a, b], shape, (ops, output), (0, 1.0), &[])
    }

    fn attention(
        &self,
        q: &[f64],
        kt: &[f64],
        v: &[f64],
        shape: AttentionShape,
        scale: Scale,
        allowed: &[(usize, usize)],
    ) -> Result<Option<Vec<f64>>, ArrowError> {
        let AttentionShape {
            batches,
            queries,
            features,
            keys,
            values,
        } = shape;
        let scale = match scale {
            Scale::One => (0, 1.0),
            Scale::Times(c) => (1, c),
            Scale::Over(c) => (2, c),
            _ => return Ok(None),
        };
        // Each pair's score, then its weight's share of each value.
        let pairs = batches.saturating_mul(queries).saturating_mul(keys);
        let ops = pairs.saturating_mul(features.saturating_add(values));
        let output = batches.saturating_mul(queries).saturating_mul(values);
        let shape = vec![batches, queries, features, keys, values];
        self.call(
            "attention",
            &[q, kt, v],
            shape,
            (ops, output),
            scale,
            allowed,
        )
    }
}
