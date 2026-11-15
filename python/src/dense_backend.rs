// SPDX-FileCopyrightText: 2026 xarray-sql contributors
// SPDX-License-Identifier: Apache-2.0

//! Python callback for validated dense arithmetic, scoped to one session.
use datafusion::arrow::array::{make_array, Array, ArrayData, Float64Array};
use datafusion::arrow::error::ArrowError;
use datafusion::arrow::pyarrow::{FromPyArrow, ToPyArrow};
use einfold_ir::Scale;
use einfold_kernels::DenseBackend;
use pyo3::prelude::*;

#[derive(Debug)]
pub(crate) struct PythonBackend {
    pub(crate) callback: Py<PyAny>,
    pub(crate) min_ops: usize,
    pub(crate) max_bytes: usize,
}
impl PythonBackend {
    fn run(
        &self,
        kind: &str,
        arrays: &[&[f64]],
        shape: (usize, usize, usize, usize),
        scale: (u8, f64),
        ranges: &[(usize, usize)],
    ) -> Result<Option<Vec<f64>>, ArrowError> {
        let ops = if kind == "matmul" {
            shape
                .0
                .saturating_mul(shape.1)
                .saturating_mul(shape.2)
                .saturating_mul(shape.3)
        } else {
            shape
                .0
                .saturating_mul(shape.2)
                .saturating_mul(shape.1.saturating_add(shape.3))
        };
        let cells = arrays.iter().map(|a| a.len()).sum::<usize>();
        let output = if kind == "matmul" {
            shape.0.saturating_mul(shape.1).saturating_mul(shape.3)
        } else {
            shape.0.saturating_mul(shape.3)
        };
        if ops < self.min_ops || cells.saturating_add(output).saturating_mul(32) > self.max_bytes {
            return Ok(None);
        }
        Python::attach(|py| -> PyResult<Option<Vec<f64>>> {
            let arrays: Vec<_> = arrays
                .iter()
                .map(|a| Float64Array::from(a.to_vec()).to_data().to_pyarrow(py))
                .collect::<PyResult<_>>()?;
            let result = self
                .callback
                .call1(py, (kind, arrays, shape, scale, ranges.to_vec()))?;
            if result.is_none(py) {
                return Ok(None);
            }
            let result = make_array(ArrayData::from_pyarrow_bound(result.bind(py))?);
            let result = result
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("backend must return Float64")
                })?;
            if result.null_count() > 0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "backend returned NULLs",
                ));
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
        shape: (usize, usize, usize, usize),
    ) -> Result<Option<Vec<f64>>, ArrowError> {
        self.run("matmul", &[a, b], shape, (0, 1.0), &[])
    }
    fn attention(
        &self,
        q: &[f64],
        kt: &[f64],
        v: &[f64],
        shape: (usize, usize, usize, usize),
        scale: Scale,
        ranges: &[(usize, usize)],
    ) -> Result<Option<Vec<f64>>, ArrowError> {
        let scale = match scale {
            Scale::One => (0, 1.0),
            Scale::Times(c) => (1, c),
            Scale::Over(c) => (2, c),
            _ => return Ok(None),
        };
        self.run("attention", &[q, kt, v], shape, scale, ranges)
    }
}
