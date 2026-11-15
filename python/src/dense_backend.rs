// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`PythonBackend`]: a session's Python function as einfold's
//! [`DenseBackend`], the host's arithmetic for the dense work einfold has
//! checked, called with each kind of work by name. Experimental and
//! temporary (see [`DenseBackend`]).

use datafusion::arrow::array::{make_array, Array, ArrayData, Float64Array};
use datafusion::arrow::error::ArrowError;
use datafusion::arrow::pyarrow::{FromPyArrow, ToPyArrow};
use einfold_ir::Scale;
use einfold_kernels::{AttentionShape, DenseBackend, DenseShape, DenseWork, ProductShape};
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
/// Later versions may offer other kinds, such as convolutions, so the
/// function returns `None` for a kind it doesn't know. Work smaller than
/// `min_ops` multiplications, or whose operands and result take more than
/// `max_bytes`, isn't offered, and einfold doesn't reserve memory for it or
/// copy it.
#[derive(Debug)]
pub(crate) struct PythonBackend {
    pub(crate) callback: Py<PyAny>,
    pub(crate) min_ops: usize,
    pub(crate) max_bytes: usize,
}

/// What a kind of work costs: its multiplications, the cells of its
/// operands, and the cells of its result.
struct Cost {
    ops: usize,
    operands: usize,
    output: usize,
}

impl Cost {
    /// The cost of work of `shape`, or `None` for a kind this function
    /// isn't told of.
    fn of(shape: DenseShape) -> Option<Cost> {
        let product = |sizes: &[usize]| sizes.iter().fold(1, |n: usize, &s| n.saturating_mul(s));
        match shape {
            DenseShape::Product(ProductShape {
                batches,
                rows,
                inner,
                columns,
            }) => Some(Cost {
                ops: product(&[batches, rows, inner, columns]),
                operands: product(&[batches, rows, inner])
                    .saturating_add(product(&[batches, inner, columns])),
                output: product(&[batches, rows, columns]),
            }),
            DenseShape::Attention(AttentionShape {
                batches,
                queries,
                features,
                keys,
                values,
            }) => Some(Cost {
                // Each pair's score, then its weight's share of each value.
                ops: product(&[batches, queries, keys, features.saturating_add(values)]),
                operands: product(&[batches, queries, features])
                    .saturating_add(product(&[batches, features, keys]))
                    .saturating_add(product(&[batches, keys, values])),
                output: product(&[batches, queries, values]),
            }),
            _ => None,
        }
    }
}

impl PythonBackend {
    /// Calls the function with `arrays` and returns its result, checked to
    /// be `output` `float64` values with no NULLs before it is copied: a
    /// longer result would take memory einfold hasn't reserved. `noun` names
    /// the work in errors.
    fn call(
        &self,
        (kind, noun): (&str, &str),
        arrays: &[&[f64]],
        shape: Vec<usize>,
        output: usize,
        scale: (u8, f64),
        ranges: &[(usize, usize)],
    ) -> Result<Option<Vec<f64>>, ArrowError> {
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
            if result.len() != output {
                return Err(PyValueError::new_err(format!(
                    "the dense backend gave {noun} {} values, not {output}",
                    result.len()
                )));
            }
            if result.null_count() > 0 {
                return Err(PyValueError::new_err("a dense backend returned NULLs"));
            }
            Ok(Some(result.values().to_vec()))
        })
        .map_err(|e| ArrowError::ExternalError(Box::new(e)))
    }
}

impl DenseBackend for PythonBackend {
    fn may_run(&self, shape: DenseShape) -> bool {
        Cost::of(shape).is_some_and(|cost| {
            let bytes = (cost.operands.saturating_add(cost.output))
                .saturating_mul(std::mem::size_of::<f64>());
            cost.ops >= self.min_ops && bytes <= self.max_bytes
        })
    }

    fn run(&self, work: DenseWork<'_>) -> Result<Option<Vec<f64>>, ArrowError> {
        let Some(cost) = Cost::of(work.shape()).filter(|_| self.may_run(work.shape())) else {
            return Ok(None);
        };
        match work {
            DenseWork::Product { a, b, shape } => {
                let ProductShape {
                    batches,
                    rows,
                    inner,
                    columns,
                } = shape;
                let shape = vec![batches, rows, inner, columns];
                self.call(
                    ("matmul", "a product"),
                    &[a, b],
                    shape,
                    cost.output,
                    (0, 1.0),
                    &[],
                )
            }
            DenseWork::Attention {
                q,
                kt,
                v,
                shape,
                scale,
                allowed,
            } => {
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
                let shape = vec![batches, queries, features, keys, values];
                let arrays = [q, kt, v];
                self.call(
                    ("attention", "attention"),
                    &arrays,
                    shape,
                    cost.output,
                    scale,
                    allowed,
                )
            }
            // A kind of work this function isn't told of: declined.
            _ => Ok(None),
        }
    }
}
