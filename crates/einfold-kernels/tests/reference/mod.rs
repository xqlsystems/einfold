// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A dense backend that computes each kind of work from its documented
//! meaning alone, in another order than einfold's (attention softmax first,
//! sums in reverse), and checks the contract einfold promises a backend:
//! operand lengths that match the shape, finite factors, and each query's
//! range of keys within the keys.

#![allow(dead_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use einfold_kernels::{AttentionShape, DenseBackend, DenseShape, DenseWork, ProductShape};

#[derive(Debug, Default)]
pub struct Reference {
    /// Work it computed.
    pub ran: AtomicUsize,
    /// Each broken promise, in words.
    pub broken: Mutex<Vec<String>>,
}

impl Reference {
    fn broke(&self, what: String) {
        self.broken.lock().unwrap().push(what);
    }

    pub fn broken(&self) -> Vec<String> {
        self.broken.lock().unwrap().clone()
    }
}

impl DenseBackend for Reference {
    fn run(&self, work: DenseWork<'_>) -> einfold_kernels::Result<Option<Vec<f64>>> {
        self.ran.fetch_add(1, Ordering::Relaxed);
        match work {
            DenseWork::Product { a, b, shape } => {
                let ProductShape {
                    batches: nw,
                    rows: nx,
                    inner: nz,
                    columns: ny,
                } = shape;
                if a.len() != nw * nx * nz || b.len() != nw * nz * ny {
                    self.broke(format!("product {shape:?}: a {} b {}", a.len(), b.len()));
                    return Ok(None);
                }
                if !a.iter().chain(b).all(|x| x.is_finite()) {
                    self.broke(format!("product {shape:?}: a factor isn't finite"));
                }
                let mut out = vec![0.0; nw * nx * ny];
                for w in 0..nw {
                    for x in 0..nx {
                        for y in 0..ny {
                            let mut s = 0.0;
                            for z in (0..nz).rev() {
                                s += a[(w * nx + x) * nz + z] * b[(w * nz + z) * ny + y];
                            }
                            out[(w * nx + x) * ny + y] = s;
                        }
                    }
                }
                Ok(Some(out))
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
                    batches: nw,
                    queries: nx,
                    features: ne,
                    keys: nj,
                    values: ny,
                } = shape;
                if q.len() != nw * nx * ne
                    || kt.len() != nw * ne * nj
                    || v.len() != nw * nj * ny
                    || allowed.len() != nx
                {
                    self.broke(format!(
                        "attention {shape:?}: q {} kt {} v {} ranges {}",
                        q.len(),
                        kt.len(),
                        v.len(),
                        allowed.len()
                    ));
                    return Ok(None);
                }
                if !q.iter().chain(kt).chain(v).all(|x| x.is_finite()) {
                    self.broke(format!("attention {shape:?}: an operand isn't finite"));
                }
                let mut out = vec![0.0; nw * nx * ny];
                for w in 0..nw {
                    for x in 0..nx {
                        let (lo, hi) = allowed[x];
                        if lo > hi || hi > nj {
                            self.broke(format!("attention {shape:?}: range {lo}..{hi} of {nj}"));
                            return Ok(None);
                        }
                        if lo == hi {
                            continue;
                        }
                        let s: Vec<f64> = (lo..hi)
                            .map(|j| {
                                let p: f64 = (0..ne)
                                    .map(|e| q[(w * nx + x) * ne + e] * kt[(w * ne + e) * nj + j])
                                    .sum();
                                scale.apply(p)
                            })
                            .collect();
                        let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        let e: Vec<f64> = s.iter().map(|s| (s - m).exp()).collect();
                        let z: f64 = e.iter().rev().sum();
                        for y in 0..ny {
                            let mut acc = 0.0;
                            for (i, j) in (lo..hi).enumerate().rev() {
                                acc += (e[i] / z) * v[(w * nj + j) * ny + y];
                            }
                            out[(w * nx + x) * ny + y] = acc;
                        }
                    }
                }
                Ok(Some(out))
            }
            _ => Ok(None),
        }
    }
}

/// A backend that computes as `Reference` does, but rounds every operand
/// and result to `float32`, as an MLX backend does.
#[derive(Debug, Default)]
pub struct Float32(pub Reference);

impl DenseBackend for Float32 {
    fn may_run(&self, shape: DenseShape) -> bool {
        self.0.may_run(shape)
    }

    fn run(&self, work: DenseWork<'_>) -> einfold_kernels::Result<Option<Vec<f64>>> {
        let r = |x: &[f64]| -> Vec<f64> { x.iter().map(|&x| f64::from(x as f32)).collect() };
        let out = match work {
            DenseWork::Product { a, b, shape } => {
                let (a, b) = (r(a), r(b));
                self.0.run(DenseWork::Product {
                    a: &a,
                    b: &b,
                    shape,
                })?
            }
            DenseWork::Attention {
                q,
                kt,
                v,
                shape,
                scale,
                allowed,
            } => {
                let (q, kt, v) = (r(q), r(kt), r(v));
                self.0.run(DenseWork::Attention {
                    q: &q,
                    kt: &kt,
                    v: &v,
                    shape,
                    scale,
                    allowed,
                })?
            }
            _ => None,
        };
        Ok(out.map(|o| r(&o)))
    }
}
