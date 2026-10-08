// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Spike S24: what do bit-for-bit repeatable dense kernels cost on ddx's
//! contractions?
//!
//! ddx (an XQL Systems project for automatic differentiation of SQL queries)
//! computes matrix products as SQL: `SELECT x, y, SUM(a.val * b.val) FROM a
//! JOIN b ON a.z = b.z GROUP BY x, y` over coordinate tables `(i, k, val)`.
//! Spike S21 found that dense positional kernels beat the hosts by 4–48× on
//! these. But S21's kernels split a long sum across threads when the output
//! has few rows, so their bits depend on the thread count, and its streaming
//! "fold" kernel adds rows in arrival order, so its bits depend on the order
//! rows arrive in. einfold's design makes repeatable bits the default in its
//! own executor. This spike measures, on the same shapes:
//!
//! - S21's kernels as they are (`dense`, `fold`);
//! - the same kernels with work split into fixed blocks chosen from the shape
//!   alone, and partial sums combined in block order (`dense blocked`), and a
//!   fold that visits rows in position order rather than arrival order (`fold
//!   by position`);
//! - a product that accumulates every output cell in a one-pass reproducible
//!   accumulator (`indexed`; see `indexed.rs`), whose bits depend on nothing
//!   but the values.
//!
//! For each, it counts the distinct output bit patterns over thread counts
//! 1–12 and three row arrival orders.

mod indexed;

use std::collections::HashSet;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::thread;
use std::time::{Duration, Instant};

use arrow_array::{Float64Array, Int64Array};
use indexed::Indexed;

const RUNS: usize = 5;
/// Fixed block of the contracted dimension, for outputs with few cells.
const ZB: usize = 4096;
/// Fixed block of output rows, for outputs with many cells.
const XB: usize = 256;
/// Below this many output cells, split the contraction; otherwise the rows.
const FEW_CELLS: usize = 4096;

// --- data ----------------------------------------------------------------------

/// A complete coordinate table over `0..e0 × 0..e1`: row `r` has `c0 = r /
/// e1`, `c1 = r % e1` and `val = scale · sin(r + phase)`, as in S21.
#[derive(Clone, Copy)]
struct Table {
    extents: (usize, usize),
    phase: f64,
    scale: f64,
}

/// Operand rows, in some arrival order.
#[derive(Clone)]
struct Rows {
    c0: Vec<i64>,
    c1: Vec<i64>,
    v: Vec<f64>,
}

impl Table {
    fn rows(&self) -> Rows {
        let (e0, e1) = self.extents;
        let n = e0 * e1;
        Rows {
            c0: (0..n).map(|r| (r / e1) as i64).collect(),
            c1: (0..n).map(|r| (r % e1) as i64).collect(),
            v: (0..n)
                .map(|r| self.scale * (r as f64 + self.phase).sin())
                .collect(),
        }
    }
}

impl Rows {
    fn permuted(&self, order: &[usize]) -> Rows {
        Rows {
            c0: order.iter().map(|&i| self.c0[i]).collect(),
            c1: order.iter().map(|&i| self.c1[i]).collect(),
            v: order.iter().map(|&i| self.v[i]).collect(),
        }
    }
}

/// `C[x, y] = Σ_z A[x, z] · B[z, y]`; `a_x`, `b_z` say which column of each
/// table holds `x` and `z`.
struct Contraction {
    label: &'static str,
    a: Table,
    b: Table,
    a_x: usize,
    b_z: usize,
}

impl Contraction {
    fn dims(&self) -> (usize, usize, usize) {
        let e = |t: &Table, i: usize| if i == 0 { t.extents.0 } else { t.extents.1 };
        (
            e(&self.a, self.a_x),
            e(&self.a, 1 - self.a_x),
            e(&self.b, 1 - self.b_z),
        )
    }
}

fn mm(n: usize) -> Vec<Contraction> {
    let a = Table {
        extents: (n, 16),
        phase: 0.0,
        scale: 1.0,
    };
    let w = Table {
        extents: (16, 8),
        phase: 0.5,
        scale: 0.1,
    };
    let zbar = Table {
        extents: (n, 8),
        phase: 1.0,
        scale: 0.01,
    };
    vec![
        Contraction {
            label: "forward a·w",
            a,
            b: w,
            a_x: 0,
            b_z: 0,
        },
        Contraction {
            label: "backward W̄ = aᵀ·z̄",
            a,
            b: zbar,
            a_x: 1,
            b_z: 0,
        },
        Contraction {
            label: "backward Ā = z̄·wᵀ",
            a: zbar,
            b: w,
            a_x: 0,
            b_z: 1,
        },
    ]
}

fn attn(l: usize) -> Vec<Contraction> {
    let q = Table {
        extents: (l, 16),
        phase: 0.0,
        scale: 1.0,
    };
    let kk = Table {
        extents: (l, 16),
        phase: 0.5,
        scale: 1.0,
    };
    let p = Table {
        extents: (l, l),
        phase: 1.0,
        scale: 0.01,
    };
    let vv = Table {
        extents: (l, 16),
        phase: 1.5,
        scale: 1.0,
    };
    vec![
        Contraction {
            label: "scores q·kᵀ",
            a: q,
            b: kk,
            a_x: 0,
            b_z: 1,
        },
        Contraction {
            label: "output p·v",
            a: p,
            b: vv,
            a_x: 0,
            b_z: 0,
        },
    ]
}

// --- shared pieces ----------------------------------------------------------------

/// Scatter rows into a dense row-major matrix, rows indexed by column `r`.
fn scatter(t: &Rows, r: usize, ncols: usize, nrows: usize) -> Vec<f64> {
    let mut m = vec![0.0; nrows * ncols];
    let (ri, ci) = if r == 0 {
        (&t.c0, &t.c1)
    } else {
        (&t.c1, &t.c0)
    };
    for ((&i, &j), &v) in ri.iter().zip(ci).zip(&t.v) {
        m[i as usize * ncols + j as usize] = v;
    }
    m
}

/// `c += a · b` by GEMM, for `a` of shape `m × k` and `b` of `k × n`.
fn gemm(a: &[f64], b: &[f64], c: &mut [f64], k: usize, n: usize) {
    let m = a.len() / k;
    // SAFETY: the slices hold m·k, k·n and m·n elements in row-major order.
    unsafe {
        matrixmultiply::dgemm(
            m,
            k,
            n,
            1.0,
            a.as_ptr(),
            k as isize,
            1,
            b.as_ptr(),
            n as isize,
            1,
            1.0,
            c.as_mut_ptr(),
            n as isize,
            1,
        );
    }
}

/// The result as coordinate rows: every output cell, as Arrow arrays.
struct Out {
    v: Float64Array,
}

fn emit(c: Vec<f64>, ny: usize, threads: usize) -> Out {
    let n = c.len();
    let mut x = vec![0i64; n];
    let mut y = vec![0i64; n];
    let chunk = n.div_ceil(threads).max(1);
    thread::scope(|s| {
        for (i, (xs, ys)) in x.chunks_mut(chunk).zip(y.chunks_mut(chunk)).enumerate() {
            s.spawn(move || {
                for (j, (xv, yv)) in xs.iter_mut().zip(ys.iter_mut()).enumerate() {
                    *xv = ((i * chunk + j) / ny) as i64;
                    *yv = ((i * chunk + j) % ny) as i64;
                }
            });
        }
    });
    let (_x, _y) = (Int64Array::from(x), Int64Array::from(y));
    Out {
        v: Float64Array::from(c),
    }
}

/// Run `f(i)` for every task `i` in `0..tasks` on `threads` threads, each
/// thread taking tasks `t, t + threads, …`. Results are returned by task, so
/// the assignment of tasks to threads never shows in them.
fn par_tasks<T: Send>(tasks: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let mut slots: Vec<Option<T>> = (0..tasks).map(|_| None).collect();
    let f = &f;
    thread::scope(|s| {
        let hs: Vec<_> = (0..threads.min(tasks).max(1))
            .map(|t| {
                s.spawn(move || {
                    (t..tasks)
                        .step_by(threads)
                        .map(|i| (i, f(i)))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in hs {
            for (i, r) in h.join().unwrap() {
                slots[i] = Some(r);
            }
        }
    });
    slots.into_iter().map(Option::unwrap).collect()
}

// --- S21's kernels ------------------------------------------------------------------

/// S21: split output rows when there are enough, else split the contraction
/// with one partial output per thread (bits depend on the thread count).
fn split_x(nx: usize, threads: usize) -> bool {
    threads == 1 || nx >= 4 * threads
}

fn dense_s21(c: &Contraction, a: &Rows, b: &Rows, threads: usize) -> Out {
    let (nx, nz, ny) = c.dims();
    let am = scatter(a, c.a_x, nz, nx);
    let bm = scatter(b, c.b_z, ny, nz);
    let mut out = vec![0.0; nx * ny];
    if split_x(nx, threads) {
        let rows = nx.div_ceil(threads);
        thread::scope(|s| {
            for (ac, cc) in am.chunks(rows * nz).zip(out.chunks_mut(rows * ny)) {
                let bm = &bm;
                s.spawn(move || gemm(ac, bm, cc, nz, ny));
            }
        });
    } else {
        let zs = nz.div_ceil(threads);
        let partials = par_tasks(threads, threads, |t| {
            let (z0, z1) = (t * zs, ((t + 1) * zs).min(nz));
            let mut p = vec![0.0; nx * ny];
            if z0 < z1 {
                let w = z1 - z0;
                let mut asl = vec![0.0; nx * w];
                for i in 0..nx {
                    asl[i * w..(i + 1) * w].copy_from_slice(&am[i * nz + z0..i * nz + z1]);
                }
                gemm(&asl, &bm[z0 * ny..z1 * ny], &mut p, w, ny);
            }
            p
        });
        for p in partials {
            for (o, v) in out.iter_mut().zip(p) {
                *o += v;
            }
        }
    }
    emit(out, ny, threads)
}

fn fold_s21(c: &Contraction, a: &Rows, b: &Rows, threads: usize) -> Out {
    let (nx, nz, ny) = c.dims();
    let bm = scatter(b, c.b_z, ny, nz);
    let (xs, zs) = if c.a_x == 0 {
        (&a.c0, &a.c1)
    } else {
        (&a.c1, &a.c0)
    };
    let accumulate = |out: &mut [f64], x0: usize, idx: &mut dyn Iterator<Item = usize>| {
        for r in idx {
            let (x, z, v) = (xs[r] as usize - x0, zs[r] as usize, a.v[r]);
            for (o, bv) in out[x * ny..(x + 1) * ny]
                .iter_mut()
                .zip(&bm[z * ny..(z + 1) * ny])
            {
                *o += v * bv;
            }
        }
    };
    let mut out = vec![0.0; nx * ny];
    let n = a.v.len();
    if split_x(nx, threads) {
        let per = nx.div_ceil(threads);
        let mut buckets = vec![Vec::new(); threads];
        for (r, &x) in xs.iter().enumerate() {
            buckets[x as usize / per].push(r);
        }
        thread::scope(|s| {
            for (t, (cc, bucket)) in out.chunks_mut(per * ny).zip(&buckets).enumerate() {
                let accumulate = &accumulate;
                s.spawn(move || accumulate(cc, t * per, &mut bucket.iter().copied()));
            }
        });
    } else {
        let per = n.div_ceil(threads);
        let partials = par_tasks(threads, threads, |t| {
            let mut p = vec![0.0; nx * ny];
            accumulate(&mut p, 0, &mut (t * per..((t + 1) * per).min(n)));
            p
        });
        for p in partials {
            for (o, v) in out.iter_mut().zip(p) {
                *o += v;
            }
        }
    }
    emit(out, ny, threads)
}

// --- deterministic kernels -------------------------------------------------------------

/// GEMM over fixed blocks chosen from the shape alone: blocks of `XB` output
/// rows when the output is large (each cell then comes from one GEMM call
/// over all of `z`), else blocks of `ZB` along `z`, each giving a partial
/// output, added in block order. Neither the thread count nor the arrival
/// order of rows can change which values are added in which order.
fn dense_blocked(c: &Contraction, a: &Rows, b: &Rows, threads: usize) -> Out {
    let (nx, nz, ny) = c.dims();
    let am = scatter(a, c.a_x, nz, nx);
    let bm = scatter(b, c.b_z, ny, nz);
    let out = if nx * ny >= FEW_CELLS {
        let blocks = par_tasks(nx.div_ceil(XB), threads, |t| {
            let (x0, x1) = (t * XB, ((t + 1) * XB).min(nx));
            let mut cc = vec![0.0; (x1 - x0) * ny];
            gemm(&am[x0 * nz..x1 * nz], &bm, &mut cc, nz, ny);
            cc
        });
        blocks.concat()
    } else {
        let partials = par_tasks(nz.div_ceil(ZB), threads, |t| {
            let (z0, z1) = (t * ZB, ((t + 1) * ZB).min(nz));
            let w = z1 - z0;
            let mut asl = vec![0.0; nx * w];
            for i in 0..nx {
                asl[i * w..(i + 1) * w].copy_from_slice(&am[i * nz + z0..i * nz + z1]);
            }
            let mut p = vec![0.0; nx * ny];
            gemm(&asl, &bm[z0 * ny..z1 * ny], &mut p, w, ny);
            p
        });
        let mut out = vec![0.0; nx * ny];
        for p in partials {
            for (o, v) in out.iter_mut().zip(p) {
                *o += v;
            }
        }
        out
    };
    emit(out, ny, threads)
}

/// The fold, visiting `a`'s rows in position order instead of arrival
/// order: first record each row's position (a scatter of row numbers, which
/// needs unique coordinates), then accumulate block by block as
/// `dense_blocked` splits its work.
fn fold_by_position(c: &Contraction, a: &Rows, b: &Rows, threads: usize) -> Out {
    let (nx, nz, ny) = c.dims();
    let bm = scatter(b, c.b_z, ny, nz);
    let (xs, zs) = if c.a_x == 0 {
        (&a.c0, &a.c1)
    } else {
        (&a.c1, &a.c0)
    };
    let mut at = vec![u32::MAX; nx * nz];
    for (r, (&x, &z)) in xs.iter().zip(zs).enumerate() {
        at[x as usize * nz + z as usize] = r as u32;
    }
    // Add a's row at (x, z), if present, times B's row z into `row`.
    let add = |row: &mut [f64], x: usize, z: usize| {
        let r = at[x * nz + z];
        if r != u32::MAX {
            let v = a.v[r as usize];
            for (o, bv) in row.iter_mut().zip(&bm[z * ny..(z + 1) * ny]) {
                *o += v * bv;
            }
        }
    };
    let out = if nx * ny >= FEW_CELLS {
        par_tasks(nx.div_ceil(XB), threads, |t| {
            let (x0, x1) = (t * XB, ((t + 1) * XB).min(nx));
            let mut cc = vec![0.0; (x1 - x0) * ny];
            for x in x0..x1 {
                for z in 0..nz {
                    add(&mut cc[(x - x0) * ny..(x - x0 + 1) * ny], x, z);
                }
            }
            cc
        })
        .concat()
    } else {
        let partials = par_tasks(nz.div_ceil(ZB), threads, |t| {
            let mut p = vec![0.0; nx * ny];
            for z in t * ZB..((t + 1) * ZB).min(nz) {
                for x in 0..nx {
                    add(&mut p[x * ny..(x + 1) * ny], x, z);
                }
            }
            p
        });
        let mut out = vec![0.0; nx * ny];
        for p in partials {
            for (o, v) in out.iter_mut().zip(p) {
                *o += v;
            }
        }
        out
    };
    emit(out, ny, threads)
}

/// Every output cell in a reproducible accumulator: the products are added
/// in whatever order is convenient, and partial accumulators are merged in
/// whatever order threads finish, without changing a bit.
fn indexed_product(c: &Contraction, a: &Rows, b: &Rows, threads: usize) -> Out {
    let (nx, nz, ny) = c.dims();
    let am = scatter(a, c.a_x, nz, nx);
    let bm = scatter(b, c.b_z, ny, nz);
    // b transposed, so each cell's products read contiguous memory.
    let mut bt = vec![0.0; ny * nz];
    for z in 0..nz {
        for y in 0..ny {
            bt[y * nz + z] = bm[z * ny + y];
        }
    }
    let cell = |acc: &mut Indexed, x: usize, y: usize, z0: usize, z1: usize| {
        for (av, bv) in am[x * nz + z0..x * nz + z1]
            .iter()
            .zip(&bt[y * nz + z0..y * nz + z1])
        {
            acc.add(av * bv);
        }
    };
    let out: Vec<f64> = if nx * ny >= FEW_CELLS {
        par_tasks(nx.div_ceil(XB), threads, |t| {
            let mut cc = Vec::with_capacity(XB * ny);
            for x in t * XB..((t + 1) * XB).min(nx) {
                for y in 0..ny {
                    let mut acc = Indexed::default();
                    cell(&mut acc, x, y, 0, nz);
                    cc.push(acc.finish());
                }
            }
            cc
        })
        .concat()
    } else {
        // Blocks along z, one accumulator per cell per block, merged in any
        // order: here, in reverse, to make the point.
        let mut parts = par_tasks(nz.div_ceil(ZB), threads, |t| {
            let (z0, z1) = (t * ZB, ((t + 1) * ZB).min(nz));
            let mut accs = vec![Indexed::default(); nx * ny];
            for x in 0..nx {
                for y in 0..ny {
                    cell(&mut accs[x * ny + y], x, y, z0, z1);
                }
            }
            accs
        });
        parts.reverse();
        let mut tot = vec![Indexed::default(); nx * ny];
        for p in &parts {
            for (t, q) in tot.iter_mut().zip(p) {
                t.merge(q);
            }
        }
        tot.iter().map(Indexed::finish).collect()
    };
    emit(out, ny, threads)
}

// --- harness ---------------------------------------------------------------------------

type Kernel = fn(&Contraction, &Rows, &Rows, usize) -> Out;

fn bits(o: &Out) -> u64 {
    let mut h = DefaultHasher::new();
    for v in o.v.values() {
        v.to_bits().hash(&mut h);
    }
    h.finish()
}

fn median(mut f: impl FnMut()) -> Duration {
    f();
    let mut ts: Vec<Duration> = (0..RUNS)
        .map(|_| {
            let t = Instant::now();
            f();
            t.elapsed()
        })
        .collect();
    ts.sort();
    ts[RUNS / 2]
}

/// Arrival orders: as generated, reversed, and a fixed shuffle.
fn orders(n: usize) -> Vec<Vec<usize>> {
    let id: Vec<usize> = (0..n).collect();
    let rev: Vec<usize> = (0..n).rev().collect();
    let mut sh = id.clone();
    let mut s = 0x2424_u64;
    for i in (1..n).rev() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        sh.swap(i, ((s >> 33) % (i as u64 + 1)) as usize);
    }
    vec![id, rev, sh]
}

fn main() {
    let threads = thread::available_parallelism().unwrap().get();
    let kernels: [(&str, Kernel); 5] = [
        ("dense (S21)", dense_s21),
        ("dense blocked", dense_blocked),
        ("fold (S21)", fold_s21),
        ("fold by position", fold_by_position),
        ("indexed", indexed_product),
    ];
    let mut workloads: Vec<(String, Vec<Contraction>)> = Vec::new();
    for n in [10_000, 50_000, 200_000] {
        workloads.push((format!("matmul n={n}"), mm(n)));
    }
    for l in [256, 1024] {
        workloads.push((format!("attn L={l}"), attn(l)));
    }
    print!("| workload | query | output cells |");
    for (name, _) in &kernels {
        print!(" {name}, ms | distinct bits |");
    }
    println!("\n|---|---|---:|{}", "---:|---:|".repeat(kernels.len()));
    for (wname, cs) in &workloads {
        for c in cs {
            let (nx, _, ny) = c.dims();
            let (a, b) = (c.a.rows(), c.b.rows());
            let reference = indexed_product(c, &a, &b, threads);
            print!("| {wname} | {} | {} |", c.label, nx * ny);
            for (_, k) in &kernels {
                let t = median(|| {
                    k(c, &a, &b, threads);
                });
                // Bits over thread counts and arrival orders of both operands.
                let mut seen = HashSet::new();
                let mut worst: f64 = 0.0;
                for order in orders(a.v.len()).iter().zip(orders(b.v.len())) {
                    let (ap, bp) = (a.permuted(order.0), b.permuted(&order.1));
                    for t in [1, 2, 3, 5, 8, 12] {
                        let o = k(c, &ap, &bp, t);
                        seen.insert(bits(&o));
                        for (x, y) in o.v.values().iter().zip(reference.v.values()) {
                            worst = worst.max((x - y).abs() / (1.0 + y.abs()));
                        }
                    }
                }
                assert!(
                    worst < 1e-9,
                    "{wname} {}: difference {worst} from the indexed result",
                    c.label
                );
                print!(" {:.1} | {} |", t.as_secs_f64() * 1e3, seen.len());
            }
            println!();
        }
    }
}
