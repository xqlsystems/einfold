// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Spike S11: at what density does EinFold's dense algorithm beat its hash
//! algorithms on CPU? (Design doc §10.2 and §10.4.)
//!
//! C[i,j] = Σ_k A[i,k]·B[k,j], with A (n×n) and B (n×n) given as coordinate
//! rows (i, k, value), each entry present independently with probability d.
//! All algorithms produce the same output rows: one per group (i, j) with at
//! least one matching pair, as SQL's join + GROUP BY does.
//!
//! - hash:      hash table on B keyed by k; probe with A in arbitrary order;
//!              accumulate into a hash table keyed by (i, j). What a SQL
//!              engine's hash join + hash aggregate does, without the join rows.
//! - gustavson: A grouped by i (sorted), B in CSR by k; one row of state at a
//!              time: Gustavson's x, xb (multiple-switch) and JC arrays.
//! - dense:     scatter A and B into dense matrices, one GEMM for values and
//!              one GEMM over 0/1 indicators to know which groups exist, then
//!              emit groups whose count is nonzero. Scatter, both GEMMs and
//!              emission are all timed.
//! - dense (values only): one GEMM, emitting every group. Only valid when the
//!              inputs are known to be complete (an Exact dense fact), but it
//!              is the dense algorithm's best case.
//!
//! Single-threaded throughout. Each timing is the median of 5 runs.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};
use std::time::Instant;

/// A small fast hasher (FxHash), as hash-aggregation engines use.
#[derive(Default)]
struct Fx(u64);
impl Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.write_u64(*b as u64);
        }
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64)
    }
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64)
    }
}
type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Coordinate rows of an n×n matrix with density d, in shuffled order.
fn coo(n: usize, d: f64, rng: &mut Lcg) -> Vec<(u32, u32, f64)> {
    let mut v = Vec::new();
    for i in 0..n {
        for k in 0..n {
            if rng.next() < d {
                v.push((i as u32, k as u32, rng.next()));
            }
        }
    }
    for x in (1..v.len()).rev() {
        let y = (rng.next() * (x + 1) as f64) as usize;
        v.swap(x, y);
    }
    v
}

type Out = Vec<(u32, u32, f64)>;

fn hash(a: &[(u32, u32, f64)], b: &[(u32, u32, f64)]) -> Out {
    let mut build: FxMap<u32, Vec<(u32, f64)>> = FxMap::default();
    for &(k, j, v) in b {
        build.entry(k).or_default().push((j, v));
    }
    let mut agg: FxMap<(u32, u32), f64> = FxMap::default();
    for &(i, k, av) in a {
        if let Some(list) = build.get(&k) {
            for &(j, bv) in list {
                *agg.entry((i, j)).or_insert(0.0) += av * bv;
            }
        }
    }
    agg.into_iter().map(|((i, j), v)| (i, j, v)).collect()
}

fn gustavson(n: usize, a: &[(u32, u32, f64)], b: &[(u32, u32, f64)]) -> Out {
    // Counting sort A by row and B by row (k): Gustavson's distribution sort.
    let csr = |m: &[(u32, u32, f64)]| {
        let mut start = vec![0usize; n + 1];
        for &(r, _, _) in m {
            start[r as usize + 1] += 1;
        }
        for r in 0..n {
            start[r + 1] += start[r];
        }
        let mut pos = start.clone();
        let mut cols = vec![0u32; m.len()];
        let mut vals = vec![0f64; m.len()];
        for &(r, c, v) in m {
            let p = &mut pos[r as usize];
            cols[*p] = c;
            vals[*p] = v;
            *p += 1;
        }
        (start, cols, vals)
    };
    let (sa, ca, va) = csr(a);
    let (sb, cb, vb) = csr(b);
    let mut x = vec![0f64; n];
    let mut xb = vec![u32::MAX; n]; // multiple switch: the row that last touched column j
    let mut jc: Vec<u32> = Vec::with_capacity(n);
    let mut out = Vec::new();
    for i in 0..n {
        jc.clear();
        for p in sa[i]..sa[i + 1] {
            let (k, av) = (ca[p] as usize, va[p]);
            for q in sb[k]..sb[k + 1] {
                let j = cb[q] as usize;
                if xb[j] != i as u32 {
                    xb[j] = i as u32;
                    x[j] = 0.0;
                    jc.push(j as u32);
                }
                x[j] += av * vb[q];
            }
        }
        for &j in &jc {
            out.push((i as u32, j, x[j as usize]));
        }
    }
    out
}

fn gemm(n: usize, a: &[f64], b: &[f64], c: &mut [f64]) {
    unsafe {
        matrixmultiply::dgemm(n, n, n, 1.0, a.as_ptr(), n as isize, 1, b.as_ptr(), n as isize, 1, 0.0, c.as_mut_ptr(), n as isize, 1);
    }
}

fn dense(n: usize, a: &[(u32, u32, f64)], b: &[(u32, u32, f64)], track_matched: bool) -> Out {
    let mut da = vec![0f64; n * n];
    let mut db = vec![0f64; n * n];
    for &(i, k, v) in a {
        da[i as usize * n + k as usize] += v;
    }
    for &(k, j, v) in b {
        db[k as usize * n + j as usize] += v;
    }
    let mut c = vec![0f64; n * n];
    gemm(n, &da, &db, &mut c);
    let mut out = Vec::new();
    if track_matched {
        let mut ia = vec![0f64; n * n];
        let mut ib = vec![0f64; n * n];
        for &(i, k, _) in a {
            ia[i as usize * n + k as usize] = 1.0;
        }
        for &(k, j, _) in b {
            ib[k as usize * n + j as usize] = 1.0;
        }
        let mut m = vec![0f64; n * n];
        gemm(n, &ia, &ib, &mut m);
        for i in 0..n {
            for j in 0..n {
                if m[i * n + j] != 0.0 {
                    out.push((i as u32, j as u32, c[i * n + j]));
                }
            }
        }
    } else {
        for i in 0..n {
            for j in 0..n {
                out.push((i as u32, j as u32, c[i * n + j]));
            }
        }
    }
    out
}

fn median_ms<F: FnMut() -> Out>(mut f: F) -> (f64, Out) {
    let mut times = Vec::new();
    let mut last = Vec::new();
    for _ in 0..5 {
        let t0 = Instant::now();
        last = f();
        times.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    (times[2], last)
}

fn same(mut x: Out, mut y: Out) -> bool {
    x.sort_by_key(|t| (t.0, t.1));
    y.sort_by_key(|t| (t.0, t.1));
    x.len() == y.len() && x.iter().zip(&y).all(|(p, q)| p.0 == q.0 && p.1 == q.1 && (p.2 - q.2).abs() <= 1e-9 * (1.0 + q.2.abs()))
}

fn main() {
    println!("| n | density | products | output groups | hash (ms) | gustavson (ms) | dense (ms) | dense, values only (ms) | fastest | results agree |");
    println!("|---|---|---|---|---|---|---|---|---|---|");
    for n in [256usize, 512, 1024] {
        for d in [0.001, 0.003, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0] {
            let mut rng = Lcg(42);
            let (a, b) = (coo(n, d, &mut rng), coo(n, d, &mut rng));
            let products: f64 = {
                let mut per_k = vec![0usize; n];
                for &(k, _, _) in &b {
                    per_k[k as usize] += 1;
                }
                a.iter().map(|&(_, k, _)| per_k[k as usize] as f64).sum()
            };
            // The hash algorithm is very slow on large dense inputs; skip past 2e8 products.
            let (th, oh) = if products <= 2e8 { median_ms(|| hash(&a, &b)) } else { (f64::NAN, Vec::new()) };
            let (tg, og) = median_ms(|| gustavson(n, &a, &b));
            let (td, od) = median_ms(|| dense(n, &a, &b, true));
            let (tv, _) = median_ms(|| dense(n, &a, &b, false));
            let agree = (oh.is_empty() || same(oh.clone(), og.clone())) && same(og.clone(), od);
            let best = [("hash", th), ("gustavson", tg), ("dense", td)]
                .into_iter()
                .filter(|(_, t)| !t.is_nan())
                .min_by(|x, y| x.1.total_cmp(&y.1))
                .unwrap()
                .0;
            let fmt = |t: f64| if t.is_nan() { "—".to_string() } else { format!("{t:.2}") };
            println!(
                "| {n} | {d} | {:.2e} | {} | {} | {} | {} | {} | {best} | {} |",
                products, og.len(), fmt(th), fmt(tg), fmt(td), fmt(tv), if agree { "yes" } else { "**no**" }
            );
        }
    }
}
