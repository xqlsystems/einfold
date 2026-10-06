// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Spike S8: what does a deterministic float sum cost on CPU? (Design doc §8.6.)
//!
//! Accumulators:
//! - plain:       left-to-right f64 sum. Deterministic only for a fixed order.
//! - fixed tree:  fixed 64 Ki-element blocks, each summed left to right, block
//!                results combined pairwise in block order. Deterministic for
//!                any thread count, but depends on the input order.
//! - Kahan:       Neumaier's compensated sum. More accurate; order-dependent.
//! - binned:      a reproducible sum after Demmel and Nguyen (2013): each value
//!                is split onto K = 3 fixed grids derived from max|x| and n, and
//!                each grid's sum is exact, so the result does not depend on
//!                order or threads. Needs max|x| first: a second pass, or a
//!                Bound from facts (chunk min/max statistics).
//! - superacc:    an exact fixed-point accumulator over the whole double range
//!                (2-word limbs per 32 bits of exponent range); order-independent.
//! - fsum:        Shewchuk's exact, correctly rounded sum (Python's math.fsum);
//!                the reference value. Correctly rounded implies deterministic.
//!
//! Data: values spanning 16 orders of magnitude with mixed signs, as spike S19.

use std::time::Instant;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn unif(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn data(n: usize, seed: u64) -> Vec<f64> {
    let mut r = Lcg(seed);
    (0..n)
        .map(|_| {
            let mag = 10f64.powf(r.unif() * 16.0 - 8.0);
            if r.next() & 1 == 0 { mag } else { -mag }
        })
        .collect()
}

// --- accumulators ---------------------------------------------------------

fn plain(x: &[f64]) -> f64 {
    x.iter().sum()
}

fn kahan(x: &[f64]) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for &v in x {
        let t = s + v;
        if s.abs() >= v.abs() { c += (s - t) + v } else { c += (v - t) + s }
        s = t;
    }
    s + c
}

const BLOCK: usize = 1 << 16;

fn fixed_tree(x: &[f64], threads: usize) -> f64 {
    let blocks: Vec<&[f64]> = x.chunks(BLOCK).collect();
    let mut sums = vec![0.0; blocks.len()];
    let per = blocks.len().div_ceil(threads);
    std::thread::scope(|s| {
        for (bs, out) in blocks.chunks(per).zip(sums.chunks_mut(per)) {
            s.spawn(move || {
                for (b, o) in bs.iter().zip(out.iter_mut()) {
                    *o = plain(b);
                }
            });
        }
    });
    while sums.len() > 1 {
        sums = sums.chunks(2).map(|p| p.iter().sum()).collect();
    }
    sums[0]
}

/// Binned reproducible sum. sigma[k] are powers of two; level k extracts the
/// part of each value on the grid of spacing ulp(sigma[k]); each level's sum
/// is exact because it holds at most n values of magnitude < sigma[k] / n on
/// that grid.
#[derive(Clone, Copy)]
struct Binned {
    sigma: [f64; 3],
}

impl Binned {
    fn new(max_abs: f64, n: usize) -> Self {
        let lg = |v: f64| v.log2().ceil() as i32;
        let mut sigma = [0.0; 3];
        let mut m = max_abs.max(f64::MIN_POSITIVE);
        for s in sigma.iter_mut() {
            *s = 2f64.powi(lg(m * n as f64) + 1);
            m = *s * 2f64.powi(-53); // the largest remainder after this level
        }
        Binned { sigma }
    }
    #[inline]
    fn add(&self, acc: &mut [f64; 3], v: f64) {
        let mut r = v;
        for k in 0..3 {
            let q = (self.sigma[k] + r) - self.sigma[k];
            acc[k] += q;
            r -= q;
        }
    }
    fn finish(acc: &[f64; 3]) -> f64 {
        acc[0] + acc[1] + acc[2]
    }
}

fn max_abs(x: &[f64]) -> f64 {
    x.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

fn binned(x: &[f64], threads: usize, known_max: Option<f64>) -> f64 {
    let m = known_max.unwrap_or_else(|| max_abs(x));
    let b = Binned::new(m, x.len());
    let per = x.len().div_ceil(threads);
    let mut parts = vec![[0.0; 3]; threads];
    std::thread::scope(|s| {
        for (chunk, acc) in x.chunks(per).zip(parts.iter_mut()) {
            s.spawn(move || {
                for &v in chunk {
                    b.add(acc, v);
                }
            });
        }
    });
    let mut tot = [0.0; 3];
    for p in &parts {
        for k in 0..3 {
            tot[k] += p[k]; // exact: same grid, within range
        }
    }
    Binned::finish(&tot)
}

/// Exact superaccumulator: 2^-1074 .. 2^1024 in 66 limbs of 32 bits, held in
/// i64 so that up to 2^31 additions need no carry propagation.
const LIMBS: usize = 72;
#[derive(Clone)]
struct Super([i64; LIMBS]);
impl Super {
    fn new() -> Self {
        Super([0; LIMBS])
    }
    #[inline]
    fn add(&mut self, v: f64) {
        let bits = v.to_bits();
        let exp = ((bits >> 52) & 0x7ff) as i64;
        let mut mant = (bits & ((1u64 << 52) - 1)) as i64;
        let shift = if exp == 0 { 0 } else { mant |= 1 << 52; exp - 1 };
        let (limb, off) = ((shift / 32) as usize, (shift % 32) as u32);
        let wide = (mant as i128) << off; // at most 85 bits
        let sign = if bits >> 63 == 1 { -1 } else { 1 };
        self.0[limb] += sign * (wide & 0xffff_ffff) as i64;
        self.0[limb + 1] += sign * ((wide >> 32) & 0xffff_ffff) as i64;
        self.0[limb + 2] += sign * (wide >> 64) as i64;
    }
    fn merge(&mut self, o: &Super) {
        for k in 0..LIMBS {
            self.0[k] += o.0[k];
        }
    }
    /// Normalize to an exact big integer and round to the nearest double.
    fn finish(&self) -> f64 {
        let mut l = self.0;
        for k in 0..LIMBS - 1 {
            let carry = l[k] >> 32; // arithmetic shift: floor division
            l[k] -= carry << 32;
            l[k + 1] += carry;
        }
        // Value = Σ l[k]·2^(32k − 1074), with 0 ≤ l[k] < 2^32 except the top limb.
        let neg = l[LIMBS - 1] < 0;
        if neg {
            // Negate the two's-complement big integer.
            let mut borrow = 0i64;
            for k in 0..LIMBS {
                let v = -l[k] - borrow;
                if k < LIMBS - 1 {
                    let c = if v < 0 { 1 } else { 0 };
                    l[k] = v + (c << 32);
                    borrow = c;
                } else {
                    l[k] = v;
                }
            }
        }
        let top = match (0..LIMBS).rev().find(|&k| l[k] != 0) {
            Some(t) => t,
            None => return 0.0,
        };
        // Take 96 bits from the top three limbs, plus a sticky bit for the rest,
        // and round once: the conversion from i128 rounds to nearest even.
        let lo = top.saturating_sub(2);
        let mut m: i128 = 0;
        for k in (lo..=top).rev() {
            m = (m << 32) | l[k] as i128;
        }
        let sticky = (0..lo).any(|k| l[k] != 0);
        let m = (m << 1) | sticky as i128;
        let r = (m as f64) * 2f64.powi(32 * lo as i32 - 1074 - 1);
        if neg { -r } else { r }
    }
}

fn superacc(x: &[f64], threads: usize) -> f64 {
    let per = x.len().div_ceil(threads);
    let mut parts = vec![Super::new(); threads];
    std::thread::scope(|s| {
        for (chunk, acc) in x.chunks(per).zip(parts.iter_mut()) {
            s.spawn(move || {
                for &v in chunk {
                    acc.add(v);
                }
            });
        }
    });
    let mut tot = Super::new();
    for p in &parts {
        tot.merge(p);
    }
    tot.finish()
}

/// Shewchuk's msum, as Python's math.fsum: exact and correctly rounded.
fn fsum(x: &[f64]) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for &v in x {
        let mut v = v;
        let mut i = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if v.abs() < y.abs() {
                std::mem::swap(&mut v, &mut y);
            }
            let hi = v + y;
            let lo = y - (hi - v);
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            v = hi;
        }
        partials.truncate(i);
        partials.push(v);
    }
    // Round the partials correctly (Python's final step).
    let mut n = partials.len();
    let mut hi = 0.0;
    if n > 0 {
        n -= 1;
        hi = partials[n];
        let mut lo = 0.0;
        while n > 0 {
            let x = hi;
            n -= 1;
            let y = partials[n];
            hi = x + y;
            let yr = hi - x;
            lo = y - yr;
            if lo != 0.0 {
                break;
            }
        }
        if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
            let y = lo * 2.0;
            let x = hi + y;
            if y == x - hi {
                hi = x;
            }
        }
    }
    hi
}

// --- harness ---------------------------------------------------------------

fn time<F: FnMut() -> f64>(mut f: F) -> (f64, f64) {
    let mut ts = Vec::new();
    let mut r = 0.0;
    for _ in 0..5 {
        let t0 = Instant::now();
        r = std::hint::black_box(f());
        ts.push(t0.elapsed().as_secs_f64() * 1e3);
    }
    ts.sort_by(|a, b| a.total_cmp(b));
    (ts[2], r)
}

fn shuffled(x: &[f64], seed: u64) -> Vec<f64> {
    let mut v = x.to_vec();
    let mut r = Lcg(seed);
    for i in (1..v.len()).rev() {
        let j = (r.next() % (i as u64 + 1)) as usize;
        v.swap(i, j);
    }
    v
}

fn main() {
    let n = 50_000_000;
    let x = data(n, 7);
    let exact = fsum(&x);
    let threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);
    let m = max_abs(&x);
    let rel = |v: f64| ((v - exact) / exact).abs();

    println!("## One long sum: {n} doubles, 16 orders of magnitude, mixed signs\n");
    println!("Exact (correctly rounded) sum: {exact:e}. Threads: {threads}.\n");
    println!("| Accumulator | Threads | ms | ns per value | vs plain | Relative error | Same bits under 3 shuffles × threads 1, 4, {threads} |");
    println!("|---|---|---|---|---|---|---|");
    let (t_plain, _) = time(|| plain(&x));
    let orders: Vec<Vec<f64>> = (1..=3).map(|s| shuffled(&x, s)).collect();
    type Acc<'a> = (&'a str, usize, Box<dyn Fn(&[f64], usize) -> f64>);
    let accs: Vec<Acc> = vec![
        ("plain", 1, Box::new(|v, _| plain(v))),
        ("Kahan (Neumaier)", 1, Box::new(|v, _| kahan(v))),
        ("fixed tree", 1, Box::new(fixed_tree)),
        ("fixed tree", threads, Box::new(fixed_tree)),
        ("binned, 2 passes", 1, Box::new(|v, t| binned(v, t, None))),
        ("binned, 2 passes", threads, Box::new(|v, t| binned(v, t, None))),
        ("binned, max from facts", 1, Box::new(move |v, t| binned(v, t, Some(m)))),
        ("binned, max from facts", threads, Box::new(move |v, t| binned(v, t, Some(m)))),
        ("superaccumulator", 1, Box::new(superacc)),
        ("superaccumulator", threads, Box::new(superacc)),
        ("fsum (Shewchuk)", 1, Box::new(|v, _| fsum(v))),
    ];
    for (name, th, f) in &accs {
        let (ms, r) = time(|| f(&x, *th));
        let mut results = std::collections::BTreeSet::new();
        for o in &orders {
            for t in [1, 4, threads] {
                results.insert(f(o, t).to_bits());
            }
        }
        results.insert(r.to_bits());
        let det = if results.len() == 1 { "yes".to_string() } else { format!("no ({} results)", results.len()) };
        println!("| {name} | {th} | {ms:.1} | {:.2} | {:.1}× | {:.1e} | {det} |", ms * 1e6 / n as f64, ms / t_plain, rel(r));
    }

    println!("\n## Grouped sums: 1 Mi groups × 64 values, arriving in random group order (hash-aggregation style)\n");
    println!("| Accumulator | State per group (bytes) | ms | vs plain | Max relative error | Same bits under a shuffle |");
    println!("|---|---|---|---|---|---|");
    let groups = 1 << 20;
    let per = 64;
    let vals = data(groups * per, 11);
    let mut r = Lcg(5);
    let mut keys: Vec<u32> = (0..groups * per).map(|i| (i / per) as u32).collect();
    for i in (1..keys.len()).rev() {
        let j = (r.next() % (i as u64 + 1)) as usize;
        keys.swap(i, j);
    }
    // Reference per group, and a second arrival order.
    let mut by_group: Vec<Vec<f64>> = vec![Vec::with_capacity(per); groups];
    for (&k, &v) in keys.iter().zip(&vals) {
        by_group[k as usize].push(v);
    }
    let exact_g: Vec<f64> = by_group.iter().map(|g| fsum(g)).collect();
    let perm: Vec<usize> = { let mut p: Vec<usize> = (0..keys.len()).collect(); let mut r = Lcg(9); for i in (1..p.len()).rev() { let j = (r.next() % (i as u64 + 1)) as usize; p.swap(i, j);} p };
    let keys2: Vec<u32> = perm.iter().map(|&i| keys[i]).collect();
    let vals2: Vec<f64> = perm.iter().map(|&i| vals[i]).collect();
    let gmax = max_abs(&vals);
    let maxrel = |res: &[f64]| res.iter().zip(&exact_g).map(|(a, e)| if *e == 0.0 { 0.0 } else { ((a - e) / e).abs() }).fold(0.0, f64::max);

    let run_plain = |k: &[u32], v: &[f64]| { let mut s = vec![0.0f64; groups]; for (&g, &x) in k.iter().zip(v) { s[g as usize] += x; } s };
    let run_kahan = |k: &[u32], v: &[f64]| {
        let mut s = vec![(0.0f64, 0.0f64); groups];
        for (&g, &x) in k.iter().zip(v) {
            let (a, c) = &mut s[g as usize];
            let t = *a + x;
            if a.abs() >= x.abs() { *c += (*a - t) + x } else { *c += (x - t) + *a }
            *a = t;
        }
        s.into_iter().map(|(a, c)| a + c).collect::<Vec<_>>()
    };
    // Binned with one global max (a Bound from facts) and n = values per group.
    let b = Binned::new(gmax, per);
    let run_binned = |k: &[u32], v: &[f64]| { let mut s = vec![[0.0f64; 3]; groups]; for (&g, &x) in k.iter().zip(v) { b.add(&mut s[g as usize], x); } s.iter().map(Binned::finish).collect::<Vec<_>>() };
    let run_super = |k: &[u32], v: &[f64]| { let mut s = vec![Super::new(); groups]; for (&g, &x) in k.iter().zip(v) { s[g as usize].add(x); } s.iter().map(|a| a.finish()).collect::<Vec<_>>() };

    let mut t0 = 0.0;
    for (name, bytes, f) in [
        ("plain", 8, &run_plain as &dyn Fn(&[u32], &[f64]) -> Vec<f64>),
        ("Kahan (Neumaier)", 16, &run_kahan),
        ("binned, max from facts", 24, &run_binned),
        ("superaccumulator", 8 * LIMBS, &run_super),
    ] {
        let mut ts = Vec::new();
        let mut res = Vec::new();
        for _ in 0..3 {
            let t = Instant::now();
            res = f(&keys, &vals);
            ts.push(t.elapsed().as_secs_f64() * 1e3);
        }
        ts.sort_by(|a, b| a.total_cmp(b));
        let ms = ts[1];
        if name == "plain" { t0 = ms; }
        let res2 = f(&keys2, &vals2);
        let same = res.iter().zip(&res2).all(|(a, b)| a.to_bits() == b.to_bits());
        println!("| {name} | {bytes} | {ms:.0} | {:.1}× | {:.1e} | {} |", ms / t0, maxrel(&res), if same { "yes" } else { "no" });
    }
}
