// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A one-pass reproducible accumulator for `f64` sums: the *indexed type* of
//! Ahrens, Nguyen and Demmel, "Efficient Reproducible Floating Point Summation
//! and BLAS" (UC Berkeley EECS-2016-121), the algorithm behind the ReproBLAS
//! library.
//!
//! The exponent range is cut into fixed bins of `W` bits. Each value is split
//! into *slices*, the parts of it that fall in each bin, rounded at the bin's
//! lower edge. The slices in one bin are added exactly, because they all lie
//! on that bin's grid. Only the `K` highest bins that the largest value so far
//! reaches are kept. Because the bins are fixed, and each slice depends only on
//! its value and its bin, the kept sums don't depend on the order of addition,
//! nor on how partial sums are grouped and merged: the result has the same
//! bits for any order, thread count or reduction tree.
//!
//! Unlike a binned sum with grids derived from `max|x|` (spike S8's), this needs
//! no maximum in advance: when a value above the current top bin arrives, the
//! accumulator shifts its bins up and drops the lowest ones, which a value
//! that large would have made irrelevant anyway.
//!
//! Simplifications: values must be finite and below 2^984 (the top bin, which
//! needs a scaled representation, is not implemented), and no more than about
//! 2^64 values may be added to one accumulator.

const W: i32 = 40;
const K: usize = 3;
/// The deposits a bin can absorb between renormalizations: 2^(p − W − 2).
const ENDURANCE: u32 = 1 << 11;
const IMAX: i32 = 51;

/// The lower exponent edge of bin `i`: values in bin `i` have exponents in
/// `(a(i), a(i) + W]`.
fn a(i: i32) -> i32 {
    1024 - (i + 1) * W
}

/// 1.5 · 2^(a(i) + 53): the value a bin's primary field starts from, so that
/// adding a slice changes only the low bits.
fn base(i: i32) -> f64 {
    1.5 * 2f64.powi(a(i) + 53)
}

/// 0.25 · 2^(a(i) + 53): the step a renormalization moves into the carry.
fn quarter(i: i32) -> f64 {
    0.25 * 2f64.powi(a(i) + 53)
}

/// The highest bin whose range contains `x`: the largest `i` with
/// `|x| < 2^(a(i) + W)`.
fn index_of(x: f64) -> i32 {
    if x == 0.0 {
        return IMAX;
    }
    let e = ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    // Keep all K bins at or above the lowest valid bin.
    ((1023 - e) / W).min(IMAX - K as i32 + 1)
}

/// `x` with the last bit of its significand set, so that adding it to a
/// primary field never meets a rounding tie.
fn odd(x: f64) -> f64 {
    f64::from_bits(x.to_bits() | 1)
}

#[derive(Clone, Copy, Debug)]
pub struct Indexed {
    /// Index of the top bin kept; `None` until the first value.
    top: Option<i32>,
    primary: [f64; K],
    carry: [i64; K],
    deposits: u32,
}

impl Default for Indexed {
    fn default() -> Self {
        Indexed {
            top: None,
            primary: [0.0; K],
            carry: [0; K],
            deposits: 0,
        }
    }
}

impl Indexed {
    /// Make the top bin at least as high as `j`, shifting the kept bins down
    /// and dropping the lowest.
    fn raise(&mut self, j: i32) {
        match self.top {
            None => {
                self.top = Some(j);
                for k in 0..K {
                    self.primary[k] = base(j + k as i32);
                    self.carry[k] = 0;
                }
            }
            Some(i) if j < i => {
                let shift = ((i - j) as usize).min(K);
                for k in (shift..K).rev() {
                    self.primary[k] = self.primary[k - shift];
                    self.carry[k] = self.carry[k - shift];
                }
                for k in 0..shift {
                    self.primary[k] = base(j + k as i32);
                    self.carry[k] = 0;
                }
                self.top = Some(j);
            }
            _ => {}
        }
    }

    /// Move each primary field back into `[1.5, 1.75) · 2^(a + 53)`, counting
    /// the steps in its carry. The represented value doesn't change.
    fn renormalize(&mut self) {
        let Some(i) = self.top else { return };
        for k in 0..K {
            let (b, q) = (base(i + k as i32), quarter(i + k as i32));
            while self.primary[k] >= b + q {
                self.primary[k] -= q;
                self.carry[k] += 1;
            }
            while self.primary[k] < b {
                self.primary[k] += q;
                self.carry[k] -= 1;
            }
        }
        self.deposits = 0;
    }

    /// Add one finite value.
    #[inline]
    pub fn add(&mut self, x: f64) {
        debug_assert!(x.is_finite() && x.abs() < 2f64.powi(984));
        // A zero adds nothing, and must not set the index: the bins below
        // the lowest valid one would underflow.
        if x == 0.0 {
            return;
        }
        let j = index_of(x);
        if self.top.is_none_or(|i| j < i) {
            self.raise(j);
        }
        if self.deposits == ENDURANCE {
            self.renormalize();
        }
        let mut r = x;
        for k in 0..K - 1 {
            let q = self.primary[k] + odd(r);
            let d = q - self.primary[k];
            self.primary[k] = q;
            r -= d;
        }
        self.primary[K - 1] += odd(r);
        self.deposits += 1;
    }

    /// Merge another accumulator in. The result is the same as adding all of
    /// its values here, in any order.
    pub fn merge(&mut self, other: &Indexed) {
        let Some(j) = other.top else { return };
        let mut o = *other;
        o.renormalize();
        self.raise(j);
        let i = self.top.unwrap();
        o.raise(i);
        self.renormalize();
        for k in 0..K {
            // Exact: both fields lie in the same binade, on the same grid.
            self.primary[k] += o.primary[k] - base(i + k as i32);
            self.carry[k] += o.carry[k];
        }
        self.renormalize();
    }

    /// The sum, rounded once.
    pub fn finish(&self) -> f64 {
        let Some(i) = self.top else { return 0.0 };
        let mut c = *self;
        c.renormalize();
        let mut parts = Vec::with_capacity(2 * K);
        for k in 0..K {
            let bin = i + k as i32;
            parts.push(c.primary[k] - base(bin));
            parts.push(c.carry[k] as f64 * quarter(bin));
        }
        fsum(&parts)
    }
}

/// Shewchuk's exact summation, correctly rounded, as Python's `math.fsum`.
pub fn fsum(x: &[f64]) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 11
        }
        fn unif(&mut self) -> f64 {
            self.next() as f64 / (1u64 << 53) as f64
        }
        fn value(&mut self) -> f64 {
            // Signs random, magnitudes over 2^-100 .. 2^100, some zeros.
            if self.unif() < 0.05 {
                return 0.0;
            }
            let s = if self.unif() < 0.5 { -1.0 } else { 1.0 };
            s * 2f64.powf(self.unif() * 200.0 - 100.0)
        }
    }

    fn sum(xs: &[f64]) -> f64 {
        let mut acc = Indexed::default();
        for &x in xs {
            acc.add(x);
        }
        acc.finish()
    }

    /// The same bits under every permutation and every grouping of partial
    /// sums tried, and close to the correctly rounded sum.
    #[test]
    fn order_and_grouping_do_not_matter() {
        let mut rng = Lcg(24);
        for case in 0..300 {
            let n = 1 + (rng.next() % 5000) as usize;
            let mut xs: Vec<f64> = (0..n).map(|_| rng.value()).collect();
            let want = sum(&xs);
            for _ in 0..4 {
                // Shuffle.
                for i in (1..n).rev() {
                    xs.swap(i, (rng.next() % (i as u64 + 1)) as usize);
                }
                assert_eq!(sum(&xs).to_bits(), want.to_bits(), "case {case}: shuffled");
                // Split into random chunks and merge them in reverse.
                let mut cuts: Vec<usize> = (0..(rng.next() % 7) as usize)
                    .map(|_| (rng.next() % n as u64) as usize)
                    .collect();
                cuts.push(0);
                cuts.push(n);
                cuts.sort();
                let mut parts: Vec<Indexed> = cuts
                    .windows(2)
                    .map(|w| {
                        let mut acc = Indexed::default();
                        for &x in &xs[w[0]..w[1]] {
                            acc.add(x);
                        }
                        acc
                    })
                    .collect();
                parts.reverse();
                let mut tot = Indexed::default();
                for p in &parts {
                    tot.merge(p);
                }
                assert_eq!(
                    tot.finish().to_bits(),
                    want.to_bits(),
                    "case {case}: merged"
                );
            }
            // Error: within n·2^-80·max|x| + 7 ulp of the exact sum (the
            // paper's bound for K = 3, W = 40).
            let exact = fsum(&xs);
            let max = xs.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let bound = n as f64 * 2f64.powi(-80) * max + 7.0 * f64::EPSILON / 2.0 * exact.abs();
            assert!(
                (want - exact).abs() <= bound,
                "case {case}: {want} vs {exact}"
            );
        }
    }

    /// Long sums cross many renormalizations.
    #[test]
    fn long_sums() {
        let mut rng = Lcg(7);
        let xs: Vec<f64> = (0..200_000).map(|_| rng.unif() - 0.5).collect();
        let mut rev = xs.clone();
        rev.reverse();
        assert_eq!(sum(&xs).to_bits(), sum(&rev).to_bits());
        assert!((sum(&xs) - fsum(&xs)).abs() <= 1e-15 * fsum(&xs).abs().max(1.0));
    }
}
