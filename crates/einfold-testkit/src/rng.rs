// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! A small deterministic generator, so that a seed reported in a failure
//! replays the same case on any machine.

/// SplitMix64, as the simulation tests of ddx (an XQL Systems project for
/// automatic differentiation of SQL queries) use: a fixed algorithm with no
/// dependencies, so a seed means the same case forever.
pub struct Rng(u64);

impl Rng {
    /// The generator for `seed`, salted so that different tests, or the
    /// different parts of one case, drawn from one seed don't share a stream.
    #[must_use]
    pub fn seeded(seed: u64, salt: u64) -> Self {
        Rng(seed.wrapping_mul(0x2545_f491_4f6c_dd1d) ^ salt)
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A number in `0..n`, or 0 when `n` is 0.
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n.max(1) as u64) as usize
    }

    /// True with probability `p`.
    pub fn chance(&mut self, p: f64) -> bool {
        self.unit() < p
    }

    /// A number in `[0, 1)`, with 53 random bits.
    pub fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) / ((1u64 << 53) as f64)
    }

    /// One of `xs`, which must not be empty.
    pub fn pick<'a, T>(&mut self, xs: &'a [T]) -> &'a T {
        &xs[self.below(xs.len())]
    }

    /// `xs` in a random order.
    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            xs.swap(i, self.below(i + 1));
        }
    }
}
