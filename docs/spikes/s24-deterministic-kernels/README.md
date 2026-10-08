<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S24: deterministic dense kernels

**Question.** What do bit-for-bit repeatable dense kernels cost on ddx's contractions? ddx is an XQL Systems project for automatic differentiation of SQL queries. The design makes repeatable bits the default in einfold's own executor (§8.6). It says this comes free on CPU, "work is split by output groups, never within one". Spike S21's kernels don't behave that way. When the output has few rows, they split a long sum across threads, so the bits depend on the thread count. And its streaming fold adds rows in arrival order, so the bits depend on how rows arrive.

**Answer.** On CPU, determinism is free if work is split into fixed blocks chosen from the shape alone, and rows are visited in position order rather than arrival order.

* **Blocked kernels.** They gave one bit pattern across 6 thread counts and 3 arrival orders on every query. They ran as fast as S21's kernels, and sometimes faster. S21's kernels gave up to 4 (dense) and 12 (fold) bit patterns.
* **The indexed accumulator.** A matrix product that keeps every output cell in a one-pass reproducible accumulator (`indexed`, after ReproBLAS) costs **2–7× the best deterministic kernel on `matmul`** and **7–25× on attention's** products. Its bits depend on nothing but the values. That suits GPU atomics and `highest` precision, not the CPU default.

Date: 2026-10-07. Intel Core i7-8700 (6 cores, 12 threads), 15 GB RAM, Rust 1.91, `matrixmultiply` 0.3 (no threading feature).

## Method

[`src/main.rs`](src/main.rs) uses S21's data and shapes: complete coordinate tables with `sin` values, ddx's `matmul(n, 16, 8)` (forward, W̄ and Ā) and attention's two products. Every kernel reads Arrow-style rows and writes every output cell as Arrow arrays, as in S21.

| kernel | how it splits work | bits depend on |
|---|---|---|
| dense (S21) | GEMM; output rows split across threads, or the contraction when the output has fewer than 4 rows per thread | thread count, for few-row outputs |
| dense blocked | GEMM over fixed blocks chosen from the shape alone: 256 output rows per block when the output has at least 4,096 cells (each cell from one GEMM call over all of `z`), else 4,096 values of `z` per block, the partial outputs added in block order | nothing (for one build on one machine) |
| fold (S21) | stream `A`'s rows in arrival order into a dense output, with `B` dense | thread count and arrival order |
| fold by position | record each row's position (a scatter of row numbers; needs unique coordinates), then accumulate in position order, in the same fixed blocks as dense blocked | nothing |
| indexed | every output cell in a reproducible accumulator; blocks along `z` merged in reverse order, to show that the merge order doesn't matter | nothing, not even the order of additions |

[`src/indexed.rs`](src/indexed.rs) implements the one-pass *indexed type* of Ahrens, Nguyen and Demmel ([UC Berkeley EECS-2016-121](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-121.html)), the algorithm behind ReproBLAS, with K = 3 bins of W = 40 bits. Unlike S8's binned sum, it needs no `max|x|` in advance: a larger value shifts the kept bins up. Its unit tests check that a sum has the same bits under every shuffle, and under random splits merged in reverse. They also check that it stays within the paper's error bound, n·2⁻⁸⁰·max|x| + 7 ulp. The tests run 300 cases of up to 5,000 values (signs random, magnitudes 2⁻¹⁰⁰ to 2¹⁰⁰, some zeros) and a 200,000-value sum. Simplifications: finite values below 2⁹⁸⁴ only, since the top bin's scaled form isn't implemented.

For each kernel and query: the median of 5 runs on 12 threads. Then the number of distinct output bit patterns over thread counts {1, 2, 3, 5, 8, 12}, times three arrival orders of both operands (as generated, reversed, shuffled). Every output agreed with the indexed result to within 1e-9·(1 + |v|).

Run with `cargo run --release`; `cargo test --release` runs the accumulator's tests.

## Results

Times in ms, on 12 threads; "bits" is the number of distinct output bit patterns over 18 runs (6 thread counts × 3 arrival orders).

| workload | query | output cells | dense (S21) | bits | dense blocked | bits | fold (S21) | bits | fold by position | bits | indexed | bits |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| matmul n=10,000 | forward a·w | 80,000 | 0.9 | 1 | 0.9 | 1 | 1.1 | 3 | 0.9 | 1 | 3.5 | 1 |
| matmul n=10,000 | W̄ = aᵀ·z̄ | 128 | 1.0 | 4 | 1.0 | 1 | 0.7 | 12 | 1.2 | 1 | 3.1 | 1 |
| matmul n=10,000 | Ā = z̄·wᵀ | 160,000 | 0.8 | 1 | 0.9 | 1 | 1.0 | 3 | 0.7 | 1 | 5.0 | 1 |
| matmul n=50,000 | forward a·w | 400,000 | 6.7 | 1 | 6.8 | 1 | 9.5 | 3 | 3.1 | 1 | 14.8 | 1 |
| matmul n=50,000 | W̄ = aᵀ·z̄ | 128 | 3.5 | 4 | 3.6 | 1 | 1.9 | 12 | 3.2 | 1 | 7.5 | 1 |
| matmul n=50,000 | Ā = z̄·wᵀ | 800,000 | 5.6 | 1 | 6.9 | 1 | 7.4 | 3 | 4.0 | 1 | 25.7 | 1 |
| matmul n=200,000 | forward a·w | 1,600,000 | 23.3 | 1 | 26.1 | 1 | 36.2 | 3 | 23.9 | 1 | 72.5 | 1 |
| matmul n=200,000 | W̄ = aᵀ·z̄ | 128 | 16.3 | 4 | 12.7 | 1 | 7.3 | 12 | 13.1 | 1 | 46.0 | 1 |
| matmul n=200,000 | Ā = z̄·wᵀ | 3,200,000 | 13.9 | 1 | 15.6 | 1 | 19.6 | 3 | 15.3 | 1 | 93.1 | 1 |
| attn L=256 | scores q·kᵀ | 65,536 | 0.5 | 1 | 0.5 | 1 | 0.5 | 3 | 0.8 | 1 | 12.5 | 1 |
| attn L=256 | output p·v | 4,096 | 0.5 | 1 | 0.4 | 1 | 0.7 | 3 | 1.0 | 1 | 5.3 | 1 |
| attn L=1024 | scores q·kᵀ | 1,048,576 | 3.2 | 1 | 3.8 | 1 | 3.2 | 3 | 4.2 | 1 | 51.0 | 1 |
| attn L=1024 | output p·v | 16,384 | 3.2 | 1 | 3.0 | 1 | 5.6 | 3 | 4.4 | 1 | 20.2 | 1 |

## Findings

1. **S21's kernels are not deterministic, and the design's mechanism doesn't describe a fix.**
   * S21's dense kernel gave 4 bit patterns for W̄, the gradient whose 128 output cells each sum over all n rows: the split depends on the thread count.
   * S21's fold gave 3 patterns on every query (one per arrival order), and 12 on W̄.

   "Split by output groups, never within one" would serialize W̄'s long sums. "Each group's values are added in input order" makes the bits depend on arrival, which hosts don't fix (S3).
2. **Fixed blocks and position order make both kernels deterministic at no cost.** Blocking is chosen from the shape alone, and partial outputs are added in block order. With it, dense GEMM gave one bit pattern everywhere and ran within 0.78–1.23× of S21's time; on W̄ at n = 200,000 it was faster. The fold that visits rows in position order gave one bit pattern everywhere, and was as fast as or faster than S21's fold on large outputs (3.1 ms against 9.5 ms for the forward product at n = 50,000), apparently because position order also improves locality. Recovering positions needs unique coordinates. Duplicate keys would need a counting sort, not measured here.
3. **The reproducible accumulator costs 2–25×, depending on how long each cell's sum is.** On `matmul` it costs 2.3–3.6× the best deterministic kernel where sums are long (W̄), and 3–7× where they are 8–16 terms. On attention it costs 7–25×, where per-cell setup dominates the short sums. ReproBLAS's own GEMM measured 12.6× MKL's. It is the right tool where order can't be fixed, as with GPU atomics, or for `highest` precision; not the CPU default.
4. **The scope of the promise must be stated.** These results hold for one build on one machine. `matrixmultiply` picks micro-kernels by CPU features (FMA, AVX2, AVX-512), so bits may differ across machines; that was not tested here. Only the indexed accumulator is independent of the machine's kernels, since it adds rounded products whose bits don't depend on blocking.

## Limitations

- One machine, one GEMM library; GPU not measured.
- Dense, complete inputs with unique coordinates, as in S21.
- The accumulator's top bin (values ≥ 2⁹⁸⁴), exceptional values and the paper's exact conversion algorithm are not implemented; the conversion here sums the exact bin values with Shewchuk's correctly rounded `fsum`.
- The indexed kernel is a plain loop of deposits, not a tuned, vectorized GEMM.
