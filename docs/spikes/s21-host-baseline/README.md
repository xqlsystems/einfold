<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S21: the host baseline

**Question.** How long do host engines take on ddx's contractions, and how much faster are hand-written kernels that read and write the same tables? ddx is an XQL Systems project that runs and differentiates neural networks as SQL. This bounds what einfold can win, before any more of einfold is built.

**Answer.** Hand-written dense, positional kernels beat the faster of Apache DataFusion and DuckDB by **4–14× on ddx's matrix products** and **13–48× on its attention**, end to end from Arrow rows to Arrow rows. No hashing is needed: when dimensions are dense ranges, coordinates are positions, and an output group is an array index. M1's hash kernel, by contrast, ran at 0.11–0.6× of DataFusion on the same shapes (see [`docs/lessons.md`](../../lessons.md)). The win exists; M1 built the wrong algorithm for it.

Date: 2026-10-07. Intel Core i7-8700 (6 cores, 12 threads), 15 GB RAM, Rust 1.91, DataFusion 54.1.0, DuckDB 1.4.3.

## Method

[`src/main.rs`](src/main.rs) builds ddx's coordinate tables `(i, k, val)`, with dense integer coordinates and values from `sin`, as ddx's benchmark does. It then times each contraction `C[x, y] = Σ_z A[x, z] · B[z, y]`, written as ddx writes it:

```sql
SELECT a.s, w.o, SUM(a.val * w.val) AS v FROM a JOIN w ON a.k = w.k GROUP BY a.s, w.o
```

The workloads are ddx's `matmul(n, d=16, h=8)`, forward and both backward products, and `attention(L, d=16)`, its scores `q·kᵀ` and output `p·v`.

| Method | What it does |
|---|---|
| DataFusion | The SQL above, with each table in memory in 12 partitions of 8192-row batches, collected to Arrow batches. 12 threads |
| DuckDB | The same SQL in DuckDB's command-line tool, as `CREATE OR REPLACE TEMP TABLE out AS …`, on tables built in DuckDB with the same formulas. 12 threads |
| dense | Scatter both operands into dense row-major matrices, multiply with a GEMM (`matrixmultiply`), and emit one Arrow row per output cell. With 12 threads, the output rows are split across threads, or the contraction when the output has few rows |
| fold | Scatter only `B` into a dense matrix. Stream `A`'s rows, adding `val · B[z, :]` into the output row `x`, then emit. The positional form of a fused join and aggregate: no hash table, no join rows. Threads own disjoint output rows, or partial outputs when there are few |

Every time is the median of 5 runs after a warm-up. Each run's output is checked against DataFusion's by the sum of the absolute values of `v`, to a relative 1e-9. All agreed.

The kernels assume two things that einfold must prove before using them:
- each dimension is a dense integer range, so coordinates are positions; for other keys, a dictionary encoding gives positions;
- both inputs are complete, so every output cell exists. Spike S11 found that tracking which cells exist roughly doubles the dense algorithm's cost when that isn't known.

Run with `cargo run --release`, with `duckdb` on the `PATH`.

## Results

Times in ms. "Joined pairs" is the number of rows the join would produce.

| workload | query | joined pairs | output rows | DataFusion | DuckDB | dense, 1 thread | dense, 12 | fold, 1 thread | fold, 12 | best kernel vs best host |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| matmul n=10000 | forward a·w | 1280000 | 80000 | 8.4 | 36.0 | 2.4 | 2.3 | 1.9 | 1.1 | 7.4× |
| matmul n=10000 | backward W̄ = aᵀ·z̄ | 1280000 | 128 | 8.7 | 15.0 | 0.6 | 1.0 | 1.6 | 0.6 | 14.3× |
| matmul n=10000 | backward Ā = z̄·wᵀ | 1280000 | 160000 | 9.3 | 47.0 | 3.3 | 2.5 | 1.8 | 1.2 | 8.0× |
| matmul n=50000 | forward a·w | 6400000 | 400000 | 40.0 | 72.0 | 10.1 | 7.4 | 9.2 | 5.0 | 8.1× |
| matmul n=50000 | backward W̄ = aᵀ·z̄ | 6400000 | 128 | 27.8 | 33.0 | 3.2 | 3.6 | 8.0 | 2.0 | 14.3× |
| matmul n=50000 | backward Ā = z̄·wᵀ | 6400000 | 800000 | 52.1 | 95.0 | 16.9 | 9.7 | 16.1 | 8.4 | 6.2× |
| matmul n=200000 | forward a·w | 25600000 | 1600000 | 136.7 | 136.0 | 47.8 | 31.7 | 71.9 | 40.5 | 4.3× |
| matmul n=200000 | backward W̄ = aᵀ·z̄ | 25600000 | 128 | 96.4 | 83.0 | 13.1 | 16.1 | 46.1 | 7.0 | 11.8× |
| matmul n=200000 | backward Ā = z̄·wᵀ | 25600000 | 3200000 | 214.6 | 194.0 | 70.9 | 39.8 | 80.1 | 38.6 | 5.0× |
| attn L=256 | scores q·kᵀ | 1048576 | 65536 | 24.2 | 35.0 | 0.6 | 0.8 | 0.7 | 0.6 | 43.6× |
| attn L=256 | output p·v | 1048576 | 4096 | 6.7 | 32.0 | 0.4 | 0.6 | 0.9 | 0.7 | 17.8× |
| attn L=1024 | scores q·kᵀ | 16777216 | 1048576 | 229.0 | 343.0 | 6.9 | 4.8 | 8.8 | 4.7 | 48.3× |
| attn L=1024 | output p·v | 16777216 | 16384 | 71.9 | 94.0 | 3.6 | 3.1 | 12.8 | 5.1 | 23.3× |
| attn L=2048 | scores q·kᵀ | 67108864 | 4194304 | 636.4 | 1239.0 | 67.8 | 34.7 | 75.3 | 35.1 | 18.3× |
| attn L=2048 | output p·v | 67108864 | 32768 | 278.0 | 252.0 | 24.8 | 22.4 | 65.0 | 19.7 | 12.8× |

## Findings

1. **The win is real, for dense positional kernels.** The best kernel beat the best host on every shape: 4.3–14.3× on matrix products, and 12.8–48.3× on attention. At the larger sizes, the hosts spend about 4–10 ns per joined pair on 12 threads (more at small sizes, where fixed costs dominate). The kernels never form a pair.
2. **Positions, not hashes.** Neither kernel hashes anything. The *fold* kernel needs only one operand and the output dense. Where each output cell sums a few values (16 or 8: the forward product, `Ā` and attention scores), it is within 1.5× of GEMM, and sometimes faster. Where each cell sums many (`W̄` sums over all n rows, and `p·v` over L), GEMM is up to 3.5× faster on one thread, although the fold catches up with 12 threads on `W̄`. Both are better starting points for einfold than the hash algorithm M1 built.
3. **Writing the output bounds the win.** Where the output has millions of rows (forward and `Ā` at n = 200,000, scores at L = 2048), 12 threads gave the kernels at most a 2.1× speedup over one, against 6.6× for `W̄`, whose output has 128 rows: building the output rows dominates. So the speedup shrinks as outputs grow: 4.3–5.0× for the forward product and `Ā` at n = 200,000. Larger wins there need efficient Arrow output (no copies, emitted in parallel), or not materializing the output at all, by fusing its consumer.
4. **The host baseline depends on input layout.** With tables in 12 in-memory partitions, DataFusion took 8.4 ms for the forward product at n = 10,000. The adversarial review measured 31.7 ms on tables created by `CREATE TABLE … AS`. This spike reports the faster setup, so its speedups are conservative for DataFusion.
5. **DuckDB was no faster than DataFusion here,** except for matrix products at n = 200,000. Its times include writing a temporary table, which DataFusion's collection of batches avoids.

## Limitations

- Dense, complete, float64 inputs with integer coordinates in a known range. Sparse data, unknown extents, existence tracking and `float32` are not measured.
- Kernels were measured outside DataFusion: a real operator also pays for collecting its inputs, the memory pool and planning.
- One machine. No GPU.
- A checksum, not a full comparison of outputs.
