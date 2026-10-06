<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S11: when does EinFold's dense algorithm win?

**Question.** At what density and size does EinFold's dense algorithm beat its hash algorithm on CPU? This also sets the switching threshold for run-time switching (design doc §10.2 and §10.4).

**Answer.** For a matrix product with both inputs at uniform density `d`:

- **Dense GEMM overtakes Gustavson's algorithm at about `d = 0.2`** (20% of entries present), at every size tested.
- **Gustavson's algorithm beats a SQL-style hash join with hash aggregation by 2.5–17×** across densities, and is faster than dense below the crossover.
- **Dense overtakes the hash join with hash aggregation much earlier, at `d ≈ 0.05–0.1`.** That is close to the 5% Staudt et al. found for their switch, which was against a different sparse implementation.
- Tracking which output groups exist roughly **doubles** the dense algorithm's cost. An Exact fact that both inputs are complete removes that cost.

So einfold's default switching threshold should be about 20% for the einsum form, on CPU. Measured densities (§10.4) decide at run time.

Date: 2026-10-06. Intel Core i7-8700, single thread, Rust 1.91, `matrixmultiply` 0.3 for GEMM.

## Method

[`src/main.rs`](src/main.rs) computes `C[i,j] = Σ_k A[i,k]·B[k,j]` for `n × n` inputs, each entry present independently with probability `d`, given as shuffled coordinate rows `(i, k, value)`. Every algorithm produces the same rows: one per group `(i, j)` that some pair of entries reaches, as SQL's join plus `GROUP BY` does. Results were checked to agree.

| Algorithm | What it does |
|---|---|
| hash | hash table on `B` keyed by `k`; probe with `A` in arbitrary order; accumulate into a hash table keyed by `(i, j)`. What a SQL engine does, minus the join rows |
| Gustavson | counting-sort `A` and `B` by row (CSR); one row of state at a time: Gustavson's `x`, `xb` (multiple switch) and `JC` arrays. EinFold's streaming hash algorithm (§10.2) |
| dense | scatter both inputs into dense matrices; one GEMM for the values, and one over 0/1 indicators to know which groups exist; emit those groups. Scatter, both GEMMs and emission are timed |
| dense, values only | one GEMM, emitting every group. Valid only with an Exact fact that both inputs are complete; the dense algorithm's best case |

Sizes `n` = 256, 512 and 1024; densities from 0.001 to 1. Each timing is the median of 5 runs. The hash algorithm was skipped above 2·10⁸ products.

Run with `cargo run --release`.

## Results

Times in ms; "products" is the number of nonzero-by-nonzero multiplications.

| n | d | products | hash | Gustavson | dense | dense, values only | fastest |
|---|---|---|---|---|---|---|---|
| 256 | 0.01 | 1.7e3 | 0.05 | 0.02 | 1.48 | 0.79 | Gustavson |
| 256 | 0.05 | 4.3e4 | 1.50 | 0.31 | 1.85 | 0.82 | Gustavson |
| 256 | 0.1 | 1.7e5 | 3.91 | 0.96 | 1.68 | 0.89 | Gustavson |
| 256 | 0.2 | 6.8e5 | 6.33 | 1.96 | 1.60 | 0.86 | dense |
| 256 | 1 | 1.7e7 | 106 | 28.2 | 1.92 | 1.00 | dense |
| 512 | 0.01 | 1.3e4 | 0.35 | 0.10 | 11.8 | 6.19 | Gustavson |
| 512 | 0.1 | 1.4e6 | 22.4 | 5.11 | 11.9 | 6.09 | Gustavson |
| 512 | 0.2 | 5.4e6 | 54.2 | 12.7 | 12.2 | 6.20 | dense |
| 512 | 1 | 1.3e8 | 1206 | 223 | 14.4 | 7.40 | dense |
| 1024 | 0.01 | 1.1e5 | 2.35 | 0.71 | 100 | 48.6 | Gustavson |
| 1024 | 0.05 | 2.7e6 | 89.0 | 14.8 | 98.0 | 48.2 | Gustavson |
| 1024 | 0.1 | 1.1e7 | 367 | 29.7 | 98.4 | 51.3 | Gustavson |
| 1024 | 0.2 | 4.3e7 | 1421 | 84.1 | 98.6 | 51.7 | Gustavson |
| 1024 | 0.5 | 2.7e8 | — | 456 | 105 | 53.4 | dense |
| 1024 | 1 | 1.1e9 | — | 1898 | 114 | 57.9 | dense |

The full table, with all nine densities per size, is in the run output.

## Findings

1. **The dense–Gustavson crossover is near 20% density.** Gustavson's cost grows with the number of products, `d²·n³`, at about 1.8 ns each. The dense algorithm's cost is fixed at `n³` multiply-adds at GEMM speed (about 37 GFLOP/s here, on one core), plus `O(n²)` scatter and emission. They meet where `d² ≈` the ratio of the two per-operation costs, about 0.04, so `d ≈ 0.2`. The crossover barely moves with `n`, because both costs scale with `n³`.
2. **Use the hash join only where Gustavson can't run.** The hash join with hash aggregation was 2.5–17× slower than Gustavson's algorithm. It pays for a hash lookup per product, where Gustavson indexes an array. Gustavson needs the free dimension `F_B` to have a known extent; the hash variant is the fallback when it doesn't.
3. **Tracking group existence costs about 2×.** Knowing which `(i, j)` groups some product reached needs a second GEMM over 0/1 indicators. An Exact fact that both inputs are complete makes every group exist, and halves the dense cost. This is a concrete payoff for density facts (§8.1).
4. **Thresholds for §10.4.** In the einsum form, switch to dense at about 20% density and back to sparse below it. Staudt et al.'s 5% is close to the crossover against a hash-based sparse algorithm (5–10% here), which is the right threshold for a host that runs only the relational form.
5. **The relational form pays the hash price.** A host running einfold's relational form runs the "hash" column at best. That is why two-operand contractions need the einsum form (§7.2), and why a dense input above 5–10% density is already faster as dense.

## Limitations

- Single-threaded CPU only. A multithreaded GEMM shifts the crossover further toward sparse, since GEMM parallelizes almost perfectly.
- Uniform random density. Structured sparsity (blocks, bands, masks) favors the block-sparse algorithm, which was not measured.
- Square matrices only; no batch dimensions.
- `matrixmultiply` is a portable GEMM, well below a vendor BLAS. A faster GEMM also moves the crossover toward sparse.
