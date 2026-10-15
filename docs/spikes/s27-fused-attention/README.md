<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S27: fused attention, demo 1's room

**Question.** Demo 1 ([`docs/demos.md`](../../demos.md)) asks einfold to find FlashAttention's plan for causal attention written in plain SQL. How do the hosts run that SQL today, in time and memory? What does a fused, tiled kernel with an online softmax give on the same rows? Does it match SQL on NaN and infinities?

**Answer.**

* **The hosts struggle.**
  * DataFusion takes 1.8 s at L = 1,024 tokens and 53 s at L = 4,096, and fails at L = 8,192: out of its 6 GB memory pool in one spelling, past 15 minutes in the other.
  * DuckDB takes 1.0 s at L = 1,024 and 28 s at L = 8,192, and runs out of memory (6 GB, plus 4 GB of spill) at L = 16,384.
  * Both hold the N² scores, as hash-aggregate groups.
* **The fused kernel is 280–390× faster than DuckDB, and 500–2,000× faster than DataFusion,** from rows to rows: 3.4 ms at L = 1,024, and 96 ms at L = 8,192.
  * It runs at L = 65,536 in 5.1 s.
  * Its memory is the inputs and the output, linear in L, plus one 64×64 tile of scores per thread.
  * The causal mask lets it skip 47–50% of key tiles.
* **It is more accurate than the host, and deterministic.** Against a compensated reference, its error is 1.3–5.1e-15 of the largest output, DataFusion's 0.6–8.7e-14, including on scores spread 40× wider. Its bits are identical across 1, 2, 5 and 12 threads.
* **Masks are absent rows.** The first version of the kernel gave masked pairs a weight of 0 inside partly masked tiles. A NaN in `v` then turned 5 output rows into NaN that SQL computes as numbers (`0 · NaN` is NaN). Skipping masked pairs fixed it. Then the kernel matched DataFusion on every special-value case: NaN and ±∞ in each input, 1e300, and a row whose scores are all −∞.

Date: 2026-10-15. Intel Core i7-8700 (6 cores, 12 threads), 15 GB RAM, Rust 1.91, DataFusion 54.1.0, DuckDB 1.4.3, `matrixmultiply` 0.3.

## Method

[`src/main.rs`](src/main.rs) builds `q`, `k` and `v`, each `(t, d, x)` with L tokens and d = 64 features, and values `sin(0.7·r + phase)`. The query is demo 1's causal attention, with scores scaled by 1/8 and a softmax over keys, in two spellings:

* **window:** `exp(s − MAX(s) OVER (PARTITION BY i))`, as demo 1 writes it;
* **groupby:** `MAX(s) … GROUP BY i` joined back to the scores, as ddx writes it. ddx is an XQL Systems project for differentiating SQL queries.

| method | what it does |
|---|---|
| DataFusion | the SQL, tables in memory in 12 partitions, a 6 GB memory pool and 4 GB of spill |
| DuckDB | the SQL in DuckDB's CLI, as `CREATE TEMP TABLE out AS …`, 12 threads, `memory_limit = 6GB`, 4 GB of spill |
| fused | from `(t, d, x)` rows: scatter to positions, then for each tile of 64 queries (one thread per tile), stream key tiles of 64 in order, skipping tiles the causal mask removes. Compute the tile's scores with a GEMM, keep a running maximum, normalizer and weighted sum per query, rescaled when the maximum grows, then write `(t, d, x)` rows |

Each host measurement is one process under GNU `time`, which reports its peak resident memory. The fused kernel's time is the median of 3 runs. The checksum, the sum of the absolute values of the outputs, agreed across all methods wherever they finished.

`accuracy <L> <scale>` compares the kernel and DataFusion with a reference that uses compensated sums and the exact row maximum. `scale` multiplies `q`, which spreads the scores and forces more rescalings. `edges` plants special values at L = 128 and compares the kernel, with masked pairs skipped and with them zeroed, against DataFusion row by row.

Run with `nix-shell -p duckdb time --run ./run.sh`, after `cargo build --release`.

## Results

Times in ms; peak resident memory in MiB in parentheses.

| L | DataFusion, window | DataFusion, groupby | DuckDB, window | DuckDB, groupby | fused, 1 thread | fused, 12 threads | fastest host ÷ fused |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 1,770 (628) | 2,780 (697) | 954 (150) | 1,067 (347) | 9.8 | 3.4 (23) | 281× |
| 2,048 | 9,653 (1,838) | 15,106 (2,089) | 3,328 (321) | 3,514 (728) | 33.3 | 8.9 (32) | 374× |
| 4,096 | 52,766 (6,082) | 86,602 (6,737) | 10,529 (835) | 11,305 (2,573) | 120.9 | 27.0 (49) | 390× |
| 8,192 | past 15 min (8,204 when stopped) | out of memory | 28,416 (2,239) | 38,992 (6,128) | 463.3 | 95.6 (83) | 297× |
| 16,384 | not run | not run | out of memory (5,757) | out of memory (5,997) | 1,838 | 334 (151) | — |
| 32,768 | | | | | | 1,293 (303) | — |
| 65,536 | | | | | | 5,059 (559) | — |

The fused kernel visited 136 of 256 key tiles at L = 1,024, and 524,800 of 1,048,576 at L = 65,536: half, plus the diagonal. Its memory is the row arrays, their dense copies and the output, all linear in L. Its working set per thread is about 64 KiB.

DataFusion's resident memory exceeded its 6 GB pool, since the pool tracks only what operators reserve.

**Accuracy** (L = 2,048), as the largest error divided by the largest output:

| scores scaled by | fused | DataFusion | fused bits across 1, 2, 5, 12 threads |
|---:|---:|---:|---|
| 1 | 1.34e-15 | 5.70e-15 | identical |
| 10 | 3.66e-15 | 1.62e-14 | identical |
| 40 | 5.05e-15 | 8.73e-14 | identical |

**Special values** (L = 128, 8,192 output rows), as output rows that differ from DataFusion's:

| case | NaN rows in DataFusion | fused, masked pairs skipped | fused, masked pairs zeroed |
|---|---:|---:|---:|
| none | 0 | 0 | 0 |
| NaN in `q` | 64 | 0 | 0 |
| NaN in `v` | 123 | 0 | **5** |
| +∞ in `q` | 64 | 0 | 0 |
| −∞ in `q` | 64 | 0 | 0 |
| 1e300 in `k` | 0 | 0 | 0 |
| a row whose scores are all −∞ | 64 | 0 | 0 |

## Findings

1. **Demo 1's room is two to three orders of magnitude, and a memory wall.** S21 found 13–48× for attention's two products taken separately. Fusing the whole softmax pipeline gives 280–390× over the faster host. Both hosts also stop at a sequence length the fused kernel handles in a third of a second.
2. **The hosts' cost is the N² intermediate, as demo 1 predicts.** Both keep every score group in a hash aggregate, then window or group and join it again. DataFusion spends more time and memory than DuckDB in both spellings. ddx's `GROUP BY` spelling is slower than the window on both hosts.
3. **The online softmax is an accumulator, and an accurate one.** Rescaling a running maximum made the result *more* accurate than DataFusion's, which adds in no fixed order, and it stayed deterministic. Its error stayed near 1e-15 as the scores spread 40× wider. That supports treating the maximum as numerics (design §8.6) rather than as algebra.
4. **Masks are absent rows, and a naive dense kernel gets them wrong.** Zeroing masked pairs inside a partly masked tile is the natural GEMM implementation, and it is wrong in SQL terms whenever a masked value is NaN. This is the same existence issue S25 found for missing rows. The design must state it for masks too.
5. **Special values match without special handling,** once masked pairs are skipped. NaN and +∞ scores make their rows NaN in both, and so does a row whose scores are all −∞: SQL computes `exp(−∞ − (−∞))`, the kernel `0 / 0`.

## Limitations

* One machine, one data distribution, d = 64, causal mask only.
* The kernel was timed outside DataFusion, from Rust vectors. A real operator also pays for collecting Arrow batches and for the memory pool.
* Positions are the coordinates, and inputs are complete: the kernel assumes facts einfold must prove.
* NULLs, absent rows other than masked ones, and duplicate keys were not tested.
* The accuracy reference uses compensated sums, not exact arithmetic.
