<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S8: the cost of deterministic sums

**Question.** What does a deterministic floating-point sum cost, compared with a plain sum, on CPU and GPU? Compare a fixed combine order, order-independent accumulators, and the precision levels (design doc §8.6).

**Answer.** A **binned reproducible sum** is the practical choice. It is order-independent, so it gives the same bits in any order, on any number of threads, and even with GPU atomics. It is also more accurate than a plain sum: on this data it equaled the correctly rounded sum.

| | Binned sum vs. plain | Notes |
|---|---|---|
| CPU, one long sum, 12 threads | 3.0× slower than a parallel plain tree | faster than a single-threaded plain sum |
| CPU, grouped sums | 4.4× slower | 24 bytes of state per group |
| GPU (GTX 1080 Ti) | 3.5× slower | consumer GPU, with slow float64 |

It needs the largest absolute value in advance. Facts can supply it: per-chunk maximum statistics, or `layout:summaries`. Otherwise it costs a second pass.

- An exact **superaccumulator** is competitive for one long parallel sum (2× a parallel plain tree). It is impractical for grouped sums, at 576 bytes of state per group, 11× slower; and on the GPU it is 83× slower.
- A **fixed combine order** costs nothing, but is deterministic only for a fixed input order, which hosts don't guarantee (spike S3).

Date: 2026-10-06. CPU: Intel Core i7-8700 (6 cores, 12 threads), Rust 1.91. GPU: NVIDIA GTX 1080 Ti, CUDA 12.8.

## Accumulators

| Name | What it does | Deterministic? |
|---|---|---|
| plain | left-to-right `f64` sum | only for one fixed order |
| Kahan (Neumaier) | compensated sum: tracks the rounding error of each addition | not in general (S19 saw DuckDB's Kahan-based `fsum` vary in parallel) |
| fixed tree | fixed 64 Ki-element blocks, each summed in order, then combined pairwise in block order | for a fixed input order, at any thread count |
| binned | after Demmel and Nguyen (2013): each value is split onto 3 fixed grids derived from `max|x|` and `n`. Each grid's sum is exact, so the total does not depend on order | yes, in any order |
| superaccumulator | an exact fixed-point accumulator over the whole `double` range, 72 limbs | yes, in any order |
| fsum (Shewchuk) | Python's `math.fsum`: exact and correctly rounded. The reference | yes |

## Method

- **CPU** ([`src/main.rs`](src/main.rs)):
  - *One long sum:* 50 million doubles spanning 16 orders of magnitude, with mixed signs, as in S19.
  - *Grouped sums:* 64 million values in 1 Mi groups, arriving in random group order, as a hash aggregation sees them.
  - Timings are medians. Determinism is checked by summing three shuffled orders on 1, 4 and 12 threads, and comparing bits.
- **GPU** ([`gpu.cu`](gpu.cu)): the same 50 million values, 20 runs of each kernel.
  - *atomic:* per-block tree sums combined with `atomicAdd`, in whatever order blocks finish.
  - *fixed tree:* block sums written out, then summed in a fixed pattern.
  - *binned:* per-thread level sums combined with `atomicAdd`. That is exact on each grid, so order cannot matter.
  - *superaccumulator:* 64-bit integer atomics.

Run with `cargo run --release`, and `nvcc -O3 -arch=sm_61 gpu.cu -o gpu && ./gpu` (use `sm_75` for a T4). The Colab notebook in [`../s05-gpudb`](../s05-gpudb/s05_s08_colab.ipynb) runs the GPU half on a T4.

## Results

### CPU, one long sum

| Accumulator | Threads | ms | vs. plain, 1 thread | Relative error | Same bits under shuffles and thread counts |
|---|---|---|---|---|---|
| plain | 1 | 48.2 | 1.0× | 3.6e-11 | no (4 results) |
| Kahan (Neumaier) | 1 | 72.0 | 1.5× | 0 | yes, on this data |
| fixed tree | 1 | 48.7 | 1.0× | 5.1e-14 | no (2 results): input order changes it |
| fixed tree | 12 | 11.9 | 0.25× | 5.1e-14 | no (2 results) |
| binned, computing the maximum (2 passes) | 1 | 125.5 | 2.6× | 0 | yes |
| binned, computing the maximum (2 passes) | 12 | 77.2 | 1.6× | 0 | yes |
| binned, maximum from facts | 1 | 94.4 | 2.0× | 0 | yes |
| binned, maximum from facts | 12 | 35.9 | 0.74× | 0 | yes |
| superaccumulator | 1 | 144.7 | 3.0× | 0 | yes |
| superaccumulator | 12 | 24.1 | 0.50× | 0 | yes |
| fsum (Shewchuk) | 1 | 1127.5 | 23× | 0 | yes |

A relative error of 0 means the result equaled the correctly rounded sum.

### CPU, grouped sums (1 Mi groups × 64 values, single thread)

| Accumulator | State per group | ms | vs. plain | Max relative error | Same bits after a shuffle |
|---|---|---|---|---|---|
| plain | 8 B | 193 | 1.0× | 1.5e-15 | no |
| Kahan (Neumaier) | 16 B | 740 | 3.8× | 0 | yes, on this data |
| binned, maximum from facts | 24 B | 855 | 4.4× | 0 | yes |
| superaccumulator | 576 B | 2137 | 11× | 0 | yes |

### GPU (GTX 1080 Ti)

| Accumulator | ms | GB/s | vs. atomic | Distinct results in 20 runs |
|---|---|---|---|---|
| atomic (scheduler order) | 1.06 | 377 | 1.0× | **6** |
| fixed tree | 1.05 | 380 | 1.0× | 1 |
| binned, maximum from facts | 3.67 | 109 | 3.5× | 1 |
| superaccumulator | 87.9 | 5 | 83× | 1 |

The GPU's binned and superaccumulator results equal the CPU's correctly rounded sum, bit for bit.

## Findings

1. **Use a binned sum for `highest` precision and for determinism in einfold's executors.** It is order-independent, so it is deterministic under any parallel schedule, including GPU atomics. On this data it was also correctly rounded, where the plain sum's relative error was 3.6e-11. In parallel it costs 3× a plain parallel sum, and 4.4× in grouped form, with 24 bytes of state per group. That makes it the first choice in §8.6, item 4.
2. **Facts make it one pass.** The binned sum needs `max|x|` before it starts. Computing it takes a second pass, which doubled the parallel cost (77 ms vs. 36 ms). Per-chunk maximum statistics (`layout:summaries`, §8.1) give it for free. For grouped sums, one global maximum suffices, at some cost in accuracy for groups of small values.
3. **A superaccumulator suits one big sum, not many small ones.** It parallelizes well (24 ms on 12 threads), but its 576-byte state makes it 11× slower for hash aggregation, and its many atomics make it 83× slower on a GPU.
4. **A fixed order is not enough on hosts.** The fixed tree is free, but it depends on input order. Spike S3 found that neither DataFusion nor DuckDB guarantees input order through a parallel scan and aggregation. It remains useful inside einfold's own executor, where einfold controls the order.
5. **Kahan matched here, but is not a determinism fix.** Neumaier's compensated sum gave the same bits on every order of this data, because its error term absorbed all rounding. Nothing guarantees that in general, and S19 saw DuckDB's Kahan-based `fsum` vary in parallel.
6. **On a GPU, deterministic doesn't have to mean slow.** The fixed tree costs nothing, and the binned sum's 3.5× is on a consumer card whose float64 throughput is 1/32 of its float32. Data-center GPUs run float64 far faster, so the binned sum should come out closer to bandwidth-bound there. The Colab T4 run (also slow float64) and a later data-center GPU would confirm.

## Limitations

- One data distribution (16 orders of magnitude, random signs). Sums with heavy cancellation stress the binned sum's 3 levels more. Its accuracy degrades gracefully, but it stays deterministic.
- The binned sum here is a simplified 3-level form, not ReproBLAS itself.
- Grouped sums were measured single-threaded only.
- No float32 inputs, and no `fast` precision level (bfloat16 or TF32), which only a host's kernels offer.
