<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S5: gpudb shapes

**Question.** Which relational-form shapes does `gpudb` fuse on the GPU, measured on matrix multiplication and attention? How should einfold handle gpudb's rule for float sums (design doc §8.6)?

**Answer.** Today, gpudb accelerates almost none of einfold's relational-form shapes. On a Colab T4, 1 of 13 shapes ran on the GPU: a `BIGINT` reduction, 4.3× faster than DuckDB. Every other shape was declined, for one of four reasons:

- the sum is over `DOUBLE`, which gpudb never rewrites;
- a many-to-many join expands too much;
- an operand is a subquery rather than a base table;
- a window function.

gpudb returned the same results as DuckDB everywhere, and its float sums were exactly as repeatable as DuckDB's: rarely. So einfold's relational form gains nothing on gpudb for float workloads (ddx, geoscience). Speedups there need gpudb to adopt the einsum form, or to carry `DOUBLE` sums on the GPU.

**Status: done.** This machine's GPU (GTX 1080 Ti, compute capability 6.1) is below gpudb's minimum. The GPU half ran on a free Colab T4 (compute capability 7.5) with [`s05_s08_colab.ipynb`](s05_s08_colab.ipynb); its raw output is [`colab-t4-results.json`](colab-t4-results.json).

Date: 2026-10-06. gpudb 0.7.0 (CUDA build from `pip install duckdb-gpudb`); DuckDB 1.5.6; Colab T4 (16 GB), 2 vCPUs.

## What was learned without a usable GPU

1. **Hardware floor.** gpudb supports NVIDIA compute capability 7.5 and up (Turing, Ampere, Ada, Hopper). Its install guide says "Volta (sm_70) and older are not supported" because CUDA 13 dropped them from `nvcc`. Pascal cards such as the GTX 1080 Ti are out. Sirius (S15) has a similar floor.
2. **The community build is CPU-only on Linux.** `INSTALL gpudb FROM community` gives `compiled=cpu runtime=cpu`. The CUDA build comes from `pip install duckdb-gpudb` (a wheel carrying the CUDA extension), or a source build.
3. **How gpudb decides.** gpudb rewrites the SQL statement *before* DuckDB plans it, in its Python client (`gpudb.connect`), and reports each decision through `last_rewrite()`. Probing einfold's shapes with [`shapes.py`](shapes.py) showed the order of its checks:
   - first, a size floor: no table of at least 1,000,000 rows means no rewrite (`reason: threshold`);
   - next, the backend: a CPU build declines everything (`reason: backend`) before it looks at the shape.

   So which shapes it accepts can only be seen with a GPU.
4. **Consequences for einfold.**
   - gpudb is a *statement* rewriter, so it sees einfold's SQL-to-SQL output, not a plan. einfold's relational form must therefore be emitted as SQL that gpudb's recognizer matches. Spike S7 showed that unparsed DataFusion plans keep the shapes einfold wrote.
   - gpudb's 1M-row floor means small einsums (ddx's weight gradients) stay on DuckDB whatever einfold emits.
   - Like einfold, gpudb declines rather than risk a wrong answer: its documentation claims identical answers on both paths. einfold's target profile for gpudb should record its floor and its per-shape decisions, measured by the notebook.

## The GPU half (Colab)

[`s05_s08_colab.ipynb`](s05_s08_colab.ipynb): open it in Colab, choose a T4 runtime, and run all cells (10–15 minutes). It:

- builds an 8-million-row matrix `a(i, k)` and smaller operands, including attention's `Q` and `K`;
- runs 13 einfold shapes through gpudb and through plain DuckDB: matrix multiplication, matrix-vector product, eager aggregation, CTE contraction order, partial aggregate with matched count, causal mask, reductions over `DOUBLE`, `BIGINT` and `DECIMAL`, retiling key, global sum, attention scores `Q·Kᵀ`, and window maximum;
- records for each whether gpudb rewrote it, why or why not, the speedup, and whether the results match;
- runs four float and decimal sums 10 times on each engine and counts distinct results, which tests gpudb's float-sum rule;
- compiles and runs spike S8's GPU kernels ([`../s08-deterministic-sums/gpu.cu`](../s08-deterministic-sums/gpu.cu)) on the T4;
- prints a JSON report at the end.

The notebook's code was smoke-tested on this machine with gpudb's CPU backend: every cell runs, and every shape is declined, as expected without a GPU.

## Results (Colab T4)

Times are medians of 3 runs. "Native" is plain DuckDB on the same data.

| Shape | Native (ms) | gpudb (ms) | Rewritten? | gpudb's reason |
|---|---|---|---|---|
| Matrix product (join + `GROUP BY` + `SUM`) | 13,428 | 13,351 | no | join expands: "512000000 rows from tables of at most 8000000" |
| Matrix-vector (join onto a unique key) | 146 | 147 | no | `DOUBLE` |
| Eager aggregation (pre-aggregated subquery) | 142 | 148 | no | "join leaf is SUBQUERY, not a base table" |
| Contraction order as a CTE | 13,270 | 13,268 | no | join expands |
| Partial aggregate with matched count | 161 | 160 | no | `DOUBLE` |
| Mask operand (causal predicate) | 7,036 | 6,450 | no | join expands |
| Reduction over `k`, `DOUBLE` | 93 | 95 | no | `DOUBLE` |
| **Reduction over `k`, `BIGINT`** | **90** | **21** | **yes (4.3×)** | resident `GROUP BY` |
| Reduction over `k`, "`DECIMAL`" (see below) | 94 | 92 | no | `DOUBLE` |
| Retiling partition key | 237 | 245 | no | `DOUBLE` |
| Global sum | 26 | 26 | no | `DOUBLE` |
| Attention scores `Q·Kᵀ` | 2,189 | 2,444 | no | tables under 1M rows |
| Window maximum (softmax) | 6,084 | 5,687 | no | window function |

All 13 results matched DuckDB's.

**A notebook bug.** The "`DECIMAL`" column was really `DOUBLE`: in DuckDB, dividing a `DECIMAL` by an integer returns a `DOUBLE`. The notebook is fixed (no division), but the `DECIMAL` case has not been rerun. Since gpudb sums `DECIMAL` on the GPU as scaled integers by its own account, a rerun should show it rewritten.

**Repeatability**, as distinct results in 10 runs of the same query:

| Query | gpudb | DuckDB |
|---|---|---|
| Grouped `DOUBLE` sum | 9 | 9 |
| Global `DOUBLE` sum | 7 | 8 |
| Grouped "`DECIMAL`" sum (actually `DOUBLE`) | 8 | 7 |
| Matrix product, `DOUBLE` | 10 | 10 |

None of these was rewritten, so both columns measure DuckDB's own parallel float sums. They confirm spike S19 on a 2-vCPU machine: almost every run gives different bits.

## Findings

1. **For gpudb, einfold's float workloads are out of reach today.** gpudb never rewrites a `DOUBLE` sum, by design, because it cannot match DuckDB's bits. Every ddx and geoscience einsum sums `DOUBLE`s. So the relational form gets no GPU help from gpudb. Only exact types do (the `BIGINT` reduction, 4.3×).
2. **gpudb only matches base-table leaves.** einfold's eager aggregation emits joins against pre-aggregated subqueries (design §10.1), which gpudb declines. A gpudb target profile must ask for base tables, which in practice means materializing intermediate results as tables.
3. **Many-to-many joins are declined on size.** A matrix product's join produces `N·D·H` rows, which gpudb refuses above its limit. That is exactly the problem EinFold solves (design §3, problem 1). gpudb's fused join-aggregate covers key/foreign-key joins, not contractions.
4. **What would change this.** gpudb carrying `DOUBLE` sums with an order-independent accumulator. Spike S8 measured a binned sum at 3.6× the cost of an atomic sum on this same T4, deterministic and accurate, which also answers gpudb's reason for declining. Or gpudb adopting the einsum form. Both belong to M6 (design §16), and are worth raising with gpudb's maintainer.
5. **For the target profile:** a 1M-row floor; base-table leaves only; no window functions; a join-expansion limit; exact types only on the GPU.
