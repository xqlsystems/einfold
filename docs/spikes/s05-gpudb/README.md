<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S5: gpudb shapes

**Question.** Which relational-form shapes does `gpudb` fuse on the GPU, measured on matrix multiplication and attention? How should einfold handle gpudb's rule for float sums (design doc §8.6)?

**Status: partly done; the GPU half is ready to run on Colab.** This machine's GPU (GTX 1080 Ti, compute capability 6.1) is below gpudb's minimum. The notebook [`s05_s08_colab.ipynb`](s05_s08_colab.ipynb) runs the GPU half on a free Colab T4 (compute capability 7.5).

Date: 2026-10-06. gpudb 0.7.0; DuckDB 1.5.6.

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

Results: pending.
