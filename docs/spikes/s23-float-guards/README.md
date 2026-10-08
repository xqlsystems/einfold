<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S23: float guards for eager aggregation

**Question.** When does eager aggregation change a floating-point result by more than rounding, and what does a guard that rules this out cost?

Eager aggregation sums a table over a dimension no later step needs, before the join: `SUM(a.v * b.v) … GROUP BY g` becomes `SUM(a2.v * b.v)` over `a2 = SELECT k, SUM(v) FROM a GROUP BY k`. Over the reals the two are equal, because multiplication distributes over addition. Design §8.6 promises float results "within the bound that any order of the same additions has", and that "NaN and infinities must agree exactly".

**Answer.** Eager aggregation changes finiteness in both directions, on both hosts:

* an exact 0 becomes NaN;
* a NaN becomes 0;
* a finite 5e307 becomes inf.

The hosts' own plain `SUM` also changes finiteness with the partitioning, even with one partition, so "NaN and infinities must agree exactly" cannot hold even without rewrites.

A simple guard makes eager aggregation safe:

* every input is finite;
* each pre-aggregated partial sum's absolute sum is below the overflow threshold;
* each output group's sum of |a·b| is too.

With it, both plans are finite and within the summation bound in all 122,987 random groups where it held. Checking it at run time costs 2.4 ms against 43 ms for the contraction it guards. Parquet statistics reveal infinities but not NaN.

Two related defects in the design's algebra:

* `VAR` decomposed as Σx² − (Σx)²/n returns 7,550 where the true value is 0.083.
* The online-softmax merge returns NaN for two empty states.

Date: 2026-10-07. Intel Core i7-8700 (6 cores, 12 threads), 15 GB RAM, DuckDB 1.5.6, DataFusion (Python) 54.1.0, NumPy, PyArrow.

## Method

[`float_guards.py`](float_guards.py) has three parts.

**1. Counterexamples.** Each case is a table `A(j, k, v)`, with several `j` for one `k`, and `B(k, v)` with one row. Both plans are written as SQL and run on DuckDB (single thread) and DataFusion (one partition):

```sql
-- original
SELECT SUM(a.v * b.v) FROM A a JOIN B b ON a.k = b.k
-- eager: A pre-aggregated over j
WITH a2 AS (SELECT k, SUM(v) AS v FROM A GROUP BY k) SELECT SUM(a2.v * b.v) FROM a2 JOIN B b ON a2.k = b.k
```

It also runs:

* one plan, `SUM(v)` over {x, x, −x, −x} with x = 1e308, under different partitionings;
* `VAR_POP` against the decomposition RFC 0001 proposes, on 100,000 values 1e9 + U[0, 1);
* the online-softmax merge `(m, d) ⊕ (m′, d′) = (max, d·e^(m−max) + d′·e^(m′−max))` on two empty states (−∞, 0).

**2. The guard.** For one output group, the original plan adds the terms `aᵢ·b_s(i)` in some order. The eager plan first sums each sub-group's `aᵢ`, then adds `A_s·b_s` over the sub-groups. Let n be the number of terms and u = 2⁻⁵³. The guard is:

* (G0) every aᵢ and b_s is finite;
* (G1) Σᵢ |aᵢ·b_s(i)| · (1 + 4nu) < MAX, the largest double;
* (G2) for every sub-group s, Σ_{i∈s} |aᵢ| · (1 + 4nu) < MAX.

*Why it suffices.* Every partial sum in the original plan is bounded in magnitude by Σ|aᵢ·b_s(i)|·(1+u)^(n+1). That stays finite by G1. Every partial sum inside a sub-group is bounded by Σ_{i∈s}|aᵢ|·(1+u)^n, finite by G2. Each eager product and partial sum of products is bounded by the G1 quantity. So neither plan overflows. Without overflow, the standard analysis of recursive summation (Higham, *Accuracy and Stability of Numerical Algorithms*, chapter 4) bounds each plan's error by γ_(n+2)·Σ|aᵢ·b_s(i)|, with γ_k = ku/(1−ku), plus an absolute allowance for gradual underflow.

The property test generates random groups:

* 1–8 terms in 1–4 sub-groups;
* random signs, and exact zeros with probability 0.1;
* magnitudes log-uniform, in one of two ranges: everywhere (1e-310 to 1e308), or near overflow (a in 1e300 to 1e308, b in 1e-10 to 1e10).

Each plan is evaluated in Python doubles, and compared exactly, with `fractions.Fraction`, against the exact sum and the bound.

A sufficient form from statistics: G2 holds if (rows of A) · max|a| · (1 + 4nu) < MAX, and G1 if (joined rows) · max|a| · max|b| · (1 + 4nu) < MAX. Both use per-chunk maxima, which Parquet and the readers' statistics provide.

**3. Cost.**

* What Parquet row-group statistics (written by PyArrow) show for a column holding inf and NaN.
* The median of 5 timings, on ddx's 800,000-value `matmul` operand (n = 50,000), of:
  * the contraction `a·w` in DataFusion;
  * a finiteness and `max|v|` check in DataFusion SQL;
  * the same check in NumPy, next to a NumPy `sum`.

Run with:

```
uv run --with duckdb --with datafusion --with numpy --with pyarrow python float_guards.py
```

## Results

### 1. Counterexamples

| case | exact | DuckDB original | DuckDB eager | DataFusion original | DataFusion eager |
|---|---|---|---|---|---|
| partial sum overflows, B is 0 | 0 | 0 | NaN | 0 | NaN |
| products overflow, partial sum cancels | 0 | NaN | 0 | NaN | 0 |
| partial sum overflows, then shrinks | 5e307 | 5e307 | inf | 5e307 | inf |

One plan, `SUM(v)` over {x, x, −x, −x}, x = 1e308, on DataFusion:

| batches (one per partition) | `SUM(v)` |
|---|---|
| [x, −x, x, −x] | NaN |
| [x, x] · [−x, −x] | NaN |
| [x, x, −x, −x] | 0 |
| [−x, −x, x, x] | 0 |

`VAR_POP` of 100,000 values 1e9 + U[0, 1), true value 0.083202:

| host | engine's `VAR_POP` | Σx² − (Σx)²/n |
|---|---|---|
| DuckDB | 0.083202 | 7549.75 |
| DataFusion | 0.083202 | 503, 671 and 839 on three runs |

Online-softmax merge of two empty states, which should give (−∞, 0): both hosts return (−∞, NaN).

### 2. The guard

| magnitudes | groups | guard held | of those: a plan non-finite or outside the bound | guard failed | of those: finiteness differs | both finite, eager outside the bound | both non-finite |
|---|---:|---:|---:|---:|---:|---:|---:|
| a, b in 1e-310 … 1e308 | 100,000 | 70,089 | 0 | 29,911 | 0 | 0 | 29,911 |
| a in 1e300 … 1e308, b in 1e-10 … 1e10 | 100,000 | 52,898 | 0 | 47,102 | 185 | 0 | 46,763 |

### 3. Cost

Parquet (PyArrow) row-group statistics for v = [1, inf, 2, NaN, −3]: min −3, max inf, null count 0. NaN leaves no trace.

| pass over ddx's 800,000-value operand (n = 50,000) | ms |
|---|---:|
| DataFusion: the contraction `a·w` | 42.6 |
| DataFusion: finiteness and max\|v\| check, as SQL | 2.4 |
| NumPy: `sum` | 0.21 |
| NumPy: `isfinite().all()` and `abs().max()` | 0.89 |

## Findings

1. **Eager aggregation changes finiteness, in both directions, on both hosts.**
   * A pre-aggregated partial can overflow where the original never forms it: 0 becomes NaN, and 5e307 becomes inf.
   * The original's products can overflow where the eager plan cancels first: NaN becomes 0.

   Design §8.6's bound assumes no overflow, so it says nothing about these.
2. **The hosts' own sums change finiteness with the partitioning.** DataFusion sums {x, x, −x, −x} to NaN or 0 depending on how the rows are split. With one partition, [x, −x, x, −x] gives NaN while [x, x, −x, −x] gives 0, apparently because its accumulator adds alternate values in separate lanes. So "NaN and infinities must agree exactly" with the host's plan can't be promised even for pure reordering. Promise instead what the guard gives: if the inputs are finite and the guard holds, both plans are finite and within the bound.
3. **The guard is sufficient and cheap.** In 122,987 random groups where it held, both plans were finite and within the bound, every time. Where it failed near overflow, 185 groups differed in finiteness, so it is not vacuous. A sufficient form needs only per-chunk maxima and row counts. Parquet's statistics give the maximum, including inf, but not NaN, so finiteness needs a reader fact or a check. The check costs 2.4 ms in DataFusion against a 43 ms contraction (6%). In a native kernel it can be fused into the scatter pass that positional kernels already make.
4. **Algebraic decomposition is not exact over floats.** RFC 0001 says level-3 aggregates "can be computed exactly" from semiring folds, and that `VAR` and `STDDEV` "decompose similarly". On values near 1e9, Σx² − (Σx)²/n returns 7,550, and on DataFusion 503–839 from run to run, where both engines' `VAR_POP` returns 0.0832. Decompose `VAR` into the parallel (Welford, Chan) merge of (count, mean, M2) instead, which is a fold but not a semiring fold. Treat the finish function as part of the error analysis.
5. **The online-softmax fold needs its identity special-cased.** Merging two empty states gives (−∞, NaN) on both hosts. Its merge must test for m = −∞ before subtracting.

## Limitations

- The guard is argued in prose and tested on random single groups in Python doubles, not on whole plans inside a host.
- The check's cost is measured as separate passes; fusing it into a kernel was not measured.
- One machine. Single-threaded DuckDB and one-partition DataFusion for the counterexamples, so that the original plans' order is fixed.
