<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S22: ddx's gradient plans

**Question.** Where does the time go in the gradients ddx emits, and what would einfold have to match to speed them up? ddx is an XQL Systems project for automatic differentiation of SQL queries: from a query computing a loss, it builds a *program* of SQL steps, each materialized as a table, that computes the loss and its gradient. Spike S21 timed the matrix products of a gradient as plain two-table queries. The design (§3) attributes ddx's slow gradients to those products ("the gap is the hash joins").

**Answer.** The products S21 timed are a small part of the gradient. In every gradient step of ddx's `matmul`, a two-layer network, and attention, ddx:

* rebuilds the forward join, only to test whether the forward product was NULL;
* joins that recomputed join to the cotangent;
* aggregates;
* left-joins back to the input table to fill missing groups.

Steps that a "two-table `SUM` of products" detector, einfold's first milestone, would match are **13–32% of step time**.

On 300 random tables with NULLs, NaNs, missing rows and duplicate keys, the NULL test is **redundant**. A two-table contraction followed by ddx's own left join gives the same gradient in every case. For Ā in `matmul` at n = 50,000:

| form | time |
|---|---:|
| ddx's step | 680 ms |
| the two-table form | 186 ms |
| a dense kernel that writes every output position (and so needs no fill) | 10 ms |

Date: 2026-10-07. Intel Core i7-8700 (6 cores, 12 threads), 15 GB RAM, Rust 1.91, DataFusion 54.1.0, ddx at its public `main`, [c6657c9](https://github.com/xqlsystems/ddx/commit/c6657c97c1d45daffe5cd8d6339240b36037e9c7).

## Method

[`src/main.rs`](src/main.rs) has three parts.

**1. Profile.** Builds each workload's tables, as ddx's benchmark `crates/ddx-datafusion/tests/ad_perf.rs` does: complete coordinate tables with `sin`/`cos` values. It then times:

* the forward query;
* building the gradient program (`ad::grad`);
* running it (`ad::run`);
* each step, the best of 3, after a full run.

For each step it reads DataFusion's logical plan and counts:

* table scans;
* inner and left joins;
* aggregates;
* `CASE WHEN … IS NULL` tests inside an aggregate's input.

It also asks a detector modeled on the first milestone's: does some aggregate compute only `SUM`s over exactly one inner join of exactly two table scans, with no such test?

The workloads:

| workload | loss | with respect to |
|---|---|---|
| `matmul` (ddx's) | `SUM(tanh(a·w)²)`, a: n×16, w: 16×8 | a, w |
| `mlp2` | `SUM(tanh(tanh(x·w1)·w2)²)`, x: n×16, w1: 16×16, w2: 16×8 | w1, w2 |
| `attn` (ddx's) | single-head self-attention with scaling and a softmax over keys, then `SUM(o²)`; L tokens, d = 16 | wq, wk, wv |

**2. Guards.** For each of 300 random `matmul` instances, it runs ddx's program and compares each gradient with a hand-written form. Each instance has:

* 1–6 values of `s`, 1–4 of `k` and 1–4 of `o`;
* each row present with probability 0.8;
* values NULL with probability 0.15, NaN with 0.03, otherwise multiples of 1/8.

In a third of the cases only `w` is differentiated, and `a`, which is then constant data, repeats keys with probability 0.3. ddx checks the keys of tables it differentiates.

The hand-written form contracts ddx's own cotangent table with the other operand, then applies ddx's own left join and `CASE`:

```sql
-- Ā
SELECT a.s, a.k, CASE WHEN a.val IS NULL THEN NULL WHEN r.v IS NULL THEN 0.0 ELSE r.v END AS val
FROM a LEFT JOIN (SELECT c.s, w.k, SUM(c.val * w.val) AS v FROM cotangent c JOIN w ON c.o = w.o
                  GROUP BY c.s, w.k) r ON a.s = r.s AND a.k = r.k
```

W̄ is symmetric. Rows must match on their keys, with the same NULLs, NaN where NaN, and values within 1e-9 relative. Duplicate keys are compared as multisets.

**3. Filling groups.** For Ā at n = 50,000 it times:

* ddx's step;
* the two-table aggregate alone;
* the aggregate with ddx's left join;
* the same, with the cotangent spread over 12 partitions instead of ddx's one;
* a dense positional kernel. It streams the cotangent's rows, adds `v · W[o, :]` into output row `s`, and writes every `(s, k)` position as an Arrow row. Every position is written, so nothing needs filling (both inputs complete).

Run with `cargo run --release`. `S22_PARTS=23` runs only parts 2 and 3. Building it needs `protoc`, for ddx's Substrait dependency.

## Results

### 1. Profile

| workload | forward | build | run | run ÷ forward | steps the two-table detector matches |
|---|---:|---:|---:|---:|---:|
| matmul n = 10,000 | 34.4 ms | 1.5 ms | 255 ms | 7.4× | 13% of step time |
| matmul n = 50,000 | 144 ms | 1.4 ms | 1,399 ms | 9.7× | 13% |
| mlp2 n = 10,000 | 83.6 ms | 2.2 ms | 462 ms | 5.5× | 18% |
| attn L = 64 | 24.7 ms | 9.3 ms | 83.2 ms | 3.4× | 32% |
| attn L = 256 | 101 ms | 9.7 ms | 538 ms | 5.3× | 13% |

`matmul`, n = 50,000:

| step | ms | rows | scans | inner joins | left joins | aggregates | NULL tests | two-table `SUM` |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `saved_0` (forward z = a·w) | 141.9 | 400,000 | 2 | 1 | 0 | 1 | 0 | yes |
| `saved_1` | 1.6 | 1 | 1 | 0 | 0 | 1 | 0 | no |
| `value` | 0.3 | 1 | 1 | 0 | 0 | 0 | 0 | no |
| `cotangent_1` | 0.6 | 1 | 1 | 0 | 0 | 1 | 1 | no |
| `cotangent_0` (z̄) | 17.7 | 400,000 | 2 | 1 | 0 | 1 | 1 | yes |
| `grad_0_a` (Ā) | 596.7 | 800,000 | 4 | 2 | 1 | 1 | 1 | no |
| `grad_1_w` (W̄) | 515.9 | 128 | 4 | 2 | 1 | 1 | 1 | no |

`mlp2`, n = 10,000:

| step | ms | rows | scans | inner joins | left joins | aggregates | NULL tests | two-table `SUM` |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `saved_0` | 56.8 | 160,000 | 2 | 1 | 0 | 1 | 0 | yes |
| `saved_1` | 20.5 | 80,000 | 2 | 1 | 0 | 1 | 0 | yes |
| `saved_2`, `value`, `cotangent_2` | 2.0 | 1 | | | | | | no |
| `cotangent_1` | 4.7 | 80,000 | 2 | 1 | 0 | 1 | 1 | yes |
| `cotangent_0` | 123.0 | 160,000 | 3 | 2 | 0 | 1 | 1 | no |
| `grad_0_w1` | 190.1 | 256 | 4 | 2 | 1 | 1 | 1 | no |
| `grad_1_w2` | 56.7 | 128 | 4 | 2 | 1 | 1 | 1 | no |

`attn`, L = 256, steps over 5 ms:

| step | ms | rows | scans | inner joins | left joins | aggregates | NULL tests | two-table `SUM` |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `saved_2` (scores) | 18.5 | 65,536 | 2 | 1 | 0 | 1 | 0 | yes |
| `saved_6` (softmax-weighted values) | 18.1 | 4,096 | 4 | 3 | 0 | 1 | 0 | no |
| `region_0` | 226.4 | 1,048,576 | 5 | 4 | 0 | 0 | 0 | no |
| `cotangent_3` | 7.8 | 256 | 4 | 2 | 0 | 3 | 2 | no |
| `cotangent_2` | 35.6 | 65,536 | 6 | 3 | 0 | 4 | 4 | yes |
| `cotangent_1` | 94.8 | 4,096 | 3 | 2 | 0 | 1 | 1 | no |
| `cotangent_0` | 91.4 | 4,096 | 3 | 2 | 0 | 1 | 1 | no |
| `grad_0_wq`, `grad_1_wk`, `grad_2_wv` | 6.7–6.9 each | 256 | 4 | 2 | 1 | 1 | 1 | no |

### 2. Guards

300 random cases: **300 agree**, 0 refused, 0 disagree. 2,904 gradient rows compared, of which 503 NULL and 354 NaN.

### 3. Filling groups (Ā, n = 50,000)

| form | ms |
|---|---:|
| ddx's step (NULL test, recomputed join, aggregate, left join) | 680.4 |
| two-table aggregate only | 134.9 |
| two-table aggregate + ddx's left join and `CASE` | 185.6 |
| the same, cotangent in 12 partitions instead of 1 | 177.1 |
| dense fold writing every (s, k), 1 thread | 18.8 |
| dense fold writing every (s, k), 12 threads | 10.0 |

## Findings

1. **The gradient's time is not where §3 puts it.** In `matmul`, the two gradient steps take 1.1 s of the 1.4 s run. But each is a three-table fold, not the two-table product S21 timed (28 and 52 ms there). Each one:
   * rebuilds the forward join `a ⋈ w` (6.4 M rows), only to compute `a.val * w.val` for a test `CASE WHEN a.val * w.val IS NULL THEN NULL ELSE z̄ END`;
   * joins those rows to the 400,000-row cotangent on (s, o);
   * aggregates;
   * left-joins back to the input table.

   Building the program is 1.4 ms, so planning is not the cost.
2. **The NULL test is redundant, so the recomputed join is too.** The two-table form agreed with ddx on all 300 random cases, including NULLs, NaNs, missing rows, and duplicate keys in constant data. The reason:
   * a NULL in `a.val` or `w.val` already makes the contribution NULL, which `SUM` skips;
   * or it makes the whole group NULL, through the step's outer `CASE`, which tests the differentiated table's own value.

   With the test gone, nothing reads the recomputed join, and each gradient is the transposed contraction, as in JAX. That takes Ā from 680 ms to 186 ms. It is a change ddx can make without einfold.
3. **The fill is the dense kernel's natural output.** ddx's left join gives every input row a gradient, 0 where no group was reached. A dense positional kernel writes every position anyway. With inputs proved complete, one EinFold node replaces both the aggregate and the left join: 10 ms against 186 ms for the best SQL form, and 680 ms for ddx's step. The single-partition tables ddx materializes are not the bottleneck (177 ms with 12 partitions).
4. **The first milestone's detector misses most of the time.** Steps it matches are 13–32% of step time: the forward products and some cotangents. Every gradient step has a NULL test, three or more operands, and a left join. In attention the largest costs are elsewhere:
   * `region_0`, a five-table join of 1 M rows with no aggregate (226 ms at L = 256);
   * two cotangent steps of three tables with NULL tests (about 93 ms each).

   To speed up ddx, einfold must handle existence tests and multi-operand folds before it handles the two-table case, or ddx must emit simpler plans.
5. **The earlier issue #30 numbers came from a side branch.** Issue #30 profiled ddx 061b680, which is not ddx's `main`. At `main` (c6657c9) the `matmul` gradient runs in 1.40 s rather than 1.79 s. The step structure, and so every conclusion, is the same.

## Limitations

- One machine; DataFusion only. ddx's programs on DuckDB were not timed.
- The guard test is a property test on small instances of one loss (`matmul`), not a proof. The other workloads' guards were classified, not tested.
- The dense kernel assumes dense integer coordinates and complete inputs, and was timed outside DataFusion, as in S21.
- "Two-table `SUM`" is this spike's model of the first milestone's detector, not its code.
