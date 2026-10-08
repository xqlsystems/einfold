<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S25: ddx emits the fold

**Question.** ddx is an XQL Systems project for automatic differentiation of SQL queries. Suppose it emitted its gradient contractions as an explicit fold node, instead of SQL that einfold has to recognize. How much would that take, how much of ddx's time would it cover, and is it correct?

**Answer.** A DataFusion logical node, physical operator and planner for a gradient contraction (`GradFold`, [`src/fold.rs`](src/fold.rs)) is about 300 lines, comments included. It has a dense positional path and a hash fallback.

* **Correct.** Built from a ddx program's own metadata, it matched ddx's gradient on all 300 random tables with NULLs, NaNs, infinities, missing rows and duplicate keys.
* **Fast.** At n = 50,000 it computes ddx's two `matmul` gradient steps in 28 ms instead of 1,235 ms: **57× and 33× faster**.
* **Coverage.** That shape covers 86–87% of `matmul`'s step time and 54% of the two-layer network's, but only 4% of attention's.

So emitting folds from ddx is cheap and pays off at once for matrix products. Attention needs other shapes (S22).

Date: 2026-10-07. Intel Core i7-8700 (6 cores, 12 threads), 15 GB RAM, Rust 1.91, DataFusion 54.1.0, ddx at its public `main` ([c6657c9](https://github.com/xqlsystems/ddx/commit/c6657c97c1d45daffe5cd8d6339240b36037e9c7)).

## Method

**The node.** `GradFold` has three inputs: a streamed operand `l (x, z, val)`, a held operand `r (z, y, val)`, and a *fill* table `f (x, y, val)`, whose rows the output has. In SQL terms it computes:

```sql
SELECT f.x, f.y, CASE WHEN f.val IS NULL THEN NULL WHEN r.v IS NULL THEN 0.0 ELSE r.v END AS val
FROM f LEFT JOIN (SELECT l.x, r.y, SUM(l.val * r.val) AS v FROM l JOIN r ON l.z = r.z
                  GROUP BY l.x, r.y) r ON f.x = r.x AND f.y = r.y
```

Spike S22 found this equals ddx's gradient step. The operator takes the dense path when every key is a small non-negative integer and `r` has one row per `(z, y)`. That path holds `r` as a matrix, streams `l` into a dense output, and looks up each row of `f` by position. A 0 that stands for a NULL or a missing row of `r` must never meet a NaN or an infinity in `l`. When that could happen, and whenever `r` repeats a key, it takes a hash path that follows SQL's bag semantics exactly. Both paths skip NULL products, as `SUM` does.

**Emitting it.** The spike plays ddx. For each gradient of `SUM(tanh(a·w)²)`, it reads the program's metadata: the gradient's output table and columns, and the cotangent step. From these it builds a `GradFold` over table scans:

* for `a`'s gradient, `l` = the cotangent `(s, o)`, `r` = `w` read as `(o, k)`, `f` = `a`;
* for `w`'s gradient, `l` = `a` read as `(k, s)`, `r` = the cotangent `(s, o)`, `f` = `w`.

DataFusion plans the node through an extension planner. In ddx itself this would be the rule that writes a gradient step for an aggregate of a product. That rule already knows these facts, but this spike doesn't change ddx's code.

**Correctness.** 300 random `matmul` instances, as in S22, with infinities added:

* 1–6 × 1–4 × 1–4 extents;
* rows present with probability 0.8;
* values NULL with probability 0.15, NaN 0.03, inf 0.02, otherwise multiples of 1/8;
* in a third of the cases only `w` is differentiated, and `a` repeats keys with probability 0.3.

Each gradient is compared with ddx's: same keys, same NULLs, NaN where NaN, values within 1e-9 relative.

**Time.** `matmul` at n = 50,000, best of 3. Both forms read the same registered tables and materialize their result as a table. ddx's step is run with `ad::run_step`.

Run with `cargo run --release`. Building it needs `protoc`, for ddx's Substrait dependency.

## Results

300 random cases: **300 agree**, 0 refused, 0 disagree. 2,782 gradient rows compared, of which 426 NULL and 416 NaN. Of the 500 `GradFold` computations, 428 took the dense path and 72 the hash path.

| gradient, n = 50,000 | ddx's step | `GradFold` | speedup | same result |
|---|---:|---:|---:|---|
| Ā (`a`'s) | 702.7 ms | 12.3 ms | 57× | yes |
| W̄ (`w`'s) | 532.1 ms | 16.0 ms | 33× | yes |

ddx's whole run took 1,405 ms, of which its two gradient steps took 1,235 ms. `GradFold` does those in 28 ms.

Coverage, from S22's per-step profile: the share of step time in steps of `GradFold`'s shape (a gradient of a two-table product, with its NULL test and fill):

| workload | covered | what is left |
|---|---:|---|
| `matmul` n = 10,000 | 86% | the forward product, the cotangent |
| `matmul` n = 50,000 | 87% | the same |
| `mlp2` n = 10,000 | 54% | the cotangent through the second layer, which is a contraction times `tanh′`: `GradFold` plus an elementwise factor |
| `attn` L = 256 | 4% | the 1 M-row region step, the softmax cotangents |

## Findings

1. **Emitting the fold is a small amount of code, and it pays immediately for matrix products.** The node, operator and planner are about 300 lines. On ddx's `matmul` benchmark they take the gradient steps from 1.2 s to 28 ms, and the whole gradient run from 1.4 s to roughly 0.2 s, now dominated by the forward product and the cotangent.
2. **The dense path's correctness hinges on existence, not just values.** A dense array stores missing rows and NULLs as 0, which is right for `SUM`, until a NaN or an infinity meets one: 0·∞ is NaN, but SQL forms no product for a missing row. The operator must know which rows exist, or fall back. This is the existence tracking the design keeps apart from the aggregate's state (§8.3), and it is not optional.
3. **The fill table is part of the fold.** ddx's left join, which gives every input row a gradient, became a lookup by position. Without it, a fold node would still need ddx's SQL left join: about 50 ms of S22's 186 ms two-table form.
4. **Coverage is by shape, and attention needs more shapes.** `GradFold` covers the gradients of two-table products. The rest of ddx's time is:
   * contractions with an elementwise factor (`mlp2`'s cotangent);
   * a large region join with no aggregate, and softmax cotangents (attention).

   Those need either more node shapes from ddx, or einfold's general fold with existence factors.

## Limitations

- One loss shape for correctness; the timing is one size on one machine.
- The spike builds the node from ddx's program metadata. Changing ddx's gradient rule to emit it, and carrying it through Substrait (ddx's plan format), are not done.
- The dense path collects all inputs in memory and runs on one thread.
