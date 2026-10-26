<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S28: detecting ddx's folds

**Question.** Can einfold find the folds in ddx's gradient programs by itself, in DataFusion's plans, with no help from ddx? ddx is an XQL Systems project for automatic differentiation of SQL queries. Spike S22 found that ddx's gradient steps rebuilt the forward join for a NULL test, so a detector for two-table folds matched only 13–32% of their time. Design v15 therefore made M1 an explicit fold that ddx would emit. ddx#126 has since removed that join. If einfold can detect ddx's folds, the two projects stay siblings, with no dependency either way, and einfold also speeds up ddx's steps wherever DataFusion runs them, including inside xarray-sql's Python `Context`.

**Answer.** Yes, for most of the time in ddx's own benchmarks. On the optimized plans an optimizer rule would see, folds a narrow detector finds hold this share of step time:

| workload | folds | masked folds (M2) |
|---|---:|---:|
| `matmul`, n = 50,000 | 63–64% | |
| `mlp2` (two layers), n = 10,000 | 76–79% | |
| `attn` (ddx's attention), L = 256 | 59–64% | |
| demo 1's causal attention gradient, L = 256, window spelling | 43% | 19% |
| demo 1's causal attention gradient, L = 256, `GROUP BY` spelling | 24–26% | 10% |

(Ranges are over two runs.) Every gradient step ends in the same **fill** shape: a left join from the differentiated table onto an aggregate by all its keys, with NULL-safe keys, which is what einfold's fill assumes. The aggregate inside it is not itself a fold, though. ddx joins the transposed contraction back to the table and groups again by the table's keys. So detection finds the inner contraction, and ddx writing the step as a plain left join from the table would let einfold fuse the fill as well.

Date: 2026-10-26. Intel Core i7-8700 (6 cores, 12 threads), 15 GB RAM, Rust 1.99, DataFusion 55.2.0, ddx at its public `main` ([921aef1](https://github.com/xqlsystems/ddx/commit/921aef1a34baa3b58ab32a5a68bd663b35060bfd)), on Tokio's multi-threaded runtime.

## Method

[`src/main.rs`](src/main.rs) builds each workload's tables as ddx's benchmark `crates/ddx-datafusion/tests/ad_perf.rs` does: complete coordinate tables with `sin`/`cos` values. It builds the gradient program with `ad::grad`, and times the whole program (`engine::execute`) and each step (`engine::run_step`), best of 3. The workloads are S22's three (`matmul`, `mlp2`, `attn`) and demo 1's causal attention in both of its spellings, with D = 64 features, exactly as `docs/demos.md` and xarray-sql's `benchmarks/ml/01_attention.py` write it. The demo's loss is half the output's squared norm, differentiated with respect to every value of Q, K and V.

For each step, the spike optimizes its logical plan with the session's optimizer, as a rule placed at the end of the optimizer would see it, and looks for three shapes:

- **fold:** an `Aggregate` computing exactly one `SUM` (no `DISTINCT`, no filter), grouped by plain columns, over projections and filters of one inner join with equality keys, where the sum's argument is a product `l * r` with `l` reading only the join's left input and `r` only its right. Columns are traced through the projections between the aggregate and the join;
- **masked fold:** the same, except that the join also has a condition reading both inputs that isn't an equality, such as causal attention's `q.t >= k.t`;
- **fill:** a left join from a relation onto an aggregate, on as many keys as the aggregate has group keys. The spike also reports whether the aggregate is a fold, and whether the join treats NULL keys as equal.

It times each outermost match on its own, as a plan run alone, to see how much einfold could replace. When a fill's aggregate isn't a fold, it times the folds inside it instead. A match's time includes reading its inputs, which are materialized tables, so it slightly overstates what einfold would replace. It also profiles demo 1's forward query.

Run with `cargo run --release`. `S28_PLANS=1` also prints the plans of steps over 15 ms that nothing matched. Building it needs `protoc`, for ddx's Substrait dependency.

## Results

Times in ms, from one run.

### `matmul`, n = 50,000

Whole program: 413 ms.

| step | ms | matched | matched, run alone |
|---|---:|---|---:|
| `saved_0` (forward z = a·w) | 38.4 | fold | 38.3 |
| `saved_1`, `value`, `cotangent_1` | 2.6 | | |
| `cotangent_0` | 17.9 | | |
| `grad_0_a` (Ā) | 262.0 | fill onto an aggregate that isn't a fold; inside it, a fold | 150.3 |
| `grad_1_w` (W̄) | 37.6 | fill onto an aggregate that isn't a fold; inside it, a fold | 36.2 |

### `mlp2`, n = 10,000

Whole program: 111 ms. The two forward products (`saved_0`, 16.7 ms; `saved_1`, 20.7), the cotangent through the second layer (`cotangent_0`, 41.9, of which the fold is 31.6) and both gradients' inner contractions (11.8 and 7.1) are folds. The rest is scalar and elementwise.

### `attn`, L = 256

Whole program: 164 ms. Folds: the three projections and their scores and output products (`saved_0`, `_1`, `_2`, `_5`, `_6`, 3–20 ms each), ddx's `contraction_0` (24.4), three cotangents (16.4, 7.7, 7.4) and the gradients' inner contractions (about 2 ms each). Unmatched: `region_1` (10.3 ms) and the softmax cotangents (`cotangent_2`, 19.8 ms, a union of aggregates over windows).

### Demo 1, L = 256

| step | window spelling | `GROUP BY` spelling | matched |
|---|---:|---:|---|
| the scores, `saved_0` | 45.5 | 45.5 | masked fold |
| the output, `SUM(w / z · v.x)` over weights ⋈ v ⋈ norm | 23.5 | 231.1 | no: a factor reads two joined tables |
| `contraction_0` | 40.4 | 40.0 | fold |
| the softmax cotangents (`cotangent_0`) | 29.5 | 15.5 | no: unions of aggregates over windows |
| the gradients of Q, K and V | 19.5, 26.4, 32.1 | 23.7, 25.8, 30.8 | fill onto an aggregate that isn't a fold; inside each, a fold of 15–31 ms |
| whole program | 248 | 437 | |

The forward query (98 ms in the window spelling, 153 ms in the `GROUP BY` spelling) contains the masked scores fold twice in the window spelling and four times in the other, about 46 ms each, because DataFusion copies a CTE into the plan at every place that reads it.

## Findings

1. **Detection now finds most of ddx's contraction time.** On ddx's three benchmarks, 59–79% of step time is inside folds a narrow detector matches, against S22's 13–32% before ddx#126. ddx doesn't need to emit anything for einfold to find them, so M1 can detect instead.
2. **The fill is a standard left join, and its keys are NULL-safe.** Every gradient step ends in `table LEFT JOIN aggregate` on all the table's keys, with DataFusion's `NullEqualsNull`, as einfold's fill assumes. But the aggregate under it groups again, after joining the contraction back to the table. In `matmul`'s Ā, the inner fold is 150 ms of the step's 262 ms. Written directly as `a LEFT JOIN t`, the step takes 182 ms on plain DataFusion, and the fill becomes a pattern einfold can fuse. That rewrite is ddx's, and it helps every host (xqlsystems/ddx#208).
3. **Demo 1 needs M2 and more.** Its scores are a masked fold, 10–19% of the gradient's step time and nearly half of the forward query. Its output product is a fold whose factor `w / z` reads two joined tables. ddx's own contraction rule already splits a join tree's inputs by which factor reads them, and a detector can do the same. In the `GROUP BY` spelling, that step (231 ms) recomputes the N² weights inside the join: the producer-consumer pair M2 fuses. The softmax cotangents are M3's.
4. **The forward plan computes the scores up to four times.** DataFusion copies a CTE into the plan wherever it is read, and each copy is a fold. Detection would replace each copy, so it would help, but planning each CTE once (DataFusion 56's `AS MATERIALIZED`) is the real fix, and it isn't einfold's.

## Limitations

- One machine, one size per workload, best of 3.
- A matched subtree's time includes reading its inputs, and whether einfold's operator beats it depends on whether the dense path applies. S21 and einfold#38 measured that separately.
- The detector is this spike's model of M1's rule, not its code. It doesn't check types (all factors here are `DOUBLE`) or NULL semantics of the join keys beyond the fill's.
- The demo's tables are built in Rust as ddx's benchmark builds them, not registered through xarray-sql, and its gradient runs on ddx-datafusion rather than through ddxdb and the Python `datafusion` wheel. The plans should be the same, since both reach DataFusion's optimizer, but that wasn't checked.
