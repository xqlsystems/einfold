<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S31: where ddx's gradients spend their time with einfold 0.1

**Question.** einfold's milestone M3 is to speed up gradients, starting with attention's. ddx, an XQL Systems project for automatic differentiation of SQL queries, computes a gradient as a *program*: SQL steps, each materialized as a table. With einfold 0.1, which steps of the gradients in xarray-sql's ML demos still take the time, and why? xarray-sql is an XQL Systems project that queries Xarray datasets with SQL, and its demos are the workloads einfold's releases are accepted on. Spike S22 asked the same question before einfold existed, against an older ddx.

**Answer.** einfold 0.1 already matches the steps that held most of the time in S22, and runs them 10–60× faster. What's left is in three places, two of them narrow:

1. **A fold joined back to its domain runs DataFusion's own plan whenever an operand has missing rows**, as a causal mask's lower triangle does. ddx fills a missing group with `0.0` (`CASE … WHEN r.v IS NULL THEN 0.0`). The detector doesn't read that fill, so it gives the node no default, and the dense algorithm declines on any operand with holes, as it must without a default. attention's dQ and dK, 0.6 s of einfold's 1.7–2.1 s per gradient at 1,024 tokens, run as slowly as on DataFusion.
2. **Weighted sums over a saved table of scores aren't matched.** ddx saves each aggregate's output, so the gradient reads the scores `S` as a table, not as the join of `Q` and `K` that M2's pipelines fuse. Three kinds of step reweigh `S` row by row (`exp(s − max) / z`) and sum it against a third table: the output `O = P·V` saved for the backward pass, `dV = Pᵀ·dO`, and the GPT's `dP`. They take 0.7–0.9 s per attention gradient, the same as on DataFusion.
3. **Steps einfold doesn't match get slower from Python.** In one training step of the GPT, the steps einfold doesn't plan take 0.92 s to run against DataFusion's 0.74 s, and planning every step takes 2.9× as long: 571 ms against 198 ms. This is xqlsystems/einfold#81, the per-query cost of datafusion-python's FFI.

If those steps then ran as fast as the folds einfold already runs (25 ms for dQ and dK, as `dP` takes, and 50 ms for each weighted sum, as the scores take), attention's gradient at 1,024 tokens would spend 0.5–0.8 s in its steps, against 3.9–4.2 s on DataFusion: 5–8× faster, up from 1.9–2.5×.

FlashAttention's backward pass, which never stores the `N²` scores, isn't something einfold can do alone. ddx saves the scores, the softmax's `N²` cotangent and `dP` as tables, each a step of its own, and einfold sees one step at a time. It needs ddx to recompute the scores in each backward step instead of saving them, as `jax.checkpoint` can. That is a change of ddx's save-or-recompute policy, and a question for ddx's maintainers.

Date: 2026-10-31. Intel Core i7-8700 (6 cores, 12 threads), 15 GB RAM, otherwise idle (load average 0.02). From PyPI: einfold 0.1.0, ddxdb 0.4.6 (ddx's Python package) and datafusion 55.0.0 (DataFusion's Python bindings). xarray-sql at [6debe24](https://github.com/xqlsystems/xarray-sql/commit/6debe24).

## Method

[`profile.py`](profile.py) runs one of xarray-sql's demos unchanged, with ddxdb's DataFusion backend wrapped. Each program step ddxdb materializes is recorded:

- its rows;
- the time to plan it, and then to run it as ddxdb does (`to_arrow_table`, which plans it again; running is the difference);
- whether einfold planned any of it, from the extension's counters;
- its optimized logical plan, and counts of its scans, joins, aggregates, windows and `IS NULL` tests.

With `S31_ANALYZE=1` it also writes each step's `EXPLAIN ANALYZE`, which works through the FFI. With `S31_DUMP=1` it writes each step's rows and the demo's tables as Parquet, and each step's SQL as ddxdb writes it (`ddxdb.ad.to_sql`). Then [`src/main.rs`](src/main.rs) runs any step's SQL again from Rust, with einfold on and off, and prints einfold's `EXPLAIN ANALYZE`.

The workloads:

- **Demo 1, attention:** `01_attention.py --sizes 1024`, the gradient of `SUM(O²)/2` with respect to `Q`, `K` and `V`, with 1,024 tokens and 64 features, in both of the demo's spellings: the row maximum as a window, and as a `GROUP BY` joined back.
- **Demo 2, the GPT:** `02_nanogpt.py --sizes 12 --steps 1 --embed 128 --heads 4 --context 64 --layers 1`, one SGD step at nanoGPT's CPU size: 4 heads, 128 channels, a context of 64 characters, and 12 sequences per batch.

Each ran once per host, so the times are single runs. Step times sum planning and running, the time ddxdb spends in the step. The demos' own timings below include the profiler's second planning of each step.

To run it, install einfold, ddxdb and datafusion beside a checkout of xarray-sql (its `.venv` has the rest), then:

```sh
python profile.py out ~/git/xarray-sql/benchmarks/ml/01_attention.py --hosts einfold datafusion --sizes 1024
S31_DUMP=1 python profile.py dump ~/git/xarray-sql/benchmarks/ml/01_attention.py --hosts datafusion --sizes 1024
cargo run --release -- dump/datafusion dump/datafusion/0012.sql
```

## Results

### Attention, 1,024 tokens

The demo's own times for the whole gradient were 2.30 s (window) and 1.89 s (`GROUP BY`) with einfold, and 3.98 s and 4.31 s on DataFusion.

Steps over 20 ms, window spelling. *Matched* means einfold planned some of the step.

| step | what it computes | rows | DataFusion | einfold | matched |
|---|---|---:|---:|---:|---|
| `saved_0` | the scores `S = QKᵀ/8`, causal | 524,800 | 594 ms | 46 ms | yes |
| `saved_2` | the output `O = softmax(S)·V`, from `S` saved | 65,536 | 358 ms | 357 ms | no |
| `contraction_0` | `dP = dO·Vᵀ` | 1,048,576 | 1,295 ms | 22 ms | yes |
| `cotangent_1` | the softmax's row sums | 1,024 | 60 ms | 54 ms | no |
| `cotangent_0` | `dS`, the softmax's backward pass | 524,800 | 388 ms | 403 ms | no |
| `grad_0_q` | `dQ = dS·K/8` | 65,536 | 166 ms | 185 ms | yes, but ran DataFusion's plan |
| `grad_1_k` | `dK = dSᵀ·Q/8` | 65,536 | 421 ms | 447 ms | yes, but ran DataFusion's plan |
| `grad_2_v` | `dV = Pᵀ·dO`, from `S` saved | 65,536 | 557 ms | 528 ms | no |
| **all 13 steps** | | | **3,856 ms** | **2,077 ms** | |

The `GROUP BY` spelling: 4,207 ms on DataFusion, 1,703 ms with einfold. Its `dP` is written inside a region step (`region_1`, 1,776 ms against 84 ms), and its `cotangent_0` is smaller (134 ms against 159 ms), since its maximum is a saved aggregate rather than a window.

**Why dQ and dK run DataFusion's plan.** Their `EXPLAIN ANALYZE` shows `EinFoldExec: a,b * a,c -> b,c + unique domain` with `host_plans=1`. That's the fold joined back to its domain (`k`), checked for unique keys as it runs. The fold's dense algorithm declined, and a node with a domain runs the host's plan when it declines. Instrumenting the dense algorithm showed which requirement failed: `b`, the cotangent `dS`, has holes, the masked upper triangle, and the domain has no default (`missing: None`). Without a default, a hole would turn a group that should be missing into a 0, so declining is right. But ddx's step fills every missing group with `0.0` itself:

```sql
CASE WHEN k.x IS NULL THEN NULL WHEN r.v IS NULL THEN 0.0 ELSE r.v END
```

With that fill, a missing group and a group of zeros give the same row, which the dense algorithm's `missing: Some(0.0)` was written for. The detector never sets it, in either of the shapes with a domain ([`join_back.rs`](../../../crates/einfold-datafusion/src/join_back.rs) and [`detect_folds.rs`](../../../crates/einfold-datafusion/src/detect_folds.rs)). Only a node built by hand, as in einfold's examples, has it. Run alone, without its domain, the same fold takes einfold's discrimination path in 127 ms, against DataFusion's 380 ms.

Every step that has a domain and reads a masked operand meets this, which in a causal model means every gradient that flows through the scores.

**The weighted sums over saved scores.** `saved_2` is the second half of M2's pipeline, `Σ_j exp(s_ij − m_i)/z_i · v_jd`. Here `s` is read from the table `saved_0`, `m` is a window over it, and `z` is a saved table. M2's detector looks for a pipeline whose scores are a fold over a join, so a table of scores doesn't match. `grad_2_v` is the same sum transposed (`Σ_i P_ij · dO_id`), behind ddx's fill.

### The GPT, one training step

The demo's own times for the step were 4.03 s with einfold and 10.9 s on DataFusion. Its 63 program steps:

| | DataFusion | einfold |
|---|---:|---:|
| steps einfold matches (25), running | 8,561 ms | 789 ms |
| steps einfold doesn't match (38), running | 740 ms | 922 ms |
| planning, all 63 | 198 ms | 571 ms |
| **all steps** | **9,498 ms** | **2,281 ms** |

The largest steps einfold doesn't match:

| step | what it computes | rows | DataFusion | einfold |
|---|---|---:|---:|---:|
| `cotangent_4` | `dS`, attention's softmax backward | 99,840 | 105 ms | 250 ms |
| `cotangent_5` | `dP`, weighted by `exp(s − m)` from the saved scores | 98,304 | 120 ms | 137 ms |
| `saved_7` | attention's output from the saved scores | 98,304 | 146 ms | 127 ms |
| `cotangent_15` | the cross-entropy's softmax backward | 49,920 | 47 ms | 80 ms |
| `saved_1` | a layer norm's variance (`AVG(power(x − μ, 2))`) | 768 | 14 ms | 51 ms |

The attention steps are the same three kinds as in demo 1. The rest are row-local work, layer norms and softmaxes, which aren't folds. Through einfold's bridge they take up to 3.6× as long, in planning and in running.

Also seen in passing: since xarray-sql 6debe24 writes the GPT without its shared-CTE workaround, its loss takes 110 s to plan and run on DataFusion, and 12 s with einfold, because DataFusion plans a CTE once per reference (xqlsystems/xarray-sql#267).

## What this means for M3

In order of time saved per unit of work:

1. **Read ddx's fill as the domain's default** (`missing: Some(0.0)`), so that the dense algorithm takes masked operands under a domain. It's small, it's in detection only, and it makes dQ and dK about as fast as `contraction_0`: roughly 0.55 s saved per attention gradient at 1,024 tokens, and more in the GPT, whose masked gradients have the same shape.
2. **Fuse weighted sums whose scores are a table:** a pipeline whose first fold is a scan. This covers `O`, `dV` and the GPT's `dP`, about 0.8 s per attention gradient. It reuses M2's fused kernel and weights, with the producer replaced by a read.
3. **Cut the bridge's cost on steps einfold doesn't plan** (#81): about 0.55 s of the GPT step's 2.3 s.
4. **FlashAttention's backward pass** needs ddx to recompute the scores in the backward pass rather than save them. Then each backward step is one query, in which einfold can find a pipeline over `Q` and `K`, as it does for the forward pass. Until then, a gradient stores `N²/2` scores, so at 16,384 tokens it holds 134 million rows, about 3 GB, in each of several tables.

Chains of three or more products, M3's other benchmark, aren't among the steps that take the time in either demo.

## After the fixes

The three changes S31 led to: #129 reads ddx's fill as the domain's default, #131 finds products over three or more relations by regrouping the joins (and computes float `/` on an operand's rows), and #132 runs DataFusion's own functions in einfold's library rather than through the FFI. With all three, from a local build, on the same machine, as medians of 3 runs of the demos themselves:

| | einfold 0.1.0 | with the three | DataFusion | DuckDB |
|---|---:|---:|---:|---:|
| attention gradient, 1,024 tokens, window | 2.30 s | 0.90 s | 4.03 s | 5.42 s |
| attention gradient, 1,024 tokens, `GROUP BY` | 1.89 s | 0.59 s | 4.67 s | 7.60 s |
| GPT training step, nanoGPT's CPU size | 4.03 s | 2.61 s | 12.05 s | |

That's 4.5× and 7.9× DataFusion on attention's gradient, and 4.6× on the GPT step. The steps that changed:

- dQ and dK: 418 ms → 32 ms, the dense algorithm under ddx's fill;
- `O = softmax(S)·V` from the saved scores: 302 ms → 33 ms, and `dV`: 653 ms → 33 ms, each a fold of regrouped operands;
- the GPT's softmax backward, which einfold doesn't plan: 335 ms → 136 ms (DataFusion: 107–129 ms). Before #132, its CPU time was 2.6× DataFusion's, a third of it converting the session's configuration for each call of a foreign function.

Two costs remain that einfold can't remove alone. The window spelling's softmax backward (`cotangent_0`, 400–540 ms) is a `UNION` of four branches, two of which recompute the other two's `N²` aggregates, which is ddx's program (xqlsystems/ddx#243). And from Python, every query pays the FFI's fixed cost, a few milliseconds: on TPC-H, which einfold leaves alone (it plans none of the 22 queries, and each gives DataFusion's rows), that's 1.21× DataFusion's time at scale factor 0.1 and 1.06× at scale factor 1.

A chain of three matrix products written flat (`SUM(a.v * b.v * c.v)` over three joins) is also found now, as a fold of `a JOIN b` and `c`. It's 4.2× faster than DataFusion and 7.4× faster than DuckDB at 96 × 96, but only as fast as DataFusion at 32 × 32, since `a JOIN b` still has `n³` rows. Choosing the order of contractions, M3's next item, would compute it as two matrix products instead.
