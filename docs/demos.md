<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# einfold target demos

Status: draft v2. Author: Alex Merose. Last updated: 2026-10-15.

This doc describes three demos that would show what XQL, ddx, and einfold can do together. Each also serves as a check on einfold's design ([`design.md`](design.md)): it says which parts of the design it exercises, and which gaps it exposes.

The three projects, briefly:

- **XQL** gives SQL access to large scientific arrays stored in Zarr, a chunked array format. A dataset becomes a table with one row per combination of coordinates.
- **ddx** differentiates SQL queries: given a query that computes a function, it produces the queries that compute its gradients. That makes model training expressible in SQL.
- **einfold** rewrites query plans so that the SQL engine running them (the *host*) computes tensor contractions fast. It never executes anything itself.

The demos are not meant to beat specialized systems on raw speed. They are meant to show that a declarative, relational description of a model, plus a principled optimizer, can recover the algorithmic ideas that those systems were hand-engineered to exploit.

## Demo 1: Rediscovering FlashAttention automatically

This is the demo einfold's research is aimed at first (design.md §16, milestone M2).

### The idea

Attention, the core of a transformer, computes `softmax(QKᵀ/√d)·V` for query, key and value matrices `Q`, `K`, `V`. A naive implementation materializes the full `N×N` matrix of scores for `N` tokens. FlashAttention (Dao et al., 2022) never does. It splits `Q`, `K` and `V` into tiles. For each tile of queries, it streams over tiles of keys and values, and keeps a running softmax that it rescales as larger scores appear (the "online softmax", Milakov and Gimelshein, 2018). Memory per tile stays small, and with a causal mask (each token attends only to earlier tokens), tiles above the diagonal are skipped entirely.

The demo: write attention in plain SQL, and have einfold produce a plan with FlashAttention's structure, without being told about FlashAttention. "Without being told" means every rule involved is general: each must also fire on a query that has nothing to do with attention (see the success criteria).

**Prior art.** Three systems have rediscovered FlashAttention automatically, all from tensor programs:
- Mirage (Wu et al., 2025), a superoptimizer that searches kernel, block and thread levels at once;
- Blockbuster (Dekel, 2025), whose fusion rules model how blocks move between memory tiers;
- EqiForge (Zhan et al., 2026), which uses equality saturation with a rule for fusing a reduction into a later reduction that depends on it.

einfold's version starts from SQL instead. That brings constraints the tensor systems don't have:
- results must match SQL's, NULLs and special values included;
- a mask is a predicate, and a masked pair is an absent row;
- the output is a plan for someone else's engine.

### Attention in SQL

For one attention head, with `Q`, `K` and `V` stored as `(t, d, x)` tables (token, feature, value):

```sql
WITH scores AS (
  SELECT q.t AS i, k.t AS j, SUM(q.x * k.x) / 8.0 AS s   -- 8 = sqrt(64 features)
  FROM Q q JOIN K k ON q.d = k.d
  WHERE q.t >= k.t                                       -- causal mask
  GROUP BY q.t, k.t),
weights AS (
  SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS w
  FROM scores),
norm AS (
  SELECT i, SUM(w) AS z FROM weights GROUP BY i)
SELECT w.i AS t, v.d, SUM(w.w * v.x) / n.z AS x
FROM weights w
JOIN V v ON w.j = v.t
JOIN norm n ON w.i = n.i
GROUP BY w.i, v.d, n.z;
```

ddx (an XQL Systems project for differentiating SQL queries) spells the maximum differently: `MAX(s) … GROUP BY i`, joined back to the scores. It may also divide inside the sum, `SUM(w / z * v.x)`. Both spellings mean the same thing, and einfold must recognize both.

### What the query computes

Read algebraically, the query is two folds over joins, one feeding the other:

1. **The scores** are an einsum, `s(i, j) = Σ_e q(i, e)·k(j, e) / 8`, over the pairs the mask allows.
2. **The output** is `x(i, d) = Σ_j e^(s(i,j) − m(i))·v(j, d) / Σ_j e^(s(i,j) − m(i))`, with `m(i)` the largest score in row `i`.

Two observations make FlashAttention's plan fall out of einfold's existing abstractions, rather than needing new algebra:

- **The maximum cancels.** `e^(−m(i))` multiplies both sums, so `x(i, d) = Σ_j e^(s(i,j))·v(j, d) / Σ_j e^(s(i,j))`. Rabe and Staats (2021) use the same rearrangement. That is a weighted average with weights `e^(s)`. In the design's terms it is an **algebraic fold**: two sums and a final division, the same class as `AVG` (design §2.2).
- **The maximum is numerics, not algebra.** Its only job is to keep `e^(s)` from overflowing. That is the **accumulator**'s concern (design §8.6), not the aggregate's. A *scaled exponential accumulator* holds a sum of exponentials as a pair, a shift `m` and a sum scaled by `e^(−m)`, and rescales when `m` grows. Merging two such pairs is associative and commutative, so it is a commutative monoid, and partial results combine like any other. The online softmax is exactly this accumulator. No log-sum-exp semiring is needed.

What remains is **fusing the two folds** (design §9.6). The output fold reads the scores through a nonlinear map, `exp`. Fusing them is legal for two reasons:
- each score's group `(i, j)` is an output group `i` paired with a value of the output's summed dimension `j`;
- each score is complete once its tile has summed all its features `e`.

So the two folds can run as one loop over (query tile, key tile) pairs, holding one tile of scores at a time.

**Masks are absent rows.** A masked pair `(i, j)` is not a score of −∞, and not a weight of 0: in SQL it is a row that doesn't exist, so it contributes nothing to any sum. The difference shows on special values:
- if `v(j, d)` is NaN, a kernel that multiplies a masked pair's weight 0 by it produces NaN where SQL produces a number;
- a query row whose every key is masked has no output rows at all, while a row whose scores are all −∞ outputs NaN.

Spike S27's first fused kernel had the first bug.

### How einfold finds FlashAttention's plan

| FlashAttention's idea | einfold's mechanism | Design | Milestone |
|---|---|---|---|
| Never store join rows for `QKᵀ` or for the product with `V` | EinFold fuses each join with its sum | §10.2 | M1 |
| Never store the `N×N` scores between the two products | Fold pipelines: the output fold consumes each tile of the scores fold as it is produced | §9.6 | M2 |
| The online softmax | The output is an algebraic fold with weights `e^(s)`, summed in a scaled exponential accumulator | §8.6, §9.6 | M2 |
| Recognize the softmax in SQL | Detect `exp(s − max)` normalized by its sum, spelled with a window or with `GROUP BY` and a join | §9.6 | M2 |
| Skip tiles above the diagonal | The predicate `q.t >= k.t` becomes a mask; each tile pair is classified as empty, full or partial from the predicate and the tile's position bounds; empty pairs are skipped, and partial ones skip masked pairs rather than zeroing them | §9.1, §10.2 | M2 |
| Work tile by tile with bounded memory | Tiles chosen from the shape alone, which also makes results deterministic (S24) | §8.6, §10.2 | M2 (fixed tiles); M7 (tiles under a memory budget, S20) |
| The backward pass recomputes scores tile by tile from a saved row statistic, `m + log z` | The gradient of the pipeline is a pipeline over the same tiles | §16 | M3 |

### Evidence

Spike S27 ([`spikes/s27-fused-attention`](spikes/s27-fused-attention/README.md)) ran this query, in both spellings, on DataFusion and DuckDB with 6 GB of memory each. It compared them with a hand-written fused kernel reading the same rows: 64×64 tiles, causal tiles skipped, and an online softmax.

| tokens L | DataFusion | DuckDB | fused kernel | fused vs. the faster host |
|---:|---:|---:|---:|---:|
| 1,024 | 1.8 s, 0.6 GiB | 0.95 s, 0.15 GiB | 3.4 ms | 281× |
| 4,096 | 53 s, 5.9 GiB | 10.5 s, 0.8 GiB | 27 ms | 390× |
| 8,192 | fails (out of memory, or past 15 min) | 28 s, 2.2 GiB | 96 ms | 297× |
| 16,384 | not run | fails (out of memory) | 0.33 s | — |
| 65,536 | | | 5.1 s | — |

(Window spelling; the `GROUP BY` spelling was slower on both hosts. The kernel's times are on 12 threads, from rows to rows.)

- **The hosts hold every score.** Both keep the `N²` score groups in a hash aggregate, and both stop at sequence lengths the fused kernel handles in a third of a second.
- **The kernel's memory** is its inputs and output, linear in `N`, plus one 64×64 tile of scores per thread. The causal mask let it skip half the key tiles.
- **It was more accurate than DataFusion and deterministic.**
  - Errors were 1–5e-15 of the largest output, against DataFusion's 0.6–9e-14, even with scores spread 40× wider.
  - Its bits were identical across thread counts.
- **Special values.** With masked pairs skipped, it matched DataFusion on NaN and ±∞ in every input, and on a row whose scores are all −∞. With masked pairs zeroed, a NaN in `V` broke 5 rows.

### Success criteria

- **Memory.** The plan never stores the `N×N` scores. Peak memory beyond the inputs and the output is at most one dense copy of the inputs' values, which the fused kernel reads its tiles from, and a constant number of tiles per thread. So it grows linearly in `N`, never quadratically. Reading the tiles from the inputs in place, without the copy, would need the inputs' rows to arrive in position order; it is left for later.
- **Masks.** With the causal mask, about half the key tiles are never visited.
- **Correct.**
  - Results are within the error bound of design §8.6 for this accumulator: the summation bound, plus one rounding of `exp` per rescaling. This is tested on ill-conditioned scores.
  - On NaN, infinities and masked rows, results are exactly what SQL gives.
  - Results repeat bit for bit across thread counts.
- **Fast.** Faster than DataFusion and DuckDB running the naive plan, at every `N` where they finish. It also runs at `N` where they exceed their memory budget.
- **Found, not told.**
  - einfold finds the plan from the SQL above, in both spellings, with no attention-specific rule.
  - The same rules fuse a query that isn't attention: a Gaussian-kernel weighted average, `SUM(exp(−dist²/h²)·y) / SUM(exp(−dist²/h²))` over a join of target and source points. That is the Nadaraya–Watson estimator, used in regridding.
- **Where.** Measured on einfold's DataFusion reference executor. The relational form can't meet the memory criterion: a host's hash aggregation keeps every `(i, j)` score group until the scores are complete.

FlashAttention's speed on GPUs also comes from managing on-chip memory, and that is the host's job. einfold produces the plan's *structure*. Running it at GPU-kernel speed requires a host that implements the `Fold` relation, such as Sirius or NVIDIA's GPU Query Engine.

## Demo 2: NanoGPT in N lines of SQL, plus an M-line diff for the sparsity record

### The idea

nanoGPT is Andrej Karpathy's small, readable implementation of GPT-2 training. The NanoGPT speedrun (Keller Jordan's modded-nanogpt) is a public competition to train a 124M-parameter GPT-2 to a validation loss of 3.28 on the FineWeb dataset as fast as possible, on 8 NVIDIA H100 GPUs. In late September 2026, Larry Dial reported a record of 39.9 seconds, down from 67.6 seconds, from skipping low-value work:

1. **Sampled softmax.** If a token doesn't appear in a batch, skip its part of the output layer some of the time.
2. **Sparse values.** Run the optimizer step only for n-gram embedding rows that occurred in the batch. (This sets Adam's `beta1` to zero, and applies `beta2` retroactively when a row is next used.)
3. **Sparse updates.** Update n-gram and value embeddings every 4 steps instead of every 2.
4. **Sparse communication.** Shard the n-gram table across GPUs, and exchange only the rows receiving updates.

The demo has two parts: GPT-2 written in `N` lines of SQL and trained through ddx, and the four tricks as a diff of `M` lines.

### Part A: GPT in SQL

| Model piece | In SQL |
|---|---|
| Token and position embeddings | A join of the token table with the embedding tables on token id and position |
| Layer normalization | Per-row aggregates (mean and variance), used as derived factors |
| Attention | Demo 1 |
| MLP | Two einsums, with the activation as a derived factor |
| Output layer and cross-entropy loss | An einsum with the unembedding matrix, a log-sum-exp per position, and a join with the target tokens |
| Gradients | ddx |
| Optimizer step | A join of each parameter table with its gradient table, producing the next version of the parameters |

`N` is a measurement, not a target. The goal is a model that reads about as clearly as nanoGPT's own model file, which is a few hundred lines of Python.

### Part B: the sparsity record as a diff

| Trick | The SQL change | Exact or approximate | einfold's role |
|---|---|---|---|
| Sparse values | Largely free: the gradient of an embedding lookup, a `GROUP BY token` over the batch, has rows only for tokens that occurred, so the optimizer's join touches only those rows. The diff adds a `last_step` column to the optimizer state, and applies `beta2` raised to `step − last_step` when a row is touched. | Exact for the gradient; the `beta1 = 0` choice is the record's own algorithmic change | Run the sparse join efficiently |
| Sparse communication | Free in a distributed relational engine: a join partitioned by token exchanges only the rows that exist | Exact | None. Distributed execution is outside einfold's scope, so this demo counts the rows that *would* be exchanged. |
| Sampled softmax | Restrict the output layer's vocabulary to the batch's tokens plus a random sample: `WHERE token IN (SELECT token FROM batch) OR random() < p` | Approximate, so written by the user (design principle 8) | Prune the support of the restricted operand |
| Sparse updates | Gate the update with `WHERE step % 4 = 0` | Approximate, so written by the user | None needed |

The point to show: two of the four tricks are close to what SQL already does by default, and the other two are one-line filters. If the MLP uses a ReLU-family activation (modded-nanogpt has used ReLU²), einfold can also *infer* exact zero skipping in the forward and backward passes ("matmul pushdown", design.md §9.2), with no diff at all.

### Success criteria

- On a tiny configuration (for example a character-level model of Shakespeare), the SQL model's training loss tracks a PyTorch reference, and ddx's gradients match JAX's.
- For each trick, measure the rows and floating-point operations skipped, against the dense plan.
- `M` stays small: tens of lines.
- Not a goal: beating 39.9 seconds. The goal is the same *algorithmic* savings, from a declarative model.

### What the demo exercises

Program mode (a training step is a program of plans), derived factors, masks, sums over `UNION ALL` (the token embedding is read both at the input and at the output layer), inferred zero elimination, the determinism setting (training would turn it off), and ddx's facts. Gaps it exposes: the online softmax from Demo 1, for efficiency; and how optimizer state is versioned between steps, which is a question for XQL and ddx rather than einfold.

## Demo 3: GraphCast in SQL, reading only what a forecast needs

### The idea

GraphCast (Lam et al., 2023) is Google DeepMind's machine-learning weather model. It predicts the global atmosphere 6 hours ahead from the two previous states, and repeats to forecast up to 10 days. It is a graph neural network with three stages:

- an **encoder** maps a 0.25° latitude–longitude grid (721 × 1440 points, with variables at 37 pressure levels) onto a "multi-mesh": 40,962 points on a refined icosahedron, with edges from every level of refinement, including very long ones;
- a **processor** runs 16 rounds of message passing on the multi-mesh;
- a **decoder** maps back to the grid.

It has 36.7 million parameters, and is trained on ERA5, the hourly global reanalysis of past weather.

The demo: GraphCast inference in SQL, over ERA5 stored in Zarr, where a query for a forecast of one region at one lead time reads only the data and computes only the nodes it needs.

### GraphCast in SQL

The graph is three edge tables (grid to mesh, mesh to mesh, mesh to grid). One round of message passing is a join of node features with an edge table, an MLP over each edge (einsums plus activations), and a `SUM` into each receiving node. Those are einsums over sparse edge tables, which is exactly what the EinFold join's hash algorithm (Gustavson's algorithm, design.md §10.2) is for. Each layer's weights are shared across all nodes and edges.

### Where pushdown helps

| Opportunity | Mechanism | Expected effect |
|---|---|---|
| Read only the variables and pressure levels the model uses | Projection pushdown, which XQL readers already do | ERA5 holds many more variables than GraphCast uses |
| Read only the two input time steps | Filter pushdown on time, down to chunk reads | Large; ERA5 spans decades |
| Compute only the requested region | The query's filter on the output's coordinates propagates backward: the decoder computes only the grid points in the region; a semi-join with the mesh-to-grid edges finds the mesh nodes they read; each processor layer adds one hop of multi-mesh edges; the encoder needs only the grid cells feeding those mesh nodes; and the reader reads only those chunks. This is semi-join reduction (§9.2), run backward from the query through all 16 layers. | Large for the decoder, the output, and reading. In the processor, uncertain (see below). |
| Reuse the fixed graph | Edge tables and mesh geometry never change, so their structure is computed once (§8.5) | Faster repeated forecasts |
| Match ERA5's chunking to spatial access | Retiling and chunk order (§9.4, the layout convention) | Depends on how the ERA5 copy is chunked |

**The honest caveat.** The multi-mesh contains edges from the coarsest icosahedron, which span thousands of kilometers. After 16 rounds of message passing, a single grid point may depend on almost the whole globe. So backward pruning may save little inside the processor, and less still for forecasts of several steps. It will clearly save in the decoder, in the output, and in what is read from storage. Measuring how much it saves in the processor is part of the demo. One variation would be a model trained without the longest edges, but that is a different model.

**On weights.** GraphCast shares each layer's weights across all nodes, so nearly all of its 36.7 million parameters are needed for any forecast. Pushdown matters here for data and activations, not for weights. A model with region-specific weights would benefit in the same way, through the same filters.

### Success criteria

- One 6-hour step matches DeepMind's reference JAX implementation within tolerance.
- For a regional forecast, measure the bytes read, the rows processed per stage, and the latency, against a global forecast.
- Report the receptive field per processor layer, to show where pruning stops paying.

### What the demo exercises

Zarr facts and readers, projection and filter pushdown into chunk reads, semi-join reduction through a deep plan, sparse einsums over edge tables, support facts propagated through graph joins, and the structure/value split. Gaps it exposes: running semi-join reduction through a plan 16 layers deep, and computing a receptive field as a bound on support (design.md §8.1).

## Summary: what the demos ask of the design

| Design feature | FlashAttention | NanoGPT | GraphCast |
|---|---|---|---|
| Masks and partly masked tiles | ✓ | ✓ | |
| EinFold join (fused join and sum) | ✓ | ✓ | ✓ |
| Fold pipelines (fusion across folds) | ✓ | ✓ | |
| Scaled exponential accumulator (softmax) | ✓ | ✓ | |
| Tiles with a memory budget (S20) | ✓ | | ✓ |
| Derived factors | ✓ | ✓ | ✓ |
| Program mode | | ✓ | |
| Exact zero elimination, declared or inferred | | ✓ | |
| Semi-join reduction and pushdown into reads | | | ✓ |
| Zarr facts and the layout convention | | | ✓ |
| ddx gradients | | ✓ | |

Design gaps found, in order of how many demos need them:

1. **Fusion across folds** with a nonlinear step in between. Designed in v16 as fold pipelines (design §9.6), for M2. FlashAttention and NanoGPT.
2. **Softmax.** Designed in v16: an algebraic fold with weights `e^(s)`, summed in a scaled exponential accumulator (design §8.6). Recognized in both its window and its `GROUP BY` spellings (design §9.6). No log-sum-exp semiring is needed. FlashAttention, and NanoGPT through its attention and loss.
3. **Masks as absent rows in dense kernels,** found by spike S27: a masked pair must be skipped, not zeroed. FlashAttention and NanoGPT.
4. **Semi-join reduction through deep plans,** and receptive fields as bounds on support. GraphCast.
5. **Versioned optimizer state,** for XQL and ddx rather than einfold. NanoGPT.

## References

- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS 2022.* https://arxiv.org/abs/2205.14135
- Milakov, M., Gimelshein, N. (2018). Online normalizer calculation for softmax. https://arxiv.org/abs/1805.02867
- Rabe, M. N., Staats, C. (2021). Self-attention Does Not Need O(n²) Memory. https://arxiv.org/abs/2112.05682
- Wu, M., Cheng, X., et al. (2025). Mirage: A Multi-Level Superoptimizer for Tensor Programs. *OSDI 2025.* https://arxiv.org/abs/2405.05751
- Dekel, O. (2025). Blockbuster, Part 1: Block-level AI Operator Fusion. https://arxiv.org/abs/2505.07829
- Zhan, Q., Hu, X., Xia, X., Li, S. (2026). Unleashing the Power of Equality Saturation for Tensor Program Superoptimization. https://arxiv.org/abs/2609.12330
- Lam, R., et al. (2023). Learning skillful medium-range global weather forecasting. *Science* 382(6677), 1416–1421. https://arxiv.org/abs/2212.12794
- GraphCast reference implementation: https://github.com/google-deepmind/graphcast
- nanoGPT: https://github.com/karpathy/nanoGPT
- modded-nanogpt (the NanoGPT speedrun): https://github.com/KellerJordan/modded-nanogpt
- Larry Dial's record post, as quoted by Tim Kellogg on Bluesky (2026-09-29): https://bsky.app/profile/timkellogg.me/post/3mwnoi35gys2s
