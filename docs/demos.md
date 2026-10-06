<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# einfold target demos

Status: draft v1. Author: Alex Merose. Last updated: 2026-10-06.

This doc describes three demos that would show what XQL, ddx, and einfold can do together. Each also serves as a check on einfold's design ([`design.md`](design.md)): it says which parts of the design it exercises, and which gaps it exposes.

The three projects, briefly:

- **XQL** gives SQL access to large scientific arrays stored in Zarr, a chunked array format. A dataset becomes a table with one row per combination of coordinates.
- **ddx** differentiates SQL queries: given a query that computes a function, it produces the queries that compute its gradients. That makes model training expressible in SQL.
- **einfold** rewrites query plans so that the SQL engine running them (the *host*) computes tensor contractions fast. It never executes anything itself.

The demos are not meant to beat specialized systems on raw speed. They are meant to show that a declarative, relational description of a model, plus a principled optimizer, can recover the algorithmic ideas that those systems were hand-engineered to exploit.

## Demo 1: Rediscovering FlashAttention automatically

### The idea

Attention, the core of a transformer, computes `softmax(QKᵀ/√d)·V` for query, key and value matrices `Q`, `K`, `V`. A naive implementation materializes the full `N×N` matrix of scores for `N` tokens. FlashAttention (Dao et al., 2022) never does. It splits `Q`, `K` and `V` into tiles. For each tile of queries, it streams over tiles of keys and values, and keeps a running softmax that it rescales as larger scores appear (the "online softmax", Milakov and Gimelshein, 2018). Memory per tile stays small, and with a causal mask (each token attends only to earlier tokens), tiles above the diagonal are skipped entirely.

The demo: write attention in plain SQL, and have einfold produce a plan with FlashAttention's structure, without being told about FlashAttention.

### Attention in SQL

For one attention head, with `Q`, `K` and `V` stored as `(t, d, v)` tables (token, feature, value):

```sql
WITH scores AS (
  SELECT q.t AS i, k.t AS j, SUM(q.v * k.v) / 8.0 AS s   -- 8 = sqrt(64 features)
  FROM Q q JOIN K k ON q.d = k.d
  WHERE q.t >= k.t                                       -- causal mask
  GROUP BY q.t, k.t),
weights AS (
  SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS w
  FROM scores),
norm AS (
  SELECT i, SUM(w) AS z FROM weights GROUP BY i)
SELECT w.i AS t, v.d, SUM(w.w * v.v) / n.z AS v
FROM weights w
JOIN V v ON w.j = v.t
JOIN norm n ON w.i = n.i
GROUP BY w.i, v.d, n.z;
```

### How einfold would find FlashAttention's plan

| FlashAttention's idea | einfold's mechanism | Status in the design |
|---|---|---|
| Skip tiles above the diagonal | The predicate `q.t >= k.t` becomes a mask operand, and the block-sparse algorithm skips absent tiles and masks the diagonal ones | Designed (design.md §9.1, §10.2) |
| Work tile by tile with bounded memory | Tiles in the algebra: block einsums with a per-task memory budget | Designed (§8.2, §9.4); prototyped for matrix multiplication in spike S20 |
| Never store join rows for `QKᵀ` or for the product with `V` | The EinFold join fuses each join with its sum | Designed (§10.2) |
| Never store the `N×N` scores *between* the two products | Fuse the scores, the softmax and the product with `V` into one task over (query tile, key tile) pairs | **Gap:** fusion across einsums, with a nonlinear step between them |
| The online softmax: a running maximum and sum, rescaled as larger scores arrive | The partial aggregate for each output row becomes the triple (maximum, normalizer, weighted sum), combined with `m = max(m₁, m₂)`, `z = z₁·e^(m₁−m) + z₂·e^(m₂−m)`, `a = a₁·e^(m₁−m) + a₂·e^(m₂−m)` | **Gap:** partial aggregates (§8.3) cover `SUM` only. This combine is associative, so it generalizes cleanly. It is the "log-sum-exp" semiring, which ties into the semiring open question (§14). |
| Recognize the softmax in SQL | Detect `exp(s − MAX(s) OVER (PARTITION BY i))`, normalized by a sum over the same partition | **Gap:** detection does not yet look at window functions |

So the demo is achievable, and the gaps are specific: generalized partial aggregates, fusion across einsums, and softmax detection.

### Success criteria

- The plan never stores the `N×N` scores. Peak memory per task grows with the tile size, not with `N`.
- With the causal mask, about half the key tiles are skipped.
- Results match the naive SQL plan within floating-point tolerance, and repeat bit for bit in the reference executor.
- Measured on einfold's DataFusion reference executor against the naive plan on DataFusion and DuckDB.

FlashAttention's speed on GPUs also comes from managing on-chip memory, and that is the host's job. einfold produces the plan's *structure*. Running it at GPU-kernel speed requires a host that implements the `Einsum` relation, such as Sirius or NVIDIA's GPU Query Engine.

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
| Mask operands and partly masked tiles | ✓ | ✓ | |
| EinFold join (fused join and sum) | ✓ | ✓ | ✓ |
| Tiles with a memory budget (S20) | ✓ | | ✓ |
| Derived factors | ✓ | ✓ | ✓ |
| Program mode | | ✓ | |
| Exact zero elimination, declared or inferred | | ✓ | |
| Semi-join reduction and pushdown into reads | | | ✓ |
| Zarr facts and the layout convention | | | ✓ |
| ddx gradients | | ✓ | |

Design gaps found, in order of how many demos need them:

1. **Generalized partial aggregates.** Associative combines beyond `SUM`, such as the online softmax's (maximum, normalizer, weighted sum). FlashAttention, and NanoGPT through its attention and loss.
2. **Fusion across einsums** with a nonlinear step in between. FlashAttention and NanoGPT.
3. **Detecting window functions,** such as `MAX(s) OVER (PARTITION BY i)` in a softmax. FlashAttention and NanoGPT.
4. **Semi-join reduction through deep plans,** and receptive fields as bounds on support. GraphCast.
5. **Versioned optimizer state,** for XQL and ddx rather than einfold. NanoGPT.

## References

- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS 2022.* https://arxiv.org/abs/2205.14135
- Milakov, M., Gimelshein, N. (2018). Online normalizer calculation for softmax. https://arxiv.org/abs/1805.02867
- Lam, R., et al. (2023). Learning skillful medium-range global weather forecasting. *Science* 382(6677), 1416–1421. https://arxiv.org/abs/2212.12794
- GraphCast reference implementation: https://github.com/google-deepmind/graphcast
- nanoGPT: https://github.com/karpathy/nanoGPT
- modded-nanogpt (the NanoGPT speedrun): https://github.com/KellerJordan/modded-nanogpt
- Larry Dial's record post, as quoted by Tim Kellogg on Bluesky (2026-09-29): https://bsky.app/profile/timkellogg.me/post/3mwnoi35gys2s
