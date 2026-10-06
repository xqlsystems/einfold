<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S20: tiles in egglog

**Question.** Can splitting, retiling, block einsums and the per-task memory analysis (design doc §9.4) be encoded as egglog terms? Does that reproduce Cubed's plan for a large matrix multiplication, and does extraction never pick a plan over the memory budget?

**Answer.** Yes, on all three counts, for matrix multiplication.

- **Reproduction.** Given only the storage tilings, egglog extracts exactly Cubed's plan for aligned inputs: the same operations, task counts and projected memory, to the megabyte.
- **The budget is a hard limit.** In 200 random configurations, no extracted plan exceeded the budget. Each extracted cost equaled the minimum found by brute-force enumeration of the same plan space.
- **Optimization.** Where Cubed plans an over-budget computation and fails when it runs, egglog either finds a plan that fits or reports that none exists, at planning time.
- **Cost.** E-graphs of at most 12,000 tuples, and 16 ms to saturate and extract.

Date: 2026-10-06. egglog 3.0.0; Cubed 0.28.0 as the reference.

## Background

[Cubed](https://github.com/cubed-dev/cubed) runs array programs as a series of tasks, each within a fixed memory budget (`allowed_mem`). Its matrix multiplication `matmul(a, b)` is:

1. a `blockwise` operation with one task per block triple `(i, j, k)`, each multiplying one chunk of `a` by one chunk of `b`. This produces one partial product per block of `k`;
2. a tree of `partial_reduce` operations, each summing groups of `split_every` partials (4 by default), until one is left.

If the chunks of `a` and `b` disagree along `k`, Cubed first rechunks. Before running, it projects each task's memory:

- for `blockwise`: `reserved + Σ inputs × 2 + output × 2`;
- for `partial_reduce`: `reserved + 7 × chunk`.

It refuses to run a plan whose projection exceeds `allowed_mem`.

## What the spike does

**Terms** ([`tiles.egg`](tiles.egg)). Each e-class is one array *at one tiling*. Tilings of the same values are different e-classes, linked by `Retile` terms, so that a parent needing a particular tiling always finds it:

| Term | Meaning | Cubed |
|---|---|---|
| `(Stored a)` | array `a` in its storage tiling | an input array |
| `(View a r c)` | array `a` tiled `(r, c)` | — |
| `(Retile x r c)` | change `x`'s tiling | `rechunk` |
| `(BlockMM x y)` | one task per block triple; partials along `k` | `blockwise` with `_matmul` |
| `(PartialSum x s)` | sum groups of `s` partials | `partial_reduce` |
| `(Partials a b ti tk tj m)` | the partial products at tiles `(ti, tk, tj)`, with `m` blocks left along `k` | an intermediate array |
| `(Result a b ti tj)`, `(MatMul a b)` | the product at a tiling, and at any tiling | the output |

Five rules build the space: views by retiling from storage, block products for every combination of candidate tiles, rounds of partial sums for every candidate fan-in, and the result once one block is left.

**Candidates.** Tile sizes are numbers, so the space is infinite. As §9.4 proposes, a planner (here, a function in [`src/main.rs`](src/main.rs)) proposes candidates per dimension, and the rules combine them. Two proposers are used:

- **storage only:** each dimension gets just the storage tile sizes along it. That is the space Cubed works in;
- **proposed:** storage sizes, their least common multiples, and halvings of the extent (20,000; 10,000; …; 625).

**Cost model.** A custom egglog `CostModel` computes each node's projected memory with Cubed's formulas, plus IO bytes and task count. A plan's cost is `IO bytes + 10 MB × tasks`, or infinity if any task's projected memory exceeds the budget. Infinity propagates, so no over-budget node can be part of an extracted plan.

**Reference.** [`cubed_reference.py`](cubed_reference.py) builds the same multiplications in Cubed, without computing them, and prints each operation of the finalized plan.

To reproduce: `python cubed_reference.py` and `cargo run --release`.

## Results

All cases multiply two 20,000 × 20,000 float64 matrices, with 100 MB of reserved memory.

### Reproducing Cubed (storage tilings, fan-in 4)

| Case | Cubed | egglog |
|---|---|---|
| `a` chunks 5000 × 2000, `b` 2000 × 5000; 2 GB | blockwise: 160 tasks, 820 MB; partial_reduce: 48 tasks, 1500 MB; partial_reduce: 16 tasks, 1500 MB | **identical** |
| Square chunks 2000; 1 GB | blockwise: 1000 tasks, 292 MB; partial_reduce: 300 tasks, 324 MB; partial_reduce: 100 tasks, 324 MB | **identical** |
| `k` misaligned: `a` 2000 × 2000, `b` 3000 × 2000; 1 GB | rechunk `b` in two stages (40 tasks, 484 MB; 20 tasks, 772 MB); then as above | rechunk `a` to 2000 × 3000 (70 tasks, 260 MB); blockwise 700 tasks; two rounds of partial sums. Different, but within budget (max 356 MB) |
| Chunks 8000 × 8000; 1 GB | builds a plan with tasks up to **3684 MB**, which fails when run | **no plan fits the budget**, reported at planning time |

The misaligned case differs because the spike's retile is a single-stage approximation of rechunker's algorithm, so it costs rechunking `a` and `b` differently than Cubed does.

### Optimizing (proposed tilings)

With more candidate tile sizes, extraction trades rechunking IO against partial-product IO:

| Case | Extracted plan | Max memory | Cost vs. Cubed's plan |
|---|---|---|---|
| 5000 × 2000 / 2000 × 5000; 2 GB | rechunk both so `k` is one block (2500 × 20,000 and 20,000 × 2500); one blockwise; no partial sums | 1800 MB | 0.60× |
| Square 2000; 1 GB | rechunk to 5000 × 5000 and 5000 × 2500; blockwise with 4 partials; one partial sum | 900 MB | 0.50× |
| Misaligned; 1 GB | rechunk to 5000 × 5000 and 5000 × 3000; blockwise with 4 partials; one partial sum | 980 MB | 0.55× |

Cubed's matrix multiplication writes one full-size partial product per block of `k`: ten 3.2 GB arrays in the first case. Rechunking the inputs once is cheaper under any cost model that counts IO.

Making the fan-in free (2, 4, 8 or 16) changed nothing in these three cases.

### Budget sweep (storage chunks 8000 × 8000, fan-in free)

| Budget | Extracted plan | Max memory |
|---|---|---|
| 1000 MB | no plan: reading one 512 MB storage chunk, with Cubed's read copy, already needs 1124 MB | — |
| 1200 MB | rechunk to 2500 × 1250 and 1250 × 2500; 16 partials; one partial sum of fan-in 16 | 1174 MB |
| 1300 MB | rechunk to 2500 × 2500; 8 partials; fan-in 8 | 1224 MB |
| 1500 MB | rechunk to 5000 × 2500 and 2500 × 5000; 8 partials; fan-in 8 | 1500 MB |
| 2000 MB | rechunk to 5000 × 8000 and 8000 × 5000; 3 partials; fan-in 4 | 1780 MB |
| 3000 MB | rechunk to 8000 × 10,000 and 10,000 × 5000; 2 partials; fan-in 2 | 2820 MB |
| 4000 MB | rechunk so `k` is one block; no partial sums | 3700 MB |

As the budget grows, tiles grow, and partial sums shrink and then disappear. With little memory, the fan-in rises to 8 or 16, because a partial sum reads its inputs one at a time, so in Cubed's model its memory does not depend on fan-in.

### Randomized check

200 configurations: storage tile sizes drawn from {1000, 2000, 2500, 3000, 4000, 5000, 8000, 10,000} for each dimension of each input, and budgets from 150 MB to 4 GB.

- Extracted plans over budget: **0**.
- Configurations with no feasible plan: 71. Brute force agreed in every one.
- Extracted cost different from the brute-force minimum: **0**.

## Findings

1. **Tiles fit the e-graph if a tiling is part of an e-class's identity.** Putting different tilings of the same values in one e-class would let extraction pick, for a child, a tiling its parent can't use. Keeping one e-class per (array, tiling), linked by `Retile`, makes extraction exact: it matched brute force every time. This is the "physical property" pattern of Cascades-style optimizers (which plan for properties such as sort order alongside the plan itself), expressed as terms.
2. **The memory budget works as an infinite cost.** No rule needs to know the budget. The cost model applies Cubed's projection to each node, and infinity propagates. Infeasibility is reported at planning time, not when a task fails.
3. **Cubed's plan is one point in the space, and often not the cheapest.** Restricted to the storage tilings, egglog reproduces Cubed exactly. With proposed tile sizes, it finds plans with 40–50% lower estimated cost, chiefly by rechunking once to avoid writing many full-size partial products. Cubed doesn't search because it executes the chunking the user chose. einfold has to search, because its users write SQL and choose no chunking.
4. **The proposer decides what is reachable.** Every plan the e-graph found used only proposed sizes. That validates the §9.4 split (a planner proposes, extraction chooses), and also locates its main risk: if the proposer misses the right size, extraction cannot recover it.
5. **Some workloads cannot fit, and that must be reported.** With 512 MB storage chunks and a 1 GB budget, no plan exists under Cubed's memory model, because each storage chunk must be read whole. einfold's equivalent is to decline the tiling rewrite and leave the host's plan as it was, which may spill to disk or fail on its own terms. Missing a speedup is acceptable (design doc principle 4); making a query fail that the host could have run is not.
6. **It is cheap.** At most 11,621 tuples and 16 ms per case, including extraction.

## Limitations

- Only matrix multiplication, with two 2-D inputs. General einsums need the same terms per dimension: `View` over a tuple of tile sizes, and `BlockMM` generalized to a block einsum. The rules are the same in shape.
- Retiling is approximated as a single stage that holds one source and one target tile per task. Cubed uses rechunker's multi-stage algorithm, which is why the misaligned case differs.
- The cost model counts IO bytes plus a fixed per-task overhead. It does not model parallelism, so it doesn't capture the main reason for Cubed's default fan-in of 4 (keeping many tasks in flight).
- In einfold, a tile is a partition key in a SQL plan, not a Zarr write (§9.4, step 4). The memory formulas here are Cubed's. A host's memory per partition would have to be modeled separately for each host.
