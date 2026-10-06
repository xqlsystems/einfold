<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S17: n-ary sum-product nodes

**Question.** Spike S16 found that, even without associativity, binary products and nested sums grow the e-graph by about 2.2× per operand. Does representing each sum-product region as **one node over a multiset of operands** fix that? How does the e-graph grow on the Einsum Benchmark's large expressions?

**Answer.** Yes. With n-ary nodes, a pure sum-product region saturates in **one iteration**, and the e-graph grows **linearly**: about 4 tuples per operand, up to 1000 operands. The binary form failed to saturate within 2 million tuples at 16 to 26 operands.

Two costs remain:

- **Distributivity is still exponential.** `k` operands that are sums give `2^k` expanded terms, and saturation timed out at `k = 8`. It needs a guard.
- **Extraction now holds the planning cost.** With the contraction planner running inside extraction's cost model, extraction took 17–26 s at 1000 operands. Saturation took 0.2 s.

Date: 2026-10-06. egglog 3.0.0, on a 12-core Linux machine.

## What the spike does

[`nary.egg`](nary.egg) replaces S16's binary `Mul` and `Sum` with one constructor:

| Term | Meaning |
|---|---|
| `(SP ops sums)` | `Σ_sums Π ops`: a multiset of operands, joined on shared dimensions, with a set of dimensions summed out |
| `(T name)`, `(Add a b)`, `(Scale i a)` | as in S16 |

A multiset has no order and a set has no nesting, so commutativity, associativity and the reordering of sums need no rules: every ordering of a region is the same term. The rules that remain:

- **Dimension analysis.** A region's free dimensions are the union of its operands' dimensions, minus the summed ones. This is computed with egglog's container functions (`unstable-multiset-map` and `unstable-multiset-reduce`).
- **Flattening.** A region nested inside another is merged into it, when its summed dimensions appear nowhere else.
- **Distributivity, both ways.** An operand that is a sum splits the region in two; two regions that share operands are factored back.
- **Broadcast factoring.** A summed dimension that no operand has becomes a `Scale`.
- **A one-operand region with nothing summed is its operand.**

Every rule was checked on small cases before the experiments: `Σ_k A·(B+C)` expands, `Σ_{lat,t} w(lat)` gives `Scale t`, and a nested region flattens.

**Extraction plans.** In [`src/main.rs`](src/main.rs), the custom `CostModel` receives each `SP` node's operands as a container, and runs a greedy contraction planner on them inside the cost function, so extraction compares candidates by their *planned* cost. This is the "call the planner from inside extraction" option in the design doc's open questions (§14). The planner uses opt_einsum's greedy rule: repeatedly join the pair that most reduces size. Its costs match opt_einsum's greedy planner on the same instances, for example 10^8.97 vs. 10^8.97 for a 10-matrix chain, and 10^41.6 vs. 10^42.6 for a 500-node random graph.

**Instances.** [`make_instances.py`](make_instances.py) uses the Einsum Benchmark's own generators (the `einsum_benchmark` package) to build the benchmark's instance families at increasing sizes: matrix chains, random 3-regular graphs, 2-D lattices, trees, matrix product states, a MaxCut QAOA circuit, language models, and random hypernetworks. The published instances (a 552 MB download from Zenodo) were not used: at about 150 KB/s, the download did not finish within its 30-minute limit.

**Experiments.**

1. Pure sum-product: the binary form with S16's rules (limits: 30 iterations, 2 million tuples, 60 s), against the n-ary form.
2. Distributivity: the same region with `k` operands replaced by sums `(t_k + u_k)`.
3. Determinism: three runs each with 1 and 4 threads.

Run: `python make_instances.py instances.txt`, then `cargo run --release -- instances.txt 26`.

## Results

### Pure sum-product regions

"Tuples" counts rows in egglog's tables, a measure of e-graph size. The binary form was run only up to 26 operands.

| Instance | Operands | Binary: tuples / iterations / saturated / s | N-ary: tuples / iterations / saturated / s | N-ary extraction (s) |
|---|---|---|---|---|
| matrix chain | 10 | 6,182 / 18 / yes / 0.03 | 43 / 1 / yes / 0.002 | 0.000 |
| matrix chain | 25 | 3,292,566 / 10 / **no** / 6.8 | 103 / 1 / yes / 0.004 | 0.002 |
| matrix chain | 200 | — | 803 / 1 / yes / 0.017 | 0.11 |
| random 3-regular | 10 | 320,626 / 20 / yes / 1.8 | 43 / 1 / yes / 0.002 | 0.001 |
| random 3-regular | 26 | 2,393,379 / 9 / **no** / 4.5 | 107 / 1 / yes / 0.004 | 0.005 |
| random 3-regular | 100 | — | 403 / 1 / yes / 0.010 | 0.10 |
| random 3-regular | 1000 | — | 4,003 / 1 / yes / 0.19 | 16.9 |
| lattice 4 × 4 | 16 | 3,204,521 / 10 / **no** / 6.3 | 67 / 1 / yes / 0.003 | 0.002 |
| lattice 24 × 24 | 576 | — | 2,307 / 1 / yes / 0.088 | 5.0 |
| tree | 1000 | — | 4,003 / 1 / yes / 0.14 | 4.7 |
| matrix product state | 200 | — | 803 / 1 / yes / 0.019 | 0.23 |
| MaxCut, 24 qubits, p = 3 | 228 | — | 915 / 1 / yes / 0.022 | 0.46 |
| language model, depth 2 | 22 | 6,773,211 / 10 / **no** / 15.3 | 91 / 1 / yes / 0.003 | 0.004 |
| language model, depth 4 | 114 | — | 459 / 1 / yes / 0.011 | 0.14 |
| random hypernetwork | 1000 | — | 4,003 / 1 / yes / 0.14 | 25.7 |

Intermediate sizes were omitted; the full table is in the run's output.

### Distributivity

| Instance | Operands that are sums (`k`) | N-ary: tuples / iterations / saturated / s | Extraction (s) | Sums left in the extracted term |
|---|---|---|---|---|
| matrix chain, 50 | 1 | 212 / 4 / yes / 0.007 | 0.016 | 1 |
| | 4 | 487 / 6 / yes / 0.08 | 0.46 | 4 |
| | 6 | 3,141 / 8 / yes / 2.2 | 4.2 | 6 |
| | 8 | 30,595 / 8 / **no** / 96 | 67 | 8 |
| random 3-regular, 100 | 1 | 412 / 4 / yes / 0.017 | 0.29 | 1 |
| | 4 | 687 / 6 / yes / 0.33 | 7.8 | 4 |
| | 6 | 3,341 / 8 / yes / 6.8 | 70 | 6 |
| | 8 | 28,491 / 7 / **no** / 125 | 638 | 8 |

For these dense operands, extraction kept every sum factored, which is correct: expanding a sum of two dense operands doubles the work.

### Determinism

The random 3-regular graph with 100 operands and the 8 × 8 lattice, both with four sums, each run three times with 1 thread and three times with 4: one e-graph size and one extracted cost per instance.

## Findings

1. **N-ary nodes solve the growth S16 found.** Without commutativity, associativity or sum-reordering rules, a pure region is one e-node. Saturation is one pass, and the e-graph is linear in the number of operands. The binary form, even without associativity, did not saturate past 16 to 26 operands.
2. **Planning inside extraction works.** Passing the operand multiset to the cost model lets extraction compare algebraic alternatives by their planned cost. This answers the open question about coupling extraction and planning (§14), at least for greedy planning. The cost moves into extraction: at 1000 operands, the planner takes seconds. A production planner needs cotengra-style incremental search, and a cache of planned costs per multiset, since egglog calls the cost function again whenever a child's cost improves.
3. **Distributivity needs a guard.** Each sum operand doubles the expanded forms, so `k` sums give `2^k`. Extraction correctly kept the dense sums factored, but saturation still built every expansion first. Options: expand only operands with a sparsity fact that can make expansion pay (S16's case E2a); cap `k` per region; or run distributivity in its own ruleset with a node limit.
4. **Container rules must run naively.** egglog requires rules that call a function through `unstable-fn` (here, `dims` over a multiset) to run as `:naive`, rematching everything each iteration. That is harmless at one iteration, but contributes to the slow distributivity runs.
5. **Determinism held,** across thread counts, as in S16.

## Limitations

- Generated instances from the benchmark's own generators, not the published instance files.
- Dense extents only; the cost model does not use the benchmark's sparse tensors.
- The flattening rule matches every pair of `SP` nodes and filters by containment. That is quadratic in the number of `SP` nodes, which was fine here, where distributivity created at most a few thousand.
- No NULL semantics, as in S16.
