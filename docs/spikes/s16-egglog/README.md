<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S16: einfold's rewrites in egglog, with a hybrid planner

**Question.** Can egglog, an equality-saturation engine, serve as einfold's logical optimizer? Specifically: does it find einfold's key algebraic rewrites, how does its search grow with einsum size, and are its results deterministic?

**Answer.** Yes, for the algebraic rewrites, in a **hybrid design**. egglog handles the rewrites; a specialized planner chooses contraction order. If associativity is left to egglog, the search explodes, and once the search is cut off, it can return far worse plans.

Date: 2026-10-05. egglog 3.0.0, Rust 1.91.1, on a 12-core Linux machine.

## Background

An **e-graph** stores many equivalent versions of an expression compactly, by grouping equal subexpressions into classes. **Equality saturation** applies rewrite rules until no new equivalent forms appear. **Extraction** then picks the cheapest version under a cost model. [egglog](https://github.com/egraphs-good/egglog) is an e-graph engine that also supports Datalog-style rules for deriving facts.

## What the spike does

[`rules.egg`](rules.egg) encodes the sum-product core of einfold's einsums as egglog terms:

| Term | Meaning |
|---|---|
| `(T name)` | an operand: one variable over its own dimensions |
| `(Mul a b)` | pointwise product: a join on shared dimensions |
| `(Add a b)` | pointwise sum |
| `(Sum i a)` | sum out dimension `i`: a `GROUP BY` on the rest |
| `(Scale i a)` | multiply by the extent of `i` |

It has:

- **An analysis, written as Datalog rules,** that computes each class's free dimensions.
- **Algebraic rewrites,** from sections 9.1 and 10.1 of the design doc:
  - commutativity of products;
  - distributivity in both directions;
  - eager aggregation (push a sum into the factor that owns the dimension, and pull it back out);
  - a sum distributing over addition;
  - broadcast factoring (`Σᵢ B = nᵢ·B` when `B` does not depend on `i`);
  - reordering nested sums.
- **Associativity of products,** in a separate ruleset, so its cost can be measured.

[`src/main.rs`](src/main.rs) is the driver:

1. **Saturation** in steps, with limits of 30 iterations, 2 million tuples, and 60 s.
2. **Extraction** with a custom egglog `CostModel`. It estimates rows with the same uniform-density formula the design doc uses (section 9.3). A join's rows are `rows(A)·rows(B)/Π(shared extents)`, capped at the dense size. Each operator's cost is the rows it reads or produces.
3. **Planning (hybrid only).** A planner reorders every region of products and sums. It is a dynamic program over subsets of operands, and it sums dimensions out as early as possible.

It compares two configurations:

- **Hybrid:** the algebra rules without associativity, followed by the planner.
- **Full AC:** the algebra rules plus associativity in egglog, with no planner.

To reproduce:

```sh
cargo run --release
```

## Results

### Algebraic rewrites (E1–E4)

Costs are estimated rows touched. "Tuples" counts rows in egglog's tables, a measure of e-graph size.

| Problem | Input cost | Hybrid cost | Full-AC cost | Hybrid result | Hybrid: tuples / iterations / saturated / seconds | Full AC: tuples / iterations / saturated / seconds |
|---|---|---|---|---|---|---|
| E1 `sum(W·H)` (SPORES's PNMF example) | 2.010e12 | 4.000e7 | 4.000e7 | `Σk (Σj H * Σi W)` | 37 / 6 / yes / 0.001 | 37 / 6 / yes / 0.001 |
| E2a `X·(A+B)`, `X` sparse | 4.000e8 | 2.001e8 | 2.001e8 | `(Σj Σi (X * A) + Σj Σi (X * B))` | 43 / 5 / yes / 0.001 | 43 / 5 / yes / 0.001 |
| E2b `X·(A+B)`, `X` dense | 7.000e8 | 7.000e8 | 7.000e8 | `Σj Σi (X * (A + B))` | 43 / 5 / yes / 0.001 | 43 / 5 / yes / 0.001 |
| E3a `Σ w(lat)·x(t,lat,lon)` by `t` | 3.115e9 | 2.078e9 | 2.078e9 | `Σlat (w * Σlon x)` | 19 / 3 / yes / 0.000 | 19 / 3 / yes / 0.000 |
| E3b `Σ_{t,lat,lon} w(lat)` | 2.164e3 | 1.444e3 | 1.444e3 | `n_t·n_lon·Σlat w` | 32 / 4 / yes / 0.000 | 32 / 4 / yes / 0.000 |
| E4 ddx gradient, written `X·(Ȳ·W2)` | 8.972e8 | 1.402e7 | 1.402e7 | `Σo (Σn (X * Ybar) * W2)` | 24 / 3 / yes / 0.000 | 38 / 4 / yes / 0.001 |

- **E1:** egglog found SPORES's rewrite `sum(WH) = Σk (Σi W)(Σj H)`, a 50,000× reduction in estimated cost.
- **E2:** distributivity is decided by cost. The product is expanded when `X` is sparse and kept factored when `X` is dense, which is the case SPORES and Galley make for cost-based distributivity.
- **E3:** broadcast factoring works, both pushing a sum past a repeated weight and turning a sum over a missing dimension into a scale factor.
- **E4:** a bad written order of ddx's two-layer gradient, `W̄1[d,h] = Σn Σo X[n,d]·Ȳ[n,o]·W2[h,o]`, is fixed 64× by both configurations. The hybrid gets there through the planner; full AC through associativity.

### Scaling with einsum size: matrix chains (E5)

A dense chain `A1·A2·…·An`, written right-nested, with extents chosen so that the order matters.

| n | Hybrid: tuples / iterations / saturated / seconds | Planner seconds | Hybrid cost | Full AC: tuples / iterations / saturated / seconds | Full-AC cost |
|---|---|---|---|---|---|
| 3 | 24 / 3 / yes / 0.000 | 0.0000 | 4.710e5 | 38 / 4 / yes / 0.001 | 4.710e5 |
| 4 | 54 / 4 / yes / 0.000 | 0.0001 | 1.641e6 | 149 / 6 / yes / 0.001 | 1.641e6 |
| 5 | 120 / 5 / yes / 0.001 | 0.0003 | 4.535e5 | 578 / 7 / yes / 0.003 | 4.535e5 |
| 6 | 266 / 6 / yes / 0.001 | 0.0011 | 2.464e6 | 2205 / 8 / yes / 0.012 | 2.464e6 |
| 7 | 588 / 7 / yes / 0.002 | 0.0038 | 1.614e6 | 8305 / 10 / yes / 0.050 | 1.614e6 |
| 8 | 1294 / 8 / yes / 0.004 | 0.0141 | 2.028e6 | 31032 / 11 / yes / 0.224 | 2.028e6 |
| 9 | 2832 / 9 / yes / 0.009 | 0.0493 | 1.858e6 | 115450 / 12 / yes / 1.037 | 1.858e6 |
| 10 | 6162 / 10 / yes / 0.019 | 0.1558 | 2.892e6 | 428663 / 13 / yes / 5.136 | 2.892e6 |
| 11 | 13332 / 11 / yes / 0.044 | 0.5155 | 2.238e6 | 1590696 / 13 / yes / 24.347 | 2.238e6 |
| 12 | 28694 / 12 / yes / 0.102 | 1.7622 | 3.039e6 | 3080186 / 6 / **no** / 21.607 | **2.420e12** |

- **Full AC grows about 3.7× per extra matrix,** reaching 1.6 million tuples and 24 s at n = 11. At n = 12 it hit the tuple limit before saturating, and extraction returned a plan **800,000× worse** than the hybrid's. An unsaturated e-graph gives no quality guarantee at all.
- **When both saturate, both find the same optimum.** The planner's dynamic program is exhaustive, so the hybrid loses nothing.
- **The hybrid's e-graph also grows, about 2.2× per matrix,** because commutativity and reordering nested sums still multiply the variants. That was fine here (29,000 tuples and 0.1 s at n = 12), but it will not scale to einsums with hundreds of operands.
- **The planner's dynamic program grows as 3ⁿ** (1.8 s at n = 12). As the design doc already says, larger einsums need a greedy or hyper-optimized planner.

### Determinism (E6)

Each of E1–E4 was run three times with 1 thread and three times with 4 threads, in both configurations.

| Problem | Distinct extracted terms (hybrid / full AC) | Distinct e-graph sizes (hybrid / full AC) |
|---|---|---|
| E1 | 1 / 1 | 1 / 1 |
| E2a | 1 / 1 | 1 / 1 |
| E2b | 1 / 1 | 1 / 1 |
| E3a | 1 / 1 | 1 / 1 |
| E3b | 1 / 1 | 1 / 1 |
| E4 | 1 / 1 | 1 / 1 |

All runs agreed, including across thread counts. The spike uses no random sampling of rule matches, which SPORES needed in order to avoid blowup.

## Findings

1. **egglog expresses einfold's algebraic layer naturally.** Dimension analysis is three Datalog rules. Each rewrite from sections 9.1 and 10.1 is one rule, with its side condition (such as "`i` is not a dimension of `a`") written as a query. Cost-based distributivity, one of the hardest decisions to hand-code, comes from extraction for free.
2. **Leave contraction order to a planner.** Associativity in the e-graph is exponential, and when the search is cut short the result can be far worse than the input. This confirms the hybrid design.
3. **Even without associativity, n-way products and nested sums grow the e-graph.** For large einsums, represent each sum-product region as one n-ary node, holding a multiset of operands and a set of summed dimensions. egglog has multisets built in, and its own examples use them to replace associativity and commutativity rules.
4. **Hand-written planners must keep the algebra's invariants.** The first run had a planner bug: it dropped a sum over a dimension that no operand has, instead of turning it into a scale factor. The egglog rule got this right. Planner and rules should share one set of tests (section 11 of the design doc).
5. **Extraction cost models are straightforward to write.** egglog's `CostModel` trait accepts a custom cost type, and our row estimates fit in about 60 lines.
6. **Saturation was deterministic here.** einfold should still fix rule schedules and avoid random sampling, and test determinism on larger e-graphs (section 8.6).

## Limitations

- The cost model is the simple uniform-density estimate, with tree cost, which counts shared subexpressions twice.
- The planner runs after extraction, so extraction chooses algebraic forms without seeing the planner's order. In these cases the result matched the full search wherever that search finished. Calling the planner from inside extraction, or extracting several candidates, is future work.
- `Scale` uses extents, which is exact only for complete dense arrays. The design doc's rule uses row counts after filters.
- No SQL NULL semantics. NULL behavior belongs in the rules' side conditions and in the shared tests.
- Small problems only, for the algebra (E1–E4). Large einsums were tested only as matrix chains.
