<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# RFC 0001: Folds over joins, not just einsums

- **Status:** Accepted in direction by @alxmrs ("take the general shape now"); the ontology review below is proposed
- **Author:** 🧭 Claude (orchestrator), from @alxmrs's review of #6 and #7
- **Design sections affected:** most of them. The vocabulary (§6), output forms (§7.2), partial aggregates (§8.3), detection (§9.1), and EinFold (§10.2) change most; see "Refactoring the docs".

## Summary

einfold's design is centered on the *einsum*: `SUM` of products over joined tables. This RFC makes the central object more general: a **fold over a join**, meaning any aggregate whose partial results can be combined, applied to a value computed from each joined row. Einsums become the most important special case, not the definition. The fused join-and-aggregate operator (EinFold) works for every fold. The algebraic rewrites (summing early, choosing contraction order) work for the subset of folds that form a **semiring**, which includes einsums but also `MIN` of sums, `MAX` of products, counting and existence. Some aggregates that are not semiring folds, notably `AVG` of products, decompose exactly into semiring folds, and so get the full set of rewrites.

## Motivation

The review of #6 and #7 asked whether the project over-samples one use case. It does:

> "I think this may oversample on ein *sums*. I'd rather generically support groupjoins. What if someone wanted to take an average of products instead? I think that should also be made fast." (@alxmrs, #7)

Concretely:

- **Users write more than `SUM`.** `AVG(a.v * b.v)`, `COUNT(*)` over a join, `MAX(a.score + b.score)` and `MIN(a.cost + b.cost)` (shortest paths) all build the same huge intermediate join that motivated einfold (design §3, problem 1). Today detection declines them, and they stay slow.
- **The project's own demos already need it.** The FlashAttention demo ([`demos.md`](../demos.md)) found that its online softmax needs a partial aggregate of three values (running maximum, normalizer, weighted sum) combined by a rule other than addition. That is a fold that is not a `SUM`.
- **The name already says it.** "einfold" joins *einsum* with *fold*, functional programming's word for a reducing pass. The fold is the general idea; the code should name it.
- **It is cheap to change now and expensive later.** The intermediate representation (#6) and the operator (#14, #16) are in review, not merged.

## Background: three levels of structure

A query of the shape

```sql
SELECT g…, AGG(f(row)) FROM t1 JOIN t2 … GROUP BY g…
```

has more or less structure depending on `AGG` and `f`. Each level permits more optimizations.

**Level 1, a fold over a join.** `AGG` is any aggregate whose state over part of the input can be *merged* with another part's state, associatively and commutatively, with an identity. In algebra this is a commutative monoid: `SUM`, `COUNT`, `MIN`, `MAX`, `AVG` (whose state is a sum and a count), `BOOL_OR`, and the online-softmax triple are all such folds. `f` is any expression over the joined row. What this level permits:

- **Fusing the join with the aggregate.** Every joined row updates its group's state directly, and the join's rows are never materialized. This is the EinFold operator (design §10.2), and it is what fixes problem 1.
- **Partial aggregates.** Computing states per partition, per chunk or per tile, and merging them later (design §8.3).

What it does not permit: moving the aggregate below the join, or reordering the joins of a multi-way query. Those need level 2.

**Level 2, a semiring fold.** `f` is a product, under an operation ⊗, of factors that each read one input table, and `AGG` is an operation ⊕ such that ⊗ distributes over ⊕: `a ⊗ (b ⊕ c) = (a ⊗ b) ⊕ (a ⊗ c)`. This structure is a semiring, and it is exactly what makes eager aggregation correct (design §10.1). The FAQ framework (functional aggregate queries, Abo Khamis, Ngo and Rudra, 2016) studies these queries in general. Examples:

| ⊕ (the aggregate) | ⊗ (the product) | SQL | Use |
|---|---|---|---|
| `SUM` | `*` | `SUM(a.v * b.v)` | einsums: linear algebra, ML |
| `SUM` | `*`, with values 1 | `COUNT(*)` | counting join results ("eager count") |
| `MIN` | `+` | `MIN(a.cost + b.cost)` | shortest paths (the tropical semiring) |
| `MAX` | `+` | `MAX(a.score + b.score)` | best paths, Viterbi decoding |
| `MAX` | `*`, non-negative values only | `MAX(a.p * b.p)` | most likely explanations |
| `OR` | `AND` | `EXISTS`, `BOOL_OR` | reachability, semi-joins |

What this level adds: eager aggregation, contraction-order planning, variable separation, distributing and factoring, pruning (a ⊕-identity factor annihilated by ⊗, such as an exact zero), and tiling as nested folds. These are the rewrites in design §9 and §10.1.

**Level 3, decomposable aggregates.** Some aggregates are not semiring folds but can be computed exactly from several of them. `AVG(a.v * b.v)` is `SUM(a.v * b.v) / COUNT(a.v * b.v)`, and both parts are sum-product folds over the same join, so each gets every level-2 rewrite. Yan and Larson's "eager count", which the design already cites, is exactly this decomposition. `VAR` and `STDDEV` decompose similarly, into sums of products and of squares.

## Proposal

1. **The central object is the fold over a join.** In the design, the vocabulary and the code, einfold optimizes *folds over joins*. An einsum is the sum-product semiring fold: the most important case, and the one most of the docs teach with. The project and the operator keep their names: *einfold* (einsum + fold) and *EinFold*.
2. **Detection classifies every query at the strongest level it provably meets.** Level 1 gets the fused operator; levels 2 and 3 also get the algebraic rewrites. M1 recognizes `SUM`, `COUNT` and `AVG`; `MIN`, `MAX` and existence follow in M2 with their semiring checks.
3. **Each aggregate carries its SQL rules:** how NULL inputs behave, what an empty group returns, and whether results are exact (`COUNT`, `MIN` and `MAX` are exact on any type; float `SUM` and `AVG` are not). The exactness invariant (design §8.6) applies per aggregate. Some semirings hold only under conditions (`MAX` of products needs non-negative factors), which become facts detection must prove.

## Ontology review

With the fold as the center, I reviewed every concept in the docs and in the code in review, layer by layer. For each, the question is whether its name and its definition match the thing.

### The data layer (XQL): mostly right

*Dataset table*, *variable*, *dimension*, *coordinate*, *position*, *extent*, *chunk*, *tile*, *layout*, *support* and *fill value* describe the data, not the query, so the fold doesn't change them. Two need sharper definitions:

- **Dimension** means two things today: an axis of the data (`lat`), and, in the code, a set of columns that a query equates (`a.k = b.k`). In the fold ontology the second sense is primary. A **dimension** is one variable of a fold: a set of columns the query equates, together with how it compares them (`=` or `IS NOT DISTINCT FROM`). A data axis becomes a dimension when a query uses it. The FAQ literature calls these *variables*; we keep "dimension" because it's what array users say.
- **Operand** was defined as "one variable over its own dimensions". In the fold ontology, an operand is any input to the fold: a table, a subquery, or a mask. A variable is the most common kind of operand, not the definition.

### The query layer: the fold replaces the einsum

| Concept | Definition | Replaces |
|---|---|---|
| **Fold** (fold over a join) | Inputs (operands), their join (dimensions), output dimensions (the `GROUP BY`), a row value computed from each joined row, and an aggregate that folds the row values in each group | *einsum* as the central object |
| **Aggregate** | A commutative monoid with SQL semantics: how to combine, its identity, NULL handling, the value of an empty group, and whether it's exact. `SUM`, `COUNT`, `MIN`, `MAX`, `AVG`, `BOOL_OR` | `SUM` hard-coded |
| **Row value** | What each joined row contributes: either a **product of factors**, one per operand, or an arbitrary expression | the implicit product |
| **Factor** | An operand's contribution to a product: an expression over that operand's columns only (formerly "derived factor" when computed) | unchanged |
| **Semiring fold** | A fold whose row value is a ⊗-product of factors, and whose aggregate is a ⊕ that ⊗ distributes over. A *derived* property of the fold, not a field set independently | `Semiring` as a free-standing field |
| **Einsum** | The sum-product semiring fold | was the center; now a special case |
| **Mask** (support-only operand) | An operand whose factor is ⊗'s identity, so it filters which rows join and contributes no value. This is semiring-independent: 1 for sum-product, 0 for min-plus, `TRUE` for existence | "mask operand", defined only for products |

### The evaluation layer: separate three things the code merges

| Concept | Definition | Today |
|---|---|---|
| **Group existence** | Whether any joined row reached a group. Identical for every aggregate, because SQL creates a group exactly when some row reaches it | the `matched` half of `PartialSum` |
| **Aggregate state** | The aggregate's own partial state: a sum; a sum and a count; a minimum | the `value` half of `PartialSum` |
| **Partial aggregate** | Group existence plus aggregate state, for part of the input, mergeable with other parts. The term stays, because it's the database-standard name and applies to every fold | `PartialSum` |
| **Accumulator** | *How* a state's ⊕ is computed numerically: a plain `f64` sum, a binned reproducible sum, and so on. This is where determinism and precision live (design §8.6) | unnamed; was going to be inside `PartialSum` |
| **Contraction** | One step of a semiring fold's evaluation: join two operands (⊗), and fold away the dimensions no one else needs (⊕). Defined for every semiring, not just sum-product | used only for sum-product |
| **Contraction tree** | A binary tree of contractions that evaluates a semiring fold. Level-1 folds don't have one: their aggregate cannot move below joins | unchanged, but scoped |

### The physical layer: name operators after what they do

| Concept | Definition | Today |
|---|---|---|
| **EinFold** | The fused join-and-fold operator: a join whose rows update partial aggregates directly, so join rows never exist. Works for every fold | correct, but described for `SUM` only |
| **EinFold's algorithms** | *Hash* (Gustavson's algorithm, generalized), *dense*, *block-sparse*. Dense and block-sparse need a semiring with fast dense kernels: GEMM for sum-product, and a tropical "GEMM" for min-plus, which hosts may lack | "EinFoldHashJoin" for the hash algorithm |
| **Relational form** | einfold's output as standard joins and aggregates | unchanged |
| **Extension form** | einfold's output as a Substrait extension relation, **`Fold`**, that carries a fold and its facts | "einsum form", "`Einsum` relation" |
| **Reference executor** | einfold's own implementation of the extension form, in DataFusion | `EinsumExec` |

### The system layer: right as is

*Host*, *reader*, *fact*, *fact provider*, *target profile*, *program mode* and *spike* name roles in the system, and are unaffected.

### Structural findings beyond naming

1. **The semiring should be derived, not declared.** The code stores a `Semiring` field next to the operands. Whether a fold *is* a semiring fold follows from its row value and aggregate, and sometimes needs facts (non-negativity). A free-standing field can contradict them. The optimizer should ask the fold.
2. **Group existence belongs to the group, not the sum.** Splitting existence from the aggregate state makes every aggregate inherit the subtle SQL rule (an all-NULL group is NULL, not absent) for free, instead of each one reimplementing it.
3. **Output dimensions and output columns are different things.** `GROUP BY a.k, b.k` with `a.k = b.k` has two output columns but one output dimension. The code handles this in the rule (D1). The ontology should name both, so later work doesn't conflate them.
4. **One key equality per dimension is a representation choice with a precondition.** SQL attaches `=` or `IS NOT DISTINCT FROM` to each join edge, not to a dimension. Storing one per dimension is valid only because detection declines dimensions whose edges disagree. That precondition should be stated where the representation is defined.
5. **Facts attach at three granularities:**
   - to a dimension: extent;
   - to an operand: size, density;
   - to an operand's use of a dimension: its coordinate map and its tiling, since two operands may tile the same dimension differently.

   Fact kinds (M3) should be organized that way.

## Renames

Names in docs and code, old to new. Exact type and function shapes are the implementer's choice (AGENTS.md); these are the concepts the names must express.

| Where | Old | New |
|---|---|---|
| Docs, everywhere | einsum (as the central object) | fold over a join; *einsum* only for the sum-product case |
| Docs | einsum form; `Einsum` relation | extension form; `Fold` relation |
| Docs | EinsumIR | fold IR (or just "the IR") |
| Docs | EinFoldHashJoin | EinFold's hash algorithm |
| Docs | mask operand | mask (a support-only operand) |
| Docs, title | "Fast Tensor Contractions for the XQL Model" | "Fast Folds over Joins, for Tensors and the XQL Model" |
| `einfold-ir` | `Einsum`, `EinsumError`, module `einsum` | `Fold`, `FoldError`, module `fold`, with a convenient way to build the sum-product case |
| `einfold-ir` | `Semiring` field | the aggregate, plus the row value; the semiring derived from them |
| `einfold-ir` | `PartialSum` | group existence plus an aggregate state, with `SUM`, `COUNT` and `AVG` states for M1 |
| `einfold-datafusion` | `kernel::contract`, `SUM_COLUMN` | a kernel named for the fused fold, and an output column named for the aggregate's value |
| `einfold-datafusion` | `EinsumExec` | `EinFoldExec` (the operator's name, with DataFusion's `Exec` suffix) |
| `einfold-datafusion` | `EinsumNode`, `EinsumRule`, `EinsumPlanner` | `FoldNode`, `FoldRule`, `EinFoldPlanner` |
| `einfold-datafusion` | `detect::Detected` | a name for "a fold found in a plan, bound to its inputs", such as `FoldMatch` |
| `einfold-testkit` | cases with an `einsum` | cases with a `fold`, over `SUM`, `COUNT` and `AVG` |

## Refactoring the docs

Once this RFC is accepted:

- **design.md:** the summary and title; §2.2, which still teaches with the einsum but introduces the fold right after; §6's vocabulary, rebuilt on the four layers above; §7.2's output forms; §8.3, with group existence, aggregate state and accumulator; §9.1, which classifies folds by level; §10.1, where eager aggregation is a semiring-fold rewrite; §10.2, where EinFold handles any fold; and §14, where the semiring question is answered.
- **supplement.md:** the same sections in detail, plus each aggregate's SQL rules.
- **demos.md:** the online softmax's gap becomes "a level-1 fold, supported by EinFold". Its algebraic optimizations are still a gap.

## Effect on M1

M1 keeps its scope (fast two-operand queries in DataFusion) and takes the general shape. Each item's issue is revised to state the new goals:

- **A2 (#6):** the fold, aggregates (`SUM`, `COUNT`, `AVG`), row values, and group existence separated from aggregate state; the semiring derived.
- **C1 (#14):** the kernel folds any M1 aggregate. Its join and group bookkeeping are unchanged.
- **C2 (#16):** renamed to `EinFoldExec`; otherwise unchanged.
- **B1 (#15):** recognizes `SUM`, `COUNT` (including `COUNT(*)`) and `AVG` of products and of single-operand values, and classifies each at its strongest level.
- **E1 (#19):** generates `COUNT` and `AVG` queries alongside `SUM`.
- **D1 (#20):** the renames; otherwise unchanged.
- **E2 (#12):** adds an `AVG` of products to the benchmark.

## Alternatives considered

- **"Groupjoin" as the central name.** It is precise for a different thing. Moerkotte and Neumann's groupjoin (2011) fuses a join with a `GROUP BY` whose groups come from *one* input (a key–foreign key join). A matrix product's groups span both inputs, which a classical groupjoin cannot express (supplement §10.1). Groupjoin names one physical operator, and EinFold generalizes it; it doesn't name the query class.
- **"Semiring fold" for everything.** That's accurate for level 2, but it would exclude `AVG`, the online softmax, and every level-1 fold that still benefits from fusion.
- **"Aggregate-join query", or FAQ.** "FAQ" is the most precise term from the literature, but obscure to most readers. "Aggregate-join" is clear, but awkward in code. "Fold over a join" says the same thing, and matches the project's name.
- **Keep "einsum" as the center, and add special cases.** That's simpler, but the special cases would multiply: `COUNT`, `AVG`, softmax. It also leaves the design's ontology wrong.
- **"Variable" instead of "dimension".** It's the FAQ literature's term, but "variable" already means an XQL data variable (temperature) in this project.

## Consequences

- **Correctness.** Each new aggregate needs its NULL, empty-group and exactness rules stated and tested. The equivalence harness extends naturally, since SQL itself is the oracle.
- **Performance.** Level-1 folds get the fused operator's main win: no materialized join. Level-2 and level-3 folds get the full optimizer.
- **Composability.** The fold and its partial states are the same abstraction that reduction at the source (design §10.5) and the `Fold` Substrait relation carry, so readers and hosts implement one concept.
- **Docs.** The design keeps teaching through einsums, the most familiar case, while stating the general object up front.

## Resolved questions

1. *Should M1 take the general shape now?* Yes (@alxmrs).
2. *What should the Substrait relation be called?* Proposed: **`Fold`**, since it carries any fold, not just einsums.
3. *Are order-sensitive aggregates in scope?* Proposed: not for now. `STRING_AGG` and `ARRAY_AGG` with `ORDER BY` are folds, but not commutative ones, so partial states can't be merged in any order. Revisit if a demo needs them.

**Open for review:** the names in "Renames", especially `Fold`, `FoldMatch` and the new title, and the five structural findings.
