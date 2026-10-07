<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# RFC 0001: Folds over joins, not just einsums

- **Status:** Proposed
- **Author:** 🧭 Claude (orchestrator), from @alxmrs's review of #6 and #7
- **Design sections affected:** §1, §2.2, §6, §7.1, §7.6, §8.3, §9.1, §10.1, §10.2, §14; supplement §9.1 and §10.2

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

1. **The central object is the fold over a join.** In the design, the vocabulary and the code, einfold optimizes *folds over joins*. An einsum is the sum-product semiring fold, the most important case and the one most of the docs explain. The project and operator names stay: *einfold* (einsum + fold) and *EinFold*.

2. **The intermediate representation names the levels explicitly.** Sketch:

   ```rust
   /// A fold over a join: GROUP BY `output`, folding `aggregate` over
   /// `value(row)` for every row of the join of `operands`.
   pub struct Fold {
       operands: Vec<Operand>,              // unchanged: dims + equality per dim
       output: Vec<Dim>,
       equality: BTreeMap<Dim, KeyEquality>,
       value: Value,
       aggregate: Aggregate,
   }

   #[non_exhaustive]
   pub enum Value {
       /// A ⊗-product of one factor per operand: a semiring fold when
       /// `aggregate` is the semiring's ⊕. Unlocks the algebraic rewrites.
       Product(Semiring),
       /// Any other expression of the joined row: fusion only.
       Opaque,
   }

   #[non_exhaustive]
   pub enum Aggregate { Sum, Count, Min, Max, Avg, BoolOr /* … */ }
   ```

   `Einsum` stays as a constructor, or an alias, for the sum-product case, so existing tests and docs read naturally. `Fold::semiring()` returns `Some` only when the value and aggregate together form a semiring fold, and the optimizer checks it before any algebraic rewrite.

3. **Partial aggregates are generic.** "Partial aggregate" remains the right name: it is the standard database term for an aggregate's state over part of the input (DataFusion's `Partial` and `Final` modes use it the same way), and it applies to every fold. In code, the state splits in two:
   - **group existence** (`matched`), which is the same for every aggregate, because SQL creates a group exactly when some joined row reaches it;
   - **the aggregate's state** (a sum; a sum and a count; a minimum; …), behind one trait with `update`, `merge` and `finish`.

   `PartialSum` becomes one implementation of that trait.

4. **Detection recognizes folds, not just sums.** It accepts `SUM`, `COUNT`, `MIN`, `MAX` and `AVG`, plus the decompositions of level 3, and classifies each query at the strongest level it provably meets. A query that is only a level-1 fold gets the fused operator and nothing else; that is still a large win.

5. **Correctness rules travel with the aggregate.** Each aggregate carries its SQL semantics: how NULL inputs behave, what an empty group returns, and whether results are exact (`COUNT`, `MIN` and `MAX` are exact on any type; `SUM` and `AVG` of floats are not). The exactness invariant (design §8.6) applies per aggregate. One distinctive case is `MAX` of products, which is a semiring only when every factor is non-negative; that is a fact detection must prove before using it.

## Effect on M1

M1 is in review. I propose to keep M1's *scope* (fast two-operand queries in DataFusion) while giving it the general *shape*:

- **A2 (#6):** rename `Einsum` to `Fold`, add `Value` and `Aggregate` (initially `Sum`, `Count` and `Avg`), and split `PartialSum` into existence plus an aggregate-state trait.
- **C1 and C2 (#14, #16):** make the kernel generic over the aggregate's state. The hash join and group bookkeeping don't change.
- **B1 (#15):** accept `SUM`, `COUNT` and `AVG` of products, and the value `1` for `COUNT(*)`. `MIN` and `MAX` follow in M2 with the semiring checks.
- **E1 (#19):** the generator also produces `COUNT` and `AVG` queries.
- **D1 (#20):** unchanged in structure.

That adds perhaps 150–250 lines across the stack. The alternative is to land M1 as is, and generalize in M2. That's cheaper now, but it means renaming merged public types and rewriting reviewed code later.

## Alternatives considered

- **"Groupjoin" as the central name.** It is precise for a different thing. Moerkotte and Neumann's groupjoin (2011) fuses a join with a `GROUP BY` whose groups come from *one* input (a key–foreign key join). A matrix product's groups span both inputs, which a classical groupjoin cannot express (supplement §10.1). Groupjoin names one physical operator, and EinFold generalizes it; it doesn't name the query class.
- **"Semiring fold" for everything.** That's accurate for level 2, but it would exclude `AVG`, the online softmax, and every level-1 fold that still benefits from fusion.
- **"Aggregate-join query", or FAQ.** "FAQ" is the most precise term from the literature, but obscure to most readers. "Aggregate-join" is clear, but awkward in code. "Fold over a join" says the same thing, and matches the project's name.
- **Keep "einsum" as the center, and add special cases.** That's simpler, but the special cases would multiply: `COUNT`, `AVG`, softmax. It also leaves the design's ontology wrong.

## Consequences

- **Correctness.** Each new aggregate needs its NULL, empty-group and exactness rules stated and tested. The equivalence harness extends naturally, since SQL itself is the oracle.
- **Performance.** Level-1 folds get the fused operator's main win: no materialized join. Level-2 and level-3 folds get the full optimizer.
- **Composability.** The fold and its partial states are the same abstraction that reduction at the source (design §10.5) and the `Einsum` Substrait relation (to be renamed `Fold`?) carry, so readers and hosts implement one concept.
- **Docs.** The design keeps teaching through einsums, the most familiar case, while stating the general object up front.

**Open questions**

1. Should M1 take the general shape now, as proposed, or land as is and generalize in M2?
2. What should the Substrait extension relation be called: `Einsum` or `Fold`?
3. Should order-sensitive aggregates (`STRING_AGG`, `ARRAY_AGG` with `ORDER BY`) ever be in scope? They are folds, but not commutative ones.
