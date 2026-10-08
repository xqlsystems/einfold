<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Lessons so far

What M1's first implementation round taught us that [`design.md`](design.md) and [`supplement.md`](supplement.md) don't say, or say only in passing. It was distilled from code review and testing of the first pull requests (#6, #14–#16, #19, #20), whose branches remain as reference. Spike results are in [`spikes/`](spikes/README.md) and are not repeated here.

## The algebra and the IR

From the maintainer's review of the IR (#6):

- **Describe things by their operations and laws, not by lists of cases.**
  - An operation (`+`, `*`, `min`, `max`, `AND`, `OR`) is a *commutative monoid*: associative, commutative, with an identity. Give it `combine` and `identity`, and let everything else be derived.
  - A *semiring* is any pair of operations where "multiply" distributes over "add". Build it only through a constructor that checks the law, so that holding one is evidence the law holds.
  - Whether a fold is a semiring fold is derived from its operations, never declared.
- **Some laws are conditional.** `*` distributes over `max` and `min` only for non-negative values. Record the condition with the semiring, and prove it from facts before relying on it.
- **Check annihilation too.** In a semiring, "add"'s identity (its zero) should *annihilate*: `0 ⊗ a = 0`. This is what lets a dense kernel pad absent entries with the zero. It holds for sum-product, min-plus, max-plus and Boolean. It fails for `max` of `*`, whose zero is −∞, since `−∞ · 0` is NaN.
- **Classify aggregates as Gray et al. do.** Gray et al.'s data-cube paper ("Data Cube: A Relational Aggregation Operator", 1997) sorts aggregates by the state a partial result needs:
  - *distributive:* lift each value, then fold with one operation. `COUNT` lifts each value to 1, then sums.
  - *algebraic:* a fixed tuple of distributive parts, then a final function. `AVG` is `SUM / COUNT`. Its state is the direct product of its parts' monoids, which is again a monoid.
  - *holistic* (`MEDIAN`), *approximate* (sketches) and *order-dependent* (`STRING_AGG`): not handled yet.

  [Apache DataFusion](https://datafusion.apache.org/)'s whole aggregate catalog sorts into these classes.
- **The shape is standard.** Lift, fold with a monoid, then finish is the aggregate interface of Algebird's `Aggregator`, Apache Beam's `CombineFn` and DataFusion's `Accumulator`. The semiring view of aggregates over joins is FAQ's (Abo Khamis, Ngo and Rudra, "FAQ: Questions Asked Frequently", 2016). A brief literature check of an abstraction is cheap and builds confidence.
- **SQL departs from the monoid on empty input.** A monoid folds no values to its identity, but SQL gives NULL for every aggregate except `COUNT`, which gives 0.
- **Use typed values.** Folding everything as `f64` turns `COUNT` and integer sums into floats, and forces logical aggregates to read numbers as truth values. Operations act on typed values (integer, float, Boolean).
- **Keep three things apart:**
  - whether a group exists (some joined row reached it, whatever its value);
  - the aggregate's state;
  - the *accumulator*, meaning how additions are computed numerically, such as plain `f64` or a binned reproducible sum.

## SQL semantics that are easy to get wrong

- **A group reached only by NULL values exists.** Its `SUM` is NULL and its `COUNT` is 0. It doesn't vanish.
- **Without `GROUP BY`,** SQL returns one row even when nothing joined. A fold over a join has rows only where something joined, so detection declines this shape for now.
- **NULL keys:** `=` never matches NULL, and `IS NOT DISTINCT FROM` matches NULL to NULL. Preserve whichever the plan used, per dimension.
- **Duplicates count.** Every joined row contributes, even when its key tuple repeats.
- **Don't introduce errors.** Moving an expression below a join evaluates it on rows the plan never evaluated it on. Move only expressions that cannot fail.
  - `/` and `%` can fail.
  - `power(0, -1)` errors in DataFusion.
  - Integer arithmetic wraps, and float arithmetic gives ±inf or NaN, so both are safe to move.
- **Exact types stay exact.** `SUM` over integers or `DECIMAL` is exact in SQL. Detection rewrites only float sums, where reordering changes at most the last bits.
- **A product is NULL exactly when one of its factors is NULL.** NaN and infinity are not NULL. So `COUNT` of a product is a sum of products of 0/1 indicators.

## DataFusion 54.1.0 behaviors

- **Float keys compare bitwise.** `-0.0` and `0.0` are different keys, and `NaN` equals `NaN`.
- **`MIN` and `MAX` order floats by IEEE 754's total order,** as Arrow does: NaN is above every number, and `-0.0` is below `0.0`.
- **Integer arithmetic wraps** on overflow.
- **Partial aggregate states have fixed schemas,** given by each aggregate's `state_fields`. For example, `SUM(double)` is `[sum]`, and `AVG` is `[count, sum]`.
- **Optimizer bugs in null equality,** filed locally with the `upstream` label:
  - mixing `=` and `IS NOT DISTINCT FROM` in one join chain makes the `=` join match NULLs (#21);
  - an `IS NOT DISTINCT FROM` join followed by a `CROSS JOIN` drops NULL matches (#22).

  einfold's rule runs after the optimizer, so it inherits these bugs rather than causing them. Tests pin them, so we notice when they are fixed.
- **The Unparser,** which turns plans back into SQL, is safe only on unoptimized plans (spike S7).

## Testing

- **Differential testing against an independent oracle found real bugs.** The oracle is a naive nested-loop evaluation of the fold, with no shared code. Its random generator of folds found #21 and #22.
- **Compare floats bit for bit,** including NaN, and compare row order where the operator promises it.
- **Tests that pin a dependency's bug fail once it's fixed,** which prompts removing the workaround.

## Process

- **Don't build on an unsettled foundation.** Five pull requests were built in parallel on the IR while its design was still changing in review. Each review round meant porting all five. [`AGENTS.md`](../AGENTS.md) now limits building past unreviewed foundations.
- **Concrete code review settles design** in ways a design doc can't. Expect the first foundation to take several rounds, and keep it small.
- **Cargo:** use resolver v3, so dependencies respect the minimum supported Rust version. Give each git worktree its own target directory.

## Open points from review

These are unresolved, and belong in the next implementation:

- **Performance of partial aggregates.** In #6, an aggregate's state loops over its parts on every value. The maintainer called this "crazy non performant". The IR may describe aggregates generically, but hot loops need specialized code per aggregate.
- **Style.** The maintainer asked why the state code doesn't read like conventional functional code: folds and maps over immutable values rather than in-place updates. Decide which style the codebase uses, and say why.
- **An adversarial review of #6** is in progress. Add its findings here.
