<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Lessons so far

What M1's first implementation round taught us that [`design.md`](design.md) and [`supplement.md`](supplement.md) don't say, or say only in passing. It was distilled from the maintainer's review, an adversarial review, and testing of the first pull requests (#6, #14–#16, #19, #20), whose branches remain as reference. The adversarial review's probes and benchmarks are on the branches [`review/ir-adversarial`](https://github.com/xqlsystems/einfold/tree/review/ir-adversarial) and [`review/d1-adversarial`](https://github.com/xqlsystems/einfold/tree/review/d1-adversarial). Spike results are in [`spikes/`](spikes/README.md) and are not repeated here.

## Evidence before scope

- **Measure against the host before building.** Spike S11's "2.5–17× faster than a hash join and aggregate" compared Gustavson's algorithm with the spike's own single-threaded loop, not with a SQL engine. M1's kernel, measured against plain DataFusion on ddx's matrix products and attention (ddx is an XQL Systems project for differentiating SQL queries), ran at **0.11–0.6×**: slower everywhere except tiny attention. The first experiment should have been the host's own time on the target workload, next to a hand-written dense loop doing the same arithmetic.
- **Order milestones by value.** M1 built the sparse hash path, whose only advantage is a constant factor over a parallel, vectorized engine. The asymptotic wins (aggregating early and choosing join order, for three or more operands) and the dense path (ddx and ERA5 are dense) came later. Dense, positional kernels and eager aggregation likely deserve to come first.
- **Gate each milestone on a benchmark** against the host, not only on an equivalence suite.
- **Check that detection fires on real plans.** M1's detection accepts only `Float64` `SUM`, `COUNT` and `AVG`, one aggregate, with `GROUP BY`, over two operands. Zarr variables are often `float32` or packed integers, and weighted means are often global `SUM(w*x) / SUM(w)`. All of these decline. Measure the hit rate on plans from xarray-sql (which exposes xarray datasets as SQL tables) before widening anything.

## Performance

From the adversarial review of the kernel (#14) and operator (#16):

- **Keep generic code out of hot loops.** The kernel cloned a key per joined pair, and updated state through the generic `PartialAggregate`, which dispatches on the operation and the value types for every value. It cost about 70 ns per joined pair, against DataFusion's 25. Describe aggregates generically in the IR, and use the generic state as the *oracle* that specialized loops for each aggregate and type are tested against.
- **Use positions, not hashes, for small extents.** Dictionary-encode each dimension to dense positions once; the group is then `i · n_j + j`, and the state is a flat array. Fall back to hashing only when the output's extents are too large.
- **Behave like a DataFusion operator:** reserve memory from the `MemoryPool`, run CPU-bound work off the async runtime, use every core, emit batches of `batch_size`, and report metrics for `EXPLAIN ANALYZE`. For repeatable bits in parallel, partition the output keys: each group's additions keep a fixed order.

## The algebra and the IR

From the maintainer's review of the IR (#6):

- **Describe things by their operations and laws, not by lists of cases.**
  - An operation (`+`, `*`, `min`, `max`, `AND`, `OR`) is a *commutative monoid*: associative, commutative, with an identity. Give it `combine` and `identity`, and let everything else be derived.
  - A *semiring* is any pair of operations where "multiply" distributes over "add". Build it only through a constructor that checks the law, so that holding one is evidence the law holds.
  - Whether a fold is a semiring fold is derived from its operations, never declared.
- **Some laws are conditional.** `*` distributes over `max` and `min` only for non-negative values. Record the condition with the semiring, and prove it from facts before relying on it.
- **Check annihilation too.** In a semiring, "add"'s identity (its zero) should *annihilate*: `0 ⊗ a = 0`. This is what lets a dense kernel pad absent entries with the zero. Over floats it holds for sum-product, min-plus, max-plus and Boolean. It fails for `max` of `*`, whose zero is −∞, since `−∞ · 0` is NaN. Over integers it fails for min-plus and max-plus: the zero is the largest or smallest integer, and wrapping `+` turns `i64::MAX + 1` into the smallest integer. Laws hold for a semiring *and a value type*, and must be checked per type.
- **Floats have edge cases that laws must cover:**
  - `+`'s identity is `-0.0`, not `0.0`: `0.0 + -0.0` is `0.0`. A sum started at `0.0` turns `SUM(-0.0)` into `0.0`, as DataFusion does; one started at the first value keeps `-0.0`. Decide which is the specification.
  - NaN has a sign. IEEE 754's total order puts a NaN with the sign bit set *below* every number. Which NaN arithmetic produces is platform-dependent: x86 gives a negative NaN for `inf + -inf`.
  - Test laws on special values (`±0.0`, `±inf`, NaNs of both signs), not just finite samples.
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
- **`MIN` and `MAX` order floats by IEEE 754's total order,** as Arrow does: a NaN with the sign bit clear is above every number, one with it set is below every number, and `-0.0` is below `0.0`.
- **Integer arithmetic wraps** on overflow.
- **Partial aggregate states have fixed schemas,** given by each aggregate's `state_fields`. For example, `SUM(double)` is `[sum]`, and `AVG` is `[count, sum]`.
- **Optimizer bugs in null equality,** filed locally with the `upstream` label:
  - mixing `=` and `IS NOT DISTINCT FROM` in one join chain makes the `=` join match NULLs (#21);
  - an `IS NOT DISTINCT FROM` join followed by a `CROSS JOIN` drops NULL matches (#22).

  einfold's rule runs after the optimizer, so it inherits these bugs rather than causing them. Tests pin them, so we notice when they are fixed.
- **The Unparser,** which turns plans back into SQL, is safe only on unoptimized plans (spike S7).

## Testing

- **Differential testing against an independent oracle found real bugs.** The oracle is a naive nested-loop evaluation of the fold, with no shared code. Its random generator of folds found #21 and #22.
- **Make the oracle see arithmetic.** M1's harness used values that are exact multiples of 0.25 in small tables, so every sum was exact in any order and no accumulation bug could show. Include signed zeros, NaNs, infinities and ill-conditioned sums. For example, `1e16, 1, -1e16, 1` sums exactly to 2, but DataFusion gave 0 and einfold gave 1.
- **Define equivalence precisely.** "Never wrong" needs a definition that holds for floats: bit-identical results, or a stated error bound, such as one scaled by the sum of the magnitudes. Compare to that definition, not to an ad hoc relative tolerance, and compare schemas (names, types, nullability) as well as values.
- **Generate plans, not just tables.** Detection's risk lives in plan shapes: projections, aliases, filters, join orders, self-joins and repeated dimensions. A generator of random plans, with einfold on and off as the oracle, covers it; hand-written cases don't.
- **Use multi-batch, multi-partition inputs,** where ordering and concatenation bugs live.
- **Keep the harness fast.** A quadratic comparison took 500 s on a benchmark's output, and the time was blamed on einfold.
- **Tests that pin a dependency's bug fail once it's fixed,** which prompts removing the workaround.

## Process

- **Don't build on an unsettled foundation.** Five pull requests were built in parallel on the IR while its design was still changing in review. Each review round meant porting all five. [`AGENTS.md`](../AGENTS.md) now limits building past unreviewed foundations.
- **Concrete code review settles design** in ways a design doc can't. Expect the first foundation to take several rounds, and keep it small.
- **Cargo:** use resolver v3, so dependencies respect the minimum supported Rust version. Give each git worktree its own target directory.

## Open questions

- **Default determinism.** The design makes repeatable bits the default where einfold executes, using a binned sum that costs 3–4.4× a plain one (spike S8). The adversarial review argues for defaulting to fast, with determinism as a switch, as hosts have it. Repeatability is also not accuracy: see the ill-conditioned sum above.
- **Scope.** The design specifies an e-graph optimizer, a Substrait extension relation, layout algebra for tiling and GPU hosts, while the one working host, DataFusion, has yet to see a speedup. Consider deferring all of it until einfold beats DataFusion on ddx's matrix products and attention.
- **Style.** The maintainer asked why the state code doesn't read like conventional functional code: folds and maps over immutable values, rather than in-place updates. Decide which style the codebase uses, and say why.

## Worth keeping

The adversarial review found these done well, and worth carrying into any reimplementation:

- the IR's separation of group existence from aggregate state;
- the checked `Semiring` constructor;
- detection's refusal to move fallible expressions (`movable()`), and its "any doubt declines" rule;
- the kernel's fixed probe, match and group order, and its emission through representative rows, which keeps exact key types;
- the rule's reuse of the `Aggregate` node's own schema, so names, types, nullability and functional dependencies are preserved;
- tests that pin dependency behaviors (bugs #21 and #22, wrapping integer arithmetic) and fail loudly when they change;
- the independent nested-loop oracle;
- self-contained comments in SQL terms.
