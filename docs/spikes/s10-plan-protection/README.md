<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S10: plan protection

**Question.** For each host, which mechanism stops the optimizer from undoing einfold's contraction tree, and what does planning cost on a large decomposed einsum? Reproduce Blacher et al.'s satisfiability example on current DuckDB and DataFusion.

**Answer.**

- **DuckDB keeps a contraction order written as CTEs, but plans inlined CTEs very slowly:** 56 s for a 91-step einsum, and more than 3 minutes for 218 steps. Writing the same CTEs `AS MATERIALIZED` plans in 0.1–0.4 s and runs in 0.06–0.2 s. This reproduces what Blacher et al. reported, an optimizer that chokes on a decomposed einsum, and shows that `MATERIALIZED` is the fix on DuckDB.
- **DataFusion keeps the order with plain CTEs, and plans them fast:** 0.13 s for 91 steps, and 0.41 s for 218.
- **On both hosts, the flat single-query form is the problem.** Its run time grows beyond 3 minutes at 218 clauses. One flat DuckDB query spilled 16 GB to disk before the system stopped it. The decomposed forms ran in well under a second.

So the target profile's plan-protection mechanism is `MATERIALIZED` CTEs for DuckDB, and plain CTEs for DataFusion. The relational form must always be decomposed, never flat.

Date: 2026-10-06. DuckDB 1.5.6, DataFusion 54 (Python), on a 6-core, 12-thread Intel i7-8700 with 15 GB of RAM.

## Background

Counting the solutions of a CNF formula is an einsum. Each variable is a dimension of extent 2, and each clause is a tensor over its variables, holding 1 for the assignments that satisfy it. Blacher et al. (SIGMOD 2023, §4.2) used this to stress databases: for a 952-clause problem, DuckDB had not finished planning after five hours.

## Method

[`sat.py`](sat.py) generates a random 3-SAT formula and builds one table per clause, holding the 7 satisfying assignments of its 3 variables (the 8th assignment, all literals false, is the zero left out), with a `DOUBLE` value of 1. Values are doubles because model counts of large formulas exceed 64-bit integers.

The formulas are **banded**: each clause's variables lie within a window of 8. Uniform random 3-SAT has a treewidth that grows with size. At 50 variables, NumPy's reference contraction already needed a 2³³-entry intermediate (64 GiB), so no engine could contract it. Structured formulas, like the one Blacher et al. solved, have small treewidth. A ratio of about 3 clauses per variable keeps the formulas satisfiable.

Three SQL forms of the same einsum:

- **flat:** one query that joins every clause table, with one `SUM` and no `GROUP BY`. The host's join optimizer chooses the order.
- **cte:** Blacher's decomposition. One CTE per pairwise contraction, in opt_einsum's greedy order, each a join with `GROUP BY` and `SUM`.
- **materialized** (DuckDB only): the same CTEs, written `AS MATERIALIZED`.

For each form, the script measures:

- planning time (`EXPLAIN`);
- run time, which includes planning again;
- whether the physical plan still has one aggregation per contraction step, counted from DuckDB's JSON plan and DataFusion's final-mode aggregates;
- whether the result matches NumPy's einsum.

Each measurement runs in its own process, killed after 180 s.

## Results

**30 variables, 91 clauses (90 contraction steps); 2,550 solutions:**

| Host | Form | Plan | Aggregations in plan | Run | Correct |
|---|---|---|---|---|---|
| DuckDB | flat | 0.29 s | 1 of 91 | 1.4 s | yes |
| DuckDB | cte | **55.6 s** | 91 of 91 | 56.5 s | yes |
| DuckDB | materialized | 0.09 s | 91 of 91 | 0.06 s | yes |
| DataFusion | flat | 0.96 s | 1 of 91 | 2.9 s | yes |
| DataFusion | cte | 0.13 s | 91 of 91 | 0.11 s | yes |

**73 variables, 218 clauses (217 steps); 103,531,944 solutions:**

| Host | Form | Plan | Aggregations in plan | Run | Correct |
|---|---|---|---|---|---|
| DuckDB | flat | 1.6 s | 1 of 218 | **> 180 s** | — |
| DuckDB | cte | **> 180 s** | — | > 180 s | — |
| DuckDB | materialized | 0.39 s | 218 of 218 | 0.22 s | yes |
| DataFusion | flat | 16.8 s | 1 of 218 | **> 180 s** | — |
| DataFusion | cte | 0.41 s | 218 of 218 | 0.34 s | yes |

**Incomplete.** The run was also to cover 430 and 718 clauses, and a sweep of DuckDB's inlined-CTE planning time against size. Claude Code stopped it, because the system ran low on memory while the 430-clause flat DuckDB query was spilling (16 GB of temporary files). The script now caps DuckDB's memory and spill size; the larger sizes can be rerun with `python sat.py`.

In an earlier run on a uniform random formula with 20 variables and 91 clauses, DuckDB's inlined CTE form took 138 s to plan and over 300 s to run, and the materialized form 0.1 s.

## Findings

1. **Decompose, always.** The flat query lets the host choose a join order for hundreds of tables. On both hosts its run time grew past 3 minutes by 218 clauses, and DuckDB spilled without bound. The decomposed forms ran in 0.06–0.34 s. This is problem 2 of the design doc (no aggregation pushdown), measured.
2. **Both hosts keep the written order.** Every decomposed plan kept one aggregation per contraction step. `GROUP BY` is a barrier neither optimizer moves joins across. So neither host *reorders* the tree; the danger is planning cost, not undoing.
3. **DuckDB: write `MATERIALIZED` CTEs.** With inlined CTEs, DuckDB's optimizer spends minutes on a plan it doesn't change. Materialized CTEs cut planning by 600× at 91 steps, from 56 s to 0.09 s. This reproduces Blacher et al.'s finding, and gives a fix that needs no change to DuckDB settings.
4. **DataFusion: plain CTEs are enough.** DataFusion planned 218 steps in 0.41 s.
5. **For the target profiles.**
   - DuckDB: plan protection by `MATERIALIZED` CTEs. DataFusion's Unparser does not write them today (spike S7), so einfold's DuckDB writer must.
   - DataFusion: plain CTEs, or a physical plan in the in-engine mode.
   - Both: never emit the flat form for more than a few operands.

## Limitations

- Two formula sizes completed. Blacher et al.'s 952-clause case was not reached.
- Banded random formulas, not SATLIB instances.
- Turning off DuckDB optimizer passes (`SET disabled_optimizers`), Blacher's own workaround, was not measured; `MATERIALIZED` made it unnecessary.
