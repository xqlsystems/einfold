<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S7: DataFusion's Unparser as einfold's SQL writer

**Question.** In SQL-to-SQL mode, einfold rewrites a DataFusion logical plan and turns it back into SQL for another engine. Is the SQL that DataFusion's `Unparser` writes for DuckDB good enough to round-trip rewritten plans?

**Answer.** Yes, but only for plans that **have not been through DataFusion's optimizer**.

- **Unoptimized plans:** all 34 queries ran in DuckDB with the right results. Two differ in result *type*, because the engines type `AVG` differently.
- **Optimized plans:** 31 of 34 produced SQL that DuckDB rejects. One, TPC-H Q22, produced SQL that DuckDB runs and that returns the **wrong answer**: 25 rows instead of 7.

So einfold should apply its rewrites to the unoptimized plan and unparse that, and leave optimization to the target engine. It must never unparse a plan that DataFusion's own optimizer has rewritten.

Date: 2026-10-06. DataFusion 54 (Python bindings, `datafusion.unparser`, DuckDB dialect); DuckDB 1.5.6.

## Method

[`unparse.py`](unparse.py), for each query:

1. plans it in DataFusion, and takes both its logical plan and its optimized logical plan;
2. unparses each with `Unparser(Dialect.duckdb())`;
3. runs the SQL in DuckDB on the same data, and compares with DataFusion's own result. Rows are compared without regard to order, with numbers rounded to 4 places.

Queries:

- **Twelve shapes einfold's relational form produces** (design doc §10.1), on small einsum operands:
  - a matrix multiplication;
  - eager aggregation into a subquery;
  - a contraction order written as CTEs;
  - a partial aggregate with its `matched` count;
  - a causal-mask predicate;
  - partial aggregates combined with `UNION ALL`;
  - a retiling partition key;
  - a window maximum, and a softmax normalizer;
  - a semi-join reduction;
  - a broadcast scale factor;
  - `ORDER BY … LIMIT`.
- **All 22 TPC-H queries** (from DataFusion's benchmark suite), at scale factor 0.01, as a control.

Run: `python unparse.py <folder with q1.sql … q22.sql>`.

## Results

| | Unoptimized plan | Optimized plan |
|---|---|---|
| Runs in DuckDB, correct result | 32 | 2 |
| Correct values, different result types | 2 (TPC-H Q1, Q8) | 0 |
| DuckDB rejects the SQL | 0 | 31 |
| Runs, **wrong result** | 0 | 1 (TPC-H Q22) |

Every einfold shape round-tripped from the unoptimized plan. From the optimized plan, only the retiling key and the window maximum did.

**Why the optimized plans fail.** DataFusion's optimizer inserts projections and subqueries without aliases, and the Unparser then refers to the original table names outside their scope. For the matrix multiplication, it wrote:

```sql
SELECT "a"."i", "b"."j", sum(("a"."v" * "b"."v")) AS "v"
FROM (SELECT "a"."i", "a"."v", "b"."j", "b"."v" FROM "a" INNER JOIN "b" ON "a"."k" = "b"."k")
GROUP BY "a"."i", "b"."j"
```

DuckDB rejects it: `"a"` is not visible outside the unaliased subquery, which also has two columns named `v`. Similarly, after the optimizer rewrites `COUNT(DISTINCT j)`, the Unparser refers to an internal column `alias1` that it never defines.

**The wrong result.** For TPC-H Q22, the optimizer decorrelates the scalar subquery `c_acctbal > (SELECT avg(c_acctbal) …)` into a join. The unparsed SQL drops both that comparison and the `substring(c_phone, 1, 2) IN (…)` filter, and keeps only the `NOT EXISTS`. DuckDB runs it and returns all 25 country codes instead of 7. Nothing signals an error.

**Types.** Q1's averages come back as `DOUBLE` from DuckDB, but as `DECIMAL(19, 6)` from DataFusion, which also truncates rather than rounds (25.575154 vs. 25.575155). Q8's market share differs the same way. The SQL is right. The engines simply type `AVG` over decimals differently.

## Findings

1. **Rewrite before optimizing, then unparse.** einfold's SQL-to-SQL mode should parse into DataFusion's unoptimized logical plan, apply only einfold's own rewrites, and unparse. The target engine optimizes the result itself. That is also the only form that round-tripped every einfold shape.
2. **The Unparser can produce silently wrong SQL.** The Q22 case is the dangerous one. Anything einfold unparses must be checked. One option is to parse the unparsed SQL back into DataFusion and compare the two plans structurally. Another is to run einfold's shared test suite (§11) through every target dialect.
3. **einfold's own rewrites must stay inside what the Unparser handles.** Every node einfold introduces needs an alias it controls, and must use no DataFusion-internal names. The shapes above are a start for a round-trip test suite. The S10 spike finds that DuckDB keeps a contraction order written as CTEs, but only when they are `MATERIALIZED` (or planning becomes very slow). The DuckDB dialect therefore needs a way to emit `AS MATERIALIZED`, which the Unparser does not do today.
4. **Result types are a host property.** Even correct SQL can change a column's type between hosts (`AVG` over `DECIMAL`). Since principle 2 says einfold never changes a result's shape, the SQL-to-SQL mode should add explicit casts wherever the source and target engines type an expression differently, or document that types follow the target engine.
5. **Worth reporting upstream.** The Q22 predicate loss and the unaliased-subquery references are bugs in DataFusion's Unparser. They were not filed.

## Limitations

- Only the DuckDB dialect, through Python bindings.
- einfold's shapes were written as SQL and planned by DataFusion. They were not produced by einfold's rewriter, which doesn't exist yet. A rewriter that builds plan nodes directly may produce plans the Unparser handles differently.
- Scale factor 0.01. Results were compared, not the SQL text, so a wrong query whose error does not show on this small data would go unnoticed.
