<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S12: reader aggregate pushdown

**Questions.** How can a DataFusion table provider take over the `Partial` phase of an aggregation? Is there an equivalent for duckdb-zarr? What partial-state format do the engines expect? (Design doc §10.5, reduction at the source.)

**Answer, in short.**

- **DataFusion** has no table-provider hook for aggregates. The working route is a physical optimizer rule, which zarr-datafusion already uses.
- That rule should replace only the `Partial` aggregate, emitting batches in DataFusion's documented partial-state schema. zarr-datafusion instead replaces the whole aggregate, which gives up DataFusion's parallel final combine, and accumulates in `f64`, which breaks exactness for large integers.
- **DuckDB** offers an extension written against its C API (as duckdb-zarr is, through duckdb-rs) no way to receive an aggregate, or even a filter. The route there is a rewrite of the SQL itself: einfold calls a reader-provided aggregating table function, and combines its partial columns in plain SQL (the relational form of partial aggregates, §8.3).

Date: 2026-10-06. DataFusion 54, zarr-datafusion at commit `190e227`, duckdb-zarr v0.1.3 (duckdb-rs `1.10505.0`), DuckDB 1.5.6.

## Method

Investigation, with small experiments:

1. Read zarr-datafusion's `ZarrAggregateExec` and the optimizer rule that installs it (`src/physical_plan/zarr_aggregate.rs`, `src/optimizer/cardinality/rule.rs`).
2. Printed DataFusion's physical plan, with schemas, for a grouped `SUM`, `AVG`, `COUNT`, `MIN` and `MAX` over float and integer columns (`SET datafusion.explain.show_schema = true`). This shows the partial-state columns.
3. Read duckdb-zarr's table function and the duckdb-rs `VTab` trait it implements.
4. Tried DuckDB's own aggregate-state SQL (`EXPORT_STATE`, `combine`, `finalize`).

## Findings

### 1. DataFusion: replace the `Partial` aggregate, not the whole aggregate

DataFusion plans a grouped aggregate in two phases, with a hash repartition between them:

```
AggregateExec: mode=FinalPartitioned
  RepartitionExec: Hash(group keys)
    AggregateExec: mode=Partial
      DataSourceExec (the reader)
```

`TableProvider` has hooks for projection, filters and limits, but none for aggregates. A reader can take over aggregation only through a `PhysicalOptimizerRule` that recognizes the subtree and swaps in its own `ExecutionPlan`.

zarr-datafusion does exactly this. Its `CardinalityRule` matches a `Single` or `Final` aggregate over its `ZarrExec`, and replaces the **whole** `AggregateExec ← ZarrExec` subtree with `ZarrAggregateExec`. That operator folds the scan's rows into per-group accumulators, after checking that the number of groups fits a budget. It is a working precedent, with three limitations einfold should avoid:

- **It discards DataFusion's final phase,** so the combine across partitions, and the spilling and memory accounting of `AggregateExec`, are lost.
- **It works row by row** on the flattened scan. It does not reduce whole decoded chunks with a dense kernel, which is the actual gain §10.5 is after.
- **It accumulates `SUM` in `f64`,** including for integer columns, as its own TODO notes: integer sums beyond 2⁵³ lose precision. `MIN` and `MAX` also go through `f64`. That breaks einfold's exactness invariant (§8.6, item 2).

The better design replaces only the `AggregateExec(mode=Partial)` and its scan, with a reader operator that emits batches in the **partial-state schema**. DataFusion's repartition and `FinalPartitioned` aggregate stay as they are. Parallelism, spilling and the final combine then remain DataFusion's, and each chunk can be reduced by a dense kernel inside the reader.

### 2. The partial-state schema DataFusion expects

The `Partial` aggregate's output schema is the group keys, followed by each aggregate's state fields, named `<aggregate>[<field>]`. From the plan for `SUM`, `AVG`, `COUNT`, `MIN` and `MAX` over a `DOUBLE` column `v` and a `BIGINT` column `n`:

| Aggregate | State fields | Notes |
|---|---|---|
| `SUM(v)`, double | `sum(v)[sum]: Float64, nullable` | NULL when the group had no non-NULL input: einfold's `matched` flag (§8.3) |
| `SUM(n)`, bigint | `sum(n)[sum]: Int64, nullable` | exact |
| `AVG(v)`, `AVG(n)` | `[count]: UInt64`, `[sum]: Float64` | an integer `AVG` sums in `Float64` |
| `COUNT(v)` | `count(v)[count]: Int64, not null` | |
| `MIN(v)`, `MAX(v)` | `[value]`, in the input type, nullable | |

These names and types come from each aggregate function's `state_fields`, part of DataFusion's `AggregateUDFImpl` interface, so a reader can build the schema from the plan's own aggregate expressions rather than hard-coding it. The two-field `AVG` state shows why a partial aggregate needs more than one value per group, which is what §8.3 generalizes.

### 3. DuckDB: no pushdown hook for C-API extensions; rewrite the SQL instead

duckdb-zarr implements duckdb-rs's `VTab` trait. Its only pushdown hook is `supports_pushdown()`, which turns on **projection** pushdown. The extension C API that duckdb-rs wraps offers no filter pushdown and no aggregate pushdown. Spike S1 saw the same from outside: DuckDB runs a `FILTER` above `READ_ZARR`. DuckDB's C++ optimizer extensions can rewrite plans, but they are not reachable from a Rust C-API extension.

DuckDB does have partial aggregate states in SQL: `sum(v) EXPORT_STATE` returns an `AGGREGATE_STATE<sum(DOUBLE)>`, and `finalize(combine(s1, s2))` merges two. But `combine` is a scalar over two states, not an aggregate over many, and the states' binary layout is internal, so a reader cannot produce them.

So, for DuckDB, the route is the one gpudb already takes: **rewrite the statement before DuckDB plans it.**

1. The reader provides an aggregating table function, for example `read_zarr_reduce(store, dims := [...], group_by := [...], aggs := [...])`. It reduces each chunk with a dense kernel and returns one row per (chunk, group) with plain partial columns: `sum`, `count`, `min`, `max`, and `matched`.
2. einfold's SQL-to-SQL mode replaces `SELECT g, SUM(x) … FROM read_zarr(…) GROUP BY g` with `SELECT g, SUM(sum) … FROM read_zarr_reduce(…) GROUP BY g`. This is the relational form of a partial aggregate (§8.3), which DuckDB plans and parallelizes normally.

This is portable: the same table function serves any engine that can call one, and the combine step is ordinary SQL.

### 4. What the reader needs to know

Either way, the reader must:

- recognize which aggregates it can reduce exactly (`SUM`, `COUNT`, `MIN`, `MAX`, and `AVG` as sum and count), over which column types;
- keep integer and `DECIMAL` sums exact (`i128` accumulators), and emit floats in the engine's state type;
- emit the `matched` flag, or the nullable sum, so that a group whose values are all missing comes out NULL rather than 0;
- agree with the engine on how missing data appears: NULL in xarray-sql and duckdb-zarr, NaN in zarr-datafusion (spike S1). A NaN fill value inside a pushed-down `SUM` gives NaN, exactly as the engine would compute it. A NULL fill value must be skipped.

## Recommendations for the design doc

- §10.5: specify reduction at the source as **replacing the `Partial` phase**, emitting the engine's partial-state schema (DataFusion), or a reader table function with partial columns combined in SQL (DuckDB and others).
- Cite zarr-datafusion's `ZarrAggregateExec` as the precedent, and note the exactness gap, worth raising with its maintainers.
- duckdb-zarr: add `read_zarr_reduce` (or equivalent) as the reduction-at-the-source entry point, since DuckDB's C API offers no aggregate pushdown.

## Limitations

- No prototype of the partial-replacing operator was built. The schema above was read from DataFusion's own plans, not exercised by a custom operator.
- DuckDB's C++ extension API (optimizer extensions) was not examined in depth, because duckdb-zarr uses the C API through Rust.
- xarray-sql was not examined. Its DataFusion table provider would use the same optimizer-rule route as zarr-datafusion.
