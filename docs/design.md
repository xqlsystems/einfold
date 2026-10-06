<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# einfold: Fast Tensor Contractions for the XQL Model

Status: draft v9. Author: Alex Merose. Last updated: 2026-10-06. Repository: [xqlsystems/einfold](https://github.com/xqlsystems/einfold). License: Apache-2.0.

## 1. Summary

einfold makes tensor computation fast inside SQL engines, on any hardware those engines run on. Its name joins *einsum*, the standard notation for tensor contractions, with *fold*, functional programming's word for a reducing pass. That is the system's core idea: evaluate an einsum by folding its sums into the joins that feed them.

**The setting.** XQL Systems builds SQL access to large scientific arrays, such as climate and weather data. In the XQL model, a dataset becomes a table with one row per combination of coordinates (for example one row per time, latitude, and longitude), and each variable (such as temperature) becomes a column. Much array math then becomes joins and aggregations. In particular, a **tensor contraction** (matrix multiplication is the simplest case) becomes a join followed by a `SUM` over groups. Contractions are usually written in **einsum** notation, short for Einstein summation, which section 2.2 explains.

**The problem.** SQL engines run contractions badly. They materialize huge intermediate joins, sum too late, and ignore the dense structure of the arrays (section 3).

**The approach.** einfold fixes this by **rewriting query plans, not by executing them.** A query plan is the tree of relational operators (scans, joins, aggregates) that an engine builds from a SQL query. einfold takes a plan in, finds the einsums inside it, and returns a plan that the engine runs faster. The engine, which this doc calls the **host**, may run on CPU or GPU. einfold never touches a device. Section 2.3 lists the hosts.

einfold aims to be a maximally composable data system. Every part (plan readers and writers, fact providers, planners, reference executors) is a separate, swappable piece behind a small interface.

**The design** has three layers:

- a shared vocabulary that maps XQL, einsum, and relational terms onto each other (section 6);
- a few core abstractions that many optimizations reuse: facts, tiles, partial aggregates, and the split between structure and values (section 8);
- a logical optimizer (section 9) and a set of physical realizations (section 10).

**Origin.** einfold grew out of [ddx](https://github.com/xqlsystems/ddx), an XQL Systems project for automatic differentiation of SQL queries. Given a query that computes a function (for example a small neural network written as joins and aggregates), ddx produces the queries that compute its gradients. Training a model this way is mostly contractions, and their slowness motivated einfold. ddx's notes on the problem ([`fast-linalg-notes.md`](https://github.com/xqlsystems/ddx/blob/main/docs/fast-linalg-notes.md)) are the starting point for this design, but everything needed is restated here. ddx benefits from einfold but does not depend on it.

## 2. Background

### 2.1 The XQL ecosystem

The data lives in **Zarr**, a storage format for large N-dimensional arrays. Zarr splits each array into fixed-size **chunks** and stores each chunk as a separate compressed object, on disk or in cloud object storage, along with JSON metadata that records the array's shape, chunk shape, data type, and dimension names. Zarr has two format versions in use, v2 and v3. **Xarray** is a Python library for labeled N-dimensional arrays: arrays with named dimensions and with coordinate values along each dimension. Most Zarr datasets in climate and weather are written and read through Xarray.

[XQL Systems](https://xql.systems) builds SQL on top of Zarr. einfold must work with all of these **readers**, the projects that turn arrays into tables:

| Reader | Engine | How arrays become tables |
|---|---|---|
| [xarray-sql](https://github.com/alxmrs/xarray-sql) (XQL Systems) | DataFusion; also DuckDB, Polars (a DataFrame library), and any database reachable through ADBC or Flight SQL (Apache Arrow's standard database connectors) | Converts each chunk of an Xarray dataset into an Arrow record batch (Arrow is a standard in-memory columnar format) inside a DataFusion table provider |
| [duckdb-zarr](https://github.com/xqlsystems/duckdb-zarr) (XQL Systems) | DuckDB | A native DuckDB extension. Reads Zarr v2 and v3, decodes CF metadata (section 6.2), and reads only the columns a query needs |
| [zarr-datafusion](https://github.com/jayendra13/zarr-datafusion) | DataFusion | A Rust table provider. Infers the table schema from Zarr metadata and dictionary-encodes coordinate columns (stores each distinct value once and refers to it by number) |
| [Zax-SQL](https://docs.earthmover.io/compute/sql) | DataFusion, as a hosted service reached over the Postgres wire protocol or Flight SQL | Built by Earthmover, a company that hosts Zarr data. Reads from Icechunk, Earthmover's versioned, transactional storage layer for Zarr. Flattens each group of arrays like Xarray's `Dataset.to_dataframe()`, repeating lower-dimensional variables across missing dimensions without copying them, and turns filters on dimensions into slices before reading chunks |

All of them flatten a chunked array into rows. einfold needs the structure they flatten away (section 8.1).

### 2.2 Einsums and their relational form

An **einsum** names each axis of each input with a letter, and names the axes of the output. Any letter that does not appear in the output is summed over. For example, matrix multiplication `C[n,h] = Σ_d X[n,d]·W[d,h]` is written `nd,dh->nh`. A **tensor contraction** is any einsum that sums over at least one shared axis.

In the XQL model, `X` and `W` are tables of `(axis…, value)` rows, and the matrix product is:

```sql
SELECT x.n, w.h, SUM(x.v * w.v) AS v
FROM X x JOIN W w ON x.d = w.d
GROUP BY x.n, w.h;
```

Contracted axes become join keys. Output axes become group keys. Blacher et al. (2023), who studied how to evaluate einsums portably in SQL databases, show that four rules turn any einsum into one SQL query of this shape:

1. list the inputs in `FROM`;
2. list the output axes in `SELECT` and `GROUP BY`;
3. take the `SUM` of the product of the values;
4. equate shared axes in `WHERE`, transitively.

### 2.3 Hosts

einfold's output must run on engines we do not control. Hosts take plans in one of two forms. **SQL text** is the usual form. **Substrait** is a cross-engine standard for serialized query plans, so that one engine (or a tool such as einfold) can hand a plan to another without going through SQL.

- **Apache DataFusion** (CPU). A query engine written in Rust, built on Arrow, and designed to be extended. xarray-sql, zarr-datafusion, and Zax-SQL all use it. It lets extensions add optimizer rules and physical operators, and it can write plans back out as SQL (through its `Unparser`) and as Substrait.
- **NVIDIA GPU Query Engine (GQE)**. NVIDIA's GPU query engine, built on libcudf, the CUDA library behind NVIDIA's GPU DataFrames. It accepts Substrait, Flight SQL, and SQL, and uses DataFusion for planning.
- **DuckDB** (CPU). An in-process analytical SQL database, like SQLite for analytics. It reads SQL, and reads Substrait through its `substrait` extension.
- **DuckDB + [`gpudb`](https://github.com/singhpratech/duckdbgpumetaldbram)** (Apple Metal and NVIDIA CUDA). A community DuckDB extension that rewrites SQL statements before DuckDB plans them, and runs parts of them on the GPU. It does not read Substrait. It runs GROUP BY, joins, and expressions inside `SUM` on the GPU, and for some shapes it fuses join and aggregate without producing join output. It keeps `SUM` and `AVG` over floating-point columns on the CPU, because floating-point sums depend on the order of addition (section 8.6).
- **DuckDB + [Sirius](https://github.com/sirius-db/sirius)** (NVIDIA CUDA). A GPU query engine built on libcudf that plugs into DuckDB as an extension. It intercepts every query through a hook in DuckDB's optimizer, converts it to Substrait, and runs it on the GPU. Operators it does not support fall back to DuckDB on the CPU. It supports filters, projections, hash and nested-loop joins, GROUP BY, aggregation, ORDER BY, top-N, LIMIT, and common table expressions (CTEs, the SQL `WITH` clause), over integer, floating-point, decimal, string, date, and timestamp types. Its `pin_table` function keeps a table resident in GPU memory between queries. Support for StarRocks, another analytical database, is announced.

Because of this, einfold must write both **Substrait and SQL**.

## 3. Problem

Building ddx exposed three inefficiencies in how SQL engines run contractions:

1. **Join rows pile up.** In the matrix product above, with `X` of size `N×D` and `W` of size `D×H`, the join emits `N·D·H` rows before `SUM` collapses them. A dense matrix-multiply routine (GEMM, the standard routine in the BLAS linear-algebra libraries) does the same arithmetic without storing any intermediate.
2. **No aggregation pushdown.** Engines do not move a `SUM` below a join. For a product of three or more inputs, the full join is built before anything is summed. The order of contractions that keeps intermediates small is invisible to a join optimizer.
3. **Lost dense structure.** A table does not say that its coordinates map arithmetically to memory offsets, so the engine cannot choose a dense kernel even when the data is dense.

**Baseline.** ddx's benchmark on DataFusion multiplies a 50,000×16 matrix by a 16×8 matrix. The forward product takes 146 ms. The gradient with respect to both inputs takes about 1.8 s, 12× longer, even though it needs only two more contractions: for `Y = X·W`, the gradients are `X̄ = Ȳ·Wᵀ` and `W̄ = Xᵀ·Ȳ`, where a bar marks the gradient of the final result with respect to that matrix. The gap is the hash joins: each contraction materializes all `N·D·H` join rows before summing them.

Three more problems come from the XQL setting:

4. **Chunk misalignment.** Zarr arrays are chunked, and the chunks of two inputs rarely line up on a shared dimension.
5. **Repeated values.** Flattening repeats every lower-dimensional variable across the dimensions it lacks, and every row pays for the repetition (section 9.1).
6. **Planning cost.** Large einsums produce large queries, and host optimizers can spend more time planning them than running them (section 9.3).

## 4. Principles

1. **einfold rewrites, hosts execute.** einfold's product is a better plan. Hardware is the host's job.
2. **The XQL logical model is the contract.** A dataset is a table with one row per coordinate tuple, its dimensions as key columns and its variables as value columns. einfold never changes what a query means or the shape of its result. Layouts, tiles, and partial states are physical, and they live inside operators.
3. **Every rewrite has a portable fallback.** If a host cannot run a richer form, einfold still gives it a plan made of standard relational operators.
4. **Never wrong.** A rewrite is applied only when it is proven equivalent under SQL semantics, including NULLs and bag semantics (SQL tables may hold duplicate rows, and aggregates count every duplicate). Values computed exactly (integers, `DECIMAL`, `COUNT`, `MIN`, `MAX`) stay exact (section 8.6). Missing a speedup is acceptable; changing a result is not.
5. **Never add nondeterminism.** einfold never makes a plan less repeatable than the plan it received. Bit-for-bit determinism is a setting users can request, and the default wherever einfold itself executes (section 8.6).
6. **A chunk is a partition.** Storage tiles are the natural unit of reading, pruning, parallelism, and partial aggregation (section 8.2).
7. **Composable by default.** Readers, writers, fact providers, and planners are plugins behind narrow interfaces. Nothing in the core knows which engine or device is downstream.

## 5. Goals and non-goals

### Goals

- Read relational plans (Substrait, DataFusion's `LogicalPlan`) and write optimized plans (Substrait, SQL for named dialects).
- Remove the aggregation-pushdown and contraction-order problems on every host, using standard operators only.
- Define an `Einsum` Substrait extension relation that carries contraction structure and facts, for hosts that implement it.
- Carry facts (dimensions, extents, layouts, tilings, statistics) from readers to the plan, Zarr first.
- Ship a reference executor for DataFusion that proves the extension is worth adopting.
- Integrate with xarray-sql, duckdb-zarr, zarr-datafusion, Zax-SQL, and ddx.

### Non-goals

- GPU or other device kernels. Hosts provide them.
- Automatic differentiation. That stays in ddx.
- Distributed execution. Plans should not prevent it.
- Changing query semantics to match Xarray (for example Xarray's `skipna`, which ignores NaN in sums). einfold preserves the SQL meaning of the plan it is given (section 6.2).

## 6. Vocabulary

einfold sits where three vocabularies meet: XQL and Xarray, einsum notation, and relational algebra. This section fixes the words the rest of the doc uses.

### 6.1 Terms

| Term | Meaning | Einsum | Relational |
|---|---|---|---|
| **Dimension** | A named axis, such as `lat` or `time` | An index label (`i`, `j`) | A dimension column, which is a key column |
| **Coordinate** | A label along a dimension, such as `30.0°N` or a timestamp | — | A value in a dimension column |
| **Position** | An integer offset `0 … n−1` along a dimension | The value an index ranges over | — |
| **Extent** | The number of positions along a dimension | The size of an index | Exact distinct count of a dimension column |
| **Variable** | A named array over a set of dimensions; Xarray distinguishes data variables (such as temperature) from coordinate variables (such as latitude) | A tensor | A value column, together with its dimensions |
| **Dataset table** | The table a reader produces: one row per coordinate tuple over all of the dataset's dimensions, one column per variable | Several tensors | A wide table |
| **Operand** | One variable over its own dimensions, as input to an einsum | An operand | A narrow table `(dims…, value)` |
| **Support** | The coordinate tuples that have a row | The entries that are not the fill value | The rows that exist |
| **Fill value** | Zarr's value for chunks that were never written | Often called "zero" in sparse-tensor work | — |
| **Tile** | A rectangular box of positions (section 8.2) | A slice or block | A partition |
| **Chunk** | A storage tile in Zarr (in v3, possibly an inner chunk of a shard; section 6.2) | — | Usually one partition of a scan |
| **Layout** | A function from positions to row order within a tile (section 8.1) | A strided array view | Row order of a scan |
| **Fact** | Something known about an operand, tagged with how it is known (section 8.1) | — | Statistics, metadata |
| **Einsum** | A sum of products over operands, with output dimensions | `ik,kj->ij` | `Aggregate(SUM(product))` over equi-joins |
| **Contraction tree** | A binary tree of pairwise contractions that evaluates an einsum | A contraction path | A tree of join-aggregates |
| **EinFold join** (or **EinFold**) | The package's fused join-and-sum operator (section 10.2). The capitalized name is the operator; lowercase **einfold** is the package | One pairwise contraction | A groupjoin (a join fused with the group-by after it; section 10.1), generalized to groups that span both inputs |
| **EinFoldHashJoin** | EinFold's hash algorithm, based on Gustavson's 1978 sparse matrix multiply (section 10.2) | Sparse contraction | — |
| **`Einsum` relation** | The Substrait extension relation that carries an einsum and its facts | — | — |
| **`EinsumExec`** | The DataFusion physical operator in einfold's reference executor | — | — |
| **Partial aggregate** | A per-group state that can be combined later (section 8.3) | A partial sum | A partial-mode aggregate |
| **Reader** | A project that turns arrays into tables (section 2.1) | — | A table provider |
| **Host** | The engine that runs einfold's output (section 2.3) | — | — |
| **Target profile** | Data describing what a host supports (section 7.4) | — | — |
| **E-graph** | A data structure that stores many equivalent versions of an expression compactly, by grouping equal subexpressions into classes (section 7.6) | — | Like the "memo" of a Cascades-style query optimizer |
| **Equality saturation** | Applying rewrite rules to an e-graph until no new equivalent forms appear, or a limit is reached | — | — |
| **Extraction** | Choosing the cheapest expression in an e-graph under a cost model | — | Choosing the cheapest plan |

This doc says **dimension** wherever einsum literature says "index". "Index" appears only in the einsum column above, in titles of cited work, and in einsum strings. That avoids collisions with database indexes and with Xarray's indexes (pandas index objects attached to coordinates).

Notation: `dims(T)` is the set of dimensions of operand `T`. `O` is the set of output dimensions. A dimension in two or more operands is **shared**. A dimension not in `O` is **summed**. A dimension in exactly one operand and not in `O` is **private** to that operand.

### 6.2 Nuances to honor

These are facts about the data model that einfold's rewrites must respect.

- **Coordinates are not positions.** Dimension columns hold coordinates (latitudes, timestamps), not integer offsets. Dense execution needs a map from coordinates to positions: affine `(start, step)` for regular coordinates, a lookup table otherwise. Times decoded from CF metadata (below) are coordinates too.
- **Joins compare coordinates exactly.** Two operands align on a shared dimension only where their coordinates are equal. Float coordinates that differ in the last bit do not join. That matches what the query says, so einfold preserves it. Dense alignment additionally requires that both operands map coordinates to positions the same way. Otherwise EinFold uses its hash algorithm.
- **The fill value decides what a missing chunk means.** Zarr arrays are logically dense: every position has a value, and chunks that were never written read as the array's fill value. The fill value may be 0, NaN, or something else. Many climate datasets follow the CF (Climate and Forecast) metadata conventions, a standard for describing units, time encodings, scaling, and missing data. Readers that decode CF metadata may turn missing data into NULL. The fill value (what unwritten chunks return) is also distinct from a missing-data sentinel, which the `missing_value` Zarr convention declares (section 8.1). They are often equal, but need not be. Which of these a reader emits decides whether skipping a missing chunk is exact (section 8.1).
- **SQL semantics, not Xarray semantics.** SQL `SUM` skips NULL but propagates NaN, while Xarray's `sum` skips NaN by default. Whether missing data reaches the plan as NULL or as NaN is the reader's decision. einfold preserves whatever the plan means.
- **Groups exist only where something joined.** A SQL group appears in the output only if at least one joined row reached it, and its `SUM` is NULL only if every contribution was NULL. Every rewrite must preserve both: which groups exist, and which values are NULL.
- **Bag semantics.** SQL sums every joined row, including duplicate coordinate tuples. That matches einsum over a coordinate list (COO, the sparse format that stores each nonzero as a coordinate tuple plus a value), where duplicate entries add.
- **Memory order is metadata.** A Zarr chunk is not always stored row-major. v2 has an `order` field (C for row-major, F for column-major), and v3 can reorder axes with a transpose codec (codecs are the encoding steps, such as compression, that Zarr applies to each chunk). v3 sharding packs many inner chunks into one stored object, called a shard, which makes storage tiling two-level.
- **Chunk grids may be irregular.** The last chunk along a dimension is partial when the chunk size does not divide the extent. Proposed Zarr extensions also allow chunk sizes that vary along a dimension.

## 7. Architecture

### 7.1 Overview

```
  frontends                        einfold core                                  writers
 ───────────        ─────────────────────────────────────────        ─────────────────────
 Substrait plan ─┐                                                   ┌─► Substrait (GQE, Sirius, DuckDB, DataFusion)
 DataFusion plan ┼─► Rel IR ─► detect ─► EinsumIR ─► logical ─► physical ┼─► SQL text (DuckDB, gpudb, any SQL host)
 SQL (via DF)  ──┘                         ▲          optimizer   realization └─► DataFusion plan (+ EinsumExec)
                                           │              ▲
                                    fact providers   path planners
                                    (Zarr, Arrow,    (greedy, DP,
                                     statistics,      cotengra-style)
                                     user, runtime)
```

- **Rel IR.** einfold's internal representation of relational plans, kept close to Substrait. Frontends convert into it and writers convert out of it.
- **EinsumIR.** einfold's internal representation of an einsum (operands, dimensions, output dimensions, and the pair of operations used for "add" and "multiply", which section 14 calls the semiring), with facts attached to each operand (section 8.1).
- **Logical optimizer** (section 9). Rewrites einsums: detection and normalization, pruning, choosing the order of contractions, tiling, and sharing work across einsums. Its algebraic rewrites run on the egglog engine, and contraction order and tiling run on specialized planners (section 7.6).
- **Physical realization** (section 10). Turns each node of the contraction tree into something a host can run.

### 7.2 Output forms and executors

einfold produces two output forms. The planner picks per subplan, based on the target profile.

**The relational form.** Standard Substrait or SQL that any host can run.

- Each node of the contraction tree becomes a join-aggregate, so dimensions are summed out as early as possible (section 10.1). This fixes problem 2 on every host.
- Each node uses the join-then-`SUM(a*b)`-then-GROUP BY shape that `gpudb` and other engines fuse.
- *Limit.* A two-operand contraction such as matrix multiplication has nothing to sum early. Its speed depends on whether the host fuses join and aggregate.

**The einsum form.** The `Einsum` Substrait extension relation (Substrait lets engines define their own relation types), which carries the contraction and its facts.

- A host that implements it can run EinFold (section 10.2), including dense and block-sparse kernels, and reach GEMM-class speed. This is the only fix for problem 1.
- The extension is published as a spec with a conformance test suite, so GQE, Sirius, DuckDB, and others can adopt it without depending on einfold's code.

**Executors.** Hosts run both forms with their own executors. einfold also ships one reference executor: `EinsumExec` in DataFusion, which implements the einsum form. It proves that the extension is worth adopting, and ddx uses it directly. Device executors belong to the hosts.

### 7.3 Deployment modes

- **In-engine rule.** In DataFusion, einfold runs as an optimizer rule. xarray-sql, zarr-datafusion, and ddx enable it with one call.
- **Plan-to-plan service.** Given a Substrait plan and a target profile, return a Substrait plan. This is how GQE is fed.
- **SQL-to-SQL rewrite.** Given SQL, a dialect, and a target profile, return SQL. This is how DuckDB, `gpudb`, and Sirius are fed, how duckdb-zarr users would use einfold, and how a client can use einfold with Zax-SQL today.
- **Program mode.** Given a batch of named plans, where some plans read the results of others by name, optimize them together and return the batch. A ddx `BackwardProgram` has exactly this shape: a list of steps, each of which may read earlier steps. An optimizer rule sees one plan at a time, so only program mode can share work across steps (section 9.5) and cache plans for a whole program (section 8.5). Program mode works with any of the output forms above.

### 7.4 Target profiles

A **target profile** describes a host: Substrait or SQL dialect, whether it implements the `Einsum` relation (and for which value types and forms), which join and aggregate shapes it fuses, its deterministic mechanisms and supported precision levels (section 8.6), how to stop it from reordering einfold's plan (section 9.3), whether its readers accept aggregate pushdown (section 10.5), and whether it scans a shared CTE once (section 9.5).

A profile also records quirks of the host's plan reader that the writers must respect. For example, DuckDB's Substrait reader honors a relation's `emit` field (which selects and reorders output columns) only on projections. On joins, filters, sorts, fetches, cross joins and set operations it silently ignores `emit` and returns the leading columns (found by ddx, ddx#117). So einfold's Substrait writer puts `emit` only on projection relations for every host, and adds an explicit projection where it needs to reorder columns.

Profiles are data, not code, so new hosts need no einfold release. Fallback is per subplan. If a host runs the `Einsum` relation for dense 64-bit floats but not for sparse integers, einfold emits the einsum form for the first subplan and the relational form for the second, in the same plan.

### 7.5 Components

einfold is written in Rust, organized as several crates (Rust packages), with Python bindings.

| Crate / package | Role | Depends on |
|---|---|---|
| `einfold-ir` | Rel IR, EinsumIR, facts, tiles, partial-aggregate states | nothing engine-specific |
| `einfold-plan` | Logical optimizer (egglog rules in `.egg` files, plus planners) and physical realization | `einfold-ir`, `egglog` |
| `einfold-substrait` | Substrait in and out; `Einsum` relation definition | `einfold-ir`, `substrait` |
| `einfold-sql` | SQL writers per dialect (DuckDB, DataFusion, Postgres) | `einfold-ir` |
| `einfold-datafusion` | `LogicalPlan` frontend, optimizer rule, reference `EinsumExec` | `einfold-plan`, DataFusion |
| `einfold-zarr` | Zarr fact provider | `einfold-ir`, a Zarr metadata reader |
| `einfold-py` | Python bindings | the above |

Only `einfold-datafusion` depends on an engine. The core can be used from DuckDB, GQE, Sirius, or a client library without pulling in DataFusion's executor.

### 7.6 Rewrite engine: egglog, in a hybrid design

The logical optimizer's algebra runs on **egglog** (Zhang et al., 2023), an open-source equality-saturation engine written in Rust. egglog is the successor to **egg** (Willsey et al., 2021), and adds Datalog-style rules for deriving facts. An e-graph stores many equivalent versions of a plan at once. Saturation fills it by applying rewrite rules, and extraction then picks the cheapest version under a cost model.

**Why.** Hand-written rewrite passes must be applied in some order, and an early pass can destroy an opportunity a later pass needed. Hand-written heuristics also have to pick one direction for rules that help in either direction. An e-graph keeps all versions and lets the cost model choose. The strongest precedent is SPORES (Wang et al., 2020). It translates linear algebra into relational algebra, which is close to einfold's own IR, then optimizes with egg, and translates back. It ran 1.2–5× faster than SystemML, a production ML system. Its wins came from algebraic choices like einfold's: distributing a product over a sum when that exploits sparsity, and factoring it back when that is cheaper.

**The hybrid split.**

| Work | Where it runs |
|---|---|
| Facts such as each class's dimensions (section 8.1) | egglog analyses, written as Datalog rules |
| Normalization (9.1), eager aggregation (10.1), pruning (9.2) | egglog rewrite rules, with their side conditions written as queries |
| Cost-based algebraic choices, such as distributivity (9.1) | egglog extraction, with einfold's cost model (section 9.3) plugged in through egglog's `CostModel` interface |
| Common subexpressions (9.5) | Free: the e-graph stores each distinct subexpression once |
| Contraction order (9.3) and tiling (9.4) | Specialized Rust planners, applied to each sum-product region of the extracted plan |
| Physical realization (section 10) | Rust |

**Evidence.** Spike S16 ([`spikes/s16-egglog`](spikes/s16-egglog/README.md)) encoded einfold's rewrites in egglog 3.0:

- egglog found SPORES's rewrite `sum(WH) = Σₖ (Σᵢ W)(Σⱼ H)` (50,000× cheaper by the cost model). It expanded `X·(A + B)` only when `X` was sparse, factored a repeated weight out of a sum, turned a sum over a missing dimension into a scale factor, and fixed a badly ordered ddx gradient (64× cheaper).
- With associativity left to egglog, the e-graph grew about 3.7× per extra matrix in a chain: 1.6 million tuples and 24 s for 11 matrices. At 12 it hit the size limit before finishing, and extraction returned a plan 800,000× worse than the hybrid's. The hybrid matched the full search wherever that search finished.
- Every result was identical across repeated runs and across 1 and 4 threads.

**Design rules from the spike.**

1. **No associativity rules over products.** Contraction order belongs to the planner.
2. **One n-ary node per sum-product region.** Even without associativity, commutativity and reordering of nested sums grew the e-graph about 2.2× per operand. A region should be a single node holding a multiset of operands and a set of summed dimensions. egglog has multisets built in, and its own examples use them in place of associativity and commutativity rules.
3. **Bounded, deterministic schedules.** Use a fixed rule order with iteration and size limits, and no random sampling of rule matches. If a limit is hit, keep the best plan found by the rules that did finish. Never rely on an unfinished associativity search.
4. **Planners and rules share tests.** In the spike, the hand-written planner first dropped a sum over a dimension that no operand has, a case the egglog rule handled correctly. The equivalence tests of section 11 run on both.

**Out of scope, and future work.** Two uses of e-graphs stay in ddx: simplifying the scalar derivative expressions inside projections (`tanh`, `exp`, `CASE`, ddx's NULL handling), which are not sum-products, and choices specific to automatic differentiation, such as saving a region versus recomputing it, or where to checkpoint a deep expression. After milestone M1, einfold's e-graph could accept non-einsum regions as opaque nodes whose costs the caller supplies. ddx could then share einfold's e-graph instead of building a second one.

**Risks.** egglog 3.0 is young (released August 2026), so egg is the fallback. RisingWave found that a SQL optimizer built on egg planned a 6-table join in 39 ms, where DuckDB took 5 ms, and that cost functions were hard to debug. For einfold, plan caching (section 8.5) absorbs repeated planning, but one-off queries pay the cost. An earlier egg-based optimizer for DataFusion's expressions, datafusion-tokomak, has been inactive since 2022, for reasons not yet investigated.

## 8. Core abstractions

Several optimizations turned out to be the same idea in different places. This section defines each shared idea once. Sections 9 and 10 use them.

### 8.1 Facts

A **fact** is something known about an operand, tagged with how it is known. Every planning decision reads facts, and every fact has a source and a precision.

**Kinds of fact.**

| Fact | Example | Used by |
|---|---|---|
| Dimensions of each variable | `w` depends only on `lat` | Variable separation (9.1) |
| Extents | `lat` has 721 positions | Contraction-tree cost (9.3), memory (9.4) |
| Coordinate map | `lat = 90 − 0.25·position` | Dense alignment (10.2) |
| Layout | row order within a chunk | Dense kernels and streaming (10.2, 10.3) |
| Tiling | chunk shape, shard shape, chunk grid | Tiling (9.4), reduction at the source (10.5) |
| Support | which chunks exist; which rows exist | Pruning (9.2), block-sparse algorithm (10.2) |
| Fill value, and how the reader emits it | fill 0, written as no row | Support (below) |
| Size and degree | number of entries; maximum rows per key | Contraction-tree cost (9.3), memory (9.4) |

**Precision.** Each fact is one of:

- **Exact.** Known from metadata or a reader's guarantee: Zarr shapes, chunk grids, a variable's dimensions. DataFusion's table statistics mark such values `Precision::Exact`.
- **Bound.** A guaranteed upper bound, such as a bound on the size of a join computed from degree statistics (below). Deeds et al. (2025) use such bounds in Galley, a query optimizer for sparse tensor programs (Deeds et al., §6.3.2; Chen et al., 2023, who survey bounds of this kind).
- **Estimate.** A best guess from statistics, such as a join size estimated by assuming uniform density. DataFusion's `Precision::Inexact`.
- **Measured.** Observed while running, such as the exact number of entries in an intermediate result that EinFold just produced (section 10.4).

**Rules.**

- Correctness may depend only on Exact facts. Variable separation, dense alignment, and skipping missing chunks all need Exact facts.
- Memory decisions use Exact facts or Bounds: slicing, and choosing which input EinFold holds in memory.
- Cost decisions may use Estimates.
- A Measured fact replaces an Estimate or Bound for the rest of the query.

**Sources (pluggable fact providers).**

- **Zarr metadata.** Shape, chunk grid, shards, dimension names (v3's `dimension_names` field, or v2's `_ARRAY_DIMENSIONS` attribute), memory order, fill value, and which chunks exist.
- **Zarr conventions.** A Zarr convention is a published, named set of metadata attributes that gives arrays extra meaning without changing how they are stored. The `spatial` convention gives affine maps from positions to X/Y coordinates. The `missing_value` convention names the value that marks missing data. XQL Systems' proposed `layout:` convention ([`layout-convention.md`](layout-convention.md)) gives curve orderings, chunk visit order, records of which chunks were written, and per-chunk summaries, each marked exact or not. It is useful to readers and engines without einfold.
- **The reader.** How it flattens: which variables depend on which dimensions, whether it dictionary-encodes a column and how, and how it emits fill values and missing data.
- **Table statistics.** Readers already pass Zarr dimension bounds to the engine as table statistics, which engines use to skip data and estimate costs. In DataFusion these are per-column `min`, `max`, and `distinct_count` in its `Statistics` structure. Statistics alone cannot prove density or functional dependencies, so they yield Estimates unless a reader marks them Exact.
- **Degree statistics.** `D(X | Y)` is the maximum number of rows for any one value of `Y`. For dense arrays these follow from extents. Missing chunks give them at chunk granularity. Sparse tables need the reader to compute them. DataFusion's column statistics have no field for them.
- **The user.** Declared facts.
- **Producers of plans, such as ddx.** ddx proves facts that einfold's dense algorithms need. Every ddx program checks that each input table's dimensions identify its rows, and that each derivative table has one row per key; ddx#120 records these as a `Verified` set that callers keep across runs. ddx also knows which columns are dimensions and which are values, and the keys of every step in its program. ddx hands these over as Exact facts on its steps, through the same carrier as reader facts.
- **The executor.** Measured facts.

**Inside the optimizer,** facts are egglog analyses: Datalog rules derive them, and egglog's merge functions combine them, keeping the most precise value (section 7.6).

**How facts reach the plan is not settled.** Spikes S1–S3 (section 13) decide the carrier: Arrow field metadata (key-value pairs attached to a column's schema), a Substrait extension attached to the read, or a side channel.

**Support and the fill value.** Support, meaning which coordinate tuples have rows, is where sparsity reasoning and cardinality estimation (a database optimizer's estimate of how many rows an operator produces) meet. For a product, the support of the result is the join of the supports of the factors. For a sum over a dimension, it is the projection that drops that dimension (Deeds et al., §6.1). Pruning (9.2), block-sparse execution (10.2), and density measurement (10.4) all reason about support. A missing Zarr chunk may be skipped only as far as its fill value allows:

| Fill value, as emitted by the reader | Can a missing chunk be skipped? |
|---|---|
| No rows (the reader omits fill entries) | Yes. The table really has no rows there. |
| Rows with value 0 | Its products are 0, so the work can be skipped. But the groups it reaches still exist, and still equal 0 where nothing else contributes. Represent the skipped tile by the partial aggregate "matched, value 0" (section 8.3). |
| Rows with value NULL | Its products are NULL, so the work can be skipped. The groups it reaches still exist. Represent it by "matched, no value". |
| Rows with value NaN | Not as if it were zero. Any group it reaches is NaN. The work can be skipped by setting those groups to NaN directly. |

### 8.2 Tiles

A **tile** is a rectangular box of positions. A Zarr chunk, a scan partition, a slice of a contraction, a block in a block-sparse operand, and a GEMM tile (the block of a matrix that a matrix-multiply kernel works on at once) are all tiles. They differ only in their role:

| Role | Example | Chosen by |
|---|---|---|
| Storage tile | Zarr chunk, inner chunk of a shard | The data's author |
| Execution tile | Scan partition, slice of a contraction | einfold (section 9.4) |
| Kernel tile | GEMM tile | The host's kernel |

einfold describes tiles with **CuTe**, the layout library inside NVIDIA's CUTLASS (a C++ library of fast matrix-multiply kernels for GPUs). In CuTe, a layout is a function from coordinates to offsets, written as a shape and a stride, which may be nested to describe hierarchies of tiles. CuTe also defines an algebra for combining and splitting layouts. A **tiling** is CuTe's `logical_divide` of a layout by a tile shape. It splits the layout into two parts: the position inside a tile, and which tile. A Zarr array with shape `N` and chunk shape `c` is `zipped_divide(layout(N), c)`. Shards make this two-level, which CuTe's nested layouts express directly.

Because they are one abstraction, these optimizations share one mechanism:

- **Chunk alignment and slicing for memory** both choose an execution tiling (section 9.4).
- **Moving between tilings** costs IO, counted with a formula from rechunker (section 9.4). That cost enters the choice of contraction tree (section 9.3).
- **Block-sparse execution** skips tiles that are not in the support (section 10.2).
- **Reduction at the source** computes a partial aggregate per storage tile (section 10.5).
- **Pruning** and **partition pruning** (skipping partitions whose statistics rule them out) remove whole tiles when their coordinate range cannot match (section 9.2).

Slicing a summed dimension produces partial aggregates that must be combined (section 8.3). Slicing a kept dimension produces disjoint outputs that are concatenated. Ragged edges (a partial last chunk) break CuTe's requirement that tile sizes divide extents evenly. They are represented as a padded tile plus a bounds check, which is how CuTe handles leftovers. Irregular grids are represented as an explicit list of tiles.

### 8.3 Partial aggregates

A **partial aggregate** is the state of one output group, computed over part of the input and combined later. Eager aggregation (10.1), EinFold's accumulator (10.2), slicing (9.4), parallel partitions, and reduction at the source (10.5) all produce partial aggregates. So their SQL semantics are defined once, here. Their numerics are defined in section 8.6.

**State.** For `SUM` over products, a group's state is:

- `matched`: did any joined row reach this group? This decides whether the group exists.
- `value`: the running sum of the non-NULL products, or "none" if there were none. "None" means the final `SUM` is NULL.

**Update.** For each joined row, set `matched`. Then, if the product is not NULL, add it to `value` (the first non-NULL product assigns it). Setting `matched` before the NULL test is what keeps an all-NULL group in the output as NULL, rather than dropping it.

**Combine.** `matched` is OR-ed. `value` is added, with "none" as the identity. When einfold controls the combine (in its reference executor, and when determinism is requested), it combines partial aggregates in a fixed order: partition order, then tile order.

**Numerics.** How `value` is accumulated (its precision, and whether its result depends on the order of additions) is set by section 8.6.

### 8.4 Order is a layout

The order in which rows arrive is a layout: which dimensions vary slowest and which fastest. Treating order this way connects three things:

- **Streaming.** EinFold can emit one output row at a time when its input arrives grouped by the output's leading dimensions (section 10.2). That is a condition on the input's layout.
- **Choosing output order.** Each node of the contraction tree can emit rows in the order its consumer wants (section 10.3).
- **Reordering.** When the order is wrong and the extents are known, a distribution counting sort (which counts how many rows fall in each bucket, then places each row directly) fixes it in linear time. Gustavson's sparse transpose is exactly that sort.

Within a Zarr chunk, row order comes from the chunk's memory order (section 6.2). CuTe's `coalesce` operation tells whether a set of dimensions forms one contiguous run in memory. That decides whether a dense kernel can read a buffer directly, and whether a matrix-multiply call must treat an input as transposed.

### 8.5 Structure and values

Most of what einfold computes depends only on **structure**: the einsum, the extents, the tilings, and the support. Only the final arithmetic depends on **values**. Separating the two lets structure be computed once and reused:

- **Plan caching.** Contraction trees are cached, keyed by the einsum in a canonical form plus extents rounded into buckets. Blacher et al. note that repeated einsums should not be re-planned. In program mode (section 7.3) the cache key covers the whole program. Keys never include table names: a step that reads another step is identified by its position in the program, not its name. ddx gives every step a fresh name per program (`__ddx_{id}_…`), so name-based keys would miss on every training step.
- **Symbolic–numeric split.** When the support is fixed and only values change, Gustavson computes the output's structure once (the symbolic pass), then runs only a numeric pass with no hashing and no "already touched?" tests (section 10.2). ddx's training steps fit this exactly: each step runs the same contractions on new values.
- **Measured facts.** Densities measured in one run (section 10.4) remain valid for later runs over the same support.

### 8.6 Numerics: determinism and precision

Floating-point addition is not associative: `(a + b) + c` can differ from `a + (b + c)` in the last bits. So a parallel sum, whose additions happen in whatever order threads finish, can give slightly different results on each run. This section sets einfold's policy, based on what SQL engines and JAX do.

#### What the ecosystem does

**SQL engines do not guarantee deterministic float sums in parallel.** Every engine we checked treats run-to-run variation as expected behavior:

- **DuckDB.** Its maintainers call varying `sum(double)` results expected, and suggest `fsum` (Kahan summation), `threads=1`, or a cast to `DECIMAL` (DuckDB discussion #12693).
- **PostgreSQL.** Serial plans give stable float sums, and parallel plans do not. This was reported in 2017 and treated as inherent to floating point.
- **BigQuery and Snowflake.** Their documentation states that `SUM` over floats can differ between runs, and recommends fixed-point types where precision matters.
- **DataFusion.** No documented guarantee. Our measurement (below) shows run-to-run variation.

The engines accumulate `DOUBLE` sums in 64 bits, and offer `DECIMAL` for exact results.

**Our measurement.** 4 million `DOUBLE` values spanning 16 orders of magnitude, with mixed signs; each configuration run 20 times on a 12-core machine; DuckDB 1.5.6 and DataFusion 54.0.0.

| Engine and setting | Distinct results in 20 runs | Relative error vs exact sum |
|---|---|---|
| DuckDB `sum`, 12 threads | 19 | 1.8e-16 |
| DuckDB `fsum` (Kahan), 12 threads | 5 | 3.5e-16 |
| DuckDB `sum`, 1 thread | 1 | 6.9e-14 |
| DuckDB `fsum`, 1 thread | 1 | 0 |
| DataFusion `sum`, 1 input partition, `target_partitions = 1` | 1 | 2.7e-15 |
| DataFusion `sum`, any other partitioning | 4–6 | 1–2e-15 |

Two lessons:

- **Compensated summation, such as Kahan's, is not a determinism fix.** It shrinks the effect of addition order but does not remove it. Only accumulators whose result is truly independent of order (below) are deterministic in parallel.
- **Determinism and accuracy are different properties.** The single-threaded sums are repeatable but the least accurate, because adding strictly left to right accumulates rounding error. A fixed order buys repeatability, not accuracy.

**JAX separates three concerns,** and handles each differently:

- *Randomness* is deterministic by design. Random numbers come from explicit keys passed through pure functions.
- *Order of operations* is fast by default and deterministic by opt-in. On GPUs, reductions use atomic operations, and the compiler's autotuner may choose different kernels between compilations. So results can vary between runs. Users opt into determinism with process-wide XLA flags (`--xla_gpu_exclude_nondeterministic_ops`, formerly `--xla_gpu_deterministic_ops`, plus `--xla_gpu_autotune_level=0`). XLA's documentation warns of substantial throughput loss.
- *Precision* is fast by default, with explicit, scoped controls. Arrays default to 32-bit floats, and 64-bit must be enabled. A float32 matrix multiply at the default precision runs in bfloat16 on TPUs and in TF32 on A100 and H100 GPUs. Users raise precision per operation (a `precision=` argument) or for a block of code (the `jax.default_matmul_precision` context manager).

#### einfold's policy

einfold follows JAX's split.

**1. Guaranteed by design: einfold never adds nondeterminism.**

- A rewritten plan is never less deterministic than the plan einfold received. einfold's own decisions (the contraction tree, the choice of algorithm, run-time switching in section 10.4) depend only on the plan, the facts, and the data, never on timing. The rewrite engine follows the same rule: fixed schedules and no random sampling (section 7.6). Spike S16 found identical results across repeated runs and thread counts.
- Rewrites still change *which* numbers are added together and when, so a rewritten plan's bits can differ from the original plan's. "Equivalent" in principle 4 means mathematically equivalent. Distributing a product over a sum can also change overflow behavior: `a·Σbⱼ` can overflow when every `a·bⱼ` does not.

**Exactness invariant.** einfold never turns an exactly computed value into a rounded one, and never changes an exact value that a comparison, filter, join key, ordering or `LIMIT` depends on. `COUNT`, `MIN`, `MAX`, and integer and `DECIMAL` arithmetic stay exact. Only floating-point sums may change in their last bits, as section 8.6's policy describes. ddx relies on exactly this split: it already assumes that a float `SUM` can differ between a saved result and its recomputation, and that the exact operations do not.

- **Integer and `DECIMAL` overflow.** Eager aggregation, distributivity and reordering can create intermediate values the original plan never computed. For example, `a·Σbⱼ` can overflow when every `a·bⱼ` does not, and reordering additions can overflow a partial sum. In SQL, integer overflow is usually an error, so such a rewrite could turn a correct result into a failed query. einfold applies these rewrites to exact types only when Exact facts or Bounds prove that no intermediate value can exceed its type. Otherwise it leaves that part of the plan unchanged.

**2. Determinism is a scoped setting, off by default on hosts.**

- By default, einfold emits the fastest forms the host supports, with the host's usual float behavior, as every SQL engine above does.
- A user can request determinism per session or per query, as with JAX's flag or DuckDB's `threads=1`. einfold then uses only mechanisms the host's target profile lists as deterministic: a single partition, the host's own deterministic mode, or a cast to `DECIMAL`. If the host offers none, einfold leaves that subplan in its original form and says why.
- `DECIMAL` is the portable deterministic path, since every engine recommends it and it is exact. Its limits are value range and speed.

**3. Deterministic by default where einfold executes.**

- The reference executor (`EinsumExec`) and the `Einsum` relation's specification are deterministic by default, and the conformance suite checks that repeated runs give identical bits. einfold controls these, and conformance testing needs repeatability, much as JAX on TPU is deterministic in practice.
- Ways to accumulate `value` deterministically, in order of preference:
  1. **An order-independent accumulator.** Reproducible summation gives the same bits in any order. It works either by binning values by exponent, as the ReproBLAS library (Demmel and Nguyen) does, or with an exact "superaccumulator" wide enough to hold any sum without rounding. Results are then deterministic, parallel, and accurate, and a rewrite that only reorders sums gives the original plan's exact bits. Cost: slower than a plain sum (spike S8).
  2. **A fixed combine order** (section 8.3). Deterministic on one host but not across hosts, harder to make fast on GPU, and only as accurate as its order.
  3. **Fixed-point values** (SQL `DECIMAL` or scaled integers) where the value range allows. Exact, and `gpudb` already sums these on the GPU. Costly to prove safe.
- Users who want maximum speed in the reference executor, for example when training models, can turn determinism off.

**4. Precision is its own setting.** Modeled on JAX's precision levels:

| Level | Accumulator | Einsum-form kernels |
|---|---|---|
| `fast` | the host's default | may use reduced-precision formats, such as TF32 or bfloat16, where the host offers them |
| `default` | 64-bit for 32-bit and 64-bit floats | full input precision |
| `highest` | an exact or reproducible accumulator | full input precision |

`default` follows SQL's convention of 64-bit sums, and Gustavson's advice to accumulate in higher precision and round once (section 10.2). Precision and determinism are set independently, except that `highest` is also deterministic.

**5. Warn where small differences become big ones.** In SQL, a sum that varies in its last bit can flip a comparison. Sums feed `WHERE` and `HAVING` filters, `ORDER BY … LIMIT`, `MIN` and `MAX` ties, `GROUP BY` on computed keys, and joins on computed values. Then a last-bit difference becomes different rows. ddx, for example, needs a tie-breaking rule for `MIN` and `MAX` because sums across partitions vary. einfold traces each float sum through the plan, and when one feeds such a decision without determinism requested, it warns and suggests deterministic mode. It does not change the plan on its own, since hosts don't either.

**In target profiles,** each host records which deterministic mechanisms it offers and which precision levels it supports.

**On GPU hosts,** the relational form still gets its plan-shape benefits. Fast deterministic float sums on GPU need host support for an order-independent accumulator, which einfold will propose to the `gpudb`, Sirius, and GQE maintainers.

## 9. Logical optimization

The logical optimizer rewrites einsums without choosing how each node will run. Sections 9.1, 9.2 and 9.5 are egglog rules and analyses. Sections 9.3 and 9.4 are specialized planners, applied to each sum-product region of the plan that egglog extracts (section 7.6).

```
detect & normalize ─► prune ─► plan contraction tree ─► choose tiling ─► share across einsums
     (9.1)             (9.2)          (9.3)                (9.4)              (9.5)
```

### 9.1 Detect and normalize

**Matches.** These plan shapes, wherever they occur in a plan, not only at its root:

```
Aggregate(group by G; SUM(e))
  over a tree of inner joins on equal dimensions, filters, and projections over scans

Aggregate(group by G; SUM(v))
  over a UNION ALL of branches, each of which matches the shape above
```

The second shape is a sum of einsums, which ddx produces when a table is read in several places (for example, a weight used by every layer gets one gradient contribution per read). Adding einsum results means what SQL means here: union all the branches, then sum per group. A group that appears in only one branch keeps that branch's value. Each branch is planned on its own, and branches share work through section 9.5.

**Produces.** One or more einsums, plus the rest of the plan around them. That includes outer projections, `HAVING`, `ORDER BY` and `LIMIT`, and also joins above the aggregate: ddx, for example, left-joins each gradient back onto its input table so that rows no gradient reached get 0.

**Algorithm.**

1. **Look through projections.** The product need not appear inside the `SUM`. ddx, for example, computes the product in a projection below the aggregate, then aggregates that one column. Inline projection expressions into `e` and into join and group keys until only scans, filters, and joins remain below the aggregate.
2. **Separate variables.** A dataset table holds many variables, each repeated across the dimensions it lacks. Split each referenced variable into its own operand over its own dimensions, using the Exact fact "this variable depends only on these dimensions" (section 8.1). That fact can come from three places:
   - the reader's dimension metadata;
   - a dimension with stride 0 in the variable's layout, meaning the value does not change along it;
   - an Arrow dictionary-encoded column whose dictionary the reader built from the variable's coordinates. In that case the dictionary's list of distinct values *is* the variable at its true shape, so the operand is read without scanning the repeated column. zarr-datafusion already dictionary-encodes coordinates this way.

   *Example.* In `SUM(w*x) GROUP BY time` over a dataset table `T(time, lat, lon, w, x)`, the weight `w` depends only on `lat`, so it becomes an operand over `{lat}`. Eager aggregation (section 10.1) then computes `Σ_lat w(lat) · (Σ_lon x(time, lat, lon))`, which does `n_lon` times fewer multiplications. A term over repeated values alone collapses entirely, since a value that does not depend on `i` gives `Σᵢ B = nᵢ · B` (Deeds et al., §4.4). A normalizer `Σ_{time,lat,lon} w(lat)` becomes `n_time · n_lon · Σ_lat w(lat)`.

   *Correctness.* The repetition factor is the number of surviving rows per group, not the extent. They are equal for a complete dense array. After filters, use a count: what Yan and Larson (1995), who introduced pre-aggregating before joins, call "eager count".
3. **Group each operand's expressions into factors.** Any expression that reads only one operand's columns becomes a *derived value column* of that operand, and so counts as one factor. For example, a gradient contribution for a layer `y = tanh(Σ x·w)` is `Σₙ x · ȳ · (1 − tanh(z)²)`, where `z` is a column of a saved aggregate. The factor `(1 − tanh(z)²)` reads only that aggregate's columns, so it is one factor of that operand. This is still exactly an einsum, since the factor is constant within the operand's row. A derived column has its operand's dimensions, so variable separation and eager aggregation treat it like any other value column. Without this step, detection would miss most of ddx's gradient contractions.
4. **Normalize the aggregated expression.** Rewrite `e` as a sum of monomials `Σ cⱼ · Π fₖ` (constants times products of factors). A single monomial is one einsum. Several monomials are several einsums whose results are added, since `SUM` is linear. Each operand contributes at most one factor per monomial. An expression that mixes columns of different operands in some way other than a product, such as `SUM(exp(a*b))` with `a` and `b` from different operands, does not match.
   - *Whether to expand.* Expanding a product over a sum can be much better or much worse, depending on sparsity. Consider the matrix-factorization loss `Σ(X − UV)²`. With sparse `X` it is far cheaper expanded, because every term then runs in time linear in `X`'s nonzeros. With dense `X`, the unexpanded form is cheaper. einfold uses Galley's greedy search: try each single application of distributivity, re-plan with section 9.3, keep it if the cost drops, and also try the fully expanded form (Deeds et al., §4.1). Galley's analysis classifies each operator in an expression as distributive, commutative with the aggregate, or blocking. It also covers expressions that mix addition and multiplication inside the sum, such as `Σⱼ A_ik·(B_ij + C_jk)`, a variant of sampled dense-dense matrix multiplication (Deeds et al., §4.4).
5. **Build dimension classes.** Run union-find over the join conditions `x.c = y.d` and `x.c IS NOT DISTINCT FROM y.d`. Each class is one einsum dimension. Equality is transitive, so `u.i = v.i AND v.i = w.i` becomes a single dimension shared by three operands (rule 4 of Blacher et al.). An equality between two columns of the same table is a diagonal (`ii->i`).
   - *The two kinds of equality.* `=` never matches NULL, so it drops rows whose key is NULL. `IS NOT DISTINCT FROM` treats NULL as one more coordinate, which matches how `GROUP BY` already treats NULL. Both define an einsum dimension. Detection records which kind each join uses, and the SQL einfold writes keeps that kind, so NULL keys behave as before. ddx joins dimensions with `IS NOT DISTINCT FROM`. Only the dense algorithms need more: an Exact fact that the dimension is never NULL (which Zarr dimensions guarantee), because a dense layout has no position for NULL.
6. **Classify filters.**
   - A range or equality on a dimension column stays with its operand as a slice. An equality to a constant removes that dimension.
   - A predicate on a value column stays with its operand. It shrinks the operand's support, and the contraction is still an einsum.
   - A predicate that spans two operands and is not an equality (for example `x.i < y.j`) does not match. Detection stops.
7. **Map group keys.** Each group key must be a column in some dimension class. The classes it names form `O`.
8. **Check the semiring.** A semiring is the pair of operations an einsum uses for "add" and "multiply". `SUM` over `*` is the default. `MIN` or `MAX` over `+` (tropical semirings, used for shortest paths) are recognized but left unmatched until section 14 decides on semirings.

**Correctness.** Every rewrite in section 9 holds under bag semantics, so detection does not require unique coordinate tuples. Dense execution does (section 10.2). A join on `=` drops rows whose key is NULL, a join on `IS NOT DISTINCT FROM` matches NULL to NULL, and `GROUP BY` keeps NULL as its own group. The relational form keeps these semantics because it is relational, and dense execution requires non-NULL dimensions, which Zarr guarantees. Anything detection cannot prove is left unchanged.

**In egglog.** Steps 2–6 are rewrite rules and analyses. Dimension classes (step 5) come from the e-graph's built-in union-find. The decision whether to expand a product (step 4) needs no separate search: the e-graph holds both forms, and extraction picks the cheaper one.

**Prior work.** Blacher et al.'s four rules, read in reverse; Galley's logical normalization (Deeds et al., §4); SPORES (Wang et al., 2020).

### 9.2 Prune the support

**Idea.** Before contracting, remove rows that cannot find a join partner anywhere in the einsum.

- For acyclic joins, Yannakakis (1981) showed how to remove all such rows with semi-joins (filters that keep a row only if it has a partner in another table), passed up and then down a join tree. Afterwards no join does wasted work. Einsums are usually acyclic: chains, trees, and stars.
- Predicate transfer (Yang et al., 2024) is a cheaper variant that passes Bloom filters (compact, approximate set-membership tests) along the join graph instead of exact semi-joins.
- Gustavson's analysis of wasted work is the two-operand case. It traces waste to rows of `A` that are empty, entries of `A` whose column matches an empty row of `B`, and the reverse (Gustavson, 1978, §3.3).
- At tile granularity this is partition pruning: a tile whose coordinate range cannot match is never read.

**When.** When facts predict a high miss rate. Dense arrays whose support is complete have no unmatched rows, so pruning is skipped for them. It matters most for sparse workloads such as graphs and triplestores (databases of subject–predicate–object facts).

**Correctness.** A semi-join only removes rows that would not join, and never duplicates rows, so inner-join results and bag semantics are unchanged.

**Realization.** Relational form: semi-joins (`WHERE EXISTS` or `IN`). Reference executor: a Bloom filter on the keys of the input held in memory, applied while scanning the other input.

### 9.3 Plan the contraction tree

**Input.** An einsum with facts. **Output.** A binary contraction tree, with a cost.

Within the tree, each node `A ⊗ B` keeps the dimensions `K = (dims(A) ∪ dims(B)) ∩ (O ∪ dims(rest))`, where `rest` is every operand outside the node's subtree. Everything else in `dims(A) ∪ dims(B)` is summed at that node.

The order of contractions matters as much as join order does. For `ij,jk,k->i`, multiplying the matrices first costs `|i|·|j|·|k|` multiplications, while multiplying `jk,k->j` first costs only `|j|·|k| + |i|·|j|`. Finding the optimal order is NP-hard in general, so practical tools use heuristics. einfold follows opt_einsum (Smith and Gray, 2018), the standard Python library for choosing contraction orders, and cotengra (Gray and Kourtis, 2021), a library for very large tensor networks.

**Algorithm.**

1. **Preprocess** (the first steps of opt_einsum's dynamic-programming planner):
   - Sum out private dimensions first (eager aggregation on single operands).
   - Split the einsum into connected components of the dimension graph. Components with no shared dimension are outer products, which are cross joins in SQL. Contract each component separately and combine them last.
   - Contract Hadamard pairs first, meaning element-wise products of pairs with identical dimension sets.
2. **Search.** Pick an algorithm by operand count, as opt_einsum's `auto` mode does:
   - up to 4 operands: exhaustive search;
   - up to about 14: branch-and-bound, or the dynamic program over subsets of operands of Pfeifer, Haegeman and Verstraete (2014), with a cost cap. The cap starts at the product of the output extents and grows by the smallest extent until a full order is found;
   - more than 14: greedy. At each step, contract the pair with the largest `size(A) + size(B) − size(result)`.
   - cotengra's hyper-optimized search (many randomized greedy runs, hypergraph partitioning, and local rewrites of subtrees) is an optional plugin for very large networks.
3. **Cost.** A combination of floating-point operations (FLOPs) and intermediate size, as in cotengra's `combo` objective, under a memory limit per intermediate.
   - *Dense operands:* FLOPs at a node = the product of the extents of `dims(A) ∪ dims(B)`. Size = the product of the extents of `K`.
   - *Sparse operands:* join rows ≈ `nnz(A)·nnz(B) / Π_{s∈S} n_s`, where `nnz` is the number of entries and the product runs over shared dimensions. This assumes uniform density, so it is an Estimate. Degree statistics give a Bound (section 8.1).
   - *Fusion:* if the profile says the host fuses join and aggregate, join rows cost compute only. Otherwise they cost memory too.
   - *Retiling:* the IO to bring each node's shared dimensions onto a common tiling (section 9.4). This lets the planner prefer trees whose shared dimensions are already chunk-aligned, and node outputs whose tiling matches the next node. It combines rechunker's IO model with contraction-order planning.
   - *Reordering:* the cost of a counting sort when a node's input does not arrive in the order it needs (section 10.3).
4. **Memory.** If no tree fits the memory limit, hand the best tree to tiling (section 9.4) to slice.
5. **Cache** the tree (section 8.5).

**Why not search orders in the e-graph.** Associativity and commutativity rules would let egglog enumerate contraction orders itself, but spike S16 showed that search growing exponentially, and returning poor plans when cut short (section 7.6). The planners here are faster and, within their size limits, exact.

**Why einfold plans inside the engine.** Blacher et al. chose contraction orders outside the database. They note the trade-off: an outside planner knows the contraction structure, while the engine knows sizes and sparsity. They conclude that for sparse tensors, the order is better chosen by the database's optimizer. einfold's in-engine mode gets both: the structure from detection and the facts from the host. In SQL-to-SQL mode, einfold relies on reader facts and supplied statistics instead.

**Protecting the plan from the host optimizer.** A decomposed einsum is a deep tree of small queries, and some host optimizers cost more than they save on it. Blacher et al. measured a query encoding a satisfiability problem with 952 clauses. HyPer, a fast research database from TU Munich, spent 0.87 s planning it and 0.08 s executing it. DuckDB had not finished planning after five hours. With its optimizer disabled, DuckDB planned in 0.20 s and ran in 0.97 s. So the output must stop the host from flattening or reordering the tree again. The target profile picks the mechanism: CTEs that the host must materialize, turning off specific optimizer passes, or, in DataFusion, handing over a physical plan directly. Spike S10 checks which mechanisms each host honors.

**Prior work.** Blacher et al.; opt_einsum; Pfeifer et al.; cotengra; rechunker (IO model).

### 9.4 Choose an execution tiling

**Two problems, one mechanism** (section 8.2):

- *Alignment.* Contracting tile by tile needs both operands tiled the same way along every shared dimension.
- *Memory.* If the best tree has an intermediate larger than the memory limit, the contraction must be split.

cotengra solves the memory problem by slicing: fix some dimensions, run one smaller contraction per combination of their values, and combine the results. A slice is an execution tile, so one step handles both problems by choosing an execution tiling.

To change tilings, einfold borrows from rechunker, a tool from the Pangeo community (which builds open-source tools for big-data geoscience) that changes the chunking of Zarr arrays within a memory budget.

**Algorithm.**

1. **Choose a target tile size for each shared dimension.** For a shared dimension `s` with storage tile sizes `c_A(s)` and `c_B(s)`:
   - if they are equal and start at the same origin, keep them;
   - otherwise choose a target `t(s)`. Candidates are each side's tile size and their least common multiple. The IO to move one side from tile size `c` to `t`, along a dimension of extent `n`, follows rechunker's count of intermediate pieces: `(n div L)·(L/c + L/t − 1)` plus a remainder term, where `L = lcm(c, t)`.
2. **Fit memory.** Grow tiles as rechunker does: walk the dimensions from last to first, enlarging each toward its limit while the tile stays under the memory budget. When the read and write tilings differ, the intermediate tiling is their elementwise minimum. If that is still too small for efficient IO, use rechunker's multi-stage plan, which spaces tile sizes geometrically between source and target.
3. **Slice for memory.** If an intermediate still exceeds the limit, pick dimensions to slice greedily, as cotengra does: prefer the dimension that most reduces the largest intermediate per unit of extra FLOPs. Prefer slice boundaries that coincide with storage tiles, since those cost no IO.
4. **Realize.** A query has no "write" step, so retiling becomes a partition key. Each row gets the key `(⌊position_s / t(s)⌋ for each tiled s)`. The relational form emits a repartition (an exchange of rows between parallel workers) on that key and runs the contraction per partition. If the key includes a summed dimension, the per-partition results are partial aggregates, combined as in section 8.3. If it covers only kept dimensions, the outputs are disjoint and need no combine.

**Why it fits XQL.** When `t(s)` matches the Zarr chunk size, partitions map one-to-one onto chunks. Each partition reads its own chunks, and xarray-sql's partition pruning applies.

**Prior work.** rechunker; cotengra slicing.

### 9.5 Share work across einsums

- **Shared scans.** ddx's two gradient contractions from section 3, `X̄[n,d] = Σ_h Ȳ[n,h]·W[d,h]` and `W̄[d,h] = Σ_n X[n,d]·Ȳ[n,h]`, both read `Ȳ`. Both can stream `Ȳ` against a hash table: one on `W` keyed by `h`, one on `X` keyed by `n`. One scan of `Ȳ` then feeds both results. Each row `(n, h, ȳ)` adds to row `n` of `X̄` through the `W` table, and to column `h` of `W̄` through the `X` table. In ddx these are two separate steps that each read the stored `Ȳ` step, so an optimizer rule seeing one plan at a time cannot share the scan. Program mode (section 7.3) can: it sees every step of the program. In general, group the einsums in one plan, or one program, that share an operand, and stream the shared operand. Holding all of a program's steps in one e-graph also shares their common subexpressions automatically. This is multiple-query optimization (Sellis, 1988), the classic technique of sharing work among queries run together, applied to einsums.
- **Common subexpressions.** Put every node of every contraction tree in a canonical form, hash it, and compute identical nodes once (Deeds et al., §5.4). In egglog this comes for free: the e-graph stores each distinct subexpression once, so einsums placed in the same e-graph share them automatically.
- **Realization.** Reference executor: an `EinsumExec` with several outputs. Relational form: the shared operand is emitted once as a CTE that both einsums reference. Whether the host then scans it once is recorded in the target profile.

## 10. Physical realization

Each node of the contraction tree becomes either a relational join-aggregate (section 10.1) or an EinFold operator (section 10.2), chosen per subplan by the target profile (section 7.4).

### 10.1 The relational form: eager aggregation

**Idea.** Sum out a dimension as soon as no later step needs it. For one pair this is eager aggregation. Over a whole contraction tree it is Blacher et al.'s decomposition of an einsum into nested CTEs.

**The rule.** In relational-algebra notation, `Γ_G F(R)` groups `R` by the columns `G` and computes the aggregate `F`, and `R ⋈_S Q` joins `R` and `Q` on the columns `S`. For a node `A ⊗ B` with kept dimensions `K` and shared dimensions `S = dims(A) ∩ dims(B)`:

```
node(A, B) = Γ_K SUM(a·b) ( A' ⋈_S B' )
where  A' = Γ_{dims(A) \ P_A} SUM(a) (A)     -- P_A: dimensions private to A at this node
       B' = Γ_{dims(B) \ P_B} SUM(b) (B)
```

Written as SQL, each node is one CTE. For `ij,jk,k->i`, with tables `A(i, j, v)`, `B(j, k, v)`, and `v(k, v)`:

```sql
WITH n1 AS (
  SELECT B.j AS j, SUM(B.v * v.v) AS v      -- sums out k
  FROM B JOIN v ON B.k = v.k GROUP BY B.j
)
SELECT A.i, SUM(A.v * n1.v) AS v
FROM A JOIN n1 ON A.j = n1.j GROUP BY A.i;
```

**Why it is correct.** Yan and Larson's theorem on pre-aggregating before a join covers aggregates whose arguments come from one side of the join. `SUM(a·b)` takes `a` from one side and `b` from the other, so their theorem does not apply directly. The rule holds instead because multiplication distributes over addition. Within a pre-aggregated group of `B`, all rows share the same join key, so every row of `A` matches all of them or none of them, and `Σⱼ a·bⱼ = a·Σⱼ bⱼ`. This is the semiring argument behind FAQ ("functional aggregate queries"; Abo Khamis, Ngo and Rudra, 2016), a framework that unifies many sum-product problems. Yan and Larson's "eager count" is the special case `b = 1`.

**NULL semantics.** Check both fields of the partial aggregate (section 8.3).

- *`matched`.* A pre-aggregated group of `B` exists exactly when some row of `B` has that key, so the join matches in both forms, for the same output groups.
- *`value`.*
  - If `a` is NULL, every `a·bⱼ` is NULL, and so is `a·Σbⱼ`. Both forms skip the contribution.
  - If all `bⱼ` are NULL, `Σbⱼ` is NULL, so `a·Σbⱼ` is NULL. Every `a·bⱼ` is also NULL. Both forms skip.
  - Otherwise both forms add the same non-NULL terms.

The rule is therefore exact in SQL semantics, up to the floating-point effects in section 8.6.

**When it helps.** Yan and Larson skip pre-aggregation when the grouping columns form a key, because then nothing gets smaller. The einsum version of that test: a node gains only if it sums out at least one dimension. For matrix multiplication `nd,dh->nh`, the shared dimension `d` is summed only after the join, and nothing is private, so eager aggregation changes nothing. Matrix multiplication needs EinFold. Eager aggregation matters for chains of three or more operands, for ddx's multi-way gradient contractions, and for operands with private dimensions (marginals, traces, weighted means).

**Relation to the groupjoin.** Moerkotte and Neumann (2011) defined the groupjoin, which fuses a join and the group-by after it into one hash table, and gave conditions under which it is correct. Those conditions require each output group to come from exactly one row of the input that is held in the hash table: the grouping columns must determine that row's identity, written `G₁, G₂⁺ → TID(e₁)`, where TID is a row identifier. That holds for joins on a key and a foreign key. It also holds for einsum steps whose output dimensions all belong to one operand, for example `ij,i->i`. It does not hold for matrix multiplication, where an output group `(n, h)` combines rows from both inputs. So matrix multiplication is not a classical groupjoin. It is Gustavson's algorithm (section 10.2).

**Prior work.** Yan and Larson (eager and lazy aggregation); Chaudhuri and Shim (1994), who added group-by to cost-based query optimization; Moerkotte and Neumann (groupjoin); Blacher et al. (CTE decomposition); FAQ.

### 10.2 The einsum form: EinFold

EinFold is the fused join-and-sum operator. It is what the `Einsum` relation asks a host to run, and what `EinsumExec` implements. For two-operand contractions it is the highest-value piece, and the only fix for problem 1.

**Idea.** Fuse the `SUM` into the hash join so the `N·D·H` join rows never exist. For matrices this is Gustavson's row-by-row sparse matrix multiply (Gustavson, 1978), which sparse BLAS libraries still use.

**Dimension roles at a node** `A ⊗ B → K`, with shared dimensions `S`:

- `Kₛ = K ∩ S`: batch dimensions, shared and kept (such as `b` in the batched matrix product `bik,bkj->bij`);
- `F_A = (K ∩ dims(A)) \ S` and `F_B = (K ∩ dims(B)) \ S`: free dimensions from each side;
- `S \ K`: contracted dimensions.

#### Hash algorithm (EinFoldHashJoin)

**Gustavson's algorithm.** It computes `C = AB` one row at a time, as `cᵢ. = Σ_{aᵢⱼ≠0} aᵢⱼ·bⱼ.`, where `cᵢ.` is row `i` of `C`. Row `i` of `C` is a linear combination of the rows of `B` picked out by the nonzeros in row `i` of `A`. Both inputs are stored row by row in compressed sparse row form (CSR: each row's nonzeros stored together, with an array marking where each row starts), so the inner loop only ever multiplies a nonzero by a nonzero. The work is proportional to `N`, the number of nonzero-by-nonzero products, plus small terms for empty rows: `O(p, r, N_A, N)` for a `p×q` by `q×r` product, where `N_A` is the number of nonzeros in `A`. Earlier algorithms matched rows of `A` against columns of `B`, and spent most of their time merging entries that never multiply. Gustavson's state for one row is:

- `x`, a dense array of length `r` holding the running sums;
- `JC`, an unordered list of the columns touched so far in this row;
- `xb`, an integer array of length `r`. Gustavson's "multiple-switch" technique: `xb[k] = i` means column `k` has already been touched in row `i`. Because it stores the row number rather than a yes/no flag, it never needs clearing between rows. It is set to zero once, at the start.

When a row is done, the operator emits `x[k]` for each `k` in `JC`. Within a row, the output is unordered.

**Generalized to einsums:**

```
build:  H ← hash table on B keyed by S; each entry is a list of (F_B, b)
probe:  for each row (s_A, f_A, a) of A:
          for each (f_B, b) in H[s_A]:
            update the partial aggregate for (kₛ, f_A, f_B) with a·b      -- section 8.3
emit:   each group with matched = true
```

The **build** step loads `B` into a hash table. The **probe** step streams `A` and looks each row up in that table. The hash table is `B` compressed along its contracted dimensions, which is what Gustavson's row-wise storage of `B` is. The probe loop is his row loop. The update follows section 8.3: set `matched` (Gustavson's `xb` and `JC`) before testing the product for NULL. Otherwise an all-NULL group would vanish instead of coming out NULL.

**Streaming.** If `A` arrives grouped by `(Kₛ, F_A)` (section 8.4), all contributions to one output row arrive together. Then only one row's state is live: Gustavson's `x`, `JC`, and `xb`, keyed by `F_B`. Use his arrays directly when `F_B` has a known extent `r`, and a small hash map otherwise. When the `(Kₛ, F_A)` key changes, emit the row. With the multiple-switch array there is nothing to reset.

- Memory drops from the size of the output to the size of the hash table plus one output row.
- Rows come out grouped by key, so the next node can stream too.
- A Zarr scan arrives grouped this way within a chunk when `(Kₛ, F_A)` are the slowest-varying dimensions of the chunk's memory order. When the input is not grouped, a counting sort can group it in `O(rows + extent)`.

**Choosing which input to stream.** Gustavson notes an asymmetry. Streaming `A` wastes a step for each entry of `A` whose row in `B` is empty, so its cost depends on `N_A`. Streaming `B` depends on `N_B`. The planner streams the input with fewer expected misses and builds the hash table on the other, as long as that table fits in memory (a Bound, section 8.1).

**Symbolic–numeric split** (section 8.5). When the support is fixed and only values change, run the symbolic pass once to compute the output's structure. Later runs then do only the numeric pass, with no hashing and no "already touched?" test. The reference executor caches the symbolic result next to ddx's cached physical plan.

**Accumulator precision.** Gustavson recommends accumulating `x` in higher precision than the inputs and rounding once, when the row is emitted (Gustavson, 1978, §3.5). einfold's precision levels (section 8.6) build on that advice.

#### Dense and block-sparse algorithms

- **Dense.** When both operands are dense over `S` and their free dimensions, with Exact coordinate maps that agree and unique coordinate tuples, skip hashing. View each tile through its layout and call a GEMM, batched over `Kₛ`. The layout's strides decide whether an input must be treated as transposed (section 8.4).
- **Block-sparse.** When an operand's support is known by tile (for example, from missing Zarr chunks), run Gustavson's algorithm over tiles instead of rows. Hash the present tiles of `B` by their tile coordinates along `S`. For each present tile of `A`, call a dense GEMM against each matching tile of `B`. Skipped tiles still contribute their partial aggregate, as the fill-value rules in section 8.1 require.
- **Choosing.** Dense when both sides are dense with Exact extents. Block-sparse when support is known by tile. Otherwise hash. Thresholds come from spike S11, and can change while running (section 10.4).

#### Correctness

- *NULLs and group existence:* section 8.3.
- *Missing tiles:* section 8.1.
- *Duplicate coordinate tuples.* The hash algorithm adds them up (bag semantics). The dense algorithms require unique tuples, which an Exact layout guarantees.
- *Determinism.* Accumulation order is the order of the streamed input times the order of each hash-table entry's list. It is deterministic if both inputs arrive in a deterministic order and the lists keep insertion order. When determinism is requested, parallel partitions are combined in a fixed order, or with an order-independent accumulator (section 8.6).

**In the relational form.** The relational form cannot express EinFold. But on hosts whose profile says they fuse join and aggregate (such as `gpudb`, for some query shapes), the plain pairwise SQL already avoids materializing join rows.

**Generality.** Gustavson also uses the algorithm for a "pseudo-multiplication": assembling the large sparse matrix of a finite-element simulation from small per-element matrices, where the "product" looks up an entry of an element matrix. Any operation with the same structure works. That supports carrying a semiring in the EinsumIR (section 14).

**Prior work.** Gustavson (1978): row-wise sparse multiply, the multiple-switch technique, the symbolic–numeric split, and sparse transpose as a distribution counting sort. Groupjoin. Sparse tensor compilers such as TACO (the Tensor Algebra Compiler), which generate code for sparse tensor expressions.

### 10.3 Output order

- **Idea.** Each node emits rows in the order its consumer wants (section 8.4), so Gustavson's streaming variant can run through a whole chain with no sorting in between.
- **How.** The contraction planner tracks output orders while it plans. This is how System R, IBM's pioneering relational database, tracked "interesting orders" while choosing join orders (Selinger et al., 1979). Galley makes each intermediate's format match the loop order of the kernel that consumes it (Deeds et al., §5.2). When an order can't be had for free, the counting sort's cost enters the planner's cost model (section 9.3).

### 10.4 Switching between dense and sparse at run time

- **The evidence.** Staudt et al. (2025) show that the density of intermediate results changes during an einsum and is hard to predict from the inputs. They measured this on the Einsum Benchmark (Blacher et al., 2024), a collection of 168 einsum problems from probabilistic models, model counting, language models, and quantum computing. 104 of the 158 instances they ran changed average density by more than 0.2 over their contraction sequence. 28 instances that did not start sparse became very sparse, and 42 became nearly dense. They also give a family of expressions where sparse evaluation is exponentially cheaper than dense.
- **What we have.** EinFold's hash algorithm is correct at any density, and facts give the inputs' densities. So inputs are handled well. Intermediates are not: their density cannot be predicted reliably at plan time.
- **Measuring is free.** EinFold knows exactly how many groups each intermediate has, and facts give the extents of its kept dimensions. So the density of each intermediate is a Measured fact before the next node runs (section 8.1).
- **Policy** (Staudt et al.):
  - track the average density of the tensors still to be contracted;
  - switch from dense to sparse when it falls below a threshold (they found 5% empirically; einfold's comes from spike S11);
  - measure only before expensive contractions, and stop measuring once density exceeds 95%.
- **Beyond the paper.** Staudt et al. only switch from dense to sparse. EinFold has both algorithms, so einfold can switch both ways, and can use the block-sparse algorithm for nodes that mix dense and sparse operands. Their other stated limitation is a fixed contraction order. With Measured facts and Bounds, the planner can re-plan the rest of the tree while it runs.
- **Scope.** Inside an `EinsumExec` (or a host's `Einsum` relation) that runs a whole contraction tree. The relational form is unaffected, since hosts already run it as a sparse form.
- **Determinism.** Switching decisions depend only on the data, so the same data gives the same decisions and the same bits.

### 10.5 Reduction at the source

- **Idea.** When a dimension is summed within a single operand, the reader can compute each storage tile's partial aggregate with a dense kernel on the decoded chunk, before the chunk is flattened into rows. The engine then only combines partial aggregates. This follows directly from principle 6.
- **Mechanism.** Engines already split aggregation into a partial phase per partition and a final merge. In DataFusion, `AggregateExec` runs in `Partial` and `Final` modes. A reader whose target profile says it accepts aggregate pushdown replaces "partial aggregate over scan" with a scan that emits partial aggregates, one per chunk. This is the same kind of pushdown as the projection pushdown (reading only needed columns) and filter pushdown (reading only needed rows) that xarray-sql, zarr-datafusion, and duckdb-zarr already do.
- **Scope.** Contractions local to one operand, including those that variable separation (section 9.1) isolates. Also products of variables in the same Zarr group that share a chunk grid: an element-wise product plus a private sum is local to each chunk.
- **Correctness.** Partial aggregates carry `matched` and `value` (section 8.3) and are combined in chunk order.
- **Why it matters.** For single-array reductions (time means, spatial averages, applying regridding weights), this attacks problem 3 at its source: the data is reduced before it is ever flattened.

### 10.6 Worst-case optimal joins (future)

- For cyclic sparse einsums such as triangle counting (`ij,jk,ki->`), every pairwise contraction order can create intermediates far larger than the result. Worst-case optimal join algorithms avoid this by joining all inputs at once, one variable at a time (Ngo et al., 2018).
- Free Join (Wang et al., 2023) unifies them with ordinary binary hash joins, using one data structure for both hash tables and tries, and matches or beats both kinds on standard benchmarks. Galley's choice of loop order plays the same role for sparse tensors (Deeds et al., §5.1).
- This would be an n-way node in the contraction tree, used only for cyclic sparse subexpressions.

### 10.7 Considered and not pursued

- **Packing dense dimensions into array columns** (for example Arrow's fixed-size list type). It breaks principle 2, the XQL logical model. Dense speed belongs to EinFold's dense algorithm and to facts.
- **Incremental maintenance across Icechunk versions.** Out of scope for now.

## 11. Verification

- **Equivalence on random inputs.** For every rewrite, compare the rewritten plan with the original on random einsums with NULLs, NaNs, duplicate coordinate tuples, ties, empty groups, and all-NULL groups. Do this on every host. ddx's "soak" test generator already produces random queries with NULLs, ties and duplicates, and checks them against JAX. einfold reuses it as its shared equivalence suite across hosts.
- **Fill values.** For each row of the fill-value table in section 8.1, check that skipping missing chunks matches a full scan.
- **Gradients.** With einfold enabled, ddx's gradients still match those of JAX, Google's numerical computing library, computed with `jax.grad` (ddx's `tests/test_v2_jax.py`).
- **Speed.** ddx's `matmul` and `attn` (attention) benchmark families (`crates/ddx-datafusion/tests/ad_perf.rs`), forward and backward, with einfold on and off. Measure the symbolic–numeric split separately: the first training step against later steps.
- **Bits versus math.** Rewrites change summation order, so plain float results may differ in the last bits. Equivalence tests compare with a tolerance. Determinism tests compare the same plan across runs bit for bit. ddx currently tolerates last-bit differences. ddx's tests that use einfold's reference executor, which is deterministic by default (section 8.6), should also check that repeated runs give identical bits.

## 12. Integration

| Project | Mode | Fact source |
|---|---|---|
| xarray-sql | In-engine rule (DataFusion), or SQL-to-SQL for other engines | Xarray dimensions, chunks, and variables |
| zarr-datafusion | In-engine rule | Zarr metadata; dictionary-encoded coordinates |
| duckdb-zarr | SQL-to-SQL; later, a DuckDB extension that calls `einfold-plan` | Zarr metadata |
| Zax-SQL | Today: SQL-to-SQL on the client before sending over the Postgres wire protocol or Flight SQL (relational form only). With Earthmover: in-engine rule in their DataFusion | Icechunk and Zarr metadata, which Zax-SQL already uses for pushdown |
| ddx | Program mode, or the in-engine rule | Facts ddx proves: unique keys (its `Verified` set), which columns are dimensions, and the keys of every step (section 8.1) |
| NVIDIA GQE | Plan-to-plan (Substrait) | from the reader |
| DuckDB + gpudb | SQL-to-SQL | from the reader |
| DuckDB + Sirius | SQL-to-SQL today. Later, the `Einsum` relation inside Sirius's Substrait pipeline | from the reader |

**What stays in ddx:**

- Caching each training step's physical plan. Plans don't change between steps, so they need planning once. Once contractions are fast, the roughly 3 ms of planning per step would otherwise dominate. einfold's program-level cache (section 8.5) complements this by caching its own rewrite decisions across steps.
- Forward-mode differentiation inside long chains of row-by-row operations. Forward mode carries derivatives alongside values instead of working backward, which keeps plans growing linearly, not quadratically, in the chain's length.
- Emitting gradient contributions in a canonical, einsum-shaped form, so detection recognizes them cleanly. Later, ddx could emit the `Einsum` relation directly, as an option in its `ddx_ad::Options` settings.
- Simplifying scalar derivative expressions, and choices specific to automatic differentiation such as saving versus recomputing a region (section 7.6).

## 13. Spikes and literature review

A spike is a short, time-boxed experiment that answers one design question. Spike code and reports live in [`docs/spikes/`](spikes/).

### 13.1 Facts

- **S1: Zarr → table → plan.** For each of xarray-sql, duckdb-zarr, and zarr-datafusion: what Zarr metadata survives into the table schema, and how is it exposed (Arrow metadata, a side table like duckdb-zarr's `read_zarr_metadata()`, or not at all)?
- **S2: Carrier survival.** Does Arrow field metadata survive DataFusion projections, filters, and joins? Can a Substrait read relation carry it through an extension? Does DuckDB keep it?
- **S3: Propagation rules.** Validate the layout rules in section 13.5 against real plans.
- **S4: Coordinate maps.** How often are coordinates affine in real datasets, such as ERA5 (ECMWF's hourly global atmospheric reanalysis) and CMIP6 (the latest round of coordinated global climate-model runs)? How should irregular coordinates be represented?
- **S13: Variable separation.** For each reader, are a variable's dimensions exposed to the plan? When a reader dictionary-encodes a column, does it guarantee that the dictionary comes from the variable's coordinates?
- **S14: Fill values.** For each reader, how are fill values and CF-masked values emitted: no rows, 0, NULL, or NaN? Is that exposed as a fact?

### 13.2 Hosts

- **S5: gpudb shapes.** Which relational-form shapes `gpudb` fuses on GPU, measured on matrix multiplication and attention. How to handle its float-sum rule (section 8.6).
- **S6: GQE Substrait.** Does GQE accept extension relations, reject plans that contain them, or ignore them? Which operators run on GPU?
- **S7: DataFusion unparser.** Is the SQL that DataFusion's `Unparser` writes for DuckDB good enough to round-trip rewritten plans?
- **S8: Cost of deterministic sums.** Measure reproducible summation against a plain sum on CPU and GPU, for matrix-multiply-sized reductions (section 8.6). Also measure a fixed combine order, and the cost of the precision levels.
- **S9: Zax-SQL.** Which rewritten SQL shapes does Zax-SQL's DataFusion run well, and does its Icechunk metadata appear in `information_schema` (the standard SQL catalog of tables and columns)?
- **S10: Plan protection.** For each host, which mechanism stops the optimizer from undoing the contraction tree (materialized CTEs, disabled passes, physical plans), and what does planning cost on a large decomposed einsum? Reproduce Blacher et al.'s satisfiability example on current DuckDB and DataFusion.
- **S11: Algorithm thresholds.** At what density and size does EinFold's dense algorithm beat its hash algorithm on CPU? This also sets the switching threshold in section 10.4.
- **S12: Reader aggregate pushdown.** How can a DataFusion table provider take over the `Partial` phase of an aggregation, and is there an equivalent for duckdb-zarr? What partial-state format do the engines expect?
- **S15: Sirius.** Does DuckDB's optimizer reorder einfold's contraction tree before Sirius's hook sees the plan? Which relational-form shapes stay on GPU, and which fall back to CPU? Is the float `SUM` in Sirius (via libcudf) deterministic (section 8.6)? Can a Substrait extension relation reach Sirius through its hook? Does `pin_table` keep ddx's weights in GPU memory across training steps?

### 13.3 Optimizer engine

- **S16: egglog as the rewrite engine. Done.** Report: [`spikes/s16-egglog`](spikes/s16-egglog/README.md). Outcome: adopt egglog for the algebraic layer, in the hybrid design of section 7.6.
- **S17: n-ary sum-product nodes.** Re-run S16 with each sum-product region as one node over a multiset of operands. Measure e-graph growth on the Einsum Benchmark's largest expressions.
- **S18: Planning time.** Measure egglog's saturation and extraction time on ddx's training plans and on TPC-H, against DataFusion's own planner.

### 13.4 Literature still to read

- Factorized databases (Olteanu and colleagues), which store and compute on joins without materializing them, and FAQ (Abo Khamis, Ngo and Rudra): the general semiring theory behind eager aggregation.
- Fent and Neumann, *A practical approach to groupjoin and nested aggregates* (VLDB 2021): groupjoins beyond key/foreign-key joins.
- Eich, Fender and Moerkotte (2018): generating plans that contain group-by, join, and groupjoin.
- Systems that run tensor or ML computations inside databases: tensor relational algebra (TRA), LMFAO (in-database learning over joins), and SystemDS (Apache's declarative ML system).
- Sparse tensor compilers (TACO).
- Tensat (Yang et al., MLSys 2021): equality saturation for tensor computation graphs.
- Why datafusion-tokomak, an egg-based optimizer for DataFusion, went inactive.
- Reproducible floating-point summation (Demmel and Nguyen, ReproBLAS; exact superaccumulators).
- Substrait's extension mechanisms.

### 13.5 Reference: layout propagation rules

To be validated by spike S3. Each rule follows from CuTe's layout algebra (section 8.2).

| Operator | Effect on layout |
|---|---|
| Projection that drops value columns | Unchanged |
| Projection that drops a dimension column | Only valid if its extent is 1; otherwise it is a reduction, handled by detection |
| Filter on a dimension range `[lo, hi)` (or a strided slice) | Composition with a slice: same strides, smaller shape, new offset |
| Filter `dimension = constant` | That dimension is removed |
| Filter on a value column | The layout becomes a superset of the support; the operand is marked sparse |
| Reordering dimension columns (transpose) | Permute the dimensions; strides move with them |
| Union of arrays along a dimension | Concatenate along that dimension (CuTe's `logical_product` of the tile with the stacking layout) |
| Contraction node | A fresh layout over `K`, in the order EinFold emits (section 10.3) |

## 14. Open questions

- **Semirings.** Support min-plus and max-times semirings for graph workloads such as shortest paths? Eager aggregation's distributivity argument holds for any commutative semiring, so the relational form extends easily. Dense kernels may not.
- **Explicit API.** Offer an `einsum(...)` table function next to automatic detection? Useful for users and tests.
- **Extension governance.** Where does the `Einsum` relation's spec live, and is it proposed upstream to Substrait?
- **Benchmarks.** Dataset sizes, hardware, and pass/fail thresholds for section 15.
- **Zax-SQL partnership.** Zax-SQL is a hosted service, so anything beyond the relational form needs Earthmover to run einfold inside their engine. Earthmover's stated goal for its compute engine ("the system should make those decisions, not the user") matches einfold's.

## 15. Success benchmarks

| Workload | Tests | Hosts |
|---|---|---|
| Matrix multiplication and attention (ddx), forward and backward | Contraction planning, EinFold, einsum form, shared scans | DataFusion, GQE, DuckDB+gpudb, DuckDB+Sirius |
| Geoscience on Zarr (ERA5): EOFs (empirical orthogonal functions, the principal components of a climate field), weighted means, regridding | Variable separation, eager aggregation, tiling, reduction at the source, data larger than memory | xarray-sql, duckdb-zarr, zarr-datafusion |
| Sparse and graph: sparse matrix multiplication, einsums over sparse dimensions | EinFold's hash and block-sparse algorithms, pruning | DataFusion, GQE |
| Large einsums: Blacher et al.'s satisfiability, triplestore, and tensor-network cases | Contraction planning, plan protection, run-time switching | DataFusion, DuckDB |
| TPC-H, the standard decision-support SQL benchmark | Detection precision: no plan changes, no slowdowns on ordinary queries | all |

Each benchmark runs with einfold off and on, on the same host. That is the measure of einfold's worth: speedup on someone else's engine.

## 16. Roadmap

### 16.1 Milestones

1. **M0: Spikes S1–S18** (S16 done). Output: a decision on how facts travel, and a target-profile schema.
2. **M1: EinFold.** Detection (including single-operand factors, `IS NOT DISTINCT FROM` joins, and sums over `UNION ALL`) and EinFold's hash algorithm as a DataFusion rule and `EinsumExec`, for two-operand contractions over sparse tables. Verified as in section 11, and benchmarked on ddx's `matmul` and `attn`.
3. **M2: Relational form and program mode.** The egglog rule set for normalization, eager aggregation, and pruning, plus the greedy contraction planner, and program mode with program-level caching, written as DataFusion plans, Substrait, and DuckDB SQL. Variable separation and distributivity (9.1), shared scans (9.5), and support pruning (9.2). Same benchmarks on DuckDB, DuckDB+gpudb, DuckDB+Sirius, and GQE. Plan protection per S10.
4. **M3: Facts.** `einfold-zarr`, layouts, fill-value rules, Bounds and degree statistics, and EinFold's dense and block-sparse algorithms. Reduction at the source (10.5). Integration with xarray-sql, zarr-datafusion, and duckdb-zarr. ERA5 benchmarks.
5. **M4: Einsum form.** The `Einsum` relation's spec and conformance tests. Run-time switching (10.4) in the reference executor.
6. **M5: Tiling.** Execution tiling and slicing (9.4), dynamic-programming and exhaustive contraction planners, retiling cost in contraction planning, and output order (10.3).
7. **M6: GPU adoption (medium term).** Work with the GQE, Sirius, and/or `gpudb` maintainers to run the einsum form on GPU. Sirius is the most natural first partner, because it already runs Substrait plans on GPU and describes itself as composable.
8. **Later.** Worst-case optimal joins (10.6).

### 16.2 Priority of the further optimizations

Agreed priority, highest first. Each lives in the section that owns it:

1. Variable separation and cost-based distributivity (9.1)
2. Shared scans and common subexpressions (9.5)
3. Support pruning (9.2)
4. Reduction at the source (10.5)
5. Bounds and degree statistics (8.1)
6. Output order (10.3)
7. Retiling cost in contraction planning (9.3)
8. Run-time switching between dense and sparse (10.4)
9. Worst-case optimal joins (10.6)

## 17. Risks

- **Wrong rewrites.** Mitigation: conservative detection; the partial-aggregate and fill-value rules (sections 8.1 and 8.3); the tests in section 11.
- **The einsum form is never adopted.** Then einfold's ceiling on GPU hosts is the relational form, which cannot speed up two-operand contractions. Mitigation: keep the relational form valuable on its own; keep the `Einsum` relation small and well tested; show results with the reference executor.
- **Host optimizers undo or choke on einfold's plans.** Mitigation: plan protection in the target profile (S10).
- **Host behavior drift.** Hosts change what they fuse and accelerate. Mitigation: target profiles are data, plus a benchmark suite per host.
- **Rewrite-engine dependency.** egglog is young, and saturation can blow up. Mitigation: the hybrid design and bounded schedules (section 7.6); egg as a fallback.
- **Facts lost in transit.** If no carrier survives the path from reader to plan, variable separation and EinFold's dense algorithms have nothing to use. Mitigation: spikes S1–S2 and S13–S14 come first.
- **Float sums on GPU hosts.** Deterministic float sums on GPU need host support. Mitigation: determinism is a setting, not a default, on hosts (section 8.6).

## 18. References

Project and systems:

- ddx, and its notes on fast linear algebra: https://github.com/xqlsystems/ddx, https://github.com/xqlsystems/ddx/blob/main/docs/fast-linalg-notes.md
- XQL Systems: https://xql.systems
- xarray-sql: https://github.com/alxmrs/xarray-sql
- duckdb-zarr: https://github.com/xqlsystems/duckdb-zarr
- zarr-datafusion: https://github.com/jayendra13/zarr-datafusion
- Zax-SQL: https://www.earthmover.io/blog/compute-roadmap, https://docs.earthmover.io/compute/sql
- NVIDIA GPU Query Engine: https://build.nvidia.com/nvidia/gpu-query-engine
- gpudb DuckDB extension: https://github.com/duckdb/community-extensions/blob/main/extensions/gpudb/description.yml, https://github.com/singhpratech/duckdbgpumetaldbram
- Sirius: https://github.com/sirius-db/sirius
- Substrait: https://substrait.io
- Zarr conventions (`spatial`, `missing_value`, `dependent-arrays`, and the conventions specification): https://github.com/zarr-conventions
- Proposed XQL Systems `layout:` convention: [`layout-convention.md`](layout-convention.md)

Papers:

- Abo Khamis, M., Ngo, H. Q., Rudra, A. (2016). FAQ: Questions Asked Frequently. *PODS 2016.*
- Blacher, M., Klaus, J., Staudt, C., Laue, S., Leis, V., Giesen, J. (2023). Efficient and Portable Einstein Summation in SQL. *Proc. ACM Manag. Data* 1(2), Article 121. https://doi.org/10.1145/3589266
- Blacher, M., Staudt, C., Klaus, J., Wenig, M., Merk, N., Breuer, A., Engel, M., Laue, S., Giesen, J. (2024). Einsum Benchmark: Enabling the Development of Next-Generation Tensor Execution Engines. *NeurIPS 2024, Datasets and Benchmarks Track.* https://proceedings.neurips.cc/paper_files/paper/2024/file/b1bbfdb9197bfc819a52c34dce493f85-Paper-Datasets_and_Benchmarks_Track.pdf
- Chaudhuri, S., Shim, K. (1994). Including Group-By in Query Optimization. *VLDB 1994*, 354–366.
- Chen, J., Huang, Y., Wang, M., Salihoglu, S., Salem, K. (2023). Accurate Summary-based Cardinality Estimation Through the Lens of Cardinality Estimation Graphs. *SIGMOD Record* 52(1), 94–102. https://doi.org/10.1145/3604437.3604458
- Deeds, K., Ahrens, W., Balazinska, M., Suciu, D. (2025). Galley: Modern Query Optimization for Sparse Tensor Programs. *Proc. ACM Manag. Data* 3(3), Article 164. https://doi.org/10.1145/3725301 (arXiv:2408.14706)
- Gray, J., Kourtis, S. (2021). Hyper-optimized tensor network contraction. *Quantum* 5, 410.
- Gustavson, F. G. (1978). Two Fast Algorithms for Sparse Matrices: Multiplication and Permuted Transposition. *ACM Trans. Math. Softw.* 4(3), 250–269. https://doi.org/10.1145/355791.355796
- Moerkotte, G., Neumann, T. (2011). Accelerating Queries with Group-By and Join by Groupjoin. *PVLDB* 4(11), 843–851. https://www.vldb.org/pvldb/vol4/p843-moerkotte.pdf
- Ngo, H. Q., Porat, E., Ré, C., Rudra, A. (2018). Worst-case Optimal Join Algorithms. *J. ACM* 65(3). (Conference version: PODS 2012.)
- Pfeifer, R. N. C., Haegeman, J., Verstraete, F. (2014). Faster identification of optimal contraction sequences for tensor networks. *Phys. Rev. E* 90, 033315. https://arxiv.org/abs/1304.6112
- Selinger, P. G., Astrahan, M. M., Chamberlin, D. D., Lorie, R. A., Price, T. G. (1979). Access Path Selection in a Relational Database Management System. *SIGMOD 1979*, 23–34.
- Sellis, T. K. (1988). Multiple-Query Optimization. *ACM Trans. Database Syst.* 13(1), 23–52.
- Smith, D. G. A., Gray, J. (2018). opt_einsum: A Python package for optimizing contraction order for einsum-like expressions. *Journal of Open Source Software* 3(26), 753.
- Staudt, C., Blacher, M., Hoffmann, T., Kasche, K., Beyersdorff, O., Giesen, J. (2025). Exploiting Dynamic Sparsity in Einsum. *NeurIPS 2025.* https://openreview.net/forum?id=ixOpURt7wC. Code: https://github.com/ti2-group/dynamic-sparsity-einsum
- Wang, Y. R., Hutchison, S., Leang, J., Howe, B., Suciu, D. (2020). SPORES: Sum-Product Optimization via Relational Equality Saturation for Large Scale Linear Algebra. *PVLDB* 13(12), 1919–1932. http://www.vldb.org/pvldb/vol13/p1919-wang.pdf
- Wang, Y. R., Willsey, M., Suciu, D. (2023). Free Join: Unifying Worst-Case Optimal and Traditional Joins. *Proc. ACM Manag. Data* 1(2), Article 150. https://doi.org/10.1145/3589295
- Willsey, M., Nandi, C., Wang, Y. R., Flatt, O., Tatlock, Z., Panchekha, P. (2021). egg: Fast and Extensible Equality Saturation. *Proc. ACM Program. Lang.* 5(POPL). https://arxiv.org/abs/2004.03082
- Yan, W. P., Larson, P.-Å. (1995). Eager Aggregation and Lazy Aggregation. *VLDB 1995*, 345–357. https://www.vldb.org/conf/1995/P345.PDF
- Yang, Y., Zhao, H., Yu, X., Koutris, P. (2024). Predicate Transfer: Efficient Pre-Filtering on Multi-Join Queries. *CIDR 2024.* https://www.cidrdb.org/cidr2024/papers/p22-yang.pdf
- Yannakakis, M. (1981). Algorithms for Acyclic Database Schemes. *VLDB 1981*, 82–94.
- Zhang, Y., Wang, Y. R., Flatt, O., Cao, D., Zucker, P., Rosenthal, E., Tatlock, Z., Willsey, M. (2023). Better Together: Unifying Datalog and Equality Saturation. *Proc. ACM Program. Lang.* 7(PLDI), Article 125. https://doi.org/10.1145/3591239

Software and documentation:

- egg: https://github.com/egraphs-good/egg; egglog: https://github.com/egraphs-good/egglog; overview of e-graphs: https://egraphs-good.github.io
- RisingWave, "Incubate Your SQL Optimizer Using Egg": https://risingwave.com/blog/incubate-your-sql-optimizer-using-egg/
- datafusion-tokomak, an egg-based optimizer for DataFusion: https://github.com/datafusion-contrib/datafusion-tokomak
- DuckDB discussion #12693, "sum of double not deterministic": https://github.com/duckdb/duckdb/discussions/12693
- DuckDB issue #26143, on how `fsum` combines partial sums: https://github.com/duckdb/duckdb/issues/26143
- PostgreSQL mailing list, "Non-deterministic behavior with floating point in parallel mode" (2017): https://www.postgresql.org/message-id/CAFRJ5K0%2BZZaUz0-ihX-aCj1h42H%3Ds-CLWO%2B2Fb6nHCvXx19Diw%40mail.gmail.com
- XLA GPU determinism: https://openxla.org/xla/determinism
- JAX discussion #10674, on GPU determinism: https://github.com/jax-ml/jax/discussions/10674
- JAX default matmul precision: https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html and JAX issue #10413

- Apache Arrow columnar format, dictionary-encoded layout: https://arrow.apache.org/docs/format/Columnar.html#dictionary-encoded-layout
- DataFusion table statistics (`Statistics`, `ColumnStatistics`, `Precision`): https://docs.rs/datafusion/latest/datafusion/common/struct.Statistics.html
- CuTe layout algebra (NVIDIA CUTLASS): https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md
- CuTe layouts as tensor indexes: https://github.com/NVlabs/CuTe/issues/4
- rechunker algorithm: https://rechunker.readthedocs.io/en/latest/algorithm.html and https://github.com/pangeo-data/rechunker/blob/master/rechunker/algorithm.py
- opt_einsum: https://github.com/dgasmith/opt_einsum
- cotengra: https://github.com/jcmgray/cotengra
- Zarr v3 core specification (chunk grids, fill value, codecs, sharding): https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html
