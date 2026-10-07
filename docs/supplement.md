<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# einfold: design supplement

Companion to the [design doc](design.md). The design doc gives the argument; this supplement gives the details behind it: algorithms, rule tables, correctness arguments, and the evidence from spikes. Sections are numbered to match the design doc: supplement section 9.1 expands design section 9.1, so a reference such as "section 8.6" points to the same topic in both.

## 2. Background, in detail

### 2.1 What the readers pass on

Spike S1 ([`spikes/s01-readers`](spikes/s01-readers/README.md)) read one small dataset through xarray-sql, duckdb-zarr and zarr-datafusion, and found they disagree on almost everything einfold needs to know:

- how missing data appears: NULL in xarray-sql and duckdb-zarr, NaN in zarr-datafusion;
- how a lower-dimensional variable is handled. zarr-datafusion pairs a 1-D data variable with its dimension wrongly, a silent wrong result;
- which statistics reach the engine;
- whether filters reach the scan;
- how times are typed.

None of them puts Zarr metadata into the Arrow schema.

### 2.3 Hosts

- **Apache DataFusion** (CPU). A query engine written in Rust, built on Arrow, and designed to be extended. xarray-sql, zarr-datafusion, and Zax-SQL all use it. It lets extensions add optimizer rules and physical operators, and it can write plans back out as SQL (through its `Unparser`) and as Substrait.
- **NVIDIA GPU Query Engine (GQE)**. NVIDIA's GPU query engine, built on libcudf, the CUDA library behind NVIDIA's GPU DataFrames. It accepts Substrait, Flight SQL, and SQL, and uses DataFusion for planning.
- **DuckDB** (CPU). An in-process analytical SQL database, like SQLite for analytics. It reads SQL, and reads Substrait through its `substrait` extension.
- **DuckDB + [`gpudb`](https://github.com/singhpratech/duckdbgpumetaldbram)** (Apple Metal and NVIDIA CUDA). A DuckDB extension that rewrites SQL statements before DuckDB plans them, and runs parts of them on the GPU. It does not read Substrait. It runs GROUP BY, joins, and expressions inside `SUM` on the GPU, and for some shapes it fuses join and aggregate without producing join output. It never rewrites `SUM` and `AVG` over floating-point columns, because floating-point sums depend on the order of addition (section 8.6), and it leaves tables under 1 million rows to DuckDB. Its CUDA build needs an NVIDIA GPU of compute capability 7.5 or newer (Turing and later), and comes from `pip install duckdb-gpudb`; the community extension is CPU-only on Linux. On a Colab T4, it declined 12 of 13 einfold shapes (spike S5): every `DOUBLE` sum; joins that expand many-to-many, as a matrix product's does; operands that are subqueries rather than base tables, which is what eager aggregation emits; and window functions. Only a `BIGINT` reduction ran on the GPU, 4.3× faster.
- **DuckDB + [Sirius](https://github.com/sirius-db/sirius)** (NVIDIA CUDA). A GPU query engine built on libcudf that plugs into DuckDB as an extension. It intercepts every query through a hook in DuckDB's optimizer, converts it to Substrait, and runs it on the GPU. Operators it does not support fall back to DuckDB on the CPU. It supports filters, projections, hash and nested-loop joins, GROUP BY, aggregation, ORDER BY, top-N, LIMIT, and common table expressions (CTEs, the SQL `WITH` clause), over integer, floating-point, decimal, string, date, and timestamp types. Its `pin_table` function keeps a table resident in GPU memory between queries. Support for StarRocks, another analytical database, is announced.

DuckDB's Substrait reader has a quirk that einfold's writers must respect; see section 7.4.

## 6. Vocabulary, in full

The full glossary, with each term's einsum and relational equivalents. The design doc's section 6 lists the core terms.

### 6.1 Terms

| Term | Meaning | Einsum | Relational |
|---|---|---|---|
| **Dimension** | One variable of a fold: a set of columns the query equates, with how it compares them. A data axis such as `lat` becomes a dimension when a query joins or groups on it | An index label (`i`, `j`) | A class of equated key columns |
| **Coordinate** | A label along a dimension, such as `30.0°N` or a timestamp | — | A value in a dimension column |
| **Position** | An integer offset `0 … n−1` along a dimension | The value an index ranges over | — |
| **Extent** | The number of positions along a dimension | The size of an index | Exact distinct count of a dimension column |
| **Variable** | A named array over a set of dimensions; Xarray distinguishes data variables (such as temperature) from coordinate variables (such as latitude) | A tensor | A value column, together with its dimensions |
| **Dataset table** | The table a reader produces: one row per coordinate tuple over all of the dataset's dimensions, one column per variable | Several tensors | A wide table |
| **Operand** | One input to a fold: a table, a subquery, or a mask; often one variable over its own dimensions | An operand | A narrow table `(dims…, value)`, or any relation |
| **Support** | The coordinate tuples that have a row | The entries that are not the fill value | The rows that exist |
| **Fill value** | Zarr's value for chunks that were never written | Often called "zero" in sparse-tensor work | — |
| **Tile** | A rectangular box of positions (section 8.2) | A slice or block | A partition |
| **Chunk** | A storage tile in Zarr (in v3, possibly an inner chunk of a shard; section 6.2) | — | Usually one partition of a scan |
| **Layout** | A function from positions to row order within a tile (section 8.2) | A strided array view | Row order of a scan |
| **Fact** | Something known about an operand, tagged with how it is known (section 8.1) | — | Statistics, metadata |
| **Fold** (over a join) | A join of operands, grouped by output dimensions, with an aggregate folding each group's row values ([RFC 0001](rfcs/0001-folds-over-joins.md)) | — | `Aggregate(AGG(row value))` over joins on dimensions |
| **Aggregate** | How a group's values combine (`SUM`, `COUNT`, `AVG`, `MIN`, `MAX`), with SQL's rules for NULLs, empty groups and exactness | ⊕ | An aggregate function |
| **Semiring fold** | A fold whose row value is a ⊗-product of per-operand factors, and whose aggregate is a ⊕ that ⊗ distributes over; derived, not declared | A tensor contraction over a semiring | An aggregate-join query that admits eager aggregation |
| **Einsum** | The sum-product semiring fold | `ik,kj->ij` | `Aggregate(SUM(product))` over joins on dimensions |
| **Mask** | An operand that only filters which rows join and contributes no value: its factor is ⊗'s identity (1 for sum-product, 0 for min-plus, `TRUE` for existence). einfold's internal form of a predicate relating dimensions of different operands, such as a causal mask `q.t >= k.t` (section 9.1) | A 0/1 tensor multiplied in | A join with the set of allowed pairs |
| **Matmul pushdown** | Skipping rows whose value is exactly zero before the join of a contraction, whether the user writes the filter or einfold proves it safe (section 9.2) | Skipping zero entries | A filter pushed below a join |
| **Derived factor** | An expression that reads only one operand's columns, treated as one of that operand's value columns (section 9.1) | One factor of a product | A computed column |
| **Contraction tree** | A binary tree of pairwise contractions that evaluates a semiring fold. Folds that are not semiring folds have none | A contraction path | A tree of join-aggregates |
| **EinFold** | The package's fused join-and-fold operator, for any fold (section 10.2). The capitalized name is the operator; lowercase **einfold** is the package | One pairwise contraction | A groupjoin (a join fused with the group-by after it; section 10.1), generalized to groups that span both inputs |
| **EinFold's hash algorithm** | Based on Gustavson's 1978 sparse matrix multiply, generalized to folds (section 10.2) | Sparse contraction | — |
| **`Fold` relation** | The Substrait extension relation that carries an einsum and its facts | — | — |
| **`EinFoldExec`** | The DataFusion physical operator in einfold's reference executor | — | — |
| **Partial aggregate** | A group's state over part of the input: group existence plus the aggregate's state, mergeable with other parts (section 8.3) | A partial sum | A partial-mode aggregate |
| **Accumulator** | How an aggregate state's additions are computed numerically: plain `f64`, a binned reproducible sum, … (section 8.6) | — | — |
| **Reader** | A project that turns arrays into tables (section 2.1) | — | A table provider |
| **Host** | The engine that runs einfold's output (section 2.3) | — | — |
| **Target profile** | Data describing what a host supports (section 7.4) | — | — |
| **Relational form**, **extension form** | einfold's two output forms: standard joins and aggregates, or the `Fold` Substrait extension relation (section 7.2) | — | — |
| **Program mode** | Optimizing a batch of plans that read each other's results together (section 7.3) | — | — |
| **Spike** | A short, time-boxed experiment that answers one design question (section 13) | — | — |
| **E-graph** | A data structure that stores many equivalent versions of an expression compactly, by grouping equal subexpressions into classes (section 7.6) | — | Like the "memo" in which a Cascades-style query optimizer (a common framework for plan search) stores equivalent plans |
| **Equality saturation** | Applying rewrite rules to an e-graph until no new equivalent forms appear, or a limit is reached | — | — |
| **Extraction** | Choosing the cheapest expression in an e-graph under a cost model | — | Choosing the cheapest plan |

These docs say **dimension** wherever einsum literature says "index". "Index" appears only in the einsum column above, in titles of cited work, and in einsum strings. That avoids collisions with database indexes and with Xarray's indexes (pandas index objects attached to coordinates).

Notation: `dims(T)` is the set of dimensions of operand `T`. `O` is the set of output dimensions. A dimension in two or more operands is **shared**. A dimension not in `O` is **summed**. A dimension in exactly one operand and not in `O` is **private** to that operand.

### 6.2 Nuances to honor

These are facts about the data model that einfold's rewrites must respect.

- **Coordinates are not positions.** Dimension columns hold coordinates (latitudes, timestamps), not integer offsets. Dense execution needs a map from coordinates to positions. Spike S4 ([`spikes/s04-coords`](spikes/s04-coords/README.md)) classified the coordinates of 127 ERA5 and CMIP6 stores:
  - Almost every one-dimensional coordinate is strictly monotone, so a **sorted lookup table** is the general exact map. A range filter on coordinates becomes a range of positions by binary search.
  - **Affine** maps (`start + k·step`) are common but must reproduce the stored values *bitwise*. Nine grids, including three of four ERA5 stores tested, are affine only up to rounding.
  - Monthly times are affine in **calendar months**, not in their stored days.
  - Vertical levels are never affine. Curvilinear and unstructured grids (most ocean models) have no coordinate map: their dimensions are positions, and latitude and longitude are data.

  Section 8.1 lists the resulting forms. Times decoded from CF metadata (below) are coordinates too, and readers type them differently, so a map is exact only for the representation a reader emits.
- **Joins compare coordinates exactly.** Two operands align on a shared dimension only where their coordinates are equal. Float coordinates that differ in the last bit do not join. That matches what the query says, so einfold preserves it. Dense alignment additionally requires that both operands map coordinates to positions the same way. Otherwise EinFold uses its hash algorithm.
- **The fill value decides what a missing chunk means.** Zarr arrays are logically dense: every position has a value, and chunks that were never written read as the array's fill value. The fill value may be 0, NaN, or something else. Many climate datasets follow the CF (Climate and Forecast) metadata conventions, a standard for describing units, time encodings, scaling, and missing data. Readers that decode CF metadata may turn missing data into NULL. The fill value (what unwritten chunks return) is also distinct from a missing-data sentinel, which the `missing_value` Zarr convention declares (section 8.1). They are often equal, but need not be. Which of these a reader emits decides whether skipping a missing chunk is exact (section 8.1).
- **SQL semantics, not Xarray semantics.** SQL `SUM` skips NULL but propagates NaN, while Xarray's `sum` skips NaN by default. Whether missing data reaches the plan as NULL or as NaN is the reader's decision. einfold preserves whatever the plan means.
- **Groups exist only where something joined.** A SQL group appears in the output only if at least one joined row reached it, and its `SUM` is NULL only if every contribution was NULL. Every rewrite must preserve both: which groups exist, and which values are NULL.
- **Bag semantics.** SQL sums every joined row, including duplicate coordinate tuples. That matches einsum over a coordinate list (COO, the sparse format that stores each nonzero as a coordinate tuple plus a value), where duplicate entries add.
- **Memory order is metadata.** A Zarr chunk is not always stored row-major. v2 has an `order` field (C for row-major, F for column-major), and v3 can reorder axes with a transpose codec (codecs are the encoding steps, such as compression, that Zarr applies to each chunk). v3 sharding packs many inner chunks into one stored object, called a shard, which makes storage tiling two-level.
- **Chunk grids may be irregular.** The last chunk along a dimension is partial when the chunk size does not divide the extent. Proposed Zarr extensions also allow chunk sizes that vary along a dimension.

## 7. Architecture, in detail

### 7.3 Deployment modes

**SQL-to-SQL.** einfold parses the SQL into DataFusion's *unoptimized* logical plan, applies only its own rewrites, and writes SQL with DataFusion's `Unparser`. The target engine then optimizes the result itself. Spike S7 ([`spikes/s07-unparser`](spikes/s07-unparser/README.md)) found this the only safe order:

- Unparsed from unoptimized plans, all 34 queries tested (12 of einfold's shapes and all of TPC-H) ran in DuckDB with correct results.
- Unparsed from DataFusion's *optimized* plans, 31 were rejected by DuckDB, and TPC-H Q22 silently returned a wrong answer: the Unparser had dropped two of its predicates.
- So everything einfold unparses is checked, by re-parsing and comparing plans, and by the shared equivalence suite run through each dialect (section 11).
- The DuckDB dialect must be able to write `MATERIALIZED` CTEs (section 9.3), which the Unparser does not do today.
- Result types can differ between engines even for correct SQL. DuckDB's `AVG` over a `DECIMAL` returns `DOUBLE`, where DataFusion's returns a `DECIMAL`. To keep a result's shape unchanged (principle 2), einfold casts wherever the source and target engines type an expression differently.

**Program mode.** A ddx `BackwardProgram` is a list of named steps, each of which may read earlier steps by name. An optimizer rule sees one plan at a time, so only program mode can share work across steps (section 9.5) and cache plans for a whole program (section 8.5). Program mode works with any output form.

### 7.4 Target profiles

A profile also records whether the host scans a shared CTE once (section 9.5), and quirks of its plan reader that the writers must respect. For example, DuckDB's Substrait reader honors a relation's `emit` field (which selects and reorders output columns) only on projections. On joins, filters, sorts, fetches, cross joins and set operations it silently ignores `emit` and returns the leading columns (found by ddx, [ddx PR #117](https://github.com/xqlsystems/ddx/pull/117)). So einfold's Substrait writer puts `emit` only on projection relations for every host, and adds an explicit projection where it needs to reorder columns.

Fallback is per subplan. If a host runs the `Fold` relation for dense 64-bit floats but not for sparse integers, einfold emits the extension form for the first subplan and the relational form for the second, in the same plan.

### 7.6 Rewrite engine: egglog, in a hybrid design

egglog (Zhang et al., 2023) is the successor to egg (Willsey et al., 2021), and adds rules in the style of Datalog, a rule language from databases that derives new facts from existing ones.

**Why.** Hand-written rewrite passes must be applied in some order, and an early pass can destroy an opportunity a later pass needed. Hand-written heuristics also have to pick one direction for rules that help in either direction. An e-graph keeps all versions and lets the cost model choose. The strongest precedent is SPORES (Wang et al., 2020). It translates linear algebra into relational algebra, which is close to einfold's own IR, then optimizes with egg, and translates back. It ran 1.2–5× faster than SystemML, a production ML system. Its wins came from algebraic choices like einfold's: distributing a product over a sum when that exploits sparsity, and factoring it back when that is cheaper.

**Evidence.** Spike S16 ([`spikes/s16-egglog`](spikes/s16-egglog/README.md)) encoded einfold's rewrites in egglog 3.0:

- egglog found SPORES's rewrite `sum(WH) = Σₖ (Σᵢ W)(Σⱼ H)` (50,000× cheaper by the cost model). It expanded `X·(A + B)` only when `X` was sparse, factored a repeated weight out of a sum, turned a sum over a missing dimension into a scale factor, and fixed a badly ordered ddx gradient (64× cheaper).
- With associativity left to egglog, the e-graph grew about 3.7× per extra matrix in a chain: 1.6 million tuples and 24 s for 11 matrices. At 12 it hit the size limit before finishing, and extraction returned a plan 800,000× worse than the hybrid's. The hybrid matched the full search wherever that search finished.
- Every result was identical across repeated runs and across 1 and 4 threads.

Three follow-up spikes tested the hybrid at scale:

- **N-ary nodes** (S17, [`spikes/s17-nary`](spikes/s17-nary/README.md)). With each sum-product region as one node over a multiset of operands, a pure region saturates in one iteration, and the e-graph grows linearly: about 4 tuples per operand, up to 1000 operands drawn from the Einsum Benchmark's instance families. The binary form stopped saturating at 16 to 26 operands. Distributivity is still exponential: `k` operands that are sums give `2^k` expansions, and saturation timed out at `k = 8`.
- **Planning time** (S18, [`spikes/s18-planning-time`](spikes/s18-planning-time/README.md)). On TPC-H's join-and-`SUM` queries and on ddx's training einsums, egglog plans in 0.5–1.2 ms, starting from an e-graph that already holds the rules. That is less than DataFusion's or DuckDB's own planning time for the same queries (0.7–4.6 ms). Parsing the rules costs about 1.5 ms, so a session loads them once and clones the e-graph per query.
- **Tiles** (S20, [`spikes/s20-tiles`](spikes/s20-tiles/README.md)). Retiling, blockwise products and partial-sum trees, encoded as terms with Cubed's memory model in the cost function, reproduce Cubed's matrix-multiplication plans exactly, and never extract a plan over the memory budget (section 9.4).

**Design rules from the spikes.**

1. **No associativity rules over products.** Contraction order belongs to the planner.
2. **One n-ary node per sum-product region.** Even without associativity, commutativity and reordering of nested sums grew the e-graph about 2.2× per operand (S16). A region is a single node, `SP(ops, sums)`, holding a multiset of operands and a set of summed dimensions, built on egglog's multisets. S17 confirmed linear growth. Rules that compute over a multiset's elements must run in egglog's naive mode (rematching everything each iteration), which is cheap only while iterations are few.
3. **The planner runs inside extraction.** egglog's cost model receives each `SP` node's operands, and runs the contraction planner on them, so extraction compares algebraic alternatives by their *planned* cost (S17). With a greedy planner that matches opt_einsum's, this takes milliseconds up to about 100 operands, and seconds at 1000. Large regions need an incremental planner and a cache of planned costs.
4. **Guard distributivity.** Expand an operand that is a sum only when a sparsity fact says the expansion can pay (S16's sparse case), and cap the number of such operands per region.
5. **Bounded, deterministic schedules.** Use a fixed rule order with iteration and size limits, and no random sampling of rule matches. If a limit is hit, keep the best plan found by the rules that did finish. Never rely on an unfinished associativity search. S16 to S18 found identical results across repeated runs and thread counts.
6. **Planners and rules share tests.** In S16, the hand-written planner first dropped a sum over a dimension that no operand has, a case the egglog rule handled correctly. The equivalence tests of section 11 run on both.
7. **Load rules once per session.** Clone the loaded e-graph for each query (S18).

**Out of scope, and future work.** Two uses of e-graphs stay in ddx: simplifying the scalar derivative expressions inside projections (`tanh`, `exp`, `CASE`, ddx's NULL handling), which are not sum-products, and choices specific to automatic differentiation, such as saving a region versus recomputing it, or where to checkpoint a deep expression. After milestone M1 (section 16), einfold's e-graph could accept non-einsum regions as opaque nodes whose costs the caller supplies. ddx could then share einfold's e-graph instead of building a second one.

**Risks.** egglog 3.0 is young (released August 2026), so egg is the fallback. RisingWave, a company that builds a streaming SQL database, found that a SQL optimizer built on egg planned a 6-table join in 39 ms, where DuckDB took 5 ms, and that cost functions were hard to debug. einfold's e-graphs are much smaller than a SQL optimizer's, since only sum-product regions enter them: S18 measured about 1 ms per query once the rules are loaded. Plan caching (section 8.5) absorbs repeated planning. An earlier egg-based optimizer for DataFusion's expressions, datafusion-tokomak, has been inactive since 2022, for reasons not yet investigated.

## 8. Core abstractions, in detail

Several optimizations turned out to be the same idea in different places. This section defines each shared idea once. Sections 9 and 10 use them.

### 8.1 Facts

A **fact** is something known about an operand, tagged with how it is known. Every planning decision reads facts, and every fact has a source and a precision.

**Kinds of fact.**

| Fact | Example | Used by |
|---|---|---|
| Dimensions of each variable | `w` depends only on `lat` | Variable separation (9.1) |
| Extents | `lat` has 721 positions | Contraction-tree cost (9.3), memory (9.4) |
| Coordinate map | `lat = 90 − 0.25·position`, or a sorted table of latitudes | Dense alignment (10.2), slicing and masks (9.1) |
| Layout | row order within a chunk | Dense kernels and streaming (10.2, 10.3) |
| Tiling | chunk shape, shard shape, chunk grid | Tiling (9.4), reduction at the source (10.5) |
| Support | which chunks exist; which rows exist | Pruning (9.2), block-sparse algorithm (10.2) |
| Fill value, and how the reader emits it | fill 0, written as no row | Support (below) |
| Size and degree | number of entries; maximum rows per key | Contraction-tree cost (9.3), memory (9.4) |
| Value statistics | minimum, maximum, null count, and zero count per chunk | Chunk skipping and zero elimination (9.2) |
| Finiteness | no NaN or infinite values in a column | Zero elimination (9.2) |
| Constraints | primary keys, uniqueness, `NOT NULL`, `CHECK` | Unique coordinate tuples (10.2), non-NULL dimensions (9.1), mask support (9.1) |

**Precision.** Each fact is one of:

- **Exact.** Known from metadata or a reader's guarantee: Zarr shapes, chunk grids, a variable's dimensions. DataFusion's table statistics mark such values `Precision::Exact`.
- **Bound.** A guaranteed upper bound, such as a bound on the size of a join computed from degree statistics (below). Deeds et al. (2025) use such bounds in Galley, a query optimizer for sparse tensor programs (Deeds et al., §6.3.2; Chen et al., 2023, who survey bounds of this kind).
- **Estimate.** A best guess from statistics, such as a join size estimated by assuming uniform density. DataFusion's `Precision::Inexact`.
- **Measured.** Observed while running, such as the exact number of entries in an intermediate result that EinFold just produced (section 10.4).

**Rules.**

- Correctness may depend only on Exact facts. Variable separation, dense alignment, skipping missing chunks, mask support, and inferred zero elimination all need Exact facts.
- Memory decisions use Exact facts or Bounds: slicing, and choosing which input EinFold holds in memory.
- Cost decisions may use Estimates.
- A Measured fact replaces an Estimate or Bound for the rest of the query.

**Sources (pluggable fact providers).**

- **Zarr metadata.** Shape, chunk grid, shards, dimension names (v3's `dimension_names` field, or v2's `_ARRAY_DIMENSIONS` attribute), memory order, fill value, and which chunks exist.
- **Zarr conventions.** A Zarr convention is a published, named set of metadata attributes that gives arrays extra meaning without changing how they are stored. The `spatial` convention gives affine maps from positions to X/Y coordinates. The `missing_value` convention names the value that marks missing data. XQL Systems' proposed `layout:` convention ([`layout-convention.md`](layout-convention.md)) gives curve orderings, chunk visit order, records of which chunks were written, and per-chunk summaries, each marked exact or not. It is useful to readers and engines without einfold.
- **The reader.** How it flattens: which variables depend on which dimensions, whether it dictionary-encodes a column and how, and how it emits fill values and missing data.
- **Table statistics.** Readers can pass Zarr dimension bounds to the engine as table statistics, which engines use to skip data and estimate costs. Today only xarray-sql does: exact row counts, and exact minimum, maximum and null count for dimension columns, none for data variables (spike S1). In DataFusion these are per-column `min`, `max`, and `distinct_count` in its `Statistics` structure. Statistics alone cannot prove density or functional dependencies, so they yield Estimates unless a reader marks them Exact.
- **SQL constraints.** A declared primary key or unique constraint says that a table's dimensions identify its rows. `NOT NULL` says a dimension is never NULL. A `CHECK` constraint such as `CHECK (i >= j)` describes where rows can exist. Like a primary key in any database, these change no results; they are Exact facts the optimizer may use. DataFusion records primary-key and unique constraints without enforcing them, and DuckDB supports `CHECK`. No reader declares its dimension columns as a key today, and only zarr-datafusion marks coordinates `NOT NULL` (spike S1). So for Zarr data these facts come from the providers, and declared constraints matter mostly for tables users create.
- **Per-chunk statistics.** Databases keep minimum, maximum and null counts per block of data: Parquet (a columnar file format) per row group, and Iceberg (a table format for data lakes) per data file. Engines such as DataFusion use them to skip blocks a filter rules out. Readers should report the same statistics per Zarr chunk, from `layout:summaries` where present, so engines skip chunks without einfold's help. None of the three readers tested does yet for data variables (spike S1). A zero count per chunk, next to the null count, makes "this chunk is all zeros" provable.
- **Degree statistics.** `D(X | Y)` is the maximum number of rows for any one value of `Y`. For dense arrays these follow from extents. Missing chunks give them at chunk granularity. Sparse tables need the reader to compute them. DataFusion's column statistics have no field for them.
- **The user.** Declared facts.
- **Producers of plans, such as ddx.** ddx proves facts that einfold's dense algorithms need. Every ddx program checks that each input table's dimensions identify its rows, and that each derivative table has one row per key; [ddx PR #120](https://github.com/xqlsystems/ddx/pull/120) records these as a `Verified` set that callers keep across runs. ddx also knows which columns are dimensions and which are values, and the keys of every step in its program. ddx hands these over as Exact facts on its steps, through the same carrier as reader facts.
- **The executor.** Measured facts.

**Inside the optimizer,** facts are egglog analyses: Datalog rules derive them, and egglog's merge functions combine them, keeping the most precise value (section 7.6).

**How facts reach the plan: a side channel.** Spikes S1–S3 compared the candidate carriers: Arrow field metadata (key-value pairs attached to a column's schema), a Substrait extension attached to the read, and a side channel.

- *No reader writes facts into Arrow metadata today* (S1).
- *Arrow metadata does not survive the trip* (S2, [`spikes/s02-carrier`](spikes/s02-carrier/README.md)). DuckDB drops all field and schema metadata at its Arrow boundary. Substrait's schema has no metadata field, so a plan carries none: metadata seemed to survive a round trip only because the consumer looked the table up again in its own catalog.
- *Inside DataFusion, metadata is kept by column name, not by meaning.* It survives filters, limits, joins and casts, which change the rows or values a fact describes. Where a union's inputs disagree, the left input's metadata silently wins.

So facts travel in einfold's own **fact table**, keyed by table and column, and filled by **per-reader fact providers**. xarray-sql's provider reads its Python context, duckdb-zarr's its metadata table functions (`read_zarr_metadata()`, `read_zarr_groups()`), and zarr-datafusion's its extended `DESCRIBE`. Arrow metadata can still be one *input* to a provider, read at the scan, and nowhere above it. For the extension form, facts that the host needs at run time travel as fields of the `Fold` relation itself.

Two rules follow:

- **A fact names the plan node where it holds.** A fact about a column's meaning (its dimensions, layout or coordinate map) holds above a filter, sort or rename. A fact about its rows or values (density, zero count, minimum and maximum, "every chunk written") does not hold above any operator that changes the rows or values. Facts are re-derived above such operators, and at every union, never inherited by name.
- **How a reader emits missing data is itself a per-reader fact.** The same store reads as NULL through one reader and NaN through another (S1). The fill-value table below needs to know which.

**Forms of coordinate map** (spike S4). A provider tries these in order, and records the first that reproduces the stored values exactly:

| Form | Exact when | Size |
|---|---|---|
| Affine: start, step, stored type | `cast(start + k·step)` equals every stored value, bitwise | constant |
| Calendar: start month, calendar, day rule | times are consecutive calendar months | constant |
| Sorted table of stored values | always, for a strictly monotone coordinate | one value per position |
| None | the coordinate is not monotone, or is 2-D or per-cell | — |

A grid that is affine only up to rounding gets an Estimate-level affine fact for costing, and its sorted table as the Exact map.

**Support and the fill value.** Support, meaning which coordinate tuples have rows, is where sparsity reasoning and cardinality estimation (a database optimizer's estimate of how many rows an operator produces) meet. For a product, the support of the result is the join of the supports of the factors. For a sum over a dimension, it is the projection that drops that dimension (Deeds et al., §6.1). Pruning (9.2), block-sparse execution (10.2), and density measurement (10.4) all reason about support. A missing Zarr chunk may be skipped only as far as its fill value allows:

| Fill value, as emitted by the reader | Can a missing chunk be skipped? |
|---|---|
| No rows (the reader omits fill entries) | Yes. The table really has no rows there. |
| Rows with value 0 | Its products are 0, so the work can be skipped. But the groups it reaches still exist, and still equal 0 where nothing else contributes. Represent the skipped tile by the partial aggregate "reached, value 0" (section 8.3). |
| Rows with value NULL | Its products are NULL, so the work can be skipped. The groups it reaches still exist. Represent it by "reached, no value". |
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

**Tiling is reindexing.** Splitting dimension `d` into tiles of size `c` writes each position as `d = c·d_blk + d_in`, so a sum over `d` becomes a nested sum, over the block index `d_blk` and then within the block. That identity is exact, which lets tiles live in the algebra (section 9.4). Cubed, a Python library that runs NumPy-style array programs within a fixed memory budget, builds everything from two such primitives: `blockwise`, which maps input chunks to output chunks, and `rechunk`, which changes the tiling. Its matrix multiplication is an einsum over blocks: one task per triple of blocks `(i, j, k)`, followed by a tree of partial sums over `k`.

Slicing a summed dimension produces partial aggregates that must be combined (section 8.3). Slicing a kept dimension produces disjoint outputs that are concatenated. Ragged edges (a partial last chunk) break CuTe's requirement that tile sizes divide extents evenly. They are represented as a padded tile plus a bounds check, which is how CuTe handles leftovers. Irregular grids are represented as an explicit list of tiles.

### 8.3 Partial aggregates

A **partial aggregate** is the state of one output group, computed over part of the input and combined later. Eager aggregation (10.1), EinFold (10.2), slicing (9.4), parallel partitions, and reduction at the source (10.5) all produce partial aggregates. So their SQL semantics are defined once, here. Their numerics are defined in section 8.6.

**State.** A group's partial aggregate has two independent parts:

- **group existence** (`reached`): did any joined row reach this group? This decides whether the group exists, and is the same for every aggregate;
- **the aggregate's state:** for `SUM`, the running sum of the non-NULL values, or "none"; for `COUNT`, the number of non-NULL values; for `AVG`, a sum and a count; for `MIN` and `MAX`, the extreme so far, or "none".

**Update.** For each joined row, set `reached`. Then, if the row's value is not NULL, fold it into the aggregate's state. Setting `reached` before the NULL test is what keeps a group reached only by NULL values in the output: as NULL for `SUM`, `AVG`, `MIN` and `MAX`, and as 0 for `COUNT`.

**Combine.** `reached` is OR-ed. The aggregate states are merged with the aggregate's own operation, with its empty state as the identity. When einfold controls the combine (in its reference executor, and when determinism is requested), it combines partial aggregates in a fixed order: partition order, then tile order.

**Finish.** `SUM` returns its sum, or NULL for "none". `COUNT` returns its count. `AVG` returns sum ÷ count, or NULL when the count is 0. A group never reached does not appear.

**Numerics.** How a float state is accumulated (its precision, and whether its result depends on the order of additions) is the *accumulator*, set by section 8.6. `COUNT`, `MIN` and `MAX` are exact on any type.

### 8.4 Order is a layout

The order in which rows arrive is a layout: which dimensions vary slowest and which fastest. Treating order this way connects three things:

- **Streaming.** EinFold can emit one output row at a time when its input arrives grouped by the output's leading dimensions (section 10.2). That is a condition on the input's layout.
- **Choosing output order.** Each node of the contraction tree can emit rows in the order its consumer wants (section 10.3).
- **Reordering.** When the order is wrong and the extents are known, a distribution counting sort (which counts how many rows fall in each bucket, then places each row directly) fixes it in linear time. Gustavson's sparse transpose is exactly that sort.

Within a Zarr chunk, row order comes from the chunk's memory order (section 6.2). CuTe's `coalesce` operation tells whether a set of dimensions forms one contiguous run in memory. That decides whether a dense kernel can read a buffer directly, and whether a matrix-multiply call must treat an input as transposed.

**A layout is two facts: the coordinate map and the row order.** Spike S3 ([`spikes/s02-carrier`](spikes/s02-carrier/README.md)) ran the layout rules of section 13.5 (in this supplement) on a 1000 × 1000 array in DataFusion and DuckDB.

- *The rules correctly predicted which coordinates every operator outputs, on both hosts.* That part of a layout, the map from coordinates, propagates as section 13.5 says.
- *Row order did not follow the rules, and the two hosts fail in opposite ways.* DataFusion knows a declared order: it tracks it through filters, projections and unions, and answers `ORDER BY` with a merge rather than a sort. But without `ORDER BY` it spreads rows across partitions, so they arrive scrambled. DuckDB keeps insertion order through every operator except an aggregate, but its planner doesn't know the order, and always sorts for `ORDER BY`.
- *Neither keeps order through a hash aggregate,* which is what a contraction node in the relational form is.

So row order is a fact only where a host promises it. On DataFusion, einfold asks for the order it needs: the operator declares a required input ordering, which costs a merge, not a sort, when the scan's order is declared. On DuckDB, einfold relies on order only if it adds the `ORDER BY` and pays for the sort. The output order of a contraction node is whatever EinFold emits (section 10.3), which only the extension form can promise.

### 8.5 Structure and values

Most of what einfold computes depends only on **structure**: the einsum, the extents, the tilings, and the support. Only the final arithmetic depends on **values**. Separating the two lets structure be computed once and reused:

- **Plan caching.** Contraction trees are cached, keyed by the einsum in a canonical form plus extents rounded into buckets. Blacher et al. note that repeated einsums should not be re-planned. In program mode (section 7.3) the cache key covers the whole program. Keys never include table names: a step that reads another step is identified by its position in the program, not its name. ddx gives every step a fresh name per program (`__ddx_{id}_…`), so name-based keys would miss on every training step.
- **Symbolic–numeric split.** When the support is fixed and only values change, Gustavson computes the output's structure once (the symbolic pass), then runs only a numeric pass with no hashing and no "already touched?" tests (section 10.2). ddx's training steps fit this exactly: each step runs the same contractions on new values.
- **Measured facts.** Densities measured in one run (section 10.4) remain valid for later runs over the same support. In program mode, a backward step's support equals that of the forward step it differentiates, including the user's filters and masks, so the forward pass's measured support becomes a fact for the backward pass.

### 8.6 Numerics: determinism and precision

Floating-point addition is not associative: `(a + b) + c` can differ from `a + (b + c)` in the last bits. So a parallel sum, whose additions happen in whatever order threads finish, can give slightly different results on each run. This section sets einfold's policy.

**What others do.** Spike S19 ([`spikes/s19-float-sums`](spikes/s19-float-sums/README.md)) surveyed SQL engines and JAX, Google's numerical computing library, and measured DuckDB and DataFusion:

- No mainstream SQL engine guarantees repeatable float sums in parallel. DuckDB, PostgreSQL, BigQuery and Snowflake all treat run-to-run variation as expected, and DuckDB's parallel `SUM` returned 19 different results in 20 runs. Engines accumulate `DOUBLE` sums in 64 bits, and offer `DECIMAL` for exact results.
- Compensated summation, such as DuckDB's Kahan-based `fsum`, is not a determinism fix: in parallel it still varied. Only accumulators whose result is independent of the order of additions are deterministic in parallel.
- Determinism and accuracy are different properties. The repeatable single-threaded sums were the least accurate, because adding strictly left to right accumulates rounding error.
- JAX treats three concerns separately. Randomness is deterministic by design. The order of operations is fast by default and deterministic by opt-in, through process-wide flags that cost throughput. Precision is fast by default (a float32 matrix multiply runs in bfloat16, a 16-bit floating-point format, on TPUs), with controls per operation and per block of code.

#### einfold's policy

einfold follows JAX's split.

**1. Guaranteed by design: einfold never adds nondeterminism.**

- A rewritten plan is never less deterministic than the plan einfold received. einfold's own decisions (the contraction tree, the choice of algorithm, run-time switching in section 10.4) depend only on the plan, the facts, and the data, never on timing. The rewrite engine follows the same rule: fixed schedules and no random sampling (section 7.6). Spike S16 found identical results across repeated runs and thread counts.
- Rewrites still change *which* numbers are added together and when, so a rewritten plan's float results can differ in their last bits from the original plan's. "Equivalent" in principle 4 means mathematically equivalent.

**2. Exactness invariant.** einfold never turns an exactly computed value into a rounded one, and never changes an exact value that a comparison, filter, join key, ordering or `LIMIT` depends on. `COUNT`, `MIN`, `MAX`, and integer and `DECIMAL` arithmetic stay exact. Only floating-point sums may change in their last bits, under the rest of this policy. ddx relies on exactly this split: it already assumes that a float `SUM` can differ between a saved result and its recomputation, and that the exact operations do not.

- **Integer and `DECIMAL` overflow.** Eager aggregation, distributivity and reordering can create intermediate values the original plan never computed. For example, `a·Σbⱼ` can overflow when every `a·bⱼ` does not, and reordering additions can overflow a partial sum. In SQL, integer overflow is usually an error, so such a rewrite could turn a correct result into a failed query. einfold applies these rewrites to exact types only when Exact facts or Bounds prove that no intermediate value can exceed its type. Otherwise it leaves that part of the plan unchanged.

**3. Determinism is a scoped setting, off by default on hosts.**

- By default, einfold emits the fastest forms the host supports, with the host's usual float behavior, as every SQL engine in spike S19 does.
- A user can request determinism per session or per query, as with JAX's flag or DuckDB's `threads=1`. einfold then uses only mechanisms the host's target profile lists as deterministic: a single partition, the host's own deterministic mode, or a cast to `DECIMAL`. If the host offers none, einfold leaves that subplan in its original form and says why.
- `DECIMAL` is the portable deterministic path, since every engine recommends it and it is exact. Its limits are value range and speed.

**4. Deterministic by default where einfold executes.**

- The reference executor (`EinFoldExec`) and the `Fold` relation's specification are deterministic by default, and the conformance suite checks that repeated runs give identical bits. einfold controls these, and conformance testing needs repeatability, much as JAX on TPU is deterministic in practice.
- Ways to accumulate `value` deterministically, in order of preference:
  1. **An order-independent accumulator.** Reproducible summation gives the same bits in any order. It works either by binning values by exponent, as the ReproBLAS library (Demmel and Nguyen) does, or with an exact "superaccumulator" wide enough to hold any sum without rounding. Results are then deterministic, parallel, and accurate, and a rewrite that only reorders sums gives the original plan's exact bits. Spike S8 ([`spikes/s08-deterministic-sums`](spikes/s08-deterministic-sums/README.md)) measured the cost:
     - A **binned sum** gave identical bits under every order, thread count, and GPU atomic schedule tried, and on the test data equaled the correctly rounded sum.
     - On CPU it cost 3× a parallel plain sum for one long sum, and 4.4× for grouped sums, with 24 bytes of state per group. On a GTX 1080 Ti it cost 3.5×.
     - It needs the largest absolute value in advance: from per-chunk statistics (section 8.1), or a second pass.
     - A **superaccumulator** suits one large parallel sum (2× a parallel plain sum), but its 576-byte state makes it 11× slower for grouped sums, and 83× slower on GPU.

     einfold's executors use the binned sum.
  2. **A fixed combine order** (section 8.3). Free on CPU and GPU (S8), but deterministic only for a fixed input order, which hosts don't guarantee through parallel scans (spike S3). Useful inside einfold's own executor, and only as accurate as its order.
  3. **Fixed-point values** (SQL `DECIMAL` or scaled integers) where the value range allows. Exact, and `gpudb` already sums these on the GPU. Costly to prove safe.
- Users who want maximum speed in the reference executor, for example when training models, can turn determinism off.

**5. Precision is its own setting.** Modeled on JAX's precision levels:

| Level | Accumulator | Einsum-form kernels |
|---|---|---|
| `fast` | the host's default | may use reduced-precision formats where the host offers them, such as bfloat16 or TF32 (NVIDIA's reduced-precision format for matrix multiplication) |
| `default` | 64-bit for 32-bit and 64-bit floats | full input precision |
| `highest` | a reproducible (binned) accumulator; exact where the type allows | full input precision |

`default` follows SQL's convention of 64-bit sums, and Gustavson's advice to accumulate in higher precision and round once (section 10.2). Precision and determinism are set independently, except that `highest` is also deterministic.

**6. Warn where small differences become big ones.** In SQL, a sum that varies in its last bit can flip a comparison. Sums feed `WHERE` and `HAVING` filters, `ORDER BY … LIMIT`, `MIN` and `MAX` ties, `GROUP BY` on computed keys, and joins on computed values. Then a last-bit difference becomes different rows. ddx, for example, needs a tie-breaking rule for `MIN` and `MAX` because sums across partitions vary. einfold traces each float sum through the plan, and when one feeds such a decision without determinism requested, it warns and suggests deterministic mode. It does not change the plan on its own, since hosts don't either.

**In target profiles,** each host records which deterministic mechanisms it offers and which precision levels it supports.

**On GPU hosts,** the relational form still gets its plan-shape benefits. Fast deterministic float sums on GPU need host support for an order-independent accumulator, which einfold will propose to the `gpudb`, Sirius, and GQE maintainers.

## 9. Logical optimization, in detail

Sections 9.1, 9.2 and 9.5 are egglog rules and analyses. Section 9.3 is a specialized planner, which egglog's extraction calls on each sum-product region. In section 9.4, a planner proposes candidate tilings and extraction chooses among them (section 7.6).

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
   - an Arrow dictionary-encoded column whose dictionary the reader built from the variable's coordinates. In that case the dictionary's list of distinct values *is* the variable at its true shape, so the operand is read without scanning the repeated column. zarr-datafusion already dictionary-encodes coordinates this way, with 16-bit keys. Its dictionaries hold coordinates, though, not lower-dimensional data variables, which it mishandles (spike S1). So this source is usable only for coordinates, and only with the provider's Exact confirmation.

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
   - A predicate that relates *dimension* columns of different operands and is not an equality, such as a causal mask `q.t >= k.t` or a sliding window `abs(q.t - k.t) < w`, becomes a **mask**: the set of allowed coordinate pairs, joined in like any other operand. That is exactly what the SQL predicate means, including which groups exist. A mask computable from coordinates has a known support, so each tile is Exact-known to be fully allowed, fully masked, or partly masked. Turning a predicate on coordinates into one on positions needs an Exact fact that the coordinate map is monotonic, which nearly all one-dimensional coordinates satisfy (spike S4). A stored table of allowed pairs, such as a graph's edges, is a mask too. In the e-graph a mask is one more leaf, so every rewrite applies to it. Users never write masks; they write SQL predicates, and the mask is einfold's internal representation (principle 8).
   - A predicate that compares *values* of different operands, other than through equality on dimensions, does not match. Detection stops.
7. **Map group keys.** Each group key must be a column in some dimension class. The classes it names form `O`.
8. **Classify the fold.** A semiring is a pair of operations, "add" (⊕) and "multiply" (⊗), with ⊗ distributing over ⊕. The fold is a *semiring fold* when its row value is a ⊗-product of the factors from step 3 and its aggregate is ⊕: `SUM` of `*` (einsums), `COUNT` (sum-product over 0/1 indicators), and in M2 `MIN` or `MAX` of `+` (tropical semirings, used for shortest paths) and `MAX` of `*` when every factor is provably non-negative. In each of these semirings, ⊕'s identity also annihilates (`0 ⊗ a = 0`), which lets a dense kernel pad absent entries with it; for `MAX` of `*` that holds only for positive values, since `−∞ · 0` is NaN, so such kernels must not pad with −∞. `AVG` is *algebraic*: `SUM / COUNT` over the same join, then a division. Any other row value or mergeable aggregate is still a fold, and gets EinFold's fusion, but no algebraic rewrite. The classification is derived from the query, never declared.

**Correctness.** Every rewrite in section 9 holds under bag semantics, so detection does not require unique coordinate tuples. Dense execution does (section 10.2). A join on `=` drops rows whose key is NULL, a join on `IS NOT DISTINCT FROM` matches NULL to NULL, and `GROUP BY` keeps NULL as its own group. The relational form keeps these semantics because it is relational, and dense execution requires non-NULL dimensions, which Zarr guarantees. Anything detection cannot prove is left unchanged.

**In egglog.** Steps 2–6 are rewrite rules and analyses. Dimension classes (step 5) come from the e-graph's built-in union-find. The decision whether to expand a product (step 4) needs no separate search: the e-graph holds both forms, and extraction picks the cheaper one.

**Prior work.** Blacher et al.'s four rules, read in reverse; Galley's logical normalization (Deeds et al., §4); SPORES (Wang et al., 2020).

### 9.2 Prune the support

**Idea.** Before contracting, remove rows that cannot change the result: rows with no join partner anywhere in the einsum, and rows whose factor is exactly zero.

- For acyclic joins, Yannakakis (1981) showed how to remove all such rows with semi-joins (filters that keep a row only if it has a partner in another table), passed up and then down a join tree. Afterwards no join does wasted work. Einsums are usually acyclic: chains, trees, and stars.
- Predicate transfer (Yang et al., 2024) is a cheaper variant that passes Bloom filters (compact, approximate set-membership tests) along the join graph instead of exact semi-joins.
- Gustavson's analysis of wasted work is the two-operand case. It traces waste to rows of `A` that are empty, entries of `A` whose column matches an empty row of `B`, and the reverse (Gustavson, 1978, §3.3).
- At tile granularity this is partition pruning: a tile whose coordinate range cannot match is never read. Per-chunk value statistics (section 8.1) extend this to filters on values: a filter `v <> 0` skips every chunk whose minimum and maximum are both 0.

**Exact zeros ("matmul pushdown").** A row whose factor is exactly 0 contributes nothing to a sum of products. Skipping such rows before the join is a filter pushed below a matrix multiplication, which is how an audience member once described this trick, as "matmul pushdown". There are two ways it happens:

- *Declared.* The user writes the filter, for example `WHERE x.v <> 0`, or ReLU as `WHERE z > 0`. That is the user's chosen semantics, including which groups exist. einfold keeps the filter at the operand, never moves it back above the join, and exploits the smaller support: EinFold measures the lower density (section 10.4), and per-chunk statistics skip chunks the filter rules out.
- *Inferred.* einfold drops zero rows on its own only when all three of these hold, each of which it can check:
  1. the zero factor feeds only a sum of products;
  2. every other factor in those products is finite (an Exact fact), because in floating point `0 × NaN` and `0 × ∞` are `NaN`;
  3. whether a group exists is not visible downstream. Either another operand guarantees every group, or, in program mode, the plan fills missing groups in later, as ddx's gradient step does with its left join that writes 0.

  (Skipping zeros can also flip the sign of a result that is itself zero, from `−0.0` to `0.0`.) ReLU is the motivating case: `GREATEST(z, 0)` produces exact zeros, and its derivative is exactly zero in the same places, so the inferred filter `z > 0` applies to the forward and backward passes alike.

Dropping values that are merely *small* is an approximation, not an exact rewrite. It stays the user's to write (principle 8).

**When.** When facts predict a high miss rate. Dense arrays whose support is complete have no unmatched rows, so pruning is skipped for them. It matters most for sparse workloads such as graphs and triplestores (databases of subject–predicate–object facts).

**Correctness.** A semi-join only removes rows that would not join, and never duplicates rows, so inner-join results and bag semantics are unchanged. Exact zeros are covered by the three conditions above.

**Realization.** Relational form: semi-joins (`WHERE EXISTS` or `IN`), and filters such as `v <> 0` placed at the operand. Reference executor: a Bloom filter on the keys of the input held in memory, applied while scanning the other input.

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

**Protecting the plan from the host optimizer.** A decomposed einsum is a deep tree of small queries, and some host optimizers cost more than they save on it. Blacher et al. measured a query encoding a satisfiability problem with 952 clauses. HyPer, a fast research database from TU Munich, spent 0.87 s planning it and 0.08 s executing it. DuckDB had not finished planning after five hours. With its optimizer disabled, DuckDB planned in 0.20 s and ran in 0.97 s. So the output must stop the host from flattening or reordering the tree again, and from spending minutes planning it. The target profile picks the mechanism.

Spike S10 ([`spikes/s10-plan-protection`](spikes/s10-plan-protection/README.md)) reproduced the satisfiability example on current hosts, with banded 3-SAT formulas of 91 and 218 clauses:

- *Both hosts keep a contraction order written as CTEs.* Every decomposed plan kept one aggregation per step, because neither optimizer moves joins across a `GROUP BY`. The danger is planning cost, not reordering.
- *DuckDB plans inlined CTEs very slowly:* 56 s at 91 steps, and over 3 minutes at 218. Written `AS MATERIALIZED`, the same CTEs planned in 0.09–0.39 s and ran in 0.06–0.22 s. So DuckDB's mechanism is `MATERIALIZED` CTEs.
- *DataFusion plans plain CTEs quickly* (0.41 s at 218 steps). Its mechanism is plain CTEs, or a physical plan in the in-engine mode.
- *The flat form fails on both.* One query joining every operand ran past 3 minutes at 218 clauses on both hosts, and DuckDB spilled 16 GB doing it. einfold never emits the flat form for more than a few operands.

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

**Tiles in the algebra.** Following Cubed (section 8.2), a tiled plan is built from two operations, both expressible as e-graph terms:

| Cubed | einfold |
|---|---|
| A chunked array | An operand whose dimensions are split into block and within-block parts |
| `blockwise`, with a function from each output chunk to the input chunks it needs | An einsum whose block dimensions are the outer loop (one task per combination) and whose within-block dimensions are the kernel. The input chunks a task needs follow from the einsum's dimensions. |
| Aligning inputs' chunks before a `blockwise` | Shared dimensions must be split the same way |
| `rechunk` | `Retile(d, c → t)`: the same values, at the IO cost counted in step 1 above |
| A reduction as a tree of `partial_reduce` steps | The sum over a block dimension, as a fixed-order combine of partial aggregates (section 8.3) |
| Projected memory per task, checked at planning time | An e-graph analysis: memory per task is the product of within-block extents over the operands the task touches, with a margin for decoding and encoding chunks (Cubed's rule of thumb is about 4× a chunk). Plans over the budget get infinite cost, so extraction cannot pick them. |
| Fusing operations, with limits on how many arrays and blocks one task reads | Merging adjacent contraction-tree nodes that share a tiling into one task, under the same kind of limits |
| Reading a task's chunks all at once, or one at a time | Hash versus streaming execution (sections 8.4 and 10.2) |

Tile sizes are numbers, so the space of tilings is infinite. As with contraction order, the planner above proposes a few candidates (the storage chunk sizes, their least common multiples, rechunker's consolidated sizes), and the e-graph holds them as alternatives. Extraction then chooses among them, with the memory budget as a hard limit.

**Evidence** (spike S20, [`spikes/s20-tiles`](spikes/s20-tiles/README.md)). Matrix multiplication of two 20,000 × 20,000 matrices, with retiling, blockwise products and rounds of partial sums as egglog terms, and Cubed's projected-memory formulas in the cost model:

- **Reproduces Cubed.** Restricted to the storage tilings, extraction returned Cubed's own plan: the same operations, task counts and projected memory, to the megabyte.
- **The budget is a hard limit.** In 200 random configurations, no extracted plan exceeded its budget, and every extracted cost equaled the brute-force minimum over the same space. Where no plan fits (512 MB storage chunks under a 1 GB budget), extraction says so at planning time. Cubed builds a plan that fails when run.
- **Finds cheaper plans.** With proposed tile sizes, extraction found plans 40–50% cheaper than Cubed's, mostly by rechunking the inputs once rather than writing one full-size partial product per block of `k`.
- **Cheap.** At most 12,000 tuples and 16 ms per case.

Two design rules come from it:

- **A tiling is part of an e-class's identity.** Each e-class is one array *at one tiling*, and `Retile` terms link the tilings. If different tilings of the same values shared an e-class, extraction could choose a child tiling its parent cannot use. This is the "physical property" pattern of Cascades-style optimizers (which plan for properties such as sort order alongside the plan itself), written as terms.
- **The proposer decides what is reachable.** Every extracted plan used only proposed sizes. Proposing too few sizes is the main way this design can miss a good plan.

**Why it fits XQL.** When `t(s)` matches the Zarr chunk size, partitions map one-to-one onto chunks. Each partition reads its own chunks, and xarray-sql's partition pruning applies.

**Prior work.** rechunker; cotengra slicing; Cubed.

### 9.5 Share work across einsums

- **Shared scans.** ddx's two gradient contractions from section 3, `X̄[n,d] = Σ_h Ȳ[n,h]·W[d,h]` and `W̄[d,h] = Σ_n X[n,d]·Ȳ[n,h]`, both read `Ȳ`. Both can stream `Ȳ` against a hash table: one on `W` keyed by `h`, one on `X` keyed by `n`. One scan of `Ȳ` then feeds both results. Each row `(n, h, ȳ)` adds to row `n` of `X̄` through the `W` table, and to column `h` of `W̄` through the `X` table. In ddx these are two separate steps that each read the stored `Ȳ` step, so an optimizer rule seeing one plan at a time cannot share the scan. Program mode (section 7.3) can: it sees every step of the program. In general, group the einsums in one plan, or one program, that share an operand, and stream the shared operand. This is multiple-query optimization (Sellis, 1988), the classic technique of sharing work among queries run together, applied to einsums.
- **Common subexpressions.** Put every node of every contraction tree in a canonical form, hash it, and compute identical nodes once (Deeds et al., §5.4). In egglog this comes for free: the e-graph stores each distinct subexpression once, so einsums placed in the same e-graph, including all the steps of a program, share them automatically.
- **Realization.** Reference executor: an `EinFoldExec` with several outputs. Relational form: the shared operand is emitted once as a CTE that both einsums reference. Whether the host then scans it once is recorded in the target profile.

## 10. Physical realization, in detail

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

- *Group existence (`reached`).* A pre-aggregated group of `B` exists exactly when some row of `B` has that key, so the join matches in both forms, for the same output groups.
- *`value`.*
  - If `a` is NULL, every `a·bⱼ` is NULL, and so is `a·Σbⱼ`. Both forms skip the contribution.
  - If all `bⱼ` are NULL, `Σbⱼ` is NULL, so `a·Σbⱼ` is NULL. Every `a·bⱼ` is also NULL. Both forms skip.
  - Otherwise both forms add the same non-NULL terms.

The rule is therefore exact in SQL semantics, up to floating-point rounding. For integer and `DECIMAL` values it is applied only when no intermediate value can overflow (section 8.6).

**When it helps.** Yan and Larson skip pre-aggregation when the grouping columns form a key, because then nothing gets smaller. The einsum version of that test: a node gains only if it sums out at least one dimension. For matrix multiplication `nd,dh->nh`, the shared dimension `d` is summed only after the join, and nothing is private, so eager aggregation changes nothing. Matrix multiplication needs EinFold. Eager aggregation matters for chains of three or more operands, for ddx's multi-way gradient contractions, and for operands with private dimensions (marginals, traces, weighted means).

**Relation to the groupjoin.** Moerkotte and Neumann (2011) defined the groupjoin, which fuses a join and the group-by after it into one hash table, and gave conditions under which it is correct. Those conditions require each output group to come from exactly one row of the input that is held in the hash table: the grouping columns must determine that row's identity, written `G₁, G₂⁺ → TID(e₁)`, where TID is a row identifier. That holds for joins on a key and a foreign key. It also holds for einsum steps whose output dimensions all belong to one operand, for example `ij,i->i`. It does not hold for matrix multiplication, where an output group `(n, h)` combines rows from both inputs. So matrix multiplication is not a classical groupjoin. It is Gustavson's algorithm (section 10.2).

**Prior work.** Yan and Larson (eager and lazy aggregation); Chaudhuri and Shim (1994), who added group-by to cost-based query optimization; Moerkotte and Neumann (groupjoin); Blacher et al. (CTE decomposition); FAQ.

### 10.2 The extension form: EinFold

EinFold is the fused join-and-sum operator. It is what the `Fold` relation asks a host to run, and what `EinFoldExec` implements. For two-operand contractions it is the highest-value piece, and the only fix for problem 1.

**Idea.** Fuse the aggregate into the hash join so the `N·D·H` join rows never exist: each joined row updates its group's partial aggregate (section 8.3) directly. This works for every fold, since it only needs partial states to merge. For matrices this is Gustavson's row-by-row sparse matrix multiply (Gustavson, 1978), which sparse BLAS libraries still use.

**Dimension roles at a node** `A ⊗ B → K`, with shared dimensions `S`:

- `Kₛ = K ∩ S`: batch dimensions, shared and kept (such as `b` in the batched matrix product `bik,bkj->bij`);
- `F_A = (K ∩ dims(A)) \ S` and `F_B = (K ∩ dims(B)) \ S`: free dimensions from each side;
- `S \ K`: contracted dimensions.

#### Hash algorithm (EinFold's hash algorithm)

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
emit:   each group with reached = true
```

The **build** step loads `B` into a hash table. The **probe** step streams `A` and looks each row up in that table. The hash table is `B` compressed along its contracted dimensions, which is what Gustavson's row-wise storage of `B` is. The probe loop is his row loop. The update follows section 8.3: set `reached` (Gustavson's `xb` and `JC`) before testing the row's value for NULL. Otherwise a group reached only by NULLs would vanish instead of coming out NULL (or 0, for `COUNT`).

**Streaming.** If `A` arrives grouped by `(Kₛ, F_A)` (section 8.4), all contributions to one output row arrive together. Then only one row's state is live: Gustavson's `x`, `JC`, and `xb`, keyed by `F_B`. Use his arrays directly when `F_B` has a known extent `r`, and a small hash map otherwise. When the `(Kₛ, F_A)` key changes, emit the row. With the multiple-switch array there is nothing to reset.

- Memory drops from the size of the output to the size of the hash table plus one output row.
- Rows come out grouped by key, so the next node can stream too.
- A Zarr scan arrives grouped this way within a chunk when `(Kₛ, F_A)` are the slowest-varying dimensions of the chunk's memory order. When the input is not grouped, a counting sort can group it in `O(rows + extent)`.

**Choosing which input to stream.** Gustavson notes an asymmetry. Streaming `A` wastes a step for each entry of `A` whose row in `B` is empty, so its cost depends on `N_A`. Streaming `B` depends on `N_B`. The planner streams the input with fewer expected misses and builds the hash table on the other, as long as that table fits in memory (a Bound, section 8.1).

**Symbolic–numeric split** (section 8.5). When the support is fixed and only values change, run the symbolic pass once to compute the output's structure. Later runs then do only the numeric pass, with no hashing and no "already touched?" test. The reference executor caches the symbolic result next to ddx's cached physical plan.

**Accumulator precision.** Gustavson recommends accumulating `x` in higher precision than the inputs and rounding once, when the row is emitted (Gustavson, 1978, §3.5). einfold's precision levels (section 8.6) build on that advice.

#### Dense and block-sparse algorithms

- **Dense.** When both operands are dense over `S` and their free dimensions, with Exact coordinate maps that agree and unique coordinate tuples, skip hashing. View each tile through its layout and call a GEMM, batched over `Kₛ`. The layout's strides decide whether an input must be treated as transposed (section 8.4).
- **Block-sparse.** When an operand's support is known by tile (for example, from missing Zarr chunks or a mask), run Gustavson's algorithm over tiles instead of rows. Hash the present tiles of `B` by their tile coordinates along `S`. For each present tile of `A`, call a dense GEMM against each matching tile of `B`. A tile has one of three states: absent (skipped), fully present (a dense kernel), or partly present (a masked kernel, which applies the mask within the tile). For a causal mask, the tiles above the diagonal are absent, those below are fully present, and only the diagonal tiles need masking, which is how FlashAttention tiles causal attention (Dao et al., 2022). Skipped tiles still contribute their partial aggregate, as the fill-value rules in section 8.1 require.
- **Choosing.** Dense when both sides are dense with Exact extents. Block-sparse when support is known by tile. Otherwise hash. Spike S11 ([`spikes/s11-thresholds`](spikes/s11-thresholds/README.md)) measured the thresholds on CPU, for matrix products at uniform density:
  - dense GEMM overtakes Gustavson's algorithm at about **20% density**, at every size tested;
  - Gustavson's array-indexed algorithm is 2.5–17× faster than a hash join followed by a hash aggregate, so the hash-map variant is only a fallback for free dimensions without a known extent;
  - knowing which output groups exist costs the dense algorithm a second, 0/1 GEMM, about 2×. An Exact fact that both inputs are complete removes it.

  Thresholds can change while running (section 10.4).

#### Correctness

- *NULLs and group existence:* section 8.3.
- *Missing tiles:* section 8.1.
- *Duplicate coordinate tuples.* The hash algorithm adds them up (bag semantics). The dense algorithms require unique tuples, which an Exact layout guarantees.
- *Determinism.* Accumulation order is the order of the streamed input times the order of each hash-table entry's list. It is deterministic if both inputs arrive in a deterministic order and the lists keep insertion order. When determinism is requested, parallel partitions are combined in a fixed order, or with an order-independent accumulator (section 8.6).

**In the relational form.** The relational form cannot express EinFold. But on hosts whose profile says they fuse join and aggregate (such as `gpudb`, for some query shapes), the plain pairwise SQL already avoids materializing join rows.

**Generality.** Gustavson also uses the algorithm for a "pseudo-multiplication": assembling the large sparse matrix of a finite-element simulation from small per-element matrices, where the "product" looks up an entry of an element matrix. Any operation with the same structure works. That supports carrying a semiring in the fold IR (section 14).

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
  - switch from dense to sparse when it falls below a threshold. They found 5% empirically. Spike S11 puts einfold's crossover near 20% against Gustavson's algorithm, and near 5–10% against a hash join and aggregate, which matches theirs;
  - measure only before expensive contractions, and stop measuring once density exceeds 95%.
- **Beyond the paper.** Staudt et al. only switch from dense to sparse. EinFold has both algorithms, so einfold can switch both ways, and can use the block-sparse algorithm for nodes that mix dense and sparse operands. Their other stated limitation is a fixed contraction order. With Measured facts and Bounds, the planner can re-plan the rest of the tree while it runs.
- **Scope.** Inside an `EinFoldExec` (or a host's `Fold` relation) that runs a whole contraction tree. The relational form is unaffected, since hosts already run it as a sparse form.
- **Determinism.** Switching decisions depend only on the data, so the same data gives the same decisions and the same bits.

### 10.5 Reduction at the source

- **Idea.** When a dimension is summed within a single operand, the reader can compute each storage tile's partial aggregate with a dense kernel on the decoded chunk, before the chunk is flattened into rows. The engine then only combines partial aggregates. This follows directly from principle 6.
- **Mechanism.** Engines already split aggregation into a partial phase per partition and a final merge. Spike S12 ([`spikes/s12-aggregate-pushdown`](spikes/s12-aggregate-pushdown/README.md)) worked out the route on each host:
  - *DataFusion.* `AggregateExec` runs in `Partial` and then `FinalPartitioned` mode, with a hash repartition between them. `TableProvider` has no aggregate hook, so the reader installs a physical optimizer rule that replaces only the `Partial` aggregate and its scan with a scan that emits partial aggregates, one per chunk. DataFusion's repartition and final phase stay, with their parallelism and spilling. The scan emits DataFusion's partial-state schema, which each aggregate function defines through its `state_fields`. For example, `SUM(double)` is one nullable `[sum]`, whose NULL coincides with "no non-NULL value yet" in section 8.3, and `AVG` is `[count, sum]`.
  - *DuckDB.* Its extension C API, which duckdb-zarr uses through Rust, offers projection pushdown only: no filters and no aggregates. So einfold's SQL-to-SQL mode rewrites the statement to call a reader-provided aggregating table function, such as a `read_zarr_reduce`, which returns plain partial columns that ordinary SQL then combines. This is the relational form of partial aggregates (section 8.3), and it works on any engine that can call a table function.
- **Precedent.** zarr-datafusion's `ZarrAggregateExec` already computes `SUM`, `COUNT`, `MIN`, `MAX` and `AVG` itself when they sit directly over its scan. It replaces the whole aggregate, though, not only its partial phase. It folds rows rather than whole chunks. And it accumulates in `f64`, which loses exactness for integers beyond 2⁵³ and breaks the exactness invariant (section 8.6). einfold's version should replace the partial phase, reduce chunks with dense kernels, and accumulate integers exactly.
- **Scope.** Contractions local to one operand, including those that variable separation (section 9.1) isolates. Also products of variables in the same Zarr group that share a chunk grid: an element-wise product plus a private sum is local to each chunk.
- **Correctness.** Partial aggregates carry group existence and the aggregate's state (section 8.3), and are combined in chunk order.
- **Why it matters.** For single-array reductions (time means, spatial averages, applying regridding weights), this attacks problem 3 at its source: the data is reduced before it is ever flattened.

### 10.6 Worst-case optimal joins (future)

- For cyclic sparse einsums such as triangle counting (`ij,jk,ki->`), every pairwise contraction order can create intermediates far larger than the result. Worst-case optimal join algorithms avoid this by joining all inputs at once, one variable at a time (Ngo et al., 2018).
- Free Join (Wang et al., 2023) unifies them with ordinary binary hash joins, using one data structure for both hash tables and tries, and matches or beats both kinds on standard benchmarks. Galley's choice of loop order plays the same role for sparse tensors (Deeds et al., §5.1).
- This would be an n-way node in the contraction tree, used only for cyclic sparse subexpressions.

### 10.7 Considered and not pursued

- **Packing dense dimensions into array columns** (for example Arrow's fixed-size list type). It breaks principle 2, the XQL logical model. Dense speed belongs to EinFold's dense algorithm and to facts.
- **Incremental maintenance across Icechunk versions.** Out of scope for now.
- **Choosing approximations for the user**, such as dropping small values, sampling a vocabulary, or skipping updates. These change results, so users write them in SQL (principle 8). The NanoGPT demo (see [`demos.md`](demos.md)) shows such tricks as small SQL diffs.

## 12. Integration, in detail

**What stays in ddx:**

- Caching each training step's physical plan. Plans don't change between steps, so they need planning once. Once contractions are fast, the roughly 3 ms of planning per step would otherwise dominate. einfold's program-level cache (section 8.5) complements this by caching its own rewrite decisions across steps.
- Forward-mode differentiation inside long chains of row-by-row operations. Forward mode carries derivatives alongside values instead of working backward, which keeps plans growing linearly, not quadratically, in the chain's length.
- Emitting gradient contributions in a canonical, einsum-shaped form, so detection recognizes them cleanly. Later, ddx could emit the `Fold` relation directly, as an option in its `ddx_ad::Options` settings.
- Simplifying scalar derivative expressions, and choices specific to automatic differentiation such as saving versus recomputing a region (section 7.6).

## 13. Spikes and literature, in detail

A spike is a short, time-boxed experiment that answers one design question. Spike code and reports live in [`docs/spikes/`](spikes/).

Status as of design v12: 17 of 20 spikes done, and three blocked on access. The index [`docs/spikes/README.md`](spikes/README.md) lists every spike with its report.

### 13.1 Facts

| Spike | Question | Outcome |
|---|---|---|
| **S1** Zarr → table → plan; **S13** variable separation; **S14** fill values. Done: [`s01-readers`](spikes/s01-readers/README.md) | What Zarr metadata reaches the plan through each reader? Are a variable's dimensions exposed? How are fill values emitted? | No reader writes facts into Arrow metadata. Readers disagree on missing data (NULL vs. NaN), lower-dimensional variables, statistics, filter pushdown and time types. Facts come from per-reader providers (section 8.1). |
| **S2** Carrier survival; **S3** propagation rules. Done: [`s02-carrier`](spikes/s02-carrier/README.md) | Does Arrow metadata survive plans and Substrait? Do the layout rules hold? | Metadata is lost in DuckDB and Substrait, and kept too eagerly in DataFusion, so facts use a side channel. The layout rules predict coordinates correctly, but row order is a separate, host-specific fact (sections 8.1 and 8.4). |
| **S4** Coordinate maps. Done: [`s04-coords`](spikes/s04-coords/README.md) | How often are ERA5 (ECMWF's hourly global reanalysis) and CMIP6 (the latest coordinated climate-model runs) coordinates affine? | Affine is common but must be bitwise; monthly times are calendar maps; sorted tables are the general exact form (sections 6.2 and 8.1). |

### 13.2 Hosts

| Spike | Question | Outcome |
|---|---|---|
| **S5** gpudb shapes. Done: [`s05-gpudb`](spikes/s05-gpudb/README.md) | Which relational-form shapes does `gpudb` run on GPU? How does its float-sum rule interact with einfold? | On a T4, 1 of 13 shapes ran on GPU (a `BIGINT` reduction, 4.3×). gpudb declines `DOUBLE` sums, expanding joins, subquery operands and windows, so einfold's float workloads get no GPU help from it today. |
| **S6** GQE Substrait. **Blocked** | Does GQE accept, reject, or ignore extension relations? Which operators run on GPU? | Needs access to NVIDIA's GPU Query Engine (offered through build.nvidia.com) and an NVIDIA GPU it supports. Not attempted. |
| **S7** DataFusion unparser. Done: [`s07-unparser`](spikes/s07-unparser/README.md) | Can DataFusion's `Unparser` hand rewritten plans to DuckDB? | Yes, from unoptimized plans (34 of 34 correct). Not from optimized plans: 31 of 34 rejected, and one silently wrong (section 7.3). |
| **S8** Cost of deterministic sums. Done: [`s08-deterministic-sums`](spikes/s08-deterministic-sums/README.md) | What do reproducible sums cost on CPU and GPU? | A binned sum is deterministic and accurate, at 3–4.4× a plain sum on CPU and 3.5× on GPU (section 8.6). |
| **S9** Zax-SQL. **Blocked** | Which rewritten shapes does Zax-SQL run well? Does its Icechunk metadata appear in `information_schema`? | Needs an Earthmover account with Zax-SQL access. |
| **S10** Plan protection. Done: [`s10-plan-protection`](spikes/s10-plan-protection/README.md) | Which mechanism protects the contraction tree on each host, and what does planning cost? | Both hosts keep CTE orders. DuckDB needs `MATERIALIZED` CTEs to plan in under a second; DataFusion plans plain CTEs quickly (section 9.3). |
| **S11** Algorithm thresholds. Done: [`s11-thresholds`](spikes/s11-thresholds/README.md) | When does EinFold's dense algorithm win? | Above about 20% density against Gustavson's algorithm, and 5–10% against a hash join and aggregate (sections 10.2 and 10.4). |
| **S12** Reader aggregate pushdown. Done: [`s12-aggregate-pushdown`](spikes/s12-aggregate-pushdown/README.md) | How does a reader take over the partial phase of an aggregation? | In DataFusion, through a physical optimizer rule that replaces the `Partial` aggregate and emits its state schema. In DuckDB, through a SQL rewrite to a reader table function (section 10.5). |
| **S15** Sirius. **Blocked** | Does DuckDB's optimizer reorder einfold's tree before Sirius sees it? Which shapes stay on GPU? Are its float sums deterministic? Can a Substrait extension relation reach it? Does `pin_table` keep ddx's weights resident? | Needs an NVIDIA GPU of compute capability 7.5 or newer (per gpudb's comparison table), and a long build against libcudf. The local GTX 1080 Ti (6.1) is too old. A Colab T4 or a cloud VM would do. |
| **S19** Float sums in hosts. Done: [`s19-float-sums`](spikes/s19-float-sums/README.md) | Are float sums repeatable in SQL engines and JAX? | No mainstream SQL engine guarantees it, so determinism is a setting on hosts (section 8.6). |

### 13.3 Optimizer engine

| Spike | Question | Outcome |
|---|---|---|
| **S16** egglog as the rewrite engine. Done: [`s16-egglog`](spikes/s16-egglog/README.md) | Can egglog find einfold's rewrites? | Yes, in the hybrid design (section 7.6). |
| **S17** n-ary sum-product nodes. Done: [`s17-nary`](spikes/s17-nary/README.md) | Does one node per region fix e-graph growth? | Yes: one iteration and linear size up to 1000 operands. Distributivity still needs a guard (section 7.6). |
| **S18** Planning time. Done: [`s18-planning-time`](spikes/s18-planning-time/README.md) | How long does egglog take on real plans? | 0.5–1.2 ms per query with rules preloaded, less than the hosts' own planning (section 7.6). |
| **S20** Tiles in egglog. Done: [`s20-tiles`](spikes/s20-tiles/README.md) | Can tilings and a memory budget live in the e-graph? | Yes. It reproduces Cubed's plans exactly, never exceeds the budget, and finds cheaper plans (section 9.4). |

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

Each rule follows from CuTe's layout algebra (section 8.2). Spike S3 confirmed every rule's prediction of which coordinates come out, on DataFusion and DuckDB. The rules describe the coordinate map only: row order is a separate fact that holds only where the host promises it (section 8.4).

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

## 18. References

Project and systems:

- ddx, and its notes on fast linear algebra: https://github.com/xqlsystems/ddx, https://github.com/xqlsystems/ddx/blob/main/docs/fast-linalg-notes.md
- XQL Systems: https://xql.systems
- xarray-sql: https://github.com/alxmrs/xarray-sql
- duckdb-zarr: https://github.com/xqlsystems/duckdb-zarr
- zarr-datafusion: https://github.com/stratoscale-io/zarr-datafusion (formerly jayendra13/zarr-datafusion)
- Zax-SQL: https://www.earthmover.io/blog/compute-roadmap, https://docs.earthmover.io/compute/sql
- NVIDIA GPU Query Engine: https://build.nvidia.com/nvidia/gpu-query-engine
- gpudb DuckDB extension: https://github.com/duckdb/community-extensions/blob/main/extensions/gpudb/description.yml, https://github.com/singhpratech/duckdbgpumetaldbram
- Sirius: https://github.com/sirius-db/sirius
- Substrait: https://substrait.io
- Zarr conventions (`spatial`, `missing_value`, `dependent-arrays`, and the conventions specification): https://github.com/zarr-conventions
- Proposed XQL Systems `layout:` convention: [`layout-convention.md`](layout-convention.md)
- Target demos: [`demos.md`](demos.md)

Papers:

- Abo Khamis, M., Ngo, H. Q., Rudra, A. (2016). FAQ: Questions Asked Frequently. *PODS 2016.*
- Blacher, M., Klaus, J., Staudt, C., Laue, S., Leis, V., Giesen, J. (2023). Efficient and Portable Einstein Summation in SQL. *Proc. ACM Manag. Data* 1(2), Article 121. https://doi.org/10.1145/3589266
- Blacher, M., Staudt, C., Klaus, J., Wenig, M., Merk, N., Breuer, A., Engel, M., Laue, S., Giesen, J. (2024). Einsum Benchmark: Enabling the Development of Next-Generation Tensor Execution Engines. *NeurIPS 2024, Datasets and Benchmarks Track.* https://proceedings.neurips.cc/paper_files/paper/2024/file/b1bbfdb9197bfc819a52c34dce493f85-Paper-Datasets_and_Benchmarks_Track.pdf
- Chaudhuri, S., Shim, K. (1994). Including Group-By in Query Optimization. *VLDB 1994*, 354–366.
- Chen, J., Huang, Y., Wang, M., Salihoglu, S., Salem, K. (2023). Accurate Summary-based Cardinality Estimation Through the Lens of Cardinality Estimation Graphs. *SIGMOD Record* 52(1), 94–102. https://doi.org/10.1145/3604437.3604458
- Deeds, K., Ahrens, W., Balazinska, M., Suciu, D. (2025). Galley: Modern Query Optimization for Sparse Tensor Programs. *Proc. ACM Manag. Data* 3(3), Article 164. https://doi.org/10.1145/3725301 (arXiv:2408.14706)
- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *NeurIPS 2022.* https://arxiv.org/abs/2205.14135
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
- Cubed: https://cubed-dev.github.io/cubed/, https://github.com/cubed-dev/cubed (design notes in `cubed/primitive/DESIGN.md`)
- Zarr v3 core specification (chunk grids, fill value, codecs, sharding): https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html
