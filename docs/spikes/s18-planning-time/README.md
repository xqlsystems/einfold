<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S18: egglog's planning time

**Question.** How long do egglog's saturation and extraction take on ddx's training plans and on TPC-H (the standard decision-support SQL benchmark), compared with the host's own planner?

**Answer.** Planning a query's sum-product regions takes **0.5–1.2 ms** when einfold keeps an e-graph with its rules already loaded. That is less than DataFusion's or DuckDB's own planning time for the same TPC-H query (0.7–4.6 ms). Starting cold adds about 1.5 ms per query, almost all of it parsing the rules, so einfold should load its rules once per session, not once per query.

Date: 2026-10-06. egglog 3.0.0, DataFusion 54 (Python), DuckDB 1.5.6, on a 12-core Linux machine.

## Method

**Host baseline.** [`host_planning.py`](host_planning.py) generates TPC-H at scale factor 0.01 with DuckDB's `tpch` extension, and registers the same tables in DataFusion. It then times planning only, with no execution, for all 22 queries from DataFusion's benchmark suite, taking the median of 25 runs:

- DataFusion: SQL to optimized physical plan (`ctx.sql(q).execution_plan()`);
- DuckDB: `EXPLAIN q` (parse, bind, optimize, physical plan).

**einfold's cost.** [`src/main.rs`](src/main.rs) uses S17's n-ary rules ([`../s17-nary/nary.egg`](../s17-nary/nary.egg)) and its cost model, with the greedy contraction planner inside extraction. Leaves are sized by their row counts. Each case is planned in two ways, median of 25 runs:

- **cold:** a new e-graph, the rules parsed, the query's terms added, saturation, and extraction of every region;
- **warm:** the same, starting from a clone of an e-graph that already holds the rules, as a long-lived einfold session would.

The cases are:

- **TPC-H's join-and-`SUM` queries** (Q1, Q3, Q5, Q7, Q8, Q9, Q10) at scale factor 1, each written as one sum-product region. Each table is an operand over its key columns, so for example Q5 has six operands over `custkey`, `orderkey`, `suppkey`, `nationkey` and `regionkey`. Q8 is TPC-H's largest join, at eight tables.
- **ddx's training plans:** the five einsums of a two-layer MLP's forward and backward pass, and the six einsums of attention's forward and backward pass, each planned together in one e-graph, as in one training step. Also S16's three-operand gradient.

Run with `python host_planning.py <queries>` and `cargo run --release`.

## Results

Host planning, TPC-H (ms):

| | DataFusion | DuckDB |
|---|---|---|
| Median over 22 queries | 2.33 | 1.73 |
| Fastest (Q6) | 0.85 | 0.71 |
| Slowest | 4.56 (Q8) | 3.10 (Q2) |
| Q3 / Q5 / Q8 / Q10 | 2.33 / 2.91 / 4.56 / 3.31 | 1.33 / 2.22 / 3.00 / 1.72 |

einfold's egglog planning (ms):

| Case | Regions | Operands | E-graph tuples | Cold: loading rules | Cold: total | Warm: cloning | Warm: total |
|---|---|---|---|---|---|---|---|
| TPC-H Q1 | 1 | 1 | 5 | 1.52 | 1.93 | 0.14 | 0.52 |
| TPC-H Q3 | 1 | 3 | 9 | 1.48 | 1.98 | 0.14 | 0.63 |
| TPC-H Q5 | 1 | 6 | 15 | 1.48 | 2.23 | 0.14 | 0.87 |
| TPC-H Q7 | 1 | 6 | 15 | 1.45 | 2.17 | 0.14 | 0.84 |
| TPC-H Q8 | 1 | 8 | 19 | 1.47 | 2.36 | 0.14 | 1.00 |
| TPC-H Q9 | 1 | 6 | 15 | 1.51 | 2.31 | 0.14 | 0.85 |
| TPC-H Q10 | 1 | 4 | 11 | 1.48 | 2.05 | 0.14 | 0.71 |
| ddx MLP, forward and backward | 5 | 10 | 27 | 1.47 | 2.47 | 0.14 | 1.12 |
| ddx attention, forward and backward | 6 | 12 | 30 | 1.47 | 2.53 | 0.15 | 1.19 |
| ddx MLP gradient as one region | 1 | 3 | 9 | 1.46 | 1.97 | 0.14 | 0.62 |

For larger regions, spike S17 measured the same rules on 10 to 1000 operands: saturation took 2 to 200 ms, and extraction, dominated by the greedy planner, 1 ms to 26 s.

## Findings

1. **Warm, egglog costs less than the host's own planning.** For TPC-H's sum-product queries, it adds 0.5–1.0 ms to a host that spends 1.3–4.6 ms planning the same query: roughly 20–45% more planning time. That time is spent only on queries in which detection finds a region; detection's own cost is not measured here.
2. **Load the rules once.** Parsing the rule file costs about 1.5 ms, three quarters of a cold plan. An einfold session should build one e-graph with the rules and clone it per query (0.14 ms). egglog 3.0's `EGraph` implements `Clone`, which makes this straightforward.
3. **ddx's per-step planning is covered.** A whole training step's einsums plan in about 1.2 ms warm, less than the roughly 3 ms per step that ddx already caches away (design doc §12). With einfold's program-level cache (§8.5), even that is paid once per program, not once per step.
4. **The cost is in extraction, not saturation, as regions grow.** These regions are small: at most 30 tuples. Spike S17 shows that the n-ary form keeps saturation cheap at hundreds of operands, and that the planner inside extraction is what grows. For einsums with thousands of operands, the planner needs the same care as cotengra's: incremental greedy search, or a time budget.

## Limitations

- TPC-H regions were written by hand from each query's joins and `SUM`, not produced by einfold's detection, which doesn't exist yet. Detection time isn't measured.
- The host baseline goes through Python bindings, which add a little per-call overhead to DataFusion's and DuckDB's numbers. That only makes the comparison less favorable to einfold.
- Only the rules from S16 and S17 were loaded. Pruning, tiling (S20) and sharing rules will add parsing time to a cold start, but not to a warm one, and add saturation work only where they match.
