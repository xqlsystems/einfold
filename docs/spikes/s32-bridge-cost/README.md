<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S32: where einfold's Python bridge spends TPC-H's time

**Question.** M3's gate asks for no slowdown on TPC-H, the standard benchmark of 22 analytic queries over orders and their line items. einfold changes none of those plans, yet einfold 0.2.0 from Python takes 1.06× DataFusion's time at scale factor 1 (SF1). Where does the extra time go, and how much of it can einfold remove on its own?

**Answer.** About 115 ms over the 22 queries, from five causes. Three are einfold's to fix, and fixing them halves the excess:

| | ms over 22 queries | einfold can fix |
|---|---:|---|
| the FFI's round trip: datafusion-python's protobuf of the logical plan, a foreign session built per query, and the physical plan's reference back | ~29 | no: upstream (#58) |
| execution with a fresh `RuntimeEnv` per query, so every Parquet footer is read and parsed again, during planning and during execution | ~25–35 | **yes**, by keeping one per session |
| einfold's library built without link-time optimization (LTO), unlike datafusion-python's | ~30 | **yes**, by building as datafusion-python does |
| one conversion of the session's task context per query, every function included, even when no table is copied | ~5 | **yes**, by converting it only when a table is copied |
| large blocks freed back to the operating system as a query runs, such as q21's 3.8 M-row hash table | ~10 | partly: a trade against peak memory (#119) |

With the three fixes, TPC-H takes 1.023–1.040× DataFusion's time over three alternating runs, down from 1.038–1.061× for 0.2.0. What's left is the FFI's round trip, 1.3 ms per query, which no change in einfold removes, and the large frees.

So "no slowdown" on TPC-H can't be reached by einfold alone while every query crosses datafusion-python's FFI. The FFI costs a fixed 1.3 ms per query, and TPC-H at SF1 has queries of 20–30 ms. The FFI's cost is upstream, and #58 now has its measurement.

Date: 2026-11-01. Intel Core i7-8700 (6 cores, 12 threads), 15 GB RAM, otherwise idle. datafusion 55.0.0 (DataFusion's Python bindings), einfold 0.2.0 from PyPI and built from `python-v0.2.0` with [`spike.patch`](spike.patch), Python 3.12. `OPENBLAS_NUM_THREADS=1` for the later runs: PyArrow imports NumPy, whose OpenBLAS threads otherwise spin through a profile.

## Method

[`generate.py`](generate.py) writes TPC-H at SF1 as Parquet, and its 22 queries, with DuckDB's tpch extension. Each script registers the Parquet files in two `SessionContext`s, one with einfold installed and one without, and times the same queries in each, as medians:

- [`phases.py`](phases.py): each query's planning (`DataFrame.execution_plan()`, median of 9) and the whole query (`to_arrow_table()`, median of 7), after one warm-up;
- [`hostphys.py`](hostphys.py): DataFusion's own physical planning, as `execution_plan()` less `optimized_logical_plan()`, against einfold's part of planning;
- [`plantime.py`](plantime.py), with `EINFOLD_TIMING=1` and the patch: each step of einfold's planner, in microseconds;
- [`execonly.py`](execonly.py): execution alone, planning outside the timer;
- [`analyze.py`](analyze.py): `EXPLAIN ANALYZE` metrics per operator, einfold on and off.

Run-to-run noise is about ±25 ms of the 2.3 s total, so each comparison alternates its builds, two or three times.

## Results

**Plans are the same.** `EXPLAIN` gives the same physical plan with einfold as without, for every query: einfold finds no fold in TPC-H, and plans each query with DataFusion's own planner and rules.

**Planning: +54 ms, of which einfold's own steps are 7 ms.** DataFusion alone plans the 22 queries in 68 ms, of which physical planning is 37 ms. With einfold 0.2.0, planning takes 122 ms. einfold's planner, step by step (`plantime.py`):

| step | 0.2.0 | patched |
|---|---:|---:|
| relink DataFusion's built-in functions | 0.9 ms | 0.9 ms |
| copy small in-memory tables (`localize`) | 5.0 ms | 0.1 ms |
| detect folds | 0.5 ms | 0.5 ms |
| physical planning | 46.7 ms | 40.3 ms |

The rest, about 29 ms, is outside einfold's planner: datafusion-python encodes the optimized logical plan as protobuf, and datafusion-ffi's wrapper builds a `ForeignSession` and decodes the plan with the session's task context. A `ForeignSession` converts every function the session has, and one conversion of the task context alone takes 234 µs.

- `localize` read the session's task context on every query, though it needs it only to copy a table. Read lazily, it takes 0.1 ms.
- Physical planning in einfold's library took 46.7 ms against DataFusion's 37 ms, because each `ListingTable` scan read its files' Parquet metadata again: datafusion-ffi gives a `ForeignSession`, and every task context it converts, a new `RuntimeEnv::default()`, so its metadata cache is empty on every query. With one `RuntimeEnv` per session, which einfold plans and runs with, physical planning takes 39–40 ms.

**Execution: +57 ms in 0.2.0, +32 ms patched**, as each query's time less its planning, averaged over three runs. Timed alone (`execonly.py`, patched), q11 and q14 run within 0.2 ms of DataFusion, q01 and q06 2.2–2.4 ms slower, and q21 9.5 ms slower. Two build differences, and one query, account for most of the excess:

- **LTO.** datafusion-python builds with `lto = "thin"` and `codegen-units = 2`; einfold builds with Cargo's defaults. einfold runs every plan in its own library, so its copy of DataFusion's operators is the one that runs. Built as datafusion-python is, the whole excess falls from 110–142 ms to 77–105 ms over two alternating runs.
- **The runtime environment**, as above: in 0.2.0, each query's Parquet scans read their metadata again as they run.
- **q21, +9.5 ms.** Its partial aggregate's `elapsed_compute` is 42–47 ms with einfold, against 0.4–1.1 ms without, though its `aggregation_time` is 0.3 ms. The hash join below it builds a table of 3.8 M rows, and freeing that table falls inside the aggregate's poll. einfold allocates blocks of 8 MiB or more with glibc, which unmaps them as they're freed (#119); datafusion-python's mimalloc keeps them for reuse. With every block from mimalloc, q21 is 7 ms faster, but the total doesn't move beyond noise, and #119's peak memory would come back.

**Totals.** Three alternating runs of `phases.py`, as einfold's time over DataFusion's:

| | run 1 | run 2 | run 3 |
|---|---:|---:|---:|
| einfold 0.2.0 (PyPI) | 1.049× | 1.061× | 1.038× |
| patched, with LTO | 1.040× | 1.023× | 1.025× |

## What follows

1. **Build einfold for Python as datafusion-python is built:** thin LTO and 2 codegen units in `python/Cargo.toml`'s release profile. CI's wheel builds take longer.
2. **One `RuntimeEnv` per session**, which einfold's planner plans with and its plans run with, so caches last from query to query; and read the session's task context in `localize` only when a table is copied.
3. **Report the FFI's per-query cost upstream, in #58:** 1.3 ms per query for the round trip, 234 µs of it per conversion of a task context. Upstream, datafusion-ffi could convert a session's functions lazily, and pass its `RuntimeEnv`. The `RuntimeEnv` also matters for correctness: einfold's side of the FFI sees neither the session's object stores, such as one registered for S3, nor its memory limit.
4. **The gate.** With 1–3, TPC-H runs at about 1.02–1.04× DataFusion's time through the FFI, within 1.3 ms of the FFI's cost per query. The maintainer accepted it for M3's gate on 2026-11-01 (1.016–1.047× with #150 and #152 merged), with what remains tracked: the FFI's cost in #58, the session's object stores across the FFI in #149, and q21's large frees in #154.
