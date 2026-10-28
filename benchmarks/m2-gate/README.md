<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# M2's gate

einfold's milestone M2 is demo 1 (`docs/demos.md`): rediscovering FlashAttention from causal attention written in plain SQL. Its gate is the demo's success criteria. This benchmark runs the demo's query, with 64 features, in its two spellings: the maximum as a window, and as `GROUP BY` joined back, which is how ddx, an XQL Systems project for automatic differentiation of SQL queries, writes it. It runs on DataFusion with einfold off and on (`einfold_datafusion::install`), and on DuckDB, each in its own process under GNU `time` for peak memory, each with a 6 GB memory budget. einfold finds the pipeline in the SQL itself; nothing tells it the query is attention.

Run with `cargo build --release`, then `./run.sh` (see its header for GNU `time` and the Python with DuckDB). DuckDB runs through its Python package, `duckdb_run.py`, so that it is the version the environment pins.

## Result

2026-10-27. Intel Core i7-8700 (6 cores, 12 threads), 15 GB RAM, Rust 1.99, DataFusion 55.2.0, DuckDB 1.5.6 (the latest 1.x, through its Python package), einfold at the commit that last changed this file. Times are from SQL to Arrow rows, one run each; peak memory is the process's.

**Window spelling**

| tokens L | DataFusion | DuckDB | einfold on DataFusion | vs. the faster host |
|---:|---:|---:|---:|---:|
| 1,024 | 1.50 s, 0.64 GB | 1.33 s, 0.16 GB | 14 ms, 0.07 GB | 93× |
| 4,096 | 45.1 s, 5.9 GB | 12.7 s, 0.71 GB | 44 ms, 0.10 GB | 291× |
| 8,192 | out of memory | 48.5 s, 1.7 GB | 118 ms, 0.14 GB | 412× |
| 16,384 | | out of memory | 0.39 s, 0.22 GB | — |
| 65,536 | | | 5.5 s, 0.69 GB | — |

**`GROUP BY` spelling**

| tokens L | DataFusion | DuckDB | einfold on DataFusion | vs. the faster host |
|---:|---:|---:|---:|---:|
| 1,024 | 2.85 s, 0.67 GB | 1.04 s, 0.26 GB | 25 ms, 0.07 GB | 41× |
| 4,096 | 73.8 s, 5.3 GB | 12.4 s, 2.0 GB | 45 ms, 0.09 GB | 274× |
| 8,192 | 369 s, 10.1 GB | 39.6 s, 6.2 GB | 136 ms, 0.14 GB | 291× |
| 16,384 | | out of memory | 0.39 s, 0.21 GB | — |
| 65,536 | | | 5.6 s, 0.68 GB | — |

All three agree on every output row's checksum wherever they finish.

## Against demo 1's criteria

- **Found, not told: both spellings, yes.** The plan has one `EinFoldExec` computing the pipeline, `a,b * c,b -> a,c where a >= c; softmax(/ 8, shifted) * c,d -> a,d`, which `PipelineNode` detection (#72) found in the SQL.
- **Masks: yes.** The causal mask leaves exactly the tiles on and below the diagonal: 136 of 256 pairs at L = 1,024, 524,800 of 1,048,576 at 65,536.
- **Fast: yes.** 41–412× faster than the faster host wherever one finishes, and it runs at L = 16,384 and 65,536, where both have run out of memory.
- **Memory: no `N²` term.** Peak memory rises from 70 MB at L = 1,024 to 690 MB at 65,536, while the inputs and output alone, as Arrow rows, are about 400 MB at 65,536. What is left over grows with the inputs: their dense copies and the output's key columns. The `N²` scores never exist; one 64 × 64 tile of them is held per thread.
- **Correct.**
  - Within the error bound, and more accurate than DataFusion. Against a compensated reference at L = 2,048, with the queries scaled 1×, 10× and 40× to spread the scores, einfold's largest error is 0.8, 3.7 and 5.1e-15 of the largest output. DataFusion's is 6.6, 22 and 61e-15.
  - The same bits on 1, 2, 5 and 12 threads.
  - Exactly SQL's results on special values. At L = 128, with a NaN in `q` or `v`, `±∞` in `q`, `1e300` in `k`, or a row whose scores are all `−∞`: the same NaN rows as DataFusion (64, 123, 64, 64, 0 and 64), and no row that differs.
- **Not yet:**
  - **ddx's attention program,** which normalizes before the product with `V`, `a = ex / z` then `SUM(a * v)`. Detection doesn't match that spelling yet.
  - **A Gaussian-kernel weighted average,** `SUM(exp(-dist² / h²) * y) / SUM(exp(-dist² / h²))`, the same rules on a query that isn't attention. Its producer, `Σ (t − s)²`, isn't a sum of products. The rule's `SUM(w * y) / SUM(w)` form is tested on other queries in `tests/detect_pipelines.rs`.
