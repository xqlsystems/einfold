<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# einfold

**Array algorithms for tensor computations written in SQL.**

Matrix products, tensor contractions and attention layers can all be written in SQL, as joins and aggregates. A matrix product, for one:

```sql
SELECT a.i, b.k, SUM(a.v * b.v) AS v
FROM a JOIN b ON a.j = b.j
GROUP BY a.i, b.k
```

A SQL engine runs that as SQL. It builds every row of the join, then hashes each into its group. That's correct, but slow, and for attention it needs memory that grows with the square of the sequence length. einfold finds these computations in a query's plan and runs them with array algorithms instead. Nothing in the query says it's a matrix product: einfold recognizes the shape.

einfold is part of [XQL Systems](https://xql.systems), which builds SQL access to large scientific arrays, such as climate and weather data stored in [Zarr](https://zarr.dev), the chunked array format. In XQL's model, a dataset is a table with one row per combination of coordinates, so array math becomes joins and sums. einfold makes those fast. The name joins *einsum* (Einstein summation, a notation for tensor contractions) with *fold*, functional programming's word for a reducing pass.

## Status

**einfold 0.1 is in release candidates, for [Apache DataFusion](https://datafusion.apache.org/)**, the Rust SQL engine, through its Python bindings. Its first two milestones are done:

- **M1, folds:** sums of products over a join, the shape of matrix products and of the gradients that [ddx](https://github.com/xqlsystems/ddx) (an XQL Systems project for automatic differentiation of SQL queries) writes.
- **M2, fold pipelines:** a weighted sum of such a sum, run tile by tile. Attention is one, so einfold rediscovers FlashAttention's algorithm from attention written in plain SQL ([demo 1](docs/demos.md)).

Next is M3: the backward pass of attention fused the same way, chains of three or more products, and wider detection ([roadmap](docs/design.md#16-roadmap)). DuckDB is planned as a second engine.

## Use it from Python

```sh
python -m pip install --upgrade --pre "einfold[datafusion]"
```

Wheels are built for Linux (x86_64, aarch64) and macOS (Apple silicon, Intel), for Python 3.10 and later. Windows isn't supported yet ([#118](https://github.com/xqlsystems/einfold/pull/118)).

```python
from datafusion import SessionContext
from einfold.datafusion import EinFold

einfold = EinFold()
ctx = SessionContext().with_extensions(einfold)

ctx.register_parquet("a", "a.parquet")
ctx.register_parquet("b", "b.parquet")
table = ctx.sql("""
    SELECT a.i, b.k, SUM(a.v * b.v) AS v
    FROM a JOIN b ON a.j = b.j
    GROUP BY a.i, b.k
""").to_arrow_table()

einfold.plans_with_einfold()   # 1: einfold ran the matrix product
```

Every query in the session is now planned by einfold. Whatever it doesn't recognize runs as DataFusion would run it. [xarray-sql](https://github.com/xqlsystems/xarray-sql), which queries xarray Datasets with SQL, is a DataFusion session, so `with_extensions(EinFold())` works there too. The [package's README](python/README.md) has the details and its limits.

## Use it from Rust

```toml
[dependencies]
datafusion = "55.2"
einfold-datafusion = "0.1"
```

einfold 0.1 supports DataFusion 55, from 55.2.

```rust
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;

let state = einfold_datafusion::install(SessionStateBuilder::new().with_default_features()).build();
let ctx = SessionContext::new_with_state(state);
```

## What it finds

- **Sums of products over a join,** grouped by columns of either side: matrix products, tensor contractions, and weighted sums such as regridding with a weight table. This includes the same sums left-joined from a table, so that every row of the table gets one, which is how ddx writes a gradient.
- **A weighted sum of such a sum, fused with it.** One query computes scores, and a second sums values weighted by a function of them. Attention is one: weights `exp(s - MAX(s))`, normalized by their sum, written with a window or with `GROUP BY` joined back. The weights can be other expressions of the score, such as a sigmoid's or a Cauchy kernel's. A plain chain of two matrix products fuses too.
- **Scores that are any expression of two rows' values,** such as a Gaussian kernel's `exp(-power(x - y, 2) / h)` over a cross join, for kernel smoothing.
- **Masks that compare keys:** causal, a sliding window, or a band. einfold skips masked tiles rather than computing and discarding them.

## How it works

einfold rewrites query plans, and doesn't parse SQL. It looks for its shapes in DataFusion's optimized logical plan, just before physical planning. It replaces each with a node that its own operator, `EinFoldExec`, runs. The operator picks a kernel when it runs, from the data it reads:

- **dense, positional kernels** when the keys are small integers, as an array's coordinates are;
- **sorting-based kernels** (radix ranking, then sparse products) for any other keys;
- **a fused, tiled kernel** for pipelines. It holds one block of scores per thread, with an online softmax as FlashAttention does, so attention's memory grows with the sequence, not its square.

Where a kernel doesn't apply, or the session's memory pool runs short, the operator runs DataFusion's own plan for that part of the query instead. So einfold changes how fast a query runs, not what it returns:

- **The same rows as DataFusion,** up to the rounding of floating-point sums. NULLs, NaN, infinities and signed zeros follow DataFusion's semantics.
- **Deterministic:** where einfold computes a sum, it gives the same bits on any number of threads and any batch size. Where DataFusion's plan ran instead, its sums can vary with the number of partitions. `EXPLAIN ANALYZE` shows this as `host_plans` on einfold's operator.

These hold under randomized testing against DataFusion itself, with a nightly simulation soak.

## How fast

Times on an Intel i7-8700 (6 cores), against DataFusion 55 and DuckDB 1.5. Attention is causal, with 64 features. It's demo 1's query in plain SQL, measured in Rust ([benchmark](benchmarks/m2-gate/README.md)):

| tokens | DataFusion | DuckDB | einfold on DataFusion |
|---:|---:|---:|---:|
| 1,024 | 1.50 s, 0.64 GB | 1.33 s, 0.16 GB | 14 ms, 0.07 GB |
| 4,096 | 45.1 s, 5.9 GB | 12.7 s, 0.71 GB | 44 ms, 0.10 GB |
| 8,192 | out of memory | 48.5 s, 1.7 GB | 118 ms, 0.14 GB |
| 65,536 | | | 5.5 s, 0.69 GB |

At 65,536 tokens, the scores below the diagonal alone would take 17 GB. einfold's results are closer to an exact reference than DataFusion's, and the same bits on 1, 2, 5 and 12 threads.

The gradient of ddx's matrix-product program (`SUM(tanh(a·w)²)`, with `a` of 50,000 × 16) takes 95 ms with einfold, against 377 ms without, also in Rust ([benchmark](benchmarks/m1-gate/README.md)).

From Python, in xarray-sql's machine-learning demos, on einfold 0.1.0rc1 ([results](https://github.com/xqlsystems/xarray-sql/issues/285)):

- **One training step of a GPT written in SQL,** at nanoGPT's CPU size (4 heads, 128 channels, context 64): 2.7 s, against 9.5 s on DataFusion and 16.4 s on DuckDB.
- **Attention's gradient isn't fused yet** (M3). At 1,024 tokens it takes 1.8–2.2 s, against 4.0–4.4 s on DataFusion.

From Python, each query costs about 1.5–3 ms to hand its plan across DataFusion's foreign-function interface. So for tiny queries, such as the GPT at the demo's own size, einfold is slower than DataFusion alone.

## Repository

- [`crates/einfold-ir`](crates/einfold-ir): the description of a fold or a pipeline, independent of any engine.
- [`crates/einfold-kernels`](crates/einfold-kernels): the kernels, on [Apache Arrow](https://arrow.apache.org/) data, also engine-independent.
- [`crates/einfold-datafusion`](crates/einfold-datafusion): detection in DataFusion's plans, and `EinFoldExec`. The only crate that depends on an engine.
- [`crates/einfold-testkit`](crates/einfold-testkit): the seeded random generator and shrinker the property tests share.
- [`python`](python): the `einfold` package for DataFusion's Python bindings.
- [`benchmarks`](benchmarks): the milestones' gates, M1 and M2.

## Documents

- [Design](docs/design.md): the problem, principles, architecture, and roadmap. The [supplement](docs/supplement.md) holds the details, numbered to match.
- [Lessons](docs/lessons.md): what review and testing have taught us that the design doesn't say.
- [Target demos](docs/demos.md): FlashAttention rediscovered from SQL; NanoGPT in SQL; GraphCast weather forecasts that read only the data they need.
- [Spikes](docs/spikes/README.md): short experiments that settled design questions, each with code and a report.
- [Layout convention](docs/layout-convention.md): a draft XQL Systems convention for Zarr datasets to declare how their data is ordered and summarized, so that engines can scan less.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md), and [`AGENTS.md`](AGENTS.md) for how AI agents work on einfold.

## Related projects

- [xarray-sql](https://github.com/xqlsystems/xarray-sql): query xarray Datasets with SQL.
- [ddx](https://github.com/xqlsystems/ddx): automatic differentiation of SQL queries, so that models written in SQL can be trained.
- [duckdb-zarr](https://github.com/xqlsystems/duckdb-zarr): query Zarr stores from DuckDB.

## License

einfold is licensed under the [Apache License, Version 2.0](LICENSES/Apache-2.0.txt). It follows the [REUSE](https://reuse.software) specification: every file states its copyright and license, and `reuse lint` checks it.
