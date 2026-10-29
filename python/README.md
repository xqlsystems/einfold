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

A SQL engine runs that as SQL: it builds every row of the join, then hashes each into its group. That's correct, but slow, and for attention it needs memory that grows with the square of the sequence length. einfold finds these computations in a query's plan and runs them with array algorithms instead:
- dense, positional kernels when the keys are small integers;
- sorting-based kernels for any other keys;
- a fused, tiled kernel for attention, that never holds more than a tile of its scores.

The rows are the same as the engine's, up to the rounding of floating-point sums. NULLs, NaN and infinities follow SQL's semantics exactly.

## Use

einfold plugs into a SQL engine. Today that's [Apache DataFusion](https://datafusion.apache.org/python/)'s Python bindings:

```sh
pip install "einfold[datafusion]"
```

```python
from datafusion import SessionContext
from einfold.datafusion import EinFold

einfold = EinFold()
ctx = SessionContext().with_extensions(einfold)

ctx.register_parquet("a", "a.parquet")
ctx.register_parquet("b", "b.parquet")
table = ctx.sql("SELECT a.i, b.k, SUM(a.v * b.v) AS v FROM a JOIN b ON a.j = b.j GROUP BY a.i, b.k").to_arrow_table()

einfold.plans_with_einfold()   # 1: einfold ran the matrix product
```

Every query in that session is planned by einfold from then on. What it doesn't recognize runs as DataFusion would run it, planned with DataFusion's default physical planner and optimizer rules. A query planner or physical optimizer rule added to the session before einfold isn't applied.

### Limits

- **Very deep plans.** A session hands einfold each plan through DataFusion's FFI, which decodes it recursively on the thread that asks. On Linux, einfold raises the main thread's stack limit, so plans of about 2,000 joins work there. On another thread, or on macOS, a plan of a few hundred joins can overflow the thread's stack, and the process exits rather than raising an error.
- **Functions named `sum` and `max`.** einfold replaces them with DataFusion's own, which it recognizes in plans. In a session with einfold, a function you register under either name isn't used.

## What it finds

- **Sums of products over a join**, grouped by columns of either side: matrix products, contractions, weighted sums and regridding with a weight table. Also the same left-joined from a table, so that every row of the table gets its sum.
- **Softmax-weighted averages of such a sum**, such as attention, written with a window (`exp(s - MAX(s) OVER (PARTITION BY i))`) or with `GROUP BY` joined back. A causal mask, such as `q.t >= k.t`, is skipped tile by tile.

## How fast

From Python, on an Intel i7-8700 (6 cores), DataFusion 55, with and without einfold:

| query | DataFusion | with einfold |
|---|---:|---:|
| causal self-attention in SQL, 1,024 tokens of 64 features | 2.07 s | 17.5 ms |
| the same, 4,096 tokens | 109 s | 46 ms |
| a matrix product, `50,000 × 16` by `16 × 8` | 37.9 ms | 20.9 ms |
| `SELECT 1`, no tensor computation at all | 0.23 ms | 1.69 ms |

The last row is a fixed cost per query. The extension planner hands each plan across DataFusion's FFI, and that costs about 1.5–3 ms per query, which matters for very short queries.

## Engines

- **DataFusion's Python bindings:** `einfold.datafusion`, now.
- **DuckDB:** planned.

einfold's algorithms work on Apache Arrow data and depend on no engine, so each engine needs only a way to hand einfold the parts of its plans.

## Projects that use einfold

- [xarray-sql](https://github.com/xqlsystems/xarray-sql) queries xarray Datasets, such as climate and weather data, with SQL. Its `Context` is a DataFusion session, so `with_extensions(EinFold())` works there too.
- [ddx](https://github.com/xqlsystems/ddx) differentiates SQL queries, so that models written in SQL can be trained. Its gradients are sums of products over joins, which einfold runs.

## License

Apache-2.0. The source is at [github.com/xqlsystems/einfold](https://github.com/xqlsystems/einfold).
