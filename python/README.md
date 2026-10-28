<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# einfold for Python

`einfold` installs einfold in a session of [datafusion-python](https://datafusion.apache.org/python/), Apache DataFusion's Python bindings, and so in the sessions that build on one: [xarray-sql](https://github.com/xqlsystems/xarray-sql)'s `Context`, which queries xarray Datasets with SQL, and [ddxdb](https://github.com/xqlsystems/ddx)'s, which differentiates SQL queries.

```python
import einfold
from xarray_sql import Context

ctx = Context()
found = einfold.install(ctx)
ctx.from_dataset("a", a)
ctx.from_dataset("w", w)
ctx.sql("SELECT a.s, w.o, SUM(a.val * w.val) FROM a JOIN w ON a.k = w.k GROUP BY a.s, w.o")
found.plans_with_einfold()   # 1
```

From then on, every query in that session is planned by einfold:
- **folds over joins** (a `SUM` of products grouped by the output's keys: matrix products, contractions, ddx's gradient steps, regridding with a weight table) run by einfold's dense or discrimination algorithms;
- **fold pipelines** (a softmax-weighted average of a fold's results, such as attention) run by its fused, tiled algorithm.

Every other query runs as DataFusion would run it. The results are the same, up to float summation's rounding. The extension counts what it planned: `plans()`, `plans_with_einfold()` and `plans_with_pipelines()`.

## How it works

`SessionContext.with_extensions` asks the extension for codecs and a query planner, through DataFusion's FFI, its stable C interface between separately built libraries:

- **The planner** applies einfold's detection to each plan and plans it with einfold's operator. It returns the plan to the session as one opaque node, to be run in this library, so that it isn't serialized node by node.
- **The codecs** pass tables and plans between the session and the planner by reference, since both are in one process. Protobuf can't encode an in-memory table without a codec, and copies an in-memory scan's data.
- **Built-in functions** cross the FFI as foreign wrappers, and einfold's detection matches `SUM` and `MAX` by type. So the planner relinks foreign functions of those names to DataFusion's own. That's right while the session's `sum` and `max` are the built-ins.

Each query pays a fixed cost for crossing the FFI, about 1.5–3 ms (einfold's spike S29, `docs/spikes/s29-python-planner`). Fixing that needs datafusion-python and its FFI, drafted in einfold#58.

On an i7-8700 (12 threads), from Python, with datafusion-python 55.0.0:

| query | without einfold | with einfold |
|---|---:|---:|
| demo 1's causal attention, 1,024 tokens, 64 features | 2.07 s | 17.5 ms |
| demo 1's causal attention, 4,096 tokens | 109 s | 46 ms |
| matrix product, `50,000 × 16 · 16 × 8` | 37.9 ms | 20.9 ms |
| `SELECT 1` | 0.23 ms | 1.69 ms |

## Building

It needs Rust and [maturin](https://www.maturin.rs/):

```sh
maturin build --release -m python/Cargo.toml -o dist
pip install dist/*.whl "datafusion==55.0.0"
```

The crate isn't a member of einfold's Cargo workspace. It reads the FFI capsules datafusion-python hands it with `unsafe`, which the workspace forbids, and it needs pyo3 and DataFusion's FFI crates, which nothing else does.

Its tests (`tests/test_einfold.py`) check that each query gives the same rows with einfold installed as without, and that einfold runs the folds and pipelines it should. They cover datafusion-python, and also xarray-sql's `Context` and ddxdb's gradients where those packages are installed. Run them from outside `python/`, so that Python imports the installed package rather than the source directory.
