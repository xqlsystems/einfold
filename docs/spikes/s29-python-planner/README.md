<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S29: einfold in xarray-sql, through datafusion-python's FFI planner

**Question.** Can einfold run inside xarray-sql today? xarray-sql is an XQL Systems project for querying xarray Datasets with SQL. It runs in Python on datafusion-python, Apache DataFusion's Python bindings. einfold is Rust, so it would plug in as a query planner, installed from Python through `SessionContext.with_extensions`. datafusion-python 55 hands the planner to the session through its FFI, DataFusion's stable C interface between separately built libraries. What breaks on the way, and what's the speedup once it works?

**Answer.** Yes, with four workarounds in the bridge and one change in einfold (xqlsystems/einfold#57). On xarray-sql tables, from Python, with the same rows as einfold off:

| query, n rows of `a` / 16 | einfold off, ms | einfold on, ms | speedup |
|---|---:|---:|---:|
| matrix product, n = 10,000 | 8.7 | 7.4 | 1.2× |
| matrix product, n = 50,000 | 47.7 | 17.0 | 2.8× |
| ddx's fold from a domain, n = 10,000 | 39.7 | 10.5 | 3.8× |
| ddx's fold from a domain, n = 50,000 | 194.8 | 22.1 | 8.8× |

ddx, an XQL Systems project for automatic differentiation of SQL queries, has a Python package, ddxdb. Its `grad()` in SQL runs M1's gate program, `matmul`'s gradient:

| gradient, through ddxdb | einfold off, ms | einfold on, ms | speedup |
|---|---:|---:|---:|
| Ā, n = 10,000 | 95.8 | 98.2 | 1.0× |
| Ā, n = 50,000 | 457.9 | 247.1 | 1.9× |
| W̄, n = 10,000 | 33.3 | 69.8 | 0.5× |
| W̄, n = 50,000 | 105.4 | 102.8 | 1.0× |

That is well short of M1's 4× in Rust (#49), for two reasons. Every planned query pays a fixed cost for crossing the FFI, 1.4–3.4 ms (below), and ddxdb plans 12 statements per gradient. Of those 12, einfold finds a fold in two: the forward contraction and the Ā step. Both costs are worth taking on before xarray-sql's demos use einfold.

Run on the machine in [the spikes index](../README.md), with datafusion-python 55.0.0 (a local wheel, built with DataFusion 55.2.0, which has `with_extensions`), xarray-sql at 74cea87, and ddxdb from ddx's `main`.

## What it took

The bridge (`src/lib.rs`) is an extension bundle, a Python object with two hooks:
- **planner hook:** wraps the session's planner in one that applies `DetectFolds`, einfold's optimizer rule, and plans with `EinFoldPlanner`;
- **components hook:** adds the codecs below.

Each step to a working query met one obstacle:

1. **Tables can't cross as protobuf.** datafusion-python serializes each logical plan as protobuf to hand it to the planner. Protobuf can't encode a table, such as xarray-sql's in-memory ones, without a codec. Every query failed with `LogicalExtensionCodec is not provided`, before einfold saw it.
   - **Workaround:** a logical codec, `ByReference`, that encodes any table as an index into a registry in this process, and decodes it back to the same table. The planner runs in the host's process, so a reference is enough.
   - **Upstream:** a codec like this, built in, for planners in the same process.
2. **Built-in functions lose their identity.** Decoded through the host's registry, `SUM` arrives as an FFI wrapper of the host's `sum`, not as DataFusion's own `Sum`. `DetectFolds` matches `SUM` by type, as DataFusion asks of rules, so it found nothing.
   - **Workaround:** the bridge relinks a foreign aggregate named `sum` to DataFusion's own before detecting. That's right only while the host's `sum` is the built-in.
   - **Upstream:** decode built-in functions as the planner library's own.
3. **The physical plan returns as protobuf too.** Its standard nodes come back as the host's own, and DataFusion's protobuf encodes an in-memory scan with its data. A plan crossing node by node copies every table it reads, twice: 800,000 rows of `a` per query, about 9 ms. Nodes protobuf doesn't know, such as `EinFoldExec`, need a physical codec.
   - **Workaround:** the bridge returns each plan inside one node, `Opaque`, which has no visible children and is passed by reference. The bridge plans every query itself, folds or not, so none crosses node by node. That skips any planner the session had below this one.
   - **Upstream:** a way for an FFI planner's result to come back as a handle rather than as protobuf.
4. **Two copies of Tokio.** Tokio is the async runtime DataFusion runs on. The bridge library has its own copy, and the host polls its plan from the host's threads, where that copy sees no runtime. Spawning there panics in an FFI function that can't unwind, which aborts Python.
   - **Workarounds:**
     - `Opaque` enters a runtime of the bridge's own whenever the host polls it;
     - `EinFoldExec` now runs where no runtime is current: it reads inputs as futures, runs its kernel inline, and runs the host's plan on its own runtime (xqlsystems/einfold#57).
   - **Upstream:** datafusion-ffi could enter the runtime a library supplied whenever it executes or polls that library's plans.

The host's physical optimizer rules reach a plan built in the bridge only as FFI proxies of its nodes, which they can't optimize. That broke a limit's plan, and each rule call crossed the FFI. So `LocalRules`, a session wrapper, plans with the bridge's own copy of DataFusion's rules.

## What remains

**A fixed cost per query.** Installing the bridge costs every query, folds or not, as `overhead.py` measures:

| query | einfold off, ms | einfold on, ms |
|---|---:|---:|
| `SELECT 1` | 0.24 | 1.65 |
| `SELECT COUNT(*) FROM a` | 0.40 | 3.08 |
| `SELECT s, SUM(val) FROM a GROUP BY s` | 0.94 | 4.34 |

Most of it is the logical plan's protobuf round trip and the session's calls across the FFI. For the GPT demo, which is bound by planning time, this would cost more than einfold saves. The fixes are upstream: a planner hook that takes the logical plan as a handle rather than protobuf, and the items above.

**Detection inside ddxdb's statements.** ddxdb's `grad()` writes its own SQL, not the plans ddx's Rust `ad::grad` builds. einfold finds folds in two of its 12 statements per gradient, where it finds three of the Rust program's steps (#49). That's worth a look at the statements, as S28 looked at the Rust program's.

**Spike shortcuts.** The registries never free what they hold, and `sum` is relinked by name. Neither belongs in a real `einfold-python` crate.

## Reproduce

On NixOS, xarray-sql's environment needs its flake's libraries on `LD_LIBRARY_PATH` (`libstdc++` and `zlib`). Then, with `CARGO_TARGET_DIR` pointing at einfold's target directory to share its build cache:

```sh
maturin build --release -o wheels -i <xarray-sql>/.venv/bin/python
python -c "import zipfile, glob; zipfile.ZipFile(glob.glob('wheels/*.whl')[0]).extractall('py')"
PYTHONPATH=py <xarray-sql>/.venv/bin/python run.py 10000 50000
PYTHONPATH=py:<ddx>/python/ddxdb <xarray-sql>/.venv/bin/python gradient.py 10000 50000
PYTHONPATH=py <xarray-sql>/.venv/bin/python overhead.py
```

`EINFOLD_SPIKE_DEBUG=1` prints each plan the bridge sees, whether einfold found a fold in it, and how long it took to plan.
