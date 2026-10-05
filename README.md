<!--
SPDX-FileCopyrightText: 2026 Alex Merose

SPDX-License-Identifier: Apache-2.0
-->

# einfold

**Fast tensor contractions inside SQL engines, for the XQL model.**

einfold makes tensor computation fast inside SQL engines, on any hardware those engines run on. It is part of [XQL Systems](https://xql.systems), which builds SQL access to large scientific arrays stored in [Zarr](https://zarr.dev).

In the XQL model, a dataset becomes a table with one row per combination of coordinates, and a tensor contraction (matrix multiplication is the simplest case) becomes a join followed by a `SUM`. SQL engines run that pattern badly: they materialize huge intermediate joins, sum too late, and ignore the dense structure of arrays.

einfold fixes this by **rewriting query plans, not by executing them**. It finds the einsums (Einstein-summation expressions) inside a relational plan and returns a plan that the engine runs faster. Engines can run it on CPU or GPU: [Apache DataFusion](https://datafusion.apache.org), [DuckDB](https://duckdb.org), DuckDB with GPU extensions such as [Sirius](https://github.com/sirius-db/sirius), or [NVIDIA's GPU Query Engine](https://build.nvidia.com/nvidia/gpu-query-engine). einfold never touches a device itself.

The name joins *einsum* with *fold*, functional programming's word for a reducing pass: einfold evaluates an einsum by folding its sums into the joins that feed them.

## Status

Design phase. There is no code yet.

## Documents

- [Design](docs/design.md): the architecture, core abstractions, optimizer, and roadmap.
- [Layout convention](docs/layout-convention.md): a draft XQL Systems convention that lets Zarr datasets declare how their data is ordered and summarized (curve orderings such as Z-order, chunk visit order, written-chunk records, and per-chunk min/max), so that query engines can scan less.

## Related projects

- [xarray-sql](https://github.com/alxmrs/xarray-sql): query Xarray datasets with SQL.
- [duckdb-zarr](https://github.com/xqlsystems/duckdb-zarr): query Zarr stores from DuckDB.
- [ddx](https://github.com/xqlsystems/ddx): automatic differentiation of SQL queries. Its gradient computations are mostly tensor contractions, which motivated einfold.

## License

einfold is licensed under the [Apache License, Version 2.0](LICENSES/Apache-2.0.txt).

The project follows the [REUSE](https://reuse.software) specification: every file states its copyright and license with SPDX tags, and license texts live in [`LICENSES/`](LICENSES). To check compliance, run:

```sh
reuse lint
```
