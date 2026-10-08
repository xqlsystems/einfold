<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S26: where xarray-sql's geospatial time goes

**Question.** xarray-sql, an XQL Systems project that exposes xarray datasets as SQL tables on Apache DataFusion, publishes [geospatial benchmarks](https://xql.systems/xarray-sql/latest/geospatial/) against plain xarray. Measured in-region on a cloud VM, SQL is 2.4–6.2× slower on ERA5 group-bys and joins, and 43× on WeatherBench 2 forecast skill. How much of that gap is engine compute, the only part einfold's positional operators could remove? The rest is reading from cloud storage, and the reader turning arrays into rows. The answer decides whether xarray-sql is a second customer for einfold, alongside ddx (an XQL Systems project for differentiating SQL queries).

**Answer.** Engine compute is a small part of the gap: **1.5–4% of the published gaps on the ERA5 cases** (02–04).

* **In memory, the gaps are small.** xarray-sql takes 74–124 ms against xarray's 9–44 ms on those cases. Absolute differences are 30–112 ms, against published gaps of 2.0–4.5 s.
* **Cold reads favor SQL, apart from registration.** Read cold from cloud storage (from this machine, not in-region), the SQL query, including its read, is as fast as xarray's read and compute, or faster. Registering the full ERA5 archive takes about 46 s against xarray's 2 s to open it.
* **So the published gaps are in xarray-sql's read path at in-region latencies, which this spike could not reproduce.**
* **Forecast skill (05) is the exception.** It is a fold over a join, and compute is a real share of it: 486 ms in xarray-sql against 44 ms in xarray, in memory. But most of that comes from xarray-sql's native path, not DataFusion: the same SQL on pre-converted Arrow tables takes 108 ms.
* **Positional kernels pay only when the reader supplies positions.** Recovering each row's grid position from its coordinate values in NumPy cost 136–817 ms, more than DataFusion's whole hash aggregation (45–108 ms). xarray, whose positions are implicit, is the floor: 9–44 ms.

So xarray-sql's lever is its read path and its native join path, not einfold's kernels.

Date: 2026-10-08. Intel Core i7-8700 (6 cores, 12 threads), 15 GB RAM, home network; xarray-sql 0.5.0 (PyPI), DataFusion (Python) 54.0.0, xarray 2026.9.0, NumPy.

## Method

[`breakdown.py`](breakdown.py) downloads each case's data window once into `fixture/` (git-ignored), as the benchmark scripts select it:

| case | data | rows |
|---|---|---:|
| 02 climatology | ERA5 `2m_temperature`, 3 days hourly, a CONUS-ish box | 1.6 M |
| 03 zonal mean | the same, one day, the whole globe | 24.9 M |
| 04 anomaly | as 02 | 1.6 M |
| 05 forecast skill | WeatherBench 2 Pangu and GraphCast forecasts at 64×32, and ERA5 truth | 3.3 M |

Then, in memory, it times each of these (median of 5):

| column | what it times |
|---|---|
| xarray | the benchmark's own xarray reference |
| xarray-sql | the benchmark's SQL through `XarrayContext` (the native DataFusion provider) on the in-memory dataset, collected as Arrow. The `WHERE` that selects the window is dropped, since the data is already the window |
| native scan | the native provider alone: every value scanned, with a trivial `SUM` |
| engine | DataFusion alone: the same SQL on the rows pre-converted to Arrow (8 partitions) |
| positional, recovered | the same reduction in NumPy, each row's grid position recovered from its coordinates by `searchsorted`, then `bincount` (or a gather, for 04 and 05's join) |
| positional, known | the same, each position derived from the row's place in the reader's chunk-ordered output, as a reader that supplies positions would allow (checked once against the coordinates) |

Every engine result was checked against both positional results (sorted values, to 1e-6).

[`cold.py`](cold.py) times cold reads from Google Cloud Storage for cases 02 and 03. Each run is a fresh process, as xarray-sql's own `run_perf.sh` does, since xarray caches a variable after its first read. It times the xarray reference against the SQL on the lazily registered full archive, and opening against querying. It ran 3 times per case and path.

Run with:

```
uv run --with "xarray-sql==0.5.0" --with gcsfs --with zarr --with netcdf4 python breakdown.py
uv run --with "xarray-sql==0.5.0" --with gcsfs --with zarr python cold.py 02 sql   # or xarray; 02 or 03
```

On NixOS, PyPI wheels need `libstdc++` and `libz` on `LD_LIBRARY_PATH`.

## Results

In memory, ms:

| case | rows | data MB | xarray | xarray-sql | native scan | engine | positional, recovered | positional, known (NumPy) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 02 climatology | 1,607,112 | 6.4 | 8.8 | 77.7 | 10.3 | 45.1 | 136.0 | 81.7 |
| 03 zonal mean | 24,917,760 | 99.7 | 43.7 | 74.0 | 45.4 | 45.2 | 809.1 | 909.2 |
| 04 anomaly | 1,607,112 | 6.4 | 12.7 | 124.3 | 10.9 | 81.4 | 144.1 | 87.2 |
| 05 forecast skill | 3,276,800 | 13.1 | 44.0 | 485.7 | 21.1 | 108.2 | 817.5 | 274.8 |

Cold reads from GCS, seconds, three runs each:

| case | xarray: open | xarray: read + compute | xarray-sql: register full archive | xarray-sql: query, read included |
|---|---|---|---|---|
| 02 climatology | 2.89, 1.77, 2.22 | 16.60, 17.01, 14.80 | 45.34, 46.31, 48.11 | 10.98, 11.95, 13.37 |
| 03 zonal mean | 1.98, 1.88, 2.36 | 4.39, 3.72, 6.84 | 45.53, 46.53, 45.62 | 3.93, 4.55, 4.32 |

Against the published in-region medians:

| case | published gap (SQL − xarray) | in-memory gap here | compute share of the gap |
|---|---:|---:|---:|
| 02 | 4.443 − 1.867 = 2.58 s | 69 ms | ~3% |
| 03 | 2.406 − 0.385 = 2.02 s | 30 ms | ~1.5% |
| 04 | 7.027 − 2.549 = 4.48 s | 112 ms | ~2.5% |
| 05 | 10.714 − 0.248 = 10.47 s, or 1.791 − 0.247 = 1.54 s in the page's multi-engine table | 442 ms | ~4%, or ~29% |

## Findings

1. **On ERA5 group-bys and joins, the gap is not compute.** In memory, xarray-sql is 30–112 ms behind xarray on cases 02–04, 1.5–4% of the published gaps. Read cold, its query (read included) matched or beat xarray's read and compute from this machine. The published gaps must therefore come from xarray-sql's read path at in-region latencies, where per-request time is small and per-partition overhead shows; this spike could not measure that. Positional operators can't remove it.
2. **Registering the whole archive costs about 46 s.** `from_dataset` over the full ARCO-ERA5 store (273 variables, about 1.3 M timesteps) took 45–48 s, against 2 s for xarray to open it. The benchmark times it separately, so it isn't in the published numbers, but every session pays it.
3. **Forecast skill has a real compute gap, but most of it is in xarray-sql's native path, not the engine.**
   * The SQL takes 486 ms through `XarrayContext`, but 108 ms when DataFusion runs it on the same rows pre-converted to Arrow, and the native scan alone takes 21 ms. About 360 ms is overhead specific to the native provider on this join.
   * The page's own figures for this case disagree: 10.7 s in its main table, 1.79 s in its multi-engine table.
4. **Positions must come from the reader.**
   * Recovering each row's grid position from its coordinate values cost more than DataFusion's whole hash aggregation, in every case: 136–817 ms against 45–108 ms.
   * xarray, whose positions are implicit in its arrays, is the floor: 9–44 ms. That would be the target for a positional operator fed positions by the reader, for example each chunk's origin with its batch.
   * The native provider does not return rows in the reader's chunk order, so positions would have to travel with each batch rather than be inferred from order.
5. **For einfold, xarray-sql is not yet a second customer.** Its measured gaps are in reading and in its native provider, not in the join-and-aggregate work einfold targets. The exception is 05's compute, and that is about 0.4 s, most of which sits in the native path.

## Limitations

- Cold reads were timed from a home network, not in-region, so the read path that dominates the published numbers is not reproduced; its share is inferred by subtraction from published medians taken on other hardware (an e2-standard-8 VM).
- Cases 01 (NDVI), 06 (zonal statistics) and 07–09 (reprojection, regridding, warp) were not measured.
- The positional kernels are NumPy, single-threaded, and several passes over the data; a native operator would do better than "positional, known" here. xarray's time is the realistic floor.
- `XarrayContext` ran on an in-memory dataset; with a lazy store its partitions also read from storage.
