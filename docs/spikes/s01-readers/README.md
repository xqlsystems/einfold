<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spikes S1, S13, S14: what XQL's readers pass to the plan

**Questions.**

- **S1.** For each reader, what Zarr metadata survives into the table schema, and how is it exposed?
- **S13.** Are a variable's dimensions exposed to the plan? How are lower-dimensional variables handled?
- **S14.** How are fill values and missing data emitted: no rows, 0, NULL, or NaN?

**Answer, in short.** No reader puts Zarr metadata into Arrow schema or field metadata. Each exposes it differently, or not at all. The readers disagree on almost everything einfold needs as facts:

- how missing data appears (NULL or NaN);
- how lower-dimensional variables are handled;
- what statistics reach the engine;
- whether filters are pushed into the scan;
- how times are typed.

So facts must come from per-reader fact providers (design doc §8.1), not from a single convention the readers already follow.

Date: 2026-10-06. Readers tested:

- xarray-sql 0.5.0, with DataFusion 54;
- zarr-datafusion's `zarr-cli` v0.1.1, the prebuilt release from `stratoscale-io/zarr-datafusion`, which the design doc lists as `jayendra13/zarr-datafusion`;
- duckdb-zarr v0.1.3, built from source for DuckDB 1.5.5.

## Method

[`make_fixture.py`](make_fixture.py) builds a small dataset in Zarr v2 and v3 (`fixture/`, not committed):

- `t2m(time, lat, lon)`, float32 with a NaN fill value. One `(2, 3, 4)` chunk is all NaN and was never written (7 of 8 chunks stored).
- `x(time, lat, lon)`, float64, with one chunk of exact zeros.
- `w(lat)`, a lower-dimensional weight, `cos(lat)`.
- Regular `lat` and `lon` coordinates, and CF-encoded times (`hours since 2000-01-01`).

Probes: [`probe_xarray_sql.py`](probe_xarray_sql.py), [`probe_zarr_datafusion.sql`](probe_zarr_datafusion.sql) (`zarr-cli -f`), and [`probe_duckdb_zarr.py`](probe_duckdb_zarr.py) (`uv run --with duckdb==1.5.5`). Each:

- registers the store and inspects the schema;
- counts NULL, NaN and zero rows;
- checks how `w` relates to `lat`;
- examines the physical plan, with DataFusion's statistics turned on, for a filter on `lat`.

## Results

| Question | xarray-sql | zarr-datafusion | duckdb-zarr |
|---|---|---|---|
| Tables for `t2m`, `x`, `w` | One per dimension group: `ds.time_lat_lon` and `ds.lat` | One table; `w` is treated as a coordinate with its own dimension | One table function per dimension group; `dims :=` is required when there are several |
| `w` paired with `lat` | Correct, in its own table | **Wrong.** Every latitude is paired with three `w` values, apparently `w`'s first chunk | Correct, in its own group |
| Never-written NaN chunk | 24 rows of **NULL** | 24 rows of **NaN** | 24 rows of **NULL** |
| Chunk of zeros | 24 rows of 0 | 24 rows of 0 | 24 rows of 0 |
| Arrow schema or field metadata | None | Not checked through SQL; `DESCRIBE` adds role, dimensions, size and chunk shape | None in the schema; `read_zarr_metadata()` and `read_zarr_groups()` return dimensions, role, chunk shape and attributes as tables |
| Dimension columns marked as keys or `NOT NULL` | No (all nullable) | Coordinates are `NOT NULL` | No (all nullable) |
| Dictionary encoding | No | Coordinates, with 16-bit keys | No |
| Time type | `timestamp[ns]` | Timestamp in microseconds, UTC | Raw `BIGINT` (hours); CF time decoding is deferred by design |
| Statistics reaching the engine | Exact row count; exact min, max and null count for dimension columns; none for data variables | None (`Rows=Absent`) | None visible in the plan |
| Filter `lat > 10` | Partitions pruned: the scan drops from 192 to 96 rows | Pushed into the scan (`filters=[lat>10]`) | Not pushed; a `FILTER` runs above `READ_ZARR` (only column selection is pushed) |
| Aggregates pushed into the reader | No | Yes: `COUNT(*)`, `SUM`, `MIN`, `MAX`, `AVG` (`ZarrAggregateExec`) | No |

## Findings

1. **No reader carries facts in Arrow metadata today.** einfold cannot rely on reading facts off the schema. Each reader needs a fact provider: xarray-sql through its Python context, which keeps the registered datasets; duckdb-zarr through its metadata table functions; zarr-datafusion through its extended `DESCRIBE`. Spike S2 asks whether Arrow metadata *could* carry facts through a plan.
2. **Two of three readers already separate variables by dimension group.** For xarray-sql and duckdb-zarr, the "repeated values" problem (design doc §3, problem 5) doesn't arise inside one table, because a lower-dimensional variable gets its own table. It remains real for Zax-SQL, which repeats such variables across a single table, and for users who join the tables back together.
3. **zarr-datafusion returns wrong results for lower-dimensional data variables.** This follows from its documented assumption that every 1-D array is a coordinate, but the answer is silently wrong rather than an error. Worth raising with its maintainers.
4. **Missing data is NULL in two readers and NaN in the third.** The same store gives different values through different readers. That confirms that the fill-value rules (design doc §8.1) must take, as a per-reader fact, how each reader emits fill values. None of the three omits rows for missing chunks.
5. **Statistics and pushdown vary widely.** Only xarray-sql reports statistics, and only for dimensions. None reports value statistics such as min and max per chunk for data variables, so value-based chunk skipping (§9.2) needs reader work in all three. Filter pushdown exists in two readers, by different mechanisms.
6. **Aggregate pushdown already exists in one reader.** zarr-datafusion's `ZarrAggregateExec` is a working precedent for reduction at the source (§10.5, spike S12).
7. **Times are typed three different ways,** so coordinate maps (§6.2) are reader-specific facts too.

## Limitations

- One small synthetic dataset, read locally. Remote stores, sharding, and irregular chunk grids were not tested.
- zarr-datafusion was tested through its command-line tool, not as a Rust library, so its Arrow field metadata was not inspected.
- Zax-SQL needs an Earthmover account, so it was not tested (spike S9).
