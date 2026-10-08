# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike S26: where does xarray-sql's time go on its geospatial benchmarks?

xarray-sql, an XQL Systems project that exposes xarray datasets as SQL tables
on Apache DataFusion, publishes geospatial benchmarks against plain xarray
(https://xql.systems/xarray-sql/latest/geospatial/). On ERA5 group-bys and
joins, SQL is 2.4-6.2x slower, and 43x on WeatherBench 2 forecast skill. Those
totals mix three costs: reading the data from cloud storage, the reader turning
arrays into Arrow rows, and the engine's compute. Only the last is what
positional operators (einfold) would change.

This script downloads each case's data window once (a few MB to ~100 MB), then
times, in memory, median of 5:

- ``xarray``: the benchmark's own xarray reference;
- ``xarray-sql``: the benchmark's SQL through ``XarrayContext`` on the
  in-memory dataset, result collected as Arrow;
- ``native scan``: xarray-sql's native reader alone: every value scanned,
  with a trivial ``SUM``;
- ``engine``: only DataFusion, the same SQL on the pre-converted Arrow table;
- ``positional, recovered``: the same reduction from the Arrow rows by
  position, in NumPy, with each row's grid position recovered from its
  coordinate values (``searchsorted``), then ``bincount`` or a gather;
- ``positional, known``: the same, with each row's grid position known from
  its place in the reader's chunk-ordered output (checked against the
  coordinates), as a reader that supplies positions would allow.

Cases: 02 climatology (GROUP BY lat, lon, hour), 03 zonal mean (GROUP BY lat),
04 anomaly (climatology CTE self-JOIN), 05 forecast skill (JOIN on valid time
and the grid, GROUP BY model, lead).

Run with (network needed for the first run; data is cached in ``fixture/``):
    uv run --with "xarray-sql==0.5.0" --with gcsfs --with zarr --with netcdf4 \\
        python breakdown.py
"""

from __future__ import annotations

import pathlib
import statistics
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import xarray as xr
import xarray_sql as xql
from datafusion import SessionConfig, SessionContext

RUNS = 5
HERE = pathlib.Path(__file__).parent
FIXTURE = HERE / "fixture"
ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
WB2 = "gs://weatherbench2/datasets"
GRID = "64x32_equiangular_conservative"
VAR = "2m_temperature"


# --- data ------------------------------------------------------------------------


def _cached(name: str, make) -> xr.Dataset:
    path = FIXTURE / f"{name}.nc"
    if not path.exists():
        FIXTURE.mkdir(exist_ok=True)
        make().to_netcdf(path)
    return xr.open_dataset(path).load()


def _open(url: str, **kw) -> xr.Dataset:
    return xr.open_zarr(url, chunks=None, storage_options={"token": "anon"}, **kw)


def era5_window() -> xr.Dataset:
    """Cases 02 and 04: three days over a CONUS-ish box."""
    return _cached(
        "era5_window",
        lambda: _open(ERA5)[[VAR]].sel(
            time=slice("2020-06-01", "2020-06-03T23"),
            latitude=slice(50.0, 25.0),
            longitude=slice(235.0, 290.0),
        ),
    )


def era5_day() -> xr.Dataset:
    """Case 03: one day, the whole globe."""
    return _cached(
        "era5_day",
        lambda: _open(ERA5)[[VAR]].sel(time=slice("2020-06-01", "2020-06-01T23")),
    )


def wb2() -> tuple[xr.Dataset, xr.Dataset]:
    """Case 05: Pangu and GraphCast forecasts, and ERA5 truth, at 64x32."""
    def forecasts():
        era5 = _open(f"{WB2}/era5/1959-2023_01_10-6h-{GRID}.zarr")
        init = slice("2020-01-01", "2020-01-10")
        pangu = _open(f"{WB2}/pangu/2018-2022_0012_{GRID}.zarr", decode_timedelta=True)[[VAR]].sel(time=init)
        graphcast = _open(
            f"{WB2}/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours-{GRID}.zarr",
            decode_timedelta=True,
        )[[VAR]].sel(time=init)
        return xr.concat([pangu, graphcast], dim="model").assign_coords(
            model=["pangu", "graphcast"], latitude=era5.latitude.values, longitude=era5.longitude.values
        )

    def truth():
        era5 = _open(f"{WB2}/era5/1959-2023_01_10-6h-{GRID}.zarr")
        f = forecasts()
        valid_max = f.time.values.max() + f.prediction_timedelta.values.max()
        return era5[[VAR]].sel(time=slice("2020-01-01", pd.Timestamp(valid_max)))

    return _cached("wb2_forecasts", forecasts), _cached("wb2_truth", truth)


# --- timing ------------------------------------------------------------------------


def median_ms(fn) -> float:
    fn()
    ts = []
    for _ in range(RUNS):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3


def rows(ds: xr.Dataset, chunks: dict) -> pa.Table:
    return xql.read_xarray(ds, chunks=chunks).read_all()


def engine(table: pa.Table, name: str, sql: str, extra: dict | None = None) -> pa.Table:
    ctx = SessionContext(SessionConfig().with_target_partitions(8))
    ctx.register_record_batches(name, [table.to_batches(max_chunksize=65536)])
    for n, t in (extra or {}).items():
        ctx.register_record_batches(n, [t.to_batches(max_chunksize=65536)])
    return ctx.sql(sql).to_arrow_table()


def xarray_sql(datasets: dict, chunks: dict, sql: str) -> pa.Table:
    ctx = xql.XarrayContext()
    for name, ds in datasets.items():
        ctx.from_dataset(name, ds, chunks=chunks)
    return ctx.sql(sql).to_arrow_table()


def native_scan(datasets: dict, chunks: dict) -> None:
    """The native reader's cost: scan every value, with a trivial aggregate."""
    ctx = xql.XarrayContext()
    for name, ds in datasets.items():
        ctx.from_dataset(name, ds, chunks=chunks)
    for name in datasets:
        ctx.sql(f'SELECT SUM("{VAR}") FROM {name}').to_arrow_table()


def grid_index(n: int, sizes: list[int]) -> list[np.ndarray]:
    """Each row's index along each dim, for rows in C order over `sizes`:
    what a reader that emits each chunk in order can supply per batch."""
    r = np.arange(n)
    out = []
    for k in range(len(sizes)):
        inner = int(np.prod(sizes[k + 1:], dtype=np.int64))
        out.append((r // inner) % sizes[k])
    return out


def positions(values: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Each value's position among the coordinate values (which are exact)."""
    order = np.argsort(coords)
    return order[np.searchsorted(coords, values, sorter=order)]


# --- cases ------------------------------------------------------------------------

CLIM_SQL = """
    SELECT latitude, longitude, date_part('hour', time) AS hour, AVG("2m_temperature") - 273.15 AS clim_c
    FROM {t} GROUP BY latitude, longitude, date_part('hour', time)
"""


def case02(out: list) -> None:
    ds = era5_window()
    chunks = {"time": 6}
    table = rows(ds, chunks)
    sql = CLIM_SQL.format(t="era5")

    def ref():
        return ds[VAR].groupby("time.hour").mean("time") - 273.15

    lat, lon = ds.latitude.values, ds.longitude.values

    def positional():
        hour = pc_hour(table["time"])
        i = positions(table["latitude"].to_numpy(), lat)
        j = positions(table["longitude"].to_numpy(), lon)
        g = (i * len(lon) + j) * 24 + hour
        n = len(lat) * len(lon) * 24
        s = np.bincount(g, weights=table[VAR].to_numpy(), minlength=n)
        c = np.bincount(g, minlength=n)
        return s[c > 0] / c[c > 0] - 273.15

    sizes = [ds.sizes[d] for d in ds[VAR].dims]  # time, latitude, longitude
    hour0 = int(pd.Timestamp(ds.time.values[0]).hour)

    def known():
        t_, i, j = grid_index(table.num_rows, sizes)
        g = (i * len(lon) + j) * 24 + (t_ + hour0) % 24
        n = len(lat) * len(lon) * 24
        s = np.bincount(g, weights=table[VAR].to_numpy(), minlength=n)
        c = np.bincount(g, minlength=n)
        return s[c > 0] / c[c > 0] - 273.15

    out.append(measure("02 climatology", ds, table, ref, lambda: xarray_sql({"era5": ds}, chunks, sql),
                       lambda: native_scan({"era5": ds}, chunks), lambda: engine(table, "era5", sql), positional, known))


def pc_hour(col: pa.ChunkedArray) -> np.ndarray:
    t = col.to_numpy().astype("datetime64[h]").astype(np.int64)
    return t % 24


def case03(out: list) -> None:
    ds = era5_day()
    chunks = {"time": 6}
    table = rows(ds, chunks)
    sql = 'SELECT latitude, AVG("2m_temperature") - 273.15 AS air_mean_c FROM era5 GROUP BY latitude'
    lat = ds.latitude.values

    def ref():
        return ds[VAR].mean(["longitude", "time"]) - 273.15

    def positional():
        i = positions(table["latitude"].to_numpy(), lat)
        s = np.bincount(i, weights=table[VAR].to_numpy(), minlength=len(lat))
        c = np.bincount(i, minlength=len(lat))
        return s[c > 0] / c[c > 0] - 273.15

    sizes = [ds.sizes[d] for d in ds[VAR].dims]

    def known():
        _, i, _ = grid_index(table.num_rows, sizes)
        s = np.bincount(i, weights=table[VAR].to_numpy(), minlength=len(lat))
        c = np.bincount(i, minlength=len(lat))
        return s[c > 0] / c[c > 0] - 273.15

    out.append(measure("03 zonal mean", ds, table, ref, lambda: xarray_sql({"era5": ds}, chunks, sql),
                       lambda: native_scan({"era5": ds}, chunks), lambda: engine(table, "era5", sql), positional, known))


def case04(out: list) -> None:
    ds = era5_window()
    chunks = {"time": 6}
    table = rows(ds, chunks)
    sql = f"""
        WITH clim AS ({CLIM_SQL.format(t='era5')})
        SELECT a.time, a.latitude, a.longitude, a."2m_temperature" - (c.clim_c + 273.15) AS anomaly
        FROM era5 a JOIN clim c
          ON a.latitude = c.latitude AND a.longitude = c.longitude AND date_part('hour', a.time) = c.hour
    """

    def ref():
        t = ds[VAR]
        return t.groupby("time.hour") - t.groupby("time.hour").mean("time")

    lat, lon = ds.latitude.values, ds.longitude.values

    def positional():
        hour = pc_hour(table["time"])
        i = positions(table["latitude"].to_numpy(), lat)
        j = positions(table["longitude"].to_numpy(), lon)
        g = (i * len(lon) + j) * 24 + hour
        n = len(lat) * len(lon) * 24
        v = table[VAR].to_numpy().astype(np.float64)
        clim = np.bincount(g, weights=v, minlength=n) / np.maximum(np.bincount(g, minlength=n), 1)
        return v - clim[g]

    sizes = [ds.sizes[d] for d in ds[VAR].dims]
    hour0 = int(pd.Timestamp(ds.time.values[0]).hour)

    def known():
        t_, i, j = grid_index(table.num_rows, sizes)
        g = (i * len(lon) + j) * 24 + (t_ + hour0) % 24
        n = len(lat) * len(lon) * 24
        v = table[VAR].to_numpy().astype(np.float64)
        clim = np.bincount(g, weights=v, minlength=n) / np.maximum(np.bincount(g, minlength=n), 1)
        return v - clim[g]

    out.append(measure("04 anomaly", ds, table, ref, lambda: xarray_sql({"era5": ds}, chunks, sql),
                       lambda: native_scan({"era5": ds}, chunks), lambda: engine(table, "era5", sql), positional, known))


def case05(out: list) -> None:
    f, e = wb2()
    chunks = {"time": 100}
    ft, et = rows(f, chunks), rows(e, chunks)
    sql = """
        SELECT f.model, f.prediction_timedelta AS lead,
               SQRT(AVG(POWER(CAST(f."2m_temperature" AS DOUBLE) - e."2m_temperature", 2))) AS rmse
        FROM forecasts f JOIN era5 e
          ON e.time = f.time + f.prediction_timedelta AND e.latitude = f.latitude AND e.longitude = f.longitude
        GROUP BY f.model, f.prediction_timedelta
    """

    def ref():
        fv, ev = f[VAR], e[VAR]
        out_ = []
        for lead in fv.prediction_timedelta.values:
            e_at = ev.sel(time=fv.time.values + lead)
            out_.append(np.sqrt(((fv.sel(prediction_timedelta=lead) - e_at.values) ** 2).mean(["time", "latitude", "longitude"])))
        return out_

    lat, lon, models = f.latitude.values, f.longitude.values, f.model.values
    leads, etimes = f.prediction_timedelta.values, e.time.values

    def positional():
        valid = ft["time"].to_numpy() + ft["prediction_timedelta"].to_numpy()
        t = positions(valid, etimes)
        i = positions(ft["latitude"].to_numpy(), lat)
        j = positions(ft["longitude"].to_numpy(), lon)
        # e's rows by position: its own (time, lat, lon) positions.
        ep = (positions(et["time"].to_numpy(), etimes) * len(lat) + positions(et["latitude"].to_numpy(), lat)) * len(lon) \
            + positions(et["longitude"].to_numpy(), lon)
        grid = np.empty(len(etimes) * len(lat) * len(lon))
        grid[ep] = et[VAR].to_numpy()
        d = ft[VAR].to_numpy().astype(np.float64) - grid[(t * len(lat) + i) * len(lon) + j]
        g = positions(ft["model"].to_numpy(zero_copy_only=False).astype(str), models.astype(str)) * len(leads) \
            + positions(ft["prediction_timedelta"].to_numpy(), leads)
        s = np.bincount(g, weights=d * d, minlength=len(models) * len(leads))
        c = np.bincount(g, minlength=len(models) * len(leads))
        return np.sqrt(s[c > 0] / c[c > 0])

    # The reader's row order over the forecasts' dims, found (not assumed) by
    # matching each candidate order against the coordinate columns.
    import itertools

    coord = {d: (ft[d].to_numpy(zero_copy_only=False) if d != "model" else ft[d].to_numpy(zero_copy_only=False).astype(str))
             for d in f[VAR].dims}
    values = {d: (f[d].values if d != "model" else f[d].values.astype(str)) for d in f[VAR].dims}
    for fdims in itertools.permutations(f[VAR].dims):
        fsizes = [f.sizes[d] for d in fdims]
        idx = dict(zip(fdims, grid_index(ft.num_rows, fsizes)))
        if all(np.array_equal(values[d][idx[d]], coord[d]) for d in fdims):
            break
    else:
        raise AssertionError("the forecasts' rows are in no C order of their dims")
    fdims = list(fdims)
    esizes = [e.sizes[d] for d in ("time", "latitude", "longitude")]
    step = (etimes[1] - etimes[0])

    def known():
        idx = dict(zip(fdims, grid_index(ft.num_rows, fsizes)))
        # The truth's time position of valid = init + lead, by arithmetic on
        # its regular time axis.
        valid = f.time.values[idx["time"]] + leads[idx["prediction_timedelta"]]
        t_ = ((valid - etimes[0]) // step).astype(np.int64)
        # The truth as a dense array over (time, latitude, longitude): WB2
        # stores (time, longitude, latitude), so transpose first.
        grid = e[VAR].transpose("time", "latitude", "longitude").values.reshape(-1).astype(np.float64)
        d = ft[VAR].to_numpy().astype(np.float64) - grid[(t_ * esizes[1] + idx["latitude"]) * esizes[2] + idx["longitude"]]
        g = idx["model"] * len(leads) + idx["prediction_timedelta"]
        s = np.bincount(g, weights=d * d, minlength=len(models) * len(leads))
        c = np.bincount(g, minlength=len(models) * len(leads))
        return np.sqrt(s[c > 0] / c[c > 0])

    out.append(measure("05 forecast skill", f, ft, ref, lambda: xarray_sql({"forecasts": f, "era5": e}, chunks, sql),
                       lambda: native_scan({"forecasts": f, "era5": e}, chunks),
                       lambda: engine(ft, "forecasts", sql, {"era5": et}), positional, known))


def measure(name, ds, table, ref, sqlfn, readerfn, enginefn, positional, known) -> dict:
    # The engine and the positional kernel must compute the same numbers.
    want = np.sort(np.asarray(enginefn().column(-1).to_numpy(zero_copy_only=False), dtype=np.float64))
    for fn in (positional, known):
        got = np.sort(np.asarray(fn(), dtype=np.float64))
        assert want.shape == got.shape, f"{name}: {want.shape} vs {got.shape}"
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6, err_msg=name)
    return {
        "case": name,
        "rows": table.num_rows,
        "MB": round(sum(v.nbytes for v in ds.data_vars.values()) / 1e6, 1),
        "xarray": median_ms(ref),
        "xarray-sql": median_ms(sqlfn),
        "reader": median_ms(readerfn),
        "engine": median_ms(enginefn),
        "positional": median_ms(positional),
        "known": median_ms(known),
    }


def main() -> None:
    out: list = []
    for case in (case02, case03, case04, case05):
        case(out)
        r = out[-1]
        print(r, flush=True)
    print("\n| case | rows | data MB | xarray | xarray-sql | native scan | engine | positional, recovered | positional, known |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in out:
        print(f"| {r['case']} | {r['rows']:,} | {r['MB']} | {r['xarray']:.1f} | {r['xarray-sql']:.1f} | "
              f"{r['reader']:.1f} | {r['engine']:.1f} | {r['positional']:.1f} | {r['known']:.1f} |")


if __name__ == "__main__":
    main()
