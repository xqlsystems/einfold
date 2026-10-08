# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike S26, cold reads: one benchmark case, one path, read from cloud storage.

Run in a fresh process per measurement (xarray caches a variable in memory
after its first read), as xarray-sql's own `run_perf.sh` does:

    uv run --with "xarray-sql==0.5.0" --with gcsfs --with zarr python cold.py 02 sql
    uv run --with "xarray-sql==0.5.0" --with gcsfs --with zarr python cold.py 02 xarray

Prints `case path open_s query_s`: opening the store lazily, then the query or
the xarray reference, including its read.
"""

from __future__ import annotations

import sys
import time

import xarray as xr
import xarray_sql as xql

ERA5 = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
VAR = "2m_temperature"

CASES = {
    # case: (window as .sel() arguments, SQL WHERE, SQL)
    "02": (
        dict(time=slice("2020-06-01", "2020-06-03T23"), latitude=slice(50.0, 25.0), longitude=slice(235.0, 290.0)),
        """SELECT latitude, longitude, date_part('hour', time) AS hour, AVG("2m_temperature") - 273.15 AS clim_c
           FROM era5.surface
           WHERE time BETWEEN TIMESTAMP '2020-06-01 00:00:00' AND TIMESTAMP '2020-06-03 23:00:00'
             AND latitude BETWEEN 25.0 AND 50.0 AND longitude BETWEEN 235.0 AND 290.0
           GROUP BY latitude, longitude, date_part('hour', time)""",
        lambda w: w.groupby("time.hour").mean("time"),
    ),
    "03": (
        dict(time=slice("2020-06-01", "2020-06-01T23")),
        """SELECT latitude, AVG("2m_temperature") - 273.15 AS air_mean_c FROM era5.surface
           WHERE time BETWEEN TIMESTAMP '2020-06-01 00:00:00' AND TIMESTAMP '2020-06-01 23:00:00'
           GROUP BY latitude""",
        lambda w: w.mean(["longitude", "time"]),
    ),
}


def main() -> None:
    case, path = sys.argv[1], sys.argv[2]
    window, sql, reduce = CASES[case]
    t0 = time.perf_counter()
    ds = xr.open_zarr(ERA5, chunks=None, storage_options={"token": "anon"})
    if path == "sql":
        ctx = xql.XarrayContext()
        ctx.from_dataset(
            "era5",
            ds,
            chunks={"time": 6},
            table_names={
                ("time", "latitude", "longitude"): "surface",
                ("time", "level", "latitude", "longitude"): "atmosphere",
            },
        )
    t1 = time.perf_counter()
    if path == "sql":
        ctx.sql(sql).to_arrow_table()
    else:
        reduce(ds[VAR].sel(**window)).values
    t2 = time.perf_counter()
    print(f"{case} {path} {t1 - t0:.2f} {t2 - t1:.2f}", flush=True)


if __name__ == "__main__":
    main()
