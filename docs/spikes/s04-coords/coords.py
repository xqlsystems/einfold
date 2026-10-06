# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike S4: how often are coordinates affine in real datasets?

Opens public ERA5 (ARCO-ERA5) and CMIP6 (Pangeo's Google Cloud copy) Zarr
stores anonymously, reads only their coordinate arrays, and classifies each
one-dimensional coordinate:

- exact:     stored == cast(start + k * step) for every index k, bitwise
- linspace:  stored == cast(linspace(first, last, n)), bitwise
- rounding:  affine to within 1e-6 of a step, but not bitwise
- monotone:  strictly monotone, not affine (e.g. Gaussian latitudes, levels)
- other:     not monotone

Times are checked in their stored (CF-encoded) units, not decoded, because
that is what a reader can expose as an exact map. Two-dimensional
coordinates (curvilinear grids) are counted separately.

Usage: python coords.py <cmip6-zarr-consolidated-stores.csv> [stores per table]
"""

import concurrent.futures as cf
import sys

import cftime
import gcsfs
import numpy as np
import pandas as pd
import xarray as xr

ERA5 = [
    "gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3",
    "gcp-public-data-arco-era5/ar/1959-2022-6h-512x256_equiangular_conservative.zarr",
    "gcp-public-data-arco-era5/ar/1959-2022-6h-64x32_equiangular_conservative.zarr",
    "gcp-public-data-arco-era5/ar/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr",
]
fs = gcsfs.GCSFileSystem(token="anon")


def classify(x):
    x = np.asarray(x)
    n = len(x)
    if n < 3 or not np.issubdtype(x.dtype, np.number):
        return "trivial"
    x64 = x.astype(np.float64)
    d = np.diff(x64)
    if not (np.all(d > 0) or np.all(d < 0)):
        return "other"
    k = np.arange(n, dtype=np.float64)
    step = x64[1] - x64[0]
    if np.array_equal((x64[0] + k * step).astype(x.dtype), x):
        return "exact"
    if np.array_equal(np.linspace(x64[0], x64[-1], n).astype(x.dtype), x):
        return "linspace"
    fit = x64[0] + k * (x64[-1] - x64[0]) / (n - 1)
    if np.max(np.abs(fit - x64)) <= 1e-6 * abs(step):
        return "rounding"
    return "monotone"


def kind(name, var):
    attrs = var.attrs
    x = np.asarray(var.values)
    if "since" in str(attrs.get("units", "")) or name.startswith("time"):
        return "time"
    if (np.issubdtype(x.dtype, np.number) and len(x) > 1 and x[0] in (0, 1)
            and np.array_equal(x, x[0] + np.arange(len(x)))):
        return "grid index"  # i, j, x, y, nlat... on curvilinear grids: positions, not coordinates
    if name in ("lat", "latitude", "y", "rlat", "j"):
        return "lat"
    if name in ("lon", "longitude", "x", "rlon", "i"):
        return "lon"
    if name in ("plev", "level", "lev", "hybrid", "olevel", "depth", "height"):
        return "vertical"
    return "other"


def probe(store, source):
    try:
        ds = xr.open_zarr(fs.get_mapper(store), decode_times=False, consolidated=None, chunks=None)
    except Exception as e:  # report and continue
        return [(source, store, "-", "-", "error: " + type(e).__name__, 0)]
    rows = []
    for name in ds.dims:
        if name in ds.coords and ds[name].ndim == 1:
            v = ds[name]
            k, c = kind(name, v), classify(v.values)
            if k == "time" and c == "monotone":
                c = calendar_class(v)
            rows.append((source, store, name, k, c, len(v)))
    # A curvilinear grid: a 2-D latitude or longitude over two spatial dimensions
    # (not a 2-D bounds array such as lat_bnds(lat, bnds)).
    # An unstructured mesh: a 1-D latitude over a cell dimension.
    bounds = ("bnds", "bounds", "vertices", "nv", "vertex", "nbnd")
    for name, v in {**ds.coords, **ds.data_vars}.items():
        if "lat" not in name.lower() or any(b in d for d in v.dims for b in bounds):
            continue
        if v.ndim == 2:
            rows.append((source, store, name, "2-D", "curvilinear", v.size))
            break
        if v.ndim == 1 and v.dims[0] != name:
            rows.append((source, store, name, "2-D", "unstructured", v.size))
            break
    return rows


def calendar_class(v):
    """Is a non-affine time affine in calendar months or years?"""
    try:
        t = cftime.num2date(v.values, v.attrs["units"], v.attrs.get("calendar", "standard"))
    except Exception:
        return "monotone"
    months = np.array([d.year * 12 + d.month for d in t])
    if np.all(np.diff(months) == 1):
        return "monthly"
    return "monotone"


def cmip6_sample(csv, per_table):
    cat = pd.read_csv(csv, usecols=["source_id", "table_id", "variable_id", "experiment_id", "zstore", "version"])
    cat = cat[(cat.experiment_id == "historical")
              & (((cat.table_id == "Amon") & (cat.variable_id == "ta"))
                 | ((cat.table_id == "Omon") & (cat.variable_id == "thetao")))]
    # One store per model and table, so each model's grid counts once.
    picks = cat.sort_values("version").groupby(["table_id", "source_id"]).tail(1)
    picks = picks.groupby("table_id").head(per_table)
    return [(z.removeprefix("gs://"), f"CMIP6 {t}") for z, t in zip(picks.zstore, picks.table_id)]


stores = [(s, "ERA5") for s in ERA5] + cmip6_sample(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 40)
with cf.ThreadPoolExecutor(16) as pool:
    rows = [r for rs in pool.map(lambda a: probe(*a), stores) for r in rs]
df = pd.DataFrame(rows, columns=["source", "store", "coord", "kind", "class", "n"])
df.to_csv("results.csv", index=False)

print(f"stores: {len(stores)}; errors: {(df['class'].str.startswith('error')).sum()}")
ok = df[~df["class"].str.startswith("error")]
print("\n## ERA5, per coordinate")
print(ok[ok.source == "ERA5"].assign(store=lambda d: d.store.str.split("/").str[-1])
      [["store", "coord", "class", "n"]].to_markdown(index=False))
print("\n## One-dimensional coordinates, by source, kind and class")
one_d = ok[ok.kind != "2-D"]  # "2-D" also marks unstructured meshes
print(one_d.groupby(["source", "kind", "class"]).size().unstack(fill_value=0).to_markdown())

print("\n## Horizontal grid per store")
def grid(g):
    if (g["class"] == "curvilinear").any():
        return "curvilinear (2-D lat/lon)"
    if (g["class"] == "unstructured").any():
        return "unstructured mesh"
    h = g[g.kind.isin(["lat", "lon"])]["class"]
    if h.empty:
        return "no 1-D lat/lon"
    if h.isin(["exact", "linspace"]).all():
        return "affine, bitwise"
    if h.isin(["exact", "linspace", "rounding"]).all():
        return "affine up to rounding"
    if h.isin(["exact", "linspace", "rounding", "monotone"]).all():
        return "monotone (not affine)"
    return "not monotone"
print(ok.groupby(["source", "store"]).apply(grid, include_groups=False).groupby(level=0)
      .value_counts().unstack(fill_value=0).to_markdown())
print("\n## Non-monotone one-dimensional coordinates")
print(one_d[one_d["class"] == "other"][["source", "store", "coord"]].to_markdown(index=False))
