# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Build a small Zarr dataset that exercises what spikes S1, S13 and S14 ask about.

- t2m(time, lat, lon): float32 with a NaN fill value; one chunk is all NaN and,
  written with write_empty_chunks=False, is never stored.
- x(time, lat, lon): float64 with one chunk that is all zeros (stored).
- w(lat): a lower-dimensional weight.
- Regular lat/lon coordinates; CF-encoded times ("hours since 2000-01-01").

Writes v2 and v3 copies next to this script, under fixture/.
"""

import pathlib
import shutil

import numpy as np
import xarray as xr

out = pathlib.Path(__file__).parent / "fixture"
shutil.rmtree(out, ignore_errors=True)

time = np.array(["2000-01-01T00", "2000-01-01T06", "2000-01-01T12", "2000-01-01T18"], dtype="datetime64[ns]")
lat = np.linspace(-45.0, 45.0, 6)
lon = np.arange(0.0, 360.0, 45.0)
rng = np.random.default_rng(0)

t2m = rng.standard_normal((4, 6, 8)).astype("float32")
t2m[0:2, 0:3, 0:4] = np.nan  # exactly one (2, 3, 4) chunk, all NaN
x = rng.standard_normal((4, 6, 8))
x[2:4, 3:6, 4:8] = 0.0  # exactly one chunk, all zeros
w = np.cos(np.deg2rad(lat))

ds = xr.Dataset(
    {
        "t2m": (("time", "lat", "lon"), t2m, {"units": "K"}),
        "x": (("time", "lat", "lon"), x),
        "w": (("lat",), w),
    },
    coords={"time": time, "lat": lat, "lon": lon},
)
ds.time.encoding.update(units="hours since 2000-01-01", calendar="proleptic_gregorian")
enc = {
    "t2m": {"chunks": (2, 3, 4), "_FillValue": np.float32("nan")},
    "x": {"chunks": (2, 3, 4)},
    "w": {"chunks": (3,)},
}
for fmt in (2, 3):
    ds.to_zarr(out / f"v{fmt}.zarr", zarr_format=fmt, encoding=enc, write_empty_chunks=False, consolidated=True)
print(ds)
