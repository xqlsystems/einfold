# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike S29: what installing einfold's FFI planner costs a query it leaves
alone."""

import time, numpy as np, xarray as xr
from xarray_sql import Context
from einfold_spike import EinFold
def best(f, k=20):
    b = 1e9
    for _ in range(k):
        t = time.perf_counter(); f(); b = min(b, time.perf_counter() - t)
    return b * 1e3
off, base = Context(), Context()
on = base.with_extensions(EinFold())
for c in (off, base):
    a = xr.Dataset({"val": (("s","k"), np.ones((1000,16)))}, coords={"s": np.arange(1000), "k": np.arange(16)})
    c.from_dataset("a", a)
for q in ["SELECT 1", "SELECT COUNT(*) FROM a", "SELECT s, SUM(val) FROM a GROUP BY s"]:
    print(f"{q!r}: off {best(lambda: off.sql(q).collect()):.2f} ms, on {best(lambda: on.sql(q).collect()):.2f} ms")
