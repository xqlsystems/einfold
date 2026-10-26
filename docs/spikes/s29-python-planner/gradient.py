# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike S29: ddx's matmul gradient (M1's gate program) through ddxdb, ddx's
Python package, in Python, with einfold off and on. ddx is an XQL Systems
project for automatic differentiation of SQL queries."""

import sys
import time

import numpy as np
import pyarrow as pa
from ddxdb import Context

from einfold_spike import EinFold


def best_ms(run, times=5):
    best = float("inf")
    for _ in range(times):
        t = time.perf_counter()
        run()
        best = min(best, (time.perf_counter() - t) * 1e3)
    return best


def context(n, einfold, d=16, h=8):
    ctx = Context()
    if einfold is not None:
        ctx.with_extensions(einfold)
    r = np.arange(n * d)
    ctx.register_record_batches("a", [[pa.record_batch(
        {"s": r // d, "k": r % d, "val": np.sin(r.astype(float))})]])
    r = np.arange(d * h)
    ctx.register_record_batches("w", [[pa.record_batch(
        {"k": r // h, "o": r % h, "val": 0.1 * np.cos(r.astype(float))})]])
    return ctx


LOSS = ("WITH c AS (SELECT a.s, w.o, SUM(a.val * w.val) AS z FROM a JOIN w ON a.k = w.k "
        "GROUP BY a.s, w.o), loss AS (SELECT SUM(tanh(z) * tanh(z)) AS l FROM c) ")
GRADIENTS = {"Ā": LOSS + "SELECT s, k, grad(loss, val) AS g FROM a ORDER BY s, k",
             "W̄": LOSS + "SELECT k, o, grad(loss, val) AS g FROM w ORDER BY k, o"}


def main(n):
    einfold = EinFold()
    off, on = context(n, None), context(n, einfold)
    print(f"n = {n}")
    for name, sql in GRADIENTS.items():
        want = off.sql(sql).to_arrow_table().column("g").to_numpy()
        got = on.sql(sql).to_arrow_table().column("g").to_numpy()
        scale = max(1.0, float(np.abs(want).max()))
        same = bool(np.all(np.abs(want - got) <= 1e-12 * scale))
        t_off = best_ms(lambda: off.sql(sql).collect())
        t_on = best_ms(lambda: on.sql(sql).collect())
        print(f"  {name}: off {t_off:.1f} ms, on {t_on:.1f} ms, {t_off / t_on:.1f}x, "
              f"same gradients: {same}")
    print(f"  planner calls {einfold.plans()}, with folds {einfold.plans_with_folds()}")


if __name__ == "__main__":
    for n in map(int, sys.argv[1:] or ["10000", "50000"]):
        main(n)
