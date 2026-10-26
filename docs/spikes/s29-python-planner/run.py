# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike S29: einfold inside xarray-sql, through datafusion-python's FFI
query planner. Runs a matrix product and ddx's fold-from-domain shape on
xarray Datasets registered with xarray-sql, with einfold off and on, and
checks that both give the same rows."""

import sys
import time

import numpy as np
import pyarrow as pa
import xarray as xr
from xarray_sql import Context

from einfold_spike import EinFold


def best_ms(run, times=5):
    best = float("inf")
    for _ in range(times):
        t = time.perf_counter()
        run()
        best = min(best, (time.perf_counter() - t) * 1e3)
    return best


def tables(ctx, n, d=16, h=8):
    rng = np.random.default_rng(0)
    a = xr.Dataset({"val": (("s", "k"), rng.standard_normal((n, d)))},
                   coords={"s": np.arange(n), "k": np.arange(d)})
    w = xr.Dataset({"val": (("k", "o"), 0.1 * rng.standard_normal((d, h)))},
                   coords={"k": np.arange(d), "o": np.arange(h)})
    ctx.from_dataset("a", a)
    ctx.from_dataset("w", w)


QUERIES = {
    "matmul": "SELECT a.s, w.o, SUM(a.val * w.val) AS v FROM a JOIN w ON a.k = w.k "
              "GROUP BY a.s, w.o",
    # ddx's gradient for a: every row of a gets its group's sum, 0 where none.
    "fold from a domain": "SELECT a.s, a.k, CASE WHEN a.val IS NULL THEN NULL "
              "WHEN g.v IS NULL THEN 0.0 ELSE g.v END AS v FROM a LEFT JOIN "
              "(SELECT c.s, w.k, SUM(c.val * w.val) AS v FROM c JOIN w ON c.o = w.o "
              "GROUP BY c.s, w.k) g ON (a.s IS NOT DISTINCT FROM g.s) "
              "AND (a.k IS NOT DISTINCT FROM g.k)",
}


def agree(want, got):
    """The same keys, and values within a relative 1e-12 of the largest,
    NULLs where NULLs are: float sums differ in their last bits with the
    order of their additions."""
    keys = [c for c in want.column_names if c != "v"]
    a = want.to_pandas().sort_values(keys).reset_index(drop=True)
    b = got.to_pandas().sort_values(keys).reset_index(drop=True)
    if len(a) != len(b) or not a[keys].equals(b[keys]):
        return False
    if not (a["v"].isna() == b["v"].isna()).all():
        return False
    scale = max(1.0, float(np.nanmax(np.abs(a["v"]))) if len(a) else 1.0)
    return bool((np.abs(a["v"] - b["v"]).fillna(0) <= 1e-12 * scale).all())


def config():
    from datafusion import SessionConfig
    return (SessionConfig().with_information_schema(True)
            .set("datafusion.execution.use_row_number_estimates_to_optimize_partitioning", "true")
            .set("datafusion.optimizer.enable_round_robin_repartition", REPARTITION))


REPARTITION = "true"


def main(n):
    off = Context(config())
    on = Context(config())
    einfold = EinFold()
    on_ext = on.with_extensions(einfold)
    for ctx in (off, on):
        tables(ctx, n)
        # The cotangent c(s, o) the gradient reads.
        ctx.sql("CREATE TABLE c AS SELECT a.s, w.o, SUM(a.val * w.val) AS val "
                "FROM a JOIN w ON a.k = w.k GROUP BY a.s, w.o").collect()
    print(f"n = {n}")
    for name, sql in QUERIES.items():
        for label, ctx in (("on (with_extensions)", on_ext), ("on (the original)", on)):
            before = einfold.plans_with_folds()
            got = ctx.sql(sql).to_arrow_table()
            found = einfold.plans_with_folds() - before
            print(f"  {name}, einfold {label}: plans with folds {found}")
        want = off.sql(sql).to_arrow_table()
        got = on_ext.sql(sql).to_arrow_table()
        same = agree(want, got)
        t_off = best_ms(lambda: off.sql(sql).collect())
        t_on = best_ms(lambda: on_ext.sql(sql).collect())
        print(f"  {name}: off {t_off:.1f} ms, on {t_on:.1f} ms, "
              f"{t_off / t_on:.1f}x, same rows: {same}")
    plan = on_ext.sql(QUERIES["matmul"]).execution_plan().display_indent()
    print("  matmul's plan with einfold:\n    " + plan.replace("\n", "\n    "))
    print(f"  planner calls {einfold.plans()}, with folds {einfold.plans_with_folds()}")


if __name__ == "__main__":
    import os
    REPARTITION = os.environ.get("REPARTITION", "true")
    for n in map(int, sys.argv[1:] or ["10000", "50000"]):
        main(n)
