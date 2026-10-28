# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Time and peak memory of queries in a DataFusion session, einfold off and on.

Each measurement runs in a process of its own, so that its peak is its own.
The peak is the process's peak resident memory while the query runs, above
what it held once its tables were loaded: Linux's VmHWM, reset just before
the query.

Usage, with einfold and datafusion-python installed:

    python memory.py                      # every case
    python memory.py contraction-256      # one case
    XARRAY_SQL=~/git/xarray-sql python memory.py gpt-loss gpt-gradients

The GPT cases run xarray-sql's GPT demo (`benchmarks/ml/02_nanogpt.py` in
xarray-sql, a GPT written and trained in SQL), and need xarray-sql and ddxdb
(ddx's Python package; ddx differentiates SQL queries) installed, and
XARRAY_SQL pointing at a checkout of xarray-sql that has the demo.
"""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
import time

import numpy as np
import pyarrow as pa

# How many times each measurement runs; the fastest run is reported, and the
# largest peak.
RUNS = 3


def _kb(field: str) -> int:
    with open("/proc/self/status") as status:
        line = next(l for l in status if l.startswith(field))
    return int(line.split()[1])


def _measure(run) -> tuple[float, float]:
    """`run()`'s time in seconds, and its peak above the memory held before it,
    in MB."""
    with open("/proc/self/clear_refs", "w") as refs:
        refs.write("5")  # resets VmHWM to the current resident memory
    base = _kb("VmRSS")
    start = time.perf_counter()
    run()
    seconds = time.perf_counter() - start
    return seconds, (_kb("VmHWM") - base) / 1024


# --- cases: each loads its tables into `ctx`, and returns what to run ---------


def _contraction(ctx, s: int):
    """x(s, p, k) · w(k, e) -> y(s, p, e), with p = 64, k = e = 128: a linear
    layer applied to s sequences of 64 positions."""
    p, k, e = 64, 128, 128
    rng = np.random.default_rng(0)
    si, pi, ki = (a.ravel() for a in np.indices((s, p, k)))
    x = pa.table({"s": si, "p": pi, "k": ki, "v": rng.normal(size=si.size)})
    ctx.register_record_batches("x", [x.to_batches(65536)])
    kk, ee = (a.ravel() for a in np.indices((k, e)))
    w = pa.table({"k": kk, "e": ee, "v": rng.normal(size=kk.size)})
    ctx.register_record_batches("w", [w.to_batches(65536)])
    sql = ("SELECT x.s, x.p, w.e, SUM(x.v * w.v) AS v FROM x JOIN w ON x.k = w.k "
           "GROUP BY x.s, x.p, w.e")
    return lambda: ctx.sql(sql).to_arrow_table()


def _attention(ctx, length: int):
    """Causal self-attention over `length` tokens of 64 features, its
    maximum written as a window."""
    d = 64
    for name, phase in (("q", 0.0), ("k", 0.5), ("v", 1.0)):
        r = np.arange(length * d)
        batch = pa.record_batch({"t": r // d, "d": r % d, "x": np.sin(r * 0.7 + phase)})
        ctx.register_record_batches(name, [[batch]])
    sql = (
        "WITH scores AS (SELECT q.t AS i, k.t AS j, SUM(q.x * k.x) / 8.0 AS s "
        "FROM q JOIN k ON q.d = k.d WHERE q.t >= k.t GROUP BY q.t, k.t), "
        "weights AS (SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS w FROM scores), "
        "norm AS (SELECT i, SUM(w) AS z FROM weights GROUP BY i) "
        "SELECT weights.i AS t, v.d, SUM(weights.w / norm.z * v.x) AS x "
        "FROM weights JOIN v ON weights.j = v.t JOIN norm ON weights.i = norm.i "
        "GROUP BY weights.i, v.d")
    return lambda: ctx.sql(sql).to_arrow_table()


def _gpt(ctx, which: str):
    """xarray-sql's GPT demo at batch 8, 64 channels, context 32, 4 heads and
    one layer: its loss, or the gradients of a training step."""
    import xarray_sql as xql

    demo = os.path.join(os.environ["XARRAY_SQL"], "benchmarks", "ml")
    sys.path.insert(0, demo)
    sizes = {"LAYERS": 1, "EMBED": 64, "HEADS": 4, "CONTEXT": 32, "HEAD": 16, "HIDDEN": 256}
    g = runpy.run_path(os.path.join(demo, "02_nanogpt.py"), run_name="gpt")
    g["model"].__globals__.update(sizes)
    text = g["shakespeare"]()
    params = g["init_params"](len(set(text)), np.random.default_rng(0))
    for name, values in params.items():
        ctx.from_dataset(name, g["as_table"](name, values))
    ctx.from_dataset("batch", next(g["batches"](text, 8, np.random.default_rng(1))))
    ctx.sql(g["model"]()).collect()
    if which == "loss":
        return lambda: ctx.sql(g["loss_statement"]()).to_arrow_table()
    statements = g["gradient_statements"]()
    return lambda: [df.to_arrow_table() for df in xql.ad.sql_all(ctx, statements)]


CASES = {
    "contraction-16": lambda ctx: _contraction(ctx, 16),
    "contraction-64": lambda ctx: _contraction(ctx, 64),
    "contraction-256": lambda ctx: _contraction(ctx, 256),
    "attention-1024": lambda ctx: _attention(ctx, 1024),
    "attention-4096": lambda ctx: _attention(ctx, 4096),
    "gpt-loss": lambda ctx: _gpt(ctx, "loss"),
    "gpt-gradients": lambda ctx: _gpt(ctx, "gradients"),
}


def _one(case: str, on: bool) -> dict:
    """Runs `case` in this process, and returns its measurements."""
    if case.startswith("gpt"):
        import xarray_sql as xql

        ctx = xql.Context()
    else:
        from datafusion import SessionContext

        ctx = SessionContext()
    if on:
        from einfold.datafusion import EinFold

        ctx.with_extensions(EinFold())
    run = CASES[case](ctx)
    runs = [_measure(run) for _ in range(RUNS)]
    return {
        "seconds": min(s for s, _ in runs),
        "peak_mb": max(mb for _, mb in runs),
    }


def main(cases: list[str]) -> None:
    gpt = "XARRAY_SQL" in os.environ
    cases = cases or [c for c in CASES if gpt or not c.startswith("gpt")]
    print("| case | einfold off | einfold on |")
    print("|---|---:|---:|")
    for case in cases:
        cells = []
        for on in (False, True):
            done = subprocess.run(
                [sys.executable, __file__, "--one", case, "on" if on else "off"],
                capture_output=True, text=True)
            if done.returncode != 0:
                cells.append(f"failed ({done.returncode})")
                continue
            m = json.loads(done.stdout.strip().splitlines()[-1])
            cells.append(f"{m['seconds'] * 1e3:,.0f} ms, {m['peak_mb']:,.0f} MB")
        print(f"| {case} | {cells[0]} | {cells[1]} |", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["--one"]:
        print(json.dumps(_one(sys.argv[2], sys.argv[3] == "on")))
    else:
        main(sys.argv[1:])
