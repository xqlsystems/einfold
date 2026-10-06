# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike S19: is a floating-point SUM repeatable across runs in DuckDB and DataFusion?

Sums 4 million doubles that span 16 orders of magnitude, with mixed signs, so that
the order of addition matters. Runs each configuration 20 times and reports how
many distinct results came back, and the relative error against an exact sum.

Run with:
    uv run --with duckdb --with datafusion --with numpy --with pyarrow python float_sums.py
"""

import math
import os

import duckdb
import numpy as np
import pyarrow as pa
from datafusion import SessionConfig, SessionContext

RUNS = 20
rng = np.random.default_rng(0)
n = 4_000_000
x = rng.standard_normal(n) * 10.0 ** rng.integers(-8, 9, n)
exact = math.fsum(x)
table = pa.table({"x": x})


def report(name, values):
    rel_err = abs(values[0] - exact) / abs(exact)
    print(f"| {name} | {len(set(values))} | {rel_err:.1e} |")


print(f"CPUs: {os.cpu_count()}, duckdb {duckdb.__version__}")
print("| Engine and setting | Distinct results in 20 runs | Relative error vs exact sum |")
print("|---|---|---|")

# DuckDB scans its own tables in parallel, one row group per task.
con = duckdb.connect()
con.register("arrow_x", table)
con.sql("create table t as select * from arrow_x")
for threads in (os.cpu_count(), 1):
    con.sql(f"set threads={threads}")
    for fn in ("sum", "fsum"):
        values = [con.sql(f"select {fn}(x) from t").fetchone()[0] for _ in range(RUNS)]
        report(f"DuckDB `{fn}`, {threads} threads", values)

# DataFusion: vary the number of input partitions and target partitions.
batches = table.to_batches(max_chunksize=50_000)
for parts, target in ((1, 1), (1, 16), (16, 1), (16, 16)):
    partitions = [batches] if parts == 1 else [batches[i::parts] for i in range(parts)]
    ctx = SessionContext(SessionConfig().with_target_partitions(target))
    ctx.register_record_batches("t", partitions)
    values = [ctx.sql("select sum(x) from t").to_pylist()[0]["sum(t.x)"] for _ in range(RUNS)]
    report(f"DataFusion `sum`, {parts} input partitions, `target_partitions = {target}`", values)
