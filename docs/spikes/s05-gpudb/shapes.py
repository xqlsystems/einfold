# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike S5 (CPU half): which of einfold's relational shapes does gpudb rewrite?

gpudb rewrites a SQL statement before DuckDB plans it, in its Python client
(`gpudb.connect`), and reports each decision through `last_rewrite()`. This
runs einfold's relational-form shapes through it, twice each (the first sight
of a shape schedules residency), and records the decision and reason.

The rewrite decision is made on the CPU; whether a rewritten statement then
runs on a GPU depends on the build. This machine's GPU (GTX 1080 Ti, compute
capability 6.1) is below gpudb's minimum (7.5), so GPU timings are not taken.

Usage: PYTHONPATH=<gpudb checkout>/python python shapes.py
"""

import json
import time

import gpudb

N = 2_000_000  # above gpudb's 1M-row floor
SHAPES = {
    "matmul: join + GROUP BY + SUM":
        "SELECT a.i, b.j, SUM(a.v * b.v) AS v FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j",
    "matrix-vector (join onto a unique key)":
        "SELECT a.i, SUM(a.v * w.v) AS v FROM a JOIN w ON a.k = w.k GROUP BY a.i",
    "eager aggregation: pre-aggregated subquery":
        "SELECT a.i, SUM(a.v * s.v) AS v FROM a JOIN (SELECT k, SUM(v) AS v FROM b GROUP BY k) s "
        "ON a.k = s.k GROUP BY a.i",
    "contraction order as CTEs":
        "WITH t0 AS (SELECT a.i, b.j, SUM(a.v * b.v) AS v FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j) "
        "SELECT i, SUM(v) AS v FROM t0 GROUP BY i",
    "partial aggregate with matched count":
        "SELECT a.i, SUM(a.v * w.v) AS v, COUNT(a.v * w.v) AS matched FROM a JOIN w ON a.k = w.k GROUP BY a.i",
    "mask operand (causal predicate)":
        "SELECT a.i, SUM(a.v * b.v) AS v FROM a JOIN b ON a.k = b.k WHERE b.j <= a.i GROUP BY a.i",
    "single-table reduction (sum over k)":
        "SELECT i, SUM(v) AS v FROM a GROUP BY i",
    "single-table reduction, exact type (BIGINT)":
        "SELECT i, SUM(n) AS v FROM a GROUP BY i",
    "retiling partition key":
        "SELECT i // 64 AS i_blk, SUM(v) AS v FROM a GROUP BY i // 64",
    "window maximum (softmax)":
        "SELECT i, k, v, MAX(v) OVER (PARTITION BY i) AS m FROM a",
}

con = gpudb.connect()
print("build:", con.extension_note if isinstance(con.extension_note, str) else con.extension_note())
raw = con._raw if hasattr(con, "_raw") else con
con.execute(f"CREATE TABLE a AS SELECT (range // 1000)::BIGINT AS i, (range % 100)::BIGINT AS k, "
            f"random() AS v, (range % 7)::BIGINT AS n FROM range({N})")
con.execute("CREATE TABLE b AS SELECT (range // 50)::BIGINT AS k, (range % 50)::BIGINT AS j, "
            "random() AS v FROM range(5000)")
con.execute("CREATE TABLE w AS SELECT range::BIGINT AS k, random() AS v FROM range(100)")

print("| Shape | Rewritten | Reason | Detail |")
print("|---|---|---|---|")
for name, sql in SHAPES.items():
    d = {}
    for _ in range(3):  # first sight schedules residency; let idle uploads run
        con.execute(sql).fetchall()
        d = con.last_rewrite() if callable(con.last_rewrite) else con.last_rewrite
        time.sleep(0.2)
    detail = d.get("detail") or d.get("why") or ""
    print(f"| {name} | {d.get('rewritten')} | {d.get('reason', '')} | {str(detail)[:120]} |")
print("\nlast decision, full:", json.dumps(d, default=str)[:600])
