# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike S18: how long do the hosts themselves take to plan TPC-H?

Generates TPC-H at scale factor 0.01 with DuckDB's tpch extension, registers
the same tables in DataFusion, and times planning only (no execution):

- DataFusion: SQL to optimized physical plan (`ctx.sql(q).execution_plan()`).
- DuckDB: `EXPLAIN q`, which parses, binds, optimizes and builds the physical plan.

Each query is planned 25 times; the median is reported.

Usage: python host_planning.py <folder with q1.sql ... q22.sql>
(the queries from apache/datafusion's benchmarks/queries).
"""

import pathlib
import statistics
import sys
import time

import duckdb
from datafusion import SessionContext

queries = pathlib.Path(sys.argv[1])
con = duckdb.connect()
con.sql("INSTALL tpch; LOAD tpch; CALL dbgen(sf = 0.01)")
ctx = SessionContext()
for (t,) in con.sql("SHOW TABLES").fetchall():
    ctx.register_record_batches(t, [con.sql(f"SELECT * FROM {t}").to_arrow_table().to_batches()])


def median_ms(f, n=25):
    f()  # warm up
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        f()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


print("| Query | DataFusion (ms) | DuckDB (ms) |")
print("|---|---|---|")
df_all, dd_all = [], []
for i in range(1, 23):
    sql = (queries / f"q{i}.sql").read_text()
    # q15 is a view definition, a query and a drop; plan its query only.
    stmts = [s for s in sql.split(";") if s.strip()]
    q = max(stmts, key=lambda s: s.lower().count("select"))
    if i == 15:
        q = stmts[1].replace("revenue0", "(select l_suppkey as supplier_no, sum(l_extendedprice * (1 - l_discount)) as total_revenue from lineitem where l_shipdate >= date '1996-01-01' and l_shipdate < date '1996-04-01' group by l_suppkey)")
    df_ms = median_ms(lambda: ctx.sql(q).execution_plan())
    dd_ms = median_ms(lambda: con.execute("EXPLAIN " + q).fetchall())
    df_all.append(df_ms)
    dd_all.append(dd_ms)
    print(f"| Q{i} | {df_ms:.2f} | {dd_ms:.2f} |")
print(f"| median | {statistics.median(df_all):.2f} | {statistics.median(dd_all):.2f} |")
print(f"| max | {max(df_all):.2f} | {max(dd_all):.2f} |")
