# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike S3: check the layout propagation rules (design doc §13.5) against real plans.

A dense 1000 x 1000 array `a(i, j, v)` is written to Parquet in row-major
order. For each operator in the rules table, the script runs the query in
DataFusion and DuckDB and checks three things:

- support: are the output rows exactly the coordinates the rule predicts?
- order: do they come out in the order the predicted layout says
  (row-major over the remaining dimensions)?
- known order: given `ORDER BY` in that order, does the host still sort, or
  does its planner already know the order (no sort operator in the plan)?
"""

import pathlib

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datafusion import SessionContext

N = 1000
tmp = pathlib.Path(__file__).parent / "fixture"
tmp.mkdir(exist_ok=True)
i, j = np.divmod(np.arange(N * N), N)
v = np.random.default_rng(0).random(N * N)
pq.write_table(pa.table({"i": i, "j": j, "v": v}), tmp / "a.parquet", row_group_size=65536)
bi, bj = np.divmod(np.arange(N * 8), 8)
pq.write_table(pa.table({"i": bi, "j": bj, "v": np.ones(N * 8)}), tmp / "b.parquet")

# Each case: SQL, the dimension columns of the result in layout order, and the
# predicted support as a set of coordinate tuples.
full = {(a, b) for a in range(N) for b in range(N)}
CASES = {
    "filter on a dimension range": (
        "SELECT i, j, v FROM a WHERE i >= 100 AND i < 300", ["i", "j"],
        {(a, b) for a in range(100, 300) for b in range(N)}),
    "strided slice": (
        "SELECT i, j, v FROM a WHERE i % 4 = 0", ["i", "j"],
        {(a, b) for a in range(0, N, 4) for b in range(N)}),
    "filter dimension = constant": (
        "SELECT j, v FROM a WHERE i = 7", ["j"], {(b,) for b in range(N)}),
    "filter on a value column": (
        "SELECT i, j, v FROM a WHERE v > 0.5", ["i", "j"], full),  # superset rule
    "projection dropping a value column": ("SELECT i, j FROM a", ["i", "j"], full),
    "transpose (reorder dimension columns)": ("SELECT j, i, v FROM a", ["i", "j"], full),
    "union along a dimension": (
        "SELECT i, j, v FROM a WHERE i < 500 UNION ALL SELECT i, j, v FROM a WHERE i >= 500",
        ["i", "j"], full),
    "contraction node": (
        "SELECT a.i, b.j, sum(a.v * b.v) AS v FROM a JOIN b ON a.j = b.i GROUP BY a.i, b.j",
        ["i", "j"], {(a, b) for a in range(N) for b in range(8)}),
}


def check(rows, dims, predicted, exact):
    coords = list(zip(*(rows[d] for d in dims)))
    support = set(coords) == predicted if exact else set(coords) <= predicted
    keys = np.ravel_multi_index(tuple(np.asarray(rows[d]) for d in dims),
                                tuple(N for _ in dims)) if dims else np.zeros(0)
    ordered = bool(np.all(np.diff(keys) > 0))
    return support, ordered


def datafusion_run():
    ctx = SessionContext()
    # Declare the files' order. (In Python, `file_sort_order=[[col("i").sort()]]`
    # declares NULLS FIRST, which doesn't match `ORDER BY i`; the DDL form does.)
    for t in ("a", "b"):
        ctx.sql(f"CREATE EXTERNAL TABLE {t} STORED AS PARQUET LOCATION '{tmp / (t + '.parquet')}' "
                "WITH ORDER (i ASC, j ASC)")
    out = {}
    for name, (sql, dims, predicted) in CASES.items():
        rows = ctx.sql(sql).to_pydict()
        support, ordered = check(rows, dims, predicted, "value" not in name)
        order_by = ", ".join(dims)
        plan = "\n".join(ctx.sql(f"EXPLAIN {sql} ORDER BY {order_by}").to_pydict()["plan"])
        out[name] = (support, ordered, "SortExec" not in plan)
    return out


def duckdb_run():
    con = duckdb.connect()
    con.sql(f"CREATE VIEW a AS SELECT * FROM '{tmp / 'a.parquet'}'")
    con.sql(f"CREATE VIEW b AS SELECT * FROM '{tmp / 'b.parquet'}'")
    out = {}
    for name, (sql, dims, predicted) in CASES.items():
        rows = con.sql(sql).fetchnumpy()
        support, ordered = check(rows, dims, predicted, "value" not in name)
        order_by = ", ".join(dims)
        plan = con.sql(f"EXPLAIN {sql} ORDER BY {order_by}").fetchall()[0][1]
        out[name] = (support, ordered, "ORDER_BY" not in plan)
    return out


def yn(b):
    return "yes" if b else "**no**"


df, dd = datafusion_run(), duckdb_run()
print(f"DuckDB threads: {duckdb.sql('SELECT current_setting(\'threads\')').fetchone()[0]}; "
      f"DataFusion target partitions: {SessionContext().sql('SHOW datafusion.execution.target_partitions').to_pydict()['value'][0]}")
print("| Rule | Support (DF / DuckDB) | Row order (DF / DuckDB) | Order known to planner (DF / DuckDB) |")
print("|---|---|---|---|")
for name in CASES:
    print(f"| {name} | {yn(df[name][0])} / {yn(dd[name][0])} | {yn(df[name][1])} / {yn(dd[name][1])} "
          f"| {yn(df[name][2])} / {yn(dd[name][2])} |")
