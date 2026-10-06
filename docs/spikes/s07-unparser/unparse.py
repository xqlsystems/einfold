# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike S7: is DataFusion's Unparser good enough to hand rewritten plans to DuckDB?

einfold's SQL-to-SQL mode would rewrite a DataFusion logical plan, then turn it
back into SQL for another engine. This checks that last step. For each query:

1. plan it in DataFusion, both before and after DataFusion's optimizer;
2. unparse each plan with the DuckDB dialect;
3. run the SQL in DuckDB and compare with DataFusion's own result.

Queries: the shapes einfold emits in its relational form (section 10.1 of the
design doc), and all 22 TPC-H queries as a control.

Usage: python unparse.py <folder with TPC-H q1.sql ... q22.sql>
"""

import decimal
import math
import pathlib
import sys

import duckdb
import numpy as np
import pyarrow as pa
from datafusion import SessionContext
from datafusion.unparser import Dialect, Unparser

con = duckdb.connect()
con.sql("INSTALL tpch; LOAD tpch; CALL dbgen(sf = 0.01)")
ctx = SessionContext()
for (t,) in con.sql("SHOW TABLES").fetchall():
    ctx.register_record_batches(t, [con.sql(f"SELECT * FROM {t}").to_arrow_table().to_batches()])

# Small einsum operands: A(i, k), B(k, j), w(k), with a NULL and a duplicate-free key.
rng = np.random.default_rng(0)
ii, kk = np.meshgrid(np.arange(6), np.arange(5), indexing="ij")
A = pa.table({"i": ii.ravel(), "k": kk.ravel(), "v": rng.random(30)})
kk2, jj = np.meshgrid(np.arange(5), np.arange(4), indexing="ij")
B = pa.table({"k": kk2.ravel(), "j": jj.ravel(), "v": rng.random(20)})
W = pa.table({"k": np.arange(5), "v": pa.array([1.0, None, 3.0, 4.0, 5.0])})
for name, t in {"a": A, "b": B, "w": W}.items():
    ctx.register_record_batches(name, [t.to_batches()])
    con.register(name, t)

EINFOLD = {
    "matmul: join + GROUP BY + SUM":
        "SELECT a.i, b.j, SUM(a.v * b.v) AS v FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j",
    "eager aggregation: pre-aggregated subquery":
        "SELECT a.i, SUM(a.v * s.v) AS v FROM a JOIN (SELECT k, SUM(v) AS v FROM b GROUP BY k) s "
        "ON a.k = s.k GROUP BY a.i",
    "contraction order as CTEs":
        "WITH t0 AS (SELECT a.i, b.j, SUM(a.v * b.v) AS v FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j), "
        "t1 AS (SELECT t0.i, SUM(t0.v) AS v FROM t0 GROUP BY t0.i) SELECT i, v FROM t1",
    "partial aggregate with matched count (NULL-aware)":
        "SELECT a.i, SUM(a.v * w.v) AS v, COUNT(a.v * w.v) AS matched FROM a JOIN w ON a.k = w.k GROUP BY a.i",
    "mask operand: causal predicate":
        "SELECT a.i, b.j, SUM(a.v * b.v) AS v FROM a JOIN b ON a.k = b.k WHERE b.j <= a.i GROUP BY a.i, b.j",
    "partial aggregates combined with UNION ALL":
        "SELECT i, SUM(v) AS v FROM (SELECT i, SUM(v) AS v FROM a WHERE k < 2 GROUP BY i "
        "UNION ALL SELECT i, SUM(v) AS v FROM a WHERE k >= 2 GROUP BY i) p GROUP BY i",
    "retiling as a partition key":
        "SELECT i / 2 AS i_blk, k / 2 AS k_blk, SUM(v) AS v FROM a GROUP BY i / 2, k / 2",
    "online softmax pieces: window max":
        "SELECT i, k, v, MAX(v) OVER (PARTITION BY i) AS m FROM a",
    "softmax normalizer":
        "SELECT a.i, SUM(EXP(a.v - m.m)) AS z FROM a JOIN (SELECT i, MAX(v) AS m FROM a GROUP BY i) m "
        "ON a.i = m.i GROUP BY a.i",
    "semi-join reduction":
        "SELECT a.i, SUM(a.v) AS v FROM a WHERE a.k IN (SELECT k FROM w WHERE v IS NOT NULL) GROUP BY a.i",
    "scale factor (broadcast factoring)":
        "SELECT SUM(v) * (SELECT COUNT(DISTINCT j) FROM b) AS v FROM w",
    "ordered output with LIMIT":
        "SELECT a.i, b.j, SUM(a.v * b.v) AS v FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j ORDER BY v DESC LIMIT 5",
}


def rows(table):
    """Order-insensitive comparable form. Floats and decimals are rounded to 4 places,
    because the hosts type AVG differently (DataFusion: truncated decimal; DuckDB: double)."""
    out = []
    for r in table.to_pylist():
        vals = [float(x) if isinstance(x, decimal.Decimal) else x for x in r.values()]
        out.append(tuple(round(x, 4) if isinstance(x, float) and not math.isnan(x) else x for x in vals))
    return sorted(out, key=repr)


def check(name, sql, ordered=False):
    df = ctx.sql(sql)
    expected = df.to_arrow_table()
    results = {}
    unparser = Unparser(Dialect.duckdb())
    for which, plan in (("unoptimized", df.logical_plan()), ("optimized", df.optimized_logical_plan())):
        try:
            text = unparser.plan_to_sql(plan)
        except Exception as e:  # report and continue
            results[which] = ("unparse error", str(e).splitlines()[0][:110])
            continue
        try:
            got = con.sql(text).to_arrow_table()
        except Exception as e:  # report and continue
            results[which] = ("DuckDB error", str(e).splitlines()[0][:110])
            continue
        if rows(got) == rows(expected):
            # Integer and decimal widths differ routinely; report only float vs. not.
            kinds = [f"{a.name}: {a.type} vs {b.type}" for a, b in zip(got.schema, expected.schema)
                     if pa.types.is_floating(a.type) != pa.types.is_floating(b.type)]
            results[which] = ("ok, types differ", "; ".join(kinds)) if kinds else ("ok", "")
        else:
            results[which] = ("**wrong result**", f"{got.num_rows} rows vs {expected.num_rows}")
    return results


print("| Query | Unoptimized plan | Optimized plan | Detail |")
print("|---|---|---|---|")
summary = {"unoptimized": {}, "optimized": {}}
cases = list(EINFOLD.items())
for i in range(1, 23):
    text = (pathlib.Path(sys.argv[1]) / f"q{i}.sql").read_text()
    stmts = [s for s in text.split(";") if s.strip()]
    q = max(stmts, key=lambda s: s.lower().count("select"))
    if i == 15:  # a view, a query and a drop: inline the view
        q = stmts[1].replace("revenue0", "(select l_suppkey as supplier_no, sum(l_extendedprice * (1 - l_discount)) "
                             "as total_revenue from lineitem where l_shipdate >= date '1996-01-01' and l_shipdate "
                             "< date '1996-04-01' group by l_suppkey) revenue0")
    cases.append((f"TPC-H Q{i}", q))
for name, sql in cases:
    r = check(name, sql)
    for which in r:
        summary[which][r[which][0]] = summary[which].get(r[which][0], 0) + 1
    detail = "; ".join(f"{w}: {d}" for w, (s, d) in r.items() if d)
    print(f"| {name} | {r['unoptimized'][0]} | {r['optimized'][0]} | {detail} |")
print()
for which, counts in summary.items():
    print(f"{which}: " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
