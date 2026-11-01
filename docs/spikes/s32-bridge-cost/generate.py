# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Writes TPC-H at scale factor 1 as Parquet, and its 22 queries, with
DuckDB's tpch extension (DuckDB 1.5.6)."""
import os, duckdb
here = os.path.dirname(os.path.abspath(__file__))
con = duckdb.connect()
con.execute("INSTALL tpch; LOAD tpch; CALL dbgen(sf=1)")
os.makedirs(f"{here}/data", exist_ok=True)
for (t,) in con.execute("SHOW TABLES").fetchall():
    con.execute(f"COPY {t} TO '{here}/data/{t}.parquet' (FORMAT parquet)")
for n, q in con.execute("SELECT query_nr, query FROM tpch_queries()").fetchall():
    open(f"{here}/q{n:02d}.sql", "w").write(q)
