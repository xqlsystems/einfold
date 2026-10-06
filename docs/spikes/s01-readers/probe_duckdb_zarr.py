# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike S1/S13/S14 probe for duckdb-zarr v0.1.3, DuckDB's `zarr` community extension.

Run with: uv run --with duckdb==1.5.5 python probe_duckdb_zarr.py [path to zarr.duckdb_extension]
Without a path, the extension is installed from DuckDB's community repository.
(As of 2026-10-06 it is published for DuckDB 1.5.4 and 1.5.5, not yet 1.5.6.)
"""

import pathlib
import sys

import duckdb

ext = sys.argv[1] if len(sys.argv) > 1 else None
here = pathlib.Path(__file__).parent
store = str(here / "fixture" / "v3.zarr")
main = f"read_zarr('{store}', dims := ['time', 'lat', 'lon'])"
weights = f"read_zarr('{store}', dims := ['lat'])"
if ext:
    con = duckdb.connect(config={"allow_unsigned_extensions": "true"})
    con.sql(f"LOAD '{ext}'")
else:
    con = duckdb.connect()
    con.sql("INSTALL zarr FROM community; LOAD zarr")


def show(title, sql):
    print(f"\n## {title}\n-- {sql}")
    try:
        print(con.sql(sql))
    except Exception as e:  # report and continue
        print("ERROR:", str(e)[:300])


show("Array metadata", f"SELECT * FROM read_zarr_metadata('{store}')")
show("Groups", f"SELECT * FROM read_zarr_groups('{store}')")
show("Schema of the main group", f"DESCRIBE SELECT * FROM {main}")
show("Schema of the lat group", f"DESCRIBE SELECT * FROM {weights}")
show("The lat group", f"SELECT * FROM {weights}")
show("Fill values and zeros",
     f"""SELECT count(*) AS rows,
               count(*) FILTER (WHERE t2m IS NULL) AS t2m_null,
               count(*) FILTER (WHERE isnan(t2m)) AS t2m_nan,
               count(*) FILTER (WHERE x = 0) AS x_zero
        FROM {main}""")

show("Plan for a dimension filter", f"EXPLAIN SELECT count(*) FROM {main} WHERE lat > 10")
