# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Runs a script from `m2-gate duck-sql` with DuckDB's Python package, and
prints the time of its query and its result, as the DuckDB CLI's `.timer`
would."""

import sys
import time

import duckdb

script = open(sys.argv[1]).read()
con = duckdb.connect()
print(f"duckdb {duckdb.__version__}")
for statement in script.split(";\n"):
    # The CLI's dot commands (`.timer`) aren't SQL: drop them.
    lines = [line for line in statement.splitlines() if not line.startswith(".")]
    statement = "\n".join(lines).strip().rstrip(";")
    if not statement:
        continue
    try:
        start = time.perf_counter()
        result = con.execute(statement).fetchall()
        elapsed = time.perf_counter() - start
    except duckdb.Error as e:
        print(f"error: {str(e)[:200]}")
        sys.exit(1)
    if statement.startswith("CREATE TEMP TABLE out"):
        print(f"Run Time (s): real {elapsed:.3f}")
    elif statement.startswith("SELECT count"):
        print(f"{result[0][0]},{result[0][1]}")
