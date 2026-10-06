# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike S2: does Arrow field metadata survive relational operators and Substrait?

Registers two Arrow tables whose fields carry metadata (as an einfold fact
would), runs one query per operator in DataFusion and DuckDB, and reports
which output columns still carry metadata. Then round-trips a plan through
DataFusion's Substrait producer and consumer.
"""

import duckdb
import pyarrow as pa
from datafusion import SessionContext
from datafusion import substrait as ss

A_META = {b"einfold.fact": b"dims=i,j;layout=row-major"}
SCHEMA_META = {b"einfold.table": b"zarr:fixture/v3.zarr"}


def table(name):
    fields = [
        pa.field("i", pa.int64(), metadata={b"einfold.dim": b"i"}),
        pa.field("j", pa.int64(), metadata={b"einfold.dim": b"j"}),
        pa.field("v", pa.float64(), metadata=A_META),
    ]
    sch = pa.schema(fields, metadata=SCHEMA_META)
    return pa.table({"i": [0, 0, 1, 1], "j": [0, 1, 0, 1], "v": [1.0, 2.0, 3.0, 4.0]}, schema=sch)


QUERIES = {
    "scan": "SELECT * FROM a",
    "projection (column)": "SELECT i, v FROM a",
    "projection (renamed)": "SELECT i AS row, v AS val FROM a",
    "projection (expression)": "SELECT i, v * 2 AS v2 FROM a",
    "filter": "SELECT * FROM a WHERE v > 1",
    "sort": "SELECT * FROM a ORDER BY v DESC",
    "limit": "SELECT * FROM a LIMIT 2",
    "join": "SELECT a.i, b.j, a.v, b.v AS bv FROM a JOIN b ON a.j = b.i",
    "aggregate": "SELECT i, sum(v) AS s FROM a GROUP BY i",
    "join + aggregate (einsum)": "SELECT a.i, b.j, sum(a.v * b.v) AS v FROM a JOIN b ON a.j = b.i GROUP BY a.i, b.j",
    "union all": "SELECT * FROM a UNION ALL SELECT * FROM b",
    "window": "SELECT i, j, v, max(v) OVER (PARTITION BY i) AS m FROM a",
}


def describe(schema):
    """Which output columns carry field metadata, and is schema metadata kept?"""
    kept = [f.name for f in schema if f.metadata]
    return f"fields with metadata: {', '.join(kept) or '—'}; schema metadata: {'yes' if schema.metadata else 'no'}"


def datafusion_results():
    ctx = SessionContext()
    ctx.register_record_batches("a", [table("a").to_batches()])
    ctx.register_record_batches("b", [table("b").to_batches()])
    out = {}
    for name, sql in QUERIES.items():
        df = ctx.sql(sql)
        out[name] = (describe(df.schema()), describe(df.to_arrow_table().schema))
    return ctx, out


def duckdb_results():
    con = duckdb.connect()
    a, b = table("a"), table("b")  # noqa: F841 (referenced by name in SQL)
    out = {}
    for name, sql in QUERIES.items():
        out[name] = describe(con.sql(sql).to_arrow_table().schema)
    return out


ctx, df_out = datafusion_results()
dd_out = duckdb_results()
print("| Operator | DataFusion plan schema | DataFusion result | DuckDB result |")
print("|---|---|---|---|")
for name in QUERIES:
    print(f"| {name} | {df_out[name][0]} | {df_out[name][1]} | {dd_out[name]} |")

print("\n## DataFusion: inputs whose metadata disagree, and value-changing operators")
ctx.register_record_batches("c", [table("c").replace_schema_metadata({b"einfold.table": b"other"}).to_batches()])
cmeta = pa.schema([f.with_metadata({b"einfold.fact": b"from-c"}) for f in table("c").schema])
ctx.deregister_table("c")
ctx.register_record_batches("c", [table("c").cast(cmeta.with_metadata({b"einfold.table": b"other"})).to_batches()])
for sql in (
    "SELECT * FROM a UNION ALL SELECT * FROM c",
    "SELECT a.v, c.v AS cv FROM a JOIN c ON a.i = c.i",
    "SELECT CAST(v AS float) AS v FROM a",
    "SELECT coalesce(v, 0) AS v FROM a",
):
    sch = ctx.sql(sql).to_arrow_table().schema
    fields = {f.name: (f.metadata or {}).get(b"einfold.fact") for f in sch}
    print(f"-- {sql}\n   field facts: {fields}; schema metadata: {sch.metadata}")

print("\n## DuckDB: does a canonical Arrow extension type survive where plain metadata does not?")
uu = pa.table({"u": pa.ExtensionArray.from_storage(pa.uuid(), pa.array([b"0123456789abcdef"], pa.binary(16)))})
for lossless in (False, True):
    con = duckdb.connect()
    con.sql(f"SET arrow_lossless_conversion = {str(lossless).lower()}")
    a = table("a")  # noqa: F841
    print(f"   arrow_lossless_conversion={lossless}: uuid column -> {con.sql('SELECT * FROM uu').to_arrow_table().schema.field('u').type};",
          describe(con.sql("SELECT * FROM a").to_arrow_table().schema))

print("\n## Substrait round trip (DataFusion producer, then consumer)")
for sql in ("SELECT * FROM a", QUERIES["join + aggregate (einsum)"]):
    plan = ss.Producer.to_substrait_plan(ctx.sql(sql).logical_plan(), ctx)
    raw = plan.encode()
    back = ss.Consumer.from_substrait_plan(ctx, plan)
    print(f"-- {sql}")
    print(f"   plan bytes: {len(raw)}; 'einfold' in plan bytes: {b'einfold' in raw}")
    print(f"   after round trip: {describe(ctx.create_dataframe_from_logical_plan(back).schema())}")
