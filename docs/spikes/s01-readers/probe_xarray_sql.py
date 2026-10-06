# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike S1/S13/S14 probe for xarray-sql: what reaches DataFusion from a Zarr store."""

import pathlib

import xarray as xr
from xarray_sql import XarrayContext

here = pathlib.Path(__file__).parent
ds = xr.open_zarr(here / "fixture" / "v3.zarr")
ctx = XarrayContext()
ctx.from_dataset("ds", ds)


def q(sql):
    return ctx.sql(sql).to_pandas()


print("## Tables registered")
cat = ctx.catalog()
for schema_name in cat.names():
    for t in cat.schema(schema_name).names():
        print(f"{schema_name}.{t}")

tables = [f"{s}.{t}" for s in cat.names() for t in cat.schema(s).names()]
for t in sorted(tables):
    print(f"\n## Table `{t}`")
    sch = ctx.table(t).schema()
    print("schema metadata:", sch.metadata)
    for f in sch:
        print(f"  {f.name}: {f.type}, nullable={f.nullable}, metadata={f.metadata}")
    print("rows:", q(f"SELECT count(*) AS n FROM {t}")["n"][0])

data_table = [t for t in tables if "t2m" in [f.name for f in ctx.table(t).schema()]][0]
print(f"\n## Fill values in `{data_table}` (one t2m chunk was never written)")
print(q(f"""SELECT count(*) AS rows,
                   sum(CASE WHEN t2m IS NULL THEN 1 ELSE 0 END) AS t2m_null,
                   sum(CASE WHEN isnan(t2m) THEN 1 ELSE 0 END) AS t2m_nan,
                   sum(CASE WHEN x = 0 THEN 1 ELSE 0 END) AS x_zero
            FROM {data_table}""").to_string(index=False))

print("\n## Plan with statistics, and partition pruning on a dimension filter")
ctx.sql("SET datafusion.explain.show_statistics = true")
for sql in (f"SELECT * FROM {data_table}", f"SELECT * FROM {data_table} WHERE lat > 10"):
    print(f"\n-- EXPLAIN {sql}")
    plan = ctx.sql(f"EXPLAIN {sql}").to_pandas()
    print(plan[plan.plan_type == "physical_plan"]["plan"].iloc[0])
