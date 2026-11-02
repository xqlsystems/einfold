# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

import glob, os, sys, re
from datafusion import SessionContext
from einfold.datafusion import EinFold
data, q = sys.argv[1], sys.argv[2]
here=os.path.dirname(os.path.abspath(__file__))
for e in (False, True):
    c = SessionContext()
    if e: c = c.with_extensions(EinFold())
    for f in glob.glob(f"{data}/*.parquet"): c.register_parquet(os.path.basename(f)[:-8], f)
    sql = open(f"{here}/{q}.sql").read()
    for _ in range(3): c.sql(sql).collect()
    t = str(c.sql("EXPLAIN ANALYZE " + sql).to_arrow_table().column(1)[0])
    print("=== einfold" if e else "=== off")
    for l in t.splitlines():
        name = l.strip().split(':')[0]
        m = re.findall(r'(elapsed_compute|output_rows|row_groups_pruned_statistics|pushdown_rows_pruned|files_ranges_pruned_statistics|time_elapsed_opening|time_elapsed_scanning_total|metadata_load_time|page_index_rows_pruned|bloom_filter\w*|build_time|join_time)=([^,\]]+)', l)
        print(" ", " " * (len(l) - len(l.lstrip()) ), name[:28], ' '.join(f"{k}={v}" for k, v in m))
