# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

import glob, os, sys, time, statistics
from datafusion import SessionContext
from einfold.datafusion import EinFold
data = sys.argv[1]; qs = sys.argv[2:]
here=os.path.dirname(os.path.abspath(__file__))
def mk(e):
    c = SessionContext()
    if e: c = c.with_extensions(EinFold())
    for f in glob.glob(f"{data}/*.parquet"): c.register_parquet(os.path.basename(f)[:-8], f)
    return c
cs = [mk(False), mk(True)]
for q in qs:
    sql = open(f"{here}/{q}.sql").read()
    res=[]
    for c in cs:
        def run():
            plan = c.sql(sql).execution_plan()
            s = time.perf_counter()
            for p in range(plan.partition_count):
                for b in c.execute(plan, p): pass
            return time.perf_counter() - s
        for _ in range(3): run()
        t=[run() for _ in range(15)]
        n = c.sql(sql).execution_plan().partition_count
        res.append(statistics.median(t)*1e3)
    print(f"{q} partitions={n} exec off {res[0]:.1f} ms, einfold {res[1]:.1f} ms ({res[1]-res[0]:+.1f})", flush=True)
