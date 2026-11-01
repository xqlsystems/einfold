# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

import glob, os, sys, time, statistics
from datafusion import SessionContext
from einfold.datafusion import EinFold
here = os.path.dirname(os.path.abspath(__file__)); data = sys.argv[1]
def mk(e):
    c = SessionContext()
    if e: c = c.with_extensions(EinFold())
    for f in glob.glob(f"{data}/*.parquet"): c.register_parquet(os.path.basename(f)[:-8], f)
    return c
off, on = mk(False), mk(True)
def med(f, n=15):
    t=[]
    for _ in range(n):
        s=time.perf_counter(); f(); t.append(time.perf_counter()-s)
    return statistics.median(t)*1e3
T=[0]*5
for q in sorted(glob.glob(f"{here}/q*.sql")):
    sql=open(q).read()
    r=[med(lambda: off.sql(sql)), med(lambda: off.sql(sql).optimized_logical_plan()), med(lambda: off.sql(sql).execution_plan()),
       med(lambda: on.sql(sql).optimized_logical_plan()), med(lambda: on.sql(sql).execution_plan())]
    T=[a+b for a,b in zip(T,r)]
print("sum ms: sql=%.1f off_opt=%.1f off_plan=%.1f on_opt=%.1f on_plan=%.1f" % tuple(T))
print("host physical ≈ %.1f ms; einfold's part of planning ≈ %.1f ms" % (T[2]-T[1], T[4]-T[3]))
