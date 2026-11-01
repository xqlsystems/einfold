# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Per TPC-H query: planning (DataFrame.execution_plan) and the whole query, einfold on and off."""
import glob, os, sys, time, statistics
from datafusion import SessionContext
from einfold.datafusion import EinFold
here = os.path.dirname(os.path.abspath(__file__))
data = sys.argv[1]
def ctx(einfold):
    c = SessionContext()
    if einfold: c = c.with_extensions(EinFold())
    for f in glob.glob(f"{data}/*.parquet"):
        c.register_parquet(os.path.basename(f)[:-8], f)
    return c
off, on = ctx(False), ctx(True)
def med(f, n):
    t = []
    for _ in range(n):
        s = time.perf_counter(); f(); t.append(time.perf_counter() - s)
    return statistics.median(t)
tot = [0]*4
for q in sorted(glob.glob(f"{here}/q*.sql")):
    sql = open(q).read()
    for c in (off, on): c.sql(sql).to_arrow_table()
    p = [med(lambda: c.sql(sql).execution_plan(), 9) for c in (off, on)]
    a = [med(lambda: c.sql(sql).to_arrow_table(), 7) for c in (off, on)]
    for i, v in enumerate(p + a): tot[i] += v
    print(f"{os.path.basename(q)[:3]} plan {p[0]*1e3:6.1f} {p[1]*1e3:6.1f} (+{(p[1]-p[0])*1e3:5.1f})  all {a[0]*1e3:6.1f} {a[1]*1e3:6.1f} (+{(a[1]-a[0])*1e3:5.1f})", flush=True)
print(f"total plan {tot[0]*1e3:.0f} {tot[1]*1e3:.0f} (+{(tot[1]-tot[0])*1e3:.0f})  all {tot[2]*1e3:.0f} {tot[3]*1e3:.0f} (+{(tot[3]-tot[2])*1e3:.0f}) ({tot[3]/tot[2]:.3f}x)")
