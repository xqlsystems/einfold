# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

import glob, os, sys, time, statistics, subprocess
from datafusion import SessionContext
from einfold.datafusion import EinFold
here = os.path.dirname(os.path.abspath(__file__)); data = sys.argv[1]
c = SessionContext().with_extensions(EinFold())
for f in glob.glob(f"{data}/*.parquet"): c.register_parquet(os.path.basename(f)[:-8], f)
for q in sorted(glob.glob(f"{here}/q*.sql")):
    sql = open(q).read()
    print("Q", os.path.basename(q)[:3], file=sys.stderr, flush=True)
    for _ in range(9): c.sql(sql).execution_plan()
