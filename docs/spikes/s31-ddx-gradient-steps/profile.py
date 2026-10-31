# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike S31: profile each step of ddx's gradient programs in xarray-sql's ML demos.

Runs one of xarray-sql's demos (``benchmarks/ml/01_attention.py`` or
``02_nanogpt.py``) unchanged, with ddxdb's DataFusion backend wrapped so
that every program step it materializes is recorded: its name, its rows,
the time to plan it and to run it, whether einfold planned any of it
(and as a fused pipeline), and its optimized logical plan, written to a
file per step for reading later.

    python profile.py OUT_DIR DEMO.py [demo arguments...]

Each host gets its own directory under OUT_DIR, with ``steps.jsonl`` and
the plans. Two settings record more:

* ``S31_ANALYZE=1``: each step's ``EXPLAIN ANALYZE``, which runs it again;
* ``S31_DUMP=1``: each step's rows as Parquet, the demo's own tables as it
  first loads them, and each step's SQL, for ``src/main.rs`` to run again
  from Rust.
"""

from __future__ import annotations

import json
import os
import re
import runpy
import sys
import time
from pathlib import Path

OUT = Path(sys.argv[1])
DEMO = Path(sys.argv[2]).resolve()
sys.argv = [str(DEMO), *sys.argv[3:]]
sys.path.insert(0, str(DEMO.parent))

import _hosts  # noqa: E402  (the demo's own module, found beside it)
from ddxdb import ad as ddx_ad  # noqa: E402

# Which host each SessionContext belongs to, and its einfold extension.
HOST_OF: dict[int, tuple[str, object]] = {}


def _track(cls):
    init = cls.__init__

    def tracked(self, *args, **kwargs):
        init(self, *args, **kwargs)
        HOST_OF[id(self.ctx)] = (self.name, getattr(self, "einfold", None))

    cls.__init__ = tracked


for cls in (_hosts.DataFusion, _hosts.EinFold):
    _track(cls)

COUNTS: dict[str, int] = {}


def _shape(plan: str) -> dict:
    """Operator counts in a logical plan's indented display."""
    ops = [line.strip().split(":")[0].split(" ")[0] for line in plan.splitlines()]
    joins = [line for line in plan.splitlines() if line.strip().startswith(("Inner Join", "Left Join", "Right Join", "Full Join", "LeftSemi", "LeftAnti", "Cross Join"))]
    return {
        "scans": sum(op == "TableScan" for op in ops),
        "inner_joins": sum("Inner Join" in j for j in joins),
        "left_joins": sum("Left Join" in j for j in joins),
        "cross_joins": sum("Cross Join" in j for j in joins),
        "other_joins": sum(not any(k in j for k in ("Inner Join", "Left Join", "Cross Join")) for j in joins),
        "aggregates": sum(op == "Aggregate" for op in ops),
        "windows": sum(op == "WindowAggr" for op in ops),
        "unions": sum(op == "Union" for op in ops),
        "is_null_tests": len(re.findall(r"IS NULL", plan)),
        "lines": len(ops),
    }


def _record(backend, name: str, plan: bytes, run) -> None:
    """Runs a step by ``run``, which returns its rows as a table, and
    records it."""
    host, einfold = HOST_OF.get(id(backend.ctx), ("unknown", None))
    out = OUT / host
    out.mkdir(parents=True, exist_ok=True)
    before = (einfold.plans_with_einfold(), einfold.plans_with_pipelines()) if einfold else (0, 0)
    df = ddx_ad._dataframe(backend.ctx, plan)
    logical = df.optimized_logical_plan().display_indent()
    # Planning alone, then the step as ddxdb runs it (planned again, then
    # run on every partition at once); running is the difference.
    t0 = time.perf_counter()
    df.execution_plan()
    t1 = time.perf_counter()
    table = run(df)
    t2 = time.perf_counter()
    t1, t2 = t1 - t0, t2 - t1
    after = (einfold.plans_with_einfold(), einfold.plans_with_pipelines()) if einfold else (0, 0)
    index = COUNTS[host] = COUNTS.get(host, 0) + 1
    (out / f"{index:04d}.plan").write_text(f"-- {name}\n{logical}\n")
    if os.environ.get("S31_ANALYZE"):
        # EXPLAIN ANALYZE, which datafusion-python prints from Rust, so it
        # goes to the process's stdout, not Python's.
        sys.stdout.flush()
        saved = os.dup(1)
        with open(out / f"{index:04d}.analyze", "w") as f:
            os.dup2(f.fileno(), 1)
            try:
                df.explain(False, True)
            except Exception as e:  # noqa: BLE001 - noted, not fatal
                print(f"explain failed: {e}")
            finally:
                os.dup2(saved, 1)
                os.close(saved)
    if os.environ.get("S31_DUMP"):
        # The step's SQL and its rows, to run again from Rust.
        import pyarrow.parquet as pq

        try:
            (out / f"{index:04d}.sql").write_text(ddx_ad.to_sql(plan, "datafusion"))
        except Exception as e:  # noqa: BLE001 - noted, not fatal
            (out / f"{index:04d}.sql").write_text(f"-- to_sql failed: {e}")
        pq.write_table(table, out / f"{name}.parquet")
    entry = {
        "index": index,
        "name": name,
        "rows": table.num_rows,
        "plan_ms": round(1e3 * t1, 2),
        "run_ms": round(1e3 * max(t2 - t1, 0.0), 2),
        "einfold": after[0] > before[0],
        "pipeline": after[1] > before[1],
        **_shape(logical),
    }
    with open(out / "steps.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    return table


def materialize(self, name, plan):
    """``DataFusionBackend.materialize``, recorded: the step's rows,
    registered under its name."""
    table = _record(self, name, plan, lambda df: df.to_arrow_table())
    ddx_ad._register(self.ctx, name, table)


ddx_ad.DataFusionBackend.materialize = materialize

if __name__ == "__main__":
    if os.environ.get("S31_DUMP"):
        # The demo's own tables, as each host loads them.
        load = _hosts.DataFusion.load

        def dumping(self, name, data, **options):
            load(self, name, data, **options)
            import pyarrow.parquet as pq

            (OUT / self.name).mkdir(parents=True, exist_ok=True)
            # Only the first load of each name: the demos load their main
            # size first, and smaller checks after.
            if (OUT / self.name / f"{name}.parquet").exists():
                return
            pq.write_table(self.ctx.table(name).to_arrow_table(), OUT / self.name / f"{name}.parquet")

        _hosts.DataFusion.load = dumping
    runpy.run_path(str(DEMO), run_name="__main__")
