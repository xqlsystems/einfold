# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike S20: Cubed's own plans for large matrix multiplications, as ground truth.

For each case, builds `matmul(a, b)` lazily in Cubed (nothing is computed) and
prints each primitive operation in the finalized plan: its name, task count,
projected memory, and output chunks. Cases whose projected memory exceeds the
budget make Cubed raise at planning time, and that is recorded too.
"""

import cubed
import cubed.array_api as xp
import cubed.random

MB = 1_000_000
CASES = [
    # name, shape (n, n), chunks of a, chunks of b, allowed_mem
    ("aligned 5000x2000 / 2000x5000, 2 GB", 20000, (5000, 2000), (2000, 5000), 2000 * MB),
    ("aligned square 2000, 1 GB", 20000, (2000, 2000), (2000, 2000), 1000 * MB),
    ("misaligned k: 2000 vs 3000, 1 GB", 20000, (2000, 2000), (3000, 2000), 1000 * MB),
    ("large chunks 8000, 1 GB (over budget)", 20000, (8000, 8000), (8000, 8000), 1000 * MB),
]

for name, n, ca, cb, mem in CASES:
    print(f"\n## {name}")
    spec = cubed.Spec(work_dir="/tmp/cubed-s20", allowed_mem=mem, reserved_mem=100 * MB)
    try:
        a = cubed.random.random((n, n), chunks=ca, spec=spec)
        b = cubed.random.random((n, n), chunks=cb, spec=spec)
        c = xp.matmul(a, b)
        plan = c.plan()
    except Exception as e:  # Cubed rejects plans over budget when building them
        print(f"rejected: {type(e).__name__}: {str(e)[:200]}")
        continue
    print(f"max projected memory: {plan.max_projected_mem / MB:.0f} MB; tasks: {plan.num_tasks}; "
          f"primitive ops: {plan.num_primitive_ops}")
    for node, d in plan.dag.nodes(data=True):
        op = d.get("primitive_op")
        if op is None:
            continue
        target = getattr(op, "target_array", None)
        chunks = getattr(target, "chunks", None)
        print(f"  {d.get('op_name', node)}: tasks={op.num_tasks}, "
              f"projected_mem={op.projected_mem / MB:.0f} MB, output chunks={chunks}, "
              f"output shape={getattr(target, 'shape', None)}")
