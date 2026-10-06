# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike S17: write einsum instances for the Rust driver.

Uses the Einsum Benchmark's own generators (pip install einsum_benchmark) for
the benchmark's instance families, at increasing sizes. Each line is

    name|operand indices, comma-separated|output indices|index=extent ...

with each index renamed to an ASCII name.
"""

import sys

import numpy as np
from einsum_benchmark import generators as g


def write(out, name, fmt, shapes):
    lhs, rhs = fmt.split("->")
    ops = lhs.split(",")
    names = {}
    ident = lambda c: names.setdefault(c, f"i{len(names)}")  # noqa: E731
    sizes = {}
    for op, shape in zip(ops, shapes):
        shape = shape if isinstance(shape, tuple) else np.shape(shape)  # some generators return arrays
        for c, s in zip(op, shape):
            sizes[ident(c)] = s
    for c in rhs:
        ident(c)
    out.write("|".join([
        name,
        ",".join(" ".join(ident(c) for c in op) for op in ops),
        " ".join(ident(c) for c in rhs),
        " ".join(f"{k}={v}" for k, v in sizes.items()),
    ]) + "\n")


cases = []
for n in (10, 25, 50, 100, 200):
    cases.append((f"matrix_chain n={n}", *g.structured.matrix_chain(num_matrices=n, seed=0)[:2]))
for n in (10, 26, 50, 100, 250, 500, 1000):
    cases.append((f"random 3-regular n={n}", *g.random.regular(n, 3, seed=0)[:2]))
for k in (4, 8, 16, 24):
    cases.append((f"lattice {k}x{k}", *g.structured.lattice((k, k), seed=0)[:2]))
for n in (100, 1000):
    cases.append((f"tree n={n}", *g.structured.tree(n=n, seed=1)[:2]))
cases.append(("mps n=100", *g.quantum_computing.matrix_product_state(n=100, seed=0)[:2]))
cases.append(("maxcut n=24 p=3", *g.quantum_computing.maxcut(n=24, reg=3, p=3, seed=1)[:2]))
for depth in (2, 3, 4):
    cases.append((f"language model depth={depth}", *g.language_model.p_first_and_last(depth, 8, 16)[:2]))
for n in (50, 200, 1000):
    fmt, shapes = g.random.connected_hypernetwork(
        n, 2.5, max_edge_order=4, number_of_output_indices=4, seed=0)[:2]
    cases.append((f"random hypernetwork n={n}", fmt, shapes))

with open(sys.argv[1] if len(sys.argv) > 1 else "instances.txt", "w") as out:
    for name, fmt, shapes in cases:
        write(out, name, fmt, shapes)
        print(name, len(shapes), "operands", file=sys.stderr)
