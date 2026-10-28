# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""einfold in DataFusion sessions: each query gives the same rows
with einfold installed as without, and einfold runs the folds and pipelines
it claims to. The algorithms' semantics are tested in einfold's Rust crates;
these test the extension around them: installation, the codecs that pass
tables and plans by reference, and the planner across the FFI."""

from __future__ import annotations

import math

import numpy as np
import pyarrow as pa
import pytest
from datafusion import SessionContext

from einfold.datafusion import EinFold


def _tables(ctx, n=300, d=16, h=8):
    r = np.arange(n * d)
    ctx.register_record_batches("a", [[pa.record_batch(
        {"s": r // d, "k": r % d, "val": np.sin(r * 0.7)})]])
    r = np.arange(d * h)
    ctx.register_record_batches("w", [[pa.record_batch(
        {"k": r // h, "o": r % h, "val": np.cos(r * 0.3)})]])


def _attention_tables(ctx, length=150, d=8):
    for name, phase in (("q", 0.0), ("k", 0.5), ("v", 1.0)):
        r = np.arange(length * d)
        ctx.register_record_batches(name, [[pa.record_batch(
            {"t": r // d, "d": r % d, "x": np.sin(r * 0.7 + phase)})]])


MATMUL = ("SELECT a.s, w.o, SUM(a.val * w.val) AS v FROM a JOIN w ON a.k = w.k "
          "GROUP BY a.s, w.o")

ATTENTION = (
    "WITH scores AS (SELECT q.t AS i, k.t AS j, SUM(q.x * k.x) / 8.0 AS s "
    "FROM q JOIN k ON q.d = k.d WHERE q.t >= k.t GROUP BY q.t, k.t), "
    "weights AS (SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS w FROM scores), "
    "norm AS (SELECT i, SUM(w) AS z FROM weights GROUP BY i) "
    "SELECT weights.i AS t, v.d, SUM(weights.w * v.x) / norm.z AS x "
    "FROM weights JOIN v ON weights.j = v.t JOIN norm ON weights.i = norm.i "
    "GROUP BY weights.i, v.d, norm.z")


def _rows(table: pa.Table):
    """The rows, sorted, as tuples."""
    columns = [table.column(i).to_pylist() for i in range(table.num_columns)]
    return sorted(zip(*columns), key=lambda r: tuple(map(str, r[:-1])))


def _assert_same(want: pa.Table, got: pa.Table):
    want, got = _rows(want), _rows(got)
    assert len(want) == len(got)
    for w, g in zip(want, got):
        assert w[:-1] == g[:-1]
        assert math.isclose(w[-1], g[-1], rel_tol=1e-12, abs_tol=1e-12), (w, g)


def _install(ctx) -> EinFold:
    """Installs einfold in `ctx`'s session, and returns the extension."""
    found = EinFold()
    ctx.with_extensions(found)
    return found


def _pair(load):
    """A session without einfold and one with it, both with `load`'s tables."""
    off, on = SessionContext(), SessionContext()
    found = _install(on)
    load(off)
    load(on)
    return off, on, found


def test_runs_a_matrix_product():
    off, on, found = _pair(_tables)
    _assert_same(off.sql(MATMUL).to_arrow_table(), on.sql(MATMUL).to_arrow_table())
    assert found.plans_with_einfold() >= 1


def test_runs_causal_attention_as_a_pipeline():
    off, on, found = _pair(_attention_tables)
    _assert_same(off.sql(ATTENTION).to_arrow_table(), on.sql(ATTENTION).to_arrow_table())
    assert found.plans_with_pipelines() >= 1


def test_leaves_other_queries_alone():
    off, on, found = _pair(_tables)
    sql = "SELECT a.s, COUNT(*) AS n, AVG(a.val) AS m FROM a GROUP BY a.s"
    want, got = off.sql(sql).to_arrow_table(), on.sql(sql).to_arrow_table()
    assert _rows(want) == _rows(got)
    assert found.plans() >= 1 and found.plans_with_einfold() == 0


def test_shares_the_session_it_was_installed_in():
    ctx = SessionContext()
    found = _install(ctx)
    _tables(ctx)
    ctx.sql(MATMUL).to_arrow_table()
    assert found.plans_with_einfold() >= 1


def test_runs_queries_again_and_tables_replaced():
    off, on, found = _pair(_tables)
    for _ in range(3):
        _assert_same(off.sql(MATMUL).to_arrow_table(), on.sql(MATMUL).to_arrow_table())
    for ctx in (off, on):
        ctx.deregister_table("w")
        r = np.arange(16 * 8)
        ctx.register_record_batches("w", [[pa.record_batch(
            {"k": r // 8, "o": r % 8, "val": -np.cos(r * 0.3)})]])
    _assert_same(off.sql(MATMUL).to_arrow_table(), on.sql(MATMUL).to_arrow_table())


def test_in_xarray_sql():
    xr = pytest.importorskip("xarray")
    xql = pytest.importorskip("xarray_sql")
    rng = np.random.default_rng(0)
    a = xr.Dataset({"val": (("s", "k"), rng.standard_normal((200, 16)))},
                   coords={"s": np.arange(200), "k": np.arange(16)})
    w = xr.Dataset({"val": (("k", "o"), rng.standard_normal((16, 8)))},
                   coords={"k": np.arange(16), "o": np.arange(8)})

    def load(ctx):
        ctx.from_dataset("a", a)
        ctx.from_dataset("w", w)

    off, on = xql.Context(), xql.Context()
    found = _install(on)
    load(off)
    load(on)
    _assert_same(off.sql(MATMUL).to_arrow_table(), on.sql(MATMUL).to_arrow_table())
    assert found.plans_with_einfold() >= 1


def test_gradients_through_ddxdb():
    ddxdb = pytest.importorskip("ddxdb")
    loss = ("WITH c AS (SELECT a.s, w.o, SUM(a.val * w.val) AS z FROM a JOIN w ON a.k = w.k "
            "GROUP BY a.s, w.o), loss AS (SELECT SUM(tanh(z) * tanh(z)) AS l FROM c) ")
    sql = loss + "SELECT s, k, grad(loss, val) AS g FROM a ORDER BY s, k"
    off, on = ddxdb.Context(), ddxdb.Context()
    found = _install(on)
    _tables(off)
    _tables(on)
    _assert_same(off.sql(sql).to_arrow_table(), on.sql(sql).to_arrow_table())
    assert found.plans_with_einfold() >= 1


def _deep(depth):
    """A query whose plan is a chain of `depth` joins."""
    ctes = ["t0 AS (SELECT i, x FROM base)"]
    for n in range(1, depth + 1):
        ctes.append(f"t{n} AS (SELECT a.i, a.x + b.x AS x FROM t{n - 1} a JOIN base b ON a.i = b.i)")
    return "WITH " + ", ".join(ctes) + f" SELECT SUM(x) AS s FROM t{depth}"


@pytest.mark.parametrize("depth", [15, 60])
def test_plans_deep_queries(depth):
    off, on = SessionContext(), SessionContext()
    _install(on)
    for ctx in (off, on):
        ctx.register_record_batches("base", [[pa.record_batch(
            {"i": [0, 1, 2], "x": [1.0, 2.0, 3.0]})]])
    sql = _deep(depth)
    assert on.sql(sql).to_pydict() == off.sql(sql).to_pydict()
