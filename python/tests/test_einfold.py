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
import subprocess
import sys
import textwrap

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


def _same_value(w, g) -> bool:
    """Whether two values agree: both NULL, both NaN, or close."""
    if w is None or g is None:
        return w is None and g is None
    if math.isnan(w) or math.isnan(g):
        return math.isnan(w) and math.isnan(g)
    return math.isclose(w, g, rel_tol=1e-12, abs_tol=1e-12)


def _assert_same(want: pa.Table, got: pa.Table):
    want, got = _rows(want), _rows(got)
    assert len(want) == len(got)
    for w, g in zip(want, got):
        assert w[:-1] == g[:-1]
        assert _same_value(w[-1], g[-1]), (w, g)


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


def test_gives_sqls_nulls_and_nans():
    """A matrix product over a NaN and a NULL factor gives SQL's NaNs, and
    skips the NULL as `SUM` does."""
    def tables(ctx):
        _tables(ctx)
        ctx.deregister_table("w")
        r = np.arange(16 * 8)
        val = pa.array(np.cos(r * 0.3)).to_pylist()
        val[3], val[10] = float("nan"), None
        ctx.register_record_batches("w", [[pa.record_batch(
            {"k": r // 8, "o": r % 8, "val": pa.array(val, pa.float64())})]])

    off, on, _ = _pair(tables)
    want, got = off.sql(MATMUL).to_arrow_table(), on.sql(MATMUL).to_arrow_table()
    assert any(math.isnan(v) for v in want.column("v").to_pylist())
    _assert_same(want, got)


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
        # Nothing a query encoded outlives it: a plan held after it ran would
        # keep what its operators hold, such as its hash joins' tables.
        assert EinFold.tables_pending() == 0
        assert EinFold.plans_pending() == 0
    for ctx in (off, on):
        ctx.deregister_table("w")
        r = np.arange(16 * 8)
        ctx.register_record_batches("w", [[pa.record_batch(
            {"k": r // 8, "o": r % 8, "val": -np.cos(r * 0.3)})]])
    _assert_same(off.sql(MATMUL).to_arrow_table(), on.sql(MATMUL).to_arrow_table())


def test_in_xarray_sql():
    """einfold in xarray-sql, an XQL Systems project for querying xarray
    datasets with SQL, whose context is a DataFusion session."""
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
    """Gradients from ddxdb, the Python package of ddx, an XQL Systems
    project for automatic differentiation of SQL queries."""
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


def _in_a_process(depth, thread_stack=None):
    """Plans and runs `_deep(depth)` in a process of its own, einfold off and
    then on, from the main thread or from a thread with a stack of
    `thread_stack` bytes, and returns the process, so that a crash fails the
    test rather than the test run. It prints whether the two answers agree."""
    script = textwrap.dedent(f"""
        import sys
        import threading
        import pyarrow as pa
        from datafusion import SessionContext
        from einfold.datafusion import EinFold

        def run():
            answers = []
            for on in (False, True):
                ctx = SessionContext()
                if on:
                    ctx.with_extensions(EinFold())
                ctx.register_record_batches("base", [[pa.record_batch(
                    {{"i": [0, 1, 2], "x": [1.0, 2.0, 3.0]}})]])
                answers.append(ctx.sql({_deep(depth)!r}).to_pydict())
            print(answers[0] == answers[1])

        # A crash writes no core file, which would take most of the test's
        # time.
        if sys.platform == "linux":
            import resource
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

        if {thread_stack!r} is None:
            run()
        else:
            threading.stack_size({thread_stack!r})
            thread = threading.Thread(target=run)
            thread.start()
            thread.join()
        """)
    return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)


@pytest.mark.skipif(sys.platform != "linux", reason="einfold deepens the stack on Linux only")
def test_plans_deeper_than_the_default_stack():
    """DataFusion's FFI decodes each plan, on the thread that asks for it,
    with about 8 KB of stack per node. A chain of 350 joins needs more than
    Linux's default 8 MB stack, which DataFusion alone plans: einfold makes
    room on the main thread's stack."""
    done = _in_a_process(350)
    assert done.returncode == 0 and done.stdout.strip() == "True", done.stderr[-2000:]


@pytest.mark.skipif(sys.platform != "linux", reason="Linux's default thread stack")
def test_decoding_a_deep_plan_overflows_another_threads_stack():
    """Pins a limit that isn't einfold's: datafusion-proto decodes a plan by
    recursing without growing its stack, unlike DataFusion's other walks of
    a plan, so a plan of 350 joins asked for on a thread with an 8 MB stack
    overflows it, and the process dies. Once the decoding grows its stack,
    this fails, and einfold's main-thread workaround can go."""
    done = _in_a_process(350, thread_stack=8 << 20)
    assert done.returncode < 0, (done.returncode, done.stdout)


ATTENTION_INSIDE = (
    "WITH scores AS (SELECT q.t AS i, k.t AS j, SUM(q.x * k.x) / 8.0 AS s "
    "FROM q JOIN k ON q.d = k.d WHERE q.t >= k.t GROUP BY q.t, k.t), "
    "weights AS (SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS w FROM scores), "
    "norm AS (SELECT i, SUM(w) AS z FROM weights GROUP BY i) "
    "SELECT weights.i AS t, v.d, SUM(weights.w / norm.z * v.x) AS x "
    "FROM weights JOIN v ON weights.j = v.t JOIN norm ON weights.i = norm.i "
    "GROUP BY weights.i, v.d")

MASKED_SCORES = (
    "SELECT q.t AS i, k.t AS j, SUM(q.x * k.x) / 8.0 AS s "
    "FROM q JOIN k ON q.d = k.d WHERE q.t >= k.t GROUP BY q.t, k.t")


def test_runs_attention_normalized_inside_the_sum():
    off, on, found = _pair(_attention_tables)
    want, got = off.sql(ATTENTION_INSIDE).to_arrow_table(), on.sql(ATTENTION_INSIDE).to_arrow_table()
    _assert_same(want, got)
    assert found.plans_with_pipelines() >= 1


def test_runs_a_masked_fold():
    off, on, found = _pair(_attention_tables)
    _assert_same(off.sql(MASKED_SCORES).to_arrow_table(), on.sql(MASKED_SCORES).to_arrow_table())
    assert found.plans_with_einfold() >= 1
