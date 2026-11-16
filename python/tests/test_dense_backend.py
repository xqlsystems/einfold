# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""A session's dense backend, the experimental function that computes the
matrix products and attention einfold has checked: with a backend that
computes them as SQL does, or one that declines, each query answers as
DataFusion does without einfold; a backend's errors and malformed results
fail the query; and a backend belongs to its own extension's sessions."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from datafusion import SessionContext

from einfold.datafusion import EinFold


class NumpyBackend:
    """Computes what einfold offers in float64 with NumPy, as SQL does, and
    records each call's kind and shape. With `decline`, it declines."""

    def __init__(self, decline=False):
        self.decline = decline
        self.calls = []

    def __call__(self, kind, arrays, shape, scale, ranges):
        self.calls.append((kind, tuple(shape)))
        if self.decline:
            return None
        arrays = [a.to_numpy() for a in arrays]
        if kind == "matmul":
            w, x, z, y = shape
            a, b = arrays[0].reshape(w, x, z), arrays[1].reshape(w, z, y)
            return pa.array((a @ b).reshape(-1), type=pa.float64())
        assert kind == "attention"
        w, x, e, j, y = shape
        q, kt, v = arrays[0].reshape(w, x, e), arrays[1].reshape(w, e, j), arrays[2].reshape(w, j, y)
        code, c = scale
        s = q @ kt
        s = {0: s, 1: s * c, 2: s / c}[code]
        allowed = np.zeros((x, j), dtype=bool)
        for i, (lo, hi) in enumerate(ranges):
            allowed[i, lo:hi] = True
        s = np.where(allowed, s, -np.inf)
        with np.errstate(invalid="ignore"):
            p = np.exp(s - s.max(axis=2, keepdims=True))
            p /= p.sum(axis=2, keepdims=True)
        return pa.array((p @ v).reshape(-1), type=pa.float64())


def _register(ctxs, name, keys, extents, values):
    cells = np.arange(int(np.prod(extents)))
    columns = {}
    for i, key in enumerate(keys):
        stride = int(np.prod(extents[i + 1:]))
        columns[key] = (cells // stride) % extents[i]
    columns["x"] = values
    batch = pa.record_batch(columns)
    for ctx in ctxs:
        ctx.deregister_table(name)
        ctx.register_record_batches(name, [[batch]])


def _rows(table):
    columns = [table.column(i).to_pylist() for i in range(table.num_columns)]
    return sorted(zip(*columns))


def _assert_same(want, got):
    want, got = _rows(want), _rows(got)
    assert [w[:-1] for w in want] == [g[:-1] for g in got]
    np.testing.assert_allclose([g[-1] for g in got], [w[-1] for w in want],
                               rtol=1e-9, atol=1e-12)


def _sessions(backend, **options):
    plain = SessionContext()
    found = EinFold(dense_backend=backend, dense_min_ops=0, **options)
    return plain, SessionContext().with_extensions(found)


MATMUL = ("SELECT a.i, b.j, SUM(a.x * b.x) AS v FROM a JOIN b ON a.k = b.k "
          "GROUP BY a.i, b.j")

BATCHED_MATMUL = ("SELECT a.b, a.i, w.j, SUM(a.x * w.x) AS v FROM a JOIN w "
                  "ON a.b = w.b AND a.k = w.k GROUP BY a.b, a.i, w.j")


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("decline", [False, True])
def test_products_answer_as_sql_after_inputs_change(seed, decline):
    rng = np.random.default_rng(seed)
    n, k, m, nb = (int(v) for v in rng.integers(2, 9, size=4))
    backend = NumpyBackend(decline)
    plain, ours = _sessions(backend)
    for _ in range(2):
        _register((plain, ours), "a", ("i", "k"), (n, k), rng.normal(size=n * k))
        _register((plain, ours), "b", ("k", "j"), (k, m), rng.normal(size=k * m))
        _assert_same(plain.sql(MATMUL).to_arrow_table(), ours.sql(MATMUL).to_arrow_table())
    # A product per batch: each batch's operands, batch first.
    _register((plain, ours), "a", ("b", "i", "k"), (nb, n, k), rng.normal(size=nb * n * k))
    _register((plain, ours), "w", ("b", "k", "j"), (nb, k, m), rng.normal(size=nb * k * m))
    _assert_same(plain.sql(BATCHED_MATMUL).to_arrow_table(),
                 ours.sql(BATCHED_MATMUL).to_arrow_table())
    assert ("matmul", (1, n, k, m)) in backend.calls
    assert ("matmul", (nb, n, k, m)) in backend.calls


def _attention(scale, mask):
    return (
        f"WITH scores AS (SELECT q.b, q.h, q.t AS i, k.t AS j, SUM(q.x * k.x){scale} AS s "
        f"FROM q JOIN k ON q.b = k.b AND q.h = k.h AND q.d = k.d{mask} "
        "GROUP BY q.b, q.h, q.t, k.t), "
        "weights AS (SELECT b, h, i, j, exp(s - MAX(s) OVER (PARTITION BY b, h, i)) AS w "
        "FROM scores), "
        "norm AS (SELECT b, h, i, SUM(w) AS z FROM weights GROUP BY b, h, i) "
        "SELECT weights.b, weights.h, weights.i, v.c, SUM(weights.w / norm.z * v.x) AS x "
        "FROM weights JOIN v ON weights.j = v.t AND weights.b = v.b AND weights.h = v.h "
        "JOIN norm ON weights.i = norm.i AND weights.b = norm.b AND weights.h = norm.h "
        "GROUP BY weights.b, weights.h, weights.i, v.c")


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("scale", [" / 8.0", " * 0.5", ""])
@pytest.mark.parametrize("mask", [" AND q.t >= k.t", " AND q.t - k.t < 3 AND q.t >= k.t", ""])
def test_attention_per_batch_and_head_answers_as_sql(seed, scale, mask):
    """The backend gets one problem per batch and head, each query's own
    range of keys, and computes the shifted softmax as SQL writes it."""
    rng = np.random.default_rng(seed)
    nb, nh = (int(v) for v in rng.integers(1, 4, size=2))
    length, d, c = (int(v) for v in rng.integers(2, 12, size=3))
    backend = NumpyBackend()
    plain, ours = _sessions(backend)
    for name, last, extent in (("q", "d", d), ("k", "d", d), ("v", "c", c)):
        values = rng.normal(size=nb * nh * length * extent)
        _register((plain, ours), name, ("b", "h", "t", last), (nb, nh, length, extent), values)
    sql = _attention(scale, mask)
    _assert_same(plain.sql(sql).to_arrow_table(), ours.sql(sql).to_arrow_table())
    assert ("attention", (nb * nh, length, d, length, c)) in backend.calls


def test_a_declining_backend_leaves_attention_to_einfold():
    backend = NumpyBackend(decline=True)
    plain, ours = _sessions(backend)
    rng = np.random.default_rng(3)
    for name, last in (("q", "d"), ("k", "d"), ("v", "c")):
        _register((plain, ours), name, ("b", "h", "t", last), (2, 2, 9, 3), rng.normal(size=108))
    sql = _attention(" / 8.0", " AND q.t >= k.t")
    _assert_same(plain.sql(sql).to_arrow_table(), ours.sql(sql).to_arrow_table())
    assert any(kind == "attention" for kind, _ in backend.calls)


def test_attention_that_could_overflow_isnt_offered():
    """Values near the largest double: SQL's `SUM(w * v) / SUM(w)` overflows
    where a softmax taken first doesn't, so einfold keeps the work, and the
    query answers as SQL does."""
    backend = NumpyBackend()
    plain, ours = _sessions(backend)
    for name, last in (("q", "d"), ("k", "d"), ("v", "c")):
        values = np.zeros(36) if name != "v" else np.full(36, 1e308)
        _register((plain, ours), name, ("b", "h", "t", last), (1, 2, 6, 3), values)
    sql = _attention("", "")
    _assert_same(plain.sql(sql).to_arrow_table(), ours.sql(sql).to_arrow_table())
    assert not any(kind == "attention" for kind, _ in backend.calls)


@pytest.mark.parametrize("q, k, scale", [
    (1.0, [-744.4, 0.0, 0.0], ""),
    # Scores of `[0, 745, 745]` from a tiny query and a large scale: its
    # square would round to 0, so einfold scales a norm by its largest
    # element.
    (1e-200, [0.0, 1.0, 1.0], " * 7.45e202"),
])
def test_attention_that_could_underflow_isnt_offered(q, k, scale):
    """Weights of about `[5e-324, 1, 1]` over values `[1e300, 1e-300,
    1e-300]`: SQL's `w / z` rounds the first to 0, where a backend that
    multiplies before it divides keeps it, and answers `5e-324 · 1e300 / 2`,
    far from SQL's `1e-300`. So einfold keeps the work, and the query
    answers as SQL does."""
    backend = NumpyBackend()
    plain, ours = _sessions(backend)
    _register((plain, ours), "q", ("b", "h", "t", "d"), (1, 1, 1, 1), np.array([q]))
    _register((plain, ours), "k", ("b", "h", "t", "d"), (1, 1, 3, 1), np.array(k))
    _register((plain, ours), "v", ("b", "h", "t", "c"), (1, 1, 3, 1),
              np.array([1e300, 1e-300, 1e-300]))
    sql = _attention(scale, "")
    want = plain.sql(sql).to_arrow_table()
    got = ours.sql(sql).to_arrow_table()
    np.testing.assert_allclose(want.column("x").to_pylist(), [1e-300], rtol=1e-9, atol=0)
    np.testing.assert_allclose(got.column("x").to_pylist(), [1e-300], rtol=1e-9, atol=0)
    assert not any(kind == "attention" for kind, _ in backend.calls)


@pytest.mark.parametrize("result, error", [
    (lambda shape: (_ for _ in ()).throw(RuntimeError("the GPU is busy")), "the GPU is busy"),
    (lambda shape: pa.array([1.0, 2.0]), "gave a product 2 values"),
    (lambda shape: pa.array([1, 2], type=pa.int64()), "must return float64"),
    (lambda shape: pa.array([None] * int(np.prod(shape[:2]) * shape[3]), type=pa.float64()),
     "returned NULLs"),
])
def test_a_backends_errors_fail_the_query(result, error):
    def backend(kind, arrays, shape, scale, ranges):
        return result(shape)

    plain, ours = _sessions(backend)
    rng = np.random.default_rng(0)
    _register((ours,), "a", ("i", "k"), (4, 3), rng.normal(size=12))
    _register((ours,), "b", ("k", "j"), (3, 5), rng.normal(size=15))
    with pytest.raises(Exception, match=error):
        ours.sql(MATMUL).to_arrow_table()


def test_a_backend_belongs_to_its_own_sessions():
    """Two extensions, one with a backend: only its session calls it, and
    small work isn't offered."""
    backend = NumpyBackend()
    with_backend = SessionContext().with_extensions(
        EinFold(dense_backend=backend, dense_min_ops=0))
    without = SessionContext().with_extensions(EinFold())
    small = SessionContext().with_extensions(EinFold(dense_backend=backend))
    rng = np.random.default_rng(1)
    _register((with_backend, without, small), "a", ("i", "k"), (6, 4), rng.normal(size=24))
    _register((with_backend, without, small), "b", ("k", "j"), (4, 5), rng.normal(size=20))
    for ctx in (without, small):
        ctx.sql(MATMUL).to_arrow_table()
    assert backend.calls == []
    with_backend.sql(MATMUL).to_arrow_table()
    assert backend.calls == [("matmul", (1, 6, 4, 5))]
