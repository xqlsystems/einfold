# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""A Python dense backend on every way datafusion-python runs a plan: each
answers as DataFusion does without einfold and none deadlocks, whether the
backend is called from a query's own thread, from other threads at once, or
runs a query itself; a sliced result is read from its offset; and a result
that isn't an Arrow array fails the query clearly."""

from __future__ import annotations

import threading

import numpy as np
import pyarrow as pa
import pytest
from datafusion import SessionContext

from einfold.datafusion import EinFold

SQL = "SELECT a.i, b.j, SUM(a.x * b.x) AS v FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j"

# Long enough for any of these queries; a deadlocked one never finishes.
TIMEOUT = 60


def _product(arrays, shape):
    w, x, z, y = shape
    a = arrays[0].to_numpy().reshape(w, x, z)
    b = arrays[1].to_numpy().reshape(w, z, y)
    return (a @ b).reshape(-1)


class Backend:
    """Computes products with NumPy, returning `wrap` of the result, and
    records the thread of each call. With `inside`, it first runs
    `inside()`, as a backend that queries another session would."""

    def __init__(self, wrap=lambda out: pa.array(out, type=pa.float64()), inside=None):
        self.wrap, self.inside = wrap, inside
        self.threads = []

    def __call__(self, kind, arrays, shape, scale, ranges):
        if kind != "matmul":
            return None
        self.threads.append(threading.current_thread().name)
        if self.inside is not None:
            self.inside()
        return self.wrap(_product(arrays, shape))


def _sessions(backend):
    plain = SessionContext()
    ours = SessionContext().with_extensions(EinFold(dense_backend=backend, dense_min_ops=0))
    rng = np.random.default_rng(0)
    for name, keys, extents in (("a", ("i", "k"), (7, 5)), ("b", ("k", "j"), (5, 6))):
        cells = np.arange(int(np.prod(extents)))
        columns = {keys[0]: cells // extents[1], keys[1]: cells % extents[1]}
        columns["x"] = rng.normal(size=cells.size)
        for ctx in (plain, ours):
            ctx.register_record_batches(name, [[pa.record_batch(columns)]])
    return plain, ours


def _rows(result):
    if not isinstance(result, pa.Table):
        result = pa.Table.from_batches(result)
    return sorted(zip(*[c.to_pylist() for c in result.columns]))


def _assert_same(want, got):
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert g[:2] == w[:2]
        assert abs(g[2] - w[2]) <= 1e-9 * (1 + abs(w[2]))


def _in_thread(run):
    """`run()` on another thread, failing if it doesn't finish."""
    out = {}

    def target():
        try:
            out["value"] = run()
        except BaseException as e:  # noqa: BLE001 - re-raised below
            out["error"] = e

    t = threading.Thread(target=target)
    t.start()
    t.join(TIMEOUT)
    assert not t.is_alive(), "deadlocked"
    if "error" in out:
        raise out["error"]
    return out["value"]


RUNS = {
    "collect": lambda df: df.collect(),
    "to_arrow_table": lambda df: df.to_arrow_table(),
    "arrow_c_stream": lambda df: pa.RecordBatchReader.from_stream(df).read_all(),
    "pa_table": lambda df: pa.table(df),
    "execute_stream": lambda df: [b.to_pyarrow() for b in df.execute_stream()],
}


@pytest.mark.parametrize("run", RUNS)
def test_each_way_to_run_a_plan_calls_the_backend_and_answers_as_sql(run):
    backend = Backend()
    plain, ours = _sessions(backend)
    want = _rows(plain.sql(SQL).to_arrow_table())
    got = _rows(_in_thread(lambda: RUNS[run](ours.sql(SQL))))
    assert backend.threads, "not offered"
    _assert_same(want, got)


def test_show_calls_the_backend():
    backend = Backend()
    _, ours = _sessions(backend)
    _in_thread(lambda: ours.sql(SQL).show())
    assert backend.threads, "not offered"


def test_queries_on_several_threads_at_once_answer_as_sql():
    backend = Backend()
    plain, ours = _sessions(backend)
    want = _rows(plain.sql(SQL).to_arrow_table())
    results = {}
    threads = [
        threading.Thread(target=lambda i=i: results.update({i: _rows(ours.sql(SQL).collect())}))
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(TIMEOUT)
    assert not any(t.is_alive() for t in threads), "deadlocked"
    assert sorted(results) == list(range(4))
    for got in results.values():
        _assert_same(want, got)


def test_a_backend_that_queries_another_session_answers_as_sql():
    other = SessionContext()
    backend = Backend(inside=lambda: other.sql("SELECT 1").collect())
    plain, ours = _sessions(backend)
    want = _rows(plain.sql(SQL).to_arrow_table())
    _assert_same(want, _rows(_in_thread(lambda: ours.sql(SQL).collect())))
    assert backend.threads, "not offered"


def test_a_backend_that_queries_its_own_session_answers_as_sql():
    holder = {}

    def inside():
        # Once: the inner query's product is offered to this backend too.
        if not holder.get("inside"):
            holder["inside"] = True
            holder["inner"] = _rows(holder["ctx"].sql(SQL).collect())

    backend = Backend(inside=inside)
    plain, ours = _sessions(backend)
    holder["ctx"] = ours
    want = _rows(plain.sql(SQL).to_arrow_table())
    _assert_same(want, _rows(_in_thread(lambda: ours.sql(SQL).collect())))
    _assert_same(want, holder["inner"])
    assert len(backend.threads) >= 2


def test_a_sliced_result_is_read_from_its_offset():
    def sliced(out):
        padded = pa.array(np.concatenate([[123.0], out]), type=pa.float64())
        return padded.slice(1)

    backend = Backend(wrap=sliced)
    plain, ours = _sessions(backend)
    want = _rows(plain.sql(SQL).to_arrow_table())
    _assert_same(want, _rows(ours.sql(SQL).collect()))
    assert backend.threads, "not offered"


@pytest.mark.parametrize("wrap", [
    lambda out: pa.chunked_array([pa.array(out, type=pa.float64())]),
    lambda out: out,
], ids=["chunked_array", "numpy"])
def test_a_result_that_isnt_an_arrow_array_fails_the_query(wrap):
    _, ours = _sessions(Backend(wrap=wrap))
    with pytest.raises(Exception, match="(?i)array"):
        ours.sql(SQL).collect()
