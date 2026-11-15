# SPDX-FileCopyrightText: 2026 OpenAI
#
# SPDX-License-Identifier: Apache-2.0

"""Compact coordinate execution agrees with relational rows across seeds."""

import datafusion
import numpy as np
import pyarrow as pa
import pytest

from einfold.dense_program import DenseProgram, _Axis, _Table, _column


@pytest.mark.parametrize("seed", range(8))
def test_compact_join_and_masked_reduction_preserve_sql_rows(seed):
    rng = np.random.default_rng(seed)
    n, m, k = rng.integers(1, 6, size=3)
    a, b = rng.normal(size=(n, k)), rng.normal(size=(k, m))
    i, r, j = (_Axis(np.arange(s, dtype=np.int64)) for s in (n, k, m))
    tables = {
        "a": _Table((i, r), [(("a", "i"), _column(i.values, (i,), i)),
                              (("a", "r"), _column(r.values, (r,), r)),
                              (("a", "v"), _column(a, (i, r)))]),
        "b": _Table((r, j), [(("b", "r"), _column(r.values, (r,), r)),
                              (("b", "j"), _column(j.values, (j,), j)),
                              (("b", "v"), _column(b, (r, j)))])
    }
    ctx = datafusion.SessionContext()
    for name, table in tables.items():
        schema = pa.schema([(key[1], pa.int64() if key[1] != "v" else pa.float64())
                            for key in table.columns])
        ctx.register_record_batches(name, [table.arrow(schema).to_batches()])
    query = (
        "SELECT a.i, b.j, SUM(CASE WHEN a.v > 0 THEN a.v * b.v ELSE 0.0 END) AS v "
        "FROM a JOIN b ON a.r = b.r GROUP BY a.i, b.j ORDER BY a.i, b.j"
    )
    frame = ctx.sql(query)
    actual = DenseProgram(frame.logical_plan(), frame.schema()).run(tables.__getitem__).arrow(frame.schema())
    expected = frame.to_arrow_table()
    assert actual.column(0).to_pylist() == expected.column(0).to_pylist()
    assert actual.column(1).to_pylist() == expected.column(1).to_pylist()
    np.testing.assert_allclose(actual.column(2).to_numpy(), expected.column(2).to_numpy(), atol=1e-12)
