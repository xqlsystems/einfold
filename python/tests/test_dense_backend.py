# SPDX-FileCopyrightText: 2026 OpenAI
#
# SPDX-License-Identifier: Apache-2.0

"""Session callbacks preserve SQL rows and their own session's policy."""

import datafusion
import numpy as np
import pyarrow as pa
import pytest

from einfold.datafusion import EinFold


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("decline", [False, True])
def test_callback_and_decline_equal_sql_after_inputs_change(seed, decline):
    rng = np.random.default_rng(seed)
    n, k, m = (int(x) for x in rng.integers(2, 8, size=3))

    def backend(kind, arrays, shape, scale, ranges):
        assert kind == "matmul"
        if decline:
            return None
        w, x, z, y = shape
        a = arrays[0].to_numpy().reshape(w, x, z)
        b = arrays[1].to_numpy().reshape(w, z, y)
        return pa.array((a @ b).reshape(-1), type=pa.float64())

    plain = datafusion.SessionContext()
    accelerated = datafusion.SessionContext().with_extensions(
        EinFold(dense_backend=backend, dense_min_ops=0)
    )
    query = (
        "SELECT a.i, b.j, SUM(a.v * b.v) AS v FROM a JOIN b ON a.k = b.k "
        "GROUP BY a.i, b.j ORDER BY a.i, b.j"
    )
    for _ in range(2):
        for name, rows, cols, keys in [("a", n, k, ("i", "k")), ("b", k, m, ("k", "j"))]:
            values = rng.normal(size=rows * cols)
            cells = np.arange(rows * cols)
            batch = pa.record_batch({keys[0]: cells // cols, keys[1]: cells % cols, "v": values})
            for ctx in (plain, accelerated):
                ctx.deregister_table(name)
                ctx.register_record_batches(name, [[batch]])
        want = plain.sql(query).to_arrow_table()
        got = accelerated.sql(query).to_arrow_table()
        assert got.column(0).to_pylist() == want.column(0).to_pylist()
        assert got.column(1).to_pylist() == want.column(1).to_pylist()
        np.testing.assert_allclose(got.column(2).to_numpy(), want.column(2).to_numpy(), atol=1e-12)
