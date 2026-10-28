# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""einfold for Python: fast folds over joins in datafusion-python sessions.

einfold finds tensor contractions written as SQL (a ``SUM`` of products over
a join, grouped by the output's coordinates), and softmax-weighted averages
of them, such as attention, in a session's query plans, and runs them with
its own algorithms rather than the engine's hash joins and aggregates. It
changes no result beyond float summation's rounding.

It installs in any session of datafusion-python, Apache DataFusion's Python
bindings, through ``SessionContext.with_extensions``, and so in sessions
that build on one, such as xarray-sql's ``Context`` and ddxdb's::

    import einfold
    from xarray_sql import Context

    ctx = Context()
    found = einfold.install(ctx)
    ctx.sql("SELECT a.s, w.o, SUM(a.v * w.v) FROM a JOIN w ON a.k = w.k GROUP BY a.s, w.o")
    found.plans_with_einfold()  # 1

It needs datafusion-python 55, whose DataFusion FFI version it is built
against.
"""

from __future__ import annotations

from einfold._native import DATAFUSION_VERSION, EinFold

__all__ = ["DATAFUSION_VERSION", "EinFold", "install"]


def install(ctx, extension: EinFold | None = None) -> EinFold:
    """Installs einfold in ``ctx``'s session, and returns the extension.

    The session's planner is replaced by einfold's for every query after
    this, through ``ctx`` and every context that shares its session. The
    extension counts what it planned: ``plans()``, and ``plans_with_einfold()``
    for those in which einfold ran a fold or a fold pipeline.

    Install it once per session: a second install would put einfold's
    planner in front of itself.
    """
    extension = EinFold() if extension is None else extension
    ctx.with_extensions(extension)
    return extension
