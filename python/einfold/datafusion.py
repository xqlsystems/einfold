# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""einfold for Apache DataFusion's Python bindings (``datafusion``).

Install einfold in a session with DataFusion's own extension mechanism::

    from datafusion import SessionContext
    from einfold.datafusion import EinFold

    einfold = EinFold()
    ctx = SessionContext().with_extensions(einfold)

Every query planned in that session afterwards, through ``ctx`` or any
context that shares its session, is planned by einfold. Where it finds a
computation it can run, it runs it; every other part of a query runs as
DataFusion would run it. The extension counts what it planned:
``plans()``, ``plans_with_einfold()`` (the plans in which einfold ran
something) and ``plans_with_pipelines()`` (those in which it ran a fused
pipeline, such as attention).

It needs ``datafusion`` 55, whose FFI it is built against.
"""

from einfold._native import EinFold

__all__ = ["EinFold"]
