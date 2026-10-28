# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""einfold: array algorithms for tensor computations written in SQL.

A matrix product, a tensor contraction or an attention layer can be written
in SQL as joins and aggregates. A SQL engine then runs it as SQL: it builds
every row of the join, and hashes each into its group. einfold finds these
computations in a query's plan and runs them with array algorithms instead,
giving the same rows, up to the rounding of floating-point sums.

einfold plugs into a SQL engine. Each engine has its own module:

* ``einfold.datafusion``, for Apache DataFusion's Python bindings
  (``pip install "einfold[datafusion]"``).
"""

from einfold._native import DATAFUSION_VERSION

__all__ = ["DATAFUSION_VERSION"]
