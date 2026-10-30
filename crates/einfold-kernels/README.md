<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# einfold-kernels

**einfold's array algorithms for tensor computations written in SQL.**

A fold over a join is SQL's spelling of a tensor contraction:

```sql
SELECT a.i, b.k, SUM(a.v * b.v) FROM a JOIN b ON a.j = b.j GROUP BY a.i, b.k
```

This crate computes such folds, and fold pipelines such as attention, from [Apache Arrow](https://arrow.apache.org/) arrays to Arrow arrays, without forming the join's rows. It has dense, positional kernels for small integer keys, sorting-based ones for any other keys, and a fused, tiled kernel for pipelines. They give SQL's answers, including for NULL keys and factors, duplicate rows, NaN and infinities.

Nothing here depends on a query engine. Most users want [`einfold-datafusion`](https://crates.io/crates/einfold-datafusion), which finds folds in Apache DataFusion's plans and runs them with these kernels. See [einfold's README](https://github.com/xqlsystems/einfold#readme).

Apache-2.0.
