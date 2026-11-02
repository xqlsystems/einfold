<!--
SPDX-FileCopyrightText: 2026 OpenAI
SPDX-License-Identifier: Apache-2.0
-->

# Optional numerical backends and compact SQL array execution

These extensions support [xarray-sql](https://github.com/xqlsystems/xarray-sql),
a SQL interface to labeled arrays, while retaining ordinary
[DataFusion](https://datafusion.apache.org/) SQL planning and
[ddx](https://github.com/xqlsystems/ddx) automatic differentiation of SQL.
[MLX](https://github.com/ml-explore/mlx), Apple's array framework, executes
GPU arithmetic downstream. EinFold does not substitute a handwritten model
or use MLX automatic differentiation.

## Session-local dense kernel callback

The Rust `DenseBackend` trait offers batched matmul and fused attention
after EinFold's SQL validation and packing. A backend can decline a call
and retain the original CPU algorithm. The Python bridge passes numerical
arrays, shapes and attention key intervals to an optional executor scoped
to the extension instance. The executor is never process-global.

The bridge checks output shape and errors rather than accepting arbitrary
callback results as valid rows. Key, duplicate, NULL and domain validation
still governs dispatch. Attention tiling and normalized ranges select
causal or mask-free fast kernels only when equivalent to the SQL ranges.
No handwritten unsafe code is introduced. Original execution remains the
default when no backend is supplied or eligibility thresholds decline it.

Downstream policies explicitly choose NumPy float64 on CPU or MLX float32
on GPU. Small kernels stay on CPU because packing, dispatch and readback
can exceed GPU savings. Matmul results are converted back to normal
EinFold rows at the existing public boundary.

Apple M5 Ultra measurements, November 1, 2026:

| Kernel | Original | Optional executor | Observed improvement |
|---|---:|---:|---:|
| 256×512×256 matrix product | 8.698 ms | NumPy float64 4.422 ms | 1.97× |
| 1024³ matrix product | CPU 64.131 ms | MLX float32 35.174 ms | 1.82× |
| 16,384-token, context-64 causal attention | 95.044 ms | MLX float32 71.292 ms | 1.33× |

NumPy float64 took 33.745 ms for the 1024³ case, slightly faster than GPU.
A smaller attention case regressed about 6% with a lower GPU threshold;
the automatic threshold keeps that work on CPU. These are kernel timings,
not end-to-end model speedups. Repeated ddx gradient examples improved
14.479→9.424 ms (1.54×) for 256 rows and 19.543→14.049 ms (1.39×) for
2,048 rows. Some small prepared DataFusion examples remained faster.

