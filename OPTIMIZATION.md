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

## Compact logical-plan array programs

The optional `einfold.dense_program.DenseProgram` lowers supported
DataFusion logical operators to compact arrays. The host supplies current
input tables on each invocation and chooses numerical execution. Axis
domains are represented once; values retain their factorization instead
of repeatedly expanding coordinate columns for intermediate Arrow tables.

Joins become gathers or contractions, grouped/window reductions retain
their coordinate domains, and UNION branches preserve bag semantics for
gradient accumulation. SQL row presence is distinct from column NULL
validity. Exact compact-axis ordering can precede Arrow expansion. A
bounded computed-value cache evicts intermediates for recomputation, while
the configured temporary-array bound declines oversized operations.

This is an explicitly invoked prototype, not an unconditional optimizer
rewrite. Unsupported operators or row semantics raise `_Decline`; the host
can execute original SQL before publishing external updates. The ordinary
query host imports changed inputs afresh and returns Arrow snapshots.
Dense program internals are private; this is not yet a stable public API.
The CPU module requires NumPy and PyArrow when imported, without adding
mandatory dependencies to the base extension package.

Device leaves, bounded integer-domain metadata and known-count bounds let
downstream hosts compile GPU gathers/scatter reductions without pulling
numerical arrays to CPU. Actual input domains and structural SQL checks
must be validated or proven, not skipped. Unsupported dynamic checks and
integer arithmetic decline. Compilation is owned by the downstream host,
which passes numerical state explicitly and enforces atomic publication.

## End-to-end measurements and memory

Machine: Apple M5 Ultra, 30 CPU cores, 64 GPU cores, 256 GiB, macOS 27.0.1.
Final runs were serial with game contention off and `MLX_ENABLE_TF32=0`.
The following five-step SQL training runs use SGD, tanh GELU and batch 1
for the larger models. The GPT-2-size vocabulary has 50,257 classes but
inputs are Shakespeare character IDs. Every loss and parameter update was
checked against a float64 JAX reference outside the timer. Warm means
exclude step 1; these are not convergence tests or confidence intervals.

| Parameters / context | Shared Arrow CPU → compact CPU | Shared Arrow GPU → compact GPU |
|---|---:|---:|
| 213,504 / 64 | 431 → 61 ms, 7.1× | 482 → 83 ms, 5.8× |
| 10,745,088 / 256 | 29.257 → 1.406 s, 20.8× | 29.937 → 0.589 s, 50.8× |
| 123,563,520 / 16 | 164.034 → 4.037 s, 40.6× | 163.646 → 2.434 s, 67.2× |

For the 123.6M/context-16 GPU model, first/warm loss-plus-update pairs
were 3.750/2.434 s and sampled RSS 17.91 GiB, versus 40.0 GiB for shared
Arrow execution. CPU RSS was 22.90 versus 38.7 GiB. The
124,337,664-parameter/context-1,024 GPU model completed five updates:
first pair 7.496 s, warm 5.575 s, sampled RSS 28.72 GiB at a 32 GiB cache
budget. CPU full-context trials were incomplete and establish no speedup.
An 8→32 GiB computed-cache trial improved warm GPU time 14.242→5.835 s;
the later final run gave 5.575 s. Cache budgets and RSS are different measures.

These end-to-end gains include xarray-sql's shared execution, plan reuse,
registration and compact host integration. They are not isolated gains
from the kernel callback or this module alone.

The subsequent real Shakespeare AdamW/dropout recipe (10.7M parameters,
batch 12, context 256) improved 2.024 s Arrow GPU→101.3 ms compiled
resident GPU per warm step, **19.98×**, over a 100-update run. That further
gain combines compact programs with downstream state ownership,
`ddxdb.Prepared.values` primal reuse, whole-step compilation and GPU dropout.
It remains 3.76× slower than native PyTorch MPS (27.0 ms). All 100 timed
steps exported zero numerical arrays; loss and validation scalars still
synchronize. First-step and driver-cache behavior were not isolated.

## Correctness and review boundaries

The randomized Python test compares compact join/conditional reduction
results and ordered coordinate rows with DataFusion over eight seeded
shapes/inputs. Existing kernel metamorphic tests remain applicable to the
default executor. xarray-sql's downstream suite covers live inputs,
NULLs/duplicates, masks, fallback, schema invalidation, compiled changing
inputs, snapshots and training state ownership; its final non-integration
run passed 714 tests. Five-step real-recipe checks compare every raw
gradient, loss, weight and optimizer moment with original nanoGPT/PyTorch
at float32 tolerance 3e-5.

The tested base is `39b25210d3b39027ef91584383fb114dfa8a4a48` (Python
0.2.0). Local benchmark wheels were labeled `0.2.0+xqlmlx.6`; upstream
versions are not bumped. The work is split into the Rust/Python kernel
bridge and the optional compact array prototype. xarray-sql's
`NANOGPT_PERFORMANCE.md` and `benchmarks/apple` contain the timing commands,
raw evidence and limitations. No default optimizer or precision policy
is silently changed by importing the extension.

The persistent checkout passed workspace and Python-extension Clippy with
warnings denied, Rust formatting, warning-free Rust docs and REUSE license
checks. The rebuilt wheel passed the combined ddx/EinFold Python suite
(126 passed, 6 optional skips), then xarray-sql passed all 714
non-integration tests against the rebuilt dependency wheels.

The full Rust workspace run had 72 passes, 4 ignored tests and one failure:
`nan_compares_by_its_sign`, a DataFusion floating-point behavior assertion.
Running that exact test on the unmodified pinned base reproduced the same
`[true, true]` versus `[true, false]` failure on this macOS host. It is a
baseline limitation, not a passing check; it is disclosed in the PRs.

The publication stack incorporates upstream main through `a60c86f`,
resolving its session-runtime changes while retaining ordered providers and
backend callbacks. Benchmark records identify their earlier pinned base;
upstream changes are not attributed to these optimization measurements.
