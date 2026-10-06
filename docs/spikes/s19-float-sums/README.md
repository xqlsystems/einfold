<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S19: are floating-point sums repeatable in SQL engines and JAX?

**Question.** einfold's design originally made bit-for-bit determinism the default everywhere. Is that what SQL engines and machine-learning systems do? If not, what do they do instead?

**Answer.** No mainstream SQL engine guarantees repeatable floating-point sums when it runs in parallel. JAX makes speed the default and offers determinism and precision as explicit settings. einfold's policy (design doc, section 8.6) follows that split.

Date: 2026-10-04, rerun 2026-10-06. DuckDB 1.5.6 and DataFusion 54.0.0, on a 12-core Linux machine.

## Why sums vary

Floating-point addition is not associative: `(a + b) + c` can differ from `a + (b + c)` in the last bits. A parallel sum adds partial results in whatever order threads finish, so the same query on the same data can return slightly different values on each run.

## What SQL engines and JAX do

- **DuckDB.** Its maintainers call varying `sum(double)` results expected behavior, not a bug. They suggest `fsum` (Kahan summation, which tracks rounding error as it adds), `threads=1`, or casting to `DECIMAL` (DuckDB discussion #12693). A separate DuckDB issue (#26143) reports that `fsum` combines partial sums incorrectly in some cases.
- **PostgreSQL.** Serial plans give stable float sums, and parallel plans do not. This was reported on the PostgreSQL mailing list in 2017 and treated as inherent to floating point.
- **BigQuery and Snowflake.** Their documentation says `SUM` over floating-point values can differ between runs, and recommends fixed-point types where precision matters. (This comes from search summaries of their documentation; we did not read the pages themselves.)
- **DataFusion.** No documented guarantee. The measurement below shows variation.
- **JAX** treats three concerns separately:
  - *Randomness* is deterministic by design. Random numbers come from explicit keys passed through pure functions.
  - *Order of operations* is fast by default and deterministic by opt-in. On GPUs, reductions use atomic operations, and the compiler may choose different kernels between compilations, so results can vary between runs. Users opt in with process-wide XLA flags (`--xla_gpu_exclude_nondeterministic_ops`, formerly `--xla_gpu_deterministic_ops`, plus `--xla_gpu_autotune_level=0`). XLA's documentation warns of substantial throughput loss.
  - *Precision* is fast by default, with explicit, scoped controls. Arrays default to 32-bit floats. A float32 matrix multiply at the default precision runs in bfloat16 on TPUs and in TF32 on A100 and H100 GPUs. Users raise precision per operation (a `precision=` argument) or for a block of code (the `jax.default_matmul_precision` context manager).

The SQL engines accumulate `DOUBLE` sums in 64 bits, and offer `DECIMAL` for exact results.

## Measurement

[`float_sums.py`](float_sums.py) sums 4 million `DOUBLE` values spanning 16 orders of magnitude, with mixed signs, and runs each configuration 20 times.

```sh
uv run --with duckdb --with datafusion --with numpy --with pyarrow python float_sums.py
```

| Engine and setting | Distinct results in 20 runs | Relative error vs exact sum (first run) |
|---|---|---|
| DuckDB `sum`, 12 threads | 19 | 1.3e-14 |
| DuckDB `fsum` (Kahan), 12 threads | 4 | 1.8e-16 |
| DuckDB `sum`, 1 thread | 1 | 6.9e-14 |
| DuckDB `fsum`, 1 thread | 1 | 0 |
| DataFusion `sum`, 1 input partition, `target_partitions = 1` | 1 | 2.7e-15 |
| DataFusion `sum`, 1 input partition, `target_partitions = 16` | 4 | 2.3e-15 |
| DataFusion `sum`, 16 input partitions, `target_partitions = 1` | 4 | 1.1e-15 |
| DataFusion `sum`, 16 input partitions, `target_partitions = 16` | 5 | 1.2e-15 |

The counts of distinct results, and the errors of the parallel runs, change from one execution of the script to the next, since that variation is what the spike measures. An earlier execution gave 19 and 5 distinct results for DuckDB's parallel `sum` and `fsum`, and 4–6 for DataFusion's parallel configurations.

## Findings

1. **Repeatable float sums need serial execution in both engines.** Parallel runs varied in DuckDB and in DataFusion.
2. **Compensated summation, such as Kahan's, is not a determinism fix.** DuckDB's maintainers suggested `fsum` as a deterministic alternative, but in parallel it still returned 4 or 5 different results in 20 runs. It shrinks the effect of addition order but does not remove it. Only accumulators whose result is truly independent of order (binned or exact accumulators) are deterministic in parallel.
3. **Determinism and accuracy are different properties.** The single-threaded sums are repeatable but the least accurate (6.9e-14 for DuckDB), because adding strictly left to right accumulates rounding error. The varying parallel sums are more accurate. A fixed order buys repeatability, not accuracy.

## Limitations

- One data distribution and one machine.
- CPU only. GPU engines were not measured.
- A float `SUM` alone, not sums inside joins or rewritten plans.

## References

- DuckDB discussion #12693, "sum of double not deterministic": https://github.com/duckdb/duckdb/discussions/12693
- DuckDB issue #26143, on how `fsum` combines partial sums: https://github.com/duckdb/duckdb/issues/26143
- PostgreSQL mailing list, "Non-deterministic behavior with floating point in parallel mode" (2017): https://www.postgresql.org/message-id/CAFRJ5K0%2BZZaUz0-ihX-aCj1h42H%3Ds-CLWO%2B2Fb6nHCvXx19Diw%40mail.gmail.com
- XLA GPU determinism: https://openxla.org/xla/determinism
- JAX discussion #10674, on GPU determinism: https://github.com/jax-ml/jax/discussions/10674
- JAX default matmul precision: https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html, and JAX issue #10413: https://github.com/jax-ml/jax/issues/10413
