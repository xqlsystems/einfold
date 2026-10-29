<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# M1's gate

einfold's milestone M1 (design §16) is gated on ddx's `matmul` gradient program, end to end, at n = 50,000: einfold on must be at least 4× faster than einfold off, on the same DataFusion. ddx is an XQL Systems project for automatic differentiation of SQL queries. This benchmark builds the program as ddx's own benchmark does (`crates/ddx-datafusion/tests/ad_perf.rs`), for `SUM(tanh(a·w)²)` with `a` of `n × 16` and `w` of `16 × 8`. It runs the program with einfold off and on (`einfold_datafusion::install`). It checks that both give the same gradients, within how far two correct evaluations can differ when each adds its sums in its own order, and that einfold's gradients are the same bits on one thread as on every thread.

It is a crate of its own, not a member of einfold's workspace, so that einfold doesn't depend on ddx. It runs on Tokio's multi-threaded runtime, as a host does; ddx's `ad_perf` runs on the single-threaded one, where DataFusion gets one core. JAX on CPU uses every core too: on the same machine, `jax.jit(jax.grad(…))` of this loss takes 1.6 ms.

Run with `cargo run --release [n …]`, with `protoc` on the `PATH` or in `PROTOC`, for ddx's Substrait dependency.

## Result

2026-10-28. Intel Core i7-8700 (6 cores, 12 threads), Rust 1.99, DataFusion 55.2.0, ddx `main` at [921aef1](https://github.com/xqlsystems/ddx/commit/921aef1a34baa3b58ab32a5a68bd663b35060bfd), einfold at the commit that last changed this file, on an otherwise idle machine. Four rounds of einfold off and on, alternating which runs first, each the best of 5 runs; each step and the whole program at its best over the rounds.

| step, n = 50,000 | einfold off, ms | einfold on, ms |
|---|---:|---:|
| ddx's checks (each table's keys identify its rows; the cotangent's shape) | 32.6 | 32.7 |
| `saved_0` (forward z = a·w) | 35.1 | 9.5 |
| `saved_1`, `value`, `cotangent_1` | 2.5 | 2.5 |
| `cotangent_0` | 16.9 | 18.0 |
| `grad_0_a` (Ā) | 261.9 | 15.5 |
| `grad_1_w` (W̄) | 36.5 | 15.3 |
| **whole program** | **377.2** | **94.9** |

**3.98× at n = 50,000, just below M1's gate of 4×.** The rounds, off against on: 404.1 and 96.6 ms, then 381.4 and 94.9, 377.2 and 102.7, 378.0 and 95.4. At n = 10,000 it's 2.5× (90.9 ms against 36.5). The benchmark exits with an error when the speedup at n = 50,000 is below 4×.

An earlier measurement, before the fixes for #60–#62, gave 4.0–4.1× (368.9 ms against 89.8), with each configuration timed in one block rather than alternated. Since those fixes, the joined-back steps compute the regroup's test as a column of the domain (#60), and detection checks that each moved expression can't fail (#61): about 5 ms with einfold on, in `grad_0_a`, `grad_1_w` and `cotangent_0`.

The gradients agree within their derived bounds (`bounds` in `src/main.rs`), and einfold's are the same bits on 1 thread as on 12.

einfold finds each gradient step whole: ddx's transposed contraction, joined back to the table, grouped again by the table's keys, and left-joined from it. It runs the step as the contraction with the table as its domain, checking as it runs that the table's keys are unique. That took Ā from 262 ms to 16.

Where the rest is, with einfold on:

- **ddx's checks, 33 ms,** run before the steps. One groups `a`'s 800,000 rows by their keys to check that they identify its rows. It isn't a fold, and einfold leaves it to DataFusion. It checks the same thing einfold's domain check does, so a host running both does that work twice.
- **`cotangent_0`, 17 ms,** is elementwise (the derivative of `tanh(z)²`), not a fold.
- **The three folds, 40 ms between them.** They are mostly linear passes: reading 800,000 rows' keys and factors, and writing 400,000 or 800,000 rows out.

So more than half of what remains is ddx's own work outside the folds. Even if einfold's three folds were free, the program would be about 7× faster, not more.
