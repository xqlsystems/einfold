<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# M1's gate

einfold's milestone M1 (design §16) is gated on ddx's `matmul` gradient program, end to end, at n = 50,000: einfold on must be at least 4× faster than einfold off, on the same DataFusion. ddx is an XQL Systems project for automatic differentiation of SQL queries. This benchmark builds the program as ddx's own benchmark does (`crates/ddx-datafusion/tests/ad_perf.rs`), for `SUM(tanh(a·w)²)` with `a` of `n × 16` and `w` of `16 × 8`. It runs the program with einfold off and on (`einfold_datafusion::install`), and checks that both give the same gradients.

It is a crate of its own, not a member of einfold's workspace, so that einfold doesn't depend on ddx. It runs on Tokio's multi-threaded runtime, as a host does; ddx's `ad_perf` runs on the single-threaded one, where DataFusion gets one core. JAX on CPU uses every core too: on the same machine, `jax.jit(jax.grad(…))` of this loss takes 1.6 ms.

Run with `cargo run --release [n …]`, with `protoc` on the `PATH` or in `PROTOC`, for ddx's Substrait dependency.

## Result

2026-10-26. Intel Core i7-8700 (6 cores, 12 threads), Rust 1.99, DataFusion 55.2.0, ddx `main` at [921aef1](https://github.com/xqlsystems/ddx/commit/921aef1a34baa3b58ab32a5a68bd663b35060bfd), einfold at the commit that last changed this file. Best of 5.

| step, n = 50,000 | einfold off, ms | einfold on, ms |
|---|---:|---:|
| ddx's checks (each table's keys identify its rows; the cotangent's shape) | 32.0 | 32.2 |
| `saved_0` (forward z = a·w) | 31.5 | 19.9 |
| `saved_1`, `value`, `cotangent_1` | 2.3 | 2.2 |
| `cotangent_0` | 17.4 | 17.8 |
| `grad_0_a` (Ā) | 246.7 | 21.5 |
| `grad_1_w` (W̄) | 33.9 | 20.2 |
| **whole program** | **364.6** | **117.2** |

**3.1–3.2× at n = 50,000 over three runs, and 2.3–2.4× at n = 10,000. The gate, 4×, is not yet passed.** The gradients agree to a relative 1e-12.

Each gradient step is ddx's transposed contraction, joined back to the table and grouped again by its keys, then left-joined from the table. einfold finds the whole step, and runs it as the contraction with the table as its domain, checking as it runs that the table's keys are unique. That took Ā from 247 ms to 22.

Where the rest is, with einfold on:

- **ddx's checks, 32 ms,** run before the steps. One groups `a`'s 800,000 rows by their keys to check that they identify its rows. It isn't a fold, and einfold leaves it to DataFusion.
- **`cotangent_0`, 18 ms,** is elementwise (the derivative of `tanh(z)²`), not a fold.
- **The three folds, 20–22 ms each,** are mostly einfold's own overhead around the arithmetic: reading the operands' keys, placing each row, and writing out the result's 400,000 or 800,000 rows, each in a single-threaded pass. The next pull request takes these on.
