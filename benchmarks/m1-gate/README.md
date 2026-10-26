<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# M1's gate

einfold's milestone M1 (design §16) is gated on ddx's `matmul` gradient program, end to end, at n = 50,000: einfold on must be at least 4× faster than einfold off, on the same DataFusion. ddx is an XQL Systems project for automatic differentiation of SQL queries. This benchmark builds the program as ddx's own benchmark does (`crates/ddx-datafusion/tests/ad_perf.rs`), for `SUM(tanh(a·w)²)` with `a` of `n × 16` and `w` of `16 × 8`. It runs the program with einfold off and on (`einfold_datafusion::install`), and checks that both give the same gradients.

It is a crate of its own, not a member of einfold's workspace, so that einfold doesn't depend on ddx. It runs on Tokio's multi-threaded runtime, as a host does; ddx's `ad_perf` runs on the single-threaded one, where DataFusion gets one core. JAX on CPU uses every core too: on the same machine, `jax.jit(jax.grad(…))` of this loss takes 1.6 ms.

Run with `cargo run --release [n …]`, with `protoc` on the `PATH` or in `PROTOC`, for ddx's Substrait dependency.

## Result

2026-10-26. Intel Core i7-8700 (6 cores, 12 threads), Rust 1.99, DataFusion 55.2.0, ddx `main` at [921aef1](https://github.com/xqlsystems/ddx/commit/921aef1a34baa3b58ab32a5a68bd663b35060bfd), einfold at the commit that adds this file. Best of 5.

| step, n = 50,000 | einfold off, ms | einfold on, ms |
|---|---:|---:|
| `saved_0` (forward z = a·w) | 33.4 | 18.9 |
| `saved_1`, `value`, `cotangent_1` | 2.5 | 2.2 |
| `cotangent_0` | 17.4 | 17.9 |
| `grad_0_a` (Ā) | 251.4 | 130.2 |
| `grad_1_w` (W̄) | 33.7 | 20.9 |
| **whole program** | **367.9** | **223.5** |

**1.6× at n = 50,000, and 1.5× at n = 10,000 (75 ms against 49). The gate is not yet passed.** The gradients agree to a relative 1e-12.

Where the rest is: einfold finds every contraction in the program, but each gradient step is ddx's transposed contraction, joined back to the table and grouped again by its keys, then left-joined from the table. Detection replaces the contraction, and the join back and second `GROUP BY` (about 120 ms of Ā's 130) stay on DataFusion. They are a no-op when the table's keys are unique, which ddx knows and einfold doesn't. Writing the step as a plain left join from the table (xqlsystems/ddx#208) would let einfold take the table as the fold's domain too. Spike S28 estimated about 4.8× then.
