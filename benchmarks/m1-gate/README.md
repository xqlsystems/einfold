<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# M1's gate

einfold's milestone M1 (design §16) is gated on ddx's `matmul` gradient program, end to end, at n = 50,000: einfold on must be at least 4× faster than einfold off, on the same DataFusion. ddx is an XQL Systems project for automatic differentiation of SQL queries. This benchmark builds the program as ddx's own benchmark does (`crates/ddx-datafusion/tests/ad_perf.rs`), for `SUM(tanh(a·w)²)` with `a` of `n × 16` and `w` of `16 × 8`. It runs the program with einfold off and on (`einfold_datafusion::install`). It checks that both give the same gradients, within how far two correct evaluations can differ when each adds its sums in its own order, and that einfold's gradients are the same bits on one thread as on every thread.

It is a crate of its own, not a member of einfold's workspace, so that einfold doesn't depend on ddx. It runs on Tokio's multi-threaded runtime, as a host does; ddx's `ad_perf` runs on the single-threaded one, where DataFusion gets one core. JAX on CPU uses every core too: on the same machine, `jax.jit(jax.grad(…))` of this loss takes 1.6 ms.

Run with `cargo run --release [n …]`, with `protoc` on the `PATH` or in `PROTOC`, for ddx's Substrait dependency.

## Result

2026-10-26. Intel Core i7-8700 (6 cores, 12 threads), Rust 1.99, DataFusion 55.2.0, ddx `main` at [921aef1](https://github.com/xqlsystems/ddx/commit/921aef1a34baa3b58ab32a5a68bd663b35060bfd), einfold at the commit that last changed this file. Best of 5.

| step, n = 50,000 | einfold off, ms | einfold on, ms |
|---|---:|---:|
| ddx's checks (each table's keys identify its rows; the cotangent's shape) | 32.1 | 32.1 |
| `saved_0` (forward z = a·w) | 37.7 | 9.3 |
| `saved_1`, `value`, `cotangent_1` | 2.3 | 2.2 |
| `cotangent_0` | 17.0 | 17.4 |
| `grad_0_a` (Ā) | 245.5 | 13.9 |
| `grad_1_w` (W̄) | 33.1 | 14.2 |
| **whole program** | **368.9** | **89.8** |

**4.0–4.1× at n = 50,000 over five runs, so M1's gate, 4×, is passed, just.** At n = 10,000 it's 2.5–2.6× (75 ms against 30).

**Since the fixes for #60–#62, 3.9–4.0×** (360–362 ms against 91.2–91.8, three runs), about 2 ms slower with einfold on: the joined-back steps now compute the regroup's test as a column of the domain (#60), and detection checks that each moved expression can't fail (#61). That is within noise of the gate, but leaves no margin.

The gradients agree within their derived bounds (`bounds` in `src/main.rs`), and einfold's are the same bits on 1 thread as on 12.

einfold finds each gradient step whole: ddx's transposed contraction, joined back to the table, grouped again by the table's keys, and left-joined from it. It runs the step as the contraction with the table as its domain, checking as it runs that the table's keys are unique. That took Ā from 246 ms to 14.

Where the rest is, with einfold on:

- **ddx's checks, 32 ms,** run before the steps. One groups `a`'s 800,000 rows by their keys to check that they identify its rows. It isn't a fold, and einfold leaves it to DataFusion. It checks the same thing einfold's domain check does, so a host running both does that work twice.
- **`cotangent_0`, 17 ms,** is elementwise (the derivative of `tanh(z)²`), not a fold.
- **The three folds, 37 ms between them.** They are mostly linear passes: reading 800,000 rows' keys and factors, and writing 400,000 or 800,000 rows out.

So more than half of what remains is ddx's own work outside the folds. Even if einfold's three folds were free, the program would be about 7× faster, not more.
