<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S30: folds over joins by discrimination

**Question.** Is discrimination a better general algorithm for einfold's folds than its hash path, and does it beat the host? Discrimination is Fritz Henglein's linear-time sorting and partitioning without comparisons or hashing ("Generic Discrimination", ICFP 2008). Henglein and Larsen built SQL-style equijoins on it ("discrimination-based joins", 2011). The maintainer asked for it on #44, as another algorithm, before the M1 stack merges.

**Answer.** Yes on both counts, except where einfold's dense algorithm already applies.
- **Against the hash path:** 9–34× faster on every workload.
- **Against DataFusion:** faster in 11 of 12 cases, by 1.1–2.7×, from Arrow rows to Arrow rows, on the same 12 threads. The exception is the matrix product with sparse integer keys at n = 50,000: 51–56 ms against DataFusion's 41–49.
- **With dense integer keys,** einfold's dense algorithm is still 3× faster than discrimination.

So it is worth building as `Algorithm::Discriminate`. It would replace the hash path as the algorithm for any keys, and could replace the host's plan as the fallback where it wins.

| workload | keys | n | output rows | discrimination, ms | einfold today, ms (algorithm) | DataFusion, ms |
|---|---|---:|---:|---:|---:|---:|
| matrix product | dense integers | 10,000 | 80,000 | 5.6–6.6 | 2.1–2.4 (dense) | 9.8–10.5 |
| matrix product | dense integers | 50,000 | 400,000 | 30.6–35.5 | 9.6–10.4 (dense) | 38.9–44.5 |
| matrix product | sparse integers | 10,000 | 80,000 | 7.9–8.2 | 204–244 (hash) | 9.9–10.8 |
| matrix product | sparse integers | 50,000 | 400,000 | 51.0–55.7 | 1,635–1,719 (hash) | 41.4–48.9 |
| matrix product | text | 10,000 | 80,000 | 9.8–10.9 | 136–139 (hash) | 14.5–16.3 |
| matrix product | text | 50,000 | 400,000 | 57.8–57.9 | 1,058–1,069 (hash) | 65.8–73.7 |
| sparse product | dense integers | 10,000 | 637,678 | 13.5–13.7 | 113 (hash) | 28.6–29.4 |
| sparse product | dense integers | 50,000 | 3,197,735 | 112–118 | 890–898 (hash) | 143–144 |
| sparse product | sparse integers | 10,000 | 637,678 | 15.7–16.1 | 136–137 (hash) | 29.3–29.5 |
| sparse product | sparse integers | 50,000 | 3,197,735 | 105–106 | 1,019–1,030 (hash) | 147–148 |
| sparse product | text | 10,000 | 637,678 | 21.4–21.5 | 119–120 (hash) | 37.1–37.4 |
| sparse product | text | 50,000 | 3,197,735 | 137–146 | 870–909 (hash) | 185 |

Ranges are over two runs, each the best of five. All three agree on every output row, to within `1e-9` relative.

The workloads compute `SELECT a.x, b.y, SUM(a.v * b.v) FROM a JOIN b ON a.z = b.z GROUP BY a.x, b.y`:
- **the matrix product** is ddx's `matmul` shape, `n × 16` by `16 × 8`. ddx is an XQL Systems project for automatic differentiation of SQL queries;
- **the sparse product** is `n × n` by `n × n` with up to 8 entries per row of each, at random, which is graph-like.

The keys come in three forms:
- **dense integers** `0..k`, which einfold's dense algorithm takes;
- **sparse integers** spread over `0..2⁴⁰`, which it declines;
- **text** (`k0000123`).

"einfold today" is `einfold_kernels::fold` at the top of the M1 stack (#57), which picks dense or hash. DataFusion 55.2.0 runs the SQL on 12 partitions of in-memory tables.

## How it works

`src/main.rs`, about 640 lines.

1. **Ranks.** For each of `x`, `z` and `y`, every distinct key gets a rank `0..k`, in key order. `z` is ranked over both tables at once, so equal keys share a rank.
   - **Integer keys** map to `u64`s that order as the integers do. They are sorted as `(word, row)` pairs by a least-significant-byte-first radix sort. Each pass counts and scatters by chunks of rows on 12 threads, and a pass over a byte that no key varies in is skipped.
   - **Other keys** go through Arrow's row format, byte strings that compare as the keys do. If every row has one width and at most 8 bytes vary, those bytes are packed into a `u64` and take the same path. Text instead takes a most-significant-byte-first radix sort, one counting pass per byte, recursing into buckets and finishing buckets of 16 or fewer by comparison.
2. **Buckets.** `a`'s rows are put in `(x, z)` order and `b`'s in `(z, y)` order by two stable counting sorts each.
3. **Gustavson's product** over the buckets, for blocks of 256 `x` ranks on the threads in turn: each of `x`'s `(z, v)` adds `v` times `b`'s bucket `z` into a dense accumulator over `y`. The touched `y`s are listed and sorted, so the accumulator needs no clearing.
4. **Output.** Each rank's key is taken from one row that has it.

Every addition happens in an order fixed by the keys' values, so the result doesn't depend on how rows were batched or ordered. That is what the hash path pays an O(n log n) sort for today.

## What's not done

This spike computes the plain fold only. To become an algorithm it needs:
- **SQL's corners,** which the hash path already has:
  - NULL keys, under `=` and `IS NOT DISTINCT FROM`;
  - the host's comparison of `-0.0` and `0.0` (`FloatKeys`), since the row format tells them apart;
  - domains.
- **Duplicate key tuples:** bag semantics already hold, since every pair counts. But duplicates add in the order they arrived, so for determinism they need a tiebreak by their values.
- **Ranking several dimensions at once:** a key of several columns is one row-format string, which takes the text path.
- **Speed:**
  - ranks take about two thirds of the time on the matrix products;
  - the three discriminations run at once, each on 12 threads, so they compete;
  - the counting sorts are single-threaded;
  - most-significant-byte-first sorting of text spends its time in per-row lookups into the row format.

## Reproduce

```sh
CARGO_TARGET_DIR=<einfold>/target cargo run --release
```

`Cargo.lock` is the workspace's own, so DataFusion's build is shared with einfold's. `S30_ONLY=matmul,Sparse,50000` runs one case, and `S30_PHASES=1` prints where its time goes.

Run on the machine in [the spikes index](../README.md), 2026-10-27.
