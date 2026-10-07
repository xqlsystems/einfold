<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# einfold spikes

A spike is a short, time-boxed experiment that answers one design question (design doc §13). Each folder holds the code and a report (`README.md`) with method, results, findings and limitations.

| Spike | Topic | Status | Headline |
|---|---|---|---|
| [S1, S13, S14](s01-readers/README.md) | What readers pass to the plan | Done | No reader carries facts in Arrow metadata; readers disagree on NULL vs. NaN, lower-dimensional variables, statistics, pushdown and time types |
| [S2, S3](s02-carrier/README.md) | Fact carriers; layout propagation | Done | Arrow metadata is lost in DuckDB and Substrait, and kept too eagerly in DataFusion, so facts use a side channel. Layout rules hold for coordinates; row order is host-specific |
| [S4](s04-coords/README.md) | Coordinate maps in ERA5 and CMIP6 | Done | Affine must be bitwise; calendar maps for monthly times; sorted tables are the general exact form |
| [S5](s05-gpudb/README.md) | gpudb shapes | Done (Colab T4) | 1 of 13 einfold shapes ran on GPU (`BIGINT` reduction, 4.3×); `DOUBLE` sums, expanding joins, subquery operands and windows are declined |
| S6 | GQE and Substrait extension relations | Blocked | Needs access to NVIDIA's GPU Query Engine |
| [S7](s07-unparser/README.md) | DataFusion's Unparser to DuckDB | Done | Unoptimized plans round-trip (34 of 34); optimized plans don't (31 rejected, one silently wrong) |
| [S8](s08-deterministic-sums/README.md) | Cost of deterministic sums | Done | Binned sums: deterministic and accurate, 3–4.4× a plain sum on CPU, 3.5–3.6× on GPU (GTX 1080 Ti, T4) |
| S9 | Zax-SQL | Blocked | Needs an Earthmover account |
| [S10](s10-plan-protection/README.md) | Plan protection | Done | Hosts keep CTE orders; DuckDB needs `MATERIALIZED` (56 s → 0.09 s planning) |
| [S11](s11-thresholds/README.md) | Dense vs. hash thresholds | Done | Dense wins above ~20% density vs. Gustavson; Gustavson beats hash join + aggregate by 2.5–17× |
| [S12](s12-aggregate-pushdown/README.md) | Reader aggregate pushdown | Done | Replace DataFusion's `Partial` aggregate via an optimizer rule; on DuckDB, rewrite SQL to a reader table function |
| S15 | Sirius | Blocked | Needs a GPU of compute capability 7.5+ and a libcudf build: a Colab T4 or a cloud VM |
| [S16](s16-egglog/README.md) | egglog as the rewrite engine | Done | Adopt egglog in a hybrid design |
| [S17](s17-nary/README.md) | n-ary sum-product nodes | Done | One iteration, linear growth to 1000 operands; distributivity needs a guard |
| [S18](s18-planning-time/README.md) | egglog planning time | Done | 0.5–1.2 ms per query with rules preloaded, under the hosts' own planning time |
| [S19](s19-float-sums/README.md) | Float sums in hosts and JAX | Done | No SQL engine guarantees repeatable float sums; determinism is a setting |
| [S20](s20-tiles/README.md) | Tiles in egglog | Done | Reproduces Cubed's plans exactly; never exceeds the budget; finds cheaper plans |

Machine for the local spikes: Intel Core i7-8700 (6 cores, 12 threads), 15 GB RAM, NVIDIA GTX 1080 Ti, NixOS.
