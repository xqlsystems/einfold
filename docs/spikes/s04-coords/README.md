<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spike S4: are coordinates affine in real datasets?

**Questions.** How often are coordinates affine (`value = start + k × step` for index `k`) in real datasets such as ERA5 and CMIP6? How should irregular coordinates be represented?

**Answer, in short.** Affine is common but far from universal, and "nearly affine" is a trap.

- **Times are affine in ERA5,** bitwise, in their stored units. Monthly CMIP6 times are not affine in days (months differ in length), but they are affine in calendar months.
- **Horizontal grids:** about half the atmosphere models and almost no ocean models have bitwise-affine latitude and longitude. The rest use Gaussian latitudes (monotone, not affine), curvilinear ocean grids (two-dimensional latitude and longitude), or unstructured meshes.
- **Vertical levels are never affine.**
- **Nine grids are affine only up to rounding,** including three of four ERA5 stores tested. An exact fact must not call them affine.
- **Every 1-D coordinate but one is strictly monotone,** and none of the non-affine ones has more than 2,160 entries.

So the general exact representation of a coordinate map is a **sorted lookup table** from index to value. An affine map, or a calendar map, is a compressed form of it, used only when it reproduces the stored values bitwise. Curvilinear and unstructured grids have no coordinate map over their dimensions; their dimensions are positions, and geography is ordinary data.

Date: 2026-10-06. Data read anonymously from Google Cloud: ARCO-ERA5 (`gs://gcp-public-data-arco-era5/ar/`) and Pangeo's CMIP6 copy (`gs://cmip6`, catalog `cmip6-zarr-consolidated-stores.csv`). xarray with zarr 3.4, cftime.

## Method

[`coords.py`](coords.py) opens each store's metadata and reads only its coordinate arrays:

- **ERA5:** four stores, the 0.25° `full_37-1h-0p25deg-chunk-1.zarr-v3` and three WeatherBench-style regridded stores (64 × 32, 240 × 121, 512 × 256). The model-level store did not return coordinates within the time box and was dropped.
- **CMIP6:** for every model with a `historical` run, the latest store of monthly air temperature (`Amon/ta`, 64 models) and of ocean potential temperature (`Omon/thetao`, 59 models). One store per model and table, so each model's grid counts once. 127 stores in all.

Each index coordinate (a 1-D coordinate named after its dimension) is classified, in its stored type, as:

| Class | Meaning |
|---|---|
| exact | `cast(start + k × step) == stored` for every `k`, bitwise |
| linspace | `cast(linspace(first, last, n)) == stored`, bitwise |
| rounding | within 10⁻⁶ of a step of an affine map, but not bitwise |
| monthly | (times only) not affine in stored units, but consecutive calendar months |
| monotone | strictly increasing or decreasing, not affine |
| other | not monotone |

Times are checked in their stored CF units (for example `hours since 1900-01-01`), not decoded, because that is what a reader can map exactly. Coordinates that are just positions (`i`, `j`, `x`, `y` holding `0, 1, 2, …`) are counted as grid indices. A store has a curvilinear grid if it has a 2-D latitude, and an unstructured mesh if its 1-D latitude runs over a cell dimension.

Run it with `python coords.py cmip6-zarr-consolidated-stores.csv 1000`. Per-coordinate results are in [`results.csv`](results.csv).

## Results

ERA5, per coordinate:

| Store | time | latitude | longitude | level |
|---|---|---|---|---|
| 0.25°, 1-hourly (`full_37-…-v3`) | exact (1,323,648) | exact (721) | exact (1440) | monotone (37) |
| 512 × 256, 6-hourly | exact | rounding | rounding | monotone (13) |
| 240 × 121, 6-hourly | exact | rounding | rounding | monotone (13) |
| 64 × 32, 6-hourly | exact | rounding | rounding | monotone (13) |

Horizontal grid, per store:

| Source | Affine, bitwise | Affine up to rounding | Monotone, not affine | Curvilinear (2-D) | Unstructured mesh |
|---|---|---|---|---|---|
| CMIP6 atmosphere (64) | 30 | 5 | 28 | 0 | 1 |
| CMIP6 ocean (59) | 9 | 1 | 2 | 44 | 3 |
| ERA5 (4) | 1 | 3 | 0 | 0 | 0 |

Time and vertical coordinates:

| Source | Time | Vertical |
|---|---|---|
| CMIP6 atmosphere | 4 exact, 60 monthly | 64 monotone (pressure levels) |
| CMIP6 ocean | 3 exact, 55 monthly, 1 not monotone | 57 monotone (depths), 2 other monotone (density, depth) |
| ERA5 | 4 exact | 4 monotone |

Sizes: the largest monotone-but-not-affine coordinate has 1440 entries, the largest "rounding" one 512, and the longest monthly time 2160. Curvilinear grids have 360 to 1.7 million points.

The one non-monotone coordinate is the time of one NorESM2-MM ocean store.

## Findings

1. **"Affine" must mean bitwise, or it is not an exact fact.** Nine grids look affine to any plotting tool but differ from every affine formula in the last bits. If einfold claimed them affine, a filter `lat = 45.0` rewritten into an index computation could select a different row than the engine would. Such a grid can carry an Estimate-level affine fact for costing, but its exact map is the lookup table.
2. **Sorted lookup tables are the general exact representation.** All but one 1-D coordinate is strictly monotone, and the non-affine ones are small (at most a few thousand entries). With a sorted table, a range filter on coordinate values becomes an index range by binary search, exactly, which is all that slicing and tiling (§8.2, §9.4) need. Affine is an optimization of it: no table, and closed-form alignment between two datasets.
3. **Times need a calendar map.** 115 of 123 CMIP6 time axes are monthly. They are affine in calendar months, not in their stored days, and the length of a month depends on the calendar (`noleap`, `360_day`, `gregorian`). A map of the form "month `k` after a start month, in calendar `c`" is exact and tiny. It is the second compressed form worth having.
4. **On curvilinear and unstructured grids, the dimensions are positions.** 47 of 59 ocean models (and one atmosphere model) index space by `i`, `j` or a cell number, with latitude and longitude as 2-D or per-cell data. There is no coordinate map to infer: the layout's dimensions are the indices, which are trivially affine. Geographic selection is a filter on data columns, and pruning by it needs per-chunk min and max statistics (§9.2), not a map.
5. **Vertical levels are always lookup tables.** None is affine, all are short (13 to 75 entries in this sample).
6. **Reader behavior matters as much as the data.** The maps above are over stored values. Readers decode times differently (spike S1): xarray-sql decodes to nanosecond timestamps, duckdb-zarr keeps raw integers. A map is exact only for the representation the reader emits, so the fact provider must state which.

## Proposed representation

A coordinate map for one dimension is one of:

| Form | Exact when | Size |
|---|---|---|
| Affine: start, step, stored type | `cast(start + k × step)` reproduces every stored value | constant |
| Calendar: start month, calendar, day-of-month rule | times are consecutive calendar months | constant |
| Sorted table: the stored values | always, for a strictly monotone coordinate | `n` values |
| None | the coordinate is not monotone, or is 2-D or per-cell | — |

A fact provider tries the forms in that order and records the first that verifies against the stored values. Verification is a one-time pass over the coordinate, which is small.

## Limitations

- Only monthly CMIP6 tables and four ERA5 stores. Daily and sub-daily CMIP6 tables, regional models and observational products were not sampled.
- CMIP6 grids were taken one per model, from the `gn` or `gr` grid each model's latest store used; regridded variants of the same model were not compared.
- Times were decoded with `cftime`; the monthly test checks consecutive months, not the day within the month.
