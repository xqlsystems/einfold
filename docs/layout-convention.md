<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# XQL Systems Layout Convention for Zarr (draft)

- **UUID**: f4cbf091-d7c1-40f7-80ba-c2ffdc4dcf69
- **Name**: `layout:`
- **Owner**: XQL Systems
- **Schema URL**: to be published in a proposed `xqlsystems/zarr-layout-convention` repository
- **Spec URL**: same repository
- **Scope**: Array, Group
- **Maturity**: Proposal
- **Status**: draft 1.1, for discussion. Author: Alex Merose. 2026-10-04.

## 1. Background

**Zarr** is a storage format for large N-dimensional arrays, widely used for climate, weather, and imaging data. Zarr splits each array into fixed-size **chunks**, stores each chunk as a separate compressed object (on disk or in cloud object storage), and describes the array in JSON metadata: its shape, chunk shape, data type, dimension names, and **fill value** (the value returned for chunks that were never written). Arrays live in a hierarchy of **groups**. Both arrays and groups can carry free-form **attributes**.

A **Zarr convention** is a published, named set of attributes that gives arrays or groups extra meaning, without changing how they are stored. The [Zarr Conventions Specification](https://github.com/zarr-conventions/zarr-conventions-spec) says how to declare one: list it in a `zarr_conventions` attribute, and prefix its attribute names (here, `layout:`) so they don't collide with other conventions. Tools that don't know a convention can ignore it safely.

**XQL Systems** builds SQL access to Zarr data. Its readers (such as xarray-sql and duckdb-zarr) turn arrays into tables with one row per combination of coordinates, so that SQL engines can query them. This convention is owned by XQL Systems. It lets the writer of a dataset tell those readers, and the engines behind them, how the data is ordered and summarized, so that queries scan less and run faster.

**CuTe** is the layout library inside NVIDIA's CUTLASS, a C++ library of fast matrix-multiply kernels for GPUs. In CuTe, a **layout** is a function from coordinates to positions, written as a **shape** and a **stride**. Shapes and strides may be nested to describe hierarchies of tiles. Layouts are a few integers, they are computed rather than looked up, and CuTe defines an algebra for slicing, tiling, and combining them. This convention writes orderings as CuTe layouts.

## 2. What the convention describes

- **Curve dimensions.** A stored dimension whose positions trace a curve through several logical dimensions. For example, a 2-D grid stored as a 1-D array in Z-order (also called Morton order), which visits the grid in recursive 2×2 blocks so that nearby cells stay nearby in storage.
- **Chunk order.** The order in which readers should visit chunks.
- **Written chunks.** A record of which chunks were written, so readers need not list every object in the store.
- **Chunk summaries.** Per-chunk statistics, such as the minimum and maximum value. Databases call these **zone maps**, and use them to skip chunks that cannot match a filter.

## 3. Design rules

1. **Interpretation only.** As the Zarr Conventions Specification requires, this convention does not change how data are encoded or stored. Every ordering in it is over the array's *logical* positions: its indices, linearized in C order (row-major, last dimension fastest) over its declared dimension names. The order of bytes inside a chunk belongs to the array's codecs (the encoding steps Zarr applies to each chunk, such as compression or the `transpose` codec) and is out of scope.
2. **Safely ignorable.** A reader that ignores this convention still reads correct data. It only loses performance.
3. **Reuse other conventions.** Maps from positions to coordinates belong to the [`spatial`](https://github.com/zarr-conventions/spatial) convention (affine maps for the two horizontal axes) and to CF coordinate variables. CF, the Climate and Forecast conventions, is the standard metadata vocabulary for climate data, and it stores coordinates as arrays alongside the data. Missing-data sentinels belong to the [`missing_value`](https://github.com/zarr-conventions/missing_value) convention. Auxiliary arrays may be declared with the [`dependent-arrays`](https://github.com/zarr-conventions/dependent-arrays) convention, which stores them under the main array. New properties are added only where no existing convention fits.
4. **Say how each fact is known.** Orderings are exact by construction. Written-chunk records and summaries describe the data at some point in time, and can go stale when the array changes. Each one says whether it is exact, and as of when (section 5.5).

## 4. Registration

```json
{
  "zarr_conventions": [
    {
      "uuid": "f4cbf091-d7c1-40f7-80ba-c2ffdc4dcf69",
      "schema_url": "https://raw.githubusercontent.com/xqlsystems/zarr-layout-convention/refs/tags/v0/schema.json",
      "spec_url": "https://github.com/xqlsystems/zarr-layout-convention/blob/v0/README.md",
      "name": "layout:",
      "description": "XQL Systems: orderings, chunk order, written chunks, and chunk summaries for efficient scans"
    }
  ]
}
```

The URLs point to the proposed repository and will resolve once it is published. As the Zarr Conventions Specification requires, tools identify the convention by its UUID, not by the `layout:` prefix.

## 5. Properties

| Property | Type | Description | Required |
|---|---|---|---|
| `layout:version` | `string` | Version of this convention | Yes |
| `layout:curve` | object | Declares that stored dimensions are a curve over logical dimensions (5.2) | No |
| `layout:chunk_order` | layout object | Preferred order for visiting chunks (5.3) | No |
| `layout:written_chunks` | object | Where the record of written chunks is (5.4) | No |
| `layout:summaries` | object | Per-chunk statistics arrays (5.4) | No |

### 5.1 Layout objects

Several properties hold a **layout object**, which describes a function from logical positions to positions in some target range.

| Field | Type | Description |
|---|---|---|
| `type` | `string` | `"cute"` (default), `"cute+swizzle"`, `"lookup"`, or a registered curve name such as `"hilbert"` |
| `dimensions` | `string[]` | The logical dimensions, in order. Each one is one top-level mode (component) of the layout. |
| `shape` | nested `integer` arrays | CuTe shape, one top-level entry per dimension. For example `[[2,2],[2,2]]`. |
| `stride` | nested `integer` arrays | CuTe stride, with the same nesting as `shape` |
| `swizzle` | `integer[3]` | For `"cute+swizzle"`: the parameters `B, M, S` of CuTe's `Swizzle`, a bit-mixing function applied after the layout |
| `array` | `string` | For `"lookup"`: the path of an integer array that holds the target position of each logical position |
| `order` | `integer` | For named curves: the curve's order (bits per dimension) |
| `extents` | object | Optional. Map from dimension name to its true extent, when the layout's shape pads it (for example to a power of two). Positions at or beyond the true extent are skipped. |

**Evaluating a CuTe layout.** Split a position `p` along a dimension whose shape is `(s₀, s₁, …)` into digits, least significant first: `p = d₀ + s₀·d₁ + s₀·s₁·d₂ + …`. The target position is the sum of every digit times its stride, over all dimensions. This is CuTe's standard meaning of a layout.

**Validity.** A layout object MUST be a one-to-one map from its domain onto `[0, total size)`. For CuTe layouts, the product of each dimension's shape is the size of its domain along that dimension. That product MUST equal the dimension's extent, or be at least the extent when `extents` declares padding.

### 5.2 `layout:curve`

Declares that the array's stored dimensions are a curve through a set of logical dimensions. Reading the stored array in C order then visits the logical positions in curve order.

| Field | Type | Description |
|---|---|---|
| `curve_dimensions` | `string[]` | Stored dimensions (from the array's dimension names) that make up the curve |
| `logical` | object | Map from logical dimension name to its extent |
| `map` | layout object | From logical positions to the linearized positions of `curve_dimensions` |
| `padded` | `boolean` | True if logical extents were padded (for example to powers of two). Padded positions hold the fill value and are not part of the data. |

A reader that understands this property:

- exposes the logical dimensions as columns, computing each row's logical positions from its stored position with the inverse of the layout, and its coordinates through the usual coordinate maps;
- exposes the curve dimensions only on request;
- drops padded positions;
- answers range filters on logical dimensions using the layout: a box of logical positions maps to a set of ranges along the curve, and so to a set of chunks.

The table a reader produces is unchanged: one row per logical coordinate tuple. Only the storage order differs.

### 5.3 `layout:chunk_order`

The order in which readers should visit chunks, as a layout object over the chunk grid (the grid of chunk indices). For example, it can declare Z-order over the spatial chunk grid while keeping row-major order inside each chunk. This changes nothing in storage. It tells readers which chunk order keeps neighboring chunks together, which helps caching, streaming, and pruning.

### 5.4 `layout:written_chunks` and `layout:summaries`

`layout:written_chunks`:

| Field | Type | Description |
|---|---|---|
| `array` | `string` | Path of a boolean array over the chunk grid; `true` means the chunk was written |
| `exact` | `boolean` | See 5.5 |
| `as_of` | `string` | Optional. See 5.5 |

A chunk that was not written reads as the array's fill value. What that value means for a query (zero, missing, or NaN) is decided by the fill value, the `missing_value` convention, and the reader, not by this convention. On cloud object stores, this record saves listing every chunk.

`layout:summaries`:

| Field | Type | Description |
|---|---|---|
| `statistics` | object | Map from statistic name (`"min"`, `"max"`, `"count_valid"`, `"count_zero"`, …) to the path of an array over the chunk grid. `"count_zero"` counts values exactly equal to zero, which lets readers prove that a chunk is all zeros. |
| `ignores` | `string[]` | Values excluded when computing the statistics, such as `"missing_value"` and `"NaN"` |
| `exact` | `boolean` | See 5.5 |
| `as_of` | `string` | Optional. See 5.5 |

Summaries serve filters on values, such as `WHERE temperature > 40`. Dimensions need no summaries, because each chunk's coordinate range follows from the coordinates and the chunk grid.

### 5.5 Staleness: `exact` and `as_of`

Written-chunk records and summaries describe the data as it was when they were written. Two fields say how far a reader may trust them:

- `exact`: the writer's promise that the record matches the data.
- `as_of`: optional. The version of the data the record describes, in whatever form the store versions data. In Icechunk (a versioned, transactional storage layer for Zarr), this is a snapshot ID. An Icechunk writer can update the data and the record in one transaction, so the record stays exact.

Readers apply these rules:

| `exact` | `as_of` | Reader may |
|---|---|---|
| true | present, and matches the version being read | trust it: skip chunks, prune, and estimate costs |
| true | present, but cannot be confirmed or does not match | use it only as a hint: ordering and cost estimates, never skipping |
| true | absent | trust it. The writer promises to update it with every write |
| false or absent | any | use it only as a hint |

Writers that cannot keep a record in sync with the data MUST set `exact` to false, or omit the record.

## 6. Examples

### 6.1 A 4×4 Z-order curve

A 16-element array stored with dimension names `["z"]`, holding a 4×4 grid over `(y, x)` in Z-order, with `x` in the even bits:

```json
{
  "zarr_conventions": [ { "uuid": "f4cbf091-d7c1-40f7-80ba-c2ffdc4dcf69", "name": "layout:" } ],
  "layout:version": "0",
  "layout:curve": {
    "curve_dimensions": ["z"],
    "logical": { "y": 4, "x": 4 },
    "map": {
      "type": "cute",
      "dimensions": ["y", "x"],
      "shape":  [[2, 2], [2, 2]],
      "stride": [[2, 8], [1, 4]]
    },
    "padded": false
  }
}
```

In CuTe's own notation this is `((2,2),(2,2)):((2,8),(1,4))`. Row `y = 0` lands at positions `0, 1, 4, 5`, and row `y = 1` at `2, 3, 6, 7`: the familiar Z pattern.

### 6.2 Z-order over the chunks of a global weather array

ERA5 is a widely used hourly global weather dataset with a 0.25° grid. Take one year of one ERA5 variable, with dimension names `["time", "lat", "lon"]`, shape `[8760, 721, 1440]`, and chunks `[24, 128, 128]`. The chunk grid is `[365, 6, 12]`. The writer wants readers to visit spatial chunks in Z-order within each day. Z-order needs power-of-two extents, so the spatial chunk grid is padded to `8 × 16`, and `extents` records its true size.

```json
{
  "layout:version": "0",
  "layout:chunk_order": {
    "type": "cute",
    "dimensions": ["time", "lat", "lon"],
    "shape":  [365, [2, 2, 2], [2, 2, 2, 2]],
    "stride": [128, [2, 8, 32], [1, 4, 16, 64]],
    "extents": { "lat": 6, "lon": 12 }
  },
  "layout:written_chunks": {
    "array": "temperature_written",
    "exact": true,
    "as_of": "<icechunk snapshot id>"
  },
  "layout:summaries": {
    "statistics": { "min": "temperature_min", "max": "temperature_max" },
    "ignores": ["missing_value", "NaN"],
    "exact": true
  }
}
```

The `lat` and `lon` digits interleave as in 6.1, with `lat` in the odd bits. `time` varies slowest, with stride `128 = 8·16`. Readers skip the padded positions.

### 6.3 A Hilbert curve, stored as a lookup

A Hilbert curve keeps neighbors even closer together than Z-order, but it is not a CuTe layout, because it rotates and reflects at each level. The writer stores the map as an integer array and points to it:

```json
{
  "layout:version": "0",
  "layout:curve": {
    "curve_dimensions": ["h"],
    "logical": { "y": 1024, "x": 1024 },
    "map": { "type": "lookup", "dimensions": ["y", "x"], "array": "hilbert_position" },
    "padded": false
  }
}
```

## 7. What readers derive

A reader that implements this convention can tell its query engine the following, without scanning any data. In DataFusion (an extensible SQL engine that several XQL readers use), these become partitions, statistics, declared orderings, and filter pushdown (the reader skipping data a filter rules out).

| Derived | From |
|---|---|
| One partition per chunk, visited in `chunk_order` | chunk grid, `layout:chunk_order` |
| Exact minimum and maximum of each logical dimension, per partition | chunk grid, `layout:curve`, coordinate maps |
| Which chunks match a range filter on dimensions | the inverse layout |
| Which chunks match a filter on values | `layout:summaries`, trusted per 5.5 |
| Which chunks hold only the fill value | `layout:written_chunks`, trusted per 5.5 |
| A declared output ordering, where the layout is row-major in the logical dimensions | `layout:curve` |

Engines that know nothing about layouts benefit through these standard hooks. Systems that do understand layouts can use the orderings directly. One example is einfold, an XQL Systems project that rewrites query plans to speed up tensor computations. It can use them to choose dense kernels or streaming algorithms.

## 8. Relationship to other conventions

| Convention | Relationship |
|---|---|
| [`spatial`](https://github.com/zarr-conventions/spatial) | Supplies affine maps from positions to X/Y coordinates. Used with `layout:curve` to give logical dimensions their coordinates. |
| CF coordinate variables | Supply maps from positions to coordinates for other dimensions |
| [`missing_value`](https://github.com/zarr-conventions/missing_value) | Says which stored values mean "missing". Readers combine it with the fill value and `layout:written_chunks`. |
| [`dependent-arrays`](https://github.com/zarr-conventions/dependent-arrays) | May declare the auxiliary arrays (written-chunk record, summaries, lookups) so that they live under the main array |
| [`multiscales`](https://github.com/zarr-conventions/multiscales) | Describes image pyramids (the same data at several resolutions). Each resolution level may carry its own `layout:` properties. |

Discrete global grid systems (DGGS) also order cells along curves. They are out of scope for this draft.

## 9. Open questions

1. **Named curves.** Which curves that are not CuTe layouts (Hilbert, Peano) deserve registered types with closed-form maps, so that no lookup array is needed?
2. **Irregular chunk grids.** If Zarr adopts chunk grids whose chunk sizes vary along a dimension, `chunk_order` needs a form that does not assume a regular grid.
3. **Coordinates for curve dimensions.** Should `layout:curve` restate coordinate maps for logical dimensions, or always defer to `spatial` and CF?
4. **Summary types.** Should summaries allow sketches, such as Bloom filters (compact, approximate set-membership tests) or histograms, as well as minimum and maximum?
5. **How layouts are written in JSON.** Nested arrays, as here, or CuTe's string form `((2,2),(2,2)):((2,8),(1,4))`, or both?

## 10. References

- Zarr Conventions Specification: https://github.com/zarr-conventions/zarr-conventions-spec
- Zarr v3 core specification: https://zarr-specs.readthedocs.io/en/latest/v3/core/index.html
- `spatial` convention: https://github.com/zarr-conventions/spatial
- `missing_value` convention: https://github.com/zarr-conventions/missing_value
- `dependent-arrays` convention: https://github.com/zarr-conventions/dependent-arrays
- `multiscales` convention: https://github.com/zarr-conventions/multiscales
- CF (Climate and Forecast) conventions: https://cfconventions.org
- CuTe layout algebra (NVIDIA CUTLASS): https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/cute/02_layout_algebra.md
- CuTe layouts as tensor indexes: https://github.com/NVlabs/CuTe/issues/4
- Icechunk: https://icechunk.io
- XQL Systems: https://xql.systems
