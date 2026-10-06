<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# Spikes S2 and S3: how facts and layouts travel through a plan

**Questions.**

- **S2.** If a reader attached einfold facts to Arrow field or schema metadata, would they reach einfold's rewrite, and would they survive Substrait?
- **S3.** Which relational operators keep that metadata? And do the layout propagation rules (design doc §13.5) hold on real plans?

**Answer, in short.**

**S2.** Arrow metadata is not a usable carrier for facts across hosts. It is also not a safe one inside DataFusion.

- **DuckDB drops all field and schema metadata** at the Arrow boundary.
- **Substrait does not carry it.** The plan contains no trace of it. Metadata reappears after a round trip only because the consumer looks the table up in its own catalog again.
- **DataFusion keeps metadata too eagerly.** It survives filters, limits, joins and casts, which change the rows or values a fact describes. Where inputs disagree, it silently keeps one side's.

This settles the carrier question left open in §8.1 in favor of a side channel: facts come from per-reader fact providers keyed by table and column, held in einfold's own fact table, not in the data's schema. Arrow metadata can at most be one *input* to a provider, read at the scan.

**S3.** The §13.5 rules correctly predict which coordinates every operator outputs, on both hosts. But row order does not follow them: DataFusion knows the order and DuckDB keeps it, and neither does both. So a layout has to be split into two facts: the coordinate map, which always propagates, and the row order, which holds only where the host promises it.

Date: 2026-10-06. DataFusion 54.0.0 (Python), DuckDB 1.5.6, pyarrow 25.0.1.

## S2: Arrow metadata as a carrier

### Method

[`carrier.py`](carrier.py) registers two small Arrow tables, `a` and `b`, with columns `i`, `j` and `v`. Every field carries metadata (for example `einfold.fact` on `v`), and each schema carries `einfold.table`. It then:

1. runs one query per operator in DataFusion and DuckDB, and records which output columns still carry metadata, both in DataFusion's plan schema and in the Arrow result;
2. in DataFusion, registers a table `c` whose metadata differs from `a`'s, and runs a union, a join, a cast and a `coalesce`;
3. in DuckDB, checks whether a canonical Arrow extension type (`arrow.uuid`) survives, with and without `arrow_lossless_conversion`;
4. produces Substrait from two DataFusion plans, looks for the metadata in the plan bytes, and consumes the plan again.

Run it with `python carrier.py`.

### Results

"Kept" lists the output columns that still carry field metadata.

| Operator | DataFusion (plan schema and result agree) | DuckDB |
|---|---|---|
| Scan | `i`, `j`, `v`; schema metadata kept | Nothing kept |
| Projection of columns | `i`, `v` | Nothing |
| Projection with renaming (`v AS val`) | `row`, `val`: metadata follows the column | Nothing |
| Projection of an expression (`v * 2`) | `i` only: the expression drops it | Nothing |
| Filter | `i`, `j`, `v` | Nothing |
| Sort, limit | `i`, `j`, `v` | Nothing |
| Join | All four output columns; each keeps its own side's metadata | Nothing |
| Aggregate (`sum(v)`) | Grouping column `i` only | Nothing |
| Join then aggregate (an einsum) | `i`, `j` only; the summed value has none | Nothing |
| Union all | `i`, `j`, `v` | Nothing |
| Window | `i`, `j`, `v`; the window output has none | Nothing |
| `CAST(v AS float)` | Kept | — |
| `coalesce(v, 0)` | Dropped | — |

When inputs disagree (DataFusion):

- **Union.** The output keeps the left input's field and schema metadata. The right input's are dropped without a warning.
- **Join.** Each field keeps its own metadata. Schema metadata comes from the left input only.

DuckDB:

- Plain field and schema metadata are always dropped.
- The canonical `arrow.uuid` extension type comes back as `string` by default, and as `arrow.uuid` with `SET arrow_lossless_conversion = true`. DuckDB keeps only extension types it knows. Registering a new one takes C++ in an extension, not SQL.

Substrait (DataFusion producer, then consumer):

- The plan for `SELECT * FROM a` is 118 bytes and contains no `einfold` metadata. Substrait's `NamedStruct` schema has no metadata field.
- After consuming, the schema has its metadata again, because the consumer resolves table `a` in its own catalog. A consumer on another host, or with a different catalog, would not see it.

### Findings

1. **No cross-host carrier.** Facts written as Arrow metadata do not cross DuckDB or Substrait. einfold's output forms both go through Substrait or a non-DataFusion host, so facts must travel in einfold's own channel: the fact table keyed by table and column (§8.1), and, for the einsum form, fields of the `Einsum` extension relation itself.
2. **Inside DataFusion, metadata is kept by name, not by meaning.** A fact about a column's *meaning* (its dimensions, its layout, its coordinate map) is still true after a filter, sort or rename. A fact about its *rows or values* (density, `count_zero`, min and max, "every chunk written") is false after a filter, a limit, a join or a cast, yet DataFusion keeps it. So einfold must not read row-set facts from metadata below an operator that changes rows. That is the same rule the design applies to statistics: a fact names the plan node it was established at.
3. **Silent conflict resolution.** A union of two inputs with different facts keeps the left one. An einfold rule reading that would apply the left input's facts to the right input's rows. Rules must re-derive facts at a union, not inherit them.
4. **Computed columns lose metadata, as they should.** `v * 2`, `sum(v)` and window outputs have none. einfold has to derive facts for computed columns itself, which it already plans to do (§8.1, fact derivation).
5. **What metadata is still good for.** At the scan, in DataFusion, metadata is a cheap way for a reader to hand facts to a fact provider. That is how a reader could publish the XQL Systems layout convention without einfold depending on it. Above the scan, einfold should ignore it.

## S3: layout propagation rules against real plans

### Method

[`layout_rules.py`](layout_rules.py) writes a dense 1000 × 1000 array `a(i, j, v)` to Parquet in row-major order (`fixture/`, not committed), plus an 8-column `b` for a contraction. For each rule in the design doc's §13.5 table, it runs the query on DataFusion (12 partitions, files declared `WITH ORDER (i ASC, j ASC)`) and DuckDB (12 threads), and checks:

- **support:** are the output coordinates exactly what the rule predicts (a subset, for a filter on values)?
- **row order:** without `ORDER BY`, do rows arrive in the predicted layout's order, row-major over the remaining dimensions?
- **order known to the planner:** with `ORDER BY` in that order, does the plan still contain a sort?

### Results

| Rule (§13.5) | Support (DataFusion / DuckDB) | Row order without `ORDER BY` | Planner knows the order |
|---|---|---|---|
| Filter on a dimension range | yes / yes | **no** / yes | yes / **no** |
| Strided slice (`i % 4 = 0`) | yes / yes | **no** / yes | yes / **no** |
| Filter `dimension = constant` | yes / yes | yes / yes | yes / **no** |
| Filter on a value column | yes (subset) / yes | **no** / yes | yes / **no** |
| Projection dropping a value column | yes / yes | yes / yes | yes / **no** |
| Transpose (reorder dimension columns) | yes / yes | yes / yes | yes / **no** |
| Union along a dimension | yes / yes | **no** / yes | yes / **no** |
| Contraction node | yes / yes | **no** / **no** | **no** / **no** |

### Findings

6. **The rules are right about support.** Every rule predicted exactly which coordinates come out, on both hosts. That part of each layout (its shape, offset and strides as a map from coordinates) can be propagated as §13.5 says.
7. **Row order is a separate fact, and hosts treat it differently.**
   - *DataFusion* knows the order. It tracks the declared order through filters, projections and unions, and answers `ORDER BY i, j` with a sort-preserving merge, never a sort. But without `ORDER BY` it spreads rows round-robin over its partitions, so arrival order is scrambled.
   - *DuckDB* is the reverse. It keeps insertion order (`preserve_insertion_order`, on by default) through every operator except the aggregate, but its planner doesn't know the order and always sorts for `ORDER BY`.
   - Neither keeps order through a hash aggregate, which is what a contraction node is.
8. **So einfold must split the layout into two facts:** the coordinate map, which the §13.5 rules propagate on every host; and the row order, which is only true where the host promises it. On DataFusion, einfold can ask for the order: a required ordering costs a merge, not a sort, because the planner already knows it. On DuckDB, einfold should not rely on order unless it adds the `ORDER BY` and pays for the sort. Either way, a contraction node's output order is whatever EinFold emits (§10.3), and only the einsum form can promise it.

## Limitations

- One file, one machine. Multiple files, hive partitioning and remote stores were not tested; DataFusion only keeps a declared order across files whose ranges don't overlap.
- The order checks ran without `ORDER BY`, so a "yes" for DuckDB is an observed behavior with `preserve_insertion_order` on, not a documented guarantee for every operator.
- Only Python bindings were used. DataFusion's Rust API behaves the same way, since the Python bindings wrap it, but extension points that exist only in Rust (for example, a custom `TableProvider` adding metadata) were not exercised.
- Substrait was tested only with DataFusion's producer and consumer. DuckDB's Substrait extension was not tested; given finding 1, it would not change the answer.
- GQE, Sirius and gpudb were not tested. They consume Substrait or DuckDB plans, so finding 1 applies to them.
