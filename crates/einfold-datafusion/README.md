<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# einfold-datafusion

**einfold for [Apache DataFusion](https://datafusion.apache.org/): array algorithms for tensor computations written in SQL.**

Matrix products, tensor contractions and attention can all be written in SQL, as joins and aggregates. DataFusion runs them as SQL: it builds every row of the join, then hashes each into its group. einfold finds these computations in DataFusion's plans and runs them with array algorithms instead: dense kernels where the keys are small integers, sorting-based ones for any other keys, and a fused, tiled kernel for attention that never holds all of its scores. Its answers are DataFusion's, up to the rounding of floating-point sums.

```rust
use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::SessionContext;

let state = einfold_datafusion::install(SessionStateBuilder::new().with_default_features()).build();
let ctx = SessionContext::new_with_state(state);
```

Every query in that session is now planned with einfold. Whatever einfold doesn't recognize runs as DataFusion would run it.

einfold 0.1 supports DataFusion 55, from 55.2. See [einfold's README](https://github.com/xqlsystems/einfold#readme) for what it finds and how fast it is, and the [`einfold` Python package](https://pypi.org/project/einfold/) for DataFusion's Python bindings.

Apache-2.0.
