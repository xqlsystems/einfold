<!--
SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors

SPDX-License-Identifier: Apache-2.0
-->

# einfold-ir

**einfold's engine-independent descriptions of tensor computations found in SQL plans.**

A fold is a sum of products over a join, SQL's spelling of a tensor contraction such as a matrix product. A fold pipeline is a fold whose results another fold reads, such as attention's scores and its weighted sum of values. This crate describes both, independently of any query engine, so that every engine integration can share them.

Most users want [`einfold-datafusion`](https://crates.io/crates/einfold-datafusion). See [einfold's README](https://github.com/xqlsystems/einfold#readme).

Apache-2.0.
