// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The allocator this library allocates with.

use mimalloc::MiMalloc;

/// mimalloc, Microsoft's general-purpose allocator, which datafusion-python
/// uses too, in its version 2, which runs beside PyArrow's own copy (see
/// `Cargo.toml`).
#[global_allocator]
static ALLOCATOR: MiMalloc = MiMalloc;
