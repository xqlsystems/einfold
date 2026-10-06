// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! einfold's engine-independent representations (design doc §7.5).
//!
//! Nothing in this crate depends on a query engine. It holds the EinsumIR
//! (design §7.1), facts (§8.1), and partial-aggregate states (§8.3).
