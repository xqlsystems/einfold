// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! einfold for Apache DataFusion (design doc §7.3, §7.5).
//!
//! The only einfold crate that depends on an engine. It will hold detection
//! (§9.1), the optimizer rule, and the reference executor `EinsumExec` (§10.2).

/// The DataFusion version this crate is built against. ddx must link the same.
pub const DATAFUSION_VERSION: &str = datafusion::DATAFUSION_VERSION;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pinned_datafusion_version() {
        assert_eq!(DATAFUSION_VERSION, "54.1.0");
    }
}
