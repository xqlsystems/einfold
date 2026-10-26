// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! What a caller tells the kernels about the host, and how to run a fold.

/// How the host compares floating-point keys, in joins and in `GROUP BY`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum FloatKeys {
    /// By their bits: `-0.0` and `0.0` are different keys, and a NaN matches
    /// a NaN with the same bits. Arrow's own comparisons of rows do this.
    #[default]
    Bitwise,
    /// By their bits, except that `-0.0` and `0.0` are one key, and a group of
    /// them is shown as `0.0`. DataFusion 55 does this for `Float16`, `Float32`
    /// and `Float64` columns, but not for dictionary-encoded ones.
    SignedZerosEqual,
}

/// What the kernels need to know about the host: the engine whose plan the
/// fold came from, which decides what SQL leaves open.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Host {
    /// How the host compares float keys.
    pub float_keys: FloatKeys,
}

/// Whether a fold's domain may have two rows that reach one group.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DomainRows {
    /// Any rows: each gets its group's value.
    #[default]
    Any,
    /// At most one row per group, or the dense algorithm declines. A plan that
    /// joins a fold back to its domain and groups again by the domain's keys
    /// is the fold from its domain exactly when they are unique. Only
    /// [`fold_dense`](crate::fold_dense) checks it, as it can decline.
    UniqueKeys,
}

/// How to run a fold. The default is one thread, a host that compares float
/// keys by their bits, and any domain rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Options {
    /// What the host decides.
    pub host: Host,
    /// The most threads to use. The result's bits don't depend on it.
    pub threads: usize,
    /// Whether the domain's rows must have unique keys.
    pub domain_rows: DomainRows,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            host: Host::default(),
            threads: 1,
            domain_rows: DomainRows::Any,
        }
    }
}
