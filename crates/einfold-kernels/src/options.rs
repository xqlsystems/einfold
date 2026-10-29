// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! What a caller tells the kernels about the host, and how to run a fold.

use std::fmt;

use arrow::error::ArrowError;

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

/// The memory a kernel may take beyond its inputs, such as a query engine's
/// memory pool. A kernel asks before it allocates its working arrays and its
/// output, and stops with [`ArrowError::MemoryError`] where the budget says
/// no, before allocating, so that its caller can run something that needs
/// less, such as a plan that spills to disk.
///
/// What a kernel asks for is an estimate of what it allocates, within a
/// small factor; it is not given back as the kernel frees it.
///
/// [`ArrowError::MemoryError`]: arrow::error::ArrowError::MemoryError
pub trait Budget: Sync + fmt::Debug {
    /// Whether `bytes` more may be allocated; if so, they are counted.
    fn grow(&self, bytes: usize) -> bool;
}

/// How to run a fold. The default is one thread, a host that compares float
/// keys by their bits, any domain rows, and no limit on memory.
#[derive(Clone, Copy, Debug)]
pub struct Options<'a> {
    /// What the host decides.
    pub host: Host,
    /// The most threads to use. The result's bits don't depend on it.
    pub threads: usize,
    /// Whether the domain's rows must have unique keys.
    pub domain_rows: DomainRows,
    /// The memory the kernel may take beyond its inputs, or `None` for no
    /// limit.
    pub budget: Option<&'a dyn Budget>,
}

impl Options<'_> {
    /// Asks the budget for `bytes` to hold `what`, failing if it says no.
    pub(crate) fn reserve(&self, bytes: usize, what: &str) -> crate::Result<()> {
        match self.budget {
            Some(budget) if !budget.grow(bytes) => Err(ArrowError::MemoryError(format!(
                "EinFold's budget can't give {bytes} bytes for {what}"
            ))),
            _ => Ok(()),
        }
    }
}

impl Default for Options<'_> {
    fn default() -> Self {
        Options {
            host: Host::default(),
            threads: 1,
            domain_rows: DomainRows::Any,
            budget: None,
        }
    }
}
