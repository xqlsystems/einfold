// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! An equivalence harness: random folds over random tables, checked against
//! SQL itself.
//!
//! A *fold* ([`einfold_ir::Fold`]) is a SQL query of the shape `SELECT
//! <output>, AGG(a.v * b.v * ...) FROM a JOIN b ON ... GROUP BY <output>`,
//! where `AGG` is `SUM`, `COUNT`, or `AVG`. An einsum is the `SUM` case. This
//! crate checks an implementation of such queries against two independent
//! oracles, so that a rewrite is only ever allowed to change how the answer is
//! computed, never the answer.
//!
//! - [`Case::generate`] makes a seeded [`Case`]: a fold, its tables, and its
//!   SQL.
//! - [`sql_reference`] runs that SQL in DataFusion; [`naive_reference`]
//!   computes the same result by nested loops, with
//!   [`PartialAggregate`](einfold_ir::PartialAggregate)s.
//! - [`assert_same_result`] compares two results as SQL would.
//! - [`check`] runs an implementation against the SQL reference on many cases.
//!
//! # Invariants checked
//!
//! 1. **Same groups.** A group appears in the result exactly when some joined
//!    row reached it, even if every value in it is NULL. Keys compare exactly,
//!    with NULL equal to NULL and floats by their bits (`-0.0` differs from
//!    `0.0`, NaN equals NaN), as DataFusion's `GROUP BY` does.
//! 2. **Same values.** `SUM` and `AVG` are NULL if every value in the group is
//!    NULL, and NaN propagates. `COUNT` is an exact `Int64`, 0 for an all-NULL
//!    group. Floats match to a relative `1e-9`, since a rewrite may add in a
//!    different order.
//! 3. **Same rows, in any order.** Rows compare as a multiset, so duplicate
//!    result rows must be duplicated.
//! 4. **The oracles agree.** `naive_reference` equals `sql_reference` on every
//!    generated case (`tests/selftest.rs`), which validates both.
//!
//! The generated inputs cover NULL and duplicate keys, NULL, NaN, and all-NULL
//! values, empty tables, and `Int64`, `Utf8`, and `Float64` keys. A table has
//! one column per dimension of its operand, named by the dimension, then a
//! `Float64` column `v`. A result has the output dimensions, then `v`. A
//! dimension appears at most once per operand here.

mod compare;
mod generate;
mod oracle;

pub use compare::{assert_same_result, compare};
pub use generate::Case;
pub use oracle::{naive_reference, sql_reference};

use datafusion::arrow::record_batch::RecordBatch;

/// The seed of case `i` in a run seeded `seed`. Pass it to [`Case::generate`]
/// to rebuild one failing case.
pub fn case_seed(seed: u64, i: usize) -> u64 {
    seed.wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
}

/// Generate `n_cases` cases from `seed`, run `implementation` on each, and
/// compare it with [`sql_reference`].
///
/// # Panics
///
/// On the first mismatch, with the case's seed, fold, SQL, tables, and a
/// diff of the two results.
pub fn check(n_cases: usize, seed: u64, implementation: impl Fn(&Case) -> RecordBatch) {
    for i in 0..n_cases {
        let cs = case_seed(seed, i);
        let case = Case::generate(cs);
        let expected = sql_reference(&case);
        let actual = implementation(&case);
        if let Err(diff) = compare(&expected, &actual) {
            panic!("case {i} of run seed {seed} failed (case seed {cs})\n{case}\n{diff}");
        }
    }
}
