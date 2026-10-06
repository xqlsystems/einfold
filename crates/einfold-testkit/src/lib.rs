// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The equivalence harness of design §11: random einsums over random tables,
//! checked against SQL itself.
//!
//! - [`Case::generate`] makes a seeded [`Case`]: an [`Einsum`](einfold_ir::Einsum),
//!   its tables, and the SQL that Blacher's four rules give it (design §2.2).
//! - [`sql_reference`] runs that SQL in DataFusion; [`naive_reference`] computes
//!   the same result by nested loops, using
//!   [`PartialSum`](einfold_ir::PartialSum) (design §8.3).
//! - [`assert_same_result`] compares two results as SQL would: rows as a
//!   multiset, keys exactly (NULL equals NULL), values to a relative `1e-9`
//!   (NULL equals NULL, NaN equals NaN).
//! - [`check`] runs an implementation against the SQL reference on many cases.
//!
//! Every table has the columns of its operand's dimensions, in order, named by
//! the dimension, followed by a `Float64` column `v`. A result has the output
//! dimensions, then `v`. A dimension appears at most once per operand here.

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
/// On the first mismatch, with the case's seed, einsum, SQL, tables, and a
/// diff of the two results (design §11).
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
