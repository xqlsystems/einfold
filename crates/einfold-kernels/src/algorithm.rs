// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`fold`] and [`fold_dense`], which run a fold by the best algorithm that
//! applies, and [`Algorithm`], which says which one ran.

use arrow::compute;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use einfold_ir::Fold;

use crate::options::{DomainRows, Options};
use crate::relation::Relation;
use crate::{bad, dense, discriminate, Result};

/// Which algorithm computed a fold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Algorithm {
    /// By positions, with no hashing: keys are small non-negative integers,
    /// used as array indices. Work is split into fixed blocks chosen from the
    /// shape alone, so the bits don't depend on the number of threads.
    Dense,
    /// Gustavson's row-by-row sparse product, generalized to folds, over
    /// keys ranked by discrimination, a radix sort of their bytes: no hashing
    /// and no comparisons of keys. Any keys, any density.
    Discriminate,
}

/// A fold's result, and the algorithm that computed it.
#[derive(Clone, Debug)]
pub struct Folded {
    /// Which algorithm ran.
    pub algorithm: Algorithm,
    /// The result's rows.
    pub batches: Vec<RecordBatch>,
}

/// [`fold`] by the dense algorithm alone, or `None` when its requirements
/// don't hold (as for a fold with [conditions](einfold_ir::Condition)), for a
/// caller that would rather run something else, such as the host's plan,
/// than the discrimination algorithm.
pub fn fold_dense(
    fold: &Fold,
    a: Relation,
    b: Relation,
    domain: Option<Relation>,
    schema: SchemaRef,
    options: Options,
) -> Result<Option<Vec<RecordBatch>>> {
    let threads = options.threads.max(1);
    let domain = domain.map(|d| d.batches);
    dense::fold(
        fold,
        a.batches,
        b.batches,
        domain,
        schema,
        threads,
        options.domain_rows,
    )
}

/// `fold` over operands `a` and `b`, and its domain if it has one, as
/// batches of `schema` (which [`schema`](crate::schema) gives): one batch per
/// domain batch, or one in all without a domain. The dense algorithm runs
/// when its requirements hold, and the discrimination one otherwise.
///
/// The result's bits depend only on the inputs' rows, not on how they were
/// batched or ordered, nor on the number of threads.
///
/// Fails if `options` asks for [`DomainRows::UniqueKeys`], which only
/// [`fold_dense`] can check, or if the fold has conditions, which neither
/// algorithm supports yet.
pub fn fold(
    fold: &Fold,
    a: Relation,
    b: Relation,
    domain: Option<Relation>,
    schema: SchemaRef,
    options: Options,
) -> Result<Folded> {
    if options.domain_rows == DomainRows::UniqueKeys {
        return bad("only fold_dense checks that a domain's keys are unique".into());
    }
    if !fold.conditions().is_empty() {
        return bad("the fold kernels don't support conditions".into());
    }
    let threads = options.threads.max(1);
    let domain = domain.map(|d| d.batches);
    let dense = dense::fold(
        fold,
        a.batches,
        b.batches,
        domain,
        schema.clone(),
        threads,
        DomainRows::Any,
    )?;
    if let Some(batches) = dense {
        return Ok(Folded {
            algorithm: Algorithm::Dense,
            batches,
        });
    }
    // The discrimination algorithm reads each operand as one batch.
    let whole = |r: Relation| compute::concat_batches(r.schema, r.batches);
    let (a, b) = (whole(a)?, whole(b)?);
    let batches = discriminate::fold(fold, &a, &b, domain, schema, options.host, threads)?;
    Ok(Folded {
        algorithm: Algorithm::Discriminate,
        batches,
    })
}
