// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`fold`] and [`fold_dense`], which run a fold by the best algorithm that
//! applies, [`Algorithm`], which says which one ran, and [`pipeline`], which
//! runs a fold pipeline.

use arrow::compute;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use einfold_ir::{Fold, Pipeline};

use crate::fused::Piped;
use crate::options::{DomainRows, Options};
use crate::relation::Relation;
use crate::{bad, dense, discriminate, fused, Result};

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
    /// A fold pipeline's fused algorithm (see [`pipeline`]): the producer
    /// computed one tile at a time and folded into the consumer, so its
    /// results never exist in full.
    Fused,
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
/// don't hold (as for a fold with [conditions](einfold_ir::Condition) on a
/// dimension it sums away), for a
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
/// [`fold_dense`] can check, or if the fold has
/// [conditions](einfold_ir::Condition) that the dense algorithm can't take:
/// the discrimination algorithm computes no masks.
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
    if !fold.conditions().is_empty() {
        return bad("the discrimination algorithm computes no masks".into());
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

/// `pipeline` over its producer's operands `q` and `k` and its values `v`, as
/// batches of `schema` (which [`pipeline_schema`](crate::pipeline_schema)
/// gives), by the fused algorithm: tile by tile, without the producer's
/// results ever existing in full. `None` when the algorithm's requirements
/// don't hold: every input complete, with one row and a non-NULL factor at
/// every position of its dimensions, keys small non-negative integers, and
/// at most one condition, between the producer's two outputs. The caller
/// then runs something else, such as the host's plan.
///
/// The result's bits depend only on the inputs' rows, not on how they were
/// batched or ordered, nor on the number of threads.
pub fn pipeline(
    pipeline: &Pipeline,
    [q, k, v]: [Relation; 3],
    schema: SchemaRef,
    options: Options,
) -> Result<Option<Piped>> {
    let threads = options.threads.max(1);
    fused::pipeline(pipeline, [q.batches, k.batches, v.batches], schema, threads)
}
