// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`Relation`]: one of a fold's inputs, as Arrow batches.

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;

/// The rows of one of a fold's inputs, as batches of one schema, in the
/// order and batches they arrived in.
#[derive(Clone, Copy, Debug)]
pub struct Relation<'a> {
    /// The batches' schema.
    pub schema: &'a SchemaRef,
    /// The rows.
    pub batches: &'a [RecordBatch],
}

impl<'a> Relation<'a> {
    /// `batches`, all of `schema`.
    #[must_use]
    pub fn new(schema: &'a SchemaRef, batches: &'a [RecordBatch]) -> Self {
        Relation { schema, batches }
    }
}
