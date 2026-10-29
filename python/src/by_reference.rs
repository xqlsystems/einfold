// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`ByReference`]: codecs that pass tables and plans by reference within
//! this process.
//!
//! datafusion-python hands its logical plans to a query planner as protobuf,
//! and takes the physical plan back the same way. Protobuf can't encode an
//! in-memory table without a codec, and encodes an in-memory scan with its
//! data, which would copy every table a query reads. The planner runs in the
//! session's own process, so a reference is enough: a table or plan is
//! encoded as an index into a registry here, and decoded back to the same
//! object.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::TableProvider;
use datafusion::common::{not_impl_err, plan_err, Result, TableReference};
use datafusion::execution::TaskContext;
use datafusion::logical_expr::{Extension, LogicalPlan};
use datafusion::physical_plan::ExecutionPlan;
use datafusion_proto::logical_plan::LogicalExtensionCodec;
use datafusion_proto::physical_plan::{PhysicalExtensionCodec, PhysicalProtoConverterExtension};

use crate::lean::Lean;

/// Marks a buffer [`ByReference`] wrote.
const MAGIC: &[u8] = b"einfold-ref:";

/// What was encoded and not yet decoded, by id. A plan is encoded on one
/// side of the FFI and decoded on the other within one planning call, so
/// each table or plan is held from its encoding until its decoding, and no
/// longer: a table the session replaces is freed with it, and so is a plan
/// once the session drops it, with what its operators keep, such as a hash
/// join's table.
struct Pending<T: ?Sized> {
    next: u64,
    held: HashMap<u64, Arc<T>>,
}

impl<T: ?Sized> Pending<T> {
    fn new() -> Self {
        Pending {
            next: 0,
            held: HashMap::new(),
        }
    }
}

static TABLES: Mutex<Option<Pending<dyn TableProvider>>> = Mutex::new(None);
static PLANS: Mutex<Option<Pending<dyn ExecutionPlan>>> = Mutex::new(None);

/// Holds `item` until it is taken, and returns its id.
fn hold<T: ?Sized>(pending: &Mutex<Option<Pending<T>>>, item: Arc<T>) -> u64 {
    let mut pending = pending
        .lock()
        .expect("no panic while holding encoded objects");
    let pending = pending.get_or_insert_with(Pending::new);
    let id = pending.next;
    pending.next += 1;
    pending.held.insert(id, item);
    id
}

/// The item held as `id`, no longer held.
fn take<T: ?Sized>(pending: &Mutex<Option<Pending<T>>>, id: u64) -> Option<Arc<T>> {
    let mut pending = pending
        .lock()
        .expect("no panic while holding encoded objects");
    pending.as_mut().and_then(|p| p.held.remove(&id))
}

/// How many items are held.
fn held<T: ?Sized>(pending: &Mutex<Option<Pending<T>>>) -> usize {
    let pending = pending
        .lock()
        .expect("no panic while holding encoded objects");
    pending.as_ref().map_or(0, |p| p.held.len())
}

fn write(buf: &mut Vec<u8>, id: u64) {
    buf.extend_from_slice(MAGIC);
    buf.extend_from_slice(&id.to_le_bytes());
}

fn read(buf: &[u8]) -> Option<u64> {
    let id = buf.strip_prefix(MAGIC)?.try_into().ok()?;
    Some(u64::from_le_bytes(id))
}

/// How many encoded tables are waiting to be decoded.
pub(crate) fn tables_pending() -> usize {
    held(&TABLES)
}

/// How many encoded plans are waiting to be decoded.
pub(crate) fn plans_pending() -> usize {
    held(&PLANS)
}

/// Codecs that pass tables and plans by reference within this process.
#[derive(Debug)]
pub(crate) struct ByReference;

impl LogicalExtensionCodec for ByReference {
    fn try_decode(&self, _: &[u8], _: &[LogicalPlan], _: &TaskContext) -> Result<Extension> {
        not_impl_err!("einfold's codec encodes only tables")
    }

    fn try_encode(&self, _: &Extension, _: &mut Vec<u8>) -> Result<()> {
        not_impl_err!("einfold's codec encodes only tables")
    }

    fn try_decode_table_provider(
        &self,
        buf: &[u8],
        _table_ref: &TableReference,
        _schema: SchemaRef,
        _ctx: &TaskContext,
    ) -> Result<Arc<dyn TableProvider>> {
        let Some(id) = read(buf) else {
            return plan_err!("not a table einfold's codec encoded");
        };
        match take(&TABLES, id) {
            Some(table) => Ok(Arc::new(Lean::new(table))),
            None => plan_err!("einfold's codec holds no table {id}"),
        }
    }

    fn try_encode_table_provider(
        &self,
        _table_ref: &TableReference,
        node: Arc<dyn TableProvider>,
        buf: &mut Vec<u8>,
    ) -> Result<()> {
        write(buf, hold(&TABLES, node));
        Ok(())
    }
}

impl PhysicalExtensionCodec for ByReference {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        _ctx: &TaskContext,
        _proto_converter: &dyn PhysicalProtoConverterExtension,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let Some(id) = read(buf) else {
            return plan_err!("not a plan einfold's codec encoded");
        };
        let Some(plan) = take(&PLANS, id) else {
            return plan_err!("einfold's codec holds no plan {id}");
        };
        match inputs.is_empty() {
            true => Ok(plan),
            // A node whose children the decoder rebuilt.
            #[allow(deprecated)]
            false => plan.with_new_children(inputs.to_vec()),
        }
    }

    fn try_encode(
        &self,
        node: Arc<dyn ExecutionPlan>,
        buf: &mut Vec<u8>,
        _proto_converter: &dyn PhysicalProtoConverterExtension,
    ) -> Result<()> {
        write(buf, hold(&PLANS, node));
        Ok(())
    }
}
