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

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::TableProvider;
use datafusion::common::{not_impl_err, plan_err, Result, TableReference};
use datafusion::execution::TaskContext;
use datafusion::logical_expr::{Extension, LogicalPlan};
use datafusion::physical_plan::ExecutionPlan;
use datafusion_proto::logical_plan::LogicalExtensionCodec;
use datafusion_proto::physical_plan::{PhysicalExtensionCodec, PhysicalProtoConverterExtension};

/// Marks a buffer [`ByReference`] wrote.
const MAGIC: &[u8] = b"einfold-ref:";

/// How many encoded plans are kept. A plan is decoded right after it is
/// encoded, when the planner hands it back to the session, so only the
/// latest few are ever needed.
const PLANS_KEPT: usize = 1024;

/// The tables encoded and not yet decoded, by id. A plan is encoded on the
/// session's side and decoded on einfold's within one planning call, so a
/// table is held from its encoding until its decoding, and no longer: a
/// table the session replaces is freed with it.
#[derive(Default)]
struct Tables {
    next: u64,
    pending: HashMap<u64, Arc<dyn TableProvider>>,
}

static TABLES: Mutex<Option<Tables>> = Mutex::new(None);

/// The plans encoded so far: the last [`PLANS_KEPT`], from id `first` on.
#[derive(Default)]
struct Plans {
    first: u64,
    kept: VecDeque<Arc<dyn ExecutionPlan>>,
}

static PLANS: Mutex<Option<Plans>> = Mutex::new(None);

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
    let tables = TABLES.lock().expect("no panic while holding the tables");
    tables.as_ref().map_or(0, |t| t.pending.len())
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
        let mut tables = TABLES.lock().expect("no panic while holding the tables");
        let table = tables.as_mut().and_then(|t| t.pending.remove(&id));
        match table {
            Some(table) => Ok(table),
            None => plan_err!("einfold's codec holds no table {id}"),
        }
    }

    fn try_encode_table_provider(
        &self,
        _table_ref: &TableReference,
        node: Arc<dyn TableProvider>,
        buf: &mut Vec<u8>,
    ) -> Result<()> {
        let mut tables = TABLES.lock().expect("no panic while holding the tables");
        let tables = tables.get_or_insert_with(Tables::default);
        let id = tables.next;
        tables.next += 1;
        tables.pending.insert(id, node);
        write(buf, id);
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
        let plan = {
            let plans = PLANS.lock().expect("no panic while holding the plans");
            let plan = plans.as_ref().and_then(|p| {
                let at = id.checked_sub(p.first)?;
                p.kept.get(at as usize).cloned()
            });
            match plan {
                Some(plan) => plan,
                None => return plan_err!("einfold's codec no longer holds plan {id}"),
            }
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
        let mut plans = PLANS.lock().expect("no panic while holding the plans");
        let plans = plans.get_or_insert_with(Plans::default);
        let id = plans.first + plans.kept.len() as u64;
        plans.kept.push_back(node);
        if plans.kept.len() > PLANS_KEPT {
            plans.kept.pop_front();
            plans.first += 1;
        }
        write(buf, id);
        Ok(())
    }
}
