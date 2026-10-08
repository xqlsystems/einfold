// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `GradFold`: a two-operand contraction whose output rows are those of a
//! third table, as a DataFusion logical node, physical operator and planner.
//!
//! It computes what ddx's gradient steps compute for a matrix product, in
//! SQL terms:
//!
//! ```sql
//! SELECT f.x, f.y, CASE WHEN f.val IS NULL THEN NULL
//!                       WHEN r.v IS NULL THEN 0.0 ELSE r.v END AS val
//! FROM f LEFT JOIN (SELECT l.x, r.y, SUM(l.val * r.val) AS v
//!                   FROM l JOIN r ON l.z = r.z GROUP BY l.x, r.y) r
//!   ON f.x = r.x AND f.y = r.y
//! ```
//!
//! `l` is streamed and `r` is held. A product with a NULL factor is skipped,
//! as `SUM` skips NULL; a group no non-NULL product reaches is 0; a row of `f`
//! whose value is NULL gets NULL. Duplicate rows count every time (bag
//! semantics), in `l`, `r` and `f` alike.
//!
//! When every key is a small non-negative integer and `r` has one row per
//! `(z, y)`, keys are used as positions in dense arrays: no hashing. Otherwise
//! a hash table on `r` is used.

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::arrow::array::{Array, ArrayRef, Float64Array, Int64Array, RecordBatch};
use datafusion::arrow::compute::concat_batches;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::common::{DFSchemaRef, DataFusionError, Result};
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::{SendableRecordBatchStream, SessionState, TaskContext};
use datafusion::logical_expr::{
    Expr, LogicalPlan, UserDefinedLogicalNode, UserDefinedLogicalNodeCore,
};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, Partitioning, PlanProperties,
};
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use futures::TryStreamExt;

/// Which column of each input holds which dimension. `l` holds `(x, z,
/// val)`, `r` holds `(z, y, val)` and `f` holds `(x, y, val)`; each field is
/// a column index, and every input's value column is given too.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd)]
pub struct Columns {
    pub l: (usize, usize, usize),
    pub r: (usize, usize, usize),
    pub f: (usize, usize, usize),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GradFold {
    pub cols: Columns,
    /// `[l, r, f]`.
    pub inputs: Vec<LogicalPlan>,
    /// `f`'s key columns and a nullable `Float64` value.
    pub schema: DFSchemaRef,
}

// The schema follows from the inputs and columns, so ordering ignores it.
impl PartialOrd for GradFold {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        (self.cols, &self.inputs).partial_cmp(&(other.cols, &other.inputs))
    }
}

impl UserDefinedLogicalNodeCore for GradFold {
    fn name(&self) -> &str {
        "GradFold"
    }
    fn inputs(&self) -> Vec<&LogicalPlan> {
        self.inputs.iter().collect()
    }
    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }
    fn expressions(&self) -> Vec<Expr> {
        vec![]
    }
    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GradFold: {:?}", self.cols)
    }
    fn with_exprs_and_inputs(&self, _exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> Result<Self> {
        Ok(GradFold {
            cols: self.cols,
            inputs,
            schema: self.schema.clone(),
        })
    }
}

#[derive(Debug)]
pub struct GradFoldExec {
    cols: Columns,
    inputs: Vec<Arc<dyn ExecutionPlan>>,
    cache: Arc<PlanProperties>,
}

impl GradFoldExec {
    fn new(cols: Columns, inputs: Vec<Arc<dyn ExecutionPlan>>, schema: SchemaRef) -> Self {
        let cache = PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        GradFoldExec {
            cols,
            inputs,
            cache: Arc::new(cache),
        }
    }
}

/// One input, every partition, as one batch.
async fn collect(plan: &Arc<dyn ExecutionPlan>, ctx: &Arc<TaskContext>) -> Result<RecordBatch> {
    let mut batches = Vec::new();
    for p in 0..plan.properties().output_partitioning().partition_count() {
        batches.extend(
            plan.execute(p, Arc::clone(ctx))?
                .try_collect::<Vec<_>>()
                .await?,
        );
    }
    Ok(concat_batches(&plan.schema(), &batches)?)
}

fn ints(b: &RecordBatch, i: usize) -> Result<&Int64Array> {
    b.column(i)
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| DataFusionError::Plan("GradFold keys must be Int64".into()))
}

fn floats(b: &RecordBatch, i: usize) -> Result<&Float64Array> {
    b.column(i)
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| DataFusionError::Plan("GradFold values must be Float64".into()))
}

/// The largest key plus one, if every key is non-negative.
fn extent(keys: &Int64Array) -> Option<usize> {
    let mut max = -1i64;
    for i in 0..keys.len() {
        let k = keys.value(i);
        if k < 0 {
            return None;
        }
        max = max.max(k);
    }
    Some((max + 1) as usize)
}

/// How many computations took the dense path and the hash path.
pub static DENSE: AtomicUsize = AtomicUsize::new(0);
pub static HASHED: AtomicUsize = AtomicUsize::new(0);

/// Grouped sums of `l.val · r.val` over `l.z = r.z`, looked up for each row
/// of `f`.
fn compute(
    cols: Columns,
    l: &RecordBatch,
    r: &RecordBatch,
    f: &RecordBatch,
) -> Result<Float64Array> {
    let (lx, lz, lv) = (ints(l, cols.l.0)?, ints(l, cols.l.1)?, floats(l, cols.l.2)?);
    let (rz, ry, rv) = (ints(r, cols.r.0)?, ints(r, cols.r.1)?, floats(r, cols.r.2)?);
    let (fx, fy, fv) = (ints(f, cols.f.0)?, ints(f, cols.f.1)?, floats(f, cols.f.2)?);
    let lookup: Box<dyn Fn(i64, i64) -> f64> =
        match (extent(lx), extent(lz).max(extent(rz)), extent(ry)) {
            (Some(nx), Some(nz), Some(ny))
                if nx.saturating_mul(ny) <= 1 << 26 && nz.saturating_mul(ny) <= 1 << 26 =>
            {
                // Dense: r as a matrix [z][y], unless a (z, y) repeats.
                let mut rm = vec![0.0; nz * ny];
                let mut seen = vec![false; nz * ny];
                let mut unique = true;
                for i in 0..r.num_rows() {
                    let p = rz.value(i) as usize * ny + ry.value(i) as usize;
                    unique &= !std::mem::replace(&mut seen[p], true);
                    // A NULL factor contributes nothing, as SUM skips NULL.
                    rm[p] = if rv.is_null(i) { 0.0 } else { rv.value(i) };
                }
                if !unique {
                    return hashed(cols, l, r, f);
                }
                let mut out = vec![0.0; nx * ny];
                for i in 0..l.num_rows() {
                    if lv.is_null(i) {
                        continue;
                    }
                    let (x, z, v) = (lx.value(i) as usize, lz.value(i) as usize, lv.value(i));
                    for (o, b) in out[x * ny..(x + 1) * ny]
                        .iter_mut()
                        .zip(&rm[z * ny..(z + 1) * ny])
                    {
                        *o += v * b;
                    }
                }
                // A 0.0 standing for a NULL or a missing row of r must not meet a
                // NaN or an infinity in l: 0 · inf is NaN, but SQL forms no such
                // product (no join row) or skips it (a NULL factor).
                let holes = rv.null_count() > 0 || seen.iter().any(|s| !s);
                if holes && (0..l.num_rows()).any(|i| !lv.is_null(i) && !lv.value(i).is_finite()) {
                    return hashed(cols, l, r, f);
                }
                DENSE.fetch_add(1, Ordering::Relaxed);
                Box::new(move |x, y| {
                    let (x, y) = (x as usize, y as usize);
                    if x < nx && y < ny {
                        out[x * ny + y]
                    } else {
                        0.0
                    }
                })
            }
            _ => return hashed(cols, l, r, f),
        };
    Ok((0..f.num_rows())
        .map(|i| {
            if fv.is_null(i) {
                None
            } else {
                Some(lookup(fx.value(i), fy.value(i)))
            }
        })
        .collect())
}

/// The general path: a hash table on `r`, keyed by `z`.
fn hashed(
    cols: Columns,
    l: &RecordBatch,
    r: &RecordBatch,
    f: &RecordBatch,
) -> Result<Float64Array> {
    HASHED.fetch_add(1, Ordering::Relaxed);
    let (lx, lz, lv) = (ints(l, cols.l.0)?, ints(l, cols.l.1)?, floats(l, cols.l.2)?);
    let (rz, ry, rv) = (ints(r, cols.r.0)?, ints(r, cols.r.1)?, floats(r, cols.r.2)?);
    let (fx, fy, fv) = (ints(f, cols.f.0)?, ints(f, cols.f.1)?, floats(f, cols.f.2)?);
    let mut table: HashMap<i64, Vec<(i64, f64)>> = HashMap::new();
    for i in 0..r.num_rows() {
        if !rv.is_null(i) {
            table
                .entry(rz.value(i))
                .or_default()
                .push((ry.value(i), rv.value(i)));
        }
    }
    let mut sums: HashMap<(i64, i64), f64> = HashMap::new();
    for i in 0..l.num_rows() {
        if lv.is_null(i) {
            continue;
        }
        if let Some(rows) = table.get(&lz.value(i)) {
            for &(y, b) in rows {
                *sums.entry((lx.value(i), y)).or_insert(0.0) += lv.value(i) * b;
            }
        }
    }
    Ok((0..f.num_rows())
        .map(|i| {
            if fv.is_null(i) {
                None
            } else {
                Some(*sums.get(&(fx.value(i), fy.value(i))).unwrap_or(&0.0))
            }
        })
        .collect())
}

impl DisplayAs for GradFoldExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "GradFoldExec: {:?}", self.cols)
    }
}

impl ExecutionPlan for GradFoldExec {
    fn name(&self) -> &'static str {
        "GradFoldExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.inputs.iter().collect()
    }
    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::UnspecifiedDistribution; 3]
    }
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false; 3]
    }
    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(GradFoldExec::new(
            self.cols,
            children,
            self.schema(),
        )))
    }
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "GradFoldExec has one partition, asked for {partition}"
            )));
        }
        let (cols, inputs, schema) = (self.cols, self.inputs.clone(), self.schema());
        let out_schema = Arc::clone(&schema);
        let fut = async move {
            let l = collect(&inputs[0], &context).await?;
            let r = collect(&inputs[1], &context).await?;
            let f = collect(&inputs[2], &context).await?;
            let v = compute(cols, &l, &r, &f)?;
            let columns: Vec<ArrayRef> = vec![
                f.column(cols.f.0).clone(),
                f.column(cols.f.1).clone(),
                Arc::new(v),
            ];
            Ok(RecordBatch::try_new(out_schema, columns)?)
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::once(fut),
        )))
    }
}

#[derive(Debug, Default)]
pub struct GradFoldPlanner;

#[async_trait]
impl ExtensionPlanner for GradFoldPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        _session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(node) = node.as_any().downcast_ref::<GradFold>() else {
            return Ok(None);
        };
        let schema = Arc::new(node.schema.as_arrow().clone());
        Ok(Some(Arc::new(GradFoldExec::new(
            node.cols,
            physical_inputs.to_vec(),
            schema,
        ))))
    }
}

#[derive(Debug, Default)]
pub struct GradFoldQueryPlanner;

#[async_trait]
impl QueryPlanner for GradFoldQueryPlanner {
    async fn create_physical_plan(
        &self,
        plan: &LogicalPlan,
        state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(GradFoldPlanner)])
            .create_physical_plan(plan, state)
            .await
    }
}
