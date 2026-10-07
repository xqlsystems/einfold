// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! `EinFoldExec`, einfold's reference physical operator in DataFusion.

use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, AsArray};
use datafusion::arrow::compute::concat_batches;
use datafusion::arrow::datatypes::{DataType, Float64Type, SchemaRef};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, Distribution, ExecutionPlan, Partitioning, PlanProperties,
};
use einfold_ir::Fold;
use futures::TryStreamExt;

use crate::kernel::{fold_join, OperandArrays};

/// A physical operator that runs a two-operand [`Fold`] with the EinFold hash
/// kernel: a join followed by `SUM`, `COUNT` or `AVG` and a `GROUP BY`,
/// computed without materializing the join's rows.
///
/// Each child produces one operand: its dimension columns in the order of the
/// operand's `dims`, then its `Float64` value column. The output has one
/// partition and the schema given at construction, which must match what
/// [`fold_join`] returns: one column per output dimension, then the
/// aggregate's value.
///
/// The output must be repeatable: float addition is not associative, so the
/// same rows added in a different order can differ in the last bits. So each
/// child's partitions are read one after another in partition-index order,
/// never interleaved by arrival, and for deterministic children the output is
/// identical on every run.
#[derive(Debug)]
pub struct EinFoldExec {
    fold: Fold,
    inputs: [Arc<dyn ExecutionPlan>; 2],
    cache: Arc<PlanProperties>,
}

impl EinFoldExec {
    /// An operator for `fold` over `a` and `b`, producing `schema`.
    ///
    /// Errors if `fold` does not have exactly two operands, or a child does
    /// not have one column per operand dimension plus a value column.
    pub fn try_new(
        fold: Fold,
        a: Arc<dyn ExecutionPlan>,
        b: Arc<dyn ExecutionPlan>,
        schema: SchemaRef,
    ) -> Result<Self> {
        if fold.operands().len() != 2 {
            return Err(DataFusionError::Plan(format!(
                "EinFoldExec folds exactly two operands, got {}",
                fold.operands().len()
            )));
        }
        for (op, child) in fold.operands().iter().zip([&a, &b]) {
            if child.schema().fields().len() != op.dims.len() + 1 {
                return Err(DataFusionError::Plan(format!(
                    "the input for `{}` needs {} dimension columns and a value column",
                    op.name,
                    op.dims.len()
                )));
            }
        }
        if schema.fields().len() != fold.output().len() + 1 {
            return Err(DataFusionError::Plan(
                "the output schema needs one column per output dimension and the aggregate's value"
                    .into(),
            ));
        }
        let cache = PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            Boundedness::Bounded,
        );
        Ok(EinFoldExec {
            fold,
            inputs: [a, b],
            cache: Arc::new(cache),
        })
    }

    /// The fold this operator computes.
    pub fn fold(&self) -> &Fold {
        &self.fold
    }
}

/// Read every partition of `plan`, in partition-index order, into one batch.
async fn collect_operand(
    plan: &Arc<dyn ExecutionPlan>,
    ctx: &Arc<TaskContext>,
) -> Result<OperandArrays> {
    let mut batches = Vec::new();
    for p in 0..plan.properties().output_partitioning().partition_count() {
        let stream = plan.execute(p, Arc::clone(ctx))?;
        batches.extend(stream.try_collect::<Vec<_>>().await?);
    }
    let schema = plan.schema();
    let all = concat_batches(&schema, &batches)?;
    let (value, dims) = all.columns().split_last().expect("checked in try_new");
    if value.data_type() != &DataType::Float64 {
        return Err(DataFusionError::Plan(format!(
            "an operand's value column must be Float64, got {}",
            value.data_type()
        )));
    }
    Ok(OperandArrays {
        dims: dims.to_vec(),
        value: value.as_primitive::<Float64Type>().clone(),
    })
}

impl DisplayAs for EinFoldExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "EinFoldExec: {}", self.fold)
    }
}

impl ExecutionPlan for EinFoldExec {
    fn name(&self) -> &'static str {
        "EinFoldExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        self.inputs.iter().collect()
    }

    /// `UnspecifiedDistribution`, so DataFusion adds no `CoalescePartitionsExec`
    /// (which would interleave a child's partitions by arrival).
    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::UnspecifiedDistribution; 2]
    }

    /// Never ask for round-robin repartitioning below this operator: it would
    /// make arrival order timing-dependent, and with it the float sums' bits.
    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false, false]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let [a, b]: [Arc<dyn ExecutionPlan>; 2] = children
            .try_into()
            .map_err(|_| DataFusionError::Plan("EinFoldExec has exactly two children".into()))?;
        Ok(Arc::new(EinFoldExec::try_new(
            self.fold.clone(),
            a,
            b,
            self.cache.eq_properties.schema().clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(format!(
                "EinFoldExec has one partition, asked for {partition}"
            )));
        }
        let (fold, inputs) = (self.fold.clone(), self.inputs.clone());
        let schema = self.schema();
        let out_schema = Arc::clone(&schema);
        // Dropping the returned stream drops this future, which cancels the
        // reads of the children at their next await point.
        let fut = async move {
            // For now both sides are collected in memory. A streaming version
            // would hash `b` and probe with `a`'s batches as they arrive.
            let a = collect_operand(&inputs[0], &context).await?;
            let b = collect_operand(&inputs[1], &context).await?;
            let out = fold_join(&fold, &a, &b)?;
            let columns: Vec<ArrayRef> = out.columns().to_vec();
            // Re-wrap under the supplied schema; this checks types and nullability.
            Ok(RecordBatch::try_new(out_schema, columns)?)
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::once(fut),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::aggregate_field;
    use datafusion::arrow::array::{Float64Array, Int64Array};
    use datafusion::arrow::datatypes::{Field, Schema};
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::physical_plan::{collect, displayable};
    use datafusion::prelude::SessionContext;
    use einfold_ir::{Aggregate, Dim, KeyEquality, Op, Operand, RowValue};

    fn fold_with(aggregate: Aggregate) -> Fold {
        let d = |s: &str| Dim::new(s);
        Fold::new(
            vec![
                Operand::new("a", [d("i"), d("k")]),
                Operand::new("b", [d("k"), d("j")]),
            ],
            vec![d("i"), d("j")],
            ["i", "j", "k"]
                .iter()
                .map(|s| (d(s), KeyEquality::Equal))
                .collect(),
            RowValue::Product(Op::Mul),
            aggregate,
        )
        .unwrap()
    }

    fn in_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int64, true),
            Field::new("y", DataType::Int64, true),
            Field::new("v", DataType::Float64, true),
        ]))
    }

    fn out_schema() -> SchemaRef {
        schema_for(Aggregate::SUM)
    }

    fn schema_for(aggregate: Aggregate) -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int64, true),
            Field::new("j", DataType::Int64, true),
            aggregate_field(aggregate),
        ]))
    }

    fn batch(rows: &[(i64, i64, f64)]) -> RecordBatch {
        RecordBatch::try_new(
            in_schema(),
            vec![
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.0))),
                Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.1))),
                Arc::new(Float64Array::from_iter_values(rows.iter().map(|r| r.2))),
            ],
        )
        .unwrap()
    }

    /// Rows with sums that depend on the order of addition.
    fn rows(seed: i64, n: i64) -> Vec<(i64, i64, f64)> {
        (0..n)
            .map(|x| {
                (
                    (x * seed) % 7,
                    (x + seed) % 5,
                    1e8 / (x + seed) as f64 + 0.1,
                )
            })
            .collect()
    }

    /// `rows` cut into partitions of several batches each.
    fn partitions(rows: &[(i64, i64, f64)], parts: usize) -> Vec<Vec<RecordBatch>> {
        let per = rows.len().div_ceil(parts);
        rows.chunks(per)
            .map(|p| p.chunks(7).map(batch).collect())
            .collect()
    }

    fn exec(a: &[Vec<RecordBatch>], b: &[Vec<RecordBatch>]) -> Arc<EinFoldExec> {
        exec_with(Aggregate::SUM, a, b)
    }

    fn exec_with(
        aggregate: Aggregate,
        a: &[Vec<RecordBatch>],
        b: &[Vec<RecordBatch>],
    ) -> Arc<EinFoldExec> {
        let src = |p: &[Vec<RecordBatch>]| -> Arc<dyn ExecutionPlan> {
            MemorySourceConfig::try_new_exec(p, in_schema(), None).unwrap()
        };
        let schema = schema_for(aggregate);
        Arc::new(EinFoldExec::try_new(fold_with(aggregate), src(a), src(b), schema).unwrap())
    }

    #[tokio::test]
    async fn every_aggregate_matches_kernel() {
        let (pa, pb) = (partitions(&rows(3, 60), 4), partitions(&rows(5, 45), 3));
        for agg in [Aggregate::SUM, Aggregate::COUNT, Aggregate::AVG] {
            let got = run(exec_with(agg, &pa, &pb)).await;
            let want = fold_join(&fold_with(agg), &operand(&pa), &operand(&pb)).unwrap();
            assert_eq!(got.columns(), want.columns(), "{agg}");
            assert_eq!(got.schema(), schema_for(agg), "{agg}");
        }
    }

    async fn run(plan: Arc<EinFoldExec>) -> RecordBatch {
        let ctx = SessionContext::new().task_ctx();
        let out = collect(plan, ctx).await.unwrap();
        assert_eq!(out.len(), 1);
        out.into_iter().next().unwrap()
    }

    fn operand(parts: &[Vec<RecordBatch>]) -> OperandArrays {
        let all: Vec<RecordBatch> = parts.iter().flatten().cloned().collect();
        let all = concat_batches(&in_schema(), &all).unwrap();
        OperandArrays {
            dims: all.columns()[..2].to_vec(),
            value: all.column(2).as_primitive::<Float64Type>().clone(),
        }
    }

    #[tokio::test]
    async fn matches_kernel_on_concatenated_input() {
        let (ra, rb) = (rows(3, 60), rows(5, 45));
        let (pa, pb) = (partitions(&ra, 4), partitions(&rb, 3));
        let got = run(exec(&pa, &pb)).await;
        let want = fold_join(&fold_with(Aggregate::SUM), &operand(&pa), &operand(&pb)).unwrap();
        assert!(got.num_rows() > 0);
        assert_eq!(got.columns(), want.columns());
        assert_eq!(got.schema(), out_schema());
    }

    #[tokio::test]
    async fn deterministic_across_runs() {
        let (pa, pb) = (partitions(&rows(3, 200), 5), partitions(&rows(5, 150), 4));
        let first = run(exec(&pa, &pb)).await;
        for _ in 0..20 {
            assert_eq!(run(exec(&pa, &pb)).await, first);
        }
    }

    #[tokio::test]
    async fn empty_inputs() {
        let none: Vec<Vec<RecordBatch>> = vec![vec![]];
        let pb = partitions(&rows(5, 10), 2);
        assert_eq!(run(exec(&none, &pb)).await.num_rows(), 0);
        assert_eq!(run(exec(&none, &none)).await.num_rows(), 0);
    }

    #[test]
    fn plan_shape() {
        let p = partitions(&rows(3, 10), 2);
        let plan = exec(&p, &p);
        assert_eq!(plan.children().len(), 2);
        let dist = plan.required_input_distribution();
        assert_eq!(dist.len(), 2);
        assert_eq!(plan.benefits_from_input_partitioning(), vec![false, false]);
        assert!(dist
            .iter()
            .all(|d| matches!(d, Distribution::UnspecifiedDistribution)));
        assert_eq!(plan.properties().output_partitioning().partition_count(), 1);
        let shown = displayable(plan.as_ref()).one_line().to_string();
        assert!(
            shown.contains("EinFoldExec: SUM(a[i,k] · b[k,j]) -> [i,j]"),
            "{shown}"
        );
        let kids: Vec<_> = plan.children().into_iter().cloned().collect();
        let again = plan.clone().with_new_children(kids).unwrap();
        assert_eq!(again.schema(), out_schema());
        assert!(plan.with_new_children(vec![]).is_err());
    }

    #[tokio::test]
    async fn only_partition_zero_and_bad_schemas() {
        let p = partitions(&rows(3, 10), 2);
        let plan = exec(&p, &p);
        assert!(plan.execute(1, SessionContext::new().task_ctx()).is_err());
        let child: Arc<dyn ExecutionPlan> =
            MemorySourceConfig::try_new_exec(&p, in_schema(), None).unwrap();
        let narrow = Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        assert!(EinFoldExec::try_new(
            fold_with(Aggregate::SUM),
            child.clone(),
            child.clone(),
            narrow
        )
        .is_err());
    }
}
