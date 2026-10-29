// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Running a query with einfold on and off, and comparing what comes back.

use std::sync::Arc;

use datafusion::arrow::array::{Array, ArrayRef, Float64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Schema};
use datafusion::arrow::util::display;
use datafusion::execution::memory_pool::FairSpillPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::execution::SessionStateBuilder;
use datafusion::physical_plan::display::DisplayableExecutionPlan;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::prelude::{SessionConfig, SessionContext};
use einfold_datafusion::{DetectFolds, EinFoldQueryPlanner};

/// Session settings a case varies, the same for the host and einfold: how
/// DataFusion plans joins and aggregates, and how much memory it may use.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Settings {
    /// Sort-merge joins rather than hash joins.
    pub sort_merge: bool,
    /// No repartitioning of joins and aggregates.
    pub no_repartition: bool,
    /// No dynamic filters pushed from joins and TopK into scans.
    pub no_dynamic_filters: bool,
    /// A memory pool of this many bytes, which DataFusion's own operators
    /// can spill from, rather than an unbounded one.
    pub pool: Option<usize>,
}

/// A session on `threads` partitions with `settings`: plain DataFusion, or
/// with einfold installed; with `fallback` false, every fold einfold finds
/// runs through its own kernels, the hash algorithm included.
pub fn session(
    threads: usize,
    einfold: Option<bool>,
    batch_size: usize,
    settings: Settings,
) -> SessionContext {
    let config = SessionConfig::new()
        .with_target_partitions(threads)
        .with_batch_size(batch_size)
        .set_bool(
            "datafusion.optimizer.prefer_hash_join",
            !settings.sort_merge,
        )
        .set_bool(
            "datafusion.optimizer.repartition_joins",
            !settings.no_repartition,
        )
        .set_bool(
            "datafusion.optimizer.repartition_aggregations",
            !settings.no_repartition,
        )
        .set_bool(
            "datafusion.optimizer.enable_dynamic_filter_pushdown",
            !settings.no_dynamic_filters,
        );
    let mut runtime = RuntimeEnvBuilder::new();
    if let Some(bytes) = settings.pool {
        runtime = runtime.with_memory_pool(Arc::new(FairSpillPool::new(bytes)));
    }
    let builder = SessionStateBuilder::new()
        .with_default_features()
        .with_config(config)
        .with_runtime_env(runtime.build_arc().expect("a runtime"));
    let builder = match einfold {
        None => builder,
        Some(fallback) => {
            let rule = match fallback {
                true => DetectFolds::default(),
                false => DetectFolds::without_fallback(),
            };
            builder.with_query_planner(Arc::new(EinFoldQueryPlanner::detecting(rule)))
        }
    };
    SessionContext::new_with_state(builder.build())
}

/// What running a query gave.
pub struct Ran {
    pub result: Result<Vec<RecordBatch>, String>,
    /// The logical schema DataFusion reports for the query.
    pub schema: Option<Arc<Schema>>,
    /// The physical plan, run, with its metrics, if it got that far.
    pub plan: Option<Arc<dyn ExecutionPlan>>,
}

impl Ran {
    /// The plan as text, with metrics.
    pub fn shown(&self) -> String {
        match &self.plan {
            Some(p) => DisplayableExecutionPlan::with_metrics(p.as_ref())
                .indent(true)
                .to_string(),
            None => "(no plan)".into(),
        }
    }

    /// How many `EinFoldExec`s ran, and how many by each route.
    pub fn routes(&self) -> Routes {
        let shown = self.shown();
        Routes {
            folds: shown.matches("EinFoldExec:").count(),
            domains: shown.matches("domain").count(),
            dense: shown.matches("dense_folds=1").count(),
            // The hash algorithm, or the discrimination algorithm that
            // replaces it: einfold's own route for any keys.
            hash: shown.matches("hash_folds=1").count()
                + shown.matches("discrimination_folds=1").count(),
            host: shown.matches("host_plans=1").count(),
            other_aggregates: shown.matches("AggregateExec").count(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Routes {
    pub folds: usize,
    pub domains: usize,
    pub dense: usize,
    pub hash: usize,
    pub host: usize,
    /// Aggregates the host runs outside einfold.
    pub other_aggregates: usize,
}

pub async fn run(ctx: &SessionContext, sql: &str) -> Ran {
    let df = match ctx.sql(sql).await {
        Ok(df) => df,
        Err(e) => {
            return Ran {
                result: Err(e.to_string()),
                schema: None,
                plan: None,
            }
        }
    };
    let schema = Some(Arc::new(df.schema().as_arrow().clone()));
    let plan = match df.create_physical_plan().await {
        Ok(p) => p,
        Err(e) => {
            return Ran {
                result: Err(e.to_string()),
                schema,
                plan: None,
            }
        }
    };
    // On a task of its own, so that a panic is reported as a failure of
    // this case rather than ending the run.
    let (p, task) = (Arc::clone(&plan), ctx.task_ctx());
    let result = match tokio::spawn(
        async move { datafusion::physical_plan::collect(p, task).await },
    )
    .await
    {
        Ok(r) => r.map_err(|e| e.to_string()),
        Err(e) => Err(format!("panicked: {e}")),
    };
    Ran {
        result,
        schema,
        plan: Some(plan),
    }
}

/// Runs a plan that already ran once, again, after resetting its state as
/// DataFusion asks of a plan it runs twice (`reset_plan_states`, which a
/// recursive query uses for each iteration).
pub async fn rerun(ctx: &SessionContext, ran: &Ran) -> Option<Result<Vec<RecordBatch>, String>> {
    let plan = ran.plan.as_ref()?;
    let plan = match datafusion::physical_plan::execution_plan::reset_plan_states(Arc::clone(plan))
    {
        Ok(p) => p,
        Err(e) => return Some(Err(e.to_string())),
    };
    let task = ctx.task_ctx();
    let collected =
        tokio::spawn(async move { datafusion::physical_plan::collect(plan, task).await });
    Some(match collected.await {
        Ok(r) => r.map_err(|e| e.to_string()),
        Err(e) => Err(format!("panicked: {e}")),
    })
}

/// A cell, as text that is equal exactly when the cells are the same value:
/// floats by their bits, except that in a value (`value` true) every NaN is
/// one, as a NaN's sign and payload depend on the order of additions, even
/// within the host; a dictionary value as the value. Keys keep their NaNs'
/// bits: the host groups NaNs of different bits apart.
fn cell(column: &ArrayRef, r: usize, value: bool) -> String {
    let column = match column.data_type() {
        DataType::Dictionary(_, v) => {
            datafusion::arrow::compute::cast(column, v).expect("a dictionary's values")
        }
        _ => Arc::clone(column),
    };
    if !column.is_valid(r) {
        return "NULL".into();
    }
    match column.as_any().downcast_ref::<Float64Array>() {
        Some(f) if value && f.value(r).is_nan() => "NaN".into(),
        Some(f) => format!("{:016x}", f.value(r).to_bits()),
        None => display::array_value_to_string(&column, r).expect("a displayable value"),
    }
}

/// A row: its key cells (every column but those named `v`), and its values.
#[derive(Clone, Debug, PartialEq, PartialOrd)]
pub struct Row {
    pub keys: Vec<String>,
    pub values: Vec<Option<f64>>,
}

pub fn rows(batches: &[RecordBatch]) -> Vec<Row> {
    let mut out = Vec::new();
    for b in batches {
        let schema = b.schema();
        let is_value = |i: usize| schema.field(i).name() == "v";
        for r in 0..b.num_rows() {
            let mut row = Row {
                keys: Vec::new(),
                values: Vec::new(),
            };
            for (i, c) in b.columns().iter().enumerate() {
                match c.as_any().downcast_ref::<Float64Array>() {
                    Some(f) if is_value(i) => row.values.push(f.is_valid(r).then(|| f.value(r))),
                    _ => row.keys.push(cell(c, r, false)),
                }
            }
            out.push(row);
        }
    }
    // Rows with the same keys in the order of their values' classes and
    // then values, so that two runs pair them alike.
    out.sort_by(|x, y| {
        let class = |r: &Row| -> Vec<(u8, f64)> {
            (r.values.iter())
                .map(|v| match v {
                    None => (0, 0.0),
                    Some(f) if f.is_nan() => (1, 0.0),
                    Some(f) => (2, *f),
                })
                .collect()
        };
        (&x.keys, class(x))
            .partial_cmp(&(&y.keys, class(y)))
            .expect("no NaN compared")
    });
    out
}

/// Every cell as text, sorted: equal exactly when the results are the same
/// multiset of rows, bit for bit (every NaN alike).
pub fn exact(batches: &[RecordBatch]) -> Vec<String> {
    let mut out = Vec::new();
    for b in batches {
        let schema = b.schema();
        for r in 0..b.num_rows() {
            let row: Vec<String> = (b.columns().iter().enumerate())
                .map(|(i, c)| cell(c, r, schema.field(i).name() == "v"))
                .collect();
            out.push(row.join("|"));
        }
    }
    out.sort();
    out
}

/// Whether two values agree: both NULL, both NaN, the same infinity, the
/// same zero, or finite and within `bound`.
pub fn agree(ours: Option<f64>, theirs: Option<f64>, bound: f64) -> bool {
    match (ours, theirs) {
        (None, None) => true,
        (Some(x), Some(y)) if x.is_nan() || y.is_nan() => x.is_nan() && y.is_nan(),
        (Some(x), Some(y)) if x == y => x.is_sign_negative() == y.is_sign_negative(),
        (Some(x), Some(y)) => x.is_finite() && y.is_finite() && (x - y).abs() <= bound,
        _ => false,
    }
}

/// Whether two schemas have the same fields: names, types, nullability and
/// metadata, in order.
pub fn same_fields(a: &Schema, b: &Schema) -> Result<(), String> {
    if a.fields().len() != b.fields().len() {
        return Err(format!(
            "{} columns, the host has {}",
            a.fields().len(),
            b.fields().len()
        ));
    }
    for (x, y) in a.fields().iter().zip(b.fields()) {
        if x.name() != y.name()
            || x.data_type() != y.data_type()
            || x.is_nullable() != y.is_nullable()
            || x.metadata() != y.metadata()
        {
            return Err(format!("column {x:?}, the host's {y:?}"));
        }
    }
    Ok(())
}
