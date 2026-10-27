// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! [`PipelineNode`]: a fold pipeline as a node of DataFusion's logical plan.
//!
//! The node stands for the producer fold, the softmax-weighted average that
//! reads it, and the joins between them (see [`einfold_ir::pipeline`]).
//! DataFusion's optimizer leaves it alone, and
//! [`EinFoldPlanner`](crate::EinFoldPlanner) turns it into
//! [`EinFoldExec`](crate::EinFoldExec).

use std::cmp::Ordering;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::{Field, Schema};
use datafusion::common::{plan_err, DFSchema, DFSchemaRef, Result};
use datafusion::logical_expr::{Expr, Extension, LogicalPlan, UserDefinedLogicalNodeCore};
use einfold_ir::{DimensionId, Normalizer, Pipeline, Shift};

use crate::fold_node;

/// A fold pipeline in a DataFusion logical plan. Its inputs are the
/// producer's two operands, then the values operand; its schema is
/// [`einfold_kernels::pipeline_schema`]'s, under the names it is given. It
/// always keeps the plan it replaced, which runs wherever the fused
/// algorithm declines.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineNode {
    pipeline: Pipeline,
    inputs: Vec<LogicalPlan>,
    schema: DFSchemaRef,
    /// The plan this node replaced: its columns are the node's, in order,
    /// though their names may differ.
    fallback: LogicalPlan,
}

impl PipelineNode {
    /// The node computing `pipeline` over `inputs` (the producer's operands,
    /// then the values), with `names` naming its columns, and `fallback`, a
    /// plan with the same columns, running wherever the fused algorithm
    /// declines. Fails when the inputs don't fit the pipeline, or when there
    /// isn't one name per column.
    pub fn try_new(
        pipeline: Pipeline,
        inputs: [LogicalPlan; 3],
        names: Vec<String>,
        fallback: LogicalPlan,
    ) -> Result<Self> {
        let arrow = |p: &LogicalPlan| p.schema().as_arrow().clone();
        let [q, k, v] = [&inputs[0], &inputs[1], &inputs[2]].map(arrow);
        let schema = einfold_kernels::pipeline_schema(&pipeline, &[&q, &k, &v])?;
        if names.len() != schema.fields().len() {
            return plan_err!(
                "a pipeline with {} columns, given {} names",
                schema.fields().len(),
                names.len()
            );
        }
        let fields: Vec<Field> = (schema.fields().iter().zip(names))
            .map(|(f, name)| f.as_ref().clone().with_name(name))
            .collect();
        let schema = Arc::new(DFSchema::try_from(Schema::new(fields))?);
        if fallback.schema().fields().len() != schema.fields().len() {
            return plan_err!("a pipeline's fallback must have its columns");
        }
        Ok(PipelineNode {
            pipeline,
            inputs: inputs.into(),
            schema,
            fallback,
        })
    }

    /// The pipeline this node computes.
    #[must_use]
    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    /// The plan this node replaced.
    #[must_use]
    pub fn fallback(&self) -> &LogicalPlan {
        &self.fallback
    }

    /// This node as a logical plan.
    #[must_use]
    pub fn into_plan(self) -> LogicalPlan {
        LogicalPlan::Extension(Extension {
            node: Arc::new(self),
        })
    }
}

// The schema follows from the pipeline and the inputs, so ordering ignores
// it.
impl PartialOrd for PipelineNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        fn key(n: &PipelineNode) -> impl PartialOrd + '_ {
            (&n.pipeline, &n.inputs, &n.fallback)
        }
        key(self).partial_cmp(&key(other))
    }
}

impl UserDefinedLogicalNodeCore for PipelineNode {
    fn name(&self) -> &str {
        "Pipeline"
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
        write!(f, "Pipeline: {}", describe(&self.pipeline))
    }

    fn with_exprs_and_inputs(&self, exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> Result<Self> {
        if !exprs.is_empty() {
            return plan_err!("a Pipeline node has no expressions");
        }
        let Ok(inputs) = <[LogicalPlan; 3]>::try_from(inputs) else {
            return plan_err!("a Pipeline node has three inputs");
        };
        let names = self.schema.fields().iter().map(|f| f.name().clone());
        PipelineNode::try_new(
            self.pipeline.clone(),
            inputs,
            names.collect(),
            self.fallback.clone(),
        )
    }
}

/// The pipeline for `EXPLAIN`: its producer as [`FoldNode`](crate::FoldNode)
/// shows a fold, then the average it feeds, such as `a,b * c,b -> a,c where a
/// >= c; softmax(/ 8, shifted) * c,d -> a,d` for causal attention.
pub(crate) fn describe(pipeline: &Pipeline) -> String {
    let producer = pipeline.producer();
    let name = |DimensionId(d): DimensionId| match u8::try_from(d) {
        Ok(d) if d < 26 => char::from(b'a' + d).to_string(),
        _ => format!("#{d}"),
    };
    let values: Vec<String> = (pipeline.values().keys.iter())
        .map(|k| name(k.dimension))
        .collect();
    let outputs: Vec<String> = pipeline.outputs().iter().map(|&d| name(d)).collect();
    let shift = match pipeline.shift() {
        Shift::KeptMaximum => ", shifted",
        _ => "",
    };
    let normalizer = match pipeline.normalizer() {
        Normalizer::JoinedRows => ", over joined rows",
        _ => "",
    };
    format!(
        "{}; softmax({}{shift}{normalizer}) * {} -> {}",
        fold_node::describe(producer, einfold_kernels::DomainRows::Any),
        pipeline.scale().to_string().trim(),
        values.join(","),
        outputs.join(",")
    )
}
