// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Detect einsums in DataFusion logical plans (design doc §9.1).
//!
//! [`detect`] reads one `Aggregate` node as an einsum: an `Aggregate(group by
//! G; SUM(e))` over a tree of inner equi-joins, filters, projections and
//! subquery aliases. Every node below that tree is a *leaf*, an opaque operand.
//! Detection only describes the einsum; replacing the node is the optimizer
//! rule's job.
//!
//! Detection declines (returns `None`) whenever it cannot prove that the einsum
//! means exactly what the plan means (design §4, supplement §6.2): which groups
//! exist, which values are NULL, how NULL keys join, and bag semantics.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, DFSchema, DFSchemaRef, NullEquality};
use datafusion::functions_aggregate::sum::Sum;
use datafusion::logical_expr::utils::{conjunction, split_conjunction_owned};
use datafusion::logical_expr::{
    BinaryExpr, Expr, ExprSchemable, Filter, JoinType, LogicalPlan, Operator,
};
use einfold_ir::{Dim, Einsum, KeyEquality, Operand, Semiring};

/// An `Aggregate` node read as an einsum, with what is needed to rebuild it
/// (design §9.1).
#[derive(Clone, Debug)]
pub struct Detected {
    /// The einsum. Its dimensions are named after their dimension classes
    /// (design §9.1, step 5); operands are the leaves in plan order.
    pub einsum: Einsum,
    /// One binding per operand, in the order of `einsum.operands()`.
    pub operands: Vec<OperandInput>,
    /// One entry per group expression of the `Aggregate`, in its order: the
    /// `Aggregate`'s output column and the dimension it holds. Two group keys
    /// in one dimension class hold the same dimension.
    pub group_outputs: Vec<(Column, Dim)>,
    /// The `Aggregate`'s output column holding the `SUM`.
    pub sum_output: Column,
    /// The `Aggregate`'s output schema, which a rewrite must reproduce.
    pub schema: DFSchemaRef,
}

/// How one einsum operand reads its leaf of the plan.
#[derive(Clone, Debug)]
pub struct OperandInput {
    /// The leaf subtree, with every filter over only its columns applied on
    /// top (design §9.1, step 6).
    pub plan: Arc<LogicalPlan>,
    /// The leaf's column for each entry of the operand's `dims`, in order.
    pub dim_columns: Vec<Column>,
    /// The operand's factor, over `plan`'s columns only: a `Float64` that
    /// multiplies into every product the operand's rows take part in. `1.0`
    /// for an operand that only joins.
    pub value: Expr,
}

/// Try to read the `Aggregate` node `plan` as an einsum (design §9.1).
///
/// Matches `SUM(e)` over `Float64`, with plain-column group keys, over inner
/// equi-joins, filters, projections and aliases down to opaque leaves. `e` must
/// be a product whose factors each read one leaf. Returns `None` whenever the
/// node is not provably an einsum, so the plan stays unchanged.
pub fn detect(plan: &LogicalPlan) -> Option<Detected> {
    let LogicalPlan::Aggregate(agg) = plan else {
        return None;
    };
    // SQL returns one row from an aggregate without GROUP BY even when nothing
    // joined, but an einsum has a row only where some joined row reached
    // (design §8.3). Decline until the IR can say which it means.
    if agg.group_expr.is_empty() || agg.aggr_expr.len() != 1 {
        return None;
    }
    let Expr::AggregateFunction(sum) = strip_alias(&agg.aggr_expr[0]) else {
        return None;
    };
    let p = &sum.params;
    if !sum.func.inner().is::<Sum>()
        || p.distinct
        || p.filter.is_some()
        || !p.order_by.is_empty()
        || p.null_treatment.is_some()
        || p.args.len() != 1
    {
        return None;
    }
    // Integer and DECIMAL sums must stay exact (design §8.6, item 2).
    if p.args[0].get_type(agg.input.schema()).ok()? != DataType::Float64 {
        return None;
    }

    let mut b = Builder::default();
    let cols = b.visit(&agg.input, None)?;
    let input_schema = agg.input.schema();
    let sum_expr = inline(&p.args[0], input_schema, &cols)?;
    let mut group_slots = Vec::new();
    for g in &agg.group_expr {
        let Expr::Column(_) = g else { return None };
        group_slots.push(slot_of(&inline(g, input_schema, &cols)?)?);
    }
    b.finish(sum_expr, group_slots, agg.schema.clone())
}

/// One opaque leaf: an operand.
#[derive(Debug)]
struct Leaf {
    plan: Arc<LogicalPlan>,
    name: String,
    /// The slot of the leaf's first column; its columns are consecutive.
    first_slot: usize,
}

/// Detection state. Every column of every leaf gets a *slot*, written as a
/// placeholder column `__einfold_<slot>`, so that two scans of one table stay
/// distinct. Expressions are inlined down to slots.
#[derive(Default, Debug)]
struct Builder {
    leaves: Vec<Leaf>,
    /// For each slot: its leaf and its type.
    slots: Vec<(usize, Field)>,
    /// Column equalities between slots, from joins and filters.
    equalities: Vec<(usize, usize, KeyEquality)>,
    /// Other filter conjuncts, over slots.
    predicates: Vec<Expr>,
}

impl Builder {
    /// Walk the tree below the aggregate. Returns, for each output column of
    /// `plan`, its expression over slots.
    fn visit(&mut self, plan: &LogicalPlan, alias: Option<&str>) -> Option<Vec<Expr>> {
        match plan {
            LogicalPlan::Projection(proj) => {
                let cols = self.visit(&proj.input, alias)?;
                let schema = proj.input.schema();
                proj.expr.iter().map(|e| inline(e, schema, &cols)).collect()
            }
            LogicalPlan::Filter(filter) => {
                let cols = self.visit(&filter.input, alias)?;
                let pred = inline(&filter.predicate, filter.input.schema(), &cols)?;
                self.predicates.extend(split_conjunction_owned(pred));
                Some(cols)
            }
            LogicalPlan::SubqueryAlias(sa) => {
                self.visit(&sa.input, Some(alias.unwrap_or(sa.alias.table())))
            }
            LogicalPlan::Join(join) => {
                if join.join_type != JoinType::Inner || join.null_aware {
                    return None;
                }
                let kind = match join.null_equality {
                    NullEquality::NullEqualsNothing => KeyEquality::Equal,
                    NullEquality::NullEqualsNull => KeyEquality::NotDistinctFrom,
                };
                let mut cols = self.visit(&join.left, None)?;
                let right = self.visit(&join.right, None)?;
                for (l, r) in &join.on {
                    let l = slot_of(&inline(l, join.left.schema(), &cols)?)?;
                    let r = slot_of(&inline(r, join.right.schema(), &right)?)?;
                    self.equalities.push((l, r, kind));
                }
                cols.extend(right);
                // An inner join's filter means what a `Filter` above it means.
                if let Some(filter) = &join.filter {
                    let pred = inline(filter, &join.schema, &cols)?;
                    self.predicates.extend(split_conjunction_owned(pred));
                }
                Some(cols)
            }
            leaf => Some(self.add_leaf(leaf, alias)),
        }
    }

    fn add_leaf(&mut self, plan: &LogicalPlan, alias: Option<&str>) -> Vec<Expr> {
        let index = self.leaves.len();
        let schema = plan.schema();
        let name = alias
            .map(str::to_owned)
            .or_else(|| {
                let (q, _) = schema.iter().next()?;
                q.map(|q| q.table().to_owned())
            })
            .unwrap_or_else(|| format!("op{index}"));
        let first_slot = self.slots.len();
        self.leaves.push(Leaf {
            plan: Arc::new(plan.clone()),
            name,
            first_slot,
        });
        schema
            .fields()
            .iter()
            .map(|f| {
                let slot = self.slots.len();
                let field = Field::new(slot_name(slot), f.data_type().clone(), f.is_nullable());
                self.slots.push((index, field));
                Expr::Column(Column::new_unqualified(slot_name(slot)))
            })
            .collect()
    }

    /// The schema of all slots, for typing expressions over them.
    fn slot_schema(&self) -> Option<DFSchema> {
        let fields = self.slots.iter().map(|(_, f)| f.clone()).collect();
        DFSchema::from_unqualified_fields(fields, HashMap::new()).ok()
    }

    /// Build the einsum from the walked tree, the aggregated expression and the
    /// group keys' slots.
    fn finish(
        mut self,
        sum_expr: Expr,
        group_slots: Vec<usize>,
        schema: DFSchemaRef,
    ) -> Option<Detected> {
        // Classify filter conjuncts (design §9.1, step 6): column equalities
        // join dimension classes; a predicate over one leaf stays with it;
        // anything else (a mask, or a comparison of values) is declined in M1.
        let mut leaf_preds: Vec<Vec<Expr>> = vec![Vec::new(); self.leaves.len()];
        for pred in std::mem::take(&mut self.predicates) {
            if let Some((l, r, kind)) = column_equality(&pred) {
                self.equalities.push((l, r, kind));
                continue;
            }
            match self.leaves_of(&pred)?.as_slice() {
                [leaf] => leaf_preds[*leaf].push(pred),
                _ => return None,
            }
        }

        // Dimension classes (step 5) and group keys (step 7).
        let mut uf = UnionFind::new(self.slots.len());
        for &(l, r, _) in &self.equalities {
            uf.union(l, r);
        }
        let mut kinds: BTreeMap<usize, KeyEquality> = BTreeMap::new();
        for &(l, _, kind) in &self.equalities {
            if *kinds.entry(uf.find(l)).or_insert(kind) != kind {
                return None; // one class mixes `=` and `IS NOT DISTINCT FROM`
            }
        }
        // A group key outside every equality is a dimension of its own; GROUP
        // BY keeps NULL as one more coordinate.
        for &s in &group_slots {
            kinds
                .entry(uf.find(s))
                .or_insert(KeyEquality::NotDistinctFrom);
        }
        let dims = self.name_classes(&kinds);

        let factors = self.factors(sum_expr)?;

        let mut operands = Vec::new();
        let mut inputs = Vec::new();
        for (index, (leaf, (factor, preds))) in self
            .leaves
            .iter()
            .zip(factors.into_iter().zip(leaf_preds))
            .enumerate()
        {
            let schema = leaf.plan.schema();
            let to_leaf = |e: Expr| self.to_leaf(index, schema, e);
            let mut plan = leaf.plan.clone();
            if let Some(pred) = conjunction(preds) {
                plan = Arc::new(LogicalPlan::Filter(
                    Filter::try_new(to_leaf(pred)?, plan).ok()?,
                ));
            }
            let mut op_dims = Vec::new();
            let mut dim_columns = Vec::new();
            for j in 0..schema.fields().len() {
                if let Some(d) = dims.get(&uf.find(leaf.first_slot + j)) {
                    op_dims.push(d.clone());
                    dim_columns.push(Column::from(schema.qualified_field(j)));
                }
            }
            operands.push(Operand::new(leaf.name.clone(), op_dims));
            inputs.push(OperandInput {
                plan,
                dim_columns,
                value: to_leaf(factor.unwrap_or(Expr::Literal(1.0f64.into(), None)))?,
            });
        }

        let group_dims: Vec<Dim> = group_slots
            .iter()
            .map(|&s| dims[&uf.find(s)].clone())
            .collect();
        let mut output: Vec<Dim> = Vec::new();
        for d in &group_dims {
            if !output.contains(d) {
                output.push(d.clone());
            }
        }
        let equality = kinds
            .iter()
            .map(|(root, k)| (dims[root].clone(), *k))
            .collect();
        let einsum = Einsum::new(operands, output, equality, Semiring::SumProduct).ok()?;
        let group_outputs = group_dims
            .into_iter()
            .enumerate()
            .map(|(i, d)| (Column::from(schema.qualified_field(i)), d))
            .collect();
        let sum_output = Column::from(schema.qualified_field(group_slots.len()));
        Some(Detected {
            einsum,
            operands: inputs,
            group_outputs,
            sum_output,
            schema,
        })
    }

    /// Name each class after its first column's name, qualified by its leaf
    /// when two classes would otherwise share a name.
    /// `kinds` is keyed by class root, which is the class's first slot.
    fn name_classes(&self, kinds: &BTreeMap<usize, KeyEquality>) -> BTreeMap<usize, Dim> {
        let mut used = BTreeSet::new();
        let mut names = BTreeMap::new();
        for &root in kinds.keys() {
            let leaf = &self.leaves[self.slots[root].0];
            let base = leaf
                .plan
                .schema()
                .field(root - leaf.first_slot)
                .name()
                .clone();
            let mut name = base.clone();
            if used.contains(&name) {
                name = format!("{}.{base}", leaf.name);
            }
            let mut n = 2;
            while used.contains(&name) {
                name = format!("{base}#{n}");
                n += 1;
            }
            used.insert(name.clone());
            names.insert(root, Dim::new(name));
        }
        names
    }

    /// Split the aggregated expression into one factor per leaf (design §9.1,
    /// steps 3 and 4). Returns `None` unless it is a product of `Float64`
    /// factors that each read at most one leaf. Factors of one leaf are
    /// multiplied together, and constant factors go to the first operand.
    fn factors(&self, sum_expr: Expr) -> Option<Vec<Option<Expr>>> {
        let schema = self.slot_schema()?;
        let mut flat = Vec::new();
        flatten_product(sum_expr, &schema, &mut flat)?;
        let mut factors: Vec<Option<Expr>> = vec![None; self.leaves.len()];
        let mut push = |i: usize, f: Expr| {
            factors[i] = Some(match factors[i].take() {
                Some(prev) => prev * f,
                None => f,
            });
        };
        let mut constants = Vec::new();
        for f in flat {
            if f.get_type(&schema).ok()? != DataType::Float64 {
                return None;
            }
            match self.leaves_of(&f)?.as_slice() {
                [] => constants.push(f),
                [leaf] => push(*leaf, f),
                _ => return None,
            }
        }
        for c in constants {
            push(0, c);
        }
        Some(factors)
    }

    /// The leaves whose columns `e` reads, in order.
    fn leaves_of(&self, e: &Expr) -> Option<Vec<usize>> {
        let mut leaves = BTreeSet::new();
        for c in e.column_refs() {
            leaves.insert(self.slots[parse_slot(&c.name)?].0);
        }
        Some(leaves.into_iter().collect())
    }

    /// Rewrite an expression over one leaf's slots into one over its columns.
    fn to_leaf(&self, leaf: usize, schema: &DFSchema, e: Expr) -> Option<Expr> {
        let first = self.leaves[leaf].first_slot;
        e.transform(|e| match e {
            Expr::Column(c) => {
                let j = parse_slot(&c.name).expect("slot column") - first;
                let col = Column::from(schema.qualified_field(j));
                Ok(Transformed::yes(Expr::Column(col)))
            }
            e => Ok(Transformed::no(e)),
        })
        .ok()
        .map(|t| t.data)
    }
}

/// Flatten nested `Float64` multiplications into their factors.
fn flatten_product(e: Expr, schema: &DFSchema, out: &mut Vec<Expr>) -> Option<()> {
    match e {
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Multiply,
            right,
        }) if left.get_type(schema).ok()? == DataType::Float64
            && right.get_type(schema).ok()? == DataType::Float64 =>
        {
            flatten_product(*left, schema, out)?;
            flatten_product(*right, schema, out)
        }
        e => {
            out.push(e);
            Some(())
        }
    }
}

/// `a = b` or `a IS NOT DISTINCT FROM b` between two slots.
fn column_equality(e: &Expr) -> Option<(usize, usize, KeyEquality)> {
    let Expr::BinaryExpr(BinaryExpr { left, op, right }) = e else {
        return None;
    };
    let kind = match op {
        Operator::Eq => KeyEquality::Equal,
        Operator::IsNotDistinctFrom => KeyEquality::NotDistinctFrom,
        _ => return None,
    };
    Some((slot_of(left)?, slot_of(right)?, kind))
}

fn strip_alias(e: &Expr) -> &Expr {
    match e {
        Expr::Alias(a) => strip_alias(&a.expr),
        e => e,
    }
}

fn slot_name(slot: usize) -> String {
    format!("__einfold_{slot}")
}

fn parse_slot(name: &str) -> Option<usize> {
    name.strip_prefix("__einfold_")?.parse().ok()
}

/// The slot an inlined expression is, if it is a plain column.
fn slot_of(e: &Expr) -> Option<usize> {
    match e {
        Expr::Column(c) if c.relation.is_none() => parse_slot(&c.name),
        _ => None,
    }
}

/// Whether detection can move `e` (evaluate it on the leaf's rows rather than
/// on the joined rows): a deterministic, row-wise expression of known kind.
fn movable(e: &Expr) -> bool {
    !e.exists(|e| {
        Ok(match e {
            Expr::ScalarFunction(_) => e.is_volatile_node(),
            Expr::Alias(_)
            | Expr::Column(_)
            | Expr::Literal(..)
            | Expr::BinaryExpr(_)
            | Expr::Like(_)
            | Expr::SimilarTo(_)
            | Expr::Not(_)
            | Expr::IsNotNull(_)
            | Expr::IsNull(_)
            | Expr::IsTrue(_)
            | Expr::IsFalse(_)
            | Expr::IsUnknown(_)
            | Expr::IsNotTrue(_)
            | Expr::IsNotFalse(_)
            | Expr::IsNotUnknown(_)
            | Expr::Negative(_)
            | Expr::Between(_)
            | Expr::Case(_)
            | Expr::Cast(_)
            | Expr::TryCast(_)
            | Expr::InList(_) => false,
            _ => true,
        })
    })
    .expect("infallible")
}

/// Rewrite `e`, an expression over `schema`, into one over slots, given each
/// column's expression `cols`. Aliases are dropped.
fn inline(e: &Expr, schema: &DFSchema, cols: &[Expr]) -> Option<Expr> {
    if !movable(e) {
        return None;
    }
    let mut missing = false;
    let out = e
        .clone()
        .transform(|e| match e {
            Expr::Column(c) => match schema.maybe_index_of_column(&c) {
                Some(i) => Ok(Transformed::yes(cols[i].clone())),
                None => {
                    missing = true;
                    Ok(Transformed::no(Expr::Column(c)))
                }
            },
            e => Ok(Transformed::no(e)),
        })
        .ok()?
        .data;
    (!missing).then(|| out.unalias_nested().data)
}

/// Union-find over slots.
struct UnionFind(Vec<usize>);

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind((0..n).collect())
    }

    fn find(&mut self, x: usize) -> usize {
        let p = self.0[x];
        if p == x {
            return x;
        }
        let root = self.find(p);
        self.0[x] = root;
        root
    }

    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        // The smaller slot is the root, so a class's root is its first column.
        self.0[a.max(b)] = a.min(b);
    }
}
