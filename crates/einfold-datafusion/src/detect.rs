// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Find folds over joins in DataFusion logical plans.
//!
//! A *fold over a join* is a join followed by an aggregate, grouped by some of
//! the join's columns:
//!
//! ```sql
//! SELECT a.i, b.j, SUM(a.v * b.v) FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j
//! ```
//!
//! [`detect`] reads one `Aggregate` node in that shape: `SUM`, `COUNT` or
//! `AVG` over a tree of inner equi-joins, filters, projections and subquery
//! aliases. Every node below that tree is a *leaf*: a subplan detection
//! doesn't look inside (a table scan, a subquery with its own `GROUP BY`, ...).
//! Each leaf becomes one *operand* of the fold. The aggregated expression must
//! be a product of *factors*, each reading the columns of one operand only;
//! the example above, a matrix product, is the most important case and is
//! called an *einsum*. Detection only describes the fold; replacing the node
//! is an optimizer rule's job.
//!
//! Detection declines (returns `None`) whenever it cannot prove that the fold
//! means exactly what the plan means. In particular it preserves:
//!
//! - **which groups exist:** a group appears only if at least one joined row
//!   reached it, whatever the aggregate;
//! - **NULLs:** every aggregate skips NULL inputs. A group whose inputs were
//!   all NULL has a NULL `SUM` and `AVG`, and a `COUNT` of 0;
//! - **how NULL keys join:** `=` never matches NULL, `IS NOT DISTINCT FROM`
//!   matches NULL to NULL;
//! - **duplicates:** every joined row counts, even if its key tuple repeats;
//! - **errors:** nothing that could fail is moved to run on rows the plan never
//!   evaluated it on.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field};
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, DFSchema, DFSchemaRef, NullEquality, ScalarValue};
use datafusion::functions_aggregate::average::Avg;
use datafusion::functions_aggregate::count::Count;
use datafusion::functions_aggregate::sum::Sum;
use datafusion::logical_expr::utils::{conjunction, split_conjunction_owned};
use datafusion::logical_expr::{
    when, BinaryExpr, Expr, ExprSchemable, Filter, JoinType, LogicalPlan, Operator,
};
use einfold_ir::{Aggregate, Dim, Fold, KeyEquality, Op, Operand, RowValue};

/// A fold found in a plan: the [`Fold`], bound to the plan's inputs and to the
/// `Aggregate` node's output, so that a rule can replace the node.
#[derive(Clone, Debug)]
pub struct FoldMatch {
    /// The fold. Each dimension is a set of columns that the joins and filters
    /// force to be equal, such as `{a.k, b.k}` for `a.k = b.k`, named after its
    /// first column. Operands are the leaves, in plan order. Its row value is
    /// always a product of the operands' factors, so whether it is a semiring
    /// fold follows from the aggregate ([`Fold::semiring`]): `SUM` and `COUNT`
    /// are, `AVG` is `SUM / COUNT` of two that are.
    pub fold: Fold,
    /// One binding per operand, in the order of `fold.operands()`.
    pub operands: Vec<OperandInput>,
    /// One entry per group expression of the `Aggregate`, in its order: the
    /// `Aggregate`'s output column and the dimension it holds. Two group keys
    /// whose columns are equated hold the same dimension.
    pub group_outputs: Vec<(Column, Dim)>,
    /// The `Aggregate`'s output column holding the aggregate's value.
    pub value_output: Column,
    /// The `Aggregate`'s output schema, which a rewrite must reproduce.
    pub schema: DFSchemaRef,
}

/// How one operand reads its leaf of the plan.
#[derive(Clone, Debug)]
pub struct OperandInput {
    /// The leaf subtree, with every filter that reads only its columns applied
    /// on top. For an inner join, filtering one side before the join equals
    /// filtering the joined rows after it.
    pub plan: Arc<LogicalPlan>,
    /// The leaf's column for each entry of the operand's `dims`, in order.
    pub dim_columns: Vec<Column>,
    /// The operand's factor, a `Float64` expression over `plan`'s columns
    /// only. Each joined row's value is the product of its operands' factors,
    /// NULL if any factor is. An operand that only joins has factor `1.0`.
    ///
    /// For `COUNT`, only whether a value is NULL matters, so each factor is
    /// `1.0` where the original factor is non-NULL and NULL where it is NULL.
    /// This also lets `COUNT` count values of any type.
    pub value: Expr,
}

/// Try to read the `Aggregate` node `plan` as a fold over a join.
///
/// Matches `SUM(e)` and `AVG(e)` over `Float64`, and `COUNT(e)` or `COUNT(*)`
/// over any type, with plain-column group keys, over inner equi-joins,
/// filters, projections and aliases down to opaque leaves. `e` must be a
/// product whose factors each read one leaf. Returns `None` whenever the node
/// is not provably such a fold, so the plan stays unchanged.
pub fn detect(plan: &LogicalPlan) -> Option<FoldMatch> {
    let LogicalPlan::Aggregate(agg) = plan else {
        return None;
    };
    // Without GROUP BY, SQL returns one row even when no row joined (with a
    // NULL sum, or a count of 0), but a fold has a row only where some joined
    // row reached. Decline until the IR can express that.
    if agg.group_expr.is_empty() || agg.aggr_expr.len() != 1 {
        return None;
    }
    let Expr::AggregateFunction(func) = strip_alias(&agg.aggr_expr[0]) else {
        return None;
    };
    let p = &func.params;
    if p.distinct
        || p.filter.is_some()
        || !p.order_by.is_empty()
        || p.null_treatment.is_some()
        || p.args.len() != 1
    {
        return None;
    }
    let udf = func.func.inner();
    let aggregate = if udf.is::<Sum>() {
        Aggregate::SUM
    } else if udf.is::<Avg>() {
        Aggregate::AVG
    } else if udf.is::<Count>() {
        Aggregate::COUNT
    } else {
        return None;
    };
    // `SUM` and `AVG` only over floats: over integers or DECIMAL they are exact
    // in SQL, and a rewrite that reorders or regroups the additions could
    // overflow where the original did not. `COUNT` is exact on any type.
    let arg_type = p.args[0].get_type(agg.input.schema()).ok()?;
    if aggregate != Aggregate::COUNT && arg_type != DataType::Float64 {
        return None;
    }

    let mut b = Builder::default();
    let cols = b.visit(&agg.input, None)?;
    let input_schema = agg.input.schema();
    let arg = inline(&p.args[0], input_schema, &cols)?;
    let mut group_slots = Vec::new();
    for g in &agg.group_expr {
        let Expr::Column(_) = g else { return None };
        group_slots.push(slot_of(&inline(g, input_schema, &cols)?)?);
    }
    b.finish(arg, aggregate, group_slots, agg.schema.clone())
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

    /// Build the fold from the walked tree, the aggregate and its argument, and
    /// the group keys' slots.
    fn finish(
        mut self,
        arg: Expr,
        aggregate: Aggregate,
        group_slots: Vec<usize>,
        schema: DFSchemaRef,
    ) -> Option<FoldMatch> {
        // Classify filter conjuncts. `x = y` and `x IS NOT DISTINCT FROM y`
        // between columns merge their dimension classes (two columns of one
        // leaf make a diagonal, like `m.i = m.j`). A predicate over one leaf
        // moves onto that leaf. Anything else, such as `a.t >= b.t`, relates
        // operands in a way a fold can't express yet, so decline.
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

        // Dimension classes: union-find over the column equalities, so that
        // `a.k = b.k AND b.k = c.k` makes one class of three columns. Each class
        // records how it compares keys. The two kinds differ only for NULL: `=`
        // drops rows with a NULL key, `IS NOT DISTINCT FROM` joins NULL to
        // NULL. A class that mixes them has no single meaning, so decline.
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

        let factors = self.factors(arg, aggregate)?;

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
        let fold = Fold::new(
            operands,
            output,
            equality,
            RowValue::Product(Op::Mul),
            aggregate,
        )
        .ok()?;
        let group_outputs = group_dims
            .into_iter()
            .enumerate()
            .map(|(i, d)| (Column::from(schema.qualified_field(i)), d))
            .collect();
        let value_output = Column::from(schema.qualified_field(group_slots.len()));
        Some(FoldMatch {
            fold,
            operands: inputs,
            group_outputs,
            value_output,
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

    /// Split the aggregate's argument into one factor per leaf. Returns `None`
    /// unless it is a product of factors that each read at most one leaf:
    /// `SUM(a.v * f(b.x, b.y))` splits, but `SUM(a.v + b.v)` and
    /// `SUM(exp(a.v * b.v))` don't. Any expression over one leaf's columns,
    /// such as `1 - b.z * b.z`, is one factor, since it is constant within
    /// that leaf's row. Factors of one leaf are multiplied together, and
    /// constant factors go to the first operand. Reordering a float product
    /// this way can change its last bits, but not its mathematical value.
    ///
    /// For `SUM` and `AVG`, every multiplication and factor must be `Float64`,
    /// so that splitting doesn't change how the product is computed. For
    /// `COUNT`, a product is NULL exactly when one of its factors is (and
    /// nothing else: NaN and overflow are not NULL), so any product splits,
    /// and each factor becomes `1.0` or NULL.
    fn factors(&self, arg: Expr, aggregate: Aggregate) -> Option<Vec<Option<Expr>>> {
        let schema = self.slot_schema()?;
        let count = aggregate == Aggregate::COUNT;
        let mut flat = Vec::new();
        flatten_product(arg, &schema, count, &mut flat)?;
        let mut factors: Vec<Option<Expr>> = vec![None; self.leaves.len()];
        let mut push = |i: usize, f: Expr| {
            factors[i] = Some(match factors[i].take() {
                Some(prev) => prev * f,
                None => f,
            });
        };
        let mut constants = Vec::new();
        for f in flat {
            let f = if count {
                // CASE WHEN f IS NOT NULL THEN 1.0 END
                when(
                    f.is_not_null(),
                    Expr::Literal(ScalarValue::Float64(Some(1.0)), None),
                )
                .end()
                .ok()?
            } else if f.get_type(&schema).ok()? == DataType::Float64 {
                f
            } else {
                return None;
            };
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

/// Flatten nested multiplications into their factors: all of them if `any`,
/// otherwise only those of two `Float64`s.
fn flatten_product(e: Expr, schema: &DFSchema, any: bool, out: &mut Vec<Expr>) -> Option<()> {
    let float = |e: &Expr| e.get_type(schema).ok() == Some(DataType::Float64);
    match e {
        Expr::BinaryExpr(BinaryExpr {
            left,
            op: Operator::Multiply,
            right,
        }) if any || (float(&left) && float(&right)) => {
            flatten_product(*left, schema, any, out)?;
            flatten_product(*right, schema, any, out)
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

/// Scalar functions that detection may move onto a leaf, when every argument
/// is `Float64`. Each returns NaN, ±inf or NULL rather than an error for any
/// input, which `tests/detect.rs` checks over NaN, ±inf, ±0, negatives and NULL.
/// `power` is left out: `power(0, -1)` raises an error in DataFusion.
pub const MOVABLE_FUNCTIONS: &[&str] = &["abs", "exp", "ln", "sqrt", "tanh", "signum"];

/// Whether detection can move `e`, an expression over `schema`: evaluate it on
/// a leaf's rows rather than on the joined rows. That is safe only for
/// deterministic, row-wise expressions that cannot raise an error: a leaf row
/// that never joins is never evaluated by the original plan, so if it made
/// `CAST('abc' AS DOUBLE)` or `1 / 0` fail, the rewrite would turn a working
/// query into an error.
fn movable(e: &Expr, schema: &DFSchema) -> bool {
    let ty = |e: &Expr| e.get_type(schema).ok();
    let arithmetic = |e: &Expr| ty(e).is_some_and(|t| t.is_integer() || t.is_floating());
    !e.exists(|e| {
        let ok = match e {
            Expr::Alias(_) | Expr::Column(_) | Expr::Literal(..) => true,
            // Integer arithmetic wraps in DataFusion; float arithmetic gives
            // inf or NaN. `/` and `%` are declined until proven infallible.
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => match op {
                Operator::Plus | Operator::Minus | Operator::Multiply => {
                    arithmetic(left) && arithmetic(right)
                }
                Operator::Eq
                | Operator::NotEq
                | Operator::Lt
                | Operator::LtEq
                | Operator::Gt
                | Operator::GtEq
                | Operator::And
                | Operator::Or
                | Operator::IsDistinctFrom
                | Operator::IsNotDistinctFrom => true,
                _ => false,
            },
            Expr::Negative(inner) => arithmetic(inner),
            Expr::Not(_)
            | Expr::IsNotNull(_)
            | Expr::IsNull(_)
            | Expr::IsTrue(_)
            | Expr::IsFalse(_)
            | Expr::IsUnknown(_)
            | Expr::IsNotTrue(_)
            | Expr::IsNotFalse(_)
            | Expr::IsNotUnknown(_)
            | Expr::Between(_)
            | Expr::Case(_)
            | Expr::TryCast(_) => true,
            Expr::InList(list) => list.list.iter().all(|e| matches!(e, Expr::Literal(..))),
            // Integers and floats convert to Float64 without error.
            Expr::Cast(cast) => {
                *cast.field.data_type() == DataType::Float64 && arithmetic(&cast.expr)
            }
            Expr::ScalarFunction(f) => {
                MOVABLE_FUNCTIONS.contains(&f.name())
                    && f.args.iter().all(|a| ty(a) == Some(DataType::Float64))
            }
            _ => false,
        };
        Ok(!ok)
    })
    .expect("infallible")
}

/// Rewrite `e`, an expression over `schema`, into one over slots, given each
/// column's expression `cols`. Aliases are dropped.
fn inline(e: &Expr, schema: &DFSchema, cols: &[Expr]) -> Option<Expr> {
    if !movable(e, schema) {
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
