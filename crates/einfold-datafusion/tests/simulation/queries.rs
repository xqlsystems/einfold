// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Random queries over `l`, `r` and `d` (see `data.rs`), spelled the many
//! ways SQL allows, so that detection meets the plans users and ddx write.
//!
//! Every query is built around a *fold*: `SUM` of a product of an expression
//! of `l` and one of `r`, over `l JOIN r`, grouped by columns. Around it go
//! the three shapes detection looks for, and look-alikes of each:
//!
//! - the fold alone, with its join written `JOIN … ON`, as a comma join
//!   with the condition in `WHERE`, with its sides swapped, through
//!   subqueries and CTEs, with filters, `HAVING`, and `=` or `IS NOT DISTINCT
//!   FROM`; its factors are expressions, some of which can fail (a cast of
//!   text, an integer division) on rows the join drops;
//! - a left join from `d` onto the fold (a *domain*), with `COALESCE` or ddx's
//!   `CASE`, or reading the fold's keys, or another kind of join;
//! - the fold joined back to `d`, grouped again by `d`'s keys, and left-joined
//!   from `d`, as ddx writes each gradient step, with the inner `SUM`'s
//!   argument the fold's value or `CASE WHEN <test of d's row> THEN NULL ELSE
//!   <value> END`, and `d` with or without duplicate keys;
//!
//! - a masked fold: a condition comparing two of the fold's outputs, or a
//!   dimension it sums away, alone or left-joined from `d`;
//!
//! and compositions: two folds under `UNION ALL`, a fold of a fold's result,
//! and `ORDER BY` with `LIMIT` above.

use einfold_testkit::Rng;

/// Which shape a query was built as, for the tallies.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Shape {
    Fold,
    FromDomain,
    JoinedBack,
    Union,
    Nested,
    /// A fold with a mask: a condition comparing two of its columns.
    Masked,
}

/// How the fold's join compares keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Eq {
    Equal,
    NotDistinct,
}

impl Eq {
    fn draw(rng: &mut Rng) -> Self {
        if rng.chance(0.5) {
            Eq::Equal
        } else {
            Eq::NotDistinct
        }
    }

    fn render(self, x: &str, y: &str) -> String {
        match self {
            Eq::Equal => format!("{x} = {y}"),
            Eq::NotDistinct => format!("({x} IS NOT DISTINCT FROM {y})"),
        }
    }
}

/// What the generator may use, decided per case.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Whether factor expressions may fail on some rows.
    pub fallible: bool,
    /// Whether values are exact (see `data::Values`): only then may factors
    /// add constants, which change the magnitudes the bound is computed from.
    pub exact: bool,
    /// Whether the query may be ordered and limited above.
    pub limit: bool,
}

/// An expression of one side's row, as a factor.
fn factor(rng: &mut Rng, t: &str, options: Options) -> String {
    let mut menu: Vec<String> = vec![
        format!("{t}.v"),
        format!("{t}.v"),
        format!("({t}.v * 2.0)"),
        format!("(-{t}.v)"),
        format!("abs({t}.v)"),
    ];
    if options.exact {
        menu.extend([
            format!("({t}.v + 1.0)"),
            format!("COALESCE({t}.v, 0.5)"),
            format!("CASE WHEN {t}.v > 0 THEN {t}.v ELSE 0.5 END"),
            // A factor from a key: NULL keys make NULL factors.
            format!("CAST({t}.a IS NULL AS DOUBLE)"),
        ]);
    }
    if options.fallible {
        menu.extend([
            // Fails on text that isn't a number.
            format!("CAST({t}.t AS DOUBLE)"),
            // Fails where `v` is 0 as an integer, or NaN or infinite.
            format!("CAST(CAST(6 AS BIGINT) / CAST({t}.v AS BIGINT) AS DOUBLE)"),
        ]);
    }
    rng.pick(&menu).clone()
}

/// A source for one side of the fold, aliased `alias`.
fn source(rng: &mut Rng, table: &str, alias: &str, ctes: &mut Vec<String>) -> String {
    match rng.below(8) {
        0 => format!("(SELECT * FROM {table}) {alias}"),
        1 => format!("(SELECT {table}.a, {table}.b, {table}.v, {table}.t FROM {table} WHERE {table}.v IS NOT NULL) {alias}"),
        2 => format!("(SELECT * FROM {table} WHERE {table}.a IS NOT NULL) {alias}"),
        3 => {
            let name = format!("cte_{table}_{}", ctes.len());
            ctes.push(format!("{name} AS (SELECT a, b, v, t FROM {table})"));
            format!("{name} {alias}")
        }
        _ => format!("{table} {alias}"),
    }
}

/// The fold's parts, rendered by [`Fold::sql`].
struct Fold {
    /// `l`'s source and `r`'s, with their aliases `l` and `r`.
    sources: [String; 2],
    swap: bool,
    comma: bool,
    on: Vec<String>,
    product: String,
    /// The select list's group keys, in their own order, and the GROUP BY.
    select: Vec<String>,
    group: Vec<String>,
    filter: Option<String>,
    having: Option<String>,
    /// Whether the aggregate is something other than a plain `SUM` of the
    /// product, which detection must leave alone.
    look_alike: bool,
    /// A condition added to the join's, written as `{MASK}` in the SQL, so
    /// that it can be replaced (see [`Mask`]).
    mask: bool,
}

impl Fold {
    /// A fold whose output columns are `g0` (from `l.a`) and `g1` (from
    /// `r.b`), as shapes 2 and 3 need, when `keyed`; any group keys otherwise.
    fn draw(rng: &mut Rng, options: Options, keyed: bool, ctes: &mut Vec<String>) -> Self {
        let eq = Eq::draw(rng);
        let mut on = vec![eq.render("l.b", "r.a")];
        // Sometimes a second key; and rarely the two equalities mixed, which
        // DataFusion keeps as a join filter.
        if !keyed && rng.chance(0.15) {
            let second = if rng.chance(0.2) { Eq::draw(rng) } else { eq };
            on.push(second.render("l.a", "r.b"));
        }
        let (mut fl, mut fr) = (factor(rng, "l", options), factor(rng, "r", options));
        if rng.chance(0.3) {
            std::mem::swap(&mut fl, &mut fr);
        }
        let mut look_alike = false;
        let product = match rng.below(20) {
            0 => format!("(2.0 * {fl}) * {fr}"),
            1 => format!("{fl} * ({fr} * 2.0)"),
            2 if !keyed => {
                look_alike = true;
                format!("{fl} * {fr} * 2.0")
            }
            // A sum, whose terms' magnitudes the bound's twin tables can't
            // give, so only with exact values.
            3 if !keyed && options.exact => {
                look_alike = true;
                format!("{fl} + {fr}")
            }
            _ => format!("{fl} * {fr}"),
        };
        let (select, group) = if keyed {
            let mut select = vec!["l.a AS g0".to_string(), "r.b AS g1".to_string()];
            if rng.chance(0.5) {
                select.reverse();
            }
            let mut group = vec!["l.a".to_string(), "r.b".to_string()];
            if rng.chance(0.5) {
                group.reverse();
            }
            (select, group)
        } else {
            let pool = ["l.a", "r.b", "l.b", "r.a"];
            let mut keys: Vec<&str> = pool.iter().copied().filter(|_| rng.chance(0.5)).collect();
            if keys.is_empty() && rng.chance(0.9) {
                keys.push("l.a");
            }
            rng.shuffle(&mut keys);
            let select = (keys.iter().enumerate())
                .map(|(i, k)| format!("{k} AS g{i}"))
                .collect();
            (select, keys.iter().map(|k| k.to_string()).collect())
        };
        if group.is_empty() {
            look_alike = true;
        }
        let filter = match rng.below(10) {
            0 => Some("l.a IS NOT NULL".to_string()),
            1 => Some("r.v IS NOT NULL".to_string()),
            2 if !keyed => {
                // A condition across both sides: a mask, left alone.
                look_alike = true;
                Some("l.a >= r.b".to_string())
            }
            _ => None,
        };
        let having = (!keyed && rng.chance(0.1)).then(|| "COUNT(*) > 0".to_string());
        if having.is_some() {
            // HAVING's COUNT is a second aggregate.
            look_alike = true;
        }
        let swap = rng.chance(0.3);
        let comma = rng.chance(0.15);
        Fold {
            sources: [source(rng, "l", "l", ctes), source(rng, "r", "r", ctes)],
            swap,
            comma,
            on,
            product,
            select,
            group,
            filter,
            having,
            look_alike,
            mask: false,
        }
    }

    fn sql(&self) -> String {
        let [l, r] = &self.sources;
        let (x, y) = if self.swap { (r, l) } else { (l, r) };
        let mut conditions: Vec<String> = Vec::new();
        let from = if self.comma {
            conditions.extend(self.on.iter().cloned());
            format!("{x}, {y}")
        } else {
            format!("{x} JOIN {y} ON {}", self.on.join(" AND "))
        };
        conditions.extend(self.filter.iter().cloned());
        if self.mask {
            conditions.push("{MASK}".to_string());
        }
        let mut s = format!(
            "SELECT {}SUM({}) AS v FROM {from}",
            self.select
                .iter()
                .map(|c| format!("{c}, "))
                .collect::<String>(),
            self.product
        );
        if !conditions.is_empty() {
            s += &format!(" WHERE {}", conditions.join(" AND "));
        }
        if !self.group.is_empty() {
            s += &format!(" GROUP BY {}", self.group.join(", "));
        }
        if let Some(h) = &self.having {
            s += &format!(" HAVING {h}");
        }
        s
    }
}

/// A mask's condition and the query it was put in, written with `{MASK}`
/// where the condition goes, so that a test can put other conditions there:
/// its negation, or whether it is NULL.
#[derive(Clone, Debug)]
pub struct Mask {
    pub template: String,
    pub condition: String,
    /// Whether the query is the masked fold alone, and its condition
    /// compares two of its outputs: then the condition, its negation and its
    /// NULL case divide the fold's groups among them, each whole.
    pub partitions: bool,
}

/// A random query and the shape it was built as.
#[derive(Clone)]
pub struct Query {
    pub sql: String,
    pub shape: Shape,
    /// Whether the generator built something detection must leave alone at
    /// its outermost fold. Only for tallies: detection may decline more.
    pub look_alike: bool,
    /// Whether some join has a condition beyond equalities, as a mask's
    /// `l.a >= r.b`: DataFusion keeps it as the join's filter.
    pub join_filter: bool,
    /// For a masked fold, its mask.
    pub mask: Option<Mask>,
}

pub fn query(rng: &mut Rng, options: Options) -> Query {
    let mut ctes = Vec::new();
    let shape = match rng.below(26) {
        0..=6 => Shape::Fold,
        7..=10 => Shape::FromDomain,
        11..=16 => Shape::JoinedBack,
        17..=18 => Shape::Union,
        19 => Shape::Nested,
        _ => Shape::Masked,
    };
    let mut mask = None;
    let (body, look_alike) = match shape {
        Shape::Fold => {
            let f = Fold::draw(rng, options, false, &mut ctes);
            (f.sql(), f.look_alike)
        }
        Shape::FromDomain => from_domain(rng, options, &mut ctes),
        Shape::JoinedBack => joined_back(rng, options, &mut ctes),
        Shape::Union => {
            let f = Fold::draw(rng, options, true, &mut ctes);
            let g = Fold::draw(rng, options, true, &mut ctes);
            // UNION ALL matches columns by position, and keyed folds may put
            // g0 and g1 in either order.
            (
                format!(
                    "SELECT g0, g1, v FROM ({}) u0 UNION ALL SELECT g0, g1, v FROM ({}) u1",
                    f.sql(),
                    g.sql()
                ),
                false,
            )
        }
        Shape::Masked => {
            let (template, condition, partitions) = masked(rng, options, &mut ctes);
            mask = Some(Mask {
                template: template.clone(),
                condition: condition.clone(),
                partitions,
            });
            (template.replace("{MASK}", &condition), false)
        }
        Shape::Nested => {
            // A fold whose first operand is a fold's result: g1 has K's
            // type, as d.b does.
            let f = Fold::draw(rng, options, true, &mut ctes);
            let eq = Eq::draw(rng);
            (
                format!(
                    "SELECT f.g0 AS g0, d.a AS h, SUM(f.v * d.v) AS v FROM ({}) f JOIN d ON {} \
                     GROUP BY f.g0, d.a",
                    f.sql(),
                    eq.render("f.g1", "d.b")
                ),
                false,
            )
        }
    };
    let prefix = match ctes.is_empty() {
        true => String::new(),
        false => format!("WITH {} ", ctes.join(", ")),
    };
    let with_ctes = |body: &str| format!("{prefix}{body}");
    let mut sql = with_ctes(&body);
    // Sometimes ordered and limited above: the order is on every column, so
    // with exact values the rows kept don't depend on the plan.
    let limited = !matches!(shape, Shape::Fold | Shape::Masked);
    if options.limit && limited && rng.chance(0.08) {
        sql = format!("SELECT * FROM ({sql}) q ORDER BY 1, 2, 3 LIMIT 5");
    }
    // DataFusion keeps as a join filter the mask (`l.a >= r.b`, the only
    // inequality the generator writes) and a second key compared otherwise
    // than the first.
    let mixed = (sql.contains("l.b = r.a") && sql.contains("(l.a IS NOT DISTINCT FROM r.b)"))
        || (sql.contains("(l.b IS NOT DISTINCT FROM r.a)") && sql.contains("l.a = r.b"));
    let join_filter = sql.contains(" >= ") || mixed || mask.is_some();
    Query {
        sql,
        shape,
        look_alike,
        join_filter,
        mask: mask.map(|m| Mask {
            template: with_ctes(&m.template),
            ..m
        }),
    }
}

/// `d LEFT JOIN (fold) s`, and its look-alikes.
fn from_domain(rng: &mut Rng, options: Options, ctes: &mut Vec<String>) -> (String, bool) {
    let f = Fold::draw(rng, options, true, ctes);
    let eq = Eq::draw(rng);
    let mut on = vec![eq.render("d.a", "s.g0"), eq.render("d.b", "s.g1")];
    if rng.chance(0.5) {
        on.reverse();
    }
    let value = rng
        .pick(&[
            "s.v",
            "COALESCE(s.v, 0.0)",
            "CASE WHEN d.v IS NULL THEN NULL WHEN s.v IS NULL THEN 0.0 ELSE s.v END",
            "(s.v * 2.0)",
        ])
        .to_string();
    let (reads, join, look_alike) = match rng.below(10) {
        0 => ("s.g0", "LEFT JOIN", true),
        1 => ("d.b", "JOIN", true),
        2 => ("d.b", "RIGHT JOIN", true),
        3 => {
            // One key only: not every output is matched.
            on.truncate(1);
            ("d.b", "LEFT JOIN", true)
        }
        _ => ("d.b", "LEFT JOIN", false),
    };
    (
        format!(
            "SELECT d.a, {reads}, {value} AS v FROM d {join} ({}) s ON {}",
            f.sql(),
            on.join(" AND ")
        ),
        look_alike,
    )
}

/// ddx's gradient step: the fold joined back to `d`, grouped again by `d`'s
/// keys, and left-joined from `d`; and its look-alikes.
fn joined_back(rng: &mut Rng, options: Options, ctes: &mut Vec<String>) -> (String, bool) {
    let f = Fold::draw(rng, options, true, ctes);
    let (inner, outer) = (Eq::draw(rng), Eq::draw(rng));
    let mut look_alike = false;
    // What the regroup sums: the fold's value, or NULL where a test of d2's
    // row holds. The tests read columns other than the keys, so duplicate
    // rows of d can disagree.
    let mut tests = vec![
        "f.v",
        "CASE WHEN d2.v IS NULL THEN NULL ELSE f.v END",
        "CASE WHEN d2.v IS NULL THEN NULL ELSE f.v END",
        "CASE WHEN d2.c = 'x' THEN NULL ELSE f.v END",
    ];
    // A test of the value's sign changes when the bound's twin tables take
    // magnitudes, so only exact cases use it.
    if options.exact {
        tests.push("CASE WHEN d2.v < 0 THEN NULL ELSE f.v END");
    }
    let e = rng.pick(&tests).to_string();
    let d2 = match rng.below(8) {
        0 => "(SELECT * FROM d) d2",
        1 => "(SELECT a, b, v, c FROM d) d2",
        2 => {
            look_alike = true;
            "(SELECT * FROM d WHERE d.v IS NOT NULL) d2"
        }
        _ => "d d2",
    };
    let (group, keys) = match rng.below(10) {
        // Grouping by a value changes what the bound's twin tables compute,
        // so only with exact values.
        0 if options.exact => {
            look_alike = true;
            ("d2.a, d2.v", "d2.a AS a, d2.v AS b")
        }
        1 => ("d2.b, d2.a", "d2.a AS a, d2.b AS b"),
        _ => ("d2.a, d2.b", "d2.a AS a, d2.b AS b"),
    };
    let regroup = format!(
        "SELECT {keys}, SUM({e}) AS v FROM {d2} JOIN ({}) f ON {} AND {} GROUP BY {group}",
        f.sql(),
        inner.render("d2.a", "f.g0"),
        inner.render("d2.b", "f.g1"),
    );
    // The outer projection: ddx's masks d's NULL rows, which hides what the
    // regroup did to them; the others don't.
    let value = rng
        .pick(&[
            "s.v",
            "COALESCE(s.v, 0.0)",
            "CASE WHEN d.v IS NULL THEN NULL WHEN s.v IS NULL THEN 0.0 ELSE s.v END",
        ])
        .to_string();
    let reads = if rng.chance(0.1) {
        look_alike = true;
        "s.b"
    } else {
        "d.b"
    };
    (
        format!(
            "SELECT d.a, {reads}, {value} AS v FROM d LEFT JOIN ({regroup}) s ON {} AND {}",
            outer.render("d.a", "s.a"),
            outer.render("d.b", "s.b"),
        ),
        look_alike,
    )
}

/// A masked fold: a keyed fold with `{MASK}` among its conditions, the mask's
/// condition, and whether the query is the fold alone. The condition usually
/// compares the fold's two outputs, which decides each group as a whole;
/// sometimes it reads a dimension the fold sums away (`l.b`), which masks
/// pairs within a group, and detection must leave it alone or get it right.
/// Half the time the fold is left-joined from `d`, as its domain.
fn masked(rng: &mut Rng, options: Options, ctes: &mut Vec<String>) -> (String, String, bool) {
    let mut f = Fold::draw(rng, options, true, ctes);
    f.mask = true;
    let op = *rng.pick(&["<", "<=", ">", ">=", "<>", "="]);
    let (x, y) = match rng.below(6) {
        0 => ("l.b", "r.b"),
        1 => ("r.b", "l.a"),
        _ => ("l.a", "r.b"),
    };
    let condition = format!("({x} {op} {y})");
    if rng.chance(0.5) {
        return (f.sql(), condition, x != "l.b");
    }
    let eq = Eq::draw(rng);
    let template = format!(
        "SELECT d.a, d.b, COALESCE(s.v, 0.0) AS v FROM d LEFT JOIN ({}) s ON {} AND {}",
        f.sql(),
        eq.render("d.a", "s.g0"),
        eq.render("d.b", "s.g1")
    );
    (template, condition, false)
}
