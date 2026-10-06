# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike S10: do hosts keep a contraction order written in SQL?

Reproduces Blacher et al.'s satisfiability example (SIGMOD 2023, §4.2):
counting the solutions of a 3-SAT formula is an einsum with one tensor per
clause and one index per variable. Each clause becomes a table of its 7
satisfying assignments (the 8th, all literals false, is the zero left out).

Three SQL forms of the same einsum, on DuckDB and DataFusion:

- flat:        one query joining every clause table, SUM and no GROUP BY;
                the host's join optimizer chooses the order.
- cte:         Blacher's decomposition: one CTE per pairwise contraction,
                in opt_einsum's order, each a join with GROUP BY and SUM.
- materialized (DuckDB only): the same, with `AS MATERIALIZED` CTEs.

For each: planning time (EXPLAIN), run time (with a time limit), the result
(checked against NumPy's einsum), and whether the physical plan still has
one aggregation per contraction step, in the written nesting.
"""

import multiprocessing as mp
import random
import sys
import time

import duckdb
import numpy as np
import opt_einsum as oe
import pyarrow as pa
from datafusion import SessionContext

LIMIT_S = 180


BAND = 8


def random_3sat(n_vars, n_clauses, seed):
    """Random 3-SAT whose clauses use variables within a window of BAND.

    Uniform random 3-SAT has treewidth that grows with size (50 variables
    already needs a 2^33-entry intermediate), so no engine could contract it.
    A banded formula keeps the treewidth small at any size, as structured
    instances like Blacher et al.'s 718-clause example must.
    """
    rng = random.Random(seed)
    clauses = []
    for _ in range(n_clauses):
        start = rng.randrange(n_vars - BAND)
        vs = rng.sample(range(start, start + BAND), 3)
        clauses.append([(v, rng.random() < 0.5) for v in vs])  # (variable, negated)
    return clauses


def clause_rows(clause):
    """The 7 satisfying assignments of a clause, as columns x<var>."""
    rows = []
    for bits in range(8):
        assign = [(bits >> k) & 1 for k in range(3)]
        sat = any((a == 0) if neg else (a == 1) for a, (_, neg) in zip(assign, clause))
        if sat:
            rows.append(assign)
    cols = {f"x{v}": [r[k] for r in rows] for k, (v, _) in enumerate(clause)}
    cols = {k: pa.array(v, pa.int64()) for k, v in cols.items()}
    # DOUBLE: model counts of large formulas exceed 64-bit integers.
    cols["val"] = pa.array([1.0] * len(rows), pa.float64())
    return pa.table(cols)


def numpy_count(clauses, n_vars):
    syms = [oe.get_symbol(v) for v in range(n_vars)]
    ops, terms = [], []
    for c in clauses:
        t = np.zeros((2, 2, 2), dtype=np.float64)
        for r in clause_rows(c).to_pylist():
            t[tuple(r[f"x{v}"] for v, _ in c)] = 1
        ops.append(t)
        terms.append("".join(syms[v] for v, _ in c))
    expr = ",".join(terms) + "->"
    path, info = oe.contract_path(expr, *ops, optimize="greedy")
    return float(oe.contract(expr, *ops, optimize=path)), path, info


def flat_sql(clauses):
    tabs = [f"c{k}" for k in range(len(clauses))]
    first = {}
    conds = []
    for k, c in enumerate(clauses):
        for v, _ in c:
            if v in first:
                conds.append(f"{first[v]}.x{v} = c{k}.x{v}")
            else:
                first[v] = f"c{k}"
    prod = " * ".join(f"{t}.val" for t in tabs)
    where = " AND ".join(conds) or "TRUE"
    return f"SELECT SUM({prod}) AS val FROM {', '.join(tabs)} WHERE {where}"


def cte_sql(clauses, path, materialized=False):
    """Blacher's rule: each pairwise contraction is a CTE with GROUP BY."""
    live = [(f"c{k}", {v for v, _ in c}) for k, c in enumerate(clauses)]
    ctes = []
    for step, pair in enumerate(path):
        a, b = sorted(pair, reverse=True)
        (na, va), (nb, vb) = live.pop(a), live.pop(b)
        rest = set().union(*(vs for _, vs in live)) if live else set()
        keep = sorted((va | vb) & rest)
        shared = sorted(va & vb)
        sel = [f"{na}.x{v}" if v in va else f"{nb}.x{v}" for v in keep]
        cols = ", ".join(f"{s} AS x{v}" for s, v in zip(sel, keep))
        where = " AND ".join(f"{na}.x{v} = {nb}.x{v}" for v in shared) or "TRUE"
        group = f" GROUP BY {', '.join(sel)}" if keep else ""
        name = f"t{step}"
        mat = " MATERIALIZED" if materialized else ""
        select_cols = (cols + ", ") if cols else ""
        ctes.append(f"{name} AS{mat} (SELECT {select_cols}SUM({na}.val * {nb}.val) AS val "
                    f"FROM {na}, {nb} WHERE {where}{group})")
        live.append((name, set(keep)))
    final = live[0][0]
    return f"WITH {', '.join(ctes)} SELECT SUM(val) AS val FROM {final}"


def run_form(host, form, n_vars, n_clauses, seed, out):
    """Plan and run one form on one host, in a child process; report through `out`."""
    clauses = random_3sat(n_vars, n_clauses, seed)
    _, path, _ = numpy_count(clauses, n_vars)
    sql = {"flat": flat_sql(clauses), "cte": cte_sql(clauses, path),
           "materialized": cte_sql(clauses, path, materialized=True)}[form]
    tables = {f"c{k}": clause_rows(c) for k, c in enumerate(clauses)}
    if host == "DuckDB":
        con = duckdb.connect()
        # A flat join over hundreds of tables can spill without bound: a first run
        # wrote 16 GB of temporary files before the system stopped it.
        con.execute("SET memory_limit = '4GB'")
        con.execute("SET max_temp_directory_size = '2GB'")
        for name, t in tables.items():
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM t")
        explain = lambda: con.execute("EXPLAIN " + sql).fetchall()  # noqa: E731
        aggs = lambda: count_aggs_duckdb(con, sql)  # noqa: E731
        run = lambda: con.execute(sql).fetchone()[0]  # noqa: E731
    else:
        ctx = SessionContext()
        for name, t in tables.items():
            ctx.register_record_batches(name, [t.to_batches()])
        explain = lambda: ctx.sql("EXPLAIN " + sql).collect()  # noqa: E731
        aggs = lambda: count_aggs_datafusion(ctx, sql)  # noqa: E731
        run = lambda: ctx.sql(sql).to_pydict()["val"][0]  # noqa: E731
    t0 = time.perf_counter()
    explain()
    out.put(("plan", time.perf_counter() - t0))
    out.put(("aggs", aggs()))
    t0 = time.perf_counter()
    val = run()
    out.put(("run", time.perf_counter() - t0, float(val)))


def measure(host, form, n_vars, n_clauses, seed, limit=LIMIT_S):
    """Run `run_form` in a process; kill it if planning plus running exceeds `limit`."""
    q = mp.Queue()
    p = mp.Process(target=run_form, args=(host, form, n_vars, n_clauses, seed, q))
    p.start()
    p.join(limit)
    if p.is_alive():
        p.kill()
        p.join()
    got = {}
    while not q.empty():
        item = q.get()
        got[item[0]] = item[1:]
    plan = f"{got['plan'][0]:.3f} s" if "plan" in got else f"> {limit} s"
    aggs = got["aggs"][0] if "aggs" in got else "—"
    run = f"{got['run'][0]:.3f} s" if "run" in got else (f"> {limit} s" if p.exitcode in (None, -9) else f"failed ({p.exitcode})")
    val = got["run"][1] if "run" in got else None
    return plan, aggs, run, val


def count_aggs_duckdb(con, sql):
    # The JSON form, because the text form elides wide plans.
    plan = con.execute("EXPLAIN (FORMAT json) " + sql).fetchall()[0][1]
    return sum(plan.count(f'"name": "{op}"') for op in
               ("HASH_GROUP_BY", "PERFECT_HASH_GROUP_BY", "UNGROUPED_AGGREGATE"))


def count_aggs_datafusion(ctx, sql):
    rows = ctx.sql("EXPLAIN " + sql).to_pydict()
    plan = [p for t, p in zip(rows["plan_type"], rows["plan"]) if t == "physical_plan"][0]
    # Count final-mode aggregates only (each logical aggregate may split into partial + final).
    return sum(1 for line in plan.splitlines() if "AggregateExec" in line and "mode=Partial" not in line)


def main(n_vars, n_clauses, seed=0):
    clauses = random_3sat(n_vars, n_clauses, seed)
    expected, path, info = numpy_count(clauses, n_vars)
    print(f"\n## {n_vars} variables, {n_clauses} clauses (seed {seed}): "
          f"{expected:.6e} solutions; opt_einsum greedy: largest intermediate {int(info.largest_intermediate)} entries")
    steps = len(path)
    print(f"contraction steps: {steps}")
    print("| Host | Form | Plan (EXPLAIN) | Aggregations in plan | Run | Result correct |")
    print("|---|---|---|---|---|---|")
    for host, forms in (("DuckDB", ("flat", "cte", "materialized")), ("DataFusion", ("flat", "cte"))):
        for form in forms:
            plan, aggs, run, val = measure(host, form, n_vars, n_clauses, seed)
            ok = "—" if val is None else ("yes" if abs(val - expected) <= 1e-9 * abs(expected) else f"no ({val:.6e})")
            print(f"| {host} | {form} | {plan} | {aggs} of {steps + 1} | {run} | {ok} |")
            sys.stdout.flush()


def cte_planning_sweep():
    """How DuckDB's planning time for inlined CTEs grows with the number of steps."""
    print("\n## DuckDB: planning time of the CTE form, inlined vs. materialized, 30 variables")
    print("| Clauses | Inlined (EXPLAIN) | Materialized (EXPLAIN) |")
    print("|---|---|---|")
    for m in (15, 30, 45, 60, 75, 91):
        inl = measure("DuckDB", "cte", 30, m, 0)[0]
        mat = measure("DuckDB", "materialized", 30, m, 0)[0]
        print(f"| {m} | {inl} | {mat} |")
        sys.stdout.flush()


if __name__ == "__main__":
    for n, m in [(30, 91), (73, 218), (143, 430), (250, 718)]:
        main(n, m)
    cte_planning_sweep()
