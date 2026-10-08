# SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Spike S23: when does eager aggregation change a float result, and what does a guard cost?

Eager aggregation sums a table over a dimension no later step needs, before the
join: `SUM(a.v * b.v) ... GROUP BY g` becomes `SUM(a2.v * b.v)` over
`a2 = SELECT k, SUM(v) FROM a GROUP BY k`. Over the reals the two are equal,
because multiplication distributes over addition. Over IEEE doubles they can
differ in finiteness, not just in the last bits. This spike:

1. runs counterexamples on DuckDB and DataFusion, both plans written as SQL;
2. checks a guard, by property test: if every input is finite, every eager
   partial sum's absolute sum stays below the overflow threshold, and every
   output group's sum of |a·b| does too, then both plans are finite and each
   is within the usual summation bound of the exact result;
3. measures what checking that guard costs: from Parquet statistics, and as a
   run-time pass next to the contraction itself.

Run with:
    uv run --with duckdb --with datafusion --with numpy --with pyarrow python float_guards.py
"""

import math
import random
import tempfile
import time
from fractions import Fraction

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from datafusion import SessionConfig, SessionContext

MAX = 1.7976931348623157e308
U = 2.0**-53  # unit roundoff of DOUBLE

# --- 1. counterexamples on the hosts -------------------------------------------

ORIGINAL = "SELECT SUM(a.v * b.v) AS s FROM A a JOIN B b ON a.k = b.k"
EAGER = (
    "WITH a2 AS (SELECT k, SUM(v) AS v FROM A GROUP BY k) "
    "SELECT SUM(a2.v * b.v) AS s FROM a2 JOIN B b ON a2.k = b.k"
)

CASES = [
    # (name, A's v values, all with k = 1 and distinct j; B's single v at k = 1)
    ("partial sum overflows, B is 0", [1e308, 1e308], 0.0),
    ("products overflow, partial sum cancels", [1e200, -1e200], 1e200),
    ("partial sum overflows, then shrinks", [1e308, 1e308, -1e308], 0.5),
]


def tables(av, bv):
    a = pa.table({"j": list(range(len(av))), "k": [1] * len(av), "v": av})
    b = pa.table({"k": [1], "v": [bv]})
    return a, b


def duck(a, b, sql):
    con = duckdb.connect()
    con.execute("SET threads = 1")
    con.register("A", a)
    con.register("B", b)
    return con.sql(sql).fetchall()[0][0]


def fusion(a, b, sql, partitions=1):
    ctx = SessionContext(SessionConfig().with_target_partitions(partitions))
    ctx.register_record_batches("A", [a.to_batches()])
    ctx.register_record_batches("B", [b.to_batches()])
    return ctx.sql(sql).to_pylist()[0]["s"]


def exact(av, bv):
    return float(sum(Fraction(x) * Fraction(bv) for x in av))


print("## 1. Counterexamples\n")
print(f"DuckDB {duckdb.__version__}, single thread; DataFusion, one partition.\n")
print("| case | exact | DuckDB original | DuckDB eager | DataFusion original | DataFusion eager |")
print("|---|---|---|---|---|---|")
for name, av, bv in CASES:
    a, b = tables(av, bv)
    row = [exact(av, bv), duck(a, b, ORIGINAL), duck(a, b, EAGER), fusion(a, b, ORIGINAL), fusion(a, b, EAGER)]
    print(f"| {name} | " + " | ".join(f"{x:g}" for x in row) + " |")

# The host alone: the same four values summed in different partitionings.
x = 1e308
print("\nOne host, one plan, different partitionings: `SUM(v)` over {x, x, −x, −x}, x = 1e308.\n")
print("| batches (one per partition) | DataFusion `SUM(v)` |")
print("|---|---|")
for parts in ([[x, -x, x, -x]], [[x, x], [-x, -x]], [[x, x, -x, -x]], [[-x, -x, x, x]]):
    ctx = SessionContext(SessionConfig().with_target_partitions(len(parts)))
    batches = [pa.record_batch({"v": p}) for p in parts]
    ctx.register_record_batches("t", [[bt] for bt in batches])
    s = ctx.sql("SELECT SUM(v) AS s FROM t").to_pylist()[0]["s"]
    shown = " · ".join("[" + ", ".join("x" if y > 0 else "−x" for y in p) + "]" for p in parts)
    print(f"| {shown} | {s:g} |")

# Algebraic decomposition of VAR: Σx² − (Σx)²/n cancels catastrophically.
rng = np.random.default_rng(0)
xs = 1e9 + rng.random(100_000)
t = pa.table({"x": xs})
var_sql = (
    "SELECT VAR_POP(x) AS engine, "
    "(SUM(x * x) - SUM(x) * SUM(x) / COUNT(x)) / COUNT(x) AS decomposed FROM t"
)
con = duckdb.connect()
con.register("t", t)
d_engine, d_dec = con.sql(var_sql).fetchall()[0]
ctx = SessionContext()
ctx.register_record_batches("t", [t.to_batches()])
f = ctx.sql(var_sql).to_pylist()[0]
true_var = float(np.var(xs - 1e9))
print(f"\n`VAR_POP` of 100,000 values 1e9 + U[0, 1) (true value {true_var:.6f}):\n")
print("| host | engine's `VAR_POP` | Σx² − (Σx)²/n |\n|---|---|---|")
print(f"| DuckDB | {d_engine:.6f} | {d_dec:.6g} |")
print(f"| DataFusion | {f['engine']:.6f} | {f['decomposed']:.6g} |")

# The online-softmax merge, whose identity is (−∞, 0): merging two empty states.
softmax_sql = (
    "SELECT greatest(m1, m2) AS m, d1 * exp(m1 - greatest(m1, m2)) + d2 * exp(m2 - greatest(m1, m2)) AS d "
    "FROM (SELECT CAST('-inf' AS DOUBLE) AS m1, 0.0 AS d1, CAST('-inf' AS DOUBLE) AS m2, 0.0 AS d2)"
)
dm, dd = duckdb.sql(softmax_sql).fetchall()[0]
fr = SessionContext().sql(softmax_sql).to_pylist()[0]
print("\nOnline-softmax merge of two empty states (−∞, 0) ⊕ (−∞, 0), whose result should be (−∞, 0):\n")
print(f"| host | m | d |\n|---|---|---|\n| DuckDB | {dm} | {dd} |\n| DataFusion | {fr['m']} | {fr['d']} |")

# --- 2. the guard, by property test ----------------------------------------------


def seq_sum(xs):
    s = 0.0
    for v in xs:
        s += v
    return s


def magnitude(r, lo, hi):
    """Log-uniform magnitudes from 10^lo to 10^hi, random sign, some exact zeros."""
    if r.random() < 0.1:
        return 0.0
    return r.choice((-1.0, 1.0)) * 10.0 ** r.uniform(lo, hi)


def guard(groups):
    """groups: {out: [(sub, a, b)]}; the eager plan sums a within (out, sub) first."""
    for terms in groups.values():
        if sum(abs(a) * abs(b) for _, a, b in terms) * (1 + 4 * len(terms) * U) >= MAX:
            return False
        subs = {}
        for s, a, _ in terms:
            subs[s] = subs.get(s, 0.0) + abs(a)
        if any(v * (1 + 4 * len(terms) * U) >= MAX for v in subs.values()):
            return False
    return True


def within(plan, terms, ex, n):
    # The bound for n additions in any order, plus one multiplication per term,
    # plus an absolute allowance for gradual underflow.
    if not math.isfinite(plan):
        return False
    absum = sum(abs(Fraction(a) * Fraction(b)) for _, a, b in terms)
    u = Fraction(U)
    gamma = (n + 2) * u / (1 - (n + 2) * u)
    return abs(Fraction(plan) - ex) <= gamma * absum + 4 * (n + 2) * Fraction(2) ** -1074


def trial(r, a_range, b_range):
    """One output group: terms a_i · b_sub(i), with a summed per sub-group first by the eager plan."""
    nsub = r.randint(1, 4)
    b = [magnitude(r, *b_range) for _ in range(nsub)]
    terms = []
    for _ in range(r.randint(1, 8)):
        s = r.randrange(nsub)
        terms.append((s, magnitude(r, *a_range), b[s]))
    r.shuffle(terms)
    original = seq_sum(a * bb for _, a, bb in terms)
    subs = {}
    for s, a, _ in terms:
        subs.setdefault(s, []).append(a)
    eager = seq_sum(seq_sum(v) * b[s] for s, v in subs.items())
    ex = sum(Fraction(a) * Fraction(bb) for _, a, bb in terms)
    return terms, original, eager, ex


print("\n## 2. The guard, by property test\n")
print("Random output groups of 1–8 terms in 1–4 sub-groups, signs and exact zeros random.\n")
print("| magnitudes of a, b | groups | guard held | of those, either plan non-finite or outside the bound "
      "| guard failed | of those: finiteness differs | both finite, eager outside the bound | both non-finite |")
print("|---|---:|---:|---:|---:|---:|---:|---:|")
r = random.Random(23)
N = 100_000
for label, a_range, b_range in [
    ("a, b in 1e-310 … 1e308", (-310, 308), (-310, 308)),
    ("a in 1e300 … 1e308, b in 1e-10 … 1e10", (300, 308), (-10, 10)),
]:
    held = violations = 0
    out = {"differs": 0, "outside": 0, "both": 0}
    for _ in range(N):
        terms, original, eager, ex = trial(r, a_range, b_range)
        n = len(terms)
        if guard({0: terms}):
            held += 1
            if not (within(original, terms, ex, n) and within(eager, terms, ex, n)):
                violations += 1
        elif math.isfinite(original) != math.isfinite(eager):
            out["differs"] += 1
        elif math.isfinite(original) and not within(eager, terms, ex, n):
            out["outside"] += 1
        elif not math.isfinite(original):
            out["both"] += 1
    print(f"| {label} | {N} | {held} | {violations} | {N - held} | {out['differs']} | {out['outside']} | {out['both']} |")

# --- 3. what the guard costs ------------------------------------------------------

print("\n## 3. What checking the guard costs\n")

# Parquet statistics: do row-group min and max reveal infinities and NaNs?
with tempfile.TemporaryDirectory() as d:
    path = f"{d}/v.parquet"
    pq.write_table(pa.table({"v": [1.0, float("inf"), 2.0, float("nan"), -3.0]}), path)
    st = pq.ParquetFile(path).metadata.row_group(0).column(0).statistics
    print("Parquet (pyarrow) row-group statistics for v = [1, inf, 2, NaN, −3]: "
          f"min {st.min}, max {st.max}, null_count {st.null_count}.\n")

# A run-time pass over ddx's matmul operand, next to the contraction it guards.
def timed(fn, runs=5):
    fn()
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[runs // 2] * 1e3


n = 50_000
rr = np.arange(n * 16)
a = pa.table({"s": rr // 16, "k": rr % 16, "val": np.sin(rr.astype(float))})
rw = np.arange(16 * 8)
w = pa.table({"k": rw // 8, "o": rw % 8, "val": 0.1 * np.cos(rw.astype(float))})
ctx = SessionContext()
ctx.register_record_batches("a", [a.to_batches(8192)])
ctx.register_record_batches("w", [w.to_batches()])
contraction = "SELECT a.s, w.o, SUM(a.val * w.val) AS z FROM a JOIN w ON a.k = w.k GROUP BY a.s, w.o"
check = (
    "SELECT MAX(abs(val)) AS m, COUNT(*) AS n, "
    "SUM(CASE WHEN isnan(val) OR abs(val) = CAST('inf' AS DOUBLE) THEN 1 ELSE 0 END) AS bad FROM a"
)
t_con = timed(lambda: ctx.sql(contraction).collect())
t_chk = timed(lambda: ctx.sql(check).collect())
av = a.column("val").to_numpy()
t_np_sum = timed(lambda: av.sum())
t_np_chk = timed(lambda: (np.isfinite(av).all(), np.abs(av).max()))
print(f"| pass over ddx's 800,000-value operand (n = {n}) | ms |\n|---|---:|")
print(f"| DataFusion: the contraction `a·w` | {t_con:.2f} |")
print(f"| DataFusion: finiteness and max|v| check, as SQL | {t_chk:.2f} |")
print(f"| NumPy: `sum` | {t_np_sum:.3f} |")
print(f"| NumPy: `isfinite().all()` and `abs().max()` | {t_np_chk:.3f} |")
