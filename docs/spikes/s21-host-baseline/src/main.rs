// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Spike S21: what do host engines take on ddx's contractions, and how close
//! do hand-written dense loops get to the hardware?
//!
//! ddx (an XQL Systems project for differentiating SQL queries) runs neural
//! networks as SQL. Its matrix products and attention are joins of
//! coordinate tables `(i, k, val)` followed by `GROUP BY` and `SUM`. This
//! spike times those exact queries on two hosts, Apache DataFusion and
//! DuckDB, and next to them two hand-written kernels that read and write the
//! same coordinate rows (Arrow arrays in, Arrow arrays out):
//!
//! - **dense:** scatter both operands into dense matrices, multiply with a
//!   GEMM (`matrixmultiply`), and emit one row per output cell;
//! - **fold:** keep only the second operand and the output dense; stream the
//!   first operand's rows, adding `val · B[k, :]` into the output row. This
//!   is the positional form of a fused join-and-aggregate: no hashing, no
//!   join rows.
//!
//! Both kernels treat coordinates as positions (`0..extent`). That needs a
//! fact that each dimension is a dense range, which ddx's tables have, and
//! they assume every output cell exists (both inputs complete).

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use datafusion::arrow::array::{Array, Float64Array, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::prelude::SessionContext;

const RUNS: usize = 5;
const BATCH: usize = 8192;

/// A coordinate table: rows `r` in `0..extent0·extent1`, with `c0 = r /
/// extent1`, `c1 = r % extent1`, and `val = scale · sin(r + phase)`.
#[derive(Clone, Copy)]
struct Table {
    name: &'static str,
    cols: (&'static str, &'static str),
    extents: (usize, usize),
    phase: f64,
    scale: f64,
}

impl Table {
    fn rows(&self) -> usize {
        self.extents.0 * self.extents.1
    }

    fn data(&self) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
        let w = self.extents.1;
        let n = self.rows();
        let c0 = (0..n).map(|r| (r / w) as i64).collect();
        let c1 = (0..n).map(|r| (r % w) as i64).collect();
        let v = (0..n).map(|r| self.scale * (r as f64 + self.phase).sin()).collect();
        (c0, c1, v)
    }

    fn duckdb_create(&self) -> String {
        format!(
            "CREATE TABLE {} AS SELECT CAST(r // {w} AS BIGINT) AS {}, CAST(r % {w} AS BIGINT) AS {}, \
             {} * sin(CAST(r AS DOUBLE) + {}) AS val FROM range(0, {}) t(r);",
            self.name,
            self.cols.0,
            self.cols.1,
            self.scale,
            self.phase,
            self.rows(),
            w = self.extents.1,
        )
    }
}

/// `C[x, y] = Σ_z A[x, z] · B[z, y]`, with the columns of each operand that
/// hold `x`, `z` and `y` (0 or 1).
struct Contraction {
    label: &'static str,
    a: Table,
    b: Table,
    a_x: usize,
    b_z: usize,
}

impl Contraction {
    fn a_z(&self) -> usize {
        1 - self.a_x
    }
    fn b_y(&self) -> usize {
        1 - self.b_z
    }
    fn col(t: &Table, i: usize) -> &'static str {
        if i == 0 {
            t.cols.0
        } else {
            t.cols.1
        }
    }
    fn extent(t: &Table, i: usize) -> usize {
        if i == 0 {
            t.extents.0
        } else {
            t.extents.1
        }
    }
    fn dims(&self) -> (usize, usize, usize) {
        (
            Self::extent(&self.a, self.a_x),
            Self::extent(&self.a, self.a_z()),
            Self::extent(&self.b, self.b_y()),
        )
    }
    fn sql(&self) -> String {
        let (a, b) = (self.a.name, self.b.name);
        format!(
            "SELECT {a}.{x}, {b}.{y}, SUM({a}.val * {b}.val) AS v FROM {a} JOIN {b} \
             ON {a}.{za} = {b}.{zb} GROUP BY {a}.{x}, {b}.{y}",
            x = Self::col(&self.a, self.a_x),
            y = Self::col(&self.b, self.b_y()),
            za = Self::col(&self.a, self.a_z()),
            zb = Self::col(&self.b, self.b_z),
        )
    }
}

fn mm(n: usize) -> Vec<Contraction> {
    let a = Table { name: "a", cols: ("s", "k"), extents: (n, 16), phase: 0.0, scale: 1.0 };
    let w = Table { name: "w", cols: ("k", "o"), extents: (16, 8), phase: 0.5, scale: 0.1 };
    let zbar = Table { name: "zbar", cols: ("s", "o"), extents: (n, 8), phase: 1.0, scale: 0.01 };
    vec![
        Contraction { label: "forward a·w", a, b: w, a_x: 0, b_z: 0 },
        Contraction { label: "backward W̄ = aᵀ·z̄", a, b: zbar, a_x: 1, b_z: 0 },
        Contraction { label: "backward Ā = z̄·wᵀ", a: zbar, b: w, a_x: 0, b_z: 1 },
    ]
}

fn attn(l: usize) -> Vec<Contraction> {
    let q = Table { name: "q", cols: ("t", "j"), extents: (l, 16), phase: 0.0, scale: 1.0 };
    let kk = Table { name: "kk", cols: ("u", "j"), extents: (l, 16), phase: 0.5, scale: 1.0 };
    let p = Table { name: "p", cols: ("t", "u"), extents: (l, l), phase: 1.0, scale: 0.01 };
    let vv = Table { name: "vv", cols: ("u", "j"), extents: (l, 16), phase: 1.5, scale: 1.0 };
    vec![
        Contraction { label: "scores q·kᵀ", a: q, b: kk, a_x: 0, b_z: 1 },
        Contraction { label: "output p·v", a: p, b: vv, a_x: 0, b_z: 0 },
    ]
}

/// Coordinate rows of the result, as Arrow arrays.
struct Out {
    x: Int64Array,
    y: Int64Array,
    v: Float64Array,
}

/// Sum of absolute values: a checksum that cancellation can't hide.
fn checksum(v: &Float64Array) -> f64 {
    v.values().iter().map(|x| x.abs()).sum()
}

fn median(mut f: impl FnMut() -> f64) -> (Duration, f64) {
    let mut sum = f();
    let mut times = Vec::new();
    for _ in 0..RUNS {
        let t = Instant::now();
        sum = f();
        times.push(t.elapsed());
    }
    times.sort();
    (times[RUNS / 2], sum)
}

// --- hosts -----------------------------------------------------------------

async fn datafusion_time(c: &Contraction) -> (Duration, f64) {
    let ctx = SessionContext::new();
    let threads = thread::available_parallelism().unwrap().get();
    let mut seen = Vec::new();
    for t in [c.a, c.b] {
        if seen.contains(&t.name) {
            continue;
        }
        seen.push(t.name);
        let schema = Arc::new(Schema::new(vec![
            Field::new(t.cols.0, DataType::Int64, false),
            Field::new(t.cols.1, DataType::Int64, false),
            Field::new("val", DataType::Float64, false),
        ]));
        let (c0, c1, v) = t.data();
        let mut parts = vec![Vec::new(); threads];
        for (i, start) in (0..t.rows()).step_by(BATCH).enumerate() {
            let end = (start + BATCH).min(t.rows());
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(c0[start..end].to_vec())),
                    Arc::new(Int64Array::from(c1[start..end].to_vec())),
                    Arc::new(Float64Array::from(v[start..end].to_vec())),
                ],
            )
            .unwrap();
            parts[i % threads].push(batch);
        }
        let table = MemTable::try_new(schema, parts).unwrap();
        ctx.register_table(t.name, Arc::new(table)).unwrap();
    }
    let sql = c.sql();
    let mut sum = run_sql(&ctx, &sql).await;
    let mut times = Vec::new();
    for _ in 0..RUNS {
        let t = Instant::now();
        sum = run_sql(&ctx, &sql).await;
        times.push(t.elapsed());
    }
    times.sort();
    (times[RUNS / 2], sum)
}

async fn run_sql(ctx: &SessionContext, sql: &str) -> f64 {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    batches
        .iter()
        .map(|b| checksum(b.column(2).as_any().downcast_ref::<Float64Array>().unwrap()))
        .sum()
}

/// Times each contraction in DuckDB's CLI, in one session per workload.
fn duckdb_times(cs: &[Contraction]) -> Vec<(Duration, f64)> {
    let mut script = String::from("SET threads = 12;\nSET memory_limit = '6GB';\n");
    let mut made = Vec::new();
    for c in cs {
        for t in [c.a, c.b] {
            if !made.contains(&t.name) {
                made.push(t.name);
                script += &t.duckdb_create();
                script += "\n";
            }
        }
    }
    for c in cs {
        script += ".timer on\n";
        for _ in 0..=RUNS {
            script += &format!("CREATE OR REPLACE TEMP TABLE out AS {};\n", c.sql());
        }
        script += ".timer off\nSELECT sum(abs(v)) FROM out;\n";
    }
    let mut child = Command::new("duckdb")
        .args(["-csv", "-noheader", ":memory:"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("duckdb on PATH");
    child.stdin.take().unwrap().write_all(script.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    let mut results = Vec::new();
    let mut times = Vec::new();
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Run Time (s): real ") {
            let secs: f64 = rest.split_whitespace().next().unwrap().parse().unwrap();
            times.push(Duration::from_secs_f64(secs));
        } else if let Ok(sum) = line.trim().parse::<f64>() {
            let mut t: Vec<Duration> = times.drain(..).skip(1).collect();
            t.sort();
            results.push((t[t.len() / 2], sum));
        }
    }
    assert_eq!(results.len(), cs.len(), "duckdb output:\n{text}\n{}", String::from_utf8_lossy(&out.stderr));
    results
}

// --- hand-written kernels ----------------------------------------------------

/// Operand data as positions: rows of (x or z, z or y, val).
struct Rows {
    c0: Vec<i64>,
    c1: Vec<i64>,
    v: Vec<f64>,
}

fn rows(t: &Table) -> Rows {
    let (c0, c1, v) = t.data();
    Rows { c0, c1, v }
}

/// Scatter rows into a dense row-major matrix whose row index is column `r`
/// and whose column index is the other column.
fn scatter(t: &Rows, r: usize, ncols: usize, nrows: usize) -> Vec<f64> {
    let mut m = vec![0.0; nrows * ncols];
    let (ri, ci) = if r == 0 { (&t.c0, &t.c1) } else { (&t.c1, &t.c0) };
    for ((&i, &j), &v) in ri.iter().zip(ci).zip(&t.v) {
        m[i as usize * ncols + j as usize] = v;
    }
    m
}

/// `c[rows] += a[rows] · b` by GEMM, for `a` of shape `m × k`, `b` of `k × n`.
fn gemm(a: &[f64], b: &[f64], c: &mut [f64], k: usize, n: usize) {
    let m = a.len() / k;
    // SAFETY: the slices hold m·k, k·n and m·n elements in row-major order.
    unsafe {
        matrixmultiply::dgemm(
            m, k, n, 1.0, a.as_ptr(), k as isize, 1, b.as_ptr(), n as isize, 1, 1.0,
            c.as_mut_ptr(), n as isize, 1,
        );
    }
}

/// Emit every output cell as a coordinate row.
fn emit(c: &[f64], ny: usize, threads: usize) -> Out {
    let n = c.len();
    let mut x = vec![0i64; n];
    let mut y = vec![0i64; n];
    let chunk = n.div_ceil(threads).max(1);
    thread::scope(|s| {
        for (i, (xs, ys)) in x.chunks_mut(chunk).zip(y.chunks_mut(chunk)).enumerate() {
            s.spawn(move || {
                for (j, (xv, yv)) in xs.iter_mut().zip(ys.iter_mut()).enumerate() {
                    let cell = i * chunk + j;
                    *xv = (cell / ny) as i64;
                    *yv = (cell % ny) as i64;
                }
            });
        }
    });
    Out { x: Int64Array::from(x), y: Int64Array::from(y), v: Float64Array::from(c.to_vec()) }
}

/// Split work over threads: over output rows `x` when there are enough of
/// them, otherwise over the contraction with one partial output per thread.
fn split_x(nx: usize, threads: usize) -> bool {
    threads == 1 || nx >= 4 * threads
}

fn dense(c: &Contraction, a: &Rows, b: &Rows, threads: usize) -> Out {
    let (nx, nz, ny) = c.dims();
    let am = scatter(a, c.a_x, nz, nx);
    let bm = scatter(b, c.b_z, ny, nz);
    let mut out = vec![0.0; nx * ny];
    if split_x(nx, threads) {
        let rows = nx.div_ceil(threads);
        thread::scope(|s| {
            for (ac, cc) in am.chunks(rows * nz).zip(out.chunks_mut(rows * ny)) {
                let bm = &bm;
                s.spawn(move || gemm(ac, bm, cc, nz, ny));
            }
        });
    } else {
        // Split z: a's columns and b's rows. Transpose a's slice per thread.
        let zs = nz.div_ceil(threads);
        let partials: Vec<Vec<f64>> = thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|t| {
                    let (am, bm) = (&am, &bm);
                    s.spawn(move || {
                        let (z0, z1) = (t * zs, ((t + 1) * zs).min(nz));
                        let mut p = vec![0.0; nx * ny];
                        if z0 < z1 {
                            let w = z1 - z0;
                            let mut asl = vec![0.0; nx * w];
                            for i in 0..nx {
                                asl[i * w..(i + 1) * w].copy_from_slice(&am[i * nz + z0..i * nz + z1]);
                            }
                            gemm(&asl, &bm[z0 * ny..z1 * ny], &mut p, w, ny);
                        }
                        p
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for p in partials {
            for (o, v) in out.iter_mut().zip(p) {
                *o += v;
            }
        }
    }
    emit(&out, ny, threads)
}

/// Stream `a`'s rows into a dense output, with `b` dense: the positional
/// fused join-and-aggregate.
fn fold(c: &Contraction, a: &Rows, b: &Rows, threads: usize) -> Out {
    let (nx, nz, ny) = c.dims();
    let bm = scatter(b, c.b_z, ny, nz);
    let (xs, zs) = if c.a_x == 0 { (&a.c0, &a.c1) } else { (&a.c1, &a.c0) };
    let accumulate = |out: &mut [f64], x0: usize, idx: &mut dyn Iterator<Item = usize>| {
        for r in idx {
            let (x, z, v) = (xs[r] as usize - x0, zs[r] as usize, a.v[r]);
            let row = &mut out[x * ny..(x + 1) * ny];
            for (o, bv) in row.iter_mut().zip(&bm[z * ny..(z + 1) * ny]) {
                *o += v * bv;
            }
        }
    };
    let mut out = vec![0.0; nx * ny];
    let n = a.v.len();
    if split_x(nx, threads) {
        // Bucket rows by output-row range, so threads own disjoint outputs.
        let per = nx.div_ceil(threads);
        let mut buckets = vec![Vec::new(); threads];
        for (r, &x) in xs.iter().enumerate() {
            buckets[x as usize / per].push(r);
        }
        thread::scope(|s| {
            for (t, (cc, bucket)) in out.chunks_mut(per * ny).zip(&buckets).enumerate() {
                let accumulate = &accumulate;
                s.spawn(move || accumulate(cc, t * per, &mut bucket.iter().copied()));
            }
        });
    } else {
        let per = n.div_ceil(threads);
        let partials: Vec<Vec<f64>> = thread::scope(|s| {
            let hs: Vec<_> = (0..threads)
                .map(|t| {
                    let accumulate = &accumulate;
                    s.spawn(move || {
                        let mut p = vec![0.0; nx * ny];
                        accumulate(&mut p, 0, &mut (t * per..((t + 1) * per).min(n)));
                        p
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for p in partials {
            for (o, v) in out.iter_mut().zip(p) {
                *o += v;
            }
        }
    }
    emit(&out, ny, threads)
}

// --- report ------------------------------------------------------------------

fn ms(d: Duration) -> String {
    format!("{:.1}", d.as_secs_f64() * 1e3)
}

#[tokio::main]
async fn main() {
    let threads = thread::available_parallelism().unwrap().get();
    let mut workloads: Vec<(String, Vec<Contraction>)> = Vec::new();
    for n in [10_000, 50_000, 200_000] {
        workloads.push((format!("matmul n={n}"), mm(n)));
    }
    for l in [256, 1024, 2048] {
        workloads.push((format!("attn L={l}"), attn(l)));
    }
    println!(
        "| workload | query | joined pairs | output rows | DataFusion | DuckDB | dense, 1 thread | dense, {threads} | fold, 1 thread | fold, {threads} | best kernel vs best host |"
    );
    println!("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for (name, cs) in &workloads {
        let duck = duckdb_times(cs);
        for (c, (duck_t, duck_sum)) in cs.iter().zip(duck) {
            let (nx, nz, ny) = c.dims();
            let (df_t, df_sum) = datafusion_time(c).await;
            let (a, b) = (rows(&c.a), rows(&c.b));
            let kernel = |f: fn(&Contraction, &Rows, &Rows, usize) -> Out, t: usize| {
                median(|| {
                    let o = f(c, &a, &b, t);
                    assert_eq!(o.x.len(), nx * ny);
                    assert_eq!(o.y.len(), nx * ny);
                    checksum(&o.v)
                })
            };
            let ks = [
                kernel(dense, 1),
                kernel(dense, threads),
                kernel(fold, 1),
                kernel(fold, threads),
            ];
            let scale = df_sum.max(1.0);
            for (t, s) in ks.iter().chain([&(duck_t, duck_sum)]) {
                assert!((s - df_sum).abs() <= 1e-9 * scale, "{name} {}: {s} vs {df_sum} ({t:?})", c.label);
            }
            let host = df_t.min(duck_t);
            let best = ks.iter().map(|k| k.0).min().unwrap();
            println!(
                "| {name} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {:.1}× |",
                c.label,
                nx * nz * ny,
                nx * ny,
                ms(df_t),
                ms(duck_t),
                ms(ks[0].0),
                ms(ks[1].0),
                ms(ks[2].0),
                ms(ks[3].0),
                host.as_secs_f64() / best.as_secs_f64(),
            );
        }
    }
}
