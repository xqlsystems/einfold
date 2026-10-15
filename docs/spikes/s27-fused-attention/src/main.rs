// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Spike S27: demo 1's causal attention, written in SQL, on the hosts, against
//! a fused, tiled kernel with an online softmax.
//!
//! The tables are `q`, `k` and `v`, each `(t, d, x)`: token, feature, value.
//! The query is demo 1's, in two spellings: with a window function, as the
//! demo writes it, and with `GROUP BY` and a join, as ddx (an XQL Systems
//! project for differentiating SQL queries) writes it.
//!
//! The kernel reads the same rows. It treats `t` and `d` as positions, which
//! needs them to be dense ranges and the tables complete, as here. For each
//! tile of 64 queries it streams over tiles of 64 keys, computes the tile's
//! scores with a GEMM, skips tiles the causal mask removes entirely, and
//! keeps a running maximum, normalizer and weighted sum per query, rescaled
//! when the maximum grows (Milakov and Gimelshein's online softmax). So it
//! never holds more than one 64×64 tile of scores per thread.
//!
//! Subcommands, one measurement per process so that peak memory is the
//! process's own:
//!
//! - `df <window|groupby> <L>`: DataFusion, with a 6 GB memory pool;
//! - `duck-sql <window|groupby> <L>`: print a DuckDB CLI script;
//! - `fused <L> <threads>`: the kernel;
//! - `accuracy <L>`: kernel and DataFusion against a compensated reference;
//! - `edges`: NaN and infinities, kernel against DataFusion.

use std::env;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use datafusion::arrow::array::{Array, Float64Array, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};

const D: usize = 64;
const SCALE: f64 = 8.0; // sqrt(D), as the demo's SQL writes it
const BQ: usize = 64;
const BK: usize = 64;

/// Values of table `name` at row `r = t·D + d`.
fn value(name: &str, r: usize) -> f64 {
    let phase = match name {
        "q" => 0.0,
        "k" => 0.5,
        _ => 1.0,
    };
    (r as f64 * 0.7 + phase).sin()
}

fn dense(name: &str, l: usize) -> Vec<f64> {
    (0..l * D).map(|r| value(name, r)).collect()
}

fn sql(form: &str) -> String {
    let scores = format!(
        "scores AS (SELECT q.t AS i, k.t AS j, SUM(q.x * k.x) / {SCALE:.1} AS s \
         FROM q JOIN k ON q.d = k.d WHERE q.t >= k.t GROUP BY q.t, k.t)"
    );
    let weights = match form {
        "window" => "weights AS (SELECT i, j, exp(s - MAX(s) OVER (PARTITION BY i)) AS w FROM scores)"
            .to_string(),
        "groupby" => "m AS (SELECT i, MAX(s) AS m FROM scores GROUP BY i), \
             weights AS (SELECT scores.i, scores.j, exp(scores.s - m.m) AS w \
             FROM scores JOIN m ON scores.i = m.i)"
            .to_string(),
        f => panic!("unknown form {f}"),
    };
    format!(
        "WITH {scores}, {weights}, norm AS (SELECT i, SUM(w) AS z FROM weights GROUP BY i) \
         SELECT weights.i AS t, v.d, SUM(weights.w * v.x) / norm.z AS x \
         FROM weights JOIN v ON weights.j = v.t JOIN norm ON weights.i = norm.i \
         GROUP BY weights.i, v.d, norm.z"
    )
}

// --- DataFusion ----------------------------------------------------------------

async fn df_context(l: usize, tables: &[(&str, Vec<f64>)]) -> SessionContext {
    let rt = RuntimeEnvBuilder::new()
        .with_memory_limit(6 << 30, 1.0)
        .with_max_temp_directory_size(4 << 30)
        .build_arc()
        .unwrap();
    let ctx = SessionContext::new_with_config_rt(SessionConfig::new(), rt);
    let schema = Arc::new(Schema::new(vec![
        Field::new("t", DataType::Int64, false),
        Field::new("d", DataType::Int64, false),
        Field::new("x", DataType::Float64, true),
    ]));
    for (name, vals) in tables {
        let n = l * D;
        let mut parts = vec![Vec::new(); 12];
        for (i, start) in (0..n).step_by(8192).enumerate() {
            let end = (start + 8192).min(n);
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from_iter_values((start..end).map(|r| (r / D) as i64))),
                    Arc::new(Int64Array::from_iter_values((start..end).map(|r| (r % D) as i64))),
                    Arc::new(Float64Array::from(vals[start..end].to_vec())),
                ],
            )
            .unwrap();
            parts[i % 12].push(batch);
        }
        let t = MemTable::try_new(schema.clone(), parts).unwrap();
        ctx.register_table(*name, Arc::new(t)).unwrap();
    }
    ctx
}

/// Run the query; return (t, d, x) for every output row.
async fn df_run(ctx: &SessionContext, form: &str) -> Vec<(usize, usize, Option<f64>)> {
    let batches = ctx.sql(&sql(form)).await.unwrap().collect().await.unwrap();
    let mut out = Vec::new();
    for b in &batches {
        let t = b.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
        let d = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
        let x = b.column(2).as_any().downcast_ref::<Float64Array>().unwrap();
        for r in 0..b.num_rows() {
            out.push((t.value(r) as usize, d.value(r) as usize, x.is_valid(r).then(|| x.value(r))));
        }
    }
    out
}

fn checksum(xs: impl Iterator<Item = f64>) -> f64 {
    xs.map(f64::abs).sum()
}

// --- the fused kernel ------------------------------------------------------------

/// `c (m×n) += a (m×k, row stride rsa, column stride csa) · b (k×n, strides rsb, csb)`.
#[allow(clippy::too_many_arguments)]
fn gemm(m: usize, k: usize, n: usize, a: &[f64], rsa: isize, csa: isize, b: &[f64], rsb: isize, csb: isize, c: &mut [f64]) {
    // SAFETY: callers pass slices covering the strided extents.
    unsafe {
        matrixmultiply::dgemm(m, k, n, 1.0, a.as_ptr(), rsa, csa, b.as_ptr(), rsb, csb, 1.0, c.as_mut_ptr(), n as isize, 1);
    }
}

/// Causal attention over dense `q`, `k`, `v` (each `l × D`, row-major), one
/// tile of queries at a time, keys streamed in tile order. Each query tile
/// is computed by one thread in a fixed order, so the bits don't depend on
/// the thread count.
///
/// With `zero_masked`, a partly masked tile gives masked pairs a weight of 0
/// and multiplies the whole tile by `V`, as a naive kernel would. That is
/// wrong in SQL terms: a masked pair is an absent row, but `0 · NaN` is NaN.
/// Without it, masked pairs are skipped.
fn fused(q: &[f64], k: &[f64], v: &[f64], l: usize, threads: usize, zero_masked: bool) -> Vec<f64> {
    let mut out = vec![0.0; l * D];
    let chunks: Vec<&mut [f64]> = out.chunks_mut(BQ * D).collect();
    let mut per_thread: Vec<Vec<(usize, &mut [f64])>> = (0..threads).map(|_| Vec::new()).collect();
    for (tile, chunk) in chunks.into_iter().enumerate() {
        per_thread[tile % threads].push((tile, chunk));
    }
    thread::scope(|s| {
        for work in per_thread {
            s.spawn(move || {
                let mut sc = vec![0.0; BQ * BK];
                let mut acc = vec![0.0; BQ * D];
                for (tile, o) in work {
                    let (i0, i1) = (tile * BQ, ((tile + 1) * BQ).min(l));
                    let bq = i1 - i0;
                    let mut m = vec![f64::NEG_INFINITY; bq];
                    let mut z = vec![0.0; bq];
                    acc[..bq * D].fill(0.0);
                    // Causal mask q.t >= k.t: key tiles starting after i1 - 1 are skipped.
                    for j0 in (0..i1).step_by(BK) {
                        let bk = (j0 + BK).min(l) - j0;
                        let s_ = &mut sc[..bq * bk];
                        s_.fill(0.0);
                        // scores = Q[I] · K[J]ᵀ
                        gemm(bq, D, bk, &q[i0 * D..], D as isize, 1, &k[j0 * D..], 1, D as isize, s_);
                        for ii in 0..bq {
                            let row = &mut s_[ii * bk..(ii + 1) * bk];
                            let i = i0 + ii;
                            let mut rmax = f64::NEG_INFINITY;
                            for (jj, x) in row.iter_mut().enumerate() {
                                if j0 + jj > i {
                                    *x = f64::NEG_INFINITY; // masked: no row in SQL
                                } else {
                                    *x /= SCALE;
                                    rmax = rmax.max(*x);
                                }
                            }
                            let m_new = m[ii].max(rmax);
                            if m_new == f64::NEG_INFINITY {
                                row.fill(0.0);
                                continue;
                            }
                            let alpha = (m[ii] - m_new).exp();
                            let mut sum = 0.0;
                            for x in row.iter_mut() {
                                *x = (*x - m_new).exp();
                                sum += *x;
                            }
                            z[ii] = z[ii] * alpha + sum;
                            for a in &mut acc[ii * D..(ii + 1) * D] {
                                *a *= alpha;
                            }
                            m[ii] = m_new;
                        }
                        // acc += P · V[J], skipping masked pairs in a partly masked tile.
                        let partial = j0 + bk > i0 + 1;
                        if !partial || zero_masked {
                            gemm(bq, bk, D, s_, bk as isize, 1, &v[j0 * D..], D as isize, 1, &mut acc[..bq * D]);
                        } else {
                            for ii in 0..bq {
                                let allowed = (i0 + ii + 1).saturating_sub(j0).min(bk);
                                let out = &mut acc[ii * D..(ii + 1) * D];
                                for jj in 0..allowed {
                                    let p = s_[ii * bk + jj];
                                    for (o, vv) in out.iter_mut().zip(&v[(j0 + jj) * D..(j0 + jj + 1) * D]) {
                                        *o += p * vv;
                                    }
                                }
                            }
                        }
                    }
                    for ii in 0..bq {
                        for e in 0..D {
                            o[ii * D + e] = acc[ii * D + e] / z[ii];
                        }
                    }
                }
            });
        }
    });
    out
}

// --- reference -------------------------------------------------------------------

/// Neumaier's compensated sum.
fn csum(xs: impl Iterator<Item = f64>) -> f64 {
    let (mut s, mut c) = (0.0f64, 0.0f64);
    for x in xs {
        let t = s + x;
        c += if s.abs() >= x.abs() { (s - t) + x } else { (x - t) + s };
        s = t;
    }
    s + c
}

/// The SQL's arithmetic with compensated sums: exact row maximum, then the
/// weights, the normalizer and the weighted sums.
fn reference(q: &[f64], k: &[f64], v: &[f64], l: usize) -> Vec<f64> {
    let mut out = vec![0.0; l * D];
    for i in 0..l {
        let s: Vec<f64> = (0..=i)
            .map(|j| csum((0..D).map(|e| q[i * D + e] * k[j * D + e])) / SCALE)
            .collect();
        let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let w: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
        let z = csum(w.iter().copied());
        for e in 0..D {
            out[i * D + e] = csum(w.iter().enumerate().map(|(j, wj)| wj * v[j * D + e])) / z;
        }
    }
    out
}

// --- subcommands -----------------------------------------------------------------

fn arg(i: usize) -> String {
    env::args().nth(i).unwrap_or_default()
}

#[tokio::main]
async fn main() {
    match arg(1).as_str() {
        "df" => {
            let (form, l) = (arg(2), arg(3).parse::<usize>().unwrap());
            let tables: Vec<(&str, Vec<f64>)> = ["q", "k", "v"].iter().map(|n| (*n, dense(n, l))).collect();
            let ctx = df_context(l, &tables).await;
            let t = Instant::now();
            let rows = df_run(&ctx, &form).await;
            let ms = t.elapsed().as_secs_f64() * 1e3;
            println!(
                "ms={ms:.1} rows={} checksum={:.9e}",
                rows.len(),
                checksum(rows.iter().filter_map(|r| r.2))
            );
        }
        "duck-sql" => {
            let (form, l) = (arg(2), arg(3).parse::<usize>().unwrap());
            println!("SET threads = 12;\nSET memory_limit = '6GB';\nSET max_temp_directory_size = '4GB';");
            for (i, name) in ["q", "k", "v"].iter().enumerate() {
                let phase = [0.0, 0.5, 1.0][i];
                println!(
                    "CREATE TABLE {name} AS SELECT CAST(r // {D} AS BIGINT) AS t, CAST(r % {D} AS BIGINT) AS d, \
                     sin(CAST(r AS DOUBLE) * 0.7 + {phase}) AS x FROM range(0, {}) tbl(r);",
                    l * D
                );
            }
            println!(".timer on\nCREATE TEMP TABLE out AS {};\n.timer off", sql(&form));
            println!("SELECT count(*), sum(abs(x)) FROM out;");
        }
        "fused" => {
            let (l, threads) = (arg(2).parse::<usize>().unwrap(), arg(3).parse::<usize>().unwrap());
            // Rows as the hosts get them: (t, d, x) columns. The time includes
            // scattering them to positions and writing the output rows back.
            let rows = |name: &str| -> (Vec<i64>, Vec<i64>, Vec<f64>) {
                let n = l * D;
                ((0..n).map(|r| (r / D) as i64).collect(), (0..n).map(|r| (r % D) as i64).collect(), dense(name, l))
            };
            let (rq, rk, rv) = (rows("q"), rows("k"), rows("v"));
            let scatter = |(t, d, x): &(Vec<i64>, Vec<i64>, Vec<f64>)| {
                let mut m = vec![0.0; l * D];
                for ((t, d), x) in t.iter().zip(d).zip(x) {
                    m[*t as usize * D + *d as usize] = *x;
                }
                m
            };
            let mut times = Vec::new();
            let mut out = Vec::new();
            for _ in 0..3 {
                let t = Instant::now();
                let (q, k, v) = (scatter(&rq), scatter(&rk), scatter(&rv));
                let dense_out = fused(&q, &k, &v, l, threads, false);
                let n = dense_out.len();
                let ot = Int64Array::from_iter_values((0..n).map(|r| (r / D) as i64));
                let od = Int64Array::from_iter_values((0..n).map(|r| (r % D) as i64));
                let ox = Float64Array::from(dense_out);
                assert_eq!(ot.len() + od.len(), 2 * ox.len());
                out = ox.values().to_vec();
                times.push(t.elapsed().as_secs_f64() * 1e3);
            }
            times.sort_by(f64::total_cmp);
            let tiles = l.div_ceil(BQ);
            let visited: usize = (0..tiles).map(|t| ((t + 1) * BQ).min(l).div_ceil(BK)).sum();
            println!(
                "ms={:.1} rows={} checksum={:.9e} key_tiles_visited={visited} of {}",
                times[1],
                out.len(),
                checksum(out.iter().copied()),
                tiles * l.div_ceil(BK)
            );
        }
        "accuracy" => {
            // `accuracy <L> <scale>`: q is multiplied by `scale`, which spreads the
            // scores and makes the running maximum rescale more often.
            let l = arg(2).parse::<usize>().unwrap();
            let qs = arg(3).parse::<f64>().unwrap_or(1.0);
            let q: Vec<f64> = dense("q", l).iter().map(|x| x * qs).collect();
            let (k, v) = (dense("k", l), dense("v", l));
            let r = reference(&q, &k, &v, l);
            let scale = r.iter().fold(0.0f64, |a, x| a.max(x.abs()));
            let f12 = fused(&q, &k, &v, l, 12, false);
            let same_bits = [1, 2, 5, 12].iter().all(|&t| {
                fused(&q, &k, &v, l, t, false).iter().zip(&f12).all(|(a, b)| a.to_bits() == b.to_bits())
            });
            let err = |xs: &[f64]| xs.iter().zip(&r).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / scale;
            let ctx = df_context(l, &[("q", q.clone()), ("k", k.clone()), ("v", v.clone())]).await;
            let mut dfo = vec![f64::NAN; l * D];
            for (t, d, x) in df_run(&ctx, "window").await {
                dfo[t * D + d] = x.unwrap_or(f64::NAN);
            }
            println!(
                "L={l} q_scale={qs} max|out|={scale:.3e} fused_err={:.2e} datafusion_err={:.2e} fused_same_bits_1_2_5_12_threads={same_bits}",
                err(&f12),
                err(&dfo)
            );
        }
        "edges" => {
            let l = 128;
            let cases: [(&str, &str, usize, f64); 7] = [
                ("none", "q", 0, 0.0),
                ("NaN in q", "q", 70 * D + 3, f64::NAN),
                ("NaN in v", "v", 5 * D + 1, f64::NAN),
                ("+inf in q", "q", 90 * D + 2, f64::INFINITY),
                ("-inf in q", "q", 100 * D + 2, f64::NEG_INFINITY),
                ("1e300 in k", "k", 3 * D + 4, 1e300),
                ("row 100's scores all -inf", "q", 100 * D, f64::NEG_INFINITY),
            ];
            for (label, table, at, x) in cases {
                let mut t: Vec<(&str, Vec<f64>)> = ["q", "k", "v"].iter().map(|n| (*n, dense(n, l))).collect();
                if label != "none" {
                    t.iter_mut().find(|(n, _)| *n == table).unwrap().1[at] = x;
                }
                if label.starts_with("row 100") {
                    // k(j, 0) = 1 for every j, so every score of row 100 is -inf.
                    for j in 0..l {
                        t[1].1[j * D] = 1.0;
                    }
                }
                let f = fused(&t[0].1, &t[1].1, &t[2].1, l, 4, false);
                let fz = fused(&t[0].1, &t[1].1, &t[2].1, l, 4, true);
                let ctx = df_context(l, &t).await;
                let rows = df_run(&ctx, "window").await;
                let same = |a: f64, b: Option<f64>| match b {
                    None => false,
                    Some(b) if b.is_nan() => a.is_nan(),
                    Some(b) if b.is_infinite() => a == b,
                    Some(b) => (a - b).abs() <= 1e-12 * (1.0 + b.abs()),
                };
                let differ = |k: &[f64]| rows.iter().filter(|(ti, di, x)| !same(k[ti * D + di], *x)).count();
                let df_nan = rows.iter().filter(|r| r.2.is_none_or(f64::is_nan)).count();
                let f_nan = f.iter().filter(|x| x.is_nan()).count();
                println!(
                    "| {label} | {} | {df_nan} | {f_nan} | {} | {} |",
                    rows.len(),
                    differ(&f),
                    differ(&fz)
                );
            }
        }
        c => panic!("unknown subcommand {c:?}"),
    }
}
