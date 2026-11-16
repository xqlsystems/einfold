// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Properties of what einfold offers a dense backend.
//!
//! 1. **Offered attention is order-independent.** Whenever attention is
//!    offered, SQL's two spellings, `SUM(w · v) / z` and `SUM(w / z · v)`,
//!    and a backend's softmax-first order agree within rounding relative to
//!    the row's own magnitude, `Σ (w/z)·|v|`, and none rounds a row to 0
//!    where another doesn't. Values are drawn near the guards' limits.
//! 2. **A float32 backend** (pinned): the guards are in float64's range, so
//!    a float32 backend can overflow where SQL doesn't, as the backend's
//!    documentation says.
//! 3. **No product is offered that overflows in einfold's order**, so a
//!    backend that sums in another order can't turn SQL's infinity finite.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use datafusion::arrow::array::{Float64Array, Int64Array, RecordBatch};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use einfold_ir::{
    Aggregate, ColumnIndex, Dimension, DimensionId, Fold, Key, KeyEquality, Normalizer, Operand,
    Output, Pipeline, RowValue, Scale, Shift, Weight,
};
use einfold_kernels::{DenseBackend, DenseWork, Options, Relation};
use einfold_testkit::Rng;

mod reference;

fn table(names: &[&str], extents: &[usize], values: &[f64]) -> (SchemaRef, RecordBatch) {
    let mut fields: Vec<Field> = (names.iter())
        .map(|n| Field::new(*n, DataType::Int64, false))
        .collect();
    fields.push(Field::new("x", DataType::Float64, true));
    let schema = Arc::new(Schema::new(fields));
    let cells: usize = extents.iter().product();
    assert_eq!(cells, values.len());
    let mut columns: Vec<Arc<dyn datafusion::arrow::array::Array>> = (0..extents.len())
        .map(|c| {
            let stride: usize = extents[c + 1..].iter().product();
            Arc::new(Int64Array::from_iter_values(
                (0..cells).map(|r| ((r / stride) % extents[c]) as i64),
            )) as _
        })
        .collect();
    columns.push(Arc::new(Float64Array::from(values.to_vec())));
    let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
    (schema, batch)
}

/// Attention `softmax(q · k) · v` over `q(t, d)`, `k(t, d)`, `v(t)`.
fn attention(scale: Scale) -> Pipeline {
    const EQ: Dimension = Dimension {
        equality: KeyEquality::Equal,
    };
    let key = |d, c| Key::new(DimensionId(d), ColumnIndex(c));
    let out = |d| Output {
        dimension: DimensionId(d),
        domain_key: None,
    };
    let ops = vec![
        Operand {
            keys: vec![key(0, 0), key(1, 1)],
            factor: Some(ColumnIndex(2)),
        },
        Operand {
            keys: vec![key(2, 0), key(1, 1)],
            factor: Some(ColumnIndex(2)),
        },
    ];
    let producer = Fold::try_new(
        vec![EQ; 3],
        ops,
        vec![out(0), out(2)],
        RowValue::Product,
        Aggregate::sum(),
        None,
    )
    .unwrap();
    Pipeline::try_new(
        producer,
        Weight::Exp {
            scale,
            shift: Shift::KeptMaximum,
        },
        vec![],
        Operand {
            keys: vec![key(2, 0)],
            factor: Some(ColumnIndex(1)),
        },
        vec![DimensionId(0)],
        Normalizer::KeptGroups,
    )
    .unwrap()
}

/// Checks, for each row of offered attention, that the three orders agree.
#[derive(Debug, Default)]
struct Orders {
    offered: AtomicUsize,
    disagreements: Mutex<Vec<String>>,
}

impl DenseBackend for Orders {
    fn run(&self, work: DenseWork<'_>) -> einfold_kernels::Result<Option<Vec<f64>>> {
        let DenseWork::Attention {
            q,
            kt,
            v,
            shape,
            scale,
            allowed,
        } = work
        else {
            return Ok(None);
        };
        self.offered.fetch_add(1, Ordering::Relaxed);
        let (nx, ne, nj, ny) = (shape.queries, shape.features, shape.keys, shape.values);
        assert_eq!(shape.batches, 1);
        let mut out = vec![0.0; nx * ny];
        for x in 0..nx {
            let (lo, hi) = allowed[x];
            let s: Vec<f64> = (lo..hi)
                .map(|j| scale.apply((0..ne).map(|e| q[x * ne + e] * kt[e * nj + j]).sum()))
                .collect();
            let m = s.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let w: Vec<f64> = s.iter().map(|s| (s - m).exp()).collect();
            let z: f64 = w.iter().sum();
            for y in 0..ny {
                let vj = |i: usize| v[(lo + i) * ny + y];
                let first: f64 = (0..w.len()).map(|i| w[i] * vj(i)).sum::<f64>() / z;
                let per_term: f64 = (0..w.len()).map(|i| w[i] / z * vj(i)).sum();
                let softmax: f64 = (0..w.len()).rev().map(|i| (w[i] / z) * vj(i)).sum();
                // The row's magnitude, computed so as not to underflow: each
                // term's |w/z · v| in logs.
                let magnitude: f64 = (0..w.len())
                    .map(|i| (w[i].ln() - z.ln() + vj(i).abs().ln()).exp())
                    .sum();
                let tol = 4.0 * (w.len() as f64 + 2.0) * f64::EPSILON * magnitude;
                let close = |a: f64, b: f64| (a - b).abs() <= tol;
                if !(close(first, per_term) && close(first, softmax)) {
                    self.disagreements.lock().unwrap().push(format!(
                        "row {x}: multiply-first {first:e}, per-term {per_term:e}, \
                         softmax-first {softmax:e}, magnitude {magnitude:e}, \
                         weights {w:?}, z {z:e}, values {:?}",
                        (0..w.len()).map(vj).collect::<Vec<_>>()
                    ));
                }
                out[x * ny + y] = softmax;
            }
        }
        Ok(Some(out))
    }
}

/// A number of magnitude about 2^e, of a random sign.
fn about(rng: &mut Rng, e: f64) -> f64 {
    let sign = if rng.chance(0.5) { -1.0 } else { 1.0 };
    sign * (1.0 + rng.unit()) * e.exp2()
}

#[test]
fn offered_attention_is_order_independent() {
    let seed: u64 = std::env::var("EINFOLD_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xadd5_0001);
    let cases: u64 = std::env::var("EINFOLD_CASES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3000);
    let (mut offered, mut cases_run) = (0, 0);
    for i in 0..cases {
        let seed = seed.wrapping_add(i);
        let mut rng = Rng::seeded(seed, 0);
        let (nq, nk, nd) = (1 + rng.below(4), 2 + rng.below(80), 1 + rng.below(2));
        // Scores up to a bound near the guard's limit, about 354.
        let reach = [1.0, 50.0, 200.0, 340.0, 354.0, 360.0, 368.0, 372.0][rng.below(8)];
        let each = (reach / nd as f64).sqrt();
        // Queries at the edge of their range, so that scores reach the bound.
        let qv: Vec<f64> = (0..nq * nd)
            .map(|_| if rng.chance(0.5) { each } else { -each })
            .collect();
        // Keys often at the extremes, so several tie for the largest score
        // (a normalizer above 1) and some reach the smallest.
        let kv: Vec<f64> = (0..nk * nd)
            .map(|_| match rng.below(4) {
                0 => each,
                1 => -each,
                _ => (2.0 * rng.unit() - 1.0) * each,
            })
            .collect();
        // Values of magnitudes across the double range, some rows' close to
        // the largest double or the smallest normal one.
        let (lo, hi) = match rng.below(4) {
            0 => (-1021.0, 1023.0),
            1 => (-1021.0, -900.0),
            2 => (900.0, 1023.0 - (nk as f64).log2() - 1.5),
            _ => (-10.0, 10.0),
        };
        let vv: Vec<f64> = (0..nk)
            .map(|_| match rng.chance(0.05) {
                true => 0.0,
                false => {
                    let e = lo + (hi - lo) * rng.unit();
                    about(&mut rng, e)
                }
            })
            .collect();
        // Half the time, values against the scores: the lower a key's score
        // with query 0, the larger its value, so the lightest weights carry
        // the largest values, where underflow would matter.
        let mut vv = vv;
        if rng.chance(0.5) {
            let s: Vec<f64> = (0..nk)
                .map(|j| (0..nd).map(|e| qv[e] * kv[j * nd + e]).sum())
                .collect();
            let (min, max) = s
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), &x| {
                    (a.min(x), b.max(x))
                });
            let top = 1023.0 - (nk as f64).log2() - 2.0;
            for j in 0..nk {
                let t = if max > min {
                    (s[j] - min) / (max - min)
                } else {
                    0.5
                };
                vv[j] = about(&mut rng, top - (top + 1021.0) * t);
            }
        }
        let scale = [Scale::One, Scale::Over(8.0), Scale::Times(-0.7)][rng.below(3)];
        let pipeline = attention(scale);
        let (sq, bq) = table(&["t", "d"], &[nq, nd], &qv);
        let (sk, bk) = table(&["t", "d"], &[nk, nd], &kv);
        let (sv, bv) = table(&["t"], &[nk], &vv);
        let schema = einfold_kernels::pipeline_schema(&pipeline, &[&sq, &sk, &sv]).unwrap();
        let orders = Orders::default();
        let (bq, bk, bv) = ([bq], [bk], [bv]);
        let inputs = [
            Relation::new(&sq, &bq),
            Relation::new(&sk, &bk),
            Relation::new(&sv, &bv),
        ];
        let options = Options {
            backend: Some(&orders),
            ..Options::default()
        };
        let _ = einfold_kernels::pipeline(&pipeline, inputs, schema, options).unwrap();
        cases_run += 1;
        offered += orders.offered.load(Ordering::Relaxed);
        let d = orders.disagreements.into_inner().unwrap();
        assert!(
            d.is_empty(),
            "seed {seed}: offered attention whose orders disagree: {d:#?}"
        );
    }
    eprintln!("offered {offered} of {cases_run} cases");
    assert!(offered * 8 > cases_run as usize, "offered only {offered}");
}

/// Pins: a float32 backend, as MLX's, is offered a product whose result
/// SQL computes as 2e40, and answers infinity: the guards are in float64's
/// range, and float32's is narrower.
#[test]
fn a_float32_backend_turns_a_finite_product_infinite() {
    const EQ: Dimension = Dimension {
        equality: KeyEquality::Equal,
    };
    let key = |d, c| Key::new(DimensionId(d), ColumnIndex(c));
    let out = |d| Output {
        dimension: DimensionId(d),
        domain_key: None,
    };
    // SELECT a.i, b.j, SUM(a.x * b.x) FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j
    let fold = Fold::try_new(
        vec![EQ; 3],
        vec![
            Operand {
                keys: vec![key(0, 0), key(1, 1)],
                factor: Some(ColumnIndex(2)),
            },
            Operand {
                keys: vec![key(1, 0), key(2, 1)],
                factor: Some(ColumnIndex(2)),
            },
        ],
        vec![out(0), out(2)],
        RowValue::Product,
        Aggregate::sum(),
        None,
    )
    .unwrap();
    let (sa, ba) = table(&["i", "k"], &[2, 2], &[1e20, 1e20, 1.0, 1.0]);
    let (sb, bb) = table(&["k", "j"], &[2, 2], &[1e20, 1.0, 1e20, 1.0]);
    let schema = einfold_kernels::schema(&fold, &[&sa, &sb], None).unwrap();
    let f32 = reference::Float32::default();
    let options = Options {
        backend: Some(&f32),
        ..Options::default()
    };
    let (ba, bb) = ([ba], [bb]);
    let folded = einfold_kernels::fold(
        &fold,
        Relation::new(&sa, &ba),
        Relation::new(&sb, &bb),
        None,
        schema,
        options,
    )
    .unwrap();
    let values: Vec<f64> = (folded.batches.iter())
        .flat_map(|b| {
            let c = b.column(b.num_columns() - 1);
            let c = c.as_any().downcast_ref::<Float64Array>().unwrap();
            c.values().to_vec()
        })
        .collect();
    assert!(f32.0.ran.load(Ordering::Relaxed) > 0, "not offered");
    // SQL: (0, 0) = 2e40, finite. The float32 backend: infinity.
    assert!(values.contains(&f64::INFINITY), "{values:?}");
}

/// A product whose partial sums overflow in einfold's order but not in the
/// backend's: SQL's answer is infinite, so einfold must not offer it.
#[tokio::test]
async fn a_product_that_overflows_in_one_order_is_not_offered() {
    const EQ: Dimension = Dimension {
        equality: KeyEquality::Equal,
    };
    let key = |d, c| Key::new(DimensionId(d), ColumnIndex(c));
    let out = |d| Output {
        dimension: DimensionId(d),
        domain_key: None,
    };
    let fold = Fold::try_new(
        vec![EQ; 3],
        vec![
            Operand {
                keys: vec![key(0, 0), key(1, 1)],
                factor: Some(ColumnIndex(2)),
            },
            Operand {
                keys: vec![key(1, 0), key(2, 1)],
                factor: Some(ColumnIndex(2)),
            },
        ],
        vec![out(0), out(2)],
        RowValue::Product,
        Aggregate::sum(),
        None,
    )
    .unwrap();
    // One output cell: 1e308 + 1e308 - 1e308 over k = 0, 1, 2.
    let (sa, ba) = table(&["i", "k"], &[1, 3], &[1e308, 1e308, -1e308]);
    let (sb, bb) = table(&["k", "j"], &[3, 1], &[1.0, 1.0, 1.0]);
    let schema = einfold_kernels::schema(&fold, &[&sa, &sb], None).unwrap();
    let run = |backend: Option<&dyn DenseBackend>| {
        let (ba, bb) = ([ba.clone()], [bb.clone()]);
        let f = einfold_kernels::fold(
            &fold,
            Relation::new(&sa, &ba),
            Relation::new(&sb, &bb),
            None,
            Arc::clone(&schema),
            Options {
                backend,
                ..Options::default()
            },
        )
        .unwrap();
        let b = &f.batches[0];
        let c = b.column(b.num_columns() - 1);
        c.as_any().downcast_ref::<Float64Array>().unwrap().value(0)
    };
    let r = reference::Reference::default();
    let own = run(None);
    let hosted = run(Some(&r));
    let ctx = datafusion::prelude::SessionContext::new();
    ctx.register_batch("a", ba.clone()).unwrap();
    ctx.register_batch("b", bb.clone()).unwrap();
    let sql = ctx
        .sql("SELECT SUM(a.x * b.x) FROM a JOIN b ON a.k = b.k GROUP BY a.i, b.j")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let sql = sql[0].column(0).as_any().downcast_ref::<Float64Array>();
    let sql = sql.unwrap().value(0);
    assert_eq!(r.ran.load(Ordering::Relaxed), 0, "offered");
    assert!(
        own.is_infinite() && sql.is_infinite(),
        "own {own:e} sql {sql:e}"
    );
    assert_eq!(hosted.to_bits(), own.to_bits());
}
