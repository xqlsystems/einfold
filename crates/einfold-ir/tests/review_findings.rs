// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Adversarial review of `einfold-ir`: each test pins a claim the docs make
//! that is false on some input. Each asserts the *current* (wrong) behavior,
//! so it passes today and fails once the claim is fixed, as
//! `AGENTS.md` asks of bugs in dependencies.

use einfold_ir::{Aggregate, Op, PartialAggregate, Semiring, Value, ValueType};
use Value::{Float, Int};

/// `algebra.rs` says: "In every semiring here, ⊕'s identity, the semiring's
/// 'zero', also annihilates: 0 ⊗ a = 0. That is what lets a dense kernel pad
/// absent entries with the zero." For `MIN_PLUS` over integers the zero is
/// `i64::MAX`, and integer `+` wraps, so padding turns into the smallest
/// integer and wins every `MIN`. The test `zero_annihilates` only tries floats.
#[test]
fn integer_min_plus_zero_does_not_annihilate() {
    let zero = Op::Min.identity(ValueType::Int).unwrap();
    assert_eq!(zero, Int(i64::MAX));
    // zero ⊗ 1 should be zero.
    let padded = Op::Add.combine(zero, Int(1)).unwrap();
    assert_eq!(padded, Int(i64::MIN)); // wrapped: NOT the zero
    // So a shortest-path step through an absent edge beats every real path.
    let real_path = Int(7);
    assert_eq!(Op::Min.combine(real_path, padded), Some(Int(i64::MIN)));
    // `Semiring::MIN_PLUS` happily claims to be this semiring over Int.
    assert_eq!(Semiring::MIN_PLUS.add(), Op::Min);
}

/// `combine` docs: "NaN is above every number". `min`/`max` use `total_cmp`,
/// which orders NaNs by their sign bit: a NaN with the sign bit set (what
/// x86 produces for `inf + -inf`, `0.0 / 0.0`) sorts *below* every number.
/// So `MAX` drops it, and the same query gives different answers on x86 and
/// on ARM (whose default NaN is positive). SQL has one NaN.
#[test]
fn nan_ordering_depends_on_the_sign_bit() {
    let neg_nan = f64::from_bits(f64::NAN.to_bits() | (1 << 63));
    assert!(neg_nan.is_nan());
    assert_eq!(Op::Max.combine(Float(1.0), Float(neg_nan)), Some(Float(1.0)));
    let Some(Float(m)) = Op::Max.combine(Float(1.0), Float(f64::NAN)) else {
        unreachable!()
    };
    assert!(m.is_nan());
    // Tropical distributivity then depends on the platform's default NaN:
    // a + min(b, c) at a = inf, b = -inf, c = 5 is `NaN` on the left and
    // min(NaN, inf) on the right, which is inf if that NaN is positive.
    let default_nan_is_negative = (f64::INFINITY + f64::NEG_INFINITY).is_sign_negative();
    let rhs = Op::Min
        .combine(Float(f64::INFINITY + f64::NEG_INFINITY), Float(f64::INFINITY))
        .unwrap();
    assert_eq!(matches!(rhs, Float(x) if x.is_nan()), default_nan_is_negative);
}

/// The identity of `+` is `0.0`, but `0.0 + -0.0 == +0.0`. A kernel that
/// starts an accumulator from the identity (the dense kernel the design plans)
/// returns `+0.0` for `SUM(-0.0)`, where SQL returns `-0.0`. The test
/// `identities_are_identities` compares with `==`, which cannot see the sign.
#[test]
fn additive_identity_loses_negative_zero() {
    let e = Op::Add.identity(ValueType::Float).unwrap();
    let Float(r) = Op::Add.combine(e, Float(-0.0)).unwrap() else {
        unreachable!()
    };
    assert!(r.is_sign_positive());
    // The sequential path (no identity) preserves it, so the two disagree.
    let mut p = PartialAggregate::new(Aggregate::SUM);
    p.update(Some(Float(-0.0)));
    let Some(Some(Float(s))) = p.finish() else {
        unreachable!()
    };
    assert!(s.is_sign_negative());
}

/// `AVG` is modelled as `SUM / COUNT` with an exact wrapping `i64` sum. No
/// engine does that: DataFusion's `avg` accumulates integers in `f64`. At
/// the extremes the IR's `AVG` is not the SQL `AVG` of the engine it targets.
#[test]
fn avg_of_large_integers_wraps() {
    let mut p = PartialAggregate::new(Aggregate::AVG);
    p.update(Some(Int(i64::MAX)));
    p.update(Some(Int(i64::MAX)));
    // The true average is 9.22e18. The IR says -1.
    assert_eq!(p.finish(), Some(Some(Float(-1.0))));
}
