// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Self-tests: the two oracles agree, which validates the harness and pins
//! down DataFusion's behavior on NULL keys and NaN.

use einfold_testkit::{assert_same_result, case_seed, check, naive_reference, sql_reference, Case};

/// Both oracles agree on `n` cases of `operands` operands, from `seed`.
fn oracles_agree(n: usize, seed: u64, operands: usize) {
    for i in 0..n {
        let cs = case_seed(seed, i);
        let case = Case::generate_with(cs, operands);
        if let Err(d) = einfold_testkit::compare(&sql_reference(&case), &naive_reference(&case)) {
            panic!("case {i} (case seed {cs}, {operands} operands): naive != sql\n{case}\n{d}");
        }
    }
}

#[test]
fn naive_matches_sql_on_two_operands() {
    oracles_agree(600, 1, 2);
}

#[test]
fn naive_matches_sql_on_one_and_three_operands() {
    oracles_agree(150, 2, 1);
    oracles_agree(150, 3, 3);
}

#[test]
#[ignore = "large run"]
fn naive_matches_sql_large() {
    oracles_agree(20_000, 99, 2);
    oracles_agree(5_000, 98, 3);
}

#[test]
fn check_accepts_a_correct_implementation() {
    check(50, 7, naive_reference);
}

#[test]
#[should_panic(expected = "case seed")]
fn check_rejects_a_wrong_implementation() {
    check(200, 7, |case| {
        let r = naive_reference(case);
        r.slice(0, r.num_rows().saturating_sub(1))
    });
}

#[test]
fn generator_is_deterministic_and_varied() {
    let a = Case::generate(5);
    assert_eq!(a.sql, Case::generate(5).sql);
    assert_eq!(a.tables, Case::generate(5).tables);
    let sqls: std::collections::HashSet<_> = (0..50).map(|s| Case::generate(s).sql).collect();
    assert!(sqls.len() > 20);
    let nulls = |c: &Case| {
        c.tables
            .iter()
            .any(|t| t.columns().iter().any(|c| c.null_count() > 0))
    };
    assert!((0..50).any(|s| nulls(&Case::generate(s))));
    assert!((0..50).any(|s| Case::generate(s).tables.iter().any(|t| t.num_rows() == 0)));
}

#[test]
fn all_three_aggregates_are_generated_with_their_sql_types() {
    use einfold_ir::Aggregate;
    for agg in [Aggregate::SUM, Aggregate::COUNT, Aggregate::AVG] {
        let case = (0..200)
            .map(Case::generate)
            .find(|c| c.fold.aggregate() == agg)
            .expect("generated");
        let ty = sql_reference(&case)
            .schema()
            .fields()
            .last()
            .unwrap()
            .data_type()
            .clone();
        let want = if agg == Aggregate::COUNT {
            "Int64"
        } else {
            "Float64"
        };
        assert_eq!(ty.to_string(), want, "{agg}");
        assert_eq!(
            naive_reference(&case)
                .schema()
                .fields()
                .last()
                .unwrap()
                .data_type(),
            &ty
        );
    }
}

#[test]
fn count_of_all_null_group_is_zero_and_not_a_float() {
    use datafusion::arrow::array::{Float64Array, Int64Array};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::arrow::record_batch::RecordBatch;
    use std::sync::Arc;
    let mk = |count: bool| {
        let (ty, col): (_, datafusion::arrow::array::ArrayRef) = if count {
            (DataType::Int64, Arc::new(Int64Array::from(vec![0])))
        } else {
            (DataType::Float64, Arc::new(Float64Array::from(vec![0.0])))
        };
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("v", ty, true)])),
            vec![col],
        )
        .unwrap()
    };
    assert!(einfold_testkit::compare(&mk(true), &mk(true)).is_ok());
    assert!(einfold_testkit::compare(&mk(true), &mk(false)).is_err());
}

#[test]
fn comparison_semantics() {
    let case = Case::generate(11);
    let r = naive_reference(&case);
    assert_same_result(&r, &r);
    // Order does not matter.
    if r.num_rows() > 1 {
        let n = r.num_rows();
        let idx =
            datafusion::arrow::array::UInt32Array::from((0..n as u32).rev().collect::<Vec<_>>());
        let rev = datafusion::arrow::compute::take_record_batch(&r, &idx).unwrap();
        assert_same_result(&r, &rev);
    }
}
