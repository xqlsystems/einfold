// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Self-tests: the two oracles agree, which validates the harness and pins
//! down DataFusion's behavior on NULL keys and NaN.

use einfold_testkit::{assert_same_result, case_seed, check, naive_reference, sql_reference, Case};

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
