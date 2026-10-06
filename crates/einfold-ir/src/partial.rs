// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Partial aggregates: the state of one output group, computed over part of
//! the input and combined later.
//!
//! An engine rarely sums a group in one go. It sums pieces in parallel, per
//! partition, per chunk, or per input of a join, and combines the pieces at the
//! end. Every piece must carry enough state for the combined result to equal
//! what SQL's `SUM` would have returned over all the rows.

/// The partial state of `SUM` over products for one output group.
///
/// SQL semantics it preserves:
/// - a group exists only if at least one joined row reached it (`matched`);
/// - its `SUM` is NULL if every product that reached it was NULL (`value` is
///   `None` while `matched` is true);
/// - NaN propagates, as SQL's `SUM` over `DOUBLE` does.
///
/// Float addition is not associative (`(a + b) + c` can differ from
/// `a + (b + c)` in the last bits), so the order of [`update`](Self::update)
/// and [`merge`](Self::merge) calls decides the result's last bits. Callers
/// that promise repeatable results must call them in a fixed order.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PartialSum {
    matched: bool,
    value: Option<f64>,
}

impl PartialSum {
    /// The state of a group no row has reached yet.
    pub const EMPTY: PartialSum = PartialSum {
        matched: false,
        value: None,
    };

    /// Add one joined row's product; `None` is a NULL product.
    ///
    /// The row marks the group as reached *before* the NULL test, so a group
    /// reached only by NULL products ends as NULL rather than vanishing.
    pub fn update(&mut self, product: Option<f64>) {
        self.matched = true;
        if let Some(p) = product {
            self.value = Some(self.value.map_or(p, |v| v + p));
        }
    }

    /// Combine another partial state of the same group into this one.
    pub fn merge(&mut self, other: &PartialSum) {
        self.matched |= other.matched;
        if let Some(p) = other.value {
            self.value = Some(self.value.map_or(p, |v| v + p));
        }
    }

    /// Whether any joined row reached the group.
    pub fn matched(&self) -> bool {
        self.matched
    }

    /// The final result: `None` if the group does not exist, `Some(None)` if
    /// its `SUM` is NULL, and `Some(Some(v))` otherwise.
    pub fn finish(&self) -> Option<Option<f64>> {
        self.matched.then_some(self.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unreached_group_does_not_exist() {
        assert_eq!(PartialSum::EMPTY.finish(), None);
    }

    #[test]
    fn group_reached_only_by_nulls_is_null() {
        let mut s = PartialSum::EMPTY;
        s.update(None);
        s.update(None);
        assert_eq!(s.finish(), Some(None));
    }

    #[test]
    fn nulls_are_skipped() {
        let mut s = PartialSum::EMPTY;
        s.update(None);
        s.update(Some(2.0));
        s.update(None);
        s.update(Some(3.0));
        assert_eq!(s.finish(), Some(Some(5.0)));
    }

    #[test]
    fn nan_propagates() {
        let mut s = PartialSum::EMPTY;
        s.update(Some(1.0));
        s.update(Some(f64::NAN));
        assert!(s.finish().unwrap().unwrap().is_nan());
    }

    #[test]
    fn merge_matches_sequential_updates() {
        let mut a = PartialSum::EMPTY;
        a.update(Some(1.0));
        let mut b = PartialSum::EMPTY;
        b.update(None);
        let mut c = PartialSum::EMPTY;
        c.update(Some(4.0));
        let mut m = PartialSum::EMPTY;
        m.merge(&a);
        m.merge(&b);
        m.merge(&c);
        assert_eq!(m.finish(), Some(Some(5.0)));

        let mut only_null = PartialSum::EMPTY;
        only_null.merge(&b);
        assert_eq!(only_null.finish(), Some(None));

        let mut empty = PartialSum::EMPTY;
        empty.merge(&PartialSum::EMPTY);
        assert_eq!(empty.finish(), None);
    }
}
