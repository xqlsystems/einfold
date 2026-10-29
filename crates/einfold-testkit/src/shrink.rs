// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Shrinking: turning a failing case into a small one that fails the same
//! way.
//!
//! A random case that breaks a property is usually large: thousands of rows,
//! several tables, settings that don't matter. Shrinking tries simpler
//! versions of it, keeps any that still fails, and repeats until no simpler
//! version fails or a budget runs out. "Fails the same way" means the same
//! property: each failure message starts with its property's tag in
//! brackets, such as `[rows]`, and a simpler case counts only if it fails
//! with the same tag. Otherwise a case that breaks `[rows]` could shrink into
//! one that breaks something else, and the bug being chased would be lost.
//!
//! Shrinking is greedy. Each round takes the first simpler version that still
//! fails and starts again from it. A case's [`Shrink::candidates`] lists the
//! biggest simplifications first, such as removing half of a table's rows
//! before removing one, so a large case loses most of its size in a few
//! rounds.

use std::cell::Cell;
use std::ops::Range;
use std::panic;
use std::sync::Once;
use std::time::{Duration, Instant};

/// A case that can propose simpler versions of itself.
pub trait Shrink: Sized {
    /// Simpler versions of `self`, the biggest simplifications first.
    ///
    /// Each must be smaller than `self` by some measure that no candidate
    /// increases, such as its number of rows, then its number of values that
    /// aren't 1, so that shrinking ends.
    fn candidates(&self) -> Box<dyn Iterator<Item = Self> + '_>;
}

/// How long shrinking may go on.
#[derive(Clone, Copy, Debug)]
pub struct Budget {
    /// How many candidates it may try.
    pub tries: usize,
    /// How long it may take, checked before each try.
    pub time: Duration,
}

impl Budget {
    /// `EINFOLD_SHRINK_TRIES` candidates (default `tries`; 0 turns shrinking
    /// off) within `EINFOLD_SHRINK_SECS` seconds (default `secs`).
    #[must_use]
    pub fn from_env(tries: usize, secs: u64) -> Self {
        let env = |name: &str| std::env::var(name).ok().and_then(|s| s.parse::<u64>().ok());
        Budget {
            tries: env("EINFOLD_SHRINK_TRIES").map_or(tries, |n| n as usize),
            time: Duration::from_secs(env("EINFOLD_SHRINK_SECS").unwrap_or(secs)),
        }
    }
}

/// A shrunk case.
#[derive(Debug)]
pub struct Shrunk<T> {
    /// The smallest failing case found.
    pub case: T,
    /// Its failure, which has the original failure's tag.
    pub failure: String,
    /// How many simplifications were kept.
    pub steps: usize,
    /// How many candidates were tried.
    pub tries: usize,
    /// Whether the budget ran out before every candidate of `case` passed.
    pub exhausted: bool,
}

/// The property tag a failure starts with, such as `[rows]`, or `""` if it
/// starts with none.
#[must_use]
pub fn tag(failure: &str) -> &str {
    match (failure.starts_with('['), failure.find(']')) {
        (true, Some(end)) => &failure[..=end],
        _ => "",
    }
}

/// Shrinks `case`, which failed with `failure`: `fails` runs the properties
/// on a candidate and returns its failure, if any. A candidate is kept only
/// if its failure has `failure`'s tag.
///
/// Wrap a `fails` that can panic in [`failure_of`], so that a panic counts
/// as a failure. While shrinking, panics on this thread aren't printed, as
/// candidates that panic are expected.
pub fn shrink<T: Shrink>(
    case: T,
    failure: String,
    budget: Budget,
    mut fails: impl FnMut(&T) -> Option<String>,
) -> Shrunk<T> {
    let _quiet = Quiet::new();
    let start = Instant::now();
    let want = tag(&failure).to_string();
    let mut shrunk = Shrunk {
        case,
        failure,
        steps: 0,
        tries: 0,
        exhausted: false,
    };
    'rounds: loop {
        let mut kept = None;
        for candidate in shrunk.case.candidates() {
            if shrunk.tries >= budget.tries || start.elapsed() >= budget.time {
                shrunk.exhausted = true;
                break 'rounds;
            }
            shrunk.tries += 1;
            if let Some(f) = fails(&candidate) {
                if tag(&f) == want {
                    kept = Some((candidate, f));
                    break;
                }
            }
        }
        match kept {
            Some((candidate, f)) => {
                shrunk.case = candidate;
                shrunk.failure = f;
                shrunk.steps += 1;
            }
            None => break,
        }
    }
    shrunk
}

/// `run`'s failure: its error, or a panic's message tagged `[panic]`.
pub fn failure_of(run: impl FnOnce() -> Result<(), String>) -> Option<String> {
    match panic::catch_unwind(panic::AssertUnwindSafe(run)) {
        Ok(result) => result.err(),
        Err(payload) => {
            let message = (payload.downcast_ref::<&str>().map(|s| s.to_string()))
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "a panic without a message".to_string());
            Some(format!("[panic] {message}"))
        }
    }
}

/// The ranges of `0..n` to try removing, biggest first: all of it, then each
/// half, each quarter, and so on down to each single index. Removing a range
/// of a table's rows this way finds, in few tries, the rows a failure needs.
pub fn removals(n: usize) -> impl Iterator<Item = Range<usize>> {
    let sizes = std::iter::successors((n > 0).then_some(n), |&k| (k > 1).then_some(k.div_ceil(2)));
    sizes.flat_map(move |k| (0..n.div_ceil(k)).map(move |i| i * k..((i + 1) * k).min(n)))
}

/// `xs` without the elements in `range`.
#[must_use]
pub fn without<T: Clone>(xs: &[T], range: Range<usize>) -> Vec<T> {
    let mut out = xs[..range.start].to_vec();
    out.extend_from_slice(&xs[range.end..]);
    out
}

thread_local! {
    static QUIET: Cell<bool> = const { Cell::new(false) };
}

/// While it lives, panics on this thread aren't printed. The panic hook is
/// process-wide, so it's installed once and asks each thread whether to
/// print, rather than swapped in and out by tests running in parallel.
struct Quiet {
    was: bool,
}

impl Quiet {
    fn new() -> Self {
        static HOOK: Once = Once::new();
        HOOK.call_once(|| {
            let print = panic::take_hook();
            panic::set_hook(Box::new(move |info| {
                if !QUIET.with(Cell::get) {
                    print(info);
                }
            }));
        });
        Quiet {
            was: QUIET.with(|q| q.replace(true)),
        }
    }
}

impl Drop for Quiet {
    fn drop(&mut self) {
        QUIET.with(|q| q.set(self.was));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Rng;

    /// A list of numbers, shrunk by removing ranges and then by halving one
    /// number.
    #[derive(Clone, Debug, PartialEq)]
    struct Numbers(Vec<u32>);

    impl Shrink for Numbers {
        fn candidates(&self) -> Box<dyn Iterator<Item = Self> + '_> {
            let removed = removals(self.0.len()).map(|r| Numbers(without(&self.0, r)));
            let halved = (0..self.0.len()).filter(|&i| self.0[i] > 0).map(|i| {
                let mut xs = self.0.clone();
                xs[i] /= 2;
                Numbers(xs)
            });
            Box::new(removed.chain(halved))
        }
    }

    fn unlimited() -> Budget {
        Budget {
            tries: usize::MAX,
            time: Duration::from_secs(600),
        }
    }

    #[test]
    fn removals_cover_every_single_index() {
        for n in 0..40 {
            let ranges: Vec<_> = removals(n).collect();
            for i in 0..n {
                assert!(ranges.contains(&(i..i + 1)), "n = {n}: no removal of {i}");
            }
            assert!(ranges.iter().all(|r| r.start < r.end && r.end <= n));
            assert_eq!(ranges.first(), (n > 0).then_some(&(0..n)));
        }
    }

    /// Without a budget, a shrunk case still fails with the original tag, and
    /// it's 1-minimal: none of its candidates fails with that tag.
    #[test]
    fn shrunk_cases_fail_the_same_way_and_are_minimal() {
        for seed in 0..200 {
            let mut rng = Rng::seeded(seed, 0x5eed);
            let len = rng.below(60);
            let xs: Vec<u32> = (0..len).map(|_| rng.below(1000) as u32).collect();
            let threshold = 1 + rng.below(5000) as u32;
            let odd = rng.below(4) as u32;
            // Two properties: a sum too large, and an element of a residue.
            let fails = |c: &Numbers| {
                if c.0.iter().sum::<u32>() >= threshold {
                    Some(format!("[sum] {}", c.0.iter().sum::<u32>()))
                } else if c.0.iter().any(|&x| x % 7 == odd + 3) {
                    Some("[residue]".to_string())
                } else {
                    None
                }
            };
            let Some(failure) = fails(&Numbers(xs.clone())) else {
                continue;
            };
            let shrunk = shrink(Numbers(xs), failure.clone(), unlimited(), fails);
            assert!(!shrunk.exhausted);
            let again = fails(&shrunk.case).expect("the shrunk case fails");
            assert_eq!(tag(&again), tag(&failure), "seed {seed}");
            for c in shrunk.case.candidates() {
                assert!(
                    fails(&c).is_none_or(|f| tag(&f) != tag(&failure)),
                    "seed {seed}: {c:?}, simpler than {:?}, fails the same way",
                    shrunk.case
                );
            }
        }
    }

    #[test]
    fn shrinking_stops_at_the_budget() {
        let xs = Numbers((0..1000).collect());
        let budget = Budget {
            tries: 17,
            time: Duration::from_secs(600),
        };
        // Every candidate fails another way, so none is kept and all are tried.
        let shrunk = shrink(xs, "[any]".into(), budget, |_| Some("[other]".into()));
        assert_eq!(shrunk.tries, 17);
        assert!(shrunk.exhausted);
    }

    #[test]
    fn panics_are_failures() {
        let shrunk = shrink(
            Numbers(vec![5, 9, 12]),
            "[panic] nine".into(),
            unlimited(),
            |c| {
                failure_of(|| {
                    assert!(!c.0.contains(&9), "nine");
                    Ok(())
                })
            },
        );
        assert_eq!(shrunk.case, Numbers(vec![9]));
        assert_eq!(shrunk.failure, "[panic] nine");
    }

    #[test]
    fn tags() {
        assert_eq!(tag("[rows] 3 rows"), "[rows]");
        assert_eq!(tag("no tag [rows]"), "");
        assert_eq!(tag("[unclosed"), "");
    }
}
