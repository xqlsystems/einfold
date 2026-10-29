// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! What einfold's property, fuzz and simulation tests share.
//!
//! Each of those tests draws random cases from a seed and checks properties
//! of them. This crate gives them one seeded generator, so that a seed in a
//! failure replays the same case on any machine. It has no dependencies, and
//! only einfold's tests use it.

pub mod rng;

pub use rng::Rng;
