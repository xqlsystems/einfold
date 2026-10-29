// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! Room on the main thread's stack for a deep plan's decoding.
//!
//! A session hands einfold each logical plan as protobuf, and DataFusion's
//! FFI decodes it here, on the thread that asked for the plan: in Python,
//! usually the main thread. The decoding (datafusion-proto's
//! `try_into_logical_plan`) recurses once per node of the plan, with frames
//! of about 8 KB, and, unlike DataFusion's own walks of a plan, without
//! growing its stack as it goes. On Linux's default 8 MB stack, a chain of
//! about 250 joins overflows it, and the process dies, where DataFusion alone
//! plans about 1,350.
//!
//! einfold has no frame between the session's call and the decoding, so it
//! can't grow the stack there. But Linux grows the main thread's stack on
//! demand, up to its soft limit as it stands when the stack grows: so
//! [`deepen`] raises that limit, as `ulimit -s` would. Linux keeps at least
//! 128 MB free below the main thread's stack, so [`MAIN_STACK`] fits. Other
//! threads' stacks are fixed when they start, so a plan asked for on one is
//! decoded within its size.

/// The soft stack limit [`deepen`] sets: room for about 2,000 joins.
pub(crate) const MAIN_STACK: u64 = 64 << 20;

/// Raises the process's soft stack limit to [`MAIN_STACK`], if it is lower,
/// and the hard limit allows. Processes started afterwards inherit it, as
/// they would a raised `ulimit -s`.
#[cfg(target_os = "linux")]
pub(crate) fn deepen() {
    let mut limit = libc::rlimit {
        rlim_cur: 0,
        rlim_max: 0,
    };
    // SAFETY: `limit` is a valid rlimit for getrlimit to write and
    // setrlimit to read.
    unsafe {
        if libc::getrlimit(libc::RLIMIT_STACK, &mut limit) != 0 {
            return;
        }
        let wanted = MAIN_STACK.min(limit.rlim_max);
        if limit.rlim_cur != libc::RLIM_INFINITY && limit.rlim_cur < wanted {
            limit.rlim_cur = wanted;
            // A refusal leaves the limit as it was.
            libc::setrlimit(libc::RLIMIT_STACK, &limit);
        }
    }
}

/// Elsewhere the main thread's stack doesn't grow past its start.
#[cfg(not(target_os = "linux"))]
pub(crate) fn deepen() {}
