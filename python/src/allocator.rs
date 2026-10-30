// SPDX-FileCopyrightText: 2026 Alexander Merose <al@merose.com> & einfold Authors
//
// SPDX-License-Identifier: Apache-2.0

//! The allocator this library allocates with: [`Split`], mimalloc for small
//! blocks and the system's allocator for large ones.
//!
//! mimalloc, Microsoft's general-purpose allocator, which datafusion-python
//! uses too, keeps small allocations from fragmenting across the threads
//! einfold computes on: with the system allocator alone, a session that ran
//! many queries grew by about 150 MB per query. It's mimalloc's version 2
//! (see `Cargo.toml`). But version 2 keeps large freed blocks for reuse,
//! which raised the peak of a query that allocates a few large buffers (the
//! dense copies of its inputs, its output's columns) by about 0.1 GB, at
//! 32,768 tokens of attention (#119). The system allocator gives a large
//! block back to the operating system when it's freed, so large blocks go
//! there. They're few, so the system calls cost no measurable time.

use std::alloc::{GlobalAlloc, Layout, System};

use mimalloc::MiMalloc;

/// The size from which a block is large: [`Split`] allocates it with the
/// system's allocator. On Linux, glibc maps blocks this large directly, and
/// unmaps them when they're freed (see [`pin_mmap_threshold`]). At 1 MiB,
/// blocks that a session allocates again and again went there too, and it
/// grew by more over many queries than with mimalloc alone; at 8 MiB it
/// grows by less.
const LARGE: usize = 8 << 20;

/// mimalloc for blocks smaller than [`LARGE`], and the system's allocator
/// for the rest. Each block is freed by the allocator that allocated it,
/// which its size tells.
pub(crate) struct Split;

impl Split {
    fn large(layout: &Layout) -> bool {
        layout.size() >= LARGE
    }
}

// SAFETY: every method hands the request to one of two allocators that
// uphold `GlobalAlloc`'s contract, chosen by the block's size, which is the
// same when a block is freed as when it was allocated. `realloc` moves a
// block between them by allocating, copying and freeing.
unsafe impl GlobalAlloc for Split {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's guarantees for `layout` pass through.
        unsafe {
            match Split::large(&layout) {
                true => System.alloc(layout),
                false => MiMalloc.alloc(layout),
            }
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's guarantees for `layout` pass through.
        unsafe {
            match Split::large(&layout) {
                true => System.alloc_zeroed(layout),
                false => MiMalloc.alloc_zeroed(layout),
            }
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` was allocated with `layout`, so by the allocator
        // its size chooses.
        unsafe {
            match Split::large(&layout) {
                true => System.dealloc(ptr, layout),
                false => MiMalloc.dealloc(ptr, layout),
            }
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: `ptr` was allocated with `layout`; `new_size` is valid
        // for its alignment, as the caller guarantees.
        unsafe {
            let new = Layout::from_size_align_unchecked(new_size, layout.align());
            match (Split::large(&layout), Split::large(&new)) {
                (true, true) => System.realloc(ptr, layout, new_size),
                (false, false) => MiMalloc.realloc(ptr, layout, new_size),
                // Across the threshold: into the other allocator.
                _ => {
                    let moved = self.alloc(new);
                    if !moved.is_null() {
                        std::ptr::copy_nonoverlapping(ptr, moved, layout.size().min(new_size));
                        self.dealloc(ptr, layout);
                    }
                    moved
                }
            }
        }
    }
}

#[global_allocator]
static ALLOCATOR: Split = Split;

/// Fixes glibc's threshold for mapping a block directly at [`LARGE`].
/// glibc raises the threshold each time a mapped block is freed, up to
/// 32 MiB, and then serves large blocks from its heap, which keeps them; a
/// fixed threshold keeps every large block mapped, so freeing it returns it.
#[cfg(target_os = "linux")]
pub(crate) fn pin_mmap_threshold() {
    // SAFETY: mallopt takes an option and a value, and only sets glibc's
    // allocator's parameters.
    unsafe {
        libc::mallopt(libc::M_MMAP_THRESHOLD, LARGE as libc::c_int);
    }
}

/// Elsewhere the system's allocator returns large blocks of its own accord.
#[cfg(not(target_os = "linux"))]
pub(crate) fn pin_mmap_threshold() {}
