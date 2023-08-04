//! SVE intrinsics
//!
//! ## Safety in this module
//!
//! Under [`target_feature_11`][] rules, several of these intrinsics are safe to call as long as
//! the caller already declares the necessary target features. In general:
//!
//! - Intrinsics that access memory or handle pointers are `unsafe`.
//!   - Most of these are memory accesses, and are treated just like any other pointer dereference.
//!   - A few, such as the prefetch hints, are not obviously `unsafe`, but perform pointer
//!     arithmetic (similar to [`pointer::offset`]) that might be.
//! - Intrinsics that can produce undefined values are `unsafe`. This is limited to the explicit
//!   `svundef_*` intrinsics. Note that these behave like [`core::mem::uninitialized`], and
//!   might be similarly unsound, but this requires further analysis.
//! - All other intrinsics operate in a well-defined manner, and are safe (subject to target
//!   feature checks).
//!   - This includes intrinsics with the "don't-care" predication strategy (with a `_x` suffix).
//!     As in [ACLE][], the value of inactive lanes is unspecified, but Rust intrinsics always
//!     ensures that they are initialised to _something_.
//!
//! [`target_feature_11`]: https://rust-lang.github.io/rfcs/2396-target-feature-1.1.html
//! [`pointer::offset`]: pointer#method.offset
//! [ACLE]: https://github.com/ARM-software/acle

mod sve;
mod sve2;
mod types;

use crate::core_arch::simd_llvm::*;

pub use sve::*;
pub use sve2::*;
pub use types::*;
