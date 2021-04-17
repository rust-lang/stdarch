//! ARM intrinsics.
//!
//! The reference for NEON is [ARM's NEON Intrinsics Reference][arm_ref]. The
//! [ARM's NEON Intrinsics Online Database][arm_dat] is also useful.
//!
//! [arm_ref]: http://infocenter.arm.com/help/topic/com.arm.doc.ihi0073a/IHI0073A_arm_neon_intrinsics_ref.pdf
//! [arm_dat]: https://developer.arm.com/technologies/neon/intrinsics

mod armclang;

pub use self::armclang::*;

mod v6;
pub use self::v6::*;

// Supported arches: 6, 7-M. See Section 10.1 of ACLE (e.g. SSAT)
#[cfg(any(target_feature = "v6", doc))]
mod sat;

#[cfg(any(target_feature = "v6", doc))]
pub use self::sat::*;

// Supported arches: 5TE, 7E-M. See Section 10.1 of ACLE (e.g. QADD)
// We also include the A profile even though DSP is deprecated on that profile as of ACLE 2.0 (see
// section 5.4.7)
// Here we workaround the difference between LLVM's +dsp and ACLE's __ARM_FEATURE_DSP by gating on
// '+v5te' rather than on '+dsp'
#[cfg(any(
    // >= v5TE but excludes v7-M
    all(target_feature = "v5te", not(target_feature = "mclass")),
    // v7E-M
    all(target_feature = "mclass", target_feature = "dsp"),
    doc,
))]
pub mod dsp;

#[cfg(any(
    // >= v5TE but excludes v7-M
    all(target_feature = "v5te", not(target_feature = "mclass")),
    // v7E-M
    all(target_feature = "mclass", target_feature = "dsp"),
    doc,
))]
pub use self::dsp::*;

// Deprecated in ACLE 2.0 for the A profile but fully supported on the M and R profiles, says
// Section 5.4.9 of ACLE. We'll expose these for the A profile even if deprecated
#[cfg(any(
    // v7-A, v7-R
    all(target_feature = "v6", not(target_feature = "mclass")),
    // v7E-M
    all(target_feature = "mclass", target_feature = "dsp"),
    doc,
))]
mod simd32;

#[cfg(any(
    // v7-A, v7-R
    all(target_feature = "v6", not(target_feature = "mclass")),
    // v7E-M
    all(target_feature = "mclass", target_feature = "dsp"),
    doc,
))]
pub use self::simd32::*;

#[cfg(any(target_feature = "v7", doc))]
mod v7;
#[cfg(any(target_feature = "v7", doc))]
pub use self::v7::*;

pub use crate::core_arch::acle::*;

#[cfg(test)]
use stdarch_test::assert_instr;

#[cfg(any(target_feature = "v7", doc))]
pub(crate) mod neon;
#[cfg(any(target_feature = "v7", doc))]
pub use neon::*;

/// Generates the trap instruction `UDF`
#[cfg(target_arch = "arm")]
#[cfg_attr(test, assert_instr(udf))]
#[inline]
pub unsafe fn udf() -> ! {
    crate::intrinsics::abort()
}
