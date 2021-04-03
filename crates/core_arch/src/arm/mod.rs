//! ARM intrinsics.
//!
//! The reference for NEON is [ARM's NEON Intrinsics Reference][arm_ref]. The
//! [ARM's NEON Intrinsics Online Database][arm_dat] is also useful.
//!
//! [arm_ref]: http://infocenter.arm.com/help/topic/com.arm.doc.ihi0073a/IHI0073A_arm_neon_intrinsics_ref.pdf
//! [arm_dat]: https://developer.arm.com/technologies/neon/intrinsics
#![allow(non_camel_case_types)]

mod armclang;

pub use self::armclang::*;

mod v6;
pub use self::v6::*;

// Supported arches: 6, 7-M. See Section 10.1 of ACLE (e.g. SSAT)
#[cfg(any(all(not(target_arch = "aarch64"), target_feature = "v6",), doc))]
mod sat;

#[cfg(any(all(not(target_arch = "aarch64"), target_feature = "v6",), doc))]
pub use self::sat::*;

// Supported arches: 5TE, 7E-M. See Section 10.1 of ACLE (e.g. QADD)
// We also include the A profile even though DSP is deprecated on that profile as of ACLE 2.0 (see
// section 5.4.7)
// Here we workaround the difference between LLVM's +dsp and ACLE's __ARM_FEATURE_DSP by gating on
// '+v5te' rather than on '+dsp'
#[cfg(any(all(
    not(target_arch = "aarch64"),
    any(
        // >= v5TE but excludes v7-M
        all(target_feature = "v5te", not(target_feature = "mclass")),
        // v7E-M
        all(target_feature = "mclass", target_feature = "dsp"),
    )
), doc))]
mod dsp;

#[cfg(any(
    all(
        not(target_arch = "aarch64"),
        any(
            all(target_feature = "v5te", not(target_feature = "mclass")),
            all(target_feature = "mclass", target_feature = "dsp"),
        )
    ),
    doc
))]
pub use self::dsp::*;

#[cfg(any(target_arch = "aarch64", target_feature = "v7"))]
mod v7;
#[cfg(any(target_arch = "aarch64", target_feature = "v7"))]
pub use self::v7::*;

#[cfg(any(target_arch = "aarch64", target_feature = "v7", doc))]
mod neon;
#[cfg(any(target_arch = "aarch64", target_feature = "v7", doc))]
pub use self::neon::*;

pub use crate::core_arch::acle::*;

#[cfg(test)]
use stdarch_test::assert_instr;

/// Generates the trap instruction `UDF`
#[cfg(target_arch = "arm")]
#[cfg_attr(test, assert_instr(udf))]
#[inline]
pub unsafe fn udf() -> ! {
    crate::intrinsics::abort()
}

#[cfg(test)]
#[cfg(any(target_arch = "aarch64", target_feature = "v7"))]
pub(crate) mod test_support;
