//! Run-time feature detection on MIPS64.

use super::bit;
use super::cache;
use super::linux;

#[macro_export]
#[unstable(feature = "stdsimd", issue = "0")]
macro_rules! is_mips64_feature_detected {
    ("msa") => {
        cfg!(target_feature = "msa") ||
            $crate::arch::detect::check_for($crate::arch::detect::Feature::msa)
    };
    ($t:tt) => { compile_error!(concat!("unknown mips64 target feature: ", $t)) };
}

/// MIPS64 CPU Feature enum. Each variant denotes a position in a bitset
/// for a particular feature.
///
/// PLEASE: do not use this, it is an implementation detail subject to change.
#[doc(hidden)]
#[allow(non_camel_case_types)]
#[repr(u8)]
pub enum Feature {
    /// MIPS SIMD Architecture (MSA)
    msa,
}

pub fn detect_features() -> cache::Initializer {
    let mut value = cache::Initializer::default();
    fill_features(&mut value);
    return value
}

fn fill_features(value: &mut cache::Initializer) {
    let mut enable_feature = |f, enable| {
        if enable {
            value.set(f as u32);
        }
    };

    // The values are part of the platform-specific [asm/cputable.h][cputable]
    //
    // [cputable]: https://github.com/torvalds/linux/blob/master/arch/mips/include/uapi/asm/hwcap.h
    if let Ok(auxv) = linux::auxv() {
        enable_feature(Feature::msa, bit::test(auxv.hwcap, 1));
        return
    }

    // TODO: fall back via cpuinfo
}
