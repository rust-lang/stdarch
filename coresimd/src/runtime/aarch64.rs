//! Run-time feature detection on ARM Aarch64.
use runtime::bit;
use runtime::arch::HasFeature;

#[macro_export]
#[doc(hidden)]
macro_rules! __unstable_detect_feature {
    ("neon") => {
        // FIXME: this should be removed once we rename Aarch64 neon to asimd
        $crate::__vendor_runtime::__unstable_detect_feature($crate::__vendor_runtime::__Feature::asimd{})
    };
    ("asimd") => {
        $crate::__vendor_runtime::__unstable_detect_feature($crate::__vendor_runtime::__Feature::asimd{})
    };
    ("pmull") => {
        $crate::__vendor_runtime::__unstable_detect_feature($crate::__vendor_runtime::__Feature::pmull{})
    };
    ($t:tt) => { compile_error!(concat!("unknown arm target feature: ", $t)) };
}

/// ARM Aarch64 CPU Feature enum. Each variant denotes a position in a bitset
/// for a particular feature.
///
/// PLEASE: do not use this, it is an implementation detail subject to change.
#[doc(hidden)]
#[allow(non_camel_case_types)]
#[repr(u8)]
pub enum __Feature {
    /// ARM Advanced SIMD (ASIMD) - Aarch64
    asimd,
    /// Polynomial Multiply
    pmull,
}

pub fn detect_features<T: HasFeature>(mut x: T) -> usize {
    let mut value: usize = 0;
    {
        let mut enable_feature = |f| {
            if x.has_feature(&f) {
                value = bit::set(value, f as u32);
            }
        };
        enable_feature(__Feature::asimd);
        enable_feature(__Feature::pmull);
    }
    value
}
