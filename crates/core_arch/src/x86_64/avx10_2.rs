use crate::core_arch::{simd::*, x86::*};

#[cfg(test)]
use stdarch_test::assert_instr;

/// Convert the lower double-precision (64-bit) floating-point element in `a` to a 64-bit
/// integer with truncation and saturation.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttsd2sis, SAE = 8)
)]
pub fn _mm_cvtts_roundsd_i64<const SAE: i32>(a: __m128d) -> i64 {
    static_assert_sae!(SAE);
    unsafe { vcvttsd2sis64(a.as_f64x2(), SAE) }
}

/// Convert the lower double-precision (64-bit) floating-point element in `a` to a 64-bit
/// integer with truncation and saturation.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
pub fn _mm_cvtts_roundsd_si64<const SAE: i32>(a: __m128d) -> i64 {
    _mm_cvtts_roundsd_i64::<SAE>(a)
}

/// Convert the lower double-precision (64-bit) floating-point element in `a` to a 64-bit
/// unsigned integer with truncation and saturation.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttsd2usis, SAE = 8)
)]
pub fn _mm_cvtts_roundsd_u64<const SAE: i32>(a: __m128d) -> u64 {
    static_assert_sae!(SAE);
    unsafe { vcvttsd2usis64(a.as_f64x2(), SAE) }
}

/// Convert the lower single-precision (32-bit) floating-point element in `a` to a 64-bit
/// integer with truncation and saturation.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttss2sis, SAE = 8)
)]
pub fn _mm_cvtts_roundss_i64<const SAE: i32>(a: __m128) -> i64 {
    static_assert_sae!(SAE);
    unsafe { vcvttss2sis64(a.as_f32x4(), SAE) }
}

/// Convert the lower single-precision (32-bit) floating-point element in `a` to a 64-bit
/// integer with truncation and saturation.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
pub fn _mm_cvtts_roundss_si64<const SAE: i32>(a: __m128) -> i64 {
    _mm_cvtts_roundss_i64::<SAE>(a)
}

/// Convert the lower single-precision (32-bit) floating-point element in `a` to a 64-bit
/// unsigned integer with truncation and saturation.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttss2usis, SAE = 8)
)]
pub fn _mm_cvtts_roundss_u64<const SAE: i32>(a: __m128) -> u64 {
    static_assert_sae!(SAE);
    unsafe { vcvttss2usis64(a.as_f32x4(), SAE) }
}

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.x86.avx10.vcvttss2sis64"]
    fn vcvttss2sis64(a: f32x4, sae: i32) -> i64;
    #[link_name = "llvm.x86.avx10.vcvttss2usis64"]
    fn vcvttss2usis64(a: f32x4, sae: i32) -> u64;

    #[link_name = "llvm.x86.avx10.vcvttsd2sis64"]
    fn vcvttsd2sis64(a: f64x2, sae: i32) -> i64;
    #[link_name = "llvm.x86.avx10.vcvttsd2usis64"]
    fn vcvttsd2usis64(a: f64x2, sae: i32) -> u64;
}

#[cfg(test)]
mod tests {
    use super::*;
    use stdarch_test::simd_test;

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_roundsd_i64() {
        let a = _mm_set_sd(8.5);
        let r = _mm_cvtts_roundsd_i64::<_MM_FROUND_NO_EXC>(a);
        assert_eq!(r, 8i64);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_roundsd_si64() {
        let a = _mm_set_sd(9.3);
        let r = _mm_cvtts_roundsd_si64::<_MM_FROUND_NO_EXC>(a);
        assert_eq!(r, 9i64);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_roundsd_u64() {
        let a = _mm_set_sd(10.7);
        let r = _mm_cvtts_roundsd_u64::<_MM_FROUND_NO_EXC>(a);
        assert_eq!(r, 10u64);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_roundss_i64() {
        let a = _mm_set_ss(11.4);
        let r = _mm_cvtts_roundss_i64::<_MM_FROUND_NO_EXC>(a);
        assert_eq!(r, 11i64);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_roundss_si64() {
        let a = _mm_set_ss(12.9);
        let r = _mm_cvtts_roundss_si64::<_MM_FROUND_NO_EXC>(a);
        assert_eq!(r, 12i64);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_roundss_u64() {
        let a = _mm_set_ss(13.6);
        let r = _mm_cvtts_roundss_u64::<_MM_FROUND_NO_EXC>(a);
        assert_eq!(r, 13u64);
    }
}
