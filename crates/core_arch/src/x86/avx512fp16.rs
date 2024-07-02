use crate::core_arch::{simd::*, x86::*};
// use core::intrinsics::simd::*;

/// Set packed half-precision (16-bit) floating-point elements in dst with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_set_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_set_ph(
    e7: f16,
    e6: f16,
    e5: f16,
    e4: f16,
    e3: f16,
    e2: f16,
    e1: f16,
    e0: f16,
) -> __m128h {
    __m128h(e0, e1, e2, e3, e4, e5, e6, e7)
}

/// Set packed half-precision (16-bit) floating-point elements in dst with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_set_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_set_ph(
    e15: f16,
    e14: f16,
    e13: f16,
    e12: f16,
    e11: f16,
    e10: f16,
    e9: f16,
    e8: f16,
    e7: f16,
    e6: f16,
    e5: f16,
    e4: f16,
    e3: f16,
    e2: f16,
    e1: f16,
    e0: f16,
) -> __m256h {
    __m256h(
        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15,
    )
}

/// Set packed half-precision (16-bit) floating-point elements in dst with the supplied values.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_set_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_set_ph(
    e31: f16,
    e30: f16,
    e29: f16,
    e28: f16,
    e27: f16,
    e26: f16,
    e25: f16,
    e24: f16,
    e23: f16,
    e22: f16,
    e21: f16,
    e20: f16,
    e19: f16,
    e18: f16,
    e17: f16,
    e16: f16,
    e15: f16,
    e14: f16,
    e13: f16,
    e12: f16,
    e11: f16,
    e10: f16,
    e9: f16,
    e8: f16,
    e7: f16,
    e6: f16,
    e5: f16,
    e4: f16,
    e3: f16,
    e2: f16,
    e1: f16,
    e0: f16,
) -> __m512h {
    __m512h(
        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19,
        e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31,
    )
}

/// Copy half-precision (16-bit) floating-point elements from a to the lower element of dst and zero
/// the upper 7 elements.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_set_sh)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_set_sh(a: f16) -> __m128h {
    __m128h(a, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
}

/// Broadcast the half-precision (16-bit) floating-point value a to all elements of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_set1_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_set1_ph(a: f16) -> __m128h {
    transmute(f16x8::splat(a))
}

/// Broadcast the half-precision (16-bit) floating-point value a to all elements of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_set1_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_set1_ph(a: f16) -> __m256h {
    transmute(f16x16::splat(a))
}

/// Broadcast the half-precision (16-bit) floating-point value a to all elements of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_set1_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_set1_ph(a: f16) -> __m512h {
    transmute(f16x32::splat(a))
}

/// Set packed half-precision (16-bit) floating-point elements in dst with the supplied values in reverse order.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_setr_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_setr_ph(
    e0: f16,
    e1: f16,
    e2: f16,
    e3: f16,
    e4: f16,
    e5: f16,
    e6: f16,
    e7: f16,
) -> __m128h {
    __m128h(e0, e1, e2, e3, e4, e5, e6, e7)
}

/// Set packed half-precision (16-bit) floating-point elements in dst with the supplied values in reverse order.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_setr_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_setr_ph(
    e0: f16,
    e1: f16,
    e2: f16,
    e3: f16,
    e4: f16,
    e5: f16,
    e6: f16,
    e7: f16,
    e8: f16,
    e9: f16,
    e10: f16,
    e11: f16,
    e12: f16,
    e13: f16,
    e14: f16,
    e15: f16,
) -> __m256h {
    __m256h(
        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15,
    )
}

/// Set packed half-precision (16-bit) floating-point elements in dst with the supplied values in reverse order.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_setr_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_setr_ph(
    e0: f16,
    e1: f16,
    e2: f16,
    e3: f16,
    e4: f16,
    e5: f16,
    e6: f16,
    e7: f16,
    e8: f16,
    e9: f16,
    e10: f16,
    e11: f16,
    e12: f16,
    e13: f16,
    e14: f16,
    e15: f16,
    e16: f16,
    e17: f16,
    e18: f16,
    e19: f16,
    e20: f16,
    e21: f16,
    e22: f16,
    e23: f16,
    e24: f16,
    e25: f16,
    e26: f16,
    e27: f16,
    e28: f16,
    e29: f16,
    e30: f16,
    e31: f16,
) -> __m512h {
    __m512h(
        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13, e14, e15, e16, e17, e18, e19,
        e20, e21, e22, e23, e24, e25, e26, e27, e28, e29, e30, e31,
    )
}

/// Return vector of type __m128h with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_setzero_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm_setzero_ph() -> __m128h {
    transmute(f16x8::splat(0.0))
}

/// Return vector of type __m256h with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_setzero_ph)
#[inline]
#[target_feature(enable = "avx512fp16,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm256_setzero_ph() -> __m256h {
    transmute(f16x16::splat(0.0))
}

/// Return vector of type __m512h with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_setzero_ph)
#[inline]
#[target_feature(enable = "avx512fp16")]
#[unstable(feature = "stdarch_x86_avx512_f16", issue = "127213")]
pub unsafe fn _mm512_setzero_ph() -> __m512h {
    transmute(f16x32::splat(0.0))
}

#[cfg(test)]
mod tests {
    use crate::core_arch::x86::*;
    use std::mem::transmute;
    use stdarch_test::simd_test;

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_set_ph() {
        let r = _mm_set_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let e = _mm_setr_ph(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm256_set_ph() {
        let r = _mm256_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let e = _mm256_setr_ph(
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_set_ph() {
        let r = _mm512_set_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let e = _mm512_setr_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_set_sh() {
        let r = _mm_set_sh(1.0);
        let e = _mm_set_ph(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_set1_ph() {
        let r = _mm_set1_ph(1.0);
        let e = _mm_set_ph(1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm256_set1_ph() {
        let r = _mm256_set1_ph(1.0);
        let e = _mm256_set_ph(
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_set1_ph() {
        let r = _mm512_set1_ph(1.0);
        let e = _mm512_set_ph(
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm_setr_ph() {
        let r = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let e = _mm_set_ph(8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm256_setr_ph() {
        let r = _mm256_setr_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let e = _mm256_set_ph(
            16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_setr_ph() {
        let r = _mm512_setr_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let e = _mm512_set_ph(
            32.0, 31.0, 30.0, 29.0, 28.0, 27.0, 26.0, 25.0, 24.0, 23.0, 22.0, 21.0, 20.0, 19.0,
            18.0, 17.0, 16.0, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0,
            3.0, 2.0, 1.0,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm_setzero_ph() {
        let r = _mm_setzero_ph();
        let e = _mm_set1_ph(0.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx512fp16,avx512vl")]
    unsafe fn test_mm256_setzero_ph() {
        let r = _mm256_setzero_ph();
        let e = _mm256_set1_ph(0.0);
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx512fp16")]
    unsafe fn test_mm512_setzero_ph() {
        let r = _mm512_setzero_ph();
        let e = _mm512_set1_ph(0.0);
        assert_eq_m512h(r, e);
    }
}
