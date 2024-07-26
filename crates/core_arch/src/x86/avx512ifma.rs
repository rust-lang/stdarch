use crate::core_arch::x86::*;
use crate::intrinsics::simd::simd_select_bitmask;

#[cfg(test)]
use stdarch_test::assert_instr;

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the high 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512IFMA52&text=_mm512_madd52hi_epu64)
#[inline]
#[target_feature(enable = "avx512ifma")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmadd52huq))]
pub unsafe fn _mm512_madd52hi_epu64(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    vpmadd52huq_512(a, b, c)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the high 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst` using writemask `k` (elements are copied
/// from `k` when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512IFMA52&text=_mm512_mask_madd52hi_epu64)
#[inline]
#[target_feature(enable = "avx512ifma")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmadd52huq))]
pub unsafe fn _mm512_mask_madd52hi_epu64(
    a: __m512i,
    k: __mmask8,
    b: __m512i,
    c: __m512i,
) -> __m512i {
    simd_select_bitmask(k, vpmadd52huq_512(a, b, c), a)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the high 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst` using writemask `k` (elements are zeroed
/// out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512IFMA52&text=_mm512_maskz_madd52hi_epu64)
#[inline]
#[target_feature(enable = "avx512ifma")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmadd52huq))]
pub unsafe fn _mm512_maskz_madd52hi_epu64(
    k: __mmask8,
    a: __m512i,
    b: __m512i,
    c: __m512i,
) -> __m512i {
    simd_select_bitmask(k, vpmadd52huq_512(a, b, c), _mm512_setzero_si512())
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the low 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512IFMA52&text=_mm512_madd52lo_epu64)
#[inline]
#[target_feature(enable = "avx512ifma")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmadd52luq))]
pub unsafe fn _mm512_madd52lo_epu64(a: __m512i, b: __m512i, c: __m512i) -> __m512i {
    vpmadd52luq_512(a, b, c)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the low 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst` using writemask `k` (elements are copied
/// from `k` when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512IFMA52&text=_mm512_mask_madd52lo_epu64)
#[inline]
#[target_feature(enable = "avx512ifma")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmadd52luq))]
pub unsafe fn _mm512_mask_madd52lo_epu64(
    a: __m512i,
    k: __mmask8,
    b: __m512i,
    c: __m512i,
) -> __m512i {
    simd_select_bitmask(k, vpmadd52luq_512(a, b, c), a)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the low 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst` using writemask `k` (elements are zeroed
/// out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512IFMA52&text=_mm512_maskz_madd52lo_epu64)
#[inline]
#[target_feature(enable = "avx512ifma")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmadd52luq))]
pub unsafe fn _mm512_maskz_madd52lo_epu64(
    k: __mmask8,
    a: __m512i,
    b: __m512i,
    c: __m512i,
) -> __m512i {
    simd_select_bitmask(k, vpmadd52luq_512(a, b, c), _mm512_setzero_si512())
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the high 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_madd52hi_avx_epu64)
#[inline]
#[target_feature(enable = "avxifma")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(
    all(test, any(target_os = "linux", target_env = "msvc")),
    assert_instr(vpmadd52huq)
)]
pub unsafe fn _mm256_madd52hi_avx_epu64(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    vpmadd52huq_256(a, b, c)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the high 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512IFMA52&text=_mm256_madd52hi_epu64)
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmadd52huq))]
pub unsafe fn _mm256_madd52hi_epu64(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    vpmadd52huq_256(a, b, c)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the high 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst` using writemask `k` (elements are copied
/// from `k` when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512IFMA52&text=_mm256_mask_madd52hi_epu64)
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmadd52huq))]
pub unsafe fn _mm256_mask_madd52hi_epu64(
    a: __m256i,
    k: __mmask8,
    b: __m256i,
    c: __m256i,
) -> __m256i {
    simd_select_bitmask(k, vpmadd52huq_256(a, b, c), a)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the high 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst` using writemask `k` (elements are zeroed
/// out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512IFMA52&text=_mm256_maskz_madd52hi_epu64)
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmadd52huq))]
pub unsafe fn _mm256_maskz_madd52hi_epu64(
    k: __mmask8,
    a: __m256i,
    b: __m256i,
    c: __m256i,
) -> __m256i {
    simd_select_bitmask(k, vpmadd52huq_256(a, b, c), _mm256_setzero_si256())
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the low 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm256_madd52lo_avx_epu64)
#[inline]
#[target_feature(enable = "avxifma")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(
    all(test, any(target_os = "linux", target_env = "msvc")),
    assert_instr(vpmadd52luq)
)]
pub unsafe fn _mm256_madd52lo_avx_epu64(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    vpmadd52luq_256(a, b, c)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the low 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512IFMA52&text=_mm256_madd52lo_epu64)
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmadd52luq))]
pub unsafe fn _mm256_madd52lo_epu64(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    vpmadd52luq_256(a, b, c)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the low 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst` using writemask `k` (elements are copied
/// from `k` when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512IFMA52&text=_mm256_mask_madd52lo_epu64)
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmadd52luq))]
pub unsafe fn _mm256_mask_madd52lo_epu64(
    a: __m256i,
    k: __mmask8,
    b: __m256i,
    c: __m256i,
) -> __m256i {
    simd_select_bitmask(k, vpmadd52luq_256(a, b, c), a)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the low 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst` using writemask `k` (elements are zeroed
/// out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512IFMA52&text=_mm256_maskz_madd52lo_epu64)
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmadd52luq))]
pub unsafe fn _mm256_maskz_madd52lo_epu64(
    k: __mmask8,
    a: __m256i,
    b: __m256i,
    c: __m256i,
) -> __m256i {
    simd_select_bitmask(k, vpmadd52luq_256(a, b, c), _mm256_setzero_si256())
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the high 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_madd52hi_avx_epu64)
#[inline]
#[target_feature(enable = "avxifma")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(
    all(test, any(target_os = "linux", target_env = "msvc")),
    assert_instr(vpmadd52huq)
)]
pub unsafe fn _mm_madd52hi_avx_epu64(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    vpmadd52huq_128(a, b, c)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the high 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512IFMA52&text=_mm_madd52hi_epu64)
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmadd52huq))]
pub unsafe fn _mm_madd52hi_epu64(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    vpmadd52huq_128(a, b, c)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the high 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst` using writemask `k` (elements are copied
/// from `k` when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512IFMA52&text=_mm_mask_madd52hi_epu64)
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmadd52huq))]
pub unsafe fn _mm_mask_madd52hi_epu64(a: __m128i, k: __mmask8, b: __m128i, c: __m128i) -> __m128i {
    simd_select_bitmask(k, vpmadd52huq_128(a, b, c), a)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the high 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst` using writemask `k` (elements are zeroed
/// out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512IFMA52&text=_mm_maskz_madd52hi_epu64)
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmadd52huq))]
pub unsafe fn _mm_maskz_madd52hi_epu64(k: __mmask8, a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    simd_select_bitmask(k, vpmadd52huq_128(a, b, c), _mm_setzero_si128())
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the low 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_madd52lo_avx_epu64)
#[inline]
#[target_feature(enable = "avxifma")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(
    all(test, any(target_os = "linux", target_env = "msvc")),
    assert_instr(vpmadd52luq)
)]
pub unsafe fn _mm_madd52lo_avx_epu64(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    vpmadd52luq_128(a, b, c)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the low 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst`.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512IFMA52&text=_mm_madd52lo_epu64)
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmadd52luq))]
pub unsafe fn _mm_madd52lo_epu64(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    vpmadd52luq_128(a, b, c)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the low 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst` using writemask `k` (elements are copied
/// from `k` when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512IFMA52&text=_mm_mask_madd52lo_epu64)
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmadd52luq))]
pub unsafe fn _mm_mask_madd52lo_epu64(a: __m128i, k: __mmask8, b: __m128i, c: __m128i) -> __m128i {
    simd_select_bitmask(k, vpmadd52luq_128(a, b, c), a)
}

/// Multiply packed unsigned 52-bit integers in each 64-bit element of
/// `b` and `c` to form a 104-bit intermediate result. Add the low 52-bit
/// unsigned integer from the intermediate result with the
/// corresponding unsigned 64-bit integer in `a`, and store the
/// results in `dst` using writemask `k` (elements are zeroed
/// out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#avx512techs=AVX512IFMA52&text=_mm_maskz_madd52lo_epu64)
#[inline]
#[target_feature(enable = "avx512ifma,avx512vl")]
#[unstable(feature = "stdarch_x86_avx512", issue = "111137")]
#[cfg_attr(test, assert_instr(vpmadd52luq))]
pub unsafe fn _mm_maskz_madd52lo_epu64(k: __mmask8, a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    simd_select_bitmask(k, vpmadd52luq_128(a, b, c), _mm_setzero_si128())
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.vpmadd52l.uq.128"]
    fn vpmadd52luq_128(z: __m128i, x: __m128i, y: __m128i) -> __m128i;
    #[link_name = "llvm.x86.avx512.vpmadd52h.uq.128"]
    fn vpmadd52huq_128(z: __m128i, x: __m128i, y: __m128i) -> __m128i;
    #[link_name = "llvm.x86.avx512.vpmadd52l.uq.256"]
    fn vpmadd52luq_256(z: __m256i, x: __m256i, y: __m256i) -> __m256i;
    #[link_name = "llvm.x86.avx512.vpmadd52h.uq.256"]
    fn vpmadd52huq_256(z: __m256i, x: __m256i, y: __m256i) -> __m256i;
    #[link_name = "llvm.x86.avx512.vpmadd52l.uq.512"]
    fn vpmadd52luq_512(z: __m512i, x: __m512i, y: __m512i) -> __m512i;
    #[link_name = "llvm.x86.avx512.vpmadd52h.uq.512"]
    fn vpmadd52huq_512(z: __m512i, x: __m512i, y: __m512i) -> __m512i;
}

#[cfg(test)]
mod tests {

    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    const K: __mmask8 = 0b01101101;

    #[simd_test(enable = "avx512ifma")]
    unsafe fn test_mm512_madd52hi_epu64() {
        let a = _mm512_set1_epi64(10 << 40);
        let b = _mm512_set1_epi64((11 << 40) + 4);
        let c = _mm512_set1_epi64((12 << 40) + 3);

        let actual = _mm512_madd52hi_epu64(a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) >> 52)
        let expected = _mm512_set1_epi64(11030549757952);

        assert_eq_m512i(expected, actual);
    }

    #[simd_test(enable = "avx512ifma")]
    unsafe fn test_mm512_mask_madd52hi_epu64() {
        let a = _mm512_set1_epi64(10 << 40);
        let b = _mm512_set1_epi64((11 << 40) + 4);
        let c = _mm512_set1_epi64((12 << 40) + 3);

        let actual = _mm512_mask_madd52hi_epu64(a, K, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) >> 52)
        let mut expected = _mm512_set1_epi64(11030549757952);
        expected = _mm512_mask_blend_epi64(K, a, expected);

        assert_eq_m512i(expected, actual);
    }

    #[simd_test(enable = "avx512ifma")]
    unsafe fn test_mm512_maskz_madd52hi_epu64() {
        let a = _mm512_set1_epi64(10 << 40);
        let b = _mm512_set1_epi64((11 << 40) + 4);
        let c = _mm512_set1_epi64((12 << 40) + 3);

        let actual = _mm512_maskz_madd52hi_epu64(K, a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) >> 52)
        let mut expected = _mm512_set1_epi64(11030549757952);
        expected = _mm512_mask_blend_epi64(K, _mm512_setzero_si512(), expected);

        assert_eq_m512i(expected, actual);
    }

    #[simd_test(enable = "avx512ifma")]
    unsafe fn test_mm512_madd52lo_epu64() {
        let a = _mm512_set1_epi64(10 << 40);
        let b = _mm512_set1_epi64((11 << 40) + 4);
        let c = _mm512_set1_epi64((12 << 40) + 3);

        let actual = _mm512_madd52lo_epu64(a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) % (1 << 52))
        let expected = _mm512_set1_epi64(100055558127628);

        assert_eq_m512i(expected, actual);
    }

    #[simd_test(enable = "avx512ifma")]
    unsafe fn test_mm512_mask_madd52lo_epu64() {
        let a = _mm512_set1_epi64(10 << 40);
        let b = _mm512_set1_epi64((11 << 40) + 4);
        let c = _mm512_set1_epi64((12 << 40) + 3);

        let actual = _mm512_mask_madd52lo_epu64(a, K, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) % (1 << 52))
        let mut expected = _mm512_set1_epi64(100055558127628);
        expected = _mm512_mask_blend_epi64(K, a, expected);

        assert_eq_m512i(expected, actual);
    }

    #[simd_test(enable = "avx512ifma")]
    unsafe fn test_mm512_maskz_madd52lo_epu64() {
        let a = _mm512_set1_epi64(10 << 40);
        let b = _mm512_set1_epi64((11 << 40) + 4);
        let c = _mm512_set1_epi64((12 << 40) + 3);

        let actual = _mm512_maskz_madd52lo_epu64(K, a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) % (1 << 52))
        let mut expected = _mm512_set1_epi64(100055558127628);
        expected = _mm512_mask_blend_epi64(K, _mm512_setzero_si512(), expected);

        assert_eq_m512i(expected, actual);
    }

    #[simd_test(enable = "avxifma")]
    unsafe fn test_mm256_madd52hi_avx_epu64() {
        let a = _mm256_set1_epi64x(10 << 40);
        let b = _mm256_set1_epi64x((11 << 40) + 4);
        let c = _mm256_set1_epi64x((12 << 40) + 3);

        let actual = _mm256_madd52hi_avx_epu64(a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) >> 52)
        let expected = _mm256_set1_epi64x(11030549757952);

        assert_eq_m256i(expected, actual);
    }

    #[simd_test(enable = "avx512ifma,avx512vl")]
    unsafe fn test_mm256_madd52hi_epu64() {
        let a = _mm256_set1_epi64x(10 << 40);
        let b = _mm256_set1_epi64x((11 << 40) + 4);
        let c = _mm256_set1_epi64x((12 << 40) + 3);

        let actual = _mm256_madd52hi_epu64(a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) >> 52)
        let expected = _mm256_set1_epi64x(11030549757952);

        assert_eq_m256i(expected, actual);
    }

    #[simd_test(enable = "avx512ifma,avx512vl")]
    unsafe fn test_mm256_mask_madd52hi_epu64() {
        let a = _mm256_set1_epi64x(10 << 40);
        let b = _mm256_set1_epi64x((11 << 40) + 4);
        let c = _mm256_set1_epi64x((12 << 40) + 3);

        let actual = _mm256_mask_madd52hi_epu64(a, K, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) >> 52)
        let mut expected = _mm256_set1_epi64x(11030549757952);
        expected = _mm256_mask_blend_epi64(K, a, expected);

        assert_eq_m256i(expected, actual);
    }

    #[simd_test(enable = "avx512ifma,avx512vl")]
    unsafe fn test_mm256_maskz_madd52hi_epu64() {
        let a = _mm256_set1_epi64x(10 << 40);
        let b = _mm256_set1_epi64x((11 << 40) + 4);
        let c = _mm256_set1_epi64x((12 << 40) + 3);

        let actual = _mm256_maskz_madd52hi_epu64(K, a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) >> 52)
        let mut expected = _mm256_set1_epi64x(11030549757952);
        expected = _mm256_mask_blend_epi64(K, _mm256_setzero_si256(), expected);

        assert_eq_m256i(expected, actual);
    }

    #[simd_test(enable = "avxifma")]
    unsafe fn test_mm256_madd52lo_avx_epu64() {
        let a = _mm256_set1_epi64x(10 << 40);
        let b = _mm256_set1_epi64x((11 << 40) + 4);
        let c = _mm256_set1_epi64x((12 << 40) + 3);

        let actual = _mm256_madd52lo_avx_epu64(a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) % (1 << 52))
        let expected = _mm256_set1_epi64x(100055558127628);

        assert_eq_m256i(expected, actual);
    }

    #[simd_test(enable = "avx512ifma,avx512vl")]
    unsafe fn test_mm256_madd52lo_epu64() {
        let a = _mm256_set1_epi64x(10 << 40);
        let b = _mm256_set1_epi64x((11 << 40) + 4);
        let c = _mm256_set1_epi64x((12 << 40) + 3);

        let actual = _mm256_madd52lo_epu64(a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) % (1 << 52))
        let expected = _mm256_set1_epi64x(100055558127628);

        assert_eq_m256i(expected, actual);
    }

    #[simd_test(enable = "avx512ifma,avx512vl")]
    unsafe fn test_mm256_mask_madd52lo_epu64() {
        let a = _mm256_set1_epi64x(10 << 40);
        let b = _mm256_set1_epi64x((11 << 40) + 4);
        let c = _mm256_set1_epi64x((12 << 40) + 3);

        let actual = _mm256_mask_madd52lo_epu64(a, K, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) % (1 << 52))
        let mut expected = _mm256_set1_epi64x(100055558127628);
        expected = _mm256_mask_blend_epi64(K, a, expected);

        assert_eq_m256i(expected, actual);
    }

    #[simd_test(enable = "avx512ifma,avx512vl")]
    unsafe fn test_mm256_maskz_madd52lo_epu64() {
        let a = _mm256_set1_epi64x(10 << 40);
        let b = _mm256_set1_epi64x((11 << 40) + 4);
        let c = _mm256_set1_epi64x((12 << 40) + 3);

        let actual = _mm256_maskz_madd52lo_epu64(K, a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) % (1 << 52))
        let mut expected = _mm256_set1_epi64x(100055558127628);
        expected = _mm256_mask_blend_epi64(K, _mm256_setzero_si256(), expected);

        assert_eq_m256i(expected, actual);
    }

    #[simd_test(enable = "avxifma")]
    unsafe fn test_mm_madd52hi_avx_epu64() {
        let a = _mm_set1_epi64x(10 << 40);
        let b = _mm_set1_epi64x((11 << 40) + 4);
        let c = _mm_set1_epi64x((12 << 40) + 3);

        let actual = _mm_madd52hi_avx_epu64(a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) >> 52)
        let expected = _mm_set1_epi64x(11030549757952);

        assert_eq_m128i(expected, actual);
    }

    #[simd_test(enable = "avx512ifma,avx512vl")]
    unsafe fn test_mm_madd52hi_epu64() {
        let a = _mm_set1_epi64x(10 << 40);
        let b = _mm_set1_epi64x((11 << 40) + 4);
        let c = _mm_set1_epi64x((12 << 40) + 3);

        let actual = _mm_madd52hi_epu64(a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) >> 52)
        let expected = _mm_set1_epi64x(11030549757952);

        assert_eq_m128i(expected, actual);
    }

    #[simd_test(enable = "avx512ifma,avx512vl")]
    unsafe fn test_mm_mask_madd52hi_epu64() {
        let a = _mm_set1_epi64x(10 << 40);
        let b = _mm_set1_epi64x((11 << 40) + 4);
        let c = _mm_set1_epi64x((12 << 40) + 3);

        let actual = _mm_mask_madd52hi_epu64(a, K, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) >> 52)
        let mut expected = _mm_set1_epi64x(11030549757952);
        expected = _mm_mask_blend_epi64(K, a, expected);

        assert_eq_m128i(expected, actual);
    }

    #[simd_test(enable = "avx512ifma,avx512vl")]
    unsafe fn test_mm_maskz_madd52hi_epu64() {
        let a = _mm_set1_epi64x(10 << 40);
        let b = _mm_set1_epi64x((11 << 40) + 4);
        let c = _mm_set1_epi64x((12 << 40) + 3);

        let actual = _mm_maskz_madd52hi_epu64(K, a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) >> 52)
        let mut expected = _mm_set1_epi64x(11030549757952);
        expected = _mm_mask_blend_epi64(K, _mm_setzero_si128(), expected);

        assert_eq_m128i(expected, actual);
    }

    #[simd_test(enable = "avxifma")]
    unsafe fn test_mm_madd52lo_avx_epu64() {
        let a = _mm_set1_epi64x(10 << 40);
        let b = _mm_set1_epi64x((11 << 40) + 4);
        let c = _mm_set1_epi64x((12 << 40) + 3);

        let actual = _mm_madd52lo_avx_epu64(a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) % (1 << 52))
        let expected = _mm_set1_epi64x(100055558127628);

        assert_eq_m128i(expected, actual);
    }

    #[simd_test(enable = "avx512ifma,avx512vl")]
    unsafe fn test_mm_madd52lo_epu64() {
        let a = _mm_set1_epi64x(10 << 40);
        let b = _mm_set1_epi64x((11 << 40) + 4);
        let c = _mm_set1_epi64x((12 << 40) + 3);

        let actual = _mm_madd52lo_epu64(a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) % (1 << 52))
        let expected = _mm_set1_epi64x(100055558127628);

        assert_eq_m128i(expected, actual);
    }

    #[simd_test(enable = "avx512ifma,avx512vl")]
    unsafe fn test_mm_mask_madd52lo_epu64() {
        let a = _mm_set1_epi64x(10 << 40);
        let b = _mm_set1_epi64x((11 << 40) + 4);
        let c = _mm_set1_epi64x((12 << 40) + 3);

        let actual = _mm_mask_madd52lo_epu64(a, K, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) % (1 << 52))
        let mut expected = _mm_set1_epi64x(100055558127628);
        expected = _mm_mask_blend_epi64(K, a, expected);

        assert_eq_m128i(expected, actual);
    }

    #[simd_test(enable = "avx512ifma,avx512vl")]
    unsafe fn test_mm_maskz_madd52lo_epu64() {
        let a = _mm_set1_epi64x(10 << 40);
        let b = _mm_set1_epi64x((11 << 40) + 4);
        let c = _mm_set1_epi64x((12 << 40) + 3);

        let actual = _mm_maskz_madd52lo_epu64(K, a, b, c);

        // (10 << 40) + ((((11 << 40) + 4) * ((12 << 40) + 3)) % (1 << 52))
        let mut expected = _mm_set1_epi64x(100055558127628);
        expected = _mm_mask_blend_epi64(K, _mm_setzero_si128(), expected);

        assert_eq_m128i(expected, actual);
    }
}
