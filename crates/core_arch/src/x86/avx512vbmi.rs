use crate::core_arch::{simd::*, simd_llvm::*, x86::*};

#[cfg(test)]
use stdarch_test::assert_instr;

/// Shuffle 8-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_permutex2var_epi8&expand=4262)
#[inline]
#[target_feature(enable = "avx512vbmi")]
#[cfg_attr(test, assert_instr(vperm))] //should be vpermi2b
pub unsafe fn _mm512_permutex2var_epi8(a: __m512i, idx: __m512i, b: __m512i) -> __m512i {
    transmute(vpermi2b(a.as_i8x64(), idx.as_i8x64(), b.as_i8x64()))
}

/// Shuffle 8-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_permutex2var_epi8&expand=4259)
#[inline]
#[target_feature(enable = "avx512vbmi")]
#[cfg_attr(test, assert_instr(vpermt2b))]
pub unsafe fn _mm512_mask_permutex2var_epi8(
    a: __m512i,
    k: __mmask64,
    idx: __m512i,
    b: __m512i,
) -> __m512i {
    let permute = _mm512_permutex2var_epi8(a, idx, b).as_i8x64();
    transmute(simd_select_bitmask(k, permute, a.as_i8x64()))
}

/// Shuffle 8-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_permutex2var_epi8&expand=4261)
#[inline]
#[target_feature(enable = "avx512vbmi")]
#[cfg_attr(test, assert_instr(vpermt2b))]
pub unsafe fn _mm512_maskz_permutex2var_epi8(
    k: __mmask64,
    a: __m512i,
    idx: __m512i,
    b: __m512i,
) -> __m512i {
    let permute = _mm512_permutex2var_epi8(a, idx, b).as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, permute, zero))
}

/// Shuffle 8-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask2_permutex2var_epi8&expand=4260)
#[inline]
#[target_feature(enable = "avx512vbmi")]
#[cfg_attr(test, assert_instr(vpermi2b))]
pub unsafe fn _mm512_mask2_permutex2var_epi8(
    a: __m512i,
    idx: __m512i,
    k: __mmask64,
    b: __m512i,
) -> __m512i {
    let permute = _mm512_permutex2var_epi8(a, idx, b).as_i8x64();
    transmute(simd_select_bitmask(k, permute, idx.as_i8x64()))
}

/// Shuffle 8-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_permutex2var_epi8&expand=4258)
#[inline]
#[target_feature(enable = "avx512vbmi,avx512vl")]
#[cfg_attr(test, assert_instr(vperm))] //should be vpermi2b
pub unsafe fn _mm256_permutex2var_epi8(a: __m256i, idx: __m256i, b: __m256i) -> __m256i {
    transmute(vpermi2b256(a.as_i8x32(), idx.as_i8x32(), b.as_i8x32()))
}

/// Shuffle 8-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_permutex2var_epi8&expand=4255)
#[inline]
#[target_feature(enable = "avx512vbmi,avx512vl")]
#[cfg_attr(test, assert_instr(vpermt2b))]
pub unsafe fn _mm256_mask_permutex2var_epi8(
    a: __m256i,
    k: __mmask32,
    idx: __m256i,
    b: __m256i,
) -> __m256i {
    let permute = _mm256_permutex2var_epi8(a, idx, b).as_i8x32();
    transmute(simd_select_bitmask(k, permute, a.as_i8x32()))
}

/// Shuffle 8-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_permutex2var_epi8&expand=4257)
#[inline]
#[target_feature(enable = "avx512vbmi,avx512vl")]
#[cfg_attr(test, assert_instr(vpermt2b))]
pub unsafe fn _mm256_maskz_permutex2var_epi8(
    k: __mmask32,
    a: __m256i,
    idx: __m256i,
    b: __m256i,
) -> __m256i {
    let permute = _mm256_permutex2var_epi8(a, idx, b).as_i8x32();
    let zero = _mm256_setzero_si256().as_i8x32();
    transmute(simd_select_bitmask(k, permute, zero))
}

/// Shuffle 8-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask2_permutex2var_epi8&expand=4256)
#[inline]
#[target_feature(enable = "avx512vbmi,avx512vl")]
#[cfg_attr(test, assert_instr(vpermi2b))]
pub unsafe fn _mm256_mask2_permutex2var_epi8(
    a: __m256i,
    idx: __m256i,
    k: __mmask32,
    b: __m256i,
) -> __m256i {
    let permute = _mm256_permutex2var_epi8(a, idx, b).as_i8x32();
    transmute(simd_select_bitmask(k, permute, idx.as_i8x32()))
}

/// Shuffle 8-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_permutex2var_epi8&expand=4254)
#[inline]
#[target_feature(enable = "avx512vbmi,avx512vl")]
#[cfg_attr(test, assert_instr(vperm))] //should be vpermi2b
pub unsafe fn _mm_permutex2var_epi8(a: __m128i, idx: __m128i, b: __m128i) -> __m128i {
    transmute(vpermi2b128(a.as_i8x16(), idx.as_i8x16(), b.as_i8x16()))
}

/// Shuffle 8-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_permutex2var_epi8&expand=4251)
#[inline]
#[target_feature(enable = "avx512vbmi,avx512vl")]
#[cfg_attr(test, assert_instr(vpermt2b))]
pub unsafe fn _mm_mask_permutex2var_epi8(
    a: __m128i,
    k: __mmask16,
    idx: __m128i,
    b: __m128i,
) -> __m128i {
    let permute = _mm_permutex2var_epi8(a, idx, b).as_i8x16();
    transmute(simd_select_bitmask(k, permute, a.as_i8x16()))
}

/// Shuffle 8-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_permutex2var_epi8&expand=4253)
#[inline]
#[target_feature(enable = "avx512vbmi,avx512vl")]
#[cfg_attr(test, assert_instr(vpermt2b))]
pub unsafe fn _mm_maskz_permutex2var_epi8(
    k: __mmask16,
    a: __m128i,
    idx: __m128i,
    b: __m128i,
) -> __m128i {
    let permute = _mm_permutex2var_epi8(a, idx, b).as_i8x16();
    let zero = _mm_setzero_si128().as_i8x16();
    transmute(simd_select_bitmask(k, permute, zero))
}

/// Shuffle 8-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask2_permutex2var_epi8&expand=4252)
#[inline]
#[target_feature(enable = "avx512vbmi,avx512vl")]
#[cfg_attr(test, assert_instr(vpermi2b))]
pub unsafe fn _mm_mask2_permutex2var_epi8(
    a: __m128i,
    idx: __m128i,
    k: __mmask16,
    b: __m128i,
) -> __m128i {
    let permute = _mm_permutex2var_epi8(a, idx, b).as_i8x16();
    transmute(simd_select_bitmask(k, permute, idx.as_i8x16()))
}

/*
/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_permutexvar_epi16&expand=4295)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpermw))]
pub unsafe fn _mm512_permutexvar_epi16(idx: __m512i, a: __m512i) -> __m512i {
    transmute(vpermw(a.as_i16x32(), idx.as_i16x32()))
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_permutexvar_epi16&expand=4293)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpermw))]
pub unsafe fn _mm512_mask_permutexvar_epi16(
    src: __m512i,
    k: __mmask32,
    idx: __m512i,
    a: __m512i,
) -> __m512i {
    let permute = _mm512_permutexvar_epi16(idx, a).as_i16x32();
    transmute(simd_select_bitmask(k, permute, src.as_i16x32()))
}

/// Shuffle 16-bit integers in a across lanes using the corresponding index in idx, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_permutexvar_epi16&expand=4294)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpermw))]
pub unsafe fn _mm512_maskz_permutexvar_epi16(k: __mmask32, idx: __m512i, a: __m512i) -> __m512i {
    let permute = _mm512_permutexvar_epi16(idx, a).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, permute, zero))
}

*/
#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.vpermi2var.qi.512"]
    fn vpermi2b(a: i8x64, idx: i8x64, b: i8x64) -> i8x64;
    #[link_name = "llvm.x86.avx512.vpermi2var.qi.256"]
    fn vpermi2b256(a: i8x32, idx: i8x32, b: i8x32) -> i8x32;
    #[link_name = "llvm.x86.avx512.vpermi2var.qi.128"]
    fn vpermi2b128(a: i8x16, idx: i8x16, b: i8x16) -> i8x16;
}

#[cfg(test)]
mod tests {

    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;
    use crate::hint::black_box;
    use crate::mem::{self};

    /*
        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm_maskz_sllv_epi16() {
            let a = _mm_set1_epi16(1 << 15);
            let count = _mm_set1_epi16(2);
            let r = _mm_maskz_sllv_epi16(0, a, count);
            assert_eq_m128i(r, _mm_setzero_si128());
            let r = _mm_maskz_sllv_epi16(0b11111111, a, count);
            let e = _mm_set1_epi16(0);
            assert_eq_m128i(r, e);
        }

        #[simd_test(enable = "avx512bw")]
        unsafe fn test_mm512_srl_epi16() {
            let a = _mm512_set1_epi16(1 << 1);
            let count = _mm_set1_epi16(2);
            let r = _mm512_srl_epi16(a, count);
            let e = _mm512_set1_epi16(0);
            assert_eq_m512i(r, e);
        }

        #[simd_test(enable = "avx512bw")]
        unsafe fn test_mm512_mask_srl_epi16() {
            let a = _mm512_set1_epi16(1 << 1);
            let count = _mm_set1_epi16(2);
            let r = _mm512_mask_srl_epi16(a, 0, a, count);
            assert_eq_m512i(r, a);
            let r = _mm512_mask_srl_epi16(a, 0b11111111_11111111_11111111_11111111, a, count);
            let e = _mm512_set1_epi16(0);
            assert_eq_m512i(r, e);
        }

        #[simd_test(enable = "avx512bw")]
        unsafe fn test_mm512_maskz_srl_epi16() {
            let a = _mm512_set1_epi16(1 << 1);
            let count = _mm_set1_epi16(2);
            let r = _mm512_maskz_srl_epi16(0, a, count);
            assert_eq_m512i(r, _mm512_setzero_si512());
            let r = _mm512_maskz_srl_epi16(0b11111111_11111111_11111111_11111111, a, count);
            let e = _mm512_set1_epi16(0);
            assert_eq_m512i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm256_mask_srl_epi16() {
            let a = _mm256_set1_epi16(1 << 1);
            let count = _mm_set1_epi16(2);
            let r = _mm256_mask_srl_epi16(a, 0, a, count);
            assert_eq_m256i(r, a);
            let r = _mm256_mask_srl_epi16(a, 0b11111111_11111111, a, count);
            let e = _mm256_set1_epi16(0);
            assert_eq_m256i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm256_maskz_srl_epi16() {
            let a = _mm256_set1_epi16(1 << 1);
            let count = _mm_set1_epi16(2);
            let r = _mm256_maskz_srl_epi16(0, a, count);
            assert_eq_m256i(r, _mm256_setzero_si256());
            let r = _mm256_maskz_srl_epi16(0b11111111_11111111, a, count);
            let e = _mm256_set1_epi16(0);
            assert_eq_m256i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm_mask_srl_epi16() {
            let a = _mm_set1_epi16(1 << 1);
            let count = _mm_set1_epi16(2);
            let r = _mm_mask_srl_epi16(a, 0, a, count);
            assert_eq_m128i(r, a);
            let r = _mm_mask_srl_epi16(a, 0b11111111, a, count);
            let e = _mm_set1_epi16(0);
            assert_eq_m128i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm_maskz_srl_epi16() {
            let a = _mm_set1_epi16(1 << 1);
            let count = _mm_set1_epi16(2);
            let r = _mm_maskz_srl_epi16(0, a, count);
            assert_eq_m128i(r, _mm_setzero_si128());
            let r = _mm_maskz_srl_epi16(0b11111111, a, count);
            let e = _mm_set1_epi16(0);
            assert_eq_m128i(r, e);
        }

        #[simd_test(enable = "avx512bw")]
        unsafe fn test_mm512_srli_epi16() {
            let a = _mm512_set1_epi16(1 << 1);
            let r = _mm512_srli_epi16(a, 2);
            let e = _mm512_set1_epi16(0);
            assert_eq_m512i(r, e);
        }

        #[simd_test(enable = "avx512bw")]
        unsafe fn test_mm512_mask_srli_epi16() {
            let a = _mm512_set1_epi16(1 << 1);
            let r = _mm512_mask_srli_epi16(a, 0, a, 2);
            assert_eq_m512i(r, a);
            let r = _mm512_mask_srli_epi16(a, 0b11111111_11111111_11111111_11111111, a, 2);
            let e = _mm512_set1_epi16(0);
            assert_eq_m512i(r, e);
        }

        #[simd_test(enable = "avx512bw")]
        unsafe fn test_mm512_maskz_srli_epi16() {
            let a = _mm512_set1_epi16(1 << 1);
            let r = _mm512_maskz_srli_epi16(0, a, 2);
            assert_eq_m512i(r, _mm512_setzero_si512());
            let r = _mm512_maskz_srli_epi16(0b11111111_11111111_11111111_11111111, a, 2);
            let e = _mm512_set1_epi16(0);
            assert_eq_m512i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm256_mask_srli_epi16() {
            let a = _mm256_set1_epi16(1 << 1);
            let r = _mm256_mask_srli_epi16(a, 0, a, 2);
            assert_eq_m256i(r, a);
            let r = _mm256_mask_srli_epi16(a, 0b11111111_11111111, a, 2);
            let e = _mm256_set1_epi16(0);
            assert_eq_m256i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm256_maskz_srli_epi16() {
            let a = _mm256_set1_epi16(1 << 1);
            let r = _mm256_maskz_srli_epi16(0, a, 2);
            assert_eq_m256i(r, _mm256_setzero_si256());
            let r = _mm256_maskz_srli_epi16(0b11111111_11111111, a, 2);
            let e = _mm256_set1_epi16(0);
            assert_eq_m256i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm_mask_srli_epi16() {
            let a = _mm_set1_epi16(1 << 1);
            let r = _mm_mask_srli_epi16(a, 0, a, 2);
            assert_eq_m128i(r, a);
            let r = _mm_mask_srli_epi16(a, 0b11111111, a, 2);
            let e = _mm_set1_epi16(0);
            assert_eq_m128i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm_maskz_srli_epi16() {
            let a = _mm_set1_epi16(1 << 1);
            let r = _mm_maskz_srli_epi16(0, a, 2);
            assert_eq_m128i(r, _mm_setzero_si128());
            let r = _mm_maskz_srli_epi16(0b11111111, a, 2);
            let e = _mm_set1_epi16(0);
            assert_eq_m128i(r, e);
        }

        #[simd_test(enable = "avx512bw")]
        unsafe fn test_mm512_srlv_epi16() {
            let a = _mm512_set1_epi16(1 << 1);
            let count = _mm512_set1_epi16(2);
            let r = _mm512_srlv_epi16(a, count);
            let e = _mm512_set1_epi16(0);
            assert_eq_m512i(r, e);
        }

        #[simd_test(enable = "avx512bw")]
        unsafe fn test_mm512_mask_srlv_epi16() {
            let a = _mm512_set1_epi16(1 << 1);
            let count = _mm512_set1_epi16(2);
            let r = _mm512_mask_srlv_epi16(a, 0, a, count);
            assert_eq_m512i(r, a);
            let r = _mm512_mask_srlv_epi16(a, 0b11111111_11111111_11111111_11111111, a, count);
            let e = _mm512_set1_epi16(0);
            assert_eq_m512i(r, e);
        }

        #[simd_test(enable = "avx512bw")]
        unsafe fn test_mm512_maskz_srlv_epi16() {
            let a = _mm512_set1_epi16(1 << 1);
            let count = _mm512_set1_epi16(2);
            let r = _mm512_maskz_srlv_epi16(0, a, count);
            assert_eq_m512i(r, _mm512_setzero_si512());
            let r = _mm512_maskz_srlv_epi16(0b11111111_11111111_11111111_11111111, a, count);
            let e = _mm512_set1_epi16(0);
            assert_eq_m512i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm256_srlv_epi16() {
            let a = _mm256_set1_epi16(1 << 1);
            let count = _mm256_set1_epi16(2);
            let r = _mm256_srlv_epi16(a, count);
            let e = _mm256_set1_epi16(0);
            assert_eq_m256i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm256_mask_srlv_epi16() {
            let a = _mm256_set1_epi16(1 << 1);
            let count = _mm256_set1_epi16(2);
            let r = _mm256_mask_srlv_epi16(a, 0, a, count);
            assert_eq_m256i(r, a);
            let r = _mm256_mask_srlv_epi16(a, 0b11111111_11111111, a, count);
            let e = _mm256_set1_epi16(0);
            assert_eq_m256i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm256_maskz_srlv_epi16() {
            let a = _mm256_set1_epi16(1 << 1);
            let count = _mm256_set1_epi16(2);
            let r = _mm256_maskz_srlv_epi16(0, a, count);
            assert_eq_m256i(r, _mm256_setzero_si256());
            let r = _mm256_maskz_srlv_epi16(0b11111111_11111111, a, count);
            let e = _mm256_set1_epi16(0);
            assert_eq_m256i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm_srlv_epi16() {
            let a = _mm_set1_epi16(1 << 1);
            let count = _mm_set1_epi16(2);
            let r = _mm_srlv_epi16(a, count);
            let e = _mm_set1_epi16(0);
            assert_eq_m128i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm_mask_srlv_epi16() {
            let a = _mm_set1_epi16(1 << 1);
            let count = _mm_set1_epi16(2);
            let r = _mm_mask_srlv_epi16(a, 0, a, count);
            assert_eq_m128i(r, a);
            let r = _mm_mask_srlv_epi16(a, 0b11111111, a, count);
            let e = _mm_set1_epi16(0);
            assert_eq_m128i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm_maskz_srlv_epi16() {
            let a = _mm_set1_epi16(1 << 1);
            let count = _mm_set1_epi16(2);
            let r = _mm_maskz_srlv_epi16(0, a, count);
            assert_eq_m128i(r, _mm_setzero_si128());
            let r = _mm_maskz_srlv_epi16(0b11111111, a, count);
            let e = _mm_set1_epi16(0);
            assert_eq_m128i(r, e);
        }

        #[simd_test(enable = "avx512bw")]
        unsafe fn test_mm512_sra_epi16() {
            let a = _mm512_set1_epi16(8);
            let count = _mm_set1_epi16(1);
            let r = _mm512_sra_epi16(a, count);
            let e = _mm512_set1_epi16(0);
            assert_eq_m512i(r, e);
        }

        #[simd_test(enable = "avx512bw")]
        unsafe fn test_mm512_mask_sra_epi16() {
            let a = _mm512_set1_epi16(8);
            let count = _mm_set1_epi16(1);
            let r = _mm512_mask_sra_epi16(a, 0, a, count);
            assert_eq_m512i(r, a);
            let r = _mm512_mask_sra_epi16(a, 0b11111111_11111111_11111111_11111111, a, count);
            let e = _mm512_set1_epi16(0);
            assert_eq_m512i(r, e);
        }

        #[simd_test(enable = "avx512bw")]
        unsafe fn test_mm512_maskz_sra_epi16() {
            let a = _mm512_set1_epi16(8);
            let count = _mm_set1_epi16(1);
            let r = _mm512_maskz_sra_epi16(0, a, count);
            assert_eq_m512i(r, _mm512_setzero_si512());
            let r = _mm512_maskz_sra_epi16(0b11111111_11111111_11111111_11111111, a, count);
            let e = _mm512_set1_epi16(0);
            assert_eq_m512i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm256_mask_sra_epi16() {
            let a = _mm256_set1_epi16(8);
            let count = _mm_set1_epi16(1);
            let r = _mm256_mask_sra_epi16(a, 0, a, count);
            assert_eq_m256i(r, a);
            let r = _mm256_mask_sra_epi16(a, 0b11111111_11111111, a, count);
            let e = _mm256_set1_epi16(0);
            assert_eq_m256i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm256_maskz_sra_epi16() {
            let a = _mm256_set1_epi16(8);
            let count = _mm_set1_epi16(1);
            let r = _mm256_maskz_sra_epi16(0, a, count);
            assert_eq_m256i(r, _mm256_setzero_si256());
            let r = _mm256_maskz_sra_epi16(0b11111111_11111111, a, count);
            let e = _mm256_set1_epi16(0);
            assert_eq_m256i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm_mask_sra_epi16() {
            let a = _mm_set1_epi16(8);
            let count = _mm_set1_epi16(1);
            let r = _mm_mask_sra_epi16(a, 0, a, count);
            assert_eq_m128i(r, a);
            let r = _mm_mask_sra_epi16(a, 0b11111111, a, count);
            let e = _mm_set1_epi16(0);
            assert_eq_m128i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm_maskz_sra_epi16() {
            let a = _mm_set1_epi16(8);
            let count = _mm_set1_epi16(1);
            let r = _mm_maskz_sra_epi16(0, a, count);
            assert_eq_m128i(r, _mm_setzero_si128());
            let r = _mm_maskz_sra_epi16(0b11111111, a, count);
            let e = _mm_set1_epi16(0);
            assert_eq_m128i(r, e);
        }

        #[simd_test(enable = "avx512bw")]
        unsafe fn test_mm512_srai_epi16() {
            let a = _mm512_set1_epi16(8);
            let r = _mm512_srai_epi16(a, 2);
            let e = _mm512_set1_epi16(2);
            assert_eq_m512i(r, e);
        }

        #[simd_test(enable = "avx512bw")]
        unsafe fn test_mm512_mask_srai_epi16() {
            let a = _mm512_set1_epi16(8);
            let r = _mm512_mask_srai_epi16(a, 0, a, 2);
            assert_eq_m512i(r, a);
            let r = _mm512_mask_srai_epi16(a, 0b11111111_11111111_11111111_11111111, a, 2);
            let e = _mm512_set1_epi16(2);
            assert_eq_m512i(r, e);
        }

        #[simd_test(enable = "avx512bw")]
        unsafe fn test_mm512_maskz_srai_epi16() {
            let a = _mm512_set1_epi16(8);
            let r = _mm512_maskz_srai_epi16(0, a, 2);
            assert_eq_m512i(r, _mm512_setzero_si512());
            let r = _mm512_maskz_srai_epi16(0b11111111_11111111_11111111_11111111, a, 2);
            let e = _mm512_set1_epi16(2);
            assert_eq_m512i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm256_mask_srai_epi16() {
            let a = _mm256_set1_epi16(8);
            let r = _mm256_mask_srai_epi16(a, 0, a, 2);
            assert_eq_m256i(r, a);
            let r = _mm256_mask_srai_epi16(a, 0b11111111_11111111, a, 2);
            let e = _mm256_set1_epi16(2);
            assert_eq_m256i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm256_maskz_srai_epi16() {
            let a = _mm256_set1_epi16(8);
            let r = _mm256_maskz_srai_epi16(0, a, 2);
            assert_eq_m256i(r, _mm256_setzero_si256());
            let r = _mm256_maskz_srai_epi16(0b11111111_11111111, a, 2);
            let e = _mm256_set1_epi16(2);
            assert_eq_m256i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm_mask_srai_epi16() {
            let a = _mm_set1_epi16(8);
            let r = _mm_mask_srai_epi16(a, 0, a, 2);
            assert_eq_m128i(r, a);
            let r = _mm_mask_srai_epi16(a, 0b11111111, a, 2);
            let e = _mm_set1_epi16(2);
            assert_eq_m128i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm_maskz_srai_epi16() {
            let a = _mm_set1_epi16(8);
            let r = _mm_maskz_srai_epi16(0, a, 2);
            assert_eq_m128i(r, _mm_setzero_si128());
            let r = _mm_maskz_srai_epi16(0b11111111, a, 2);
            let e = _mm_set1_epi16(2);
            assert_eq_m128i(r, e);
        }

        #[simd_test(enable = "avx512bw")]
        unsafe fn test_mm512_srav_epi16() {
            let a = _mm512_set1_epi16(8);
            let count = _mm512_set1_epi16(2);
            let r = _mm512_srav_epi16(a, count);
            let e = _mm512_set1_epi16(2);
            assert_eq_m512i(r, e);
        }

        #[simd_test(enable = "avx512bw")]
        unsafe fn test_mm512_mask_srav_epi16() {
            let a = _mm512_set1_epi16(8);
            let count = _mm512_set1_epi16(2);
            let r = _mm512_mask_srav_epi16(a, 0, a, count);
            assert_eq_m512i(r, a);
            let r = _mm512_mask_srav_epi16(a, 0b11111111_11111111_11111111_11111111, a, count);
            let e = _mm512_set1_epi16(2);
            assert_eq_m512i(r, e);
        }

        #[simd_test(enable = "avx512bw")]
        unsafe fn test_mm512_maskz_srav_epi16() {
            let a = _mm512_set1_epi16(8);
            let count = _mm512_set1_epi16(2);
            let r = _mm512_maskz_srav_epi16(0, a, count);
            assert_eq_m512i(r, _mm512_setzero_si512());
            let r = _mm512_maskz_srav_epi16(0b11111111_11111111_11111111_11111111, a, count);
            let e = _mm512_set1_epi16(2);
            assert_eq_m512i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm256_srav_epi16() {
            let a = _mm256_set1_epi16(8);
            let count = _mm256_set1_epi16(2);
            let r = _mm256_srav_epi16(a, count);
            let e = _mm256_set1_epi16(2);
            assert_eq_m256i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm256_mask_srav_epi16() {
            let a = _mm256_set1_epi16(8);
            let count = _mm256_set1_epi16(2);
            let r = _mm256_mask_srav_epi16(a, 0, a, count);
            assert_eq_m256i(r, a);
            let r = _mm256_mask_srav_epi16(a, 0b11111111_11111111, a, count);
            let e = _mm256_set1_epi16(2);
            assert_eq_m256i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm256_maskz_srav_epi16() {
            let a = _mm256_set1_epi16(8);
            let count = _mm256_set1_epi16(2);
            let r = _mm256_maskz_srav_epi16(0, a, count);
            assert_eq_m256i(r, _mm256_setzero_si256());
            let r = _mm256_maskz_srav_epi16(0b11111111_11111111, a, count);
            let e = _mm256_set1_epi16(2);
            assert_eq_m256i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm_srav_epi16() {
            let a = _mm_set1_epi16(8);
            let count = _mm_set1_epi16(2);
            let r = _mm_srav_epi16(a, count);
            let e = _mm_set1_epi16(2);
            assert_eq_m128i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm_mask_srav_epi16() {
            let a = _mm_set1_epi16(8);
            let count = _mm_set1_epi16(2);
            let r = _mm_mask_srav_epi16(a, 0, a, count);
            assert_eq_m128i(r, a);
            let r = _mm_mask_srav_epi16(a, 0b11111111, a, count);
            let e = _mm_set1_epi16(2);
            assert_eq_m128i(r, e);
        }

        #[simd_test(enable = "avx512bw,avx512vl")]
        unsafe fn test_mm_maskz_srav_epi16() {
            let a = _mm_set1_epi16(8);
            let count = _mm_set1_epi16(2);
            let r = _mm_maskz_srav_epi16(0, a, count);
            assert_eq_m128i(r, _mm_setzero_si128());
            let r = _mm_maskz_srav_epi16(0b11111111, a, count);
            let e = _mm_set1_epi16(2);
            assert_eq_m128i(r, e);
        }
    */
    #[simd_test(enable = "avx512vbmi")]
    unsafe fn test_mm512_permutex2var_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        #[rustfmt::skip]
        let idx = _mm512_set_epi8(1,  1<<6, 2,  1<<6, 3,  1<<6, 4,  1<<6, 5,  1<<6, 6,  1<<6, 7,  1<<6, 8,  1<<6,
                                  9,  1<<6, 10, 1<<6, 11, 1<<6, 12, 1<<6, 13, 1<<6, 14, 1<<6, 15, 1<<6, 16, 1<<6,
                                  17, 1<<6, 18, 1<<6, 19, 1<<6, 20, 1<<6, 21, 1<<6, 22, 1<<6, 23, 1<<6, 24, 1<<6,
                                  25, 1<<6, 26, 1<<6, 27, 1<<6, 28, 1<<6, 29, 1<<6, 30, 1<<6, 31, 1<<6, 32, 1<<6);
        let b = _mm512_set1_epi8(100);
        let r = _mm512_permutex2var_epi8(a, idx, b);
        #[rustfmt::skip]
        let e = _mm512_set_epi8(
            62, 100, 61, 100, 60, 100, 59, 100, 58, 100, 57, 100, 56, 100, 55, 100,
            54, 100, 53, 100, 52, 100, 51, 100, 50, 100, 49, 100, 48, 100, 47, 100,
            46, 100, 45, 100, 44, 100, 43, 100, 42, 100, 41, 100, 40, 100, 39, 100,
            38, 100, 37, 100, 36, 100, 35, 100, 34, 100, 33, 100, 32, 100, 31, 100,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi")]
    unsafe fn test_mm512_mask_permutex2var_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        #[rustfmt::skip]
        let idx = _mm512_set_epi8(1,  1<<6, 2,  1<<6, 3,  1<<6, 4,  1<<6, 5,  1<<6, 6,  1<<6, 7,  1<<6, 8,  1<<6,
                                  9,  1<<6, 10, 1<<6, 11, 1<<6, 12, 1<<6, 13, 1<<6, 14, 1<<6, 15, 1<<6, 16, 1<<6,
                                  17, 1<<6, 18, 1<<6, 19, 1<<6, 20, 1<<6, 21, 1<<6, 22, 1<<6, 23, 1<<6, 24, 1<<6,
                                  25, 1<<6, 26, 1<<6, 27, 1<<6, 28, 1<<6, 29, 1<<6, 30, 1<<6, 31, 1<<6, 32, 1<<6);
        let b = _mm512_set1_epi8(100);
        let r = _mm512_mask_permutex2var_epi8(a, 0, idx, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_permutex2var_epi8(
            a,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            idx,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(
            62, 100, 61, 100, 60, 100, 59, 100, 58, 100, 57, 100, 56, 100, 55, 100,
            54, 100, 53, 100, 52, 100, 51, 100, 50, 100, 49, 100, 48, 100, 47, 100,
            46, 100, 45, 100, 44, 100, 43, 100, 42, 100, 41, 100, 40, 100, 39, 100,
            38, 100, 37, 100, 36, 100, 35, 100, 34, 100, 33, 100, 32, 100, 31, 100,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi")]
    unsafe fn test_mm512_maskz_permutex2var_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        #[rustfmt::skip]
        let idx = _mm512_set_epi8(1,  1<<6, 2,  1<<6, 3,  1<<6, 4,  1<<6, 5,  1<<6, 6,  1<<6, 7,  1<<6, 8,  1<<6,
                                  9,  1<<6, 10, 1<<6, 11, 1<<6, 12, 1<<6, 13, 1<<6, 14, 1<<6, 15, 1<<6, 16, 1<<6,
                                  17, 1<<6, 18, 1<<6, 19, 1<<6, 20, 1<<6, 21, 1<<6, 22, 1<<6, 23, 1<<6, 24, 1<<6,
                                  25, 1<<6, 26, 1<<6, 27, 1<<6, 28, 1<<6, 29, 1<<6, 30, 1<<6, 31, 1<<6, 32, 1<<6);
        let b = _mm512_set1_epi8(100);
        let r = _mm512_maskz_permutex2var_epi8(0, a, idx, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_permutex2var_epi8(
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            a,
            idx,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(
            62, 100, 61, 100, 60, 100, 59, 100, 58, 100, 57, 100, 56, 100, 55, 100,
            54, 100, 53, 100, 52, 100, 51, 100, 50, 100, 49, 100, 48, 100, 47, 100,
            46, 100, 45, 100, 44, 100, 43, 100, 42, 100, 41, 100, 40, 100, 39, 100,
            38, 100, 37, 100, 36, 100, 35, 100, 34, 100, 33, 100, 32, 100, 31, 100,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi")]
    unsafe fn test_mm512_mask2_permutex2var_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        #[rustfmt::skip]
        let idx = _mm512_set_epi8(1,  1<<6, 2,  1<<6, 3,  1<<6, 4,  1<<6, 5,  1<<6, 6,  1<<6, 7,  1<<6, 8,  1<<6,
                                  9,  1<<6, 10, 1<<6, 11, 1<<6, 12, 1<<6, 13, 1<<6, 14, 1<<6, 15, 1<<6, 16, 1<<6,
                                  17, 1<<6, 18, 1<<6, 19, 1<<6, 20, 1<<6, 21, 1<<6, 22, 1<<6, 23, 1<<6, 24, 1<<6,
                                  25, 1<<6, 26, 1<<6, 27, 1<<6, 28, 1<<6, 29, 1<<6, 30, 1<<6, 31, 1<<6, 32, 1<<6);
        let b = _mm512_set1_epi8(100);
        let r = _mm512_mask2_permutex2var_epi8(a, idx, 0, b);
        assert_eq_m512i(r, idx);
        let r = _mm512_mask2_permutex2var_epi8(
            a,
            idx,
            0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111,
            b,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(
            62, 100, 61, 100, 60, 100, 59, 100, 58, 100, 57, 100, 56, 100, 55, 100,
            54, 100, 53, 100, 52, 100, 51, 100, 50, 100, 49, 100, 48, 100, 47, 100,
            46, 100, 45, 100, 44, 100, 43, 100, 42, 100, 41, 100, 40, 100, 39, 100,
            38, 100, 37, 100, 36, 100, 35, 100, 34, 100, 33, 100, 32, 100, 31, 100,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi,avx512vl")]
    unsafe fn test_mm256_permutex2var_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        #[rustfmt::skip]
        let idx = _mm256_set_epi8(1,  1<<5, 2,  1<<5, 3,  1<<5, 4,  1<<5, 5,  1<<5, 6,  1<<5, 7,  1<<5, 8,  1<<5,
                                  9,  1<<5, 10, 1<<5, 11, 1<<5, 12, 1<<5, 13, 1<<5, 14, 1<<5, 15, 1<<5, 16, 1<<5);
        let b = _mm256_set1_epi8(100);
        let r = _mm256_permutex2var_epi8(a, idx, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(
            30, 100, 29, 100, 28, 100, 27, 100, 26, 100, 25, 100, 24, 100, 23, 100,
            22, 100, 21, 100, 20, 100, 19, 100, 18, 100, 17, 100, 16, 100, 15, 100,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi,avx512vl")]
    unsafe fn test_mm256_mask_permutex2var_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        #[rustfmt::skip]
        let idx = _mm256_set_epi8(1,  1<<5, 2,  1<<5, 3,  1<<5, 4,  1<<5, 5,  1<<5, 6,  1<<5, 7,  1<<5, 8,  1<<5,
                                  9,  1<<5, 10, 1<<5, 11, 1<<5, 12, 1<<5, 13, 1<<5, 14, 1<<5, 15, 1<<5, 16, 1<<5);
        let b = _mm256_set1_epi8(100);
        let r = _mm256_mask_permutex2var_epi8(a, 0, idx, b);
        assert_eq_m256i(r, a);
        let r = _mm256_mask_permutex2var_epi8(a, 0b11111111_11111111_11111111_11111111, idx, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(
            30, 100, 29, 100, 28, 100, 27, 100, 26, 100, 25, 100, 24, 100, 23, 100,
            22, 100, 21, 100, 20, 100, 19, 100, 18, 100, 17, 100, 16, 100, 15, 100,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi,avx512vl")]
    unsafe fn test_mm256_maskz_permutex2var_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        #[rustfmt::skip]
        let idx = _mm256_set_epi8(1,  1<<5, 2,  1<<5, 3,  1<<5, 4,  1<<5, 5,  1<<5, 6,  1<<5, 7,  1<<5, 8,  1<<5,
                                  9,  1<<5, 10, 1<<5, 11, 1<<5, 12, 1<<5, 13, 1<<5, 14, 1<<5, 15, 1<<5, 16, 1<<5);
        let b = _mm256_set1_epi8(100);
        let r = _mm256_maskz_permutex2var_epi8(0, a, idx, b);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm256_maskz_permutex2var_epi8(0b11111111_11111111_11111111_11111111, a, idx, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(
            30, 100, 29, 100, 28, 100, 27, 100, 26, 100, 25, 100, 24, 100, 23, 100,
            22, 100, 21, 100, 20, 100, 19, 100, 18, 100, 17, 100, 16, 100, 15, 100,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi,avx512vl")]
    unsafe fn test_mm256_mask2_permutex2var_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        #[rustfmt::skip]
        let idx = _mm256_set_epi8(1,  1<<5, 2,  1<<5, 3,  1<<5, 4,  1<<5, 5,  1<<5, 6,  1<<5, 7,  1<<5, 8,  1<<5,
                                  9,  1<<5, 10, 1<<5, 11, 1<<5, 12, 1<<5, 13, 1<<5, 14, 1<<5, 15, 1<<5, 16, 1<<5);
        let b = _mm256_set1_epi8(100);
        let r = _mm256_mask2_permutex2var_epi8(a, idx, 0, b);
        assert_eq_m256i(r, idx);
        let r = _mm256_mask2_permutex2var_epi8(a, idx, 0b11111111_11111111_11111111_11111111, b);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(
            30, 100, 29, 100, 28, 100, 27, 100, 26, 100, 25, 100, 24, 100, 23, 100,
            22, 100, 21, 100, 20, 100, 19, 100, 18, 100, 17, 100, 16, 100, 15, 100,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi,avx512vl")]
    unsafe fn test_mm_permutex2var_epi8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let idx = _mm_set_epi8(1, 1 << 4, 2, 1 << 4, 3, 1 << 4, 4, 1 << 4, 5, 1 << 4, 6, 1 << 4, 7, 1 << 4, 8, 1 << 4);
        let b = _mm_set1_epi8(100);
        let r = _mm_permutex2var_epi8(a, idx, b);
        let e = _mm_set_epi8(
            14, 100, 13, 100, 12, 100, 11, 100, 10, 100, 9, 100, 8, 100, 7, 100,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi,avx512vl")]
    unsafe fn test_mm_mask_permutex2var_epi8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let idx = _mm_set_epi8(1, 1 << 4, 2, 1 << 4, 3, 1 << 4, 4, 1 << 4, 5, 1 << 4, 6, 1 << 4, 7, 1 << 4, 8, 1 << 4);
        let b = _mm_set1_epi8(100);
        let r = _mm_mask_permutex2var_epi8(a, 0, idx, b);
        assert_eq_m128i(r, a);
        let r = _mm_mask_permutex2var_epi8(a, 0b11111111_11111111, idx, b);
        let e = _mm_set_epi8(
            14, 100, 13, 100, 12, 100, 11, 100, 10, 100, 9, 100, 8, 100, 7, 100,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi,avx512vl")]
    unsafe fn test_mm_maskz_permutex2var_epi8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let idx = _mm_set_epi8(1, 1 << 4, 2, 1 << 4, 3, 1 << 4, 4, 1 << 4, 5, 1 << 4, 6, 1 << 4, 7, 1 << 4, 8, 1 << 4);
        let b = _mm_set1_epi8(100);
        let r = _mm_maskz_permutex2var_epi8(0, a, idx, b);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm_maskz_permutex2var_epi8(0b11111111_11111111, a, idx, b);
        let e = _mm_set_epi8(
            14, 100, 13, 100, 12, 100, 11, 100, 10, 100, 9, 100, 8, 100, 7, 100,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi,avx512vl")]
    unsafe fn test_mm_mask2_permutex2var_epi8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        #[rustfmt::skip]
        let idx = _mm_set_epi8(1, 1 << 4, 2, 1 << 4, 3, 1 << 4, 4, 1 << 4, 5, 1 << 4, 6, 1 << 4, 7, 1 << 4, 8, 1 << 4);
        let b = _mm_set1_epi8(100);
        let r = _mm_mask2_permutex2var_epi8(a, idx, 0, b);
        assert_eq_m128i(r, idx);
        let r = _mm_mask2_permutex2var_epi8(a, idx, 0b11111111_11111111, b);
        let e = _mm_set_epi8(
            14, 100, 13, 100, 12, 100, 11, 100, 10, 100, 9, 100, 8, 100, 7, 100,
        );
        assert_eq_m128i(r, e);
    }
    /*
    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_permutexvar_epi16() {
        let idx = _mm512_set1_epi16(1);
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm512_permutexvar_epi16(idx, a);
        let e = _mm512_set1_epi16(30);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_permutexvar_epi16() {
        let idx = _mm512_set1_epi16(1);
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm512_mask_permutexvar_epi16(a, 0, idx, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_permutexvar_epi16(a, 0b11111111_11111111_11111111_11111111, idx, a);
        let e = _mm512_set1_epi16(30);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_permutexvar_epi16() {
        let idx = _mm512_set1_epi16(1);
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm512_maskz_permutexvar_epi16(0, idx, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_permutexvar_epi16(0b11111111_11111111_11111111_11111111, idx, a);
        let e = _mm512_set1_epi16(30);
        assert_eq_m512i(r, e);
    }

    */
}
