use crate::core_arch::{simd::*, /*simd_llvm::*,*/ x86::*};

#[cfg(test)]
use stdarch_test::assert_instr;

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in writemask k) to dst, and pass through the remaining elements from src.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_compress_epi16&expand=1192)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[cfg_attr(test, assert_instr(vpcompressw))]
pub unsafe fn _mm512_mask_compress_epi16(src: __m512i, k: __mmask32, a: __m512i) -> __m512i {
    transmute(vpcompressw(a.as_i16x32(), src.as_i16x32(), k))
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in zeromask k) to dst, and set the remaining elements to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_compress_epi16&expand=1193)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[cfg_attr(test, assert_instr(vpcompressw))]
pub unsafe fn _mm512_maskz_compress_epi16(k: __mmask32, a: __m512i) -> __m512i {
    transmute(vpcompressw(
        a.as_i16x32(),
        _mm512_setzero_si512().as_i16x32(),
        k,
    ))
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in writemask k) to dst, and pass through the remaining elements from src.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_compress_epi16&expand=1190)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpcompressw))]
pub unsafe fn _mm256_mask_compress_epi16(src: __m256i, k: __mmask16, a: __m256i) -> __m256i {
    transmute(vpcompressw256(a.as_i16x16(), src.as_i16x16(), k))
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in zeromask k) to dst, and set the remaining elements to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_compress_epi16&expand=1191)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpcompressw))]
pub unsafe fn _mm256_maskz_compress_epi16(k: __mmask16, a: __m256i) -> __m256i {
    transmute(vpcompressw256(
        a.as_i16x16(),
        _mm256_setzero_si256().as_i16x16(),
        k,
    ))
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in writemask k) to dst, and pass through the remaining elements from src.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_compress_epi16&expand=1188)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpcompressw))]
pub unsafe fn _mm_mask_compress_epi16(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    transmute(vpcompressw128(a.as_i16x8(), src.as_i16x8(), k))
}

/// Contiguously store the active 16-bit integers in a (those with their respective bit set in zeromask k) to dst, and set the remaining elements to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_compress_epi16&expand=1189)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpcompressw))]
pub unsafe fn _mm_maskz_compress_epi16(k: __mmask8, a: __m128i) -> __m128i {
    transmute(vpcompressw128(
        a.as_i16x8(),
        _mm_setzero_si128().as_i16x8(),
        k,
    ))
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in writemask k) to dst, and pass through the remaining elements from src.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_compress_epi8&expand=1210)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[cfg_attr(test, assert_instr(vpcompressb))]
pub unsafe fn _mm512_mask_compress_epi8(src: __m512i, k: __mmask64, a: __m512i) -> __m512i {
    transmute(vpcompressb(a.as_i8x64(), src.as_i8x64(), k))
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in zeromask k) to dst, and set the remaining elements to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_compress_epi8&expand=1211)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[cfg_attr(test, assert_instr(vpcompressb))]
pub unsafe fn _mm512_maskz_compress_epi8(k: __mmask64, a: __m512i) -> __m512i {
    transmute(vpcompressb(
        a.as_i8x64(),
        _mm512_setzero_si512().as_i8x64(),
        k,
    ))
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in writemask k) to dst, and pass through the remaining elements from src.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_compress_epi8&expand=1208)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpcompressb))]
pub unsafe fn _mm256_mask_compress_epi8(src: __m256i, k: __mmask32, a: __m256i) -> __m256i {
    transmute(vpcompressb256(a.as_i8x32(), src.as_i8x32(), k))
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in zeromask k) to dst, and set the remaining elements to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_compress_epi8&expand=1209)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpcompressb))]
pub unsafe fn _mm256_maskz_compress_epi8(k: __mmask32, a: __m256i) -> __m256i {
    transmute(vpcompressb256(
        a.as_i8x32(),
        _mm256_setzero_si256().as_i8x32(),
        k,
    ))
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in writemask k) to dst, and pass through the remaining elements from src.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_compress_epi8&expand=1206)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpcompressb))]
pub unsafe fn _mm_mask_compress_epi8(src: __m128i, k: __mmask16, a: __m128i) -> __m128i {
    transmute(vpcompressb128(a.as_i8x16(), src.as_i8x16(), k))
}

/// Contiguously store the active 8-bit integers in a (those with their respective bit set in zeromask k) to dst, and set the remaining elements to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_compress_epi8&expand=1207)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpcompressb))]
pub unsafe fn _mm_maskz_compress_epi8(k: __mmask16, a: __m128i) -> __m128i {
    transmute(vpcompressb128(
        a.as_i8x16(),
        _mm_setzero_si128().as_i8x16(),
        k,
    ))
}

/// Load contiguous active 16-bit integers from a (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_expand_epi16&expand=2310)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[cfg_attr(test, assert_instr(vpexpandw))]
pub unsafe fn _mm512_mask_expand_epi16(src: __m512i, k: __mmask32, a: __m512i) -> __m512i {
    transmute(vpexpandw(a.as_i16x32(), src.as_i16x32(), k))
}

/// Load contiguous active 16-bit integers from a (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_expand_epi16&expand=2311)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[cfg_attr(test, assert_instr(vpexpandw))]
pub unsafe fn _mm512_maskz_expand_epi16(k: __mmask32, a: __m512i) -> __m512i {
    transmute(vpexpandw(
        a.as_i16x32(),
        _mm512_setzero_si512().as_i16x32(),
        k,
    ))
}

/// Load contiguous active 16-bit integers from a (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_expand_epi16&expand=2308)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpexpandw))]
pub unsafe fn _mm256_mask_expand_epi16(src: __m256i, k: __mmask16, a: __m256i) -> __m256i {
    transmute(vpexpandw256(a.as_i16x16(), src.as_i16x16(), k))
}

/// Load contiguous active 16-bit integers from a (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_expand_epi16&expand=2309)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpexpandw))]
pub unsafe fn _mm256_maskz_expand_epi16(k: __mmask16, a: __m256i) -> __m256i {
    transmute(vpexpandw256(
        a.as_i16x16(),
        _mm256_setzero_si256().as_i16x16(),
        k,
    ))
}

/// Load contiguous active 16-bit integers from a (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_expand_epi16&expand=2306)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpexpandw))]
pub unsafe fn _mm_mask_expand_epi16(src: __m128i, k: __mmask8, a: __m128i) -> __m128i {
    transmute(vpexpandw128(a.as_i16x8(), src.as_i16x8(), k))
}

/// Load contiguous active 16-bit integers from a (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_expand_epi16&expand=2307)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpexpandw))]
pub unsafe fn _mm_maskz_expand_epi16(k: __mmask8, a: __m128i) -> __m128i {
    transmute(vpexpandw128(
        a.as_i16x8(),
        _mm_setzero_si128().as_i16x8(),
        k,
    ))
}

/// Load contiguous active 8-bit integers from a (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_expand_epi8&expand=2328)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[cfg_attr(test, assert_instr(vpexpandb))]
pub unsafe fn _mm512_mask_expand_epi8(src: __m512i, k: __mmask64, a: __m512i) -> __m512i {
    transmute(vpexpandb(a.as_i8x64(), src.as_i8x64(), k))
}

/// Load contiguous active 8-bit integers from a (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_expand_epi8&expand=2329)
#[inline]
#[target_feature(enable = "avx512vbmi2")]
#[cfg_attr(test, assert_instr(vpexpandb))]
pub unsafe fn _mm512_maskz_expand_epi8(k: __mmask64, a: __m512i) -> __m512i {
    transmute(vpexpandb(
        a.as_i8x64(),
        _mm512_setzero_si512().as_i8x64(),
        k,
    ))
}

/// Load contiguous active 8-bit integers from a (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_mask_expand_epi8&expand=2326)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpexpandb))]
pub unsafe fn _mm256_mask_expand_epi8(src: __m256i, k: __mmask32, a: __m256i) -> __m256i {
    transmute(vpexpandb256(a.as_i8x32(), src.as_i8x32(), k))
}

/// Load contiguous active 8-bit integers from a (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm256_maskz_expand_epi8&expand=2327)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpexpandb))]
pub unsafe fn _mm256_maskz_expand_epi8(k: __mmask32, a: __m256i) -> __m256i {
    transmute(vpexpandb256(
        a.as_i8x32(),
        _mm256_setzero_si256().as_i8x32(),
        k,
    ))
}

/// Load contiguous active 8-bit integers from a (those with their respective bit set in mask k), and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_mask_expand_epi8&expand=2324)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpexpandb))]
pub unsafe fn _mm_mask_expand_epi8(src: __m128i, k: __mmask16, a: __m128i) -> __m128i {
    transmute(vpexpandb128(a.as_i8x16(), src.as_i8x16(), k))
}

/// Load contiguous active 8-bit integers from a (those with their respective bit set in mask k), and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm_maskz_expand_epi8&expand=2325)
#[inline]
#[target_feature(enable = "avx512vbmi2,avx512vl")]
#[cfg_attr(test, assert_instr(vpexpandb))]
pub unsafe fn _mm_maskz_expand_epi8(k: __mmask16, a: __m128i) -> __m128i {
    transmute(vpexpandb128(
        a.as_i8x16(),
        _mm_setzero_si128().as_i8x16(),
        k,
    ))
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.mask.compress.w.512"]
    fn vpcompressw(a: i16x32, src: i16x32, mask: u32) -> i16x32;
    #[link_name = "llvm.x86.avx512.mask.compress.w.256"]
    fn vpcompressw256(a: i16x16, src: i16x16, mask: u16) -> i16x16;
    #[link_name = "llvm.x86.avx512.mask.compress.w.128"]
    fn vpcompressw128(a: i16x8, src: i16x8, mask: u8) -> i16x8;

    #[link_name = "llvm.x86.avx512.mask.compress.b.512"]
    fn vpcompressb(a: i8x64, src: i8x64, mask: u64) -> i8x64;
    #[link_name = "llvm.x86.avx512.mask.compress.b.256"]
    fn vpcompressb256(a: i8x32, src: i8x32, mask: u32) -> i8x32;
    #[link_name = "llvm.x86.avx512.mask.compress.b.128"]
    fn vpcompressb128(a: i8x16, src: i8x16, mask: u16) -> i8x16;

    #[link_name = "llvm.x86.avx512.mask.expand.w.512"]
    fn vpexpandw(a: i16x32, src: i16x32, mask: u32) -> i16x32;
    #[link_name = "llvm.x86.avx512.mask.expand.w.256"]
    fn vpexpandw256(a: i16x16, src: i16x16, mask: u16) -> i16x16;
    #[link_name = "llvm.x86.avx512.mask.expand.w.128"]
    fn vpexpandw128(a: i16x8, src: i16x8, mask: u8) -> i16x8;

    #[link_name = "llvm.x86.avx512.mask.expand.b.512"]
    fn vpexpandb(a: i8x64, src: i8x64, mask: u64) -> i8x64;
    #[link_name = "llvm.x86.avx512.mask.expand.b.256"]
    fn vpexpandb256(a: i8x32, src: i8x32, mask: u32) -> i8x32;
    #[link_name = "llvm.x86.avx512.mask.expand.b.128"]
    fn vpexpandb128(a: i8x16, src: i8x16, mask: u16) -> i8x16;
}

#[cfg(test)]
mod tests {

    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_compress_epi16() {
        let src = _mm512_set1_epi16(200);
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm512_mask_compress_epi16(src, 0b01010101_01010101_01010101_01010101, a);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(
            200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200,
            1,   3,   5,   7,   9,   11,  13,  15,  17,  19,  21,  23,  25,  27,  29,  31,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_compress_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm512_maskz_compress_epi16(0b01010101_01010101_01010101_01010101, a);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(
            0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_compress_epi16() {
        let src = _mm256_set1_epi16(200);
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm256_mask_compress_epi16(src, 0b01010101_01010101, a);
        let e = _mm256_set_epi16(
            200, 200, 200, 200, 200, 200, 200, 200, 1, 3, 5, 7, 9, 11, 13, 15,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_compress_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm256_maskz_compress_epi16(0b01010101_01010101, a);
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 7, 9, 11, 13, 15);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_compress_epi16() {
        let src = _mm_set1_epi16(200);
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_mask_compress_epi16(src, 0b01010101, a);
        let e = _mm_set_epi16(200, 200, 200, 200, 1, 3, 5, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_compress_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_maskz_compress_epi16(0b01010101, a);
        let e = _mm_set_epi16(0, 0, 0, 0, 1, 3, 5, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_compress_epi8() {
        let src = _mm512_set1_epi8(100);
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        let r = _mm512_mask_compress_epi8(
            src,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101,
            a,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            1,   3,   5,   7,   9,   11,  13,  15,  17,  19,  21,  23,  25,  27,  29,  31,
            33,  35,  37,  39,  41,  43,  45,  47,  49,  51,  53,  55,  57,  59,  61,  63,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_compress_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        let r = _mm512_maskz_compress_epi8(
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101,
            a,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
            33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_compress_epi8() {
        let src = _mm256_set1_epi8(100);
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm256_mask_compress_epi8(src, 0b01010101_01010101_01010101_01010101, a);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            1,   3,   5,   7,   9,   11,  13,  15,  17,  19,  21,  23,  25,  27,  29,  31,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_compress_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm256_maskz_compress_epi8(0b01010101_01010101_01010101_01010101, a);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(
            0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
            1,  3,  5,  7,  9,  11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_compress_epi8() {
        let src = _mm_set1_epi8(100);
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm_mask_compress_epi8(src, 0b01010101_01010101, a);
        let e = _mm_set_epi8(
            100, 100, 100, 100, 100, 100, 100, 100, 1, 3, 5, 7, 9, 11, 13, 15,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_compress_epi8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm_maskz_compress_epi8(0b01010101_01010101, a);
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 5, 7, 9, 11, 13, 15);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_expand_epi16() {
        let src = _mm512_set1_epi16(200);
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm512_mask_expand_epi16(src, 0b01010101_01010101_01010101_01010101, a);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(
            200, 16, 200, 17, 200, 18, 200, 19, 200, 20, 200, 21, 200, 22, 200, 23,
            200, 24, 200, 25, 200, 26, 200, 27, 200, 28, 200, 29, 200, 30, 200, 31,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_expand_epi16() {
        #[rustfmt::skip]
        let a = _mm512_set_epi16(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm512_maskz_expand_epi16(0b01010101_01010101_01010101_01010101, a);
        #[rustfmt::skip]
        let e = _mm512_set_epi16(0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23,
                                 0, 24, 0, 25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_expand_epi16() {
        let src = _mm256_set1_epi16(200);
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm256_mask_expand_epi16(src, 0b01010101_01010101, a);
        let e = _mm256_set_epi16(
            200, 8, 200, 9, 200, 10, 200, 11, 200, 12, 200, 13, 200, 14, 200, 15,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_expand_epi16() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm256_maskz_expand_epi16(0b01010101_01010101, a);
        let e = _mm256_set_epi16(0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_expand_epi16() {
        let src = _mm_set1_epi16(200);
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_mask_expand_epi16(src, 0b01010101, a);
        let e = _mm_set_epi16(200, 4, 200, 5, 200, 6, 200, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_expand_epi16() {
        let a = _mm_set_epi16(0, 1, 2, 3, 4, 5, 6, 7);
        let r = _mm_maskz_expand_epi16(0b01010101, a);
        let e = _mm_set_epi16(0, 4, 0, 5, 0, 6, 0, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_mask_expand_epi8() {
        let src = _mm512_set1_epi8(100);
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        let r = _mm512_mask_expand_epi8(
            src,
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101,
            a,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(
            100, 32, 100, 33, 100, 34, 100, 35, 100, 36, 100, 37, 100, 38, 100, 39,
            100, 40, 100, 41, 100, 42, 100, 43, 100, 44, 100, 45, 100, 46, 100, 47,
            100, 48, 100, 49, 100, 50, 100, 51, 100, 52, 100, 53, 100, 54, 100, 55,
            100, 56, 100, 57, 100, 58, 100, 59, 100, 60, 100, 61, 100, 62, 100, 63,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2")]
    unsafe fn test_mm512_maskz_expand_epi8() {
        #[rustfmt::skip]
        let a = _mm512_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                                32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
                                48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63);
        let r = _mm512_maskz_expand_epi8(
            0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101,
            a,
        );
        #[rustfmt::skip]
        let e = _mm512_set_epi8(
            0, 32, 0, 33, 0, 34, 0, 35, 0, 36, 0, 37, 0, 38, 0, 39,
            0, 40, 0, 41, 0, 42, 0, 43, 0, 44, 0, 45, 0, 46, 0, 47,
            0, 48, 0, 49, 0, 50, 0, 51, 0, 52, 0, 53, 0, 54, 0, 55,
            0, 56, 0, 57, 0, 58, 0, 59, 0, 60, 0, 61, 0, 62, 0, 63,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_mask_expand_epi8() {
        let src = _mm256_set1_epi8(100);
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm256_mask_expand_epi8(src, 0b01010101_01010101_01010101_01010101, a);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(
            100, 16, 100, 17, 100, 18, 100, 19, 100, 20, 100, 21, 100, 22, 100, 23,
            100, 24, 100, 25, 100, 26, 100, 27, 100, 28, 100, 29, 100, 30, 100, 31,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm256_maskz_expand_epi8() {
        #[rustfmt::skip]
        let a = _mm256_set_epi8(0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let r = _mm256_maskz_expand_epi8(0b01010101_01010101_01010101_01010101, a);
        #[rustfmt::skip]
        let e = _mm256_set_epi8(
            0, 16, 0, 17, 0, 18, 0, 19, 0, 20, 0, 21, 0, 22, 0, 23,
            0, 24, 0, 25, 0, 26, 0, 27, 0, 28, 0, 29, 0, 30, 0, 31,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_mask_expand_epi8() {
        let src = _mm_set1_epi8(100);
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm_mask_expand_epi8(src, 0b01010101_01010101, a);
        let e = _mm_set_epi8(
            100, 8, 100, 9, 100, 10, 100, 11, 100, 12, 100, 13, 100, 14, 100, 15,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512vbmi2,avx512vl")]
    unsafe fn test_mm_maskz_expand_epi8() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm_maskz_expand_epi8(0b01010101_01010101, a);
        let e = _mm_set_epi8(0, 8, 0, 9, 0, 10, 0, 11, 0, 12, 0, 13, 0, 14, 0, 15);
        assert_eq_m128i(r, e);
    }
}
