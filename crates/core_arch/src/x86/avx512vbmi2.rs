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

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.mask.compress.w.512"]
    fn vpcompressw(a: i16x32, src: i16x32, mask: u32) -> i16x32;

    #[link_name = "llvm.x86.avx512.mask.compress.b.512"]
    fn vpcompressb(a: i8x64, src: i8x64, mask: u64) -> i8x64;
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
}
