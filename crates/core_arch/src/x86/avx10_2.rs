use crate::core_arch::{simd::*, x86::*};
use crate::intrinsics::simd::*;

#[cfg(test)]
use stdarch_test::assert_instr;

/// Copies the lower 32 bits of `a` to the lower 32 bits of `dst`, zeroing the upper bits.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovd))]
pub fn _mm_move_epi32(a: __m128i) -> __m128i {
    unsafe {
        let b: u32x4 = simd_shuffle!(a.as_u32x4(), u32x4::ZERO, [0, 4, 4, 4]);
        b.as_m128i()
    }
}

/// Copies the lower 16 bits of `a` to the lower 16 bits of `dst`, zeroing the upper bits.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovw))]
pub fn _mm_move_epi16(a: __m128i) -> __m128i {
    unsafe {
        let b: u16x8 = simd_shuffle!(a.as_u16x8(), u16x8::ZERO, [0, 8, 8, 8, 8, 8, 8, 8]);
        b.as_m128i()
    }
}

/// Computes the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in `a`
/// compared to those in `b`, and stores the 16-bit results in dst using writemask `k` (elements are
/// copied from src when the corresponding mask bit is not set). Eight SADs are performed using one
/// quadruplet from `b` and eight quadruplets from `a`. One quadruplet is selected from `b` starting
/// at on the offset specified in `imm8`. Eight quadruplets are formed from sequential 8-bit integers
/// selected from `a` starting at the offset specified in `imm8`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vmpsadbw, IMM8 = 0)
)]
pub fn _mm_mask_mpsadbw_epu8<const IMM8: i32>(
    src: __m128i,
    k: __mmask8,
    a: __m128i,
    b: __m128i,
) -> __m128i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe {
        simd_select_bitmask(k, _mm_mpsadbw_epu8::<IMM8>(a, b).as_u16x8(), src.as_u16x8()).as_m128i()
    }
}

/// Computes the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in `a`
/// compared to those in `b`, and stores the 16-bit results in dst using zeromask `k` (elements are
/// zeroed out when the corresponding mask bit is not set). Eight SADs are performed using one
/// quadruplet from `b` and eight quadruplets from `a`. One quadruplet is selected from `b` starting
/// at on the offset specified in `imm8`. Eight quadruplets are formed from sequential 8-bit integers
/// selected from `a` starting at the offset specified in `imm8`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vmpsadbw, IMM8 = 0)
)]
pub fn _mm_maskz_mpsadbw_epu8<const IMM8: i32>(k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe {
        simd_select_bitmask(k, _mm_mpsadbw_epu8::<IMM8>(a, b).as_u16x8(), u16x8::ZERO).as_m128i()
    }
}

/// Computes the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in `a`
/// compared to those in `b`, and stores the 16-bit results in dst using writemask `k` (elements are
/// copied from src when the corresponding mask bit is not set). Eight SADs are performed for each
/// 128-bit lane using one quadruplet from `b` and eight quadruplets from `a`. One quadruplet is
/// selected from `b` starting at on the offset specified in `imm8`. Eight quadruplets are formed from
/// sequential 8-bit integers selected from `a` starting at the offset specified in `imm8`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vmpsadbw, IMM8 = 0)
)]
pub fn _mm256_mask_mpsadbw_epu8<const IMM8: i32>(
    src: __m256i,
    k: __mmask16,
    a: __m256i,
    b: __m256i,
) -> __m256i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe {
        simd_select_bitmask(
            k,
            _mm256_mpsadbw_epu8::<IMM8>(a, b).as_u16x16(),
            src.as_u16x16(),
        )
        .as_m256i()
    }
}

/// Computes the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in `a`
/// compared to those in `b`, and stores the 16-bit results in dst using zeromask `k` (elements are
/// zeroed out when the corresponding mask bit is not set). Eight SADs are performed for each 128-bit
/// lane using one quadruplet from `b` and eight quadruplets from `a`. One quadruplet is selected from
/// `b` starting at on the offset specified in `imm8`. Eight quadruplets are formed from sequential
/// 8-bit integers selected from `a` starting at the offset specified in `imm8`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vmpsadbw, IMM8 = 0)
)]
pub fn _mm256_maskz_mpsadbw_epu8<const IMM8: i32>(k: __mmask16, a: __m256i, b: __m256i) -> __m256i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe {
        simd_select_bitmask(
            k,
            _mm256_mpsadbw_epu8::<IMM8>(a, b).as_u16x16(),
            u16x16::ZERO,
        )
        .as_m256i()
    }
}

/// Computes the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in `a`
/// compared to those in `b`, and stores the 16-bit results in dst. Eight SADs are performed for each
/// 128-bit lane using one quadruplet from `b` and eight quadruplets from `a`. One quadruplet is
/// selected from `b` starting at on the offset specified in `imm8`. Eight quadruplets are formed from
/// sequential 8-bit integers selected from `a` starting at the offset specified in `imm8`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vmpsadbw, IMM8 = 0)
)]
pub fn _mm512_mpsadbw_epu8<const IMM8: i32>(a: __m512i, b: __m512i) -> __m512i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { vmpsadbw512(a.as_u8x64(), b.as_u8x64(), IMM8 as i8).as_m512i() }
}

/// Computes the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in `a`
/// compared to those in `b`, and stores the 16-bit results in dst using writemask `k` (elements are
/// copied from src when the corresponding mask bit is not set). Eight SADs are performed for each
/// 128-bit lane using one quadruplet from `b` and eight quadruplets from `a`. One quadruplet is
/// selected from `b` starting at on the offset specified in `imm8`. Eight quadruplets are formed from
/// sequential 8-bit integers selected from `a` starting at the offset specified in `imm8`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vmpsadbw, IMM8 = 0)
)]
pub fn _mm512_mask_mpsadbw_epu8<const IMM8: i32>(
    src: __m512i,
    k: __mmask32,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe {
        simd_select_bitmask(
            k,
            _mm512_mpsadbw_epu8::<IMM8>(a, b).as_u16x32(),
            src.as_u16x32(),
        )
        .as_m512i()
    }
}

/// Computes the sum of absolute differences (SADs) of quadruplets of unsigned 8-bit integers in `a`
/// compared to those in `b`, and stores the 16-bit results in dst using zeromask `k` (elements are
/// zeroed out when the corresponding mask bit is not set). Eight SADs are performed for each 128-bit
/// lane using one quadruplet from `b` and eight quadruplets from `a`. One quadruplet is selected from
/// `b` starting at on the offset specified in `imm8`. Eight quadruplets are formed from sequential
/// 8-bit integers selected from `a` starting at the offset specified in `imm8`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vmpsadbw, IMM8 = 0)
)]
pub fn _mm512_maskz_mpsadbw_epu8<const IMM8: i32>(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe {
        simd_select_bitmask(
            k,
            _mm512_mpsadbw_epu8::<IMM8>(a, b).as_u16x32(),
            u16x32::ZERO,
        )
        .as_m512i()
    }
}

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.x86.avx10.vmpsadbw.512"]
    fn vmpsadbw512(a: u8x64, b: u8x64, imm8: i8) -> u16x32;

}

#[cfg(test)]
mod tests {
    use crate::core_arch::x86::*;
    use stdarch_test::simd_test;

    #[simd_test(enable = "avx10.2")]
    fn test_mm_move_epi32() {
        let a = _mm_set_epi32(0x12345678, 0x7ABCDEF0, 0x11111111, 0x22222222);
        let r = _mm_move_epi32(a);
        let e = _mm_set_epi32(0, 0, 0, 0x22222222);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_move_epi16() {
        let a = _mm_set_epi16(
            0x1234, 0x5678, 0x7ABC, 0x5EF0, 0x1111, 0x2222, 0x3333, 0x4444,
        );
        let r = _mm_move_epi16(a);
        let e = _mm_set_epi16(0, 0, 0, 0, 0, 0, 0, 0x4444);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_mpsadbw_epu8() {
        let src = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(5);
        let k = 0b01010101;
        let r = _mm_mask_mpsadbw_epu8::<0>(src, k, a, b);
        // Each SAD result is: abs(10-5) * 4 = 20
        let e = _mm_setr_epi16(20, 2, 20, 4, 20, 6, 20, 8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_mpsadbw_epu8() {
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(5);
        let k = 0b01010101;
        let r = _mm_maskz_mpsadbw_epu8::<0>(k, a, b);
        let e = _mm_setr_epi16(20, 0, 20, 0, 20, 0, 20, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_mpsadbw_epu8() {
        let src = _mm256_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(5);
        let k = 0b0101010101010101;
        let r = _mm256_mask_mpsadbw_epu8::<0>(src, k, a, b);
        let e = _mm256_setr_epi16(20, 2, 20, 4, 20, 6, 20, 8, 20, 10, 20, 12, 20, 14, 20, 16);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_mpsadbw_epu8() {
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(5);
        let k = 0b0101010101010101;
        let r = _mm256_maskz_mpsadbw_epu8::<0>(k, a, b);
        let e = _mm256_setr_epi16(20, 0, 20, 0, 20, 0, 20, 0, 20, 0, 20, 0, 20, 0, 20, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mpsadbw_epu8() {
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(5);
        let r = _mm512_mpsadbw_epu8::<0>(a, b);
        let e = _mm512_set1_epi16(20);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_mpsadbw_epu8() {
        let src = _mm512_set_epi16(
            32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11,
            10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
        );
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(5);
        let k = 0b01010101010101010101010101010101;
        let r = _mm512_mask_mpsadbw_epu8::<0>(src, k, a, b);
        let e = _mm512_set_epi16(
            32, 20, 30, 20, 28, 20, 26, 20, 24, 20, 22, 20, 20, 20, 18, 20, 16, 20, 14, 20, 12, 20,
            10, 20, 8, 20, 6, 20, 4, 20, 2, 20,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_mpsadbw_epu8() {
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(5);
        let k = 0b01010101010101010101010101010101;
        let r = _mm512_maskz_mpsadbw_epu8::<0>(k, a, b);
        let e = _mm512_set_epi16(
            0, 20, 0, 20, 0, 20, 0, 20, 0, 20, 0, 20, 0, 20, 0, 20, 0, 20, 0, 20, 0, 20, 0, 20, 0,
            20, 0, 20, 0, 20, 0, 20,
        );
        assert_eq_m512i(r, e);
    }
}
