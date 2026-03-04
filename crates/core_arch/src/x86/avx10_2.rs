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

/// Multiply groups of 2 adjacent pairs of half-precision (16-bit) floating-point numbers in a with
/// corresponding half-precision (16-bit) floating-point numbers in b, producing 2 intermediate
/// single-precision (32-bit) floating-point results. Sum these 2 results with the corresponding
/// single-precision (32-bit) floating-point number in src, and store the packed 32-bit results in dst.
///
/// This neither consults nor updates MXCSR.RC, rather the rounding semantics are fixed to:
///
///  - Round to nearest, with ties to even
///  - The two multiplications and two additions are fused into two FMA operations
///  - Input denormals are treated as zero, and output denormals are flushed to zero
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vdpphps))]
pub fn _mm_dpph_ps(src: __m128, a: __m128h, b: __m128h) -> __m128 {
    unsafe { vdpphps128(src.as_f32x4(), a.as_f16x8(), b.as_f16x8()).as_m128() }
}

/// Multiply groups of 2 adjacent pairs of half-precision (16-bit) floating-point numbers in a with
/// corresponding half-precision (16-bit) floating-point numbers in b, producing 2 intermediate
/// single-precision (32-bit) floating-point results. Sum these 2 results with the corresponding
/// single-precision (32-bit) floating-point number in src, and store the packed 32-bit results in dst
/// using writemask `k` (elements are copied from src when the corresponding mask bit is not set).
///
/// This neither consults nor updates MXCSR.RC, rather the rounding semantics are fixed to:
///
///  - Round to nearest, with ties to even
///  - The two multiplications and two additions are fused into two FMA operations
///  - Input denormals are treated as zero, and output denormals are flushed to zero
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vdpphps))]
pub fn _mm_mask_dpph_ps(src: __m128, k: __mmask8, a: __m128h, b: __m128h) -> __m128 {
    unsafe { simd_select_bitmask(k, _mm_dpph_ps(src, a, b), src) }
}

/// Multiply groups of 2 adjacent pairs of half-precision (16-bit) floating-point numbers in a with
/// corresponding half-precision (16-bit) floating-point numbers in b, producing 2 intermediate
/// single-precision (32-bit) floating-point results. Sum these 2 results with the corresponding
/// single-precision (32-bit) floating-point number in src, and store the packed 32-bit results in dst
/// using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
///
/// This neither consults nor updates MXCSR.RC, rather the rounding semantics are fixed to:
///
///  - Round to nearest, with ties to even
///  - The two multiplications and two additions are fused into two FMA operations
///  - Input denormals are treated as zero, and output denormals are flushed to zero
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vdpphps))]
pub fn _mm_maskz_dpph_ps(k: __mmask8, src: __m128, a: __m128h, b: __m128h) -> __m128 {
    unsafe { simd_select_bitmask(k, _mm_dpph_ps(src, a, b), _mm_setzero_ps()) }
}

/// Multiply groups of 2 adjacent pairs of half-precision (16-bit) floating-point numbers in a with
/// corresponding half-precision (16-bit) floating-point numbers in b, producing 2 intermediate
/// single-precision (32-bit) floating-point results. Sum these 2 results with the corresponding
/// single-precision (32-bit) floating-point number in src, and store the packed 32-bit results in dst.
///
/// This neither consults nor updates MXCSR.RC, rather the rounding semantics are fixed to:
///
///  - Round to nearest, with ties to even
///  - The two multiplications and two additions are fused into two FMA operations
///  - Input denormals are treated as zero, and output denormals are flushed to zero
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vdpphps))]
pub fn _mm256_dpph_ps(src: __m256, a: __m256h, b: __m256h) -> __m256 {
    unsafe { vdpphps256(src.as_f32x8(), a.as_f16x16(), b.as_f16x16()).as_m256() }
}

/// Multiply groups of 2 adjacent pairs of half-precision (16-bit) floating-point numbers in a with
/// corresponding half-precision (16-bit) floating-point numbers in b, producing 2 intermediate
/// single-precision (32-bit) floating-point results. Sum these 2 results with the corresponding
/// single-precision (32-bit) floating-point number in src, and store the packed 32-bit results in dst
/// using writemask `k` (elements are copied from src when the corresponding mask bit is not set).
///
/// This neither consults nor updates MXCSR.RC, rather the rounding semantics are fixed to:
///
///  - Round to nearest, with ties to even
///  - The two multiplications and two additions are fused into two FMA operations
///  - Input denormals are treated as zero, and output denormals are flushed to zero
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vdpphps))]
pub fn _mm256_mask_dpph_ps(src: __m256, k: __mmask8, a: __m256h, b: __m256h) -> __m256 {
    unsafe { simd_select_bitmask(k, _mm256_dpph_ps(src, a, b), src) }
}

/// Multiply groups of 2 adjacent pairs of half-precision (16-bit) floating-point numbers in a with
/// corresponding half-precision (16-bit) floating-point numbers in b, producing 2 intermediate
/// single-precision (32-bit) floating-point results. Sum these 2 results with the corresponding
/// single-precision (32-bit) floating-point number in src, and store the packed 32-bit results in dst
/// using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
///
/// This neither consults nor updates MXCSR.RC, rather the rounding semantics are fixed to:
///
///  - Round to nearest, with ties to even
///  - The two multiplications and two additions are fused into two FMA operations
///  - Input denormals are treated as zero, and output denormals are flushed to zero
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vdpphps))]
pub fn _mm256_maskz_dpph_ps(k: __mmask8, src: __m256, a: __m256h, b: __m256h) -> __m256 {
    unsafe { simd_select_bitmask(k, _mm256_dpph_ps(src, a, b), _mm256_setzero_ps()) }
}

/// Multiply groups of 2 adjacent pairs of half-precision (16-bit) floating-point numbers in a with
/// corresponding half-precision (16-bit) floating-point numbers in b, producing 2 intermediate
/// single-precision (32-bit) floating-point results. Sum these 2 results with the corresponding
/// single-precision (32-bit) floating-point number in src, and store the packed 32-bit results in dst.
///
/// This neither consults nor updates MXCSR.RC, rather the rounding semantics are fixed to:
///
///  - Round to nearest, with ties to even
///  - The two multiplications and two additions are fused into two FMA operations
///  - Input denormals are treated as zero, and output denormals are flushed to zero
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vdpphps))]
pub fn _mm512_dpph_ps(src: __m512, a: __m512h, b: __m512h) -> __m512 {
    unsafe { vdpphps512(src.as_f32x16(), a.as_f16x32(), b.as_f16x32()).as_m512() }
}

/// Multiply groups of 2 adjacent pairs of half-precision (16-bit) floating-point numbers in a with
/// corresponding half-precision (16-bit) floating-point numbers in b, producing 2 intermediate
/// single-precision (32-bit) floating-point results. Sum these 2 results with the corresponding
/// single-precision (32-bit) floating-point number in src, and store the packed 32-bit results in dst
/// using writemask `k` (elements are copied from src when the corresponding mask bit is not set).
///
/// This neither consults nor updates MXCSR.RC, rather the rounding semantics are fixed to:
///
///  - Round to nearest, with ties to even
///  - The two multiplications and two additions are fused into two FMA operations
///  - Input denormals are treated as zero, and output denormals are flushed to zero
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vdpphps))]
pub fn _mm512_mask_dpph_ps(src: __m512, k: __mmask16, a: __m512h, b: __m512h) -> __m512 {
    unsafe { simd_select_bitmask(k, _mm512_dpph_ps(src, a, b), src) }
}

/// Multiply groups of 2 adjacent pairs of half-precision (16-bit) floating-point numbers in a with
/// corresponding half-precision (16-bit) floating-point numbers in b, producing 2 intermediate
/// single-precision (32-bit) floating-point results. Sum these 2 results with the corresponding
/// single-precision (32-bit) floating-point number in src, and store the packed 32-bit results in dst
/// using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
///
/// This neither consults nor updates MXCSR.RC, rather the rounding semantics are fixed to:
///
///  - Round to nearest, with ties to even
///  - The two multiplications and two additions are fused into two FMA operations
///  - Input denormals are treated as zero, and output denormals are flushed to zero
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vdpphps))]
pub fn _mm512_maskz_dpph_ps(k: __mmask16, src: __m512, a: __m512h, b: __m512h) -> __m512 {
    unsafe { simd_select_bitmask(k, _mm512_dpph_ps(src, a, b), _mm512_setzero_ps()) }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding signed 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using writemask `k`
/// (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbssd))]
pub fn _mm_mask_dpbssd_epi32(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpbssd_epi32(src, a, b).as_i32x4(), src.as_i32x4()).as_m128i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding signed 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using zeromask `k`
/// (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbssd))]
pub fn _mm_maskz_dpbssd_epi32(k: __mmask8, src: __m128i, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpbssd_epi32(src, a, b).as_i32x4(), i32x4::ZERO).as_m128i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding signed 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using writemask `k`
/// (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbssd))]
pub fn _mm256_mask_dpbssd_epi32(src: __m256i, k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(k, _mm256_dpbssd_epi32(src, a, b).as_i32x8(), src.as_i32x8()).as_m256i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding signed 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using zeromask `k`
/// (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbssd))]
pub fn _mm256_maskz_dpbssd_epi32(k: __mmask8, src: __m256i, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(k, _mm256_dpbssd_epi32(src, a, b).as_i32x8(), i32x8::ZERO).as_m256i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding signed 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbssd))]
pub fn _mm512_dpbssd_epi32(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe { vdpbssd512(src.as_i32x16(), a.as_i8x64(), b.as_i8x64()).as_m512i() }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding signed 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using writemask `k`
/// (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbssd))]
pub fn _mm512_mask_dpbssd_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(
            k,
            _mm512_dpbssd_epi32(src, a, b).as_i32x16(),
            src.as_i32x16(),
        )
        .as_m512i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding signed 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using zeromask `k`
/// (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbssd))]
pub fn _mm512_maskz_dpbssd_epi32(k: __mmask16, src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(k, _mm512_dpbssd_epi32(src, a, b).as_i32x16(), i32x16::ZERO).as_m512i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding signed 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using writemask `k` (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbssds))]
pub fn _mm_mask_dpbssds_epi32(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpbssds_epi32(src, a, b).as_i32x4(), src.as_i32x4()).as_m128i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding signed 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbssds))]
pub fn _mm_maskz_dpbssds_epi32(k: __mmask8, src: __m128i, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpbssds_epi32(src, a, b).as_i32x4(), i32x4::ZERO).as_m128i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding signed 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using writemask `k` (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbssds))]
pub fn _mm256_mask_dpbssds_epi32(src: __m256i, k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(
            k,
            _mm256_dpbssds_epi32(src, a, b).as_i32x8(),
            src.as_i32x8(),
        )
        .as_m256i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding signed 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbssds))]
pub fn _mm256_maskz_dpbssds_epi32(k: __mmask8, src: __m256i, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(k, _mm256_dpbssds_epi32(src, a, b).as_i32x8(), i32x8::ZERO).as_m256i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding signed 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbssds))]
pub fn _mm512_dpbssds_epi32(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe { vdpbssds512(src.as_i32x16(), a.as_i8x64(), b.as_i8x64()).as_m512i() }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding signed 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using writemask `k` (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbssds))]
pub fn _mm512_mask_dpbssds_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(
            k,
            _mm512_dpbssds_epi32(src, a, b).as_i32x16(),
            src.as_i32x16(),
        )
        .as_m512i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding signed 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbssds))]
pub fn _mm512_maskz_dpbssds_epi32(k: __mmask16, src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(k, _mm512_dpbssds_epi32(src, a, b).as_i32x16(), i32x16::ZERO).as_m512i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using writemask `k`
/// (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbsud))]
pub fn _mm_mask_dpbsud_epi32(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpbsud_epi32(src, a, b).as_i32x4(), src.as_i32x4()).as_m128i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using zeromask `k`
/// (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbsud))]
pub fn _mm_maskz_dpbsud_epi32(k: __mmask8, src: __m128i, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpbsud_epi32(src, a, b).as_i32x4(), i32x4::ZERO).as_m128i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using writemask `k`
/// (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbsud))]
pub fn _mm256_mask_dpbsud_epi32(src: __m256i, k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(k, _mm256_dpbsud_epi32(src, a, b).as_i32x8(), src.as_i32x8()).as_m256i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using zeromask `k`
/// (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbsud))]
pub fn _mm256_maskz_dpbsud_epi32(k: __mmask8, src: __m256i, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(k, _mm256_dpbsud_epi32(src, a, b).as_i32x8(), i32x8::ZERO).as_m256i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbsud))]
pub fn _mm512_dpbsud_epi32(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe { vdpbsud512(src.as_i32x16(), a.as_i8x64(), b.as_u8x64()).as_m512i() }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using writemask `k`
/// (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbsud))]
pub fn _mm512_mask_dpbsud_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(
            k,
            _mm512_dpbsud_epi32(src, a, b).as_i32x16(),
            src.as_i32x16(),
        )
        .as_m512i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using zeromask `k`
/// (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbsud))]
pub fn _mm512_maskz_dpbsud_epi32(k: __mmask16, src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(k, _mm512_dpbsud_epi32(src, a, b).as_i32x16(), i32x16::ZERO).as_m512i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using writemask `k` (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbsuds))]
pub fn _mm_mask_dpbsuds_epi32(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpbsuds_epi32(src, a, b).as_i32x4(), src.as_i32x4()).as_m128i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbsuds))]
pub fn _mm_maskz_dpbsuds_epi32(k: __mmask8, src: __m128i, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpbsuds_epi32(src, a, b).as_i32x4(), i32x4::ZERO).as_m128i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using writemask `k` (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbsuds))]
pub fn _mm256_mask_dpbsuds_epi32(src: __m256i, k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(
            k,
            _mm256_dpbsuds_epi32(src, a, b).as_i32x8(),
            src.as_i32x8(),
        )
        .as_m256i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbsuds))]
pub fn _mm256_maskz_dpbsuds_epi32(k: __mmask8, src: __m256i, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(k, _mm256_dpbsuds_epi32(src, a, b).as_i32x8(), i32x8::ZERO).as_m256i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbsuds))]
pub fn _mm512_dpbsuds_epi32(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe { vdpbsuds512(src.as_i32x16(), a.as_i8x64(), b.as_u8x64()).as_m512i() }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using writemask `k` (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbsuds))]
pub fn _mm512_mask_dpbsuds_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(
            k,
            _mm512_dpbsuds_epi32(src, a, b).as_i32x16(),
            src.as_i32x16(),
        )
        .as_m512i()
    }
}

/// Multiply groups of 4 adjacent pairs of signed 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbsuds))]
pub fn _mm512_maskz_dpbsuds_epi32(k: __mmask16, src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(k, _mm512_dpbsuds_epi32(src, a, b).as_i32x16(), i32x16::ZERO).as_m512i()
    }
}

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using writemask `k`
/// (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbuud))]
pub fn _mm_mask_dpbuud_epi32(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpbuud_epi32(src, a, b).as_i32x4(), src.as_i32x4()).as_m128i()
    }
}

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using zeromask `k`
/// (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbuud))]
pub fn _mm_maskz_dpbuud_epi32(k: __mmask8, src: __m128i, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpbuud_epi32(src, a, b).as_i32x4(), i32x4::ZERO).as_m128i()
    }
}

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using writemask `k`
/// (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbuud))]
pub fn _mm256_mask_dpbuud_epi32(src: __m256i, k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(k, _mm256_dpbuud_epi32(src, a, b).as_i32x8(), src.as_i32x8()).as_m256i()
    }
}

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using zeromask `k`
/// (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbuud))]
pub fn _mm256_maskz_dpbuud_epi32(k: __mmask8, src: __m256i, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(k, _mm256_dpbuud_epi32(src, a, b).as_i32x8(), i32x8::ZERO).as_m256i()
    }
}

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbuud))]
pub fn _mm512_dpbuud_epi32(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe { vdpbuud512(src.as_i32x16(), a.as_u8x64(), b.as_u8x64()).as_m512i() }
}

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using writemask `k`
/// (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbuud))]
pub fn _mm512_mask_dpbuud_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(
            k,
            _mm512_dpbuud_epi32(src, a, b).as_i32x16(),
            src.as_i32x16(),
        )
        .as_m512i()
    }
}

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using zeromask `k`
/// (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbuud))]
pub fn _mm512_maskz_dpbuud_epi32(k: __mmask16, src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(k, _mm512_dpbuud_epi32(src, a, b).as_i32x16(), i32x16::ZERO).as_m512i()
    }
}

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using writemask `k` (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbuuds))]
pub fn _mm_mask_dpbuuds_epi32(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpbuuds_epi32(src, a, b).as_i32x4(), src.as_i32x4()).as_m128i()
    }
}

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbuuds))]
pub fn _mm_maskz_dpbuuds_epi32(k: __mmask8, src: __m128i, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpbuuds_epi32(src, a, b).as_i32x4(), i32x4::ZERO).as_m128i()
    }
}

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using writemask `k` (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbuuds))]
pub fn _mm256_mask_dpbuuds_epi32(src: __m256i, k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(
            k,
            _mm256_dpbuuds_epi32(src, a, b).as_i32x8(),
            src.as_i32x8(),
        )
        .as_m256i()
    }
}

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbuuds))]
pub fn _mm256_maskz_dpbuuds_epi32(k: __mmask8, src: __m256i, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(k, _mm256_dpbuuds_epi32(src, a, b).as_i32x8(), i32x8::ZERO).as_m256i()
    }
}

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbuuds))]
pub fn _mm512_dpbuuds_epi32(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe { vdpbuuds512(src.as_i32x16(), a.as_u8x64(), b.as_u8x64()).as_m512i() }
}

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using writemask `k` (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbuuds))]
pub fn _mm512_mask_dpbuuds_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(
            k,
            _mm512_dpbuuds_epi32(src, a, b).as_i32x16(),
            src.as_i32x16(),
        )
        .as_m512i()
    }
}

/// Multiply groups of 4 adjacent pairs of unsigned 8-bit integers in a with corresponding unsigned 8-bit
/// integers in b, producing 4 intermediate signed 16-bit results. Sum these 4 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpbuuds))]
pub fn _mm512_maskz_dpbuuds_epi32(k: __mmask16, src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(k, _mm512_dpbuuds_epi32(src, a, b).as_i32x16(), i32x16::ZERO).as_m512i()
    }
}

/// Multiply groups of 2 adjacent pairs of signed 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using writemask `k`
/// (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwsud))]
pub fn _mm_mask_dpwsud_epi32(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpwsud_epi32(src, a, b).as_i32x4(), src.as_i32x4()).as_m128i()
    }
}

/// Multiply groups of 2 adjacent pairs of signed 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using zeromask `k`
/// (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwsud))]
pub fn _mm_maskz_dpwsud_epi32(k: __mmask8, src: __m128i, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpwsud_epi32(src, a, b).as_i32x4(), i32x4::ZERO).as_m128i()
    }
}

/// Multiply groups of 2 adjacent pairs of signed 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using writemask `k`
/// (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwsud))]
pub fn _mm256_mask_dpwsud_epi32(src: __m256i, k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(k, _mm256_dpwsud_epi32(src, a, b).as_i32x8(), src.as_i32x8()).as_m256i()
    }
}

/// Multiply groups of 2 adjacent pairs of signed 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using zeromask `k`
/// (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwsud))]
pub fn _mm256_maskz_dpwsud_epi32(k: __mmask8, src: __m256i, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(k, _mm256_dpwsud_epi32(src, a, b).as_i32x8(), i32x8::ZERO).as_m256i()
    }
}

/// Multiply groups of 2 adjacent pairs of signed 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwsud))]
pub fn _mm512_dpwsud_epi32(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe { vdpwsud512(src.as_i32x16(), a.as_i16x32(), b.as_u16x32()).as_m512i() }
}

/// Multiply groups of 2 adjacent pairs of signed 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using writemask `k`
/// (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwsud))]
pub fn _mm512_mask_dpwsud_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(
            k,
            _mm512_dpwsud_epi32(src, a, b).as_i32x16(),
            src.as_i32x16(),
        )
        .as_m512i()
    }
}

/// Multiply groups of 2 adjacent pairs of signed 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using zeromask `k`
/// (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwsud))]
pub fn _mm512_maskz_dpwsud_epi32(k: __mmask16, src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(k, _mm512_dpwsud_epi32(src, a, b).as_i32x16(), i32x16::ZERO).as_m512i()
    }
}

/// Multiply groups of 2 adjacent pairs of signed 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using writemask `k` (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwsuds))]
pub fn _mm_mask_dpwsuds_epi32(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpwsuds_epi32(src, a, b).as_i32x4(), src.as_i32x4()).as_m128i()
    }
}

/// Multiply groups of 2 adjacent pairs of signed 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwsuds))]
pub fn _mm_maskz_dpwsuds_epi32(k: __mmask8, src: __m128i, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpwsuds_epi32(src, a, b).as_i32x4(), i32x4::ZERO).as_m128i()
    }
}

/// Multiply groups of 2 adjacent pairs of signed 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using writemask `k` (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwsuds))]
pub fn _mm256_mask_dpwsuds_epi32(src: __m256i, k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(
            k,
            _mm256_dpwsuds_epi32(src, a, b).as_i32x8(),
            src.as_i32x8(),
        )
        .as_m256i()
    }
}

/// Multiply groups of 2 adjacent pairs of signed 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwsuds))]
pub fn _mm256_maskz_dpwsuds_epi32(k: __mmask8, src: __m256i, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(k, _mm256_dpwsuds_epi32(src, a, b).as_i32x8(), i32x8::ZERO).as_m256i()
    }
}

/// Multiply groups of 2 adjacent pairs of signed 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwsuds))]
pub fn _mm512_dpwsuds_epi32(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe { vdpwsuds512(src.as_i32x16(), a.as_i16x32(), b.as_u16x32()).as_m512i() }
}

/// Multiply groups of 2 adjacent pairs of signed 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using writemask `k` (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwsuds))]
pub fn _mm512_mask_dpwsuds_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(
            k,
            _mm512_dpwsuds_epi32(src, a, b).as_i32x16(),
            src.as_i32x16(),
        )
        .as_m512i()
    }
}

/// Multiply groups of 2 adjacent pairs of signed 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwsuds))]
pub fn _mm512_maskz_dpwsuds_epi32(k: __mmask16, src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(k, _mm512_dpwsuds_epi32(src, a, b).as_i32x16(), i32x16::ZERO).as_m512i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding signed 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using writemask `k`
/// (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwusd))]
pub fn _mm_mask_dpwusd_epi32(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpwusd_epi32(src, a, b).as_i32x4(), src.as_i32x4()).as_m128i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding signed 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using zeromask `k`
/// (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwusd))]
pub fn _mm_maskz_dpwusd_epi32(k: __mmask8, src: __m128i, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpwusd_epi32(src, a, b).as_i32x4(), i32x4::ZERO).as_m128i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding signed 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using writemask `k`
/// (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwusd))]
pub fn _mm256_mask_dpwusd_epi32(src: __m256i, k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(k, _mm256_dpwusd_epi32(src, a, b).as_i32x8(), src.as_i32x8()).as_m256i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding signed 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using zeromask `k`
/// (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwusd))]
pub fn _mm256_maskz_dpwusd_epi32(k: __mmask8, src: __m256i, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(k, _mm256_dpwusd_epi32(src, a, b).as_i32x8(), i32x8::ZERO).as_m256i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding signed 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwusd))]
pub fn _mm512_dpwusd_epi32(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe { vdpwusd512(src.as_i32x16(), a.as_u16x32(), b.as_i16x32()).as_m512i() }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding signed 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using writemask `k`
/// (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwusd))]
pub fn _mm512_mask_dpwusd_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(
            k,
            _mm512_dpwusd_epi32(src, a, b).as_i32x16(),
            src.as_i32x16(),
        )
        .as_m512i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding signed 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using zeromask `k`
/// (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwusd))]
pub fn _mm512_maskz_dpwusd_epi32(k: __mmask16, src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(k, _mm512_dpwusd_epi32(src, a, b).as_i32x16(), i32x16::ZERO).as_m512i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding signed 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using writemask `k` (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwusds))]
pub fn _mm_mask_dpwusds_epi32(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpwusds_epi32(src, a, b).as_i32x4(), src.as_i32x4()).as_m128i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding signed 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwusds))]
pub fn _mm_maskz_dpwusds_epi32(k: __mmask8, src: __m128i, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpwusds_epi32(src, a, b).as_i32x4(), i32x4::ZERO).as_m128i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding signed 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using writemask `k` (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwusds))]
pub fn _mm256_mask_dpwusds_epi32(src: __m256i, k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(
            k,
            _mm256_dpwusds_epi32(src, a, b).as_i32x8(),
            src.as_i32x8(),
        )
        .as_m256i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding signed 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwusds))]
pub fn _mm256_maskz_dpwusds_epi32(k: __mmask8, src: __m256i, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(k, _mm256_dpwusds_epi32(src, a, b).as_i32x8(), i32x8::ZERO).as_m256i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding signed 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwusds))]
pub fn _mm512_dpwusds_epi32(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe { vdpwusds512(src.as_i32x16(), a.as_u16x32(), b.as_i16x32()).as_m512i() }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding signed 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using writemask `k` (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwusds))]
pub fn _mm512_mask_dpwusds_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(
            k,
            _mm512_dpwusds_epi32(src, a, b).as_i32x16(),
            src.as_i32x16(),
        )
        .as_m512i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding signed 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwusds))]
pub fn _mm512_maskz_dpwusds_epi32(k: __mmask16, src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(k, _mm512_dpwusds_epi32(src, a, b).as_i32x16(), i32x16::ZERO).as_m512i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using writemask `k`
/// (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwuud))]
pub fn _mm_mask_dpwuud_epi32(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpwuud_epi32(src, a, b).as_i32x4(), src.as_i32x4()).as_m128i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using zeromask `k`
/// (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwuud))]
pub fn _mm_maskz_dpwuud_epi32(k: __mmask8, src: __m128i, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpwuud_epi32(src, a, b).as_i32x4(), i32x4::ZERO).as_m128i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using writemask `k`
/// (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwuud))]
pub fn _mm256_mask_dpwuud_epi32(src: __m256i, k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(k, _mm256_dpwuud_epi32(src, a, b).as_i32x8(), src.as_i32x8()).as_m256i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using zeromask `k`
/// (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwuud))]
pub fn _mm256_maskz_dpwuud_epi32(k: __mmask8, src: __m256i, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(k, _mm256_dpwuud_epi32(src, a, b).as_i32x8(), i32x8::ZERO).as_m256i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwuud))]
pub fn _mm512_dpwuud_epi32(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe { vdpwuud512(src.as_i32x16(), a.as_u16x32(), b.as_u16x32()).as_m512i() }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using writemask `k`
/// (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwuud))]
pub fn _mm512_mask_dpwuud_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(
            k,
            _mm512_dpwuud_epi32(src, a, b).as_i32x16(),
            src.as_i32x16(),
        )
        .as_m512i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src, and store the packed 32-bit results in dst using zeromask `k`
/// (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwuud))]
pub fn _mm512_maskz_dpwuud_epi32(k: __mmask16, src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(k, _mm512_dpwuud_epi32(src, a, b).as_i32x16(), i32x16::ZERO).as_m512i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using writemask `k` (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwuuds))]
pub fn _mm_mask_dpwuuds_epi32(src: __m128i, k: __mmask8, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpwuuds_epi32(src, a, b).as_i32x4(), src.as_i32x4()).as_m128i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwuuds))]
pub fn _mm_maskz_dpwuuds_epi32(k: __mmask8, src: __m128i, a: __m128i, b: __m128i) -> __m128i {
    unsafe {
        simd_select_bitmask(k, _mm_dpwuuds_epi32(src, a, b).as_i32x4(), i32x4::ZERO).as_m128i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using writemask `k` (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwuuds))]
pub fn _mm256_mask_dpwuuds_epi32(src: __m256i, k: __mmask8, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(
            k,
            _mm256_dpwuuds_epi32(src, a, b).as_i32x8(),
            src.as_i32x8(),
        )
        .as_m256i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwuuds))]
pub fn _mm256_maskz_dpwuuds_epi32(k: __mmask8, src: __m256i, a: __m256i, b: __m256i) -> __m256i {
    unsafe {
        simd_select_bitmask(k, _mm256_dpwuuds_epi32(src, a, b).as_i32x8(), i32x8::ZERO).as_m256i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwuuds))]
pub fn _mm512_dpwuuds_epi32(src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe { vdpwuuds512(src.as_i32x16(), a.as_u16x32(), b.as_u16x32()).as_m512i() }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using writemask `k` (elements are copied from src when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwuuds))]
pub fn _mm512_mask_dpwuuds_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(
            k,
            _mm512_dpwuuds_epi32(src, a, b).as_i32x16(),
            src.as_i32x16(),
        )
        .as_m512i()
    }
}

/// Multiply groups of 2 adjacent pairs of unsigned 16-bit integers in a with corresponding unsigned 16-bit
/// integers in b, producing 2 intermediate signed 32-bit results. Sum these 2 results with the
/// corresponding 32-bit integer in src with signed saturation, and store the packed 32-bit results
/// in dst using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set)
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vpdpwuuds))]
pub fn _mm512_maskz_dpwuuds_epi32(k: __mmask16, src: __m512i, a: __m512i, b: __m512i) -> __m512i {
    unsafe {
        simd_select_bitmask(k, _mm512_dpwuuds_epi32(src, a, b).as_i32x16(), i32x16::ZERO).as_m512i()
    }
}

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.x86.avx10.vmpsadbw.512"]
    fn vmpsadbw512(a: u8x64, b: u8x64, imm8: i8) -> u16x32;

    #[link_name = "llvm.x86.avx10.vdpphps.128"]
    fn vdpphps128(src: f32x4, a: f16x8, b: f16x8) -> f32x4;
    #[link_name = "llvm.x86.avx10.vdpphps.256"]
    fn vdpphps256(src: f32x8, a: f16x16, b: f16x16) -> f32x8;
    #[link_name = "llvm.x86.avx10.vdpphps.512"]
    fn vdpphps512(src: f32x16, a: f16x32, b: f16x32) -> f32x16;

    #[link_name = "llvm.x86.avx10.vpdpbssd.512"]
    fn vdpbssd512(src: i32x16, a: i8x64, b: i8x64) -> i32x16;
    #[link_name = "llvm.x86.avx10.vpdpbssds.512"]
    fn vdpbssds512(src: i32x16, a: i8x64, b: i8x64) -> i32x16;
    #[link_name = "llvm.x86.avx10.vpdpbsud.512"]
    fn vdpbsud512(src: i32x16, a: i8x64, b: u8x64) -> i32x16;
    #[link_name = "llvm.x86.avx10.vpdpbsuds.512"]
    fn vdpbsuds512(src: i32x16, a: i8x64, b: u8x64) -> i32x16;
    #[link_name = "llvm.x86.avx10.vpdpbuud.512"]
    fn vdpbuud512(src: i32x16, a: u8x64, b: u8x64) -> i32x16;
    #[link_name = "llvm.x86.avx10.vpdpbuuds.512"]
    fn vdpbuuds512(src: i32x16, a: u8x64, b: u8x64) -> i32x16;

    #[link_name = "llvm.x86.avx10.vpdpwsud.512"]
    fn vdpwsud512(src: i32x16, a: i16x32, b: u16x32) -> i32x16;
    #[link_name = "llvm.x86.avx10.vpdpwsuds.512"]
    fn vdpwsuds512(src: i32x16, a: i16x32, b: u16x32) -> i32x16;
    #[link_name = "llvm.x86.avx10.vpdpwusd.512"]
    fn vdpwusd512(src: i32x16, a: u16x32, b: i16x32) -> i32x16;
    #[link_name = "llvm.x86.avx10.vpdpwusds.512"]
    fn vdpwusds512(src: i32x16, a: u16x32, b: i16x32) -> i32x16;
    #[link_name = "llvm.x86.avx10.vpdpwuud.512"]
    fn vdpwuud512(src: i32x16, a: u16x32, b: u16x32) -> i32x16;
    #[link_name = "llvm.x86.avx10.vpdpwuuds.512"]
    fn vdpwuuds512(src: i32x16, a: u16x32, b: u16x32) -> i32x16;
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

    #[simd_test(enable = "avx10.2")]
    fn test_mm_dpph_ps() {
        let src = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let a = _mm_set1_ph(10.0);
        let b = _mm_set1_ph(20.0);
        let r = _mm_dpph_ps(src, a, b);
        // Each result is: src[i] + (a[2*i] * b[2*i] + a[2*i+1] * b[2*i+1])
        let e = _mm_setr_ps(401.0, 402.0, 403.0, 404.0);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_dpph_ps() {
        let src = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let a = _mm_set1_ph(10.0);
        let b = _mm_set1_ph(20.0);
        let k = 0b0101;
        let r = _mm_mask_dpph_ps(src, k, a, b);
        let e = _mm_setr_ps(401.0, 2.0, 403.0, 4.0);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_dpph_ps() {
        let src = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let a = _mm_set1_ph(10.0);
        let b = _mm_set1_ph(20.0);
        let k = 0b0101;
        let r = _mm_maskz_dpph_ps(k, src, a, b);
        let e = _mm_setr_ps(401.0, 0.0, 403.0, 0.0);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_dpph_ps() {
        let src = _mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let a = _mm256_set1_ph(10.0);
        let b = _mm256_set1_ph(20.0);
        let r = _mm256_dpph_ps(src, a, b);
        let e = _mm256_setr_ps(401.0, 402.0, 403.0, 404.0, 405.0, 406.0, 407.0, 408.0);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_dpph_ps() {
        let src = _mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let a = _mm256_set1_ph(10.0);
        let b = _mm256_set1_ph(20.0);
        let k = 0b01010101;
        let r = _mm256_mask_dpph_ps(src, k, a, b);
        let e = _mm256_setr_ps(401.0, 2.0, 403.0, 4.0, 405.0, 6.0, 407.0, 8.0);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_dpph_ps() {
        let src = _mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let a = _mm256_set1_ph(10.0);
        let b = _mm256_set1_ph(20.0);
        let k = 0b01010101;
        let r = _mm256_maskz_dpph_ps(k, src, a, b);
        let e = _mm256_setr_ps(401.0, 0.0, 403.0, 0.0, 405.0, 0.0, 407.0, 0.0);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_dpph_ps() {
        let src = _mm512_setr_ps(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let a = _mm512_set1_ph(10.0);
        let b = _mm512_set1_ph(20.0);
        let r = _mm512_dpph_ps(src, a, b);
        let e = _mm512_setr_ps(
            401.0, 402.0, 403.0, 404.0, 405.0, 406.0, 407.0, 408.0, 409.0, 410.0, 411.0, 412.0,
            413.0, 414.0, 415.0, 416.0,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_dpph_ps() {
        let src = _mm512_setr_ps(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let a = _mm512_set1_ph(10.0);
        let b = _mm512_set1_ph(20.0);
        let k = 0b0101010101010101;
        let r = _mm512_mask_dpph_ps(src, k, a, b);
        let e = _mm512_setr_ps(
            401.0, 2.0, 403.0, 4.0, 405.0, 6.0, 407.0, 8.0, 409.0, 10.0, 411.0, 12.0, 413.0, 14.0,
            415.0, 16.0,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_dpph_ps() {
        let src = _mm512_setr_ps(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let a = _mm512_set1_ph(10.0);
        let b = _mm512_set1_ph(20.0);
        let k = 0b0101010101010101;
        let r = _mm512_maskz_dpph_ps(k, src, a, b);
        let e = _mm512_setr_ps(
            401.0, 0.0, 403.0, 0.0, 405.0, 0.0, 407.0, 0.0, 409.0, 0.0, 411.0, 0.0, 413.0, 0.0,
            415.0, 0.0,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_dpbssd_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_mask_dpbssd_epi32(src, k, a, b);
        let e = _mm_setr_epi32(801, 2, 803, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_dpbssd_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_maskz_dpbssd_epi32(k, src, a, b);
        let e = _mm_setr_epi32(801, 0, 803, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_dpbssd_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpbssd_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(801, 2, 803, 4, 805, 6, 807, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_dpbssd_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(20);
        let k = 0b01010101;
        let r = _mm256_maskz_dpbssd_epi32(k, src, a, b);
        let e = _mm256_setr_epi32(801, 0, 803, 0, 805, 0, 807, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_dpbssd_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(20);
        let r = _mm512_dpbssd_epi32(src, a, b);
        let e = _mm512_setr_epi32(
            801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_dpbssd_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(20);
        let k = 0b0101010101010101;
        let r = _mm512_mask_dpbssd_epi32(src, k, a, b);
        let e = _mm512_setr_epi32(
            801, 2, 803, 4, 805, 6, 807, 8, 809, 10, 811, 12, 813, 14, 815, 16,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_dpbssd_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(20);
        let k = 0b0101010101010101;
        let r = _mm512_maskz_dpbssd_epi32(k, src, a, b);
        let e = _mm512_setr_epi32(
            801, 0, 803, 0, 805, 0, 807, 0, 809, 0, 811, 0, 813, 0, 815, 0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_dpbssds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_mask_dpbssds_epi32(src, k, a, b);
        let e = _mm_setr_epi32(801, 2, 803, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_dpbssds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_maskz_dpbssds_epi32(k, src, a, b);
        let e = _mm_setr_epi32(801, 0, 803, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_dpbssds_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpbssds_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(801, 2, 803, 4, 805, 6, 807, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_dpbssds_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(20);
        let k = 0b01010101;
        let r = _mm256_maskz_dpbssds_epi32(k, src, a, b);
        let e = _mm256_setr_epi32(801, 0, 803, 0, 805, 0, 807, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_dpbssds_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(20);
        let r = _mm512_dpbssds_epi32(src, a, b);
        let e = _mm512_setr_epi32(
            801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_dpbssds_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(20);
        let k = 0b0101010101010101;
        let r = _mm512_mask_dpbssds_epi32(src, k, a, b);
        let e = _mm512_setr_epi32(
            801, 2, 803, 4, 805, 6, 807, 8, 809, 10, 811, 12, 813, 14, 815, 16,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_dpbssds_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(20);
        let k = 0b0101010101010101;
        let r = _mm512_maskz_dpbssds_epi32(k, src, a, b);
        let e = _mm512_setr_epi32(
            801, 0, 803, 0, 805, 0, 807, 0, 809, 0, 811, 0, 813, 0, 815, 0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_dpbsud_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_mask_dpbsud_epi32(src, k, a, b);
        let e = _mm_setr_epi32(801, 2, 803, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_dpbsud_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_maskz_dpbsud_epi32(k, src, a, b);
        let e = _mm_setr_epi32(801, 0, 803, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_dpbsud_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpbsud_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(801, 2, 803, 4, 805, 6, 807, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_dpbsud_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(20);
        let k = 0b01010101;
        let r = _mm256_maskz_dpbsud_epi32(k, src, a, b);
        let e = _mm256_setr_epi32(801, 0, 803, 0, 805, 0, 807, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_dpbsud_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(20);
        let r = _mm512_dpbsud_epi32(src, a, b);
        let e = _mm512_setr_epi32(
            801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_dpbsud_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(20);
        let k = 0b0101010101010101;
        let r = _mm512_mask_dpbsud_epi32(src, k, a, b);
        let e = _mm512_setr_epi32(
            801, 2, 803, 4, 805, 6, 807, 8, 809, 10, 811, 12, 813, 14, 815, 16,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_dpbsud_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(20);
        let k = 0b0101010101010101;
        let r = _mm512_maskz_dpbsud_epi32(k, src, a, b);
        let e = _mm512_setr_epi32(
            801, 0, 803, 0, 805, 0, 807, 0, 809, 0, 811, 0, 813, 0, 815, 0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_dpbsuds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_mask_dpbsuds_epi32(src, k, a, b);
        let e = _mm_setr_epi32(801, 2, 803, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_dpbsuds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_maskz_dpbsuds_epi32(k, src, a, b);
        let e = _mm_setr_epi32(801, 0, 803, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_dpbsuds_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpbsuds_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(801, 2, 803, 4, 805, 6, 807, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_dpbsuds_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(20);
        let k = 0b01010101;
        let r = _mm256_maskz_dpbsuds_epi32(k, src, a, b);
        let e = _mm256_setr_epi32(801, 0, 803, 0, 805, 0, 807, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_dpbsuds_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(20);
        let r = _mm512_dpbsuds_epi32(src, a, b);
        let e = _mm512_setr_epi32(
            801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_dpbsuds_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(20);
        let k = 0b0101010101010101;
        let r = _mm512_mask_dpbsuds_epi32(src, k, a, b);
        let e = _mm512_setr_epi32(
            801, 2, 803, 4, 805, 6, 807, 8, 809, 10, 811, 12, 813, 14, 815, 16,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_dpbsuds_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(20);
        let k = 0b0101010101010101;
        let r = _mm512_maskz_dpbsuds_epi32(k, src, a, b);
        let e = _mm512_setr_epi32(
            801, 0, 803, 0, 805, 0, 807, 0, 809, 0, 811, 0, 813, 0, 815, 0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_dpbuud_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_mask_dpbuud_epi32(src, k, a, b);
        let e = _mm_setr_epi32(801, 2, 803, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_dpbuud_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_maskz_dpbuud_epi32(k, src, a, b);
        let e = _mm_setr_epi32(801, 0, 803, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_dpbuud_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpbuud_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(801, 2, 803, 4, 805, 6, 807, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_dpbuud_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(20);
        let k = 0b01010101;
        let r = _mm256_maskz_dpbuud_epi32(k, src, a, b);
        let e = _mm256_setr_epi32(801, 0, 803, 0, 805, 0, 807, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_dpbuud_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(20);
        let r = _mm512_dpbuud_epi32(src, a, b);
        let e = _mm512_setr_epi32(
            801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_dpbuud_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(20);
        let k = 0b0101010101010101;
        let r = _mm512_mask_dpbuud_epi32(src, k, a, b);
        let e = _mm512_setr_epi32(
            801, 2, 803, 4, 805, 6, 807, 8, 809, 10, 811, 12, 813, 14, 815, 16,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_dpbuud_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(20);
        let k = 0b0101010101010101;
        let r = _mm512_maskz_dpbuud_epi32(k, src, a, b);
        let e = _mm512_setr_epi32(
            801, 0, 803, 0, 805, 0, 807, 0, 809, 0, 811, 0, 813, 0, 815, 0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_dpbuuds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_mask_dpbuuds_epi32(src, k, a, b);
        let e = _mm_setr_epi32(801, 2, 803, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_dpbuuds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_maskz_dpbuuds_epi32(k, src, a, b);
        let e = _mm_setr_epi32(801, 0, 803, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_dpbuuds_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpbuuds_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(801, 2, 803, 4, 805, 6, 807, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_dpbuuds_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(20);
        let k = 0b01010101;
        let r = _mm256_maskz_dpbuuds_epi32(k, src, a, b);
        let e = _mm256_setr_epi32(801, 0, 803, 0, 805, 0, 807, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_dpbuuds_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(20);
        let r = _mm512_dpbuuds_epi32(src, a, b);
        let e = _mm512_setr_epi32(
            801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_dpbuuds_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(20);
        let k = 0b0101010101010101;
        let r = _mm512_mask_dpbuuds_epi32(src, k, a, b);
        let e = _mm512_setr_epi32(
            801, 2, 803, 4, 805, 6, 807, 8, 809, 10, 811, 12, 813, 14, 815, 16,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_dpbuuds_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi8(10);
        let b = _mm512_set1_epi8(20);
        let k = 0b0101010101010101;
        let r = _mm512_maskz_dpbuuds_epi32(k, src, a, b);
        let e = _mm512_setr_epi32(
            801, 0, 803, 0, 805, 0, 807, 0, 809, 0, 811, 0, 813, 0, 815, 0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_dpwsud_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_mask_dpwsud_epi32(src, k, a, b);
        let e = _mm_setr_epi32(401, 2, 403, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_dpwsud_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_maskz_dpwsud_epi32(k, src, a, b);
        let e = _mm_setr_epi32(401, 0, 403, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_dpwsud_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi16(10);
        let b = _mm256_set1_epi16(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpwsud_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(401, 2, 403, 4, 405, 6, 407, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_dpwsud_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi16(10);
        let b = _mm256_set1_epi16(20);
        let k = 0b01010101;
        let r = _mm256_maskz_dpwsud_epi32(k, src, a, b);
        let e = _mm256_setr_epi32(401, 0, 403, 0, 405, 0, 407, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_dpwsud_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi16(10);
        let b = _mm512_set1_epi16(20);
        let r = _mm512_dpwsud_epi32(src, a, b);
        let e = _mm512_setr_epi32(
            401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_dpwsud_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi16(10);
        let b = _mm512_set1_epi16(20);
        let k = 0b0101010101010101;
        let r = _mm512_mask_dpwsud_epi32(src, k, a, b);
        let e = _mm512_setr_epi32(
            401, 2, 403, 4, 405, 6, 407, 8, 409, 10, 411, 12, 413, 14, 415, 16,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_dpwsud_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi16(10);
        let b = _mm512_set1_epi16(20);
        let k = 0b0101010101010101;
        let r = _mm512_maskz_dpwsud_epi32(k, src, a, b);
        let e = _mm512_setr_epi32(
            401, 0, 403, 0, 405, 0, 407, 0, 409, 0, 411, 0, 413, 0, 415, 0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_dpwsuds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_mask_dpwsuds_epi32(src, k, a, b);
        let e = _mm_setr_epi32(401, 2, 403, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_dpwsuds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_maskz_dpwsuds_epi32(k, src, a, b);
        let e = _mm_setr_epi32(401, 0, 403, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_dpwsuds_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi16(10);
        let b = _mm256_set1_epi16(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpwsuds_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(401, 2, 403, 4, 405, 6, 407, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_dpwsuds_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi16(10);
        let b = _mm256_set1_epi16(20);
        let k = 0b01010101;
        let r = _mm256_maskz_dpwsuds_epi32(k, src, a, b);
        let e = _mm256_setr_epi32(401, 0, 403, 0, 405, 0, 407, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_dpwsuds_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi16(10);
        let b = _mm512_set1_epi16(20);
        let r = _mm512_dpwsuds_epi32(src, a, b);
        let e = _mm512_setr_epi32(
            401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_dpwsuds_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi16(10);
        let b = _mm512_set1_epi16(20);
        let k = 0b0101010101010101;
        let r = _mm512_mask_dpwsuds_epi32(src, k, a, b);
        let e = _mm512_setr_epi32(
            401, 2, 403, 4, 405, 6, 407, 8, 409, 10, 411, 12, 413, 14, 415, 16,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_dpwsuds_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi16(10);
        let b = _mm512_set1_epi16(20);
        let k = 0b0101010101010101;
        let r = _mm512_maskz_dpwsuds_epi32(k, src, a, b);
        let e = _mm512_setr_epi32(
            401, 0, 403, 0, 405, 0, 407, 0, 409, 0, 411, 0, 413, 0, 415, 0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_dpwusd_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_mask_dpwusd_epi32(src, k, a, b);
        let e = _mm_setr_epi32(401, 2, 403, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_dpwusd_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_maskz_dpwusd_epi32(k, src, a, b);
        let e = _mm_setr_epi32(401, 0, 403, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_dpwusd_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi16(10);
        let b = _mm256_set1_epi16(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpwusd_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(401, 2, 403, 4, 405, 6, 407, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_dpwusd_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi16(10);
        let b = _mm256_set1_epi16(20);
        let k = 0b01010101;
        let r = _mm256_maskz_dpwusd_epi32(k, src, a, b);
        let e = _mm256_setr_epi32(401, 0, 403, 0, 405, 0, 407, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_dpwusd_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi16(10);
        let b = _mm512_set1_epi16(20);
        let r = _mm512_dpwusd_epi32(src, a, b);
        let e = _mm512_setr_epi32(
            401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_dpwusd_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi16(10);
        let b = _mm512_set1_epi16(20);
        let k = 0b0101010101010101;
        let r = _mm512_mask_dpwusd_epi32(src, k, a, b);
        let e = _mm512_setr_epi32(
            401, 2, 403, 4, 405, 6, 407, 8, 409, 10, 411, 12, 413, 14, 415, 16,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_dpwusd_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi16(10);
        let b = _mm512_set1_epi16(20);
        let k = 0b0101010101010101;
        let r = _mm512_maskz_dpwusd_epi32(k, src, a, b);
        let e = _mm512_setr_epi32(
            401, 0, 403, 0, 405, 0, 407, 0, 409, 0, 411, 0, 413, 0, 415, 0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_dpwusds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_mask_dpwusds_epi32(src, k, a, b);
        let e = _mm_setr_epi32(401, 2, 403, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_dpwusds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_maskz_dpwusds_epi32(k, src, a, b);
        let e = _mm_setr_epi32(401, 0, 403, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_dpwusds_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi16(10);
        let b = _mm256_set1_epi16(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpwusds_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(401, 2, 403, 4, 405, 6, 407, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_dpwusds_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi16(10);
        let b = _mm256_set1_epi16(20);
        let k = 0b01010101;
        let r = _mm256_maskz_dpwusds_epi32(k, src, a, b);
        let e = _mm256_setr_epi32(401, 0, 403, 0, 405, 0, 407, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_dpwusds_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi16(10);
        let b = _mm512_set1_epi16(20);
        let r = _mm512_dpwusds_epi32(src, a, b);
        let e = _mm512_setr_epi32(
            401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_dpwusds_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi16(10);
        let b = _mm512_set1_epi16(20);
        let k = 0b0101010101010101;
        let r = _mm512_mask_dpwusds_epi32(src, k, a, b);
        let e = _mm512_setr_epi32(
            401, 2, 403, 4, 405, 6, 407, 8, 409, 10, 411, 12, 413, 14, 415, 16,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_dpwusds_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi16(10);
        let b = _mm512_set1_epi16(20);
        let k = 0b0101010101010101;
        let r = _mm512_maskz_dpwusds_epi32(k, src, a, b);
        let e = _mm512_setr_epi32(
            401, 0, 403, 0, 405, 0, 407, 0, 409, 0, 411, 0, 413, 0, 415, 0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_dpwuud_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_mask_dpwuud_epi32(src, k, a, b);
        let e = _mm_setr_epi32(401, 2, 403, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_dpwuud_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_maskz_dpwuud_epi32(k, src, a, b);
        let e = _mm_setr_epi32(401, 0, 403, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_dpwuud_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi16(10);
        let b = _mm256_set1_epi16(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpwuud_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(401, 2, 403, 4, 405, 6, 407, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_dpwuud_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi16(10);
        let b = _mm256_set1_epi16(20);
        let k = 0b01010101;
        let r = _mm256_maskz_dpwuud_epi32(k, src, a, b);
        let e = _mm256_setr_epi32(401, 0, 403, 0, 405, 0, 407, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_dpwuud_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi16(10);
        let b = _mm512_set1_epi16(20);
        let r = _mm512_dpwuud_epi32(src, a, b);
        let e = _mm512_setr_epi32(
            401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_dpwuud_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi16(10);
        let b = _mm512_set1_epi16(20);
        let k = 0b0101010101010101;
        let r = _mm512_mask_dpwuud_epi32(src, k, a, b);
        let e = _mm512_setr_epi32(
            401, 2, 403, 4, 405, 6, 407, 8, 409, 10, 411, 12, 413, 14, 415, 16,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_dpwuud_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi16(10);
        let b = _mm512_set1_epi16(20);
        let k = 0b0101010101010101;
        let r = _mm512_maskz_dpwuud_epi32(k, src, a, b);
        let e = _mm512_setr_epi32(
            401, 0, 403, 0, 405, 0, 407, 0, 409, 0, 411, 0, 413, 0, 415, 0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_dpwuuds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_mask_dpwuuds_epi32(src, k, a, b);
        let e = _mm_setr_epi32(401, 2, 403, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_dpwuuds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_maskz_dpwuuds_epi32(k, src, a, b);
        let e = _mm_setr_epi32(401, 0, 403, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_dpwuuds_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi16(10);
        let b = _mm256_set1_epi16(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpwuuds_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(401, 2, 403, 4, 405, 6, 407, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_dpwuuds_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi16(10);
        let b = _mm256_set1_epi16(20);
        let k = 0b01010101;
        let r = _mm256_maskz_dpwuuds_epi32(k, src, a, b);
        let e = _mm256_setr_epi32(401, 0, 403, 0, 405, 0, 407, 0);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_dpwuuds_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi16(10);
        let b = _mm512_set1_epi16(20);
        let r = _mm512_dpwuuds_epi32(src, a, b);
        let e = _mm512_setr_epi32(
            401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_dpwuuds_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi16(10);
        let b = _mm512_set1_epi16(20);
        let k = 0b0101010101010101;
        let r = _mm512_mask_dpwuuds_epi32(src, k, a, b);
        let e = _mm512_setr_epi32(
            401, 2, 403, 4, 405, 6, 407, 8, 409, 10, 411, 12, 413, 14, 415, 16,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_dpwuuds_epi32() {
        let src = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let a = _mm512_set1_epi16(10);
        let b = _mm512_set1_epi16(20);
        let k = 0b0101010101010101;
        let r = _mm512_maskz_dpwuuds_epi32(k, src, a, b);
        let e = _mm512_setr_epi32(
            401, 0, 403, 0, 405, 0, 407, 0, 409, 0, 411, 0, 413, 0, 415, 0,
        );
        assert_eq_m512i(r, e);
    }
}
