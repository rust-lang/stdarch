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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint8")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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
#[target_feature(enable = "avx10.2,avxvnniint16")]
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

/// Convert the lower double-precision (64-bit) floating-point element in `a` to a 32-bit
/// integer with truncation and saturation.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttsd2sis, SAE = 8)
)]
pub fn _mm_cvtts_roundsd_i32<const SAE: i32>(a: __m128d) -> i32 {
    static_assert_sae!(SAE);
    unsafe { vcvttsd2sis(a.as_f64x2(), SAE) }
}

/// Convert the lower double-precision (64-bit) floating-point element in `a` to a 32-bit
/// integer with truncation and saturation.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
pub fn _mm_cvtts_roundsd_si32<const SAE: i32>(a: __m128d) -> i32 {
    _mm_cvtts_roundsd_i32::<SAE>(a)
}

/// Convert the lower double-precision (64-bit) floating-point element in `a` to a 32-bit
/// unsigned integer with truncation and saturation.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttsd2usis, SAE = 8)
)]
pub fn _mm_cvtts_roundsd_u32<const SAE: i32>(a: __m128d) -> u32 {
    static_assert_sae!(SAE);
    unsafe { vcvttsd2usis(a.as_f64x2(), SAE) }
}

/// Convert the lower single-precision (32-bit) floating-point element in `a` to a 32-bit
/// integer with truncation and saturation.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttss2sis, SAE = 8)
)]
pub fn _mm_cvtts_roundss_i32<const SAE: i32>(a: __m128) -> i32 {
    static_assert_sae!(SAE);
    unsafe { vcvttss2sis(a.as_f32x4(), SAE) }
}

/// Convert the lower single-precision (32-bit) floating-point element in `a` to a 32-bit
/// integer with truncation and saturation.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
pub fn _mm_cvtts_roundss_si32<const SAE: i32>(a: __m128) -> i32 {
    _mm_cvtts_roundss_i32::<SAE>(a)
}

/// Convert the lower single-precision (32-bit) floating-point element in `a` to a 32-bit
/// unsigned integer with truncation and saturation.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttss2usis, SAE = 8)
)]
pub fn _mm_cvtts_roundss_u32<const SAE: i32>(a: __m128) -> u32 {
    static_assert_sae!(SAE);
    unsafe { vcvttss2usis(a.as_f32x4(), SAE) }
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst`.
/// The upper 64 bits of `dst` are zeroed out.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2dqs))]
pub fn _mm_cvtts_pd_epi32(a: __m128d) -> __m128i {
    _mm_mask_cvtts_pd_epi32(_mm_undefined_si128(), !0, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
/// The upper 64 bits of `dst` are zeroed out.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2dqs))]
pub fn _mm_mask_cvtts_pd_epi32(src: __m128i, k: __mmask8, a: __m128d) -> __m128i {
    unsafe { vcvttpd2dqs_128(a.as_f64x2(), src.as_i32x4(), k).as_m128i() }
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// The upper 64 bits of `dst` are zeroed out.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2dqs))]
pub fn _mm_maskz_cvtts_pd_epi32(k: __mmask8, a: __m128d) -> __m128i {
    _mm_mask_cvtts_pd_epi32(_mm_setzero_si128(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2dqs))]
pub fn _mm256_cvtts_pd_epi32(a: __m256d) -> __m128i {
    _mm256_mask_cvtts_pd_epi32(_mm_undefined_si128(), !0, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2dqs))]
pub fn _mm256_mask_cvtts_pd_epi32(src: __m128i, k: __mmask8, a: __m256d) -> __m128i {
    unsafe { vcvttpd2dqs_256(a.as_f64x4(), src.as_i32x4(), k).as_m128i() }
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2dqs))]
pub fn _mm256_maskz_cvtts_pd_epi32(k: __mmask8, a: __m256d) -> __m128i {
    _mm256_mask_cvtts_pd_epi32(_mm_setzero_si128(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2dqs))]
pub fn _mm512_cvtts_pd_epi32(a: __m512d) -> __m256i {
    _mm512_mask_cvtts_pd_epi32(_mm256_undefined_si256(), !0, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2dqs))]
pub fn _mm512_mask_cvtts_pd_epi32(src: __m256i, k: __mmask8, a: __m512d) -> __m256i {
    _mm512_mask_cvtts_roundpd_epi32::<_MM_FROUND_CUR_DIRECTION>(src, k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2dqs))]
pub fn _mm512_maskz_cvtts_pd_epi32(k: __mmask8, a: __m512d) -> __m256i {
    _mm512_mask_cvtts_pd_epi32(_mm256_setzero_si256(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttpd2dqs, SAE = 8)
)]
pub fn _mm512_cvtts_roundpd_epi32<const SAE: i32>(a: __m512d) -> __m256i {
    _mm512_mask_cvtts_roundpd_epi32::<SAE>(_mm256_undefined_si256(), !0, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttpd2dqs, SAE = 8)
)]
pub fn _mm512_mask_cvtts_roundpd_epi32<const SAE: i32>(
    src: __m256i,
    k: __mmask8,
    a: __m512d,
) -> __m256i {
    static_assert_sae!(SAE);
    unsafe { vcvttpd2dqs_512(a.as_f64x8(), src.as_i32x8(), k, SAE).as_m256i() }
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttpd2dqs, SAE = 8)
)]
pub fn _mm512_maskz_cvtts_roundpd_epi32<const SAE: i32>(k: __mmask8, a: __m512d) -> __m256i {
    _mm512_mask_cvtts_roundpd_epi32::<SAE>(_mm256_setzero_si256(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst`.
/// The upper 64 bits of `dst` are zeroed out.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2udqs))]
pub fn _mm_cvtts_pd_epu32(a: __m128d) -> __m128i {
    _mm_mask_cvtts_pd_epu32(_mm_undefined_si128(), !0, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
/// The upper 64 bits of `dst` are zeroed out.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2udqs))]
pub fn _mm_mask_cvtts_pd_epu32(src: __m128i, k: __mmask8, a: __m128d) -> __m128i {
    unsafe { vcvttpd2udqs_128(a.as_f64x2(), src.as_u32x4(), k).as_m128i() }
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// The upper 64 bits of `dst` are zeroed out.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2udqs))]
pub fn _mm_maskz_cvtts_pd_epu32(k: __mmask8, a: __m128d) -> __m128i {
    _mm_mask_cvtts_pd_epu32(_mm_setzero_si128(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2udqs))]
pub fn _mm256_cvtts_pd_epu32(a: __m256d) -> __m128i {
    _mm256_mask_cvtts_pd_epu32(_mm_undefined_si128(), !0, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2udqs))]
pub fn _mm256_mask_cvtts_pd_epu32(src: __m128i, k: __mmask8, a: __m256d) -> __m128i {
    unsafe { vcvttpd2udqs_256(a.as_f64x4(), src.as_u32x4(), k).as_m128i() }
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2udqs))]
pub fn _mm256_maskz_cvtts_pd_epu32(k: __mmask8, a: __m256d) -> __m128i {
    _mm256_mask_cvtts_pd_epu32(_mm_setzero_si128(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2udqs))]
pub fn _mm512_cvtts_pd_epu32(a: __m512d) -> __m256i {
    _mm512_mask_cvtts_pd_epu32(_mm256_undefined_si256(), !0, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2udqs))]
pub fn _mm512_mask_cvtts_pd_epu32(src: __m256i, k: __mmask8, a: __m512d) -> __m256i {
    _mm512_mask_cvtts_roundpd_epu32::<_MM_FROUND_CUR_DIRECTION>(src, k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2udqs))]
pub fn _mm512_maskz_cvtts_pd_epu32(k: __mmask8, a: __m512d) -> __m256i {
    _mm512_mask_cvtts_pd_epu32(_mm256_setzero_si256(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttpd2udqs, SAE = 8)
)]
pub fn _mm512_cvtts_roundpd_epu32<const SAE: i32>(a: __m512d) -> __m256i {
    _mm512_mask_cvtts_roundpd_epu32::<SAE>(_mm256_undefined_si256(), !0, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttpd2udqs, SAE = 8)
)]
pub fn _mm512_mask_cvtts_roundpd_epu32<const SAE: i32>(
    src: __m256i,
    k: __mmask8,
    a: __m512d,
) -> __m256i {
    static_assert_sae!(SAE);
    unsafe { vcvttpd2udqs_512(a.as_f64x8(), src.as_u32x8(), k, SAE).as_m256i() }
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttpd2udqs, SAE = 8)
)]
pub fn _mm512_maskz_cvtts_roundpd_epu32<const SAE: i32>(k: __mmask8, a: __m512d) -> __m256i {
    _mm512_mask_cvtts_roundpd_epu32::<SAE>(_mm256_setzero_si256(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2qqs))]
pub fn _mm_cvtts_pd_epi64(a: __m128d) -> __m128i {
    _mm_mask_cvtts_pd_epi64(_mm_undefined_si128(), !0, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2qqs))]
pub fn _mm_mask_cvtts_pd_epi64(src: __m128i, k: __mmask8, a: __m128d) -> __m128i {
    unsafe { vcvttpd2qqs_128(a.as_f64x2(), src.as_i64x2(), k).as_m128i() }
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2qqs))]
pub fn _mm_maskz_cvtts_pd_epi64(k: __mmask8, a: __m128d) -> __m128i {
    _mm_mask_cvtts_pd_epi64(_mm_setzero_si128(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2qqs))]
pub fn _mm256_cvtts_pd_epi64(a: __m256d) -> __m256i {
    _mm256_mask_cvtts_pd_epi64(_mm256_undefined_si256(), !0, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2qqs))]
pub fn _mm256_mask_cvtts_pd_epi64(src: __m256i, k: __mmask8, a: __m256d) -> __m256i {
    unsafe { vcvttpd2qqs_256(a.as_f64x4(), src.as_i64x4(), k).as_m256i() }
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2qqs))]
pub fn _mm256_maskz_cvtts_pd_epi64(k: __mmask8, a: __m256d) -> __m256i {
    _mm256_mask_cvtts_pd_epi64(_mm256_setzero_si256(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2qqs))]
pub fn _mm512_cvtts_pd_epi64(a: __m512d) -> __m512i {
    _mm512_mask_cvtts_pd_epi64(_mm512_undefined_epi32(), !0, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2qqs))]
pub fn _mm512_mask_cvtts_pd_epi64(src: __m512i, k: __mmask8, a: __m512d) -> __m512i {
    _mm512_mask_cvtts_roundpd_epi64::<_MM_FROUND_CUR_DIRECTION>(src, k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2qqs))]
pub fn _mm512_maskz_cvtts_pd_epi64(k: __mmask8, a: __m512d) -> __m512i {
    _mm512_mask_cvtts_pd_epi64(_mm512_setzero_si512(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttpd2qqs, SAE = 8)
)]
pub fn _mm512_cvtts_roundpd_epi64<const SAE: i32>(a: __m512d) -> __m512i {
    _mm512_mask_cvtts_roundpd_epi64::<SAE>(_mm512_undefined_epi32(), !0, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttpd2qqs, SAE = 8)
)]
pub fn _mm512_mask_cvtts_roundpd_epi64<const SAE: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m512d,
) -> __m512i {
    static_assert_sae!(SAE);
    unsafe { vcvttpd2qqs_512(a.as_f64x8(), src.as_i64x8(), k, SAE).as_m512i() }
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttpd2qqs, SAE = 8)
)]
pub fn _mm512_maskz_cvtts_roundpd_epi64<const SAE: i32>(k: __mmask8, a: __m512d) -> __m512i {
    _mm512_mask_cvtts_roundpd_epi64::<SAE>(_mm512_setzero_si512(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2uqqs))]
pub fn _mm_cvtts_pd_epu64(a: __m128d) -> __m128i {
    _mm_mask_cvtts_pd_epu64(_mm_undefined_si128(), !0, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2uqqs))]
pub fn _mm_mask_cvtts_pd_epu64(src: __m128i, k: __mmask8, a: __m128d) -> __m128i {
    unsafe { vcvttpd2uqqs_128(a.as_f64x2(), src.as_u64x2(), k).as_m128i() }
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2uqqs))]
pub fn _mm_maskz_cvtts_pd_epu64(k: __mmask8, a: __m128d) -> __m128i {
    _mm_mask_cvtts_pd_epu64(_mm_setzero_si128(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2uqqs))]
pub fn _mm256_cvtts_pd_epu64(a: __m256d) -> __m256i {
    _mm256_mask_cvtts_pd_epu64(_mm256_undefined_si256(), !0, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2uqqs))]
pub fn _mm256_mask_cvtts_pd_epu64(src: __m256i, k: __mmask8, a: __m256d) -> __m256i {
    unsafe { vcvttpd2uqqs_256(a.as_f64x4(), src.as_u64x4(), k).as_m256i() }
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2uqqs))]
pub fn _mm256_maskz_cvtts_pd_epu64(k: __mmask8, a: __m256d) -> __m256i {
    _mm256_mask_cvtts_pd_epu64(_mm256_setzero_si256(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst`
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2uqqs))]
pub fn _mm512_cvtts_pd_epu64(a: __m512d) -> __m512i {
    _mm512_mask_cvtts_pd_epu64(_mm512_undefined_epi32(), !0, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2uqqs))]
pub fn _mm512_mask_cvtts_pd_epu64(src: __m512i, k: __mmask8, a: __m512d) -> __m512i {
    _mm512_mask_cvtts_roundpd_epu64::<_MM_FROUND_CUR_DIRECTION>(src, k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttpd2uqqs))]
pub fn _mm512_maskz_cvtts_pd_epu64(k: __mmask8, a: __m512d) -> __m512i {
    _mm512_mask_cvtts_pd_epu64(_mm512_setzero_si512(), k, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttpd2uqqs, SAE = 8)
)]
pub fn _mm512_cvtts_roundpd_epu64<const SAE: i32>(a: __m512d) -> __m512i {
    _mm512_mask_cvtts_roundpd_epu64::<SAE>(_mm512_undefined_epi32(), !0, a)
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttpd2uqqs, SAE = 8)
)]
pub fn _mm512_mask_cvtts_roundpd_epu64<const SAE: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m512d,
) -> __m512i {
    static_assert_sae!(SAE);
    unsafe { vcvttpd2uqqs_512(a.as_f64x8(), src.as_u64x8(), k, SAE).as_m512i() }
}

/// Convert packed double-precision (64-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttpd2uqqs, SAE = 8)
)]
pub fn _mm512_maskz_cvtts_roundpd_epu64<const SAE: i32>(k: __mmask8, a: __m512d) -> __m512i {
    _mm512_mask_cvtts_roundpd_epu64::<SAE>(_mm512_setzero_si512(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2dqs))]
pub fn _mm_cvtts_ps_epi32(a: __m128) -> __m128i {
    _mm_mask_cvtts_ps_epi32(_mm_undefined_si128(), !0, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2dqs))]
pub fn _mm_mask_cvtts_ps_epi32(src: __m128i, k: __mmask8, a: __m128) -> __m128i {
    unsafe { vcvttps2dqs_128(a.as_f32x4(), src.as_i32x4(), k).as_m128i() }
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2dqs))]
pub fn _mm_maskz_cvtts_ps_epi32(k: __mmask8, a: __m128) -> __m128i {
    _mm_mask_cvtts_ps_epi32(_mm_setzero_si128(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2dqs))]
pub fn _mm256_cvtts_ps_epi32(a: __m256) -> __m256i {
    _mm256_mask_cvtts_ps_epi32(_mm256_undefined_si256(), !0, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2dqs))]
pub fn _mm256_mask_cvtts_ps_epi32(src: __m256i, k: __mmask8, a: __m256) -> __m256i {
    unsafe { vcvttps2dqs_256(a.as_f32x8(), src.as_i32x8(), k).as_m256i() }
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2dqs))]
pub fn _mm256_maskz_cvtts_ps_epi32(k: __mmask8, a: __m256) -> __m256i {
    _mm256_mask_cvtts_ps_epi32(_mm256_setzero_si256(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2dqs))]
pub fn _mm512_cvtts_ps_epi32(a: __m512) -> __m512i {
    _mm512_mask_cvtts_ps_epi32(_mm512_undefined_epi32(), !0, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2dqs))]
pub fn _mm512_mask_cvtts_ps_epi32(src: __m512i, k: __mmask16, a: __m512) -> __m512i {
    _mm512_mask_cvtts_roundps_epi32::<_MM_FROUND_CUR_DIRECTION>(src, k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2dqs))]
pub fn _mm512_maskz_cvtts_ps_epi32(k: __mmask16, a: __m512) -> __m512i {
    _mm512_mask_cvtts_ps_epi32(_mm512_setzero_si512(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttps2dqs, SAE = 8)
)]
pub fn _mm512_cvtts_roundps_epi32<const SAE: i32>(a: __m512) -> __m512i {
    _mm512_mask_cvtts_roundps_epi32::<SAE>(_mm512_undefined_epi32(), !0, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttps2dqs, SAE = 8)
)]
pub fn _mm512_mask_cvtts_roundps_epi32<const SAE: i32>(
    src: __m512i,
    k: __mmask16,
    a: __m512,
) -> __m512i {
    unsafe { vcvttps2dqs_512(a.as_f32x16(), src.as_i32x16(), k, SAE).as_m512i() }
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttps2dqs, SAE = 8)
)]
pub fn _mm512_maskz_cvtts_roundps_epi32<const SAE: i32>(k: __mmask16, a: __m512) -> __m512i {
    _mm512_mask_cvtts_roundps_epi32::<SAE>(_mm512_setzero_si512(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2udqs))]
pub fn _mm_cvtts_ps_epu32(a: __m128) -> __m128i {
    _mm_mask_cvtts_ps_epu32(_mm_undefined_si128(), !0, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2udqs))]
pub fn _mm_mask_cvtts_ps_epu32(src: __m128i, k: __mmask8, a: __m128) -> __m128i {
    unsafe { vcvttps2udqs_128(a.as_f32x4(), src.as_u32x4(), k).as_m128i() }
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2udqs))]
pub fn _mm_maskz_cvtts_ps_epu32(k: __mmask8, a: __m128) -> __m128i {
    _mm_mask_cvtts_ps_epu32(_mm_setzero_si128(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2udqs))]
pub fn _mm256_cvtts_ps_epu32(a: __m256) -> __m256i {
    _mm256_mask_cvtts_ps_epu32(_mm256_undefined_si256(), !0, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2udqs))]
pub fn _mm256_mask_cvtts_ps_epu32(src: __m256i, k: __mmask8, a: __m256) -> __m256i {
    unsafe { vcvttps2udqs_256(a.as_f32x8(), src.as_u32x8(), k).as_m256i() }
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2udqs))]
pub fn _mm256_maskz_cvtts_ps_epu32(k: __mmask8, a: __m256) -> __m256i {
    _mm256_mask_cvtts_ps_epu32(_mm256_setzero_si256(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst`
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2udqs))]
pub fn _mm512_cvtts_ps_epu32(a: __m512) -> __m512i {
    _mm512_mask_cvtts_ps_epu32(_mm512_undefined_epi32(), !0, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2udqs))]
pub fn _mm512_mask_cvtts_ps_epu32(src: __m512i, k: __mmask16, a: __m512) -> __m512i {
    _mm512_mask_cvtts_roundps_epu32::<_MM_FROUND_CUR_DIRECTION>(src, k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2udqs))]
pub fn _mm512_maskz_cvtts_ps_epu32(k: __mmask16, a: __m512) -> __m512i {
    _mm512_mask_cvtts_ps_epu32(_mm512_setzero_si512(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttps2udqs, SAE = 8)
)]
pub fn _mm512_cvtts_roundps_epu32<const SAE: i32>(a: __m512) -> __m512i {
    _mm512_mask_cvtts_roundps_epu32::<SAE>(_mm512_undefined_epi32(), !0, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttps2udqs, SAE = 8)
)]
pub fn _mm512_mask_cvtts_roundps_epu32<const SAE: i32>(
    src: __m512i,
    k: __mmask16,
    a: __m512,
) -> __m512i {
    static_assert_sae!(SAE);
    unsafe { vcvttps2udqs_512(a.as_f32x16(), src.as_u32x16(), k, SAE).as_m512i() }
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 32-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttps2udqs, SAE = 8)
)]
pub fn _mm512_maskz_cvtts_roundps_epu32<const SAE: i32>(k: __mmask16, a: __m512) -> __m512i {
    _mm512_mask_cvtts_roundps_epu32::<SAE>(_mm512_setzero_si512(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2qqs))]
pub fn _mm_cvtts_ps_epi64(a: __m128) -> __m128i {
    _mm_mask_cvtts_ps_epi64(_mm_undefined_si128(), !0, a)
}

/// Convert packed single-precision (32-bit) floating-point elements from the lower 64 bits of `a`
/// to packed 64-bit integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2qqs))]
pub fn _mm_mask_cvtts_ps_epi64(src: __m128i, k: __mmask8, a: __m128) -> __m128i {
    unsafe { vcvttps2qqs_128(a.as_f32x4(), src.as_i64x2(), k).as_m128i() }
}

/// Convert packed single-precision (32-bit) floating-point elements from the lower 64 bits of  `a`
/// to packed 64-bit integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2qqs))]
pub fn _mm_maskz_cvtts_ps_epi64(k: __mmask8, a: __m128) -> __m128i {
    _mm_mask_cvtts_ps_epi64(_mm_setzero_si128(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements from the lower 64 bits of  `a`
/// to packed 64-bit integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2qqs))]
pub fn _mm256_cvtts_ps_epi64(a: __m128) -> __m256i {
    _mm256_mask_cvtts_ps_epi64(_mm256_undefined_si256(), !0, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2qqs))]
pub fn _mm256_mask_cvtts_ps_epi64(src: __m256i, k: __mmask8, a: __m128) -> __m256i {
    unsafe { vcvttps2qqs_256(a.as_f32x4(), src.as_i64x4(), k).as_m256i() }
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2qqs))]
pub fn _mm256_maskz_cvtts_ps_epi64(k: __mmask8, a: __m128) -> __m256i {
    _mm256_mask_cvtts_ps_epi64(_mm256_setzero_si256(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2qqs))]
pub fn _mm512_cvtts_ps_epi64(a: __m256) -> __m512i {
    _mm512_mask_cvtts_ps_epi64(_mm512_undefined_epi32(), !0, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2qqs))]
pub fn _mm512_mask_cvtts_ps_epi64(src: __m512i, k: __mmask8, a: __m256) -> __m512i {
    _mm512_mask_cvtts_roundps_epi64::<_MM_FROUND_CUR_DIRECTION>(src, k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2qqs))]
pub fn _mm512_maskz_cvtts_ps_epi64(k: __mmask8, a: __m256) -> __m512i {
    _mm512_mask_cvtts_ps_epi64(_mm512_setzero_si512(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttps2qqs, SAE = 8)
)]
pub fn _mm512_cvtts_roundps_epi64<const SAE: i32>(a: __m256) -> __m512i {
    _mm512_mask_cvtts_roundps_epi64::<SAE>(_mm512_undefined_epi32(), !0, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttps2qqs, SAE = 8)
)]
pub fn _mm512_mask_cvtts_roundps_epi64<const SAE: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m256,
) -> __m512i {
    static_assert_sae!(SAE);
    unsafe { vcvttps2qqs_512(a.as_f32x8(), src.as_i64x8(), k, SAE).as_m512i() }
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 64-bit integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttps2qqs, SAE = 8)
)]
pub fn _mm512_maskz_cvtts_roundps_epi64<const SAE: i32>(k: __mmask8, a: __m256) -> __m512i {
    _mm512_mask_cvtts_roundps_epi64::<SAE>(_mm512_setzero_si512(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements from the lower 64 bits of  `a`
/// to packed 64-bit unsigned integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2uqqs))]
pub fn _mm_cvtts_ps_epu64(a: __m128) -> __m128i {
    _mm_mask_cvtts_ps_epu64(_mm_undefined_si128(), !0, a)
}

/// Convert packed single-precision (32-bit) floating-point elements from the lower 64 bits of  `a`
/// to packed 64-bit unsigned integers with truncation and saturation, and store the results in `dst`
/// using writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2uqqs))]
pub fn _mm_mask_cvtts_ps_epu64(src: __m128i, k: __mmask8, a: __m128) -> __m128i {
    unsafe { vcvttps2uqqs_128(a.as_f32x4(), src.as_u64x2(), k).as_m128i() }
}

/// Convert packed single-precision (32-bit) floating-point elements from the lower 64 bits of  `a`
/// to packed 64-bit unsigned integers with truncation and saturation, and store the results in `dst`
/// using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2uqqs))]
pub fn _mm_maskz_cvtts_ps_epu64(k: __mmask8, a: __m128) -> __m128i {
    _mm_mask_cvtts_ps_epu64(_mm_setzero_si128(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2uqqs))]
pub fn _mm256_cvtts_ps_epu64(a: __m128) -> __m256i {
    _mm256_mask_cvtts_ps_epu64(_mm256_undefined_si256(), !0, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2uqqs))]
pub fn _mm256_mask_cvtts_ps_epu64(src: __m256i, k: __mmask8, a: __m128) -> __m256i {
    unsafe { vcvttps2uqqs_256(a.as_f32x4(), src.as_u64x4(), k).as_m256i() }
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2uqqs))]
pub fn _mm256_maskz_cvtts_ps_epu64(k: __mmask8, a: __m128) -> __m256i {
    _mm256_mask_cvtts_ps_epu64(_mm256_setzero_si256(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst`.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2uqqs))]
pub fn _mm512_cvtts_ps_epu64(a: __m256) -> __m512i {
    _mm512_mask_cvtts_ps_epu64(_mm512_undefined_epi32(), !0, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2uqqs))]
pub fn _mm512_mask_cvtts_ps_epu64(src: __m512i, k: __mmask8, a: __m256) -> __m512i {
    _mm512_mask_cvtts_roundps_epu64::<_MM_FROUND_CUR_DIRECTION>(src, k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vcvttps2uqqs))]
pub fn _mm512_maskz_cvtts_ps_epu64(k: __mmask8, a: __m256) -> __m512i {
    _mm512_mask_cvtts_ps_epu64(_mm512_setzero_si512(), k, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttps2uqqs, SAE = 8)
)]
pub fn _mm512_cvtts_roundps_epu64<const SAE: i32>(a: __m256) -> __m512i {
    _mm512_mask_cvtts_roundps_epu64::<SAE>(_mm512_undefined_epi32(), !0, a)
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// writemask `k` (elements are copied from `src` when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttps2uqqs, SAE = 8)
)]
pub fn _mm512_mask_cvtts_roundps_epu64<const SAE: i32>(
    src: __m512i,
    k: __mmask8,
    a: __m256,
) -> __m512i {
    static_assert_sae!(SAE);
    unsafe { vcvttps2uqqs_512(a.as_f32x8(), src.as_u64x8(), k, SAE).as_m512i() }
}

/// Convert packed single-precision (32-bit) floating-point elements in `a` to packed
/// 64-bit unsigned integers with truncation and saturation, and store the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vcvttps2uqqs, SAE = 8)
)]
pub fn _mm512_maskz_cvtts_roundps_epu64<const SAE: i32>(k: __mmask8, a: __m256) -> __m512i {
    _mm512_mask_cvtts_roundps_epu64::<SAE>(_mm512_setzero_si512(), k, a)
}

/// Performs a min/max comparison between packed double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`.
///
/// -------------------------------------------------------------------------------------------------------------------------------
/// |`IMM8[4]`|`IMM8[1:0]`|        Operation        |         Description                                                         |
/// |:-------:|:---------:|:-----------------------:|:----------------------------------------------------------------------------|
/// |    0    |    00     |         Minimum         | `a` if `a<=b`, `b` if `b<a`, and qNan if either operand is NaN              |
/// |    0    |    01     |         Maximum         | `a` if `a>=b`, `b` if `b>a`, and qNan if either operand is NaN              |
/// |    0    |    10     |     MinimumMagnitude    | `a` if `\|a\|<\|b\|`, `b` if `\|b\|<\|a\|`, otherwise `Minimum(a, b)`       |
/// |    0    |    11     |     MaximumMagnitude    | `a` if `\|a\|>\|b\|`, `b` if `\|b\|>\|a\|`, otherwise `Maximum(a, b)`       |
/// |    1    |    00     |       MinimumNumber     | `a` if `a<=b`, `b` if `b<a`. If only one operand is NaN, the other one is returned. If both operands are NaNs, a qNaN is returned |
/// |    1    |    01     |       MaximumNumber     | `a` if `a>=b`, `b` if `b>a`. If only one operand is NaN, the other one is returned. If both operands are NaNs, a qNaN is returned |
/// |    1    |    10     |  MinimumMagnitudeNumber | `a` if `\|a\|<\|b\|`, `b` if `\|b\|<\|x\|`, otherwise `MinimumNumber(a, b)` |
/// |    1    |    10     |  MaximumMagnitudeNumber | `a` if `\|a\|>\|b\|`, `b` if `\|b\|>\|x\|`, otherwise `MaximumNumber(a, b)` |
/// -------------------------------------------------------------------------------------------------------------------------------
///
/// The sign of the output is decided using `IMM[3:2]`
///
/// ---------------------------------------------
/// |`IMM8[3:2]`|              Sign             |
/// |:---------:|:-----------------------------:|
/// |     00    | Use sign of the first operand |
/// |     01    | Preserve sign of the result   |
/// |     10    | Set sign to +ve               |
/// |     11    | Set sign to -ve               |
/// ---------------------------------------------
///
/// For more details, including behaviour for NaNs and denormals, refer to the [AVX10.2 Spec].
///
/// [AVX10.2 Spec]: https://www.intel.com/content/www/us/en/content-details/913918/intel-advanced-vector-extensions-10-2-intel-avx10-2-architecture-specification.html
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxpd, IMM8 = 0)
)]
pub fn _mm_minmax_pd<const IMM8: i32>(a: __m128d, b: __m128d) -> __m128d {
    _mm_mask_minmax_pd::<IMM8>(_mm_undefined_pd(), !0, a, b)
}

/// Performs a min/max comparison between packed double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// writemask `k` (elements are copied from src when the corresponding mask bit is not set).
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxpd, IMM8 = 0)
)]
pub fn _mm_mask_minmax_pd<const IMM8: i32>(
    src: __m128d,
    k: __mmask8,
    a: __m128d,
    b: __m128d,
) -> __m128d {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { vminmaxpd128(a.as_f64x2(), b.as_f64x2(), IMM8, src.as_f64x2(), k as u8).as_m128d() }
}

/// Performs a min/max comparison between packed double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxpd, IMM8 = 0)
)]
pub fn _mm_maskz_minmax_pd<const IMM8: i32>(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    _mm_mask_minmax_pd::<IMM8>(_mm_setzero_pd(), k, a, b)
}

/// Performs a min/max comparison between packed double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxpd, IMM8 = 0)
)]
pub fn _mm256_minmax_pd<const IMM8: i32>(a: __m256d, b: __m256d) -> __m256d {
    _mm256_mask_minmax_pd::<IMM8>(_mm256_undefined_pd(), !0, a, b)
}

/// Performs a min/max comparison between packed double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// writemask `k` (elements are copied from src when the corresponding mask bit is not set).
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxpd, IMM8 = 0)
)]
pub fn _mm256_mask_minmax_pd<const IMM8: i32>(
    src: __m256d,
    k: __mmask8,
    a: __m256d,
    b: __m256d,
) -> __m256d {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { vminmaxpd256(a.as_f64x4(), b.as_f64x4(), IMM8, src.as_f64x4(), k as u8).as_m256d() }
}

/// Performs a min/max comparison between packed double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxpd, IMM8 = 0)
)]
pub fn _mm256_maskz_minmax_pd<const IMM8: i32>(k: __mmask8, a: __m256d, b: __m256d) -> __m256d {
    _mm256_mask_minmax_pd::<IMM8>(_mm256_setzero_pd(), k, a, b)
}

/// Performs a min/max comparison between packed double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxpd, IMM8 = 0)
)]
pub fn _mm512_minmax_pd<const IMM8: i32>(a: __m512d, b: __m512d) -> __m512d {
    _mm512_mask_minmax_pd::<IMM8>(_mm512_undefined_pd(), !0, a, b)
}

/// Performs a min/max comparison between packed double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// writemask `k` (elements are copied from src when the corresponding mask bit is not set).
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxpd, IMM8 = 0)
)]
pub fn _mm512_mask_minmax_pd<const IMM8: i32>(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    b: __m512d,
) -> __m512d {
    _mm512_mask_minmax_round_pd::<IMM8, _MM_FROUND_CUR_DIRECTION>(src, k, a, b)
}

/// Performs a min/max comparison between packed double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxpd, IMM8 = 0)
)]
pub fn _mm512_maskz_minmax_pd<const IMM8: i32>(k: __mmask8, a: __m512d, b: __m512d) -> __m512d {
    _mm512_mask_minmax_pd::<IMM8>(_mm512_setzero_pd(), k, a, b)
}

/// Performs a min/max comparison between packed double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxpd, IMM8 = 0, SAE = 8)
)]
pub fn _mm512_minmax_round_pd<const IMM8: i32, const SAE: i32>(a: __m512d, b: __m512d) -> __m512d {
    _mm512_mask_minmax_round_pd::<IMM8, SAE>(_mm512_undefined_pd(), !0, a, b)
}

/// Performs a min/max comparison between packed double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// writemask `k` (elements are copied from src when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxpd, IMM8 = 0, SAE = 8)
)]
pub fn _mm512_mask_minmax_round_pd<const IMM8: i32, const SAE: i32>(
    src: __m512d,
    k: __mmask8,
    a: __m512d,
    b: __m512d,
) -> __m512d {
    static_assert_uimm_bits!(IMM8, 8);
    static_assert_sae!(SAE);
    unsafe {
        vminmaxpd512(
            a.as_f64x8(),
            b.as_f64x8(),
            IMM8,
            src.as_f64x8(),
            k as u8,
            SAE,
        )
        .as_m512d()
    }
}

/// Performs a min/max comparison between packed double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxpd, IMM8 = 0, SAE = 8)
)]
pub fn _mm512_maskz_minmax_round_pd<const IMM8: i32, const SAE: i32>(
    k: __mmask8,
    a: __m512d,
    b: __m512d,
) -> __m512d {
    _mm512_mask_minmax_round_pd::<IMM8, SAE>(_mm512_setzero_pd(), k, a, b)
}

/// Performs a min/max comparison between packed single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxps, IMM8 = 0)
)]
pub fn _mm_minmax_ps<const IMM8: i32>(a: __m128, b: __m128) -> __m128 {
    _mm_mask_minmax_ps::<IMM8>(_mm_undefined_ps(), !0, a, b)
}

/// Performs a min/max comparison between packed single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// writemask `k` (elements are copied from src when the corresponding mask bit is not set).
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxps, IMM8 = 0)
)]
pub fn _mm_mask_minmax_ps<const IMM8: i32>(
    src: __m128,
    k: __mmask8,
    a: __m128,
    b: __m128,
) -> __m128 {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { vminmaxps128(a.as_f32x4(), b.as_f32x4(), IMM8, src.as_f32x4(), k as u8).as_m128() }
}

/// Performs a min/max comparison between packed single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxps, IMM8 = 0)
)]
pub fn _mm_maskz_minmax_ps<const IMM8: i32>(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    _mm_mask_minmax_ps::<IMM8>(_mm_setzero_ps(), k, a, b)
}

/// Performs a min/max comparison between packed single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxps, IMM8 = 0)
)]
pub fn _mm256_minmax_ps<const IMM8: i32>(a: __m256, b: __m256) -> __m256 {
    _mm256_mask_minmax_ps::<IMM8>(_mm256_undefined_ps(), !0, a, b)
}

/// Performs a min/max comparison between packed single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// writemask `k` (elements are copied from src when the corresponding mask bit is not set).
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxps, IMM8 = 0)
)]
pub fn _mm256_mask_minmax_ps<const IMM8: i32>(
    src: __m256,
    k: __mmask8,
    a: __m256,
    b: __m256,
) -> __m256 {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { vminmaxps256(a.as_f32x8(), b.as_f32x8(), IMM8, src.as_f32x8(), k as u8).as_m256() }
}

/// Performs a min/max comparison between packed single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxps, IMM8 = 0)
)]
pub fn _mm256_maskz_minmax_ps<const IMM8: i32>(k: __mmask8, a: __m256, b: __m256) -> __m256 {
    _mm256_mask_minmax_ps::<IMM8>(_mm256_setzero_ps(), k, a, b)
}

/// Performs a min/max comparison between packed single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxps, IMM8 = 0)
)]
pub fn _mm512_minmax_ps<const IMM8: i32>(a: __m512, b: __m512) -> __m512 {
    _mm512_mask_minmax_ps::<IMM8>(_mm512_undefined_ps(), !0, a, b)
}

/// Performs a min/max comparison between packed single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// writemask `k` (elements are copied from src when the corresponding mask bit is not set).
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxps, IMM8 = 0)
)]
pub fn _mm512_mask_minmax_ps<const IMM8: i32>(
    src: __m512,
    k: __mmask16,
    a: __m512,
    b: __m512,
) -> __m512 {
    _mm512_mask_minmax_round_ps::<IMM8, _MM_FROUND_CUR_DIRECTION>(src, k, a, b)
}

/// Performs a min/max comparison between packed single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxps, IMM8 = 0)
)]
pub fn _mm512_maskz_minmax_ps<const IMM8: i32>(k: __mmask16, a: __m512, b: __m512) -> __m512 {
    _mm512_mask_minmax_ps::<IMM8>(_mm512_setzero_ps(), k, a, b)
}

/// Performs a min/max comparison between packed single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxps, IMM8 = 0, SAE = 8)
)]
pub fn _mm512_minmax_round_ps<const IMM8: i32, const SAE: i32>(a: __m512, b: __m512) -> __m512 {
    _mm512_mask_minmax_round_ps::<IMM8, SAE>(_mm512_undefined_ps(), !0, a, b)
}

/// Performs a min/max comparison between packed single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// writemask `k` (elements are copied from src when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxps, IMM8 = 0, SAE = 8)
)]
pub fn _mm512_mask_minmax_round_ps<const IMM8: i32, const SAE: i32>(
    src: __m512,
    k: __mmask16,
    a: __m512,
    b: __m512,
) -> __m512 {
    static_assert_uimm_bits!(IMM8, 8);
    static_assert_sae!(SAE);
    unsafe {
        vminmaxps512(
            a.as_f32x16(),
            b.as_f32x16(),
            IMM8,
            src.as_f32x16(),
            k as u16,
            SAE,
        )
        .as_m512()
    }
}

/// Performs a min/max comparison between packed single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxps, IMM8 = 0, SAE = 8)
)]
pub fn _mm512_maskz_minmax_round_ps<const IMM8: i32, const SAE: i32>(
    k: __mmask16,
    a: __m512,
    b: __m512,
) -> __m512 {
    _mm512_mask_minmax_round_ps::<IMM8, SAE>(_mm512_setzero_ps(), k, a, b)
}

/// Performs a min/max comparison between packed half-precision (16-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxph, IMM8 = 0)
)]
pub fn _mm_minmax_ph<const IMM8: i32>(a: __m128h, b: __m128h) -> __m128h {
    _mm_mask_minmax_ph::<IMM8>(_mm_undefined_ph(), !0, a, b)
}

/// Performs a min/max comparison between packed half-precision (16-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`. and stores the results in `dst` using
/// writemask `k` (elements are copied from src when the corresponding mask bit is not set).
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxph, IMM8 = 0)
)]
pub fn _mm_mask_minmax_ph<const IMM8: i32>(
    src: __m128h,
    k: __mmask8,
    a: __m128h,
    b: __m128h,
) -> __m128h {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe { vminmaxph128(a.as_f16x8(), b.as_f16x8(), IMM8, src.as_f16x8(), k as u8).as_m128h() }
}

/// Performs a min/max comparison between packed half-precision (16-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`. and stores the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxph, IMM8 = 0)
)]
pub fn _mm_maskz_minmax_ph<const IMM8: i32>(k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    _mm_mask_minmax_ph::<IMM8>(_mm_setzero_ph(), k, a, b)
}

/// Performs a min/max comparison between packed half-precision (16-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxph, IMM8 = 0)
)]
pub fn _mm256_minmax_ph<const IMM8: i32>(a: __m256h, b: __m256h) -> __m256h {
    _mm256_mask_minmax_ph::<IMM8>(_mm256_undefined_ph(), !0, a, b)
}

/// Performs a min/max comparison between packed half-precision (16-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`. and stores the results in `dst` using
/// writemask `k` (elements are copied from src when the corresponding mask bit is not set).
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxph, IMM8 = 0)
)]
pub fn _mm256_mask_minmax_ph<const IMM8: i32>(
    src: __m256h,
    k: __mmask16,
    a: __m256h,
    b: __m256h,
) -> __m256h {
    static_assert_uimm_bits!(IMM8, 8);
    unsafe {
        vminmaxph256(
            a.as_f16x16(),
            b.as_f16x16(),
            IMM8,
            src.as_f16x16(),
            k as u16,
        )
        .as_m256h()
    }
}

/// Performs a min/max comparison between packed half-precision (16-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`. and stores the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxph, IMM8 = 0)
)]
pub fn _mm256_maskz_minmax_ph<const IMM8: i32>(k: __mmask16, a: __m256h, b: __m256h) -> __m256h {
    _mm256_mask_minmax_ph::<IMM8>(_mm256_setzero_ph(), k, a, b)
}

/// Performs a min/max comparison between packed half-precision (16-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxph, IMM8 = 0)
)]
pub fn _mm512_minmax_ph<const IMM8: i32>(a: __m512h, b: __m512h) -> __m512h {
    _mm512_mask_minmax_ph::<IMM8>(_mm512_undefined_ph(), !0, a, b)
}

/// Performs a min/max comparison between packed half-precision (16-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`. and stores the results in `dst` using
/// writemask `k` (elements are copied from src when the corresponding mask bit is not set).
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxph, IMM8 = 0)
)]
pub fn _mm512_mask_minmax_ph<const IMM8: i32>(
    src: __m512h,
    k: __mmask32,
    a: __m512h,
    b: __m512h,
) -> __m512h {
    _mm512_mask_minmax_round_ph::<IMM8, _MM_FROUND_CUR_DIRECTION>(src, k, a, b)
}

/// Performs a min/max comparison between packed half-precision (16-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`. and stores the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxph, IMM8 = 0)
)]
pub fn _mm512_maskz_minmax_ph<const IMM8: i32>(k: __mmask32, a: __m512h, b: __m512h) -> __m512h {
    _mm512_mask_minmax_ph::<IMM8>(_mm512_setzero_ph(), k, a, b)
}

/// Performs a min/max comparison between packed half-precision (16-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxph, IMM8 = 0, SAE = 8)
)]
pub fn _mm512_minmax_round_ph<const IMM8: i32, const SAE: i32>(a: __m512h, b: __m512h) -> __m512h {
    _mm512_mask_minmax_round_ph::<IMM8, SAE>(_mm512_undefined_ph(), !0, a, b)
}

/// Performs a min/max comparison between packed half-precision (16-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`. and stores the results in `dst` using
/// writemask `k` (elements are copied from src when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxph, IMM8 = 0, SAE = 8)
)]
pub fn _mm512_mask_minmax_round_ph<const IMM8: i32, const SAE: i32>(
    src: __m512h,
    k: __mmask32,
    a: __m512h,
    b: __m512h,
) -> __m512h {
    static_assert_uimm_bits!(IMM8, 8);
    static_assert_sae!(SAE);
    unsafe {
        vminmaxph512(
            a.as_f16x32(),
            b.as_f16x32(),
            IMM8,
            src.as_f16x32(),
            k as u32,
            SAE,
        )
        .as_m512h()
    }
}

/// Performs a min/max comparison between packed half-precision (16-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`. and stores the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set).
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxph, IMM8 = 0, SAE = 8)
)]
pub fn _mm512_maskz_minmax_round_ph<const IMM8: i32, const SAE: i32>(
    k: __mmask32,
    a: __m512h,
    b: __m512h,
) -> __m512h {
    _mm512_mask_minmax_round_ph::<IMM8, SAE>(_mm512_setzero_ph(), k, a, b)
}

/// Performs a min/max comparison between the lower double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and copy the upper element from
/// `a` to the upper element of `dst`.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxsd, IMM8 = 0)
)]
pub fn _mm_minmax_sd<const IMM8: i32>(a: __m128d, b: __m128d) -> __m128d {
    _mm_mask_minmax_sd::<IMM8>(_mm_undefined_pd(), !0, a, b)
}

/// Performs a min/max comparison between the lower double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// writemask `k` (elements are copied from src when the corresponding mask bit is not set), and
/// copy the upper element from `a` to the upper element of `dst`.
///
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxsd, IMM8 = 0)
)]
pub fn _mm_mask_minmax_sd<const IMM8: i32>(
    src: __m128d,
    k: __mmask8,
    a: __m128d,
    b: __m128d,
) -> __m128d {
    _mm_mask_minmax_round_sd::<IMM8, _MM_FROUND_CUR_DIRECTION>(src, k, a, b)
}

/// Performs a min/max comparison between the lower double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst`
/// using zeromask `k` (elements are zeroed out when the corresponding mask bit is not set), and
/// copy the upper element from `a` to the upper element of `dst`.
///
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxsd, IMM8 = 0)
)]
pub fn _mm_maskz_minmax_sd<const IMM8: i32>(k: __mmask8, a: __m128d, b: __m128d) -> __m128d {
    _mm_mask_minmax_sd::<IMM8>(_mm_setzero_pd(), k, a, b)
}

/// Performs a min/max comparison between the lower double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set), and copy
/// the upper element from `a` to the upper element of `dst`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxsd, IMM8 = 0, SAE = 8)
)]
pub fn _mm_minmax_round_sd<const IMM8: i32, const SAE: i32>(a: __m128d, b: __m128d) -> __m128d {
    _mm_mask_minmax_round_sd::<IMM8, SAE>(_mm_undefined_pd(), !0, a, b)
}

/// Performs a min/max comparison between the lower double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// writemask `k` (elements are copied from src when the corresponding mask bit is not set), and
/// copy the upper element from `a` to the upper element of `dst`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxsd, IMM8 = 0, SAE = 8)
)]
pub fn _mm_mask_minmax_round_sd<const IMM8: i32, const SAE: i32>(
    src: __m128d,
    k: __mmask8,
    a: __m128d,
    b: __m128d,
) -> __m128d {
    static_assert_uimm_bits!(IMM8, 8);
    static_assert_sae!(SAE);
    unsafe {
        vminmaxsd(
            a.as_f64x2(),
            b.as_f64x2(),
            IMM8,
            src.as_f64x2(),
            k as u8,
            SAE,
        )
        .as_m128d()
    }
}

/// Performs a min/max comparison between the lower double-precision (64-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set), and
/// copy the upper element from `a` to the upper element of `dst`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxsd, IMM8 = 0, SAE = 8)
)]
pub fn _mm_maskz_minmax_round_sd<const IMM8: i32, const SAE: i32>(
    k: __mmask8,
    a: __m128d,
    b: __m128d,
) -> __m128d {
    _mm_mask_minmax_round_sd::<IMM8, SAE>(_mm_setzero_pd(), k, a, b)
}

/// Performs a min/max comparison between the lower single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and copy the upper 3 packed elements
/// from `a` to the upper elements of `dst`.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxss, IMM8 = 0)
)]
pub fn _mm_minmax_ss<const IMM8: i32>(a: __m128, b: __m128) -> __m128 {
    _mm_mask_minmax_ss::<IMM8>(_mm_undefined_ps(), !0, a, b)
}

/// Performs a min/max comparison between the lower single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// writemask `k` (elements are copied from src when the corresponding mask bit is not set), and
/// copy the upper 3 packed elements from `a` to the upper elements of `dst`.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxss, IMM8 = 0)
)]
pub fn _mm_mask_minmax_ss<const IMM8: i32>(
    src: __m128,
    k: __mmask8,
    a: __m128,
    b: __m128,
) -> __m128 {
    _mm_mask_minmax_round_ss::<IMM8, _MM_FROUND_CUR_DIRECTION>(src, k, a, b)
}

/// Performs a min/max comparison between the lower single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set), and copy
/// the upper 3 packed elements from `a` to the upper elements of `dst`.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxss, IMM8 = 0)
)]
pub fn _mm_maskz_minmax_ss<const IMM8: i32>(k: __mmask8, a: __m128, b: __m128) -> __m128 {
    _mm_mask_minmax_ss::<IMM8>(_mm_setzero_ps(), k, a, b)
}

/// Performs a min/max comparison between the lower single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and copy the upper 3 packed elements
/// from `a` to the upper elements of `dst`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxss, IMM8 = 0, SAE = 8)
)]
pub fn _mm_minmax_round_ss<const IMM8: i32, const SAE: i32>(a: __m128, b: __m128) -> __m128 {
    _mm_mask_minmax_round_ss::<IMM8, SAE>(_mm_undefined_ps(), !0, a, b)
}

/// Performs a min/max comparison between the lower single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// writemask `k` (elements are copied from src when the corresponding mask bit is not set), and
/// copy the upper 3 packed elements from `a` to the upper elements of `dst`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxss, IMM8 = 0, SAE = 8)
)]
pub fn _mm_mask_minmax_round_ss<const IMM8: i32, const SAE: i32>(
    src: __m128,
    k: __mmask8,
    a: __m128,
    b: __m128,
) -> __m128 {
    static_assert_uimm_bits!(IMM8, 8);
    static_assert_sae!(SAE);
    unsafe {
        vminmaxss(
            a.as_f32x4(),
            b.as_f32x4(),
            IMM8,
            src.as_f32x4(),
            k as u8,
            SAE,
        )
        .as_m128()
    }
}

/// Performs a min/max comparison between the lower single-precision (32-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set), and copy
/// the upper 3 packed elements from `a` to the upper elements of `dst`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxss, IMM8 = 0, SAE = 8)
)]
pub fn _mm_maskz_minmax_round_ss<const IMM8: i32, const SAE: i32>(
    k: __mmask8,
    a: __m128,
    b: __m128,
) -> __m128 {
    _mm_mask_minmax_round_ss::<IMM8, SAE>(_mm_setzero_ps(), k, a, b)
}

/// Performs a min/max comparison between packed half-precision (16-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and copy the upper 7 packed
/// elements from `a` to the upper elements of `dst`.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxsh, IMM8 = 0)
)]
pub fn _mm_minmax_sh<const IMM8: i32>(a: __m128h, b: __m128h) -> __m128h {
    _mm_mask_minmax_sh::<IMM8>(_mm_undefined_ph(), !0, a, b)
}

/// Performs a min/max comparison between packed half-precision (16-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// writemask `k` (elements are copied from src when the corresponding mask bit is not set), and
/// copy the upper 7 packed elements from `a` to the upper elements of `dst`.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxsh, IMM8 = 0)
)]
pub fn _mm_mask_minmax_sh<const IMM8: i32>(
    src: __m128h,
    k: __mmask8,
    a: __m128h,
    b: __m128h,
) -> __m128h {
    _mm_mask_minmax_round_sh::<IMM8, _MM_FROUND_CUR_DIRECTION>(src, k, a, b)
}

/// Performs a min/max comparison between packed half-precision (16-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set), and copy
/// the upper 7 packed elements from `a` to the upper elements of `dst`.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxsh, IMM8 = 0)
)]
pub fn _mm_maskz_minmax_sh<const IMM8: i32>(k: __mmask8, a: __m128h, b: __m128h) -> __m128h {
    _mm_mask_minmax_sh::<IMM8>(_mm_setzero_ph(), k, a, b)
}

/// Performs a min/max comparison between packed half-precision (16-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and copy the upper 7 packed
/// elements from `a` to the upper elements of `dst`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxsh, IMM8 = 0, SAE = 8)
)]
pub fn _mm_minmax_round_sh<const IMM8: i32, const SAE: i32>(a: __m128h, b: __m128h) -> __m128h {
    _mm_mask_minmax_round_sh::<IMM8, SAE>(_mm_undefined_ph(), !0, a, b)
}

/// Performs a min/max comparison between packed half-precision (16-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// writemask `k` (elements are copied from src when the corresponding mask bit is not set), and
/// copy the upper 7 packed elements from `a` to the upper elements of `dst`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxsh, IMM8 = 0, SAE = 8)
)]
pub fn _mm_mask_minmax_round_sh<const IMM8: i32, const SAE: i32>(
    src: __m128h,
    k: __mmask8,
    a: __m128h,
    b: __m128h,
) -> __m128h {
    static_assert_uimm_bits!(IMM8, 8);
    static_assert_sae!(SAE);
    unsafe {
        vminmaxsh(
            a.as_f16x8(),
            b.as_f16x8(),
            IMM8,
            src.as_f16x8(),
            k as u8,
            SAE,
        )
        .as_m128h()
    }
}

/// Performs a min/max comparison between packed half-precision (16-bit) floating-point
/// elements in `a` and `b` based on the control in `IMM8`, and stores the results in `dst` using
/// zeromask `k` (elements are zeroed out when the corresponding mask bit is not set), and copy
/// the upper 7 packed elements from `a` to the upper elements of `dst`.
/// Exceptions can be suppressed by passing [`_MM_FROUND_NO_EXC`]` in the SAE parameter.
/// For more details, see [`_mm_minmax_pd`].
#[inline]
#[target_feature(enable = "avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(
    all(test, not(target_vendor = "apple")),
    assert_instr(vminmaxsh, IMM8 = 0, SAE = 8)
)]
pub fn _mm_maskz_minmax_round_sh<const IMM8: i32, const SAE: i32>(
    k: __mmask8,
    a: __m128h,
    b: __m128h,
) -> __m128h {
    _mm_mask_minmax_round_sh::<IMM8, SAE>(_mm_setzero_ph(), k, a, b)
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

    #[link_name = "llvm.x86.avx10.vcvttss2sis"]
    fn vcvttss2sis(a: f32x4, sae: i32) -> i32;
    #[link_name = "llvm.x86.avx10.vcvttss2usis"]
    fn vcvttss2usis(a: f32x4, sae: i32) -> u32;

    #[link_name = "llvm.x86.avx10.vcvttsd2sis"]
    fn vcvttsd2sis(a: f64x2, sae: i32) -> i32;
    #[link_name = "llvm.x86.avx10.vcvttsd2usis"]
    fn vcvttsd2usis(a: f64x2, sae: i32) -> u32;

    #[link_name = "llvm.x86.avx10.mask.vcvttpd2dqs.128"]
    fn vcvttpd2dqs_128(a: f64x2, src: i32x4, mask: u8) -> i32x4;
    #[link_name = "llvm.x86.avx10.mask.vcvttpd2dqs.256"]
    fn vcvttpd2dqs_256(a: f64x4, src: i32x4, mask: u8) -> i32x4;
    #[link_name = "llvm.x86.avx10.mask.vcvttpd2dqs.round.512"]
    fn vcvttpd2dqs_512(a: f64x8, src: i32x8, mask: u8, sae: i32) -> i32x8;

    #[link_name = "llvm.x86.avx10.mask.vcvttpd2udqs.128"]
    fn vcvttpd2udqs_128(a: f64x2, src: u32x4, mask: u8) -> u32x4;
    #[link_name = "llvm.x86.avx10.mask.vcvttpd2udqs.256"]
    fn vcvttpd2udqs_256(a: f64x4, src: u32x4, mask: u8) -> u32x4;
    #[link_name = "llvm.x86.avx10.mask.vcvttpd2udqs.round.512"]
    fn vcvttpd2udqs_512(a: f64x8, src: u32x8, mask: u8, sae: i32) -> u32x8;

    #[link_name = "llvm.x86.avx10.mask.vcvttpd2qqs.128"]
    fn vcvttpd2qqs_128(a: f64x2, src: i64x2, mask: u8) -> i64x2;
    #[link_name = "llvm.x86.avx10.mask.vcvttpd2qqs.256"]
    fn vcvttpd2qqs_256(a: f64x4, src: i64x4, mask: u8) -> i64x4;
    #[link_name = "llvm.x86.avx10.mask.vcvttpd2qqs.round.512"]
    fn vcvttpd2qqs_512(a: f64x8, src: i64x8, mask: u8, sae: i32) -> i64x8;

    #[link_name = "llvm.x86.avx10.mask.vcvttpd2uqqs.128"]
    fn vcvttpd2uqqs_128(a: f64x2, src: u64x2, mask: u8) -> u64x2;
    #[link_name = "llvm.x86.avx10.mask.vcvttpd2uqqs.256"]
    fn vcvttpd2uqqs_256(a: f64x4, src: u64x4, mask: u8) -> u64x4;
    #[link_name = "llvm.x86.avx10.mask.vcvttpd2uqqs.round.512"]
    fn vcvttpd2uqqs_512(a: f64x8, src: u64x8, mask: u8, sae: i32) -> u64x8;

    #[link_name = "llvm.x86.avx10.mask.vcvttps2dqs.128"]
    fn vcvttps2dqs_128(a: f32x4, src: i32x4, mask: u8) -> i32x4;
    #[link_name = "llvm.x86.avx10.mask.vcvttps2dqs.256"]
    fn vcvttps2dqs_256(a: f32x8, src: i32x8, mask: u8) -> i32x8;
    #[link_name = "llvm.x86.avx10.mask.vcvttps2dqs.round.512"]
    fn vcvttps2dqs_512(a: f32x16, src: i32x16, mask: u16, sae: i32) -> i32x16;

    #[link_name = "llvm.x86.avx10.mask.vcvttps2udqs.128"]
    fn vcvttps2udqs_128(a: f32x4, src: u32x4, mask: u8) -> u32x4;
    #[link_name = "llvm.x86.avx10.mask.vcvttps2udqs.256"]
    fn vcvttps2udqs_256(a: f32x8, src: u32x8, mask: u8) -> u32x8;
    #[link_name = "llvm.x86.avx10.mask.vcvttps2udqs.round.512"]
    fn vcvttps2udqs_512(a: f32x16, src: u32x16, mask: u16, sae: i32) -> u32x16;

    #[link_name = "llvm.x86.avx10.mask.vcvttps2qqs.128"]
    fn vcvttps2qqs_128(a: f32x4, src: i64x2, mask: u8) -> i64x2;
    #[link_name = "llvm.x86.avx10.mask.vcvttps2qqs.256"]
    fn vcvttps2qqs_256(a: f32x4, src: i64x4, mask: u8) -> i64x4;
    #[link_name = "llvm.x86.avx10.mask.vcvttps2qqs.round.512"]
    fn vcvttps2qqs_512(a: f32x8, src: i64x8, mask: u8, sae: i32) -> i64x8;

    #[link_name = "llvm.x86.avx10.mask.vcvttps2uqqs.128"]
    fn vcvttps2uqqs_128(a: f32x4, src: u64x2, mask: u8) -> u64x2;
    #[link_name = "llvm.x86.avx10.mask.vcvttps2uqqs.256"]
    fn vcvttps2uqqs_256(a: f32x4, src: u64x4, mask: u8) -> u64x4;
    #[link_name = "llvm.x86.avx10.mask.vcvttps2uqqs.round.512"]
    fn vcvttps2uqqs_512(a: f32x8, src: u64x8, mask: u8, sae: i32) -> u64x8;

    #[link_name = "llvm.x86.avx10.mask.vminmaxpd128"]
    fn vminmaxpd128(a: f64x2, b: f64x2, imm8: i32, src: f64x2, k: u8) -> f64x2;
    #[link_name = "llvm.x86.avx10.mask.vminmaxpd256"]
    fn vminmaxpd256(a: f64x4, b: f64x4, imm8: i32, src: f64x4, k: u8) -> f64x4;
    #[link_name = "llvm.x86.avx10.mask.vminmaxpd.round"]
    fn vminmaxpd512(a: f64x8, b: f64x8, imm8: i32, src: f64x8, k: u8, sae: i32) -> f64x8;

    #[link_name = "llvm.x86.avx10.mask.vminmaxps128"]
    fn vminmaxps128(a: f32x4, b: f32x4, imm8: i32, src: f32x4, k: u8) -> f32x4;
    #[link_name = "llvm.x86.avx10.mask.vminmaxps256"]
    fn vminmaxps256(a: f32x8, b: f32x8, imm8: i32, src: f32x8, k: u8) -> f32x8;
    #[link_name = "llvm.x86.avx10.mask.vminmaxps.round"]
    fn vminmaxps512(a: f32x16, b: f32x16, imm8: i32, src: f32x16, k: u16, sae: i32) -> f32x16;

    #[link_name = "llvm.x86.avx10.mask.vminmaxph128"]
    fn vminmaxph128(a: f16x8, b: f16x8, imm8: i32, src: f16x8, k: u8) -> f16x8;
    #[link_name = "llvm.x86.avx10.mask.vminmaxph256"]
    fn vminmaxph256(a: f16x16, b: f16x16, imm8: i32, src: f16x16, k: u16) -> f16x16;
    #[link_name = "llvm.x86.avx10.mask.vminmaxph.round"]
    fn vminmaxph512(a: f16x32, b: f16x32, imm8: i32, src: f16x32, k: u32, sae: i32) -> f16x32;

    #[link_name = "llvm.x86.avx10.mask.vminmaxsd.round"]
    fn vminmaxsd(a: f64x2, b: f64x2, imm8: i32, src: f64x2, k: u8, sae: i32) -> f64x2;
    #[link_name = "llvm.x86.avx10.mask.vminmaxss.round"]
    fn vminmaxss(a: f32x4, b: f32x4, imm8: i32, src: f32x4, k: u8, sae: i32) -> f32x4;
    #[link_name = "llvm.x86.avx10.mask.vminmaxsh.round"]
    fn vminmaxsh(a: f16x8, b: f16x8, imm8: i32, src: f16x8, k: u8, sae: i32) -> f16x8;
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

    #[simd_test(enable = "avx10.2,avxvnniint8")]
    fn test_mm_mask_dpbssd_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_mask_dpbssd_epi32(src, k, a, b);
        let e = _mm_setr_epi32(801, 2, 803, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint8")]
    fn test_mm_maskz_dpbssd_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_maskz_dpbssd_epi32(k, src, a, b);
        let e = _mm_setr_epi32(801, 0, 803, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint8")]
    fn test_mm256_mask_dpbssd_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpbssd_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(801, 2, 803, 4, 805, 6, 807, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint8")]
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

    #[simd_test(enable = "avx10.2,avxvnniint8")]
    fn test_mm_mask_dpbssds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_mask_dpbssds_epi32(src, k, a, b);
        let e = _mm_setr_epi32(801, 2, 803, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint8")]
    fn test_mm_maskz_dpbssds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_maskz_dpbssds_epi32(k, src, a, b);
        let e = _mm_setr_epi32(801, 0, 803, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint8")]
    fn test_mm256_mask_dpbssds_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpbssds_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(801, 2, 803, 4, 805, 6, 807, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint8")]
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

    #[simd_test(enable = "avx10.2,avxvnniint8")]
    fn test_mm_mask_dpbsud_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_mask_dpbsud_epi32(src, k, a, b);
        let e = _mm_setr_epi32(801, 2, 803, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint8")]
    fn test_mm_maskz_dpbsud_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_maskz_dpbsud_epi32(k, src, a, b);
        let e = _mm_setr_epi32(801, 0, 803, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint8")]
    fn test_mm256_mask_dpbsud_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpbsud_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(801, 2, 803, 4, 805, 6, 807, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint8")]
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

    #[simd_test(enable = "avx10.2,avxvnniint8")]
    fn test_mm_mask_dpbsuds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_mask_dpbsuds_epi32(src, k, a, b);
        let e = _mm_setr_epi32(801, 2, 803, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint8")]
    fn test_mm_maskz_dpbsuds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_maskz_dpbsuds_epi32(k, src, a, b);
        let e = _mm_setr_epi32(801, 0, 803, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint8")]
    fn test_mm256_mask_dpbsuds_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpbsuds_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(801, 2, 803, 4, 805, 6, 807, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint8")]
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

    #[simd_test(enable = "avx10.2,avxvnniint8")]
    fn test_mm_mask_dpbuud_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_mask_dpbuud_epi32(src, k, a, b);
        let e = _mm_setr_epi32(801, 2, 803, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint8")]
    fn test_mm_maskz_dpbuud_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_maskz_dpbuud_epi32(k, src, a, b);
        let e = _mm_setr_epi32(801, 0, 803, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint8")]
    fn test_mm256_mask_dpbuud_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpbuud_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(801, 2, 803, 4, 805, 6, 807, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint8")]
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

    #[simd_test(enable = "avx10.2,avxvnniint8")]
    fn test_mm_mask_dpbuuds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_mask_dpbuuds_epi32(src, k, a, b);
        let e = _mm_setr_epi32(801, 2, 803, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint8")]
    fn test_mm_maskz_dpbuuds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi8(10);
        let b = _mm_set1_epi8(20);
        let k = 0b0101;
        let r = _mm_maskz_dpbuuds_epi32(k, src, a, b);
        let e = _mm_setr_epi32(801, 0, 803, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint8")]
    fn test_mm256_mask_dpbuuds_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi8(10);
        let b = _mm256_set1_epi8(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpbuuds_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(801, 2, 803, 4, 805, 6, 807, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint8")]
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

    #[simd_test(enable = "avx10.2,avxvnniint16")]
    fn test_mm_mask_dpwsud_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_mask_dpwsud_epi32(src, k, a, b);
        let e = _mm_setr_epi32(401, 2, 403, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint16")]
    fn test_mm_maskz_dpwsud_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_maskz_dpwsud_epi32(k, src, a, b);
        let e = _mm_setr_epi32(401, 0, 403, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint16")]
    fn test_mm256_mask_dpwsud_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi16(10);
        let b = _mm256_set1_epi16(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpwsud_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(401, 2, 403, 4, 405, 6, 407, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint16")]
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

    #[simd_test(enable = "avx10.2,avxvnniint16")]
    fn test_mm_mask_dpwsuds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_mask_dpwsuds_epi32(src, k, a, b);
        let e = _mm_setr_epi32(401, 2, 403, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint16")]
    fn test_mm_maskz_dpwsuds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_maskz_dpwsuds_epi32(k, src, a, b);
        let e = _mm_setr_epi32(401, 0, 403, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint16")]
    fn test_mm256_mask_dpwsuds_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi16(10);
        let b = _mm256_set1_epi16(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpwsuds_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(401, 2, 403, 4, 405, 6, 407, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint16")]
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

    #[simd_test(enable = "avx10.2,avxvnniint16")]
    fn test_mm_mask_dpwusd_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_mask_dpwusd_epi32(src, k, a, b);
        let e = _mm_setr_epi32(401, 2, 403, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint16")]
    fn test_mm_maskz_dpwusd_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_maskz_dpwusd_epi32(k, src, a, b);
        let e = _mm_setr_epi32(401, 0, 403, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint16")]
    fn test_mm256_mask_dpwusd_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi16(10);
        let b = _mm256_set1_epi16(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpwusd_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(401, 2, 403, 4, 405, 6, 407, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint16")]
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

    #[simd_test(enable = "avx10.2,avxvnniint16")]
    fn test_mm_mask_dpwusds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_mask_dpwusds_epi32(src, k, a, b);
        let e = _mm_setr_epi32(401, 2, 403, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint16")]
    fn test_mm_maskz_dpwusds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_maskz_dpwusds_epi32(k, src, a, b);
        let e = _mm_setr_epi32(401, 0, 403, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint16")]
    fn test_mm256_mask_dpwusds_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi16(10);
        let b = _mm256_set1_epi16(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpwusds_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(401, 2, 403, 4, 405, 6, 407, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint16")]
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

    #[simd_test(enable = "avx10.2,avxvnniint16")]
    fn test_mm_mask_dpwuud_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_mask_dpwuud_epi32(src, k, a, b);
        let e = _mm_setr_epi32(401, 2, 403, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint16")]
    fn test_mm_maskz_dpwuud_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_maskz_dpwuud_epi32(k, src, a, b);
        let e = _mm_setr_epi32(401, 0, 403, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint16")]
    fn test_mm256_mask_dpwuud_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi16(10);
        let b = _mm256_set1_epi16(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpwuud_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(401, 2, 403, 4, 405, 6, 407, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint16")]
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

    #[simd_test(enable = "avx10.2,avxvnniint16")]
    fn test_mm_mask_dpwuuds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_mask_dpwuuds_epi32(src, k, a, b);
        let e = _mm_setr_epi32(401, 2, 403, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint16")]
    fn test_mm_maskz_dpwuuds_epi32() {
        let src = _mm_setr_epi32(1, 2, 3, 4);
        let a = _mm_set1_epi16(10);
        let b = _mm_set1_epi16(20);
        let k = 0b0101;
        let r = _mm_maskz_dpwuuds_epi32(k, src, a, b);
        let e = _mm_setr_epi32(401, 0, 403, 0);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint16")]
    fn test_mm256_mask_dpwuuds_epi32() {
        let src = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let a = _mm256_set1_epi16(10);
        let b = _mm256_set1_epi16(20);
        let k = 0b01010101;
        let r = _mm256_mask_dpwuuds_epi32(src, k, a, b);
        let e = _mm256_setr_epi32(401, 2, 403, 4, 405, 6, 407, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx10.2,avxvnniint16")]
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

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_roundsd_i32() {
        let a = _mm_set_sd(2.0);
        let r = _mm_cvtts_roundsd_i32::<_MM_FROUND_NO_EXC>(a);
        assert_eq!(r, 2i32);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_roundsd_si32() {
        let a = _mm_set_sd(3.7);
        let r = _mm_cvtts_roundsd_si32::<_MM_FROUND_NO_EXC>(a);
        assert_eq!(r, 3i32);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_roundsd_u32() {
        let a = _mm_set_sd(5.9);
        let r = _mm_cvtts_roundsd_u32::<_MM_FROUND_NO_EXC>(a);
        assert_eq!(r, 5u32);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_roundss_i32() {
        let a = _mm_set_ss(4.2);
        let r = _mm_cvtts_roundss_i32::<_MM_FROUND_NO_EXC>(a);
        assert_eq!(r, 4i32);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_roundss_si32() {
        let a = _mm_set_ss(6.8);
        let r = _mm_cvtts_roundss_si32::<_MM_FROUND_NO_EXC>(a);
        assert_eq!(r, 6i32);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_roundss_u32() {
        let a = _mm_set_ss(7.1);
        let r = _mm_cvtts_roundss_u32::<_MM_FROUND_NO_EXC>(a);
        assert_eq!(r, 7u32);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_pd_epi32() {
        let a = _mm_setr_pd(14.2, 15.8);
        let r = _mm_cvtts_pd_epi32(a);
        let expected = _mm_setr_epi32(14, 15, 0, 0);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_cvtts_pd_epi32() {
        let a = _mm_setr_pd(16.9, 17.1);
        let src = _mm_setr_epi32(100, 200, 0, 0);
        let r = _mm_mask_cvtts_pd_epi32(src, 0b01, a);
        let expected = _mm_setr_epi32(16, 200, 0, 0);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_cvtts_pd_epi32() {
        let a = _mm_setr_pd(18.5, 19.3);
        let r = _mm_maskz_cvtts_pd_epi32(0b10, a);
        let expected = _mm_setr_epi32(0, 19, 0, 0);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_cvtts_pd_epi32() {
        let a = _mm256_setr_pd(20.7, 21.4, 22.9, 23.1);
        let r = _mm256_cvtts_pd_epi32(a);
        let expected = _mm_setr_epi32(20, 21, 22, 23);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_cvtts_pd_epi32() {
        let a = _mm256_setr_pd(24.8, 25.2, 26.6, 27.3);
        let src = _mm_setr_epi32(100, 200, 300, 400);
        let r = _mm256_mask_cvtts_pd_epi32(src, 0b0101, a);
        let expected = _mm_setr_epi32(24, 200, 26, 400);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_cvtts_pd_epi32() {
        let a = _mm256_setr_pd(28.9, 29.1, 30.4, 31.7);
        let r = _mm256_maskz_cvtts_pd_epi32(0b1010, a);
        let expected = _mm_setr_epi32(0, 29, 0, 31);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_cvtts_pd_epi32() {
        let a = _mm512_setr_pd(32.5, 33.2, 34.8, 35.3, 36.6, 37.9, 38.1, 39.4);
        let r = _mm512_cvtts_pd_epi32(a);
        let expected = _mm256_setr_epi32(32, 33, 34, 35, 36, 37, 38, 39);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_cvtts_pd_epi32() {
        let a = _mm512_setr_pd(40.7, 41.3, 42.9, 43.2, 44.5, 45.8, 46.1, 47.6);
        let src = _mm256_setr_epi32(100, 200, 300, 400, 500, 600, 700, 800);
        let r = _mm512_mask_cvtts_pd_epi32(src, 0b10101010, a);
        let expected = _mm256_setr_epi32(100, 41, 300, 43, 500, 45, 700, 47);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_cvtts_pd_epi32() {
        let a = _mm512_setr_pd(48.4, 49.8, 50.3, 51.7, 52.1, 53.9, 54.2, 55.6);
        let r = _mm512_maskz_cvtts_pd_epi32(0b11110000, a);
        let expected = _mm256_setr_epi32(0, 0, 0, 0, 52, 53, 54, 55);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_cvtts_roundpd_epi32() {
        let a = _mm512_setr_pd(56.5, 57.2, 58.8, 59.3, 60.6, 61.9, 62.1, 63.4);
        let r = _mm512_cvtts_roundpd_epi32::<_MM_FROUND_NO_EXC>(a);
        let expected = _mm256_setr_epi32(56, 57, 58, 59, 60, 61, 62, 63);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_cvtts_roundpd_epi32() {
        let a = _mm512_setr_pd(64.7, 65.3, 66.9, 67.2, 68.5, 69.8, 70.1, 71.6);
        let src = _mm256_setr_epi32(100, 200, 300, 400, 500, 600, 700, 800);
        let r = _mm512_mask_cvtts_roundpd_epi32::<_MM_FROUND_NO_EXC>(src, 0b01010101, a);
        let expected = _mm256_setr_epi32(64, 200, 66, 400, 68, 600, 70, 800);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_cvtts_roundpd_epi32() {
        let a = _mm512_setr_pd(72.4, 73.8, 74.3, 75.7, 76.1, 77.9, 78.2, 79.6);
        let r = _mm512_maskz_cvtts_roundpd_epi32::<_MM_FROUND_NO_EXC>(0b00001111, a);
        let expected = _mm256_setr_epi32(72, 73, 74, 75, 0, 0, 0, 0);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_pd_epu32() {
        let a = _mm_setr_pd(14.2, 15.8);
        let r = _mm_cvtts_pd_epu32(a);
        let expected = _mm_setr_epi32(14, 15, 0, 0);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_cvtts_pd_epu32() {
        let a = _mm_setr_pd(16.9, 17.1);
        let src = _mm_setr_epi32(100, 200, 0, 0);
        let r = _mm_mask_cvtts_pd_epu32(src, 0b01, a);
        let expected = _mm_setr_epi32(16, 200, 0, 0);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_cvtts_pd_epu32() {
        let a = _mm_setr_pd(18.5, 19.3);
        let r = _mm_maskz_cvtts_pd_epu32(0b10, a);
        let expected = _mm_setr_epi32(0, 19, 0, 0);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_cvtts_pd_epu32() {
        let a = _mm256_setr_pd(20.7, 21.4, 22.9, 23.1);
        let r = _mm256_cvtts_pd_epu32(a);
        let expected = _mm_setr_epi32(20, 21, 22, 23);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_cvtts_pd_epu32() {
        let a = _mm256_setr_pd(24.8, 25.2, 26.6, 27.3);
        let src = _mm_setr_epi32(100, 200, 300, 400);
        let r = _mm256_mask_cvtts_pd_epu32(src, 0b0101, a);
        let expected = _mm_setr_epi32(24, 200, 26, 400);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_cvtts_pd_epu32() {
        let a = _mm256_setr_pd(28.9, 29.1, 30.4, 31.7);
        let r = _mm256_maskz_cvtts_pd_epu32(0b1010, a);
        let expected = _mm_setr_epi32(0, 29, 0, 31);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_cvtts_pd_epu32() {
        let a = _mm512_setr_pd(32.5, 33.2, 34.8, 35.3, 36.6, 37.9, 38.1, 39.4);
        let r = _mm512_cvtts_pd_epu32(a);
        let expected = _mm256_setr_epi32(32, 33, 34, 35, 36, 37, 38, 39);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_cvtts_pd_epu32() {
        let a = _mm512_setr_pd(40.7, 41.3, 42.9, 43.2, 44.5, 45.8, 46.1, 47.6);
        let src = _mm256_setr_epi32(100, 200, 300, 400, 500, 600, 700, 800);
        let r = _mm512_mask_cvtts_pd_epu32(src, 0b10101010, a);
        let expected = _mm256_setr_epi32(100, 41, 300, 43, 500, 45, 700, 47);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_cvtts_pd_epu32() {
        let a = _mm512_setr_pd(48.4, 49.8, 50.3, 51.7, 52.1, 53.9, 54.2, 55.6);
        let r = _mm512_maskz_cvtts_pd_epu32(0b11110000, a);
        let expected = _mm256_setr_epi32(0, 0, 0, 0, 52, 53, 54, 55);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_cvtts_roundpd_epu32() {
        let a = _mm512_setr_pd(56.5, 57.2, 58.8, 59.3, 60.6, 61.9, 62.1, 63.4);
        let r = _mm512_cvtts_roundpd_epu32::<_MM_FROUND_NO_EXC>(a);
        let expected = _mm256_setr_epi32(56, 57, 58, 59, 60, 61, 62, 63);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_cvtts_roundpd_epu32() {
        let a = _mm512_setr_pd(64.7, 65.3, 66.9, 67.2, 68.5, 69.8, 70.1, 71.6);
        let src = _mm256_setr_epi32(100, 200, 300, 400, 500, 600, 700, 800);
        let r = _mm512_mask_cvtts_roundpd_epu32::<_MM_FROUND_NO_EXC>(src, 0b01010101, a);
        let expected = _mm256_setr_epi32(64, 200, 66, 400, 68, 600, 70, 800);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_cvtts_roundpd_epu32() {
        let a = _mm512_setr_pd(72.4, 73.8, 74.3, 75.7, 76.1, 77.9, 78.2, 79.6);
        let r = _mm512_maskz_cvtts_roundpd_epu32::<_MM_FROUND_NO_EXC>(0b00001111, a);
        let expected = _mm256_setr_epi32(72, 73, 74, 75, 0, 0, 0, 0);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_pd_epi64() {
        let a = _mm_setr_pd(80.5, 81.9);
        let r = _mm_cvtts_pd_epi64(a);
        let expected = _mm_setr_epi64x(80, 81);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_cvtts_pd_epi64() {
        let a = _mm_setr_pd(82.3, 83.7);
        let src = _mm_setr_epi64x(100, 200);
        let r = _mm_mask_cvtts_pd_epi64(src, 0b01, a);
        let expected = _mm_setr_epi64x(82, 200);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_cvtts_pd_epi64() {
        let a = _mm_setr_pd(84.8, 85.2);
        let r = _mm_maskz_cvtts_pd_epi64(0b10, a);
        let expected = _mm_setr_epi64x(0, 85);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_cvtts_pd_epi64() {
        let a = _mm256_setr_pd(86.4, 87.6, 88.1, 89.9);
        let r = _mm256_cvtts_pd_epi64(a);
        let expected = _mm256_setr_epi64x(86, 87, 88, 89);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_cvtts_pd_epi64() {
        let a = _mm256_setr_pd(90.7, 91.3, 92.8, 93.2);
        let src = _mm256_setr_epi64x(100, 200, 300, 400);
        let r = _mm256_mask_cvtts_pd_epi64(src, 0b0101, a);
        let expected = _mm256_setr_epi64x(90, 200, 92, 400);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_cvtts_pd_epi64() {
        let a = _mm256_setr_pd(94.5, 95.1, 96.9, 97.4);
        let r = _mm256_maskz_cvtts_pd_epi64(0b1010, a);
        let expected = _mm256_setr_epi64x(0, 95, 0, 97);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_cvtts_pd_epi64() {
        let a = _mm512_setr_pd(98.6, 99.2, 100.8, 101.3, 102.7, 103.9, 104.1, 105.5);
        let r = _mm512_cvtts_pd_epi64(a);
        let expected = _mm512_setr_epi64(98, 99, 100, 101, 102, 103, 104, 105);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_cvtts_pd_epi64() {
        let a = _mm512_setr_pd(106.4, 107.8, 108.2, 109.6, 110.3, 111.7, 112.9, 113.1);
        let src = _mm512_setr_epi64(100, 200, 300, 400, 500, 600, 700, 800);
        let r = _mm512_mask_cvtts_pd_epi64(src, 0b10101010, a);
        let expected = _mm512_setr_epi64(100, 107, 300, 109, 500, 111, 700, 113);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_cvtts_pd_epi64() {
        let a = _mm512_setr_pd(114.5, 115.8, 116.3, 117.7, 118.2, 119.9, 120.4, 121.6);
        let r = _mm512_maskz_cvtts_pd_epi64(0b11110000, a);
        let expected = _mm512_setr_epi64(0, 0, 0, 0, 118, 119, 120, 121);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_cvtts_roundpd_epi64() {
        let a = _mm512_setr_pd(122.7, 123.3, 124.9, 125.1, 126.5, 127.8, 128.2, 129.6);
        let r = _mm512_cvtts_roundpd_epi64::<_MM_FROUND_NO_EXC>(a);
        let expected = _mm512_setr_epi64(122, 123, 124, 125, 126, 127, 128, 129);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_cvtts_roundpd_epi64() {
        let a = _mm512_setr_pd(130.4, 131.8, 132.3, 133.7, 134.1, 135.9, 136.5, 137.2);
        let src = _mm512_setr_epi64(100, 200, 300, 400, 500, 600, 700, 800);
        let r = _mm512_mask_cvtts_roundpd_epi64::<_MM_FROUND_NO_EXC>(src, 0b01010101, a);
        let expected = _mm512_setr_epi64(130, 200, 132, 400, 134, 600, 136, 800);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_cvtts_roundpd_epi64() {
        let a = _mm512_setr_pd(138.6, 139.4, 140.8, 141.2, 142.7, 143.9, 144.3, 145.5);
        let r = _mm512_maskz_cvtts_roundpd_epi64::<_MM_FROUND_NO_EXC>(0b00001111, a);
        let expected = _mm512_setr_epi64(138, 139, 140, 141, 0, 0, 0, 0);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_pd_epu64() {
        let a = _mm_setr_pd(146.7, 147.9);
        let r = _mm_cvtts_pd_epu64(a);
        let expected = _mm_setr_epi64x(146, 147);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_cvtts_pd_epu64() {
        let a = _mm_setr_pd(148.3, 149.6);
        let src = _mm_setr_epi64x(100, 200);
        let r = _mm_mask_cvtts_pd_epu64(src, 0b01, a);
        let expected = _mm_setr_epi64x(148, 200);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_cvtts_pd_epu64() {
        let a = _mm_setr_pd(150.8, 151.2);
        let r = _mm_maskz_cvtts_pd_epu64(0b10, a);
        let expected = _mm_setr_epi64x(0, 151);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_cvtts_pd_epu64() {
        let a = _mm256_setr_pd(152.4, 153.7, 154.1, 155.9);
        let r = _mm256_cvtts_pd_epu64(a);
        let expected = _mm256_setr_epi64x(152, 153, 154, 155);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_cvtts_pd_epu64() {
        let a = _mm256_setr_pd(156.5, 157.3, 158.8, 159.2);
        let src = _mm256_setr_epi64x(100, 200, 300, 400);
        let r = _mm256_mask_cvtts_pd_epu64(src, 0b0101, a);
        let expected = _mm256_setr_epi64x(156, 200, 158, 400);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_cvtts_pd_epu64() {
        let a = _mm256_setr_pd(160.6, 161.1, 162.9, 163.4);
        let r = _mm256_maskz_cvtts_pd_epu64(0b1010, a);
        let expected = _mm256_setr_epi64x(0, 161, 0, 163);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_cvtts_pd_epu64() {
        let a = _mm512_setr_pd(164.7, 165.2, 166.8, 167.3, 168.5, 169.9, 170.1, 171.6);
        let r = _mm512_cvtts_pd_epu64(a);
        let expected = _mm512_setr_epi64(164, 165, 166, 167, 168, 169, 170, 171);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_cvtts_pd_epu64() {
        let a = _mm512_setr_pd(172.4, 173.8, 174.2, 175.6, 176.3, 177.7, 178.9, 179.1);
        let src = _mm512_setr_epi64(100, 200, 300, 400, 500, 600, 700, 800);
        let r = _mm512_mask_cvtts_pd_epu64(src, 0b10101010, a);
        let expected = _mm512_setr_epi64(100, 173, 300, 175, 500, 177, 700, 179);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_cvtts_pd_epu64() {
        let a = _mm512_setr_pd(180.5, 181.8, 182.3, 183.7, 184.2, 185.9, 186.4, 187.6);
        let r = _mm512_maskz_cvtts_pd_epu64(0b11110000, a);
        let expected = _mm512_setr_epi64(0, 0, 0, 0, 184, 185, 186, 187);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_cvtts_roundpd_epu64() {
        let a = _mm512_setr_pd(188.7, 189.3, 190.9, 191.1, 192.5, 193.8, 194.2, 195.6);
        let r = _mm512_cvtts_roundpd_epu64::<_MM_FROUND_NO_EXC>(a);
        let expected = _mm512_setr_epi64(188, 189, 190, 191, 192, 193, 194, 195);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_cvtts_roundpd_epu64() {
        let a = _mm512_setr_pd(196.4, 197.8, 198.3, 199.7, 200.1, 201.9, 202.5, 203.2);
        let src = _mm512_setr_epi64(100, 200, 300, 400, 500, 600, 700, 800);
        let r = _mm512_mask_cvtts_roundpd_epu64::<_MM_FROUND_NO_EXC>(src, 0b01010101, a);
        let expected = _mm512_setr_epi64(196, 200, 198, 400, 200, 600, 202, 800);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_cvtts_roundpd_epu64() {
        let a = _mm512_setr_pd(204.6, 205.4, 206.8, 207.2, 208.7, 209.9, 210.3, 211.5);
        let r = _mm512_maskz_cvtts_roundpd_epu64::<_MM_FROUND_NO_EXC>(0b00001111, a);
        let expected = _mm512_setr_epi64(204, 205, 206, 207, 0, 0, 0, 0);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_ps_epi32() {
        let a = _mm_setr_ps(212.3, 213.7, 214.9, 215.2);
        let r = _mm_cvtts_ps_epi32(a);
        let expected = _mm_setr_epi32(212, 213, 214, 215);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_cvtts_ps_epi32() {
        let a = _mm_setr_ps(216.5, 217.8, 218.1, 219.6);
        let src = _mm_setr_epi32(100, 200, 300, 400);
        let r = _mm_mask_cvtts_ps_epi32(src, 0b0101, a);
        let expected = _mm_setr_epi32(216, 200, 218, 400);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_cvtts_ps_epi32() {
        let a = _mm_setr_ps(220.4, 221.9, 222.3, 223.7);
        let r = _mm_maskz_cvtts_ps_epi32(0b1010, a);
        let expected = _mm_setr_epi32(0, 221, 0, 223);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_cvtts_ps_epi32() {
        let a = _mm256_setr_ps(224.6, 225.2, 226.8, 227.3, 228.5, 229.9, 230.1, 231.7);
        let r = _mm256_cvtts_ps_epi32(a);
        let expected = _mm256_setr_epi32(224, 225, 226, 227, 228, 229, 230, 231);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_cvtts_ps_epi32() {
        let a = _mm256_setr_ps(232.4, 233.8, 234.2, 235.6, 236.3, 237.7, 238.9, 239.1);
        let src = _mm256_setr_epi32(100, 200, 300, 400, 500, 600, 700, 800);
        let r = _mm256_mask_cvtts_ps_epi32(src, 0b10101010, a);
        let expected = _mm256_setr_epi32(100, 233, 300, 235, 500, 237, 700, 239);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_cvtts_ps_epi32() {
        let a = _mm256_setr_ps(240.5, 241.8, 242.3, 243.7, 244.2, 245.9, 246.4, 247.6);
        let r = _mm256_maskz_cvtts_ps_epi32(0b11110000, a);
        let expected = _mm256_setr_epi32(0, 0, 0, 0, 244, 245, 246, 247);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_cvtts_ps_epi32() {
        let a = _mm512_setr_ps(
            248.7, 249.3, 250.9, 251.1, 252.5, 253.8, 254.2, 255.6, 256.4, 257.8, 258.3, 259.7,
            260.1, 261.9, 262.5, 263.2,
        );
        let r = _mm512_cvtts_ps_epi32(a);
        let expected = _mm512_setr_epi32(
            248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263,
        );
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_cvtts_ps_epi32() {
        let a = _mm512_setr_ps(
            264.6, 265.4, 266.8, 267.2, 268.7, 269.9, 270.3, 271.5, 272.1, 273.6, 274.9, 275.3,
            276.8, 277.2, 278.5, 279.7,
        );
        let src = _mm512_setr_epi32(
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
        );
        let r = _mm512_mask_cvtts_ps_epi32(src, 0b1010101010101010, a);
        let expected = _mm512_setr_epi32(
            100, 265, 300, 267, 500, 269, 700, 271, 900, 273, 1100, 275, 1300, 277, 1500, 279,
        );
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_cvtts_ps_epi32() {
        let a = _mm512_setr_ps(
            280.4, 281.9, 282.3, 283.7, 284.1, 285.8, 286.5, 287.2, 288.6, 289.4, 290.8, 291.3,
            292.7, 293.1, 294.9, 295.5,
        );
        let r = _mm512_maskz_cvtts_ps_epi32(0b1111111100000000, a);
        let expected = _mm512_setr_epi32(
            0, 0, 0, 0, 0, 0, 0, 0, 288, 289, 290, 291, 292, 293, 294, 295,
        );
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_cvtts_roundps_epi32() {
        let a = _mm512_setr_ps(
            296.7, 297.3, 298.9, 299.1, 300.5, 301.8, 302.2, 303.6, 304.4, 305.8, 306.3, 307.7,
            308.1, 309.9, 310.5, 311.2,
        );
        let r = _mm512_cvtts_roundps_epi32::<_MM_FROUND_NO_EXC>(a);
        let expected = _mm512_setr_epi32(
            296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311,
        );
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_cvtts_roundps_epi32() {
        let a = _mm512_setr_ps(
            312.6, 313.4, 314.8, 315.2, 316.7, 317.9, 318.3, 319.5, 320.1, 321.6, 322.9, 323.3,
            324.8, 325.2, 326.5, 327.7,
        );
        let src = _mm512_setr_epi32(
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
        );
        let r = _mm512_mask_cvtts_roundps_epi32::<_MM_FROUND_NO_EXC>(src, 0b0101010101010101, a);
        let expected = _mm512_setr_epi32(
            312, 200, 314, 400, 316, 600, 318, 800, 320, 1000, 322, 1200, 324, 1400, 326, 1600,
        );
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_cvtts_roundps_epi32() {
        let a = _mm512_setr_ps(
            328.4, 329.9, 330.3, 331.7, 332.1, 333.8, 334.5, 335.2, 336.6, 337.4, 338.8, 339.3,
            340.7, 341.1, 342.9, 343.5,
        );
        let r = _mm512_maskz_cvtts_roundps_epi32::<_MM_FROUND_NO_EXC>(0b0000000011111111, a);
        let expected = _mm512_setr_epi32(
            328, 329, 330, 331, 332, 333, 334, 335, 0, 0, 0, 0, 0, 0, 0, 0,
        );
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_ps_epu32() {
        let a = _mm_setr_ps(344.7, 345.3, 346.9, 347.1);
        let r = _mm_cvtts_ps_epu32(a);
        let expected = _mm_setr_epi32(344, 345, 346, 347);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_cvtts_ps_epu32() {
        let a = _mm_setr_ps(348.5, 349.8, 350.2, 351.6);
        let src = _mm_setr_epi32(100, 200, 300, 400);
        let r = _mm_mask_cvtts_ps_epu32(src, 0b0101, a);
        let expected = _mm_setr_epi32(348, 200, 350, 400);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_cvtts_ps_epu32() {
        let a = _mm_setr_ps(352.4, 353.9, 354.3, 355.7);
        let r = _mm_maskz_cvtts_ps_epu32(0b1010, a);
        let expected = _mm_setr_epi32(0, 353, 0, 355);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_cvtts_ps_epu32() {
        let a = _mm256_setr_ps(356.6, 357.2, 358.8, 359.3, 360.5, 361.9, 362.1, 363.7);
        let r = _mm256_cvtts_ps_epu32(a);
        let expected = _mm256_setr_epi32(356, 357, 358, 359, 360, 361, 362, 363);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_cvtts_ps_epu32() {
        let a = _mm256_setr_ps(364.4, 365.8, 366.2, 367.6, 368.3, 369.7, 370.9, 371.1);
        let src = _mm256_setr_epi32(100, 200, 300, 400, 500, 600, 700, 800);
        let r = _mm256_mask_cvtts_ps_epu32(src, 0b10101010, a);
        let expected = _mm256_setr_epi32(100, 365, 300, 367, 500, 369, 700, 371);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_cvtts_ps_epu32() {
        let a = _mm256_setr_ps(372.5, 373.8, 374.3, 375.7, 376.2, 377.9, 378.4, 379.6);
        let r = _mm256_maskz_cvtts_ps_epu32(0b11110000, a);
        let expected = _mm256_setr_epi32(0, 0, 0, 0, 376, 377, 378, 379);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_cvtts_ps_epu32() {
        let a = _mm512_setr_ps(
            380.7, 381.3, 382.9, 383.1, 384.5, 385.8, 386.2, 387.6, 388.4, 389.8, 390.3, 391.7,
            392.1, 393.9, 394.5, 395.2,
        );
        let r = _mm512_cvtts_ps_epu32(a);
        let expected = _mm512_setr_epi32(
            380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395,
        );
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_cvtts_ps_epu32() {
        let a = _mm512_setr_ps(
            396.6, 397.4, 398.8, 399.2, 400.7, 401.9, 402.3, 403.5, 404.1, 405.6, 406.9, 407.3,
            408.8, 409.2, 410.5, 411.7,
        );
        let src = _mm512_setr_epi32(
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
        );
        let r = _mm512_mask_cvtts_ps_epu32(src, 0b1010101010101010, a);
        let expected = _mm512_setr_epi32(
            100, 397, 300, 399, 500, 401, 700, 403, 900, 405, 1100, 407, 1300, 409, 1500, 411,
        );
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_cvtts_ps_epu32() {
        let a = _mm512_setr_ps(
            412.4, 413.9, 414.3, 415.7, 416.1, 417.8, 418.5, 419.2, 420.6, 421.4, 422.8, 423.3,
            424.7, 425.1, 426.9, 427.5,
        );
        let r = _mm512_maskz_cvtts_ps_epu32(0b1111111100000000, a);
        let expected = _mm512_setr_epi32(
            0, 0, 0, 0, 0, 0, 0, 0, 420, 421, 422, 423, 424, 425, 426, 427,
        );
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_cvtts_roundps_epu32() {
        let a = _mm512_setr_ps(
            428.7, 429.3, 430.9, 431.1, 432.5, 433.8, 434.2, 435.6, 436.4, 437.8, 438.3, 439.7,
            440.1, 441.9, 442.5, 443.2,
        );
        let r = _mm512_cvtts_roundps_epu32::<_MM_FROUND_NO_EXC>(a);
        let expected = _mm512_setr_epi32(
            428, 429, 430, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 441, 442, 443,
        );
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_cvtts_roundps_epu32() {
        let a = _mm512_setr_ps(
            444.6, 445.4, 446.8, 447.2, 448.7, 449.9, 450.3, 451.5, 452.1, 453.6, 454.9, 455.3,
            456.8, 457.2, 458.5, 459.7,
        );
        let src = _mm512_setr_epi32(
            100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
        );
        let r = _mm512_mask_cvtts_roundps_epu32::<_MM_FROUND_NO_EXC>(src, 0b0101010101010101, a);
        let expected = _mm512_setr_epi32(
            444, 200, 446, 400, 448, 600, 450, 800, 452, 1000, 454, 1200, 456, 1400, 458, 1600,
        );
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_cvtts_roundps_epu32() {
        let a = _mm512_setr_ps(
            460.4, 461.9, 462.3, 463.7, 464.1, 465.8, 466.5, 467.2, 468.6, 469.4, 470.8, 471.3,
            472.7, 473.1, 474.9, 475.5,
        );
        let r = _mm512_maskz_cvtts_roundps_epu32::<_MM_FROUND_NO_EXC>(0b0000000011111111, a);
        let expected = _mm512_setr_epi32(
            460, 461, 462, 463, 464, 465, 466, 467, 0, 0, 0, 0, 0, 0, 0, 0,
        );
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_ps_epi64() {
        let a = _mm_setr_ps(476.7, 477.3, 478.9, 479.1);
        let r = _mm_cvtts_ps_epi64(a);
        let expected = _mm_setr_epi64x(476, 477);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_cvtts_ps_epi64() {
        let a = _mm_setr_ps(480.5, 481.8, 482.2, 483.6);
        let src = _mm_setr_epi64x(100, 200);
        let r = _mm_mask_cvtts_ps_epi64(src, 0b01, a);
        let expected = _mm_setr_epi64x(480, 200);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_cvtts_ps_epi64() {
        let a = _mm_setr_ps(484.4, 485.9, 486.3, 487.7);
        let r = _mm_maskz_cvtts_ps_epi64(0b10, a);
        let expected = _mm_setr_epi64x(0, 485);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_cvtts_ps_epi64() {
        let a = _mm_setr_ps(488.6, 489.2, 490.8, 491.3);
        let r = _mm256_cvtts_ps_epi64(a);
        let expected = _mm256_setr_epi64x(488, 489, 490, 491);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_cvtts_ps_epi64() {
        let a = _mm_setr_ps(496.4, 497.8, 498.2, 499.6);
        let src = _mm256_setr_epi64x(100, 200, 300, 400);
        let r = _mm256_mask_cvtts_ps_epi64(src, 0b0101, a);
        let expected = _mm256_setr_epi64x(496, 200, 498, 400);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_cvtts_ps_epi64() {
        let a = _mm_setr_ps(504.5, 505.8, 506.3, 507.7);
        let r = _mm256_maskz_cvtts_ps_epi64(0b1010, a);
        let expected = _mm256_setr_epi64x(0, 505, 0, 507);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_cvtts_ps_epi64() {
        let a = _mm256_setr_ps(512.7, 513.3, 514.9, 515.1, 516.5, 517.8, 518.2, 519.6);
        let r = _mm512_cvtts_ps_epi64(a);
        let expected = _mm512_setr_epi64(512, 513, 514, 515, 516, 517, 518, 519);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_cvtts_ps_epi64() {
        let a = _mm256_setr_ps(520.4, 521.8, 522.3, 523.7, 524.1, 525.9, 526.5, 527.2);
        let src = _mm512_setr_epi64(100, 200, 300, 400, 500, 600, 700, 800);
        let r = _mm512_mask_cvtts_ps_epi64(src, 0b10101010, a);
        let expected = _mm512_setr_epi64(100, 521, 300, 523, 500, 525, 700, 527);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_cvtts_ps_epi64() {
        let a = _mm256_setr_ps(528.6, 529.4, 530.8, 531.2, 532.7, 533.9, 534.3, 535.5);
        let r = _mm512_maskz_cvtts_ps_epi64(0b11110000, a);
        let expected = _mm512_setr_epi64(0, 0, 0, 0, 532, 533, 534, 535);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_cvtts_roundps_epi64() {
        let a = _mm256_setr_ps(536.7, 537.3, 538.9, 539.1, 540.5, 541.8, 542.2, 543.6);
        let r = _mm512_cvtts_roundps_epi64::<_MM_FROUND_NO_EXC>(a);
        let expected = _mm512_setr_epi64(536, 537, 538, 539, 540, 541, 542, 543);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_cvtts_roundps_epi64() {
        let a = _mm256_setr_ps(544.4, 545.8, 546.3, 547.7, 548.1, 549.9, 550.5, 551.2);
        let src = _mm512_setr_epi64(100, 200, 300, 400, 500, 600, 700, 800);
        let r = _mm512_mask_cvtts_roundps_epi64::<_MM_FROUND_NO_EXC>(src, 0b01010101, a);
        let expected = _mm512_setr_epi64(544, 200, 546, 400, 548, 600, 550, 800);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_cvtts_roundps_epi64() {
        let a = _mm256_setr_ps(552.6, 553.4, 554.8, 555.3, 556.7, 557.9, 558.2, 559.5);
        let r = _mm512_maskz_cvtts_roundps_epi64::<_MM_FROUND_NO_EXC>(0b00001111, a);
        let expected = _mm512_setr_epi64(552, 553, 554, 555, 0, 0, 0, 0);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_cvtts_ps_epu64() {
        let a = _mm_setr_ps(560.7, 561.3, 562.9, 563.1);
        let r = _mm_cvtts_ps_epu64(a);
        let expected = _mm_setr_epi64x(560, 561);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_cvtts_ps_epu64() {
        let a = _mm_setr_ps(564.5, 565.8, 566.2, 567.6);
        let src = _mm_setr_epi64x(100, 200);
        let r = _mm_mask_cvtts_ps_epu64(src, 0b01, a);
        let expected = _mm_setr_epi64x(564, 200);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_cvtts_ps_epu64() {
        let a = _mm_setr_ps(568.4, 569.9, 570.3, 571.7);
        let r = _mm_maskz_cvtts_ps_epu64(0b10, a);
        let expected = _mm_setr_epi64x(0, 569);
        assert_eq_m128i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_cvtts_ps_epu64() {
        let a = _mm_setr_ps(572.6, 573.2, 574.8, 575.3);
        let r = _mm256_cvtts_ps_epu64(a);
        let expected = _mm256_setr_epi64x(572, 573, 574, 575);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_cvtts_ps_epu64() {
        let a = _mm_setr_ps(580.4, 581.8, 582.2, 583.6);
        let src = _mm256_setr_epi64x(100, 200, 300, 400);
        let r = _mm256_mask_cvtts_ps_epu64(src, 0b0101, a);
        let expected = _mm256_setr_epi64x(580, 200, 582, 400);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_cvtts_ps_epu64() {
        let a = _mm_setr_ps(588.5, 589.8, 590.3, 591.7);
        let r = _mm256_maskz_cvtts_ps_epu64(0b1010, a);
        let expected = _mm256_setr_epi64x(0, 589, 0, 591);
        assert_eq_m256i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_cvtts_ps_epu64() {
        let a = _mm256_setr_ps(596.7, 597.3, 598.9, 599.1, 600.5, 601.8, 602.2, 603.6);
        let r = _mm512_cvtts_ps_epu64(a);
        let expected = _mm512_setr_epi64(596, 597, 598, 599, 600, 601, 602, 603);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_cvtts_ps_epu64() {
        let a = _mm256_setr_ps(604.4, 605.8, 606.3, 607.7, 608.1, 609.9, 610.5, 611.2);
        let src = _mm512_setr_epi64(100, 200, 300, 400, 500, 600, 700, 800);
        let r = _mm512_mask_cvtts_ps_epu64(src, 0b10101010, a);
        let expected = _mm512_setr_epi64(100, 605, 300, 607, 500, 609, 700, 611);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_cvtts_ps_epu64() {
        let a = _mm256_setr_ps(612.6, 613.4, 614.8, 615.2, 616.7, 617.9, 618.3, 619.5);
        let r = _mm512_maskz_cvtts_ps_epu64(0b11110000, a);
        let expected = _mm512_setr_epi64(0, 0, 0, 0, 616, 617, 618, 619);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_cvtts_roundps_epu64() {
        let a = _mm256_setr_ps(620.7, 621.3, 622.9, 623.1, 624.5, 625.8, 626.2, 627.6);
        let r = _mm512_cvtts_roundps_epu64::<_MM_FROUND_NO_EXC>(a);
        let expected = _mm512_setr_epi64(620, 621, 622, 623, 624, 625, 626, 627);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_cvtts_roundps_epu64() {
        let a = _mm256_setr_ps(628.4, 629.8, 630.3, 631.7, 632.1, 633.9, 634.5, 635.2);
        let src = _mm512_setr_epi64(100, 200, 300, 400, 500, 600, 700, 800);
        let r = _mm512_mask_cvtts_roundps_epu64::<_MM_FROUND_NO_EXC>(src, 0b01010101, a);
        let expected = _mm512_setr_epi64(628, 200, 630, 400, 632, 600, 634, 800);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_cvtts_roundps_epu64() {
        let a = _mm256_setr_ps(636.6, 637.4, 638.8, 639.3, 640.7, 641.9, 642.2, 643.5);
        let r = _mm512_maskz_cvtts_roundps_epu64::<_MM_FROUND_NO_EXC>(0b00001111, a);
        let expected = _mm512_setr_epi64(636, 637, 638, 639, 0, 0, 0, 0);
        assert_eq_m512i(r, expected);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_minmax_pd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(3.0, 0.0);

        let r = _mm_minmax_pd::<0>(a, b);
        let e = _mm_setr_pd(1.0, 0.0);
        assert_eq_m128d(r, e);

        let r = _mm_minmax_pd::<1>(a, b);
        let e = _mm_setr_pd(3.0, 2.0);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_minmax_pd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(3.0, 0.0);
        let src = _mm_setr_pd(20.0, 30.0);

        let r = _mm_mask_minmax_pd::<0>(src, 0b01, a, b);
        let e = _mm_setr_pd(1.0, 30.0);
        assert_eq_m128d(r, e);

        let r = _mm_mask_minmax_pd::<1>(src, 0b01, a, b);
        let e = _mm_setr_pd(3.0, 30.0);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_minmax_pd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(3.0, 0.0);

        let r = _mm_maskz_minmax_pd::<0>(0b01, a, b);
        let e = _mm_setr_pd(1.0, 0.0);
        assert_eq_m128d(r, e);

        let r = _mm_maskz_minmax_pd::<1>(0b01, a, b);
        let e = _mm_setr_pd(3.0, 0.0);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_minmax_pd() {
        let a = _mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
        let b = _mm256_setr_pd(5.0, 6.0, 7.0, 8.0);

        let r = _mm256_minmax_pd::<0>(a, b);
        let e = _mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
        assert_eq_m256d(r, e);

        let r = _mm256_minmax_pd::<1>(a, b);
        let e = _mm256_setr_pd(5.0, 6.0, 7.0, 8.0);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_minmax_pd() {
        let a = _mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
        let b = _mm256_setr_pd(5.0, 6.0, 7.0, 8.0);
        let src = _mm256_setr_pd(20.0, 30.0, 40.0, 50.0);

        let r = _mm256_mask_minmax_pd::<0>(src, 0b0101, a, b);
        let e = _mm256_setr_pd(1.0, 30.0, 3.0, 50.0);
        assert_eq_m256d(r, e);

        let r = _mm256_mask_minmax_pd::<1>(src, 0b0101, a, b);
        let e = _mm256_setr_pd(5.0, 30.0, 7.0, 50.0);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_minmax_pd() {
        let a = _mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
        let b = _mm256_setr_pd(5.0, 6.0, 7.0, 8.0);

        let r = _mm256_maskz_minmax_pd::<0>(0b0101, a, b);
        let e = _mm256_setr_pd(1.0, 0.0, 3.0, 0.0);
        assert_eq_m256d(r, e);

        let r = _mm256_maskz_minmax_pd::<1>(0b0101, a, b);
        let e = _mm256_setr_pd(5.0, 0.0, 7.0, 0.0);
        assert_eq_m256d(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_minmax_pd() {
        let a = _mm512_setr_pd(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm512_setr_pd(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);

        let r = _mm512_minmax_pd::<0>(a, b);
        let e = _mm512_setr_pd(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m512d(r, e);

        let r = _mm512_minmax_pd::<1>(a, b);
        let e = _mm512_setr_pd(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_minmax_pd() {
        let a = _mm512_setr_pd(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm512_setr_pd(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
        let src = _mm512_setr_pd(20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0);

        let r = _mm512_mask_minmax_pd::<0>(src, 0b01010101, a, b);
        let e = _mm512_setr_pd(1.0, 30.0, 3.0, 50.0, 5.0, 70.0, 7.0, 90.0);
        assert_eq_m512d(r, e);

        let r = _mm512_mask_minmax_pd::<1>(src, 0b01010101, a, b);
        let e = _mm512_setr_pd(9.0, 30.0, 11.0, 50.0, 13.0, 70.0, 15.0, 90.0);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_minmax_pd() {
        let a = _mm512_setr_pd(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm512_setr_pd(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);

        let r = _mm512_maskz_minmax_pd::<0>(0b01010101, a, b);
        let e = _mm512_setr_pd(1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0);
        assert_eq_m512d(r, e);

        let r = _mm512_maskz_minmax_pd::<1>(0b01010101, a, b);
        let e = _mm512_setr_pd(9.0, 0.0, 11.0, 0.0, 13.0, 0.0, 15.0, 0.0);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_minmax_round_pd() {
        let a = _mm512_setr_pd(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm512_setr_pd(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);

        let r = _mm512_minmax_round_pd::<0, _MM_FROUND_NO_EXC>(a, b);
        let e = _mm512_setr_pd(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m512d(r, e);

        let r = _mm512_minmax_round_pd::<1, _MM_FROUND_NO_EXC>(a, b);
        let e = _mm512_setr_pd(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_minmax_round_pd() {
        let a = _mm512_setr_pd(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm512_setr_pd(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
        let src = _mm512_setr_pd(20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0);

        let r = _mm512_mask_minmax_round_pd::<0, _MM_FROUND_NO_EXC>(src, 0b01010101, a, b);
        let e = _mm512_setr_pd(1.0, 30.0, 3.0, 50.0, 5.0, 70.0, 7.0, 90.0);
        assert_eq_m512d(r, e);

        let r = _mm512_mask_minmax_round_pd::<1, _MM_FROUND_NO_EXC>(src, 0b01010101, a, b);
        let e = _mm512_setr_pd(9.0, 30.0, 11.0, 50.0, 13.0, 70.0, 15.0, 90.0);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_minmax_round_pd() {
        let a = _mm512_setr_pd(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm512_setr_pd(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);

        let r = _mm512_maskz_minmax_round_pd::<0, _MM_FROUND_NO_EXC>(0b01010101, a, b);
        let e = _mm512_setr_pd(1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0);
        assert_eq_m512d(r, e);

        let r = _mm512_maskz_minmax_round_pd::<1, _MM_FROUND_NO_EXC>(0b01010101, a, b);
        let e = _mm512_setr_pd(9.0, 0.0, 11.0, 0.0, 13.0, 0.0, 15.0, 0.0);
        assert_eq_m512d(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_minmax_ps() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);

        let r = _mm_minmax_ps::<0>(a, b);
        let e = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        assert_eq_m128(r, e);

        let r = _mm_minmax_ps::<1>(a, b);
        let e = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_minmax_ps() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);
        let src = _mm_setr_ps(20.0, 30.0, 40.0, 50.0);

        let r = _mm_mask_minmax_ps::<0>(src, 0b0101, a, b);
        let e = _mm_setr_ps(1.0, 30.0, 3.0, 50.0);
        assert_eq_m128(r, e);

        let r = _mm_mask_minmax_ps::<1>(src, 0b0101, a, b);
        let e = _mm_setr_ps(5.0, 30.0, 7.0, 50.0);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_minmax_ps() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);

        let r = _mm_maskz_minmax_ps::<0>(0b0101, a, b);
        let e = _mm_setr_ps(1.0, 0.0, 3.0, 0.0);
        assert_eq_m128(r, e);

        let r = _mm_maskz_minmax_ps::<1>(0b0101, a, b);
        let e = _mm_setr_ps(5.0, 0.0, 7.0, 0.0);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_minmax_ps() {
        let a = _mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm256_setr_ps(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);

        let r = _mm256_minmax_ps::<0>(a, b);
        let e = _mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m256(r, e);

        let r = _mm256_minmax_ps::<1>(a, b);
        let e = _mm256_setr_ps(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_minmax_ps() {
        let a = _mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm256_setr_ps(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
        let src = _mm256_setr_ps(20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0);

        let r = _mm256_mask_minmax_ps::<0>(src, 0b01010101, a, b);
        let e = _mm256_setr_ps(1.0, 30.0, 3.0, 50.0, 5.0, 70.0, 7.0, 90.0);
        assert_eq_m256(r, e);

        let r = _mm256_mask_minmax_ps::<1>(src, 0b01010101, a, b);
        let e = _mm256_setr_ps(9.0, 30.0, 11.0, 50.0, 13.0, 70.0, 15.0, 90.0);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_minmax_ps() {
        let a = _mm256_setr_ps(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm256_setr_ps(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);

        let r = _mm256_maskz_minmax_ps::<0>(0b01010101, a, b);
        let e = _mm256_setr_ps(1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0);
        assert_eq_m256(r, e);

        let r = _mm256_maskz_minmax_ps::<1>(0b01010101, a, b);
        let e = _mm256_setr_ps(9.0, 0.0, 11.0, 0.0, 13.0, 0.0, 15.0, 0.0);
        assert_eq_m256(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_minmax_ps() {
        let a = _mm512_setr_ps(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm512_setr_ps(
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );

        let r = _mm512_minmax_ps::<0>(a, b);
        let e = _mm512_setr_ps(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        assert_eq_m512(r, e);

        let r = _mm512_minmax_ps::<1>(a, b);
        let e = _mm512_setr_ps(
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_minmax_ps() {
        let a = _mm512_setr_ps(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm512_setr_ps(
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let src = _mm512_setr_ps(
            20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0,
            150.0, 160.0, 170.0,
        );

        let r = _mm512_mask_minmax_ps::<0>(src, 0b0101010101010101, a, b);
        let e = _mm512_setr_ps(
            1.0, 30.0, 3.0, 50.0, 5.0, 70.0, 7.0, 90.0, 9.0, 110.0, 11.0, 130.0, 13.0, 150.0, 15.0,
            170.0,
        );
        assert_eq_m512(r, e);

        let r = _mm512_mask_minmax_ps::<1>(src, 0b0101010101010101, a, b);
        let e = _mm512_setr_ps(
            17.0, 30.0, 19.0, 50.0, 21.0, 70.0, 23.0, 90.0, 25.0, 110.0, 27.0, 130.0, 29.0, 150.0,
            31.0, 170.0,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_minmax_ps() {
        let a = _mm512_setr_ps(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm512_setr_ps(
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );

        let r = _mm512_maskz_minmax_ps::<0>(0b0101010101010101, a, b);
        let e = _mm512_setr_ps(
            1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0, 0.0, 11.0, 0.0, 13.0, 0.0, 15.0, 0.0,
        );
        assert_eq_m512(r, e);

        let r = _mm512_maskz_minmax_ps::<1>(0b0101010101010101, a, b);
        let e = _mm512_setr_ps(
            17.0, 0.0, 19.0, 0.0, 21.0, 0.0, 23.0, 0.0, 25.0, 0.0, 27.0, 0.0, 29.0, 0.0, 31.0, 0.0,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_minmax_round_ps() {
        let a = _mm512_setr_ps(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm512_setr_ps(
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );

        let r = _mm512_minmax_round_ps::<0, _MM_FROUND_NO_EXC>(a, b);
        let e = _mm512_setr_ps(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        assert_eq_m512(r, e);

        let r = _mm512_minmax_round_ps::<1, _MM_FROUND_NO_EXC>(a, b);
        let e = _mm512_setr_ps(
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_minmax_round_ps() {
        let a = _mm512_setr_ps(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm512_setr_ps(
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let src = _mm512_setr_ps(
            20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0,
            150.0, 160.0, 170.0,
        );

        let r = _mm512_mask_minmax_round_ps::<0, _MM_FROUND_NO_EXC>(src, 0b0101010101010101, a, b);
        let e = _mm512_setr_ps(
            1.0, 30.0, 3.0, 50.0, 5.0, 70.0, 7.0, 90.0, 9.0, 110.0, 11.0, 130.0, 13.0, 150.0, 15.0,
            170.0,
        );
        assert_eq_m512(r, e);

        let r = _mm512_mask_minmax_round_ps::<1, _MM_FROUND_NO_EXC>(src, 0b0101010101010101, a, b);
        let e = _mm512_setr_ps(
            17.0, 30.0, 19.0, 50.0, 21.0, 70.0, 23.0, 90.0, 25.0, 110.0, 27.0, 130.0, 29.0, 150.0,
            31.0, 170.0,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_minmax_round_ps() {
        let a = _mm512_setr_ps(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm512_setr_ps(
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );

        let r = _mm512_maskz_minmax_round_ps::<0, _MM_FROUND_NO_EXC>(0b0101010101010101, a, b);
        let e = _mm512_setr_ps(
            1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0, 0.0, 11.0, 0.0, 13.0, 0.0, 15.0, 0.0,
        );
        assert_eq_m512(r, e);

        let r = _mm512_maskz_minmax_round_ps::<1, _MM_FROUND_NO_EXC>(0b0101010101010101, a, b);
        let e = _mm512_setr_ps(
            17.0, 0.0, 19.0, 0.0, 21.0, 0.0, 23.0, 0.0, 25.0, 0.0, 27.0, 0.0, 29.0, 0.0, 31.0, 0.0,
        );
        assert_eq_m512(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_minmax_ph() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_setr_ph(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);

        let r = _mm_minmax_ph::<0>(a, b);
        let e = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);

        let r = _mm_minmax_ph::<1>(a, b);
        let e = _mm_setr_ph(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_minmax_ph() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_setr_ph(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
        let src = _mm_setr_ph(20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0);

        let r = _mm_mask_minmax_ph::<0>(src, 0b01010101, a, b);
        let e = _mm_setr_ph(1.0, 30.0, 3.0, 50.0, 5.0, 70.0, 7.0, 90.0);
        assert_eq_m128h(r, e);

        let r = _mm_mask_minmax_ph::<1>(src, 0b01010101, a, b);
        let e = _mm_setr_ph(9.0, 30.0, 11.0, 50.0, 13.0, 70.0, 15.0, 90.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_minmax_ph() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_setr_ph(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);

        let r = _mm_maskz_minmax_ph::<0>(0b01010101, a, b);
        let e = _mm_setr_ph(1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0);
        assert_eq_m128h(r, e);

        let r = _mm_maskz_minmax_ph::<1>(0b01010101, a, b);
        let e = _mm_setr_ph(9.0, 0.0, 11.0, 0.0, 13.0, 0.0, 15.0, 0.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_minmax_ph() {
        let a = _mm256_setr_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm256_setr_ph(
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );

        let r = _mm256_minmax_ph::<0>(a, b);
        let e = _mm256_setr_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        assert_eq_m256h(r, e);

        let r = _mm256_minmax_ph::<1>(a, b);
        let e = _mm256_setr_ph(
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_mask_minmax_ph() {
        let a = _mm256_setr_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm256_setr_ph(
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let src = _mm256_setr_ph(
            20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0,
            150.0, 160.0, 170.0,
        );

        let r = _mm256_mask_minmax_ph::<0>(src, 0b0101010101010101, a, b);
        let e = _mm256_setr_ph(
            1.0, 30.0, 3.0, 50.0, 5.0, 70.0, 7.0, 90.0, 9.0, 110.0, 11.0, 130.0, 13.0, 150.0, 15.0,
            170.0,
        );
        assert_eq_m256h(r, e);

        let r = _mm256_mask_minmax_ph::<1>(src, 0b0101010101010101, a, b);
        let e = _mm256_setr_ph(
            17.0, 30.0, 19.0, 50.0, 21.0, 70.0, 23.0, 90.0, 25.0, 110.0, 27.0, 130.0, 29.0, 150.0,
            31.0, 170.0,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm256_maskz_minmax_ph() {
        let a = _mm256_setr_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        );
        let b = _mm256_setr_ph(
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );

        let r = _mm256_maskz_minmax_ph::<0>(0b0101010101010101, a, b);
        let e = _mm256_setr_ph(
            1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0, 0.0, 11.0, 0.0, 13.0, 0.0, 15.0, 0.0,
        );
        assert_eq_m256h(r, e);

        let r = _mm256_maskz_minmax_ph::<1>(0b0101010101010101, a, b);
        let e = _mm256_setr_ph(
            17.0, 0.0, 19.0, 0.0, 21.0, 0.0, 23.0, 0.0, 25.0, 0.0, 27.0, 0.0, 29.0, 0.0, 31.0, 0.0,
        );
        assert_eq_m256h(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_minmax_ph() {
        let a = _mm512_setr_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_setr_ph(
            33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0,
            47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
            61.0, 62.0, 63.0, 64.0,
        );

        let r = _mm512_minmax_ph::<0>(a, b);
        let e = _mm512_setr_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        assert_eq_m512h(r, e);

        let r = _mm512_minmax_ph::<1>(a, b);
        let e = _mm512_setr_ph(
            33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0,
            47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
            61.0, 62.0, 63.0, 64.0,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_minmax_ph() {
        let a = _mm512_setr_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_setr_ph(
            33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0,
            47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
            61.0, 62.0, 63.0, 64.0,
        );
        let src = _mm512_setr_ph(
            65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0,
            79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0,
            93.0, 94.0, 95.0, 96.0,
        );

        let r = _mm512_mask_minmax_ph::<0>(src, 0b01010101010101010101010101010101, a, b);
        let e = _mm512_setr_ph(
            1.0, 66.0, 3.0, 68.0, 5.0, 70.0, 7.0, 72.0, 9.0, 74.0, 11.0, 76.0, 13.0, 78.0, 15.0,
            80.0, 17.0, 82.0, 19.0, 84.0, 21.0, 86.0, 23.0, 88.0, 25.0, 90.0, 27.0, 92.0, 29.0,
            94.0, 31.0, 96.0,
        );
        assert_eq_m512h(r, e);

        let r = _mm512_mask_minmax_ph::<1>(src, 0b01010101010101010101010101010101, a, b);
        let e = _mm512_setr_ph(
            33.0, 66.0, 35.0, 68.0, 37.0, 70.0, 39.0, 72.0, 41.0, 74.0, 43.0, 76.0, 45.0, 78.0,
            47.0, 80.0, 49.0, 82.0, 51.0, 84.0, 53.0, 86.0, 55.0, 88.0, 57.0, 90.0, 59.0, 92.0,
            61.0, 94.0, 63.0, 96.0,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_minmax_ph() {
        let a = _mm512_setr_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_setr_ph(
            33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0,
            47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
            61.0, 62.0, 63.0, 64.0,
        );

        let r = _mm512_maskz_minmax_ph::<0>(0b01010101010101010101010101010101, a, b);
        let e = _mm512_setr_ph(
            1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0, 0.0, 11.0, 0.0, 13.0, 0.0, 15.0, 0.0,
            17.0, 0.0, 19.0, 0.0, 21.0, 0.0, 23.0, 0.0, 25.0, 0.0, 27.0, 0.0, 29.0, 0.0, 31.0, 0.0,
        );
        assert_eq_m512h(r, e);

        let r = _mm512_maskz_minmax_ph::<1>(0b01010101010101010101010101010101, a, b);
        let e = _mm512_setr_ph(
            33.0, 0.0, 35.0, 0.0, 37.0, 0.0, 39.0, 0.0, 41.0, 0.0, 43.0, 0.0, 45.0, 0.0, 47.0, 0.0,
            49.0, 0.0, 51.0, 0.0, 53.0, 0.0, 55.0, 0.0, 57.0, 0.0, 59.0, 0.0, 61.0, 0.0, 63.0, 0.0,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_minmax_round_ph() {
        let a = _mm512_setr_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_setr_ph(
            33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0,
            47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
            61.0, 62.0, 63.0, 64.0,
        );

        let r = _mm512_minmax_round_ph::<0, _MM_FROUND_NO_EXC>(a, b);
        let e = _mm512_setr_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        assert_eq_m512h(r, e);

        let r = _mm512_minmax_round_ph::<1, _MM_FROUND_NO_EXC>(a, b);
        let e = _mm512_setr_ph(
            33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0,
            47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
            61.0, 62.0, 63.0, 64.0,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_mask_minmax_round_ph() {
        let a = _mm512_setr_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_setr_ph(
            33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0,
            47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
            61.0, 62.0, 63.0, 64.0,
        );
        let src = _mm512_setr_ph(
            65.0, 66.0, 67.0, 68.0, 69.0, 70.0, 71.0, 72.0, 73.0, 74.0, 75.0, 76.0, 77.0, 78.0,
            79.0, 80.0, 81.0, 82.0, 83.0, 84.0, 85.0, 86.0, 87.0, 88.0, 89.0, 90.0, 91.0, 92.0,
            93.0, 94.0, 95.0, 96.0,
        );

        let r = _mm512_mask_minmax_round_ph::<0, _MM_FROUND_NO_EXC>(
            src,
            0b01010101010101010101010101010101,
            a,
            b,
        );
        let e = _mm512_setr_ph(
            1.0, 66.0, 3.0, 68.0, 5.0, 70.0, 7.0, 72.0, 9.0, 74.0, 11.0, 76.0, 13.0, 78.0, 15.0,
            80.0, 17.0, 82.0, 19.0, 84.0, 21.0, 86.0, 23.0, 88.0, 25.0, 90.0, 27.0, 92.0, 29.0,
            94.0, 31.0, 96.0,
        );
        assert_eq_m512h(r, e);

        let r = _mm512_mask_minmax_round_ph::<1, _MM_FROUND_NO_EXC>(
            src,
            0b01010101010101010101010101010101,
            a,
            b,
        );
        let e = _mm512_setr_ph(
            33.0, 66.0, 35.0, 68.0, 37.0, 70.0, 39.0, 72.0, 41.0, 74.0, 43.0, 76.0, 45.0, 78.0,
            47.0, 80.0, 49.0, 82.0, 51.0, 84.0, 53.0, 86.0, 55.0, 88.0, 57.0, 90.0, 59.0, 92.0,
            61.0, 94.0, 63.0, 96.0,
        );
        assert_eq_m512h(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm512_maskz_minmax_round_ph() {
        let a = _mm512_setr_ph(
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
            31.0, 32.0,
        );
        let b = _mm512_setr_ph(
            33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0,
            47.0, 48.0, 49.0, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0, 58.0, 59.0, 60.0,
            61.0, 62.0, 63.0, 64.0,
        );

        let r = _mm512_maskz_minmax_round_ph::<0, _MM_FROUND_NO_EXC>(
            0b01010101010101010101010101010101,
            a,
            b,
        );
        let e = _mm512_setr_ph(
            1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0, 0.0, 11.0, 0.0, 13.0, 0.0, 15.0, 0.0,
            17.0, 0.0, 19.0, 0.0, 21.0, 0.0, 23.0, 0.0, 25.0, 0.0, 27.0, 0.0, 29.0, 0.0, 31.0, 0.0,
        );
        assert_eq_m512h(r, e);

        let r = _mm512_maskz_minmax_round_ph::<1, _MM_FROUND_NO_EXC>(
            0b01010101010101010101010101010101,
            a,
            b,
        );
        let e = _mm512_setr_ph(
            33.0, 0.0, 35.0, 0.0, 37.0, 0.0, 39.0, 0.0, 41.0, 0.0, 43.0, 0.0, 45.0, 0.0, 47.0, 0.0,
            49.0, 0.0, 51.0, 0.0, 53.0, 0.0, 55.0, 0.0, 57.0, 0.0, 59.0, 0.0, 61.0, 0.0, 63.0, 0.0,
        );
        assert_eq_m512h(r, e);
    }

    // FIXME: the following tests do not pass due to a LLVM miscompilation bug. See llvm/llvm-project#184245

    #[simd_test(enable = "avx10.2")]
    fn test_mm_minmax_sd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(3.0, 4.0);

        let r = _mm_minmax_sd::<0>(a, b);
        let e = _mm_setr_pd(1.0, 2.0);
        assert_eq_m128d(r, e);

        let r = _mm_minmax_sd::<1>(a, b);
        let e = _mm_setr_pd(3.0, 2.0);
        assert_eq_m128d(r, e);
    }

    #[ignore]
    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_minmax_sd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(3.0, 4.0);
        let src = _mm_setr_pd(20.0, 30.0);

        let r = _mm_mask_minmax_sd::<0>(src, 1, a, b);
        let e = _mm_setr_pd(1.0, 2.0);
        assert_eq_m128d(r, e);

        let r = _mm_mask_minmax_sd::<0>(src, 0, a, b);
        let e = _mm_setr_pd(20.0, 2.0);
        assert_eq_m128d(r, e);

        let r = _mm_mask_minmax_sd::<1>(src, 1, a, b);
        let e = _mm_setr_pd(3.0, 2.0);
        assert_eq_m128d(r, e);
    }

    #[ignore]
    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_minmax_sd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(3.0, 4.0);

        let r = _mm_maskz_minmax_sd::<0>(1, a, b);
        let e = _mm_setr_pd(1.0, 2.0);
        assert_eq_m128d(r, e);

        let r = _mm_maskz_minmax_sd::<0>(0, a, b);
        let e = _mm_setr_pd(0.0, 2.0);
        assert_eq_m128d(r, e);

        let r = _mm_maskz_minmax_sd::<1>(1, a, b);
        let e = _mm_setr_pd(3.0, 2.0);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_minmax_round_sd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(3.0, 4.0);

        let r = _mm_minmax_round_sd::<0, _MM_FROUND_NO_EXC>(a, b);
        let e = _mm_setr_pd(1.0, 2.0);
        assert_eq_m128d(r, e);

        let r = _mm_minmax_round_sd::<1, _MM_FROUND_NO_EXC>(a, b);
        let e = _mm_setr_pd(3.0, 2.0);
        assert_eq_m128d(r, e);
    }

    #[ignore]
    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_minmax_round_sd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(3.0, 4.0);
        let src = _mm_setr_pd(20.0, 30.0);

        let r = _mm_mask_minmax_round_sd::<0, _MM_FROUND_NO_EXC>(src, 1, a, b);
        let e = _mm_setr_pd(1.0, 2.0);
        assert_eq_m128d(r, e);

        let r = _mm_mask_minmax_round_sd::<0, _MM_FROUND_NO_EXC>(src, 0, a, b);
        let e = _mm_setr_pd(20.0, 2.0);
        assert_eq_m128d(r, e);

        let r = _mm_mask_minmax_round_sd::<1, _MM_FROUND_NO_EXC>(src, 1, a, b);
        let e = _mm_setr_pd(3.0, 2.0);
        assert_eq_m128d(r, e);
    }

    #[ignore]
    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_minmax_round_sd() {
        let a = _mm_setr_pd(1.0, 2.0);
        let b = _mm_setr_pd(3.0, 4.0);

        let r = _mm_maskz_minmax_round_sd::<0, _MM_FROUND_NO_EXC>(1, a, b);
        let e = _mm_setr_pd(1.0, 2.0);
        assert_eq_m128d(r, e);

        let r = _mm_maskz_minmax_round_sd::<0, _MM_FROUND_NO_EXC>(0, a, b);
        let e = _mm_setr_pd(0.0, 2.0);
        assert_eq_m128d(r, e);

        let r = _mm_maskz_minmax_round_sd::<1, _MM_FROUND_NO_EXC>(1, a, b);
        let e = _mm_setr_pd(3.0, 2.0);
        assert_eq_m128d(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_minmax_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);

        let r = _mm_minmax_ss::<0>(a, b);
        let e = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        assert_eq_m128(r, e);

        let r = _mm_minmax_ss::<1>(a, b);
        let e = _mm_setr_ps(5.0, 2.0, 3.0, 4.0);
        assert_eq_m128(r, e);
    }

    #[ignore]
    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_minmax_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);
        let src = _mm_setr_ps(20.0, 30.0, 40.0, 50.0);

        let r = _mm_mask_minmax_ss::<0>(src, 1, a, b);
        let e = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        assert_eq_m128(r, e);

        let r = _mm_mask_minmax_ss::<0>(src, 0, a, b);
        let e = _mm_setr_ps(20.0, 2.0, 3.0, 4.0);
        assert_eq_m128(r, e);

        let r = _mm_mask_minmax_ss::<1>(src, 1, a, b);
        let e = _mm_setr_ps(5.0, 2.0, 3.0, 4.0);
        assert_eq_m128(r, e);
    }

    #[ignore]
    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_minmax_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);

        let r = _mm_maskz_minmax_ss::<0>(1, a, b);
        let e = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        assert_eq_m128(r, e);

        let r = _mm_maskz_minmax_ss::<0>(0, a, b);
        let e = _mm_setr_ps(0.0, 2.0, 3.0, 4.0);
        assert_eq_m128(r, e);

        let r = _mm_maskz_minmax_ss::<1>(1, a, b);
        let e = _mm_setr_ps(5.0, 2.0, 3.0, 4.0);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_minmax_round_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);

        let r = _mm_minmax_round_ss::<0, _MM_FROUND_NO_EXC>(a, b);
        let e = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        assert_eq_m128(r, e);

        let r = _mm_minmax_round_ss::<1, _MM_FROUND_NO_EXC>(a, b);
        let e = _mm_setr_ps(5.0, 2.0, 3.0, 4.0);
        assert_eq_m128(r, e);
    }

    #[ignore]
    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_minmax_round_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);
        let src = _mm_setr_ps(20.0, 30.0, 40.0, 50.0);

        let r = _mm_mask_minmax_round_ss::<0, _MM_FROUND_NO_EXC>(src, 1, a, b);
        let e = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        assert_eq_m128(r, e);

        let r = _mm_mask_minmax_round_ss::<0, _MM_FROUND_NO_EXC>(src, 0, a, b);
        let e = _mm_setr_ps(20.0, 2.0, 3.0, 4.0);
        assert_eq_m128(r, e);

        let r = _mm_mask_minmax_round_ss::<1, _MM_FROUND_NO_EXC>(src, 1, a, b);
        let e = _mm_setr_ps(5.0, 2.0, 3.0, 4.0);
        assert_eq_m128(r, e);
    }

    #[ignore]
    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_minmax_round_ss() {
        let a = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        let b = _mm_setr_ps(5.0, 6.0, 7.0, 8.0);

        let r = _mm_maskz_minmax_round_ss::<0, _MM_FROUND_NO_EXC>(1, a, b);
        let e = _mm_setr_ps(1.0, 2.0, 3.0, 4.0);
        assert_eq_m128(r, e);

        let r = _mm_maskz_minmax_round_ss::<0, _MM_FROUND_NO_EXC>(0, a, b);
        let e = _mm_setr_ps(0.0, 2.0, 3.0, 4.0);
        assert_eq_m128(r, e);

        let r = _mm_maskz_minmax_round_ss::<1, _MM_FROUND_NO_EXC>(1, a, b);
        let e = _mm_setr_ps(5.0, 2.0, 3.0, 4.0);
        assert_eq_m128(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_minmax_sh() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_setr_ph(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);

        let r = _mm_minmax_sh::<0>(a, b);
        let e = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);

        let r = _mm_minmax_sh::<1>(a, b);
        let e = _mm_setr_ph(9.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);
    }

    #[ignore]
    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_minmax_sh() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_setr_ph(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
        let src = _mm_setr_ph(20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0);

        let r = _mm_mask_minmax_sh::<0>(src, 1, a, b);
        let e = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);

        let r = _mm_mask_minmax_sh::<0>(src, 0, a, b);
        let e = _mm_setr_ph(20.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);

        let r = _mm_mask_minmax_sh::<1>(src, 1, a, b);
        let e = _mm_setr_ph(9.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);
    }

    #[ignore]
    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_minmax_sh() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_setr_ph(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);

        let r = _mm_maskz_minmax_sh::<0>(1, a, b);
        let e = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);

        let r = _mm_maskz_minmax_sh::<0>(0, a, b);
        let e = _mm_setr_ph(0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);

        let r = _mm_maskz_minmax_sh::<1>(1, a, b);
        let e = _mm_setr_ph(9.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);
    }

    #[simd_test(enable = "avx10.2")]
    fn test_mm_minmax_round_sh() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_setr_ph(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);

        let r = _mm_minmax_round_sh::<0, _MM_FROUND_NO_EXC>(a, b);
        let e = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);

        let r = _mm_minmax_round_sh::<1, _MM_FROUND_NO_EXC>(a, b);
        let e = _mm_setr_ph(9.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);
    }

    #[ignore]
    #[simd_test(enable = "avx10.2")]
    fn test_mm_mask_minmax_round_sh() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_setr_ph(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);
        let src = _mm_setr_ph(20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0);

        let r = _mm_mask_minmax_round_sh::<0, _MM_FROUND_NO_EXC>(src, 1, a, b);
        let e = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);

        let r = _mm_mask_minmax_round_sh::<0, _MM_FROUND_NO_EXC>(src, 0, a, b);
        let e = _mm_setr_ph(20.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);

        let r = _mm_mask_minmax_round_sh::<1, _MM_FROUND_NO_EXC>(src, 1, a, b);
        let e = _mm_setr_ph(9.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);
    }

    #[ignore]
    #[simd_test(enable = "avx10.2")]
    fn test_mm_maskz_minmax_round_sh() {
        let a = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let b = _mm_setr_ph(9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0);

        let r = _mm_maskz_minmax_round_sh::<0, _MM_FROUND_NO_EXC>(1, a, b);
        let e = _mm_setr_ph(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);

        let r = _mm_maskz_minmax_round_sh::<0, _MM_FROUND_NO_EXC>(0, a, b);
        let e = _mm_setr_ph(0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);

        let r = _mm_maskz_minmax_round_sh::<1, _MM_FROUND_NO_EXC>(1, a, b);
        let e = _mm_setr_ph(9.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        assert_eq_m128h(r, e);
    }
}
