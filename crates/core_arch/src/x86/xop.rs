use crate::core_arch::{simd::*, x86::*};
use crate::intrinsics::simd::*;

/// Multiplies each packed 16-bit signed integer value in `a` by its corresponding packed 16-
/// bit signed integer value in `b`, then adds the 32-bit signed integer products to the
/// corresponding packed 16-bit signed integer value in `c`. The saturated results are written
/// to the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpmacssww))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_maccs_epi16(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    transmute(vpmacssww(a.as_i16x8(), b.as_i16x8(), c.as_i16x8()))
}

/// Multiplies each packed 16-bit signed integer value in `a` by the corresponding packed 16-
/// bit signed integer value in `b`, then adds each 32-bit signed integer product to the
/// corresponding packed 16-bit signed integer value in `c`. The eight results are written to
/// the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpmacsww))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_macc_epi16(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    transmute(vpmacsww(a.as_i16x8(), b.as_i16x8(), c.as_i16x8()))
}

/// Multiplies the odd-numbered packed 16-bit signed integer values in `a` by the
/// corresponding packed 16-bit signed integer values in `b`, then adds the 32-bit signed
/// integer products to the corresponding packed 32-bit signed integer values in `c`. The
/// saturated results are written to the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpmacsswd))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_maccsd_epi16(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    transmute(vpmacsswd(a.as_i16x8(), b.as_i16x8(), c.as_i32x4()))
}

/// Multiplies each odd-numbered packed 16-bit signed integer value in `a` by the
/// corresponding packed 16-bit signed integer value in `b`, then adds the 32-bit signed
/// integer products to the corresponding packed 32-bit signed integer value in `c`. The four
/// results are written to the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpmacswd))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_maccd_epi16(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    transmute(vpmacswd(a.as_i16x8(), b.as_i16x8(), c.as_i32x4()))
}

/// Multiplies each packed 32-bit signed integer value in `a` by the corresponding packed 32-
/// bit signed integer value in `b`, then adds each 64-bit signed integer product to the
/// corresponding packed 32-bit signed integer value in `c`. The saturated results are written
/// to the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpmacssdd))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_maccs_epi32(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    transmute(vpmacssdd(a.as_i32x4(), b.as_i32x4(), c.as_i32x4()))
}

/// Multiplies each packed 32-bit signed integer value in `a` by the corresponding packed 32-
/// bit signed integer value in `b`, then adds the 64-bit signed integer product to the
/// corresponding packed 32-bit signed integer value in `c`. The four resulting 32-bit sums
/// are stored in the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpmacsdd))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_macc_epi32(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    transmute(vpmacsdd(a.as_i32x4(), b.as_i32x4(), c.as_i32x4()))
}

/// Multiplies the low-order 32-bit signed integer value of `a` by the low-order 32-bit signed
/// integer value in `b`, then adds the 64-bit signed integer product to the low-order 64-bit
/// signed integer value in `c`. Simultaneously, multiplies the third 32-bit signed integer
/// value of `a` by the third 32-bit signed integer value in `b`, then adds the 64-
/// bit signed integer product to the high-order 64-bit signed integer value in `c`. The
/// saturated results are written to the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpmacssdql))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_maccslo_epi32(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    transmute(vpmacssdql(a.as_i32x4(), b.as_i32x4(), c.as_i64x2()))
}

/// Multiplies the low-order 32-bit signed integer value of `a` by the low-order 32-bit signed
/// integer value in `b`, then adds the 64-bit signed integer product to the low-order 64-bit
/// signed integer value in `c`. Simultaneously, multiplies the third 32-bit signed integer
/// value of `a` by the corresponding 32-bit signed integer value in `b`, then
/// adds the 64-bit signed integer product to the second 64-bit signed integer value in `c`. The
/// results are written to the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpmacsdql))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_macclo_epi32(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    transmute(vpmacsdql(a.as_i32x4(), b.as_i32x4(), c.as_i64x2()))
}

/// Multiplies the second 32-bit signed integer value of `a` by the second 32-bit signed integer
/// value in `b`, then adds the 64-bit signed integer product to the low-order 64-bit signed
/// integer value in `c`. Simultaneously, multiplies the fourth 32-bit signed integer value of
/// `a` by the fourth 32-bit signed integer value in `b`, then adds the 64-bit
/// signed integer product to the high-order 64-bit signed integer value in `c`. The saturated
/// results are written to the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpmacssdqh))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_maccshi_epi32(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    transmute(vpmacssdqh(a.as_i32x4(), b.as_i32x4(), c.as_i64x2()))
}

/// Multiplies the second 32-bit signed integer value of `a` by the second 32-bit signed integer
/// value in `b`, then adds the 64-bit signed integer product to the low-order 64-bit signed
/// integer value in `c`. Simultaneously, multiplies the fourth 32-bit signed integer value of
/// `a` by the fourth 32-bit signed integer value in `b`, then adds the 64-bit
/// signed integer product to the second 64-bit signed integer value in `c`.The results are
/// written to the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpmacsdqh))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_macchi_epi32(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    transmute(vpmacsdqh(a.as_i32x4(), b.as_i32x4(), c.as_i64x2()))
}

/// Multiplies each packed 16-bit signed integer value in `a` by the corresponding packed 16-
/// bit signed integer value in `b`, then adds the 32-bit signed integer products of the even-odd
/// adjacent words. Each resulting sum is then added to the corresponding packed 32-bit signed
/// integer value in `c`. The four saturated results are written to the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpmadcsswd))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_maddsd_epi16(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    transmute(vpmadcsswd(a.as_i16x8(), b.as_i16x8(), c.as_i32x4()))
}

/// Multiplies each packed 16-bit signed integer value in `a` by the corresponding packed 16-
/// bit signed integer value in `b`, then adds the 32-bit signed integer products of the even-odd
/// adjacent words together and adds their sum to the corresponding packed 32-bit signed integer
/// values in `c`. The four results are written to the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpmadcswd))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_maddd_epi16(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    transmute(vpmadcswd(a.as_i16x8(), b.as_i16x8(), c.as_i32x4()))
}

/// Adds each adjacent pair of 8-bit signed integer values from `a` and packs the sign-extended 16-
/// bit integer result of each addition in the corresponding 16-bit element of the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vphaddbw))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_haddw_epi8(a: __m128i) -> __m128i {
    transmute(vphaddbw(a.as_i8x16()))
}

/// Adds four successive 8-bit signed integer values from `a` and packs the sign-extended results
/// of the additions in the corresponding 32-bit element in the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vphaddbd))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_haddd_epi8(a: __m128i) -> __m128i {
    transmute(vphaddbd(a.as_i8x16()))
}

/// Adds eight successive 8-bit signed integer values from `a` and packs the sign-extended results
/// of the additions in the corresponding 64-bit element in the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vphaddbq))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_haddq_epi8(a: __m128i) -> __m128i {
    transmute(vphaddbq(a.as_i8x16()))
}

/// Adds each adjacent pair of 16-bit signed integer values from `a` and packs the sign-extended
/// results of the addition in the corresponding 32-bit element in the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vphaddwd))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_haddd_epi16(a: __m128i) -> __m128i {
    transmute(vphaddwd(a.as_i16x8()))
}

/// Adds four successive 16-bit signed integer values from the `a` and packs the sign-extended
/// results of each addition in the corresponding 64-bit element in the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vphaddwq))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_haddq_epi16(a: __m128i) -> __m128i {
    transmute(vphaddwq(a.as_i16x8()))
}

/// Adds each adjacent pair of signed 32-bit integer values in `a` and packs the sign-extended
/// sums of each addition in the corresponding 64-bit element in the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vphadddq))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_haddq_epi32(a: __m128i) -> __m128i {
    transmute(vphadddq(a.as_i32x4()))
}

/// Adds each adjacent pair of 8-bit unsigned integer values from `a` and packs the 16-bit integer
/// results of each addition in the corresponding word in the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vphaddubw))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_haddw_epu8(a: __m128i) -> __m128i {
    transmute(vphaddubw(a.as_u8x16()))
}

/// Adds four successive 8-bit unsigned integer values from `a` and packs the results of the additions
/// in the corresponding 32-bit element in the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vphaddubd))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_haddd_epu8(a: __m128i) -> __m128i {
    transmute(vphaddubd(a.as_u8x16()))
}

/// Adds eight successive 8-bit unsigned integer values from `a` and packs the results of the additions
/// in the corresponding 64-bit element in the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vphaddubq))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_haddq_epu8(a: __m128i) -> __m128i {
    transmute(vphaddubq(a.as_u8x16()))
}

/// Adds each adjacent pair of 16-bit unsigned integer values from `a` and packs the results of the
/// additions in the corresponding 32-bit element in the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vphadduwd))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_haddd_epu16(a: __m128i) -> __m128i {
    transmute(vphadduwd(a.as_u16x8()))
}

/// Adds four successive 16-bit unsigned integer values from `a` and packs the results of the additions
/// in the corresponding 64-bit element in the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vphadduwq))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_haddq_epu16(a: __m128i) -> __m128i {
    transmute(vphadduwq(a.as_u16x8()))
}

/// Adds each adjacent pair of unsigned 32-bit integer values in `a` and packs the sums of each
/// addition in the corresponding 64-bit element in the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vphaddudq))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_haddq_epu32(a: __m128i) -> __m128i {
    transmute(vphaddudq(a.as_u32x4()))
}

/// Subtracts the most significant signed 8-bit integer from the least significant signed 8-bit integer of each
/// 16-bit element in `a` and packs the sign-extended 16-bit integer results of each subtraction in the
/// destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vphsubbw))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_hsubw_epi8(a: __m128i) -> __m128i {
    transmute(vphsubbw(a.as_i8x16()))
}

/// Subtracts the most significant signed 16-bit integer from the least significant signed 16-bit integer of each
/// 32-bit element in `a` and packs the sign-extended 32-bit integer results of each subtraction in the
/// destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vphsubwd))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_hsubd_epi16(a: __m128i) -> __m128i {
    transmute(vphsubwd(a.as_i16x8()))
}

/// Subtracts the most significant signed 32-bit integer from the least significant signed 32-bit integer of each
/// 64-bit element in `a` and packs the sign-extended 64-bit integer results of each subtraction in the
/// destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vphsubdq))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_hsubq_epi32(a: __m128i) -> __m128i {
    transmute(vphsubdq(a.as_i32x4()))
}

/// Moves bits of either `a` or `b` into their corresponding positions in the destination, depending
/// on the value of the corresponding selector bit in `c`. If the selector bit is set to 1, the corresponding
/// bit in `a` is moved to the destination; otherwise, the corresponding bit from `b` is moved to the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpcmov))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_cmov_si128(a: __m128i, b: __m128i, c: __m128i) -> __m128i {
    simd_or(
        simd_and(a, c),
        simd_and(b, simd_xor(_mm_set1_epi64x(-1), c)),
    )
}

/// Moves bits of either `a` or `b` into their corresponding positions in the destination, depending
/// on the value of the corresponding selector bit in `c`. If the selector bit is set to 1, the corresponding
/// bit in `a` is moved to the destination; otherwise, the corresponding bit from `b` is moved to the destination.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpcmov))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm256_cmov_si256(a: __m256i, b: __m256i, c: __m256i) -> __m256i {
    simd_or(
        simd_and(a, c),
        simd_and(b, simd_xor(_mm256_set1_epi64x(-1), c)),
    )
}

/// Rotate each 8-bit element in `a` left by the number of bits specified in the corresponding element of `b`.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vprotb))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_rot_epi8(a: __m128i, b: __m128i) -> __m128i {
    transmute(vprotb(a.as_u8x16(), b.as_u8x16()))
}

/// Rotate each 16-bit element in `a` left by the number of bits specified in the corresponding element of `b`.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vprotw))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_rot_epi16(a: __m128i, b: __m128i) -> __m128i {
    transmute(vprotw(a.as_u16x8(), b.as_u16x8()))
}

/// Rotate each 32-bit element in `a` left by the number of bits specified in the corresponding element of `b`.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vprotd))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_rot_epi32(a: __m128i, b: __m128i) -> __m128i {
    transmute(vprotd(a.as_u32x4(), b.as_u32x4()))
}

/// Rotate each 64-bit element in `a` left by the number of bits specified in the corresponding element of `b`.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vprotq))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_rot_epi64(a: __m128i, b: __m128i) -> __m128i {
    transmute(vprotq(a.as_u64x2(), b.as_u64x2()))
}

/// Rotate each 8-bit element in `a` left by `IMM8` bits.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vprotb, IMM8 = 3))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_roti_epi8<const IMM8: i32>(a: __m128i) -> __m128i {
    _mm_rot_epi8(a, _mm_set1_epi8(IMM8 as i8))
}

/// Rotate each 16-bit element in `a` left by `IMM8` bits.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vprotw, IMM8 = 3))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_roti_epi16<const IMM8: i32>(a: __m128i) -> __m128i {
    _mm_rot_epi16(a, _mm_set1_epi16(IMM8 as i16))
}

/// Rotate each 32-bit element in `a` left by `IMM8` bits.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vprotd, IMM8 = 3))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_roti_epi32<const IMM8: i32>(a: __m128i) -> __m128i {
    _mm_rot_epi32(a, _mm_set1_epi32(IMM8))
}

/// Rotate each 64-bit element in `a` left by `IMM8` bits.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vprotq, IMM8 = 3))]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_roti_epi64<const IMM8: i32>(a: __m128i) -> __m128i {
    _mm_rot_epi64(a, _mm_set1_epi64x(IMM8 as i64))
}

/// Shifts each 8-bit element of `a` by the number of bits specified in the corresponding element of `count`.
///
/// If the `count` value is positive, bits are shifted to the left, while shifting in zeroes.
///
/// If the `count` value is negative, bits are shifted to the right, while shifting in zeroes.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpshlb))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_shl_epi8(a: __m128i, count: __m128i) -> __m128i {
    transmute(vpshlb(a.as_i8x16(), count.as_i8x16()))
}

/// Shifts each 16-bit element of `a` by the number of bits specified in the corresponding element of `count`.
///
/// If the `count` value is positive, bits are shifted to the left, while shifting in zeroes.
///
/// If the `count` value is negative, bits are shifted to the right, while shifting in zeroes.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpshlw))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_shl_epi16(a: __m128i, count: __m128i) -> __m128i {
    transmute(vpshlw(a.as_i16x8(), count.as_i16x8()))
}

/// Shifts each 32-bit element of `a` by the number of bits specified in the corresponding element of `count`.
///
/// If the `count` value is positive, bits are shifted to the left, while shifting in zeroes.
///
/// If the `count` value is negative, bits are shifted to the right, while shifting in zeroes.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpshld))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_shl_epi32(a: __m128i, count: __m128i) -> __m128i {
    transmute(vpshld(a.as_i32x4(), count.as_i32x4()))
}

/// Shifts each 64-bit element of `a` by the number of bits specified in the corresponding element of `count`.
///
/// If the `count` value is positive, bits are shifted to the left, while shifting in zeroes.
///
/// If the `count` value is negative, bits are shifted to the right, while shifting in zeroes.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpshlq))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_shl_epi64(a: __m128i, count: __m128i) -> __m128i {
    transmute(vpshlq(a.as_i64x2(), count.as_i64x2()))
}

/// Shifts each 8-bit element of `a` by the number of bits specified in the corresponding element of `count`.
///
/// If the `count` value is positive, bits are shifted to the left, while shifting in zeroes.
///
/// If the `count` value is negative, bits are shifted to the right, while shifting in sign bits.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpshab))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_sha_epi8(a: __m128i, count: __m128i) -> __m128i {
    transmute(vpshab(a.as_i8x16(), count.as_i8x16()))
}

/// Shifts each 16-bit element of `a` by the number of bits specified in the corresponding element of `count`.
///
/// If the `count` value is positive, bits are shifted to the left, while shifting in zeroes.
///
/// If the `count` value is negative, bits are shifted to the right, while shifting in sign bits.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpshaw))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_sha_epi16(a: __m128i, count: __m128i) -> __m128i {
    transmute(vpshaw(a.as_i16x8(), count.as_i16x8()))
}

/// Shifts each 32-bit element of `a` by the number of bits specified in the corresponding element of `count`.
///
/// If the `count` value is positive, bits are shifted to the left, while shifting in zeroes.
///
/// If the `count` value is negative, bits are shifted to the right, while shifting in sign bits.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpshad))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_sha_epi32(a: __m128i, count: __m128i) -> __m128i {
    transmute(vpshad(a.as_i32x4(), count.as_i32x4()))
}

/// Shifts each 64-bit element of `a` by the number of bits specified in the corresponding element of `count`.
///
/// If the `count` value is positive, bits are shifted to the left, while shifting in zeroes.
///
/// If the `count` value is negative, bits are shifted to the right, while shifting in sign bits.
#[inline]
#[target_feature(enable = "xop")]
#[cfg_attr(test, assert_instr(vpshaq))]
#[unstable(feature = "xop_target_feature", issue = "127208")]
pub unsafe fn _mm_sha_epi64(a: __m128i, count: __m128i) -> __m128i {
    transmute(vpshaq(a.as_i64x2(), count.as_i64x2()))
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.xop.vpmacssww"]
    fn vpmacssww(a: i16x8, b: i16x8, c: i16x8) -> i16x8;
    #[link_name = "llvm.x86.xop.vpmacsww"]
    fn vpmacsww(a: i16x8, b: i16x8, c: i16x8) -> i16x8;
    #[link_name = "llvm.x86.xop.vpmacsswd"]
    fn vpmacsswd(a: i16x8, b: i16x8, c: i32x4) -> i32x4;
    #[link_name = "llvm.x86.xop.vpmacswd"]
    fn vpmacswd(a: i16x8, b: i16x8, c: i32x4) -> i32x4;
    #[link_name = "llvm.x86.xop.vpmacssdd"]
    fn vpmacssdd(a: i32x4, b: i32x4, c: i32x4) -> i32x4;
    #[link_name = "llvm.x86.xop.vpmacsdd"]
    fn vpmacsdd(a: i32x4, b: i32x4, c: i32x4) -> i32x4;
    #[link_name = "llvm.x86.xop.vpmacssdql"]
    fn vpmacssdql(a: i32x4, b: i32x4, c: i64x2) -> i64x2;
    #[link_name = "llvm.x86.xop.vpmacsdql"]
    fn vpmacsdql(a: i32x4, b: i32x4, c: i64x2) -> i64x2;
    #[link_name = "llvm.x86.xop.vpmacssdqh"]
    fn vpmacssdqh(a: i32x4, b: i32x4, c: i64x2) -> i64x2;
    #[link_name = "llvm.x86.xop.vpmacsdqh"]
    fn vpmacsdqh(a: i32x4, b: i32x4, c: i64x2) -> i64x2;
    #[link_name = "llvm.x86.xop.vpmadcsswd"]
    fn vpmadcsswd(a: i16x8, b: i16x8, c: i32x4) -> i32x4;
    #[link_name = "llvm.x86.xop.vpmadcswd"]
    fn vpmadcswd(a: i16x8, b: i16x8, c: i32x4) -> i32x4;

    #[link_name = "llvm.x86.xop.vphaddbw"]
    fn vphaddbw(a: i8x16) -> i16x8;
    #[link_name = "llvm.x86.xop.vphaddbd"]
    fn vphaddbd(a: i8x16) -> i32x4;
    #[link_name = "llvm.x86.xop.vphaddbq"]
    fn vphaddbq(a: i8x16) -> i64x2;
    #[link_name = "llvm.x86.xop.vphaddwd"]
    fn vphaddwd(a: i16x8) -> i32x4;
    #[link_name = "llvm.x86.xop.vphaddwq"]
    fn vphaddwq(a: i16x8) -> i64x2;
    #[link_name = "llvm.x86.xop.vphadddq"]
    fn vphadddq(a: i32x4) -> i64x2;
    #[link_name = "llvm.x86.xop.vphaddubw"]
    fn vphaddubw(a: u8x16) -> u16x8;
    #[link_name = "llvm.x86.xop.vphaddubd"]
    fn vphaddubd(a: u8x16) -> u32x4;
    #[link_name = "llvm.x86.xop.vphaddubq"]
    fn vphaddubq(a: u8x16) -> u64x2;
    #[link_name = "llvm.x86.xop.vphadduwd"]
    fn vphadduwd(a: u16x8) -> u32x4;
    #[link_name = "llvm.x86.xop.vphadduwq"]
    fn vphadduwq(a: u16x8) -> u64x2;
    #[link_name = "llvm.x86.xop.vphaddudq"]
    fn vphaddudq(a: u32x4) -> u64x2;
    #[link_name = "llvm.x86.xop.vphsubbw"]
    fn vphsubbw(a: i8x16) -> i16x8;
    #[link_name = "llvm.x86.xop.vphsubwd"]
    fn vphsubwd(a: i16x8) -> i32x4;
    #[link_name = "llvm.x86.xop.vphsubdq"]
    fn vphsubdq(a: i32x4) -> i64x2;

    #[link_name = "llvm.x86.xop.vpperm"]
    fn vpperm(a: i8x16, b: i8x16, c: i8x16) -> i8x16;

    #[link_name = "llvm.x86.xop.vprotb"]
    fn vprotb(a: u8x16, b: u8x16) -> u8x16;
    #[link_name = "llvm.x86.xop.vprotw"]
    fn vprotw(a: u16x8, b: u16x8) -> u16x8;
    #[link_name = "llvm.x86.xop.vprotd"]
    fn vprotd(a: u32x4, b: u32x4) -> u32x4;
    #[link_name = "llvm.x86.xop.vprotq"]
    fn vprotq(a: u64x2, b: u64x2) -> u64x2;

    #[link_name = "llvm.x86.xop.vpshlb"]
    fn vpshlb(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.x86.xop.vpshlw"]
    fn vpshlw(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.x86.xop.vpshld"]
    fn vpshld(a: i32x4, b: i32x4) -> i32x4;
    #[link_name = "llvm.x86.xop.vpshlq"]
    fn vpshlq(a: i64x2, b: i64x2) -> i64x2;

    #[link_name = "llvm.x86.xop.vpshab"]
    fn vpshab(a: i8x16, b: i8x16) -> i8x16;
    #[link_name = "llvm.x86.xop.vpshaw"]
    fn vpshaw(a: i16x8, b: i16x8) -> i16x8;
    #[link_name = "llvm.x86.xop.vpshad"]
    fn vpshad(a: i32x4, b: i32x4) -> i32x4;
    #[link_name = "llvm.x86.xop.vpshaq"]
    fn vpshaq(a: i64x2, b: i64x2) -> i64x2;

    #[link_name = "llvm.x86.xop.vpermil2pd"]
    fn vpermil2pd_128(a: __m128d, b: __m128d, select: i64x2, imm8: i8) -> __m128d;
    #[link_name = "llvm.x86.xop.vpermil2pd.256"]
    fn vpermil2pd_256(a: __m256d, b: __m256d, select: i64x4, imm8: i8) -> __m256d;
    #[link_name = "llvm.x86.xop.vpermil2ps"]
    fn vpermil2ps_128(a: __m128, b: __m128, select: i32x4, imm8: i8) -> __m128;
    #[link_name = "llvm.x86.xop.vpermil2ps.256"]
    fn vpermil2ps_256(a: __m256, b: __m256, select: i32x8, imm8: i8) -> __m256;

    #[link_name = "llvm.x86.xop.vfrcz.pd"]
    fn vfrczpd_128(a: __m128d) -> __m128d;
    #[link_name = "llvm.x86.xop.vfrcz.pd.256"]
    fn vfrczpd_256(a: __m256d) -> __m256d;
    #[link_name = "llvm.x86.xop.vfrcz.ps"]
    fn vfrczps_128(a: __m128) -> __m128;
    #[link_name = "llvm.x86.xop.vfrcz.ps.256"]
    fn vfrczps_256(a: __m256) -> __m256;
    #[link_name = "llvm.x86.xop.vfrcz.sd"]
    fn vfrczsd(a: __m128d) -> __m128d;
    #[link_name = "llvm.x86.xop.vfrcz.ss"]
    fn vfrczss(a: __m128) -> __m128;

    #[link_name = "llvm.x86.xop.vpcomb"]
    fn vpcomb(a: i8x16, b: i8x16, imm8: i8) -> i8x16;
    #[link_name = "llvm.x86.xop.vpcomw"]
    fn vpcomw(a: i16x8, b: i16x8, imm8: i8) -> i16x8;
    #[link_name = "llvm.x86.xop.vpcomd"]
    fn vpcomd(a: i32x4, b: i32x4, imm8: i8) -> i32x4;
    #[link_name = "llvm.x86.xop.vpcomq"]
    fn vpcomq(a: i64x2, b: i64x2, imm8: i8) -> i64x2;
    #[link_name = "llvm.x86.xop.vpcomub"]
    fn vpcomub(a: u8x16, b: u8x16, imm8: i8) -> i8x16;
    #[link_name = "llvm.x86.xop.vpcomuw"]
    fn vpcomuw(a: u16x8, b: u16x8, imm8: i8) -> i16x8;
    #[link_name = "llvm.x86.xop.vpcomud"]
    fn vpcomud(a: u32x4, b: u32x4, imm8: i8) -> i32x4;
    #[link_name = "llvm.x86.xop.vpcomuq"]
    fn vpcomuq(a: u64x2, b: u64x2, imm8: i8) -> i64x2;

}

#[cfg(test)]
mod tests {
    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_maccs_epi16() {
        let a = _mm_setr_epi16(1, 2, 1, 2, 1, 2, 1, 2);
        let b = _mm_setr_epi16(3, 4, 3, 4, 3, 4, 3, 4);
        let c = _mm_setr_epi16(5, 6, 5, 6, 5, 6, 5, 6);
        let r = _mm_maccs_epi16(a, b, c);
        let e = _mm_setr_epi16(8, 14, 8, 14, 8, 14, 8, 14);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_macc_epi16() {
        let a = _mm_setr_epi16(1, 2, 1, 2, 1, 2, 1, 2);
        let b = _mm_setr_epi16(3, 4, 3, 4, 3, 4, 3, 4);
        let c = _mm_setr_epi16(5, 6, 5, 6, 5, 6, 5, 6);
        let r = _mm_macc_epi16(a, b, c);
        let e = _mm_setr_epi16(8, 14, 8, 14, 8, 14, 8, 14);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_maccsd_epi16() {
        let a = _mm_setr_epi16(1, 2, 1, 2, 0, 0, 0, 0);
        let b = _mm_setr_epi16(3, 4, 3, 4, 0, 0, 0, 0);
        let c = _mm_setr_epi32(5, 6, 5, 6);
        let r = _mm_maccsd_epi16(a, b, c);
        let e = _mm_setr_epi32(8, 14, 8, 14);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_maccd_epi16() {
        let a = _mm_setr_epi16(1, 2, 1, 2, 0, 0, 0, 0);
        let b = _mm_setr_epi16(3, 4, 3, 4, 0, 0, 0, 0);
        let c = _mm_setr_epi32(5, 6, 5, 6);
        let r = _mm_maccd_epi16(a, b, c);
        let e = _mm_setr_epi32(8, 14, 8, 14);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_maccs_epi32() {
        let a = _mm_setr_epi32(1, 2, 1, 2);
        let b = _mm_setr_epi32(3, 4, 3, 4);
        let c = _mm_setr_epi32(5, 6, 5, 6);
        let r = _mm_maccs_epi32(a, b, c);
        let e = _mm_setr_epi32(8, 14, 8, 14);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_macc_epi32() {
        let a = _mm_setr_epi32(1, 2, 1, 2);
        let b = _mm_setr_epi32(3, 4, 3, 4);
        let c = _mm_setr_epi32(5, 6, 5, 6);
        let r = _mm_macc_epi32(a, b, c);
        let e = _mm_setr_epi32(8, 14, 8, 14);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_maccslo_epi32() {
        let a = _mm_setr_epi32(1, 2, 1, 2);
        let b = _mm_setr_epi32(3, 4, 3, 4);
        let c = _mm_setr_epi64x(5, 6);
        let r = _mm_maccslo_epi32(a, b, c);
        let e = _mm_setr_epi64x(8, 9);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_macclo_epi32() {
        let a = _mm_setr_epi32(1, 2, 1, 2);
        let b = _mm_setr_epi32(3, 4, 3, 4);
        let c = _mm_setr_epi64x(5, 6);
        let r = _mm_macclo_epi32(a, b, c);
        let e = _mm_setr_epi64x(8, 9);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_maccshi_epi32() {
        let a = _mm_setr_epi32(1, 2, 1, 2);
        let b = _mm_setr_epi32(3, 4, 3, 4);
        let c = _mm_setr_epi64x(5, 6);
        let r = _mm_maccshi_epi32(a, b, c);
        let e = _mm_setr_epi64x(13, 14);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_macchi_epi32() {
        let a = _mm_setr_epi32(1, 2, 1, 2);
        let b = _mm_setr_epi32(3, 4, 3, 4);
        let c = _mm_setr_epi64x(5, 6);
        let r = _mm_macchi_epi32(a, b, c);
        let e = _mm_setr_epi64x(13, 14);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_maddsd_epi16() {
        let a = _mm_setr_epi16(1, 2, 1, 2, 1, 2, 1, 2);
        let b = _mm_setr_epi16(3, 4, 3, 4, 3, 4, 3, 4);
        let c = _mm_setr_epi32(5, 6, 5, 6);
        let r = _mm_maddsd_epi16(a, b, c);
        let e = _mm_setr_epi32(16, 17, 16, 17);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_maddd_epi16() {
        let a = _mm_setr_epi16(1, 2, 1, 2, 1, 2, 1, 2);
        let b = _mm_setr_epi16(3, 4, 3, 4, 3, 4, 3, 4);
        let c = _mm_setr_epi32(5, 6, 5, 6);
        let r = _mm_maddd_epi16(a, b, c);
        let e = _mm_setr_epi32(16, 17, 16, 17);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_haddw_epi8() {
        let a = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm_haddw_epi8(a);
        let e = _mm_setr_epi16(3, 7, 11, 15, 19, 23, 27, 31);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_haddd_epi8() {
        let a = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm_haddd_epi8(a);
        let e = _mm_setr_epi32(10, 26, 42, 58);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_haddq_epi8() {
        let a = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm_haddq_epi8(a);
        let e = _mm_setr_epi64x(36, 100);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_haddd_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm_haddd_epi16(a);
        let e = _mm_setr_epi32(3, 7, 11, 15);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_haddq_epi16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm_haddq_epi16(a);
        let e = _mm_setr_epi64x(10, 26);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_haddq_epi32() {
        let a = _mm_setr_epi32(1, 2, 3, 4);
        let r = _mm_haddq_epi32(a);
        let e = _mm_setr_epi64x(3, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_haddw_epu8() {
        let a = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm_haddw_epu8(a);
        let e = _mm_setr_epi16(3, 7, 11, 15, 19, 23, 27, 31);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_haddd_epu8() {
        let a = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm_haddd_epu8(a);
        let e = _mm_setr_epi32(10, 26, 42, 58);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_haddq_epu8() {
        let a = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = _mm_haddq_epu8(a);
        let e = _mm_setr_epi64x(36, 100);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_haddd_epu16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm_haddd_epu16(a);
        let e = _mm_setr_epi32(3, 7, 11, 15);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_haddq_epu16() {
        let a = _mm_setr_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let r = _mm_haddq_epu16(a);
        let e = _mm_setr_epi64x(10, 26);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_haddq_epu32() {
        let a = _mm_setr_epi32(1, 2, 3, 4);
        let r = _mm_haddq_epu32(a);
        let e = _mm_setr_epi64x(3, 7);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_hsubw_epi8() {
        let a = _mm_setr_epi8(1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8, -8);
        let r = _mm_hsubw_epi8(a);
        let e = _mm_setr_epi16(2, 4, 6, 8, 10, 12, 14, 16);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_hsubd_epi16() {
        let a = _mm_setr_epi16(1, -1, 2, -2, 3, -3, 4, -4);
        let r = _mm_hsubd_epi16(a);
        let e = _mm_setr_epi32(2, 4, 6, 8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_hsubq_epi32() {
        let a = _mm_setr_epi32(1, -1, 2, -2);
        let r = _mm_hsubq_epi32(a);
        let e = _mm_setr_epi64x(2, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_cmov_si128() {
        let a = _mm_setr_epi32(1, 2, 3, 4);
        let b = _mm_setr_epi32(5, 6, 7, 8);
        let mask = _mm_setr_epi32(-1, 0, -1, 0);
        let r = _mm_cmov_si128(a, b, mask);
        let e = _mm_setr_epi32(1, 6, 3, 8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm256_cmov_si256() {
        let a = _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let b = _mm256_setr_epi32(9, 10, 11, 12, 13, 14, 15, 16);
        let mask = _mm256_setr_epi32(-1, 0, -1, 0, -1, 0, -1, 0);
        let r = _mm256_cmov_si256(a, b, mask);
        let e = _mm256_setr_epi32(1, 10, 3, 12, 5, 14, 7, 16);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_rot_epi8() {
        let a = _mm_set1_epi8(0b0101_1100);
        let b = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8);
        let r = _mm_rot_epi8(a, b);
        let e = _mm_setr_epi8(
            0b10111000_u8 as i8,
            0b01110001,
            0b11100010_u8 as i8,
            0b11000101_u8 as i8,
            0b10001011_u8 as i8,
            0b00010111,
            0b00101110,
            0b01011100,
            0b00101110,
            0b00010111,
            0b10001011_u8 as i8,
            0b11000101_u8 as i8,
            0b11100010_u8 as i8,
            0b01110001,
            0b10111000_u8 as i8,
            0b01011100,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_rot_epi16() {
        let a = _mm_set1_epi16(0b1110011001010111_u16 as i16);
        let b = _mm_setr_epi16(1, 2, 3, 4, -1, -2, -3, -4);
        let r = _mm_rot_epi16(a, b);
        let e = _mm_setr_epi16(
            0b1100110010101111_u16 as i16,
            0b1001100101011111_u16 as i16,
            0b0011001010111111,
            0b0110010101111110,
            0b1111001100101011_u16 as i16,
            0b1111100110010101_u16 as i16,
            0b1111110011001010_u16 as i16,
            0b0111111001100101,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_rot_epi32() {
        let a = _mm_set1_epi32(0xff00ff00_u32 as i32);
        let b = _mm_setr_epi32(1, 2, -1, -2);
        let r = _mm_rot_epi32(a, b);
        let e = _mm_setr_epi32(
            0xfe01fe01_u32 as i32,
            0xfc03fc03_u32 as i32,
            0x7f807f80,
            0x3fb03fb0,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_rot_epi64() {
        let a = _mm_set1_epi64x(0xff00ff00_ff00ff00_u64 as i64);
        let b = _mm_setr_epi64x(1, -1);
        let r = _mm_rot_epi64(a, b);
        let e = _mm_setr_epi64x(
            0xfe01fe01_fe01fe01_u64 as i64,
            0x7f807f80_7f807f80_u64 as i64,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_roti_epi8() {
        let a = _mm_set1_epi8(0b10100011_u8 as i8);
        let r = _mm_roti_epi8::<3>(a);
        let e = _mm_set1_epi8(0b00011101);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_roti_epi16() {
        let a = _mm_set1_epi16(0b1010001100100011_u16 as i16);
        let r = _mm_roti_epi16::<3>(a);
        let e = _mm_set1_epi16(0b0001100100011101_u16 as i16);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_roti_epi32() {
        let a = _mm_set1_epi32(0xff00ff00_u32 as i32);
        let r = _mm_roti_epi32::<3>(a);
        let e = _mm_set1_epi32(0xf807f807_u32 as i32);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_roti_epi64() {
        let a = _mm_set1_epi64x(0xff00ff00_ff00ff00_u64 as i64);
        let r = _mm_roti_epi64::<3>(a);
        let e = _mm_set1_epi64x(0xf807f807_f807f807_u64 as i64);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_shl_epi8() {
        let a = _mm_set1_epi8(-1);
        let b = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8);
        let r = _mm_shl_epi8(a, b);
        let e = _mm_setr_epi8(
            -2, -4, -8, -16, -32, -64, -128, -1, 127, 63, 31, 15, 7, 3, 1, -1,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_shl_epi16() {
        let a = _mm_set1_epi16(-1);
        let b = _mm_setr_epi16(1, 2, 3, 4, -1, -2, -3, -4);
        let r = _mm_shl_epi16(a, b);
        let e = _mm_setr_epi16(-2, -4, -8, -16, 0x7fff, 0x3fff, 0x1fff, 0x0fff);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_shl_epi32() {
        let a = _mm_set1_epi32(-1);
        let b = _mm_setr_epi32(1, 2, -1, -2);
        let r = _mm_shl_epi32(a, b);
        let e = _mm_setr_epi32(-2, -4, 0x7fffffff, 0x3fffffff);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_shl_epi64() {
        let a = _mm_set1_epi64x(-1);
        let b = _mm_setr_epi64x(1, -1);
        let r = _mm_shl_epi64(a, b);
        let e = _mm_setr_epi64x(-2, i64::MAX);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_sha_epi8() {
        let a = _mm_set1_epi8(-1);
        let b = _mm_setr_epi8(1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8);
        let r = _mm_sha_epi8(a, b);
        let e = _mm_setr_epi8(
            -2, -4, -8, -16, -32, -64, -128, -1, -1, -1, -1, -1, -1, -1, -1, -1,
        );
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_sha_epi16() {
        let a = _mm_set1_epi16(-1);
        let b = _mm_setr_epi16(1, 2, 3, 4, -1, -2, -3, -4);
        let r = _mm_sha_epi16(a, b);
        let e = _mm_setr_epi16(-2, -4, -8, -16, -1, -1, -1, -1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_sha_epi32() {
        let a = _mm_set1_epi32(-1);
        let b = _mm_setr_epi32(1, 2, -1, -2);
        let r = _mm_sha_epi32(a, b);
        let e = _mm_setr_epi32(-2, -4, -1, -1);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "xop")]
    unsafe fn test_mm_sha_epi64() {
        let a = _mm_set1_epi64x(-1);
        let b = _mm_setr_epi64x(1, -1);
        let r = _mm_sha_epi64(a, b);
        let e = _mm_setr_epi64x(-2, -1);
        assert_eq_m128i(r, e);
    }
}
