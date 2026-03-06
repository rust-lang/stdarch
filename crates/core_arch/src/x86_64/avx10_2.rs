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

/// Moves 16 bytes from the source to the destination, with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsb))]
pub unsafe fn _mm_loadrs_epi8(p: *const i8) -> __m128i {
    vmovrsb128(p).as_m128i()
}

/// Moves 16 bytes from the source to the destination using writemask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsb))]
pub unsafe fn _mm_mask_loadrs_epi8(src: __m128i, k: __mmask16, p: *const i8) -> __m128i {
    _mm_mask_mov_epi8(src, k, _mm_loadrs_epi8(p))
}

/// Moves 16 bytes from the source to the destination using zeromask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsb))]
pub unsafe fn _mm_maskz_loadrs_epi8(k: __mmask16, p: *const i8) -> __m128i {
    _mm_maskz_mov_epi8(k, _mm_loadrs_epi8(p))
}

/// Moves 32 bytes from the source to the destination, with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsb))]
pub unsafe fn _mm256_loadrs_epi8(p: *const i8) -> __m256i {
    vmovrsb256(p).as_m256i()
}

/// Moves 32 bytes from the source to the destination using writemask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsb))]
pub unsafe fn _mm256_mask_loadrs_epi8(src: __m256i, k: __mmask32, p: *const i8) -> __m256i {
    _mm256_mask_mov_epi8(src, k, _mm256_loadrs_epi8(p))
}

/// Moves 32 bytes from the source to the destination using zeromask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsb))]
pub unsafe fn _mm256_maskz_loadrs_epi8(k: __mmask32, p: *const i8) -> __m256i {
    _mm256_maskz_mov_epi8(k, _mm256_loadrs_epi8(p))
}

/// Moves 64 bytes from the source to the destination, with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsb))]
pub unsafe fn _mm512_loadrs_epi8(p: *const i8) -> __m512i {
    vmovrsb512(p).as_m512i()
}

/// Moves 64 bytes from the source to the destination using writemask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsb))]
pub unsafe fn _mm512_mask_loadrs_epi8(src: __m512i, k: __mmask64, p: *const i8) -> __m512i {
    _mm512_mask_mov_epi8(src, k, _mm512_loadrs_epi8(p))
}

/// Moves 64 bytes from the source to the destination using zeromask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsb))]
pub unsafe fn _mm512_maskz_loadrs_epi8(k: __mmask64, p: *const i8) -> __m512i {
    _mm512_maskz_mov_epi8(k, _mm512_loadrs_epi8(p))
}

/// Moves 8 16-bit words from the source to the destination, with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsw))]
pub unsafe fn _mm_loadrs_epi16(p: *const i16) -> __m128i {
    vmovrsw128(p).as_m128i()
}

/// Moves 8 16-bit words from the source to the destination using writemask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsw))]
pub unsafe fn _mm_mask_loadrs_epi16(src: __m128i, k: __mmask8, p: *const i16) -> __m128i {
    _mm_mask_mov_epi16(src, k, _mm_loadrs_epi16(p))
}

/// Moves 8 16-bit words from the source to the destination using zeromask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsw))]
pub unsafe fn _mm_maskz_loadrs_epi16(k: __mmask8, p: *const i16) -> __m128i {
    _mm_maskz_mov_epi16(k, _mm_loadrs_epi16(p))
}

/// Moves 16 16-bit words from the source to the destination, with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsw))]
pub unsafe fn _mm256_loadrs_epi16(p: *const i16) -> __m256i {
    vmovrsw256(p).as_m256i()
}

/// Moves 16 16-bit words from the source to the destination using writemask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsw))]
pub unsafe fn _mm256_mask_loadrs_epi16(src: __m256i, k: __mmask16, p: *const i16) -> __m256i {
    _mm256_mask_mov_epi16(src, k, _mm256_loadrs_epi16(p))
}

/// Moves 16 16-bit words from the source to the destination using zeromask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsw))]
pub unsafe fn _mm256_maskz_loadrs_epi16(k: __mmask16, p: *const i16) -> __m256i {
    _mm256_maskz_mov_epi16(k, _mm256_loadrs_epi16(p))
}

/// Moves 32 16-bit words from the source to the destination, with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsw))]
pub unsafe fn _mm512_loadrs_epi16(p: *const i16) -> __m512i {
    vmovrsw512(p).as_m512i()
}

/// Moves 32 16-bit words from the source to the destination using writemask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsw))]
pub unsafe fn _mm512_mask_loadrs_epi16(src: __m512i, k: __mmask32, p: *const i16) -> __m512i {
    _mm512_mask_mov_epi16(src, k, _mm512_loadrs_epi16(p))
}

/// Moves 32 16-bit words from the source to the destination using zeromask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsw))]
pub unsafe fn _mm512_maskz_loadrs_epi16(k: __mmask32, p: *const i16) -> __m512i {
    _mm512_maskz_mov_epi16(k, _mm512_loadrs_epi16(p))
}

/// Moves 4 32-bit doublewords from the source to the destination, with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsd))]
pub unsafe fn _mm_loadrs_epi32(p: *const i32) -> __m128i {
    vmovrsd128(p).as_m128i()
}

/// Moves 4 32-bit doublewords from the source to the destination using writemask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsd))]
pub unsafe fn _mm_mask_loadrs_epi32(src: __m128i, k: __mmask8, p: *const i32) -> __m128i {
    _mm_mask_mov_epi32(src, k, _mm_loadrs_epi32(p))
}

/// Moves 4 32-bit doublewords from the source to the destination using zeromask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsd))]
pub unsafe fn _mm_maskz_loadrs_epi32(k: __mmask8, p: *const i32) -> __m128i {
    _mm_maskz_mov_epi32(k, _mm_loadrs_epi32(p))
}

/// Moves 8 32-bit doublewords from the source to the destination, with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsd))]
pub unsafe fn _mm256_loadrs_epi32(p: *const i32) -> __m256i {
    vmovrsd256(p).as_m256i()
}

/// Moves 8 32-bit doublewords from the source to the destination using writemask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsd))]
pub unsafe fn _mm256_mask_loadrs_epi32(src: __m256i, k: __mmask8, p: *const i32) -> __m256i {
    _mm256_mask_mov_epi32(src, k, _mm256_loadrs_epi32(p))
}

/// Moves 8 32-bit doublewords from the source to the destination using zeromask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsd))]
pub unsafe fn _mm256_maskz_loadrs_epi32(k: __mmask8, p: *const i32) -> __m256i {
    _mm256_maskz_mov_epi32(k, _mm256_loadrs_epi32(p))
}

/// Moves 16 32-bit doublewords from the source to the destination, with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsd))]
pub unsafe fn _mm512_loadrs_epi32(p: *const i32) -> __m512i {
    vmovrsd512(p).as_m512i()
}

/// Moves 16 32-bit doublewords from the source to the destination using writemask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsd))]
pub unsafe fn _mm512_mask_loadrs_epi32(src: __m512i, k: __mmask16, p: *const i32) -> __m512i {
    _mm512_mask_mov_epi32(src, k, _mm512_loadrs_epi32(p))
}

/// Moves 16 32-bit doublewords from the source to the destination using zeromask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsd))]
pub unsafe fn _mm512_maskz_loadrs_epi32(k: __mmask16, p: *const i32) -> __m512i {
    _mm512_maskz_mov_epi32(k, _mm512_loadrs_epi32(p))
}

/// Moves 2 64-bit quadwords from the source to the destination, with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsq))]
pub unsafe fn _mm_loadrs_epi64(p: *const i64) -> __m128i {
    vmovrsq128(p).as_m128i()
}

/// Moves 2 64-bit quadwords from the source to the destination using writemask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsq))]
pub unsafe fn _mm_mask_loadrs_epi64(src: __m128i, k: __mmask8, p: *const i64) -> __m128i {
    _mm_mask_mov_epi64(src, k, _mm_loadrs_epi64(p))
}

/// Moves 2 64-bit quadwords from the source to the destination using zeromask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsq))]
pub unsafe fn _mm_maskz_loadrs_epi64(k: __mmask8, p: *const i64) -> __m128i {
    _mm_maskz_mov_epi64(k, _mm_loadrs_epi64(p))
}

/// Moves 4 64-bit quadwords from the source to the destination, with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsq))]
pub unsafe fn _mm256_loadrs_epi64(p: *const i64) -> __m256i {
    vmovrsq256(p).as_m256i()
}

/// Moves 4 64-bit quadwords from the source to the destination using writemask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsq))]
pub unsafe fn _mm256_mask_loadrs_epi64(src: __m256i, k: __mmask8, p: *const i64) -> __m256i {
    _mm256_mask_mov_epi64(src, k, _mm256_loadrs_epi64(p))
}

/// Moves 4 64-bit quadwords from the source to the destination using zeromask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsq))]
pub unsafe fn _mm256_maskz_loadrs_epi64(k: __mmask8, p: *const i64) -> __m256i {
    _mm256_maskz_mov_epi64(k, _mm256_loadrs_epi64(p))
}

/// Moves 8 64-bit quadwords from the source to the destination, with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsq))]
pub unsafe fn _mm512_loadrs_epi64(p: *const i64) -> __m512i {
    vmovrsq512(p).as_m512i()
}

/// Moves 8 64-bit quadwords from the source to the destination using writemask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsq))]
pub unsafe fn _mm512_mask_loadrs_epi64(src: __m512i, k: __mmask8, p: *const i64) -> __m512i {
    _mm512_mask_mov_epi64(src, k, _mm512_loadrs_epi64(p))
}

/// Moves 8 64-bit quadwords from the source to the destination using zeromask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set), with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs,avx10.2")]
#[unstable(feature = "stdarch_x86_avx10_2", issue = "153417")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(vmovrsq))]
pub unsafe fn _mm512_maskz_loadrs_epi64(k: __mmask8, p: *const i64) -> __m512i {
    _mm512_maskz_mov_epi64(k, _mm512_loadrs_epi64(p))
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

    #[link_name = "llvm.x86.avx10.vmovrsb128"]
    fn vmovrsb128(p: *const i8) -> i8x16;
    #[link_name = "llvm.x86.avx10.vmovrsb256"]
    fn vmovrsb256(p: *const i8) -> i8x32;
    #[link_name = "llvm.x86.avx10.vmovrsb512"]
    fn vmovrsb512(p: *const i8) -> i8x64;

    #[link_name = "llvm.x86.avx10.vmovrsw128"]
    fn vmovrsw128(p: *const i16) -> i16x8;
    #[link_name = "llvm.x86.avx10.vmovrsw256"]
    fn vmovrsw256(p: *const i16) -> i16x16;
    #[link_name = "llvm.x86.avx10.vmovrsw512"]
    fn vmovrsw512(p: *const i16) -> i16x32;

    #[link_name = "llvm.x86.avx10.vmovrsd128"]
    fn vmovrsd128(p: *const i32) -> i32x4;
    #[link_name = "llvm.x86.avx10.vmovrsd256"]
    fn vmovrsd256(p: *const i32) -> i32x8;
    #[link_name = "llvm.x86.avx10.vmovrsd512"]
    fn vmovrsd512(p: *const i32) -> i32x16;

    #[link_name = "llvm.x86.avx10.vmovrsq128"]
    fn vmovrsq128(p: *const i64) -> i64x2;
    #[link_name = "llvm.x86.avx10.vmovrsq256"]
    fn vmovrsq256(p: *const i64) -> i64x4;
    #[link_name = "llvm.x86.avx10.vmovrsq512"]
    fn vmovrsq512(p: *const i64) -> i64x8;
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

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm_loadrs_epi8() {
        let data = _mm_set_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = unsafe { _mm_loadrs_epi8((&raw const data).cast()) };
        assert_eq_m128i(r, data);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm_mask_loadrs_epi8() {
        let data = _mm_set_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let src = _mm_set_epi8(
            65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        );
        let k = 0b0101010101010101;
        let r = unsafe { _mm_mask_loadrs_epi8(src, k, (&raw const data).cast()) };
        let e = _mm_set_epi8(65, 2, 67, 4, 69, 6, 71, 8, 73, 10, 75, 12, 77, 14, 79, 16);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm_maskz_loadrs_epi8() {
        let data = _mm_set_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let k = 0b0101010101010101;
        let r = unsafe { _mm_maskz_loadrs_epi8(k, (&raw const data).cast()) };
        let e = _mm_set_epi8(0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12, 0, 14, 0, 16);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm256_loadrs_epi8() {
        let data = _mm256_set_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = unsafe { _mm256_loadrs_epi8((&raw const data).cast()) };
        assert_eq_m256i(r, data);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm256_mask_loadrs_epi8() {
        let data = _mm256_set_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let src = _mm256_set_epi8(
            65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
            87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
        );
        let k = 0b01010101010101010101010101010101;
        let r = unsafe { _mm256_mask_loadrs_epi8(src, k, (&raw const data).cast()) };
        let e = _mm256_set_epi8(
            65, 2, 67, 4, 69, 6, 71, 8, 73, 10, 75, 12, 77, 14, 79, 16, 81, 18, 83, 20, 85, 22, 87,
            24, 89, 26, 91, 28, 93, 30, 95, 32,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm256_maskz_loadrs_epi8() {
        let data = _mm256_set_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let k = 0b01010101010101010101010101010101;
        let r = unsafe { _mm256_maskz_loadrs_epi8(k, (&raw const data).cast()) };
        let e = _mm256_set_epi8(
            0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12, 0, 14, 0, 16, 0, 18, 0, 20, 0, 22, 0, 24, 0, 26,
            0, 28, 0, 30, 0, 32,
        );
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm512_loadrs_epi8() {
        let data = _mm512_set_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        );
        let r = unsafe { _mm512_loadrs_epi8((&raw const data).cast()) };
        assert_eq_m512i(r, data);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm512_mask_loadrs_epi8() {
        let data = _mm512_set_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        );
        let src = _mm512_set_epi8(
            65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
            87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
            107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123,
            124, 125, 126, 127, -128,
        );
        let k = 0b0101010101010101010101010101010101010101010101010101010101010101;
        let r = unsafe { _mm512_mask_loadrs_epi8(src, k, (&raw const data).cast()) };
        let e = _mm512_set_epi8(
            65, 2, 67, 4, 69, 6, 71, 8, 73, 10, 75, 12, 77, 14, 79, 16, 81, 18, 83, 20, 85, 22, 87,
            24, 89, 26, 91, 28, 93, 30, 95, 32, 97, 34, 99, 36, 101, 38, 103, 40, 105, 42, 107, 44,
            109, 46, 111, 48, 113, 50, 115, 52, 117, 54, 119, 56, 121, 58, 123, 60, 125, 62, 127,
            64,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm512_maskz_loadrs_epi8() {
        let data = _mm512_set_epi8(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
        );
        let k = 0b0101010101010101010101010101010101010101010101010101010101010101;
        let r = unsafe { _mm512_maskz_loadrs_epi8(k, (&raw const data).cast()) };
        let e = _mm512_set_epi8(
            0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12, 0, 14, 0, 16, 0, 18, 0, 20, 0, 22, 0, 24, 0, 26,
            0, 28, 0, 30, 0, 32, 0, 34, 0, 36, 0, 38, 0, 40, 0, 42, 0, 44, 0, 46, 0, 48, 0, 50, 0,
            52, 0, 54, 0, 56, 0, 58, 0, 60, 0, 62, 0, 64,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm_loadrs_epi16() {
        let data = _mm_set_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let r = unsafe { _mm_loadrs_epi16((&raw const data).cast()) };
        assert_eq_m128i(r, data);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm_mask_loadrs_epi16() {
        let data = _mm_set_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let src = _mm_set_epi16(65, 66, 67, 68, 69, 70, 71, 72);
        let k = 0b01010101;
        let r = unsafe { _mm_mask_loadrs_epi16(src, k, (&raw const data).cast()) };
        let e = _mm_set_epi16(65, 2, 67, 4, 69, 6, 71, 8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm_maskz_loadrs_epi16() {
        let data = _mm_set_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let k = 0b01010101;
        let r = unsafe { _mm_maskz_loadrs_epi16(k, (&raw const data).cast()) };
        let e = _mm_set_epi16(0, 2, 0, 4, 0, 6, 0, 8);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm256_loadrs_epi16() {
        let data = _mm256_set_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = unsafe { _mm256_loadrs_epi16((&raw const data).cast()) };
        assert_eq_m256i(r, data);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm256_mask_loadrs_epi16() {
        let data = _mm256_set_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let src = _mm256_set_epi16(
            65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        );
        let k = 0b0101010101010101;
        let r = unsafe { _mm256_mask_loadrs_epi16(src, k, (&raw const data).cast()) };
        let e = _mm256_set_epi16(65, 2, 67, 4, 69, 6, 71, 8, 73, 10, 75, 12, 77, 14, 79, 16);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm256_maskz_loadrs_epi16() {
        let data = _mm256_set_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let k = 0b0101010101010101;
        let r = unsafe { _mm256_maskz_loadrs_epi16(k, (&raw const data).cast()) };
        let e = _mm256_set_epi16(0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12, 0, 14, 0, 16);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm512_loadrs_epi16() {
        let data = _mm512_set_epi16(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = unsafe { _mm512_loadrs_epi16((&raw const data).cast()) };
        assert_eq_m512i(r, data);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm512_mask_loadrs_epi16() {
        let data = _mm512_set_epi16(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let src = _mm512_set_epi16(
            65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
            87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
        );
        let k = 0b01010101010101010101010101010101;
        let r = unsafe { _mm512_mask_loadrs_epi16(src, k, (&raw const data).cast()) };
        let e = _mm512_set_epi16(
            65, 2, 67, 4, 69, 6, 71, 8, 73, 10, 75, 12, 77, 14, 79, 16, 81, 18, 83, 20, 85, 22, 87,
            24, 89, 26, 91, 28, 93, 30, 95, 32,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm512_maskz_loadrs_epi16() {
        let data = _mm512_set_epi16(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        );
        let k = 0b01010101010101010101010101010101;
        let r = unsafe { _mm512_maskz_loadrs_epi16(k, (&raw const data).cast()) };
        let e = _mm512_set_epi16(
            0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12, 0, 14, 0, 16, 0, 18, 0, 20, 0, 22, 0, 24, 0, 26,
            0, 28, 0, 30, 0, 32,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm_loadrs_epi32() {
        let data = _mm_set_epi32(1, 2, 3, 4);
        let r = unsafe { _mm_loadrs_epi32((&raw const data).cast()) };
        assert_eq_m128i(r, data);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm_mask_loadrs_epi32() {
        let data = _mm_set_epi32(1, 2, 3, 4);
        let src = _mm_set_epi32(65, 66, 67, 68);
        let k = 0b0101;
        let r = unsafe { _mm_mask_loadrs_epi32(src, k, (&raw const data).cast()) };
        let e = _mm_set_epi32(65, 2, 67, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm_maskz_loadrs_epi32() {
        let data = _mm_set_epi32(1, 2, 3, 4);
        let k = 0b0101;
        let r = unsafe { _mm_maskz_loadrs_epi32(k, (&raw const data).cast()) };
        let e = _mm_set_epi32(0, 2, 0, 4);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm256_loadrs_epi32() {
        let data = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let r = unsafe { _mm256_loadrs_epi32((&raw const data).cast()) };
        assert_eq_m256i(r, data);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm256_mask_loadrs_epi32() {
        let data = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let src = _mm256_set_epi32(65, 66, 67, 68, 69, 70, 71, 72);
        let k = 0b01010101;
        let r = unsafe { _mm256_mask_loadrs_epi32(src, k, (&raw const data).cast()) };
        let e = _mm256_set_epi32(65, 2, 67, 4, 69, 6, 71, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm256_maskz_loadrs_epi32() {
        let data = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let k = 0b01010101;
        let r = unsafe { _mm256_maskz_loadrs_epi32(k, (&raw const data).cast()) };
        let e = _mm256_set_epi32(0, 2, 0, 4, 0, 6, 0, 8);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm512_loadrs_epi32() {
        let data = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = unsafe { _mm512_loadrs_epi32((&raw const data).cast()) };
        assert_eq_m512i(r, data);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm512_mask_loadrs_epi32() {
        let data = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let src = _mm512_set_epi32(
            65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80,
        );
        let k = 0b0101010101010101;
        let r = unsafe { _mm512_mask_loadrs_epi32(src, k, (&raw const data).cast()) };
        let e = _mm512_set_epi32(65, 2, 67, 4, 69, 6, 71, 8, 73, 10, 75, 12, 77, 14, 79, 16);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm512_maskz_loadrs_epi32() {
        let data = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let k = 0b0101010101010101;
        let r = unsafe { _mm512_maskz_loadrs_epi32(k, (&raw const data).cast()) };
        let e = _mm512_set_epi32(0, 2, 0, 4, 0, 6, 0, 8, 0, 10, 0, 12, 0, 14, 0, 16);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm_loadrs_epi64() {
        let data = _mm_set_epi64x(1, 2);
        let r = unsafe { _mm_loadrs_epi64((&raw const data).cast()) };
        assert_eq_m128i(r, data);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm_mask_loadrs_epi64() {
        let data = _mm_set_epi64x(1, 2);
        let src = _mm_set_epi64x(65, 66);
        let k = 0b01;
        let r = unsafe { _mm_mask_loadrs_epi64(src, k, (&raw const data).cast()) };
        let e = _mm_set_epi64x(65, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm_maskz_loadrs_epi64() {
        let data = _mm_set_epi64x(1, 2);
        let k = 0b01;
        let r = unsafe { _mm_maskz_loadrs_epi64(k, (&raw const data).cast()) };
        let e = _mm_set_epi64x(0, 2);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm256_loadrs_epi64() {
        let data = _mm256_set_epi64x(1, 2, 3, 4);
        let r = unsafe { _mm256_loadrs_epi64((&raw const data).cast()) };
        assert_eq_m256i(r, data);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm256_mask_loadrs_epi64() {
        let data = _mm256_set_epi64x(1, 2, 3, 4);
        let src = _mm256_set_epi64x(65, 66, 67, 68);
        let k = 0b0101;
        let r = unsafe { _mm256_mask_loadrs_epi64(src, k, (&raw const data).cast()) };
        let e = _mm256_set_epi64x(65, 2, 67, 4);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm256_maskz_loadrs_epi64() {
        let data = _mm256_set_epi64x(1, 2, 3, 4);
        let k = 0b0101;
        let r = unsafe { _mm256_maskz_loadrs_epi64(k, (&raw const data).cast()) };
        let e = _mm256_set_epi64x(0, 2, 0, 4);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm512_loadrs_epi64() {
        let data = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = unsafe { _mm512_loadrs_epi64((&raw const data).cast()) };
        assert_eq_m512i(r, data);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm512_mask_loadrs_epi64() {
        let data = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let src = _mm512_set_epi64(65, 66, 67, 68, 69, 70, 71, 72);
        let k = 0b01010101;
        let r = unsafe { _mm512_mask_loadrs_epi64(src, k, (&raw const data).cast()) };
        let e = _mm512_set_epi64(65, 2, 67, 4, 69, 6, 71, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm512_maskz_loadrs_epi64() {
        let data = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let k = 0b01010101;
        let r = unsafe { _mm512_maskz_loadrs_epi64(k, (&raw const data).cast()) };
        let e = _mm512_set_epi64(0, 2, 0, 4, 0, 6, 0, 8);
        assert_eq_m512i(r, e);
    }
}
