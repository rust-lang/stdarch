//! Read-shared Move instructions

use crate::core_arch::{simd::*, x86::*};

#[cfg(test)]
use stdarch_test::assert_instr;

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.x86.movrsqi"]
    fn movrsqi(src: *const i8) -> i8;
    #[link_name = "llvm.x86.movrshi"]
    fn movrshi(src: *const i16) -> i16;
    #[link_name = "llvm.x86.movrssi"]
    fn movrssi(src: *const i32) -> i32;
    #[link_name = "llvm.x86.movrsdi"]
    fn movrsdi(src: *const i64) -> i64;

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

/// Moves a byte from the source to the destination, with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(movrs))]
#[unstable(feature = "movrs_target_feature", issue = "137976")]
pub unsafe fn _movrs_i8(src: *const i8) -> i8 {
    movrsqi(src)
}

/// Moves a 16-bit word from the source to the destination, with an indication that the source memory
/// location is likely to become read-shared by multiple processors, i.e., read in the future by at
/// least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(movrs))]
#[unstable(feature = "movrs_target_feature", issue = "137976")]
pub unsafe fn _movrs_i16(src: *const i16) -> i16 {
    movrshi(src)
}

/// Moves a 32-bit doubleword from the source to the destination, with an indication that the source
/// memory location is likely to become read-shared by multiple processors, i.e., read in the future
/// by at least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(movrs))]
#[unstable(feature = "movrs_target_feature", issue = "137976")]
pub unsafe fn _movrs_i32(src: *const i32) -> i32 {
    movrssi(src)
}

/// Moves a 64-bit quadword from the source to the destination, with an indication that the source
/// memory location is likely to become read-shared by multiple processors, i.e., read in the future
/// by at least one other processor before it is written, assuming it is ever written in the future.
#[inline]
#[target_feature(enable = "movrs")]
#[cfg_attr(all(test, not(target_vendor = "apple")), assert_instr(movrs))]
#[unstable(feature = "movrs_target_feature", issue = "137976")]
pub unsafe fn _movrs_i64(src: *const i64) -> i64 {
    movrsdi(src)
}

// FIXME: add masked versions of movrs, currently LLVM doesn't support them

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

#[cfg(test)]
mod tests {
    use stdarch_test::simd_test;

    use super::*;

    #[simd_test(enable = "movrs")]
    fn test_movrs_i8() {
        let x: i8 = 42;
        let y = unsafe { _movrs_i8(&x) };
        assert_eq!(x, y);
    }

    #[simd_test(enable = "movrs")]
    fn test_movrs_i16() {
        let x: i16 = 42;
        let y = unsafe { _movrs_i16(&x) };
        assert_eq!(x, y);
    }

    #[simd_test(enable = "movrs")]
    fn test_movrs_i32() {
        let x: i32 = 42;
        let y = unsafe { _movrs_i32(&x) };
        assert_eq!(x, y);
    }

    #[simd_test(enable = "movrs")]
    fn test_movrs_i64() {
        let x: i64 = 42;
        let y = unsafe { _movrs_i64(&x) };
        assert_eq!(x, y);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm_loadrs_epi8() {
        let data = _mm_set_epi8(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = unsafe { _mm_loadrs_epi8((&raw const data).cast()) };
        assert_eq_m128i(r, data);
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
    fn test_mm_loadrs_epi16() {
        let data = _mm_set_epi16(1, 2, 3, 4, 5, 6, 7, 8);
        let r = unsafe { _mm_loadrs_epi16((&raw const data).cast()) };
        assert_eq_m128i(r, data);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm256_loadrs_epi16() {
        let data = _mm256_set_epi16(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = unsafe { _mm256_loadrs_epi16((&raw const data).cast()) };
        assert_eq_m256i(r, data);
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
    fn test_mm_loadrs_epi32() {
        let data = _mm_set_epi32(1, 2, 3, 4);
        let r = unsafe { _mm_loadrs_epi32((&raw const data).cast()) };
        assert_eq_m128i(r, data);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm256_loadrs_epi32() {
        let data = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
        let r = unsafe { _mm256_loadrs_epi32((&raw const data).cast()) };
        assert_eq_m256i(r, data);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm512_loadrs_epi32() {
        let data = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r = unsafe { _mm512_loadrs_epi32((&raw const data).cast()) };
        assert_eq_m512i(r, data);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm_loadrs_epi64() {
        let data = _mm_set_epi64x(1, 2);
        let r = unsafe { _mm_loadrs_epi64((&raw const data).cast()) };
        assert_eq_m128i(r, data);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm256_loadrs_epi64() {
        let data = _mm256_set_epi64x(1, 2, 3, 4);
        let r = unsafe { _mm256_loadrs_epi64((&raw const data).cast()) };
        assert_eq_m256i(r, data);
    }

    #[simd_test(enable = "movrs,avx10.2")]
    fn test_mm512_loadrs_epi64() {
        let data = _mm512_set_epi64(1, 2, 3, 4, 5, 6, 7, 8);
        let r = unsafe { _mm512_loadrs_epi64((&raw const data).cast()) };
        assert_eq_m512i(r, data);
    }
}
