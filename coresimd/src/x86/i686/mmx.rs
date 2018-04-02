//! `i586` MMX instruction set.
//!
//! The intrinsics here roughly correspond to those in the `mmintrin.h` C
//! header.
//!
//! The reference is [Intel 64 and IA-32 Architectures Software Developer's
//! Manual Volume 2: Instruction Set Reference, A-Z][intel64_ref].
//!
//! [intel64_ref]: http://www.intel.de/content/dam/www/public/us/en/documents/manuals/64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf

use v64::*;
use core::mem;

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Constructs a 64-bit integer vector initialized to zero.
#[inline]
#[target_feature(enable = "mmx")]
// FIXME: this produces a movl instead of xorps on x86
// FIXME: this produces a xor intrinsic instead of xorps on x86_64
#[cfg_attr(all(test, target_arch = "x86_64"), assert_instr(xor))]
pub unsafe fn _mm_setzero_si64() -> __m64 {
    mem::transmute(0_i64)
}

/// Add packed 8-bit integers in `a` and `b`.
#[inline]
#[target_feature(enable = "mmx")]
#[cfg_attr(test, assert_instr(paddb))]
pub unsafe fn _mm_add_pi8(a: __m64, b: __m64) -> __m64 {
    paddb(a, b)
}

/// Add packed 16-bit integers in `a` and `b`.
#[inline]
#[target_feature(enable = "mmx")]
#[cfg_attr(test, assert_instr(paddw))]
pub unsafe fn _mm_add_pi16(a: __m64, b: __m64) -> __m64 {
    paddw(a, b)
}

/// Add packed 32-bit integers in `a` and `b`.
#[inline]
#[target_feature(enable = "mmx")]
#[cfg_attr(test, assert_instr(paddd))]
pub unsafe fn _mm_add_pi32(a: __m64, b: __m64) -> __m64 {
    paddd(a, b)
}

/// Add packed 8-bit integers in `a` and `b` using saturation.
#[inline]
#[target_feature(enable = "mmx")]
#[cfg_attr(test, assert_instr(paddsb))]
pub unsafe fn _mm_adds_pi8(a: __m64, b: __m64) -> __m64 {
    paddsb(a, b)
}

/// Add packed 16-bit integers in `a` and `b` using saturation.
#[inline]
#[target_feature(enable = "mmx")]
#[cfg_attr(test, assert_instr(paddsw))]
pub unsafe fn _mm_adds_pi16(a: __m64, b: __m64) -> __m64 {
    paddsw(a, b)
}

/// Add packed unsigned 8-bit integers in `a` and `b` using saturation.
#[inline]
#[target_feature(enable = "mmx")]
#[cfg_attr(test, assert_instr(paddusb))]
pub unsafe fn _mm_adds_pu8(a: __m64, b: __m64) -> __m64 {
    paddusb(a, b)
}

/// Add packed unsigned 16-bit integers in `a` and `b` using saturation.
#[inline]
#[target_feature(enable = "mmx")]
#[cfg_attr(test, assert_instr(paddusw))]
pub unsafe fn _mm_adds_pu16(a: __m64, b: __m64) -> __m64 {
    paddusw(a, b)
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.mmx.padd.b"]
    fn paddb(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.padd.w"]
    fn paddw(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.padd.d"]
    fn paddd(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.padds.b"]
    fn paddsb(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.padds.w"]
    fn paddsw(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.paddus.b"]
    fn paddusb(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.paddus.w"]
    fn paddusw(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.packsswb"]
    fn packsswb(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.packssdw"]
    fn packssdw(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.pcmpgt.b"]
    fn pcmpgtb(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.pcmpgt.w"]
    fn pcmpgtw(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.pcmpgt.d"]
    fn pcmpgtd(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.punpckhwd"]
    fn punpckhwd(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.punpcklwd"]
    fn punpcklwd(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.punpckhbw"]
    fn punpckhbw(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.punpcklbw"]
    fn punpcklbw(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.punpckhdq"]
    fn punpckhdq(a: __m64, b: __m64) -> __m64;
    #[link_name = "llvm.x86.mmx.punpckldq"]
    fn punpckldq(a: __m64, b: __m64) -> __m64;
}

#[cfg(test)]
mod tests {
    use v64::{__m64, i16x4, i32x2, i8x8, u16x4, u8x8};
    use x86::i686::mmx;
    use stdsimd_test::simd_test;

    #[simd_test = "mmx"]
    unsafe fn _mm_setzero_si64() {
        let r: __m64 = ::std::mem::transmute(0_i64);
        assert_eq!(r, mmx::_mm_setzero_si64());
    }

    #[simd_test = "mmx"]
    unsafe fn _mm_add_pi8() {
        let a = i8x8::new(-1, -1, 1, 1, -1, 0, 1, 0);
        let b = i8x8::new(-127, 101, 99, 126, 0, -1, 0, 1);
        let r = i8x8::from(mmx::_mm_add_pi8(a.into(), b.into()));
        let e = i8x8::new(-128, 100, 100, 127, -1, -1, 1, 1);
        assert_eq!(r, e);
    }

    #[simd_test = "mmx"]
    unsafe fn _mm_add_pi16() {
        let a = i16x4::new(-1, -1, 1, 1);
        let b = i16x4::new(
            i16::min_value() + 1,
            30001,
            -30001,
            i16::max_value() - 1,
        );
        let r = i16x4::from(mmx::_mm_add_pi16(a.into(), b.into()));
        let e = i16x4::new(i16::min_value(), 30000, -30000, i16::max_value());
        assert_eq!(r, e);
    }

    #[simd_test = "mmx"]
    unsafe fn _mm_add_pi32() {
        let a = i32x2::new(1, -1);
        let b = i32x2::new(i32::max_value() - 1, i32::min_value() + 1);
        let r = i32x2::from(mmx::_mm_add_pi32(a.into(), b.into()));
        let e = i32x2::new(i32::max_value(), i32::min_value());
        assert_eq!(r, e);
    }

    #[simd_test = "mmx"]
    unsafe fn _mm_adds_pi8() {
        let a = i8x8::new(-100, -1, 1, 100, -1, 0, 1, 0);
        let b = i8x8::new(-100, 1, -1, 100, 0, -1, 0, 1);
        let r = i8x8::from(mmx::_mm_adds_pi8(a.into(), b.into()));
        let e =
            i8x8::new(i8::min_value(), 0, 0, i8::max_value(), -1, -1, 1, 1);
        assert_eq!(r, e);
    }

    #[simd_test = "mmx"]
    unsafe fn _mm_adds_pi16() {
        let a = i16x4::new(-32000, 32000, 4, 0);
        let b = i16x4::new(-32000, 32000, -5, 1);
        let r = i16x4::from(mmx::_mm_adds_pi16(a.into(), b.into()));
        let e = i16x4::new(i16::min_value(), i16::max_value(), -1, 1);
        assert_eq!(r, e);
    }

    #[simd_test = "mmx"]
    unsafe fn _mm_adds_pu8() {
        let a = u8x8::new(0, 1, 2, 3, 4, 5, 6, 200);
        let b = u8x8::new(0, 10, 20, 30, 40, 50, 60, 200);
        let r = u8x8::from(mmx::_mm_adds_pu8(a.into(), b.into()));
        let e = u8x8::new(0, 11, 22, 33, 44, 55, 66, u8::max_value());
        assert_eq!(r, e);
    }

    #[simd_test = "mmx"]
    unsafe fn _mm_adds_pu16() {
        let a = u16x4::new(0, 1, 2, 60000);
        let b = u16x4::new(0, 10, 20, 60000);
        let r = u16x4::from(mmx::_mm_adds_pu16(a.into(), b.into()));
        let e = u16x4::new(0, 11, 22, u16::max_value());
        assert_eq!(r, e);
    }
}
