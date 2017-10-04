#[cfg(test)]
use stdsimd_test::assert_instr;

use v128::*;
use x86::__m128i;

#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(pblendvb))]
pub unsafe fn _mm_blendv_epi8(
    a: __m128i,
    b: __m128i,
    mask: __m128i,
) -> __m128i {
    pblendvb(a, b, mask)
}

#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(pblendw, imm8=0xF0))]
pub unsafe fn _mm_blend_epi16(a: i16x8, b: i16x8, imm8: u8) -> i16x8 {
    macro_rules! call {
        ($imm8:expr) => { pblendw(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Blend packed double-precision (64-bit) floating-point elements from `a` and `b` using `mask`
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(blendvpd))]
pub unsafe fn _mm_blendv_pd(a: f64x2, b: f64x2, mask: f64x2) -> f64x2 {
    blendvpd(a, b, mask)
}

/// Blend packed single-precision (32-bit) floating-point elements from `a` and `b` using `mask`
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(blendvps))]
pub unsafe fn _mm_blendv_ps(a: f32x4, b: f32x4, mask: f32x4) -> f32x4 {
    blendvps(a, b, mask)
}

/// Blend packed double-precision (64-bit) floating-point elements from `a` and `b` using control mask `imm2`
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(blendpd, imm2=0b10))]
pub unsafe fn _mm_blend_pd(a: f64x2, b: f64x2, imm2: u8) -> f64x2 {
    macro_rules! call {
        ($imm2:expr) => { blendpd(a, b, $imm2) }
    }
    constify_imm2!(imm2, call)
}

/// Blend packed single-precision (32-bit) floating-point elements from `a` and `b` using mask `imm4`
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(blendps, imm4=0b0101))]
pub unsafe fn _mm_blend_ps(a: f32x4, b: f32x4, imm4: u8) -> f32x4 {
    macro_rules! call {
        ($imm4:expr) => { blendps(a, b, $imm4) }
    }
    constify_imm4!(imm4, call)
}

/// Returns the dot product of two f64x2 vectors.
///
/// `imm8[1:0]` is the broadcast mask, and `imm8[5:4]` is the condition mask.
/// If a condition mask bit is zero, the corresponding multiplication is
/// replaced by a value of `0.0`. If a broadcast mask bit is one, the result of
/// the dot product will be stored in the return value component. Otherwise if
/// the broadcast mask bit is zero then the return component will be zero.
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(dppd, imm8 = 0))]
pub unsafe fn _mm_dp_pd(a: f64x2, b: f64x2, imm8: u8) -> f64x2 {
    macro_rules! call {
        ($imm8:expr) => { dppd(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

/// Returns the dot product of two f32x4 vectors.
///
/// `imm8[3:0]` is the broadcast mask, and `imm8[7:4]` is the condition mask.
/// If a condition mask bit is zero, the corresponding multiplication is
/// replaced by a value of `0.0`. If a broadcast mask bit is one, the result of
/// the dot product will be stored in the return value component. Otherwise if
/// the broadcast mask bit is zero then the return component will be zero.
#[inline(always)]
#[target_feature = "+sse4.1"]
#[cfg_attr(test, assert_instr(dpps, imm8 = 0))]
pub unsafe fn _mm_dp_ps(a: f32x4, b: f32x4, imm8: u8) -> f32x4 {
    macro_rules! call {
        ($imm8:expr) => { dpps(a, b, $imm8) }
    }
    constify_imm8!(imm8, call)
}

#[allow(improper_ctypes)]
extern {
    #[link_name = "llvm.x86.sse41.pblendvb"]
    fn pblendvb(a: __m128i, b: __m128i, mask: __m128i) -> __m128i;
    #[link_name = "llvm.x86.sse41.blendvpd"]
    fn blendvpd(a: f64x2, b: f64x2, mask: f64x2) -> f64x2;
    #[link_name = "llvm.x86.sse41.blendvps"]
    fn blendvps(a: f32x4, b: f32x4, mask: f32x4) -> f32x4;
    #[link_name = "llvm.x86.sse41.blendpd"]
    fn blendpd(a: f64x2, b: f64x2, imm2: u8) -> f64x2;
    #[link_name = "llvm.x86.sse41.blendps"]
    fn blendps(a: f32x4, b: f32x4, imm4: u8) -> f32x4;
    #[link_name = "llvm.x86.sse41.pblendw"]
    fn pblendw(a: i16x8, b: i16x8, imm8: u8) -> i16x8;
    #[link_name = "llvm.x86.sse41.dppd"]
    fn dppd(a: f64x2, b: f64x2, imm8: u8) -> f64x2;
    #[link_name = "llvm.x86.sse41.dpps"]
    fn dpps(a: f32x4, b: f32x4, imm8: u8) -> f32x4;
}

#[cfg(test)]
mod tests {
    use stdsimd_test::simd_test;

    use v128::*;
    use x86::sse41;

    #[simd_test = "sse4.1"]
    unsafe fn _mm_blendv_epi8() {
        let a = i8x16::new(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = i8x16::new(
            16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
        let mask = i8x16::new(
            0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1);
        let e = i8x16::new(
            0, 17, 2, 19, 4, 21, 6, 23, 8, 25, 10, 27, 12, 29, 14, 31);
        assert_eq!(sse41::_mm_blendv_epi8(a, b, mask), e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_blendv_pd() {
        let a = f64x2::splat(0.0);
        let b = f64x2::splat(1.0);
        let mask = ::std::mem::transmute(i64x2::new(0, -1));
        let r = sse41::_mm_blendv_pd(a, b, mask);
        let e = f64x2::new(0.0, 1.0);
        assert_eq!(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_blendv_ps() {
        let a = f32x4::splat(0.0);
        let b = f32x4::splat(1.0);
        let mask = ::std::mem::transmute(i32x4::new(0,-1, 0, -1));
        let r = sse41::_mm_blendv_ps(a, b, mask);
        let e = f32x4::new(0.0, 1.0, 0.0, 1.0);
        assert_eq!(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_blend_pd() {
        let a = f64x2::splat(0.0);
        let b = f64x2::splat(1.0);
        let r = sse41::_mm_blend_pd(a, b, 0b10);
        let e = f64x2::new(0.0, 1.0);
        assert_eq!(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_blend_ps() {
        let a = f32x4::splat(0.0);
        let b = f32x4::splat(1.0);
        let r = sse41::_mm_blend_ps(a, b, 0b1010);
        let e = f32x4::new(0.0, 1.0, 0.0, 1.0);
        assert_eq!(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_blend_epi16() {
        let a = i16x8::splat(0);
        let b = i16x8::splat(1);
        let r = sse41::_mm_blend_epi16(a, b, 0b1010_1100);
        let e = i16x8::new(0, 0, 1, 1, 0, 1, 0, 1);
        assert_eq!(r, e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_dp_pd() {
        let a = f64x2::new(2.0, 3.0);
        let b = f64x2::new(1.0, 4.0);
        let e = f64x2::new(14.0, 0.0);
        assert_eq!(sse41::_mm_dp_pd(a, b, 0b00110001), e);
    }

    #[simd_test = "sse4.1"]
    unsafe fn _mm_dp_ps() {
        let a = f32x4::new(2.0, 3.0, 1.0, 10.0);
        let b = f32x4::new(1.0, 4.0, 0.5, 10.0);
        let e = f32x4::new(14.5, 0.0, 14.5, 0.0);
        assert_eq!(sse41::_mm_dp_ps(a, b, 0b01110101), e);
    }
}
