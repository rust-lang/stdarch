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
}
