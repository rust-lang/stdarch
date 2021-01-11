use crate::{
    core_arch::{simd::*, /*simd_llvm::*,*/ x86::*},
    mem::transmute,
};

#[cfg(test)]
use stdarch_test::assert_instr;

/// Compute intersection of packed 32-bit integer vectors a and b, and store indication of match in the corresponding bit of two mask registers specified by k1 and k2. A match in corresponding elements of a and b is indicated by a set bit in the corresponding bit of the mask registers.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_2intersect_epi32&expand=2)
#[inline]
#[target_feature(enable = "avx512vp2intersect,avx512f")]
#[cfg_attr(test, assert_instr(vp2intersectd))]
pub unsafe fn _mm512_2intersect_epi32(a: __m512i, b: __m512i, k1: *mut u16, k2: *mut u16) {
    transmute(vp2intersectd(a.as_i32x16(), b.as_i32x16(), k1, k2))
}

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.vp2intersect.d.512"]
    fn vp2intersectd(a: i32x16, b: i32x16, k1: *mut u16, k2: *mut u16);
}

#[cfg(test)]
mod tests {

    use crate::core_arch::x86::*;
    use stdarch_test::simd_test;

    #[simd_test(enable = "avx512vp2intersect,avx512f")]
    unsafe fn test_mm512_2intersect_epi32() {
        let a = _mm512_set1_epi32(1);
        let b = _mm512_set1_epi32(1);
        let mut r1: u16 = 0;
        let mut r2: u16 = 0;
        _mm512_2intersect_epi32(a, b, &mut r1 as *mut _ as *mut u16, &mut r2 as *mut _ as *mut u16);
        //assert_eq!(r1, 0b11111111_11111111);
        //assert_eq!(r2, 0b11111111_11111111);
        assert_eq!(r1, 0);
        assert_eq!(r2, 0);
    }
}
