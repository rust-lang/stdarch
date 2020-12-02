use crate::{
    core_arch::{simd::*, simd_llvm::*, x86::*},
    mem::transmute,
};

#[cfg(test)]
use stdarch_test::assert_instr;

/// Broadcast the low 16-bits from input mask k to all 32-bit elements of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_broadcastmw_epi32&expand=553)
#[inline]
#[target_feature(enable = "avx512cd")]
#[cfg_attr(test, assert_instr(vpbroadcast))] // should be vpbroadcastmw2d
pub unsafe fn _mm512_broadcastmw_epi32(k: __mmask16) -> __m512i {
    _mm512_set1_epi32(k as i32)
}

/// Broadcast the low 8-bits from input mask k to all 64-bit elements of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_broadcastmb_epi64&expand=550)
#[inline]
#[target_feature(enable = "avx512cd")]
#[cfg_attr(test, assert_instr(vpbroadcast))] // should be vpbroadcastmb2q
pub unsafe fn _mm512_broadcastmb_epi64(k: __mmask8) -> __m512i {
    _mm512_set1_epi64(k as i64)
}

/// Test each 32-bit element of a for equality with all other elements in a closer to the least significant bit. Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_conflict_epi32&expand=1248)
#[inline]
#[target_feature(enable = "avx512cd")]
#[cfg_attr(test, assert_instr(vpconflictd))]
pub unsafe fn _mm512_conflict_epi32(a: __m512i) -> __m512i {
    transmute(vpconflictd(a.as_i32x16()))
}

/// Test each 32-bit element of a for equality with all other elements in a closer to the least significant bit using writemask k (elements are copied from src when the corresponding mask bit is not set). Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_conflict_epi32&expand=1249)
#[inline]
#[target_feature(enable = "avx512cd")]
#[cfg_attr(test, assert_instr(vpconflictd))]
pub unsafe fn _mm512_mask_conflict_epi32(src: __m512i, k: __mmask16, a: __m512i) -> __m512i {
    let conflict = _mm512_conflict_epi32(a).as_i32x16();
    transmute(simd_select_bitmask(k, conflict, src.as_i32x16()))
}

/// Test each 32-bit element of a for equality with all other elements in a closer to the least significant bit using zeromask k (elements are zeroed out when the corresponding mask bit is not set). Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_conflict_epi32&expand=1250)
#[inline]
#[target_feature(enable = "avx512cd")]
#[cfg_attr(test, assert_instr(vpconflictd))]
pub unsafe fn _mm512_maskz_conflict_epi32(k: __mmask16, a: __m512i) -> __m512i {
    let conflict = _mm512_conflict_epi32(a).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, conflict, zero))
}

/// Test each 64-bit element of a for equality with all other elements in a closer to the least significant bit. Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_conflict_epi64&expand=1257)
#[inline]
#[target_feature(enable = "avx512cd")]
#[cfg_attr(test, assert_instr(vpconflictq))]
pub unsafe fn _mm512_conflict_epi64(a: __m512i) -> __m512i {
    transmute(vpconflictq(a.as_i64x8()))
}

/// Test each 64-bit element of a for equality with all other elements in a closer to the least significant bit using writemask k (elements are copied from src when the corresponding mask bit is not set). Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_conflict_epi64&expand=1258)
#[inline]
#[target_feature(enable = "avx512cd")]
#[cfg_attr(test, assert_instr(vpconflictq))]
pub unsafe fn _mm512_mask_conflict_epi64(src: __m512i, k: __mmask8, a: __m512i) -> __m512i {
    let conflict = _mm512_conflict_epi64(a).as_i64x8();
    transmute(simd_select_bitmask(k, conflict, src.as_i64x8()))
}

/// Test each 64-bit element of a for equality with all other elements in a closer to the least significant bit using zeromask k (elements are zeroed out when the corresponding mask bit is not set). Each element's comparison forms a zero extended bit vector in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_conflict_epi64&expand=1259)
#[inline]
#[target_feature(enable = "avx512cd")]
#[cfg_attr(test, assert_instr(vpconflictq))]
pub unsafe fn _mm512_maskz_conflict_epi64(k: __mmask8, a: __m512i) -> __m512i {
    let conflict = _mm512_conflict_epi64(a).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, conflict, zero))
}
/*
/// Counts the number of leading zero bits in each packed 32-bit integer in a, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_lzcnt_epi32&expand=3491)
#[inline]
#[target_feature(enable = "avx512cd")]
#[cfg_attr(test, assert_instr(vplzcntd))]
pub unsafe fn _mm512_lzcnt_epi32(a: __m512i) -> __m512i {
}
*/
#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.conflict.d.512"]
    fn vpconflictd(a: i32x16) -> i32x16;
    #[link_name = "llvm.x86.avx512.conflict.q.512"]
    fn vpconflictq(a: i64x8) -> i64x8;
}

#[cfg(test)]
mod tests {

    use crate::core_arch::x86::*;
    use stdarch_test::simd_test;

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_broadcastmw_epi32() {
        let a: __mmask16 = 2;
        let r = _mm512_broadcastmw_epi32(a);
        let e = _mm512_set1_epi32(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_broadcastmb_epi64() {
        let a: __mmask8 = 2;
        let r = _mm512_broadcastmb_epi64(a);
        let e = _mm512_set1_epi64(2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_conflict_epi32() {
        let a = _mm512_set1_epi32(1);
        let r = _mm512_conflict_epi32(a);
        let e = _mm512_set_epi32(
            1 << 14
                | 1 << 13
                | 1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 13
                | 1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 9 | 1 << 8 | 1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 8 | 1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 2 | 1 << 1 | 1 << 0,
            1 << 1 | 1 << 0,
            1 << 0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_mask_conflict_epi32() {
        let a = _mm512_set1_epi32(1);
        let r = _mm512_mask_conflict_epi32(a, 0, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_conflict_epi32(a, 0b11111111_11111111, a);
        let e = _mm512_set_epi32(
            1 << 14
                | 1 << 13
                | 1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 13
                | 1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 9 | 1 << 8 | 1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 8 | 1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 2 | 1 << 1 | 1 << 0,
            1 << 1 | 1 << 0,
            1 << 0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_maskz_conflict_epi32() {
        let a = _mm512_set1_epi32(1);
        let r = _mm512_maskz_conflict_epi32(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_conflict_epi32(0b11111111_11111111, a);
        let e = _mm512_set_epi32(
            1 << 14
                | 1 << 13
                | 1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 13
                | 1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 12
                | 1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 11
                | 1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 10
                | 1 << 9
                | 1 << 8
                | 1 << 7
                | 1 << 6
                | 1 << 5
                | 1 << 4
                | 1 << 3
                | 1 << 2
                | 1 << 1
                | 1 << 0,
            1 << 9 | 1 << 8 | 1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 8 | 1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 7 | 1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 2 | 1 << 1 | 1 << 0,
            1 << 1 | 1 << 0,
            1 << 0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_conflict_epi64() {
        let a = _mm512_set1_epi64(1);
        let r = _mm512_conflict_epi64(a);
        let e = _mm512_set_epi64(
            1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 2 | 1 << 1 | 1 << 0,
            1 << 1 | 1 << 0,
            1 << 0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_mask_conflict_epi64() {
        let a = _mm512_set1_epi64(1);
        let r = _mm512_mask_conflict_epi64(a, 0, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_conflict_epi64(a, 0b11111111, a);
        let e = _mm512_set_epi64(
            1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 2 | 1 << 1 | 1 << 0,
            1 << 1 | 1 << 0,
            1 << 0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512cd")]
    unsafe fn test_mm512_maskz_conflict_epi64() {
        let a = _mm512_set1_epi64(1);
        let r = _mm512_maskz_conflict_epi64(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_conflict_epi64(0b11111111, a);
        let e = _mm512_set_epi64(
            1 << 6 | 1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 5 | 1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 4 | 1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 3 | 1 << 2 | 1 << 1 | 1 << 0,
            1 << 2 | 1 << 1 | 1 << 0,
            1 << 1 | 1 << 0,
            1 << 0,
            0,
        );
        assert_eq_m512i(r, e);
    }
}
