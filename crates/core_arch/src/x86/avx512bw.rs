use crate::{
    core_arch::{simd::*, simd_llvm::*, x86::*},
    //mem::{self, transmute},
    //ptr,
};

#[cfg(test)]
use stdarch_test::assert_instr;
/*
/// Computes the absolute values of packed 32-bit integers in `a`.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990,33&text=_mm512_abs_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpabsd))]
pub unsafe fn _mm512_abs_epi32(a: __m512i) -> __m512i {
    let a = a.as_i32x16();
    // all-0 is a properly initialized i32x16
    let zero: i32x16 = mem::zeroed();
    let sub = simd_sub(zero, a);
    let cmp: i32x16 = simd_gt(a, zero);
    transmute(simd_select(cmp, a, sub))
}

/// Computes the absolute value of packed 32-bit integers in `a`, and store the
/// unsigned results in `dst` using writemask `k` (elements are copied from
/// `src` when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990,33&text=_mm512_abs_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpabsd))]
pub unsafe fn _mm512_mask_abs_epi32(src: __m512i, k: __mmask16, a: __m512i) -> __m512i {
    let abs = _mm512_abs_epi32(a).as_i32x16();
    transmute(simd_select_bitmask(k, abs, src.as_i32x16()))
}

/// Computes the absolute value of packed 32-bit integers in `a`, and store the
/// unsigned results in `dst` using zeromask `k` (elements are zeroed out when
/// the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990,33,34,35,35&text=_mm512_maskz_abs_epi32)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpabsd))]
pub unsafe fn _mm512_maskz_abs_epi32(k: __mmask16, a: __m512i) -> __m512i {
    let abs = _mm512_abs_epi32(a).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, abs, zero))
}

/// Compute the absolute value of packed signed 64-bit integers in a, and store the unsigned results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_abs_epi64&expand=48)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpabsq))]
pub unsafe fn _mm512_abs_epi64(a: __m512i) -> __m512i {
    let a = a.as_i64x8();
    // all-0 is a properly initialized i64x8
    let zero: i64x8 = mem::zeroed();
    let sub = simd_sub(zero, a);
    let cmp: i64x8 = simd_gt(a, zero);
    transmute(simd_select(cmp, a, sub))
}

/// Compute the absolute value of packed signed 64-bit integers in a, and store the unsigned results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_abs_epi64&expand=49)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpabsq))]
pub unsafe fn _mm512_mask_abs_epi64(src: __m512i, k: __mmask8, a: __m512i) -> __m512i {
    let abs = _mm512_abs_epi64(a).as_i64x8();
    transmute(simd_select_bitmask(k, abs, src.as_i64x8()))
}

/// Compute the absolute value of packed signed 64-bit integers in a, and store the unsigned results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_abs_epi64&expand=50)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpabsq))]
pub unsafe fn _mm512_maskz_abs_epi64(k: __mmask8, a: __m512i) -> __m512i {
    let abs = _mm512_abs_epi64(a).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, abs, zero))
}

/// Move packed 32-bit integers from a to dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_mov_epi32&expand=3801)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmovdqa32))]
pub unsafe fn _mm512_mask_mov_epi32(src: __m512i, k: __mmask16, a: __m512i) -> __m512i {
    let mov = a.as_i32x16();
    transmute(simd_select_bitmask(k, mov, src.as_i32x16()))
}

/// Move packed 32-bit integers from a into dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_mov_epi32&expand=3802)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmovdqa32))]
pub unsafe fn _mm512_maskz_mov_epi32(k: __mmask16, a: __m512i) -> __m512i {
    let mov = a.as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, mov, zero))
}

/// Move packed 64-bit integers from a to dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_mov_epi64&expand=3807)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmovdqa64))]
pub unsafe fn _mm512_mask_mov_epi64(src: __m512i, k: __mmask8, a: __m512i) -> __m512i {
    let mov = a.as_i64x8();
    transmute(simd_select_bitmask(k, mov, src.as_i64x8()))
}

/// Move packed 64-bit integers from a into dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_mov_epi64&expand=3808)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmovdqa64))]
pub unsafe fn _mm512_maskz_mov_epi64(k: __mmask8, a: __m512i) -> __m512i {
    let mov = a.as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, mov, zero))
}

/// Add packed 32-bit integers in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_add_epi32&expand=100)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm512_add_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_add(a.as_i32x16(), b.as_i32x16()))
}

/// Add packed 32-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_add_epi32&expand=101)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm512_mask_add_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let add = _mm512_add_epi32(a, b).as_i32x16();
    transmute(simd_select_bitmask(k, add, src.as_i32x16()))
}

/// Add packed 32-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_add_epi32&expand=102)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddd))]
pub unsafe fn _mm512_maskz_add_epi32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let add = _mm512_add_epi32(a, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, add, zero))
}

/// Add packed 64-bit integers in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_add_epi64&expand=109)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddq))]
pub unsafe fn _mm512_add_epi64(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_add(a.as_i64x8(), b.as_i64x8()))
}

/// Add packed 64-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_add_epi64&expand=110)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddq))]
pub unsafe fn _mm512_mask_add_epi64(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let add = _mm512_add_epi64(a, b).as_i64x8();
    transmute(simd_select_bitmask(k, add, src.as_i64x8()))
}

/// Add packed 64-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_add_epi64&expand=111)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpaddq))]
pub unsafe fn _mm512_maskz_add_epi64(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let add = _mm512_add_epi64(a, b).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, add, zero))
}

/// Subtract packed 32-bit integers in b from packed 32-bit integers in a, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sub_epi32&expand=5694)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsubd))]
pub unsafe fn _mm512_sub_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_sub(a.as_i32x16(), b.as_i32x16()))
}

/// Subtract packed 32-bit integers in b from packed 32-bit integers in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sub_epi32&expand=5692)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsubd))]
pub unsafe fn _mm512_mask_sub_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let sub = _mm512_sub_epi32(a, b).as_i32x16();
    transmute(simd_select_bitmask(k, sub, src.as_i32x16()))
}

/// Subtract packed 32-bit integers in b from packed 32-bit integers in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sub_epi32&expand=5693)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsubd))]
pub unsafe fn _mm512_maskz_sub_epi32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let sub = _mm512_sub_epi32(a, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, sub, zero))
}

/// Subtract packed 64-bit integers in b from packed 64-bit integers in a, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sub_epi64&expand=5703)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsubq))]
pub unsafe fn _mm512_sub_epi64(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_sub(a.as_i64x8(), b.as_i64x8()))
}

/// Subtract packed 64-bit integers in b from packed 64-bit integers in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sub_epi64&expand=5701)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsubq))]
pub unsafe fn _mm512_mask_sub_epi64(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let sub = _mm512_sub_epi64(a, b).as_i64x8();
    transmute(simd_select_bitmask(k, sub, src.as_i64x8()))
}

/// Subtract packed 64-bit integers in b from packed 64-bit integers in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sub_epi64&expand=5702)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsubq))]
pub unsafe fn _mm512_maskz_sub_epi64(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let add = _mm512_sub_epi64(a, b).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, add, zero))
}

/// Multiply the low signed 32-bit integers from each packed 64-bit element in a and b, and store the signed 64-bit results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mul_epi32&expand=3907)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmuldq))]
pub unsafe fn _mm512_mul_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmuldq(a.as_i32x16(), b.as_i32x16()))
}

/// Multiply the low signed 32-bit integers from each packed 64-bit element in a and b, and store the signed 64-bit results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_mul_epi32&expand=3905)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmuldq))]
pub unsafe fn _mm512_mask_mul_epi32(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mul_epi32(a, b).as_i64x8();
    transmute(simd_select_bitmask(k, mul, src.as_i64x8()))
}

/// Multiply the low signed 32-bit integers from each packed 64-bit element in a and b, and store the signed 64-bit results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_mul_epi32&expand=3906)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmuldq))]
pub unsafe fn _mm512_maskz_mul_epi32(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mul_epi32(a, b).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiply the packed 32-bit integers in a and b, producing intermediate 64-bit integers, and store the low 32 bits of the intermediate integers in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mullo_epi&expand=4005)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmulld))]
pub unsafe fn _mm512_mullo_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_mul(a.as_i32x16(), b.as_i32x16()))
}

/// Multiply the packed 32-bit integers in a and b, producing intermediate 64-bit integers, and store the low 32 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_mullo_epi32&expand=4003)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmulld))]
pub unsafe fn _mm512_mask_mullo_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let mul = _mm512_mullo_epi32(a, b).as_i32x16();
    transmute(simd_select_bitmask(k, mul, src.as_i32x16()))
}

/// Multiply the packed 32-bit integers in a and b, producing intermediate 64-bit integers, and store the low 32 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_mullo_epi32&expand=4004)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmulld))]
pub unsafe fn _mm512_maskz_mullo_epi32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mullo_epi32(a, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiplies elements in packed 64-bit integer vectors a and b together, storing the lower 64 bits of the result in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mullox_epi64&expand=4017)
///
/// This intrinsic generates a sequence of instructions, which may perform worse than a native instruction. Consider the performance impact of this intrinsic.
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_mullox_epi64(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_mul(a.as_i64x8(), b.as_i64x8()))
}

/// Multiplies elements in packed 64-bit integer vectors a and b together, storing the lower 64 bits of the result in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_mullox&expand=4016)
///
/// This intrinsic generates a sequence of instructions, which may perform worse than a native instruction. Consider the performance impact of this intrinsic.
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_mask_mullox_epi64(
    src: __m512i,
    k: __mmask8,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let mul = _mm512_mullox_epi64(a, b).as_i64x8();
    transmute(simd_select_bitmask(k, mul, src.as_i64x8()))
}

/// Compare packed signed 32-bit integers in a and b, and store packed maximum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_max_epi32&expand=3582)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmaxsd))]
pub unsafe fn _mm512_max_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmaxsd(a.as_i32x16(), b.as_i32x16()))
}

/// Compare packed signed 32-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_max_epi32&expand=3580)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmaxsd))]
pub unsafe fn _mm512_mask_max_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epi32(a, b).as_i32x16();
    transmute(simd_select_bitmask(k, max, src.as_i32x16()))
}

/// Compare packed signed 32-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_max_epi32&expand=3581)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmaxsd))]
pub unsafe fn _mm512_maskz_max_epi32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epi32(a, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed signed 64-bit integers in a and b, and store packed maximum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_max_epi64&expand=3591)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmaxsq))]
pub unsafe fn _mm512_max_epi64(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmaxsq(a.as_i64x8(), b.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_max_epi64&expand=3589)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmaxsq))]
pub unsafe fn _mm512_mask_max_epi64(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epi64(a, b).as_i64x8();
    transmute(simd_select_bitmask(k, max, src.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_max_epi64&expand=3590)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmaxsq))]
pub unsafe fn _mm512_maskz_max_epi64(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epi64(a, b).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed signed 32-bit integers in a and b, and store packed minimum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_min_epi32&expand=3696)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpminsd))]
pub unsafe fn _mm512_min_epi32(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpminsd(a.as_i32x16(), b.as_i32x16()))
}

/// Compare packed signed 32-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_min_epi32&expand=3694)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpminsd))]
pub unsafe fn _mm512_mask_min_epi32(src: __m512i, k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_min_epi32(a, b).as_i32x16();
    transmute(simd_select_bitmask(k, max, src.as_i32x16()))
}

/// Compare packed signed 32-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_min_epi32&expand=3695)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpminsd))]
pub unsafe fn _mm512_maskz_min_epi32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_min_epi32(a, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed signed 64-bit integers in a and b, and store packed minimum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_min_epi64&expand=3705)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpminsq))]
pub unsafe fn _mm512_min_epi64(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpminsq(a.as_i64x8(), b.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b, and store packed minimum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_min_epi64&expand=3703)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpminsq))]
pub unsafe fn _mm512_mask_min_epi64(src: __m512i, k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_min_epi64(a, b).as_i64x8();
    transmute(simd_select_bitmask(k, max, src.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b, and store packed minimum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_min_epi64&expand=3704)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpminsq))]
pub unsafe fn _mm512_maskz_min_epi64(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_min_epi64(a, b).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Sign extend packed 8-bit integers in a to packed 32-bit integers, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtepi8_epi32&expand=1535)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsxbd))]
pub unsafe fn _mm512_cvtepi8_epi32(a: __m128i) -> __m512i {
    let a = a.as_i8x16();
    transmute::<i32x16, _>(simd_cast(a))
}

/// Sign extend packed 8-bit integers in a to packed 32-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtepi8_epi32&expand=1536)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsxbd))]
pub unsafe fn _mm512_mask_cvtepi8_epi32(src: __m512i, k: __mmask16, a: __m128i) -> __m512i {
    let convert = _mm512_cvtepi8_epi32(a).as_i32x16();
    transmute(simd_select_bitmask(k, convert, src.as_i32x16()))
}

/// Sign extend packed 8-bit integers in a to packed 32-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtepi8_epi32&expand=1537)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsxbd))]
pub unsafe fn _mm512_maskz_cvtepi8_epi32(k: __mmask16, a: __m128i) -> __m512i {
    let convert = _mm512_cvtepi8_epi32(a).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, convert, zero))
}

/// Sign extend packed 8-bit integers in the low 8 bytes of a to packed 64-bit integers, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtepi8_epi64&expand=1544)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsxbq))]
pub unsafe fn _mm512_cvtepi8_epi64(a: __m128i) -> __m512i {
    let a = a.as_i8x16();
    let v64: i8x8 = simd_shuffle8(a, a, [0, 1, 2, 3, 4, 5, 6, 7]);
    transmute::<i64x8, _>(simd_cast(v64))
}

/// Sign extend packed 8-bit integers in the low 8 bytes of a to packed 64-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtepi8_epi64&expand=1545)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsxbq))]
pub unsafe fn _mm512_mask_cvtepi8_epi64(src: __m512i, k: __mmask8, a: __m128i) -> __m512i {
    let convert = _mm512_cvtepi8_epi64(a).as_i64x8();
    transmute(simd_select_bitmask(k, convert, src.as_i64x8()))
}

/// Sign extend packed 8-bit integers in the low 8 bytes of a to packed 64-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtepi8_epi64&expand=1546)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsxbq))]
pub unsafe fn _mm512_maskz_cvtepi8_epi64(k: __mmask8, a: __m128i) -> __m512i {
    let convert = _mm512_cvtepi8_epi64(a).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, convert, zero))
}

/// Sign extend packed 16-bit integers in a to packed 32-bit integers, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtepi16_epi32&expand=1389)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsxwd))]
pub unsafe fn _mm512_cvtepi16_epi32(a: __m256i) -> __m512i {
    let a = a.as_i16x16();
    transmute::<i32x16, _>(simd_cast(a))
}

/// Sign extend packed 16-bit integers in a to packed 32-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtepi16_epi32&expand=1390)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsxwd))]
pub unsafe fn _mm512_mask_cvtepi16_epi32(src: __m512i, k: __mmask16, a: __m256i) -> __m512i {
    let convert = _mm512_cvtepi16_epi32(a).as_i32x16();
    transmute(simd_select_bitmask(k, convert, src.as_i32x16()))
}

/// Sign extend packed 16-bit integers in a to packed 32-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtepi16_epi32&expand=1391)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsxwd))]
pub unsafe fn _mm512_maskz_cvtepi16_epi32(k: __mmask16, a: __m256i) -> __m512i {
    let convert = _mm512_cvtepi16_epi32(a).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, convert, zero))
}

/// Sign extend packed 16-bit integers in a to packed 64-bit integers, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtepi16_epi64&expand=1398)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsxwq))]
pub unsafe fn _mm512_cvtepi16_epi64(a: __m128i) -> __m512i {
    let a = a.as_i16x8();
    transmute::<i64x8, _>(simd_cast(a))
}

/// Sign extend packed 16-bit integers in a to packed 64-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtepi16_epi64&expand=1399)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsxwq))]
pub unsafe fn _mm512_mask_cvtepi16_epi64(src: __m512i, k: __mmask8, a: __m128i) -> __m512i {
    let convert = _mm512_cvtepi16_epi64(a).as_i64x8();
    transmute(simd_select_bitmask(k, convert, src.as_i64x8()))
}

/// Sign extend packed 16-bit integers in a to packed 64-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtepi16_epi64&expand=1400)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsxwq))]
pub unsafe fn _mm512_maskz_cvtepi16_epi64(k: __mmask8, a: __m128i) -> __m512i {
    let convert = _mm512_cvtepi16_epi64(a).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, convert, zero))
}

/// Sign extend packed 32-bit integers in a to packed 64-bit integers, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtepi32_epi64&expand=1428)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsxdq))]
pub unsafe fn _mm512_cvtepi32_epi64(a: __m256i) -> __m512i {
    let a = a.as_i32x8();
    transmute::<i64x8, _>(simd_cast(a))
}

/// Sign extend packed 32-bit integers in a to packed 64-bit integers, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtepi32_epi64&expand=1429)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsxdq))]
pub unsafe fn _mm512_mask_cvtepi32_epi64(src: __m512i, k: __mmask8, a: __m256i) -> __m512i {
    let convert = _mm512_cvtepi32_epi64(a).as_i64x8();
    transmute(simd_select_bitmask(k, convert, src.as_i64x8()))
}

/// Sign extend packed 32-bit integers in a to packed 64-bit integers, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtepi32_epi64&expand=1430)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsxdq))]
pub unsafe fn _mm512_maskz_cvtepi32_epi64(k: __mmask8, a: __m256i) -> __m512i {
    let convert = _mm512_cvtepi32_epi64(a).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, convert, zero))
}

/// Convert packed 32-bit integers in a to packed 16-bit integers with truncation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtepi32_epi16&expand=1419)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovdw))]
pub unsafe fn _mm512_cvtepi32_epi16(a: __m512i) -> __m256i {
    let a = a.as_i32x16();
    transmute::<i16x16, _>(simd_cast(a))
}

/// Convert packed 32-bit integers in a to packed 16-bit integers with truncation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtepi32_epi16&expand=1420)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovdw))]
pub unsafe fn _mm512_mask_cvtepi32_epi16(src: __m256i, k: __mmask16, a: __m512i) -> __m256i {
    let convert = _mm512_cvtepi32_epi16(a).as_i16x16();
    transmute(simd_select_bitmask(k, convert, src.as_i16x16()))
}

/// Convert packed 32-bit integers in a to packed 16-bit integers with truncation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtepi32_epi16&expand=1421)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovdw))]
pub unsafe fn _mm512_maskz_cvtepi32_epi16(k: __mmask16, a: __m512i) -> __m256i {
    let convert = _mm512_cvtepi32_epi16(a).as_i16x16();
    let zero = _mm256_setzero_si256().as_i16x16();
    transmute(simd_select_bitmask(k, convert, zero))
}

/// Convert packed 32-bit integers in a to packed 8-bit integers with truncation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtepi32_epi8&expand=1437)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovdb))]
pub unsafe fn _mm512_cvtepi32_epi8(a: __m512i) -> __m128i {
    let a = a.as_i32x16();
    transmute::<i8x16, _>(simd_cast(a))
}

/// Convert packed 32-bit integers in a to packed 8-bit integers with truncation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtepi32_epi8&expand=1438)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovdb))]
pub unsafe fn _mm512_mask_cvtepi32_epi8(src: __m128i, k: __mmask16, a: __m512i) -> __m128i {
    let convert = _mm512_cvtepi32_epi8(a).as_i8x16();
    transmute(simd_select_bitmask(k, convert, src.as_i8x16()))
}

/// Convert packed 32-bit integers in a to packed 8-bit integers with truncation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtepi32_epi8&expand=1439)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovdb))]
pub unsafe fn _mm512_maskz_cvtepi32_epi8(k: __mmask16, a: __m512i) -> __m128i {
    let convert = _mm512_cvtepi32_epi8(a).as_i8x16();
    let zero = _mm_setzero_si128().as_i8x16();
    transmute(simd_select_bitmask(k, convert, zero))
}

/// Convert packed 64-bit integers in a to packed 32-bit integers with truncation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtepi64_epi32&expand=1481)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovqd))]
pub unsafe fn _mm512_cvtepi64_epi32(a: __m512i) -> __m256i {
    let a = a.as_i64x8();
    transmute::<i32x8, _>(simd_cast(a))
}

/// Convert packed 64-bit integers in a to packed 32-bit integers with truncation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtepi64_epi32&expand=1482)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovqd))]
pub unsafe fn _mm512_mask_cvtepi64_epi32(src: __m256i, k: __mmask8, a: __m512i) -> __m256i {
    let convert = _mm512_cvtepi64_epi32(a).as_i32x8();
    transmute(simd_select_bitmask(k, convert, src.as_i32x8()))
}

/// Convert packed 64-bit integers in a to packed 32-bit integers with truncation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtepi64_epi32&expand=1483)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovqd))]
pub unsafe fn _mm512_maskz_cvtepi64_epi32(k: __mmask8, a: __m512i) -> __m256i {
    let convert = _mm512_cvtepi64_epi32(a).as_i32x8();
    let zero = _mm256_setzero_si256().as_i32x8();
    transmute(simd_select_bitmask(k, convert, zero))
}

/// Convert packed 64-bit integers in a to packed 16-bit integers with truncation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtepi64_epi16&expand=1472)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovqw))]
pub unsafe fn _mm512_cvtepi64_epi16(a: __m512i) -> __m128i {
    let a = a.as_i64x8();
    transmute::<i16x8, _>(simd_cast(a))
}

/// Convert packed 64-bit integers in a to packed 16-bit integers with truncation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtepi64_epi16&expand=1473)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovqw))]
pub unsafe fn _mm512_mask_cvtepi64_epi16(src: __m128i, k: __mmask8, a: __m512i) -> __m128i {
    let convert = _mm512_cvtepi64_epi16(a).as_i16x8();
    transmute(simd_select_bitmask(k, convert, src.as_i16x8()))
}

/// Convert packed 64-bit integers in a to packed 16-bit integers with truncation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtepi64_epi16&expand=1474)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovqw))]
pub unsafe fn _mm512_maskz_cvtepi64_epi16(k: __mmask8, a: __m512i) -> __m128i {
    let convert = _mm512_cvtepi64_epi16(a).as_i16x8();
    let zero = _mm_setzero_si128().as_i16x8();
    transmute(simd_select_bitmask(k, convert, zero))
}

/// Convert packed 64-bit integers in a to packed 8-bit integers with truncation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtepi64_epi8&expand=1490)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovqb))]
pub unsafe fn _mm512_cvtepi64_epi8(a: __m512i) -> __m128i {
    transmute(vpmovqb(
        a.as_i64x8(),
        _mm_setzero_si128().as_i8x16(),
        0b11111111,
    ))
}

/// Convert packed 64-bit integers in a to packed 8-bit integers with truncation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtepi64_epi8&expand=1491)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovqb))]
pub unsafe fn _mm512_mask_cvtepi64_epi8(src: __m128i, k: __mmask8, a: __m512i) -> __m128i {
    transmute(vpmovqb(a.as_i64x8(), src.as_i8x16(), k))
}

/// Convert packed 64-bit integers in a to packed 8-bit integers with truncation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtepi64_epi8&expand=1492)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovqb))]
pub unsafe fn _mm512_maskz_cvtepi64_epi8(k: __mmask8, a: __m512i) -> __m128i {
    transmute(vpmovqb(a.as_i64x8(), _mm_setzero_si128().as_i8x16(), k))
}

/// Convert packed signed 32-bit integers in a to packed 16-bit integers with signed saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtsepi32_epi16&expand=1819)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsdw))]
pub unsafe fn _mm512_cvtsepi32_epi16(a: __m512i) -> __m256i {
    transmute(vpmovsdw(
        a.as_i32x16(),
        _mm256_setzero_si256().as_i16x16(),
        0b11111111_11111111,
    ))
}

/// Convert packed signed 32-bit integers in a to packed 16-bit integers with signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtsepi32_epi16&expand=1820)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsdw))]
pub unsafe fn _mm512_mask_cvtsepi32_epi16(src: __m256i, k: __mmask16, a: __m512i) -> __m256i {
    transmute(vpmovsdw(a.as_i32x16(), src.as_i16x16(), k))
}

/// Convert packed signed 32-bit integers in a to packed 16-bit integers with signed saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtsepi32_epi16&expand=1819)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsdw))]
pub unsafe fn _mm512_maskz_cvtsepi32_epi16(k: __mmask16, a: __m512i) -> __m256i {
    transmute(vpmovsdw(
        a.as_i32x16(),
        _mm256_setzero_si256().as_i16x16(),
        k,
    ))
}

/// Convert packed signed 32-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtsepi32_epi8&expand=1828)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsdb))]
pub unsafe fn _mm512_cvtsepi32_epi8(a: __m512i) -> __m128i {
    transmute(vpmovsdb(
        a.as_i32x16(),
        _mm_setzero_si128().as_i8x16(),
        0b11111111_11111111,
    ))
}

/// Convert packed signed 32-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtsepi32_epi8&expand=1829)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsdb))]
pub unsafe fn _mm512_mask_cvtsepi32_epi8(src: __m128i, k: __mmask16, a: __m512i) -> __m128i {
    transmute(vpmovsdb(a.as_i32x16(), src.as_i8x16(), k))
}

/// Convert packed signed 32-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtsepi32_epi8&expand=1830)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsdb))]
pub unsafe fn _mm512_maskz_cvtsepi32_epi8(k: __mmask16, a: __m512i) -> __m128i {
    transmute(vpmovsdb(a.as_i32x16(), _mm_setzero_si128().as_i8x16(), k))
}

/// Convert packed signed 64-bit integers in a to packed 32-bit integers with signed saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtsepi64_epi32&expand=1852)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsqd))]
pub unsafe fn _mm512_cvtsepi64_epi32(a: __m512i) -> __m256i {
    transmute(vpmovsqd(
        a.as_i64x8(),
        _mm256_setzero_si256().as_i32x8(),
        0b11111111,
    ))
}

/// Convert packed signed 64-bit integers in a to packed 32-bit integers with signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtsepi64_epi32&expand=1853)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsqd))]
pub unsafe fn _mm512_mask_cvtsepi64_epi32(src: __m256i, k: __mmask8, a: __m512i) -> __m256i {
    transmute(vpmovsqd(a.as_i64x8(), src.as_i32x8(), k))
}

/// Convert packed signed 64-bit integers in a to packed 32-bit integers with signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtsepi64_epi32&expand=1854)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsqd))]
pub unsafe fn _mm512_maskz_cvtsepi64_epi32(k: __mmask8, a: __m512i) -> __m256i {
    transmute(vpmovsqd(a.as_i64x8(), _mm256_setzero_si256().as_i32x8(), k))
}

/// Convert packed signed 64-bit integers in a to packed 16-bit integers with signed saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtsepi64_epi16&expand=1843)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsqw))]
pub unsafe fn _mm512_cvtsepi64_epi16(a: __m512i) -> __m128i {
    transmute(vpmovsqw(
        a.as_i64x8(),
        _mm_setzero_si128().as_i16x8(),
        0b11111111,
    ))
}

/// Convert packed signed 64-bit integers in a to packed 16-bit integers with signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtsepi64_epi16&expand=1844)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsqw))]
pub unsafe fn _mm512_mask_cvtsepi64_epi16(src: __m128i, k: __mmask8, a: __m512i) -> __m128i {
    transmute(vpmovsqw(a.as_i64x8(), src.as_i16x8(), k))
}

/// Convert packed signed 64-bit integers in a to packed 16-bit integers with signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtsepi64_epi16&expand=1845)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsqw))]
pub unsafe fn _mm512_maskz_cvtsepi64_epi16(k: __mmask8, a: __m512i) -> __m128i {
    transmute(vpmovsqw(a.as_i64x8(), _mm_setzero_si128().as_i16x8(), k))
}

/// Convert packed signed 64-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtsepi64_epi8&expand=1861)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsqb))]
pub unsafe fn _mm512_cvtsepi64_epi8(a: __m512i) -> __m128i {
    transmute(vpmovsqb(
        a.as_i64x8(),
        _mm_setzero_si128().as_i8x16(),
        0b11111111,
    ))
}

/// Convert packed signed 64-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtsepi64_epi8&expand=1862)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsqb))]
pub unsafe fn _mm512_mask_cvtsepi64_epi8(src: __m128i, k: __mmask8, a: __m512i) -> __m128i {
    transmute(vpmovsqb(a.as_i64x8(), src.as_i8x16(), k))
}

/// Convert packed signed 64-bit integers in a to packed 8-bit integers with signed saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtsepi64_epi8&expand=1863)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovsqb))]
pub unsafe fn _mm512_maskz_cvtsepi64_epi8(k: __mmask8, a: __m512i) -> __m128i {
    transmute(vpmovsqb(a.as_i64x8(), _mm_setzero_si128().as_i8x16(), k))
}

/// Convert packed unsigned 32-bit integers in a to packed unsigned 16-bit integers with unsigned saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtusepi32_epi16&expand=2054)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovusdw))]
pub unsafe fn _mm512_cvtusepi32_epi16(a: __m512i) -> __m256i {
    transmute(vpmovusdw(
        a.as_u32x16(),
        _mm256_setzero_si256().as_u16x16(),
        0b11111111_11111111,
    ))
}

/// Convert packed unsigned 32-bit integers in a to packed unsigned 16-bit integers with unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtusepi32_epi16&expand=2055)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovusdw))]
pub unsafe fn _mm512_mask_cvtusepi32_epi16(src: __m256i, k: __mmask16, a: __m512i) -> __m256i {
    transmute(vpmovusdw(a.as_u32x16(), src.as_u16x16(), k))
}

/// Convert packed unsigned 32-bit integers in a to packed unsigned 16-bit integers with unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtusepi32_epi16&expand=2056)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovusdw))]
pub unsafe fn _mm512_maskz_cvtusepi32_epi16(k: __mmask16, a: __m512i) -> __m256i {
    transmute(vpmovusdw(
        a.as_u32x16(),
        _mm256_setzero_si256().as_u16x16(),
        k,
    ))
}

/// Convert packed unsigned 32-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtusepi32_epi8&expand=2063)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovusdb))]
pub unsafe fn _mm512_cvtusepi32_epi8(a: __m512i) -> __m128i {
    transmute(vpmovusdb(
        a.as_u32x16(),
        _mm_setzero_si128().as_u8x16(),
        0b11111111_11111111,
    ))
}

/// Convert packed unsigned 32-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtusepi32_epi8&expand=2064)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovusdb))]
pub unsafe fn _mm512_mask_cvtusepi32_epi8(src: __m128i, k: __mmask16, a: __m512i) -> __m128i {
    transmute(vpmovusdb(a.as_u32x16(), src.as_u8x16(), k))
}

/// Convert packed unsigned 32-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtusepi32_epi8&expand=2065)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovusdb))]
pub unsafe fn _mm512_maskz_cvtusepi32_epi8(k: __mmask16, a: __m512i) -> __m128i {
    transmute(vpmovusdb(a.as_u32x16(), _mm_setzero_si128().as_u8x16(), k))
}

/// Convert packed unsigned 64-bit integers in a to packed unsigned 32-bit integers with unsigned saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtusepi64_epi32&expand=2087)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovusqd))]
pub unsafe fn _mm512_cvtusepi64_epi32(a: __m512i) -> __m256i {
    transmute(vpmovusqd(
        a.as_u64x8(),
        _mm256_setzero_si256().as_u32x8(),
        0b11111111,
    ))
}

/// Convert packed unsigned 64-bit integers in a to packed unsigned 32-bit integers with unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtusepi64_epi32&expand=2088)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovusqd))]
pub unsafe fn _mm512_mask_cvtusepi64_epi32(src: __m256i, k: __mmask8, a: __m512i) -> __m256i {
    transmute(vpmovusqd(a.as_u64x8(), src.as_u32x8(), k))
}

/// Convert packed unsigned 64-bit integers in a to packed unsigned 32-bit integers with unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtusepi64_epi32&expand=2089)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovusqd))]
pub unsafe fn _mm512_maskz_cvtusepi64_epi32(k: __mmask8, a: __m512i) -> __m256i {
    transmute(vpmovusqd(
        a.as_u64x8(),
        _mm256_setzero_si256().as_u32x8(),
        k,
    ))
}

/// Convert packed unsigned 64-bit integers in a to packed unsigned 16-bit integers with unsigned saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtusepi64_epi16&expand=2078)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovusqw))]
pub unsafe fn _mm512_cvtusepi64_epi16(a: __m512i) -> __m128i {
    transmute(vpmovusqw(
        a.as_u64x8(),
        _mm_setzero_si128().as_u16x8(),
        0b11111111,
    ))
}

/// Convert packed unsigned 64-bit integers in a to packed unsigned 16-bit integers with unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtusepi64_epi16&expand=2079)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovusqw))]
pub unsafe fn _mm512_mask_cvtusepi64_epi16(src: __m128i, k: __mmask8, a: __m512i) -> __m128i {
    transmute(vpmovusqw(a.as_u64x8(), src.as_u16x8(), k))
}

/// Convert packed unsigned 64-bit integers in a to packed unsigned 16-bit integers with unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtusepi64_epi16&expand=2080)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovusqw))]
pub unsafe fn _mm512_maskz_cvtusepi64_epi16(k: __mmask8, a: __m512i) -> __m128i {
    transmute(vpmovusqw(a.as_u64x8(), _mm_setzero_si128().as_u16x8(), k))
}

/// Convert packed unsigned 64-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cvtusepi64_epi8&expand=2096)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovusqb))]
pub unsafe fn _mm512_cvtusepi64_epi8(a: __m512i) -> __m128i {
    transmute(vpmovusqb(
        a.as_u64x8(),
        _mm_setzero_si128().as_u8x16(),
        0b11111111,
    ))
}

/// Convert packed unsigned 64-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cvtusepi64_epi8&expand=2097)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovusqb))]
pub unsafe fn _mm512_mask_cvtusepi64_epi8(src: __m128i, k: __mmask8, a: __m512i) -> __m128i {
    transmute(vpmovusqb(a.as_u64x8(), src.as_u8x16(), k))
}

/// Convert packed unsigned 64-bit integers in a to packed unsigned 8-bit integers with unsigned saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_cvtusepi64_epi8&expand=2098)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpmovusqb))]
pub unsafe fn _mm512_maskz_cvtusepi64_epi8(k: __mmask8, a: __m512i) -> __m128i {
    transmute(vpmovusqb(a.as_u64x8(), _mm_setzero_si128().as_u8x16(), k))
}

/// Shift packed 32-bit integers in a left by imm8 while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_slli_epi32&expand=5310)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpslld, imm8 = 5))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_slli_epi32(a: __m512i, imm8: u32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vpsllid(a.as_i32x16(), $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(r)
}

/// Shift packed 32-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_slli_epi32&expand=5308)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpslld, imm8 = 5))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_slli_epi32(src: __m512i, k: __mmask16, a: __m512i, imm8: u32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vpsllid(a.as_i32x16(), $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
}

/// Shift packed 32-bit integers in a left by imm8 while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_slli_epi32&expand=5309)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpslld, imm8 = 5))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_slli_epi32(k: __mmask16, a: __m512i, imm8: u32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vpsllid(a.as_i32x16(), $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 32-bit integers in a right by imm8 while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_srli_epi32&expand=5522)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrld, imm8 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_srli_epi32(a: __m512i, imm8: u32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vpsrlid(a.as_i32x16(), $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(r)
}

/// Shift packed 32-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_srli_epi32&expand=5520)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrld, imm8 = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_srli_epi32(src: __m512i, k: __mmask16, a: __m512i, imm8: u32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vpsrlid(a.as_i32x16(), $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
}

/// Shift packed 32-bit integers in a right by imm8 while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_srli_epi32&expand=5521)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrld, imm8 = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_srli_epi32(k: __mmask16, a: __m512i, imm8: u32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vpsrlid(a.as_i32x16(), $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 32-bit integers in a left by count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sll_epi32&expand=5280)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpslld))]
pub unsafe fn _mm512_sll_epi32(a: __m512i, count: __m128i) -> __m512i {
    transmute(vpslld(a.as_i32x16(), count.as_i32x4()))
}

/// Shift packed 32-bit integers in a left by count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sll_epi32&expand=5278)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpslld))]
pub unsafe fn _mm512_mask_sll_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    count: __m128i,
) -> __m512i {
    let shf = _mm512_sll_epi32(a, count).as_i32x16();
    transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
}

/// Shift packed 32-bit integers in a left by count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sll_epi32&expand=5279)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpslld))]
pub unsafe fn _mm512_maskz_sll_epi32(k: __mmask16, a: __m512i, count: __m128i) -> __m512i {
    let shf = _mm512_sll_epi32(a, count).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 32-bit integers in a right by count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_srl_epi32&expand=5492)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrld))]
pub unsafe fn _mm512_srl_epi32(a: __m512i, count: __m128i) -> __m512i {
    transmute(vpsrld(a.as_i32x16(), count.as_i32x4()))
}

/// Shift packed 32-bit integers in a right by count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_srl_epi32&expand=5490)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrld))]
pub unsafe fn _mm512_mask_srl_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    count: __m128i,
) -> __m512i {
    let shf = _mm512_srl_epi32(a, count).as_i32x16();
    transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
}

/// Shift packed 32-bit integers in a right by count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_srl_epi32&expand=5491)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrld))]
pub unsafe fn _mm512_maskz_srl_epi32(k: __mmask16, a: __m512i, count: __m128i) -> __m512i {
    let shf = _mm512_srl_epi32(a, count).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 32-bit integers in a right by count while shifting in sign bits, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sra_epi32&expand=5407)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrad))]
pub unsafe fn _mm512_sra_epi32(a: __m512i, count: __m128i) -> __m512i {
    transmute(vpsrad(a.as_i32x16(), count.as_i32x4()))
}

/// Shift packed 32-bit integers in a right by count while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sra_epi32&expand=5405)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrad))]
pub unsafe fn _mm512_mask_sra_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    count: __m128i,
) -> __m512i {
    let shf = _mm512_sra_epi32(a, count).as_i32x16();
    transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
}

/// Shift packed 32-bit integers in a right by count while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sra_epi32&expand=5406)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrad))]
pub unsafe fn _mm512_maskz_sra_epi32(k: __mmask16, a: __m512i, count: __m128i) -> __m512i {
    let shf = _mm512_sra_epi32(a, count).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 32-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_srai_epi32&expand=5436)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrad, imm8 = 1))]
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_srai_epi32(a: __m512i, imm8: u32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vpsraid(a.as_i32x16(), $imm8)
        };
    }
    let r = constify_imm8_sae!(imm8, call);
    transmute(r)
}

/// Shift packed 32-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_srai_epi32&expand=5434)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrad, imm8 = 1))]
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_srai_epi32(src: __m512i, k: __mmask16, a: __m512i, imm8: u32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vpsraid(a.as_i32x16(), $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
}

/// Shift packed 32-bit integers in a right by imm8 while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_srai_epi32&expand=5435)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrad, imm8 = 1))]
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_srai_epi32(k: __mmask16, a: __m512i, imm8: u32) -> __m512i {
    macro_rules! call {
        ($imm8:expr) => {
            vpsraid(a.as_i32x16(), $imm8)
        };
    }
    let shf = constify_imm8_sae!(imm8, call);
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 32-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_srav_epi32&expand=5465)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsravd))]
pub unsafe fn _mm512_srav_epi32(a: __m512i, count: __m512i) -> __m512i {
    transmute(vpsravd(a.as_i32x16(), count.as_i32x16()))
}

/// Shift packed 32-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_srav_epi32&expand=5463)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsravd))]
pub unsafe fn _mm512_mask_srav_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    count: __m512i,
) -> __m512i {
    let shf = _mm512_srav_epi32(a, count).as_i32x16();
    transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
}

/// Shift packed 32-bit integers in a right by the amount specified by the corresponding element in count while shifting in sign bits, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_srav_epi32&expand=5464)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsravd))]
pub unsafe fn _mm512_maskz_srav_epi32(k: __mmask16, a: __m512i, count: __m512i) -> __m512i {
    let shf = _mm512_srav_epi32(a, count).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 32-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_sllv_epi32&expand=5342)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsllvd))]
pub unsafe fn _mm512_sllv_epi32(a: __m512i, count: __m512i) -> __m512i {
    transmute(vpsllvd(a.as_i32x16(), count.as_i32x16()))
}

/// Shift packed 32-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_sllv_epi32&expand=5340)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsllvd))]
pub unsafe fn _mm512_mask_sllv_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    count: __m512i,
) -> __m512i {
    let shf = _mm512_sllv_epi32(a, count).as_i32x16();
    transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
}

/// Shift packed 32-bit integers in a left by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_sllv_epi32&expand=5341)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsllvd))]
pub unsafe fn _mm512_maskz_sllv_epi32(k: __mmask16, a: __m512i, count: __m512i) -> __m512i {
    let shf = _mm512_sllv_epi32(a, count).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shift packed 32-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_srlv_epi32&expand=5554)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrlvd))]
pub unsafe fn _mm512_srlv_epi32(a: __m512i, count: __m512i) -> __m512i {
    transmute(vpsrlvd(a.as_i32x16(), count.as_i32x16()))
}

/// Shift packed 32-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_srlv_epi32&expand=5552)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrlvd))]
pub unsafe fn _mm512_mask_srlv_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    count: __m512i,
) -> __m512i {
    let shf = _mm512_srlv_epi32(a, count).as_i32x16();
    transmute(simd_select_bitmask(k, shf, src.as_i32x16()))
}

/// Shift packed 32-bit integers in a right by the amount specified by the corresponding element in count while shifting in zeros, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_srlv_epi32&expand=5553)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpsrlvd))]
pub unsafe fn _mm512_maskz_srlv_epi32(k: __mmask16, a: __m512i, count: __m512i) -> __m512i {
    let shf = _mm512_srlv_epi32(a, count).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, shf, zero))
}

/// Shuffle 32-bit integers in a across lanes using the corresponding index in idx, and store the results in dst. Note that this intrinsic shuffles across 128-bit lanes, unlike past intrinsics that use the permutevar name. This intrinsic is identical to _mm512_permutexvar_epi32, and it is recommended that you use that intrinsic name.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_permutevar_epi32&expand=4182)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vperm))] //should be vpermd, but generate vpermps. It generates vpermd with mask
pub unsafe fn _mm512_permutevar_epi32(idx: __m512i, a: __m512i) -> __m512i {
    transmute(vpermd(a.as_i32x16(), idx.as_i32x16()))
}

/// Shuffle 32-bit integers in a across lanes using the corresponding index in idx, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). Note that this intrinsic shuffles across 128-bit lanes, unlike past intrinsics that use the permutevar name. This intrinsic is identical to _mm512_mask_permutexvar_epi32, and it is recommended that you use that intrinsic name.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_permutevar_epi32&expand=4181)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpermd))]
pub unsafe fn _mm512_mask_permutevar_epi32(
    src: __m512i,
    k: __mmask16,
    idx: __m512i,
    a: __m512i,
) -> __m512i {
    let permute = _mm512_permutevar_epi32(idx, a).as_i32x16();
    transmute(simd_select_bitmask(k, permute, src.as_i32x16()))
}

/// Shuffle 32-bit integers in a across lanes using the corresponding index in idx, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_permutexvar_epi32&expand=4301)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vperm))] //should be vpermd, but generate vpermps. It generates vpermd with mask
pub unsafe fn _mm512_permutexvar_epi32(idx: __m512i, a: __m512i) -> __m512i {
    transmute(vpermd(a.as_i32x16(), idx.as_i32x16()))
}

/// Shuffle 32-bit integers in a across lanes using the corresponding index in idx, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_permutexvar_epi32&expand=4299)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpermd))]
pub unsafe fn _mm512_mask_permutexvar_epi32(
    src: __m512i,
    k: __mmask16,
    idx: __m512i,
    a: __m512i,
) -> __m512i {
    let permute = _mm512_permutexvar_epi32(idx, a).as_i32x16();
    transmute(simd_select_bitmask(k, permute, src.as_i32x16()))
}

/// Shuffle 32-bit integers in a across lanes using the corresponding index in idx, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_permutexvar_epi32&expand=4300)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpermd))]
pub unsafe fn _mm512_maskz_permutexvar_epi32(k: __mmask16, idx: __m512i, a: __m512i) -> __m512i {
    let permute = _mm512_permutexvar_epi32(idx, a).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, permute, zero))
}

/// Shuffle 32-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_permutex2var_epi32&expand=4238)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vperm))] //vpermi2d or vpermt2d
pub unsafe fn _mm512_permutex2var_epi32(a: __m512i, idx: __m512i, b: __m512i) -> __m512i {
    transmute(vpermi2d(a.as_i32x16(), idx.as_i32x16(), b.as_i32x16()))
}

/// Shuffle 32-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from a when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_permutex2var_epi32&expand=4235)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpermt2d))]
pub unsafe fn _mm512_mask_permutex2var_epi32(
    a: __m512i,
    k: __mmask16,
    idx: __m512i,
    b: __m512i,
) -> __m512i {
    let permute = _mm512_permutex2var_epi32(a, idx, b).as_i32x16();
    transmute(simd_select_bitmask(k, permute, a.as_i32x16()))
}

/// Shuffle 32-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_permutex2var_epi32&expand=4237)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vperm))] //vpermi2d or vpermt2d
pub unsafe fn _mm512_maskz_permutex2var_epi32(
    k: __mmask16,
    a: __m512i,
    idx: __m512i,
    b: __m512i,
) -> __m512i {
    let permute = _mm512_permutex2var_epi32(a, idx, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, permute, zero))
}

/// Shuffle 32-bit integers in a and b across lanes using the corresponding selector and index in idx, and store the results in dst using writemask k (elements are copied from idx when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask2_permutex2var_epi32&expand=4236)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpermi2d))]
pub unsafe fn _mm512_mask2_permutex2var_epi32(
    a: __m512i,
    idx: __m512i,
    k: __mmask16,
    b: __m512i,
) -> __m512i {
    let permute = _mm512_permutex2var_epi32(a, idx, b).as_i32x16();
    transmute(simd_select_bitmask(k, permute, idx.as_i32x16()))
}

/// Shuffle single-precision (32-bit) floating-point elements in a within 128-bit lanes using the control in imm8, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_shuffle_epi32&expand=5150)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpermilps, imm8 = 9))] //should be vpshufd, but generate vpermilps
#[rustc_args_required_const(1)]
pub unsafe fn _mm512_shuffle_epi32(a: __m512i, imm8: _MM_PERM_ENUM) -> __m512i {
    let imm8 = (imm8 & 0xFF) as u8;

    let a = a.as_i32x16();
    macro_rules! shuffle4 {
        (
            $a:expr,
            $b:expr,
            $c:expr,
            $d:expr,
            $e:expr,
            $f:expr,
            $g:expr,
            $h:expr,
            $i:expr,
            $j:expr,
            $k:expr,
            $l:expr,
            $m:expr,
            $n:expr,
            $o:expr,
            $p:expr
        ) => {
            simd_shuffle16(
                a,
                a,
                [
                    $a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k, $l, $m, $n, $o, $p,
                ],
            )
        };
    }
    macro_rules! shuffle3 {
        ($a:expr, $b:expr, $c:expr, $e:expr, $f:expr, $g:expr, $i:expr, $j:expr, $k:expr, $m:expr, $n:expr, $o:expr) => {
            match (imm8 >> 6) & 0x3 {
                0 => shuffle4!($a, $b, $c, 16, $e, $f, $g, 20, $i, $j, $k, 24, $m, $n, $o, 28),
                1 => shuffle4!($a, $b, $c, 17, $e, $f, $g, 21, $i, $j, $k, 25, $m, $n, $o, 29),
                2 => shuffle4!($a, $b, $c, 18, $e, $f, $g, 22, $i, $j, $k, 26, $m, $n, $o, 30),
                _ => shuffle4!($a, $b, $c, 19, $e, $f, $g, 23, $i, $j, $k, 27, $m, $n, $o, 31),
            }
        };
    }
    macro_rules! shuffle2 {
        ($a:expr, $b:expr, $e:expr, $f:expr, $i:expr, $j:expr, $m:expr, $n:expr) => {
            match (imm8 >> 4) & 0x3 {
                0 => shuffle3!($a, $b, 16, $e, $f, 20, $i, $j, 24, $m, $n, 28),
                1 => shuffle3!($a, $b, 17, $e, $f, 21, $i, $j, 25, $m, $n, 29),
                2 => shuffle3!($a, $b, 18, $e, $f, 22, $i, $j, 26, $m, $n, 30),
                _ => shuffle3!($a, $b, 19, $e, $f, 23, $i, $j, 27, $m, $n, 31),
            }
        };
    }
    macro_rules! shuffle1 {
        ($a:expr, $e:expr, $i: expr, $m: expr) => {
            match (imm8 >> 2) & 0x3 {
                0 => shuffle2!($a, 0, $e, 4, $i, 8, $m, 12),
                1 => shuffle2!($a, 1, $e, 5, $i, 9, $m, 13),
                2 => shuffle2!($a, 2, $e, 6, $i, 10, $m, 14),
                _ => shuffle2!($a, 3, $e, 7, $i, 11, $m, 15),
            }
        };
    }
    let r: i32x16 = match imm8 & 0x3 {
        0 => shuffle1!(0, 4, 8, 12),
        1 => shuffle1!(1, 5, 9, 13),
        2 => shuffle1!(2, 6, 10, 14),
        _ => shuffle1!(3, 7, 11, 15),
    };
    transmute(r)
}

/// Shuffle 32-bit integers in a within 128-bit lanes using the control in imm8, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_shuffle_epi32&expand=5148)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpshufd, imm8 = 9))] //should be vpshufd, but generate vpermilps
#[rustc_args_required_const(3)]
pub unsafe fn _mm512_mask_shuffle_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    imm8: _MM_PERM_ENUM,
) -> __m512i {
    let imm8 = (imm8 & 0xFF) as u8;

    let a = a.as_i32x16();
    macro_rules! shuffle4 {
        (
            $a:expr,
            $b:expr,
            $c:expr,
            $d:expr,
            $e:expr,
            $f:expr,
            $g:expr,
            $h:expr,
            $i:expr,
            $j:expr,
            $k:expr,
            $l:expr,
            $m:expr,
            $n:expr,
            $o:expr,
            $p:expr
        ) => {
            simd_shuffle16(
                a,
                a,
                [
                    $a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k, $l, $m, $n, $o, $p,
                ],
            )
        };
    }
    macro_rules! shuffle3 {
        ($a:expr, $b:expr, $c:expr, $e:expr, $f:expr, $g:expr, $i:expr, $j:expr, $k:expr, $m:expr, $n:expr, $o:expr) => {
            match (imm8 >> 6) & 0x3 {
                0 => shuffle4!($a, $b, $c, 16, $e, $f, $g, 20, $i, $j, $k, 24, $m, $n, $o, 28),
                1 => shuffle4!($a, $b, $c, 17, $e, $f, $g, 21, $i, $j, $k, 25, $m, $n, $o, 29),
                2 => shuffle4!($a, $b, $c, 18, $e, $f, $g, 22, $i, $j, $k, 26, $m, $n, $o, 30),
                _ => shuffle4!($a, $b, $c, 19, $e, $f, $g, 23, $i, $j, $k, 27, $m, $n, $o, 31),
            }
        };
    }
    macro_rules! shuffle2 {
        ($a:expr, $b:expr, $e:expr, $f:expr, $i:expr, $j:expr, $m:expr, $n:expr) => {
            match (imm8 >> 4) & 0x3 {
                0 => shuffle3!($a, $b, 16, $e, $f, 20, $i, $j, 24, $m, $n, 28),
                1 => shuffle3!($a, $b, 17, $e, $f, 21, $i, $j, 25, $m, $n, 29),
                2 => shuffle3!($a, $b, 18, $e, $f, 22, $i, $j, 26, $m, $n, 30),
                _ => shuffle3!($a, $b, 19, $e, $f, 23, $i, $j, 27, $m, $n, 31),
            }
        };
    }
    macro_rules! shuffle1 {
        ($a:expr, $e:expr, $i: expr, $m: expr) => {
            match (imm8 >> 2) & 0x3 {
                0 => shuffle2!($a, 0, $e, 4, $i, 8, $m, 12),
                1 => shuffle2!($a, 1, $e, 5, $i, 9, $m, 13),
                2 => shuffle2!($a, 2, $e, 6, $i, 10, $m, 14),
                _ => shuffle2!($a, 3, $e, 7, $i, 11, $m, 15),
            }
        };
    }
    let shuffle: i32x16 = match imm8 & 0x3 {
        0 => shuffle1!(0, 4, 8, 12),
        1 => shuffle1!(1, 5, 9, 13),
        2 => shuffle1!(2, 6, 10, 14),
        _ => shuffle1!(3, 7, 11, 15),
    };
    transmute(simd_select_bitmask(k, shuffle, src.as_i32x16()))
}

/// Shuffle 32-bit integers in a within 128-bit lanes using the control in imm8, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_shuffle_epi32&expand=5149)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpshufd, imm8 = 9))] //should be vpshufd, but generate vpermilps
#[rustc_args_required_const(2)]
pub unsafe fn _mm512_maskz_shuffle_epi32(k: __mmask16, a: __m512i, imm8: _MM_PERM_ENUM) -> __m512i {
    let imm8 = (imm8 & 0xFF) as u8;

    let a = a.as_i32x16();
    macro_rules! shuffle4 {
        (
            $a:expr,
            $b:expr,
            $c:expr,
            $d:expr,
            $e:expr,
            $f:expr,
            $g:expr,
            $h:expr,
            $i:expr,
            $j:expr,
            $k:expr,
            $l:expr,
            $m:expr,
            $n:expr,
            $o:expr,
            $p:expr
        ) => {
            simd_shuffle16(
                a,
                a,
                [
                    $a, $b, $c, $d, $e, $f, $g, $h, $i, $j, $k, $l, $m, $n, $o, $p,
                ],
            )
        };
    }
    macro_rules! shuffle3 {
        ($a:expr, $b:expr, $c:expr, $e:expr, $f:expr, $g:expr, $i:expr, $j:expr, $k:expr, $m:expr, $n:expr, $o:expr) => {
            match (imm8 >> 6) & 0x3 {
                0 => shuffle4!($a, $b, $c, 16, $e, $f, $g, 20, $i, $j, $k, 24, $m, $n, $o, 28),
                1 => shuffle4!($a, $b, $c, 17, $e, $f, $g, 21, $i, $j, $k, 25, $m, $n, $o, 29),
                2 => shuffle4!($a, $b, $c, 18, $e, $f, $g, 22, $i, $j, $k, 26, $m, $n, $o, 30),
                _ => shuffle4!($a, $b, $c, 19, $e, $f, $g, 23, $i, $j, $k, 27, $m, $n, $o, 31),
            }
        };
    }
    macro_rules! shuffle2 {
        ($a:expr, $b:expr, $e:expr, $f:expr, $i:expr, $j:expr, $m:expr, $n:expr) => {
            match (imm8 >> 4) & 0x3 {
                0 => shuffle3!($a, $b, 16, $e, $f, 20, $i, $j, 24, $m, $n, 28),
                1 => shuffle3!($a, $b, 17, $e, $f, 21, $i, $j, 25, $m, $n, 29),
                2 => shuffle3!($a, $b, 18, $e, $f, 22, $i, $j, 26, $m, $n, 30),
                _ => shuffle3!($a, $b, 19, $e, $f, 23, $i, $j, 27, $m, $n, 31),
            }
        };
    }
    macro_rules! shuffle1 {
        ($a:expr, $e:expr, $i: expr, $m: expr) => {
            match (imm8 >> 2) & 0x3 {
                0 => shuffle2!($a, 0, $e, 4, $i, 8, $m, 12),
                1 => shuffle2!($a, 1, $e, 5, $i, 9, $m, 13),
                2 => shuffle2!($a, 2, $e, 6, $i, 10, $m, 14),
                _ => shuffle2!($a, 3, $e, 7, $i, 11, $m, 15),
            }
        };
    }
    let shuffle: i32x16 = match imm8 & 0x3 {
        0 => shuffle1!(0, 4, 8, 12),
        1 => shuffle1!(1, 5, 9, 13),
        2 => shuffle1!(2, 6, 10, 14),
        _ => shuffle1!(3, 7, 11, 15),
    };
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, shuffle, zero))
}

/// Unpack and interleave 32-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_unpackhi_epi32&expand=6021)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vunpckhps))] //should be vpunpckhdq
pub unsafe fn _mm512_unpackhi_epi32(a: __m512i, b: __m512i) -> __m512i {
    let a = a.as_i32x16();
    let b = b.as_i32x16();
    let r: i32x16 = simd_shuffle16(
        a,
        b,
        [
            2,
            18,
            3,
            19,
            2 + 4,
            18 + 4,
            3 + 4,
            19 + 4,
            2 + 8,
            18 + 8,
            3 + 8,
            19 + 8,
            2 + 12,
            18 + 12,
            3 + 12,
            19 + 12,
        ],
    );
    transmute(r)
}

/// Unpack and interleave 32-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_unpackhi_epi32&expand=6019)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpunpckhdq))]
pub unsafe fn _mm512_mask_unpackhi_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let unpackhi = _mm512_unpackhi_epi32(a, b).as_i32x16();
    transmute(simd_select_bitmask(k, unpackhi, src.as_i32x16()))
}

/// Unpack and interleave 32-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_unpackhi_epi32&expand=6020)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpunpckhdq))]
pub unsafe fn _mm512_maskz_unpackhi_epi32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let unpackhi = _mm512_unpackhi_epi32(a, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, unpackhi, zero))
}

/// Unpack and interleave 64-bit integers from the high half of each 128-bit lane in a and b, and
/// store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_unpackhi_epi64&expand=6030)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vunpckhpd))] //should be vpunpckhqdq
pub unsafe fn _mm512_unpackhi_epi64(a: __m512i, b: __m512i) -> __m512i {
    simd_shuffle8(a, b, [1, 9, 1 + 2, 9 + 2, 1 + 4, 9 + 4, 1 + 6, 9 + 6])
}

/// Unpack and interleave 64-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_unpackhi_epi64&expand=6028)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpunpckhqdq))]
pub unsafe fn _mm512_mask_unpackhi_epi64(
    src: __m512i,
    k: __mmask8,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let unpackhi = _mm512_unpackhi_epi64(a, b).as_i64x8();
    transmute(simd_select_bitmask(k, unpackhi, src.as_i64x8()))
}

/// Unpack and interleave 64-bit integers from the high half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_unpackhi_epi64&expand=6029)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpunpckhqdq))]
pub unsafe fn _mm512_maskz_unpackhi_epi64(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let unpackhi = _mm512_unpackhi_epi64(a, b).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, unpackhi, zero))
}

/// Unpack and interleave 32-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_unpacklo_epi32&expand=6078)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vunpcklps))] //should be vpunpckldq
pub unsafe fn _mm512_unpacklo_epi32(a: __m512i, b: __m512i) -> __m512i {
    let a = a.as_i32x16();
    let b = b.as_i32x16();
    let r: i32x16 = simd_shuffle16(
        a,
        b,
        [
            0,
            16,
            1,
            17,
            0 + 4,
            16 + 4,
            1 + 4,
            17 + 4,
            0 + 8,
            16 + 8,
            1 + 8,
            17 + 8,
            0 + 12,
            16 + 12,
            1 + 12,
            17 + 12,
        ],
    );
    transmute(r)
}

/// Unpack and interleave 32-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_unpacklo_epi32&expand=6076)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpunpckldq))]
pub unsafe fn _mm512_mask_unpacklo_epi32(
    src: __m512i,
    k: __mmask16,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let unpackhi = _mm512_unpacklo_epi32(a, b).as_i32x16();
    transmute(simd_select_bitmask(k, unpackhi, src.as_i32x16()))
}

/// Unpack and interleave 32-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_unpacklo_epi32&expand=6077)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpunpckldq))]
pub unsafe fn _mm512_maskz_unpacklo_epi32(k: __mmask16, a: __m512i, b: __m512i) -> __m512i {
    let unpackhi = _mm512_unpacklo_epi32(a, b).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, unpackhi, zero))
}

/// Unpack and interleave 64-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_unpacklo_epi64&expand=6087)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vunpcklpd))] //should be vpunpcklqdq
pub unsafe fn _mm512_unpacklo_epi64(a: __m512i, b: __m512i) -> __m512i {
    simd_shuffle8(a, b, [0, 8, 0 + 2, 8 + 2, 0 + 4, 8 + 4, 0 + 6, 8 + 6])
}

/// Unpack and interleave 64-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_unpacklo_epi64&expand=6085)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpunpcklqdq))]
pub unsafe fn _mm512_mask_unpacklo_epi64(
    src: __m512i,
    k: __mmask8,
    a: __m512i,
    b: __m512i,
) -> __m512i {
    let unpackhi = _mm512_unpacklo_epi64(a, b).as_i64x8();
    transmute(simd_select_bitmask(k, unpackhi, src.as_i64x8()))
}

/// Unpack and interleave 64-bit integers from the low half of each 128-bit lane in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_unpacklo_epi64&expand=6086)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpunpcklqdq))]
pub unsafe fn _mm512_maskz_unpacklo_epi64(k: __mmask8, a: __m512i, b: __m512i) -> __m512i {
    let unpackhi = _mm512_unpacklo_epi64(a, b).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, unpackhi, zero))
}

/// Broadcast the low packed 32-bit integer from a to all elements of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_broadcastd_epi32&expand=545)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vbroadcast))] //should be vpbroadcastd
pub unsafe fn _mm512_broadcastd_epi32(a: __m128i) -> __m512i {
    let a = _mm512_castsi128_si512(a).as_i32x16();
    let ret: i32x16 = simd_shuffle16(a, a, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    transmute(ret)
}

/// Broadcast the low packed 32-bit integer from a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_broadcastd_epi32&expand=546)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpbroadcast))] //should be vpbroadcastd
pub unsafe fn _mm512_mask_broadcastd_epi32(src: __m512i, k: __mmask16, a: __m128i) -> __m512i {
    let broadcast = _mm512_broadcastd_epi32(a).as_i32x16();
    transmute(simd_select_bitmask(k, broadcast, src.as_i32x16()))
}

/// Broadcast the low packed 32-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_broadcastd_epi32&expand=547)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpbroadcast))] //should be vpbroadcastd
pub unsafe fn _mm512_maskz_broadcastd_epi32(k: __mmask16, a: __m128i) -> __m512i {
    let broadcast = _mm512_broadcastd_epi32(a).as_i32x16();
    let zero = _mm512_setzero_si512().as_i32x16();
    transmute(simd_select_bitmask(k, broadcast, zero))
}

/// Broadcast the low packed 64-bit integer from a to all elements of dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_broadcastq_epi64&expand=560)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vbroadcas))] //should be vpbroadcastq
pub unsafe fn _mm512_broadcastq_epi64(a: __m128i) -> __m512i {
    simd_shuffle8(a, a, [0, 0, 0, 0, 0, 0, 0, 0])
}

/// Broadcast the low packed 64-bit integer from a to all elements of dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_broadcastq_epi64&expand=561)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpbroadcast))] //should be vpbroadcastq
pub unsafe fn _mm512_mask_broadcastq_epi64(src: __m512i, k: __mmask8, a: __m128i) -> __m512i {
    let broadcast = _mm512_broadcastq_epi64(a).as_i64x8();
    transmute(simd_select_bitmask(k, broadcast, src.as_i64x8()))
}

/// Broadcast the low packed 64-bit integer from a to all elements of dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_maskz_broadcastq_epi64&expand=562)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpbroadcast))] //should be vpbroadcastq
pub unsafe fn _mm512_maskz_broadcastq_epi64(k: __mmask8, a: __m128i) -> __m512i {
    let broadcast = _mm512_broadcastq_epi64(a).as_i64x8();
    let zero = _mm512_setzero_si512().as_i64x8();
    transmute(simd_select_bitmask(k, broadcast, zero))
}

/// Compute the bitwise AND of 16-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=kand_mask16&expand=3212)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(and))] // generate normal and code instead of kandw
pub unsafe fn _kand_mask16(a: __mmask16, b: __mmask16) -> __mmask16 {
    transmute(a & b)
}

/// Compute the bitwise AND of 16-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_kand&expand=3210)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(and))] // generate normal and code instead of kandw
pub unsafe fn _mm512_kand(a: __mmask16, b: __mmask16) -> __mmask16 {
    transmute(a & b)
}

/// Compute the bitwise OR of 16-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=kor_mask16&expand=3239)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(or))] // generate normal or code instead of korw
pub unsafe fn _kor_mask16(a: __mmask16, b: __mmask16) -> __mmask16 {
    transmute(a | b)
}

/// Compute the bitwise OR of 16-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_kor&expand=3237)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(or))] // generate normal or code instead of korw
pub unsafe fn _mm512_kor(a: __mmask16, b: __mmask16) -> __mmask16 {
    transmute(a | b)
}

/// Compute the bitwise XOR of 16-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=kxor_mask16&expand=3291)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(xor))] // generate normal xor code instead of kxorw
pub unsafe fn _kxor_mask16(a: __mmask16, b: __mmask16) -> __mmask16 {
    transmute(a ^ b)
}

/// Compute the bitwise XOR of 16-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_kxor&expand=3289)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(xor))] // generate normal xor code instead of kxorw
pub unsafe fn _mm512_kxor(a: __mmask16, b: __mmask16) -> __mmask16 {
    transmute(a ^ b)
}

/// Compute the bitwise NOT of 16-bit mask a, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=knot_mask16&expand=3233)
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _knot_mask16(a: __mmask16) -> __mmask16 {
    transmute(a ^ 0b11111111_11111111)
}

/// Compute the bitwise NOT of 16-bit mask a, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_knot&expand=3231)
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_knot(a: __mmask16) -> __mmask16 {
    transmute(a ^ 0b11111111_11111111)
}

/// Compute the bitwise NOT of 16-bit masks a and then AND with b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=kandn_mask16&expand=3218)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(not))] // generate normal and, not code instead of kandnw
pub unsafe fn _kandn_mask16(a: __mmask16, b: __mmask16) -> __mmask16 {
    _mm512_kand(_mm512_knot(a), b)
}

/// Compute the bitwise NOT of 16-bit masks a and then AND with b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_kandn&expand=3216)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(not))] // generate normal and code instead of kandw
pub unsafe fn _mm512_kandn(a: __mmask16, b: __mmask16) -> __mmask16 {
    _mm512_kand(_mm512_knot(a), b)
}

/// Compute the bitwise XNOR of 16-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=kxnor_mask16&expand=3285)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(xor))] // generate normal xor, not code instead of kxnorw
pub unsafe fn _kxnor_mask16(a: __mmask16, b: __mmask16) -> __mmask16 {
    _mm512_knot(_mm512_kxor(a, b))
}

/// Compute the bitwise XNOR of 16-bit masks a and b, and store the result in k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_kxnor&expand=3283)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(xor))] // generate normal and code instead of kandw
pub unsafe fn _mm512_kxnor(a: __mmask16, b: __mmask16) -> __mmask16 {
    _mm512_knot(_mm512_kxor(a, b))
}

/// Copy 16-bit mask a to k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=mm512_kmov&expand=3228)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(mov))] // generate normal and code instead of kmovw
pub unsafe fn _mm512_kmov(a: __mmask16) -> __mmask16 {
    let r: u16 = a;
    transmute(r)
}

/// Performs bitwise OR between k1 and k2, storing the result in dst. CF flag is set if dst consists of all 1's.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_kortestc&expand=3247)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(cmp))] // generate normal and code instead of kortestw
pub unsafe fn _mm512_kortestc(a: __mmask16, b: __mmask16) -> i32 {
    let r = a | b;
    if r == 0b11111111_11111111 {
        1
    } else {
        0
    }
}

/// Compute the bitwise AND of packed 32-bit integers in a and b, producing intermediate 32-bit values, and set the corresponding bit in result mask k if the intermediate value is non-zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_test_epi32_mask&expand=5890)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vptestmd))]
pub unsafe fn _mm512_test_epi32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    let and = _mm512_and_epi32(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_cmpneq_epi32_mask(and, zero)
}

/// Compute the bitwise AND of packed 32-bit integers in a and b, producing intermediate 32-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is non-zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_test_epi32_mask&expand=5889)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vptestmd))]
pub unsafe fn _mm512_mask_test_epi32_mask(k: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    let and = _mm512_and_epi32(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_mask_cmpneq_epi32_mask(k, and, zero)
}

/// Compute the bitwise AND of packed 64-bit integers in a and b, producing intermediate 64-bit values, and set the corresponding bit in result mask k if the intermediate value is non-zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_test_epi64_mask&expand=5896)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vptestmq))]
pub unsafe fn _mm512_test_epi64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    let and = _mm512_and_epi64(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_cmpneq_epi64_mask(and, zero)
}

/// Compute the bitwise AND of packed 64-bit integers in a and b, producing intermediate 64-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is non-zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_test_epi64_mask&expand=5895)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vptestmq))]
pub unsafe fn _mm512_mask_test_epi64_mask(k: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    let and = _mm512_and_epi64(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_mask_cmpneq_epi64_mask(k, and, zero)
}

/// Compute the bitwise NAND of packed 32-bit integers in a and b, producing intermediate 32-bit values, and set the corresponding bit in result mask k if the intermediate value is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_testn_epi32_mask&expand=5921)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vptestnmd))]
pub unsafe fn _mm512_testn_epi32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    let and = _mm512_and_epi32(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_cmpeq_epi32_mask(and, zero)
}

/// Compute the bitwise NAND of packed 32-bit integers in a and b, producing intermediate 32-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_testn_epi32_mask&expand=5920)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vptestnmd))]
pub unsafe fn _mm512_mask_testn_epi32_mask(k: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    let and = _mm512_and_epi32(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_mask_cmpeq_epi32_mask(k, and, zero)
}

/// Compute the bitwise NAND of packed 64-bit integers in a and b, producing intermediate 64-bit values, and set the corresponding bit in result mask k if the intermediate value is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_testn_epi64_mask&expand=5927)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vptestnmq))]
pub unsafe fn _mm512_testn_epi64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    let and = _mm512_and_epi64(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_cmpeq_epi64_mask(and, zero)
}

/// Compute the bitwise NAND of packed 64-bit integers in a and b, producing intermediate 64-bit values, and set the corresponding bit in result mask k (subject to writemask k) if the intermediate value is zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_testn_epi64_mask&expand=5926)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vptestnmq))]
pub unsafe fn _mm512_mask_testn_epi64_mask(k: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    let and = _mm512_and_epi64(a, b);
    let zero = _mm512_setzero_si512();
    _mm512_mask_cmpeq_epi64_mask(k, and, zero)
}
*/
/// Compare packed unsigned 16-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cmplt_epu16_mask&expand=1050)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<u16x32, _>(simd_lt(a.as_u16x32(), b.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cmple_epu16_mask&expand=990)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmplt_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cmple_epu8_mask&expand=1007)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<u8x64, _>(simd_lt(a.as_u8x64(), b.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cmple_epu8_mask&expand=1008)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmplt_epu8_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cmplt_epi16_mask&expand=1022)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<i16x32, _>(simd_lt(a.as_i16x32(), b.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cmplt_epi16_mask&expand=1023)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmplt_epi16_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cmplt_epi8_mask&expand=1044)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<i8x64, _>(simd_lt(a.as_i8x64(), b.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cmplt_epi8_mask&expand=1045)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmplt_epi8_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cmpgt_epu16_mask&expand=927)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpgt_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<u16x32, _>(simd_gt(a.as_u16x32(), b.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cmpgt_epu16_mask&expand=928)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpgt_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmpgt_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cmpgt_epu8_mask&expand=945)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpgt_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<u8x64, _>(simd_gt(a.as_u8x64(), b.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cmpgt_epu8_mask&expand=946)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpgt_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmpgt_epu8_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cmpgt_epi16_mask&expand=897)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpgt_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<i16x32, _>(simd_gt(a.as_i16x32(), b.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cmpgt_epi16_mask&expand=898)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpgt_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmpgt_epi16_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cmpgt_epi8_mask&expand=921)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpgt_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<i8x64, _>(simd_gt(a.as_i8x64(), b.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_mask_cmpgt_epi8_mask&expand=922)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpgt_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmpgt_epi8_mask(a, b) & k1
}

/*
/// Compare packed unsigned 32-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmple_epu32_mask&expand=995)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmple_epu32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<u32x16, _>(simd_le(a.as_u32x16(), b.as_u32x16()))
}

/// Compare packed unsigned 32-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmple_epu32_mask&expand=996)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmple_epu32_mask(k1: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmple_epu32_mask(a, b) & k1
}

/// Compare packed unsigned 32-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpge_epu32_mask&expand=873)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpge_epu32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<u32x16, _>(simd_ge(a.as_u32x16(), b.as_u32x16()))
}

/// Compare packed unsigned 32-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpge_epu32_mask&expand=874)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpge_epu32_mask(k1: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmpge_epu32_mask(a, b) & k1
}

/// Compare packed unsigned 32-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpeq_epu32_mask&expand=807)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpeq_epu32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<u32x16, _>(simd_eq(a.as_u32x16(), b.as_u32x16()))
}

/// Compare packed unsigned 32-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpeq_epu32_mask&expand=808)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpeq_epu32_mask(k1: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmpeq_epu32_mask(a, b) & k1
}

/// Compare packed unsigned 32-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpneq_epu32_mask&expand=1112)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpneq_epu32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<u32x16, _>(simd_ne(a.as_u32x16(), b.as_u32x16()))
}

/// Compare packed unsigned 32-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpneq_epu32_mask&expand=1113)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpneq_epu32_mask(k1: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmpneq_epu32_mask(a, b) & k1
}

/// Compare packed unsigned 32-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmp_epu32_mask&expand=721)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_cmp_epu32_mask(a: __m512i, b: __m512i, imm8: _MM_CMPINT_ENUM) -> __mmask16 {
    let neg_one = -1;
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpud(a.as_i32x16(), b.as_i32x16(), $imm3, neg_one)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed unsigned 32-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmp_epu32_mask&expand=722)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_mask_cmp_epu32_mask(
    k1: __mmask16,
    a: __m512i,
    b: __m512i,
    imm8: _MM_CMPINT_ENUM,
) -> __mmask16 {
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpud(a.as_i32x16(), b.as_i32x16(), $imm3, k1 as i16)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 32-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmple_epi32_mask&expand=971)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmple_epi32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<i32x16, _>(simd_le(a.as_i32x16(), b.as_i32x16()))
}

/// Compare packed signed 32-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmple_epi32_mask&expand=972)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmple_epi32_mask(k1: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmple_epi32_mask(a, b) & k1
}

/// Compare packed signed 32-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpge_epi32_mask&expand=849)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpge_epi32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<i32x16, _>(simd_ge(a.as_i32x16(), b.as_i32x16()))
}

/// Compare packed signed 32-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpge_epi32_mask&expand=850)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpge_epi32_mask(k1: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmpge_epi32_mask(a, b) & k1
}

/// Compare packed 32-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpeq_epi32_mask&expand=779)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpeq_epi32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<i32x16, _>(simd_eq(a.as_i32x16(), b.as_i32x16()))
}

/// Compare packed 32-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpeq_epi32_mask&expand=780)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpeq_epi32_mask(k1: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmpeq_epi32_mask(a, b) & k1
}

/// Compare packed 32-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpneq_epi32_mask&expand=1088)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpneq_epi32_mask(a: __m512i, b: __m512i) -> __mmask16 {
    simd_bitmask::<i32x16, _>(simd_ne(a.as_i32x16(), b.as_i32x16()))
}

/// Compare packed 32-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpneq_epi32_mask&expand=1089)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpneq_epi32_mask(k1: __mmask16, a: __m512i, b: __m512i) -> __mmask16 {
    _mm512_cmpneq_epi32_mask(a, b) & k1
}

/// Compare packed signed 32-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmp_epi32_mask&expand=697)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_cmp_epi32_mask(a: __m512i, b: __m512i, imm8: _MM_CMPINT_ENUM) -> __mmask16 {
    let neg_one = -1;
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpd(a.as_i32x16(), b.as_i32x16(), $imm3, neg_one)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 32-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmp_epi32_mask&expand=698)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_mask_cmp_epi32_mask(
    k1: __mmask16,
    a: __m512i,
    b: __m512i,
    imm8: _MM_CMPINT_ENUM,
) -> __mmask16 {
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpd(a.as_i32x16(), b.as_i32x16(), $imm3, k1 as i16)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed unsigned 64-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmplt_epu64_mask&expand=1062)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epu64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_lt(a.as_u64x8(), b.as_u64x8()))
}

/// Compare packed unsigned 64-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmplt_epu64_mask&expand=1063)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epu64_mask(k1: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmplt_epu64_mask(a, b) & k1
}

/// Compare packed unsigned 64-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmple_epu64_mask&expand=1001)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmple_epu64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_le(a.as_u64x8(), b.as_u64x8()))
}

/// Compare packed unsigned 64-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmple_epu64_mask&expand=1002)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmple_epu64_mask(k1: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmple_epu64_mask(a, b) & k1
}

/// Compare packed unsigned 64-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpge_epu64_mask&expand=879)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpge_epu64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_ge(a.as_u64x8(), b.as_u64x8()))
}

/// Compare packed unsigned 64-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpge_epu64_mask&expand=880)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpge_epu64_mask(k1: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpge_epu64_mask(b, a) & k1
}

/// Compare packed unsigned 64-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpeq_epu64_mask&expand=813)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpeq_epu64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_eq(a.as_u64x8(), b.as_u64x8()))
}

/// Compare packed unsigned 64-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpeq_epu64_mask&expand=814)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpeq_epu64_mask(k1: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpeq_epu64_mask(a, b) & k1
}

/// Compare packed unsigned 64-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpneq_epu64_mask&expand=1118)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpneq_epu64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_ne(a.as_u64x8(), b.as_u64x8()))
}

/// Compare packed unsigned 64-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpneq_epu64_mask&expand=1119)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpneq_epu64_mask(k1: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpneq_epu64_mask(a, b) & k1
}

/// Compare packed unsigned 64-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmp_epu64_mask&expand=727)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_cmp_epu64_mask(a: __m512i, b: __m512i, imm8: _MM_CMPINT_ENUM) -> __mmask8 {
    let neg_one = -1;
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpuq(a.as_i64x8(), b.as_i64x8(), $imm3, neg_one)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed unsigned 64-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmp_epu64_mask&expand=728)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_mask_cmp_epu64_mask(
    k1: __mmask8,
    a: __m512i,
    b: __m512i,
    imm8: _MM_CMPINT_ENUM,
) -> __mmask8 {
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpuq(a.as_i64x8(), b.as_i64x8(), $imm3, k1 as i8)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 64-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmplt_epi64_mask&expand=1037)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epi64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_lt(a.as_i64x8(), b.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmplt_epi64_mask&expand=1038)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epi64_mask(k1: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmplt_epi64_mask(a, b) & k1
}

/// Compare packed signed 64-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmple_epi64_mask&expand=977)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmple_epi64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_le(a.as_i64x8(), b.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmple_epi64_mask&expand=978)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmple_epi64_mask(k1: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmple_epi64_mask(a, b) & k1
}

/// Compare packed signed 64-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpge_epi64_mask&expand=855)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpge_epi64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_ge(a.as_i64x8(), b.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpge_epi64_mask&expand=856)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpge_epi64_mask(k1: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpge_epi64_mask(b, a) & k1
}

/// Compare packed 64-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpeq_epi64_mask&expand=787)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpeq_epi64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_eq(a.as_i64x8(), b.as_i64x8()))
}

/// Compare packed 64-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpeq_epi64_mask&expand=788)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpeq_epi64_mask(k1: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpeq_epi64_mask(a, b) & k1
}

/// Compare packed signed 64-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpneq_epi64_mask&expand=1094)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpneq_epi64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_ne(a.as_i64x8(), b.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpneq_epi64_mask&expand=1095)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpneq_epi64_mask(k1: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpneq_epi64_mask(a, b) & k1
}

/// Compare packed signed 64-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmp_epi64_mask&expand=703)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_cmp_epi64_mask(a: __m512i, b: __m512i, imm8: _MM_CMPINT_ENUM) -> __mmask8 {
    let neg_one = -1;
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpq(a.as_i64x8(), b.as_i64x8(), $imm3, neg_one)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 64-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmp_epi64_mask&expand=704)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_mask_cmp_epi64_mask(
    k1: __mmask8,
    a: __m512i,
    b: __m512i,
    imm8: _MM_CMPINT_ENUM,
) -> __mmask8 {
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpq(a.as_i64x8(), b.as_i64x8(), $imm3, k1 as i8)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}*/
/*
/// Load 512-bits (composed of 16 packed 32-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_loadu_epi32&expand=3377)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu32
pub unsafe fn _mm512_loadu_epi32(mem_addr: *const i32) -> __m512i {
    ptr::read_unaligned(mem_addr as *const __m512i)
}

/// Store 512-bits (composed of 16 packed 32-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_storeu_epi32&expand=5628)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu32
pub unsafe fn _mm512_storeu_epi32(mem_addr: *mut i32, a: __m512i) {
    ptr::write_unaligned(mem_addr as *mut __m512i, a);
}

/// Load 512-bits (composed of 8 packed 64-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_loadu_epi64&expand=3386)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu64
pub unsafe fn _mm512_loadu_epi64(mem_addr: *const i64) -> __m512i {
    ptr::read_unaligned(mem_addr as *const __m512i)
}

/// Store 512-bits (composed of 8 packed 64-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_storeu_epi64&expand=5634)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu64
pub unsafe fn _mm512_storeu_epi64(mem_addr: *mut i64, a: __m512i) {
    ptr::write_unaligned(mem_addr as *mut __m512i, a);
}

/// Load 512-bits of integer data from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_loadu_si512&expand=3420)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu32
pub unsafe fn _mm512_loadu_si512(mem_addr: *const i32) -> __m512i {
    ptr::read_unaligned(mem_addr as *const __m512i)
}

/// Store 512-bits of integer data from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_storeu_si512&expand=5657)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu32
pub unsafe fn _mm512_storeu_si512(mem_addr: *mut i32, a: __m512i) {
    ptr::write_unaligned(mem_addr as *mut __m512i, a);
}
*/

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.pmul.dq.512"]
    fn vpmuldq(a: i32x16, b: i32x16) -> i64x8;
    #[link_name = "llvm.x86.avx512.pmulu.dq.512"]
    fn vpmuludq(a: u32x16, b: u32x16) -> u64x8;

    #[link_name = "llvm.x86.avx512.mask.pmaxs.d.512"]
    fn vpmaxsd(a: i32x16, b: i32x16) -> i32x16;
    #[link_name = "llvm.x86.avx512.mask.pmaxs.q.512"]
    fn vpmaxsq(a: i64x8, b: i64x8) -> i64x8;
    #[link_name = "llvm.x86.avx512.mask.pmins.d.512"]
    fn vpminsd(a: i32x16, b: i32x16) -> i32x16;
    #[link_name = "llvm.x86.avx512.mask.pmins.q.512"]
    fn vpminsq(a: i64x8, b: i64x8) -> i64x8;

    #[link_name = "llvm.x86.avx512.mask.pmaxu.d.512"]
    fn vpmaxud(a: u32x16, b: u32x16) -> u32x16;
    #[link_name = "llvm.x86.avx512.mask.pmaxu.q.512"]
    fn vpmaxuq(a: u64x8, b: u64x8) -> i64x8;
    #[link_name = "llvm.x86.avx512.mask.pminu.d.512"]
    fn vpminud(a: u32x16, b: u32x16) -> u32x16;
    #[link_name = "llvm.x86.avx512.mask.pminu.q.512"]
    fn vpminuq(a: u64x8, b: u64x8) -> i64x8;

    #[link_name = "llvm.x86.avx512.mask.cvtps2dq.512"]
    fn vcvtps2dq(a: f32x16, src: i32x16, mask: u16, rounding: i32) -> i32x16;
    #[link_name = "llvm.x86.avx512.mask.cvtps2udq.512"]
    fn vcvtps2udq(a: f32x16, src: u32x16, mask: u16, rounding: i32) -> u32x16;
    #[link_name = "llvm.x86.avx512.mask.cvtpd2dq.512"]
    fn vcvtpd2dq(a: f64x8, src: i32x8, mask: u8, rounding: i32) -> i32x8;
    #[link_name = "llvm.x86.avx512.mask.cvtpd2udq.512"]
    fn vcvtpd2udq(a: f64x8, src: u32x8, mask: u8, rounding: i32) -> u32x8;

    #[link_name = "llvm.x86.avx512.mask.cvttps2dq.512"]
    fn vcvttps2dq(a: f32x16, src: i32x16, mask: u16, rounding: i32) -> i32x16;
    #[link_name = "llvm.x86.avx512.mask.cvttps2udq.512"]
    fn vcvttps2udq(a: f32x16, src: i32x16, mask: u16, rounding: i32) -> u32x16;
    #[link_name = "llvm.x86.avx512.mask.cvttpd2dq.512"]
    fn vcvttpd2dq(a: f64x8, src: i32x8, mask: u8, rounding: i32) -> i32x8;
    #[link_name = "llvm.x86.avx512.mask.cvttpd2udq.512"]
    fn vcvttpd2udq(a: f64x8, src: i32x8, mask: u8, rounding: i32) -> u32x8;

    #[link_name = "llvm.x86.avx512.mask.pmov.qb.512"]
    fn vpmovqb(a: i64x8, src: i8x16, mask: u8) -> i8x16;
    #[link_name = "llvm.x86.avx512.mask.pmovs.dw.512"]
    fn vpmovsdw(a: i32x16, src: i16x16, mask: u16) -> i16x16;
    #[link_name = "llvm.x86.avx512.mask.pmovs.db.512"]
    fn vpmovsdb(a: i32x16, src: i8x16, mask: u16) -> i8x16;
    #[link_name = "llvm.x86.avx512.mask.pmovs.qd.512"]
    fn vpmovsqd(a: i64x8, src: i32x8, mask: u8) -> i32x8;
    #[link_name = "llvm.x86.avx512.mask.pmovs.qw.512"]
    fn vpmovsqw(a: i64x8, src: i16x8, mask: u8) -> i16x8;
    #[link_name = "llvm.x86.avx512.mask.pmovs.qb.512"]
    fn vpmovsqb(a: i64x8, src: i8x16, mask: u8) -> i8x16;
    #[link_name = "llvm.x86.avx512.mask.pmovus.dw.512"]
    fn vpmovusdw(a: u32x16, src: u16x16, mask: u16) -> u16x16;
    #[link_name = "llvm.x86.avx512.mask.pmovus.db.512"]
    fn vpmovusdb(a: u32x16, src: u8x16, mask: u16) -> u8x16;
    #[link_name = "llvm.x86.avx512.mask.pmovus.qd.512"]
    fn vpmovusqd(a: u64x8, src: u32x8, mask: u8) -> u32x8;
    #[link_name = "llvm.x86.avx512.mask.pmovus.qw.512"]
    fn vpmovusqw(a: u64x8, src: u16x8, mask: u8) -> u16x8;
    #[link_name = "llvm.x86.avx512.mask.pmovus.qb.512"]
    fn vpmovusqb(a: u64x8, src: u8x16, mask: u8) -> u8x16;

    #[link_name = "llvm.x86.avx512.mask.cmp.ss"]
    fn vcmpss(a: __m128, b: __m128, op: i32, m: i8, sae: i32) -> i8;
    #[link_name = "llvm.x86.avx512.mask.cmp.sd"]
    fn vcmpsd(a: __m128d, b: __m128d, op: i32, m: i8, sae: i32) -> i8;
    #[link_name = "llvm.x86.avx512.mask.cmp.ps.512"]
    fn vcmpps(a: f32x16, b: f32x16, op: i32, m: i16, sae: i32) -> i16;
    #[link_name = "llvm.x86.avx512.mask.cmp.pd.512"]
    fn vcmppd(a: f64x8, b: f64x8, op: i32, m: i8, sae: i32) -> i8;
    #[link_name = "llvm.x86.avx512.mask.ucmp.q.512"]
    fn vpcmpuq(a: i64x8, b: i64x8, op: i32, m: i8) -> i8;
    #[link_name = "llvm.x86.avx512.mask.cmp.q.512"]
    fn vpcmpq(a: i64x8, b: i64x8, op: i32, m: i8) -> i8;
    #[link_name = "llvm.x86.avx512.mask.ucmp.d.512"]
    fn vpcmpud(a: i32x16, b: i32x16, op: i32, m: i16) -> i16;
    #[link_name = "llvm.x86.avx512.mask.cmp.d.512"]
    fn vpcmpd(a: i32x16, b: i32x16, op: i32, m: i16) -> i16;

    #[link_name = "llvm.x86.avx512.mask.prol.d.512"]
    fn vprold(a: i32x16, i8: i32) -> i32x16;
    #[link_name = "llvm.x86.avx512.mask.pror.d.512"]
    fn vprord(a: i32x16, i8: i32) -> i32x16;
    #[link_name = "llvm.x86.avx512.mask.prol.q.512"]
    fn vprolq(a: i64x8, i8: i32) -> i64x8;
    #[link_name = "llvm.x86.avx512.mask.pror.q.512"]
    fn vprorq(a: i64x8, i8: i32) -> i64x8;

    #[link_name = "llvm.x86.avx512.mask.prolv.d.512"]
    fn vprolvd(a: i32x16, b: i32x16) -> i32x16;
    #[link_name = "llvm.x86.avx512.mask.prorv.d.512"]
    fn vprorvd(a: i32x16, b: i32x16) -> i32x16;
    #[link_name = "llvm.x86.avx512.mask.prolv.q.512"]
    fn vprolvq(a: i64x8, b: i64x8) -> i64x8;
    #[link_name = "llvm.x86.avx512.mask.prorv.q.512"]
    fn vprorvq(a: i64x8, b: i64x8) -> i64x8;

    #[link_name = "llvm.x86.avx512.psllv.d.512"]
    fn vpsllvd(a: i32x16, b: i32x16) -> i32x16;
    #[link_name = "llvm.x86.avx512.psrlv.d.512"]
    fn vpsrlvd(a: i32x16, b: i32x16) -> i32x16;
    #[link_name = "llvm.x86.avx512.psllv.q.512"]
    fn vpsllvq(a: i64x8, b: i64x8) -> i64x8;
    #[link_name = "llvm.x86.avx512.psrlv.q.512"]
    fn vpsrlvq(a: i64x8, b: i64x8) -> i64x8;

    #[link_name = "llvm.x86.avx512.pslli.d.512"]
    fn vpsllid(a: i32x16, imm8: u32) -> i32x16;
    #[link_name = "llvm.x86.avx512.psrli.d.512"]
    fn vpsrlid(a: i32x16, imm8: u32) -> i32x16;
    #[link_name = "llvm.x86.avx512.pslli.q.512"]
    fn vpslliq(a: i64x8, imm8: u32) -> i64x8;
    #[link_name = "llvm.x86.avx512.psrli.q.512"]
    fn vpsrliq(a: i64x8, imm8: u32) -> i64x8;

    #[link_name = "llvm.x86.avx512.psll.d.512"]
    fn vpslld(a: i32x16, count: i32x4) -> i32x16;
    #[link_name = "llvm.x86.avx512.psrl.d.512"]
    fn vpsrld(a: i32x16, count: i32x4) -> i32x16;
    #[link_name = "llvm.x86.avx512.psll.q.512"]
    fn vpsllq(a: i64x8, count: i64x2) -> i64x8;
    #[link_name = "llvm.x86.avx512.psrl.q.512"]
    fn vpsrlq(a: i64x8, count: i64x2) -> i64x8;

    #[link_name = "llvm.x86.avx512.psra.d.512"]
    fn vpsrad(a: i32x16, count: i32x4) -> i32x16;
    #[link_name = "llvm.x86.avx512.psra.q.512"]
    fn vpsraq(a: i64x8, count: i64x2) -> i64x8;

    #[link_name = "llvm.x86.avx512.psrai.d.512"]
    fn vpsraid(a: i32x16, imm8: u32) -> i32x16;
    #[link_name = "llvm.x86.avx512.psrai.q.512"]
    fn vpsraiq(a: i64x8, imm8: u32) -> i64x8;

    #[link_name = "llvm.x86.avx512.psrav.d.512"]
    fn vpsravd(a: i32x16, count: i32x16) -> i32x16;
    #[link_name = "llvm.x86.avx512.psrav.q.512"]
    fn vpsravq(a: i64x8, count: i64x8) -> i64x8;

    #[link_name = "llvm.x86.avx512.permvar.si.512"]
    fn vpermd(a: i32x16, idx: i32x16) -> i32x16;
    #[link_name = "llvm.x86.avx512.permvar.di.512"]
    fn vpermq(a: i64x8, idx: i64x8) -> i64x8;

    #[link_name = "llvm.x86.avx512.vpermi2var.d.512"]
    fn vpermi2d(a: i32x16, idx: i32x16, b: i32x16) -> i32x16;
    #[link_name = "llvm.x86.avx512.vpermi2var.q.512"]
    fn vpermi2q(a: i64x8, idx: i64x8, b: i64x8) -> i64x8;

    #[link_name = "llvm.x86.avx512.vcvtss2si32"]
    fn vcvtss2si(a: f32x4, rounding: i32) -> i32;
    #[link_name = "llvm.x86.avx512.vcvtss2si64"]
    fn vcvtss2si64(a: f32x4, rounding: i32) -> i64;
    #[link_name = "llvm.x86.avx512.vcvtss2usi32"]
    fn vcvtss2usi(a: f32x4, rounding: i32) -> u32;
    #[link_name = "llvm.x86.avx512.vcvtss2usi64"]
    fn vcvtss2usi64(a: f32x4, rounding: i32) -> u64;
    #[link_name = "llvm.x86.avx512.vcvtsd2si32"]
    fn vcvtsd2si(a: f64x2, rounding: i32) -> i32;
    #[link_name = "llvm.x86.avx512.vcvtsd2si64"]
    fn vcvtsd2si64(a: f64x2, rounding: i32) -> i64;
    #[link_name = "llvm.x86.avx512.vcvtsd2usi32"]
    fn vcvtsd2usi(a: f64x2, rounding: i32) -> u32;
    #[link_name = "llvm.x86.avx512.vcvtsd2usi64"]
    fn vcvtsd2usi64(a: f64x2, rounding: i32) -> u64;
}

#[cfg(test)]
mod tests {

    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;
    use crate::hint::black_box;
    use crate::mem::{self};
/*
    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_abs_epi32() {
        #[rustfmt::skip]
        let a = _mm512_setr_epi32(
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
        );
        let r = _mm512_abs_epi32(a);
        let e = _mm512_setr_epi32(
            0,
            1,
            1,
            i32::MAX,
            i32::MAX.wrapping_add(1),
            100,
            100,
            32,
            0,
            1,
            1,
            i32::MAX,
            i32::MAX.wrapping_add(1),
            100,
            100,
            32,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_abs_epi32() {
        #[rustfmt::skip]
        let a = _mm512_setr_epi32(
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
        );
        let r = _mm512_mask_abs_epi32(a, 0, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_abs_epi32(a, 0b00000000_11111111, a);
        let e = _mm512_setr_epi32(
            0,
            1,
            1,
            i32::MAX,
            i32::MAX.wrapping_add(1),
            100,
            100,
            32,
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_abs_epi32() {
        #[rustfmt::skip]
        let a = _mm512_setr_epi32(
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
        );
        let r = _mm512_maskz_abs_epi32(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_abs_epi32(0b00000000_11111111, a);
        let e = _mm512_setr_epi32(
            0,
            1,
            1,
            i32::MAX,
            i32::MAX.wrapping_add(1),
            100,
            100,
            32,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_mov_epi32() {
        let src = _mm512_set1_epi32(1);
        let a = _mm512_set1_epi32(2);
        let r = _mm512_mask_mov_epi32(src, 0, a);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_mov_epi32(src, 0b11111111_11111111, a);
        assert_eq_m512i(r, a);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_mov_epi32() {
        let a = _mm512_set1_epi32(2);
        let r = _mm512_maskz_mov_epi32(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_mov_epi32(0b11111111_11111111, a);
        assert_eq_m512i(r, a);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_add_epi32() {
        let a = _mm512_setr_epi32(
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
        );
        let b = _mm512_set1_epi32(1);
        let r = _mm512_add_epi32(a, b);
        let e = _mm512_setr_epi32(
            1,
            2,
            0,
            i32::MIN,
            i32::MIN + 1,
            101,
            -99,
            -31,
            1,
            2,
            0,
            i32::MIN,
            i32::MIN + 1,
            101,
            -99,
            -31,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_add_epi32() {
        #[rustfmt::skip]
        let a = _mm512_setr_epi32(
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
        );
        let b = _mm512_set1_epi32(1);
        let r = _mm512_mask_add_epi32(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_add_epi32(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(
            1,
            2,
            0,
            i32::MIN,
            i32::MIN + 1,
            101,
            -99,
            -31,
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_add_epi32() {
        #[rustfmt::skip]
        let a = _mm512_setr_epi32(
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
            0, 1, -1, i32::MAX,
            i32::MIN, 100, -100, -32,
        );
        let b = _mm512_set1_epi32(1);
        let r = _mm512_maskz_add_epi32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_add_epi32(0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(
            1,
            2,
            0,
            i32::MIN,
            i32::MIN + 1,
            101,
            -99,
            -31,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_sub_epi32() {
        let a = _mm512_setr_epi32(
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
        );
        let b = _mm512_set1_epi32(1);
        let r = _mm512_sub_epi32(a, b);
        let e = _mm512_setr_epi32(
            -1,
            0,
            -2,
            i32::MAX - 1,
            i32::MAX,
            99,
            -101,
            -33,
            -1,
            0,
            -2,
            i32::MAX - 1,
            i32::MAX,
            99,
            -101,
            -33,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_sub_epi32() {
        let a = _mm512_setr_epi32(
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
        );
        let b = _mm512_set1_epi32(1);
        let r = _mm512_mask_sub_epi32(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_sub_epi32(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(
            -1,
            0,
            -2,
            i32::MAX - 1,
            i32::MAX,
            99,
            -101,
            -33,
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_sub_epi32() {
        let a = _mm512_setr_epi32(
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
        );
        let b = _mm512_set1_epi32(1);
        let r = _mm512_maskz_sub_epi32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_sub_epi32(0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(
            -1,
            0,
            -2,
            i32::MAX - 1,
            i32::MAX,
            99,
            -101,
            -33,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mullo_epi32() {
        let a = _mm512_setr_epi32(
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
        );
        let b = _mm512_set1_epi32(2);
        let r = _mm512_mullo_epi32(a, b);
        let e = _mm512_setr_epi32(
            0, 2, -2, -2, 0, 200, -200, -64, 0, 2, -2, -2, 0, 200, -200, -64,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_mullo_epi32() {
        let a = _mm512_setr_epi32(
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
        );
        let b = _mm512_set1_epi32(2);
        let r = _mm512_mask_mullo_epi32(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_mullo_epi32(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(
            0,
            2,
            -2,
            -2,
            0,
            200,
            -200,
            -64,
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_mullo_epi32() {
        let a = _mm512_setr_epi32(
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
            0,
            1,
            -1,
            i32::MAX,
            i32::MIN,
            100,
            -100,
            -32,
        );
        let b = _mm512_set1_epi32(2);
        let r = _mm512_maskz_mullo_epi32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_mullo_epi32(0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(0, 2, -2, -2, 0, 200, -200, -64, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_max_epi32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_max_epi32(a, b);
        let e = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_max_epi32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_max_epi32(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_max_epi32(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_max_epi32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_max_epi32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_max_epi32(0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_max_epu32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_max_epu32(a, b);
        let e = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_max_epu32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_max_epu32(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_max_epu32(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_max_epu32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_max_epu32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_max_epu32(0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_min_epi32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_min_epi32(a, b);
        let e = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_min_epi32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_min_epi32(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_min_epi32(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_min_epi32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_min_epi32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_min_epi32(0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_min_epu32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_min_epu32(a, b);
        let e = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_min_epu32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_min_epu32(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_min_epu32(a, 0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_min_epu32() {
        let a = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_min_epu32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_min_epu32(0b00000000_11111111, a, b);
        let e = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cvtepi8_epi32() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm512_cvtepi8_epi32(a);
        let e = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cvtepi8_epi32() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let src = _mm512_set1_epi32(-1);
        let r = _mm512_mask_cvtepi8_epi32(src, 0, a);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_cvtepi8_epi32(src, 0b00000000_11111111, a);
        let e = _mm512_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_cvtepi8_epi32() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm512_maskz_cvtepi8_epi32(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_cvtepi8_epi32(0b00000000_11111111, a);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cvtepu8_epi32() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm512_cvtepu8_epi32(a);
        let e = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cvtepu8_epi32() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let src = _mm512_set1_epi32(-1);
        let r = _mm512_mask_cvtepu8_epi32(src, 0, a);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_cvtepu8_epi32(src, 0b00000000_11111111, a);
        let e = _mm512_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_cvtepu8_epi32() {
        let a = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm512_maskz_cvtepu8_epi32(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_cvtepu8_epi32(0b00000000_11111111, a);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cvtepi16_epi32() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm512_cvtepi16_epi32(a);
        let e = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cvtepi16_epi32() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let src = _mm512_set1_epi32(-1);
        let r = _mm512_mask_cvtepi16_epi32(src, 0, a);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_cvtepi16_epi32(src, 0b00000000_11111111, a);
        let e = _mm512_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_cvtepi16_epi32() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm512_maskz_cvtepi16_epi32(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_cvtepi16_epi32(0b00000000_11111111, a);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cvtepu16_epi32() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm512_cvtepu16_epi32(a);
        let e = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cvtepu16_epi32() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let src = _mm512_set1_epi32(-1);
        let r = _mm512_mask_cvtepu16_epi32(src, 0, a);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_cvtepu16_epi32(src, 0b00000000_11111111, a);
        let e = _mm512_set_epi32(-1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_cvtepu16_epi32() {
        let a = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm512_maskz_cvtepu16_epi32(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_cvtepu16_epi32(0b00000000_11111111, a);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cvtepi32_epi16() {
        let a = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm512_cvtepi32_epi16(a);
        let e = _mm256_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cvtepi32_epi16() {
        let a = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let src = _mm256_set1_epi16(-1);
        let r = _mm512_mask_cvtepi32_epi16(src, 0, a);
        assert_eq_m256i(r, src);
        let r = _mm512_mask_cvtepi32_epi16(src, 0b00000000_11111111, a);
        let e = _mm256_set_epi16(-1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_cvtepi32_epi16() {
        let a = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm512_maskz_cvtepi32_epi16(0, a);
        assert_eq_m256i(r, _mm256_setzero_si256());
        let r = _mm512_maskz_cvtepi32_epi16(0b00000000_11111111, a);
        let e = _mm256_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m256i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cvtepi32_epi8() {
        let a = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm512_cvtepi32_epi8(a);
        let e = _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cvtepi32_epi8() {
        let a = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let src = _mm_set1_epi8(-1);
        let r = _mm512_mask_cvtepi32_epi8(src, 0, a);
        assert_eq_m128i(r, src);
        let r = _mm512_mask_cvtepi32_epi8(src, 0b00000000_11111111, a);
        let e = _mm_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m128i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_cvtepi32_epi8() {
        let a = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm512_maskz_cvtepi32_epi8(0, a);
        assert_eq_m128i(r, _mm_setzero_si128());
        let r = _mm512_maskz_cvtepi32_epi8(0b00000000_11111111, a);
        let e = _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m128i(r, e);
    }
*/
    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmplt_epu16_mask() {
        let a = _mm512_set1_epi16(-2);
        let b = _mm512_set1_epi16(-1);
        let m = _mm512_cmplt_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmplt_epu16_mask() {
        let a = _mm512_set1_epi16(-2);
        let b = _mm512_set1_epi16(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmplt_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmplt_epu8_mask() {
        let a = _mm512_set1_epi8(-2);
        let b = _mm512_set1_epi8(-1);
        let m = _mm512_cmplt_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmplt_epu8_mask() {
        let a = _mm512_set1_epi8(-2);
        let b = _mm512_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmplt_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmplt_epi16_mask() {
        let a = _mm512_set1_epi16(-2);
        let b = _mm512_set1_epi16(-1);
        let m = _mm512_cmplt_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmplt_epi16_mask() {
        let a = _mm512_set1_epi16(-2);
        let b = _mm512_set1_epi16(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmplt_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmplt_epi8_mask() {
        let a = _mm512_set1_epi8(-2);
        let b = _mm512_set1_epi8(-1);
        let m = _mm512_cmplt_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmplt_epi8_mask() {
        let a = _mm512_set1_epi8(-2);
        let b = _mm512_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmplt_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpgt_epu16_mask() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(1);
        let m = _mm512_cmpgt_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpgt_epu16_mask() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpgt_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpgt_epu8_mask() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(1);
        let m = _mm512_cmpgt_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpgt_epu8_mask() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpgt_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpgt_epi16_mask() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(-1);
        let m = _mm512_cmpgt_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpgt_epi16_mask() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpgt_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpgt_epi8_mask() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(-1);
        let m = _mm512_cmpgt_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpgt_epi8_mask() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpgt_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101);
    }
/*
    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmple_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        assert_eq!(
            _mm512_cmple_epu32_mask(a, b),
            !_mm512_cmpgt_epu32_mask(a, b)
        )
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmple_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let mask = 0b01111010_01111010;
        assert_eq!(
            _mm512_mask_cmple_epu32_mask(mask, a, b),
            0b01111010_01111010
        );
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpge_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        assert_eq!(
            _mm512_cmpge_epu32_mask(a, b),
            !_mm512_cmplt_epu32_mask(a, b)
        )
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpge_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let mask = 0b01111010_01111010;
        assert_eq!(_mm512_mask_cmpge_epu32_mask(mask, a, b), 0b01100000_0110000);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpeq_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        #[rustfmt::skip]
        let b = _mm512_set_epi32(0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100);
        let m = _mm512_cmpeq_epu32_mask(b, a);
        assert_eq!(m, 0b11001111_11001111);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpeq_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        #[rustfmt::skip]
        let b = _mm512_set_epi32(0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100);
        let mask = 0b01111010_01111010;
        let r = _mm512_mask_cmpeq_epu32_mask(mask, b, a);
        assert_eq!(r, 0b01001010_01001010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpneq_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        #[rustfmt::skip]
        let b = _mm512_set_epi32(0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100);
        let m = _mm512_cmpneq_epu32_mask(b, a);
        assert_eq!(m, !_mm512_cmpeq_epu32_mask(b, a));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpneq_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, -100, 100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, -100, 100);
        #[rustfmt::skip]
        let b = _mm512_set_epi32(0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100);
        let mask = 0b01111010_01111010;
        let r = _mm512_mask_cmpneq_epu32_mask(mask, b, a);
        assert_eq!(r, 0b00110010_00110010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmp_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let m = _mm512_cmp_epu32_mask(a, b, _MM_CMPINT_LT);
        assert_eq!(m, 0b11001111_11001111);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmp_epu32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let mask = 0b01111010_01111010;
        let r = _mm512_mask_cmp_epu32_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(r, 0b01001010_01001010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmple_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        assert_eq!(
            _mm512_cmple_epi32_mask(a, b),
            !_mm512_cmpgt_epi32_mask(a, b)
        )
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmple_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let mask = 0b01111010_01111010;
        assert_eq!(_mm512_mask_cmple_epi32_mask(mask, a, b), 0b01100000_0110000);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpge_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        assert_eq!(
            _mm512_cmpge_epi32_mask(a, b),
            !_mm512_cmplt_epi32_mask(a, b)
        )
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpge_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, u32::MAX as i32, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let mask = 0b01111010_01111010;
        assert_eq!(
            _mm512_mask_cmpge_epi32_mask(mask, a, b),
            0b01111010_01111010
        );
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpeq_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100);
        #[rustfmt::skip]
        let b = _mm512_set_epi32(0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100);
        let m = _mm512_cmpeq_epi32_mask(b, a);
        assert_eq!(m, 0b11001111_11001111);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpeq_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100);
        #[rustfmt::skip]
        let b = _mm512_set_epi32(0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100);
        let mask = 0b01111010_01111010;
        let r = _mm512_mask_cmpeq_epi32_mask(mask, b, a);
        assert_eq!(r, 0b01001010_01001010);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmpneq_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100);
        #[rustfmt::skip]
        let b = _mm512_set_epi32(0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100);
        let m = _mm512_cmpneq_epi32_mask(b, a);
        assert_eq!(m, !_mm512_cmpeq_epi32_mask(b, a));
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmpneq_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, 13, i32::MAX, i32::MIN, -100, 100,
                                 0, 1, -1, 13, i32::MAX, i32::MIN, -100, 100);
        #[rustfmt::skip]
        let b = _mm512_set_epi32(0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, 13, 42, i32::MAX, i32::MIN, 100, -100);
        let mask = 0b01111010_01111010;
        let r = _mm512_mask_cmpneq_epi32_mask(mask, b, a);
        assert_eq!(r, 0b00110010_00110010)
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_cmp_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let m = _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_LT);
        assert_eq!(m, 0b00000101_00000101);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_cmp_epi32_mask() {
        #[rustfmt::skip]
        let a = _mm512_set_epi32(0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100,
                                 0, 1, -1, 13, i32::MAX, i32::MIN, 100, -100);
        let b = _mm512_set1_epi32(-1);
        let mask = 0b01100110_01100110;
        let r = _mm512_mask_cmp_epi32_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(r, 0b00000100_00000100);
    }*/
/*
    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_slli_epi32() {
        let a = _mm512_set_epi32(1 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let r = _mm512_slli_epi32(a, 1);
        let e = _mm512_set_epi32(0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_slli_epi32() {
        let a = _mm512_set_epi32(1 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let r = _mm512_mask_slli_epi32(a, 0, a, 1);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_slli_epi32(a, 0b11111111_11111111, a, 1);
        let e = _mm512_set_epi32(0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_slli_epi32() {
        let a = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 << 31);
        let r = _mm512_maskz_slli_epi32(0, a, 1);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_slli_epi32(0b00000000_11111111, a, 1);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_srli_epi32() {
        let a = _mm512_set_epi32(0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let r = _mm512_srli_epi32(a, 1);
        let e = _mm512_set_epi32(0 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_srli_epi32() {
        let a = _mm512_set_epi32(0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let r = _mm512_mask_srli_epi32(a, 0, a, 1);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_srli_epi32(a, 0b11111111_11111111, a, 1);
        let e = _mm512_set_epi32(0 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_srli_epi32() {
        let a = _mm512_set_epi32(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0);
        let r = _mm512_maskz_srli_epi32(0, a, 1);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_srli_epi32(0b00000000_11111111, a, 1);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0 << 31);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_sllv_epi32() {
        let a = _mm512_set_epi32(1 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let count = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let r = _mm512_sllv_epi32(a, count);

        let e = _mm512_set_epi32(0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_sllv_epi32() {
        let a = _mm512_set_epi32(1 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let count = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let r = _mm512_mask_sllv_epi32(a, 0, a, count);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_sllv_epi32(a, 0b11111111_11111111, a, count);

        let e = _mm512_set_epi32(0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_sllv_epi32() {
        let a = _mm512_set_epi32(1 << 31, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 << 31);
        let count = _mm512_set_epi32(0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let r = _mm512_maskz_sllv_epi32(0, a, count);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_sllv_epi32(0b00000000_11111111, a, count);

        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_srlv_epi32() {
        let a = _mm512_set_epi32(0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let count = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let r = _mm512_srlv_epi32(a, count);

        let e = _mm512_set_epi32(0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_srlv_epi32() {
        let a = _mm512_set_epi32(0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let count = _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let r = _mm512_mask_srlv_epi32(a, 0, a, count);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_srlv_epi32(a, 0b11111111_11111111, a, count);

        let e = _mm512_set_epi32(0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_srlv_epi32() {
        let a = _mm512_set_epi32(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0);
        let count = _mm512_set_epi32(0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);

        let r = _mm512_maskz_srlv_epi32(0, a, count);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_srlv_epi32(0b00000000_11111111, a, count);

        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_sll_epi32() {
        let a = _mm512_set_epi32(
            1 << 31,
            1 << 0,
            1 << 1,
            1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let count = _mm_set_epi32(0, 0, 0, 2);
        let r = _mm512_sll_epi32(a, count);
        let e = _mm512_set_epi32(
            0,
            1 << 2,
            1 << 3,
            1 << 4,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_sll_epi32() {
        let a = _mm512_set_epi32(
            1 << 31,
            1 << 0,
            1 << 1,
            1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let count = _mm_set_epi32(0, 0, 0, 2);
        let r = _mm512_mask_sll_epi32(a, 0, a, count);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_sll_epi32(a, 0b11111111_11111111, a, count);
        let e = _mm512_set_epi32(
            0,
            1 << 2,
            1 << 3,
            1 << 4,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_sll_epi32() {
        let a = _mm512_set_epi32(
            1 << 31,
            1 << 0,
            1 << 1,
            1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 31,
        );
        let count = _mm_set_epi32(2, 0, 0, 2);
        let r = _mm512_maskz_sll_epi32(0, a, count);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_sll_epi32(0b00000000_11111111, a, count);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_srl_epi32() {
        let a = _mm512_set_epi32(
            1 << 31,
            1 << 0,
            1 << 1,
            1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let count = _mm_set_epi32(0, 0, 0, 2);
        let r = _mm512_srl_epi32(a, count);
        let e = _mm512_set_epi32(1 << 29, 0, 0, 1 << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_srl_epi32() {
        let a = _mm512_set_epi32(
            1 << 31,
            1 << 0,
            1 << 1,
            1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        );
        let count = _mm_set_epi32(0, 0, 0, 2);
        let r = _mm512_mask_srl_epi32(a, 0, a, count);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_srl_epi32(a, 0b11111111_11111111, a, count);
        let e = _mm512_set_epi32(1 << 29, 0, 0, 1 << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_srl_epi32() {
        let a = _mm512_set_epi32(
            1 << 31,
            1 << 0,
            1 << 1,
            1 << 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            1 << 31,
        );
        let count = _mm_set_epi32(2, 0, 0, 2);
        let r = _mm512_maskz_srl_epi32(0, a, count);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_srl_epi32(0b00000000_11111111, a, count);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 << 29);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_sra_epi32() {
        let a = _mm512_set_epi32(8, -8, 16, -15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
        let count = _mm_set_epi32(1, 0, 0, 2);
        let r = _mm512_sra_epi32(a, count);
        let e = _mm512_set_epi32(2, -2, 4, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_sra_epi32() {
        let a = _mm512_set_epi32(8, -8, 16, -15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16);
        let count = _mm_set_epi32(0, 0, 0, 2);
        let r = _mm512_mask_sra_epi32(a, 0, a, count);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_sra_epi32(a, 0b11111111_11111111, a, count);
        let e = _mm512_set_epi32(2, -2, 4, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_sra_epi32() {
        let a = _mm512_set_epi32(8, -8, 16, -15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -15, -14);
        let count = _mm_set_epi32(2, 0, 0, 2);
        let r = _mm512_maskz_sra_epi32(0, a, count);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_sra_epi32(0b00000000_11111111, a, count);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -4);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_srav_epi32() {
        let a = _mm512_set_epi32(8, -8, 16, -15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
        let count = _mm512_set_epi32(2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r = _mm512_srav_epi32(a, count);
        let e = _mm512_set_epi32(2, -2, 4, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_srav_epi32() {
        let a = _mm512_set_epi32(8, -8, 16, -15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16);
        let count = _mm512_set_epi32(2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1);
        let r = _mm512_mask_srav_epi32(a, 0, a, count);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_srav_epi32(a, 0b11111111_11111111, a, count);
        let e = _mm512_set_epi32(2, -2, 4, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_srav_epi32() {
        let a = _mm512_set_epi32(8, -8, 16, -15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -15, -14);
        let count = _mm512_set_epi32(2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2);
        let r = _mm512_maskz_srav_epi32(0, a, count);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_srav_epi32(0b00000000_11111111, a, count);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, -4);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_srai_epi32() {
        let a = _mm512_set_epi32(8, -8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, -15);
        let r = _mm512_srai_epi32(a, 2);
        let e = _mm512_set_epi32(2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, -4);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_srai_epi32() {
        let a = _mm512_set_epi32(8, -8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, -15);
        let r = _mm512_mask_srai_epi32(a, 0, a, 2);
        assert_eq_m512i(r, a);

        let r = _mm512_mask_srai_epi32(a, 0b11111111_11111111, a, 2);
        let e = _mm512_set_epi32(2, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -4);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_srai_epi32() {
        let a = _mm512_set_epi32(8, -8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 15, -15);
        let r = _mm512_maskz_srai_epi32(0, a, 2);
        assert_eq_m512i(r, _mm512_setzero_si512());

        let r = _mm512_maskz_srai_epi32(0b00000000_11111111, a, 2);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, -4);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_permutevar_epi32() {
        let idx = _mm512_set1_epi32(1);
        let a = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm512_permutevar_epi32(idx, a);
        let e = _mm512_set1_epi32(14);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_permutevar_epi32() {
        let idx = _mm512_set1_epi32(1);
        let a = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm512_mask_permutevar_epi32(a, 0, idx, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_permutevar_epi32(a, 0b11111111_11111111, idx, a);
        let e = _mm512_set1_epi32(14);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_permutexvar_epi32() {
        let idx = _mm512_set1_epi32(1);
        let a = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm512_permutexvar_epi32(idx, a);
        let e = _mm512_set1_epi32(14);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_permutexvar_epi32() {
        let idx = _mm512_set1_epi32(1);
        let a = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm512_mask_permutexvar_epi32(a, 0, idx, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_permutexvar_epi32(a, 0b11111111_11111111, idx, a);
        let e = _mm512_set1_epi32(14);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_permutexvar_epi32() {
        let idx = _mm512_set1_epi32(1);
        let a = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r = _mm512_maskz_permutexvar_epi32(0, idx, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_permutexvar_epi32(0b00000000_11111111, idx, a);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 14, 14, 14, 14, 14, 14, 14, 14);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_permutex2var_epi32() {
        let a = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let idx = _mm512_set_epi32(
            1,
            1 << 4,
            2,
            1 << 4,
            3,
            1 << 4,
            4,
            1 << 4,
            5,
            1 << 4,
            6,
            1 << 4,
            7,
            1 << 4,
            8,
            1 << 4,
        );
        let b = _mm512_set1_epi32(100);
        let r = _mm512_permutex2var_epi32(a, idx, b);
        let e = _mm512_set_epi32(
            14, 100, 13, 100, 12, 100, 11, 100, 10, 100, 9, 100, 8, 100, 7, 100,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_permutex2var_epi32() {
        let a = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let idx = _mm512_set_epi32(
            1,
            1 << 4,
            2,
            1 << 4,
            3,
            1 << 4,
            4,
            1 << 4,
            5,
            1 << 4,
            6,
            1 << 4,
            7,
            1 << 4,
            8,
            1 << 4,
        );
        let b = _mm512_set1_epi32(100);
        let r = _mm512_mask_permutex2var_epi32(a, 0, idx, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_permutex2var_epi32(a, 0b11111111_11111111, idx, b);
        let e = _mm512_set_epi32(
            14, 100, 13, 100, 12, 100, 11, 100, 10, 100, 9, 100, 8, 100, 7, 100,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_permutex2var_epi32() {
        let a = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let idx = _mm512_set_epi32(
            1,
            1 << 4,
            2,
            1 << 4,
            3,
            1 << 4,
            4,
            1 << 4,
            5,
            1 << 4,
            6,
            1 << 4,
            7,
            1 << 4,
            8,
            1 << 4,
        );
        let b = _mm512_set1_epi32(100);
        let r = _mm512_maskz_permutex2var_epi32(0, a, idx, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_permutex2var_epi32(0b00000000_11111111, a, idx, b);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 10, 100, 9, 100, 8, 100, 7, 100);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask2_permutex2var_epi32() {
        let a = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let idx = _mm512_set_epi32(
            1000,
            1 << 4,
            2000,
            1 << 4,
            3000,
            1 << 4,
            4000,
            1 << 4,
            5,
            1 << 4,
            6,
            1 << 4,
            7,
            1 << 4,
            8,
            1 << 4,
        );
        let b = _mm512_set1_epi32(100);
        let r = _mm512_mask2_permutex2var_epi32(a, idx, 0, b);
        assert_eq_m512i(r, idx);
        let r = _mm512_mask2_permutex2var_epi32(a, idx, 0b00000000_11111111, b);
        let e = _mm512_set_epi32(
            1000,
            1 << 4,
            2000,
            1 << 4,
            3000,
            1 << 4,
            4000,
            1 << 4,
            10,
            100,
            9,
            100,
            8,
            100,
            7,
            100,
        );
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_shuffle_epi32() {
        let a = _mm512_setr_epi32(1, 4, 5, 8, 9, 12, 13, 16, 1, 4, 5, 8, 9, 12, 13, 16);
        let r = _mm512_shuffle_epi32(a, _MM_PERM_AADD);
        let e = _mm512_setr_epi32(8, 8, 1, 1, 16, 16, 9, 9, 8, 8, 1, 1, 16, 16, 9, 9);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_shuffle_epi32() {
        let a = _mm512_setr_epi32(1, 4, 5, 8, 9, 12, 13, 16, 1, 4, 5, 8, 9, 12, 13, 16);
        let r = _mm512_mask_shuffle_epi32(a, 0, a, _MM_PERM_AADD);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_shuffle_epi32(a, 0b11111111_11111111, a, _MM_PERM_AADD);
        let e = _mm512_setr_epi32(8, 8, 1, 1, 16, 16, 9, 9, 8, 8, 1, 1, 16, 16, 9, 9);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_shuffle_epi32() {
        let a = _mm512_setr_epi32(1, 4, 5, 8, 9, 12, 13, 16, 1, 4, 5, 8, 9, 12, 13, 16);
        let r = _mm512_maskz_shuffle_epi32(0, a, _MM_PERM_AADD);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_shuffle_epi32(0b00000000_11111111, a, _MM_PERM_AADD);
        let e = _mm512_setr_epi32(8, 8, 1, 1, 16, 16, 9, 9, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_broadcastd_epi32() {
        let a = _mm_set_epi32(17, 18, 19, 20);
        let r = _mm512_broadcastd_epi32(a);
        let e = _mm512_set1_epi32(20);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_broadcastd_epi32() {
        let src = _mm512_set1_epi32(20);
        let a = _mm_set_epi32(17, 18, 19, 20);
        let r = _mm512_mask_broadcastd_epi32(src, 0, a);
        assert_eq_m512i(r, src);
        let r = _mm512_mask_broadcastd_epi32(src, 0b11111111_11111111, a);
        let e = _mm512_set1_epi32(20);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_broadcastd_epi32() {
        let a = _mm_set_epi32(17, 18, 19, 20);
        let r = _mm512_maskz_broadcastd_epi32(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_broadcastd_epi32(0b00000000_11111111, a);
        let e = _mm512_setr_epi32(20, 20, 20, 20, 20, 20, 20, 20, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_unpackhi_epi32() {
        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm512_set_epi32(
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = _mm512_unpackhi_epi32(a, b);
        let e = _mm512_set_epi32(17, 1, 18, 2, 21, 5, 22, 6, 25, 9, 26, 10, 29, 13, 30, 14);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_unpackhi_epi32() {
        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm512_set_epi32(
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = _mm512_mask_unpackhi_epi32(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_unpackhi_epi32(a, 0b11111111_11111111, a, b);
        let e = _mm512_set_epi32(17, 1, 18, 2, 21, 5, 22, 6, 25, 9, 26, 10, 29, 13, 30, 14);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_unpackhi_epi32() {
        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm512_set_epi32(
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = _mm512_maskz_unpackhi_epi32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_unpackhi_epi32(0b00000000_11111111, a, b);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 25, 9, 26, 10, 29, 13, 30, 14);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_unpacklo_epi32() {
        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm512_set_epi32(
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = _mm512_unpacklo_epi32(a, b);
        let e = _mm512_set_epi32(19, 3, 20, 4, 23, 7, 24, 8, 27, 11, 28, 12, 31, 15, 32, 16);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_unpacklo_epi32() {
        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm512_set_epi32(
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = _mm512_mask_unpacklo_epi32(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_unpacklo_epi32(a, 0b11111111_11111111, a, b);
        let e = _mm512_set_epi32(19, 3, 20, 4, 23, 7, 24, 8, 27, 11, 28, 12, 31, 15, 32, 16);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_unpacklo_epi32() {
        let a = _mm512_set_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b = _mm512_set_epi32(
            17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
        );
        let r = _mm512_maskz_unpacklo_epi32(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_unpacklo_epi32(0b00000000_11111111, a, b);
        let e = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 27, 11, 28, 12, 31, 15, 32, 16);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_kand() {
        let a: u16 = 0b11001100_00110011;
        let b: u16 = 0b11001100_00110011;
        let r = _mm512_kand(a, b);
        let e: u16 = 0b11001100_00110011;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_kand_mask16() {
        let a: u16 = 0b11001100_00110011;
        let b: u16 = 0b11001100_00110011;
        let r = _kand_mask16(a, b);
        let e: u16 = 0b11001100_00110011;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_kor() {
        let a: u16 = 0b11001100_00110011;
        let b: u16 = 0b00101110_00001011;
        let r = _mm512_kor(a, b);
        let e: u16 = 0b11101110_00111011;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_kor_mask16() {
        let a: u16 = 0b11001100_00110011;
        let b: u16 = 0b00101110_00001011;
        let r = _kor_mask16(a, b);
        let e: u16 = 0b11101110_00111011;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_kxor() {
        let a: u16 = 0b11001100_00110011;
        let b: u16 = 0b00101110_00001011;
        let r = _mm512_kxor(a, b);
        let e: u16 = 0b11100010_00111000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_kxor_mask16() {
        let a: u16 = 0b11001100_00110011;
        let b: u16 = 0b00101110_00001011;
        let r = _kxor_mask16(a, b);
        let e: u16 = 0b11100010_00111000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_knot() {
        let a: u16 = 0b11001100_00110011;
        let r = _mm512_knot(a);
        let e: u16 = 0b00110011_11001100;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_knot_mask16() {
        let a: u16 = 0b11001100_00110011;
        let r = _knot_mask16(a);
        let e: u16 = 0b00110011_11001100;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_kandn() {
        let a: u16 = 0b11001100_00110011;
        let b: u16 = 0b00101110_00001011;
        let r = _mm512_kandn(a, b);
        let e: u16 = 0b00100010_00001000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_kandn_mask16() {
        let a: u16 = 0b11001100_00110011;
        let b: u16 = 0b00101110_00001011;
        let r = _kandn_mask16(a, b);
        let e: u16 = 0b00100010_00001000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_kxnor() {
        let a: u16 = 0b11001100_00110011;
        let b: u16 = 0b00101110_00001011;
        let r = _mm512_kxnor(a, b);
        let e: u16 = 0b00011101_11000111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_kxnor_mask16() {
        let a: u16 = 0b11001100_00110011;
        let b: u16 = 0b00101110_00001011;
        let r = _kxnor_mask16(a, b);
        let e: u16 = 0b00011101_11000111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_kmov() {
        let a: u16 = 0b11001100_00110011;
        let r = _mm512_kmov(a);
        let e: u16 = 0b11001100_00110011;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_kunpackb() {
        let a: u16 = 0b11001100_00110011;
        let b: u16 = 0b00101110_00001011;
        let r = _mm512_kunpackb(a, b);
        let e: u16 = 0b00101110_00110011;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_kortestc() {
        let a: u16 = 0b11001100_00110011;
        let b: u16 = 0b00101110_00001011;
        let r = _mm512_kortestc(a, b);
        assert_eq!(r, 0);
        let b: u16 = 0b11111111_11111111;
        let r = _mm512_kortestc(a, b);
        assert_eq!(r, 1);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_test_epi32_mask() {
        let a = _mm512_set1_epi32(1 << 0);
        let b = _mm512_set1_epi32(1 << 0 | 1 << 1);
        let r = _mm512_test_epi32_mask(a, b);
        let e: __mmask16 = 0b11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_test_epi32_mask() {
        let a = _mm512_set1_epi32(1 << 0);
        let b = _mm512_set1_epi32(1 << 0 | 1 << 1);
        let r = _mm512_mask_test_epi32_mask(0, a, b);
        assert_eq!(r, 0);
        let r = _mm512_mask_test_epi32_mask(0b11111111_11111111, a, b);
        let e: __mmask16 = 0b11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_testn_epi32_mask() {
        let a = _mm512_set1_epi32(1 << 0);
        let b = _mm512_set1_epi32(1 << 0 | 1 << 1);
        let r = _mm512_testn_epi32_mask(a, b);
        let e: __mmask16 = 0b00000000_00000000;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_testn_epi32_mask() {
        let a = _mm512_set1_epi32(1 << 0);
        let b = _mm512_set1_epi32(1 << 1);
        let r = _mm512_mask_test_epi32_mask(0, a, b);
        assert_eq!(r, 0);
        let r = _mm512_mask_testn_epi32_mask(0b11111111_11111111, a, b);
        let e: __mmask16 = 0b11111111_11111111;
        assert_eq!(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_loadu_epi32() {
        let a = &[4, 3, 2, 5, 8, 9, 64, 50, -4, -3, -2, -5, -8, -9, -64, -50];
        let p = a.as_ptr();
        let r = _mm512_loadu_epi32(black_box(p));
        let e = _mm512_setr_epi32(4, 3, 2, 5, 8, 9, 64, 50, -4, -3, -2, -5, -8, -9, -64, -50);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_storeu_epi32() {
        let a = _mm512_set1_epi32(9);
        let mut r = _mm512_undefined_epi32();
        _mm512_storeu_epi32(&mut r as *mut _ as *mut i32, a);
        assert_eq_m512i(r, a);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_load_epi32() {
        #[repr(align(64))]
        struct Align {
            data: [i32; 16], // 64 bytes
        }
        let a = Align {
            data: [4, 3, 2, 5, 8, 9, 64, 50, -4, -3, -2, -5, -8, -9, -64, -50],
        };
        let p = (a.data).as_ptr();
        let r = _mm512_load_epi32(black_box(p));
        let e = _mm512_setr_epi32(4, 3, 2, 5, 8, 9, 64, 50, -4, -3, -2, -5, -8, -9, -64, -50);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_store_epi32() {
        let a = _mm512_set1_epi32(9);
        let mut r = _mm512_undefined_epi32();
        _mm512_store_epi32(&mut r as *mut _ as *mut i32, a);
        assert_eq_m512i(r, a);
    }
    */
}
