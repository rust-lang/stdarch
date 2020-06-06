use crate::{
    core_arch::{simd::*, simd_llvm::*, x86::*},
    mem::{self, transmute},
};

#[cfg(test)]
use stdarch_test::assert_instr;

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

/// Returns vector of type `__m512i` with all elements set to zero.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#avx512techs=AVX512F&expand=33,34,4990&text=_mm512_setzero_si512)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vxorps))]
pub unsafe fn _mm512_setzero_si512() -> __m512i {
    // All-0 is a properly initialized __m512i
    mem::zeroed()
}

/// Sets packed 32-bit integers in `dst` with the supplied values in reverse
/// order.
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_setr_epi32(
    e15: i32,
    e14: i32,
    e13: i32,
    e12: i32,
    e11: i32,
    e10: i32,
    e9: i32,
    e8: i32,
    e7: i32,
    e6: i32,
    e5: i32,
    e4: i32,
    e3: i32,
    e2: i32,
    e1: i32,
    e0: i32,
) -> __m512i {
    let r = i32x16(
        e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0,
    );
    transmute(r)
}

/// Broadcast 64-bit integer `a` to all elements of `dst`.
#[inline]
#[target_feature(enable = "avx512f")]
pub unsafe fn _mm512_set1_epi64(a: i64) -> __m512i {
    transmute(i64x8::splat(a))
}

/// Compare packed unsigned 64-bit integers in a and b for less-than, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmplt_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epu64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_lt(a.as_u64x8(), b.as_u64x8()))
}

/// Compare packed unsigned 64-bit integers in a and b for less-than, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmplt_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epu64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmplt_epu64_mask(a, b) & m
}

/// Compare packed unsigned 64-bit integers in a and b for greater-than, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpgt_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpgt_epu64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_gt(a.as_u64x8(), b.as_u64x8()))
}

/// Compare packed unsigned 64-bit integers in a and b for greater-than, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpgt_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpgt_epu64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpgt_epu64_mask(a, b) & m
}

/// Compare packed unsigned 64-bit integers in a and b for less-than-or-equal, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmple_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmple_epu64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpgt_epu64_mask(b, a)
}

/// Compare packed unsigned 64-bit integers in a and b for less-than-or-equal, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmple_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmple_epu64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpgt_epu64_mask(b, a) & m
}

/// Compare packed unsigned 64-bit integers in a and b for greater-than-or-equal, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpge_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpge_epu64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmplt_epu64_mask(b, a)
}

/// Compare packed unsigned 64-bit integers in a and b for greater-than-or-equal, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpge_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpge_epu64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmplt_epu64_mask(b, a) & m
}

/// Compare packed unsigned 64-bit integers in a and b for equality, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpeq_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpeq_epu64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_eq(a.as_u64x8(), b.as_u64x8()))
}

/// Compare packed unsigned 64-bit integers in a and b for equality, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpeq_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpeq_epu64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpeq_epu64_mask(a, b) & m
}

/// Compare packed unsigned 64-bit integers in a and b for inequality, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpneq_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpneq_epu64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_ne(a.as_u64x8(), b.as_u64x8()))
}

///Compare packed unsigned 64-bit integers in a and b for inequality, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpneq_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpneq_epu64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpneq_epu64_mask(a, b) & m
}

/// Compare packed unsigned 64-bit integers in a and b based on the comparison operand specified by op.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, op = 0))]
pub unsafe fn _mm512_cmp_epu64_mask(a: __m512i, b: __m512i, op: _MM_CMPINT_ENUM) -> __mmask8 {
    let neg_one = -1;
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpuq(a.as_i64x8(), b.as_i64x8(), $imm3, neg_one)
        };
    }
    let r = constify_imm3!(op, call);
    transmute(r)
}

/// Compare packed unsigned 64-bit integers in a and b based on the comparison operand specified by op,
///  using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_epu64)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, op = 0))]
pub unsafe fn _mm512_mask_cmp_epu64_mask(
    m: __mmask8,
    a: __m512i,
    b: __m512i,
    op: _MM_CMPINT_ENUM,
) -> __mmask8 {
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpuq(a.as_i64x8(), b.as_i64x8(), $imm3, m as i8)
        };
    }
    let r = constify_imm3!(op, call);
    transmute(r)
}

/// Compare packed signed 64-bit integers in a and b for less-than, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmplt_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epi64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_lt(a.as_i64x8(), b.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b for less-than, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmplt_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epi64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmplt_epi64_mask(a, b) & m
}

/// Compare packed signed 64-bit integers in a and b for greater-than, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpgt_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpgt_epi64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_gt(a.as_i64x8(), b.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b for greater-than, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpgt_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpgt_epi64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpgt_epi64_mask(a, b) & m
}

/// Compare packed signed 64-bit integers in a and b for less-than-or-equal, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmple_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmple_epi64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpgt_epi64_mask(b, a)
}

/// Compare packed signed 64-bit integers in a and b for less-than-or-equal, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmple_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmple_epi64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpgt_epi64_mask(b, a) & m
}

/// Compare packed signed 64-bit integers in a and b for greater-than-or-equal, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpge_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpge_epi64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmplt_epi64_mask(b, a)
}

/// Compare packed signed 64-bit integers in a and b for greater-than-or-equal, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpge_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpge_epi64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmplt_epi64_mask(b, a) & m
}

/// Compare packed signed 64-bit integers in a and b for equality, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpeq_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpeq_epi64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_eq(a.as_i64x8(), b.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b for equality, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpeq_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpeq_epi64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpeq_epi64_mask(a, b) & m
}

/// Compare packed signed 64-bit integers in a and b for inequality, and store the results in a mask vector.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062&text=_mm512_cmpneq_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpneq_epi64_mask(a: __m512i, b: __m512i) -> __mmask8 {
    simd_bitmask::<__m512i, _>(simd_ne(a.as_i64x8(), b.as_i64x8()))
}

/// Compare packed signed 64-bit integers in a and b for inequality, and store the results in a mask vector k
/// using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmpneq_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpneq_epi64_mask(m: __mmask8, a: __m512i, b: __m512i) -> __mmask8 {
    _mm512_cmpneq_epi64_mask(a, b) & m
}

/// Compare packed signed 64-bit integers in a and b based on the comparison operand specified by op.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, op = 0))]
pub unsafe fn _mm512_cmp_epi64_mask(a: __m512i, b: __m512i, op: _MM_CMPINT_ENUM) -> __mmask8 {
    let neg_one = -1;
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpq(a.as_i64x8(), b.as_i64x8(), $imm3, neg_one)
        };
    }
    let r = constify_imm3!(op, call);
    transmute(r)
}

/// Compare packed signed 64-bit integers in a and b based on the comparison operand specified by op,
///  using zeromask m (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#expand=727,1063,4909,1062,1062,1063&text=_mm512_mask_cmp_epi64)
#[inline]
#[target_feature(enable = "avx512f")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, op = 0))]
pub unsafe fn _mm512_mask_cmp_epi64_mask(
    m: __mmask8,
    a: __m512i,
    b: __m512i,
    op: _MM_CMPINT_ENUM,
) -> __mmask8 {
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpq(a.as_i64x8(), b.as_i64x8(), $imm3, m as i8)
        };
    }
    let r = constify_imm3!(op, call);
    transmute(r)
}

/// Equal
pub const _MM_CMPINT_EQ: _MM_CMPINT_ENUM = 0x00;
/// Less-than
pub const _MM_CMPINT_LT: _MM_CMPINT_ENUM = 0x01;
/// Less-than-or-equal
pub const _MM_CMPINT_LE: _MM_CMPINT_ENUM = 0x02;
/// False
pub const _MM_CMPINT_FALSE: _MM_CMPINT_ENUM = 0x03;
/// Not-equal
pub const _MM_CMPINT_NE: _MM_CMPINT_ENUM = 0x04;
/// Not less-than
pub const _MM_CMPINT_NLT: _MM_CMPINT_ENUM = 0x05;
/// Not less-than-or-equal
pub const _MM_CMPINT_NLE: _MM_CMPINT_ENUM = 0x06;
/// True
pub const _MM_CMPINT_TRUE: _MM_CMPINT_ENUM = 0x07;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.x86.avx512.mask.ucmp.q.512"]
    fn vpcmpuq(a: i64x8, b: i64x8, op: i32, m: i8) -> i8;
    #[link_name = "llvm.x86.avx512.mask.cmp.q.512"]
    fn vpcmpq(a: i64x8, b: i64x8, op: i32, m: i8) -> i8;
}

#[cfg(test)]
mod tests {
    use std;
    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;

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
        let r = _mm512_mask_abs_epi32(a, 0b11111111, a);
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
        let r = _mm512_maskz_abs_epi32(0b11111111, a);
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
}
