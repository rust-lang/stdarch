//! `LoongArch32` intrinsics

use crate::arch::asm;

#[allow(improper_ctypes)]
unsafe extern "unadjusted" {
    #[link_name = "llvm.loongarch.cacop.w"]
    fn __cacop(a: i32, b: i32, c: i32);
    #[link_name = "llvm.loongarch.csrrd.w"]
    fn __csrrd(a: i32) -> i32;
    #[link_name = "llvm.loongarch.csrwr.w"]
    fn __csrwr(a: i32, b: i32) -> i32;
    #[link_name = "llvm.loongarch.csrxchg.w"]
    fn __csrxchg(a: i32, b: i32, c: i32) -> i32;
}

/// Generates the cache operation instruction
#[inline]
#[rustc_legacy_const_generics(0, 2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn cacop<const IMM5: i32, const IMM_S12: i32>(b: i32) {
    static_assert_uimm_bits!(IMM5, 5);
    static_assert_simm_bits!(IMM_S12, 12);
    __cacop(IMM5, b, IMM_S12);
}

/// Reads the CSR
#[inline]
#[rustc_legacy_const_generics(0)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn csrrd<const IMM14: i32>() -> i32 {
    static_assert_uimm_bits!(IMM14, 14);
    __csrrd(IMM14)
}

/// Writes the CSR
#[inline]
#[rustc_legacy_const_generics(1)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn csrwr<const IMM14: i32>(a: i32) -> i32 {
    static_assert_uimm_bits!(IMM14, 14);
    __csrwr(a, IMM14)
}

/// Exchanges the CSR
#[inline]
#[rustc_legacy_const_generics(2)]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
pub unsafe fn csrxchg<const IMM14: i32>(a: i32, b: i32) -> i32 {
    static_assert_uimm_bits!(IMM14, 14);
    __csrxchg(a, b, IMM14)
}
