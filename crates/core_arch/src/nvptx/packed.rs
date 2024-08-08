//! NVPTX Packed data types (SIMD)
//!
//! Packed Data Types is what PTX calls SIMD types. See [PTX ISA (Packed Data Types)](https://docs.nvidia.com/cuda/parallel-thread-execution/#packed-data-types) for a full reference.
//!
//! Note: #[assert_instr] tests are not actually being run on nvptx due to being a `no_std` target incapable of running tests. Something like FileCheck would be appropriate for verifying the correct instruction is used.

use crate::intrinsics::simd::*;

#[allow(improper_ctypes)]
extern "C" {
    #[link_name = "llvm.minimum.v2f16"]
    fn llvm_f16x2_min(a: f16x2, b: f16x2) -> f16x2;
    #[link_name = "llvm.maximum.v2f16"]
    fn llvm_f16x2_max(a: f16x2, b: f16x2) -> f16x2;
}

#[allow(unused)]
macro_rules! simd_types {
    ($(
        $(#[$doc:meta])*
        pub struct $name:ident($($fields:tt)*);
    )*) => ($(
        $(#[$doc])*
        #[derive(Copy, Clone, Debug)]
        #[allow(non_camel_case_types)]
        #[repr(simd)]
        #[allow(clippy::missing_inline_in_public_items)]
        pub struct $name($($fields)*);
    )*)
}

simd_types! {
    /// PTX-specific 32-bit wide floating point (f16 x 2) vector type
    #[unstable(feature = "stdarch_nvptx", issue = "111199")]
    pub struct f16x2(f16, f16);

}

/// Add two values
///
/// https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-add
#[inline]
#[cfg_attr(test, assert_instr(add.rn.f16x22))]
#[unstable(feature = "stdarch_nvptx", issue = "111199")]
pub unsafe fn f16x2_add(a: f16x2, b: f16x2) -> f16x2 {
    simd_add(a, b)
}

/// Subtract two values
///
/// https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-sub
#[inline]
#[cfg_attr(test, assert_instr(sub.rn.f16x2))]
#[unstable(feature = "stdarch_nvptx", issue = "111199")]
pub unsafe fn f16x2_sub(a: f16x2, b: f16x2) -> f16x2 {
    simd_sub(a, b)
}

/// Multiply two values
///
/// https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-mul
#[inline]
#[cfg_attr(test, assert_instr(mul.rn.f16x2))]
#[unstable(feature = "stdarch_nvptx", issue = "111199")]
pub unsafe fn f16x2_mul(a: f16x2, b: f16x2) -> f16x2 {
    simd_mul(a, b)
}

/// Fused multiply-add
///
/// https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-fma
#[inline]
#[cfg_attr(test, assert_instr(fma.rn.f16x2))]
#[unstable(feature = "stdarch_nvptx", issue = "111199")]
pub unsafe fn f16x2_fma(a: f16x2, b: f16x2, c: f16x2) -> f16x2 {
    simd_fma(a, b, c)
}

/// Arithmetic negate
///
/// https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-neg
#[inline]
#[cfg_attr(test, assert_instr(neg.f16x2))]
#[unstable(feature = "stdarch_nvptx", issue = "111199")]
pub unsafe fn f16x2_neg(a: f16x2) -> f16x2 {
    simd_neg(a)
}

/// Find the minimum of two values
///
/// https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-min
#[inline]
#[cfg_attr(test, assert_instr(min.NaN.f16x2))]
#[unstable(feature = "stdarch_nvptx", issue = "111199")]
pub unsafe fn f16x2_min(a: f16x2, b: f16x2) -> f16x2 {
    llvm_f16x2_min(a, b)
}

/// Find the maximum of two values
///
/// https://docs.nvidia.com/cuda/parallel-thread-execution/#half-precision-floating-point-instructions-max
#[inline]
#[cfg_attr(test, assert_instr(max.NaN.f16x2))]
#[unstable(feature = "stdarch_nvptx", issue = "111199")]
pub unsafe fn f16x2_max(a: f16x2, b: f16x2) -> f16x2 {
    llvm_f16x2_max(a, b)
}
