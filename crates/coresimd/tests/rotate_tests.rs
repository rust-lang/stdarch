//! rotate instruction tests

#![feature(stdsimd)]
#![feature(proc_macro)]
#![feature(avx512_target_feature)]
#![feature(abi_vectorcall)]

extern crate stdsimd_test;
extern crate stdsimd;

use stdsimd::simd::*;
use stdsimd_test::assert_instr;

// Verify that supported hardware compiles rotates into single instructions

#[inline]
#[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), target_feature(enable = "avx512f"))]
#[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), assert_instr(vprorvq))]
unsafe fn rotate_right_variable(x: u64x8) -> u64x8 {
    x.rotate_right(u64x8::new(0, 1, 2, 3, 4, 5, 6, 7))
}

#[inline]
#[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), target_feature(enable = "avx512f"))]
#[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), assert_instr(vprolvq))]
unsafe fn rotate_left_variable(x: u64x8) -> u64x8 {
    x.rotate_left(u64x8::new(0, 1, 2, 3, 4, 5, 6, 7))
}

#[inline]
#[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), target_feature(enable = "avx512f"))]
#[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), assert_instr(vprorq))]
unsafe fn rotate_right(x: u64x8) -> u64x8 {
    x.rotate_right(u64x8::splat(12))
}

#[inline]
#[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), target_feature(enable = "avx512f"))]
#[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), assert_instr(vprolq))]
unsafe fn rotate_left(x: u64x8) -> u64x8 {
    x.rotate_left(u64x8::splat(12))
}

#[inline]
#[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), target_feature(enable = "avx512f"))]
#[cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), assert_instr(vprolq))]
unsafe fn rotate_left_x2(x: u64x2) -> u64x2 {
    x.rotate_left(u64x2::splat(12))
}
