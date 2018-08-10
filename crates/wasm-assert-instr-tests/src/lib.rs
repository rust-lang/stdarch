//! Includes assert_instr tests for wasm that currently pass.
#![feature(stdsimd)]
#![cfg_attr(test, feature(use_extern_macros))]

extern crate coresimd;
#[cfg(test)]
extern crate stdsimd_test;
#[cfg(all(test, target_arch = "wasm32"))]
extern crate wasm_bindgen_test;

use coresimd::arch::wasm32::*;

#[cfg(test)]
use stdsimd_test::assert_instr;

#[cfg(test)]
use wasm_bindgen_test::*;

#[cfg_attr(test, assert_instr(foo))]
pub fn i8x16_add(a: v128, b: v128) -> v128 {
    unsafe { i8x16::add(a, b) }
}
