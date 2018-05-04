//! SIMD and vendor intrinsics support library.
//!
//! This documentation is for the `coresimd` crate, but you probably want to
//! use the [`stdsimd` crate][stdsimd] which should have more complete
//! documentation.
//!
//! [stdsimd]: https://rust-lang-nursery.github.io/stdsimd/x86_64/stdsimd/

#![cfg_attr(stdsimd_strict, deny(warnings))]
#![allow(dead_code)]
#![allow(unused_features)]
#![feature(const_fn, link_llvm_intrinsics, platform_intrinsics, repr_simd,
           simd_ffi, asm,
           integer_atomics, stmt_expr_attributes, core_intrinsics,
           crate_in_paths, no_core, attr_literals, rustc_attrs, stdsimd,
           staged_api, core_float, core_slice_ext, align_offset,
           doc_cfg, mmx_target_feature, tbm_target_feature,
           sse4a_target_feature, arm_target_feature, aarch64_target_feature,
           mips_target_feature)]
#![cfg_attr(test,
            feature(proc_macro, test, attr_literals, abi_vectorcall,
                    untagged_unions))]
#![cfg_attr(feature = "cargo-clippy",
            allow(inline_always, too_many_arguments, cast_sign_loss,
                  cast_lossless, cast_possible_wrap,
                  cast_possible_truncation, cast_precision_loss,
                  shadow_reuse, cyclomatic_complexity, similar_names,
                  many_single_char_names))]
#![cfg_attr(test, allow(unused_imports))]
#![no_core]
#![unstable(feature = "stdsimd", issue = "0")]
#![doc(test(attr(deny(warnings))),
       test(attr(allow(dead_code, deprecated, unused_variables,
                       unused_mut))))]

#[cfg_attr(not(test), macro_use)]
extern crate core as _core;
#[cfg(test)]
#[macro_use]
extern crate std;
#[cfg(test)]
#[macro_use]
extern crate stdsimd;
#[cfg(test)]
extern crate stdsimd_test;
#[cfg(test)]
extern crate test;

#[doc(hidden)]
macro_rules! cfg_if {
    ($(
        if #[cfg($($meta:meta),*)] { $($it:item)* }
    ) else * else {
        $($it2:item)*
    }) => {
        __cfg_if_items! {
            () ;
            $( ( ($($meta),*) ($($it)*) ), )*
            ( () ($($it2)*) ),
        }
    };
    (
        if #[cfg($($i_met:meta),*)] { $($i_it:item)* }
        $(
            else if #[cfg($($e_met:meta),*)] { $($e_it:item)* }
        )*
    ) => {
        __cfg_if_items! {
            () ;
            ( ($($i_met),*) ($($i_it)*) ),
            $( ( ($($e_met),*) ($($e_it)*) ), )*
            ( () () ),
        }
    }
}

#[doc(hidden)]
macro_rules! __cfg_if_items {
    (($($not:meta,)*) ; ) => {};
    (($($not:meta,)*) ; ( ($($m:meta),*) ($($it:item)*) ), $($rest:tt)*) => {
        __cfg_if_apply! { cfg(all($($m,)* not(any($($not),*)))), $($it)* }
        __cfg_if_items! { ($($not,)* $($m,)*) ; $($rest)* }
    }
}

#[doc(hidden)]
macro_rules! __cfg_if_apply {
    ($m:meta, $($it:item)*) => {
        $(#[$m] $it)*
    }
}

macro_rules! test_v16 {
    ($item:item) => {};
}
macro_rules! test_v32 {
    ($item:item) => {};
}
macro_rules! test_v64 {
    ($item:item) => {};
}
macro_rules! test_v128 {
    ($item:item) => {};
}
macro_rules! test_v256 {
    ($item:item) => {};
}
macro_rules! test_v512 {
    ($item:item) => {};
}
macro_rules! vector_impl {
    ($([$f:ident, $($args:tt)*]),*) => { $($f!($($args)*);)* }
}

#[path = "../../../coresimd/mod.rs"]
mod coresimd;

pub use coresimd::arch;
pub use coresimd::simd;

#[allow(unused_imports)]
use _core::clone;
#[allow(unused_imports)]
use _core::cmp;
#[allow(unused_imports)]
use _core::convert;
#[allow(unused_imports)]
use _core::default;
#[allow(unused_imports)]
use _core::fmt;
#[allow(unused_imports)]
use _core::hash;
#[allow(unused_imports)]
use _core::intrinsics;
#[allow(unused_imports)]
use _core::iter;
#[allow(unused_imports)]
use _core::marker;
#[allow(unused_imports)]
use _core::mem;
#[allow(unused_imports)]
use _core::num;
#[allow(unused_imports)]
use _core::ops;
#[allow(unused_imports)]
use _core::option;
#[allow(unused_imports)]
use _core::prelude;
#[allow(unused_imports)]
use _core::ptr;
#[allow(unused_imports)]
use _core::result;
#[allow(unused_imports)]
use _core::slice;
