#![doc = include_str!("core_arch_docs.md")]
#![allow(improper_ctypes_definitions)]
#![allow(dead_code)]
#![allow(unused_features)]
#![allow(internal_features)]
#![deny(rust_2018_idioms)]
#![feature(
    custom_inner_attributes,
    link_llvm_intrinsics,
    platform_intrinsics,
    repr_simd,
    simd_ffi,
    proc_macro_hygiene,
    stmt_expr_attributes,
    intrinsics,
    no_core,
    rustc_attrs,
    staged_api,
    doc_cfg,
    tbm_target_feature,
    sse4a_target_feature,
    riscv_target_feature,
    arm_target_feature,
    avx512_target_feature,
    mips_target_feature,
    powerpc_target_feature,
    wasm_target_feature,
    abi_unadjusted,
    rtm_target_feature,
    allow_internal_unstable,
    decl_macro,
    asm_const,
    target_feature_11,
    inline_const,
    generic_arg_infer
)]
#![cfg_attr(test, feature(test, abi_vectorcall))]
#![deny(clippy::missing_inline_in_public_items)]
#![allow(
    clippy::identity_op,
    clippy::inline_always,
    clippy::too_many_arguments,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cognitive_complexity,
    clippy::many_single_char_names,
    clippy::missing_safety_doc,
    clippy::shadow_reuse,
    clippy::similar_names,
    clippy::unusual_byte_groupings,
    clippy::wrong_self_convention
)]
#![cfg_attr(test, allow(unused_imports))]
#![no_std]
#![stable(feature = "stdsimd", since = "1.27.0")]
#![doc(
    test(attr(deny(warnings))),
    test(attr(allow(dead_code, deprecated, unused_variables, unused_mut)))
)]
#![cfg_attr(
    test,
    feature(stdarch_arm_feature_detection, stdarch_powerpc_feature_detection)
)]

#[cfg(test)]
#[macro_use]
extern crate std;
#[cfg(test)]
#[macro_use]
extern crate std_detect;
#[path = "mod.rs"]
mod core_arch;

#[stable(feature = "stdsimd", since = "1.27.0")]
pub mod arch {
    #[stable(feature = "stdsimd", since = "1.27.0")]
    #[allow(unused_imports)]
    pub use crate::core_arch::arch::*;
    #[stable(feature = "stdsimd", since = "1.27.0")]
    pub use core::arch::asm;
}

#[allow(unused_imports)]
use core::{convert, ffi, hint, marker, mem, ops, ptr, sync};

// `core` is changing the feature name for the `intrinsics` module.
// To permit that transition, we avoid using that feature for now.
mod intrinsics {
    extern "rust-intrinsic" {
        /// Emits a `!nontemporal` store according to LLVM (see their docs).
        /// Probably will never become stable.
        #[rustc_nounwind]
        pub fn nontemporal_store<T>(ptr: *mut T, val: T);

        /// Aborts the execution of the process.
        ///
        /// Note that, unlike most intrinsics, this is safe to call;
        /// it does not require an `unsafe` block.
        /// Therefore, implementations must not require the user to uphold
        /// any safety invariants.
        ///
        /// [`std::process::abort`](../../std/process/fn.abort.html) is to be preferred if possible,
        /// as its behavior is more user-friendly and more stable.
        ///
        /// The current implementation of `intrinsics::abort` is to invoke an invalid instruction,
        /// on most platforms.
        /// On Unix, the
        /// process will probably terminate with a signal like `SIGABRT`, `SIGILL`, `SIGTRAP`, `SIGSEGV` or
        /// `SIGBUS`.  The precise behaviour is not guaranteed and not stable.
        #[rustc_safe_intrinsic]
        #[rustc_nounwind]
        pub fn abort() -> !;

        /// Stores a value if the current value is the same as the `old` value.
        ///
        /// The stabilized version of this intrinsic is available on the
        /// [`atomic`] types via the `compare_exchange` method by passing
        /// [`Ordering::Relaxed`] as both the success and failure parameters.
        /// For example, [`AtomicBool::compare_exchange`].
        #[rustc_nounwind]
        pub fn atomic_cxchg_relaxed_relaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
        /// Stores a value if the current value is the same as the `old` value.
        ///
        /// The stabilized version of this intrinsic is available on the
        /// [`atomic`] types via the `compare_exchange` method by passing
        /// [`Ordering::Relaxed`] and [`Ordering::Acquire`] as the success and failure parameters.
        /// For example, [`AtomicBool::compare_exchange`].
        #[rustc_nounwind]
        pub fn atomic_cxchg_relaxed_acquire<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
        /// Stores a value if the current value is the same as the `old` value.
        ///
        /// The stabilized version of this intrinsic is available on the
        /// [`atomic`] types via the `compare_exchange` method by passing
        /// [`Ordering::Relaxed`] and [`Ordering::SeqCst`] as the success and failure parameters.
        /// For example, [`AtomicBool::compare_exchange`].
        #[rustc_nounwind]
        pub fn atomic_cxchg_relaxed_seqcst<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
        /// Stores a value if the current value is the same as the `old` value.
        ///
        /// The stabilized version of this intrinsic is available on the
        /// [`atomic`] types via the `compare_exchange` method by passing
        /// [`Ordering::Acquire`] and [`Ordering::Relaxed`] as the success and failure parameters.
        /// For example, [`AtomicBool::compare_exchange`].
        #[rustc_nounwind]
        pub fn atomic_cxchg_acquire_relaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
        /// Stores a value if the current value is the same as the `old` value.
        ///
        /// The stabilized version of this intrinsic is available on the
        /// [`atomic`] types via the `compare_exchange` method by passing
        /// [`Ordering::Acquire`] as both the success and failure parameters.
        /// For example, [`AtomicBool::compare_exchange`].
        #[rustc_nounwind]
        pub fn atomic_cxchg_acquire_acquire<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
        /// Stores a value if the current value is the same as the `old` value.
        ///
        /// The stabilized version of this intrinsic is available on the
        /// [`atomic`] types via the `compare_exchange` method by passing
        /// [`Ordering::Acquire`] and [`Ordering::SeqCst`] as the success and failure parameters.
        /// For example, [`AtomicBool::compare_exchange`].
        #[rustc_nounwind]
        pub fn atomic_cxchg_acquire_seqcst<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
        /// Stores a value if the current value is the same as the `old` value.
        ///
        /// The stabilized version of this intrinsic is available on the
        /// [`atomic`] types via the `compare_exchange` method by passing
        /// [`Ordering::Release`] and [`Ordering::Relaxed`] as the success and failure parameters.
        /// For example, [`AtomicBool::compare_exchange`].
        #[rustc_nounwind]
        pub fn atomic_cxchg_release_relaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
        /// Stores a value if the current value is the same as the `old` value.
        ///
        /// The stabilized version of this intrinsic is available on the
        /// [`atomic`] types via the `compare_exchange` method by passing
        /// [`Ordering::Release`] and [`Ordering::Acquire`] as the success and failure parameters.
        /// For example, [`AtomicBool::compare_exchange`].
        #[rustc_nounwind]
        pub fn atomic_cxchg_release_acquire<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
        /// Stores a value if the current value is the same as the `old` value.
        ///
        /// The stabilized version of this intrinsic is available on the
        /// [`atomic`] types via the `compare_exchange` method by passing
        /// [`Ordering::Release`] and [`Ordering::SeqCst`] as the success and failure parameters.
        /// For example, [`AtomicBool::compare_exchange`].
        #[rustc_nounwind]
        pub fn atomic_cxchg_release_seqcst<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
        /// Stores a value if the current value is the same as the `old` value.
        ///
        /// The stabilized version of this intrinsic is available on the
        /// [`atomic`] types via the `compare_exchange` method by passing
        /// [`Ordering::AcqRel`] and [`Ordering::Relaxed`] as the success and failure parameters.
        /// For example, [`AtomicBool::compare_exchange`].
        #[rustc_nounwind]
        pub fn atomic_cxchg_acqrel_relaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
        /// Stores a value if the current value is the same as the `old` value.
        ///
        /// The stabilized version of this intrinsic is available on the
        /// [`atomic`] types via the `compare_exchange` method by passing
        /// [`Ordering::AcqRel`] and [`Ordering::Acquire`] as the success and failure parameters.
        /// For example, [`AtomicBool::compare_exchange`].
        #[rustc_nounwind]
        pub fn atomic_cxchg_acqrel_acquire<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
        /// Stores a value if the current value is the same as the `old` value.
        ///
        /// The stabilized version of this intrinsic is available on the
        /// [`atomic`] types via the `compare_exchange` method by passing
        /// [`Ordering::AcqRel`] and [`Ordering::SeqCst`] as the success and failure parameters.
        /// For example, [`AtomicBool::compare_exchange`].
        #[rustc_nounwind]
        pub fn atomic_cxchg_acqrel_seqcst<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
        /// Stores a value if the current value is the same as the `old` value.
        ///
        /// The stabilized version of this intrinsic is available on the
        /// [`atomic`] types via the `compare_exchange` method by passing
        /// [`Ordering::SeqCst`] and [`Ordering::Relaxed`] as the success and failure parameters.
        /// For example, [`AtomicBool::compare_exchange`].
        #[rustc_nounwind]
        pub fn atomic_cxchg_seqcst_relaxed<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
        /// Stores a value if the current value is the same as the `old` value.
        ///
        /// The stabilized version of this intrinsic is available on the
        /// [`atomic`] types via the `compare_exchange` method by passing
        /// [`Ordering::SeqCst`] and [`Ordering::Acquire`] as the success and failure parameters.
        /// For example, [`AtomicBool::compare_exchange`].
        #[rustc_nounwind]
        pub fn atomic_cxchg_seqcst_acquire<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
        /// Stores a value if the current value is the same as the `old` value.
        ///
        /// The stabilized version of this intrinsic is available on the
        /// [`atomic`] types via the `compare_exchange` method by passing
        /// [`Ordering::SeqCst`] as both the success and failure parameters.
        /// For example, [`AtomicBool::compare_exchange`].
        #[rustc_nounwind]
        pub fn atomic_cxchg_seqcst_seqcst<T: Copy>(dst: *mut T, old: T, src: T) -> (T, bool);
    }
}
