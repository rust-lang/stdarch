//! ARM SVE (Scalable Vector Extension) intrinsics.
//!
//! This module provides Rust bindings for ARM SVE and SVE2 SIMD intrinsics.
//! SVE vectors are *scalable* — their width is determined at runtime by the
//! hardware (128–2048 bits in 128-bit increments). The types defined here
//! use `#[rustc_scalable_vector]` to represent these runtime-sized vectors.
//!
//! # SVE2 Bit Permutation (FEAT_SVE_BitPerm)
//!
//! The `sve2_bitperm` sub-module exposes three bit-permutation instructions
//! added by the SVE2 BitPerm extension:
//!
//! - **BDEP** (Bit Deposit) — scatter source bits into positions selected by a mask.
//!   Equivalent to x86 `PDEP`.
//! - **BEXT** (Bit Extract) — gather bits from positions selected by a mask.
//!   Equivalent to x86 `PEXT`.
//! - **BGRP** (Bit Group) — partition bits into two groups: those selected by the
//!   mask (placed at the bottom) and those not selected (placed at the top).
//!
//! Gated by `#[target_feature(enable = "sve2-bitperm")]`.
//!
//! # References
//!
//! - [ARM Architecture Reference Manual — SVE2](https://developer.arm.com/documentation/ddi0602/)
//! - [ARM C Language Extensions (ACLE) for SVE](https://arm-software.github.io/acle/main/acle-sve.html)

mod types;
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub use self::types::*;

mod sve2_bitperm;
#[unstable(feature = "stdarch_aarch64_sve2_bitperm", issue = "none")]
pub use self::sve2_bitperm::*;
