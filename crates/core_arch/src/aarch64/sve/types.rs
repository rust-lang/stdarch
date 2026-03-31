//! SVE scalable vector type definitions.
//!
//! These types represent ARM SVE scalable vectors whose width is determined
//! at runtime by the hardware (128–2048 bits in 128-bit increments).
//!
//! Each type maps to an LLVM scalable vector IR type:
//!
//! | Rust type    | LLVM IR type              | ACLE C type   | Min elements |
//! |-------------|---------------------------|---------------|-------------|
//! | `svuint8_t`  | `<vscale x 16 x i8>`     | `svuint8_t`   | 16          |
//! | `svuint16_t` | `<vscale x 8 x i16>`     | `svuint16_t`  | 8           |
//! | `svuint32_t` | `<vscale x 4 x i32>`     | `svuint32_t`  | 4           |
//! | `svuint64_t` | `<vscale x 2 x i64>`     | `svuint64_t`  | 2           |
//! | `svint8_t`   | `<vscale x 16 x i8>`     | `svint8_t`    | 16          |
//! | `svint16_t`  | `<vscale x 8 x i16>`     | `svint16_t`   | 8           |
//! | `svint32_t`  | `<vscale x 4 x i32>`     | `svint32_t`   | 4           |
//! | `svint64_t`  | `<vscale x 2 x i64>`     | `svint64_t`   | 2           |
//!
//! The `vscale` factor is determined by hardware: a 128-bit SVE implementation
//! has `vscale = 1` (16 × u8 lanes), 256-bit has `vscale = 2` (32 × u8 lanes),
//! up to 2048-bit with `vscale = 16` (256 × u8 lanes).
//!
//! # Compiler support
//!
//! These types use `#[rustc_scalable_vector(N)]` (rust-lang/rust#145052),
//! available on nightly behind `#![feature(rustc_attrs)]`.

#![allow(non_camel_case_types)]

/// Scalable vector of unsigned 8-bit integers.
///
/// Maps to ACLE `svuint8_t` / LLVM `<vscale x 16 x i8>`.
/// At runtime, contains `16 * vscale` elements.
#[rustc_scalable_vector(16)]
#[derive(Copy, Clone)]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub struct svuint8_t(u8);

/// Scalable vector of unsigned 16-bit integers.
///
/// Maps to ACLE `svuint16_t` / LLVM `<vscale x 8 x i16>`.
/// At runtime, contains `8 * vscale` elements.
#[rustc_scalable_vector(8)]
#[derive(Copy, Clone)]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub struct svuint16_t(u16);

/// Scalable vector of unsigned 32-bit integers.
///
/// Maps to ACLE `svuint32_t` / LLVM `<vscale x 4 x i32>`.
/// At runtime, contains `4 * vscale` elements.
#[rustc_scalable_vector(4)]
#[derive(Copy, Clone)]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub struct svuint32_t(u32);

/// Scalable vector of unsigned 64-bit integers.
///
/// Maps to ACLE `svuint64_t` / LLVM `<vscale x 2 x i64>`.
/// At runtime, contains `2 * vscale` elements.
#[rustc_scalable_vector(2)]
#[derive(Copy, Clone)]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub struct svuint64_t(u64);

/// Scalable vector of signed 8-bit integers.
///
/// Maps to ACLE `svint8_t` / LLVM `<vscale x 16 x i8>`.
#[rustc_scalable_vector(16)]
#[derive(Copy, Clone)]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub struct svint8_t(i8);

/// Scalable vector of signed 16-bit integers.
///
/// Maps to ACLE `svint16_t` / LLVM `<vscale x 8 x i16>`.
#[rustc_scalable_vector(8)]
#[derive(Copy, Clone)]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub struct svint16_t(i16);

/// Scalable vector of signed 32-bit integers.
///
/// Maps to ACLE `svint32_t` / LLVM `<vscale x 4 x i32>`.
#[rustc_scalable_vector(4)]
#[derive(Copy, Clone)]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub struct svint32_t(i32);

/// Scalable vector of signed 64-bit integers.
///
/// Maps to ACLE `svint64_t` / LLVM `<vscale x 2 x i64>`.
#[rustc_scalable_vector(2)]
#[derive(Copy, Clone)]
#[unstable(feature = "stdarch_aarch64_sve", issue = "none")]
pub struct svint64_t(i64);
