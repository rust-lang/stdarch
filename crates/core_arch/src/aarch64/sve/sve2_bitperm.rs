#![allow(improper_ctypes)]

//! SVE2 Bit Permutation intrinsics (FEAT_SVE_BitPerm).
//!
//! Three bit-permutation instructions operating on scalable vectors:
//!
//! | Instruction | ACLE           | x86 equivalent | Operation                                     |
//! |-------------|----------------|----------------|-----------------------------------------------|
//! | `BDEP`      | `svbdep[_u*]`  | `PDEP`         | Scatter source bits to mask-selected positions |
//! | `BEXT`      | `svbext[_u*]`  | `PEXT`         | Gather bits from mask-selected positions       |
//! | `BGRP`      | `svbgrp[_u*]`  | —              | Partition bits by mask into two groups          |
//!
//! All intrinsics require `#[target_feature(enable = "sve2-bitperm")]`.
//!
//! # References
//!
//! - [ARM Architecture Reference Manual — SVE2 BDEP, BEXT, BGRP](https://developer.arm.com/documentation/ddi0602/)
//! - [ARM ACLE for SVE2](https://arm-software.github.io/acle/main/acle-sve2.html)

#[cfg(test)]
use stdarch_test::assert_instr;

use super::types::*;

// ---------------------------------------------------------------------------
// LLVM intrinsic declarations
// ---------------------------------------------------------------------------
//
// The `.x.` suffix indicates unpredicated (don't-care predication) variants.
// SVE2 bitperm instructions do not use governing predicates.
//
// Naming: llvm.aarch64.sve.{bdep|bext|bgrp}.x.nxv{N}i{M}

// --- BDEP ---

unsafe extern "unadjusted" {
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.bdep.x.nxv16i8")]
    fn _svbdep_u8(op1: svuint8_t, op2: svuint8_t) -> svuint8_t;
}
unsafe extern "unadjusted" {
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.bdep.x.nxv8i16")]
    fn _svbdep_u16(op1: svuint16_t, op2: svuint16_t) -> svuint16_t;
}
unsafe extern "unadjusted" {
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.bdep.x.nxv4i32")]
    fn _svbdep_u32(op1: svuint32_t, op2: svuint32_t) -> svuint32_t;
}
unsafe extern "unadjusted" {
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.bdep.x.nxv2i64")]
    fn _svbdep_u64(op1: svuint64_t, op2: svuint64_t) -> svuint64_t;
}

// --- BEXT ---

unsafe extern "unadjusted" {
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.bext.x.nxv16i8")]
    fn _svbext_u8(op1: svuint8_t, op2: svuint8_t) -> svuint8_t;
}
unsafe extern "unadjusted" {
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.bext.x.nxv8i16")]
    fn _svbext_u16(op1: svuint16_t, op2: svuint16_t) -> svuint16_t;
}
unsafe extern "unadjusted" {
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.bext.x.nxv4i32")]
    fn _svbext_u32(op1: svuint32_t, op2: svuint32_t) -> svuint32_t;
}
unsafe extern "unadjusted" {
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.bext.x.nxv2i64")]
    fn _svbext_u64(op1: svuint64_t, op2: svuint64_t) -> svuint64_t;
}

// --- BGRP ---

unsafe extern "unadjusted" {
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.bgrp.x.nxv16i8")]
    fn _svbgrp_u8(op1: svuint8_t, op2: svuint8_t) -> svuint8_t;
}
unsafe extern "unadjusted" {
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.bgrp.x.nxv8i16")]
    fn _svbgrp_u16(op1: svuint16_t, op2: svuint16_t) -> svuint16_t;
}
unsafe extern "unadjusted" {
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.bgrp.x.nxv4i32")]
    fn _svbgrp_u32(op1: svuint32_t, op2: svuint32_t) -> svuint32_t;
}
unsafe extern "unadjusted" {
    #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.sve.bgrp.x.nxv2i64")]
    fn _svbgrp_u64(op1: svuint64_t, op2: svuint64_t) -> svuint64_t;
}

// ===========================================================================
// BDEP — Bit Deposit (SVE2 equivalent of x86 PDEP)
// ===========================================================================
//
// For each element, scatter consecutive low bits from `data` into the bit
// positions where `mask` has a 1. Remaining result bits are zero.

/// SVE2 bit deposit (8-bit elements).
///
/// For each 8-bit element, scatter consecutive low bits from `data` into
/// the bit positions selected by `mask`.
///
/// Equivalent to x86 BMI2 `PDEP` but operating on scalable vectors.
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbdep[_u8])
#[inline(always)]
#[target_feature(enable = "sve2-bitperm")]
#[cfg_attr(test, assert_instr(bdep))]
#[unstable(feature = "stdarch_aarch64_sve2_bitperm", issue = "none")]
pub fn svbdep_u8(data: svuint8_t, mask: svuint8_t) -> svuint8_t {
    unsafe { _svbdep_u8(data, mask) }
}

/// SVE2 bit deposit (16-bit elements).
///
/// For each 16-bit element, scatter consecutive low bits from `data` into
/// the bit positions selected by `mask`.
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbdep[_u16])
#[inline(always)]
#[target_feature(enable = "sve2-bitperm")]
#[cfg_attr(test, assert_instr(bdep))]
#[unstable(feature = "stdarch_aarch64_sve2_bitperm", issue = "none")]
pub fn svbdep_u16(data: svuint16_t, mask: svuint16_t) -> svuint16_t {
    unsafe { _svbdep_u16(data, mask) }
}

/// SVE2 bit deposit (32-bit elements).
///
/// For each 32-bit element, scatter consecutive low bits from `data` into
/// the bit positions selected by `mask`.
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbdep[_u32])
#[inline(always)]
#[target_feature(enable = "sve2-bitperm")]
#[cfg_attr(test, assert_instr(bdep))]
#[unstable(feature = "stdarch_aarch64_sve2_bitperm", issue = "none")]
pub fn svbdep_u32(data: svuint32_t, mask: svuint32_t) -> svuint32_t {
    unsafe { _svbdep_u32(data, mask) }
}

/// SVE2 bit deposit (64-bit elements).
///
/// For each 64-bit element, scatter consecutive low bits from `data` into
/// the bit positions selected by `mask`.
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbdep[_u64])
#[inline(always)]
#[target_feature(enable = "sve2-bitperm")]
#[cfg_attr(test, assert_instr(bdep))]
#[unstable(feature = "stdarch_aarch64_sve2_bitperm", issue = "none")]
pub fn svbdep_u64(data: svuint64_t, mask: svuint64_t) -> svuint64_t {
    unsafe { _svbdep_u64(data, mask) }
}

// ===========================================================================
// BEXT — Bit Extract (SVE2 equivalent of x86 PEXT)
// ===========================================================================
//
// For each element, gather bits from `data` at positions where `mask` has
// a 1 and pack them into consecutive low bits of the result.

/// SVE2 bit extract (8-bit elements).
///
/// For each 8-bit element, gather bits from `data` at the positions
/// selected by `mask` and pack them into consecutive low bits.
///
/// Equivalent to x86 BMI2 `PEXT` but operating on scalable vectors.
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbext[_u8])
#[inline(always)]
#[target_feature(enable = "sve2-bitperm")]
#[cfg_attr(test, assert_instr(bext))]
#[unstable(feature = "stdarch_aarch64_sve2_bitperm", issue = "none")]
pub fn svbext_u8(data: svuint8_t, mask: svuint8_t) -> svuint8_t {
    unsafe { _svbext_u8(data, mask) }
}

/// SVE2 bit extract (16-bit elements).
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbext[_u16])
#[inline(always)]
#[target_feature(enable = "sve2-bitperm")]
#[cfg_attr(test, assert_instr(bext))]
#[unstable(feature = "stdarch_aarch64_sve2_bitperm", issue = "none")]
pub fn svbext_u16(data: svuint16_t, mask: svuint16_t) -> svuint16_t {
    unsafe { _svbext_u16(data, mask) }
}

/// SVE2 bit extract (32-bit elements).
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbext[_u32])
#[inline(always)]
#[target_feature(enable = "sve2-bitperm")]
#[cfg_attr(test, assert_instr(bext))]
#[unstable(feature = "stdarch_aarch64_sve2_bitperm", issue = "none")]
pub fn svbext_u32(data: svuint32_t, mask: svuint32_t) -> svuint32_t {
    unsafe { _svbext_u32(data, mask) }
}

/// SVE2 bit extract (64-bit elements).
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbext[_u64])
#[inline(always)]
#[target_feature(enable = "sve2-bitperm")]
#[cfg_attr(test, assert_instr(bext))]
#[unstable(feature = "stdarch_aarch64_sve2_bitperm", issue = "none")]
pub fn svbext_u64(data: svuint64_t, mask: svuint64_t) -> svuint64_t {
    unsafe { _svbext_u64(data, mask) }
}

// ===========================================================================
// BGRP — Bit Group (no x86 equivalent)
// ===========================================================================
//
// For each element, partition bits of `data` into two groups by `mask`:
// - Bits where mask=1 are packed into the low portion of the result
// - Bits where mask=0 are packed into the high portion
//
// Formally: result = BEXT(data, mask) | (BEXT(data, ~mask) << popcount(mask))

/// SVE2 bit group (8-bit elements).
///
/// For each 8-bit element, partition the bits of `data` into two groups
/// based on `mask`: bits at mask-selected positions are packed into the
/// low bits, remaining bits are packed into the high bits.
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbgrp[_u8])
#[inline(always)]
#[target_feature(enable = "sve2-bitperm")]
#[cfg_attr(test, assert_instr(bgrp))]
#[unstable(feature = "stdarch_aarch64_sve2_bitperm", issue = "none")]
pub fn svbgrp_u8(data: svuint8_t, mask: svuint8_t) -> svuint8_t {
    unsafe { _svbgrp_u8(data, mask) }
}

/// SVE2 bit group (16-bit elements).
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbgrp[_u16])
#[inline(always)]
#[target_feature(enable = "sve2-bitperm")]
#[cfg_attr(test, assert_instr(bgrp))]
#[unstable(feature = "stdarch_aarch64_sve2_bitperm", issue = "none")]
pub fn svbgrp_u16(data: svuint16_t, mask: svuint16_t) -> svuint16_t {
    unsafe { _svbgrp_u16(data, mask) }
}

/// SVE2 bit group (32-bit elements).
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbgrp[_u32])
#[inline(always)]
#[target_feature(enable = "sve2-bitperm")]
#[cfg_attr(test, assert_instr(bgrp))]
#[unstable(feature = "stdarch_aarch64_sve2_bitperm", issue = "none")]
pub fn svbgrp_u32(data: svuint32_t, mask: svuint32_t) -> svuint32_t {
    unsafe { _svbgrp_u32(data, mask) }
}

/// SVE2 bit group (64-bit elements).
///
/// [Arm's documentation](https://developer.arm.com/architectures/instruction-sets/intrinsics/svbgrp[_u64])
#[inline(always)]
#[target_feature(enable = "sve2-bitperm")]
#[cfg_attr(test, assert_instr(bgrp))]
#[unstable(feature = "stdarch_aarch64_sve2_bitperm", issue = "none")]
pub fn svbgrp_u64(data: svuint64_t, mask: svuint64_t) -> svuint64_t {
    unsafe { _svbgrp_u64(data, mask) }
}

// ===========================================================================
// Scalar broadcast variants (_n_ suffix)
// ===========================================================================
//
// ACLE defines `svbdep_n_u*` / `svbext_n_u*` / `svbgrp_n_u*` where the
// second operand is a scalar broadcast to all lanes. These depend on
// `svdup_n_u*` (SVE DUP instruction) which will be provided by the base
// SVE intrinsics module. They are intentionally deferred to that PR to
// keep this contribution minimal and self-contained.
//
// TODO: Add _n_ variants once base SVE1 intrinsics (svdup_n_u*) land.

// ===========================================================================
// Tests — Scalar reference implementations
// ===========================================================================

#[cfg(test)]
mod tests {
    // Scalar reference implementations for correctness verification.
    // These run on any host (no SVE hardware needed).
    //
    // Integration tests that call the actual SVE2 intrinsics require either:
    // - Hardware with SVE2 + BitPerm (e.g., AWS Graviton 3)
    // - QEMU user-mode: qemu-aarch64 -cpu max ./test_binary

    /// Scalar BDEP: scatter consecutive low bits of `data` to mask-selected positions.
    fn scalar_bdep(data: u64, mask: u64) -> u64 {
        let mut result: u64 = 0;
        let mut m = mask;
        let mut src_bit: u32 = 0;
        while m != 0 {
            let pos = m.trailing_zeros();
            if (data >> src_bit) & 1 != 0 {
                result |= 1u64 << pos;
            }
            m &= m - 1; // clear lowest set bit
            src_bit += 1;
        }
        result
    }

    /// Scalar BEXT: gather bits from mask-selected positions into consecutive low bits.
    fn scalar_bext(data: u64, mask: u64) -> u64 {
        let mut result: u64 = 0;
        let mut m = mask;
        let mut dst_bit: u32 = 0;
        while m != 0 {
            let pos = m.trailing_zeros();
            if (data >> pos) & 1 != 0 {
                result |= 1u64 << dst_bit;
            }
            m &= m - 1;
            dst_bit += 1;
        }
        result
    }

    /// Scalar BGRP: partition bits by mask — selected bits low, unselected bits high.
    fn scalar_bgrp(data: u64, mask: u64) -> u64 {
        let selected = scalar_bext(data, mask);
        let not_selected = scalar_bext(data, !mask);
        let shift = mask.count_ones();
        if shift >= 64 {
            selected
        } else {
            selected | (not_selected << shift)
        }
    }

    // --- Correctness of reference implementations ---

    #[test]
    fn test_bdep_bext_inverse_property() {
        // Key property: BEXT(BDEP(x, m), m) == x & ((1 << popcount(m)) - 1)
        let data: u64 = 0b1011;
        let mask: u64 = 0b1010_1010;

        let deposited = scalar_bdep(data, mask);
        // bits 1,0,1,1 deposited at positions 1,3,5,7
        // data bits 1,1,0,1 deposited at mask positions 1,3,5,7
        // = (1<<1) | (1<<3) | (0<<5) | (1<<7) = 2+8+128 = 0b1000_1010
        assert_eq!(deposited, 0b1000_1010);

        let extracted = scalar_bext(deposited, mask);
        let significant = (1u64 << mask.count_ones()) - 1;
        assert_eq!(extracted, data & significant);
    }

    #[test]
    fn test_bdep_bext_roundtrip_exhaustive_u8() {
        // Exhaustive roundtrip for all 8-bit values
        for mask in 0u8..=255 {
            let popcount = mask.count_ones();
            let max_data = if popcount >= 8 {
                255u64
            } else {
                (1u64 << popcount) - 1
            };
            for data in 0..=max_data.min(255) {
                let deposited = scalar_bdep(data, mask as u64) as u8;
                let extracted = scalar_bext(deposited as u64, mask as u64) as u8;
                assert_eq!(
                    extracted,
                    (data as u8) & (max_data as u8),
                    "roundtrip failed: data={data:#010b}, mask={mask:#010b}"
                );
            }
        }
    }

    #[test]
    fn test_bgrp_partitions_correctly() {
        let data: u64 = 0b1100_1010;
        let mask: u64 = 0b1111_0000;

        let grouped = scalar_bgrp(data, mask);
        // mask=1 positions (bits 7-4) of data: 0b1100 → packed low
        // mask=0 positions (bits 3-0) of data: 0b1010 → packed high
        // result = 0b1100 | (0b1010 << 4) = 0b1010_1100
        assert_eq!(grouped, 0b1010_1100);
    }

    #[test]
    fn test_bdep_zero_mask() {
        assert_eq!(scalar_bdep(0xFF, 0x00), 0);
    }

    #[test]
    fn test_bext_zero_mask() {
        assert_eq!(scalar_bext(0xFF, 0x00), 0);
    }

    #[test]
    fn test_bdep_full_mask() {
        assert_eq!(scalar_bdep(0xAB, 0xFF), 0xAB);
    }

    #[test]
    fn test_bext_full_mask() {
        assert_eq!(scalar_bext(0xAB, 0xFF), 0xAB);
    }

    #[test]
    fn test_bgrp_full_mask() {
        // All bits selected → identity
        assert_eq!(scalar_bgrp(0xAB, 0xFF) as u8, 0xAB);
    }

    #[test]
    fn test_bgrp_zero_mask() {
        // No bits selected → all bits shift up by 0 → data unchanged (moved to "high" group)
        let data: u64 = 0xAB;
        let mask: u64 = 0x00;
        // popcount(0) = 0, so not_selected << 0 = data, selected = 0
        assert_eq!(scalar_bgrp(data, mask) as u8, data as u8);
    }
}
