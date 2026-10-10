#[cfg(test)]
use stdarch_test::assert_instr;

unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.riscv.xperm8.i32"]
    fn _xperm8(rs1: i32, rs2: i32) -> i32;

    #[link_name = "llvm.riscv.xperm4.i32"]
    fn _xperm4(rs1: i32, rs2: i32) -> i32;

    #[link_name = "llvm.riscv.orc.b.i32"]
    fn _orc_b(rs: i32) -> i32;

    #[link_name = "llvm.riscv.clmul.i32"]
    fn _clmul(rs1: i32, rs2: i32) -> i32;

    #[link_name = "llvm.riscv.clmulh.i32"]
    fn _clmulh(rs1: i32, rs2: i32) -> i32;

    #[link_name = "llvm.riscv.clmulr.i32"]
    fn _clmulr(rs1: i32, rs2: i32) -> i32;
}

/// Byte-wise lookup of indices into a vector in registers.
///
/// The xperm8 instruction operates on bytes. The rs1 register contains a vector of XLEN/8
/// 8-bit elements. The rs2 register contains a vector of XLEN/8 8-bit indexes. The result is
/// each element in rs2 replaced by the indexed element in rs1, or zero if the index into rs2
/// is out of bounds.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.47
#[stable(feature = "riscv_zb_intrinsics", since = "CURRENT_RUSTC_VERSION")]
#[target_feature(enable = "zbkx")]
#[cfg_attr(test, assert_instr(xperm8))]
#[inline]
pub fn xperm8(rs1: u32, rs2: u32) -> u32 {
    unsafe { _xperm8(rs1 as i32, rs2 as i32) as u32 }
}

/// Nibble-wise lookup of indices into a vector.
///
/// The xperm4 instruction operates on nibbles. The rs1 register contains a vector of XLEN/4
/// 4-bit elements. The rs2 register contains a vector of XLEN/4 4-bit indexes. The result is
/// each element in rs2 replaced by the indexed element in rs1, or zero if the index into rs2
/// is out of bounds.
///
/// Source: RISC-V Cryptography Extensions Volume I: Scalar & Entropy Source Instructions
///
/// Version: v1.0.1
///
/// Section: 3.48
#[stable(feature = "riscv_zb_intrinsics", since = "CURRENT_RUSTC_VERSION")]
#[target_feature(enable = "zbkx")]
#[cfg_attr(test, assert_instr(xperm4))]
#[inline]
pub fn xperm4(rs1: u32, rs2: u32) -> u32 {
    unsafe { _xperm4(rs1 as i32, rs2 as i32) as u32 }
}

/// Bitwise OR-Combine, byte granule
///
/// Combines the bits within every byte through a reciprocal bitwise logical OR. This sets the bits of each byte in
/// the result rd to all zeros if no bit within the respective byte of rs is set, or to all ones if any bit within the
/// respective byte of rs is set.
///
/// Source: RISC-V Bit-Manipulation ISA-extensions
///
/// Version: v1.0.0
///
/// Section: 2.24
#[stable(feature = "riscv_zb_intrinsics", since = "CURRENT_RUSTC_VERSION")]
#[target_feature(enable = "zbb")]
#[cfg_attr(test, assert_instr(orc.b))]
#[inline]
pub fn orc_b(rs: u32) -> u32 {
    unsafe { _orc_b(rs as i32) as u32 }
}

/// Carry-less multiply (low-part)
///
/// clmul produces the lower half of the 2·XLEN carry-less product.
///
/// Source: RISC-V Bit-Manipulation ISA-extensions
///
/// Version: v1.0.0
///
/// Section: 2.11
#[stable(feature = "riscv_zb_intrinsics", since = "CURRENT_RUSTC_VERSION")]
#[target_feature(enable = "zbkc")]
#[cfg_attr(test, assert_instr(clmul))]
#[inline]
pub fn clmul(rs1: u32, rs2: u32) -> u32 {
    unsafe { _clmul(rs1 as i32, rs2 as i32) as u32 }
}

/// Carry-less multiply (high-part)
///
/// clmulh produces the upper half of the 2·XLEN carry-less product.
///
/// Source: RISC-V Bit-Manipulation ISA-extensions
///
/// Version: v1.0.0
///
/// Section: 2.12
#[stable(feature = "riscv_zb_intrinsics", since = "CURRENT_RUSTC_VERSION")]
#[target_feature(enable = "zbkc")]
#[cfg_attr(test, assert_instr(clmulh))]
#[inline]
pub fn clmulh(rs1: u32, rs2: u32) -> u32 {
    unsafe { _clmulh(rs1 as i32, rs2 as i32) as u32 }
}

/// Carry-less multiply (reversed)
///
/// clmulr produces bits 2·XLEN−2:XLEN-1 of the 2·XLEN carry-less product.
///
/// Source: RISC-V Bit-Manipulation ISA-extensions
///
/// Version: v1.0.0
///
/// Section: 2.13
#[stable(feature = "riscv_zb_intrinsics", since = "CURRENT_RUSTC_VERSION")]
#[target_feature(enable = "zbc")]
#[cfg_attr(test, assert_instr(clmulr))]
#[inline]
pub fn clmulr(rs1: u32, rs2: u32) -> u32 {
    unsafe { _clmulr(rs1 as i32, rs2 as i32) as u32 }
}
