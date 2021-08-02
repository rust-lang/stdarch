// This code is automatically generated. DO NOT MODIFY.
//
// Instead, modify `crates/stdarch-gen/neon.spec` and run the following command to re-generate this file:
//
// ```
// OUT_DIR=`pwd`/crates/core_arch cargo run -p stdarch-gen -- crates/stdarch-gen/neon.spec
// ```
use super::*;
#[cfg(test)]
use stdarch_test::assert_instr;

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vand_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_and(a, b)
}

/// Vector bitwise and
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vand))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(and))]
pub unsafe fn vandq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_and(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorr_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_or(a, b)
}

/// Vector bitwise or (immediate, inclusive)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vorr))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(orr))]
pub unsafe fn vorrq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_or(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veor_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_xor(a, b)
}

/// Vector bitwise exclusive or (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(veor))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(eor))]
pub unsafe fn veorq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_xor(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sabd))]
pub unsafe fn vabd_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabds.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sabd.v8i8")]
        fn vabd_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vabd_s8_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sabd))]
pub unsafe fn vabdq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabds.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sabd.v16i8")]
        fn vabdq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vabdq_s8_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sabd))]
pub unsafe fn vabd_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabds.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sabd.v4i16")]
        fn vabd_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vabd_s16_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sabd))]
pub unsafe fn vabdq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabds.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sabd.v8i16")]
        fn vabdq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vabdq_s16_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sabd))]
pub unsafe fn vabd_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabds.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sabd.v2i32")]
        fn vabd_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vabd_s32_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sabd))]
pub unsafe fn vabdq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabds.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sabd.v4i32")]
        fn vabdq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vabdq_s32_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uabd))]
pub unsafe fn vabd_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabdu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uabd.v8i8")]
        fn vabd_u8_(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    }
vabd_u8_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uabd))]
pub unsafe fn vabdq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabdu.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uabd.v16i8")]
        fn vabdq_u8_(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
    }
vabdq_u8_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uabd))]
pub unsafe fn vabd_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabdu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uabd.v4i16")]
        fn vabd_u16_(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    }
vabd_u16_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uabd))]
pub unsafe fn vabdq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabdu.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uabd.v8i16")]
        fn vabdq_u16_(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t;
    }
vabdq_u16_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uabd))]
pub unsafe fn vabd_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabdu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uabd.v2i32")]
        fn vabd_u32_(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    }
vabd_u32_(a, b)
}

/// Absolute difference between the arguments
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uabd))]
pub unsafe fn vabdq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabdu.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uabd.v4i32")]
        fn vabdq_u32_(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    }
vabdq_u32_(a, b)
}

/// Absolute difference between the arguments of Floating
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fabd))]
pub unsafe fn vabd_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabds.v2f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fabd.v2f32")]
        fn vabd_f32_(a: float32x2_t, b: float32x2_t) -> float32x2_t;
    }
vabd_f32_(a, b)
}

/// Absolute difference between the arguments of Floating
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabd.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fabd))]
pub unsafe fn vabdq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vabds.v4f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fabd.v4f32")]
        fn vabdq_f32_(a: float32x4_t, b: float32x4_t) -> float32x4_t;
    }
vabdq_f32_(a, b)
}

/// Unsigned Absolute difference Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabdl.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uabdl))]
pub unsafe fn vabdl_u8(a: uint8x8_t, b: uint8x8_t) -> uint16x8_t {
    simd_cast(vabd_u8(a, b))
}

/// Unsigned Absolute difference Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabdl.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uabdl))]
pub unsafe fn vabdl_u16(a: uint16x4_t, b: uint16x4_t) -> uint32x4_t {
    simd_cast(vabd_u16(a, b))
}

/// Unsigned Absolute difference Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabdl.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uabdl))]
pub unsafe fn vabdl_u32(a: uint32x2_t, b: uint32x2_t) -> uint64x2_t {
    simd_cast(vabd_u32(a, b))
}

/// Signed Absolute difference Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabdl.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sabdl))]
pub unsafe fn vabdl_s8(a: int8x8_t, b: int8x8_t) -> int16x8_t {
    let c: uint8x8_t = simd_cast(vabd_s8(a, b));
    simd_cast(c)
}

/// Signed Absolute difference Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabdl.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sabdl))]
pub unsafe fn vabdl_s16(a: int16x4_t, b: int16x4_t) -> int32x4_t {
    let c: uint16x4_t = simd_cast(vabd_s16(a, b));
    simd_cast(c)
}

/// Signed Absolute difference Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabdl.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sabdl))]
pub unsafe fn vabdl_s32(a: int32x2_t, b: int32x2_t) -> int64x2_t {
    let c: uint32x2_t = simd_cast(vabd_s32(a, b));
    simd_cast(c)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_s8(a: int8x8_t, b: int8x8_t) -> uint8x8_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_s16(a: int16x4_t, b: int16x4_t) -> uint16x4_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_s32(a: int32x2_t, b: int32x2_t) -> uint32x2_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceq_p8(a: poly8x8_t, b: poly8x8_t) -> uint8x8_t {
    simd_eq(a, b)
}

/// Compare bitwise Equal (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmeq))]
pub unsafe fn vceqq_p8(a: poly8x16_t, b: poly8x16_t) -> uint8x16_t {
    simd_eq(a, b)
}

/// Floating-point compare equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmeq))]
pub unsafe fn vceq_f32(a: float32x2_t, b: float32x2_t) -> uint32x2_t {
    simd_eq(a, b)
}

/// Floating-point compare equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vceq.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmeq))]
pub unsafe fn vceqq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    simd_eq(a, b)
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtst_s8(a: int8x8_t, b: int8x8_t) -> uint8x8_t {
    let c: int8x8_t = simd_and(a, b);
    let d: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtstq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    let c: int8x16_t = simd_and(a, b);
    let d: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtst_s16(a: int16x4_t, b: int16x4_t) -> uint16x4_t {
    let c: int16x4_t = simd_and(a, b);
    let d: i16x4 = i16x4::new(0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtstq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    let c: int16x8_t = simd_and(a, b);
    let d: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtst_s32(a: int32x2_t, b: int32x2_t) -> uint32x2_t {
    let c: int32x2_t = simd_and(a, b);
    let d: i32x2 = i32x2::new(0, 0);
    simd_ne(c, transmute(d))
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtstq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    let c: int32x4_t = simd_and(a, b);
    let d: i32x4 = i32x4::new(0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtst_p8(a: poly8x8_t, b: poly8x8_t) -> uint8x8_t {
    let c: poly8x8_t = simd_and(a, b);
    let d: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Signed compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtstq_p8(a: poly8x16_t, b: poly8x16_t) -> uint8x16_t {
    let c: poly8x16_t = simd_and(a, b);
    let d: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Unsigned compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtst_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    let c: uint8x8_t = simd_and(a, b);
    let d: u8x8 = u8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Unsigned compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtstq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    let c: uint8x16_t = simd_and(a, b);
    let d: u8x16 = u8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Unsigned compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtst_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    let c: uint16x4_t = simd_and(a, b);
    let d: u16x4 = u16x4::new(0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Unsigned compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtstq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    let c: uint16x8_t = simd_and(a, b);
    let d: u16x8 = u16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Unsigned compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtst_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    let c: uint32x2_t = simd_and(a, b);
    let d: u32x2 = u32x2::new(0, 0);
    simd_ne(c, transmute(d))
}

/// Unsigned compare bitwise Test bits nonzero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vtst))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmtst))]
pub unsafe fn vtstq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    let c: uint32x4_t = simd_and(a, b);
    let d: u32x4 = u32x4::new(0, 0, 0, 0);
    simd_ne(c, transmute(d))
}

/// Floating-point absolute value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vabs))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fabs))]
pub unsafe fn vabs_f32(a: float32x2_t) -> float32x2_t {
    simd_fabs(a)
}

/// Floating-point absolute value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vabs))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fabs))]
pub unsafe fn vabsq_f32(a: float32x4_t) -> float32x4_t {
    simd_fabs(a)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcgt_s8(a: int8x8_t, b: int8x8_t) -> uint8x8_t {
    simd_gt(a, b)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcgtq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    simd_gt(a, b)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcgt_s16(a: int16x4_t, b: int16x4_t) -> uint16x4_t {
    simd_gt(a, b)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcgtq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    simd_gt(a, b)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcgt_s32(a: int32x2_t, b: int32x2_t) -> uint32x2_t {
    simd_gt(a, b)
}

/// Compare signed greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcgtq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcgt_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcgtq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcgt_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcgtq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcgt_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_gt(a, b)
}

/// Compare unsigned highe
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcgtq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_gt(a, b)
}

/// Floating-point compare greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmgt))]
pub unsafe fn vcgt_f32(a: float32x2_t, b: float32x2_t) -> uint32x2_t {
    simd_gt(a, b)
}

/// Floating-point compare greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmgt))]
pub unsafe fn vcgtq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    simd_gt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vclt_s8(a: int8x8_t, b: int8x8_t) -> uint8x8_t {
    simd_lt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcltq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    simd_lt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vclt_s16(a: int16x4_t, b: int16x4_t) -> uint16x4_t {
    simd_lt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcltq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    simd_lt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vclt_s32(a: int32x2_t, b: int32x2_t) -> uint32x2_t {
    simd_lt(a, b)
}

/// Compare signed less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmgt))]
pub unsafe fn vcltq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vclt_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcltq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vclt_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcltq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vclt_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_lt(a, b)
}

/// Compare unsigned less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhi))]
pub unsafe fn vcltq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_lt(a, b)
}

/// Floating-point compare less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmgt))]
pub unsafe fn vclt_f32(a: float32x2_t, b: float32x2_t) -> uint32x2_t {
    simd_lt(a, b)
}

/// Floating-point compare less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcgt.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmgt))]
pub unsafe fn vcltq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    simd_lt(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcle_s8(a: int8x8_t, b: int8x8_t) -> uint8x8_t {
    simd_le(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcleq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    simd_le(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcle_s16(a: int16x4_t, b: int16x4_t) -> uint16x4_t {
    simd_le(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcleq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    simd_le(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcle_s32(a: int32x2_t, b: int32x2_t) -> uint32x2_t {
    simd_le(a, b)
}

/// Compare signed less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcleq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcle_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcleq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcle_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcleq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcle_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_le(a, b)
}

/// Compare unsigned less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcleq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_le(a, b)
}

/// Floating-point compare less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmge))]
pub unsafe fn vcle_f32(a: float32x2_t, b: float32x2_t) -> uint32x2_t {
    simd_le(a, b)
}

/// Floating-point compare less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmge))]
pub unsafe fn vcleq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    simd_le(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcge_s8(a: int8x8_t, b: int8x8_t) -> uint8x8_t {
    simd_ge(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcgeq_s8(a: int8x16_t, b: int8x16_t) -> uint8x16_t {
    simd_ge(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcge_s16(a: int16x4_t, b: int16x4_t) -> uint16x4_t {
    simd_ge(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcgeq_s16(a: int16x8_t, b: int16x8_t) -> uint16x8_t {
    simd_ge(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcge_s32(a: int32x2_t, b: int32x2_t) -> uint32x2_t {
    simd_ge(a, b)
}

/// Compare signed greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmge))]
pub unsafe fn vcgeq_s32(a: int32x4_t, b: int32x4_t) -> uint32x4_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcge_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcgeq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcge_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcgeq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcge_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_ge(a, b)
}

/// Compare unsigned greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cmhs))]
pub unsafe fn vcgeq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_ge(a, b)
}

/// Floating-point compare greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmge))]
pub unsafe fn vcge_f32(a: float32x2_t, b: float32x2_t) -> uint32x2_t {
    simd_ge(a, b)
}

/// Floating-point compare greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcge.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcmge))]
pub unsafe fn vcgeq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    simd_ge(a, b)
}

/// Count leading sign bits
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcls.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cls))]
pub unsafe fn vcls_s8(a: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vcls.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.cls.v8i8")]
        fn vcls_s8_(a: int8x8_t) -> int8x8_t;
    }
vcls_s8_(a)
}

/// Count leading sign bits
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcls.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cls))]
pub unsafe fn vclsq_s8(a: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vcls.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.cls.v16i8")]
        fn vclsq_s8_(a: int8x16_t) -> int8x16_t;
    }
vclsq_s8_(a)
}

/// Count leading sign bits
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcls.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cls))]
pub unsafe fn vcls_s16(a: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vcls.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.cls.v4i16")]
        fn vcls_s16_(a: int16x4_t) -> int16x4_t;
    }
vcls_s16_(a)
}

/// Count leading sign bits
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcls.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cls))]
pub unsafe fn vclsq_s16(a: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vcls.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.cls.v8i16")]
        fn vclsq_s16_(a: int16x8_t) -> int16x8_t;
    }
vclsq_s16_(a)
}

/// Count leading sign bits
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcls.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cls))]
pub unsafe fn vcls_s32(a: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vcls.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.cls.v2i32")]
        fn vcls_s32_(a: int32x2_t) -> int32x2_t;
    }
vcls_s32_(a)
}

/// Count leading sign bits
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vcls.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(cls))]
pub unsafe fn vclsq_s32(a: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vcls.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.cls.v4i32")]
        fn vclsq_s32_(a: int32x4_t) -> int32x4_t;
    }
vclsq_s32_(a)
}

/// Signed count leading sign bits
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vclz.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(clz))]
pub unsafe fn vclz_s8(a: int8x8_t) -> int8x8_t {
    vclz_s8_(a)
}

/// Signed count leading sign bits
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vclz.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(clz))]
pub unsafe fn vclzq_s8(a: int8x16_t) -> int8x16_t {
    vclzq_s8_(a)
}

/// Signed count leading sign bits
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vclz.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(clz))]
pub unsafe fn vclz_s16(a: int16x4_t) -> int16x4_t {
    vclz_s16_(a)
}

/// Signed count leading sign bits
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vclz.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(clz))]
pub unsafe fn vclzq_s16(a: int16x8_t) -> int16x8_t {
    vclzq_s16_(a)
}

/// Signed count leading sign bits
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vclz.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(clz))]
pub unsafe fn vclz_s32(a: int32x2_t) -> int32x2_t {
    vclz_s32_(a)
}

/// Signed count leading sign bits
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vclz.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(clz))]
pub unsafe fn vclzq_s32(a: int32x4_t) -> int32x4_t {
    vclzq_s32_(a)
}

/// Unsigned count leading sign bits
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vclz.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(clz))]
pub unsafe fn vclz_u8(a: uint8x8_t) -> uint8x8_t {
    transmute(vclz_s8_(transmute(a)))
}

/// Unsigned count leading sign bits
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vclz.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(clz))]
pub unsafe fn vclzq_u8(a: uint8x16_t) -> uint8x16_t {
    transmute(vclzq_s8_(transmute(a)))
}

/// Unsigned count leading sign bits
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vclz.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(clz))]
pub unsafe fn vclz_u16(a: uint16x4_t) -> uint16x4_t {
    transmute(vclz_s16_(transmute(a)))
}

/// Unsigned count leading sign bits
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vclz.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(clz))]
pub unsafe fn vclzq_u16(a: uint16x8_t) -> uint16x8_t {
    transmute(vclzq_s16_(transmute(a)))
}

/// Unsigned count leading sign bits
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vclz.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(clz))]
pub unsafe fn vclz_u32(a: uint32x2_t) -> uint32x2_t {
    transmute(vclz_s32_(transmute(a)))
}

/// Unsigned count leading sign bits
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vclz.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(clz))]
pub unsafe fn vclzq_u32(a: uint32x4_t) -> uint32x4_t {
    transmute(vclzq_s32_(transmute(a)))
}

/// Floating-point absolute compare greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vacgt.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(facgt))]
pub unsafe fn vcagt_f32(a: float32x2_t, b: float32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vacgt.v2i32.v2f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.facgt.v2i32.v2f32")]
        fn vcagt_f32_(a: float32x2_t, b: float32x2_t) -> uint32x2_t;
    }
vcagt_f32_(a, b)
}

/// Floating-point absolute compare greater than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vacgt.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(facgt))]
pub unsafe fn vcagtq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vacgt.v4i32.v4f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.facgt.v4i32.v4f32")]
        fn vcagtq_f32_(a: float32x4_t, b: float32x4_t) -> uint32x4_t;
    }
vcagtq_f32_(a, b)
}

/// Floating-point absolute compare greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vacge.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(facge))]
pub unsafe fn vcage_f32(a: float32x2_t, b: float32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vacge.v2i32.v2f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.facge.v2i32.v2f32")]
        fn vcage_f32_(a: float32x2_t, b: float32x2_t) -> uint32x2_t;
    }
vcage_f32_(a, b)
}

/// Floating-point absolute compare greater than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vacge.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(facge))]
pub unsafe fn vcageq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vacge.v4i32.v4f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.facge.v4i32.v4f32")]
        fn vcageq_f32_(a: float32x4_t, b: float32x4_t) -> uint32x4_t;
    }
vcageq_f32_(a, b)
}

/// Floating-point absolute compare less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vacgt.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(facgt))]
pub unsafe fn vcalt_f32(a: float32x2_t, b: float32x2_t) -> uint32x2_t {
    vcagt_f32(b, a)
}

/// Floating-point absolute compare less than
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vacgt.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(facgt))]
pub unsafe fn vcaltq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    vcagtq_f32(b, a)
}

/// Floating-point absolute compare less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vacge.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(facge))]
pub unsafe fn vcale_f32(a: float32x2_t, b: float32x2_t) -> uint32x2_t {
    vcage_f32(b, a)
}

/// Floating-point absolute compare less than or equal
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vacge.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(facge))]
pub unsafe fn vcaleq_f32(a: float32x4_t, b: float32x4_t) -> uint32x4_t {
    vcageq_f32(b, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop))]
pub unsafe fn vcreate_s8(a: u64) -> int8x8_t {
    transmute(a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop))]
pub unsafe fn vcreate_s32(a: u64) -> int32x2_t {
    transmute(a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop))]
pub unsafe fn vcreate_s64(a: u64) -> int64x1_t {
    transmute(a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop))]
pub unsafe fn vcreate_u8(a: u64) -> uint8x8_t {
    transmute(a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop))]
pub unsafe fn vcreate_u32(a: u64) -> uint32x2_t {
    transmute(a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop))]
pub unsafe fn vcreate_u64(a: u64) -> uint64x1_t {
    transmute(a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop))]
pub unsafe fn vcreate_p8(a: u64) -> poly8x8_t {
    transmute(a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop))]
pub unsafe fn vcreate_p16(a: u64) -> poly16x4_t {
    transmute(a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "crypto,v8"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop))]
pub unsafe fn vcreate_p64(a: u64) -> poly64x1_t {
    transmute(a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop))]
pub unsafe fn vcreate_f32(a: u64) -> float32x2_t {
    transmute(a)
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vcvt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(scvtf))]
pub unsafe fn vcvt_f32_s32(a: int32x2_t) -> float32x2_t {
    simd_cast(a)
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vcvt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(scvtf))]
pub unsafe fn vcvtq_f32_s32(a: int32x4_t) -> float32x4_t {
    simd_cast(a)
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vcvt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ucvtf))]
pub unsafe fn vcvt_f32_u32(a: uint32x2_t) -> float32x2_t {
    simd_cast(a)
}

/// Fixed-point convert to floating-point
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vcvt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ucvtf))]
pub unsafe fn vcvtq_f32_u32(a: uint32x4_t) -> float32x4_t {
    simd_cast(a)
}

/// Fixed-point convert to floating-point
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vcvt, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvt_n_f32_s32<const N: i32>(a: int32x2_t) -> float32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vcvtfxs2fp.v2f32.v2i32")]
        fn vcvt_n_f32_s32_(a: int32x2_t, n: i32) -> float32x2_t;
    }
vcvt_n_f32_s32_(a, N)
}

/// Fixed-point convert to floating-point
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(scvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvt_n_f32_s32<const N: i32>(a: int32x2_t) -> float32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfxs2fp.v2f32.v2i32")]
        fn vcvt_n_f32_s32_(a: int32x2_t, n: i32) -> float32x2_t;
    }
vcvt_n_f32_s32_(a, N)
}

/// Fixed-point convert to floating-point
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vcvt, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvtq_n_f32_s32<const N: i32>(a: int32x4_t) -> float32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vcvtfxs2fp.v4f32.v4i32")]
        fn vcvtq_n_f32_s32_(a: int32x4_t, n: i32) -> float32x4_t;
    }
vcvtq_n_f32_s32_(a, N)
}

/// Fixed-point convert to floating-point
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(scvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvtq_n_f32_s32<const N: i32>(a: int32x4_t) -> float32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfxs2fp.v4f32.v4i32")]
        fn vcvtq_n_f32_s32_(a: int32x4_t, n: i32) -> float32x4_t;
    }
vcvtq_n_f32_s32_(a, N)
}

/// Fixed-point convert to floating-point
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vcvt, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvt_n_f32_u32<const N: i32>(a: uint32x2_t) -> float32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vcvtfxu2fp.v2f32.v2i32")]
        fn vcvt_n_f32_u32_(a: uint32x2_t, n: i32) -> float32x2_t;
    }
vcvt_n_f32_u32_(a, N)
}

/// Fixed-point convert to floating-point
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ucvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvt_n_f32_u32<const N: i32>(a: uint32x2_t) -> float32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfxu2fp.v2f32.v2i32")]
        fn vcvt_n_f32_u32_(a: uint32x2_t, n: i32) -> float32x2_t;
    }
vcvt_n_f32_u32_(a, N)
}

/// Fixed-point convert to floating-point
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vcvt, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvtq_n_f32_u32<const N: i32>(a: uint32x4_t) -> float32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vcvtfxu2fp.v4f32.v4i32")]
        fn vcvtq_n_f32_u32_(a: uint32x4_t, n: i32) -> float32x4_t;
    }
vcvtq_n_f32_u32_(a, N)
}

/// Fixed-point convert to floating-point
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ucvtf, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvtq_n_f32_u32<const N: i32>(a: uint32x4_t) -> float32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfxu2fp.v4f32.v4i32")]
        fn vcvtq_n_f32_u32_(a: uint32x4_t, n: i32) -> float32x4_t;
    }
vcvtq_n_f32_u32_(a, N)
}

/// Floating-point convert to fixed-point, rounding toward zero
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vcvt, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvt_n_s32_f32<const N: i32>(a: float32x2_t) -> int32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vcvtfp2fxs.v2i32.v2f32")]
        fn vcvt_n_s32_f32_(a: float32x2_t, n: i32) -> int32x2_t;
    }
vcvt_n_s32_f32_(a, N)
}

/// Floating-point convert to fixed-point, rounding toward zero
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcvtzs, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvt_n_s32_f32<const N: i32>(a: float32x2_t) -> int32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfp2fxs.v2i32.v2f32")]
        fn vcvt_n_s32_f32_(a: float32x2_t, n: i32) -> int32x2_t;
    }
vcvt_n_s32_f32_(a, N)
}

/// Floating-point convert to fixed-point, rounding toward zero
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vcvt, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvtq_n_s32_f32<const N: i32>(a: float32x4_t) -> int32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vcvtfp2fxs.v4i32.v4f32")]
        fn vcvtq_n_s32_f32_(a: float32x4_t, n: i32) -> int32x4_t;
    }
vcvtq_n_s32_f32_(a, N)
}

/// Floating-point convert to fixed-point, rounding toward zero
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcvtzs, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvtq_n_s32_f32<const N: i32>(a: float32x4_t) -> int32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfp2fxs.v4i32.v4f32")]
        fn vcvtq_n_s32_f32_(a: float32x4_t, n: i32) -> int32x4_t;
    }
vcvtq_n_s32_f32_(a, N)
}

/// Floating-point convert to fixed-point, rounding toward zero
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vcvt, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvt_n_u32_f32<const N: i32>(a: float32x2_t) -> uint32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vcvtfp2fxu.v2i32.v2f32")]
        fn vcvt_n_u32_f32_(a: float32x2_t, n: i32) -> uint32x2_t;
    }
vcvt_n_u32_f32_(a, N)
}

/// Floating-point convert to fixed-point, rounding toward zero
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcvtzu, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvt_n_u32_f32<const N: i32>(a: float32x2_t) -> uint32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfp2fxu.v2i32.v2f32")]
        fn vcvt_n_u32_f32_(a: float32x2_t, n: i32) -> uint32x2_t;
    }
vcvt_n_u32_f32_(a, N)
}

/// Floating-point convert to fixed-point, rounding toward zero
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vcvt, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvtq_n_u32_f32<const N: i32>(a: float32x4_t) -> uint32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vcvtfp2fxu.v4i32.v4f32")]
        fn vcvtq_n_u32_f32_(a: float32x4_t, n: i32) -> uint32x4_t;
    }
vcvtq_n_u32_f32_(a, N)
}

/// Floating-point convert to fixed-point, rounding toward zero
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcvtzu, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vcvtq_n_u32_f32<const N: i32>(a: float32x4_t) -> uint32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.vcvtfp2fxu.v4i32.v4f32")]
        fn vcvtq_n_u32_f32_(a: float32x4_t, n: i32) -> uint32x4_t;
    }
vcvtq_n_u32_f32_(a, N)
}

/// Floating-point convert to signed fixed-point, rounding toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vcvt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcvtzs))]
pub unsafe fn vcvt_s32_f32(a: float32x2_t) -> int32x2_t {
    simd_cast(a)
}

/// Floating-point convert to signed fixed-point, rounding toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vcvt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcvtzs))]
pub unsafe fn vcvtq_s32_f32(a: float32x4_t) -> int32x4_t {
    simd_cast(a)
}

/// Floating-point convert to unsigned fixed-point, rounding toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vcvt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcvtzu))]
pub unsafe fn vcvt_u32_f32(a: float32x2_t) -> uint32x2_t {
    simd_cast(a)
}

/// Floating-point convert to unsigned fixed-point, rounding toward zero
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vcvt))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fcvtzu))]
pub unsafe fn vcvtq_u32_f32(a: float32x4_t) -> uint32x4_t {
    simd_cast(a)
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.8", N = 4))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 4))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_lane_s8<const N: i32>(a: int8x8_t) -> int8x8_t {
    static_assert_imm3!(N);
    simd_shuffle8!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.8", N = 8))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 8))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_laneq_s8<const N: i32>(a: int8x16_t) -> int8x16_t {
    static_assert_imm4!(N);
    simd_shuffle16!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.16", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_lane_s16<const N: i32>(a: int16x4_t) -> int16x4_t {
    static_assert_imm2!(N);
    simd_shuffle4!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.16", N = 4))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 4))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_laneq_s16<const N: i32>(a: int16x8_t) -> int16x8_t {
    static_assert_imm3!(N);
    simd_shuffle8!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.32", N = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_lane_s32<const N: i32>(a: int32x2_t) -> int32x2_t {
    static_assert_imm1!(N);
    simd_shuffle2!(a, a, <const N: i32> [N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.32", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_laneq_s32<const N: i32>(a: int32x4_t) -> int32x4_t {
    static_assert_imm2!(N);
    simd_shuffle4!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.8", N = 8))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 8))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_laneq_s8<const N: i32>(a: int8x16_t) -> int8x8_t {
    static_assert_imm4!(N);
    simd_shuffle8!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.16", N = 4))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 4))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_laneq_s16<const N: i32>(a: int16x8_t) -> int16x4_t {
    static_assert_imm3!(N);
    simd_shuffle4!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.32", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_laneq_s32<const N: i32>(a: int32x4_t) -> int32x2_t {
    static_assert_imm2!(N);
    simd_shuffle2!(a, a, <const N: i32> [N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.8", N = 4))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 4))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_lane_s8<const N: i32>(a: int8x8_t) -> int8x16_t {
    static_assert_imm3!(N);
    simd_shuffle16!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.16", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_lane_s16<const N: i32>(a: int16x4_t) -> int16x8_t {
    static_assert_imm2!(N);
    simd_shuffle8!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.32", N = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_lane_s32<const N: i32>(a: int32x2_t) -> int32x4_t {
    static_assert_imm1!(N);
    simd_shuffle4!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.8", N = 4))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 4))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_lane_u8<const N: i32>(a: uint8x8_t) -> uint8x8_t {
    static_assert_imm3!(N);
    simd_shuffle8!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.8", N = 8))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 8))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_laneq_u8<const N: i32>(a: uint8x16_t) -> uint8x16_t {
    static_assert_imm4!(N);
    simd_shuffle16!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.16", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_lane_u16<const N: i32>(a: uint16x4_t) -> uint16x4_t {
    static_assert_imm2!(N);
    simd_shuffle4!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.16", N = 4))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 4))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_laneq_u16<const N: i32>(a: uint16x8_t) -> uint16x8_t {
    static_assert_imm3!(N);
    simd_shuffle8!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.32", N = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_lane_u32<const N: i32>(a: uint32x2_t) -> uint32x2_t {
    static_assert_imm1!(N);
    simd_shuffle2!(a, a, <const N: i32> [N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.32", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_laneq_u32<const N: i32>(a: uint32x4_t) -> uint32x4_t {
    static_assert_imm2!(N);
    simd_shuffle4!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.8", N = 8))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 8))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_laneq_u8<const N: i32>(a: uint8x16_t) -> uint8x8_t {
    static_assert_imm4!(N);
    simd_shuffle8!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.16", N = 4))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 4))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_laneq_u16<const N: i32>(a: uint16x8_t) -> uint16x4_t {
    static_assert_imm3!(N);
    simd_shuffle4!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.32", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_laneq_u32<const N: i32>(a: uint32x4_t) -> uint32x2_t {
    static_assert_imm2!(N);
    simd_shuffle2!(a, a, <const N: i32> [N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.8", N = 4))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 4))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_lane_u8<const N: i32>(a: uint8x8_t) -> uint8x16_t {
    static_assert_imm3!(N);
    simd_shuffle16!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.16", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_lane_u16<const N: i32>(a: uint16x4_t) -> uint16x8_t {
    static_assert_imm2!(N);
    simd_shuffle8!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.32", N = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_lane_u32<const N: i32>(a: uint32x2_t) -> uint32x4_t {
    static_assert_imm1!(N);
    simd_shuffle4!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.8", N = 4))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 4))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_lane_p8<const N: i32>(a: poly8x8_t) -> poly8x8_t {
    static_assert_imm3!(N);
    simd_shuffle8!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.8", N = 8))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 8))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_laneq_p8<const N: i32>(a: poly8x16_t) -> poly8x16_t {
    static_assert_imm4!(N);
    simd_shuffle16!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.16", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_lane_p16<const N: i32>(a: poly16x4_t) -> poly16x4_t {
    static_assert_imm2!(N);
    simd_shuffle4!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.16", N = 4))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 4))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_laneq_p16<const N: i32>(a: poly16x8_t) -> poly16x8_t {
    static_assert_imm3!(N);
    simd_shuffle8!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.8", N = 8))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 8))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_laneq_p8<const N: i32>(a: poly8x16_t) -> poly8x8_t {
    static_assert_imm4!(N);
    simd_shuffle8!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.16", N = 4))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 4))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_laneq_p16<const N: i32>(a: poly16x8_t) -> poly16x4_t {
    static_assert_imm3!(N);
    simd_shuffle4!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.8", N = 4))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 4))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_lane_p8<const N: i32>(a: poly8x8_t) -> poly8x16_t {
    static_assert_imm3!(N);
    simd_shuffle16!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.16", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_lane_p16<const N: i32>(a: poly16x4_t) -> poly16x8_t {
    static_assert_imm2!(N);
    simd_shuffle8!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmov, N = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_laneq_s64<const N: i32>(a: int64x2_t) -> int64x2_t {
    static_assert_imm1!(N);
    simd_shuffle2!(a, a, <const N: i32> [N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmov, N = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 0))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_lane_s64<const N: i32>(a: int64x1_t) -> int64x2_t {
    static_assert!(N : i32 where N == 0);
    simd_shuffle2!(a, a, <const N: i32> [N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmov, N = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_laneq_u64<const N: i32>(a: uint64x2_t) -> uint64x2_t {
    static_assert_imm1!(N);
    simd_shuffle2!(a, a, <const N: i32> [N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmov, N = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 0))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_lane_u64<const N: i32>(a: uint64x1_t) -> uint64x2_t {
    static_assert!(N : i32 where N == 0);
    simd_shuffle2!(a, a, <const N: i32> [N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.32", N = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_lane_f32<const N: i32>(a: float32x2_t) -> float32x2_t {
    static_assert_imm1!(N);
    simd_shuffle2!(a, a, <const N: i32> [N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.32", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_laneq_f32<const N: i32>(a: float32x4_t) -> float32x4_t {
    static_assert_imm2!(N);
    simd_shuffle4!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.32", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_laneq_f32<const N: i32>(a: float32x4_t) -> float32x2_t {
    static_assert_imm2!(N);
    simd_shuffle2!(a, a, <const N: i32> [N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vdup.32", N = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(dup, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdupq_lane_f32<const N: i32>(a: float32x2_t) -> float32x4_t {
    static_assert_imm1!(N);
    simd_shuffle4!(a, a, <const N: i32> [N as u32, N as u32, N as u32, N as u32])
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, N = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, N = 0))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_lane_s64<const N: i32>(a: int64x1_t) -> int64x1_t {
    static_assert!(N : i32 where N == 0);
    a
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, N = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, N = 0))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_lane_u64<const N: i32>(a: uint64x1_t) -> uint64x1_t {
    static_assert!(N : i32 where N == 0);
    a
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmov, N = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_laneq_s64<const N: i32>(a: int64x2_t) -> int64x1_t {
    static_assert_imm1!(N);
    transmute::<i64, _>(simd_extract(a, N as u32))
}

/// Set all vector lanes to the same value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmov, N = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, N = 1))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vdup_laneq_u64<const N: i32>(a: uint64x2_t) -> uint64x1_t {
    static_assert_imm1!(N);
    transmute::<u64, _>(simd_extract(a, N as u32))
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vext.8", N = 4))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 4))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vext_s8<const N: i32>(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    static_assert_imm3!(N);
    match N & 0b111 {
        0 => simd_shuffle8!(a, b, [0, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle8!(a, b, [1, 2, 3, 4, 5, 6, 7, 8]),
        2 => simd_shuffle8!(a, b, [2, 3, 4, 5, 6, 7, 8, 9]),
        3 => simd_shuffle8!(a, b, [3, 4, 5, 6, 7, 8, 9, 10]),
        4 => simd_shuffle8!(a, b, [4, 5, 6, 7, 8, 9, 10, 11]),
        5 => simd_shuffle8!(a, b, [5, 6, 7, 8, 9, 10, 11, 12]),
        6 => simd_shuffle8!(a, b, [6, 7, 8, 9, 10, 11, 12, 13]),
        7 => simd_shuffle8!(a, b, [7, 8, 9, 10, 11, 12, 13, 14]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vext.8", N = 8))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 8))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vextq_s8<const N: i32>(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    static_assert_imm4!(N);
    match N & 0b1111 {
        0 => simd_shuffle16!(a, b, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        1 => simd_shuffle16!(a, b, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
        2 => simd_shuffle16!(a, b, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
        3 => simd_shuffle16!(a, b, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
        4 => simd_shuffle16!(a, b, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
        5 => simd_shuffle16!(a, b, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        6 => simd_shuffle16!(a, b, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]),
        7 => simd_shuffle16!(a, b, [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]),
        8 => simd_shuffle16!(a, b, [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),
        9 => simd_shuffle16!(a, b, [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]),
        10 => simd_shuffle16!(a, b, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]),
        11 => simd_shuffle16!(a, b, [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]),
        12 => simd_shuffle16!(a, b, [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]),
        13 => simd_shuffle16!(a, b, [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]),
        14 => simd_shuffle16!(a, b, [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
        15 => simd_shuffle16!(a, b, [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vext.8", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vext_s16<const N: i32>(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    static_assert_imm2!(N);
    match N & 0b11 {
        0 => simd_shuffle4!(a, b, [0, 1, 2, 3]),
        1 => simd_shuffle4!(a, b, [1, 2, 3, 4]),
        2 => simd_shuffle4!(a, b, [2, 3, 4, 5]),
        3 => simd_shuffle4!(a, b, [3, 4, 5, 6]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vext.8", N = 4))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 4))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vextq_s16<const N: i32>(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    static_assert_imm3!(N);
    match N & 0b111 {
        0 => simd_shuffle8!(a, b, [0, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle8!(a, b, [1, 2, 3, 4, 5, 6, 7, 8]),
        2 => simd_shuffle8!(a, b, [2, 3, 4, 5, 6, 7, 8, 9]),
        3 => simd_shuffle8!(a, b, [3, 4, 5, 6, 7, 8, 9, 10]),
        4 => simd_shuffle8!(a, b, [4, 5, 6, 7, 8, 9, 10, 11]),
        5 => simd_shuffle8!(a, b, [5, 6, 7, 8, 9, 10, 11, 12]),
        6 => simd_shuffle8!(a, b, [6, 7, 8, 9, 10, 11, 12, 13]),
        7 => simd_shuffle8!(a, b, [7, 8, 9, 10, 11, 12, 13, 14]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vext.8", N = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vext_s32<const N: i32>(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    static_assert_imm1!(N);
    match N & 0b1 {
        0 => simd_shuffle2!(a, b, [0, 1]),
        1 => simd_shuffle2!(a, b, [1, 2]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vext.8", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vextq_s32<const N: i32>(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    static_assert_imm2!(N);
    match N & 0b11 {
        0 => simd_shuffle4!(a, b, [0, 1, 2, 3]),
        1 => simd_shuffle4!(a, b, [1, 2, 3, 4]),
        2 => simd_shuffle4!(a, b, [2, 3, 4, 5]),
        3 => simd_shuffle4!(a, b, [3, 4, 5, 6]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vext.8", N = 4))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 4))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vext_u8<const N: i32>(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    static_assert_imm3!(N);
    match N & 0b111 {
        0 => simd_shuffle8!(a, b, [0, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle8!(a, b, [1, 2, 3, 4, 5, 6, 7, 8]),
        2 => simd_shuffle8!(a, b, [2, 3, 4, 5, 6, 7, 8, 9]),
        3 => simd_shuffle8!(a, b, [3, 4, 5, 6, 7, 8, 9, 10]),
        4 => simd_shuffle8!(a, b, [4, 5, 6, 7, 8, 9, 10, 11]),
        5 => simd_shuffle8!(a, b, [5, 6, 7, 8, 9, 10, 11, 12]),
        6 => simd_shuffle8!(a, b, [6, 7, 8, 9, 10, 11, 12, 13]),
        7 => simd_shuffle8!(a, b, [7, 8, 9, 10, 11, 12, 13, 14]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vext.8", N = 8))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 8))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vextq_u8<const N: i32>(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    static_assert_imm4!(N);
    match N & 0b1111 {
        0 => simd_shuffle16!(a, b, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        1 => simd_shuffle16!(a, b, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
        2 => simd_shuffle16!(a, b, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
        3 => simd_shuffle16!(a, b, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
        4 => simd_shuffle16!(a, b, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
        5 => simd_shuffle16!(a, b, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        6 => simd_shuffle16!(a, b, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]),
        7 => simd_shuffle16!(a, b, [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]),
        8 => simd_shuffle16!(a, b, [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),
        9 => simd_shuffle16!(a, b, [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]),
        10 => simd_shuffle16!(a, b, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]),
        11 => simd_shuffle16!(a, b, [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]),
        12 => simd_shuffle16!(a, b, [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]),
        13 => simd_shuffle16!(a, b, [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]),
        14 => simd_shuffle16!(a, b, [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
        15 => simd_shuffle16!(a, b, [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vext.8", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vext_u16<const N: i32>(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    static_assert_imm2!(N);
    match N & 0b11 {
        0 => simd_shuffle4!(a, b, [0, 1, 2, 3]),
        1 => simd_shuffle4!(a, b, [1, 2, 3, 4]),
        2 => simd_shuffle4!(a, b, [2, 3, 4, 5]),
        3 => simd_shuffle4!(a, b, [3, 4, 5, 6]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vext.8", N = 4))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 4))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vextq_u16<const N: i32>(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    static_assert_imm3!(N);
    match N & 0b111 {
        0 => simd_shuffle8!(a, b, [0, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle8!(a, b, [1, 2, 3, 4, 5, 6, 7, 8]),
        2 => simd_shuffle8!(a, b, [2, 3, 4, 5, 6, 7, 8, 9]),
        3 => simd_shuffle8!(a, b, [3, 4, 5, 6, 7, 8, 9, 10]),
        4 => simd_shuffle8!(a, b, [4, 5, 6, 7, 8, 9, 10, 11]),
        5 => simd_shuffle8!(a, b, [5, 6, 7, 8, 9, 10, 11, 12]),
        6 => simd_shuffle8!(a, b, [6, 7, 8, 9, 10, 11, 12, 13]),
        7 => simd_shuffle8!(a, b, [7, 8, 9, 10, 11, 12, 13, 14]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vext.8", N = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vext_u32<const N: i32>(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    static_assert_imm1!(N);
    match N & 0b1 {
        0 => simd_shuffle2!(a, b, [0, 1]),
        1 => simd_shuffle2!(a, b, [1, 2]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vext.8", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vextq_u32<const N: i32>(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    static_assert_imm2!(N);
    match N & 0b11 {
        0 => simd_shuffle4!(a, b, [0, 1, 2, 3]),
        1 => simd_shuffle4!(a, b, [1, 2, 3, 4]),
        2 => simd_shuffle4!(a, b, [2, 3, 4, 5]),
        3 => simd_shuffle4!(a, b, [3, 4, 5, 6]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vext.8", N = 4))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 4))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vext_p8<const N: i32>(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t {
    static_assert_imm3!(N);
    match N & 0b111 {
        0 => simd_shuffle8!(a, b, [0, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle8!(a, b, [1, 2, 3, 4, 5, 6, 7, 8]),
        2 => simd_shuffle8!(a, b, [2, 3, 4, 5, 6, 7, 8, 9]),
        3 => simd_shuffle8!(a, b, [3, 4, 5, 6, 7, 8, 9, 10]),
        4 => simd_shuffle8!(a, b, [4, 5, 6, 7, 8, 9, 10, 11]),
        5 => simd_shuffle8!(a, b, [5, 6, 7, 8, 9, 10, 11, 12]),
        6 => simd_shuffle8!(a, b, [6, 7, 8, 9, 10, 11, 12, 13]),
        7 => simd_shuffle8!(a, b, [7, 8, 9, 10, 11, 12, 13, 14]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vext.8", N = 8))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 8))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vextq_p8<const N: i32>(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    static_assert_imm4!(N);
    match N & 0b1111 {
        0 => simd_shuffle16!(a, b, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
        1 => simd_shuffle16!(a, b, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]),
        2 => simd_shuffle16!(a, b, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]),
        3 => simd_shuffle16!(a, b, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]),
        4 => simd_shuffle16!(a, b, [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]),
        5 => simd_shuffle16!(a, b, [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]),
        6 => simd_shuffle16!(a, b, [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]),
        7 => simd_shuffle16!(a, b, [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]),
        8 => simd_shuffle16!(a, b, [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]),
        9 => simd_shuffle16!(a, b, [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]),
        10 => simd_shuffle16!(a, b, [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]),
        11 => simd_shuffle16!(a, b, [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]),
        12 => simd_shuffle16!(a, b, [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]),
        13 => simd_shuffle16!(a, b, [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]),
        14 => simd_shuffle16!(a, b, [14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]),
        15 => simd_shuffle16!(a, b, [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vext.8", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vext_p16<const N: i32>(a: poly16x4_t, b: poly16x4_t) -> poly16x4_t {
    static_assert_imm2!(N);
    match N & 0b11 {
        0 => simd_shuffle4!(a, b, [0, 1, 2, 3]),
        1 => simd_shuffle4!(a, b, [1, 2, 3, 4]),
        2 => simd_shuffle4!(a, b, [2, 3, 4, 5]),
        3 => simd_shuffle4!(a, b, [3, 4, 5, 6]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vext.8", N = 4))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 4))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vextq_p16<const N: i32>(a: poly16x8_t, b: poly16x8_t) -> poly16x8_t {
    static_assert_imm3!(N);
    match N & 0b111 {
        0 => simd_shuffle8!(a, b, [0, 1, 2, 3, 4, 5, 6, 7]),
        1 => simd_shuffle8!(a, b, [1, 2, 3, 4, 5, 6, 7, 8]),
        2 => simd_shuffle8!(a, b, [2, 3, 4, 5, 6, 7, 8, 9]),
        3 => simd_shuffle8!(a, b, [3, 4, 5, 6, 7, 8, 9, 10]),
        4 => simd_shuffle8!(a, b, [4, 5, 6, 7, 8, 9, 10, 11]),
        5 => simd_shuffle8!(a, b, [5, 6, 7, 8, 9, 10, 11, 12]),
        6 => simd_shuffle8!(a, b, [6, 7, 8, 9, 10, 11, 12, 13]),
        7 => simd_shuffle8!(a, b, [7, 8, 9, 10, 11, 12, 13, 14]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmov, N = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vextq_s64<const N: i32>(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    static_assert_imm1!(N);
    match N & 0b1 {
        0 => simd_shuffle2!(a, b, [0, 1]),
        1 => simd_shuffle2!(a, b, [1, 2]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmov, N = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vextq_u64<const N: i32>(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    static_assert_imm1!(N);
    match N & 0b1 {
        0 => simd_shuffle2!(a, b, [0, 1]),
        1 => simd_shuffle2!(a, b, [1, 2]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vext.8", N = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vext_f32<const N: i32>(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    static_assert_imm1!(N);
    match N & 0b1 {
        0 => simd_shuffle2!(a, b, [0, 1]),
        1 => simd_shuffle2!(a, b, [1, 2]),
        _ => unreachable_unchecked(),
    }
}

/// Extract vector from pair of vectors
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vext.8", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ext, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vextq_f32<const N: i32>(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    static_assert_imm2!(N);
    match N & 0b11 {
        0 => simd_shuffle4!(a, b, [0, 1, 2, 3]),
        1 => simd_shuffle4!(a, b, [1, 2, 3, 4]),
        2 => simd_shuffle4!(a, b, [2, 3, 4, 5]),
        3 => simd_shuffle4!(a, b, [3, 4, 5, 6]),
        _ => unreachable_unchecked(),
    }
}

/// Multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmla_s8(a: int8x8_t, b: int8x8_t, c: int8x8_t) -> int8x8_t {
    simd_add(a, simd_mul(b, c))
}

/// Multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmlaq_s8(a: int8x16_t, b: int8x16_t, c: int8x16_t) -> int8x16_t {
    simd_add(a, simd_mul(b, c))
}

/// Multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmla_s16(a: int16x4_t, b: int16x4_t, c: int16x4_t) -> int16x4_t {
    simd_add(a, simd_mul(b, c))
}

/// Multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmlaq_s16(a: int16x8_t, b: int16x8_t, c: int16x8_t) -> int16x8_t {
    simd_add(a, simd_mul(b, c))
}

/// Multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmla_s32(a: int32x2_t, b: int32x2_t, c: int32x2_t) -> int32x2_t {
    simd_add(a, simd_mul(b, c))
}

/// Multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmlaq_s32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
    simd_add(a, simd_mul(b, c))
}

/// Multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmla_u8(a: uint8x8_t, b: uint8x8_t, c: uint8x8_t) -> uint8x8_t {
    simd_add(a, simd_mul(b, c))
}

/// Multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmlaq_u8(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t {
    simd_add(a, simd_mul(b, c))
}

/// Multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmla_u16(a: uint16x4_t, b: uint16x4_t, c: uint16x4_t) -> uint16x4_t {
    simd_add(a, simd_mul(b, c))
}

/// Multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmlaq_u16(a: uint16x8_t, b: uint16x8_t, c: uint16x8_t) -> uint16x8_t {
    simd_add(a, simd_mul(b, c))
}

/// Multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmla_u32(a: uint32x2_t, b: uint32x2_t, c: uint32x2_t) -> uint32x2_t {
    simd_add(a, simd_mul(b, c))
}

/// Multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmlaq_u32(a: uint32x4_t, b: uint32x4_t, c: uint32x4_t) -> uint32x4_t {
    simd_add(a, simd_mul(b, c))
}

/// Floating-point multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul))]
pub unsafe fn vmla_f32(a: float32x2_t, b: float32x2_t, c: float32x2_t) -> float32x2_t {
    simd_add(a, simd_mul(b, c))
}

/// Floating-point multiply-add to accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul))]
pub unsafe fn vmlaq_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    simd_add(a, simd_mul(b, c))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmla_n_s16(a: int16x4_t, b: int16x4_t, c: i16) -> int16x4_t {
    vmla_s16(a, b, vdup_n_s16(c))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmlaq_n_s16(a: int16x8_t, b: int16x8_t, c: i16) -> int16x8_t {
    vmlaq_s16(a, b, vdupq_n_s16(c))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmla_n_s32(a: int32x2_t, b: int32x2_t, c: i32) -> int32x2_t {
    vmla_s32(a, b, vdup_n_s32(c))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmlaq_n_s32(a: int32x4_t, b: int32x4_t, c: i32) -> int32x4_t {
    vmlaq_s32(a, b, vdupq_n_s32(c))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmla_n_u16(a: uint16x4_t, b: uint16x4_t, c: u16) -> uint16x4_t {
    vmla_u16(a, b, vdup_n_u16(c))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmlaq_n_u16(a: uint16x8_t, b: uint16x8_t, c: u16) -> uint16x8_t {
    vmlaq_u16(a, b, vdupq_n_u16(c))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmla_n_u32(a: uint32x2_t, b: uint32x2_t, c: u32) -> uint32x2_t {
    vmla_u32(a, b, vdup_n_u32(c))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla))]
pub unsafe fn vmlaq_n_u32(a: uint32x4_t, b: uint32x4_t, c: u32) -> uint32x4_t {
    vmlaq_u32(a, b, vdupq_n_u32(c))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul))]
pub unsafe fn vmla_n_f32(a: float32x2_t, b: float32x2_t, c: f32) -> float32x2_t {
    vmla_f32(a, b, vdup_n_f32(c))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul))]
pub unsafe fn vmlaq_n_f32(a: float32x4_t, b: float32x4_t, c: f32) -> float32x4_t {
    vmlaq_f32(a, b, vdupq_n_f32(c))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmla_lane_s16<const LANE: i32>(a: int16x4_t, b: int16x4_t, c: int16x4_t) -> int16x4_t {
    static_assert_imm2!(LANE);
    vmla_s16(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmla_laneq_s16<const LANE: i32>(a: int16x4_t, b: int16x4_t, c: int16x8_t) -> int16x4_t {
    static_assert_imm3!(LANE);
    vmla_s16(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlaq_lane_s16<const LANE: i32>(a: int16x8_t, b: int16x8_t, c: int16x4_t) -> int16x8_t {
    static_assert_imm2!(LANE);
    vmlaq_s16(a, b, simd_shuffle8!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlaq_laneq_s16<const LANE: i32>(a: int16x8_t, b: int16x8_t, c: int16x8_t) -> int16x8_t {
    static_assert_imm3!(LANE);
    vmlaq_s16(a, b, simd_shuffle8!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmla_lane_s32<const LANE: i32>(a: int32x2_t, b: int32x2_t, c: int32x2_t) -> int32x2_t {
    static_assert_imm1!(LANE);
    vmla_s32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmla_laneq_s32<const LANE: i32>(a: int32x2_t, b: int32x2_t, c: int32x4_t) -> int32x2_t {
    static_assert_imm2!(LANE);
    vmla_s32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlaq_lane_s32<const LANE: i32>(a: int32x4_t, b: int32x4_t, c: int32x2_t) -> int32x4_t {
    static_assert_imm1!(LANE);
    vmlaq_s32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlaq_laneq_s32<const LANE: i32>(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
    static_assert_imm2!(LANE);
    vmlaq_s32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmla_lane_u16<const LANE: i32>(a: uint16x4_t, b: uint16x4_t, c: uint16x4_t) -> uint16x4_t {
    static_assert_imm2!(LANE);
    vmla_u16(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmla_laneq_u16<const LANE: i32>(a: uint16x4_t, b: uint16x4_t, c: uint16x8_t) -> uint16x4_t {
    static_assert_imm3!(LANE);
    vmla_u16(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlaq_lane_u16<const LANE: i32>(a: uint16x8_t, b: uint16x8_t, c: uint16x4_t) -> uint16x8_t {
    static_assert_imm2!(LANE);
    vmlaq_u16(a, b, simd_shuffle8!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlaq_laneq_u16<const LANE: i32>(a: uint16x8_t, b: uint16x8_t, c: uint16x8_t) -> uint16x8_t {
    static_assert_imm3!(LANE);
    vmlaq_u16(a, b, simd_shuffle8!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmla_lane_u32<const LANE: i32>(a: uint32x2_t, b: uint32x2_t, c: uint32x2_t) -> uint32x2_t {
    static_assert_imm1!(LANE);
    vmla_u32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmla_laneq_u32<const LANE: i32>(a: uint32x2_t, b: uint32x2_t, c: uint32x4_t) -> uint32x2_t {
    static_assert_imm2!(LANE);
    vmla_u32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlaq_lane_u32<const LANE: i32>(a: uint32x4_t, b: uint32x4_t, c: uint32x2_t) -> uint32x4_t {
    static_assert_imm1!(LANE);
    vmlaq_u32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.i32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mla, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlaq_laneq_u32<const LANE: i32>(a: uint32x4_t, b: uint32x4_t, c: uint32x4_t) -> uint32x4_t {
    static_assert_imm2!(LANE);
    vmlaq_u32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.f32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmla_lane_f32<const LANE: i32>(a: float32x2_t, b: float32x2_t, c: float32x2_t) -> float32x2_t {
    static_assert_imm1!(LANE);
    vmla_f32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.f32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmla_laneq_f32<const LANE: i32>(a: float32x2_t, b: float32x2_t, c: float32x4_t) -> float32x2_t {
    static_assert_imm2!(LANE);
    vmla_f32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.f32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlaq_lane_f32<const LANE: i32>(a: float32x4_t, b: float32x4_t, c: float32x2_t) -> float32x4_t {
    static_assert_imm1!(LANE);
    vmlaq_f32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmla.f32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlaq_laneq_f32<const LANE: i32>(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    static_assert_imm2!(LANE);
    vmlaq_f32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Signed multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlal.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smlal))]
pub unsafe fn vmlal_s8(a: int16x8_t, b: int8x8_t, c: int8x8_t) -> int16x8_t {
    simd_add(a, vmull_s8(b, c))
}

/// Signed multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlal.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smlal))]
pub unsafe fn vmlal_s16(a: int32x4_t, b: int16x4_t, c: int16x4_t) -> int32x4_t {
    simd_add(a, vmull_s16(b, c))
}

/// Signed multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlal.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smlal))]
pub unsafe fn vmlal_s32(a: int64x2_t, b: int32x2_t, c: int32x2_t) -> int64x2_t {
    simd_add(a, vmull_s32(b, c))
}

/// Unsigned multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlal.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umlal))]
pub unsafe fn vmlal_u8(a: uint16x8_t, b: uint8x8_t, c: uint8x8_t) -> uint16x8_t {
    simd_add(a, vmull_u8(b, c))
}

/// Unsigned multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlal.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umlal))]
pub unsafe fn vmlal_u16(a: uint32x4_t, b: uint16x4_t, c: uint16x4_t) -> uint32x4_t {
    simd_add(a, vmull_u16(b, c))
}

/// Unsigned multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlal.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umlal))]
pub unsafe fn vmlal_u32(a: uint64x2_t, b: uint32x2_t, c: uint32x2_t) -> uint64x2_t {
    simd_add(a, vmull_u32(b, c))
}

/// Vector widening multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlal.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smlal))]
pub unsafe fn vmlal_n_s16(a: int32x4_t, b: int16x4_t, c: i16) -> int32x4_t {
    vmlal_s16(a, b, vdup_n_s16(c))
}

/// Vector widening multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlal.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smlal))]
pub unsafe fn vmlal_n_s32(a: int64x2_t, b: int32x2_t, c: i32) -> int64x2_t {
    vmlal_s32(a, b, vdup_n_s32(c))
}

/// Vector widening multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlal.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umlal))]
pub unsafe fn vmlal_n_u16(a: uint32x4_t, b: uint16x4_t, c: u16) -> uint32x4_t {
    vmlal_u16(a, b, vdup_n_u16(c))
}

/// Vector widening multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlal.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umlal))]
pub unsafe fn vmlal_n_u32(a: uint64x2_t, b: uint32x2_t, c: u32) -> uint64x2_t {
    vmlal_u32(a, b, vdup_n_u32(c))
}

/// Vector widening multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlal.s16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smlal, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlal_lane_s16<const LANE: i32>(a: int32x4_t, b: int16x4_t, c: int16x4_t) -> int32x4_t {
    static_assert_imm2!(LANE);
    vmlal_s16(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector widening multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlal.s16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smlal, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlal_laneq_s16<const LANE: i32>(a: int32x4_t, b: int16x4_t, c: int16x8_t) -> int32x4_t {
    static_assert_imm3!(LANE);
    vmlal_s16(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector widening multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlal.s32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smlal, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlal_lane_s32<const LANE: i32>(a: int64x2_t, b: int32x2_t, c: int32x2_t) -> int64x2_t {
    static_assert_imm1!(LANE);
    vmlal_s32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector widening multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlal.s32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smlal, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlal_laneq_s32<const LANE: i32>(a: int64x2_t, b: int32x2_t, c: int32x4_t) -> int64x2_t {
    static_assert_imm2!(LANE);
    vmlal_s32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector widening multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlal.u16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umlal, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlal_lane_u16<const LANE: i32>(a: uint32x4_t, b: uint16x4_t, c: uint16x4_t) -> uint32x4_t {
    static_assert_imm2!(LANE);
    vmlal_u16(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector widening multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlal.u16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umlal, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlal_laneq_u16<const LANE: i32>(a: uint32x4_t, b: uint16x4_t, c: uint16x8_t) -> uint32x4_t {
    static_assert_imm3!(LANE);
    vmlal_u16(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector widening multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlal.u32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umlal, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlal_lane_u32<const LANE: i32>(a: uint64x2_t, b: uint32x2_t, c: uint32x2_t) -> uint64x2_t {
    static_assert_imm1!(LANE);
    vmlal_u32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector widening multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlal.u32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umlal, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlal_laneq_u32<const LANE: i32>(a: uint64x2_t, b: uint32x2_t, c: uint32x4_t) -> uint64x2_t {
    static_assert_imm2!(LANE);
    vmlal_u32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmls_s8(a: int8x8_t, b: int8x8_t, c: int8x8_t) -> int8x8_t {
    simd_sub(a, simd_mul(b, c))
}

/// Multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmlsq_s8(a: int8x16_t, b: int8x16_t, c: int8x16_t) -> int8x16_t {
    simd_sub(a, simd_mul(b, c))
}

/// Multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmls_s16(a: int16x4_t, b: int16x4_t, c: int16x4_t) -> int16x4_t {
    simd_sub(a, simd_mul(b, c))
}

/// Multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmlsq_s16(a: int16x8_t, b: int16x8_t, c: int16x8_t) -> int16x8_t {
    simd_sub(a, simd_mul(b, c))
}

/// Multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmls_s32(a: int32x2_t, b: int32x2_t, c: int32x2_t) -> int32x2_t {
    simd_sub(a, simd_mul(b, c))
}

/// Multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmlsq_s32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
    simd_sub(a, simd_mul(b, c))
}

/// Multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmls_u8(a: uint8x8_t, b: uint8x8_t, c: uint8x8_t) -> uint8x8_t {
    simd_sub(a, simd_mul(b, c))
}

/// Multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmlsq_u8(a: uint8x16_t, b: uint8x16_t, c: uint8x16_t) -> uint8x16_t {
    simd_sub(a, simd_mul(b, c))
}

/// Multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmls_u16(a: uint16x4_t, b: uint16x4_t, c: uint16x4_t) -> uint16x4_t {
    simd_sub(a, simd_mul(b, c))
}

/// Multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmlsq_u16(a: uint16x8_t, b: uint16x8_t, c: uint16x8_t) -> uint16x8_t {
    simd_sub(a, simd_mul(b, c))
}

/// Multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmls_u32(a: uint32x2_t, b: uint32x2_t, c: uint32x2_t) -> uint32x2_t {
    simd_sub(a, simd_mul(b, c))
}

/// Multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmlsq_u32(a: uint32x4_t, b: uint32x4_t, c: uint32x4_t) -> uint32x4_t {
    simd_sub(a, simd_mul(b, c))
}

/// Floating-point multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul))]
pub unsafe fn vmls_f32(a: float32x2_t, b: float32x2_t, c: float32x2_t) -> float32x2_t {
    simd_sub(a, simd_mul(b, c))
}

/// Floating-point multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul))]
pub unsafe fn vmlsq_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    simd_sub(a, simd_mul(b, c))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmls_n_s16(a: int16x4_t, b: int16x4_t, c: i16) -> int16x4_t {
    vmls_s16(a, b, vdup_n_s16(c))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmlsq_n_s16(a: int16x8_t, b: int16x8_t, c: i16) -> int16x8_t {
    vmlsq_s16(a, b, vdupq_n_s16(c))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmls_n_s32(a: int32x2_t, b: int32x2_t, c: i32) -> int32x2_t {
    vmls_s32(a, b, vdup_n_s32(c))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmlsq_n_s32(a: int32x4_t, b: int32x4_t, c: i32) -> int32x4_t {
    vmlsq_s32(a, b, vdupq_n_s32(c))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmls_n_u16(a: uint16x4_t, b: uint16x4_t, c: u16) -> uint16x4_t {
    vmls_u16(a, b, vdup_n_u16(c))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmlsq_n_u16(a: uint16x8_t, b: uint16x8_t, c: u16) -> uint16x8_t {
    vmlsq_u16(a, b, vdupq_n_u16(c))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmls_n_u32(a: uint32x2_t, b: uint32x2_t, c: u32) -> uint32x2_t {
    vmls_u32(a, b, vdup_n_u32(c))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls))]
pub unsafe fn vmlsq_n_u32(a: uint32x4_t, b: uint32x4_t, c: u32) -> uint32x4_t {
    vmlsq_u32(a, b, vdupq_n_u32(c))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul))]
pub unsafe fn vmls_n_f32(a: float32x2_t, b: float32x2_t, c: f32) -> float32x2_t {
    vmls_f32(a, b, vdup_n_f32(c))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul))]
pub unsafe fn vmlsq_n_f32(a: float32x4_t, b: float32x4_t, c: f32) -> float32x4_t {
    vmlsq_f32(a, b, vdupq_n_f32(c))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmls_lane_s16<const LANE: i32>(a: int16x4_t, b: int16x4_t, c: int16x4_t) -> int16x4_t {
    static_assert_imm2!(LANE);
    vmls_s16(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmls_laneq_s16<const LANE: i32>(a: int16x4_t, b: int16x4_t, c: int16x8_t) -> int16x4_t {
    static_assert_imm3!(LANE);
    vmls_s16(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsq_lane_s16<const LANE: i32>(a: int16x8_t, b: int16x8_t, c: int16x4_t) -> int16x8_t {
    static_assert_imm2!(LANE);
    vmlsq_s16(a, b, simd_shuffle8!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsq_laneq_s16<const LANE: i32>(a: int16x8_t, b: int16x8_t, c: int16x8_t) -> int16x8_t {
    static_assert_imm3!(LANE);
    vmlsq_s16(a, b, simd_shuffle8!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmls_lane_s32<const LANE: i32>(a: int32x2_t, b: int32x2_t, c: int32x2_t) -> int32x2_t {
    static_assert_imm1!(LANE);
    vmls_s32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmls_laneq_s32<const LANE: i32>(a: int32x2_t, b: int32x2_t, c: int32x4_t) -> int32x2_t {
    static_assert_imm2!(LANE);
    vmls_s32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsq_lane_s32<const LANE: i32>(a: int32x4_t, b: int32x4_t, c: int32x2_t) -> int32x4_t {
    static_assert_imm1!(LANE);
    vmlsq_s32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsq_laneq_s32<const LANE: i32>(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
    static_assert_imm2!(LANE);
    vmlsq_s32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmls_lane_u16<const LANE: i32>(a: uint16x4_t, b: uint16x4_t, c: uint16x4_t) -> uint16x4_t {
    static_assert_imm2!(LANE);
    vmls_u16(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmls_laneq_u16<const LANE: i32>(a: uint16x4_t, b: uint16x4_t, c: uint16x8_t) -> uint16x4_t {
    static_assert_imm3!(LANE);
    vmls_u16(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsq_lane_u16<const LANE: i32>(a: uint16x8_t, b: uint16x8_t, c: uint16x4_t) -> uint16x8_t {
    static_assert_imm2!(LANE);
    vmlsq_u16(a, b, simd_shuffle8!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsq_laneq_u16<const LANE: i32>(a: uint16x8_t, b: uint16x8_t, c: uint16x8_t) -> uint16x8_t {
    static_assert_imm3!(LANE);
    vmlsq_u16(a, b, simd_shuffle8!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmls_lane_u32<const LANE: i32>(a: uint32x2_t, b: uint32x2_t, c: uint32x2_t) -> uint32x2_t {
    static_assert_imm1!(LANE);
    vmls_u32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmls_laneq_u32<const LANE: i32>(a: uint32x2_t, b: uint32x2_t, c: uint32x4_t) -> uint32x2_t {
    static_assert_imm2!(LANE);
    vmls_u32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsq_lane_u32<const LANE: i32>(a: uint32x4_t, b: uint32x4_t, c: uint32x2_t) -> uint32x4_t {
    static_assert_imm1!(LANE);
    vmlsq_u32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.i32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mls, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsq_laneq_u32<const LANE: i32>(a: uint32x4_t, b: uint32x4_t, c: uint32x4_t) -> uint32x4_t {
    static_assert_imm2!(LANE);
    vmlsq_u32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.f32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmls_lane_f32<const LANE: i32>(a: float32x2_t, b: float32x2_t, c: float32x2_t) -> float32x2_t {
    static_assert_imm1!(LANE);
    vmls_f32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.f32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmls_laneq_f32<const LANE: i32>(a: float32x2_t, b: float32x2_t, c: float32x4_t) -> float32x2_t {
    static_assert_imm2!(LANE);
    vmls_f32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.f32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsq_lane_f32<const LANE: i32>(a: float32x4_t, b: float32x4_t, c: float32x2_t) -> float32x4_t {
    static_assert_imm1!(LANE);
    vmlsq_f32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmls.f32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsq_laneq_f32<const LANE: i32>(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    static_assert_imm2!(LANE);
    vmlsq_f32(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Signed multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlsl.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smlsl))]
pub unsafe fn vmlsl_s8(a: int16x8_t, b: int8x8_t, c: int8x8_t) -> int16x8_t {
    simd_sub(a, vmull_s8(b, c))
}

/// Signed multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlsl.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smlsl))]
pub unsafe fn vmlsl_s16(a: int32x4_t, b: int16x4_t, c: int16x4_t) -> int32x4_t {
    simd_sub(a, vmull_s16(b, c))
}

/// Signed multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlsl.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smlsl))]
pub unsafe fn vmlsl_s32(a: int64x2_t, b: int32x2_t, c: int32x2_t) -> int64x2_t {
    simd_sub(a, vmull_s32(b, c))
}

/// Unsigned multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlsl.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umlsl))]
pub unsafe fn vmlsl_u8(a: uint16x8_t, b: uint8x8_t, c: uint8x8_t) -> uint16x8_t {
    simd_sub(a, vmull_u8(b, c))
}

/// Unsigned multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlsl.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umlsl))]
pub unsafe fn vmlsl_u16(a: uint32x4_t, b: uint16x4_t, c: uint16x4_t) -> uint32x4_t {
    simd_sub(a, vmull_u16(b, c))
}

/// Unsigned multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlsl.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umlsl))]
pub unsafe fn vmlsl_u32(a: uint64x2_t, b: uint32x2_t, c: uint32x2_t) -> uint64x2_t {
    simd_sub(a, vmull_u32(b, c))
}

/// Vector widening multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlsl.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smlsl))]
pub unsafe fn vmlsl_n_s16(a: int32x4_t, b: int16x4_t, c: i16) -> int32x4_t {
    vmlsl_s16(a, b, vdup_n_s16(c))
}

/// Vector widening multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlsl.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smlsl))]
pub unsafe fn vmlsl_n_s32(a: int64x2_t, b: int32x2_t, c: i32) -> int64x2_t {
    vmlsl_s32(a, b, vdup_n_s32(c))
}

/// Vector widening multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlsl.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umlsl))]
pub unsafe fn vmlsl_n_u16(a: uint32x4_t, b: uint16x4_t, c: u16) -> uint32x4_t {
    vmlsl_u16(a, b, vdup_n_u16(c))
}

/// Vector widening multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlsl.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umlsl))]
pub unsafe fn vmlsl_n_u32(a: uint64x2_t, b: uint32x2_t, c: u32) -> uint64x2_t {
    vmlsl_u32(a, b, vdup_n_u32(c))
}

/// Vector widening multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlsl.s16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smlsl, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsl_lane_s16<const LANE: i32>(a: int32x4_t, b: int16x4_t, c: int16x4_t) -> int32x4_t {
    static_assert_imm2!(LANE);
    vmlsl_s16(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector widening multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlsl.s16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smlsl, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsl_laneq_s16<const LANE: i32>(a: int32x4_t, b: int16x4_t, c: int16x8_t) -> int32x4_t {
    static_assert_imm3!(LANE);
    vmlsl_s16(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector widening multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlsl.s32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smlsl, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsl_lane_s32<const LANE: i32>(a: int64x2_t, b: int32x2_t, c: int32x2_t) -> int64x2_t {
    static_assert_imm1!(LANE);
    vmlsl_s32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector widening multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlsl.s32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smlsl, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsl_laneq_s32<const LANE: i32>(a: int64x2_t, b: int32x2_t, c: int32x4_t) -> int64x2_t {
    static_assert_imm2!(LANE);
    vmlsl_s32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector widening multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlsl.u16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umlsl, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsl_lane_u16<const LANE: i32>(a: uint32x4_t, b: uint16x4_t, c: uint16x4_t) -> uint32x4_t {
    static_assert_imm2!(LANE);
    vmlsl_u16(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector widening multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlsl.u16", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umlsl, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsl_laneq_u16<const LANE: i32>(a: uint32x4_t, b: uint16x4_t, c: uint16x8_t) -> uint32x4_t {
    static_assert_imm3!(LANE);
    vmlsl_u16(a, b, simd_shuffle4!(c, c, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector widening multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlsl.u32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umlsl, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsl_lane_u32<const LANE: i32>(a: uint64x2_t, b: uint32x2_t, c: uint32x2_t) -> uint64x2_t {
    static_assert_imm1!(LANE);
    vmlsl_u32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector widening multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmlsl.u32", LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umlsl, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vmlsl_laneq_u32<const LANE: i32>(a: uint64x2_t, b: uint32x2_t, c: uint32x4_t) -> uint64x2_t {
    static_assert_imm2!(LANE);
    vmlsl_u32(a, b, simd_shuffle2!(c, c, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vneg.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(neg))]
pub unsafe fn vneg_s8(a: int8x8_t) -> int8x8_t {
    simd_neg(a)
}

/// Negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vneg.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(neg))]
pub unsafe fn vnegq_s8(a: int8x16_t) -> int8x16_t {
    simd_neg(a)
}

/// Negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vneg.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(neg))]
pub unsafe fn vneg_s16(a: int16x4_t) -> int16x4_t {
    simd_neg(a)
}

/// Negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vneg.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(neg))]
pub unsafe fn vnegq_s16(a: int16x8_t) -> int16x8_t {
    simd_neg(a)
}

/// Negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vneg.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(neg))]
pub unsafe fn vneg_s32(a: int32x2_t) -> int32x2_t {
    simd_neg(a)
}

/// Negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vneg.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(neg))]
pub unsafe fn vnegq_s32(a: int32x4_t) -> int32x4_t {
    simd_neg(a)
}

/// Negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vneg.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fneg))]
pub unsafe fn vneg_f32(a: float32x2_t) -> float32x2_t {
    simd_neg(a)
}

/// Negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vneg.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fneg))]
pub unsafe fn vnegq_f32(a: float32x4_t) -> float32x4_t {
    simd_neg(a)
}

/// Signed saturating negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqneg.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqneg))]
pub unsafe fn vqneg_s8(a: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqneg.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqneg.v8i8")]
        fn vqneg_s8_(a: int8x8_t) -> int8x8_t;
    }
vqneg_s8_(a)
}

/// Signed saturating negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqneg.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqneg))]
pub unsafe fn vqnegq_s8(a: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqneg.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqneg.v16i8")]
        fn vqnegq_s8_(a: int8x16_t) -> int8x16_t;
    }
vqnegq_s8_(a)
}

/// Signed saturating negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqneg.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqneg))]
pub unsafe fn vqneg_s16(a: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqneg.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqneg.v4i16")]
        fn vqneg_s16_(a: int16x4_t) -> int16x4_t;
    }
vqneg_s16_(a)
}

/// Signed saturating negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqneg.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqneg))]
pub unsafe fn vqnegq_s16(a: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqneg.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqneg.v8i16")]
        fn vqnegq_s16_(a: int16x8_t) -> int16x8_t;
    }
vqnegq_s16_(a)
}

/// Signed saturating negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqneg.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqneg))]
pub unsafe fn vqneg_s32(a: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqneg.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqneg.v2i32")]
        fn vqneg_s32_(a: int32x2_t) -> int32x2_t;
    }
vqneg_s32_(a)
}

/// Signed saturating negate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqneg.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqneg))]
pub unsafe fn vqnegq_s32(a: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqneg.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqneg.v4i32")]
        fn vqnegq_s32_(a: int32x4_t) -> int32x4_t;
    }
vqnegq_s32_(a)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsub_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.usub.sat.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v8i8")]
        fn vqsub_u8_(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    }
vqsub_u8_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsubq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.usub.sat.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v16i8")]
        fn vqsubq_u8_(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
    }
vqsubq_u8_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsub_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.usub.sat.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v4i16")]
        fn vqsub_u16_(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    }
vqsub_u16_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsubq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.usub.sat.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v8i16")]
        fn vqsubq_u16_(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t;
    }
vqsubq_u16_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsub_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.usub.sat.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v2i32")]
        fn vqsub_u32_(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    }
vqsub_u32_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsubq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.usub.sat.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v4i32")]
        fn vqsubq_u32_(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    }
vqsubq_u32_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.u64"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsub_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.usub.sat.v1i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v1i64")]
        fn vqsub_u64_(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t;
    }
vqsub_u64_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.u64"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqsub))]
pub unsafe fn vqsubq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.usub.sat.v2i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqsub.v2i64")]
        fn vqsubq_u64_(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t;
    }
vqsubq_u64_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsub_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.ssub.sat.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v8i8")]
        fn vqsub_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vqsub_s8_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsubq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.ssub.sat.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v16i8")]
        fn vqsubq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vqsubq_s8_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsub_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.ssub.sat.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v4i16")]
        fn vqsub_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vqsub_s16_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsubq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.ssub.sat.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v8i16")]
        fn vqsubq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vqsubq_s16_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsub_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.ssub.sat.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v2i32")]
        fn vqsub_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vqsub_s32_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsubq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.ssub.sat.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v4i32")]
        fn vqsubq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vqsubq_s32_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.s64"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsub_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.ssub.sat.v1i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v1i64")]
        fn vqsub_s64_(a: int64x1_t, b: int64x1_t) -> int64x1_t;
    }
vqsub_s64_(a, b)
}

/// Saturating subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqsub.s64"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqsub))]
pub unsafe fn vqsubq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.ssub.sat.v2i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqsub.v2i64")]
        fn vqsubq_s64_(a: int64x2_t, b: int64x2_t) -> int64x2_t;
    }
vqsubq_s64_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhadd))]
pub unsafe fn vhadd_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhaddu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhadd.v8i8")]
        fn vhadd_u8_(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    }
vhadd_u8_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhadd))]
pub unsafe fn vhaddq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhaddu.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhadd.v16i8")]
        fn vhaddq_u8_(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
    }
vhaddq_u8_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhadd))]
pub unsafe fn vhadd_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhaddu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhadd.v4i16")]
        fn vhadd_u16_(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    }
vhadd_u16_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhadd))]
pub unsafe fn vhaddq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhaddu.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhadd.v8i16")]
        fn vhaddq_u16_(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t;
    }
vhaddq_u16_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhadd))]
pub unsafe fn vhadd_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhaddu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhadd.v2i32")]
        fn vhadd_u32_(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    }
vhadd_u32_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhadd))]
pub unsafe fn vhaddq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhaddu.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhadd.v4i32")]
        fn vhaddq_u32_(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    }
vhaddq_u32_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shadd))]
pub unsafe fn vhadd_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhadds.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shadd.v8i8")]
        fn vhadd_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vhadd_s8_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shadd))]
pub unsafe fn vhaddq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhadds.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shadd.v16i8")]
        fn vhaddq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vhaddq_s8_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shadd))]
pub unsafe fn vhadd_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhadds.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shadd.v4i16")]
        fn vhadd_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vhadd_s16_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shadd))]
pub unsafe fn vhaddq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhadds.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shadd.v8i16")]
        fn vhaddq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vhaddq_s16_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shadd))]
pub unsafe fn vhadd_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhadds.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shadd.v2i32")]
        fn vhadd_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vhadd_s32_(a, b)
}

/// Halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhadd.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shadd))]
pub unsafe fn vhaddq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhadds.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shadd.v4i32")]
        fn vhaddq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vhaddq_s32_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urhadd))]
pub unsafe fn vrhadd_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhaddu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urhadd.v8i8")]
        fn vrhadd_u8_(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    }
vrhadd_u8_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urhadd))]
pub unsafe fn vrhaddq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhaddu.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urhadd.v16i8")]
        fn vrhaddq_u8_(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
    }
vrhaddq_u8_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urhadd))]
pub unsafe fn vrhadd_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhaddu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urhadd.v4i16")]
        fn vrhadd_u16_(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    }
vrhadd_u16_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urhadd))]
pub unsafe fn vrhaddq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhaddu.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urhadd.v8i16")]
        fn vrhaddq_u16_(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t;
    }
vrhaddq_u16_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urhadd))]
pub unsafe fn vrhadd_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhaddu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urhadd.v2i32")]
        fn vrhadd_u32_(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    }
vrhadd_u32_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urhadd))]
pub unsafe fn vrhaddq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhaddu.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urhadd.v4i32")]
        fn vrhaddq_u32_(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    }
vrhaddq_u32_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srhadd))]
pub unsafe fn vrhadd_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhadds.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srhadd.v8i8")]
        fn vrhadd_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vrhadd_s8_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srhadd))]
pub unsafe fn vrhaddq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhadds.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srhadd.v16i8")]
        fn vrhaddq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vrhaddq_s8_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srhadd))]
pub unsafe fn vrhadd_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhadds.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srhadd.v4i16")]
        fn vrhadd_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vrhadd_s16_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srhadd))]
pub unsafe fn vrhaddq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhadds.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srhadd.v8i16")]
        fn vrhaddq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vrhaddq_s16_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srhadd))]
pub unsafe fn vrhadd_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhadds.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srhadd.v2i32")]
        fn vrhadd_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vrhadd_s32_(a, b)
}

/// Rounding halving add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vrhadd.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srhadd))]
pub unsafe fn vrhaddq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrhadds.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srhadd.v4i32")]
        fn vrhaddq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vrhaddq_s32_(a, b)
}

/// Floating-point round to integral, to nearest with ties to even
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "fp-armv8,v8"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrintn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(frintn))]
pub unsafe fn vrndn_f32(a: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrintn.v2f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.frintn.v2f32")]
        fn vrndn_f32_(a: float32x2_t) -> float32x2_t;
    }
vrndn_f32_(a)
}

/// Floating-point round to integral, to nearest with ties to even
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "fp-armv8,v8"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrintn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(frintn))]
pub unsafe fn vrndnq_f32(a: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrintn.v4f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.frintn.v4f32")]
        fn vrndnq_f32_(a: float32x4_t) -> float32x4_t;
    }
vrndnq_f32_(a)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqadd_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.uadd.sat.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v8i8")]
        fn vqadd_u8_(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    }
vqadd_u8_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqaddq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.uadd.sat.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v16i8")]
        fn vqaddq_u8_(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
    }
vqaddq_u8_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqadd_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.uadd.sat.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v4i16")]
        fn vqadd_u16_(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    }
vqadd_u16_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqaddq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.uadd.sat.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v8i16")]
        fn vqaddq_u16_(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t;
    }
vqaddq_u16_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqadd_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.uadd.sat.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v2i32")]
        fn vqadd_u32_(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    }
vqadd_u32_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqaddq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.uadd.sat.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v4i32")]
        fn vqaddq_u32_(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    }
vqaddq_u32_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.u64"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqadd_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.uadd.sat.v1i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v1i64")]
        fn vqadd_u64_(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t;
    }
vqadd_u64_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.u64"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqadd))]
pub unsafe fn vqaddq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.uadd.sat.v2i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqadd.v2i64")]
        fn vqaddq_u64_(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t;
    }
vqaddq_u64_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqadd_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.sadd.sat.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v8i8")]
        fn vqadd_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vqadd_s8_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqaddq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.sadd.sat.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v16i8")]
        fn vqaddq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vqaddq_s8_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqadd_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.sadd.sat.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v4i16")]
        fn vqadd_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vqadd_s16_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqaddq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.sadd.sat.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v8i16")]
        fn vqaddq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vqaddq_s16_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqadd_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.sadd.sat.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v2i32")]
        fn vqadd_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vqadd_s32_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqaddq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.sadd.sat.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v4i32")]
        fn vqaddq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vqaddq_s32_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.s64"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqadd_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.sadd.sat.v1i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v1i64")]
        fn vqadd_s64_(a: int64x1_t, b: int64x1_t) -> int64x1_t;
    }
vqadd_s64_(a, b)
}

/// Saturating add
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqadd.s64"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqadd))]
pub unsafe fn vqaddq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.sadd.sat.v2i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqadd.v2i64")]
        fn vqaddq_s64_(a: int64x2_t, b: int64x2_t) -> int64x2_t;
    }
vqaddq_s64_(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmul_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmulq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmul_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmulq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmul_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmulq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmul_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmulq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmul_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmulq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmul_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmulq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_mul(a, b)
}

/// Polynomial multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(pmul))]
pub unsafe fn vmul_p8(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmulp.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.pmul.v8i8")]
        fn vmul_p8_(a: poly8x8_t, b: poly8x8_t) -> poly8x8_t;
    }
vmul_p8_(a, b)
}

/// Polynomial multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(pmul))]
pub unsafe fn vmulq_p8(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmulp.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.pmul.v16i8")]
        fn vmulq_p8_(a: poly8x16_t, b: poly8x16_t) -> poly8x16_t;
    }
vmulq_p8_(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul))]
pub unsafe fn vmul_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_mul(a, b)
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmul.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul))]
pub unsafe fn vmulq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_mul(a, b)
}

/// Vector multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmul_n_s16(a: int16x4_t, b: i16) -> int16x4_t {
    simd_mul(a, vdup_n_s16(b))
}

/// Vector multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmulq_n_s16(a: int16x8_t, b: i16) -> int16x8_t {
    simd_mul(a, vdupq_n_s16(b))
}

/// Vector multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmul_n_s32(a: int32x2_t, b: i32) -> int32x2_t {
    simd_mul(a, vdup_n_s32(b))
}

/// Vector multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmulq_n_s32(a: int32x4_t, b: i32) -> int32x4_t {
    simd_mul(a, vdupq_n_s32(b))
}

/// Vector multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmul_n_u16(a: uint16x4_t, b: u16) -> uint16x4_t {
    simd_mul(a, vdup_n_u16(b))
}

/// Vector multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmulq_n_u16(a: uint16x8_t, b: u16) -> uint16x8_t {
    simd_mul(a, vdupq_n_u16(b))
}

/// Vector multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmul_n_u32(a: uint32x2_t, b: u32) -> uint32x2_t {
    simd_mul(a, vdup_n_u32(b))
}

/// Vector multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul))]
pub unsafe fn vmulq_n_u32(a: uint32x4_t, b: u32) -> uint32x4_t {
    simd_mul(a, vdupq_n_u32(b))
}

/// Vector multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul))]
pub unsafe fn vmul_n_f32(a: float32x2_t, b: f32) -> float32x2_t {
    simd_mul(a, vdup_n_f32(b))
}

/// Vector multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul))]
pub unsafe fn vmulq_n_f32(a: float32x4_t, b: f32) -> float32x4_t {
    simd_mul(a, vdupq_n_f32(b))
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmul_lane_s16<const LANE: i32>(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    static_assert_imm2!(LANE);
    simd_mul(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmul_laneq_s16<const LANE: i32>(a: int16x4_t, b: int16x8_t) -> int16x4_t {
    static_assert_imm3!(LANE);
    simd_mul(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulq_lane_s16<const LANE: i32>(a: int16x8_t, b: int16x4_t) -> int16x8_t {
    static_assert_imm2!(LANE);
    simd_mul(a, simd_shuffle8!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulq_laneq_s16<const LANE: i32>(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    static_assert_imm3!(LANE);
    simd_mul(a, simd_shuffle8!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmul_lane_s32<const LANE: i32>(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    static_assert_imm1!(LANE);
    simd_mul(a, simd_shuffle2!(b, b, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmul_laneq_s32<const LANE: i32>(a: int32x2_t, b: int32x4_t) -> int32x2_t {
    static_assert_imm2!(LANE);
    simd_mul(a, simd_shuffle2!(b, b, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulq_lane_s32<const LANE: i32>(a: int32x4_t, b: int32x2_t) -> int32x4_t {
    static_assert_imm1!(LANE);
    simd_mul(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulq_laneq_s32<const LANE: i32>(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    static_assert_imm2!(LANE);
    simd_mul(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmul_lane_u16<const LANE: i32>(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    static_assert_imm2!(LANE);
    simd_mul(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmul_laneq_u16<const LANE: i32>(a: uint16x4_t, b: uint16x8_t) -> uint16x4_t {
    static_assert_imm3!(LANE);
    simd_mul(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulq_lane_u16<const LANE: i32>(a: uint16x8_t, b: uint16x4_t) -> uint16x8_t {
    static_assert_imm2!(LANE);
    simd_mul(a, simd_shuffle8!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulq_laneq_u16<const LANE: i32>(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    static_assert_imm3!(LANE);
    simd_mul(a, simd_shuffle8!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmul_lane_u32<const LANE: i32>(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    static_assert_imm1!(LANE);
    simd_mul(a, simd_shuffle2!(b, b, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmul_laneq_u32<const LANE: i32>(a: uint32x2_t, b: uint32x4_t) -> uint32x2_t {
    static_assert_imm2!(LANE);
    simd_mul(a, simd_shuffle2!(b, b, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulq_lane_u32<const LANE: i32>(a: uint32x4_t, b: uint32x2_t) -> uint32x4_t {
    static_assert_imm1!(LANE);
    simd_mul(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(mul, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulq_laneq_u32<const LANE: i32>(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    static_assert_imm2!(LANE);
    simd_mul(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Floating-point multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmul_lane_f32<const LANE: i32>(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    static_assert_imm1!(LANE);
    simd_mul(a, simd_shuffle2!(b, b, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Floating-point multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmul_laneq_f32<const LANE: i32>(a: float32x2_t, b: float32x4_t) -> float32x2_t {
    static_assert_imm2!(LANE);
    simd_mul(a, simd_shuffle2!(b, b, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Floating-point multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulq_lane_f32<const LANE: i32>(a: float32x4_t, b: float32x2_t) -> float32x4_t {
    static_assert_imm1!(LANE);
    simd_mul(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Floating-point multiply
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmul, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmul, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmulq_laneq_f32<const LANE: i32>(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    static_assert_imm2!(LANE);
    simd_mul(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Signed multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmull.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smull))]
pub unsafe fn vmull_s8(a: int8x8_t, b: int8x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmulls.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smull.v8i8")]
        fn vmull_s8_(a: int8x8_t, b: int8x8_t) -> int16x8_t;
    }
vmull_s8_(a, b)
}

/// Signed multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmull.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smull))]
pub unsafe fn vmull_s16(a: int16x4_t, b: int16x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmulls.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smull.v4i16")]
        fn vmull_s16_(a: int16x4_t, b: int16x4_t) -> int32x4_t;
    }
vmull_s16_(a, b)
}

/// Signed multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmull.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smull))]
pub unsafe fn vmull_s32(a: int32x2_t, b: int32x2_t) -> int64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmulls.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smull.v2i32")]
        fn vmull_s32_(a: int32x2_t, b: int32x2_t) -> int64x2_t;
    }
vmull_s32_(a, b)
}

/// Unsigned multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmull.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umull))]
pub unsafe fn vmull_u8(a: uint8x8_t, b: uint8x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmullu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umull.v8i8")]
        fn vmull_u8_(a: uint8x8_t, b: uint8x8_t) -> uint16x8_t;
    }
vmull_u8_(a, b)
}

/// Unsigned multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmull.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umull))]
pub unsafe fn vmull_u16(a: uint16x4_t, b: uint16x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmullu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umull.v4i16")]
        fn vmull_u16_(a: uint16x4_t, b: uint16x4_t) -> uint32x4_t;
    }
vmull_u16_(a, b)
}

/// Unsigned multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmull.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umull))]
pub unsafe fn vmull_u32(a: uint32x2_t, b: uint32x2_t) -> uint64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmullu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umull.v2i32")]
        fn vmull_u32_(a: uint32x2_t, b: uint32x2_t) -> uint64x2_t;
    }
vmull_u32_(a, b)
}

/// Polynomial multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vmull.p8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(pmull))]
pub unsafe fn vmull_p8(a: poly8x8_t, b: poly8x8_t) -> poly16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmullp.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.pmull.v8i8")]
        fn vmull_p8_(a: poly8x8_t, b: poly8x8_t) -> poly16x8_t;
    }
vmull_p8_(a, b)
}

/// Vector long multiply with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmull))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smull))]
pub unsafe fn vmullh_n_s16(a: int16x4_t, b: i16) -> int32x4_t {
    vmull_s16(a, vdup_n_s16(b))
}

/// Vector long multiply with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmull))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smull))]
pub unsafe fn vmulls_n_s32(a: int32x2_t, b: i32) -> int64x2_t {
    vmull_s32(a, vdup_n_s32(b))
}

/// Vector long multiply with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmull))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umull))]
pub unsafe fn vmullh_n_u16(a: uint16x4_t, b: u16) -> uint32x4_t {
    vmull_u16(a, vdup_n_u16(b))
}

/// Vector long multiply with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmull))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umull))]
pub unsafe fn vmulls_n_u32(a: uint32x2_t, b: u32) -> uint64x2_t {
    vmull_u32(a, vdup_n_u32(b))
}

/// Vector long multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmull, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smull, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmull_lane_s16<const LANE: i32>(a: int16x4_t, b: int16x4_t) -> int32x4_t {
    static_assert_imm2!(LANE);
    vmull_s16(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector long multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmull, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smull, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmull_laneq_s16<const LANE: i32>(a: int16x4_t, b: int16x8_t) -> int32x4_t {
    static_assert_imm3!(LANE);
    vmull_s16(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector long multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmull, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smull, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmull_lane_s32<const LANE: i32>(a: int32x2_t, b: int32x2_t) -> int64x2_t {
    static_assert_imm1!(LANE);
    vmull_s32(a, simd_shuffle2!(b, b, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector long multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmull, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smull, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmull_laneq_s32<const LANE: i32>(a: int32x2_t, b: int32x4_t) -> int64x2_t {
    static_assert_imm2!(LANE);
    vmull_s32(a, simd_shuffle2!(b, b, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector long multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmull, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umull, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmull_lane_u16<const LANE: i32>(a: uint16x4_t, b: uint16x4_t) -> uint32x4_t {
    static_assert_imm2!(LANE);
    vmull_u16(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector long multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmull, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umull, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmull_laneq_u16<const LANE: i32>(a: uint16x4_t, b: uint16x8_t) -> uint32x4_t {
    static_assert_imm3!(LANE);
    vmull_u16(a, simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]))
}

/// Vector long multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmull, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umull, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmull_lane_u32<const LANE: i32>(a: uint32x2_t, b: uint32x2_t) -> uint64x2_t {
    static_assert_imm1!(LANE);
    vmull_u32(a, simd_shuffle2!(b, b, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Vector long multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmull, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umull, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vmull_laneq_u32<const LANE: i32>(a: uint32x2_t, b: uint32x4_t) -> uint64x2_t {
    static_assert_imm2!(LANE);
    vmull_u32(a, simd_shuffle2!(b, b, <const LANE: i32> [LANE as u32, LANE as u32]))
}

/// Floating-point fused Multiply-Add to accumulator(vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "fp-armv8,v8"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vfma))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmla))]
pub unsafe fn vfma_f32(a: float32x2_t, b: float32x2_t, c: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.fma.v2f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.fma.v2f32")]
        fn vfma_f32_(a: float32x2_t, b: float32x2_t, c: float32x2_t) -> float32x2_t;
    }
vfma_f32_(b, c, a)
}

/// Floating-point fused Multiply-Add to accumulator(vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "fp-armv8,v8"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vfma))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmla))]
pub unsafe fn vfmaq_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.fma.v4f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.fma.v4f32")]
        fn vfmaq_f32_(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t;
    }
vfmaq_f32_(b, c, a)
}

/// Floating-point fused Multiply-Add to accumulator(vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "fp-armv8,v8"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vfma))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmla))]
pub unsafe fn vfma_n_f32(a: float32x2_t, b: float32x2_t, c: f32) -> float32x2_t {
    vfma_f32(a, b, vdup_n_f32(c))
}

/// Floating-point fused Multiply-Add to accumulator(vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "fp-armv8,v8"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vfma))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmla))]
pub unsafe fn vfmaq_n_f32(a: float32x4_t, b: float32x4_t, c: f32) -> float32x4_t {
    vfmaq_f32(a, b, vdupq_n_f32(c))
}

/// Floating-point fused multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "fp-armv8,v8"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vfms))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmls))]
pub unsafe fn vfms_f32(a: float32x2_t, b: float32x2_t, c: float32x2_t) -> float32x2_t {
    let b: float32x2_t = simd_neg(b);
    vfma_f32(a, b, c)
}

/// Floating-point fused multiply-subtract from accumulator
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "fp-armv8,v8"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vfms))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmls))]
pub unsafe fn vfmsq_f32(a: float32x4_t, b: float32x4_t, c: float32x4_t) -> float32x4_t {
    let b: float32x4_t = simd_neg(b);
    vfmaq_f32(a, b, c)
}

/// Floating-point fused Multiply-subtract to accumulator(vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "fp-armv8,v8"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vfms))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmls))]
pub unsafe fn vfms_n_f32(a: float32x2_t, b: float32x2_t, c: f32) -> float32x2_t {
    vfms_f32(a, b, vdup_n_f32(c))
}

/// Floating-point fused Multiply-subtract to accumulator(vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "fp-armv8,v8"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vfms))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmls))]
pub unsafe fn vfmsq_n_f32(a: float32x4_t, b: float32x4_t, c: f32) -> float32x4_t {
    vfmsq_f32(a, b, vdupq_n_f32(c))
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsub_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsubq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsub_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsubq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsub_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsubq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsub_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsubq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsub_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsubq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsub_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsubq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i64"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsub_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i64"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsubq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i64"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsub_u64(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.i64"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sub))]
pub unsafe fn vsubq_u64(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fsub))]
pub unsafe fn vsub_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    simd_sub(a, b)
}

/// Subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vsub.f32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fsub))]
pub unsafe fn vsubq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    simd_sub(a, b)
}

/// Subtract returning high narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubhn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(subhn))]
pub unsafe fn vsubhn_s16(a: int16x8_t, b: int16x8_t) -> int8x8_t {
    let c: i16x8 = i16x8::new(8, 8, 8, 8, 8, 8, 8, 8);
    simd_cast(simd_shr(simd_sub(a, b), transmute(c)))
}

/// Subtract returning high narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubhn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(subhn))]
pub unsafe fn vsubhn_s32(a: int32x4_t, b: int32x4_t) -> int16x4_t {
    let c: i32x4 = i32x4::new(16, 16, 16, 16);
    simd_cast(simd_shr(simd_sub(a, b), transmute(c)))
}

/// Subtract returning high narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubhn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(subhn))]
pub unsafe fn vsubhn_s64(a: int64x2_t, b: int64x2_t) -> int32x2_t {
    let c: i64x2 = i64x2::new(32, 32);
    simd_cast(simd_shr(simd_sub(a, b), transmute(c)))
}

/// Subtract returning high narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubhn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(subhn))]
pub unsafe fn vsubhn_u16(a: uint16x8_t, b: uint16x8_t) -> uint8x8_t {
    let c: u16x8 = u16x8::new(8, 8, 8, 8, 8, 8, 8, 8);
    simd_cast(simd_shr(simd_sub(a, b), transmute(c)))
}

/// Subtract returning high narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubhn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(subhn))]
pub unsafe fn vsubhn_u32(a: uint32x4_t, b: uint32x4_t) -> uint16x4_t {
    let c: u32x4 = u32x4::new(16, 16, 16, 16);
    simd_cast(simd_shr(simd_sub(a, b), transmute(c)))
}

/// Subtract returning high narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubhn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(subhn))]
pub unsafe fn vsubhn_u64(a: uint64x2_t, b: uint64x2_t) -> uint32x2_t {
    let c: u64x2 = u64x2::new(32, 32);
    simd_cast(simd_shr(simd_sub(a, b), transmute(c)))
}

/// Subtract returning high narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubhn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(subhn2))]
pub unsafe fn vsubhn_high_s16(a: int8x8_t, b: int16x8_t, c: int16x8_t) -> int8x16_t {
    let d: int8x8_t = vsubhn_s16(b, c);
    simd_shuffle16!(a, d, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Subtract returning high narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubhn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(subhn2))]
pub unsafe fn vsubhn_high_s32(a: int16x4_t, b: int32x4_t, c: int32x4_t) -> int16x8_t {
    let d: int16x4_t = vsubhn_s32(b, c);
    simd_shuffle8!(a, d, [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Subtract returning high narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubhn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(subhn2))]
pub unsafe fn vsubhn_high_s64(a: int32x2_t, b: int64x2_t, c: int64x2_t) -> int32x4_t {
    let d: int32x2_t = vsubhn_s64(b, c);
    simd_shuffle4!(a, d, [0, 1, 2, 3])
}

/// Subtract returning high narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubhn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(subhn2))]
pub unsafe fn vsubhn_high_u16(a: uint8x8_t, b: uint16x8_t, c: uint16x8_t) -> uint8x16_t {
    let d: uint8x8_t = vsubhn_u16(b, c);
    simd_shuffle16!(a, d, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
}

/// Subtract returning high narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubhn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(subhn2))]
pub unsafe fn vsubhn_high_u32(a: uint16x4_t, b: uint32x4_t, c: uint32x4_t) -> uint16x8_t {
    let d: uint16x4_t = vsubhn_u32(b, c);
    simd_shuffle8!(a, d, [0, 1, 2, 3, 4, 5, 6, 7])
}

/// Subtract returning high narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubhn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(subhn2))]
pub unsafe fn vsubhn_high_u64(a: uint32x2_t, b: uint64x2_t, c: uint64x2_t) -> uint32x4_t {
    let d: uint32x2_t = vsubhn_u64(b, c);
    simd_shuffle4!(a, d, [0, 1, 2, 3])
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhsub))]
pub unsafe fn vhsub_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhsub.v8i8")]
        fn vhsub_u8_(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    }
vhsub_u8_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhsub))]
pub unsafe fn vhsubq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubu.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhsub.v16i8")]
        fn vhsubq_u8_(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
    }
vhsubq_u8_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhsub))]
pub unsafe fn vhsub_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhsub.v4i16")]
        fn vhsub_u16_(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    }
vhsub_u16_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhsub))]
pub unsafe fn vhsubq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubu.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhsub.v8i16")]
        fn vhsubq_u16_(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t;
    }
vhsubq_u16_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhsub))]
pub unsafe fn vhsub_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhsub.v2i32")]
        fn vhsub_u32_(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    }
vhsub_u32_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uhsub))]
pub unsafe fn vhsubq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubu.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uhsub.v4i32")]
        fn vhsubq_u32_(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    }
vhsubq_u32_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shsub))]
pub unsafe fn vhsub_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubs.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shsub.v8i8")]
        fn vhsub_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vhsub_s8_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shsub))]
pub unsafe fn vhsubq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubs.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shsub.v16i8")]
        fn vhsubq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vhsubq_s8_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shsub))]
pub unsafe fn vhsub_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubs.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shsub.v4i16")]
        fn vhsub_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vhsub_s16_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shsub))]
pub unsafe fn vhsubq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubs.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shsub.v8i16")]
        fn vhsubq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vhsubq_s16_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shsub))]
pub unsafe fn vhsub_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubs.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shsub.v2i32")]
        fn vhsub_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vhsub_s32_(a, b)
}

/// Signed halving subtract
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vhsub.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shsub))]
pub unsafe fn vhsubq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vhsubs.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.shsub.v4i32")]
        fn vhsubq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vhsubq_s32_(a, b)
}

/// Signed Subtract Wide
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubw))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ssubw))]
pub unsafe fn vsubw_s8(a: int16x8_t, b: int8x8_t) -> int16x8_t {
    simd_sub(a, simd_cast(b))
}

/// Signed Subtract Wide
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubw))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ssubw))]
pub unsafe fn vsubw_s16(a: int32x4_t, b: int16x4_t) -> int32x4_t {
    simd_sub(a, simd_cast(b))
}

/// Signed Subtract Wide
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubw))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ssubw))]
pub unsafe fn vsubw_s32(a: int64x2_t, b: int32x2_t) -> int64x2_t {
    simd_sub(a, simd_cast(b))
}

/// Unsigned Subtract Wide
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubw))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(usubw))]
pub unsafe fn vsubw_u8(a: uint16x8_t, b: uint8x8_t) -> uint16x8_t {
    simd_sub(a, simd_cast(b))
}

/// Unsigned Subtract Wide
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubw))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(usubw))]
pub unsafe fn vsubw_u16(a: uint32x4_t, b: uint16x4_t) -> uint32x4_t {
    simd_sub(a, simd_cast(b))
}

/// Unsigned Subtract Wide
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubw))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(usubw))]
pub unsafe fn vsubw_u32(a: uint64x2_t, b: uint32x2_t) -> uint64x2_t {
    simd_sub(a, simd_cast(b))
}

/// Signed Subtract Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ssubl))]
pub unsafe fn vsubl_s8(a: int8x8_t, b: int8x8_t) -> int16x8_t {
    let c: int16x8_t = simd_cast(a);
    let d: int16x8_t = simd_cast(b);
    simd_sub(c, d)
}

/// Signed Subtract Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ssubl))]
pub unsafe fn vsubl_s16(a: int16x4_t, b: int16x4_t) -> int32x4_t {
    let c: int32x4_t = simd_cast(a);
    let d: int32x4_t = simd_cast(b);
    simd_sub(c, d)
}

/// Signed Subtract Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ssubl))]
pub unsafe fn vsubl_s32(a: int32x2_t, b: int32x2_t) -> int64x2_t {
    let c: int64x2_t = simd_cast(a);
    let d: int64x2_t = simd_cast(b);
    simd_sub(c, d)
}

/// Unsigned Subtract Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(usubl))]
pub unsafe fn vsubl_u8(a: uint8x8_t, b: uint8x8_t) -> uint16x8_t {
    let c: uint16x8_t = simd_cast(a);
    let d: uint16x8_t = simd_cast(b);
    simd_sub(c, d)
}

/// Unsigned Subtract Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(usubl))]
pub unsafe fn vsubl_u16(a: uint16x4_t, b: uint16x4_t) -> uint32x4_t {
    let c: uint32x4_t = simd_cast(a);
    let d: uint32x4_t = simd_cast(b);
    simd_sub(c, d)
}

/// Unsigned Subtract Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsubl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(usubl))]
pub unsafe fn vsubl_u32(a: uint32x2_t, b: uint32x2_t) -> uint64x2_t {
    let c: uint64x2_t = simd_cast(a);
    let d: uint64x2_t = simd_cast(b);
    simd_sub(c, d)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smax))]
pub unsafe fn vmax_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxs.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smax.v8i8")]
        fn vmax_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vmax_s8_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smax))]
pub unsafe fn vmaxq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxs.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smax.v16i8")]
        fn vmaxq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vmaxq_s8_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smax))]
pub unsafe fn vmax_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxs.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smax.v4i16")]
        fn vmax_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vmax_s16_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smax))]
pub unsafe fn vmaxq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxs.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smax.v8i16")]
        fn vmaxq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vmaxq_s16_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smax))]
pub unsafe fn vmax_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxs.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smax.v2i32")]
        fn vmax_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vmax_s32_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smax))]
pub unsafe fn vmaxq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxs.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smax.v4i32")]
        fn vmaxq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vmaxq_s32_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umax))]
pub unsafe fn vmax_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umax.v8i8")]
        fn vmax_u8_(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    }
vmax_u8_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umax))]
pub unsafe fn vmaxq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxu.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umax.v16i8")]
        fn vmaxq_u8_(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
    }
vmaxq_u8_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umax))]
pub unsafe fn vmax_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umax.v4i16")]
        fn vmax_u16_(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    }
vmax_u16_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umax))]
pub unsafe fn vmaxq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxu.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umax.v8i16")]
        fn vmaxq_u16_(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t;
    }
vmaxq_u16_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umax))]
pub unsafe fn vmax_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umax.v2i32")]
        fn vmax_u32_(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    }
vmax_u32_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umax))]
pub unsafe fn vmaxq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxu.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umax.v4i32")]
        fn vmaxq_u32_(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    }
vmaxq_u32_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmax))]
pub unsafe fn vmax_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxs.v2f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmax.v2f32")]
        fn vmax_f32_(a: float32x2_t, b: float32x2_t) -> float32x2_t;
    }
vmax_f32_(a, b)
}

/// Maximum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmax))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmax))]
pub unsafe fn vmaxq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxs.v4f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmax.v4f32")]
        fn vmaxq_f32_(a: float32x4_t, b: float32x4_t) -> float32x4_t;
    }
vmaxq_f32_(a, b)
}

/// Floating-point Maximun Number (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "fp-armv8,v8"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmaxnm))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmaxnm))]
pub unsafe fn vmaxnm_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxnm.v2f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmaxnm.v2f32")]
        fn vmaxnm_f32_(a: float32x2_t, b: float32x2_t) -> float32x2_t;
    }
vmaxnm_f32_(a, b)
}

/// Floating-point Maximun Number (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "fp-armv8,v8"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmaxnm))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmaxnm))]
pub unsafe fn vmaxnmq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmaxnm.v4f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmaxnm.v4f32")]
        fn vmaxnmq_f32_(a: float32x4_t, b: float32x4_t) -> float32x4_t;
    }
vmaxnmq_f32_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smin))]
pub unsafe fn vmin_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmins.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smin.v8i8")]
        fn vmin_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vmin_s8_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smin))]
pub unsafe fn vminq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmins.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smin.v16i8")]
        fn vminq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vminq_s8_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smin))]
pub unsafe fn vmin_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmins.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smin.v4i16")]
        fn vmin_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vmin_s16_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smin))]
pub unsafe fn vminq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmins.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smin.v8i16")]
        fn vminq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vminq_s16_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smin))]
pub unsafe fn vmin_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmins.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smin.v2i32")]
        fn vmin_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vmin_s32_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(smin))]
pub unsafe fn vminq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmins.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.smin.v4i32")]
        fn vminq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vminq_s32_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umin))]
pub unsafe fn vmin_u8(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vminu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umin.v8i8")]
        fn vmin_u8_(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t;
    }
vmin_u8_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umin))]
pub unsafe fn vminq_u8(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vminu.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umin.v16i8")]
        fn vminq_u8_(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t;
    }
vminq_u8_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umin))]
pub unsafe fn vmin_u16(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vminu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umin.v4i16")]
        fn vmin_u16_(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t;
    }
vmin_u16_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umin))]
pub unsafe fn vminq_u16(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vminu.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umin.v8i16")]
        fn vminq_u16_(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t;
    }
vminq_u16_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umin))]
pub unsafe fn vmin_u32(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vminu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umin.v2i32")]
        fn vmin_u32_(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t;
    }
vmin_u32_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(umin))]
pub unsafe fn vminq_u32(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vminu.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.umin.v4i32")]
        fn vminq_u32_(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t;
    }
vminq_u32_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmin))]
pub unsafe fn vmin_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmins.v2f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmin.v2f32")]
        fn vmin_f32_(a: float32x2_t, b: float32x2_t) -> float32x2_t;
    }
vmin_f32_(a, b)
}

/// Minimum (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vmin))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fmin))]
pub unsafe fn vminq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vmins.v4f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fmin.v4f32")]
        fn vminq_f32_(a: float32x4_t, b: float32x4_t) -> float32x4_t;
    }
vminq_f32_(a, b)
}

/// Floating-point Minimun Number (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "fp-armv8,v8"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vminnm))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fminnm))]
pub unsafe fn vminnm_f32(a: float32x2_t, b: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vminnm.v2f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fminnm.v2f32")]
        fn vminnm_f32_(a: float32x2_t, b: float32x2_t) -> float32x2_t;
    }
vminnm_f32_(a, b)
}

/// Floating-point Minimun Number (vector)
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "fp-armv8,v8"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vminnm))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(fminnm))]
pub unsafe fn vminnmq_f32(a: float32x4_t, b: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vminnm.v4f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.fminnm.v4f32")]
        fn vminnmq_f32_(a: float32x4_t, b: float32x4_t) -> float32x4_t;
    }
vminnmq_f32_(a, b)
}

/// Signed saturating doubling multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmull))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmull))]
pub unsafe fn vqdmull_s16(a: int16x4_t, b: int16x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqdmull.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqdmull.v4i32")]
        fn vqdmull_s16_(a: int16x4_t, b: int16x4_t) -> int32x4_t;
    }
vqdmull_s16_(a, b)
}

/// Signed saturating doubling multiply long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmull))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmull))]
pub unsafe fn vqdmull_s32(a: int32x2_t, b: int32x2_t) -> int64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqdmull.v2i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqdmull.v2i64")]
        fn vqdmull_s32_(a: int32x2_t, b: int32x2_t) -> int64x2_t;
    }
vqdmull_s32_(a, b)
}

/// Vector saturating doubling long multiply with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmull))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmull))]
pub unsafe fn vqdmull_n_s16(a: int16x4_t, b: i16) -> int32x4_t {
    vqdmull_s16(a, vdup_n_s16(b))
}

/// Vector saturating doubling long multiply with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmull))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmull))]
pub unsafe fn vqdmull_n_s32(a: int32x2_t, b: i32) -> int64x2_t {
    vqdmull_s32(a, vdup_n_s32(b))
}

/// Vector saturating doubling long multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmull, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmull, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqdmull_lane_s16<const N: i32>(a: int16x4_t, b: int16x4_t) -> int32x4_t {
    static_assert_imm2!(N);
    let b: int16x4_t = simd_shuffle4!(b, b, <const N: i32> [N as u32, N as u32, N as u32, N as u32]);
    vqdmull_s16(a, b)
}

/// Vector saturating doubling long multiply by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmull, N = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmull, N = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqdmull_lane_s32<const N: i32>(a: int32x2_t, b: int32x2_t) -> int64x2_t {
    static_assert_imm1!(N);
    let b: int32x2_t = simd_shuffle2!(b, b, <const N: i32> [N as u32, N as u32]);
    vqdmull_s32(a, b)
}

/// Signed saturating doubling multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmlal))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmlal))]
pub unsafe fn vqdmlal_s16(a: int32x4_t, b: int16x4_t, c: int16x4_t) -> int32x4_t {
    vqaddq_s32(a, vqdmull_s16(b, c))
}

/// Signed saturating doubling multiply-add long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmlal))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmlal))]
pub unsafe fn vqdmlal_s32(a: int64x2_t, b: int32x2_t, c: int32x2_t) -> int64x2_t {
    vqaddq_s64(a, vqdmull_s32(b, c))
}

/// Vector widening saturating doubling multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmlal))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmlal))]
pub unsafe fn vqdmlal_n_s16(a: int32x4_t, b: int16x4_t, c: i16) -> int32x4_t {
    vqaddq_s32(a, vqdmull_n_s16(b, c))
}

/// Vector widening saturating doubling multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmlal))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmlal))]
pub unsafe fn vqdmlal_n_s32(a: int64x2_t, b: int32x2_t, c: i32) -> int64x2_t {
    vqaddq_s64(a, vqdmull_n_s32(b, c))
}

/// Vector widening saturating doubling multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmlal, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmlal, N = 2))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqdmlal_lane_s16<const N: i32>(a: int32x4_t, b: int16x4_t, c: int16x4_t) -> int32x4_t {
    static_assert_imm2!(N);
    vqaddq_s32(a, vqdmull_lane_s16::<N>(b, c))
}

/// Vector widening saturating doubling multiply accumulate with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmlal, N = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmlal, N = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqdmlal_lane_s32<const N: i32>(a: int64x2_t, b: int32x2_t, c: int32x2_t) -> int64x2_t {
    static_assert_imm1!(N);
    vqaddq_s64(a, vqdmull_lane_s32::<N>(b, c))
}

/// Signed saturating doubling multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmlsl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmlsl))]
pub unsafe fn vqdmlsl_s16(a: int32x4_t, b: int16x4_t, c: int16x4_t) -> int32x4_t {
    vqsubq_s32(a, vqdmull_s16(b, c))
}

/// Signed saturating doubling multiply-subtract long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmlsl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmlsl))]
pub unsafe fn vqdmlsl_s32(a: int64x2_t, b: int32x2_t, c: int32x2_t) -> int64x2_t {
    vqsubq_s64(a, vqdmull_s32(b, c))
}

/// Vector widening saturating doubling multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmlsl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmlsl))]
pub unsafe fn vqdmlsl_n_s16(a: int32x4_t, b: int16x4_t, c: i16) -> int32x4_t {
    vqsubq_s32(a, vqdmull_n_s16(b, c))
}

/// Vector widening saturating doubling multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmlsl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmlsl))]
pub unsafe fn vqdmlsl_n_s32(a: int64x2_t, b: int32x2_t, c: i32) -> int64x2_t {
    vqsubq_s64(a, vqdmull_n_s32(b, c))
}

/// Vector widening saturating doubling multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmlsl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmlsl, N = 2))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqdmlsl_lane_s16<const N: i32>(a: int32x4_t, b: int16x4_t, c: int16x4_t) -> int32x4_t {
    static_assert_imm2!(N);
    vqsubq_s32(a, vqdmull_lane_s16::<N>(b, c))
}

/// Vector widening saturating doubling multiply subtract with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmlsl, N = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmlsl, N = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqdmlsl_lane_s32<const N: i32>(a: int64x2_t, b: int32x2_t, c: int32x2_t) -> int64x2_t {
    static_assert_imm1!(N);
    vqsubq_s64(a, vqdmull_lane_s32::<N>(b, c))
}

/// Signed saturating doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmulh))]
pub unsafe fn vqdmulh_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqdmulh.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqdmulh.v4i16")]
        fn vqdmulh_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vqdmulh_s16_(a, b)
}

/// Signed saturating doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmulh))]
pub unsafe fn vqdmulhq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqdmulh.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqdmulh.v8i16")]
        fn vqdmulhq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vqdmulhq_s16_(a, b)
}

/// Signed saturating doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmulh))]
pub unsafe fn vqdmulh_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqdmulh.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqdmulh.v2i32")]
        fn vqdmulh_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vqdmulh_s32_(a, b)
}

/// Signed saturating doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmulh))]
pub unsafe fn vqdmulhq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqdmulh.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqdmulh.v4i32")]
        fn vqdmulhq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vqdmulhq_s32_(a, b)
}

/// Vector saturating doubling multiply high with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmulh))]
pub unsafe fn vqdmulh_n_s16(a: int16x4_t, b: i16) -> int16x4_t {
    let b: int16x4_t = vdup_n_s16(b);
    vqdmulh_s16(a, b)
}

/// Vector saturating doubling multiply high with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmulh))]
pub unsafe fn vqdmulh_n_s32(a: int32x2_t, b: i32) -> int32x2_t {
    let b: int32x2_t = vdup_n_s32(b);
    vqdmulh_s32(a, b)
}

/// Vector saturating doubling multiply high with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmulh))]
pub unsafe fn vqdmulhq_nq_s16(a: int16x8_t, b: i16) -> int16x8_t {
    let b: int16x8_t = vdupq_n_s16(b);
    vqdmulhq_s16(a, b)
}

/// Vector saturating doubling multiply high with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqdmulh))]
pub unsafe fn vqdmulhq_nq_s32(a: int32x4_t, b: i32) -> int32x4_t {
    let b: int32x4_t = vdupq_n_s32(b);
    vqdmulhq_s32(a, b)
}

/// Signed saturating extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqmovn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqxtn))]
pub unsafe fn vqmovn_s16(a: int16x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqmovns.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqxtn.v8i8")]
        fn vqmovn_s16_(a: int16x8_t) -> int8x8_t;
    }
vqmovn_s16_(a)
}

/// Signed saturating extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqmovn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqxtn))]
pub unsafe fn vqmovn_s32(a: int32x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqmovns.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqxtn.v4i16")]
        fn vqmovn_s32_(a: int32x4_t) -> int16x4_t;
    }
vqmovn_s32_(a)
}

/// Signed saturating extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqmovn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqxtn))]
pub unsafe fn vqmovn_s64(a: int64x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqmovns.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqxtn.v2i32")]
        fn vqmovn_s64_(a: int64x2_t) -> int32x2_t;
    }
vqmovn_s64_(a)
}

/// Unsigned saturating extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqmovn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqxtn))]
pub unsafe fn vqmovn_u16(a: uint16x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqmovnu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqxtn.v8i8")]
        fn vqmovn_u16_(a: uint16x8_t) -> uint8x8_t;
    }
vqmovn_u16_(a)
}

/// Unsigned saturating extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqmovn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqxtn))]
pub unsafe fn vqmovn_u32(a: uint32x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqmovnu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqxtn.v4i16")]
        fn vqmovn_u32_(a: uint32x4_t) -> uint16x4_t;
    }
vqmovn_u32_(a)
}

/// Unsigned saturating extract narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqmovn))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqxtn))]
pub unsafe fn vqmovn_u64(a: uint64x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqmovnu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqxtn.v2i32")]
        fn vqmovn_u64_(a: uint64x2_t) -> uint32x2_t;
    }
vqmovn_u64_(a)
}

/// Signed saturating extract unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqmovun))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqxtun))]
pub unsafe fn vqmovun_s16(a: int16x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqmovnsu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqxtun.v8i8")]
        fn vqmovun_s16_(a: int16x8_t) -> uint8x8_t;
    }
vqmovun_s16_(a)
}

/// Signed saturating extract unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqmovun))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqxtun))]
pub unsafe fn vqmovun_s32(a: int32x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqmovnsu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqxtun.v4i16")]
        fn vqmovun_s32_(a: int32x4_t) -> uint16x4_t;
    }
vqmovun_s32_(a)
}

/// Signed saturating extract unsigned narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqmovun))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqxtun))]
pub unsafe fn vqmovun_s64(a: int64x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqmovnsu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqxtun.v2i32")]
        fn vqmovun_s64_(a: int64x2_t) -> uint32x2_t;
    }
vqmovun_s64_(a)
}

/// Signed saturating rounding doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh))]
pub unsafe fn vqrdmulh_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrdmulh.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrdmulh.v4i16")]
        fn vqrdmulh_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vqrdmulh_s16_(a, b)
}

/// Signed saturating rounding doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh))]
pub unsafe fn vqrdmulhq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrdmulh.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrdmulh.v8i16")]
        fn vqrdmulhq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vqrdmulhq_s16_(a, b)
}

/// Signed saturating rounding doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh))]
pub unsafe fn vqrdmulh_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrdmulh.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrdmulh.v2i32")]
        fn vqrdmulh_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vqrdmulh_s32_(a, b)
}

/// Signed saturating rounding doubling multiply returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh))]
pub unsafe fn vqrdmulhq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrdmulh.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrdmulh.v4i32")]
        fn vqrdmulhq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vqrdmulhq_s32_(a, b)
}

/// Vector saturating rounding doubling multiply high with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh))]
pub unsafe fn vqrdmulh_n_s16(a: int16x4_t, b: i16) -> int16x4_t {
    vqrdmulh_s16(a, vdup_n_s16(b))
}

/// Vector saturating rounding doubling multiply high with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh))]
pub unsafe fn vqrdmulhq_n_s16(a: int16x8_t, b: i16) -> int16x8_t {
    vqrdmulhq_s16(a, vdupq_n_s16(b))
}

/// Vector saturating rounding doubling multiply high with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh))]
pub unsafe fn vqrdmulh_n_s32(a: int32x2_t, b: i32) -> int32x2_t {
    vqrdmulh_s32(a, vdup_n_s32(b))
}

/// Vector saturating rounding doubling multiply high with scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh))]
pub unsafe fn vqrdmulhq_n_s32(a: int32x4_t, b: i32) -> int32x4_t {
    vqrdmulhq_s32(a, vdupq_n_s32(b))
}

/// Vector rounding saturating doubling multiply high by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrdmulh_lane_s16<const LANE: i32>(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    static_assert_imm2!(LANE);
    let b: int16x4_t = simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]);
    vqrdmulh_s16(a, b)
}

/// Vector rounding saturating doubling multiply high by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrdmulh_laneq_s16<const LANE: i32>(a: int16x4_t, b: int16x8_t) -> int16x4_t {
    static_assert_imm3!(LANE);
    let b: int16x4_t = simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]);
    vqrdmulh_s16(a, b)
}

/// Vector rounding saturating doubling multiply high by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrdmulhq_lane_s16<const LANE: i32>(a: int16x8_t, b: int16x4_t) -> int16x8_t {
    static_assert_imm2!(LANE);
    let b: int16x8_t = simd_shuffle8!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]);
    vqrdmulhq_s16(a, b)
}

/// Vector rounding saturating doubling multiply high by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrdmulhq_laneq_s16<const LANE: i32>(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    static_assert_imm3!(LANE);
    let b: int16x8_t = simd_shuffle8!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32, LANE as u32]);
    vqrdmulhq_s16(a, b)
}

/// Vector rounding saturating doubling multiply high by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrdmulh_lane_s32<const LANE: i32>(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    static_assert_imm1!(LANE);
    let b: int32x2_t = simd_shuffle2!(b, b, <const LANE: i32> [LANE as u32, LANE as u32]);
    vqrdmulh_s32(a, b)
}

/// Vector rounding saturating doubling multiply high by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrdmulh_laneq_s32<const LANE: i32>(a: int32x2_t, b: int32x4_t) -> int32x2_t {
    static_assert_imm2!(LANE);
    let b: int32x2_t = simd_shuffle2!(b, b, <const LANE: i32> [LANE as u32, LANE as u32]);
    vqrdmulh_s32(a, b)
}

/// Vector rounding saturating doubling multiply high by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrdmulhq_lane_s32<const LANE: i32>(a: int32x4_t, b: int32x2_t) -> int32x4_t {
    static_assert_imm1!(LANE);
    let b: int32x4_t = simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]);
    vqrdmulhq_s32(a, b)
}

/// Vector rounding saturating doubling multiply high by scalar
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vqrdmulhq_laneq_s32<const LANE: i32>(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    static_assert_imm2!(LANE);
    let b: int32x4_t = simd_shuffle4!(b, b, <const LANE: i32> [LANE as u32, LANE as u32, LANE as u32, LANE as u32]);
    vqrdmulhq_s32(a, b)
}

/// Signed saturating rounding doubling multiply accumulate returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh))]
pub unsafe fn vqrdmlah_s16(a: int16x4_t, b: int16x4_t, c: int16x4_t) -> int16x4_t {
    vqadd_s16(a, vqrdmulh_s16(b, c))
}

/// Signed saturating rounding doubling multiply accumulate returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh))]
pub unsafe fn vqrdmlahq_s16(a: int16x8_t, b: int16x8_t, c: int16x8_t) -> int16x8_t {
    vqaddq_s16(a, vqrdmulhq_s16(b, c))
}

/// Signed saturating rounding doubling multiply accumulate returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh))]
pub unsafe fn vqrdmlah_s32(a: int32x2_t, b: int32x2_t, c: int32x2_t) -> int32x2_t {
    vqadd_s32(a, vqrdmulh_s32(b, c))
}

/// Signed saturating rounding doubling multiply accumulate returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh))]
pub unsafe fn vqrdmlahq_s32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
    vqaddq_s32(a, vqrdmulhq_s32(b, c))
}

/// Signed saturating rounding doubling multiply accumulate returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlah_lane_s16<const LANE: i32>(a: int16x4_t, b: int16x4_t, c: int16x4_t) -> int16x4_t {
    static_assert_imm2!(LANE);
    vqadd_s16(a, vqrdmulh_lane_s16::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply accumulate returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlah_laneq_s16<const LANE: i32>(a: int16x4_t, b: int16x4_t, c: int16x8_t) -> int16x4_t {
    static_assert_imm3!(LANE);
    vqadd_s16(a, vqrdmulh_laneq_s16::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply accumulate returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlahq_lane_s16<const LANE: i32>(a: int16x8_t, b: int16x8_t, c: int16x4_t) -> int16x8_t {
    static_assert_imm2!(LANE);
    vqaddq_s16(a, vqrdmulhq_lane_s16::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply accumulate returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlahq_laneq_s16<const LANE: i32>(a: int16x8_t, b: int16x8_t, c: int16x8_t) -> int16x8_t {
    static_assert_imm3!(LANE);
    vqaddq_s16(a, vqrdmulhq_laneq_s16::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply accumulate returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlah_lane_s32<const LANE: i32>(a: int32x2_t, b: int32x2_t, c: int32x2_t) -> int32x2_t {
    static_assert_imm1!(LANE);
    vqadd_s32(a, vqrdmulh_lane_s32::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply accumulate returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlah_laneq_s32<const LANE: i32>(a: int32x2_t, b: int32x2_t, c: int32x4_t) -> int32x2_t {
    static_assert_imm2!(LANE);
    vqadd_s32(a, vqrdmulh_laneq_s32::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply accumulate returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlahq_lane_s32<const LANE: i32>(a: int32x4_t, b: int32x4_t, c: int32x2_t) -> int32x4_t {
    static_assert_imm1!(LANE);
    vqaddq_s32(a, vqrdmulhq_lane_s32::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply accumulate returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlahq_laneq_s32<const LANE: i32>(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
    static_assert_imm2!(LANE);
    vqaddq_s32(a, vqrdmulhq_laneq_s32::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply subtract returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh))]
pub unsafe fn vqrdmlsh_s16(a: int16x4_t, b: int16x4_t, c: int16x4_t) -> int16x4_t {
    vqsub_s16(a, vqrdmulh_s16(b, c))
}

/// Signed saturating rounding doubling multiply subtract returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh))]
pub unsafe fn vqrdmlshq_s16(a: int16x8_t, b: int16x8_t, c: int16x8_t) -> int16x8_t {
    vqsubq_s16(a, vqrdmulhq_s16(b, c))
}

/// Signed saturating rounding doubling multiply subtract returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh))]
pub unsafe fn vqrdmlsh_s32(a: int32x2_t, b: int32x2_t, c: int32x2_t) -> int32x2_t {
    vqsub_s32(a, vqrdmulh_s32(b, c))
}

/// Signed saturating rounding doubling multiply subtract returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh))]
pub unsafe fn vqrdmlshq_s32(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
    vqsubq_s32(a, vqrdmulhq_s32(b, c))
}

/// Signed saturating rounding doubling multiply subtract returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlsh_lane_s16<const LANE: i32>(a: int16x4_t, b: int16x4_t, c: int16x4_t) -> int16x4_t {
    static_assert_imm2!(LANE);
    vqsub_s16(a, vqrdmulh_lane_s16::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply subtract returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlsh_laneq_s16<const LANE: i32>(a: int16x4_t, b: int16x4_t, c: int16x8_t) -> int16x4_t {
    static_assert_imm3!(LANE);
    vqsub_s16(a, vqrdmulh_laneq_s16::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply subtract returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlshq_lane_s16<const LANE: i32>(a: int16x8_t, b: int16x8_t, c: int16x4_t) -> int16x8_t {
    static_assert_imm2!(LANE);
    vqsubq_s16(a, vqrdmulhq_lane_s16::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply subtract returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlshq_laneq_s16<const LANE: i32>(a: int16x8_t, b: int16x8_t, c: int16x8_t) -> int16x8_t {
    static_assert_imm3!(LANE);
    vqsubq_s16(a, vqrdmulhq_laneq_s16::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply subtract returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlsh_lane_s32<const LANE: i32>(a: int32x2_t, b: int32x2_t, c: int32x2_t) -> int32x2_t {
    static_assert_imm1!(LANE);
    vqsub_s32(a, vqrdmulh_lane_s32::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply subtract returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlsh_laneq_s32<const LANE: i32>(a: int32x2_t, b: int32x2_t, c: int32x4_t) -> int32x2_t {
    static_assert_imm2!(LANE);
    vqsub_s32(a, vqrdmulh_laneq_s32::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply subtract returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlshq_lane_s32<const LANE: i32>(a: int32x4_t, b: int32x4_t, c: int32x2_t) -> int32x4_t {
    static_assert_imm1!(LANE);
    vqsubq_s32(a, vqrdmulhq_lane_s32::<LANE>(b, c))
}

/// Signed saturating rounding doubling multiply subtract returning high half
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrdmulh, LANE = 1))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrdmulh, LANE = 1))]
#[rustc_legacy_const_generics(3)]
pub unsafe fn vqrdmlshq_laneq_s32<const LANE: i32>(a: int32x4_t, b: int32x4_t, c: int32x4_t) -> int32x4_t {
    static_assert_imm2!(LANE);
    vqsubq_s32(a, vqrdmulhq_laneq_s32::<LANE>(b, c))
}

/// Signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrshl))]
pub unsafe fn vqrshl_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshifts.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrshl.v8i8")]
        fn vqrshl_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vqrshl_s8_(a, b)
}

/// Signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrshl))]
pub unsafe fn vqrshlq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshifts.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrshl.v16i8")]
        fn vqrshlq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vqrshlq_s8_(a, b)
}

/// Signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrshl))]
pub unsafe fn vqrshl_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshifts.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrshl.v4i16")]
        fn vqrshl_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vqrshl_s16_(a, b)
}

/// Signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrshl))]
pub unsafe fn vqrshlq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshifts.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrshl.v8i16")]
        fn vqrshlq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vqrshlq_s16_(a, b)
}

/// Signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrshl))]
pub unsafe fn vqrshl_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshifts.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrshl.v2i32")]
        fn vqrshl_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vqrshl_s32_(a, b)
}

/// Signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrshl))]
pub unsafe fn vqrshlq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshifts.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrshl.v4i32")]
        fn vqrshlq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vqrshlq_s32_(a, b)
}

/// Signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrshl))]
pub unsafe fn vqrshl_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshifts.v1i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrshl.v1i64")]
        fn vqrshl_s64_(a: int64x1_t, b: int64x1_t) -> int64x1_t;
    }
vqrshl_s64_(a, b)
}

/// Signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrshl))]
pub unsafe fn vqrshlq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshifts.v2i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrshl.v2i64")]
        fn vqrshlq_s64_(a: int64x2_t, b: int64x2_t) -> int64x2_t;
    }
vqrshlq_s64_(a, b)
}

/// Unsigned signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqrshl))]
pub unsafe fn vqrshl_u8(a: uint8x8_t, b: int8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshiftu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqrshl.v8i8")]
        fn vqrshl_u8_(a: uint8x8_t, b: int8x8_t) -> uint8x8_t;
    }
vqrshl_u8_(a, b)
}

/// Unsigned signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqrshl))]
pub unsafe fn vqrshlq_u8(a: uint8x16_t, b: int8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshiftu.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqrshl.v16i8")]
        fn vqrshlq_u8_(a: uint8x16_t, b: int8x16_t) -> uint8x16_t;
    }
vqrshlq_u8_(a, b)
}

/// Unsigned signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqrshl))]
pub unsafe fn vqrshl_u16(a: uint16x4_t, b: int16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshiftu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqrshl.v4i16")]
        fn vqrshl_u16_(a: uint16x4_t, b: int16x4_t) -> uint16x4_t;
    }
vqrshl_u16_(a, b)
}

/// Unsigned signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqrshl))]
pub unsafe fn vqrshlq_u16(a: uint16x8_t, b: int16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshiftu.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqrshl.v8i16")]
        fn vqrshlq_u16_(a: uint16x8_t, b: int16x8_t) -> uint16x8_t;
    }
vqrshlq_u16_(a, b)
}

/// Unsigned signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqrshl))]
pub unsafe fn vqrshl_u32(a: uint32x2_t, b: int32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshiftu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqrshl.v2i32")]
        fn vqrshl_u32_(a: uint32x2_t, b: int32x2_t) -> uint32x2_t;
    }
vqrshl_u32_(a, b)
}

/// Unsigned signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqrshl))]
pub unsafe fn vqrshlq_u32(a: uint32x4_t, b: int32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshiftu.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqrshl.v4i32")]
        fn vqrshlq_u32_(a: uint32x4_t, b: int32x4_t) -> uint32x4_t;
    }
vqrshlq_u32_(a, b)
}

/// Unsigned signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqrshl))]
pub unsafe fn vqrshl_u64(a: uint64x1_t, b: int64x1_t) -> uint64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshiftu.v1i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqrshl.v1i64")]
        fn vqrshl_u64_(a: uint64x1_t, b: int64x1_t) -> uint64x1_t;
    }
vqrshl_u64_(a, b)
}

/// Unsigned signed saturating rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqrshl))]
pub unsafe fn vqrshlq_u64(a: uint64x2_t, b: int64x2_t) -> uint64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshiftu.v2i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqrshl.v2i64")]
        fn vqrshlq_u64_(a: uint64x2_t, b: int64x2_t) -> uint64x2_t;
    }
vqrshlq_u64_(a, b)
}

/// Signed saturating rounded shift right narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrn_n_s16<const N: i32>(a: int16x8_t) -> int8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshiftns.v8i8")]
        fn vqrshrn_n_s16_(a: int16x8_t, n: int16x8_t) -> int8x8_t;
    }
vqrshrn_n_s16_(a, int16x8_t(-N as i16, -N as i16, -N as i16, -N as i16, -N as i16, -N as i16, -N as i16, -N as i16))
}

/// Signed saturating rounded shift right narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrn_n_s16<const N: i32>(a: int16x8_t) -> int8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrshrn.v8i8")]
        fn vqrshrn_n_s16_(a: int16x8_t, n: i32) -> int8x8_t;
    }
vqrshrn_n_s16_(a, N)
}

/// Signed saturating rounded shift right narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrn_n_s32<const N: i32>(a: int32x4_t) -> int16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshiftns.v4i16")]
        fn vqrshrn_n_s32_(a: int32x4_t, n: int32x4_t) -> int16x4_t;
    }
vqrshrn_n_s32_(a, int32x4_t(-N as i32, -N as i32, -N as i32, -N as i32))
}

/// Signed saturating rounded shift right narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrn_n_s32<const N: i32>(a: int32x4_t) -> int16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrshrn.v4i16")]
        fn vqrshrn_n_s32_(a: int32x4_t, n: i32) -> int16x4_t;
    }
vqrshrn_n_s32_(a, N)
}

/// Signed saturating rounded shift right narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrn_n_s64<const N: i32>(a: int64x2_t) -> int32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshiftns.v2i32")]
        fn vqrshrn_n_s64_(a: int64x2_t, n: int64x2_t) -> int32x2_t;
    }
vqrshrn_n_s64_(a, int64x2_t(-N as i64, -N as i64))
}

/// Signed saturating rounded shift right narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrn_n_s64<const N: i32>(a: int64x2_t) -> int32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrshrn.v2i32")]
        fn vqrshrn_n_s64_(a: int64x2_t, n: i32) -> int32x2_t;
    }
vqrshrn_n_s64_(a, N)
}

/// Unsigned signed saturating rounded shift right narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrn_n_u16<const N: i32>(a: uint16x8_t) -> uint8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshiftnu.v8i8")]
        fn vqrshrn_n_u16_(a: uint16x8_t, n: uint16x8_t) -> uint8x8_t;
    }
vqrshrn_n_u16_(a, uint16x8_t(-N as u16, -N as u16, -N as u16, -N as u16, -N as u16, -N as u16, -N as u16, -N as u16))
}

/// Unsigned signed saturating rounded shift right narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrn_n_u16<const N: i32>(a: uint16x8_t) -> uint8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqrshrn.v8i8")]
        fn vqrshrn_n_u16_(a: uint16x8_t, n: i32) -> uint8x8_t;
    }
vqrshrn_n_u16_(a, N)
}

/// Unsigned signed saturating rounded shift right narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrn_n_u32<const N: i32>(a: uint32x4_t) -> uint16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshiftnu.v4i16")]
        fn vqrshrn_n_u32_(a: uint32x4_t, n: uint32x4_t) -> uint16x4_t;
    }
vqrshrn_n_u32_(a, uint32x4_t(-N as u32, -N as u32, -N as u32, -N as u32))
}

/// Unsigned signed saturating rounded shift right narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrn_n_u32<const N: i32>(a: uint32x4_t) -> uint16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqrshrn.v4i16")]
        fn vqrshrn_n_u32_(a: uint32x4_t, n: i32) -> uint16x4_t;
    }
vqrshrn_n_u32_(a, N)
}

/// Unsigned signed saturating rounded shift right narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrn_n_u64<const N: i32>(a: uint64x2_t) -> uint32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshiftnu.v2i32")]
        fn vqrshrn_n_u64_(a: uint64x2_t, n: uint64x2_t) -> uint32x2_t;
    }
vqrshrn_n_u64_(a, uint64x2_t(-N as u64, -N as u64))
}

/// Unsigned signed saturating rounded shift right narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrn_n_u64<const N: i32>(a: uint64x2_t) -> uint32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqrshrn.v2i32")]
        fn vqrshrn_n_u64_(a: uint64x2_t, n: i32) -> uint32x2_t;
    }
vqrshrn_n_u64_(a, N)
}

/// Signed saturating rounded shift right unsigned narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrun_n_s16<const N: i32>(a: int16x8_t) -> uint8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshiftnsu.v8i8")]
        fn vqrshrun_n_s16_(a: int16x8_t, n: int16x8_t) -> uint8x8_t;
    }
vqrshrun_n_s16_(a, int16x8_t(-N as i16, -N as i16, -N as i16, -N as i16, -N as i16, -N as i16, -N as i16, -N as i16))
}

/// Signed saturating rounded shift right unsigned narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrun_n_s16<const N: i32>(a: int16x8_t) -> uint8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrshrun.v8i8")]
        fn vqrshrun_n_s16_(a: int16x8_t, n: i32) -> uint8x8_t;
    }
vqrshrun_n_s16_(a, N)
}

/// Signed saturating rounded shift right unsigned narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrun_n_s32<const N: i32>(a: int32x4_t) -> uint16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshiftnsu.v4i16")]
        fn vqrshrun_n_s32_(a: int32x4_t, n: int32x4_t) -> uint16x4_t;
    }
vqrshrun_n_s32_(a, int32x4_t(-N as i32, -N as i32, -N as i32, -N as i32))
}

/// Signed saturating rounded shift right unsigned narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrun_n_s32<const N: i32>(a: int32x4_t) -> uint16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrshrun.v4i16")]
        fn vqrshrun_n_s32_(a: int32x4_t, n: i32) -> uint16x4_t;
    }
vqrshrun_n_s32_(a, N)
}

/// Signed saturating rounded shift right unsigned narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqrshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrun_n_s64<const N: i32>(a: int64x2_t) -> uint32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqrshiftnsu.v2i32")]
        fn vqrshrun_n_s64_(a: int64x2_t, n: int64x2_t) -> uint32x2_t;
    }
vqrshrun_n_s64_(a, int64x2_t(-N as i64, -N as i64))
}

/// Signed saturating rounded shift right unsigned narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqrshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqrshrun_n_s64<const N: i32>(a: int64x2_t) -> uint32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqrshrun.v2i32")]
        fn vqrshrun_n_s64_(a: int64x2_t, n: i32) -> uint32x2_t;
    }
vqrshrun_n_s64_(a, N)
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshl))]
pub unsafe fn vqshl_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshifts.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqshl.v8i8")]
        fn vqshl_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vqshl_s8_(a, b)
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshl))]
pub unsafe fn vqshlq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshifts.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqshl.v16i8")]
        fn vqshlq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vqshlq_s8_(a, b)
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshl))]
pub unsafe fn vqshl_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshifts.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqshl.v4i16")]
        fn vqshl_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vqshl_s16_(a, b)
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshl))]
pub unsafe fn vqshlq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshifts.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqshl.v8i16")]
        fn vqshlq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vqshlq_s16_(a, b)
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshl))]
pub unsafe fn vqshl_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshifts.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqshl.v2i32")]
        fn vqshl_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vqshl_s32_(a, b)
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshl))]
pub unsafe fn vqshlq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshifts.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqshl.v4i32")]
        fn vqshlq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vqshlq_s32_(a, b)
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshl))]
pub unsafe fn vqshl_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshifts.v1i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqshl.v1i64")]
        fn vqshl_s64_(a: int64x1_t, b: int64x1_t) -> int64x1_t;
    }
vqshl_s64_(a, b)
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshl))]
pub unsafe fn vqshlq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshifts.v2i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqshl.v2i64")]
        fn vqshlq_s64_(a: int64x2_t, b: int64x2_t) -> int64x2_t;
    }
vqshlq_s64_(a, b)
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqshl))]
pub unsafe fn vqshl_u8(a: uint8x8_t, b: int8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshiftu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqshl.v8i8")]
        fn vqshl_u8_(a: uint8x8_t, b: int8x8_t) -> uint8x8_t;
    }
vqshl_u8_(a, b)
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqshl))]
pub unsafe fn vqshlq_u8(a: uint8x16_t, b: int8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshiftu.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqshl.v16i8")]
        fn vqshlq_u8_(a: uint8x16_t, b: int8x16_t) -> uint8x16_t;
    }
vqshlq_u8_(a, b)
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqshl))]
pub unsafe fn vqshl_u16(a: uint16x4_t, b: int16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshiftu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqshl.v4i16")]
        fn vqshl_u16_(a: uint16x4_t, b: int16x4_t) -> uint16x4_t;
    }
vqshl_u16_(a, b)
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqshl))]
pub unsafe fn vqshlq_u16(a: uint16x8_t, b: int16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshiftu.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqshl.v8i16")]
        fn vqshlq_u16_(a: uint16x8_t, b: int16x8_t) -> uint16x8_t;
    }
vqshlq_u16_(a, b)
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqshl))]
pub unsafe fn vqshl_u32(a: uint32x2_t, b: int32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshiftu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqshl.v2i32")]
        fn vqshl_u32_(a: uint32x2_t, b: int32x2_t) -> uint32x2_t;
    }
vqshl_u32_(a, b)
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqshl))]
pub unsafe fn vqshlq_u32(a: uint32x4_t, b: int32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshiftu.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqshl.v4i32")]
        fn vqshlq_u32_(a: uint32x4_t, b: int32x4_t) -> uint32x4_t;
    }
vqshlq_u32_(a, b)
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqshl))]
pub unsafe fn vqshl_u64(a: uint64x1_t, b: int64x1_t) -> uint64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshiftu.v1i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqshl.v1i64")]
        fn vqshl_u64_(a: uint64x1_t, b: int64x1_t) -> uint64x1_t;
    }
vqshl_u64_(a, b)
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqshl))]
pub unsafe fn vqshlq_u64(a: uint64x2_t, b: int64x2_t) -> uint64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshiftu.v2i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqshl.v2i64")]
        fn vqshlq_u64_(a: uint64x2_t, b: int64x2_t) -> uint64x2_t;
    }
vqshlq_u64_(a, b)
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshl_n_s8<const N: i32>(a: int8x8_t) -> int8x8_t {
    static_assert_imm3!(N);
    vqshl_s8(a, vdup_n_s8(N.try_into().unwrap()))
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshlq_n_s8<const N: i32>(a: int8x16_t) -> int8x16_t {
    static_assert_imm3!(N);
    vqshlq_s8(a, vdupq_n_s8(N.try_into().unwrap()))
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshl_n_s16<const N: i32>(a: int16x4_t) -> int16x4_t {
    static_assert_imm4!(N);
    vqshl_s16(a, vdup_n_s16(N.try_into().unwrap()))
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshlq_n_s16<const N: i32>(a: int16x8_t) -> int16x8_t {
    static_assert_imm4!(N);
    vqshlq_s16(a, vdupq_n_s16(N.try_into().unwrap()))
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshl_n_s32<const N: i32>(a: int32x2_t) -> int32x2_t {
    static_assert_imm5!(N);
    vqshl_s32(a, vdup_n_s32(N.try_into().unwrap()))
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshlq_n_s32<const N: i32>(a: int32x4_t) -> int32x4_t {
    static_assert_imm5!(N);
    vqshlq_s32(a, vdupq_n_s32(N.try_into().unwrap()))
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshl_n_s64<const N: i32>(a: int64x1_t) -> int64x1_t {
    static_assert_imm6!(N);
    vqshl_s64(a, vdup_n_s64(N.try_into().unwrap()))
}

/// Signed saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshlq_n_s64<const N: i32>(a: int64x2_t) -> int64x2_t {
    static_assert_imm6!(N);
    vqshlq_s64(a, vdupq_n_s64(N.try_into().unwrap()))
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshl_n_u8<const N: i32>(a: uint8x8_t) -> uint8x8_t {
    static_assert_imm3!(N);
    vqshl_u8(a, vdup_n_s8(N.try_into().unwrap()))
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshlq_n_u8<const N: i32>(a: uint8x16_t) -> uint8x16_t {
    static_assert_imm3!(N);
    vqshlq_u8(a, vdupq_n_s8(N.try_into().unwrap()))
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshl_n_u16<const N: i32>(a: uint16x4_t) -> uint16x4_t {
    static_assert_imm4!(N);
    vqshl_u16(a, vdup_n_s16(N.try_into().unwrap()))
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshlq_n_u16<const N: i32>(a: uint16x8_t) -> uint16x8_t {
    static_assert_imm4!(N);
    vqshlq_u16(a, vdupq_n_s16(N.try_into().unwrap()))
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshl_n_u32<const N: i32>(a: uint32x2_t) -> uint32x2_t {
    static_assert_imm5!(N);
    vqshl_u32(a, vdup_n_s32(N.try_into().unwrap()))
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshlq_n_u32<const N: i32>(a: uint32x4_t) -> uint32x4_t {
    static_assert_imm5!(N);
    vqshlq_u32(a, vdupq_n_s32(N.try_into().unwrap()))
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshl_n_u64<const N: i32>(a: uint64x1_t) -> uint64x1_t {
    static_assert_imm6!(N);
    vqshl_u64(a, vdup_n_s64(N.try_into().unwrap()))
}

/// Unsigned saturating shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqshl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshlq_n_u64<const N: i32>(a: uint64x2_t) -> uint64x2_t {
    static_assert_imm6!(N);
    vqshlq_u64(a, vdupq_n_s64(N.try_into().unwrap()))
}

/// Signed saturating shift right narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrn_n_s16<const N: i32>(a: int16x8_t) -> int8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshiftns.v8i8")]
        fn vqshrn_n_s16_(a: int16x8_t, n: int16x8_t) -> int8x8_t;
    }
vqshrn_n_s16_(a, int16x8_t(-N as i16, -N as i16, -N as i16, -N as i16, -N as i16, -N as i16, -N as i16, -N as i16))
}

/// Signed saturating shift right narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrn_n_s16<const N: i32>(a: int16x8_t) -> int8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqshrn.v8i8")]
        fn vqshrn_n_s16_(a: int16x8_t, n: i32) -> int8x8_t;
    }
vqshrn_n_s16_(a, N)
}

/// Signed saturating shift right narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrn_n_s32<const N: i32>(a: int32x4_t) -> int16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshiftns.v4i16")]
        fn vqshrn_n_s32_(a: int32x4_t, n: int32x4_t) -> int16x4_t;
    }
vqshrn_n_s32_(a, int32x4_t(-N as i32, -N as i32, -N as i32, -N as i32))
}

/// Signed saturating shift right narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrn_n_s32<const N: i32>(a: int32x4_t) -> int16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqshrn.v4i16")]
        fn vqshrn_n_s32_(a: int32x4_t, n: i32) -> int16x4_t;
    }
vqshrn_n_s32_(a, N)
}

/// Signed saturating shift right narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrn_n_s64<const N: i32>(a: int64x2_t) -> int32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshiftns.v2i32")]
        fn vqshrn_n_s64_(a: int64x2_t, n: int64x2_t) -> int32x2_t;
    }
vqshrn_n_s64_(a, int64x2_t(-N as i64, -N as i64))
}

/// Signed saturating shift right narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrn_n_s64<const N: i32>(a: int64x2_t) -> int32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqshrn.v2i32")]
        fn vqshrn_n_s64_(a: int64x2_t, n: i32) -> int32x2_t;
    }
vqshrn_n_s64_(a, N)
}

/// Unsigned saturating shift right narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrn_n_u16<const N: i32>(a: uint16x8_t) -> uint8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshiftnu.v8i8")]
        fn vqshrn_n_u16_(a: uint16x8_t, n: uint16x8_t) -> uint8x8_t;
    }
vqshrn_n_u16_(a, uint16x8_t(-N as u16, -N as u16, -N as u16, -N as u16, -N as u16, -N as u16, -N as u16, -N as u16))
}

/// Unsigned saturating shift right narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrn_n_u16<const N: i32>(a: uint16x8_t) -> uint8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqshrn.v8i8")]
        fn vqshrn_n_u16_(a: uint16x8_t, n: i32) -> uint8x8_t;
    }
vqshrn_n_u16_(a, N)
}

/// Unsigned saturating shift right narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrn_n_u32<const N: i32>(a: uint32x4_t) -> uint16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshiftnu.v4i16")]
        fn vqshrn_n_u32_(a: uint32x4_t, n: uint32x4_t) -> uint16x4_t;
    }
vqshrn_n_u32_(a, uint32x4_t(-N as u32, -N as u32, -N as u32, -N as u32))
}

/// Unsigned saturating shift right narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrn_n_u32<const N: i32>(a: uint32x4_t) -> uint16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqshrn.v4i16")]
        fn vqshrn_n_u32_(a: uint32x4_t, n: i32) -> uint16x4_t;
    }
vqshrn_n_u32_(a, N)
}

/// Unsigned saturating shift right narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrn_n_u64<const N: i32>(a: uint64x2_t) -> uint32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshiftnu.v2i32")]
        fn vqshrn_n_u64_(a: uint64x2_t, n: uint64x2_t) -> uint32x2_t;
    }
vqshrn_n_u64_(a, uint64x2_t(-N as u64, -N as u64))
}

/// Unsigned saturating shift right narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uqshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrn_n_u64<const N: i32>(a: uint64x2_t) -> uint32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.uqshrn.v2i32")]
        fn vqshrn_n_u64_(a: uint64x2_t, n: i32) -> uint32x2_t;
    }
vqshrn_n_u64_(a, N)
}

/// Signed saturating shift right unsigned narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrun_n_s16<const N: i32>(a: int16x8_t) -> uint8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshiftnsu.v8i8")]
        fn vqshrun_n_s16_(a: int16x8_t, n: int16x8_t) -> uint8x8_t;
    }
vqshrun_n_s16_(a, int16x8_t(-N as i16, -N as i16, -N as i16, -N as i16, -N as i16, -N as i16, -N as i16, -N as i16))
}

/// Signed saturating shift right unsigned narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrun_n_s16<const N: i32>(a: int16x8_t) -> uint8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqshrun.v8i8")]
        fn vqshrun_n_s16_(a: int16x8_t, n: i32) -> uint8x8_t;
    }
vqshrun_n_s16_(a, N)
}

/// Signed saturating shift right unsigned narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrun_n_s32<const N: i32>(a: int32x4_t) -> uint16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshiftnsu.v4i16")]
        fn vqshrun_n_s32_(a: int32x4_t, n: int32x4_t) -> uint16x4_t;
    }
vqshrun_n_s32_(a, int32x4_t(-N as i32, -N as i32, -N as i32, -N as i32))
}

/// Signed saturating shift right unsigned narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrun_n_s32<const N: i32>(a: int32x4_t) -> uint16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqshrun.v4i16")]
        fn vqshrun_n_s32_(a: int32x4_t, n: i32) -> uint16x4_t;
    }
vqshrun_n_s32_(a, N)
}

/// Signed saturating shift right unsigned narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vqshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrun_n_s64<const N: i32>(a: int64x2_t) -> uint32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqshiftnsu.v2i32")]
        fn vqshrun_n_s64_(a: int64x2_t, n: int64x2_t) -> uint32x2_t;
    }
vqshrun_n_s64_(a, int64x2_t(-N as i64, -N as i64))
}

/// Signed saturating shift right unsigned narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqshrun, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vqshrun_n_s64<const N: i32>(a: int64x2_t) -> uint32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqshrun.v2i32")]
        fn vqshrun_n_s64_(a: int64x2_t, n: i32) -> uint32x2_t;
    }
vqshrun_n_s64_(a, N)
}

/// Reciprocal square-root estimate.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsqrte))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(frsqrte))]
pub unsafe fn vrsqrte_f32(a: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrsqrte.v2f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.frsqrte.v2f32")]
        fn vrsqrte_f32_(a: float32x2_t) -> float32x2_t;
    }
vrsqrte_f32_(a)
}

/// Reciprocal square-root estimate.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsqrte))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(frsqrte))]
pub unsafe fn vrsqrteq_f32(a: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrsqrte.v4f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.frsqrte.v4f32")]
        fn vrsqrteq_f32_(a: float32x4_t) -> float32x4_t;
    }
vrsqrteq_f32_(a)
}

/// Reciprocal estimate.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrecpe))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(frecpe))]
pub unsafe fn vrecpe_f32(a: float32x2_t) -> float32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrecpe.v2f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.frecpe.v2f32")]
        fn vrecpe_f32_(a: float32x2_t) -> float32x2_t;
    }
vrecpe_f32_(a)
}

/// Reciprocal estimate.
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrecpe))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(frecpe))]
pub unsafe fn vrecpeq_f32(a: float32x4_t) -> float32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrecpe.v4f32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.frecpe.v4f32")]
        fn vrecpeq_f32_(a: float32x4_t) -> float32x4_t;
    }
vrecpeq_f32_(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s8_u8(a: uint8x8_t) -> int8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s8_p8(a: poly8x8_t) -> int8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s16_p16(a: poly16x4_t) -> int16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s16_u16(a: uint16x4_t) -> int16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s32_u32(a: uint32x2_t) -> int32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s64_u64(a: uint64x1_t) -> int64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s8_u8(a: uint8x16_t) -> int8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s8_p8(a: poly8x16_t) -> int8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s16_p16(a: poly16x8_t) -> int16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s16_u16(a: uint16x8_t) -> int16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s32_u32(a: uint32x4_t) -> int32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s64_u64(a: uint64x2_t) -> int64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u8_p8(a: poly8x8_t) -> uint8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u8_s8(a: int8x8_t) -> uint8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u16_p16(a: poly16x4_t) -> uint16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u16_s16(a: int16x4_t) -> uint16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u32_s32(a: int32x2_t) -> uint32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u64_s64(a: int64x1_t) -> uint64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u8_p8(a: poly8x16_t) -> uint8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u8_s8(a: int8x16_t) -> uint8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u16_p16(a: poly16x8_t) -> uint16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u16_s16(a: int16x8_t) -> uint16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u32_s32(a: int32x4_t) -> uint32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u64_s64(a: int64x2_t) -> uint64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p8_s8(a: int8x8_t) -> poly8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p8_u8(a: uint8x8_t) -> poly8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p16_s16(a: int16x4_t) -> poly16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p16_u16(a: uint16x4_t) -> poly16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p8_s8(a: int8x16_t) -> poly8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p8_u8(a: uint8x16_t) -> poly8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p16_s16(a: int16x8_t) -> poly16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p16_u16(a: uint16x8_t) -> poly16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s8_s16(a: int16x4_t) -> int8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s8_u16(a: uint16x4_t) -> int8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s8_p16(a: poly16x4_t) -> int8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s16_s32(a: int32x2_t) -> int16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s16_u32(a: uint32x2_t) -> int16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s32_s64(a: int64x1_t) -> int32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s32_u64(a: uint64x1_t) -> int32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s8_s16(a: int16x8_t) -> int8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s8_u16(a: uint16x8_t) -> int8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s8_p16(a: poly16x8_t) -> int8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s16_s32(a: int32x4_t) -> int16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s16_u32(a: uint32x4_t) -> int16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s32_s64(a: int64x2_t) -> int32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s32_u64(a: uint64x2_t) -> int32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u8_p16(a: poly16x4_t) -> uint8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u8_s16(a: int16x4_t) -> uint8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u8_u16(a: uint16x4_t) -> uint8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u16_s32(a: int32x2_t) -> uint16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u16_u32(a: uint32x2_t) -> uint16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u32_s64(a: int64x1_t) -> uint32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u32_u64(a: uint64x1_t) -> uint32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u8_p16(a: poly16x8_t) -> uint8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u8_s16(a: int16x8_t) -> uint8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u8_u16(a: uint16x8_t) -> uint8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u16_s32(a: int32x4_t) -> uint16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u16_u32(a: uint32x4_t) -> uint16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u32_s64(a: int64x2_t) -> uint32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u32_u64(a: uint64x2_t) -> uint32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p8_p16(a: poly16x4_t) -> poly8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p8_s16(a: int16x4_t) -> poly8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p8_u16(a: uint16x4_t) -> poly8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p16_s32(a: int32x2_t) -> poly16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p16_u32(a: uint32x2_t) -> poly16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p8_p16(a: poly16x8_t) -> poly8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p8_s16(a: int16x8_t) -> poly8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p8_u16(a: uint16x8_t) -> poly8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p16_s32(a: int32x4_t) -> poly16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p16_u32(a: uint32x4_t) -> poly16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s16_p8(a: poly8x8_t) -> int16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s16_s8(a: int8x8_t) -> int16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s16_u8(a: uint8x8_t) -> int16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s32_p16(a: poly16x4_t) -> int32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s32_s16(a: int16x4_t) -> int32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s32_u16(a: uint16x4_t) -> int32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s64_s32(a: int32x2_t) -> int64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s64_u32(a: uint32x2_t) -> int64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s16_p8(a: poly8x16_t) -> int16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s16_s8(a: int8x16_t) -> int16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s16_u8(a: uint8x16_t) -> int16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s32_p16(a: poly16x8_t) -> int32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s32_s16(a: int16x8_t) -> int32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s32_u16(a: uint16x8_t) -> int32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s64_s32(a: int32x4_t) -> int64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s64_u32(a: uint32x4_t) -> int64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u16_p8(a: poly8x8_t) -> uint16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u16_s8(a: int8x8_t) -> uint16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u16_u8(a: uint8x8_t) -> uint16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u32_p16(a: poly16x4_t) -> uint32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u32_s16(a: int16x4_t) -> uint32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u32_u16(a: uint16x4_t) -> uint32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u64_s32(a: int32x2_t) -> uint64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u64_u32(a: uint32x2_t) -> uint64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u16_p8(a: poly8x16_t) -> uint16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u16_s8(a: int8x16_t) -> uint16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u16_u8(a: uint8x16_t) -> uint16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u32_p16(a: poly16x8_t) -> uint32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u32_s16(a: int16x8_t) -> uint32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u32_u16(a: uint16x8_t) -> uint32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u64_s32(a: int32x4_t) -> uint64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u64_u32(a: uint32x4_t) -> uint64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p16_p8(a: poly8x8_t) -> poly16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p16_s8(a: int8x8_t) -> poly16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p16_u8(a: uint8x8_t) -> poly16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p16_p8(a: poly8x16_t) -> poly16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p16_s8(a: int8x16_t) -> poly16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p16_u8(a: uint8x16_t) -> poly16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s8_s32(a: int32x2_t) -> int8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s8_u32(a: uint32x2_t) -> int8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s16_s64(a: int64x1_t) -> int16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s16_u64(a: uint64x1_t) -> int16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s8_s32(a: int32x4_t) -> int8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s8_u32(a: uint32x4_t) -> int8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s16_s64(a: int64x2_t) -> int16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s16_u64(a: uint64x2_t) -> int16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u8_s32(a: int32x2_t) -> uint8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u8_u32(a: uint32x2_t) -> uint8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u16_s64(a: int64x1_t) -> uint16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u16_u64(a: uint64x1_t) -> uint16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u8_s32(a: int32x4_t) -> uint8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u8_u32(a: uint32x4_t) -> uint8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u16_s64(a: int64x2_t) -> uint16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u16_u64(a: uint64x2_t) -> uint16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p8_s32(a: int32x2_t) -> poly8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p8_u32(a: uint32x2_t) -> poly8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p16_s64(a: int64x1_t) -> poly16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p16_u64(a: uint64x1_t) -> poly16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p8_s32(a: int32x4_t) -> poly8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p8_u32(a: uint32x4_t) -> poly8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p16_s64(a: int64x2_t) -> poly16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p16_u64(a: uint64x2_t) -> poly16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s32_p8(a: poly8x8_t) -> int32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s32_s8(a: int8x8_t) -> int32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s32_u8(a: uint8x8_t) -> int32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s64_p16(a: poly16x4_t) -> int64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s64_s16(a: int16x4_t) -> int64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s64_u16(a: uint16x4_t) -> int64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s32_p8(a: poly8x16_t) -> int32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s32_s8(a: int8x16_t) -> int32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s32_u8(a: uint8x16_t) -> int32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s64_p16(a: poly16x8_t) -> int64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s64_s16(a: int16x8_t) -> int64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s64_u16(a: uint16x8_t) -> int64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u32_p8(a: poly8x8_t) -> uint32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u32_s8(a: int8x8_t) -> uint32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u32_u8(a: uint8x8_t) -> uint32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u64_p16(a: poly16x4_t) -> uint64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u64_s16(a: int16x4_t) -> uint64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u64_u16(a: uint16x4_t) -> uint64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u32_p8(a: poly8x16_t) -> uint32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u32_s8(a: int8x16_t) -> uint32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u32_u8(a: uint8x16_t) -> uint32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u64_p16(a: poly16x8_t) -> uint64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u64_s16(a: int16x8_t) -> uint64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u64_u16(a: uint16x8_t) -> uint64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s8_s64(a: int64x1_t) -> int8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s8_u64(a: uint64x1_t) -> int8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u8_s64(a: int64x1_t) -> uint8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u8_u64(a: uint64x1_t) -> uint8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p8_s64(a: int64x1_t) -> poly8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p8_u64(a: uint64x1_t) -> poly8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s8_s64(a: int64x2_t) -> int8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s8_u64(a: uint64x2_t) -> int8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u8_s64(a: int64x2_t) -> uint8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u8_u64(a: uint64x2_t) -> uint8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p8_s64(a: int64x2_t) -> poly8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p8_u64(a: uint64x2_t) -> poly8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s64_p8(a: poly8x8_t) -> int64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s64_s8(a: int8x8_t) -> int64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s64_u8(a: uint8x8_t) -> int64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u64_p8(a: poly8x8_t) -> uint64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u64_s8(a: int8x8_t) -> uint64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u64_u8(a: uint8x8_t) -> uint64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s64_p8(a: poly8x16_t) -> int64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s64_s8(a: int8x16_t) -> int64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s64_u8(a: uint8x16_t) -> int64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u64_p8(a: poly8x16_t) -> uint64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u64_s8(a: int8x16_t) -> uint64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u64_u8(a: uint8x16_t) -> uint64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s8_f32(a: float32x2_t) -> int8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s16_f32(a: float32x2_t) -> int16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s32_f32(a: float32x2_t) -> int32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_s64_f32(a: float32x2_t) -> int64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s8_f32(a: float32x4_t) -> int8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s16_f32(a: float32x4_t) -> int16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s32_f32(a: float32x4_t) -> int32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_s64_f32(a: float32x4_t) -> int64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u8_f32(a: float32x2_t) -> uint8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u16_f32(a: float32x2_t) -> uint16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u32_f32(a: float32x2_t) -> uint32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_u64_f32(a: float32x2_t) -> uint64x1_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u8_f32(a: float32x4_t) -> uint8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u16_f32(a: float32x4_t) -> uint16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u32_f32(a: float32x4_t) -> uint32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_u64_f32(a: float32x4_t) -> uint64x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p8_f32(a: float32x2_t) -> poly8x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_p16_f32(a: float32x2_t) -> poly16x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p8_f32(a: float32x4_t) -> poly8x16_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_p16_f32(a: float32x4_t) -> poly16x8_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_f32_s8(a: int8x8_t) -> float32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_f32_s16(a: int16x4_t) -> float32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_f32_s32(a: int32x2_t) -> float32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_f32_s64(a: int64x1_t) -> float32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_f32_s8(a: int8x16_t) -> float32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_f32_s16(a: int16x8_t) -> float32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_f32_s32(a: int32x4_t) -> float32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_f32_s64(a: int64x2_t) -> float32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_f32_u8(a: uint8x8_t) -> float32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_f32_u16(a: uint16x4_t) -> float32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_f32_u32(a: uint32x2_t) -> float32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_f32_u64(a: uint64x1_t) -> float32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_f32_u8(a: uint8x16_t) -> float32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_f32_u16(a: uint16x8_t) -> float32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_f32_u32(a: uint32x4_t) -> float32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_f32_u64(a: uint64x2_t) -> float32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_f32_p8(a: poly8x8_t) -> float32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpret_f32_p16(a: poly16x4_t) -> float32x2_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_f32_p8(a: poly8x16_t) -> float32x4_t {
    transmute(a)
}

/// Vector reinterpret cast operation
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(str))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(str))]
pub unsafe fn vreinterpretq_f32_p16(a: poly16x8_t) -> float32x4_t {
    transmute(a)
}

/// Signed rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srshl))]
pub unsafe fn vrshl_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrshifts.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srshl.v8i8")]
        fn vrshl_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vrshl_s8_(a, b)
}

/// Signed rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srshl))]
pub unsafe fn vrshlq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrshifts.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srshl.v16i8")]
        fn vrshlq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vrshlq_s8_(a, b)
}

/// Signed rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srshl))]
pub unsafe fn vrshl_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrshifts.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srshl.v4i16")]
        fn vrshl_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vrshl_s16_(a, b)
}

/// Signed rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srshl))]
pub unsafe fn vrshlq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrshifts.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srshl.v8i16")]
        fn vrshlq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vrshlq_s16_(a, b)
}

/// Signed rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srshl))]
pub unsafe fn vrshl_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrshifts.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srshl.v2i32")]
        fn vrshl_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vrshl_s32_(a, b)
}

/// Signed rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srshl))]
pub unsafe fn vrshlq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrshifts.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srshl.v4i32")]
        fn vrshlq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vrshlq_s32_(a, b)
}

/// Signed rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srshl))]
pub unsafe fn vrshl_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrshifts.v1i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srshl.v1i64")]
        fn vrshl_s64_(a: int64x1_t, b: int64x1_t) -> int64x1_t;
    }
vrshl_s64_(a, b)
}

/// Signed rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srshl))]
pub unsafe fn vrshlq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrshifts.v2i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.srshl.v2i64")]
        fn vrshlq_s64_(a: int64x2_t, b: int64x2_t) -> int64x2_t;
    }
vrshlq_s64_(a, b)
}

/// Unsigned rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urshl))]
pub unsafe fn vrshl_u8(a: uint8x8_t, b: int8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrshiftu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urshl.v8i8")]
        fn vrshl_u8_(a: uint8x8_t, b: int8x8_t) -> uint8x8_t;
    }
vrshl_u8_(a, b)
}

/// Unsigned rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urshl))]
pub unsafe fn vrshlq_u8(a: uint8x16_t, b: int8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrshiftu.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urshl.v16i8")]
        fn vrshlq_u8_(a: uint8x16_t, b: int8x16_t) -> uint8x16_t;
    }
vrshlq_u8_(a, b)
}

/// Unsigned rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urshl))]
pub unsafe fn vrshl_u16(a: uint16x4_t, b: int16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrshiftu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urshl.v4i16")]
        fn vrshl_u16_(a: uint16x4_t, b: int16x4_t) -> uint16x4_t;
    }
vrshl_u16_(a, b)
}

/// Unsigned rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urshl))]
pub unsafe fn vrshlq_u16(a: uint16x8_t, b: int16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrshiftu.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urshl.v8i16")]
        fn vrshlq_u16_(a: uint16x8_t, b: int16x8_t) -> uint16x8_t;
    }
vrshlq_u16_(a, b)
}

/// Unsigned rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urshl))]
pub unsafe fn vrshl_u32(a: uint32x2_t, b: int32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrshiftu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urshl.v2i32")]
        fn vrshl_u32_(a: uint32x2_t, b: int32x2_t) -> uint32x2_t;
    }
vrshl_u32_(a, b)
}

/// Unsigned rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urshl))]
pub unsafe fn vrshlq_u32(a: uint32x4_t, b: int32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrshiftu.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urshl.v4i32")]
        fn vrshlq_u32_(a: uint32x4_t, b: int32x4_t) -> uint32x4_t;
    }
vrshlq_u32_(a, b)
}

/// Unsigned rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urshl))]
pub unsafe fn vrshl_u64(a: uint64x1_t, b: int64x1_t) -> uint64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrshiftu.v1i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urshl.v1i64")]
        fn vrshl_u64_(a: uint64x1_t, b: int64x1_t) -> uint64x1_t;
    }
vrshl_u64_(a, b)
}

/// Unsigned rounding shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urshl))]
pub unsafe fn vrshlq_u64(a: uint64x2_t, b: int64x2_t) -> uint64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrshiftu.v2i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.urshl.v2i64")]
        fn vrshlq_u64_(a: uint64x2_t, b: int64x2_t) -> uint64x2_t;
    }
vrshlq_u64_(a, b)
}

/// Signed rounding shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshr, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshr_n_s8<const N: i32>(a: int8x8_t) -> int8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    vrshl_s8(a, vdup_n_s8((-N).try_into().unwrap()))
}

/// Signed rounding shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshr, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshrq_n_s8<const N: i32>(a: int8x16_t) -> int8x16_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    vrshlq_s8(a, vdupq_n_s8((-N).try_into().unwrap()))
}

/// Signed rounding shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshr, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshr_n_s16<const N: i32>(a: int16x4_t) -> int16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    vrshl_s16(a, vdup_n_s16((-N).try_into().unwrap()))
}

/// Signed rounding shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshr, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshrq_n_s16<const N: i32>(a: int16x8_t) -> int16x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    vrshlq_s16(a, vdupq_n_s16((-N).try_into().unwrap()))
}

/// Signed rounding shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshr, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshr_n_s32<const N: i32>(a: int32x2_t) -> int32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    vrshl_s32(a, vdup_n_s32((-N).try_into().unwrap()))
}

/// Signed rounding shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshr, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshrq_n_s32<const N: i32>(a: int32x4_t) -> int32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    vrshlq_s32(a, vdupq_n_s32((-N).try_into().unwrap()))
}

/// Signed rounding shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshr, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshr_n_s64<const N: i32>(a: int64x1_t) -> int64x1_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    vrshl_s64(a, vdup_n_s64((-N).try_into().unwrap()))
}

/// Signed rounding shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshr, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshrq_n_s64<const N: i32>(a: int64x2_t) -> int64x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    vrshlq_s64(a, vdupq_n_s64((-N).try_into().unwrap()))
}

/// Unsigned rounding shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshr, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshr_n_u8<const N: i32>(a: uint8x8_t) -> uint8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    vrshl_u8(a, vdup_n_s8((-N).try_into().unwrap()))
}

/// Unsigned rounding shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshr, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshrq_n_u8<const N: i32>(a: uint8x16_t) -> uint8x16_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    vrshlq_u8(a, vdupq_n_s8((-N).try_into().unwrap()))
}

/// Unsigned rounding shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshr, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshr_n_u16<const N: i32>(a: uint16x4_t) -> uint16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    vrshl_u16(a, vdup_n_s16((-N).try_into().unwrap()))
}

/// Unsigned rounding shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshr, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshrq_n_u16<const N: i32>(a: uint16x8_t) -> uint16x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    vrshlq_u16(a, vdupq_n_s16((-N).try_into().unwrap()))
}

/// Unsigned rounding shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshr, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshr_n_u32<const N: i32>(a: uint32x2_t) -> uint32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    vrshl_u32(a, vdup_n_s32((-N).try_into().unwrap()))
}

/// Unsigned rounding shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshr, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshrq_n_u32<const N: i32>(a: uint32x4_t) -> uint32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    vrshlq_u32(a, vdupq_n_s32((-N).try_into().unwrap()))
}

/// Unsigned rounding shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshr, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshr_n_u64<const N: i32>(a: uint64x1_t) -> uint64x1_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    vrshl_u64(a, vdup_n_s64((-N).try_into().unwrap()))
}

/// Unsigned rounding shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshr, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(urshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshrq_n_u64<const N: i32>(a: uint64x2_t) -> uint64x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    vrshlq_u64(a, vdupq_n_s64((-N).try_into().unwrap()))
}

/// Rounding shift right narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshrn_n_s16<const N: i32>(a: int16x8_t) -> int8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrshiftn.v8i8")]
        fn vrshrn_n_s16_(a: int16x8_t, n: int16x8_t) -> int8x8_t;
    }
vrshrn_n_s16_(a, int16x8_t(-N as i16, -N as i16, -N as i16, -N as i16, -N as i16, -N as i16, -N as i16, -N as i16))
}

/// Rounding shift right narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(rshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshrn_n_s16<const N: i32>(a: int16x8_t) -> int8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.rshrn.v8i8")]
        fn vrshrn_n_s16_(a: int16x8_t, n: i32) -> int8x8_t;
    }
vrshrn_n_s16_(a, N)
}

/// Rounding shift right narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshrn_n_s32<const N: i32>(a: int32x4_t) -> int16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrshiftn.v4i16")]
        fn vrshrn_n_s32_(a: int32x4_t, n: int32x4_t) -> int16x4_t;
    }
vrshrn_n_s32_(a, int32x4_t(-N as i32, -N as i32, -N as i32, -N as i32))
}

/// Rounding shift right narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(rshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshrn_n_s32<const N: i32>(a: int32x4_t) -> int16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.rshrn.v4i16")]
        fn vrshrn_n_s32_(a: int32x4_t, n: i32) -> int16x4_t;
    }
vrshrn_n_s32_(a, N)
}

/// Rounding shift right narrow
#[inline]
#[cfg(target_arch = "arm")]
#[target_feature(enable = "neon,v7")]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshrn_n_s64<const N: i32>(a: int64x2_t) -> int32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vrshiftn.v2i32")]
        fn vrshrn_n_s64_(a: int64x2_t, n: int64x2_t) -> int32x2_t;
    }
vrshrn_n_s64_(a, int64x2_t(-N as i64, -N as i64))
}

/// Rounding shift right narrow
#[inline]
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(rshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshrn_n_s64<const N: i32>(a: int64x2_t) -> int32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.rshrn.v2i32")]
        fn vrshrn_n_s64_(a: int64x2_t, n: i32) -> int32x2_t;
    }
vrshrn_n_s64_(a, N)
}

/// Rounding shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshrn, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(rshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshrn_n_u16<const N: i32>(a: uint16x8_t) -> uint8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    transmute(vrshrn_n_s16::<N>(transmute(a)))
}

/// Rounding shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshrn, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(rshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshrn_n_u32<const N: i32>(a: uint32x4_t) -> uint16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    transmute(vrshrn_n_s32::<N>(transmute(a)))
}

/// Rounding shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrshrn, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(rshrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vrshrn_n_u64<const N: i32>(a: uint64x2_t) -> uint32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    transmute(vrshrn_n_s64::<N>(transmute(a)))
}

/// Signed rounding shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srsra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrsra_n_s8<const N: i32>(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_add(a, vrshr_n_s8::<N>(b))
}

/// Signed rounding shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srsra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrsraq_n_s8<const N: i32>(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_add(a, vrshrq_n_s8::<N>(b))
}

/// Signed rounding shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srsra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrsra_n_s16<const N: i32>(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_add(a, vrshr_n_s16::<N>(b))
}

/// Signed rounding shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srsra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrsraq_n_s16<const N: i32>(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_add(a, vrshrq_n_s16::<N>(b))
}

/// Signed rounding shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srsra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrsra_n_s32<const N: i32>(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_add(a, vrshr_n_s32::<N>(b))
}

/// Signed rounding shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srsra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrsraq_n_s32<const N: i32>(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_add(a, vrshrq_n_s32::<N>(b))
}

/// Signed rounding shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srsra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrsra_n_s64<const N: i32>(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    simd_add(a, vrshr_n_s64::<N>(b))
}

/// Signed rounding shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(srsra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrsraq_n_s64<const N: i32>(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    simd_add(a, vrshrq_n_s64::<N>(b))
}

/// Unsigned rounding shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ursra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrsra_n_u8<const N: i32>(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_add(a, vrshr_n_u8::<N>(b))
}

/// Unsigned rounding shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ursra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrsraq_n_u8<const N: i32>(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_add(a, vrshrq_n_u8::<N>(b))
}

/// Unsigned rounding shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ursra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrsra_n_u16<const N: i32>(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_add(a, vrshr_n_u16::<N>(b))
}

/// Unsigned rounding shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ursra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrsraq_n_u16<const N: i32>(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_add(a, vrshrq_n_u16::<N>(b))
}

/// Unsigned rounding shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ursra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrsra_n_u32<const N: i32>(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_add(a, vrshr_n_u32::<N>(b))
}

/// Unsigned rounding shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ursra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrsraq_n_u32<const N: i32>(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_add(a, vrshrq_n_u32::<N>(b))
}

/// Unsigned rounding shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ursra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrsra_n_u64<const N: i32>(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    simd_add(a, vrshr_n_u64::<N>(b))
}

/// Unsigned rounding shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vrsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ursra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vrsraq_n_u64<const N: i32>(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    simd_add(a, vrshrq_n_u64::<N>(b))
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vset_lane_s8<const LANE: i32>(a: i8, b: int8x8_t) -> int8x8_t {
    static_assert_imm3!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vset_lane_s16<const LANE: i32>(a: i16, b: int16x4_t) -> int16x4_t {
    static_assert_imm2!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vset_lane_s32<const LANE: i32>(a: i32, b: int32x2_t) -> int32x2_t {
    static_assert_imm1!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vset_lane_s64<const LANE: i32>(a: i64, b: int64x1_t) -> int64x1_t {
    static_assert!(LANE : i32 where LANE == 0);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vset_lane_u8<const LANE: i32>(a: u8, b: uint8x8_t) -> uint8x8_t {
    static_assert_imm3!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vset_lane_u16<const LANE: i32>(a: u16, b: uint16x4_t) -> uint16x4_t {
    static_assert_imm2!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vset_lane_u32<const LANE: i32>(a: u32, b: uint32x2_t) -> uint32x2_t {
    static_assert_imm1!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vset_lane_u64<const LANE: i32>(a: u64, b: uint64x1_t) -> uint64x1_t {
    static_assert!(LANE : i32 where LANE == 0);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vset_lane_p8<const LANE: i32>(a: p8, b: poly8x8_t) -> poly8x8_t {
    static_assert_imm3!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vset_lane_p16<const LANE: i32>(a: p16, b: poly16x4_t) -> poly16x4_t {
    static_assert_imm2!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "crypto,v8"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vset_lane_p64<const LANE: i32>(a: p64, b: poly64x1_t) -> poly64x1_t {
    static_assert!(LANE : i32 where LANE == 0);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsetq_lane_s8<const LANE: i32>(a: i8, b: int8x16_t) -> int8x16_t {
    static_assert_imm4!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsetq_lane_s16<const LANE: i32>(a: i16, b: int16x8_t) -> int16x8_t {
    static_assert_imm3!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsetq_lane_s32<const LANE: i32>(a: i32, b: int32x4_t) -> int32x4_t {
    static_assert_imm2!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsetq_lane_s64<const LANE: i32>(a: i64, b: int64x2_t) -> int64x2_t {
    static_assert_imm1!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsetq_lane_u8<const LANE: i32>(a: u8, b: uint8x16_t) -> uint8x16_t {
    static_assert_imm4!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsetq_lane_u16<const LANE: i32>(a: u16, b: uint16x8_t) -> uint16x8_t {
    static_assert_imm3!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsetq_lane_u32<const LANE: i32>(a: u32, b: uint32x4_t) -> uint32x4_t {
    static_assert_imm2!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsetq_lane_u64<const LANE: i32>(a: u64, b: uint64x2_t) -> uint64x2_t {
    static_assert_imm1!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsetq_lane_p8<const LANE: i32>(a: p8, b: poly8x16_t) -> poly8x16_t {
    static_assert_imm4!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsetq_lane_p16<const LANE: i32>(a: p16, b: poly16x8_t) -> poly16x8_t {
    static_assert_imm3!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon,aes")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "crypto,v8"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsetq_lane_p64<const LANE: i32>(a: p64, b: poly64x2_t) -> poly64x2_t {
    static_assert_imm1!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vset_lane_f32<const LANE: i32>(a: f32, b: float32x2_t) -> float32x2_t {
    static_assert_imm1!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Insert vector element from another vector element
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(nop, LANE = 0))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(nop, LANE = 0))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsetq_lane_f32<const LANE: i32>(a: f32, b: float32x4_t) -> float32x4_t {
    static_assert_imm2!(LANE);
    simd_insert(b, LANE as u32, a)
}

/// Signed Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sshl))]
pub unsafe fn vshl_s8(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vshifts.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sshl.v8i8")]
        fn vshl_s8_(a: int8x8_t, b: int8x8_t) -> int8x8_t;
    }
vshl_s8_(a, b)
}

/// Signed Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sshl))]
pub unsafe fn vshlq_s8(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vshifts.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sshl.v16i8")]
        fn vshlq_s8_(a: int8x16_t, b: int8x16_t) -> int8x16_t;
    }
vshlq_s8_(a, b)
}

/// Signed Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sshl))]
pub unsafe fn vshl_s16(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vshifts.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sshl.v4i16")]
        fn vshl_s16_(a: int16x4_t, b: int16x4_t) -> int16x4_t;
    }
vshl_s16_(a, b)
}

/// Signed Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sshl))]
pub unsafe fn vshlq_s16(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vshifts.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sshl.v8i16")]
        fn vshlq_s16_(a: int16x8_t, b: int16x8_t) -> int16x8_t;
    }
vshlq_s16_(a, b)
}

/// Signed Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sshl))]
pub unsafe fn vshl_s32(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vshifts.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sshl.v2i32")]
        fn vshl_s32_(a: int32x2_t, b: int32x2_t) -> int32x2_t;
    }
vshl_s32_(a, b)
}

/// Signed Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sshl))]
pub unsafe fn vshlq_s32(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vshifts.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sshl.v4i32")]
        fn vshlq_s32_(a: int32x4_t, b: int32x4_t) -> int32x4_t;
    }
vshlq_s32_(a, b)
}

/// Signed Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sshl))]
pub unsafe fn vshl_s64(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vshifts.v1i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sshl.v1i64")]
        fn vshl_s64_(a: int64x1_t, b: int64x1_t) -> int64x1_t;
    }
vshl_s64_(a, b)
}

/// Signed Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sshl))]
pub unsafe fn vshlq_s64(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vshifts.v2i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sshl.v2i64")]
        fn vshlq_s64_(a: int64x2_t, b: int64x2_t) -> int64x2_t;
    }
vshlq_s64_(a, b)
}

/// Unsigned Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ushl))]
pub unsafe fn vshl_u8(a: uint8x8_t, b: int8x8_t) -> uint8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vshiftu.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.ushl.v8i8")]
        fn vshl_u8_(a: uint8x8_t, b: int8x8_t) -> uint8x8_t;
    }
vshl_u8_(a, b)
}

/// Unsigned Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ushl))]
pub unsafe fn vshlq_u8(a: uint8x16_t, b: int8x16_t) -> uint8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vshiftu.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.ushl.v16i8")]
        fn vshlq_u8_(a: uint8x16_t, b: int8x16_t) -> uint8x16_t;
    }
vshlq_u8_(a, b)
}

/// Unsigned Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ushl))]
pub unsafe fn vshl_u16(a: uint16x4_t, b: int16x4_t) -> uint16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vshiftu.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.ushl.v4i16")]
        fn vshl_u16_(a: uint16x4_t, b: int16x4_t) -> uint16x4_t;
    }
vshl_u16_(a, b)
}

/// Unsigned Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ushl))]
pub unsafe fn vshlq_u16(a: uint16x8_t, b: int16x8_t) -> uint16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vshiftu.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.ushl.v8i16")]
        fn vshlq_u16_(a: uint16x8_t, b: int16x8_t) -> uint16x8_t;
    }
vshlq_u16_(a, b)
}

/// Unsigned Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ushl))]
pub unsafe fn vshl_u32(a: uint32x2_t, b: int32x2_t) -> uint32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vshiftu.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.ushl.v2i32")]
        fn vshl_u32_(a: uint32x2_t, b: int32x2_t) -> uint32x2_t;
    }
vshl_u32_(a, b)
}

/// Unsigned Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ushl))]
pub unsafe fn vshlq_u32(a: uint32x4_t, b: int32x4_t) -> uint32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vshiftu.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.ushl.v4i32")]
        fn vshlq_u32_(a: uint32x4_t, b: int32x4_t) -> uint32x4_t;
    }
vshlq_u32_(a, b)
}

/// Unsigned Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ushl))]
pub unsafe fn vshl_u64(a: uint64x1_t, b: int64x1_t) -> uint64x1_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vshiftu.v1i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.ushl.v1i64")]
        fn vshl_u64_(a: uint64x1_t, b: int64x1_t) -> uint64x1_t;
    }
vshl_u64_(a, b)
}

/// Unsigned Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ushl))]
pub unsafe fn vshlq_u64(a: uint64x2_t, b: int64x2_t) -> uint64x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vshiftu.v2i64")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.ushl.v2i64")]
        fn vshlq_u64_(a: uint64x2_t, b: int64x2_t) -> uint64x2_t;
    }
vshlq_u64_(a, b)
}

/// Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshl_n_s8<const N: i32>(a: int8x8_t) -> int8x8_t {
    static_assert_imm3!(N);
    simd_shl(a, vdup_n_s8(N.try_into().unwrap()))
}

/// Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshlq_n_s8<const N: i32>(a: int8x16_t) -> int8x16_t {
    static_assert_imm3!(N);
    simd_shl(a, vdupq_n_s8(N.try_into().unwrap()))
}

/// Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshl_n_s16<const N: i32>(a: int16x4_t) -> int16x4_t {
    static_assert_imm4!(N);
    simd_shl(a, vdup_n_s16(N.try_into().unwrap()))
}

/// Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshlq_n_s16<const N: i32>(a: int16x8_t) -> int16x8_t {
    static_assert_imm4!(N);
    simd_shl(a, vdupq_n_s16(N.try_into().unwrap()))
}

/// Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshl_n_s32<const N: i32>(a: int32x2_t) -> int32x2_t {
    static_assert_imm5!(N);
    simd_shl(a, vdup_n_s32(N.try_into().unwrap()))
}

/// Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshlq_n_s32<const N: i32>(a: int32x4_t) -> int32x4_t {
    static_assert_imm5!(N);
    simd_shl(a, vdupq_n_s32(N.try_into().unwrap()))
}

/// Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshl_n_u8<const N: i32>(a: uint8x8_t) -> uint8x8_t {
    static_assert_imm3!(N);
    simd_shl(a, vdup_n_u8(N.try_into().unwrap()))
}

/// Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshlq_n_u8<const N: i32>(a: uint8x16_t) -> uint8x16_t {
    static_assert_imm3!(N);
    simd_shl(a, vdupq_n_u8(N.try_into().unwrap()))
}

/// Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshl_n_u16<const N: i32>(a: uint16x4_t) -> uint16x4_t {
    static_assert_imm4!(N);
    simd_shl(a, vdup_n_u16(N.try_into().unwrap()))
}

/// Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshlq_n_u16<const N: i32>(a: uint16x8_t) -> uint16x8_t {
    static_assert_imm4!(N);
    simd_shl(a, vdupq_n_u16(N.try_into().unwrap()))
}

/// Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshl_n_u32<const N: i32>(a: uint32x2_t) -> uint32x2_t {
    static_assert_imm5!(N);
    simd_shl(a, vdup_n_u32(N.try_into().unwrap()))
}

/// Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshlq_n_u32<const N: i32>(a: uint32x4_t) -> uint32x4_t {
    static_assert_imm5!(N);
    simd_shl(a, vdupq_n_u32(N.try_into().unwrap()))
}

/// Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshl_n_s64<const N: i32>(a: int64x1_t) -> int64x1_t {
    static_assert_imm6!(N);
    simd_shl(a, vdup_n_s64(N.try_into().unwrap()))
}

/// Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshlq_n_s64<const N: i32>(a: int64x2_t) -> int64x2_t {
    static_assert_imm6!(N);
    simd_shl(a, vdupq_n_s64(N.try_into().unwrap()))
}

/// Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshl_n_u64<const N: i32>(a: uint64x1_t) -> uint64x1_t {
    static_assert_imm6!(N);
    simd_shl(a, vdup_n_u64(N.try_into().unwrap()))
}

/// Shift left
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vshl, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shl, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshlq_n_u64<const N: i32>(a: uint64x2_t) -> uint64x2_t {
    static_assert_imm6!(N);
    simd_shl(a, vdupq_n_u64(N.try_into().unwrap()))
}

/// Signed shift left long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshll.s8", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sshll, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshll_n_s8<const N: i32>(a: int8x8_t) -> int16x8_t {
    static_assert!(N : i32 where N >= 0 && N <= 8);
    simd_shl(simd_cast(a), vdupq_n_s16(N.try_into().unwrap()))
}

/// Signed shift left long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshll.s16", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sshll, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshll_n_s16<const N: i32>(a: int16x4_t) -> int32x4_t {
    static_assert!(N : i32 where N >= 0 && N <= 16);
    simd_shl(simd_cast(a), vdupq_n_s32(N.try_into().unwrap()))
}

/// Signed shift left long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshll.s32", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sshll, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshll_n_s32<const N: i32>(a: int32x2_t) -> int64x2_t {
    static_assert!(N : i32 where N >= 0 && N <= 32);
    simd_shl(simd_cast(a), vdupq_n_s64(N.try_into().unwrap()))
}

/// Signed shift left long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshll.u8", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ushll, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshll_n_u8<const N: i32>(a: uint8x8_t) -> uint16x8_t {
    static_assert!(N : i32 where N >= 0 && N <= 8);
    simd_shl(simd_cast(a), vdupq_n_u16(N.try_into().unwrap()))
}

/// Signed shift left long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshll.u16", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ushll, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshll_n_u16<const N: i32>(a: uint16x4_t) -> uint32x4_t {
    static_assert!(N : i32 where N >= 0 && N <= 16);
    simd_shl(simd_cast(a), vdupq_n_u32(N.try_into().unwrap()))
}

/// Signed shift left long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshll.u32", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ushll, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshll_n_u32<const N: i32>(a: uint32x2_t) -> uint64x2_t {
    static_assert!(N : i32 where N >= 0 && N <= 32);
    simd_shl(simd_cast(a), vdupq_n_u64(N.try_into().unwrap()))
}

/// Shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshr.s8", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshr_n_s8<const N: i32>(a: int8x8_t) -> int8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_shr(a, vdup_n_s8(N.try_into().unwrap()))
}

/// Shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshr.s8", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshrq_n_s8<const N: i32>(a: int8x16_t) -> int8x16_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_shr(a, vdupq_n_s8(N.try_into().unwrap()))
}

/// Shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshr.s16", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshr_n_s16<const N: i32>(a: int16x4_t) -> int16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_shr(a, vdup_n_s16(N.try_into().unwrap()))
}

/// Shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshr.s16", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshrq_n_s16<const N: i32>(a: int16x8_t) -> int16x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_shr(a, vdupq_n_s16(N.try_into().unwrap()))
}

/// Shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshr.s32", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshr_n_s32<const N: i32>(a: int32x2_t) -> int32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_shr(a, vdup_n_s32(N.try_into().unwrap()))
}

/// Shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshr.s32", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshrq_n_s32<const N: i32>(a: int32x4_t) -> int32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_shr(a, vdupq_n_s32(N.try_into().unwrap()))
}

/// Shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshr.s64", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshr_n_s64<const N: i32>(a: int64x1_t) -> int64x1_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    simd_shr(a, vdup_n_s64(N.try_into().unwrap()))
}

/// Shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshr.s64", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sshr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshrq_n_s64<const N: i32>(a: int64x2_t) -> int64x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    simd_shr(a, vdupq_n_s64(N.try_into().unwrap()))
}

/// Shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshr.u8", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ushr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshr_n_u8<const N: i32>(a: uint8x8_t) -> uint8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_shr(a, vdup_n_u8(N.try_into().unwrap()))
}

/// Shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshr.u8", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ushr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshrq_n_u8<const N: i32>(a: uint8x16_t) -> uint8x16_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_shr(a, vdupq_n_u8(N.try_into().unwrap()))
}

/// Shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshr.u16", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ushr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshr_n_u16<const N: i32>(a: uint16x4_t) -> uint16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_shr(a, vdup_n_u16(N.try_into().unwrap()))
}

/// Shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshr.u16", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ushr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshrq_n_u16<const N: i32>(a: uint16x8_t) -> uint16x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_shr(a, vdupq_n_u16(N.try_into().unwrap()))
}

/// Shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshr.u32", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ushr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshr_n_u32<const N: i32>(a: uint32x2_t) -> uint32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_shr(a, vdup_n_u32(N.try_into().unwrap()))
}

/// Shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshr.u32", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ushr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshrq_n_u32<const N: i32>(a: uint32x4_t) -> uint32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_shr(a, vdupq_n_u32(N.try_into().unwrap()))
}

/// Shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshr.u64", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ushr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshr_n_u64<const N: i32>(a: uint64x1_t) -> uint64x1_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    simd_shr(a, vdup_n_u64(N.try_into().unwrap()))
}

/// Shift right
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshr.u64", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ushr, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshrq_n_u64<const N: i32>(a: uint64x2_t) -> uint64x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    simd_shr(a, vdupq_n_u64(N.try_into().unwrap()))
}

/// Shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshrn.i16", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshrn_n_s16<const N: i32>(a: int16x8_t) -> int8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_cast(simd_shr(a, vdupq_n_s16(N.try_into().unwrap())))
}

/// Shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshrn.i32", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshrn_n_s32<const N: i32>(a: int32x4_t) -> int16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_cast(simd_shr(a, vdupq_n_s32(N.try_into().unwrap())))
}

/// Shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshrn.i64", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshrn_n_s64<const N: i32>(a: int64x2_t) -> int32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_cast(simd_shr(a, vdupq_n_s64(N.try_into().unwrap())))
}

/// Shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshrn.i16", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshrn_n_u16<const N: i32>(a: uint16x8_t) -> uint8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_cast(simd_shr(a, vdupq_n_u16(N.try_into().unwrap())))
}

/// Shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshrn.i32", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshrn_n_u32<const N: i32>(a: uint32x4_t) -> uint16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_cast(simd_shr(a, vdupq_n_u32(N.try_into().unwrap())))
}

/// Shift right narrow
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vshrn.i64", N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(shrn, N = 2))]
#[rustc_legacy_const_generics(1)]
pub unsafe fn vshrn_n_u64<const N: i32>(a: uint64x2_t) -> uint32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_cast(simd_shr(a, vdupq_n_u64(N.try_into().unwrap())))
}

/// Signed shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ssra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsra_n_s8<const N: i32>(a: int8x8_t, b: int8x8_t) -> int8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_add(a, vshr_n_s8::<N>(b))
}

/// Signed shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ssra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsraq_n_s8<const N: i32>(a: int8x16_t, b: int8x16_t) -> int8x16_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_add(a, vshrq_n_s8::<N>(b))
}

/// Signed shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ssra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsra_n_s16<const N: i32>(a: int16x4_t, b: int16x4_t) -> int16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_add(a, vshr_n_s16::<N>(b))
}

/// Signed shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ssra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsraq_n_s16<const N: i32>(a: int16x8_t, b: int16x8_t) -> int16x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_add(a, vshrq_n_s16::<N>(b))
}

/// Signed shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ssra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsra_n_s32<const N: i32>(a: int32x2_t, b: int32x2_t) -> int32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_add(a, vshr_n_s32::<N>(b))
}

/// Signed shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ssra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsraq_n_s32<const N: i32>(a: int32x4_t, b: int32x4_t) -> int32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_add(a, vshrq_n_s32::<N>(b))
}

/// Signed shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ssra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsra_n_s64<const N: i32>(a: int64x1_t, b: int64x1_t) -> int64x1_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    simd_add(a, vshr_n_s64::<N>(b))
}

/// Signed shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(ssra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsraq_n_s64<const N: i32>(a: int64x2_t, b: int64x2_t) -> int64x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    simd_add(a, vshrq_n_s64::<N>(b))
}

/// Unsigned shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(usra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsra_n_u8<const N: i32>(a: uint8x8_t, b: uint8x8_t) -> uint8x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_add(a, vshr_n_u8::<N>(b))
}

/// Unsigned shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(usra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsraq_n_u8<const N: i32>(a: uint8x16_t, b: uint8x16_t) -> uint8x16_t {
    static_assert!(N : i32 where N >= 1 && N <= 8);
    simd_add(a, vshrq_n_u8::<N>(b))
}

/// Unsigned shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(usra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsra_n_u16<const N: i32>(a: uint16x4_t, b: uint16x4_t) -> uint16x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_add(a, vshr_n_u16::<N>(b))
}

/// Unsigned shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(usra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsraq_n_u16<const N: i32>(a: uint16x8_t, b: uint16x8_t) -> uint16x8_t {
    static_assert!(N : i32 where N >= 1 && N <= 16);
    simd_add(a, vshrq_n_u16::<N>(b))
}

/// Unsigned shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(usra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsra_n_u32<const N: i32>(a: uint32x2_t, b: uint32x2_t) -> uint32x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_add(a, vshr_n_u32::<N>(b))
}

/// Unsigned shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(usra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsraq_n_u32<const N: i32>(a: uint32x4_t, b: uint32x4_t) -> uint32x4_t {
    static_assert!(N : i32 where N >= 1 && N <= 32);
    simd_add(a, vshrq_n_u32::<N>(b))
}

/// Unsigned shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(usra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsra_n_u64<const N: i32>(a: uint64x1_t, b: uint64x1_t) -> uint64x1_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    simd_add(a, vshr_n_u64::<N>(b))
}

/// Unsigned shift right and accumulate
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr(vsra, N = 2))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(usra, N = 2))]
#[rustc_legacy_const_generics(2)]
pub unsafe fn vsraq_n_u64<const N: i32>(a: uint64x2_t, b: uint64x2_t) -> uint64x2_t {
    static_assert!(N : i32 where N >= 1 && N <= 64);
    simd_add(a, vshrq_n_u64::<N>(b))
}

/// Unsigned Absolute difference and Accumulate Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabal.u8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uabal))]
pub unsafe fn vabal_u8(a: uint16x8_t, b: uint8x8_t, c: uint8x8_t) -> uint16x8_t {
    let d: uint8x8_t = vabd_u8(b, c);
    simd_add(a, simd_cast(d))
}

/// Unsigned Absolute difference and Accumulate Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabal.u16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uabal))]
pub unsafe fn vabal_u16(a: uint32x4_t, b: uint16x4_t, c: uint16x4_t) -> uint32x4_t {
    let d: uint16x4_t = vabd_u16(b, c);
    simd_add(a, simd_cast(d))
}

/// Unsigned Absolute difference and Accumulate Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabal.u32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(uabal))]
pub unsafe fn vabal_u32(a: uint64x2_t, b: uint32x2_t, c: uint32x2_t) -> uint64x2_t {
    let d: uint32x2_t = vabd_u32(b, c);
    simd_add(a, simd_cast(d))
}

/// Signed Absolute difference and Accumulate Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabal.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sabal))]
pub unsafe fn vabal_s8(a: int16x8_t, b: int8x8_t, c: int8x8_t) -> int16x8_t {
    let d: int8x8_t = vabd_s8(b, c);
    let e: uint8x8_t = simd_cast(d);
    simd_add(a, simd_cast(e))
}

/// Signed Absolute difference and Accumulate Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabal.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sabal))]
pub unsafe fn vabal_s16(a: int32x4_t, b: int16x4_t, c: int16x4_t) -> int32x4_t {
    let d: int16x4_t = vabd_s16(b, c);
    let e: uint16x4_t = simd_cast(d);
    simd_add(a, simd_cast(e))
}

/// Signed Absolute difference and Accumulate Long
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vabal.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sabal))]
pub unsafe fn vabal_s32(a: int64x2_t, b: int32x2_t, c: int32x2_t) -> int64x2_t {
    let d: int32x2_t = vabd_s32(b, c);
    let e: uint32x2_t = simd_cast(d);
    simd_add(a, simd_cast(e))
}

/// Singned saturating Absolute value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqabs.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqabs))]
pub unsafe fn vqabs_s8(a: int8x8_t) -> int8x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqabs.v8i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqabs.v8i8")]
        fn vqabs_s8_(a: int8x8_t) -> int8x8_t;
    }
vqabs_s8_(a)
}

/// Singned saturating Absolute value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqabs.s8"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqabs))]
pub unsafe fn vqabsq_s8(a: int8x16_t) -> int8x16_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqabs.v16i8")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqabs.v16i8")]
        fn vqabsq_s8_(a: int8x16_t) -> int8x16_t;
    }
vqabsq_s8_(a)
}

/// Singned saturating Absolute value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqabs.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqabs))]
pub unsafe fn vqabs_s16(a: int16x4_t) -> int16x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqabs.v4i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqabs.v4i16")]
        fn vqabs_s16_(a: int16x4_t) -> int16x4_t;
    }
vqabs_s16_(a)
}

/// Singned saturating Absolute value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqabs.s16"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqabs))]
pub unsafe fn vqabsq_s16(a: int16x8_t) -> int16x8_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqabs.v8i16")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqabs.v8i16")]
        fn vqabsq_s16_(a: int16x8_t) -> int16x8_t;
    }
vqabsq_s16_(a)
}

/// Singned saturating Absolute value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqabs.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqabs))]
pub unsafe fn vqabs_s32(a: int32x2_t) -> int32x2_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqabs.v2i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqabs.v2i32")]
        fn vqabs_s32_(a: int32x2_t) -> int32x2_t;
    }
vqabs_s32_(a)
}

/// Singned saturating Absolute value
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr("vqabs.s32"))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr(sqabs))]
pub unsafe fn vqabsq_s32(a: int32x4_t) -> int32x4_t {
    #[allow(improper_ctypes)]
    extern "C" {
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.vqabs.v4i32")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.sqabs.v4i32")]
        fn vqabsq_s32_(a: int32x4_t) -> int32x4_t;
    }
vqabsq_s32_(a)
}

#[cfg(test)]
#[allow(overflowing_literals)]
mod test {
    use super::*;
    use crate::core_arch::simd::*;
    use std::mem::transmute;
    use stdarch_test::simd_test;

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s8() {
        let a: i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i8x8 = i8x8::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let e: i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: i8x8 = transmute(vand_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i8x8 = i8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i8x8 = i8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let r: i8x8 = transmute(vand_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s8() {
        let a: i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00);
        let b: i8x16 = i8x16::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let e: i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00);
        let r: i8x16 = transmute(vandq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00);
        let b: i8x16 = i8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i8x16 = i8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let r: i8x16 = transmute(vandq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s16() {
        let a: i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let b: i16x4 = i16x4::new(0x0F, 0x0F, 0x0F, 0x0F);
        let e: i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let r: i16x4 = transmute(vand_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let b: i16x4 = i16x4::new(0x00, 0x00, 0x00, 0x00);
        let e: i16x4 = i16x4::new(0x00, 0x00, 0x00, 0x00);
        let r: i16x4 = transmute(vand_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s16() {
        let a: i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i16x8 = i16x8::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let e: i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: i16x8 = transmute(vandq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i16x8 = i16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i16x8 = i16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let r: i16x8 = transmute(vandq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s32() {
        let a: i32x2 = i32x2::new(0x00, 0x01);
        let b: i32x2 = i32x2::new(0x0F, 0x0F);
        let e: i32x2 = i32x2::new(0x00, 0x01);
        let r: i32x2 = transmute(vand_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i32x2 = i32x2::new(0x00, 0x01);
        let b: i32x2 = i32x2::new(0x00, 0x00);
        let e: i32x2 = i32x2::new(0x00, 0x00);
        let r: i32x2 = transmute(vand_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s32() {
        let a: i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let b: i32x4 = i32x4::new(0x0F, 0x0F, 0x0F, 0x0F);
        let e: i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let r: i32x4 = transmute(vandq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let b: i32x4 = i32x4::new(0x00, 0x00, 0x00, 0x00);
        let e: i32x4 = i32x4::new(0x00, 0x00, 0x00, 0x00);
        let r: i32x4 = transmute(vandq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u8() {
        let a: u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u8x8 = u8x8::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let e: u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: u8x8 = transmute(vand_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u8x8 = u8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: u8x8 = u8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let r: u8x8 = transmute(vand_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u8() {
        let a: u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00);
        let b: u8x16 = u8x16::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let e: u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00);
        let r: u8x16 = transmute(vandq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x00);
        let b: u8x16 = u8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: u8x16 = u8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let r: u8x16 = transmute(vandq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u16() {
        let a: u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let b: u16x4 = u16x4::new(0x0F, 0x0F, 0x0F, 0x0F);
        let e: u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let r: u16x4 = transmute(vand_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let b: u16x4 = u16x4::new(0x00, 0x00, 0x00, 0x00);
        let e: u16x4 = u16x4::new(0x00, 0x00, 0x00, 0x00);
        let r: u16x4 = transmute(vand_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u16() {
        let a: u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u16x8 = u16x8::new(0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F, 0x0F);
        let e: u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: u16x8 = transmute(vandq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u16x8 = u16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: u16x8 = u16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let r: u16x8 = transmute(vandq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u32() {
        let a: u32x2 = u32x2::new(0x00, 0x01);
        let b: u32x2 = u32x2::new(0x0F, 0x0F);
        let e: u32x2 = u32x2::new(0x00, 0x01);
        let r: u32x2 = transmute(vand_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u32x2 = u32x2::new(0x00, 0x01);
        let b: u32x2 = u32x2::new(0x00, 0x00);
        let e: u32x2 = u32x2::new(0x00, 0x00);
        let r: u32x2 = transmute(vand_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u32() {
        let a: u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let b: u32x4 = u32x4::new(0x0F, 0x0F, 0x0F, 0x0F);
        let e: u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let r: u32x4 = transmute(vandq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let b: u32x4 = u32x4::new(0x00, 0x00, 0x00, 0x00);
        let e: u32x4 = u32x4::new(0x00, 0x00, 0x00, 0x00);
        let r: u32x4 = transmute(vandq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_s64() {
        let a: i64x1 = i64x1::new(0x00);
        let b: i64x1 = i64x1::new(0x0F);
        let e: i64x1 = i64x1::new(0x00);
        let r: i64x1 = transmute(vand_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i64x1 = i64x1::new(0x00);
        let b: i64x1 = i64x1::new(0x00);
        let e: i64x1 = i64x1::new(0x00);
        let r: i64x1 = transmute(vand_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_s64() {
        let a: i64x2 = i64x2::new(0x00, 0x01);
        let b: i64x2 = i64x2::new(0x0F, 0x0F);
        let e: i64x2 = i64x2::new(0x00, 0x01);
        let r: i64x2 = transmute(vandq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i64x2 = i64x2::new(0x00, 0x01);
        let b: i64x2 = i64x2::new(0x00, 0x00);
        let e: i64x2 = i64x2::new(0x00, 0x00);
        let r: i64x2 = transmute(vandq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vand_u64() {
        let a: u64x1 = u64x1::new(0x00);
        let b: u64x1 = u64x1::new(0x0F);
        let e: u64x1 = u64x1::new(0x00);
        let r: u64x1 = transmute(vand_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u64x1 = u64x1::new(0x00);
        let b: u64x1 = u64x1::new(0x00);
        let e: u64x1 = u64x1::new(0x00);
        let r: u64x1 = transmute(vand_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vandq_u64() {
        let a: u64x2 = u64x2::new(0x00, 0x01);
        let b: u64x2 = u64x2::new(0x0F, 0x0F);
        let e: u64x2 = u64x2::new(0x00, 0x01);
        let r: u64x2 = transmute(vandq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u64x2 = u64x2::new(0x00, 0x01);
        let b: u64x2 = u64x2::new(0x00, 0x00);
        let e: u64x2 = u64x2::new(0x00, 0x00);
        let r: u64x2 = transmute(vandq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s8() {
        let a: i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i8x8 = i8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: i8x8 = transmute(vorr_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s8() {
        let a: i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let b: i8x16 = i8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let r: i8x16 = transmute(vorrq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s16() {
        let a: i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let b: i16x4 = i16x4::new(0x00, 0x00, 0x00, 0x00);
        let e: i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let r: i16x4 = transmute(vorr_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s16() {
        let a: i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i16x8 = i16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: i16x8 = transmute(vorrq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s32() {
        let a: i32x2 = i32x2::new(0x00, 0x01);
        let b: i32x2 = i32x2::new(0x00, 0x00);
        let e: i32x2 = i32x2::new(0x00, 0x01);
        let r: i32x2 = transmute(vorr_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s32() {
        let a: i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let b: i32x4 = i32x4::new(0x00, 0x00, 0x00, 0x00);
        let e: i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let r: i32x4 = transmute(vorrq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u8() {
        let a: u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u8x8 = u8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: u8x8 = transmute(vorr_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u8() {
        let a: u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let b: u8x16 = u8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let r: u8x16 = transmute(vorrq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u16() {
        let a: u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let b: u16x4 = u16x4::new(0x00, 0x00, 0x00, 0x00);
        let e: u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let r: u16x4 = transmute(vorr_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u16() {
        let a: u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u16x8 = u16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: u16x8 = transmute(vorrq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u32() {
        let a: u32x2 = u32x2::new(0x00, 0x01);
        let b: u32x2 = u32x2::new(0x00, 0x00);
        let e: u32x2 = u32x2::new(0x00, 0x01);
        let r: u32x2 = transmute(vorr_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u32() {
        let a: u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let b: u32x4 = u32x4::new(0x00, 0x00, 0x00, 0x00);
        let e: u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let r: u32x4 = transmute(vorrq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_s64() {
        let a: i64x1 = i64x1::new(0x00);
        let b: i64x1 = i64x1::new(0x00);
        let e: i64x1 = i64x1::new(0x00);
        let r: i64x1 = transmute(vorr_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_s64() {
        let a: i64x2 = i64x2::new(0x00, 0x01);
        let b: i64x2 = i64x2::new(0x00, 0x00);
        let e: i64x2 = i64x2::new(0x00, 0x01);
        let r: i64x2 = transmute(vorrq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorr_u64() {
        let a: u64x1 = u64x1::new(0x00);
        let b: u64x1 = u64x1::new(0x00);
        let e: u64x1 = u64x1::new(0x00);
        let r: u64x1 = transmute(vorr_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vorrq_u64() {
        let a: u64x2 = u64x2::new(0x00, 0x01);
        let b: u64x2 = u64x2::new(0x00, 0x00);
        let e: u64x2 = u64x2::new(0x00, 0x01);
        let r: u64x2 = transmute(vorrq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s8() {
        let a: i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i8x8 = i8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i8x8 = i8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: i8x8 = transmute(veor_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s8() {
        let a: i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let b: i8x16 = i8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i8x16 = i8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let r: i8x16 = transmute(veorq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s16() {
        let a: i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let b: i16x4 = i16x4::new(0x00, 0x00, 0x00, 0x00);
        let e: i16x4 = i16x4::new(0x00, 0x01, 0x02, 0x03);
        let r: i16x4 = transmute(veor_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s16() {
        let a: i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i16x8 = i16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i16x8 = i16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: i16x8 = transmute(veorq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s32() {
        let a: i32x2 = i32x2::new(0x00, 0x01);
        let b: i32x2 = i32x2::new(0x00, 0x00);
        let e: i32x2 = i32x2::new(0x00, 0x01);
        let r: i32x2 = transmute(veor_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s32() {
        let a: i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let b: i32x4 = i32x4::new(0x00, 0x00, 0x00, 0x00);
        let e: i32x4 = i32x4::new(0x00, 0x01, 0x02, 0x03);
        let r: i32x4 = transmute(veorq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u8() {
        let a: u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u8x8 = u8x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: u8x8 = u8x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: u8x8 = transmute(veor_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u8() {
        let a: u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let b: u8x16 = u8x16::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: u8x16 = u8x16::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F);
        let r: u8x16 = transmute(veorq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u16() {
        let a: u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let b: u16x4 = u16x4::new(0x00, 0x00, 0x00, 0x00);
        let e: u16x4 = u16x4::new(0x00, 0x01, 0x02, 0x03);
        let r: u16x4 = transmute(veor_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u16() {
        let a: u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u16x8 = u16x8::new(0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: u16x8 = u16x8::new(0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let r: u16x8 = transmute(veorq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u32() {
        let a: u32x2 = u32x2::new(0x00, 0x01);
        let b: u32x2 = u32x2::new(0x00, 0x00);
        let e: u32x2 = u32x2::new(0x00, 0x01);
        let r: u32x2 = transmute(veor_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u32() {
        let a: u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let b: u32x4 = u32x4::new(0x00, 0x00, 0x00, 0x00);
        let e: u32x4 = u32x4::new(0x00, 0x01, 0x02, 0x03);
        let r: u32x4 = transmute(veorq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_s64() {
        let a: i64x1 = i64x1::new(0x00);
        let b: i64x1 = i64x1::new(0x00);
        let e: i64x1 = i64x1::new(0x00);
        let r: i64x1 = transmute(veor_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_s64() {
        let a: i64x2 = i64x2::new(0x00, 0x01);
        let b: i64x2 = i64x2::new(0x00, 0x00);
        let e: i64x2 = i64x2::new(0x00, 0x01);
        let r: i64x2 = transmute(veorq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veor_u64() {
        let a: u64x1 = u64x1::new(0x00);
        let b: u64x1 = u64x1::new(0x00);
        let e: u64x1 = u64x1::new(0x00);
        let r: u64x1 = transmute(veor_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_veorq_u64() {
        let a: u64x2 = u64x2::new(0x00, 0x01);
        let b: u64x2 = u64x2::new(0x00, 0x00);
        let e: u64x2 = u64x2::new(0x00, 0x01);
        let r: u64x2 = transmute(veorq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabd_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: i8x8 = i8x8::new(15, 13, 11, 9, 7, 5, 3, 1);
        let r: i8x8 = transmute(vabd_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdq_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        let e: i8x16 = i8x16::new(15, 13, 11, 9, 7, 5, 3, 1, 1, 3, 5, 7, 9, 11, 13, 15);
        let r: i8x16 = transmute(vabdq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabd_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(16, 15, 14, 13);
        let e: i16x4 = i16x4::new(15, 13, 11, 9);
        let r: i16x4 = transmute(vabd_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: i16x8 = i16x8::new(15, 13, 11, 9, 7, 5, 3, 1);
        let r: i16x8 = transmute(vabdq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabd_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(16, 15);
        let e: i32x2 = i32x2::new(15, 13);
        let r: i32x2 = transmute(vabd_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(16, 15, 14, 13);
        let e: i32x4 = i32x4::new(15, 13, 11, 9);
        let r: i32x4 = transmute(vabdq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabd_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u8x8 = u8x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: u8x8 = u8x8::new(15, 13, 11, 9, 7, 5, 3, 1);
        let r: u8x8 = transmute(vabd_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdq_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: u8x16 = u8x16::new(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        let e: u8x16 = u8x16::new(15, 13, 11, 9, 7, 5, 3, 1, 1, 3, 5, 7, 9, 11, 13, 15);
        let r: u8x16 = transmute(vabdq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabd_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(16, 15, 14, 13);
        let e: u16x4 = u16x4::new(15, 13, 11, 9);
        let r: u16x4 = transmute(vabd_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u16x8 = u16x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: u16x8 = u16x8::new(15, 13, 11, 9, 7, 5, 3, 1);
        let r: u16x8 = transmute(vabdq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabd_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(16, 15);
        let e: u32x2 = u32x2::new(15, 13);
        let r: u32x2 = transmute(vabd_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32x4 = u32x4::new(16, 15, 14, 13);
        let e: u32x4 = u32x4::new(15, 13, 11, 9);
        let r: u32x4 = transmute(vabdq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabd_f32() {
        let a: f32x2 = f32x2::new(1.0, 2.0);
        let b: f32x2 = f32x2::new(9.0, 3.0);
        let e: f32x2 = f32x2::new(8.0, 1.0);
        let r: f32x2 = transmute(vabd_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdq_f32() {
        let a: f32x4 = f32x4::new(1.0, 2.0, 5.0, -4.0);
        let b: f32x4 = f32x4::new(9.0, 3.0, 2.0, 8.0);
        let e: f32x4 = f32x4::new(8.0, 1.0, 3.0, 12.0);
        let r: f32x4 = transmute(vabdq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdl_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 4, 3, 2, 1);
        let b: u8x8 = u8x8::new(10, 10, 10, 10, 10, 10, 10, 10);
        let e: u16x8 = u16x8::new(9, 8, 7, 6, 6, 7, 8, 9);
        let r: u16x8 = transmute(vabdl_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdl_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(10, 10, 10, 10);
        let e: u32x4 = u32x4::new(9, 8, 7, 6);
        let r: u32x4 = transmute(vabdl_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdl_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(10, 10);
        let e: u64x2 = u64x2::new(9, 8);
        let r: u64x2 = transmute(vabdl_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdl_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 4, 3, 2, 1);
        let b: i8x8 = i8x8::new(10, 10, 10, 10, 10, 10, 10, 10);
        let e: i16x8 = i16x8::new(9, 8, 7, 6, 6, 7, 8, 9);
        let r: i16x8 = transmute(vabdl_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdl_s16() {
        let a: i16x4 = i16x4::new(1, 2, 11, 12);
        let b: i16x4 = i16x4::new(10, 10, 10, 10);
        let e: i32x4 = i32x4::new(9, 8, 1, 2);
        let r: i32x4 = transmute(vabdl_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabdl_s32() {
        let a: i32x2 = i32x2::new(1, 11);
        let b: i32x2 = i32x2::new(10, 10);
        let e: i64x2 = i64x2::new(9, 1);
        let r: i64x2 = transmute(vabdl_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u8() {
        let a: u8x8 = u8x8::new(0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u8x8 = u8x8::new(0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vceq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u8x8 = u8x8::new(0, 0, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u8x8 = u8x8::new(0, 0xFF, 0x02, 0x04, 0x04, 0x00, 0x06, 0x08);
        let e: u8x8 = u8x8::new(0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0);
        let r: u8x8 = transmute(vceq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u8() {
        let a: u8x16 = u8x16::new(0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0xFF);
        let b: u8x16 = u8x16::new(0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0xFF);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vceqq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u8x16 = u8x16::new(0, 0, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xCC, 0x0D, 0xEE, 0xFF);
        let b: u8x16 = u8x16::new(0, 0xFF, 0x02, 0x04, 0x04, 0x00, 0x06, 0x08, 0x08, 0x00, 0x0A, 0x0A, 0xCC, 0xD0, 0xEE, 0);
        let e: u8x16 = u8x16::new(0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0);
        let r: u8x16 = transmute(vceqq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u16() {
        let a: u16x4 = u16x4::new(0, 0x01, 0x02, 0x03);
        let b: u16x4 = u16x4::new(0, 0x01, 0x02, 0x03);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vceq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u16x4 = u16x4::new(0, 0, 0x02, 0x03);
        let b: u16x4 = u16x4::new(0, 0xFF_FF, 0x02, 0x04);
        let e: u16x4 = u16x4::new(0xFF_FF, 0, 0xFF_FF, 0);
        let r: u16x4 = transmute(vceq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u16() {
        let a: u16x8 = u16x8::new(0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u16x8 = u16x8::new(0, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vceqq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u16x8 = u16x8::new(0, 0, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: u16x8 = u16x8::new(0, 0xFF_FF, 0x02, 0x04, 0x04, 0x00, 0x06, 0x08);
        let e: u16x8 = u16x8::new(0xFF_FF, 0, 0xFF_FF, 0, 0xFF_FF, 0, 0xFF_FF, 0);
        let r: u16x8 = transmute(vceqq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_u32() {
        let a: u32x2 = u32x2::new(0, 0x01);
        let b: u32x2 = u32x2::new(0, 0x01);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vceq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u32x2 = u32x2::new(0, 0);
        let b: u32x2 = u32x2::new(0, 0xFF_FF_FF_FF);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0);
        let r: u32x2 = transmute(vceq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_u32() {
        let a: u32x4 = u32x4::new(0, 0x01, 0x02, 0x03);
        let b: u32x4 = u32x4::new(0, 0x01, 0x02, 0x03);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vceqq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: u32x4 = u32x4::new(0, 0, 0x02, 0x03);
        let b: u32x4 = u32x4::new(0, 0xFF_FF_FF_FF, 0x02, 0x04);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0, 0xFF_FF_FF_FF, 0);
        let r: u32x4 = transmute(vceqq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s8() {
        let a: i8x8 = i8x8::new(-128, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i8x8 = i8x8::new(-128, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vceq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i8x8 = i8x8::new(-128, -128, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i8x8 = i8x8::new(-128, 0x7F, 0x02, 0x04, 0x04, 0x00, 0x06, 0x08);
        let e: u8x8 = u8x8::new(0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0);
        let r: u8x8 = transmute(vceq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s8() {
        let a: i8x16 = i8x16::new(-128, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x7F);
        let b: i8x16 = i8x16::new(-128, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x7F);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vceqq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i8x16 = i8x16::new(-128, -128, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xCC, 0x0D, 0xEE, 0x7F);
        let b: i8x16 = i8x16::new(-128, 0x7F, 0x02, 0x04, 0x04, 0x00, 0x06, 0x08, 0x08, 0x00, 0x0A, 0x0A, 0xCC, 0xD0, 0xEE, -128);
        let e: u8x16 = u8x16::new(0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0);
        let r: u8x16 = transmute(vceqq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s16() {
        let a: i16x4 = i16x4::new(-32768, 0x01, 0x02, 0x03);
        let b: i16x4 = i16x4::new(-32768, 0x01, 0x02, 0x03);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vceq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i16x4 = i16x4::new(-32768, -32768, 0x02, 0x03);
        let b: i16x4 = i16x4::new(-32768, 0x7F_FF, 0x02, 0x04);
        let e: u16x4 = u16x4::new(0xFF_FF, 0, 0xFF_FF, 0);
        let r: u16x4 = transmute(vceq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s16() {
        let a: i16x8 = i16x8::new(-32768, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i16x8 = i16x8::new(-32768, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vceqq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i16x8 = i16x8::new(-32768, -32768, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i16x8 = i16x8::new(-32768, 0x7F_FF, 0x02, 0x04, 0x04, 0x00, 0x06, 0x08);
        let e: u16x8 = u16x8::new(0xFF_FF, 0, 0xFF_FF, 0, 0xFF_FF, 0, 0xFF_FF, 0);
        let r: u16x8 = transmute(vceqq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_s32() {
        let a: i32x2 = i32x2::new(-2147483648, 0x01);
        let b: i32x2 = i32x2::new(-2147483648, 0x01);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vceq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i32x2 = i32x2::new(-2147483648, -2147483648);
        let b: i32x2 = i32x2::new(-2147483648, 0x7F_FF_FF_FF);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0);
        let r: u32x2 = transmute(vceq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_s32() {
        let a: i32x4 = i32x4::new(-2147483648, 0x01, 0x02, 0x03);
        let b: i32x4 = i32x4::new(-2147483648, 0x01, 0x02, 0x03);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vceqq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i32x4 = i32x4::new(-2147483648, -2147483648, 0x02, 0x03);
        let b: i32x4 = i32x4::new(-2147483648, 0x7F_FF_FF_FF, 0x02, 0x04);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0, 0xFF_FF_FF_FF, 0);
        let r: u32x4 = transmute(vceqq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_p8() {
        let a: i8x8 = i8x8::new(-128, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i8x8 = i8x8::new(-128, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vceq_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i8x8 = i8x8::new(-128, -128, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07);
        let b: i8x8 = i8x8::new(-128, 0x7F, 0x02, 0x04, 0x04, 0x00, 0x06, 0x08);
        let e: u8x8 = u8x8::new(0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0);
        let r: u8x8 = transmute(vceq_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_p8() {
        let a: i8x16 = i8x16::new(-128, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x7F);
        let b: i8x16 = i8x16::new(-128, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x7F);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vceqq_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);

        let a: i8x16 = i8x16::new(-128, -128, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0xCC, 0x0D, 0xEE, 0x7F);
        let b: i8x16 = i8x16::new(-128, 0x7F, 0x02, 0x04, 0x04, 0x00, 0x06, 0x08, 0x08, 0x00, 0x0A, 0x0A, 0xCC, 0xD0, 0xEE, -128);
        let e: u8x16 = u8x16::new(0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0);
        let r: u8x16 = transmute(vceqq_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceq_f32() {
        let a: f32x2 = f32x2::new(1.2, 3.4);
        let b: f32x2 = f32x2::new(1.2, 3.4);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vceq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vceqq_f32() {
        let a: f32x4 = f32x4::new(1.2, 3.4, 5.6, 7.8);
        let b: f32x4 = f32x4::new(1.2, 3.4, 5.6, 7.8);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vceqq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtst_s8() {
        let a: i8x8 = i8x8::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let b: i8x8 = i8x8::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let e: u8x8 = u8x8::new(0xFF, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vtst_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtstq_s8() {
        let a: i8x16 = i8x16::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x7F);
        let b: i8x16 = i8x16::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x7F);
        let e: u8x16 = u8x16::new(0xFF, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vtstq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtst_s16() {
        let a: i16x4 = i16x4::new(-32768, 0x00, 0x01, 0x02);
        let b: i16x4 = i16x4::new(-32768, 0x00, 0x01, 0x02);
        let e: u16x4 = u16x4::new(0xFF_FF, 0, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vtst_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtstq_s16() {
        let a: i16x8 = i16x8::new(-32768, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let b: i16x8 = i16x8::new(-32768, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let e: u16x8 = u16x8::new(0xFF_FF, 0, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vtstq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtst_s32() {
        let a: i32x2 = i32x2::new(-2147483648, 0x00);
        let b: i32x2 = i32x2::new(-2147483648, 0x00);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0);
        let r: u32x2 = transmute(vtst_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtstq_s32() {
        let a: i32x4 = i32x4::new(-2147483648, 0x00, 0x01, 0x02);
        let b: i32x4 = i32x4::new(-2147483648, 0x00, 0x01, 0x02);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vtstq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtst_p8() {
        let a: i8x8 = i8x8::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let b: i8x8 = i8x8::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let e: u8x8 = u8x8::new(0xFF, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vtst_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtstq_p8() {
        let a: i8x16 = i8x16::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x7F);
        let b: i8x16 = i8x16::new(-128, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x7F);
        let e: u8x16 = u8x16::new(0xFF, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vtstq_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtst_u8() {
        let a: u8x8 = u8x8::new(0, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let b: u8x8 = u8x8::new(0, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let e: u8x8 = u8x8::new(0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vtst_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtstq_u8() {
        let a: u8x16 = u8x16::new(0, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0xFF);
        let b: u8x16 = u8x16::new(0, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0xFF);
        let e: u8x16 = u8x16::new(0, 0, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vtstq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtst_u16() {
        let a: u16x4 = u16x4::new(0, 0x00, 0x01, 0x02);
        let b: u16x4 = u16x4::new(0, 0x00, 0x01, 0x02);
        let e: u16x4 = u16x4::new(0, 0, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vtst_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtstq_u16() {
        let a: u16x8 = u16x8::new(0, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let b: u16x8 = u16x8::new(0, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06);
        let e: u16x8 = u16x8::new(0, 0, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vtstq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtst_u32() {
        let a: u32x2 = u32x2::new(0, 0x00);
        let b: u32x2 = u32x2::new(0, 0x00);
        let e: u32x2 = u32x2::new(0, 0);
        let r: u32x2 = transmute(vtst_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vtstq_u32() {
        let a: u32x4 = u32x4::new(0, 0x00, 0x01, 0x02);
        let b: u32x4 = u32x4::new(0, 0x00, 0x01, 0x02);
        let e: u32x4 = u32x4::new(0, 0, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vtstq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabs_f32() {
        let a: f32x2 = f32x2::new(-0.1, -2.2);
        let e: f32x2 = f32x2::new(0.1, 2.2);
        let r: f32x2 = transmute(vabs_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabsq_f32() {
        let a: f32x4 = f32x4::new(-0.1, -2.2, -3.3, -6.6);
        let e: f32x4 = f32x4::new(0.1, 2.2, 3.3, 6.6);
        let r: f32x4 = transmute(vabsq_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcgt_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcgtq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(0, 1, 2, 3);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vcgt_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcgtq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(0, 1);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcgt_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(0, 1, 2, 3);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgtq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcgt_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcgtq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(0, 1, 2, 3);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vcgt_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcgtq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(0, 1);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcgt_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32x4 = u32x4::new(0, 1, 2, 3);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgtq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgt_f32() {
        let a: f32x2 = f32x2::new(1.2, 2.3);
        let b: f32x2 = f32x2::new(0.1, 1.2);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcgt_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgtq_f32() {
        let a: f32x4 = f32x4::new(1.2, 2.3, 3.4, 4.5);
        let b: f32x4 = f32x4::new(0.1, 1.2, 2.3, 3.4);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgtq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s8() {
        let a: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vclt_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s8() {
        let a: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcltq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vclt_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcltq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_s32() {
        let a: i32x2 = i32x2::new(0, 1);
        let b: i32x2 = i32x2::new(1, 2);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vclt_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_s32() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let b: i32x4 = i32x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcltq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u8() {
        let a: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vclt_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u8() {
        let a: u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcltq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u16() {
        let a: u16x4 = u16x4::new(0, 1, 2, 3);
        let b: u16x4 = u16x4::new(1, 2, 3, 4);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vclt_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u16() {
        let a: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcltq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_u32() {
        let a: u32x2 = u32x2::new(0, 1);
        let b: u32x2 = u32x2::new(1, 2);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vclt_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_u32() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let b: u32x4 = u32x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcltq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclt_f32() {
        let a: f32x2 = f32x2::new(0.1, 1.2);
        let b: f32x2 = f32x2::new(1.2, 2.3);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vclt_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcltq_f32() {
        let a: f32x4 = f32x4::new(0.1, 1.2, 2.3, 3.4);
        let b: f32x4 = f32x4::new(1.2, 2.3, 3.4, 4.5);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcltq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s8() {
        let a: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcle_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s8() {
        let a: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcleq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vcle_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcleq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_s32() {
        let a: i32x2 = i32x2::new(0, 1);
        let b: i32x2 = i32x2::new(1, 2);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcle_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_s32() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let b: i32x4 = i32x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcleq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u8() {
        let a: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcle_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u8() {
        let a: u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcleq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u16() {
        let a: u16x4 = u16x4::new(0, 1, 2, 3);
        let b: u16x4 = u16x4::new(1, 2, 3, 4);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vcle_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u16() {
        let a: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcleq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_u32() {
        let a: u32x2 = u32x2::new(0, 1);
        let b: u32x2 = u32x2::new(1, 2);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcle_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_u32() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let b: u32x4 = u32x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcleq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcle_f32() {
        let a: f32x2 = f32x2::new(0.1, 1.2);
        let b: f32x2 = f32x2::new(1.2, 2.3);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcle_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcleq_f32() {
        let a: f32x4 = f32x4::new(0.1, 1.2, 2.3, 3.4);
        let b: f32x4 = f32x4::new(1.2, 2.3, 3.4, 4.5);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcleq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcge_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcgeq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(0, 1, 2, 3);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vcge_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcgeq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(0, 1);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcge_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(0, 1, 2, 3);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgeq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vcge_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u8x16 = u8x16::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x16 = transmute(vcgeq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(0, 1, 2, 3);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vcge_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x8 = transmute(vcgeq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(0, 1);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcge_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32x4 = u32x4::new(0, 1, 2, 3);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgeq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcge_f32() {
        let a: f32x2 = f32x2::new(1.2, 2.3);
        let b: f32x2 = f32x2::new(0.1, 1.2);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcge_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcgeq_f32() {
        let a: f32x4 = f32x4::new(1.2, 2.3, 3.4, 4.5);
        let b: f32x4 = f32x4::new(0.1, 1.2, 2.3, 3.4);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcgeq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcls_s8() {
        let a: i8x8 = i8x8::new(-128, -1, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i8x8 = i8x8::new(0, 7, 7, 7, 7, 7, 7, 7);
        let r: i8x8 = transmute(vcls_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclsq_s8() {
        let a: i8x16 = i8x16::new(-128, -1, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7F);
        let e: i8x16 = i8x16::new(0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0);
        let r: i8x16 = transmute(vclsq_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcls_s16() {
        let a: i16x4 = i16x4::new(-32768, -1, 0x00, 0x00);
        let e: i16x4 = i16x4::new(0, 15, 15, 15);
        let r: i16x4 = transmute(vcls_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclsq_s16() {
        let a: i16x8 = i16x8::new(-32768, -1, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00);
        let e: i16x8 = i16x8::new(0, 15, 15, 15, 15, 15, 15, 15);
        let r: i16x8 = transmute(vclsq_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcls_s32() {
        let a: i32x2 = i32x2::new(-2147483648, -1);
        let e: i32x2 = i32x2::new(0, 31);
        let r: i32x2 = transmute(vcls_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclsq_s32() {
        let a: i32x4 = i32x4::new(-2147483648, -1, 0x00, 0x00);
        let e: i32x4 = i32x4::new(0, 31, 31, 31);
        let r: i32x4 = transmute(vclsq_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclz_s8() {
        let a: i8x8 = i8x8::new(-128, -1, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01);
        let e: i8x8 = i8x8::new(0, 0, 8, 7, 7, 7, 7, 7);
        let r: i8x8 = transmute(vclz_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclzq_s8() {
        let a: i8x16 = i8x16::new(-128, -1, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x7F);
        let e: i8x16 = i8x16::new(0, 0, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 1);
        let r: i8x16 = transmute(vclzq_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclz_s16() {
        let a: i16x4 = i16x4::new(-32768, -1, 0x00, 0x01);
        let e: i16x4 = i16x4::new(0, 0, 16, 15);
        let r: i16x4 = transmute(vclz_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclzq_s16() {
        let a: i16x8 = i16x8::new(-32768, -1, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01);
        let e: i16x8 = i16x8::new(0, 0, 16, 15, 15, 15, 15, 15);
        let r: i16x8 = transmute(vclzq_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclz_s32() {
        let a: i32x2 = i32x2::new(-2147483648, -1);
        let e: i32x2 = i32x2::new(0, 0);
        let r: i32x2 = transmute(vclz_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclzq_s32() {
        let a: i32x4 = i32x4::new(-2147483648, -1, 0x00, 0x01);
        let e: i32x4 = i32x4::new(0, 0, 32, 31);
        let r: i32x4 = transmute(vclzq_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclz_u8() {
        let a: u8x8 = u8x8::new(0, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01);
        let e: u8x8 = u8x8::new(8, 8, 7, 7, 7, 7, 7, 7);
        let r: u8x8 = transmute(vclz_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclzq_u8() {
        let a: u8x16 = u8x16::new(0, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0xFF);
        let e: u8x16 = u8x16::new(8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 0);
        let r: u8x16 = transmute(vclzq_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclz_u16() {
        let a: u16x4 = u16x4::new(0, 0x00, 0x01, 0x01);
        let e: u16x4 = u16x4::new(16, 16, 15, 15);
        let r: u16x4 = transmute(vclz_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclzq_u16() {
        let a: u16x8 = u16x8::new(0, 0x00, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01);
        let e: u16x8 = u16x8::new(16, 16, 15, 15, 15, 15, 15, 15);
        let r: u16x8 = transmute(vclzq_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclz_u32() {
        let a: u32x2 = u32x2::new(0, 0x00);
        let e: u32x2 = u32x2::new(32, 32);
        let r: u32x2 = transmute(vclz_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vclzq_u32() {
        let a: u32x4 = u32x4::new(0, 0x00, 0x01, 0x01);
        let e: u32x4 = u32x4::new(32, 32, 31, 31);
        let r: u32x4 = transmute(vclzq_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcagt_f32() {
        let a: f32x2 = f32x2::new(-1.2, 0.0);
        let b: f32x2 = f32x2::new(-1.1, 0.0);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0);
        let r: u32x2 = transmute(vcagt_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcagtq_f32() {
        let a: f32x4 = f32x4::new(-1.2, 0.0, 1.2, 2.3);
        let b: f32x4 = f32x4::new(-1.1, 0.0, 1.1, 2.4);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0, 0xFF_FF_FF_FF, 0);
        let r: u32x4 = transmute(vcagtq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcage_f32() {
        let a: f32x2 = f32x2::new(-1.2, 0.0);
        let b: f32x2 = f32x2::new(-1.1, 0.0);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcage_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcageq_f32() {
        let a: f32x4 = f32x4::new(-1.2, 0.0, 1.2, 2.3);
        let b: f32x4 = f32x4::new(-1.1, 0.0, 1.1, 2.4);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0);
        let r: u32x4 = transmute(vcageq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcalt_f32() {
        let a: f32x2 = f32x2::new(-1.2, 0.0);
        let b: f32x2 = f32x2::new(-1.1, 0.0);
        let e: u32x2 = u32x2::new(0, 0);
        let r: u32x2 = transmute(vcalt_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcaltq_f32() {
        let a: f32x4 = f32x4::new(-1.2, 0.0, 1.2, 2.3);
        let b: f32x4 = f32x4::new(-1.1, 0.0, 1.1, 2.4);
        let e: u32x4 = u32x4::new(0, 0, 0, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcaltq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcale_f32() {
        let a: f32x2 = f32x2::new(-1.2, 0.0);
        let b: f32x2 = f32x2::new(-1.1, 0.0);
        let e: u32x2 = u32x2::new(0, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vcale_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcaleq_f32() {
        let a: f32x4 = f32x4::new(-1.2, 0.0, 1.2, 2.3);
        let b: f32x4 = f32x4::new(-1.1, 0.0, 1.1, 2.4);
        let e: u32x4 = u32x4::new(0, 0xFF_FF_FF_FF, 0, 0xFF_FF_FF_FF);
        let r: u32x4 = transmute(vcaleq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcreate_s8() {
        let a: u64 = 1;
        let e: i8x8 = i8x8::new(1, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x8 = transmute(vcreate_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcreate_s32() {
        let a: u64 = 1;
        let e: i32x2 = i32x2::new(1, 0);
        let r: i32x2 = transmute(vcreate_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcreate_s64() {
        let a: u64 = 1;
        let e: i64x1 = i64x1::new(1);
        let r: i64x1 = transmute(vcreate_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcreate_u8() {
        let a: u64 = 1;
        let e: u8x8 = u8x8::new(1, 0, 0, 0, 0, 0, 0, 0);
        let r: u8x8 = transmute(vcreate_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcreate_u32() {
        let a: u64 = 1;
        let e: u32x2 = u32x2::new(1, 0);
        let r: u32x2 = transmute(vcreate_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcreate_u64() {
        let a: u64 = 1;
        let e: u64x1 = u64x1::new(1);
        let r: u64x1 = transmute(vcreate_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcreate_p8() {
        let a: u64 = 1;
        let e: i8x8 = i8x8::new(1, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x8 = transmute(vcreate_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcreate_p16() {
        let a: u64 = 1;
        let e: i16x4 = i16x4::new(1, 0, 0, 0);
        let r: i16x4 = transmute(vcreate_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcreate_p64() {
        let a: u64 = 1;
        let e: i64x1 = i64x1::new(1);
        let r: i64x1 = transmute(vcreate_p64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcreate_f32() {
        let a: u64 = 0;
        let e: f32x2 = f32x2::new(0., 0.);
        let r: f32x2 = transmute(vcreate_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_f32_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let e: f32x2 = f32x2::new(1., 2.);
        let r: f32x2 = transmute(vcvt_f32_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_f32_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let e: f32x4 = f32x4::new(1., 2., 3., 4.);
        let r: f32x4 = transmute(vcvtq_f32_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_f32_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let e: f32x2 = f32x2::new(1., 2.);
        let r: f32x2 = transmute(vcvt_f32_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_f32_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let e: f32x4 = f32x4::new(1., 2., 3., 4.);
        let r: f32x4 = transmute(vcvtq_f32_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_n_f32_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let e: f32x2 = f32x2::new(0.25, 0.5);
        let r: f32x2 = transmute(vcvt_n_f32_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_n_f32_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let e: f32x4 = f32x4::new(0.25, 0.5, 0.75, 1.);
        let r: f32x4 = transmute(vcvtq_n_f32_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_n_f32_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let e: f32x2 = f32x2::new(0.25, 0.5);
        let r: f32x2 = transmute(vcvt_n_f32_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_n_f32_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let e: f32x4 = f32x4::new(0.25, 0.5, 0.75, 1.);
        let r: f32x4 = transmute(vcvtq_n_f32_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_n_s32_f32() {
        let a: f32x2 = f32x2::new(0.25, 0.5);
        let e: i32x2 = i32x2::new(1, 2);
        let r: i32x2 = transmute(vcvt_n_s32_f32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_n_s32_f32() {
        let a: f32x4 = f32x4::new(0.25, 0.5, 0.75, 1.);
        let e: i32x4 = i32x4::new(1, 2, 3, 4);
        let r: i32x4 = transmute(vcvtq_n_s32_f32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_n_u32_f32() {
        let a: f32x2 = f32x2::new(0.25, 0.5);
        let e: u32x2 = u32x2::new(1, 2);
        let r: u32x2 = transmute(vcvt_n_u32_f32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_n_u32_f32() {
        let a: f32x4 = f32x4::new(0.25, 0.5, 0.75, 1.);
        let e: u32x4 = u32x4::new(1, 2, 3, 4);
        let r: u32x4 = transmute(vcvtq_n_u32_f32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_s32_f32() {
        let a: f32x2 = f32x2::new(-1.1, 2.1);
        let e: i32x2 = i32x2::new(-1, 2);
        let r: i32x2 = transmute(vcvt_s32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_s32_f32() {
        let a: f32x4 = f32x4::new(-1.1, 2.1, -2.9, 3.9);
        let e: i32x4 = i32x4::new(-1, 2, -2, 3);
        let r: i32x4 = transmute(vcvtq_s32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvt_u32_f32() {
        let a: f32x2 = f32x2::new(1.1, 2.1);
        let e: u32x2 = u32x2::new(1, 2);
        let r: u32x2 = transmute(vcvt_u32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vcvtq_u32_f32() {
        let a: f32x4 = f32x4::new(1.1, 2.1, 2.9, 3.9);
        let e: u32x4 = u32x4::new(1, 2, 2, 3);
        let r: u32x4 = transmute(vcvtq_u32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_lane_s8() {
        let a: i8x8 = i8x8::new(1, 1, 1, 4, 1, 6, 7, 8);
        let e: i8x8 = i8x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let r: i8x8 = transmute(vdup_lane_s8::<4>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_laneq_s8() {
        let a: i8x16 = i8x16::new(1, 1, 1, 4, 1, 6, 7, 8, 1, 10, 11, 12, 13, 14, 15, 16);
        let e: i8x16 = i8x16::new(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let r: i8x16 = transmute(vdupq_laneq_s8::<8>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_lane_s16() {
        let a: i16x4 = i16x4::new(1, 1, 1, 4);
        let e: i16x4 = i16x4::new(1, 1, 1, 1);
        let r: i16x4 = transmute(vdup_lane_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_laneq_s16() {
        let a: i16x8 = i16x8::new(1, 1, 1, 4, 1, 6, 7, 8);
        let e: i16x8 = i16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let r: i16x8 = transmute(vdupq_laneq_s16::<4>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_lane_s32() {
        let a: i32x2 = i32x2::new(1, 1);
        let e: i32x2 = i32x2::new(1, 1);
        let r: i32x2 = transmute(vdup_lane_s32::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_laneq_s32() {
        let a: i32x4 = i32x4::new(1, 1, 1, 4);
        let e: i32x4 = i32x4::new(1, 1, 1, 1);
        let r: i32x4 = transmute(vdupq_laneq_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_laneq_s8() {
        let a: i8x16 = i8x16::new(1, 1, 1, 4, 1, 6, 7, 8, 1, 10, 11, 12, 13, 14, 15, 16);
        let e: i8x8 = i8x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let r: i8x8 = transmute(vdup_laneq_s8::<8>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_laneq_s16() {
        let a: i16x8 = i16x8::new(1, 1, 1, 4, 1, 6, 7, 8);
        let e: i16x4 = i16x4::new(1, 1, 1, 1);
        let r: i16x4 = transmute(vdup_laneq_s16::<4>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_laneq_s32() {
        let a: i32x4 = i32x4::new(1, 1, 1, 4);
        let e: i32x2 = i32x2::new(1, 1);
        let r: i32x2 = transmute(vdup_laneq_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_lane_s8() {
        let a: i8x8 = i8x8::new(1, 1, 1, 4, 1, 6, 7, 8);
        let e: i8x16 = i8x16::new(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let r: i8x16 = transmute(vdupq_lane_s8::<4>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_lane_s16() {
        let a: i16x4 = i16x4::new(1, 1, 1, 4);
        let e: i16x8 = i16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let r: i16x8 = transmute(vdupq_lane_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_lane_s32() {
        let a: i32x2 = i32x2::new(1, 1);
        let e: i32x4 = i32x4::new(1, 1, 1, 1);
        let r: i32x4 = transmute(vdupq_lane_s32::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_lane_u8() {
        let a: u8x8 = u8x8::new(1, 1, 1, 4, 1, 6, 7, 8);
        let e: u8x8 = u8x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let r: u8x8 = transmute(vdup_lane_u8::<4>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_laneq_u8() {
        let a: u8x16 = u8x16::new(1, 1, 1, 4, 1, 6, 7, 8, 1, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let r: u8x16 = transmute(vdupq_laneq_u8::<8>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_lane_u16() {
        let a: u16x4 = u16x4::new(1, 1, 1, 4);
        let e: u16x4 = u16x4::new(1, 1, 1, 1);
        let r: u16x4 = transmute(vdup_lane_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_laneq_u16() {
        let a: u16x8 = u16x8::new(1, 1, 1, 4, 1, 6, 7, 8);
        let e: u16x8 = u16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let r: u16x8 = transmute(vdupq_laneq_u16::<4>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_lane_u32() {
        let a: u32x2 = u32x2::new(1, 1);
        let e: u32x2 = u32x2::new(1, 1);
        let r: u32x2 = transmute(vdup_lane_u32::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_laneq_u32() {
        let a: u32x4 = u32x4::new(1, 1, 1, 4);
        let e: u32x4 = u32x4::new(1, 1, 1, 1);
        let r: u32x4 = transmute(vdupq_laneq_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_laneq_u8() {
        let a: u8x16 = u8x16::new(1, 1, 1, 4, 1, 6, 7, 8, 1, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x8 = u8x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let r: u8x8 = transmute(vdup_laneq_u8::<8>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_laneq_u16() {
        let a: u16x8 = u16x8::new(1, 1, 1, 4, 1, 6, 7, 8);
        let e: u16x4 = u16x4::new(1, 1, 1, 1);
        let r: u16x4 = transmute(vdup_laneq_u16::<4>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_laneq_u32() {
        let a: u32x4 = u32x4::new(1, 1, 1, 4);
        let e: u32x2 = u32x2::new(1, 1);
        let r: u32x2 = transmute(vdup_laneq_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_lane_u8() {
        let a: u8x8 = u8x8::new(1, 1, 1, 4, 1, 6, 7, 8);
        let e: u8x16 = u8x16::new(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let r: u8x16 = transmute(vdupq_lane_u8::<4>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_lane_u16() {
        let a: u16x4 = u16x4::new(1, 1, 1, 4);
        let e: u16x8 = u16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let r: u16x8 = transmute(vdupq_lane_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_lane_u32() {
        let a: u32x2 = u32x2::new(1, 1);
        let e: u32x4 = u32x4::new(1, 1, 1, 1);
        let r: u32x4 = transmute(vdupq_lane_u32::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_lane_p8() {
        let a: i8x8 = i8x8::new(1, 1, 1, 4, 1, 6, 7, 8);
        let e: i8x8 = i8x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let r: i8x8 = transmute(vdup_lane_p8::<4>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_laneq_p8() {
        let a: i8x16 = i8x16::new(1, 1, 1, 4, 1, 6, 7, 8, 1, 10, 11, 12, 13, 14, 15, 16);
        let e: i8x16 = i8x16::new(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let r: i8x16 = transmute(vdupq_laneq_p8::<8>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_lane_p16() {
        let a: i16x4 = i16x4::new(1, 1, 1, 4);
        let e: i16x4 = i16x4::new(1, 1, 1, 1);
        let r: i16x4 = transmute(vdup_lane_p16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_laneq_p16() {
        let a: i16x8 = i16x8::new(1, 1, 1, 4, 1, 6, 7, 8);
        let e: i16x8 = i16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let r: i16x8 = transmute(vdupq_laneq_p16::<4>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_laneq_p8() {
        let a: i8x16 = i8x16::new(1, 1, 1, 4, 1, 6, 7, 8, 1, 10, 11, 12, 13, 14, 15, 16);
        let e: i8x8 = i8x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let r: i8x8 = transmute(vdup_laneq_p8::<8>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_laneq_p16() {
        let a: i16x8 = i16x8::new(1, 1, 1, 4, 1, 6, 7, 8);
        let e: i16x4 = i16x4::new(1, 1, 1, 1);
        let r: i16x4 = transmute(vdup_laneq_p16::<4>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_lane_p8() {
        let a: i8x8 = i8x8::new(1, 1, 1, 4, 1, 6, 7, 8);
        let e: i8x16 = i8x16::new(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let r: i8x16 = transmute(vdupq_lane_p8::<4>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_lane_p16() {
        let a: i16x4 = i16x4::new(1, 1, 1, 4);
        let e: i16x8 = i16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let r: i16x8 = transmute(vdupq_lane_p16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_laneq_s64() {
        let a: i64x2 = i64x2::new(1, 1);
        let e: i64x2 = i64x2::new(1, 1);
        let r: i64x2 = transmute(vdupq_laneq_s64::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_lane_s64() {
        let a: i64x1 = i64x1::new(1);
        let e: i64x2 = i64x2::new(1, 1);
        let r: i64x2 = transmute(vdupq_lane_s64::<0>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_laneq_u64() {
        let a: u64x2 = u64x2::new(1, 1);
        let e: u64x2 = u64x2::new(1, 1);
        let r: u64x2 = transmute(vdupq_laneq_u64::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_lane_u64() {
        let a: u64x1 = u64x1::new(1);
        let e: u64x2 = u64x2::new(1, 1);
        let r: u64x2 = transmute(vdupq_lane_u64::<0>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_lane_f32() {
        let a: f32x2 = f32x2::new(1., 1.);
        let e: f32x2 = f32x2::new(1., 1.);
        let r: f32x2 = transmute(vdup_lane_f32::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_laneq_f32() {
        let a: f32x4 = f32x4::new(1., 1., 1., 4.);
        let e: f32x4 = f32x4::new(1., 1., 1., 1.);
        let r: f32x4 = transmute(vdupq_laneq_f32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_laneq_f32() {
        let a: f32x4 = f32x4::new(1., 1., 1., 4.);
        let e: f32x2 = f32x2::new(1., 1.);
        let r: f32x2 = transmute(vdup_laneq_f32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdupq_lane_f32() {
        let a: f32x2 = f32x2::new(1., 1.);
        let e: f32x4 = f32x4::new(1., 1., 1., 1.);
        let r: f32x4 = transmute(vdupq_lane_f32::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_lane_s64() {
        let a: i64x1 = i64x1::new(0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vdup_lane_s64::<0>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_lane_u64() {
        let a: u64x1 = u64x1::new(0);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vdup_lane_u64::<0>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_laneq_s64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: i64x1 = i64x1::new(1);
        let r: i64x1 = transmute(vdup_laneq_s64::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vdup_laneq_u64() {
        let a: u64x2 = u64x2::new(0, 1);
        let e: u64x1 = u64x1::new(1);
        let r: u64x1 = transmute(vdup_laneq_u64::<1>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vext_s8() {
        let a: i8x8 = i8x8::new(0, 8, 8, 9, 8, 9, 9, 11);
        let b: i8x8 = i8x8::new(9, 11, 14, 15, 16, 17, 18, 19);
        let e: i8x8 = i8x8::new(8, 9, 9, 11, 9, 11, 14, 15);
        let r: i8x8 = transmute(vext_s8::<4>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vextq_s8() {
        let a: i8x16 = i8x16::new(0, 8, 8, 9, 8, 9, 9, 11, 8, 9, 9, 11, 9, 11, 14, 15);
        let b: i8x16 = i8x16::new(9, 11, 14, 15, 16, 17, 18, 19, 0, 8, 8, 9, 8, 9, 9, 11);
        let e: i8x16 = i8x16::new(8, 9, 9, 11, 9, 11, 14, 15, 9, 11, 14, 15, 16, 17, 18, 19);
        let r: i8x16 = transmute(vextq_s8::<8>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vext_s16() {
        let a: i16x4 = i16x4::new(0, 8, 8, 9);
        let b: i16x4 = i16x4::new(9, 11, 14, 15);
        let e: i16x4 = i16x4::new(8, 9, 9, 11);
        let r: i16x4 = transmute(vext_s16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vextq_s16() {
        let a: i16x8 = i16x8::new(0, 8, 8, 9, 8, 9, 9, 11);
        let b: i16x8 = i16x8::new(9, 11, 14, 15, 16, 17, 18, 19);
        let e: i16x8 = i16x8::new(8, 9, 9, 11, 9, 11, 14, 15);
        let r: i16x8 = transmute(vextq_s16::<4>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vext_s32() {
        let a: i32x2 = i32x2::new(0, 8);
        let b: i32x2 = i32x2::new(9, 11);
        let e: i32x2 = i32x2::new(8, 9);
        let r: i32x2 = transmute(vext_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vextq_s32() {
        let a: i32x4 = i32x4::new(0, 8, 8, 9);
        let b: i32x4 = i32x4::new(9, 11, 14, 15);
        let e: i32x4 = i32x4::new(8, 9, 9, 11);
        let r: i32x4 = transmute(vextq_s32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vext_u8() {
        let a: u8x8 = u8x8::new(0, 8, 8, 9, 8, 9, 9, 11);
        let b: u8x8 = u8x8::new(9, 11, 14, 15, 16, 17, 18, 19);
        let e: u8x8 = u8x8::new(8, 9, 9, 11, 9, 11, 14, 15);
        let r: u8x8 = transmute(vext_u8::<4>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vextq_u8() {
        let a: u8x16 = u8x16::new(0, 8, 8, 9, 8, 9, 9, 11, 8, 9, 9, 11, 9, 11, 14, 15);
        let b: u8x16 = u8x16::new(9, 11, 14, 15, 16, 17, 18, 19, 0, 8, 8, 9, 8, 9, 9, 11);
        let e: u8x16 = u8x16::new(8, 9, 9, 11, 9, 11, 14, 15, 9, 11, 14, 15, 16, 17, 18, 19);
        let r: u8x16 = transmute(vextq_u8::<8>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vext_u16() {
        let a: u16x4 = u16x4::new(0, 8, 8, 9);
        let b: u16x4 = u16x4::new(9, 11, 14, 15);
        let e: u16x4 = u16x4::new(8, 9, 9, 11);
        let r: u16x4 = transmute(vext_u16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vextq_u16() {
        let a: u16x8 = u16x8::new(0, 8, 8, 9, 8, 9, 9, 11);
        let b: u16x8 = u16x8::new(9, 11, 14, 15, 16, 17, 18, 19);
        let e: u16x8 = u16x8::new(8, 9, 9, 11, 9, 11, 14, 15);
        let r: u16x8 = transmute(vextq_u16::<4>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vext_u32() {
        let a: u32x2 = u32x2::new(0, 8);
        let b: u32x2 = u32x2::new(9, 11);
        let e: u32x2 = u32x2::new(8, 9);
        let r: u32x2 = transmute(vext_u32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vextq_u32() {
        let a: u32x4 = u32x4::new(0, 8, 8, 9);
        let b: u32x4 = u32x4::new(9, 11, 14, 15);
        let e: u32x4 = u32x4::new(8, 9, 9, 11);
        let r: u32x4 = transmute(vextq_u32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vext_p8() {
        let a: i8x8 = i8x8::new(0, 8, 8, 9, 8, 9, 9, 11);
        let b: i8x8 = i8x8::new(9, 11, 14, 15, 16, 17, 18, 19);
        let e: i8x8 = i8x8::new(8, 9, 9, 11, 9, 11, 14, 15);
        let r: i8x8 = transmute(vext_p8::<4>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vextq_p8() {
        let a: i8x16 = i8x16::new(0, 8, 8, 9, 8, 9, 9, 11, 8, 9, 9, 11, 9, 11, 14, 15);
        let b: i8x16 = i8x16::new(9, 11, 14, 15, 16, 17, 18, 19, 0, 8, 8, 9, 8, 9, 9, 11);
        let e: i8x16 = i8x16::new(8, 9, 9, 11, 9, 11, 14, 15, 9, 11, 14, 15, 16, 17, 18, 19);
        let r: i8x16 = transmute(vextq_p8::<8>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vext_p16() {
        let a: i16x4 = i16x4::new(0, 8, 8, 9);
        let b: i16x4 = i16x4::new(9, 11, 14, 15);
        let e: i16x4 = i16x4::new(8, 9, 9, 11);
        let r: i16x4 = transmute(vext_p16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vextq_p16() {
        let a: i16x8 = i16x8::new(0, 8, 8, 9, 8, 9, 9, 11);
        let b: i16x8 = i16x8::new(9, 11, 14, 15, 16, 17, 18, 19);
        let e: i16x8 = i16x8::new(8, 9, 9, 11, 9, 11, 14, 15);
        let r: i16x8 = transmute(vextq_p16::<4>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vextq_s64() {
        let a: i64x2 = i64x2::new(0, 8);
        let b: i64x2 = i64x2::new(9, 11);
        let e: i64x2 = i64x2::new(8, 9);
        let r: i64x2 = transmute(vextq_s64::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vextq_u64() {
        let a: u64x2 = u64x2::new(0, 8);
        let b: u64x2 = u64x2::new(9, 11);
        let e: u64x2 = u64x2::new(8, 9);
        let r: u64x2 = transmute(vextq_u64::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vext_f32() {
        let a: f32x2 = f32x2::new(0., 2.);
        let b: f32x2 = f32x2::new(3., 4.);
        let e: f32x2 = f32x2::new(2., 3.);
        let r: f32x2 = transmute(vext_f32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vextq_f32() {
        let a: f32x4 = f32x4::new(0., 2., 2., 3.);
        let b: f32x4 = f32x4::new(3., 4., 5., 6.);
        let e: f32x4 = f32x4::new(2., 3., 3., 4.);
        let r: f32x4 = transmute(vextq_f32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_s8() {
        let a: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: i8x8 = i8x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: i8x8 = i8x8::new(3, 3, 3, 3, 3, 3, 3, 3);
        let e: i8x8 = i8x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let r: i8x8 = transmute(vmla_s8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_s8() {
        let a: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b: i8x16 = i8x16::new(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let c: i8x16 = i8x16::new(3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3);
        let e: i8x16 = i8x16::new(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21);
        let r: i8x16 = transmute(vmlaq_s8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_s16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let c: i16x4 = i16x4::new(3, 3, 3, 3);
        let e: i16x4 = i16x4::new(6, 7, 8, 9);
        let r: i16x4 = transmute(vmla_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_s16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: i16x8 = i16x8::new(3, 3, 3, 3, 3, 3, 3, 3);
        let e: i16x8 = i16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let r: i16x8 = transmute(vmlaq_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_s32() {
        let a: i32x2 = i32x2::new(0, 1);
        let b: i32x2 = i32x2::new(2, 2);
        let c: i32x2 = i32x2::new(3, 3);
        let e: i32x2 = i32x2::new(6, 7);
        let r: i32x2 = transmute(vmla_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_s32() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let c: i32x4 = i32x4::new(3, 3, 3, 3);
        let e: i32x4 = i32x4::new(6, 7, 8, 9);
        let r: i32x4 = transmute(vmlaq_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_u8() {
        let a: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: u8x8 = u8x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: u8x8 = u8x8::new(3, 3, 3, 3, 3, 3, 3, 3);
        let e: u8x8 = u8x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let r: u8x8 = transmute(vmla_u8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_u8() {
        let a: u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b: u8x16 = u8x16::new(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let c: u8x16 = u8x16::new(3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3);
        let e: u8x16 = u8x16::new(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21);
        let r: u8x16 = transmute(vmlaq_u8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_u16() {
        let a: u16x4 = u16x4::new(0, 1, 2, 3);
        let b: u16x4 = u16x4::new(2, 2, 2, 2);
        let c: u16x4 = u16x4::new(3, 3, 3, 3);
        let e: u16x4 = u16x4::new(6, 7, 8, 9);
        let r: u16x4 = transmute(vmla_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_u16() {
        let a: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: u16x8 = u16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: u16x8 = u16x8::new(3, 3, 3, 3, 3, 3, 3, 3);
        let e: u16x8 = u16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let r: u16x8 = transmute(vmlaq_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_u32() {
        let a: u32x2 = u32x2::new(0, 1);
        let b: u32x2 = u32x2::new(2, 2);
        let c: u32x2 = u32x2::new(3, 3);
        let e: u32x2 = u32x2::new(6, 7);
        let r: u32x2 = transmute(vmla_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_u32() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let b: u32x4 = u32x4::new(2, 2, 2, 2);
        let c: u32x4 = u32x4::new(3, 3, 3, 3);
        let e: u32x4 = u32x4::new(6, 7, 8, 9);
        let r: u32x4 = transmute(vmlaq_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_f32() {
        let a: f32x2 = f32x2::new(0., 1.);
        let b: f32x2 = f32x2::new(2., 2.);
        let c: f32x2 = f32x2::new(3., 3.);
        let e: f32x2 = f32x2::new(6., 7.);
        let r: f32x2 = transmute(vmla_f32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_f32() {
        let a: f32x4 = f32x4::new(0., 1., 2., 3.);
        let b: f32x4 = f32x4::new(2., 2., 2., 2.);
        let c: f32x4 = f32x4::new(3., 3., 3., 3.);
        let e: f32x4 = f32x4::new(6., 7., 8., 9.);
        let r: f32x4 = transmute(vmlaq_f32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_n_s16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let c: i16 = 3;
        let e: i16x4 = i16x4::new(6, 7, 8, 9);
        let r: i16x4 = transmute(vmla_n_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_n_s16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: i16 = 3;
        let e: i16x8 = i16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let r: i16x8 = transmute(vmlaq_n_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_n_s32() {
        let a: i32x2 = i32x2::new(0, 1);
        let b: i32x2 = i32x2::new(2, 2);
        let c: i32 = 3;
        let e: i32x2 = i32x2::new(6, 7);
        let r: i32x2 = transmute(vmla_n_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_n_s32() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let c: i32 = 3;
        let e: i32x4 = i32x4::new(6, 7, 8, 9);
        let r: i32x4 = transmute(vmlaq_n_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_n_u16() {
        let a: u16x4 = u16x4::new(0, 1, 2, 3);
        let b: u16x4 = u16x4::new(2, 2, 2, 2);
        let c: u16 = 3;
        let e: u16x4 = u16x4::new(6, 7, 8, 9);
        let r: u16x4 = transmute(vmla_n_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_n_u16() {
        let a: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: u16x8 = u16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: u16 = 3;
        let e: u16x8 = u16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let r: u16x8 = transmute(vmlaq_n_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_n_u32() {
        let a: u32x2 = u32x2::new(0, 1);
        let b: u32x2 = u32x2::new(2, 2);
        let c: u32 = 3;
        let e: u32x2 = u32x2::new(6, 7);
        let r: u32x2 = transmute(vmla_n_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_n_u32() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let b: u32x4 = u32x4::new(2, 2, 2, 2);
        let c: u32 = 3;
        let e: u32x4 = u32x4::new(6, 7, 8, 9);
        let r: u32x4 = transmute(vmlaq_n_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_n_f32() {
        let a: f32x2 = f32x2::new(0., 1.);
        let b: f32x2 = f32x2::new(2., 2.);
        let c: f32 = 3.;
        let e: f32x2 = f32x2::new(6., 7.);
        let r: f32x2 = transmute(vmla_n_f32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_n_f32() {
        let a: f32x4 = f32x4::new(0., 1., 2., 3.);
        let b: f32x4 = f32x4::new(2., 2., 2., 2.);
        let c: f32 = 3.;
        let e: f32x4 = f32x4::new(6., 7., 8., 9.);
        let r: f32x4 = transmute(vmlaq_n_f32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_lane_s16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let c: i16x4 = i16x4::new(0, 3, 0, 0);
        let e: i16x4 = i16x4::new(6, 7, 8, 9);
        let r: i16x4 = transmute(vmla_lane_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_laneq_s16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let c: i16x8 = i16x8::new(0, 3, 0, 0, 0, 0, 0, 0);
        let e: i16x4 = i16x4::new(6, 7, 8, 9);
        let r: i16x4 = transmute(vmla_laneq_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_lane_s16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: i16x4 = i16x4::new(0, 3, 0, 0);
        let e: i16x8 = i16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let r: i16x8 = transmute(vmlaq_lane_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_laneq_s16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: i16x8 = i16x8::new(0, 3, 0, 0, 0, 0, 0, 0);
        let e: i16x8 = i16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let r: i16x8 = transmute(vmlaq_laneq_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_lane_s32() {
        let a: i32x2 = i32x2::new(0, 1);
        let b: i32x2 = i32x2::new(2, 2);
        let c: i32x2 = i32x2::new(0, 3);
        let e: i32x2 = i32x2::new(6, 7);
        let r: i32x2 = transmute(vmla_lane_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_laneq_s32() {
        let a: i32x2 = i32x2::new(0, 1);
        let b: i32x2 = i32x2::new(2, 2);
        let c: i32x4 = i32x4::new(0, 3, 0, 0);
        let e: i32x2 = i32x2::new(6, 7);
        let r: i32x2 = transmute(vmla_laneq_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_lane_s32() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let c: i32x2 = i32x2::new(0, 3);
        let e: i32x4 = i32x4::new(6, 7, 8, 9);
        let r: i32x4 = transmute(vmlaq_lane_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_laneq_s32() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let c: i32x4 = i32x4::new(0, 3, 0, 0);
        let e: i32x4 = i32x4::new(6, 7, 8, 9);
        let r: i32x4 = transmute(vmlaq_laneq_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_lane_u16() {
        let a: u16x4 = u16x4::new(0, 1, 2, 3);
        let b: u16x4 = u16x4::new(2, 2, 2, 2);
        let c: u16x4 = u16x4::new(0, 3, 0, 0);
        let e: u16x4 = u16x4::new(6, 7, 8, 9);
        let r: u16x4 = transmute(vmla_lane_u16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_laneq_u16() {
        let a: u16x4 = u16x4::new(0, 1, 2, 3);
        let b: u16x4 = u16x4::new(2, 2, 2, 2);
        let c: u16x8 = u16x8::new(0, 3, 0, 0, 0, 0, 0, 0);
        let e: u16x4 = u16x4::new(6, 7, 8, 9);
        let r: u16x4 = transmute(vmla_laneq_u16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_lane_u16() {
        let a: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: u16x8 = u16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: u16x4 = u16x4::new(0, 3, 0, 0);
        let e: u16x8 = u16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let r: u16x8 = transmute(vmlaq_lane_u16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_laneq_u16() {
        let a: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: u16x8 = u16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: u16x8 = u16x8::new(0, 3, 0, 0, 0, 0, 0, 0);
        let e: u16x8 = u16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let r: u16x8 = transmute(vmlaq_laneq_u16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_lane_u32() {
        let a: u32x2 = u32x2::new(0, 1);
        let b: u32x2 = u32x2::new(2, 2);
        let c: u32x2 = u32x2::new(0, 3);
        let e: u32x2 = u32x2::new(6, 7);
        let r: u32x2 = transmute(vmla_lane_u32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_laneq_u32() {
        let a: u32x2 = u32x2::new(0, 1);
        let b: u32x2 = u32x2::new(2, 2);
        let c: u32x4 = u32x4::new(0, 3, 0, 0);
        let e: u32x2 = u32x2::new(6, 7);
        let r: u32x2 = transmute(vmla_laneq_u32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_lane_u32() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let b: u32x4 = u32x4::new(2, 2, 2, 2);
        let c: u32x2 = u32x2::new(0, 3);
        let e: u32x4 = u32x4::new(6, 7, 8, 9);
        let r: u32x4 = transmute(vmlaq_lane_u32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_laneq_u32() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let b: u32x4 = u32x4::new(2, 2, 2, 2);
        let c: u32x4 = u32x4::new(0, 3, 0, 0);
        let e: u32x4 = u32x4::new(6, 7, 8, 9);
        let r: u32x4 = transmute(vmlaq_laneq_u32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_lane_f32() {
        let a: f32x2 = f32x2::new(0., 1.);
        let b: f32x2 = f32x2::new(2., 2.);
        let c: f32x2 = f32x2::new(0., 3.);
        let e: f32x2 = f32x2::new(6., 7.);
        let r: f32x2 = transmute(vmla_lane_f32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmla_laneq_f32() {
        let a: f32x2 = f32x2::new(0., 1.);
        let b: f32x2 = f32x2::new(2., 2.);
        let c: f32x4 = f32x4::new(0., 3., 0., 0.);
        let e: f32x2 = f32x2::new(6., 7.);
        let r: f32x2 = transmute(vmla_laneq_f32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_lane_f32() {
        let a: f32x4 = f32x4::new(0., 1., 2., 3.);
        let b: f32x4 = f32x4::new(2., 2., 2., 2.);
        let c: f32x2 = f32x2::new(0., 3.);
        let e: f32x4 = f32x4::new(6., 7., 8., 9.);
        let r: f32x4 = transmute(vmlaq_lane_f32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlaq_laneq_f32() {
        let a: f32x4 = f32x4::new(0., 1., 2., 3.);
        let b: f32x4 = f32x4::new(2., 2., 2., 2.);
        let c: f32x4 = f32x4::new(0., 3., 0., 0.);
        let e: f32x4 = f32x4::new(6., 7., 8., 9.);
        let r: f32x4 = transmute(vmlaq_laneq_f32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_s8() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: i8x8 = i8x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: i8x8 = i8x8::new(3, 3, 3, 3, 3, 3, 3, 3);
        let e: i16x8 = i16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let r: i16x8 = transmute(vmlal_s8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_s16() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let c: i16x4 = i16x4::new(3, 3, 3, 3);
        let e: i32x4 = i32x4::new(6, 7, 8, 9);
        let r: i32x4 = transmute(vmlal_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_s32() {
        let a: i64x2 = i64x2::new(0, 1);
        let b: i32x2 = i32x2::new(2, 2);
        let c: i32x2 = i32x2::new(3, 3);
        let e: i64x2 = i64x2::new(6, 7);
        let r: i64x2 = transmute(vmlal_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_u8() {
        let a: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: u8x8 = u8x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: u8x8 = u8x8::new(3, 3, 3, 3, 3, 3, 3, 3);
        let e: u16x8 = u16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let r: u16x8 = transmute(vmlal_u8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_u16() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let b: u16x4 = u16x4::new(2, 2, 2, 2);
        let c: u16x4 = u16x4::new(3, 3, 3, 3);
        let e: u32x4 = u32x4::new(6, 7, 8, 9);
        let r: u32x4 = transmute(vmlal_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_u32() {
        let a: u64x2 = u64x2::new(0, 1);
        let b: u32x2 = u32x2::new(2, 2);
        let c: u32x2 = u32x2::new(3, 3);
        let e: u64x2 = u64x2::new(6, 7);
        let r: u64x2 = transmute(vmlal_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_n_s16() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let c: i16 = 3;
        let e: i32x4 = i32x4::new(6, 7, 8, 9);
        let r: i32x4 = transmute(vmlal_n_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_n_s32() {
        let a: i64x2 = i64x2::new(0, 1);
        let b: i32x2 = i32x2::new(2, 2);
        let c: i32 = 3;
        let e: i64x2 = i64x2::new(6, 7);
        let r: i64x2 = transmute(vmlal_n_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_n_u16() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let b: u16x4 = u16x4::new(2, 2, 2, 2);
        let c: u16 = 3;
        let e: u32x4 = u32x4::new(6, 7, 8, 9);
        let r: u32x4 = transmute(vmlal_n_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_n_u32() {
        let a: u64x2 = u64x2::new(0, 1);
        let b: u32x2 = u32x2::new(2, 2);
        let c: u32 = 3;
        let e: u64x2 = u64x2::new(6, 7);
        let r: u64x2 = transmute(vmlal_n_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_lane_s16() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let c: i16x4 = i16x4::new(0, 3, 0, 0);
        let e: i32x4 = i32x4::new(6, 7, 8, 9);
        let r: i32x4 = transmute(vmlal_lane_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_laneq_s16() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let c: i16x8 = i16x8::new(0, 3, 0, 0, 0, 0, 0, 0);
        let e: i32x4 = i32x4::new(6, 7, 8, 9);
        let r: i32x4 = transmute(vmlal_laneq_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_lane_s32() {
        let a: i64x2 = i64x2::new(0, 1);
        let b: i32x2 = i32x2::new(2, 2);
        let c: i32x2 = i32x2::new(0, 3);
        let e: i64x2 = i64x2::new(6, 7);
        let r: i64x2 = transmute(vmlal_lane_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_laneq_s32() {
        let a: i64x2 = i64x2::new(0, 1);
        let b: i32x2 = i32x2::new(2, 2);
        let c: i32x4 = i32x4::new(0, 3, 0, 0);
        let e: i64x2 = i64x2::new(6, 7);
        let r: i64x2 = transmute(vmlal_laneq_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_lane_u16() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let b: u16x4 = u16x4::new(2, 2, 2, 2);
        let c: u16x4 = u16x4::new(0, 3, 0, 0);
        let e: u32x4 = u32x4::new(6, 7, 8, 9);
        let r: u32x4 = transmute(vmlal_lane_u16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_laneq_u16() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let b: u16x4 = u16x4::new(2, 2, 2, 2);
        let c: u16x8 = u16x8::new(0, 3, 0, 0, 0, 0, 0, 0);
        let e: u32x4 = u32x4::new(6, 7, 8, 9);
        let r: u32x4 = transmute(vmlal_laneq_u16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_lane_u32() {
        let a: u64x2 = u64x2::new(0, 1);
        let b: u32x2 = u32x2::new(2, 2);
        let c: u32x2 = u32x2::new(0, 3);
        let e: u64x2 = u64x2::new(6, 7);
        let r: u64x2 = transmute(vmlal_lane_u32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlal_laneq_u32() {
        let a: u64x2 = u64x2::new(0, 1);
        let b: u32x2 = u32x2::new(2, 2);
        let c: u32x4 = u32x4::new(0, 3, 0, 0);
        let e: u64x2 = u64x2::new(6, 7);
        let r: u64x2 = transmute(vmlal_laneq_u32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_s8() {
        let a: i8x8 = i8x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let b: i8x8 = i8x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: i8x8 = i8x8::new(3, 3, 3, 3, 3, 3, 3, 3);
        let e: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i8x8 = transmute(vmls_s8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_s8() {
        let a: i8x16 = i8x16::new(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21);
        let b: i8x16 = i8x16::new(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let c: i8x16 = i8x16::new(3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3);
        let e: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r: i8x16 = transmute(vmlsq_s8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_s16() {
        let a: i16x4 = i16x4::new(6, 7, 8, 9);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let c: i16x4 = i16x4::new(3, 3, 3, 3);
        let e: i16x4 = i16x4::new(0, 1, 2, 3);
        let r: i16x4 = transmute(vmls_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_s16() {
        let a: i16x8 = i16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: i16x8 = i16x8::new(3, 3, 3, 3, 3, 3, 3, 3);
        let e: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i16x8 = transmute(vmlsq_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_s32() {
        let a: i32x2 = i32x2::new(6, 7);
        let b: i32x2 = i32x2::new(2, 2);
        let c: i32x2 = i32x2::new(3, 3);
        let e: i32x2 = i32x2::new(0, 1);
        let r: i32x2 = transmute(vmls_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_s32() {
        let a: i32x4 = i32x4::new(6, 7, 8, 9);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let c: i32x4 = i32x4::new(3, 3, 3, 3);
        let e: i32x4 = i32x4::new(0, 1, 2, 3);
        let r: i32x4 = transmute(vmlsq_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_u8() {
        let a: u8x8 = u8x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let b: u8x8 = u8x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: u8x8 = u8x8::new(3, 3, 3, 3, 3, 3, 3, 3);
        let e: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: u8x8 = transmute(vmls_u8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_u8() {
        let a: u8x16 = u8x16::new(6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21);
        let b: u8x16 = u8x16::new(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let c: u8x16 = u8x16::new(3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3);
        let e: u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r: u8x16 = transmute(vmlsq_u8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_u16() {
        let a: u16x4 = u16x4::new(6, 7, 8, 9);
        let b: u16x4 = u16x4::new(2, 2, 2, 2);
        let c: u16x4 = u16x4::new(3, 3, 3, 3);
        let e: u16x4 = u16x4::new(0, 1, 2, 3);
        let r: u16x4 = transmute(vmls_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_u16() {
        let a: u16x8 = u16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let b: u16x8 = u16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: u16x8 = u16x8::new(3, 3, 3, 3, 3, 3, 3, 3);
        let e: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: u16x8 = transmute(vmlsq_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_u32() {
        let a: u32x2 = u32x2::new(6, 7);
        let b: u32x2 = u32x2::new(2, 2);
        let c: u32x2 = u32x2::new(3, 3);
        let e: u32x2 = u32x2::new(0, 1);
        let r: u32x2 = transmute(vmls_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_u32() {
        let a: u32x4 = u32x4::new(6, 7, 8, 9);
        let b: u32x4 = u32x4::new(2, 2, 2, 2);
        let c: u32x4 = u32x4::new(3, 3, 3, 3);
        let e: u32x4 = u32x4::new(0, 1, 2, 3);
        let r: u32x4 = transmute(vmlsq_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_f32() {
        let a: f32x2 = f32x2::new(6., 7.);
        let b: f32x2 = f32x2::new(2., 2.);
        let c: f32x2 = f32x2::new(3., 3.);
        let e: f32x2 = f32x2::new(0., 1.);
        let r: f32x2 = transmute(vmls_f32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_f32() {
        let a: f32x4 = f32x4::new(6., 7., 8., 9.);
        let b: f32x4 = f32x4::new(2., 2., 2., 2.);
        let c: f32x4 = f32x4::new(3., 3., 3., 3.);
        let e: f32x4 = f32x4::new(0., 1., 2., 3.);
        let r: f32x4 = transmute(vmlsq_f32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_n_s16() {
        let a: i16x4 = i16x4::new(6, 7, 8, 9);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let c: i16 = 3;
        let e: i16x4 = i16x4::new(0, 1, 2, 3);
        let r: i16x4 = transmute(vmls_n_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_n_s16() {
        let a: i16x8 = i16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: i16 = 3;
        let e: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i16x8 = transmute(vmlsq_n_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_n_s32() {
        let a: i32x2 = i32x2::new(6, 7);
        let b: i32x2 = i32x2::new(2, 2);
        let c: i32 = 3;
        let e: i32x2 = i32x2::new(0, 1);
        let r: i32x2 = transmute(vmls_n_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_n_s32() {
        let a: i32x4 = i32x4::new(6, 7, 8, 9);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let c: i32 = 3;
        let e: i32x4 = i32x4::new(0, 1, 2, 3);
        let r: i32x4 = transmute(vmlsq_n_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_n_u16() {
        let a: u16x4 = u16x4::new(6, 7, 8, 9);
        let b: u16x4 = u16x4::new(2, 2, 2, 2);
        let c: u16 = 3;
        let e: u16x4 = u16x4::new(0, 1, 2, 3);
        let r: u16x4 = transmute(vmls_n_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_n_u16() {
        let a: u16x8 = u16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let b: u16x8 = u16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: u16 = 3;
        let e: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: u16x8 = transmute(vmlsq_n_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_n_u32() {
        let a: u32x2 = u32x2::new(6, 7);
        let b: u32x2 = u32x2::new(2, 2);
        let c: u32 = 3;
        let e: u32x2 = u32x2::new(0, 1);
        let r: u32x2 = transmute(vmls_n_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_n_u32() {
        let a: u32x4 = u32x4::new(6, 7, 8, 9);
        let b: u32x4 = u32x4::new(2, 2, 2, 2);
        let c: u32 = 3;
        let e: u32x4 = u32x4::new(0, 1, 2, 3);
        let r: u32x4 = transmute(vmlsq_n_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_n_f32() {
        let a: f32x2 = f32x2::new(6., 7.);
        let b: f32x2 = f32x2::new(2., 2.);
        let c: f32 = 3.;
        let e: f32x2 = f32x2::new(0., 1.);
        let r: f32x2 = transmute(vmls_n_f32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_n_f32() {
        let a: f32x4 = f32x4::new(6., 7., 8., 9.);
        let b: f32x4 = f32x4::new(2., 2., 2., 2.);
        let c: f32 = 3.;
        let e: f32x4 = f32x4::new(0., 1., 2., 3.);
        let r: f32x4 = transmute(vmlsq_n_f32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_lane_s16() {
        let a: i16x4 = i16x4::new(6, 7, 8, 9);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let c: i16x4 = i16x4::new(0, 3, 0, 0);
        let e: i16x4 = i16x4::new(0, 1, 2, 3);
        let r: i16x4 = transmute(vmls_lane_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_laneq_s16() {
        let a: i16x4 = i16x4::new(6, 7, 8, 9);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let c: i16x8 = i16x8::new(0, 3, 0, 0, 0, 0, 0, 0);
        let e: i16x4 = i16x4::new(0, 1, 2, 3);
        let r: i16x4 = transmute(vmls_laneq_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_lane_s16() {
        let a: i16x8 = i16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: i16x4 = i16x4::new(0, 3, 0, 0);
        let e: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i16x8 = transmute(vmlsq_lane_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_laneq_s16() {
        let a: i16x8 = i16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: i16x8 = i16x8::new(0, 3, 0, 0, 0, 0, 0, 0);
        let e: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i16x8 = transmute(vmlsq_laneq_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_lane_s32() {
        let a: i32x2 = i32x2::new(6, 7);
        let b: i32x2 = i32x2::new(2, 2);
        let c: i32x2 = i32x2::new(0, 3);
        let e: i32x2 = i32x2::new(0, 1);
        let r: i32x2 = transmute(vmls_lane_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_laneq_s32() {
        let a: i32x2 = i32x2::new(6, 7);
        let b: i32x2 = i32x2::new(2, 2);
        let c: i32x4 = i32x4::new(0, 3, 0, 0);
        let e: i32x2 = i32x2::new(0, 1);
        let r: i32x2 = transmute(vmls_laneq_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_lane_s32() {
        let a: i32x4 = i32x4::new(6, 7, 8, 9);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let c: i32x2 = i32x2::new(0, 3);
        let e: i32x4 = i32x4::new(0, 1, 2, 3);
        let r: i32x4 = transmute(vmlsq_lane_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_laneq_s32() {
        let a: i32x4 = i32x4::new(6, 7, 8, 9);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let c: i32x4 = i32x4::new(0, 3, 0, 0);
        let e: i32x4 = i32x4::new(0, 1, 2, 3);
        let r: i32x4 = transmute(vmlsq_laneq_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_lane_u16() {
        let a: u16x4 = u16x4::new(6, 7, 8, 9);
        let b: u16x4 = u16x4::new(2, 2, 2, 2);
        let c: u16x4 = u16x4::new(0, 3, 0, 0);
        let e: u16x4 = u16x4::new(0, 1, 2, 3);
        let r: u16x4 = transmute(vmls_lane_u16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_laneq_u16() {
        let a: u16x4 = u16x4::new(6, 7, 8, 9);
        let b: u16x4 = u16x4::new(2, 2, 2, 2);
        let c: u16x8 = u16x8::new(0, 3, 0, 0, 0, 0, 0, 0);
        let e: u16x4 = u16x4::new(0, 1, 2, 3);
        let r: u16x4 = transmute(vmls_laneq_u16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_lane_u16() {
        let a: u16x8 = u16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let b: u16x8 = u16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: u16x4 = u16x4::new(0, 3, 0, 0);
        let e: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: u16x8 = transmute(vmlsq_lane_u16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_laneq_u16() {
        let a: u16x8 = u16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let b: u16x8 = u16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: u16x8 = u16x8::new(0, 3, 0, 0, 0, 0, 0, 0);
        let e: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: u16x8 = transmute(vmlsq_laneq_u16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_lane_u32() {
        let a: u32x2 = u32x2::new(6, 7);
        let b: u32x2 = u32x2::new(2, 2);
        let c: u32x2 = u32x2::new(0, 3);
        let e: u32x2 = u32x2::new(0, 1);
        let r: u32x2 = transmute(vmls_lane_u32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_laneq_u32() {
        let a: u32x2 = u32x2::new(6, 7);
        let b: u32x2 = u32x2::new(2, 2);
        let c: u32x4 = u32x4::new(0, 3, 0, 0);
        let e: u32x2 = u32x2::new(0, 1);
        let r: u32x2 = transmute(vmls_laneq_u32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_lane_u32() {
        let a: u32x4 = u32x4::new(6, 7, 8, 9);
        let b: u32x4 = u32x4::new(2, 2, 2, 2);
        let c: u32x2 = u32x2::new(0, 3);
        let e: u32x4 = u32x4::new(0, 1, 2, 3);
        let r: u32x4 = transmute(vmlsq_lane_u32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_laneq_u32() {
        let a: u32x4 = u32x4::new(6, 7, 8, 9);
        let b: u32x4 = u32x4::new(2, 2, 2, 2);
        let c: u32x4 = u32x4::new(0, 3, 0, 0);
        let e: u32x4 = u32x4::new(0, 1, 2, 3);
        let r: u32x4 = transmute(vmlsq_laneq_u32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_lane_f32() {
        let a: f32x2 = f32x2::new(6., 7.);
        let b: f32x2 = f32x2::new(2., 2.);
        let c: f32x2 = f32x2::new(0., 3.);
        let e: f32x2 = f32x2::new(0., 1.);
        let r: f32x2 = transmute(vmls_lane_f32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmls_laneq_f32() {
        let a: f32x2 = f32x2::new(6., 7.);
        let b: f32x2 = f32x2::new(2., 2.);
        let c: f32x4 = f32x4::new(0., 3., 0., 0.);
        let e: f32x2 = f32x2::new(0., 1.);
        let r: f32x2 = transmute(vmls_laneq_f32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_lane_f32() {
        let a: f32x4 = f32x4::new(6., 7., 8., 9.);
        let b: f32x4 = f32x4::new(2., 2., 2., 2.);
        let c: f32x2 = f32x2::new(0., 3.);
        let e: f32x4 = f32x4::new(0., 1., 2., 3.);
        let r: f32x4 = transmute(vmlsq_lane_f32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsq_laneq_f32() {
        let a: f32x4 = f32x4::new(6., 7., 8., 9.);
        let b: f32x4 = f32x4::new(2., 2., 2., 2.);
        let c: f32x4 = f32x4::new(0., 3., 0., 0.);
        let e: f32x4 = f32x4::new(0., 1., 2., 3.);
        let r: f32x4 = transmute(vmlsq_laneq_f32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_s8() {
        let a: i16x8 = i16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let b: i8x8 = i8x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: i8x8 = i8x8::new(3, 3, 3, 3, 3, 3, 3, 3);
        let e: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i16x8 = transmute(vmlsl_s8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_s16() {
        let a: i32x4 = i32x4::new(6, 7, 8, 9);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let c: i16x4 = i16x4::new(3, 3, 3, 3);
        let e: i32x4 = i32x4::new(0, 1, 2, 3);
        let r: i32x4 = transmute(vmlsl_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_s32() {
        let a: i64x2 = i64x2::new(6, 7);
        let b: i32x2 = i32x2::new(2, 2);
        let c: i32x2 = i32x2::new(3, 3);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vmlsl_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_u8() {
        let a: u16x8 = u16x8::new(6, 7, 8, 9, 10, 11, 12, 13);
        let b: u8x8 = u8x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let c: u8x8 = u8x8::new(3, 3, 3, 3, 3, 3, 3, 3);
        let e: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: u16x8 = transmute(vmlsl_u8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_u16() {
        let a: u32x4 = u32x4::new(6, 7, 8, 9);
        let b: u16x4 = u16x4::new(2, 2, 2, 2);
        let c: u16x4 = u16x4::new(3, 3, 3, 3);
        let e: u32x4 = u32x4::new(0, 1, 2, 3);
        let r: u32x4 = transmute(vmlsl_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_u32() {
        let a: u64x2 = u64x2::new(6, 7);
        let b: u32x2 = u32x2::new(2, 2);
        let c: u32x2 = u32x2::new(3, 3);
        let e: u64x2 = u64x2::new(0, 1);
        let r: u64x2 = transmute(vmlsl_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_n_s16() {
        let a: i32x4 = i32x4::new(6, 7, 8, 9);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let c: i16 = 3;
        let e: i32x4 = i32x4::new(0, 1, 2, 3);
        let r: i32x4 = transmute(vmlsl_n_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_n_s32() {
        let a: i64x2 = i64x2::new(6, 7);
        let b: i32x2 = i32x2::new(2, 2);
        let c: i32 = 3;
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vmlsl_n_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_n_u16() {
        let a: u32x4 = u32x4::new(6, 7, 8, 9);
        let b: u16x4 = u16x4::new(2, 2, 2, 2);
        let c: u16 = 3;
        let e: u32x4 = u32x4::new(0, 1, 2, 3);
        let r: u32x4 = transmute(vmlsl_n_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_n_u32() {
        let a: u64x2 = u64x2::new(6, 7);
        let b: u32x2 = u32x2::new(2, 2);
        let c: u32 = 3;
        let e: u64x2 = u64x2::new(0, 1);
        let r: u64x2 = transmute(vmlsl_n_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_lane_s16() {
        let a: i32x4 = i32x4::new(6, 7, 8, 9);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let c: i16x4 = i16x4::new(0, 3, 0, 0);
        let e: i32x4 = i32x4::new(0, 1, 2, 3);
        let r: i32x4 = transmute(vmlsl_lane_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_laneq_s16() {
        let a: i32x4 = i32x4::new(6, 7, 8, 9);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let c: i16x8 = i16x8::new(0, 3, 0, 0, 0, 0, 0, 0);
        let e: i32x4 = i32x4::new(0, 1, 2, 3);
        let r: i32x4 = transmute(vmlsl_laneq_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_lane_s32() {
        let a: i64x2 = i64x2::new(6, 7);
        let b: i32x2 = i32x2::new(2, 2);
        let c: i32x2 = i32x2::new(0, 3);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vmlsl_lane_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_laneq_s32() {
        let a: i64x2 = i64x2::new(6, 7);
        let b: i32x2 = i32x2::new(2, 2);
        let c: i32x4 = i32x4::new(0, 3, 0, 0);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vmlsl_laneq_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_lane_u16() {
        let a: u32x4 = u32x4::new(6, 7, 8, 9);
        let b: u16x4 = u16x4::new(2, 2, 2, 2);
        let c: u16x4 = u16x4::new(0, 3, 0, 0);
        let e: u32x4 = u32x4::new(0, 1, 2, 3);
        let r: u32x4 = transmute(vmlsl_lane_u16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_laneq_u16() {
        let a: u32x4 = u32x4::new(6, 7, 8, 9);
        let b: u16x4 = u16x4::new(2, 2, 2, 2);
        let c: u16x8 = u16x8::new(0, 3, 0, 0, 0, 0, 0, 0);
        let e: u32x4 = u32x4::new(0, 1, 2, 3);
        let r: u32x4 = transmute(vmlsl_laneq_u16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_lane_u32() {
        let a: u64x2 = u64x2::new(6, 7);
        let b: u32x2 = u32x2::new(2, 2);
        let c: u32x2 = u32x2::new(0, 3);
        let e: u64x2 = u64x2::new(0, 1);
        let r: u64x2 = transmute(vmlsl_lane_u32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmlsl_laneq_u32() {
        let a: u64x2 = u64x2::new(6, 7);
        let b: u32x2 = u32x2::new(2, 2);
        let c: u32x4 = u32x4::new(0, 3, 0, 0);
        let e: u64x2 = u64x2::new(0, 1);
        let r: u64x2 = transmute(vmlsl_laneq_u32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vneg_s8() {
        let a: i8x8 = i8x8::new(0, 1, -1, 2, -2, 3, -3, 4);
        let e: i8x8 = i8x8::new(0, -1, 1, -2, 2, -3, 3, -4);
        let r: i8x8 = transmute(vneg_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vnegq_s8() {
        let a: i8x16 = i8x16::new(0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7, 8);
        let e: i8x16 = i8x16::new(0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8);
        let r: i8x16 = transmute(vnegq_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vneg_s16() {
        let a: i16x4 = i16x4::new(0, 1, -1, 2);
        let e: i16x4 = i16x4::new(0, -1, 1, -2);
        let r: i16x4 = transmute(vneg_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vnegq_s16() {
        let a: i16x8 = i16x8::new(0, 1, -1, 2, -2, 3, -3, 4);
        let e: i16x8 = i16x8::new(0, -1, 1, -2, 2, -3, 3, -4);
        let r: i16x8 = transmute(vnegq_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vneg_s32() {
        let a: i32x2 = i32x2::new(0, 1);
        let e: i32x2 = i32x2::new(0, -1);
        let r: i32x2 = transmute(vneg_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vnegq_s32() {
        let a: i32x4 = i32x4::new(0, 1, -1, 2);
        let e: i32x4 = i32x4::new(0, -1, 1, -2);
        let r: i32x4 = transmute(vnegq_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vneg_f32() {
        let a: f32x2 = f32x2::new(0., 1.);
        let e: f32x2 = f32x2::new(0., -1.);
        let r: f32x2 = transmute(vneg_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vnegq_f32() {
        let a: f32x4 = f32x4::new(0., 1., -1., 2.);
        let e: f32x4 = f32x4::new(0., -1., 1., -2.);
        let r: f32x4 = transmute(vnegq_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqneg_s8() {
        let a: i8x8 = i8x8::new(-128, 0, 1, -1, 2, -2, 3, -3);
        let e: i8x8 = i8x8::new(0x7F, 0, -1, 1, -2, 2, -3, 3);
        let r: i8x8 = transmute(vqneg_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqnegq_s8() {
        let a: i8x16 = i8x16::new(-128, 0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7);
        let e: i8x16 = i8x16::new(0x7F, 0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7);
        let r: i8x16 = transmute(vqnegq_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqneg_s16() {
        let a: i16x4 = i16x4::new(-32768, 0, 1, -1);
        let e: i16x4 = i16x4::new(0x7F_FF, 0, -1, 1);
        let r: i16x4 = transmute(vqneg_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqnegq_s16() {
        let a: i16x8 = i16x8::new(-32768, 0, 1, -1, 2, -2, 3, -3);
        let e: i16x8 = i16x8::new(0x7F_FF, 0, -1, 1, -2, 2, -3, 3);
        let r: i16x8 = transmute(vqnegq_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqneg_s32() {
        let a: i32x2 = i32x2::new(-2147483648, 0);
        let e: i32x2 = i32x2::new(0x7F_FF_FF_FF, 0);
        let r: i32x2 = transmute(vqneg_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqnegq_s32() {
        let a: i32x4 = i32x4::new(-2147483648, 0, 1, -1);
        let e: i32x4 = i32x4::new(0x7F_FF_FF_FF, 0, -1, 1);
        let r: i32x4 = transmute(vqnegq_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_u8() {
        let a: u8x8 = u8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(41, 40, 39, 38, 37, 36, 35, 34);
        let r: u8x8 = transmute(vqsub_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_u8() {
        let a: u8x16 = u8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26);
        let r: u8x16 = transmute(vqsubq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_u16() {
        let a: u16x4 = u16x4::new(42, 42, 42, 42);
        let b: u16x4 = u16x4::new(1, 2, 3, 4);
        let e: u16x4 = u16x4::new(41, 40, 39, 38);
        let r: u16x4 = transmute(vqsub_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_u16() {
        let a: u16x8 = u16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(41, 40, 39, 38, 37, 36, 35, 34);
        let r: u16x8 = transmute(vqsubq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_u32() {
        let a: u32x2 = u32x2::new(42, 42);
        let b: u32x2 = u32x2::new(1, 2);
        let e: u32x2 = u32x2::new(41, 40);
        let r: u32x2 = transmute(vqsub_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_u32() {
        let a: u32x4 = u32x4::new(42, 42, 42, 42);
        let b: u32x4 = u32x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(41, 40, 39, 38);
        let r: u32x4 = transmute(vqsubq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_u64() {
        let a: u64x1 = u64x1::new(42);
        let b: u64x1 = u64x1::new(1);
        let e: u64x1 = u64x1::new(41);
        let r: u64x1 = transmute(vqsub_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_u64() {
        let a: u64x2 = u64x2::new(42, 42);
        let b: u64x2 = u64x2::new(1, 2);
        let e: u64x2 = u64x2::new(41, 40);
        let r: u64x2 = transmute(vqsubq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_s8() {
        let a: i8x8 = i8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i8x8 = i8x8::new(41, 40, 39, 38, 37, 36, 35, 34);
        let r: i8x8 = transmute(vqsub_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_s8() {
        let a: i8x16 = i8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: i8x16 = i8x16::new(41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 29, 28, 27, 26);
        let r: i8x16 = transmute(vqsubq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_s16() {
        let a: i16x4 = i16x4::new(42, 42, 42, 42);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let e: i16x4 = i16x4::new(41, 40, 39, 38);
        let r: i16x4 = transmute(vqsub_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_s16() {
        let a: i16x8 = i16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i16x8 = i16x8::new(41, 40, 39, 38, 37, 36, 35, 34);
        let r: i16x8 = transmute(vqsubq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_s32() {
        let a: i32x2 = i32x2::new(42, 42);
        let b: i32x2 = i32x2::new(1, 2);
        let e: i32x2 = i32x2::new(41, 40);
        let r: i32x2 = transmute(vqsub_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_s32() {
        let a: i32x4 = i32x4::new(42, 42, 42, 42);
        let b: i32x4 = i32x4::new(1, 2, 3, 4);
        let e: i32x4 = i32x4::new(41, 40, 39, 38);
        let r: i32x4 = transmute(vqsubq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsub_s64() {
        let a: i64x1 = i64x1::new(42);
        let b: i64x1 = i64x1::new(1);
        let e: i64x1 = i64x1::new(41);
        let r: i64x1 = transmute(vqsub_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqsubq_s64() {
        let a: i64x2 = i64x2::new(42, 42);
        let b: i64x2 = i64x2::new(1, 2);
        let e: i64x2 = i64x2::new(41, 40);
        let r: i64x2 = transmute(vqsubq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_u8() {
        let a: u8x8 = u8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(21, 22, 22, 23, 23, 24, 24, 25);
        let r: u8x8 = transmute(vhadd_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_u8() {
        let a: u8x16 = u8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29);
        let r: u8x16 = transmute(vhaddq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_u16() {
        let a: u16x4 = u16x4::new(42, 42, 42, 42);
        let b: u16x4 = u16x4::new(1, 2, 3, 4);
        let e: u16x4 = u16x4::new(21, 22, 22, 23);
        let r: u16x4 = transmute(vhadd_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_u16() {
        let a: u16x8 = u16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(21, 22, 22, 23, 23, 24, 24, 25);
        let r: u16x8 = transmute(vhaddq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_u32() {
        let a: u32x2 = u32x2::new(42, 42);
        let b: u32x2 = u32x2::new(1, 2);
        let e: u32x2 = u32x2::new(21, 22);
        let r: u32x2 = transmute(vhadd_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_u32() {
        let a: u32x4 = u32x4::new(42, 42, 42, 42);
        let b: u32x4 = u32x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(21, 22, 22, 23);
        let r: u32x4 = transmute(vhaddq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_s8() {
        let a: i8x8 = i8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i8x8 = i8x8::new(21, 22, 22, 23, 23, 24, 24, 25);
        let r: i8x8 = transmute(vhadd_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_s8() {
        let a: i8x16 = i8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: i8x16 = i8x16::new(21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29);
        let r: i8x16 = transmute(vhaddq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_s16() {
        let a: i16x4 = i16x4::new(42, 42, 42, 42);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let e: i16x4 = i16x4::new(21, 22, 22, 23);
        let r: i16x4 = transmute(vhadd_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_s16() {
        let a: i16x8 = i16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i16x8 = i16x8::new(21, 22, 22, 23, 23, 24, 24, 25);
        let r: i16x8 = transmute(vhaddq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhadd_s32() {
        let a: i32x2 = i32x2::new(42, 42);
        let b: i32x2 = i32x2::new(1, 2);
        let e: i32x2 = i32x2::new(21, 22);
        let r: i32x2 = transmute(vhadd_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhaddq_s32() {
        let a: i32x4 = i32x4::new(42, 42, 42, 42);
        let b: i32x4 = i32x4::new(1, 2, 3, 4);
        let e: i32x4 = i32x4::new(21, 22, 22, 23);
        let r: i32x4 = transmute(vhaddq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_u8() {
        let a: u8x8 = u8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(22, 22, 23, 23, 24, 24, 25, 25);
        let r: u8x8 = transmute(vrhadd_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_u8() {
        let a: u8x16 = u8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29);
        let r: u8x16 = transmute(vrhaddq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_u16() {
        let a: u16x4 = u16x4::new(42, 42, 42, 42);
        let b: u16x4 = u16x4::new(1, 2, 3, 4);
        let e: u16x4 = u16x4::new(22, 22, 23, 23);
        let r: u16x4 = transmute(vrhadd_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_u16() {
        let a: u16x8 = u16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(22, 22, 23, 23, 24, 24, 25, 25);
        let r: u16x8 = transmute(vrhaddq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_u32() {
        let a: u32x2 = u32x2::new(42, 42);
        let b: u32x2 = u32x2::new(1, 2);
        let e: u32x2 = u32x2::new(22, 22);
        let r: u32x2 = transmute(vrhadd_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_u32() {
        let a: u32x4 = u32x4::new(42, 42, 42, 42);
        let b: u32x4 = u32x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(22, 22, 23, 23);
        let r: u32x4 = transmute(vrhaddq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_s8() {
        let a: i8x8 = i8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i8x8 = i8x8::new(22, 22, 23, 23, 24, 24, 25, 25);
        let r: i8x8 = transmute(vrhadd_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_s8() {
        let a: i8x16 = i8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: i8x16 = i8x16::new(22, 22, 23, 23, 24, 24, 25, 25, 26, 26, 27, 27, 28, 28, 29, 29);
        let r: i8x16 = transmute(vrhaddq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_s16() {
        let a: i16x4 = i16x4::new(42, 42, 42, 42);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let e: i16x4 = i16x4::new(22, 22, 23, 23);
        let r: i16x4 = transmute(vrhadd_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_s16() {
        let a: i16x8 = i16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i16x8 = i16x8::new(22, 22, 23, 23, 24, 24, 25, 25);
        let r: i16x8 = transmute(vrhaddq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhadd_s32() {
        let a: i32x2 = i32x2::new(42, 42);
        let b: i32x2 = i32x2::new(1, 2);
        let e: i32x2 = i32x2::new(22, 22);
        let r: i32x2 = transmute(vrhadd_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrhaddq_s32() {
        let a: i32x4 = i32x4::new(42, 42, 42, 42);
        let b: i32x4 = i32x4::new(1, 2, 3, 4);
        let e: i32x4 = i32x4::new(22, 22, 23, 23);
        let r: i32x4 = transmute(vrhaddq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndn_f32() {
        let a: f32x2 = f32x2::new(-1.5, 0.5);
        let e: f32x2 = f32x2::new(-2.0, 0.0);
        let r: f32x2 = transmute(vrndn_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrndnq_f32() {
        let a: f32x4 = f32x4::new(-1.5, 0.5, 1.5, 2.5);
        let e: f32x4 = f32x4::new(-2.0, 0.0, 2.0, 2.0);
        let r: f32x4 = transmute(vrndnq_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_u8() {
        let a: u8x8 = u8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(43, 44, 45, 46, 47, 48, 49, 50);
        let r: u8x8 = transmute(vqadd_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_u8() {
        let a: u8x16 = u8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58);
        let r: u8x16 = transmute(vqaddq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_u16() {
        let a: u16x4 = u16x4::new(42, 42, 42, 42);
        let b: u16x4 = u16x4::new(1, 2, 3, 4);
        let e: u16x4 = u16x4::new(43, 44, 45, 46);
        let r: u16x4 = transmute(vqadd_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_u16() {
        let a: u16x8 = u16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(43, 44, 45, 46, 47, 48, 49, 50);
        let r: u16x8 = transmute(vqaddq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_u32() {
        let a: u32x2 = u32x2::new(42, 42);
        let b: u32x2 = u32x2::new(1, 2);
        let e: u32x2 = u32x2::new(43, 44);
        let r: u32x2 = transmute(vqadd_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_u32() {
        let a: u32x4 = u32x4::new(42, 42, 42, 42);
        let b: u32x4 = u32x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(43, 44, 45, 46);
        let r: u32x4 = transmute(vqaddq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_u64() {
        let a: u64x1 = u64x1::new(42);
        let b: u64x1 = u64x1::new(1);
        let e: u64x1 = u64x1::new(43);
        let r: u64x1 = transmute(vqadd_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_u64() {
        let a: u64x2 = u64x2::new(42, 42);
        let b: u64x2 = u64x2::new(1, 2);
        let e: u64x2 = u64x2::new(43, 44);
        let r: u64x2 = transmute(vqaddq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_s8() {
        let a: i8x8 = i8x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i8x8 = i8x8::new(43, 44, 45, 46, 47, 48, 49, 50);
        let r: i8x8 = transmute(vqadd_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_s8() {
        let a: i8x16 = i8x16::new(42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42);
        let b: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: i8x16 = i8x16::new(43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58);
        let r: i8x16 = transmute(vqaddq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_s16() {
        let a: i16x4 = i16x4::new(42, 42, 42, 42);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let e: i16x4 = i16x4::new(43, 44, 45, 46);
        let r: i16x4 = transmute(vqadd_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_s16() {
        let a: i16x8 = i16x8::new(42, 42, 42, 42, 42, 42, 42, 42);
        let b: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i16x8 = i16x8::new(43, 44, 45, 46, 47, 48, 49, 50);
        let r: i16x8 = transmute(vqaddq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_s32() {
        let a: i32x2 = i32x2::new(42, 42);
        let b: i32x2 = i32x2::new(1, 2);
        let e: i32x2 = i32x2::new(43, 44);
        let r: i32x2 = transmute(vqadd_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_s32() {
        let a: i32x4 = i32x4::new(42, 42, 42, 42);
        let b: i32x4 = i32x4::new(1, 2, 3, 4);
        let e: i32x4 = i32x4::new(43, 44, 45, 46);
        let r: i32x4 = transmute(vqaddq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqadd_s64() {
        let a: i64x1 = i64x1::new(42);
        let b: i64x1 = i64x1::new(1);
        let e: i64x1 = i64x1::new(43);
        let r: i64x1 = transmute(vqadd_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqaddq_s64() {
        let a: i64x2 = i64x2::new(42, 42);
        let b: i64x2 = i64x2::new(1, 2);
        let e: i64x2 = i64x2::new(43, 44);
        let r: i64x2 = transmute(vqaddq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_s8() {
        let a: i8x8 = i8x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let b: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i8x8 = i8x8::new(1, 4, 3, 8, 5, 12, 7, 16);
        let r: i8x8 = transmute(vmul_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_s8() {
        let a: i8x16 = i8x16::new(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2);
        let b: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: i8x16 = i8x16::new(1, 4, 3, 8, 5, 12, 7, 16, 9, 20, 11, 24, 13, 28, 15, 32);
        let r: i8x16 = transmute(vmulq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_s16() {
        let a: i16x4 = i16x4::new(1, 2, 1, 2);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let e: i16x4 = i16x4::new(1, 4, 3, 8);
        let r: i16x4 = transmute(vmul_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let b: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i16x8 = i16x8::new(1, 4, 3, 8, 5, 12, 7, 16);
        let r: i16x8 = transmute(vmulq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(1, 2);
        let e: i32x2 = i32x2::new(1, 4);
        let r: i32x2 = transmute(vmul_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 1, 2);
        let b: i32x4 = i32x4::new(1, 2, 3, 4);
        let e: i32x4 = i32x4::new(1, 4, 3, 8);
        let r: i32x4 = transmute(vmulq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_u8() {
        let a: u8x8 = u8x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let b: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(1, 4, 3, 8, 5, 12, 7, 16);
        let r: u8x8 = transmute(vmul_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_u8() {
        let a: u8x16 = u8x16::new(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2);
        let b: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(1, 4, 3, 8, 5, 12, 7, 16, 9, 20, 11, 24, 13, 28, 15, 32);
        let r: u8x16 = transmute(vmulq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_u16() {
        let a: u16x4 = u16x4::new(1, 2, 1, 2);
        let b: u16x4 = u16x4::new(1, 2, 3, 4);
        let e: u16x4 = u16x4::new(1, 4, 3, 8);
        let r: u16x4 = transmute(vmul_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let b: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(1, 4, 3, 8, 5, 12, 7, 16);
        let r: u16x8 = transmute(vmulq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(1, 2);
        let e: u32x2 = u32x2::new(1, 4);
        let r: u32x2 = transmute(vmul_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 1, 2);
        let b: u32x4 = u32x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(1, 4, 3, 8);
        let r: u32x4 = transmute(vmulq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_p8() {
        let a: i8x8 = i8x8::new(1, 3, 1, 3, 1, 3, 1, 3);
        let b: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i8x8 = i8x8::new(1, 6, 3, 12, 5, 10, 7, 24);
        let r: i8x8 = transmute(vmul_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_p8() {
        let a: i8x16 = i8x16::new(1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3);
        let b: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: i8x16 = i8x16::new(1, 6, 3, 12, 5, 10, 7, 24, 9, 30, 11, 20, 13, 18, 15, 48);
        let r: i8x16 = transmute(vmulq_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_f32() {
        let a: f32x2 = f32x2::new(1.0, 2.0);
        let b: f32x2 = f32x2::new(2.0, 3.0);
        let e: f32x2 = f32x2::new(2.0, 6.0);
        let r: f32x2 = transmute(vmul_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_f32() {
        let a: f32x4 = f32x4::new(1.0, 2.0, 1.0, 2.0);
        let b: f32x4 = f32x4::new(2.0, 3.0, 4.0, 5.0);
        let e: f32x4 = f32x4::new(2.0, 6.0, 4.0, 10.0);
        let r: f32x4 = transmute(vmulq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_n_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16 = 2;
        let e: i16x4 = i16x4::new(2, 4, 6, 8);
        let r: i16x4 = transmute(vmul_n_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_n_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16 = 2;
        let e: i16x8 = i16x8::new(2, 4, 6, 8, 10, 12, 14, 16);
        let r: i16x8 = transmute(vmulq_n_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_n_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32 = 2;
        let e: i32x2 = i32x2::new(2, 4);
        let r: i32x2 = transmute(vmul_n_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_n_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32 = 2;
        let e: i32x4 = i32x4::new(2, 4, 6, 8);
        let r: i32x4 = transmute(vmulq_n_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_n_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16 = 2;
        let e: u16x4 = u16x4::new(2, 4, 6, 8);
        let r: u16x4 = transmute(vmul_n_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_n_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u16 = 2;
        let e: u16x8 = u16x8::new(2, 4, 6, 8, 10, 12, 14, 16);
        let r: u16x8 = transmute(vmulq_n_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_n_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32 = 2;
        let e: u32x2 = u32x2::new(2, 4);
        let r: u32x2 = transmute(vmul_n_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_n_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32 = 2;
        let e: u32x4 = u32x4::new(2, 4, 6, 8);
        let r: u32x4 = transmute(vmulq_n_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_n_f32() {
        let a: f32x2 = f32x2::new(1., 2.);
        let b: f32 = 2.;
        let e: f32x2 = f32x2::new(2., 4.);
        let r: f32x2 = transmute(vmul_n_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_n_f32() {
        let a: f32x4 = f32x4::new(1., 2., 3., 4.);
        let b: f32 = 2.;
        let e: f32x4 = f32x4::new(2., 4., 6., 8.);
        let r: f32x4 = transmute(vmulq_n_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_lane_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(0, 2, 0, 0);
        let e: i16x4 = i16x4::new(2, 4, 6, 8);
        let r: i16x4 = transmute(vmul_lane_s16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_laneq_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x8 = i16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: i16x4 = i16x4::new(2, 4, 6, 8);
        let r: i16x4 = transmute(vmul_laneq_s16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_lane_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x4 = i16x4::new(0, 2, 0, 0);
        let e: i16x8 = i16x8::new(2, 4, 6, 8, 10, 12, 14, 16);
        let r: i16x8 = transmute(vmulq_lane_s16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_laneq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: i16x8 = i16x8::new(2, 4, 6, 8, 10, 12, 14, 16);
        let r: i16x8 = transmute(vmulq_laneq_s16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_lane_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(0, 2);
        let e: i32x2 = i32x2::new(2, 4);
        let r: i32x2 = transmute(vmul_lane_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_laneq_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x4 = i32x4::new(0, 2, 0, 0);
        let e: i32x2 = i32x2::new(2, 4);
        let r: i32x2 = transmute(vmul_laneq_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_lane_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x2 = i32x2::new(0, 2);
        let e: i32x4 = i32x4::new(2, 4, 6, 8);
        let r: i32x4 = transmute(vmulq_lane_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_laneq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(0, 2, 0, 0);
        let e: i32x4 = i32x4::new(2, 4, 6, 8);
        let r: i32x4 = transmute(vmulq_laneq_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_lane_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(0, 2, 0, 0);
        let e: u16x4 = u16x4::new(2, 4, 6, 8);
        let r: u16x4 = transmute(vmul_lane_u16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_laneq_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x8 = u16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: u16x4 = u16x4::new(2, 4, 6, 8);
        let r: u16x4 = transmute(vmul_laneq_u16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_lane_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u16x4 = u16x4::new(0, 2, 0, 0);
        let e: u16x8 = u16x8::new(2, 4, 6, 8, 10, 12, 14, 16);
        let r: u16x8 = transmute(vmulq_lane_u16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_laneq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u16x8 = u16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: u16x8 = u16x8::new(2, 4, 6, 8, 10, 12, 14, 16);
        let r: u16x8 = transmute(vmulq_laneq_u16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_lane_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(0, 2);
        let e: u32x2 = u32x2::new(2, 4);
        let r: u32x2 = transmute(vmul_lane_u32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_laneq_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x4 = u32x4::new(0, 2, 0, 0);
        let e: u32x2 = u32x2::new(2, 4);
        let r: u32x2 = transmute(vmul_laneq_u32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_lane_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32x2 = u32x2::new(0, 2);
        let e: u32x4 = u32x4::new(2, 4, 6, 8);
        let r: u32x4 = transmute(vmulq_lane_u32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_laneq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32x4 = u32x4::new(0, 2, 0, 0);
        let e: u32x4 = u32x4::new(2, 4, 6, 8);
        let r: u32x4 = transmute(vmulq_laneq_u32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_lane_f32() {
        let a: f32x2 = f32x2::new(1., 2.);
        let b: f32x2 = f32x2::new(2., 0.);
        let e: f32x2 = f32x2::new(2., 4.);
        let r: f32x2 = transmute(vmul_lane_f32::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmul_laneq_f32() {
        let a: f32x2 = f32x2::new(1., 2.);
        let b: f32x4 = f32x4::new(2., 0., 0., 0.);
        let e: f32x2 = f32x2::new(2., 4.);
        let r: f32x2 = transmute(vmul_laneq_f32::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_lane_f32() {
        let a: f32x4 = f32x4::new(1., 2., 3., 4.);
        let b: f32x2 = f32x2::new(2., 0.);
        let e: f32x4 = f32x4::new(2., 4., 6., 8.);
        let r: f32x4 = transmute(vmulq_lane_f32::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulq_laneq_f32() {
        let a: f32x4 = f32x4::new(1., 2., 3., 4.);
        let b: f32x4 = f32x4::new(2., 0., 0., 0.);
        let e: f32x4 = f32x4::new(2., 4., 6., 8.);
        let r: f32x4 = transmute(vmulq_laneq_f32::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: i16x8 = i16x8::new(1, 4, 3, 8, 5, 12, 7, 16);
        let r: i16x8 = transmute(vmull_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(1, 2, 1, 2);
        let e: i32x4 = i32x4::new(1, 4, 3, 8);
        let r: i32x4 = transmute(vmull_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(1, 2);
        let e: i64x2 = i64x2::new(1, 4);
        let r: i64x2 = transmute(vmull_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u8x8 = u8x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: u16x8 = u16x8::new(1, 4, 3, 8, 5, 12, 7, 16);
        let r: u16x8 = transmute(vmull_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(1, 2, 1, 2);
        let e: u32x4 = u32x4::new(1, 4, 3, 8);
        let r: u32x4 = transmute(vmull_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(1, 2);
        let e: u64x2 = u64x2::new(1, 4);
        let r: u64x2 = transmute(vmull_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_p8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(1, 3, 1, 3, 1, 3, 1, 3);
        let e: i16x8 = i16x8::new(1, 6, 3, 12, 5, 10, 7, 24);
        let r: i16x8 = transmute(vmull_p8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmullh_n_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16 = 2;
        let e: i32x4 = i32x4::new(2, 4, 6, 8);
        let r: i32x4 = transmute(vmullh_n_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulls_n_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32 = 2;
        let e: i64x2 = i64x2::new(2, 4);
        let r: i64x2 = transmute(vmulls_n_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmullh_n_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16 = 2;
        let e: u32x4 = u32x4::new(2, 4, 6, 8);
        let r: u32x4 = transmute(vmullh_n_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmulls_n_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32 = 2;
        let e: u64x2 = u64x2::new(2, 4);
        let r: u64x2 = transmute(vmulls_n_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_lane_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(0, 2, 0, 0);
        let e: i32x4 = i32x4::new(2, 4, 6, 8);
        let r: i32x4 = transmute(vmull_lane_s16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_laneq_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x8 = i16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: i32x4 = i32x4::new(2, 4, 6, 8);
        let r: i32x4 = transmute(vmull_laneq_s16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_lane_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(0, 2);
        let e: i64x2 = i64x2::new(2, 4);
        let r: i64x2 = transmute(vmull_lane_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_laneq_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x4 = i32x4::new(0, 2, 0, 0);
        let e: i64x2 = i64x2::new(2, 4);
        let r: i64x2 = transmute(vmull_laneq_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_lane_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(0, 2, 0, 0);
        let e: u32x4 = u32x4::new(2, 4, 6, 8);
        let r: u32x4 = transmute(vmull_lane_u16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_laneq_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x8 = u16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: u32x4 = u32x4::new(2, 4, 6, 8);
        let r: u32x4 = transmute(vmull_laneq_u16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_lane_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(0, 2);
        let e: u64x2 = u64x2::new(2, 4);
        let r: u64x2 = transmute(vmull_lane_u32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmull_laneq_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x4 = u32x4::new(0, 2, 0, 0);
        let e: u64x2 = u64x2::new(2, 4);
        let r: u64x2 = transmute(vmull_laneq_u32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfma_f32() {
        let a: f32x2 = f32x2::new(8.0, 18.0);
        let b: f32x2 = f32x2::new(6.0, 4.0);
        let c: f32x2 = f32x2::new(2.0, 3.0);
        let e: f32x2 = f32x2::new(20.0, 30.0);
        let r: f32x2 = transmute(vfma_f32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmaq_f32() {
        let a: f32x4 = f32x4::new(8.0, 18.0, 12.0, 10.0);
        let b: f32x4 = f32x4::new(6.0, 4.0, 7.0, 8.0);
        let c: f32x4 = f32x4::new(2.0, 3.0, 4.0, 5.0);
        let e: f32x4 = f32x4::new(20.0, 30.0, 40.0, 50.0);
        let r: f32x4 = transmute(vfmaq_f32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfma_n_f32() {
        let a: f32x2 = f32x2::new(2.0, 3.0);
        let b: f32x2 = f32x2::new(6.0, 4.0);
        let c: f32 = 8.0;
        let e: f32x2 = f32x2::new(50.0, 35.0);
        let r: f32x2 = transmute(vfma_n_f32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmaq_n_f32() {
        let a: f32x4 = f32x4::new(2.0, 3.0, 4.0, 5.0);
        let b: f32x4 = f32x4::new(6.0, 4.0, 7.0, 8.0);
        let c: f32 = 8.0;
        let e: f32x4 = f32x4::new(50.0, 35.0, 60.0, 69.0);
        let r: f32x4 = transmute(vfmaq_n_f32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfms_f32() {
        let a: f32x2 = f32x2::new(20.0, 30.0);
        let b: f32x2 = f32x2::new(6.0, 4.0);
        let c: f32x2 = f32x2::new(2.0, 3.0);
        let e: f32x2 = f32x2::new(8.0, 18.0);
        let r: f32x2 = transmute(vfms_f32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmsq_f32() {
        let a: f32x4 = f32x4::new(20.0, 30.0, 40.0, 50.0);
        let b: f32x4 = f32x4::new(6.0, 4.0, 7.0, 8.0);
        let c: f32x4 = f32x4::new(2.0, 3.0, 4.0, 5.0);
        let e: f32x4 = f32x4::new(8.0, 18.0, 12.0, 10.0);
        let r: f32x4 = transmute(vfmsq_f32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfms_n_f32() {
        let a: f32x2 = f32x2::new(50.0, 35.0);
        let b: f32x2 = f32x2::new(6.0, 4.0);
        let c: f32 = 8.0;
        let e: f32x2 = f32x2::new(2.0, 3.0);
        let r: f32x2 = transmute(vfms_n_f32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vfmsq_n_f32() {
        let a: f32x4 = f32x4::new(50.0, 35.0, 60.0, 69.0);
        let b: f32x4 = f32x4::new(6.0, 4.0, 7.0, 8.0);
        let c: f32 = 8.0;
        let e: f32x4 = f32x4::new(2.0, 3.0, 4.0, 5.0);
        let r: f32x4 = transmute(vfmsq_n_f32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: i8x8 = i8x8::new(0, 0, 2, 2, 4, 4, 6, 6);
        let r: i8x8 = transmute(vsub_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2);
        let e: i8x16 = i8x16::new(0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14);
        let r: i8x16 = transmute(vsubq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(1, 2, 1, 2);
        let e: i16x4 = i16x4::new(0, 0, 2, 2);
        let r: i16x4 = transmute(vsub_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: i16x8 = i16x8::new(0, 0, 2, 2, 4, 4, 6, 6);
        let r: i16x8 = transmute(vsubq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(1, 2);
        let e: i32x2 = i32x2::new(0, 0);
        let r: i32x2 = transmute(vsub_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(1, 2, 1, 2);
        let e: i32x4 = i32x4::new(0, 0, 2, 2);
        let r: i32x4 = transmute(vsubq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u8x8 = u8x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: u8x8 = u8x8::new(0, 0, 2, 2, 4, 4, 6, 6);
        let r: u8x8 = transmute(vsub_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: u8x16 = u8x16::new(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2);
        let e: u8x16 = u8x16::new(0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10, 12, 12, 14, 14);
        let r: u8x16 = transmute(vsubq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(1, 2, 1, 2);
        let e: u16x4 = u16x4::new(0, 0, 2, 2);
        let r: u16x4 = transmute(vsub_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u16x8 = u16x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: u16x8 = u16x8::new(0, 0, 2, 2, 4, 4, 6, 6);
        let r: u16x8 = transmute(vsubq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(1, 2);
        let e: u32x2 = u32x2::new(0, 0);
        let r: u32x2 = transmute(vsub_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32x4 = u32x4::new(1, 2, 1, 2);
        let e: u32x4 = u32x4::new(0, 0, 2, 2);
        let r: u32x4 = transmute(vsubq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_s64() {
        let a: i64x1 = i64x1::new(1);
        let b: i64x1 = i64x1::new(1);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vsub_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_s64() {
        let a: i64x2 = i64x2::new(1, 2);
        let b: i64x2 = i64x2::new(1, 2);
        let e: i64x2 = i64x2::new(0, 0);
        let r: i64x2 = transmute(vsubq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_u64() {
        let a: u64x1 = u64x1::new(1);
        let b: u64x1 = u64x1::new(1);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vsub_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_u64() {
        let a: u64x2 = u64x2::new(1, 2);
        let b: u64x2 = u64x2::new(1, 2);
        let e: u64x2 = u64x2::new(0, 0);
        let r: u64x2 = transmute(vsubq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsub_f32() {
        let a: f32x2 = f32x2::new(1.0, 4.0);
        let b: f32x2 = f32x2::new(1.0, 2.0);
        let e: f32x2 = f32x2::new(0.0, 2.0);
        let r: f32x2 = transmute(vsub_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubq_f32() {
        let a: f32x4 = f32x4::new(1.0, 4.0, 3.0, 8.0);
        let b: f32x4 = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let e: f32x4 = f32x4::new(0.0, 2.0, 0.0, 4.0);
        let r: f32x4 = transmute(vsubq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubhn_s16() {
        let a: i16x8 = i16x8::new(0x7F_FF, -32768, 1, 1, 0x7F_FF, -32768, 1, 1);
        let b: i16x8 = i16x8::new(1, 0, 0, 0, 1, 0, 0, 0);
        let e: i8x8 = i8x8::new(0x7F, -128, 0, 0, 0x7F, -128, 0, 0);
        let r: i8x8 = transmute(vsubhn_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubhn_s32() {
        let a: i32x4 = i32x4::new(0x7F_FF_FF_FF, -2147483648, 1, 1);
        let b: i32x4 = i32x4::new(1, 0, 0, 0);
        let e: i16x4 = i16x4::new(0x7F_FF, -32768, 0, 0);
        let r: i16x4 = transmute(vsubhn_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubhn_s64() {
        let a: i64x2 = i64x2::new(0x7F_FF_FF_FF_FF_FF_FF_FF, -9223372036854775808);
        let b: i64x2 = i64x2::new(1, 0);
        let e: i32x2 = i32x2::new(0x7F_FF_FF_FF, -2147483648);
        let r: i32x2 = transmute(vsubhn_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubhn_u16() {
        let a: u16x8 = u16x8::new(0xFF_FF, 0, 1, 1, 0xFF_FF, 0, 1, 1);
        let b: u16x8 = u16x8::new(1, 0, 0, 0, 1, 0, 0, 0);
        let e: u8x8 = u8x8::new(0xFF, 0, 0, 0, 0xFF, 0, 0, 0);
        let r: u8x8 = transmute(vsubhn_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubhn_u32() {
        let a: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0, 1, 1);
        let b: u32x4 = u32x4::new(1, 0, 0, 0);
        let e: u16x4 = u16x4::new(0xFF_FF, 0, 0, 0);
        let r: u16x4 = transmute(vsubhn_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubhn_u64() {
        let a: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0);
        let b: u64x2 = u64x2::new(1, 0);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0);
        let r: u32x2 = transmute(vsubhn_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubhn_high_s16() {
        let a: i8x8 = i8x8::new(0x7F, 0, 0x7F, 0, 0x7F, 0, 0x7F, 0);
        let b: i16x8 = i16x8::new(0x7F_FF, 1, 0x7F_FF, 1, 0x7F_FF, 1, 0x7F_FF, 1);
        let c: i16x8 = i16x8::new(1, 0, 1, 0, 1, 0, 1, 0);
        let e: i8x16 = i8x16::new(0x7F, 0, 0x7F, 0, 0x7F, 0, 0x7F, 0, 0x7F, 0, 0x7F, 0, 0x7F, 0, 0x7F, 0);
        let r: i8x16 = transmute(vsubhn_high_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubhn_high_s32() {
        let a: i16x4 = i16x4::new(0x7F_FF, 0, 0x7F_FF, 0);
        let b: i32x4 = i32x4::new(0x7F_FF_FF_FF, 1, 0x7F_FF_FF_FF, 1);
        let c: i32x4 = i32x4::new(1, 0, 1, 0);
        let e: i16x8 = i16x8::new(0x7F_FF, 0, 0x7F_FF, 0, 0x7F_FF, 0, 0x7F_FF, 0);
        let r: i16x8 = transmute(vsubhn_high_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubhn_high_s64() {
        let a: i32x2 = i32x2::new(0x7F_FF_FF_FF, 0);
        let b: i64x2 = i64x2::new(0x7F_FF_FF_FF_FF_FF_FF_FF, 1);
        let c: i64x2 = i64x2::new(1, 0);
        let e: i32x4 = i32x4::new(0x7F_FF_FF_FF, 0, 0x7F_FF_FF_FF, 0);
        let r: i32x4 = transmute(vsubhn_high_s64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubhn_high_u16() {
        let a: u8x8 = u8x8::new(0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0);
        let b: u16x8 = u16x8::new(0xFF_FF, 1, 0xFF_FF, 1, 0xFF_FF, 1, 0xFF_FF, 1);
        let c: u16x8 = u16x8::new(1, 0, 1, 0, 1, 0, 1, 0);
        let e: u8x16 = u8x16::new(0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0, 0xFF, 0);
        let r: u8x16 = transmute(vsubhn_high_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubhn_high_u32() {
        let a: u16x4 = u16x4::new(0xFF_FF, 0, 0xFF_FF, 0);
        let b: u32x4 = u32x4::new(0xFF_FF_FF_FF, 1, 0xFF_FF_FF_FF, 1);
        let c: u32x4 = u32x4::new(1, 0, 1, 0);
        let e: u16x8 = u16x8::new(0xFF_FF, 0, 0xFF_FF, 0, 0xFF_FF, 0, 0xFF_FF, 0);
        let r: u16x8 = transmute(vsubhn_high_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubhn_high_u64() {
        let a: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0);
        let b: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 1);
        let c: u64x2 = u64x2::new(1, 0);
        let e: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0, 0xFF_FF_FF_FF, 0);
        let r: u32x4 = transmute(vsubhn_high_u64(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u8x8 = u8x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: u8x8 = u8x8::new(0, 0, 1, 1, 2, 2, 3, 3);
        let r: u8x8 = transmute(vhsub_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: u8x16 = u8x16::new(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2);
        let e: u8x16 = u8x16::new(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
        let r: u8x16 = transmute(vhsubq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(1, 2, 1, 2);
        let e: u16x4 = u16x4::new(0, 0, 1, 1);
        let r: u16x4 = transmute(vhsub_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u16x8 = u16x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: u16x8 = u16x8::new(0, 0, 1, 1, 2, 2, 3, 3);
        let r: u16x8 = transmute(vhsubq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(1, 2);
        let e: u32x2 = u32x2::new(0, 0);
        let r: u32x2 = transmute(vhsub_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32x4 = u32x4::new(1, 2, 1, 2);
        let e: u32x4 = u32x4::new(0, 0, 1, 1);
        let r: u32x4 = transmute(vhsubq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: i8x8 = i8x8::new(0, 0, 1, 1, 2, 2, 3, 3);
        let r: i8x8 = transmute(vhsub_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2);
        let e: i8x16 = i8x16::new(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
        let r: i8x16 = transmute(vhsubq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(1, 2, 1, 2);
        let e: i16x4 = i16x4::new(0, 0, 1, 1);
        let r: i16x4 = transmute(vhsub_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(1, 2, 1, 2, 1, 2, 1, 2);
        let e: i16x8 = i16x8::new(0, 0, 1, 1, 2, 2, 3, 3);
        let r: i16x8 = transmute(vhsubq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsub_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(1, 2);
        let e: i32x2 = i32x2::new(0, 0);
        let r: i32x2 = transmute(vhsub_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vhsubq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(1, 2, 1, 2);
        let e: i32x4 = i32x4::new(0, 0, 1, 1);
        let r: i32x4 = transmute(vhsubq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubw_s8() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: i16x8 = transmute(vsubw_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubw_s16() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let b: i16x4 = i16x4::new(0, 1, 2, 3);
        let e: i32x4 = i32x4::new(0, 0, 0, 0);
        let r: i32x4 = transmute(vsubw_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubw_s32() {
        let a: i64x2 = i64x2::new(0, 1);
        let b: i32x2 = i32x2::new(0, 1);
        let e: i64x2 = i64x2::new(0, 0);
        let r: i64x2 = transmute(vsubw_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubw_u8() {
        let a: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u16x8 = u16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: u16x8 = transmute(vsubw_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubw_u16() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let b: u16x4 = u16x4::new(0, 1, 2, 3);
        let e: u32x4 = u32x4::new(0, 0, 0, 0);
        let r: u32x4 = transmute(vsubw_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubw_u32() {
        let a: u64x2 = u64x2::new(0, 1);
        let b: u32x2 = u32x2::new(0, 1);
        let e: u64x2 = u64x2::new(0, 0);
        let r: u64x2 = transmute(vsubw_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubl_s8() {
        let a: i8x8 = i8x8::new(0x7F, -128, 2, 3, 4, 5, 6, 7);
        let b: i8x8 = i8x8::new(0x7F, -128, 2, 3, 4, 5, 6, 7);
        let e: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: i16x8 = transmute(vsubl_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubl_s16() {
        let a: i16x4 = i16x4::new(0x7F_FF, -32768, 2, 3);
        let b: i16x4 = i16x4::new(0x7F_FF, -32768, 2, 3);
        let e: i32x4 = i32x4::new(0, 0, 0, 0);
        let r: i32x4 = transmute(vsubl_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubl_s32() {
        let a: i32x2 = i32x2::new(0x7F_FF_FF_FF, -2147483648);
        let b: i32x2 = i32x2::new(0x7F_FF_FF_FF, -2147483648);
        let e: i64x2 = i64x2::new(0, 0);
        let r: i64x2 = transmute(vsubl_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubl_u8() {
        let a: u8x8 = u8x8::new(0xFF, 0, 2, 3, 4, 5, 6, 7);
        let b: u8x8 = u8x8::new(0xFF, 0, 2, 3, 4, 5, 6, 7);
        let e: u16x8 = u16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: u16x8 = transmute(vsubl_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubl_u16() {
        let a: u16x4 = u16x4::new(0xFF_FF, 0, 2, 3);
        let b: u16x4 = u16x4::new(0xFF_FF, 0, 2, 3);
        let e: u32x4 = u32x4::new(0, 0, 0, 0);
        let r: u32x4 = transmute(vsubl_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsubl_u32() {
        let a: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0);
        let b: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0);
        let e: u64x2 = u64x2::new(0, 0);
        let r: u64x2 = transmute(vsubl_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmax_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: i8x8 = i8x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let r: i8x8 = transmute(vmax_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxq_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        let e: i8x16 = i8x16::new(16, 15, 14, 13, 12, 11, 10, 9, 9, 10, 11, 12, 13, 14, 15, 16);
        let r: i8x16 = transmute(vmaxq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmax_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(16, 15, 14, 13);
        let e: i16x4 = i16x4::new(16, 15, 14, 13);
        let r: i16x4 = transmute(vmax_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: i16x8 = i16x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let r: i16x8 = transmute(vmaxq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmax_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(16, 15);
        let e: i32x2 = i32x2::new(16, 15);
        let r: i32x2 = transmute(vmax_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(16, 15, 14, 13);
        let e: i32x4 = i32x4::new(16, 15, 14, 13);
        let r: i32x4 = transmute(vmaxq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmax_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u8x8 = u8x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: u8x8 = u8x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let r: u8x8 = transmute(vmax_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxq_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: u8x16 = u8x16::new(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        let e: u8x16 = u8x16::new(16, 15, 14, 13, 12, 11, 10, 9, 9, 10, 11, 12, 13, 14, 15, 16);
        let r: u8x16 = transmute(vmaxq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmax_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(16, 15, 14, 13);
        let e: u16x4 = u16x4::new(16, 15, 14, 13);
        let r: u16x4 = transmute(vmax_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u16x8 = u16x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: u16x8 = u16x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let r: u16x8 = transmute(vmaxq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmax_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(16, 15);
        let e: u32x2 = u32x2::new(16, 15);
        let r: u32x2 = transmute(vmax_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32x4 = u32x4::new(16, 15, 14, 13);
        let e: u32x4 = u32x4::new(16, 15, 14, 13);
        let r: u32x4 = transmute(vmaxq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmax_f32() {
        let a: f32x2 = f32x2::new(1.0, -2.0);
        let b: f32x2 = f32x2::new(0.0, 3.0);
        let e: f32x2 = f32x2::new(1.0, 3.0);
        let r: f32x2 = transmute(vmax_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxq_f32() {
        let a: f32x4 = f32x4::new(1.0, -2.0, 3.0, -4.0);
        let b: f32x4 = f32x4::new(0.0, 3.0, 2.0, 8.0);
        let e: f32x4 = f32x4::new(1.0, 3.0, 3.0, 8.0);
        let r: f32x4 = transmute(vmaxq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxnm_f32() {
        let a: f32x2 = f32x2::new(1.0, 2.0);
        let b: f32x2 = f32x2::new(8.0, 16.0);
        let e: f32x2 = f32x2::new(8.0, 16.0);
        let r: f32x2 = transmute(vmaxnm_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmaxnmq_f32() {
        let a: f32x4 = f32x4::new(1.0, 2.0, 3.0, -4.0);
        let b: f32x4 = f32x4::new(8.0, 16.0, -1.0, 6.0);
        let e: f32x4 = f32x4::new(8.0, 16.0, 3.0, 6.0);
        let r: f32x4 = transmute(vmaxnmq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmin_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i8x8 = transmute(vmin_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminq_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        let e: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1);
        let r: i8x16 = transmute(vminq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmin_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(16, 15, 14, 13);
        let e: i16x4 = i16x4::new(1, 2, 3, 4);
        let r: i16x4 = transmute(vmin_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i16x8 = transmute(vminq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmin_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(16, 15);
        let e: i32x2 = i32x2::new(1, 2);
        let r: i32x2 = transmute(vmin_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(16, 15, 14, 13);
        let e: i32x4 = i32x4::new(1, 2, 3, 4);
        let r: i32x4 = transmute(vminq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmin_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u8x8 = u8x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u8x8 = transmute(vmin_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminq_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: u8x16 = u8x16::new(16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        let e: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 5, 4, 3, 2, 1);
        let r: u8x16 = transmute(vminq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmin_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(16, 15, 14, 13);
        let e: u16x4 = u16x4::new(1, 2, 3, 4);
        let r: u16x4 = transmute(vmin_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u16x8 = u16x8::new(16, 15, 14, 13, 12, 11, 10, 9);
        let e: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u16x8 = transmute(vminq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmin_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: u32x2 = u32x2::new(16, 15);
        let e: u32x2 = u32x2::new(1, 2);
        let r: u32x2 = transmute(vmin_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u32x4 = u32x4::new(16, 15, 14, 13);
        let e: u32x4 = u32x4::new(1, 2, 3, 4);
        let r: u32x4 = transmute(vminq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vmin_f32() {
        let a: f32x2 = f32x2::new(1.0, -2.0);
        let b: f32x2 = f32x2::new(0.0, 3.0);
        let e: f32x2 = f32x2::new(0.0, -2.0);
        let r: f32x2 = transmute(vmin_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminq_f32() {
        let a: f32x4 = f32x4::new(1.0, -2.0, 3.0, -4.0);
        let b: f32x4 = f32x4::new(0.0, 3.0, 2.0, 8.0);
        let e: f32x4 = f32x4::new(0.0, -2.0, 2.0, -4.0);
        let r: f32x4 = transmute(vminq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminnm_f32() {
        let a: f32x2 = f32x2::new(1.0, 2.0);
        let b: f32x2 = f32x2::new(8.0, 16.0);
        let e: f32x2 = f32x2::new(1.0, 2.0);
        let r: f32x2 = transmute(vminnm_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vminnmq_f32() {
        let a: f32x4 = f32x4::new(1.0, 2.0, 3.0, -4.0);
        let b: f32x4 = f32x4::new(8.0, 16.0, -1.0, 6.0);
        let e: f32x4 = f32x4::new(1.0, 2.0, -1.0, -4.0);
        let r: f32x4 = transmute(vminnmq_f32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmull_s16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let e: i32x4 = i32x4::new(0, 4, 12, 24);
        let r: i32x4 = transmute(vqdmull_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmull_s32() {
        let a: i32x2 = i32x2::new(0, 1);
        let b: i32x2 = i32x2::new(1, 2);
        let e: i64x2 = i64x2::new(0, 4);
        let r: i64x2 = transmute(vqdmull_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmull_n_s16() {
        let a: i16x4 = i16x4::new(2, 4, 6, 8);
        let b: i16 = 2;
        let e: i32x4 = i32x4::new(8, 16, 24, 32);
        let r: i32x4 = transmute(vqdmull_n_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmull_n_s32() {
        let a: i32x2 = i32x2::new(2, 4);
        let b: i32 = 2;
        let e: i64x2 = i64x2::new(8, 16);
        let r: i64x2 = transmute(vqdmull_n_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmull_lane_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(0, 2, 2, 0);
        let e: i32x4 = i32x4::new(4, 8, 12, 16);
        let r: i32x4 = transmute(vqdmull_lane_s16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmull_lane_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(0, 2);
        let e: i64x2 = i64x2::new(4, 8);
        let r: i64x2 = transmute(vqdmull_lane_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlal_s16() {
        let a: i32x4 = i32x4::new(1, 1, 1, 1);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let c: i16x4 = i16x4::new(2, 2, 2, 2);
        let e: i32x4 = i32x4::new(5, 9, 13, 17);
        let r: i32x4 = transmute(vqdmlal_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlal_s32() {
        let a: i64x2 = i64x2::new(1, 1);
        let b: i32x2 = i32x2::new(1, 2);
        let c: i32x2 = i32x2::new(2, 2);
        let e: i64x2 = i64x2::new(5, 9);
        let r: i64x2 = transmute(vqdmlal_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlal_n_s16() {
        let a: i32x4 = i32x4::new(1, 1, 1, 1);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let c: i16 = 2;
        let e: i32x4 = i32x4::new(5, 9, 13, 17);
        let r: i32x4 = transmute(vqdmlal_n_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlal_n_s32() {
        let a: i64x2 = i64x2::new(1, 1);
        let b: i32x2 = i32x2::new(1, 2);
        let c: i32 = 2;
        let e: i64x2 = i64x2::new(5, 9);
        let r: i64x2 = transmute(vqdmlal_n_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlal_lane_s16() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let c: i16x4 = i16x4::new(0, 2, 2, 0);
        let e: i32x4 = i32x4::new(5, 10, 15, 20);
        let r: i32x4 = transmute(vqdmlal_lane_s16::<2>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlal_lane_s32() {
        let a: i64x2 = i64x2::new(1, 2);
        let b: i32x2 = i32x2::new(1, 2);
        let c: i32x2 = i32x2::new(0, 2);
        let e: i64x2 = i64x2::new(5, 10);
        let r: i64x2 = transmute(vqdmlal_lane_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlsl_s16() {
        let a: i32x4 = i32x4::new(3, 7, 11, 15);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let c: i16x4 = i16x4::new(2, 2, 2, 2);
        let e: i32x4 = i32x4::new(-1, -1, -1, -1);
        let r: i32x4 = transmute(vqdmlsl_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlsl_s32() {
        let a: i64x2 = i64x2::new(3, 7);
        let b: i32x2 = i32x2::new(1, 2);
        let c: i32x2 = i32x2::new(2, 2);
        let e: i64x2 = i64x2::new(-1, -1);
        let r: i64x2 = transmute(vqdmlsl_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlsl_n_s16() {
        let a: i32x4 = i32x4::new(3, 7, 11, 15);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let c: i16 = 2;
        let e: i32x4 = i32x4::new(-1, -1, -1, -1);
        let r: i32x4 = transmute(vqdmlsl_n_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlsl_n_s32() {
        let a: i64x2 = i64x2::new(3, 7);
        let b: i32x2 = i32x2::new(1, 2);
        let c: i32 = 2;
        let e: i64x2 = i64x2::new(-1, -1);
        let r: i64x2 = transmute(vqdmlsl_n_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlsl_lane_s16() {
        let a: i32x4 = i32x4::new(3, 6, 9, 12);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let c: i16x4 = i16x4::new(0, 2, 2, 0);
        let e: i32x4 = i32x4::new(-1, -2, -3, -4);
        let r: i32x4 = transmute(vqdmlsl_lane_s16::<2>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmlsl_lane_s32() {
        let a: i64x2 = i64x2::new(3, 6);
        let b: i32x2 = i32x2::new(1, 2);
        let c: i32x2 = i32x2::new(0, 2);
        let e: i64x2 = i64x2::new(-1, -2);
        let r: i64x2 = transmute(vqdmlsl_lane_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmulh_s16() {
        let a: i16x4 = i16x4::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let e: i16x4 = i16x4::new(1, 1, 1, 1);
        let r: i16x4 = transmute(vqdmulh_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmulhq_s16() {
        let a: i16x8 = i16x8::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: i16x8 = i16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let r: i16x8 = transmute(vqdmulhq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmulh_s32() {
        let a: i32x2 = i32x2::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let b: i32x2 = i32x2::new(2, 2);
        let e: i32x2 = i32x2::new(1, 1);
        let r: i32x2 = transmute(vqdmulh_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmulhq_s32() {
        let a: i32x4 = i32x4::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let e: i32x4 = i32x4::new(1, 1, 1, 1);
        let r: i32x4 = transmute(vqdmulhq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmulh_n_s16() {
        let a: i16x4 = i16x4::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let b: i16 = 2;
        let e: i16x4 = i16x4::new(1, 1, 1, 1);
        let r: i16x4 = transmute(vqdmulh_n_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmulh_n_s32() {
        let a: i32x2 = i32x2::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let b: i32 = 2;
        let e: i32x2 = i32x2::new(1, 1);
        let r: i32x2 = transmute(vqdmulh_n_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmulhq_nq_s16() {
        let a: i16x8 = i16x8::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let b: i16 = 2;
        let e: i16x8 = i16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let r: i16x8 = transmute(vqdmulhq_nq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqdmulhq_nq_s32() {
        let a: i32x4 = i32x4::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let b: i32 = 2;
        let e: i32x4 = i32x4::new(1, 1, 1, 1);
        let r: i32x4 = transmute(vqdmulhq_nq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovn_s16() {
        let a: i16x8 = i16x8::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let e: i8x8 = i8x8::new(0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F, 0x7F);
        let r: i8x8 = transmute(vqmovn_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovn_s32() {
        let a: i32x4 = i32x4::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let e: i16x4 = i16x4::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let r: i16x4 = transmute(vqmovn_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovn_s64() {
        let a: i64x2 = i64x2::new(0x7F_FF_FF_FF_FF_FF_FF_FF, 0x7F_FF_FF_FF_FF_FF_FF_FF);
        let e: i32x2 = i32x2::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let r: i32x2 = transmute(vqmovn_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovn_u16() {
        let a: u16x8 = u16x8::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let e: u8x8 = u8x8::new(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);
        let r: u8x8 = transmute(vqmovn_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovn_u32() {
        let a: u32x4 = u32x4::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let e: u16x4 = u16x4::new(0xFF_FF, 0xFF_FF, 0xFF_FF, 0xFF_FF);
        let r: u16x4 = transmute(vqmovn_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovn_u64() {
        let a: u64x2 = u64x2::new(0xFF_FF_FF_FF_FF_FF_FF_FF, 0xFF_FF_FF_FF_FF_FF_FF_FF);
        let e: u32x2 = u32x2::new(0xFF_FF_FF_FF, 0xFF_FF_FF_FF);
        let r: u32x2 = transmute(vqmovn_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovun_s16() {
        let a: i16x8 = i16x8::new(-1, -1, -1, -1, -1, -1, -1, -1);
        let e: u8x8 = u8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: u8x8 = transmute(vqmovun_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovun_s32() {
        let a: i32x4 = i32x4::new(-1, -1, -1, -1);
        let e: u16x4 = u16x4::new(0, 0, 0, 0);
        let r: u16x4 = transmute(vqmovun_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqmovun_s64() {
        let a: i64x2 = i64x2::new(-1, -1);
        let e: u32x2 = u32x2::new(0, 0);
        let r: u32x2 = transmute(vqmovun_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulh_s16() {
        let a: i16x4 = i16x4::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let e: i16x4 = i16x4::new(2, 2, 2, 2);
        let r: i16x4 = transmute(vqrdmulh_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulhq_s16() {
        let a: i16x8 = i16x8::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let r: i16x8 = transmute(vqrdmulhq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulh_s32() {
        let a: i32x2 = i32x2::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let b: i32x2 = i32x2::new(2, 2);
        let e: i32x2 = i32x2::new(2, 2);
        let r: i32x2 = transmute(vqrdmulh_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulhq_s32() {
        let a: i32x4 = i32x4::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let e: i32x4 = i32x4::new(2, 2, 2, 2);
        let r: i32x4 = transmute(vqrdmulhq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulh_n_s16() {
        let a: i16x4 = i16x4::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let b: i16 = 2;
        let e: i16x4 = i16x4::new(2, 2, 2, 2);
        let r: i16x4 = transmute(vqrdmulh_n_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulhq_n_s16() {
        let a: i16x8 = i16x8::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let b: i16 = 2;
        let e: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let r: i16x8 = transmute(vqrdmulhq_n_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulh_n_s32() {
        let a: i32x2 = i32x2::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let b: i32 = 2;
        let e: i32x2 = i32x2::new(2, 2);
        let r: i32x2 = transmute(vqrdmulh_n_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulhq_n_s32() {
        let a: i32x4 = i32x4::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let b: i32 = 2;
        let e: i32x4 = i32x4::new(2, 2, 2, 2);
        let r: i32x4 = transmute(vqrdmulhq_n_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulh_lane_s16() {
        let a: i16x4 = i16x4::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let b: i16x4 = i16x4::new(0, 2, 0, 0);
        let e: i16x4 = i16x4::new(2, 2, 2, 2);
        let r: i16x4 = transmute(vqrdmulh_lane_s16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulh_laneq_s16() {
        let a: i16x4 = i16x4::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let b: i16x8 = i16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: i16x4 = i16x4::new(2, 2, 2, 2);
        let r: i16x4 = transmute(vqrdmulh_laneq_s16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulhq_lane_s16() {
        let a: i16x8 = i16x8::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let b: i16x4 = i16x4::new(0, 2, 0, 0);
        let e: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let r: i16x8 = transmute(vqrdmulhq_lane_s16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulhq_laneq_s16() {
        let a: i16x8 = i16x8::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let b: i16x8 = i16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let r: i16x8 = transmute(vqrdmulhq_laneq_s16::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulh_lane_s32() {
        let a: i32x2 = i32x2::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let b: i32x2 = i32x2::new(0, 2);
        let e: i32x2 = i32x2::new(2, 2);
        let r: i32x2 = transmute(vqrdmulh_lane_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulh_laneq_s32() {
        let a: i32x2 = i32x2::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let b: i32x4 = i32x4::new(0, 2, 0, 0);
        let e: i32x2 = i32x2::new(2, 2);
        let r: i32x2 = transmute(vqrdmulh_laneq_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulhq_lane_s32() {
        let a: i32x4 = i32x4::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let b: i32x2 = i32x2::new(0, 2);
        let e: i32x4 = i32x4::new(2, 2, 2, 2);
        let r: i32x4 = transmute(vqrdmulhq_lane_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmulhq_laneq_s32() {
        let a: i32x4 = i32x4::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let b: i32x4 = i32x4::new(0, 2, 0, 0);
        let e: i32x4 = i32x4::new(2, 2, 2, 2);
        let r: i32x4 = transmute(vqrdmulhq_laneq_s32::<1>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlah_s16() {
        let a: i16x4 = i16x4::new(1, 1, 1, 1);
        let b: i16x4 = i16x4::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let c: i16x4 = i16x4::new(2, 2, 2, 2);
        let e: i16x4 = i16x4::new(3, 3, 3, 3);
        let r: i16x4 = transmute(vqrdmlah_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlahq_s16() {
        let a: i16x8 = i16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let b: i16x8 = i16x8::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let c: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: i16x8 = i16x8::new(3, 3, 3, 3, 3, 3, 3, 3);
        let r: i16x8 = transmute(vqrdmlahq_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlah_s32() {
        let a: i32x2 = i32x2::new(1, 1);
        let b: i32x2 = i32x2::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let c: i32x2 = i32x2::new(2, 2);
        let e: i32x2 = i32x2::new(3, 3);
        let r: i32x2 = transmute(vqrdmlah_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlahq_s32() {
        let a: i32x4 = i32x4::new(1, 1, 1, 1);
        let b: i32x4 = i32x4::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let c: i32x4 = i32x4::new(2, 2, 2, 2);
        let e: i32x4 = i32x4::new(3, 3, 3, 3);
        let r: i32x4 = transmute(vqrdmlahq_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlah_lane_s16() {
        let a: i16x4 = i16x4::new(1, 1, 1, 1);
        let b: i16x4 = i16x4::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let c: i16x4 = i16x4::new(0, 2, 0, 0);
        let e: i16x4 = i16x4::new(3, 3, 3, 3);
        let r: i16x4 = transmute(vqrdmlah_lane_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlah_laneq_s16() {
        let a: i16x4 = i16x4::new(1, 1, 1, 1);
        let b: i16x4 = i16x4::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let c: i16x8 = i16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: i16x4 = i16x4::new(3, 3, 3, 3);
        let r: i16x4 = transmute(vqrdmlah_laneq_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlahq_lane_s16() {
        let a: i16x8 = i16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let b: i16x8 = i16x8::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let c: i16x4 = i16x4::new(0, 2, 0, 0);
        let e: i16x8 = i16x8::new(3, 3, 3, 3, 3, 3, 3, 3);
        let r: i16x8 = transmute(vqrdmlahq_lane_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlahq_laneq_s16() {
        let a: i16x8 = i16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let b: i16x8 = i16x8::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let c: i16x8 = i16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: i16x8 = i16x8::new(3, 3, 3, 3, 3, 3, 3, 3);
        let r: i16x8 = transmute(vqrdmlahq_laneq_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlah_lane_s32() {
        let a: i32x2 = i32x2::new(1, 1);
        let b: i32x2 = i32x2::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let c: i32x2 = i32x2::new(0, 2);
        let e: i32x2 = i32x2::new(3, 3);
        let r: i32x2 = transmute(vqrdmlah_lane_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlah_laneq_s32() {
        let a: i32x2 = i32x2::new(1, 1);
        let b: i32x2 = i32x2::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let c: i32x4 = i32x4::new(0, 2, 0, 0);
        let e: i32x2 = i32x2::new(3, 3);
        let r: i32x2 = transmute(vqrdmlah_laneq_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlahq_lane_s32() {
        let a: i32x4 = i32x4::new(1, 1, 1, 1);
        let b: i32x4 = i32x4::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let c: i32x2 = i32x2::new(0, 2);
        let e: i32x4 = i32x4::new(3, 3, 3, 3);
        let r: i32x4 = transmute(vqrdmlahq_lane_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlahq_laneq_s32() {
        let a: i32x4 = i32x4::new(1, 1, 1, 1);
        let b: i32x4 = i32x4::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let c: i32x4 = i32x4::new(0, 2, 0, 0);
        let e: i32x4 = i32x4::new(3, 3, 3, 3);
        let r: i32x4 = transmute(vqrdmlahq_laneq_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlsh_s16() {
        let a: i16x4 = i16x4::new(1, 1, 1, 1);
        let b: i16x4 = i16x4::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let c: i16x4 = i16x4::new(2, 2, 2, 2);
        let e: i16x4 = i16x4::new(-1, -1, -1, -1);
        let r: i16x4 = transmute(vqrdmlsh_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlshq_s16() {
        let a: i16x8 = i16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let b: i16x8 = i16x8::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let c: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: i16x8 = i16x8::new(-1, -1, -1, -1, -1, -1, -1, -1);
        let r: i16x8 = transmute(vqrdmlshq_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlsh_s32() {
        let a: i32x2 = i32x2::new(1, 1);
        let b: i32x2 = i32x2::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let c: i32x2 = i32x2::new(2, 2);
        let e: i32x2 = i32x2::new(-1, -1);
        let r: i32x2 = transmute(vqrdmlsh_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlshq_s32() {
        let a: i32x4 = i32x4::new(1, 1, 1, 1);
        let b: i32x4 = i32x4::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let c: i32x4 = i32x4::new(2, 2, 2, 2);
        let e: i32x4 = i32x4::new(-1, -1, -1, -1);
        let r: i32x4 = transmute(vqrdmlshq_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlsh_lane_s16() {
        let a: i16x4 = i16x4::new(1, 1, 1, 1);
        let b: i16x4 = i16x4::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let c: i16x4 = i16x4::new(0, 2, 0, 0);
        let e: i16x4 = i16x4::new(-1, -1, -1, -1);
        let r: i16x4 = transmute(vqrdmlsh_lane_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlsh_laneq_s16() {
        let a: i16x4 = i16x4::new(1, 1, 1, 1);
        let b: i16x4 = i16x4::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let c: i16x8 = i16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: i16x4 = i16x4::new(-1, -1, -1, -1);
        let r: i16x4 = transmute(vqrdmlsh_laneq_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlshq_lane_s16() {
        let a: i16x8 = i16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let b: i16x8 = i16x8::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let c: i16x4 = i16x4::new(0, 2, 0, 0);
        let e: i16x8 = i16x8::new(-1, -1, -1, -1, -1, -1, -1, -1);
        let r: i16x8 = transmute(vqrdmlshq_lane_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlshq_laneq_s16() {
        let a: i16x8 = i16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let b: i16x8 = i16x8::new(0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF, 0x7F_FF);
        let c: i16x8 = i16x8::new(0, 2, 0, 0, 0, 0, 0, 0);
        let e: i16x8 = i16x8::new(-1, -1, -1, -1, -1, -1, -1, -1);
        let r: i16x8 = transmute(vqrdmlshq_laneq_s16::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlsh_lane_s32() {
        let a: i32x2 = i32x2::new(1, 1);
        let b: i32x2 = i32x2::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let c: i32x2 = i32x2::new(0, 2);
        let e: i32x2 = i32x2::new(-1, -1);
        let r: i32x2 = transmute(vqrdmlsh_lane_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlsh_laneq_s32() {
        let a: i32x2 = i32x2::new(1, 1);
        let b: i32x2 = i32x2::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let c: i32x4 = i32x4::new(0, 2, 0, 0);
        let e: i32x2 = i32x2::new(-1, -1);
        let r: i32x2 = transmute(vqrdmlsh_laneq_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlshq_lane_s32() {
        let a: i32x4 = i32x4::new(1, 1, 1, 1);
        let b: i32x4 = i32x4::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let c: i32x2 = i32x2::new(0, 2);
        let e: i32x4 = i32x4::new(-1, -1, -1, -1);
        let r: i32x4 = transmute(vqrdmlshq_lane_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrdmlshq_laneq_s32() {
        let a: i32x4 = i32x4::new(1, 1, 1, 1);
        let b: i32x4 = i32x4::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let c: i32x4 = i32x4::new(0, 2, 0, 0);
        let e: i32x4 = i32x4::new(-1, -1, -1, -1);
        let r: i32x4 = transmute(vqrdmlshq_laneq_s32::<1>(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshl_s8() {
        let a: i8x8 = i8x8::new(2, -128, 0x7F, 3, 4, 5, 6, 7);
        let b: i8x8 = i8x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: i8x8 = i8x8::new(8, -128, 0x7F, 12, 16, 20, 24, 28);
        let r: i8x8 = transmute(vqrshl_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshlq_s8() {
        let a: i8x16 = i8x16::new(2, -128, 0x7F, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b: i8x16 = i8x16::new(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let e: i8x16 = i8x16::new(8, -128, 0x7F, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60);
        let r: i8x16 = transmute(vqrshlq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshl_s16() {
        let a: i16x4 = i16x4::new(2, -32768, 0x7F_FF, 3);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let e: i16x4 = i16x4::new(8, -32768, 0x7F_FF, 12);
        let r: i16x4 = transmute(vqrshl_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshlq_s16() {
        let a: i16x8 = i16x8::new(2, -32768, 0x7F_FF, 3, 4, 5, 6, 7);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: i16x8 = i16x8::new(8, -32768, 0x7F_FF, 12, 16, 20, 24, 28);
        let r: i16x8 = transmute(vqrshlq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshl_s32() {
        let a: i32x2 = i32x2::new(2, -2147483648);
        let b: i32x2 = i32x2::new(2, 2);
        let e: i32x2 = i32x2::new(8, -2147483648);
        let r: i32x2 = transmute(vqrshl_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshlq_s32() {
        let a: i32x4 = i32x4::new(2, -2147483648, 0x7F_FF_FF_FF, 3);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let e: i32x4 = i32x4::new(8, -2147483648, 0x7F_FF_FF_FF, 12);
        let r: i32x4 = transmute(vqrshlq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshl_s64() {
        let a: i64x1 = i64x1::new(2);
        let b: i64x1 = i64x1::new(2);
        let e: i64x1 = i64x1::new(8);
        let r: i64x1 = transmute(vqrshl_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshlq_s64() {
        let a: i64x2 = i64x2::new(2, -9223372036854775808);
        let b: i64x2 = i64x2::new(2, 2);
        let e: i64x2 = i64x2::new(8, -9223372036854775808);
        let r: i64x2 = transmute(vqrshlq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshl_u8() {
        let a: u8x8 = u8x8::new(2, 0, 0xFF, 3, 4, 5, 6, 7);
        let b: i8x8 = i8x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: u8x8 = u8x8::new(8, 0, 0xFF, 12, 16, 20, 24, 28);
        let r: u8x8 = transmute(vqrshl_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshlq_u8() {
        let a: u8x16 = u8x16::new(2, 0, 0xFF, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b: i8x16 = i8x16::new(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let e: u8x16 = u8x16::new(8, 0, 0xFF, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60);
        let r: u8x16 = transmute(vqrshlq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshl_u16() {
        let a: u16x4 = u16x4::new(2, 0, 0xFF_FF, 3);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let e: u16x4 = u16x4::new(8, 0, 0xFF_FF, 12);
        let r: u16x4 = transmute(vqrshl_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshlq_u16() {
        let a: u16x8 = u16x8::new(2, 0, 0xFF_FF, 3, 4, 5, 6, 7);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: u16x8 = u16x8::new(8, 0, 0xFF_FF, 12, 16, 20, 24, 28);
        let r: u16x8 = transmute(vqrshlq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshl_u32() {
        let a: u32x2 = u32x2::new(2, 0);
        let b: i32x2 = i32x2::new(2, 2);
        let e: u32x2 = u32x2::new(8, 0);
        let r: u32x2 = transmute(vqrshl_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshlq_u32() {
        let a: u32x4 = u32x4::new(2, 0, 0xFF_FF_FF_FF, 3);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let e: u32x4 = u32x4::new(8, 0, 0xFF_FF_FF_FF, 12);
        let r: u32x4 = transmute(vqrshlq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshl_u64() {
        let a: u64x1 = u64x1::new(2);
        let b: i64x1 = i64x1::new(2);
        let e: u64x1 = u64x1::new(8);
        let r: u64x1 = transmute(vqrshl_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshlq_u64() {
        let a: u64x2 = u64x2::new(2, 0);
        let b: i64x2 = i64x2::new(2, 2);
        let e: u64x2 = u64x2::new(8, 0);
        let r: u64x2 = transmute(vqrshlq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrn_n_s16() {
        let a: i16x8 = i16x8::new(-32768, 4, 8, 12, 16, 20, 24, 28);
        let e: i8x8 = i8x8::new(-128, 1, 2, 3, 4, 5, 6, 7);
        let r: i8x8 = transmute(vqrshrn_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrn_n_s32() {
        let a: i32x4 = i32x4::new(-2147483648, 4, 8, 12);
        let e: i16x4 = i16x4::new(-32768, 1, 2, 3);
        let r: i16x4 = transmute(vqrshrn_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrn_n_s64() {
        let a: i64x2 = i64x2::new(-9223372036854775808, 4);
        let e: i32x2 = i32x2::new(-2147483648, 1);
        let r: i32x2 = transmute(vqrshrn_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrn_n_u16() {
        let a: u16x8 = u16x8::new(0, 4, 8, 12, 16, 20, 24, 28);
        let e: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: u8x8 = transmute(vqrshrn_n_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrn_n_u32() {
        let a: u32x4 = u32x4::new(0, 4, 8, 12);
        let e: u16x4 = u16x4::new(0, 1, 2, 3);
        let r: u16x4 = transmute(vqrshrn_n_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrn_n_u64() {
        let a: u64x2 = u64x2::new(0, 4);
        let e: u32x2 = u32x2::new(0, 1);
        let r: u32x2 = transmute(vqrshrn_n_u64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrun_n_s16() {
        let a: i16x8 = i16x8::new(0, 4, 8, 12, 16, 20, 24, 28);
        let e: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: u8x8 = transmute(vqrshrun_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrun_n_s32() {
        let a: i32x4 = i32x4::new(0, 4, 8, 12);
        let e: u16x4 = u16x4::new(0, 1, 2, 3);
        let r: u16x4 = transmute(vqrshrun_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqrshrun_n_s64() {
        let a: i64x2 = i64x2::new(0, 4);
        let e: u32x2 = u32x2::new(0, 1);
        let r: u32x2 = transmute(vqrshrun_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshl_s8() {
        let a: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: i8x8 = i8x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: i8x8 = i8x8::new(0, 4, 8, 12, 16, 20, 24, 28);
        let r: i8x8 = transmute(vqshl_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlq_s8() {
        let a: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b: i8x16 = i8x16::new(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let e: i8x16 = i8x16::new(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60);
        let r: i8x16 = transmute(vqshlq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshl_s16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let e: i16x4 = i16x4::new(0, 4, 8, 12);
        let r: i16x4 = transmute(vqshl_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlq_s16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: i16x8 = i16x8::new(0, 4, 8, 12, 16, 20, 24, 28);
        let r: i16x8 = transmute(vqshlq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshl_s32() {
        let a: i32x2 = i32x2::new(0, 1);
        let b: i32x2 = i32x2::new(2, 2);
        let e: i32x2 = i32x2::new(0, 4);
        let r: i32x2 = transmute(vqshl_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlq_s32() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let e: i32x4 = i32x4::new(0, 4, 8, 12);
        let r: i32x4 = transmute(vqshlq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshl_s64() {
        let a: i64x1 = i64x1::new(0);
        let b: i64x1 = i64x1::new(2);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vqshl_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlq_s64() {
        let a: i64x2 = i64x2::new(0, 1);
        let b: i64x2 = i64x2::new(2, 2);
        let e: i64x2 = i64x2::new(0, 4);
        let r: i64x2 = transmute(vqshlq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshl_u8() {
        let a: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: i8x8 = i8x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: u8x8 = u8x8::new(0, 4, 8, 12, 16, 20, 24, 28);
        let r: u8x8 = transmute(vqshl_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlq_u8() {
        let a: u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b: i8x16 = i8x16::new(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let e: u8x16 = u8x16::new(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60);
        let r: u8x16 = transmute(vqshlq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshl_u16() {
        let a: u16x4 = u16x4::new(0, 1, 2, 3);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let e: u16x4 = u16x4::new(0, 4, 8, 12);
        let r: u16x4 = transmute(vqshl_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlq_u16() {
        let a: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: u16x8 = u16x8::new(0, 4, 8, 12, 16, 20, 24, 28);
        let r: u16x8 = transmute(vqshlq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshl_u32() {
        let a: u32x2 = u32x2::new(0, 1);
        let b: i32x2 = i32x2::new(2, 2);
        let e: u32x2 = u32x2::new(0, 4);
        let r: u32x2 = transmute(vqshl_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlq_u32() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let e: u32x4 = u32x4::new(0, 4, 8, 12);
        let r: u32x4 = transmute(vqshlq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshl_u64() {
        let a: u64x1 = u64x1::new(0);
        let b: i64x1 = i64x1::new(2);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vqshl_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlq_u64() {
        let a: u64x2 = u64x2::new(0, 1);
        let b: i64x2 = i64x2::new(2, 2);
        let e: u64x2 = u64x2::new(0, 4);
        let r: u64x2 = transmute(vqshlq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshl_n_s8() {
        let a: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i8x8 = i8x8::new(0, 4, 8, 12, 16, 20, 24, 28);
        let r: i8x8 = transmute(vqshl_n_s8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlq_n_s8() {
        let a: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: i8x16 = i8x16::new(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60);
        let r: i8x16 = transmute(vqshlq_n_s8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshl_n_s16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let e: i16x4 = i16x4::new(0, 4, 8, 12);
        let r: i16x4 = transmute(vqshl_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlq_n_s16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i16x8 = i16x8::new(0, 4, 8, 12, 16, 20, 24, 28);
        let r: i16x8 = transmute(vqshlq_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshl_n_s32() {
        let a: i32x2 = i32x2::new(0, 1);
        let e: i32x2 = i32x2::new(0, 4);
        let r: i32x2 = transmute(vqshl_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlq_n_s32() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let e: i32x4 = i32x4::new(0, 4, 8, 12);
        let r: i32x4 = transmute(vqshlq_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshl_n_s64() {
        let a: i64x1 = i64x1::new(0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vqshl_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlq_n_s64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: i64x2 = i64x2::new(0, 4);
        let r: i64x2 = transmute(vqshlq_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshl_n_u8() {
        let a: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u8x8 = u8x8::new(0, 4, 8, 12, 16, 20, 24, 28);
        let r: u8x8 = transmute(vqshl_n_u8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlq_n_u8() {
        let a: u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u8x16 = u8x16::new(0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60);
        let r: u8x16 = transmute(vqshlq_n_u8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshl_n_u16() {
        let a: u16x4 = u16x4::new(0, 1, 2, 3);
        let e: u16x4 = u16x4::new(0, 4, 8, 12);
        let r: u16x4 = transmute(vqshl_n_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlq_n_u16() {
        let a: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u16x8 = u16x8::new(0, 4, 8, 12, 16, 20, 24, 28);
        let r: u16x8 = transmute(vqshlq_n_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshl_n_u32() {
        let a: u32x2 = u32x2::new(0, 1);
        let e: u32x2 = u32x2::new(0, 4);
        let r: u32x2 = transmute(vqshl_n_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlq_n_u32() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let e: u32x4 = u32x4::new(0, 4, 8, 12);
        let r: u32x4 = transmute(vqshlq_n_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshl_n_u64() {
        let a: u64x1 = u64x1::new(0);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vqshl_n_u64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshlq_n_u64() {
        let a: u64x2 = u64x2::new(0, 1);
        let e: u64x2 = u64x2::new(0, 4);
        let r: u64x2 = transmute(vqshlq_n_u64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrn_n_s16() {
        let a: i16x8 = i16x8::new(0, 4, 8, 12, 16, 20, 24, 28);
        let e: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i8x8 = transmute(vqshrn_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrn_n_s32() {
        let a: i32x4 = i32x4::new(0, 4, 8, 12);
        let e: i16x4 = i16x4::new(0, 1, 2, 3);
        let r: i16x4 = transmute(vqshrn_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrn_n_s64() {
        let a: i64x2 = i64x2::new(0, 4);
        let e: i32x2 = i32x2::new(0, 1);
        let r: i32x2 = transmute(vqshrn_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrn_n_u16() {
        let a: u16x8 = u16x8::new(0, 4, 8, 12, 16, 20, 24, 28);
        let e: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: u8x8 = transmute(vqshrn_n_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrn_n_u32() {
        let a: u32x4 = u32x4::new(0, 4, 8, 12);
        let e: u16x4 = u16x4::new(0, 1, 2, 3);
        let r: u16x4 = transmute(vqshrn_n_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrn_n_u64() {
        let a: u64x2 = u64x2::new(0, 4);
        let e: u32x2 = u32x2::new(0, 1);
        let r: u32x2 = transmute(vqshrn_n_u64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrun_n_s16() {
        let a: i16x8 = i16x8::new(0, 4, 8, 12, 16, 20, 24, 28);
        let e: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: u8x8 = transmute(vqshrun_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrun_n_s32() {
        let a: i32x4 = i32x4::new(0, 4, 8, 12);
        let e: u16x4 = u16x4::new(0, 1, 2, 3);
        let r: u16x4 = transmute(vqshrun_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqshrun_n_s64() {
        let a: i64x2 = i64x2::new(0, 4);
        let e: u32x2 = u32x2::new(0, 1);
        let r: u32x2 = transmute(vqshrun_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsqrte_f32() {
        let a: f32x2 = f32x2::new(1.0, 2.0);
        let e: f32x2 = f32x2::new(0.998046875, 0.705078125);
        let r: f32x2 = transmute(vrsqrte_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsqrteq_f32() {
        let a: f32x4 = f32x4::new(1.0, 2.0, 3.0, 4.0);
        let e: f32x4 = f32x4::new(0.998046875, 0.705078125, 0.576171875, 0.4990234375);
        let r: f32x4 = transmute(vrsqrteq_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrecpe_f32() {
        let a: f32x2 = f32x2::new(4.0, 3.0);
        let e: f32x2 = f32x2::new(0.24951171875, 0.3330078125);
        let r: f32x2 = transmute(vrecpe_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrecpeq_f32() {
        let a: f32x4 = f32x4::new(4.0, 3.0, 2.0, 1.0);
        let e: f32x4 = f32x4::new(0.24951171875, 0.3330078125, 0.4990234375, 0.998046875);
        let r: f32x4 = transmute(vrecpeq_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s8_u8() {
        let a: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i8x8 = transmute(vreinterpret_s8_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s8_p8() {
        let a: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i8x8 = transmute(vreinterpret_s8_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s16_p16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let e: i16x4 = i16x4::new(0, 1, 2, 3);
        let r: i16x4 = transmute(vreinterpret_s16_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s16_u16() {
        let a: u16x4 = u16x4::new(0, 1, 2, 3);
        let e: i16x4 = i16x4::new(0, 1, 2, 3);
        let r: i16x4 = transmute(vreinterpret_s16_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s32_u32() {
        let a: u32x2 = u32x2::new(0, 1);
        let e: i32x2 = i32x2::new(0, 1);
        let r: i32x2 = transmute(vreinterpret_s32_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s64_u64() {
        let a: u64x1 = u64x1::new(0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_s64_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s8_u8() {
        let a: u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r: i8x16 = transmute(vreinterpretq_s8_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s8_p8() {
        let a: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r: i8x16 = transmute(vreinterpretq_s8_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s16_p16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i16x8 = transmute(vreinterpretq_s16_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s16_u16() {
        let a: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i16x8 = transmute(vreinterpretq_s16_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s32_u32() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let e: i32x4 = i32x4::new(0, 1, 2, 3);
        let r: i32x4 = transmute(vreinterpretq_s32_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s64_u64() {
        let a: u64x2 = u64x2::new(0, 1);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_s64_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u8_p8() {
        let a: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: u8x8 = transmute(vreinterpret_u8_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u8_s8() {
        let a: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: u8x8 = transmute(vreinterpret_u8_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u16_p16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let e: u16x4 = u16x4::new(0, 1, 2, 3);
        let r: u16x4 = transmute(vreinterpret_u16_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u16_s16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let e: u16x4 = u16x4::new(0, 1, 2, 3);
        let r: u16x4 = transmute(vreinterpret_u16_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u32_s32() {
        let a: i32x2 = i32x2::new(0, 1);
        let e: u32x2 = u32x2::new(0, 1);
        let r: u32x2 = transmute(vreinterpret_u32_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u64_s64() {
        let a: i64x1 = i64x1::new(0);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vreinterpret_u64_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u8_p8() {
        let a: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r: u8x16 = transmute(vreinterpretq_u8_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u8_s8() {
        let a: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r: u8x16 = transmute(vreinterpretq_u8_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u16_p16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: u16x8 = transmute(vreinterpretq_u16_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u16_s16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: u16x8 = transmute(vreinterpretq_u16_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u32_s32() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let e: u32x4 = u32x4::new(0, 1, 2, 3);
        let r: u32x4 = transmute(vreinterpretq_u32_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u64_s64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: u64x2 = u64x2::new(0, 1);
        let r: u64x2 = transmute(vreinterpretq_u64_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p8_s8() {
        let a: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i8x8 = transmute(vreinterpret_p8_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p8_u8() {
        let a: u8x8 = u8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i8x8 = i8x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i8x8 = transmute(vreinterpret_p8_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p16_s16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let e: i16x4 = i16x4::new(0, 1, 2, 3);
        let r: i16x4 = transmute(vreinterpret_p16_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p16_u16() {
        let a: u16x4 = u16x4::new(0, 1, 2, 3);
        let e: i16x4 = i16x4::new(0, 1, 2, 3);
        let r: i16x4 = transmute(vreinterpret_p16_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p8_s8() {
        let a: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r: i8x16 = transmute(vreinterpretq_p8_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p8_u8() {
        let a: u8x16 = u8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let e: i8x16 = i8x16::new(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let r: i8x16 = transmute(vreinterpretq_p8_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p16_s16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i16x8 = transmute(vreinterpretq_p16_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p16_u16() {
        let a: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i16x8 = transmute(vreinterpretq_p16_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s8_s16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let e: i8x8 = i8x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let r: i8x8 = transmute(vreinterpret_s8_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s8_u16() {
        let a: u16x4 = u16x4::new(0, 1, 2, 3);
        let e: i8x8 = i8x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let r: i8x8 = transmute(vreinterpret_s8_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s8_p16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let e: i8x8 = i8x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let r: i8x8 = transmute(vreinterpret_s8_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s16_s32() {
        let a: i32x2 = i32x2::new(0, 1);
        let e: i16x4 = i16x4::new(0, 0, 1, 0);
        let r: i16x4 = transmute(vreinterpret_s16_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s16_u32() {
        let a: u32x2 = u32x2::new(0, 1);
        let e: i16x4 = i16x4::new(0, 0, 1, 0);
        let r: i16x4 = transmute(vreinterpret_s16_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s32_s64() {
        let a: i64x1 = i64x1::new(0);
        let e: i32x2 = i32x2::new(0, 0);
        let r: i32x2 = transmute(vreinterpret_s32_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s32_u64() {
        let a: u64x1 = u64x1::new(0);
        let e: i32x2 = i32x2::new(0, 0);
        let r: i32x2 = transmute(vreinterpret_s32_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s8_s16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i8x16 = i8x16::new(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0);
        let r: i8x16 = transmute(vreinterpretq_s8_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s8_u16() {
        let a: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i8x16 = i8x16::new(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0);
        let r: i8x16 = transmute(vreinterpretq_s8_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s8_p16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i8x16 = i8x16::new(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0);
        let r: i8x16 = transmute(vreinterpretq_s8_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s16_s32() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let e: i16x8 = i16x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let r: i16x8 = transmute(vreinterpretq_s16_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s16_u32() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let e: i16x8 = i16x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let r: i16x8 = transmute(vreinterpretq_s16_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s32_s64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: i32x4 = i32x4::new(0, 0, 1, 0);
        let r: i32x4 = transmute(vreinterpretq_s32_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s32_u64() {
        let a: u64x2 = u64x2::new(0, 1);
        let e: i32x4 = i32x4::new(0, 0, 1, 0);
        let r: i32x4 = transmute(vreinterpretq_s32_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u8_p16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let e: u8x8 = u8x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let r: u8x8 = transmute(vreinterpret_u8_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u8_s16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let e: u8x8 = u8x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let r: u8x8 = transmute(vreinterpret_u8_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u8_u16() {
        let a: u16x4 = u16x4::new(0, 1, 2, 3);
        let e: u8x8 = u8x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let r: u8x8 = transmute(vreinterpret_u8_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u16_s32() {
        let a: i32x2 = i32x2::new(0, 1);
        let e: u16x4 = u16x4::new(0, 0, 1, 0);
        let r: u16x4 = transmute(vreinterpret_u16_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u16_u32() {
        let a: u32x2 = u32x2::new(0, 1);
        let e: u16x4 = u16x4::new(0, 0, 1, 0);
        let r: u16x4 = transmute(vreinterpret_u16_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u32_s64() {
        let a: i64x1 = i64x1::new(0);
        let e: u32x2 = u32x2::new(0, 0);
        let r: u32x2 = transmute(vreinterpret_u32_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u32_u64() {
        let a: u64x1 = u64x1::new(0);
        let e: u32x2 = u32x2::new(0, 0);
        let r: u32x2 = transmute(vreinterpret_u32_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u8_p16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u8x16 = u8x16::new(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0);
        let r: u8x16 = transmute(vreinterpretq_u8_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u8_s16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u8x16 = u8x16::new(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0);
        let r: u8x16 = transmute(vreinterpretq_u8_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u8_u16() {
        let a: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: u8x16 = u8x16::new(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0);
        let r: u8x16 = transmute(vreinterpretq_u8_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u16_s32() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let e: u16x8 = u16x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let r: u16x8 = transmute(vreinterpretq_u16_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u16_u32() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let e: u16x8 = u16x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let r: u16x8 = transmute(vreinterpretq_u16_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u32_s64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: u32x4 = u32x4::new(0, 0, 1, 0);
        let r: u32x4 = transmute(vreinterpretq_u32_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u32_u64() {
        let a: u64x2 = u64x2::new(0, 1);
        let e: u32x4 = u32x4::new(0, 0, 1, 0);
        let r: u32x4 = transmute(vreinterpretq_u32_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p8_p16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let e: i8x8 = i8x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let r: i8x8 = transmute(vreinterpret_p8_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p8_s16() {
        let a: i16x4 = i16x4::new(0, 1, 2, 3);
        let e: i8x8 = i8x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let r: i8x8 = transmute(vreinterpret_p8_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p8_u16() {
        let a: u16x4 = u16x4::new(0, 1, 2, 3);
        let e: i8x8 = i8x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let r: i8x8 = transmute(vreinterpret_p8_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p16_s32() {
        let a: i32x2 = i32x2::new(0, 1);
        let e: i16x4 = i16x4::new(0, 0, 1, 0);
        let r: i16x4 = transmute(vreinterpret_p16_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p16_u32() {
        let a: u32x2 = u32x2::new(0, 1);
        let e: i16x4 = i16x4::new(0, 0, 1, 0);
        let r: i16x4 = transmute(vreinterpret_p16_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p8_p16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i8x16 = i8x16::new(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0);
        let r: i8x16 = transmute(vreinterpretq_p8_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p8_s16() {
        let a: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i8x16 = i8x16::new(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0);
        let r: i8x16 = transmute(vreinterpretq_p8_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p8_u16() {
        let a: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let e: i8x16 = i8x16::new(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0);
        let r: i8x16 = transmute(vreinterpretq_p8_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p16_s32() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let e: i16x8 = i16x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let r: i16x8 = transmute(vreinterpretq_p16_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p16_u32() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let e: i16x8 = i16x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let r: i16x8 = transmute(vreinterpretq_p16_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s16_p8() {
        let a: i8x8 = i8x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let e: i16x4 = i16x4::new(0, 1, 2, 3);
        let r: i16x4 = transmute(vreinterpret_s16_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s16_s8() {
        let a: i8x8 = i8x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let e: i16x4 = i16x4::new(0, 1, 2, 3);
        let r: i16x4 = transmute(vreinterpret_s16_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s16_u8() {
        let a: u8x8 = u8x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let e: i16x4 = i16x4::new(0, 1, 2, 3);
        let r: i16x4 = transmute(vreinterpret_s16_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s32_p16() {
        let a: i16x4 = i16x4::new(0, 0, 1, 0);
        let e: i32x2 = i32x2::new(0, 1);
        let r: i32x2 = transmute(vreinterpret_s32_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s32_s16() {
        let a: i16x4 = i16x4::new(0, 0, 1, 0);
        let e: i32x2 = i32x2::new(0, 1);
        let r: i32x2 = transmute(vreinterpret_s32_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s32_u16() {
        let a: u16x4 = u16x4::new(0, 0, 1, 0);
        let e: i32x2 = i32x2::new(0, 1);
        let r: i32x2 = transmute(vreinterpret_s32_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s64_s32() {
        let a: i32x2 = i32x2::new(0, 0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_s64_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s64_u32() {
        let a: u32x2 = u32x2::new(0, 0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_s64_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s16_p8() {
        let a: i8x16 = i8x16::new(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0);
        let e: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i16x8 = transmute(vreinterpretq_s16_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s16_s8() {
        let a: i8x16 = i8x16::new(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0);
        let e: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i16x8 = transmute(vreinterpretq_s16_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s16_u8() {
        let a: u8x16 = u8x16::new(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0);
        let e: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i16x8 = transmute(vreinterpretq_s16_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s32_p16() {
        let a: i16x8 = i16x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let e: i32x4 = i32x4::new(0, 1, 2, 3);
        let r: i32x4 = transmute(vreinterpretq_s32_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s32_s16() {
        let a: i16x8 = i16x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let e: i32x4 = i32x4::new(0, 1, 2, 3);
        let r: i32x4 = transmute(vreinterpretq_s32_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s32_u16() {
        let a: u16x8 = u16x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let e: i32x4 = i32x4::new(0, 1, 2, 3);
        let r: i32x4 = transmute(vreinterpretq_s32_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s64_s32() {
        let a: i32x4 = i32x4::new(0, 0, 1, 0);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_s64_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s64_u32() {
        let a: u32x4 = u32x4::new(0, 0, 1, 0);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_s64_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u16_p8() {
        let a: i8x8 = i8x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let e: u16x4 = u16x4::new(0, 1, 2, 3);
        let r: u16x4 = transmute(vreinterpret_u16_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u16_s8() {
        let a: i8x8 = i8x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let e: u16x4 = u16x4::new(0, 1, 2, 3);
        let r: u16x4 = transmute(vreinterpret_u16_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u16_u8() {
        let a: u8x8 = u8x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let e: u16x4 = u16x4::new(0, 1, 2, 3);
        let r: u16x4 = transmute(vreinterpret_u16_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u32_p16() {
        let a: i16x4 = i16x4::new(0, 0, 1, 0);
        let e: u32x2 = u32x2::new(0, 1);
        let r: u32x2 = transmute(vreinterpret_u32_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u32_s16() {
        let a: i16x4 = i16x4::new(0, 0, 1, 0);
        let e: u32x2 = u32x2::new(0, 1);
        let r: u32x2 = transmute(vreinterpret_u32_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u32_u16() {
        let a: u16x4 = u16x4::new(0, 0, 1, 0);
        let e: u32x2 = u32x2::new(0, 1);
        let r: u32x2 = transmute(vreinterpret_u32_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u64_s32() {
        let a: i32x2 = i32x2::new(0, 0);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vreinterpret_u64_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u64_u32() {
        let a: u32x2 = u32x2::new(0, 0);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vreinterpret_u64_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u16_p8() {
        let a: i8x16 = i8x16::new(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0);
        let e: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: u16x8 = transmute(vreinterpretq_u16_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u16_s8() {
        let a: i8x16 = i8x16::new(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0);
        let e: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: u16x8 = transmute(vreinterpretq_u16_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u16_u8() {
        let a: u8x16 = u8x16::new(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0);
        let e: u16x8 = u16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: u16x8 = transmute(vreinterpretq_u16_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u32_p16() {
        let a: i16x8 = i16x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let e: u32x4 = u32x4::new(0, 1, 2, 3);
        let r: u32x4 = transmute(vreinterpretq_u32_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u32_s16() {
        let a: i16x8 = i16x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let e: u32x4 = u32x4::new(0, 1, 2, 3);
        let r: u32x4 = transmute(vreinterpretq_u32_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u32_u16() {
        let a: u16x8 = u16x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let e: u32x4 = u32x4::new(0, 1, 2, 3);
        let r: u32x4 = transmute(vreinterpretq_u32_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u64_s32() {
        let a: i32x4 = i32x4::new(0, 0, 1, 0);
        let e: u64x2 = u64x2::new(0, 1);
        let r: u64x2 = transmute(vreinterpretq_u64_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u64_u32() {
        let a: u32x4 = u32x4::new(0, 0, 1, 0);
        let e: u64x2 = u64x2::new(0, 1);
        let r: u64x2 = transmute(vreinterpretq_u64_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p16_p8() {
        let a: i8x8 = i8x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let e: i16x4 = i16x4::new(0, 1, 2, 3);
        let r: i16x4 = transmute(vreinterpret_p16_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p16_s8() {
        let a: i8x8 = i8x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let e: i16x4 = i16x4::new(0, 1, 2, 3);
        let r: i16x4 = transmute(vreinterpret_p16_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p16_u8() {
        let a: u8x8 = u8x8::new(0, 0, 1, 0, 2, 0, 3, 0);
        let e: i16x4 = i16x4::new(0, 1, 2, 3);
        let r: i16x4 = transmute(vreinterpret_p16_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p16_p8() {
        let a: i8x16 = i8x16::new(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0);
        let e: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i16x8 = transmute(vreinterpretq_p16_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p16_s8() {
        let a: i8x16 = i8x16::new(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0);
        let e: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i16x8 = transmute(vreinterpretq_p16_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p16_u8() {
        let a: u8x16 = u8x16::new(0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0);
        let e: i16x8 = i16x8::new(0, 1, 2, 3, 4, 5, 6, 7);
        let r: i16x8 = transmute(vreinterpretq_p16_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s8_s32() {
        let a: i32x2 = i32x2::new(0, 1);
        let e: i8x8 = i8x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let r: i8x8 = transmute(vreinterpret_s8_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s8_u32() {
        let a: u32x2 = u32x2::new(0, 1);
        let e: i8x8 = i8x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let r: i8x8 = transmute(vreinterpret_s8_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s16_s64() {
        let a: i64x1 = i64x1::new(0);
        let e: i16x4 = i16x4::new(0, 0, 0, 0);
        let r: i16x4 = transmute(vreinterpret_s16_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s16_u64() {
        let a: u64x1 = u64x1::new(0);
        let e: i16x4 = i16x4::new(0, 0, 0, 0);
        let r: i16x4 = transmute(vreinterpret_s16_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s8_s32() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let e: i8x16 = i8x16::new(0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0);
        let r: i8x16 = transmute(vreinterpretq_s8_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s8_u32() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let e: i8x16 = i8x16::new(0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0);
        let r: i8x16 = transmute(vreinterpretq_s8_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s16_s64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: i16x8 = i16x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let r: i16x8 = transmute(vreinterpretq_s16_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s16_u64() {
        let a: u64x2 = u64x2::new(0, 1);
        let e: i16x8 = i16x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let r: i16x8 = transmute(vreinterpretq_s16_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u8_s32() {
        let a: i32x2 = i32x2::new(0, 1);
        let e: u8x8 = u8x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let r: u8x8 = transmute(vreinterpret_u8_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u8_u32() {
        let a: u32x2 = u32x2::new(0, 1);
        let e: u8x8 = u8x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let r: u8x8 = transmute(vreinterpret_u8_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u16_s64() {
        let a: i64x1 = i64x1::new(0);
        let e: u16x4 = u16x4::new(0, 0, 0, 0);
        let r: u16x4 = transmute(vreinterpret_u16_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u16_u64() {
        let a: u64x1 = u64x1::new(0);
        let e: u16x4 = u16x4::new(0, 0, 0, 0);
        let r: u16x4 = transmute(vreinterpret_u16_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u8_s32() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let e: u8x16 = u8x16::new(0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0);
        let r: u8x16 = transmute(vreinterpretq_u8_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u8_u32() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let e: u8x16 = u8x16::new(0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0);
        let r: u8x16 = transmute(vreinterpretq_u8_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u16_s64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: u16x8 = u16x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let r: u16x8 = transmute(vreinterpretq_u16_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u16_u64() {
        let a: u64x2 = u64x2::new(0, 1);
        let e: u16x8 = u16x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let r: u16x8 = transmute(vreinterpretq_u16_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p8_s32() {
        let a: i32x2 = i32x2::new(0, 1);
        let e: i8x8 = i8x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let r: i8x8 = transmute(vreinterpret_p8_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p8_u32() {
        let a: u32x2 = u32x2::new(0, 1);
        let e: i8x8 = i8x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let r: i8x8 = transmute(vreinterpret_p8_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p16_s64() {
        let a: i64x1 = i64x1::new(0);
        let e: i16x4 = i16x4::new(0, 0, 0, 0);
        let r: i16x4 = transmute(vreinterpret_p16_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p16_u64() {
        let a: u64x1 = u64x1::new(0);
        let e: i16x4 = i16x4::new(0, 0, 0, 0);
        let r: i16x4 = transmute(vreinterpret_p16_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p8_s32() {
        let a: i32x4 = i32x4::new(0, 1, 2, 3);
        let e: i8x16 = i8x16::new(0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0);
        let r: i8x16 = transmute(vreinterpretq_p8_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p8_u32() {
        let a: u32x4 = u32x4::new(0, 1, 2, 3);
        let e: i8x16 = i8x16::new(0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0);
        let r: i8x16 = transmute(vreinterpretq_p8_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p16_s64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: i16x8 = i16x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let r: i16x8 = transmute(vreinterpretq_p16_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p16_u64() {
        let a: u64x2 = u64x2::new(0, 1);
        let e: i16x8 = i16x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let r: i16x8 = transmute(vreinterpretq_p16_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s32_p8() {
        let a: i8x8 = i8x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let e: i32x2 = i32x2::new(0, 1);
        let r: i32x2 = transmute(vreinterpret_s32_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s32_s8() {
        let a: i8x8 = i8x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let e: i32x2 = i32x2::new(0, 1);
        let r: i32x2 = transmute(vreinterpret_s32_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s32_u8() {
        let a: u8x8 = u8x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let e: i32x2 = i32x2::new(0, 1);
        let r: i32x2 = transmute(vreinterpret_s32_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s64_p16() {
        let a: i16x4 = i16x4::new(0, 0, 0, 0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_s64_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s64_s16() {
        let a: i16x4 = i16x4::new(0, 0, 0, 0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_s64_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s64_u16() {
        let a: u16x4 = u16x4::new(0, 0, 0, 0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_s64_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s32_p8() {
        let a: i8x16 = i8x16::new(0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0);
        let e: i32x4 = i32x4::new(0, 1, 2, 3);
        let r: i32x4 = transmute(vreinterpretq_s32_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s32_s8() {
        let a: i8x16 = i8x16::new(0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0);
        let e: i32x4 = i32x4::new(0, 1, 2, 3);
        let r: i32x4 = transmute(vreinterpretq_s32_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s32_u8() {
        let a: u8x16 = u8x16::new(0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0);
        let e: i32x4 = i32x4::new(0, 1, 2, 3);
        let r: i32x4 = transmute(vreinterpretq_s32_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s64_p16() {
        let a: i16x8 = i16x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_s64_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s64_s16() {
        let a: i16x8 = i16x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_s64_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s64_u16() {
        let a: u16x8 = u16x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_s64_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u32_p8() {
        let a: i8x8 = i8x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let e: u32x2 = u32x2::new(0, 1);
        let r: u32x2 = transmute(vreinterpret_u32_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u32_s8() {
        let a: i8x8 = i8x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let e: u32x2 = u32x2::new(0, 1);
        let r: u32x2 = transmute(vreinterpret_u32_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u32_u8() {
        let a: u8x8 = u8x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let e: u32x2 = u32x2::new(0, 1);
        let r: u32x2 = transmute(vreinterpret_u32_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u64_p16() {
        let a: i16x4 = i16x4::new(0, 0, 0, 0);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vreinterpret_u64_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u64_s16() {
        let a: i16x4 = i16x4::new(0, 0, 0, 0);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vreinterpret_u64_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u64_u16() {
        let a: u16x4 = u16x4::new(0, 0, 0, 0);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vreinterpret_u64_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u32_p8() {
        let a: i8x16 = i8x16::new(0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0);
        let e: u32x4 = u32x4::new(0, 1, 2, 3);
        let r: u32x4 = transmute(vreinterpretq_u32_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u32_s8() {
        let a: i8x16 = i8x16::new(0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0);
        let e: u32x4 = u32x4::new(0, 1, 2, 3);
        let r: u32x4 = transmute(vreinterpretq_u32_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u32_u8() {
        let a: u8x16 = u8x16::new(0, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0);
        let e: u32x4 = u32x4::new(0, 1, 2, 3);
        let r: u32x4 = transmute(vreinterpretq_u32_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u64_p16() {
        let a: i16x8 = i16x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let e: u64x2 = u64x2::new(0, 1);
        let r: u64x2 = transmute(vreinterpretq_u64_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u64_s16() {
        let a: i16x8 = i16x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let e: u64x2 = u64x2::new(0, 1);
        let r: u64x2 = transmute(vreinterpretq_u64_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u64_u16() {
        let a: u16x8 = u16x8::new(0, 0, 0, 0, 1, 0, 0, 0);
        let e: u64x2 = u64x2::new(0, 1);
        let r: u64x2 = transmute(vreinterpretq_u64_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s8_s64() {
        let a: i64x1 = i64x1::new(0);
        let e: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x8 = transmute(vreinterpret_s8_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s8_u64() {
        let a: u64x1 = u64x1::new(0);
        let e: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x8 = transmute(vreinterpret_s8_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u8_s64() {
        let a: i64x1 = i64x1::new(0);
        let e: u8x8 = u8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: u8x8 = transmute(vreinterpret_u8_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u8_u64() {
        let a: u64x1 = u64x1::new(0);
        let e: u8x8 = u8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: u8x8 = transmute(vreinterpret_u8_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p8_s64() {
        let a: i64x1 = i64x1::new(0);
        let e: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x8 = transmute(vreinterpret_p8_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p8_u64() {
        let a: u64x1 = u64x1::new(0);
        let e: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x8 = transmute(vreinterpret_p8_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s8_s64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x16 = transmute(vreinterpretq_s8_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s8_u64() {
        let a: u64x2 = u64x2::new(0, 1);
        let e: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x16 = transmute(vreinterpretq_s8_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u8_s64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: u8x16 = u8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
        let r: u8x16 = transmute(vreinterpretq_u8_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u8_u64() {
        let a: u64x2 = u64x2::new(0, 1);
        let e: u8x16 = u8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
        let r: u8x16 = transmute(vreinterpretq_u8_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p8_s64() {
        let a: i64x2 = i64x2::new(0, 1);
        let e: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x16 = transmute(vreinterpretq_p8_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p8_u64() {
        let a: u64x2 = u64x2::new(0, 1);
        let e: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x16 = transmute(vreinterpretq_p8_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s64_p8() {
        let a: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_s64_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s64_s8() {
        let a: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_s64_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s64_u8() {
        let a: u8x8 = u8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_s64_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u64_p8() {
        let a: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vreinterpret_u64_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u64_s8() {
        let a: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vreinterpret_u64_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u64_u8() {
        let a: u8x8 = u8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vreinterpret_u64_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s64_p8() {
        let a: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_s64_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s64_s8() {
        let a: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_s64_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s64_u8() {
        let a: u8x16 = u8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
        let e: i64x2 = i64x2::new(0, 1);
        let r: i64x2 = transmute(vreinterpretq_s64_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u64_p8() {
        let a: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
        let e: u64x2 = u64x2::new(0, 1);
        let r: u64x2 = transmute(vreinterpretq_u64_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u64_s8() {
        let a: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
        let e: u64x2 = u64x2::new(0, 1);
        let r: u64x2 = transmute(vreinterpretq_u64_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u64_u8() {
        let a: u8x16 = u8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0);
        let e: u64x2 = u64x2::new(0, 1);
        let r: u64x2 = transmute(vreinterpretq_u64_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s8_f32() {
        let a: f32x2 = f32x2::new(0., 0.);
        let e: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x8 = transmute(vreinterpret_s8_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s16_f32() {
        let a: f32x2 = f32x2::new(0., 0.);
        let e: i16x4 = i16x4::new(0, 0, 0, 0);
        let r: i16x4 = transmute(vreinterpret_s16_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s32_f32() {
        let a: f32x2 = f32x2::new(0., 0.);
        let e: i32x2 = i32x2::new(0, 0);
        let r: i32x2 = transmute(vreinterpret_s32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_s64_f32() {
        let a: f32x2 = f32x2::new(0., 0.);
        let e: i64x1 = i64x1::new(0);
        let r: i64x1 = transmute(vreinterpret_s64_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s8_f32() {
        let a: f32x4 = f32x4::new(0., 0., 0., 0.);
        let e: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x16 = transmute(vreinterpretq_s8_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s16_f32() {
        let a: f32x4 = f32x4::new(0., 0., 0., 0.);
        let e: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: i16x8 = transmute(vreinterpretq_s16_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s32_f32() {
        let a: f32x4 = f32x4::new(0., 0., 0., 0.);
        let e: i32x4 = i32x4::new(0, 0, 0, 0);
        let r: i32x4 = transmute(vreinterpretq_s32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_s64_f32() {
        let a: f32x4 = f32x4::new(0., 0., 0., 0.);
        let e: i64x2 = i64x2::new(0, 0);
        let r: i64x2 = transmute(vreinterpretq_s64_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u8_f32() {
        let a: f32x2 = f32x2::new(0., 0.);
        let e: u8x8 = u8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: u8x8 = transmute(vreinterpret_u8_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u16_f32() {
        let a: f32x2 = f32x2::new(0., 0.);
        let e: u16x4 = u16x4::new(0, 0, 0, 0);
        let r: u16x4 = transmute(vreinterpret_u16_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u32_f32() {
        let a: f32x2 = f32x2::new(0., 0.);
        let e: u32x2 = u32x2::new(0, 0);
        let r: u32x2 = transmute(vreinterpret_u32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_u64_f32() {
        let a: f32x2 = f32x2::new(0., 0.);
        let e: u64x1 = u64x1::new(0);
        let r: u64x1 = transmute(vreinterpret_u64_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u8_f32() {
        let a: f32x4 = f32x4::new(0., 0., 0., 0.);
        let e: u8x16 = u8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r: u8x16 = transmute(vreinterpretq_u8_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u16_f32() {
        let a: f32x4 = f32x4::new(0., 0., 0., 0.);
        let e: u16x8 = u16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: u16x8 = transmute(vreinterpretq_u16_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u32_f32() {
        let a: f32x4 = f32x4::new(0., 0., 0., 0.);
        let e: u32x4 = u32x4::new(0, 0, 0, 0);
        let r: u32x4 = transmute(vreinterpretq_u32_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_u64_f32() {
        let a: f32x4 = f32x4::new(0., 0., 0., 0.);
        let e: u64x2 = u64x2::new(0, 0);
        let r: u64x2 = transmute(vreinterpretq_u64_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p8_f32() {
        let a: f32x2 = f32x2::new(0., 0.);
        let e: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x8 = transmute(vreinterpret_p8_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_p16_f32() {
        let a: f32x2 = f32x2::new(0., 0.);
        let e: i16x4 = i16x4::new(0, 0, 0, 0);
        let r: i16x4 = transmute(vreinterpret_p16_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p8_f32() {
        let a: f32x4 = f32x4::new(0., 0., 0., 0.);
        let e: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let r: i8x16 = transmute(vreinterpretq_p8_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_p16_f32() {
        let a: f32x4 = f32x4::new(0., 0., 0., 0.);
        let e: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let r: i16x8 = transmute(vreinterpretq_p16_f32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f32_s8() {
        let a: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: f32x2 = f32x2::new(0., 0.);
        let r: f32x2 = transmute(vreinterpret_f32_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f32_s16() {
        let a: i16x4 = i16x4::new(0, 0, 0, 0);
        let e: f32x2 = f32x2::new(0., 0.);
        let r: f32x2 = transmute(vreinterpret_f32_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f32_s32() {
        let a: i32x2 = i32x2::new(0, 0);
        let e: f32x2 = f32x2::new(0., 0.);
        let r: f32x2 = transmute(vreinterpret_f32_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f32_s64() {
        let a: i64x1 = i64x1::new(0);
        let e: f32x2 = f32x2::new(0., 0.);
        let r: f32x2 = transmute(vreinterpret_f32_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f32_s8() {
        let a: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let e: f32x4 = f32x4::new(0., 0., 0., 0.);
        let r: f32x4 = transmute(vreinterpretq_f32_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f32_s16() {
        let a: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: f32x4 = f32x4::new(0., 0., 0., 0.);
        let r: f32x4 = transmute(vreinterpretq_f32_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f32_s32() {
        let a: i32x4 = i32x4::new(0, 0, 0, 0);
        let e: f32x4 = f32x4::new(0., 0., 0., 0.);
        let r: f32x4 = transmute(vreinterpretq_f32_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f32_s64() {
        let a: i64x2 = i64x2::new(0, 0);
        let e: f32x4 = f32x4::new(0., 0., 0., 0.);
        let r: f32x4 = transmute(vreinterpretq_f32_s64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f32_u8() {
        let a: u8x8 = u8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: f32x2 = f32x2::new(0., 0.);
        let r: f32x2 = transmute(vreinterpret_f32_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f32_u16() {
        let a: u16x4 = u16x4::new(0, 0, 0, 0);
        let e: f32x2 = f32x2::new(0., 0.);
        let r: f32x2 = transmute(vreinterpret_f32_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f32_u32() {
        let a: u32x2 = u32x2::new(0, 0);
        let e: f32x2 = f32x2::new(0., 0.);
        let r: f32x2 = transmute(vreinterpret_f32_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f32_u64() {
        let a: u64x1 = u64x1::new(0);
        let e: f32x2 = f32x2::new(0., 0.);
        let r: f32x2 = transmute(vreinterpret_f32_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f32_u8() {
        let a: u8x16 = u8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let e: f32x4 = f32x4::new(0., 0., 0., 0.);
        let r: f32x4 = transmute(vreinterpretq_f32_u8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f32_u16() {
        let a: u16x8 = u16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: f32x4 = f32x4::new(0., 0., 0., 0.);
        let r: f32x4 = transmute(vreinterpretq_f32_u16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f32_u32() {
        let a: u32x4 = u32x4::new(0, 0, 0, 0);
        let e: f32x4 = f32x4::new(0., 0., 0., 0.);
        let r: f32x4 = transmute(vreinterpretq_f32_u32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f32_u64() {
        let a: u64x2 = u64x2::new(0, 0);
        let e: f32x4 = f32x4::new(0., 0., 0., 0.);
        let r: f32x4 = transmute(vreinterpretq_f32_u64(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f32_p8() {
        let a: i8x8 = i8x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: f32x2 = f32x2::new(0., 0.);
        let r: f32x2 = transmute(vreinterpret_f32_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpret_f32_p16() {
        let a: i16x4 = i16x4::new(0, 0, 0, 0);
        let e: f32x2 = f32x2::new(0., 0.);
        let r: f32x2 = transmute(vreinterpret_f32_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f32_p8() {
        let a: i8x16 = i8x16::new(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        let e: f32x4 = f32x4::new(0., 0., 0., 0.);
        let r: f32x4 = transmute(vreinterpretq_f32_p8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vreinterpretq_f32_p16() {
        let a: i16x8 = i16x8::new(0, 0, 0, 0, 0, 0, 0, 0);
        let e: f32x4 = f32x4::new(0., 0., 0., 0.);
        let r: f32x4 = transmute(vreinterpretq_f32_p16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshl_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: i8x8 = i8x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let r: i8x8 = transmute(vrshl_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshlq_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let e: i8x16 = i8x16::new(4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64);
        let r: i8x16 = transmute(vrshlq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshl_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let e: i16x4 = i16x4::new(4, 8, 12, 16);
        let r: i16x4 = transmute(vrshl_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshlq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: i16x8 = i16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let r: i16x8 = transmute(vrshlq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshl_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(2, 2);
        let e: i32x2 = i32x2::new(4, 8);
        let r: i32x2 = transmute(vrshl_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshlq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let e: i32x4 = i32x4::new(4, 8, 12, 16);
        let r: i32x4 = transmute(vrshlq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshl_s64() {
        let a: i64x1 = i64x1::new(1);
        let b: i64x1 = i64x1::new(2);
        let e: i64x1 = i64x1::new(4);
        let r: i64x1 = transmute(vrshl_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshlq_s64() {
        let a: i64x2 = i64x2::new(1, 2);
        let b: i64x2 = i64x2::new(2, 2);
        let e: i64x2 = i64x2::new(4, 8);
        let r: i64x2 = transmute(vrshlq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshl_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: u8x8 = u8x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let r: u8x8 = transmute(vrshl_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshlq_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let e: u8x16 = u8x16::new(4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64);
        let r: u8x16 = transmute(vrshlq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshl_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let e: u16x4 = u16x4::new(4, 8, 12, 16);
        let r: u16x4 = transmute(vrshl_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshlq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: u16x8 = u16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let r: u16x8 = transmute(vrshlq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshl_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: i32x2 = i32x2::new(2, 2);
        let e: u32x2 = u32x2::new(4, 8);
        let r: u32x2 = transmute(vrshl_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshlq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let e: u32x4 = u32x4::new(4, 8, 12, 16);
        let r: u32x4 = transmute(vrshlq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshl_u64() {
        let a: u64x1 = u64x1::new(1);
        let b: i64x1 = i64x1::new(2);
        let e: u64x1 = u64x1::new(4);
        let r: u64x1 = transmute(vrshl_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshlq_u64() {
        let a: u64x2 = u64x2::new(1, 2);
        let b: i64x2 = i64x2::new(2, 2);
        let e: u64x2 = u64x2::new(4, 8);
        let r: u64x2 = transmute(vrshlq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshr_n_s8() {
        let a: i8x8 = i8x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i8x8 = transmute(vrshr_n_s8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrq_n_s8() {
        let a: i8x16 = i8x16::new(4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64);
        let e: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r: i8x16 = transmute(vrshrq_n_s8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshr_n_s16() {
        let a: i16x4 = i16x4::new(4, 8, 12, 16);
        let e: i16x4 = i16x4::new(1, 2, 3, 4);
        let r: i16x4 = transmute(vrshr_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrq_n_s16() {
        let a: i16x8 = i16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i16x8 = transmute(vrshrq_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshr_n_s32() {
        let a: i32x2 = i32x2::new(4, 8);
        let e: i32x2 = i32x2::new(1, 2);
        let r: i32x2 = transmute(vrshr_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrq_n_s32() {
        let a: i32x4 = i32x4::new(4, 8, 12, 16);
        let e: i32x4 = i32x4::new(1, 2, 3, 4);
        let r: i32x4 = transmute(vrshrq_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshr_n_s64() {
        let a: i64x1 = i64x1::new(4);
        let e: i64x1 = i64x1::new(1);
        let r: i64x1 = transmute(vrshr_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrq_n_s64() {
        let a: i64x2 = i64x2::new(4, 8);
        let e: i64x2 = i64x2::new(1, 2);
        let r: i64x2 = transmute(vrshrq_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshr_n_u8() {
        let a: u8x8 = u8x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u8x8 = transmute(vrshr_n_u8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrq_n_u8() {
        let a: u8x16 = u8x16::new(4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64);
        let e: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r: u8x16 = transmute(vrshrq_n_u8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshr_n_u16() {
        let a: u16x4 = u16x4::new(4, 8, 12, 16);
        let e: u16x4 = u16x4::new(1, 2, 3, 4);
        let r: u16x4 = transmute(vrshr_n_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrq_n_u16() {
        let a: u16x8 = u16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u16x8 = transmute(vrshrq_n_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshr_n_u32() {
        let a: u32x2 = u32x2::new(4, 8);
        let e: u32x2 = u32x2::new(1, 2);
        let r: u32x2 = transmute(vrshr_n_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrq_n_u32() {
        let a: u32x4 = u32x4::new(4, 8, 12, 16);
        let e: u32x4 = u32x4::new(1, 2, 3, 4);
        let r: u32x4 = transmute(vrshrq_n_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshr_n_u64() {
        let a: u64x1 = u64x1::new(4);
        let e: u64x1 = u64x1::new(1);
        let r: u64x1 = transmute(vrshr_n_u64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrq_n_u64() {
        let a: u64x2 = u64x2::new(4, 8);
        let e: u64x2 = u64x2::new(1, 2);
        let r: u64x2 = transmute(vrshrq_n_u64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrn_n_s16() {
        let a: i16x8 = i16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i8x8 = transmute(vrshrn_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrn_n_s32() {
        let a: i32x4 = i32x4::new(4, 8, 12, 16);
        let e: i16x4 = i16x4::new(1, 2, 3, 4);
        let r: i16x4 = transmute(vrshrn_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrn_n_s64() {
        let a: i64x2 = i64x2::new(4, 8);
        let e: i32x2 = i32x2::new(1, 2);
        let r: i32x2 = transmute(vrshrn_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrn_n_u16() {
        let a: u16x8 = u16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u8x8 = transmute(vrshrn_n_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrn_n_u32() {
        let a: u32x4 = u32x4::new(4, 8, 12, 16);
        let e: u16x4 = u16x4::new(1, 2, 3, 4);
        let r: u16x4 = transmute(vrshrn_n_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrshrn_n_u64() {
        let a: u64x2 = u64x2::new(4, 8);
        let e: u32x2 = u32x2::new(1, 2);
        let r: u32x2 = transmute(vrshrn_n_u64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsra_n_s8() {
        let a: i8x8 = i8x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let b: i8x8 = i8x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: i8x8 = i8x8::new(2, 3, 4, 5, 6, 7, 8, 9);
        let r: i8x8 = transmute(vrsra_n_s8::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsraq_n_s8() {
        let a: i8x16 = i8x16::new(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let b: i8x16 = i8x16::new(4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64);
        let e: i8x16 = i8x16::new(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17);
        let r: i8x16 = transmute(vrsraq_n_s8::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsra_n_s16() {
        let a: i16x4 = i16x4::new(1, 1, 1, 1);
        let b: i16x4 = i16x4::new(4, 8, 12, 16);
        let e: i16x4 = i16x4::new(2, 3, 4, 5);
        let r: i16x4 = transmute(vrsra_n_s16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsraq_n_s16() {
        let a: i16x8 = i16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let b: i16x8 = i16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: i16x8 = i16x8::new(2, 3, 4, 5, 6, 7, 8, 9);
        let r: i16x8 = transmute(vrsraq_n_s16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsra_n_s32() {
        let a: i32x2 = i32x2::new(1, 1);
        let b: i32x2 = i32x2::new(4, 8);
        let e: i32x2 = i32x2::new(2, 3);
        let r: i32x2 = transmute(vrsra_n_s32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsraq_n_s32() {
        let a: i32x4 = i32x4::new(1, 1, 1, 1);
        let b: i32x4 = i32x4::new(4, 8, 12, 16);
        let e: i32x4 = i32x4::new(2, 3, 4, 5);
        let r: i32x4 = transmute(vrsraq_n_s32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsra_n_s64() {
        let a: i64x1 = i64x1::new(1);
        let b: i64x1 = i64x1::new(4);
        let e: i64x1 = i64x1::new(2);
        let r: i64x1 = transmute(vrsra_n_s64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsraq_n_s64() {
        let a: i64x2 = i64x2::new(1, 1);
        let b: i64x2 = i64x2::new(4, 8);
        let e: i64x2 = i64x2::new(2, 3);
        let r: i64x2 = transmute(vrsraq_n_s64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsra_n_u8() {
        let a: u8x8 = u8x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let b: u8x8 = u8x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: u8x8 = u8x8::new(2, 3, 4, 5, 6, 7, 8, 9);
        let r: u8x8 = transmute(vrsra_n_u8::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsraq_n_u8() {
        let a: u8x16 = u8x16::new(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let b: u8x16 = u8x16::new(4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64);
        let e: u8x16 = u8x16::new(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17);
        let r: u8x16 = transmute(vrsraq_n_u8::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsra_n_u16() {
        let a: u16x4 = u16x4::new(1, 1, 1, 1);
        let b: u16x4 = u16x4::new(4, 8, 12, 16);
        let e: u16x4 = u16x4::new(2, 3, 4, 5);
        let r: u16x4 = transmute(vrsra_n_u16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsraq_n_u16() {
        let a: u16x8 = u16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let b: u16x8 = u16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: u16x8 = u16x8::new(2, 3, 4, 5, 6, 7, 8, 9);
        let r: u16x8 = transmute(vrsraq_n_u16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsra_n_u32() {
        let a: u32x2 = u32x2::new(1, 1);
        let b: u32x2 = u32x2::new(4, 8);
        let e: u32x2 = u32x2::new(2, 3);
        let r: u32x2 = transmute(vrsra_n_u32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsraq_n_u32() {
        let a: u32x4 = u32x4::new(1, 1, 1, 1);
        let b: u32x4 = u32x4::new(4, 8, 12, 16);
        let e: u32x4 = u32x4::new(2, 3, 4, 5);
        let r: u32x4 = transmute(vrsraq_n_u32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsra_n_u64() {
        let a: u64x1 = u64x1::new(1);
        let b: u64x1 = u64x1::new(4);
        let e: u64x1 = u64x1::new(2);
        let r: u64x1 = transmute(vrsra_n_u64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vrsraq_n_u64() {
        let a: u64x2 = u64x2::new(1, 1);
        let b: u64x2 = u64x2::new(4, 8);
        let e: u64x2 = u64x2::new(2, 3);
        let r: u64x2 = transmute(vrsraq_n_u64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vset_lane_s8() {
        let a: i8 = 1;
        let b: i8x8 = i8x8::new(0, 2, 3, 4, 5, 6, 7, 8);
        let e: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i8x8 = transmute(vset_lane_s8::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vset_lane_s16() {
        let a: i16 = 1;
        let b: i16x4 = i16x4::new(0, 2, 3, 4);
        let e: i16x4 = i16x4::new(1, 2, 3, 4);
        let r: i16x4 = transmute(vset_lane_s16::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vset_lane_s32() {
        let a: i32 = 1;
        let b: i32x2 = i32x2::new(0, 2);
        let e: i32x2 = i32x2::new(1, 2);
        let r: i32x2 = transmute(vset_lane_s32::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vset_lane_s64() {
        let a: i64 = 1;
        let b: i64x1 = i64x1::new(0);
        let e: i64x1 = i64x1::new(1);
        let r: i64x1 = transmute(vset_lane_s64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vset_lane_u8() {
        let a: u8 = 1;
        let b: u8x8 = u8x8::new(0, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u8x8 = transmute(vset_lane_u8::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vset_lane_u16() {
        let a: u16 = 1;
        let b: u16x4 = u16x4::new(0, 2, 3, 4);
        let e: u16x4 = u16x4::new(1, 2, 3, 4);
        let r: u16x4 = transmute(vset_lane_u16::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vset_lane_u32() {
        let a: u32 = 1;
        let b: u32x2 = u32x2::new(0, 2);
        let e: u32x2 = u32x2::new(1, 2);
        let r: u32x2 = transmute(vset_lane_u32::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vset_lane_u64() {
        let a: u64 = 1;
        let b: u64x1 = u64x1::new(0);
        let e: u64x1 = u64x1::new(1);
        let r: u64x1 = transmute(vset_lane_u64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vset_lane_p8() {
        let a: p8 = 1;
        let b: i8x8 = i8x8::new(0, 2, 3, 4, 5, 6, 7, 8);
        let e: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i8x8 = transmute(vset_lane_p8::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vset_lane_p16() {
        let a: p16 = 1;
        let b: i16x4 = i16x4::new(0, 2, 3, 4);
        let e: i16x4 = i16x4::new(1, 2, 3, 4);
        let r: i16x4 = transmute(vset_lane_p16::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vset_lane_p64() {
        let a: p64 = 1;
        let b: i64x1 = i64x1::new(0);
        let e: i64x1 = i64x1::new(1);
        let r: i64x1 = transmute(vset_lane_p64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsetq_lane_s8() {
        let a: i8 = 1;
        let b: i8x16 = i8x16::new(0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r: i8x16 = transmute(vsetq_lane_s8::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsetq_lane_s16() {
        let a: i16 = 1;
        let b: i16x8 = i16x8::new(0, 2, 3, 4, 5, 6, 7, 8);
        let e: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i16x8 = transmute(vsetq_lane_s16::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsetq_lane_s32() {
        let a: i32 = 1;
        let b: i32x4 = i32x4::new(0, 2, 3, 4);
        let e: i32x4 = i32x4::new(1, 2, 3, 4);
        let r: i32x4 = transmute(vsetq_lane_s32::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsetq_lane_s64() {
        let a: i64 = 1;
        let b: i64x2 = i64x2::new(0, 2);
        let e: i64x2 = i64x2::new(1, 2);
        let r: i64x2 = transmute(vsetq_lane_s64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsetq_lane_u8() {
        let a: u8 = 1;
        let b: u8x16 = u8x16::new(0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r: u8x16 = transmute(vsetq_lane_u8::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsetq_lane_u16() {
        let a: u16 = 1;
        let b: u16x8 = u16x8::new(0, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u16x8 = transmute(vsetq_lane_u16::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsetq_lane_u32() {
        let a: u32 = 1;
        let b: u32x4 = u32x4::new(0, 2, 3, 4);
        let e: u32x4 = u32x4::new(1, 2, 3, 4);
        let r: u32x4 = transmute(vsetq_lane_u32::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsetq_lane_u64() {
        let a: u64 = 1;
        let b: u64x2 = u64x2::new(0, 2);
        let e: u64x2 = u64x2::new(1, 2);
        let r: u64x2 = transmute(vsetq_lane_u64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsetq_lane_p8() {
        let a: p8 = 1;
        let b: i8x16 = i8x16::new(0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r: i8x16 = transmute(vsetq_lane_p8::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsetq_lane_p16() {
        let a: p16 = 1;
        let b: i16x8 = i16x8::new(0, 2, 3, 4, 5, 6, 7, 8);
        let e: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i16x8 = transmute(vsetq_lane_p16::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsetq_lane_p64() {
        let a: p64 = 1;
        let b: i64x2 = i64x2::new(0, 2);
        let e: i64x2 = i64x2::new(1, 2);
        let r: i64x2 = transmute(vsetq_lane_p64::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vset_lane_f32() {
        let a: f32 = 1.;
        let b: f32x2 = f32x2::new(0., 2.);
        let e: f32x2 = f32x2::new(1., 2.);
        let r: f32x2 = transmute(vset_lane_f32::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsetq_lane_f32() {
        let a: f32 = 1.;
        let b: f32x4 = f32x4::new(0., 2., 3., 4.);
        let e: f32x4 = f32x4::new(1., 2., 3., 4.);
        let r: f32x4 = transmute(vsetq_lane_f32::<0>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshl_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: i8x8 = i8x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let r: i8x8 = transmute(vshl_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshlq_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let e: i8x16 = i8x16::new(4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64);
        let r: i8x16 = transmute(vshlq_s8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshl_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let e: i16x4 = i16x4::new(4, 8, 12, 16);
        let r: i16x4 = transmute(vshl_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshlq_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: i16x8 = i16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let r: i16x8 = transmute(vshlq_s16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshl_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let b: i32x2 = i32x2::new(2, 2);
        let e: i32x2 = i32x2::new(4, 8);
        let r: i32x2 = transmute(vshl_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshlq_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let e: i32x4 = i32x4::new(4, 8, 12, 16);
        let r: i32x4 = transmute(vshlq_s32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshl_s64() {
        let a: i64x1 = i64x1::new(1);
        let b: i64x1 = i64x1::new(2);
        let e: i64x1 = i64x1::new(4);
        let r: i64x1 = transmute(vshl_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshlq_s64() {
        let a: i64x2 = i64x2::new(1, 2);
        let b: i64x2 = i64x2::new(2, 2);
        let e: i64x2 = i64x2::new(4, 8);
        let r: i64x2 = transmute(vshlq_s64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshl_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: u8x8 = u8x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let r: u8x8 = transmute(vshl_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshlq_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let b: i8x16 = i8x16::new(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2);
        let e: u8x16 = u8x16::new(4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64);
        let r: u8x16 = transmute(vshlq_u8(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshl_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(2, 2, 2, 2);
        let e: u16x4 = u16x4::new(4, 8, 12, 16);
        let r: u16x4 = transmute(vshl_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshlq_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i16x8 = i16x8::new(2, 2, 2, 2, 2, 2, 2, 2);
        let e: u16x8 = u16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let r: u16x8 = transmute(vshlq_u16(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshl_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let b: i32x2 = i32x2::new(2, 2);
        let e: u32x2 = u32x2::new(4, 8);
        let r: u32x2 = transmute(vshl_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshlq_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: i32x4 = i32x4::new(2, 2, 2, 2);
        let e: u32x4 = u32x4::new(4, 8, 12, 16);
        let r: u32x4 = transmute(vshlq_u32(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshl_u64() {
        let a: u64x1 = u64x1::new(1);
        let b: i64x1 = i64x1::new(2);
        let e: u64x1 = u64x1::new(4);
        let r: u64x1 = transmute(vshl_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshlq_u64() {
        let a: u64x2 = u64x2::new(1, 2);
        let b: i64x2 = i64x2::new(2, 2);
        let e: u64x2 = u64x2::new(4, 8);
        let r: u64x2 = transmute(vshlq_u64(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshl_n_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i8x8 = i8x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let r: i8x8 = transmute(vshl_n_s8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshlq_n_s8() {
        let a: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: i8x16 = i8x16::new(4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64);
        let r: i8x16 = transmute(vshlq_n_s8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshl_n_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let e: i16x4 = i16x4::new(4, 8, 12, 16);
        let r: i16x4 = transmute(vshl_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshlq_n_s16() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i16x8 = i16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let r: i16x8 = transmute(vshlq_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshl_n_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let e: i32x2 = i32x2::new(4, 8);
        let r: i32x2 = transmute(vshl_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshlq_n_s32() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let e: i32x4 = i32x4::new(4, 8, 12, 16);
        let r: i32x4 = transmute(vshlq_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshl_n_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u8x8 = u8x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let r: u8x8 = transmute(vshl_n_u8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshlq_n_u8() {
        let a: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let e: u8x16 = u8x16::new(4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64);
        let r: u8x16 = transmute(vshlq_n_u8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshl_n_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let e: u16x4 = u16x4::new(4, 8, 12, 16);
        let r: u16x4 = transmute(vshl_n_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshlq_n_u16() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let r: u16x8 = transmute(vshlq_n_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshl_n_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let e: u32x2 = u32x2::new(4, 8);
        let r: u32x2 = transmute(vshl_n_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshlq_n_u32() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(4, 8, 12, 16);
        let r: u32x4 = transmute(vshlq_n_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshl_n_s64() {
        let a: i64x1 = i64x1::new(1);
        let e: i64x1 = i64x1::new(4);
        let r: i64x1 = transmute(vshl_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshlq_n_s64() {
        let a: i64x2 = i64x2::new(1, 2);
        let e: i64x2 = i64x2::new(4, 8);
        let r: i64x2 = transmute(vshlq_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshl_n_u64() {
        let a: u64x1 = u64x1::new(1);
        let e: u64x1 = u64x1::new(4);
        let r: u64x1 = transmute(vshl_n_u64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshlq_n_u64() {
        let a: u64x2 = u64x2::new(1, 2);
        let e: u64x2 = u64x2::new(4, 8);
        let r: u64x2 = transmute(vshlq_n_u64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshll_n_s8() {
        let a: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: i16x8 = i16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let r: i16x8 = transmute(vshll_n_s8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshll_n_s16() {
        let a: i16x4 = i16x4::new(1, 2, 3, 4);
        let e: i32x4 = i32x4::new(4, 8, 12, 16);
        let r: i32x4 = transmute(vshll_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshll_n_s32() {
        let a: i32x2 = i32x2::new(1, 2);
        let e: i64x2 = i64x2::new(4, 8);
        let r: i64x2 = transmute(vshll_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshll_n_u8() {
        let a: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let e: u16x8 = u16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let r: u16x8 = transmute(vshll_n_u8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshll_n_u16() {
        let a: u16x4 = u16x4::new(1, 2, 3, 4);
        let e: u32x4 = u32x4::new(4, 8, 12, 16);
        let r: u32x4 = transmute(vshll_n_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshll_n_u32() {
        let a: u32x2 = u32x2::new(1, 2);
        let e: u64x2 = u64x2::new(4, 8);
        let r: u64x2 = transmute(vshll_n_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshr_n_s8() {
        let a: i8x8 = i8x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i8x8 = transmute(vshr_n_s8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrq_n_s8() {
        let a: i8x16 = i8x16::new(4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64);
        let e: i8x16 = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r: i8x16 = transmute(vshrq_n_s8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshr_n_s16() {
        let a: i16x4 = i16x4::new(4, 8, 12, 16);
        let e: i16x4 = i16x4::new(1, 2, 3, 4);
        let r: i16x4 = transmute(vshr_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrq_n_s16() {
        let a: i16x8 = i16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i16x8 = transmute(vshrq_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshr_n_s32() {
        let a: i32x2 = i32x2::new(4, 8);
        let e: i32x2 = i32x2::new(1, 2);
        let r: i32x2 = transmute(vshr_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrq_n_s32() {
        let a: i32x4 = i32x4::new(4, 8, 12, 16);
        let e: i32x4 = i32x4::new(1, 2, 3, 4);
        let r: i32x4 = transmute(vshrq_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshr_n_s64() {
        let a: i64x1 = i64x1::new(4);
        let e: i64x1 = i64x1::new(1);
        let r: i64x1 = transmute(vshr_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrq_n_s64() {
        let a: i64x2 = i64x2::new(4, 8);
        let e: i64x2 = i64x2::new(1, 2);
        let r: i64x2 = transmute(vshrq_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshr_n_u8() {
        let a: u8x8 = u8x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u8x8 = transmute(vshr_n_u8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrq_n_u8() {
        let a: u8x16 = u8x16::new(4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64);
        let e: u8x16 = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);
        let r: u8x16 = transmute(vshrq_n_u8::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshr_n_u16() {
        let a: u16x4 = u16x4::new(4, 8, 12, 16);
        let e: u16x4 = u16x4::new(1, 2, 3, 4);
        let r: u16x4 = transmute(vshr_n_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrq_n_u16() {
        let a: u16x8 = u16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u16x8 = transmute(vshrq_n_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshr_n_u32() {
        let a: u32x2 = u32x2::new(4, 8);
        let e: u32x2 = u32x2::new(1, 2);
        let r: u32x2 = transmute(vshr_n_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrq_n_u32() {
        let a: u32x4 = u32x4::new(4, 8, 12, 16);
        let e: u32x4 = u32x4::new(1, 2, 3, 4);
        let r: u32x4 = transmute(vshrq_n_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshr_n_u64() {
        let a: u64x1 = u64x1::new(4);
        let e: u64x1 = u64x1::new(1);
        let r: u64x1 = transmute(vshr_n_u64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrq_n_u64() {
        let a: u64x2 = u64x2::new(4, 8);
        let e: u64x2 = u64x2::new(1, 2);
        let r: u64x2 = transmute(vshrq_n_u64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrn_n_s16() {
        let a: i16x8 = i16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: i8x8 = transmute(vshrn_n_s16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrn_n_s32() {
        let a: i32x4 = i32x4::new(4, 8, 12, 16);
        let e: i16x4 = i16x4::new(1, 2, 3, 4);
        let r: i16x4 = transmute(vshrn_n_s32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrn_n_s64() {
        let a: i64x2 = i64x2::new(4, 8);
        let e: i32x2 = i32x2::new(1, 2);
        let r: i32x2 = transmute(vshrn_n_s64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrn_n_u16() {
        let a: u16x8 = u16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let r: u8x8 = transmute(vshrn_n_u16::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrn_n_u32() {
        let a: u32x4 = u32x4::new(4, 8, 12, 16);
        let e: u16x4 = u16x4::new(1, 2, 3, 4);
        let r: u16x4 = transmute(vshrn_n_u32::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vshrn_n_u64() {
        let a: u64x2 = u64x2::new(4, 8);
        let e: u32x2 = u32x2::new(1, 2);
        let r: u32x2 = transmute(vshrn_n_u64::<2>(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsra_n_s8() {
        let a: i8x8 = i8x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let b: i8x8 = i8x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: i8x8 = i8x8::new(2, 3, 4, 5, 6, 7, 8, 9);
        let r: i8x8 = transmute(vsra_n_s8::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsraq_n_s8() {
        let a: i8x16 = i8x16::new(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let b: i8x16 = i8x16::new(4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64);
        let e: i8x16 = i8x16::new(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17);
        let r: i8x16 = transmute(vsraq_n_s8::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsra_n_s16() {
        let a: i16x4 = i16x4::new(1, 1, 1, 1);
        let b: i16x4 = i16x4::new(4, 8, 12, 16);
        let e: i16x4 = i16x4::new(2, 3, 4, 5);
        let r: i16x4 = transmute(vsra_n_s16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsraq_n_s16() {
        let a: i16x8 = i16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let b: i16x8 = i16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: i16x8 = i16x8::new(2, 3, 4, 5, 6, 7, 8, 9);
        let r: i16x8 = transmute(vsraq_n_s16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsra_n_s32() {
        let a: i32x2 = i32x2::new(1, 1);
        let b: i32x2 = i32x2::new(4, 8);
        let e: i32x2 = i32x2::new(2, 3);
        let r: i32x2 = transmute(vsra_n_s32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsraq_n_s32() {
        let a: i32x4 = i32x4::new(1, 1, 1, 1);
        let b: i32x4 = i32x4::new(4, 8, 12, 16);
        let e: i32x4 = i32x4::new(2, 3, 4, 5);
        let r: i32x4 = transmute(vsraq_n_s32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsra_n_s64() {
        let a: i64x1 = i64x1::new(1);
        let b: i64x1 = i64x1::new(4);
        let e: i64x1 = i64x1::new(2);
        let r: i64x1 = transmute(vsra_n_s64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsraq_n_s64() {
        let a: i64x2 = i64x2::new(1, 1);
        let b: i64x2 = i64x2::new(4, 8);
        let e: i64x2 = i64x2::new(2, 3);
        let r: i64x2 = transmute(vsraq_n_s64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsra_n_u8() {
        let a: u8x8 = u8x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let b: u8x8 = u8x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: u8x8 = u8x8::new(2, 3, 4, 5, 6, 7, 8, 9);
        let r: u8x8 = transmute(vsra_n_u8::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsraq_n_u8() {
        let a: u8x16 = u8x16::new(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        let b: u8x16 = u8x16::new(4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64);
        let e: u8x16 = u8x16::new(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17);
        let r: u8x16 = transmute(vsraq_n_u8::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsra_n_u16() {
        let a: u16x4 = u16x4::new(1, 1, 1, 1);
        let b: u16x4 = u16x4::new(4, 8, 12, 16);
        let e: u16x4 = u16x4::new(2, 3, 4, 5);
        let r: u16x4 = transmute(vsra_n_u16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsraq_n_u16() {
        let a: u16x8 = u16x8::new(1, 1, 1, 1, 1, 1, 1, 1);
        let b: u16x8 = u16x8::new(4, 8, 12, 16, 20, 24, 28, 32);
        let e: u16x8 = u16x8::new(2, 3, 4, 5, 6, 7, 8, 9);
        let r: u16x8 = transmute(vsraq_n_u16::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsra_n_u32() {
        let a: u32x2 = u32x2::new(1, 1);
        let b: u32x2 = u32x2::new(4, 8);
        let e: u32x2 = u32x2::new(2, 3);
        let r: u32x2 = transmute(vsra_n_u32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsraq_n_u32() {
        let a: u32x4 = u32x4::new(1, 1, 1, 1);
        let b: u32x4 = u32x4::new(4, 8, 12, 16);
        let e: u32x4 = u32x4::new(2, 3, 4, 5);
        let r: u32x4 = transmute(vsraq_n_u32::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsra_n_u64() {
        let a: u64x1 = u64x1::new(1);
        let b: u64x1 = u64x1::new(4);
        let e: u64x1 = u64x1::new(2);
        let r: u64x1 = transmute(vsra_n_u64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vsraq_n_u64() {
        let a: u64x2 = u64x2::new(1, 1);
        let b: u64x2 = u64x2::new(4, 8);
        let e: u64x2 = u64x2::new(2, 3);
        let r: u64x2 = transmute(vsraq_n_u64::<2>(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabal_u8() {
        let a: u16x8 = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: u8x8 = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let c: u8x8 = u8x8::new(10, 10, 10, 10, 10, 10, 10, 10);
        let e: u16x8 = u16x8::new(10, 10, 10, 10, 10, 10, 10, 10);
        let r: u16x8 = transmute(vabal_u8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabal_u16() {
        let a: u32x4 = u32x4::new(1, 2, 3, 4);
        let b: u16x4 = u16x4::new(1, 2, 3, 4);
        let c: u16x4 = u16x4::new(10, 10, 10, 10);
        let e: u32x4 = u32x4::new(10, 10, 10, 10);
        let r: u32x4 = transmute(vabal_u16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabal_u32() {
        let a: u64x2 = u64x2::new(1, 2);
        let b: u32x2 = u32x2::new(1, 2);
        let c: u32x2 = u32x2::new(10, 10);
        let e: u64x2 = u64x2::new(10, 10);
        let r: u64x2 = transmute(vabal_u32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabal_s8() {
        let a: i16x8 = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let b: i8x8 = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);
        let c: i8x8 = i8x8::new(10, 10, 10, 10, 10, 10, 10, 10);
        let e: i16x8 = i16x8::new(10, 10, 10, 10, 10, 10, 10, 10);
        let r: i16x8 = transmute(vabal_s8(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabal_s16() {
        let a: i32x4 = i32x4::new(1, 2, 3, 4);
        let b: i16x4 = i16x4::new(1, 2, 3, 4);
        let c: i16x4 = i16x4::new(10, 10, 10, 10);
        let e: i32x4 = i32x4::new(10, 10, 10, 10);
        let r: i32x4 = transmute(vabal_s16(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vabal_s32() {
        let a: i64x2 = i64x2::new(1, 2);
        let b: i32x2 = i32x2::new(1, 2);
        let c: i32x2 = i32x2::new(10, 10);
        let e: i64x2 = i64x2::new(10, 10);
        let r: i64x2 = transmute(vabal_s32(transmute(a), transmute(b), transmute(c)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqabs_s8() {
        let a: i8x8 = i8x8::new(-128, 0x7F, -6, -5, -4, -3, -2, -1);
        let e: i8x8 = i8x8::new(0x7F, 0x7F, 6, 5, 4, 3, 2, 1);
        let r: i8x8 = transmute(vqabs_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqabsq_s8() {
        let a: i8x16 = i8x16::new(-128, 0x7F, -6, -5, -4, -3, -2, -1, 0, -127, 127, 1, 2, 3, 4, 5);
        let e: i8x16 = i8x16::new(0x7F, 0x7F, 6, 5, 4, 3, 2, 1, 0, 127, 127, 1, 2, 3, 4, 5);
        let r: i8x16 = transmute(vqabsq_s8(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqabs_s16() {
        let a: i16x4 = i16x4::new(-32768, 0x7F_FF, -6, -5);
        let e: i16x4 = i16x4::new(0x7F_FF, 0x7F_FF, 6, 5);
        let r: i16x4 = transmute(vqabs_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqabsq_s16() {
        let a: i16x8 = i16x8::new(-32768, 0x7F_FF, -6, -5, -4, -3, -2, -1);
        let e: i16x8 = i16x8::new(0x7F_FF, 0x7F_FF, 6, 5, 4, 3, 2, 1);
        let r: i16x8 = transmute(vqabsq_s16(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqabs_s32() {
        let a: i32x2 = i32x2::new(-2147483648, 0x7F_FF_FF_FF);
        let e: i32x2 = i32x2::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF);
        let r: i32x2 = transmute(vqabs_s32(transmute(a)));
        assert_eq!(r, e);
    }

    #[simd_test(enable = "neon")]
    unsafe fn test_vqabsq_s32() {
        let a: i32x4 = i32x4::new(-2147483648, 0x7F_FF_FF_FF, -6, -5);
        let e: i32x4 = i32x4::new(0x7F_FF_FF_FF, 0x7F_FF_FF_FF, 6, 5);
        let r: i32x4 = transmute(vqabsq_s32(transmute(a)));
        assert_eq!(r, e);
    }
}
