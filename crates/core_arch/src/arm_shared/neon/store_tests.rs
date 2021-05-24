//! Tests for ARM+v7+neon store (vst1) intrinsics.
//!
//! These are included in `{arm, aarch64}::neon`.

use super::*;

#[cfg(target_arch = "arm")]
use crate::core_arch::arm::*;

#[cfg(target_arch = "aarch64")]
use crate::core_arch::aarch64::*;

use crate::core_arch::simd::*;
use stdarch_test::simd_test;

#[simd_test(enable = "neon")]
unsafe fn test_vst1_s8() {
    let mut vals = [0_i8; 9];
    let a = i8x8::new(1, 2, 3, 4, 5, 6, 7, 8);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1_s8(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
    assert_eq!(vals[ofs + 1], 2);
    assert_eq!(vals[ofs + 2], 3);
    assert_eq!(vals[ofs + 3], 4);
    assert_eq!(vals[ofs + 4], 5);
    assert_eq!(vals[ofs + 5], 6);
    assert_eq!(vals[ofs + 6], 7);
    assert_eq!(vals[ofs + 7], 8);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_s8() {
    let mut vals = [0_i8; 17];
    let a = i8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1q_s8(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
    assert_eq!(vals[ofs + 1], 2);
    assert_eq!(vals[ofs + 2], 3);
    assert_eq!(vals[ofs + 3], 4);
    assert_eq!(vals[ofs + 4], 5);
    assert_eq!(vals[ofs + 5], 6);
    assert_eq!(vals[ofs + 6], 7);
    assert_eq!(vals[ofs + 7], 8);
    assert_eq!(vals[ofs + 8], 9);
    assert_eq!(vals[ofs + 9], 10);
    assert_eq!(vals[ofs + 10], 11);
    assert_eq!(vals[ofs + 11], 12);
    assert_eq!(vals[ofs + 12], 13);
    assert_eq!(vals[ofs + 13], 14);
    assert_eq!(vals[ofs + 14], 15);
    assert_eq!(vals[ofs + 15], 16);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_s16() {
    let mut vals = [0_i16; 5];
    let a = i16x4::new(1, 2, 3, 4);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1_s16(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
    assert_eq!(vals[ofs + 1], 2);
    assert_eq!(vals[ofs + 2], 3);
    assert_eq!(vals[ofs + 3], 4);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_s16() {
    let mut vals = [0_i16; 9];
    let a = i16x8::new(1, 2, 3, 4, 5, 6, 7, 8);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1q_s16(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
    assert_eq!(vals[ofs + 1], 2);
    assert_eq!(vals[ofs + 2], 3);
    assert_eq!(vals[ofs + 3], 4);
    assert_eq!(vals[ofs + 4], 5);
    assert_eq!(vals[ofs + 5], 6);
    assert_eq!(vals[ofs + 6], 7);
    assert_eq!(vals[ofs + 7], 8);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_s32() {
    let mut vals = [0_i32; 3];
    let a = i32x2::new(1, 2);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1_s32(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
    assert_eq!(vals[ofs + 1], 2);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_s32() {
    let mut vals = [0_i32; 5];
    let a = i32x4::new(1, 2, 3, 4);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1q_s32(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
    assert_eq!(vals[ofs + 1], 2);
    assert_eq!(vals[ofs + 2], 3);
    assert_eq!(vals[ofs + 3], 4);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_s64() {
    let mut vals = [0_i64; 2];
    let a = i64x1::new(1);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1_s64(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_s64() {
    let mut vals = [0_i64; 3];
    let a = i64x2::new(1, 2);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1q_s64(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
    assert_eq!(vals[ofs + 1], 2);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_u8() {
    let mut vals = [0_u8; 9];
    let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1_u8(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
    assert_eq!(vals[ofs + 1], 2);
    assert_eq!(vals[ofs + 2], 3);
    assert_eq!(vals[ofs + 3], 4);
    assert_eq!(vals[ofs + 4], 5);
    assert_eq!(vals[ofs + 5], 6);
    assert_eq!(vals[ofs + 6], 7);
    assert_eq!(vals[ofs + 7], 8);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_u8() {
    let mut vals = [0_u8; 17];
    let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1q_u8(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
    assert_eq!(vals[ofs + 1], 2);
    assert_eq!(vals[ofs + 2], 3);
    assert_eq!(vals[ofs + 3], 4);
    assert_eq!(vals[ofs + 4], 5);
    assert_eq!(vals[ofs + 5], 6);
    assert_eq!(vals[ofs + 6], 7);
    assert_eq!(vals[ofs + 7], 8);
    assert_eq!(vals[ofs + 8], 9);
    assert_eq!(vals[ofs + 9], 10);
    assert_eq!(vals[ofs + 10], 11);
    assert_eq!(vals[ofs + 11], 12);
    assert_eq!(vals[ofs + 12], 13);
    assert_eq!(vals[ofs + 13], 14);
    assert_eq!(vals[ofs + 14], 15);
    assert_eq!(vals[ofs + 15], 16);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_u16() {
    let mut vals = [0_u16; 5];
    let a = u16x4::new(1, 2, 3, 4);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1_u16(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
    assert_eq!(vals[ofs + 1], 2);
    assert_eq!(vals[ofs + 2], 3);
    assert_eq!(vals[ofs + 3], 4);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_u16() {
    let mut vals = [0_u16; 9];
    let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1q_u16(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
    assert_eq!(vals[ofs + 1], 2);
    assert_eq!(vals[ofs + 2], 3);
    assert_eq!(vals[ofs + 3], 4);
    assert_eq!(vals[ofs + 4], 5);
    assert_eq!(vals[ofs + 5], 6);
    assert_eq!(vals[ofs + 6], 7);
    assert_eq!(vals[ofs + 7], 8);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_u32() {
    let mut vals = [0_u32; 3];
    let a = u32x2::new(1, 2);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1_u32(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
    assert_eq!(vals[ofs + 1], 2);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_u32() {
    let mut vals = [0_u32; 5];
    let a = u32x4::new(1, 2, 3, 4);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1q_u32(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
    assert_eq!(vals[ofs + 1], 2);
    assert_eq!(vals[ofs + 2], 3);
    assert_eq!(vals[ofs + 3], 4);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_u64() {
    let mut vals = [0_u64; 2];
    let a = u64x1::new(1);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1_u64(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_u64() {
    let mut vals = [0_u64; 3];
    let a = u64x2::new(1, 2);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1q_u64(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
    assert_eq!(vals[ofs + 1], 2);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_p8() {
    let mut vals = [0_u8; 9];
    let a = u8x8::new(1, 2, 3, 4, 5, 6, 7, 8);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1_p8(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
    assert_eq!(vals[ofs + 1], 2);
    assert_eq!(vals[ofs + 2], 3);
    assert_eq!(vals[ofs + 3], 4);
    assert_eq!(vals[ofs + 4], 5);
    assert_eq!(vals[ofs + 5], 6);
    assert_eq!(vals[ofs + 6], 7);
    assert_eq!(vals[ofs + 7], 8);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_p8() {
    let mut vals = [0_u8; 17];
    let a = u8x16::new(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1q_p8(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
    assert_eq!(vals[ofs + 1], 2);
    assert_eq!(vals[ofs + 2], 3);
    assert_eq!(vals[ofs + 3], 4);
    assert_eq!(vals[ofs + 4], 5);
    assert_eq!(vals[ofs + 5], 6);
    assert_eq!(vals[ofs + 6], 7);
    assert_eq!(vals[ofs + 7], 8);
    assert_eq!(vals[ofs + 8], 9);
    assert_eq!(vals[ofs + 9], 10);
    assert_eq!(vals[ofs + 10], 11);
    assert_eq!(vals[ofs + 11], 12);
    assert_eq!(vals[ofs + 12], 13);
    assert_eq!(vals[ofs + 13], 14);
    assert_eq!(vals[ofs + 14], 15);
    assert_eq!(vals[ofs + 15], 16);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_f32() {
    let mut vals = [0_f32; 3];
    let a = f32x2::new(1., 2.);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1_f32(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0.);
    }
    assert_eq!(vals[ofs + 0], 1.);
    assert_eq!(vals[ofs + 1], 2.);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1_p16() {
    let mut vals = [0_u16; 5];
    let a = u16x4::new(1, 2, 3, 4);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1_p16(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
    assert_eq!(vals[ofs + 1], 2);
    assert_eq!(vals[ofs + 2], 3);
    assert_eq!(vals[ofs + 3], 4);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_p16() {
    let mut vals = [0_u16; 9];
    let a = u16x8::new(1, 2, 3, 4, 5, 6, 7, 8);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1q_p16(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0);
    }
    assert_eq!(vals[ofs + 0], 1);
    assert_eq!(vals[ofs + 1], 2);
    assert_eq!(vals[ofs + 2], 3);
    assert_eq!(vals[ofs + 3], 4);
    assert_eq!(vals[ofs + 4], 5);
    assert_eq!(vals[ofs + 5], 6);
    assert_eq!(vals[ofs + 6], 7);
    assert_eq!(vals[ofs + 7], 8);
}

#[simd_test(enable = "neon")]
unsafe fn test_vst1q_f32() {
    let mut vals = [0_f32; 5];
    let a = f32x4::new(1., 2., 3., 4.);

    let mut ofs = 0;
    let mut p = vals.as_mut_ptr();

    // Make sure p is **not** aligned to 16-byte boundary
    if (p as usize) & 0xf == 0 {
        ofs = 1;
        p = p.offset(1);
    }

    vst1q_f32(p, transmute(a));

    if ofs > 0 {
        assert_eq!(vals[ofs - 1], 0.);
    }
    assert_eq!(vals[ofs + 0], 1.);
    assert_eq!(vals[ofs + 1], 2.);
    assert_eq!(vals[ofs + 2], 3.);
    assert_eq!(vals[ofs + 3], 4.);
}
