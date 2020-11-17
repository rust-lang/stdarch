use crate::{
    core_arch::{simd::*, simd_llvm::*, x86::*},
    mem::{self, transmute},
    ptr,
};

#[cfg(test)]
use stdarch_test::assert_instr;

/// Compute the absolute value of packed signed 16-bit integers in a, and store the unsigned results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_abs_epi16&expand=30)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpabsw))]
pub unsafe fn _mm512_abs_epi16(a: __m512i) -> __m512i {
    let a = a.as_i16x32();
    // all-0 is a properly initialized i16x32
    let zero: i16x32 = mem::zeroed();
    let sub = simd_sub(zero, a);
    let cmp: i16x32 = simd_gt(a, zero);
    transmute(simd_select(cmp, a, sub))
}

/// Compute the absolute value of packed signed 16-bit integers in a, and store the unsigned results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_abs_epi16&expand=31)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpabsw))]
pub unsafe fn _mm512_mask_abs_epi16(src: __m512i, k: __mmask32, a: __m512i) -> __m512i {
    let abs = _mm512_abs_epi16(a).as_i16x32();
    transmute(simd_select_bitmask(k, abs, src.as_i16x32()))
}

/// Compute the absolute value of packed signed 16-bit integers in a, and store the unsigned results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_abs_epi16&expand=32)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpabsw))]
pub unsafe fn _mm512_maskz_abs_epi16(k: __mmask32, a: __m512i) -> __m512i {
    let abs = _mm512_abs_epi16(a).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, abs, zero))
}

/// Compute the absolute value of packed signed 8-bit integers in a, and store the unsigned results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_abs_epi8&expand=57)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpabsb))]
pub unsafe fn _mm512_abs_epi8(a: __m512i) -> __m512i {
    let a = a.as_i8x64();
    // all-0 is a properly initialized i8x64
    let zero: i8x64 = mem::zeroed();
    let sub = simd_sub(zero, a);
    let cmp: i8x64 = simd_gt(a, zero);
    transmute(simd_select(cmp, a, sub))
}

/// Compute the absolute value of packed signed 8-bit integers in a, and store the unsigned results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_abs_epi8&expand=58)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpabsb))]
pub unsafe fn _mm512_mask_abs_epi8(src: __m512i, k: __mmask64, a: __m512i) -> __m512i {
    let abs = _mm512_abs_epi8(a).as_i8x64();
    transmute(simd_select_bitmask(k, abs, src.as_i8x64()))
}

/// Compute the absolute value of packed signed 8-bit integers in a, and store the unsigned results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_abs_epi8&expand=59)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpabsb))]
pub unsafe fn _mm512_maskz_abs_epi8(k: __mmask64, a: __m512i) -> __m512i {
    let abs = _mm512_abs_epi8(a).as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, abs, zero))
}

/// Add packed 16-bit integers in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_add_epi16&expand=91)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddw))]
pub unsafe fn _mm512_add_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_add(a.as_i16x32(), b.as_i16x32()))
}

/// Add packed 16-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set). 
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_add_epi16&expand=92)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddw))]
pub unsafe fn _mm512_mask_add_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let add = _mm512_add_epi16(a, b).as_i16x32();
    transmute(simd_select_bitmask(k, add, src.as_i16x32()))
}

/// Add packed 16-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_add_epi16&expand=93)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddw))]
pub unsafe fn _mm512_maskz_add_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let add = _mm512_add_epi16(a, b).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, add, zero))
}

/// Add packed 8-bit integers in a and b, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_add_epi8&expand=118)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddb))]
pub unsafe fn _mm512_add_epi8(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_add(a.as_i8x64(), b.as_i8x64()))
}

/// Add packed 8-bit integers in a and b, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_add_epi8&expand=119)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddb))]
pub unsafe fn _mm512_mask_add_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let add = _mm512_add_epi8(a, b).as_i8x64();
    transmute(simd_select_bitmask(k, add, src.as_i8x64()))
}

/// Add packed 8-bit integers in a and b, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set). 
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_add_epi8&expand=120)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddb))]
pub unsafe fn _mm512_maskz_add_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let add = _mm512_add_epi8(a, b).as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, add, zero))
}

/// Add packed unsigned 16-bit integers in a and b using saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_adds_epu16&expand=197)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddusw))]
pub unsafe fn _mm512_adds_epu16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddusw(a.as_u16x32(), b.as_u16x32(), _mm512_setzero_si512().as_u16x32(), 0b11111111_11111111_11111111_11111111))
}

/// Add packed unsigned 16-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_adds_epu16&expand=198)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddusw))]
pub unsafe fn _mm512_mask_adds_epu16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddusw(a.as_u16x32(), b.as_u16x32(), src.as_u16x32(), k))
}

/// Add packed unsigned 16-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_adds_epu16&expand=199)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddusw))]
pub unsafe fn _mm512_maskz_adds_epu16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddusw(a.as_u16x32(), b.as_u16x32(), _mm512_setzero_si512().as_u16x32(), k))
}

/// Add packed unsigned 8-bit integers in a and b using saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_adds_epu8&expand=206)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddusb))]
pub unsafe fn _mm512_adds_epu8(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddusb(a.as_u8x64(), b.as_u8x64(), _mm512_setzero_si512().as_u8x64(), 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111))
}

/// Add packed unsigned 8-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_adds_epu8&expand=207)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddusb))]
pub unsafe fn _mm512_mask_adds_epu8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddusb(a.as_u8x64(), b.as_u8x64(), src.as_u8x64(), k))
}

/// Add packed unsigned 8-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_adds_epu8&expand=208)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddusb))]
pub unsafe fn _mm512_maskz_adds_epu8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddusb(a.as_u8x64(), b.as_u8x64(), _mm512_setzero_si512().as_u8x64(), k))
}

/// Add packed signed 16-bit integers in a and b using saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_adds_epi16&expand=179)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddsw))]
pub unsafe fn _mm512_adds_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddsw(a.as_i16x32(), b.as_i16x32(), _mm512_setzero_si512().as_i16x32(), 0b11111111_11111111_11111111_11111111))
}

/// Add packed signed 16-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_adds_epi16&expand=180)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddsw))]
pub unsafe fn _mm512_mask_adds_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddsw(a.as_i16x32(), b.as_i16x32(), src.as_i16x32(), k))
}

/// Add packed signed 16-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_adds_epi16&expand=181)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddsw))]
pub unsafe fn _mm512_maskz_adds_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddsw(a.as_i16x32(), b.as_i16x32(), _mm512_setzero_si512().as_i16x32(), k))
}

/// Add packed signed 8-bit integers in a and b using saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_adds_epi8&expand=188)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddsb))]
pub unsafe fn _mm512_adds_epi8(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddsb(a.as_i8x64(), b.as_i8x64(), _mm512_setzero_si512().as_i8x64(), 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111))
}

/// Add packed signed 8-bit integers in a and b using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_adds_epi8&expand=189)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddsb))]
pub unsafe fn _mm512_mask_adds_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddsb(a.as_i8x64(), b.as_i8x64(), src.as_i8x64(), k))
}

/// Add packed signed 8-bit integers in a and b using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_adds_epi8&expand=190)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpaddsb))]
pub unsafe fn _mm512_maskz_adds_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpaddsb(a.as_i8x64(), b.as_i8x64(), _mm512_setzero_si512().as_i8x64(), k))
}

/// Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_sub_epi16&expand=5685)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubw))]
pub unsafe fn _mm512_sub_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_sub(a.as_i16x32(), b.as_i16x32()))
}

/// Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_sub_epi16&expand=5683)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubw))]
pub unsafe fn _mm512_mask_sub_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let sub = _mm512_sub_epi16(a, b).as_i16x32();
    transmute(simd_select_bitmask(k, sub, src.as_i16x32()))
}

/// Subtract packed 16-bit integers in b from packed 16-bit integers in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_sub_epi16&expand=5684)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubw))]
pub unsafe fn _mm512_maskz_sub_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let sub = _mm512_sub_epi16(a, b).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, sub, zero))
}

/// Subtract packed 8-bit integers in b from packed 8-bit integers in a, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_sub_epi8&expand=5712)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubb))]
pub unsafe fn _mm512_sub_epi8(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_sub(a.as_i8x64(), b.as_i8x64()))
}

/// Subtract packed 8-bit integers in b from packed 8-bit integers in a, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_sub_epi8&expand=5710)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubb))]
pub unsafe fn _mm512_mask_sub_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let sub = _mm512_sub_epi8(a, b).as_i8x64();
    transmute(simd_select_bitmask(k, sub, src.as_i8x64()))
}

/// Subtract packed 8-bit integers in b from packed 8-bit integers in a, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_sub_epi8&expand=5711)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubb))]
pub unsafe fn _mm512_maskz_sub_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let sub = _mm512_sub_epi8(a, b).as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, sub, zero))
}

/// Subtract packed unsigned 16-bit integers in b from packed unsigned 16-bit integers in a using saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_subs_epu16&expand=5793)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubusw))]
pub unsafe fn _mm512_subs_epu16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubusw(a.as_u16x32(), b.as_u16x32(), _mm512_setzero_si512().as_u16x32(), 0b11111111_11111111_11111111_11111111))
}

/// Subtract packed unsigned 16-bit integers in b from packed unsigned 16-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_subs_epu16&expand=5791)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubusw))]
pub unsafe fn _mm512_mask_subs_epu16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubusw(a.as_u16x32(), b.as_u16x32(), src.as_u16x32(), k))
}

/// Subtract packed unsigned 16-bit integers in b from packed unsigned 16-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_subs_epu16&expand=5792)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubusw))]
pub unsafe fn _mm512_maskz_subs_epu16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubusw(a.as_u16x32(), b.as_u16x32(), _mm512_setzero_si512().as_u16x32(), k))
}

/// Subtract packed unsigned 8-bit integers in b from packed unsigned 8-bit integers in a using saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_subs_epu8&expand=5802)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubusb))]
pub unsafe fn _mm512_subs_epu8(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubusb(a.as_u8x64(), b.as_u8x64(), _mm512_setzero_si512().as_u8x64(), 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111))
}

/// Subtract packed unsigned 8-bit integers in b from packed unsigned 8-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_subs_epu8&expand=5800)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubusb))]
pub unsafe fn _mm512_mask_subs_epu8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubusb(a.as_u8x64(), b.as_u8x64(), src.as_u8x64(), k))
}

/// Subtract packed unsigned 8-bit integers in b from packed unsigned 8-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_subs_epu8&expand=5801)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubusb))]
pub unsafe fn _mm512_maskz_subs_epu8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubusb(a.as_u8x64(), b.as_u8x64(), _mm512_setzero_si512().as_u8x64(), k))
}

/// Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_subs_epi16&expand=5775)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubsw))]
pub unsafe fn _mm512_subs_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubsw(a.as_i16x32(), b.as_i16x32(), _mm512_setzero_si512().as_i16x32(), 0b11111111_11111111_11111111_11111111))
}

/// Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_subs_epi16&expand=5773)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubsw))]
pub unsafe fn _mm512_mask_subs_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubsw(a.as_i16x32(), b.as_i16x32(), src.as_i16x32(), k))
}

/// Subtract packed signed 16-bit integers in b from packed 16-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_subs_epi16&expand=5774)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubsw))]
pub unsafe fn _mm512_maskz_subs_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubsw(a.as_i16x32(), b.as_i16x32(), _mm512_setzero_si512().as_i16x32(), k))
}

/// Subtract packed signed 8-bit integers in b from packed 8-bit integers in a using saturation, and store the results in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_subs_epi8&expand=5784)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubsb))]
pub unsafe fn _mm512_subs_epi8(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubsb(a.as_i8x64(), b.as_i8x64(), _mm512_setzero_si512().as_i8x64(), 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111))
}

/// Subtract packed signed 8-bit integers in b from packed 8-bit integers in a using saturation, and store the results in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_subs_epi8&expand=5782)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubsb))]
pub unsafe fn _mm512_mask_subs_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubsb(a.as_i8x64(), b.as_i8x64(), src.as_i8x64(), k))
}

/// Subtract packed signed 8-bit integers in b from packed 8-bit integers in a using saturation, and store the results in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_subs_epi8&expand=5783)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpsubsb))]
pub unsafe fn _mm512_maskz_subs_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    transmute(vpsubsb(a.as_i8x64(), b.as_i8x64(), _mm512_setzero_si512().as_i8x64(), k))
}

/// Multiply the packed unsigned 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mulhi_epu16&expand=3973)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmulhuw))]
pub unsafe fn _mm512_mulhi_epu16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmulhuw(a.as_u16x32(), b.as_u16x32()))
}

/// Multiply the packed unsigned 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_mulhi_epu16&expand=3971)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmulhuw))]
pub unsafe fn _mm512_mask_mulhi_epu16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mulhi_epu16(a, b).as_u16x32();
    transmute(simd_select_bitmask(k, mul, src.as_u16x32()))
}

/// Multiply the packed unsigned 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_mulhi_epu16&expand=3972)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmulhuw))]
pub unsafe fn _mm512_maskz_mulhi_epu16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mulhi_epu16(a, b).as_u16x32();
    let zero = _mm512_setzero_si512().as_u16x32();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiply the packed signed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mulhi_epi16&expand=3962)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmulhw))]
pub unsafe fn _mm512_mulhi_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmulhw(a.as_i16x32(), b.as_i16x32()))
}

/// Multiply the packed signed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_mulhi_epi16&expand=3960)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmulhw))]
pub unsafe fn _mm512_mask_mulhi_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mulhi_epi16(a, b).as_i16x32();
    transmute(simd_select_bitmask(k, mul, src.as_i16x32()))
}

/// Multiply the packed signed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the high 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_mulhi_epi16&expand=3961)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmulhw))]
pub unsafe fn _mm512_maskz_mulhi_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mulhi_epi16(a, b).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Truncate each intermediate integer to the 18 most significant bits, round by adding 1, and store bits \[16:1\] to dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mulhrs_epi16&expand=3986)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmulhrsw))]
pub unsafe fn _mm512_mulhrs_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmulhrsw(a.as_i16x32(), b.as_i16x32()))
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Truncate each intermediate integer to the 18 most significant bits, round by adding 1, and store bits [16:1] to dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_mulhrs_epi16&expand=3984)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmulhrsw))]
pub unsafe fn _mm512_mask_mulhrs_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mulhrs_epi16(a, b).as_i16x32();
    transmute(simd_select_bitmask(k, mul, src.as_i16x32()))
}

/// Multiply packed signed 16-bit integers in a and b, producing intermediate signed 32-bit integers. Truncate each intermediate integer to the 18 most significant bits, round by adding 1, and store bits \[16:1\] to dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_mulhrs_epi16&expand=3985)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmulhrsw))]
pub unsafe fn _mm512_maskz_mulhrs_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mulhrs_epi16(a, b).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mullo_epi16&expand=3996)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmullw))]
pub unsafe fn _mm512_mullo_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(simd_mul(a.as_i16x32(), b.as_i16x32()))
}

/// Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_mullo_epi16&expand=3994)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmullw))]
pub unsafe fn _mm512_mask_mullo_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mullo_epi16(a, b).as_i16x32();
    transmute(simd_select_bitmask(k, mul, src.as_i16x32()))
}

/// Multiply the packed 16-bit integers in a and b, producing intermediate 32-bit integers, and store the low 16 bits of the intermediate integers in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_mullo_epi16&expand=3995)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmullw))]
pub unsafe fn _mm512_maskz_mullo_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let mul = _mm512_mullo_epi16(a, b).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, mul, zero))
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed maximum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_max_epu16&expand=3609)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxuw))]
pub unsafe fn _mm512_max_epu16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmaxuw(a.as_u16x32(), b.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_max_epu16&expand=3607)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxuw))]
pub unsafe fn _mm512_mask_max_epu16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epu16(a, b).as_u16x32();
    transmute(simd_select_bitmask(k, max, src.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_max_epu16&expand=3608)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxuw))]
pub unsafe fn _mm512_maskz_max_epu16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epu16(a, b).as_u16x32();
    let zero = _mm512_setzero_si512().as_u16x32();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed maximum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_max_epu8&expand=3636)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxub))]
pub unsafe fn _mm512_max_epu8(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmaxub(a.as_u8x64(), b.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_max_epu8&expand=3634)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxub))]
pub unsafe fn _mm512_mask_max_epu8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epu8(a, b).as_u8x64();
    transmute(simd_select_bitmask(k, max, src.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_max_epu8&expand=3635)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxub))]
pub unsafe fn _mm512_maskz_max_epu8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epu8(a, b).as_u8x64();
    let zero = _mm512_setzero_si512().as_u8x64();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed signed 16-bit integers in a and b, and store packed maximum values in dst. 
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_max_epi16&expand=3573)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxsw))]
pub unsafe fn _mm512_max_epi16(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmaxsw(a.as_i16x32(), b.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_max_epi16&expand=3571)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxsw))]
pub unsafe fn _mm512_mask_max_epi16(src: __m512i, k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epi16(a, b).as_i16x32();
    transmute(simd_select_bitmask(k, max, src.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_max_epi16&expand=3572)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxsw))]
pub unsafe fn _mm512_maskz_max_epi16(k: __mmask32, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epi16(a, b).as_i16x32();
    let zero = _mm512_setzero_si512().as_i16x32();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed signed 8-bit integers in a and b, and store packed maximum values in dst.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_max_epi8&expand=3600)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxsb))]
pub unsafe fn _mm512_max_epi8(a: __m512i, b: __m512i) -> __m512i {
    transmute(vpmaxsb(a.as_i8x64(), b.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b, and store packed maximum values in dst using writemask k (elements are copied from src when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_max_epi8&expand=3598)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxsb))]
pub unsafe fn _mm512_mask_max_epi8(src: __m512i, k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epi8(a, b).as_i8x64();
    transmute(simd_select_bitmask(k, max, src.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b, and store packed maximum values in dst using zeromask k (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_maskz_max_epi8&expand=3599)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpmaxsb))]
pub unsafe fn _mm512_maskz_max_epi8(k: __mmask64, a: __m512i, b: __m512i) -> __m512i {
    let max = _mm512_max_epi8(a, b).as_i8x64();
    let zero = _mm512_setzero_si512().as_i8x64();
    transmute(simd_select_bitmask(k, max, zero))
}

/// Compare packed unsigned 16-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=512_cmplt_epu16_mask&expand=1050)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<u16x32, _>(simd_lt(a.as_u16x32(), b.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmplt_epu16_mask&expand=1051)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmplt_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=mm512_cmplt_epu8_mask&expand=1068)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<u8x64, _>(simd_lt(a.as_u8x64(), b.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmplt_epu8_mask&expand=1069)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmplt_epu8_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmplt_epi16_mask&expand=1022)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<i16x32, _>(simd_lt(a.as_i16x32(), b.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmplt_epi16_mask&expand=1023)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmplt_epi16_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for less-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmplt_epi8_mask&expand=1044)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmplt_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<i8x64, _>(simd_lt(a.as_i8x64(), b.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b for less-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmplt_epi8_mask&expand=1045)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmplt_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmplt_epi8_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpgt_epu16_mask&expand=927)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpgt_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<u16x32, _>(simd_gt(a.as_u16x32(), b.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpgt_epu16_mask&expand=928)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpgt_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmpgt_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpgt_epu8_mask&expand=945)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpgt_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<u8x64, _>(simd_gt(a.as_u8x64(), b.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpgt_epu8_mask&expand=946)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpgt_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmpgt_epu8_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpgt_epi16_mask&expand=897)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpgt_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<i16x32, _>(simd_gt(a.as_i16x32(), b.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpgt_epi16_mask&expand=898)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpgt_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmpgt_epi16_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for greater-than, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpgt_epi8_mask&expand=921)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpgt_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<i8x64, _>(simd_gt(a.as_i8x64(), b.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b for greater-than, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpgt_epi8_mask&expand=922)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpgt_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmpgt_epi8_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmple_epu16_mask&expand=989)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmple_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<u16x32, _>(simd_le(a.as_u16x32(), b.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmple_epu16_mask&expand=990)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmple_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmple_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.   
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmple_epu8_mask&expand=1007)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmple_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<u8x64, _>(simd_le(a.as_u8x64(), b.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmple_epu8_mask&expand=1008)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmple_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmple_epu8_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmple_epi16_mask&expand=965)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmple_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<i16x32, _>(simd_le(a.as_i16x32(), b.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmple_epi16_mask&expand=966)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmple_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmple_epi16_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmple_epi8_mask&expand=983)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmple_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<i8x64, _>(simd_le(a.as_i8x64(), b.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b for less-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmple_epi8_mask&expand=984)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmple_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmple_epi8_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpge_epu16_mask&expand=867)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpge_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<u16x32, _>(simd_ge(a.as_u16x32(), b.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpge_epu16_mask&expand=868)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpge_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmpge_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpge_epu8_mask&expand=885)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpge_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<u8x64, _>(simd_ge(a.as_u8x64(), b.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpge_epu8_mask&expand=886)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpge_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmpge_epu8_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpge_epi16_mask&expand=843)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpge_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<i16x32, _>(simd_ge(a.as_i16x32(), b.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpge_epi16_mask&expand=844)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpge_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmpge_epi16_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpge_epi8_mask&expand=861)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpge_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<i8x64, _>(simd_ge(a.as_i8x64(), b.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b for greater-than-or-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpge_epi8_mask&expand=862)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpge_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmpge_epi8_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpeq_epu16_mask&expand=801)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpeq_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<u16x32, _>(simd_eq(a.as_u16x32(), b.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpeq_epu16_mask&expand=802)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpeq_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmpeq_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpeq_epu8_mask&expand=819)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpeq_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<u8x64, _>(simd_eq(a.as_u8x64(), b.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpeq_epu8_mask&expand=820)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpeq_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmpeq_epu8_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpeq_epi16_mask&expand=771)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpeq_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<i16x32, _>(simd_eq(a.as_i16x32(), b.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set). 
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpeq_epi16_mask&expand=772)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpeq_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmpeq_epi16_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for equality, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpeq_epi8_mask&expand=795)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpeq_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<i8x64, _>(simd_eq(a.as_i8x64(), b.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b for equality, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpeq_epi8_mask&expand=796)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpeq_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmpeq_epi8_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpneq_epu16_mask&expand=1106)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpneq_epu16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<u16x32, _>(simd_ne(a.as_u16x32(), b.as_u16x32()))
}

/// Compare packed unsigned 16-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpneq_epu16_mask&expand=1107)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpneq_epu16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmpneq_epu16_mask(a, b) & k1
}

/// Compare packed unsigned 8-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpneq_epu8_mask&expand=1124)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpneq_epu8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<u8x64, _>(simd_ne(a.as_u8x64(), b.as_u8x64()))
}

/// Compare packed unsigned 8-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpneq_epu8_mask&expand=1125)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpneq_epu8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmpneq_epu8_mask(a, b) & k1
}

/// Compare packed signed 16-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpneq_epi16_mask&expand=1082)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpneq_epi16_mask(a: __m512i, b: __m512i) -> __mmask32 {
    simd_bitmask::<i16x32, _>(simd_ne(a.as_i16x32(), b.as_i16x32()))
}

/// Compare packed signed 16-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpneq_epi16_mask&expand=1083)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpneq_epi16_mask(k1: __mmask32, a: __m512i, b: __m512i) -> __mmask32 {
    _mm512_cmpneq_epi16_mask(a, b) & k1
}

/// Compare packed signed 8-bit integers in a and b for not-equal, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmpneq_epi8_mask&expand=1100)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_cmpneq_epi8_mask(a: __m512i, b: __m512i) -> __mmask64 {
    simd_bitmask::<i8x64, _>(simd_ne(a.as_i8x64(), b.as_i8x64()))
}

/// Compare packed signed 8-bit integers in a and b for not-equal, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmpneq_epi8_mask&expand=1101)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vpcmp))]
pub unsafe fn _mm512_mask_cmpneq_epi8_mask(k1: __mmask64, a: __m512i, b: __m512i) -> __mmask64 {
    _mm512_cmpneq_epi8_mask(a, b) & k1
}

/// Compare packed unsigned 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmp_epu16_mask&expand=715)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_cmp_epu16_mask(a: __m512i, b: __m512i, imm8: i32) -> __mmask32 {
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpuw(a.as_u16x32(), b.as_u16x32(), $imm3, 0b11111111_11111111_11111111_11111111)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed unsigned 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmp_epu16_mask&expand=716)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_mask_cmp_epu16_mask(
    k1: __mmask32,
    a: __m512i,
    b: __m512i,
    imm8: i32,
) -> __mmask32 {
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpuw(a.as_u16x32(), b.as_u16x32(), $imm3, k1)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed unsigned 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmp_epu8_mask&expand=733)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_cmp_epu8_mask(a: __m512i, b: __m512i, imm8: i32) -> __mmask64 {
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpub(a.as_u8x64(), b.as_u8x64(), $imm3, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed unsigned 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmp_epu8_mask&expand=734)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_mask_cmp_epu8_mask(
    k1: __mmask64,
    a: __m512i,
    b: __m512i,
    imm8: i32,
) -> __mmask64 {
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpub(a.as_u8x64(), b.as_u8x64(), $imm3, k1)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmp_epi16_mask&expand=691)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_cmp_epi16_mask(a: __m512i, b: __m512i, imm8: i32) -> __mmask32 {
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpw(a.as_i16x32(), b.as_i16x32(), $imm3, 0b11111111_11111111_11111111_11111111)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 16-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmp_epi16_mask&expand=692)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_mask_cmp_epi16_mask(
    k1: __mmask32,
    a: __m512i,
    b: __m512i,
    imm8: i32,
) -> __mmask32 {
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpw(a.as_i16x32(), b.as_i16x32(), $imm3, k1)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_cmp_epi8_mask&expand=709)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_args_required_const(2)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_cmp_epi8_mask(a: __m512i, b: __m512i, imm8: i32) -> __mmask64 {
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpb(a.as_i8x64(), b.as_i8x64(), $imm3, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Compare packed signed 8-bit integers in a and b based on the comparison operand specified by imm8, and store the results in mask vector k using zeromask k1 (elements are zeroed out when the corresponding mask bit is not set).
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_mask_cmp_epi8_mask&expand=710)
#[inline]
#[target_feature(enable = "avx512bw")]
#[rustc_args_required_const(3)]
#[cfg_attr(test, assert_instr(vpcmp, imm8 = 0))]
pub unsafe fn _mm512_mask_cmp_epi8_mask(
    k1: __mmask64,
    a: __m512i,
    b: __m512i,
    imm8: i32,
) -> __mmask64 {
    macro_rules! call {
        ($imm3:expr) => {
            vpcmpb(a.as_i8x64(), b.as_i8x64(), $imm3, k1)
        };
    }
    let r = constify_imm3!(imm8, call);
    transmute(r)
}

/// Load 512-bits (composed of 32 packed 16-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_loadu_epi16&expand=3368)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu16
pub unsafe fn _mm512_loadu_epi16(mem_addr: *const i16) -> __m512i {
    ptr::read_unaligned(mem_addr as *const __m512i)
}

/// Load 512-bits (composed of 64 packed 8-bit integers) from memory into dst. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_loadu_epi8&expand=3395)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu8
pub unsafe fn _mm512_loadu_epi8(mem_addr: *const i8) -> __m512i {
    ptr::read_unaligned(mem_addr as *const __m512i)
}

/// Store 512-bits (composed of 32 packed 16-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_storeu_epi16&expand=5622)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu32
pub unsafe fn _mm512_storeu_epi16(mem_addr: *mut i16, a: __m512i) {
    ptr::write_unaligned(mem_addr as *mut __m512i, a);
}

/// Store 512-bits (composed of 64 packed 8-bit integers) from a into memory. mem_addr does not need to be aligned on any particular boundary.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=_mm512_storeu_epi8&expand=5640)
#[inline]
#[target_feature(enable = "avx512bw")]
#[cfg_attr(test, assert_instr(vmovups))] //should be vmovdqu8
pub unsafe fn _mm512_storeu_epi8(mem_addr: *mut i8, a: __m512i) {
    ptr::write_unaligned(mem_addr as *mut __m512i, a);
}

#[allow(improper_ctypes)]
extern "C" {

    #[link_name = "llvm.x86.avx512.mask.paddus.w.512"]
    fn vpaddusw(a: u16x32, b: u16x32, src: u16x32, mask: u32) -> u16x32;
    #[link_name = "llvm.x86.avx512.mask.paddus.b.512"]
    fn vpaddusb(a: u8x64, b: u8x64, src: u8x64, mask: u64) -> u8x64;
    #[link_name = "llvm.x86.avx512.mask.padds.w.512"]
    fn vpaddsw(a: i16x32, b: i16x32, src: i16x32, mask: u32) -> i16x32;
    #[link_name = "llvm.x86.avx512.mask.padds.b.512"]
    fn vpaddsb(a: i8x64, b: i8x64, src: i8x64, mask: u64) -> i8x64;

    #[link_name = "llvm.x86.avx512.mask.psubus.w.512"]
    fn vpsubusw(a: u16x32, b: u16x32, src: u16x32, mask: u32) -> u16x32;
    #[link_name = "llvm.x86.avx512.mask.psubus.b.512"]
    fn vpsubusb(a: u8x64, b: u8x64, src: u8x64, mask: u64) -> u8x64;
    #[link_name = "llvm.x86.avx512.mask.psubs.w.512"]
    fn vpsubsw(a: i16x32, b: i16x32, src: i16x32, mask: u32) -> i16x32;
    #[link_name = "llvm.x86.avx512.mask.psubs.b.512"]
    fn vpsubsb(a: i8x64, b: i8x64, src: i8x64, mask: u64) -> i8x64;

    #[link_name = "llvm.x86.avx512.pmulhu.w.512"]
    fn vpmulhuw(a: u16x32, b: u16x32) -> u16x32;
    #[link_name = "llvm.x86.avx512.pmulh.w.512"]
    fn vpmulhw(a: i16x32, b: i16x32) -> i16x32;
    #[link_name = "llvm.x86.avx512.pmul.hr.sw.512"]
    fn vpmulhrsw(a: i16x32, b: i16x32) -> i16x32;

    #[link_name = "llvm.x86.avx512.mask.ucmp.w.512"]
    fn vpcmpuw(a: u16x32, b: u16x32, op: i32, mask: u32) -> u32;
    #[link_name = "llvm.x86.avx512.mask.ucmp.b.512"]
    fn vpcmpub(a: u8x64, b: u8x64, op: i32, mask: u64) -> u64;
    #[link_name = "llvm.x86.avx512.mask.cmp.w.512"]
    fn vpcmpw(a: i16x32, b: i16x32, op: i32, mask: u32) -> u32;
    #[link_name = "llvm.x86.avx512.mask.cmp.b.512"]
    fn vpcmpb(a: i8x64, b: i8x64, op: i32, mask: u64) -> u64;

    #[link_name = "llvm.x86.avx512.mask.pmaxu.w.512"]
    fn vpmaxuw(a: u16x32, b: u16x32) -> u16x32;
    #[link_name = "llvm.x86.avx512.mask.pmaxu.b.512"]
    fn vpmaxub(a: u8x64, b: u8x64) -> u8x64;
    #[link_name = "llvm.x86.avx512.mask.pmaxs.w.512"]
    fn vpmaxsw(a: i16x32, b: i16x32) -> i16x32;
    #[link_name = "llvm.x86.avx512.mask.pmaxs.b.512"]
    fn vpmaxsb(a: i8x64, b: i8x64) -> i8x64;

    #[link_name = "llvm.x86.avx512.mask.pmins.q.512"]
    fn vpminsq(a: i64x8, b: i64x8) -> i64x8;

    #[link_name = "llvm.x86.avx512.mask.pmaxu.d.512"]
    fn vpmaxud(a: u32x16, b: u32x16) -> u32x16;
    #[link_name = "llvm.x86.avx512.mask.pmaxu.q.512"]
    fn vpmaxuq(a: u64x8, b: u64x8) -> i64x8;
    #[link_name = "llvm.x86.avx512.mask.pminu.d.512"]
    fn vpminud(a: u32x16, b: u32x16) -> u32x16;
    #[link_name = "llvm.x86.avx512.mask.pminu.q.512"]
    fn vpminuq(a: u64x8, b: u64x8) -> i64x8;
}

#[cfg(test)]
mod tests {

    use stdarch_test::simd_test;

    use crate::core_arch::x86::*;
    use crate::hint::black_box;
    use crate::mem::{self};

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_abs_epi16() {
        let a = _mm512_set1_epi16(-1);
        let r = _mm512_abs_epi16(a);
        let e = _mm512_set1_epi16(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_abs_epi16() {
        let a = _mm512_set1_epi16(-1);
        let r = _mm512_mask_abs_epi16(a, 0, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_abs_epi16(a, 0b00000000_11111111_00000000_11111111, a);
        let e = _mm512_set_epi16(-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_abs_epi16() {
        let a = _mm512_set1_epi16(-1);
        let r = _mm512_maskz_abs_epi16(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_abs_epi16(0b00000000_11111111_00000000_11111111, a);
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                  0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_abs_epi8() {
        let a = _mm512_set1_epi8(-1);
        let r = _mm512_abs_epi8(a);
        let e = _mm512_set1_epi8(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_abs_epi8() {
        let a = _mm512_set1_epi8(-1);
        let r = _mm512_mask_abs_epi8(a, 0, a);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_abs_epi8(a, 0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111, a);
        let e = _mm512_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
                                -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
                                -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1,
                                -1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_abs_epi8() {
        let a = _mm512_set1_epi8(-1);
        let r = _mm512_maskz_abs_epi8(0, a);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_abs_epi8(0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111, a);
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
                                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_add_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(2);
        let r = _mm512_add_epi16(a, b);
        let e = _mm512_set1_epi16(3);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_add_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(2);
        let r = _mm512_mask_add_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_add_epi16(a, 0b00000000_11111111_00000000_11111111, a, b);
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3,
                                 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_add_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(2);
        let r = _mm512_maskz_add_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_add_epi16(0b00000000_11111111_00000000_11111111, a, b);
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3,
                                 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_add_epi8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(2);
        let r = _mm512_add_epi8(a, b);
        let e = _mm512_set1_epi8(3);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_add_epi8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(2);
        let r = _mm512_mask_add_epi8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_add_epi8(a, 0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111, a, b);
        let e = _mm512_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3,
                                1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_add_epi8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(2);
        let r = _mm512_maskz_add_epi8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_add_epi8(0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111, a, b);
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3,
                                0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3,
                                0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3,
                                0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 3);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_adds_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(u16::MAX as i16);
        let r = _mm512_adds_epu16(a, b);
        let e = _mm512_set1_epi16(u16::MAX as i16);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_adds_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(u16::MAX as i16);
        let r = _mm512_mask_adds_epu16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_adds_epu16(a, 0b00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, u16::MAX as i16, u16::MAX as i16, u16::MAX as i16, u16::MAX as i16);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_adds_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(u16::MAX as i16);
        let r = _mm512_maskz_adds_epu16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_adds_epu16(0b00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, u16::MAX as i16, u16::MAX as i16, u16::MAX as i16, u16::MAX as i16);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_adds_epu8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(u8::MAX as i8);
        let r = _mm512_adds_epu8(a, b);
        let e = _mm512_set1_epi8(u8::MAX as i8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_adds_epu8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(u8::MAX as i8);
        let r = _mm512_mask_adds_epu8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_adds_epu8(a, 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, u8::MAX as i8, u8::MAX as i8, u8::MAX as i8, u8::MAX as i8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_adds_epu8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(u8::MAX as i8);
        let r = _mm512_maskz_adds_epu8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_adds_epu8(0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, u8::MAX as i8, u8::MAX as i8, u8::MAX as i8, u8::MAX as i8);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_adds_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(i16::MAX);
        let r = _mm512_adds_epi16(a, b);
        let e = _mm512_set1_epi16(i16::MAX);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_adds_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(i16::MAX);
        let r = _mm512_mask_adds_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_adds_epi16(a, 0b00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, i16::MAX, i16::MAX, i16::MAX, i16::MAX);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_adds_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(i16::MAX);
        let r = _mm512_maskz_adds_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_adds_epi16(0b00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i16::MAX, i16::MAX, i16::MAX, i16::MAX);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_adds_epi8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(i8::MAX);
        let r = _mm512_adds_epi8(a, b);
        let e = _mm512_set1_epi8(i8::MAX);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_adds_epi8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(i8::MAX);
        let r = _mm512_mask_adds_epi8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_adds_epi8(a, 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, i8::MAX, i8::MAX, i8::MAX, i8::MAX);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_adds_epi8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(i8::MAX);
        let r = _mm512_maskz_adds_epi8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_adds_epi8(0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i8::MAX, i8::MAX, i8::MAX, i8::MAX);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_sub_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(2);
        let r = _mm512_sub_epi16(a, b);
        let e = _mm512_set1_epi16(-1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_sub_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(2);
        let r = _mm512_mask_sub_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_sub_epi16(a, 0b00000000_11111111_00000000_11111111, a, b);
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1,
                                 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_sub_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(2);
        let r = _mm512_maskz_sub_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_sub_epi16(0b00000000_11111111_00000000_11111111, a, b);
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1,
                                 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_sub_epi8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(2);
        let r = _mm512_sub_epi8(a, b);
        let e = _mm512_set1_epi8(-1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_sub_epi8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(2);
        let r = _mm512_mask_sub_epi8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_sub_epi8(a, 0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111, a, b);
        let e = _mm512_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1,
                                1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1,
                                1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1,
                                1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_sub_epi8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(2);
        let r = _mm512_maskz_sub_epi8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_sub_epi8(0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111, a, b);
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1,
                                0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1,
                                0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1,
                                0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_subs_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(u16::MAX as i16);
        let r = _mm512_subs_epu16(a, b);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_subs_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(u16::MAX as i16);
        let r = _mm512_mask_subs_epu16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_subs_epu16(a, 0b00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_subs_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(u16::MAX as i16);
        let r = _mm512_maskz_subs_epu16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_subs_epu16(0b00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_subs_epu8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(u8::MAX as i8);
        let r = _mm512_subs_epu8(a, b);
        let e = _mm512_set1_epi8(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_subs_epu8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(u8::MAX as i8);
        let r = _mm512_mask_subs_epu8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_subs_epu8(a, 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_subs_epu8() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(u8::MAX as i8);
        let r = _mm512_maskz_subs_epu8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_subs_epu8(0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_subs_epi16() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(i16::MAX);
        let r = _mm512_subs_epi16(a, b);
        let e = _mm512_set1_epi16(i16::MIN);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_subs_epi16() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(i16::MAX);
        let r = _mm512_mask_subs_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_subs_epi16(a, 0b00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi16(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                 -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, i16::MIN, i16::MIN, i16::MIN, i16::MIN);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_subs_epi16() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(i16::MAX);
        let r = _mm512_maskz_subs_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_subs_epi16(0b00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i16::MIN, i16::MIN, i16::MIN, i16::MIN);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_subs_epi8() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(i8::MAX);
        let r = _mm512_subs_epi8(a, b);
        let e = _mm512_set1_epi8(i8::MIN);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_subs_epi8() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(i8::MAX);
        let r = _mm512_mask_subs_epi8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_subs_epi8(a, 0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, i8::MIN, i8::MIN, i8::MIN, i8::MIN);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_subs_epi8() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(i8::MAX);
        let r = _mm512_maskz_subs_epi8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_subs_epi8(0b00000000_00000000_00000000_00000000_00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i8::MIN, i8::MIN, i8::MIN, i8::MIN);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mulhi_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_mulhi_epu16(a, b);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_mulhi_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_mask_mulhi_epu16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_mulhi_epu16(a, 0b00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_mulhi_epu16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_maskz_mulhi_epu16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_mulhi_epu16(0b00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mulhi_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_mulhi_epi16(a, b);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_mulhi_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_mask_mulhi_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_mulhi_epi16(a, 0b00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_mulhi_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_maskz_mulhi_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_mulhi_epi16(0b00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mulhrs_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_mulhrs_epi16(a, b);
        let e = _mm512_set1_epi16(0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_mulhrs_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_mask_mulhrs_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_mulhrs_epi16(a, 0b00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_mulhrs_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_maskz_mulhrs_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_mulhrs_epi16(0b00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mullo_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_mullo_epi16(a, b);
        let e = _mm512_set1_epi16(1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_mullo_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_mask_mullo_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_mullo_epi16(a, 0b00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi16(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_maskz_mullo_epi16() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let r = _mm512_maskz_mullo_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_mullo_epi16(0b00000000_00000000_00000000_00001111, a, b);
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_max_epu16() {
        let a = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_max_epu16(a, b);
        let e = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15,
                                 15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_max_epu16() {
        let a = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_max_epu16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_max_epu16(a, 0b00000000_11111111_00000000_11111111, a, b);
        let e = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_max_epu16() {
        let a = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_max_epu16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_max_epu16(0b00000000_11111111_00000000_11111111, a, b);
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_max_epu8() {
        let a = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_max_epu8(a, b);
        let e = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15,
                                15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15,
                                15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15,
                                15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_max_epu8() {
        let a = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_max_epu8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_max_epu8(a, 0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111, a, b);
        let e = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_max_epu8() {
        let a = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_max_epu8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_max_epu8(0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111, a, b);
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_max_epi16() {
        let a = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_max_epi16(a, b);
        let e = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15,
                                 15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_max_epi16() {
        let a = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_max_epi16(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_max_epi16(a, 0b00000000_11111111_00000000_11111111, a, b);
        let e = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_max_epi16() {
        let a = _mm512_set_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_set_epi16(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_max_epi16(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_max_epi16(0b00000000_11111111_00000000_11111111, a, b);
        let e = _mm512_set_epi16(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15,
                                 0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_max_epi8() {
        let a = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_max_epi8(a, b);
        let e = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15,
                                15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15,
                                15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15,
                                15, 14, 13, 12, 11, 10, 9, 8, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_mask_max_epi8() {
        let a = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_mask_max_epi8(a, 0, a, b);
        assert_eq_m512i(r, a);
        let r = _mm512_mask_max_epi8(a, 0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111, a, b);
        let e = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512f")]
    unsafe fn test_mm512_maskz_max_epi8() {
        let a = _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        let b = _mm512_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                                15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        let r = _mm512_maskz_max_epi8(0, a, b);
        assert_eq_m512i(r, _mm512_setzero_si512());
        let r = _mm512_maskz_max_epi8(0b00000000_11111111_00000000_11111111_00000000_11111111_00000000_11111111, a, b);
        let e = _mm512_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15,
                                0, 0, 0, 0, 0, 0, 0, 0, 8, 9, 10, 11, 12, 13, 14, 15);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmplt_epu16_mask() {
        let a = _mm512_set1_epi16(-2);
        let b = _mm512_set1_epi16(-1);
        let m = _mm512_cmplt_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmplt_epu16_mask() {
        let a = _mm512_set1_epi16(-2);
        let b = _mm512_set1_epi16(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmplt_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmplt_epu8_mask() {
        let a = _mm512_set1_epi8(-2);
        let b = _mm512_set1_epi8(-1);
        let m = _mm512_cmplt_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmplt_epu8_mask() {
        let a = _mm512_set1_epi8(-2);
        let b = _mm512_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmplt_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmplt_epi16_mask() {
        let a = _mm512_set1_epi16(-2);
        let b = _mm512_set1_epi16(-1);
        let m = _mm512_cmplt_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmplt_epi16_mask() {
        let a = _mm512_set1_epi16(-2);
        let b = _mm512_set1_epi16(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmplt_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmplt_epi8_mask() {
        let a = _mm512_set1_epi8(-2);
        let b = _mm512_set1_epi8(-1);
        let m = _mm512_cmplt_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmplt_epi8_mask() {
        let a = _mm512_set1_epi8(-2);
        let b = _mm512_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmplt_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpgt_epu16_mask() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(1);
        let m = _mm512_cmpgt_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpgt_epu16_mask() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpgt_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpgt_epu8_mask() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(1);
        let m = _mm512_cmpgt_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpgt_epu8_mask() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpgt_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpgt_epi16_mask() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(-1);
        let m = _mm512_cmpgt_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpgt_epi16_mask() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpgt_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpgt_epi8_mask() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(-1);
        let m = _mm512_cmpgt_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpgt_epi8_mask() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpgt_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmple_epu16_mask() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(-1);
        let m = _mm512_cmple_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmple_epu16_mask() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmple_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmple_epu8_mask() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(-1);
        let m = _mm512_cmple_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmple_epu8_mask() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmple_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmple_epi16_mask() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(-1);
        let m = _mm512_cmple_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmple_epi16_mask() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmple_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmple_epi8_mask() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(-1);
        let m = _mm512_cmple_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmple_epi8_mask() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmple_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpge_epu16_mask() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let m = _mm512_cmpge_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpge_epu16_mask() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpge_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpge_epu8_mask() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(1);
        let m = _mm512_cmpge_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpge_epu8_mask() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpge_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpge_epi16_mask() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(-1);
        let m = _mm512_cmpge_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpge_epi16_mask() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpge_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpge_epi8_mask() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(-1);
        let m = _mm512_cmpge_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpge_epi8_mask() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpge_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpeq_epu16_mask() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let m = _mm512_cmpeq_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpeq_epu16_mask() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpeq_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpeq_epu8_mask() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(1);
        let m = _mm512_cmpeq_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpeq_epu8_mask() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpeq_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpeq_epi16_mask() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(-1);
        let m = _mm512_cmpeq_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpeq_epi16_mask() {
        let a = _mm512_set1_epi16(-1);
        let b = _mm512_set1_epi16(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpeq_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpeq_epi8_mask() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(-1);
        let m = _mm512_cmpeq_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpeq_epi8_mask() {
        let a = _mm512_set1_epi8(-1);
        let b = _mm512_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpeq_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpneq_epu16_mask() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(1);
        let m = _mm512_cmpneq_epu16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpneq_epu16_mask() {
        let a = _mm512_set1_epi16(2);
        let b = _mm512_set1_epi16(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpneq_epu16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpneq_epu8_mask() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(1);
        let m = _mm512_cmpneq_epu8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpneq_epu8_mask() {
        let a = _mm512_set1_epi8(2);
        let b = _mm512_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpneq_epu8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpneq_epi16_mask() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(-1);
        let m = _mm512_cmpneq_epi16_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpneq_epi16_mask() {
        let a = _mm512_set1_epi16(1);
        let b = _mm512_set1_epi16(-1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpneq_epi16_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmpneq_epi8_mask() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(-1);
        let m = _mm512_cmpneq_epi8_mask(a, b);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmpneq_epi8_mask() {
        let a = _mm512_set1_epi8(1);
        let b = _mm512_set1_epi8(-1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmpneq_epi8_mask(mask, a, b);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmp_epu16_mask() {
        let a = _mm512_set1_epi16(0);
        let b = _mm512_set1_epi16(1);
        let m = _mm512_cmp_epu16_mask(a, b, _MM_CMPINT_LT);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmp_epu16_mask() {
        let a = _mm512_set1_epi16(0);
        let b = _mm512_set1_epi16(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmp_epu16_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmp_epu8_mask() {
        let a = _mm512_set1_epi8(0);
        let b = _mm512_set1_epi8(1);
        let m = _mm512_cmp_epu8_mask(a, b, _MM_CMPINT_LT);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmp_epu8_mask() {
        let a = _mm512_set1_epi8(0);
        let b = _mm512_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmp_epu8_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmp_epi16_mask() {
        let a = _mm512_set1_epi16(0);
        let b = _mm512_set1_epi16(1);
        let m = _mm512_cmp_epi16_mask(a, b, _MM_CMPINT_LT);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmp_epi16_mask() {
        let a = _mm512_set1_epi16(0);
        let b = _mm512_set1_epi16(1);
        let mask = 0b01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmp_epi16_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_cmp_epi8_mask() {
        let a = _mm512_set1_epi8(0);
        let b = _mm512_set1_epi8(1);
        let m = _mm512_cmp_epi8_mask(a, b, _MM_CMPINT_LT);
        assert_eq!(m, 0b11111111_11111111_11111111_11111111_11111111_11111111_11111111_11111111);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_mask_cmp_epi8_mask() {
        let a = _mm512_set1_epi8(0);
        let b = _mm512_set1_epi8(1);
        let mask = 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101;
        let r = _mm512_mask_cmp_epi8_mask(mask, a, b, _MM_CMPINT_LT);
        assert_eq!(r, 0b01010101_01010101_01010101_01010101_01010101_01010101_01010101_01010101);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_loadu_epi16() {
        let a: [i16; 32] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
        let r = _mm512_loadu_epi16(&a[0]);
        let e = _mm512_set_epi16(32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_loadu_epi8() {
        let a: [i8; 64] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                           1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32];
        let r = _mm512_loadu_epi8(&a[0]);
        let e = _mm512_set_epi8(32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                                32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1);
        assert_eq_m512i(r, e);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_storeu_epi16() {
        let a = _mm512_set1_epi16(9);
        let mut r = _mm512_undefined_epi32();
        _mm512_storeu_epi16(&mut r as *mut _ as *mut i16, a);
        assert_eq_m512i(r, a);
    }

    #[simd_test(enable = "avx512bw")]
    unsafe fn test_mm512_storeu_epi8() {
        let a = _mm512_set1_epi8(9);
        let mut r = _mm512_undefined_epi32();
        _mm512_storeu_epi8(&mut r as *mut _ as *mut i8, a);
        assert_eq_m512i(r, a);
    }
}
