//! `MOVDIR64B` 64-byte direct store.

#[cfg(test)]
use stdarch_test::assert_instr;

#[allow(improper_ctypes)]
unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.x86.movdir64b"]
    fn movdir64b(dst: *mut u8, src: *const u8);
}

/// Moves a 64-byte value from `src` to `dst` using a direct store.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_movdir64b)
///
/// # Safety
///
/// `src` must be valid for reads of 64 bytes; it does not need to be aligned. `dst` must be valid
/// for writes of 64 bytes and aligned to 64 bytes (`MOVDIR64B` raises a general-protection fault
/// otherwise).
///
/// # Safety of direct stores
///
/// Like a non-temporal store, the direct store is weakly ordered with respect to other stores.
/// After using this intrinsic, but before any other access to the memory that this intrinsic
/// mutates, a call to [`_mm_sfence`] must be performed by the thread that used the intrinsic. In
/// particular, functions that call this intrinsic should generally call `_mm_sfence` before they
/// return.
///
/// See [`_mm_sfence`] for details.
///
/// [`_mm_sfence`]: crate::arch::x86::_mm_sfence
#[inline]
#[target_feature(enable = "movdir64b")]
#[cfg_attr(test, assert_instr(movdir64b))]
#[unstable(feature = "simd_x86_movdir64b", issue = "163741")]
pub unsafe fn _movdir64b(dst: *mut u8, src: *const u8) {
    movdir64b(dst, src);
}

#[cfg(test)]
mod tests {
    use crate::core_arch::x86::*;
    use stdarch_test::simd_test;

    #[repr(C, align(64))]
    struct Line([u8; 64]);

    #[simd_test(enable = "movdir64b")]
    unsafe fn test_movdir64b() {
        let src = Line(core::array::from_fn(|i| i as u8));
        let mut dst = Line([0; 64]);
        _movdir64b(dst.0.as_mut_ptr(), src.0.as_ptr());
        _mm_sfence();
        assert_eq!(dst.0, src.0);
    }
}
