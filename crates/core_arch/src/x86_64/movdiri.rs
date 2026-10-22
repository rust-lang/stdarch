//! `MOVDIRI` quadword direct store.

#[cfg(test)]
use stdarch_test::assert_instr;

#[allow(improper_ctypes)]
unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.x86.directstore64"]
    fn directstore64(dst: *mut u64, val: u64);
}

/// Stores the 64-bit integer `val` to `dst` using a direct store.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_directstoreu_u64)
///
/// # Safety
///
/// `dst` must be valid for writes of 8 bytes; it does not need to be aligned.
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
/// [`_mm_sfence`]: crate::arch::x86_64::_mm_sfence
#[inline]
#[target_feature(enable = "movdiri")]
#[cfg_attr(test, assert_instr(movdiri))]
#[unstable(feature = "simd_x86_movdiri", issue = "163741")]
pub unsafe fn _directstoreu_u64(dst: *mut u64, val: u64) {
    directstore64(dst, val);
}

#[cfg(test)]
mod tests {
    use crate::core_arch::{x86::*, x86_64::*};
    use stdarch_test::simd_test;

    #[simd_test(enable = "movdiri")]
    unsafe fn test_directstoreu_u64() {
        let mut x = 0_u64;
        _directstoreu_u64(&mut x, 0x0123_4567_89ab_cdef);
        _mm_sfence();
        assert_eq!(x, 0x0123_4567_89ab_cdef);
    }
}
