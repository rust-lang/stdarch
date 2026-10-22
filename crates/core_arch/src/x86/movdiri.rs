//! `MOVDIRI` doubleword direct store.

#[cfg(test)]
use stdarch_test::assert_instr;

#[allow(improper_ctypes)]
unsafe extern "llvm-intrinsic" {
    #[link_name = "llvm.x86.directstore32"]
    fn directstore32(dst: *mut u32, val: u32);
}

/// Stores the 32-bit integer `val` to `dst` using a direct store.
///
/// [Intel's documentation](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_directstoreu_u32)
///
/// # Safety
///
/// `dst` must be valid for writes of 4 bytes; it does not need to be aligned.
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
#[target_feature(enable = "movdiri")]
#[cfg_attr(test, assert_instr(movdiri))]
#[unstable(feature = "simd_x86_movdiri", issue = "163741")]
pub unsafe fn _directstoreu_u32(dst: *mut u32, val: u32) {
    directstore32(dst, val);
}

#[cfg(test)]
mod tests {
    use crate::core_arch::x86::*;
    use stdarch_test::simd_test;

    #[simd_test(enable = "movdiri")]
    unsafe fn test_directstoreu_u32() {
        let mut x = 0_u32;
        _directstoreu_u32(&mut x, 0xdead_beef);
        _mm_sfence();
        assert_eq!(x, 0xdead_beef);
    }
}
