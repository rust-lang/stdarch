//! `i386` intrinsics

/// Reads EFLAGS.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=__readeflags)
#[cfg(target_arch = "x86")]
#[inline(always)]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[rustc_deprecated(
    since = "1.29.0",
    reason = "See issue #51810 - use inline assembly instead"
)]
#[doc(hidden)]
pub unsafe fn __readeflags() -> u32 {
    let eflags: u32;
    asm!("pushfd; popl {}", out(reg) eflags, options(att_syntax, preserves_flags, nomem, pure));
    eflags
}

/// Reads EFLAGS.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=__readeflags)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[rustc_deprecated(
    since = "1.29.0",
    reason = "See issue #51810 - use inline assembly instead"
)]
#[doc(hidden)]
pub unsafe fn __readeflags() -> u64 {
    let eflags: u64;
    asm!("pushfq; popq {}", out(reg) eflags, options(att_syntax, preserves_flags, nomem, pure));
    eflags
}

/// Write EFLAGS.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=__writeeflags)
#[cfg(target_arch = "x86")]
#[inline(always)]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[rustc_deprecated(
    since = "1.29.0",
    reason = "See issue #51810 - use inline assembly instead"
)]
#[doc(hidden)]
pub unsafe fn __writeeflags(eflags: u32) {
    asm!("pushl {}; popfd", in(reg) eflags, options(att_syntax, nomem));
}

/// Write EFLAGS.
///
/// [Intel's documentation](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#text=__writeeflags)
#[cfg(target_arch = "x86_64")]
#[inline(always)]
#[stable(feature = "simd_x86", since = "1.27.0")]
#[rustc_deprecated(
    since = "1.29.0",
    reason = "See issue #51810 - use inline assembly instead"
)]
#[doc(hidden)]
pub unsafe fn __writeeflags(eflags: u64) {
    asm!("pushq {}; popfq", in(reg) eflags, options(att_syntax, nomem));
}

#[cfg(test)]
mod tests {
    use crate::core_arch::x86::*;

    #[test]
    #[allow(deprecated)]
    fn test_eflags() {
        unsafe {
            // reads eflags, writes them back, reads them again,
            // and compare for equality:
            let v = __readeflags();
            __writeeflags(v);
            let u = __readeflags();
            assert_eq!(v, u);
        }
    }
}
