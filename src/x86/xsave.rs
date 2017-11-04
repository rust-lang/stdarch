//! `xsave` target feature intrinsics

#[cfg(test)]
use stdsimd_test::assert_instr;

/// Reads the contents of the extended control register `XCR`
/// specified in `xcr_no`.
#[inline(always)]
// #[target_feature = "+xsave"] // FIXME: see
// https://github.com/rust-lang-nursery/stdsimd/issues/167
#[cfg_attr(test, assert_instr(xgetbv))]
pub unsafe fn _xgetbv(xcr_no: u32) -> u64 {
    let eax: u32;
    let edx: u32;

    asm!("xgetbv"
         : "={eax}"(eax),  "={edx}"(edx)
         : "{ecx}"(xcr_no)
         : :);

    ((edx as u64) << 32) | (eax as u64)
}
