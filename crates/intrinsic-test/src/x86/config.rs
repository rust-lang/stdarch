pub fn build_notices(line_prefix: &str) -> String {
    format!(
        "\
{line_prefix}This is a transient test file, not intended for distribution. Some aspects of the
{line_prefix}test are derived from an XML specification, published under the same license as the
{line_prefix}`intrinsic-test` crate.\n
"
    )
}

// Format f16 values (and vectors containing them) in a way that is consistent with C.
pub const F16_FORMATTING_DEF: &str = r#"
#[repr(transparent)]
struct Hex<T>(T);
 "#;

pub const X86_CONFIGURATIONS: &str = r#"
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_avx512))]
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_avx2))]
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_avx))]
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_sse42))]
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_sse41))]
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_ssse3))]
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_sse3))]
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_sse2))]
#![cfg_attr(target_arch = "x86", feature(stdarch_x86_sse))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx512))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx2))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_avx))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_fma))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_aes))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_sha))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_bmi1))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_bmi2))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_lzcnt))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_popcnt))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_rdrand))]
#![cfg_attr(target_arch = "x86_64", feature(stdarch_x86_rdseed))]
#![cfg_attr(any(target_arch = "x86", target_arch = "x86_64"), feature(stdarch_x86_mmx))]
#![feature(fmt_helpers_for_derive)]
"#;
