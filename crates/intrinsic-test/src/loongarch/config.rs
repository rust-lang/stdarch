pub fn build_notices(line_prefix: &str) -> String {
    format!(
        "\
{line_prefix}This is a transient test file, not intended for distribution. Some aspects of the
{line_prefix}test are derived from `stdarch-gen-loongarch/lsx.spec` and `stdarch-gen-loongarch/lasx.spec`, 
{line_prefix}published under the same license as the `intrinsic-test` crate.\n
"
    )
}

pub const LOONGARCH_CONFIGURATIONS: &str = r#"
#![cfg_attr(any(target_arch = "loongarch64", target_arch = "loongarch32"), feature(stdarch_loongarch))]
"#;

// Format f16 values (and vectors containing them) in a way that is consistent with C.
pub const F16_FORMATTING_DEF: &str = r#"
/// Used to continue `Debug`ging SIMD types as `MySimd(1, 2, 3, 4)`, as they
/// were before moving to array-based simd.
#[inline]
fn debug_simd_finish<T: core::fmt::Debug, const N: usize>(
    formatter: &mut core::fmt::Formatter<'_>,
    type_name: &str,
    array: &[T; N],
) -> core::fmt::Result {
    core::fmt::Formatter::debug_tuple_fields_finish(
        formatter,
        type_name,
        &core::array::from_fn::<&dyn core::fmt::Debug, N, _>(|i| &array[i]),
    )
}

#[repr(transparent)]
struct Hex<T>(T);

impl<T: DebugHexF16> core::fmt::Debug for Hex<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        <T as DebugHexF16>::fmt(&self.0, f)
    }
}

fn debug_f16<T: DebugHexF16>(x: T) -> impl core::fmt::Debug {
    Hex(x)
}

trait DebugHexF16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result;
}

impl DebugHexF16 for f16 {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:#06x?}", self.to_bits())
    }
}
 "#;
