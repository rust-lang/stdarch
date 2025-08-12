pub fn build_notices(line_prefix: &str) -> String {
    format!(
        "\
{line_prefix}This is a transient test file, not intended for distribution. Some aspects of the
{line_prefix}test are derived from `stdarch-gen-loongarch/lsx.spec` and `stdarch-gen-loongarch/lasx.spec`, 
{line_prefix}published under the same license as the `intrinsic-test` crate.\n
"
    )
}
