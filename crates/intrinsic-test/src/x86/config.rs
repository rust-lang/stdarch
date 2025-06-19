pub fn build_notices(line_prefix: &str) -> String {
    format!(
        "\
{line_prefix}This is a transient test file, not intended for distribution. Some aspects of the
{line_prefix}test are derived from an XML specification, published under the same license as the
{line_prefix}`intrinsic-test` crate.\n
"
    )
}
