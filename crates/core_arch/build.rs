use std::env;

fn main() {
    println!("cargo:rustc-cfg=core_arch_docs");

    // Used to tell our `#[assert_instr]` annotations that all simd intrinsics
    // are available to test their codegen, since some are gated behind an extra
    // `-Ctarget-feature=+unimplemented-simd128` that doesn't have any
    // equivalent in `#[target_feature]` right now.
    if env::var("RUSTFLAGS")
        .unwrap_or_default()
        .contains("unimplemented-simd128")
    {
        println!("cargo:rust-cfg=all_simd");
    }
}
