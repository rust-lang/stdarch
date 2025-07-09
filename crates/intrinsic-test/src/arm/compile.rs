use crate::common::compile_c::CompilationCommandBuilder;

pub fn compile_c_arm(
    compiler: &str,
    target: &str,
    cxx_toolchain_dir: Option<&str>,
    inputs: &[String],
    output: Option<&str>,
) -> bool {
    // -ffp-contract=off emulates Rust's approach of not fusing separate mul-add operations
    let mut command = CompilationCommandBuilder::new()
        .add_arch_flags(vec!["armv8.6-a", "crypto", "crc", "dotprod", "fp16"])
        .set_compiler(compiler)
        .set_target(target)
        .set_opt_level("2")
        .set_cxx_toolchain_dir(cxx_toolchain_dir)
        .set_project_root("c_programs")
        .add_extra_flags(vec!["-ffp-contract=off", "-Wno-narrowing"]);

    if !target.contains("v7") {
        command = command.add_arch_flags(vec!["faminmax", "lut", "sha3"]);
    }

    /*
     * clang++ cannot link an aarch64_be object file, so we invoke
     * aarch64_be-unknown-linux-gnu's C++ linker. This ensures that we
     * are testing the intrinsics against LLVM.
     *
     * Note: setting `--sysroot=<...>` which is the obvious thing to do
     * does not work as it gets caught up with `#include_next <stdlib.h>`
     * not existing...
     */
    if target.contains("aarch64_be") {
        command = command
            .set_linker(
                cxx_toolchain_dir.unwrap_or("").to_string() + "/bin/aarch64_be-none-linux-gnu-g++",
            )
            .set_include_paths(vec![
                "/include",
                "/aarch64_be-none-linux-gnu/include",
                "/aarch64_be-none-linux-gnu/include/c++/14.2.1",
                "/aarch64_be-none-linux-gnu/include/c++/14.2.1/aarch64_be-none-linux-gnu",
                "/aarch64_be-none-linux-gnu/include/c++/14.2.1/backward",
                "/aarch64_be-none-linux-gnu/libc/usr/include",
            ]);
    }

    if !compiler.contains("clang") {
        command = command.add_extra_flag("-flax-vector-conversions");
    }

    let mut command = command.into_command();
    command.command_mut().current_dir("c_programs");

    for input in inputs {
        assert!(
            std::path::Path::new("c_programs").join(input).exists(),
            "{}",
            input
        );
    }
    command.command_mut().args(inputs);

    if let Some(output) = output {
        trace!("running {compiler} to produce {output}");
        if output.ends_with(".o") {
            command.command_mut().arg("-c");
        }
        command.command_mut().args(["-o", output]);
    } else {
        trace!("running {compiler}");
    }

    trace!("running {compiler}\n{:?}", &command);

    if log::log_enabled!(log::Level::Trace) {
        command.command_mut().stdout(std::process::Stdio::inherit());
        command.command_mut().stderr(std::process::Stdio::inherit());
    }

    let output = command.output();

    trace!("{compiler} is done");

    if let Ok(output) = output {
        if output.status.success() {
            true
        } else {
            error!(
                "Failed to compile code for intrinsics: \n\nstdout:\n{}\n\nstderr:\n{}",
                std::str::from_utf8(&output.stdout).unwrap_or(""),
                std::str::from_utf8(&output.stderr).unwrap_or("")
            );
            false
        }
    } else {
        error!("Command failed: {output:#?}");
        false
    }
}
