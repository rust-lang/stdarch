#![feature(slice_partition_dedup)]
#[macro_use]
extern crate lazy_static;
#[macro_use]
extern crate log;

use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

use intrinsic::Intrinsic;
use itertools::Itertools;
use rayon::prelude::*;
use types::TypeKind;

use crate::argument::Argument;
use crate::format::Indentation;
use crate::json_parser::{get_neon_intrinsics, get_sve_intrinsics};

mod argument;
mod format;
mod intrinsic;
mod json_parser;
mod types;
mod values;

// The number of times each intrinsic will be called per constraint (also per-predicate pattern,
// for SVE intrinsics).
const PASSES: u32 = 20;

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Language {
    Rust,
    C,
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub enum Extension {
    NEON,
    SVE,
}

fn gen_code_c(
    indentation: Indentation,
    intrinsic: &Intrinsic,
    preset_vals: &[&Argument],
    context: String,
    mode: Extension,
    is_aarch32: bool,
) -> String {
    if let Some((current, preset_vals)) = preset_vals.split_last() {
        let name = &current.name;
        if current.is_predicate() {
            let passes = current
                .get_predicate_decls(indentation.nested(), Language::C)
                .into_iter()
                .enumerate()
                .map(|(i, p)| {
                    format!(
                        r#"{indentation}{{
{p}
{pass}
{indentation}}}"#,
                        pass = gen_code_c(
                            indentation.nested(),
                            intrinsic,
                            preset_vals,
                            format!("{context} {name}=pat{i}"),
                            mode,
                            is_aarch32
                        )
                    )
                })
                .join("\n");
            format!("{indentation}svbool_t {name};\n{passes}")
        } else if current.ty.kind() == TypeKind::Bool {
            // Some bool intrinics nest quite deeply, so prefer looping
            format!(
                r"{indentation}bool {name}_vals[] = {{true, false}};
{indentation}for(bool {name}: {name}_vals) {{
{pass}
{indentation}}}",
                pass = gen_code_c(
                    indentation.nested(),
                    intrinsic,
                    preset_vals,
                    format!("{context} {name}=\" << {name} << \""),
                    mode,
                    is_aarch32
                )
            )
        } else {
            current
                .constraints
                .iter()
                .flat_map(|c| c.iter())
                .map(|i| {
                    let ty = current.ty.c_type();
                    let val = if current.ty.kind().is_enum() {
                        format!("static_cast<{ty}>({i})")
                    } else {
                        i.to_string()
                    };
                    let indentation_1 = indentation.nested();
                    format!(
                        r#"{indentation}{{
{indentation_1}{ty} {name} = {val};
{pass}
{indentation}}}"#,
                        pass = gen_code_c(
                            indentation_1,
                            intrinsic,
                            preset_vals,
                            format!("{context} {name}={i}"),
                            mode,
                            is_aarch32
                        )
                    )
                })
                .join("\n")
        }
    } else {
        intrinsic.generate_loop_c(indentation, &context, PASSES, mode, is_aarch32)
    }
}

fn generate_c_program(
    notices: &str,
    header_files: &[&str],
    intrinsic: &Intrinsic,
    mode: Extension,
    is_aarch32: bool,
) -> String {
    let preset_vals = intrinsic
        .arguments
        .iter()
        .filter(|i| i.uses_set_values())
        .collect_vec();

    let indentation = Indentation::default();

    let neon_poly128_override = r#"#ifdef __aarch64__
std::ostream& operator<<(std::ostream& os, poly128_t value) {{
  std::stringstream temp;
  do {{
    int n = value % 10;
    value /= 10;
    temp << n;
  }} while (value != 0);
  std::string tempstr(temp.str());
  std::string res(tempstr.rbegin(), tempstr.rend());
  os << res;
  return os;
}}
#endif
"#;
    let indentation_1 = indentation.nested();
    let main_body = format!(
        r#"{results_array}
{element_count}

{passes}
{indentation_1}return 0;
}}"#,
        results_array = intrinsic.gen_results_array_c(indentation_1),
        element_count = intrinsic.gen_element_count_c(indentation_1, Language::C),
        passes = gen_code_c(
            indentation_1,
            intrinsic,
            preset_vals.as_slice(),
            Default::default(),
            mode,
            is_aarch32
        )
    );

    format!(
        r#"{notices}{header_files}
#include <iostream>
#include <cstring>
#include <iomanip>
#include <sstream>

template<typename T1, typename T2> T1 cast(T2 x) {{
  static_assert(sizeof(T1) == sizeof(T2), "sizeof T1 and T2 must be the same");
  T1 ret{{}};
  memcpy(&ret, &x, sizeof(T1));
  return ret;
}}

{neon_poly128_override}

{arglists}

int main(int argc, char **argv) {{
{body}"#,
        header_files = header_files
            .iter()
            .map(|header| format!("#include <{header}>"))
            .join("\n"),
        arglists = intrinsic.arguments.gen_arglists_c(indentation, PASSES),
        body = main_body,
        neon_poly128_override = if let Extension::NEON = mode {
            neon_poly128_override
        } else {
            ""
        }
    )
}

fn gen_code_rust(
    indentation: Indentation,
    intrinsic: &Intrinsic,
    preset_vals: &[&Argument],
    context: String,
    mode: Extension,
    is_aarch32: bool,
) -> String {
    if let Some((current, preset_vals)) = preset_vals.split_last() {
        let name = &current.name;
        if current.is_predicate() {
            current
                .get_predicate_decls(indentation, Language::Rust)
                .into_iter()
                .enumerate()
                .map(|(i, p)| {
                    format!(
                        r#"{p}
{pass}"#,
                        pass = gen_code_rust(
                            indentation,
                            intrinsic,
                            preset_vals,
                            format!("{context} {name}=pat{i}"),
                            mode,
                            is_aarch32
                        )
                    )
                })
                .join("\n")
        } else if current.ty.kind() == TypeKind::Bool {
            // Some bool intrinics nest quite deeply, so prefer looping
            format!(
                r"{indentation}for {name} in [true, false] {{
{pass}
{indentation}}}",
                pass = gen_code_rust(
                    indentation.nested(),
                    intrinsic,
                    preset_vals,
                    format!("{context} {name}={{{name}}}"),
                    mode,
                    is_aarch32
                )
            )
        } else {
            current
                .constraints
                .iter()
                .flat_map(|c| c.iter())
                .map(|i| {
                    let ty = current.ty.rust_type();
                    let val = if current.ty.kind().is_enum() {
                        // This is defined behaviour as enums in types.rs are `#[repr(i32)]`
                        // in order to facilitating passing them as const-generics
                        format!("unsafe {{ core::mem::transmute::<i32, _>({i}) }}")
                    } else {
                        i.to_string()
                    };
                    let indentation_1 = indentation.nested();
                    format!(
                        r#"{indentation}{{
{indentation_1}const {name}: {ty} = {val};
{pass}
{indentation}}}"#,
                        pass = gen_code_rust(
                            indentation_1,
                            intrinsic,
                            preset_vals,
                            format!("{context} {name}={i}"),
                            mode,
                            is_aarch32
                        )
                    )
                })
                .join("\n")
        }
    } else {
        intrinsic.generate_loop_rust(indentation, &context, PASSES, mode, is_aarch32)
    }
}

fn generate_rust_program(
    notices: &str,
    intrinsic: &Intrinsic,
    mode: Extension,
    is_aarch32: bool,
) -> String {
    let preset_vals = intrinsic
        .arguments
        .iter()
        .filter(|i| i.uses_set_values())
        .collect_vec();

    let indentation = Indentation::default();
    format!(
        r#"{notices}#![feature(simd_ffi)]
#![feature(link_llvm_intrinsics)]
#![feature(unsized_fn_params)]
#![feature(unsized_locals)]
#![cfg_attr(target_arch = "arm", feature(stdarch_arm_neon_intrinsics))]
#![feature(stdarch_arm_crc32)]
#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_fcma))]
#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_dotprod))]
#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_i8mm))]
#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_sha3))]
#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_sm4))]
#![cfg_attr(target_arch = "aarch64", feature(stdarch_neon_ftts))]
#![cfg_attr(target_arch = "aarch64", feature(stdarch_aarch64_sve))]
#![allow(non_upper_case_globals)]
#![allow(internal_features)]
#![allow(incomplete_features)]
use core_arch::arch::{target_arch}::{extension};

fn main() {{
{results_array}
{element_count}

{arglists}
{passes}
}}
"#,
        target_arch = if is_aarch32 { "arm" } else { "aarch64" },
        extension = if let Extension::SVE = mode {
            "sve::*"
        } else {
            "*"
        },
        arglists = intrinsic
            .arguments
            .gen_arglists_rust(indentation.nested(), PASSES),
        passes = gen_code_rust(
            indentation.nested(),
            intrinsic,
            &preset_vals,
            Default::default(),
            mode,
            is_aarch32
        ),
        results_array = intrinsic.gen_results_array_rust(indentation.nested()),
        element_count = intrinsic.gen_element_count_c(indentation.nested(), Language::Rust),
    )
}

fn compile_c(
    c_filename: &str,
    intrinsic: &Intrinsic,
    compiler: &str,
    mode: Extension,
    is_aarch32: bool,
) -> bool {
    let flags = std::env::var("CPPFLAGS").unwrap_or_default();
    let mut a64_archflags = String::from("-march=armv8.6-a+crypto+sha3+sm4+crc+dotprod");
    if let Extension::SVE = mode {
        a64_archflags.push_str("+sve2-aes+sve2-sm4+sve2-sha3+sve2-bitperm+f32mm+f64mm");
    }

    let output = Command::new("sh")
        .arg("-c")
        .arg(format!(
            // -ffp-contract=off emulates Rust's approach of not fusing separate mul-add operations
            "{cpp} {cppflags} {arch_flags} -ffp-contract=off -Wno-narrowing -O2 -target {target} -o c_programs/{intrinsic} {filename}",
            target = if is_aarch32 { "armv7-unknown-linux-gnueabihf" } else { "aarch64-unknown-linux-gnu" },
            arch_flags = if is_aarch32 { "-march=armv8.6-a+crypto+crc+dotprod" } else { a64_archflags.as_str() },
            filename = c_filename,
            intrinsic = intrinsic.name,
            cpp = compiler,
            cppflags = flags,
        )).output();
    if let Ok(output) = output {
        if output.status.success() {
            true
        } else {
            error!(
                "Failed to compile code for intrinsic: {}\n\nstdout:\n{}\n\nstderr:\n{}",
                intrinsic.name,
                std::str::from_utf8(&output.stdout).unwrap_or(""),
                std::str::from_utf8(&output.stderr).unwrap_or("")
            );
            false
        }
    } else {
        error!("Command failed: {:#?}", output);
        false
    }
}

fn build_notices(line_prefix: &str) -> String {
    format!(
        "\
{line_prefix}This is a transient test file, not intended for distribution. Some aspects of the
{line_prefix}test are derived from a JSON specification, published under the same license as the
{line_prefix}`intrinsic-test` crate.\n
"
    )
}

fn build_c(
    notices: &str,
    intrinsics: &Vec<Intrinsic>,
    compiler: Option<&str>,
    mode: Extension,
    is_aarch32: bool,
) -> bool {
    let _ = std::fs::create_dir("c_programs");
    intrinsics
        .par_iter()
        .map(|i| {
            let c_filename = format!(r#"c_programs/{}.cpp"#, i.name);
            let mut file = File::create(&c_filename).unwrap();

            let header = if let Extension::SVE = mode {
                "arm_sve.h"
            } else {
                "arm_neon.h"
            };
            let c_code = generate_c_program(notices, &[header, "arm_acle.h"], i, mode, is_aarch32);
            file.write_all(c_code.into_bytes().as_slice()).unwrap();
            match compiler {
                None => true,
                Some(compiler) => compile_c(&c_filename, i, compiler, mode, is_aarch32),
            }
        })
        .find_any(|x| !x)
        .is_none()
}

fn build_rust(
    notices: &str,
    intrinsics: &[Intrinsic],
    toolchain: Option<&str>,
    mode: Extension,
    is_aarch32: bool,
) -> bool {
    intrinsics.iter().for_each(|i| {
        let rust_dir = format!(r#"rust_programs/{}"#, i.name);
        let _ = std::fs::create_dir_all(&rust_dir);
        let rust_filename = format!(r#"{rust_dir}/main.rs"#);
        let mut file = File::create(rust_filename).unwrap();

        let c_code = generate_rust_program(notices, i, mode, is_aarch32);
        file.write_all(c_code.into_bytes().as_slice()).unwrap();
    });

    let mut cargo = File::create("rust_programs/Cargo.toml").unwrap();
    cargo
        .write_all(
            format!(
                r#"[package]
name = "intrinsic-test-programs"
version = "{version}"
authors = [{authors}]
license = "{license}"
edition = "2018"
[workspace]
[dependencies]
core_arch = {{ path = "../crates/core_arch" }}
{binaries}"#,
                version = env!("CARGO_PKG_VERSION"),
                authors = env!("CARGO_PKG_AUTHORS")
                    .split(":")
                    .format_with(", ", |author, fmt| fmt(&format_args!("\"{author}\""))),
                license = env!("CARGO_PKG_LICENSE"),
                binaries = intrinsics
                    .iter()
                    .map(|i| {
                        format!(
                            r#"[[bin]]
name = "{intrinsic}"
path = "{intrinsic}/main.rs""#,
                            intrinsic = i.name
                        )
                    })
                    .join("\n")
            )
            .into_bytes()
            .as_slice(),
        )
        .unwrap();

    let toolchain = match toolchain {
        None => return true,
        Some(t) => t,
    };

    let features = if mode == Extension::SVE {
        "-Ctarget-feature=+sve,+sve2,+sve2-aes,+sve2-sm4,+sve2-sha3,+sve2-bitperm,+f32mm,+f64mm"
    } else {
        ""
    };

    let output = Command::new("sh")
        .current_dir("rust_programs")
        .arg("-c")
        .arg(format!(
            "cargo {toolchain} build --target {target} --release",
            toolchain = toolchain,
            target = if is_aarch32 {
                "armv7-unknown-linux-gnueabihf"
            } else {
                "aarch64-unknown-linux-gnu"
            },
        ))
        .env("RUSTFLAGS", format!("-Cdebuginfo=0 {features}"))
        .output();
    if let Ok(output) = output {
        if output.status.success() {
            true
        } else {
            error!(
                "Failed to compile code for intrinsics\n\nstdout:\n{}\n\nstderr:\n{}",
                std::str::from_utf8(&output.stdout).unwrap_or(""),
                std::str::from_utf8(&output.stderr).unwrap_or("")
            );
            false
        }
    } else {
        error!("Command failed: {:#?}", output);
        false
    }
}

/// Intrinsic test tool
#[derive(clap::Parser)]
#[command(
    name = "Intrinsic test tool",
    about = "Generates Rust and C programs for intrinsics and compares the output"
)]
struct Cli {
    /// The input file containing the intrinsics
    input: PathBuf,

    /// The rust toolchain to use for building the rust code
    #[arg(long)]
    toolchain: Option<String>,

    /// The C++ compiler to use for compiling the c++ code
    #[arg(long, default_value_t = String::from("clang++"))]
    cppcompiler: String,

    /// Run the C programs under emulation with this command
    #[arg(long)]
    runner: Option<String>,

    /// Filename for a list of intrinsics to skip (one per line)
    #[arg(long)]
    skip: Option<PathBuf>,

    /// Run tests for A32 instrinsics instead of A64
    #[arg(long)]
    a32: bool,

    /// Regenerate test programs, but don't build or run them
    #[arg(long)]
    generate_only: bool,

    /// Run tests for SVE instead of Neon
    #[arg(long)]
    sve: bool,
}

fn main() {
    pretty_env_logger::init();

    let args: Cli = clap::Parser::parse();

    let filename = args.input;
    let c_runner = args.runner.unwrap_or_else(String::new);
    let skip = if let Some(filename) = args.skip {
        let data = std::fs::read_to_string(&filename).expect("Failed to open file");
        data.lines()
            .map(str::trim)
            .filter(|s| !s.contains('#'))
            .map(String::from)
            .collect_vec()
    } else {
        Default::default()
    };

    let a32 = args.a32;

    let (mode, mut intrinsics) = if args.sve {
        (
            Extension::SVE,
            get_sve_intrinsics(&filename).expect("Error parsing input file"),
        )
    } else {
        (
            Extension::NEON,
            get_neon_intrinsics(&filename).expect("Error parsing input file"),
        )
    };

    intrinsics.sort_by(|a, b| a.name.cmp(&b.name));

    let mut intrinsics = intrinsics
        .into_iter()
        // Void intrinsics consist of stores, prefetch and svwrffr, all of which we can't test here
        .filter(|i| i.results.kind() != TypeKind::Void)
        // Most pointer intrinsics access memory, which we handle with separate tests
        .filter(|i| {
            !i.arguments.iter().any(|a| a.is_ptr())
                || i.name.starts_with("svwhilewr")
                || i.name.starts_with("svwhilerw")
        })
        // Bases arguments are really pointers, but memory isn't accessed for address calculation
        // intrinsics
        .filter(|i| !i.arguments.iter().any(|a| a.name == "bases") || i.name.starts_with("svadr"))
        .filter(|i| {
            !i.arguments
                .iter()
                .any(|a| !a.is_predicate() && a.ty.inner_size() == 128)
        })
        .filter(|i| !skip.contains(&i.name))
        .filter(|i| !(a32 && i.a64_only))
        .collect::<Vec<_>>();

    intrinsics.dedup();
    println!("Testing {} intrinsics", intrinsics.len());

    let (toolchain, cpp_compiler) = if args.generate_only {
        (None, None)
    } else {
        (
            Some(args.toolchain.map_or_else(String::new, |t| format!("+{t}"))),
            Some(args.cppcompiler),
        )
    };

    let notices = build_notices("// ");

    if !build_c(&notices, &intrinsics, cpp_compiler.as_deref(), mode, a32) {
        std::process::exit(2);
    }

    if !build_rust(&notices, &intrinsics, toolchain.as_deref(), mode, a32) {
        std::process::exit(3);
    }

    if let Some(ref toolchain) = toolchain {
        if !compare_outputs(&intrinsics, toolchain, &c_runner, mode, a32) {
            std::process::exit(1)
        }
    }
}

enum FailureReason {
    RunC(String),
    RunRust(String),
    Difference(String, String, String),
}

fn compare_outputs(
    intrinsics: &Vec<Intrinsic>,
    toolchain: &str,
    runner: &str,
    mode: Extension,
    is_aarch32: bool,
) -> bool {
    let features = if mode == Extension::SVE {
        "-Ctarget-feature=+sve,+sve2,+sve2-aes,+sve2-sm4,+sve2-sha3,+sve2-bitperm,+f32mm,+f64mm"
    } else {
        ""
    };
    let intrinsics = intrinsics
        .par_iter()
        .filter_map(|intrinsic| {
            let c = Command::new("sh")
                .arg("-c")
                .arg(format!(
                    "{runner} ./c_programs/{intrinsic}",
                    runner = runner,
                    intrinsic = intrinsic.name,
                ))
                .output();
            let rust = Command::new("sh")
                .current_dir("rust_programs")
                .arg("-c")
                .arg(format!(
                    "cargo {toolchain} run --target {target} --bin {intrinsic} --release",
                    intrinsic = intrinsic.name,
                    toolchain = toolchain,
                    target = if is_aarch32 {
                        "armv7-unknown-linux-gnueabihf"
                    } else {
                        "aarch64-unknown-linux-gnu"
                    },
                ))
                .env("RUSTFLAGS", format!("-Cdebuginfo=0 {features}"))
                .output();

            let (c, rust) = match (c, rust) {
                (Ok(c), Ok(rust)) => (c, rust),
                a => panic!("{a:#?}"),
            };

            if !c.status.success() {
                error!("Failed to run C program for intrinsic {}", intrinsic.name);
                error!("stdout: {}", std::str::from_utf8(&c.stdout).unwrap());
                error!("stderr: {}", std::str::from_utf8(&c.stderr).unwrap());
                return Some(FailureReason::RunC(intrinsic.name.clone()));
            }

            if !rust.status.success() {
                error!(
                    "Failed to run rust program for intrinsic {}",
                    intrinsic.name
                );
                error!("stdout: {}", std::str::from_utf8(&rust.stdout).unwrap());
                error!("stderr: {}", std::str::from_utf8(&rust.stderr).unwrap());
                return Some(FailureReason::RunRust(intrinsic.name.clone()));
            }

            info!("Comparing intrinsic: {}", intrinsic.name);

            let c = std::str::from_utf8(&c.stdout)
                .unwrap()
                .to_lowercase()
                .replace("-nan", "nan");
            let rust = std::str::from_utf8(&rust.stdout)
                .unwrap()
                .to_lowercase()
                .replace("-nan", "nan");

            if c == rust {
                None
            } else {
                Some(FailureReason::Difference(intrinsic.name.clone(), c, rust))
            }
        })
        .collect::<Vec<_>>();

    intrinsics.iter().for_each(|reason| match reason {
        FailureReason::Difference(intrinsic, c, rust) => {
            println!("Difference for intrinsic: {intrinsic}");
            let diff = diff::lines(c, rust);
            diff.iter().for_each(|diff| match diff {
                diff::Result::Left(c) => println!("C: {c}"),
                diff::Result::Right(rust) => println!("Rust: {rust}"),
                diff::Result::Both(_, _) => (),
            });
            println!("****************************************************************");
        }
        FailureReason::RunC(intrinsic) => {
            println!("Failed to run C program for intrinsic {intrinsic}")
        }
        FailureReason::RunRust(intrinsic) => {
            println!("Failed to run rust program for intrinsic {intrinsic}")
        }
    });
    println!("{} differences found", intrinsics.len());
    intrinsics.is_empty()
}
