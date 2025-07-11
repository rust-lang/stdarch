mod compile;
mod config;
mod intrinsic;
mod json_parser;
mod types;

use std::fs::File;

use rayon::prelude::*;

use crate::arm::config::POLY128_OSTREAM_DEF;
use crate::common::SupportedArchitectureTest;
use crate::common::cli::ProcessedCli;
use crate::common::compare::compare_outputs;
use crate::common::gen_c::{write_main_cpp, write_mod_cpp};
use crate::common::gen_rust::{
    compile_rust_programs, write_bin_cargo_toml, write_lib_cargo_toml, write_main_rs,
};
use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_helpers::TypeKind;
use config::{AARCH_CONFIGURATIONS, F16_FORMATTING_DEF, build_notices};
use intrinsic::ArmIntrinsicType;
use json_parser::get_neon_intrinsics;

pub struct ArmArchitectureTest {
    intrinsics: Vec<Intrinsic<ArmIntrinsicType>>,
    cli_options: ProcessedCli,
}

impl SupportedArchitectureTest for ArmArchitectureTest {
    fn create(cli_options: ProcessedCli) -> Box<Self> {
        let a32 = cli_options.target.contains("v7");
        let mut intrinsics = get_neon_intrinsics(&cli_options.filename, &cli_options.target)
            .expect("Error parsing input file");

        intrinsics.sort_by(|a, b| a.name.cmp(&b.name));

        let mut intrinsics = intrinsics
            .into_iter()
            // Not sure how we would compare intrinsic that returns void.
            .filter(|i| i.results.kind() != TypeKind::Void)
            .filter(|i| i.results.kind() != TypeKind::BFloat)
            .filter(|i| !i.arguments.iter().any(|a| a.ty.kind() == TypeKind::BFloat))
            // Skip pointers for now, we would probably need to look at the return
            // type to work out how many elements we need to point to.
            .filter(|i| !i.arguments.iter().any(|a| a.is_ptr()))
            .filter(|i| !i.arguments.iter().any(|a| a.ty.inner_size() == 128))
            .filter(|i| !cli_options.skip.contains(&i.name))
            .filter(|i| !(a32 && i.arch_tags == vec!["A64".to_string()]))
            .collect::<Vec<_>>();
        intrinsics.dedup();

        Box::new(Self {
            intrinsics,
            cli_options,
        })
    }

    fn build_c_file(&self) -> bool {
        let c_target = "aarch64";
        let platform_headers = &["arm_neon.h", "arm_acle.h", "arm_fp16.h"];

        let available_parallelism = std::thread::available_parallelism().unwrap().get();
        let chunk_count = Ord::min(available_parallelism, self.intrinsics.len());
        let chunk_size = self.intrinsics.len().div_ceil(chunk_count);

        let cpp_compiler = compile::configure_cpp_compiler(&self.cli_options).unwrap();

        let notice = &build_notices("// ");
        self.intrinsics
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                let c_filename = format!("c_programs/mod_{i}.cpp");
                info!("writing {c_filename}");
                let mut file = File::create(&c_filename).unwrap();
                write_mod_cpp(&mut file, notice, c_target, platform_headers, chunk).unwrap();

                // compile this cpp file into a .o file
                info!("compiling {c_filename}");
                let output = cpp_compiler
                    .compile_object_file(&format!("mod_{i}.cpp"), &format!("mod_{i}.o"))?;
                assert!(output.status.success());
                info!("done compiling {c_filename}");

                Ok(())
            })
            .collect::<Result<(), std::io::Error>>()
            .unwrap();

        info!("writing main.cpp");
        let mut file = File::create("c_programs/main.cpp").unwrap();
        write_main_cpp(
            &mut file,
            c_target,
            POLY128_OSTREAM_DEF,
            self.intrinsics.iter().map(|i| i.name.as_str()),
        )
        .unwrap();

        // compile this cpp file into a .o file
        info!("compiling main.cpp");
        let output = cpp_compiler
            .compile_object_file("main.cpp", "intrinsic-test-programs.o")
            .unwrap();
        assert!(output.status.success());

        let object_files = (0..chunk_count)
            .map(|i| format!("mod_{i}.o"))
            .chain(["intrinsic-test-programs.o".to_owned()]);

        let output = match &self.cli_options.linker {
            Some(custom_linker) => {
                let mut linker = std::process::Command::new(custom_linker);

                linker.current_dir("c_programs");

                linker.args(object_files);
                linker.args(["-o", "intrinsic-test-programs"]);

                info!("linking final C binary with {custom_linker}");
                linker.output()
            }
            None => cpp_compiler.link_executable(object_files, "intrinsic-test-programs"),
        };

        let output = output.unwrap();
        assert!(output.status.success());

        true
    }

    fn build_rust_file(&self) -> bool {
        std::fs::create_dir_all("rust_programs/src").unwrap();

        let architecture = if self.cli_options.target.contains("v7") {
            "arm"
        } else {
            "aarch64"
        };

        // Experimentally, keeping 2 cores free is fastest for cargo.
        let available_parallelism = std::thread::available_parallelism().unwrap().get();
        let chunk_count = Ord::min(available_parallelism, self.intrinsics.len());
        let chunk_size = self.intrinsics.len().div_ceil(chunk_count);

        let mut cargo = File::create("rust_programs/Cargo.toml").unwrap();
        write_bin_cargo_toml(&mut cargo, chunk_count).unwrap();

        let mut main_rs = File::create("rust_programs/src/main.rs").unwrap();
        write_main_rs(
            &mut main_rs,
            chunk_count,
            AARCH_CONFIGURATIONS,
            "",
            self.intrinsics.iter().map(|i| i.name.as_str()),
        )
        .unwrap();

        let target = &self.cli_options.target;
        let toolchain = self.cli_options.toolchain.as_deref();
        let linker = self.cli_options.linker.as_deref();

        let notice = &build_notices("// ");
        self.intrinsics
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                use std::io::Write;

                std::fs::create_dir_all(format!("rust_programs/mod_{i}/src"))?;

                let rust_filename = format!("rust_programs/mod_{i}/src/lib.rs");
                trace!("generating `{rust_filename}`");
                let mut file = File::create(rust_filename).unwrap();

                write!(file, "{notice}")?;

                writeln!(file, "#![feature(simd_ffi)]")?;
                writeln!(file, "#![feature(f16)]")?;
                writeln!(file, "#![allow(unused)]")?;

                // Cargo will spam the logs if these warnings are not silenced.
                writeln!(file, "#![allow(non_upper_case_globals)]")?;
                writeln!(file, "#![allow(non_camel_case_types)]")?;
                writeln!(file, "#![allow(non_snake_case)]")?;

                let cfg = AARCH_CONFIGURATIONS;
                writeln!(file, "{cfg}")?;

                writeln!(file, "use core_arch::arch::{architecture}::*;")?;

                let definitions = F16_FORMATTING_DEF;
                writeln!(file, "{definitions}")?;

                for intrinsic in chunk {
                    crate::common::gen_rust::create_rust_test_module(&mut file, intrinsic)?;
                }

                let toml_filename = format!("rust_programs/mod_{i}/Cargo.toml");
                trace!("generating `{toml_filename}`");
                let mut file = File::create(toml_filename).unwrap();

                write_lib_cargo_toml(&mut file, &format!("mod_{i}"))?;

                Ok(())
            })
            .collect::<Result<(), std::io::Error>>()
            .unwrap();

        compile_rust_programs(toolchain, target, linker)
    }

    fn compare_outputs(&self) -> bool {
        if self.cli_options.toolchain.is_some() {
            let intrinsics_name_list = self
                .intrinsics
                .iter()
                .map(|i| i.name.clone())
                .collect::<Vec<_>>();

            compare_outputs(
                &intrinsics_name_list,
                &self.cli_options.c_runner,
                &self.cli_options.target,
            )
        } else {
            true
        }
    }
}
