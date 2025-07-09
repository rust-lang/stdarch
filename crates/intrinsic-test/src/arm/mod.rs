use std::fs::File;

use rayon::prelude::*;

mod compile;
mod config;
mod intrinsic;
mod json_parser;
mod types;

use crate::common::SupportedArchitectureTest;
use crate::common::cli::ProcessedCli;
use crate::common::compare::compare_outputs;
use crate::common::gen_c::{write_main_cpp, write_mod_cpp};
use crate::common::gen_rust::{compile_rust_programs, write_cargo_toml, write_main_rs};
use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_helpers::TypeKind;
use config::{AARCH_CONFIGURATIONS, F16_FORMATTING_DEF, POLY128_OSTREAM_DEF, build_notices};
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
        let compiler = self.cli_options.cpp_compiler.as_deref();
        let target = &self.cli_options.target;
        let cxx_toolchain_dir = self.cli_options.cxx_toolchain_dir.as_deref();
        let c_target = "aarch64";

        let available_parallelism = std::thread::available_parallelism().unwrap().get();
        let chunk_size = self.intrinsics.len().div_ceil(available_parallelism);

        let notice = &build_notices("// ");
        self.intrinsics
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                let c_filename = format!("c_programs/mod_{i}.cpp");
                let mut file = File::create(&c_filename).unwrap();
                write_mod_cpp(&mut file, &notice, chunk).unwrap();

                // compile this cpp file into a .o file

                // clang++ -march=armv8.6-a+crypto+crc+dotprod+fp16+faminmax+lut+sha3 -O2 -ffp-contract=off -Wno-narrowing --target=aarch64-unknown-linux-gnu c_file_0.c -c
                let mut cmd = std::process::Command::new("clang++");
                cmd.current_dir("c_programs");

                cmd.arg("-march=armv8.6-a+crypto+crc+dotprod+fp16+faminmax+lut+sha3");
                cmd.arg("-O2");
                cmd.arg("-ffp-contract=off");
                cmd.arg("-Wno-narrowing");
                cmd.arg("--target=aarch64-unknown-linux-gnu");
                cmd.arg("-c");
                cmd.arg(format!("mod_{i}.cpp"));

                let output = cmd.output();
                eprintln!(
                    "{}",
                    String::from_utf8_lossy(&output.as_ref().unwrap().stderr)
                );
                assert!(output.unwrap().status.success());

                Ok(())
            })
            .collect::<Result<(), std::io::Error>>()
            .unwrap();

        let c_filename = format!("c_programs/main.cpp");
        let mut file = File::create(&c_filename).unwrap();
        write_main_cpp(
            &mut file,
            c_target,
            POLY128_OSTREAM_DEF,
            self.intrinsics.iter().map(|i| i.name.as_str()),
        )
        .unwrap();

        let mut cmd = std::process::Command::new("clang++");
        cmd.current_dir("c_programs");

        cmd.arg("-march=armv8.6-a+crypto+crc+dotprod+fp16+faminmax+lut+sha3");
        cmd.arg("-O2");
        cmd.arg("-ffp-contract=off");
        cmd.arg("-Wno-narrowing");
        cmd.arg("--target=aarch64-unknown-linux-gnu");
        cmd.arg(format!("main.cpp"));
        for i in 0..Ord::min(available_parallelism, self.intrinsics.len()) {
            cmd.arg(format!("mod_{i}.o"));
        }
        cmd.args(&["-o", "intrinsic-test-programs"]);

        let output = cmd.output();
        eprintln!(
            "{}",
            String::from_utf8_lossy(&output.as_ref().unwrap().stderr)
        );
        assert!(output.unwrap().status.success());

        //        match compiler {
        //            None => true,
        //            Some(compiler) => compile_c_arm(
        //                intrinsics_name_list.unwrap().as_slice(),
        //                compiler,
        //                target,
        //                cxx_toolchain_dir,
        //            ),
        //        }

        true
    }

    fn build_rust_file(&self) -> bool {
        std::fs::create_dir_all("rust_programs/src").unwrap();

        let architecture = if self.cli_options.target.contains("v7") {
            "arm"
        } else {
            "aarch64"
        };

        let available_parallelism = std::thread::available_parallelism().unwrap().get();
        let chunk_size = self.intrinsics.len().div_ceil(available_parallelism);

        let mut cargo = File::create("rust_programs/Cargo.toml").unwrap();
        write_cargo_toml(&mut cargo, &[]).unwrap();

        let mut main_rs = File::create("rust_programs/src/main.rs").unwrap();
        write_main_rs(
            &mut main_rs,
            available_parallelism,
            architecture,
            AARCH_CONFIGURATIONS,
            F16_FORMATTING_DEF,
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

                let rust_filename = format!("rust_programs/src/mod_{i}.rs");
                let mut file = File::create(rust_filename).unwrap();

                write!(file, "{notice}")?;

                writeln!(file, "use core_arch::arch::{architecture}::*;")?;
                writeln!(file, "use crate::{{debug_simd_finish, debug_f16}};")?;

                for intrinsic in chunk {
                    crate::common::gen_rust::create_rust_test_module(&mut file, intrinsic)?;
                }

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
                &self.cli_options.runner,
                &self.cli_options.target,
            )
        } else {
            true
        }
    }
}
