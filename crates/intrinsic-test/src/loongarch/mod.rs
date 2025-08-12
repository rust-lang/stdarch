mod compile;
mod config;
mod intrinsic;
mod parser;
mod types;

use std::fs::{self, File};

use rayon::prelude::*;

use crate::common::cli::ProcessedCli;
use crate::common::compare::compare_outputs;
use crate::common::gen_c::{write_main_cpp, write_mod_cpp};
use crate::common::gen_rust::{
    compile_rust_programs, write_bin_cargo_toml, write_lib_cargo_toml, write_lib_rs, write_main_rs,
};
use crate::common::intrinsic_helpers::TypeKind;
use crate::common::{SupportedArchitectureTest, chunk_info};

use crate::common::intrinsic::Intrinsic;
use crate::loongarch::config::{F16_FORMATTING_DEF, LOONGARCH_CONFIGURATIONS, build_notices};
use crate::loongarch::parser::get_loongson_intrinsics;
use intrinsic::LoongArchIntrinsicType;

pub struct LoongArchArchitectureTest {
    intrinsics: Vec<Intrinsic<LoongArchIntrinsicType>>,
    cli_options: ProcessedCli,
}

impl SupportedArchitectureTest for LoongArchArchitectureTest {
    fn create(cli_options: ProcessedCli) -> Box<Self> {
        let mut intrinsics = get_loongson_intrinsics(&cli_options.filename, &cli_options.target)
            .expect("Error parsing input file");

        intrinsics.sort_by(|a, b| a.name.cmp(&b.name));

        let mut intrinsics = intrinsics
            .into_iter()
            .filter(|i| i.results.kind() != TypeKind::Void)
            .filter(|i| !i.arguments.iter().any(|a| a.is_ptr()))
            .filter(|i| !cli_options.skip.contains(&i.name))
            .collect::<Vec<_>>();
        intrinsics.dedup();

        Box::new(Self {
            intrinsics,
            cli_options,
        })
    }

    fn build_c_file(&self) -> bool {
        let c_target = "loongarch";
        let platform_headers = &["lasxintrin.h", "lsxintrin.h"];

        let (chunk_size, chunk_count) = chunk_info(self.intrinsics.len());

        let cpp_compiler_wrapped = compile::build_cpp_compilation(&self.cli_options);

        let notice = &build_notices("// ");
        fs::create_dir_all("c_programs").unwrap();
        self.intrinsics
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                let c_filename = format!("c_programs/mod_{i}.cpp");
                let mut file = File::create(&c_filename).unwrap();
                write_mod_cpp(&mut file, notice, c_target, platform_headers, chunk).unwrap();

                // compile this cpp file into a .o file.
                //
                // This is done because `cpp_compiler_wrapped` is None when
                // the --generate-only flag is passed
                if let Some(cpp_compiler) = cpp_compiler_wrapped.as_ref() {
                    let output = cpp_compiler
                        .compile_object_file(&format!("mod_{i}.cpp"), &format!("mod_{i}.o"))?;
                    assert!(output.status.success(), "{output:?}");
                }

                Ok(())
            })
            .collect::<Result<(), std::io::Error>>()
            .unwrap();

        let mut file = File::create("c_programs/main.cpp").unwrap();
        write_main_cpp(
            &mut file,
            c_target,
            "\n",
            self.intrinsics.iter().map(|i| i.name.as_str()),
        )
        .unwrap();

        // This is done because `cpp_compiler_wrapped` is None when
        // the --generate-only flag is passed
        if let Some(cpp_compiler) = cpp_compiler_wrapped.as_ref() {
            // compile this cpp file into a .o file
            info!("compiling main.cpp");
            let output = cpp_compiler
                .compile_object_file("main.cpp", "intrinsic-test-programs.o")
                .unwrap();
            assert!(output.status.success(), "{output:?}");

            let object_files = (0..chunk_count)
                .map(|i| format!("mod_{i}.o"))
                .chain(["intrinsic-test-programs.o".to_owned()]);

            let output = cpp_compiler
                .link_executable(object_files, "intrinsic-test-programs")
                .unwrap();
            assert!(output.status.success(), "{output:?}");
        }

        true
    }

    fn build_rust_file(&self) -> bool {
        std::fs::create_dir_all("rust_programs/src").unwrap();

        let architecture = "loongarch64";

        let (chunk_size, chunk_count) = chunk_info(self.intrinsics.len());

        let mut cargo = File::create("rust_programs/Cargo.toml").unwrap();
        write_bin_cargo_toml(&mut cargo, chunk_count).unwrap();

        let mut main_rs = File::create("rust_programs/src/main.rs").unwrap();
        write_main_rs(
            &mut main_rs,
            chunk_count,
            LOONGARCH_CONFIGURATIONS,
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
                std::fs::create_dir_all(format!("rust_programs/mod_{i}/src"))?;

                let rust_filename = format!("rust_programs/mod_{i}/src/lib.rs");
                trace!("generating `{rust_filename}`");
                let mut file = File::create(rust_filename)?;

                let cfg = LOONGARCH_CONFIGURATIONS;
                let definitions = F16_FORMATTING_DEF;
                write_lib_rs(&mut file, architecture, notice, cfg, definitions, chunk)?;

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
                &self.cli_options.runner,
                &self.cli_options.target,
            )
        } else {
            true
        }
    }
}
