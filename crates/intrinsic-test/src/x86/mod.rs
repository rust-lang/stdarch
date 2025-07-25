mod compile;
mod config;
mod constraint;
mod intrinsic;
mod types;
mod xml_parser;

use rayon::prelude::*;
use std::fs;

use crate::common::cli::ProcessedCli;
use crate::common::compare::compare_outputs;
use crate::common::gen_c::{write_main_cpp, write_mod_cpp};
use crate::common::gen_rust::compile_rust_programs;
use crate::common::intrinsic::{Intrinsic, IntrinsicDefinition};
use crate::common::intrinsic_helpers::TypeKind;
use crate::common::write_file::write_rust_testfiles;
use crate::common::{SupportedArchitectureTest, chunk_info};
use crate::x86::config::{F16_FORMATTING_DEF, X86_CONFIGURATIONS};
use config::build_notices;
use intrinsic::X86IntrinsicType;
use xml_parser::get_xml_intrinsics;

pub struct X86ArchitectureTest {
    intrinsics: Vec<Intrinsic<X86IntrinsicType>>,
    cli_options: ProcessedCli,
}

impl SupportedArchitectureTest for X86ArchitectureTest {
    fn create(cli_options: ProcessedCli) -> Box<Self> {
        let intrinsics =
            get_xml_intrinsics(&cli_options.filename).expect("Error parsing input file");

        let mut intrinsics = intrinsics
            .into_iter()
            // Not sure how we would compare intrinsic that returns void.
            .filter(|i| i.results.kind() != TypeKind::Void)
            .filter(|i| i.results.kind() != TypeKind::BFloat)
            .filter(|i| i.arguments().args.len() > 0)
            .filter(|i| !i.arguments.iter().any(|a| a.ty.kind() == TypeKind::BFloat))
            // Skip pointers for now, we would probably need to look at the return
            // type to work out how many elements we need to point to.
            .filter(|i| !i.arguments.iter().any(|a| a.is_ptr()))
            .filter(|i| !i.arguments.iter().any(|a| a.ty.inner_size() == 128))
            .filter(|i| !cli_options.skip.contains(&i.name))
            .collect::<Vec<_>>();

        intrinsics.sort_by(|a, b| a.name.cmp(&b.name));
        Box::new(Self {
            intrinsics: intrinsics,
            cli_options: cli_options,
        })
    }

    fn build_c_file(&self) -> bool {
        let c_target = "x86_64";
        let (chunk_size, chunk_count) = chunk_info(self.intrinsics.len());
        let notice = &build_notices("// ");
        let platform_headers = &["immintrin.h"];

        let cpp_compiler = compile::build_cpp_compilation(&self.cli_options);

        match fs::exists("c_programs") {
            Ok(false) => fs::create_dir("c_programs").unwrap(),
            Ok(true) => {}
            _ => return false,
        }

        self.intrinsics
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                let c_filename = format!("c_programs/mod_{i}.cpp");
                let mut file = fs::File::create(&c_filename).unwrap();
                write_mod_cpp(&mut file, notice, c_target, platform_headers, chunk).unwrap();

                // compile this cpp file into a .o file
                if let Some(compiler) = cpp_compiler.as_ref() {
                    let output = compiler
                        .compile_object_file(&format!("mod_{i}.cpp"), &format!("mod_{i}.o"))?;
                    assert!(output.status.success(), "{output:?}");
                }
                Ok(())
            })
            .collect::<Result<(), std::io::Error>>()
            .unwrap();

        let mut file = fs::File::create("c_programs/main.cpp").unwrap();
        write_main_cpp(
            &mut file,
            c_target,
            "",
            Vec::from(platform_headers),
            self.intrinsics.iter().map(|i| i.name.as_str()),
        )
        .unwrap();

        // compile this cpp file into a .o file
        if let Some(compiler) = cpp_compiler.as_ref() {
            info!("compiling main.cpp");
            let output = compiler
                .compile_object_file("main.cpp", "intrinsic-test-programs.o")
                .unwrap();
            assert!(output.status.success(), "{output:?}");

            let object_files = (0..chunk_count)
                .map(|i| format!("mod_{i}.o"))
                .chain(["intrinsic-test-programs.o".to_owned()]);

            let output = compiler
                .link_executable(object_files, "intrinsic-test-programs")
                .unwrap();
            assert!(output.status.success(), "{output:?}");
        }

        true
    }

    fn build_rust_file(&self) -> bool {
        // this is the module that handles the specific intrinsic
        // within the core_arch crate in std::arch
        let rust_target = if self.cli_options.target.contains("v7") {
            "x86"
        } else {
            "x86_64"
        };
        let target = &self.cli_options.target;
        let toolchain = self.cli_options.toolchain.as_deref();
        let linker = self.cli_options.linker.as_deref();
        let intrinsics_name_list = write_rust_testfiles(
            self.intrinsics
                .iter()
                .map(|i| i as &dyn IntrinsicDefinition<_>)
                .collect::<Vec<_>>(),
            rust_target,
            &build_notices("// "),
            F16_FORMATTING_DEF,
            X86_CONFIGURATIONS,
        );

        compile_rust_programs(intrinsics_name_list, toolchain, target, linker)
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
