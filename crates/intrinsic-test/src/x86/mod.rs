mod config;
mod intrinsic;
mod types;
mod xml_parser;

use crate::common::SupportedArchitectureTest;
use crate::common::cli::ProcessedCli;
use crate::common::intrinsic::{Intrinsic, IntrinsicDefinition};
use crate::common::write_file::{write_c_testfiles, write_rust_testfiles};
use config::build_notices;
use intrinsic::X86IntrinsicType;
use xml_parser::get_xml_intrinsics;

pub struct X86ArchitectureTest {
    intrinsics: Vec<Intrinsic<X86IntrinsicType>>,
    cli_options: ProcessedCli,
}

impl SupportedArchitectureTest for X86ArchitectureTest {
    fn create(cli_options: ProcessedCli) -> Box<Self> {
        let intrinsics = get_xml_intrinsics(&cli_options.filename, &cli_options.target)
            .expect("Error parsing input file");

        Box::new(Self {
            intrinsics: intrinsics,
            cli_options: cli_options,
        })
    }

    fn build_c_file(&self) -> bool {
        let compiler = self.cli_options.cpp_compiler.as_deref();
        let target = &self.cli_options.target;
        let cxx_toolchain_dir = self.cli_options.cxx_toolchain_dir.as_deref();
        let c_target = "x86_64";

        let intrinsics_name_list = write_c_testfiles(
            &self
                .intrinsics
                .iter()
                .map(|i| i as &dyn IntrinsicDefinition<_>)
                .collect::<Vec<_>>(),
            target,
            c_target,
            &["immintrin.h"],
            &build_notices("// "),
            &[],
        );

        true
    }

    fn build_rust_file(&self) -> bool {
        todo!("build_rust_file in X86ArchitectureTest is not implemented")
    }

    fn compare_outputs(&self) -> bool {
        todo!("compare_outputs in X86ArchitectureTest is not implemented")
    }
}
