mod intrinsic;
mod types;
mod xml_parser;

use crate::common::SupportedArchitectureTest;
use crate::common::cli::ProcessedCli;
use crate::common::intrinsic::Intrinsic;
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
        todo!("build_c_file in X86ArchitectureTest is not implemented")
    }

    fn build_rust_file(&self) -> bool {
        todo!("build_rust_file in X86ArchitectureTest is not implemented")
    }

    fn compare_outputs(&self) -> bool {
        todo!("compare_outputs in X86ArchitectureTest is not implemented")
    }
}
