mod config;
mod intrinsic;
mod types;
mod xml_parser;

use crate::common::SupportedArchitectureTest;
use crate::common::cli::ProcessedCli;
use crate::common::gen_rust::compile_rust_programs;
use crate::common::intrinsic::{Intrinsic, IntrinsicDefinition};
use crate::common::intrinsic_helpers::TypeKind;
use crate::common::write_file::{write_c_testfiles, write_rust_testfiles};
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
        let mut intrinsics =
            get_xml_intrinsics(&cli_options.filename).expect("Error parsing input file");

        intrinsics.sort_by(|a, b| a.name.cmp(&b.name));
        let intrinsics = intrinsics
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

        Box::new(Self {
            intrinsics: intrinsics,
            cli_options: cli_options,
        })
    }

    fn build_c_file(&self) -> bool {
        // let compiler = self.cli_options.cpp_compiler.as_deref();
        let target = &self.cli_options.target;
        // let cxx_toolchain_dir = self.cli_options.cxx_toolchain_dir.as_deref();
        let c_target = "x86_64";

        /* let intrinsics_name_list = */
        write_c_testfiles(
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
        todo!("compare_outputs in X86ArchitectureTest is not implemented")
    }
}
