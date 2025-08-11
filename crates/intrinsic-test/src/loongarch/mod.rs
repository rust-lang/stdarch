mod intrinsic;
mod parser;
mod types;

use std::fs::{self, File};

use rayon::prelude::*;

use crate::common::intrinsic_helpers::TypeKind;
use crate::common::SupportedArchitectureTest;
use crate::common::cli::ProcessedCli;
use crate::common::compare::compare_outputs;

use crate::common::intrinsic::Intrinsic;
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
        unimplemented!("build_c_file of LoongArchIntrinsicType is not defined!")
    }

    fn build_rust_file(&self) -> bool {
        unimplemented!("build_rust_file of LoongArchIntrinsicType is not defined!")
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
