mod intrinsic;
mod types;

use std::fs::{self, File};

use rayon::prelude::*;

use crate::common::SupportedArchitectureTest;
use crate::common::cli::ProcessedCli;
use crate::common::compare::compare_outputs;

use crate::common::intrinsic::Intrinsic;
use intrinsic::LoongArchIntrinsicType;

pub struct LoongArchArchitectureTest {
    intrinsics: Vec<Intrinsic<LoongArchIntrinsicType>>,
    cli_options: ProcessedCli,
}

impl SupportedArchitectureTest for LoongArchArchitectureTest {
    fn create(cli_options: ProcessedCli) -> Box<Self> {
        unimplemented!("create of LoongArchIntrinsicType is not defined!")
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
