mod intrinsic;
mod parser;
mod types;

use std::path::Path;

use crate::common::SupportedArchitecture;
use crate::common::cli::ProcessedCli;
use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_helpers::TypeKind;
use crate::riscv::parser::get_intrinsics;
use intrinsic::RiscvIntrinsicType;

#[derive(PartialEq)]
pub struct Riscv {
    intrinsics: Vec<Intrinsic<Riscv>>,
}

impl SupportedArchitecture for Riscv {
    type Type = RiscvIntrinsicType;

    fn intrinsics(&self) -> &[Intrinsic<Self>] {
        &self.intrinsics
    }

    fn create(cli_options: &ProcessedCli) -> Self {
        let rv32 = cli_options.target.starts_with("riscv32");

        let mut intrinsics =
            load_intrinsics(&cli_options.filename).expect("Error parsing input file");

        intrinsics.sort_by(|a, b| a.name.cmp(&b.name));
        intrinsics.dedup_by(|a, b| {
            a.name == b.name && a.results == b.results && a.arguments == b.arguments
        });

        let sample_percentage: usize = cli_options.sample_percentage as usize;
        let sample_size = (intrinsics.len() * sample_percentage) / 100;

        let intrinsics = intrinsics
            .into_iter()
            // Skip intrinsics that don't return a value.
            .filter(|intrinsic| intrinsic.results.kind() != TypeKind::Void)
            .filter(|intrinsic| !intrinsic.arguments.args.is_empty())
            .filter(|intrinsic| !cli_options.skip.contains(&intrinsic.name))
            .filter(|intrinsic| {
                intrinsic.arch_tags.is_empty() || ((intrinsic.arch_tags[0] == "rv64") ^ rv32)
            })
            .take(sample_size)
            .collect::<Vec<_>>();

        Self { intrinsics }
    }

    const NOTICE: &str = r#"
// This is a transient test file, not intended for distribution. Some aspects of the
// test are derived from RISC-V specification files, published under the same license as the
// `intrinsic-test` crate.
    "#;

    const C_PRELUDE: &str = r#"
#include <riscv_bitmanip.h>
#include <riscv_crypto.h>
"#;

    const RUST_PRELUDE: &str = RUST_PRELUDE;

    const C_NAME_PREFIX: &str = "__riscv_";

    fn c_compiler_flags(&self, cli_options: &ProcessedCli) -> Vec<&str> {
        let arch_str = if cli_options.target.starts_with("riscv32") {
            "-march=rv32gc_zbb_zbc_zk_zks"
        } else {
            "-march=rv64gc_zbb_zbc_zk_zks"
        };
        vec![arch_str]
    }

    fn predicate_function(_: u32) -> String {
        unimplemented!("scalable vector intrinsics are currently unimplemented for RISC-V")
    }
}

const RUST_PRELUDE: &str = r#"
#![feature(riscv_ext_intrinsics)]

#[cfg(target_arch = "riscv64")]
use core_arch::arch::riscv64::*;
#[cfg(target_arch = "riscv32")]
use core_arch::arch::riscv32::*;
"#;

fn load_intrinsics(path: &Path) -> Result<Vec<Intrinsic<Riscv>>, Box<dyn std::error::Error>> {
    if path.is_dir() {
        let mut intrinsics = Vec::new();
        for spec in ["crypto.spec", "bitmanip.spec"] {
            let spec_path = path.join(spec);
            intrinsics.extend(get_intrinsics(&spec_path)?);
        }
        return Ok(intrinsics);
    }

    get_intrinsics(path)
}
