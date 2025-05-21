#![feature(slice_partition_dedup)]
#[macro_use]
extern crate log;

mod arm;
mod common;

use arm::ArmTestProcessor;
use common::SupportedArchitectureTest;
use common::types::{Cli, ProcessedCli};

fn main() {
    pretty_env_logger::init();
    let args: Cli = clap::Parser::parse();
    let processed_cli_options = ProcessedCli::new(args);

    let test_environment_result = match processed_cli_options.target.as_str() {
        "aarch64-unknown-linux-gnu"
        | "armv7-unknown-linux-gnueabihf"
        | "aarch64_be-unknown-linux-gnu" => Some(ArmTestProcessor::create(processed_cli_options)),

        _ => None,
    };

    if test_environment_result.is_none() {
        std::process::exit(0);
    }

    let test_environment = test_environment_result.unwrap();

    if !test_environment.build_c_file() {
        std::process::exit(2);
    }
    if !test_environment.build_rust_file() {
        std::process::exit(3);
    }
    if !test_environment.compare_outputs() {
        std::process::exit(1);
    }
}
