use rayon::prelude::*;
use std::process::Command;

enum FailureReason {
    RunC(String),
    RunRust(String),
    Difference(String, String, String),
}

pub fn compare_outputs(
    intrinsic_name_list: &Vec<String>,
    toolchain: &str,
    runner: &str,
    target: &str,
) -> bool {
    let intrinsics = intrinsic_name_list
        .par_iter()
        .filter_map(|intrinsic_name| {
            let c = Command::new("sh")
                .arg("-c")
                .arg(format!(
                    "{runner} ./c_programs/{intrinsic_name}",
                    runner = runner,
                    intrinsic_name = intrinsic_name,
                ))
                .output();

            let rust = if target != "aarch64_be-unknown-linux-gnu" {
                Command::new("sh")
                    .current_dir("rust_programs")
                    .arg("-c")
                    .arg(format!(
                        "cargo {toolchain} run --target {target} --bin {intrinsic_name} --release",
                        intrinsic_name = intrinsic_name,
                        toolchain = toolchain,
                        target = target
                    ))
                    .env("RUSTFLAGS", "-Cdebuginfo=0")
                    .output()
            } else {
                Command::new("sh")
                    .arg("-c")
                    .arg(format!(
                        "{runner} ./rust_programs/target/{target}/release/{intrinsic_name}",
                        runner = runner,
                        target = target,
                        intrinsic_name = intrinsic_name,
                    ))
                    .output()
            };

            let (c, rust) = match (c, rust) {
                (Ok(c), Ok(rust)) => (c, rust),
                a => panic!("{a:#?}"),
            };

            if !c.status.success() {
                error!("Failed to run C program for intrinsic {}", intrinsic_name);
                return Some(FailureReason::RunC(intrinsic_name.clone()));
            }

            if !rust.status.success() {
                error!(
                    "Failed to run rust program for intrinsic {}",
                    intrinsic_name
                );
                return Some(FailureReason::RunRust(intrinsic_name.clone()));
            }

            info!("Comparing intrinsic: {}", intrinsic_name);

            let c = std::str::from_utf8(&c.stdout)
                .unwrap()
                .to_lowercase()
                .replace("-nan", "nan");
            let rust = std::str::from_utf8(&rust.stdout)
                .unwrap()
                .to_lowercase()
                .replace("-nan", "nan");

            if c == rust {
                None
            } else {
                Some(FailureReason::Difference(intrinsic_name.clone(), c, rust))
            }
        })
        .collect::<Vec<_>>();

    intrinsics.iter().for_each(|reason| match reason {
        FailureReason::Difference(intrinsic, c, rust) => {
            println!("Difference for intrinsic: {intrinsic}");
            let diff = diff::lines(c, rust);
            diff.iter().for_each(|diff| match diff {
                diff::Result::Left(c) => println!("C: {c}"),
                diff::Result::Right(rust) => println!("Rust: {rust}"),
                diff::Result::Both(_, _) => (),
            });
            println!("****************************************************************");
        }
        FailureReason::RunC(intrinsic) => {
            println!("Failed to run C program for intrinsic {intrinsic}")
        }
        FailureReason::RunRust(intrinsic) => {
            println!("Failed to run rust program for intrinsic {intrinsic}")
        }
    });
    println!("{} differences found", intrinsics.len());
    intrinsics.is_empty()
}
