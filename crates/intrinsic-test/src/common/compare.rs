use itertools::Itertools;
use rayon::prelude::*;
use std::{collections::HashMap, process::Command};

pub const INTRINSIC_DELIMITER: &str = "############";
fn runner_command(runner: &str) -> Command {
    let mut it = runner.split_whitespace();
    let mut cmd = Command::new(it.next().unwrap());
    cmd.args(it);

    cmd
}

pub fn compare_outputs(intrinsic_name_list: &Vec<String>, runner: &str, target: &str) -> bool {
    let available_parallelism = std::thread::available_parallelism().unwrap().get();
    let c_outputs = (0..available_parallelism)
        .into_par_iter()
        .map(|i| {
            runner_command(runner)
                .arg(format!("./intrinsic-test-programs-{i}"))
                .current_dir("c_programs")
                .output()
        })
        .collect::<Vec<_>>();

    let rust_outputs = (0..available_parallelism)
        .into_par_iter()
        .map(|i| {
            runner_command(runner)
                .arg(format!(
                    "./target/{target}/release/intrinsic-test-programs-{i}"
                ))
                .current_dir("rust_programs")
                .output()
        })
        .collect::<Vec<_>>();

    let c_error = c_outputs.iter().filter(|elem| elem.is_err()).next();
    let rust_error = rust_outputs.iter().filter(|elem| elem.is_err()).next();
    match (c_error, rust_error) {
        (None, None) => (),
        failure => panic!("Failed to run: {failure:#?}"),
    };

    let c_stdout = c_outputs
        .into_iter()
        .map(|c_elem| {
            let c = c_elem.unwrap();
            let c_stdout = std::str::from_utf8(&c.stdout).unwrap_or("").to_string();
            if !c.status.success() {
                error!(
                    "Failed to run C program.\nstdout: {c_stdout}\nstderr: {stderr}",
                    stderr = std::str::from_utf8(&c.stderr).unwrap_or(""),
                );
            }
            c_stdout
        })
        .collect_vec()
        .join("\n");

    let rust_stdout = rust_outputs
        .into_iter()
        .map(|rust_elem| {
            let rust = rust_elem.unwrap();
            let rust_stdout = std::str::from_utf8(&rust.stdout).unwrap_or("").to_string();
            if !rust.status.success() {
                error!(
                    "Failed to run Rust program.\nstdout: {rust_stdout}\nstderr: {stderr}",
                    stderr = std::str::from_utf8(&rust.stderr).unwrap_or(""),
                );
            }
            rust_stdout
        })
        .collect_vec()
        .join("\n");

    info!("Completed running C++ and Rust test binaries");
    let c = c_stdout.to_lowercase().replace("-nan", "nan");
    let rust = rust_stdout.to_lowercase().replace("-nan", "nan");

    let c_output_map = c
        .split(INTRINSIC_DELIMITER)
        .filter_map(|output| output.trim().split_once("\n"))
        .collect::<HashMap<&str, &str>>();
    let rust_output_map = rust
        .split(INTRINSIC_DELIMITER)
        .filter_map(|output| output.trim().split_once("\n"))
        .collect::<HashMap<&str, &str>>();

    let intrinsics = c_output_map
        .keys()
        .chain(rust_output_map.keys())
        .unique()
        .collect_vec();

    info!("Comparing outputs");
    let intrinsics_diff_count = intrinsics
        .par_iter()
        .filter_map(|&&intrinsic| {
            let c_output = c_output_map.get(intrinsic).unwrap();
            let rust_output = rust_output_map.get(intrinsic).unwrap();
            if rust_output.eq(c_output) {
                None
            } else {
                let diff = diff::lines(c_output, rust_output);
                let diffs = diff
                    .into_iter()
                    .filter_map(|diff| match diff {
                        diff::Result::Left(_) | diff::Result::Right(_) => Some(diff),
                        diff::Result::Both(_, _) => None,
                    })
                    .collect_vec();
                if diffs.len() > 0 {
                    Some((intrinsic, diffs))
                } else {
                    None
                }
            }
        })
        .inspect(|(intrinsic, diffs)| {
            println!("Difference for intrinsic: {intrinsic}");
            diffs.into_iter().for_each(|diff| match diff {
                diff::Result::Left(c) => println!("C: {c}"),
                diff::Result::Right(rust) => println!("Rust: {rust}"),
                _ => (),
            });
            println!("****************************************************************");
        })
        .count();

    println!(
        "{} differences found (tested {} intrinsics)",
        intrinsics_diff_count,
        intrinsic_name_list.len()
    );

    intrinsics_diff_count == 0
}
