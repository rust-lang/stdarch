use std::fs::File;
use std::io::Write;

use super::gen_rust::{create_rust_test_program, setup_rust_file_paths};
use super::intrinsic::IntrinsicDefinition;
use super::intrinsic_helpers::IntrinsicTypeDefinition;

pub fn write_file(filename: &String, code: String) {
    let mut file = File::create(filename).unwrap();
    file.write_all(code.into_bytes().as_slice()).unwrap();
}

pub fn write_c_testfiles<T: IntrinsicTypeDefinition + Sized>(
    intrinsics: &[&dyn IntrinsicDefinition<T>],
    target: &str,
    c_target: &str,
    headers: &[&str],
    notice: &str,
    arch_specific_definitions: &[&str],
) -> std::io::Result<Vec<String>> {
    std::fs::create_dir_all("c_programs")?;

    intrinsics
        .iter()
        .map(|intrinsic| {
            let identifier = intrinsic.name().to_owned();
            let mut file = File::create(format!("c_programs/{identifier}.cpp")).unwrap();

            // write_c_test_program(&mut file, intrinsic)?;
            let c_code = crate::common::gen_c::create_c_test_program(
                *intrinsic,
                headers,
                target,
                c_target,
                notice,
                arch_specific_definitions,
            );

            file.write_all(c_code.as_bytes())?;

            Ok(identifier)
        })
        .collect()
}

pub fn write_rust_testfiles<T: IntrinsicTypeDefinition>(
    intrinsics: Vec<&dyn IntrinsicDefinition<T>>,
    rust_target: &str,
    notice: &str,
    definitions: &str,
    cfg: &str,
) -> Vec<String> {
    let intrinsics_name_list = intrinsics
        .iter()
        .map(|i| i.name().clone())
        .collect::<Vec<_>>();
    let filename_mapping = setup_rust_file_paths(&intrinsics_name_list);

    intrinsics.iter().for_each(|&i| {
        let rust_code = create_rust_test_program(i, rust_target, notice, definitions, cfg);
        if let Some(filename) = filename_mapping.get(&i.name()) {
            write_file(filename, rust_code)
        }
    });

    intrinsics_name_list
}
