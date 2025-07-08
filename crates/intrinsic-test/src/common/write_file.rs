use std::fs::File;
use std::io::Write;

use super::gen_rust::create_rust_test_program;
use super::intrinsic::IntrinsicDefinition;
use super::intrinsic_helpers::IntrinsicTypeDefinition;

pub fn write_c_testfiles<'a, T, I>(
    intrinsics: I,
    target: &str,
    c_target: &str,
    headers: &[&str],
    notice: &str,
    arch_specific_definitions: &[&str],
) -> std::io::Result<Vec<String>>
where
    T: IntrinsicTypeDefinition + Sized + 'a,
    I: Iterator<Item = &'a dyn IntrinsicDefinition<T>>,
{
    std::fs::create_dir_all("c_programs")?;

    intrinsics
        .map(|intrinsic| {
            let identifier = intrinsic.name().to_owned();
            let mut file = File::create(format!("c_programs/{identifier}.cpp")).unwrap();

            // write_c_test_program(&mut file, intrinsic)?;
            let c_code = crate::common::gen_c::create_c_test_program(
                intrinsic,
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

pub fn write_rust_testfiles<'a, T, I>(
    intrinsics: I,
    rust_target: &str,
    notice: &str,
    definitions: &str,
    cfg: &str,
) -> std::io::Result<Vec<String>>
where
    T: IntrinsicTypeDefinition + Sized + 'a,
    I: Iterator<Item = &'a dyn IntrinsicDefinition<T>>,
{
    std::fs::create_dir_all("rust_programs")?;

    intrinsics
        .map(|intrinsic| {
            let identifier = intrinsic.name().to_owned();

            let rust_dir = format!("rust_programs/{identifier}");
            std::fs::create_dir_all(&rust_dir)?;
            let rust_filename = format!("{rust_dir}/main.rs");
            let mut file = File::create(rust_filename).unwrap();

            let rust_code =
                create_rust_test_program(intrinsic, rust_target, notice, definitions, cfg);

            file.write_all(rust_code.as_bytes())?;

            Ok(identifier)
        })
        .collect()
}
