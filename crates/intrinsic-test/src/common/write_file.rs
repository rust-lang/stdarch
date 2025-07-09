use std::fs::File;

use super::gen_rust::create_rust_test_program;
use super::intrinsic::IntrinsicDefinition;
use super::intrinsic_helpers::IntrinsicTypeDefinition;

use rayon::prelude::*;

pub fn write_c_testfiles<'a, T, I, E>(
    intrinsics: I,
    target: &str,
    c_target: &str,
    headers: &[&str],
    notice: &str,
    arch_specific_definitions: &[&str],
) -> std::io::Result<Vec<String>>
where
    T: IntrinsicTypeDefinition + Sized + 'a,
    I: ParallelIterator<Item = &'a E>,
    E: IntrinsicDefinition<T> + 'a,
{
    std::fs::create_dir_all("c_programs")?;

    intrinsics
        .map(|intrinsic| {
            let identifier = intrinsic.name().to_owned();
            let mut file = File::create(format!("c_programs/{identifier}.cpp")).unwrap();

            crate::common::gen_c::create_c_test_program(
                &mut file,
                intrinsic,
                headers,
                target,
                c_target,
                notice,
                arch_specific_definitions,
            )?;

            Ok(identifier)
        })
        .collect()
}

pub fn write_rust_testfiles<'a, T, I, E>(
    intrinsics: I,
    rust_target: &str,
    notice: &str,
    definitions: &str,
    cfg: &str,
) -> std::io::Result<Vec<String>>
where
    T: IntrinsicTypeDefinition + Sized + 'a,
    I: ParallelIterator<Item = &'a E>,
    E: IntrinsicDefinition<T> + 'a,
{
    std::fs::create_dir_all("rust_programs")?;

    intrinsics
        .map(|intrinsic| {
            let identifier = intrinsic.name().to_owned();

            let rust_dir = format!("rust_programs/{identifier}");
            std::fs::create_dir_all(&rust_dir)?;
            let rust_filename = format!("{rust_dir}/main.rs");
            let mut file = File::create(rust_filename).unwrap();

            create_rust_test_program(&mut file, intrinsic, rust_target, notice, definitions, cfg)?;

            Ok(identifier)
        })
        .collect()
}
