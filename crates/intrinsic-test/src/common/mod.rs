use std::fs::File;

use rayon::prelude::*;

use cli::ProcessedCli;

use crate::common::{
    compile_c::CppCompilation,
    gen_c::{write_main_cpp, write_mod_cpp},
    gen_rust::{
        compile_rust_programs, write_bin_cargo_toml, write_lib_cargo_toml, write_lib_rs,
        write_main_rs,
    },
    intrinsic::Intrinsic,
    intrinsic_helpers::IntrinsicTypeDefinition,
};

pub mod argument;
pub mod cli;
pub mod compare;
pub mod compile_c;
pub mod constraint;
pub mod gen_c;
pub mod gen_rust;
pub mod indentation;
pub mod intrinsic;
pub mod intrinsic_helpers;
pub mod values;

/// Architectures must support this trait
/// to be successfully tested.
pub trait SupportedArchitectureTest
where
    Self: Sync + Send,
{
    type IntrinsicImpl: IntrinsicTypeDefinition + Sync;

    fn cli_options(&self) -> &ProcessedCli;
    fn intrinsics(&self) -> &[Intrinsic<Self::IntrinsicImpl>];

    fn create(cli_options: ProcessedCli) -> Self;

    const NOTICE: &str;

    const PLATFORM_C_HEADERS: &[&str];
    const PLATFORM_C_DEFINITIONS: &str;
    const PLATFORM_C_FORWARD_DECLARATIONS: &str;

    const PLATFORM_RUST_CFGS: &str;
    const PLATFORM_RUST_DEFINITIONS: &str;

    fn cpp_compilation(&self) -> Option<CppCompilation>;

    fn build_c_file(&self) -> bool {
        let (chunk_size, chunk_count) = manual_chunk(self.intrinsics().len(), 400);

        let cpp_compiler_wrapped = self.cpp_compilation();

        std::fs::create_dir_all("c_programs").unwrap();
        self.intrinsics()
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                let c_filename = format!("c_programs/mod_{i}.cpp");
                let mut file = File::create(&c_filename).unwrap();
                let mod_file_write_result = write_mod_cpp(
                    &mut file,
                    Self::NOTICE,
                    Self::PLATFORM_C_HEADERS,
                    Self::PLATFORM_C_FORWARD_DECLARATIONS,
                    chunk,
                );

                if let Err(error) = mod_file_write_result {
                    return Err(format!("Error writing to mod_{i}.cpp: {error:?}"));
                }

                // compile this cpp file into a .o file.
                //
                // This is done because `cpp_compiler_wrapped` is None when
                // the --generate-only flag is passed
                trace!("compiling mod_{i}.cpp");
                if let Some(cpp_compiler) = cpp_compiler_wrapped.as_ref() {
                    let compile_output = cpp_compiler
                        .compile_object_file(&format!("mod_{i}.cpp"), &format!("mod_{i}.o"))
                        .map_err(|e| format!("Error compiling mod_{i}.cpp: {e:?}"))?;

                    assert!(
                        compile_output.status.success(),
                        "{}",
                        String::from_utf8_lossy(&compile_output.stderr)
                    );

                    trace!("finished compiling mod_{i}.cpp");
                }
                Ok(())
            })
            .collect::<Result<(), String>>()
            .unwrap();

        let (auto_chunk_size, auto_chunk_count) = auto_chunk(self.intrinsics().len());

        self.intrinsics()
            .par_chunks(auto_chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                let mut file = File::create(format!("c_programs/main_{i}.cpp")).unwrap();
                write_main_cpp(
                    &mut file,
                    Self::PLATFORM_C_DEFINITIONS,
                    Self::PLATFORM_C_HEADERS,
                    chunk.iter().map(|i| i.name.as_str()),
                )
            })
            .collect::<Result<(), std::io::Error>>()
            .unwrap();

        if let Some(cpp_compiler) = cpp_compiler_wrapped.as_ref() {
            (0..auto_chunk_count)
                .into_par_iter()
                .map(|index| {
                    // This is done because `cpp_compiler_wrapped` is None when
                    // the --generate-only flag is passed
                    // compile this cpp file into a .o file
                    trace!("compiling main_{index}.cpp");
                    let output = cpp_compiler.compile_object_file(
                        format!("main_{index}.cpp").as_str(),
                        format!("main_{index}.o").as_str(),
                    );

                    if output.is_err() {
                        return output;
                    };

                    let object_files = (0..chunk_count)
                        .map(|i| format!("mod_{i}.o"))
                        .chain([format!("main_{index}.o").to_owned()]);

                    let output = cpp_compiler.link_executable(
                        object_files,
                        format!("intrinsic-test-programs-{index}").as_str(),
                    );
                    trace!("finished compiling main_{index}.cpp");

                    return output;
                })
                .inspect(|output| {
                    assert!(output.is_ok(), "{output:?}");
                    if let Ok(out) = &output {
                        assert!(out.status.success(), "{output:?}")
                    }
                })
                .all(|output| output.is_ok())
        } else {
            true
        }
    }

    fn build_rust_file(&self) -> bool {
        std::fs::create_dir_all("rust_programs/src").unwrap();

        let (chunk_size, chunk_count) = manual_chunk(self.intrinsics().len(), 400);
        let (auto_chunk_size, auto_chunk_count) = auto_chunk(self.intrinsics().len());

        let mut cargo = File::create("rust_programs/Cargo.toml").unwrap();
        write_bin_cargo_toml(&mut cargo, chunk_count, auto_chunk_count).unwrap();

        self.intrinsics()
            .par_chunks(auto_chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                let mut main_rs = File::create(format!("rust_programs/src/main_{i}.rs")).unwrap();
                write_main_rs(
                    &mut main_rs,
                    chunk_count,
                    Self::PLATFORM_RUST_CFGS,
                    "",
                    chunk.iter().map(|i| i.name.as_str()),
                )
            })
            .collect::<Result<(), std::io::Error>>()
            .unwrap();

        let target = &self.cli_options().target;
        let toolchain = self.cli_options().toolchain.as_deref();
        let linker = self.cli_options().linker.as_deref();

        self.intrinsics()
            .par_chunks(chunk_size)
            .enumerate()
            .map(|(i, chunk)| {
                std::fs::create_dir_all(format!("rust_programs/mod_{i}/src"))?;

                let rust_filename = format!("rust_programs/mod_{i}/src/lib.rs");
                trace!("generating `{rust_filename}`");
                let mut file = File::create(rust_filename)?;

                write_lib_rs(
                    &mut file,
                    Self::NOTICE,
                    Self::PLATFORM_RUST_CFGS,
                    Self::PLATFORM_RUST_DEFINITIONS,
                    chunk,
                )?;

                let toml_filename = format!("rust_programs/mod_{i}/Cargo.toml");
                trace!("generating `{toml_filename}`");
                let mut file = File::create(toml_filename).unwrap();

                write_lib_cargo_toml(&mut file, &format!("mod_{i}"))?;

                Ok(())
            })
            .collect::<Result<(), std::io::Error>>()
            .unwrap();

        compile_rust_programs(toolchain, target, linker)
    }

    fn compare_outputs(&self) -> bool {
        if self.cli_options().toolchain.is_some() {
            let intrinsics_name_list = self
                .intrinsics()
                .iter()
                .map(|i| i.name.clone())
                .collect::<Vec<_>>();

            compare::compare_outputs(
                &intrinsics_name_list,
                &self.cli_options().runner,
                &self.cli_options().target,
            )
        } else {
            true
        }
    }
}

pub fn auto_chunk(intrinsic_count: usize) -> (usize, usize) {
    let available_parallelism = std::thread::available_parallelism().unwrap().get();
    let chunk_size = intrinsic_count.div_ceil(Ord::min(available_parallelism, intrinsic_count));

    (chunk_size, intrinsic_count.div_ceil(chunk_size))
}

pub fn manual_chunk(intrinsic_count: usize, chunk_size: usize) -> (usize, usize) {
    (chunk_size, intrinsic_count.div_ceil(chunk_size))
}
