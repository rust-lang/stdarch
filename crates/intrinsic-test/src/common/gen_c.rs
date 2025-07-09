use rayon::prelude::*;
use std::process::Command;

use super::argument::Argument;
use super::indentation::Indentation;
use super::intrinsic::IntrinsicDefinition;
use super::intrinsic_helpers::IntrinsicTypeDefinition;

// The number of times each intrinsic will be called.
const PASSES: u32 = 20;

pub fn compile_c_programs(compiler_commands: &[String]) -> bool {
    compiler_commands
        .par_iter()
        .map(|compiler_command| {
            let output = Command::new("sh").arg("-c").arg(compiler_command).output();
            if let Ok(output) = output {
                if output.status.success() {
                    true
                } else {
                    error!(
                        "Failed to compile code for intrinsics: \n\nstdout:\n{}\n\nstderr:\n{}",
                        std::str::from_utf8(&output.stdout).unwrap_or(""),
                        std::str::from_utf8(&output.stderr).unwrap_or("")
                    );
                    false
                }
            } else {
                error!("Command failed: {output:#?}");
                false
            }
        })
        .find_any(|x| !x)
        .is_none()
}

pub fn generate_c_test_loop<T: IntrinsicTypeDefinition + Sized>(
    w: &mut impl std::io::Write,
    intrinsic: &dyn IntrinsicDefinition<T>,
    indentation: Indentation,
    additional: &str,
    passes: u32,
) -> std::io::Result<()> {
    let body_indentation = indentation.nested();
    write!(
        w,
        "{indentation}for (int i=0; i<{passes}; i++) {{\n\
            {loaded_args}\
            {body_indentation}auto __return_value = {intrinsic_call}({args});\n\
            {print_result}\n\
        {indentation}}}",
        loaded_args = intrinsic.arguments().load_values_c(body_indentation),
        intrinsic_call = intrinsic.name(),
        args = intrinsic.arguments().as_call_param_c(),
        print_result = intrinsic.print_result_c(body_indentation, additional)
    )
}

pub fn generate_c_constraint_blocks<'a, T: IntrinsicTypeDefinition + 'a>(
    w: &mut impl std::io::Write,
    intrinsic: &dyn IntrinsicDefinition<T>,
    indentation: Indentation,
    constraints: &mut (impl Iterator<Item = &'a Argument<T>> + Clone),
    name: String,
) -> std::io::Result<()> {
    let Some(current) = constraints.next() else {
        return generate_c_test_loop(w, intrinsic, indentation, &name, PASSES);
    };

    let body_indentation = indentation.nested();
    for i in current.constraint.iter().flat_map(|c| c.to_range()) {
        let ty = current.ty.c_type();

        writeln!(w, "{indentation}{{")?;
        writeln!(w, "{body_indentation}{ty} {} = {i};", current.name)?;

        generate_c_constraint_blocks(
            w,
            intrinsic,
            body_indentation,
            &mut constraints.clone(),
            format!("{name}-{i}"),
        )?;

        writeln!(w, "{indentation}}}")?;
    }

    Ok(())
}

// Compiles C test programs using specified compiler
pub fn create_c_test_program<T: IntrinsicTypeDefinition>(
    w: &mut impl std::io::Write,
    intrinsic: &dyn IntrinsicDefinition<T>,
    header_files: &[&str],
    _target: &str,
    c_target: &str,
    notices: &str,
    arch_specific_definitions: &[&str],
) -> std::io::Result<()> {
    let indentation = Indentation::default();

    write!(w, "{notices}")?;

    for header in header_files {
        writeln!(w, "#include <{header}>")?;
    }

    writeln!(
        w,
        r#"
#include <iostream>
#include <cstring>
#include <iomanip>
#include <sstream>

template<typename T1, typename T2> T1 cast(T2 x) {{
  static_assert(sizeof(T1) == sizeof(T2), "sizeof T1 and T2 must be the same");
  T1 ret{{}};
  memcpy(&ret, &x, sizeof(T1));
  return ret;
}}

std::ostream& operator<<(std::ostream& os, float16_t value) {{
    uint16_t temp = 0;
    memcpy(&temp, &value, sizeof(float16_t));
    std::stringstream ss;
    ss << "0x" << std::setfill('0') << std::setw(4) << std::hex << temp;
    os << ss.str();
    return os;
}}
"#
    )?;

    let arch_identifier = c_target;
    writeln!(w, "#ifdef __{arch_identifier}__")?;
    for def in arch_specific_definitions {
        writeln!(w, "{def}")?;
    }
    writeln!(w, "#endif")?;

    writeln!(w, "int main(int argc, char **argv) {{")?;

    // Define the arrays of arguments.
    let arguments = intrinsic.arguments();
    arguments.gen_arglists_c(w, indentation.nested(), PASSES)?;

    generate_c_constraint_blocks(
        w,
        intrinsic,
        indentation.nested(),
        &mut arguments.iter().rev().filter(|&i| i.has_constraint()),
        Default::default(),
    )?;

    writeln!(w, "    return 0;")?;
    writeln!(w, "}}")?;

    Ok(())
}
