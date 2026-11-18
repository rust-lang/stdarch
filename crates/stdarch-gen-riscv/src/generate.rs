use std::{fmt, fs, path::Path};

use regex::regex;

pub fn generate(crate_dir: &Path, out_dir: &Path, header: &str) -> anyhow::Result<()> {
    let in_file = crate_dir.join(format!("riscv_{header}.h"));
    let contents = fs::read_to_string(in_file)?;
    let intrinsic_prototypes = parse_header(contents);
    let output = generate_spec_file(intrinsic_prototypes);
    let out_file = out_dir.join(format!("{header}.spec"));
    fs::write(out_file, output)?;
    Ok(())
}

#[derive(PartialEq, Copy, Clone)]
pub enum RiscvArch {
    Both,
    Rv32,
    Rv64,
}

impl fmt::Display for RiscvArch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RiscvArch::Both => f.write_str("both"),
            RiscvArch::Rv32 => f.write_str("rv32"),
            RiscvArch::Rv64 => f.write_str("rv64"),
        }
    }
}

pub struct RiscvIntrinsic {
    pub name: String,
    pub arch: RiscvArch,
    pub return_type: String,
    pub arguments: String,
    pub extensions: Vec<String>,
}

fn parse_header(contents: String) -> Vec<RiscvIntrinsic> {
    let mut intrinsics = vec![];

    let mut lines = contents.lines().map(|line| line.trim());
    let mut current_extensions = None;
    let mut current_arch = RiscvArch::Both;
    // Some #if statements have the xlen condition in the same line as the extension condition,
    // which complicates parsing
    let mut xlen_if_nesting = false;
    while let Some(mut line) = lines.next() {
        // Parse extensions: start of block
        let line_extensions: Vec<_> = regex!(r"defined\(__riscv_(\w+)\)")
            .captures_iter(line)
            .map(|captures| captures[1].to_string())
            .collect();
        if !line_extensions.is_empty() && (line.contains("\\") || !line.starts_with("#if ")) {
            // These are the "meta" extension definitions at the start of the file
            continue;
        }
        if current_extensions.is_some() && !line_extensions.is_empty() {
            panic!(
                "parser error: should not have nested extensions.\nLine:{line}\nActive extensions:\n{current_extensions:?}"
            );
        } else if !line_extensions.is_empty() {
            current_extensions = Some(line_extensions);
        }

        // Parse extensions: end of block
        let Some(extensions) = &current_extensions else {
            continue;
        };
        let end_if_comment = extensions
            .iter()
            .map(|extension| format!("__riscv_{extension}"))
            .reduce(|a, b| format!("{a} || {b}"))
            .unwrap();
        let end_line = format!("#endif // {end_if_comment}");
        if line == end_line {
            current_extensions = None;
            current_arch = RiscvArch::Both;
            continue;
        }

        // Parse bit width
        if line.contains("__riscv_xlen == 32") {
            assert!(
                current_arch == RiscvArch::Both,
                "parser error: should not have nested xlen\nLine:\n{line}"
            );
            current_arch = RiscvArch::Rv32;
            xlen_if_nesting = line == "#if __riscv_xlen == 32";
        }
        if line.contains("__riscv_xlen == 64") {
            assert!(
                current_arch == RiscvArch::Both,
                "parser error: should not have nested xlen\nLine:\n{line}"
            );
            current_arch = RiscvArch::Rv64;
            xlen_if_nesting = line == "#if __riscv_xlen == 64";
        }
        if current_arch != RiscvArch::Both && line == "#endif" && xlen_if_nesting {
            current_arch = RiscvArch::Both;
        }

        // Parse function declaration
        let Some(return_type) = regex!("extern __inline (.*)")
            .captures(line)
            .map(|captures| captures[1].to_string())
        else {
            continue;
        };

        let (name, arguments) = loop {
            let Some(captures) = regex!(r"__riscv_(\w+)\((.*?)\)").captures(line) else {
                line = lines.next().unwrap();
                continue;
            };
            let name = captures[1].to_string();
            let arguments = captures[2].to_string();

            break (name, arguments);
        };

        // Skip to the end of this function
        while line != "}" {
            line = lines.next().unwrap();
        }

        intrinsics.push(RiscvIntrinsic {
            name,
            arch: current_arch,
            return_type,
            arguments,
            extensions: extensions.clone(),
        });
    }
    intrinsics
}

fn generate_spec_file(intrinsics: Vec<RiscvIntrinsic>) -> String {
    let mut output = r#"// This code is automatically generated. DO NOT MODIFY.
// ```
// cargo run -p stdarch-gen-riscv
// ```
"#
    .to_string();

    output.extend(intrinsics.into_iter().map(|intrinsic| {
        format!(
            "
name = {}
arch = {}
return_type = {}
arguments = {}
extensions = {}
",
            intrinsic.name,
            intrinsic.arch,
            intrinsic.return_type,
            intrinsic.arguments,
            intrinsic.extensions.join(", ")
        )
    }));

    output
}
