use clap::Parser;
use std::collections::HashSet;
use std::env;
use std::fmt;
use std::fs::File;
use std::io::prelude::*;
use std::io::{self, BufReader};
use std::path::Path;
use std::path::PathBuf;
use stdarch_gen_common::{GeneratorCtx, Mode, run_generator};

/// Complete lines of generated source.
///
/// This enables common generation tasks to be factored out without precluding basic
/// context-specific formatting.
///
/// The convention in this generator is to prefix (not suffix) lines with a newline, so the
/// implementation of `std::fmt::Display` behaves in the same way.
struct Lines {
    indent: usize,
    lines: Vec<String>,
}

impl Lines {
    fn single(line: String) -> Self {
        Self::from(vec![line])
    }
}

impl From<Vec<String>> for Lines {
    fn from(lines: Vec<String>) -> Self {
        Self { indent: 0, lines }
    }
}

impl std::fmt::Display for Lines {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
        for line in self.lines.iter() {
            write!(f, "\n{:width$}{line}", "", width = self.indent)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq)]
enum TargetFeature {
    Lsx,
    Lasx,
}

impl TargetFeature {
    fn new(ext: &str) -> TargetFeature {
        match ext {
            "lasx" => Self::Lasx,
            _ => Self::Lsx,
        }
    }

    /// A string for use with `#[target_feature(...)]`.
    fn as_target_feature_arg(&self, ins: &str) -> String {
        let vec = match *self {
            // Features included with LoongArch64 LSX and LASX.
            Self::Lsx => "lsx",
            Self::Lasx => "lasx",
        };
        let frecipe = match ins {
            "lsx_vfrecipe_s" | "lsx_vfrecipe_d" | "lsx_vfrsqrte_s" | "lsx_vfrsqrte_d"
            | "lasx_xvfrecipe_s" | "lasx_xvfrecipe_d" | "lasx_xvfrsqrte_s" | "lasx_xvfrsqrte_d" => {
                ",frecipe"
            }
            _ => "",
        };
        format!("{vec}{frecipe}")
    }

    fn attr(name: &str, value: impl fmt::Display) -> String {
        format!(r#"#[{name}(enable = "{value}")]"#)
    }

    /// Generate a target_feature attribute
    fn to_target_feature_attr(self, ins: &str) -> Lines {
        Lines::single(Self::attr(
            "target_feature",
            self.as_target_feature_arg(ins),
        ))
    }
}

fn portable_intrinsics() -> HashSet<&'static str> {
    include_str!("portable-intrinsics.txt")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .collect()
}

fn gen_spec(in_file: String, ext_name: &str) -> io::Result<()> {
    let f = File::open(in_file.clone()).unwrap_or_else(|_| panic!("Failed to open {in_file}"));
    let f = BufReader::new(f);
    let mut out = format!(
        r#"// This code is automatically generated. DO NOT MODIFY.
// ```
// OUT_DIR=`pwd`/crates/stdarch-gen-loongarch cargo run -p stdarch-gen-loongarch -- {in_file}
// ```
"#
    );
    out.push('\n');

    let mut asm_fmts = String::new();
    let mut data_types = String::new();
    let fn_pat = format!("__{ext_name}_");
    let portable_intrinsics = portable_intrinsics();
    for line in f.lines() {
        let line = line.unwrap();
        if line.is_empty() {
            continue;
        }

        if let Some(s) = line.find("/* Assembly instruction format:") {
            let e = line.find('.').unwrap();
            asm_fmts = line.get(s + 31..e).unwrap().trim().to_string();
        } else if let Some(s) = line.find("/* Data types in instruction templates:") {
            let e = line.find('.').unwrap();
            data_types = line.get(s + 39..e).unwrap().trim().to_string();
        } else if let Some(s) = line.find(fn_pat.as_str()) {
            let e = line.find('(').unwrap();
            let name = line.get(s + 2..e).unwrap().trim().to_string();
            out.push_str(&format!("/// {name}\n"));
            if portable_intrinsics.contains(name.as_str()) {
                out.push_str("impl = portable\n");
            }
            out.push_str(&format!("name = {name}\n"));
            out.push_str(&format!("asm-fmts = {asm_fmts}\n"));
            out.push_str(&format!("data-types = {data_types}\n"));
            out.push('\n');
        }
    }

    let out_dir_path: PathBuf = PathBuf::from(env::var("OUT_DIR").unwrap());
    std::fs::create_dir_all(&out_dir_path)?;
    let mut f = File::create(out_dir_path.join(format!("{ext_name}.spec")))?;
    f.write_all(out.as_bytes())?;
    Ok(())
}

fn gen_bind(in_file: &str, ext_name: &str, out_path: &Path) -> io::Result<()> {
    let f = File::open(in_file).unwrap_or_else(|_| panic!("Failed to open {in_file}"));
    let f = BufReader::new(f);

    let target: TargetFeature = TargetFeature::new(ext_name);
    let mut para_num;
    let mut current_name: Option<String> = None;
    let mut asm_fmts: Vec<String> = Vec::new();
    let mut link_function_str = String::new();
    let mut function_str = String::new();
    let mut out = String::new();
    let mut skip = false;

    out.push_str(&format!(
        r#"// This code is automatically generated. DO NOT MODIFY.
//
// Instead, modify `{in_file}` and run the following command to re-generate this file:
//
// ```
// OUT_DIR=`pwd`/crates/core_arch cargo run -p stdarch-gen-loongarch -- {in_file}
// ```

use crate::mem::transmute;
use super::super::*;
"#
    ));

    out.push_str(
        r#"
#[allow(improper_ctypes)]
unsafe extern "llvm-intrinsic" {
"#,
    );

    for line in f.lines() {
        let line = line.unwrap();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("impl = portable") {
            skip = true;
        } else if let Some(name) = line.strip_prefix("name = ") {
            current_name = Some(String::from(name));
        } else if line.starts_with("asm-fmts = ") {
            asm_fmts = line[10..]
                .split(',')
                .map(|v| v.trim().to_string())
                .collect();
        } else if line.starts_with("data-types = ") {
            let current_name = current_name.clone().unwrap();
            let data_types: Vec<&str> = line
                .get(12..)
                .unwrap()
                .split(',')
                .map(|e| e.trim())
                .collect();
            let in_t;
            let out_t;
            if data_types.len() == 2 {
                in_t = [data_types[1], "NULL", "NULL", "NULL"];
                out_t = data_types[0];
                para_num = 1;
            } else if data_types.len() == 3 {
                in_t = [data_types[1], data_types[2], "NULL", "NULL"];
                out_t = data_types[0];
                para_num = 2;
            } else if data_types.len() == 4 {
                in_t = [data_types[1], data_types[2], data_types[3], "NULL"];
                out_t = data_types[0];
                para_num = 3;
            } else if data_types.len() == 5 {
                in_t = [data_types[1], data_types[2], data_types[3], data_types[4]];
                out_t = data_types[0];
                para_num = 4;
            } else {
                panic!("DEBUG: line: {0} len: {1}", line, data_types.len());
            }

            if skip {
                skip = false;
                continue;
            }

            let (link_function, function) =
                gen_bind_body(&current_name, &asm_fmts, &in_t, out_t, para_num, target);
            link_function_str.push_str(&link_function);
            function_str.push_str(&function);
        }
    }
    out.push_str(&link_function_str);
    out.push_str("}\n");
    out.push_str(&function_str);

    std::fs::create_dir_all(out_path)?;
    let mut file = File::create(out_path.join("generated.rs"))?;
    file.write_all(out.as_bytes())?;
    Ok(())
}

fn gen_bind_body(
    current_name: &str,
    asm_fmts: &[String],
    in_t: &[&str; 4],
    out_t: &str,
    para_num: i32,
    target: TargetFeature,
) -> (String, String) {
    enum TypeKind {
        Vector,
        Intrinsic,
    }
    use TypeKind::*;
    let type_to_rst = |t: &str, s: bool, k: TypeKind| -> &str {
        match (t, s, k) {
            ("V16QI", _, Vector) => "__v16i8",
            ("V16QI", _, Intrinsic) => "m128i",
            ("V32QI", _, Vector) => "__v32i8",
            ("V32QI", _, Intrinsic) => "m256i",
            ("V8HI", _, Vector) => "__v8i16",
            ("V8HI", _, Intrinsic) => "m128i",
            ("V16HI", _, Vector) => "__v16i16",
            ("V16HI", _, Intrinsic) => "m256i",
            ("V4SI", _, Vector) => "__v4i32",
            ("V4SI", _, Intrinsic) => "m128i",
            ("V8SI", _, Vector) => "__v8i32",
            ("V8SI", _, Intrinsic) => "m256i",
            ("V2DI", _, Vector) => "__v2i64",
            ("V2DI", _, Intrinsic) => "m128i",
            ("V4DI", _, Vector) => "__v4i64",
            ("V4DI", _, Intrinsic) => "m256i",
            ("UV16QI", _, Vector) => "__v16u8",
            ("UV16QI", _, Intrinsic) => "m128i",
            ("UV32QI", _, Vector) => "__v32u8",
            ("UV32QI", _, Intrinsic) => "m256i",
            ("UV8HI", _, Vector) => "__v8u16",
            ("UV8HI", _, Intrinsic) => "m128i",
            ("UV16HI", _, Vector) => "__v16u16",
            ("UV16HI", _, Intrinsic) => "m256i",
            ("UV4SI", _, Vector) => "__v4u32",
            ("UV4SI", _, Intrinsic) => "m128i",
            ("UV8SI", _, Vector) => "__v8u32",
            ("UV8SI", _, Intrinsic) => "m256i",
            ("UV2DI", _, Vector) => "__v2u64",
            ("UV2DI", _, Intrinsic) => "m128i",
            ("UV4DI", _, Vector) => "__v4u64",
            ("UV4DI", _, Intrinsic) => "m256i",
            ("SI", _, _) => "i32",
            ("DI", _, _) => "i64",
            ("USI", _, _) => "u32",
            ("UDI", _, _) => "u64",
            ("V4SF", _, Vector) => "__v4f32",
            ("V4SF", _, Intrinsic) => "m128",
            ("V8SF", _, Vector) => "__v8f32",
            ("V8SF", _, Intrinsic) => "m256",
            ("V2DF", _, Vector) => "__v2f64",
            ("V2DF", _, Intrinsic) => "m128d",
            ("V4DF", _, Vector) => "__v4f64",
            ("V4DF", _, Intrinsic) => "m256d",
            ("UQI", _, _) => "u32",
            ("QI", _, _) => "i32",
            ("CVPOINTER", false, _) => "*const i8",
            ("CVPOINTER", true, _) => "*mut i8",
            ("HI", _, _) => "i32",
            (_, _, _) => panic!("unknown type: {t}"),
        }
    };

    let is_mem = in_t.iter().any(|s| s.contains("POINTER"));
    let is_store = current_name.to_string().contains("vst");
    let link_function = {
        let fn_decl = {
            let fn_output = if out_t.to_lowercase() == "void" {
                String::new()
            } else {
                format!(" -> {}", type_to_rst(out_t, is_store, Vector))
            };
            let fn_inputs = match para_num {
                1 => format!("(a: {})", type_to_rst(in_t[0], is_store, Vector)),
                2 => format!(
                    "(a: {}, b: {})",
                    type_to_rst(in_t[0], is_store, Vector),
                    type_to_rst(in_t[1], is_store, Vector)
                ),
                3 => format!(
                    "(a: {}, b: {}, c: {})",
                    type_to_rst(in_t[0], is_store, Vector),
                    type_to_rst(in_t[1], is_store, Vector),
                    type_to_rst(in_t[2], is_store, Vector)
                ),
                4 => format!(
                    "(a: {}, b: {}, c: {}, d: {})",
                    type_to_rst(in_t[0], is_store, Vector),
                    type_to_rst(in_t[1], is_store, Vector),
                    type_to_rst(in_t[2], is_store, Vector),
                    type_to_rst(in_t[3], is_store, Vector)
                ),
                _ => panic!("unsupported parameter number"),
            };
            format!("fn __{current_name}{fn_inputs}{fn_output};")
        };
        let function = format!(
            r#"    #[link_name = "llvm.loongarch.{}"]
    {fn_decl}
"#,
            current_name.replace('_', ".")
        );
        function
    };

    let type_to_imm = |t| -> i8 {
        match t {
            'b' => 4,
            'h' => 3,
            'w' => 2,
            'd' => 1,
            _ => panic!("unsupported type"),
        }
    };
    let mut rustc_legacy_const_generics = "";
    let fn_decl = {
        let fn_output = if out_t.to_lowercase() == "void" {
            String::new()
        } else {
            format!("-> {} ", type_to_rst(out_t, is_store, Intrinsic))
        };
        let mut fn_inputs = match para_num {
            1 => format!("(a: {})", type_to_rst(in_t[0], is_store, Intrinsic)),
            2 => format!(
                "(a: {}, b: {})",
                type_to_rst(in_t[0], is_store, Intrinsic),
                type_to_rst(in_t[1], is_store, Intrinsic)
            ),
            3 => format!(
                "(a: {}, b: {}, c: {})",
                type_to_rst(in_t[0], is_store, Intrinsic),
                type_to_rst(in_t[1], is_store, Intrinsic),
                type_to_rst(in_t[2], is_store, Intrinsic)
            ),
            4 => format!(
                "(a: {}, b: {}, c: {}, d: {})",
                type_to_rst(in_t[0], is_store, Intrinsic),
                type_to_rst(in_t[1], is_store, Intrinsic),
                type_to_rst(in_t[2], is_store, Intrinsic),
                type_to_rst(in_t[3], is_store, Intrinsic)
            ),
            _ => panic!("unsupported parameter number"),
        };
        if para_num == 1 && in_t[0] == "HI" {
            fn_inputs = match asm_fmts[1].as_str() {
                "si13" | "i13" => format!(
                    "<const IMM_S13: {}>()",
                    type_to_rst(in_t[0], is_store, Intrinsic)
                ),
                "si10" => format!(
                    "<const IMM_S10: {}>()",
                    type_to_rst(in_t[0], is_store, Intrinsic)
                ),
                _ => panic!("unsupported assembly format: {}", asm_fmts[1]),
            };
            rustc_legacy_const_generics = "rustc_legacy_const_generics(0)";
        } else if para_num == 2 && (in_t[1] == "UQI" || in_t[1] == "USI") {
            fn_inputs = if asm_fmts[2].starts_with("ui") {
                format!(
                    "<const IMM{2}: {1}>(a: {0})",
                    type_to_rst(in_t[0], is_store, Intrinsic),
                    type_to_rst(in_t[1], is_store, Intrinsic),
                    asm_fmts[2].get(2..).unwrap()
                )
            } else {
                panic!("unsupported assembly format: {}", asm_fmts[2]);
            };
            rustc_legacy_const_generics = "rustc_legacy_const_generics(1)";
        } else if para_num == 2 && in_t[1] == "QI" {
            fn_inputs = if asm_fmts[2].starts_with("si") {
                format!(
                    "<const IMM_S{2}: {1}>(a: {0})",
                    type_to_rst(in_t[0], is_store, Intrinsic),
                    type_to_rst(in_t[1], is_store, Intrinsic),
                    asm_fmts[2].get(2..).unwrap()
                )
            } else {
                panic!("unsupported assembly format: {}", asm_fmts[2]);
            };
            rustc_legacy_const_generics = "rustc_legacy_const_generics(1)";
        } else if para_num == 2 && in_t[0] == "CVPOINTER" && in_t[1] == "SI" {
            fn_inputs = if asm_fmts[2].starts_with("si") {
                format!(
                    "<const IMM_S{2}: {1}>(mem_addr: {0})",
                    type_to_rst(in_t[0], is_store, Intrinsic),
                    type_to_rst(in_t[1], is_store, Intrinsic),
                    asm_fmts[2].get(2..).unwrap()
                )
            } else {
                panic!("unsupported assembly format: {}", asm_fmts[2]);
            };
            rustc_legacy_const_generics = "rustc_legacy_const_generics(1)";
        } else if para_num == 2 && in_t[0] == "CVPOINTER" && in_t[1] == "DI" {
            fn_inputs = match asm_fmts[2].as_str() {
                "rk" => format!(
                    "(mem_addr: {}, b: {})",
                    type_to_rst(in_t[0], is_store, Intrinsic),
                    type_to_rst(in_t[1], is_store, Intrinsic)
                ),
                _ => panic!("unsupported assembly format: {}", asm_fmts[2]),
            };
        } else if para_num == 3 && (in_t[2] == "USI" || in_t[2] == "UQI") {
            fn_inputs = if asm_fmts[2].starts_with("ui") {
                format!(
                    "<const IMM{3}: {2}>(a: {0}, b: {1})",
                    type_to_rst(in_t[0], is_store, Intrinsic),
                    type_to_rst(in_t[1], is_store, Intrinsic),
                    type_to_rst(in_t[2], is_store, Intrinsic),
                    asm_fmts[2].get(2..).unwrap()
                )
            } else {
                panic!("unsupported assembly format: {}", asm_fmts[2])
            };
            rustc_legacy_const_generics = "rustc_legacy_const_generics(2)";
        } else if para_num == 3 && in_t[1] == "CVPOINTER" && in_t[2] == "SI" {
            fn_inputs = match asm_fmts[2].as_str() {
                "si12" => format!(
                    "<const IMM_S12: {2}>(a: {0}, mem_addr: {1})",
                    type_to_rst(in_t[0], is_store, Intrinsic),
                    type_to_rst(in_t[1], is_store, Intrinsic),
                    type_to_rst(in_t[2], is_store, Intrinsic)
                ),
                _ => panic!("unsupported assembly format: {}", asm_fmts[2]),
            };
            rustc_legacy_const_generics = "rustc_legacy_const_generics(2)";
        } else if para_num == 3 && in_t[1] == "CVPOINTER" && in_t[2] == "DI" {
            fn_inputs = match asm_fmts[2].as_str() {
                "rk" => format!(
                    "(a: {}, mem_addr: {}, b: {})",
                    type_to_rst(in_t[0], is_store, Intrinsic),
                    type_to_rst(in_t[1], is_store, Intrinsic),
                    type_to_rst(in_t[2], is_store, Intrinsic)
                ),
                _ => panic!("unsupported assembly format: {}", asm_fmts[2]),
            };
        } else if para_num == 4 {
            fn_inputs = match (asm_fmts[2].as_str(), current_name.chars().last().unwrap()) {
                ("si8", t) => format!(
                    "<const IMM_S8: {2}, const IMM{4}: {3}>(a: {0}, mem_addr: {1})",
                    type_to_rst(in_t[0], is_store, Intrinsic),
                    type_to_rst(in_t[1], is_store, Intrinsic),
                    type_to_rst(in_t[2], is_store, Intrinsic),
                    type_to_rst(in_t[3], is_store, Intrinsic),
                    type_to_imm(t),
                ),
                (_, _) => panic!(
                    "unsupported assembly format: {} for {}",
                    asm_fmts[2], current_name
                ),
            };
            rustc_legacy_const_generics = "rustc_legacy_const_generics(2, 3)";
        }
        format!(
            "pub {}fn {current_name}{fn_inputs} {fn_output}",
            if is_mem { "unsafe " } else { "" }
        )
    };
    let unsafe_start = if !is_mem { "unsafe { " } else { "" };
    let unsafe_end = if !is_mem { " }" } else { "" };
    let mut call_params = {
        match para_num {
            1 => format!("{unsafe_start}transmute(__{current_name}(transmute(a))){unsafe_end}"),
            2 => format!(
                "{unsafe_start}transmute(__{current_name}(transmute(a), transmute(b))){unsafe_end}"
            ),
            3 => format!(
                "{unsafe_start}transmute(__{current_name}(transmute(a), transmute(b), transmute(c))){unsafe_end}"
            ),
            4 => format!(
                "{unsafe_start}transmute(__{current_name}(transmute(a), transmute(b), transmute(c), transmute(d))){unsafe_end}"
            ),
            _ => panic!("unsupported parameter number"),
        }
    };
    if para_num == 1 && in_t[0] == "HI" {
        call_params = match asm_fmts[1].as_str() {
            "si10" => {
                format!(
                    "static_assert_simm_bits!(IMM_S10, 10);\n    {unsafe_start}transmute(__{current_name}(IMM_S10)){unsafe_end}"
                )
            }
            "i13" => {
                format!(
                    "static_assert_simm_bits!(IMM_S13, 13);\n    {unsafe_start}transmute(__{current_name}(IMM_S13)){unsafe_end}"
                )
            }
            _ => panic!("unsupported assembly format: {}", asm_fmts[2]),
        }
    } else if para_num == 2 && (in_t[1] == "UQI" || in_t[1] == "USI") {
        call_params = if asm_fmts[2].starts_with("ui") {
            format!(
                "static_assert_uimm_bits!(IMM{0}, {0});\n    {unsafe_start}transmute(__{current_name}(transmute(a), IMM{0})){unsafe_end}",
                asm_fmts[2].get(2..).unwrap()
            )
        } else {
            panic!("unsupported assembly format: {}", asm_fmts[2])
        };
    } else if para_num == 2 && in_t[1] == "QI" {
        call_params = match asm_fmts[2].as_str() {
            "si5" => {
                format!(
                    "static_assert_simm_bits!(IMM_S5, 5);\n    {unsafe_start}transmute(__{current_name}(transmute(a), IMM_S5)){unsafe_end}"
                )
            }
            _ => panic!("unsupported assembly format: {}", asm_fmts[2]),
        };
    } else if para_num == 2 && in_t[0] == "CVPOINTER" && in_t[1] == "SI" {
        call_params = if asm_fmts[2].starts_with("si") {
            format!(
                "static_assert_simm_bits!(IMM_S{0}, {0});\n    {unsafe_start}transmute(__{current_name}(mem_addr, IMM_S{0})){unsafe_end}",
                asm_fmts[2].get(2..).unwrap()
            )
        } else {
            panic!("unsupported assembly format: {}", asm_fmts[2])
        }
    } else if para_num == 2 && in_t[0] == "CVPOINTER" && in_t[1] == "DI" {
        call_params = match asm_fmts[2].as_str() {
            "rk" => format!(
                "{unsafe_start}transmute(__{current_name}(mem_addr, transmute(b))){unsafe_end}"
            ),
            _ => panic!("unsupported assembly format: {}", asm_fmts[2]),
        };
    } else if para_num == 3 && (in_t[2] == "USI" || in_t[2] == "UQI") {
        call_params = if asm_fmts[2].starts_with("ui") {
            format!(
                "static_assert_uimm_bits!(IMM{0}, {0});\n    {unsafe_start}transmute(__{current_name}(transmute(a), transmute(b), IMM{0})){unsafe_end}",
                asm_fmts[2].get(2..).unwrap()
            )
        } else {
            panic!("unsupported assembly format: {}", asm_fmts[2])
        }
    } else if para_num == 3 && in_t[1] == "CVPOINTER" && in_t[2] == "SI" {
        call_params = match asm_fmts[2].as_str() {
            "si12" => format!(
                "static_assert_simm_bits!(IMM_S12, 12);\n    {unsafe_start}__{current_name}(transmute(a), mem_addr, IMM_S12){unsafe_end}"
            ),
            _ => panic!("unsupported assembly format: {}", asm_fmts[2]),
        };
    } else if para_num == 3 && in_t[1] == "CVPOINTER" && in_t[2] == "DI" {
        call_params = match asm_fmts[2].as_str() {
            "rk" => format!(
                "{unsafe_start}__{current_name}(transmute(a), mem_addr, transmute(b)){unsafe_end}"
            ),
            _ => panic!("unsupported assembly format: {}", asm_fmts[2]),
        };
    } else if para_num == 4 {
        call_params = match (asm_fmts[2].as_str(), current_name.chars().last().unwrap()) {
            ("si8", t) => format!(
                "static_assert_simm_bits!(IMM_S8, 8);\n    static_assert_uimm_bits!(IMM{0}, {0});\n    {unsafe_start}__{current_name}(transmute(a), mem_addr, IMM_S8, IMM{0}){unsafe_end}",
                type_to_imm(t)
            ),
            (_, _) => panic!(
                "unsupported assembly format: {} for {}",
                asm_fmts[2], current_name
            ),
        }
    }
    let function = if !rustc_legacy_const_generics.is_empty() {
        format!(
            r#"
#[inline]{target_feature}
#[{rustc_legacy_const_generics}]
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
{fn_decl}{{
    {call_params}
}}
"#,
            target_feature = target.to_target_feature_attr(current_name)
        )
    } else {
        format!(
            r#"
#[inline]{target_feature}
#[unstable(feature = "stdarch_loongarch", issue = "117427")]
{fn_decl}{{
    {call_params}
}}
"#,
            target_feature = target.to_target_feature_attr(current_name)
        )
    };
    (link_function, function)
}

/// Runs the check/bless harness for `lsx`/`lasx` when invoked with
/// no args or a bare ext name.
#[derive(clap::Parser, Debug)]
struct Args {
    /// Either:
    /// - The extension (lsx/lasx) to generate, or:
    /// - A path to a <extension>intrin.h file to generate the spec file from.
    arguments: Vec<String>,
    /// Generation mode.
    #[arg(long, env = "STDARCH_GEN_MODE")]
    mode: Option<Mode>,
    /// Path to a rustfmt binary that will be used to reformat the generated code.
    /// If unset, it will just use "rustfmt" from the environment.
    #[arg(long)]
    rustfmt_path: Option<PathBuf>,
}

pub fn main() -> Result<(), String> {
    let args = Args::parse();

    let crate_dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"));
    let core_arch_src = crate_dir.join("../core_arch/src");
    let mode = args.mode.unwrap_or_default();
    let ctx = GeneratorCtx::new(args.rustfmt_path);

    let arguments = &args.arguments;
    if arguments.len() == 1 && (arguments[0] == "lsx" || arguments[0] == "lasx") {
        let extension = arguments[0].as_str();
        let spec_rel = format!("crates/stdarch-gen-loongarch/{extension}.spec");
        let committed = core_arch_src.join("loongarch64").join(extension);
        run_generator(&ctx, &committed, mode, |out_dir| {
            gen_bind(&spec_rel, extension, out_dir)
        })
        .map_err(|e| e.to_string())?;
    } else {
        let in_file = arguments[0].clone();
        let in_file_name = PathBuf::from(&in_file)
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let ext_name = if in_file_name.starts_with("lasx") {
            "lasx"
        } else {
            "lsx"
        };
        if in_file_name.ends_with(".h") {
            return gen_spec(in_file, ext_name).map_err(|e| e.to_string());
        }
        // Note: this does not apply rustfmt formatting
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap_or("crates/core_arch".to_string()))
            .join("src")
            .join("loongarch64")
            .join(ext_name);
        gen_bind(&in_file, ext_name, &out_path).map_err(|e| e.to_string())?;
    }
    Ok(())
}
