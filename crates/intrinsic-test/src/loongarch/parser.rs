use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_helpers::IntrinsicType;
use crate::loongarch::intrinsic::LoongArchIntrinsicType;

pub fn get_loongson_intrinsics(
    path: &Path,
    target: &str
) -> Result<Vec<Intrinsic<LoongArchIntrinsicType>>, Box<dyn std::error::Error>> {
    let f = File::open(path).unwrap_or_else(|_| panic!("Failed to open {}", path.display()));
    let f = BufReader::new(f);
    
    let mut para_num;
    let mut current_name: Option<String> = None;
    let mut asm_fmts: Vec<String> = Vec::new();
    let mut impl_function_str = String::new();
    let mut call_function_str = String::new();
    let mut out = String::new();

    let mut intrinsics: Vec<Intrinsic<LoongArchIntrinsicType>> = Vec::new();
    for line in f.lines() {
        let line = line.unwrap();
        if line.is_empty() {
            continue;
        }
        if let Some(name) = line.strip_prefix("name = ") {
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

            // TODO: implement the below functions
            // create list of intrinsics
            let intrinsic = gen_intrinsic(current_name.as_str(), asm_fmts.as_slice(), &in_t, out_t, para_num, target);
            if intrinsic.is_ok() {
                intrinsics.push(intrinsic.unwrap());
            }
        }
    };
    return Ok(intrinsics)
}

fn gen_intrinsic(
    current_name: &str,
    asm_fmts: &[String],
    in_t: &[&str; 4],
    out_t: &str,
    para_num: i32,
    target: &str,
) -> Result<Intrinsic<LoongArchIntrinsicType>, Box<dyn std::error::Error>> {
    let type_to_ct = |t: &str| -> &str {
        match t {
            "V16QI" => "union v16qi",
            "V32QI" => "union v32qi",
            "V8HI" => "union v8hi",
            "V16HI" => "union v16hi",
            "V4SI" => "union v4si",
            "V8SI" => "union v8si",
            "V2DI" => "union v2di",
            "V4DI" => "union v4di",
            "UV16QI" => "union uv16qi",
            "UV32QI" => "union uv32qi",
            "UV8HI" => "union uv8hi",
            "UV16HI" => "union uv16hi",
            "UV4SI" => "union uv4si",
            "UV8SI" => "union uv8si",
            "UV2DI" => "union uv2di",
            "UV4DI" => "union uv4di",
            "SI" => "int32_t",
            "DI" => "int64_t",
            "USI" => "uint32_t",
            "UDI" => "uint64_t",
            "V4SF" => "union v4sf",
            "V8SF" => "union v8sf",
            "V2DF" => "union v2df",
            "V4DF" => "union v4df",
            "UQI" => "uint32_t",
            "QI" => "int32_t",
            "CVPOINTER" => "void*",
            "HI" => "int32_t",
            _ => panic!("unknown type: {t}"),
        }
    };
    let type_to_size = |v: &str, t: &str| -> u32 {
        let n = if v.starts_with('_') {
            v.get(1..).unwrap()
        } else {
            v
        };
        match t {
            "A16QI" => 8,
            "AM16QI" => 8,
            "V16QI" => 8,
            "V32QI" => 8,
            "A32QI" => 8,
            "AM32QI" => 8,
            "V8HI" => 16,
            "V16HI" => 16,
            "V4SI" => 32,
            "V8SI" => 32,
            "V2DI" => 64,
            "V4DI" => 64,
            "UV16QI" => 8,
            "UV32QI" => 8,
            "UV8HI" => 16,
            "UV16HI" => 16,
            "UV4SI" => 32,
            "UV8SI" => 32,
            "UV2DI" => 64,
            "UV4DI" => 64,
            "V4SF" => 32,
            "V8SF" => 32,
            "V2DF" => 64,
            "V4DF" => 64,
            "SI" | "DI" | "USI" | "UDI" | "UQI" | "QI" | "CVPOINTER" | "HI" => 0,
            _ => panic!("unknown type: {t}"),
        }
    };
    let type_to_rp = |t: &str| -> Option<u32> {
        match t {
            "SI" | "DI" | "USI" | "UDI" | "UQI" | "QI" | "HI" | => None,
            "V32QI" | "V16HI" | "V8SI" | "V4DI" | "UV32QI" | "UV16HI" | "UV8SI" | "UV4DI"
            | "V8SF" | "V4DF" => Some(4)
            _ => Some(2),
        }
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

    Ok(Intrinsic {
        name: current_name.to_string(),
        arguments,
        results: results,
        arch_tags: vec![target.to_string()],
    })
}