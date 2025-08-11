use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::common::argument::{Argument, ArgumentList};
use crate::common::intrinsic::Intrinsic;
use crate::loongarch::intrinsic::LoongArchIntrinsicType;

pub fn get_loongson_intrinsics(
    path: &Path,
    target: &str
) -> Result<Vec<Intrinsic<LoongArchIntrinsicType>>, Box<dyn std::error::Error>> {
    let f = File::open(path).unwrap_or_else(|_| panic!("Failed to open {}", path.display()));
    let f = BufReader::new(f);
    
    let mut current_name: Option<String> = None;
    let mut asm_fmts: Vec<String> = Vec::new();

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
            let mut data_types: Vec<String> = line
                .get(12..)
                .unwrap()
                .split(',')
                .map(|e| e.trim().to_string())
                .collect();
            let arguments;
            let return_type;
            let data_types_len = data_types.len();
            if data_types_len > 0 && data_types_len < 6 {
                arguments = data_types.split_off(1);
                
                // Being explicit here with the variable name
                return_type = data_types.get(0).unwrap();
            }  else {
                panic!("DEBUG: line: {0} len: {1}", line, data_types.len());
            }

            let intrinsic = gen_intrinsic(current_name.as_str(), asm_fmts.clone(), arguments, return_type, target);
            if intrinsic.is_ok() {
                intrinsics.push(intrinsic.unwrap());
            }
        }
    };
    return Ok(intrinsics)
}

fn gen_intrinsic(
    current_name: &str,
    asm_fmts: Vec<String>,
    args: Vec<String>,
    return_type: &String,
    target: &str,
) -> Result<Intrinsic<LoongArchIntrinsicType>, Box<dyn std::error::Error>> {
    let para_num = args.len();
    let mut arguments = asm_fmts
        .iter()
        .zip(args.iter())
        .enumerate()
        .map(|(i, (asm_fmt, arg_type))| {
            let ty = LoongArchIntrinsicType::from_values(asm_fmt, arg_type).unwrap();
            let arg = Argument::<LoongArchIntrinsicType>::new(i, format!("_{i}_{}", arg_type), ty, None);
            return arg;
        })
        .collect::<Vec<Argument<LoongArchIntrinsicType>>>();

    if para_num == 1 && args[0] == "HI" {
        match asm_fmts[1].as_str() {
            "si13" | "i13" => arguments[0].ty.constant = true,
            "si10" => arguments[0].ty.constant = true,
            _ => panic!("unsupported assembly format: {:?}", asm_fmts),
        };
    } else if para_num == 2 && (args[1] == "UQI" || args[1] == "USI") {
        if asm_fmts[2].starts_with("ui") {
            arguments[1].ty.constant = true;
        } else {
            panic!("unsupported assembly format: {:?}", asm_fmts);
        };
    } else if para_num == 2 && args[1] == "QI" {
        if asm_fmts[2].starts_with("si") {
            arguments[1].ty.constant = true;
        } else {
            panic!("unsupported assembly format: {:?}", asm_fmts);
        };
    } else if para_num == 2 && args[0] == "CVPOINTER" && args[1] == "SI" {
        if asm_fmts[2].starts_with("si") {
            arguments[1].ty.constant = true;
        } else {
            panic!("unsupported assembly format: {:?}", asm_fmts);
        };
    } else if para_num == 3 && (args[2] == "USI" || args[2] == "UQI") {
        if asm_fmts[2].starts_with("ui") {
            arguments[2].ty.constant = true;
        } else {
            panic!("unsupported assembly format: {:?}", asm_fmts);
        };
    } else if para_num == 3 && args[1] == "CVPOINTER" && args[2] == "SI" {
        match asm_fmts[2].as_str() {
            "si12" => arguments[2].ty.constant = true,
            _ => panic!("unsupported assembly format: {:?}", asm_fmts),
        };
    } else if para_num == 4 {
        match (asm_fmts[3].as_str(), current_name.chars().last().unwrap()) {
            ("si8", t) => {
                arguments[2].ty.constant = true;
                arguments[3].ty.constant = true;
            },
            (_, _) => panic!(
                "unsupported assembly format: {:?} for {}",
                asm_fmts, current_name
            ),
        };
    }
    let results = LoongArchIntrinsicType::from_values(return_type, &asm_fmts[0])?;
    let arguments = ArgumentList::<LoongArchIntrinsicType> { args: arguments };
    Ok(Intrinsic {
        name: current_name.to_string(),
        arguments,
        results: results,
        arch_tags: vec![target.to_string()],
    })
}