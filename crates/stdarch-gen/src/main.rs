use std::fs::File;
use std::io::prelude::*;
use std::io::{self, BufReader};
use std::process::Command;

const IN: &str = "neon.spec";
const ARM_OUT: &str = "arm.rs";
const AARCH64_OUT: &str = "aarch64.rs";

const UINT_TYPES: [&'static str; 6] = [
    "uint8x8_t",
    "uint8x16_t",
    "uint16x4_t",
    "uint16x8_t",
    "uint32x2_t",
    "uint32x4_t",
];

const UINT_TYPES_64: [&'static str; 2] = ["uint64x1_t", "uint64x2_t"];

const INT_TYPES: [&'static str; 6] = [
    "int8x8_t",
    "int8x16_t",
    "int16x4_t",
    "int16x8_t",
    "int32x2_t",
    "int32x4_t",
];

const INT_TYPES_64: [&'static str; 2] = ["int64x1_t", "int64x2_t"];

const FLOAT_TYPES: [&'static str; 2] = [
    //"float8x8_t", not supported by rust
    //"float8x16_t", not supported by rust
    //"float16x4_t", not supported by rust
    //"float16x8_t", not supported by rust
    "float32x2_t",
    "float32x4_t",
];

const FLOAT_TYPES_64: [&'static str; 2] = [
    //"float8x8_t", not supported by rust
    //"float8x16_t", not supported by rust
    //"float16x4_t", not supported by rust
    //"float16x8_t", not supported by rust
    "float64x1_t",
    "float64x2_t",
];

fn type_len(t: &str) -> usize {
    match t {
        "int8x8_t" => 8,
        "int8x16_t" => 16,
        "int16x4_t" => 4,
        "int16x8_t" => 8,
        "int32x2_t" => 2,
        "int32x4_t" => 4,
        "int64x1_t" => 1,
        "int64x2_t" => 2,
        "uint8x8_t" => 8,
        "uint8x16_t" => 16,
        "uint16x4_t" => 4,
        "uint16x8_t" => 8,
        "uint32x2_t" => 2,
        "uint32x4_t" => 4,
        "uint64x1_t" => 1,
        "uint64x2_t" => 2,
        "float16x4_t" => 4,
        "float16x8_t" => 8,
        "float32x2_t" => 2,
        "float32x4_t" => 4,
        "float64x1_t" => 1,
        "float64x2_t" => 2,
        "poly64x1_t" => 1,
        "poly64x2_t" => 2,
        _ => panic!("unknown type: {}", t),
    }
}

fn type_to_suffix(t: &str) -> &str {
    match t {
        "int8x8_t" => "_s8",
        "int8x16_t" => "q_s8",
        "int16x4_t" => "_s16",
        "int16x8_t" => "q_s16",
        "int32x2_t" => "_s32",
        "int32x4_t" => "q_s32",
        "int64x1_t" => "_s64",
        "int64x2_t" => "q_s64",
        "uint8x8_t" => "_u8",
        "uint8x16_t" => "q_u8",
        "uint16x4_t" => "_u16",
        "uint16x8_t" => "q_u16",
        "uint32x2_t" => "_u32",
        "uint32x4_t" => "q_u32",
        "uint64x1_t" => "_u64",
        "uint64x2_t" => "q_u64",
        "float16x4_t" => "_f16",
        "float16x8_t" => "q_f16",
        "float32x2_t" => "_f32",
        "float32x4_t" => "q_f32",
        "float64x1_t" => "_f64",
        "float64x2_t" => "q_f64",
        "poly64x1_t" => "_p64",
        "poly64x2_t" => "q_p64",
        _ => panic!("unknown type: {}", t),
    }
}

fn type_to_global_type(t: &str) -> &str {
    match t {
        "int8x8_t" => "i8x8",
        "int8x16_t" => "i8x16",
        "int16x4_t" => "i16x4",
        "int16x8_t" => "i16x8",
        "int32x2_t" => "i32x2",
        "int32x4_t" => "i32x4",
        "int64x1_t" => "i64x1",
        "int64x2_t" => "i64x2",
        "uint8x8_t" => "u8x8",
        "uint8x16_t" => "u8x16",
        "uint16x4_t" => "u16x4",
        "uint16x8_t" => "u16x8",
        "uint32x2_t" => "u32x2",
        "uint32x4_t" => "u32x4",
        "uint64x1_t" => "u64x1",
        "uint64x2_t" => "u64x2",
        "float16x4_t" => "f16x4",
        "float16x8_t" => "f16x8",
        "float32x2_t" => "f32x2",
        "float32x4_t" => "f32x4",
        "float64x1_t" => "f64",
        "float64x2_t" => "f64x2",
        "poly64x1_t" => "i64x1",
        "poly64x2_t" => "i64x2",
        _ => panic!("unknown type: {}", t),
    }
}

fn type_to_ext(t: &str) -> &str {
    match t {
        "int8x8_t" => "v8i8",
        "int8x16_t" => "v16i8",
        "int16x4_t" => "v4i16",
        "int16x8_t" => "v8i16",
        "int32x2_t" => "v2i32",
        "int32x4_t" => "v4i32",
        "int64x1_t" => "v1i64",
        "int64x2_t" => "v2i64",
        "uint8x8_t" => "v8i8",
        "uint8x16_t" => "v16i8",
        "uint16x4_t" => "v4i16",
        "uint16x8_t" => "v8i16",
        "uint32x2_t" => "v2i32",
        "uint32x4_t" => "v4i32",
        "uint64x1_t" => "v1i64",
        "uint64x2_t" => "v2i64",
        "float16x4_t" => "v4f16",
        "float16x8_t" => "v8f16",
        "float32x2_t" => "v2f32",
        "float32x4_t" => "v4f32",
        "float64x1_t" => "v1f64",
        "float64x2_t" => "v2f64",
        /*
        "poly64x1_t" => "i64x1",
        "poly64x2_t" => "i64x2",
        */
        _ => panic!("unknown type for extension: {}", t),
    }
}

fn values(t: &str, vs: &[String]) -> String {
    if vs.len() == 1 && !t.contains("x") {
        format!(": {} = {}", t, vs[0])
    } else {
        format!(
            ": {} = {}::new({})",
            t,
            t,
            vs.iter()
                .map(|v| map_val(t, v))
                .collect::<Vec<&str>>()
                .join(", ")
        )
    }
}

fn map_val<'v>(t: &str, v: &'v str) -> &'v str {
    match v {
        "FALSE" => "0",
        "TRUE" => match &t[..3] {
            "u8x" => "0xFF",
            "u16" => "0xFF_FF",
            "u32" => "0xFF_FF_FF_FF",
            "u64" => "0xFF_FF_FF_FF_FF_FF_FF_FF",
            _ => panic!("No TRUE for type {}", t),
        },
        o => o,
    }
}

fn main() -> io::Result<()> {
    let f = File::open(IN).expect("Failed to open neon.spec");
    let f = BufReader::new(f);

    let mut current_comment = String::new();
    let mut current_name: Option<String> = None;
    let mut current_fn: Option<String> = None;
    let mut current_arm: Option<String> = None;
    let mut current_aarch64: Option<String> = None;
    let mut link_arm: Option<String> = None;
    let mut link_aarch64: Option<String> = None;
    let mut a: Vec<String> = Vec::new();
    let mut b: Vec<String> = Vec::new();
    let mut e: Vec<String> = Vec::new();
    //
    // THIS FILE IS GENERATED FORM neon.spec DO NOT CHANGE IT MANUALLY
    //
    let mut out_arm = String::from(
        r#"
use super::*;
#[cfg(test)]
use stdarch_test::assert_instr;
"#,
    );
    let mut tests_arm = String::from(
        r#"
#[cfg(test)]
mod test {
    use super::*;
    use crate::core_arch::simd::*;
    use std::mem::transmute;
    use stdarch_test::simd_test;
"#,
    );
    //
    // THIS FILE IS GENERATED FORM neon.spec DO NOT CHANGE IT MANUALLY
    //
    let mut out_aarch64 = String::from(
        r#"
use super::*;
#[cfg(test)]
use stdarch_test::assert_instr;
"#,
    );
    let mut tests_aarch64 = String::from(
        r#"
#[cfg(test)]
mod test {
    use super::*;
    use crate::core_arch::simd::*;
    use std::mem::transmute;
    use stdarch_test::simd_test;
"#,
    );

    for line in f.lines() {
        let line = line.unwrap();
        if line.is_empty() {
            continue;
        }
        if line.starts_with("/// ") {
            current_comment = line;
            current_name = None;
            current_fn = None;
            current_arm = None;
            current_aarch64 = None;
            link_aarch64 = None;
            link_arm = None;
        } else if line.starts_with("//") {
        } else if line.starts_with("name = ") {
            current_name = Some(String::from(&line[7..]));
        } else if line.starts_with("fn = ") {
            current_fn = Some(String::from(&line[5..]));
        } else if line.starts_with("arm = ") {
            current_arm = Some(String::from(&line[6..]));
        } else if line.starts_with("aarch64 = ") {
            current_aarch64 = Some(String::from(&line[10..]));
        } else if line.starts_with("a = ") {
            a = line[4..].split(',').map(|v| v.trim().to_string()).collect();
        } else if line.starts_with("b = ") {
            b = line[4..].split(',').map(|v| v.trim().to_string()).collect();
        } else if line.starts_with("e = ") {
            e = line[4..].split(',').map(|v| v.trim().to_string()).collect();
        } else if line.starts_with("link-aarch64 = ") {
            link_aarch64 = Some(String::from(&line[15..]));
        } else if line.starts_with("link-arm = ") {
            link_arm = Some(String::from(&line[11..]));
        } else if line.starts_with("generate ") {
            let line = &line[9..];
            let types: Vec<String> = line
                .split(",")
                .map(|v| v.trim().to_string())
                .flat_map(|v| match v.as_str() {
                    "uint*_t" => UINT_TYPES.iter().map(|v| v.to_string()).collect(),
                    "uint64x*_t" => UINT_TYPES_64.iter().map(|v| v.to_string()).collect(),
                    "int*_t" => INT_TYPES.iter().map(|v| v.to_string()).collect(),
                    "int64x*_t" => INT_TYPES_64.iter().map(|v| v.to_string()).collect(),
                    "float*_t" => FLOAT_TYPES.iter().map(|v| v.to_string()).collect(),
                    "float64x*_t" => FLOAT_TYPES_64.iter().map(|v| v.to_string()).collect(),
                    _ => vec![v],
                })
                .collect();

            for line in types {
                let spec: Vec<&str> = line.split(':').map(|e| e.trim()).collect();
                let in_t;
                let out_t;
                if spec.len() == 1 {
                    in_t = spec[0];
                    out_t = spec[0];
                } else if spec.len() == 2 {
                    in_t = spec[0];
                    out_t = spec[1];
                } else {
                    panic!("Bad spec: {}", line)
                }
                let current_name = current_name.clone().unwrap();
                let name = format!("{}{}", current_name, type_to_suffix(in_t),);
                let a: Vec<String> = a.iter().take(type_len(in_t)).cloned().collect();
                let b: Vec<String> = b.iter().take(type_len(in_t)).cloned().collect();
                let e: Vec<String> = e.iter().take(type_len(in_t)).cloned().collect();
                let globla_t = type_to_global_type(in_t);
                let globla_ret_t = type_to_global_type(out_t);

                if let Some(current_arm) = current_arm.clone() {
                    let current_aarch64 = current_aarch64.clone().unwrap_or(current_arm.clone());

                    let current_fn = if let Some(current_fn) = current_fn.clone() {
                        if link_aarch64.is_some() || link_arm.is_some() {
                            panic!(
                                "[{}] Can't specify link and function at the same time. {} / {:?} / {:?}",
                                name, current_fn, link_aarch64, link_arm
                            )
                        }
                        current_fn
                    } else {
                        if link_aarch64.is_none() || link_arm.is_none() {
                            panic!(
                                "[{}] Either fn or link-arm and link-aarch have to be specified.",
                                name
                            )
                        }
                        format!("{}_", name)
                    };

                    let link = if let (Some(link_arm), Some(link_aarch64)) =
                        (link_arm.clone(), link_aarch64.clone())
                    {
                        let ext = type_to_ext(in_t);

                        format!(
                            r#"#[allow(improper_ctypes)]
    extern "C" {{
        #[cfg_attr(target_arch = "arm", link_name = "llvm.arm.neon.{}")]
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.{}")]
        fn {}(a: {}, b: {}) -> {};
    }}
"#,
                            link_arm.replace("_EXT_", ext),
                            link_aarch64.replace("_EXT_", ext),
                            current_fn,
                            in_t,
                            in_t,
                            out_t
                        )
                    } else {
                        String::new()
                    };

                    let function = format!(
                        r#"
{}
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(target_arch = "arm", target_feature(enable = "v7"))]
#[cfg_attr(all(test, target_arch = "arm"), assert_instr({}))]
#[cfg_attr(all(test, target_arch = "aarch64"), assert_instr({}))]
pub unsafe fn {}(a: {}, b: {}) -> {} {{
    {}{}(a, b)
}}
"#,
                        current_comment,
                        current_arm,
                        current_aarch64,
                        name,
                        in_t,
                        in_t,
                        out_t,
                        link,
                        current_fn,
                    );
                    let test = format!(
                        r#"
    #[simd_test(enable = "neon")]
    unsafe fn test_{}() {{
        let a{};
        let b{};
        let e{};
        let r: {} = transmute({}(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }}
"#,
                        name,
                        values(globla_t, &a),
                        values(globla_t, &b),
                        values(globla_ret_t, &e),
                        globla_ret_t,
                        name
                    );
                    //out_arm.push_str(&link);
                    out_arm.push_str(&function);
                    tests_arm.push_str(&test);
                } else {
                    let current_fn = if let Some(current_fn) = current_fn.clone() {
                        if link_aarch64.is_some() {
                            panic!("[{}] Can't specify link and fn at the same time.", name)
                        }
                        current_fn
                    } else {
                        if link_aarch64.is_none() {
                            panic!("[{}] Either fn or link-aarch have to be specified.", name)
                        }
                        format!("{}_", name)
                    };
                    let current_aarch64 = current_aarch64.clone().unwrap();
                    let link = if let Some(link_aarch64) = link_aarch64.clone() {
                        let ext = type_to_ext(in_t);

                        format!(
                            r#"
    #[allow(improper_ctypes)]
    extern "C" {{
        #[cfg_attr(target_arch = "aarch64", link_name = "llvm.aarch64.neon.{}")]
        fn {}(a: {}, a: {}) -> {};
    }}
"#,
                            link_aarch64.replace("_EXT_", ext),
                            current_fn,
                            in_t,
                            in_t,
                            out_t
                        )
                    } else {
                        String::new()
                    };
                    let function = format!(
                        r#"
{}
#[inline]
#[target_feature(enable = "neon")]
#[cfg_attr(test, assert_instr({}))]
pub unsafe fn {}(a: {}, b: {}) -> {} {{
    {}{}(a, b)
}}
"#,
                        current_comment, current_aarch64, name, in_t, in_t, out_t, link, current_fn,
                    );
                    let test = format!(
                        r#"
    #[simd_test(enable = "neon")]
    unsafe fn test_{}() {{
        let a{};
        let b{};
        let e{};
        let r: {} = transmute({}(transmute(a), transmute(b)));
        assert_eq!(r, e);
    }}
"#,
                        name,
                        values(globla_t, &a),
                        values(globla_t, &b),
                        values(globla_ret_t, &e),
                        globla_ret_t,
                        name
                    );
                    // out_aarch64.push_str(&link);
                    out_aarch64.push_str(&function);
                    tests_aarch64.push_str(&test);
                }
            }
        }
    }
    tests_arm.push('}');
    tests_arm.push('\n');
    tests_aarch64.push('}');
    tests_aarch64.push('\n');

    let mut file_arm = File::create(ARM_OUT)?;
    file_arm.write_all(out_arm.as_bytes())?;
    file_arm.write_all(tests_arm.as_bytes())?;

    let mut file_aarch = File::create(AARCH64_OUT)?;
    file_aarch.write_all(out_aarch64.as_bytes())?;
    file_aarch.write_all(tests_aarch64.as_bytes())?;
    Command::new("rustfmt")
        .arg(ARM_OUT)
        .arg(AARCH64_OUT)
        .status()
        .expect("failed to execute process");
    Ok(())
}
