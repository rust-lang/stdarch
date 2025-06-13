use crate::common::argument::{Argument, ArgumentList};
use crate::common::intrinsic::Intrinsic;
use crate::common::intrinsic_helpers::{IntrinsicType, IntrinsicTypeDefinition, TypeKind};

use serde::{Deserialize, Deserializer};
use std::path::Path;

use super::intrinsic::X86IntrinsicType;

// Custom deserializer function to convert "TRUE"/"FALSE" strings to boolean
fn string_to_bool<'de, D>(deserializer: D) -> Result<bool, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    match s.as_str() {
        "TRUE" => Ok(true),
        "FALSE" => Ok(false),
        _ => Ok(false), // Default to false for any other value
    }
}

// Custom deserializer function to convert strings to u16
fn string_to_u16<'de, D>(deserializer: D) -> Result<u16, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    return Ok(s.as_str().parse::<u16>().unwrap_or(0u16));
}

#[derive(Deserialize)]
struct Data {
    #[serde(rename = "intrinsic", default)]
    intrinsics: Vec<XMLIntrinsic>,
}

#[derive(Deserialize)]
struct XMLIntrinsic {
    #[serde(rename = "return")]
    return_data: Parameter,
    #[serde(rename = "@name")]
    name: String,
    #[serde(rename = "@tech")]
    tech: String,
    #[serde(rename = "CPUID", default)]
    cpuid: Vec<String>,
    #[serde(rename = "parameter", default)]
    parameters: Vec<Parameter>,
    #[serde(rename = "@sequence", default, deserialize_with = "string_to_bool")]
    generates_sequence: bool,
    #[serde(default)]
    instruction: Vec<Instruction>,
}

#[derive(Deserialize)]
struct Parameter {
    #[serde(rename = "@type")]
    type_data: String,
    #[serde(rename = "@etype", default)]
    etype: String,
    #[serde(rename = "@memwidth", default, deserialize_with = "string_to_u16")]
    memwidth: u16,
    #[serde(rename = "@varname", default)]
    var_name: String,
}

#[derive(Deserialize, Debug)]
struct Instruction {
    #[serde(rename = "@name")]
    name: String,
}

pub fn get_xml_intrinsics(
    filename: &Path,
    target: &String,
) -> Result<Vec<Intrinsic<X86IntrinsicType>>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(filename)?;
    let reader = std::io::BufReader::new(file);
    let data: Data =
        quick_xml::de::from_reader(reader).expect("failed to deserialize the source XML file");

    let parsed_intrinsics: Vec<Intrinsic<X86IntrinsicType>> = data
        .intrinsics
        .into_iter()
        .filter_map(|intr| {
            // Some(xml_to_intrinsic(intr, target).expect("Couldn't parse XML properly!"))
            xml_to_intrinsic(intr, target).ok()
        })
        .collect();

    Ok(parsed_intrinsics)
}

fn parse_observable(param: &Parameter, target: &String) -> Result<X86IntrinsicType, String> {
    let ty = X86IntrinsicType::from_c(param.type_data.as_str(), target);

    if let Err(_) = ty {
        return ty;
    }
    let mut ty_bit_len = param.etype.clone();
    ty_bit_len.retain(|c| c.is_numeric());
    let ty_bit_len = str::parse::<u32>(ty_bit_len.as_str()).ok();
    let mut ty = ty.unwrap();
    let ty_bit_len = match ty_bit_len {
        None => match ty.kind {
            TypeKind::Int(_) => Some(32),
            TypeKind::Float => Some(32),
            TypeKind::BFloat => Some(16),
            TypeKind::Void => Some(param.memwidth as u32),
            _ => None,
        },
        ty => ty,
    };
    ty.set_bit_len(ty_bit_len);
    Ok(ty)
}

fn xml_to_intrinsic(
    intr: XMLIntrinsic,
    target: &String,
) -> Result<Intrinsic<X86IntrinsicType>, Box<dyn std::error::Error>> {
    let name = intr.name;
    let result = parse_observable(&intr.return_data, target);
    let args_check = intr.parameters.into_iter().enumerate().map(|(i, param)| {
        let ty = parse_observable(&param, target);
        if let Err(_) = ty {
            return None;
        }
        let constraint = None;
        let mut arg = Argument::<X86IntrinsicType>::new(i, param.var_name, ty.unwrap(), constraint);
        let IntrinsicType {
            ref mut constant, ..
        } = arg.ty.0;
        if param.etype == "IMM" {
            *constant = true
        }
        Some(arg)
    });

    let args = args_check.collect::<Vec<_>>();
    if args.iter().any(|elem| elem.is_none()) {
        return Err(Box::from("intrinsic isn't fully supported in this test!"));
    }
    let args = args
        .into_iter()
        .map(|e| e.unwrap())
        .filter(|arg| arg.ty.ptr || arg.ty.kind != TypeKind::Void)
        .collect::<Vec<_>>();
    let arguments = ArgumentList::<X86IntrinsicType> { args };

    if let Err(message) = result {
        return Err(Box::from(message));
    }
    Ok(Intrinsic {
        name,
        arguments,
        results: result.unwrap(),
        arch_tags: intr.cpuid,
    })
}
