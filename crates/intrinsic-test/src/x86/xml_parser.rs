use crate::common::{
    argument::Argument, intrinsic::Intrinsic, intrinsic_helpers::IntrinsicTypeDefinition,
};
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

#[derive(Deserialize)]
struct Data {
    #[serde(rename = "intrinsic", default)]
    intrinsics: Vec<XMLIntrinsic>,
}

#[derive(Deserialize)]
struct XMLIntrinsic {
    #[serde(rename = "return")]
    return_data: Return,
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
}

#[derive(Deserialize)]
struct Return {
    #[serde(rename = "@type", default)]
    type_data: String,
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

    // println!("{} intrinsics found", data.intrinsics.len());
    let parsed_intrinsics: Vec<Intrinsic<X86IntrinsicType>> = data
        .intrinsics
        .into_iter()
        .filter_map(|intr| {
            Some(xml_to_intrinsic(intr, target).expect("Couldn't parse XML properly!"))
        })
        .collect();

    Ok(parsed_intrinsics)
}

pub fn xml_to_intrinsic(
    mut intr: XMLIntrinsic,
    target: &String,
) -> Result<Intrinsic<X86IntrinsicType>, Box<dyn std::error::Error>> {
    let name = intr.name;
    let results = X86IntrinsicType::from_c(&intr.return_data.type_data, target)?;

    let arguments: Vec<_> = intr
        .parameters
        .into_iter()
        .enumerate()
        .map(|(i, arg)| {
            // let arg_name = Argument::<X86IntrinsicType>::type_and_name_from_c(&arg).1;
        })
        .collect();

    todo!("xml_to_intrinsic needs to collect the arguments properly!");
}
