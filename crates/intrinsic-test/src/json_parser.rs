use std::collections::HashMap;
use std::path::Path;

use regex::Regex;
use serde::Deserialize;

use crate::argument::{Argument, ArgumentList, Constraint};
use crate::intrinsic::{Intrinsic, Predication};
use crate::types::IntrinsicType;

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct ReturnType {
    element_bit_size: String,
    value: String,
}

#[derive(Deserialize, Debug)]
#[serde(untagged, deny_unknown_fields)]
pub enum ArgPrep {
    Register {
        #[serde(rename = "register")]
        reg: String,
    },
    Immediate {
        #[serde(rename = "minimum")]
        min: i64,
        #[serde(rename = "maximum")]
        max: i64,
    },
    Nothing {},
}

impl ArgPrep {
    pub fn get_element_size(&self) -> Result<u32, String> {
        // We only rely on argument preparation for the element size of predicates
        // All other sizes are implicit in the type
        lazy_static! {
            // Syntax examples:
            //   Pg.B
            //   Pop2.S
            //   Ptied.Q
            static ref PREP_REGEX: Regex = Regex::new(r"^P[a-z]+\d?\.(B|H|S|D|Q)?").unwrap();
        }
        match self {
            ArgPrep::Register { reg } => {
                if let Some(caps) = PREP_REGEX.captures(reg) {
                    let size_kind = caps.get(1).unwrap().as_str();
                    match size_kind {
                        "B" => Ok(8),
                        "H" => Ok(16),
                        "S" => Ok(32),
                        "D" => Ok(64),
                        "Q" => Ok(128),
                        _ => panic!("{size_kind}?"),
                    }
                } else {
                    Err(format!("Couldn't get element size for register {reg}"))
                }
            }
            _ => Err(format!(
                "Couldn't get element size from argument preparation {self:?}"
            )),
        }
    }
}

#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
struct JsonIntrinsic {
    #[serde(rename = "SIMD_ISA")]
    simd_isa: String,
    name: String,
    arguments: Vec<String>,
    return_type: ReturnType,
    #[serde(rename = "Arguments_Preparation")]
    args_prep: Option<HashMap<String, ArgPrep>>,
    #[serde(rename = "Architectures")]
    architectures: Vec<String>,
    #[serde(rename = "instructions")]
    _ins: Option<Vec<Vec<String>>>,
}

pub fn get_neon_intrinsics(filename: &Path) -> Result<Vec<Intrinsic>, Box<dyn std::error::Error>> {
    get_intrinsics(filename, false)
}

pub fn get_sve_intrinsics(filename: &Path) -> Result<Vec<Intrinsic>, Box<dyn std::error::Error>> {
    get_intrinsics(filename, true)
}

fn get_intrinsics(
    filename: &Path,
    is_sve: bool,
) -> Result<Vec<Intrinsic>, Box<dyn std::error::Error>> {
    let arch = if is_sve { "sve" } else { "Neon" };
    let file = std::fs::File::open(filename)?;
    let reader = std::io::BufReader::new(file);
    let json: Vec<JsonIntrinsic> = serde_json::from_reader(reader).expect("Couldn't parse JSON");

    let parsed = json
        .into_iter()
        .filter_map(|intr| {
            if intr.simd_isa.starts_with(arch) {
                Some(json_to_intrinsic(intr).expect("Couldn't parse JSON"))
            } else {
                None
            }
        })
        .collect();
    Ok(parsed)
}

fn json_to_intrinsic(mut intr: JsonIntrinsic) -> Result<Intrinsic, Box<dyn std::error::Error>> {
    let name = intr.name.replace(['[', ']'], "");

    let mut results = IntrinsicType::from_c(&intr.return_type.value)?;
    results.set_inner_size(intr.return_type.element_bit_size.parse::<u32>()?);

    let mut args_prep = intr.args_prep.as_mut();
    let args = intr
        .arguments
        .into_iter()
        .enumerate()
        .map(|(i, arg)| {
            let arg_name = Argument::type_and_name_from_c(&arg).1;
            let arg_prep = args_prep.as_mut().and_then(|a| a.remove(arg_name));
            let mut arg = Argument::from_c(i, &arg, arg_prep);

            // The JSON doesn't list immediates as const
            if let IntrinsicType::Type {
                ref mut constant, ..
            } = arg.ty
            {
                if arg.name.starts_with("imm") {
                    *constant = true
                }
            }
            if (name.starts_with("svcadd_") || name.starts_with("svqcadd_"))
                && arg_name == "imm_rotation"
            {
                arg.constraints = vec![Constraint::ImmRotationAdd];
            }
            arg
        })
        .collect();

    let arguments = ArgumentList { args };

    let predication = if name.ends_with("_m") {
        Predication::Merging
    } else if name.ends_with("_x") {
        Predication::DontCare
    } else if name.ends_with("_z") || arguments.iter().any(|a| a.is_predicate() && a.name == "pg") {
        // Predicated intrinsics with only a zeroing form typically lack a _z suffix
        Predication::Zeroing
    } else {
        Predication::None
    };

    Ok(Intrinsic {
        name,
        arguments,
        results,
        a64_only: intr.architectures == vec!["A64".to_string()],
        predication,
    })
}
