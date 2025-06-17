use std::collections::HashMap;
use std::str::FromStr;

use itertools::Itertools;

use super::intrinsic::X86IntrinsicType;
use crate::common::cli::Language;
use crate::common::intrinsic_helpers::{IntrinsicType, IntrinsicTypeDefinition, TypeKind};

impl IntrinsicTypeDefinition for X86IntrinsicType {
    /// Gets a string containing the type in C format.
    /// This function assumes that this value is present in the metadata hashmap.
    fn c_type(&self) -> String {
        self.metadata
            .get("type")
            .expect("Failed to extract the C typename in X86!")
            .to_string()
    }

    fn c_single_vector_type(&self) -> String {
        todo!("c_single_vector_type for X86IntrinsicType needs to be implemented!");
    }

    fn rust_type(&self) -> String {
        todo!("rust_type for X86IntrinsicType needs to be implemented!");
    }

    /// Determines the load function for this type.
    fn get_load_function(&self, _language: Language) -> String {
        todo!("get_load_function for X86IntrinsicType needs to be implemented!");
    }

    /// Determines the get lane function for this type.
    fn get_lane_function(&self) -> String {
        todo!("get_lane_function for X86IntrinsicType needs to be implemented!");
    }

    fn from_c(s: &str) -> Result<Self, String> {
        let mut s_copy = s.to_string();
        let mut metadata: HashMap<String, String> = HashMap::new();
        metadata.insert("type".to_string(), s.to_string());
        s_copy = s_copy
            .replace("*", "")
            .replace("_", "")
            .replace("constexpr", "")
            .replace("const", "")
            .replace("literal", "");

        let s_split = s_copy
            .split(" ")
            .filter_map(|s| if s.len() == 0 { None } else { Some(s) })
            .last();

        let s_split = s_split.map(|s| s.chars().filter(|c| !c.is_numeric()).join(""));

        // TODO: make the unwrapping safe
        let kind = TypeKind::from_str(s_split.unwrap().trim()).unwrap_or(TypeKind::Void);

        let kind = if s.find("unsigned").is_some() {
            match kind {
                TypeKind::Int(_) => TypeKind::Int(false),
                TypeKind::Char(_) => TypeKind::Char(false),
                a => a,
            }
        } else {
            kind
        };

        let ptr_constant = false;
        let constant = s.matches("const").next().is_some();
        let ptr = s.matches("*").next().is_some();

        Ok(X86IntrinsicType(IntrinsicType {
            ptr,
            ptr_constant,
            constant,
            kind,
            bit_len: None,
            simd_len: None,
            vec_len: None,
            metadata,
        }))
    }
}

impl X86IntrinsicType {
    pub fn from_param(param: &Parameter) -> Result<Self, String> {
        match Self::from_c(param.type_data.as_str()) {
            Err(message) => Err(message),
            Ok(mut ret) => {
                // First correct the type of the parameter using param.etype.
                // The assumption is that the parameter of type void may have param.type
                // as "__m128i", "__mmask8" and the like.
                ret.set_metadata("etype".to_string(), param.etype.clone());
                if !param.etype.is_empty() {
                    match TypeKind::from_str(param.etype.as_str()) {
                        Ok(value) => {
                            ret.kind = value;
                        }
                        Err(_) => {}
                    };
                }

                // check for param.etype.
                // extract the numeric part and set as bit-len
                // If param.etype is not present, guess the default bit-len

                let mut etype_processed = param.etype.clone();
                etype_processed.retain(|c| c.is_numeric());

                match str::parse::<u32>(etype_processed.as_str()) {
                    Ok(value) => ret.bit_len = Some(value),
                    Err(_) => {
                        ret.bit_len = match ret.kind() {
                            TypeKind::Char(_) => Some(8),
                            TypeKind::BFloat => Some(16),
                            TypeKind::Int(_) => Some(32),
                            TypeKind::Float => Some(32),
                            _ => None,
                        };
                    }
                }

                // then check the param.type and extract numeric part if there are double
                // underscores. divide this number with bit-len and set this as simd-len.

                let mut type_processed = param.etype.clone();
                type_processed.retain(|c| c.is_numeric());

                ret.vec_len = match str::parse::<u32>(etype_processed.as_str()) {
                    // If bit_len is None, vec_len will be None.
                    // Else vec_len will be (num_bits / bit_len).
                    Ok(num_bits) => ret.bit_len.and(Some(num_bits / ret.bit_len.unwrap())),
                    Err(_) => None,
                };

                // if param.etype == IMM, then it is a constant.
                // else it stays unchanged.
                ret.constant |= param.etype == "IMM";

                Ok(ret)
            }
        }
        // Tile types won't currently reach here, since the intrinsic that involve them
        // often return "null" type. Such intrinsics are not tested in `intrinsic-test`
        // currently and are filtered out at `mod.rs`.
    }
}
