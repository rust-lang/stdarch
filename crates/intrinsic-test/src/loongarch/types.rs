use super::intrinsic::LoongArchIntrinsicType;
use crate::common::cli::Language;
<<<<<<< HEAD
=======
use crate::common::intrinsic_helpers::Sign;
>>>>>>> b094de07 (chenj)
use crate::common::intrinsic_helpers::{IntrinsicType, IntrinsicTypeDefinition, TypeKind};

impl IntrinsicTypeDefinition for LoongArchIntrinsicType {
    /// Gets a string containing the type in C format.
    /// This function assumes that this value is present in the metadata hashmap.
    fn c_type(&self) -> String {
        unimplemented!("c_type for LoongArchIntrinsicType is not implemented!")
    }

    fn c_single_vector_type(&self) -> String {
        unimplemented!("c_single_vector_type for LoongArchIntrinsicType is not implemented!")
    }

    /// Determines the load function for this type.
    fn get_load_function(&self, _language: Language) -> String {
        unimplemented!("get_load_function for LoongArchIntrinsicType is not implemented!")
    }

    /// Determines the get lane function for this type.
    fn get_lane_function(&self) -> String {
        todo!("get_lane_function for LoongArchIntrinsicType needs to be implemented!");
    }
}

impl LoongArchIntrinsicType {
    /// Accepts X, Y and Z.
    /// Returns a `LoongArchType`
    pub fn from_values(asm_fmt: &String, data_type: &String) -> Result<Self, String> {
        let bit_len = match data_type.as_str() {
            "A16QI" => Some(8),
            "AM16QI" => Some(8),
            "V16QI" => Some(8),
            "V32QI" => Some(8),
            "A32QI" => Some(8),
            "AM32QI" => Some(8),
            "V8HI" => Some(16),
            "V16HI" => Some(16),
            "V4SI" => Some(32),
            "V8SI" => Some(32),
            "V2DI" => Some(64),
            "V4DI" => Some(64),
            "UV16QI" => Some(8),
            "UV32QI" => Some(8),
            "UV8HI" => Some(16),
            "UV16HI" => Some(16),
            "UV4SI" => Some(32),
            "UV8SI" => Some(32),
            "UV2DI" => Some(64),
            "UV4DI" => Some(64),
            "V4SF" => Some(32),
            "V8SF" => Some(32),
            "V2DF" => Some(64),
            "V4DF" => Some(64),
            "SI" | "DI" | "USI" | "UDI" | "UQI" | "QI" | "CVPOINTER" | "HI" => None,
            _ => panic!("unknown type {data_type} with ASM {asm_fmt}"),
        };

        let vec_len = match data_type.as_str() {
            "SI" | "DI" | "USI" | "UDI" | "UQI" | "QI" | "HI" | => None,
            "V32QI" | "V16HI" | "V8SI" | "V4DI" | "UV32QI" | "UV16HI" | "UV8SI" | "UV4DI"
            | "V8SF" | "V4DF" => Some(4),
            _ => Some(2),
        };

        Ok(LoongArchIntrinsicType {
            data: IntrinsicType {
                constant: false,
                ptr_constant: false,
                ptr: false,
                kind: TypeKind::Mask,
                bit_len,
                vec_len,
                simd_len: None
            }
        })
    }
}
