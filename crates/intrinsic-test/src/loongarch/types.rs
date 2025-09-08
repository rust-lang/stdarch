use super::intrinsic::LoongArchIntrinsicType;
use crate::common::cli::Language;
use crate::common::intrinsic_helpers::Sign;
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
        let (bit_len, vec_len, type_kind) = match data_type.as_str() {
            "A16QI" => (Some(8), Some(16), TypeKind::Int(Sign::Signed)),
            "AM16QI" => (Some(8), Some(16), TypeKind::Int(Sign::Signed)),
            "V16QI" => (Some(8), Some(16), TypeKind::Int(Sign::Signed)),
            "V32QI" => (Some(8), Some(32), TypeKind::Int(Sign::Signed)),
            "A32QI" => (Some(8), Some(32), TypeKind::Int(Sign::Signed)),
            "AM32QI" => (Some(8), Some(32), TypeKind::Int(Sign::Signed)),
            "V8HI" => (Some(16), Some(8), TypeKind::Int(Sign::Signed)),
            "V16HI" => (Some(16), Some(16), TypeKind::Int(Sign::Signed)),
            "V4SI" => (Some(32), Some(4), TypeKind::Int(Sign::Signed)),
            "V8SI" => (Some(32), Some(8), TypeKind::Int(Sign::Signed)),
            "V2DI" => (Some(64), Some(2), TypeKind::Int(Sign::Signed)),
            "V4DI" => (Some(64), Some(4), TypeKind::Int(Sign::Signed)),
            "UV16QI" => (Some(8), Some(16), TypeKind::Int(Sign::Unsigned)),
            "UV32QI" => (Some(8), Some(32), TypeKind::Int(Sign::Unsigned)),
            "UV8HI" => (Some(16), Some(8), TypeKind::Int(Sign::Unsigned)),
            "UV16HI" => (Some(16), Some(16), TypeKind::Int(Sign::Unsigned)),
            "UV4SI" => (Some(32), Some(4), TypeKind::Int(Sign::Unsigned)),
            "UV8SI" => (Some(32), Some(8), TypeKind::Int(Sign::Unsigned)),
            "UV2DI" => (Some(64), Some(2), TypeKind::Int(Sign::Unsigned)),
            "UV4DI" => (Some(64), Some(4), TypeKind::Int(Sign::Unsigned)),
            "V4SF" => (Some(32), Some(4), TypeKind::Float),
            "V8SF" => (Some(32), Some(8), TypeKind::Float),
            "V2DF" => (Some(64), Some(2), TypeKind::Float),
            "V4DF" => (Some(64), Some(4), TypeKind::Float),
            "SI" | "DI" | "USI" | "UDI" | "UQI" | "QI" | "CVPOINTER" | "HI" => {
                (None, None, TypeKind::Int(Sign::Signed))
            }
            _ => panic!("unknown type {data_type} with ASM {asm_fmt}"),
        };

        Ok(LoongArchIntrinsicType {
            data: IntrinsicType {
                constant: false,
                ptr_constant: false,
                ptr: false,
                kind: TypeKind::Mask,
                bit_len,
                vec_len,
                simd_len: None,
            },
        })
    }
}
