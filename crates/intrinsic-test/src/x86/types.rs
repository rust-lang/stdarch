use std::str::FromStr;

use super::intrinsic::X86IntrinsicType;
use crate::common::cli::Language;
use crate::common::intrinsic_helpers::{IntrinsicType, IntrinsicTypeDefinition, TypeKind};

impl IntrinsicTypeDefinition for X86IntrinsicType {
    /// Gets a string containing the typename for this type in C format.
    fn c_type(&self) -> String {
        let part_0 = if self.constant { "const" } else { "" };
        let part_1 = match self.kind {
            TypeKind::Int(false) => "unsigned int",
            TypeKind::Char(false) => "unsigned char",
            _ => self.kind.c_prefix(),
        };
        let part_2 = if self.ptr { "*" } else { "" };

        String::from(vec![part_0, part_1, part_2].join(" ").trim())
    }

    fn c_single_vector_type(&self) -> String {
        todo!("c_single_vector_type for X86IntrinsicType needs to be implemented!");
    }

    fn rust_type(&self) -> String {
        todo!("rust_type for X86IntrinsicType needs to be implemented!");
    }

    /// Determines the load function for this type.
    fn get_load_function(&self, language: Language) -> String {
        todo!("get_load_function for X86IntrinsicType needs to be implemented!");
    }

    /// Determines the get lane function for this type.
    fn get_lane_function(&self) -> String {
        todo!("get_lane_function for X86IntrinsicType needs to be implemented!");
    }

    fn from_c(s: &str, target: &String) -> Result<Self, String> {
        let mut s_copy = s.to_string();
        s_copy = s_copy
            .replace("*", "")
            .replace("constexpr", "")
            .replace("const", "")
            .replace("literal", "");

        let s_split = s_copy
            .split(" ")
            .filter_map(|s| if s.len() == 0 { None } else { Some(s) })
            .last();

        // TODO: add more intrinsics by modifying
        // functionality below this line.
        // Currently all the intrinsics that have an "_"
        // is ignored.
        if s.matches("_").next().is_some() {
            return Err(String::from("This functionality needs to be implemented"));
        };

        // TODO: make the unwrapping safe
        let kind = TypeKind::from_str(s_split.unwrap()).expect("Unable to parse type!");

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
        let mut constant = false;
        let mut ptr = false;

        if s.matches("const").next().is_some() {
            constant = true;
        };
        if s.matches("*").next().is_some() {
            ptr = true;
        };

        Ok(X86IntrinsicType(IntrinsicType {
            ptr,
            ptr_constant,
            constant,
            kind,
            bit_len: None,
            simd_len: None,
            vec_len: None,
            target: target.to_string(),
        }))
    }
}
