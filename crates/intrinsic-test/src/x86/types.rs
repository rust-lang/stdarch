use super::intrinsic::X86IntrinsicType;
use crate::common::cli::Language;
use crate::common::intrinsic_helpers::IntrinsicTypeDefinition;

impl IntrinsicTypeDefinition for X86IntrinsicType {
    /// Gets a string containing the typename for this type in C format.
    fn c_type(&self) -> String {
        todo!("c_type for X86IntrinsicType needs to be implemented!");
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
        todo!("from_c for X86IntrinsicType needs to be implemented!");
    }
}
