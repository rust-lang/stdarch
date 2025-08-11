use super::intrinsic::LoongArchIntrinsicType;
use crate::common::cli::Language;
use crate::common::intrinsic_helpers::IntrinsicTypeDefinition;

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

impl LoongArchIntrinsicType
