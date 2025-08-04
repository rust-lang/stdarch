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

    // fn rust_type(&self) -> String {
    //     // handling edge cases first
    //     // the general handling is implemented below
    //     if let Some(val) = self.metadata.get("type") {
    //         match val.as_str() {
    //             "__m128 const *" => {
    //                 return "&__m128".to_string();
    //             }
    //             "__m128d const *" => {
    //                 return "&__m128d".to_string();
    //             }
    //             "const void*" => {
    //                 return "&__m128d".to_string();
    //             }
    //             _ => {}
    //         }
    //     }

    //     if self.kind() == TypeKind::Void && self.ptr {
    //         // this has been handled by default settings in
    //         // the from_param function of X86IntrinsicType
    //         unreachable!()
    //     }

    //     // general handling cases
    //     let core_part = if self.kind() == TypeKind::Mask {
    //         // all types of __mmask<int> are handled here
    //         format!("__mask{}", self.bit_len.unwrap())
    //     } else if self.simd_len.is_some() {
    //         // all types of __m<int> vector types are handled here
    //         let re = Regex::new(r"\__m\d+[a-z]*").unwrap();
    //         let rust_type = self
    //             .metadata
    //             .get("type")
    //             .map(|val| re.find(val).unwrap().as_str());
    //         rust_type.unwrap().to_string()
    //     } else {
    //         format!(
    //             "{}{}",
    //             self.kind.rust_prefix().to_string(),
    //             self.bit_len.unwrap()
    //         )
    //     };

    //     // extracting "memsize" so that even vector types can be involved
    //     let memwidth = self
    //         .metadata
    //         .get("memwidth")
    //         .map(|n| str::parse::<u32>(n).unwrap());
    //     let prefix_part = if self.ptr && self.constant && self.bit_len.eq(&memwidth) {
    //         "&"
    //     } else if self.ptr && self.bit_len.eq(&memwidth) {
    //         "&mut "
    //     } else if self.ptr && self.constant {
    //         "*const "
    //     } else if self.ptr {
    //         "*mut "
    //     } else {
    //         ""
    //     };

    //     return prefix_part.to_string() + core_part.as_str();
    // }

    /// Determines the load function for this type.
    fn get_load_function(&self, _language: Language) -> String {
        unimplemented!("get_load_function for LoongArchIntrinsicType is not implemented!")
    }

    /// Determines the get lane function for this type.
    fn get_lane_function(&self) -> String {
        todo!("get_lane_function for LoongArchIntrinsicType needs to be implemented!");
    }

    fn from_c(s: &str, target: &str) -> Result<Self, String> {
        todo!("from_c for LoongArchIntrinsicType needs to be implemented!");
    }
}
