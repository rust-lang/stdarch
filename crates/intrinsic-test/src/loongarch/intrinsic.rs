use crate::common::argument::ArgumentList;
use crate::common::indentation::Indentation;
use crate::common::intrinsic::{Intrinsic, IntrinsicDefinition};
use crate::common::intrinsic_helpers::{IntrinsicType, IntrinsicTypeDefinition, Sign, TypeKind};
use std::ops::{Deref, DerefMut};

#[derive(Debug, Clone, PartialEq)]
pub struct LoongArchIntrinsicType {
    pub data: IntrinsicType,
}

impl Deref for LoongArchIntrinsicType {
    type Target = IntrinsicType;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for LoongArchIntrinsicType {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl IntrinsicDefinition<LoongArchIntrinsicType> for Intrinsic<LoongArchIntrinsicType> {
    fn arguments(&self) -> ArgumentList<LoongArchIntrinsicType> {
        self.arguments.clone()
    }

    fn results(&self) -> LoongArchIntrinsicType {
        self.results.clone()
    }

    fn name(&self) -> String {
        self.name.clone()
    }

    /// Generates a std::cout for the intrinsics results that will match the
    /// rust debug output format for the return type. The generated line assumes
    /// there is an int i in scope which is the current pass number.
    fn print_result_c(&self, indentation: Indentation, additional: &str) -> String {
        unimplemented!("print_result_c of IntrinsicDefinition<LoongArchIntrinsicType> is not defined!")
    }
}
