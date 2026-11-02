use crate::common::intrinsic_helpers::IntrinsicType;
use std::ops::{Deref, DerefMut};

#[derive(Debug, Clone, PartialEq)]
pub struct RiscvIntrinsicType(pub IntrinsicType);

impl Deref for RiscvIntrinsicType {
    type Target = IntrinsicType;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for RiscvIntrinsicType {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
