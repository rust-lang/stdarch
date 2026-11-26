use crate::{
    common::intrinsic_helpers::{Sign, TypeDefinition, TypeKind},
    riscv::intrinsic::RiscvIntrinsicType,
};

impl TypeDefinition for RiscvIntrinsicType {
    fn load_function(&self) -> String {
        unimplemented!("internal error: reached unimplemented intrinsic kind {self:?}");
    }

    fn c_type(&self) -> String {
        assert!(
            !self.ptr && !self.ptr_constant && self.vec_len.is_none() && self.simd_len.is_none(),
            "internal error: reached unimplemented intrinsic type {self:?}"
        );

        let TypeKind::Int(sign) = self.kind else {
            panic!(
                "internal error: reached unimplemented intrinsic type kind {:?}",
                self.kind
            );
        };
        let bit_len = self.bit_len.unwrap();
        let base_int_type = scalar_c_type_name(sign, bit_len);

        if self.constant {
            format!("const {base_int_type}")
        } else {
            base_int_type.to_string()
        }
    }

    fn rust_type(&self) -> String {
        assert!(
            !self.ptr && !self.ptr_constant && self.vec_len.is_none() && self.simd_len.is_none(),
            "internal error: reached unimplemented intrinsic type {self:?}"
        );

        let TypeKind::Int(sign) = self.kind else {
            panic!(
                "internal error: reached unimplemented intrinsic type kind {:?}",
                self.kind
            );
        };
        let bit_len = self.bit_len.unwrap();
        scalar_rust_type_name(sign, bit_len).to_string()
    }
}

fn scalar_c_type_name(sign: Sign, bit_len: u32) -> &'static str {
    match (sign, bit_len) {
        (Sign::Signed, 32) => "int32_t",
        (Sign::Signed, 64) => "int64_t",
        (Sign::Unsigned, 32) => "uint32_t",
        (Sign::Unsigned, 64) => "uint64_t",
        _ => unreachable!("unsupported RISC-V scalar width {bit_len}"),
    }
}

fn scalar_rust_type_name(sign: Sign, bit_len: u32) -> &'static str {
    match (sign, bit_len) {
        (Sign::Signed, 8) => "i8",
        (Sign::Signed, 16) => "i16",
        (Sign::Signed, 32) => "i32",
        (Sign::Signed, 64) => "i64",
        (Sign::Unsigned, 8) => "u8",
        (Sign::Unsigned, 16) => "u16",
        (Sign::Unsigned, 32) => "u32",
        (Sign::Unsigned, 64) => "u64",
        _ => unreachable!("unsupported RISC-V scalar width {bit_len}"),
    }
}
