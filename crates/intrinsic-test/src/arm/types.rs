use std::collections::HashMap;

use super::intrinsic::ArmIntrinsicType;
use crate::common::cli::Language;
use crate::common::intrinsic_helpers::{IntrinsicType, IntrinsicTypeDefinition, Sign, TypeKind};

impl IntrinsicTypeDefinition for ArmIntrinsicType {
    /// Gets a string containing the typename for this type in C format.
    /// This assumes that the metadata hashmap contains this value at the
    /// "type" key
    fn c_type(&self) -> String {
        self.metadata
            .get("type")
            .expect("Failed to extract the C typename in Aarch!")
            .replace("*", "")
            .replace("  ", "")
            .trim()
            .to_string()
    }

    fn c_single_vector_type(&self) -> String {
        if let (Some(bit_len), Some(simd_len)) = (self.0.bit_len, self.0.simd_len) {
            format!(
                "{prefix}{bit_len}x{simd_len}_t",
                prefix = self.0.kind.c_prefix()
            )
        } else {
            unreachable!("Shouldn't be called on this type")
        }
    }

    fn rust_type(&self) -> String {
        let rust_prefix = self.0.kind.rust_prefix();
        let c_prefix = self.0.kind.c_prefix();
        if self.0.ptr_constant {
            self.c_type()
        } else if let (Some(bit_len), simd_len, vec_len) =
            (self.0.bit_len, self.0.simd_len, self.0.vec_len)
        {
            match (simd_len, vec_len) {
                (None, None) => format!("{rust_prefix}{bit_len}"),
                (Some(simd), None) => format!("{c_prefix}{bit_len}x{simd}_t"),
                (Some(simd), Some(vec)) => format!("{c_prefix}{bit_len}x{simd}x{vec}_t"),
                (None, Some(_)) => todo!("{:#?}", self), // Likely an invalid case
            }
        } else {
            todo!("{:#?}", self)
        }
    }

    /// Determines the load function for this type.
    fn get_load_function(&self, language: Language) -> String {
        if let IntrinsicType {
            kind: k,
            bit_len: Some(bl),
            simd_len,
            vec_len,
            metadata,
            ..
        } = &self.0
        {
            let quad = if simd_len.unwrap_or(1) * bl > 64 {
                "q"
            } else {
                ""
            };

            let choose_workaround = language == Language::C
                && metadata
                    .get("target")
                    .filter(|value| value.contains("v7"))
                    .is_some();
            format!(
                "vld{len}{quad}_{type}{size}",
                type = match k {
                    TypeKind::Int(Sign::Unsigned) => "u",
                    TypeKind::Int(Sign::Signed) => "s",
                    TypeKind::Float => "f",
                    // The ACLE doesn't support 64-bit polynomial loads on Armv7
                    // if armv7 and bl == 64, use "s", else "p"
                    TypeKind::Poly => if choose_workaround && *bl == 64 {"s"} else {"p"},
                    x => todo!("get_load_function TypeKind: {:#?}", x),
                },
                size = bl,
                quad = quad,
                len = vec_len.unwrap_or(1),
            )
        } else {
            todo!("get_load_function IntrinsicType: {:#?}", self)
        }
    }

    /// Determines the get lane function for this type.
    fn get_lane_function(&self) -> String {
        if let IntrinsicType {
            kind: k,
            bit_len: Some(bl),
            simd_len,
            ..
        } = &self.0
        {
            let quad = if (simd_len.unwrap_or(1) * bl) > 64 {
                "q"
            } else {
                ""
            };
            format!(
                "vget{quad}_lane_{type}{size}",
                type = match k {
                    TypeKind::Int(Sign::Unsigned) => "u",
                    TypeKind::Int(Sign::Signed) => "s",
                    TypeKind::Float => "f",
                    TypeKind::Poly => "p",
                    x => todo!("get_load_function TypeKind: {:#?}", x),
                },
                size = bl,
                quad = quad,
            )
        } else {
            todo!("get_lane_function IntrinsicType: {:#?}", self)
        }
    }

    fn from_c(s: &str) -> Result<Self, String> {
        const CONST_STR: &str = "const";
        let mut metadata: HashMap<String, String> = HashMap::new();
        metadata.insert("type".to_string(), s.to_string());
        if let Some(s) = s.strip_suffix('*') {
            let (s, constant) = match s.trim().strip_suffix(CONST_STR) {
                Some(stripped) => (stripped, true),
                None => (s, false),
            };
            let s = s.trim_end();
            let temp_return = ArmIntrinsicType::from_c(s);
            temp_return.map(|mut op| {
                op.ptr = true;
                op.ptr_constant = constant;
                op
            })
        } else {
            // [const ]TYPE[{bitlen}[x{simdlen}[x{vec_len}]]][_t]
            let (mut s, constant) = match s.strip_prefix(CONST_STR) {
                Some(stripped) => (stripped.trim(), true),
                None => (s, false),
            };
            s = s.strip_suffix("_t").unwrap_or(s);
            let mut parts = s.split('x'); // [[{bitlen}], [{simdlen}], [{vec_len}] ]
            let start = parts.next().ok_or("Impossible to parse type")?;
            if let Some(digit_start) = start.find(|c: char| c.is_ascii_digit()) {
                let (arg_kind, bit_len) = start.split_at(digit_start);
                let arg_kind = arg_kind.parse::<TypeKind>()?;
                let bit_len = bit_len.parse::<u32>().map_err(|err| err.to_string())?;
                let simd_len = match parts.next() {
                    Some(part) => Some(
                        part.parse::<u32>()
                            .map_err(|_| "Couldn't parse simd_len: {part}")?,
                    ),
                    None => None,
                };
                let vec_len = match parts.next() {
                    Some(part) => Some(
                        part.parse::<u32>()
                            .map_err(|_| "Couldn't parse vec_len: {part}")?,
                    ),
                    None => None,
                };
                Ok(ArmIntrinsicType(IntrinsicType {
                    ptr: false,
                    ptr_constant: false,
                    constant,
                    kind: arg_kind,
                    bit_len: Some(bit_len),
                    simd_len,
                    vec_len,
                    metadata,
                }))
            } else {
                let kind = start.parse::<TypeKind>()?;
                let bit_len = match kind {
                    TypeKind::Int(_) => Some(32),
                    _ => None,
                };
                Ok(ArmIntrinsicType(IntrinsicType {
                    ptr: false,
                    ptr_constant: false,
                    constant,
                    kind: start.parse::<TypeKind>()?,
                    bit_len,
                    simd_len: None,
                    vec_len: None,
                    metadata,
                }))
            }
        }
    }
}
