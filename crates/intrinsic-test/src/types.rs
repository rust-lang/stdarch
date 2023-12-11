use std::fmt;
use std::str::FromStr;

use crate::format::Indentation;
use crate::values::{value_for_array, MAX_SVE_BITS};
use crate::Language;

use itertools::Itertools;

#[allow(dead_code)]
#[derive(Debug, PartialEq, Copy, Clone)]
pub enum VecLen {
    Scalable,
    Fixed(u32),
}

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum TypeKind {
    BFloat,
    Bool,
    Float,
    Int,
    UInt,
    Poly,
    Void,
    SvPattern,
    SvPrefetchOp,
}

impl FromStr for TypeKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "svbool" | "bool" => Ok(Self::Bool),
            "svbfloat" | "bfloat" => Ok(Self::BFloat),
            "svfloat" | "float" => Ok(Self::Float),
            "svint" | "int" => Ok(Self::Int),
            "svuint" | "uint" | "unsigned" => Ok(Self::UInt),
            "poly" => Ok(Self::Poly),
            "void" => Ok(Self::Void),
            "svpattern" => Ok(Self::SvPattern),
            "svprfop" => Ok(Self::SvPrefetchOp),
            _ => Err(format!("Impossible to parse argument kind {s}")),
        }
    }
}

impl fmt::Display for TypeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Bool => "bool",
                Self::BFloat => "bfloat",
                Self::Float => "float",
                Self::Int => "int",
                Self::UInt => "uint",
                Self::Poly => "poly",
                Self::Void => "void",
                Self::SvPattern => "svpattern",
                Self::SvPrefetchOp => "svprfop",
            }
        )
    }
}

impl TypeKind {
    /// Gets the type part of a c typedef for a type that's in the form of {type}{size}_t.
    pub fn c_prefix(&self) -> &str {
        match self {
            Self::Bool => "bool",
            Self::Float => "float",
            Self::Int => "int",
            Self::UInt => "uint",
            Self::Poly => "poly",
            _ => unreachable!("Not used: {:#?}", self),
        }
    }

    /// Gets the rust prefix for the type kind i.e. i, u, f.
    pub fn rust_prefix(&self) -> &str {
        match self {
            Self::Float => "f",
            Self::Int => "i",
            Self::UInt => "u",
            Self::Poly => "u",
            _ => unreachable!("Unused type kind: {:#?}", self),
        }
    }

    pub fn is_enum(&self) -> bool {
        self == &TypeKind::SvPattern || self == &TypeKind::SvPrefetchOp
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum IntrinsicType {
    Ptr {
        constant: bool,
        child: Box<IntrinsicType>,
    },
    Type {
        constant: bool,
        kind: TypeKind,
        /// The bit length of this type (e.g. 32 for u32).
        /// For predicates, this means the length of the element each predicate bit represents
        bit_len: Option<u32>,

        /// Length of the vector (i.e. Fixed(4) for uint32x4_t), A value of `None`
        /// means this is not a simd type. A value of `None` can be assumed to
        /// be Fixed(1), although in some places a distinction is needed between `u64` and
        /// `uint64x1_t` this signals that.
        simd_len: Option<VecLen>,

        /// The number of rows for SIMD matrices (i.e. 2 for uint8x8x2_t).
        /// A value of `None` represents a type that does not contain any
        /// rows encoded in the type (e.g. uint8x8_t).
        /// A value of `None` can be assumed to be 1 though.
        vec_len: Option<u32>,
    },
}

impl IntrinsicType {
    /// Get the TypeKind for this type, recursing into pointers.
    pub fn kind(&self) -> TypeKind {
        match *self {
            IntrinsicType::Ptr { ref child, .. } => child.kind(),
            IntrinsicType::Type { kind, .. } => kind,
        }
    }

    /// Get the size of a single element inside this type, recursing into
    /// pointers, i.e. a pointer to a u16 would be 16 rather than the size
    /// of a pointer.
    pub fn inner_size(&self) -> u32 {
        match self {
            IntrinsicType::Ptr { child, .. } => child.inner_size(),
            IntrinsicType::Type {
                bit_len: Some(bl), ..
            } => *bl,
            _ => unreachable!("{self:?}"),
        }
    }

    pub fn set_inner_size(&mut self, size: u32) {
        match self {
            IntrinsicType::Ptr { child, .. } => child.set_inner_size(size),
            IntrinsicType::Type { bit_len, .. } => *bit_len = Some(size),
        }
    }

    pub fn num_lanes(&self) -> Option<VecLen> {
        match *self {
            IntrinsicType::Ptr { ref child, .. } => child.num_lanes(),
            IntrinsicType::Type { simd_len, .. } => simd_len,
        }
    }

    pub fn num_vectors(&self) -> u32 {
        match *self {
            IntrinsicType::Ptr { ref child, .. } => child.num_vectors(),
            IntrinsicType::Type {
                vec_len: Some(vl), ..
            } => vl,
            _ => 1,
        }
    }

    /// Determine if the type is a simd type, this will treat a type such as
    /// `uint64x1` as simd.
    pub fn is_simd(&self) -> bool {
        match *self {
            IntrinsicType::Ptr { ref child, .. } => child.is_simd(),
            IntrinsicType::Type {
                simd_len: None,
                vec_len: None,
                ..
            } => false,
            _ => true,
        }
    }

    pub fn is_scalable(&self) -> bool {
        match *self {
            IntrinsicType::Ptr { ref child, .. } => child.is_scalable(),
            IntrinsicType::Type {
                simd_len: Some(VecLen::Scalable),
                ..
            } => true,
            _ => false,
        }
    }

    pub fn is_ptr(&self) -> bool {
        match *self {
            IntrinsicType::Ptr { .. } => true,
            IntrinsicType::Type { .. } => false,
        }
    }

    pub fn is_predicate(&self) -> bool {
        matches!(
            *self,
            IntrinsicType::Type {
                kind: TypeKind::Bool,
                simd_len: Some(_),
                ..
            }
        )
    }

    pub fn is_p64(&self) -> bool {
        match *self {
            IntrinsicType::Ptr { ref child, .. } => child.is_p64(),
            IntrinsicType::Type {
                kind: TypeKind::Poly,
                bit_len: Some(64),
                ..
            } => true,
            _ => false,
        }
    }

    pub fn c_scalar_type(&self) -> String {
        if self.kind() == TypeKind::Bool {
            "bool".to_string()
        } else {
            format!(
                "{prefix}{bits}_t",
                prefix = self.kind().c_prefix(),
                bits = self.inner_size()
            )
        }
    }

    pub fn rust_scalar_type(&self) -> String {
        if self.kind() == TypeKind::Bool {
            "bool".to_string()
        } else {
            format!(
                "{prefix}{bits}",
                prefix = self.kind().rust_prefix(),
                bits = self.inner_size()
            )
        }
    }

    /// Gets a string containing the typename for this type in C format.
    pub fn c_type(&self) -> String {
        match self {
            IntrinsicType::Ptr { child, .. } => format!("{}*", child.c_type()),
            IntrinsicType::Type {
                constant,
                kind,
                bit_len: Some(bit_len),
                simd_len: None,
                vec_len: None,
                ..
            } => {
                if kind.is_enum() {
                    format!("const {kind}")
                } else if *kind == TypeKind::Bool {
                    kind.c_prefix().to_string()
                } else {
                    format!(
                        "{}{}{}_t",
                        if *constant { "const " } else { "" },
                        kind.c_prefix(),
                        bit_len
                    )
                }
            }
            IntrinsicType::Type {
                kind,
                bit_len: Some(bit_len),
                simd_len: Some(VecLen::Fixed(simd_len)),
                vec_len: Some(1) | None,
                ..
            } => format!("{}{bit_len}x{simd_len}_t", kind.c_prefix()),
            IntrinsicType::Type {
                kind,
                bit_len: Some(bit_len),
                simd_len: Some(VecLen::Fixed(simd_len)),
                vec_len: Some(vec_len),
                ..
            } => format!("{}{bit_len}x{simd_len}x{vec_len}_t", kind.c_prefix()),
            IntrinsicType::Type {
                kind,
                bit_len: Some(bit_len),
                simd_len: Some(VecLen::Scalable),
                vec_len,
                ..
            } => format!(
                "sv{}{bit_len}{}_t",
                kind.c_prefix(),
                match vec_len {
                    Some(len) if *len > 1 => format!("x{len}"),
                    _ => "".to_string(),
                }
            ),
            _ => unreachable!("{self:#?}"),
        }
    }

    pub fn rust_type(&self) -> String {
        match self {
            IntrinsicType::Ptr { child, .. } => format!("{}*", child.rust_type()),
            IntrinsicType::Type {
                constant,
                kind,
                bit_len: Some(bit_len),
                simd_len: None,
                vec_len: None,
                ..
            } => {
                if kind.is_enum() {
                    kind.to_string()
                } else if *constant {
                    // We make all const generic parameters i32s - this will cause issues with
                    // pointers to const data but the tool doesn't test those intrinsics
                    "i32".to_string()
                } else if *kind == TypeKind::Bool {
                    "bool".to_string()
                } else {
                    format!("{}{bit_len}", kind.rust_prefix())
                }
            }
            IntrinsicType::Type {
                kind,
                bit_len: Some(bit_len),
                simd_len: Some(VecLen::Fixed(simd_len)),
                vec_len: Some(1) | None,
                ..
            } => format!("{}{bit_len}x{simd_len}_t", kind.c_prefix()),
            IntrinsicType::Type {
                kind,
                bit_len: Some(bit_len),
                simd_len: Some(VecLen::Fixed(simd_len)),
                vec_len: Some(vec_len),
                ..
            } => format!("{}{bit_len}x{simd_len}x{vec_len}_t", kind.c_prefix()),
            IntrinsicType::Type {
                kind,
                bit_len: Some(bit_len),
                simd_len: Some(VecLen::Scalable),
                vec_len,
                ..
            } => format!(
                "sv{}{}{}_t",
                kind.c_prefix(),
                bit_len,
                match vec_len {
                    Some(len) if *len > 1 => format!("x{len}"),
                    _ => "".to_string(),
                }
            ),
            _ => unreachable!("{self:#?}"),
        }
    }

    /// Gets a cast for this type if needs promotion.
    /// This is required for 8 bit types due to printing as the 8 bit types use
    /// a char and when using that in `std::cout` it will print as a character,
    /// which means value of 0 will be printed as a null byte.
    ///
    /// This is also needed for polynomial types because we want them to be
    /// printed as unsigned integers to match Rust's `Debug` impl.
    pub fn c_promotion(&self) -> &str {
        match *self {
            IntrinsicType::Type {
                kind,
                bit_len: Some(8),
                ..
            } => match kind {
                TypeKind::Int | TypeKind::Bool => "(int)",
                TypeKind::UInt => "(unsigned int)",
                TypeKind::Poly => "(unsigned int)(uint8_t)",
                _ => "",
            },
            IntrinsicType::Type {
                kind: TypeKind::Poly,
                bit_len: Some(bit_len),
                ..
            } => match bit_len {
                8 => unreachable!("handled above"),
                16 => "(uint16_t)",
                32 => "(uint32_t)",
                64 => "(uint64_t)",
                128 => "",
                _ => panic!("invalid bit_len"),
            },
            _ => "",
        }
    }

    /// Generates an initialiser for an array, which can be used to initialise an argument for the
    /// intrinsic call.
    ///
    /// This is determistic based on the pass number.
    ///
    /// * `loads`: The number of values that need to be loaded from the argument array
    /// * e.g for argument type uint32x2, loads=2 results in a string representing 4 32-bit values
    ///
    /// Returns a string such as
    /// * `{0x1, 0x7F, 0xFF}` if `language` is `Language::C`
    /// * `[0x1 as _, 0x7F as _, 0xFF as _]` if `language` is `Language::Rust`
    pub fn populate_random(
        &self,
        indentation: Indentation,
        loads: u32,
        language: &Language,
    ) -> String {
        match self {
            IntrinsicType::Ptr { .. } => {
                let (prefix, suffix) = match language {
                    Language::Rust => ("[", "]"),
                    Language::C => ("{", "}"),
                };
                format!(
                    "{indentation}{prefix}{body}{suffix}",
                    body = (0..sliding_window_value_count(64, None, 1, loads))
                        .map(|i| {
                            format!(
                                "{}{}",
                                value_for_array(64, i),
                                match *language {
                                    Language::Rust => " as usize",
                                    Language::C => "",
                                }
                            )
                        })
                        .join(",")
                )
            }
            IntrinsicType::Type {
                bit_len: Some(bit_len @ (8 | 16 | 32 | 64)),
                kind: kind @ (TypeKind::Int | TypeKind::UInt | TypeKind::Poly),
                simd_len,
                vec_len,
                ..
            } => {
                let (prefix, suffix) = match language {
                    Language::Rust => ("[", "]"),
                    Language::C => ("{", "}"),
                };

                format!(
                    "{prefix}{body}{indentation}{suffix}",
                    body = (0..sliding_window_value_count(
                        *bit_len,
                        *simd_len,
                        vec_len.unwrap_or(1),
                        loads
                    ))
                        .format_with(", ", |i, fmt| {
                            let src = value_for_array(*bit_len, i);
                            assert!(src == 0 || src.ilog2() < *bit_len);
                            if *kind == TypeKind::Int && (src >> (*bit_len - 1)) != 0 {
                                // `src` is a two's complement representation of a negative value.
                                let mask = !0u64 >> (64 - *bit_len);
                                let ones_compl = src ^ mask;
                                let twos_compl = ones_compl + 1;
                                if (twos_compl == src) && (language == &Language::C) {
                                    // `src` is INT*_MIN. C requires `-0x7fffffff - 1` to avoid
                                    // undefined literal overflow behaviour.
                                    fmt(&format_args!("-{ones_compl:#x} - 1"))
                                } else {
                                    fmt(&format_args!("-{twos_compl:#x}"))
                                }
                            } else {
                                fmt(&format_args!("{src:#x}"))
                            }
                        })
                )
            }
            IntrinsicType::Type {
                kind: TypeKind::Float,
                bit_len: Some(bit_len @ (32 | 64)),
                simd_len,
                vec_len,
                ..
            } => {
                let (prefix, cast_prefix, cast_suffix, suffix) = match (language, bit_len) {
                    (&Language::Rust, 32) => ("[", "f32::from_bits(", ")", "]"),
                    (&Language::Rust, 64) => ("[", "f64::from_bits(", ")", "]"),
                    (&Language::C, 32) => ("{", "cast<float, uint32_t>(", ")", "}"),
                    (&Language::C, 64) => ("{", "cast<double, uint64_t>(", ")", "}"),
                    _ => unreachable!(),
                };
                format!(
                    "{prefix}{body}{indentation}{suffix}",
                    body = (0..sliding_window_value_count(
                        *bit_len,
                        *simd_len,
                        vec_len.unwrap_or(1),
                        loads
                    ))
                        .format_with(", ", |i, fmt| fmt(&format_args!(
                            "{indentation}{cast_prefix}{src:#x}{cast_suffix}",
                            src = value_for_array(*bit_len, i)
                        )))
                )
            }
            _ => unimplemented!("populate random: {:#?}", self),
        }
    }

    /// Determines the load function for this type.
    pub fn get_load_function(&self, is_aarch32: bool) -> String {
        match self {
            IntrinsicType::Ptr { child, .. } => child.get_load_function(is_aarch32),
            IntrinsicType::Type {
                kind: k,
                bit_len: Some(bl),
                simd_len: Some(VecLen::Fixed(sl)),
                vec_len,
                ..
            } => {
                let quad = if (sl * bl) > 64 { "q" } else { "" };

                format!(
                    "vld{len}{quad}_{type}{size}",
                    type = match k {
                        TypeKind::UInt => "u",
                        TypeKind::Int => "s",
                        TypeKind::Float => "f",
                        // The ACLE doesn't support 64-bit polynomial loads on Armv7
                        TypeKind::Poly => if is_aarch32 && *bl == 64 {"s"} else {"p"},
                        x => unreachable!("get_load_function: {x:#?}"),
                    },
                    size = bl,
                    quad = quad,
                    len = vec_len.unwrap_or(1),
                )
            }
            _ => unreachable!("get_load_function {self:#?}"),
        }
    }

    pub fn get_load_function_sve(&self) -> String {
        match self {
            IntrinsicType::Ptr { child, .. } => child.get_load_function_sve(),
            IntrinsicType::Type {
                kind: k,
                bit_len: Some(bl),
                simd_len: Some(VecLen::Scalable),
                vec_len,
                ..
            } => {
                format!(
                    "svld{len}_{type}{size}",
                    type = match k {
                        TypeKind::UInt => "u",
                        TypeKind::Int => "s",
                        TypeKind::Float => "f",
                        x => unreachable!("get_load_function {x:#?}"),
                    },
                    size = bl,
                    len = vec_len.unwrap_or(1),
                )
            }
            _ => unreachable!("get_load_function_sve {self:#?}"),
        }
    }

    /// Determines the store function for this type.
    pub fn get_store_function(&self, is_aarch32: bool) -> String {
        match self {
            IntrinsicType::Ptr { child, .. } => child.get_store_function(is_aarch32),
            IntrinsicType::Type {
                kind: k,
                bit_len: Some(bl),
                simd_len: Some(sl),
                vec_len,
                ..
            } => {
                let quad = match sl {
                    VecLen::Fixed(len) if len * bl > 64 => "q",
                    _ => "",
                };

                format!(
                    "{sve}vst{len}{quad}_{ty}{size}",
                    ty = match k {
                        TypeKind::UInt => "u",
                        // Predicates are converted to ints
                        TypeKind::Int | TypeKind::Bool => "s",
                        TypeKind::Float => "f",
                        TypeKind::Poly =>
                            if is_aarch32 && *bl == 64 {
                                "s"
                            } else {
                                "p"
                            },
                        x => unreachable!("get_store_function {x:#?}"),
                    },
                    sve = if self.is_scalable() { "s" } else { "" },
                    size = bl,
                    quad = quad,
                    len = vec_len.unwrap_or(1),
                )
            }
            _ => unreachable!("get_store_function IntrinsicType: {self:#?}"),
        }
    }

    pub fn from_c(s: &str) -> Result<IntrinsicType, String> {
        const CONST_STR: &str = "const ";
        const ENUM_STR: &str = "enum ";

        if let Some(s) = s.strip_suffix('*') {
            let (s, constant) = if s.ends_with(CONST_STR) || s.starts_with(CONST_STR) {
                (
                    s.trim_start_matches(CONST_STR).trim_end_matches(CONST_STR),
                    true,
                )
            } else {
                (s, false)
            };

            let s = s.trim_end();

            Ok(IntrinsicType::Ptr {
                constant,
                child: Box::new(IntrinsicType::from_c(s)?),
            })
        } else {
            // [const ][sv]TYPE[{bitlen}[x{simdlen}[x{vec_len}]]]_t
            //   | [enum ]TYPE
            let (mut s, constant) = match (s.strip_prefix(CONST_STR), s.strip_prefix(ENUM_STR)) {
                (Some(const_strip), _) => (const_strip, true),
                (_, Some(enum_strip)) => (enum_strip, true),
                (None, None) => (s, false),
            };
            s = s.strip_suffix("_t").unwrap_or(s);
            let sve = s.starts_with("sv");

            let mut parts = s.split('x'); // [[{bitlen}], [{simdlen}], [{vec_len}]]

            let start = parts.next().ok_or("Impossible to parse type")?;

            if let Some(digit_start) = start.find(|c: char| c.is_ascii_digit()) {
                let (arg_kind, bit_len) = start.split_at(digit_start);

                let arg_kind = arg_kind.parse::<TypeKind>()?;
                let bit_len = bit_len.parse::<u32>().map_err(|err| err.to_string())?;
                let n1 = match parts.next() {
                    Some(part) => Some(
                        part.parse::<u32>()
                            .map_err(|_| "Couldn't parse simd_len: {part}")?,
                    ),
                    None => None,
                };
                let n2 = match parts.next() {
                    Some(part) => Some(
                        part.parse::<u32>()
                            .map_err(|_| "Couldn't parse vec_len: {part}")?,
                    ),
                    None => None,
                };

                Ok(IntrinsicType::Type {
                    constant,
                    kind: arg_kind,
                    bit_len: Some(bit_len),
                    simd_len: if sve {
                        Some(VecLen::Scalable)
                    } else {
                        n1.map(VecLen::Fixed)
                    },
                    vec_len: if sve { n1 } else { n2 },
                })
            } else {
                let kind = start.parse::<TypeKind>()?;
                let bit_len = match kind {
                    TypeKind::SvPattern | TypeKind::SvPrefetchOp | TypeKind::Int => {
                        // All these are represented as i32
                        Some(32)
                    }
                    TypeKind::Bool => Some(8),
                    _ => None,
                };
                Ok(IntrinsicType::Type {
                    constant,
                    kind,
                    bit_len,
                    simd_len: if sve && !kind.is_enum() {
                        Some(VecLen::Scalable)
                    } else {
                        None
                    },
                    vec_len: None,
                })
            }
        }
    }
}

// Returns the number of values needed to load `num_vectors` vectors of length `simd_len` with
// values, `loads` times.
fn sliding_window_value_count(
    bit_len: u32,
    simd_len: Option<VecLen>,
    num_vectors: u32,
    loads: u32,
) -> u32 {
    // If it's SVE, assume the vector has the largest possible length given the data type.
    let vector_length = simd_len.map_or(1, |v| {
        if let VecLen::Fixed(n) = v {
            n
        } else {
            MAX_SVE_BITS / bit_len
        }
    });
    vector_length * num_vectors + loads - 1
}
