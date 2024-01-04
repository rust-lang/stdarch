use std::iter::Iterator;
use std::ops::Range;

use crate::format::Indentation;
use crate::json_parser::ArgPrep;
use crate::types::{IntrinsicType, TypeKind, VecLen};
use crate::values::{MAX_SVE_BITS, PRED_PATTERNS, SVE_GRANULE_BITS};
use crate::{Extension, Language};

use itertools::Itertools;

/// An argument for the intrinsic.
#[derive(Debug, PartialEq, Clone)]
pub struct Argument {
    /// The argument's index in the intrinsic function call.
    pub pos: usize,
    /// The argument name.
    pub name: String,
    /// The type of the argument.
    pub ty: IntrinsicType,
    /// Any constraints that are on this argument
    pub constraints: Vec<Constraint>,
}

#[derive(Debug, PartialEq, Clone)]
pub enum Constraint {
    Equal(i64),
    Range(Range<i64>),
    Svpattern,
    Svprfop,
    ImmRotation,
    ImmRotationAdd,
}

impl TryFrom<ArgPrep> for Constraint {
    type Error = ();

    fn try_from(prep: ArgPrep) -> Result<Self, Self::Error> {
        let parsed_ints = match prep {
            ArgPrep::Immediate { min, max } => Ok((min, max)),
            _ => Err(()),
        };
        if let Ok((min, max)) = parsed_ints {
            if min == max {
                Ok(Constraint::Equal(min))
            } else {
                Ok(Constraint::Range(min..max + 1))
            }
        } else {
            Err(())
        }
    }
}

impl Constraint {
    pub fn iter(&self) -> Box<dyn Iterator<Item = i64>> {
        match self {
            Constraint::Equal(eq) => Box::new(std::iter::once(*eq)),
            Constraint::Range(range) => Box::new(range.clone()),
            Constraint::Svpattern => Box::new((0..14).chain(29..32)),
            Constraint::Svprfop => Box::new((0..6).chain(8..14)),
            Constraint::ImmRotation => Box::new((0..271).step_by(90)),
            Constraint::ImmRotationAdd => Box::new((90..271).step_by(180)),
        }
    }
}

impl Argument {
    fn to_c_type(&self) -> String {
        self.ty.c_type()
    }

    fn is_simd(&self) -> bool {
        self.ty.is_simd()
    }

    pub fn is_ptr(&self) -> bool {
        self.ty.is_ptr()
    }

    pub fn is_predicate(&self) -> bool {
        self.ty.is_predicate()
    }

    // Values for predicates, bools and immediates aren't loaded from a "populate_random" array and
    // we instead use a new block for each preset value
    pub fn uses_set_values(&self) -> bool {
        self.has_constraint() || self.ty.kind() == TypeKind::Bool
    }

    pub fn has_constraint(&self) -> bool {
        !self.constraints.is_empty()
    }

    pub fn type_and_name_from_c(arg: &str) -> (&str, &str) {
        let split_index = arg
            .rfind([' ', '*'])
            .expect("Couldn't split type and argname");

        (arg[..split_index + 1].trim_end(), &arg[split_index + 1..])
    }

    pub fn from_c(pos: usize, arg: &str, arg_prep: Option<ArgPrep>) -> Argument {
        let (ty, var_name) = Self::type_and_name_from_c(arg);

        let mut ty = IntrinsicType::from_c(ty)
            .unwrap_or_else(|_| panic!("Failed to parse argument '{arg}'"));

        if ty.is_predicate() {
            if let Some(ap) = arg_prep.as_ref() {
                let bit_len = ap.get_element_size().unwrap_or_else(|e| panic!("{e}"));
                ty.set_inner_size(bit_len);
            } else {
                // Assume 8-bit lanes
                // For example, svptest_* allow any length of predicate
                ty.set_inner_size(8);
            }
        }

        let constraint = arg_prep.and_then(|a| a.try_into().ok()).or_else(|| {
            if ty.kind() == TypeKind::SvPattern {
                Some(Constraint::Svpattern)
            } else if ty.kind() == TypeKind::SvPrefetchOp {
                Some(Constraint::Svprfop)
            } else if var_name == "imm_rotation" {
                Some(Constraint::ImmRotation)
            } else {
                None
            }
        });

        Argument {
            pos,
            name: String::from(var_name),
            ty,
            constraints: constraint.map_or(vec![], |r| vec![r]),
        }
    }

    fn is_rust_vals_array_const(&self) -> bool {
        use TypeKind::*;
        match self.ty {
            // Floats have to be loaded at runtime for stable NaN conversion.
            IntrinsicType::Type { kind: Float, .. } => false,
            IntrinsicType::Type {
                kind: Int | UInt | Poly,
                ..
            } => true,
            IntrinsicType::Ptr { .. } => true,
            ref ty => unimplemented!("{:#?}", ty),
        }
    }

    /// The binding keyword (e.g. "const" or "let") for the array of possible test inputs.
    pub fn rust_vals_array_binding(&self) -> impl std::fmt::Display {
        if self.is_rust_vals_array_const() {
            "const"
        } else {
            "let"
        }
    }

    /// The name (e.g. "A_VALS" or "a_vals") for the array of possible test inputs.
    pub fn rust_vals_array_name(&self) -> impl std::fmt::Display {
        if self.is_rust_vals_array_const() {
            format!("{}_VALS", self.name.to_uppercase())
        } else {
            format!("{}_vals", self.name.to_lowercase())
        }
    }

    /// Returns a vector of predication setup statements for this argument
    pub fn get_predicate_decls(&self, indentation: Indentation, language: Language) -> Vec<String> {
        assert!(self.is_predicate());
        let psize = self.ty.inner_size();
        let (bind, open, close) = if let Language::Rust = language {
            ("let ", "unsafe {", "}")
        } else {
            ("", "", "")
        };

        PRED_PATTERNS
            .iter()
            .map(|pat| {
                let pat_string = pat
                    .iter()
                    .take((SVE_GRANULE_BITS / psize) as usize)
                    .map(|b| b.to_string())
                    .join(", ");

                format!(
                    "{indentation}{bind}{} = {open}svdupq_n_b{psize}({pat_string}){close};",
                    self.name
                )
            })
            .collect()
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct ArgumentList {
    pub args: Vec<Argument>,
}

impl ArgumentList {
    /// Converts the argument list into the call parameters for a C function call.
    /// e.g. this would generate something like `a, b, c`
    pub fn as_call_param_c(&self) -> String {
        self.args.iter().map(|arg| &arg.name).join(", ")
    }

    /// Converts the argument list into the call parameters for a Rust function.
    /// e.g. this would generate something like `a, b, c`
    pub fn as_call_param_rust(&self) -> String {
        self.args
            .iter()
            .filter(|a| !a.has_constraint())
            .map(|arg| arg.name.to_string())
            .join(", ")
    }

    pub fn as_constraint_parameters_rust(&self) -> String {
        self.args
            .iter()
            .filter(|a| a.has_constraint())
            .map(|arg| arg.name.clone())
            .join(", ")
    }

    /// Creates a line for each argument that initializes an array for C from which `loads` argument
    /// values can be loaded  as a sliding window.
    /// e.g `const int32x2_t a_vals = {0x3effffff, 0x3effffff, 0x3f7fffff}`, if loads=2.
    pub fn gen_arglists_c(&self, indentation: Indentation, loads: u32) -> String {
        self.iter()
            .filter(|arg| !arg.uses_set_values())
            .filter_map(|arg| {
                let ty = if arg.is_ptr() {
                    "uintptr_t".to_string()
                } else {
                    arg.ty.c_scalar_type()
                };

                (!arg.has_constraint()).then(|| {
                    format!(
                        "{indentation}const {ty} {name}_vals[] = {values};",
                        name = arg.name,
                        values = arg.ty.populate_random(indentation, loads, &Language::C)
                    )
                })
            })
            .join("\n")
    }

    /// Creates a line for each argument that initializes an array for Rust from which `loads` argument
    /// values can be loaded as a sliding window, e.g `const A_VALS: [u32; 20]  = [...];`
    pub fn gen_arglists_rust(&self, indentation: Indentation, loads: u32) -> String {
        self.iter()
            .filter(|arg| !arg.uses_set_values())
            .filter_map(|arg| {
                (!arg.has_constraint()).then(|| {
                    let vlen = arg.ty.num_lanes().map_or(1, |v| {
                        if let VecLen::Fixed(n) = v {
                            n
                        } else {
                            MAX_SVE_BITS / arg.ty.inner_size()
                        }
                    });
                    let load_size = vlen * arg.ty.num_vectors() + loads - 1;

                    let ty = if arg.is_ptr() {
                        "usize".to_string()
                    } else {
                        arg.ty.rust_scalar_type()
                    };
                    format!(
                        "{indentation}{bind} {name}: [{ty}; {load_size}] = {values};",
                        bind = arg.rust_vals_array_binding(),
                        name = arg.rust_vals_array_name(),
                        values = arg.ty.populate_random(indentation, loads, &Language::Rust)
                    )
                })
            })
            .join("\n")
    }

    /// Creates a line that initalizes this argument from a pointer p_[arg] using a
    /// load intrinsic, e.g. `uint8x8_t a = vld1_u8(p_a++);`
    pub fn load_values_c(
        &self,
        indentation: Indentation,
        mode: Extension,
        is_aarch32: bool,
    ) -> String {
        if let Extension::SVE = mode {
            self.iter()
                .filter_map(|arg| {
                    (!arg.uses_set_values()).then(|| {
                        if arg.is_simd() {
                            format!(
                                "{indentation}{ty} {name} = {load}(svptrue_b{psize}(), &{name}_vals[i]);",
                                psize = arg.ty.inner_size(),
                                ty = arg.to_c_type(),
                                name = arg.name,
                                load = arg.ty.get_load_function_sve()
                            )
                        } else {
                            format!(
                                "{indentation}{ty} {name} = {cast}{name}_vals[i];",
                                ty = arg.to_c_type(),
                                name = arg.name,
                                cast = if arg.is_ptr() {
                                    format!("({})", arg.to_c_type())
                                } else {
                                    String::new()
                                },
                            )
                        }
                    })
                })
                .join("\n")
        } else {
            self.iter()
                .filter_map(|arg| {
                    // The ACLE doesn't support 64-bit polynomial loads on Armv7
                    // This and the cast are a workaround for this
                    let armv7_p64 = if arg.ty.is_p64() { is_aarch32 } else { false };

                    let (open_cast, close_cast) = if armv7_p64 {
                        (format!("cast<{}>(", arg.to_c_type()), ")")
                    } else {
                        ("".to_string(), "")
                    };

                    (!arg.uses_set_values()).then(|| {
                        if arg.is_simd() {
                            format!(
                                "{indentation}{ty} {name} = {open_cast}{load}(&{name}_vals[i]){close_cast};",
                                ty = arg.to_c_type(),
                                name = arg.name,
                                load = arg.ty.get_load_function(is_aarch32),
                                open_cast = open_cast,
                                close_cast = close_cast
                            )
                        } else {
                            format!(
                                "{indentation}{ty} {name} = {open_cast} {name}_vals[i] {close_cast};",
                                ty = arg.to_c_type(),
                                name = arg.name,
                                open_cast = open_cast,
                                close_cast = close_cast
                            )
                        }
                    })
                })
                .join("\n")
        }
    }

    /// Creates a line for each argument that initializes the argument from array `[ARG]_VALS` at
    /// an offset `i` using a load intrinsic, in Rust.
    /// e.g `let a = vld1_u8(A_VALS.as_ptr().offset(i));`
    pub fn load_values_rust(&self, indentation: Indentation, mode: Extension) -> String {
        self.iter()
            .filter_map(|arg| {
                (!arg.uses_set_values()).then(|| {
                    if arg.is_simd() {
                        format!(
                            "{indentation}let {name} = {load}({predicate}{array_name}.as_ptr().offset(i));",
                            name = arg.name,
                            array_name = arg.rust_vals_array_name(),
                            load = if let Extension::SVE = mode {
                                arg.ty.get_load_function_sve()
                            } else {
                                arg.ty.get_load_function(false)
                            },
                            predicate = if let Extension::SVE = mode {
                                format!("svptrue_b{}(), ", arg.ty.inner_size())
                            } else {
                                "".to_string()
                            }
                        )
                    } else {
                        format!(
                            "{indentation}let {name} = {array_name}[i as usize]{cast};",
                            name = arg.name,
                            array_name = arg.rust_vals_array_name(),
                            cast = if arg.is_ptr() {
                                format!(" as *const {}", arg.ty.rust_scalar_type())
                            } else {
                                String::new()
                            },
                        )
                    }
                })
            })
            .join("\n")
    }

    pub fn iter(&self) -> std::slice::Iter<'_, Argument> {
        self.args.iter()
    }
}
