use crate::{
    format::Indentation,
    types::{IntrinsicType, TypeKind, VecLen},
    values::MAX_SVE_BITS,
    Extension, Language,
};

use super::argument::ArgumentList;

#[derive(Debug, PartialEq, Copy, Clone)]
pub enum Predication {
    None,
    Merging,
    Zeroing,
    DontCare,
}

/// An intrinsic
#[derive(Debug, PartialEq, Clone)]
pub struct Intrinsic {
    /// The function name of this intrinsic.
    pub name: String,

    /// Any arguments for this intrinsic.
    pub arguments: ArgumentList,

    /// The return type of this intrinsic.
    pub results: IntrinsicType,

    /// Whether this intrinsic is only available on A64.
    pub a64_only: bool,

    /// The type of predication (if any) this intrinsic uses.
    pub predication: Predication,
}

impl Intrinsic {
    pub fn print_results_c(
        &self,
        indentation: Indentation,
        var_name: &str,
        context: &str,
    ) -> String {
        let open = format!(
            r#"std::cout << std::boolalpha << "Result{context} "{iter} << ": {ty}" << std::fixed << std::setprecision(150);"#,
            iter = if self.arguments.iter().any(|a| !a.uses_set_values()) {
                " << i+1"
            } else {
                ""
            },
            ty = if self.results.is_simd() {
                format!("{}(", self.results.c_type())
            } else {
                "".to_string()
            }
        );

        let close = format!(
            r#"std::cout << "{brace}" << std::endl;"#,
            brace = if self.results.is_simd() { ")" } else { "" },
        );
        let indentation_1 = indentation.nested();
        let indentation_2 = indentation_1.nested();
        format!(
            r#"{indentation}{open}
{indentation}for (int j=0; j<element_count; j++) {{
{indentation_1}std::cout << {cast}{var_name}[j];
{indentation_1}if (j < element_count-1) {{
{indentation_2}std::cout << ", ";
{indentation_1}}}
{indentation}}}
{indentation}{close}"#,
            open = open,
            close = close,
            var_name = var_name,
            cast = self.results.c_promotion()
        )
    }

    pub fn print_results_rust(&self, indentation: Indentation, context: &str) -> String {
        let open = format!(
            r#"print!("Result{context} {{}}: {ty}", {iter});"#,
            iter = if self.arguments.iter().any(|a| !a.uses_set_values()) {
                "i+1"
            } else {
                "\"\""
            },
            ty = if self.results.is_simd() {
                format!("{}(", self.results.rust_type())
            } else {
                "".to_string()
            },
        );

        let close = format!(
            r#"println!("{brace}")"#,
            brace = if self.results.is_simd() { ")" } else { "" },
        );

        let bool_cast = if self.results.kind() == TypeKind::Bool && !self.results.is_predicate() {
            // Match C's bool printing behaviour
            " as i32"
        } else {
            ""
        };

        let indentation_1 = indentation.nested();
        let indentation_2 = indentation_1.nested();
        format!(
            r#"{indentation}{open}
{indentation}for j in 0..element_count {{
{indentation_1}print!("{{:.150?}}", results_array[j as usize]{bool_cast});
{indentation_1}if j < element_count-1 {{
{indentation_2}print!(", ");
{indentation_1}}}
{indentation}}}
{indentation}{close}"#,
        )
    }

    pub fn gen_results_array_c(&self, indentation: Indentation) -> String {
        let ty = if self.results.is_predicate() {
            // We'll convert predicates to ints later
            format!("int{}_t", self.results.inner_size())
        } else {
            self.results.c_scalar_type()
        };

        format!(
            "{indentation}{ty} results_array[{size}] = {{0}};",
            size = if self.results.is_simd() {
                match self.results.num_lanes().unwrap() {
                    // If an SVE vector is returned, assume the largest possible vector size
                    VecLen::Scalable => {
                        (MAX_SVE_BITS / self.results.inner_size()) * self.results.num_vectors()
                    }
                    VecLen::Fixed(n) => n * self.results.num_vectors(),
                }
            } else {
                1
            }
        )
    }

    pub fn gen_results_array_rust(&self, indentation: Indentation) -> String {
        let ty = if self.results.is_predicate() {
            // Predicates are converted to ints
            format!("i{}", self.results.inner_size())
        } else {
            self.results.rust_scalar_type()
        };
        format!(
            "{indentation}let mut results_array: [{ty}; {size}] = [Default::default(); {size}];",
            size = if self.results.is_simd() {
                match self.results.num_lanes().unwrap() {
                    // If an SVE vector is returned, assume the largest possible vector size
                    VecLen::Scalable => {
                        (MAX_SVE_BITS / self.results.inner_size()) * self.results.num_vectors()
                    }
                    VecLen::Fixed(n) => n * self.results.num_vectors(),
                }
            } else {
                1
            }
        )
    }

    /// Returns a line which stores the result of this intrinsic to array `results_array`, in the
    /// provided language.
    /// e.g `svst1_u32(svtrue_b32(), results_array.as_ptr().offset(i), __return_value);`, for intrinsic
    /// svadd_s32_z and language Rust.
    fn store_result(
        &self,
        indentation: Indentation,
        language: Language,
        is_aarch32: bool,
    ) -> String {
        let results = &self.results;
        if results.is_simd() {
            let arg_result = if results.is_predicate() {
                format!("svdup_n_s{}_z(__return_value, 1)", results.inner_size())
            } else if results.is_p64() && is_aarch32 && language == Language::C {
                "cast<int64x1_t>(__return_value)".to_string()
            } else {
                "__return_value".to_string()
            };

            format!(
                "{indentation}{store}({predicate}{arg_array}, {arg_result});",
                store = results.get_store_function(language == Language::C && is_aarch32),
                predicate = if self.predication == Predication::DontCare {
                    "pg, ".to_string()
                } else if results.is_scalable() {
                    format!("svptrue_b{}(), ", results.inner_size())
                } else {
                    "".to_string()
                },
                arg_array = if language == Language::C {
                    "results_array"
                } else {
                    "results_array.as_mut_ptr()"
                },
            )
        } else {
            format!("{indentation}results_array[0] = __return_value;")
        }
    }

    /// Returns a line which stores the number of elements in one intrinsic result in a variable
    /// named `element_count`. For Neon this will be a fixed number, for SVE this will be either
    /// a fixed number or a call to one of the `svcnt` intrinsics (possibly multiplied by some
    /// factor, if multiple vectors are returned).
    pub fn gen_element_count_c(&self, indentation: Indentation, language: Language) -> String {
        format!(
            "{indentation}{rtype} element_count = {call};",
            rtype = if let Language::Rust = language {
                "let"
            } else {
                "uint64_t"
            },
            call = match self.results {
                IntrinsicType::Type {
                    bit_len: Some(bit_len),
                    simd_len: Some(VecLen::Scalable),
                    vec_len,
                    ..
                } => format!(
                    "{ropen} {num_vectors} * svcnt{size}() {rclose}",
                    num_vectors = vec_len.unwrap_or(1),
                    size = match bit_len {
                        64 => "d",
                        32 => "w",
                        16 => "h",
                        8 => "b",
                        _ => unreachable!("non-SVE result bit-length"),
                    },
                    ropen = if let Language::Rust = language {
                        "unsafe {"
                    } else {
                        ""
                    },
                    rclose = if let Language::Rust = language {
                        "}"
                    } else {
                        ""
                    }
                ),
                IntrinsicType::Type {
                    simd_len: Some(VecLen::Fixed(sl)),
                    vec_len,
                    ..
                } => format!("{}", vec_len.unwrap_or(1) * sl),
                IntrinsicType::Type {
                    simd_len: None,
                    vec_len: None,
                    ..
                } => "1".to_string(),
                _ => unreachable!("Shouldn't be called on this type"),
            }
        )
    }

    /// Returns a call to this intrinsic in the given language, storing the result in a variable
    /// `varname`. For each pair in `overrides`, the any argument at position `pair.0`
    /// will be called with `pair.1` instead of its actual name.
    pub fn generate_call(
        &self,
        indentation: Indentation,
        varname: &str,
        language: Language,
    ) -> String {
        let constraints = self.arguments.as_constraint_parameters_rust();
        let constraints = if !constraints.is_empty() {
            format!("::<{constraints}>")
        } else {
            constraints
        };

        let (decl_var, constraints, args) = match language {
            Language::Rust => ("let", constraints, self.arguments.as_call_param_rust()),
            Language::C => ("auto", "".to_string(), self.arguments.as_call_param_c()),
        };

        format!(
            "{indentation}{bind} {varname} = {intrinsic_call}{const}({args});",
            bind = decl_var,
            intrinsic_call = self.name,
            const = constraints,
        )
    }

    pub fn generate_loop_c(
        &self,
        indentation: Indentation,
        additional: &str,
        passes: u32,
        mode: Extension,
        is_aarch32: bool,
    ) -> String {
        let block_indentation = indentation.nested();
        let start = if self.arguments.iter().any(|a| !a.uses_set_values()) {
            format!("{indentation}for (int i=0; i<{passes}; i++) {{")
        } else {
            format!("{indentation}{{")
        };

        format!(
            r#"{start}
{loaded_args}
{intrinsic_call}

{store_result}
{print_result}
{indentation}}}
"#,
            loaded_args = self
                .arguments
                .load_values_c(block_indentation, mode, is_aarch32),
            intrinsic_call = self.generate_call(block_indentation, "__return_value", Language::C),
            store_result = self.store_result(block_indentation, Language::C, is_aarch32),
            print_result = self.print_results_c(block_indentation, "results_array", additional)
        )
    }

    pub fn generate_loop_rust(
        &self,
        indentation: Indentation,
        additional: &str,
        passes: u32,
        mode: Extension,
        is_aarch32: bool,
    ) -> String {
        let block_indentation = if self.arguments.iter().any(|a| !a.uses_set_values()) {
            indentation.nested()
        } else {
            indentation
        };

        let mut block = format!(
            r#"{block_indentation}unsafe {{
{loaded_args}
{intrinsic_call}

{store_result}
{print_result}
{block_indentation}}}
"#,
            loaded_args = self
                .arguments
                .load_values_rust(block_indentation.nested(), mode),
            intrinsic_call =
                self.generate_call(block_indentation.nested(), "__return_value", Language::Rust),
            store_result =
                self.store_result(block_indentation.nested(), Language::Rust, is_aarch32),
            print_result = self.print_results_rust(block_indentation.nested(), additional),
        );

        if self.arguments.iter().any(|a| !a.uses_set_values()) {
            block = format!(
                r#"{indentation}for i in 0..{passes} {{
{block}
{indentation}}}"#,
            );
        }

        block
    }
}
