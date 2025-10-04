use crate::utils::leaf_nodes_from_grammar_name;
use std::ops::Add;
use tree_sitter::Node;

#[derive(Debug)]
pub struct RustIntrinsic<'a> {
    pub intrinsic: &'a str,
    pub arg_names: Vec<&'a str>,
    pub arg_types: Vec<&'a str>,
    pub generic_arg_names: Vec<&'a str>,
    pub generic_arg_types: Vec<&'a str>,
    pub return_type: Option<&'a str>,
}

#[derive(Debug)]
pub struct CIntrinsic<'a> {
    pub intrinsic: &'a str,
    pub arg_names: Vec<&'a str>,
    pub arg_types: Vec<&'a str>,
    pub specifier: &'a str,
    pub return_type: Option<&'a str>,
}

impl<'a> CIntrinsic<'a> {
    pub fn new(node: Node, source: &'a str) -> Self {
        // Take an intrinsic definition for example:
        //
        // static __inline__ v128_t __DEFAULT_FN_ATTRS wasm_u32x4_make(uint32_t __c0, uint32_t __c1, uint32_t __c2, uint32_t __c3) {...}
        //
        // For a C intrinsic, the immediate children
        // would have their grammar names as:
        //
        // "storage_class_specifier" (which is `static`)
        // "storage_class_specifier" (which is `__inline__`)
        // "identifier" (which is `v128_t`. The parser doesn't recognize that it is a type, instead thinks that it is an identifier)
        // "ERROR" (which points to the keyword `__DEFAULT_FN_ATTRS`. The parser doesn't recognize it as a valid part of the tree, and annotates it as ERROR.)
        // "function_declarator" (points to `wasm_u32x4_make(uint32_t __c0, uint32_t __c1, uint32_t __c2, uint32_t __c3)`)
        // "compound_statement" (the body of the function)
        let mut cursor = node.walk();

        let return_type = node
            .children(&mut cursor)
            .find(|node| {
                node.grammar_name() == "identifier" || node.grammar_name() == "primitive_type"
            })
            .map(|node| source.get(node.byte_range()).unwrap());

        let specifier = source
            .get(
                node.children(&mut cursor)
                    .find(|node| node.grammar_name() == "ERROR")
                    .unwrap()
                    .byte_range(),
            )
            .unwrap();

        // The immediate children of the `function_declarator` node would have
        // their grammar as follows:
        //
        // "identifier" (which is the intrinsic name)
        // "parameter_list" (which is the arguments to the intrinsic)
        let declarator_node = node
            .children(&mut cursor)
            .find(|node| node.grammar_name() == "function_declarator")
            .unwrap();

        // The immediate children of a `parameter_list` node would have
        // their grammar as follows (assuming 2 arguments):
        //
        // "(" -> The opening bracket that denotes the start of the arguments definition
        // "parameter_declaration" -> The definition for the first argument
        // "," -> The comma that separates the first and the second arguments
        // "parameter_declaration" -> The definition for the first argument
        // ")"  -> The closing bracket that denotes the start of the arguments definition
        //
        // Each node with grammar name as `parameter_declaration` could have their children as
        // (incase of `int x`):
        // 1. "primitive_type" -> Points to `int`
        // 2. "indentifier" -> Points to `x`
        //
        // or have (incase of `v128_t x`):
        // 1. "identifier" -> Points to `v128_t` which is actually a type (but the parser is unaware of it)
        // 2. "identifier" -> Points to `x`
        //
        // or have (incase of `const void *__mem`):
        // 1. "type_qualifier" -> Points to `const`
        // 2. "primitive_type" -> Points to `void`
        // 3. "pointer_declarator" -> breaks down into "*" and "identifier" (which is `__mem`)
        //
        let intrinsic_name = source
            .get(
                declarator_node
                    .children(&mut cursor)
                    .find(|node| node.grammar_name() == "identifier")
                    .unwrap()
                    .byte_range(),
            )
            .unwrap();

        let args = declarator_node
            .children(&mut cursor)
            .find(|node| node.grammar_name() == "parameter_list");

        let (arg_names, arg_types): (Vec<&str>, Vec<&str>) = if let Some(args) = args {
            let arg_name_nodes = leaf_nodes_from_grammar_name(args, "identifier");
            let arg_name_nodes = arg_name_nodes.iter();

            arg_name_nodes
                .map(|arg_name_node| {
                    // Since the type could be identified as either `primitive_type, `indentifier`,
                    // or a combination of `type_qualifier`, `primitive_type` and `*` (in the case of "const void *")
                    // this approach first calculates the end index (which is right before the start of an argument variable)
                    //
                    // And then searches backwards until it finds a break (either a comma
                    // or the opening bracket). The entire portion contained within this range
                    // is then considered as the type of the argument.
                    let end_index = arg_name_node.byte_range().start;
                    let start_index = source
                        .get(0..end_index)
                        .unwrap()
                        .bytes()
                        .rposition(|character| character == b',' || character == b'(')
                        .unwrap()
                        .add(1);
                    (
                        source
                            .get(arg_name_node.byte_range())
                            .expect("C arg name construction")
                            .trim(),
                        source
                            .get(start_index..end_index)
                            .expect("C arg type construction")
                            .trim(),
                    )
                })
                .filter(|(_, arg_type)| arg_type.len() > 0)
                .filter(|(_, arg_type)| !arg_type.starts_with("\""))
                .unzip()
        } else {
            (Vec::new(), Vec::new())
        };

        Self {
            intrinsic: intrinsic_name,
            arg_names,
            arg_types,
            return_type,
            specifier: specifier,
        }
    }
}

impl<'a> RustIntrinsic<'a> {
    pub fn new(node: Node, source: &'a str) -> Self {
        // For a Rust intrinsic, the immediate children
        // would have their grammar names as:
        //
        // 1. "visibility_modifier"  (for `pub`)
        // 2. "function_modifiers" (for `unsafe`. May not always be present)
        // 3. "fn" (the actual keyword `fn`)
        // 4. "identifier" (the name of the function)
        // 5. "type_parameters" (the const generic arguments. This is not always present)
        // 6. "parameters" (the arguments passed to the function)
        // 7. "->" (the arrow used to specify return type)
        // 8. "identifier" (the return type of the function)
        // 9. "block" (the body of the function)
        //
        let mut cursor = node.walk();
        let intrinsic_name = source
            .get(
                node.children(&mut cursor)
                    .find(|node| node.grammar_name() == "identifier")
                    .unwrap()
                    .byte_range(),
            )
            .unwrap();

        let arrow_pos = node
            .children(&mut cursor)
            .position(|node| node.grammar_name() == "->");

        let return_type = arrow_pos.map(|index| {
            source
                .get(node.child(index + 1).unwrap().byte_range())
                .unwrap()
        });

        let generic_args = node
            .children(&mut cursor)
            .find(|node| node.grammar_name() == "type_parameters");

        let args = node
            .children(&mut cursor)
            .find(|node| node.grammar_name() == "parameters");

        let mut generic_arg_names: Vec<&str> = Vec::new();
        let mut generic_arg_types: Vec<&str> = Vec::new();
        if let Some(generic_args) = generic_args {
            // The children of this node have their grammar_names as the following
            // (assuming 2 generic arguments):
            //
            // "<" (The opening angle bracket that starts the generic arguments definition)
            // "const_parameter" (The first const generic argument)
            // "," (The comma that denotes the end of definition of the first const generic argument)
            // "const_parameter" (The second const generic argument)
            // ">" (The closing angle bracket that concludes the generic arguments definition)
            //
            (generic_arg_names, generic_arg_types) = generic_args
                .children(&mut cursor)
                .filter(|arg| arg.grammar_name() == "const_parameter")
                .map(|arg| {
                    (
                        source
                            .get(arg.named_child(0).unwrap().byte_range())
                            .unwrap(),
                        source
                            .get(arg.named_child(1).unwrap().byte_range())
                            .unwrap(),
                    )
                })
                .unzip();
        }

        let mut arg_names: Vec<&str> = Vec::new();
        let mut arg_types: Vec<&str> = Vec::new();
        if let Some(args) = args {
            // The children of this node have their grammar_names as the following
            // (assuming 2 generic arguments):
            //
            // "(" (The opening circular bracket that starts the arguments definition)
            // "parameter" (The first argument)
            // "," (The comma that denotes the end of definition of the first argument)
            // "parameter" (The second argument)
            // ")" (The closing circular bracket that concludes the arguments definition)
            (arg_names, arg_types) = args
                .children(&mut cursor)
                .filter(|arg| arg.grammar_name() == "parameter")
                .map(|arg| {
                    (
                        source
                            .get(arg.named_child(0).unwrap().byte_range())
                            .unwrap(),
                        source
                            .get(arg.named_child(1).unwrap().byte_range())
                            .unwrap(),
                    )
                })
                .unzip();
        }

        Self {
            intrinsic: intrinsic_name,
            arg_names,
            arg_types,
            generic_arg_names,
            generic_arg_types,
            return_type,
        }
    }
}
