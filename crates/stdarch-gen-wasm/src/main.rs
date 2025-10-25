mod cli;
mod matcher;
mod structs;
mod utils;

use std::fs;

use clap::Parser as ClapParser;
use cli::Args;
use tree_sitter::{Parser, Tree};

use crate::matcher::match_intrinsic_definitions;
use crate::structs::{CIntrinsic, RustIntrinsic};
use crate::utils::leaf_nodes_from_grammar_name;

/// Read the Rust source code and returns its AST
fn process_rust_code(source: String) -> (String, Tree) {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_rust::LANGUAGE.into())
        .expect("Error loading Rust grammar");
    let source_code = fs::read_to_string(source);

    if let Err(ref err) = source_code {
        eprintln!("Rust parsing error: {}", err);
        panic!()
    }
    let source_code_string = source_code.unwrap();
    let tree = parser.parse(source_code_string.clone(), None).unwrap();
    (source_code_string, tree)
}

/// Reads the C source code and returns its AST
fn process_c_code(source: String) -> (String, Tree) {
    let mut parser = Parser::new();
    parser
        .set_language(&tree_sitter_c::LANGUAGE.into())
        .expect("Error loading Rust grammar");
    let source_code = fs::read_to_string(source);

    if let Err(ref err) = source_code {
        eprintln!("C parsing error: {}", err);
        panic!()
    }
    let source_code_string = source_code.unwrap();
    let tree = parser.parse(source_code_string.clone(), None).unwrap();
    (source_code_string, tree)
}

/// Creates an entry in the spec sheet that corresponds to a specific intrinsic
fn generate_spec(c_intrinsic: &CIntrinsic, rust_intrinsic: &RustIntrinsic) -> String {
    format!(
        "/// {}
c-intrinsic-name = {}
c-arguments = {}
c-arguments-data-types = {}
c-return-type = {}
rust-intrinsic-name = {}
rust-arguments = {}
rust-arguments-data-types = {}
rust-const-generic-arguments = {}
rust-const-generic-arguments-data-types = {}
rust-return-type = {}",
        rust_intrinsic.intrinsic,
        c_intrinsic.intrinsic,
        c_intrinsic.arg_names.join(", "),
        c_intrinsic.arg_types.join(", "),
        c_intrinsic.return_type.unwrap_or(""),
        rust_intrinsic.intrinsic,
        rust_intrinsic.arg_names.join(", "),
        rust_intrinsic.arg_types.join(", "),
        rust_intrinsic.generic_arg_names.join(", "),
        rust_intrinsic.generic_arg_types.join(", "),
        rust_intrinsic.return_type.unwrap_or(""),
    )
}

/// Create the spec sheet.
///
/// Fields that would be present in the spec sheet:
/// 1. c-intrinsic-name
/// 2. c-arguments
/// 3. c-arguments-data-types
/// 4. c-return-type
/// 5. rust-intrinsic-name
/// 6. rust-arguments
/// 7. rust-arguments-data-types
/// 8. rust-const-generic-arguments
/// 9. rust-const-generic-arguments-data-types
/// 10. rust-return-type
fn main() {
    // Read the file-paths from CLI arguments
    // obtain the tree of tokens from the code
    let args = Args::parse();
    let (c_source, c_tree) = process_c_code(args.c);
    let preproc_node = c_tree.root_node();

    let c_intrinsics = leaf_nodes_from_grammar_name(preproc_node, "function_definition")
        .iter()
        .map(|&node| CIntrinsic::new(node, &c_source))
        .collect::<Vec<_>>();

    let rust_intrinsics_interim = args
        .rust
        .into_iter()
        .map(|path| process_rust_code(path))
        .collect::<Vec<_>>();

    let rust_intrinsics = rust_intrinsics_interim
        .iter()
        .map(|(rust_source, rust_tree)| {
            let rust_source_str = rust_source.as_str();
            let mut rust_cursor = rust_tree.root_node().walk();
            rust_tree
                .root_node()
                .children(&mut rust_cursor)
                .filter(|node| node.grammar_name() == "function_item")
                .map(|node| RustIntrinsic::new(node, rust_source_str))
                .collect::<Vec<_>>()
        })
        .flatten()
        .collect::<Vec<_>>();

    let matching_intrinsics = match_intrinsic_definitions(&c_intrinsics, &rust_intrinsics);
    println!(
        "// This code is automatically generated. DO NOT MODIFY.
// Number of matched intrinsics: {}\n",
        matching_intrinsics.len()
    );

    let spec_details = matching_intrinsics
        .iter()
        .map(|&(c_intrinsic, rust_intrinsic)| generate_spec(c_intrinsic, rust_intrinsic))
        .collect::<Vec<_>>()
        .join("\n\n");
    println!("{}", spec_details);
}
