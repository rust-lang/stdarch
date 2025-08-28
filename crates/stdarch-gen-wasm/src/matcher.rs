use std::collections::HashSet;

use crate::structs::{CIntrinsic, RustIntrinsic};

/// Matches the set of intrinsics in Rust to their C counterpart.
/// 
/// This function assumes that the list of Rust definitions
/// will be a subset of the list of definitions in C.
pub fn match_intrinsic_definitions<'a>(
    c_definitions: &'a Vec<CIntrinsic>,
    rust_definitions: &'a Vec<RustIntrinsic>,
) -> Vec<(&'a CIntrinsic<'a>, &'a RustIntrinsic<'a>)> {
    // This function assumes that the list of Rust definitions
    // will be a subset of the list of definitions in C

    let mut matched_definitions: Vec<(&'a CIntrinsic, &'a RustIntrinsic)> = Vec::new();
    matched_definitions.reserve(rust_definitions.len());

    for rust_definition in rust_definitions.iter() {
        let c_definition = c_definitions
            .iter()
            .find(|&c_def| match_intrinsic_definition(c_def.intrinsic, rust_definition.intrinsic));
        if let Some(c_def) = c_definition {
            matched_definitions.push((c_def, rust_definition));
        }
    }

    matched_definitions
}

/// checks if the function name of the intrinsic in Rust
/// matches that of the intrinsic in C.
fn match_intrinsic_definition(c_definition: &str, rust_definition: &str) -> bool {
    // Most intrinsics in C are of the format: `wasm_v128_load`.
    // Its Rust counterpart is named `v128_load`.
    // 
    // Another one is `wasm_i8x16_const_splat`, and its Rust counterpart is `i8x16_splat`.
    // 
    // The pattern that is observed is that, each keyword "chunk" that constructs 
    // the intrinsic name in Rust will also be used to construct the intrinsic name in C.
    // These names are constructed by joining the chunks with an underscore (_).

    let c_definition_split: HashSet<_> = c_definition.split('_').collect();
    rust_definition
        .split('_')
        .all(|keyword| c_definition_split.contains(keyword))
}
