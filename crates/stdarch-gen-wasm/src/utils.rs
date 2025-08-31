use tree_sitter::Node;

/// Recursively searches the node and its children for a node
/// that matches its grammar name, using Depth-first search.
pub fn leaf_nodes_from_grammar_name<'a>(node: Node<'a>, name: &str) -> Vec<Node<'a>> {
    if node.grammar_name() == name {
        // handle the base case
        vec![node]
    } else if node.child_count() > 0 {
        // search through the children
        (0..node.child_count())
            .into_iter()
            .filter_map(|index| {
                let tags = leaf_nodes_from_grammar_name(node.child(index).unwrap(), name);
                if tags.len() == 0 { None } else { Some(tags) }
            })
            .flatten()
            .collect()
    } else {
        // Node has no children and doesnt have its grammar name as `name`.
        // Return empty vector.
        Vec::new()
    }
}
