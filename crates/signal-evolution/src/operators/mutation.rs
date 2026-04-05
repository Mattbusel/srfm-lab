/// Mutation operators for expression tree evolution.
///
/// Three mutation types:
///
/// 1. Point mutation   — Replace a random terminal node with another random terminal.
/// 2. Subtree mutation — Replace a random subtree with a freshly grown random tree.
/// 3. Hoist mutation   — Replace a node with one of its own subtrees (tree simplification).

use crate::expression_tree::Node;
use crate::primitives::Terminal;
use rand::prelude::SliceRandom;
use rand::Rng;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Apply a random mutation to `tree`. Returns the mutated tree (original unchanged).
pub fn mutate(tree: &Node, max_depth: usize, rng: &mut impl Rng) -> Node {
    let choice: u8 = rng.gen_range(0..3);
    match choice {
        0 => point_mutation(tree, rng),
        1 => subtree_mutation(tree, max_depth, rng),
        _ => hoist_mutation(tree, rng),
    }
}

/// Replace a random terminal node with a different random terminal.
pub fn point_mutation(tree: &Node, rng: &mut impl Rng) -> Node {
    let terminal_paths = collect_terminal_paths(tree);
    if terminal_paths.is_empty() {
        return tree.clone();
    }
    let path = terminal_paths.choose(rng).unwrap().clone();
    let all_terminals = Terminal::all_variants();
    let new_terminal = all_terminals.choose(rng).unwrap().clone();
    let mut child = tree.clone();
    replace_node_at_path(&mut child, &path, Node::Terminal(new_terminal));
    child
}

/// Replace a random subtree with a freshly grown random tree of bounded depth.
pub fn subtree_mutation(tree: &Node, max_depth: usize, rng: &mut impl Rng) -> Node {
    let all_paths = collect_all_paths(tree);
    if all_paths.is_empty() {
        return tree.clone();
    }
    // Prefer non-root for replacement
    let path = if all_paths.len() > 1 {
        all_paths[1..].choose(rng).unwrap().clone()
    } else {
        all_paths[0].clone()
    };
    let new_subtree_depth = rng.gen_range(1..=max_depth.min(3));
    let new_subtree = Node::random_tree(new_subtree_depth, rng);
    let mut child = tree.clone();
    replace_node_at_path(&mut child, &path, new_subtree);
    // Enforce depth constraint
    if child.depth() > max_depth {
        return tree.clone();
    }
    child
}

/// Replace a random internal node with one of its own subtrees (simplification).
pub fn hoist_mutation(tree: &Node, rng: &mut impl Rng) -> Node {
    let function_paths = collect_function_paths(tree);
    if function_paths.is_empty() {
        // Nothing to hoist — return as-is
        return tree.clone();
    }
    let path = function_paths.choose(rng).unwrap().clone();
    // Get the node at that path
    let node = get_node_at_path(tree, &path);
    // Pick a random child subtree to hoist up
    if let Node::Function(_, children) = &node {
        if children.is_empty() {
            return tree.clone();
        }
        let child_idx = rng.gen_range(0..children.len());
        let hoisted = children[child_idx].clone();
        let mut result = tree.clone();
        replace_node_at_path(&mut result, &path, hoisted);
        result
    } else {
        tree.clone()
    }
}

// ---------------------------------------------------------------------------
// Path navigation helpers
// ---------------------------------------------------------------------------

fn collect_all_paths(node: &Node) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    collect_paths_recursive(node, &[], &mut result);
    result
}

fn collect_terminal_paths(node: &Node) -> Vec<Vec<usize>> {
    collect_all_paths(node)
        .into_iter()
        .filter(|path| matches!(get_node_at_path(node, path), Node::Terminal(_)))
        .collect()
}

fn collect_function_paths(node: &Node) -> Vec<Vec<usize>> {
    collect_all_paths(node)
        .into_iter()
        .filter(|path| matches!(get_node_at_path(node, path), Node::Function(_, _)))
        .collect()
}

fn collect_paths_recursive(node: &Node, path: &[usize], result: &mut Vec<Vec<usize>>) {
    result.push(path.to_vec());
    if let Node::Function(_, children) = node {
        for (i, child) in children.iter().enumerate() {
            let mut child_path = path.to_vec();
            child_path.push(i);
            collect_paths_recursive(child, &child_path, result);
        }
    }
}

fn get_node_at_path(node: &Node, path: &[usize]) -> Node {
    if path.is_empty() {
        return node.clone();
    }
    match node {
        Node::Function(_, children) => {
            let idx = path[0];
            if idx < children.len() {
                get_node_at_path(&children[idx], &path[1..])
            } else {
                node.clone()
            }
        }
        Node::Terminal(_) => node.clone(),
    }
}

fn replace_node_at_path(node: &mut Node, path: &[usize], replacement: Node) {
    if path.is_empty() {
        *node = replacement;
        return;
    }
    if let Node::Function(_, children) = node {
        let idx = path[0];
        if idx < children.len() {
            replace_node_at_path(&mut children[idx], &path[1..], replacement);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expression_tree::Node;
    use crate::primitives::Terminal;
    use crate::operators::Operator;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn sample_tree() -> Node {
        Node::Function(
            Operator::Add,
            vec![
                Node::Terminal(Terminal::Price),
                Node::Function(
                    Operator::Mul,
                    vec![
                        Node::Terminal(Terminal::RSI { period: 14 }),
                        Node::Terminal(Terminal::Volume),
                    ],
                ),
            ],
        )
    }

    #[test]
    fn point_mutation_produces_valid_tree() {
        let mut rng = SmallRng::seed_from_u64(1);
        let tree = sample_tree();
        let mutated = point_mutation(&tree, &mut rng);
        assert!(mutated.complexity() >= 1);
    }

    #[test]
    fn subtree_mutation_respects_depth() {
        let mut rng = SmallRng::seed_from_u64(2);
        let tree = sample_tree();
        for _ in 0..20 {
            let mutated = subtree_mutation(&tree, 6, &mut rng);
            assert!(mutated.depth() <= 6, "Depth exceeded: {}", mutated.depth());
        }
    }

    #[test]
    fn hoist_reduces_or_preserves_complexity() {
        let mut rng = SmallRng::seed_from_u64(3);
        let tree = sample_tree();
        let orig_complexity = tree.complexity();
        let hoisted = hoist_mutation(&tree, &mut rng);
        assert!(
            hoisted.complexity() <= orig_complexity,
            "Hoist increased complexity: {} > {}",
            hoisted.complexity(),
            orig_complexity
        );
    }

    #[test]
    fn mutate_never_panics() {
        let mut rng = SmallRng::seed_from_u64(77);
        let tree = sample_tree();
        for _ in 0..50 {
            let m = mutate(&tree, 7, &mut rng);
            assert!(m.complexity() >= 1);
        }
    }
}
