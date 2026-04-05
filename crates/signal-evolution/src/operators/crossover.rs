/// Subtree crossover operator for expression tree evolution.
///
/// Algorithm:
///   1. Enumerate all nodes in each parent tree.
///   2. Pick a random non-root node from parent A (crossover point A).
///   3. Pick a random node from parent B (crossover point B).
///   4. Replace the subtree at point A with the subtree at point B.
///   5. Enforce max-depth constraint: if resulting tree exceeds max_depth,
///      replace the subtree at point A with a random terminal instead.

use crate::expression_tree::Node;
use rand::Rng;

/// Perform subtree crossover between two parent trees.
/// Returns a new child tree. `max_depth` enforces structural constraints.
pub fn subtree_crossover(
    parent_a: &Node,
    parent_b: &Node,
    max_depth: usize,
    rng: &mut impl Rng,
) -> Node {
    // Collect all node paths (as index vectors for navigation)
    let nodes_a = collect_node_indices(parent_a);
    let nodes_b = collect_node_indices(parent_b);

    if nodes_a.is_empty() || nodes_b.is_empty() {
        return parent_a.clone();
    }

    // Pick crossover points — prefer internal nodes in A (skip root by using range 1..)
    let idx_a = if nodes_a.len() > 1 {
        rng.gen_range(1..nodes_a.len())
    } else {
        0
    };
    let idx_b = rng.gen_range(0..nodes_b.len());

    let path_a = &nodes_a[idx_a];
    let path_b = &nodes_b[idx_b];

    // Extract the donor subtree from parent B
    let donor = get_subtree(parent_b, path_b);

    // Build child by replacing subtree in parent A at path_a with donor
    let mut child = parent_a.clone();
    replace_subtree(&mut child, path_a, donor);

    // Enforce depth constraint
    if child.depth() > max_depth {
        // Fallback: return parent_a unchanged to preserve validity
        return parent_a.clone();
    }

    child
}

/// A path is a Vec<usize> giving child indices to traverse from root.
/// [] = root, [0] = first child of root, [0, 1] = second child of first child, etc.
fn collect_node_indices(node: &Node) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    collect_recursive(node, &[], &mut result);
    result
}

fn collect_recursive(node: &Node, path: &[usize], result: &mut Vec<Vec<usize>>) {
    result.push(path.to_vec());
    if let Node::Function(_, children) = node {
        for (i, child) in children.iter().enumerate() {
            let mut child_path = path.to_vec();
            child_path.push(i);
            collect_recursive(child, &child_path, result);
        }
    }
}

/// Navigate to the node at `path` and return a clone of it.
fn get_subtree(node: &Node, path: &[usize]) -> Node {
    if path.is_empty() {
        return node.clone();
    }
    match node {
        Node::Function(_, children) => {
            let idx = path[0];
            if idx < children.len() {
                get_subtree(&children[idx], &path[1..])
            } else {
                node.clone()
            }
        }
        Node::Terminal(_) => node.clone(),
    }
}

/// Replace the subtree at `path` in `node` with `replacement` in-place.
fn replace_subtree(node: &mut Node, path: &[usize], replacement: Node) {
    if path.is_empty() {
        *node = replacement;
        return;
    }
    if let Node::Function(_, children) = node {
        let idx = path[0];
        if idx < children.len() {
            replace_subtree(&mut children[idx], &path[1..], replacement);
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

    fn make_simple_tree() -> Node {
        Node::Function(
            Operator::Add,
            vec![
                Node::Terminal(Terminal::Price),
                Node::Terminal(Terminal::Volume),
            ],
        )
    }

    #[test]
    fn crossover_preserves_depth_constraint() {
        let mut rng = SmallRng::seed_from_u64(42);
        let a = make_simple_tree();
        let b = make_simple_tree();
        let child = subtree_crossover(&a, &b, 5, &mut rng);
        assert!(child.depth() <= 5, "Depth constraint violated");
    }

    #[test]
    fn crossover_returns_valid_tree() {
        let mut rng = SmallRng::seed_from_u64(99);
        let a = make_simple_tree();
        let b = Node::Terminal(Terminal::RSI { period: 14 });
        let child = subtree_crossover(&a, &b, 5, &mut rng);
        // Should not panic and should be valid
        assert!(child.complexity() >= 1);
    }

    #[test]
    fn collect_indices_counts_nodes() {
        let tree = make_simple_tree();
        let indices = collect_node_indices(&tree);
        // Root + 2 children = 3
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn get_subtree_root() {
        let tree = make_simple_tree();
        let sub = get_subtree(&tree, &[]);
        assert_eq!(sub.complexity(), tree.complexity());
    }

    #[test]
    fn get_subtree_child() {
        let tree = make_simple_tree();
        let sub = get_subtree(&tree, &[0]);
        // Should be Terminal(Price)
        assert!(matches!(sub, Node::Terminal(Terminal::Price)));
    }
}
