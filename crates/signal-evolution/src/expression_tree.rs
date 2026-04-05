/// Signal expression tree — the core data structure of the genetic programming engine.
///
/// A tree is either a Terminal leaf or a Function node with child subtrees.
/// Trees are evaluated against a BarData slice and produce a Vec<f64> signal series.

use crate::data_loader::BarData;
use crate::operators::Operator;
use crate::primitives::Terminal;
use rand::prelude::SliceRandom;
use rand::Rng;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Node enum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Node {
    Terminal(Terminal),
    Function(Operator, Vec<Node>),
}

impl Node {
    // ------------------------------------------------------------------
    // Evaluation
    // ------------------------------------------------------------------

    /// Evaluate the expression tree against a bar history, producing one value per bar.
    /// Returns a zero-filled Vec<f64> on empty input.
    pub fn evaluate(&self, bars: &[BarData]) -> Vec<f64> {
        if bars.is_empty() {
            return Vec::new();
        }
        match self {
            Node::Terminal(t) => t.evaluate(bars),
            Node::Function(op, children) => {
                let child_vals: Vec<Vec<f64>> = children.iter().map(|c| c.evaluate(bars)).collect();
                op.apply(&child_vals)
            }
        }
    }

    // ------------------------------------------------------------------
    // Structural properties
    // ------------------------------------------------------------------

    /// Maximum depth of the tree. A single terminal has depth 1.
    pub fn depth(&self) -> usize {
        match self {
            Node::Terminal(_) => 1,
            Node::Function(_, children) => {
                1 + children.iter().map(|c| c.depth()).max().unwrap_or(0)
            }
        }
    }

    /// Total number of nodes (complexity measure). Larger = more complex.
    pub fn complexity(&self) -> usize {
        match self {
            Node::Terminal(_) => 1,
            Node::Function(_, children) => 1 + children.iter().map(|c| c.complexity()).sum::<usize>(),
        }
    }

    /// Number of terminal nodes in the tree.
    pub fn terminal_count(&self) -> usize {
        match self {
            Node::Terminal(_) => 1,
            Node::Function(_, children) => children.iter().map(|c| c.terminal_count()).sum(),
        }
    }

    /// Number of function nodes in the tree.
    pub fn function_count(&self) -> usize {
        match self {
            Node::Terminal(_) => 0,
            Node::Function(_, children) => 1 + children.iter().map(|c| c.function_count()).sum::<usize>(),
        }
    }

    // ------------------------------------------------------------------
    // Display
    // ------------------------------------------------------------------

    /// Human-readable formula string, e.g. "(Price + RSI(14))".
    pub fn to_formula(&self) -> String {
        match self {
            Node::Terminal(t) => t.name(),
            Node::Function(op, children) => {
                let child_strs: Vec<String> = children.iter().map(|c| c.to_formula()).collect();
                match op.arity() {
                    1 => format!("{}({})", op.name(), child_strs[0]),
                    2 => format!("({} {} {})", child_strs[0], op.name(), child_strs[1]),
                    _ => format!("{}({})", op.name(), child_strs.join(", ")),
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Random tree generation
    // ------------------------------------------------------------------

    /// Generate a random expression tree with bounded depth.
    /// Uses ramped half-and-half strategy: 50% full, 50% grow.
    pub fn random_tree(max_depth: usize, rng: &mut impl Rng) -> Node {
        if rng.gen_bool(0.5) {
            Node::grow(max_depth, rng)
        } else {
            Node::full(max_depth, rng)
        }
    }

    /// "Full" method: all branches extended to max_depth.
    pub fn full(max_depth: usize, rng: &mut impl Rng) -> Node {
        if max_depth <= 1 {
            return Node::random_terminal(rng);
        }
        let op = Node::random_operator(rng);
        let arity = op.arity();
        let children = (0..arity).map(|_| Node::full(max_depth - 1, rng)).collect();
        Node::Function(op, children)
    }

    /// "Grow" method: branches can terminate early.
    pub fn grow(max_depth: usize, rng: &mut impl Rng) -> Node {
        if max_depth <= 1 {
            return Node::random_terminal(rng);
        }
        // 40% chance to become terminal at any depth > 1
        if rng.gen_bool(0.4) {
            return Node::random_terminal(rng);
        }
        let op = Node::random_operator(rng);
        let arity = op.arity();
        let children = (0..arity).map(|_| Node::grow(max_depth - 1, rng)).collect();
        Node::Function(op, children)
    }

    fn random_terminal(rng: &mut impl Rng) -> Node {
        let variants = Terminal::all_variants();
        let t = variants.choose(rng).unwrap().clone();
        Node::Terminal(t)
    }

    fn random_operator(rng: &mut impl Rng) -> Operator {
        // Bias toward binary operators (more interesting trees)
        if rng.gen_bool(0.7) {
            let variants = Operator::binary_variants();
            variants.choose(rng).unwrap().clone()
        } else {
            let variants = Operator::unary_variants();
            variants.choose(rng).unwrap().clone()
        }
    }

    // ------------------------------------------------------------------
    // Serialisation
    // ------------------------------------------------------------------

    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("Node serialisation is infallible")
    }

    pub fn from_json(s: &str) -> anyhow::Result<Self> {
        serde_json::from_str(s).map_err(Into::into)
    }
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_formula())
    }
}

// ---------------------------------------------------------------------------
// Signal tree wrapper (individual in the population)
// ---------------------------------------------------------------------------

/// A named, fitness-annotated expression tree that represents an evolved signal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalTree {
    pub id: String,
    pub tree: Node,
    pub generation: u32,
    pub fitness: Option<crate::fitness::FitnessVector>,
}

impl SignalTree {
    pub fn new(tree: Node, generation: u32) -> Self {
        Self {
            id: Self::generate_id(),
            tree,
            generation,
            fitness: None,
        }
    }

    pub fn random(max_depth: usize, generation: u32, rng: &mut impl Rng) -> Self {
        let depth = rng.gen_range(2..=max_depth);
        let tree = Node::random_tree(depth, rng);
        Self::new(tree, generation)
    }

    pub fn evaluate(&self, bars: &[BarData]) -> Vec<f64> {
        self.tree.evaluate(bars)
    }

    pub fn formula(&self) -> String {
        self.tree.to_formula()
    }

    pub fn depth(&self) -> usize {
        self.tree.depth()
    }

    pub fn complexity(&self) -> usize {
        self.tree.complexity()
    }

    pub fn ic(&self) -> f64 {
        self.fitness.as_ref().map(|f| f.ic).unwrap_or(f64::NEG_INFINITY)
    }

    pub fn icir(&self) -> f64 {
        self.fitness.as_ref().map(|f| f.icir).unwrap_or(f64::NEG_INFINITY)
    }

    fn generate_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        format!("sig_{:08x}", t)
    }
}

impl std::fmt::Display for SignalTree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "SignalTree(id={}, depth={}, complexity={}, ic={:.4})",
            self.id,
            self.depth(),
            self.complexity(),
            self.ic()
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_loader::synthetic_bars;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    fn rng() -> SmallRng {
        SmallRng::seed_from_u64(42)
    }

    #[test]
    fn terminal_evaluates_to_correct_length() {
        let bars = synthetic_bars(100, 100.0);
        let node = Node::Terminal(Terminal::Price);
        let vals = node.evaluate(&bars);
        assert_eq!(vals.len(), 100);
    }

    #[test]
    fn function_node_evaluates() {
        let bars = synthetic_bars(50, 100.0);
        let node = Node::Function(
            Operator::Add,
            vec![
                Node::Terminal(Terminal::Price),
                Node::Terminal(Terminal::Volume),
            ],
        );
        let vals = node.evaluate(&bars);
        assert_eq!(vals.len(), 50);
    }

    #[test]
    fn depth_single_terminal() {
        let node = Node::Terminal(Terminal::Price);
        assert_eq!(node.depth(), 1);
    }

    #[test]
    fn depth_nested() {
        let node = Node::Function(
            Operator::Add,
            vec![
                Node::Terminal(Terminal::Price),
                Node::Function(
                    Operator::Mul,
                    vec![
                        Node::Terminal(Terminal::Volume),
                        Node::Terminal(Terminal::RSI { period: 14 }),
                    ],
                ),
            ],
        );
        assert_eq!(node.depth(), 3);
    }

    #[test]
    fn complexity_counts_all_nodes() {
        let node = Node::Function(
            Operator::Add,
            vec![Node::Terminal(Terminal::Price), Node::Terminal(Terminal::Volume)],
        );
        // 1 function + 2 terminals = 3
        assert_eq!(node.complexity(), 3);
    }

    #[test]
    fn formula_produces_string() {
        let node = Node::Function(
            Operator::Add,
            vec![Node::Terminal(Terminal::Price), Node::Terminal(Terminal::Volume)],
        );
        let f = node.to_formula();
        assert!(f.contains('+'));
        assert!(f.contains("Price"));
        assert!(f.contains("Volume"));
    }

    #[test]
    fn random_tree_respects_max_depth() {
        let mut rng = rng();
        for _ in 0..30 {
            let tree = Node::random_tree(5, &mut rng);
            assert!(tree.depth() <= 5, "depth {} exceeds max 5", tree.depth());
        }
    }

    #[test]
    fn random_tree_evaluate_no_panic() {
        let bars = synthetic_bars(100, 100.0);
        let mut rng = rng();
        for _ in 0..20 {
            let tree = Node::random_tree(4, &mut rng);
            let vals = tree.evaluate(&bars);
            assert_eq!(vals.len(), 100);
            // No NaN allowed
            for v in &vals {
                assert!(!v.is_nan(), "NaN in tree output: {}", tree.to_formula());
            }
        }
    }

    #[test]
    fn json_round_trip() {
        let mut rng = rng();
        let tree = Node::random_tree(3, &mut rng);
        let json = tree.to_json();
        let restored = Node::from_json(&json).unwrap();
        let bars = synthetic_bars(50, 100.0);
        let v1 = tree.evaluate(&bars);
        let v2 = restored.evaluate(&bars);
        for (a, b) in v1.iter().zip(v2.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn signal_tree_wrapper() {
        let mut rng = rng();
        let st = SignalTree::random(5, 0, &mut rng);
        assert!(st.depth() >= 1);
        let bars = synthetic_bars(80, 100.0);
        let vals = st.evaluate(&bars);
        assert_eq!(vals.len(), 80);
    }
}
