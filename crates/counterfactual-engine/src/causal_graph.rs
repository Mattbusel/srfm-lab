//! Causal graph for the LARSA signal system.
//!
//! Represents the causal structure between signals, filters, actions and
//! outcomes in the IAE / LARSA trading strategy.
//!
//! # Graph structure
//!
//! ```text
//! BH_MASS ---------> ENTRY_SIGNAL ------> POSITION_SIZE --> OUTCOME
//!                         ^                     ^
//!                         |                     |
//! NAV_GEODESIC --> ENTRY_GATE (blocks)   HURST_H --> SIZE_MODIFIER
//! ```
//!
//! Each edge carries a `strength` (correlation-based coefficient in [-1, 1]).
//! `total_effect(cause, effect)` sums coefficients over all directed paths.

use std::collections::{HashMap, HashSet, VecDeque};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Node types
// ---------------------------------------------------------------------------

/// Type of a causal node in the LARSA signal graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeType {
    /// A raw market signal (e.g., BH_MASS, HURST_H, NAV_GEODESIC).
    Signal,
    /// A filter that can gate or block downstream signals.
    Filter,
    /// An action taken by the strategy (e.g., ENTRY, SIZE_MODIFIER).
    Action,
    /// A measurable outcome (P&L, POSITION_SIZE, DRAWDOWN).
    Outcome,
}

impl NodeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            NodeType::Signal  => "Signal",
            NodeType::Filter  => "Filter",
            NodeType::Action  => "Action",
            NodeType::Outcome => "Outcome",
        }
    }
}

// ---------------------------------------------------------------------------
// CausalNode
// ---------------------------------------------------------------------------

/// A node in the causal graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalNode {
    /// Unique variable name (e.g., "BH_MASS", "ENTRY_SIGNAL").
    pub name: String,
    /// Semantic role in the signal pipeline.
    pub node_type: NodeType,
    /// Optional description.
    pub description: String,
}

impl CausalNode {
    pub fn new(name: &str, node_type: NodeType, description: &str) -> Self {
        Self {
            name: name.to_string(),
            node_type,
            description: description.to_string(),
        }
    }

    pub fn signal(name: &str, description: &str) -> Self {
        Self::new(name, NodeType::Signal, description)
    }

    pub fn filter(name: &str, description: &str) -> Self {
        Self::new(name, NodeType::Filter, description)
    }

    pub fn action(name: &str, description: &str) -> Self {
        Self::new(name, NodeType::Action, description)
    }

    pub fn outcome(name: &str, description: &str) -> Self {
        Self::new(name, NodeType::Outcome, description)
    }
}

// ---------------------------------------------------------------------------
// Edge types
// ---------------------------------------------------------------------------

/// Semantic type of a causal edge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum EdgeType {
    /// Direct causal influence from cause to effect.
    Direct,
    /// Modulation: scales or adjusts (not binary on/off).
    Modulated,
    /// Blocking: the source node can suppress the target node.
    Blocked,
}

impl EdgeType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EdgeType::Direct    => "Direct",
            EdgeType::Modulated => "Modulated",
            EdgeType::Blocked   => "Blocked",
        }
    }
}

// ---------------------------------------------------------------------------
// CausalEdge
// ---------------------------------------------------------------------------

/// A directed causal edge between two nodes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEdge {
    /// Source node name.
    pub from: String,
    /// Destination node name.
    pub to: String,
    /// Type of causal relationship.
    pub edge_type: EdgeType,
    /// Correlation-based strength in [-1, 1].
    /// Positive = reinforcing, negative = inhibiting.
    pub strength: f64,
    /// Optional label (e.g., "bh_mass_thresh_gate").
    pub label: String,
}

impl CausalEdge {
    pub fn new(from: &str, to: &str, edge_type: EdgeType, strength: f64) -> Self {
        Self {
            from: from.to_string(),
            to: to.to_string(),
            edge_type,
            strength: strength.clamp(-1.0, 1.0),
            label: String::new(),
        }
    }

    pub fn with_label(mut self, label: &str) -> Self {
        self.label = label.to_string();
        self
    }

    /// Direct edge with given strength.
    pub fn direct(from: &str, to: &str, strength: f64) -> Self {
        Self::new(from, to, EdgeType::Direct, strength)
    }

    /// Modulated edge with given strength.
    pub fn modulated(from: &str, to: &str, strength: f64) -> Self {
        Self::new(from, to, EdgeType::Modulated, strength)
    }

    /// Blocking edge (negative strength = suppression).
    pub fn blocked(from: &str, to: &str, strength: f64) -> Self {
        Self::new(from, to, EdgeType::Blocked, -strength.abs())
    }
}

// ---------------------------------------------------------------------------
// CausalGraph
// ---------------------------------------------------------------------------

/// Directed causal graph for the LARSA signal pipeline.
///
/// Nodes represent variables; edges represent causal relationships with
/// empirically estimated or theoretically motivated strengths.
///
/// # Example: build the default LARSA causal graph
///
/// ```
/// use counterfactual_engine::causal_graph::CausalGraph;
/// let g = CausalGraph::larsa_default();
/// let paths = g.find_paths("BH_MASS", "OUTCOME");
/// let total = g.total_effect("BH_MASS", "OUTCOME");
/// println!("BH_MASS -> OUTCOME total effect: {total:.4}");
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalGraph {
    /// All nodes, keyed by name.
    nodes: HashMap<String, CausalNode>,
    /// All edges.
    edges: Vec<CausalEdge>,
    /// Adjacency list: from -> list of (to, edge_index).
    adjacency: HashMap<String, Vec<usize>>,
}

impl CausalGraph {
    /// Create an empty causal graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            adjacency: HashMap::new(),
        }
    }

    /// Add a node. Returns `true` if inserted, `false` if already present.
    pub fn add_node(&mut self, node: CausalNode) -> bool {
        let name = node.name.clone();
        if self.nodes.contains_key(&name) {
            return false;
        }
        self.nodes.insert(name.clone(), node);
        self.adjacency.entry(name).or_default();
        true
    }

    /// Add an edge. Both endpoints must already be nodes.
    ///
    /// Returns `Err` if either endpoint is missing.
    pub fn add_edge(&mut self, edge: CausalEdge) -> Result<(), String> {
        if !self.nodes.contains_key(&edge.from) {
            return Err(format!("Source node '{}' not found", edge.from));
        }
        if !self.nodes.contains_key(&edge.to) {
            return Err(format!("Target node '{}' not found", edge.to));
        }
        let idx = self.edges.len();
        let from = edge.from.clone();
        self.edges.push(edge);
        self.adjacency.entry(from).or_default().push(idx);
        Ok(())
    }

    /// Retrieve a node by name.
    pub fn get_node(&self, name: &str) -> Option<&CausalNode> {
        self.nodes.get(name)
    }

    /// Retrieve all edges from a given node.
    pub fn edges_from(&self, node: &str) -> Vec<&CausalEdge> {
        self.adjacency
            .get(node)
            .map(|indices| indices.iter().map(|&i| &self.edges[i]).collect())
            .unwrap_or_default()
    }

    /// All node names.
    pub fn node_names(&self) -> Vec<&str> {
        self.nodes.keys().map(|s| s.as_str()).collect()
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Find all simple (acyclic) directed paths from `from` to `to`.
    ///
    /// Returns each path as a `Vec<String>` of node names (inclusive).
    /// Uses BFS with visited set to avoid cycles.
    pub fn find_paths(&self, from: &str, to: &str) -> Vec<Vec<String>> {
        if !self.nodes.contains_key(from) || !self.nodes.contains_key(to) {
            return Vec::new();
        }

        let mut all_paths: Vec<Vec<String>> = Vec::new();
        // Queue: (current_node, path_so_far, visited_set)
        let mut queue: VecDeque<(String, Vec<String>, HashSet<String>)> = VecDeque::new();

        let mut initial_visited = HashSet::new();
        initial_visited.insert(from.to_string());
        queue.push_back((from.to_string(), vec![from.to_string()], initial_visited));

        while let Some((current, path, visited)) = queue.pop_front() {
            if current == to {
                if path.len() > 1 {
                    all_paths.push(path);
                }
                continue;
            }

            if path.len() > self.nodes.len() {
                // Guard against very deep graphs
                continue;
            }

            for edge in self.edges_from(&current) {
                if !visited.contains(&edge.to) {
                    let mut new_path = path.clone();
                    new_path.push(edge.to.clone());
                    let mut new_visited = visited.clone();
                    new_visited.insert(edge.to.clone());
                    queue.push_back((edge.to.clone(), new_path, new_visited));
                }
            }
        }

        all_paths
    }

    /// Compute the path coefficient for a given path (product of edge strengths).
    ///
    /// The path is specified as a sequence of node names.  Returns 0.0 if
    /// any edge in the path is missing.
    pub fn compute_path_coefficient(&self, path: &[String]) -> f64 {
        if path.len() < 2 {
            return 1.0;
        }
        let mut product = 1.0f64;
        for w in path.windows(2) {
            let from = &w[0];
            let to = &w[1];
            // Find the edge from -> to
            let edge_opt = self.adjacency
                .get(from)
                .and_then(|indices| {
                    indices.iter().find_map(|&i| {
                        if self.edges[i].to == *to {
                            Some(&self.edges[i])
                        } else {
                            None
                        }
                    })
                });
            match edge_opt {
                Some(e) => product *= e.strength,
                None => return 0.0, // path has a missing edge
            }
        }
        product
    }

    /// Compute the total causal effect of `cause` on `effect`.
    ///
    /// Total effect = sum of path coefficients over all directed paths.
    ///
    /// Interpretation:
    ///   Positive total effect = cause reinforces effect.
    ///   Negative total effect = cause suppresses effect.
    ///   Magnitude = expected change in effect per unit change in cause.
    pub fn total_effect(&self, cause: &str, effect: &str) -> f64 {
        let paths = self.find_paths(cause, effect);
        paths.iter().map(|p| self.compute_path_coefficient(p)).sum()
    }

    /// Find all nodes that are ancestors of `target` (direct or indirect causes).
    pub fn ancestors(&self, target: &str) -> HashSet<String> {
        let mut result = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        queue.push_back(target.to_string());

        // Build reverse adjacency
        let mut reverse: HashMap<&str, Vec<&str>> = HashMap::new();
        for edge in &self.edges {
            reverse.entry(edge.to.as_str()).or_default().push(edge.from.as_str());
        }

        while let Some(node) = queue.pop_front() {
            if let Some(preds) = reverse.get(node.as_str()) {
                for &pred in preds {
                    if result.insert(pred.to_string()) {
                        queue.push_back(pred.to_string());
                    }
                }
            }
        }
        result
    }

    /// Find all nodes that are descendants of `source` (direct or indirect effects).
    pub fn descendants(&self, source: &str) -> HashSet<String> {
        let mut result = HashSet::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        queue.push_back(source.to_string());

        while let Some(node) = queue.pop_front() {
            for edge in self.edges_from(&node) {
                if result.insert(edge.to.clone()) {
                    queue.push_back(edge.to.clone());
                }
            }
        }
        result
    }

    /// Check whether there is any directed path from `from` to `to`.
    pub fn is_cause_of(&self, from: &str, to: &str) -> bool {
        !self.find_paths(from, to).is_empty()
    }

    /// Return the topological order of nodes (Kahn's algorithm).
    /// Returns `None` if the graph has a cycle.
    pub fn topological_order(&self) -> Option<Vec<String>> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        for name in self.nodes.keys() {
            in_degree.insert(name.as_str(), 0);
        }
        for edge in &self.edges {
            *in_degree.entry(edge.to.as_str()).or_insert(0) += 1;
        }

        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&n, _)| n)
            .collect();

        let mut order: Vec<String> = Vec::new();

        while let Some(node) = queue.pop_front() {
            order.push(node.to_string());
            for edge in self.edges_from(node) {
                let d = in_degree.get_mut(edge.to.as_str()).unwrap();
                *d -= 1;
                if *d == 0 {
                    queue.push_back(edge.to.as_str());
                }
            }
        }

        if order.len() == self.nodes.len() {
            Some(order)
        } else {
            None // cycle detected
        }
    }

    /// Compute the direct effect (single-hop strength) from `from` to `to`.
    /// Returns 0.0 if no direct edge exists.
    pub fn direct_effect(&self, from: &str, to: &str) -> f64 {
        self.adjacency
            .get(from)
            .and_then(|indices| {
                indices.iter().find_map(|&i| {
                    if self.edges[i].to == to {
                        Some(self.edges[i].strength)
                    } else {
                        None
                    }
                })
            })
            .unwrap_or(0.0)
    }

    /// Build the default LARSA causal graph with empirically motivated edge strengths.
    ///
    /// # Graph topology
    ///
    /// ```text
    /// BH_MASS (0.85) --Direct--> ENTRY_SIGNAL (0.72) --Direct--> POSITION_SIZE --> OUTCOME
    ///                                   |
    /// NAV_GEODESIC --Blocked(0.60)--> ENTRY_GATE --Blocked--> ENTRY_SIGNAL (gates it)
    ///
    /// HURST_H (0.45) --Direct--> SIZE_MODIFIER (0.55) --Modulated--> POSITION_SIZE
    /// ```
    pub fn larsa_default() -> Self {
        let mut g = CausalGraph::new();

        // Nodes
        g.add_node(CausalNode::signal("BH_MASS",
            "Black Hole mass -- primary entry signal"));
        g.add_node(CausalNode::signal("HURST_H",
            "Hurst exponent -- persistence / regime signal"));
        g.add_node(CausalNode::signal("NAV_GEODESIC",
            "NAV geodesic distance -- portfolio health signal"));
        g.add_node(CausalNode::filter("ENTRY_GATE",
            "Gate that can block entry signal when NAV too low"));
        g.add_node(CausalNode::action("ENTRY_SIGNAL",
            "Combined entry signal after all filters"));
        g.add_node(CausalNode::action("SIZE_MODIFIER",
            "Hurst-based position size modifier"));
        g.add_node(CausalNode::action("POSITION_SIZE",
            "Final position size after all modifiers"));
        g.add_node(CausalNode::outcome("OUTCOME",
            "Strategy P&L outcome"));

        // Edges
        // BH_MASS -> ENTRY_SIGNAL: strong direct effect
        g.add_edge(CausalEdge::direct("BH_MASS", "ENTRY_SIGNAL", 0.85)).unwrap();

        // NAV_GEODESIC -> ENTRY_GATE: when geodesic is low, gate activates
        g.add_edge(CausalEdge::blocked("NAV_GEODESIC", "ENTRY_GATE", 0.60)).unwrap();

        // ENTRY_GATE -> ENTRY_SIGNAL: gate can block the signal
        g.add_edge(CausalEdge::blocked("ENTRY_GATE", "ENTRY_SIGNAL", 0.45)).unwrap();

        // ENTRY_SIGNAL -> POSITION_SIZE: signal directly sizes position
        g.add_edge(CausalEdge::direct("ENTRY_SIGNAL", "POSITION_SIZE", 0.72)).unwrap();

        // HURST_H -> SIZE_MODIFIER: high Hurst -> larger modifier
        g.add_edge(CausalEdge::direct("HURST_H", "SIZE_MODIFIER", 0.55)).unwrap();

        // SIZE_MODIFIER -> POSITION_SIZE: modulates (multiplies) position
        g.add_edge(CausalEdge::modulated("SIZE_MODIFIER", "POSITION_SIZE", 0.40)).unwrap();

        // POSITION_SIZE -> OUTCOME: position size determines P&L magnitude
        g.add_edge(CausalEdge::direct("POSITION_SIZE", "OUTCOME", 0.78)).unwrap();

        g
    }
}

impl Default for CausalGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Path analysis utilities
// ---------------------------------------------------------------------------

/// A directed path with its coefficient and component edges.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathAnalysis {
    pub path: Vec<String>,
    pub coefficient: f64,
    pub edge_types: Vec<String>,
    pub edge_strengths: Vec<f64>,
}

impl PathAnalysis {
    /// Compute path analysis for a given path in a graph.
    pub fn compute(graph: &CausalGraph, path: &[String]) -> Self {
        let coefficient = graph.compute_path_coefficient(path);
        let mut edge_types = Vec::new();
        let mut edge_strengths = Vec::new();

        for w in path.windows(2) {
            let from = &w[0];
            let to = &w[1];
            if let Some(indices) = graph.adjacency.get(from.as_str()) {
                for &i in indices {
                    if graph.edges[i].to == *to {
                        edge_types.push(graph.edges[i].edge_type.as_str().to_string());
                        edge_strengths.push(graph.edges[i].strength);
                        break;
                    }
                }
            }
        }

        PathAnalysis {
            path: path.to_vec(),
            coefficient,
            edge_types,
            edge_strengths,
        }
    }
}

/// Analyze all paths from a source to a target and report coefficients.
pub fn analyze_paths(graph: &CausalGraph, from: &str, to: &str) -> Vec<PathAnalysis> {
    graph.find_paths(from, to)
        .iter()
        .map(|p| PathAnalysis::compute(graph, p))
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_simple_graph() -> CausalGraph {
        let mut g = CausalGraph::new();
        g.add_node(CausalNode::signal("A", "source signal"));
        g.add_node(CausalNode::action("B", "intermediate action"));
        g.add_node(CausalNode::outcome("C", "outcome"));
        g.add_edge(CausalEdge::direct("A", "B", 0.8)).unwrap();
        g.add_edge(CausalEdge::direct("B", "C", 0.6)).unwrap();
        g
    }

    #[test]
    fn test_add_node_and_edge() {
        let g = build_simple_graph();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_find_paths_simple() {
        let g = build_simple_graph();
        let paths = g.find_paths("A", "C");
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], vec!["A", "B", "C"]);
    }

    #[test]
    fn test_find_paths_no_path() {
        let g = build_simple_graph();
        let paths = g.find_paths("C", "A"); // reverse direction -- no path
        assert!(paths.is_empty());
    }

    #[test]
    fn test_path_coefficient_product_of_strengths() {
        let g = build_simple_graph();
        let path: Vec<String> = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let coef = g.compute_path_coefficient(&path);
        assert!((coef - 0.48).abs() < 1e-9, "0.8 * 0.6 = 0.48, got {coef}");
    }

    #[test]
    fn test_total_effect_single_path() {
        let g = build_simple_graph();
        let effect = g.total_effect("A", "C");
        assert!((effect - 0.48).abs() < 1e-9, "expected 0.48, got {effect}");
    }

    #[test]
    fn test_total_effect_two_paths() {
        let mut g = build_simple_graph();
        // Add second path A -> D -> C
        g.add_node(CausalNode::action("D", "bypass"));
        g.add_edge(CausalEdge::direct("A", "D", 0.5)).unwrap();
        g.add_edge(CausalEdge::direct("D", "C", 0.4)).unwrap();

        let effect = g.total_effect("A", "C");
        // path 1: A->B->C = 0.8*0.6 = 0.48
        // path 2: A->D->C = 0.5*0.4 = 0.20
        // total = 0.68
        assert!((effect - 0.68).abs() < 1e-9, "expected 0.68, got {effect}");
    }

    #[test]
    fn test_direct_effect() {
        let g = build_simple_graph();
        assert!((g.direct_effect("A", "B") - 0.8).abs() < 1e-9);
        assert!((g.direct_effect("A", "C")).abs() < 1e-9); // no direct edge
    }

    #[test]
    fn test_ancestors() {
        let g = build_simple_graph();
        let anc = g.ancestors("C");
        assert!(anc.contains("A"));
        assert!(anc.contains("B"));
        assert!(!anc.contains("C"));
    }

    #[test]
    fn test_descendants() {
        let g = build_simple_graph();
        let desc = g.descendants("A");
        assert!(desc.contains("B"));
        assert!(desc.contains("C"));
        assert!(!desc.contains("A"));
    }

    #[test]
    fn test_is_cause_of() {
        let g = build_simple_graph();
        assert!(g.is_cause_of("A", "C"));
        assert!(!g.is_cause_of("C", "A"));
    }

    #[test]
    fn test_topological_order() {
        let g = build_simple_graph();
        let order = g.topological_order();
        assert!(order.is_some());
        let order = order.unwrap();
        let pos_a = order.iter().position(|n| n == "A").unwrap();
        let pos_b = order.iter().position(|n| n == "B").unwrap();
        let pos_c = order.iter().position(|n| n == "C").unwrap();
        assert!(pos_a < pos_b, "A should come before B");
        assert!(pos_b < pos_c, "B should come before C");
    }

    #[test]
    fn test_larsa_default_graph_structure() {
        let g = CausalGraph::larsa_default();
        assert!(g.node_count() >= 7, "expected at least 7 nodes");
        assert!(g.edge_count() >= 6, "expected at least 6 edges");
    }

    #[test]
    fn test_bh_mass_causes_outcome_in_larsa() {
        let g = CausalGraph::larsa_default();
        assert!(g.is_cause_of("BH_MASS", "OUTCOME"),
            "BH_MASS should cause OUTCOME through the signal chain");
    }

    #[test]
    fn test_hurst_h_causes_outcome() {
        let g = CausalGraph::larsa_default();
        assert!(g.is_cause_of("HURST_H", "OUTCOME"),
            "HURST_H should cause OUTCOME via SIZE_MODIFIER -> POSITION_SIZE");
    }

    #[test]
    fn test_larsa_causal_path_coefficient_positive() {
        let g = CausalGraph::larsa_default();
        let coef = g.total_effect("BH_MASS", "OUTCOME");
        // All Direct edges are positive so total effect should be positive
        assert!(coef > 0.0, "BH_MASS -> OUTCOME total effect should be positive, got {coef}");
    }

    #[test]
    fn test_blocking_edge_negative_strength() {
        let g = CausalGraph::larsa_default();
        let d = g.direct_effect("NAV_GEODESIC", "ENTRY_GATE");
        assert!(d < 0.0, "Blocking edge NAV_GEODESIC -> ENTRY_GATE should have negative strength, got {d}");
    }

    #[test]
    fn test_path_analysis_edge_types_len_matches_path_len() {
        let g = CausalGraph::larsa_default();
        let paths = g.find_paths("BH_MASS", "OUTCOME");
        for path in &paths {
            let analysis = PathAnalysis::compute(&g, path);
            assert_eq!(
                analysis.edge_types.len(),
                path.len() - 1,
                "edge_types length should equal path length - 1"
            );
        }
    }

    #[test]
    fn test_missing_node_add_edge_errors() {
        let mut g = CausalGraph::new();
        g.add_node(CausalNode::signal("X", ""));
        let result = g.add_edge(CausalEdge::direct("X", "MISSING", 0.5));
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_node_not_added() {
        let mut g = CausalGraph::new();
        g.add_node(CausalNode::signal("X", "first"));
        let added = g.add_node(CausalNode::signal("X", "second"));
        assert!(!added, "duplicate node should not be inserted");
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_analyze_paths_utility() {
        let g = CausalGraph::larsa_default();
        let analyses = analyze_paths(&g, "BH_MASS", "OUTCOME");
        assert!(!analyses.is_empty(), "should find at least one path");
        for a in &analyses {
            assert!(a.coefficient.is_finite(), "coefficient should be finite");
        }
    }
}
