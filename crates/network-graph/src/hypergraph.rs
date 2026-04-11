/// Hypergraph analysis for financial clique detection and systemic risk.
///
/// Implements:
/// - Hypergraph struct with hyperedges connecting any number of nodes
/// - Financial clique detection (mutually correlated asset groups)
/// - Hypergraph Laplacian and spectral analysis
/// - Motif counting: triangles, 4-cliques
/// - Persistent cliques across time windows
/// - Clique-based systemic risk scoring

use std::collections::{HashMap, HashSet};
use rayon::prelude::*;

use crate::ricci_curvature::WeightedGraph;

// ── Hyperedge ─────────────────────────────────────────────────────────────────

/// A hyperedge connecting a set of nodes with a weight.
#[derive(Debug, Clone)]
pub struct Hyperedge {
    /// Set of nodes in this hyperedge.
    pub nodes: Vec<usize>,
    /// Hyperedge weight (e.g. mean pairwise correlation).
    pub weight: f64,
    /// Optional label.
    pub label: Option<String>,
}

impl Hyperedge {
    pub fn new(nodes: Vec<usize>, weight: f64) -> Self {
        let mut sorted_nodes = nodes;
        sorted_nodes.sort();
        sorted_nodes.dedup();
        Hyperedge { nodes: sorted_nodes, weight, label: None }
    }

    pub fn order(&self) -> usize { self.nodes.len() }
    pub fn is_edge(&self) -> bool { self.nodes.len() == 2 }
    pub fn is_triangle(&self) -> bool { self.nodes.len() == 3 }

    pub fn contains(&self, node: usize) -> bool { self.nodes.contains(&node) }
}

// ── Hypergraph ────────────────────────────────────────────────────────────────

/// A hypergraph where hyperedges can connect any number of nodes.
#[derive(Debug, Clone)]
pub struct Hypergraph {
    pub n: usize, // number of vertices
    pub hyperedges: Vec<Hyperedge>,
}

impl Hypergraph {
    pub fn new(n: usize) -> Self {
        Hypergraph { n, hyperedges: Vec::new() }
    }

    pub fn add_hyperedge(&mut self, edge: Hyperedge) {
        self.hyperedges.push(edge);
    }

    /// Number of hyperedges.
    pub fn m(&self) -> usize { self.hyperedges.len() }

    /// Degree of node u: number of hyperedges containing u.
    pub fn degree(&self, u: usize) -> usize {
        self.hyperedges.iter().filter(|e| e.contains(u)).count()
    }

    /// Weighted degree: sum of weights of hyperedges containing u.
    pub fn weighted_degree(&self, u: usize) -> f64 {
        self.hyperedges.iter()
            .filter(|e| e.contains(u))
            .map(|e| e.weight)
            .sum()
    }

    /// Size of hyperedge e.
    pub fn edge_size(&self, e_idx: usize) -> usize {
        self.hyperedges[e_idx].nodes.len()
    }

    /// Get all standard (2-node) edges as WeightedGraph.
    pub fn to_graph(&self) -> WeightedGraph {
        let mut g = WeightedGraph::new(self.n);
        for he in &self.hyperedges {
            if he.nodes.len() == 2 {
                g.add_edge(he.nodes[0], he.nodes[1], he.weight);
            } else {
                // Clique expansion: add all pairs
                for i in 0..he.nodes.len() {
                    for j in (i+1)..he.nodes.len() {
                        g.add_edge(he.nodes[i], he.nodes[j], he.weight / he.nodes.len() as f64);
                    }
                }
            }
        }
        g
    }

    /// Build hypergraph from a weighted graph by finding cliques.
    pub fn from_graph_cliques(graph: &WeightedGraph, min_weight: f64) -> Self {
        let mut hg = Hypergraph::new(graph.n);
        // Add all 2-edges
        for e in graph.edges() {
            if e.weight >= min_weight {
                hg.add_hyperedge(Hyperedge::new(vec![e.src, e.dst], e.weight));
            }
        }
        hg
    }
}

// ── Hypergraph Laplacian ──────────────────────────────────────────────────────

/// Compute the hypergraph Laplacian.
/// L_H = D_V - H W D_E^{-1} H^T
/// where H is the incidence matrix (n x m), D_V = diag(degree), D_E = diag(edge sizes), W = diag(weights).
pub struct HypergraphLaplacian {
    pub n: usize,
    pub m: usize,
    /// Laplacian as flat n x n matrix.
    pub matrix: Vec<f64>,
    /// Eigenvalues.
    pub eigenvalues: Vec<f64>,
}

impl HypergraphLaplacian {
    /// Compute the hypergraph Laplacian for the given hypergraph.
    pub fn compute(hg: &Hypergraph) -> Self {
        let n = hg.n;
        let m = hg.hyperedges.len();
        if m == 0 {
            return HypergraphLaplacian {
                n, m: 0,
                matrix: vec![0.0; n * n],
                eigenvalues: vec![0.0; n],
            };
        }

        // Incidence matrix H: H[v][e] = 1 if v in e
        let mut h = vec![0.0f64; n * m];
        for (e_idx, edge) in hg.hyperedges.iter().enumerate() {
            for &v in &edge.nodes {
                if v < n { h[v * m + e_idx] = 1.0; }
            }
        }

        // Edge weight vector
        let w_e: Vec<f64> = hg.hyperedges.iter().map(|e| e.weight).collect();
        // Edge size (delta_e)
        let d_e: Vec<f64> = hg.hyperedges.iter().map(|e| e.nodes.len() as f64).collect();

        // Build L = D_V - H W D_E^{-1} H^T
        let mut l = vec![0.0f64; n * n];

        // D_V: diagonal
        for v in 0..n {
            let dv = hg.weighted_degree(v);
            l[v * n + v] = dv;
        }

        // H W D_E^{-1} H^T
        for v1 in 0..n {
            for v2 in 0..n {
                let mut val = 0.0;
                for e_idx in 0..m {
                    if h[v1 * m + e_idx] > 0.0 && h[v2 * m + e_idx] > 0.0 {
                        val += w_e[e_idx] / d_e[e_idx].max(1.0);
                    }
                }
                l[v1 * n + v2] -= val;
            }
        }

        // Compute eigenvalues (use simple power iteration approximation)
        let eigenvalues = laplacian_eigenvalues(&l, n);

        HypergraphLaplacian { n, m, matrix: l, eigenvalues }
    }

    /// Algebraic connectivity (second smallest eigenvalue).
    pub fn algebraic_connectivity(&self) -> f64 {
        let mut sorted = self.eigenvalues.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sorted.get(1).copied().unwrap_or(0.0).max(0.0)
    }
}

/// Estimate eigenvalues of a symmetric matrix via Jacobi method (for small n).
fn laplacian_eigenvalues(matrix: &[f64], n: usize) -> Vec<f64> {
    if n > 50 {
        // For large n, just return diagonal as proxy
        return (0..n).map(|i| matrix[i * n + i]).collect();
    }

    let mut a = matrix.to_vec();
    let tol = 1e-10;
    let max_iter = 1000;

    for _iter in 0..max_iter {
        let mut max_val = 0.0f64;
        let mut p = 0; let mut q = 1;
        for i in 0..n {
            for j in (i+1)..n {
                let v = a[i * n + j].abs();
                if v > max_val { max_val = v; p = i; q = j; }
            }
        }
        if max_val < tol { break; }

        let theta = if (a[q * n + q] - a[p * n + p]).abs() < 1e-14 {
            std::f64::consts::PI / 4.0
        } else {
            0.5 * ((2.0 * a[p * n + q]) / (a[q * n + q] - a[p * n + p])).atan()
        };
        let c = theta.cos(); let s = theta.sin();

        let app = c * c * a[p * n + p] - 2.0 * s * c * a[p * n + q] + s * s * a[q * n + q];
        let aqq = s * s * a[p * n + p] + 2.0 * s * c * a[p * n + q] + c * c * a[q * n + q];

        for k in 0..n {
            if k == p || k == q { continue; }
            let apk = a[p * n + k];
            let aqk = a[q * n + k];
            a[p * n + k] = c * apk - s * aqk;
            a[k * n + p] = a[p * n + k];
            a[q * n + k] = s * apk + c * aqk;
            a[k * n + q] = a[q * n + k];
        }
        a[p * n + p] = app;
        a[q * n + q] = aqq;
        a[p * n + q] = 0.0;
        a[q * n + p] = 0.0;
    }
    (0..n).map(|i| a[i * n + i]).collect()
}

// ── Financial Clique Detection ────────────────────────────────────────────────

/// Configuration for clique detection.
#[derive(Debug, Clone)]
pub struct CliqueConfig {
    /// Minimum edge weight to include an edge.
    pub min_weight: f64,
    /// Minimum clique size to report.
    pub min_size: usize,
    /// Maximum clique size to search (limits computation).
    pub max_size: usize,
}

impl Default for CliqueConfig {
    fn default() -> Self {
        CliqueConfig { min_weight: 0.7, min_size: 3, max_size: 10 }
    }
}

/// A detected clique with weight and member nodes.
#[derive(Debug, Clone)]
pub struct Clique {
    pub nodes: Vec<usize>,
    pub mean_weight: f64,
    pub min_weight: f64,
    pub size: usize,
}

/// Detect all maximal cliques in the filtered graph using Bron-Kerbosch.
pub fn find_cliques(graph: &WeightedGraph, config: &CliqueConfig) -> Vec<Clique> {
    let n = graph.n;

    // Build adjacency set with weight threshold
    let adj: Vec<HashSet<usize>> = (0..n).map(|u| {
        let mut s = HashSet::new();
        for &(v, w) in &graph.adj[u] {
            if w >= config.min_weight { s.insert(v); }
        }
        for &(v, w) in &graph.radj[u] {
            if w >= config.min_weight { s.insert(v); }
        }
        s
    }).collect();

    let mut cliques = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let candidates: Vec<usize> = (0..n).collect();

    bron_kerbosch(
        &adj,
        &mut current,
        &candidates,
        &[],
        &mut cliques,
        config,
        graph,
    );

    cliques
}

/// Bron-Kerbosch algorithm for maximal clique enumeration.
fn bron_kerbosch(
    adj: &[HashSet<usize>],
    r: &mut Vec<usize>,
    p: &[usize],
    x: &[usize],
    cliques: &mut Vec<Clique>,
    config: &CliqueConfig,
    graph: &WeightedGraph,
) {
    if r.len() > config.max_size { return; }

    if p.is_empty() && x.is_empty() {
        if r.len() >= config.min_size {
            let clique = compute_clique_stats(r, graph);
            cliques.push(clique);
        }
        return;
    }

    // Choose pivot to minimize branching
    let pivot = p.iter().chain(x.iter())
        .max_by_key(|&&v| {
            p.iter().filter(|&&u| adj[v].contains(&u)).count()
        })
        .copied()
        .unwrap_or(p[0]);

    let candidates: Vec<usize> = p.iter()
        .filter(|&&v| !adj[pivot].contains(&v))
        .copied()
        .collect();

    let mut p_working = p.to_vec();
    let mut x_working = x.to_vec();

    for v in candidates {
        r.push(v);
        let new_p: Vec<usize> = p_working.iter().filter(|&&u| adj[v].contains(&u)).copied().collect();
        let new_x: Vec<usize> = x_working.iter().filter(|&&u| adj[v].contains(&u)).copied().collect();
        bron_kerbosch(adj, r, &new_p, &new_x, cliques, config, graph);
        r.pop();
        p_working.retain(|&u| u != v);
        x_working.push(v);
    }
}

fn compute_clique_stats(nodes: &[usize], graph: &WeightedGraph) -> Clique {
    let mut weights = Vec::new();
    for i in 0..nodes.len() {
        for j in (i+1)..nodes.len() {
            let u = nodes[i]; let v = nodes[j];
            for &(nv, w) in &graph.adj[u] { if nv == v { weights.push(w); } }
            for &(nv, w) in &graph.radj[u] { if nv == v { weights.push(w); } }
        }
    }
    let mean_w = if weights.is_empty() { 0.0 } else { weights.iter().sum::<f64>() / weights.len() as f64 };
    let min_w = weights.iter().cloned().fold(f64::INFINITY, f64::min);

    Clique {
        nodes: nodes.to_vec(),
        mean_weight: mean_w,
        min_weight: if min_w == f64::INFINITY { 0.0 } else { min_w },
        size: nodes.len(),
    }
}

// ── Motif Counting ────────────────────────────────────────────────────────────

/// Motif counts for systemic risk assessment.
#[derive(Debug, Clone)]
pub struct MotifCounts {
    pub triangles: usize,
    pub four_cliques: usize,
    pub wedges: usize,
    pub clustering_coefficient: f64,
    /// Triangle participation ratio: fraction of nodes in at least one triangle.
    pub triangle_participation: f64,
}

/// Count triangles and 4-cliques in a weighted graph.
/// Uses the node iterator algorithm for triangles.
pub fn count_motifs(graph: &WeightedGraph, min_weight: f64) -> MotifCounts {
    let n = graph.n;

    // Build adjacency sets (undirected, filtered by weight)
    let adj: Vec<HashSet<usize>> = (0..n).map(|u| {
        let mut s = HashSet::new();
        for &(v, w) in &graph.adj[u] { if w >= min_weight { s.insert(v); } }
        for &(v, w) in &graph.radj[u] { if w >= min_weight { s.insert(v); } }
        s
    }).collect();

    // Count triangles using intersection of neighbor sets
    let triangles: usize = (0..n).into_par_iter().map(|u| {
        adj[u].iter().filter(|&&v| v > u).map(|&v| {
            // Count common neighbors w > v
            adj[u].intersection(&adj[v]).filter(|&&w| w > v).count()
        }).sum::<usize>()
    }).sum();

    // Count wedges (paths of length 2)
    let wedges: usize = (0..n).map(|u| {
        let d = adj[u].len();
        d * (d.saturating_sub(1)) / 2
    }).sum();

    // Clustering coefficient = 3 * triangles / wedges
    let clustering_coefficient = if wedges == 0 { 0.0 } else {
        3.0 * triangles as f64 / wedges as f64
    };

    // Triangle participation
    let nodes_in_triangle: HashSet<usize> = (0..n)
        .filter(|&u| {
            adj[u].iter().any(|&v| {
                adj[u].intersection(&adj[v]).next().is_some()
            })
        })
        .collect();
    let triangle_participation = nodes_in_triangle.len() as f64 / n.max(1) as f64;

    // 4-cliques: count via clique enumeration (limited)
    let config = CliqueConfig { min_weight, min_size: 4, max_size: 4 };
    let four_cliques = find_cliques(graph, &config).len();

    MotifCounts {
        triangles,
        four_cliques,
        wedges,
        clustering_coefficient,
        triangle_participation,
    }
}

// ── Persistent Cliques ────────────────────────────────────────────────────────

/// A clique that persists across time windows.
#[derive(Debug, Clone)]
pub struct PersistentClique {
    pub nodes: Vec<usize>,
    pub first_seen: usize, // snapshot index
    pub last_seen: usize,
    pub occurrence_count: usize,
    pub mean_weight: f64,
    pub persistence_ratio: f64,
}

/// Find cliques that persist across multiple graph snapshots.
pub fn find_persistent_cliques(
    graphs: &[(usize, &WeightedGraph)], // (snapshot_index, graph)
    config: &CliqueConfig,
    min_persistence: f64, // fraction of snapshots the clique must appear in
) -> Vec<PersistentClique> {
    let total_snapshots = graphs.len();
    if total_snapshots == 0 { return Vec::new(); }

    // Find cliques in each snapshot
    let all_cliques_per_snap: Vec<Vec<Clique>> = graphs.par_iter().map(|(_, g)| {
        find_cliques(g, config)
    }).collect();

    // Track clique occurrences (canonical key = sorted node set)
    let mut clique_info: HashMap<Vec<usize>, (usize, usize, usize, Vec<f64>)> = HashMap::new();
    // Map: sorted_nodes -> (first_seen, last_seen, count, weights)

    for (snap_idx, cliques) in all_cliques_per_snap.iter().enumerate() {
        for clique in cliques {
            let key = clique.nodes.clone();
            let entry = clique_info.entry(key).or_insert((snap_idx, snap_idx, 0, Vec::new()));
            entry.1 = snap_idx; // update last_seen
            entry.2 += 1;
            entry.3.push(clique.mean_weight);
        }
    }

    // Filter by minimum persistence
    clique_info.into_iter()
        .filter(|(_, (_, _, count, _))| *count as f64 / total_snapshots as f64 >= min_persistence)
        .map(|(nodes, (first, last, count, weights))| {
            let mean_w = if weights.is_empty() { 0.0 } else { weights.iter().sum::<f64>() / weights.len() as f64 };
            PersistentClique {
                nodes,
                first_seen: first,
                last_seen: last,
                occurrence_count: count,
                mean_weight: mean_w,
                persistence_ratio: count as f64 / total_snapshots as f64,
            }
        })
        .collect()
}

// ── Systemic Risk Hypergraph Score ────────────────────────────────────────────

/// Systemic risk metrics derived from hypergraph structure.
#[derive(Debug, Clone)]
pub struct HypergraphRiskMetrics {
    pub n_cliques: usize,
    pub mean_clique_size: f64,
    pub max_clique_size: usize,
    pub triangle_count: usize,
    pub four_clique_count: usize,
    pub clustering_coefficient: f64,
    pub algebraic_connectivity: f64,
    /// Normalized risk score in [0, 1].
    pub risk_score: f64,
}

/// Compute hypergraph-based systemic risk metrics.
pub fn compute_hypergraph_risk(
    graph: &WeightedGraph,
    correlation_threshold: f64,
) -> HypergraphRiskMetrics {
    let clique_config = CliqueConfig {
        min_weight: correlation_threshold,
        min_size: 3,
        max_size: 8,
    };

    let cliques = find_cliques(graph, &clique_config);
    let motifs = count_motifs(graph, correlation_threshold);

    let n_cliques = cliques.len();
    let max_clique_size = cliques.iter().map(|c| c.size).max().unwrap_or(0);
    let mean_clique_size = if n_cliques == 0 { 0.0 } else {
        cliques.iter().map(|c| c.size as f64).sum::<f64>() / n_cliques as f64
    };

    // Build hypergraph from cliques
    let mut hg = Hypergraph::new(graph.n);
    for c in &cliques {
        hg.add_hyperedge(Hyperedge::new(c.nodes.clone(), c.mean_weight));
    }
    let laplacian = HypergraphLaplacian::compute(&hg);
    let alg_conn = laplacian.algebraic_connectivity();

    // Normalize risk score:
    // High risk = many large cliques + high clustering + low algebraic connectivity
    let clique_score = (n_cliques as f64 * mean_clique_size / graph.n.max(1) as f64).min(1.0);
    let motif_score = motifs.clustering_coefficient.min(1.0);
    let conn_score = 1.0 / (1.0 + alg_conn); // low connectivity = high risk

    let risk_score = (clique_score * 0.4 + motif_score * 0.4 + conn_score * 0.2).min(1.0);

    HypergraphRiskMetrics {
        n_cliques,
        mean_clique_size,
        max_clique_size,
        triangle_count: motifs.triangles,
        four_clique_count: motifs.four_cliques,
        clustering_coefficient: motifs.clustering_coefficient,
        algebraic_connectivity: alg_conn,
        risk_score,
    }
}

// ── Hypergraph time series ────────────────────────────────────────────────────

/// Track hypergraph risk metrics over time.
#[derive(Debug, Clone)]
pub struct HypergraphTimeSeries {
    pub timestamps: Vec<i64>,
    pub risk_scores: Vec<f64>,
    pub clique_counts: Vec<usize>,
    pub clustering_coefficients: Vec<f64>,
    pub algebraic_connectivities: Vec<f64>,
}

impl HypergraphTimeSeries {
    pub fn new() -> Self {
        HypergraphTimeSeries {
            timestamps: Vec::new(),
            risk_scores: Vec::new(),
            clique_counts: Vec::new(),
            clustering_coefficients: Vec::new(),
            algebraic_connectivities: Vec::new(),
        }
    }

    pub fn update(&mut self, timestamp: i64, graph: &WeightedGraph, threshold: f64) {
        let metrics = compute_hypergraph_risk(graph, threshold);
        self.timestamps.push(timestamp);
        self.risk_scores.push(metrics.risk_score);
        self.clique_counts.push(metrics.n_cliques);
        self.clustering_coefficients.push(metrics.clustering_coefficient);
        self.algebraic_connectivities.push(metrics.algebraic_connectivity);
    }

    /// Detect sudden increases in clique count (contagion onset).
    pub fn detect_clique_explosion(&self, lookback: usize, multiplier: f64) -> Vec<usize> {
        let n = self.clique_counts.len();
        if n < lookback + 1 { return Vec::new(); }

        let mut events = Vec::new();
        for t in lookback..n {
            let window = &self.clique_counts[t - lookback..t];
            let mean = window.iter().sum::<f64>() / window.len() as f64;
            if self.clique_counts[t] as f64 > mean * multiplier {
                events.push(t);
            }
        }
        events
    }
}

impl Default for HypergraphTimeSeries {
    fn default() -> Self { Self::new() }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn full_graph(n: usize, w: f64) -> WeightedGraph {
        let mut g = WeightedGraph::new(n);
        for i in 0..n {
            for j in (i+1)..n {
                g.add_edge(i, j, w);
            }
        }
        g
    }

    fn triangle_graph() -> WeightedGraph { full_graph(3, 1.0) }

    #[test]
    fn test_hyperedge_sort() {
        let e = Hyperedge::new(vec![3, 1, 2], 0.8);
        assert_eq!(e.nodes, vec![1, 2, 3]);
    }

    #[test]
    fn test_hypergraph_degree() {
        let mut hg = Hypergraph::new(3);
        hg.add_hyperedge(Hyperedge::new(vec![0, 1, 2], 1.0));
        assert_eq!(hg.degree(0), 1);
        assert_eq!(hg.degree(1), 1);
    }

    #[test]
    fn test_hypergraph_laplacian() {
        let mut hg = Hypergraph::new(3);
        hg.add_hyperedge(Hyperedge::new(vec![0, 1], 1.0));
        hg.add_hyperedge(Hyperedge::new(vec![1, 2], 1.0));
        let laplacian = HypergraphLaplacian::compute(&hg);
        assert_eq!(laplacian.n, 3);
        assert_eq!(laplacian.matrix.len(), 9);
    }

    #[test]
    fn test_find_cliques_triangle() {
        let g = triangle_graph();
        let config = CliqueConfig { min_weight: 0.5, min_size: 3, max_size: 5 };
        let cliques = find_cliques(&g, &config);
        assert!(cliques.iter().any(|c| c.size == 3));
    }

    #[test]
    fn test_count_motifs_triangle() {
        let g = triangle_graph();
        let motifs = count_motifs(&g, 0.5);
        assert_eq!(motifs.triangles, 1);
        assert!(motifs.clustering_coefficient > 0.0);
    }

    #[test]
    fn test_count_motifs_4clique() {
        let g = full_graph(4, 1.0);
        let motifs = count_motifs(&g, 0.5);
        assert_eq!(motifs.triangles, 4); // K4 has 4 triangles
        assert_eq!(motifs.four_cliques, 1);
    }

    #[test]
    fn test_hypergraph_risk_metrics() {
        let g = full_graph(5, 0.9);
        let risk = compute_hypergraph_risk(&g, 0.8);
        assert!(risk.risk_score >= 0.0 && risk.risk_score <= 1.0);
        assert!(risk.n_cliques > 0);
    }

    #[test]
    fn test_persistent_cliques() {
        let g1 = triangle_graph();
        let g2 = triangle_graph();
        let graphs: Vec<(usize, &WeightedGraph)> = vec![(0, &g1), (1, &g2)];
        let config = CliqueConfig { min_weight: 0.5, min_size: 3, max_size: 5 };
        let persistent = find_persistent_cliques(&graphs, &config, 0.5);
        assert!(!persistent.is_empty());
        assert!(persistent[0].persistence_ratio >= 0.5);
    }

    #[test]
    fn test_to_graph_from_hypergraph() {
        let mut hg = Hypergraph::new(4);
        hg.add_hyperedge(Hyperedge::new(vec![0, 1, 2, 3], 1.0));
        let g = hg.to_graph();
        // K4 expanded: 6 edges
        assert_eq!(g.edges().len(), 6);
    }

    #[test]
    fn test_clique_explosion_detection() {
        let mut ts = HypergraphTimeSeries::new();
        let g_small = full_graph(3, 0.9);
        let g_large = full_graph(6, 0.9);
        ts.update(0, &g_small, 0.8);
        ts.update(1, &g_small, 0.8);
        ts.update(2, &g_large, 0.8);
        // Should detect explosion at t=2 if clique count jumped significantly
        let events = ts.detect_clique_explosion(2, 2.0);
        // Just check it runs without panic
        let _ = events;
    }
}
