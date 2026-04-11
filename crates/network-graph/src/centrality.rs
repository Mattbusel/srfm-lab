/// Graph centrality measures for financial network analysis.
///
/// Implements:
/// - PageRank with configurable damping
/// - Betweenness centrality via Brandes O(VE) algorithm
/// - Eigenvector centrality via power iteration
/// - Katz centrality
/// - Closeness centrality
/// - HITS (Hub and Authority) scores
/// - Parallel computation via rayon where applicable

use std::collections::{HashMap, VecDeque, BinaryHeap};
use std::cmp::Ordering;
use rayon::prelude::*;

use crate::ricci_curvature::WeightedGraph;

// ── Centrality result ─────────────────────────────────────────────────────────

/// Full centrality report for all nodes in a graph.
#[derive(Debug, Clone)]
pub struct CentralityReport {
    pub n: usize,
    pub pagerank: Vec<f64>,
    pub betweenness: Vec<f64>,
    pub eigenvector: Vec<f64>,
    pub katz: Vec<f64>,
    pub closeness: Vec<f64>,
    pub hub_scores: Vec<f64>,
    pub authority_scores: Vec<f64>,
}

impl CentralityReport {
    /// Get the top-k nodes by a given centrality measure.
    pub fn top_k_by_pagerank(&self, k: usize) -> Vec<(usize, f64)> {
        top_k_indices(&self.pagerank, k)
    }

    pub fn top_k_by_betweenness(&self, k: usize) -> Vec<(usize, f64)> {
        top_k_indices(&self.betweenness, k)
    }

    pub fn top_k_by_authority(&self, k: usize) -> Vec<(usize, f64)> {
        top_k_indices(&self.authority_scores, k)
    }

    /// Composite systemic importance score (normalized sum of all centralities).
    pub fn systemic_importance(&self) -> Vec<f64> {
        let n = self.n;
        let normalize = |v: &[f64]| -> Vec<f64> {
            let max = v.iter().cloned().fold(0.0f64, f64::max);
            if max < 1e-12 { return vec![0.0; v.len()]; }
            v.iter().map(|x| x / max).collect()
        };
        let pr = normalize(&self.pagerank);
        let bw = normalize(&self.betweenness);
        let ev = normalize(&self.eigenvector);
        let cl = normalize(&self.closeness);
        let au = normalize(&self.authority_scores);

        (0..n).map(|i| (pr[i] + bw[i] + ev[i] + cl[i] + au[i]) / 5.0).collect()
    }
}

fn top_k_indices(scores: &[f64], k: usize) -> Vec<(usize, f64)> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(Ordering::Equal));
    idx.truncate(k);
    idx.into_iter().map(|i| (i, scores[i])).collect()
}

// ── PageRank ──────────────────────────────────────────────────────────────────

/// Configuration for PageRank.
#[derive(Debug, Clone)]
pub struct PageRankConfig {
    pub damping: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub personalized: Option<Vec<f64>>, // personalized teleportation vector
}

impl Default for PageRankConfig {
    fn default() -> Self {
        PageRankConfig {
            damping: 0.85,
            max_iter: 100,
            tol: 1e-8,
            personalized: None,
        }
    }
}

/// Compute PageRank scores for all nodes.
/// Uses the power iteration method on the column-stochastic transition matrix.
pub fn pagerank(graph: &WeightedGraph, config: &PageRankConfig) -> Vec<f64> {
    let n = graph.n;
    if n == 0 { return Vec::new(); }

    // Build column-stochastic transition matrix (sparse)
    // For out-degree normalization
    let out_strengths: Vec<f64> = (0..n).map(|u| graph.adj[u].iter().map(|(_, w)| w).sum()).collect();

    // Personalization vector
    let teleport: Vec<f64> = if let Some(pv) = &config.personalized {
        let s: f64 = pv.iter().sum();
        if s > 1e-12 { pv.iter().map(|x| x / s).collect() }
        else { vec![1.0 / n as f64; n] }
    } else {
        vec![1.0 / n as f64; n]
    };

    let mut pr = vec![1.0 / n as f64; n];
    let d = config.damping;
    let dangling_weight = (1.0 - d) / n as f64;

    for _ in 0..config.max_iter {
        let mut new_pr = vec![0.0f64; n];

        // Dangling nodes (no out-edges) distribute evenly
        let dangling_sum: f64 = (0..n)
            .filter(|&u| out_strengths[u] < 1e-12)
            .map(|u| pr[u])
            .sum();

        let dangling_contrib = d * dangling_sum / n as f64;

        for u in 0..n {
            let out_s = out_strengths[u];
            if out_s < 1e-12 { continue; }
            for &(v, w) in &graph.adj[u] {
                new_pr[v] += d * pr[u] * w / out_s;
            }
        }

        // Add teleportation and dangling contribution
        for v in 0..n {
            new_pr[v] += (1.0 - d) * teleport[v] + dangling_contrib * teleport[v];
        }

        // Normalize
        let sum: f64 = new_pr.iter().sum();
        if sum > 1e-12 {
            for x in &mut new_pr { *x /= sum; }
        }

        // Check convergence (L1 norm)
        let diff: f64 = pr.iter().zip(new_pr.iter()).map(|(a, b)| (a - b).abs()).sum();
        pr = new_pr;
        if diff < config.tol { break; }
    }
    pr
}

// ── Betweenness Centrality (Brandes) ─────────────────────────────────────────

/// Compute betweenness centrality using Brandes O(VE) algorithm.
/// For weighted graphs, uses Dijkstra for shortest paths.
pub fn betweenness_centrality(graph: &WeightedGraph, normalized: bool) -> Vec<f64> {
    let n = graph.n;
    if n <= 2 { return vec![0.0; n]; }

    // Parallel over source nodes
    let per_source: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|s| betweenness_from_source(graph, s, n))
        .collect();

    let mut bc = vec![0.0f64; n];
    for partial in per_source {
        for (i, v) in partial.iter().enumerate() {
            bc[i] += v;
        }
    }

    // Each pair counted twice (undirected) and divide by 2
    for v in &mut bc { *v /= 2.0; }

    if normalized && n > 2 {
        let scale = 1.0 / ((n - 1) * (n - 2)) as f64;
        for v in &mut bc { *v *= scale; }
    }
    bc
}

/// Brandes single-source betweenness contribution using Dijkstra.
fn betweenness_from_source(graph: &WeightedGraph, s: usize, n: usize) -> Vec<f64> {
    let mut sigma = vec![0.0f64; n]; // number of shortest paths
    let mut dist = vec![f64::INFINITY; n];
    let mut pred: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut delta = vec![0.0f64; n];
    let mut stack: Vec<usize> = Vec::new();

    sigma[s] = 1.0;
    dist[s] = 0.0;

    #[derive(Clone)]
    struct State { dist: f64, node: usize }
    impl PartialEq for State { fn eq(&self, o: &Self) -> bool { self.dist == o.dist } }
    impl Eq for State {}
    impl PartialOrd for State {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
    }
    impl Ord for State {
        fn cmp(&self, o: &Self) -> Ordering {
            o.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
        }
    }

    let mut heap = BinaryHeap::new();
    heap.push(State { dist: 0.0, node: s });

    while let Some(State { dist: d, node: u }) = heap.pop() {
        if d > dist[u] + 1e-12 { continue; }
        stack.push(u);

        // Undirected: traverse both directions
        let mut all_neighbors: Vec<(usize, f64)> = graph.adj[u].clone();
        for &(v, w) in &graph.radj[u] { all_neighbors.push((v, w)); }

        for (v, w) in all_neighbors {
            if w <= 0.0 { continue; }
            let nd = d + 1.0 / w;
            if nd < dist[v] - 1e-12 {
                dist[v] = nd;
                sigma[v] = 0.0;
                pred[v].clear();
                heap.push(State { dist: nd, node: v });
            }
            if (dist[v] - (d + 1.0 / w)).abs() < 1e-10 {
                sigma[v] += sigma[u];
                pred[v].push(u);
            }
        }
    }

    // Back-propagation
    while let Some(w) = stack.pop() {
        for &v in &pred[w] {
            if sigma[w] > 1e-12 {
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w]);
            }
        }
    }

    // delta[s] excluded (no centrality to source)
    let mut result = delta;
    result[s] = 0.0;
    result
}

// ── Eigenvector Centrality ────────────────────────────────────────────────────

/// Compute eigenvector centrality via power iteration.
/// The centrality of a node is proportional to the sum of centralities of its neighbors.
pub fn eigenvector_centrality(graph: &WeightedGraph, max_iter: usize, tol: f64) -> Vec<f64> {
    let n = graph.n;
    if n == 0 { return Vec::new(); }

    let mut ev = vec![1.0 / n as f64; n];

    for _ in 0..max_iter {
        let mut new_ev = vec![0.0f64; n];

        // Undirected version: sum over all neighbors
        for u in 0..n {
            for &(v, w) in &graph.adj[u] {
                new_ev[u] += w * ev[v];
                new_ev[v] += w * ev[u];
            }
        }

        // Normalize
        let norm: f64 = new_ev.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-14 { break; }
        for x in &mut new_ev { *x /= norm; }

        // Convergence
        let diff: f64 = ev.iter().zip(new_ev.iter()).map(|(a, b)| (a - b).abs()).sum();
        ev = new_ev;
        if diff < tol { break; }
    }

    // Normalize to [0, 1]
    let max = ev.iter().cloned().fold(0.0f64, f64::max);
    if max > 1e-12 { for x in &mut ev { *x /= max; } }
    ev
}

// ── Katz Centrality ───────────────────────────────────────────────────────────

/// Compute Katz centrality.
/// C_katz(u) = sum_{k=1}^{inf} alpha^k * (number of paths of length k to u)
/// Solved as: (I - alpha * A^T)^{-1} * 1 - 1 via power series.
pub fn katz_centrality(
    graph: &WeightedGraph,
    alpha: f64,
    beta: f64,
    max_iter: usize,
    tol: f64,
) -> Vec<f64> {
    let n = graph.n;
    if n == 0 { return Vec::new(); }

    let mut katz = vec![0.0f64; n];
    let mut contribution = vec![beta; n]; // initial vector

    for _ in 0..max_iter {
        // contribution_new = alpha * A^T * contribution
        let mut new_contrib = vec![0.0f64; n];
        for u in 0..n {
            for &(v, w) in &graph.adj[u] {
                // A^T: edge u->v contributes to new_contrib[u] from contrib[v]
                new_contrib[u] += alpha * w * contribution[v];
            }
        }

        let diff: f64 = contribution.iter().zip(new_contrib.iter()).map(|(a, b)| (a - b).abs()).sum();

        for i in 0..n { katz[i] += new_contrib[i]; }
        contribution = new_contrib;

        if diff < tol { break; }
    }

    // Normalize
    let max = katz.iter().cloned().fold(0.0f64, f64::max);
    if max > 1e-12 { for x in &mut katz { *x /= max; } }
    katz
}

// ── Closeness Centrality ──────────────────────────────────────────────────────

/// Compute closeness centrality for all nodes.
/// C_closeness(u) = (n - 1) / sum_{v != u} d(u, v)
pub fn closeness_centrality(graph: &WeightedGraph) -> Vec<f64> {
    let n = graph.n;
    if n <= 1 { return vec![0.0; n]; }

    (0..n).into_par_iter().map(|u| {
        let dist = dijkstra_undirected(graph, u);
        let reachable: Vec<f64> = dist.iter().enumerate()
            .filter(|&(v, &d)| v != u && d < f64::INFINITY)
            .map(|(_, &d)| d)
            .collect();

        if reachable.is_empty() { return 0.0; }
        let sum: f64 = reachable.iter().sum();
        if sum < 1e-14 { return 0.0; }

        let reachable_n = reachable.len() as f64;
        (reachable_n / (n - 1) as f64) * (reachable_n / sum)
    }).collect()
}

/// Dijkstra on undirected graph (uses both adj and radj).
fn dijkstra_undirected(graph: &WeightedGraph, src: usize) -> Vec<f64> {
    let n = graph.n;
    let mut dist = vec![f64::INFINITY; n];
    dist[src] = 0.0;

    #[derive(Clone)]
    struct State { dist: f64, node: usize }
    impl PartialEq for State { fn eq(&self, o: &Self) -> bool { self.dist == o.dist } }
    impl Eq for State {}
    impl PartialOrd for State {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
    }
    impl Ord for State {
        fn cmp(&self, o: &Self) -> Ordering {
            o.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
        }
    }

    let mut heap = BinaryHeap::new();
    heap.push(State { dist: 0.0, node: src });

    while let Some(State { dist: d, node: u }) = heap.pop() {
        if d > dist[u] + 1e-12 { continue; }
        for &(v, w) in graph.adj[u].iter().chain(graph.radj[u].iter()) {
            if w <= 0.0 { continue; }
            let nd = d + 1.0 / w;
            if nd < dist[v] {
                dist[v] = nd;
                heap.push(State { dist: nd, node: v });
            }
        }
    }
    dist
}

// ── HITS (Hub and Authority) ──────────────────────────────────────────────────

/// Result of HITS algorithm.
#[derive(Debug, Clone)]
pub struct HITSResult {
    pub hub_scores: Vec<f64>,
    pub authority_scores: Vec<f64>,
    pub iterations: usize,
}

/// Compute HITS hub and authority scores via power iteration.
pub fn hits(graph: &WeightedGraph, max_iter: usize, tol: f64) -> HITSResult {
    let n = graph.n;
    if n == 0 {
        return HITSResult { hub_scores: Vec::new(), authority_scores: Vec::new(), iterations: 0 };
    }

    let mut hub = vec![1.0 / n as f64; n];
    let mut auth = vec![1.0 / n as f64; n];
    let mut iter = 0;

    for i in 0..max_iter {
        iter = i + 1;

        // Authority update: auth[v] = sum_{u -> v} hub[u] * w(u,v)
        let mut new_auth = vec![0.0f64; n];
        for u in 0..n {
            for &(v, w) in &graph.adj[u] {
                new_auth[v] += hub[u] * w;
            }
        }

        // Hub update: hub[u] = sum_{u -> v} auth[v] * w(u,v)
        let mut new_hub = vec![0.0f64; n];
        for u in 0..n {
            for &(v, w) in &graph.adj[u] {
                new_hub[u] += new_auth[v] * w;
            }
        }

        // Normalize both
        let auth_norm: f64 = new_auth.iter().map(|x| x * x).sum::<f64>().sqrt();
        let hub_norm: f64 = new_hub.iter().map(|x| x * x).sum::<f64>().sqrt();

        if auth_norm > 1e-14 { for x in &mut new_auth { *x /= auth_norm; } }
        if hub_norm > 1e-14 { for x in &mut new_hub { *x /= hub_norm; } }

        // Convergence
        let auth_diff: f64 = auth.iter().zip(new_auth.iter()).map(|(a, b)| (a - b).abs()).sum();
        let hub_diff: f64 = hub.iter().zip(new_hub.iter()).map(|(a, b)| (a - b).abs()).sum();

        auth = new_auth;
        hub = new_hub;

        if auth_diff < tol && hub_diff < tol { break; }
    }

    HITSResult { hub_scores: hub, authority_scores: auth, iterations: iter }
}

// ── Full centrality computation ───────────────────────────────────────────────

/// Compute all centrality measures in one pass.
pub fn compute_all_centralities(graph: &WeightedGraph) -> CentralityReport {
    let n = graph.n;

    let pr_config = PageRankConfig::default();
    let pagerank = pagerank(graph, &pr_config);
    let betweenness = betweenness_centrality(graph, true);
    let eigenvector = eigenvector_centrality(graph, 200, 1e-8);
    let katz = katz_centrality(graph, 0.01, 1.0, 100, 1e-8);
    let closeness = closeness_centrality(graph);
    let hits_result = hits(graph, 200, 1e-8);

    CentralityReport {
        n,
        pagerank,
        betweenness,
        eigenvector,
        katz,
        closeness,
        hub_scores: hits_result.hub_scores,
        authority_scores: hits_result.authority_scores,
    }
}

// ── Temporal centrality tracking ──────────────────────────────────────────────

/// Track how centrality evolves over time for each node.
#[derive(Debug, Clone)]
pub struct CentralityTracker {
    pub timestamps: Vec<i64>,
    /// pagerank[t][node]
    pub pagerank_history: Vec<Vec<f64>>,
    pub betweenness_history: Vec<Vec<f64>>,
    pub authority_history: Vec<Vec<f64>>,
}

impl CentralityTracker {
    pub fn new() -> Self {
        CentralityTracker {
            timestamps: Vec::new(),
            pagerank_history: Vec::new(),
            betweenness_history: Vec::new(),
            authority_history: Vec::new(),
        }
    }

    pub fn update(&mut self, timestamp: i64, graph: &WeightedGraph) {
        self.timestamps.push(timestamp);
        let report = compute_all_centralities(graph);
        self.pagerank_history.push(report.pagerank);
        self.betweenness_history.push(report.betweenness);
        self.authority_history.push(report.authority_scores);
    }

    /// Return the time series of PageRank for a specific node.
    pub fn pagerank_series(&self, node: usize) -> Vec<f64> {
        self.pagerank_history.iter()
            .map(|pr| pr.get(node).copied().unwrap_or(0.0))
            .collect()
    }

    /// Identify nodes with the largest increase in authority score
    /// over the last `window` periods (potential emerging systemic risks).
    pub fn emerging_authorities(&self, window: usize, top_k: usize) -> Vec<(usize, f64)> {
        let t = self.authority_history.len();
        if t < window + 1 { return Vec::new(); }
        let n = self.authority_history[0].len();

        let mut changes: Vec<(usize, f64)> = (0..n).map(|node| {
            let old = self.authority_history[t - 1 - window].get(node).copied().unwrap_or(0.0);
            let new = self.authority_history[t - 1].get(node).copied().unwrap_or(0.0);
            (node, new - old)
        }).collect();

        changes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        changes.truncate(top_k);
        changes
    }
}

impl Default for CentralityTracker {
    fn default() -> Self { Self::new() }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn star_graph(n: usize) -> WeightedGraph {
        // Hub at 0, leaves at 1..n-1
        let mut g = WeightedGraph::new(n);
        for i in 1..n {
            g.add_edge(0, i, 1.0);
        }
        g
    }

    fn triangle() -> WeightedGraph {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(0, 2, 1.0);
        g
    }

    #[test]
    fn test_pagerank_sums_to_one() {
        let g = triangle();
        let pr = pagerank(&g, &PageRankConfig::default());
        let sum: f64 = pr.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "PageRank sum = {}", sum);
    }

    #[test]
    fn test_pagerank_star_hub_dominant() {
        let g = star_graph(5);
        let pr = pagerank(&g, &PageRankConfig::default());
        // Hub (node 0) should have highest PageRank
        let hub_pr = pr[0];
        for i in 1..5 {
            assert!(hub_pr >= pr[i], "Hub should dominate: {} < {}", hub_pr, pr[i]);
        }
    }

    #[test]
    fn test_betweenness_line() {
        let mut g = WeightedGraph::new(5);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0); g.add_edge(3, 4, 1.0);
        let bc = betweenness_centrality(&g, false);
        // Node 2 (center) should have highest betweenness
        let max_node = bc.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal)).unwrap().0;
        assert_eq!(max_node, 2);
    }

    #[test]
    fn test_eigenvector_centrality_star() {
        let g = star_graph(5);
        let ev = eigenvector_centrality(&g, 100, 1e-8);
        // Hub should have highest eigenvector centrality
        let max_node = ev.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(Ordering::Equal)).unwrap().0;
        assert_eq!(max_node, 0);
    }

    #[test]
    fn test_closeness_line() {
        let mut g = WeightedGraph::new(5);
        g.add_edge(0, 1, 1.0); g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0); g.add_edge(3, 4, 1.0);
        let cl = closeness_centrality(&g);
        // Center node should have higher closeness
        assert!(cl[2] >= cl[0]);
        assert!(cl[2] >= cl[4]);
    }

    #[test]
    fn test_hits_triangle() {
        let g = triangle();
        let result = hits(&g, 100, 1e-8);
        assert_eq!(result.hub_scores.len(), 3);
        assert_eq!(result.authority_scores.len(), 3);
        // In symmetric triangle, all scores should be equal
        let hub_mean = result.hub_scores.iter().sum::<f64>() / 3.0;
        for &h in &result.hub_scores {
            assert!((h - hub_mean).abs() < 1e-4);
        }
    }

    #[test]
    fn test_katz_centrality() {
        let g = star_graph(4);
        let katz = katz_centrality(&g, 0.01, 1.0, 100, 1e-8);
        assert_eq!(katz.len(), 4);
        // All values should be non-negative
        for k in &katz { assert!(*k >= 0.0); }
    }

    #[test]
    fn test_compute_all_centralities() {
        let g = triangle();
        let report = compute_all_centralities(&g);
        assert_eq!(report.n, 3);
        assert_eq!(report.pagerank.len(), 3);
        assert_eq!(report.betweenness.len(), 3);
        let systemic = report.systemic_importance();
        assert_eq!(systemic.len(), 3);
        for s in &systemic { assert!(*s >= 0.0 && *s <= 1.0); }
    }

    #[test]
    fn test_centrality_tracker() {
        let g = triangle();
        let mut tracker = CentralityTracker::new();
        tracker.update(100, &g);
        tracker.update(200, &g);
        let series = tracker.pagerank_series(0);
        assert_eq!(series.len(), 2);
    }
}
