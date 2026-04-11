/// Temporal graph analysis: sequences of graph snapshots, structural change detection,
/// wormhole detection, contagion path tracing, and persistent topology features.

use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Ordering;
use rayon::prelude::*;

use crate::ricci_curvature::{WeightedGraph, Edge};

// ── Snapshot ──────────────────────────────────────────────────────────────────

/// A single time-stamped graph snapshot.
#[derive(Debug, Clone)]
pub struct GraphSnapshot {
    pub timestamp: i64,
    pub graph: WeightedGraph,
    /// Optional label (e.g. regime name)
    pub label: Option<String>,
}

impl GraphSnapshot {
    pub fn new(timestamp: i64, graph: WeightedGraph) -> Self {
        GraphSnapshot { timestamp, graph, label: None }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Edge set as (src, dst) -> weight map.
    pub fn edge_map(&self) -> HashMap<(usize, usize), f64> {
        self.graph.edges().into_iter().map(|e| ((e.src, e.dst), e.weight)).collect()
    }

    /// Set of node indices with at least one edge.
    pub fn active_nodes(&self) -> HashSet<usize> {
        let mut s = HashSet::new();
        for e in self.graph.edges() {
            s.insert(e.src);
            s.insert(e.dst);
        }
        s
    }
}

// ── Temporal Graph ────────────────────────────────────────────────────────────

/// Sequence of graph snapshots ordered by time.
#[derive(Debug, Clone)]
pub struct TemporalGraph {
    pub snapshots: Vec<GraphSnapshot>,
    pub n_nodes: usize,
}

impl TemporalGraph {
    pub fn new(n_nodes: usize) -> Self {
        TemporalGraph { snapshots: Vec::new(), n_nodes }
    }

    pub fn push(&mut self, snapshot: GraphSnapshot) {
        self.snapshots.push(snapshot);
        // Keep sorted by timestamp
        self.snapshots.sort_by_key(|s| s.timestamp);
    }

    pub fn len(&self) -> usize { self.snapshots.len() }
    pub fn is_empty(&self) -> bool { self.snapshots.is_empty() }

    /// Get the latest snapshot.
    pub fn latest(&self) -> Option<&GraphSnapshot> { self.snapshots.last() }

    /// Get snapshot at index.
    pub fn at(&self, idx: usize) -> Option<&GraphSnapshot> { self.snapshots.get(idx) }

    /// Compute pairwise structural change metrics between consecutive snapshots.
    pub fn compute_structural_changes(&self) -> Vec<StructuralChange> {
        if self.snapshots.len() < 2 { return Vec::new(); }

        self.snapshots
            .par_windows(2)
            .map(|w| structural_change(&w[0], &w[1]))
            .collect()
    }

    /// Compute velocity of structural change (d(topology)/dt).
    pub fn topology_velocity(&self) -> Vec<f64> {
        let changes = self.compute_structural_changes();
        changes.iter().map(|c| c.edit_distance as f64 / c.dt.max(1.0) as f64).collect()
    }

    /// Detect wormhole edges: sudden high-weight edges between normally distant nodes.
    pub fn detect_wormholes(&self, lookback: usize, z_threshold: f64) -> Vec<WormholeEvent> {
        if self.snapshots.len() < lookback + 1 { return Vec::new(); }
        let n = self.snapshots.len();
        let mut events = Vec::new();

        for t in lookback..n {
            let current = &self.snapshots[t];
            let history = &self.snapshots[t - lookback..t];
            let wormholes = detect_wormhole_edges(current, history, z_threshold);
            events.extend(wormholes);
        }
        events
    }

    /// Find persistent edges: edges present in at least `min_count` snapshots.
    pub fn persistent_edges(&self, min_count: usize) -> Vec<PersistentEdge> {
        let mut counts: HashMap<(usize, usize), (usize, f64)> = HashMap::new();
        for snap in &self.snapshots {
            for e in snap.graph.edges() {
                let entry = counts.entry((e.src, e.dst)).or_insert((0, 0.0));
                entry.0 += 1;
                entry.1 += e.weight;
            }
        }
        counts.into_iter()
            .filter(|(_, (count, _))| *count >= min_count)
            .map(|((src, dst), (count, total_w))| PersistentEdge {
                src, dst,
                occurrence_count: count,
                mean_weight: total_w / count as f64,
                persistence_ratio: count as f64 / self.snapshots.len() as f64,
            })
            .collect()
    }

    /// Edge birth/death rates over the temporal graph.
    pub fn edge_birth_death_rates(&self) -> Vec<EdgeFluxMetrics> {
        if self.snapshots.len() < 2 { return Vec::new(); }
        let n = self.snapshots.len();
        let mut metrics = Vec::with_capacity(n - 1);

        for i in 1..n {
            let prev_edges: HashSet<(usize, usize)> = self.snapshots[i-1].graph.edges()
                .into_iter().map(|e| (e.src, e.dst)).collect();
            let curr_edges: HashSet<(usize, usize)> = self.snapshots[i].graph.edges()
                .into_iter().map(|e| (e.src, e.dst)).collect();

            let born: usize = curr_edges.difference(&prev_edges).count();
            let died: usize = prev_edges.difference(&curr_edges).count();
            let survived: usize = curr_edges.intersection(&prev_edges).count();
            let dt = (self.snapshots[i].timestamp - self.snapshots[i-1].timestamp).max(1);

            metrics.push(EdgeFluxMetrics {
                t_start: self.snapshots[i-1].timestamp,
                t_end: self.snapshots[i].timestamp,
                edges_born: born,
                edges_died: died,
                edges_survived: survived,
                birth_rate: born as f64 / dt as f64,
                death_rate: died as f64 / dt as f64,
                flux_ratio: if survived > 0 { (born + died) as f64 / survived as f64 } else { f64::INFINITY },
            });
        }
        metrics
    }

    /// Node degree volatility: variance of degree across time for each node.
    pub fn degree_volatility(&self) -> Vec<f64> {
        let n = self.n_nodes;
        let mut degree_over_time: Vec<Vec<f64>> = vec![Vec::new(); n];

        for snap in &self.snapshots {
            let mut deg = vec![0.0f64; n];
            for e in snap.graph.edges() {
                if e.src < n { deg[e.src] += e.weight; }
                if e.dst < n { deg[e.dst] += e.weight; }
            }
            for i in 0..n {
                degree_over_time[i].push(deg[i]);
            }
        }

        degree_over_time.iter().map(|d| {
            if d.is_empty() { return 0.0; }
            let mean = d.iter().sum::<f64>() / d.len() as f64;
            d.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / d.len() as f64
        }).collect()
    }
}

// ── Structural change metrics ─────────────────────────────────────────────────

/// Structural change between two consecutive snapshots.
#[derive(Debug, Clone)]
pub struct StructuralChange {
    pub t_from: i64,
    pub t_to: i64,
    pub dt: i64,
    /// Graph edit distance (edge insertions + deletions + re-weighting cost).
    pub edit_distance: usize,
    /// Number of edges born.
    pub edges_born: usize,
    /// Number of edges died.
    pub edges_died: usize,
    /// Mean absolute weight change on surviving edges.
    pub mean_weight_change: f64,
    /// Jaccard similarity of edge sets.
    pub jaccard_similarity: f64,
    /// Normalized spectral distance (||L1 - L2||_F / ||L1||_F).
    pub spectral_distance: f64,
}

/// Compute structural change between two snapshots.
pub fn structural_change(a: &GraphSnapshot, b: &GraphSnapshot) -> StructuralChange {
    let a_edges: HashMap<(usize, usize), f64> = a.edge_map();
    let b_edges: HashMap<(usize, usize), f64> = b.edge_map();

    let a_set: HashSet<(usize, usize)> = a_edges.keys().copied().collect();
    let b_set: HashSet<(usize, usize)> = b_edges.keys().copied().collect();

    let born = b_set.difference(&a_set).count();
    let died = a_set.difference(&b_set).count();
    let common: HashSet<_> = a_set.intersection(&b_set).copied().collect();

    let mean_weight_change = if common.is_empty() {
        0.0
    } else {
        common.iter().map(|k| (a_edges[k] - b_edges[k]).abs()).sum::<f64>() / common.len() as f64
    };

    let union_size = a_set.union(&b_set).count();
    let intersection_size = common.len();
    let jaccard = if union_size == 0 { 1.0 } else { intersection_size as f64 / union_size as f64 };

    // Edit distance = born + died + (count of weight changes > 0.1)
    let weight_changes: usize = common.iter().filter(|k| (a_edges[k] - b_edges[k]).abs() > 0.1).count();
    let edit_distance = born + died + weight_changes;

    // Spectral distance: Frobenius norm of Laplacian difference
    let spectral_distance = laplacian_frobenius_distance(&a.graph, &b.graph);

    StructuralChange {
        t_from: a.timestamp,
        t_to: b.timestamp,
        dt: (b.timestamp - a.timestamp).max(1),
        edit_distance,
        edges_born: born,
        edges_died: died,
        mean_weight_change,
        jaccard_similarity: jaccard,
        spectral_distance,
    }
}

/// Frobenius distance between two graph Laplacians.
fn laplacian_frobenius_distance(a: &WeightedGraph, b: &WeightedGraph) -> f64 {
    let n = a.n.max(b.n);
    if n == 0 { return 0.0; }
    // Build degree maps
    let mut la = vec![vec![0.0f64; n]; n];
    let mut lb = vec![vec![0.0f64; n]; n];

    for e in a.edges() {
        if e.src < n && e.dst < n {
            la[e.src][e.dst] -= e.weight;
            la[e.dst][e.src] -= e.weight;
            la[e.src][e.src] += e.weight;
            la[e.dst][e.dst] += e.weight;
        }
    }
    for e in b.edges() {
        if e.src < n && e.dst < n {
            lb[e.src][e.dst] -= e.weight;
            lb[e.dst][e.src] -= e.weight;
            lb[e.src][e.src] += e.weight;
            lb[e.dst][e.dst] += e.weight;
        }
    }

    let mut frob_sq = 0.0f64;
    let mut norm_a_sq = 0.0f64;
    for i in 0..n {
        for j in 0..n {
            let diff = la[i][j] - lb[i][j];
            frob_sq += diff * diff;
            norm_a_sq += la[i][j] * la[i][j];
        }
    }
    if norm_a_sq < 1e-14 { return frob_sq.sqrt(); }
    frob_sq.sqrt() / norm_a_sq.sqrt()
}

// ── Persistent features ───────────────────────────────────────────────────────

/// An edge that persists across multiple time windows.
#[derive(Debug, Clone)]
pub struct PersistentEdge {
    pub src: usize,
    pub dst: usize,
    pub occurrence_count: usize,
    pub mean_weight: f64,
    pub persistence_ratio: f64,
}

/// Edge flux metrics for a time interval.
#[derive(Debug, Clone)]
pub struct EdgeFluxMetrics {
    pub t_start: i64,
    pub t_end: i64,
    pub edges_born: usize,
    pub edges_died: usize,
    pub edges_survived: usize,
    pub birth_rate: f64,
    pub death_rate: f64,
    pub flux_ratio: f64,
}

// ── Wormhole detection ────────────────────────────────────────────────────────

/// A wormhole event: sudden high-weight edge between normally distant nodes.
#[derive(Debug, Clone)]
pub struct WormholeEvent {
    pub timestamp: i64,
    pub src: usize,
    pub dst: usize,
    pub current_weight: f64,
    pub historical_mean: f64,
    pub historical_std: f64,
    pub z_score: f64,
    /// Shortest path distance in the historical graphs (normal graph distance).
    pub historical_distance: f64,
}

/// Detect wormhole edges in the current snapshot relative to historical snapshots.
fn detect_wormhole_edges(
    current: &GraphSnapshot,
    history: &[GraphSnapshot],
    z_threshold: f64,
) -> Vec<WormholeEvent> {
    // Build per-edge weight history
    let mut edge_history: HashMap<(usize, usize), Vec<f64>> = HashMap::new();
    for snap in history {
        for e in snap.graph.edges() {
            edge_history.entry((e.src, e.dst)).or_default().push(e.weight);
        }
    }

    let mut events = Vec::new();

    for current_edge in current.graph.edges() {
        let key = (current_edge.src, current_edge.dst);
        let hist = edge_history.get(&key);

        let (hist_mean, hist_std) = if let Some(h) = hist {
            if h.is_empty() { (0.0, 1.0) } else {
                let mean = h.iter().sum::<f64>() / h.len() as f64;
                let var = h.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / h.len() as f64;
                (mean, var.sqrt().max(1e-6))
            }
        } else {
            // Edge never appeared before: use mean of all historical edges as baseline
            let all_weights: Vec<f64> = history.iter()
                .flat_map(|s| s.graph.edges())
                .map(|e| e.weight)
                .collect();
            let mean = if all_weights.is_empty() { 0.0 } else {
                all_weights.iter().sum::<f64>() / all_weights.len() as f64
            };
            (mean, 1.0)
        };

        let z = (current_edge.weight - hist_mean) / hist_std;
        if z > z_threshold {
            // Compute historical graph distance between src and dst
            // Use the last historical snapshot for distance
            let hist_dist = if let Some(last) = history.last() {
                bfs_distance(&last.graph, current_edge.src, current_edge.dst)
            } else {
                f64::INFINITY
            };

            events.push(WormholeEvent {
                timestamp: current.timestamp,
                src: current_edge.src,
                dst: current_edge.dst,
                current_weight: current_edge.weight,
                historical_mean: hist_mean,
                historical_std: hist_std,
                z_score: z,
                historical_distance: hist_dist,
            });
        }
    }
    events
}

/// BFS to compute shortest hop distance between src and dst.
fn bfs_distance(graph: &WeightedGraph, src: usize, dst: usize) -> f64 {
    if src == dst { return 0.0; }
    let n = graph.n;
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();
    visited[src] = true;
    queue.push_back((src, 0usize));

    while let Some((u, d)) = queue.pop_front() {
        if u == dst { return d as f64; }
        for &(v, _) in &graph.adj[u] {
            if !visited[v] {
                visited[v] = true;
                queue.push_back((v, d + 1));
            }
        }
        for &(v, _) in &graph.radj[u] {
            if !visited[v] {
                visited[v] = true;
                queue.push_back((v, d + 1));
            }
        }
    }
    f64::INFINITY
}

// ── Graph Edit Distance ───────────────────────────────────────────────────────

/// Compute the weighted graph edit distance between two snapshots.
/// GED = cost of insertions + deletions + substitutions.
pub fn graph_edit_distance(a: &GraphSnapshot, b: &GraphSnapshot) -> f64 {
    let a_edges = a.edge_map();
    let b_edges = b.edge_map();

    let a_set: HashSet<(usize, usize)> = a_edges.keys().copied().collect();
    let b_set: HashSet<(usize, usize)> = b_edges.keys().copied().collect();

    // Insertion cost: edges in b not in a
    let insert_cost: f64 = b_set.difference(&a_set).map(|k| b_edges[k]).sum();
    // Deletion cost: edges in a not in b
    let delete_cost: f64 = a_set.difference(&b_set).map(|k| a_edges[k]).sum();
    // Substitution cost: weight change for common edges
    let sub_cost: f64 = a_set.intersection(&b_set)
        .map(|k| (a_edges[k] - b_edges[k]).abs())
        .sum();

    insert_cost + delete_cost + sub_cost
}

// ── Contagion path tracing ────────────────────────────────────────────────────

/// A contagion path from source to target.
#[derive(Debug, Clone)]
pub struct ContagionPath {
    pub source: usize,
    pub target: usize,
    pub path: Vec<usize>,
    /// Total contagion "speed" = sum of weights along path.
    pub path_weight: f64,
    /// Number of hops.
    pub hops: usize,
}

/// Find all shortest contagion paths from `source` to all other nodes.
/// Uses Dijkstra where higher weight = faster contagion (lower cost = 1/weight).
pub fn shortest_contagion_paths(
    graph: &WeightedGraph,
    source: usize,
) -> Vec<ContagionPath> {
    let n = graph.n;
    let mut dist = vec![f64::INFINITY; n];
    let mut prev: Vec<Option<usize>> = vec![None; n];
    dist[source] = 0.0;

    #[derive(Clone)]
    struct State { cost: f64, node: usize }
    impl PartialEq for State { fn eq(&self, o: &Self) -> bool { self.cost == o.cost } }
    impl Eq for State {}
    impl PartialOrd for State {
        fn partial_cmp(&self, o: &Self) -> Option<Ordering> { Some(self.cmp(o)) }
    }
    impl Ord for State {
        fn cmp(&self, o: &Self) -> Ordering {
            o.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
        }
    }

    let mut heap = BinaryHeap::new();
    heap.push(State { cost: 0.0, node: source });

    while let Some(State { cost, node: u }) = heap.pop() {
        if cost > dist[u] + 1e-12 { continue; }
        for &(v, w) in &graph.adj[u] {
            // Higher weight = faster contagion = lower cost
            let new_cost = cost + if w > 1e-10 { 1.0 / w } else { 1e10 };
            if new_cost < dist[v] {
                dist[v] = new_cost;
                prev[v] = Some(u);
                heap.push(State { cost: new_cost, node: v });
            }
        }
    }

    // Reconstruct paths
    let mut paths = Vec::new();
    for target in 0..n {
        if target == source || dist[target] == f64::INFINITY { continue; }

        let mut path = Vec::new();
        let mut cur = target;
        while let Some(p) = prev[cur] {
            path.push(cur);
            cur = p;
        }
        path.push(source);
        path.reverse();

        let path_weight: f64 = path.windows(2).map(|w| {
            graph.adj[w[0]].iter()
                .find(|&&(v, _)| v == w[1])
                .map(|&(_, wt)| wt)
                .unwrap_or(0.0)
        }).sum();

        paths.push(ContagionPath {
            source,
            target,
            hops: path.len() - 1,
            path_weight,
            path,
        });
    }
    paths
}

/// BFS-based contagion tracing: find all nodes reachable within `hops` hops
/// where edge weight >= `min_weight`.
pub fn contagion_reachability(
    graph: &WeightedGraph,
    source: usize,
    max_hops: usize,
    min_weight: f64,
) -> Vec<(usize, usize, f64)> {
    // Returns (target, hops, cumulative_weight)
    let n = graph.n;
    let mut visited = vec![false; n];
    let mut result = Vec::new();
    visited[source] = true;

    let mut queue: VecDeque<(usize, usize, f64)> = VecDeque::new();
    queue.push_back((source, 0, 0.0));

    while let Some((u, hops, cum_w)) = queue.pop_front() {
        if hops > 0 { result.push((u, hops, cum_w)); }
        if hops >= max_hops { continue; }

        for &(v, w) in &graph.adj[u] {
            if !visited[v] && w >= min_weight {
                visited[v] = true;
                queue.push_back((v, hops + 1, cum_w + w));
            }
        }
    }
    result
}

/// Identify critical contagion nodes: nodes whose removal would maximally
/// increase the shortest contagion path length.
pub fn critical_contagion_nodes(
    graph: &WeightedGraph,
    source: usize,
) -> Vec<(usize, f64)> {
    let n = graph.n;
    let base_paths = shortest_contagion_paths(graph, source);
    let base_total: f64 = base_paths.iter().map(|p| p.path_weight).sum::<f64>();

    (0..n).filter(|&v| v != source).map(|remove_node| {
        // Build graph without remove_node
        let filtered_edges: Vec<Edge> = graph.edges().into_iter()
            .filter(|e| e.src != remove_node && e.dst != remove_node)
            .collect();
        let reduced = WeightedGraph::from_edges(n, &filtered_edges);
        let new_paths = shortest_contagion_paths(&reduced, source);
        let new_total: f64 = new_paths.iter().map(|p| p.path_weight).sum::<f64>();
        let importance = if base_total > 1e-12 {
            (new_total - base_total) / base_total
        } else { 0.0 };
        (remove_node, importance)
    }).collect()
}

// ── Temporal motifs ───────────────────────────────────────────────────────────

/// Count temporal motifs: sequences of edges that form patterns across time.
/// This focuses on the "flash crash" motif: an edge that appears, strengthens,
/// then disappears within a short window.
#[derive(Debug, Clone)]
pub struct TemporalMotif {
    pub src: usize,
    pub dst: usize,
    pub t_born: i64,
    pub t_peak: i64,
    pub t_died: i64,
    pub peak_weight: f64,
    pub motif_type: MotifType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MotifType {
    FlashEdge,       // born, peaked, died within 3 snapshots
    PersistentEdge,  // present throughout
    StepChange,      // sudden sustained weight increase
    StepDecay,       // sudden sustained weight decrease
}

/// Detect temporal motifs across a sequence of snapshots.
pub fn detect_temporal_motifs(temporal: &TemporalGraph) -> Vec<TemporalMotif> {
    let n_snaps = temporal.snapshots.len();
    if n_snaps < 3 { return Vec::new(); }

    let mut all_edge_histories: HashMap<(usize, usize), Vec<(i64, f64)>> = HashMap::new();

    for snap in &temporal.snapshots {
        for e in snap.graph.edges() {
            all_edge_histories.entry((e.src, e.dst))
                .or_default()
                .push((snap.timestamp, e.weight));
        }
    }

    let mut motifs = Vec::new();

    for ((src, dst), history) in &all_edge_histories {
        if history.len() < 2 { continue; }

        let max_w = history.iter().map(|(_, w)| *w).fold(f64::NEG_INFINITY, f64::max);
        let peak_idx = history.iter().position(|(_, w)| *w == max_w).unwrap_or(0);
        let t_peak = history[peak_idx].0;

        let t_born = history.first().map(|(t, _)| *t).unwrap_or(0);
        let t_died = history.last().map(|(t, _)| *t).unwrap_or(0);

        // Classify motif
        let motif_type = if history.len() <= 3
            && peak_idx > 0 && peak_idx < history.len() - 1
        {
            MotifType::FlashEdge
        } else if history.len() == n_snaps {
            MotifType::PersistentEdge
        } else {
            // Check for step change
            let first_half: Vec<f64> = history[..history.len()/2].iter().map(|(_, w)| *w).collect();
            let second_half: Vec<f64> = history[history.len()/2..].iter().map(|(_, w)| *w).collect();
            let mean1 = first_half.iter().sum::<f64>() / first_half.len() as f64;
            let mean2 = second_half.iter().sum::<f64>() / second_half.len() as f64;
            if mean2 > mean1 * 1.5 { MotifType::StepChange }
            else if mean1 > mean2 * 1.5 { MotifType::StepDecay }
            else { MotifType::PersistentEdge }
        };

        motifs.push(TemporalMotif {
            src: *src,
            dst: *dst,
            t_born,
            t_peak,
            t_died,
            peak_weight: max_w,
            motif_type,
        });
    }
    motifs
}

// ── Structural velocity (d(topology)/dt) ─────────────────────────────────────

/// Compute the velocity of structural change as a scalar time series.
/// Returns (timestamp, velocity) pairs.
pub fn structural_velocity(temporal: &TemporalGraph) -> Vec<(i64, f64)> {
    if temporal.snapshots.len() < 2 { return Vec::new(); }

    temporal.snapshots.windows(2).map(|w| {
        let sc = structural_change(&w[0], &w[1]);
        let velocity = sc.edit_distance as f64 / sc.dt.max(1) as f64;
        (w[1].timestamp, velocity)
    }).collect()
}

/// Detect anomalous structural velocity events (velocity > mean + k*std).
pub fn detect_velocity_spikes(
    temporal: &TemporalGraph,
    sigma_threshold: f64,
) -> Vec<(i64, f64)> {
    let velocities = structural_velocity(temporal);
    if velocities.len() < 3 { return Vec::new(); }

    let vals: Vec<f64> = velocities.iter().map(|(_, v)| *v).collect();
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let var = vals.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / vals.len() as f64;
    let std = var.sqrt();

    velocities.into_iter()
        .filter(|(_, v)| *v > mean + sigma_threshold * std)
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_triangle(ts: i64) -> GraphSnapshot {
        let mut g = WeightedGraph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(0, 2, 1.0);
        GraphSnapshot::new(ts, g)
    }

    fn make_line(ts: i64, w: f64) -> GraphSnapshot {
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, w);
        g.add_edge(1, 2, w);
        g.add_edge(2, 3, w);
        GraphSnapshot::new(ts, g)
    }

    #[test]
    fn test_temporal_graph_push_sorted() {
        let mut tg = TemporalGraph::new(3);
        tg.push(make_triangle(200));
        tg.push(make_triangle(100));
        assert_eq!(tg.snapshots[0].timestamp, 100);
        assert_eq!(tg.snapshots[1].timestamp, 200);
    }

    #[test]
    fn test_structural_change_same_graph() {
        let a = make_triangle(0);
        let b = make_triangle(1);
        let sc = structural_change(&a, &b);
        assert_eq!(sc.edges_born, 0);
        assert_eq!(sc.edges_died, 0);
        assert!((sc.jaccard_similarity - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_structural_change_different() {
        let a = make_triangle(0);
        let b = make_line(1, 1.0);
        let sc = structural_change(&a, &b);
        assert!(sc.edges_born > 0 || sc.edges_died > 0);
    }

    #[test]
    fn test_persistent_edges() {
        let mut tg = TemporalGraph::new(3);
        tg.push(make_triangle(0));
        tg.push(make_triangle(1));
        tg.push(make_triangle(2));
        let pe = tg.persistent_edges(3);
        assert_eq!(pe.len(), 3); // all 3 edges persist
    }

    #[test]
    fn test_edge_flux() {
        let mut tg = TemporalGraph::new(4);
        tg.push(make_line(0, 1.0));
        tg.push(make_line(1, 1.5));
        let flux = tg.edge_birth_death_rates();
        assert_eq!(flux.len(), 1);
        assert_eq!(flux[0].edges_born, 0); // same topology
        assert_eq!(flux[0].edges_died, 0);
    }

    #[test]
    fn test_contagion_paths() {
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 2.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(2, 3, 2.0);
        let paths = shortest_contagion_paths(&g, 0);
        assert!(paths.iter().any(|p| p.target == 3));
    }

    #[test]
    fn test_bfs_distance() {
        let mut g = WeightedGraph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        assert_eq!(bfs_distance(&g, 0, 3), 3.0);
        assert_eq!(bfs_distance(&g, 0, 0), 0.0);
    }

    #[test]
    fn test_temporal_motifs() {
        let mut tg = TemporalGraph::new(3);
        tg.push(make_triangle(0));
        tg.push(make_triangle(1));
        tg.push(make_triangle(2));
        let motifs = detect_temporal_motifs(&tg);
        assert!(motifs.iter().any(|m| m.motif_type == MotifType::PersistentEdge));
    }

    #[test]
    fn test_degree_volatility() {
        let mut tg = TemporalGraph::new(3);
        tg.push(make_triangle(0));
        tg.push(make_triangle(1));
        let vol = tg.degree_volatility();
        assert_eq!(vol.len(), 3);
        // Constant graph = zero volatility
        for v in &vol { assert!(v.abs() < 1e-10); }
    }
}
