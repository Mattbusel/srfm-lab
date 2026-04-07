/// Dynamic correlation network for a crypto asset universe.
///
/// Maintains per-pair rolling Pearson correlations using the Welford online
/// covariance algorithm (window = 30 bars). Provides cluster detection,
/// hub-asset identification, and regime classification.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Default rolling window in bars.
pub const DEFAULT_WINDOW: usize = 30;

/// Average |r| above this threshold => HIGH regime.
const REGIME_HIGH: f64 = 0.8;
/// Average |r| below this threshold => LOW regime.
const REGIME_LOW: f64 = 0.5;

// ── CorrelationRegime ─────────────────────────────────────────────────────────

/// Classification of the prevailing correlation regime across the universe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CorrelationRegime {
    /// Average |r| > 0.8 -- most assets moving together (risk-on or risk-off panic).
    High,
    /// 0.5 <= average |r| <= 0.8 -- moderate co-movement.
    Medium,
    /// Average |r| < 0.5 -- assets trading largely independently.
    Low,
}

impl std::fmt::Display for CorrelationRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CorrelationRegime::High => write!(f, "HIGH"),
            CorrelationRegime::Medium => write!(f, "MEDIUM"),
            CorrelationRegime::Low => write!(f, "LOW"),
        }
    }
}

// ── CorrelationEdge ───────────────────────────────────────────────────────────

/// A weighted, undirected edge in the correlation graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationEdge {
    /// First asset (lexicographically smaller for canonical ordering).
    pub asset_a: String,
    /// Second asset.
    pub asset_b: String,
    /// Pearson correlation coefficient in [-1, 1].
    pub pearson_r: f64,
    /// Window size used for this computation.
    pub rolling_window_bars: usize,
    /// Bar index at which this edge was last updated.
    pub last_updated: u64,
}

// ── WelfordPair ───────────────────────────────────────────────────────────────

/// Per-pair Welford online covariance state with a fixed-size ring buffer.
#[derive(Debug, Clone)]
struct WelfordPair {
    /// Ring buffer for series X.
    buf_x: Vec<f64>,
    /// Ring buffer for series Y.
    buf_y: Vec<f64>,
    /// Write cursor (next slot to overwrite).
    head: usize,
    /// Number of valid elements currently in the buffer.
    count: usize,
    /// Rolling window capacity.
    window: usize,
}

impl WelfordPair {
    fn new(window: usize) -> Self {
        WelfordPair {
            buf_x: vec![0.0; window],
            buf_y: vec![0.0; window],
            head: 0,
            count: 0,
            window,
        }
    }

    /// Push a new (x, y) observation.
    fn push(&mut self, x: f64, y: f64) {
        self.buf_x[self.head] = x;
        self.buf_y[self.head] = y;
        self.head = (self.head + 1) % self.window;
        if self.count < self.window {
            self.count += 1;
        }
    }

    /// Compute Pearson r from the current buffer contents.
    /// Returns None if fewer than 3 observations are available.
    fn pearson_r(&self) -> Option<f64> {
        let n = self.count;
        if n < 3 {
            return None;
        }

        let mut mean_x = 0.0_f64;
        let mut mean_y = 0.0_f64;

        // Iterate over valid entries.
        for i in 0..n {
            mean_x += self.buf_x[i];
            mean_y += self.buf_y[i];
        }
        mean_x /= n as f64;
        mean_y /= n as f64;

        let mut cov = 0.0_f64;
        let mut var_x = 0.0_f64;
        let mut var_y = 0.0_f64;

        for i in 0..n {
            let dx = self.buf_x[i] - mean_x;
            let dy = self.buf_y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denom = (var_x * var_y).sqrt();
        if denom < 1e-14 {
            return None;
        }
        let r = (cov / denom).clamp(-1.0, 1.0);
        Some(r)
    }

    /// True if at least 3 observations have been pushed.
    fn is_ready(&self) -> bool {
        self.count >= 3
    }
}

// ── SingleSeries ─────────────────────────────────────────────────────────────

/// Per-symbol return buffer (ring buffer, window=30).
#[derive(Debug, Clone)]
struct SingleSeries {
    buf: Vec<f64>,
    head: usize,
    count: usize,
    window: usize,
}

impl SingleSeries {
    fn new(window: usize) -> Self {
        SingleSeries { buf: vec![0.0; window], head: 0, count: 0, window }
    }

    fn push(&mut self, v: f64) {
        self.buf[self.head] = v;
        self.head = (self.head + 1) % self.window;
        if self.count < self.window {
            self.count += 1;
        }
    }

    fn latest(&self) -> Option<f64> {
        if self.count == 0 {
            return None;
        }
        // The most-recently written slot is (head - 1 + window) % window.
        let slot = (self.head + self.window - 1) % self.window;
        Some(self.buf[slot])
    }
}

// ── CorrelationGraph ──────────────────────────────────────────────────────────

/// Dynamic correlation network for a universe of crypto assets.
///
/// Internally maintains a per-symbol return ring buffer (for new-pair
/// bootstrap) and a per-pair `WelfordPair` accumulator.
pub struct CorrelationGraph {
    /// Rolling window in bars.
    window: usize,

    /// Monotonically increasing bar counter -- advanced on each `update_returns` call.
    bar_idx: u64,

    /// Per-symbol latest return buffer.
    symbol_returns: HashMap<String, SingleSeries>,

    /// Per-pair covariance state. Key: (sym_a, sym_b) with a < b lexicographically.
    pair_state: HashMap<(String, String), WelfordPair>,
}

impl CorrelationGraph {
    /// Create a new graph with the given rolling window (bars).
    pub fn new(window: usize) -> Self {
        CorrelationGraph {
            window,
            bar_idx: 0,
            symbol_returns: HashMap::new(),
            pair_state: HashMap::new(),
        }
    }

    /// Create with the default 30-bar window.
    pub fn default_window() -> Self {
        Self::new(DEFAULT_WINDOW)
    }

    // ── Update ────────────────────────────────────────────────────────────────

    /// Record a new return observation for `symbol`.
    ///
    /// Updates all existing pair states that include this symbol.
    /// Newly discovered symbols are added automatically.
    pub fn update_returns(&mut self, symbol: &str, ret: f64) {
        // Ensure the symbol's own buffer exists.
        self.symbol_returns
            .entry(symbol.to_string())
            .or_insert_with(|| SingleSeries::new(self.window))
            .push(ret);

        // Snapshot the current latest returns for all *other* symbols.
        let peers: Vec<(String, f64)> = self
            .symbol_returns
            .iter()
            .filter(|(k, _)| k.as_str() != symbol)
            .filter_map(|(k, s)| s.latest().map(|v| (k.clone(), v)))
            .collect();

        // Update (or create) each pair state.
        for (peer, peer_ret) in peers {
            let key = canonical_key(symbol, &peer);
            let wp = self
                .pair_state
                .entry(key)
                .or_insert_with(|| WelfordPair::new(self.window));

            // Canonical ordering: key.0 <= key.1 lexicographically.
            if symbol <= peer.as_str() {
                wp.push(ret, peer_ret);
            } else {
                wp.push(peer_ret, ret);
            }
        }

        self.bar_idx += 1;
    }

    // ── Queries ───────────────────────────────────────────────────────────────

    /// Return the current Pearson correlation between two assets.
    /// Returns `None` if fewer than 3 joint observations exist.
    pub fn get_correlation(&self, a: &str, b: &str) -> Option<f64> {
        if a == b {
            return Some(1.0);
        }
        let key = canonical_key(a, b);
        self.pair_state.get(&key)?.pearson_r()
    }

    /// Return the full pairwise correlation matrix as a flat HashMap.
    pub fn get_full_matrix(&self) -> HashMap<(String, String), f64> {
        let mut out = HashMap::new();
        for ((a, b), wp) in &self.pair_state {
            if let Some(r) = wp.pearson_r() {
                out.insert((a.clone(), b.clone()), r);
                out.insert((b.clone(), a.clone()), r);
            }
        }
        // Self-correlations.
        for sym in self.symbol_returns.keys() {
            out.insert((sym.clone(), sym.clone()), 1.0);
        }
        out
    }

    /// Get all edges with valid Pearson r values.
    pub fn get_edges(&self) -> Vec<CorrelationEdge> {
        let mut edges = Vec::new();
        for ((a, b), wp) in &self.pair_state {
            if let Some(r) = wp.pearson_r() {
                edges.push(CorrelationEdge {
                    asset_a: a.clone(),
                    asset_b: b.clone(),
                    pearson_r: r,
                    rolling_window_bars: self.window,
                    last_updated: self.bar_idx,
                });
            }
        }
        // Sort for deterministic output.
        edges.sort_by(|a, b| {
            a.asset_a
                .cmp(&b.asset_a)
                .then(a.asset_b.cmp(&b.asset_b))
        });
        edges
    }

    // ── Cluster Detection ─────────────────────────────────────────────────────

    /// Greedy cluster detection: instruments with |corr| > `threshold` are in
    /// the same cluster.
    ///
    /// Algorithm: union-find over all pairs whose |r| > threshold.
    pub fn find_clusters(&self, threshold: f64) -> Vec<Vec<String>> {
        let symbols: Vec<String> = {
            let mut v: Vec<String> = self.symbol_returns.keys().cloned().collect();
            v.sort();
            v
        };
        let n = symbols.len();
        if n == 0 {
            return vec![];
        }

        // Union-Find with path compression.
        let mut parent: Vec<usize> = (0..n).collect();

        fn find(parent: &mut Vec<usize>, i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find(parent, parent[i]);
            }
            parent[i]
        }

        fn union(parent: &mut Vec<usize>, i: usize, j: usize) {
            let ri = find(parent, i);
            let rj = find(parent, j);
            if ri != rj {
                parent[ri] = rj;
            }
        }

        for i in 0..n {
            for j in (i + 1)..n {
                if let Some(r) = self.get_correlation(&symbols[i], &symbols[j]) {
                    if r.abs() > threshold {
                        union(&mut parent, i, j);
                    }
                }
            }
        }

        // Group symbols by their root.
        let mut clusters: HashMap<usize, Vec<String>> = HashMap::new();
        for (idx, sym) in symbols.iter().enumerate() {
            let root = find(&mut parent, idx);
            clusters.entry(root).or_default().push(sym.clone());
        }

        let mut result: Vec<Vec<String>> = clusters.into_values().collect();
        // Sort clusters by size (descending), then internally by name.
        result.sort_by(|a, b| b.len().cmp(&a.len()).then(a[0].cmp(&b[0])));
        for cluster in &mut result {
            cluster.sort();
        }
        result
    }

    // ── Hub Assets ────────────────────────────────────────────────────────────

    /// Return assets sorted by their average absolute correlation to all other
    /// assets -- descending. These are the most "systemic" assets.
    ///
    /// Returns (symbol, avg_abs_corr) pairs.
    pub fn get_hub_assets(&self) -> Vec<(String, f64)> {
        let symbols: Vec<String> = {
            let mut v: Vec<String> = self.symbol_returns.keys().cloned().collect();
            v.sort();
            v
        };
        let n = symbols.len();
        if n < 2 {
            return symbols.into_iter().map(|s| (s, 0.0)).collect();
        }

        let mut scores: Vec<(String, f64)> = symbols
            .iter()
            .map(|sym| {
                let sum: f64 = symbols
                    .iter()
                    .filter(|other| *other != sym)
                    .filter_map(|other| self.get_correlation(sym, other))
                    .map(|r| r.abs())
                    .sum();
                let count = symbols
                    .iter()
                    .filter(|other| *other != sym)
                    .filter(|other| self.get_correlation(sym, other).is_some())
                    .count();
                let avg = if count > 0 { sum / count as f64 } else { 0.0 };
                (sym.clone(), avg)
            })
            .collect();

        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    // ── Regime ────────────────────────────────────────────────────────────────

    /// Classify the current correlation regime across the universe.
    pub fn correlation_regime(&self) -> CorrelationRegime {
        let values: Vec<f64> = self
            .pair_state
            .values()
            .filter_map(|wp| wp.pearson_r())
            .map(|r| r.abs())
            .collect();

        if values.is_empty() {
            return CorrelationRegime::Low;
        }

        let avg = values.iter().sum::<f64>() / values.len() as f64;

        if avg > REGIME_HIGH {
            CorrelationRegime::High
        } else if avg >= REGIME_LOW {
            CorrelationRegime::Medium
        } else {
            CorrelationRegime::Low
        }
    }

    // ── Diagnostics ───────────────────────────────────────────────────────────

    /// Return the number of symbols currently tracked.
    pub fn num_symbols(&self) -> usize {
        self.symbol_returns.len()
    }

    /// Return the number of pairs with at least 3 joint observations.
    pub fn num_active_pairs(&self) -> usize {
        self.pair_state.values().filter(|wp| wp.is_ready()).count()
    }

    /// Average absolute correlation across all valid pairs (universe-wide
    /// correlation level).
    pub fn average_abs_correlation(&self) -> f64 {
        let vals: Vec<f64> = self
            .pair_state
            .values()
            .filter_map(|wp| wp.pearson_r())
            .map(|r| r.abs())
            .collect();
        if vals.is_empty() {
            return 0.0;
        }
        vals.iter().sum::<f64>() / vals.len() as f64
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Produce a canonical (lexicographically ordered) key for a pair.
fn canonical_key(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn push_series(graph: &mut CorrelationGraph, symbols: &[&str], returns: &[&[f64]]) {
        let n_bars = returns[0].len();
        for bar in 0..n_bars {
            for (sym_idx, sym) in symbols.iter().enumerate() {
                graph.update_returns(sym, returns[sym_idx][bar]);
            }
        }
    }

    #[test]
    fn test_perfect_correlation() {
        let mut g = CorrelationGraph::new(30);
        let series: Vec<f64> = (0..40).map(|i| (i as f64 * 0.1).sin()).collect();
        push_series(&mut g, &["A", "B"], &[&series, &series]);
        let r = g.get_correlation("A", "B").unwrap();
        assert!((r - 1.0).abs() < 1e-9, "expected 1.0, got {}", r);
    }

    #[test]
    fn test_perfect_negative_correlation() {
        let mut g = CorrelationGraph::new(30);
        let x: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|v| -*v).collect();
        push_series(&mut g, &["A", "B"], &[&x, &y]);
        let r = g.get_correlation("A", "B").unwrap();
        assert!((r + 1.0).abs() < 1e-9, "expected -1.0, got {}", r);
    }

    #[test]
    fn test_self_correlation() {
        let mut g = CorrelationGraph::new(30);
        let s: Vec<f64> = (0..20).map(|i| i as f64).collect();
        for v in &s {
            g.update_returns("X", *v);
        }
        assert_eq!(g.get_correlation("X", "X"), Some(1.0));
    }

    #[test]
    fn test_unknown_pair_returns_none() {
        let g = CorrelationGraph::new(30);
        assert_eq!(g.get_correlation("A", "B"), None);
    }

    #[test]
    fn test_hub_assets_ordering() {
        // BTC highly correlated with ETH and SOL; USDT uncorrelated with all.
        let mut g = CorrelationGraph::new(30);
        let btc: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2).sin()).collect();
        let eth: Vec<f64> = btc.iter().map(|v| v + 0.01).collect();
        let sol: Vec<f64> = btc.iter().map(|v| v - 0.01).collect();
        // USDT: random-ish, uncorrelated.
        let usdt: Vec<f64> = (0..50).map(|i| ((i * 7 + 3) as f64 % 2.0) - 1.0).collect();
        push_series(&mut g, &["BTC", "ETH", "SOL", "USDT"], &[&btc, &eth, &sol, &usdt]);

        let hubs = g.get_hub_assets();
        // USDT should not be first (BTC, ETH, or SOL should be higher).
        assert_ne!(hubs[0].0, "USDT", "USDT should not be the hub");
    }

    #[test]
    fn test_find_clusters_all_correlated() {
        let mut g = CorrelationGraph::new(30);
        let btc: Vec<f64> = (0..50).map(|i| (i as f64 * 0.15).sin()).collect();
        let eth: Vec<f64> = btc.clone();
        let sol: Vec<f64> = btc.clone();
        push_series(&mut g, &["BTC", "ETH", "SOL"], &[&btc, &eth, &sol]);
        let clusters = g.find_clusters(0.5);
        // All three should be in one cluster.
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].len(), 3);
    }

    #[test]
    fn test_regime_high() {
        let mut g = CorrelationGraph::new(30);
        let s: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        // Near-identical series => r > 0.8 for all pairs.
        let s2: Vec<f64> = s.iter().map(|v| v + 0.0001).collect();
        let s3: Vec<f64> = s.iter().map(|v| v + 0.0002).collect();
        push_series(&mut g, &["A", "B", "C"], &[&s, &s2, &s3]);
        assert_eq!(g.correlation_regime(), CorrelationRegime::High);
    }

    #[test]
    fn test_full_matrix_symmetric() {
        let mut g = CorrelationGraph::new(30);
        let s: Vec<f64> = (0..40).map(|i| (i as f64 * 0.3).cos()).collect();
        let t: Vec<f64> = (0..40).map(|i| (i as f64 * 0.5).sin()).collect();
        push_series(&mut g, &["X", "Y"], &[&s, &t]);
        let m = g.get_full_matrix();
        let xy = m.get(&("X".to_string(), "Y".to_string())).copied().unwrap_or(f64::NAN);
        let yx = m.get(&("Y".to_string(), "X".to_string())).copied().unwrap_or(f64::NAN);
        assert!((xy - yx).abs() < 1e-12, "matrix not symmetric: {} vs {}", xy, yx);
    }

    #[test]
    fn test_canonical_key_ordering() {
        assert_eq!(canonical_key("ETH", "BTC"), ("BTC".to_string(), "ETH".to_string()));
        assert_eq!(canonical_key("BTC", "ETH"), ("BTC".to_string(), "ETH".to_string()));
    }
}
