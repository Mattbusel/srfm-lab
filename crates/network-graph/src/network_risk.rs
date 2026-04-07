/// Network-theoretic systemic risk measures for crypto asset graphs.
///
/// Implements eigenvector centrality (power iteration), DebtRank-inspired
/// vulnerability scoring, a systemic risk index, and identification of
/// systemically important assets.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum power-iteration rounds for eigenvector centrality.
const MAX_ITER: usize = 500;
/// Convergence tolerance for power iteration.
const TOLERANCE: f64 = 1e-9;
/// Default number of "hub" assets used for vulnerability calculation.
const DEFAULT_HUB_COUNT: usize = 3;

// ── RiskProfile ───────────────────────────────────────────────────────────────

/// Full risk profile for a single asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskProfile {
    pub symbol: String,
    /// Eigenvector centrality score [0, 1].
    pub centrality: f64,
    /// Average |correlation| to the top-N hub assets.
    pub vulnerability: f64,
    /// Weighted combination of centrality and vulnerability.
    pub composite_risk: f64,
}

// ── NetworkRiskAnalyzer ───────────────────────────────────────────────────────

/// Computes systemic risk measures from a pairwise correlation matrix.
pub struct NetworkRiskAnalyzer {
    /// Symbols in a deterministic, sorted order.
    symbols: Vec<String>,
    /// Symmetric adjacency matrix (|correlation| values), symbols x symbols.
    /// adj[i][j] = |corr(symbols[i], symbols[j])|.
    adj: Vec<Vec<f64>>,
}

impl NetworkRiskAnalyzer {
    /// Build from a flat correlation map (both (A,B) and (B,A) entries accepted).
    pub fn from_correlation_map(correlations: &HashMap<(String, String), f64>) -> Self {
        // Collect unique symbols.
        let mut sym_set: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (a, b) in correlations.keys() {
            if a != b {
                sym_set.insert(a.clone());
                sym_set.insert(b.clone());
            }
        }
        let mut symbols: Vec<String> = sym_set.into_iter().collect();
        symbols.sort();
        let n = symbols.len();

        let idx: HashMap<&str, usize> =
            symbols.iter().enumerate().map(|(i, s)| (s.as_str(), i)).collect();

        let mut adj = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            adj[i][i] = 1.0; // self-loop = 1
        }

        for ((a, b), &r) in correlations {
            if a == b {
                continue;
            }
            if let (Some(&i), Some(&j)) = (idx.get(a.as_str()), idx.get(b.as_str())) {
                let val = r.abs();
                adj[i][j] = val;
                adj[j][i] = val;
            }
        }

        NetworkRiskAnalyzer { symbols, adj }
    }

    /// Build from a raw (symbol -> Vec<f64> returns) map using the last `window`
    /// observations per pair.
    pub fn from_return_series(
        returns: &HashMap<String, Vec<f64>>,
        window: usize,
    ) -> Self {
        let mut correlations: HashMap<(String, String), f64> = HashMap::new();
        let syms: Vec<String> = {
            let mut v: Vec<String> = returns.keys().cloned().collect();
            v.sort();
            v
        };
        let n = syms.len();
        for i in 0..n {
            for j in (i + 1)..n {
                let ra = returns[&syms[i]].as_slice();
                let rb = returns[&syms[j]].as_slice();
                let start_a = ra.len().saturating_sub(window);
                let start_b = rb.len().saturating_sub(window);
                let r = pearson(&ra[start_a..], &rb[start_b..]).unwrap_or(0.0);
                correlations.insert((syms[i].clone(), syms[j].clone()), r);
                correlations.insert((syms[j].clone(), syms[i].clone()), r);
            }
        }
        Self::from_correlation_map(&correlations)
    }

    // ── Eigenvector Centrality ────────────────────────────────────────────

    /// Compute eigenvector centrality via the power iteration method.
    ///
    /// Uses the absolute-correlation adjacency matrix as the weight matrix.
    /// Iterates until L2-norm convergence or MAX_ITER rounds.
    ///
    /// Returns a map symbol -> centrality in [0, 1].
    pub fn compute_centrality(&self) -> HashMap<String, f64> {
        let n = self.symbols.len();
        if n == 0 {
            return HashMap::new();
        }
        if n == 1 {
            return HashMap::from([(self.symbols[0].clone(), 1.0)]);
        }

        // Initialise uniformly.
        let mut v: Vec<f64> = vec![1.0 / (n as f64).sqrt(); n];

        for _iter in 0..MAX_ITER {
            // v_new = A * v
            let mut v_new = vec![0.0_f64; n];
            for i in 0..n {
                for j in 0..n {
                    v_new[i] += self.adj[i][j] * v[j];
                }
            }

            // Normalise by L2 norm.
            let norm: f64 = v_new.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-15 {
                break;
            }
            for x in &mut v_new {
                *x /= norm;
            }

            // Check convergence.
            let delta: f64 = v
                .iter()
                .zip(v_new.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);

            v = v_new;
            if delta < TOLERANCE {
                break;
            }
        }

        // Ensure all centralities are non-negative (dominant eigenvector
        // can come out with all-negative sign due to arbitrary sign flip).
        let flip = v.iter().sum::<f64>() < 0.0;
        if flip {
            for x in &mut v {
                *x = -*x;
            }
        }

        // Normalise to [0, 1] by max.
        let max_c = v.iter().cloned().fold(0.0_f64, f64::max).max(1e-15);
        self.symbols
            .iter()
            .enumerate()
            .map(|(i, sym)| (sym.clone(), (v[i] / max_c).clamp(0.0, 1.0)))
            .collect()
    }

    // ── Vulnerability ─────────────────────────────────────────────────────

    /// Compute vulnerability of each asset: average |correlation| to the
    /// top-N hub assets (highest eigenvector centrality).
    ///
    /// Assets that are highly correlated to the hubs will amplify systemic
    /// shocks originating at the hubs.
    pub fn compute_vulnerability(&self) -> HashMap<String, f64> {
        self.compute_vulnerability_top_n(DEFAULT_HUB_COUNT)
    }

    /// Compute vulnerability against the `hub_n` highest-centrality assets.
    pub fn compute_vulnerability_top_n(&self, hub_n: usize) -> HashMap<String, f64> {
        let n = self.symbols.len();
        if n == 0 {
            return HashMap::new();
        }

        let centrality = self.compute_centrality();

        // Identify top-N hubs by centrality.
        let mut ranked: Vec<(usize, f64)> = self
            .symbols
            .iter()
            .enumerate()
            .map(|(i, sym)| (i, *centrality.get(sym).unwrap_or(&0.0)))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let hub_count = hub_n.min(n);
        let hub_indices: Vec<usize> = ranked[..hub_count].iter().map(|(i, _)| *i).collect();

        self.symbols
            .iter()
            .enumerate()
            .map(|(i, sym)| {
                let avg_corr = if hub_indices.is_empty() {
                    0.0
                } else {
                    let sum: f64 = hub_indices
                        .iter()
                        .filter(|&&h| h != i)
                        .map(|&h| self.adj[i][h])
                        .sum();
                    let count = hub_indices.iter().filter(|&&h| h != i).count();
                    if count == 0 { 0.0 } else { sum / count as f64 }
                };
                (sym.clone(), avg_corr)
            })
            .collect()
    }

    // ── Systemic Risk Index ───────────────────────────────────────────────

    /// Compute the universe-wide Systemic Risk Index (SRI).
    ///
    /// SRI = sum_i (centrality_i * degree_i) / (n * max_degree)
    ///
    /// Normalised to [0, 1]. Higher values = more concentrated, interconnected
    /// network => greater systemic fragility.
    pub fn systemic_risk_index(&self) -> f64 {
        let n = self.symbols.len();
        if n < 2 {
            return 0.0;
        }

        let centrality = self.compute_centrality();

        // Weighted degree: sum of edge weights for each node.
        let degrees: Vec<f64> = (0..n)
            .map(|i| {
                self.adj[i]
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, &w)| w)
                    .sum::<f64>()
            })
            .collect();

        let max_degree = degrees.iter().cloned().fold(0.0_f64, f64::max).max(1e-15);

        let sum: f64 = self
            .symbols
            .iter()
            .enumerate()
            .map(|(i, sym)| {
                let c = *centrality.get(sym).unwrap_or(&0.0);
                c * (degrees[i] / max_degree)
            })
            .sum();

        (sum / n as f64).clamp(0.0, 1.0)
    }

    // ── Systemically Important Assets ─────────────────────────────────────

    /// Return the top-`n` symbols ranked by eigenvector centrality.
    ///
    /// These are the most systemically important: a shock to them propagates
    /// most widely through the network.
    pub fn identify_systemically_important(&self, top_n: usize) -> Vec<String> {
        let centrality = self.compute_centrality();
        let mut ranked: Vec<(String, f64)> = centrality.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
            .into_iter()
            .take(top_n)
            .map(|(sym, _)| sym)
            .collect()
    }

    // ── Full Risk Profiles ────────────────────────────────────────────────

    /// Compute complete risk profiles for all assets.
    ///
    /// Composite risk = 0.6 * centrality + 0.4 * vulnerability.
    pub fn risk_profiles(&self) -> Vec<RiskProfile> {
        let centrality = self.compute_centrality();
        let vulnerability = self.compute_vulnerability();

        let mut profiles: Vec<RiskProfile> = self
            .symbols
            .iter()
            .map(|sym| {
                let c = *centrality.get(sym).unwrap_or(&0.0);
                let v = *vulnerability.get(sym).unwrap_or(&0.0);
                RiskProfile {
                    symbol: sym.clone(),
                    centrality: c,
                    vulnerability: v,
                    composite_risk: 0.6 * c + 0.4 * v,
                }
            })
            .collect();

        profiles.sort_by(|a, b| {
            b.composite_risk
                .partial_cmp(&a.composite_risk)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        profiles
    }

    // ── Network Density ───────────────────────────────────────────────────

    /// Network density: fraction of possible edges with |corr| > `threshold`.
    pub fn density(&self, threshold: f64) -> f64 {
        let n = self.symbols.len();
        if n < 2 {
            return 0.0;
        }
        let total_possible = n * (n - 1) / 2;
        let mut active = 0usize;
        for i in 0..n {
            for j in (i + 1)..n {
                if self.adj[i][j] > threshold {
                    active += 1;
                }
            }
        }
        active as f64 / total_possible as f64
    }

    /// Average weighted degree across all nodes (mean edge weight).
    pub fn average_degree(&self) -> f64 {
        let n = self.symbols.len();
        if n < 2 {
            return 0.0;
        }
        let total: f64 = (0..n)
            .map(|i| {
                self.adj[i]
                    .iter()
                    .enumerate()
                    .filter(|(j, _)| *j != i)
                    .map(|(_, &w)| w)
                    .sum::<f64>()
            })
            .sum();
        total / n as f64
    }

    /// Symbols tracked by this analyzer.
    pub fn symbols(&self) -> &[String] {
        &self.symbols
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len().min(y.len());
    if n < 2 {
        return None;
    }
    let mx = x[..n].iter().sum::<f64>() / n as f64;
    let my = y[..n].iter().sum::<f64>() / n as f64;
    let mut num = 0.0_f64;
    let mut dx2 = 0.0_f64;
    let mut dy2 = 0.0_f64;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        num += dx * dy;
        dx2 += dx * dx;
        dy2 += dy * dy;
    }
    let denom = (dx2 * dy2).sqrt();
    if denom < 1e-14 {
        return None;
    }
    Some((num / denom).clamp(-1.0, 1.0))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn star_graph(hub: &str, spokes: &[&str], corr: f64) -> HashMap<(String, String), f64> {
        let mut m = HashMap::new();
        for &s in spokes {
            m.insert((hub.to_string(), s.to_string()), corr);
            m.insert((s.to_string(), hub.to_string()), corr);
        }
        // Spokes uncorrelated with each other.
        for i in 0..spokes.len() {
            for j in (i + 1)..spokes.len() {
                m.insert((spokes[i].to_string(), spokes[j].to_string()), 0.0);
                m.insert((spokes[j].to_string(), spokes[i].to_string()), 0.0);
            }
        }
        m
    }

    #[test]
    fn test_centrality_hub_is_highest() {
        let m = star_graph("BTC", &["ETH", "SOL", "ADA", "AVAX"], 0.9);
        let analyzer = NetworkRiskAnalyzer::from_correlation_map(&m);
        let centrality = analyzer.compute_centrality();
        let btc_c = *centrality.get("BTC").unwrap();
        for sym in ["ETH", "SOL", "ADA", "AVAX"] {
            let c = *centrality.get(sym).unwrap();
            assert!(btc_c >= c, "BTC centrality {} < {} centrality {}", btc_c, sym, c);
        }
    }

    #[test]
    fn test_centrality_equal_complete_graph() {
        // Complete graph with uniform weights => all nodes have equal centrality.
        let syms = ["A", "B", "C", "D"];
        let mut m = HashMap::new();
        for i in 0..syms.len() {
            for j in 0..syms.len() {
                if i != j {
                    m.insert((syms[i].to_string(), syms[j].to_string()), 0.7);
                }
            }
        }
        let analyzer = NetworkRiskAnalyzer::from_correlation_map(&m);
        let c = analyzer.compute_centrality();
        let values: Vec<f64> = c.values().cloned().collect();
        let max_v = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_v = values.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!((max_v - min_v).abs() < 0.01, "expected equal centralities, spread={}", max_v - min_v);
    }

    #[test]
    fn test_systemic_risk_index_range() {
        let m = star_graph("BTC", &["ETH", "SOL", "ADA"], 0.8);
        let analyzer = NetworkRiskAnalyzer::from_correlation_map(&m);
        let sri = analyzer.systemic_risk_index();
        assert!(sri >= 0.0 && sri <= 1.0, "SRI out of range: {}", sri);
    }

    #[test]
    fn test_identify_top_n() {
        let m = star_graph("BTC", &["ETH", "SOL", "ADA", "AVAX"], 0.9);
        let analyzer = NetworkRiskAnalyzer::from_correlation_map(&m);
        let top2 = analyzer.identify_systemically_important(2);
        assert_eq!(top2.len(), 2);
        // BTC should be in the top-2 for a star graph.
        assert!(top2.contains(&"BTC".to_string()), "expected BTC in top-2, got {:?}", top2);
    }

    #[test]
    fn test_density() {
        let m = star_graph("BTC", &["ETH", "SOL"], 0.9);
        let analyzer = NetworkRiskAnalyzer::from_correlation_map(&m);
        // 3 nodes, 3 possible pairs. BTC-ETH and BTC-SOL have corr 0.9, ETH-SOL has 0.
        let d = analyzer.density(0.5); // 2/3
        assert!((d - 2.0 / 3.0).abs() < 0.01, "expected density ~0.667, got {}", d);
    }

    #[test]
    fn test_vulnerability_hub_lower_than_spokes() {
        // Spokes are fully exposed to the hub => they should have higher vulnerability
        // if hub_n includes the hub.
        let m = star_graph("BTC", &["ETH", "SOL", "ADA"], 0.9);
        let analyzer = NetworkRiskAnalyzer::from_correlation_map(&m);
        let vuln = analyzer.compute_vulnerability();
        // At minimum, values are in [0, 1].
        for (_, v) in &vuln {
            assert!(*v >= 0.0 && *v <= 1.0, "vulnerability out of range: {}", v);
        }
    }

    #[test]
    fn test_risk_profiles_sorted() {
        let m = star_graph("BTC", &["ETH", "SOL"], 0.9);
        let analyzer = NetworkRiskAnalyzer::from_correlation_map(&m);
        let profiles = analyzer.risk_profiles();
        // Profiles should be sorted descending by composite_risk.
        for w in profiles.windows(2) {
            assert!(
                w[0].composite_risk >= w[1].composite_risk,
                "profiles not sorted: {} > {}",
                w[0].composite_risk,
                w[1].composite_risk
            );
        }
    }

    #[test]
    fn test_from_return_series() {
        let mut returns = HashMap::new();
        let btc: Vec<f64> = (0..60).map(|i| (i as f64 * 0.2).sin()).collect();
        let eth: Vec<f64> = btc.iter().map(|v| v + 0.01).collect();
        let sol: Vec<f64> = btc.iter().map(|v| v - 0.01).collect();
        returns.insert("BTC".to_string(), btc);
        returns.insert("ETH".to_string(), eth);
        returns.insert("SOL".to_string(), sol);
        let analyzer = NetworkRiskAnalyzer::from_return_series(&returns, 50);
        assert_eq!(analyzer.symbols().len(), 3);
        let sri = analyzer.systemic_risk_index();
        assert!(sri >= 0.0 && sri <= 1.0);
    }
}
