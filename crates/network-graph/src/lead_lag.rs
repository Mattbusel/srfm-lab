use crate::correlation_matrix::pearson_stable;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Describes the lead/lag relationship between two assets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LeadLagDirection {
    /// X moves before Y (positive lag).
    XLeads,
    /// Y moves before X (negative lag).
    YLeads,
    /// No significant lead/lag (lag ≈ 0).
    Contemporaneous,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeadLagResult {
    /// Symbol X.
    pub sym_x: String,
    /// Symbol Y.
    pub sym_y: String,
    /// Lag in bars at which |correlation| is maximised.
    /// Positive ⟹ X leads Y by `lag` bars.
    /// Negative ⟹ Y leads X by `|lag|` bars.
    pub lag: i32,
    /// Correlation value at the peak lag.
    pub correlation: f64,
    pub direction: LeadLagDirection,
}

/// Directed edge in the lead-lag network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeadLagEdge {
    /// The leader symbol.
    pub leader: String,
    /// The follower symbol.
    pub follower: String,
    /// Lag in bars (always positive: leader moves this many bars before follower).
    pub lag_bars: u32,
    /// Correlation strength at the detected lag.
    pub correlation: f64,
}

/// Directed lead-lag network for all analysed pairs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeadLagNetwork {
    pub edges: Vec<LeadLagEdge>,
    /// Per-symbol in-degree (being led by others).
    pub in_degree: HashMap<String, u32>,
    /// Per-symbol out-degree (leading others).
    pub out_degree: HashMap<String, u32>,
}

/// Compute cross-correlation of x and y at integer lags from -max_lag to +max_lag.
///
/// Returns a vector of length 2*max_lag+1. Index `max_lag + k` = correlation at lag k.
/// Positive lag k means x[t] is compared with y[t+k] (x leads y by k).
pub fn cross_correlation(x: &[f64], y: &[f64], max_lag: usize) -> Vec<f64> {
    let n = x.len().min(y.len());
    let total = 2 * max_lag + 1;
    let mut result = vec![0.0_f64; total];

    for (out_idx, lag_offset) in (0..total).enumerate() {
        let lag = lag_offset as i64 - max_lag as i64;
        // x[t], y[t + lag] => correlate x[start_x..end_x] with y[start_y..end_y]
        let (start_x, start_y) = if lag >= 0 {
            (0usize, lag as usize)
        } else {
            ((-lag) as usize, 0usize)
        };
        let available = n.saturating_sub(lag.unsigned_abs() as usize);
        if available < 3 {
            continue;
        }
        let xs = &x[start_x..start_x + available];
        let ys = &y[start_y..start_y + available];
        result[out_idx] = pearson_stable(xs, ys).unwrap_or(0.0);
    }

    result
}

/// Find the dominant lead/lag between two series.
///
/// Scans lags from -max_lag to +max_lag and returns the one with the highest |correlation|.
pub fn find_lead_lag(
    sym_x: &str,
    sym_y: &str,
    x: &[f64],
    y: &[f64],
    max_lag: usize,
) -> LeadLagResult {
    let xcorr = cross_correlation(x, y, max_lag);
    let _total = xcorr.len();

    let (best_idx, &best_corr) = xcorr
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.abs().partial_cmp(&b.abs()).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((max_lag, &0.0));

    let lag = best_idx as i32 - max_lag as i32;

    let direction = if lag.abs() <= 0 {
        LeadLagDirection::Contemporaneous
    } else if lag > 0 {
        LeadLagDirection::XLeads
    } else {
        LeadLagDirection::YLeads
    };

    LeadLagResult {
        sym_x: sym_x.to_string(),
        sym_y: sym_y.to_string(),
        lag,
        correlation: best_corr,
        direction,
    }
}

/// Build the lead-lag network for all pairs in the return series.
///
/// `max_lag` — maximum lag to scan in bars (default 5).
/// `min_corr` — minimum |correlation| to include an edge (default 0.3).
pub fn build_lead_lag_network(
    returns: &HashMap<String, Vec<f64>>,
    max_lag: usize,
    min_corr: f64,
) -> LeadLagNetwork {
    let symbols: Vec<String> = {
        let mut v: Vec<String> = returns.keys().cloned().collect();
        v.sort();
        v
    };
    let n = symbols.len();

    let mut edges: Vec<LeadLagEdge> = Vec::new();
    let mut in_degree: HashMap<String, u32> = symbols.iter().map(|s| (s.clone(), 0)).collect();
    let mut out_degree: HashMap<String, u32> = symbols.iter().map(|s| (s.clone(), 0)).collect();

    for i in 0..n {
        for j in (i + 1)..n {
            let x = &returns[&symbols[i]];
            let y = &returns[&symbols[j]];
            let result = find_lead_lag(&symbols[i], &symbols[j], x, y, max_lag);

            if result.correlation.abs() < min_corr {
                continue;
            }

            match result.direction {
                LeadLagDirection::XLeads => {
                    let leader = symbols[i].clone();
                    let follower = symbols[j].clone();
                    *out_degree.entry(leader.clone()).or_insert(0) += 1;
                    *in_degree.entry(follower.clone()).or_insert(0) += 1;
                    edges.push(LeadLagEdge {
                        leader,
                        follower,
                        lag_bars: result.lag as u32,
                        correlation: result.correlation,
                    });
                }
                LeadLagDirection::YLeads => {
                    let leader = symbols[j].clone();
                    let follower = symbols[i].clone();
                    *out_degree.entry(leader.clone()).or_insert(0) += 1;
                    *in_degree.entry(follower.clone()).or_insert(0) += 1;
                    edges.push(LeadLagEdge {
                        leader,
                        follower,
                        lag_bars: (-result.lag) as u32,
                        correlation: result.correlation,
                    });
                }
                LeadLagDirection::Contemporaneous => {
                    // No directed edge for contemporaneous moves.
                }
            }
        }
    }

    // Sort edges by correlation strength descending.
    edges.sort_unstable_by(|a, b| {
        b.correlation
            .abs()
            .partial_cmp(&a.correlation.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    LeadLagNetwork { edges, in_degree, out_degree }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build x and y such that x leads y by `lag` bars.
    /// y[t] = x[t - lag] => y is a delayed copy of x => x precedes y.
    fn lagged_series(n: usize, lag: usize) -> (Vec<f64>, Vec<f64>) {
        let signal: Vec<f64> = (0..n + lag).map(|i| (i as f64 * 0.3).sin()).collect();
        // x = signal[lag..lag+n]  (the "early" part)
        // y = signal[0..n]        (delayed by lag relative to x)
        // => x[t] = signal[t + lag], y[t] = signal[t]
        // => x[t] = y[t + lag], so x is ahead of y by `lag` bars (x leads y)
        let x: Vec<f64> = signal[lag..lag + n].to_vec();
        let y: Vec<f64> = signal[..n].to_vec();
        (x, y)
    }

    #[test]
    fn test_cross_correlation_lag_zero_identity() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2).sin()).collect();
        let xcorr = cross_correlation(&x, &x, 5);
        let center = 5; // index for lag=0
        assert!((xcorr[center] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_find_lead_lag_detects_x_leads() {
        let (x, y) = lagged_series(80, 2);
        let result = find_lead_lag("X", "Y", &x, &y, 5);
        // x leads y by 2 bars, so lag should be positive (XLeads direction).
        assert_eq!(result.direction, LeadLagDirection::XLeads,
            "expected XLeads, got {:?}, lag={}", result.direction, result.lag);
        assert!(result.lag > 0, "expected positive lag, got {}", result.lag);
    }

    #[test]
    fn test_find_lead_lag_contemporaneous() {
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        let result = find_lead_lag("X", "Y", &x, &x, 5);
        // Comparing identical series should show lag=0.
        assert_eq!(result.lag, 0);
        assert!((result.correlation - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_build_network_edge_count() {
        let mut returns = HashMap::new();
        let (x, y) = lagged_series(100, 1);
        returns.insert("BTC".to_string(), x);
        returns.insert("ETH".to_string(), y);
        let net = build_lead_lag_network(&returns, 5, 0.1);
        // Should detect one edge (BTC leads ETH or ETH leads BTC).
        assert!(!net.edges.is_empty());
    }

    #[test]
    fn test_out_degree_leader() {
        let mut returns = HashMap::new();
        // BTC leads ETH and SOL.
        let btc: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin()).collect();
        let eth: Vec<f64> = btc[1..].iter().chain(std::iter::once(&0.0_f64)).cloned().collect();
        let sol: Vec<f64> = btc[2..].iter().chain([0.0_f64, 0.0_f64].iter()).cloned().collect();
        returns.insert("BTC".to_string(), btc);
        returns.insert("ETH".to_string(), eth);
        returns.insert("SOL".to_string(), sol);
        let net = build_lead_lag_network(&returns, 5, 0.0);
        // BTC should have higher out_degree than in_degree.
        let btc_out = net.out_degree.get("BTC").copied().unwrap_or(0);
        let btc_in = net.in_degree.get("BTC").copied().unwrap_or(0);
        assert!(btc_out >= btc_in, "BTC out={} in={}", btc_out, btc_in);
    }
}
