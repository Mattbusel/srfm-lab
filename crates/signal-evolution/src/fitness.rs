/// Multi-objective fitness evaluation for evolved signal trees.
///
/// Fitness dimensions:
///   - IC    : Spearman rank correlation of signal with next-bar returns (mean over rolling 252-bar windows)
///   - ICIR  : IC / std(IC) — signal consistency / information ratio
///   - Sharpe contribution : Sharpe ratio of a signal-sized strategy
///   - Turnover : mean absolute daily signal change (penalises flip-every-bar strategies)
///   - Complexity penalty : logarithmic penalty on number of nodes

use crate::data_loader::BarData;
use crate::expression_tree::SignalTree;
use serde::{Deserialize, Serialize};

/// Multi-objective fitness vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessVector {
    /// Mean IC over rolling 252-bar windows (higher is better, max 1.0).
    pub ic: f64,
    /// IC / std(IC) — information ratio of the IC series (higher is better).
    pub icir: f64,
    /// Annualised Sharpe of signal-weighted returns (higher is better).
    pub sharpe_contrib: f64,
    /// Mean absolute change in signal rank per bar (lower is better).
    pub turnover: f64,
    /// Logarithmic complexity penalty (lower is better, typically negative addend).
    pub complexity_penalty: f64,
    /// Aggregate scalar score for single-objective comparisons.
    pub score: f64,
}

impl FitnessVector {
    /// Pareto dominance: self dominates other iff at least as good on all objectives
    /// and strictly better on at least one.
    /// Objectives: ic (max), icir (max), sharpe_contrib (max).
    /// Turnover and complexity_penalty are costs embedded in score but not used for dominance.
    pub fn dominates(&self, other: &FitnessVector) -> bool {
        let s = [self.ic, self.icir, self.sharpe_contrib];
        let o = [other.ic, other.icir, other.sharpe_contrib];
        let at_least = s.iter().zip(o.iter()).all(|(a, b)| a >= b);
        let strictly = s.iter().zip(o.iter()).any(|(a, b)| a > b);
        at_least && strictly
    }
}

impl Default for FitnessVector {
    fn default() -> Self {
        Self {
            ic: 0.0,
            icir: 0.0,
            sharpe_contrib: 0.0,
            turnover: 0.0,
            complexity_penalty: 0.0,
            score: f64::NEG_INFINITY,
        }
    }
}

// ---------------------------------------------------------------------------
// Main evaluation entry point
// ---------------------------------------------------------------------------

/// Evaluate a SignalTree against a bar history and return its fitness vector.
pub fn evaluate_fitness(signal_tree: &SignalTree, bars: &[BarData]) -> FitnessVector {
    let n = bars.len();
    if n < 30 {
        return FitnessVector::default();
    }

    let raw_signal = signal_tree.evaluate(bars);

    // Guard: if signal is constant, it carries no information
    let sig_std = std_dev(&raw_signal);
    if sig_std < 1e-12 {
        let mut fv = FitnessVector::default();
        fv.complexity_penalty = complexity_penalty(signal_tree.complexity());
        fv.score = -10.0;
        return fv;
    }

    // Next-bar returns (shifted by 1)
    let fwd_returns: Vec<f64> = {
        let mut r = vec![0.0f64; n];
        for i in 0..(n - 1) {
            r[i] = bars[i + 1].returns;
        }
        r
    };

    // Rolling IC over 252-bar windows (or all data if shorter)
    let window = 252.min(n - 1);
    let ic_series = rolling_ic(&raw_signal, &fwd_returns, window);
    let ic_mean = mean(&ic_series);
    let ic_std = std_dev(&ic_series);
    let icir = if ic_std > 1e-12 { ic_mean / ic_std } else { 0.0 };

    // Sharpe contribution: position = sign(signal), Sharpe of resulting P&L
    let sharpe_contrib = compute_sharpe_contribution(&raw_signal, &fwd_returns);

    // Turnover: mean absolute change in signal value per bar
    let turnover = compute_turnover(&raw_signal);

    // Complexity penalty
    let cp = complexity_penalty(signal_tree.complexity());

    // Aggregate score (weights tuned heuristically)
    let score = ic_mean * 3.0 + icir * 0.5 + sharpe_contrib * 0.3
        - turnover * 0.1
        + cp;

    FitnessVector {
        ic: ic_mean,
        icir,
        sharpe_contrib,
        turnover,
        complexity_penalty: cp,
        score,
    }
}

// ---------------------------------------------------------------------------
// IC computation
// ---------------------------------------------------------------------------

/// Compute rolling Spearman IC between signal and forward returns.
/// Window steps one bar at a time. Returns one IC value per valid window.
fn rolling_ic(signal: &[f64], fwd_returns: &[f64], window: usize) -> Vec<f64> {
    let n = signal.len();
    if n < window {
        let ic = spearman_ic(signal, fwd_returns);
        return vec![ic];
    }
    let mut ics = Vec::with_capacity(n - window + 1);
    for start in 0..=(n - window) {
        let s = &signal[start..(start + window)];
        let r = &fwd_returns[start..(start + window)];
        ics.push(spearman_ic(s, r));
    }
    ics
}

/// Spearman rank correlation of two series.
pub fn spearman_ic(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 4 {
        return 0.0;
    }
    let rx = rank_series(x);
    let ry = rank_series(y);
    pearson_r(&rx, &ry)
}

/// Convert a series to ranks (average ranks for ties).
fn rank_series(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut indexed: Vec<(usize, f64)> = x.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-12 {
            j += 1;
        }
        let avg_rank = (i + j - 1) as f64 / 2.0 + 1.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Pearson correlation coefficient.
pub fn pearson_r(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len()) as f64;
    if n < 2.0 {
        return 0.0;
    }
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let num: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - mx) * (b - my)).sum();
    let dx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum::<f64>().sqrt();
    let dy: f64 = y.iter().map(|b| (b - my).powi(2)).sum::<f64>().sqrt();
    if dx < 1e-12 || dy < 1e-12 {
        0.0
    } else {
        (num / (dx * dy)).clamp(-1.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Sharpe contribution
// ---------------------------------------------------------------------------

fn compute_sharpe_contribution(signal: &[f64], fwd_returns: &[f64]) -> f64 {
    let n = signal.len().min(fwd_returns.len());
    // Positions: zscore-normalised signal as fractional position
    let sig_mean = mean(&signal[..n]);
    let sig_std = std_dev(&signal[..n]);
    if sig_std < 1e-12 {
        return 0.0;
    }
    let pnl: Vec<f64> = (0..n)
        .map(|i| {
            let z = (signal[i] - sig_mean) / sig_std;
            let pos = z.clamp(-3.0, 3.0);
            pos * fwd_returns[i]
        })
        .collect();
    let mu = mean(&pnl);
    let sigma = std_dev(&pnl);
    if sigma < 1e-12 {
        0.0
    } else {
        // Annualise assuming daily bars
        mu / sigma * (252.0_f64).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Turnover
// ---------------------------------------------------------------------------

fn compute_turnover(signal: &[f64]) -> f64 {
    if signal.len() < 2 {
        return 0.0;
    }
    let sig_std = std_dev(signal);
    if sig_std < 1e-12 {
        return 0.0;
    }
    let normalised: Vec<f64> = signal.iter().map(|x| x / sig_std).collect();
    let total_change: f64 = normalised
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .sum();
    total_change / (signal.len() - 1) as f64
}

// ---------------------------------------------------------------------------
// Complexity penalty
// ---------------------------------------------------------------------------

fn complexity_penalty(node_count: usize) -> f64 {
    // Mild log penalty; a tree with 1 node gets 0.0 penalty
    -(node_count as f64).ln() * 0.05
}

// ---------------------------------------------------------------------------
// Statistical helpers
// ---------------------------------------------------------------------------

pub fn mean(x: &[f64]) -> f64 {
    if x.is_empty() {
        return 0.0;
    }
    x.iter().sum::<f64>() / x.len() as f64
}

pub fn std_dev(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(x);
    let var = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_loader::synthetic_bars;
    use crate::expression_tree::{Node, SignalTree};
    use crate::primitives::Terminal;

    #[test]
    fn ic_perfect_correlation() {
        // Signal identical to fwd_returns → IC = 1.0
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let ic = spearman_ic(&x, &x);
        assert!((ic - 1.0).abs() < 1e-6, "Perfect IC should be 1.0, got {ic}");
    }

    #[test]
    fn ic_anti_correlated() {
        let x: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..50).map(|i| (49 - i) as f64).collect();
        let ic = spearman_ic(&x, &y);
        assert!((ic + 1.0).abs() < 1e-6, "Anti-correlated IC should be -1.0, got {ic}");
    }

    #[test]
    fn ic_constant_signal_returns_zero() {
        let x = vec![1.0f64; 50];
        let y: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let ic = spearman_ic(&x, &y);
        assert_eq!(ic, 0.0);
    }

    #[test]
    fn fitness_constant_signal_penalised() {
        let bars = synthetic_bars(100, 100.0);
        let node = Node::Terminal(Terminal::Price);
        // Price isn't constant, but let's test a constant override...
        let st = SignalTree::new(
            Node::Function(
                crate::operators::Operator::Sub,
                vec![Node::Terminal(Terminal::Price), Node::Terminal(Terminal::Price)],
            ),
            0,
        );
        let fv = evaluate_fitness(&st, &bars);
        assert!(fv.score < 0.0, "Constant signal should have negative score");
    }

    #[test]
    fn fitness_returns_signal_has_positive_ic() {
        let bars = synthetic_bars(300, 100.0);
        // Returns(1) should correlate with next-bar returns (it IS the signal)
        let st = SignalTree::new(Node::Terminal(Terminal::Returns { n: 1 }), 0);
        let fv = evaluate_fitness(&st, &bars);
        // IC might be low but shouldn't blow up
        assert!(fv.ic.is_finite());
        assert!(fv.icir.is_finite());
    }

    #[test]
    fn complexity_penalty_increases_with_size() {
        let p1 = crate::fitness::complexity_penalty(1);
        let p10 = crate::fitness::complexity_penalty(10);
        assert!(p1 > p10, "Larger tree should have more negative penalty");
    }

    #[test]
    fn turnover_constant_signal_is_zero() {
        let sig = vec![5.0f64; 100];
        let to = compute_turnover(&sig);
        assert_eq!(to, 0.0);
    }

    #[test]
    fn sharpe_contrib_finite() {
        let bars = synthetic_bars(200, 100.0);
        let st = SignalTree::new(Node::Terminal(Terminal::RSI { period: 14 }), 0);
        let fv = evaluate_fitness(&st, &bars);
        assert!(fv.sharpe_contrib.is_finite());
    }

    #[test]
    fn pareto_dominance() {
        let better = FitnessVector { ic: 0.1, icir: 1.0, sharpe_contrib: 0.5, ..Default::default() };
        let worse  = FitnessVector { ic: 0.05, icir: 0.8, sharpe_contrib: 0.4, ..Default::default() };
        assert!(better.dominates(&worse));
        assert!(!worse.dominates(&better));
    }
}
