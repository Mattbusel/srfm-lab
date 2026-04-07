// alpha_combination.rs
// Combine multiple alpha signals into a composite.
// Methods: equal weight, IC-weighted, max diversification, Markowitz on IC series.

use serde::{Deserialize, Serialize};
use crate::{pearson_corr, spearman_rank_corr};

/// Combination method for multiple alpha signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CombinationMethod {
    /// Each signal receives equal weight.
    EqualWeight,
    /// Weights proportional to recent mean IC.
    IcWeighted,
    /// Minimize correlation between combined signal and each constituent.
    MaxDiversification,
    /// Markowitz on IC series: maximize IC_combined / vol_combined.
    MarkowitzIc,
}

/// Combined alpha signal output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinedAlpha {
    pub method: CombinationMethod,
    /// Weights assigned to each input signal (sum to 1).
    pub weights: Vec<f64>,
    /// Combined signal scores (cross-section at one bar).
    pub scores: Vec<f64>,
    /// Expected IC of the combined signal (estimated).
    pub expected_ic: f64,
    /// Expected ICIR of the combined signal.
    pub expected_icir: f64,
    /// Diversification ratio: sum of individual signal ICs / combined IC.
    pub diversification_ratio: f64,
}

/// Input: a named alpha signal with its cross-section scores and IC history.
#[derive(Debug, Clone)]
pub struct AlphaSignal {
    pub name: String,
    /// Cross-section scores at the current bar (one per asset).
    pub scores: Vec<f64>,
    /// Recent IC history (rolling window).
    pub ic_history: Vec<f64>,
}

impl AlphaSignal {
    pub fn mean_ic(&self) -> f64 {
        if self.ic_history.is_empty() {
            return 0.0;
        }
        self.ic_history.iter().sum::<f64>() / self.ic_history.len() as f64
    }

    pub fn std_ic(&self) -> f64 {
        if self.ic_history.len() < 2 {
            return 1e-6;
        }
        let mean = self.mean_ic();
        let var = self.ic_history.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / self.ic_history.len() as f64;
        var.sqrt().max(1e-10)
    }

    pub fn icir(&self) -> f64 {
        let std = self.std_ic();
        if std > 1e-10 { self.mean_ic() / std } else { 0.0 }
    }
}

/// Compute pairwise IC correlation matrix from historical IC series.
/// Returns flattened [n x n] correlation matrix.
pub fn ic_correlation_matrix(signals: &[AlphaSignal]) -> Vec<Vec<f64>> {
    let n = signals.len();
    let min_len = signals.iter().map(|s| s.ic_history.len()).min().unwrap_or(0);
    let mut corr = vec![vec![1.0f64; n]; n];
    if min_len < 2 {
        return corr;
    }
    for i in 0..n {
        let xi: Vec<f64> = signals[i].ic_history[signals[i].ic_history.len() - min_len..].to_vec();
        for j in (i + 1)..n {
            let xj: Vec<f64> =
                signals[j].ic_history[signals[j].ic_history.len() - min_len..].to_vec();
            let c = pearson_corr(&xi, &xj);
            corr[i][j] = c;
            corr[j][i] = c;
        }
    }
    corr
}

/// Normalize weights to sum to 1 (or all-equal if degenerate).
fn normalize(w: &[f64]) -> Vec<f64> {
    let sum: f64 = w.iter().sum::<f64>().abs();
    if sum < 1e-14 {
        let n = w.len();
        return vec![1.0 / n as f64; n];
    }
    w.iter().map(|v| v / sum).collect()
}

/// Combine scores using given weights (per-asset weighted average).
fn combine_scores(signals: &[AlphaSignal], weights: &[f64]) -> Vec<f64> {
    let n_assets = signals[0].scores.len();
    let mut combined = vec![0.0f64; n_assets];
    for (signal, &w) in signals.iter().zip(weights.iter()) {
        let scores = &signal.scores;
        let n = scores.len().min(n_assets);
        for i in 0..n {
            combined[i] += w * scores[i];
        }
    }
    combined
}

/// Alpha signal combiner.
pub struct AlphaCombiner {
    /// Long-run IC correlation matrix (updated incrementally).
    ic_corr: Option<Vec<Vec<f64>>>,
}

impl AlphaCombiner {
    pub fn new() -> Self {
        AlphaCombiner { ic_corr: None }
    }

    /// Combine multiple signals using the specified method.
    pub fn combine(
        &mut self,
        signals: &[AlphaSignal],
        method: CombinationMethod,
    ) -> Option<CombinedAlpha> {
        if signals.is_empty() {
            return None;
        }
        let n = signals.len();

        let weights = match method {
            CombinationMethod::EqualWeight => self.equal_weights(n),
            CombinationMethod::IcWeighted => self.ic_weights(signals),
            CombinationMethod::MaxDiversification => self.max_diversification_weights(signals),
            CombinationMethod::MarkowitzIc => self.markowitz_ic_weights(signals),
        };

        let scores = combine_scores(signals, &weights);

        // Estimate combined IC as weighted sum of individual ICs.
        let expected_ic: f64 = signals
            .iter()
            .zip(weights.iter())
            .map(|(s, w)| w * s.mean_ic())
            .sum();

        // Estimate combined IC std via portfolio variance formula.
        let corr = ic_correlation_matrix(signals);
        let mut var_ic = 0.0f64;
        for i in 0..n {
            for j in 0..n {
                var_ic += weights[i] * weights[j] * signals[i].std_ic() * signals[j].std_ic() * corr[i][j];
            }
        }
        let std_ic = var_ic.sqrt().max(1e-10);
        let expected_icir = expected_ic / std_ic;

        let sum_individual_ics: f64 = signals
            .iter()
            .zip(weights.iter())
            .map(|(s, w)| w * s.mean_ic().abs())
            .sum();
        let diversification_ratio = if expected_ic.abs() > 1e-10 {
            sum_individual_ics / expected_ic.abs()
        } else {
            1.0
        };

        self.ic_corr = Some(corr);

        Some(CombinedAlpha {
            method,
            weights,
            scores,
            expected_ic,
            expected_icir,
            diversification_ratio,
        })
    }

    fn equal_weights(&self, n: usize) -> Vec<f64> {
        vec![1.0 / n as f64; n]
    }

    fn ic_weights(&self, signals: &[AlphaSignal]) -> Vec<f64> {
        let raw: Vec<f64> = signals.iter().map(|s| s.mean_ic().max(0.0)).collect();
        normalize(&raw)
    }

    /// Max diversification: weights proportional to individual ICIR / sum of all pairwise correlations.
    fn max_diversification_weights(&self, signals: &[AlphaSignal]) -> Vec<f64> {
        let n = signals.len();
        let corr = ic_correlation_matrix(signals);
        let icirs: Vec<f64> = signals.iter().map(|s| s.icir().abs().max(1e-6)).collect();

        // Iterative reweighting: w_i proportional to ICIR_i / (corr with portfolio).
        let mut w = vec![1.0f64 / n as f64; n];
        for _ in 0..50 {
            // Compute portfolio correlation with each signal.
            let mut new_w = vec![0.0f64; n];
            for i in 0..n {
                let port_corr: f64 = (0..n).map(|j| w[j] * corr[i][j]).sum();
                new_w[i] = if port_corr.abs() > 1e-10 {
                    icirs[i] / port_corr.abs()
                } else {
                    icirs[i]
                };
            }
            let norm = normalize(&new_w);
            let diff: f64 = norm.iter().zip(w.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            w = norm;
            if diff < 1e-9 {
                break;
            }
        }
        w
    }

    /// Markowitz on IC series: maximize expected_IC / std_IC (max Sharpe on IC space).
    /// Uses mean-variance optimization with closed-form solution for long-only.
    fn markowitz_ic_weights(&self, signals: &[AlphaSignal]) -> Vec<f64> {
        let n = signals.len();
        if n == 1 {
            return vec![1.0];
        }

        let mean_ics: Vec<f64> = signals.iter().map(|s| s.mean_ic()).collect();
        let stds: Vec<f64> = signals.iter().map(|s| s.std_ic()).collect();
        let corr = ic_correlation_matrix(signals);

        // Build covariance matrix from stds and corr.
        let mut cov = vec![vec![0.0f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                cov[i][j] = stds[i] * stds[j] * corr[i][j];
            }
        }

        // Unconstrained Markowitz: w proportional to Sigma^{-1} * mu.
        // Solve Sigma * w = mu via Gaussian elimination.
        let w_raw = solve_linear_system(&cov, &mean_ics);

        // Project to non-negative (long-only constraint).
        let w_pos: Vec<f64> = w_raw.iter().map(|&v| v.max(0.0)).collect();
        let sum = w_pos.iter().sum::<f64>();
        if sum < 1e-14 {
            // Fall back to IC-weighted.
            return self.ic_weights(signals);
        }
        normalize(&w_pos)
    }

    /// Access the cached IC correlation matrix.
    pub fn ic_correlation(&self) -> Option<&Vec<Vec<f64>>> {
        self.ic_corr.as_ref()
    }

    /// Compute portfolio of combined alphas across multiple combination methods,
    /// return the method with highest expected ICIR.
    pub fn best_combination(
        &mut self,
        signals: &[AlphaSignal],
    ) -> Option<CombinedAlpha> {
        let methods = [
            CombinationMethod::EqualWeight,
            CombinationMethod::IcWeighted,
            CombinationMethod::MaxDiversification,
            CombinationMethod::MarkowitzIc,
        ];
        let mut best: Option<CombinedAlpha> = None;
        for &m in &methods {
            if let Some(combined) = self.combine(signals, m) {
                let is_better = best
                    .as_ref()
                    .map(|b| combined.expected_icir > b.expected_icir)
                    .unwrap_or(true);
                if is_better {
                    best = Some(combined);
                }
            }
        }
        best
    }
}

/// Solve n x n linear system Ax = b.
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .zip(b.iter())
        .map(|(row, &bi)| {
            let mut r = row.clone();
            r.push(bi);
            r
        })
        .collect();

    for col in 0..n {
        let mut max_row = col;
        for row in (col + 1)..n {
            if aug[row][col].abs() > aug[max_row][col].abs() {
                max_row = row;
            }
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        if pivot.abs() < 1e-14 {
            continue;
        }
        for row in (col + 1)..n {
            let factor = aug[row][col] / pivot;
            for j in col..=n {
                let val = aug[col][j] * factor;
                aug[row][j] -= val;
            }
        }
    }
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        if aug[i][i].abs() > 1e-14 {
            x[i] /= aug[i][i];
        }
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_signals(n: usize, n_assets: usize) -> Vec<AlphaSignal> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(55);
        (0..n)
            .map(|i| AlphaSignal {
                name: format!("signal_{}", i),
                scores: (0..n_assets).map(|_| rng.gen::<f64>() * 2.0 - 1.0).collect(),
                ic_history: (0..100)
                    .map(|_| 0.05 + rng.gen::<f64>() * 0.10 - 0.03)
                    .collect(),
            })
            .collect()
    }

    #[test]
    fn test_equal_weight_sums_to_one() {
        let signals = make_signals(4, 50);
        let mut combiner = AlphaCombiner::new();
        let result = combiner.combine(&signals, CombinationMethod::EqualWeight).unwrap();
        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_ic_weighted_favors_high_ic() {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(77);
        let signals = vec![
            AlphaSignal {
                name: "high_ic".to_string(),
                scores: (0..50).map(|_| rng.gen::<f64>()).collect(),
                ic_history: vec![0.20; 100],
            },
            AlphaSignal {
                name: "low_ic".to_string(),
                scores: (0..50).map(|_| rng.gen::<f64>()).collect(),
                ic_history: vec![0.02; 100],
            },
        ];
        let mut combiner = AlphaCombiner::new();
        let result = combiner.combine(&signals, CombinationMethod::IcWeighted).unwrap();
        assert!(result.weights[0] > result.weights[1], "High IC signal should have higher weight");
    }

    #[test]
    fn test_markowitz_weights_positive() {
        let signals = make_signals(3, 50);
        let mut combiner = AlphaCombiner::new();
        let result = combiner.combine(&signals, CombinationMethod::MarkowitzIc).unwrap();
        for &w in &result.weights {
            assert!(w >= 0.0, "Markowitz weights should be non-negative");
        }
    }
}
