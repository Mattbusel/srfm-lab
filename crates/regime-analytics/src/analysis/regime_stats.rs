// regime_stats.rs
// Conditional statistics by regime.
// Computes mean, vol, skew, kurtosis per regime.
// Persistence matrix P(stay | current regime).
// Expected regime duration.

use serde::{Deserialize, Serialize};

/// Conditional statistics for a single regime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeConditionalStats {
    pub regime: usize,
    pub regime_name: String,
    pub n_obs: usize,
    /// Fraction of total time in this regime.
    pub freq: f64,
    pub mean: f64,
    pub std: f64,
    pub skewness: f64,
    pub excess_kurtosis: f64,
    pub sharpe_annualized: f64,
    pub max_drawdown: f64,
    pub min_obs: f64,
    pub max_obs: f64,
    pub median: f64,
    /// 5th percentile (Value-at-Risk proxy).
    pub p05: f64,
}

impl RegimeConditionalStats {
    fn compute(regime: usize, name: String, obs: &[f64], total_n: usize) -> Self {
        let n = obs.len();
        if n == 0 {
            return RegimeConditionalStats {
                regime,
                regime_name: name,
                n_obs: 0,
                freq: 0.0,
                mean: 0.0,
                std: 0.0,
                skewness: 0.0,
                excess_kurtosis: 0.0,
                sharpe_annualized: 0.0,
                max_drawdown: 0.0,
                min_obs: 0.0,
                max_obs: 0.0,
                median: 0.0,
                p05: 0.0,
            };
        }

        let nf = n as f64;
        let mean = obs.iter().sum::<f64>() / nf;
        let var = obs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / nf;
        let std = var.sqrt().max(1e-10);

        // Skewness and kurtosis.
        let m3 = obs.iter().map(|v| ((v - mean) / std).powi(3)).sum::<f64>() / nf;
        let m4 = obs.iter().map(|v| ((v - mean) / std).powi(4)).sum::<f64>() / nf;
        let skewness = m3;
        let excess_kurtosis = m4 - 3.0;

        let sharpe_annualized = mean / std * (252.0f64).sqrt();

        // Max drawdown.
        let max_drawdown = compute_max_drawdown(obs);

        let mut sorted = obs.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let min_obs = sorted[0];
        let max_obs = sorted[sorted.len() - 1];
        let median = sorted[n / 2];
        let p05 = sorted[(0.05 * n as f64) as usize];

        RegimeConditionalStats {
            regime,
            regime_name: name,
            n_obs: n,
            freq: n as f64 / total_n as f64,
            mean,
            std,
            skewness,
            excess_kurtosis,
            sharpe_annualized,
            max_drawdown,
            min_obs,
            max_obs,
            median,
            p05,
        }
    }
}

/// Compute max drawdown from a returns series.
fn compute_max_drawdown(returns: &[f64]) -> f64 {
    let mut cum = 1.0f64;
    let mut peak = 1.0f64;
    let mut max_dd = 0.0f64;
    for &r in returns {
        cum *= 1.0 + r;
        if cum > peak {
            peak = cum;
        }
        let dd = (peak - cum) / peak.max(1e-10);
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

/// Transition persistence matrix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceMatrix {
    pub n_regimes: usize,
    /// p[i][j] = empirical P(next = j | current = i).
    pub p: Vec<Vec<f64>>,
    /// Expected duration per regime: 1 / (1 - p[i][i]).
    pub expected_durations: Vec<f64>,
    /// Raw transition counts.
    pub counts: Vec<Vec<usize>>,
}

impl PersistenceMatrix {
    /// Compute empirical persistence matrix from a sequence of regime labels.
    pub fn from_states(states: &[usize], n_regimes: usize) -> Self {
        let mut counts = vec![vec![0usize; n_regimes]; n_regimes];
        for window in states.windows(2) {
            let from = window[0].min(n_regimes - 1);
            let to = window[1].min(n_regimes - 1);
            counts[from][to] += 1;
        }
        // Normalize rows.
        let mut p = vec![vec![0.0f64; n_regimes]; n_regimes];
        for i in 0..n_regimes {
            let row_sum: usize = counts[i].iter().sum();
            if row_sum > 0 {
                for j in 0..n_regimes {
                    p[i][j] = counts[i][j] as f64 / row_sum as f64;
                }
            } else {
                p[i][i] = 1.0;
            }
        }
        let expected_durations: Vec<f64> = (0..n_regimes)
            .map(|i| {
                let p_stay = p[i][i];
                if (1.0 - p_stay) < 1e-10 {
                    f64::INFINITY
                } else {
                    1.0 / (1.0 - p_stay)
                }
            })
            .collect();

        PersistenceMatrix { n_regimes, p, expected_durations, counts }
    }
}

/// Overall regime statistics container.
pub struct RegimeStats {
    pub n_regimes: usize,
    pub regime_names: Vec<String>,
}

impl RegimeStats {
    pub fn new(n_regimes: usize, names: Option<Vec<String>>) -> Self {
        let regime_names = names.unwrap_or_else(|| {
            (0..n_regimes).map(|i| format!("Regime{}", i)).collect()
        });
        RegimeStats { n_regimes, regime_names }
    }

    /// Compute conditional statistics for each regime.
    pub fn compute_conditional(
        &self,
        returns: &[f64],
        states: &[usize],
    ) -> Vec<RegimeConditionalStats> {
        let total_n = returns.len();
        (0..self.n_regimes)
            .map(|regime| {
                let obs: Vec<f64> = returns
                    .iter()
                    .zip(states.iter())
                    .filter(|(_, &s)| s == regime)
                    .map(|(&r, _)| r)
                    .collect();
                RegimeConditionalStats::compute(
                    regime,
                    self.regime_names[regime].clone(),
                    &obs,
                    total_n,
                )
            })
            .collect()
    }

    /// Compute the persistence matrix.
    pub fn persistence_matrix(&self, states: &[usize]) -> PersistenceMatrix {
        PersistenceMatrix::from_states(states, self.n_regimes)
    }

    /// Compute full analysis: conditional stats + persistence.
    pub fn full_analysis(
        &self,
        returns: &[f64],
        states: &[usize],
    ) -> (Vec<RegimeConditionalStats>, PersistenceMatrix) {
        let cond = self.compute_conditional(returns, states);
        let persist = self.persistence_matrix(states);
        (cond, persist)
    }

    /// Compute information ratio per regime.
    pub fn ir_by_regime(
        &self,
        returns: &[f64],
        states: &[usize],
    ) -> Vec<(usize, f64)> {
        let cond = self.compute_conditional(returns, states);
        cond.iter()
            .map(|s| (s.regime, s.sharpe_annualized))
            .collect()
    }

    /// Regime-conditional Sharpe vs. overall Sharpe.
    pub fn sharpe_comparison(
        &self,
        returns: &[f64],
        states: &[usize],
    ) -> Vec<(String, f64, f64)> {
        let n = returns.len() as f64;
        let overall_mean = returns.iter().sum::<f64>() / n;
        let overall_std = (returns.iter().map(|v| (v - overall_mean).powi(2)).sum::<f64>() / n)
            .sqrt()
            .max(1e-10);
        let overall_sharpe = overall_mean / overall_std * (252.0f64).sqrt();
        let cond = self.compute_conditional(returns, states);
        cond.iter()
            .map(|s| {
                (
                    s.regime_name.clone(),
                    s.sharpe_annualized,
                    s.sharpe_annualized - overall_sharpe,
                )
            })
            .collect()
    }

    /// Regime volatility ratio: vol_regime / vol_overall.
    pub fn vol_ratio(&self, returns: &[f64], states: &[usize]) -> Vec<(String, f64)> {
        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let overall_vol = (returns.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n)
            .sqrt()
            .max(1e-10);
        let cond = self.compute_conditional(returns, states);
        cond.iter()
            .map(|s| (s.regime_name.clone(), s.std / overall_vol))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_regime_data(n: usize) -> (Vec<f64>, Vec<usize>) {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(314);
        let mut returns = Vec::with_capacity(n);
        let mut states = Vec::with_capacity(n);
        for i in 0..n {
            let s = if i < n / 2 { 0 } else { 1 };
            let r = if s == 0 {
                0.001 + rng.gen::<f64>() * 0.01
            } else {
                -0.001 + rng.gen::<f64>() * 0.02 - 0.01
            };
            returns.push(r);
            states.push(s);
        }
        (returns, states)
    }

    #[test]
    fn test_conditional_stats_computed() {
        let (returns, states) = two_regime_data(400);
        let rs = RegimeStats::new(2, None);
        let cond = rs.compute_conditional(&returns, &states);
        assert_eq!(cond.len(), 2);
        assert!(cond[0].n_obs > 0);
        assert!(cond[1].n_obs > 0);
        // Regime 0 should have higher mean.
        assert!(cond[0].mean > cond[1].mean, "Regime 0 should have higher mean return");
    }

    #[test]
    fn test_persistence_matrix_rows_sum_to_one() {
        let (_, states) = two_regime_data(500);
        let persist = PersistenceMatrix::from_states(&states, 2);
        for row in &persist.p {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "Transition row should sum to 1: {}", sum);
        }
    }

    #[test]
    fn test_expected_duration_positive() {
        let states: Vec<usize> = (0..100).map(|i| i / 20 % 2).collect();
        let persist = PersistenceMatrix::from_states(&states, 2);
        for dur in &persist.expected_durations {
            assert!(*dur > 0.0 || dur.is_infinite());
        }
    }
}
