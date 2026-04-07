// regime_backtest.rs
// Backtest P&L conditional on regime.
// Sharpe per regime.
// Regime transition P&L cost.
// Optimal strategy by regime (long/short/flat).

use serde::{Deserialize, Serialize};

/// P&L statistics conditioned on a specific regime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimePnl {
    pub regime: usize,
    pub regime_name: String,
    /// Number of bars in this regime.
    pub n_bars: usize,
    /// Cumulative P&L while in this regime.
    pub cumulative_pnl: f64,
    /// Mean daily P&L.
    pub mean_daily_pnl: f64,
    /// Daily P&L standard deviation.
    pub std_daily_pnl: f64,
    /// Annualized Sharpe ratio.
    pub sharpe: f64,
    /// Win rate (fraction of positive-pnl days).
    pub win_rate: f64,
    /// Max drawdown while in this regime.
    pub max_drawdown: f64,
    /// Calmar ratio: annualized return / max drawdown.
    pub calmar: f64,
    /// Position taken in this regime: 1 = long, -1 = short, 0 = flat.
    pub optimal_position: i32,
}

/// Cost of transitioning between regimes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeTransitionCost {
    pub from_regime: usize,
    pub to_regime: usize,
    /// Average P&L in the bars immediately after a transition.
    pub post_transition_pnl: f64,
    /// Number of transitions observed.
    pub n_transitions: usize,
    /// Lookback bars used.
    pub lookback: usize,
    /// Transition cost estimate: difference from in-regime mean.
    pub cost_estimate: f64,
}

/// Strategy specification per regime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeStrategy {
    pub regime: usize,
    pub position_size: f64,
    /// Minimum Sharpe to apply the strategy in this regime.
    pub sharpe_threshold: f64,
    pub active: bool,
}

/// Regime-conditional backtest engine.
pub struct RegimeBacktest {
    pub n_regimes: usize,
    pub regime_names: Vec<String>,
    /// Transaction cost per unit position change.
    pub transaction_cost: f64,
}

impl RegimeBacktest {
    pub fn new(n_regimes: usize, names: Option<Vec<String>>, transaction_cost: f64) -> Self {
        let regime_names = names.unwrap_or_else(|| {
            (0..n_regimes).map(|i| format!("Regime{}", i)).collect()
        });
        RegimeBacktest { n_regimes, regime_names, transaction_cost }
    }

    /// Compute per-regime P&L statistics for a given signal applied with unit position.
    /// `returns`: asset returns per bar.
    /// `states`: regime label per bar.
    /// `positions`: position per bar (can be +1, 0, -1 or continuous).
    pub fn compute_regime_pnl(
        &self,
        returns: &[f64],
        states: &[usize],
        positions: &[f64],
    ) -> Vec<RegimePnl> {
        assert_eq!(returns.len(), states.len());
        assert_eq!(returns.len(), positions.len());
        let n_total = returns.len();

        (0..self.n_regimes)
            .map(|regime| {
                let regime_data: Vec<(f64, f64)> = returns
                    .iter()
                    .zip(states.iter())
                    .zip(positions.iter())
                    .filter(|((_, &s), _)| s == regime)
                    .map(|((r, _), p)| (*r, *p))
                    .collect();

                if regime_data.is_empty() {
                    return RegimePnl {
                        regime,
                        regime_name: self.regime_names[regime].clone(),
                        n_bars: 0,
                        cumulative_pnl: 0.0,
                        mean_daily_pnl: 0.0,
                        std_daily_pnl: 0.0,
                        sharpe: 0.0,
                        win_rate: 0.0,
                        max_drawdown: 0.0,
                        calmar: 0.0,
                        optimal_position: 0,
                    };
                }

                let daily_pnls: Vec<f64> = regime_data.iter().map(|(r, p)| r * p).collect();
                let n = daily_pnls.len() as f64;
                let mean = daily_pnls.iter().sum::<f64>() / n;
                let std = (daily_pnls.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n)
                    .sqrt()
                    .max(1e-10);
                let sharpe = mean / std * (252.0f64).sqrt();
                let cumulative_pnl: f64 = daily_pnls.iter().sum();
                let win_rate = daily_pnls.iter().filter(|&&v| v > 0.0).count() as f64 / n;
                let max_dd = compute_max_drawdown_vec(&daily_pnls);
                let annualized_ret = mean * 252.0;
                let calmar = if max_dd > 1e-10 { annualized_ret / max_dd } else { 0.0 };

                let optimal_position = if mean > 0.0 { 1 } else if mean < 0.0 { -1 } else { 0 };

                RegimePnl {
                    regime,
                    regime_name: self.regime_names[regime].clone(),
                    n_bars: regime_data.len(),
                    cumulative_pnl,
                    mean_daily_pnl: mean,
                    std_daily_pnl: std,
                    sharpe,
                    win_rate,
                    max_drawdown: max_dd,
                    calmar,
                    optimal_position,
                }
            })
            .collect()
    }

    /// Compute P&L cost of regime transitions.
    /// For each transition from regime i to j, compute mean P&L in `lookback` bars after.
    pub fn transition_costs(
        &self,
        returns: &[f64],
        states: &[usize],
        lookback: usize,
    ) -> Vec<RegimeTransitionCost> {
        let n = returns.len();
        let mut result = Vec::new();

        // Baseline per-regime mean.
        let regime_means: Vec<f64> = (0..self.n_regimes)
            .map(|regime| {
                let obs: Vec<f64> = returns
                    .iter()
                    .zip(states.iter())
                    .filter(|(_, &s)| s == regime)
                    .map(|(r, _)| *r)
                    .collect();
                if obs.is_empty() {
                    0.0
                } else {
                    obs.iter().sum::<f64>() / obs.len() as f64
                }
            })
            .collect();

        for from in 0..self.n_regimes {
            for to in 0..self.n_regimes {
                if from == to {
                    continue;
                }
                // Find all transitions from -> to.
                let transitions: Vec<usize> = states
                    .windows(2)
                    .enumerate()
                    .filter(|(_, w)| w[0] == from && w[1] == to)
                    .map(|(i, _)| i + 1) // bar index after transition.
                    .collect();

                let n_transitions = transitions.len();
                if n_transitions == 0 {
                    continue;
                }

                // Average P&L in `lookback` bars following each transition.
                let post_pnls: Vec<f64> = transitions
                    .iter()
                    .flat_map(|&t_start| {
                        let t_end = (t_start + lookback).min(n);
                        returns[t_start..t_end].iter().copied()
                    })
                    .collect();

                let post_transition_pnl = if post_pnls.is_empty() {
                    0.0
                } else {
                    post_pnls.iter().sum::<f64>() / post_pnls.len() as f64
                };

                let cost_estimate = regime_means[to] - post_transition_pnl;

                result.push(RegimeTransitionCost {
                    from_regime: from,
                    to_regime: to,
                    post_transition_pnl,
                    n_transitions,
                    lookback,
                    cost_estimate,
                });
            }
        }
        result
    }

    /// Simulate a regime-switching strategy:
    /// Take position `long_size` in bull regime, `short_size` in bear regime, 0 in neutral.
    pub fn simulate_strategy(
        &self,
        returns: &[f64],
        states: &[usize],
        strategies: &[RegimeStrategy],
    ) -> BacktestResult {
        let n = returns.len();
        assert_eq!(n, states.len());

        // Build position vector.
        let mut positions = vec![0.0f64; n];
        for (t, &s) in states.iter().enumerate() {
            if let Some(strat) = strategies.iter().find(|st| st.regime == s) {
                if strat.active {
                    positions[t] = strat.position_size;
                }
            }
        }

        // Apply transaction costs.
        let mut daily_pnls = vec![0.0f64; n];
        let mut prev_pos = 0.0f64;
        for t in 0..n {
            let gross = returns[t] * positions[t];
            let tc = (positions[t] - prev_pos).abs() * self.transaction_cost;
            daily_pnls[t] = gross - tc;
            prev_pos = positions[t];
        }

        let n_f = n as f64;
        let mean = daily_pnls.iter().sum::<f64>() / n_f;
        let std = (daily_pnls.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n_f)
            .sqrt()
            .max(1e-10);
        let sharpe = mean / std * (252.0f64).sqrt();
        let cum_pnl: f64 = daily_pnls.iter().sum();
        let max_dd = compute_max_drawdown_vec(&daily_pnls);
        let calmar = if max_dd > 1e-10 { mean * 252.0 / max_dd } else { 0.0 };

        // Compute regime P&L.
        let regime_pnl = self.compute_regime_pnl(returns, states, &positions);

        BacktestResult {
            daily_pnls,
            positions,
            total_mean_pnl: mean,
            total_std_pnl: std,
            total_sharpe: sharpe,
            total_cum_pnl: cum_pnl,
            max_drawdown: max_dd,
            calmar,
            regime_pnl,
        }
    }

    /// Determine optimal per-regime strategy: long, short, or flat based on regime Sharpe.
    pub fn optimal_strategies(&self, pnl_stats: &[RegimePnl]) -> Vec<RegimeStrategy> {
        pnl_stats
            .iter()
            .map(|p| {
                let position_size = if p.sharpe > 0.5 {
                    1.0
                } else if p.sharpe < -0.5 {
                    -1.0
                } else {
                    0.0
                };
                RegimeStrategy {
                    regime: p.regime,
                    position_size,
                    sharpe_threshold: 0.5,
                    active: position_size.abs() > 0.01,
                }
            })
            .collect()
    }
}

/// Full backtest result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub daily_pnls: Vec<f64>,
    pub positions: Vec<f64>,
    pub total_mean_pnl: f64,
    pub total_std_pnl: f64,
    pub total_sharpe: f64,
    pub total_cum_pnl: f64,
    pub max_drawdown: f64,
    pub calmar: f64,
    pub regime_pnl: Vec<RegimePnl>,
}

fn compute_max_drawdown_vec(pnls: &[f64]) -> f64 {
    let mut cum = 0.0f64;
    let mut peak = 0.0f64;
    let mut max_dd = 0.0f64;
    for &p in pnls {
        cum += p;
        if cum > peak {
            peak = cum;
        }
        let dd = if peak > 0.0 { (peak - cum) / peak } else { 0.0 };
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_two_regime_data(n: usize) -> (Vec<f64>, Vec<usize>) {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(2025);
        let states: Vec<usize> = (0..n).map(|i| i / (n / 2)).collect();
        let returns: Vec<f64> = states
            .iter()
            .map(|&s| {
                if s == 0 {
                    0.001 + rng.gen::<f64>() * 0.01
                } else {
                    -0.001 + rng.gen::<f64>() * 0.02 - 0.01
                }
            })
            .collect();
        (returns, states)
    }

    #[test]
    fn test_regime_pnl_computed() {
        let (returns, states) = make_two_regime_data(400);
        let positions: Vec<f64> = vec![1.0; 400];
        let bt = RegimeBacktest::new(2, None, 0.0);
        let pnls = bt.compute_regime_pnl(&returns, &states, &positions);
        assert_eq!(pnls.len(), 2);
        assert!(pnls[0].n_bars > 0);
        assert!(pnls[1].n_bars > 0);
    }

    #[test]
    fn test_strategy_simulation() {
        let (returns, states) = make_two_regime_data(400);
        let bt = RegimeBacktest::new(2, None, 0.001);
        let positions = vec![1.0; 400];
        let stats = bt.compute_regime_pnl(&returns, &states, &positions);
        let strategies = bt.optimal_strategies(&stats);
        let result = bt.simulate_strategy(&returns, &states, &strategies);
        assert_eq!(result.daily_pnls.len(), 400);
        assert!(result.total_sharpe.is_finite());
    }

    #[test]
    fn test_transition_costs_computed() {
        let (returns, states) = make_two_regime_data(400);
        let bt = RegimeBacktest::new(2, None, 0.001);
        let costs = bt.transition_costs(&returns, &states, 5);
        // Should have one transition from 0->1.
        assert!(!costs.is_empty());
        for c in &costs {
            assert!(c.n_transitions > 0);
        }
    }

    #[test]
    fn test_sharpe_differences_across_regimes() {
        let (returns, states) = make_two_regime_data(400);
        let positions = vec![1.0; 400];
        let bt = RegimeBacktest::new(2, None, 0.0);
        let pnls = bt.compute_regime_pnl(&returns, &states, &positions);
        // Regime 0 (positive returns) should have higher Sharpe.
        assert!(pnls[0].sharpe > pnls[1].sharpe, "Regime 0 should have better Sharpe");
    }
}
