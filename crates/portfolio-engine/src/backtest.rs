/// Portfolio-level backtesting engine.

use crate::covariance::Matrix;
use crate::risk::{max_drawdown, sharpe_ratio, sortino_ratio, calmar_ratio, ulcer_index};

// ── Rebalance Frequency ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebalanceFreq {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
}

impl RebalanceFreq {
    pub fn bars_between(&self) -> usize {
        match self {
            RebalanceFreq::Daily => 1,
            RebalanceFreq::Weekly => 5,
            RebalanceFreq::Monthly => 21,
            RebalanceFreq::Quarterly => 63,
        }
    }
}

// ── Result ────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct PortfolioMetrics {
    pub total_return: f64,
    pub annualised_return: f64,
    pub annualised_volatility: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub calmar_ratio: f64,
    pub max_drawdown: f64,
    pub ulcer_index: f64,
    pub total_turnover: f64,
    pub avg_daily_turnover: f64,
}

#[derive(Debug)]
pub struct PortfolioBacktestResult {
    pub equity_curve: Vec<f64>,
    pub daily_returns: Vec<f64>,
    pub weights_history: Vec<Vec<f64>>,
    pub turnover_series: Vec<f64>,
    pub metrics: PortfolioMetrics,
}

// ── Main Backtest Runner ──────────────────────────────────────────────────────

/// Run a portfolio backtest.
///
/// # Arguments
/// * `universe_returns` — T × N matrix of asset returns (rows = time, cols = assets).
/// * `rebalance_fn` — closure that receives the T_so_far × N returns matrix seen so far
///   and returns the new target weights (N-vector).
/// * `freq` — how often to rebalance.
/// * `initial_capital` — starting portfolio value.
pub fn run<F>(
    universe_returns: &Matrix,
    rebalance_fn: F,
    freq: RebalanceFreq,
    initial_capital: f64,
) -> PortfolioBacktestResult
where
    F: Fn(&Matrix) -> Vec<f64>,
{
    let t = universe_returns.len();
    let n = universe_returns.first().map_or(0, |r| r.len());
    if t == 0 || n == 0 {
        return PortfolioBacktestResult {
            equity_curve: vec![initial_capital],
            daily_returns: vec![],
            weights_history: vec![],
            turnover_series: vec![],
            metrics: PortfolioMetrics {
                total_return: 0.0, annualised_return: 0.0, annualised_volatility: 0.0,
                sharpe_ratio: 0.0, sortino_ratio: 0.0, calmar_ratio: 0.0,
                max_drawdown: 0.0, ulcer_index: 0.0, total_turnover: 0.0, avg_daily_turnover: 0.0,
            },
        };
    }

    let bars_between = freq.bars_between();
    let mut equity = initial_capital;
    let mut equity_curve = vec![equity];
    let mut daily_returns: Vec<f64> = Vec::with_capacity(t);
    let mut weights_history: Vec<Vec<f64>> = Vec::with_capacity(t / bars_between + 1);
    let mut turnover_series: Vec<f64> = Vec::with_capacity(t / bars_between);

    // Start with equal weights.
    let mut current_weights: Vec<f64> = vec![1.0 / n as f64; n];
    let mut prev_weights: Vec<f64> = current_weights.clone();

    for bar in 0..t {
        // Rebalance if it's time (and we have at least a lookback worth of data).
        if bar > 0 && bar % bars_between == 0 {
            let history = universe_returns[..bar].to_vec();
            let new_weights = rebalance_fn(&history);
            // Compute turnover.
            let turn: f64 = new_weights
                .iter()
                .zip(current_weights.iter())
                .map(|(nw, cw)| (nw - cw).abs())
                .sum::<f64>()
                / 2.0;
            turnover_series.push(turn);
            weights_history.push(new_weights.clone());
            prev_weights = current_weights.clone();
            current_weights = new_weights;
        }

        // Apply returns for this bar.
        let bar_return: f64 = universe_returns[bar]
            .iter()
            .zip(current_weights.iter())
            .map(|(r, w)| r * w)
            .sum();

        equity *= 1.0 + bar_return;
        equity_curve.push(equity);
        daily_returns.push(bar_return);

        // Drift weights with realised returns (buy-and-hold between rebalances).
        let new_w: Vec<f64> = (0..n)
            .map(|i| current_weights[i] * (1.0 + universe_returns[bar][i]))
            .collect();
        let w_sum: f64 = new_w.iter().sum::<f64>().max(1e-12);
        current_weights = new_w.iter().map(|w| w / w_sum).collect();
    }

    let total_ret = (equity - initial_capital) / initial_capital;
    let t_f = t as f64;
    let ann_ret = (1.0 + total_ret).powf(252.0 / t_f) - 1.0;
    let ann_vol = {
        let m = daily_returns.iter().sum::<f64>() / t_f;
        let var = daily_returns.iter().map(|r| (r - m).powi(2)).sum::<f64>() / t_f;
        var.sqrt() * 252_f64.sqrt()
    };
    let sr = sharpe_ratio(&daily_returns, 0.0);
    let srt = sortino_ratio(&daily_returns, 0.0);
    let mdd = max_drawdown(&equity_curve);
    let cal = calmar_ratio(&daily_returns, mdd);
    let ui = ulcer_index(&equity_curve);
    let total_turnover: f64 = turnover_series.iter().sum();
    let avg_to = if turnover_series.is_empty() { 0.0 } else { total_turnover / turnover_series.len() as f64 };

    PortfolioBacktestResult {
        equity_curve,
        daily_returns,
        weights_history,
        turnover_series,
        metrics: PortfolioMetrics {
            total_return: total_ret,
            annualised_return: ann_ret,
            annualised_volatility: ann_vol,
            sharpe_ratio: sr,
            sortino_ratio: srt,
            calmar_ratio: cal,
            max_drawdown: mdd,
            ulcer_index: ui,
            total_turnover,
            avg_daily_turnover: avg_to,
        },
    }
}

// ── Transaction Cost Impact ───────────────────────────────────────────────────

/// Compute the total transaction cost drag on the portfolio.
///
/// * `weights_history` — list of weight vectors at each rebalance.
/// * `bid_ask_spreads` — per-asset half-spread (fraction of price) vectors, one per rebalance.
/// Returns total cost fraction.
pub fn transaction_cost_impact(
    weights_history: &[Vec<f64>],
    bid_ask_spreads: &[Vec<f64>],
) -> f64 {
    weights_history
        .windows(2)
        .enumerate()
        .map(|(i, w)| {
            let prev = &w[0];
            let next = &w[1];
            let spreads = bid_ask_spreads.get(i).cloned().unwrap_or_else(|| vec![0.001; prev.len()]);
            prev.iter()
                .zip(next.iter())
                .zip(spreads.iter())
                .map(|((pw, nw), s)| (nw - pw).abs() * s)
                .sum::<f64>()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_returns_uniform(t: usize, n: usize, daily_ret: f64) -> Matrix {
        vec![vec![daily_ret; n]; t]
    }

    #[test]
    fn backtest_equity_grows_with_positive_returns() {
        let returns = mock_returns_uniform(252, 5, 0.001);
        let result = run(&returns, |_| vec![0.2; 5], RebalanceFreq::Monthly, 1_000_000.0);
        assert!(result.equity_curve.last().copied().unwrap_or(0.0) > 1_000_000.0);
    }

    #[test]
    fn backtest_equity_length_correct() {
        let t = 100;
        let returns = mock_returns_uniform(t, 3, 0.0);
        let result = run(&returns, |_| vec![1.0 / 3.0; 3], RebalanceFreq::Daily, 100.0);
        assert_eq!(result.equity_curve.len(), t + 1);
    }
}
