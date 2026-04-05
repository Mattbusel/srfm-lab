use crate::bar_data::{BarData, DataStore};
use crate::bh_engine::{BHState, GARCHState, OUState};
use crate::params::StrategyParams;
use crate::portfolio::Portfolio;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

const TRANSACTION_COST: f64 = 0.0007; // 7 bps per side (taker + slippage)
const BARS_PER_YEAR: f64 = 35040.0; // 15-min bars in a year

/// Summary statistics from a single backtest run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub final_equity: f64,
    pub cagr: f64,
    pub sharpe: f64,
    pub max_drawdown: f64,
    pub total_trades: u64,
    pub win_rate: f64,
    pub profit_factor: f64,
    /// Annual P&L keyed by calendar year (unix year from timestamp).
    pub annual_pnl: HashMap<i32, f64>,
}

impl Default for BacktestResult {
    fn default() -> Self {
        Self {
            final_equity: 1.0,
            cagr: 0.0,
            sharpe: f64::NEG_INFINITY,
            max_drawdown: 1.0,
            total_trades: 0,
            win_rate: 0.0,
            profit_factor: 0.0,
            annual_pnl: HashMap::new(),
        }
    }
}

/// Run a complete backtest of the BH strategy with the given parameters
/// against the provided OHLCV data.
///
/// Returns a `BacktestResult` summarising performance.
pub fn run_backtest(data: &DataStore, params: &StrategyParams) -> BacktestResult {
    if data.is_empty() {
        return BacktestResult::default();
    }

    // ── Align bars across symbols ──────────────────────────────────────────
    // Build a unified sorted list of timestamps present in *any* symbol.
    let mut all_timestamps: Vec<i64> = data
        .values()
        .flat_map(|bars| bars.iter().map(|b| b.timestamp))
        .collect();
    all_timestamps.sort_unstable();
    all_timestamps.dedup();

    // Build per-symbol bar lookup.
    let bar_lookup: HashMap<String, HashMap<i64, &BarData>> = data
        .iter()
        .map(|(sym, bars)| {
            let lut: HashMap<i64, &BarData> = bars.iter().map(|b| (b.timestamp, b)).collect();
            (sym.clone(), lut)
        })
        .collect();

    // ── State machines ─────────────────────────────────────────────────────
    let symbols: Vec<String> = data.keys().cloned().collect();
    let mut bh_states: HashMap<String, BHState> = symbols
        .iter()
        .map(|s| (s.clone(), BHState::new(params.stale_15m_move * 5.0)))
        .collect();
    let mut garch_states: HashMap<String, GARCHState> = symbols
        .iter()
        .map(|s| (s.clone(), GARCHState::default_crypto()))
        .collect();
    let mut ou_states: HashMap<String, OUState> = symbols
        .iter()
        .map(|s| (s.clone(), OUState::new(50)))
        .collect();

    // Rolling daily returns (approximate with per-bar returns, 96 bars = 1 day).
    let mut recent_returns: HashMap<String, Vec<f64>> = symbols
        .iter()
        .map(|s| (s.clone(), Vec::with_capacity(256)))
        .collect();
    // Smoothed daily (96-bar) return accumulator.
    let mut bar_count_for_day: u32 = 0;
    let mut day_open_prices: HashMap<String, f64> = HashMap::new();

    let mut portfolio = Portfolio::new(1.0);
    let mut prev_equity = 1.0_f64;

    // Trade tracking.
    let mut total_trades: u64 = 0;
    let mut winning_trades: u64 = 0;
    let mut gross_profit = 0.0_f64;
    let mut gross_loss = 0.0_f64;

    // Annual P&L tracking.
    let mut annual_start: HashMap<i32, f64> = HashMap::new();

    const GARCH_WARMUP: usize = 20;

    for &ts in &all_timestamps {
        // Gather current prices.
        let curr_prices: HashMap<String, f64> = symbols
            .iter()
            .filter_map(|sym| {
                bar_lookup[sym].get(&ts).map(|b| (sym.clone(), b.close))
            })
            .collect();

        if curr_prices.is_empty() {
            continue;
        }

        // Update state machines.
        for sym in &symbols {
            if let Some(&price) = curr_prices.get(sym) {
                let bh = bh_states.get_mut(sym).unwrap();

                // Fast path: use fixed vol for GARCH warmup period.
                let garch = garch_states.get_mut(sym).unwrap();
                if garch.n >= GARCH_WARMUP {
                    bh.cf_scale = 1.0 / garch.inv_vol_scale(params.garch_target_vol).max(0.1);
                }

                bh.update(price);

                // Compute log return for GARCH.
                if let Some(prev_bars) = data.get(sym) {
                    if let Ok(pos) = prev_bars.binary_search_by_key(&ts, |b| b.timestamp) {
                        if pos > 0 {
                            let prev_close = prev_bars[pos - 1].close;
                            let ret = (price / prev_close).ln();
                            garch.update(ret);
                            let rets = recent_returns.get_mut(sym).unwrap();
                            rets.push(ret);
                            if rets.len() > 2000 {
                                rets.drain(0..500);
                            }
                        }
                    }
                }

                ou_states.get_mut(sym).unwrap().push(price);
            }
        }

        // Accumulate daily returns (every 96 bars ≈ 1 day for 15-min data).
        bar_count_for_day += 1;
        if bar_count_for_day == 1 {
            for (sym, &price) in &curr_prices {
                day_open_prices.insert(sym.clone(), price);
            }
        }
        let mut daily_returns: HashMap<String, Vec<f64>> = HashMap::new();
        if bar_count_for_day >= 96 {
            for (sym, &close) in &curr_prices {
                let open = *day_open_prices.get(sym).unwrap_or(&close);
                let day_ret = (close / open).ln();
                daily_returns.entry(sym.clone()).or_default().push(day_ret);
            }
            bar_count_for_day = 0;
        }
        // Use recent bar-level returns as proxy for daily returns when needed.
        for (sym, rets) in &recent_returns {
            daily_returns.entry(sym.clone()).or_insert_with(|| rets.clone());
        }

        // UTC hour from timestamp.
        let hour_utc = ((ts / 3600) % 24) as u8;

        // Force-close winners per protection rule.
        let force_close = portfolio.tick_positions(&curr_prices, params);
        for sym in force_close {
            if let Some(&frac) = portfolio.positions.get(&sym) {
                let price = curr_prices.get(&sym).copied().unwrap_or(1.0);
                let entry = portfolio.entry_prices.get(&sym).copied().unwrap_or(price);
                let pnl = (price - entry) / entry * frac.signum();
                if pnl > 0.0 {
                    winning_trades += 1;
                    gross_profit += pnl * frac.abs() * portfolio.equity;
                } else {
                    gross_loss += pnl.abs() * frac.abs() * portfolio.equity;
                }
                total_trades += 1;
                portfolio.positions.remove(&sym);
                portfolio.entry_prices.remove(&sym);
                portfolio.bars_held.remove(&sym);
            }
        }

        // Compute new targets.
        let targets = portfolio.compute_targets(
            &bh_states,
            &garch_states,
            &ou_states,
            params,
            &curr_prices,
            hour_utc,
            &daily_returns,
        );

        // Count trades (position changes).
        for (sym, &target) in &targets {
            let current = portfolio.positions.get(sym).copied().unwrap_or(0.0);
            if (target - current).abs() > portfolio.pos_floor {
                total_trades += 1;
                // Closed positions: record win/loss.
                if current.abs() > portfolio.pos_floor {
                    let price = curr_prices.get(sym).copied().unwrap_or(1.0);
                    let entry = portfolio.entry_prices.get(sym).copied().unwrap_or(price);
                    let pnl = (price - entry) / entry * current.signum();
                    if pnl > 0.0 {
                        winning_trades += 1;
                        gross_profit += pnl * current.abs() * portfolio.equity;
                    } else {
                        gross_loss += pnl.abs() * current.abs() * portfolio.equity;
                    }
                }
            }
        }

        portfolio.apply_targets(&targets, &curr_prices, TRANSACTION_COST);

        // Mark-to-market open positions.
        let mut mtm_pnl = 0.0_f64;
        for (sym, &frac) in &portfolio.positions {
            if let Some(&price) = curr_prices.get(sym) {
                let entry = portfolio.entry_prices.get(sym).copied().unwrap_or(price);
                mtm_pnl += frac * (price - entry) / entry;
            }
        }
        // Apply MTM as equity adjustment (approximate: fracs don't sum to 1 necessarily).
        portfolio.equity += portfolio.equity * mtm_pnl * 0.01; // damped MTM
        portfolio.equity = portfolio.equity.max(1e-6);

        // Annual P&L tracking.
        let year = ts_to_year(ts);
        annual_start.entry(year).or_insert(prev_equity);
        prev_equity = portfolio.equity;
    }

    // ── Compute summary statistics ─────────────────────────────────────────
    let curve = &portfolio.equity_curve;
    let n_bars = curve.len() as f64;
    let years = n_bars / BARS_PER_YEAR;

    let final_equity = portfolio.equity;
    let cagr = if years > 0.0 && final_equity > 0.0 {
        (final_equity / 1.0_f64).powf(1.0 / years) - 1.0
    } else {
        0.0
    };

    let sharpe = compute_sharpe(curve);
    let max_drawdown = compute_max_drawdown(curve);

    let win_rate = if total_trades > 0 {
        winning_trades as f64 / total_trades as f64
    } else {
        0.0
    };
    let profit_factor = if gross_loss > 1e-10 {
        gross_profit / gross_loss
    } else if gross_profit > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    // Compute annual P&L.
    let mut annual_pnl: HashMap<i32, f64> = HashMap::new();
    if let Some(&last_ts) = all_timestamps.last() {
        let last_year = ts_to_year(last_ts);
        if let Some(&start) = annual_start.get(&last_year) {
            annual_pnl.insert(last_year, (final_equity - start) / start);
        }
    }
    for (&year, &start) in &annual_start {
        annual_pnl.entry(year).or_insert_with(|| {
            // approximate year end equity from curve fraction
            (final_equity - start) / start.max(1e-10)
        });
    }

    BacktestResult {
        final_equity,
        cagr,
        sharpe,
        max_drawdown,
        total_trades,
        win_rate,
        profit_factor,
        annual_pnl,
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn compute_sharpe(equity_curve: &[f64]) -> f64 {
    if equity_curve.len() < 5 {
        return f64::NEG_INFINITY;
    }
    let rets: Vec<f64> = equity_curve
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();
    let n = rets.len() as f64;
    let mean = rets.iter().sum::<f64>() / n;
    let var = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let std = var.sqrt();
    if std < 1e-12 {
        return 0.0;
    }
    // Annualise: 35040 bars/year.
    mean / std * BARS_PER_YEAR.sqrt()
}

fn compute_max_drawdown(equity_curve: &[f64]) -> f64 {
    let mut peak = equity_curve.first().copied().unwrap_or(1.0);
    let mut max_dd = 0.0_f64;
    for &e in equity_curve {
        if e > peak {
            peak = e;
        }
        let dd = (peak - e) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

fn ts_to_year(ts: i64) -> i32 {
    // Rough: 1970 + seconds / seconds_per_year.
    1970 + (ts / 31_557_600) as i32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bar_data::BarData;

    fn synthetic_trend_data(n_bars: usize, start_price: f64, daily_drift: f64) -> DataStore {
        let mut data = DataStore::new();
        let bars: Vec<BarData> = (0..n_bars)
            .map(|i| {
                let t = 1_600_000_000_i64 + (i as i64) * 900; // 15-min bars
                let price = start_price * (1.0 + daily_drift).powf(i as f64 / 96.0);
                BarData {
                    symbol: "BTC".to_string(),
                    timestamp: t,
                    open: price * 0.999,
                    high: price * 1.002,
                    low: price * 0.997,
                    close: price,
                    volume: 10.0,
                }
            })
            .collect();
        data.insert("BTC".to_string(), bars);
        data
    }

    #[test]
    fn test_backtest_returns_result() {
        let data = synthetic_trend_data(500, 50000.0, 0.001);
        let params = StrategyParams::default();
        let result = run_backtest(&data, &params);
        // Should complete without panic and return a valid result.
        assert!(result.final_equity > 0.0);
        assert!(!result.max_drawdown.is_nan());
    }

    #[test]
    fn test_backtest_empty_data() {
        let data = DataStore::new();
        let params = StrategyParams::default();
        let result = run_backtest(&data, &params);
        assert_eq!(result.final_equity, 1.0);
    }

    #[test]
    fn test_sharpe_flat_equity() {
        let curve: Vec<f64> = vec![1.0; 100];
        let s = compute_sharpe(&curve);
        assert_eq!(s, 0.0);
    }

    #[test]
    fn test_max_drawdown_monotone_up() {
        let curve: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let dd = compute_max_drawdown(&curve);
        assert_eq!(dd, 0.0);
    }

    #[test]
    fn test_max_drawdown_50pct() {
        let curve = vec![1.0, 2.0, 1.0]; // drops 50% from peak.
        let dd = compute_max_drawdown(&curve);
        assert!((dd - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_backtest_win_rate_range() {
        let data = synthetic_trend_data(2000, 50000.0, 0.0005);
        let params = StrategyParams::default();
        let result = run_backtest(&data, &params);
        assert!((0.0..=1.0).contains(&result.win_rate));
    }
}
