// engine.rs — Bar-by-bar backtest engine with multi-timeframe BH physics
//
// Execution model
// ───────────────
// The engine iterates 15-minute bars as the base timeframe.  For each 15-m
// bar it looks up the aligned 1-h and 1-d bars (by timestamp ≤ current).
// An entry fires when the BH becomes active on *any* timeframe AND the
// DeltaScore clears a minimum threshold.  The position is sized as:
//
//   dollar_pos = equity × max_position_frac × delta_score.value
//
// Exits:
//   1. BH goes inactive on ALL timeframes.
//   2. Price hits the stop-loss (entry − stop_atr_mult × ATR).
//   3. Price hits the take-profit (entry + tp_atr_mult × ATR).
//   4. min_hold_bars not yet reached → exits suppressed.
//
// P&L accounts for round-trip transaction costs in basis points.

use crate::bh_physics::BHState;
use crate::csv_loader::BarsSet;
use crate::indicators::{ATR, EMA, RegimeClassifier};
use crate::types::{
    BacktestMetrics, Bar, DeltaScore, Position, RegimeStats, Regime, TFScore, Trade,
};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// BacktestConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestConfig {
    pub sym: String,
    /// Speed-of-light parameter for BH physics.
    pub cf: f64,
    /// BH activation mass threshold.
    pub bh_form: f64,
    /// BH deactivation mass threshold.
    pub bh_collapse: f64,
    /// Per-bar mass decay factor for spacelike moves.
    pub bh_decay: f64,
    /// Starting account equity (dollars).
    pub starting_equity: f64,
    /// Maximum fraction of equity per position (0–1).
    pub max_position_frac: f64,
    /// One-way transaction cost in basis points.
    pub transaction_cost_bps: f64,
    /// Stop-loss multiplier of ATR (None = no stop).
    pub stop_loss_atr_mult: Option<f64>,
    /// Take-profit multiplier of ATR (None = no TP).
    pub take_profit_atr_mult: Option<f64>,
    /// Minimum bars held before an exit is allowed.
    pub min_hold_bars: usize,
    /// Minimum DeltaScore.value required for entry.
    pub min_delta_score: f64,
}

impl BacktestConfig {
    pub fn default_for(sym: impl Into<String>) -> Self {
        Self {
            sym: sym.into(),
            cf: 0.001,
            bh_form: 1.5,
            bh_collapse: 1.2,
            bh_decay: 0.95,
            starting_equity: 1_000_000.0,
            max_position_frac: 0.10,
            transaction_cost_bps: 2.0,
            stop_loss_atr_mult: Some(2.0),
            take_profit_atr_mult: Some(4.0),
            min_hold_bars: 2,
            min_delta_score: 0.10,
        }
    }
}

// ---------------------------------------------------------------------------
// BacktestResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacktestResult {
    pub trades: Vec<Trade>,
    /// (unix_ms, equity) pairs at each exit bar.
    pub equity_curve: Vec<(i64, f64)>,
    pub final_equity: f64,
    pub metrics: BacktestMetrics,
    pub regime_breakdown: HashMap<String, RegimeStats>,
}

impl BacktestResult {
    pub fn empty(starting_equity: f64) -> Self {
        Self {
            trades: vec![],
            equity_curve: vec![],
            final_equity: starting_equity,
            metrics: BacktestMetrics::default(),
            regime_breakdown: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Internal open-position tracking
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct OpenPosition {
    position: Position,
    entry_time: i64,
    stop_price: Option<f64>,
    tp_price: Option<f64>,
    hold_bars: usize,
    entry_regime: Regime,
    entry_tf_score: TFScore,
    entry_mass: f64,
}

// ---------------------------------------------------------------------------
// BacktestEngine
// ---------------------------------------------------------------------------

pub struct BacktestEngine {
    pub config: BacktestConfig,
}

impl BacktestEngine {
    pub fn new(config: BacktestConfig) -> Self {
        Self { config }
    }

    /// Run the full bar-by-bar simulation.
    pub fn run(
        &mut self,
        bars_1d: &[Bar],
        bars_1h: &[Bar],
        bars_15m: &[Bar],
    ) -> Result<BacktestResult> {
        if bars_15m.is_empty() {
            return Ok(BacktestResult::empty(self.config.starting_equity));
        }

        let cfg = &self.config;

        // BH state for each timeframe
        let mut bh_1d = BHState::new(cfg.cf, cfg.bh_form, cfg.bh_collapse, cfg.bh_decay);
        let mut bh_1h = BHState::new(cfg.cf, cfg.bh_form, cfg.bh_collapse, cfg.bh_decay);

        // Pre-warm daily and hourly states on all their bars
        let mut bh_1d_by_ts: Vec<(i64, crate::bh_physics::BHUpdate)> =
            Vec::with_capacity(bars_1d.len());
        for bar in bars_1d {
            let upd = bh_1d.update(bar.close);
            bh_1d_by_ts.push((bar.timestamp, upd));
        }

        let mut bh_1h_by_ts: Vec<(i64, crate::bh_physics::BHUpdate)> =
            Vec::with_capacity(bars_1h.len());
        for bar in bars_1h {
            let upd = bh_1h.update(bar.close);
            bh_1h_by_ts.push((bar.timestamp, upd));
        }

        // 15m BH — created fresh after daily/hourly pre-warm
        let mut bh_15m = BHState::new(cfg.cf, cfg.bh_form, cfg.bh_collapse, cfg.bh_decay);

        // Indicators
        let mut atr_15m = ATR::new(14);
        let mut regime_clf = RegimeClassifier::new(0.025);
        let mut ema50_15m = EMA::new(50);

        // State
        let mut equity = cfg.starting_equity;
        let mut open_pos: Option<OpenPosition> = None;
        let mut trades: Vec<Trade> = Vec::new();
        let mut equity_curve: Vec<(i64, f64)> = vec![(bars_15m[0].timestamp, equity)];
        let mut regime_breakdown: HashMap<String, RegimeStats> = HashMap::new();

        for (i, bar) in bars_15m.iter().enumerate() {
            let ts = bar.timestamp;

            // Update 15 m indicators
            let atr_val = atr_15m.update(bar.high, bar.low, bar.close);
            let regime = regime_clf.update(bar.high, bar.low, bar.close);
            let _ema50 = ema50_15m.update(bar.close);

            // Update 15 m BH
            let upd_15m = bh_15m.update(bar.close);

            // Look up most-recent daily and hourly BH updates
            let upd_1d = latest_update_before(&bh_1d_by_ts, ts);
            let upd_1h = latest_update_before(&bh_1h_by_ts, ts);

            // Compute TF score components
            let score_1d = bh_signal_score(upd_1d);
            let score_1h = bh_signal_score(upd_1h);
            let score_15m = bh_signal_score(Some(upd_15m));
            let tf_score = TFScore::new(score_1d, score_1h, score_15m);

            let mass_15m = upd_15m.mass;
            let delta_score = DeltaScore::new(tf_score, mass_15m, atr_val);

            // ── Exit logic ──────────────────────────────────────────────
            if let Some(ref mut pos) = open_pos {
                pos.hold_bars += 1;
                let can_exit = pos.hold_bars >= cfg.min_hold_bars;

                let bh_all_dead = !upd_15m.active
                    && upd_1h.map(|u| !u.active).unwrap_or(true)
                    && upd_1d.map(|u| !u.active).unwrap_or(true);

                let stop_hit = pos
                    .stop_price
                    .map(|s| bar.low <= s)
                    .unwrap_or(false);
                let tp_hit = pos
                    .tp_price
                    .map(|t| bar.high >= t)
                    .unwrap_or(false);

                if can_exit && (bh_all_dead || stop_hit || tp_hit) {
                    // Determine exit price
                    let exit_price = if stop_hit {
                        pos.stop_price.unwrap()
                    } else if tp_hit {
                        pos.tp_price.unwrap()
                    } else {
                        bar.close
                    };

                    let trade = close_position(
                        pos,
                        exit_price,
                        ts,
                        cfg.transaction_cost_bps,
                        i,
                    );

                    equity += trade.pnl;
                    equity_curve.push((ts, equity));

                    let rs = regime_breakdown
                        .entry(trade.regime.to_str().to_string())
                        .or_default();
                    rs.add_trade(&trade);

                    trades.push(trade);
                    open_pos = None;
                }
            }

            // ── Entry logic ─────────────────────────────────────────────
            if open_pos.is_none() {
                let any_bh_active = upd_15m.active
                    || upd_1h.map(|u| u.active).unwrap_or(false)
                    || upd_1d.map(|u| u.active).unwrap_or(false);

                if any_bh_active && delta_score.value >= cfg.min_delta_score {
                    let entry_price = bar.close;
                    let dollar_pos =
                        equity * cfg.max_position_frac * delta_score.value.min(1.0);
                    let shares = if entry_price > 0.0 {
                        dollar_pos / entry_price
                    } else {
                        0.0
                    };

                    let stop_price = if let (Some(mult), false) = (cfg.stop_loss_atr_mult, atr_val.is_nan()) {
                        Some(entry_price - mult * atr_val)
                    } else {
                        None
                    };
                    let tp_price = if let (Some(mult), false) = (cfg.take_profit_atr_mult, atr_val.is_nan()) {
                        Some(entry_price + mult * atr_val)
                    } else {
                        None
                    };

                    let pos = Position::new(&cfg.sym, shares, entry_price);
                    open_pos = Some(OpenPosition {
                        position: pos,
                        entry_time: ts,
                        stop_price,
                        tp_price,
                        hold_bars: 0,
                        entry_regime: regime,
                        entry_tf_score: tf_score,
                        entry_mass: mass_15m,
                    });
                }
            }
        }

        // Force-close any open position at the last bar
        if let Some(ref mut pos) = open_pos {
            let last_bar = bars_15m.last().unwrap();
            let trade = close_position(
                pos,
                last_bar.close,
                last_bar.timestamp,
                cfg.transaction_cost_bps,
                bars_15m.len(),
            );
            equity += trade.pnl;
            equity_curve.push((last_bar.timestamp, equity));
            let rs = regime_breakdown
                .entry(trade.regime.to_str().to_string())
                .or_default();
            rs.add_trade(&trade);
            trades.push(trade);
        }

        let metrics = compute_metrics(&trades, &equity_curve, cfg.starting_equity);

        Ok(BacktestResult {
            final_equity: equity,
            equity_curve,
            trades,
            metrics,
            regime_breakdown,
        })
    }

    /// Convenience: run from a `BarsSet`.
    pub fn run_barsset(&mut self, bars: &BarsSet) -> Result<BacktestResult> {
        self.run(&bars.bars_1d, &bars.bars_1h, &bars.bars_15m)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert an Option<BHUpdate> into a 0/1 signal score (active = 1.0).
fn bh_signal_score(upd: Option<crate::bh_physics::BHUpdate>) -> f64 {
    upd.map(|u| if u.active { u.mass / 2.0 } else { 0.0 })
        .unwrap_or(0.0)
}

/// Binary search for the latest BH update whose timestamp ≤ `ts`.
fn latest_update_before(
    updates: &[(i64, crate::bh_physics::BHUpdate)],
    ts: i64,
) -> Option<crate::bh_physics::BHUpdate> {
    if updates.is_empty() {
        return None;
    }
    let idx = updates.partition_point(|(t, _)| *t <= ts);
    if idx == 0 {
        None
    } else {
        Some(updates[idx - 1].1)
    }
}

/// Build a `Trade` from an open position being closed.
fn close_position(
    pos: &OpenPosition,
    exit_price: f64,
    exit_time: i64,
    tc_bps: f64,
    _bar_idx: usize,
) -> Trade {
    let entry_price = pos.position.avg_cost;
    let shares = pos.position.size;
    let dollar_pos = shares * entry_price;

    // Gross PnL
    let gross_pnl = (exit_price - entry_price) * shares;

    // Round-trip transaction cost
    let tc = dollar_pos * tc_bps / 10_000.0 * 2.0;
    let net_pnl = gross_pnl - tc;

    Trade::new(
        pos.position.sym.clone(),
        pos.entry_time,
        exit_time,
        entry_price,
        exit_price,
        net_pnl,
        dollar_pos,
        pos.hold_bars,
        pos.entry_regime,
        pos.entry_tf_score,
        pos.entry_mass,
    )
}

// ---------------------------------------------------------------------------
// compute_metrics
// ---------------------------------------------------------------------------

/// Compute standard backtest performance metrics from a completed run.
pub fn compute_metrics(
    trades: &[Trade],
    equity_curve: &[(i64, f64)],
    starting_equity: f64,
) -> BacktestMetrics {
    if trades.is_empty() || equity_curve.is_empty() {
        return BacktestMetrics::default();
    }

    let n = trades.len();
    let winners: usize = trades.iter().filter(|t| t.is_winner()).count();
    let win_rate = winners as f64 / n as f64;

    let gross_profit: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
    let gross_loss: f64 = trades.iter().filter(|t| t.pnl < 0.0).map(|t| t.pnl.abs()).sum();
    let profit_factor = if gross_loss < 1e-9 {
        f64::INFINITY
    } else {
        gross_profit / gross_loss
    };

    let avg_hold_bars =
        trades.iter().map(|t| t.hold_bars as f64).sum::<f64>() / n as f64;

    let avg_return_per_trade =
        trades.iter().map(|t| t.return_frac()).sum::<f64>() / n as f64;

    // Sharpe: annualise daily returns from equity curve
    let returns = equity_returns(equity_curve);
    let sharpe = annualised_sharpe(&returns, 252.0);

    // Max drawdown
    let max_drawdown = max_drawdown_frac(equity_curve);

    // CAGR
    let final_eq = equity_curve.last().map(|(_, e)| *e).unwrap_or(starting_equity);
    let cagr = if equity_curve.len() >= 2 {
        let t0 = equity_curve[0].0 as f64;
        let t1 = equity_curve.last().unwrap().0 as f64;
        let years = (t1 - t0) / (365.25 * 24.0 * 3600.0 * 1000.0);
        if years > 0.01 {
            (final_eq / starting_equity).powf(1.0 / years) - 1.0
        } else {
            0.0
        }
    } else {
        0.0
    };

    let calmar_ratio = if max_drawdown.abs() > 1e-9 {
        cagr / max_drawdown.abs()
    } else {
        f64::INFINITY
    };

    BacktestMetrics {
        total_trades: n,
        win_rate,
        profit_factor,
        sharpe,
        max_drawdown,
        cagr,
        calmar_ratio,
        avg_hold_bars,
        avg_return_per_trade,
    }
}

fn equity_returns(curve: &[(i64, f64)]) -> Vec<f64> {
    curve
        .windows(2)
        .map(|w| {
            let (_, e0) = w[0];
            let (_, e1) = w[1];
            if e0.abs() < 1e-9 { 0.0 } else { e1 / e0 - 1.0 }
        })
        .collect()
}

fn annualised_sharpe(returns: &[f64], ann_factor: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = var.sqrt();
    if std < 1e-12 {
        return 0.0;
    }
    mean / std * ann_factor.sqrt()
}

fn max_drawdown_frac(curve: &[(i64, f64)]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0_f64;
    for (_, eq) in curve {
        if *eq > peak {
            peak = *eq;
        }
        let dd = if peak > 0.0 { (peak - eq) / peak } else { 0.0 };
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Bar;

    fn make_bars(prices: &[f64], start_ts: i64, interval_ms: i64) -> Vec<Bar> {
        prices
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                Bar::new(
                    start_ts + i as i64 * interval_ms,
                    p * 0.999,
                    p * 1.002,
                    p * 0.998,
                    p,
                    1000.0,
                )
            })
            .collect()
    }

    #[test]
    fn engine_runs_without_panic() {
        let prices: Vec<f64> = (0..200)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0 + i as f64 * 0.05)
            .collect();

        let bars = make_bars(&prices, 1_700_000_000_000, 15 * 60 * 1000);

        let config = BacktestConfig::default_for("TEST");
        let mut engine = BacktestEngine::new(config);
        let result = engine.run(&[], &[], &bars).expect("engine.run failed");

        // Equity should remain positive
        assert!(result.final_equity > 0.0);
    }

    #[test]
    fn metrics_on_empty_trades() {
        let m = compute_metrics(&[], &[], 100_000.0);
        assert_eq!(m.total_trades, 0);
        assert_eq!(m.win_rate, 0.0);
    }
}
