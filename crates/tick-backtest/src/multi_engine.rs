// multi_engine.rs — Parallel multi-instrument backtest engine
//
// Uses rayon to run each instrument's backtest on a separate thread,
// then aggregates results into a portfolio-level summary.

use crate::csv_loader::BarsSet;
use crate::engine::{BacktestConfig, BacktestEngine, BacktestResult, compute_metrics};
use crate::types::{BacktestMetrics, RegimeStats, Trade};
use anyhow::{Context, Result};
use ndarray::Array2;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// MultiBacktestConfig
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiBacktestConfig {
    /// Per-instrument configurations keyed by symbol.
    pub instruments: HashMap<String, BacktestConfig>,
    /// Shared starting equity (allocated equally across instruments).
    pub starting_equity: f64,
    /// Whether to normalise each instrument's equity curve to a common start
    /// before portfolio aggregation (default: true).
    pub normalise_equity: bool,
}

impl MultiBacktestConfig {
    pub fn new(starting_equity: f64) -> Self {
        Self {
            instruments: HashMap::new(),
            starting_equity,
            normalise_equity: true,
        }
    }

    pub fn add_instrument(&mut self, config: BacktestConfig) {
        self.instruments.insert(config.sym.clone(), config);
    }

    pub fn with_default_params(mut self, syms: &[&str], cf: f64, bh_form: f64, bh_collapse: f64, bh_decay: f64) -> Self {
        let per_sym_equity = self.starting_equity / syms.len().max(1) as f64;
        for &sym in syms {
            let mut cfg = BacktestConfig::default_for(sym);
            cfg.cf = cf;
            cfg.bh_form = bh_form;
            cfg.bh_collapse = bh_collapse;
            cfg.bh_decay = bh_decay;
            cfg.starting_equity = per_sym_equity;
            self.instruments.insert(sym.to_string(), cfg);
        }
        self
    }
}

// ---------------------------------------------------------------------------
// MultiBacktestResult
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiBacktestResult {
    /// Per-instrument results.
    pub per_instrument: HashMap<String, BacktestResult>,
    /// Portfolio-level equity curve (sum of all instrument equity curves,
    /// sampled at a common set of timestamps).
    pub portfolio_equity_curve: Vec<(i64, f64)>,
    /// Portfolio-level metrics.
    pub portfolio_metrics: BacktestMetrics,
    /// All trades from all instruments combined.
    pub all_trades: Vec<Trade>,
    /// Cross-instrument return correlation matrix.
    /// Row/col order is `sorted_syms`.
    pub return_correlation: Vec<Vec<f64>>,
    pub sorted_syms: Vec<String>,
}

// ---------------------------------------------------------------------------
// MultiBacktestEngine
// ---------------------------------------------------------------------------

pub struct MultiBacktestEngine {
    pub config: MultiBacktestConfig,
}

impl MultiBacktestEngine {
    pub fn new(config: MultiBacktestConfig) -> Self {
        Self { config }
    }

    /// Run per-instrument backtests in parallel using rayon.
    pub fn run_parallel(
        &self,
        all_bars: &HashMap<String, BarsSet>,
    ) -> Result<MultiBacktestResult> {
        let syms: Vec<String> = {
            let mut s: Vec<String> = self.config.instruments.keys().cloned().collect();
            s.sort();
            s
        };

        // Parallel per-instrument runs
        let results: Vec<(String, Result<BacktestResult>)> = syms
            .par_iter()
            .map(|sym| {
                let bars = all_bars.get(sym).cloned().unwrap_or_default();
                let cfg = self.config.instruments[sym].clone();
                let mut engine = BacktestEngine::new(cfg);
                let res = engine
                    .run_barsset(&bars)
                    .with_context(|| format!("Backtest failed for {sym}"));
                (sym.clone(), res)
            })
            .collect();

        // Collect and surface errors
        let mut per_instrument: HashMap<String, BacktestResult> = HashMap::new();
        for (sym, res) in results {
            per_instrument.insert(sym, res?);
        }

        // Aggregate equity curves
        let portfolio_equity_curve =
            aggregate_equity_curves(&per_instrument, self.config.starting_equity);

        // Portfolio metrics
        let all_trades: Vec<Trade> = per_instrument
            .values()
            .flat_map(|r| r.trades.iter().cloned())
            .collect();

        let portfolio_metrics = compute_metrics(
            &all_trades,
            &portfolio_equity_curve,
            self.config.starting_equity,
        );

        // Return correlation
        let (return_correlation, sorted_syms) =
            compute_return_correlation(&per_instrument, &syms);

        Ok(MultiBacktestResult {
            per_instrument,
            portfolio_equity_curve,
            portfolio_metrics,
            all_trades,
            return_correlation,
            sorted_syms,
        })
    }
}

// ---------------------------------------------------------------------------
// Portfolio equity aggregation
// ---------------------------------------------------------------------------

/// Merge all per-instrument equity curves by summing equity at each
/// unique timestamp.  Carries forward the last known equity for
/// instruments that haven't had an update yet.
fn aggregate_equity_curves(
    results: &HashMap<String, BacktestResult>,
    starting_equity: f64,
) -> Vec<(i64, f64)> {
    // Collect all timestamps
    let mut all_ts: Vec<i64> = results
        .values()
        .flat_map(|r| r.equity_curve.iter().map(|(ts, _)| *ts))
        .collect();
    all_ts.sort_unstable();
    all_ts.dedup();

    if all_ts.is_empty() {
        return vec![];
    }

    // For each instrument: build a lookup: ts → equity (sorted).
    // We carry forward the last known value.
    let per_sym_curves: Vec<Vec<(i64, f64)>> = results
        .values()
        .map(|r| {
            let mut c = r.equity_curve.clone();
            c.sort_by_key(|(ts, _)| *ts);
            c
        })
        .collect();

    let n_syms = per_sym_curves.len();
    // Per-symbol starting equities
    let per_sym_start = starting_equity / n_syms.max(1) as f64;

    all_ts
        .iter()
        .map(|&ts| {
            let sum: f64 = per_sym_curves
                .iter()
                .map(|curve| {
                    // Binary search for the last point ≤ ts
                    let idx = curve.partition_point(|(t, _)| *t <= ts);
                    if idx == 0 {
                        per_sym_start
                    } else {
                        curve[idx - 1].1
                    }
                })
                .sum();
            (ts, sum)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Return correlation
// ---------------------------------------------------------------------------

/// Compute pairwise Pearson correlation of equity-curve returns.
/// Returns (n×n correlation matrix as Vec<Vec<f64>>, sorted symbol list).
pub fn compute_return_correlation(
    results: &HashMap<String, BacktestResult>,
    sorted_syms: &[String],
) -> (Vec<Vec<f64>>, Vec<String>) {
    let n = sorted_syms.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    // Collect log-returns for each symbol
    let series: Vec<Vec<f64>> = sorted_syms
        .iter()
        .map(|sym| {
            let curve = &results[sym].equity_curve;
            curve
                .windows(2)
                .map(|w| {
                    let (_, e0) = w[0];
                    let (_, e1) = w[1];
                    if e0.abs() < 1e-9 { 0.0 } else { (e1 / e0).ln() }
                })
                .collect::<Vec<f64>>()
        })
        .collect();

    // Compute pair-wise Pearson
    let mut corr = vec![vec![1.0_f64; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let r = pearson_corr(&series[i], &series[j]);
            corr[i][j] = r;
            corr[j][i] = r;
        }
    }

    (corr, sorted_syms.to_vec())
}

/// Same as above but returns an ndarray Array2<f64>.
pub fn correlate_returns_ndarray(
    results: &HashMap<String, BacktestResult>,
    sorted_syms: &[String],
) -> Array2<f64> {
    let (corr_vec, _) = compute_return_correlation(results, sorted_syms);
    let n = corr_vec.len();
    if n == 0 {
        return Array2::zeros((0, 0));
    }
    let flat: Vec<f64> = corr_vec.into_iter().flatten().collect();
    Array2::from_shape_vec((n, n), flat).unwrap_or_else(|_| Array2::eye(n))
}

fn pearson_corr(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    if len < 2 {
        return 0.0;
    }
    let n = len as f64;
    let a = &a[..len];
    let b = &b[..len];
    let mean_a = a.iter().sum::<f64>() / n;
    let mean_b = b.iter().sum::<f64>() / n;
    let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - mean_a) * (y - mean_b)).sum::<f64>() / n;
    let std_a = (a.iter().map(|x| (x - mean_a).powi(2)).sum::<f64>() / n).sqrt();
    let std_b = (b.iter().map(|y| (y - mean_b).powi(2)).sum::<f64>() / n).sqrt();
    if std_a < 1e-12 || std_b < 1e-12 {
        0.0
    } else {
        cov / (std_a * std_b)
    }
}

// ---------------------------------------------------------------------------
// Portfolio-level regime breakdown
// ---------------------------------------------------------------------------

impl MultiBacktestResult {
    /// Aggregate regime stats across all instruments.
    pub fn portfolio_regime_breakdown(&self) -> HashMap<String, RegimeStats> {
        let mut combined: HashMap<String, RegimeStats> = HashMap::new();
        for result in self.per_instrument.values() {
            for (regime, stats) in &result.regime_breakdown {
                let entry = combined.entry(regime.clone()).or_default();
                // Merge stats
                entry.count += stats.count;
                entry.winners += stats.winners;
                entry.total_pnl += stats.total_pnl;
                // Re-compute running avg_return
                if entry.count > 0 {
                    entry.avg_return = entry.total_pnl / entry.count as f64;
                }
            }
        }
        combined
    }

    /// Summary string for logging / CLI output.
    pub fn summary(&self) -> String {
        let m = &self.portfolio_metrics;
        format!(
            "Portfolio: {} instruments | {} total trades | \
             Sharpe {:.2} | CAGR {:.1}% | MaxDD {:.1}% | WinRate {:.1}%",
            self.per_instrument.len(),
            self.all_trades.len(),
            m.sharpe,
            m.cagr * 100.0,
            m.max_drawdown * 100.0,
            m.win_rate * 100.0,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Bar;

    fn make_bars(n: usize, start: i64, interval_ms: i64, start_price: f64) -> Vec<Bar> {
        (0..n)
            .map(|i| {
                let p = start_price + (i as f64 * 0.1).sin() * 3.0 + i as f64 * 0.02;
                Bar::new(
                    start + i as i64 * interval_ms,
                    p * 0.999,
                    p * 1.003,
                    p * 0.997,
                    p,
                    500.0,
                )
            })
            .collect()
    }

    #[test]
    fn multi_engine_two_instruments() {
        let mut cfg = MultiBacktestConfig::new(200_000.0);

        let syms = ["AAA", "BBB"];
        for sym in &syms {
            let mut c = BacktestConfig::default_for(*sym);
            c.starting_equity = 100_000.0;
            cfg.add_instrument(c);
        }

        let mut all_bars: HashMap<String, BarsSet> = HashMap::new();
        for sym in &syms {
            let bars = make_bars(300, 1_700_000_000_000, 15 * 60 * 1000, 100.0);
            all_bars.insert(sym.to_string(), BarsSet::new(vec![], vec![], bars));
        }

        let engine = MultiBacktestEngine::new(cfg);
        let result = engine.run_parallel(&all_bars).expect("multi engine failed");

        assert_eq!(result.per_instrument.len(), 2);
        assert!(!result.portfolio_equity_curve.is_empty());
    }

    #[test]
    fn pearson_corr_identical_series() {
        let a: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let r = pearson_corr(&a, &a);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pearson_corr_anti_correlated() {
        let a: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let b: Vec<f64> = a.iter().map(|&x| -x).collect();
        let r = pearson_corr(&a, &b);
        assert!((r + 1.0).abs() < 1e-9);
    }
}
