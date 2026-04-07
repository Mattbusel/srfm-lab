//! "What if" counterfactual scenario engine for strategy decisions.
//!
//! This module lets you replay historical bars with modified parameters or
//! signals and compare the resulting P&L to the baseline.
//!
//! # Scenarios
//!
//! - "What if BH_MASS_THRESH was 1.5 instead of 2.0?"
//! - "What if there was NO event calendar filter?"
//! - "What if MIN_HOLD was 8 bars instead of 3?"
//! - "What if corr_factor was 0.40?"
//!
//! # Architecture
//!
//! ```text
//! ScenarioSpec      -- describes the counterfactual modification
//! WhatIfEngine      -- replays bars and computes P&L for any ScenarioSpec
//! ScenarioResult    -- baseline vs counterfactual P&L + interpretation
//! ParameterGrid     -- Cartesian product of parameter values for sweeps
//! ```

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Bar data
// ---------------------------------------------------------------------------

/// A single OHLCV bar with pre-computed signal values.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bar {
    /// Bar index (0-based).
    pub idx: usize,
    /// Open price.
    pub open: f64,
    /// High price.
    pub high: f64,
    /// Low price.
    pub low: f64,
    /// Close price.
    pub close: f64,
    /// Volume.
    pub volume: f64,
    /// Black Hole mass signal (pre-computed).
    pub bh_mass: f64,
    /// Hurst exponent (rolling 128-bar).
    pub hurst_h: f64,
    /// Realised volatility percentile.
    pub vol_percentile: f64,
    /// True when an event calendar block is active.
    pub event_calendar_block: bool,
    /// Regime label: "bull", "bear", "neutral".
    pub regime: String,
}

impl Bar {
    pub fn new(idx: usize, close: f64, bh_mass: f64, hurst_h: f64) -> Self {
        Self {
            idx,
            open: close,
            high: close,
            low: close,
            close,
            volume: 1_000_000.0,
            bh_mass,
            hurst_h,
            vol_percentile: 0.5,
            event_calendar_block: false,
            regime: "neutral".to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// BaselineParams
// ---------------------------------------------------------------------------

/// Strategy parameters used in the baseline simulation.
///
/// All parameters match the IAE genome parameter names.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineParams {
    /// Black Hole mass threshold to generate an entry signal.
    pub bh_mass_thresh: f64,
    /// Minimum hold duration in bars before exit is considered.
    pub min_hold_bars: usize,
    /// Maximum position size as fraction of NAV.
    pub pos_size_cap: f64,
    /// Corr factor for position scaling.
    pub corr_factor: f64,
    /// Whether the event calendar filter is active.
    pub event_calendar_filter: bool,
    /// NAV geodesic gate threshold.
    pub nav_geodesic_gate: f64,
    /// Stop-loss fraction.
    pub stop_loss_frac: f64,
    /// Take-profit fraction.
    pub take_profit_frac: f64,
    /// Regime-conditional scaling factors.
    pub cf_scale: HashMap<String, f64>,
}

impl Default for BaselineParams {
    fn default() -> Self {
        let mut cf_scale = HashMap::new();
        cf_scale.insert("bull".to_string(), 1.2);
        cf_scale.insert("bear".to_string(), 0.7);
        cf_scale.insert("neutral".to_string(), 1.0);

        Self {
            bh_mass_thresh: 2.0,
            min_hold_bars: 3,
            pos_size_cap: 0.10,
            corr_factor: 0.30,
            event_calendar_filter: true,
            nav_geodesic_gate: 0.95,
            stop_loss_frac: 0.02,
            take_profit_frac: 0.04,
            cf_scale,
        }
    }
}

// ---------------------------------------------------------------------------
// ScenarioSpec
// ---------------------------------------------------------------------------

/// Describes a single counterfactual scenario.
///
/// Any subset of the baseline parameters can be overridden.
/// `signal_override` forces a fixed signal value for all bars (used for
/// "what if signal was always ON/OFF").
/// `regime_override` forces all bars to be treated as if in the given regime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioSpec {
    /// Human-readable name for this scenario.
    pub name: String,
    /// Parameter overrides (applied on top of `BaselineParams`).
    pub parameter_override: HashMap<String, f64>,
    /// If `Some(v)`, every bar's bh_mass is replaced with `v`.
    pub signal_override: Option<f64>,
    /// If `Some(r)`, every bar's regime is replaced with `r`.
    pub regime_override: Option<String>,
    /// If `Some(false)`, event_calendar_block is ignored (filter disabled).
    pub disable_event_filter: bool,
}

impl ScenarioSpec {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            parameter_override: HashMap::new(),
            signal_override: None,
            regime_override: None,
            disable_event_filter: false,
        }
    }

    /// Override a single numeric parameter.
    pub fn with_param(mut self, key: &str, value: f64) -> Self {
        self.parameter_override.insert(key.to_string(), value);
        self
    }

    /// Disable the event calendar filter.
    pub fn without_event_filter(mut self) -> Self {
        self.disable_event_filter = true;
        self
    }

    /// Force all bars into a given regime.
    pub fn with_regime(mut self, regime: &str) -> Self {
        self.regime_override = Some(regime.to_string());
        self
    }

    /// Override the BH mass signal globally.
    pub fn with_signal(mut self, bh_mass: f64) -> Self {
        self.signal_override = Some(bh_mass);
        self
    }

    /// Build ScenarioSpec for "what if BH_MASS_THRESH was lower?" question.
    pub fn lower_bh_threshold(new_thresh: f64) -> Self {
        Self::new(&format!("bh_mass_thresh={new_thresh}"))
            .with_param("bh_mass_thresh", new_thresh)
    }

    /// Build ScenarioSpec for "what if there was NO event calendar filter?" question.
    pub fn no_event_filter() -> Self {
        Self::new("no_event_calendar_filter").without_event_filter()
    }

    /// Build ScenarioSpec for "what if MIN_HOLD was N bars?" question.
    pub fn min_hold(bars: usize) -> Self {
        Self::new(&format!("min_hold={bars}"))
            .with_param("min_hold_bars", bars as f64)
    }
}

// ---------------------------------------------------------------------------
// ScenarioResult
// ---------------------------------------------------------------------------

/// Result of a single "what if" scenario run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    /// Scenario name.
    pub name: String,
    /// Cumulative P&L under baseline parameters.
    pub baseline_pnl: f64,
    /// Cumulative P&L under counterfactual parameters.
    pub counterfactual_pnl: f64,
    /// Absolute P&L difference: counterfactual - baseline.
    pub pnl_delta: f64,
    /// Number of bars where the trade decision differed.
    pub trades_affected: usize,
    /// Number of trades executed under baseline.
    pub baseline_trades: usize,
    /// Number of trades executed under counterfactual.
    pub counterfactual_trades: usize,
    /// Human-readable interpretation.
    pub interpretation: String,
    /// Per-bar P&L under baseline (cumulative).
    pub baseline_equity_curve: Vec<f64>,
    /// Per-bar P&L under counterfactual (cumulative).
    pub counterfactual_equity_curve: Vec<f64>,
    /// Max drawdown under baseline.
    pub baseline_max_dd: f64,
    /// Max drawdown under counterfactual.
    pub counterfactual_max_dd: f64,
}

impl ScenarioResult {
    /// Sharpe ratio from equity curve (annualised, 252 bars/year).
    pub fn sharpe_from_curve(curve: &[f64], bars_per_year: f64) -> f64 {
        if curve.len() < 2 {
            return 0.0;
        }
        let returns: Vec<f64> = curve.windows(2).map(|w| w[1] - w[0]).collect();
        let mean_r = returns.iter().sum::<f64>() / returns.len() as f64;
        let var = returns.iter().map(|r| (r - mean_r).powi(2)).sum::<f64>()
            / (returns.len() - 1).max(1) as f64;
        let sd = var.sqrt();
        if sd < 1e-12 {
            return 0.0;
        }
        mean_r / sd * bars_per_year.sqrt()
    }

    /// Was the counterfactual strictly better? (higher P&L and lower drawdown).
    pub fn is_better(&self) -> bool {
        self.pnl_delta > 0.0 && self.counterfactual_max_dd <= self.baseline_max_dd
    }
}

// ---------------------------------------------------------------------------
// Internal simulation state
// ---------------------------------------------------------------------------

struct SimState {
    in_position: bool,
    entry_price: f64,
    entry_bar: usize,
    position_size: f64,
    cumulative_pnl: f64,
    equity_curve: Vec<f64>,
    trade_count: usize,
    peak_equity: f64,
    max_dd: f64,
}

impl SimState {
    fn new() -> Self {
        Self {
            in_position: false,
            entry_price: 0.0,
            entry_bar: 0,
            position_size: 0.0,
            cumulative_pnl: 0.0,
            equity_curve: Vec::new(),
            trade_count: 0,
            peak_equity: 0.0,
            max_dd: 0.0,
        }
    }

    fn record(&mut self) {
        self.equity_curve.push(self.cumulative_pnl);
        if self.cumulative_pnl > self.peak_equity {
            self.peak_equity = self.cumulative_pnl;
        }
        let dd = self.peak_equity - self.cumulative_pnl;
        if dd > self.max_dd {
            self.max_dd = dd;
        }
    }
}

// ---------------------------------------------------------------------------
// WhatIfEngine
// ---------------------------------------------------------------------------

/// Replays historical bars with modified parameters or signals and computes
/// the P&L difference versus the baseline strategy.
pub struct WhatIfEngine;

impl WhatIfEngine {
    pub fn new() -> Self {
        Self
    }

    /// Simulate the strategy on `bars` using the given `params`.
    /// Returns `(equity_curve, trade_count)`.
    fn simulate(bars: &[Bar], params: &BaselineParams) -> SimState {
        let mut state = SimState::new();
        let n = bars.len();

        for i in 0..n {
            let bar = &bars[i];
            let effective_regime = &bar.regime;
            let cf_scale = params.cf_scale.get(effective_regime.as_str()).copied().unwrap_or(1.0);
            let pos_size = (params.pos_size_cap * cf_scale * params.corr_factor).clamp(0.0, 0.20);

            // Check entry conditions
            if !state.in_position {
                let signal_ok = bar.bh_mass >= params.bh_mass_thresh;
                let filter_ok = !params.event_calendar_filter || !bar.event_calendar_block;
                let gate_ok = bar.hurst_h >= params.nav_geodesic_gate * 0.5; // simplified gate

                if signal_ok && filter_ok && gate_ok {
                    state.in_position = true;
                    state.entry_price = bar.close;
                    state.entry_bar = i;
                    state.position_size = pos_size;
                    state.trade_count += 1;
                }
            } else {
                let bars_held = i - state.entry_bar;
                let pnl_frac = (bar.close - state.entry_price) / state.entry_price;

                // Exit conditions
                let hit_sl = pnl_frac <= -params.stop_loss_frac;
                let hit_tp = pnl_frac >= params.take_profit_frac;
                let min_hold_met = bars_held >= params.min_hold_bars;
                let signal_gone = bar.bh_mass < params.bh_mass_thresh * 0.8;

                if hit_sl || hit_tp || (min_hold_met && signal_gone) {
                    let trade_pnl = pnl_frac * state.position_size * 100_000.0;
                    state.cumulative_pnl += trade_pnl;
                    state.in_position = false;
                    state.entry_price = 0.0;
                    state.position_size = 0.0;
                }
            }

            state.record();
        }

        // Close any open position at end of bars
        if state.in_position && n > 0 {
            let last_close = bars[n - 1].close;
            let pnl_frac = (last_close - state.entry_price) / state.entry_price;
            let trade_pnl = pnl_frac * state.position_size * 100_000.0;
            state.cumulative_pnl += trade_pnl;
            if let Some(last) = state.equity_curve.last_mut() {
                *last = state.cumulative_pnl;
            }
        }

        state
    }

    /// Apply a `ScenarioSpec` on top of `BaselineParams`, producing a modified
    /// `(Vec<Bar>, BaselineParams)` pair ready for simulation.
    fn apply_scenario(
        bars: &[Bar],
        baseline: &BaselineParams,
        scenario: &ScenarioSpec,
    ) -> (Vec<Bar>, BaselineParams) {
        let mut params = baseline.clone();

        // Apply numeric overrides
        for (key, &val) in &scenario.parameter_override {
            match key.as_str() {
                "bh_mass_thresh"      => params.bh_mass_thresh = val,
                "min_hold_bars"       => params.min_hold_bars = val as usize,
                "pos_size_cap"        => params.pos_size_cap = val,
                "corr_factor"         => params.corr_factor = val,
                "nav_geodesic_gate"   => params.nav_geodesic_gate = val,
                "stop_loss_frac"      => params.stop_loss_frac = val,
                "take_profit_frac"    => params.take_profit_frac = val,
                "cf_scale_bull"       => { params.cf_scale.insert("bull".to_string(), val); }
                "cf_scale_bear"       => { params.cf_scale.insert("bear".to_string(), val); }
                "cf_scale_neutral"    => { params.cf_scale.insert("neutral".to_string(), val); }
                _ => {} // unknown keys silently ignored
            }
        }

        // Disable event calendar filter
        if scenario.disable_event_filter {
            params.event_calendar_filter = false;
        }

        // Modify bars for signal/regime overrides
        let modified_bars: Vec<Bar> = bars.iter().map(|b| {
            let mut bar = b.clone();
            if let Some(bh) = scenario.signal_override {
                bar.bh_mass = bh;
            }
            if let Some(ref r) = scenario.regime_override {
                bar.regime = r.clone();
            }
            bar
        }).collect();

        (modified_bars, params)
    }

    /// Run a single counterfactual scenario against the baseline.
    pub fn run_counterfactual(
        &self,
        historical_bars: &[Bar],
        baseline_params: &BaselineParams,
        scenario: ScenarioSpec,
    ) -> ScenarioResult {
        // Simulate baseline
        let baseline_state = Self::simulate(historical_bars, baseline_params);

        // Apply scenario and simulate counterfactual
        let (cf_bars, cf_params) = Self::apply_scenario(historical_bars, baseline_params, &scenario);
        let cf_state = Self::simulate(&cf_bars, &cf_params);

        // Count bars where decisions differed (proxy for trades affected)
        let trades_affected = baseline_state
            .equity_curve
            .iter()
            .zip(cf_state.equity_curve.iter())
            .filter(|(a, b)| (*a - *b).abs() > 1e-9)
            .count();

        let pnl_delta = cf_state.cumulative_pnl - baseline_state.cumulative_pnl;
        let interpretation = Self::interpret(&scenario.name, pnl_delta, trades_affected);

        ScenarioResult {
            name: scenario.name,
            baseline_pnl: baseline_state.cumulative_pnl,
            counterfactual_pnl: cf_state.cumulative_pnl,
            pnl_delta,
            trades_affected,
            baseline_trades: baseline_state.trade_count,
            counterfactual_trades: cf_state.trade_count,
            interpretation,
            baseline_equity_curve: baseline_state.equity_curve,
            counterfactual_equity_curve: cf_state.equity_curve,
            baseline_max_dd: baseline_state.max_dd,
            counterfactual_max_dd: cf_state.max_dd,
        }
    }

    /// Run a batch of scenarios and return all results.
    pub fn run_batch(
        &self,
        historical_bars: &[Bar],
        baseline_params: &BaselineParams,
        scenarios: Vec<ScenarioSpec>,
    ) -> Vec<ScenarioResult> {
        scenarios
            .into_iter()
            .map(|s| self.run_counterfactual(historical_bars, baseline_params, s))
            .collect()
    }

    /// Run a parameter sweep over a grid of values for a single parameter.
    ///
    /// Returns results sorted by `pnl_delta` descending (best first).
    pub fn run_parameter_sweep(
        &self,
        historical_bars: &[Bar],
        baseline_params: &BaselineParams,
        param_grid: &ParameterGrid,
    ) -> Vec<ScenarioResult> {
        let mut results: Vec<ScenarioResult> = param_grid
            .scenarios()
            .into_iter()
            .map(|s| self.run_counterfactual(historical_bars, baseline_params, s))
            .collect();

        results.sort_by(|a, b| b.pnl_delta.partial_cmp(&a.pnl_delta).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    fn interpret(name: &str, pnl_delta: f64, trades_affected: usize) -> String {
        let direction = if pnl_delta > 0.0 { "better" } else { "worse" };
        let magnitude = pnl_delta.abs();
        format!(
            "Scenario '{name}': counterfactual was {direction} by ${magnitude:.2} P&L \
             across {trades_affected} affected bars."
        )
    }
}

impl Default for WhatIfEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ParameterGrid
// ---------------------------------------------------------------------------

/// Cartesian product of parameter values for a parameter sweep.
///
/// Each entry in `axes` is `(param_name, values)`.
/// The grid generates one `ScenarioSpec` for each combination.
pub struct ParameterGrid {
    /// Each axis: (parameter name, list of values to test).
    pub axes: Vec<(String, Vec<f64>)>,
}

impl ParameterGrid {
    pub fn new() -> Self {
        Self { axes: Vec::new() }
    }

    /// Add an axis to the grid.
    pub fn add_axis(mut self, param: &str, values: Vec<f64>) -> Self {
        self.axes.push((param.to_string(), values));
        self
    }

    /// Enumerate all `ScenarioSpec`s in the grid (Cartesian product).
    pub fn scenarios(&self) -> Vec<ScenarioSpec> {
        if self.axes.is_empty() {
            return Vec::new();
        }
        // Start with a single empty scenario
        let mut scenarios: Vec<HashMap<String, f64>> = vec![HashMap::new()];

        for (param_name, values) in &self.axes {
            let mut expanded = Vec::new();
            for base in &scenarios {
                for &val in values {
                    let mut s = base.clone();
                    s.insert(param_name.clone(), val);
                    expanded.push(s);
                }
            }
            scenarios = expanded;
        }

        scenarios
            .into_iter()
            .map(|overrides| {
                let name = overrides
                    .iter()
                    .map(|(k, v)| format!("{k}={v:.3}"))
                    .collect::<Vec<_>>()
                    .join(",");
                ScenarioSpec {
                    name,
                    parameter_override: overrides,
                    signal_override: None,
                    regime_override: None,
                    disable_event_filter: false,
                }
            })
            .collect()
    }

    /// Total number of scenarios in this grid.
    pub fn len(&self) -> usize {
        if self.axes.is_empty() {
            return 0;
        }
        self.axes.iter().map(|(_, v)| v.len()).product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for ParameterGrid {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Convenience constructors for standard LARSA scenarios
// ---------------------------------------------------------------------------

/// Build the standard set of LARSA "what if" scenarios used in research.
pub fn standard_larsa_scenarios() -> Vec<ScenarioSpec> {
    vec![
        ScenarioSpec::lower_bh_threshold(1.5),
        ScenarioSpec::lower_bh_threshold(1.8),
        ScenarioSpec::no_event_filter(),
        ScenarioSpec::min_hold(1),
        ScenarioSpec::min_hold(8),
        ScenarioSpec::min_hold(15),
        ScenarioSpec::new("high_pos_cap").with_param("pos_size_cap", 0.15),
        ScenarioSpec::new("low_pos_cap").with_param("pos_size_cap", 0.05),
        ScenarioSpec::new("bull_regime").with_regime("bull"),
        ScenarioSpec::new("bear_regime").with_regime("bear"),
        ScenarioSpec::new("tight_sl").with_param("stop_loss_frac", 0.01),
        ScenarioSpec::new("wide_sl").with_param("stop_loss_frac", 0.05),
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_trending_bars(n: usize) -> Vec<Bar> {
        let mut bars = Vec::with_capacity(n);
        let mut price = 100.0f64;
        let mut state = 42u64;
        for i in 0..n {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let r = (state as f64 / u64::MAX as f64) * 0.004 - 0.001; // mild drift up
            price *= 1.0 + r + 0.001;
            let mut bar = Bar::new(i, price, 2.5, 0.65); // bh_mass=2.5 > default thresh 2.0
            bar.event_calendar_block = (i % 10 == 5); // block every 10th bar
            bar.regime = "bull".to_string();
            bars.push(bar);
        }
        bars
    }

    fn make_blocked_bars(n: usize) -> Vec<Bar> {
        // All bars have event calendar block active and borderline bh_mass
        let mut bars = Vec::with_capacity(n);
        let mut price = 100.0f64;
        for i in 0..n {
            price *= 1.002; // steady uptrend
            let mut bar = Bar::new(i, price, 2.5, 0.65);
            bar.event_calendar_block = true; // all blocked
            bars.push(bar);
        }
        bars
    }

    #[test]
    fn test_baseline_simulation_returns_equity_curve() {
        let bars = make_trending_bars(50);
        let params = BaselineParams::default();
        let engine = WhatIfEngine::new();
        let result = engine.run_counterfactual(&bars, &params, ScenarioSpec::new("baseline_check"));
        assert_eq!(result.baseline_equity_curve.len(), bars.len());
    }

    #[test]
    fn test_no_event_filter_more_trades_on_blocked_bars() {
        let bars = make_blocked_bars(50);
        let params = BaselineParams::default(); // event_calendar_filter = true

        let engine = WhatIfEngine::new();
        let result = engine.run_counterfactual(&bars, &params, ScenarioSpec::no_event_filter());

        // Without filter, more trades should fire since all bars have blocks
        assert!(
            result.counterfactual_trades >= result.baseline_trades,
            "Without event filter, trades ({}) should be >= baseline trades ({})",
            result.counterfactual_trades,
            result.baseline_trades
        );
    }

    #[test]
    fn test_lower_bh_threshold_more_trades() {
        let bars = make_trending_bars(100);
        let params = BaselineParams::default(); // bh_mass_thresh = 2.0

        let engine = WhatIfEngine::new();
        // Lower threshold -> more bars qualify -> more trades
        let result = engine.run_counterfactual(
            &bars,
            &params,
            ScenarioSpec::lower_bh_threshold(1.0), // much lower than default 2.0
        );

        assert!(
            result.counterfactual_trades >= result.baseline_trades,
            "Lower threshold: cf_trades={} should be >= baseline={}",
            result.counterfactual_trades,
            result.baseline_trades
        );
    }

    #[test]
    fn test_higher_bh_threshold_fewer_trades() {
        let bars = make_trending_bars(100);
        // bars have bh_mass = 2.5; raising threshold to 3.0 should block entries
        let params = BaselineParams::default();

        let engine = WhatIfEngine::new();
        let result = engine.run_counterfactual(
            &bars,
            &params,
            ScenarioSpec::new("high_thresh").with_param("bh_mass_thresh", 3.0),
        );

        assert!(
            result.counterfactual_trades <= result.baseline_trades,
            "Higher threshold: cf_trades={} should be <= baseline={}",
            result.counterfactual_trades,
            result.baseline_trades
        );
    }

    #[test]
    fn test_min_hold_8_bars_scenario() {
        let bars = make_trending_bars(100);
        let params = BaselineParams::default(); // min_hold = 3

        let engine = WhatIfEngine::new();
        let result = engine.run_counterfactual(&bars, &params, ScenarioSpec::min_hold(8));

        assert_eq!(result.name, "min_hold=8");
        // Result should be computed without panic
        assert!(result.baseline_pnl.is_finite());
        assert!(result.counterfactual_pnl.is_finite());
    }

    #[test]
    fn test_pnl_delta_equals_cf_minus_baseline() {
        let bars = make_trending_bars(80);
        let params = BaselineParams::default();
        let engine = WhatIfEngine::new();
        let result = engine.run_counterfactual(&bars, &params, ScenarioSpec::no_event_filter());

        let expected_delta = result.counterfactual_pnl - result.baseline_pnl;
        assert!(
            (result.pnl_delta - expected_delta).abs() < 1e-6,
            "pnl_delta mismatch: {} vs {expected_delta}",
            result.pnl_delta
        );
    }

    #[test]
    fn test_parameter_sweep_sorted_by_pnl_delta() {
        let bars = make_trending_bars(80);
        let params = BaselineParams::default();
        let engine = WhatIfEngine::new();

        let grid = ParameterGrid::new()
            .add_axis("bh_mass_thresh", vec![1.0, 1.5, 2.0, 2.5, 3.0]);

        let results = engine.run_parameter_sweep(&bars, &params, &grid);
        assert_eq!(results.len(), 5);

        // Results should be sorted descending by pnl_delta
        for w in results.windows(2) {
            assert!(
                w[0].pnl_delta >= w[1].pnl_delta,
                "Results not sorted: {} < {}",
                w[0].pnl_delta,
                w[1].pnl_delta
            );
        }
    }

    #[test]
    fn test_parameter_grid_cartesian_product() {
        let grid = ParameterGrid::new()
            .add_axis("bh_mass_thresh", vec![1.5, 2.0, 2.5])
            .add_axis("min_hold_bars", vec![3.0, 8.0]);

        assert_eq!(grid.len(), 6, "3 x 2 = 6 scenarios");
        let scenarios = grid.scenarios();
        assert_eq!(scenarios.len(), 6);
    }

    #[test]
    fn test_batch_run_returns_all_results() {
        let bars = make_trending_bars(60);
        let params = BaselineParams::default();
        let engine = WhatIfEngine::new();
        let scenarios = standard_larsa_scenarios();
        let n = scenarios.len();
        let results = engine.run_batch(&bars, &params, scenarios);
        assert_eq!(results.len(), n);
    }

    #[test]
    fn test_equity_curve_length_matches_bars() {
        let bars = make_trending_bars(75);
        let params = BaselineParams::default();
        let engine = WhatIfEngine::new();
        let result = engine.run_counterfactual(
            &bars,
            &params,
            ScenarioSpec::new("length_check"),
        );
        assert_eq!(result.baseline_equity_curve.len(), bars.len());
        assert_eq!(result.counterfactual_equity_curve.len(), bars.len());
    }

    #[test]
    fn test_disable_filter_scenario_spec() {
        let spec = ScenarioSpec::no_event_filter();
        assert!(spec.disable_event_filter);
        assert!(spec.parameter_override.is_empty());
    }

    #[test]
    fn test_interpretation_string_non_empty() {
        let bars = make_trending_bars(40);
        let params = BaselineParams::default();
        let engine = WhatIfEngine::new();
        let result = engine.run_counterfactual(&bars, &params, ScenarioSpec::no_event_filter());
        assert!(!result.interpretation.is_empty());
        assert!(result.interpretation.contains("no_event_calendar_filter"));
    }
}
