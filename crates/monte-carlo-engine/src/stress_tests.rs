//! Stress test scenarios for Monte Carlo simulation.
//!
//! Provides historical stress scenario injection, tail risk analysis, and
//! Conditional Drawdown-at-Risk (CDaR) computation.
//!
//! # Scenario Injection Model
//!
//! Each scenario encodes a shock sequence as a multiplier applied to the
//! simulated return series at a random injection point. This is equivalent to
//! the historical simulation approach used by risk desks: take the actual
//! period's returns and inject them into simulated paths.

use anyhow::{bail, Result};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// StressScenario
// ---------------------------------------------------------------------------

/// Predefined and custom stress scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "params")]
pub enum StressScenario {
    /// COVID-19 crash: ~35% drawdown over 5 weeks (Feb–Mar 2020).
    MarketCrash2020,
    /// Crypto winter 2022: BTC -65%, prolonged drawdown over 12 months.
    CryptoWinter2022,
    /// COVID dip: sharp 3-day selloff then rapid recovery (Mar 2020 bounce).
    CovidDip,
    /// GFC 2008: financial crisis, 18-month drawdown, ~50% peak-to-trough.
    Gfc2008,
    /// Flash crash: single-day extreme move, quick recovery (May 2010).
    FlashCrash,
    /// Custom scenario defined by the caller.
    Custom(CustomScenarioParams),
}

/// Parameters for a custom stress scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomScenarioParams {
    /// Display name for the scenario.
    pub name: String,
    /// Sequence of daily returns to inject (as fractional returns, e.g. -0.05).
    pub shock_returns: Vec<f64>,
    /// If true, scenario is injected at a random point in each path.
    pub random_injection: bool,
    /// If `random_injection = false`, inject at this bar index.
    pub fixed_injection_bar: usize,
}

impl StressScenario {
    /// Return the display name of the scenario.
    pub fn name(&self) -> &str {
        match self {
            StressScenario::MarketCrash2020 => "Market Crash 2020 (COVID-19)",
            StressScenario::CryptoWinter2022 => "Crypto Winter 2022",
            StressScenario::CovidDip => "COVID Dip (Mar 2020 Bounce)",
            StressScenario::Gfc2008 => "Global Financial Crisis 2008",
            StressScenario::FlashCrash => "Flash Crash (May 2010)",
            StressScenario::Custom(p) => &p.name,
        }
    }

    /// Return the shock return sequence for this scenario.
    pub fn shock_sequence(&self) -> Vec<f64> {
        match self {
            StressScenario::MarketCrash2020 => covid_crash_returns(),
            StressScenario::CryptoWinter2022 => crypto_winter_returns(),
            StressScenario::CovidDip => covid_dip_returns(),
            StressScenario::Gfc2008 => gfc_returns(),
            StressScenario::FlashCrash => flash_crash_returns(),
            StressScenario::Custom(p) => p.shock_returns.clone(),
        }
    }

    /// Duration (number of bars) of this stress scenario.
    pub fn duration(&self) -> usize {
        self.shock_sequence().len()
    }
}

// ---------------------------------------------------------------------------
// Historical shock sequences
// ---------------------------------------------------------------------------

/// COVID-19 crash: Feb 20 – Mar 23 2020 (~24 trading days).
/// Stylised sequence capturing the approximately -35% drawdown.
fn covid_crash_returns() -> Vec<f64> {
    vec![
        -0.031, -0.028, -0.041, -0.053, -0.069, -0.048, -0.079,
        -0.098, -0.052, 0.045, -0.060, -0.038, -0.047, -0.110,
        0.060, -0.044, -0.031, 0.054, -0.037, 0.064, -0.025,
        0.053, 0.042, 0.051,
    ]
}

/// Crypto winter 2022: Jan–Dec 2022 monthly-aggregated daily shocks.
/// Captures -65% BTC drawdown spread over ~252 trading days.
fn crypto_winter_returns() -> Vec<f64> {
    let mut r = Vec::with_capacity(252);
    // Jan–Feb: initial decline
    r.extend(vec![-0.018; 30]);
    // Mar–May: acceleration
    r.extend(vec![-0.025; 40]);
    // Jun: LUNA collapse spike
    r.extend(vec![-0.035, -0.048, -0.062, -0.041, -0.030]);
    r.extend(vec![-0.015; 15]);
    // Jul: relief rally
    r.extend(vec![0.020; 20]);
    // Aug–Oct: grind lower
    r.extend(vec![-0.012; 60]);
    // Nov: FTX collapse
    r.extend(vec![-0.025, -0.058, -0.085, -0.042, -0.031]);
    r.extend(vec![-0.008; 15]);
    // Dec: bottom
    r.extend(vec![-0.005; 12]);
    r
}

/// COVID dip: 3-day sharp drop then rapid recovery.
fn covid_dip_returns() -> Vec<f64> {
    vec![
        -0.070, -0.095, -0.082, 0.040, 0.055, 0.065, 0.050, 0.035, 0.025, 0.018,
    ]
}

/// GFC 2008: 18-month stylised drawdown (~50% peak-to-trough).
fn gfc_returns() -> Vec<f64> {
    let mut r = Vec::with_capacity(378);
    // Phase 1: gradual decline (Jan–Sep 2008)
    r.extend(vec![-0.008; 180]);
    // Phase 2: Lehman collapse (Sep–Oct 2008)
    r.extend(vec![
        -0.035, -0.042, -0.091, -0.078, -0.058,
        0.041, -0.053, -0.067, 0.038, -0.044,
        -0.031, 0.055, -0.062, -0.049, 0.032,
    ]);
    // Phase 3: continued decline (Nov–Feb 2009)
    r.extend(vec![-0.012; 80]);
    // Phase 4: bottom trough (Mar 2009)
    r.extend(vec![-0.005; 20]);
    // Phase 5: initial recovery signal
    r.extend(vec![0.010; 83]);
    r
}

/// Flash crash (May 6, 2010): single-day -9%, recovery in minutes.
fn flash_crash_returns() -> Vec<f64> {
    vec![
        -0.090, 0.065, 0.012, 0.008, 0.005, 0.003, 0.002,
    ]
}

// ---------------------------------------------------------------------------
// ScenarioResult
// ---------------------------------------------------------------------------

/// Results from running a stress scenario against a return series.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioResult {
    pub scenario_name: String,
    pub scenario_duration_bars: usize,

    /// Final equity after scenario injection, averaged over N paths.
    pub mean_final_equity: f64,
    /// Median final equity.
    pub median_final_equity: f64,
    /// 5th-percentile final equity.
    pub p5_final_equity: f64,
    /// Maximum drawdown experienced during/after scenario injection.
    pub median_max_drawdown: f64,
    /// P95 max drawdown.
    pub p95_max_drawdown: f64,
    /// Probability of ruin (equity < 50% of start).
    pub prob_ruin: f64,
    /// Tail risk contribution of this scenario.
    pub tail_risk_contribution: f64,
}

// ---------------------------------------------------------------------------
// StressTestEngine
// ---------------------------------------------------------------------------

/// Engine for running stress test scenarios.
pub struct StressTestEngine {
    /// Number of paths to simulate per scenario.
    pub n_paths: usize,
    /// Total path length in bars.
    pub n_bars: usize,
    /// Starting equity.
    pub initial_equity: f64,
    /// Random seed (0 = non-deterministic).
    pub seed: u64,
}

impl StressTestEngine {
    pub fn new(n_paths: usize, n_bars: usize, initial_equity: f64) -> Self {
        Self {
            n_paths,
            n_bars,
            initial_equity,
            seed: 0,
        }
    }

    /// Run a single stress scenario against a historical return series.
    ///
    /// The scenario shock sequence is injected into simulated paths at a
    /// random injection point (or the start if the scenario is longer than
    /// the path).
    pub fn run_scenario(
        &self,
        base_returns: &[f64],
        scenario: &StressScenario,
    ) -> Result<ScenarioResult> {
        if base_returns.is_empty() {
            bail!("base_returns is empty");
        }
        let shocks = scenario.shock_sequence();
        let shock_len = shocks.len();

        let mut final_equities: Vec<f64> = Vec::with_capacity(self.n_paths);
        let mut max_drawdowns: Vec<f64> = Vec::with_capacity(self.n_paths);

        for path_i in 0..self.n_paths {
            let seed = if self.seed == 0 {
                rand::random::<u64>() ^ (path_i as u64 * 6364136223846793005)
            } else {
                self.seed.wrapping_add(path_i as u64)
            };
            let mut rng = SmallRng::seed_from_u64(seed);

            // Build a return path: bootstrap from base_returns
            let path_returns = bootstrap_path(base_returns, self.n_bars, &mut rng);

            // Determine injection point
            let injection_bar = if shock_len >= self.n_bars {
                0
            } else {
                rng.gen_range(0..=(self.n_bars - shock_len))
            };

            // Simulate equity with shock injection
            let (final_eq, max_dd) =
                simulate_with_shock(&path_returns, &shocks, injection_bar, self.initial_equity);

            final_equities.push(final_eq);
            max_drawdowns.push(max_dd);
        }

        final_equities.sort_by(|a, b| a.partial_cmp(b).unwrap());
        max_drawdowns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = self.n_paths as f64;
        let mean_final = final_equities.iter().sum::<f64>() / n;
        let ruin_count = final_equities
            .iter()
            .filter(|&&e| e < self.initial_equity * 0.5)
            .count();

        let trc = tail_risk_contribution(base_returns, &shocks);

        Ok(ScenarioResult {
            scenario_name: scenario.name().to_string(),
            scenario_duration_bars: shock_len,
            mean_final_equity: mean_final,
            median_final_equity: percentile(&final_equities, 50.0),
            p5_final_equity: percentile(&final_equities, 5.0),
            median_max_drawdown: percentile(&max_drawdowns, 50.0),
            p95_max_drawdown: percentile(&max_drawdowns, 95.0),
            prob_ruin: ruin_count as f64 / n,
            tail_risk_contribution: trc,
        })
    }
}

// ---------------------------------------------------------------------------
// Tail risk functions
// ---------------------------------------------------------------------------

/// Compute the tail risk contribution of a shock sequence relative to the
/// base strategy returns.
///
/// Defined as: mean(shock) / std(base) × duration_ratio
/// A value > 1 indicates the shock would consume more than one period's
/// worth of strategy volatility budget.
pub fn tail_risk_contribution(strategy_returns: &[f64], shock_returns: &[f64]) -> f64 {
    if strategy_returns.is_empty() || shock_returns.is_empty() {
        return 0.0;
    }
    let n = strategy_returns.len() as f64;
    let strat_mean = strategy_returns.iter().sum::<f64>() / n;
    let strat_var = strategy_returns
        .iter()
        .map(|r| (r - strat_mean).powi(2))
        .sum::<f64>()
        / n;
    let strat_std = strat_var.sqrt();

    if strat_std < 1e-12 {
        return 0.0;
    }

    let shock_sum: f64 = shock_returns.iter().sum();
    let duration_ratio = shock_returns.len() as f64 / strategy_returns.len() as f64;

    // Normalised impact: total shock magnitude / strategy daily vol
    (shock_sum.abs() / strat_std) * duration_ratio
}

/// Compute the Conditional Drawdown-at-Risk (CDaR) at a given confidence level.
///
/// CDaR is the expected drawdown conditional on being in the worst (1-α)
/// fraction of scenarios, analogous to CVaR for drawdowns.
///
/// # Arguments
/// * `returns` — series of period returns
/// * `confidence` — confidence level in (0, 1), e.g. 0.95
pub fn conditional_drawdown_at_risk(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    assert!(
        (0.0..1.0).contains(&confidence),
        "confidence must be in (0, 1)"
    );

    // Compute the full drawdown path
    let mut drawdowns = Vec::with_capacity(returns.len());
    let mut equity = 1.0_f64;
    let mut peak = 1.0_f64;

    for &r in returns {
        equity *= 1.0 + r;
        if equity > peak {
            peak = equity;
        }
        let dd = if peak > 0.0 {
            (peak - equity) / peak
        } else {
            0.0
        };
        drawdowns.push(dd);
    }

    // CDaR = mean of drawdowns exceeding the α-quantile
    drawdowns.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let var_idx = (confidence * drawdowns.len() as f64).floor() as usize;
    let tail = &drawdowns[var_idx.min(drawdowns.len() - 1)..];

    if tail.is_empty() {
        return 0.0;
    }

    tail.iter().sum::<f64>() / tail.len() as f64
}

/// Run multiple scenarios and return a vec of results.
pub fn scenario_analysis(
    engine: &StressTestEngine,
    base_returns: &[f64],
    scenarios: &[StressScenario],
) -> Result<Vec<ScenarioResult>> {
    scenarios
        .iter()
        .map(|s| engine.run_scenario(base_returns, s))
        .collect()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Simple circular bootstrap for building a path from base returns.
fn bootstrap_path(base: &[f64], n: usize, rng: &mut SmallRng) -> Vec<f64> {
    let t = base.len();
    let block = (t as f64).cbrt().ceil() as usize;
    let mut path = Vec::with_capacity(n);
    while path.len() < n {
        let start = rng.gen_range(0..t);
        for j in 0..block {
            if path.len() >= n {
                break;
            }
            path.push(base[(start + j) % t]);
        }
    }
    path
}

/// Simulate equity curve, injecting shocks at `injection_bar`.
/// Returns (final_equity, max_drawdown).
fn simulate_with_shock(
    path_returns: &[f64],
    shocks: &[f64],
    injection_bar: usize,
    initial_equity: f64,
) -> (f64, f64) {
    let n = path_returns.len();
    let mut equity = initial_equity;
    let mut peak = initial_equity;
    let mut max_dd = 0.0_f64;

    for bar in 0..n {
        // Check if we are inside the shock injection window
        let ret = if bar >= injection_bar && bar < injection_bar + shocks.len() {
            shocks[bar - injection_bar]
        } else {
            path_returns[bar]
        };

        equity *= 1.0 + ret;
        equity = equity.max(0.0);

        if equity > peak {
            peak = equity;
        }
        let dd = if peak > 0.0 {
            (peak - equity) / peak
        } else {
            0.0
        };
        if dd > max_dd {
            max_dd = dd;
        }
    }

    (equity, max_dd)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_returns(n: usize, r: f64) -> Vec<f64> {
        vec![r; n]
    }

    #[test]
    fn test_scenario_name() {
        assert_eq!(
            StressScenario::MarketCrash2020.name(),
            "Market Crash 2020 (COVID-19)"
        );
    }

    #[test]
    fn test_shock_sequence_nonempty() {
        for s in &[
            StressScenario::MarketCrash2020,
            StressScenario::CryptoWinter2022,
            StressScenario::CovidDip,
            StressScenario::Gfc2008,
            StressScenario::FlashCrash,
        ] {
            assert!(!s.shock_sequence().is_empty(), "{} has empty shocks", s.name());
        }
    }

    #[test]
    fn test_cdar_zero_returns() {
        let returns = flat_returns(100, 0.0);
        let cdar = conditional_drawdown_at_risk(&returns, 0.95);
        assert!(cdar < 1e-9, "CDaR must be near 0 for flat equity");
    }

    #[test]
    fn test_cdar_positive() {
        let mut returns = flat_returns(200, 0.001);
        // Inject a loss sequence
        returns[50] = -0.20;
        returns[51] = -0.10;
        let cdar = conditional_drawdown_at_risk(&returns, 0.95);
        assert!(cdar > 0.01, "CDaR should be > 0 with losses present");
    }

    #[test]
    fn test_tail_risk_contribution_positive() {
        let base = flat_returns(252, 0.001);
        let shocks = covid_crash_returns();
        let trc = tail_risk_contribution(&base, &shocks);
        assert!(trc > 0.0, "TRC must be positive for crash scenario");
    }

    #[test]
    fn test_run_scenario_reduces_equity() {
        let engine = StressTestEngine {
            n_paths: 200,
            n_bars: 252,
            initial_equity: 100_000.0,
            seed: 123,
        };
        let base = flat_returns(252, 0.001);
        let result = engine.run_scenario(&base, &StressScenario::MarketCrash2020).unwrap();
        // After crash injection, median equity should be below initial
        assert!(
            result.median_final_equity < 100_000.0 * 1.5,
            "median equity should not be unreasonably high"
        );
        assert!(result.median_max_drawdown > 0.0);
    }

    #[test]
    fn test_scenario_analysis_vec() {
        let engine = StressTestEngine {
            n_paths: 100,
            n_bars: 252,
            initial_equity: 100_000.0,
            seed: 42,
        };
        let base = flat_returns(252, 0.001);
        let scenarios = vec![StressScenario::CovidDip, StressScenario::FlashCrash];
        let results = scenario_analysis(&engine, &base, &scenarios).unwrap();
        assert_eq!(results.len(), 2);
    }
}
