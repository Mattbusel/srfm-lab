// stress.rs — Stress testing: historical scenarios, hypothetical, reverse stress, P&L

/// A stress scenario definition
#[derive(Clone, Debug)]
pub struct StressScenario {
    pub name: String,
    pub equity_shock: f64,       // e.g., -0.30 for -30%
    pub rate_shock_bp: f64,      // basis points change
    pub spread_shock_bp: f64,    // credit spread change bp
    pub vol_multiplier: f64,     // e.g., 2.0 for 2x vol
    pub fx_shock: f64,           // e.g., -0.10 for -10%
    pub commodity_shock: f64,    // e.g., -0.20
    pub correlation_override: Option<f64>, // override correlation to this value
    pub probability: f64,        // scenario probability weight
}

impl StressScenario {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            equity_shock: 0.0,
            rate_shock_bp: 0.0,
            spread_shock_bp: 0.0,
            vol_multiplier: 1.0,
            fx_shock: 0.0,
            commodity_shock: 0.0,
            correlation_override: None,
            probability: 1.0,
        }
    }

    /// 2008 Global Financial Crisis scenario
    pub fn gfc_2008() -> Self {
        Self {
            name: "GFC 2008".to_string(),
            equity_shock: -0.40,
            rate_shock_bp: -200.0,
            spread_shock_bp: 300.0,
            vol_multiplier: 3.0,
            fx_shock: -0.15,
            commodity_shock: -0.50,
            correlation_override: Some(0.8),
            probability: 0.01,
        }
    }

    /// 2020 COVID crash
    pub fn covid_2020() -> Self {
        Self {
            name: "COVID 2020".to_string(),
            equity_shock: -0.34,
            rate_shock_bp: -150.0,
            spread_shock_bp: 200.0,
            vol_multiplier: 4.0,
            fx_shock: -0.08,
            commodity_shock: -0.65,
            correlation_override: Some(0.85),
            probability: 0.02,
        }
    }

    /// 2022 Rate shock
    pub fn rate_shock_2022() -> Self {
        Self {
            name: "Rate Shock 2022".to_string(),
            equity_shock: -0.25,
            rate_shock_bp: 300.0,
            spread_shock_bp: 100.0,
            vol_multiplier: 2.0,
            fx_shock: 0.10,
            commodity_shock: 0.20,
            correlation_override: None,
            probability: 0.03,
        }
    }

    /// Flash crash scenario
    pub fn flash_crash() -> Self {
        Self {
            name: "Flash Crash".to_string(),
            equity_shock: -0.10,
            rate_shock_bp: -50.0,
            spread_shock_bp: 50.0,
            vol_multiplier: 5.0,
            fx_shock: -0.03,
            commodity_shock: -0.05,
            correlation_override: Some(0.95),
            probability: 0.05,
        }
    }

    /// Emerging market crisis
    pub fn em_crisis() -> Self {
        Self {
            name: "EM Crisis".to_string(),
            equity_shock: -0.30,
            rate_shock_bp: 200.0,
            spread_shock_bp: 500.0,
            vol_multiplier: 2.5,
            fx_shock: -0.25,
            commodity_shock: -0.30,
            correlation_override: Some(0.7),
            probability: 0.02,
        }
    }

    /// Stagflation scenario
    pub fn stagflation() -> Self {
        Self {
            name: "Stagflation".to_string(),
            equity_shock: -0.20,
            rate_shock_bp: 200.0,
            spread_shock_bp: 150.0,
            vol_multiplier: 1.5,
            fx_shock: -0.05,
            commodity_shock: 0.40,
            correlation_override: None,
            probability: 0.05,
        }
    }

    /// Custom rate shock
    pub fn rate_shock(bp: f64) -> Self {
        let mut s = Self::new("Rate Shock");
        s.rate_shock_bp = bp;
        // Estimate equity impact from rate shock
        let duration = 7.0; // assume 7yr duration for equity sensitivity
        s.equity_shock = -bp / 10000.0 * duration * 0.5;
        s.spread_shock_bp = bp * 0.3;
        s
    }

    /// Custom equity shock
    pub fn equity_shock(pct: f64) -> Self {
        let mut s = Self::new("Equity Shock");
        s.equity_shock = pct;
        s.vol_multiplier = 1.0 + (-pct * 3.0).min(4.0);
        s.spread_shock_bp = -pct * 500.0;
        s
    }

    /// Custom vol spike
    pub fn vol_spike(multiplier: f64) -> Self {
        let mut s = Self::new("Vol Spike");
        s.vol_multiplier = multiplier;
        s.equity_shock = -(multiplier - 1.0) * 0.05;
        s.correlation_override = Some(0.6 + (multiplier - 1.0) * 0.1);
        s
    }

    /// Correlation spike
    pub fn correlation_spike(target_corr: f64) -> Self {
        let mut s = Self::new("Correlation Spike");
        s.correlation_override = Some(target_corr);
        s.equity_shock = -(target_corr - 0.3) * 0.1;
        s.vol_multiplier = 1.0 + (target_corr - 0.3);
        s
    }
}

/// Portfolio position for stress testing
#[derive(Clone, Debug)]
pub struct StressPosition {
    pub name: String,
    pub asset_class: AssetClass,
    pub notional: f64,
    pub market_value: f64,
    pub duration: f64,        // for fixed income
    pub convexity: f64,       // for fixed income
    pub beta: f64,            // equity beta
    pub delta: f64,           // option delta
    pub gamma: f64,           // option gamma
    pub vega: f64,            // option vega
    pub fx_exposure: f64,     // FX notional
    pub commodity_exposure: f64,
    pub credit_spread_dv01: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AssetClass {
    Equity,
    FixedIncome,
    FX,
    Commodity,
    Option,
    CreditDerivative,
}

impl StressPosition {
    pub fn equity(name: &str, market_value: f64, beta: f64) -> Self {
        Self {
            name: name.to_string(), asset_class: AssetClass::Equity,
            notional: market_value, market_value, duration: 0.0, convexity: 0.0,
            beta, delta: 1.0, gamma: 0.0, vega: 0.0, fx_exposure: 0.0,
            commodity_exposure: 0.0, credit_spread_dv01: 0.0,
        }
    }

    pub fn bond(name: &str, market_value: f64, duration: f64, convexity: f64, spread_dv01: f64) -> Self {
        Self {
            name: name.to_string(), asset_class: AssetClass::FixedIncome,
            notional: market_value, market_value, duration, convexity,
            beta: 0.0, delta: 0.0, gamma: 0.0, vega: 0.0, fx_exposure: 0.0,
            commodity_exposure: 0.0, credit_spread_dv01: spread_dv01,
        }
    }

    pub fn option(name: &str, market_value: f64, delta: f64, gamma: f64, vega: f64) -> Self {
        Self {
            name: name.to_string(), asset_class: AssetClass::Option,
            notional: market_value, market_value, duration: 0.0, convexity: 0.0,
            beta: 0.0, delta, gamma, vega, fx_exposure: 0.0,
            commodity_exposure: 0.0, credit_spread_dv01: 0.0,
        }
    }

    pub fn fx(name: &str, notional: f64) -> Self {
        Self {
            name: name.to_string(), asset_class: AssetClass::FX,
            notional, market_value: notional, duration: 0.0, convexity: 0.0,
            beta: 0.0, delta: 0.0, gamma: 0.0, vega: 0.0, fx_exposure: notional,
            commodity_exposure: 0.0, credit_spread_dv01: 0.0,
        }
    }

    pub fn commodity(name: &str, notional: f64) -> Self {
        Self {
            name: name.to_string(), asset_class: AssetClass::Commodity,
            notional, market_value: notional, duration: 0.0, convexity: 0.0,
            beta: 0.0, delta: 0.0, gamma: 0.0, vega: 0.0, fx_exposure: 0.0,
            commodity_exposure: notional, credit_spread_dv01: 0.0,
        }
    }

    /// Compute P&L for a single position under a stress scenario
    pub fn stress_pnl(&self, scenario: &StressScenario) -> f64 {
        let mut pnl = 0.0;

        match self.asset_class {
            AssetClass::Equity => {
                pnl += self.market_value * self.beta * scenario.equity_shock;
            }
            AssetClass::FixedIncome => {
                let rate_change = scenario.rate_shock_bp / 10000.0;
                // Duration + convexity approximation
                pnl += self.market_value * (-self.duration * rate_change
                    + 0.5 * self.convexity * rate_change * rate_change);
                // Spread impact
                pnl -= self.credit_spread_dv01 * scenario.spread_shock_bp;
            }
            AssetClass::Option => {
                // Delta + Gamma + Vega
                let underlying_move = scenario.equity_shock;
                pnl += self.delta * self.notional * underlying_move;
                pnl += 0.5 * self.gamma * self.notional * underlying_move * underlying_move;
                // Vega: vol change in percentage points
                let vol_change = (scenario.vol_multiplier - 1.0) * 20.0; // assume 20% base vol
                pnl += self.vega * vol_change;
            }
            AssetClass::FX => {
                pnl += self.fx_exposure * scenario.fx_shock;
            }
            AssetClass::Commodity => {
                pnl += self.commodity_exposure * scenario.commodity_shock;
            }
            AssetClass::CreditDerivative => {
                pnl -= self.credit_spread_dv01 * scenario.spread_shock_bp;
            }
        }

        pnl
    }
}

/// Stress test result
#[derive(Clone, Debug)]
pub struct StressResult {
    pub scenario_name: String,
    pub total_pnl: f64,
    pub position_pnls: Vec<(String, f64)>,
    pub pnl_pct: f64,
}

/// Run stress test on portfolio
pub fn run_stress_test(positions: &[StressPosition], scenario: &StressScenario) -> StressResult {
    let total_mv: f64 = positions.iter().map(|p| p.market_value.abs()).sum();
    let mut position_pnls = Vec::with_capacity(positions.len());
    let mut total_pnl = 0.0;

    for pos in positions {
        let pnl = pos.stress_pnl(scenario);
        total_pnl += pnl;
        position_pnls.push((pos.name.clone(), pnl));
    }

    let pnl_pct = if total_mv > 1e-15 { total_pnl / total_mv } else { 0.0 };

    StressResult {
        scenario_name: scenario.name.clone(),
        total_pnl,
        position_pnls,
        pnl_pct,
    }
}

/// Run multiple stress scenarios
pub fn run_stress_battery(positions: &[StressPosition], scenarios: &[StressScenario]) -> Vec<StressResult> {
    scenarios.iter().map(|s| run_stress_test(positions, s)).collect()
}

/// Standard stress battery (all predefined scenarios)
pub fn standard_stress_battery() -> Vec<StressScenario> {
    vec![
        StressScenario::gfc_2008(),
        StressScenario::covid_2020(),
        StressScenario::rate_shock_2022(),
        StressScenario::flash_crash(),
        StressScenario::em_crisis(),
        StressScenario::stagflation(),
        StressScenario::rate_shock(300.0),
        StressScenario::rate_shock(-200.0),
        StressScenario::equity_shock(-0.30),
        StressScenario::equity_shock(-0.50),
        StressScenario::vol_spike(2.0),
        StressScenario::vol_spike(3.0),
        StressScenario::correlation_spike(0.9),
    ]
}

/// Probability-weighted expected loss across scenarios
pub fn probability_weighted_loss(results: &[StressResult], scenarios: &[StressScenario]) -> f64 {
    let mut expected_loss = 0.0;
    let total_prob: f64 = scenarios.iter().map(|s| s.probability).sum();
    for (result, scenario) in results.iter().zip(scenarios) {
        let p = scenario.probability / total_prob;
        if result.total_pnl < 0.0 {
            expected_loss += p * (-result.total_pnl);
        }
    }
    expected_loss
}

/// Worst-case scenario from battery
pub fn worst_case_scenario(results: &[StressResult]) -> Option<&StressResult> {
    results.iter().min_by(|a, b| a.total_pnl.partial_cmp(&b.total_pnl).unwrap_or(std::cmp::Ordering::Equal))
}

/// Reverse stress test: find the minimum shock magnitude that causes target loss
pub fn reverse_stress_equity(
    positions: &[StressPosition], target_loss: f64,
) -> f64 {
    // Binary search for equity shock that produces target loss
    let mut lo = -0.99;
    let mut hi = 0.0;
    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        let scenario = StressScenario::equity_shock(mid);
        let result = run_stress_test(positions, &scenario);
        if -result.total_pnl > target_loss { hi = mid; } else { lo = mid; }
        if (hi - lo).abs() < 1e-6 { break; }
    }
    0.5 * (lo + hi)
}

/// Reverse stress for rates
pub fn reverse_stress_rates(
    positions: &[StressPosition], target_loss: f64,
) -> f64 {
    // Search over rate shocks
    let mut lo = -500.0;
    let mut hi = 500.0;

    // First determine direction
    let up_scenario = StressScenario::rate_shock(300.0);
    let down_scenario = StressScenario::rate_shock(-300.0);
    let up_pnl = run_stress_test(positions, &up_scenario).total_pnl;
    let down_pnl = run_stress_test(positions, &down_scenario).total_pnl;

    if -up_pnl > -down_pnl {
        lo = 0.0; hi = 1000.0;
    } else {
        lo = -1000.0; hi = 0.0;
    }

    for _ in 0..100 {
        let mid = 0.5 * (lo + hi);
        let scenario = StressScenario::rate_shock(mid);
        let result = run_stress_test(positions, &scenario);
        if -result.total_pnl > target_loss { hi = mid; } else { lo = mid; }
        if (hi - lo).abs() < 0.1 { break; }
    }
    0.5 * (lo + hi)
}

/// Scenario sensitivity analysis: vary one factor while keeping others at base
pub fn sensitivity_analysis(
    positions: &[StressPosition],
    factor: &str,
    range: &[f64],
) -> Vec<(f64, f64)> {
    range.iter().map(|&shock| {
        let scenario = match factor {
            "equity" => StressScenario::equity_shock(shock),
            "rates" => StressScenario::rate_shock(shock),
            "vol" => StressScenario::vol_spike(shock),
            "fx" => {
                let mut s = StressScenario::new("FX Shock");
                s.fx_shock = shock;
                s
            }
            "commodity" => {
                let mut s = StressScenario::new("Commodity Shock");
                s.commodity_shock = shock;
                s
            }
            "spread" => {
                let mut s = StressScenario::new("Spread Shock");
                s.spread_shock_bp = shock;
                s
            }
            _ => StressScenario::new("Unknown"),
        };
        let result = run_stress_test(positions, &scenario);
        (shock, result.total_pnl)
    }).collect()
}

/// Two-factor stress: combine equity and rate shocks
pub fn two_factor_stress(
    positions: &[StressPosition],
    equity_shocks: &[f64],
    rate_shocks: &[f64],
) -> Vec<Vec<f64>> {
    let mut grid = Vec::with_capacity(equity_shocks.len());
    for &eq in equity_shocks {
        let mut row = Vec::with_capacity(rate_shocks.len());
        for &rt in rate_shocks {
            let mut scenario = StressScenario::new("Two-Factor");
            scenario.equity_shock = eq;
            scenario.rate_shock_bp = rt;
            scenario.spread_shock_bp = rt * 0.3 - eq * 500.0;
            scenario.vol_multiplier = 1.0 + (-eq * 3.0).min(4.0);
            let result = run_stress_test(positions, &scenario);
            row.push(result.total_pnl);
        }
        grid.push(row);
    }
    grid
}

/// Maximum loss across a grid of scenarios
pub fn max_loss_grid(grid: &[Vec<f64>]) -> f64 {
    grid.iter().flat_map(|row| row.iter())
        .copied()
        .fold(f64::INFINITY, f64::min)
}

/// Stress test P&L attribution by asset class
pub fn pnl_attribution_by_class(positions: &[StressPosition], scenario: &StressScenario) -> Vec<(String, f64)> {
    let classes = ["Equity", "FixedIncome", "FX", "Commodity", "Option", "CreditDerivative"];
    classes.iter().map(|&cls| {
        let pnl: f64 = positions.iter()
            .filter(|p| format!("{:?}", p.asset_class) == cls)
            .map(|p| p.stress_pnl(scenario))
            .sum();
        (cls.to_string(), pnl)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stress_pnl() {
        let pos = StressPosition::equity("SPY", 1_000_000.0, 1.0);
        let scenario = StressScenario::gfc_2008();
        let pnl = pos.stress_pnl(&scenario);
        assert!((pnl - (-400_000.0)).abs() < 1.0);
    }

    #[test]
    fn test_bond_stress() {
        let pos = StressPosition::bond("UST 10Y", 1_000_000.0, 8.0, 75.0, 80.0);
        let scenario = StressScenario::rate_shock(100.0);
        let pnl = pos.stress_pnl(&scenario);
        assert!(pnl < 0.0); // rates up → bond price down
    }

    #[test]
    fn test_stress_battery() {
        let portfolio = vec![
            StressPosition::equity("SPY", 500_000.0, 1.0),
            StressPosition::bond("UST", 300_000.0, 7.0, 60.0, 50.0),
        ];
        let results = run_stress_battery(&portfolio, &standard_stress_battery());
        assert!(results.len() > 5);
    }
}
