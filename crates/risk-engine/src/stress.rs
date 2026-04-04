/// Stress testing: historical scenarios and Monte Carlo stress.

use std::collections::HashMap;

// ── Scenario ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Scenario {
    pub name: String,
    pub description: String,
    /// Per-asset shocked returns. Key = asset identifier.
    pub asset_shocks: HashMap<String, f64>,
    /// Duration of the shock in trading days.
    pub duration_days: usize,
    /// Correlation increase factor (>1 = correlations rise in stress).
    pub correlation_multiplier: f64,
}

impl Scenario {
    pub fn new(name: &str, description: &str, shocks: HashMap<String, f64>, duration: usize) -> Self {
        Scenario {
            name: name.to_string(),
            description: description.to_string(),
            asset_shocks: shocks,
            duration_days: duration,
            correlation_multiplier: 1.0,
        }
    }
}

// ── Historical Scenarios ──────────────────────────────────────────────────────

/// Build the 2008 Global Financial Crisis scenario.
pub fn scenario_2008_gfc() -> Scenario {
    let mut shocks = HashMap::new();
    shocks.insert("SPY".to_string(), -0.565);   // S&P 500 peak-to-trough
    shocks.insert("QQQ".to_string(), -0.507);
    shocks.insert("IEF".to_string(), 0.15);     // 10yr Treasury (flight to safety)
    shocks.insert("GLD".to_string(), 0.02);     // Gold modest
    shocks.insert("HYG".to_string(), -0.35);    // High yield credit
    shocks.insert("VIX".to_string(), 3.5);      // VIX tripled
    shocks.insert("USD".to_string(), 0.12);     // Dollar strength
    Scenario {
        name: "2008_GFC".to_string(),
        description: "2008 Global Financial Crisis (Sep 2008 - Mar 2009)".to_string(),
        asset_shocks: shocks,
        duration_days: 126,
        correlation_multiplier: 1.8,
    }
}

/// COVID crash (Feb–Mar 2020).
pub fn scenario_covid_crash() -> Scenario {
    let mut shocks = HashMap::new();
    shocks.insert("SPY".to_string(), -0.335);
    shocks.insert("QQQ".to_string(), -0.279);
    shocks.insert("IEF".to_string(), 0.08);
    shocks.insert("GLD".to_string(), -0.03);
    shocks.insert("HYG".to_string(), -0.22);
    shocks.insert("OIL".to_string(), -0.65);
    shocks.insert("VIX".to_string(), 4.5);
    Scenario {
        name: "COVID_crash".to_string(),
        description: "COVID-19 market crash (Feb 20 – Mar 23, 2020)".to_string(),
        asset_shocks: shocks,
        duration_days: 23,
        correlation_multiplier: 2.0,
    }
}

/// 2022 rate hike shock.
pub fn scenario_2022_rate_hike() -> Scenario {
    let mut shocks = HashMap::new();
    shocks.insert("SPY".to_string(), -0.195);
    shocks.insert("QQQ".to_string(), -0.329);
    shocks.insert("IEF".to_string(), -0.156);   // Bonds sold off
    shocks.insert("TLT".to_string(), -0.310);
    shocks.insert("GLD".to_string(), -0.01);
    shocks.insert("HYG".to_string(), -0.137);
    Scenario {
        name: "2022_rate_hike".to_string(),
        description: "2022 Fed rate hike cycle (Jan – Oct 2022)".to_string(),
        asset_shocks: shocks,
        duration_days: 210,
        correlation_multiplier: 1.5,
    }
}

/// Volmageddon (Feb 2018) — VIX ETF blowup.
pub fn scenario_volmageddon() -> Scenario {
    let mut shocks = HashMap::new();
    shocks.insert("SPY".to_string(), -0.108);
    shocks.insert("VIX".to_string(), 3.15);
    shocks.insert("SVXY".to_string(), -0.96); // Short vol ETF
    shocks.insert("XIV".to_string(), -0.96);
    shocks.insert("QQQ".to_string(), -0.096);
    Scenario {
        name: "Volmageddon".to_string(),
        description: "Volmageddon / short vol blowup (Feb 5-6, 2018)".to_string(),
        asset_shocks: shocks,
        duration_days: 2,
        correlation_multiplier: 2.5,
    }
}

/// Dot-com crash scenario.
pub fn scenario_dot_com() -> Scenario {
    let mut shocks = HashMap::new();
    shocks.insert("QQQ".to_string(), -0.83);
    shocks.insert("SPY".to_string(), -0.49);
    shocks.insert("IEF".to_string(), 0.20);
    shocks.insert("GLD".to_string(), -0.05);
    Scenario {
        name: "dot_com".to_string(),
        description: "Dot-com crash (Mar 2000 – Oct 2002)".to_string(),
        asset_shocks: shocks,
        duration_days: 632,
        correlation_multiplier: 1.3,
    }
}

/// Return all built-in historical scenarios.
pub fn all_scenarios() -> Vec<Scenario> {
    vec![
        scenario_2008_gfc(),
        scenario_covid_crash(),
        scenario_2022_rate_hike(),
        scenario_volmageddon(),
        scenario_dot_com(),
    ]
}

/// Create a user-defined scenario from a map of asset → shocked return.
pub fn user_defined_scenario(
    name: &str,
    description: &str,
    shocks: HashMap<String, f64>,
    duration_days: usize,
) -> Scenario {
    Scenario::new(name, description, shocks, duration_days)
}

// ── Apply Scenario ────────────────────────────────────────────────────────────

/// Apply a scenario to a portfolio and return the expected PnL.
///
/// * `portfolio_weights` — map of asset → weight.
/// * `scenario` — the stress scenario.
pub fn apply_scenario(portfolio_weights: &HashMap<String, f64>, scenario: &Scenario) -> f64 {
    portfolio_weights
        .iter()
        .map(|(asset, weight)| {
            let shock = scenario.asset_shocks.get(asset).copied().unwrap_or(0.0);
            weight * shock
        })
        .sum()
}

// ── Monte Carlo Stress ────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct StressResult {
    pub var_95: f64,
    pub var_99: f64,
    pub cvar_95: f64,
    pub cvar_99: f64,
    pub worst_case: f64,
    pub best_case: f64,
    pub portfolio_returns: Vec<f64>,
}

/// Monte Carlo stress test using a multivariate normal shock model.
///
/// `correlation_breakdown` = true applies a stress correlation multiplier.
pub fn monte_carlo_stress(
    weights: &[f64],
    asset_vols: &[f64],
    correlation_matrix: &[Vec<f64>],
    n_sims: usize,
    correlation_breakdown: bool,
) -> StressResult {
    let n = weights.len();
    assert_eq!(asset_vols.len(), n);
    assert_eq!(correlation_matrix.len(), n);

    // Cholesky decomposition of correlation matrix (with optional stress).
    let stress_factor = if correlation_breakdown { 1.5 } else { 1.0 };
    let mut stressed_corr: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    if i == j {
                        1.0
                    } else {
                        (correlation_matrix[i][j] * stress_factor).clamp(-0.999, 0.999)
                    }
                })
                .collect()
        })
        .collect();

    let chol = cholesky_decomp(&stressed_corr, n);

    // Simulate portfolio returns.
    let mut state = 77777_u64;
    let mut pseudo_normal = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = (state >> 11) as f64 / (1u64 << 53) as f64;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;
        let u1 = u1.max(1e-15);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    let mut portfolio_returns: Vec<f64> = Vec::with_capacity(n_sims);

    for _ in 0..n_sims {
        // Draw uncorrelated normals.
        let z: Vec<f64> = (0..n).map(|_| pseudo_normal()).collect();
        // Apply Cholesky to get correlated shocks.
        let correlated: Vec<f64> = (0..n)
            .map(|i| {
                (0..=i).map(|j| chol[i][j] * z[j]).sum::<f64>()
            })
            .collect();
        // Scale by volatilities and compute portfolio return.
        let port_ret: f64 = weights
            .iter()
            .zip(correlated.iter().zip(asset_vols.iter()))
            .map(|(w, (z_i, vol))| w * z_i * vol)
            .sum();
        portfolio_returns.push(port_ret);
    }

    portfolio_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let var_95_idx = ((1.0 - 0.95) * n_sims as f64) as usize;
    let var_99_idx = ((1.0 - 0.99) * n_sims as f64) as usize;

    let var_95 = -portfolio_returns[var_95_idx.min(n_sims - 1)];
    let var_99 = -portfolio_returns[var_99_idx.min(n_sims - 1)];

    let cvar_95 = -portfolio_returns[..var_95_idx.max(1)]
        .iter()
        .sum::<f64>()
        / var_95_idx.max(1) as f64;
    let cvar_99 = -portfolio_returns[..var_99_idx.max(1)]
        .iter()
        .sum::<f64>()
        / var_99_idx.max(1) as f64;

    StressResult {
        var_95,
        var_99,
        cvar_95,
        cvar_99,
        worst_case: -portfolio_returns[0],
        best_case: portfolio_returns[n_sims - 1],
        portfolio_returns,
    }
}

/// Cholesky decomposition (lower triangular L such that A = L * L').
fn cholesky_decomp(a: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut l = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in 0..=i {
            let sum: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();
            if i == j {
                l[i][j] = (a[i][i] - sum).max(0.0).sqrt();
            } else if l[j][j] > 1e-12 {
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }
    l
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_scenario_2008_buy_and_hold_spy() {
        let mut weights = HashMap::new();
        weights.insert("SPY".to_string(), 1.0);
        let s = scenario_2008_gfc();
        let pnl = apply_scenario(&weights, &s);
        assert!(pnl < -0.5 && pnl > -0.7, "pnl={pnl}");
    }

    #[test]
    fn monte_carlo_stress_produces_negative_var() {
        let weights = vec![0.5, 0.5];
        let vols = vec![0.02, 0.025];
        let corr = vec![vec![1.0, 0.5], vec![0.5, 1.0]];
        let result = monte_carlo_stress(&weights, &vols, &corr, 10_000, false);
        assert!(result.var_95 >= 0.0, "var95={}", result.var_95);
    }
}
