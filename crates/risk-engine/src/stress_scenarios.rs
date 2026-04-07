/// Stress scenario library: built-in historical scenarios, scenario application,
/// and reverse stress testing.

use std::collections::HashMap;

// ── Position / Portfolio types ────────────────────────────────────────────────

/// A single position in the portfolio.
#[derive(Debug, Clone)]
pub struct Position {
    /// Unique identifier (symbol / instrument ID).
    pub symbol: String,
    /// Market value in base currency (positive = long).
    pub market_value: f64,
    /// Current delta (price sensitivity per unit spot move, fraction).
    pub delta: f64,
    /// Current vega (price sensitivity per 1-unit vol move).
    pub vega: f64,
    /// Current gamma (second-order price sensitivity).
    pub gamma: f64,
}

/// A portfolio is a collection of positions.
#[derive(Debug, Clone, Default)]
pub struct Portfolio {
    pub positions: Vec<Position>,
    /// Total NAV / equity.
    pub nav: f64,
    /// Current margin balance.
    pub margin_balance: f64,
}

impl Portfolio {
    pub fn new(positions: Vec<Position>, nav: f64, margin_balance: f64) -> Self {
        Portfolio { positions, nav, margin_balance }
    }

    /// Net market value of all positions.
    pub fn net_market_value(&self) -> f64 {
        self.positions.iter().map(|p| p.market_value).sum()
    }
}

// ── Stress scenario definition ────────────────────────────────────────────────

/// A parameterized stress scenario.
#[derive(Debug, Clone)]
pub struct StressScenario {
    pub name: String,
    /// Fractional spot return (e.g. -0.35 = -35%).
    pub spot_shock: f64,
    /// Fractional vol shock applied to implied vol (e.g. 3.0 = +300%).
    pub vol_shock: f64,
    /// Absolute rate shock in decimal (e.g. 0.02 = +200bps).
    pub rate_shock: f64,
    /// Additive correlation shock (e.g. 0.30 = all pairwise correlations shift +0.30).
    pub correlation_shock: f64,
    /// Optional description.
    pub description: String,
}

impl StressScenario {
    /// Construct a named scenario.
    pub fn new(
        name: &str,
        spot_shock: f64,
        vol_shock: f64,
        rate_shock: f64,
        correlation_shock: f64,
        description: &str,
    ) -> Self {
        StressScenario {
            name: name.to_string(),
            spot_shock,
            vol_shock,
            rate_shock,
            correlation_shock,
            description: description.to_string(),
        }
    }
}

// ── Built-in scenarios ────────────────────────────────────────────────────────

/// Crypto winter 2022: BTC/ETH down ~75%, implied vol up 150%.
pub const CRYPTO_WINTER_2022: fn() -> StressScenario = || StressScenario::new(
    "CRYPTO_WINTER_2022",
    -0.75,
    1.50,
    0.005,
    0.20,
    "Crypto bear market 2022: spot -75%, IV +150%",
);

/// Flash crash May 2010: S&P 500 dropped ~10% in minutes.
pub const FLASH_CRASH_2010: fn() -> StressScenario = || StressScenario::new(
    "FLASH_CRASH_2010",
    -0.10,
    0.50,
    -0.005,
    0.15,
    "Flash crash May 6 2010: spot -10% instantaneous, IV spike",
);

/// COVID March 2020: equities dropped ~35%, VIX from 15 to 85 (+300% vol shock approximation).
pub const COVID_MARCH_2020: fn() -> StressScenario = || StressScenario::new(
    "COVID_MARCH_2020",
    -0.35,
    3.00,
    -0.015,
    0.30,
    "COVID crash March 2020: spot -35%, IV +300%",
);

/// LUNA/UST collapse May 2022: LUNA down 99%, vol up 500%.
pub const LUNA_COLLAPSE: fn() -> StressScenario = || StressScenario::new(
    "LUNA_COLLAPSE",
    -0.99,
    5.00,
    0.0,
    0.40,
    "LUNA/UST collapse May 2022: spot -99%, IV +500%",
);

/// Fed rate hike shock: rapid +200bps rate shock, equity correction -15%.
pub const FED_HIKE_SHOCK: fn() -> StressScenario = || StressScenario::new(
    "FED_HIKE_SHOCK",
    -0.15,
    0.40,
    0.02,
    0.10,
    "Fed rate hike shock: rate +200bps, spot -15%",
);

/// Correlation spike: all pairwise correlations converge to 0.95.
/// Spot shock mild, vol moderate.
pub const CORRELATION_SPIKE: fn() -> StressScenario = || StressScenario::new(
    "CORRELATION_SPIKE",
    -0.05,
    0.25,
    0.002,
    0.95, // Used as the target correlation rather than additive shift
    "Correlation crisis: all pairwise correlations -> 0.95",
);

/// Return all built-in scenarios as a vector.
pub fn all_builtin_scenarios() -> Vec<StressScenario> {
    vec![
        CRYPTO_WINTER_2022(),
        FLASH_CRASH_2010(),
        COVID_MARCH_2020(),
        LUNA_COLLAPSE(),
        FED_HIKE_SHOCK(),
        CORRELATION_SPIKE(),
    ]
}

// ── Stress result ─────────────────────────────────────────────────────────────

/// Result of applying a stress scenario to a portfolio.
#[derive(Debug, Clone)]
pub struct StressResult {
    /// Total portfolio P&L under the scenario.
    pub portfolio_pnl: f64,
    /// Symbol of the worst-performing position.
    pub worst_position: String,
    /// P&L of the worst-performing position.
    pub worst_position_pnl: f64,
    /// Whether a margin call would be triggered (portfolio PnL + margin_balance < 0).
    pub margin_call_triggered: bool,
    /// Scenario that was applied.
    pub scenario_name: String,
}

// ── Scenario application ──────────────────────────────────────────────────────

/// Apply a stress scenario to a portfolio and return the stress result.
/// Uses first-order Taylor expansion with delta and vega Greeks.
/// For a spot shock dS/S = spot_shock and vol shock dSigma/sigma = vol_shock,
/// position P&L = delta * market_value * spot_shock + vega * vol_shock * market_value.
pub fn apply_scenario(portfolio: &Portfolio, scenario: &StressScenario) -> StressResult {
    let mut total_pnl = 0.0_f64;
    let mut worst_pnl = f64::INFINITY;
    let mut worst_symbol = String::new();

    for pos in &portfolio.positions {
        // Delta component: P&L = position_value * delta * spot_shock
        let delta_pnl = pos.market_value * pos.delta * scenario.spot_shock;
        // Gamma component: 0.5 * gamma * (dS)^2 -- approximate dS = spot_shock * market_value
        let gamma_pnl = 0.5 * pos.gamma * (pos.market_value * scenario.spot_shock).powi(2);
        // Vega component: vega * vol_shock (absolute vol move)
        let vega_pnl = pos.vega * scenario.vol_shock;
        // Rate shock impact: approximate bond-like exposure -- for equity/options this is rho * rate_shock
        // We use a simplified factor: -0.5 * duration_proxy * rate_shock * market_value
        let rate_pnl = -0.5 * scenario.rate_shock * pos.market_value;

        let position_pnl = delta_pnl + gamma_pnl + vega_pnl + rate_pnl;
        total_pnl += position_pnl;

        if position_pnl < worst_pnl {
            worst_pnl = position_pnl;
            worst_symbol = pos.symbol.clone();
        }
    }

    let margin_call_triggered = total_pnl + portfolio.margin_balance < 0.0;

    StressResult {
        portfolio_pnl: total_pnl,
        worst_position: worst_symbol,
        worst_position_pnl: if worst_pnl.is_finite() { worst_pnl } else { 0.0 },
        margin_call_triggered,
        scenario_name: scenario.name.clone(),
    }
}

// ── Historical scenario replay ────────────────────────────────────────────────

/// A historical return path: ordered daily returns for a given symbol.
#[derive(Debug, Clone)]
pub struct HistoricalPath {
    pub symbol: String,
    /// Daily log returns.
    pub returns: Vec<f64>,
}

/// Replay a historical return path against the portfolio.
/// For each day, compute P&L as sum of (delta * price_change) across positions.
/// Returns a vector of daily P&L values.
pub fn replay_historical_scenario(
    portfolio: &Portfolio,
    paths: &[HistoricalPath],
) -> Vec<f64> {
    if paths.is_empty() || portfolio.positions.is_empty() {
        return vec![];
    }
    let n_days = paths[0].returns.len();
    // Build a map from symbol to return series.
    let return_map: HashMap<&str, &Vec<f64>> = paths
        .iter()
        .map(|p| (p.symbol.as_str(), &p.returns))
        .collect();

    (0..n_days)
        .map(|day| {
            portfolio.positions.iter().map(|pos| {
                let r = return_map
                    .get(pos.symbol.as_str())
                    .and_then(|v| v.get(day).copied())
                    .unwrap_or(0.0);
                // Simplified: P&L = market_value * delta * return
                pos.market_value * pos.delta * r
            }).sum::<f64>()
        })
        .collect()
}

// ── Reverse stress test ───────────────────────────────────────────────────────

/// Find the minimum spot shock (binary search) that causes a portfolio loss
/// of at least max_loss (expressed as a positive number -- a loss).
/// Uses vol_shock_ratio = vol_shock / spot_shock to maintain a consistent relationship.
pub fn find_breaking_scenario(
    portfolio: &Portfolio,
    max_loss: f64,
    vol_shock_ratio: f64,
) -> StressScenario {
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64; // 100% spot decline

    // First check if max_loss is achievable at all.
    let extreme = StressScenario::new("probe", -hi, hi * vol_shock_ratio, 0.0, 0.0, "");
    let result_extreme = apply_scenario(portfolio, &extreme);
    if -result_extreme.portfolio_pnl < max_loss {
        // Even full wipe-out doesn't reach max_loss -- return a null scenario.
        return StressScenario::new(
            "NO_BREAKING_SCENARIO",
            -1.0,
            vol_shock_ratio,
            0.0,
            0.0,
            "Portfolio cannot be broken by any spot-only shock",
        );
    }

    // Binary search for minimum shock magnitude.
    for _ in 0..60 {
        let mid = (lo + hi) / 2.0;
        let scenario = StressScenario::new(
            "binary_probe",
            -mid,
            mid * vol_shock_ratio,
            0.0,
            0.0,
            "",
        );
        let result = apply_scenario(portfolio, &scenario);
        // Loss is negative PnL.
        if -result.portfolio_pnl >= max_loss {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    let breaking_shock = (lo + hi) / 2.0;
    StressScenario::new(
        "BREAKING_SCENARIO",
        -breaking_shock,
        breaking_shock * vol_shock_ratio,
        0.0,
        0.0,
        &format!("Minimum shock causing loss >= {:.2}: spot -{:.4}%", max_loss, breaking_shock * 100.0),
    )
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_portfolio() -> Portfolio {
        Portfolio::new(
            vec![
                Position {
                    symbol: "BTC".to_string(),
                    market_value: 100_000.0,
                    delta: 1.0,
                    vega: 500.0,
                    gamma: 0.001,
                },
                Position {
                    symbol: "ETH".to_string(),
                    market_value: 50_000.0,
                    delta: 0.8,
                    vega: 200.0,
                    gamma: 0.002,
                },
            ],
            200_000.0,
            50_000.0,
        )
    }

    #[test]
    fn test_apply_scenario_large_negative_shock() {
        let port = test_portfolio();
        let scenario = LUNA_COLLAPSE();
        let result = apply_scenario(&port, &scenario);
        // 99% spot decline should cause a large loss.
        assert!(result.portfolio_pnl < -50_000.0, "LUNA collapse should cause large loss");
    }

    #[test]
    fn test_apply_scenario_margin_call() {
        let port = Portfolio::new(
            vec![Position {
                symbol: "LUNA".to_string(),
                market_value: 200_000.0,
                delta: 1.0,
                vega: 0.0,
                gamma: 0.0,
            }],
            200_000.0,
            10_000.0, // Only 10k margin
        );
        let result = apply_scenario(&port, &LUNA_COLLAPSE());
        assert!(result.margin_call_triggered, "Should trigger margin call");
    }

    #[test]
    fn test_apply_zero_shock_no_loss() {
        let port = test_portfolio();
        let scenario = StressScenario::new("ZERO", 0.0, 0.0, 0.0, 0.0, "");
        let result = apply_scenario(&port, &scenario);
        assert!(result.portfolio_pnl.abs() < 1e-6, "Zero shock should give zero PnL");
    }

    #[test]
    fn test_all_builtin_scenarios_count() {
        let scenarios = all_builtin_scenarios();
        assert_eq!(scenarios.len(), 6);
    }

    #[test]
    fn test_all_builtin_scenarios_have_names() {
        for s in all_builtin_scenarios() {
            assert!(!s.name.is_empty(), "Scenario must have a name");
        }
    }

    #[test]
    fn test_crypto_winter_spot_shock() {
        let s = CRYPTO_WINTER_2022();
        assert!((s.spot_shock + 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_fed_hike_rate_shock() {
        let s = FED_HIKE_SHOCK();
        assert!((s.rate_shock - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_find_breaking_scenario_exists() {
        let port = test_portfolio();
        // A small loss should be reachable.
        let breaking = find_breaking_scenario(&port, 10_000.0, 2.0);
        assert!(
            breaking.name != "NO_BREAKING_SCENARIO",
            "Should find a breaking scenario for a loss of 10k"
        );
        // Verify it actually breaks.
        let verify = apply_scenario(&port, &breaking);
        assert!(
            -verify.portfolio_pnl >= 10_000.0 - 1.0, // 1 unit tolerance
            "Breaking scenario must achieve target loss: pnl={}",
            verify.portfolio_pnl
        );
    }

    #[test]
    fn test_worst_position_identified() {
        let port = test_portfolio();
        let scenario = CRYPTO_WINTER_2022();
        let result = apply_scenario(&port, &scenario);
        // BTC has larger market value * delta, so should be worst.
        assert_eq!(result.worst_position, "BTC");
    }

    #[test]
    fn test_historical_replay_length() {
        let port = test_portfolio();
        let paths = vec![
            HistoricalPath {
                symbol: "BTC".to_string(),
                returns: vec![-0.05, 0.03, -0.02, 0.01, 0.04],
            },
        ];
        let pnl = replay_historical_scenario(&port, &paths);
        assert_eq!(pnl.len(), 5);
    }

    #[test]
    fn test_historical_replay_positive_return() {
        let port = Portfolio::new(
            vec![Position {
                symbol: "BTC".to_string(),
                market_value: 100_000.0,
                delta: 1.0,
                vega: 0.0,
                gamma: 0.0,
            }],
            100_000.0,
            0.0,
        );
        let paths = vec![HistoricalPath {
            symbol: "BTC".to_string(),
            returns: vec![0.10], // +10% day
        }];
        let pnl = replay_historical_scenario(&port, &paths);
        assert!((pnl[0] - 10_000.0).abs() < 1e-6, "Expected 10k PnL, got {}", pnl[0]);
    }
}
