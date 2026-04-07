/// Scenario P&L analysis: grid-based scenario matrix, Gamma/theta decomposition.

use greeks_local::{bs_price, delta, gamma, vega, theta};

// ── Local re-export of BS math to avoid cross-module coupling ─────────────────

mod greeks_local {
    use std::f64::consts::PI;

    fn norm_cdf(x: f64) -> f64 {
        if x < -8.0 { return 0.0; }
        if x > 8.0  { return 1.0; }
        let t = 1.0 / (1.0 + 0.3275911_f64 * x.abs());
        let poly = t * (0.254829592_f64
            + t * (-0.284496736_f64
                + t * (1.421413741_f64
                    + t * (-1.453152027_f64 + t * 1.061405429_f64))));
        let y = 1.0 - poly * (-x * x).exp();
        let raw = if x < 0.0 { (1.0 - y) / 2.0 } else { (1.0 + y) / 2.0 };
        raw.clamp(0.0, 1.0)
    }

    pub fn norm_pdf(x: f64) -> f64 {
        (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
    }

    fn d1_d2(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> Option<(f64, f64)> {
        if s <= 0.0 || k <= 0.0 || t <= 0.0 || sigma <= 0.0 { return None; }
        let sq = t.sqrt();
        let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * sq);
        let d2 = d1 - sigma * sq;
        Some((d1, d2))
    }

    pub fn bs_price(s: f64, k: f64, t: f64, r: f64, sigma: f64, is_call: bool) -> f64 {
        match d1_d2(s, k, t, r, sigma) {
            Some((d1, d2)) => {
                let disc = (-r * t).exp();
                if is_call {
                    s * norm_cdf(d1) - k * disc * norm_cdf(d2)
                } else {
                    k * disc * norm_cdf(-d2) - s * norm_cdf(-d1)
                }
            }
            None => if is_call { (s - k).max(0.0) } else { (k - s).max(0.0) },
        }
    }

    pub fn delta(s: f64, k: f64, t: f64, r: f64, sigma: f64, is_call: bool) -> f64 {
        match d1_d2(s, k, t, r, sigma) {
            Some((d1, _)) => if is_call { norm_cdf(d1) } else { norm_cdf(d1) - 1.0 },
            None => 0.0,
        }
    }

    pub fn gamma(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
        match d1_d2(s, k, t, r, sigma) {
            Some((d1, _)) => norm_pdf(d1) / (s * sigma * t.sqrt()),
            None => 0.0,
        }
    }

    pub fn vega(s: f64, k: f64, t: f64, r: f64, sigma: f64) -> f64 {
        match d1_d2(s, k, t, r, sigma) {
            Some((d1, _)) => s * norm_pdf(d1) * t.sqrt(),
            None => 0.0,
        }
    }

    pub fn theta(s: f64, k: f64, t: f64, r: f64, sigma: f64, is_call: bool) -> f64 {
        match d1_d2(s, k, t, r, sigma) {
            Some((d1, d2)) => {
                let term1 = -(s * norm_pdf(d1) * sigma) / (2.0 * t.sqrt());
                if is_call {
                    (term1 - r * k * (-r * t).exp() * norm_cdf(d2)) / 365.0
                } else {
                    (term1 + r * k * (-r * t).exp() * norm_cdf(-d2)) / 365.0
                }
            }
            None => 0.0,
        }
    }
}

// ── Re-use OptionSpec from greeks_matrix ─────────────────────────────────────

/// Specification for a single option position.
#[derive(Debug, Clone, Copy)]
pub struct OptionSpec {
    pub strike: f64,
    /// Time to expiry in years.
    pub expiry: f64,
    pub is_call: bool,
    /// Signed position size.
    pub position: f64,
}

// ── Scenario grid ─────────────────────────────────────────────────────────────

/// Grid of spot and vol shocks to evaluate.
/// Shocks are fractional, e.g. 0.10 = +10%, -0.20 = -20%.
#[derive(Debug, Clone)]
pub struct ScenarioGrid {
    /// Fractional spot shocks applied to the current spot.
    pub spot_shocks: Vec<f64>,
    /// Fractional vol shocks applied to the current vol.
    pub vol_shocks: Vec<f64>,
}

impl ScenarioGrid {
    /// Standard grid: spot +-{5,10,15,20}%, vol +-{25,50}%.
    pub fn standard() -> Self {
        ScenarioGrid {
            spot_shocks: vec![-0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20],
            vol_shocks: vec![-0.50, -0.25, 0.0, 0.25, 0.50],
        }
    }

    /// Crypto-style grid: wider spot shocks, larger vol shocks.
    pub fn crypto() -> Self {
        ScenarioGrid {
            spot_shocks: vec![-0.50, -0.30, -0.20, -0.10, 0.0, 0.10, 0.20, 0.30, 0.50],
            vol_shocks: vec![-0.50, -0.25, 0.0, 0.50, 1.00, 2.00],
        }
    }
}

// ── Scenario matrix result ────────────────────────────────────────────────────

/// 2D P&L matrix: rows = spot shocks, columns = vol shocks.
#[derive(Debug, Clone)]
pub struct ScenarioMatrix {
    pub spot_shocks: Vec<f64>,
    pub vol_shocks: Vec<f64>,
    /// pnl[i][j] = P&L for spot_shocks[i] and vol_shocks[j].
    pub pnl: Vec<Vec<f64>>,
}

impl ScenarioMatrix {
    /// Find the minimum P&L across all scenarios.
    pub fn worst_case(&self) -> f64 {
        self.pnl
            .iter()
            .flat_map(|row| row.iter().copied())
            .fold(f64::INFINITY, f64::min)
    }

    /// Find the maximum P&L across all scenarios.
    pub fn best_case(&self) -> f64 {
        self.pnl
            .iter()
            .flat_map(|row| row.iter().copied())
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Return the (spot_shock, vol_shock) pair for the worst-case cell.
    pub fn worst_case_scenario(&self) -> (f64, f64) {
        let mut worst = f64::INFINITY;
        let mut ws = 0.0;
        let mut wv = 0.0;
        for (i, row) in self.pnl.iter().enumerate() {
            for (j, &p) in row.iter().enumerate() {
                if p < worst {
                    worst = p;
                    ws = self.spot_shocks[i];
                    wv = self.vol_shocks[j];
                }
            }
        }
        (ws, wv)
    }
}

// ── ScenarioEngine ────────────────────────────────────────────────────────────

/// Computes P&L under various spot/vol scenarios.
pub struct ScenarioEngine {
    /// Risk-free rate used in pricing.
    pub rf: f64,
}

impl ScenarioEngine {
    pub fn new(rf: f64) -> Self {
        ScenarioEngine { rf }
    }

    /// Compute portfolio value at a single (spot, vol, optional dt in years).
    fn portfolio_value(&self, portfolio: &[OptionSpec], spot: f64, vol: f64) -> f64 {
        portfolio.iter().map(|spec| {
            let price = bs_price(spot, spec.strike, spec.expiry, self.rf, vol.max(1e-6), spec.is_call);
            spec.position * price
        }).sum()
    }

    /// Full scenario P&L matrix.
    pub fn scenario_pnl(
        &self,
        portfolio: &[OptionSpec],
        spot: f64,
        vol: f64,
        grid: &ScenarioGrid,
    ) -> ScenarioMatrix {
        let base_value = self.portfolio_value(portfolio, spot, vol);
        let n_spot = grid.spot_shocks.len();
        let n_vol = grid.vol_shocks.len();
        let mut pnl = vec![vec![0.0_f64; n_vol]; n_spot];
        for (i, &ds) in grid.spot_shocks.iter().enumerate() {
            let shocked_spot = spot * (1.0 + ds);
            for (j, &dv) in grid.vol_shocks.iter().enumerate() {
                let shocked_vol = (vol * (1.0 + dv)).max(1e-6);
                let shocked_value = self.portfolio_value(portfolio, shocked_spot, shocked_vol);
                pnl[i][j] = shocked_value - base_value;
            }
        }
        ScenarioMatrix {
            spot_shocks: grid.spot_shocks.clone(),
            vol_shocks: grid.vol_shocks.clone(),
            pnl,
        }
    }

    /// Compute scenario P&L including time decay (dt in years).
    pub fn scenario_pnl_with_decay(
        &self,
        portfolio: &[OptionSpec],
        spot: f64,
        vol: f64,
        grid: &ScenarioGrid,
        dt_years: f64,
    ) -> ScenarioMatrix {
        let base_value = self.portfolio_value(portfolio, spot, vol);
        let n_spot = grid.spot_shocks.len();
        let n_vol = grid.vol_shocks.len();
        let mut pnl = vec![vec![0.0_f64; n_vol]; n_spot];
        for (i, &ds) in grid.spot_shocks.iter().enumerate() {
            let shocked_spot = spot * (1.0 + ds);
            for (j, &dv) in grid.vol_shocks.iter().enumerate() {
                let shocked_vol = (vol * (1.0 + dv)).max(1e-6);
                // Compute with reduced time to expiry.
                let aged_portfolio: Vec<OptionSpec> = portfolio.iter().map(|spec| OptionSpec {
                    expiry: (spec.expiry - dt_years).max(1e-6),
                    ..*spec
                }).collect();
                let shocked_value = self.portfolio_value(&aged_portfolio, shocked_spot, shocked_vol);
                pnl[i][j] = shocked_value - base_value;
            }
        }
        ScenarioMatrix {
            spot_shocks: grid.spot_shocks.clone(),
            vol_shocks: grid.vol_shocks.clone(),
            pnl,
        }
    }
}

// ── Standalone scenario_pnl function ─────────────────────────────────────────

/// Compute scenario P&L matrix with default rf = 0.05.
pub fn scenario_pnl(
    portfolio: &[OptionSpec],
    spot: f64,
    vol: f64,
    grid: &ScenarioGrid,
) -> ScenarioMatrix {
    ScenarioEngine::new(0.05).scenario_pnl(portfolio, spot, vol, grid)
}

// ── P&L decomposition ─────────────────────────────────────────────────────────

/// Components of a Taylor-expanded P&L.
#[derive(Debug, Clone, Copy, Default)]
pub struct PnLDecomposition {
    /// P&L from directional move (delta * dS).
    pub delta_pnl: f64,
    /// P&L from convexity (0.5 * gamma * dS^2).
    pub gamma_pnl: f64,
    /// P&L from vol change (vega * dVol).
    pub vega_pnl: f64,
    /// P&L from time decay (theta * dt).
    pub theta_pnl: f64,
    /// Cross-term: vanna * dS * dVol.
    pub cross_terms: f64,
    /// Sum of all components.
    pub total: f64,
}

/// Decompose P&L into Greeks contributions for a finite move (dS, dVol, dt in years).
/// Uses first-order Taylor expansion with gamma convexity and vanna cross-term.
pub fn decompose_pnl(
    portfolio: &[OptionSpec],
    spot: f64,
    vol: f64,
    rf: f64,
    d_s: f64,
    d_vol: f64,
    dt: f64,
) -> PnLDecomposition {
    let mut d_delta = 0.0_f64;
    let mut d_gamma = 0.0_f64;
    let mut d_vega = 0.0_f64;
    let mut d_theta = 0.0_f64;
    // Vanna: d(delta)/d(sigma) -- cross-term coefficient.
    let mut d_vanna = 0.0_f64;

    for spec in portfolio {
        let pos = spec.position;
        let k = spec.strike;
        let t = spec.expiry;
        let v = vega(spot, k, t, rf, vol);
        let g = gamma(spot, k, t, rf, vol);
        let th = theta(spot, k, t, rf, vol, spec.is_call);
        let dl = delta(spot, k, t, rf, vol, spec.is_call);

        // Vanna: dv/dS = norm_pdf(d1)*d2/(S*sigma) ... use finite difference.
        let eps = spot * 0.001;
        let vega_up = vega(spot + eps, k, t, rf, vol);
        let vega_dn = vega(spot - eps, k, t, rf, vol);
        let vanna_val = (vega_up - vega_dn) / (2.0 * eps);

        d_delta += pos * dl;
        d_gamma += pos * g;
        d_vega  += pos * v;
        d_theta += pos * th;
        d_vanna += pos * vanna_val;
    }

    let delta_pnl = d_delta * d_s;
    let gamma_pnl = 0.5 * d_gamma * d_s * d_s;
    let vega_pnl  = d_vega * d_vol;
    let theta_pnl = d_theta * dt * 365.0; // theta is per-day, dt is years
    let cross_terms = d_vanna * d_s * d_vol;
    let total = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + cross_terms;

    PnLDecomposition { delta_pnl, gamma_pnl, vega_pnl, theta_pnl, cross_terms, total }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn straddle(spot: f64, expiry: f64) -> Vec<OptionSpec> {
        vec![
            OptionSpec { strike: spot, expiry, is_call: true,  position: 1.0 },
            OptionSpec { strike: spot, expiry, is_call: false, position: 1.0 },
        ]
    }

    #[test]
    fn test_scenario_matrix_dimensions() {
        let port = straddle(100.0, 0.25);
        let grid = ScenarioGrid::standard();
        let mat = scenario_pnl(&port, 100.0, 0.20, &grid);
        assert_eq!(mat.pnl.len(), grid.spot_shocks.len());
        assert_eq!(mat.pnl[0].len(), grid.vol_shocks.len());
    }

    #[test]
    fn test_scenario_zero_shock_zero_pnl() {
        let port = straddle(100.0, 0.25);
        let grid = ScenarioGrid {
            spot_shocks: vec![0.0],
            vol_shocks: vec![0.0],
        };
        let mat = scenario_pnl(&port, 100.0, 0.20, &grid);
        assert!(mat.pnl[0][0].abs() < 1e-10, "Zero shock must give zero PnL");
    }

    #[test]
    fn test_straddle_symmetric_pnl() {
        let port = straddle(100.0, 0.25);
        let grid = ScenarioGrid {
            spot_shocks: vec![-0.10, 0.10],
            vol_shocks: vec![0.0],
        };
        let mat = scenario_pnl(&port, 100.0, 0.20, &grid);
        // Straddle P&L should be similar for equal up/down moves.
        let pnl_dn = mat.pnl[0][0];
        let pnl_up = mat.pnl[1][0];
        assert!(
            (pnl_dn - pnl_up).abs() < 1.0,
            "Straddle symmetry violated: dn={}, up={}",
            pnl_dn, pnl_up
        );
    }

    #[test]
    fn test_long_call_benefits_from_spot_up() {
        let port = vec![OptionSpec { strike: 100.0, expiry: 0.25, is_call: true, position: 1.0 }];
        let grid = ScenarioGrid {
            spot_shocks: vec![-0.10, 0.0, 0.10],
            vol_shocks: vec![0.0],
        };
        let mat = scenario_pnl(&port, 100.0, 0.20, &grid);
        assert!(mat.pnl[2][0] > 0.0, "Long call profits when spot rises");
        assert!(mat.pnl[0][0] < 0.0, "Long call loses when spot falls");
    }

    #[test]
    fn test_vol_increase_benefits_long_straddle() {
        let port = straddle(100.0, 0.25);
        let grid = ScenarioGrid {
            spot_shocks: vec![0.0],
            vol_shocks: vec![-0.50, 0.0, 0.50],
        };
        let mat = scenario_pnl(&port, 100.0, 0.20, &grid);
        // Vol up should improve P&L.
        assert!(mat.pnl[0][2] > mat.pnl[0][1], "Higher vol improves long straddle P&L");
        assert!(mat.pnl[0][0] < mat.pnl[0][1], "Lower vol hurts long straddle P&L");
    }

    #[test]
    fn test_worst_case_is_minimum() {
        let port = straddle(100.0, 0.25);
        let grid = ScenarioGrid::standard();
        let mat = scenario_pnl(&port, 100.0, 0.20, &grid);
        let wc = mat.worst_case();
        for row in &mat.pnl {
            for &p in row {
                assert!(p >= wc - 1e-10);
            }
        }
    }

    #[test]
    fn test_best_case_is_maximum() {
        let port = straddle(100.0, 0.25);
        let grid = ScenarioGrid::standard();
        let mat = scenario_pnl(&port, 100.0, 0.20, &grid);
        let bc = mat.best_case();
        for row in &mat.pnl {
            for &p in row {
                assert!(p <= bc + 1e-10);
            }
        }
    }

    #[test]
    fn test_decompose_pnl_delta_dominates_small_move() {
        let port = vec![OptionSpec { strike: 100.0, expiry: 0.25, is_call: true, position: 1.0 }];
        let d = decompose_pnl(&port, 100.0, 0.20, 0.05, 1.0, 0.0, 0.0);
        // For a small spot move, delta PnL should be largest component.
        assert!(d.delta_pnl.abs() > d.gamma_pnl.abs());
    }

    #[test]
    fn test_decompose_pnl_gamma_positive_long_call() {
        let port = vec![OptionSpec { strike: 100.0, expiry: 0.25, is_call: true, position: 1.0 }];
        let d = decompose_pnl(&port, 100.0, 0.20, 0.05, 5.0, 0.0, 0.0);
        // Long option has positive gamma PnL for any finite spot move.
        assert!(d.gamma_pnl > 0.0, "Gamma PnL must be positive: {}", d.gamma_pnl);
    }

    #[test]
    fn test_decompose_pnl_vega_positive_for_vol_up() {
        let port = vec![OptionSpec { strike: 100.0, expiry: 0.25, is_call: true, position: 1.0 }];
        let d = decompose_pnl(&port, 100.0, 0.20, 0.05, 0.0, 0.05, 0.0);
        assert!(d.vega_pnl > 0.0, "Long call benefits from vol increase");
    }

    #[test]
    fn test_decompose_pnl_theta_negative_per_day() {
        let port = vec![OptionSpec { strike: 100.0, expiry: 0.25, is_call: true, position: 1.0 }];
        // dt = 1 day = 1/365 years.
        let d = decompose_pnl(&port, 100.0, 0.20, 0.05, 0.0, 0.0, 1.0 / 365.0);
        assert!(d.theta_pnl < 0.0, "Long call decays over time: {}", d.theta_pnl);
    }

    #[test]
    fn test_scenario_grid_crypto_dimensions() {
        let grid = ScenarioGrid::crypto();
        let port = straddle(100.0, 0.25);
        let mat = scenario_pnl(&port, 100.0, 0.80, &grid);
        assert_eq!(mat.pnl.len(), grid.spot_shocks.len());
        assert_eq!(mat.pnl[0].len(), grid.vol_shocks.len());
    }

    #[test]
    fn test_scenario_engine_with_decay() {
        let port = straddle(100.0, 0.25);
        let engine = ScenarioEngine::new(0.05);
        let grid = ScenarioGrid {
            spot_shocks: vec![0.0],
            vol_shocks: vec![0.0],
        };
        let mat = engine.scenario_pnl_with_decay(&port, 100.0, 0.20, &grid, 1.0 / 365.0);
        // With time decay and no other moves, long straddle should lose value.
        assert!(mat.pnl[0][0] < 0.0, "Long straddle should decay: {}", mat.pnl[0][0]);
    }
}
