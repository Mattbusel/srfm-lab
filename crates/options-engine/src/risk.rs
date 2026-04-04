use crate::black_scholes::{BlackScholes, OptionType, Greeks};
use crate::OptionsError;
use std::collections::HashMap;

/// Greeks for a single position
#[derive(Debug, Clone)]
pub struct PositionGreeks {
    pub position_id: String,
    pub symbol: String,
    pub expiry: f64,
    pub strike: f64,
    pub opt_type: OptionType,
    pub quantity: f64,       // signed (positive = long, negative = short)
    pub spot: f64,
    pub greeks: Greeks,
    /// Dollar greeks
    pub dollar_delta: f64,   // delta * spot * multiplier
    pub dollar_gamma: f64,   // 0.5 * gamma * spot^2 * multiplier (P&L for 1% spot move)
    pub dollar_vega: f64,    // vega * multiplier (P&L for 1% vol move)
    pub dollar_theta: f64,   // theta * multiplier (P&L per day)
}

impl PositionGreeks {
    pub fn new(
        position_id: String,
        symbol: String,
        expiry: f64,
        strike: f64,
        opt_type: OptionType,
        quantity: f64,
        spot: f64,
        r: f64,
        q: f64,
        sigma: f64,
        t: f64,
        multiplier: f64,
    ) -> Self {
        let raw_greeks = BlackScholes::all_greeks(spot, strike, r, q, sigma, t, opt_type);
        let notional = quantity * multiplier;
        let dollar_delta = raw_greeks.delta * spot * notional;
        let dollar_gamma = 0.5 * raw_greeks.gamma * spot * spot * notional * 0.01;
        let dollar_vega = raw_greeks.vega * notional;
        let dollar_theta = raw_greeks.theta * notional;

        PositionGreeks {
            position_id,
            symbol,
            expiry,
            strike,
            opt_type,
            quantity,
            spot,
            greeks: raw_greeks,
            dollar_delta,
            dollar_gamma,
            dollar_vega,
            dollar_theta,
        }
    }

    pub fn unit_delta(&self) -> f64 { self.greeks.delta * self.quantity }
    pub fn unit_gamma(&self) -> f64 { self.greeks.gamma * self.quantity }
    pub fn unit_vega(&self) -> f64 { self.greeks.vega * self.quantity }
    pub fn unit_theta(&self) -> f64 { self.greeks.theta * self.quantity }
    pub fn unit_rho(&self) -> f64 { self.greeks.rho * self.quantity }
}

/// Portfolio-level Greeks
#[derive(Debug, Clone, Default)]
pub struct PortfolioGreeks {
    pub total_delta: f64,
    pub total_gamma: f64,
    pub total_vega: f64,
    pub total_theta: f64,
    pub total_rho: f64,
    pub total_vanna: f64,
    pub total_volga: f64,
    /// Dollar Greeks
    pub dollar_delta: f64,
    pub dollar_gamma: f64,
    pub dollar_vega: f64,
    pub dollar_theta: f64,
    /// Vega by expiry bucket
    pub vega_by_expiry: HashMap<String, f64>,
    /// Delta by symbol
    pub delta_by_symbol: HashMap<String, f64>,
}

impl PortfolioGreeks {
    pub fn new() -> Self { Default::default() }

    pub fn gamma_pnl_estimate(&self, spot_move_pct: f64) -> f64 {
        0.5 * self.total_gamma * spot_move_pct * spot_move_pct
    }

    pub fn vega_pnl_estimate(&self, vol_move_pct: f64) -> f64 {
        self.total_vega * vol_move_pct
    }
}

/// Aggregates Greeks across a portfolio
pub struct GreeksAggregator {
    positions: Vec<PositionGreeks>,
}

impl GreeksAggregator {
    pub fn new() -> Self {
        GreeksAggregator { positions: Vec::new() }
    }

    pub fn add_position(&mut self, pos: PositionGreeks) {
        self.positions.push(pos);
    }

    pub fn add_option(
        &mut self,
        position_id: String,
        symbol: String,
        expiry: f64,
        strike: f64,
        opt_type: OptionType,
        quantity: f64,
        spot: f64,
        r: f64,
        q: f64,
        sigma: f64,
        t: f64,
        multiplier: f64,
    ) {
        let pos = PositionGreeks::new(
            position_id, symbol, expiry, strike, opt_type,
            quantity, spot, r, q, sigma, t, multiplier,
        );
        self.positions.push(pos);
    }

    pub fn aggregate(&self) -> PortfolioGreeks {
        let mut pg = PortfolioGreeks::new();

        for pos in &self.positions {
            pg.total_delta += pos.unit_delta();
            pg.total_gamma += pos.unit_gamma();
            pg.total_vega += pos.unit_vega();
            pg.total_theta += pos.unit_theta();
            pg.total_rho += pos.unit_rho();
            pg.total_vanna += pos.greeks.vanna * pos.quantity;
            pg.total_volga += pos.greeks.volga * pos.quantity;
            pg.dollar_delta += pos.dollar_delta;
            pg.dollar_gamma += pos.dollar_gamma;
            pg.dollar_vega += pos.dollar_vega;
            pg.dollar_theta += pos.dollar_theta;

            // Vega by expiry bucket
            let expiry_key = format!("{:.2}", pos.expiry);
            *pg.vega_by_expiry.entry(expiry_key).or_insert(0.0) += pos.unit_vega();

            // Delta by symbol
            *pg.delta_by_symbol.entry(pos.symbol.clone()).or_insert(0.0) += pos.unit_delta();
        }

        pg
    }

    /// Compute dollar delta of a portfolio (full revalue approach)
    pub fn dollar_delta_bump(&self, bump_pct: f64) -> Result<f64, OptionsError> {
        if self.positions.is_empty() {
            return Ok(0.0);
        }
        // Re-evaluate at bumped spot
        let mut base_pv = 0.0;
        let mut bump_pv = 0.0;
        // We need original params: use greeks to reconstruct
        // This is a simplified version using BSM greeks
        for pos in &self.positions {
            base_pv += pos.greeks.price * pos.quantity;
            let bumped_spot = pos.spot * (1.0 + bump_pct);
            // We don't have original vol/r/q/t directly, use the approximation:
            // P(S+dS) ~= P(S) + delta*dS + 0.5*gamma*dS^2
            let ds = pos.spot * bump_pct;
            bump_pv += (pos.greeks.price + pos.greeks.delta * ds + 0.5 * pos.greeks.gamma * ds * ds) * pos.quantity;
        }
        Ok(bump_pv - base_pv)
    }

    /// Vega bucketing: vega sensitivity per expiry slice (in years)
    pub fn vega_buckets(&self, bucket_boundaries: &[f64]) -> Vec<(String, f64)> {
        let mut buckets: Vec<f64> = vec![0.0; bucket_boundaries.len() + 1];

        for pos in &self.positions {
            let idx = bucket_boundaries.partition_point(|&b| b <= pos.expiry);
            buckets[idx] += pos.unit_vega();
        }

        let mut result = Vec::new();
        let label = |i: usize| -> String {
            if i == 0 {
                format!("< {:.2}Y", bucket_boundaries[0])
            } else if i < bucket_boundaries.len() {
                format!("{:.2}Y - {:.2}Y", bucket_boundaries[i-1], bucket_boundaries[i])
            } else {
                format!("> {:.2}Y", bucket_boundaries[bucket_boundaries.len() - 1])
            }
        };
        for (i, &v) in buckets.iter().enumerate() {
            result.push((label(i), v));
        }
        result
    }

    /// Compute portfolio value
    pub fn portfolio_value(&self) -> f64 {
        self.positions.iter().map(|p| p.greeks.price * p.quantity).sum()
    }

    /// Delta-neutral hedge quantity for a given instrument
    pub fn delta_hedge_quantity(&self, hedge_delta_per_unit: f64) -> f64 {
        if hedge_delta_per_unit.abs() < 1e-12 { return 0.0; }
        -self.aggregate().total_delta / hedge_delta_per_unit
    }

    /// Gamma P&L for a given spot move
    pub fn gamma_pnl(&self, spot_move: f64) -> f64 {
        self.positions.iter().map(|p| {
            0.5 * p.greeks.gamma * spot_move * spot_move * p.quantity
        }).sum()
    }

    /// Theta decay over n_days
    pub fn theta_decay(&self, n_days: f64) -> f64 {
        self.positions.iter().map(|p| p.unit_theta() * n_days).sum()
    }

    pub fn positions(&self) -> &[PositionGreeks] {
        &self.positions
    }
}

impl Default for GreeksAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Stress test scenarios
#[derive(Debug, Clone)]
pub struct StressScenario {
    pub name: String,
    pub spot_move_pct: f64,
    pub vol_move_pct: f64,
    pub rate_move_bps: f64,
}

impl StressScenario {
    pub fn new(name: &str, spot_pct: f64, vol_pct: f64, rate_bps: f64) -> Self {
        StressScenario {
            name: name.to_string(),
            spot_move_pct: spot_pct,
            vol_move_pct: vol_pct,
            rate_move_bps: rate_bps,
        }
    }

    pub fn standard_scenarios() -> Vec<StressScenario> {
        vec![
            StressScenario::new("Down 10% / Vol +10%", -0.10, 0.10, 0.0),
            StressScenario::new("Down 20% / Vol +20%", -0.20, 0.20, 0.0),
            StressScenario::new("Up 10% / Vol -5%", 0.10, -0.05, 0.0),
            StressScenario::new("Up 20% / Vol -10%", 0.20, -0.10, 0.0),
            StressScenario::new("Flat / Vol +15%", 0.0, 0.15, 0.0),
            StressScenario::new("Flat / Vol -15%", 0.0, -0.15, 0.0),
            StressScenario::new("Rates +100bp", 0.0, 0.0, 100.0),
            StressScenario::new("Rates -100bp", 0.0, 0.0, -100.0),
        ]
    }

    /// P&L approximation via dollar Greeks
    pub fn pnl_estimate(&self, pg: &PortfolioGreeks, portfolio_vega: f64) -> f64 {
        let delta_pnl = pg.dollar_delta * self.spot_move_pct;
        let gamma_pnl = pg.dollar_gamma * self.spot_move_pct * self.spot_move_pct;
        let vega_pnl = portfolio_vega * self.vol_move_pct;
        let rho_pnl = pg.total_rho * self.rate_move_bps * 0.0001;
        delta_pnl + gamma_pnl + vega_pnl + rho_pnl
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_atm_call(qty: f64) -> PositionGreeks {
        PositionGreeks::new(
            "POS001".to_string(), "AAPL".to_string(),
            1.0, 100.0, OptionType::Call,
            qty, 100.0, 0.05, 0.02, 0.2, 1.0, 100.0,
        )
    }

    fn make_atm_put(qty: f64) -> PositionGreeks {
        PositionGreeks::new(
            "POS002".to_string(), "AAPL".to_string(),
            1.0, 100.0, OptionType::Put,
            qty, 100.0, 0.05, 0.02, 0.2, 1.0, 100.0,
        )
    }

    #[test]
    fn test_position_greeks_values() {
        let pos = make_atm_call(1.0);
        assert!(pos.greeks.delta > 0.0 && pos.greeks.delta < 1.0);
        assert!(pos.greeks.gamma > 0.0);
        assert!(pos.greeks.vega > 0.0);
        assert!(pos.greeks.theta < 0.0);
    }

    #[test]
    fn test_straddle_delta_near_zero() {
        let mut agg = GreeksAggregator::new();
        agg.add_position(make_atm_call(1.0));
        agg.add_position(make_atm_put(1.0));
        let pg = agg.aggregate();
        // ATM straddle: call delta + put delta = e^{-q*T}*(2*N(d1)-1)
        // For ATM with dividends this is non-zero; check it's between -0.3 and 0.3
        assert!(pg.total_delta.abs() < 0.30, "Straddle delta = {:.4}", pg.total_delta);
        // Gamma should be positive (doubled)
        assert!(pg.total_gamma > 0.0);
    }

    #[test]
    fn test_vega_bucketing() {
        let mut agg = GreeksAggregator::new();
        agg.add_option("P1".to_string(), "AAPL".to_string(), 0.25, 100.0, OptionType::Call,
            10.0, 100.0, 0.05, 0.02, 0.2, 0.25, 100.0);
        agg.add_option("P2".to_string(), "AAPL".to_string(), 1.0, 100.0, OptionType::Call,
            10.0, 100.0, 0.05, 0.02, 0.2, 1.0, 100.0);
        agg.add_option("P3".to_string(), "AAPL".to_string(), 2.0, 100.0, OptionType::Call,
            10.0, 100.0, 0.05, 0.02, 0.2, 2.0, 100.0);

        let buckets = agg.vega_buckets(&[0.5, 1.0, 1.5]);
        assert_eq!(buckets.len(), 4);
        // First bucket should have the 0.25Y position's vega
        assert!(buckets[0].1 > 0.0);
    }

    #[test]
    fn test_portfolio_value() {
        let mut agg = GreeksAggregator::new();
        agg.add_position(make_atm_call(10.0));
        let val = agg.portfolio_value();
        let bsm_price = BlackScholes::price(100.0, 100.0, 0.05, 0.02, 0.2, 1.0, OptionType::Call);
        assert!((val - 10.0 * bsm_price).abs() < 1e-10);
    }

    #[test]
    fn test_gamma_pnl() {
        let mut agg = GreeksAggregator::new();
        agg.add_position(make_atm_call(1.0));
        let spot_move = 5.0;
        let pnl = agg.gamma_pnl(spot_move);
        assert!(pnl > 0.0); // Long gamma => positive P&L from large moves
    }

    #[test]
    fn test_stress_scenarios() {
        let mut agg = GreeksAggregator::new();
        agg.add_position(make_atm_call(100.0));
        let pg = agg.aggregate();
        let vega = pg.total_vega;
        for scenario in StressScenario::standard_scenarios() {
            let pnl = scenario.pnl_estimate(&pg, vega);
            // Just check it runs without panic
            let _ = pnl;
        }
    }

    #[test]
    fn test_delta_hedge_quantity() {
        let mut agg = GreeksAggregator::new();
        agg.add_position(make_atm_call(100.0));
        // Hedge with stock (delta = 1.0)
        let hedge_qty = agg.delta_hedge_quantity(1.0);
        let pg = agg.aggregate();
        // After hedge, net delta should be near zero
        let hedged_delta = pg.total_delta + hedge_qty * 1.0;
        assert!(hedged_delta.abs() < 1e-10);
    }

    #[test]
    fn test_vega_by_expiry() {
        let mut agg = GreeksAggregator::new();
        agg.add_option("P1".to_string(), "AAPL".to_string(), 0.5, 100.0, OptionType::Call,
            1.0, 100.0, 0.05, 0.02, 0.2, 0.5, 1.0);
        agg.add_option("P2".to_string(), "AAPL".to_string(), 1.0, 100.0, OptionType::Put,
            1.0, 100.0, 0.05, 0.02, 0.2, 1.0, 1.0);
        let pg = agg.aggregate();
        assert!(pg.vega_by_expiry.len() >= 2);
    }
}
