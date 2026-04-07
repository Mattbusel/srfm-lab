/// Liquidity risk assessment: liquidation horizons, market impact, L-VaR,
/// and concentration metrics.

// ── LiquidityMetrics ─────────────────────────────────────────────────────────

/// Full liquidity profile for a single position.
#[derive(Debug, Clone, Copy)]
pub struct LiquidityMetrics {
    /// Estimated number of days to liquidate the position.
    pub liquidation_horizon_days: f64,
    /// Total cost to liquidate as basis points of position notional.
    pub liquidation_cost_bps: f64,
    /// Estimated market impact in basis points.
    pub market_impact_bps: f64,
    /// Half-spread cost in basis points (bid-ask).
    pub bid_ask_cost_bps: f64,
    /// Absolute dollar cost to fully liquidate.
    pub total_liquidation_cost: f64,
}

// ── Position (reused from stress_scenarios, kept local to avoid coupling) ────

/// Lightweight position descriptor for liquidity analysis.
#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    /// Absolute position size in units (e.g. number of coins or shares).
    pub qty: f64,
    /// Current market price per unit.
    pub price: f64,
    /// Average Daily Volume in the same units as qty.
    pub adv: f64,
    /// Daily return volatility (e.g. 0.03 = 3%).
    pub daily_vol: f64,
    /// Bid-ask spread in basis points (e.g. 20.0 = 20 bps = 0.20%).
    pub spread_bps: f64,
}

impl Position {
    /// Notional market value of the position.
    pub fn notional(&self) -> f64 {
        self.qty * self.price
    }
}

// ── LiquidityRiskEngine ───────────────────────────────────────────────────────

/// Engine for computing liquidity-adjusted risk metrics.
pub struct LiquidityRiskEngine {
    /// Maximum fraction of ADV that can be traded per day without significant impact.
    pub max_adv_fraction: f64,
    /// Market impact coefficient (Kyle lambda proxy).
    /// impact_bps = impact_coeff * (qty / adv)^0.6 * 10000
    pub impact_coeff: f64,
}

impl Default for LiquidityRiskEngine {
    fn default() -> Self {
        LiquidityRiskEngine {
            max_adv_fraction: 0.10,
            impact_coeff: 0.50,
        }
    }
}

impl LiquidityRiskEngine {
    pub fn new(max_adv_fraction: f64, impact_coeff: f64) -> Self {
        LiquidityRiskEngine { max_adv_fraction, impact_coeff }
    }

    /// Compute liquidation horizon, market impact, and total cost for a single position.
    pub fn assess_position(&self, position: &Position) -> LiquidityMetrics {
        assess_position_with_params(
            &position.symbol,
            position.qty,
            position.adv,
            position.spread_bps,
            position.price,
            position.daily_vol,
            self.max_adv_fraction,
            self.impact_coeff,
        )
    }

    /// Assess all positions in a portfolio.
    pub fn assess_portfolio(&self, positions: &[Position]) -> Vec<(String, LiquidityMetrics)> {
        positions
            .iter()
            .map(|p| (p.symbol.clone(), self.assess_position(p)))
            .collect()
    }

    /// Liquidity-adjusted VaR: VaR + liquidation cost at each position.
    /// Base VaR uses the square root of time rule scaled by daily vol.
    pub fn portfolio_liquidity_adjusted_var(
        &self,
        positions: &[Position],
        confidence: f64,
    ) -> f64 {
        portfolio_liquidity_adjusted_var(positions, confidence, self.max_adv_fraction, self.impact_coeff)
    }
}

// ── Standalone assess_position function ──────────────────────────────────────

/// Assess liquidity risk for a single position.
/// - qty: position size in units
/// - adv: average daily volume in units
/// - spread_bps: bid-ask spread in basis points
/// - price: current price per unit
/// - daily_vol: daily return volatility (e.g. 0.02 = 2%)
pub fn assess_position(
    symbol: &str,
    qty: f64,
    adv: f64,
    spread_bps: f64,
) -> LiquidityMetrics {
    // Use reasonable defaults for price and daily_vol when not provided.
    assess_position_with_params(symbol, qty, adv, spread_bps, 1.0, 0.02, 0.10, 0.50)
}

fn assess_position_with_params(
    _symbol: &str,
    qty: f64,
    adv: f64,
    spread_bps: f64,
    price: f64,
    _daily_vol: f64,
    max_adv_fraction: f64,
    impact_coeff: f64,
) -> LiquidityMetrics {
    let notional = qty * price;
    if qty <= 0.0 || adv <= 0.0 {
        return LiquidityMetrics {
            liquidation_horizon_days: 0.0,
            liquidation_cost_bps: 0.0,
            market_impact_bps: 0.0,
            bid_ask_cost_bps: spread_bps / 2.0, // one-way half-spread
            total_liquidation_cost: 0.0,
        };
    }

    // Liquidation horizon: number of days to liquidate at max_adv_fraction per day.
    let daily_liquidation_qty = max_adv_fraction * adv;
    let horizon_days = (qty / daily_liquidation_qty).ceil();

    // Bid-ask cost: half-spread on each day's notional sold.
    // Total cost = half_spread * total_notional.
    let half_spread_bps = spread_bps / 2.0;
    let bid_ask_cost = notional * half_spread_bps / 10_000.0;

    // Market impact: price deterioration during liquidation.
    // Model: each day we sell daily_liquidation_qty / adv fraction of ADV.
    // Impact per day (bps) = impact_coeff * (participation_rate)^0.6 * 10000.
    let participation_rate = (daily_liquidation_qty / adv).min(1.0);
    let daily_impact_bps = impact_coeff * participation_rate.powf(0.6) * 10_000.0;

    // Price deterioration: sqrt(T) scaling for random walk price drift during liquidation.
    // Total impact = daily_impact * sqrt(T) * T = accumulated drift over horizon days.
    let price_drift_cost = notional * daily_impact_bps / 10_000.0 * horizon_days.sqrt();

    // Total cost.
    let total_cost = bid_ask_cost + price_drift_cost;
    let total_cost_bps = if notional > 0.0 { total_cost / notional * 10_000.0 } else { 0.0 };
    let market_impact_bps = if notional > 0.0 { price_drift_cost / notional * 10_000.0 } else { 0.0 };

    LiquidityMetrics {
        liquidation_horizon_days: horizon_days,
        liquidation_cost_bps: total_cost_bps,
        market_impact_bps,
        bid_ask_cost_bps: half_spread_bps,
        total_liquidation_cost: total_cost,
    }
}

// ── Liquidity-adjusted VaR ────────────────────────────────────────────────────

/// Compute portfolio L-VaR: parametric VaR scaled by liquidation horizon
/// plus liquidation cost for each position.
/// confidence: e.g. 0.99
pub fn portfolio_liquidity_adjusted_var(
    positions: &[Position],
    confidence: f64,
    max_adv_fraction: f64,
    impact_coeff: f64,
) -> f64 {
    if positions.is_empty() {
        return 0.0;
    }

    // z-score for confidence level.
    let z = inv_normal_quantile(confidence);

    let mut total_l_var = 0.0_f64;
    for pos in positions {
        let notional = pos.notional();
        let metrics = assess_position_with_params(
            &pos.symbol,
            pos.qty,
            pos.adv,
            pos.spread_bps,
            pos.price,
            pos.daily_vol,
            max_adv_fraction,
            impact_coeff,
        );
        let h = metrics.liquidation_horizon_days.max(1.0);
        // VaR over liquidation horizon: VaR_h = z * daily_vol * sqrt(h) * notional.
        let var_h = z * pos.daily_vol * h.sqrt() * notional;
        // L-VaR = VaR_h + liquidation cost.
        let l_var = var_h + metrics.total_liquidation_cost;
        total_l_var += l_var;
    }

    total_l_var
}

// ── Herfindahl-Hirschman Index ────────────────────────────────────────────────

/// Concentration index: sum of squared portfolio weights.
/// HHI = 1 means 100% in one position (maximum concentration).
/// HHI = 1/n means equal weight (minimum concentration for n positions).
pub fn concentration_hhi(positions: &[Position]) -> f64 {
    if positions.is_empty() {
        return 0.0;
    }
    let total_notional: f64 = positions.iter().map(|p| p.notional().abs()).sum();
    if total_notional <= 0.0 {
        return 0.0;
    }
    positions
        .iter()
        .map(|p| {
            let w = p.notional().abs() / total_notional;
            w * w
        })
        .sum()
}

/// Normalized HHI: (HHI - 1/n) / (1 - 1/n), ranges [0, 1].
/// 0 = perfectly diversified, 1 = fully concentrated.
pub fn concentration_hhi_normalized(positions: &[Position]) -> f64 {
    let n = positions.len() as f64;
    if n <= 1.0 {
        return 1.0;
    }
    let hhi = concentration_hhi(positions);
    let min_hhi = 1.0 / n;
    if (1.0 - min_hhi).abs() < 1e-12 {
        return 0.0;
    }
    ((hhi - min_hhi) / (1.0 - min_hhi)).clamp(0.0, 1.0)
}

/// Weighted average liquidation horizon for a portfolio.
pub fn weighted_liquidation_horizon(
    positions: &[Position],
    max_adv_fraction: f64,
) -> f64 {
    if positions.is_empty() {
        return 0.0;
    }
    let total_notional: f64 = positions.iter().map(|p| p.notional().abs()).sum();
    if total_notional <= 0.0 {
        return 0.0;
    }
    positions
        .iter()
        .map(|p| {
            let daily_qty = max_adv_fraction * p.adv;
            let h = if daily_qty > 0.0 { (p.qty / daily_qty).ceil() } else { f64::MAX };
            let w = p.notional().abs() / total_notional;
            w * h
        })
        .sum()
}

// ── Normal quantile helper ────────────────────────────────────────────────────

fn inv_normal_quantile(p: f64) -> f64 {
    let p = p.clamp(1e-10, 1.0 - 1e-10);
    let sgn = if p < 0.5 { -1.0 } else { 1.0 };
    let pp = if p < 0.5 { p } else { 1.0 - p };
    let t = (-2.0 * pp.ln()).sqrt();
    let num = 2.515517 + t * (0.802853 + t * 0.010328);
    let den = 1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308));
    sgn * (t - num / den)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_position(symbol: &str, qty: f64, price: f64, adv: f64, spread_bps: f64) -> Position {
        Position {
            symbol: symbol.to_string(),
            qty,
            price,
            adv,
            daily_vol: 0.03,
            spread_bps,
        }
    }

    #[test]
    fn test_liquidation_horizon_formula() {
        // 1000 units, ADV = 10000, max_adv_fraction = 0.10 => daily_qty = 1000.
        // horizon = ceil(1000 / 1000) = 1 day.
        let pos = make_position("TEST", 1000.0, 100.0, 10_000.0, 20.0);
        let engine = LiquidityRiskEngine::default();
        let metrics = engine.assess_position(&pos);
        assert_eq!(metrics.liquidation_horizon_days, 1.0);
    }

    #[test]
    fn test_liquidation_horizon_large_position() {
        // 50000 units, ADV = 10000 => daily_qty = 1000 => horizon = 50 days.
        let pos = make_position("TEST", 50_000.0, 100.0, 10_000.0, 10.0);
        let engine = LiquidityRiskEngine::default();
        let metrics = engine.assess_position(&pos);
        assert_eq!(metrics.liquidation_horizon_days, 50.0);
    }

    #[test]
    fn test_bid_ask_cost_bps_is_half_spread() {
        let pos = make_position("TEST", 1000.0, 100.0, 10_000.0, 40.0);
        let engine = LiquidityRiskEngine::default();
        let metrics = engine.assess_position(&pos);
        assert!((metrics.bid_ask_cost_bps - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_total_cost_positive() {
        let pos = make_position("TEST", 5000.0, 100.0, 10_000.0, 30.0);
        let engine = LiquidityRiskEngine::default();
        let metrics = engine.assess_position(&pos);
        assert!(metrics.total_liquidation_cost > 0.0);
    }

    #[test]
    fn test_hhi_single_position() {
        let positions = vec![make_position("BTC", 1.0, 50_000.0, 1000.0, 20.0)];
        let hhi = concentration_hhi(&positions);
        assert!((hhi - 1.0).abs() < 1e-10, "Single position HHI must be 1.0");
    }

    #[test]
    fn test_hhi_equal_weight() {
        let positions = vec![
            make_position("A", 1.0, 100.0, 1000.0, 10.0),
            make_position("B", 1.0, 100.0, 1000.0, 10.0),
            make_position("C", 1.0, 100.0, 1000.0, 10.0),
            make_position("D", 1.0, 100.0, 1000.0, 10.0),
        ];
        let hhi = concentration_hhi(&positions);
        assert!((hhi - 0.25).abs() < 1e-10, "Equal 4-position HHI must be 0.25, got {}", hhi);
    }

    #[test]
    fn test_hhi_normalized_equal_weight_near_zero() {
        let positions = vec![
            make_position("A", 1.0, 100.0, 1000.0, 10.0),
            make_position("B", 1.0, 100.0, 1000.0, 10.0),
            make_position("C", 1.0, 100.0, 1000.0, 10.0),
        ];
        let hhi_n = concentration_hhi_normalized(&positions);
        assert!(hhi_n.abs() < 1e-10, "Normalized HHI for equal weights must be 0");
    }

    #[test]
    fn test_portfolio_l_var_positive() {
        let positions = vec![
            make_position("BTC", 10.0, 50_000.0, 500.0, 30.0),
            make_position("ETH", 50.0, 3_000.0, 10_000.0, 20.0),
        ];
        let l_var = portfolio_liquidity_adjusted_var(&positions, 0.99, 0.10, 0.50);
        assert!(l_var > 0.0, "L-VaR must be positive");
    }

    #[test]
    fn test_l_var_exceeds_unadjusted_var() {
        let positions = vec![make_position("BTC", 100.0, 50_000.0, 200.0, 50.0)];
        let l_var = portfolio_liquidity_adjusted_var(&positions, 0.99, 0.10, 0.50);
        let z = inv_normal_quantile(0.99);
        let unadjusted = z * 0.03 * positions[0].notional();
        assert!(l_var > unadjusted, "L-VaR must exceed unadjusted VaR");
    }

    #[test]
    fn test_weighted_liquidation_horizon() {
        let positions = vec![
            // ADV = 100, qty = 10, daily_qty = 10 => h=1
            make_position("LIQUID", 10.0, 100.0, 100.0, 5.0),
            // ADV = 10, qty = 100, daily_qty = 1 => h=100
            make_position("ILLIQUID", 100.0, 100.0, 10.0, 100.0),
        ];
        let wh = weighted_liquidation_horizon(&positions, 0.10);
        // Equal notional => weighted avg = (1 + 100) / 2 = 50.5
        assert!((wh - 50.5).abs() < 1e-9, "Weighted horizon = {}", wh);
    }

    #[test]
    fn test_assess_portfolio_length() {
        let positions = vec![
            make_position("A", 100.0, 50.0, 1000.0, 10.0),
            make_position("B", 200.0, 25.0, 2000.0, 15.0),
        ];
        let engine = LiquidityRiskEngine::default();
        let results = engine.assess_portfolio(&positions);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_zero_qty_returns_zero_horizon() {
        let metrics = assess_position("TEST", 0.0, 1000.0, 20.0);
        assert_eq!(metrics.liquidation_horizon_days, 0.0);
    }
}
