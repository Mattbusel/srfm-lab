use crate::venue::{Venue, MarketData};
use crate::SorError;

/// Breakdown of total trading cost
#[derive(Debug, Clone, Default)]
pub struct CostBreakdown {
    /// Bid-ask spread cost (half-spread as fraction of price)
    pub spread_cost: f64,
    /// Market impact (price movement caused by our order)
    pub market_impact: f64,
    /// Explicit fee/commission
    pub fee_cost: f64,
    /// Timing risk (opportunity cost of not trading immediately)
    pub timing_risk: f64,
    /// Total implementation shortfall
    pub total: f64,
}

impl CostBreakdown {
    pub fn recompute_total(&mut self) {
        self.total = self.spread_cost + self.market_impact + self.fee_cost + self.timing_risk;
    }
}

/// Result of cost estimation for a trade
#[derive(Debug, Clone)]
pub struct TradeCost {
    pub venue_id: String,
    pub quantity: f64,
    pub price: f64,
    pub notional: f64,
    pub breakdown: CostBreakdown,
    /// Cost in basis points of notional
    pub total_bps: f64,
}

impl TradeCost {
    pub fn dollar_cost(&self) -> f64 {
        self.notional * self.breakdown.total
    }
}

/// Implementation shortfall cost model
#[derive(Clone)]
pub struct CostModel {
    /// Volatility of the underlying (daily)
    pub daily_vol: f64,
    /// Typical bid-ask spread as fraction of price
    pub typical_spread: f64,
    /// Market impact exponent (0.5 = square root law)
    pub impact_exponent: f64,
    /// Market impact coefficient (calibrated per stock)
    pub impact_coefficient: f64,
    /// Risk aversion parameter for timing risk
    pub risk_aversion: f64,
}

impl CostModel {
    pub fn new(daily_vol: f64, typical_spread: f64) -> Self {
        CostModel {
            daily_vol,
            typical_spread,
            impact_exponent: 0.5,
            impact_coefficient: 0.5,
            risk_aversion: 1.0,
        }
    }

    pub fn with_impact(mut self, coeff: f64, exponent: f64) -> Self {
        self.impact_coefficient = coeff;
        self.impact_exponent = exponent;
        self
    }

    pub fn with_risk_aversion(mut self, ra: f64) -> Self {
        self.risk_aversion = ra;
        self
    }

    /// Square-root market impact model: MI = sigma * coefficient * sqrt(Q/ADV)
    /// Returns impact as fraction of price
    pub fn market_impact(&self, qty: f64, adv: f64, price: f64) -> f64 {
        if adv <= 0.0 { return 0.0; }
        let pov = qty / adv;  // participation rate
        self.impact_coefficient * self.daily_vol * pov.powf(self.impact_exponent)
    }

    /// Spread cost: half-spread for a market order
    pub fn spread_cost(&self, md: &MarketData, is_buy: bool) -> f64 {
        let mid = md.mid();
        if mid <= 0.0 { return 0.0; }
        let half_spread = md.spread() / (2.0 * mid);
        half_spread
    }

    /// Timing risk: cost of waiting, modeled as vol^2 * time * risk_aversion
    /// time_to_complete is in days
    pub fn timing_risk(&self, qty: f64, adv: f64, price: f64, time_to_complete_days: f64) -> f64 {
        let pov = if adv > 0.0 { qty / adv } else { 0.0 };
        // Simplified: risk ~ sigma * sqrt(time) * risk_aversion * pov
        self.risk_aversion * self.daily_vol * time_to_complete_days.sqrt() * pov.min(1.0)
    }

    /// Full cost estimate for a single venue trade
    pub fn estimate_cost(
        &self,
        venue: &Venue,
        qty: f64,
        price: f64,
        md: &MarketData,
        is_buy: bool,
        is_taker: bool,
        time_horizon_days: f64,
    ) -> TradeCost {
        let notional = qty * price;
        let fee_cost = venue.effective_fee(is_taker);
        let spread = self.spread_cost(md, is_buy);
        let impact = self.market_impact(qty, md.adv, price);
        let timing = self.timing_risk(qty, md.adv, price, time_horizon_days);

        let mut breakdown = CostBreakdown {
            spread_cost: spread,
            market_impact: impact,
            fee_cost,
            timing_risk: timing,
            total: 0.0,
        };
        breakdown.recompute_total();

        TradeCost {
            venue_id: venue.id.clone(),
            quantity: qty,
            price,
            notional,
            total_bps: breakdown.total * 10_000.0,
            breakdown,
        }
    }

    /// Estimate total cost across a split allocation
    pub fn total_cost_for_allocation(
        &self,
        allocations: &[(String, f64)], // (venue_id, qty)
        price: f64,
        md_by_venue: &std::collections::HashMap<String, MarketData>,
        venues: &[&crate::venue::Venue],
        is_buy: bool,
        time_horizon_days: f64,
    ) -> f64 {
        let total_qty: f64 = allocations.iter().map(|(_, q)| q).sum();
        if total_qty <= 0.0 { return 0.0; }

        allocations.iter().map(|(vid, qty)| {
            let venue = match venues.iter().find(|v| &v.id == vid) {
                Some(v) => v,
                None => return 0.0,
            };
            let md = match md_by_venue.get(vid) {
                Some(m) => m,
                None => return 0.0,
            };
            let cost = self.estimate_cost(venue, *qty, price, md, is_buy, true, time_horizon_days);
            cost.dollar_cost()
        }).sum()
    }

    /// Compute implementation shortfall vs. arrival price
    pub fn implementation_shortfall(
        &self,
        arrival_price: f64,
        fills: &[(f64, f64)], // (price, qty)
        side_sign: f64, // +1 for buy, -1 for sell
    ) -> f64 {
        let total_qty: f64 = fills.iter().map(|(_, q)| q).sum();
        if total_qty <= 0.0 { return 0.0; }
        let vwap: f64 = fills.iter().map(|(p, q)| p * q).sum::<f64>() / total_qty;
        side_sign * (vwap - arrival_price) / arrival_price
    }

    /// Almgren-Chriss optimal execution: optimal trading rate
    /// Minimizes E[cost] + lambda * Var[cost]
    pub fn almgren_chriss_rate(
        &self,
        total_qty: f64,
        adv: f64,
        time_horizon_secs: f64,
        n_intervals: usize,
    ) -> Vec<f64> {
        // Simplified: uniform participation subject to POV constraint
        let max_pov = 0.25;
        let interval_secs = time_horizon_secs / n_intervals as f64;
        let max_per_interval = adv * max_pov * interval_secs / 86400.0;
        let base_qty = total_qty / n_intervals as f64;
        let per_interval = base_qty.min(max_per_interval);

        let mut schedule = vec![per_interval; n_intervals];

        // Adjust last interval for rounding
        let allocated: f64 = schedule[..n_intervals.saturating_sub(1)].iter().sum();
        if n_intervals > 0 {
            schedule[n_intervals - 1] = (total_qty - allocated).max(0.0);
        }

        schedule
    }
}

/// VWAP benchmark computation
pub fn vwap(fills: &[(f64, f64)]) -> f64 {
    let total_qty: f64 = fills.iter().map(|(_, q)| q).sum();
    if total_qty <= 0.0 { return 0.0; }
    fills.iter().map(|(p, q)| p * q).sum::<f64>() / total_qty
}

/// TWAP: average of prices over time
pub fn twap(prices: &[f64]) -> f64 {
    if prices.is_empty() { return 0.0; }
    prices.iter().sum::<f64>() / prices.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::venue::{Venue, ExchangeType, MarketData};

    fn make_md(spread_frac: f64, price: f64, adv: f64) -> MarketData {
        let half = price * spread_frac / 2.0;
        MarketData {
            bid: price - half, ask: price + half,
            bid_size: adv * 0.01, ask_size: adv * 0.01,
            last_trade: price, daily_volume: adv, adv,
            bid_levels: vec![],
            ask_levels: vec![],
        }
    }

    fn make_venue() -> Venue {
        Venue::new("NYSE", "NYSE", ExchangeType::LitExchange)
            .with_fees(-0.002, 0.003)
    }

    #[test]
    fn test_spread_cost() {
        let model = CostModel::new(0.02, 0.001);
        let md = make_md(0.001, 100.0, 1_000_000.0);
        let spread = model.spread_cost(&md, true);
        assert!((spread - 0.0005).abs() < 1e-8, "spread={:.6}", spread);
    }

    #[test]
    fn test_market_impact_sqrt_law() {
        let model = CostModel::new(0.02, 0.001);
        let price = 100.0;
        let adv = 1_000_000.0;
        // 10% participation
        let mi_10pct = model.market_impact(100_000.0, adv, price);
        // 40% participation: impact ~ sqrt(4) * mi_10pct
        let mi_40pct = model.market_impact(400_000.0, adv, price);
        let ratio = mi_40pct / mi_10pct;
        assert!((ratio - 2.0).abs() < 0.01, "sqrt ratio = {:.4}", ratio);
    }

    #[test]
    fn test_full_cost_estimate() {
        let model = CostModel::new(0.02, 0.001);
        let venue = make_venue();
        let md = make_md(0.001, 100.0, 1_000_000.0);
        let cost = model.estimate_cost(&venue, 10_000.0, 100.0, &md, true, true, 0.5);
        assert!(cost.breakdown.total > 0.0);
        assert!(cost.total_bps > 0.0);
        println!("Cost breakdown: {:?}", cost.breakdown);
    }

    #[test]
    fn test_implementation_shortfall() {
        let model = CostModel::new(0.02, 0.001);
        let fills = vec![(100.10, 500.0), (100.20, 500.0)];
        let is = model.implementation_shortfall(100.0, &fills, 1.0);
        // Buy at avg 100.15 vs arrival 100.00 = 15bps shortfall
        assert!(is > 0.0);
        let expected = (100.15 - 100.0) / 100.0;
        assert!((is - expected).abs() < 1e-10);
    }

    #[test]
    fn test_vwap() {
        let fills = vec![(100.0, 1000.0), (101.0, 500.0), (99.0, 500.0)];
        let v = vwap(&fills);
        // (100*1000 + 101*500 + 99*500) / 2000 = (100000 + 50500 + 49500) / 2000 = 200000/2000 = 100.0
        assert!((v - 100.0).abs() < 1e-10, "vwap={}", v);
    }

    #[test]
    fn test_almgren_chriss_schedule() {
        let model = CostModel::new(0.02, 0.001);
        let schedule = model.almgren_chriss_rate(100_000.0, 2_000_000.0, 3600.0, 12);
        assert_eq!(schedule.len(), 12);
        let total: f64 = schedule.iter().sum();
        assert!((total - 100_000.0).abs() < 1e-6, "total={}", total);
    }

    #[test]
    fn test_twap() {
        let prices = vec![100.0, 101.0, 102.0, 99.0];
        let t = twap(&prices);
        assert!((t - 100.5).abs() < 1e-10);
    }

    #[test]
    fn test_timing_risk_grows_with_time() {
        let model = CostModel::new(0.02, 0.001);
        let r1 = model.timing_risk(10_000.0, 1_000_000.0, 100.0, 0.5);
        let r5 = model.timing_risk(10_000.0, 1_000_000.0, 100.0, 5.0);
        assert!(r5 > r1, "Timing risk should grow with time: r1={:.6} r5={:.6}", r1, r5);
    }
}
