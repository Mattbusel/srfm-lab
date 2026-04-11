/// Agent self-impact measurement and footprint analysis.
///
/// Tracks how an RL agent's own trades move the simulated LOB:
/// - Footprint analysis (fills at each price level)
/// - Self-impact decay (how quickly the impact dissipates)
/// - Queue position tracking and fill probability estimation
/// - Order-flow imbalance contribution

use std::collections::{HashMap, VecDeque};
use crate::lob_engine::{LobFill, Side, Qty, Nanos, from_tick, to_tick, TickPrice};

// ── Fill Record ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AgentFill {
    pub timestamp: Nanos,
    pub price: f64,
    pub qty: Qty,
    pub side: Side,
    /// Mid-price at time of fill.
    pub mid_at_fill: f64,
    /// Signed slippage vs mid: positive = worse than mid.
    pub slippage: f64,
}

impl AgentFill {
    pub fn from_lob_fill(fill: &LobFill, mid: f64) -> Self {
        let slippage = match fill.side {
            Side::Bid => fill.price - mid,   // buy above mid = negative impact
            Side::Ask => mid - fill.price,   // sell below mid = negative impact
        };
        AgentFill {
            timestamp: fill.timestamp,
            price: fill.price,
            qty: fill.qty,
            side: fill.side,
            mid_at_fill: mid,
            slippage,
        }
    }
}

// ── Footprint ─────────────────────────────────────────────────────────────────

/// Price-level breakdown of agent fills (footprint chart).
#[derive(Debug, Default, Clone)]
pub struct Footprint {
    /// Buy volume at each price level.
    pub bid_volume: HashMap<TickPrice, Qty>,
    /// Sell volume at each price level.
    pub ask_volume: HashMap<TickPrice, Qty>,
    /// Total buy volume.
    pub total_buy: Qty,
    /// Total sell volume.
    pub total_sell: Qty,
    /// VWAP of buys.
    pub vwap_buy: f64,
    /// VWAP of sells.
    pub vwap_sell: f64,
}

impl Footprint {
    pub fn add_fill(&mut self, fill: &AgentFill) {
        let tick = to_tick(fill.price);
        match fill.side {
            Side::Bid => {
                *self.bid_volume.entry(tick).or_insert(0.0) += fill.qty;
                self.total_buy += fill.qty;
            }
            Side::Ask => {
                *self.ask_volume.entry(tick).or_insert(0.0) += fill.qty;
                self.total_sell += fill.qty;
            }
        }
    }

    pub fn compute_vwaps(&mut self) {
        if self.total_buy > 1e-9 {
            self.vwap_buy = self.bid_volume.iter()
                .map(|(&p, &q)| from_tick(p) * q)
                .sum::<f64>() / self.total_buy;
        }
        if self.total_sell > 1e-9 {
            self.vwap_sell = self.ask_volume.iter()
                .map(|(&p, &q)| from_tick(p) * q)
                .sum::<f64>() / self.total_sell;
        }
    }

    /// Net signed volume (buy - sell).
    pub fn net_volume(&self) -> Qty {
        self.total_buy - self.total_sell
    }

    /// Dollar value of buys.
    pub fn buy_notional(&self) -> f64 {
        self.bid_volume.iter()
            .map(|(&p, &q)| from_tick(p) * q)
            .sum()
    }

    /// Dollar value of sells.
    pub fn sell_notional(&self) -> f64 {
        self.ask_volume.iter()
            .map(|(&p, &q)| from_tick(p) * q)
            .sum()
    }

    pub fn clear(&mut self) {
        self.bid_volume.clear();
        self.ask_volume.clear();
        self.total_buy = 0.0;
        self.total_sell = 0.0;
        self.vwap_buy = 0.0;
        self.vwap_sell = 0.0;
    }
}

// ── Self-Impact Decay ─────────────────────────────────────────────────────────

/// Model of self-impact: mid-price displacement due to the agent's own trades.
///
/// Uses an exponential decay model:
///   impact(t) = Σᵢ λᵢ · sign(side) · exp(−γ · (t − tᵢ))
///
/// where λ = impact per unit volume, γ = decay rate.
#[derive(Debug, Clone)]
pub struct SelfImpactModel {
    /// Impact per unit volume (in price units).
    pub lambda: f64,
    /// Decay rate γ (per nanosecond).
    pub gamma: f64,
    /// History of (timestamp, signed_impact).
    impact_history: VecDeque<(Nanos, f64)>,
    /// Accumulated impact (can be approximated via recursion).
    current_impact: f64,
    /// Timestamp of last update.
    last_update: Nanos,
}

impl SelfImpactModel {
    pub fn new(lambda: f64, gamma_per_second: f64) -> Self {
        // Convert gamma from per-second to per-nanosecond.
        let gamma = gamma_per_second / 1e9;
        SelfImpactModel {
            lambda,
            gamma,
            impact_history: VecDeque::new(),
            current_impact: 0.0,
            last_update: 0,
        }
    }

    /// Record a new fill and update impact.
    pub fn record_fill(&mut self, fill: &AgentFill) {
        let ts = fill.timestamp;
        let signed_qty = match fill.side {
            Side::Bid => fill.qty,
            Side::Ask => -fill.qty,
        };
        let instantaneous_impact = self.lambda * signed_qty;

        // Decay existing impact to current time.
        if self.last_update > 0 {
            let dt = (ts - self.last_update) as f64;
            self.current_impact *= (-self.gamma * dt).exp();
        }

        self.current_impact += instantaneous_impact;
        self.last_update = ts;
        self.impact_history.push_back((ts, instantaneous_impact));

        // Trim old history (keep last 10 seconds worth of events).
        let cutoff = ts.saturating_sub(10_000_000_000);
        while self.impact_history.front().map_or(false, |&(t, _)| t < cutoff) {
            self.impact_history.pop_front();
        }
    }

    /// Get current self-impact estimate at time `now`.
    pub fn current_impact_at(&self, now: Nanos) -> f64 {
        if self.last_update == 0 { return 0.0; }
        let dt = (now - self.last_update) as f64;
        self.current_impact * (-self.gamma * dt).exp()
    }

    /// Compute full impact from history at time `now`.
    pub fn full_impact_at(&self, now: Nanos) -> f64 {
        self.impact_history.iter()
            .map(|&(t, imp)| {
                let dt = (now - t) as f64;
                imp * (-self.gamma * dt).exp()
            })
            .sum()
    }

    /// Half-life of impact decay in seconds.
    pub fn half_life_seconds(&self) -> f64 {
        2.0_f64.ln() / (self.gamma * 1e9)
    }

    pub fn reset(&mut self) {
        self.impact_history.clear();
        self.current_impact = 0.0;
        self.last_update = 0;
    }
}

// ── Queue Position Tracker ────────────────────────────────────────────────────

/// Tracks estimated queue position and fill probability for pending limit orders.
#[derive(Debug, Clone)]
pub struct QueuePositionTracker {
    /// Estimated qty ahead of our order at a given price level.
    pub qty_ahead: Qty,
    /// Estimated total qty at level (excluding ours).
    pub level_total: Qty,
    /// Our order qty.
    pub our_qty: Qty,
    /// Price level.
    pub price: f64,
    /// Side.
    pub side: Side,
    /// Total volume that has traded at this level since we joined the queue.
    pub traded_through: Qty,
    /// Timestamp when we joined the queue.
    pub joined_at: Nanos,
}

impl QueuePositionTracker {
    pub fn new(price: f64, side: Side, qty_ahead: Qty, level_total: Qty, our_qty: Qty, ts: Nanos) -> Self {
        QueuePositionTracker {
            qty_ahead,
            level_total,
            our_qty,
            price,
            side,
            traded_through: 0.0,
            joined_at: ts,
        }
    }

    /// Record a trade at this level (updates traded_through and estimated position).
    pub fn record_trade(&mut self, traded_qty: Qty) {
        self.traded_through += traded_qty;
        // Reduce qty_ahead by the traded qty.
        self.qty_ahead = (self.qty_ahead - traded_qty).max(0.0);
    }

    /// Estimated probability of fill based on queue position.
    /// Uses a simple model: P(fill) ≈ P(enough volume arrives to consume our queue).
    ///
    /// If queue_arrival_rate = expected volume per unit time at this level,
    /// then time-to-fill ~ Geometric(p) where p ∝ queue_arrival_rate.
    pub fn fill_probability_horizon(&self, arrival_rate: Qty, horizon_secs: f64) -> f64 {
        if self.qty_ahead < 1e-9 {
            return 1.0; // We're at the front.
        }
        // Expected volume to arrive in horizon: arrival_rate * horizon_secs.
        let expected_volume = arrival_rate * horizon_secs;
        // P(volume >= qty_ahead): approximate as 1 - Poisson CDF.
        // Use complement of exponential approximation.
        let rate = expected_volume / self.qty_ahead;
        1.0 - (-rate).exp()
    }

    /// Position as fraction of level: 0 = front, 1 = back.
    pub fn queue_fraction(&self) -> f64 {
        if self.level_total < 1e-9 { return 0.0; }
        self.qty_ahead / self.level_total
    }

    pub fn is_likely_filled(&self) -> bool {
        self.qty_ahead < 1e-9
    }
}

// ── Agent Impact Tracker ──────────────────────────────────────────────────────

/// Master tracker combining all agent impact measurements.
pub struct AgentImpactTracker {
    pub agent_id: u32,
    pub footprint: Footprint,
    pub self_impact: SelfImpactModel,
    pub fills: Vec<AgentFill>,
    pub pending_orders: HashMap<u64, QueuePositionTracker>,
    /// Running VWAP of all fills.
    vwap_notional: f64,
    vwap_qty: Qty,
    /// Running P&L from fills vs mid at fill time.
    pub implementation_shortfall: f64,
    /// Count of fills.
    pub fill_count: u64,
}

impl AgentImpactTracker {
    pub fn new(agent_id: u32, impact_lambda: f64, impact_gamma: f64) -> Self {
        AgentImpactTracker {
            agent_id,
            footprint: Footprint::default(),
            self_impact: SelfImpactModel::new(impact_lambda, impact_gamma),
            fills: Vec::new(),
            pending_orders: HashMap::new(),
            vwap_notional: 0.0,
            vwap_qty: 0.0,
            implementation_shortfall: 0.0,
            fill_count: 0,
        }
    }

    /// Record a fill for this agent.
    pub fn record_fill(&mut self, lob_fill: &LobFill, mid: f64) {
        let agent_fill = AgentFill::from_lob_fill(lob_fill, mid);
        self.footprint.add_fill(&agent_fill);
        self.self_impact.record_fill(&agent_fill);

        // VWAP update.
        self.vwap_notional += agent_fill.price * agent_fill.qty;
        self.vwap_qty += agent_fill.qty;

        // Implementation shortfall: slippage vs mid.
        self.implementation_shortfall += agent_fill.slippage * agent_fill.qty;

        self.fill_count += 1;
        self.fills.push(agent_fill);
    }

    /// Track a new pending limit order.
    pub fn track_order(
        &mut self,
        order_id: u64,
        price: f64,
        side: Side,
        qty_ahead: Qty,
        level_total: Qty,
        our_qty: Qty,
        ts: Nanos,
    ) {
        self.pending_orders.insert(
            order_id,
            QueuePositionTracker::new(price, side, qty_ahead, level_total, our_qty, ts),
        );
    }

    /// Update queue position after a trade at a price level.
    pub fn update_queue(&mut self, price: f64, side: Side, traded_qty: Qty) {
        for tracker in self.pending_orders.values_mut() {
            if (tracker.price - price).abs() < 1e-9 && tracker.side == side {
                tracker.record_trade(traded_qty);
            }
        }
    }

    /// Remove a tracked order (filled or cancelled).
    pub fn remove_order(&mut self, order_id: u64) {
        self.pending_orders.remove(&order_id);
    }

    /// Current aggregate self-impact at time `now`.
    pub fn self_impact_at(&self, now: Nanos) -> f64 {
        self.self_impact.current_impact_at(now)
    }

    /// Overall VWAP of all fills.
    pub fn overall_vwap(&self) -> f64 {
        if self.vwap_qty < 1e-9 { return 0.0; }
        self.vwap_notional / self.vwap_qty
    }

    /// Average implementation shortfall per unit.
    pub fn avg_implementation_shortfall(&self) -> f64 {
        if self.vwap_qty < 1e-9 { return 0.0; }
        self.implementation_shortfall / self.vwap_qty
    }

    /// Signed order-flow imbalance contribution of this agent.
    pub fn agent_order_flow_imbalance(&self) -> f64 {
        let total = self.footprint.total_buy + self.footprint.total_sell;
        if total < 1e-9 { return 0.0; }
        (self.footprint.total_buy - self.footprint.total_sell) / total
    }

    /// Breakdown of fills by price bucket.
    pub fn price_distribution(&self, bucket_size: f64) -> Vec<(f64, Qty, Qty)> {
        let mut buckets: HashMap<i64, (Qty, Qty)> = HashMap::new();
        for fill in &self.fills {
            let bucket = (fill.price / bucket_size).floor() as i64;
            let entry = buckets.entry(bucket).or_insert((0.0, 0.0));
            match fill.side {
                Side::Bid => entry.0 += fill.qty,
                Side::Ask => entry.1 += fill.qty,
            }
        }
        let mut result: Vec<(f64, Qty, Qty)> = buckets.into_iter()
            .map(|(b, (buy, sell))| (b as f64 * bucket_size, buy, sell))
            .collect();
        result.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        result
    }

    /// Reset all statistics (start of new episode).
    pub fn reset(&mut self) {
        self.footprint.clear();
        self.self_impact.reset();
        self.fills.clear();
        self.pending_orders.clear();
        self.vwap_notional = 0.0;
        self.vwap_qty = 0.0;
        self.implementation_shortfall = 0.0;
        self.fill_count = 0;
    }

    /// Summary report.
    pub fn summary(&self, now: Nanos) -> ImpactSummary {
        let mut fp = self.footprint.clone();
        fp.compute_vwaps();
        ImpactSummary {
            agent_id: self.agent_id,
            total_buy_qty: self.footprint.total_buy,
            total_sell_qty: self.footprint.total_sell,
            net_position: self.footprint.net_volume(),
            overall_vwap: self.overall_vwap(),
            avg_slippage: self.avg_implementation_shortfall(),
            current_self_impact: self.self_impact_at(now),
            fill_count: self.fill_count,
            pending_order_count: self.pending_orders.len(),
        }
    }
}

/// Summary of agent impact metrics.
#[derive(Debug, Clone)]
pub struct ImpactSummary {
    pub agent_id: u32,
    pub total_buy_qty: Qty,
    pub total_sell_qty: Qty,
    pub net_position: Qty,
    pub overall_vwap: f64,
    pub avg_slippage: f64,
    pub current_self_impact: f64,
    pub fill_count: u64,
    pub pending_order_count: usize,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lob_engine::LobFill;

    fn make_fill(id: u64, price: f64, qty: f64, side: Side, ts: Nanos) -> LobFill {
        LobFill {
            aggressor_id: id,
            passive_id: id + 1000,
            price,
            qty,
            side,
            timestamp: ts,
            aggressor_agent: 1,
            passive_agent: 99,
        }
    }

    #[test]
    fn test_agent_fill_slippage() {
        let fill = make_fill(1, 100.05, 10.0, Side::Bid, 1_000_000_000);
        let agent_fill = AgentFill::from_lob_fill(&fill, 100.0);
        assert!((agent_fill.slippage - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_footprint_accumulation() {
        let mut fp = Footprint::default();
        let fill1 = AgentFill { timestamp: 1, price: 100.0, qty: 50.0, side: Side::Bid, mid_at_fill: 100.0, slippage: 0.0 };
        let fill2 = AgentFill { timestamp: 2, price: 99.0, qty: 30.0, side: Side::Ask, mid_at_fill: 99.5, slippage: 0.0 };
        fp.add_fill(&fill1);
        fp.add_fill(&fill2);
        fp.compute_vwaps();
        assert!((fp.total_buy - 50.0).abs() < 1e-9);
        assert!((fp.total_sell - 30.0).abs() < 1e-9);
        assert!((fp.vwap_buy - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_self_impact_decay() {
        let mut model = SelfImpactModel::new(0.001, 1.0); // 1 sec half-life
        let fill = AgentFill { timestamp: 1_000_000_000, price: 100.0, qty: 100.0, side: Side::Bid, mid_at_fill: 100.0, slippage: 0.0 };
        model.record_fill(&fill);
        let impact0 = model.current_impact_at(1_000_000_000);
        assert!((impact0 - 0.1).abs() < 1e-9); // 0.001 * 100 = 0.1

        // After 1 second (1e9 ns), impact should be ~0.1/2 = 0.05.
        let impact1 = model.current_impact_at(2_000_000_000);
        assert!((impact1 - 0.05).abs() < 1e-3);
    }

    #[test]
    fn test_queue_tracker() {
        let tracker = QueuePositionTracker::new(100.0, Side::Bid, 500.0, 1000.0, 100.0, 0);
        assert!((tracker.queue_fraction() - 0.5).abs() < 1e-9);
        let p = tracker.fill_probability_horizon(100.0, 5.0); // 500 qty in 5 secs at rate 100/sec
        assert!(p > 0.0 && p <= 1.0);
    }

    #[test]
    fn test_agent_impact_tracker() {
        let mut tracker = AgentImpactTracker::new(1, 0.001, 2.0);
        let fill = make_fill(1, 100.0, 50.0, Side::Bid, 1_000_000_000);
        tracker.record_fill(&fill, 99.9);
        let summary = tracker.summary(1_500_000_000);
        assert_eq!(summary.fill_count, 1);
        assert!((summary.total_buy_qty - 50.0).abs() < 1e-9);
        assert!((summary.avg_slippage - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_reset() {
        let mut tracker = AgentImpactTracker::new(1, 0.001, 2.0);
        let fill = make_fill(1, 100.0, 50.0, Side::Bid, 0);
        tracker.record_fill(&fill, 100.0);
        tracker.reset();
        assert_eq!(tracker.fill_count, 0);
        assert!((tracker.footprint.total_buy).abs() < 1e-9);
    }
}
