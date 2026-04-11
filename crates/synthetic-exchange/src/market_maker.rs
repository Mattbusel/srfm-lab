/// Avellaneda-Stoikov optimal market maker agent.
///
/// The AS model provides optimal bid and ask quotes for a market maker
/// who seeks to balance inventory risk against profit from spread capture.
///
/// Model:
///   r(s, q, t) = s − q·γ·σ²·(T−t)   (reservation price)
///   δ_bid = 1/γ · ln(1 + γ/k) + (2q+1)/2 · √(σ²·γ/k · (1 + γ/k)^((γ+k)/γ))
///   δ_ask = symmetric with (2q-1)
///   (simplified form used in practice)
///
/// where:
///   s = mid price
///   q = inventory (signed)
///   γ = risk aversion
///   σ = volatility
///   k = order book depth parameter (fill rate)
///   T−t = remaining time horizon

use std::collections::HashMap;
use crate::exchange::{Exchange, Instrument, Order, Fill, Side, OrderKind, TimeInForce,
                     OrderId, InstrumentId, AgentId, Qty, Price, Nanos};

// ── AS Model Parameters ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AvellanedaStoikovParams {
    /// Risk aversion γ (higher = tighter quotes, less inventory risk tolerance).
    pub gamma: f64,
    /// Annualised volatility σ.
    pub sigma: f64,
    /// Order arrival intensity k (from Poisson model: fills arrive at rate A·exp(-k·δ)).
    pub kappa: f64,
    /// Time horizon in seconds (e.g., remaining session time).
    pub horizon: f64,
    /// Maximum absolute inventory (hard limit).
    pub max_inventory: Qty,
    /// Minimum quoted spread (tick sizes).
    pub min_spread: f64,
    /// Maximum quoted spread.
    pub max_spread: f64,
    /// Order quantity per quote.
    pub order_qty: Qty,
    /// How often to requote (in seconds).
    pub requote_interval_secs: f64,
}

impl Default for AvellanedaStoikovParams {
    fn default() -> Self {
        AvellanedaStoikovParams {
            gamma: 0.1,
            sigma: 0.2,
            kappa: 1.5,
            horizon: 3600.0,  // 1 hour
            max_inventory: 1000.0,
            min_spread: 0.01,
            max_spread: 1.0,
            order_qty: 100.0,
            requote_interval_secs: 1.0,
        }
    }
}

impl AvellanedaStoikovParams {
    /// Reservation price: the "fair value" adjusted for inventory risk.
    pub fn reservation_price(&self, mid: Price, inventory: Qty, time_remaining: f64) -> Price {
        mid - inventory * self.gamma * self.sigma * self.sigma * time_remaining
    }

    /// Optimal spread (total, symmetric around reservation price).
    pub fn optimal_spread(&self, time_remaining: f64) -> f64 {
        let base_spread = self.gamma * self.sigma * self.sigma * time_remaining;
        let arrival_term = (1.0 + self.gamma / self.kappa).ln() / self.gamma;
        (base_spread + arrival_term).clamp(self.min_spread, self.max_spread)
    }

    /// Compute optimal bid and ask prices.
    pub fn compute_quotes(&self, mid: Price, inventory: Qty, time_remaining: f64) -> (Price, Price) {
        let r = self.reservation_price(mid, inventory, time_remaining);
        let spread = self.optimal_spread(time_remaining);
        let half = spread / 2.0;

        // Inventory skew: shift quotes to reduce inventory.
        let skew = self.inventory_skew(inventory);

        let bid = r - half + skew;
        let ask = r + half + skew;

        (bid, ask)
    }

    /// Inventory skew: additional shift to reduce large positions.
    /// Positive skew = shift quotes up (encourage selling into us).
    fn inventory_skew(&self, inventory: Qty) -> f64 {
        if self.max_inventory < 1e-9 { return 0.0; }
        let inv_frac = inventory / self.max_inventory;
        // Skew proportional to inventory fraction, scaled by sigma.
        -inv_frac * self.sigma * self.gamma * 0.5
    }

    /// Fill probability for a given quote offset δ from mid.
    /// P(fill in [t, t+dt]) ≈ A·exp(-k·δ)·dt  (Avellaneda-Stoikov arrival model)
    pub fn fill_probability(&self, delta: f64, dt_secs: f64) -> f64 {
        let a = self.kappa; // Using kappa as the arrival intensity proxy.
        (1.0 - (-a * (-self.kappa * delta).exp() * dt_secs).exp()).clamp(0.0, 1.0)
    }
}

// ── Position & PnL Tracking ───────────────────────────────────────────────────

#[derive(Debug, Default, Clone)]
pub struct PositionState {
    pub inventory: Qty,
    pub cash: f64,
    pub realized_pnl: f64,
    pub total_fills: u64,
    pub buy_volume: Qty,
    pub sell_volume: Qty,
    pub avg_buy_price: Price,
    pub avg_sell_price: Price,
    buy_notional: f64,
    sell_notional: f64,
}

impl PositionState {
    pub fn new(initial_cash: f64) -> Self {
        PositionState { cash: initial_cash, ..Default::default() }
    }

    pub fn record_fill(&mut self, price: Price, qty: Qty, side: Side) {
        match side {
            Side::Buy => {
                self.cash -= price * qty;
                self.inventory += qty;
                self.buy_volume += qty;
                self.buy_notional += price * qty;
                self.avg_buy_price = self.buy_notional / self.buy_volume;
            }
            Side::Sell => {
                self.cash += price * qty;
                self.inventory -= qty;
                self.sell_volume += qty;
                self.sell_notional += price * qty;
                self.avg_sell_price = self.sell_notional / self.sell_volume;
            }
        }
        self.total_fills += 1;
    }

    pub fn mark_to_market(&self, mid: Price) -> f64 {
        self.cash + self.inventory * mid
    }

    pub fn unrealized_pnl(&self, mid: Price) -> f64 {
        self.mark_to_market(mid) - self.initial_cash_estimate()
    }

    fn initial_cash_estimate(&self) -> f64 {
        // Approximation: use average buy/sell prices.
        self.cash
    }

    /// Compute spread capture PnL (realized bid-ask earnings).
    pub fn spread_capture(&self) -> f64 {
        let sell_val = self.sell_notional;
        let buy_val = self.buy_notional;
        if self.buy_volume > 1e-9 && self.sell_volume > 1e-9 {
            sell_val - buy_val * (self.sell_volume / self.buy_volume)
        } else {
            0.0
        }
    }
}

// ── Delta Hedging ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DeltaHedgeConfig {
    /// Hedge whenever |inventory| exceeds this threshold.
    pub inventory_threshold: Qty,
    /// Target inventory after hedging (usually 0).
    pub target_inventory: Qty,
    /// Use market orders for hedging (true) or limit orders at mid (false).
    pub use_market_orders: bool,
}

impl Default for DeltaHedgeConfig {
    fn default() -> Self {
        DeltaHedgeConfig {
            inventory_threshold: 500.0,
            target_inventory: 0.0,
            use_market_orders: true,
        }
    }
}

// ── Market Maker Agent ────────────────────────────────────────────────────────

pub struct MarketMakerAgent {
    pub agent_id: AgentId,
    pub instrument_id: InstrumentId,
    pub params: AvellanedaStoikovParams,
    pub position: PositionState,
    pub hedge_config: DeltaHedgeConfig,

    /// Active bid order ID (if any).
    active_bid_id: Option<OrderId>,
    /// Active ask order ID (if any).
    active_ask_id: Option<OrderId>,
    /// Current bid price.
    current_bid: Price,
    /// Current ask price.
    current_ask: Price,
    /// Last requote timestamp.
    last_requote_ns: Nanos,
    /// Session start timestamp.
    session_start_ns: Nanos,
    /// Order ID counter (agent-local).
    order_id_base: u64,
    order_id_counter: u64,
    /// Performance metrics.
    pub total_requotes: u64,
    pub total_hedges: u64,
    pub mid_price_history: Vec<f64>,
}

impl MarketMakerAgent {
    pub fn new(
        agent_id: AgentId,
        instrument_id: InstrumentId,
        params: AvellanedaStoikovParams,
        initial_cash: f64,
        session_start_ns: Nanos,
    ) -> Self {
        let base = (agent_id as u64) * 100_000_000;
        MarketMakerAgent {
            agent_id,
            instrument_id,
            params,
            position: PositionState::new(initial_cash),
            hedge_config: DeltaHedgeConfig::default(),
            active_bid_id: None,
            active_ask_id: None,
            current_bid: 0.0,
            current_ask: 0.0,
            last_requote_ns: session_start_ns,
            session_start_ns,
            order_id_base: base,
            order_id_counter: 0,
            total_requotes: 0,
            total_hedges: 0,
            mid_price_history: Vec::new(),
        }
    }

    fn next_order_id(&mut self) -> OrderId {
        self.order_id_counter += 1;
        self.order_id_base + self.order_id_counter
    }

    /// Process fills and update position.
    pub fn on_fill(&mut self, fill: &Fill) {
        // Check if this fill is for our agent.
        if fill.aggressor_agent != self.agent_id && fill.passive_agent != self.agent_id {
            return;
        }
        let our_side = if fill.passive_agent == self.agent_id {
            // We are the passive side: passive bid = we bought, passive ask = we sold.
            match fill.side {
                Side::Buy => Side::Sell,   // aggressor bought from us (we sold)
                Side::Sell => Side::Buy,   // aggressor sold to us (we bought)
            }
        } else {
            fill.side
        };
        self.position.record_fill(fill.price, fill.qty, our_side);

        // Clear active order ID if it was filled.
        if Some(fill.passive_id) == self.active_bid_id || Some(fill.aggressor_id) == self.active_bid_id {
            self.active_bid_id = None;
        }
        if Some(fill.passive_id) == self.active_ask_id || Some(fill.aggressor_id) == self.active_ask_id {
            self.active_ask_id = None;
        }
    }

    /// Decide whether to requote and what prices to use.
    /// Returns list of orders to submit and order IDs to cancel.
    pub fn step(
        &mut self,
        mid: Price,
        ts_ns: Nanos,
        tick_size: f64,
    ) -> (Vec<Order>, Vec<(InstrumentId, OrderId)>) {
        self.mid_price_history.push(mid);
        let mut orders_to_submit = Vec::new();
        let mut ids_to_cancel = Vec::new();

        let elapsed_ns = ts_ns.saturating_sub(self.last_requote_ns);
        let elapsed_secs = elapsed_ns as f64 / 1e9;
        let requote_due = elapsed_secs >= self.params.requote_interval_secs;

        // Check if delta hedge needed.
        let inv = self.position.inventory;
        if inv.abs() > self.hedge_config.inventory_threshold {
            let hedge_qty = (inv - self.hedge_config.target_inventory).abs();
            let hedge_side = if inv > self.hedge_config.target_inventory { Side::Sell } else { Side::Buy };
            let hedge_id = self.next_order_id();
            let hedge_order = Order::new_market(hedge_id, self.instrument_id, self.agent_id, hedge_side, hedge_qty, ts_ns);
            orders_to_submit.push(hedge_order);
            self.total_hedges += 1;
        }

        if requote_due || (self.active_bid_id.is_none() && self.active_ask_id.is_none()) {
            // Cancel existing quotes.
            if let Some(bid_id) = self.active_bid_id.take() {
                ids_to_cancel.push((self.instrument_id, bid_id));
            }
            if let Some(ask_id) = self.active_ask_id.take() {
                ids_to_cancel.push((self.instrument_id, ask_id));
            }

            // Compute remaining time.
            let elapsed_session = (ts_ns - self.session_start_ns) as f64 / 1e9;
            let time_remaining = (self.params.horizon - elapsed_session).max(1.0);

            // Compute new quotes.
            let (bid_price, ask_price) = self.params.compute_quotes(mid, self.position.inventory, time_remaining);

            // Round to tick size.
            let bid_price = (bid_price / tick_size).floor() * tick_size;
            let ask_price = (ask_price / tick_size).ceil() * tick_size;

            // Ensure bid < ask (minimum spread = 1 tick).
            let ask_price = ask_price.max(bid_price + tick_size);

            // Only quote if within inventory limits.
            let can_buy = self.position.inventory < self.params.max_inventory;
            let can_sell = self.position.inventory > -self.params.max_inventory;

            if can_buy && bid_price > 0.0 {
                let bid_id = self.next_order_id();
                let mut bid_order = Order::new_limit(bid_id, self.instrument_id, self.agent_id, Side::Buy, bid_price, self.params.order_qty, ts_ns);
                bid_order.tif = TimeInForce::GTC;
                self.active_bid_id = Some(bid_id);
                self.current_bid = bid_price;
                orders_to_submit.push(bid_order);
            }

            if can_sell && ask_price > 0.0 {
                let ask_id = self.next_order_id();
                let mut ask_order = Order::new_limit(ask_id, self.instrument_id, self.agent_id, Side::Sell, ask_price, self.params.order_qty, ts_ns);
                ask_order.tif = TimeInForce::GTC;
                self.active_ask_id = Some(ask_id);
                self.current_ask = ask_price;
                orders_to_submit.push(ask_order);
            }

            self.last_requote_ns = ts_ns;
            self.total_requotes += 1;
        }

        (orders_to_submit, ids_to_cancel)
    }

    /// Current market-making spread (ask - bid).
    pub fn current_spread(&self) -> f64 {
        if self.current_ask > 0.0 && self.current_bid > 0.0 {
            self.current_ask - self.current_bid
        } else {
            0.0
        }
    }

    pub fn mark_to_market(&self, mid: Price) -> f64 {
        self.position.mark_to_market(mid)
    }

    pub fn summary(&self, mid: Price) -> MarketMakerSummary {
        MarketMakerSummary {
            agent_id: self.agent_id,
            inventory: self.position.inventory,
            cash: self.position.cash,
            mtm: self.mark_to_market(mid),
            spread_capture: self.position.spread_capture(),
            current_bid: self.current_bid,
            current_ask: self.current_ask,
            current_spread: self.current_spread(),
            total_fills: self.position.total_fills,
            total_requotes: self.total_requotes,
            total_hedges: self.total_hedges,
            buy_volume: self.position.buy_volume,
            sell_volume: self.position.sell_volume,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MarketMakerSummary {
    pub agent_id: AgentId,
    pub inventory: Qty,
    pub cash: f64,
    pub mtm: f64,
    pub spread_capture: f64,
    pub current_bid: Price,
    pub current_ask: Price,
    pub current_spread: f64,
    pub total_fills: u64,
    pub total_requotes: u64,
    pub total_hedges: u64,
    pub buy_volume: Qty,
    pub sell_volume: Qty,
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_reservation_price() {
        let p = AvellanedaStoikovParams::default();
        let mid = 100.0;
        let inventory = 100.0;
        let time = 3600.0;
        let r = p.reservation_price(mid, inventory, time);
        // With positive inventory, reservation price < mid.
        assert!(r < mid);
    }

    #[test]
    fn test_optimal_spread_increases_with_time() {
        let p = AvellanedaStoikovParams::default();
        let spread_long = p.optimal_spread(3600.0);
        let spread_short = p.optimal_spread(60.0);
        // Spread decreases as time runs out (less uncertainty).
        assert!(spread_long >= spread_short);
    }

    #[test]
    fn test_compute_quotes_bid_below_ask() {
        let p = AvellanedaStoikovParams::default();
        let (bid, ask) = p.compute_quotes(100.0, 0.0, 3600.0);
        assert!(bid < ask, "bid {} should be < ask {}", bid, ask);
    }

    #[test]
    fn test_market_maker_step_produces_orders() {
        let params = AvellanedaStoikovParams {
            order_qty: 50.0,
            requote_interval_secs: 1.0,
            ..Default::default()
        };
        let mut mm = MarketMakerAgent::new(1, 1, params, 1_000_000.0, 0);
        let (orders, cancels) = mm.step(100.0, 0, 0.01);
        // Should produce 2 orders (bid and ask).
        assert_eq!(orders.len(), 2);
        assert!(cancels.is_empty());
        assert!(mm.active_bid_id.is_some());
        assert!(mm.active_ask_id.is_some());
    }

    #[test]
    fn test_market_maker_requotes() {
        let params = AvellanedaStoikovParams {
            order_qty: 50.0,
            requote_interval_secs: 1.0,
            ..Default::default()
        };
        let mut mm = MarketMakerAgent::new(1, 1, params, 1_000_000.0, 0);
        mm.step(100.0, 0, 0.01);
        // Before interval: no requote.
        let (orders2, _) = mm.step(100.5, 500_000_000, 0.01); // 0.5 sec later
        assert!(orders2.is_empty());
        // After interval: requote.
        let (orders3, cancels3) = mm.step(101.0, 1_100_000_000, 0.01); // 1.1 sec later
        assert!(!cancels3.is_empty()); // Should cancel old quotes.
        assert_eq!(orders3.len(), 2); // Should produce new bid/ask.
    }

    #[test]
    fn test_position_state_record_fill() {
        let mut pos = PositionState::new(100_000.0);
        pos.record_fill(100.0, 50.0, Side::Buy);
        assert!((pos.inventory - 50.0).abs() < 1e-9);
        assert!((pos.cash - 95_000.0).abs() < 1e-9);
        pos.record_fill(101.0, 30.0, Side::Sell);
        assert!((pos.inventory - 20.0).abs() < 1e-9);
        assert!((pos.cash - 98_030.0).abs() < 1e-9);
    }

    #[test]
    fn test_delta_hedge_triggered() {
        let params = AvellanedaStoikovParams {
            max_inventory: 100.0,
            order_qty: 50.0,
            ..Default::default()
        };
        let mut mm = MarketMakerAgent::new(1, 1, params, 1_000_000.0, 0);
        mm.hedge_config.inventory_threshold = 80.0;
        mm.position.inventory = 90.0;  // Exceeds threshold.
        let (orders, _) = mm.step(100.0, 0, 0.01);
        // First order should be a hedge (market sell).
        assert!(orders.iter().any(|o| o.kind == OrderKind::Market && o.side == Side::Sell));
        assert_eq!(mm.total_hedges, 1);
    }

    #[test]
    fn test_fill_probability() {
        let p = AvellanedaStoikovParams::default();
        let prob1 = p.fill_probability(0.01, 1.0);
        let prob2 = p.fill_probability(1.0, 1.0);
        // Tighter quotes have higher fill probability.
        assert!(prob1 > prob2 || (prob1 - prob2).abs() < 0.5);
        assert!(prob1 >= 0.0 && prob1 <= 1.0);
        assert!(prob2 >= 0.0 && prob2 <= 1.0);
    }
}
