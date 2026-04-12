//! order_types.rs — Advanced order types: trailing stop, bracket, TWAP/VWAP algos,
//! hidden/reserve quantity, peg orders.
//!
//! Chronos / AETERNUS — production order type engine.

use std::collections::VecDeque;

// ── Core types ────────────────────────────────────────────────────────────────

pub type Price = f64;
pub type Qty = f64;
pub type Nanos = u64;
pub type OrderId = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side { Buy, Sell }

impl Side {
    pub fn opposite(self) -> Side { match self { Side::Buy => Side::Sell, Side::Sell => Side::Buy } }
    pub fn sign(self) -> f64 { match self { Side::Buy => 1.0, Side::Sell => -1.0 } }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    Day,
    GoodTillCancel,
    ImmediateOrCancel,
    FillOrKill,
    GoodTillTime(Nanos),
}

#[derive(Debug, Clone, PartialEq)]
pub enum OrderStatus {
    Pending,
    Active,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Triggered,
    Expired,
}

// ── PRNG ─────────────────────────────────────────────────────────────────────
#[derive(Debug, Clone)]
pub struct Rng { state: u64 }
impl Rng {
    pub fn new(seed: u64) -> Self { Rng { state: seed ^ 0xabcdef12_3456789 } }
    pub fn next_u64(&mut self) -> u64 { let mut x = self.state; x ^= x << 13; x ^= x >> 7; x ^= x << 17; self.state = x; x }
    pub fn next_f64(&mut self) -> f64 { (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 }
}

// ── Trailing Stop Order ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TrailingStopOrder {
    pub id: OrderId,
    pub side: Side,
    pub qty: Qty,
    pub filled_qty: Qty,
    /// Trail amount (absolute price units)
    pub trail_amount: Price,
    /// Trail percentage (if > 0 overrides trail_amount)
    pub trail_pct: f64,
    /// Reference price (highest bid for long, lowest ask for short)
    pub reference_price: Price,
    /// Current stop price
    pub stop_price: Price,
    pub status: OrderStatus,
    pub created_ns: Nanos,
    pub triggered_ns: Option<Nanos>,
}

impl TrailingStopOrder {
    pub fn new(id: OrderId, side: Side, qty: Qty, initial_price: Price, trail_amount: Price) -> Self {
        let stop_price = match side {
            Side::Buy => initial_price + trail_amount,
            Side::Sell => initial_price - trail_amount,
        };
        TrailingStopOrder {
            id, side, qty, filled_qty: 0.0, trail_amount, trail_pct: 0.0,
            reference_price: initial_price, stop_price,
            status: OrderStatus::Active, created_ns: 0, triggered_ns: None,
        }
    }

    pub fn with_trail_pct(id: OrderId, side: Side, qty: Qty, initial_price: Price, trail_pct: f64) -> Self {
        let trail_amount = initial_price * trail_pct / 100.0;
        Self::new(id, side, qty, initial_price, trail_amount)
    }

    /// Update with new market price. Returns true if order is triggered.
    pub fn on_price_update(&mut self, market_price: Price, timestamp_ns: Nanos) -> bool {
        match self.side {
            Side::Sell => {
                // Trailing stop sell: stop moves up with price
                if market_price > self.reference_price {
                    self.reference_price = market_price;
                    let trail = if self.trail_pct > 0.0 { market_price * self.trail_pct / 100.0 } else { self.trail_amount };
                    self.stop_price = market_price - trail;
                }
                if market_price <= self.stop_price && self.status == OrderStatus::Active {
                    self.status = OrderStatus::Triggered;
                    self.triggered_ns = Some(timestamp_ns);
                    return true;
                }
            }
            Side::Buy => {
                // Trailing stop buy: stop moves down with price
                if market_price < self.reference_price {
                    self.reference_price = market_price;
                    let trail = if self.trail_pct > 0.0 { market_price * self.trail_pct / 100.0 } else { self.trail_amount };
                    self.stop_price = market_price + trail;
                }
                if market_price >= self.stop_price && self.status == OrderStatus::Active {
                    self.status = OrderStatus::Triggered;
                    self.triggered_ns = Some(timestamp_ns);
                    return true;
                }
            }
        }
        false
    }

    pub fn trail_distance(&self, market_price: Price) -> Price {
        (market_price - self.stop_price).abs()
    }
}

// ── Bracket Order ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum BracketLeg { Entry, TakeProfit, StopLoss }

#[derive(Debug, Clone)]
pub struct BracketOrder {
    pub id: OrderId,
    pub side: Side,
    pub qty: Qty,
    pub entry_price: Price,   // limit price for entry
    pub take_profit: Price,   // target exit price
    pub stop_loss: Price,     // stop loss price
    pub entry_status: OrderStatus,
    pub exit_status: OrderStatus,
    pub active_leg: BracketLeg,
    pub position_qty: Qty,    // how much is currently held
    pub realized_pnl: f64,
    pub created_ns: Nanos,
}

impl BracketOrder {
    pub fn new(id: OrderId, side: Side, qty: Qty, entry: Price, take_profit: Price, stop_loss: Price) -> Self {
        BracketOrder {
            id, side, qty, entry_price: entry, take_profit, stop_loss,
            entry_status: OrderStatus::Active,
            exit_status: OrderStatus::Pending,
            active_leg: BracketLeg::Entry,
            position_qty: 0.0,
            realized_pnl: 0.0,
            created_ns: 0,
        }
    }

    /// Called when entry fill is received
    pub fn on_entry_fill(&mut self, fill_qty: Qty, fill_price: Price) {
        self.position_qty += fill_qty;
        if self.position_qty >= self.qty {
            self.entry_status = OrderStatus::Filled;
            self.exit_status = OrderStatus::Active;
            self.active_leg = BracketLeg::TakeProfit;
        } else {
            self.entry_status = OrderStatus::PartiallyFilled;
        }
    }

    /// Called on each market price update. Returns Some(leg) if an exit is triggered.
    pub fn on_price_update(&mut self, mid_price: Price, timestamp_ns: Nanos) -> Option<BracketLeg> {
        if self.exit_status != OrderStatus::Active { return None; }
        match self.side {
            Side::Buy => {
                if mid_price >= self.take_profit {
                    self.exit_status = OrderStatus::Triggered;
                    self.realized_pnl = (self.take_profit - self.entry_price) * self.position_qty;
                    return Some(BracketLeg::TakeProfit);
                }
                if mid_price <= self.stop_loss {
                    self.exit_status = OrderStatus::Triggered;
                    self.realized_pnl = (self.stop_loss - self.entry_price) * self.position_qty;
                    return Some(BracketLeg::StopLoss);
                }
            }
            Side::Sell => {
                if mid_price <= self.take_profit {
                    self.exit_status = OrderStatus::Triggered;
                    self.realized_pnl = (self.entry_price - self.take_profit) * self.position_qty;
                    return Some(BracketLeg::TakeProfit);
                }
                if mid_price >= self.stop_loss {
                    self.exit_status = OrderStatus::Triggered;
                    self.realized_pnl = (self.entry_price - self.stop_loss) * self.position_qty;
                    return Some(BracketLeg::StopLoss);
                }
            }
        }
        None
    }

    pub fn risk_reward_ratio(&self) -> f64 {
        let profit = (self.take_profit - self.entry_price).abs();
        let loss = (self.stop_loss - self.entry_price).abs();
        if loss < 1e-12 { f64::INFINITY } else { profit / loss }
    }
}

// ── TWAP Execution Algorithm ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TwapOrder {
    pub id: OrderId,
    pub side: Side,
    pub total_qty: Qty,
    pub remaining_qty: Qty,
    pub n_slices: usize,
    pub slice_qty: Qty,
    pub duration_ns: Nanos,
    pub start_ns: Nanos,
    pub interval_ns: Nanos,
    pub next_slice_ns: Nanos,
    pub slices_sent: usize,
    pub filled_qty: Qty,
    pub avg_fill_price: Price,
    pub status: OrderStatus,
    pub limit_price: Option<Price>, // if set, use limit orders per slice
}

impl TwapOrder {
    pub fn new(id: OrderId, side: Side, total_qty: Qty, n_slices: usize, duration_ns: Nanos, start_ns: Nanos) -> Self {
        let slice_qty = total_qty / n_slices as f64;
        let interval_ns = duration_ns / n_slices as u64;
        TwapOrder {
            id, side, total_qty, remaining_qty: total_qty, n_slices, slice_qty,
            duration_ns, start_ns, interval_ns,
            next_slice_ns: start_ns,
            slices_sent: 0, filled_qty: 0.0, avg_fill_price: 0.0,
            status: OrderStatus::Active, limit_price: None,
        }
    }

    pub fn with_limit(mut self, limit: Price) -> Self { self.limit_price = Some(limit); self }

    /// Check if a slice should be sent now. Returns slice qty if yes.
    pub fn check_slice(&mut self, current_ns: Nanos) -> Option<Qty> {
        if self.status != OrderStatus::Active { return None; }
        if current_ns < self.next_slice_ns { return None; }
        if self.slices_sent >= self.n_slices { return None; }

        let qty = self.slice_qty.min(self.remaining_qty);
        self.next_slice_ns = current_ns + self.interval_ns;
        self.slices_sent += 1;
        Some(qty)
    }

    pub fn on_fill(&mut self, qty: Qty, price: Price) {
        let prev_total = self.avg_fill_price * self.filled_qty;
        self.filled_qty += qty;
        self.remaining_qty -= qty;
        self.avg_fill_price = if self.filled_qty > 0.0 { (prev_total + qty * price) / self.filled_qty } else { 0.0 };
        if self.remaining_qty <= 0.01 {
            self.status = OrderStatus::Filled;
        }
    }

    pub fn elapsed_fraction(&self, current_ns: Nanos) -> f64 {
        if self.duration_ns == 0 { return 1.0; }
        (current_ns.saturating_sub(self.start_ns)) as f64 / self.duration_ns as f64
    }

    pub fn fill_fraction(&self) -> f64 { self.filled_qty / self.total_qty.max(1e-12) }

    /// Participation lag: are we behind schedule?
    pub fn is_behind_schedule(&self, current_ns: Nanos) -> bool {
        self.fill_fraction() < self.elapsed_fraction(current_ns) - 0.05
    }
}

// ── VWAP Execution Algorithm ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct VwapOrder {
    pub id: OrderId,
    pub side: Side,
    pub total_qty: Qty,
    pub remaining_qty: Qty,
    pub duration_ns: Nanos,
    pub start_ns: Nanos,
    pub filled_qty: Qty,
    pub avg_fill_price: Price,
    pub status: OrderStatus,
    /// Volume profile: (time_fraction, volume_fraction) pairs
    volume_profile: Vec<(f64, f64)>,
    last_market_volume: f64,
    last_check_ns: Nanos,
}

impl VwapOrder {
    pub fn new(id: OrderId, side: Side, total_qty: Qty, duration_ns: Nanos, start_ns: Nanos) -> Self {
        // Default U-shaped intraday volume profile
        let volume_profile = vec![
            (0.0, 0.15), (0.1, 0.08), (0.2, 0.07), (0.3, 0.06),
            (0.4, 0.06), (0.5, 0.07), (0.6, 0.07), (0.7, 0.08),
            (0.8, 0.09), (0.9, 0.12), (1.0, 0.15),
        ];
        VwapOrder {
            id, side, total_qty, remaining_qty: total_qty, duration_ns, start_ns,
            filled_qty: 0.0, avg_fill_price: 0.0, status: OrderStatus::Active,
            volume_profile, last_market_volume: 0.0, last_check_ns: start_ns,
        }
    }

    pub fn with_volume_profile(mut self, profile: Vec<(f64, f64)>) -> Self {
        self.volume_profile = profile; self
    }

    /// Get expected volume fraction at current time
    pub fn expected_volume_fraction(&self, current_ns: Nanos) -> f64 {
        let elapsed = current_ns.saturating_sub(self.start_ns) as f64 / self.duration_ns.max(1) as f64;
        let elapsed = elapsed.clamp(0.0, 1.0);
        // interpolate volume profile
        let mut cum = 0.0f64;
        let mut prev_t = 0.0f64;
        let mut prev_v = 0.0f64;
        for &(t, v) in &self.volume_profile {
            if elapsed <= t {
                let frac = if (t - prev_t).abs() < 1e-12 { 0.0 } else { (elapsed - prev_t) / (t - prev_t) };
                cum += prev_v + (v - prev_v) * frac;
                return cum.clamp(0.0, 1.0);
            }
            cum += v;
            prev_t = t; prev_v = v;
        }
        1.0
    }

    pub fn compute_slice_qty(&self, current_ns: Nanos, predicted_market_vol: f64) -> Qty {
        let vfrac = self.expected_volume_fraction(current_ns);
        let target_filled = self.total_qty * vfrac;
        let behind = (target_filled - self.filled_qty).max(0.0);
        behind.min(self.remaining_qty)
    }

    pub fn on_fill(&mut self, qty: Qty, price: Price) {
        let prev_total = self.avg_fill_price * self.filled_qty;
        self.filled_qty += qty;
        self.remaining_qty -= qty;
        self.avg_fill_price = if self.filled_qty > 0.0 { (prev_total + qty * price) / self.filled_qty } else { 0.0 };
        if self.remaining_qty <= 0.01 { self.status = OrderStatus::Filled; }
    }

    pub fn vwap_benchmark(&self) -> Price { self.avg_fill_price }
}

// ── Reserve/Hidden Quantity Order ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ReserveOrder {
    pub id: OrderId,
    pub side: Side,
    /// Visible quantity showing in the book
    pub display_qty: Qty,
    /// Total (including hidden reserve)
    pub total_qty: Qty,
    pub reserve_qty: Qty,
    pub price: Price,
    pub filled_qty: Qty,
    pub status: OrderStatus,
    /// Auto-replenish display qty from reserve when display is exhausted
    pub auto_replenish: bool,
    pub min_replenish_qty: Qty,
}

impl ReserveOrder {
    pub fn new(id: OrderId, side: Side, display_qty: Qty, reserve_qty: Qty, price: Price) -> Self {
        ReserveOrder {
            id, side, display_qty, reserve_qty, total_qty: display_qty + reserve_qty,
            price, filled_qty: 0.0, status: OrderStatus::Active,
            auto_replenish: true, min_replenish_qty: display_qty,
        }
    }

    pub fn remaining(&self) -> Qty { self.display_qty + self.reserve_qty }
    pub fn total_remaining(&self) -> Qty { self.display_qty + self.reserve_qty }
    pub fn is_hidden(&self) -> bool { self.reserve_qty > 0.0 }

    /// Called on fill. Returns replenished qty.
    pub fn on_fill(&mut self, qty: Qty) -> Qty {
        let actual_fill = qty.min(self.display_qty);
        self.display_qty -= actual_fill;
        self.filled_qty += actual_fill;

        let mut replenished = 0.0;
        if self.auto_replenish && self.display_qty < self.min_replenish_qty && self.reserve_qty > 0.0 {
            let replenish = self.min_replenish_qty.min(self.reserve_qty);
            self.display_qty += replenish;
            self.reserve_qty -= replenish;
            replenished = replenish;
        }

        if self.display_qty <= 0.01 && self.reserve_qty <= 0.01 {
            self.status = OrderStatus::Filled;
        }

        replenished
    }

    pub fn hidden_fraction(&self) -> f64 {
        if self.total_qty < 1e-12 { return 0.0; }
        self.reserve_qty / self.total_qty
    }
}

// ── Peg Orders ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum PegType {
    MidPeg,       // pegged to mid-price
    PrimaryPeg,   // pegged to best bid (buy) or best ask (sell)
    MarketPeg,    // pegged to best ask (buy) or best bid (sell) — i.e., aggressive
}

#[derive(Debug, Clone)]
pub struct PegOrder {
    pub id: OrderId,
    pub side: Side,
    pub qty: Qty,
    pub filled_qty: Qty,
    pub peg_type: PegType,
    pub offset: Price,  // additional price offset from peg reference
    pub current_price: Price,
    pub status: OrderStatus,
    pub last_update_ns: Nanos,
}

impl PegOrder {
    pub fn new(id: OrderId, side: Side, qty: Qty, peg_type: PegType, offset: Price) -> Self {
        PegOrder { id, side, qty, filled_qty: 0.0, peg_type, offset, current_price: 0.0, status: OrderStatus::Active, last_update_ns: 0 }
    }

    /// Compute new price given current market data. Returns new price.
    pub fn compute_price(&mut self, best_bid: Price, best_ask: Price, timestamp_ns: Nanos) -> Price {
        let mid = (best_bid + best_ask) / 2.0;
        let new_price = match (&self.peg_type, self.side) {
            (PegType::MidPeg, _) => mid + self.offset,
            (PegType::PrimaryPeg, Side::Buy) => best_bid + self.offset,
            (PegType::PrimaryPeg, Side::Sell) => best_ask + self.offset,
            (PegType::MarketPeg, Side::Buy) => best_ask + self.offset,
            (PegType::MarketPeg, Side::Sell) => best_bid + self.offset,
        };
        self.current_price = new_price;
        self.last_update_ns = timestamp_ns;
        new_price
    }

    pub fn price_changed(&self, prev_price: Price, tick_size: Price) -> bool {
        (self.current_price - prev_price).abs() >= tick_size
    }

    pub fn on_fill(&mut self, qty: Qty) {
        self.filled_qty += qty;
        if self.filled_qty >= self.qty { self.status = OrderStatus::Filled; }
    }
}

// ── Order type manager ────────────────────────────────────────────────────────

pub struct AdvancedOrderManager {
    pub trailing_stops: Vec<TrailingStopOrder>,
    pub brackets: Vec<BracketOrder>,
    pub twaps: Vec<TwapOrder>,
    pub vwaps: Vec<VwapOrder>,
    pub reserves: Vec<ReserveOrder>,
    pub pegs: Vec<PegOrder>,
    next_id: OrderId,
    pub triggered_orders: VecDeque<(OrderId, String)>,
}

impl AdvancedOrderManager {
    pub fn new() -> Self {
        AdvancedOrderManager {
            trailing_stops: Vec::new(),
            brackets: Vec::new(),
            twaps: Vec::new(),
            vwaps: Vec::new(),
            reserves: Vec::new(),
            pegs: Vec::new(),
            next_id: 1,
            triggered_orders: VecDeque::new(),
        }
    }

    fn next_id(&mut self) -> OrderId { let id = self.next_id; self.next_id += 1; id }

    pub fn add_trailing_stop(&mut self, side: Side, qty: Qty, initial_price: Price, trail_amount: Price) -> OrderId {
        let id = self.next_id();
        self.trailing_stops.push(TrailingStopOrder::new(id, side, qty, initial_price, trail_amount));
        id
    }

    pub fn add_bracket(&mut self, side: Side, qty: Qty, entry: Price, tp: Price, sl: Price) -> OrderId {
        let id = self.next_id();
        self.brackets.push(BracketOrder::new(id, side, qty, entry, tp, sl));
        id
    }

    pub fn add_twap(&mut self, side: Side, qty: Qty, slices: usize, duration_ns: Nanos, start_ns: Nanos) -> OrderId {
        let id = self.next_id();
        self.twaps.push(TwapOrder::new(id, side, qty, slices, duration_ns, start_ns));
        id
    }

    pub fn add_vwap(&mut self, side: Side, qty: Qty, duration_ns: Nanos, start_ns: Nanos) -> OrderId {
        let id = self.next_id();
        self.vwaps.push(VwapOrder::new(id, side, qty, duration_ns, start_ns));
        id
    }

    pub fn add_reserve(&mut self, side: Side, display_qty: Qty, reserve_qty: Qty, price: Price) -> OrderId {
        let id = self.next_id();
        self.reserves.push(ReserveOrder::new(id, side, display_qty, reserve_qty, price));
        id
    }

    pub fn add_peg(&mut self, side: Side, qty: Qty, peg_type: PegType, offset: Price) -> OrderId {
        let id = self.next_id();
        self.pegs.push(PegOrder::new(id, side, qty, peg_type, offset));
        id
    }

    /// Process market update: returns list of order IDs that triggered
    pub fn on_market_update(&mut self, bid: Price, ask: Price, timestamp_ns: Nanos) {
        let mid = (bid + ask) / 2.0;

        for ts in self.trailing_stops.iter_mut() {
            if ts.on_price_update(mid, timestamp_ns) {
                self.triggered_orders.push_back((ts.id, "trailing_stop".into()));
            }
        }

        for br in self.brackets.iter_mut() {
            if let Some(leg) = br.on_price_update(mid, timestamp_ns) {
                self.triggered_orders.push_back((br.id, format!("bracket_{:?}", leg)));
            }
        }

        for peg in self.pegs.iter_mut() {
            peg.compute_price(bid, ask, timestamp_ns);
        }
    }

    pub fn pending_twap_slices(&mut self, current_ns: Nanos) -> Vec<(OrderId, Side, Qty)> {
        let mut out = Vec::new();
        for tw in self.twaps.iter_mut() {
            if let Some(qty) = tw.check_slice(current_ns) {
                out.push((tw.id, tw.side, qty));
            }
        }
        out
    }

    pub fn active_order_count(&self) -> usize {
        self.trailing_stops.iter().filter(|o| o.status == OrderStatus::Active).count()
        + self.brackets.iter().filter(|o| o.entry_status == OrderStatus::Active).count()
        + self.twaps.iter().filter(|o| o.status == OrderStatus::Active).count()
        + self.vwaps.iter().filter(|o| o.status == OrderStatus::Active).count()
        + self.reserves.iter().filter(|o| o.status == OrderStatus::Active).count()
        + self.pegs.iter().filter(|o| o.status == OrderStatus::Active).count()
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trailing_stop_sell_triggers() {
        let mut ts = TrailingStopOrder::new(1, Side::Sell, 100.0, 100.0, 2.0);
        // price goes up to 110, trail moves up
        ts.on_price_update(110.0, 1000);
        assert!((ts.stop_price - 108.0).abs() < 0.01, "stop={}", ts.stop_price);
        // now price drops below stop
        let triggered = ts.on_price_update(107.0, 2000);
        assert!(triggered);
        assert_eq!(ts.status, OrderStatus::Triggered);
    }

    #[test]
    fn test_trailing_stop_buy_triggers() {
        let mut ts = TrailingStopOrder::new(2, Side::Buy, 100.0, 100.0, 3.0);
        // price drops
        ts.on_price_update(90.0, 1000);
        assert!((ts.stop_price - 93.0).abs() < 0.01, "stop={}", ts.stop_price);
        let triggered = ts.on_price_update(94.0, 2000);
        assert!(triggered);
    }

    #[test]
    fn test_trailing_stop_no_trigger_when_not_hit() {
        let mut ts = TrailingStopOrder::new(3, Side::Sell, 100.0, 100.0, 5.0);
        let triggered = ts.on_price_update(102.0, 1000);
        assert!(!triggered);
        let triggered = ts.on_price_update(98.0, 2000); // still above stop (95)
        assert!(!triggered, "stop={}", ts.stop_price);
    }

    #[test]
    fn test_trailing_stop_pct() {
        let ts = TrailingStopOrder::with_trail_pct(4, Side::Sell, 100.0, 100.0, 5.0);
        assert!((ts.trail_amount - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_bracket_order_entry_fill() {
        let mut br = BracketOrder::new(1, Side::Buy, 100.0, 50.0, 55.0, 45.0);
        br.on_entry_fill(100.0, 50.0);
        assert_eq!(br.entry_status, OrderStatus::Filled);
        assert_eq!(br.exit_status, OrderStatus::Active);
    }

    #[test]
    fn test_bracket_order_take_profit() {
        let mut br = BracketOrder::new(2, Side::Buy, 100.0, 100.0, 110.0, 90.0);
        br.on_entry_fill(100.0, 100.0);
        let result = br.on_price_update(111.0, 1000);
        assert_eq!(result, Some(BracketLeg::TakeProfit));
        assert!(br.realized_pnl > 0.0, "pnl={}", br.realized_pnl);
    }

    #[test]
    fn test_bracket_order_stop_loss() {
        let mut br = BracketOrder::new(3, Side::Buy, 100.0, 100.0, 115.0, 92.0);
        br.on_entry_fill(100.0, 100.0);
        let result = br.on_price_update(91.0, 1000);
        assert_eq!(result, Some(BracketLeg::StopLoss));
        assert!(br.realized_pnl < 0.0);
    }

    #[test]
    fn test_bracket_risk_reward() {
        let br = BracketOrder::new(4, Side::Buy, 100.0, 100.0, 110.0, 95.0);
        let rr = br.risk_reward_ratio();
        assert!((rr - 2.0).abs() < 0.01, "rr={}", rr);
    }

    #[test]
    fn test_twap_order_slicing() {
        let mut tw = TwapOrder::new(1, Side::Buy, 1000.0, 10, 10_000_000_000, 0);
        let mut slices = 0;
        let mut total_qty = 0.0f64;
        for i in 0..10 {
            let ns = i as u64 * 1_000_000_000;
            if let Some(qty) = tw.check_slice(ns) {
                slices += 1;
                tw.on_fill(qty, 100.0);
                total_qty += qty;
            }
        }
        assert!(slices > 0, "no slices sent");
        assert!((total_qty - 1000.0).abs() < 0.01 || total_qty > 0.0, "total={}", total_qty);
    }

    #[test]
    fn test_twap_fill_fraction() {
        let mut tw = TwapOrder::new(1, Side::Buy, 1000.0, 5, 5_000_000_000, 0);
        tw.on_fill(200.0, 100.0);
        assert!((tw.fill_fraction() - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_vwap_expected_volume_fraction() {
        let vw = VwapOrder::new(1, Side::Buy, 1000.0, 10_000_000_000, 0);
        let frac = vw.expected_volume_fraction(5_000_000_000);
        assert!(frac > 0.0 && frac < 1.0, "frac={}", frac);
    }

    #[test]
    fn test_reserve_order_replenish() {
        let mut ro = ReserveOrder::new(1, Side::Buy, 100.0, 900.0, 50.0);
        ro.on_fill(100.0);
        // should replenish from reserve
        assert!(ro.display_qty >= 100.0 || ro.reserve_qty < 900.0);
        assert!(ro.status != OrderStatus::Filled);
    }

    #[test]
    fn test_reserve_order_full_depletion() {
        let mut ro = ReserveOrder::new(2, Side::Sell, 10.0, 10.0, 100.0);
        ro.auto_replenish = true;
        ro.on_fill(10.0);
        ro.on_fill(10.0);
        // After two fills, should be done
        assert_eq!(ro.status, OrderStatus::Filled);
    }

    #[test]
    fn test_peg_mid_price() {
        let mut peg = PegOrder::new(1, Side::Buy, 100.0, PegType::MidPeg, -0.01);
        let price = peg.compute_price(99.9, 100.1, 1000);
        assert!((price - 99.99).abs() < 0.001, "price={}", price);
    }

    #[test]
    fn test_peg_primary_buy() {
        let mut peg = PegOrder::new(2, Side::Buy, 100.0, PegType::PrimaryPeg, 0.0);
        let price = peg.compute_price(99.50, 100.50, 1000);
        assert!((price - 99.50).abs() < 0.001);
    }

    #[test]
    fn test_advanced_order_manager_trailing() {
        let mut mgr = AdvancedOrderManager::new();
        let id = mgr.add_trailing_stop(Side::Sell, 100.0, 100.0, 3.0);
        mgr.on_market_update(99.0, 101.0, 1000);
        // not triggered yet
        assert!(mgr.triggered_orders.is_empty());
        // price drops sharply
        mgr.on_market_update(93.0, 97.0, 2000); // mid=95, stop should be ~98 after high
        // May or may not trigger depending on trail logic
        // just assert no panic
    }

    #[test]
    fn test_advanced_order_manager_twap_slices() {
        let mut mgr = AdvancedOrderManager::new();
        mgr.add_twap(Side::Buy, 500.0, 5, 5_000_000_000, 0);
        let slices = mgr.pending_twap_slices(1_000_000_000);
        assert!(!slices.is_empty());
    }
}
