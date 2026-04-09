//! Synthetic Exchange Agent Engine
//!
//! A full agent-based exchange simulator with heterogeneous trader populations.
//! Implements Avellaneda-Stoikov market making, noise traders, informed traders,
//! momentum followers, mean-reversion strategies, latency simulation, and
//! aggregate population management with metrics collection.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// Pseudo-random number generator (xoshiro256** -- no external crate needed)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut s = [0u64; 4];
        let mut z = seed.wrapping_add(0x9e3779b97f4a7c15);
        for slot in s.iter_mut() {
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            z ^= z >> 31;
            *slot = z;
        }
        Self { s }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let result = (self.s[1].wrapping_mul(5)).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform f64 in [0, 1)
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform f64 in [lo, hi)
    pub fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Exponential distribution with given mean
    pub fn exponential(&mut self, mean: f64) -> f64 {
        let u = self.uniform();
        if u <= 1e-18 {
            mean * 40.0
        } else {
            -mean * u.ln()
        }
    }

    /// Standard normal via Box-Muller
    pub fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-18);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Normal with given mean and std
    pub fn normal_params(&mut self, mean: f64, std: f64) -> f64 {
        mean + std * self.normal()
    }

    /// Poisson-distributed integer with given lambda
    pub fn poisson(&mut self, lambda: f64) -> u64 {
        if lambda < 30.0 {
            // Knuth small-lambda algorithm
            let l = (-lambda).exp();
            let mut k: u64 = 0;
            let mut p: f64 = 1.0;
            loop {
                k += 1;
                p *= self.uniform();
                if p < l {
                    break;
                }
            }
            k - 1
        } else {
            // Normal approximation for large lambda
            let n = self.normal_params(lambda, lambda.sqrt());
            n.round().max(0.0) as u64
        }
    }

    /// Random integer in [0, n)
    pub fn rand_int(&mut self, n: u64) -> u64 {
        if n == 0 {
            return 0;
        }
        self.next_u64() % n
    }

    /// Bernoulli trial with given probability
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.uniform() < p
    }
}

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Buy,
    Sell,
}

impl Side {
    pub fn opposite(self) -> Self {
        match self {
            Side::Buy => Side::Sell,
            Side::Sell => Side::Buy,
        }
    }

    pub fn sign(self) -> f64 {
        match self {
            Side::Buy => 1.0,
            Side::Sell => -1.0,
        }
    }
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Buy => write!(f, "BUY"),
            Side::Sell => write!(f, "SELL"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderType {
    Market,
    Limit,
}

impl fmt::Display for OrderType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderType::Market => write!(f, "MKT"),
            OrderType::Limit => write!(f, "LMT"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct OrderFlow {
    pub order_id: u64,
    pub agent_id: u64,
    pub symbol: u32,
    pub side: Side,
    pub order_type: OrderType,
    pub price: f64,
    pub quantity: f64,
    pub timestamp: u64,
}

impl OrderFlow {
    pub fn new(
        order_id: u64,
        agent_id: u64,
        symbol: u32,
        side: Side,
        order_type: OrderType,
        price: f64,
        quantity: f64,
        timestamp: u64,
    ) -> Self {
        Self {
            order_id,
            agent_id,
            symbol,
            side,
            order_type,
            price,
            quantity,
            timestamp,
        }
    }

    pub fn market_order(
        order_id: u64,
        agent_id: u64,
        symbol: u32,
        side: Side,
        quantity: f64,
        timestamp: u64,
    ) -> Self {
        Self {
            order_id,
            agent_id,
            symbol,
            side,
            order_type: OrderType::Market,
            price: 0.0,
            quantity,
            timestamp,
        }
    }

    pub fn limit_order(
        order_id: u64,
        agent_id: u64,
        symbol: u32,
        side: Side,
        price: f64,
        quantity: f64,
        timestamp: u64,
    ) -> Self {
        Self {
            order_id,
            agent_id,
            symbol,
            side,
            order_type: OrderType::Limit,
            price,
            quantity,
            timestamp,
        }
    }

    pub fn notional(&self) -> f64 {
        self.price * self.quantity
    }

    pub fn is_aggressive(&self) -> bool {
        self.order_type == OrderType::Market
    }
}

#[derive(Debug, Clone)]
pub struct FillEvent {
    pub order_id: u64,
    pub agent_id: u64,
    pub fill_price: f64,
    pub fill_qty: f64,
    pub timestamp: u64,
    pub aggressor_side: Side,
    pub symbol: u32,
    pub fee: f64,
}

impl FillEvent {
    pub fn new(
        order_id: u64,
        agent_id: u64,
        fill_price: f64,
        fill_qty: f64,
        timestamp: u64,
        aggressor_side: Side,
        symbol: u32,
    ) -> Self {
        Self {
            order_id,
            agent_id,
            fill_price,
            fill_qty,
            timestamp,
            aggressor_side,
            symbol,
            fee: 0.0,
        }
    }

    pub fn with_fee(mut self, fee: f64) -> Self {
        self.fee = fee;
        self
    }

    pub fn notional(&self) -> f64 {
        self.fill_price * self.fill_qty
    }

    pub fn signed_qty(&self) -> f64 {
        match self.aggressor_side {
            Side::Buy => self.fill_qty,
            Side::Sell => -self.fill_qty,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MarketUpdate {
    pub best_bid: f64,
    pub best_ask: f64,
    pub last_trade_price: f64,
    pub last_trade_qty: f64,
    pub mid_price: f64,
    pub spread: f64,
    pub timestamp: u64,
    pub symbol: u32,
    pub bid_depth: f64,
    pub ask_depth: f64,
    pub trade_count: u64,
    pub cumulative_volume: f64,
    pub vwap: f64,
    pub high: f64,
    pub low: f64,
    pub open: f64,
}

impl MarketUpdate {
    pub fn new(best_bid: f64, best_ask: f64, timestamp: u64, symbol: u32) -> Self {
        let mid = (best_bid + best_ask) * 0.5;
        let spread = best_ask - best_bid;
        Self {
            best_bid,
            best_ask,
            last_trade_price: mid,
            last_trade_qty: 0.0,
            mid_price: mid,
            spread,
            timestamp,
            symbol,
            bid_depth: 0.0,
            ask_depth: 0.0,
            trade_count: 0,
            cumulative_volume: 0.0,
            vwap: mid,
            high: mid,
            low: mid,
            open: mid,
        }
    }

    pub fn with_last_trade(mut self, price: f64, qty: f64) -> Self {
        self.last_trade_price = price;
        self.last_trade_qty = qty;
        self
    }

    pub fn with_depth(mut self, bid_depth: f64, ask_depth: f64) -> Self {
        self.bid_depth = bid_depth;
        self.ask_depth = ask_depth;
        self
    }

    pub fn with_ohlcv(mut self, open: f64, high: f64, low: f64, volume: f64, count: u64) -> Self {
        self.open = open;
        self.high = high;
        self.low = low;
        self.cumulative_volume = volume;
        self.trade_count = count;
        self
    }

    pub fn with_vwap(mut self, vwap: f64) -> Self {
        self.vwap = vwap;
        self
    }

    pub fn imbalance(&self) -> f64 {
        let total = self.bid_depth + self.ask_depth;
        if total < 1e-15 {
            0.0
        } else {
            (self.bid_depth - self.ask_depth) / total
        }
    }

    pub fn microprice(&self) -> f64 {
        let total = self.bid_depth + self.ask_depth;
        if total < 1e-15 {
            self.mid_price
        } else {
            (self.best_bid * self.ask_depth + self.best_ask * self.bid_depth) / total
        }
    }

    pub fn relative_spread(&self) -> f64 {
        if self.mid_price.abs() < 1e-15 {
            0.0
        } else {
            self.spread / self.mid_price
        }
    }
}

// ---------------------------------------------------------------------------
// Agent trait
// ---------------------------------------------------------------------------

pub trait Agent: fmt::Debug {
    fn agent_id(&self) -> u64;
    fn agent_type(&self) -> &str;
    fn on_market_update(&mut self, update: &MarketUpdate);
    fn generate_orders(&mut self, timestamp: u64) -> Vec<OrderFlow>;
    fn on_fill(&mut self, fill: &FillEvent);
    fn inventory(&self) -> f64;
    fn pnl(&self) -> f64;
    fn reset(&mut self);
}

// ---------------------------------------------------------------------------
// Avellaneda-Stoikov Market Maker Agent (~250 lines)
// ---------------------------------------------------------------------------

/// Avellaneda-Stoikov optimal market-making model.
///
/// The reservation price is:  r = s - q * gamma * sigma^2 * (T - t)
/// The optimal spread is:     delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)
///
/// Where:
///   s     = mid price
///   q     = inventory
///   gamma = risk aversion parameter
///   sigma = volatility (per-period)
///   T     = terminal time horizon
///   t     = current time
///   k     = order arrival intensity parameter
#[derive(Debug, Clone)]
pub struct MarketMakerAgent {
    pub id: u64,
    pub gamma: f64,
    pub sigma: f64,
    pub time_horizon: f64,
    pub k: f64,
    pub max_inventory: f64,
    pub base_qty: f64,
    pub inventory_qty: f64,
    pub cash: f64,
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub quote_count: u64,
    pub fill_count: u64,
    pub total_volume: f64,
    current_time: f64,
    current_mid: f64,
    current_bid: f64,
    current_ask: f64,
    reservation_price: f64,
    optimal_spread: f64,
    optimal_bid: f64,
    optimal_ask: f64,
    bid_depth_factor: f64,
    ask_depth_factor: f64,
    last_fill_time: u64,
    price_history: VecDeque<f64>,
    realized_vol: f64,
    vol_lookback: usize,
    adaptive_gamma: bool,
    min_spread: f64,
    inventory_skew_factor: f64,
    rng: Rng,
    order_id_counter: u64,
    symbol: u32,
    active_bid_id: Option<u64>,
    active_ask_id: Option<u64>,
    fill_prices: Vec<f64>,
    position_entry_price: f64,
    max_drawdown: f64,
    peak_pnl: f64,
}

impl MarketMakerAgent {
    pub fn new(
        id: u64,
        gamma: f64,
        sigma: f64,
        time_horizon: f64,
        k: f64,
        max_inventory: f64,
        base_qty: f64,
        symbol: u32,
    ) -> Self {
        Self {
            id,
            gamma,
            sigma,
            time_horizon,
            k,
            max_inventory,
            base_qty,
            inventory_qty: 0.0,
            cash: 0.0,
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            quote_count: 0,
            fill_count: 0,
            total_volume: 0.0,
            current_time: 0.0,
            current_mid: 100.0,
            current_bid: 99.95,
            current_ask: 100.05,
            reservation_price: 100.0,
            optimal_spread: 0.1,
            optimal_bid: 99.95,
            optimal_ask: 100.05,
            bid_depth_factor: 1.0,
            ask_depth_factor: 1.0,
            last_fill_time: 0,
            price_history: VecDeque::with_capacity(512),
            realized_vol: sigma,
            vol_lookback: 100,
            adaptive_gamma: true,
            min_spread: 0.01,
            inventory_skew_factor: 0.5,
            rng: Rng::new(id.wrapping_mul(0xdeadbeef) ^ 0x12345678),
            order_id_counter: id * 1_000_000,
            symbol,
            active_bid_id: None,
            active_ask_id: None,
            fill_prices: Vec::with_capacity(1024),
            position_entry_price: 0.0,
            max_drawdown: 0.0,
            peak_pnl: 0.0,
        }
    }

    fn next_order_id(&mut self) -> u64 {
        self.order_id_counter += 1;
        self.order_id_counter
    }

    /// Compute realized volatility from recent price changes
    fn compute_realized_vol(&mut self) {
        if self.price_history.len() < 3 {
            return;
        }
        let n = self.price_history.len().min(self.vol_lookback);
        let mut sum_sq = 0.0;
        let mut count = 0usize;
        let start = self.price_history.len() - n;
        for i in (start + 1)..self.price_history.len() {
            let prev = self.price_history[i - 1];
            let curr = self.price_history[i];
            if prev > 1e-15 {
                let ret = (curr / prev).ln();
                sum_sq += ret * ret;
                count += 1;
            }
        }
        if count > 1 {
            self.realized_vol = (sum_sq / count as f64).sqrt();
            // Floor volatility to avoid degenerate spreads
            if self.realized_vol < 1e-6 {
                self.realized_vol = 1e-6;
            }
        }
    }

    /// Avellaneda-Stoikov reservation price
    fn compute_reservation_price(&self) -> f64 {
        let tau = (self.time_horizon - self.current_time).max(0.001);
        let sigma = self.effective_sigma();
        // r = s - q * gamma * sigma^2 * tau
        self.current_mid - self.inventory_qty * self.gamma * sigma * sigma * tau
    }

    /// Avellaneda-Stoikov optimal spread
    fn compute_optimal_spread(&self) -> f64 {
        let tau = (self.time_horizon - self.current_time).max(0.001);
        let sigma = self.effective_sigma();
        let g = self.effective_gamma();
        // delta = gamma * sigma^2 * tau + (2/gamma) * ln(1 + gamma/k)
        let intensity_component = if self.k > 1e-15 {
            (2.0 / g) * (1.0 + g / self.k).ln()
        } else {
            sigma * 2.0
        };
        let volatility_component = g * sigma * sigma * tau;
        (volatility_component + intensity_component).max(self.min_spread)
    }

    fn effective_sigma(&self) -> f64 {
        // Blend model sigma with realized vol
        0.5 * self.sigma + 0.5 * self.realized_vol
    }

    fn effective_gamma(&self) -> f64 {
        if !self.adaptive_gamma {
            return self.gamma;
        }
        // Increase risk aversion when inventory is large
        let inv_ratio = (self.inventory_qty.abs() / self.max_inventory).min(1.0);
        self.gamma * (1.0 + 2.0 * inv_ratio * inv_ratio)
    }

    /// Compute inventory-dependent quote sizes
    fn compute_quote_sizes(&self) -> (f64, f64) {
        let inv_ratio = self.inventory_qty / self.max_inventory;
        // Reduce size on the side where we are already long/short
        let bid_size = self.base_qty * (1.0 - self.inventory_skew_factor * inv_ratio).max(0.1);
        let ask_size = self.base_qty * (1.0 + self.inventory_skew_factor * inv_ratio).max(0.1);
        (bid_size, ask_size)
    }

    /// Skew quotes based on order-book imbalance
    fn apply_imbalance_skew(&mut self, imbalance: f64) {
        // Positive imbalance = more bid depth = price pressure up
        let skew = imbalance * self.effective_sigma() * 0.3;
        self.optimal_bid += skew;
        self.optimal_ask += skew;
    }

    fn update_pnl(&mut self) {
        self.unrealized_pnl = self.inventory_qty * self.current_mid;
        let total = self.cash + self.unrealized_pnl;
        if total > self.peak_pnl {
            self.peak_pnl = total;
        }
        let dd = self.peak_pnl - total;
        if dd > self.max_drawdown {
            self.max_drawdown = dd;
        }
    }

    /// Whether the agent should widen quotes to reduce risk
    fn should_widen_for_risk(&self) -> bool {
        self.inventory_qty.abs() > self.max_inventory * 0.8
    }

    /// Compute the full quote update cycle
    fn recompute_quotes(&mut self, imbalance: f64) {
        self.compute_realized_vol();
        self.reservation_price = self.compute_reservation_price();
        self.optimal_spread = self.compute_optimal_spread();

        if self.should_widen_for_risk() {
            // Emergency widening
            self.optimal_spread *= 1.5;
        }

        let half = self.optimal_spread * 0.5;
        self.optimal_bid = self.reservation_price - half;
        self.optimal_ask = self.reservation_price + half;

        self.apply_imbalance_skew(imbalance);

        // Clamp to not cross the market
        if self.optimal_bid >= self.current_ask {
            self.optimal_bid = self.current_ask - self.min_spread;
        }
        if self.optimal_ask <= self.current_bid {
            self.optimal_ask = self.current_bid + self.min_spread;
        }

        // Floor prices
        if self.optimal_bid < 0.01 {
            self.optimal_bid = 0.01;
        }
        if self.optimal_ask < self.optimal_bid + self.min_spread {
            self.optimal_ask = self.optimal_bid + self.min_spread;
        }
    }
}

impl Agent for MarketMakerAgent {
    fn agent_id(&self) -> u64 {
        self.id
    }

    fn agent_type(&self) -> &str {
        "MarketMaker_AvellanedaStoikov"
    }

    fn on_market_update(&mut self, update: &MarketUpdate) {
        self.current_mid = update.mid_price;
        self.current_bid = update.best_bid;
        self.current_ask = update.best_ask;
        self.current_time = update.timestamp as f64;

        if self.price_history.len() >= 512 {
            self.price_history.pop_front();
        }
        self.price_history.push_back(update.mid_price);

        let imbalance = update.imbalance();
        self.recompute_quotes(imbalance);
        self.update_pnl();
    }

    fn generate_orders(&mut self, timestamp: u64) -> Vec<OrderFlow> {
        let mut orders = Vec::with_capacity(2);
        let (bid_size, ask_size) = self.compute_quote_sizes();

        // Cancel-and-replace: always send fresh two-sided quote
        if self.inventory_qty.abs() < self.max_inventory || self.inventory_qty > 0.0 {
            // Post ask (sell) if inventory allows or we are long
            let ask_id = self.next_order_id();
            orders.push(OrderFlow::limit_order(
                ask_id,
                self.id,
                self.symbol,
                Side::Sell,
                self.optimal_ask,
                ask_size,
                timestamp,
            ));
            self.active_ask_id = Some(ask_id);
        }

        if self.inventory_qty.abs() < self.max_inventory || self.inventory_qty < 0.0 {
            // Post bid (buy) if inventory allows or we are short
            let bid_id = self.next_order_id();
            orders.push(OrderFlow::limit_order(
                bid_id,
                self.id,
                self.symbol,
                Side::Buy,
                self.optimal_bid,
                bid_size,
                timestamp,
            ));
            self.active_bid_id = Some(bid_id);
        }

        // If inventory dangerously high, send aggressive unwind order
        if self.inventory_qty.abs() > self.max_inventory * 0.95 {
            let unwind_side = if self.inventory_qty > 0.0 {
                Side::Sell
            } else {
                Side::Buy
            };
            let unwind_qty = (self.inventory_qty.abs() * 0.25).max(self.base_qty);
            let unwind_id = self.next_order_id();
            orders.push(OrderFlow::market_order(
                unwind_id,
                self.id,
                self.symbol,
                unwind_side,
                unwind_qty,
                timestamp,
            ));
        }

        self.quote_count += orders.len() as u64;
        orders
    }

    fn on_fill(&mut self, fill: &FillEvent) {
        let signed_qty = match fill.aggressor_side {
            Side::Buy => {
                // We were the passive seller (ask got lifted)
                if Some(fill.order_id) == self.active_ask_id {
                    -fill.fill_qty
                } else {
                    fill.fill_qty
                }
            }
            Side::Sell => {
                // We were the passive buyer (bid got hit)
                if Some(fill.order_id) == self.active_bid_id {
                    fill.fill_qty
                } else {
                    -fill.fill_qty
                }
            }
        };

        // Update position tracking
        let old_inv = self.inventory_qty;
        self.inventory_qty += signed_qty;
        self.cash -= signed_qty * fill.fill_price;
        self.cash -= fill.fee;
        self.fill_count += 1;
        self.total_volume += fill.fill_qty * fill.fill_price;
        self.last_fill_time = fill.timestamp;
        self.fill_prices.push(fill.fill_price);

        // Realized P&L when crossing zero or reducing position
        if old_inv.signum() != 0.0
            && (old_inv.signum() != self.inventory_qty.signum()
                || self.inventory_qty.abs() < old_inv.abs())
        {
            let closed_qty = if old_inv.signum() != self.inventory_qty.signum() {
                old_inv.abs()
            } else {
                old_inv.abs() - self.inventory_qty.abs()
            };
            let pnl_per_unit = if old_inv > 0.0 {
                fill.fill_price - self.position_entry_price
            } else {
                self.position_entry_price - fill.fill_price
            };
            self.realized_pnl += pnl_per_unit * closed_qty;
        }

        // Update entry price for new position
        if self.inventory_qty.abs() > 1e-15 {
            if old_inv.signum() != self.inventory_qty.signum() {
                self.position_entry_price = fill.fill_price;
            } else if self.inventory_qty.abs() > old_inv.abs() {
                // Average in
                let total_cost =
                    self.position_entry_price * old_inv.abs() + fill.fill_price * fill.fill_qty;
                self.position_entry_price = total_cost / self.inventory_qty.abs();
            }
        }

        self.update_pnl();
    }

    fn inventory(&self) -> f64 {
        self.inventory_qty
    }

    fn pnl(&self) -> f64 {
        self.cash + self.inventory_qty * self.current_mid
    }

    fn reset(&mut self) {
        self.inventory_qty = 0.0;
        self.cash = 0.0;
        self.realized_pnl = 0.0;
        self.unrealized_pnl = 0.0;
        self.quote_count = 0;
        self.fill_count = 0;
        self.total_volume = 0.0;
        self.current_time = 0.0;
        self.price_history.clear();
        self.fill_prices.clear();
        self.active_bid_id = None;
        self.active_ask_id = None;
        self.peak_pnl = 0.0;
        self.max_drawdown = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Noise Trader Agent (~200 lines)
// ---------------------------------------------------------------------------

/// Simulates retail/random flow. Sends market orders with exponentially
/// distributed sizes at Poisson-distributed intervals.
#[derive(Debug, Clone)]
pub struct NoiseTraderAgent {
    pub id: u64,
    pub intensity: f64,
    pub mean_size: f64,
    pub buy_bias: f64,
    pub max_position: f64,
    pub inventory_qty: f64,
    pub cash: f64,
    pub realized_pnl: f64,
    pub fill_count: u64,
    pub total_volume: f64,
    current_mid: f64,
    last_order_time: u64,
    cooldown_bars: u64,
    bars_since_order: u64,
    rng: Rng,
    order_id_counter: u64,
    symbol: u32,
    position_entry_price: f64,
    session_high_pnl: f64,
    session_low_pnl: f64,
    cluster_mode: bool,
    cluster_remaining: u32,
    cluster_side: Side,
    panic_threshold: f64,
    momentum_sensitivity: f64,
    recent_returns: VecDeque<f64>,
}

impl NoiseTraderAgent {
    pub fn new(
        id: u64,
        intensity: f64,
        mean_size: f64,
        buy_bias: f64,
        symbol: u32,
    ) -> Self {
        Self {
            id,
            intensity,
            mean_size,
            buy_bias,
            max_position: mean_size * 20.0,
            inventory_qty: 0.0,
            cash: 0.0,
            realized_pnl: 0.0,
            fill_count: 0,
            total_volume: 0.0,
            current_mid: 100.0,
            last_order_time: 0,
            cooldown_bars: 0,
            bars_since_order: 0,
            rng: Rng::new(id.wrapping_mul(0xcafebabe) ^ 0x87654321),
            order_id_counter: id * 1_000_000,
            symbol,
            position_entry_price: 0.0,
            session_high_pnl: 0.0,
            session_low_pnl: 0.0,
            cluster_mode: false,
            cluster_remaining: 0,
            cluster_side: Side::Buy,
            panic_threshold: 0.05,
            momentum_sensitivity: 0.3,
            recent_returns: VecDeque::with_capacity(32),
        }
    }

    fn next_order_id(&mut self) -> u64 {
        self.order_id_counter += 1;
        self.order_id_counter
    }

    fn should_trade(&mut self) -> bool {
        self.bars_since_order += 1;
        if self.bars_since_order < self.cooldown_bars {
            return false;
        }
        // Poisson arrival: probability of at least one order this bar
        let prob = 1.0 - (-self.intensity).exp();
        if self.cluster_mode && self.cluster_remaining > 0 {
            return true;
        }
        self.rng.bernoulli(prob)
    }

    fn choose_side(&mut self) -> Side {
        if self.cluster_mode && self.cluster_remaining > 0 {
            self.cluster_remaining -= 1;
            if self.cluster_remaining == 0 {
                self.cluster_mode = false;
            }
            return self.cluster_side;
        }

        // Position-aware: bias against current position
        let inv_ratio = if self.max_position > 1e-15 {
            self.inventory_qty / self.max_position
        } else {
            0.0
        };
        let adjusted_bias = (self.buy_bias - inv_ratio * 0.3).clamp(0.05, 0.95);

        // Momentum chasing: retail tends to chase
        let momentum_adj = if self.recent_returns.len() >= 3 {
            let sum: f64 = self.recent_returns.iter().rev().take(3).sum();
            sum * self.momentum_sensitivity
        } else {
            0.0
        };

        let final_bias = (adjusted_bias + momentum_adj).clamp(0.05, 0.95);

        if self.rng.bernoulli(final_bias) {
            Side::Buy
        } else {
            Side::Sell
        }
    }

    fn choose_size(&mut self) -> f64 {
        let raw = self.rng.exponential(self.mean_size);
        // Clamp to reasonable bounds and round to tick
        let clamped = raw.clamp(self.mean_size * 0.1, self.mean_size * 5.0);
        (clamped * 100.0).round() / 100.0
    }

    fn maybe_enter_cluster(&mut self) {
        // Occasionally retail traders "herd" -- burst of same-direction orders
        if self.rng.bernoulli(0.02) {
            self.cluster_mode = true;
            self.cluster_remaining = (self.rng.rand_int(4) + 2) as u32;
            self.cluster_side = if self.rng.bernoulli(self.buy_bias) {
                Side::Buy
            } else {
                Side::Sell
            };
        }
    }

    fn check_panic_liquidation(&mut self) -> Option<(Side, f64)> {
        // If losing badly, panic-close
        let total_pnl = self.cash + self.inventory_qty * self.current_mid;
        if self.inventory_qty.abs() > self.mean_size
            && total_pnl < self.session_low_pnl - self.panic_threshold * self.current_mid
        {
            let side = if self.inventory_qty > 0.0 {
                Side::Sell
            } else {
                Side::Buy
            };
            return Some((side, self.inventory_qty.abs()));
        }
        None
    }

    fn update_session_pnl(&mut self) {
        let pnl = self.cash + self.inventory_qty * self.current_mid;
        if pnl > self.session_high_pnl {
            self.session_high_pnl = pnl;
        }
        if pnl < self.session_low_pnl {
            self.session_low_pnl = pnl;
        }
    }
}

impl Agent for NoiseTraderAgent {
    fn agent_id(&self) -> u64 {
        self.id
    }

    fn agent_type(&self) -> &str {
        "NoiseTrader"
    }

    fn on_market_update(&mut self, update: &MarketUpdate) {
        let prev_mid = self.current_mid;
        self.current_mid = update.mid_price;

        if prev_mid > 1e-15 {
            let ret = (update.mid_price / prev_mid).ln();
            if self.recent_returns.len() >= 32 {
                self.recent_returns.pop_front();
            }
            self.recent_returns.push_back(ret);
        }

        self.update_session_pnl();
        self.maybe_enter_cluster();
    }

    fn generate_orders(&mut self, timestamp: u64) -> Vec<OrderFlow> {
        let mut orders = Vec::new();

        // Check for panic liquidation first
        if let Some((side, qty)) = self.check_panic_liquidation() {
            let oid = self.next_order_id();
            orders.push(OrderFlow::market_order(oid, self.id, self.symbol, side, qty, timestamp));
            self.bars_since_order = 0;
            self.last_order_time = timestamp;
            return orders;
        }

        if !self.should_trade() {
            return orders;
        }

        let side = self.choose_side();
        let qty = self.choose_size();

        // Check position limit
        let new_inv = self.inventory_qty + side.sign() * qty;
        if new_inv.abs() > self.max_position {
            return orders;
        }

        let oid = self.next_order_id();
        orders.push(OrderFlow::market_order(oid, self.id, self.symbol, side, qty, timestamp));
        self.bars_since_order = 0;
        self.last_order_time = timestamp;
        self.cooldown_bars = self.rng.rand_int(3);

        orders
    }

    fn on_fill(&mut self, fill: &FillEvent) {
        let signed = fill.signed_qty();
        let old_inv = self.inventory_qty;
        self.inventory_qty += signed;
        self.cash -= signed * fill.fill_price;
        self.cash -= fill.fee;
        self.fill_count += 1;
        self.total_volume += fill.fill_qty * fill.fill_price;

        if old_inv.signum() != 0.0 && self.inventory_qty.abs() < old_inv.abs() {
            let closed = old_inv.abs() - self.inventory_qty.abs();
            let entry = self.position_entry_price;
            let pnl = if old_inv > 0.0 {
                (fill.fill_price - entry) * closed
            } else {
                (entry - fill.fill_price) * closed
            };
            self.realized_pnl += pnl;
        }

        if self.inventory_qty.abs() > 1e-15 {
            if old_inv.signum() != self.inventory_qty.signum() || old_inv.abs() < 1e-15 {
                self.position_entry_price = fill.fill_price;
            } else if self.inventory_qty.abs() > old_inv.abs() {
                let tc = self.position_entry_price * old_inv.abs()
                    + fill.fill_price * fill.fill_qty;
                self.position_entry_price = tc / self.inventory_qty.abs();
            }
        }
    }

    fn inventory(&self) -> f64 {
        self.inventory_qty
    }

    fn pnl(&self) -> f64 {
        self.cash + self.inventory_qty * self.current_mid
    }

    fn reset(&mut self) {
        self.inventory_qty = 0.0;
        self.cash = 0.0;
        self.realized_pnl = 0.0;
        self.fill_count = 0;
        self.total_volume = 0.0;
        self.recent_returns.clear();
        self.cluster_mode = false;
        self.cluster_remaining = 0;
        self.session_high_pnl = 0.0;
        self.session_low_pnl = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Informed Trader Agent (~200 lines)
// ---------------------------------------------------------------------------

/// Has a private signal of "true value". Trades when market price deviates
/// from this signal by more than a configurable threshold.
#[derive(Debug, Clone)]
pub struct InformedTraderAgent {
    pub id: u64,
    pub true_value: f64,
    pub threshold: f64,
    pub aggression: f64,
    pub max_position: f64,
    pub base_qty: f64,
    pub signal_noise: f64,
    pub signal_decay: f64,
    pub inventory_qty: f64,
    pub cash: f64,
    pub realized_pnl: f64,
    pub fill_count: u64,
    pub total_volume: f64,
    current_mid: f64,
    perceived_value: f64,
    signal_strength: f64,
    rng: Rng,
    order_id_counter: u64,
    symbol: u32,
    position_entry_price: f64,
    last_signal_update: u64,
    signal_history: VecDeque<f64>,
    conviction: f64,
    max_orders_per_bar: u32,
    orders_this_bar: u32,
    use_limit_orders: bool,
    limit_offset_ticks: f64,
    stealth_mode: bool,
    stealth_max_size_ratio: f64,
    total_signal_pnl: f64,
    correct_predictions: u64,
    total_predictions: u64,
}

impl InformedTraderAgent {
    pub fn new(
        id: u64,
        true_value: f64,
        threshold: f64,
        aggression: f64,
        max_position: f64,
        base_qty: f64,
        symbol: u32,
    ) -> Self {
        Self {
            id,
            true_value,
            threshold,
            aggression,
            max_position,
            base_qty,
            signal_noise: 0.001,
            signal_decay: 0.999,
            inventory_qty: 0.0,
            cash: 0.0,
            realized_pnl: 0.0,
            fill_count: 0,
            total_volume: 0.0,
            current_mid: true_value,
            perceived_value: true_value,
            signal_strength: 1.0,
            rng: Rng::new(id.wrapping_mul(0xfeedface) ^ 0xabcdef01),
            order_id_counter: id * 1_000_000,
            symbol,
            position_entry_price: 0.0,
            last_signal_update: 0,
            signal_history: VecDeque::with_capacity(256),
            conviction: 1.0,
            max_orders_per_bar: 3,
            orders_this_bar: 0,
            use_limit_orders: false,
            limit_offset_ticks: 0.02,
            stealth_mode: true,
            stealth_max_size_ratio: 0.3,
            total_signal_pnl: 0.0,
            correct_predictions: 0,
            total_predictions: 0,
        }
    }

    pub fn with_stealth(mut self, enabled: bool, max_ratio: f64) -> Self {
        self.stealth_mode = enabled;
        self.stealth_max_size_ratio = max_ratio;
        self
    }

    pub fn with_limit_orders(mut self, enabled: bool, offset: f64) -> Self {
        self.use_limit_orders = enabled;
        self.limit_offset_ticks = offset;
        self
    }

    fn next_order_id(&mut self) -> u64 {
        self.order_id_counter += 1;
        self.order_id_counter
    }

    /// Update the internal signal with noise and decay
    fn update_signal(&mut self, timestamp: u64) {
        // Signal decays toward current mid over time
        let decay = self.signal_decay;
        let noise = self.rng.normal() * self.signal_noise * self.current_mid;
        self.perceived_value = self.true_value * decay + self.current_mid * (1.0 - decay) + noise;
        self.signal_strength *= 0.9995; // gradual loss of edge
        self.last_signal_update = timestamp;

        if self.signal_history.len() >= 256 {
            self.signal_history.pop_front();
        }
        self.signal_history.push_back(self.perceived_value);
    }

    /// Compute the deviation signal
    fn deviation(&self) -> f64 {
        if self.current_mid < 1e-15 {
            return 0.0;
        }
        (self.perceived_value - self.current_mid) / self.current_mid
    }

    /// Compute desired trade size based on deviation magnitude
    fn desired_size(&mut self) -> f64 {
        let dev = self.deviation().abs();
        let excess = dev - self.threshold;
        if excess <= 0.0 {
            return 0.0;
        }
        // Size proportional to excess deviation, scaled by aggression
        let raw = self.base_qty * (1.0 + self.aggression * excess / self.threshold);
        let clamped = raw.min(self.base_qty * 5.0);

        // Stealth: randomize size to avoid detection
        if self.stealth_mode {
            let jitter = self.rng.uniform_range(0.5, 1.0);
            (clamped * jitter * self.stealth_max_size_ratio).max(self.base_qty * 0.1)
        } else {
            clamped
        }
    }

    fn direction(&self) -> Option<Side> {
        let dev = self.deviation();
        if dev > self.threshold {
            Some(Side::Buy) // undervalued
        } else if dev < -self.threshold {
            Some(Side::Sell) // overvalued
        } else {
            None
        }
    }

    fn track_prediction(&mut self, side: Side) {
        self.total_predictions += 1;
        // Check if the last prediction was correct
        if self.signal_history.len() >= 2 {
            let prev = self.signal_history[self.signal_history.len() - 2];
            let curr = self.current_mid;
            let was_correct = match side {
                Side::Buy => curr > prev,
                Side::Sell => curr < prev,
            };
            if was_correct {
                self.correct_predictions += 1;
            }
        }
    }

    pub fn prediction_accuracy(&self) -> f64 {
        if self.total_predictions == 0 {
            0.5
        } else {
            self.correct_predictions as f64 / self.total_predictions as f64
        }
    }

    /// Set the true value externally (e.g. from simulation driver)
    pub fn set_true_value(&mut self, v: f64) {
        self.true_value = v;
    }
}

impl Agent for InformedTraderAgent {
    fn agent_id(&self) -> u64 {
        self.id
    }

    fn agent_type(&self) -> &str {
        "InformedTrader"
    }

    fn on_market_update(&mut self, update: &MarketUpdate) {
        self.current_mid = update.mid_price;
        self.update_signal(update.timestamp);
        self.orders_this_bar = 0;
    }

    fn generate_orders(&mut self, timestamp: u64) -> Vec<OrderFlow> {
        let mut orders = Vec::new();

        if self.orders_this_bar >= self.max_orders_per_bar {
            return orders;
        }

        let side = match self.direction() {
            Some(s) => s,
            None => return orders,
        };

        // Check position limit
        let new_inv = self.inventory_qty + side.sign() * self.base_qty;
        if new_inv.abs() > self.max_position {
            // Try to unwind opposite side instead
            if self.inventory_qty.abs() > self.base_qty * 0.5 {
                let unwind_side = if self.inventory_qty > 0.0 {
                    Side::Sell
                } else {
                    Side::Buy
                };
                let oid = self.next_order_id();
                let qty = (self.inventory_qty.abs() * 0.5).min(self.base_qty);
                orders.push(OrderFlow::market_order(
                    oid, self.id, self.symbol, unwind_side, qty, timestamp,
                ));
            }
            return orders;
        }

        let qty = self.desired_size();
        if qty < 1e-10 {
            return orders;
        }

        self.track_prediction(side);

        let oid = self.next_order_id();
        if self.use_limit_orders {
            let price = match side {
                Side::Buy => self.current_mid - self.limit_offset_ticks,
                Side::Sell => self.current_mid + self.limit_offset_ticks,
            };
            orders.push(OrderFlow::limit_order(
                oid, self.id, self.symbol, side, price, qty, timestamp,
            ));
        } else {
            orders.push(OrderFlow::market_order(
                oid, self.id, self.symbol, side, qty, timestamp,
            ));
        }

        self.orders_this_bar += 1;
        orders
    }

    fn on_fill(&mut self, fill: &FillEvent) {
        let signed = fill.signed_qty();
        let old_inv = self.inventory_qty;
        self.inventory_qty += signed;
        self.cash -= signed * fill.fill_price;
        self.cash -= fill.fee;
        self.fill_count += 1;
        self.total_volume += fill.fill_qty * fill.fill_price;

        if old_inv.signum() != 0.0 && self.inventory_qty.abs() < old_inv.abs() {
            let closed = old_inv.abs() - self.inventory_qty.abs();
            let pnl = if old_inv > 0.0 {
                (fill.fill_price - self.position_entry_price) * closed
            } else {
                (self.position_entry_price - fill.fill_price) * closed
            };
            self.realized_pnl += pnl;
            self.total_signal_pnl += pnl;
        }

        if self.inventory_qty.abs() > 1e-15 {
            if old_inv.signum() != self.inventory_qty.signum() || old_inv.abs() < 1e-15 {
                self.position_entry_price = fill.fill_price;
            } else if self.inventory_qty.abs() > old_inv.abs() {
                let tc = self.position_entry_price * old_inv.abs()
                    + fill.fill_price * fill.fill_qty;
                self.position_entry_price = tc / self.inventory_qty.abs();
            }
        }
    }

    fn inventory(&self) -> f64 {
        self.inventory_qty
    }

    fn pnl(&self) -> f64 {
        self.cash + self.inventory_qty * self.current_mid
    }

    fn reset(&mut self) {
        self.inventory_qty = 0.0;
        self.cash = 0.0;
        self.realized_pnl = 0.0;
        self.fill_count = 0;
        self.total_volume = 0.0;
        self.signal_history.clear();
        self.signal_strength = 1.0;
        self.total_signal_pnl = 0.0;
        self.correct_predictions = 0;
        self.total_predictions = 0;
    }
}

// ---------------------------------------------------------------------------
// Momentum Trader Agent (~200 lines)
// ---------------------------------------------------------------------------

/// Follows short-term price trends. Buys on upticks, sells on downticks.
/// Uses a configurable lookback window to compute momentum signal.
#[derive(Debug, Clone)]
pub struct MomentumTraderAgent {
    pub id: u64,
    pub lookback: usize,
    pub entry_threshold: f64,
    pub exit_threshold: f64,
    pub base_qty: f64,
    pub max_position: f64,
    pub momentum_decay: f64,
    pub inventory_qty: f64,
    pub cash: f64,
    pub realized_pnl: f64,
    pub fill_count: u64,
    pub total_volume: f64,
    current_mid: f64,
    price_history: VecDeque<f64>,
    momentum_signal: f64,
    ema_fast: f64,
    ema_slow: f64,
    ema_fast_alpha: f64,
    ema_slow_alpha: f64,
    ema_initialized: bool,
    rng: Rng,
    order_id_counter: u64,
    symbol: u32,
    position_entry_price: f64,
    position_entry_time: u64,
    max_hold_bars: u64,
    bars_in_position: u64,
    trailing_stop_pct: f64,
    trailing_stop_price: f64,
    stop_active: bool,
    trade_count: u64,
    winning_trades: u64,
    losing_trades: u64,
    consecutive_losses: u32,
    max_consecutive_losses: u32,
    cooldown_after_losses: u32,
    cooldown_remaining: u32,
    volume_filter: bool,
    min_volume_threshold: f64,
    last_volume: f64,
}

impl MomentumTraderAgent {
    pub fn new(
        id: u64,
        lookback: usize,
        entry_threshold: f64,
        base_qty: f64,
        max_position: f64,
        symbol: u32,
    ) -> Self {
        let fast_periods = (lookback as f64 * 0.3).max(2.0) as usize;
        let slow_periods = lookback;
        Self {
            id,
            lookback,
            entry_threshold,
            exit_threshold: entry_threshold * 0.3,
            base_qty,
            max_position,
            momentum_decay: 0.95,
            inventory_qty: 0.0,
            cash: 0.0,
            realized_pnl: 0.0,
            fill_count: 0,
            total_volume: 0.0,
            current_mid: 100.0,
            price_history: VecDeque::with_capacity(lookback + 16),
            momentum_signal: 0.0,
            ema_fast: 0.0,
            ema_slow: 0.0,
            ema_fast_alpha: 2.0 / (fast_periods as f64 + 1.0),
            ema_slow_alpha: 2.0 / (slow_periods as f64 + 1.0),
            ema_initialized: false,
            rng: Rng::new(id.wrapping_mul(0xbadf00d) ^ 0x55555555),
            order_id_counter: id * 1_000_000,
            symbol,
            position_entry_price: 0.0,
            position_entry_time: 0,
            max_hold_bars: (lookback * 3) as u64,
            bars_in_position: 0,
            trailing_stop_pct: 0.02,
            trailing_stop_price: 0.0,
            stop_active: false,
            trade_count: 0,
            winning_trades: 0,
            losing_trades: 0,
            consecutive_losses: 0,
            max_consecutive_losses: 5,
            cooldown_after_losses: 10,
            cooldown_remaining: 0,
            volume_filter: false,
            min_volume_threshold: 0.0,
            last_volume: 0.0,
        }
    }

    fn next_order_id(&mut self) -> u64 {
        self.order_id_counter += 1;
        self.order_id_counter
    }

    fn compute_momentum(&mut self) {
        if self.price_history.len() < 2 {
            self.momentum_signal = 0.0;
            return;
        }

        // Simple return-based momentum
        let n = self.price_history.len().min(self.lookback);
        let start_idx = self.price_history.len() - n;
        let start_price = self.price_history[start_idx];
        let end_price = *self.price_history.back().unwrap();

        if start_price > 1e-15 {
            let raw_mom = (end_price - start_price) / start_price;
            self.momentum_signal =
                self.momentum_decay * self.momentum_signal + (1.0 - self.momentum_decay) * raw_mom;
        }
    }

    fn update_emas(&mut self, price: f64) {
        if !self.ema_initialized {
            self.ema_fast = price;
            self.ema_slow = price;
            self.ema_initialized = true;
        } else {
            self.ema_fast = self.ema_fast_alpha * price + (1.0 - self.ema_fast_alpha) * self.ema_fast;
            self.ema_slow = self.ema_slow_alpha * price + (1.0 - self.ema_slow_alpha) * self.ema_slow;
        }
    }

    fn ema_crossover_signal(&self) -> f64 {
        if !self.ema_initialized || self.ema_slow.abs() < 1e-15 {
            return 0.0;
        }
        (self.ema_fast - self.ema_slow) / self.ema_slow
    }

    fn combined_signal(&self) -> f64 {
        // Blend raw momentum with EMA crossover
        0.6 * self.momentum_signal + 0.4 * self.ema_crossover_signal()
    }

    fn update_trailing_stop(&mut self) {
        if self.inventory_qty.abs() < 1e-15 {
            self.stop_active = false;
            return;
        }
        if self.inventory_qty > 0.0 {
            let new_stop = self.current_mid * (1.0 - self.trailing_stop_pct);
            if !self.stop_active || new_stop > self.trailing_stop_price {
                self.trailing_stop_price = new_stop;
                self.stop_active = true;
            }
        } else {
            let new_stop = self.current_mid * (1.0 + self.trailing_stop_pct);
            if !self.stop_active || new_stop < self.trailing_stop_price {
                self.trailing_stop_price = new_stop;
                self.stop_active = true;
            }
        }
    }

    fn check_stop_triggered(&self) -> bool {
        if !self.stop_active {
            return false;
        }
        if self.inventory_qty > 0.0 {
            self.current_mid <= self.trailing_stop_price
        } else {
            self.current_mid >= self.trailing_stop_price
        }
    }

    fn check_time_exit(&self) -> bool {
        self.bars_in_position > self.max_hold_bars
    }

    pub fn win_rate(&self) -> f64 {
        if self.trade_count == 0 {
            0.5
        } else {
            self.winning_trades as f64 / self.trade_count as f64
        }
    }
}

impl Agent for MomentumTraderAgent {
    fn agent_id(&self) -> u64 {
        self.id
    }

    fn agent_type(&self) -> &str {
        "MomentumTrader"
    }

    fn on_market_update(&mut self, update: &MarketUpdate) {
        self.current_mid = update.mid_price;
        self.last_volume = update.cumulative_volume;

        if self.price_history.len() >= self.lookback + 16 {
            self.price_history.pop_front();
        }
        self.price_history.push_back(update.mid_price);

        self.update_emas(update.mid_price);
        self.compute_momentum();
        self.update_trailing_stop();

        if self.inventory_qty.abs() > 1e-15 {
            self.bars_in_position += 1;
        }

        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
        }
    }

    fn generate_orders(&mut self, timestamp: u64) -> Vec<OrderFlow> {
        let mut orders = Vec::new();

        if self.cooldown_remaining > 0 {
            return orders;
        }

        // Check for exit conditions first
        if self.inventory_qty.abs() > 1e-15 {
            let should_exit =
                self.check_stop_triggered() || self.check_time_exit();

            // Also exit if momentum reversed
            let signal = self.combined_signal();
            let momentum_exit = (self.inventory_qty > 0.0 && signal < -self.exit_threshold)
                || (self.inventory_qty < 0.0 && signal > self.exit_threshold);

            if should_exit || momentum_exit {
                let side = if self.inventory_qty > 0.0 {
                    Side::Sell
                } else {
                    Side::Buy
                };
                let oid = self.next_order_id();
                orders.push(OrderFlow::market_order(
                    oid,
                    self.id,
                    self.symbol,
                    side,
                    self.inventory_qty.abs(),
                    timestamp,
                ));
                return orders;
            }
        }

        // Entry logic
        if self.price_history.len() < self.lookback {
            return orders;
        }

        let signal = self.combined_signal();
        if signal.abs() < self.entry_threshold {
            return orders;
        }

        // Volume filter
        if self.volume_filter && self.last_volume < self.min_volume_threshold {
            return orders;
        }

        let side = if signal > 0.0 { Side::Buy } else { Side::Sell };

        // Scale size by signal strength
        let size_multiplier = (signal.abs() / self.entry_threshold).min(3.0);
        let qty = self.base_qty * size_multiplier;

        let new_inv = self.inventory_qty + side.sign() * qty;
        if new_inv.abs() > self.max_position {
            return orders;
        }

        let oid = self.next_order_id();
        orders.push(OrderFlow::market_order(
            oid, self.id, self.symbol, side, qty, timestamp,
        ));
        orders
    }

    fn on_fill(&mut self, fill: &FillEvent) {
        let signed = fill.signed_qty();
        let old_inv = self.inventory_qty;
        self.inventory_qty += signed;
        self.cash -= signed * fill.fill_price;
        self.cash -= fill.fee;
        self.fill_count += 1;
        self.total_volume += fill.fill_qty * fill.fill_price;

        // Track realized PnL and win/loss
        if old_inv.signum() != 0.0 && self.inventory_qty.abs() < old_inv.abs() {
            let closed = old_inv.abs() - self.inventory_qty.abs();
            let pnl = if old_inv > 0.0 {
                (fill.fill_price - self.position_entry_price) * closed
            } else {
                (self.position_entry_price - fill.fill_price) * closed
            };
            self.realized_pnl += pnl;

            // If fully closed
            if self.inventory_qty.abs() < 1e-15 {
                self.trade_count += 1;
                if pnl > 0.0 {
                    self.winning_trades += 1;
                    self.consecutive_losses = 0;
                } else {
                    self.losing_trades += 1;
                    self.consecutive_losses += 1;
                    if self.consecutive_losses >= self.max_consecutive_losses {
                        self.cooldown_remaining = self.cooldown_after_losses;
                        self.consecutive_losses = 0;
                    }
                }
                self.bars_in_position = 0;
                self.stop_active = false;
            }
        }

        if self.inventory_qty.abs() > 1e-15 {
            if old_inv.abs() < 1e-15 || old_inv.signum() != self.inventory_qty.signum() {
                self.position_entry_price = fill.fill_price;
                self.position_entry_time = fill.timestamp;
                self.bars_in_position = 0;
                self.stop_active = false;
            } else if self.inventory_qty.abs() > old_inv.abs() {
                let tc = self.position_entry_price * old_inv.abs()
                    + fill.fill_price * fill.fill_qty;
                self.position_entry_price = tc / self.inventory_qty.abs();
            }
        }
    }

    fn inventory(&self) -> f64 {
        self.inventory_qty
    }

    fn pnl(&self) -> f64 {
        self.cash + self.inventory_qty * self.current_mid
    }

    fn reset(&mut self) {
        self.inventory_qty = 0.0;
        self.cash = 0.0;
        self.realized_pnl = 0.0;
        self.fill_count = 0;
        self.total_volume = 0.0;
        self.price_history.clear();
        self.momentum_signal = 0.0;
        self.ema_initialized = false;
        self.stop_active = false;
        self.bars_in_position = 0;
        self.trade_count = 0;
        self.winning_trades = 0;
        self.losing_trades = 0;
        self.consecutive_losses = 0;
        self.cooldown_remaining = 0;
    }
}

// ---------------------------------------------------------------------------
// Mean Reversion Trader Agent (~200 lines)
// ---------------------------------------------------------------------------

/// Trades against extreme moves by placing limit orders at deviations from VWAP.
/// Profits from price returning to mean. Uses Bollinger-band style thresholds.
#[derive(Debug, Clone)]
pub struct MeanReversionTrader {
    pub id: u64,
    pub vwap_lookback: usize,
    pub entry_z_score: f64,
    pub exit_z_score: f64,
    pub base_qty: f64,
    pub max_position: f64,
    pub limit_offset: f64,
    pub inventory_qty: f64,
    pub cash: f64,
    pub realized_pnl: f64,
    pub fill_count: u64,
    pub total_volume: f64,
    current_mid: f64,
    current_vwap: f64,
    price_history: VecDeque<f64>,
    volume_history: VecDeque<f64>,
    running_vwap_num: f64,
    running_vwap_den: f64,
    running_mean: f64,
    running_var: f64,
    var_count: usize,
    z_score: f64,
    rng: Rng,
    order_id_counter: u64,
    symbol: u32,
    position_entry_price: f64,
    active_orders: Vec<u64>,
    max_active_orders: usize,
    order_ttl_bars: u32,
    order_ages: HashMap<u64, u32>,
    bands_upper: f64,
    bands_lower: f64,
    band_width_multiplier: f64,
    reversion_speed_estimate: f64,
    price_mean_history: VecDeque<f64>,
    regime: MeanReversionRegime,
    regime_lookback: usize,
    vol_regime_threshold: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum MeanReversionRegime {
    LowVol,
    NormalVol,
    HighVol,
    Trending,
}

impl MeanReversionTrader {
    pub fn new(
        id: u64,
        vwap_lookback: usize,
        entry_z_score: f64,
        base_qty: f64,
        max_position: f64,
        symbol: u32,
    ) -> Self {
        Self {
            id,
            vwap_lookback,
            entry_z_score,
            exit_z_score: entry_z_score * 0.2,
            base_qty,
            max_position,
            limit_offset: 0.01,
            inventory_qty: 0.0,
            cash: 0.0,
            realized_pnl: 0.0,
            fill_count: 0,
            total_volume: 0.0,
            current_mid: 100.0,
            current_vwap: 100.0,
            price_history: VecDeque::with_capacity(vwap_lookback + 16),
            volume_history: VecDeque::with_capacity(vwap_lookback + 16),
            running_vwap_num: 0.0,
            running_vwap_den: 0.0,
            running_mean: 0.0,
            running_var: 0.0,
            var_count: 0,
            z_score: 0.0,
            rng: Rng::new(id.wrapping_mul(0xdecaf) ^ 0x99999999),
            order_id_counter: id * 1_000_000,
            symbol,
            position_entry_price: 0.0,
            active_orders: Vec::with_capacity(16),
            max_active_orders: 6,
            order_ttl_bars: 20,
            order_ages: HashMap::new(),
            bands_upper: 100.0,
            bands_lower: 100.0,
            band_width_multiplier: 2.0,
            reversion_speed_estimate: 0.0,
            price_mean_history: VecDeque::with_capacity(256),
            regime: MeanReversionRegime::NormalVol,
            regime_lookback: 50,
            vol_regime_threshold: 1.5,
        }
    }

    fn next_order_id(&mut self) -> u64 {
        self.order_id_counter += 1;
        self.order_id_counter
    }

    fn update_vwap(&mut self, price: f64, volume: f64) {
        if self.price_history.len() >= self.vwap_lookback + 16 {
            // Remove oldest contribution
            let old_p = self.price_history.pop_front().unwrap();
            let old_v = self.volume_history.pop_front().unwrap();
            self.running_vwap_num -= old_p * old_v;
            self.running_vwap_den -= old_v;
        }
        self.price_history.push_back(price);
        let vol = if volume < 1e-15 { 1.0 } else { volume };
        self.volume_history.push_back(vol);
        self.running_vwap_num += price * vol;
        self.running_vwap_den += vol;

        if self.running_vwap_den > 1e-15 {
            self.current_vwap = self.running_vwap_num / self.running_vwap_den;
        }
    }

    fn update_statistics(&mut self, price: f64) {
        self.var_count += 1;
        let n = self.var_count as f64;
        let delta = price - self.running_mean;
        self.running_mean += delta / n;
        let delta2 = price - self.running_mean;
        self.running_var += delta * delta2;

        // Use only recent window for variance
        let effective_var = if self.var_count > 2 {
            let window = self.var_count.min(self.vwap_lookback);
            self.running_var / window as f64
        } else {
            1e-10
        };

        let std_dev = effective_var.sqrt().max(1e-10);
        self.z_score = (price - self.current_vwap) / std_dev;

        self.bands_upper = self.current_vwap + self.band_width_multiplier * std_dev;
        self.bands_lower = self.current_vwap - self.band_width_multiplier * std_dev;

        if self.price_mean_history.len() >= 256 {
            self.price_mean_history.pop_front();
        }
        self.price_mean_history.push_back(self.current_vwap);
    }

    fn detect_regime(&mut self) {
        if self.price_history.len() < self.regime_lookback {
            return;
        }

        let n = self.regime_lookback;
        let start = self.price_history.len() - n;
        let mut sum_sq_ret = 0.0;
        let mut sum_ret = 0.0;
        let mut count = 0;
        for i in (start + 1)..self.price_history.len() {
            let prev = self.price_history[i - 1];
            if prev > 1e-15 {
                let r = (self.price_history[i] / prev).ln();
                sum_ret += r;
                sum_sq_ret += r * r;
                count += 1;
            }
        }

        if count < 2 {
            return;
        }

        let mean_ret = sum_ret / count as f64;
        let var_ret = sum_sq_ret / count as f64 - mean_ret * mean_ret;
        let vol = var_ret.sqrt();

        // Check if trending (autocorrelation proxy: are returns consistently same sign?)
        let mut same_sign_count = 0u32;
        for i in (start + 2)..self.price_history.len() {
            let r1 = self.price_history[i] - self.price_history[i - 1];
            let r2 = self.price_history[i - 1] - self.price_history[i - 2];
            if r1.signum() == r2.signum() {
                same_sign_count += 1;
            }
        }
        let trend_ratio = same_sign_count as f64 / (count - 1) as f64;

        let base_vol = 0.01; // ~1% per bar baseline
        self.regime = if trend_ratio > 0.65 {
            MeanReversionRegime::Trending
        } else if vol > base_vol * self.vol_regime_threshold {
            MeanReversionRegime::HighVol
        } else if vol < base_vol / self.vol_regime_threshold {
            MeanReversionRegime::LowVol
        } else {
            MeanReversionRegime::NormalVol
        };
    }

    fn effective_entry_z(&self) -> f64 {
        match self.regime {
            MeanReversionRegime::LowVol => self.entry_z_score * 0.8,
            MeanReversionRegime::NormalVol => self.entry_z_score,
            MeanReversionRegime::HighVol => self.entry_z_score * 1.5,
            MeanReversionRegime::Trending => self.entry_z_score * 3.0, // very conservative
        }
    }

    fn age_orders(&mut self) {
        let ttl = self.order_ttl_bars;
        self.order_ages.values_mut().for_each(|age| *age += 1);
        let expired: Vec<u64> = self
            .order_ages
            .iter()
            .filter(|(_, age)| **age >= ttl)
            .map(|(id, _)| *id)
            .collect();
        for id in &expired {
            self.order_ages.remove(id);
            self.active_orders.retain(|o| o != id);
        }
    }
}

impl Agent for MeanReversionTrader {
    fn agent_id(&self) -> u64 {
        self.id
    }

    fn agent_type(&self) -> &str {
        "MeanReversionTrader"
    }

    fn on_market_update(&mut self, update: &MarketUpdate) {
        self.current_mid = update.mid_price;
        let vol = if update.last_trade_qty > 0.0 {
            update.last_trade_qty
        } else {
            1.0
        };
        self.update_vwap(update.mid_price, vol);
        self.update_statistics(update.mid_price);
        self.detect_regime();
        self.age_orders();
    }

    fn generate_orders(&mut self, timestamp: u64) -> Vec<OrderFlow> {
        let mut orders = Vec::new();

        if self.price_history.len() < self.vwap_lookback / 2 {
            return orders;
        }

        // Don't trade if trending -- mean reversion fails in trends
        if self.regime == MeanReversionRegime::Trending {
            // But still unwind existing positions
            if self.inventory_qty.abs() > 1e-15 {
                let side = if self.inventory_qty > 0.0 {
                    Side::Sell
                } else {
                    Side::Buy
                };
                let oid = self.next_order_id();
                orders.push(OrderFlow::market_order(
                    oid,
                    self.id,
                    self.symbol,
                    side,
                    self.inventory_qty.abs(),
                    timestamp,
                ));
            }
            return orders;
        }

        // Exit: z-score returned to mean
        if self.inventory_qty.abs() > 1e-15 {
            let should_exit = (self.inventory_qty > 0.0 && self.z_score > -self.exit_z_score)
                || (self.inventory_qty < 0.0 && self.z_score < self.exit_z_score);
            if should_exit {
                let side = if self.inventory_qty > 0.0 {
                    Side::Sell
                } else {
                    Side::Buy
                };
                let oid = self.next_order_id();
                orders.push(OrderFlow::limit_order(
                    oid,
                    self.id,
                    self.symbol,
                    side,
                    self.current_vwap,
                    self.inventory_qty.abs(),
                    timestamp,
                ));
                self.active_orders.push(oid);
                self.order_ages.insert(oid, 0);
                return orders;
            }
        }

        // Entry: z-score exceeds threshold
        if self.active_orders.len() >= self.max_active_orders {
            return orders;
        }

        let ez = self.effective_entry_z();
        if self.z_score.abs() < ez {
            return orders;
        }

        let side = if self.z_score < -ez {
            Side::Buy // price below VWAP -- buy
        } else {
            Side::Sell // price above VWAP -- sell
        };

        let new_inv = self.inventory_qty + side.sign() * self.base_qty;
        if new_inv.abs() > self.max_position {
            return orders;
        }

        // Scale size by z-score magnitude
        let size_mult = (self.z_score.abs() / ez).min(2.5);
        let qty = self.base_qty * size_mult;

        // Place limit order at a level between current price and VWAP
        let limit_price = match side {
            Side::Buy => (self.current_mid - self.limit_offset).min(self.bands_lower),
            Side::Sell => (self.current_mid + self.limit_offset).max(self.bands_upper),
        };

        let oid = self.next_order_id();
        orders.push(OrderFlow::limit_order(
            oid, self.id, self.symbol, side, limit_price, qty, timestamp,
        ));
        self.active_orders.push(oid);
        self.order_ages.insert(oid, 0);

        // Layer in a second order closer to VWAP for better fill rate
        if self.active_orders.len() < self.max_active_orders {
            let mid_price = (limit_price + self.current_vwap) * 0.5;
            let oid2 = self.next_order_id();
            orders.push(OrderFlow::limit_order(
                oid2,
                self.id,
                self.symbol,
                side,
                mid_price,
                qty * 0.5,
                timestamp,
            ));
            self.active_orders.push(oid2);
            self.order_ages.insert(oid2, 0);
        }

        orders
    }

    fn on_fill(&mut self, fill: &FillEvent) {
        let signed = fill.signed_qty();
        let old_inv = self.inventory_qty;
        self.inventory_qty += signed;
        self.cash -= signed * fill.fill_price;
        self.cash -= fill.fee;
        self.fill_count += 1;
        self.total_volume += fill.fill_qty * fill.fill_price;

        self.active_orders.retain(|o| *o != fill.order_id);
        self.order_ages.remove(&fill.order_id);

        if old_inv.signum() != 0.0 && self.inventory_qty.abs() < old_inv.abs() {
            let closed = old_inv.abs() - self.inventory_qty.abs();
            let pnl = if old_inv > 0.0 {
                (fill.fill_price - self.position_entry_price) * closed
            } else {
                (self.position_entry_price - fill.fill_price) * closed
            };
            self.realized_pnl += pnl;
        }

        if self.inventory_qty.abs() > 1e-15 {
            if old_inv.abs() < 1e-15 || old_inv.signum() != self.inventory_qty.signum() {
                self.position_entry_price = fill.fill_price;
            } else if self.inventory_qty.abs() > old_inv.abs() {
                let tc = self.position_entry_price * old_inv.abs()
                    + fill.fill_price * fill.fill_qty;
                self.position_entry_price = tc / self.inventory_qty.abs();
            }
        }
    }

    fn inventory(&self) -> f64 {
        self.inventory_qty
    }

    fn pnl(&self) -> f64 {
        self.cash + self.inventory_qty * self.current_mid
    }

    fn reset(&mut self) {
        self.inventory_qty = 0.0;
        self.cash = 0.0;
        self.realized_pnl = 0.0;
        self.fill_count = 0;
        self.total_volume = 0.0;
        self.price_history.clear();
        self.volume_history.clear();
        self.running_vwap_num = 0.0;
        self.running_vwap_den = 0.0;
        self.running_mean = 0.0;
        self.running_var = 0.0;
        self.var_count = 0;
        self.z_score = 0.0;
        self.active_orders.clear();
        self.order_ages.clear();
        self.price_mean_history.clear();
        self.regime = MeanReversionRegime::NormalVol;
    }
}

// ---------------------------------------------------------------------------
// Latency Simulator (~150 lines)
// ---------------------------------------------------------------------------

/// Models network latency with Poisson-distributed jitter. Different agent
/// classes get different base latencies (HFT ~1us, retail ~100ms).
#[derive(Debug, Clone)]
pub struct LatencySimulator {
    profiles: HashMap<u64, LatencyProfile>,
    delayed_orders: BTreeMap<u64, Vec<OrderFlow>>,
    delayed_fills: BTreeMap<u64, Vec<FillEvent>>,
    rng: Rng,
    total_delayed_orders: u64,
    total_delayed_fills: u64,
    max_delay: u64,
    enable_jitter: bool,
    enable_drops: bool,
    drop_rate: f64,
    congestion_factor: f64,
    congestion_threshold: u64,
    orders_in_flight: u64,
}

#[derive(Debug, Clone)]
pub struct LatencyProfile {
    pub agent_id: u64,
    pub base_latency_us: u64,
    pub jitter_lambda: f64,
    pub colocated: bool,
    pub priority: u8,
}

impl LatencyProfile {
    pub fn hft(agent_id: u64) -> Self {
        Self {
            agent_id,
            base_latency_us: 1,
            jitter_lambda: 0.5,
            colocated: true,
            priority: 0,
        }
    }

    pub fn professional(agent_id: u64) -> Self {
        Self {
            agent_id,
            base_latency_us: 500,
            jitter_lambda: 50.0,
            colocated: false,
            priority: 1,
        }
    }

    pub fn retail(agent_id: u64) -> Self {
        Self {
            agent_id,
            base_latency_us: 50_000,
            jitter_lambda: 10_000.0,
            colocated: false,
            priority: 2,
        }
    }

    pub fn institutional(agent_id: u64) -> Self {
        Self {
            agent_id,
            base_latency_us: 5_000,
            jitter_lambda: 1_000.0,
            colocated: false,
            priority: 1,
        }
    }
}

impl LatencySimulator {
    pub fn new(seed: u64) -> Self {
        Self {
            profiles: HashMap::new(),
            delayed_orders: BTreeMap::new(),
            delayed_fills: BTreeMap::new(),
            rng: Rng::new(seed),
            total_delayed_orders: 0,
            total_delayed_fills: 0,
            max_delay: 1_000_000, // 1 second max
            enable_jitter: true,
            enable_drops: false,
            drop_rate: 0.0001,
            congestion_factor: 1.0,
            congestion_threshold: 1000,
            orders_in_flight: 0,
        }
    }

    pub fn register_agent(&mut self, profile: LatencyProfile) {
        self.profiles.insert(profile.agent_id, profile);
    }

    pub fn set_congestion(&mut self, factor: f64) {
        self.congestion_factor = factor;
    }

    fn compute_delay(&mut self, agent_id: u64) -> u64 {
        let profile = match self.profiles.get(&agent_id) {
            Some(p) => p.clone(),
            None => return 0,
        };

        let base = profile.base_latency_us as f64;
        let jitter = if self.enable_jitter {
            self.rng.poisson(profile.jitter_lambda) as f64
        } else {
            0.0
        };

        // Congestion multiplier: more orders in flight = more latency
        let congestion = if self.orders_in_flight > self.congestion_threshold {
            let excess =
                (self.orders_in_flight - self.congestion_threshold) as f64 / 1000.0;
            1.0 + excess * self.congestion_factor
        } else {
            1.0
        };

        let total = ((base + jitter) * congestion) as u64;
        total.min(self.max_delay)
    }

    fn should_drop(&mut self) -> bool {
        self.enable_drops && self.rng.bernoulli(self.drop_rate)
    }

    /// Submit an order through the latency simulator. Returns the delivery timestamp.
    pub fn submit_order(&mut self, order: OrderFlow, current_time: u64) -> Option<u64> {
        if self.should_drop() {
            return None;
        }

        let delay = self.compute_delay(order.agent_id);
        let delivery_time = current_time + delay;
        self.total_delayed_orders += 1;
        self.orders_in_flight += 1;

        self.delayed_orders
            .entry(delivery_time)
            .or_insert_with(Vec::new)
            .push(order);

        Some(delivery_time)
    }

    /// Submit a fill event through the latency simulator.
    pub fn submit_fill(&mut self, fill: FillEvent, current_time: u64) -> Option<u64> {
        let delay = self.compute_delay(fill.agent_id);
        let delivery_time = current_time + delay;
        self.total_delayed_fills += 1;

        self.delayed_fills
            .entry(delivery_time)
            .or_insert_with(Vec::new)
            .push(fill);

        Some(delivery_time)
    }

    /// Drain all orders that should have arrived by `current_time`.
    pub fn drain_orders(&mut self, current_time: u64) -> Vec<OrderFlow> {
        let mut result = Vec::new();
        let keys: Vec<u64> = self
            .delayed_orders
            .range(..=current_time)
            .map(|(k, _)| *k)
            .collect();
        for key in keys {
            if let Some(mut orders) = self.delayed_orders.remove(&key) {
                self.orders_in_flight =
                    self.orders_in_flight.saturating_sub(orders.len() as u64);
                result.append(&mut orders);
            }
        }
        // Sort by priority (colocated first)
        result.sort_by(|a, b| {
            let pa = self
                .profiles
                .get(&a.agent_id)
                .map_or(255, |p| p.priority);
            let pb = self
                .profiles
                .get(&b.agent_id)
                .map_or(255, |p| p.priority);
            pa.cmp(&pb).then(a.timestamp.cmp(&b.timestamp))
        });
        result
    }

    /// Drain all fills that should have arrived by `current_time`.
    pub fn drain_fills(&mut self, current_time: u64) -> Vec<FillEvent> {
        let mut result = Vec::new();
        let keys: Vec<u64> = self
            .delayed_fills
            .range(..=current_time)
            .map(|(k, _)| *k)
            .collect();
        for key in keys {
            if let Some(mut fills) = self.delayed_fills.remove(&key) {
                result.append(&mut fills);
            }
        }
        result
    }

    pub fn pending_order_count(&self) -> usize {
        self.delayed_orders.values().map(|v| v.len()).sum()
    }

    pub fn pending_fill_count(&self) -> usize {
        self.delayed_fills.values().map(|v| v.len()).sum()
    }
}

// ---------------------------------------------------------------------------
// Simulation Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SimulationConfig {
    pub n_market_makers: u32,
    pub n_noise_traders: u32,
    pub n_informed: u32,
    pub n_momentum: u32,
    pub n_mean_rev: u32,
    pub volatility: f64,
    pub true_value_drift: f64,
    pub simulation_bars: u64,
    pub initial_price: f64,
    pub tick_size: f64,
    pub maker_fee: f64,
    pub taker_fee: f64,
    pub symbol: u32,
    pub latency_enabled: bool,
    pub seed: u64,
    pub mm_gamma: f64,
    pub mm_k: f64,
    pub mm_max_inventory: f64,
    pub mm_base_qty: f64,
    pub noise_intensity: f64,
    pub noise_mean_size: f64,
    pub informed_threshold: f64,
    pub informed_aggression: f64,
    pub momentum_lookback: usize,
    pub momentum_threshold: f64,
    pub meanrev_lookback: usize,
    pub meanrev_z_score: f64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            n_market_makers: 5,
            n_noise_traders: 50,
            n_informed: 3,
            n_momentum: 10,
            n_mean_rev: 8,
            volatility: 0.02,
            true_value_drift: 0.0001,
            simulation_bars: 10_000,
            initial_price: 100.0,
            tick_size: 0.01,
            maker_fee: -0.0001,
            taker_fee: 0.0003,
            symbol: 0,
            latency_enabled: true,
            seed: 42,
            mm_gamma: 0.1,
            mm_k: 1.5,
            mm_max_inventory: 100.0,
            mm_base_qty: 1.0,
            noise_intensity: 0.3,
            noise_mean_size: 0.5,
            informed_threshold: 0.005,
            informed_aggression: 2.0,
            momentum_lookback: 20,
            momentum_threshold: 0.003,
            meanrev_lookback: 50,
            meanrev_z_score: 2.0,
        }
    }
}

impl SimulationConfig {
    pub fn total_agents(&self) -> u32 {
        self.n_market_makers
            + self.n_noise_traders
            + self.n_informed
            + self.n_momentum
            + self.n_mean_rev
    }
}

// ---------------------------------------------------------------------------
// Simulation Metrics
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SimulationMetrics {
    pub trade_count: u64,
    pub total_volume: f64,
    pub total_notional: f64,
    pub avg_spread: f64,
    pub min_spread: f64,
    pub max_spread: f64,
    pub price_volatility: f64,
    pub mm_pnl_distribution: Vec<f64>,
    pub informed_pnl: f64,
    pub noise_pnl: f64,
    pub momentum_pnl: f64,
    pub meanrev_pnl: f64,
    pub price_discovery_efficiency: f64,
    pub herfindahl_mm_share: f64,
    pub avg_mm_inventory: f64,
    pub max_mm_inventory: f64,
    pub bars_processed: u64,
    pub price_series: Vec<f64>,
    pub spread_series: Vec<f64>,
    pub volume_series: Vec<f64>,
    pub true_value_series: Vec<f64>,
    pub order_count: u64,
    pub fill_rate: f64,
    pub cancel_rate: f64,
    pub avg_latency_us: f64,
    pub market_impact_bps: f64,
}

impl SimulationMetrics {
    pub fn new() -> Self {
        Self {
            trade_count: 0,
            total_volume: 0.0,
            total_notional: 0.0,
            avg_spread: 0.0,
            min_spread: f64::MAX,
            max_spread: 0.0,
            price_volatility: 0.0,
            mm_pnl_distribution: Vec::new(),
            informed_pnl: 0.0,
            noise_pnl: 0.0,
            momentum_pnl: 0.0,
            meanrev_pnl: 0.0,
            price_discovery_efficiency: 0.0,
            herfindahl_mm_share: 0.0,
            avg_mm_inventory: 0.0,
            max_mm_inventory: 0.0,
            bars_processed: 0,
            price_series: Vec::with_capacity(10_000),
            spread_series: Vec::with_capacity(10_000),
            volume_series: Vec::with_capacity(10_000),
            true_value_series: Vec::with_capacity(10_000),
            order_count: 0,
            fill_rate: 0.0,
            cancel_rate: 0.0,
            avg_latency_us: 0.0,
            market_impact_bps: 0.0,
        }
    }

    pub fn record_bar(&mut self, spread: f64, price: f64, volume: f64, true_value: f64) {
        self.spread_series.push(spread);
        self.price_series.push(price);
        self.volume_series.push(volume);
        self.true_value_series.push(true_value);
        self.bars_processed += 1;

        if spread < self.min_spread {
            self.min_spread = spread;
        }
        if spread > self.max_spread {
            self.max_spread = spread;
        }
    }

    /// Compute all final aggregate metrics
    pub fn finalize(&mut self) {
        // Average spread
        if !self.spread_series.is_empty() {
            let sum: f64 = self.spread_series.iter().sum();
            self.avg_spread = sum / self.spread_series.len() as f64;
        }

        // Price volatility (realized, annualized proxy)
        if self.price_series.len() > 1 {
            let mut sum_sq = 0.0;
            let mut count = 0;
            for i in 1..self.price_series.len() {
                let prev = self.price_series[i - 1];
                if prev > 1e-15 {
                    let r = (self.price_series[i] / prev).ln();
                    sum_sq += r * r;
                    count += 1;
                }
            }
            if count > 0 {
                self.price_volatility = (sum_sq / count as f64).sqrt();
            }
        }

        // Price discovery efficiency: 1 - avg(|price - true_value| / true_value)
        if !self.price_series.is_empty() && !self.true_value_series.is_empty() {
            let n = self.price_series.len().min(self.true_value_series.len());
            let mut sum_dev = 0.0;
            for i in 0..n {
                let tv = self.true_value_series[i];
                if tv.abs() > 1e-15 {
                    sum_dev += ((self.price_series[i] - tv) / tv).abs();
                }
            }
            let avg_dev = sum_dev / n as f64;
            self.price_discovery_efficiency = (1.0 - avg_dev * 100.0).max(0.0);
        }

        // Fill rate
        if self.order_count > 0 {
            self.fill_rate = self.trade_count as f64 / self.order_count as f64;
        }
    }

    /// Compute Herfindahl index of market maker market share by volume
    pub fn compute_herfindahl(&mut self, mm_volumes: &[f64]) {
        let total: f64 = mm_volumes.iter().sum();
        if total < 1e-15 {
            self.herfindahl_mm_share = 1.0;
            return;
        }
        let mut hhi = 0.0;
        for v in mm_volumes {
            let share = v / total;
            hhi += share * share;
        }
        self.herfindahl_mm_share = hhi;
    }
}

impl Default for SimulationMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Agent Population Manager
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct AgentPopulation {
    market_makers: Vec<MarketMakerAgent>,
    noise_traders: Vec<NoiseTraderAgent>,
    informed_traders: Vec<InformedTraderAgent>,
    momentum_traders: Vec<MomentumTraderAgent>,
    mean_reversion_traders: Vec<MeanReversionTrader>,
    agent_count: u64,
    total_orders_generated: u64,
    total_fills_processed: u64,
}

impl AgentPopulation {
    pub fn new() -> Self {
        Self {
            market_makers: Vec::new(),
            noise_traders: Vec::new(),
            informed_traders: Vec::new(),
            momentum_traders: Vec::new(),
            mean_reversion_traders: Vec::new(),
            agent_count: 0,
            total_orders_generated: 0,
            total_fills_processed: 0,
        }
    }

    pub fn from_config(config: &SimulationConfig) -> Self {
        let mut pop = Self::new();
        let mut next_id: u64 = 1;
        let mut rng = Rng::new(config.seed);

        // Create market makers with slight parameter variation
        for _ in 0..config.n_market_makers {
            let gamma = config.mm_gamma * rng.uniform_range(0.7, 1.3);
            let k = config.mm_k * rng.uniform_range(0.8, 1.2);
            let max_inv = config.mm_max_inventory * rng.uniform_range(0.5, 1.5);
            let mm = MarketMakerAgent::new(
                next_id,
                gamma,
                config.volatility,
                config.simulation_bars as f64,
                k,
                max_inv,
                config.mm_base_qty,
                config.symbol,
            );
            pop.market_makers.push(mm);
            next_id += 1;
        }

        // Create noise traders
        for _ in 0..config.n_noise_traders {
            let intensity = config.noise_intensity * rng.uniform_range(0.3, 2.0);
            let mean_size = config.noise_mean_size * rng.uniform_range(0.5, 2.0);
            let buy_bias = rng.uniform_range(0.4, 0.6);
            let nt = NoiseTraderAgent::new(next_id, intensity, mean_size, buy_bias, config.symbol);
            pop.noise_traders.push(nt);
            next_id += 1;
        }

        // Create informed traders
        for _ in 0..config.n_informed {
            let threshold = config.informed_threshold * rng.uniform_range(0.5, 1.5);
            let aggression = config.informed_aggression * rng.uniform_range(0.8, 1.2);
            let it = InformedTraderAgent::new(
                next_id,
                config.initial_price,
                threshold,
                aggression,
                config.mm_max_inventory * 0.5,
                config.mm_base_qty * 2.0,
                config.symbol,
            );
            pop.informed_traders.push(it);
            next_id += 1;
        }

        // Create momentum traders
        for _ in 0..config.n_momentum {
            let lookback = (config.momentum_lookback as f64 * rng.uniform_range(0.5, 2.0)) as usize;
            let threshold = config.momentum_threshold * rng.uniform_range(0.5, 1.5);
            let mt = MomentumTraderAgent::new(
                next_id,
                lookback.max(3),
                threshold,
                config.mm_base_qty,
                config.mm_max_inventory * 0.3,
                config.symbol,
            );
            pop.momentum_traders.push(mt);
            next_id += 1;
        }

        // Create mean reversion traders
        for _ in 0..config.n_mean_rev {
            let lookback =
                (config.meanrev_lookback as f64 * rng.uniform_range(0.5, 2.0)) as usize;
            let z = config.meanrev_z_score * rng.uniform_range(0.8, 1.3);
            let mr = MeanReversionTrader::new(
                next_id,
                lookback.max(10),
                z,
                config.mm_base_qty,
                config.mm_max_inventory * 0.3,
                config.symbol,
            );
            pop.mean_reversion_traders.push(mr);
            next_id += 1;
        }

        pop.agent_count = next_id - 1;
        pop
    }

    pub fn agent_count(&self) -> u64 {
        self.agent_count
    }

    /// Dispatch a market update to all agents
    pub fn dispatch_market_update(&mut self, update: &MarketUpdate) {
        for mm in &mut self.market_makers {
            mm.on_market_update(update);
        }
        for nt in &mut self.noise_traders {
            nt.on_market_update(update);
        }
        for it in &mut self.informed_traders {
            it.on_market_update(update);
        }
        for mt in &mut self.momentum_traders {
            mt.on_market_update(update);
        }
        for mr in &mut self.mean_reversion_traders {
            mr.on_market_update(update);
        }
    }

    /// Collect orders from all agents
    pub fn collect_orders(&mut self, timestamp: u64) -> Vec<OrderFlow> {
        let mut all_orders = Vec::with_capacity(
            self.market_makers.len() * 2 + self.noise_traders.len() + 16,
        );

        for mm in &mut self.market_makers {
            let orders = mm.generate_orders(timestamp);
            all_orders.extend(orders);
        }
        for nt in &mut self.noise_traders {
            let orders = nt.generate_orders(timestamp);
            all_orders.extend(orders);
        }
        for it in &mut self.informed_traders {
            let orders = it.generate_orders(timestamp);
            all_orders.extend(orders);
        }
        for mt in &mut self.momentum_traders {
            let orders = mt.generate_orders(timestamp);
            all_orders.extend(orders);
        }
        for mr in &mut self.mean_reversion_traders {
            let orders = mr.generate_orders(timestamp);
            all_orders.extend(orders);
        }

        self.total_orders_generated += all_orders.len() as u64;
        all_orders
    }

    /// Dispatch fill events to the appropriate agents
    pub fn dispatch_fills(&mut self, fills: &[FillEvent]) {
        // Build agent_id -> fill mapping for efficiency
        let mut fill_map: HashMap<u64, Vec<&FillEvent>> = HashMap::new();
        for fill in fills {
            fill_map.entry(fill.agent_id).or_default().push(fill);
        }

        for mm in &mut self.market_makers {
            if let Some(agent_fills) = fill_map.get(&mm.id) {
                for fill in agent_fills {
                    mm.on_fill(fill);
                }
            }
        }
        for nt in &mut self.noise_traders {
            if let Some(agent_fills) = fill_map.get(&nt.id) {
                for fill in agent_fills {
                    nt.on_fill(fill);
                }
            }
        }
        for it in &mut self.informed_traders {
            if let Some(agent_fills) = fill_map.get(&it.id) {
                for fill in agent_fills {
                    it.on_fill(fill);
                }
            }
        }
        for mt in &mut self.momentum_traders {
            if let Some(agent_fills) = fill_map.get(&mt.id) {
                for fill in agent_fills {
                    mt.on_fill(fill);
                }
            }
        }
        for mr in &mut self.mean_reversion_traders {
            if let Some(agent_fills) = fill_map.get(&mr.id) {
                for fill in agent_fills {
                    mr.on_fill(fill);
                }
            }
        }

        self.total_fills_processed += fills.len() as u64;
    }

    /// Update the true value for all informed traders
    pub fn update_true_value(&mut self, true_value: f64) {
        for it in &mut self.informed_traders {
            it.set_true_value(true_value);
        }
    }

    /// Get aggregate statistics for metrics computation
    pub fn aggregate_stats(&self) -> PopulationStats {
        let mut stats = PopulationStats::default();

        for mm in &self.market_makers {
            let p = mm.pnl();
            stats.mm_pnls.push(p);
            stats.mm_volumes.push(mm.total_volume);
            stats.mm_inventories.push(mm.inventory_qty.abs());
            stats.mm_fill_counts.push(mm.fill_count);
        }

        for nt in &self.noise_traders {
            stats.noise_total_pnl += nt.pnl();
            stats.noise_total_volume += nt.total_volume;
        }

        for it in &self.informed_traders {
            stats.informed_total_pnl += it.pnl();
            stats.informed_total_volume += it.total_volume;
        }

        for mt in &self.momentum_traders {
            stats.momentum_total_pnl += mt.pnl();
            stats.momentum_total_volume += mt.total_volume;
        }

        for mr in &self.mean_reversion_traders {
            stats.meanrev_total_pnl += mr.pnl();
            stats.meanrev_total_volume += mr.total_volume;
        }

        stats
    }

    pub fn reset_all(&mut self) {
        for mm in &mut self.market_makers {
            mm.reset();
        }
        for nt in &mut self.noise_traders {
            nt.reset();
        }
        for it in &mut self.informed_traders {
            it.reset();
        }
        for mt in &mut self.momentum_traders {
            mt.reset();
        }
        for mr in &mut self.mean_reversion_traders {
            mr.reset();
        }
        self.total_orders_generated = 0;
        self.total_fills_processed = 0;
    }
}

impl Default for AgentPopulation {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Default)]
pub struct PopulationStats {
    pub mm_pnls: Vec<f64>,
    pub mm_volumes: Vec<f64>,
    pub mm_inventories: Vec<f64>,
    pub mm_fill_counts: Vec<u64>,
    pub noise_total_pnl: f64,
    pub noise_total_volume: f64,
    pub informed_total_pnl: f64,
    pub informed_total_volume: f64,
    pub momentum_total_pnl: f64,
    pub momentum_total_volume: f64,
    pub meanrev_total_pnl: f64,
    pub meanrev_total_volume: f64,
}

// ---------------------------------------------------------------------------
// Simple Order Book (for internal matching when no external engine is used)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BookLevel {
    price: f64,
    orders: VecDeque<(u64, u64, f64)>, // (order_id, agent_id, remaining_qty)
}

#[derive(Debug, Clone)]
struct SimpleOrderBook {
    bids: Vec<BookLevel>,
    asks: Vec<BookLevel>,
    tick_size: f64,
}

impl SimpleOrderBook {
    fn new(tick_size: f64) -> Self {
        Self {
            bids: Vec::with_capacity(64),
            asks: Vec::with_capacity(64),
            tick_size,
        }
    }

    fn round_price(&self, price: f64) -> f64 {
        (price / self.tick_size).round() * self.tick_size
    }

    fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    fn bid_depth(&self) -> f64 {
        self.bids
            .iter()
            .flat_map(|l| l.orders.iter())
            .map(|(_, _, q)| q)
            .sum()
    }

    fn ask_depth(&self) -> f64 {
        self.asks
            .iter()
            .flat_map(|l| l.orders.iter())
            .map(|(_, _, q)| q)
            .sum()
    }

    fn insert_bid(&mut self, price: f64, order_id: u64, agent_id: u64, qty: f64) {
        let price = self.round_price(price);
        // Find or create level (bids sorted descending)
        let pos = self.bids.iter().position(|l| l.price <= price);
        match pos {
            Some(i) if (self.bids[i].price - price).abs() < self.tick_size * 0.5 => {
                self.bids[i]
                    .orders
                    .push_back((order_id, agent_id, qty));
            }
            Some(i) => {
                let mut orders = VecDeque::new();
                orders.push_back((order_id, agent_id, qty));
                self.bids.insert(i, BookLevel { price, orders });
            }
            None => {
                let mut orders = VecDeque::new();
                orders.push_back((order_id, agent_id, qty));
                self.bids.push(BookLevel { price, orders });
            }
        }
    }

    fn insert_ask(&mut self, price: f64, order_id: u64, agent_id: u64, qty: f64) {
        let price = self.round_price(price);
        // Asks sorted ascending
        let pos = self.asks.iter().position(|l| l.price >= price);
        match pos {
            Some(i) if (self.asks[i].price - price).abs() < self.tick_size * 0.5 => {
                self.asks[i]
                    .orders
                    .push_back((order_id, agent_id, qty));
            }
            Some(i) => {
                let mut orders = VecDeque::new();
                orders.push_back((order_id, agent_id, qty));
                self.asks.insert(i, BookLevel { price, orders });
            }
            None => {
                let mut orders = VecDeque::new();
                orders.push_back((order_id, agent_id, qty));
                self.asks.push(BookLevel { price, orders });
            }
        }
    }

    /// Match an incoming order against the book. Returns fills.
    fn match_order(&mut self, order: &OrderFlow, timestamp: u64) -> Vec<FillEvent> {
        let mut fills = Vec::new();
        let mut remaining = order.quantity;

        match order.side {
            Side::Buy => {
                // Match against asks
                while remaining > 1e-15 && !self.asks.is_empty() {
                    let can_match = order.order_type == OrderType::Market
                        || order.price >= self.asks[0].price;
                    if !can_match {
                        break;
                    }
                    let level = &mut self.asks[0];
                    while remaining > 1e-15 && !level.orders.is_empty() {
                        let (resting_oid, resting_aid, ref mut resting_qty) =
                            level.orders[0];
                        let fill_qty = remaining.min(*resting_qty);
                        let fill_price = level.price;

                        // Fill for the aggressor
                        fills.push(FillEvent::new(
                            order.order_id,
                            order.agent_id,
                            fill_price,
                            fill_qty,
                            timestamp,
                            Side::Buy,
                            order.symbol,
                        ));
                        // Fill for the passive side
                        fills.push(FillEvent::new(
                            resting_oid,
                            resting_aid,
                            fill_price,
                            fill_qty,
                            timestamp,
                            Side::Buy,
                            order.symbol,
                        ));

                        *resting_qty -= fill_qty;
                        remaining -= fill_qty;

                        if *resting_qty < 1e-15 {
                            level.orders.pop_front();
                        }
                    }
                    if level.orders.is_empty() {
                        self.asks.remove(0);
                    }
                }
                // If limit order with remaining qty, add to book
                if remaining > 1e-15 && order.order_type == OrderType::Limit {
                    self.insert_bid(order.price, order.order_id, order.agent_id, remaining);
                }
            }
            Side::Sell => {
                while remaining > 1e-15 && !self.bids.is_empty() {
                    let can_match = order.order_type == OrderType::Market
                        || order.price <= self.bids[0].price;
                    if !can_match {
                        break;
                    }
                    let level = &mut self.bids[0];
                    while remaining > 1e-15 && !level.orders.is_empty() {
                        let (resting_oid, resting_aid, ref mut resting_qty) =
                            level.orders[0];
                        let fill_qty = remaining.min(*resting_qty);
                        let fill_price = level.price;

                        fills.push(FillEvent::new(
                            order.order_id,
                            order.agent_id,
                            fill_price,
                            fill_qty,
                            timestamp,
                            Side::Sell,
                            order.symbol,
                        ));
                        fills.push(FillEvent::new(
                            resting_oid,
                            resting_aid,
                            fill_price,
                            fill_qty,
                            timestamp,
                            Side::Sell,
                            order.symbol,
                        ));

                        *resting_qty -= fill_qty;
                        remaining -= fill_qty;

                        if *resting_qty < 1e-15 {
                            level.orders.pop_front();
                        }
                    }
                    if level.orders.is_empty() {
                        self.bids.remove(0);
                    }
                }
                if remaining > 1e-15 && order.order_type == OrderType::Limit {
                    self.insert_ask(order.price, order.order_id, order.agent_id, remaining);
                }
            }
        }

        fills
    }

    fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
    }

    /// Remove stale orders older than given max age. For simplicity this
    /// just prunes levels beyond a certain depth.
    fn prune(&mut self, max_levels: usize) {
        if self.bids.len() > max_levels {
            self.bids.truncate(max_levels);
        }
        if self.asks.len() > max_levels {
            self.asks.truncate(max_levels);
        }
    }
}

// ---------------------------------------------------------------------------
// Exchange Simulator (main orchestrator)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct ExchangeSimulator {
    config: SimulationConfig,
    population: AgentPopulation,
    latency: LatencySimulator,
    book: SimpleOrderBook,
    metrics: SimulationMetrics,
    rng: Rng,
    current_time: u64,
    true_value: f64,
    last_trade_price: f64,
    last_trade_qty: f64,
    bar_volume: f64,
    bar_trade_count: u64,
    bar_vwap_num: f64,
    bar_vwap_den: f64,
    bar_high: f64,
    bar_low: f64,
    bar_open: f64,
}

impl ExchangeSimulator {
    pub fn new(config: SimulationConfig) -> Self {
        let population = AgentPopulation::from_config(&config);
        let mut latency = LatencySimulator::new(config.seed.wrapping_add(1));

        // Register latency profiles for all agents
        if config.latency_enabled {
            let mut id: u64 = 1;
            for _ in 0..config.n_market_makers {
                latency.register_agent(LatencyProfile::hft(id));
                id += 1;
            }
            for _ in 0..config.n_noise_traders {
                latency.register_agent(LatencyProfile::retail(id));
                id += 1;
            }
            for _ in 0..config.n_informed {
                latency.register_agent(LatencyProfile::institutional(id));
                id += 1;
            }
            for _ in 0..config.n_momentum {
                latency.register_agent(LatencyProfile::professional(id));
                id += 1;
            }
            for _ in 0..config.n_mean_rev {
                latency.register_agent(LatencyProfile::professional(id));
                id += 1;
            }
        }

        let initial = config.initial_price;
        Self {
            population,
            latency,
            book: SimpleOrderBook::new(config.tick_size),
            metrics: SimulationMetrics::new(),
            rng: Rng::new(config.seed.wrapping_add(2)),
            current_time: 0,
            true_value: initial,
            last_trade_price: initial,
            last_trade_qty: 0.0,
            bar_volume: 0.0,
            bar_trade_count: 0,
            bar_vwap_num: 0.0,
            bar_vwap_den: 0.0,
            bar_high: initial,
            bar_low: initial,
            bar_open: initial,
            config,
        }
    }

    /// Generate the true value for this bar (random walk with drift)
    fn evolve_true_value(&mut self) {
        let drift = self.config.true_value_drift;
        let vol = self.config.volatility;
        let shock = self.rng.normal() * vol;
        self.true_value *= (drift + shock).exp();
        // Floor
        if self.true_value < 0.01 {
            self.true_value = 0.01;
        }
    }

    /// Generate the market update from current book state and true value
    fn generate_market_update(&self) -> MarketUpdate {
        let best_bid = self.book.best_bid().unwrap_or(self.last_trade_price - 0.05);
        let best_ask = self.book.best_ask().unwrap_or(self.last_trade_price + 0.05);
        let vwap = if self.bar_vwap_den > 1e-15 {
            self.bar_vwap_num / self.bar_vwap_den
        } else {
            self.last_trade_price
        };

        MarketUpdate::new(best_bid, best_ask, self.current_time, self.config.symbol)
            .with_last_trade(self.last_trade_price, self.last_trade_qty)
            .with_depth(self.book.bid_depth(), self.book.ask_depth())
            .with_ohlcv(
                self.bar_open,
                self.bar_high,
                self.bar_low,
                self.bar_volume,
                self.bar_trade_count,
            )
            .with_vwap(vwap)
    }

    /// Process fills: update OHLCV, metrics
    fn process_fills(&mut self, fills: &[FillEvent]) {
        for fill in fills {
            self.last_trade_price = fill.fill_price;
            self.last_trade_qty = fill.fill_qty;
            self.bar_volume += fill.fill_qty;
            self.bar_trade_count += 1;
            self.bar_vwap_num += fill.fill_price * fill.fill_qty;
            self.bar_vwap_den += fill.fill_qty;

            if fill.fill_price > self.bar_high {
                self.bar_high = fill.fill_price;
            }
            if fill.fill_price < self.bar_low {
                self.bar_low = fill.fill_price;
            }

            self.metrics.trade_count += 1;
            self.metrics.total_volume += fill.fill_qty;
            self.metrics.total_notional += fill.fill_qty * fill.fill_price;
        }
    }

    /// Run a single simulation bar
    fn step(&mut self) {
        // 1. Evolve true value
        self.evolve_true_value();

        // 2. Update informed traders with true value
        self.population.update_true_value(self.true_value);

        // 3. Generate market update from current state
        let update = self.generate_market_update();

        // 4. Dispatch to all agents
        self.population.dispatch_market_update(&update);

        // 5. Collect orders from all agents
        let orders = self.population.collect_orders(self.current_time);
        self.metrics.order_count += orders.len() as u64;

        // 6. Process orders through latency simulator (or directly)
        let orders_to_match = if self.config.latency_enabled {
            for order in orders {
                self.latency.submit_order(order, self.current_time);
            }
            self.latency.drain_orders(self.current_time)
        } else {
            orders
        };

        // 7. Match orders against the book
        let mut all_fills = Vec::new();
        for order in &orders_to_match {
            let fills = self.book.match_order(order, self.current_time);
            all_fills.extend(fills);
        }

        // 8. Process fills (update OHLCV etc)
        self.process_fills(&all_fills);

        // 9. Dispatch fills to agents (potentially through latency)
        if self.config.latency_enabled {
            for fill in &all_fills {
                self.latency
                    .submit_fill(fill.clone(), self.current_time);
            }
            let delivered_fills = self.latency.drain_fills(self.current_time);
            self.population.dispatch_fills(&delivered_fills);
        } else {
            self.population.dispatch_fills(&all_fills);
        }

        // 10. Record bar metrics
        let spread = update.spread;
        self.metrics.record_bar(
            spread,
            self.last_trade_price,
            self.bar_volume,
            self.true_value,
        );

        // 11. Prune stale book levels
        self.book.prune(50);

        // 12. Reset bar accumulators
        self.bar_open = self.last_trade_price;
        self.bar_high = self.last_trade_price;
        self.bar_low = self.last_trade_price;
        self.bar_volume = 0.0;
        self.bar_trade_count = 0;
        self.bar_vwap_num = 0.0;
        self.bar_vwap_den = 0.0;

        self.current_time += 1;
    }

    /// Run the full simulation
    pub fn run(&mut self) -> &SimulationMetrics {
        // Seed the book with initial quotes from market makers
        let initial_update =
            MarketUpdate::new(
                self.config.initial_price - 0.05,
                self.config.initial_price + 0.05,
                0,
                self.config.symbol,
            )
            .with_last_trade(self.config.initial_price, 0.0);

        self.population.dispatch_market_update(&initial_update);
        let seed_orders = self.population.collect_orders(0);
        for order in &seed_orders {
            self.book.match_order(order, 0);
        }

        // Main simulation loop
        for _ in 0..self.config.simulation_bars {
            self.step();
        }

        // Finalize metrics
        let stats = self.population.aggregate_stats();
        self.metrics.mm_pnl_distribution = stats.mm_pnls;
        self.metrics.informed_pnl = stats.informed_total_pnl;
        self.metrics.noise_pnl = stats.noise_total_pnl;
        self.metrics.momentum_pnl = stats.momentum_total_pnl;
        self.metrics.meanrev_pnl = stats.meanrev_total_pnl;
        self.metrics.compute_herfindahl(&stats.mm_volumes);

        if !stats.mm_inventories.is_empty() {
            self.metrics.avg_mm_inventory =
                stats.mm_inventories.iter().sum::<f64>() / stats.mm_inventories.len() as f64;
            self.metrics.max_mm_inventory = stats
                .mm_inventories
                .iter()
                .cloned()
                .fold(0.0f64, f64::max);
        }

        self.metrics.finalize();
        &self.metrics
    }

    pub fn metrics(&self) -> &SimulationMetrics {
        &self.metrics
    }

    pub fn population(&self) -> &AgentPopulation {
        &self.population
    }

    pub fn reset(&mut self) {
        self.population.reset_all();
        self.book.clear();
        self.metrics = SimulationMetrics::new();
        self.current_time = 0;
        self.true_value = self.config.initial_price;
        self.last_trade_price = self.config.initial_price;
        self.last_trade_qty = 0.0;
        self.bar_volume = 0.0;
        self.bar_trade_count = 0;
        self.bar_vwap_num = 0.0;
        self.bar_vwap_den = 0.0;
        self.bar_high = self.config.initial_price;
        self.bar_low = self.config.initial_price;
        self.bar_open = self.config.initial_price;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_basic() {
        let mut rng = Rng::new(42);
        let u = rng.uniform();
        assert!(u >= 0.0 && u < 1.0);
        let n = rng.normal();
        assert!(n.is_finite());
        let e = rng.exponential(1.0);
        assert!(e >= 0.0);
        let p = rng.poisson(5.0);
        assert!(p < 100);
    }

    #[test]
    fn test_market_update() {
        let mu = MarketUpdate::new(99.95, 100.05, 1, 0);
        assert!((mu.mid_price - 100.0).abs() < 1e-10);
        assert!((mu.spread - 0.10).abs() < 1e-10);
    }

    #[test]
    fn test_market_maker_quotes() {
        let mut mm = MarketMakerAgent::new(1, 0.1, 0.02, 1000.0, 1.5, 100.0, 1.0, 0);
        let update = MarketUpdate::new(99.95, 100.05, 1, 0);
        mm.on_market_update(&update);
        let orders = mm.generate_orders(1);
        assert!(orders.len() >= 2);
        // Should have both a bid and an ask
        let has_bid = orders.iter().any(|o| o.side == Side::Buy);
        let has_ask = orders.iter().any(|o| o.side == Side::Sell);
        assert!(has_bid);
        assert!(has_ask);
    }

    #[test]
    fn test_noise_trader() {
        let mut nt = NoiseTraderAgent::new(1, 0.9, 1.0, 0.5, 0);
        let update = MarketUpdate::new(99.95, 100.05, 1, 0);
        nt.on_market_update(&update);
        // Run multiple times to get at least one order (probabilistic)
        let mut got_order = false;
        for t in 0..100 {
            let orders = nt.generate_orders(t);
            if !orders.is_empty() {
                got_order = true;
                break;
            }
        }
        assert!(got_order);
    }

    #[test]
    fn test_informed_trader_signal() {
        let mut it = InformedTraderAgent::new(1, 105.0, 0.01, 2.0, 50.0, 1.0, 0);
        // Market is at 100, true value is 105 -> 5% deviation -> should buy
        let update = MarketUpdate::new(99.95, 100.05, 1, 0);
        it.on_market_update(&update);
        let orders = it.generate_orders(1);
        assert!(!orders.is_empty());
        assert_eq!(orders[0].side, Side::Buy);
    }

    #[test]
    fn test_simple_order_book() {
        let mut book = SimpleOrderBook::new(0.01);
        book.insert_bid(99.95, 1, 100, 10.0);
        book.insert_ask(100.05, 2, 101, 10.0);
        assert!((book.best_bid().unwrap() - 99.95).abs() < 0.01);
        assert!((book.best_ask().unwrap() - 100.05).abs() < 0.01);

        // Market buy should match against ask
        let buy = OrderFlow::market_order(3, 200, 0, Side::Buy, 5.0, 1);
        let fills = book.match_order(&buy, 1);
        assert_eq!(fills.len(), 2); // aggressor fill + passive fill
        assert!((fills[0].fill_price - 100.05).abs() < 0.01);
    }

    #[test]
    fn test_simulation_runs() {
        let config = SimulationConfig {
            n_market_makers: 2,
            n_noise_traders: 5,
            n_informed: 1,
            n_momentum: 2,
            n_mean_rev: 1,
            simulation_bars: 100,
            ..Default::default()
        };
        let mut sim = ExchangeSimulator::new(config);
        let metrics = sim.run();
        assert!(metrics.bars_processed == 100);
        assert!(metrics.price_series.len() == 100);
    }

    #[test]
    fn test_latency_simulator() {
        let mut lat = LatencySimulator::new(42);
        lat.register_agent(LatencyProfile::retail(1));
        let order = OrderFlow::market_order(1, 1, 0, Side::Buy, 1.0, 100);
        let delivery = lat.submit_order(order, 100);
        assert!(delivery.is_some());
        assert!(delivery.unwrap() > 100); // retail has latency

        // HFT should have near-zero latency
        lat.register_agent(LatencyProfile::hft(2));
        let order2 = OrderFlow::market_order(2, 2, 0, Side::Sell, 1.0, 100);
        let delivery2 = lat.submit_order(order2, 100);
        assert!(delivery2.is_some());
        // HFT delivery should be <= retail delivery
        assert!(delivery2.unwrap() <= delivery.unwrap());
    }

    #[test]
    fn test_population_from_config() {
        let config = SimulationConfig::default();
        let pop = AgentPopulation::from_config(&config);
        assert_eq!(pop.agent_count(), config.total_agents() as u64);
    }

    #[test]
    fn test_avellaneda_stoikov_reservation_price() {
        // With zero inventory, reservation price should equal mid
        let mm = MarketMakerAgent::new(1, 0.1, 0.02, 1000.0, 1.5, 100.0, 1.0, 0);
        let rp = mm.compute_reservation_price();
        assert!((rp - mm.current_mid).abs() < 1e-6);
    }

    #[test]
    fn test_mean_reversion_regime() {
        let mut mr = MeanReversionTrader::new(1, 20, 2.0, 1.0, 10.0, 0);
        // Feed constant prices -> should remain NormalVol or LowVol (no variance)
        for i in 0..50 {
            let update = MarketUpdate::new(99.99, 100.01, i, 0);
            mr.on_market_update(&update);
        }
        // With near-zero price variation, regime should not be Trending or HighVol
        assert!(mr.regime != MeanReversionRegime::Trending);
        assert!(mr.regime != MeanReversionRegime::HighVol);
    }
}
