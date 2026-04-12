// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// execution_engine.rs — Order execution and routing logic
// =============================================================================
//! Provides:
//! - `Order` struct and `OrderStatus` lifecycle
//! - `ExecutionEngine` managing pending/filled orders
//! - Slippage/market impact simulation
//! - Position tracking and PnL computation
//! - Fill statistics and execution quality metrics

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::now_ns;

// ---------------------------------------------------------------------------
// Order types and enums
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderSide {
    Buy,
    Sell,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderType {
    Market,
    Limit,
    StopMarket,
    StopLimit,
    TrailingStop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderStatus {
    Pending,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
    Expired,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    GoodTillCancel,
    Day,
    ImmediateOrCancel,
    FillOrKill,
}

// ---------------------------------------------------------------------------
// Order
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Order {
    pub order_id:       u64,
    pub asset_id:       u32,
    pub side:           OrderSide,
    pub order_type:     OrderType,
    pub tif:            TimeInForce,
    pub quantity:       f64,
    pub limit_price:    Option<f64>,
    pub stop_price:     Option<f64>,
    pub filled_qty:     f64,
    pub avg_fill_price: f64,
    pub status:         OrderStatus,
    pub submitted_ns:   u64,
    pub updated_ns:     u64,
    pub client_tag:     String,
}

impl Order {
    pub fn market(asset_id: u32, side: OrderSide, quantity: f64) -> Self {
        let now = now_ns();
        Self {
            order_id:       0,
            asset_id,
            side,
            order_type:     OrderType::Market,
            tif:            TimeInForce::ImmediateOrCancel,
            quantity,
            limit_price:    None,
            stop_price:     None,
            filled_qty:     0.0,
            avg_fill_price: 0.0,
            status:         OrderStatus::Pending,
            submitted_ns:   now,
            updated_ns:     now,
            client_tag:     String::new(),
        }
    }

    pub fn limit(asset_id: u32, side: OrderSide, quantity: f64, price: f64) -> Self {
        let mut o = Self::market(asset_id, side, quantity);
        o.order_type  = OrderType::Limit;
        o.limit_price = Some(price);
        o.tif         = TimeInForce::GoodTillCancel;
        o
    }

    pub fn remaining_qty(&self) -> f64 {
        self.quantity - self.filled_qty
    }

    pub fn is_active(&self) -> bool {
        matches!(self.status, OrderStatus::Pending | OrderStatus::PartiallyFilled)
    }

    pub fn notional_value(&self) -> f64 {
        self.avg_fill_price * self.filled_qty
    }
}

// ---------------------------------------------------------------------------
// Fill
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Fill {
    pub order_id:   u64,
    pub asset_id:   u32,
    pub side:       OrderSide,
    pub quantity:   f64,
    pub price:      f64,
    pub commission: f64,
    pub timestamp:  u64,
    pub is_maker:   bool,
}

impl Fill {
    pub fn net_amount(&self) -> f64 {
        let gross = self.price * self.quantity;
        match self.side {
            OrderSide::Buy  => -(gross + self.commission),
            OrderSide::Sell =>  (gross - self.commission),
        }
    }
}

// ---------------------------------------------------------------------------
// SlippageModel
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SlippageConfig {
    pub spread_half_bps:   f64,
    pub impact_bps_per_pct_adv: f64,
    pub taker_fee_bps:     f64,
    pub maker_fee_bps:     f64,
    pub min_tick:          f64,
}

impl Default for SlippageConfig {
    fn default() -> Self {
        Self {
            spread_half_bps:   2.0,
            impact_bps_per_pct_adv: 20.0,
            taker_fee_bps:     5.0,
            maker_fee_bps:     2.0,
            min_tick:          0.01,
        }
    }
}

pub struct SlippageModel {
    config: SlippageConfig,
}

impl SlippageModel {
    pub fn new(config: SlippageConfig) -> Self {
        Self { config }
    }

    pub fn simulate_fill(
        &self,
        order: &Order,
        mid_price: f64,
        adv: f64,
    ) -> Fill {
        let impact_frac = if adv > 1e-10 {
            order.quantity * mid_price / adv * self.config.impact_bps_per_pct_adv / 1e4
        } else {
            0.0
        };

        let spread_frac = self.config.spread_half_bps / 1e4;
        let is_maker = matches!(order.order_type, OrderType::Limit);

        let fill_price = match order.side {
            OrderSide::Buy  => mid_price * (1.0 + spread_frac + impact_frac),
            OrderSide::Sell => mid_price * (1.0 - spread_frac - impact_frac),
        };

        // Round to tick
        let tick = self.config.min_tick;
        let fill_price = if tick > 1e-15 {
            (fill_price / tick).round() * tick
        } else {
            fill_price
        };

        let fee_bps = if is_maker {
            self.config.maker_fee_bps
        } else {
            self.config.taker_fee_bps
        };
        let commission = fill_price * order.quantity * fee_bps / 1e4;

        Fill {
            order_id:   order.order_id,
            asset_id:   order.asset_id,
            side:       order.side,
            quantity:   order.quantity,
            price:      fill_price,
            commission,
            timestamp:  now_ns(),
            is_maker,
        }
    }
}

// ---------------------------------------------------------------------------
// Position
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default)]
pub struct Position {
    pub asset_id:       u32,
    pub quantity:       f64,    // positive = long, negative = short
    pub avg_cost:       f64,    // average cost basis
    pub realized_pnl:  f64,
    pub total_commission: f64,
    pub n_fills:        u64,
}

impl Position {
    pub fn apply_fill(&mut self, fill: &Fill) {
        let qty_signed = match fill.side {
            OrderSide::Buy  =>  fill.quantity,
            OrderSide::Sell => -fill.quantity,
        };

        if self.quantity == 0.0 {
            // Opening new position
            self.avg_cost = fill.price;
            self.quantity = qty_signed;
        } else if (self.quantity > 0.0 && qty_signed > 0.0) ||
                  (self.quantity < 0.0 && qty_signed < 0.0) {
            // Adding to position
            let total_cost = self.avg_cost * self.quantity.abs()
                + fill.price * fill.quantity;
            self.quantity += qty_signed;
            self.avg_cost = if self.quantity.abs() > 1e-12 {
                total_cost / self.quantity.abs()
            } else {
                0.0
            };
        } else {
            // Reducing or flipping position
            let close_qty = qty_signed.abs().min(self.quantity.abs());
            let pnl_sign  = if self.quantity > 0.0 { 1.0 } else { -1.0 };
            self.realized_pnl += pnl_sign * close_qty * (fill.price - self.avg_cost);

            let remaining = self.quantity + qty_signed;
            if remaining.abs() < 1e-12 {
                self.quantity = 0.0;
                self.avg_cost = 0.0;
            } else if remaining.signum() != self.quantity.signum() {
                // Flip
                self.quantity = remaining;
                self.avg_cost = fill.price;
            } else {
                self.quantity = remaining;
            }
        }

        self.total_commission += fill.commission;
        self.n_fills          += 1;
    }

    pub fn unrealized_pnl(&self, current_price: f64) -> f64 {
        if self.quantity.abs() < 1e-12 { return 0.0; }
        let sign = if self.quantity > 0.0 { 1.0 } else { -1.0 };
        sign * self.quantity.abs() * (current_price - self.avg_cost)
    }

    pub fn total_pnl(&self, current_price: f64) -> f64 {
        self.realized_pnl + self.unrealized_pnl(current_price) - self.total_commission
    }

    pub fn market_value(&self, current_price: f64) -> f64 {
        self.quantity * current_price
    }
}

// ---------------------------------------------------------------------------
// Execution statistics
// ---------------------------------------------------------------------------

#[derive(Debug, Default)]
pub struct ExecutionStats {
    pub orders_submitted:  u64,
    pub orders_filled:     u64,
    pub orders_cancelled:  u64,
    pub orders_rejected:   u64,
    pub total_fills:       u64,
    pub total_notional:    f64,
    pub total_commissions: f64,
    pub total_slippage:    f64,
    pub mean_fill_latency_ns: f64,
    pub n_latency_samples: u64,
}

impl ExecutionStats {
    pub fn record_fill(&mut self, fill: &Fill, expected_price: f64, latency_ns: u64) {
        self.total_fills       += 1;
        self.total_notional    += fill.price * fill.quantity;
        self.total_commissions += fill.commission;
        let slippage = (fill.price - expected_price).abs() * fill.quantity;
        self.total_slippage    += slippage;

        // Running mean of fill latency
        let n = self.n_latency_samples as f64;
        self.mean_fill_latency_ns = (self.mean_fill_latency_ns * n + latency_ns as f64) / (n + 1.0);
        self.n_latency_samples    += 1;
    }

    pub fn fill_rate(&self) -> f64 {
        if self.orders_submitted == 0 { return 0.0; }
        self.orders_filled as f64 / self.orders_submitted as f64
    }

    pub fn slippage_bps(&self) -> f64 {
        if self.total_notional < 1e-10 { return 0.0; }
        self.total_slippage / self.total_notional * 1e4
    }

    pub fn commission_bps(&self) -> f64 {
        if self.total_notional < 1e-10 { return 0.0; }
        self.total_commissions / self.total_notional * 1e4
    }
}

// ---------------------------------------------------------------------------
// ExecutionEngine
// ---------------------------------------------------------------------------

static ORDER_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

pub struct ExecutionEngine {
    pending_orders:  Mutex<HashMap<u64, Order>>,
    filled_orders:   Mutex<Vec<Order>>,
    fills:           Mutex<Vec<Fill>>,
    positions:       Mutex<HashMap<u32, Position>>,
    stats:           Mutex<ExecutionStats>,
    slippage_model:  SlippageModel,
}

impl ExecutionEngine {
    pub fn new(slippage_config: SlippageConfig) -> Arc<Self> {
        Arc::new(Self {
            pending_orders: Mutex::new(HashMap::new()),
            filled_orders:  Mutex::new(Vec::new()),
            fills:          Mutex::new(Vec::new()),
            positions:      Mutex::new(HashMap::new()),
            stats:          Mutex::new(ExecutionStats::default()),
            slippage_model: SlippageModel::new(slippage_config),
        })
    }

    /// Submit a new order; returns assigned order_id
    pub fn submit(&self, mut order: Order) -> u64 {
        let id = ORDER_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        order.order_id = id;
        order.status   = OrderStatus::Pending;
        order.submitted_ns = now_ns();

        let mut pending = self.pending_orders.lock().unwrap();
        pending.insert(id, order);

        let mut stats = self.stats.lock().unwrap();
        stats.orders_submitted += 1;

        id
    }

    /// Simulate immediate market execution
    pub fn execute_market(
        &self,
        order_id: u64,
        mid_price: f64,
        adv: f64,
    ) -> Option<Fill> {
        let order = {
            let mut pending = self.pending_orders.lock().unwrap();
            pending.remove(&order_id)?
        };

        let fill = self.slippage_model.simulate_fill(&order, mid_price, adv);
        let latency_ns = now_ns().saturating_sub(order.submitted_ns);

        // Update position
        {
            let mut positions = self.positions.lock().unwrap();
            let pos = positions.entry(order.asset_id).or_insert_with(|| {
                Position { asset_id: order.asset_id, ..Default::default() }
            });
            pos.apply_fill(&fill);
        }

        // Update stats
        {
            let mut stats = self.stats.lock().unwrap();
            stats.orders_filled += 1;
            stats.record_fill(&fill, mid_price, latency_ns);
        }

        // Archive order
        let mut filled_order = order;
        filled_order.filled_qty     = fill.quantity;
        filled_order.avg_fill_price = fill.price;
        filled_order.status         = OrderStatus::Filled;
        filled_order.updated_ns     = now_ns();
        self.filled_orders.lock().unwrap().push(filled_order);

        let fill_clone = fill.clone();
        self.fills.lock().unwrap().push(fill);
        Some(fill_clone)
    }

    /// Cancel a pending order
    pub fn cancel(&self, order_id: u64) -> bool {
        let mut pending = self.pending_orders.lock().unwrap();
        if let Some(mut order) = pending.remove(&order_id) {
            order.status = OrderStatus::Cancelled;
            order.updated_ns = now_ns();
            self.filled_orders.lock().unwrap().push(order);
            self.stats.lock().unwrap().orders_cancelled += 1;
            return true;
        }
        false
    }

    /// Get position for asset
    pub fn position(&self, asset_id: u32) -> Option<Position> {
        self.positions.lock().unwrap().get(&asset_id).cloned()
    }

    /// Get all positions
    pub fn all_positions(&self) -> Vec<Position> {
        self.positions.lock().unwrap().values().cloned().collect()
    }

    /// Total portfolio PnL given current prices
    pub fn portfolio_pnl(&self, prices: &HashMap<u32, f64>) -> f64 {
        self.positions.lock().unwrap().iter().map(|(aid, pos)| {
            prices.get(aid).map(|&p| pos.total_pnl(p)).unwrap_or(pos.realized_pnl)
        }).sum()
    }

    /// Gross notional exposure
    pub fn gross_exposure(&self, prices: &HashMap<u32, f64>) -> f64 {
        self.positions.lock().unwrap().iter().map(|(aid, pos)| {
            let price = prices.get(aid).copied().unwrap_or(pos.avg_cost);
            (pos.quantity * price).abs()
        }).sum()
    }

    /// Number of pending orders
    pub fn n_pending(&self) -> usize {
        self.pending_orders.lock().unwrap().len()
    }

    /// Recent fills
    pub fn recent_fills(&self, n: usize) -> Vec<Fill> {
        let fills = self.fills.lock().unwrap();
        fills.iter().rev().take(n).cloned().collect()
    }

    /// Execution statistics snapshot
    pub fn stats_snapshot(&self) -> ExecutionStats {
        let s = self.stats.lock().unwrap();
        ExecutionStats {
            orders_submitted:  s.orders_submitted,
            orders_filled:     s.orders_filled,
            orders_cancelled:  s.orders_cancelled,
            orders_rejected:   s.orders_rejected,
            total_fills:       s.total_fills,
            total_notional:    s.total_notional,
            total_commissions: s.total_commissions,
            total_slippage:    s.total_slippage,
            mean_fill_latency_ns: s.mean_fill_latency_ns,
            n_latency_samples: s.n_latency_samples,
        }
    }

    /// Prometheus metrics
    pub fn prometheus_metrics(&self) -> String {
        let s = self.stats.lock().unwrap();
        format!(
            "rtel_exec_orders_submitted {}\n\
             rtel_exec_orders_filled {}\n\
             rtel_exec_orders_cancelled {}\n\
             rtel_exec_fill_rate {:.4}\n\
             rtel_exec_total_notional {:.2}\n\
             rtel_exec_slippage_bps {:.4}\n\
             rtel_exec_commission_bps {:.4}\n\
             rtel_exec_mean_fill_latency_ns {:.1}\n",
            s.orders_submitted,
            s.orders_filled,
            s.orders_cancelled,
            s.fill_rate(),
            s.total_notional,
            s.slippage_bps(),
            s.commission_bps(),
            s.mean_fill_latency_ns,
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> Arc<ExecutionEngine> {
        ExecutionEngine::new(SlippageConfig::default())
    }

    #[test]
    fn test_market_order_fill() {
        let eng = make_engine();
        let order = Order::market(0, OrderSide::Buy, 100.0);
        let id = eng.submit(order);
        let fill = eng.execute_market(id, 150.0, 1_000_000.0).unwrap();
        assert!(fill.price > 150.0);  // slippage on buy
        assert_eq!(fill.quantity, 100.0);
        let stats = eng.stats_snapshot();
        assert_eq!(stats.orders_submitted, 1);
        assert_eq!(stats.orders_filled,    1);
        assert_eq!(stats.total_fills,      1);
    }

    #[test]
    fn test_position_tracking() {
        let eng = make_engine();
        // Buy 100
        let id = eng.submit(Order::market(1, OrderSide::Buy, 100.0));
        eng.execute_market(id, 100.0, 1e6).unwrap();
        // Buy 50 more
        let id2 = eng.submit(Order::market(1, OrderSide::Buy, 50.0));
        eng.execute_market(id2, 105.0, 1e6).unwrap();

        let pos = eng.position(1).unwrap();
        assert!(pos.quantity > 140.0);  // ~150 minus tiny slippage fill
        assert!(pos.avg_cost > 100.0 && pos.avg_cost < 110.0);
    }

    #[test]
    fn test_cancel_order() {
        let eng = make_engine();
        let order = Order::limit(0, OrderSide::Buy, 100.0, 95.0);
        let id = eng.submit(order);
        assert_eq!(eng.n_pending(), 1);
        assert!(eng.cancel(id));
        assert_eq!(eng.n_pending(), 0);
        let stats = eng.stats_snapshot();
        assert_eq!(stats.orders_cancelled, 1);
    }

    #[test]
    fn test_pnl_long_position() {
        let eng = make_engine();
        let id = eng.submit(Order::market(2, OrderSide::Buy, 1.0));
        eng.execute_market(id, 100.0, 1e6).unwrap();

        let mut prices = HashMap::new();
        prices.insert(2u32, 110.0f64);
        let pnl = eng.portfolio_pnl(&prices);
        assert!(pnl > 0.0, "long position with price up should have positive pnl");
    }

    #[test]
    fn test_slippage_model() {
        let model = SlippageModel::new(SlippageConfig::default());
        let buy_order = Order::market(0, OrderSide::Buy, 100.0);
        let sell_order = Order::market(0, OrderSide::Sell, 100.0);

        let buy_fill  = model.simulate_fill(&buy_order, 100.0, 1e6);
        let sell_fill = model.simulate_fill(&sell_order, 100.0, 1e6);

        assert!(buy_fill.price > 100.0,  "buy should have positive slippage");
        assert!(sell_fill.price < 100.0, "sell should have negative slippage");
        assert!(buy_fill.commission > 0.0);
    }

    #[test]
    fn test_position_flip() {
        let mut pos = Position { asset_id: 0, ..Default::default() };
        // Open long 100
        pos.apply_fill(&Fill {
            order_id: 1, asset_id: 0, side: OrderSide::Buy,
            quantity: 100.0, price: 100.0, commission: 0.0,
            timestamp: 0, is_maker: false,
        });
        assert!((pos.quantity - 100.0).abs() < 1e-9);
        assert_eq!(pos.avg_cost, 100.0);

        // Sell 150 (flip to short 50)
        pos.apply_fill(&Fill {
            order_id: 2, asset_id: 0, side: OrderSide::Sell,
            quantity: 150.0, price: 110.0, commission: 0.0,
            timestamp: 0, is_maker: false,
        });
        assert!((pos.quantity + 50.0).abs() < 1e-9, "should be short 50");
        assert!((pos.realized_pnl - 1000.0).abs() < 1e-6, "100 units × $10 gain");
    }
}
