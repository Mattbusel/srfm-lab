use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Side of an order: buy or sell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl OrderSide {
    pub fn opposite(&self) -> OrderSide {
        match self {
            OrderSide::Buy => OrderSide::Sell,
            OrderSide::Sell => OrderSide::Buy,
        }
    }
}

/// The type of an order.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OrderType {
    /// Limit order — rests at a specified price.
    Limit,
    /// Market order — matches immediately at best available price.
    Market,
    /// Stop-limit order — becomes a limit order when stop is triggered.
    StopLimit { stop_price: f64, limit_price: f64 },
}

/// Lifecycle status of an order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    New,
    PartialFill,
    Filled,
    Cancelled,
    Rejected,
}

/// A single resting or active order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    pub id: u64,
    pub side: OrderSide,
    /// Limit price (for Market orders this is ignored during matching but stored as 0.0).
    pub price: f64,
    /// Original quantity submitted.
    pub qty: f64,
    /// Remaining quantity (unfilled).
    pub remaining_qty: f64,
    pub order_type: OrderType,
    pub timestamp: DateTime<Utc>,
    pub status: OrderStatus,
    /// Sequence number used for time priority within a price level.
    pub sequence: u64,
}

impl Order {
    pub fn new_limit(id: u64, side: OrderSide, price: f64, qty: f64, timestamp: DateTime<Utc>, sequence: u64) -> Self {
        Order {
            id,
            side,
            price,
            qty,
            remaining_qty: qty,
            order_type: OrderType::Limit,
            timestamp,
            status: OrderStatus::New,
            sequence,
        }
    }

    pub fn new_market(id: u64, side: OrderSide, qty: f64, timestamp: DateTime<Utc>, sequence: u64) -> Self {
        Order {
            id,
            side,
            price: 0.0,
            qty,
            remaining_qty: qty,
            order_type: OrderType::Market,
            timestamp,
            status: OrderStatus::New,
            sequence,
        }
    }

    pub fn new_stop_limit(
        id: u64,
        side: OrderSide,
        stop_price: f64,
        limit_price: f64,
        qty: f64,
        timestamp: DateTime<Utc>,
        sequence: u64,
    ) -> Self {
        Order {
            id,
            side,
            price: limit_price,
            qty,
            remaining_qty: qty,
            order_type: OrderType::StopLimit { stop_price, limit_price },
            timestamp,
            status: OrderStatus::New,
            sequence,
        }
    }

    /// Whether the order has been fully consumed.
    pub fn is_complete(&self) -> bool {
        matches!(self.status, OrderStatus::Filled | OrderStatus::Cancelled | OrderStatus::Rejected)
    }

    /// Fill `filled_qty` from this order, updating status accordingly.
    pub fn apply_fill(&mut self, filled_qty: f64) {
        self.remaining_qty -= filled_qty;
        if self.remaining_qty <= 0.0 {
            self.remaining_qty = 0.0;
            self.status = OrderStatus::Filled;
        } else {
            self.status = OrderStatus::PartialFill;
        }
    }
}
