use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use crate::order::OrderSide;

/// A single execution / fill event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Fill {
    /// ID of the order that was filled.
    pub order_id: u64,
    /// Counterparty order ID (the resting order that was matched against).
    pub passive_order_id: u64,
    /// Execution price.
    pub price: f64,
    /// Quantity executed in this fill.
    pub qty: f64,
    /// Timestamp of the fill.
    pub timestamp: DateTime<Utc>,
    /// Which side was the aggressor (the incoming order).
    pub aggressor_side: OrderSide,
}

impl Fill {
    pub fn new(
        order_id: u64,
        passive_order_id: u64,
        price: f64,
        qty: f64,
        timestamp: DateTime<Utc>,
        aggressor_side: OrderSide,
    ) -> Self {
        Fill {
            order_id,
            passive_order_id,
            price,
            qty,
            timestamp,
            aggressor_side,
        }
    }

    /// Dollar value of this fill.
    pub fn notional(&self) -> f64 {
        self.price * self.qty
    }
}

/// A collection of fills for summary statistics.
#[derive(Debug, Default)]
pub struct FillSummary {
    pub fills: Vec<Fill>,
}

impl FillSummary {
    pub fn new() -> Self {
        FillSummary { fills: Vec::new() }
    }

    pub fn add(&mut self, fill: Fill) {
        self.fills.push(fill);
    }

    /// Volume-weighted average fill price.
    pub fn vwap(&self) -> f64 {
        let total_qty: f64 = self.fills.iter().map(|f| f.qty).sum();
        if total_qty == 0.0 {
            return 0.0;
        }
        let total_notional: f64 = self.fills.iter().map(|f| f.notional()).sum();
        total_notional / total_qty
    }

    /// Total quantity filled.
    pub fn total_qty(&self) -> f64 {
        self.fills.iter().map(|f| f.qty).sum()
    }

    /// Total notional filled.
    pub fn total_notional(&self) -> f64 {
        self.fills.iter().map(|f| f.notional()).sum()
    }

    /// Number of partial fills.
    pub fn count(&self) -> usize {
        self.fills.len()
    }

    /// Highest fill price.
    pub fn max_price(&self) -> f64 {
        self.fills.iter().map(|f| f.price).fold(f64::NEG_INFINITY, f64::max)
    }

    /// Lowest fill price.
    pub fn min_price(&self) -> f64 {
        self.fills.iter().map(|f| f.price).fold(f64::INFINITY, f64::min)
    }
}
