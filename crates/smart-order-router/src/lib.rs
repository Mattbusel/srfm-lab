pub mod venue;
pub mod router;
pub mod simulator;
pub mod cost_model;
pub mod optimizer;

pub use venue::{Venue, VenueId, VenueRegistry, ExchangeType};
pub use router::{SmartOrderRouter, RoutingStrategy, ChildOrder, RoutingResult};
pub use simulator::{VenueSimulator, SimFill, SimulationConfig};
pub use cost_model::{CostModel, TradeCost, CostBreakdown};
pub use optimizer::{VenueAllocator, AllocationConstraints, AllocationResult};

#[derive(Debug, thiserror::Error)]
pub enum SorError {
    #[error("No venues available")]
    NoVenues,
    #[error("Venue not found: {0}")]
    VenueNotFound(String),
    #[error("Insufficient liquidity: need {needed:.0} have {available:.0}")]
    InsufficientLiquidity { needed: f64, available: f64 },
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("Optimization failure: {0}")]
    OptimizationFailure(String),
}

/// Side of an order
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OrderSide {
    Buy,
    Sell,
}

impl OrderSide {
    pub fn sign(self) -> f64 {
        match self { OrderSide::Buy => 1.0, OrderSide::Sell => -1.0 }
    }
}

/// Parent order to be routed
#[derive(Debug, Clone)]
pub struct ParentOrder {
    pub id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub limit_price: Option<f64>,
    pub urgency: f64,  // 0.0 = patient, 1.0 = aggressive
}

impl ParentOrder {
    pub fn new(id: &str, symbol: &str, side: OrderSide, quantity: f64) -> Self {
        ParentOrder {
            id: id.to_string(),
            symbol: symbol.to_string(),
            side,
            quantity,
            limit_price: None,
            urgency: 0.5,
        }
    }

    pub fn with_limit(mut self, price: f64) -> Self {
        self.limit_price = Some(price);
        self
    }

    pub fn with_urgency(mut self, urgency: f64) -> Self {
        self.urgency = urgency.max(0.0).min(1.0);
        self
    }
}
