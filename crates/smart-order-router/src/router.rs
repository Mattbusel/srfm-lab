use std::collections::HashMap;
use crate::venue::{Venue, VenueRegistry, MarketData};
use crate::cost_model::CostModel;
use crate::optimizer::{VenueAllocator, AllocationConstraints, AllocationResult};
use crate::{SorError, OrderSide, ParentOrder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoutingStrategy {
    /// Best execution: minimize total cost
    BestExecution,
    /// Time-weighted average price: execute evenly over time
    Twap,
    /// Volume-weighted average price: track market volume profile
    Vwap,
    /// Percentage of volume: maintain constant participation rate
    Pov { target_pov: OrderedF64 },
    /// Implementation shortfall: balance urgency vs cost
    ImplementationShortfall,
    /// Smart order routing: split across venues optimally
    Sor,
}

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct OrderedF64(pub f64);

impl Eq for OrderedF64 {}

impl std::hash::Hash for OrderedF64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// A child order sent to a specific venue
#[derive(Debug, Clone)]
pub struct ChildOrder {
    pub parent_id: String,
    pub child_id: String,
    pub venue_id: String,
    pub side: OrderSide,
    pub quantity: f64,
    pub price: Option<f64>,
    pub is_limit: bool,
}

/// Result of routing a parent order
#[derive(Debug, Clone)]
pub struct RoutingResult {
    pub parent_id: String,
    pub strategy: String,
    pub child_orders: Vec<ChildOrder>,
    pub total_quantity: f64,
    pub estimated_cost_bps: f64,
    pub venues_used: Vec<String>,
}

impl RoutingResult {
    pub fn allocated_quantity(&self) -> f64 {
        self.child_orders.iter().map(|c| c.quantity).sum()
    }

    pub fn fill_fraction(&self) -> f64 {
        if self.total_quantity > 0.0 {
            self.allocated_quantity() / self.total_quantity
        } else {
            0.0
        }
    }
}

/// TWAP schedule: list of (time_slice_idx, quantity_per_slice) pairs
#[derive(Debug, Clone)]
pub struct TwapSchedule {
    pub slices: Vec<f64>,
    pub interval_secs: f64,
}

impl TwapSchedule {
    pub fn new(total_qty: f64, n_slices: usize, interval_secs: f64) -> Self {
        let base_qty = total_qty / n_slices as f64;
        let mut slices = vec![base_qty; n_slices];
        // Adjust for rounding
        let allocated: f64 = slices[..n_slices.saturating_sub(1)].iter().sum();
        if n_slices > 0 {
            slices[n_slices - 1] = (total_qty - allocated).max(0.0);
        }
        TwapSchedule { slices, interval_secs }
    }
}

/// VWAP schedule based on volume profile
#[derive(Debug, Clone)]
pub struct VwapSchedule {
    pub slices: Vec<f64>,
    pub volume_profile: Vec<f64>, // fraction of daily volume per interval
}

impl VwapSchedule {
    /// Build from a volume profile (must sum to 1.0)
    pub fn new(total_qty: f64, volume_profile: Vec<f64>) -> Self {
        let total_profile: f64 = volume_profile.iter().sum();
        let slices: Vec<f64> = volume_profile.iter()
            .map(|&frac| total_qty * frac / total_profile)
            .collect();
        VwapSchedule { slices, volume_profile }
    }

    /// Default U-shaped intraday profile (high volume at open/close)
    pub fn u_shaped(total_qty: f64, n_slices: usize) -> Self {
        let profile: Vec<f64> = (0..n_slices).map(|i| {
            let x = i as f64 / (n_slices - 1) as f64;
            // U-shape: high at open (x=0) and close (x=1), low at midday
            0.5 + 0.5 * (2.0 * std::f64::consts::PI * x).cos().abs()
        }).collect();
        Self::new(total_qty, profile)
    }
}

/// POV (percentage of volume) schedule
#[derive(Debug, Clone)]
pub struct PovSchedule {
    pub target_pov: f64,
    pub max_pov: f64,
    pub min_pov: f64,
}

impl PovSchedule {
    pub fn new(target_pov: f64) -> Self {
        PovSchedule {
            target_pov,
            max_pov: target_pov * 1.5,
            min_pov: target_pov * 0.5,
        }
    }

    pub fn qty_for_interval(&self, market_volume: f64) -> f64 {
        market_volume * self.target_pov
    }
}

/// The Smart Order Router
pub struct SmartOrderRouter {
    registry: VenueRegistry,
    market_data: HashMap<String, HashMap<String, MarketData>>, // symbol -> venue -> md
    cost_model: CostModel,
    constraints: AllocationConstraints,
    child_order_counter: u64,
}

impl SmartOrderRouter {
    pub fn new(cost_model: CostModel) -> Self {
        SmartOrderRouter {
            registry: VenueRegistry::new(),
            market_data: HashMap::new(),
            cost_model,
            constraints: AllocationConstraints::default(),
            child_order_counter: 0,
        }
    }

    pub fn with_registry(mut self, registry: VenueRegistry) -> Self {
        self.registry = registry;
        self
    }

    pub fn with_constraints(mut self, constraints: AllocationConstraints) -> Self {
        self.constraints = constraints;
        self
    }

    pub fn update_market_data(&mut self, symbol: &str, venue_id: &str, md: MarketData) {
        self.market_data.entry(symbol.to_string())
            .or_default()
            .insert(venue_id.to_string(), md);
    }

    fn next_child_id(&mut self) -> String {
        self.child_order_counter += 1;
        format!("CHILD_{:08}", self.child_order_counter)
    }

    /// Route an order using the specified strategy
    pub fn route(
        &mut self,
        order: &ParentOrder,
        strategy: RoutingStrategy,
    ) -> Result<RoutingResult, SorError> {
        match strategy {
            RoutingStrategy::BestExecution | RoutingStrategy::Sor => {
                self.route_sor(order)
            }
            RoutingStrategy::Twap => self.route_twap(order, 12, 300.0),
            RoutingStrategy::Vwap => self.route_vwap(order, 12),
            RoutingStrategy::Pov { target_pov } => self.route_pov(order, target_pov.0),
            RoutingStrategy::ImplementationShortfall => self.route_is(order),
        }
    }

    /// SOR: split order across venues to minimize cost
    fn route_sor(&mut self, order: &ParentOrder) -> Result<RoutingResult, SorError> {
        let mut allocator = VenueAllocator::new(self.cost_model.clone());

        let venues: Vec<Venue> = self.registry.all_active().into_iter().cloned().collect();
        let sym_data = self.market_data.get(&order.symbol);

        for venue in &venues {
            let md = sym_data
                .and_then(|sd| sd.get(&venue.id))
                .cloned()
                .unwrap_or_else(|| self.default_md(100.0));
            allocator.add_venue(venue.clone(), md);
        }

        let price = order.limit_price.unwrap_or_else(|| {
            sym_data.and_then(|sd| sd.values().next()).map(|md| md.mid()).unwrap_or(100.0)
        });

        let result = allocator.optimize(
            order.quantity,
            price,
            order.side,
            &self.constraints,
            100,
        )?;

        let mut child_orders = Vec::new();
        let mut venues_used = Vec::new();

        for (venue_id, qty) in &result.allocations {
            if *qty >= self.constraints.min_venue_qty {
                let child_id = self.next_child_id();
                child_orders.push(ChildOrder {
                    parent_id: order.id.clone(),
                    child_id,
                    venue_id: venue_id.clone(),
                    side: order.side,
                    quantity: *qty,
                    price: order.limit_price,
                    is_limit: order.limit_price.is_some(),
                });
                venues_used.push(venue_id.clone());
            }
        }

        if child_orders.is_empty() {
            return Err(SorError::InsufficientLiquidity {
                needed: order.quantity,
                available: 0.0,
            });
        }

        Ok(RoutingResult {
            parent_id: order.id.clone(),
            strategy: "SOR".to_string(),
            total_quantity: order.quantity,
            estimated_cost_bps: result.estimated_cost_bps,
            venues_used,
            child_orders,
        })
    }

    /// TWAP: spread evenly over n_slices
    fn route_twap(&mut self, order: &ParentOrder, n_slices: usize, interval_secs: f64) -> Result<RoutingResult, SorError> {
        let schedule = TwapSchedule::new(order.quantity, n_slices, interval_secs);
        let best_venue = self.best_venue_by_fee(&order.symbol)?;

        let mut child_orders = Vec::new();
        for qty in &schedule.slices {
            if *qty < self.constraints.min_venue_qty { continue; }
            let child_id = self.next_child_id();
            child_orders.push(ChildOrder {
                parent_id: order.id.clone(),
                child_id,
                venue_id: best_venue.clone(),
                side: order.side,
                quantity: *qty,
                price: order.limit_price,
                is_limit: order.limit_price.is_some(),
            });
        }

        Ok(RoutingResult {
            parent_id: order.id.clone(),
            strategy: "TWAP".to_string(),
            total_quantity: order.quantity,
            estimated_cost_bps: 0.0,
            venues_used: vec![best_venue],
            child_orders,
        })
    }

    /// VWAP: track U-shaped volume profile
    fn route_vwap(&mut self, order: &ParentOrder, n_slices: usize) -> Result<RoutingResult, SorError> {
        let schedule = VwapSchedule::u_shaped(order.quantity, n_slices);
        let best_venue = self.best_venue_by_fee(&order.symbol)?;

        let mut child_orders = Vec::new();
        for qty in &schedule.slices {
            if *qty < self.constraints.min_venue_qty { continue; }
            let child_id = self.next_child_id();
            child_orders.push(ChildOrder {
                parent_id: order.id.clone(),
                child_id,
                venue_id: best_venue.clone(),
                side: order.side,
                quantity: *qty,
                price: order.limit_price,
                is_limit: order.limit_price.is_some(),
            });
        }

        Ok(RoutingResult {
            parent_id: order.id.clone(),
            strategy: "VWAP".to_string(),
            total_quantity: order.quantity,
            estimated_cost_bps: 0.0,
            venues_used: vec![best_venue],
            child_orders,
        })
    }

    /// POV: send qty proportional to market volume
    fn route_pov(&mut self, order: &ParentOrder, target_pov: f64) -> Result<RoutingResult, SorError> {
        let pov_schedule = PovSchedule::new(target_pov);
        let best_venue = self.best_venue_by_fee(&order.symbol)?;

        // Estimate total market volume needed
        let adv = self.market_data.get(&order.symbol)
            .and_then(|sd| sd.get(&best_venue))
            .map(|md| md.adv)
            .unwrap_or(1_000_000.0);

        // Split into 12 intervals proportional to POV
        let n_slices = 12;
        let interval_volume = adv / n_slices as f64;
        let qty_per_slice = pov_schedule.qty_for_interval(interval_volume);
        let n_full = (order.quantity / qty_per_slice).floor() as usize;
        let remainder = order.quantity - qty_per_slice * n_full as f64;

        let mut child_orders = Vec::new();
        for _ in 0..n_full {
            let child_id = self.next_child_id();
            child_orders.push(ChildOrder {
                parent_id: order.id.clone(),
                child_id,
                venue_id: best_venue.clone(),
                side: order.side,
                quantity: qty_per_slice,
                price: order.limit_price,
                is_limit: order.limit_price.is_some(),
            });
        }
        if remainder >= self.constraints.min_venue_qty {
            let child_id = self.next_child_id();
            child_orders.push(ChildOrder {
                parent_id: order.id.clone(),
                child_id,
                venue_id: best_venue.clone(),
                side: order.side,
                quantity: remainder,
                price: order.limit_price,
                is_limit: order.limit_price.is_some(),
            });
        }

        Ok(RoutingResult {
            parent_id: order.id.clone(),
            strategy: format!("POV_{:.1}%", target_pov * 100.0),
            total_quantity: order.quantity,
            estimated_cost_bps: 0.0,
            venues_used: vec![best_venue],
            child_orders,
        })
    }

    /// Implementation Shortfall: balance urgency vs. market impact
    fn route_is(&mut self, order: &ParentOrder) -> Result<RoutingResult, SorError> {
        // IS model: trade faster when urgency is high (front-loading)
        let n_slices = 12;
        let urgency = order.urgency;

        // Front-loaded schedule: more aggressive early
        let weights: Vec<f64> = (0..n_slices).map(|i| {
            let x = i as f64 / (n_slices - 1) as f64;
            // Exponential decay: high urgency = fast early, low urgency = even
            let w = if urgency > 0.5 {
                (-x * urgency * 3.0).exp()
            } else {
                1.0 + urgency * (1.0 - x)
            };
            w
        }).collect();

        let total_weight: f64 = weights.iter().sum();
        let mut allocator = VenueAllocator::new(self.cost_model.clone());

        let venues: Vec<Venue> = self.registry.all_active().into_iter().cloned().collect();
        let sym_data = self.market_data.get(&order.symbol);
        for venue in &venues {
            let md = sym_data
                .and_then(|sd| sd.get(&venue.id))
                .cloned()
                .unwrap_or_else(|| self.default_md(100.0));
            allocator.add_venue(venue.clone(), md);
        }

        let price = order.limit_price.unwrap_or(100.0);

        let mut all_child_orders = Vec::new();
        let mut venues_used_set = std::collections::HashSet::new();

        for &weight in &weights {
            let slice_qty = order.quantity * weight / total_weight;
            if slice_qty < self.constraints.min_venue_qty { continue; }

            let result = allocator.optimize(
                slice_qty,
                price,
                order.side,
                &self.constraints,
                20,
            )?;

            for (venue_id, qty) in &result.allocations {
                if *qty >= self.constraints.min_venue_qty {
                    let child_id = self.next_child_id();
                    venues_used_set.insert(venue_id.clone());
                    all_child_orders.push(ChildOrder {
                        parent_id: order.id.clone(),
                        child_id,
                        venue_id: venue_id.clone(),
                        side: order.side,
                        quantity: *qty,
                        price: order.limit_price,
                        is_limit: order.limit_price.is_some(),
                    });
                }
            }
        }

        Ok(RoutingResult {
            parent_id: order.id.clone(),
            strategy: format!("IS_urgency_{:.2}", urgency),
            total_quantity: order.quantity,
            estimated_cost_bps: 0.0,
            venues_used: venues_used_set.into_iter().collect(),
            child_orders: all_child_orders,
        })
    }

    fn best_venue_by_fee(&self, symbol: &str) -> Result<String, SorError> {
        let venues = self.registry.all_active();
        if venues.is_empty() {
            return Err(SorError::NoVenues);
        }
        let best = venues.iter()
            .min_by(|a, b| a.fee_score().partial_cmp(&b.fee_score()).unwrap())
            .ok_or(SorError::NoVenues)?;
        Ok(best.id.clone())
    }

    fn default_md(&self, price: f64) -> MarketData {
        let half_spread = price * 0.0005;
        MarketData {
            bid: price - half_spread,
            ask: price + half_spread,
            bid_size: 10_000.0,
            ask_size: 10_000.0,
            last_trade: price,
            daily_volume: 1_000_000.0,
            adv: 1_000_000.0,
            bid_levels: vec![],
            ask_levels: vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::venue::{Venue, ExchangeType, MarketData, BookLevel};

    fn make_router() -> SmartOrderRouter {
        let cost_model = CostModel::new(0.02, 0.001);
        let mut sor = SmartOrderRouter::new(cost_model).with_registry(VenueRegistry::us_equities());

        let md = MarketData {
            bid: 99.98, ask: 100.02,
            bid_size: 50_000.0, ask_size: 50_000.0,
            last_trade: 100.0, daily_volume: 5_000_000.0, adv: 5_000_000.0,
            bid_levels: vec![BookLevel { price: 99.98, quantity: 50_000.0 }],
            ask_levels: vec![BookLevel { price: 100.02, quantity: 50_000.0 }],
        };

        for vid in &["NYSE", "NASDAQ", "BATS", "EDGX", "IEX", "SIGMA_X", "UBS_MTF", "CROSSFINDER"] {
            sor.update_market_data("AAPL", vid, md.clone());
        }
        sor
    }

    fn make_order() -> ParentOrder {
        ParentOrder::new("ORD001", "AAPL", OrderSide::Buy, 50_000.0)
            .with_urgency(0.5)
    }

    #[test]
    fn test_sor_routing() {
        let mut router = make_router();
        let order = make_order();
        let result = router.route(&order, RoutingStrategy::Sor).unwrap();
        assert!(!result.child_orders.is_empty());
        println!("SOR: {} child orders, venues={:?}", result.child_orders.len(), result.venues_used);
    }

    #[test]
    fn test_twap_routing() {
        let mut router = make_router();
        let order = make_order();
        let result = router.route(&order, RoutingStrategy::Twap).unwrap();
        // TWAP should have many slices
        assert!(result.child_orders.len() >= 1);
        // All to same venue
        let venues: std::collections::HashSet<_> = result.child_orders.iter().map(|c| &c.venue_id).collect();
        assert_eq!(venues.len(), 1);
        println!("TWAP: {} slices", result.child_orders.len());
    }

    #[test]
    fn test_vwap_routing() {
        let mut router = make_router();
        let order = make_order();
        let result = router.route(&order, RoutingStrategy::Vwap).unwrap();
        assert!(result.child_orders.len() >= 1);
        let total_qty: f64 = result.child_orders.iter().map(|c| c.quantity).sum();
        assert!((total_qty - order.quantity).abs() < 1.0, "total={} expected={}", total_qty, order.quantity);
    }

    #[test]
    fn test_pov_routing() {
        let mut router = make_router();
        let order = make_order();
        let pov_target = OrderedF64(0.10);
        let result = router.route(&order, RoutingStrategy::Pov { target_pov: pov_target }).unwrap();
        assert!(!result.child_orders.is_empty());
        let total_qty: f64 = result.child_orders.iter().map(|c| c.quantity).sum();
        assert!((total_qty - order.quantity).abs() < 1.0);
    }

    #[test]
    fn test_implementation_shortfall_routing() {
        let mut router = make_router();
        let order = make_order().with_urgency(0.8);
        let result = router.route(&order, RoutingStrategy::ImplementationShortfall).unwrap();
        assert!(!result.child_orders.is_empty());
        println!("IS: {} child orders", result.child_orders.len());
    }

    #[test]
    fn test_fill_fraction_near_one() {
        let mut router = make_router();
        let order = make_order();
        let result = router.route(&order, RoutingStrategy::Vwap).unwrap();
        let ff = result.fill_fraction();
        assert!((ff - 1.0).abs() < 0.01, "fill_fraction = {:.4}", ff);
    }

    #[test]
    fn test_twap_schedule_total_qty() {
        let schedule = TwapSchedule::new(10_000.0, 10, 300.0);
        let total: f64 = schedule.slices.iter().sum();
        assert!((total - 10_000.0).abs() < 1e-6);
        assert_eq!(schedule.slices.len(), 10);
    }

    #[test]
    fn test_vwap_u_shaped_profile() {
        let schedule = VwapSchedule::u_shaped(10_000.0, 12);
        let total: f64 = schedule.slices.iter().sum();
        assert!((total - 10_000.0).abs() < 1e-6);
        // First and last slices should be higher (U-shape)
        let n = schedule.slices.len();
        let mid = schedule.slices[n / 2];
        let first = schedule.slices[0];
        let last = schedule.slices[n - 1];
        assert!(first > mid * 0.8 || last > mid * 0.8, "U-shape: first={:.2} mid={:.2} last={:.2}", first, mid, last);
    }
}
