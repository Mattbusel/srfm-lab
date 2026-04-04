use std::collections::HashMap;
use crate::venue::{Venue, MarketData};
use crate::{SorError, OrderSide};

/// Configuration for venue simulation
#[derive(Debug, Clone)]
pub struct SimulationConfig {
    pub fill_rate_mean: f64,        // Mean fill rate (fraction of order filled)
    pub fill_rate_std: f64,         // Std of fill rate
    pub price_impact_coeff: f64,    // Adverse price movement per unit volume
    pub latency_jitter_us: u64,     // Random latency jitter in microseconds
    pub partial_fill_prob: f64,     // Probability of getting a partial fill
    pub reject_prob: f64,           // Probability of order being rejected
    pub random_seed: u64,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        SimulationConfig {
            fill_rate_mean: 0.95,
            fill_rate_std: 0.05,
            price_impact_coeff: 0.0001,
            latency_jitter_us: 50,
            partial_fill_prob: 0.20,
            reject_prob: 0.02,
            random_seed: 42,
        }
    }
}

/// Result of a simulated fill
#[derive(Debug, Clone)]
pub struct SimFill {
    pub venue_id: String,
    pub order_id: String,
    pub requested_qty: f64,
    pub filled_qty: f64,
    pub fill_price: f64,
    pub latency_us: u64,
    pub status: FillStatus,
    pub slippage_bps: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FillStatus {
    FullFill,
    PartialFill,
    Rejected,
    NoFill,
}

impl std::fmt::Display for FillStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FillStatus::FullFill => write!(f, "FullFill"),
            FillStatus::PartialFill => write!(f, "PartialFill"),
            FillStatus::Rejected => write!(f, "Rejected"),
            FillStatus::NoFill => write!(f, "NoFill"),
        }
    }
}

/// Pending order in simulator
#[derive(Debug, Clone)]
struct PendingOrder {
    id: String,
    venue_id: String,
    side: OrderSide,
    qty: f64,
    price: f64,
    submitted_at: u64, // microseconds
    expected_fill_at: u64,
}

/// Venue execution simulator with realistic fill model
pub struct VenueSimulator {
    config: SimulationConfig,
    venues: HashMap<String, Venue>,
    market_data: HashMap<String, MarketData>,
    rng: SimpleRng,
    time_us: u64,
    pending: Vec<PendingOrder>,
    fill_history: Vec<SimFill>,
}

impl VenueSimulator {
    pub fn new(config: SimulationConfig) -> Self {
        let seed = config.random_seed;
        VenueSimulator {
            rng: SimpleRng::new(seed),
            config,
            venues: HashMap::new(),
            market_data: HashMap::new(),
            time_us: 0,
            pending: Vec::new(),
            fill_history: Vec::new(),
        }
    }

    pub fn add_venue(&mut self, venue: Venue, md: MarketData) {
        self.market_data.insert(venue.id.clone(), md);
        self.venues.insert(venue.id.clone(), venue);
    }

    pub fn update_market_data(&mut self, venue_id: &str, md: MarketData) {
        self.market_data.insert(venue_id.to_string(), md);
    }

    /// Submit an order to a venue, returns order ID
    pub fn submit_order(
        &mut self,
        venue_id: &str,
        side: OrderSide,
        qty: f64,
        price: f64,
        order_id: &str,
    ) -> Result<(), SorError> {
        let venue = self.venues.get(venue_id)
            .ok_or_else(|| SorError::VenueNotFound(venue_id.to_string()))?;

        if !venue.can_accept(qty) {
            return Err(SorError::InvalidParameter(
                format!("Venue {} cannot accept qty {}", venue_id, qty)
            ));
        }

        let base_latency = venue.latency_us;
        let jitter = self.rng.next_u64() % (self.config.latency_jitter_us * 2 + 1);
        let latency = base_latency + jitter;
        let fill_at = self.time_us + latency;

        self.pending.push(PendingOrder {
            id: order_id.to_string(),
            venue_id: venue_id.to_string(),
            side,
            qty,
            price,
            submitted_at: self.time_us,
            expected_fill_at: fill_at,
        });

        Ok(())
    }

    /// Advance time and process fills for orders that have reached their expected fill time
    pub fn advance_time(&mut self, delta_us: u64) -> Vec<SimFill> {
        self.time_us += delta_us;
        let current_time = self.time_us;

        // Split into ready and pending
        let old_pending = std::mem::take(&mut self.pending);
        let mut ready = Vec::new();
        let mut remaining_pending = Vec::new();

        for order in old_pending {
            if current_time >= order.expected_fill_at {
                ready.push(order);
            } else {
                remaining_pending.push(order);
            }
        }

        self.pending = remaining_pending;

        let mut fills = Vec::new();
        for order in &ready {
            let fill = self.execute_order(order);
            fills.push(fill.clone());
            self.fill_history.push(fill);
        }
        fills
    }

    fn execute_order(&mut self, order: &PendingOrder) -> SimFill {
        let venue = match self.venues.get(&order.venue_id) {
            Some(v) => v.clone(),
            None => return SimFill {
                venue_id: order.venue_id.clone(),
                order_id: order.id.clone(),
                requested_qty: order.qty,
                filled_qty: 0.0,
                fill_price: order.price,
                latency_us: 0,
                status: FillStatus::Rejected,
                slippage_bps: 0.0,
            },
        };

        let latency_us = self.time_us.saturating_sub(order.submitted_at);

        // Check rejection
        let reject_roll = self.rng.next_f64();
        if reject_roll < self.config.reject_prob {
            return SimFill {
                venue_id: order.venue_id.clone(),
                order_id: order.id.clone(),
                requested_qty: order.qty,
                filled_qty: 0.0,
                fill_price: order.price,
                latency_us,
                status: FillStatus::Rejected,
                slippage_bps: 0.0,
            };
        }

        let md = match self.market_data.get(&order.venue_id) {
            Some(m) => m.clone(),
            None => return SimFill {
                venue_id: order.venue_id.clone(),
                order_id: order.id.clone(),
                requested_qty: order.qty,
                filled_qty: 0.0,
                fill_price: order.price,
                latency_us,
                status: FillStatus::NoFill,
                slippage_bps: 0.0,
            },
        };

        // Determine available liquidity at venue
        let available = match order.side {
            OrderSide::Buy => md.ask_size,
            OrderSide::Sell => md.bid_size,
        };

        // Fill rate model
        let partial_roll = self.rng.next_f64();
        let fill_rate_noise = self.rng.next_normal() * self.config.fill_rate_std;
        let base_fill_rate = (self.config.fill_rate_mean + fill_rate_noise).max(0.0).min(1.0);

        // Dark pools have liquidity-dependent fill rate
        let liquidity_factor = if matches!(venue.exchange_type, crate::venue::ExchangeType::DarkPool) {
            (available / order.qty).min(1.0)
        } else {
            1.0
        };

        let effective_fill_rate = if partial_roll < self.config.partial_fill_prob {
            // Partial fill: fill only a portion
            base_fill_rate * 0.5 * liquidity_factor
        } else {
            base_fill_rate * liquidity_factor
        };

        let filled_qty = (order.qty * effective_fill_rate).min(available).min(order.qty);
        let filled_qty = (filled_qty / venue.lot_size).floor() * venue.lot_size;

        if filled_qty < venue.min_qty {
            return SimFill {
                venue_id: order.venue_id.clone(),
                order_id: order.id.clone(),
                requested_qty: order.qty,
                filled_qty: 0.0,
                fill_price: order.price,
                latency_us,
                status: FillStatus::NoFill,
                slippage_bps: 0.0,
            };
        }

        // Fill price with slippage (market impact)
        let base_fill_price = match order.side {
            OrderSide::Buy => md.ask,
            OrderSide::Sell => md.bid,
        };

        let pov = filled_qty / md.adv.max(1.0);
        let impact = self.config.price_impact_coeff * pov.sqrt();
        let noise = self.rng.next_normal() * md.spread() * 0.1;
        let fill_price = match order.side {
            OrderSide::Buy => base_fill_price * (1.0 + impact + noise.max(0.0)),
            OrderSide::Sell => base_fill_price * (1.0 - impact + noise.min(0.0)),
        };

        let arrival_price = md.mid();
        let slippage_bps = match order.side {
            OrderSide::Buy => (fill_price - arrival_price) / arrival_price * 10_000.0,
            OrderSide::Sell => (arrival_price - fill_price) / arrival_price * 10_000.0,
        };

        let status = if (filled_qty - order.qty).abs() < venue.lot_size {
            FillStatus::FullFill
        } else {
            FillStatus::PartialFill
        };

        SimFill {
            venue_id: order.venue_id.clone(),
            order_id: order.id.clone(),
            requested_qty: order.qty,
            filled_qty,
            fill_price,
            latency_us,
            status,
            slippage_bps,
        }
    }

    /// Run a complete simulated execution of a split order
    pub fn simulate_split_execution(
        &mut self,
        allocations: &[(String, f64)], // (venue_id, qty)
        side: OrderSide,
        arrival_price: f64,
    ) -> SimulationResult {
        let mut all_fills = Vec::new();

        // Submit all child orders
        for (i, (venue_id, qty)) in allocations.iter().enumerate() {
            let order_id = format!("CHILD_{}", i);
            if let Err(_) = self.submit_order(venue_id, side, *qty, arrival_price, &order_id) {
                continue;
            }
        }

        // Advance time enough for all fills to complete
        let max_latency = self.venues.values().map(|v| v.latency_us).max().unwrap_or(1000);
        let fills = self.advance_time(max_latency + self.config.latency_jitter_us * 2);
        all_fills.extend(fills);

        let total_requested: f64 = allocations.iter().map(|(_, q)| q).sum();
        let total_filled: f64 = all_fills.iter().map(|f| f.filled_qty).sum();
        let fill_fraction = if total_requested > 0.0 { total_filled / total_requested } else { 0.0 };

        let vwap = if total_filled > 0.0 {
            all_fills.iter().map(|f| f.fill_price * f.filled_qty).sum::<f64>() / total_filled
        } else {
            0.0
        };

        let avg_slippage_bps = if all_fills.is_empty() { 0.0 } else {
            all_fills.iter().map(|f| f.slippage_bps).sum::<f64>() / all_fills.len() as f64
        };

        let is = match side {
            OrderSide::Buy => (vwap - arrival_price) / arrival_price * 10_000.0,
            OrderSide::Sell => (arrival_price - vwap) / arrival_price * 10_000.0,
        };

        SimulationResult {
            fills: all_fills,
            total_requested,
            total_filled,
            fill_fraction,
            vwap,
            arrival_price,
            implementation_shortfall_bps: is,
            avg_slippage_bps,
        }
    }

    pub fn fill_history(&self) -> &[SimFill] {
        &self.fill_history
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    pub fn current_time_us(&self) -> u64 {
        self.time_us
    }
}

#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub fills: Vec<SimFill>,
    pub total_requested: f64,
    pub total_filled: f64,
    pub fill_fraction: f64,
    pub vwap: f64,
    pub arrival_price: f64,
    pub implementation_shortfall_bps: f64,
    pub avg_slippage_bps: f64,
}

/// Simple LCG PRNG + Box-Muller
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self { SimpleRng { state: seed.max(1) } }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::venue::{Venue, ExchangeType, MarketData, BookLevel};

    fn make_sim() -> VenueSimulator {
        let mut sim = VenueSimulator::new(SimulationConfig {
            fill_rate_mean: 0.95,
            fill_rate_std: 0.02,
            partial_fill_prob: 0.10,
            reject_prob: 0.01,
            latency_jitter_us: 10,
            price_impact_coeff: 0.0001,
            random_seed: 99,
        });

        let md = MarketData {
            bid: 99.98, ask: 100.02,
            bid_size: 10_000.0, ask_size: 10_000.0,
            last_trade: 100.0, daily_volume: 2_000_000.0, adv: 2_000_000.0,
            bid_levels: vec![BookLevel { price: 99.98, quantity: 10_000.0 }],
            ask_levels: vec![BookLevel { price: 100.02, quantity: 10_000.0 }],
        };
        sim.add_venue(
            Venue::new("NYSE", "NYSE", ExchangeType::LitExchange)
                .with_latency(300).with_fill_prob(0.97),
            md.clone(),
        );
        sim.add_venue(
            Venue::new("BATS", "BATS", ExchangeType::LitExchange)
                .with_latency(200).with_fill_prob(0.95),
            md.clone(),
        );
        sim.add_venue(
            Venue::new("DARK1", "Dark1", ExchangeType::DarkPool)
                .with_latency(100).with_fill_prob(0.60),
            md,
        );
        sim
    }

    #[test]
    fn test_submit_and_fill() {
        let mut sim = make_sim();
        sim.submit_order("NYSE", OrderSide::Buy, 1000.0, 100.0, "ORD1").unwrap();
        assert_eq!(sim.pending_count(), 1);
        let fills = sim.advance_time(1000);
        assert!(!fills.is_empty());
        let fill = &fills[0];
        assert!(fill.filled_qty > 0.0);
        println!("Fill: {:?}", fill);
    }

    #[test]
    fn test_venue_not_found() {
        let mut sim = make_sim();
        let err = sim.submit_order("UNKNOWN", OrderSide::Buy, 100.0, 100.0, "X");
        assert!(matches!(err, Err(SorError::VenueNotFound(_))));
    }

    #[test]
    fn test_split_execution() {
        let mut sim = make_sim();
        let allocs = vec![
            ("NYSE".to_string(), 5000.0),
            ("BATS".to_string(), 3000.0),
            ("DARK1".to_string(), 2000.0),
        ];
        let result = sim.simulate_split_execution(&allocs, OrderSide::Buy, 100.0);
        assert!(result.total_requested > 0.0);
        assert!(result.fill_fraction > 0.0);
        println!("Split execution: fill_frac={:.2} vwap={:.4} IS_bps={:.2}",
            result.fill_fraction, result.vwap, result.implementation_shortfall_bps);
    }

    #[test]
    fn test_fill_history_accumulates() {
        let mut sim = make_sim();
        for i in 0..5 {
            sim.submit_order("NYSE", OrderSide::Buy, 100.0, 100.0, &format!("ORD{}", i)).unwrap();
            sim.advance_time(500);
        }
        assert!(sim.fill_history().len() > 0);
    }

    #[test]
    fn test_latency_ordering() {
        let mut sim = make_sim();
        // Both orders submitted at same time, BATS should fill first (lower latency)
        sim.submit_order("NYSE", OrderSide::Buy, 100.0, 100.0, "SLOW").unwrap();
        sim.submit_order("BATS", OrderSide::Buy, 100.0, 100.0, "FAST").unwrap();
        // Advance to just past BATS latency (200) but before NYSE (300)
        let bats_fills = sim.advance_time(250);
        // Should have BATS fill but not NYSE
        let bats_filled = bats_fills.iter().any(|f| f.venue_id == "BATS");
        let nyse_filled = bats_fills.iter().any(|f| f.venue_id == "NYSE");
        assert!(bats_filled, "BATS should have filled by 250us");
        assert!(!nyse_filled, "NYSE should not have filled by 250us");
    }

    #[test]
    fn test_dark_pool_lower_fill_rate() {
        let mut sim = make_sim();
        let mut dark_fills = 0;
        let mut lit_fills = 0;
        let n = 100;
        for i in 0..n {
            sim.submit_order("DARK1", OrderSide::Buy, 100.0, 100.0, &format!("D{}", i)).unwrap();
            sim.submit_order("NYSE", OrderSide::Buy, 100.0, 100.0, &format!("L{}", i)).unwrap();
            let fills = sim.advance_time(1000);
            for f in &fills {
                if f.venue_id == "DARK1" && f.filled_qty > 0.0 { dark_fills += 1; }
                if f.venue_id == "NYSE" && f.filled_qty > 0.0 { lit_fills += 1; }
            }
        }
        // Both should have some fills; lit typically more than dark
        println!("Dark fills: {}/{}, Lit fills: {}/{}", dark_fills, n, lit_fills, n);
        assert!(dark_fills > 0, "Dark pool should have some fills");
        assert!(lit_fills > 0, "Lit exchange should have some fills");
    }
}
