use std::collections::HashMap;
use crate::SorError;

pub type VenueId = String;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExchangeType {
    LitExchange,
    DarkPool,
    AlternativeTradingSystem,
    InternalCrossing,
    MarketMaker,
}

impl std::fmt::Display for ExchangeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExchangeType::LitExchange => write!(f, "LitExchange"),
            ExchangeType::DarkPool => write!(f, "DarkPool"),
            ExchangeType::AlternativeTradingSystem => write!(f, "ATS"),
            ExchangeType::InternalCrossing => write!(f, "InternalCrossing"),
            ExchangeType::MarketMaker => write!(f, "MarketMaker"),
        }
    }
}

/// Market depth level (one side of book)
#[derive(Debug, Clone, Copy)]
pub struct BookLevel {
    pub price: f64,
    pub quantity: f64,
}

/// Market data snapshot for a venue/symbol
#[derive(Debug, Clone)]
pub struct MarketData {
    pub bid: f64,
    pub ask: f64,
    pub bid_size: f64,
    pub ask_size: f64,
    pub last_trade: f64,
    pub daily_volume: f64,
    pub adv: f64,           // average daily volume
    pub bid_levels: Vec<BookLevel>,
    pub ask_levels: Vec<BookLevel>,
}

impl MarketData {
    pub fn spread(&self) -> f64 { self.ask - self.bid }
    pub fn mid(&self) -> f64 { (self.bid + self.ask) / 2.0 }

    pub fn available_liquidity(&self, is_buy: bool) -> f64 {
        if is_buy { self.ask_size } else { self.bid_size }
    }

    /// Total depth available within a price limit
    pub fn depth_within_limit(&self, price_limit: f64, is_buy: bool) -> f64 {
        if is_buy {
            self.ask_levels.iter()
                .filter(|l| l.price <= price_limit)
                .map(|l| l.quantity)
                .sum()
        } else {
            self.bid_levels.iter()
                .filter(|l| l.price >= price_limit)
                .map(|l| l.quantity)
                .sum()
        }
    }
}

/// Represents a trading venue
#[derive(Debug, Clone)]
pub struct Venue {
    pub id: VenueId,
    pub name: String,
    pub exchange_type: ExchangeType,
    /// Maker fee as fraction of notional (negative = rebate)
    pub fee_maker: f64,
    /// Taker fee as fraction of notional
    pub fee_taker: f64,
    /// One-way latency in microseconds
    pub latency_us: u64,
    /// Probability of fill given order reaches venue (0-1)
    pub fill_prob: f64,
    /// Minimum order size
    pub min_qty: f64,
    /// Maximum order size per instruction
    pub max_qty: f64,
    /// Lot size (quantity must be multiple of this)
    pub lot_size: f64,
    /// Venue is currently active
    pub active: bool,
    /// Participation rate cap (max fraction of ADV in one trade)
    pub pov_cap: f64,
}

impl Venue {
    pub fn new(id: &str, name: &str, exchange_type: ExchangeType) -> Self {
        Venue {
            id: id.to_string(),
            name: name.to_string(),
            exchange_type,
            fee_maker: -0.0002,  // 2bps rebate for providing liquidity
            fee_taker: 0.0003,   // 3bps for taking liquidity
            latency_us: 500,
            fill_prob: 0.95,
            min_qty: 1.0,
            max_qty: f64::INFINITY,
            lot_size: 1.0,
            active: true,
            pov_cap: 0.30,
        }
    }

    pub fn with_fees(mut self, maker: f64, taker: f64) -> Self {
        self.fee_maker = maker;
        self.fee_taker = taker;
        self
    }

    pub fn with_latency(mut self, latency_us: u64) -> Self {
        self.latency_us = latency_us;
        self
    }

    pub fn with_fill_prob(mut self, prob: f64) -> Self {
        self.fill_prob = prob.max(0.0).min(1.0);
        self
    }

    pub fn with_min_qty(mut self, min_qty: f64) -> Self {
        self.min_qty = min_qty;
        self
    }

    pub fn with_max_qty(mut self, max_qty: f64) -> Self {
        self.max_qty = max_qty;
        self
    }

    pub fn with_lot_size(mut self, lot_size: f64) -> Self {
        self.lot_size = lot_size;
        self
    }

    pub fn with_pov_cap(mut self, cap: f64) -> Self {
        self.pov_cap = cap.max(0.0).min(1.0);
        self
    }

    /// Round a quantity to the nearest lot size
    pub fn round_to_lot(&self, qty: f64) -> f64 {
        if self.lot_size <= 0.0 { return qty; }
        (qty / self.lot_size).floor() * self.lot_size
    }

    /// Check if the venue can accept an order of given quantity
    pub fn can_accept(&self, qty: f64) -> bool {
        self.active && qty >= self.min_qty && qty <= self.max_qty
    }

    /// Effective fee for a given order (maker if posting, taker if crossing)
    pub fn effective_fee(&self, is_taker: bool) -> f64 {
        if is_taker { self.fee_taker } else { self.fee_maker }
    }

    /// Fee advantage score: lower = better
    pub fn fee_score(&self) -> f64 {
        // Use taker fee as the relevant cost for market orders
        self.fee_taker
    }

    /// Latency score: lower = better
    pub fn latency_score(&self) -> f64 {
        self.latency_us as f64
    }
}

/// Registry of known venues
pub struct VenueRegistry {
    venues: HashMap<VenueId, Venue>,
}

impl VenueRegistry {
    pub fn new() -> Self {
        VenueRegistry { venues: HashMap::new() }
    }

    pub fn add(&mut self, venue: Venue) {
        self.venues.insert(venue.id.clone(), venue);
    }

    pub fn get(&self, id: &str) -> Option<&Venue> {
        self.venues.get(id)
    }

    pub fn all_active(&self) -> Vec<&Venue> {
        self.venues.values().filter(|v| v.active).collect()
    }

    pub fn all(&self) -> Vec<&Venue> {
        self.venues.values().collect()
    }

    pub fn count(&self) -> usize { self.venues.len() }

    pub fn disable(&mut self, id: &str) {
        if let Some(v) = self.venues.get_mut(id) {
            v.active = false;
        }
    }

    pub fn enable(&mut self, id: &str) {
        if let Some(v) = self.venues.get_mut(id) {
            v.active = true;
        }
    }

    /// Create a standard set of US equity venues
    pub fn us_equities() -> Self {
        let mut reg = Self::new();
        reg.add(Venue::new("NYSE", "New York Stock Exchange", ExchangeType::LitExchange)
            .with_fees(-0.0020, 0.0030)
            .with_latency(300)
            .with_fill_prob(0.97)
            .with_pov_cap(0.25));
        reg.add(Venue::new("NASDAQ", "Nasdaq", ExchangeType::LitExchange)
            .with_fees(-0.0020, 0.0030)
            .with_latency(200)
            .with_fill_prob(0.96)
            .with_pov_cap(0.25));
        reg.add(Venue::new("BATS", "Cboe BZX", ExchangeType::LitExchange)
            .with_fees(-0.0032, 0.0030)
            .with_latency(150)
            .with_fill_prob(0.95)
            .with_pov_cap(0.20));
        reg.add(Venue::new("EDGX", "Cboe EDGX", ExchangeType::LitExchange)
            .with_fees(-0.0032, 0.0030)
            .with_latency(160)
            .with_fill_prob(0.94)
            .with_pov_cap(0.20));
        reg.add(Venue::new("IEX", "IEX Exchange", ExchangeType::LitExchange)
            .with_fees(0.0009, 0.0009)
            .with_latency(650)
            .with_fill_prob(0.90)
            .with_pov_cap(0.15));
        reg.add(Venue::new("SIGMA_X", "Goldman Sachs Sigma X", ExchangeType::DarkPool)
            .with_fees(0.0010, 0.0010)
            .with_latency(100)
            .with_fill_prob(0.60)
            .with_pov_cap(0.10));
        reg.add(Venue::new("UBS_MTF", "UBS MTF", ExchangeType::DarkPool)
            .with_fees(0.0010, 0.0010)
            .with_latency(120)
            .with_fill_prob(0.55)
            .with_pov_cap(0.10));
        reg.add(Venue::new("CROSSFINDER", "Credit Suisse Crossfinder", ExchangeType::DarkPool)
            .with_fees(0.0008, 0.0008)
            .with_latency(90)
            .with_fill_prob(0.50)
            .with_pov_cap(0.10));
        reg
    }
}

impl Default for VenueRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_venue_creation() {
        let v = Venue::new("NYSE", "NYSE", ExchangeType::LitExchange)
            .with_fees(-0.002, 0.003)
            .with_latency(300)
            .with_fill_prob(0.97)
            .with_min_qty(1.0)
            .with_lot_size(1.0);
        assert!(v.can_accept(100.0));
        assert!(!v.can_accept(0.5)); // below min
    }

    #[test]
    fn test_lot_rounding() {
        let v = Venue::new("V1", "V1", ExchangeType::LitExchange).with_lot_size(100.0);
        assert_eq!(v.round_to_lot(350.0), 300.0);
        assert_eq!(v.round_to_lot(400.0), 400.0);
        assert_eq!(v.round_to_lot(99.0), 0.0);
    }

    #[test]
    fn test_registry_us_equities() {
        let reg = VenueRegistry::us_equities();
        assert_eq!(reg.count(), 8);
        let active = reg.all_active();
        assert_eq!(active.len(), 8);
    }

    #[test]
    fn test_registry_disable_enable() {
        let mut reg = VenueRegistry::us_equities();
        reg.disable("IEX");
        let active = reg.all_active();
        assert_eq!(active.len(), 7);
        reg.enable("IEX");
        assert_eq!(reg.all_active().len(), 8);
    }

    #[test]
    fn test_market_data() {
        let md = MarketData {
            bid: 99.95, ask: 100.05, bid_size: 1000.0, ask_size: 500.0,
            last_trade: 100.0, daily_volume: 1_000_000.0, adv: 2_000_000.0,
            bid_levels: vec![BookLevel { price: 99.95, quantity: 1000.0 }],
            ask_levels: vec![BookLevel { price: 100.05, quantity: 500.0 }],
        };
        assert!((md.spread() - 0.10).abs() < 1e-10);
        assert!((md.mid() - 100.0).abs() < 1e-10);
        assert_eq!(md.available_liquidity(true), 500.0);
        assert_eq!(md.available_liquidity(false), 1000.0);
    }

    #[test]
    fn test_depth_within_limit() {
        let md = MarketData {
            bid: 100.0, ask: 100.10,
            bid_size: 500.0, ask_size: 300.0,
            last_trade: 100.05, daily_volume: 1e6, adv: 2e6,
            bid_levels: vec![
                BookLevel { price: 100.0, quantity: 500.0 },
                BookLevel { price: 99.9, quantity: 300.0 },
            ],
            ask_levels: vec![
                BookLevel { price: 100.10, quantity: 300.0 },
                BookLevel { price: 100.20, quantity: 200.0 },
            ],
        };
        // Buy within 100.15 limit: should get the 100.10 level only
        let depth = md.depth_within_limit(100.15, true);
        assert!((depth - 300.0).abs() < 1e-10);
    }
}
