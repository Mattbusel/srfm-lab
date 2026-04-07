// liquidity_aggregator.rs
// Multi-venue liquidity aggregation.
// Builds a synthetic consolidated order book, computes NBBO, walks the
// aggregated book to fill a quantity, and estimates effective/realized spread.

use std::collections::HashMap;
use crate::venue::{BookLevel, MarketData};
use crate::SorError;

// ---- Side ------------------------------------------------------------------

/// Alias re-exported from crate root for convenience within this module.
pub use crate::OrderSide as Side;

// ---- OrderBook ------------------------------------------------------------

/// Single-venue order book snapshot.
#[derive(Debug, Clone)]
pub struct OrderBook {
    pub venue_id: String,
    pub bid_levels: Vec<BookLevel>,
    pub ask_levels: Vec<BookLevel>,
    /// Timestamp of last update (microseconds since epoch, or 0 if unknown)
    pub last_update_us: u64,
}

impl OrderBook {
    pub fn from_market_data(venue_id: &str, md: &MarketData) -> Self {
        let mut bid_levels = md.bid_levels.clone();
        let mut ask_levels = md.ask_levels.clone();
        // Ensure sorted: bids descending, asks ascending
        bid_levels.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap_or(std::cmp::Ordering::Equal));
        ask_levels.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap_or(std::cmp::Ordering::Equal));
        // If no levels were provided but top-of-book is available, synthesise one level
        if bid_levels.is_empty() && md.bid > 0.0 {
            bid_levels.push(BookLevel { price: md.bid, quantity: md.bid_size });
        }
        if ask_levels.is_empty() && md.ask > 0.0 {
            ask_levels.push(BookLevel { price: md.ask, quantity: md.ask_size });
        }
        OrderBook {
            venue_id: venue_id.to_string(),
            bid_levels,
            ask_levels,
            last_update_us: 0,
        }
    }

    /// Best bid price, or 0 if empty.
    pub fn best_bid(&self) -> f64 {
        self.bid_levels.first().map(|l| l.price).unwrap_or(0.0)
    }

    /// Best ask price, or f64::MAX if empty.
    pub fn best_ask(&self) -> f64 {
        self.ask_levels.first().map(|l| l.price).unwrap_or(f64::MAX)
    }

    /// Total quantity available on the bid side.
    pub fn total_bid_size(&self) -> f64 {
        self.bid_levels.iter().map(|l| l.quantity).sum()
    }

    /// Total quantity available on the ask side.
    pub fn total_ask_size(&self) -> f64 {
        self.ask_levels.iter().map(|l| l.quantity).sum()
    }
}

// ---- NbboCache ------------------------------------------------------------

/// National Best Bid/Offer cache across venues.
#[derive(Debug, Clone, Default)]
pub struct NbboCache {
    /// venue_id -> (bid, ask, bid_sz, ask_sz)
    quotes: HashMap<String, (f64, f64, f64, f64)>,
}

impl NbboCache {
    pub fn new() -> Self {
        NbboCache { quotes: HashMap::new() }
    }

    /// Update the cached quote for a single venue.
    pub fn update(&mut self, venue: &str, bid: f64, ask: f64, bid_sz: f64, ask_sz: f64) {
        self.quotes.insert(venue.to_string(), (bid, ask, bid_sz, ask_sz));
    }

    /// Best bid and best ask across all venues.
    pub fn get_nbbo(&self) -> (f64, f64) {
        let best_bid = self.quotes.values()
            .map(|(b, _, _, _)| *b)
            .fold(0.0_f64, f64::max);
        let best_ask = self.quotes.values()
            .map(|(_, a, _, _)| *a)
            .fold(f64::MAX, f64::min);
        (best_bid, best_ask)
    }

    /// Size available at the best bid across all venues.
    pub fn best_bid_size(&self) -> f64 {
        let (nbbo_bid, _) = self.get_nbbo();
        self.quotes.values()
            .filter(|(b, _, _, _)| (*b - nbbo_bid).abs() < 1e-9)
            .map(|(_, _, bsz, _)| *bsz)
            .sum()
    }

    /// Size available at the best ask across all venues.
    pub fn best_ask_size(&self) -> f64 {
        let (_, nbbo_ask) = self.get_nbbo();
        self.quotes.values()
            .filter(|(_, a, _, _)| (*a - nbbo_ask).abs() < 1e-9)
            .map(|(_, _, _, asz)| *asz)
            .sum()
    }

    /// Quoted spread = best_ask - best_bid.
    pub fn quoted_spread(&self) -> f64 {
        let (bid, ask) = self.get_nbbo();
        if ask == f64::MAX || bid <= 0.0 { return f64::NAN; }
        ask - bid
    }

    /// Mid-price.
    pub fn mid(&self) -> f64 {
        let (bid, ask) = self.get_nbbo();
        if ask == f64::MAX || bid <= 0.0 { return f64::NAN; }
        (bid + ask) / 2.0
    }
}

// ---- DepthLevel -----------------------------------------------------------

/// An aggregated depth level combining quantity across all venues at a price.
#[derive(Debug, Clone)]
pub struct DepthLevel {
    pub price: f64,
    pub total_quantity: f64,
    /// Venue contributions: (venue_id, quantity)
    pub venue_contributions: Vec<(String, f64)>,
}

// ---- AggregatedLiquidity --------------------------------------------------

/// Combined order book across all venues.
#[derive(Debug, Clone)]
pub struct AggregatedLiquidity {
    pub best_bid: f64,
    pub best_ask: f64,
    pub total_bid_size: f64,
    pub total_ask_size: f64,
    /// Aggregated bid depth levels sorted by price descending
    pub bid_depth: Vec<DepthLevel>,
    /// Aggregated ask depth levels sorted by price ascending
    pub ask_depth: Vec<DepthLevel>,
}

impl AggregatedLiquidity {
    /// Quoted spread.
    pub fn spread(&self) -> f64 {
        self.best_ask - self.best_bid
    }

    /// Mid-price.
    pub fn mid(&self) -> f64 {
        (self.best_bid + self.best_ask) / 2.0
    }

    /// Total depth available within `price_limit` on the given side.
    pub fn depth_within_limit(&self, price_limit: f64, side: Side) -> f64 {
        match side {
            Side::Buy => {
                self.ask_depth.iter()
                    .filter(|l| l.price <= price_limit)
                    .map(|l| l.total_quantity)
                    .sum()
            }
            Side::Sell => {
                self.bid_depth.iter()
                    .filter(|l| l.price >= price_limit)
                    .map(|l| l.total_quantity)
                    .sum()
            }
        }
    }
}

// ---- VenueLiquidity -------------------------------------------------------

/// Liquidity available at a specific venue and price level.
#[derive(Debug, Clone)]
pub struct VenueLiquidity {
    pub venue_id: String,
    pub price: f64,
    pub available_qty: f64,
    /// Incremental cost vs mid in bps
    pub cost_bps: f64,
}

// ---- EstimatedFillSchedule ------------------------------------------------

/// A planned fill: one venue, one price level, a quantity.
#[derive(Debug, Clone)]
pub struct FillTier {
    pub venue_id: String,
    pub price: f64,
    pub size: f64,
    pub cumulative_size: f64,
    /// VWAP of fills through this tier
    pub vwap_through: f64,
}

/// Full fill schedule for a given quantity, sorted by cost (best price first).
#[derive(Debug, Clone)]
pub struct EstimatedFillSchedule {
    pub tiers: Vec<FillTier>,
    pub total_qty: f64,
    pub filled_qty: f64,
    pub vwap: f64,
    /// Slippage vs mid-price in bps
    pub slippage_bps: f64,
}

impl EstimatedFillSchedule {
    pub fn is_fully_filled(&self) -> bool {
        (self.filled_qty - self.total_qty).abs() < 1e-6
    }
}

// ---- SpreadCalculator -----------------------------------------------------

/// Computes spread metrics from a sequence of fills.
#[derive(Debug, Clone, Default)]
pub struct SpreadCalculator {
    /// Decision price (mid at time of order): the reference for price impact
    pub decision_mid: f64,
    /// Fill records: (fill_price, fill_qty)
    fills: Vec<(f64, f64)>,
    /// Post-fill mid prices recorded some time after each fill
    post_mids: Vec<f64>,
}

impl SpreadCalculator {
    pub fn new(decision_mid: f64) -> Self {
        SpreadCalculator {
            decision_mid,
            fills: Vec::new(),
            post_mids: Vec::new(),
        }
    }

    pub fn add_fill(&mut self, price: f64, qty: f64) {
        self.fills.push((price, qty));
    }

    /// Add a post-fill mid price for computing realized spread.
    pub fn add_post_mid(&mut self, mid: f64) {
        self.post_mids.push(mid);
    }

    /// VWAP of all fills.
    pub fn fill_vwap(&self) -> f64 {
        let total_qty: f64 = self.fills.iter().map(|(_, q)| q).sum();
        if total_qty <= 0.0 { return 0.0; }
        let total_notional: f64 = self.fills.iter().map(|(p, q)| p * q).sum();
        total_notional / total_qty
    }

    /// Effective spread = 2 * |fill_vwap - decision_mid| in bps of decision_mid.
    /// This measures total round-trip cost.
    pub fn effective_spread_bps(&self) -> f64 {
        if self.decision_mid <= 0.0 { return 0.0; }
        let vwap = self.fill_vwap();
        2.0 * (vwap - self.decision_mid).abs() / self.decision_mid * 10_000.0
    }

    /// Price impact = fill_vwap - decision_mid (in bps) for buy orders.
    /// Positive = we paid more than mid, negative = price improvement.
    pub fn price_impact_bps(&self, side: Side) -> f64 {
        if self.decision_mid <= 0.0 { return 0.0; }
        let vwap = self.fill_vwap();
        let raw = (vwap - self.decision_mid) / self.decision_mid * 10_000.0;
        match side {
            Side::Buy  => raw,    // positive = bad for buyer
            Side::Sell => -raw,   // negative = bad for seller (we sold at lower price)
        }
    }

    /// Realized spread = 2 * (fill_price - post_mid) for buys (in bps).
    /// Measures how much of the spread we paid that was permanent vs. transient.
    pub fn realized_spread_bps(&self, side: Side) -> f64 {
        if self.fills.is_empty() || self.post_mids.is_empty() { return 0.0; }
        if self.decision_mid <= 0.0 { return 0.0; }
        let n = self.fills.len().min(self.post_mids.len());
        let sum: f64 = (0..n).map(|i| {
            let (fill_price, _) = self.fills[i];
            let post_mid = self.post_mids[i];
            match side {
                Side::Buy  =>  2.0 * (fill_price - post_mid),
                Side::Sell => -2.0 * (fill_price - post_mid),
            }
        }).sum();
        sum / (n as f64 * self.decision_mid) * 10_000.0
    }
}

// ---- LiquidityAggregator --------------------------------------------------

/// Aggregates order books across multiple venues to provide a consolidated view.
pub struct LiquidityAggregator {
    /// venue_id -> order book
    venues: HashMap<String, OrderBook>,
    pub nbbo_cache: NbboCache,
}

impl LiquidityAggregator {
    pub fn new() -> Self {
        LiquidityAggregator {
            venues: HashMap::new(),
            nbbo_cache: NbboCache::new(),
        }
    }

    /// Add or replace the order book for a venue.
    pub fn update_venue(&mut self, venue_id: &str, book: OrderBook) {
        // Update NBBO cache
        let bid = book.best_bid();
        let ask = book.best_ask();
        let bid_sz = book.bid_levels.first().map(|l| l.quantity).unwrap_or(0.0);
        let ask_sz = book.ask_levels.first().map(|l| l.quantity).unwrap_or(0.0);
        self.nbbo_cache.update(venue_id, bid, ask, bid_sz, ask_sz);
        self.venues.insert(venue_id.to_string(), book);
    }

    /// Update from MarketData directly.
    pub fn update_from_md(&mut self, venue_id: &str, md: &MarketData) {
        let book = OrderBook::from_market_data(venue_id, md);
        self.update_venue(venue_id, book);
    }

    /// Number of venues currently tracked.
    pub fn venue_count(&self) -> usize {
        self.venues.len()
    }

    /// Build the aggregated liquidity snapshot.
    pub fn aggregate(&self) -> AggregatedLiquidity {
        let (best_bid, best_ask) = self.nbbo_cache.get_nbbo();

        // Collect all bid and ask levels across venues
        let mut bid_map: HashMap<u64, DepthLevel> = HashMap::new();
        let mut ask_map: HashMap<u64, DepthLevel> = HashMap::new();

        for (venue_id, book) in &self.venues {
            for level in &book.bid_levels {
                let key = price_to_key(level.price);
                let entry = bid_map.entry(key).or_insert_with(|| DepthLevel {
                    price: level.price,
                    total_quantity: 0.0,
                    venue_contributions: Vec::new(),
                });
                entry.total_quantity += level.quantity;
                entry.venue_contributions.push((venue_id.clone(), level.quantity));
            }
            for level in &book.ask_levels {
                let key = price_to_key(level.price);
                let entry = ask_map.entry(key).or_insert_with(|| DepthLevel {
                    price: level.price,
                    total_quantity: 0.0,
                    venue_contributions: Vec::new(),
                });
                entry.total_quantity += level.quantity;
                entry.venue_contributions.push((venue_id.clone(), level.quantity));
            }
        }

        let mut bid_depth: Vec<DepthLevel> = bid_map.into_values().collect();
        let mut ask_depth: Vec<DepthLevel> = ask_map.into_values().collect();

        // Sort: bids descending, asks ascending
        bid_depth.sort_by(|a, b| b.price.partial_cmp(&a.price).unwrap_or(std::cmp::Ordering::Equal));
        ask_depth.sort_by(|a, b| a.price.partial_cmp(&b.price).unwrap_or(std::cmp::Ordering::Equal));

        let total_bid_size = bid_depth.iter().map(|l| l.total_quantity).sum();
        let total_ask_size = ask_depth.iter().map(|l| l.total_quantity).sum();

        AggregatedLiquidity {
            best_bid,
            best_ask: if best_ask == f64::MAX { 0.0 } else { best_ask },
            total_bid_size,
            total_ask_size,
            bid_depth,
            ask_depth,
        }
    }

    /// Walk the aggregated book to find available liquidity for a given quantity.
    /// Returns a list of (venue, price, available_qty) sorted by best price first.
    pub fn get_available_liquidity(&self, side: Side, qty: f64) -> Vec<VenueLiquidity> {
        let liq = self.aggregate();
        let mid = self.nbbo_cache.mid();
        let levels = match side {
            Side::Buy  => &liq.ask_depth, // buying -- walk asks ascending
            Side::Sell => &liq.bid_depth, // selling -- walk bids descending
        };

        let mut result = Vec::new();
        let mut remaining = qty;

        for level in levels {
            if remaining <= 0.0 { break; }
            for (venue_id, venue_qty) in &level.venue_contributions {
                if remaining <= 0.0 { break; }
                let take = venue_qty.min(remaining);
                let cost_bps = if mid > 0.0 {
                    match side {
                        Side::Buy  => (level.price - mid) / mid * 10_000.0,
                        Side::Sell => (mid - level.price) / mid * 10_000.0,
                    }
                } else {
                    0.0
                };
                result.push(VenueLiquidity {
                    venue_id: venue_id.clone(),
                    price: level.price,
                    available_qty: take,
                    cost_bps,
                });
                remaining -= take;
            }
        }
        result
    }

    /// Compute the estimated fill schedule for a given quantity.
    /// Returns tiered fills sorted from best to worst price.
    pub fn fill_schedule(&self, side: Side, qty: f64) -> Result<EstimatedFillSchedule, SorError> {
        if self.venues.is_empty() {
            return Err(SorError::NoVenues);
        }
        let liquidity = self.get_available_liquidity(side, qty);
        let available: f64 = liquidity.iter().map(|l| l.available_qty).sum();

        if available < qty - 1e-6 {
            return Err(SorError::InsufficientLiquidity { needed: qty, available });
        }

        let mut tiers = Vec::new();
        let mut cumulative = 0.0;
        let mut cum_notional = 0.0;

        for liq in &liquidity {
            cumulative += liq.available_qty;
            cum_notional += liq.price * liq.available_qty;
            let vwap_through = if cumulative > 0.0 { cum_notional / cumulative } else { 0.0 };
            tiers.push(FillTier {
                venue_id: liq.venue_id.clone(),
                price: liq.price,
                size: liq.available_qty,
                cumulative_size: cumulative,
                vwap_through,
            });
        }

        let filled_qty = cumulative.min(qty);
        let vwap = if filled_qty > 0.0 { cum_notional / cumulative } else { 0.0 };
        let mid = self.nbbo_cache.mid();
        let slippage_bps = if mid > 0.0 {
            match side {
                Side::Buy  => (vwap - mid) / mid * 10_000.0,
                Side::Sell => (mid - vwap) / mid * 10_000.0,
            }
        } else { 0.0 };

        Ok(EstimatedFillSchedule {
            tiers,
            total_qty: qty,
            filled_qty,
            vwap,
            slippage_bps,
        })
    }
}

impl Default for LiquidityAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a price to a HashMap key using fixed 6-decimal precision to avoid
/// floating-point equality issues when merging levels across venues.
fn price_to_key(price: f64) -> u64 {
    // Round to nearest 0.0001 (4 decimal places) and encode as integer
    ((price * 10_000.0).round() as i64).unsigned_abs()
}

// ---- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_md(bid: f64, ask: f64, bid_sz: f64, ask_sz: f64) -> MarketData {
        MarketData {
            bid,
            ask,
            bid_size: bid_sz,
            ask_size: ask_sz,
            last_trade: (bid + ask) / 2.0,
            daily_volume: 1_000_000.0,
            adv: 1_000_000.0,
            bid_levels: vec![BookLevel { price: bid, quantity: bid_sz }],
            ask_levels: vec![BookLevel { price: ask, quantity: ask_sz }],
        }
    }

    fn three_venue_aggregator() -> LiquidityAggregator {
        let mut agg = LiquidityAggregator::new();
        agg.update_from_md("NYSE",   &make_md(99.98, 100.02, 10_000.0, 10_000.0));
        agg.update_from_md("NASDAQ", &make_md(99.97, 100.03,  8_000.0,  8_000.0));
        agg.update_from_md("BATS",   &make_md(99.99, 100.01, 12_000.0, 12_000.0));
        agg
    }

    #[test]
    fn test_nbbo_best_bid_ask() {
        let agg = three_venue_aggregator();
        let (bid, ask) = agg.nbbo_cache.get_nbbo();
        assert!((bid - 99.99).abs() < 1e-9, "best_bid={}", bid);
        assert!((ask - 100.01).abs() < 1e-9, "best_ask={}", ask);
    }

    #[test]
    fn test_aggregate_total_sizes() {
        let agg = three_venue_aggregator();
        let liq = agg.aggregate();
        assert!((liq.total_ask_size - 30_000.0).abs() < 1e-6,
            "total_ask_size={}", liq.total_ask_size);
        assert!((liq.total_bid_size - 30_000.0).abs() < 1e-6,
            "total_bid_size={}", liq.total_bid_size);
    }

    #[test]
    fn test_aggregate_best_prices() {
        let agg = three_venue_aggregator();
        let liq = agg.aggregate();
        assert!((liq.best_bid - 99.99).abs() < 1e-9);
        assert!((liq.best_ask - 100.01).abs() < 1e-9);
    }

    #[test]
    fn test_get_available_liquidity_buy() {
        let agg = three_venue_aggregator();
        // Ask levels across venues: 100.01 (12000), 100.02 (10000), 100.03 (8000)
        // Request 25000 -- should span first two levels at least
        let liq = agg.get_available_liquidity(Side::Buy, 25_000.0);
        let total: f64 = liq.iter().map(|l| l.available_qty).sum();
        assert!((total - 25_000.0).abs() < 1e-6, "total={}", total);
        // All cost_bps should be >= 0 for buys
        for l in &liq {
            assert!(l.cost_bps >= -1e-6, "negative cost_bps={}", l.cost_bps);
        }
    }

    #[test]
    fn test_get_available_liquidity_sell() {
        let agg = three_venue_aggregator();
        let liq = agg.get_available_liquidity(Side::Sell, 15_000.0);
        let total: f64 = liq.iter().map(|l| l.available_qty).sum();
        assert!((total - 15_000.0).abs() < 1e-6, "total={}", total);
    }

    #[test]
    fn test_fill_schedule_fully_filled() {
        let agg = three_venue_aggregator();
        let sched = agg.fill_schedule(Side::Buy, 20_000.0).unwrap();
        assert!(sched.is_fully_filled());
        assert!(sched.vwap > 100.0);
    }

    #[test]
    fn test_fill_schedule_insufficient_liquidity() {
        let agg = three_venue_aggregator();
        let err = agg.fill_schedule(Side::Buy, 100_000.0);
        assert!(matches!(err, Err(SorError::InsufficientLiquidity { .. })));
    }

    #[test]
    fn test_spread_calculator_effective_spread() {
        let mut calc = SpreadCalculator::new(100.0);
        calc.add_fill(100.05, 500.0);
        calc.add_fill(100.10, 500.0);
        // VWAP = 100.075, effective spread = 2 * |100.075 - 100| / 100 * 10000 = 15 bps
        let eff = calc.effective_spread_bps();
        assert!((eff - 15.0).abs() < 0.5, "eff_spread={}", eff);
    }

    #[test]
    fn test_spread_calculator_price_impact() {
        let mut calc = SpreadCalculator::new(100.0);
        calc.add_fill(100.05, 1000.0);
        let impact = calc.price_impact_bps(Side::Buy);
        assert!((impact - 5.0).abs() < 0.1, "impact={}", impact);
    }

    #[test]
    fn test_nbbo_cache_quoted_spread() {
        let mut cache = NbboCache::new();
        cache.update("V1", 99.98, 100.02, 1000.0, 1000.0);
        cache.update("V2", 99.99, 100.01, 500.0, 500.0);
        let spread = cache.quoted_spread();
        assert!((spread - 0.02).abs() < 1e-9, "spread={}", spread);
    }

    #[test]
    fn test_orderbook_from_market_data() {
        let md = make_md(99.0, 101.0, 1000.0, 2000.0);
        let book = OrderBook::from_market_data("TEST", &md);
        assert!((book.best_bid() - 99.0).abs() < 1e-9);
        assert!((book.best_ask() - 101.0).abs() < 1e-9);
        assert_eq!(book.total_bid_size(), 1000.0);
        assert_eq!(book.total_ask_size(), 2000.0);
    }
}
