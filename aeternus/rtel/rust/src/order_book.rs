// =============================================================================
// AETERNUS Real-Time Execution Layer (RTEL)
// order_book.rs — High-performance order book implementation in Rust
// =============================================================================
//! Lock-free, cache-friendly limit order book implementation.
//! Designed to feed directly into the RTEL StatePublisher.
//!
//! Uses integer price levels (scaled by 1e8) for exact arithmetic.
//! BTreeMap for O(log n) level access, with best-bid/ask cached.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::{now_ns, state_publisher::LobLevel, MAX_LOB_LEVELS};

// ---------------------------------------------------------------------------
// Order side
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Bid,
    Ask,
}

// ---------------------------------------------------------------------------
// Order event types
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderAction {
    Add,
    Modify,
    Cancel,
    Trade,
}

#[derive(Debug, Clone, Copy)]
pub struct OrderEvent {
    pub order_id:   u64,
    pub timestamp:  u64,
    pub side:       Side,
    pub action:     OrderAction,
    pub price:      i64,    // scaled integer price (×1e8)
    pub size:       i64,    // scaled integer size  (×1e8)
    pub asset_id:   u32,
}

// ---------------------------------------------------------------------------
// PriceLevel — aggregated quantity at a price
// ---------------------------------------------------------------------------
#[derive(Debug, Clone, Copy, Default)]
pub struct PriceLevel {
    pub price:      i64,   // scaled
    pub total_size: i64,   // scaled
    pub n_orders:   u32,
}

impl PriceLevel {
    fn to_lob_level(&self, scale: f64) -> LobLevel {
        LobLevel {
            price: self.price as f64 * scale,
            size:  self.total_size as f64 * scale,
        }
    }
}

// ---------------------------------------------------------------------------
// LimitOrderBook
// ---------------------------------------------------------------------------
pub struct LimitOrderBook {
    asset_id:   u32,
    price_scale:f64,   // e.g. 1e-8 to convert to dollars
    size_scale: f64,

    // Bids: higher price = better (we want rev iter for top-of-book)
    bids: BTreeMap<i64, PriceLevel>,
    // Asks: lower price = better
    asks: BTreeMap<i64, PriceLevel>,

    sequence:     AtomicU64,
    last_trade_price: i64,
    last_trade_size:  i64,

    // Cache top-of-book for O(1) access
    best_bid:    i64,
    best_ask:    i64,

    // Stats
    n_events:    u64,
    n_trades:    u64,
    volume:      i64,
}

impl LimitOrderBook {
    pub fn new(asset_id: u32, price_scale: f64, size_scale: f64) -> Self {
        Self {
            asset_id,
            price_scale,
            size_scale,
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            sequence: AtomicU64::new(0),
            last_trade_price: 0,
            last_trade_size:  0,
            best_bid:  i64::MIN,
            best_ask:  i64::MAX,
            n_events:  0,
            n_trades:  0,
            volume:    0,
        }
    }

    pub fn process(&mut self, ev: &OrderEvent) {
        self.n_events += 1;
        self.sequence.fetch_add(1, Ordering::Relaxed);

        match ev.action {
            OrderAction::Add | OrderAction::Modify => {
                let book = match ev.side {
                    Side::Bid => &mut self.bids,
                    Side::Ask => &mut self.asks,
                };
                let level = book.entry(ev.price).or_insert_with(|| PriceLevel {
                    price: ev.price, total_size: 0, n_orders: 0
                });
                if ev.action == OrderAction::Modify {
                    level.total_size = ev.size;
                } else {
                    level.total_size += ev.size;
                    level.n_orders   += 1;
                }
                if level.total_size <= 0 {
                    book.remove(&ev.price);
                }
                self.update_best(ev.side);
            }
            OrderAction::Cancel => {
                let book = match ev.side {
                    Side::Bid => &mut self.bids,
                    Side::Ask => &mut self.asks,
                };
                if let Some(level) = book.get_mut(&ev.price) {
                    level.total_size -= ev.size;
                    level.n_orders    = level.n_orders.saturating_sub(1);
                    if level.total_size <= 0 {
                        book.remove(&ev.price);
                    }
                }
                self.update_best(ev.side);
            }
            OrderAction::Trade => {
                self.last_trade_price = ev.price;
                self.last_trade_size  = ev.size;
                self.n_trades += 1;
                self.volume   += ev.size;
                // Remove traded quantity from passive side
                let book = match ev.side {
                    Side::Bid => &mut self.asks,  // buyer hits ask
                    Side::Ask => &mut self.bids,  // seller hits bid
                };
                if let Some(level) = book.get_mut(&ev.price) {
                    level.total_size -= ev.size;
                    if level.total_size <= 0 {
                        book.remove(&ev.price);
                    }
                }
                self.update_best(ev.side.opposite());
            }
        }
    }

    fn update_best(&mut self, side: Side) {
        match side {
            Side::Bid => {
                self.best_bid = self.bids.keys().next_back().copied().unwrap_or(i64::MIN);
            }
            Side::Ask => {
                self.best_ask = self.asks.keys().next().copied().unwrap_or(i64::MAX);
            }
        }
    }

    pub fn best_bid(&self) -> Option<f64> {
        if self.best_bid == i64::MIN { None }
        else { Some(self.best_bid as f64 * self.price_scale) }
    }

    pub fn best_ask(&self) -> Option<f64> {
        if self.best_ask == i64::MAX { None }
        else { Some(self.best_ask as f64 * self.price_scale) }
    }

    pub fn mid_price(&self) -> Option<f64> {
        Some((self.best_bid()? + self.best_ask()?) * 0.5)
    }

    pub fn spread(&self) -> Option<f64> {
        Some(self.best_ask()? - self.best_bid()?)
    }

    pub fn bid_levels(&self, n: usize) -> Vec<LobLevel> {
        self.bids
            .iter()
            .rev()
            .take(n)
            .map(|(_, level)| level.to_lob_level(self.price_scale))
            .collect()
    }

    pub fn ask_levels(&self, n: usize) -> Vec<LobLevel> {
        self.asks
            .iter()
            .take(n)
            .map(|(_, level)| level.to_lob_level(self.price_scale))
            .collect()
    }

    pub fn bid_depth(&self, n_levels: usize) -> f64 {
        self.bids.values().rev().take(n_levels)
            .map(|l| l.total_size as f64 * self.size_scale)
            .sum()
    }

    pub fn ask_depth(&self, n_levels: usize) -> f64 {
        self.asks.values().take(n_levels)
            .map(|l| l.total_size as f64 * self.size_scale)
            .sum()
    }

    pub fn imbalance(&self, n_levels: usize) -> f64 {
        let bd = self.bid_depth(n_levels);
        let ad = self.ask_depth(n_levels);
        let total = bd + ad;
        if total > 1e-10 { (bd - ad) / total } else { 0.0 }
    }

    pub fn vwap(&self, side: Side, n_levels: usize) -> f64 {
        let iter: Box<dyn Iterator<Item=(&i64, &PriceLevel)>> = match side {
            Side::Bid => Box::new(self.bids.iter().rev()),
            Side::Ask => Box::new(self.asks.iter()),
        };
        let (vol_sum, size_sum) = iter.take(n_levels).fold((0.0f64, 0.0f64), |(vs, ss), (_, l)| {
            let p = l.price as f64 * self.price_scale;
            let s = l.total_size as f64 * self.size_scale;
            (vs + p * s, ss + s)
        });
        if size_sum > 1e-10 { vol_sum / size_sum } else { 0.0 }
    }

    pub fn to_lob_snapshot(&self) -> crate::state_publisher::LobSnapshot {
        use crate::state_publisher::LobSnapshot;
        let bids = self.bid_levels(MAX_LOB_LEVELS);
        let asks = self.ask_levels(MAX_LOB_LEVELS);
        let mut snap = LobSnapshot {
            asset_id:       self.asset_id,
            exchange_ts_ns: now_ns(),
            sequence:       self.sequence.load(Ordering::Relaxed),
            bids,
            asks,
            ..Default::default()
        };
        snap.mid_price     = self.mid_price().unwrap_or(0.0);
        snap.spread        = self.spread().unwrap_or(0.0);
        snap.bid_imbalance = self.imbalance(MAX_LOB_LEVELS);
        snap.vwap_bid      = self.vwap(Side::Bid, MAX_LOB_LEVELS);
        snap.vwap_ask      = self.vwap(Side::Ask, MAX_LOB_LEVELS);
        snap
    }

    pub fn n_bid_levels(&self) -> usize { self.bids.len() }
    pub fn n_ask_levels(&self) -> usize { self.asks.len() }
    pub fn sequence(&self) -> u64 { self.sequence.load(Ordering::Relaxed) }
    pub fn n_events(&self) -> u64 { self.n_events }
    pub fn n_trades(&self) -> u64 { self.n_trades }
    pub fn total_volume(&self) -> f64 { self.volume as f64 * self.size_scale }

    pub fn clear(&mut self) {
        self.bids.clear();
        self.asks.clear();
        self.best_bid = i64::MIN;
        self.best_ask = i64::MAX;
    }
}

impl Side {
    fn opposite(self) -> Self {
        match self {
            Side::Bid => Side::Ask,
            Side::Ask => Side::Bid,
        }
    }
}

// ---------------------------------------------------------------------------
// MultiAssetLOB — manages order books for multiple assets
// ---------------------------------------------------------------------------
pub struct MultiAssetLOB {
    books:        std::collections::HashMap<u32, LimitOrderBook>,
    price_scale:  f64,
    size_scale:   f64,
    n_events:     u64,
}

impl MultiAssetLOB {
    pub fn new(price_scale: f64, size_scale: f64) -> Self {
        Self {
            books: std::collections::HashMap::new(),
            price_scale,
            size_scale,
            n_events: 0,
        }
    }

    pub fn process(&mut self, ev: &OrderEvent) {
        let book = self.books.entry(ev.asset_id)
            .or_insert_with(|| LimitOrderBook::new(
                ev.asset_id, self.price_scale, self.size_scale));
        book.process(ev);
        self.n_events += 1;
    }

    pub fn book(&self, asset_id: u32) -> Option<&LimitOrderBook> {
        self.books.get(&asset_id)
    }

    pub fn all_snapshots(&self) -> Vec<crate::state_publisher::LobSnapshot> {
        self.books.values()
            .map(|b| b.to_lob_snapshot())
            .collect()
    }

    pub fn n_assets(&self) -> usize { self.books.len() }
    pub fn n_events(&self) -> u64 { self.n_events }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn make_book() -> LimitOrderBook {
        LimitOrderBook::new(0, 1e-8, 1e-8)
    }

    fn add_ev(asset_id: u32, side: Side, price_cents: i64, size: i64) -> OrderEvent {
        OrderEvent {
            order_id:  1,
            timestamp: now_ns(),
            side,
            action: OrderAction::Add,
            price:  price_cents * 1_000_000,  // scale ×1e6 for cents
            size:   size * 100_000_000,        // scale ×1e8
            asset_id,
        }
    }

    #[test]
    fn test_basic_add() {
        let mut book = make_book();
        book.process(&add_ev(0, Side::Bid, 150_00, 100));  // $150.00 bid
        book.process(&add_ev(0, Side::Ask, 150_05, 100));  // $150.05 ask

        assert!(book.best_bid().is_some());
        assert!(book.best_ask().is_some());
        println!("bid={:.4} ask={:.4} mid={:.4} spread={:.6}",
                 book.best_bid().unwrap(),
                 book.best_ask().unwrap(),
                 book.mid_price().unwrap_or(0.0),
                 book.spread().unwrap_or(0.0));
    }

    #[test]
    fn test_imbalance_bid_heavy() {
        let mut book = make_book();
        // 3× more bid quantity than ask
        for _ in 0..3 {
            book.process(&add_ev(0, Side::Bid, 150_00, 100));
        }
        book.process(&add_ev(0, Side::Ask, 150_05, 100));
        let imbal = book.imbalance(10);
        assert!(imbal > 0.0, "bid-heavy book should have positive imbalance: {}", imbal);
    }

    #[test]
    fn test_cancel_removes_level() {
        let mut book = make_book();
        book.process(&add_ev(0, Side::Bid, 150_00, 100));
        assert_eq!(book.n_bid_levels(), 1);

        let cancel = OrderEvent {
            action: OrderAction::Cancel,
            size:   100 * 100_000_000,
            ..add_ev(0, Side::Bid, 150_00, 100)
        };
        book.process(&cancel);
        assert_eq!(book.n_bid_levels(), 0);
    }

    #[test]
    fn test_to_snapshot() {
        let mut book = make_book();
        for i in 1..=5i64 {
            book.process(&add_ev(0, Side::Bid, 150_00 - i, 100));
            book.process(&add_ev(0, Side::Ask, 150_05 + i, 100));
        }
        let snap = book.to_lob_snapshot();
        assert_eq!(snap.bids.len(), 5);
        assert_eq!(snap.asks.len(), 5);
        assert!(snap.mid_price > 0.0);
    }

    #[test]
    fn test_multi_asset_lob() {
        let mut mlob = MultiAssetLOB::new(1e-8, 1e-8);
        for asset_id in 0..5u32 {
            mlob.process(&add_ev(asset_id, Side::Bid, 100_00, 100));
            mlob.process(&add_ev(asset_id, Side::Ask, 100_05, 100));
        }
        assert_eq!(mlob.n_assets(), 5);
        let snaps = mlob.all_snapshots();
        assert_eq!(snaps.len(), 5);
    }
}
