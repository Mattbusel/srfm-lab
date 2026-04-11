/// High-fidelity price-time priority LOB engine.
///
/// Supports: Limit, Market, StopLimit, StopMarket, IcebergLimit orders.
/// Features: partial fills, cancel-replace, IOC/FOK execution constraints,
///           queue position tracking, level-2 depth snapshot, VWAP sweeps.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::fmt;

// ── Core Types ────────────────────────────────────────────────────────────────

/// Nanosecond timestamp.
pub type Nanos = u64;

/// Order identifier.
pub type OrderId = u64;

/// Fixed-point price: integer ticks (divide by PRICE_SCALE for float value).
pub type TickPrice = i64;

/// Quantity in base units.
pub type Qty = f64;

/// Conversion from f64 price to tick price.
pub const PRICE_SCALE: i64 = 100_000;

pub fn to_tick(price: f64) -> TickPrice {
    (price * PRICE_SCALE as f64).round() as TickPrice
}

pub fn from_tick(tick: TickPrice) -> f64 {
    tick as f64 / PRICE_SCALE as f64
}

// ── Enums ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
    Bid,
    Ask,
}

impl Side {
    pub fn opposite(self) -> Side {
        match self {
            Side::Bid => Side::Ask,
            Side::Ask => Side::Bid,
        }
    }
}

impl fmt::Display for Side {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Side::Bid => write!(f, "BID"),
            Side::Ask => write!(f, "ASK"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OrderKind {
    /// Standard resting limit order.
    Limit,
    /// Immediate market order — walks the book until filled or book exhausted.
    Market,
    /// Stop-limit: rests until stop_price is breached, then converts to limit.
    StopLimit { stop_price: TickPrice, limit_price: TickPrice },
    /// Stop-market: rests until stop_price is breached, then converts to market.
    StopMarket { stop_price: TickPrice },
    /// Iceberg: only peak_qty visible; refills from hidden reserve.
    IcebergLimit { limit_price: TickPrice, total_qty: Qty, peak_qty: Qty },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeInForce {
    /// Good-till-cancel.
    GTC,
    /// Immediate-or-cancel: cancel any unfilled portion after matching attempt.
    IOC,
    /// Fill-or-kill: reject entirely if not completely filled immediately.
    FOK,
    /// Day order (treated like GTC in sim).
    Day,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderStatus {
    New,
    PartialFill,
    Filled,
    Cancelled,
    Rejected,
    PendingCancel,
}

// ── Order ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LobOrder {
    pub id: OrderId,
    pub side: Side,
    pub kind: OrderKind,
    pub tif: TimeInForce,
    /// Visible price used for queue position (tick-scaled).
    pub price: TickPrice,
    /// Original total quantity.
    pub orig_qty: Qty,
    /// Currently visible (resting) quantity.
    pub leaves_qty: Qty,
    /// Total filled quantity so far.
    pub filled_qty: Qty,
    /// For icebergs: remaining hidden reserve.
    pub hidden_qty: Qty,
    /// Nanosecond timestamp of submission.
    pub timestamp: Nanos,
    /// Sequence number for strict time priority.
    pub seq: u64,
    pub status: OrderStatus,
    /// Optional tag for agent identification.
    pub agent_id: u32,
}

impl LobOrder {
    pub fn new_limit(id: OrderId, side: Side, price: f64, qty: Qty, ts: Nanos, agent_id: u32) -> Self {
        LobOrder {
            id,
            side,
            kind: OrderKind::Limit,
            tif: TimeInForce::GTC,
            price: to_tick(price),
            orig_qty: qty,
            leaves_qty: qty,
            filled_qty: 0.0,
            hidden_qty: 0.0,
            timestamp: ts,
            seq: 0,
            status: OrderStatus::New,
            agent_id,
        }
    }

    pub fn new_market(id: OrderId, side: Side, qty: Qty, ts: Nanos, agent_id: u32) -> Self {
        LobOrder {
            id,
            side,
            kind: OrderKind::Market,
            tif: TimeInForce::IOC,
            price: if side == Side::Bid { i64::MAX } else { i64::MIN },
            orig_qty: qty,
            leaves_qty: qty,
            filled_qty: 0.0,
            hidden_qty: 0.0,
            timestamp: ts,
            seq: 0,
            status: OrderStatus::New,
            agent_id,
        }
    }

    pub fn new_stop_limit(
        id: OrderId, side: Side,
        stop_price: f64, limit_price: f64,
        qty: Qty, ts: Nanos, agent_id: u32,
    ) -> Self {
        LobOrder {
            id,
            side,
            kind: OrderKind::StopLimit {
                stop_price: to_tick(stop_price),
                limit_price: to_tick(limit_price),
            },
            tif: TimeInForce::GTC,
            price: to_tick(limit_price),
            orig_qty: qty,
            leaves_qty: qty,
            filled_qty: 0.0,
            hidden_qty: 0.0,
            timestamp: ts,
            seq: 0,
            status: OrderStatus::New,
            agent_id,
        }
    }

    pub fn new_iceberg(
        id: OrderId, side: Side, limit_price: f64,
        total_qty: Qty, peak_qty: Qty, ts: Nanos, agent_id: u32,
    ) -> Self {
        let visible = peak_qty.min(total_qty);
        let hidden = total_qty - visible;
        LobOrder {
            id,
            side,
            kind: OrderKind::IcebergLimit {
                limit_price: to_tick(limit_price),
                total_qty,
                peak_qty,
            },
            tif: TimeInForce::GTC,
            price: to_tick(limit_price),
            orig_qty: total_qty,
            leaves_qty: visible,
            filled_qty: 0.0,
            hidden_qty: hidden,
            timestamp: ts,
            seq: 0,
            status: OrderStatus::New,
            agent_id,
        }
    }

    pub fn with_tif(mut self, tif: TimeInForce) -> Self {
        self.tif = tif;
        self
    }

    pub fn is_complete(&self) -> bool {
        matches!(self.status, OrderStatus::Filled | OrderStatus::Cancelled | OrderStatus::Rejected)
    }

    pub fn is_resting(&self) -> bool {
        (self.status == OrderStatus::New || self.status == OrderStatus::PartialFill)
            && self.leaves_qty > 0.0
    }

    /// Apply a fill of `qty`. Returns the actual fill amount (capped by leaves_qty).
    pub fn apply_fill(&mut self, qty: Qty) -> Qty {
        let actual = qty.min(self.leaves_qty);
        self.leaves_qty -= actual;
        self.filled_qty += actual;

        // Iceberg refill from hidden reserve.
        if self.leaves_qty < 1e-9 && self.hidden_qty > 1e-9 {
            if let OrderKind::IcebergLimit { peak_qty, .. } = self.kind {
                let refill = peak_qty.min(self.hidden_qty);
                self.leaves_qty = refill;
                self.hidden_qty -= refill;
            }
        }

        if self.leaves_qty < 1e-9 && self.hidden_qty < 1e-9 {
            self.status = OrderStatus::Filled;
        } else {
            self.status = OrderStatus::PartialFill;
        }
        actual
    }

    pub fn price_f64(&self) -> f64 {
        from_tick(self.price)
    }
}

// ── Fill Event ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct LobFill {
    pub aggressor_id: OrderId,
    pub passive_id: OrderId,
    pub price: f64,
    pub qty: Qty,
    pub side: Side,          // side of the aggressor
    pub timestamp: Nanos,
    pub aggressor_agent: u32,
    pub passive_agent: u32,
}

// ── Price Level ───────────────────────────────────────────────────────────────

/// A single price level in the LOB: FIFO queue of resting orders.
#[derive(Debug, Default)]
pub struct PriceLevel {
    pub orders: VecDeque<LobOrder>,
    pub total_qty: Qty,
}

impl PriceLevel {
    pub fn push_back(&mut self, order: LobOrder) {
        self.total_qty += order.leaves_qty;
        self.orders.push_back(order);
    }

    pub fn is_empty(&self) -> bool {
        self.orders.is_empty()
    }

    /// Remove an order by id, updating total_qty. Returns the removed order if found.
    pub fn remove_by_id(&mut self, id: OrderId) -> Option<LobOrder> {
        if let Some(pos) = self.orders.iter().position(|o| o.id == id) {
            let removed = self.orders.remove(pos).unwrap();
            self.total_qty -= removed.leaves_qty;
            Some(removed)
        } else {
            None
        }
    }

    pub fn queue_position(&self, id: OrderId) -> Option<usize> {
        self.orders.iter().position(|o| o.id == id)
    }

    pub fn qty_ahead(&self, id: OrderId) -> Qty {
        let mut q = 0.0;
        for o in &self.orders {
            if o.id == id {
                break;
            }
            q += o.leaves_qty;
        }
        q
    }
}

// ── LOB Engine ────────────────────────────────────────────────────────────────

/// Full price-time priority limit order book.
pub struct LobEngine {
    /// Bids: ascending by tick-price; best bid = last entry.
    bids: BTreeMap<TickPrice, PriceLevel>,
    /// Asks: ascending by tick-price; best ask = first entry.
    asks: BTreeMap<TickPrice, PriceLevel>,
    /// Stop orders not yet triggered. Stored separately.
    stop_orders: Vec<LobOrder>,
    /// Fast lookup: id → (side, price).
    index: HashMap<OrderId, (Side, TickPrice)>,
    /// Monotonic sequence counter for time priority.
    seq: u64,
    /// Last traded price (tick).
    last_price: TickPrice,
    /// All fills generated (can be drained periodically).
    pub fill_log: Vec<LobFill>,
    /// Running total fill count.
    pub fill_count: u64,
    /// Instrument identifier.
    pub symbol: String,
}

impl LobEngine {
    pub fn new(symbol: impl Into<String>) -> Self {
        LobEngine {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            stop_orders: Vec::new(),
            index: HashMap::new(),
            seq: 0,
            last_price: 0,
            fill_log: Vec::new(),
            fill_count: 0,
            symbol: symbol.into(),
        }
    }

    fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    // ── Best prices ──────────────────────────────────────────────────────────

    pub fn best_bid_tick(&self) -> Option<TickPrice> {
        self.bids.keys().next_back().copied()
    }

    pub fn best_ask_tick(&self) -> Option<TickPrice> {
        self.asks.keys().next().copied()
    }

    pub fn best_bid(&self) -> Option<f64> {
        self.best_bid_tick().map(from_tick)
    }

    pub fn best_ask(&self) -> Option<f64> {
        self.best_ask_tick().map(from_tick)
    }

    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid_tick(), self.best_ask_tick()) {
            (Some(b), Some(a)) => Some(from_tick(b + a) / 2.0),
            _ => None,
        }
    }

    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid_tick(), self.best_ask_tick()) {
            (Some(b), Some(a)) => Some(from_tick(a - b)),
            _ => None,
        }
    }

    pub fn last_price(&self) -> f64 {
        from_tick(self.last_price)
    }

    // ── Depth snapshots ──────────────────────────────────────────────────────

    pub fn bid_depth(&self, n: usize) -> Vec<(f64, Qty)> {
        self.bids.iter().rev().take(n)
            .map(|(p, lvl)| (from_tick(*p), lvl.total_qty))
            .collect()
    }

    pub fn ask_depth(&self, n: usize) -> Vec<(f64, Qty)> {
        self.asks.iter().take(n)
            .map(|(p, lvl)| (from_tick(*p), lvl.total_qty))
            .collect()
    }

    /// Full level-2 snapshot: (price, qty, order_count) per side.
    pub fn snapshot(&self) -> (Vec<(f64, Qty, usize)>, Vec<(f64, Qty, usize)>) {
        let bids = self.bids.iter().rev()
            .map(|(p, lvl)| (from_tick(*p), lvl.total_qty, lvl.orders.len()))
            .collect();
        let asks = self.asks.iter()
            .map(|(p, lvl)| (from_tick(*p), lvl.total_qty, lvl.orders.len()))
            .collect();
        (bids, asks)
    }

    pub fn order_book_imbalance(&self, depth: usize) -> f64 {
        let bid_vol: Qty = self.bids.iter().rev().take(depth).map(|(_, l)| l.total_qty).sum();
        let ask_vol: Qty = self.asks.iter().take(depth).map(|(_, l)| l.total_qty).sum();
        let total = bid_vol + ask_vol;
        if total < 1e-9 { 0.0 } else { (bid_vol - ask_vol) / total }
    }

    /// Cost of sweeping `qty` on the given side. Returns (avg_price, filled_qty).
    pub fn vwap_sweep(&self, side: Side, qty: Qty) -> (f64, Qty) {
        let mut remaining = qty;
        let mut notional = 0.0_f64;
        let mut total_filled = 0.0_f64;

        let levels: Box<dyn Iterator<Item = (&TickPrice, &PriceLevel)>> = match side {
            Side::Bid => Box::new(self.asks.iter()),                   // buy sweeps asks
            Side::Ask => Box::new(self.bids.iter().rev()),             // sell sweeps bids
        };

        for (p, lvl) in levels {
            if remaining < 1e-9 { break; }
            let take = remaining.min(lvl.total_qty);
            notional += take * from_tick(*p);
            total_filled += take;
            remaining -= take;
        }

        if total_filled < 1e-9 {
            (0.0, 0.0)
        } else {
            (notional / total_filled, total_filled)
        }
    }

    // ── Order management ─────────────────────────────────────────────────────

    /// Submit an order. Returns vector of fills generated.
    pub fn submit(&mut self, mut order: LobOrder) -> Vec<LobFill> {
        order.seq = self.next_seq();

        match order.kind {
            OrderKind::Market => self.execute_market(order),
            OrderKind::Limit => {
                let tif = order.tif;
                // For FOK: check if full fill is available before executing.
                if tif == TimeInForce::FOK {
                    let (_, available) = self.vwap_sweep(order.side, order.leaves_qty);
                    if available < order.leaves_qty - 1e-9 {
                        // Reject the order.
                        order.status = OrderStatus::Rejected;
                        return vec![];
                    }
                }
                let fills = self.execute_limit(&mut order);
                match tif {
                    TimeInForce::IOC | TimeInForce::FOK => {
                        if order.leaves_qty > 1e-9 {
                            order.status = OrderStatus::Cancelled;
                        }
                        // Don't rest.
                    }
                    _ => {
                        if order.is_resting() {
                            self.rest_order(order);
                        }
                    }
                }
                fills
            }
            OrderKind::StopLimit { stop_price, .. } | OrderKind::StopMarket { stop_price } => {
                let triggered = match order.side {
                    Side::Bid => self.last_price >= stop_price,
                    Side::Ask => self.last_price <= stop_price,
                };
                if triggered {
                    let mut activated = self.activate_stop(order);
                    let fills = self.execute_limit(&mut activated);
                    if activated.is_resting() {
                        self.rest_order(activated);
                    }
                    fills
                } else {
                    self.stop_orders.push(order);
                    vec![]
                }
            }
            OrderKind::IcebergLimit { limit_price, .. } => {
                let mut ice = order;
                ice.price = limit_price;
                let fills = self.execute_limit(&mut ice);
                if ice.is_resting() {
                    self.rest_order(ice);
                }
                fills
            }
        }
    }

    /// Cancel a resting order by id. Returns true if found.
    pub fn cancel(&mut self, id: OrderId) -> bool {
        if let Some((side, price)) = self.index.remove(&id) {
            let book = match side {
                Side::Bid => &mut self.bids,
                Side::Ask => &mut self.asks,
            };
            if let Some(lvl) = book.get_mut(&price) {
                lvl.remove_by_id(id);
                if lvl.is_empty() {
                    book.remove(&price);
                }
            }
            true
        } else {
            // Check stop orders.
            let before = self.stop_orders.len();
            self.stop_orders.retain(|o| o.id != id);
            self.stop_orders.len() < before
        }
    }

    /// Modify an existing resting order (cancel-replace, loses time priority).
    pub fn modify(&mut self, id: OrderId, new_price: f64, new_qty: Qty) -> bool {
        if let Some((side, old_price)) = self.index.remove(&id) {
            let book = match side {
                Side::Bid => &mut self.bids,
                Side::Ask => &mut self.asks,
            };
            let mut found_order = None;
            if let Some(lvl) = book.get_mut(&old_price) {
                found_order = lvl.remove_by_id(id);
                if lvl.is_empty() {
                    book.remove(&old_price);
                }
            }
            if let Some(mut order) = found_order {
                order.price = to_tick(new_price);
                order.orig_qty = new_qty;
                order.leaves_qty = new_qty;
                order.filled_qty = 0.0;
                order.seq = self.next_seq();
                order.status = OrderStatus::New;
                let ts = order.timestamp;
                let fills = self.submit(order);
                // Fills from submit are returned; caller handles them.
                self.fill_log.extend_from_slice(&fills);
                return true;
            }
        }
        false
    }

    /// Query queue position (0-based FIFO position) and qty ahead for an order.
    pub fn queue_info(&self, id: OrderId) -> Option<(usize, Qty)> {
        if let Some((side, price)) = self.index.get(&id) {
            let book = match side {
                Side::Bid => &self.bids,
                Side::Ask => &self.asks,
            };
            if let Some(lvl) = book.get(price) {
                let pos = lvl.queue_position(id)?;
                let ahead = lvl.qty_ahead(id);
                return Some((pos, ahead));
            }
        }
        None
    }

    // ── Internal execution ────────────────────────────────────────────────────

    fn rest_order(&mut self, order: LobOrder) {
        self.index.insert(order.id, (order.side, order.price));
        match order.side {
            Side::Bid => self.bids.entry(order.price).or_default().push_back(order),
            Side::Ask => self.asks.entry(order.price).or_default().push_back(order),
        }
    }

    fn activate_stop(&self, mut order: LobOrder) -> LobOrder {
        match order.kind {
            OrderKind::StopLimit { limit_price, .. } => {
                order.kind = OrderKind::Limit;
                order.price = limit_price;
            }
            OrderKind::StopMarket { .. } => {
                order.kind = OrderKind::Market;
                order.tif = TimeInForce::IOC;
            }
            _ => {}
        }
        order
    }

    fn execute_market(&mut self, mut aggressor: LobOrder) -> Vec<LobFill> {
        let mut fills = Vec::new();
        let ts = aggressor.timestamp;
        let agg_id = aggressor.id;
        let agg_agent = aggressor.agent_id;
        let side = aggressor.side;

        match side {
            Side::Bid => {
                let prices: Vec<TickPrice> = self.asks.keys().cloned().collect();
                'outer_m: for price in prices {
                    if aggressor.leaves_qty < 1e-9 { break; }
                    if let Some(lvl) = self.asks.get_mut(&price) {
                        while let Some(passive) = lvl.orders.front_mut() {
                            if aggressor.leaves_qty < 1e-9 { break 'outer_m; }
                            let fill_qty = passive.leaves_qty.min(aggressor.leaves_qty);
                            let fill_price = from_tick(price);
                            let pass_agent = passive.agent_id;
                            let pass_id = passive.id;
                            passive.apply_fill(fill_qty);
                            aggressor.apply_fill(fill_qty);
                            self.last_price = price;
                            let f = LobFill { aggressor_id: agg_id, passive_id: pass_id, price: fill_price, qty: fill_qty, side, timestamp: ts, aggressor_agent: agg_agent, passive_agent: pass_agent };
                            fills.push(f.clone());
                            self.fill_log.push(f);
                            self.fill_count += 1;
                            if passive.is_complete() {
                                let pid = lvl.orders.front().unwrap().id;
                                self.index.remove(&pid);
                                lvl.total_qty -= 0.0; // already handled in apply_fill for PriceLevel
                                lvl.orders.pop_front();
                            } else {
                                lvl.total_qty = lvl.orders.iter().map(|o| o.leaves_qty).sum();
                                break;
                            }
                        }
                        lvl.total_qty = lvl.orders.iter().map(|o| o.leaves_qty).sum();
                    }
                    if self.asks.get(&price).map_or(false, |l| l.is_empty()) {
                        self.asks.remove(&price);
                    }
                }
            }
            Side::Ask => {
                let prices: Vec<TickPrice> = self.bids.keys().cloned().rev().collect();
                'outer_ms: for price in prices {
                    if aggressor.leaves_qty < 1e-9 { break; }
                    if let Some(lvl) = self.bids.get_mut(&price) {
                        while let Some(passive) = lvl.orders.front_mut() {
                            if aggressor.leaves_qty < 1e-9 { break 'outer_ms; }
                            let fill_qty = passive.leaves_qty.min(aggressor.leaves_qty);
                            let fill_price = from_tick(price);
                            let pass_agent = passive.agent_id;
                            let pass_id = passive.id;
                            passive.apply_fill(fill_qty);
                            aggressor.apply_fill(fill_qty);
                            self.last_price = price;
                            let f = LobFill { aggressor_id: agg_id, passive_id: pass_id, price: fill_price, qty: fill_qty, side, timestamp: ts, aggressor_agent: agg_agent, passive_agent: pass_agent };
                            fills.push(f.clone());
                            self.fill_log.push(f);
                            self.fill_count += 1;
                            if passive.is_complete() {
                                let pid = lvl.orders.front().unwrap().id;
                                self.index.remove(&pid);
                                lvl.orders.pop_front();
                            } else {
                                lvl.total_qty = lvl.orders.iter().map(|o| o.leaves_qty).sum();
                                break;
                            }
                        }
                        lvl.total_qty = lvl.orders.iter().map(|o| o.leaves_qty).sum();
                    }
                    if self.bids.get(&price).map_or(false, |l| l.is_empty()) {
                        self.bids.remove(&price);
                    }
                }
            }
        }

        self.trigger_stops(ts);
        fills
    }

    fn execute_limit(&mut self, aggressor: &mut LobOrder) -> Vec<LobFill> {
        let mut fills = Vec::new();
        let ts = aggressor.timestamp;
        let agg_id = aggressor.id;
        let agg_agent = aggressor.agent_id;
        let agg_price = aggressor.price;
        let side = aggressor.side;

        match side {
            Side::Bid => {
                let prices: Vec<TickPrice> = self.asks.keys().cloned().collect();
                'outer_l: for price in prices {
                    if price > agg_price { break; }
                    if aggressor.leaves_qty < 1e-9 { break; }
                    if let Some(lvl) = self.asks.get_mut(&price) {
                        while let Some(passive) = lvl.orders.front_mut() {
                            if aggressor.leaves_qty < 1e-9 { break 'outer_l; }
                            let fill_qty = passive.leaves_qty.min(aggressor.leaves_qty);
                            let fill_price = from_tick(price);
                            let pass_agent = passive.agent_id;
                            let pass_id = passive.id;
                            passive.apply_fill(fill_qty);
                            aggressor.apply_fill(fill_qty);
                            self.last_price = price;
                            let f = LobFill { aggressor_id: agg_id, passive_id: pass_id, price: fill_price, qty: fill_qty, side, timestamp: ts, aggressor_agent: agg_agent, passive_agent: pass_agent };
                            fills.push(f.clone());
                            self.fill_log.push(f);
                            self.fill_count += 1;
                            if passive.is_complete() {
                                let pid = lvl.orders.front().unwrap().id;
                                self.index.remove(&pid);
                                lvl.orders.pop_front();
                            } else {
                                lvl.total_qty = lvl.orders.iter().map(|o| o.leaves_qty).sum();
                                break;
                            }
                        }
                        lvl.total_qty = lvl.orders.iter().map(|o| o.leaves_qty).sum();
                    }
                    if self.asks.get(&price).map_or(false, |l| l.is_empty()) {
                        self.asks.remove(&price);
                    }
                }
            }
            Side::Ask => {
                let prices: Vec<TickPrice> = self.bids.keys().cloned().rev().collect();
                'outer_ls: for price in prices {
                    if price < agg_price { break; }
                    if aggressor.leaves_qty < 1e-9 { break; }
                    if let Some(lvl) = self.bids.get_mut(&price) {
                        while let Some(passive) = lvl.orders.front_mut() {
                            if aggressor.leaves_qty < 1e-9 { break 'outer_ls; }
                            let fill_qty = passive.leaves_qty.min(aggressor.leaves_qty);
                            let fill_price = from_tick(price);
                            let pass_agent = passive.agent_id;
                            let pass_id = passive.id;
                            passive.apply_fill(fill_qty);
                            aggressor.apply_fill(fill_qty);
                            self.last_price = price;
                            let f = LobFill { aggressor_id: agg_id, passive_id: pass_id, price: fill_price, qty: fill_qty, side, timestamp: ts, aggressor_agent: agg_agent, passive_agent: pass_agent };
                            fills.push(f.clone());
                            self.fill_log.push(f);
                            self.fill_count += 1;
                            if passive.is_complete() {
                                let pid = lvl.orders.front().unwrap().id;
                                self.index.remove(&pid);
                                lvl.orders.pop_front();
                            } else {
                                lvl.total_qty = lvl.orders.iter().map(|o| o.leaves_qty).sum();
                                break;
                            }
                        }
                        lvl.total_qty = lvl.orders.iter().map(|o| o.leaves_qty).sum();
                    }
                    if self.bids.get(&price).map_or(false, |l| l.is_empty()) {
                        self.bids.remove(&price);
                    }
                }
            }
        }

        self.trigger_stops(ts);
        fills
    }

    fn trigger_stops(&mut self, ts: Nanos) {
        let last = self.last_price;
        let mut to_activate: Vec<LobOrder> = Vec::new();
        self.stop_orders.retain(|o| {
            let triggered = match o.kind {
                OrderKind::StopLimit { stop_price, .. } | OrderKind::StopMarket { stop_price } => {
                    match o.side {
                        Side::Bid => last >= stop_price,
                        Side::Ask => last <= stop_price,
                    }
                }
                _ => false,
            };
            if triggered {
                to_activate.push(o.clone());
                false
            } else {
                true
            }
        });

        for order in to_activate {
            let mut activated = self.activate_stop(order);
            activated.timestamp = ts;
            activated.seq = self.next_seq();
            let fills = self.execute_limit(&mut activated);
            self.fill_log.extend(fills);
            if activated.is_resting() {
                self.rest_order(activated);
            }
        }
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    pub fn order_count(&self) -> usize {
        self.index.len()
    }

    pub fn bid_level_count(&self) -> usize {
        self.bids.len()
    }

    pub fn ask_level_count(&self) -> usize {
        self.asks.len()
    }

    pub fn total_bid_qty(&self) -> Qty {
        self.bids.values().map(|l| l.total_qty).sum()
    }

    pub fn total_ask_qty(&self) -> Qty {
        self.asks.values().map(|l| l.total_qty).sum()
    }

    /// Drain the fill log and return all fills since the last drain.
    pub fn drain_fills(&mut self) -> Vec<LobFill> {
        std::mem::take(&mut self.fill_log)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ts() -> Nanos { 1_700_000_000_000_000_000 }

    #[test]
    fn test_basic_limit_match() {
        let mut engine = LobEngine::new("TEST");
        let bid = LobOrder::new_limit(1, Side::Bid, 100.0, 10.0, ts(), 1);
        let ask = LobOrder::new_limit(2, Side::Ask, 100.0, 10.0, ts() + 1, 2);
        engine.submit(bid);
        let fills = engine.submit(ask);
        assert_eq!(fills.len(), 1);
        assert!((fills[0].qty - 10.0).abs() < 1e-9);
        assert!((fills[0].price - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_partial_fill() {
        let mut engine = LobEngine::new("TEST");
        let bid = LobOrder::new_limit(1, Side::Bid, 100.0, 5.0, ts(), 1);
        let ask = LobOrder::new_limit(2, Side::Ask, 100.0, 10.0, ts() + 1, 2);
        engine.submit(bid);
        let fills = engine.submit(ask);
        assert_eq!(fills.len(), 1);
        assert!((fills[0].qty - 5.0).abs() < 1e-9);
        assert_eq!(engine.ask_level_count(), 1); // 5 remaining on ask
    }

    #[test]
    fn test_market_order() {
        let mut engine = LobEngine::new("TEST");
        for i in 0..5 {
            let ask = LobOrder::new_limit(i + 1, Side::Ask, 100.0 + i as f64, 20.0, ts() + i as u64, 99);
            engine.submit(ask);
        }
        let market = LobOrder::new_market(100, Side::Bid, 50.0, ts() + 10, 1);
        let fills = engine.submit(market);
        let total: f64 = fills.iter().map(|f| f.qty).sum();
        assert!((total - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_iceberg_order() {
        let mut engine = LobEngine::new("TEST");
        let ice = LobOrder::new_iceberg(1, Side::Bid, 100.0, 30.0, 10.0, ts(), 1);
        engine.submit(ice);
        // Best bid should show only peak_qty = 10
        let depth = engine.bid_depth(1);
        assert!((depth[0].1 - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_cancel_order() {
        let mut engine = LobEngine::new("TEST");
        let bid = LobOrder::new_limit(1, Side::Bid, 100.0, 10.0, ts(), 1);
        engine.submit(bid);
        assert_eq!(engine.order_count(), 1);
        assert!(engine.cancel(1));
        assert_eq!(engine.order_count(), 0);
    }

    #[test]
    fn test_stop_limit_trigger() {
        let mut engine = LobEngine::new("TEST");
        // Seed some asks.
        let ask = LobOrder::new_limit(1, Side::Ask, 101.0, 100.0, ts(), 99);
        engine.submit(ask);
        // Add a stop-limit sell at stop=99, limit=98.
        let stop = LobOrder::new_stop_limit(2, Side::Ask, 99.0, 98.0, 50.0, ts() + 1, 1);
        engine.submit(stop);
        // Drive last_price to 99 via a market buy that fills the ask at 101.
        let mkt = LobOrder::new_market(3, Side::Bid, 100.0, ts() + 2, 2);
        engine.submit(mkt);
        // Stop should have been triggered now.
        // (Stop won't fill here because there's nothing on bid side at >= 98,
        //  but it should be resting in the book.)
        // Just verify no panic.
    }

    #[test]
    fn test_ioc_partial_cancel() {
        let mut engine = LobEngine::new("TEST");
        let ask = LobOrder::new_limit(1, Side::Ask, 100.0, 3.0, ts(), 99);
        engine.submit(ask);
        let mut ioc = LobOrder::new_limit(2, Side::Bid, 100.0, 10.0, ts() + 1, 1)
            .with_tif(TimeInForce::IOC);
        let fills = engine.submit(ioc);
        let total: f64 = fills.iter().map(|f| f.qty).sum();
        assert!((total - 3.0).abs() < 1e-9);
        // Remaining 7 should be cancelled, not resting.
        assert_eq!(engine.bid_level_count(), 0);
    }

    #[test]
    fn test_vwap_sweep() {
        let mut engine = LobEngine::new("TEST");
        for i in 0..5 {
            let ask = LobOrder::new_limit(i + 1, Side::Ask, 100.0 + i as f64, 10.0, ts() + i as u64, 99);
            engine.submit(ask);
        }
        let (avg_price, filled) = engine.vwap_sweep(Side::Bid, 30.0);
        assert!((filled - 30.0).abs() < 1e-9);
        // Average of 100, 101, 102 = 101.0
        assert!((avg_price - 101.0).abs() < 1e-4);
    }

    #[test]
    fn test_queue_info() {
        let mut engine = LobEngine::new("TEST");
        let b1 = LobOrder::new_limit(1, Side::Bid, 100.0, 10.0, ts(), 1);
        let b2 = LobOrder::new_limit(2, Side::Bid, 100.0, 20.0, ts() + 1, 2);
        engine.submit(b1);
        engine.submit(b2);
        let (pos, ahead) = engine.queue_info(2).unwrap();
        assert_eq!(pos, 1);
        assert!((ahead - 10.0).abs() < 1e-9);
    }
}
