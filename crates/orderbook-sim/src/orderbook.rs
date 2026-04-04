use std::collections::{BTreeMap, HashMap, VecDeque};
use chrono::{DateTime, Utc};
use ordered_float::OrderedFloat;
use crate::fills::Fill;
use crate::order::{Order, OrderSide, OrderStatus, OrderType};

/// Full Level-2 order book with price-time priority matching.
pub struct OrderBook {
    /// Bids: highest price first (max-heap semantics via reverse BTreeMap).
    /// Key = OrderedFloat(price), stored in ascending order; we iterate from end.
    bids: BTreeMap<OrderedFloat<f64>, VecDeque<Order>>,
    /// Asks: lowest price first.
    asks: BTreeMap<OrderedFloat<f64>, VecDeque<Order>>,
    /// Fast lookup: order_id → (side, price).
    order_index: HashMap<u64, (OrderSide, f64)>,
    /// All-time fill log.
    pub fills: Vec<Fill>,
    /// Stop orders waiting to be triggered.
    stop_orders: Vec<Order>,
    /// Last traded price (used to trigger stop orders).
    last_price: f64,
    /// Global sequence counter for time priority.
    seq: u64,
}

impl OrderBook {
    pub fn new() -> Self {
        OrderBook {
            bids: BTreeMap::new(),
            asks: BTreeMap::new(),
            order_index: HashMap::new(),
            fills: Vec::new(),
            stop_orders: Vec::new(),
            last_price: 0.0,
            seq: 0,
        }
    }

    fn next_seq(&mut self) -> u64 {
        self.seq += 1;
        self.seq
    }

    // ── Best prices ──────────────────────────────────────────────────────────

    pub fn best_bid(&self) -> Option<f64> {
        self.bids.keys().next_back().map(|k| k.0)
    }

    pub fn best_ask(&self) -> Option<f64> {
        self.asks.keys().next().map(|k| k.0)
    }

    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some((b + a) / 2.0),
            _ => None,
        }
    }

    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some(a - b),
            _ => None,
        }
    }

    // ── Depth ─────────────────────────────────────────────────────────────────

    /// Top N bid levels as (price, cumulative_qty) sorted best-first.
    pub fn bid_depth(&self, n: usize) -> Vec<(f64, f64)> {
        self.bids
            .iter()
            .rev()
            .take(n)
            .map(|(k, q)| (k.0, q.iter().map(|o| o.remaining_qty).sum()))
            .collect()
    }

    /// Top N ask levels as (price, cumulative_qty) sorted best-first.
    pub fn ask_depth(&self, n: usize) -> Vec<(f64, f64)> {
        self.asks
            .iter()
            .take(n)
            .map(|(k, q)| (k.0, q.iter().map(|o| o.remaining_qty).sum()))
            .collect()
    }

    // ── VWAP for a given notional sweep ──────────────────────────────────────

    /// Compute the VWAP price to fill `qty` on the ask side (i.e., a buy sweep).
    pub fn vwap_buy(&self, qty: f64) -> Option<f64> {
        let mut remaining = qty;
        let mut notional = 0.0;
        for (price_key, queue) in &self.asks {
            for order in queue {
                let take = order.remaining_qty.min(remaining);
                notional += take * price_key.0;
                remaining -= take;
                if remaining <= 0.0 {
                    return Some(notional / qty);
                }
            }
        }
        None // insufficient liquidity
    }

    /// Compute the VWAP price to fill `qty` on the bid side (i.e., a sell sweep).
    pub fn vwap_sell(&self, qty: f64) -> Option<f64> {
        let mut remaining = qty;
        let mut notional = 0.0;
        for (price_key, queue) in self.bids.iter().rev() {
            for order in queue {
                let take = order.remaining_qty.min(remaining);
                notional += take * price_key.0;
                remaining -= take;
                if remaining <= 0.0 {
                    return Some(notional / qty);
                }
            }
        }
        None
    }

    // ── Order-book imbalance ─────────────────────────────────────────────────

    /// Imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) over top N levels.
    /// Returns value in [-1, 1]: +1 = pure bid pressure, -1 = pure ask pressure.
    pub fn imbalance(&self, depth_levels: usize) -> f64 {
        let bid_vol: f64 = self
            .bids
            .iter()
            .rev()
            .take(depth_levels)
            .flat_map(|(_, q)| q.iter().map(|o| o.remaining_qty))
            .sum();
        let ask_vol: f64 = self
            .asks
            .iter()
            .take(depth_levels)
            .flat_map(|(_, q)| q.iter().map(|o| o.remaining_qty))
            .sum();
        let total = bid_vol + ask_vol;
        if total == 0.0 {
            0.0
        } else {
            (bid_vol - ask_vol) / total
        }
    }

    // ── Add / Cancel / Modify ────────────────────────────────────────────────

    /// Add an order to the book, running matching if required.
    /// Returns any fills generated.
    pub fn add_order(&mut self, mut order: Order) -> Vec<Fill> {
        order.sequence = self.next_seq();

        match order.order_type {
            OrderType::Market => self.match_market(order),
            OrderType::Limit => {
                let fills = self.match_limit(&mut order);
                if !order.is_complete() {
                    self.rest_order(order);
                }
                fills
            }
            OrderType::StopLimit { stop_price, .. } => {
                // Queue stop order; it will be activated when last price crosses stop.
                let triggered = match order.side {
                    OrderSide::Buy => self.last_price >= stop_price,
                    OrderSide::Sell => self.last_price <= stop_price,
                };
                if triggered {
                    // Convert to limit immediately.
                    let mut limit = order.clone();
                    limit.order_type = OrderType::Limit;
                    let fills = self.match_limit(&mut limit);
                    if !limit.is_complete() {
                        self.rest_order(limit);
                    }
                    fills
                } else {
                    self.stop_orders.push(order);
                    vec![]
                }
            }
        }
    }

    /// Cancel an existing resting order by ID. Returns true if found.
    pub fn cancel_order(&mut self, order_id: u64) -> bool {
        if let Some((side, price)) = self.order_index.remove(&order_id) {
            let key = OrderedFloat(price);
            let book_side = match side {
                OrderSide::Buy => &mut self.bids,
                OrderSide::Sell => &mut self.asks,
            };
            if let Some(queue) = book_side.get_mut(&key) {
                queue.retain(|o| o.id != order_id);
                if queue.is_empty() {
                    book_side.remove(&key);
                }
            }
            true
        } else {
            false
        }
    }

    /// Modify price/qty of a resting order (cancel-and-replace, preserving time priority loss).
    pub fn modify_order(&mut self, order_id: u64, new_price: f64, new_qty: f64) -> bool {
        // Find and remove old order.
        if let Some((side, old_price)) = self.order_index.remove(&order_id) {
            let key = OrderedFloat(old_price);
            let book_side = match side {
                OrderSide::Buy => &mut self.bids,
                OrderSide::Sell => &mut self.asks,
            };
            let maybe_order = if let Some(queue) = book_side.get_mut(&key) {
                let pos = queue.iter().position(|o| o.id == order_id);
                pos.map(|p| queue.remove(p).unwrap())
            } else {
                None
            };
            if let Some(queue) = book_side.get(&key) {
                if queue.is_empty() {
                    let k = key;
                    match side {
                        OrderSide::Buy => { self.bids.remove(&k); }
                        OrderSide::Sell => { self.asks.remove(&k); }
                    }
                }
            }
            if let Some(mut order) = maybe_order {
                order.price = new_price;
                order.qty = new_qty;
                order.remaining_qty = new_qty;
                // Re-insert with new sequence (loses time priority).
                order.sequence = self.next_seq();
                self.rest_order(order);
                return true;
            }
        }
        false
    }

    // ── Internal matching ─────────────────────────────────────────────────────

    fn rest_order(&mut self, order: Order) {
        let key = OrderedFloat(order.price);
        self.order_index.insert(order.id, (order.side, order.price));
        match order.side {
            OrderSide::Buy => self.bids.entry(key).or_default().push_back(order),
            OrderSide::Sell => self.asks.entry(key).or_default().push_back(order),
        }
    }

    /// Match a market order against the book. Consumes as much as available.
    fn match_market(&mut self, mut aggressor: Order) -> Vec<Fill> {
        let mut fills = Vec::new();
        let ts = aggressor.timestamp;
        let agg_id = aggressor.id;
        let side = aggressor.side;

        match aggressor.side {
            OrderSide::Buy => {
                // Walk up asks (lowest first).
                let prices: Vec<OrderedFloat<f64>> = self.asks.keys().cloned().collect();
                'outer: for price_key in prices {
                    if let Some(queue) = self.asks.get_mut(&price_key) {
                        while let Some(passive) = queue.front_mut() {
                            if aggressor.remaining_qty <= 0.0 {
                                break 'outer;
                            }
                            let fill_qty = passive.remaining_qty.min(aggressor.remaining_qty);
                            let fill_price = passive.price;
                            passive.apply_fill(fill_qty);
                            aggressor.apply_fill(fill_qty);
                            self.last_price = fill_price;
                            let f = Fill::new(agg_id, passive.id, fill_price, fill_qty, ts, side);
                            fills.push(f.clone());
                            self.fills.push(f);
                            if passive.is_complete() {
                                let pid = passive.id;
                                queue.pop_front();
                                self.order_index.remove(&pid);
                            }
                        }
                    }
                    if self.asks.get(&price_key).map_or(false, |q| q.is_empty()) {
                        self.asks.remove(&price_key);
                    }
                    if aggressor.remaining_qty <= 0.0 {
                        break;
                    }
                }
            }
            OrderSide::Sell => {
                let prices: Vec<OrderedFloat<f64>> = self.bids.keys().cloned().rev().collect();
                'outer: for price_key in prices {
                    if let Some(queue) = self.bids.get_mut(&price_key) {
                        while let Some(passive) = queue.front_mut() {
                            if aggressor.remaining_qty <= 0.0 {
                                break 'outer;
                            }
                            let fill_qty = passive.remaining_qty.min(aggressor.remaining_qty);
                            let fill_price = passive.price;
                            passive.apply_fill(fill_qty);
                            aggressor.apply_fill(fill_qty);
                            self.last_price = fill_price;
                            let f = Fill::new(agg_id, passive.id, fill_price, fill_qty, ts, side);
                            fills.push(f.clone());
                            self.fills.push(f);
                            if passive.is_complete() {
                                let pid = passive.id;
                                queue.pop_front();
                                self.order_index.remove(&pid);
                            }
                        }
                    }
                    if self.bids.get(&price_key).map_or(false, |q| q.is_empty()) {
                        self.bids.remove(&price_key);
                    }
                    if aggressor.remaining_qty <= 0.0 {
                        break;
                    }
                }
            }
        }

        // Trigger any stop orders that may have crossed.
        self.check_stops(ts);
        fills
    }

    /// Match a limit order. Returns fills; caller is responsible for resting remainder.
    fn match_limit(&mut self, aggressor: &mut Order) -> Vec<Fill> {
        let mut fills = Vec::new();
        let ts = aggressor.timestamp;
        let agg_id = aggressor.id;
        let agg_price = aggressor.price;
        let side = aggressor.side;

        match side {
            OrderSide::Buy => {
                let prices: Vec<OrderedFloat<f64>> = self.asks.keys().cloned().collect();
                for price_key in prices {
                    if price_key.0 > agg_price {
                        break; // No more matchable levels.
                    }
                    if aggressor.remaining_qty <= 0.0 {
                        break;
                    }
                    if let Some(queue) = self.asks.get_mut(&price_key) {
                        while let Some(passive) = queue.front_mut() {
                            if aggressor.remaining_qty <= 0.0 {
                                break;
                            }
                            let fill_qty = passive.remaining_qty.min(aggressor.remaining_qty);
                            let fill_price = passive.price;
                            passive.apply_fill(fill_qty);
                            aggressor.apply_fill(fill_qty);
                            self.last_price = fill_price;
                            let f = Fill::new(agg_id, passive.id, fill_price, fill_qty, ts, side);
                            fills.push(f.clone());
                            self.fills.push(f);
                            if passive.is_complete() {
                                let pid = passive.id;
                                queue.pop_front();
                                self.order_index.remove(&pid);
                            }
                        }
                    }
                    if self.asks.get(&price_key).map_or(false, |q| q.is_empty()) {
                        self.asks.remove(&price_key);
                    }
                }
            }
            OrderSide::Sell => {
                let prices: Vec<OrderedFloat<f64>> = self.bids.keys().cloned().rev().collect();
                for price_key in prices {
                    if price_key.0 < agg_price {
                        break;
                    }
                    if aggressor.remaining_qty <= 0.0 {
                        break;
                    }
                    if let Some(queue) = self.bids.get_mut(&price_key) {
                        while let Some(passive) = queue.front_mut() {
                            if aggressor.remaining_qty <= 0.0 {
                                break;
                            }
                            let fill_qty = passive.remaining_qty.min(aggressor.remaining_qty);
                            let fill_price = passive.price;
                            passive.apply_fill(fill_qty);
                            aggressor.apply_fill(fill_qty);
                            self.last_price = fill_price;
                            let f = Fill::new(agg_id, passive.id, fill_price, fill_qty, ts, side);
                            fills.push(f.clone());
                            self.fills.push(f);
                            if passive.is_complete() {
                                let pid = passive.id;
                                queue.pop_front();
                                self.order_index.remove(&pid);
                            }
                        }
                    }
                    if self.bids.get(&price_key).map_or(false, |q| q.is_empty()) {
                        self.bids.remove(&price_key);
                    }
                }
            }
        }
        self.check_stops(ts);
        fills
    }

    /// Check stop orders and trigger any that have crossed the last price.
    fn check_stops(&mut self, ts: DateTime<Utc>) {
        let last = self.last_price;
        let mut to_activate: Vec<Order> = Vec::new();
        self.stop_orders.retain(|o| {
            let triggered = match o.side {
                OrderSide::Buy => {
                    if let OrderType::StopLimit { stop_price, .. } = o.order_type {
                        last >= stop_price
                    } else {
                        false
                    }
                }
                OrderSide::Sell => {
                    if let OrderType::StopLimit { stop_price, .. } = o.order_type {
                        last <= stop_price
                    } else {
                        false
                    }
                }
            };
            if triggered {
                to_activate.push(o.clone());
                false
            } else {
                true
            }
        });

        for mut order in to_activate {
            order.order_type = OrderType::Limit;
            order.timestamp = ts;
            order.sequence = self.next_seq();
            let fills = self.match_limit(&mut order);
            self.fills.extend(fills);
            if !order.is_complete() {
                self.rest_order(order);
            }
        }
    }

    // ── Utility ───────────────────────────────────────────────────────────────

    pub fn order_count(&self) -> usize {
        self.order_index.len()
    }

    pub fn bid_level_count(&self) -> usize {
        self.bids.len()
    }

    pub fn ask_level_count(&self) -> usize {
        self.asks.len()
    }

    /// Total resting bid volume.
    pub fn total_bid_qty(&self) -> f64 {
        self.bids.values().flat_map(|q| q.iter().map(|o| o.remaining_qty)).sum()
    }

    /// Total resting ask volume.
    pub fn total_ask_qty(&self) -> f64 {
        self.asks.values().flat_map(|q| q.iter().map(|o| o.remaining_qty)).sum()
    }
}

impl Default for OrderBook {
    fn default() -> Self {
        Self::new()
    }
}
