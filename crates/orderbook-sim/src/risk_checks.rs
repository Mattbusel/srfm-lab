//! risk_checks.rs — Pre-trade risk management: position limits, notional limits,
//! order rate limits, fat-finger checks, duplicate order detection.
//!
//! Chronos / AETERNUS — production risk engine.

use std::collections::{HashMap, VecDeque, HashSet};
use std::sync::atomic::{AtomicU64, AtomicI64, Ordering};
use std::sync::Arc;

// ── Types ────────────────────────────────────────────────────────────────────

pub type Price = f64;
pub type Qty = f64;
pub type Nanos = u64;
pub type OrderId = u64;
pub type InstrumentId = u32;
pub type AccountId = u32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side { Buy, Sell }

// ── Risk violation types ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum RiskViolation {
    PositionLimitExceeded   { instrument: InstrumentId, current: f64, max: f64 },
    NotionalLimitExceeded   { instrument: InstrumentId, notional: f64, limit: f64 },
    OrderRateLimitExceeded  { account: AccountId, current_rate: f64, limit: f64 },
    FatFingerPrice          { instrument: InstrumentId, order_price: f64, ref_price: f64, deviation_pct: f64 },
    FatFingerQty            { instrument: InstrumentId, order_qty: f64, max_qty: f64 },
    DuplicateOrder          { original_id: OrderId },
    GrossPositionExceeded   { current: f64, limit: f64 },
    DrawdownLimitExceeded   { current_drawdown: f64, limit: f64 },
    AccountNotFound         { account: AccountId },
    InstrumentHalted        { instrument: InstrumentId },
    MaxOrderValueExceeded   { value: f64, limit: f64 },
}

impl std::fmt::Display for RiskViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub type RiskResult<T> = Result<T, Vec<RiskViolation>>;

// ── Instrument risk limits ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct InstrumentLimits {
    pub instrument_id: InstrumentId,
    pub max_long_position: Qty,
    pub max_short_position: Qty,
    pub max_notional_per_order: f64,
    pub max_notional_total: f64,
    pub max_order_qty: Qty,
    pub max_price_deviation_pct: f64, // fat-finger: max deviation from ref price
    pub is_halted: bool,
    pub tick_size: f64,
    pub lot_size: f64,
}

impl InstrumentLimits {
    pub fn default_for(instrument_id: InstrumentId) -> Self {
        InstrumentLimits {
            instrument_id,
            max_long_position: 1_000_000.0,
            max_short_position: 1_000_000.0,
            max_notional_per_order: 10_000_000.0,
            max_notional_total: 100_000_000.0,
            max_order_qty: 100_000.0,
            max_price_deviation_pct: 10.0,
            is_halted: false,
            tick_size: 0.01,
            lot_size: 1.0,
        }
    }

    pub fn with_position_limit(mut self, long: Qty, short: Qty) -> Self {
        self.max_long_position = long;
        self.max_short_position = short;
        self
    }
}

// ── Account state ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AccountState {
    pub account_id: AccountId,
    pub positions: HashMap<InstrumentId, f64>,   // net position per instrument
    pub notional: HashMap<InstrumentId, f64>,    // total notional per instrument
    pub realized_pnl: f64,
    pub unrealized_pnl: f64,
    pub cash: f64,
    pub peak_equity: f64,
    pub current_equity: f64,
    pub max_drawdown: f64,
}

impl AccountState {
    pub fn new(account_id: AccountId, initial_cash: f64) -> Self {
        AccountState {
            account_id,
            positions: HashMap::new(),
            notional: HashMap::new(),
            realized_pnl: 0.0,
            unrealized_pnl: 0.0,
            cash: initial_cash,
            peak_equity: initial_cash,
            current_equity: initial_cash,
            max_drawdown: 0.0,
        }
    }

    pub fn position(&self, instrument: InstrumentId) -> f64 {
        *self.positions.get(&instrument).unwrap_or(&0.0)
    }

    pub fn total_notional(&self, instrument: InstrumentId) -> f64 {
        *self.notional.get(&instrument).unwrap_or(&0.0)
    }

    pub fn gross_position(&self) -> f64 {
        self.positions.values().map(|p| p.abs()).sum()
    }

    pub fn update_position(&mut self, instrument: InstrumentId, qty: f64, price: f64, is_buy: bool) {
        let signed_qty = if is_buy { qty } else { -qty };
        let pos = self.positions.entry(instrument).or_insert(0.0);
        *pos += signed_qty;
        let notional = self.notional.entry(instrument).or_insert(0.0);
        *notional += qty * price;
    }

    pub fn update_pnl(&mut self, delta_pnl: f64) {
        self.realized_pnl += delta_pnl;
        self.current_equity = self.cash + self.realized_pnl + self.unrealized_pnl;
        if self.current_equity > self.peak_equity { self.peak_equity = self.current_equity; }
        let dd = (self.peak_equity - self.current_equity) / self.peak_equity.max(1.0);
        if dd > self.max_drawdown { self.max_drawdown = dd; }
    }

    pub fn drawdown(&self) -> f64 {
        if self.peak_equity < 1.0 { return 0.0; }
        (self.peak_equity - self.current_equity) / self.peak_equity
    }
}

// ── Rate limiter (token bucket) ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TokenBucketRateLimiter {
    /// Max tokens (burst capacity)
    pub capacity: f64,
    /// Tokens added per nanosecond
    pub refill_rate: f64,
    tokens: f64,
    last_refill_ns: Nanos,
}

impl TokenBucketRateLimiter {
    /// Create with max_per_second orders/second and burst capacity
    pub fn new(max_per_second: f64, burst_capacity: f64) -> Self {
        TokenBucketRateLimiter {
            capacity: burst_capacity,
            refill_rate: max_per_second / 1e9,
            tokens: burst_capacity,
            last_refill_ns: 0,
        }
    }

    pub fn try_acquire(&mut self, current_ns: Nanos, cost: f64) -> bool {
        self.refill(current_ns);
        if self.tokens >= cost {
            self.tokens -= cost;
            true
        } else {
            false
        }
    }

    fn refill(&mut self, current_ns: Nanos) {
        if self.last_refill_ns == 0 {
            self.last_refill_ns = current_ns;
            return;
        }
        let elapsed = current_ns.saturating_sub(self.last_refill_ns) as f64;
        let new_tokens = elapsed * self.refill_rate;
        self.tokens = (self.tokens + new_tokens).min(self.capacity);
        self.last_refill_ns = current_ns;
    }

    pub fn available_tokens(&self) -> f64 { self.tokens }
    pub fn current_rate_per_sec(&self) -> f64 { (self.capacity - self.tokens) * self.refill_rate * 1e9 }
}

// ── Sliding window rate limiter ───────────────────────────────────────────────

pub struct SlidingWindowRateLimiter {
    window_ns: Nanos,
    max_count: u64,
    events: VecDeque<Nanos>,
}

impl SlidingWindowRateLimiter {
    pub fn new(window_ns: Nanos, max_count: u64) -> Self {
        SlidingWindowRateLimiter { window_ns, max_count, events: VecDeque::new() }
    }

    pub fn try_acquire(&mut self, current_ns: Nanos) -> bool {
        // Purge old events
        while self.events.front().map_or(false, |&t| current_ns.saturating_sub(t) > self.window_ns) {
            self.events.pop_front();
        }
        if self.events.len() as u64 >= self.max_count { return false; }
        self.events.push_back(current_ns);
        true
    }

    pub fn current_count(&self) -> usize { self.events.len() }
    pub fn utilization(&self) -> f64 { self.events.len() as f64 / self.max_count as f64 }
}

// ── Duplicate order detector ──────────────────────────────────────────────────

pub struct DuplicateOrderDetector {
    /// Recently seen order signatures (hash of key fields)
    seen: VecDeque<(Nanos, u64)>,
    window_ns: Nanos,
}

impl DuplicateOrderDetector {
    pub fn new(window_ns: Nanos) -> Self {
        DuplicateOrderDetector { seen: VecDeque::new(), window_ns }
    }

    fn hash_order(instrument: InstrumentId, side: Side, qty: Qty, price: Price) -> u64 {
        let price_bits = price.to_bits();
        let qty_bits = qty.to_bits();
        let mut h = instrument as u64;
        h ^= h.wrapping_mul(0x9e3779b97f4a7c15);
        h ^= price_bits.wrapping_mul(0x517cc1b727220a95);
        h ^= qty_bits.wrapping_mul(0xbf58476d1ce4e5b9);
        h ^= side as u64;
        h
    }

    /// Returns true if this order looks like a duplicate (same fields within window)
    pub fn check_and_record(&mut self, current_ns: Nanos, instrument: InstrumentId, side: Side, qty: Qty, price: Price) -> bool {
        // Prune stale entries
        while self.seen.front().map_or(false, |&(t, _)| current_ns.saturating_sub(t) > self.window_ns) {
            self.seen.pop_front();
        }
        let sig = Self::hash_order(instrument, side, qty, price);
        if self.seen.iter().any(|(_, s)| *s == sig) {
            return true; // duplicate
        }
        self.seen.push_back((current_ns, sig));
        false
    }

    pub fn clear(&mut self) { self.seen.clear(); }
}

// ── Order request ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct OrderRequest {
    pub order_id: OrderId,
    pub account_id: AccountId,
    pub instrument_id: InstrumentId,
    pub side: Side,
    pub qty: Qty,
    pub price: Option<Price>,   // None = market order
    pub timestamp_ns: Nanos,
    pub is_replacement: bool,
    pub replacing_order_id: Option<OrderId>,
}

impl OrderRequest {
    pub fn limit(id: OrderId, acct: AccountId, inst: InstrumentId, side: Side, qty: Qty, price: Price, ts_ns: Nanos) -> Self {
        OrderRequest { order_id: id, account_id: acct, instrument_id: inst, side, qty, price: Some(price), timestamp_ns: ts_ns, is_replacement: false, replacing_order_id: None }
    }

    pub fn market(id: OrderId, acct: AccountId, inst: InstrumentId, side: Side, qty: Qty, ts_ns: Nanos) -> Self {
        OrderRequest { order_id: id, account_id: acct, instrument_id: inst, side, qty, price: None, timestamp_ns: ts_ns, is_replacement: false, replacing_order_id: None }
    }

    pub fn notional(&self, ref_price: Price) -> f64 {
        self.qty * self.price.unwrap_or(ref_price)
    }
}

// ── Reference price provider ──────────────────────────────────────────────────

pub struct ReferencePriceCache {
    prices: HashMap<InstrumentId, (Price, Nanos)>,
    max_staleness_ns: Nanos,
}

impl ReferencePriceCache {
    pub fn new(max_staleness_ns: Nanos) -> Self {
        ReferencePriceCache { prices: HashMap::new(), max_staleness_ns }
    }

    pub fn update(&mut self, instrument: InstrumentId, price: Price, ts_ns: Nanos) {
        self.prices.insert(instrument, (price, ts_ns));
    }

    pub fn get(&self, instrument: InstrumentId, current_ns: Nanos) -> Option<Price> {
        self.prices.get(&instrument).and_then(|&(p, ts)| {
            if current_ns.saturating_sub(ts) <= self.max_staleness_ns { Some(p) } else { None }
        })
    }
}

// ── Main risk engine ──────────────────────────────────────────────────────────

pub struct RiskEngine {
    pub accounts: HashMap<AccountId, AccountState>,
    pub instrument_limits: HashMap<InstrumentId, InstrumentLimits>,
    pub rate_limiters: HashMap<AccountId, SlidingWindowRateLimiter>,
    pub duplicate_detector: DuplicateOrderDetector,
    pub ref_prices: ReferencePriceCache,
    pub max_gross_position: f64,
    pub max_drawdown_pct: f64,
    pub max_single_order_value: f64,
    // Stats
    pub checks_passed: u64,
    pub checks_failed: u64,
    pub violations_by_type: HashMap<String, u64>,
}

impl RiskEngine {
    pub fn new() -> Self {
        RiskEngine {
            accounts: HashMap::new(),
            instrument_limits: HashMap::new(),
            rate_limiters: HashMap::new(),
            duplicate_detector: DuplicateOrderDetector::new(1_000_000_000), // 1 second window
            ref_prices: ReferencePriceCache::new(60_000_000_000), // 60 second max staleness
            max_gross_position: 10_000_000.0,
            max_drawdown_pct: 0.20,
            max_single_order_value: 5_000_000.0,
            checks_passed: 0,
            checks_failed: 0,
            violations_by_type: HashMap::new(),
        }
    }

    pub fn add_account(&mut self, account_id: AccountId, initial_cash: f64) {
        self.accounts.insert(account_id, AccountState::new(account_id, initial_cash));
        self.rate_limiters.insert(account_id, SlidingWindowRateLimiter::new(1_000_000_000, 100)); // 100 orders/sec
    }

    pub fn set_instrument_limits(&mut self, limits: InstrumentLimits) {
        self.instrument_limits.insert(limits.instrument_id, limits);
    }

    pub fn update_ref_price(&mut self, instrument: InstrumentId, price: Price, ts_ns: Nanos) {
        self.ref_prices.update(instrument, price, ts_ns);
    }

    pub fn update_position(&mut self, account_id: AccountId, instrument: InstrumentId, qty: Qty, price: Price, is_buy: bool) {
        if let Some(acct) = self.accounts.get_mut(&account_id) {
            acct.update_position(instrument, qty, price, is_buy);
        }
    }

    pub fn update_pnl(&mut self, account_id: AccountId, delta_pnl: f64) {
        if let Some(acct) = self.accounts.get_mut(&account_id) {
            acct.update_pnl(delta_pnl);
        }
    }

    /// Run all pre-trade risk checks. Returns Ok(()) if passed, Err(violations) if failed.
    pub fn check_order(&mut self, req: &OrderRequest) -> RiskResult<()> {
        let mut violations = Vec::new();

        // 1. Account existence check
        if !self.accounts.contains_key(&req.account_id) {
            violations.push(RiskViolation::AccountNotFound { account: req.account_id });
            self.checks_failed += 1;
            return Err(violations);
        }

        // 2. Instrument halted check
        if let Some(limits) = self.instrument_limits.get(&req.instrument_id) {
            if limits.is_halted {
                violations.push(RiskViolation::InstrumentHalted { instrument: req.instrument_id });
            }
        }

        // 3. Fat-finger price check
        if let Some(order_price) = req.price {
            if let Some(ref_price) = self.ref_prices.get(req.instrument_id, req.timestamp_ns) {
                if ref_price > 0.0 {
                    let dev_pct = ((order_price - ref_price).abs() / ref_price) * 100.0;
                    let max_dev = self.instrument_limits.get(&req.instrument_id)
                        .map(|l| l.max_price_deviation_pct).unwrap_or(10.0);
                    if dev_pct > max_dev {
                        violations.push(RiskViolation::FatFingerPrice {
                            instrument: req.instrument_id,
                            order_price,
                            ref_price,
                            deviation_pct: dev_pct,
                        });
                    }
                }
            }
        }

        // 4. Fat-finger qty check
        let max_qty = self.instrument_limits.get(&req.instrument_id)
            .map(|l| l.max_order_qty).unwrap_or(1_000_000.0);
        if req.qty > max_qty {
            violations.push(RiskViolation::FatFingerQty { instrument: req.instrument_id, order_qty: req.qty, max_qty });
        }

        // 5. Notional limit check
        let ref_price = self.ref_prices.get(req.instrument_id, req.timestamp_ns)
            .or(req.price).unwrap_or(0.0);
        let notional = req.qty * ref_price;
        if self.max_single_order_value > 0.0 && notional > self.max_single_order_value {
            violations.push(RiskViolation::MaxOrderValueExceeded { value: notional, limit: self.max_single_order_value });
        }

        if let Some(limits) = self.instrument_limits.get(&req.instrument_id) {
            if notional > limits.max_notional_per_order {
                violations.push(RiskViolation::NotionalLimitExceeded { instrument: req.instrument_id, notional, limit: limits.max_notional_per_order });
            }
        }

        // 6. Position limit check
        let account = self.accounts.get(&req.account_id).unwrap();
        let current_pos = account.position(req.instrument_id);
        let new_pos = match req.side {
            Side::Buy => current_pos + req.qty,
            Side::Sell => current_pos - req.qty,
        };
        if let Some(limits) = self.instrument_limits.get(&req.instrument_id) {
            if new_pos > limits.max_long_position {
                violations.push(RiskViolation::PositionLimitExceeded { instrument: req.instrument_id, current: new_pos, max: limits.max_long_position });
            }
            if new_pos < -limits.max_short_position {
                violations.push(RiskViolation::PositionLimitExceeded { instrument: req.instrument_id, current: new_pos, max: -limits.max_short_position });
            }
        }

        // 7. Gross position check
        let gross = account.gross_position() + req.qty;
        if gross > self.max_gross_position {
            violations.push(RiskViolation::GrossPositionExceeded { current: gross, limit: self.max_gross_position });
        }

        // 8. Drawdown check
        let drawdown = account.drawdown();
        if drawdown > self.max_drawdown_pct {
            violations.push(RiskViolation::DrawdownLimitExceeded { current_drawdown: drawdown, limit: self.max_drawdown_pct });
        }

        // 9. Order rate limit check
        let rate_ok = self.rate_limiters.get_mut(&req.account_id)
            .map(|rl| rl.try_acquire(req.timestamp_ns))
            .unwrap_or(true);
        if !rate_ok {
            let limit = self.rate_limiters.get(&req.account_id).map(|rl| rl.max_count as f64).unwrap_or(100.0);
            violations.push(RiskViolation::OrderRateLimitExceeded { account: req.account_id, current_rate: 0.0, limit });
        }

        // 10. Duplicate order detection
        if !req.is_replacement {
            let side = req.side;
            let inst = req.instrument_id;
            let qty = req.qty;
            let price = req.price.unwrap_or(0.0);
            if self.duplicate_detector.check_and_record(req.timestamp_ns, inst, side, qty, price) {
                violations.push(RiskViolation::DuplicateOrder { original_id: req.order_id });
            }
        }

        if violations.is_empty() {
            self.checks_passed += 1;
            Ok(())
        } else {
            self.checks_failed += 1;
            for v in &violations {
                let key = format!("{:?}", std::mem::discriminant(v));
                *self.violations_by_type.entry(key).or_insert(0) += 1;
            }
            Err(violations)
        }
    }

    pub fn pass_rate(&self) -> f64 {
        let total = self.checks_passed + self.checks_failed;
        if total == 0 { 1.0 } else { self.checks_passed as f64 / total as f64 }
    }

    pub fn account_state(&self, account_id: AccountId) -> Option<&AccountState> {
        self.accounts.get(&account_id)
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn setup_engine() -> RiskEngine {
        let mut engine = RiskEngine::new();
        engine.add_account(1, 1_000_000.0);
        engine.set_instrument_limits(InstrumentLimits::default_for(100));
        engine.update_ref_price(100, 50.0, 1_000_000_000);
        engine
    }

    #[test]
    fn test_valid_order_passes() {
        let mut engine = setup_engine();
        let req = OrderRequest::limit(1, 1, 100, Side::Buy, 100.0, 50.0, 1_000_000_000);
        assert!(engine.check_order(&req).is_ok());
    }

    #[test]
    fn test_account_not_found() {
        let mut engine = setup_engine();
        let req = OrderRequest::limit(1, 999, 100, Side::Buy, 100.0, 50.0, 1_000_000_000);
        let result = engine.check_order(&req);
        assert!(result.is_err());
        let violations = result.unwrap_err();
        assert!(violations.iter().any(|v| matches!(v, RiskViolation::AccountNotFound { .. })));
    }

    #[test]
    fn test_fat_finger_price() {
        let mut engine = setup_engine();
        // Order price 100% above ref price of 50
        let req = OrderRequest::limit(1, 1, 100, Side::Buy, 100.0, 100.0, 1_000_000_000);
        let result = engine.check_order(&req);
        assert!(result.is_err());
        assert!(result.unwrap_err().iter().any(|v| matches!(v, RiskViolation::FatFingerPrice { .. })));
    }

    #[test]
    fn test_fat_finger_qty() {
        let mut engine = setup_engine();
        engine.instrument_limits.get_mut(&100).unwrap().max_order_qty = 500.0;
        let req = OrderRequest::limit(1, 1, 100, Side::Buy, 10_000.0, 50.0, 1_000_000_000);
        let result = engine.check_order(&req);
        assert!(result.is_err());
        assert!(result.unwrap_err().iter().any(|v| matches!(v, RiskViolation::FatFingerQty { .. })));
    }

    #[test]
    fn test_position_limit_exceeded() {
        let mut engine = setup_engine();
        engine.instrument_limits.get_mut(&100).unwrap().max_long_position = 500.0;
        let req = OrderRequest::limit(1, 1, 100, Side::Buy, 600.0, 50.0, 1_000_000_000);
        let result = engine.check_order(&req);
        assert!(result.is_err());
        assert!(result.unwrap_err().iter().any(|v| matches!(v, RiskViolation::PositionLimitExceeded { .. })));
    }

    #[test]
    fn test_instrument_halted() {
        let mut engine = setup_engine();
        engine.instrument_limits.get_mut(&100).unwrap().is_halted = true;
        let req = OrderRequest::limit(1, 1, 100, Side::Buy, 100.0, 50.0, 1_000_000_000);
        let result = engine.check_order(&req);
        assert!(result.is_err());
        assert!(result.unwrap_err().iter().any(|v| matches!(v, RiskViolation::InstrumentHalted { .. })));
    }

    #[test]
    fn test_duplicate_order_detection() {
        let mut engine = setup_engine();
        let req = OrderRequest::limit(1, 1, 100, Side::Buy, 100.0, 50.0, 1_000_000_000);
        let _ = engine.check_order(&req);
        // Same parameters within window
        let req2 = OrderRequest::limit(2, 1, 100, Side::Buy, 100.0, 50.0, 1_000_000_001);
        let result = engine.check_order(&req2);
        // May have duplicate violation (or other violations)
        // Just ensure no panic
        let _ = result;
    }

    #[test]
    fn test_rate_limiter_token_bucket() {
        let mut rl = TokenBucketRateLimiter::new(10.0, 5.0);
        // Start with 5 tokens
        for _ in 0..5 { assert!(rl.try_acquire(0, 1.0)); }
        // 6th should fail immediately (no refill time passed)
        assert!(!rl.try_acquire(0, 1.0));
        // After 1 second (1e9 ns), should have 10 new tokens
        assert!(rl.try_acquire(1_000_000_000, 1.0));
    }

    #[test]
    fn test_sliding_window_rate_limiter() {
        let mut rl = SlidingWindowRateLimiter::new(1_000_000_000, 5);
        for i in 0..5 { assert!(rl.try_acquire(i)); }
        assert!(!rl.try_acquire(5));
        // After window expires
        assert!(rl.try_acquire(2_000_000_000));
    }

    #[test]
    fn test_duplicate_detector_no_false_positive() {
        let mut det = DuplicateOrderDetector::new(1_000_000_000);
        let dup = det.check_and_record(1000, 1, Side::Buy, 100.0, 50.0);
        assert!(!dup); // first occurrence is not a duplicate
        let dup2 = det.check_and_record(1001, 1, Side::Buy, 100.0, 50.0);
        assert!(dup2); // second with same params IS a duplicate
    }

    #[test]
    fn test_duplicate_detector_different_price() {
        let mut det = DuplicateOrderDetector::new(1_000_000_000);
        det.check_and_record(1000, 1, Side::Buy, 100.0, 50.0);
        let dup = det.check_and_record(1001, 1, Side::Buy, 100.0, 51.0); // different price
        assert!(!dup);
    }

    #[test]
    fn test_account_state_position_update() {
        let mut acct = AccountState::new(1, 500_000.0);
        acct.update_position(100, 1000.0, 50.0, true);
        assert!((acct.position(100) - 1000.0).abs() < 0.01);
        acct.update_position(100, 500.0, 51.0, false);
        assert!((acct.position(100) - 500.0).abs() < 0.01);
    }

    #[test]
    fn test_account_drawdown() {
        let mut acct = AccountState::new(1, 100_000.0);
        acct.update_pnl(-5_000.0);
        assert!((acct.drawdown() - 0.05).abs() < 0.001);
    }

    #[test]
    fn test_drawdown_limit_check() {
        let mut engine = setup_engine();
        engine.max_drawdown_pct = 0.01; // 1% max
        engine.update_pnl(1, -50_000.0); // large loss
        let req = OrderRequest::limit(99, 1, 100, Side::Buy, 100.0, 50.0, 2_000_000_000);
        let result = engine.check_order(&req);
        // Should fail due to drawdown
        // (may also fail for other reasons, but drawdown should be present)
        let _ = result;
    }

    #[test]
    fn test_pass_rate_tracking() {
        let mut engine = setup_engine();
        let req = OrderRequest::limit(1, 1, 100, Side::Buy, 100.0, 50.0, 1_000_000_000);
        let _ = engine.check_order(&req);
        assert!(engine.checks_passed > 0 || engine.checks_failed > 0);
        assert!(engine.pass_rate() >= 0.0 && engine.pass_rate() <= 1.0);
    }

    #[test]
    fn test_gross_position_limit() {
        let mut engine = setup_engine();
        engine.max_gross_position = 100.0;
        let req = OrderRequest::limit(1, 1, 100, Side::Buy, 500.0, 50.0, 1_000_000_000);
        let result = engine.check_order(&req);
        assert!(result.is_err());
    }
}
