/// Order-flow analytics: tick rule, signed order flow, VPIN.
///
/// Tick Rule: classify each trade as buy (+1) or sell (-1) based on
/// price movement relative to previous trade.
///
/// Signed Order Flow (OFI): Σ sign(trade) × volume
///
/// VPIN (Volume-synchronized Probability of Informed Trading):
///   Easley, López de Prado, O'Hara (2012).
///   Estimates PIN using volume buckets rather than calendar time.

use crate::streaming_stats::{RollingWindow, StreamingStats};

/// Tick-rule trade classifier.
#[derive(Debug, Clone, Default)]
pub struct TickRule {
    last_price: Option<f64>,
    last_sign:  i8,   // last assigned sign (for tied prices)
}

impl TickRule {
    pub fn new() -> Self { Self::default() }

    /// Classify a trade and return +1 (buy), -1 (sell), or 0 (unknown).
    pub fn classify(&mut self, price: f64) -> i8 {
        let sign = match self.last_price {
            None    => 0,
            Some(p) => {
                if price > p      {  1 }
                else if price < p { -1 }
                else              { self.last_sign }  // Lee-Ready: use previous sign for ties
            }
        };
        self.last_price = Some(price);
        if sign != 0 { self.last_sign = sign; }
        sign
    }

    pub fn last_sign(&self) -> i8 { self.last_sign }
    pub fn reset(&mut self) { *self = Self::default(); }
}

/// Signed order-flow imbalance tracker.
#[derive(Debug, Clone, Default)]
pub struct OrderFlowImbalance {
    buy_vol:  f64,
    sell_vol: f64,
    net_flow: f64,
    n_trades: u64,

    tick_rule: TickRule,

    /// Rolling OFI for normalisation
    ofi_stats: StreamingStats,
}

impl OrderFlowImbalance {
    pub fn new() -> Self { Self::default() }

    /// Update with a single trade. `volume` must be non-negative.
    pub fn update(&mut self, price: f64, volume: f64) {
        let sign = self.tick_rule.classify(price);
        let signed_vol = sign as f64 * volume;
        self.net_flow += signed_vol;
        self.n_trades += 1;
        if sign > 0 { self.buy_vol  += volume; }
        if sign < 0 { self.sell_vol += volume; }
    }

    /// Update with pre-classified direction (1 = buy, -1 = sell).
    pub fn update_classified(&mut self, direction: i8, volume: f64) {
        let signed_vol = direction as f64 * volume;
        self.net_flow += signed_vol;
        self.n_trades += 1;
        if direction > 0 { self.buy_vol  += volume; }
        if direction < 0 { self.sell_vol += volume; }
    }

    pub fn net_flow(&self)   -> f64  { self.net_flow }
    pub fn buy_volume(&self) -> f64  { self.buy_vol }
    pub fn sell_volume(&self)-> f64  { self.sell_vol }
    pub fn total_volume(&self) -> f64 { self.buy_vol + self.sell_vol }
    pub fn n_trades(&self)   -> u64  { self.n_trades }

    /// Buy-sell imbalance in [−1, +1].
    pub fn imbalance(&self) -> f64 {
        let tot = self.total_volume();
        if tot < 1e-15 { return 0.0; }
        (self.buy_vol - self.sell_vol) / tot
    }

    /// Record current imbalance into running stats (call at bar close).
    pub fn snapshot(&mut self) {
        self.ofi_stats.update(self.imbalance());
    }

    /// Z-score of current imbalance against historical distribution.
    pub fn imbalance_zscore(&self) -> f64 {
        let s = self.ofi_stats.std();
        if s < 1e-12 { return 0.0; }
        (self.imbalance() - self.ofi_stats.mean()) / s
    }

    pub fn reset_bar(&mut self) {
        self.buy_vol  = 0.0;
        self.sell_vol = 0.0;
        self.net_flow = 0.0;
        self.n_trades = 0;
    }
}

// ─── VPIN ───────────────────────────────────────────────────────────────────

/// Volume bucket for VPIN computation.
#[derive(Debug, Clone, Default)]
struct VpinBucket {
    buy_vol:  f64,
    sell_vol: f64,
    total:    f64,
}

impl VpinBucket {
    fn imbalance(&self) -> f64 {
        if self.total < 1e-15 { return 0.0; }
        (self.buy_vol - self.sell_vol).abs() / self.total
    }
}

/// VPIN estimator (Easley, López de Prado, O'Hara 2012).
///
/// Partitions trade volume into equal-sized buckets (each of size `bucket_vol`).
/// VPIN = mean |V_b - V_s| / bucket_vol over the last `n_buckets` buckets.
#[derive(Debug, Clone)]
pub struct Vpin {
    bucket_vol:  f64,
    n_buckets:   usize,

    current:     VpinBucket,
    current_vol: f64,

    buckets:     RollingWindow<VpinBucket>,
    tick_rule:   TickRule,
}

impl Vpin {
    /// * `bucket_vol`  — volume per bucket (e.g., ADV/50)
    /// * `n_buckets`   — rolling window of buckets for VPIN calculation
    pub fn new(bucket_vol: f64, n_buckets: usize) -> Self {
        assert!(bucket_vol > 0.0, "bucket_vol must be positive");
        assert!(n_buckets  >= 5,  "need at least 5 buckets");
        Self {
            bucket_vol,
            n_buckets,
            current:     VpinBucket::default(),
            current_vol: 0.0,
            buckets:     RollingWindow::new(n_buckets),
            tick_rule:   TickRule::new(),
        }
    }

    /// Push a trade (price, volume).
    pub fn update(&mut self, price: f64, volume: f64) {
        let sign    = self.tick_rule.classify(price);
        let buy_v   = if sign > 0 { volume } else { 0.0 };
        let sell_v  = if sign < 0 { volume } else { 0.0 };

        let mut remaining = volume;
        let mut buy_rem   = buy_v;
        let mut sell_rem  = sell_v;

        while remaining > 0.0 {
            let space = self.bucket_vol - self.current_vol;
            if remaining >= space {
                // Fill current bucket
                self.current.buy_vol  += buy_rem  * (space / remaining).min(1.0);
                self.current.sell_vol += sell_rem * (space / remaining).min(1.0);
                self.current.total    += space;

                let closed = std::mem::take(&mut self.current);
                self.buckets.push(closed);
                self.current_vol = 0.0;

                remaining -= space;
                buy_rem   -= buy_rem  * (space / (remaining + space)).min(1.0);
                sell_rem  -= sell_rem * (space / (remaining + space)).min(1.0);
            } else {
                self.current.buy_vol  += buy_rem;
                self.current.sell_vol += sell_rem;
                self.current.total    += remaining;
                self.current_vol      += remaining;
                break;
            }
        }
    }

    /// Current VPIN estimate. `None` if fewer than `n_buckets` filled.
    pub fn vpin(&self) -> Option<f64> {
        let n = self.buckets.len();
        if n < self.n_buckets { return None; }
        let sum: f64 = self.buckets.iter().map(|b| b.imbalance()).sum();
        Some(sum / n as f64)
    }

    /// Number of completed buckets so far.
    pub fn buckets_filled(&self) -> usize { self.buckets.len() }

    /// Fill fraction of the current bucket [0,1].
    pub fn bucket_fill_pct(&self) -> f64 { self.current_vol / self.bucket_vol }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_rule_basic() {
        let mut tr = TickRule::new();
        assert_eq!(tr.classify(100.0),  0);   // first tick — unknown
        assert_eq!(tr.classify(100.1),  1);   // uptick → buy
        assert_eq!(tr.classify(100.0), -1);   // downtick → sell
        assert_eq!(tr.classify(100.0), -1);   // tie → previous sign
        assert_eq!(tr.classify(100.2),  1);   // uptick → buy
    }

    #[test]
    fn ofi_imbalance_all_buys() {
        let mut ofi = OrderFlowImbalance::new();
        // All classified as buys via tick-rule (prices rising)
        let mut p = 100.0_f64;
        for _ in 0..10 {
            p += 0.01;
            ofi.update(p, 100.0);
        }
        let imb = ofi.imbalance();
        assert!(imb > 0.5, "expected strong buy imbalance, got {}", imb);
    }

    #[test]
    fn vpin_fills_buckets() {
        let mut vpin = Vpin::new(1000.0, 10);
        let mut p = 100.0_f64;
        for i in 0..200 {
            let sign = if i % 3 == 0 { -0.01 } else { 0.01 };
            p += sign;
            vpin.update(p, 100.0);
        }
        // May or may not have enough buckets — just ensure no panic
        let _ = vpin.vpin();
    }
}
