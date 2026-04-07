/// trade_flow_analysis.rs
/// ======================
/// Trade flow analysis: order flow imbalance, delta divergence, VPIN,
/// aggressiveness ratio, and tick-rule classification utilities.
///
/// All computations are O(1) per tick (or O(window) for windowed queries).

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Delta divergence classification
// ---------------------------------------------------------------------------

/// Relationship between price direction and signed delta.
///
/// BullishDivergence : price fell but cumulative delta is positive
///                     (buyers absorbing the drop -- potential reversal up)
/// BearishDivergence : price rose but cumulative delta is negative
///                     (sellers absorbing the rise -- potential reversal down)
/// Confirmed         : price and delta move in the same direction
/// Neutral           : no clear price move
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaDivergence {
    BullishDivergence,
    BearishDivergence,
    Confirmed,
    Neutral,
}

// ---------------------------------------------------------------------------
// Per-tick state snapshot
// ---------------------------------------------------------------------------

/// Snapshot of trade flow state after each tick update.
#[derive(Debug, Clone)]
pub struct TradeFlowState {
    /// Total buy volume in rolling window.
    pub buy_volume: f64,
    /// Total sell volume in rolling window.
    pub sell_volume: f64,
    /// Number of buy-initiated trades in rolling window.
    pub buy_count: u64,
    /// Number of sell-initiated trades in rolling window.
    pub sell_count: u64,
    /// Volume-weighted average price of buy trades.
    pub vwap_buy: f64,
    /// Volume-weighted average price of sell trades.
    pub vwap_sell: f64,
    /// Delta = buy_volume - sell_volume for the current window.
    pub delta: f64,
    /// Cumulative delta across all ticks seen so far (not windowed).
    pub cumulative_delta: f64,
}

// ---------------------------------------------------------------------------
// Single-tick record stored in the rolling buffer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TickRecord {
    price: f64,
    volume: f64,
    is_buy: bool,
}

// ---------------------------------------------------------------------------
// TradeFlowAnalyzer
// ---------------------------------------------------------------------------

/// Maintains a rolling window of ticks and computes trade flow metrics.
///
/// `window_size` controls how many ticks are kept for windowed metrics such
/// as order flow imbalance and VPIN.  Cumulative delta is tracked globally.
#[derive(Debug, Clone)]
pub struct TradeFlowAnalyzer {
    window_size: usize,
    buffer: VecDeque<TickRecord>,
    cumulative_delta: f64,
    /// Running buy VWAP numerator (price * volume) for the window.
    buy_pv_sum: f64,
    /// Running sell VWAP numerator.
    sell_pv_sum: f64,
    /// Running buy volume for the window.
    buy_vol_sum: f64,
    /// Running sell volume for the window.
    sell_vol_sum: f64,
    /// Running buy count for the window.
    buy_count: u64,
    /// Running sell count for the window.
    sell_count: u64,
}

impl TradeFlowAnalyzer {
    /// Create a new analyzer with the given rolling window size (in ticks).
    pub fn new(window_size: usize) -> Self {
        assert!(window_size > 0, "window_size must be positive");
        Self {
            window_size,
            buffer: VecDeque::with_capacity(window_size + 1),
            cumulative_delta: 0.0,
            buy_pv_sum: 0.0,
            sell_pv_sum: 0.0,
            buy_vol_sum: 0.0,
            sell_vol_sum: 0.0,
            buy_count: 0,
            sell_count: 0,
        }
    }

    /// Incorporate a new trade tick and return the updated state snapshot.
    ///
    /// - `price`  : trade execution price
    /// - `volume` : trade size (positive)
    /// - `is_buy` : true if buyer-initiated (aggressor is buyer)
    pub fn update_tick(&mut self, price: f64, volume: f64, is_buy: bool) -> TradeFlowState {
        debug_assert!(volume > 0.0, "volume must be positive");

        // Update cumulative delta first (global, never evicted).
        if is_buy {
            self.cumulative_delta += volume;
        } else {
            self.cumulative_delta -= volume;
        }

        // Add new record to window.
        self.buffer.push_back(TickRecord { price, volume, is_buy });
        self.add_to_accumulators(price, volume, is_buy);

        // Evict oldest record if window is exceeded.
        if self.buffer.len() > self.window_size {
            let old = self.buffer.pop_front().unwrap();
            self.remove_from_accumulators(old.price, old.volume, old.is_buy);
        }

        self.snapshot()
    }

    // -- Private accumulator helpers --

    fn add_to_accumulators(&mut self, price: f64, volume: f64, is_buy: bool) {
        if is_buy {
            self.buy_vol_sum += volume;
            self.buy_pv_sum += price * volume;
            self.buy_count += 1;
        } else {
            self.sell_vol_sum += volume;
            self.sell_pv_sum += price * volume;
            self.sell_count += 1;
        }
    }

    fn remove_from_accumulators(&mut self, price: f64, volume: f64, is_buy: bool) {
        if is_buy {
            self.buy_vol_sum -= volume;
            self.buy_pv_sum -= price * volume;
            self.buy_count = self.buy_count.saturating_sub(1);
        } else {
            self.sell_vol_sum -= volume;
            self.sell_pv_sum -= price * volume;
            self.sell_count = self.sell_count.saturating_sub(1);
        }
        // Guard against floating-point drift to zero.
        if self.buy_vol_sum < 0.0 { self.buy_vol_sum = 0.0; }
        if self.sell_vol_sum < 0.0 { self.sell_vol_sum = 0.0; }
    }

    fn snapshot(&self) -> TradeFlowState {
        let vwap_buy = if self.buy_vol_sum > 0.0 {
            self.buy_pv_sum / self.buy_vol_sum
        } else {
            0.0
        };
        let vwap_sell = if self.sell_vol_sum > 0.0 {
            self.sell_pv_sum / self.sell_vol_sum
        } else {
            0.0
        };
        let delta = self.buy_vol_sum - self.sell_vol_sum;

        TradeFlowState {
            buy_volume: self.buy_vol_sum,
            sell_volume: self.sell_vol_sum,
            buy_count: self.buy_count,
            sell_count: self.sell_count,
            vwap_buy,
            vwap_sell,
            delta,
            cumulative_delta: self.cumulative_delta,
        }
    }

    // -----------------------------------------------------------------------
    // Public metrics
    // -----------------------------------------------------------------------

    /// Order Flow Imbalance over the most recent `window` ticks.
    ///
    /// OFI = (buy_vol - sell_vol) / (buy_vol + sell_vol)
    /// Returns 0 if total volume is zero.  Clamps to [-1, 1].
    pub fn order_flow_imbalance(&self, window: usize) -> f64 {
        let window = window.min(self.buffer.len());
        if window == 0 {
            return 0.0;
        }
        let start = self.buffer.len() - window;
        let mut bv = 0.0_f64;
        let mut sv = 0.0_f64;
        for record in self.buffer.iter().skip(start) {
            if record.is_buy {
                bv += record.volume;
            } else {
                sv += record.volume;
            }
        }
        let total = bv + sv;
        if total == 0.0 {
            return 0.0;
        }
        ((bv - sv) / total).clamp(-1.0, 1.0)
    }

    /// Classify the relationship between a price change and current window delta.
    ///
    /// `price_change` is the signed change in mid/trade price over the window.
    pub fn delta_divergence(&self, price_change: f64) -> DeltaDivergence {
        let delta = self.buy_vol_sum - self.sell_vol_sum;
        let price_threshold = 1e-10;
        let delta_threshold = 1e-10;

        let price_up = price_change > price_threshold;
        let price_down = price_change < -price_threshold;
        let delta_pos = delta > delta_threshold;
        let delta_neg = delta < -delta_threshold;

        if price_down && delta_pos {
            DeltaDivergence::BullishDivergence
        } else if price_up && delta_neg {
            DeltaDivergence::BearishDivergence
        } else if (price_up && delta_pos) || (price_down && delta_neg) {
            DeltaDivergence::Confirmed
        } else {
            DeltaDivergence::Neutral
        }
    }

    /// Simplified VPIN estimate using volume buckets over the current window.
    ///
    /// Divides the window into `n_buckets` equal-volume buckets and computes
    /// the average absolute imbalance fraction across buckets.
    ///
    /// Returns a value in [0, 1].  A value near 1 indicates highly informed
    /// (one-sided) order flow.
    pub fn vpin_estimate(&self, n_buckets: usize) -> f64 {
        if n_buckets == 0 || self.buffer.is_empty() {
            return 0.0;
        }
        let total_vol: f64 = self.buffer.iter().map(|r| r.volume).sum();
        if total_vol == 0.0 {
            return 0.0;
        }
        let bucket_vol = total_vol / n_buckets as f64;
        if bucket_vol == 0.0 {
            return 0.0;
        }

        let mut imbalance_sum = 0.0_f64;
        let mut bucket_buy = 0.0_f64;
        let mut bucket_sell = 0.0_f64;
        let mut bucket_total = 0.0_f64;
        let mut completed_buckets = 0usize;

        for record in &self.buffer {
            let mut remaining = record.volume;
            while remaining > 0.0 {
                let space = bucket_vol - bucket_total;
                let fill = remaining.min(space);
                if record.is_buy {
                    bucket_buy += fill;
                } else {
                    bucket_sell += fill;
                }
                bucket_total += fill;
                remaining -= fill;

                if (bucket_total - bucket_vol).abs() < 1e-12 || remaining > 0.0 {
                    // Bucket complete.
                    let bkt_sum = bucket_buy + bucket_sell;
                    if bkt_sum > 0.0 {
                        imbalance_sum += (bucket_buy - bucket_sell).abs() / bkt_sum;
                        completed_buckets += 1;
                    }
                    bucket_buy = 0.0;
                    bucket_sell = 0.0;
                    bucket_total = 0.0;
                }
            }
        }

        if completed_buckets == 0 {
            return 0.0;
        }
        (imbalance_sum / completed_buckets as f64).clamp(0.0, 1.0)
    }

    /// Aggressiveness ratio = buyer-initiated trade count / total trade count.
    ///
    /// A value above 0.5 indicates net buying aggression.
    /// Returns 0.5 when no trades are in the window.
    pub fn aggressiveness_ratio(&self) -> f64 {
        let total = self.buy_count + self.sell_count;
        if total == 0 {
            return 0.5;
        }
        self.buy_count as f64 / total as f64
    }

    /// Number of ticks currently in the rolling window.
    pub fn window_len(&self) -> usize {
        self.buffer.len()
    }

    /// Current cumulative delta (all-time, not windowed).
    pub fn cumulative_delta(&self) -> f64 {
        self.cumulative_delta
    }

    /// Current window delta (buy_vol - sell_vol in window).
    pub fn window_delta(&self) -> f64 {
        self.buy_vol_sum - self.sell_vol_sum
    }
}

// ---------------------------------------------------------------------------
// Tick-rule classification (stateless helper)
// ---------------------------------------------------------------------------

/// Classify a trade using the Lee-Ready tick rule.
///
/// Returns:
///   +1  : uptick  (curr > prev)
///   -1  : downtick (curr < prev)
///    0  : equal-tick (no change possible to determine direction)
///
/// The equal-tick case returns 0 because without external state this function
/// cannot apply the Lee-Ready continuation rule.  For stateful classification
/// use `crate::order_flow::TickRule`.
pub fn tick_rule_classify(prev_price: f64, curr_price: f64) -> i8 {
    if curr_price > prev_price {
        1
    } else if curr_price < prev_price {
        -1
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn build_analyzer(window: usize) -> TradeFlowAnalyzer {
        TradeFlowAnalyzer::new(window)
    }

    // 1. Basic update and state fields
    #[test]
    fn test_update_tick_buy() {
        let mut a = build_analyzer(10);
        let s = a.update_tick(100.0, 5.0, true);
        assert_eq!(s.buy_volume, 5.0);
        assert_eq!(s.sell_volume, 0.0);
        assert_eq!(s.buy_count, 1);
        assert_eq!(s.sell_count, 0);
        assert_eq!(s.vwap_buy, 100.0);
        assert_eq!(s.delta, 5.0);
        assert_eq!(s.cumulative_delta, 5.0);
    }

    // 2. Sell tick reduces delta
    #[test]
    fn test_update_tick_sell() {
        let mut a = build_analyzer(10);
        a.update_tick(100.0, 5.0, true);
        let s = a.update_tick(99.0, 3.0, false);
        assert_eq!(s.sell_volume, 3.0);
        assert!((s.delta - 2.0).abs() < 1e-10);
        assert!((s.cumulative_delta - 2.0).abs() < 1e-10);
    }

    // 3. Window eviction keeps window size bounded
    #[test]
    fn test_window_eviction() {
        let mut a = build_analyzer(3);
        for i in 0..5 {
            a.update_tick(100.0 + i as f64, 1.0, true);
        }
        assert_eq!(a.window_len(), 3);
    }

    // 4. VWAP computation
    #[test]
    fn test_vwap_buy() {
        let mut a = build_analyzer(10);
        a.update_tick(100.0, 2.0, true);
        let s = a.update_tick(102.0, 4.0, true);
        // VWAP = (100*2 + 102*4) / 6 = (200 + 408) / 6 = 608/6
        let expected = 608.0 / 6.0;
        assert!((s.vwap_buy - expected).abs() < 1e-9);
    }

    // 5. Order flow imbalance -- all buys
    #[test]
    fn test_ofi_all_buys() {
        let mut a = build_analyzer(10);
        for _ in 0..5 {
            a.update_tick(100.0, 1.0, true);
        }
        let ofi = a.order_flow_imbalance(5);
        assert!((ofi - 1.0).abs() < 1e-10);
    }

    // 6. Order flow imbalance -- balanced
    #[test]
    fn test_ofi_balanced() {
        let mut a = build_analyzer(10);
        for _ in 0..4 {
            a.update_tick(100.0, 1.0, true);
            a.update_tick(100.0, 1.0, false);
        }
        let ofi = a.order_flow_imbalance(8);
        assert!(ofi.abs() < 1e-10);
    }

    // 7. Delta divergence -- bullish
    #[test]
    fn test_delta_divergence_bullish() {
        let mut a = build_analyzer(10);
        // Lots of buying but price dropped.
        a.update_tick(100.0, 10.0, true);
        a.update_tick(100.0, 1.0, false);
        let div = a.delta_divergence(-0.5);
        assert_eq!(div, DeltaDivergence::BullishDivergence);
    }

    // 8. Delta divergence -- bearish
    #[test]
    fn test_delta_divergence_bearish() {
        let mut a = build_analyzer(10);
        // Lots of selling but price rose.
        a.update_tick(100.0, 1.0, true);
        a.update_tick(100.0, 10.0, false);
        let div = a.delta_divergence(0.5);
        assert_eq!(div, DeltaDivergence::BearishDivergence);
    }

    // 9. Delta divergence -- confirmed
    #[test]
    fn test_delta_divergence_confirmed() {
        let mut a = build_analyzer(10);
        a.update_tick(100.0, 5.0, true);
        a.update_tick(100.0, 1.0, false);
        let div = a.delta_divergence(0.5);
        assert_eq!(div, DeltaDivergence::Confirmed);
    }

    // 10. VPIN estimate basic sanity
    #[test]
    fn test_vpin_all_buys() {
        let mut a = build_analyzer(100);
        for _ in 0..20 {
            a.update_tick(100.0, 1.0, true);
        }
        // All one-sided: VPIN should be 1.0
        let v = a.vpin_estimate(4);
        assert!((v - 1.0).abs() < 1e-6, "expected ~1.0 got {v}");
    }

    // 11. VPIN mixed flow
    #[test]
    fn test_vpin_mixed() {
        let mut a = build_analyzer(100);
        for _ in 0..10 {
            a.update_tick(100.0, 1.0, true);
            a.update_tick(100.0, 1.0, false);
        }
        let v = a.vpin_estimate(4);
        // Perfectly balanced flow -> VPIN near 0
        assert!(v < 0.1, "expected near 0 got {v}");
    }

    // 12. Aggressiveness ratio
    #[test]
    fn test_aggressiveness_ratio() {
        let mut a = build_analyzer(10);
        a.update_tick(100.0, 1.0, true);
        a.update_tick(100.0, 1.0, true);
        a.update_tick(100.0, 1.0, false);
        let r = a.aggressiveness_ratio();
        assert!((r - 2.0 / 3.0).abs() < 1e-10);
    }

    // 13. Aggressiveness ratio when no trades
    #[test]
    fn test_aggressiveness_ratio_empty() {
        let a = build_analyzer(10);
        assert!((a.aggressiveness_ratio() - 0.5).abs() < 1e-10);
    }

    // 14. tick_rule_classify
    #[test]
    fn test_tick_rule_classify() {
        assert_eq!(tick_rule_classify(100.0, 101.0), 1);
        assert_eq!(tick_rule_classify(100.0, 99.0), -1);
        assert_eq!(tick_rule_classify(100.0, 100.0), 0);
    }

    // 15. Cumulative delta persists across window eviction
    #[test]
    fn test_cumulative_delta_persists() {
        let mut a = build_analyzer(2);
        a.update_tick(100.0, 3.0, true);  // cum = +3
        a.update_tick(100.0, 1.0, false); // cum = +2
        a.update_tick(100.0, 2.0, true);  // evict first; cum = +4
        // Window only has last 2 ticks, but cumulative should be 4
        assert!((a.cumulative_delta() - 4.0).abs() < 1e-10);
    }
}
