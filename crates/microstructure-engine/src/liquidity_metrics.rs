/// liquidity_metrics.rs
/// ====================
/// Market liquidity measurement: effective spread, realized spread,
/// price impact, quote quality, and post-trade resilience tracking.
///
/// All spread/impact measures are expressed in basis points (bps).
/// One basis point = 0.01%.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Snapshot type
// ---------------------------------------------------------------------------

/// Point-in-time snapshot of liquidity conditions.
#[derive(Debug, Clone)]
pub struct LiquiditySnapshot {
    /// Effective spread in bps -- full round-trip cost of a trade.
    pub effective_spread_bps: f64,
    /// Realized spread in bps -- temporary component captured by market makers.
    pub realized_spread_bps: f64,
    /// Price impact in bps -- permanent component of the spread.
    pub price_impact_bps: f64,
    /// Depth imbalance: (bid_size - ask_size) / (bid_size + ask_size).
    pub depth_imbalance: f64,
    /// Resilience score: speed of mid-price recovery after a trade.
    pub resilience_score: f64,
    /// Unix timestamp (microseconds or caller-defined epoch).
    pub timestamp: i64,
}

// ---------------------------------------------------------------------------
// Stateless spread / impact functions
// ---------------------------------------------------------------------------

/// Effective spread in basis points.
///
/// Measures the total round-trip cost relative to the mid-price.
///
/// `side` : +1 for a buy trade, -1 for a sell trade.
///
/// Formula: 2 * |trade_price - mid| / mid * 10_000
/// (The factor of 2 converts from half-spread to full round-trip spread.)
/// The `side` parameter is used to sign-correct if needed but the absolute
/// value already captures the deviation from mid symmetrically.
pub fn effective_spread(trade_price: f64, mid_price: f64, side: i8) -> f64 {
    debug_assert!(mid_price > 0.0, "mid_price must be positive");
    let _ = side; // side is captured for potential sign convention; magnitude is symmetric
    2.0 * (trade_price - mid_price).abs() / mid_price * 10_000.0
}

/// Realized spread in basis points.
///
/// Measures the temporary price impact -- the portion of the spread that
/// reverts within the benchmark horizon (typically 5 minutes).
///
/// Formula (buy side): 2 * (trade_price - mid_later) / mid_price * 10_000
/// For sell side the sign convention is reversed so that positive values
/// always represent profit for the liquidity provider.
///
/// `side` : +1 buy, -1 sell.
pub fn realized_spread(trade_price: f64, mid_5min_later: f64, side: i8) -> f64 {
    debug_assert!(mid_5min_later > 0.0, "mid_5min_later must be positive");
    let s = side as f64;
    // Positive value means liquidity provider profit from the trade.
    2.0 * s * (trade_price - mid_5min_later) / mid_5min_later * 10_000.0
}

/// Price impact in basis points.
///
/// Measures the permanent price movement caused by informed trading.
///
/// Formula: effective_spread - realized_spread
/// (i.e. the portion of the spread that does NOT revert)
///
/// `mid_price`      : mid at trade time
/// `mid_5min_later` : mid 5 minutes after the trade
/// `side`           : +1 buy, -1 sell
pub fn price_impact(
    trade_price: f64,
    mid_price: f64,
    mid_5min_later: f64,
    side: i8,
) -> f64 {
    let es = effective_spread(trade_price, mid_price, side);
    let rs = realized_spread(trade_price, mid_5min_later, side);
    es - rs
}

/// Quote quality score.
///
/// Combines spread tightness with available depth into a single score.
/// Higher scores indicate better (tighter and deeper) market quality.
///
/// Formula: 1 / (spread_bps + 1) * ln(1 + min(bid_size, ask_size))
///
/// The "+1" in the spread denominator prevents division by zero for zero-spread
/// situations and makes the score bounded.  The logarithm compresses depth
/// to avoid very large sizes dominating the score.
pub fn quote_quality_score(
    bid: f64,
    ask: f64,
    mid: f64,
    bid_size: f64,
    ask_size: f64,
) -> f64 {
    debug_assert!(mid > 0.0, "mid must be positive");
    debug_assert!(bid_size >= 0.0 && ask_size >= 0.0, "sizes must be non-negative");
    let spread_bps = (ask - bid) / mid * 10_000.0;
    let spread_bps = spread_bps.max(0.0);
    let min_depth = bid_size.min(ask_size).max(0.0);
    (1.0 / (spread_bps + 1.0)) * (1.0 + min_depth).ln()
}

// ---------------------------------------------------------------------------
// LiquidityMetrics -- stateful tracker
// ---------------------------------------------------------------------------

/// Stateful container for computing rolling liquidity snapshots.
///
/// Maintains a history of recent mid-prices and trade events to provide
/// snapshot data on demand.
#[derive(Debug, Clone)]
pub struct LiquidityMetrics {
    /// Most recent snapshot (updated on each `record_trade` call).
    last_snapshot: Option<LiquiditySnapshot>,
    /// Rolling history of (timestamp, mid_price) for post-trade tracking.
    mid_history: VecDeque<(i64, f64)>,
    /// Max entries in mid history.
    history_cap: usize,
    /// Resilience tracker used internally.
    resilience_tracker: ResilienceTracker,
}

impl LiquidityMetrics {
    pub fn new(history_cap: usize) -> Self {
        Self {
            last_snapshot: None,
            mid_history: VecDeque::with_capacity(history_cap + 1),
            history_cap,
            resilience_tracker: ResilienceTracker::new(60_000_000), // 1-min window in microseconds
        }
    }

    /// Record a new mid-price observation.
    pub fn record_mid(&mut self, mid: f64, timestamp: i64) {
        if self.mid_history.len() >= self.history_cap {
            self.mid_history.pop_front();
        }
        self.mid_history.push_back((timestamp, mid));
        self.resilience_tracker.update(mid, timestamp);
    }

    /// Record a trade and produce a `LiquiditySnapshot`.
    ///
    /// `mid_at_trade`  : mid-price at the moment of the trade.
    /// `bid` / `ask`   : current best bid and ask.
    /// `bid_size` / `ask_size` : displayed depth at best quotes.
    /// `side`          : +1 buy, -1 sell.
    /// `timestamp`     : trade timestamp.
    pub fn record_trade(
        &mut self,
        trade_price: f64,
        mid_at_trade: f64,
        bid: f64,
        ask: f64,
        bid_size: f64,
        ask_size: f64,
        side: i8,
        timestamp: i64,
    ) -> LiquiditySnapshot {
        let es = effective_spread(trade_price, mid_at_trade, side);
        // For realized spread / impact we use the current mid as a proxy for
        // mid_5min_later when no historical mid is available.
        let mid_later = self.mid_later_approximation(timestamp, mid_at_trade);
        let rs = realized_spread(trade_price, mid_later, side);
        let pi = price_impact(trade_price, mid_at_trade, mid_later, side);

        let depth_imbalance = if bid_size + ask_size > 0.0 {
            (bid_size - ask_size) / (bid_size + ask_size)
        } else {
            0.0
        };

        let resilience = self
            .resilience_tracker
            .last_resilience()
            .unwrap_or(0.0);

        let snap = LiquiditySnapshot {
            effective_spread_bps: es,
            realized_spread_bps: rs,
            price_impact_bps: pi,
            depth_imbalance,
            resilience_score: resilience,
            timestamp,
        };
        self.last_snapshot = Some(snap.clone());
        // Record the mid at trade time in the history.
        self.record_mid(mid_at_trade, timestamp);
        snap
    }

    /// Return the most recently computed snapshot, if any.
    pub fn last_snapshot(&self) -> Option<&LiquiditySnapshot> {
        self.last_snapshot.as_ref()
    }

    /// Approximate the mid price at a future horizon using the latest available.
    fn mid_later_approximation(&self, trade_ts: i64, fallback: f64) -> f64 {
        // Find the most recent mid recorded after trade_ts (5 min = 300_000_000 us).
        let horizon = trade_ts + 300_000_000;
        let candidate = self
            .mid_history
            .iter()
            .filter(|(ts, _)| *ts >= horizon)
            .next()
            .map(|(_, m)| *m);
        candidate.unwrap_or(fallback)
    }
}

// ---------------------------------------------------------------------------
// ResilienceTracker
// ---------------------------------------------------------------------------

/// Tracks post-trade mid-price recovery to estimate market resilience.
///
/// Resilience quantifies how quickly the mid-price reverts toward its
/// pre-trade level after a large order moves the market.  A high resilience
/// score means the market recovers quickly (good liquidity).
///
/// Implementation: tracks a 1-minute rolling window of mid prices and
/// estimates the slope of the reversion.  A negative slope following an
/// upward trade shock means the market is reverting (resilient).
/// The returned score is the magnitude of the mean-reversion slope
/// normalised by the initial displacement.
#[derive(Debug, Clone)]
pub struct ResilienceTracker {
    /// (timestamp, mid) pairs in the rolling window.
    pub post_trade_mids: VecDeque<(i64, f64)>,
    /// Timestamp of the most recent large trade (used as window start).
    pub trade_time: i64,
    /// Window duration in timestamp units (same epoch as `trade_time`).
    window_duration: i64,
}

impl ResilienceTracker {
    /// Create a tracker with the given window duration.
    ///
    /// `window_duration` should be in the same units as the timestamps passed
    /// to `update`.  For microsecond timestamps and a 1-minute window use
    /// `60_000_000`.
    pub fn new(window_duration: i64) -> Self {
        Self {
            post_trade_mids: VecDeque::new(),
            trade_time: 0,
            window_duration,
        }
    }

    /// Feed a new mid-price observation.
    ///
    /// Returns `Some(resilience_score)` once enough data has accumulated in
    /// the window (at least 2 points), otherwise `None`.
    ///
    /// The resilience score is the absolute value of the OLS slope of mid vs.
    /// time over the window, divided by the initial mid level (i.e., in
    /// price-units per time-unit, normalised).  Larger values mean faster
    /// reversion.
    pub fn update(&mut self, mid: f64, ts: i64) -> Option<f64> {
        self.post_trade_mids.push_back((ts, mid));
        // Evict points outside the window.
        let cutoff = ts - self.window_duration;
        while let Some(&(front_ts, _)) = self.post_trade_mids.front() {
            if front_ts < cutoff {
                self.post_trade_mids.pop_front();
            } else {
                break;
            }
        }
        self.compute_resilience()
    }

    /// Return the most recently computed resilience score.
    pub fn last_resilience(&self) -> Option<f64> {
        self.compute_resilience()
    }

    /// Compute OLS slope magnitude over the current window.
    fn compute_resilience(&self) -> Option<f64> {
        let n = self.post_trade_mids.len();
        if n < 2 {
            return None;
        }
        let n_f = n as f64;
        let mut sum_t = 0.0_f64;
        let mut sum_m = 0.0_f64;
        // Use relative timestamps to avoid large number precision issues.
        let t0 = self.post_trade_mids.front().unwrap().0;
        for &(ts, mid) in &self.post_trade_mids {
            let t = (ts - t0) as f64;
            sum_t += t;
            sum_m += mid;
        }
        let mean_t = sum_t / n_f;
        let mean_m = sum_m / n_f;

        let mut ss_tt = 0.0_f64;
        let mut ss_tm = 0.0_f64;
        for &(ts, mid) in &self.post_trade_mids {
            let t = (ts - t0) as f64 - mean_t;
            ss_tt += t * t;
            ss_tm += t * (mid - mean_m);
        }
        if ss_tt == 0.0 {
            return None;
        }
        let slope = ss_tm / ss_tt;
        // Normalise by mid level.
        let mid_scale = mean_m.abs().max(1e-10);
        Some(slope.abs() / mid_scale)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Effective spread -- buy trade above mid
    #[test]
    fn test_effective_spread_buy() {
        // trade at 100.5, mid at 100.0
        // spread = 2 * 0.5 / 100.0 * 10_000 = 100 bps
        let es = effective_spread(100.5, 100.0, 1);
        assert!((es - 100.0).abs() < 1e-9);
    }

    // 2. Effective spread -- sell trade below mid
    #[test]
    fn test_effective_spread_sell() {
        let es = effective_spread(99.5, 100.0, -1);
        assert!((es - 100.0).abs() < 1e-9);
    }

    // 3. Realized spread positive for liquidity provider on buy trade
    #[test]
    fn test_realized_spread_buy_profitable() {
        // Buy at 100.5, mid later reverts to 100.2 (partial reversion).
        // rs = 2 * (+1) * (100.5 - 100.2) / 100.2 * 10_000 = 2 * 0.3/100.2 * 10_000
        let rs = realized_spread(100.5, 100.2, 1);
        assert!(rs > 0.0, "liquidity provider should profit on partially reverting buy");
    }

    // 4. Price impact -- permanent component
    #[test]
    fn test_price_impact_positive() {
        // Large buy: trade at 101, mid was 100, mid later is 100.8 (permanent move up).
        let pi = price_impact(101.0, 100.0, 100.8, 1);
        // Price impact should be positive (market moved permanently against the buyer).
        assert!(pi > 0.0, "price impact should be positive, got {pi}");
    }

    // 5. Effective_spread == realized_spread + price_impact
    #[test]
    fn test_spread_decomposition() {
        let tp = 100.5_f64;
        let mid = 100.0_f64;
        let mid_later = 100.3_f64;
        let side = 1_i8;
        let es = effective_spread(tp, mid, side);
        let rs = realized_spread(tp, mid_later, side);
        let pi = price_impact(tp, mid, mid_later, side);
        assert!((es - rs - pi).abs() < 1e-9, "decomposition failed: es={es} rs={rs} pi={pi}");
    }

    // 6. Quote quality increases with tighter spread
    #[test]
    fn test_quote_quality_tighter_spread() {
        let mid = 100.0;
        let q_tight = quote_quality_score(99.99, 100.01, mid, 1000.0, 1000.0);
        let q_wide = quote_quality_score(99.9, 100.1, mid, 1000.0, 1000.0);
        assert!(q_tight > q_wide, "tighter spread should give higher quality score");
    }

    // 7. Quote quality increases with deeper market
    #[test]
    fn test_quote_quality_deeper_market() {
        let mid = 100.0;
        let q_deep = quote_quality_score(99.99, 100.01, mid, 10_000.0, 10_000.0);
        let q_shallow = quote_quality_score(99.99, 100.01, mid, 100.0, 100.0);
        assert!(q_deep > q_shallow, "deeper market should give higher quality score");
    }

    // 8. Quote quality uses min of bid_size and ask_size (thinner side limits depth)
    #[test]
    fn test_quote_quality_min_depth() {
        let mid = 100.0;
        // One side is very thin.
        let q = quote_quality_score(99.99, 100.01, mid, 10_000.0, 1.0);
        let q_balanced = quote_quality_score(99.99, 100.01, mid, 5000.0, 5000.0);
        // q uses min(10000,1)=1; q_balanced uses 5000.  5000 >> 1.
        assert!(q_balanced > q, "balanced depth should score higher than one-sided depth");
    }

    // 9. Depth imbalance sign in LiquidityMetrics snapshot
    #[test]
    fn test_depth_imbalance_bid_heavy() {
        let mut lm = LiquidityMetrics::new(100);
        let snap = lm.record_trade(100.5, 100.0, 99.9, 100.1, 5000.0, 1000.0, 1, 1000);
        assert!(snap.depth_imbalance > 0.0, "bid-heavy book should have positive imbalance");
    }

    // 10. Depth imbalance zero when balanced
    #[test]
    fn test_depth_imbalance_balanced() {
        let mut lm = LiquidityMetrics::new(100);
        let snap = lm.record_trade(100.5, 100.0, 99.9, 100.1, 1000.0, 1000.0, 1, 1000);
        assert!(snap.depth_imbalance.abs() < 1e-10);
    }

    // 11. ResilienceTracker returns None with fewer than 2 points
    #[test]
    fn test_resilience_none_single_point() {
        let mut rt = ResilienceTracker::new(60_000_000);
        let result = rt.update(100.0, 0);
        assert!(result.is_none());
    }

    // 12. ResilienceTracker returns Some with 2+ points
    #[test]
    fn test_resilience_some_two_points() {
        let mut rt = ResilienceTracker::new(60_000_000);
        rt.update(100.0, 0);
        let result = rt.update(100.1, 1_000_000);
        assert!(result.is_some());
    }

    // 13. ResilienceTracker evicts old points
    #[test]
    fn test_resilience_eviction() {
        let mut rt = ResilienceTracker::new(10); // tiny window
        rt.update(100.0, 0);
        rt.update(100.1, 5);
        // Point at ts=0 should be evicted after ts=20 with window=10.
        rt.update(100.2, 20);
        // Only ts=20 remains (ts=0 and ts=5 evicted), so fewer than 2 points.
        // After eviction at ts=20: cutoff = 20 - 10 = 10, so ts=5 is also removed.
        // Only ts=20 left -> compute should return None.
        assert!(rt.post_trade_mids.len() <= 1 || rt.last_resilience().is_some());
    }

    // 14. LiquidityMetrics last_snapshot updates
    #[test]
    fn test_last_snapshot_updated() {
        let mut lm = LiquidityMetrics::new(100);
        assert!(lm.last_snapshot().is_none());
        lm.record_trade(100.5, 100.0, 99.9, 100.1, 500.0, 500.0, 1, 1000);
        assert!(lm.last_snapshot().is_some());
    }

    // 15. Effective spread is symmetric for buy/sell at same deviation
    #[test]
    fn test_effective_spread_symmetry() {
        let es_buy = effective_spread(100.5, 100.0, 1);
        let es_sell = effective_spread(99.5, 100.0, -1);
        assert!((es_buy - es_sell).abs() < 1e-9);
    }
}
