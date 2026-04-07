/// VPIN -- Volume-Synchronized Probability of Informed Trading.
///
/// Production implementation using tick-by-tick volume synchronization.
/// Reference: Easley, Lopez de Prado, O'Hara (2012).
///
/// Algorithm:
/// 1. Accumulate trade ticks until a fixed volume bucket is filled.
/// 2. Classify each tick as buy or sell using the tick rule (or caller-supplied side).
/// 3. For each completed bucket compute |buy_vol - sell_vol| / bucket_volume.
/// 4. VPIN = rolling mean of the last `bucket_count` bucket imbalances.
///
/// VPIN in [0, 1].  Values above 0.35 indicate elevated adverse-selection risk.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Public constants
// ---------------------------------------------------------------------------

/// Default number of buckets in the rolling window.
pub const DEFAULT_BUCKET_COUNT: usize = 50;

/// VPIN threshold above which toxicity is considered elevated.
pub const ELEVATED_TOXICITY_THRESHOLD: f64 = 0.35;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the VPIN estimator.
#[derive(Debug, Clone)]
pub struct VPINConfig {
    /// Number of volume buckets in the rolling window.
    pub bucket_count: usize,
    /// Volume per bucket.  Set automatically from total_volume / bucket_count
    /// when calling `VPINConfig::from_total_volume`, or supplied directly.
    pub volume_per_bucket: f64,
}

impl VPINConfig {
    /// Create config with explicit volume per bucket.
    pub fn new(bucket_count: usize, volume_per_bucket: f64) -> Self {
        assert!(bucket_count > 0, "bucket_count must be > 0");
        assert!(volume_per_bucket > 0.0, "volume_per_bucket must be > 0");
        VPINConfig { bucket_count, volume_per_bucket }
    }

    /// Auto-set volume_per_bucket = total_volume / bucket_count.
    pub fn from_total_volume(total_volume: f64, bucket_count: usize) -> Self {
        assert!(total_volume > 0.0, "total_volume must be > 0");
        assert!(bucket_count > 0, "bucket_count must be > 0");
        let vpb = total_volume / bucket_count as f64;
        VPINConfig { bucket_count, volume_per_bucket: vpb }
    }
}

impl Default for VPINConfig {
    fn default() -> Self {
        // Reasonable default: caller should override volume_per_bucket.
        VPINConfig { bucket_count: DEFAULT_BUCKET_COUNT, volume_per_bucket: 1000.0 }
    }
}

// ---------------------------------------------------------------------------
// Trade side
// ---------------------------------------------------------------------------

/// Direction of a single trade tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

// ---------------------------------------------------------------------------
// Volume bucket
// ---------------------------------------------------------------------------

/// A completed volume-synchronized bucket.
#[derive(Debug, Clone)]
pub struct VolumeBucket {
    /// Volume classified as buyer-initiated.
    pub buy_volume: f64,
    /// Volume classified as seller-initiated.
    pub sell_volume: f64,
    /// |buy_volume - sell_volume| / (buy_volume + sell_volume).
    pub imbalance: f64,
    /// Unix timestamp (ms) of the first trade contributing to this bucket.
    pub timestamp_start: u64,
    /// Unix timestamp (ms) of the last trade contributing to this bucket.
    pub timestamp_end: u64,
}

impl VolumeBucket {
    fn new(buy: f64, sell: f64, ts_start: u64, ts_end: u64) -> Self {
        let total = buy + sell;
        let imbalance = if total < 1e-12 { 0.0 } else { (buy - sell).abs() / total };
        VolumeBucket {
            buy_volume: buy,
            sell_volume: sell,
            imbalance: imbalance.clamp(0.0, 1.0),
            timestamp_start: ts_start,
            timestamp_end: ts_end,
        }
    }
}

// ---------------------------------------------------------------------------
// Internal state (split out so it can live inside the Mutex)
// ---------------------------------------------------------------------------

struct VPINState {
    config: VPINConfig,
    // Tick-rule state
    last_price: Option<f64>,
    last_side: Option<Side>,
    // Current in-progress bucket
    current_buy: f64,
    current_sell: f64,
    current_volume: f64,
    bucket_start_ts: u64,
    // Completed buckets (rolling window)
    buckets: VecDeque<VolumeBucket>,
    // O(1) running imbalance sum for the window
    imbalance_sum: f64,
    // Historical VPIN readings (all time, for get_history)
    vpin_history: VecDeque<f64>,
    history_capacity: usize,
    // Counters
    pub ticks_processed: u64,
    pub buckets_completed: u64,
}

impl VPINState {
    fn new(config: VPINConfig) -> Self {
        let history_capacity = config.bucket_count * 10;
        VPINState {
            config,
            last_price: None,
            last_side: None,
            current_buy: 0.0,
            current_sell: 0.0,
            current_volume: 0.0,
            bucket_start_ts: 0,
            buckets: VecDeque::new(),
            imbalance_sum: 0.0,
            vpin_history: VecDeque::new(),
            history_capacity,
            ticks_processed: 0,
            buckets_completed: 0,
        }
    }

    /// Classify trade using tick rule; caller-supplied side overrides.
    fn classify(&mut self, price: f64, supplied_side: Option<Side>) -> Side {
        if let Some(s) = supplied_side {
            self.last_side = Some(s);
            self.last_price = Some(price);
            return s;
        }
        let side = match self.last_price {
            None => Side::Buy, // first tick -- no prior context, default buy
            Some(prev) => {
                if price > prev {
                    Side::Buy
                } else if price < prev {
                    Side::Sell
                } else {
                    // Zero-tick: carry forward last non-trivial side
                    self.last_side.unwrap_or(Side::Buy)
                }
            }
        };
        self.last_price = Some(price);
        self.last_side = Some(side);
        side
    }

    /// Push one trade into the bucket accumulator.
    /// Returns the completed `VolumeBucket` if this trade caused a bucket to close,
    /// or `None` if the current bucket is still filling.
    ///
    /// Note: if a single large trade spans multiple buckets only the *last*
    /// completed bucket is returned; callers that need every bucket should
    /// use `push_trade_all`.
    fn push_trade_inner(
        &mut self,
        price: f64,
        volume: f64,
        supplied_side: Option<Side>,
        timestamp_ms: u64,
    ) -> Option<VolumeBucket> {
        self.ticks_processed += 1;

        let side = self.classify(price, supplied_side);

        if self.current_volume == 0.0 && self.bucket_start_ts == 0 {
            self.bucket_start_ts = timestamp_ms;
        }

        let mut remaining = volume;
        let mut last_completed: Option<VolumeBucket> = None;

        while remaining > 1e-12 {
            let capacity = self.config.volume_per_bucket - self.current_volume;
            let alloc = remaining.min(capacity);

            match side {
                Side::Buy => self.current_buy += alloc,
                Side::Sell => self.current_sell += alloc,
            }
            self.current_volume += alloc;
            remaining -= alloc;

            if (self.config.volume_per_bucket - self.current_volume) < 1e-9 {
                // Bucket is full -- close it
                let bucket = VolumeBucket::new(
                    self.current_buy,
                    self.current_sell,
                    self.bucket_start_ts,
                    timestamp_ms,
                );

                // Update rolling window
                self.imbalance_sum += bucket.imbalance;
                self.buckets.push_back(bucket.clone());
                if self.buckets.len() > self.config.bucket_count {
                    let evicted = self.buckets.pop_front().unwrap();
                    self.imbalance_sum -= evicted.imbalance;
                }

                // Record VPIN reading
                let v = self.vpin_raw();
                self.vpin_history.push_back(v);
                if self.vpin_history.len() > self.history_capacity {
                    self.vpin_history.pop_front();
                }

                self.buckets_completed += 1;
                last_completed = Some(bucket);

                // Reset in-progress bucket
                self.current_buy = 0.0;
                self.current_sell = 0.0;
                self.current_volume = 0.0;
                self.bucket_start_ts = if remaining > 1e-12 { timestamp_ms } else { 0 };
            }
        }

        last_completed
    }

    fn vpin_raw(&self) -> f64 {
        if self.buckets.is_empty() {
            return 0.0;
        }
        (self.imbalance_sum / self.buckets.len() as f64).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Public estimator
// ---------------------------------------------------------------------------

/// Thread-safe VPIN estimator.
///
/// Wraps internal state in `Arc<Mutex<VPINState>>` so it can be shared across
/// producer and consumer threads.
#[derive(Clone)]
pub struct VPINEstimator {
    inner: Arc<Mutex<VPINState>>,
}

impl VPINEstimator {
    /// Create a new estimator with the given configuration.
    pub fn new(config: VPINConfig) -> Self {
        VPINEstimator { inner: Arc::new(Mutex::new(VPINState::new(config))) }
    }

    /// Create with default 50-bucket config and explicit volume per bucket.
    pub fn with_volume_per_bucket(volume_per_bucket: f64) -> Self {
        let cfg = VPINConfig::new(DEFAULT_BUCKET_COUNT, volume_per_bucket);
        Self::new(cfg)
    }

    /// Push a trade.
    ///
    /// * `price`   -- trade price
    /// * `volume`  -- trade volume (always positive)
    /// * `side`    -- `Some(Side)` for pre-classified trades, `None` to use tick rule
    ///
    /// Returns the completed `VolumeBucket` if this tick closed a bucket.
    pub fn push_trade(
        &self,
        price: f64,
        volume: f64,
        side: Option<Side>,
    ) -> Option<VolumeBucket> {
        self.push_trade_ts(price, volume, side, 0)
    }

    /// Same as `push_trade` but with an explicit millisecond timestamp.
    pub fn push_trade_ts(
        &self,
        price: f64,
        volume: f64,
        side: Option<Side>,
        timestamp_ms: u64,
    ) -> Option<VolumeBucket> {
        let mut state = self.inner.lock().unwrap();
        state.push_trade_inner(price, volume, side, timestamp_ms)
    }

    /// Current VPIN over the last `bucket_count` completed buckets.
    ///
    /// Returns 0.0 until at least one bucket has been completed.
    pub fn vpin(&self) -> f64 {
        self.inner.lock().unwrap().vpin_raw()
    }

    /// Returns `true` when VPIN exceeds the elevated-toxicity threshold (0.35).
    pub fn is_elevated(&self) -> bool {
        self.vpin() > ELEVATED_TOXICITY_THRESHOLD
    }

    /// Last `n` historical VPIN readings (one per completed bucket).
    ///
    /// Returns fewer than `n` values if not enough buckets have been completed.
    pub fn get_history(&self, n: usize) -> Vec<f64> {
        let state = self.inner.lock().unwrap();
        let hist = &state.vpin_history;
        let skip = hist.len().saturating_sub(n);
        hist.iter().skip(skip).copied().collect()
    }

    /// Number of completed buckets currently in the rolling window.
    pub fn bucket_count_filled(&self) -> usize {
        self.inner.lock().unwrap().buckets.len()
    }

    /// Total number of ticks processed since creation.
    pub fn ticks_processed(&self) -> u64 {
        self.inner.lock().unwrap().ticks_processed
    }

    /// Total number of buckets completed since creation.
    pub fn buckets_completed(&self) -> u64 {
        self.inner.lock().unwrap().buckets_completed
    }

    /// Current window of completed bucket imbalances.
    pub fn bucket_imbalances(&self) -> Vec<f64> {
        self.inner.lock().unwrap().buckets.iter().map(|b| b.imbalance).collect()
    }

    /// Reset all state (e.g. at session open).
    pub fn reset(&self) {
        let mut state = self.inner.lock().unwrap();
        let config = state.config.clone();
        *state = VPINState::new(config);
    }
}

// ---------------------------------------------------------------------------
// Convenience free function
// ---------------------------------------------------------------------------

/// Compute VPIN over a slice of `(price, volume)` tick pairs using the tick rule.
///
/// Returns one VPIN value per tick (0.0 until first bucket completes).
pub fn compute_vpin_from_ticks(
    ticks: &[(f64, f64)],
    volume_per_bucket: f64,
    bucket_count: usize,
) -> Vec<f64> {
    let cfg = VPINConfig::new(bucket_count, volume_per_bucket);
    let estimator = VPINEstimator::new(cfg);
    ticks
        .iter()
        .map(|&(price, vol)| {
            estimator.push_trade(price, vol, None);
            estimator.vpin()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_estimator(vpb: f64) -> VPINEstimator {
        VPINEstimator::with_volume_per_bucket(vpb)
    }

    #[test]
    fn test_vpin_range_zero_to_one() {
        let est = make_estimator(1000.0);
        // Feed mixed trades
        for i in 0..500 {
            let price = 100.0 + (i % 10) as f64 * 0.01;
            est.push_trade(price, 100.0, None);
        }
        let v = est.vpin();
        assert!(v >= 0.0 && v <= 1.0, "VPIN out of [0,1]: {}", v);
    }

    #[test]
    fn test_vpin_all_buys_approaches_one() {
        let est = make_estimator(1000.0);
        // All up-ticks -> all buys -> imbalance = 1.0 -> VPIN -> 1.0
        for i in 0..300 {
            est.push_trade(100.0 + i as f64 * 0.01, 100.0, Some(Side::Buy));
        }
        let v = est.vpin();
        assert!(v > 0.9, "all-buy VPIN should be near 1.0, got {}", v);
    }

    #[test]
    fn test_vpin_balanced_near_zero() {
        let est = make_estimator(1000.0);
        for i in 0..500 {
            let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
            est.push_trade(100.0, 100.0, Some(side));
        }
        let v = est.vpin();
        assert!(v < 0.05, "balanced flow should give low VPIN, got {}", v);
    }

    #[test]
    fn test_vpin_elevated_detection() {
        let est = make_estimator(1000.0);
        // Pure buy flow -> elevated toxicity
        for i in 0..300 {
            est.push_trade(100.0 + i as f64 * 0.01, 100.0, Some(Side::Buy));
        }
        assert!(est.is_elevated(), "pure buy flow should be elevated");
    }

    #[test]
    fn test_vpin_not_elevated_balanced() {
        let est = make_estimator(1000.0);
        for i in 0..500 {
            let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
            est.push_trade(100.0, 100.0, Some(side));
        }
        assert!(!est.is_elevated(), "balanced flow should not be elevated");
    }

    #[test]
    fn test_bucket_completed_on_volume_fill() {
        let est = make_estimator(1000.0);
        // Exactly one bucket worth of volume in one trade
        let bucket = est.push_trade(100.0, 1000.0, Some(Side::Buy));
        assert!(bucket.is_some(), "should complete a bucket");
        assert_eq!(est.buckets_completed(), 1);
    }

    #[test]
    fn test_get_history_length_bounded() {
        let est = make_estimator(100.0);
        for i in 0..300 {
            est.push_trade(100.0 + i as f64 * 0.001, 10.0, Some(Side::Buy));
        }
        let hist = est.get_history(10);
        assert!(hist.len() <= 10);
    }

    #[test]
    fn test_get_history_values_in_range() {
        let est = make_estimator(100.0);
        for i in 0..200 {
            let side = if i % 3 == 0 { Side::Sell } else { Side::Buy };
            est.push_trade(100.0, 10.0, Some(side));
        }
        for v in est.get_history(50) {
            assert!(v >= 0.0 && v <= 1.0, "history value out of range: {}", v);
        }
    }

    #[test]
    fn test_thread_safety_concurrent_push() {
        use std::thread;
        let est = make_estimator(500.0);
        let mut handles = Vec::new();
        for _ in 0..4 {
            let est_clone = est.clone();
            handles.push(thread::spawn(move || {
                for i in 0..100 {
                    let side = if i % 2 == 0 { Side::Buy } else { Side::Sell };
                    est_clone.push_trade(100.0 + i as f64 * 0.001, 50.0, Some(side));
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        let v = est.vpin();
        assert!(v >= 0.0 && v <= 1.0);
    }

    #[test]
    fn test_reset_clears_state() {
        let est = make_estimator(100.0);
        for i in 0..200 {
            est.push_trade(100.0 + i as f64 * 0.01, 10.0, Some(Side::Buy));
        }
        assert!(est.buckets_completed() > 0);
        est.reset();
        assert_eq!(est.buckets_completed(), 0);
        assert_eq!(est.vpin(), 0.0);
    }

    #[test]
    fn test_tick_rule_uptick_buy() {
        let est = make_estimator(1000.0);
        // Seed with one trade at 100.0, then up-tick at 100.1
        est.push_trade(100.0, 10.0, None);
        let est2 = make_estimator(1000.0);
        est2.push_trade(100.0, 10.0, None);
        est2.push_trade(100.1, 10.0, None); // up-tick -> buy
        // The second estimator should have more buy volume in the current bucket
        // We verify by checking that VPIN is consistent after many up-ticks
        let est3 = make_estimator(1000.0);
        let mut price = 100.0;
        for _ in 0..300 {
            price += 0.01;
            est3.push_trade(price, 100.0, None);
        }
        let v = est3.vpin();
        assert!(v > 0.8, "all up-ticks -> high VPIN, got {}", v);
    }

    #[test]
    fn test_config_from_total_volume() {
        let cfg = VPINConfig::from_total_volume(50_000.0, 50);
        assert!((cfg.volume_per_bucket - 1000.0).abs() < 1e-9);
        assert_eq!(cfg.bucket_count, 50);
    }

    #[test]
    fn test_compute_vpin_from_ticks_length() {
        let ticks: Vec<(f64, f64)> = (0..200).map(|i| (100.0 + i as f64 * 0.01, 50.0)).collect();
        let series = compute_vpin_from_ticks(&ticks, 500.0, 10);
        assert_eq!(series.len(), 200);
    }
}
