/// VPIN -- Volume-Synchronized Probability of Informed Trading.
///
/// Algorithm:
/// 1. Divide total volume into equal-sized buckets of V_bucket units.
/// 2. For each bucket, classify volume as buy or sell (using BVC from tick_classifier).
/// 3. Compute |buy_vol - sell_vol| / V_bucket for each bucket.
/// 4. VPIN = rolling mean of the last `n_buckets` bucket imbalances.
///
/// Interpretation:
/// - VPIN in [0, 1]; higher = more informed trading activity.
/// - VPIN > 0.5 -> elevated adverse selection risk; be cautious about entering.
/// - Entry filter: skip entries when VPIN > 0.4.

pub const VPIN_INFORMED_THRESHOLD: f64 = 0.5;
pub const VPIN_ENTRY_FILTER_THRESHOLD: f64 = 0.4;

/// A completed volume bucket with its imbalance.
#[derive(Debug, Clone)]
pub struct VpinBucket {
    pub buy_volume: f64,
    pub sell_volume: f64,
    pub bucket_volume: f64,
}

impl VpinBucket {
    /// Absolute volume imbalance normalized by bucket size.
    pub fn imbalance(&self) -> f64 {
        if self.bucket_volume < 1e-9 {
            return 0.0;
        }
        ((self.buy_volume - self.sell_volume).abs() / self.bucket_volume).min(1.0)
    }
}

/// VPIN calculator.
///
/// Processes a stream of (buy_volume, sell_volume, bar_total_volume) tuples.
/// Fills volume buckets and tracks rolling VPIN over `n_buckets` buckets.
pub struct VpinCalculator {
    /// Target volume per bucket
    bucket_volume: f64,
    /// Number of buckets to average
    n_buckets: usize,
    /// Volume remaining to fill the current bucket
    remaining_in_bucket: f64,
    /// Buy volume accumulated in current bucket
    current_buy: f64,
    /// Sell volume accumulated in current bucket
    current_sell: f64,
    /// Completed buckets (circular)
    completed: std::collections::VecDeque<VpinBucket>,
    /// Running sum of imbalances for O(1) mean
    imbalance_sum: f64,
    /// Total bars processed
    pub bars_processed: u64,
    /// Total buckets completed
    pub buckets_completed: u64,
}

impl VpinCalculator {
    /// Create a new VPIN calculator.
    ///
    /// * `bucket_volume`  -- volume per bucket (e.g., average daily volume / 50)
    /// * `n_buckets`      -- number of buckets for rolling average (e.g., 50)
    pub fn new(bucket_volume: f64, n_buckets: usize) -> Self {
        assert!(bucket_volume > 0.0);
        assert!(n_buckets > 0);
        VpinCalculator {
            bucket_volume,
            n_buckets,
            remaining_in_bucket: bucket_volume,
            current_buy: 0.0,
            current_sell: 0.0,
            completed: std::collections::VecDeque::new(),
            imbalance_sum: 0.0,
            bars_processed: 0,
            buckets_completed: 0,
        }
    }

    /// Feed one bar's classified volume. Returns updated VPIN (or None if fewer
    /// than n_buckets have been completed).
    pub fn push(&mut self, buy_vol: f64, sell_vol: f64) -> Option<f64> {
        self.bars_processed += 1;
        let mut remaining_buy = buy_vol;
        let mut remaining_sell = sell_vol;
        let mut new_bucket_completed = false;

        // Distribute bar volume across (possibly multiple) buckets
        loop {
            let total_remaining = remaining_buy + remaining_sell;
            if total_remaining <= 0.0 {
                break;
            }

            let fraction = (self.remaining_in_bucket / total_remaining).min(1.0);
            let alloc_buy = remaining_buy * fraction;
            let alloc_sell = remaining_sell * fraction;

            self.current_buy += alloc_buy;
            self.current_sell += alloc_sell;
            self.remaining_in_bucket -= alloc_buy + alloc_sell;

            remaining_buy -= alloc_buy;
            remaining_sell -= alloc_sell;

            // Bucket is full
            if self.remaining_in_bucket <= 1e-9 {
                let bucket = VpinBucket {
                    buy_volume: self.current_buy,
                    sell_volume: self.current_sell,
                    bucket_volume: self.bucket_volume,
                };
                let imb = bucket.imbalance();
                self.imbalance_sum += imb;
                self.completed.push_back(bucket);
                self.buckets_completed += 1;

                // Evict oldest bucket if over window
                if self.completed.len() > self.n_buckets {
                    let old = self.completed.pop_front().unwrap();
                    self.imbalance_sum -= old.imbalance();
                }

                // Reset bucket
                self.current_buy = 0.0;
                self.current_sell = 0.0;
                self.remaining_in_bucket = self.bucket_volume;
                new_bucket_completed = true;
            }
        }

        let _ = new_bucket_completed;

        if self.completed.len() >= self.n_buckets {
            Some(self.vpin())
        } else {
            // Not enough buckets yet; return partial estimate if > 0
            if self.completed.is_empty() {
                None
            } else {
                Some(self.vpin_partial())
            }
        }
    }

    /// VPIN: mean imbalance over the last `n_buckets` completed buckets.
    pub fn vpin(&self) -> f64 {
        if self.completed.is_empty() {
            return 0.0;
        }
        (self.imbalance_sum / self.completed.len() as f64).clamp(0.0, 1.0)
    }

    /// Partial VPIN: uses fewer than n_buckets (early in time series).
    fn vpin_partial(&self) -> f64 {
        self.vpin()
    }

    /// Is informed trading elevated? (VPIN > threshold)
    pub fn is_informed_trading(&self) -> bool {
        self.vpin() > VPIN_INFORMED_THRESHOLD
    }

    /// Should we suppress entry? (VPIN > entry filter threshold)
    pub fn should_filter_entry(&self) -> bool {
        self.vpin() > VPIN_ENTRY_FILTER_THRESHOLD
    }

    /// Completed bucket count currently in the rolling window.
    pub fn bucket_count(&self) -> usize {
        self.completed.len()
    }

    /// Recent bucket imbalances for inspection.
    pub fn bucket_imbalances(&self) -> Vec<f64> {
        self.completed.iter().map(|b| b.imbalance()).collect()
    }
}

/// Compute VPIN for a full series of (buy_vol, sell_vol) pairs.
/// Returns a Vec of VPIN values (None entries replaced with 0.0).
pub fn compute_vpin_series(
    classified_bars: &[(f64, f64)], // (buy_vol, sell_vol)
    bucket_volume: f64,
    n_buckets: usize,
) -> Vec<f64> {
    let mut calc = VpinCalculator::new(bucket_volume, n_buckets);
    classified_bars
        .iter()
        .map(|&(b, s)| calc.push(b, s).unwrap_or(0.0))
        .collect()
}

/// Estimate a reasonable bucket volume from an OHLCV series.
/// Uses median bar volume / 2 as a starting point.
pub fn estimate_bucket_volume(volumes: &[f64]) -> f64 {
    if volumes.is_empty() {
        return 1000.0;
    }
    let mut sorted = volumes.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];
    (median / 2.0).max(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_bars(n: usize, buy: f64, sell: f64) -> Vec<(f64, f64)> {
        vec![(buy, sell); n]
    }

    #[test]
    fn test_vpin_all_informed() {
        // All buys -> max imbalance -> VPIN = 1.0
        let bars = uniform_bars(200, 1000.0, 0.0);
        let series = compute_vpin_series(&bars, 1000.0, 10);
        let last = *series.last().unwrap();
        assert!((last - 1.0).abs() < 0.05, "expected VPIN~1, got {}", last);
    }

    #[test]
    fn test_vpin_balanced() {
        let bars = uniform_bars(200, 500.0, 500.0);
        let series = compute_vpin_series(&bars, 1000.0, 10);
        let last = *series.last().unwrap();
        assert!(last < 0.1, "balanced flow -> low VPIN, got {}", last);
    }

    #[test]
    fn test_vpin_range() {
        let bars: Vec<(f64, f64)> = (0..100)
            .map(|i| {
                let b = (i as f64 % 10.0) * 100.0;
                let s = 1000.0 - b;
                (b, s)
            })
            .collect();
        let series = compute_vpin_series(&bars, 500.0, 5);
        for &v in &series {
            assert!(v >= 0.0 && v <= 1.0, "VPIN out of range: {}", v);
        }
    }

    #[test]
    fn test_entry_filter() {
        let mut calc = VpinCalculator::new(1000.0, 5);
        // Push strongly imbalanced bars
        for _ in 0..50 {
            calc.push(900.0, 100.0);
        }
        assert!(calc.should_filter_entry(), "high VPIN should filter entry");
    }

    #[test]
    fn test_bucket_count_bounded() {
        let mut calc = VpinCalculator::new(1000.0, 10);
        for _ in 0..200 {
            calc.push(500.0, 500.0);
        }
        assert!(calc.bucket_count() <= 10);
    }

    #[test]
    fn test_estimate_bucket_volume() {
        let volumes: Vec<f64> = (1..=100).map(|x| x as f64 * 100.0).collect();
        let bv = estimate_bucket_volume(&volumes);
        assert!(bv > 0.0);
        // Median of 100..10000 is ~5050, half = ~2525
        assert!(bv > 1000.0 && bv < 5000.0);
    }
}
