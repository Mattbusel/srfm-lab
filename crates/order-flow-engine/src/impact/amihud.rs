/// Amihud Illiquidity Ratio.
///
/// The Amihud (2002) illiquidity measure is:
///
///   ILLIQ_t = |R_t| / DV_t
///
/// where R_t is the absolute return for bar t and DV_t is the dollar volume.
/// A rolling mean of this ratio estimates the average price impact per unit
/// of dollar trading volume.
///
/// Higher values signal lower market liquidity.
///
/// Reference:
///   Amihud, Y. (2002). "Illiquidity and stock returns: cross-section and
///   time-series effects." Journal of Financial Markets 5: 31-56.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Estimator
// ---------------------------------------------------------------------------

/// Amihud illiquidity estimator with rolling mean and z-score support.
pub struct AmihudEstimator {
    /// Full rolling history (bounded to `history_cap` bars).
    history: VecDeque<f64>,
    /// Maximum bars retained for z-score computation.
    history_cap: usize,
    /// Separate short window for "30-day" mean used by `is_illiquid`.
    short_window: usize,
    /// Running sum of the last `short_window` values.
    short_sum: f64,
    /// Deque used to maintain the short-window running sum.
    short_buf: VecDeque<f64>,
    /// Total bars ever pushed.
    pub bar_count: u64,
}

impl AmihudEstimator {
    /// Create an estimator.
    ///
    /// * `history_cap`   -- maximum bars to retain (default 252 for z-score).
    /// * `short_window`  -- window for the "30-day" mean (default 22 trading days).
    pub fn new(history_cap: usize, short_window: usize) -> Self {
        AmihudEstimator {
            history: VecDeque::new(),
            history_cap,
            short_window,
            short_sum: 0.0,
            short_buf: VecDeque::new(),
            bar_count: 0,
        }
    }

    /// Push one bar.
    ///
    /// * `abs_return`    -- |close/prev_close - 1| or any absolute return measure.
    /// * `dollar_volume` -- dollar-denominated turnover for the bar.
    ///
    /// Returns the raw Amihud ratio for this bar.
    pub fn push_bar(&mut self, abs_return: f64, dollar_volume: f64) -> f64 {
        self.bar_count += 1;
        let ratio = if dollar_volume < 1e-9 { 0.0 } else { abs_return / dollar_volume };

        // Maintain full history
        self.history.push_back(ratio);
        if self.history.len() > self.history_cap {
            self.history.pop_front();
        }

        // Maintain short-window running sum
        self.short_sum += ratio;
        self.short_buf.push_back(ratio);
        if self.short_buf.len() > self.short_window {
            let evicted = self.short_buf.pop_front().unwrap();
            self.short_sum -= evicted;
        }

        ratio
    }

    /// Current (most recent bar) Amihud ratio.
    ///
    /// Returns 0.0 if no bars have been pushed.
    pub fn illiquidity(&self) -> f64 {
        self.history.back().copied().unwrap_or(0.0)
    }

    /// Rolling mean over the last `n` bars.
    ///
    /// If fewer than `n` bars are available the mean is computed over all
    /// available bars.
    pub fn rolling_mean(&self, n: usize) -> f64 {
        if self.history.is_empty() {
            return 0.0;
        }
        let window = n.min(self.history.len());
        let sum: f64 = self.history.iter().rev().take(window).sum();
        sum / window as f64
    }

    /// Mean over the short-window (default 22 bars).
    pub fn short_mean(&self) -> f64 {
        if self.short_buf.is_empty() {
            return 0.0;
        }
        self.short_sum / self.short_buf.len() as f64
    }

    /// Z-score of the current illiquidity ratio relative to the full history window.
    ///
    /// Returns 0.0 if fewer than 2 observations exist.
    pub fn illiquidity_z_score(&self, window: usize) -> f64 {
        let win = window.min(self.history.len());
        if win < 2 {
            return 0.0;
        }
        let slice: Vec<f64> = self.history.iter().rev().take(win).copied().collect();
        let mean = slice.iter().sum::<f64>() / win as f64;
        let var = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (win - 1) as f64;
        let std = var.sqrt();
        if std < 1e-30 {
            return 0.0;
        }
        let current = self.illiquidity();
        (current - mean) / std
    }

    /// Returns `true` when the current ratio exceeds twice the short-window mean.
    ///
    /// Signals that the current bar is significantly more illiquid than the
    /// recent average -- i.e. market depth has dried up.
    pub fn is_illiquid(&self) -> bool {
        let sm = self.short_mean();
        if sm < 1e-30 {
            return false;
        }
        self.illiquidity() > 2.0 * sm
    }

    /// Number of bars currently in the full history window.
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// Export the full history as a Vec (oldest first).
    pub fn history_vec(&self) -> Vec<f64> {
        self.history.iter().copied().collect()
    }
}

impl Default for AmihudEstimator {
    /// 252-bar history, 22-bar short window (standard trading-calendar defaults).
    fn default() -> Self {
        Self::new(252, 22)
    }
}

// ---------------------------------------------------------------------------
// Free function helper
// ---------------------------------------------------------------------------

/// Compute a series of Amihud ratios from slices of (abs_return, dollar_volume).
pub fn compute_amihud_series(bars: &[(f64, f64)]) -> Vec<f64> {
    let mut est = AmihudEstimator::default();
    bars.iter().map(|&(r, dv)| est.push_bar(r, dv)).collect()
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amihud_ratio_direction() {
        // Higher absolute return with same dollar volume -> higher ratio.
        let mut est = AmihudEstimator::default();
        let r1 = est.push_bar(0.01, 1_000_000.0); // 1% return, $1M volume
        est.push_bar(0.01, 1_000_000.0);
        let r2_high = est.push_bar(0.05, 1_000_000.0); // 5% return, same volume
        assert!(r2_high > r1, "higher return -> higher illiquidity, got {} vs {}", r2_high, r1);
    }

    #[test]
    fn test_amihud_ratio_inverse_volume() {
        // Same return, lower dollar volume -> higher ratio.
        let mut est = AmihudEstimator::default();
        let r_high_vol = est.push_bar(0.01, 1_000_000.0);
        let r_low_vol = est.push_bar(0.01, 100_000.0);
        assert!(
            r_low_vol > r_high_vol,
            "lower dollar volume -> higher illiquidity, got {} vs {}",
            r_low_vol,
            r_high_vol
        );
    }

    #[test]
    fn test_amihud_zero_volume_returns_zero() {
        let mut est = AmihudEstimator::default();
        let ratio = est.push_bar(0.05, 0.0);
        assert_eq!(ratio, 0.0, "zero dollar volume should return 0.0");
    }

    #[test]
    fn test_amihud_rolling_mean_decreases_with_more_observations() {
        let mut est = AmihudEstimator::default();
        // Push many low-illiquidity bars
        for _ in 0..50 {
            est.push_bar(0.001, 1_000_000.0);
        }
        // Then push one very high illiquidity bar
        est.push_bar(0.10, 100.0);
        let mean_all = est.rolling_mean(51);
        let mean_recent = est.rolling_mean(5);
        // The full-window mean is diluted by 50 low bars; recent mean is high
        assert!(mean_recent > mean_all, "recent mean should be higher after illiquid spike");
    }

    #[test]
    fn test_z_score_positive_on_illiquid_bar() {
        let mut est = AmihudEstimator::default();
        // Establish baseline with normal bars
        for _ in 0..100 {
            est.push_bar(0.005, 1_000_000.0);
        }
        // Push a highly illiquid bar
        est.push_bar(0.10, 100.0);
        let z = est.illiquidity_z_score(101);
        assert!(z > 1.0, "illiquid bar should have positive z-score, got {}", z);
    }

    #[test]
    fn test_z_score_near_zero_for_normal_bar() {
        let mut est = AmihudEstimator::default();
        for _ in 0..100 {
            est.push_bar(0.005, 1_000_000.0);
        }
        // Push another normal bar
        est.push_bar(0.005, 1_000_000.0);
        let z = est.illiquidity_z_score(100);
        assert!(z.abs() < 2.0, "normal bar should have z-score near 0, got {}", z);
    }

    #[test]
    fn test_is_illiquid_flag() {
        let mut est = AmihudEstimator::default();
        // Seed with normal bars to establish short-window mean
        for _ in 0..30 {
            est.push_bar(0.005, 1_000_000.0); // ratio ~ 5e-9
        }
        assert!(!est.is_illiquid(), "normal bar should not be illiquid");
        // Now push an extreme bar
        est.push_bar(0.20, 100.0); // ratio = 0.002 >> 2x 5e-9
        assert!(est.is_illiquid(), "extreme illiquid bar should trigger flag");
    }

    #[test]
    fn test_history_len_bounded() {
        let mut est = AmihudEstimator::new(50, 10);
        for _ in 0..200 {
            est.push_bar(0.01, 1_000_000.0);
        }
        assert_eq!(est.history_len(), 50, "history should be capped at history_cap");
    }

    #[test]
    fn test_compute_amihud_series_length() {
        let bars: Vec<(f64, f64)> = (0..100).map(|i| (0.001 * (1 + i % 5) as f64, 1e6)).collect();
        let series = compute_amihud_series(&bars);
        assert_eq!(series.len(), 100);
    }

    #[test]
    fn test_illiquidity_returns_last_bar() {
        let mut est = AmihudEstimator::default();
        est.push_bar(0.01, 1_000_000.0);
        let last = est.push_bar(0.05, 500_000.0);
        assert!((est.illiquidity() - last).abs() < 1e-15);
    }
}
