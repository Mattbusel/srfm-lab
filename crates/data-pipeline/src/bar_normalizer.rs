/// Bar data normalization pipeline.
///
/// Handles spike detection, gap marking, volume z-scoring, and adjusted-price
/// computation. Designed to run as a streaming stage: each bar is processed
/// against a rolling history without lookahead.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::ohlcv::Bar;

// ── NormError ─────────────────────────────────────────────────────────────────

/// Errors returned by the normalizer when a bar cannot be safely normalized.
#[derive(Debug, Clone, Error, Serialize, Deserialize)]
pub enum NormError {
    #[error("price spike detected: |log_return| = {0:.4} > threshold {1:.4}")]
    Spike(f64, f64),

    #[error("zero or negative volume: {0}")]
    ZeroVolume(f64),

    #[error("invalid OHLC: open={open}, high={high}, low={low}, close={close}")]
    InvalidOHLC { open: f64, high: f64, low: f64, close: f64 },

    #[error("missing data: {0}")]
    MissingData(String),
}

// ── RawBar ────────────────────────────────────────────────────────────────────

/// Raw, unadjusted OHLCV bar plus an optional adjustment factor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawBar {
    pub timestamp_ns: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    /// Cumulative split/dividend adjustment factor (1.0 = no adjustment).
    pub adj_factor: f64,
    /// True if this bar was synthesized to fill a gap.
    pub is_gap: bool,
}

impl RawBar {
    pub fn new(ts: i64, o: f64, h: f64, l: f64, c: f64, v: f64) -> Self {
        RawBar {
            timestamp_ns: ts,
            open: o,
            high: h,
            low: l,
            close: c,
            volume: v,
            adj_factor: 1.0,
            is_gap: false,
        }
    }

    /// Convert from the shared `Bar` type.
    pub fn from_bar(bar: &Bar, adj_factor: f64) -> Self {
        RawBar {
            timestamp_ns: bar.timestamp.timestamp_nanos_opt().unwrap_or(0),
            open: bar.open,
            high: bar.high,
            low: bar.low,
            close: bar.close,
            volume: bar.volume,
            adj_factor,
            is_gap: false,
        }
    }
}

// ── NormalizedBar ─────────────────────────────────────────────────────────────

/// A bar after full normalization and quality checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedBar {
    pub timestamp_ns: i64,
    /// Adjustment-factor-corrected prices.
    pub adj_open: f64,
    pub adj_high: f64,
    pub adj_low: f64,
    pub adj_close: f64,
    /// Raw volume.
    pub volume: f64,
    /// Volume z-score vs rolling 20-bar history.
    pub volume_z: f64,
    /// Log return vs previous adjusted close.
    pub log_return: f64,
    /// True if this bar was classified as a gap bar.
    pub is_gap: bool,
    /// True if the log return was flagged as a spike (but not rejected --
    /// this flag is set only when the bar is returned by the tolerant path).
    pub is_spike_flagged: bool,
}

// ── BarHistory ────────────────────────────────────────────────────────────────

/// Compact rolling history for normalization context.
///
/// Maintains a ring buffer of the last N adjusted closes and volumes.
#[derive(Debug, Clone)]
pub struct BarHistory {
    /// Ring buffer of adjusted close prices.
    closes: Vec<f64>,
    /// Ring buffer of volumes.
    volumes: Vec<f64>,
    /// Write position.
    head: usize,
    /// Number of valid entries.
    count: usize,
    /// Buffer capacity (= rolling window).
    capacity: usize,
}

impl BarHistory {
    /// Create a new history buffer for the given window size.
    pub fn new(window: usize) -> Self {
        let cap = window.max(2);
        BarHistory {
            closes: vec![f64::NAN; cap],
            volumes: vec![f64::NAN; cap],
            head: 0,
            count: 0,
            capacity: cap,
        }
    }

    /// Push a new (adj_close, volume) observation.
    pub fn push(&mut self, adj_close: f64, volume: f64) {
        self.closes[self.head] = adj_close;
        self.volumes[self.head] = volume;
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Last (most recent) adjusted close, or NaN if empty.
    pub fn last_close(&self) -> f64 {
        if self.count == 0 {
            return f64::NAN;
        }
        let slot = (self.head + self.capacity - 1) % self.capacity;
        self.closes[slot]
    }

    /// Rolling std of log returns over all valid entries (minimum 2).
    pub fn rolling_return_std(&self) -> f64 {
        if self.count < 2 {
            return f64::NAN;
        }
        let mut log_rets = Vec::with_capacity(self.count - 1);
        for i in 1..self.count {
            let prev_slot = (self.head + self.capacity - i - 1) % self.capacity;
            let cur_slot = (self.head + self.capacity - i) % self.capacity;
            let prev = self.closes[prev_slot];
            let cur = self.closes[cur_slot];
            if prev > 0.0 && cur > 0.0 {
                log_rets.push((cur / prev).ln());
            }
        }
        std_f(&log_rets)
    }

    /// Rolling mean of volumes over valid entries.
    pub fn volume_mean(&self) -> f64 {
        if self.count == 0 {
            return f64::NAN;
        }
        let sum: f64 = (0..self.count)
            .map(|i| {
                let slot = (self.head + self.capacity - 1 - i) % self.capacity;
                self.volumes[slot]
            })
            .filter(|v| !v.is_nan())
            .sum();
        let cnt = (0..self.count)
            .filter(|&i| {
                let slot = (self.head + self.capacity - 1 - i) % self.capacity;
                !self.volumes[slot].is_nan()
            })
            .count();
        if cnt == 0 { f64::NAN } else { sum / cnt as f64 }
    }

    /// Rolling std of volumes over valid entries.
    pub fn volume_std(&self) -> f64 {
        if self.count < 2 {
            return f64::NAN;
        }
        let vals: Vec<f64> = (0..self.count)
            .map(|i| {
                let slot = (self.head + self.capacity - 1 - i) % self.capacity;
                self.volumes[slot]
            })
            .filter(|v| !v.is_nan())
            .collect();
        std_f(&vals)
    }

    /// Number of valid entries currently stored.
    pub fn count(&self) -> usize {
        self.count
    }
}

// ── BarNormalizer ─────────────────────────────────────────────────────────────

/// Configurable bar normalization pipeline.
pub struct BarNormalizer {
    /// Number of sigma for spike detection (default: 5.0).
    spike_sigma: f64,
    /// Rolling window for volume z-scoring (default: 20).
    volume_window: usize,
    /// Whether to hard-reject spikes (true) or flag and pass through (false).
    reject_spikes: bool,
}

impl BarNormalizer {
    /// Create a normalizer with default parameters.
    pub fn new() -> Self {
        BarNormalizer {
            spike_sigma: 5.0,
            volume_window: 20,
            reject_spikes: true,
        }
    }

    /// Override spike threshold (number of rolling std deviations).
    pub fn with_spike_sigma(mut self, sigma: f64) -> Self {
        self.spike_sigma = sigma;
        self
    }

    /// Pass spikes through with a flag instead of rejecting them.
    pub fn flag_spikes_only(mut self) -> Self {
        self.reject_spikes = false;
        self
    }

    // ── normalize ─────────────────────────────────────────────────────────

    /// Normalize a raw bar against a rolling history.
    ///
    /// Steps:
    /// 1. Validate OHLC consistency (H >= L, H >= O, H >= C, etc.).
    /// 2. Check volume > 0.
    /// 3. Apply adjustment factor to prices.
    /// 4. Compute log return; detect spikes (|lr| > spike_sigma * rolling_std).
    /// 5. Compute volume z-score.
    ///
    /// The history is NOT mutated here -- callers must call `history.push()`
    /// after processing if they want to update the rolling state.
    pub fn normalize(
        &self,
        bar: &RawBar,
        history: &BarHistory,
    ) -> Result<NormalizedBar, NormError> {
        // Step 1: OHLC sanity.
        if bar.high < bar.low
            || bar.high < bar.open
            || bar.high < bar.close
            || bar.low > bar.open
            || bar.low > bar.close
            || bar.open <= 0.0
            || bar.close <= 0.0
        {
            return Err(NormError::InvalidOHLC {
                open: bar.open,
                high: bar.high,
                low: bar.low,
                close: bar.close,
            });
        }

        // Step 2: Volume check.
        if bar.volume <= 0.0 {
            return Err(NormError::ZeroVolume(bar.volume));
        }

        // Step 3: Adjusted prices.
        let f = bar.adj_factor.max(1e-12);
        let adj_open = bar.open * f;
        let adj_high = bar.high * f;
        let adj_low = bar.low * f;
        let adj_close = bar.close * f;

        // Step 4: Log return and spike detection.
        let prev_close = history.last_close();
        let (log_return, is_spike_flagged) = if prev_close.is_nan() || prev_close <= 0.0 {
            // No prior close -- return = 0, no spike check.
            (0.0, false)
        } else {
            let lr = (adj_close / prev_close).ln();
            let rolling_std = history.rolling_return_std();

            let spike = if rolling_std.is_nan() || rolling_std < 1e-14 {
                false
            } else {
                lr.abs() > self.spike_sigma * rolling_std
            };

            if spike && self.reject_spikes {
                return Err(NormError::Spike(lr.abs(), self.spike_sigma * rolling_std));
            }
            (lr, spike)
        };

        // Step 5: Volume z-score.
        let vol_mean = history.volume_mean();
        let vol_std = history.volume_std();
        let volume_z = if vol_mean.is_nan() || vol_std.is_nan() || vol_std < 1e-10 {
            0.0
        } else {
            (bar.volume - vol_mean) / vol_std
        };

        Ok(NormalizedBar {
            timestamp_ns: bar.timestamp_ns,
            adj_open,
            adj_high,
            adj_low,
            adj_close,
            volume: bar.volume,
            volume_z,
            log_return,
            is_gap: bar.is_gap,
            is_spike_flagged,
        })
    }

    // ── Gap interpolation ──────────────────────────────────────────────────

    /// Interpolate or flag missing bars between `prev` and `next`.
    ///
    /// If `interpolate` is true, linearly interpolate O/H/L/C between the two
    /// known bars. Otherwise, return gap-flagged copies of `prev` at each
    /// missing timestamp.
    ///
    /// `gap_timestamps_ns` is a sorted list of the missing bar timestamps.
    pub fn fill_gaps(
        prev: &NormalizedBar,
        next: &NormalizedBar,
        gap_timestamps_ns: &[i64],
        interpolate: bool,
    ) -> Vec<NormalizedBar> {
        let n = gap_timestamps_ns.len();
        if n == 0 {
            return vec![];
        }

        gap_timestamps_ns
            .iter()
            .enumerate()
            .map(|(i, &ts)| {
                let (o, h, l, c) = if interpolate {
                    let t = (i + 1) as f64 / (n + 1) as f64;
                    let interp =
                        |a: f64, b: f64| a + t * (b - a);
                    (
                        interp(prev.adj_close, next.adj_open),
                        interp(prev.adj_high, next.adj_high),
                        interp(prev.adj_low, next.adj_low),
                        interp(prev.adj_close, next.adj_close),
                    )
                } else {
                    (prev.adj_close, prev.adj_high, prev.adj_low, prev.adj_close)
                };

                NormalizedBar {
                    timestamp_ns: ts,
                    adj_open: o,
                    adj_high: h,
                    adj_low: l,
                    adj_close: c,
                    volume: 0.0,
                    volume_z: 0.0,
                    log_return: 0.0,
                    is_gap: true,
                    is_spike_flagged: false,
                }
            })
            .collect()
    }
}

impl Default for BarNormalizer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn std_f(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = v.iter().sum::<f64>() / n as f64;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bar(o: f64, h: f64, l: f64, c: f64, v: f64) -> RawBar {
        RawBar::new(1_000_000, o, h, l, c, v)
    }

    fn warm_history(window: usize, start_close: f64) -> BarHistory {
        let mut h = BarHistory::new(window);
        for i in 0..window {
            h.push(start_close + i as f64 * 0.01, 1_000.0 + i as f64 * 10.0);
        }
        h
    }

    #[test]
    fn test_normalize_valid_bar() {
        let norm = BarNormalizer::new();
        let bar = make_bar(100.0, 101.0, 99.0, 100.5, 5000.0);
        let history = warm_history(20, 100.0);
        let result = norm.normalize(&bar, &history);
        assert!(result.is_ok(), "expected Ok, got {:?}", result.err());
        let nb = result.unwrap();
        assert!((nb.adj_close - 100.5).abs() < 1e-9);
        assert!(!nb.is_spike_flagged);
    }

    #[test]
    fn test_normalize_zero_volume_err() {
        let norm = BarNormalizer::new();
        let bar = make_bar(100.0, 101.0, 99.0, 100.5, 0.0);
        let history = warm_history(20, 100.0);
        let result = norm.normalize(&bar, &history);
        assert!(matches!(result, Err(NormError::ZeroVolume(_))));
    }

    #[test]
    fn test_normalize_invalid_ohlc_err() {
        let norm = BarNormalizer::new();
        // High < Low -- invalid.
        let bar = make_bar(100.0, 98.0, 102.0, 100.5, 1000.0);
        let history = warm_history(20, 100.0);
        let result = norm.normalize(&bar, &history);
        assert!(matches!(result, Err(NormError::InvalidOHLC { .. })));
    }

    #[test]
    fn test_spike_rejection() {
        let norm = BarNormalizer::new().with_spike_sigma(3.0);
        // Build a history of closes near 100.0.
        let mut history = BarHistory::new(20);
        for _ in 0..20 {
            history.push(100.0, 1000.0);
        }
        // Push a close very far from 100 to create nonzero std.
        history.push(100.1, 1000.0);

        // A 50% move should trigger spike detection.
        let bar = make_bar(100.0, 150.0, 100.0, 150.0, 5000.0);
        let result = norm.normalize(&bar, &history);
        assert!(matches!(result, Err(NormError::Spike(_, _))), "expected spike error");
    }

    #[test]
    fn test_spike_flag_only() {
        let norm = BarNormalizer::new().with_spike_sigma(0.001).flag_spikes_only();
        let mut history = BarHistory::new(20);
        for i in 0..20 {
            history.push(100.0 + i as f64 * 0.001, 1000.0);
        }
        let bar = make_bar(100.0, 200.0, 100.0, 200.0, 5000.0);
        let result = norm.normalize(&bar, &history);
        assert!(result.is_ok());
        let nb = result.unwrap();
        assert!(nb.is_spike_flagged);
    }

    #[test]
    fn test_adj_factor_applied() {
        let norm = BarNormalizer::new();
        let mut bar = make_bar(100.0, 102.0, 98.0, 100.0, 1000.0);
        bar.adj_factor = 0.5;
        let mut history = BarHistory::new(20);
        history.push(50.0, 1000.0); // prior adj_close = 50
        let nb = norm.normalize(&bar, &history).unwrap();
        assert!((nb.adj_close - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_volume_z_score() {
        let norm = BarNormalizer::new();
        let mut history = BarHistory::new(20);
        for _ in 0..20 {
            history.push(100.0, 1_000.0);
        }
        // Volume exactly at mean => z = 0 (if std ~0, we get 0 by guard).
        let bar = make_bar(99.5, 100.5, 99.0, 100.0, 1_000.0);
        let nb = norm.normalize(&bar, &history).unwrap();
        // z should be 0 or very close (std from identical history is ~0).
        assert!(nb.volume_z.abs() < 1.0);
    }

    #[test]
    fn test_fill_gaps_interpolated() {
        let prev = NormalizedBar {
            timestamp_ns: 0,
            adj_open: 100.0, adj_high: 101.0, adj_low: 99.0, adj_close: 100.0,
            volume: 1000.0, volume_z: 0.0, log_return: 0.0,
            is_gap: false, is_spike_flagged: false,
        };
        let next = NormalizedBar {
            timestamp_ns: 4_000,
            adj_open: 104.0, adj_high: 105.0, adj_low: 103.0, adj_close: 104.0,
            volume: 1000.0, volume_z: 0.0, log_return: 0.0,
            is_gap: false, is_spike_flagged: false,
        };
        let gaps = BarNormalizer::fill_gaps(&prev, &next, &[1_000, 2_000, 3_000], true);
        assert_eq!(gaps.len(), 3);
        assert!(gaps[0].is_gap);
        // Interpolated close should be strictly between 100 and 104.
        assert!(gaps[0].adj_close > 100.0 && gaps[0].adj_close < 104.0);
    }
}
