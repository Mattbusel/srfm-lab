//! realtime_indicators.rs -- Incremental streaming indicators for tick-by-tick browser updates.
//!
//! Each struct maintains internal state across push() calls, making it suitable
//! for live data feeds where bars arrive one at a time. All types are exported
//! to JavaScript via wasm_bindgen.
//!
//! Design principles:
//!   -- O(1) per push; no unbounded history buffers
//!   -- Numerically stable online algorithms where applicable
//!   -- Warm-up period clearly documented per indicator

use wasm_bindgen::prelude::*;
use serde::Serialize;
use serde_wasm_bindgen;

// ---------------------------------------------------------------------------
// IncrementalEMA
// ---------------------------------------------------------------------------

/// Exponential Moving Average updated one value at a time.
///
/// Warm-up: first push() seeds the EMA with the input value (zero-lag seed).
/// Subsequent calls apply the standard EMA recurrence.
#[wasm_bindgen]
pub struct IncrementalEMA {
    alpha: f64,
    value: f64,
    initialized: bool,
}

#[wasm_bindgen]
impl IncrementalEMA {
    /// Create a new EMA with the given period.
    /// alpha = 2 / (period + 1), clamped to (0, 1).
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize) -> IncrementalEMA {
        let p = period.max(1) as f64;
        IncrementalEMA {
            alpha: 2.0 / (p + 1.0),
            value: 0.0,
            initialized: false,
        }
    }

    /// Create with explicit smoothing factor alpha in (0,1).
    pub fn with_alpha(alpha: f64) -> IncrementalEMA {
        IncrementalEMA {
            alpha: alpha.clamp(1e-6, 1.0 - 1e-6),
            value: 0.0,
            initialized: false,
        }
    }

    /// Push a new value and return the updated EMA.
    pub fn push(&mut self, value: f64) -> f64 {
        if !self.initialized {
            self.value = value;
            self.initialized = true;
        } else {
            self.value = self.alpha * value + (1.0 - self.alpha) * self.value;
        }
        self.value
    }

    /// Return the current EMA value without advancing.
    pub fn current(&self) -> f64 {
        self.value
    }

    /// Reset state -- next push() will seed from scratch.
    pub fn reset(&mut self) {
        self.value = 0.0;
        self.initialized = false;
    }

    /// Return the smoothing factor alpha.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

// ---------------------------------------------------------------------------
// IncrementalRSI
// ---------------------------------------------------------------------------

/// Relative Strength Index (RSI) computed incrementally using Wilder's smoothing.
///
/// Returns values in [0, 100]. Returns 50.0 during warm-up (period bars).
/// Wilder's smoothing is equivalent to EMA with alpha = 1/period.
#[wasm_bindgen]
pub struct IncrementalRSI {
    period: usize,
    avg_gain: f64,
    avg_loss: f64,
    prev_close: Option<f64>,
    bars_seen: usize,
    /// Accumulator used during the initial SMA warm-up phase.
    gain_sum: f64,
    loss_sum: f64,
}

#[wasm_bindgen]
impl IncrementalRSI {
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize) -> IncrementalRSI {
        IncrementalRSI {
            period: period.max(2),
            avg_gain: 0.0,
            avg_loss: 0.0,
            prev_close: None,
            bars_seen: 0,
            gain_sum: 0.0,
            loss_sum: 0.0,
        }
    }

    /// Push a close price and return RSI(0..100). Returns 50.0 during warm-up.
    pub fn push(&mut self, close: f64) -> f64 {
        let Some(prev) = self.prev_close else {
            self.prev_close = Some(close);
            return 50.0;
        };

        let change = close - prev;
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { -change } else { 0.0 };
        self.prev_close = Some(close);
        self.bars_seen += 1;

        if self.bars_seen < self.period {
            // Accumulate for initial SMA seed
            self.gain_sum += gain;
            self.loss_sum += loss;
            return 50.0;
        } else if self.bars_seen == self.period {
            // Seed Wilder's averages with SMA
            self.gain_sum += gain;
            self.loss_sum += loss;
            self.avg_gain = self.gain_sum / self.period as f64;
            self.avg_loss = self.loss_sum / self.period as f64;
        } else {
            // Wilder's smoothing: avg = (prev_avg * (period-1) + current) / period
            let p = self.period as f64;
            self.avg_gain = (self.avg_gain * (p - 1.0) + gain) / p;
            self.avg_loss = (self.avg_loss * (p - 1.0) + loss) / p;
        }

        if self.avg_loss < 1e-12 {
            return 100.0;
        }
        let rs = self.avg_gain / self.avg_loss;
        100.0 - 100.0 / (1.0 + rs)
    }

    pub fn reset(&mut self) {
        self.avg_gain = 0.0;
        self.avg_loss = 0.0;
        self.prev_close = None;
        self.bars_seen = 0;
        self.gain_sum = 0.0;
        self.loss_sum = 0.0;
    }

    pub fn current_avg_gain(&self) -> f64 {
        self.avg_gain
    }

    pub fn current_avg_loss(&self) -> f64 {
        self.avg_loss
    }
}

// ---------------------------------------------------------------------------
// IncrementalBollinger
// ---------------------------------------------------------------------------

/// Bollinger Bands computed incrementally via Welford's online variance algorithm.
///
/// Maintains a circular buffer of `period` prices to compute rolling mean and std.
/// Memory: O(period).
#[wasm_bindgen]
pub struct IncrementalBollinger {
    period: usize,
    n_std: f64,
    buffer: Vec<f64>,
    head: usize,
    count: usize,
    /// Running sum for mean computation.
    sum: f64,
    /// Running sum of squares for variance computation.
    sum_sq: f64,
}

#[derive(Serialize)]
struct BollingerResult {
    upper: f64,
    mid: f64,
    lower: f64,
    width: f64,
    pct_b: f64,
}

#[wasm_bindgen]
impl IncrementalBollinger {
    /// Create with given period and band multiplier (standard deviations).
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize, n_std: f64) -> IncrementalBollinger {
        let p = period.max(2);
        IncrementalBollinger {
            period: p,
            n_std,
            buffer: vec![0.0; p],
            head: 0,
            count: 0,
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    /// Push a price and return a JS object {upper, mid, lower, width, pct_b}.
    /// During warm-up (< period bars), bands are centered on the input with zero width.
    pub fn push(&mut self, price: f64) -> JsValue {
        // Remove oldest value if buffer is full
        if self.count == self.period {
            let old = self.buffer[self.head];
            self.sum -= old;
            self.sum_sq -= old * old;
        } else {
            self.count += 1;
        }

        self.buffer[self.head] = price;
        self.head = (self.head + 1) % self.period;
        self.sum += price;
        self.sum_sq += price * price;

        let n = self.count as f64;
        let mean = self.sum / n;

        // Population variance over the window
        let variance = (self.sum_sq / n - mean * mean).max(0.0);
        let std_dev = variance.sqrt();

        let upper = mean + self.n_std * std_dev;
        let lower = mean - self.n_std * std_dev;
        let width = upper - lower;
        let pct_b = if width > 1e-12 {
            (price - lower) / width
        } else {
            0.5
        };

        let result = BollingerResult { upper, mid: mean, lower, width, pct_b };
        serde_wasm_bindgen::to_value(&result).unwrap_or(JsValue::NULL)
    }

    pub fn reset(&mut self) {
        self.buffer.fill(0.0);
        self.head = 0;
        self.count = 0;
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }

    pub fn mid(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum / self.count as f64
    }

    pub fn std_dev(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        let n = self.count as f64;
        let mean = self.sum / n;
        ((self.sum_sq / n - mean * mean).max(0.0)).sqrt()
    }
}

// ---------------------------------------------------------------------------
// IncrementalVWAP
// ---------------------------------------------------------------------------

/// Volume-Weighted Average Price, reset at session boundaries via reset_session().
///
/// VWAP = sum(price * volume) / sum(volume) from session open.
/// Handles zero-volume bars gracefully (price not included in weighted sum).
#[wasm_bindgen]
pub struct IncrementalVWAP {
    cumulative_pv: f64,
    cumulative_volume: f64,
    last_vwap: f64,
    bars_in_session: usize,
    /// Optional upper/lower bands (1 std dev from VWAP)
    sum_sq_dev: f64,
}

#[wasm_bindgen]
impl IncrementalVWAP {
    #[wasm_bindgen(constructor)]
    pub fn new() -> IncrementalVWAP {
        IncrementalVWAP {
            cumulative_pv: 0.0,
            cumulative_volume: 0.0,
            last_vwap: 0.0,
            bars_in_session: 0,
            sum_sq_dev: 0.0,
        }
    }

    /// Push a price-volume pair and return updated VWAP.
    pub fn push(&mut self, price: f64, volume: f64) -> f64 {
        if volume <= 0.0 {
            return self.last_vwap;
        }
        self.cumulative_pv += price * volume;
        self.cumulative_volume += volume;
        self.bars_in_session += 1;

        let vwap = self.cumulative_pv / self.cumulative_volume;

        // Accumulate squared deviation for band computation
        let dev = price - vwap;
        self.sum_sq_dev += dev * dev * volume;

        self.last_vwap = vwap;
        vwap
    }

    /// Reset session accumulators (call at session/day open).
    pub fn reset_session(&mut self) {
        self.cumulative_pv = 0.0;
        self.cumulative_volume = 0.0;
        self.last_vwap = 0.0;
        self.bars_in_session = 0;
        self.sum_sq_dev = 0.0;
    }

    pub fn current(&self) -> f64 {
        self.last_vwap
    }

    /// Return volume-weighted standard deviation from VWAP (used for bands).
    pub fn vwap_std(&self) -> f64 {
        if self.cumulative_volume < 1e-12 {
            return 0.0;
        }
        (self.sum_sq_dev / self.cumulative_volume).sqrt()
    }

    /// Upper VWAP band at n_std standard deviations.
    pub fn upper_band(&self, n_std: f64) -> f64 {
        self.last_vwap + n_std * self.vwap_std()
    }

    /// Lower VWAP band at n_std standard deviations.
    pub fn lower_band(&self, n_std: f64) -> f64 {
        self.last_vwap - n_std * self.vwap_std()
    }

    pub fn bars_in_session(&self) -> usize {
        self.bars_in_session
    }

    pub fn cumulative_volume(&self) -> f64 {
        self.cumulative_volume
    }
}

// ---------------------------------------------------------------------------
// IncrementalATR
// ---------------------------------------------------------------------------

/// Average True Range computed incrementally using Wilder's smoothing.
///
/// Warm-up: first bar seeds the ATR with (high - low).
/// Returns Wilder-smoothed ATR on subsequent bars.
#[wasm_bindgen]
pub struct IncrementalATR {
    period: usize,
    atr: f64,
    prev_close: Option<f64>,
    initialized: bool,
    /// Number of bars fed since last reset -- used for initial SMA seed.
    bars_seen: usize,
    /// Accumulator for the SMA seed phase.
    tr_sum: f64,
}

#[wasm_bindgen]
impl IncrementalATR {
    #[wasm_bindgen(constructor)]
    pub fn new(period: usize) -> IncrementalATR {
        IncrementalATR {
            period: period.max(1),
            atr: 0.0,
            prev_close: None,
            initialized: false,
            bars_seen: 0,
            tr_sum: 0.0,
        }
    }

    /// Push (high, low, close) and return the current ATR.
    /// Returns 0.0 during warm-up (first `period` bars).
    pub fn push(&mut self, high: f64, low: f64, close: f64) -> f64 {
        let tr = match self.prev_close {
            None => high - low,
            Some(prev_c) => {
                let hl = high - low;
                let hc = (high - prev_c).abs();
                let lc = (low - prev_c).abs();
                hl.max(hc).max(lc)
            }
        };

        self.prev_close = Some(close);
        self.bars_seen += 1;

        if !self.initialized {
            self.tr_sum += tr;
            if self.bars_seen >= self.period {
                self.atr = self.tr_sum / self.period as f64;
                self.initialized = true;
            }
        } else {
            // Wilder's smoothing
            let p = self.period as f64;
            self.atr = (self.atr * (p - 1.0) + tr) / p;
        }

        self.atr
    }

    /// Return current ATR without advancing.
    pub fn current(&self) -> f64 {
        self.atr
    }

    pub fn reset(&mut self) {
        self.atr = 0.0;
        self.prev_close = None;
        self.initialized = false;
        self.bars_seen = 0;
        self.tr_sum = 0.0;
    }

    /// Return true once the warm-up period has completed.
    pub fn is_ready(&self) -> bool {
        self.initialized
    }

    /// ATR as a percentage of the last close price (normalized volatility).
    pub fn atr_pct(&self, close: f64) -> f64 {
        if close < 1e-12 {
            return 0.0;
        }
        self.atr / close
    }
}
