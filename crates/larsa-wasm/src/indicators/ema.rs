//! ema.rs -- EMA, double EMA, and MACD indicator computations.
//!
//! All functions accept JavaScript Float64Array slices and return Vec<f64>
//! which wasm-bindgen automatically converts to Float64Array on the JS side.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// EmaState -- stateful EMA for streaming tick-by-tick updates
// ---------------------------------------------------------------------------

/// Stateful EMA for streaming use. Maintains the current EMA value and alpha.
/// JS usage:
///   const ema = new EmaState(0.1);
///   ema.update(price);
///   const val = ema.get();
#[wasm_bindgen]
pub struct EmaState {
    alpha: f64,
    value: f64,
    initialized: bool,
    period: u32,
}

#[wasm_bindgen]
impl EmaState {
    /// Create a new EMA with the given smoothing period.
    /// Alpha = 2 / (period + 1), which is the standard Wilder formula.
    #[wasm_bindgen(constructor)]
    pub fn new(period: u32) -> EmaState {
        let alpha = 2.0 / (period as f64 + 1.0);
        EmaState {
            alpha,
            value: 0.0,
            initialized: false,
            period,
        }
    }

    /// Create with explicit alpha (for RiskMetrics or custom decay rates).
    pub fn with_alpha(alpha: f64) -> EmaState {
        EmaState {
            alpha,
            value: 0.0,
            initialized: false,
            period: 0,
        }
    }

    /// Feed the next price observation. Seeds from first value.
    pub fn update(&mut self, x: f64) {
        if !self.initialized {
            self.value = x;
            self.initialized = true;
        } else {
            self.value = self.alpha * x + (1.0 - self.alpha) * self.value;
        }
    }

    /// Returns the current EMA value.
    pub fn get(&self) -> f64 {
        self.value
    }

    pub fn period(&self) -> u32 {
        self.period
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn reset(&mut self) {
        self.value = 0.0;
        self.initialized = false;
    }
}

// ---------------------------------------------------------------------------
// Internal helpers (not exported to WASM)
// ---------------------------------------------------------------------------

/// Compute a full EMA series from a price slice. Returns Vec same length as prices.
/// First `period-1` values are seeded with SMA warmup, then EMA begins.
pub(crate) fn ema_series(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    if n == 0 || period == 0 {
        return Vec::new();
    }
    let period = period.min(n);
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut result = vec![f64::NAN; n];

    // Seed with SMA of first `period` values
    let seed: f64 = prices[..period].iter().sum::<f64>() / period as f64;
    result[period - 1] = seed;
    let mut ema = seed;

    for i in period..n {
        ema = alpha * prices[i] + (1.0 - alpha) * ema;
        result[i] = ema;
    }
    result
}

// ---------------------------------------------------------------------------
// Batch WASM-exported functions
// ---------------------------------------------------------------------------

/// Compute EMA series for an entire price array.
/// Returns Float64Array of same length; NaN for warmup bars (first period-1 bars).
#[wasm_bindgen]
pub fn compute_ema(prices: &[f64], period: u32) -> Vec<f64> {
    ema_series(prices, period as usize)
}

/// Compute Double EMA (DEMA = 2*EMA - EMA(EMA)) for trend smoothing.
/// Reduces lag compared to single EMA.
#[wasm_bindgen]
pub fn compute_double_ema(prices: &[f64], period: u32) -> Vec<f64> {
    let p = period as usize;
    let ema1 = ema_series(prices, p);

    // Filter out NaN values for inner EMA computation
    // We need the valid portion of ema1 to compute ema2
    let n = prices.len();
    if n < 2 * p {
        return vec![f64::NAN; n];
    }

    // Compute EMA of EMA over the valid range
    let valid_start = p - 1;
    let valid_ema1: Vec<f64> = ema1[valid_start..].to_vec();
    let ema2_partial = ema_series(&valid_ema1, p);

    let mut result = vec![f64::NAN; n];
    let dema_start = valid_start + p - 1;
    for (i, &e2) in ema2_partial[p - 1..].iter().enumerate() {
        let idx = dema_start + i;
        if idx < n {
            let e1 = ema1[idx];
            result[idx] = 2.0 * e1 - e2;
        }
    }
    result
}

/// MACD result returned as three parallel arrays packed into one Vec.
/// Layout: [macd_line..., signal_line..., histogram...]
/// Each sub-array has length n. Use `n = prices.len()` to slice.
/// In JavaScript: const n = prices.length; macd = result.slice(0,n); etc.
#[wasm_bindgen]
pub fn compute_macd(
    prices: &[f64],
    fast_period: u32,
    slow_period: u32,
    signal_period: u32,
) -> Vec<f64> {
    let n = prices.len();
    if n == 0 {
        return Vec::new();
    }

    let fast = ema_series(prices, fast_period as usize);
    let slow = ema_series(prices, slow_period as usize);

    // MACD line = fast EMA - slow EMA
    let mut macd_line = vec![f64::NAN; n];
    for i in 0..n {
        if !fast[i].is_nan() && !slow[i].is_nan() {
            macd_line[i] = fast[i] - slow[i];
        }
    }

    // Signal line = EMA of MACD line over valid portion
    let slow_p = slow_period as usize;
    let valid_start = slow_p.saturating_sub(1);
    let valid_macd: Vec<f64> = macd_line[valid_start..]
        .iter()
        .filter(|v| !v.is_nan())
        .copied()
        .collect();

    let signal_partial = ema_series(&valid_macd, signal_period as usize);
    let mut signal_line = vec![f64::NAN; n];

    let signal_offset = signal_period as usize - 1;
    let mut valid_idx = 0usize;
    for i in valid_start..n {
        if !macd_line[i].is_nan() {
            if valid_idx >= signal_offset && valid_idx < signal_partial.len() {
                signal_line[i] = signal_partial[valid_idx];
            }
            valid_idx += 1;
        }
    }

    // Histogram = MACD line - signal line
    let mut histogram = vec![f64::NAN; n];
    for i in 0..n {
        if !macd_line[i].is_nan() && !signal_line[i].is_nan() {
            histogram[i] = macd_line[i] - signal_line[i];
        }
    }

    // Pack into single array: [macd | signal | histogram]
    let mut packed = Vec::with_capacity(3 * n);
    packed.extend_from_slice(&macd_line);
    packed.extend_from_slice(&signal_line);
    packed.extend_from_slice(&histogram);
    packed
}
