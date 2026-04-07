//! rsi.rs -- Relative Strength Index computation.
//!
//! Standard RSI uses Wilder's exponential smoothing (alpha = 1/period).
//! Returns an array of the same length as the input price array.
//! First `period` values are NaN (warmup period).

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Internal RSI computation kernel
// ---------------------------------------------------------------------------

/// Compute RSI using Wilder's smoothed moving average method.
/// Returns Vec<f64> of same length as prices; NaN for warmup bars.
pub(crate) fn rsi_series(prices: &[f64], period: usize) -> Vec<f64> {
    let n = prices.len();
    if n <= period || period == 0 {
        return vec![f64::NAN; n];
    }

    let mut result = vec![f64::NAN; n];

    // Compute initial average gain/loss over first `period` bars
    let mut avg_gain = 0.0f64;
    let mut avg_loss = 0.0f64;

    for i in 1..=period {
        let change = prices[i] - prices[i - 1];
        if change > 0.0 {
            avg_gain += change;
        } else {
            avg_loss += -change;
        }
    }
    avg_gain /= period as f64;
    avg_loss /= period as f64;

    // First RSI value at index `period`
    if avg_loss == 0.0 {
        result[period] = 100.0;
    } else {
        let rs = avg_gain / avg_loss;
        result[period] = 100.0 - 100.0 / (1.0 + rs);
    }

    // Wilder's smoothing for subsequent bars
    let alpha = 1.0 / period as f64;
    for i in (period + 1)..n {
        let change = prices[i] - prices[i - 1];
        let gain = if change > 0.0 { change } else { 0.0 };
        let loss = if change < 0.0 { -change } else { 0.0 };

        avg_gain = avg_gain * (1.0 - alpha) + gain * alpha;
        avg_loss = avg_loss * (1.0 - alpha) + loss * alpha;

        if avg_loss == 0.0 {
            result[i] = 100.0;
        } else {
            let rs = avg_gain / avg_loss;
            result[i] = 100.0 - 100.0 / (1.0 + rs);
        }
    }

    result
}

// ---------------------------------------------------------------------------
// WASM exports
// ---------------------------------------------------------------------------

/// Compute RSI(period) for the given price array.
/// Returns Float64Array of same length; NaN for warmup bars.
/// Overbought threshold: 70. Oversold threshold: 30.
#[wasm_bindgen]
pub fn compute_rsi(prices: &[f64], period: u32) -> Vec<f64> {
    rsi_series(prices, period as usize)
}

/// Stochastic RSI: RSI normalized within its own rolling min/max window.
/// Returns values in [0, 100]. NaN during warmup.
#[wasm_bindgen]
pub fn compute_stoch_rsi(prices: &[f64], rsi_period: u32, stoch_period: u32) -> Vec<f64> {
    let rsi = rsi_series(prices, rsi_period as usize);
    let n = rsi.len();
    let sp = stoch_period as usize;
    let mut result = vec![f64::NAN; n];

    for i in sp..n {
        let window = &rsi[i - sp + 1..=i];
        // Collect non-NaN values in the window
        let valid: Vec<f64> = window.iter().filter(|v| !v.is_nan()).copied().collect();
        if valid.len() < sp {
            continue;
        }
        let min_rsi = valid.iter().cloned().fold(f64::MAX, f64::min);
        let max_rsi = valid.iter().cloned().fold(f64::MIN, f64::max);
        let range = max_rsi - min_rsi;
        if range < 1e-10 {
            result[i] = 50.0;
        } else {
            result[i] = (rsi[i] - min_rsi) / range * 100.0;
        }
    }

    result
}

/// RSI divergence signal: returns +1.0 where bullish divergence (price lower low,
/// RSI higher low), -1.0 for bearish divergence, 0.0 otherwise.
/// Looks back `lookback` bars for swing detection.
#[wasm_bindgen]
pub fn compute_rsi_divergence(
    prices: &[f64],
    rsi_period: u32,
    lookback: u32,
) -> Vec<f64> {
    let n = prices.len();
    let lb = lookback as usize;
    let rsi = rsi_series(prices, rsi_period as usize);
    let mut signals = vec![0.0f64; n];

    if n < lb + 2 {
        return signals;
    }

    for i in lb..n {
        let price_now = prices[i];
        let rsi_now = rsi[i];
        if rsi_now.is_nan() {
            continue;
        }

        // Find local low in lookback window
        let mut price_low = f64::MAX;
        let mut rsi_at_price_low = f64::NAN;
        for j in (i - lb)..i {
            if prices[j] < price_low && !rsi[j].is_nan() {
                price_low = prices[j];
                rsi_at_price_low = rsi[j];
            }
        }

        if rsi_at_price_low.is_nan() {
            continue;
        }

        // Bullish divergence: price makes lower low, RSI makes higher low
        if price_now < price_low && rsi_now > rsi_at_price_low {
            signals[i] = 1.0;
        }

        // Find local high in lookback window for bearish divergence
        let mut price_high = f64::MIN;
        let mut rsi_at_price_high = f64::NAN;
        for j in (i - lb)..i {
            if prices[j] > price_high && !rsi[j].is_nan() {
                price_high = prices[j];
                rsi_at_price_high = rsi[j];
            }
        }

        if !rsi_at_price_high.is_nan()
            && price_now > price_high
            && rsi_now < rsi_at_price_high
        {
            signals[i] = -1.0;
        }
    }

    signals
}
