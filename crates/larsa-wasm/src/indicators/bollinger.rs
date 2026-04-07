//! bollinger.rs -- Bollinger Bands with configurable period and sigma multiplier.
//!
//! Standard parameters: period=20, sigma=2.0.
//! Returns upper, middle (SMA), and lower band arrays packed into one Vec.
//! Layout: [upper_0..upper_n, middle_0..middle_n, lower_0..lower_n]
//! In JS: const n = prices.length; const upper = result.slice(0,n); etc.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Internal computation
// ---------------------------------------------------------------------------

/// Compute rolling SMA and standard deviation for Bollinger Bands.
/// Returns (upper, middle, lower) as three Vec<f64> of length n.
pub(crate) fn bollinger_series(
    prices: &[f64],
    period: usize,
    sigma: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = prices.len();
    let mut upper  = vec![f64::NAN; n];
    let mut middle = vec![f64::NAN; n];
    let mut lower  = vec![f64::NAN; n];

    if n < period || period == 0 {
        return (upper, middle, lower);
    }

    for i in (period - 1)..n {
        let window = &prices[(i + 1 - period)..=i];

        // Welford mean
        let mean: f64 = window.iter().sum::<f64>() / period as f64;

        // Population std dev for Bollinger (Bollinger himself uses population std dev)
        let var: f64 = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / period as f64;
        let std_dev = var.sqrt();

        middle[i] = mean;
        upper[i]  = mean + sigma * std_dev;
        lower[i]  = mean - sigma * std_dev;
    }

    (upper, middle, lower)
}

// ---------------------------------------------------------------------------
// WASM exports
// ---------------------------------------------------------------------------

/// Compute Bollinger Bands for the given price array.
/// Returns packed Float64Array of length 3*n.
/// Slice in JS: upper=result.slice(0,n), middle=result.slice(n,2*n), lower=result.slice(2*n,3*n)
#[wasm_bindgen]
pub fn compute_bollinger(prices: &[f64], period: u32, sigma: f64) -> Vec<f64> {
    let n = prices.len();
    let (upper, middle, lower) = bollinger_series(prices, period as usize, sigma);
    let mut packed = Vec::with_capacity(3 * n);
    packed.extend_from_slice(&upper);
    packed.extend_from_slice(&middle);
    packed.extend_from_slice(&lower);
    packed
}

/// Compute %B indicator: (price - lower) / (upper - lower).
/// Values > 1.0 indicate price above upper band; < 0.0 below lower band.
#[wasm_bindgen]
pub fn compute_percent_b(prices: &[f64], period: u32, sigma: f64) -> Vec<f64> {
    let n = prices.len();
    let (upper, _middle, lower) = bollinger_series(prices, period as usize, sigma);
    let mut pct_b = vec![f64::NAN; n];

    for i in 0..n {
        if !upper[i].is_nan() {
            let bw = upper[i] - lower[i];
            if bw > 1e-12 {
                pct_b[i] = (prices[i] - lower[i]) / bw;
            } else {
                pct_b[i] = 0.5;
            }
        }
    }
    pct_b
}

/// Bandwidth = (upper - lower) / middle. Measures band width as fraction of price.
/// High bandwidth = high volatility; low bandwidth = squeeze.
#[wasm_bindgen]
pub fn compute_bandwidth(prices: &[f64], period: u32, sigma: f64) -> Vec<f64> {
    let n = prices.len();
    let (upper, middle, lower) = bollinger_series(prices, period as usize, sigma);
    let mut bw = vec![f64::NAN; n];

    for i in 0..n {
        if !upper[i].is_nan() && middle[i].abs() > 1e-12 {
            bw[i] = (upper[i] - lower[i]) / middle[i];
        }
    }
    bw
}

/// Squeeze indicator: returns 1.0 when Bollinger bandwidth is below its own
/// rolling `squeeze_period` minimum (indicating compression), 0.0 otherwise.
/// Useful for detecting low-volatility consolidation before breakouts.
#[wasm_bindgen]
pub fn compute_squeeze_signal(
    prices: &[f64],
    bb_period: u32,
    sigma: f64,
    squeeze_period: u32,
) -> Vec<f64> {
    let n = prices.len();
    let bw = compute_bandwidth(prices, bb_period, sigma);
    let sp = squeeze_period as usize;
    let mut signals = vec![0.0f64; n];

    for i in sp..n {
        let window = &bw[i - sp + 1..=i];
        let valid: Vec<f64> = window.iter().filter(|v| !v.is_nan()).copied().collect();
        if valid.is_empty() {
            continue;
        }
        let min_bw = valid.iter().cloned().fold(f64::MAX, f64::min);
        if bw[i] <= min_bw + 1e-10 {
            signals[i] = 1.0;
        }
    }

    signals
}
