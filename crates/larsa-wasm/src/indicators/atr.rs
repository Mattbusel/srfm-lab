//! atr.rs -- Average True Range (ATR) and derived volatility indicators.
//!
//! Standard ATR uses Wilder's exponential smoothing over true range.
//! True range = max(high-low, |high-prev_close|, |low-prev_close|)

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Internal true range computation
// ---------------------------------------------------------------------------

/// Compute the true range series from OHLC data.
/// First bar has TR = high - low (no previous close available).
pub(crate) fn true_range_series(highs: &[f64], lows: &[f64], closes: &[f64]) -> Vec<f64> {
    let n = highs.len();
    assert_eq!(n, lows.len());
    assert_eq!(n, closes.len());

    let mut tr = Vec::with_capacity(n);
    if n == 0 {
        return tr;
    }

    // First bar: no previous close
    tr.push(highs[0] - lows[0]);

    for i in 1..n {
        let hl    = highs[i] - lows[i];
        let hpc   = (highs[i] - closes[i - 1]).abs();
        let lpc   = (lows[i] - closes[i - 1]).abs();
        tr.push(hl.max(hpc).max(lpc));
    }
    tr
}

/// Compute ATR series using Wilder's smoothed average.
/// Returns Vec<f64> of same length as input; NaN for warmup bars.
pub(crate) fn atr_series(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> Vec<f64> {
    let n = highs.len();
    if n < period || period == 0 {
        return vec![f64::NAN; n];
    }

    let tr = true_range_series(highs, lows, closes);
    let mut atr = vec![f64::NAN; n];

    // Seed with SMA of first `period` TR values
    let seed: f64 = tr[..period].iter().sum::<f64>() / period as f64;
    atr[period - 1] = seed;
    let mut prev_atr = seed;

    // Wilder's smoothing: ATR_t = (ATR_{t-1} * (period-1) + TR_t) / period
    for i in period..n {
        let next = (prev_atr * (period as f64 - 1.0) + tr[i]) / period as f64;
        atr[i] = next;
        prev_atr = next;
    }

    atr
}

// ---------------------------------------------------------------------------
// WASM exports
// ---------------------------------------------------------------------------

/// Compute ATR(period) from high/low/close arrays.
/// Returns Float64Array of same length; NaN for warmup bars.
#[wasm_bindgen]
pub fn compute_atr(highs: &[f64], lows: &[f64], closes: &[f64], period: u32) -> Vec<f64> {
    if highs.len() != lows.len() || highs.len() != closes.len() {
        return Vec::new();
    }
    atr_series(highs, lows, closes, period as usize)
}

/// Compute ATR as a percentage of close price (normalized volatility).
/// Useful for position sizing: ATR% = ATR / close * 100.
#[wasm_bindgen]
pub fn compute_atr_percent(highs: &[f64], lows: &[f64], closes: &[f64], period: u32) -> Vec<f64> {
    let atr = atr_series(highs, lows, closes, period as usize);
    let n = closes.len();
    let mut atr_pct = vec![f64::NAN; n];
    for i in 0..n {
        if !atr[i].is_nan() && closes[i].abs() > 1e-12 {
            atr_pct[i] = atr[i] / closes[i] * 100.0;
        }
    }
    atr_pct
}

/// Compute Chandelier Exit stops.
/// Long stop  = highest high over `period` - ATR * multiplier
/// Short stop = lowest low over `period`  + ATR * multiplier
/// Returns packed [long_stop..., short_stop...] of length 2*n.
#[wasm_bindgen]
pub fn compute_chandelier_exit(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    period: u32,
    multiplier: f64,
) -> Vec<f64> {
    let n = highs.len();
    if n != lows.len() || n != closes.len() {
        return Vec::new();
    }
    let p = period as usize;
    let atr = atr_series(highs, lows, closes, p);
    let mut long_stop  = vec![f64::NAN; n];
    let mut short_stop = vec![f64::NAN; n];

    for i in (p - 1)..n {
        if atr[i].is_nan() {
            continue;
        }
        // Rolling highest high and lowest low over `period`
        let window_h = &highs[i + 1 - p..=i];
        let window_l = &lows[i + 1 - p..=i];
        let hh = window_h.iter().cloned().fold(f64::MIN, f64::max);
        let ll = window_l.iter().cloned().fold(f64::MAX, f64::min);
        long_stop[i]  = hh - multiplier * atr[i];
        short_stop[i] = ll + multiplier * atr[i];
    }

    let mut packed = Vec::with_capacity(2 * n);
    packed.extend_from_slice(&long_stop);
    packed.extend_from_slice(&short_stop);
    packed
}

/// Compute the Keltner Channel: middle = EMA, upper = EMA + mult*ATR, lower = EMA - mult*ATR.
/// Returns packed [upper..., middle..., lower...] of length 3*n.
#[wasm_bindgen]
pub fn compute_keltner_channel(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    ema_period: u32,
    atr_period: u32,
    multiplier: f64,
) -> Vec<f64> {
    let n = closes.len();
    if n != highs.len() || n != lows.len() {
        return Vec::new();
    }
    let atr    = atr_series(highs, lows, closes, atr_period as usize);
    let ep     = ema_period as usize;
    let alpha  = 2.0 / (ep as f64 + 1.0);

    let mut ema    = vec![f64::NAN; n];
    let mut upper  = vec![f64::NAN; n];
    let mut lower  = vec![f64::NAN; n];

    // Seed EMA with SMA
    if n >= ep {
        let seed: f64 = closes[..ep].iter().sum::<f64>() / ep as f64;
        ema[ep - 1] = seed;
        let mut e = seed;
        for i in ep..n {
            e = alpha * closes[i] + (1.0 - alpha) * e;
            ema[i] = e;
        }

        for i in 0..n {
            if !ema[i].is_nan() && !atr[i].is_nan() {
                upper[i] = ema[i] + multiplier * atr[i];
                lower[i] = ema[i] - multiplier * atr[i];
            }
        }
    }

    let mut packed = Vec::with_capacity(3 * n);
    packed.extend_from_slice(&upper);
    packed.extend_from_slice(&ema);
    packed.extend_from_slice(&lower);
    packed
}
