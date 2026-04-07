/// timeseries_ops.rs -- time series operations on OHLCV bar slices.
///
/// Provides resampling, alignment, gap-filling, rolling windows,
/// exponential smoothing, seasonal decomposition, differencing, and
/// autocorrelation for quantitative research pipelines.

use crate::ohlcv::Bar;
use chrono::{DateTime, TimeZone, Utc};

// ── Resampling ────────────────────────────────────────────────────────────────

/// Upsample bars from a finer interval to a coarser interval.
///
/// Rules:
///   open  = open of first bar in bucket
///   high  = max of all highs in bucket
///   low   = min of all lows in bucket
///   close = close of last bar in bucket
///   volume = sum of volumes in bucket
///   timestamp = start of bucket (aligned to to_interval_s grid)
///
/// Panics if to_interval_s < from_interval_s (downsampling not supported).
pub fn resample(bars: &[Bar], from_interval_s: i64, to_interval_s: i64) -> Vec<Bar> {
    assert!(
        to_interval_s >= from_interval_s,
        "resample only supports upsampling (to_interval_s >= from_interval_s)"
    );
    assert!(
        to_interval_s % from_interval_s == 0,
        "to_interval_s must be an exact multiple of from_interval_s"
    );

    if bars.is_empty() {
        return Vec::new();
    }

    let mut result: Vec<Bar> = Vec::new();

    // Group bars by bucket = floor(ts / to_interval_s) * to_interval_s
    let bucket_for = |ts: i64| -> i64 { (ts / to_interval_s) * to_interval_s };

    let mut bucket_start = bucket_for(bars[0].timestamp.timestamp());
    let mut bucket_open = bars[0].open;
    let mut bucket_high = bars[0].high;
    let mut bucket_low = bars[0].low;
    let mut bucket_close = bars[0].close;
    let mut bucket_volume = bars[0].volume;

    for bar in &bars[1..] {
        let b = bucket_for(bar.timestamp.timestamp());
        if b == bucket_start {
            // Same bucket -- accumulate.
            if bar.high > bucket_high { bucket_high = bar.high; }
            if bar.low  < bucket_low  { bucket_low  = bar.low;  }
            bucket_close = bar.close;
            bucket_volume += bar.volume;
        } else {
            // Emit completed bucket.
            result.push(Bar::new(
                Utc.timestamp_opt(bucket_start, 0).unwrap(),
                bucket_open,
                bucket_high,
                bucket_low,
                bucket_close,
                bucket_volume,
            ));
            // Start new bucket.
            bucket_start  = b;
            bucket_open   = bar.open;
            bucket_high   = bar.high;
            bucket_low    = bar.low;
            bucket_close  = bar.close;
            bucket_volume = bar.volume;
        }
    }

    // Emit final bucket.
    result.push(Bar::new(
        Utc.timestamp_opt(bucket_start, 0).unwrap(),
        bucket_open,
        bucket_high,
        bucket_low,
        bucket_close,
        bucket_volume,
    ));

    result
}

// ── Alignment ─────────────────────────────────────────────────────────────────

/// Inner join two bar series on timestamp (second precision).
/// Returns matched pairs in ascending timestamp order.
pub fn align_bars(series_a: &[Bar], series_b: &[Bar]) -> (Vec<Bar>, Vec<Bar>) {
    use std::collections::HashMap;

    // Index series_b by timestamp for O(n) matching.
    let index_b: HashMap<i64, &Bar> = series_b
        .iter()
        .map(|b| (b.timestamp.timestamp(), b))
        .collect();

    let mut out_a: Vec<Bar> = Vec::new();
    let mut out_b: Vec<Bar> = Vec::new();

    for bar in series_a {
        let ts = bar.timestamp.timestamp();
        if let Some(&b) = index_b.get(&ts) {
            out_a.push(bar.clone());
            out_b.push(b.clone());
        }
    }

    (out_a, out_b)
}

// ── Gap filling ───────────────────────────────────────────────────────────────

/// Fill missing intervals with synthetic bars using carry-forward close.
///
/// For each gap larger than interval_s, inserts bars with:
///   open = high = low = close = previous close
///   volume = 0
pub fn fill_gaps(bars: &[Bar], interval_s: i64) -> Vec<Bar> {
    if bars.is_empty() {
        return Vec::new();
    }

    let mut result: Vec<Bar> = Vec::with_capacity(bars.len());
    result.push(bars[0].clone());

    for pair in bars.windows(2) {
        let prev = &pair[0];
        let curr = &pair[1];
        let prev_ts = prev.timestamp.timestamp();
        let curr_ts = curr.timestamp.timestamp();
        let gap = curr_ts - prev_ts;

        if gap > interval_s {
            // Insert synthetic bars to fill the gap.
            let mut fill_ts = prev_ts + interval_s;
            while fill_ts < curr_ts {
                let synthetic = Bar::new(
                    Utc.timestamp_opt(fill_ts, 0).unwrap(),
                    prev.close,
                    prev.close,
                    prev.close,
                    prev.close,
                    0.0,
                );
                result.push(synthetic);
                fill_ts += interval_s;
            }
        }
        result.push(curr.clone());
    }

    result
}

// ── Rolling window ────────────────────────────────────────────────────────────

/// Return all overlapping windows of size `window` from `data`.
///
/// Output length = max(0, data.len() - window + 1).
pub fn rolling_window<T: Clone>(data: &[T], window: usize) -> Vec<Vec<T>> {
    if window == 0 || window > data.len() {
        return Vec::new();
    }
    (0..=(data.len() - window))
        .map(|i| data[i..i + window].to_vec())
        .collect()
}

// ── Exponential smoothing ─────────────────────────────────────────────────────

/// Single exponential smoothing: s_t = alpha * x_t + (1 - alpha) * s_{t-1}.
///
/// Initializes with s_0 = x_0.
/// `alpha` must be in (0, 1].
pub fn exponential_smoothing(data: &[f64], alpha: f64) -> Vec<f64> {
    assert!(
        alpha > 0.0 && alpha <= 1.0,
        "alpha must be in (0, 1], got {}",
        alpha
    );
    if data.is_empty() {
        return Vec::new();
    }
    let mut result = Vec::with_capacity(data.len());
    let mut s = data[0];
    result.push(s);
    for &x in &data[1..] {
        s = alpha * x + (1.0 - alpha) * s;
        result.push(s);
    }
    result
}

// ── Seasonal decomposition ────────────────────────────────────────────────────

/// Additive seasonal decomposition: y = trend + seasonal + residual.
///
/// Algorithm:
///   1. Trend = centered moving average of length `period` (odd) or
///      2x12 MA style for even period.
///   2. Detrended = y - trend.
///   3. Seasonal = average of detrended values at each period position.
///   4. Residual = y - trend - seasonal.
///
/// Returns (trend, seasonal, residual) each of length data.len().
/// Trend has NaNs at the edges where the window does not fit.
pub fn seasonal_decompose(
    data: &[f64],
    period: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = data.len();
    assert!(period >= 2, "period must be >= 2");
    assert!(n >= 2 * period, "data length must be at least 2 * period");

    // -- Step 1: centered moving average trend --------------------------------
    let half = period / 2;
    let mut trend = vec![f64::NAN; n];

    if period % 2 == 1 {
        // Odd period -- simple centered MA.
        for i in half..(n - half) {
            let sum: f64 = data[i - half..=i + half].iter().sum();
            trend[i] = sum / period as f64;
        }
    } else {
        // Even period -- 2x MA: average of two consecutive MAs of length period.
        for i in half..(n - half) {
            let sum_left: f64 = data[i - half..i + half].iter().sum();
            let sum_right: f64 = data[i - half + 1..=i + half].iter().sum();
            trend[i] = (sum_left + sum_right) / (2.0 * period as f64);
        }
    }

    // -- Step 2: detrended series ---------------------------------------------
    let detrended: Vec<f64> = data
        .iter()
        .zip(trend.iter())
        .map(|(&y, &t)| if t.is_nan() { f64::NAN } else { y - t })
        .collect();

    // -- Step 3: seasonal component = avg detrended by position ---------------
    let mut season_sum = vec![0.0f64; period];
    let mut season_cnt = vec![0usize; period];

    for (i, &d) in detrended.iter().enumerate() {
        if !d.is_nan() {
            season_sum[i % period] += d;
            season_cnt[i % period] += 1;
        }
    }

    // Average, then center so seasonal sums to 0 over one period.
    let mut season_avg: Vec<f64> = season_sum
        .iter()
        .zip(season_cnt.iter())
        .map(|(&s, &c)| if c > 0 { s / c as f64 } else { 0.0 })
        .collect();

    let season_mean: f64 = season_avg.iter().sum::<f64>() / period as f64;
    for v in &mut season_avg {
        *v -= season_mean;
    }

    let seasonal: Vec<f64> = (0..n).map(|i| season_avg[i % period]).collect();

    // -- Step 4: residual ------------------------------------------------------
    let residual: Vec<f64> = data
        .iter()
        .zip(trend.iter())
        .zip(seasonal.iter())
        .map(|((&y, &t), &s)| if t.is_nan() { f64::NAN } else { y - t - s })
        .collect();

    (trend, seasonal, residual)
}

// ── Differencing ──────────────────────────────────────────────────────────────

/// Compute d-th order differences of `data`.
///
/// d=1: [x1-x0, x2-x1, ...]
/// d=2: second differences, etc.
/// Returns slice of length max(0, n - d).
pub fn differencing(data: &[f64], d: u32) -> Vec<f64> {
    let mut current = data.to_vec();
    for _ in 0..d {
        if current.len() < 2 {
            return Vec::new();
        }
        current = current
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();
    }
    current
}

// ── Autocorrelation ───────────────────────────────────────────────────────────

/// Compute sample autocorrelation function (ACF) for lags 0..=max_lag.
///
/// ACF(lag) = sum((x_t - mu)(x_{t+lag} - mu)) / sum((x_t - mu)^2)
///
/// Returns vec of length max_lag + 1; index 0 is always 1.0.
pub fn autocorrelation(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let variance: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();

    if variance == 0.0 {
        // Constant series -- ACF undefined; return 1 at lag 0, NaN elsewhere.
        let mut out = vec![f64::NAN; max_lag + 1];
        out[0] = 1.0;
        return out;
    }

    (0..=max_lag)
        .map(|lag| {
            if lag >= n {
                return f64::NAN;
            }
            let cov: f64 = data[..n - lag]
                .iter()
                .zip(&data[lag..])
                .map(|(&a, &b)| (a - mean) * (b - mean))
                .sum();
            cov / variance
        })
        .collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn bar_at(ts_s: i64, o: f64, h: f64, l: f64, c: f64, v: f64) -> Bar {
        Bar::new(Utc.timestamp_opt(ts_s, 0).unwrap(), o, h, l, c, v)
    }

    // -- resample tests -------------------------------------------------------

    #[test]
    fn test_resample_four_bars_to_one() {
        // Four 15-min bars -> one 60-min bar.
        let t0 = 0i64;
        let bars = vec![
            bar_at(t0,            100.0, 102.0, 99.0, 101.0, 100.0),
            bar_at(t0 + 900,      101.0, 103.0, 100.0, 102.0, 200.0),
            bar_at(t0 + 1800,     102.0, 104.0, 101.0, 103.0, 150.0),
            bar_at(t0 + 2700,     103.0, 106.0, 102.0, 105.0, 50.0),
        ];
        let out = resample(&bars, 900, 3600);
        assert_eq!(out.len(), 1);
        let b = &out[0];
        assert_eq!(b.open, 100.0);
        assert_eq!(b.high, 106.0);
        assert_eq!(b.low, 99.0);
        assert_eq!(b.close, 105.0);
        assert_eq!(b.volume, 500.0);
    }

    #[test]
    fn test_resample_empty_input() {
        let out = resample(&[], 60, 3600);
        assert!(out.is_empty());
    }

    #[test]
    fn test_resample_two_buckets() {
        let t0 = 0i64;
        let bars = vec![
            bar_at(t0,       10.0, 11.0, 9.0, 10.5, 100.0),
            bar_at(t0 + 60,  10.5, 12.0, 10.0, 11.0, 200.0),
            bar_at(t0 + 120, 11.0, 11.5, 10.8, 11.2, 300.0),
            bar_at(t0 + 180, 11.2, 12.0, 11.0, 11.8, 400.0),
        ];
        // 2 bars -> 2-min bucket
        let out = resample(&bars, 60, 120);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].open, 10.0);
        assert_eq!(out[1].open, 11.0);
    }

    // -- align_bars tests -----------------------------------------------------

    #[test]
    fn test_align_bars_inner_join() {
        let t = 1_000_000i64;
        let a = vec![
            bar_at(t,       100.0, 101.0, 99.0, 100.5, 10.0),
            bar_at(t + 60,  101.0, 102.0, 100.0, 101.5, 20.0),
            bar_at(t + 120, 102.0, 103.0, 101.0, 102.5, 30.0),
        ];
        let b = vec![
            bar_at(t,       50.0, 51.0, 49.0, 50.5, 5.0),
            bar_at(t + 120, 52.0, 53.0, 51.0, 52.5, 15.0),
            bar_at(t + 240, 53.0, 54.0, 52.0, 53.5, 25.0),
        ];
        let (ra, rb) = align_bars(&a, &b);
        assert_eq!(ra.len(), 2);
        assert_eq!(rb.len(), 2);
        assert_eq!(ra[0].timestamp.timestamp(), t);
        assert_eq!(ra[1].timestamp.timestamp(), t + 120);
    }

    #[test]
    fn test_align_bars_no_overlap() {
        let a = vec![bar_at(1000, 1.0, 1.1, 0.9, 1.0, 1.0)];
        let b = vec![bar_at(2000, 2.0, 2.1, 1.9, 2.0, 1.0)];
        let (ra, rb) = align_bars(&a, &b);
        assert!(ra.is_empty());
        assert!(rb.is_empty());
    }

    // -- fill_gaps tests ------------------------------------------------------

    #[test]
    fn test_fill_gaps_single_missing() {
        let t0 = 0i64;
        let bars = vec![
            bar_at(t0,       100.0, 101.0, 99.0, 100.5, 10.0),
            // gap: t0+120 is missing
            bar_at(t0 + 180, 100.5, 102.0, 100.0, 101.0, 20.0),
        ];
        let filled = fill_gaps(&bars, 60);
        // Should have 4 bars: t0, t0+60, t0+120, t0+180
        assert_eq!(filled.len(), 4);
        // Synthetic bar at t0+60 carries prev close
        assert_eq!(filled[1].close, 100.5);
        assert_eq!(filled[1].volume, 0.0);
    }

    #[test]
    fn test_fill_gaps_no_gap() {
        let t0 = 0i64;
        let bars = vec![
            bar_at(t0,      100.0, 101.0, 99.0, 100.0, 10.0),
            bar_at(t0 + 60, 100.0, 102.0, 99.5, 101.0, 20.0),
        ];
        let filled = fill_gaps(&bars, 60);
        assert_eq!(filled.len(), 2);
    }

    // -- rolling_window tests -------------------------------------------------

    #[test]
    fn test_rolling_window_basic() {
        let data = vec![1, 2, 3, 4, 5];
        let windows = rolling_window(&data, 3);
        assert_eq!(windows.len(), 3);
        assert_eq!(windows[0], vec![1, 2, 3]);
        assert_eq!(windows[2], vec![3, 4, 5]);
    }

    #[test]
    fn test_rolling_window_too_large() {
        let data = vec![1, 2, 3];
        let windows = rolling_window(&data, 5);
        assert!(windows.is_empty());
    }

    // -- exponential_smoothing tests ------------------------------------------

    #[test]
    fn test_exponential_smoothing_alpha_one() {
        // With alpha=1 output = input.
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let out = exponential_smoothing(&data, 1.0);
        for (a, b) in out.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_exponential_smoothing_alpha_half() {
        let data = vec![4.0, 0.0, 0.0, 0.0];
        let out = exponential_smoothing(&data, 0.5);
        // s0=4, s1=0.5*0+0.5*4=2, s2=0.5*0+0.5*2=1, s3=0.5
        assert!((out[0] - 4.0).abs() < 1e-12);
        assert!((out[1] - 2.0).abs() < 1e-12);
        assert!((out[2] - 1.0).abs() < 1e-12);
        assert!((out[3] - 0.5).abs() < 1e-12);
    }

    // -- differencing tests ---------------------------------------------------

    #[test]
    fn test_differencing_first_order() {
        let data = vec![1.0, 3.0, 6.0, 10.0];
        let d1 = differencing(&data, 1);
        assert_eq!(d1, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_differencing_second_order() {
        let data = vec![1.0, 3.0, 6.0, 10.0];
        let d2 = differencing(&data, 2);
        // first diff: [2,3,4], second diff: [1,1]
        assert_eq!(d2, vec![1.0, 1.0]);
    }

    #[test]
    fn test_differencing_zero_order() {
        let data = vec![1.0, 2.0, 3.0];
        let d0 = differencing(&data, 0);
        assert_eq!(d0, data);
    }

    // -- autocorrelation tests ------------------------------------------------

    #[test]
    fn test_autocorrelation_lag0_is_one() {
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let acf = autocorrelation(&data, 5);
        assert!((acf[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_autocorrelation_white_noise_near_zero() {
        // Alternating +1/-1 should have lag-1 ACF near -1.
        let data: Vec<f64> = (0..40)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let acf = autocorrelation(&data, 1);
        // lag-1 should be strongly negative
        assert!(acf[1] < -0.8);
    }

    // -- seasonal_decompose tests ---------------------------------------------

    #[test]
    fn test_seasonal_decompose_trend_extracted() {
        // Linear trend + simple seasonal pattern.
        let period = 4;
        let n = 24;
        let seasonal_pattern = [1.0, -1.0, 2.0, -2.0];
        let data: Vec<f64> = (0..n)
            .map(|i| i as f64 + seasonal_pattern[i % period])
            .collect();
        let (trend, _seasonal, residual) = seasonal_decompose(&data, period);

        // Trend should be close to linear in the non-NaN interior.
        let interior: Vec<f64> = trend
            .iter()
            .cloned()
            .filter(|v| !v.is_nan())
            .collect();
        assert!(!interior.is_empty(), "trend should have non-NaN values");

        // Residuals in the interior should be small.
        let max_residual = residual
            .iter()
            .filter(|v| !v.is_nan())
            .map(|v| v.abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_residual < 1.5,
            "residuals should be small for synthetic data, got {}",
            max_residual
        );
    }

    #[test]
    fn test_rolling_window_single_element() {
        let data = vec![42.0f64];
        let windows = rolling_window(&data, 1);
        assert_eq!(windows.len(), 1);
        assert_eq!(windows[0], vec![42.0]);
    }
}
