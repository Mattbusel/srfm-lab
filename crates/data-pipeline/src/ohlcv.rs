/// OHLCV bar types, BarSeries, resampling, gap-filling, and basic analytics.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

// ── Bar ───────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bar {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

impl Bar {
    pub fn new(ts: DateTime<Utc>, o: f64, h: f64, l: f64, c: f64, v: f64) -> Self {
        Bar { timestamp: ts, open: o, high: h, low: l, close: c, volume: v }
    }

    pub fn typical_price(&self) -> f64 {
        (self.high + self.low + self.close) / 3.0
    }

    pub fn range(&self) -> f64 {
        self.high - self.low
    }

    pub fn body(&self) -> f64 {
        (self.close - self.open).abs()
    }

    pub fn is_bullish(&self) -> bool {
        self.close >= self.open
    }
}

// ── Frequency ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Frequency {
    Tick,
    Minute1,
    Minute5,
    Minute15,
    Minute30,
    Hour1,
    Hour4,
    Daily,
    Weekly,
    Monthly,
}

impl Frequency {
    pub fn duration_seconds(&self) -> i64 {
        match self {
            Frequency::Tick => 0,
            Frequency::Minute1 => 60,
            Frequency::Minute5 => 300,
            Frequency::Minute15 => 900,
            Frequency::Minute30 => 1800,
            Frequency::Hour1 => 3600,
            Frequency::Hour4 => 14400,
            Frequency::Daily => 86400,
            Frequency::Weekly => 604800,
            Frequency::Monthly => 2592000, // ~30 days
        }
    }
}

// ── BarSeries ─────────────────────────────────────────────────────────────────

/// Efficient time series of OHLCV bars.
#[derive(Debug, Clone, Default)]
pub struct BarSeries {
    pub bars: Vec<Bar>,
    pub symbol: String,
    pub frequency: Option<Frequency>,
}

impl BarSeries {
    pub fn new(symbol: &str) -> Self {
        BarSeries { bars: Vec::new(), symbol: symbol.to_string(), frequency: None }
    }

    pub fn from_bars(bars: Vec<Bar>, symbol: &str) -> Self {
        BarSeries { bars, symbol: symbol.to_string(), frequency: None }
    }

    pub fn len(&self) -> usize { self.bars.len() }
    pub fn is_empty(&self) -> bool { self.bars.is_empty() }

    pub fn push(&mut self, bar: Bar) { self.bars.push(bar); }

    pub fn closes(&self) -> Vec<f64> { self.bars.iter().map(|b| b.close).collect() }
    pub fn opens(&self) -> Vec<f64> { self.bars.iter().map(|b| b.open).collect() }
    pub fn highs(&self) -> Vec<f64> { self.bars.iter().map(|b| b.high).collect() }
    pub fn lows(&self) -> Vec<f64> { self.bars.iter().map(|b| b.low).collect() }
    pub fn volumes(&self) -> Vec<f64> { self.bars.iter().map(|b| b.volume).collect() }
    pub fn timestamps(&self) -> Vec<DateTime<Utc>> { self.bars.iter().map(|b| b.timestamp).collect() }

    /// Last bar (O(1)).
    pub fn last(&self) -> Option<&Bar> { self.bars.last() }
    /// First bar (O(1)).
    pub fn first(&self) -> Option<&Bar> { self.bars.first() }
}

// ── Resample ──────────────────────────────────────────────────────────────────

/// Resample a BarSeries to a lower frequency.
pub fn resample(bars: &BarSeries, target_freq: Frequency) -> BarSeries {
    if bars.is_empty() { return BarSeries::new(&bars.symbol); }
    let bucket_secs = target_freq.duration_seconds();
    if bucket_secs == 0 { return bars.clone(); }

    // Group bars by bucket.
    let first_ts = bars.bars[0].timestamp.timestamp();
    let mut buckets: Vec<Vec<&Bar>> = Vec::new();
    let mut current_bucket_start = (first_ts / bucket_secs) * bucket_secs;
    let mut current_bucket: Vec<&Bar> = Vec::new();

    for bar in &bars.bars {
        let ts = bar.timestamp.timestamp();
        let bucket = (ts / bucket_secs) * bucket_secs;
        if bucket != current_bucket_start && !current_bucket.is_empty() {
            buckets.push(current_bucket.clone());
            current_bucket = Vec::new();
            current_bucket_start = bucket;
        }
        current_bucket.push(bar);
    }
    if !current_bucket.is_empty() { buckets.push(current_bucket); }

    // Aggregate each bucket.
    let agg_bars: Vec<Bar> = buckets
        .iter()
        .map(|bucket| {
            let o = bucket[0].open;
            let h = bucket.iter().map(|b| b.high).fold(f64::NEG_INFINITY, f64::max);
            let l = bucket.iter().map(|b| b.low).fold(f64::INFINITY, f64::min);
            let c = bucket.last().unwrap().close;
            let v: f64 = bucket.iter().map(|b| b.volume).sum();
            Bar::new(bucket[0].timestamp, o, h, l, c, v)
        })
        .collect();

    BarSeries { bars: agg_bars, symbol: bars.symbol.clone(), frequency: Some(target_freq) }
}

// ── Merge ─────────────────────────────────────────────────────────────────────

/// Merge two bar series by aligning on timestamp (inner join).
pub fn merge(series_a: &BarSeries, series_b: &BarSeries) -> (BarSeries, BarSeries) {
    use std::collections::HashMap;
    let map_b: HashMap<i64, &Bar> = series_b.bars.iter().map(|b| (b.timestamp.timestamp(), b)).collect();
    let mut out_a = BarSeries::new(&series_a.symbol);
    let mut out_b = BarSeries::new(&series_b.symbol);
    for bar_a in &series_a.bars {
        if let Some(&bar_b) = map_b.get(&bar_a.timestamp.timestamp()) {
            out_a.push(bar_a.clone());
            out_b.push(bar_b.clone());
        }
    }
    (out_a, out_b)
}

// ── Gap Filling ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub enum FillMethod {
    ForwardFill,
    Interpolate,
    Zero,
}

/// Fill gaps in a bar series.
/// `expected_freq` is the expected interval between bars in seconds.
pub fn fill_gaps(bars: &BarSeries, expected_freq_secs: i64, method: FillMethod) -> BarSeries {
    if bars.bars.len() < 2 { return bars.clone(); }
    let mut result = BarSeries::new(&bars.symbol);

    for window in bars.bars.windows(2) {
        let (a, b) = (&window[0], &window[1]);
        result.push(a.clone());

        let gap_secs = (b.timestamp.timestamp() - a.timestamp.timestamp()).max(0);
        let n_missing = (gap_secs / expected_freq_secs) as usize;

        for k in 1..n_missing {
            let frac = k as f64 / n_missing as f64;
            let ts = a.timestamp + Duration::seconds(k as i64 * expected_freq_secs);
            let bar = match method {
                FillMethod::ForwardFill => Bar::new(ts, a.open, a.high, a.low, a.close, 0.0),
                FillMethod::Interpolate => {
                    let interp = |va: f64, vb: f64| va + frac * (vb - va);
                    Bar::new(ts, interp(a.open, b.open), interp(a.high, b.high),
                        interp(a.low, b.low), interp(a.close, b.close), 0.0)
                }
                FillMethod::Zero => Bar::new(ts, a.close, a.close, a.close, a.close, 0.0),
            };
            result.push(bar);
        }
    }
    result.push(bars.bars.last().unwrap().clone());
    result
}

// ── Returns ───────────────────────────────────────────────────────────────────

/// Compute log returns from a bar series.
pub fn returns(bars: &BarSeries) -> Vec<f64> {
    bars.bars.windows(2)
        .map(|w| (w[1].close / w[0].close.max(1e-10)).ln())
        .collect()
}

/// Simple returns.
pub fn simple_returns(bars: &BarSeries) -> Vec<f64> {
    bars.bars.windows(2)
        .map(|w| (w[1].close - w[0].close) / w[0].close.max(1e-10))
        .collect()
}

// ── Rolling Statistics ────────────────────────────────────────────────────────

/// Rolling mean and standard deviation of log returns.
/// Returns Vec<(mean, std)> of length len - window.
pub fn rolling_stats(bars: &BarSeries, window: usize) -> Vec<(f64, f64)> {
    let rets = returns(bars);
    let n = rets.len();
    if n < window { return vec![]; }
    (window..=n)
        .map(|i| {
            let slice = &rets[i - window..i];
            let m = slice.iter().sum::<f64>() / window as f64;
            let v = slice.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (window - 1) as f64;
            (m, v.sqrt())
        })
        .collect()
}

// ── Renko Bars ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenkoBar {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub close: f64,
    pub brick_size: f64,
    pub direction: i8, // +1 = up, -1 = down
}

/// Generate Renko bars from a bar series.
pub fn renko(bars: &BarSeries, brick_size: f64) -> Vec<RenkoBar> {
    if bars.is_empty() { return vec![]; }
    let mut renko_bars = Vec::new();
    let mut current_level = bars.bars[0].close;

    for bar in &bars.bars {
        let price = bar.close;
        // Check for up bricks.
        while price >= current_level + brick_size {
            let o = current_level;
            let c = current_level + brick_size;
            renko_bars.push(RenkoBar {
                timestamp: bar.timestamp,
                open: o, close: c,
                brick_size, direction: 1,
            });
            current_level = c;
        }
        // Check for down bricks.
        while price <= current_level - brick_size {
            let o = current_level;
            let c = current_level - brick_size;
            renko_bars.push(RenkoBar {
                timestamp: bar.timestamp,
                open: o, close: c,
                brick_size, direction: -1,
            });
            current_level = c;
        }
    }
    renko_bars
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bars(n: usize, start_price: f64) -> BarSeries {
        let start = Utc::now();
        let bars: Vec<Bar> = (0..n)
            .map(|i| {
                let p = start_price + i as f64;
                Bar::new(start + Duration::days(i as i64), p, p + 1.0, p - 0.5, p + 0.5, 1000.0)
            })
            .collect();
        BarSeries::from_bars(bars, "TEST")
    }

    #[test]
    fn returns_length() {
        let bars = make_bars(10, 100.0);
        let rets = returns(&bars);
        assert_eq!(rets.len(), 9);
    }

    #[test]
    fn resample_reduces_bars() {
        let bars = make_bars(100, 50.0);
        let weekly = resample(&bars, Frequency::Weekly);
        assert!(weekly.len() < bars.len());
    }
}
