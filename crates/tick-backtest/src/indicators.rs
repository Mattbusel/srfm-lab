// indicators.rs — Technical indicators for the tick-backtest engine
//
// All indicators are online (one-bar-at-a-time), allocation-free in the
// hot path, and return `f64::NAN` until they have enough history.

use crate::types::Regime;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// EMA — Exponential Moving Average
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct EMA {
    period: usize,
    k: f64,
    value: f64,
    count: usize,
    /// Accumulator for the seed SMA.
    seed_sum: f64,
}

impl EMA {
    pub fn new(period: usize) -> Self {
        assert!(period >= 1, "EMA period must be >= 1");
        let k = 2.0 / (period as f64 + 1.0);
        Self { period, k, value: f64::NAN, count: 0, seed_sum: 0.0 }
    }

    /// Feed the next value and return the current EMA (NAN until warmed up).
    pub fn update(&mut self, value: f64) -> f64 {
        if value.is_nan() {
            return self.value;
        }
        self.count += 1;
        if self.count < self.period {
            self.seed_sum += value;
            // Not ready yet
        } else if self.count == self.period {
            self.seed_sum += value;
            self.value = self.seed_sum / self.period as f64;
        } else {
            // EMA formula
            self.value = value * self.k + self.value * (1.0 - self.k);
        }
        self.value
    }

    pub fn value(&self) -> f64 {
        self.value
    }

    pub fn is_ready(&self) -> bool {
        !self.value.is_nan()
    }

    pub fn reset(&mut self) {
        self.value = f64::NAN;
        self.count = 0;
        self.seed_sum = 0.0;
    }
}

// ---------------------------------------------------------------------------
// SMA — Simple Moving Average (circular buffer)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SMA {
    period: usize,
    buffer: VecDeque<f64>,
    sum: f64,
}

impl SMA {
    pub fn new(period: usize) -> Self {
        assert!(period >= 1, "SMA period must be >= 1");
        Self { period, buffer: VecDeque::with_capacity(period + 1), sum: 0.0 }
    }

    pub fn update(&mut self, value: f64) -> f64 {
        if value.is_nan() {
            return f64::NAN;
        }
        self.buffer.push_back(value);
        self.sum += value;
        if self.buffer.len() > self.period {
            if let Some(old) = self.buffer.pop_front() {
                self.sum -= old;
            }
        }
        if self.buffer.len() == self.period {
            self.sum / self.period as f64
        } else {
            f64::NAN
        }
    }

    pub fn value(&self) -> f64 {
        if self.buffer.len() == self.period {
            self.sum / self.period as f64
        } else {
            f64::NAN
        }
    }

    pub fn is_ready(&self) -> bool {
        self.buffer.len() == self.period
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
        self.sum = 0.0;
    }
}

// ---------------------------------------------------------------------------
// ATR — Average True Range (Wilder's EMA)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ATR {
    period: usize,
    prev_close: f64,
    smoother: EMA,
}

impl ATR {
    pub fn new(period: usize) -> Self {
        Self { period, prev_close: f64::NAN, smoother: EMA::new(period) }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> f64 {
        let tr = if self.prev_close.is_nan() {
            high - low
        } else {
            let hl = high - low;
            let hc = (high - self.prev_close).abs();
            let lc = (low - self.prev_close).abs();
            hl.max(hc).max(lc)
        };
        self.prev_close = close;
        self.smoother.update(tr)
    }

    pub fn value(&self) -> f64 {
        self.smoother.value()
    }

    pub fn period(&self) -> usize {
        self.period
    }

    pub fn is_ready(&self) -> bool {
        self.smoother.is_ready()
    }

    pub fn reset(&mut self) {
        self.prev_close = f64::NAN;
        self.smoother.reset();
    }
}

// ---------------------------------------------------------------------------
// BollingerBands
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct BBands {
    pub upper: f64,
    pub middle: f64,
    pub lower: f64,
}

#[derive(Debug, Clone)]
pub struct BollingerBands {
    period: usize,
    n_std: f64,
    buffer: VecDeque<f64>,
    sum: f64,
    sum_sq: f64,
}

impl BollingerBands {
    pub fn new(period: usize, n_std: f64) -> Self {
        assert!(period >= 2, "Bollinger period must be >= 2");
        Self {
            period,
            n_std,
            buffer: VecDeque::with_capacity(period + 1),
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    pub fn update(&mut self, close: f64) -> Option<BBands> {
        self.buffer.push_back(close);
        self.sum += close;
        self.sum_sq += close * close;
        if self.buffer.len() > self.period {
            if let Some(old) = self.buffer.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }
        if self.buffer.len() < self.period {
            return None;
        }
        let n = self.period as f64;
        let mean = self.sum / n;
        let variance = (self.sum_sq / n - mean * mean).max(0.0);
        let std = variance.sqrt();
        Some(BBands {
            upper: mean + self.n_std * std,
            middle: mean,
            lower: mean - self.n_std * std,
        })
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }
}

// ---------------------------------------------------------------------------
// RSI — Wilder's RSI
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RSI {
    period: usize,
    prev_close: f64,
    avg_gain: f64,
    avg_loss: f64,
    count: usize,
    seed_gain: f64,
    seed_loss: f64,
}

impl RSI {
    pub fn new(period: usize) -> Self {
        assert!(period >= 2, "RSI period must be >= 2");
        Self {
            period,
            prev_close: f64::NAN,
            avg_gain: f64::NAN,
            avg_loss: f64::NAN,
            count: 0,
            seed_gain: 0.0,
            seed_loss: 0.0,
        }
    }

    pub fn update(&mut self, close: f64) -> f64 {
        if self.prev_close.is_nan() {
            self.prev_close = close;
            return f64::NAN;
        }
        let diff = close - self.prev_close;
        let gain = if diff > 0.0 { diff } else { 0.0 };
        let loss = if diff < 0.0 { -diff } else { 0.0 };
        self.prev_close = close;
        self.count += 1;

        if self.count < self.period {
            self.seed_gain += gain;
            self.seed_loss += loss;
            return f64::NAN;
        } else if self.count == self.period {
            self.seed_gain += gain;
            self.seed_loss += loss;
            self.avg_gain = self.seed_gain / self.period as f64;
            self.avg_loss = self.seed_loss / self.period as f64;
        } else {
            // Wilder's smoothing
            let k = 1.0 / self.period as f64;
            self.avg_gain = self.avg_gain * (1.0 - k) + gain * k;
            self.avg_loss = self.avg_loss * (1.0 - k) + loss * k;
        }

        if self.avg_loss < 1e-12 {
            return 100.0;
        }
        let rs = self.avg_gain / self.avg_loss;
        100.0 - 100.0 / (1.0 + rs)
    }

    pub fn reset(&mut self) {
        self.prev_close = f64::NAN;
        self.avg_gain = f64::NAN;
        self.avg_loss = f64::NAN;
        self.count = 0;
        self.seed_gain = 0.0;
        self.seed_loss = 0.0;
    }
}

// ---------------------------------------------------------------------------
// MACD
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct MACDValue {
    pub macd_line: f64,
    pub signal_line: f64,
    pub histogram: f64,
}

#[derive(Debug, Clone)]
pub struct MACD {
    fast: EMA,
    slow: EMA,
    signal: EMA,
}

impl MACD {
    /// Standard defaults: fast=12, slow=26, signal=9.
    pub fn new(fast_period: usize, slow_period: usize, signal_period: usize) -> Self {
        Self {
            fast: EMA::new(fast_period),
            slow: EMA::new(slow_period),
            signal: EMA::new(signal_period),
        }
    }

    pub fn update(&mut self, close: f64) -> Option<MACDValue> {
        let fast_val = self.fast.update(close);
        let slow_val = self.slow.update(close);
        if fast_val.is_nan() || slow_val.is_nan() {
            return None;
        }
        let macd_line = fast_val - slow_val;
        let signal_line = self.signal.update(macd_line);
        if signal_line.is_nan() {
            return None;
        }
        Some(MACDValue {
            macd_line,
            signal_line,
            histogram: macd_line - signal_line,
        })
    }

    pub fn reset(&mut self) {
        self.fast.reset();
        self.slow.reset();
        self.signal.reset();
    }
}

// ---------------------------------------------------------------------------
// VWAP — Volume Weighted Average Price (session-level, resets each day)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct VWAP {
    cum_pv: f64,
    cum_vol: f64,
}

impl VWAP {
    pub fn new() -> Self {
        Self { cum_pv: 0.0, cum_vol: 0.0 }
    }

    /// `typical_price` = (high + low + close) / 3.
    pub fn update(&mut self, high: f64, low: f64, close: f64, volume: f64) -> f64 {
        let tp = (high + low + close) / 3.0;
        self.cum_pv += tp * volume;
        self.cum_vol += volume;
        if self.cum_vol < 1e-12 {
            return f64::NAN;
        }
        self.cum_pv / self.cum_vol
    }

    pub fn reset(&mut self) {
        self.cum_pv = 0.0;
        self.cum_vol = 0.0;
    }

    pub fn value(&self) -> f64 {
        if self.cum_vol < 1e-12 { f64::NAN } else { self.cum_pv / self.cum_vol }
    }
}

impl Default for VWAP {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Momentum — n-period price momentum
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Momentum {
    period: usize,
    buffer: VecDeque<f64>,
}

impl Momentum {
    pub fn new(period: usize) -> Self {
        assert!(period >= 1);
        Self { period, buffer: VecDeque::with_capacity(period + 2) }
    }

    /// Returns (close / close[n_periods_ago]) − 1, or NAN if not warm.
    pub fn update(&mut self, close: f64) -> f64 {
        self.buffer.push_back(close);
        if self.buffer.len() > self.period + 1 {
            self.buffer.pop_front();
        }
        if self.buffer.len() <= self.period {
            return f64::NAN;
        }
        let past = self.buffer.front().copied().unwrap_or(f64::NAN);
        if past.abs() < 1e-12 {
            return f64::NAN;
        }
        close / past - 1.0
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
    }
}

// ---------------------------------------------------------------------------
// RollingVol — rolling annualised volatility of log-returns
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RollingVol {
    period: usize,
    annualise: f64,
    prev_close: f64,
    returns: VecDeque<f64>,
    sum: f64,
    sum_sq: f64,
}

impl RollingVol {
    /// `bars_per_year` used for annualisation (e.g. 252 for daily bars).
    pub fn new(period: usize, bars_per_year: f64) -> Self {
        assert!(period >= 2);
        Self {
            period,
            annualise: bars_per_year.sqrt(),
            prev_close: f64::NAN,
            returns: VecDeque::with_capacity(period + 1),
            sum: 0.0,
            sum_sq: 0.0,
        }
    }

    pub fn update(&mut self, close: f64) -> f64 {
        if self.prev_close.is_nan() || self.prev_close <= 0.0 {
            self.prev_close = close;
            return f64::NAN;
        }
        let r = (close / self.prev_close).ln();
        self.prev_close = close;
        self.returns.push_back(r);
        self.sum += r;
        self.sum_sq += r * r;
        if self.returns.len() > self.period {
            if let Some(old) = self.returns.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
            }
        }
        if self.returns.len() < self.period {
            return f64::NAN;
        }
        let n = self.period as f64;
        let mean = self.sum / n;
        let variance = ((self.sum_sq / n) - mean * mean).max(0.0);
        variance.sqrt() * self.annualise
    }

    pub fn reset(&mut self) {
        self.prev_close = f64::NAN;
        self.returns.clear();
        self.sum = 0.0;
        self.sum_sq = 0.0;
    }
}

// ---------------------------------------------------------------------------
// RegimeClassifier
// ---------------------------------------------------------------------------

/// Classifies current market regime using EMA-200, ATR-based vol.
///
/// Rules (simple, tunable):
///   HIGH_VOL  — annualised vol > vol_threshold
///   BULL      — close > ema200 and atr_pct < vol_threshold
///   BEAR      — close < ema200 and atr_pct < vol_threshold
///   SIDEWAYS  — otherwise
#[derive(Debug, Clone)]
pub struct RegimeClassifier {
    ema200: EMA,
    atr14: ATR,
    vol_threshold: f64,
}

impl RegimeClassifier {
    pub fn new(vol_threshold: f64) -> Self {
        Self {
            ema200: EMA::new(200),
            atr14: ATR::new(14),
            vol_threshold,
        }
    }

    pub fn update(&mut self, high: f64, low: f64, close: f64) -> Regime {
        let ema = self.ema200.update(close);
        let atr = self.atr14.update(high, low, close);
        self.classify(close, atr, ema)
    }

    pub fn classify(&self, close: f64, atr: f64, ema200: f64) -> Regime {
        if atr.is_nan() || ema200.is_nan() {
            return Regime::Sideways;
        }
        let atr_pct = if close > 0.0 { atr / close } else { 0.0 };
        if atr_pct > self.vol_threshold {
            Regime::HighVol
        } else if close > ema200 {
            Regime::Bull
        } else if close < ema200 {
            Regime::Bear
        } else {
            Regime::Sideways
        }
    }

    pub fn ema200(&self) -> f64 {
        self.ema200.value()
    }

    pub fn atr14(&self) -> f64 {
        self.atr14.value()
    }

    pub fn reset(&mut self) {
        self.ema200.reset();
        self.atr14.reset();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_warms_up_at_period() {
        let mut ema = EMA::new(3);
        assert!(ema.update(1.0).is_nan());
        assert!(ema.update(2.0).is_nan());
        let v = ema.update(3.0);
        assert!(!v.is_nan(), "EMA should be ready at period");
        // seed SMA = 2.0
        assert!((v - 2.0).abs() < 1e-9);
    }

    #[test]
    fn sma_warms_up_exactly() {
        let mut sma = SMA::new(3);
        assert!(sma.update(1.0).is_nan());
        assert!(sma.update(2.0).is_nan());
        let v = sma.update(3.0);
        assert!((v - 2.0).abs() < 1e-9);
    }

    #[test]
    fn rsi_bounds() {
        let mut rsi = RSI::new(14);
        let prices: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        for &p in &prices {
            let v = rsi.update(p);
            if !v.is_nan() {
                assert!(v >= 0.0 && v <= 100.0, "RSI out of bounds: {v}");
            }
        }
    }

    #[test]
    fn bollinger_spread_positive() {
        let mut bb = BollingerBands::new(20, 2.0);
        for i in 0..25 {
            if let Some(bands) = bb.update(100.0 + (i % 5) as f64) {
                assert!(bands.upper >= bands.middle);
                assert!(bands.middle >= bands.lower);
            }
        }
    }

    #[test]
    fn vwap_single_bar() {
        let mut vwap = VWAP::new();
        let v = vwap.update(102.0, 98.0, 100.0, 1000.0);
        assert!((v - 100.0).abs() < 1e-9);
    }
}
