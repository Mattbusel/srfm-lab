// streaming/event_detector.rs -- Real-time market event detection.
//
// Detectors: VolatilityBreakout, MomentumShift, VolumeAnomaly, GapDetector,
// OrderFlowReversal. All emit typed events through a generic EventEmitter<T>
// backed by a crossbeam channel.

use std::collections::VecDeque;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// ─── EventEmitter ─────────────────────────────────────────────────────────────

/// Generic event emitter backed by an unbounded in-memory queue.
///
/// Producers call `emit(event)` and consumers call `drain()` to get all
/// buffered events since the last drain.
#[derive(Debug, Clone)]
pub struct EventEmitter<T: Clone> {
    queue: VecDeque<T>,
}

impl<T: Clone> EventEmitter<T> {
    pub fn new() -> Self {
        Self { queue: VecDeque::new() }
    }

    /// Emit one event.
    pub fn emit(&mut self, event: T) {
        self.queue.push_back(event);
    }

    /// Drain all pending events.
    pub fn drain(&mut self) -> Vec<T> {
        self.queue.drain(..).collect()
    }

    /// Peek at pending events without consuming them.
    pub fn pending(&self) -> usize {
        self.queue.len()
    }
}

impl<T: Clone> Default for EventEmitter<T> {
    fn default() -> Self { Self::new() }
}

// ─── VolatilityBreakoutEvent ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityBreakoutEvent {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    /// ATR at the moment of breakout.
    pub current_atr: f64,
    /// 20-bar ATR mean at the moment of breakout.
    pub baseline_atr: f64,
    /// current_atr / baseline_atr -- always >= 2.0 on a real event.
    pub magnitude: f64,
    /// True if this is a new breakout start (false = continuation update).
    pub is_new: bool,
}

/// Detects when ATR expands beyond 2x the 20-bar ATR mean.
///
/// Uses a 14-bar ATR (Wilder) and a 20-bar rolling mean of ATR values.
#[derive(Debug)]
pub struct VolatilityBreakout {
    symbol: String,
    atr_period: usize,
    mean_period: usize,
    threshold: f64,
    prev_close: Option<f64>,
    atr_buf: VecDeque<f64>,
    atr_mean_buf: VecDeque<f64>,
    current_atr: f64,
    in_breakout: bool,
    pub emitter: EventEmitter<VolatilityBreakoutEvent>,
}

impl VolatilityBreakout {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self::with_params(symbol, 14, 20, 2.0)
    }

    pub fn with_params(
        symbol: impl Into<String>,
        atr_period: usize,
        mean_period: usize,
        threshold: f64,
    ) -> Self {
        Self {
            symbol: symbol.into(),
            atr_period,
            mean_period,
            threshold,
            prev_close: None,
            atr_buf: VecDeque::with_capacity(atr_period + 1),
            atr_mean_buf: VecDeque::with_capacity(mean_period + 1),
            current_atr: 0.0,
            in_breakout: false,
            emitter: EventEmitter::new(),
        }
    }

    /// Feed OHLC bar data (open/high/low/close).
    pub fn update(&mut self, high: f64, low: f64, close: f64) {
        let tr = match self.prev_close {
            Some(pc) => {
                let hl = high - low;
                let hpc = (high - pc).abs();
                let lpc = (low - pc).abs();
                hl.max(hpc).max(lpc)
            }
            None => high - low,
        };
        self.prev_close = Some(close);

        // Wilder smoothed ATR
        if self.atr_buf.len() < self.atr_period {
            self.atr_buf.push_back(tr);
            if self.atr_buf.len() == self.atr_period {
                self.current_atr = self.atr_buf.iter().sum::<f64>() / self.atr_period as f64;
            }
            return;
        }
        self.current_atr = (self.current_atr * (self.atr_period as f64 - 1.0) + tr)
            / self.atr_period as f64;

        // Rolling mean of ATR values
        if self.atr_mean_buf.len() == self.mean_period {
            self.atr_mean_buf.pop_front();
        }
        self.atr_mean_buf.push_back(self.current_atr);

        if self.atr_mean_buf.len() < self.mean_period {
            return;
        }
        let baseline: f64 =
            self.atr_mean_buf.iter().sum::<f64>() / self.atr_mean_buf.len() as f64;

        if baseline < 1e-12 {
            return;
        }
        let magnitude = self.current_atr / baseline;

        if magnitude >= self.threshold {
            let is_new = !self.in_breakout;
            self.in_breakout = true;
            self.emitter.emit(VolatilityBreakoutEvent {
                timestamp: Utc::now(),
                symbol: self.symbol.clone(),
                current_atr: self.current_atr,
                baseline_atr: baseline,
                magnitude,
                is_new,
            });
        } else {
            self.in_breakout = false;
        }
    }
}

// ─── MomentumShiftEvent ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MomentumDirection {
    Positive,
    Negative,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MomentumShiftEvent {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    /// Direction prior to shift.
    pub from: MomentumDirection,
    /// Direction after shift.
    pub to: MomentumDirection,
    /// 5-bar momentum value at shift.
    pub momentum_value: f64,
}

/// Detects momentum exhaustion: sign change in 5-bar momentum after
/// at least 3 consecutive bars of same-sign momentum.
#[derive(Debug)]
pub struct MomentumShift {
    symbol: String,
    price_buf: VecDeque<f64>,
    momentum_period: usize,
    consecutive_period: usize,
    /// Sign of the last N momentum values.
    sign_buf: VecDeque<i8>,
    pub emitter: EventEmitter<MomentumShiftEvent>,
}

impl MomentumShift {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            price_buf: VecDeque::with_capacity(6),
            momentum_period: 5,
            consecutive_period: 3,
            sign_buf: VecDeque::with_capacity(10),
            emitter: EventEmitter::new(),
        }
    }

    pub fn update(&mut self, price: f64) {
        if self.price_buf.len() == self.momentum_period + 1 {
            self.price_buf.pop_front();
        }
        self.price_buf.push_back(price);

        if self.price_buf.len() < self.momentum_period + 1 {
            return;
        }

        let n = self.price_buf.len();
        let momentum = self.price_buf[n - 1] - self.price_buf[n - 1 - self.momentum_period];
        let sign: i8 = if momentum > 0.0 { 1 } else if momentum < 0.0 { -1 } else { 0 };

        if sign == 0 {
            self.sign_buf.push_back(sign);
            if self.sign_buf.len() > 20 {
                self.sign_buf.pop_front();
            }
            return;
        }

        // Check if there's a sign change after >= consecutive_period of same sign
        if let Some(&prev_sign) = self.sign_buf.back() {
            if prev_sign != 0 && prev_sign != sign {
                // Count consecutive prior same-sign
                let consecutive = self
                    .sign_buf
                    .iter()
                    .rev()
                    .take_while(|&&s| s == prev_sign)
                    .count();
                if consecutive >= self.consecutive_period {
                    let from = if prev_sign > 0 {
                        MomentumDirection::Positive
                    } else {
                        MomentumDirection::Negative
                    };
                    let to = if sign > 0 {
                        MomentumDirection::Positive
                    } else {
                        MomentumDirection::Negative
                    };
                    self.emitter.emit(MomentumShiftEvent {
                        timestamp: Utc::now(),
                        symbol: self.symbol.clone(),
                        from,
                        to,
                        momentum_value: momentum,
                    });
                }
            }
        }

        self.sign_buf.push_back(sign);
        if self.sign_buf.len() > 20 {
            self.sign_buf.pop_front();
        }
    }
}

// ─── VolumeAnomalyEvent ───────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeAnomalyEvent {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub volume: f64,
    pub threshold: f64,
    /// volume / threshold ratio.
    pub ratio: f64,
}

/// Tukey outlier detection on volume.
///
/// Maintains a rolling window and computes Q1, Q3, IQR on each update.
/// Flags volume > Q3 + 3*IQR as an anomaly (extreme outlier per Tukey).
#[derive(Debug)]
pub struct VolumeAnomaly {
    symbol: String,
    window: usize,
    buf: VecDeque<f64>,
    pub emitter: EventEmitter<VolumeAnomalyEvent>,
}

impl VolumeAnomaly {
    pub fn new(symbol: impl Into<String>, window: usize) -> Self {
        Self {
            symbol: symbol.into(),
            window,
            buf: VecDeque::with_capacity(window + 1),
            emitter: EventEmitter::new(),
        }
    }

    pub fn update(&mut self, volume: f64) {
        if self.buf.len() == self.window {
            self.buf.pop_front();
        }
        self.buf.push_back(volume);

        if self.buf.len() < 10 {
            return; // not enough data
        }

        let mut sorted: Vec<f64> = self.buf.iter().cloned().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();

        let q1 = sorted[n / 4];
        let q3 = sorted[3 * n / 4];
        let iqr = q3 - q1;
        let threshold = q3 + 3.0 * iqr;

        if volume > threshold && threshold > 0.0 {
            self.emitter.emit(VolumeAnomalyEvent {
                timestamp: Utc::now(),
                symbol: self.symbol.clone(),
                volume,
                threshold,
                ratio: volume / threshold,
            });
        }
    }
}

// ─── GapDetector ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GapDirection {
    Up,
    Down,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GapEvent {
    pub symbol: String,
    pub gap_pct: f64,
    pub gap_direction: GapDirection,
    /// True once the gap has been filled (price crossed back into pre-gap range).
    pub filled: bool,
    pub timestamp: DateTime<Utc>,
    /// Previous session close.
    pub prev_close: f64,
    /// Opening price of the gapping bar.
    pub open: f64,
}

/// Detects open gaps and tracks fill status.
///
/// Gap threshold is configurable (default 0.5% = 0.005).
#[derive(Debug)]
pub struct GapDetector {
    symbol: String,
    min_gap_pct: f64,
    prev_close: Option<f64>,
    open_gaps: Vec<GapEvent>,
    pub emitter: EventEmitter<GapEvent>,
}

impl GapDetector {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self::with_threshold(symbol, 0.005)
    }

    pub fn with_threshold(symbol: impl Into<String>, min_gap_pct: f64) -> Self {
        Self {
            symbol: symbol.into(),
            min_gap_pct,
            prev_close: None,
            open_gaps: Vec::new(),
            emitter: EventEmitter::new(),
        }
    }

    /// Feed the opening price of a new bar. Call this at bar open.
    pub fn on_open(&mut self, open: f64) {
        if let Some(pc) = self.prev_close {
            if pc > 1e-12 {
                let gap_pct = (open - pc) / pc;
                if gap_pct.abs() >= self.min_gap_pct {
                    let direction = if gap_pct > 0.0 { GapDirection::Up } else { GapDirection::Down };
                    let event = GapEvent {
                        symbol: self.symbol.clone(),
                        gap_pct: gap_pct.abs(),
                        gap_direction: direction,
                        filled: false,
                        timestamp: Utc::now(),
                        prev_close: pc,
                        open,
                    };
                    self.open_gaps.push(event.clone());
                    self.emitter.emit(event);
                }
            }
        }
    }

    /// Feed bar prices (high and low) to check gap fills.
    pub fn on_bar(&mut self, high: f64, low: f64, close: f64) {
        for gap in self.open_gaps.iter_mut() {
            if gap.filled {
                continue;
            }
            // Fill check: price re-enters the pre-gap zone
            match gap.gap_direction {
                GapDirection::Up => {
                    // Filled when price trades at or below prev_close
                    if low <= gap.prev_close {
                        gap.filled = true;
                        self.emitter.emit(gap.clone());
                    }
                }
                GapDirection::Down => {
                    // Filled when price trades at or above prev_close
                    if high >= gap.prev_close {
                        gap.filled = true;
                        self.emitter.emit(gap.clone());
                    }
                }
            }
        }
        // Remove filled gaps to avoid unbounded growth
        self.open_gaps.retain(|g| !g.filled);
        self.prev_close = Some(close);
    }

    /// Close price setter for session end.
    pub fn set_prev_close(&mut self, close: f64) {
        self.prev_close = Some(close);
    }
}

// ─── OrderFlowReversalEvent ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderFlowReversalEvent {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    /// Cumulative delta at the time of reversal.
    pub cumulative_delta: f64,
    /// Recent average cumulative delta magnitude.
    pub avg_delta: f64,
    /// cumulative_delta.abs() / avg_delta -- always >= 1.5 on event.
    pub magnitude: f64,
}

/// Detects order flow reversals.
///
/// Tracks cumulative delta (buy_vol - sell_vol). When the delta flips sign
/// and its magnitude exceeds 1.5x the recent average magnitude -> reversal.
#[derive(Debug)]
pub struct OrderFlowReversal {
    symbol: String,
    cumulative_delta: f64,
    prev_sign: i8,
    delta_buf: VecDeque<f64>,
    window: usize,
    threshold: f64,
    pub emitter: EventEmitter<OrderFlowReversalEvent>,
}

impl OrderFlowReversal {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self::with_params(symbol, 20, 1.5)
    }

    pub fn with_params(symbol: impl Into<String>, window: usize, threshold: f64) -> Self {
        Self {
            symbol: symbol.into(),
            cumulative_delta: 0.0,
            prev_sign: 0,
            delta_buf: VecDeque::with_capacity(window + 1),
            window,
            threshold,
            emitter: EventEmitter::new(),
        }
    }

    /// Feed a signed volume tick (+buy, -sell).
    pub fn update(&mut self, signed_volume: f64) {
        self.cumulative_delta += signed_volume;
        let sign: i8 = if self.cumulative_delta > 0.0 { 1 } else if self.cumulative_delta < 0.0 { -1 } else { 0 };

        // Record magnitude in rolling buffer
        if self.delta_buf.len() == self.window {
            self.delta_buf.pop_front();
        }
        self.delta_buf.push_back(self.cumulative_delta.abs());

        if sign != 0 && self.prev_sign != 0 && sign != self.prev_sign {
            // Sign flip detected -- check magnitude
            if self.delta_buf.len() >= 5 {
                let avg_delta = self.delta_buf.iter().sum::<f64>() / self.delta_buf.len() as f64;
                if avg_delta > 1e-12 {
                    let magnitude = self.cumulative_delta.abs() / avg_delta;
                    if magnitude >= self.threshold {
                        self.emitter.emit(OrderFlowReversalEvent {
                            timestamp: Utc::now(),
                            symbol: self.symbol.clone(),
                            cumulative_delta: self.cumulative_delta,
                            avg_delta,
                            magnitude,
                        });
                    }
                }
            }
        }

        if sign != 0 {
            self.prev_sign = sign;
        }
    }

    pub fn cumulative_delta(&self) -> f64 { self.cumulative_delta }

    /// Reset cumulative delta (e.g. at session start).
    pub fn reset_delta(&mut self) {
        self.cumulative_delta = 0.0;
        self.prev_sign = 0;
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn volatility_breakout_emits_on_spike() {
        let mut vb = VolatilityBreakout::new("TEST");
        // Prime with 40 low-vol bars (range = 1)
        for i in 0..40 {
            vb.update(100.0 + i as f64 * 0.01, 99.9 + i as f64 * 0.01, 100.0 + i as f64 * 0.01);
        }
        // High-vol bar
        vb.update(110.0, 90.0, 100.0);
        let events = vb.emitter.drain();
        assert!(!events.is_empty(), "should emit breakout event");
        assert!(events[0].magnitude >= 2.0);
    }

    #[test]
    fn volume_anomaly_detects_spike() {
        let mut va = VolumeAnomaly::new("TEST", 50);
        for _ in 0..49 {
            va.update(1000.0);
        }
        va.update(100_000.0); // huge spike
        let events = va.emitter.drain();
        assert!(!events.is_empty(), "should detect volume anomaly");
    }
}
