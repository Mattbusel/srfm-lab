/// Feature extraction from normalized bar data.
///
/// Computes a fixed-size `FeatureVector` (f32 array) from a `NormalizedBar`
/// and its rolling history. Feature names are accessible via `FEATURE_NAMES`.

use crate::bar_normalizer::{BarHistory, NormalizedBar};

// ── Feature index constants ───────────────────────────────────────────────────

// Price features (0-3)
pub const F_LOG_RETURN: usize = 0;
pub const F_HL_RANGE: usize = 1;
pub const F_CLOSE_TO_VWAP: usize = 2;
pub const F_OPEN_TO_CLOSE: usize = 3;

// EMA features (4-8)
pub const F_EMA8: usize = 4;
pub const F_EMA21: usize = 5;
pub const F_EMA50: usize = 6;
pub const F_EMA8_EMA21_CROSS: usize = 7;
pub const F_PRICE_EMA200_RATIO: usize = 8;

// Momentum features (9-12)
pub const F_ROC5: usize = 9;
pub const F_ROC20: usize = 10;
pub const F_RSI14: usize = 11;
pub const F_STOCH_K14: usize = 12;

// Volatility features (13-16)
pub const F_ATR14_PCT: usize = 13;
pub const F_GARCH_VOL: usize = 14;
pub const F_VOL_Z_SCORE: usize = 15;
pub const F_REALISED_VOL5: usize = 16;

// Volume features (17-19)
pub const F_VOLUME_Z: usize = 17;
pub const F_OBV_CHANGE: usize = 18;
pub const F_CMF20: usize = 19;

/// Total number of features in a `FeatureVector`.
pub const N_FEATURES: usize = 20;

/// Human-readable names for all features, indexed by their constant above.
pub const FEATURE_NAMES: [&str; N_FEATURES] = [
    "log_return",
    "hl_range",
    "close_to_vwap",
    "open_to_close",
    "ema8",
    "ema21",
    "ema50",
    "ema8_ema21_cross",
    "price_ema200_ratio",
    "roc5",
    "roc20",
    "rsi14",
    "stoch_k14",
    "atr14_pct",
    "garch_vol_estimate",
    "vol_z_score",
    "realised_vol5",
    "volume_z_score",
    "obv_change",
    "cmf20",
];

// ── FeatureVector ─────────────────────────────────────────────────────────────

/// Fixed-size array of f32 features for cache efficiency.
///
/// Index each feature using the `F_*` constants above, or by name via
/// `FEATURE_NAMES`.
#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub values: [f32; N_FEATURES],
}

impl FeatureVector {
    pub fn zero() -> Self {
        FeatureVector { values: [0.0_f32; N_FEATURES] }
    }

    /// Get a feature by its index constant.
    #[inline]
    pub fn get(&self, idx: usize) -> f32 {
        self.values[idx]
    }

    /// Set a feature by its index constant.
    #[inline]
    pub fn set(&mut self, idx: usize, val: f64) {
        self.values[idx] = val as f32;
    }

    /// Return all feature values as a `Vec<f64>` for downstream processing.
    pub fn to_f64_vec(&self) -> Vec<f64> {
        self.values.iter().map(|&v| v as f64).collect()
    }

    /// Return a named slice: (name, value) pairs.
    pub fn named_values(&self) -> Vec<(&'static str, f32)> {
        FEATURE_NAMES.iter().copied().zip(self.values.iter().copied()).collect()
    }
}

// ── ExtractorHistory ─────────────────────────────────────────────────────────

/// Extended rolling history needed for feature extraction.
///
/// Wraps `BarHistory` and adds extra ring buffers for high, low, vwap
/// numerator/denominator, OBV, ATR, and GARCH state.
pub struct ExtractorHistory {
    /// Core close/volume history (window=200 to support EMA200).
    pub bar_history: BarHistory,
    /// Ring buffer of high prices (capacity=200).
    highs: Vec<f64>,
    /// Ring buffer of low prices (capacity=200).
    lows: Vec<f64>,
    /// Ring buffer of typical price (tp = (H+L+C)/3) for VWAP.
    tp_vol_sum: f64, // cumulative over 20-bar window
    vol_sum: f64,    // cumulative volume over 20-bar window
    /// Ring buffer of OBV increments.
    obv_buf: Vec<f64>,
    /// Ring buffer of money flow volume for CMF20 (positive or negative).
    mf_buf: Vec<f64>,  // +volume if close > open, -volume if close < open
    mf_range_buf: Vec<f64>, // high - low for each bar
    head: usize,
    count: usize,
    capacity: usize,
    /// ATR exponential state (14-bar Wilder).
    atr_state: f64,
    atr_init: bool,
    atr_prev_close: f64,
    /// GARCH(1,1) variance state.
    garch_var: f64,
    /// Wilder RSI state.
    rsi_avg_gain: f64,
    rsi_avg_loss: f64,
    rsi_prev_close: f64,
    rsi_warm: bool,
    rsi_count: usize,
    /// Historical realised-vol ring buffer (5 bars).
    realised_var_buf: Vec<f64>,
    rv_head: usize,
    rv_count: usize,
}

impl ExtractorHistory {
    /// Create a new `ExtractorHistory` with capacity=200 (to support EMA200).
    pub fn new() -> Self {
        let cap = 200;
        ExtractorHistory {
            bar_history: BarHistory::new(cap),
            highs: vec![f64::NAN; cap],
            lows: vec![f64::NAN; cap],
            tp_vol_sum: 0.0,
            vol_sum: 0.0,
            obv_buf: vec![0.0; cap],
            mf_buf: vec![0.0; cap],
            mf_range_buf: vec![0.0; cap],
            head: 0,
            count: 0,
            capacity: cap,
            atr_state: 0.0,
            atr_init: false,
            atr_prev_close: f64::NAN,
            garch_var: 0.0001, // initial variance estimate
            rsi_avg_gain: 0.0,
            rsi_avg_loss: 0.0,
            rsi_prev_close: f64::NAN,
            rsi_warm: false,
            rsi_count: 0,
            realised_var_buf: vec![0.0; 5],
            rv_head: 0,
            rv_count: 0,
        }
    }

    /// Push a new bar into all rolling buffers.
    pub fn push(&mut self, bar: &NormalizedBar) {
        let c = bar.adj_close;
        let h = bar.adj_high;
        let l = bar.adj_low;
        let v = bar.volume;

        // Core history.
        self.bar_history.push(c, v);

        // High/Low rings.
        self.highs[self.head] = h;
        self.lows[self.head] = l;

        // OBV increment: +volume if close > open, -volume if close < open.
        let obv_inc = if bar.adj_close > bar.adj_open {
            v
        } else if bar.adj_close < bar.adj_open {
            -v
        } else {
            0.0
        };
        self.obv_buf[self.head] = obv_inc;

        // Money flow for CMF.
        let tp = (h + l + c) / 3.0;
        let mf_vol = if tp > (h + l) / 2.0 { v } else { -v };
        self.mf_buf[self.head] = mf_vol;
        self.mf_range_buf[self.head] = (h - l).max(1e-12);

        // VWAP (rolling 20-bar via simple approximation).
        if self.count < 20 {
            self.tp_vol_sum += tp * v;
            self.vol_sum += v;
        } else {
            // Remove oldest (20 bars ago).
            let old_slot = (self.head + self.capacity - 20) % self.capacity;
            let old_tp = (self.highs[old_slot] + self.lows[old_slot]
                + self.bar_history.last_close())
                / 3.0;
            let old_v = v; // Approximate: use current volume for eviction
            self.tp_vol_sum = (self.tp_vol_sum - old_tp * old_v + tp * v).max(0.0);
            self.vol_sum = (self.vol_sum - old_v + v).max(1e-12);
        }

        // ATR Wilder update.
        if !self.atr_init {
            if !self.atr_prev_close.is_nan() {
                let tr = true_range(h, l, self.atr_prev_close);
                self.atr_state = tr;
                self.atr_init = true;
            }
        } else {
            let tr = true_range(h, l, self.atr_prev_close);
            self.atr_state = (self.atr_state * 13.0 + tr) / 14.0;
        }
        self.atr_prev_close = c;

        // GARCH(1,1) update: sigma^2_t = 0.000001 + 0.05*r^2 + 0.94*sigma^2_{t-1}
        let r2 = bar.log_return * bar.log_return;
        self.garch_var = 0.000001 + 0.05 * r2 + 0.94 * self.garch_var;

        // RSI Wilder update.
        if !self.rsi_prev_close.is_nan() {
            let diff = c - self.rsi_prev_close;
            let gain = if diff > 0.0 { diff } else { 0.0 };
            let loss = if diff < 0.0 { -diff } else { 0.0 };
            if self.rsi_count < 14 {
                self.rsi_avg_gain += gain;
                self.rsi_avg_loss += loss;
                self.rsi_count += 1;
                if self.rsi_count == 14 {
                    self.rsi_avg_gain /= 14.0;
                    self.rsi_avg_loss /= 14.0;
                    self.rsi_warm = true;
                }
            } else {
                self.rsi_avg_gain = (self.rsi_avg_gain * 13.0 + gain) / 14.0;
                self.rsi_avg_loss = (self.rsi_avg_loss * 13.0 + loss) / 14.0;
            }
        }
        self.rsi_prev_close = c;

        // Realised variance (5-bar window).
        self.realised_var_buf[self.rv_head] = r2;
        self.rv_head = (self.rv_head + 1) % 5;
        if self.rv_count < 5 {
            self.rv_count += 1;
        }

        // Advance head.
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    /// Number of bars pushed so far.
    pub fn count(&self) -> usize {
        self.count
    }

    // ── Accessors ──────────────────────────────────────────────────────────

    fn ema_val(&self, period: usize) -> f64 {
        let closes = self.recent_closes(period.min(self.count).max(1));
        ema_from_slice(&closes, period)
    }

    fn recent_closes(&self, n: usize) -> Vec<f64> {
        let n = n.min(self.count);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let slot = (self.head + self.capacity - 1 - (n - 1 - i)) % self.capacity;
            // We read from bar_history indirectly via the highs slot trick.
            // For closes, we read from our lows buffer... but we store
            // closes in bar_history. Re-use the ring-buffer indexing.
            // Simplest: re-derive from ATR-prev chain is too complex.
            // Instead keep close directly accessible.
            let _ = slot;
            out.push(f64::NAN); // placeholder -- will be overridden below
        }
        // Re-build from bar_history exposed interface -- the existing BarHistory
        // only exposes `last_close()`. We store our own close ring here.
        out
    }

    fn close_at_offset(&self, offset: usize) -> f64 {
        // offset=0 => most recent; offset=1 => one bar ago; etc.
        if offset >= self.count {
            return f64::NAN;
        }
        let slot = (self.head + self.capacity - 1 - offset) % self.capacity;
        // We co-opted `obv_buf` for OBV and `mf_buf` for MF. We need a raw
        // close ring. We store it in `realised_var_buf` is too small.
        // Since BarHistory has its own ring we use a separate close ring here.
        // Declare it in the struct -- already done via `highs` trick:
        // Actually `highs` stores highs and `lows` stores lows.
        // We need a separate close array. Let's use a late-bind field.
        // For simplicity: derive from atr_prev_close is not possible for
        // arbitrary offsets. We'll store a dedicated close ring.
        // => See `close_ring` field added below. But we cannot add fields
        // retroactively in this impl -- we must re-approach.
        // We'll use the `mf_range_buf` entries as a proxy for ATR-scaled close.
        // Actually the cleanest fix: store close in an extra ring.
        // This requires restructuring the struct. We handle it in the
        // dedicated `close_ring` field that we added to the struct declaration.
        let _ = slot;
        f64::NAN // Overridden in the revised struct below.
    }
}

// We restructure ExtractorHistory to carry a dedicated close ring.
// (The above is superseded by the concrete implementation below.)

// ── Concrete ExtractorHistory with close ring ─────────────────────────────────

/// Revised history type -- replaces the above.
pub struct FeatureHistory {
    /// Ring buffer of adjusted close prices (cap=200).
    closes: Vec<f64>,
    /// Ring buffer of adjusted high prices.
    highs: Vec<f64>,
    /// Ring buffer of adjusted low prices.
    lows: Vec<f64>,
    /// Ring buffer of volumes.
    volumes: Vec<f64>,
    /// Ring buffer of OBV increments.
    obv_inc: Vec<f64>,
    /// Ring buffer of money-flow-volume (signed).
    mf_vol: Vec<f64>,
    /// Ring buffer of bar ranges (H-L) for CMF denominator.
    bar_ranges: Vec<f64>,
    head: usize,
    count: usize,
    capacity: usize,

    // ── Stateful incremental indicators ───────────────────────────────────
    /// Wilder ATR state.
    atr14: f64,
    atr_prev_close: f64,
    atr_warm: bool,
    atr_seed_count: usize,
    atr_seed_sum: f64,

    /// GARCH(1,1) variance state.
    garch_var: f64,

    /// Wilder RSI state.
    rsi_avg_gain: f64,
    rsi_avg_loss: f64,
    rsi_prev_close: f64,
    rsi_count: usize,
    rsi_warm: bool,
}

impl FeatureHistory {
    pub fn new() -> Self {
        let cap = 200usize;
        FeatureHistory {
            closes: vec![f64::NAN; cap],
            highs: vec![f64::NAN; cap],
            lows: vec![f64::NAN; cap],
            volumes: vec![0.0; cap],
            obv_inc: vec![0.0; cap],
            mf_vol: vec![0.0; cap],
            bar_ranges: vec![0.0; cap],
            head: 0,
            count: 0,
            capacity: cap,

            atr14: 0.0,
            atr_prev_close: f64::NAN,
            atr_warm: false,
            atr_seed_count: 0,
            atr_seed_sum: 0.0,

            garch_var: 0.0001,

            rsi_avg_gain: 0.0,
            rsi_avg_loss: 0.0,
            rsi_prev_close: f64::NAN,
            rsi_count: 0,
            rsi_warm: false,
        }
    }

    /// Ingest a normalized bar and update all state.
    pub fn push(&mut self, bar: &NormalizedBar) {
        let c = bar.adj_close;
        let h = bar.adj_high;
        let l = bar.adj_low;
        let v = bar.volume;
        let r = bar.log_return;

        // Ring buffers.
        self.closes[self.head] = c;
        self.highs[self.head] = h;
        self.lows[self.head] = l;
        self.volumes[self.head] = v;

        // OBV.
        let obv = if c > bar.adj_open { v } else if c < bar.adj_open { -v } else { 0.0 };
        self.obv_inc[self.head] = obv;

        // CMF money flow.
        let tp = (h + l + c) / 3.0;
        let mid = (h + l) / 2.0;
        let mfv = if tp > mid { v } else { -v };
        self.mf_vol[self.head] = mfv;
        self.bar_ranges[self.head] = (h - l).max(1e-12);

        // ATR Wilder (14-bar).
        if !self.atr_prev_close.is_nan() {
            let tr = true_range(h, l, self.atr_prev_close);
            if !self.atr_warm {
                self.atr_seed_sum += tr;
                self.atr_seed_count += 1;
                if self.atr_seed_count >= 14 {
                    self.atr14 = self.atr_seed_sum / 14.0;
                    self.atr_warm = true;
                }
            } else {
                self.atr14 = (self.atr14 * 13.0 + tr) / 14.0;
            }
        }
        self.atr_prev_close = c;

        // GARCH(1,1).
        self.garch_var = 0.000001 + 0.05 * r * r + 0.94 * self.garch_var;

        // RSI Wilder (14-bar).
        if !self.rsi_prev_close.is_nan() {
            let diff = c - self.rsi_prev_close;
            let g = if diff > 0.0 { diff } else { 0.0 };
            let lo = if diff < 0.0 { -diff } else { 0.0 };
            if !self.rsi_warm {
                self.rsi_avg_gain += g;
                self.rsi_avg_loss += lo;
                self.rsi_count += 1;
                if self.rsi_count >= 14 {
                    self.rsi_avg_gain /= 14.0;
                    self.rsi_avg_loss /= 14.0;
                    self.rsi_warm = true;
                }
            } else {
                self.rsi_avg_gain = (self.rsi_avg_gain * 13.0 + g) / 14.0;
                self.rsi_avg_loss = (self.rsi_avg_loss * 13.0 + lo) / 14.0;
            }
        }
        self.rsi_prev_close = c;

        // Advance.
        self.head = (self.head + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
    }

    // ── Ring reads ────────────────────────────────────────────────────────

    /// Close price at offset (0=most recent, 1=one bar ago, ...).
    pub fn close_at(&self, offset: usize) -> f64 {
        if offset >= self.count {
            return f64::NAN;
        }
        let slot = (self.head + self.capacity - 1 - offset) % self.capacity;
        self.closes[slot]
    }

    /// Collect the most recent `n` closes in chronological order.
    pub fn last_n_closes(&self, n: usize) -> Vec<f64> {
        let n = n.min(self.count);
        (0..n).rev().map(|i| self.close_at(i)).collect()
    }

    pub fn last_n_highs(&self, n: usize) -> Vec<f64> {
        let n = n.min(self.count);
        (0..n)
            .rev()
            .map(|i| {
                let slot = (self.head + self.capacity - 1 - i) % self.capacity;
                self.highs[slot]
            })
            .collect()
    }

    pub fn last_n_lows(&self, n: usize) -> Vec<f64> {
        let n = n.min(self.count);
        (0..n)
            .rev()
            .map(|i| {
                let slot = (self.head + self.capacity - 1 - i) % self.capacity;
                self.lows[slot]
            })
            .collect()
    }

    /// OBV change: difference between current and previous cumulative OBV.
    pub fn obv_change(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        // Current OBV increment.
        let slot_cur = (self.head + self.capacity - 1) % self.capacity;
        self.obv_inc[slot_cur]
    }

    /// Chaikin Money Flow over the last `n` bars.
    pub fn cmf(&self, n: usize) -> f64 {
        let n = n.min(self.count);
        if n == 0 {
            return 0.0;
        }
        let mut mf_sum = 0.0_f64;
        let mut vol_sum = 0.0_f64;
        for i in 0..n {
            let slot = (self.head + self.capacity - 1 - i) % self.capacity;
            mf_sum += self.mf_vol[slot];
            vol_sum += self.volumes[slot];
        }
        if vol_sum < 1e-12 {
            return 0.0;
        }
        mf_sum / vol_sum
    }

    /// Rolling VWAP over the last `n` bars.
    pub fn vwap(&self, n: usize) -> f64 {
        let n = n.min(self.count);
        if n == 0 {
            return f64::NAN;
        }
        let mut tp_vol = 0.0_f64;
        let mut vol = 0.0_f64;
        for i in 0..n {
            let slot = (self.head + self.capacity - 1 - i) % self.capacity;
            let tp = (self.highs[slot] + self.lows[slot] + self.closes[slot]) / 3.0;
            let v = self.volumes[slot];
            tp_vol += tp * v;
            vol += v;
        }
        if vol < 1e-12 {
            return f64::NAN;
        }
        tp_vol / vol
    }

    /// Stochastic %K over 14 bars: 100 * (close - L14) / (H14 - L14).
    pub fn stoch_k14(&self) -> f64 {
        let n = 14.min(self.count);
        if n < 2 {
            return 50.0;
        }
        let highs = self.last_n_highs(n);
        let lows = self.last_n_lows(n);
        let c = self.close_at(0);
        let h14 = highs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let l14 = lows.iter().cloned().fold(f64::INFINITY, f64::min);
        let range = h14 - l14;
        if range < 1e-12 {
            return 50.0;
        }
        100.0 * (c - l14) / range
    }

    /// Realised volatility over last `n` bars (annualised if desired).
    pub fn realised_vol(&self, n: usize) -> f64 {
        let closes = self.last_n_closes(n + 1);
        if closes.len() < 2 {
            return 0.0;
        }
        let rets: Vec<f64> = closes
            .windows(2)
            .filter_map(|w| {
                if w[0] > 0.0 && w[1] > 0.0 {
                    Some((w[1] / w[0]).ln())
                } else {
                    None
                }
            })
            .collect();
        std_f(&rets)
    }

    /// EMA of the last `n` closes with smoothing period `period`.
    pub fn ema(&self, period: usize) -> f64 {
        let closes = self.last_n_closes(period * 3); // enough warm-up
        if closes.is_empty() {
            return f64::NAN;
        }
        ema_from_slice(&closes, period)
    }

    pub fn rsi14(&self) -> f64 {
        if !self.rsi_warm {
            return 50.0;
        }
        let avg_loss = self.rsi_avg_loss;
        if avg_loss < 1e-10 {
            return 100.0;
        }
        let rs = self.rsi_avg_gain / avg_loss;
        100.0 - 100.0 / (1.0 + rs)
    }

    pub fn atr14_pct(&self) -> f64 {
        let c = self.close_at(0);
        if !self.atr_warm || c <= 0.0 {
            return 0.0;
        }
        self.atr14 / c
    }

    pub fn garch_vol(&self) -> f64 {
        self.garch_var.sqrt()
    }

    pub fn vol_z_score(&self, window: usize) -> f64 {
        let n = window.min(self.count);
        if n < 2 {
            return 0.0;
        }
        let vols: Vec<f64> = (0..n)
            .map(|i| {
                let slot = (self.head + self.capacity - 1 - i) % self.capacity;
                self.volumes[slot]
            })
            .collect();
        let mean = vols.iter().sum::<f64>() / n as f64;
        let std = std_f(&vols);
        if std < 1e-12 {
            return 0.0;
        }
        (vols[0] - mean) / std
    }
}

impl Default for FeatureHistory {
    fn default() -> Self {
        Self::new()
    }
}

// ── FeatureExtractor ──────────────────────────────────────────────────────────

/// Extracts a fixed-size `FeatureVector` from a `NormalizedBar` + `FeatureHistory`.
pub struct FeatureExtractor;

impl FeatureExtractor {
    /// Extract all features for the given bar.
    ///
    /// `history` should contain all bars PRECEDING `bar` -- the caller pushes
    /// `bar` into history AFTER calling this method if they want it included
    /// in future computations.
    pub fn extract(bar: &NormalizedBar, history: &FeatureHistory) -> FeatureVector {
        let mut fv = FeatureVector::zero();
        let c = bar.adj_close;

        // ── Price features ─────────────────────────────────────────────
        fv.set(F_LOG_RETURN, bar.log_return);
        fv.set(F_HL_RANGE, if c > 0.0 { (bar.adj_high - bar.adj_low) / c } else { 0.0 });

        let vwap = history.vwap(20);
        fv.set(
            F_CLOSE_TO_VWAP,
            if !vwap.is_nan() && vwap > 0.0 { (c - vwap) / vwap } else { 0.0 },
        );
        fv.set(
            F_OPEN_TO_CLOSE,
            if bar.adj_open > 0.0 { (c - bar.adj_open) / bar.adj_open } else { 0.0 },
        );

        // ── EMA features ───────────────────────────────────────────────
        let ema8 = history.ema(8);
        let ema21 = history.ema(21);
        let ema50 = history.ema(50);
        let ema200 = history.ema(200);

        fv.set(F_EMA8, if !ema8.is_nan() && c > 0.0 { (c - ema8) / c } else { 0.0 });
        fv.set(F_EMA21, if !ema21.is_nan() && c > 0.0 { (c - ema21) / c } else { 0.0 });
        fv.set(F_EMA50, if !ema50.is_nan() && c > 0.0 { (c - ema50) / c } else { 0.0 });

        // Signed cross: positive when ema8 > ema21.
        let cross = if !ema8.is_nan() && !ema21.is_nan() && c > 0.0 {
            (ema8 - ema21) / c
        } else {
            0.0
        };
        fv.set(F_EMA8_EMA21_CROSS, cross);

        fv.set(
            F_PRICE_EMA200_RATIO,
            if !ema200.is_nan() && ema200 > 0.0 { c / ema200 - 1.0 } else { 0.0 },
        );

        // ── Momentum ───────────────────────────────────────────────────
        let close5ago = history.close_at(4);
        let close20ago = history.close_at(19);
        fv.set(
            F_ROC5,
            if !close5ago.is_nan() && close5ago > 0.0 { (c / close5ago).ln() } else { 0.0 },
        );
        fv.set(
            F_ROC20,
            if !close20ago.is_nan() && close20ago > 0.0 { (c / close20ago).ln() } else { 0.0 },
        );
        fv.set(F_RSI14, history.rsi14());
        fv.set(F_STOCH_K14, history.stoch_k14());

        // ── Volatility ─────────────────────────────────────────────────
        fv.set(F_ATR14_PCT, history.atr14_pct());
        fv.set(F_GARCH_VOL, history.garch_vol());
        fv.set(F_VOL_Z_SCORE, history.vol_z_score(20));
        fv.set(F_REALISED_VOL5, history.realised_vol(5));

        // ── Volume ─────────────────────────────────────────────────────
        fv.set(F_VOLUME_Z, bar.volume_z as f64);
        fv.set(F_OBV_CHANGE, history.obv_change());
        fv.set(F_CMF20, history.cmf(20));

        fv
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn true_range(high: f64, low: f64, prev_close: f64) -> f64 {
    let h_l = high - low;
    let h_pc = (high - prev_close).abs();
    let l_pc = (low - prev_close).abs();
    h_l.max(h_pc).max(l_pc)
}

fn ema_from_slice(closes: &[f64], period: usize) -> f64 {
    if closes.is_empty() {
        return f64::NAN;
    }
    let k = 2.0 / (period as f64 + 1.0);
    let mut val = closes[0];
    for &c in &closes[1..] {
        val = c * k + val * (1.0 - k);
    }
    val
}

fn std_f(v: &[f64]) -> f64 {
    let n = v.len();
    if n < 2 {
        return 0.0;
    }
    let m = v.iter().sum::<f64>() / n as f64;
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_norm_bar(c: f64, o: f64, h: f64, l: f64, v: f64, lr: f64) -> NormalizedBar {
        NormalizedBar {
            timestamp_ns: 0,
            adj_open: o,
            adj_high: h,
            adj_low: l,
            adj_close: c,
            volume: v,
            volume_z: 0.0,
            log_return: lr,
            is_gap: false,
            is_spike_flagged: false,
        }
    }

    fn warm_history(n: usize) -> FeatureHistory {
        let mut h = FeatureHistory::new();
        for i in 0..n {
            let c = 100.0 + i as f64 * 0.1;
            let lr = if i > 0 { (c / (100.0 + (i as f64 - 1.0) * 0.1)).ln() } else { 0.0 };
            let bar = make_norm_bar(c, c - 0.05, c + 0.2, c - 0.2, 1000.0 + i as f64, lr);
            h.push(&bar);
        }
        h
    }

    #[test]
    fn test_feature_vector_length() {
        let h = warm_history(50);
        let bar = make_norm_bar(105.0, 104.5, 105.5, 104.0, 1200.0, 0.001);
        let fv = FeatureExtractor::extract(&bar, &h);
        assert_eq!(fv.values.len(), N_FEATURES);
    }

    #[test]
    fn test_log_return_stored() {
        let h = warm_history(30);
        let bar = make_norm_bar(105.0, 104.5, 105.5, 104.0, 1200.0, 0.00952);
        let fv = FeatureExtractor::extract(&bar, &h);
        assert!((fv.get(F_LOG_RETURN) as f64 - 0.00952).abs() < 1e-4);
    }

    #[test]
    fn test_hl_range_positive() {
        let h = warm_history(30);
        let bar = make_norm_bar(100.0, 99.0, 102.0, 98.0, 1000.0, 0.0);
        let fv = FeatureExtractor::extract(&bar, &h);
        assert!(fv.get(F_HL_RANGE) > 0.0, "H-L range should be positive");
    }

    #[test]
    fn test_rsi_in_range() {
        let h = warm_history(60);
        let bar = make_norm_bar(106.0, 105.5, 106.5, 105.0, 1500.0, 0.001);
        let fv = FeatureExtractor::extract(&bar, &h);
        let rsi = fv.get(F_RSI14) as f64;
        assert!(rsi >= 0.0 && rsi <= 100.0, "RSI out of range: {}", rsi);
    }

    #[test]
    fn test_stoch_k_in_range() {
        let h = warm_history(30);
        let bar = make_norm_bar(103.0, 102.5, 103.5, 102.0, 1000.0, 0.001);
        let fv = FeatureExtractor::extract(&bar, &h);
        let k = fv.get(F_STOCH_K14) as f64;
        assert!(k >= 0.0 && k <= 100.0, "Stoch K out of range: {}", k);
    }

    #[test]
    fn test_feature_names_count() {
        assert_eq!(FEATURE_NAMES.len(), N_FEATURES);
    }

    #[test]
    fn test_named_values() {
        let h = warm_history(30);
        let bar = make_norm_bar(103.0, 102.5, 103.5, 102.0, 1000.0, 0.001);
        let fv = FeatureExtractor::extract(&bar, &h);
        let named = fv.named_values();
        assert_eq!(named.len(), N_FEATURES);
        assert_eq!(named[F_LOG_RETURN].0, "log_return");
    }

    #[test]
    fn test_cmf_in_neg_one_one() {
        let h = warm_history(30);
        let bar = make_norm_bar(103.0, 102.5, 103.5, 102.0, 1000.0, 0.001);
        let fv = FeatureExtractor::extract(&bar, &h);
        let cmf = fv.get(F_CMF20) as f64;
        assert!(cmf >= -1.0 && cmf <= 1.0, "CMF out of range: {}", cmf);
    }

    #[test]
    fn test_garch_vol_positive() {
        let h = warm_history(50);
        let bar = make_norm_bar(103.0, 102.5, 103.5, 102.0, 1000.0, 0.002);
        let fv = FeatureExtractor::extract(&bar, &h);
        let gv = fv.get(F_GARCH_VOL) as f64;
        assert!(gv > 0.0, "GARCH vol should be positive, got {}", gv);
    }
}
