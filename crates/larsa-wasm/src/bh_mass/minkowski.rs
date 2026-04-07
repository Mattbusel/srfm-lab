//! minkowski.rs -- BH mass computation using Minkowski spacetime metric.
//!
//! Theoretical basis:
//! The SRFM model maps price dynamics onto a 1+1 dimensional Minkowski spacetime
//! where time is bar index and space is log-price. The Minkowski interval is:
//!
//!   ds^2 = c^2 * dt^2 - dx^2
//!
//! where c is the "speed of information" (compressibility factor) and dx is the
//! log-price change. When ds^2 > 0 (timelike motion), the market is in a
//! "causal" or compressed regime -- BH mass accumulates. When ds^2 < 0
//! (spacelike), the move is too fast relative to c, and mass decays.
//!
//! BH mass is an EWMA of the Minkowski interval, tracking how long the market
//! has been in a compressed (CTL) state. High mass signals potential reversal
//! or breakout from compression.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// BhMassState -- streaming state machine for tick-by-tick BH mass updates
// ---------------------------------------------------------------------------

/// Per-bar BH mass state. Maintains EWMA of Minkowski interval.
#[wasm_bindgen]
pub struct BhMassState {
    /// Compressibility factor c (speed-of-light analog). Typical: 0.001 - 0.01.
    cf: f64,
    /// Mass formation threshold. When mass >= bh_form, a BH is active.
    bh_form: f64,
    /// Mass decay rate per spacelike bar (ds^2 < 0). Typical: 0.94 - 0.97.
    bh_decay: f64,
    /// EWMA smoothing factor for mass accumulation.
    mass_alpha: f64,
    /// Current accumulated BH mass.
    pub mass: f64,
    /// Previous log-price.
    prev_log_price: f64,
    /// Whether a BH is currently active.
    pub bh_active: bool,
    /// Number of consecutive timelike (CTL) bars.
    pub ctl_count: i32,
    /// Last computed Minkowski interval ds^2.
    pub last_ds2: f64,
    /// Initialized flag.
    initialized: bool,
}

#[wasm_bindgen]
impl BhMassState {
    /// Create a new BH mass state machine.
    /// - cf: compressibility factor (0.001 for equity, 0.0001 for crypto)
    /// - bh_form: mass threshold for BH formation (0.6 - 0.8 typical)
    /// - bh_decay: mass decay multiplier per disruptive bar (0.90 - 0.97)
    #[wasm_bindgen(constructor)]
    pub fn new(cf: f64, bh_form: f64, bh_decay: f64) -> BhMassState {
        BhMassState {
            cf,
            bh_form,
            bh_decay,
            mass_alpha: 0.03,  // EWMA alpha for mass accumulation
            mass: 0.0,
            prev_log_price: 0.0,
            bh_active: false,
            ctl_count: 0,
            last_ds2: 0.0,
            initialized: false,
        }
    }

    /// Feed the next OHLCV bar. Uses close price for mass computation.
    /// Returns the current BH mass value.
    pub fn update_close(&mut self, close: f64) -> f64 {
        let log_price = close.ln();
        if !self.initialized {
            self.prev_log_price = log_price;
            self.initialized = true;
            return 0.0;
        }

        let dx = log_price - self.prev_log_price;
        let dt = 1.0; // normalized bar time
        let ds2 = self.cf * self.cf * dt * dt - dx * dx;

        self.last_ds2 = ds2;

        if ds2 > 0.0 {
            // Timelike (CTL): market moves slower than c, mass accumulates
            self.mass = self.mass * (1.0 - self.mass_alpha) + self.mass_alpha;
            self.ctl_count += 1;
        } else {
            // Spacelike: market moves too fast, mass decays
            self.mass *= self.bh_decay;
            self.ctl_count = 0;
        }

        self.bh_active = self.mass >= self.bh_form && self.ctl_count >= 5;
        self.prev_log_price = log_price;
        self.mass
    }

    /// Update using high/low/close for a richer ds^2 estimate.
    /// Uses the bar's typical price and range to adjust the effective dx.
    pub fn update_bar(&mut self, high: f64, low: f64, close: f64) -> f64 {
        let log_price = ((high + low + close) / 3.0).ln();
        if !self.initialized {
            self.prev_log_price = log_price;
            self.initialized = true;
            return 0.0;
        }

        // Range-adjusted dx: use bar range as uncertainty in price displacement
        let range_frac = (high - low) / (close.abs() + 1e-12);
        let dx = log_price - self.prev_log_price;
        // Effective displacement includes intra-bar range
        let dx_eff = dx.abs() + range_frac * 0.5;

        let ds2 = self.cf * self.cf - dx_eff * dx_eff;
        self.last_ds2 = ds2;

        if ds2 > 0.0 {
            self.mass = self.mass * (1.0 - self.mass_alpha) + self.mass_alpha;
            self.ctl_count += 1;
        } else {
            self.mass *= self.bh_decay;
            self.ctl_count = 0;
        }

        self.bh_active = self.mass >= self.bh_form && self.ctl_count >= 5;
        self.prev_log_price = log_price;
        self.mass
    }

    pub fn reset(&mut self) {
        self.mass = 0.0;
        self.ctl_count = 0;
        self.bh_active = false;
        self.last_ds2 = 0.0;
        self.initialized = false;
    }
}

// ---------------------------------------------------------------------------
// MultiTimeframeMass -- three-timeframe BH mass for dashboard visualization
// ---------------------------------------------------------------------------

/// Three-timeframe BH mass state for simultaneous 15m/1h/4h computation.
/// Higher timeframes use proportionally larger cf (less sensitive).
#[wasm_bindgen]
pub struct MultiTimeframeMass {
    m15: BhMassState,   // 15-minute timeframe
    m60: BhMassState,   // 1-hour timeframe
    m240: BhMassState,  // 4-hour timeframe
    /// Bar counter for timeframe aggregation
    bar_count: u64,
    /// Accumulated OHLC for 1h aggregation (4 x 15m bars)
    h1_open: f64,
    h1_high: f64,
    h1_low: f64,
    h1_close: f64,
    h1_bars: u8,
    /// Accumulated OHLC for 4h aggregation (16 x 15m bars)
    h4_open: f64,
    h4_high: f64,
    h4_low: f64,
    h4_close: f64,
    h4_bars: u8,
}

#[wasm_bindgen]
impl MultiTimeframeMass {
    /// Create multi-timeframe mass state.
    /// cf_15m: base compressibility factor for 15m timeframe.
    /// 1h cf = cf_15m * sqrt(4), 4h cf = cf_15m * sqrt(16) (volatility scaling).
    #[wasm_bindgen(constructor)]
    pub fn new(cf_15m: f64, bh_form: f64, bh_decay: f64) -> MultiTimeframeMass {
        MultiTimeframeMass {
            m15:  BhMassState::new(cf_15m,          bh_form, bh_decay),
            m60:  BhMassState::new(cf_15m * 2.0,    bh_form, bh_decay),
            m240: BhMassState::new(cf_15m * 4.0,    bh_form, bh_decay),
            bar_count: 0,
            h1_open: 0.0, h1_high: 0.0, h1_low: f64::MAX, h1_close: 0.0, h1_bars: 0,
            h4_open: 0.0, h4_high: 0.0, h4_low: f64::MAX, h4_close: 0.0, h4_bars: 0,
        }
    }

    /// Feed one 15-minute OHLC bar.
    /// Automatically aggregates into 1h and 4h bars internally.
    /// Returns [mass_15m, mass_1h, mass_4h].
    pub fn update(&mut self, open: f64, high: f64, low: f64, close: f64) -> Vec<f64> {
        // Update 15m
        self.m15.update_bar(high, low, close);

        // Accumulate for 1h (every 4 x 15m bars)
        if self.h1_bars == 0 {
            self.h1_open  = open;
            self.h1_high  = high;
            self.h1_low   = low;
            self.h1_close = close;
        } else {
            if high > self.h1_high { self.h1_high = high; }
            if low  < self.h1_low  { self.h1_low  = low; }
            self.h1_close = close;
        }
        self.h1_bars += 1;
        if self.h1_bars >= 4 {
            self.m60.update_bar(self.h1_high, self.h1_low, self.h1_close);
            self.h1_bars = 0;
        }

        // Accumulate for 4h (every 16 x 15m bars)
        if self.h4_bars == 0 {
            self.h4_open  = open;
            self.h4_high  = high;
            self.h4_low   = low;
            self.h4_close = close;
        } else {
            if high > self.h4_high { self.h4_high = high; }
            if low  < self.h4_low  { self.h4_low  = low; }
            self.h4_close = close;
        }
        self.h4_bars += 1;
        if self.h4_bars >= 16 {
            self.m240.update_bar(self.h4_high, self.h4_low, self.h4_close);
            self.h4_bars = 0;
        }

        self.bar_count += 1;
        vec![self.m15.mass, self.m60.mass, self.m240.mass]
    }

    pub fn mass_15m(&self) -> f64  { self.m15.mass }
    pub fn mass_1h(&self) -> f64   { self.m60.mass }
    pub fn mass_4h(&self) -> f64   { self.m240.mass }
    pub fn bh_active_15m(&self) -> bool { self.m15.bh_active }
    pub fn bh_active_1h(&self)  -> bool { self.m60.bh_active }
    pub fn bh_active_4h(&self)  -> bool { self.m240.bh_active }
}

// ---------------------------------------------------------------------------
// Batch computation over historical OHLCV data
// ---------------------------------------------------------------------------

/// Compute BH mass time series for an entire OHLCV history.
/// Accepts parallel arrays: opens, highs, lows, closes (all same length).
/// Returns packed array of length 3*n: [mass_15m_0..n, mass_1h_0..n, mass_4h_0..n]
/// where 1h and 4h arrays contain the most recently computed values repeated
/// (step-function update) to maintain alignment with 15m bars.
#[wasm_bindgen]
pub fn compute_bh_mass_series(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    cf_15m: f64,
    bh_form: f64,
    bh_decay: f64,
) -> Vec<f64> {
    let n = closes.len();
    if n == 0 || highs.len() != n || lows.len() != n {
        return Vec::new();
    }

    let mut state = MultiTimeframeMass::new(cf_15m, bh_form, bh_decay);
    let mut mass_15m = vec![0.0f64; n];
    let mut mass_1h  = vec![0.0f64; n];
    let mut mass_4h  = vec![0.0f64; n];

    // Track last known 1h and 4h mass for step-function fill
    let mut last_1h  = 0.0f64;
    let mut last_4h  = 0.0f64;

    for i in 0..n {
        let open = closes[i.saturating_sub(1)]; // approximate open
        let vals = state.update(open, highs[i], lows[i], closes[i]);
        mass_15m[i] = vals[0];

        // 1h and 4h update only on aggregation boundaries; otherwise repeat last
        if state.h1_bars == 0 { last_1h = vals[1]; }
        if state.h4_bars == 0 { last_4h = vals[2]; }
        mass_1h[i] = last_1h;
        mass_4h[i] = last_4h;
    }

    let mut packed = Vec::with_capacity(3 * n);
    packed.extend_from_slice(&mass_15m);
    packed.extend_from_slice(&mass_1h);
    packed.extend_from_slice(&mass_4h);
    packed
}

/// Compute binary CTL (Compressed Timelike Leg) signal for a close series.
/// Returns 1.0 for bars where ds^2 > 0 (timelike), 0.0 otherwise.
#[wasm_bindgen]
pub fn compute_ctl_signal(closes: &[f64], cf: f64) -> Vec<f64> {
    let n = closes.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let mut signal = vec![0.0f64; n];
    let log_prices: Vec<f64> = closes.iter().map(|&c| c.ln()).collect();

    for i in 1..n {
        let dx = log_prices[i] - log_prices[i - 1];
        let ds2 = cf * cf - dx * dx;
        if ds2 > 0.0 {
            signal[i] = 1.0;
        }
    }
    signal
}
