//! regime.rs -- Market regime classification from BH mass and price action.
//!
//! Regimes follow the Wyckoff-inspired four-phase model:
//!   ACCUMULATION  -- sideways/flat, building potential, BH mass rising
//!   MARKUP        -- trending up, BH active, price expanding
//!   DISTRIBUTION  -- sideways/flat after markup, BH mass falling
//!   MARKDOWN      -- trending down, high volatility, mass collapsed
//!
//! Classification uses BH mass level, trend direction (slope of EMA), and
//! volatility (rolling ATR/close ratio) to assign each bar to a regime.

use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Regime enum
// ---------------------------------------------------------------------------

/// Market regime label.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime {
    Accumulation = 0,
    Markup       = 1,
    Distribution = 2,
    Markdown     = 3,
    Undefined    = 4,
}

impl Regime {
    fn to_f64(self) -> f64 {
        self as i32 as f64
    }
}

// ---------------------------------------------------------------------------
// RegimeClassifier -- stateful streaming classifier
// ---------------------------------------------------------------------------

/// Stateful regime classifier. Maintains internal EMA and ATR estimates.
#[wasm_bindgen]
pub struct RegimeClassifier {
    /// BH mass formation threshold.
    bh_form: f64,
    /// BH mass decay threshold (below this, consider BH dissolved).
    bh_dissolve: f64,
    /// EMA period for trend direction.
    ema_period: usize,
    /// ATR period for volatility estimation.
    atr_period: usize,
    /// Current EMA value.
    ema: f64,
    /// Current ATR value.
    atr: f64,
    /// Previous close.
    prev_close: f64,
    /// Previous high.
    prev_high: f64,
    /// Previous low.
    prev_low: f64,
    /// EMA alpha.
    ema_alpha: f64,
    /// ATR Wilder alpha.
    atr_alpha: f64,
    /// Current regime.
    regime: Regime,
    /// Count of consecutive bars in current regime (for hysteresis).
    regime_bars: u32,
    initialized: bool,
    warmup_count: usize,
}

#[wasm_bindgen]
impl RegimeClassifier {
    /// Create a new regime classifier.
    /// - bh_form: BH mass threshold for "active" state (e.g. 0.65)
    /// - ema_period: trend EMA period (e.g. 20)
    /// - atr_period: ATR period (e.g. 14)
    #[wasm_bindgen(constructor)]
    pub fn new(bh_form: f64, ema_period: u32, atr_period: u32) -> RegimeClassifier {
        let ep = ema_period as usize;
        let ap = atr_period as usize;
        RegimeClassifier {
            bh_form,
            bh_dissolve: bh_form * 0.5,
            ema_period: ep,
            atr_period: ap,
            ema: 0.0,
            atr: 0.0,
            prev_close: 0.0,
            prev_high: 0.0,
            prev_low: 0.0,
            ema_alpha: 2.0 / (ep as f64 + 1.0),
            atr_alpha: 1.0 / ap as f64,
            regime: Regime::Undefined,
            regime_bars: 0,
            initialized: false,
            warmup_count: 0,
        }
    }

    /// Feed one bar. Returns regime label as f64 (cast Regime enum).
    /// 0=Accumulation, 1=Markup, 2=Distribution, 3=Markdown, 4=Undefined
    pub fn update(&mut self, high: f64, low: f64, close: f64, bh_mass: f64) -> f64 {
        if !self.initialized {
            self.ema = close;
            self.atr = high - low;
            self.prev_close = close;
            self.prev_high = high;
            self.prev_low = low;
            self.initialized = true;
            self.warmup_count += 1;
            return Regime::Undefined.to_f64();
        }

        // Update EMA
        self.ema = self.ema_alpha * close + (1.0 - self.ema_alpha) * self.ema;

        // Update ATR (Wilder's method)
        let hl   = high - low;
        let hpc  = (high - self.prev_close).abs();
        let lpc  = (low  - self.prev_close).abs();
        let tr   = hl.max(hpc).max(lpc);
        self.atr = self.atr * (1.0 - self.atr_alpha) + tr * self.atr_alpha;

        self.prev_close = close;
        self.prev_high  = high;
        self.prev_low   = low;
        self.warmup_count += 1;

        // Require enough warmup before classifying
        let min_warmup = self.ema_period.max(self.atr_period);
        if self.warmup_count < min_warmup {
            return Regime::Undefined.to_f64();
        }

        // Trend direction: close vs EMA
        let above_ema  = close > self.ema;
        let atr_ratio  = if close.abs() > 1e-12 { self.atr / close } else { 0.0 };
        let high_vol   = atr_ratio > 0.015; // >1.5% ATR/price = elevated vol

        let new_regime = classify(bh_mass, self.bh_form, self.bh_dissolve, above_ema, high_vol);

        // Hysteresis: require 2 consecutive bars before switching regime
        if new_regime == self.regime {
            self.regime_bars += 1;
        } else if self.regime_bars == 0 || new_regime != self.regime {
            // Allow immediate switch if current regime has been held for < 2 bars
            if self.regime_bars < 2 {
                self.regime = new_regime;
                self.regime_bars = 1;
            } else {
                // Pending switch: decrement hysteresis counter
                self.regime_bars = self.regime_bars.saturating_sub(1);
                if self.regime_bars == 0 {
                    self.regime = new_regime;
                    self.regime_bars = 1;
                }
            }
        }

        self.regime.to_f64()
    }

    pub fn current_regime(&self) -> f64 {
        self.regime.to_f64()
    }

    pub fn regime_name(&self) -> String {
        match self.regime {
            Regime::Accumulation => "ACCUMULATION".to_string(),
            Regime::Markup       => "MARKUP".to_string(),
            Regime::Distribution => "DISTRIBUTION".to_string(),
            Regime::Markdown     => "MARKDOWN".to_string(),
            Regime::Undefined    => "UNDEFINED".to_string(),
        }
    }

    pub fn reset(&mut self) {
        self.ema = 0.0;
        self.atr = 0.0;
        self.regime = Regime::Undefined;
        self.regime_bars = 0;
        self.initialized = false;
        self.warmup_count = 0;
    }
}

// ---------------------------------------------------------------------------
// Classification logic
// ---------------------------------------------------------------------------

fn classify(
    bh_mass: f64,
    bh_form: f64,
    bh_dissolve: f64,
    above_ema: bool,
    high_vol: bool,
) -> Regime {
    match (bh_mass >= bh_form, bh_mass > bh_dissolve, above_ema, high_vol) {
        // BH active, price above trend, low vol: compression markup in progress
        (true, _, true, false) => Regime::Markup,
        // BH active, price below trend, low vol: accumulation
        (true, _, false, false) => Regime::Accumulation,
        // BH dissolving, price above trend, high vol: distribution phase
        (false, true, true, true) => Regime::Distribution,
        // BH dissolved or dissolving, price below trend, high vol: markdown
        (false, _, false, true) => Regime::Markdown,
        // BH dissolved, price above trend, low vol: late markup / consolidation
        (false, false, true, false) => Regime::Markup,
        // BH dissolving, above EMA: still in distribution
        (false, true, true, false) => Regime::Distribution,
        // Default fallback
        _ => Regime::Accumulation,
    }
}

// ---------------------------------------------------------------------------
// Batch regime classification
// ---------------------------------------------------------------------------

/// Classify regimes for an entire historical series.
/// bh_masses must be pre-computed (e.g. from compute_bh_mass_series).
/// Returns Float64Array of regime codes (0-4) of length n.
#[wasm_bindgen]
pub fn classify_regimes(
    highs: &[f64],
    lows: &[f64],
    closes: &[f64],
    bh_masses: &[f64],
    bh_form: f64,
    ema_period: u32,
    atr_period: u32,
) -> Vec<f64> {
    let n = closes.len();
    if n == 0 || highs.len() != n || lows.len() != n || bh_masses.len() != n {
        return Vec::new();
    }

    let mut classifier = RegimeClassifier::new(bh_form, ema_period, atr_period);
    let mut regimes = Vec::with_capacity(n);

    for i in 0..n {
        let r = classifier.update(highs[i], lows[i], closes[i], bh_masses[i]);
        regimes.push(r);
    }
    regimes
}

/// Returns regime label strings for an array of regime codes.
/// Output is a newline-separated string of N regime names (for simple JS parsing).
#[wasm_bindgen]
pub fn regime_codes_to_names(codes: &[f64]) -> String {
    codes.iter().map(|&c| {
        match c as u32 {
            0 => "ACCUMULATION",
            1 => "MARKUP",
            2 => "DISTRIBUTION",
            3 => "MARKDOWN",
            _ => "UNDEFINED",
        }
    }).collect::<Vec<_>>().join("\n")
}

/// Compute regime duration histogram.
/// Returns packed array [accum_count, markup_count, distrib_count, markdown_count].
#[wasm_bindgen]
pub fn regime_duration_histogram(codes: &[f64]) -> Vec<f64> {
    let mut counts = [0u64; 4];
    for &c in codes {
        let idx = c as usize;
        if idx < 4 {
            counts[idx] += 1;
        }
    }
    counts.iter().map(|&c| c as f64).collect()
}
