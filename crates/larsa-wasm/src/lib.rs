//! larsa-wasm -- WebAssembly analytics module for the SRFM dashboard.
//!
//! Exports rolling technical indicators, BH mass computation, regime classification,
//! and equity curve analytics to JavaScript via wasm-bindgen.
//!
//! All heavy computation happens client-side in the browser; no API round-trips needed
//! for indicator rendering once the WASM module is loaded.

use wasm_bindgen::prelude::*;

pub mod indicators;
pub mod bh_mass;
pub mod portfolio;
pub mod signal_visualizer;
pub mod realtime_indicators;
pub mod portfolio_analytics_wasm;

// Re-export key types at crate root so JS sees them under the module namespace.
pub use indicators::ema::{EmaState, compute_ema, compute_double_ema, compute_macd};
pub use indicators::rsi::compute_rsi;
pub use indicators::bollinger::compute_bollinger;
pub use indicators::atr::compute_atr;
pub use bh_mass::minkowski::{BhMassState, compute_bh_mass_series, MultiTimeframeMass};
pub use bh_mass::regime::{RegimeClassifier, classify_regimes};
pub use portfolio::equity_curve::{EquityAnalytics, compute_equity_analytics};

// ---------------------------------------------------------------------------
// Original SRFMState -- preserved from v0.1, extended with new features
// ---------------------------------------------------------------------------

/// Per-tick SRFM state machine. Computes BH mass and CTL formation in real time.
#[wasm_bindgen]
pub struct SRFMState {
    pub bh_mass: f64,
    pub bh_active: bool,
    pub ctl: i32,
    pub beta: f64,
    pub equity: f64,
    cf: f64,
    bh_form: f64,
    bh_decay: f64,
    prev_close: f64,
    mass: f64,
    ctl_count: i32,
    equity_val: f64,
}

#[wasm_bindgen]
impl SRFMState {
    #[wasm_bindgen(constructor)]
    pub fn new(cf: f64, bh_form: f64, bh_decay: f64, initial_close: f64) -> SRFMState {
        SRFMState {
            cf,
            bh_form,
            bh_decay,
            prev_close: initial_close,
            mass: 0.0,
            ctl_count: 0,
            bh_mass: 0.0,
            bh_active: false,
            ctl: 0,
            beta: 0.0,
            equity: 1.0,
            equity_val: 1.0,
        }
    }

    /// Process one price bar. Returns position signal in [0, 1].
    pub fn step(&mut self, close: f64) -> f64 {
        let beta = (close - self.prev_close).abs() / (self.prev_close * self.cf + 1e-12);
        self.beta = beta;

        if beta < 1.0 {
            self.mass = self.mass * 0.97 + 0.03;
            self.ctl_count += 1;
        } else {
            self.mass *= self.bh_decay;
            self.ctl_count = 0;
        }

        self.bh_mass = self.mass;
        self.bh_active = self.mass >= self.bh_form && self.ctl_count >= 5;
        self.ctl = self.ctl_count;
        self.prev_close = close;

        if self.bh_active {
            0.65
        } else if self.ctl_count >= 3 {
            0.325
        } else {
            0.0
        }
    }

    pub fn reset(&mut self) {
        self.mass = 0.0;
        self.ctl_count = 0;
        self.bh_active = false;
        self.equity_val = 1.0;
        self.equity = 1.0;
    }
}

/// Batch simulate an entire price series. Returns equity curve as Float64Array.
#[wasm_bindgen]
pub fn simulate_series(closes: &[f64], cf: f64, bh_form: f64) -> Vec<f64> {
    if closes.is_empty() {
        return Vec::new();
    }
    let mut state = SRFMState::new(cf, bh_form, 0.95, closes[0]);
    let mut equity = vec![1.0f64; closes.len()];
    for i in 1..closes.len() {
        let pos = state.step(closes[i]);
        let ret = closes[i] / closes[i - 1] - 1.0;
        equity[i] = equity[i - 1] * (1.0 + pos * ret);
    }
    equity
}

// ---------------------------------------------------------------------------
// Module version query
// ---------------------------------------------------------------------------

/// Returns the module version string.
#[wasm_bindgen]
pub fn larsa_wasm_version() -> String {
    "0.2.0".to_string()
}
