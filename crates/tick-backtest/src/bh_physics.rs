// bh_physics.rs — Rust port of BH physics from lib/srfm_core.py
//
// Black-Hole market physics: bars are modelled as spacetime intervals.
// A "timelike" interval (ds² > 0) means price moved slower than the
// speed-of-light constant cf; a "spacelike" one means it exceeded cf.
// Timelike moves add Lorentz-gamma to BH mass; spacelike moves decay it.

use std::f64::consts::PI;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// MinkowskiClassifier
// ---------------------------------------------------------------------------

/// Classifies each bar's price move as timelike or spacelike using a
/// (1+1)-D Minkowski metric:
///
///   ds² = c²·dt² − dx²
///
/// where dt = 1 (one bar), dx = close/prev_close − 1 (log-like return),
/// c = `cf` (speed-of-light constant, tunable parameter).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MinkowskiClassifier {
    /// Speed-of-light parameter (tune per instrument, e.g. 0.001).
    pub cf: f64,
    pub prev_close: f64,
}

impl MinkowskiClassifier {
    pub fn new(cf: f64) -> Self {
        Self { cf, prev_close: f64::NAN }
    }

    /// Feed the next close price.  Returns `(is_timelike, ds_squared)`.
    /// On the very first bar (no prev_close yet) returns `(false, f64::NAN)`.
    pub fn update(&mut self, close: f64) -> (bool, f64) {
        if self.prev_close.is_nan() || self.prev_close == 0.0 {
            self.prev_close = close;
            return (false, f64::NAN);
        }
        let dx = close / self.prev_close - 1.0;
        let c = self.cf;
        // dt = 1 bar
        let ds2 = c * c - dx * dx;
        let is_timelike = ds2 > 0.0;
        self.prev_close = close;
        (is_timelike, ds2)
    }

    /// Reset to uninitialised state.
    pub fn reset(&mut self) {
        self.prev_close = f64::NAN;
    }
}

// ---------------------------------------------------------------------------
// BlackHoleDetector
// ---------------------------------------------------------------------------

/// Accumulates "mass" from timelike bar moves via Lorentz γ.
/// When mass exceeds `bh_form` the BH becomes active (entry signal).
/// When mass drops below `bh_collapse` the BH dies (exit signal).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackHoleDetector {
    /// Mass formation threshold — BH activates above this.
    pub bh_form: f64,
    /// Collapse threshold — BH deactivates below this.
    pub bh_collapse: f64,
    /// Per-bar decay multiplier for spacelike moves (< 1.0).
    pub bh_decay: f64,
    /// Current accumulated mass (0.0 – 2.0).
    pub mass: f64,
    /// Whether the BH is currently active.
    pub active: bool,
    /// Speed-of-light constant (needed for γ calculation).
    cf: f64,
}

impl BlackHoleDetector {
    pub fn new(cf: f64, bh_form: f64, bh_collapse: f64, bh_decay: f64) -> Self {
        Self {
            bh_form,
            bh_collapse,
            bh_decay,
            mass: 0.0,
            active: false,
            cf,
        }
    }

    /// Update detector with latest bar classification.
    ///
    /// # Arguments
    /// * `is_timelike` — from `MinkowskiClassifier::update`
    /// * `dx` — price return (close/prev_close − 1)
    ///
    /// Returns `true` if the BH is active after this update.
    pub fn update(&mut self, is_timelike: bool, dx: f64) -> bool {
        if is_timelike {
            // Lorentz factor: γ = 1 / √(1 − (dx/c)²)
            // Guard against |dx/c| ≥ 1 (shouldn't happen for timelike, but be safe)
            let beta = (dx / self.cf).abs().min(0.9999);
            let gamma = 1.0 / (1.0 - beta * beta).sqrt();
            self.mass = (self.mass + gamma).min(2.0);
        } else {
            self.mass *= self.bh_decay;
            if self.mass < 1e-9 {
                self.mass = 0.0;
            }
        }

        // Hysteresis: activate at bh_form, deactivate below bh_collapse
        if !self.active && self.mass >= self.bh_form {
            self.active = true;
        } else if self.active && self.mass < self.bh_collapse {
            self.active = false;
        }

        self.active
    }

    pub fn reset(&mut self) {
        self.mass = 0.0;
        self.active = false;
    }
}

// ---------------------------------------------------------------------------
// HawkingTemperature
// ---------------------------------------------------------------------------

/// Hawking radiation temperature proxy: T_H = 1 / (8π·M).
/// Used as a volatility / regime instability indicator.
pub struct HawkingTemperature;

impl HawkingTemperature {
    #[inline]
    pub fn compute(mass: f64) -> f64 {
        1.0 / (8.0 * PI * mass.max(1e-9))
    }
}

// ---------------------------------------------------------------------------
// BHUpdate — snapshot returned per bar
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BHUpdate {
    pub is_timelike: bool,
    pub ds_squared: f64,
    pub mass: f64,
    pub active: bool,
    pub hawking_temp: f64,
    /// Raw price return dx = close/prev_close − 1.
    pub dx: f64,
}

// ---------------------------------------------------------------------------
// BHState — combined component
// ---------------------------------------------------------------------------

/// Convenience wrapper combining `MinkowskiClassifier`,
/// `BlackHoleDetector`, and `HawkingTemperature` into a single stateful
/// object that processes one bar at a time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BHState {
    pub classifier: MinkowskiClassifier,
    pub detector: BlackHoleDetector,
    /// Speed-of-light constant (kept here for convenience).
    pub cf: f64,
}

impl BHState {
    pub fn new(cf: f64, bh_form: f64, bh_collapse: f64, bh_decay: f64) -> Self {
        Self {
            cf,
            classifier: MinkowskiClassifier::new(cf),
            detector: BlackHoleDetector::new(cf, bh_form, bh_collapse, bh_decay),
        }
    }

    /// Process the next close price and return a full [`BHUpdate`].
    pub fn update(&mut self, close: f64) -> BHUpdate {
        let prev_close = self.classifier.prev_close;
        let (is_timelike, ds_squared) = self.classifier.update(close);

        // Compute dx from the stored prev_close (before it was overwritten)
        let dx = if prev_close.is_nan() || prev_close == 0.0 {
            0.0
        } else {
            close / prev_close - 1.0
        };

        let active = if ds_squared.is_nan() {
            // First bar — no update to detector
            false
        } else {
            self.detector.update(is_timelike, dx)
        };

        let mass = self.detector.mass;
        let hawking_temp = HawkingTemperature::compute(mass);

        BHUpdate { is_timelike, ds_squared, mass, active, hawking_temp, dx }
    }

    /// Reset all internal state.
    pub fn reset(&mut self) {
        self.classifier.reset();
        self.detector.reset();
    }

    /// Convenience: is the BH currently active?
    pub fn is_active(&self) -> bool {
        self.detector.active
    }

    /// Current mass.
    pub fn mass(&self) -> f64 {
        self.detector.mass
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minkowski_first_bar_no_output() {
        let mut mc = MinkowskiClassifier::new(0.001);
        let (is_tl, ds2) = mc.update(100.0);
        assert!(!is_tl);
        assert!(ds2.is_nan());
    }

    #[test]
    fn minkowski_small_move_is_timelike() {
        let mut mc = MinkowskiClassifier::new(0.001);
        mc.update(100.0);
        // tiny move: dx = 0.0001 < c = 0.001
        let (is_tl, ds2) = mc.update(100.01);
        assert!(is_tl, "small move should be timelike");
        assert!(ds2 > 0.0);
    }

    #[test]
    fn minkowski_large_move_is_spacelike() {
        let mut mc = MinkowskiClassifier::new(0.001);
        mc.update(100.0);
        // large move: dx = 0.05 >> c = 0.001
        let (is_tl, ds2) = mc.update(105.0);
        assert!(!is_tl, "large move should be spacelike");
        assert!(ds2 < 0.0);
    }

    #[test]
    fn bh_activates_after_enough_timelike_bars() {
        let mut state = BHState::new(0.001, 1.5, 1.2, 0.95);
        let mut price = 100.0;
        let mut activated = false;
        for _ in 0..50 {
            // tiny upward ticks — all timelike
            price *= 1.00005;
            let upd = state.update(price);
            if upd.active {
                activated = true;
                break;
            }
        }
        assert!(activated, "BH should activate after enough timelike bars");
    }

    #[test]
    fn hawking_temp_inverse_mass() {
        let t1 = HawkingTemperature::compute(1.0);
        let t2 = HawkingTemperature::compute(2.0);
        assert!(t1 > t2, "higher mass → lower Hawking temp");
    }
}
