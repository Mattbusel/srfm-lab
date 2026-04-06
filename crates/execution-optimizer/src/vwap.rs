/// VWAP slicer — volume-weighted average price execution.
///
/// Distributes order quantity proportionally to an intraday volume profile
/// so that participation rate is roughly constant through the day.

use serde::Serialize;
use crate::schedule::{ExecutionSchedule, SliceOrder};

/// A typical intraday U-shaped volume profile (30-minute buckets, 13 buckets for 6.5h day).
/// Values are relative fractions (sum = 1.0).
pub const DEFAULT_VOLUME_PROFILE: [f64; 13] = [
    0.120, // 09:30–10:00  (heavy open)
    0.090, // 10:00–10:30
    0.075, // 10:30–11:00
    0.065, // 11:00–11:30
    0.060, // 11:30–12:00
    0.055, // 12:00–12:30
    0.055, // 12:30–13:00
    0.060, // 13:00–13:30
    0.065, // 13:30–14:00
    0.070, // 14:00–14:30
    0.075, // 14:30–15:00
    0.090, // 15:00–15:30
    0.120, // 15:30–16:00  (heavy close)
];

/// VWAP execution slicer.
#[derive(Debug, Clone)]
pub struct VwapSlicer {
    pub total_qty:    f64,
    pub volume_profile: Vec<f64>,   // fractions summing to 1.0
    pub slice_secs:   f64,
    pub max_part:     f64,
    pub adv:          f64,
}

impl VwapSlicer {
    /// Create with standard U-shaped profile.
    pub fn new(total_qty: f64) -> Self {
        let profile: Vec<f64> = DEFAULT_VOLUME_PROFILE.to_vec();
        let n = profile.len();
        Self {
            total_qty,
            volume_profile: profile,
            slice_secs: 1800.0,   // 30 min buckets
            max_part: 1.0,
            adv: f64::INFINITY,
        }
    }

    /// Create with a custom volume profile.
    pub fn with_profile(total_qty: f64, profile: Vec<f64>, slice_secs: f64) -> Self {
        let sum: f64 = profile.iter().sum();
        assert!(sum > 0.0, "volume profile must sum to positive value");
        let normalised: Vec<f64> = profile.iter().map(|v| v / sum).collect();
        Self {
            total_qty,
            volume_profile: normalised,
            slice_secs,
            max_part: 1.0,
            adv: f64::INFINITY,
        }
    }

    pub fn with_participation(mut self, max_part: f64, adv: f64) -> Self {
        self.max_part = max_part.clamp(0.0, 1.0);
        self.adv      = adv;
        self
    }

    /// Build the execution schedule.
    pub fn build(&self) -> ExecutionSchedule {
        let n = self.volume_profile.len();
        let adv_per_slice = self.adv * (self.slice_secs / 23_400.0);
        let cap = self.max_part * adv_per_slice;

        let slices: Vec<SliceOrder> = (0..n).map(|i| {
            let target = self.total_qty * self.volume_profile[i];
            let qty = if target.abs() > cap { cap * target.signum() } else { target };
            SliceOrder {
                start_secs:  i as f64 * self.slice_secs,
                end_secs:    (i + 1) as f64 * self.slice_secs,
                qty,
                limit_price: None,
            }
        }).collect();

        let total_scheduled: f64 = slices.iter().map(|s| s.qty).sum();
        ExecutionSchedule { slices, total_qty: self.total_qty, total_scheduled }
    }

    /// Execution shortfall vs. flat VWAP (0 = perfect VWAP tracking).
    pub fn tracking_error(&self, executed_qtys: &[f64]) -> f64 {
        let n = self.volume_profile.len().min(executed_qtys.len());
        let mut err = 0.0;
        for i in 0..n {
            let target = self.total_qty * self.volume_profile[i];
            err += (executed_qtys[i] - target).powi(2);
        }
        (err / n as f64).sqrt()
    }
}

/// Adaptive VWAP: updates the volume profile in real time using exponential smoothing.
#[derive(Debug, Clone)]
pub struct AdaptiveVwap {
    profile:  Vec<f64>,
    alpha:    f64,
    n:        usize,
}

impl AdaptiveVwap {
    pub fn new(initial_profile: Vec<f64>, alpha: f64) -> Self {
        let n = initial_profile.len();
        Self { profile: initial_profile, alpha, n }
    }

    /// Update slice `i`'s observed fraction.
    pub fn update_slice(&mut self, slice_idx: usize, observed_fraction: f64) {
        if slice_idx >= self.n { return; }
        self.profile[slice_idx] = (1.0 - self.alpha) * self.profile[slice_idx]
                                  + self.alpha * observed_fraction;
        // Re-normalise
        let sum: f64 = self.profile.iter().sum();
        if sum > 0.0 { self.profile.iter_mut().for_each(|v| *v /= sum); }
    }

    pub fn profile(&self) -> &[f64] { &self.profile }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vwap_profile_sums_to_total() {
        let slicer = VwapSlicer::new(10_000.0);
        let sched  = slicer.build();
        let total: f64 = sched.slices.iter().map(|s| s.qty).sum();
        assert!((total - 10_000.0).abs() < 1e-6, "total={}", total);
    }

    #[test]
    fn vwap_open_close_heavy() {
        let slicer = VwapSlicer::new(1_000.0);
        let sched  = slicer.build();
        // First and last slices should be heavier than middle
        let first  = sched.slices[0].qty;
        let middle = sched.slices[6].qty;
        assert!(first > middle, "open slice {} should exceed midday {}", first, middle);
    }

    #[test]
    fn adaptive_vwap_normalises() {
        let profile = DEFAULT_VOLUME_PROFILE.to_vec();
        let mut avwap = AdaptiveVwap::new(profile, 0.1);
        avwap.update_slice(0, 0.20);  // observed heavier open
        let sum: f64 = avwap.profile().iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "profile sum = {}", sum);
    }
}
