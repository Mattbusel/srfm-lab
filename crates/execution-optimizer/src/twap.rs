/// TWAP slicer — time-weighted average price execution.
///
/// Divides total order quantity evenly across N time intervals.
/// Supports participation-rate constraints and intraday scheduling.

use serde::Serialize;
use crate::schedule::{ExecutionSchedule, SliceOrder};

/// TWAP execution slicer.
#[derive(Debug, Clone)]
pub struct TwapSlicer {
    /// Total quantity to execute (positive = buy, negative = sell)
    pub total_qty:    f64,
    /// Number of time slices
    pub n_slices:     usize,
    /// Duration of each slice (seconds)
    pub slice_secs:   f64,
    /// Maximum participation rate [0,1] (fraction of ADV per slice)
    pub max_part:     f64,
    /// Estimated ADV for participation check
    pub adv:          f64,
}

impl TwapSlicer {
    pub fn new(total_qty: f64, n_slices: usize, total_secs: f64) -> Self {
        Self {
            total_qty,
            n_slices,
            slice_secs: total_secs / n_slices as f64,
            max_part: 1.0,
            adv: f64::INFINITY,
        }
    }

    pub fn with_participation(mut self, max_part: f64, adv: f64) -> Self {
        self.max_part = max_part.clamp(0.0, 1.0);
        self.adv = adv;
        self
    }

    /// Build the execution schedule.
    pub fn build(&self) -> ExecutionSchedule {
        let base_qty = self.total_qty / self.n_slices as f64;
        // Participation cap
        let adv_per_slice = self.adv * (self.slice_secs / 23_400.0); // 6.5h day
        let cap = self.max_part * adv_per_slice;
        let capped_qty = if base_qty.abs() > cap { cap * base_qty.signum() } else { base_qty };

        let slices: Vec<SliceOrder> = (0..self.n_slices).map(|i| {
            SliceOrder {
                start_secs:  i as f64 * self.slice_secs,
                end_secs:    (i + 1) as f64 * self.slice_secs,
                qty:         capped_qty,
                limit_price: None,
            }
        }).collect();

        let total_scheduled: f64 = slices.iter().map(|s| s.qty).sum();
        ExecutionSchedule { slices, total_qty: self.total_qty, total_scheduled }
    }

    /// Number of slices needed to execute full quantity under participation cap.
    pub fn slices_needed(&self) -> usize {
        let adv_per_slice = self.adv * (self.slice_secs / 23_400.0);
        let cap = self.max_part * adv_per_slice;
        if cap <= 0.0 { return usize::MAX; }
        (self.total_qty.abs() / cap).ceil() as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twap_slices_sum_to_total() {
        let slicer = TwapSlicer::new(1_000.0, 10, 3600.0);
        let sched  = slicer.build();
        let total: f64 = sched.slices.iter().map(|s| s.qty).sum();
        assert!((total - 1_000.0).abs() < 1e-6);
    }

    #[test]
    fn twap_uniform_slices() {
        let slicer = TwapSlicer::new(500.0, 5, 1800.0);
        let sched  = slicer.build();
        for s in &sched.slices {
            assert!((s.qty - 100.0).abs() < 1e-6);
        }
    }

    #[test]
    fn twap_participation_cap() {
        // ADV = 10_000, 10 slices over 1 hour (10 intervals × 360s)
        // ADV per slice ≈ 10_000 * 360/23400 ≈ 153.8
        // max_part = 0.5 → cap ≈ 76.9 per slice
        let slicer = TwapSlicer::new(5_000.0, 10, 3600.0)
            .with_participation(0.5, 10_000.0);
        let sched = slicer.build();
        for s in &sched.slices {
            assert!(s.qty.abs() <= 77.0 + 0.1, "qty {} exceeded cap", s.qty);
        }
    }
}
