/// Unified execution schedule types shared by all schedulers.

use serde::Serialize;

/// A single execution slice (child order).
#[derive(Debug, Clone, Serialize)]
pub struct SliceOrder {
    /// Slice start time (seconds from execution start)
    pub start_secs:  f64,
    /// Slice end time (seconds from execution start)
    pub end_secs:    f64,
    /// Target quantity to trade (positive = buy, negative = sell)
    pub qty:         f64,
    /// Optional limit price for this slice
    pub limit_price: Option<f64>,
}

impl SliceOrder {
    /// Duration of this slice in seconds.
    pub fn duration_secs(&self) -> f64 { (self.end_secs - self.start_secs).max(0.0) }

    /// True if this slice's window includes `t` (seconds from start).
    pub fn is_active_at(&self, t: f64) -> bool {
        t >= self.start_secs && t < self.end_secs
    }
}

/// Complete execution schedule.
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionSchedule {
    pub slices:           Vec<SliceOrder>,
    /// Original total quantity requested
    pub total_qty:        f64,
    /// Sum of scheduled slice quantities (may differ if participation-capped)
    pub total_scheduled:  f64,
}

impl ExecutionSchedule {
    /// Fraction of order scheduled (1.0 = fully scheduled).
    pub fn fill_fraction(&self) -> f64 {
        if self.total_qty.abs() < 1e-15 { return 1.0; }
        self.total_scheduled / self.total_qty
    }

    /// Find the active slice for time `t`.
    pub fn active_slice(&self, t: f64) -> Option<&SliceOrder> {
        self.slices.iter().find(|s| s.is_active_at(t))
    }

    /// Cumulative scheduled qty up to (but not including) slice `i`.
    pub fn cumulative_qty(&self, up_to: usize) -> f64 {
        self.slices[..up_to.min(self.slices.len())].iter().map(|s| s.qty).sum()
    }

    /// Total duration of the schedule in seconds.
    pub fn total_duration_secs(&self) -> f64 {
        self.slices.last().map(|s| s.end_secs).unwrap_or(0.0)
    }

    /// Execution summary statistics.
    pub fn summary(&self) -> ScheduleSummary {
        let n = self.slices.len();
        let qtys: Vec<f64> = self.slices.iter().map(|s| s.qty).collect();
        let max_slice = qtys.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_slice = qtys.iter().cloned().fold(f64::INFINITY, f64::min);
        let mean_slice = if n > 0 { self.total_scheduled / n as f64 } else { 0.0 };

        ScheduleSummary {
            n_slices:        n,
            total_qty:       self.total_qty,
            total_scheduled: self.total_scheduled,
            fill_fraction:   self.fill_fraction(),
            max_slice_qty:   max_slice,
            min_slice_qty:   min_slice,
            mean_slice_qty:  mean_slice,
            duration_secs:   self.total_duration_secs(),
        }
    }
}

/// Summary statistics for a schedule.
#[derive(Debug, Clone, Serialize)]
pub struct ScheduleSummary {
    pub n_slices:        usize,
    pub total_qty:       f64,
    pub total_scheduled: f64,
    pub fill_fraction:   f64,
    pub max_slice_qty:   f64,
    pub min_slice_qty:   f64,
    pub mean_slice_qty:  f64,
    pub duration_secs:   f64,
}

/// Implementation shortfall tracker.
/// Compares actual fills against the decision price (price at order arrival).
#[derive(Debug, Clone, Default)]
pub struct ImplementationShortfall {
    decision_price: Option<f64>,
    fills:          Vec<(f64, f64)>,  // (price, qty)
}

impl ImplementationShortfall {
    pub fn new() -> Self { Self::default() }

    pub fn set_decision_price(&mut self, price: f64) {
        self.decision_price = Some(price);
    }

    pub fn record_fill(&mut self, fill_price: f64, fill_qty: f64) {
        self.fills.push((fill_price, fill_qty));
    }

    /// Total filled quantity.
    pub fn total_filled(&self) -> f64 {
        self.fills.iter().map(|(_, q)| q.abs()).sum()
    }

    /// Volume-weighted average fill price.
    pub fn vwap_fill(&self) -> Option<f64> {
        let total_vol = self.total_filled();
        if total_vol < 1e-15 { return None; }
        let sum_pv: f64 = self.fills.iter().map(|(p, q)| p * q.abs()).sum();
        Some(sum_pv / total_vol)
    }

    /// Implementation shortfall in basis points.
    /// IS = (vwap_fill - decision_price) / decision_price × 10_000
    /// (positive = bought above decision price, i.e. slippage)
    pub fn shortfall_bps(&self) -> Option<f64> {
        let dp   = self.decision_price?;
        let vwap = self.vwap_fill()?;
        if dp.abs() < 1e-15 { return None; }
        Some((vwap - dp) / dp * 10_000.0)
    }

    /// Slippage vs. arrival mid.
    pub fn slippage_bps(&self, arrival_mid: f64) -> Option<f64> {
        let vwap = self.vwap_fill()?;
        if arrival_mid.abs() < 1e-15 { return None; }
        Some((vwap - arrival_mid) / arrival_mid * 10_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_fill_fraction_full() {
        let slices = vec![
            SliceOrder { start_secs: 0.0, end_secs: 60.0, qty: 100.0, limit_price: None },
            SliceOrder { start_secs: 60.0, end_secs: 120.0, qty: 100.0, limit_price: None },
        ];
        let sched = ExecutionSchedule { slices, total_qty: 200.0, total_scheduled: 200.0 };
        assert!((sched.fill_fraction() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn is_tracker_shortfall() {
        let mut is = ImplementationShortfall::new();
        is.set_decision_price(100.0);
        is.record_fill(100.5, 500.0);
        is.record_fill(100.3, 500.0);
        let bps = is.shortfall_bps().unwrap();
        // VWAP fill = 100.4, IS = 0.4/100 * 10000 = 40 bps
        assert!((bps - 40.0).abs() < 0.5);
    }

    #[test]
    fn slice_active_at() {
        let s = SliceOrder { start_secs: 100.0, end_secs: 200.0, qty: 50.0, limit_price: None };
        assert!( s.is_active_at(150.0));
        assert!(!s.is_active_at(99.9));
        assert!(!s.is_active_at(200.0));
    }
}
