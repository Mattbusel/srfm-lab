// participation_rate.rs
// Participation rate controller for VWAP/TWAP/PoV execution.
// Tracks rolling market volume, computes adaptive participation rates,
// manages slippage budgets, and sizes order slices.

use std::collections::VecDeque;

// ---- VolumeTracker --------------------------------------------------------

/// A rolling window of (timestamp_secs, volume) observations.
/// Supports computing the total volume seen in a recent window.
#[derive(Debug, Clone)]
pub struct VolumeTracker {
    /// Circular buffer: (timestamp_secs, volume)
    window: VecDeque<(u64, f64)>,
    /// Maximum age of observations to retain (seconds)
    window_secs: u64,
    /// Running sum of volumes in the window
    running_total: f64,
}

impl VolumeTracker {
    pub fn new(window_secs: u64) -> Self {
        VolumeTracker {
            window: VecDeque::new(),
            window_secs,
            running_total: 0.0,
        }
    }

    /// Add a new (timestamp, volume) observation and evict stale entries.
    pub fn push(&mut self, timestamp: u64, volume: f64) {
        self.window.push_back((timestamp, volume));
        self.running_total += volume;
        self.evict_before(timestamp.saturating_sub(self.window_secs));
    }

    /// Remove entries older than `cutoff`.
    fn evict_before(&mut self, cutoff: u64) {
        while let Some(&(ts, vol)) = self.window.front() {
            if ts < cutoff {
                self.running_total -= vol;
                // Guard against floating-point drift going negative
                if self.running_total < 0.0 {
                    self.running_total = 0.0;
                }
                self.window.pop_front();
            } else {
                break;
            }
        }
    }

    /// Total volume observed since `since_secs` (inclusive).
    /// O(n) scan -- kept simple because windows are typically small.
    pub fn compute_rate(&self, since: u64) -> f64 {
        self.window.iter()
            .filter(|(ts, _)| *ts >= since)
            .map(|(_, v)| v)
            .sum()
    }

    /// Total volume currently in the window.
    pub fn total_in_window(&self) -> f64 {
        self.running_total.max(0.0)
    }

    /// Number of observations currently retained.
    pub fn len(&self) -> usize {
        self.window.len()
    }

    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    /// Average volume per second over the window (0 if no data).
    pub fn avg_rate_per_sec(&self) -> f64 {
        if self.window_secs == 0 {
            return 0.0;
        }
        self.running_total / self.window_secs as f64
    }
}

// ---- SlippageBudget -------------------------------------------------------

/// Tracks a slippage allowance expressed in basis points.
#[derive(Debug, Clone)]
pub struct SlippageBudget {
    pub total_bps: f64,
    pub spent_bps: f64,
}

impl SlippageBudget {
    pub fn new(total_bps: f64) -> Self {
        assert!(total_bps >= 0.0, "Slippage budget must be non-negative");
        SlippageBudget { total_bps, spent_bps: 0.0 }
    }

    /// Remaining budget in basis points.
    pub fn remaining(&self) -> f64 {
        (self.total_bps - self.spent_bps).max(0.0)
    }

    /// True when the full budget has been consumed.
    pub fn is_exhausted(&self) -> bool {
        self.spent_bps >= self.total_bps
    }

    /// Record actual slippage from a fill (in bps, can be negative if price improved).
    pub fn record_fill(&mut self, slippage_bps: f64) {
        self.spent_bps += slippage_bps;
    }

    /// Fraction of budget consumed (0 .. >1.0 if over-budget).
    pub fn utilization(&self) -> f64 {
        if self.total_bps <= 0.0 { return 1.0; }
        self.spent_bps / self.total_bps
    }

    /// Maximum notional cost (bps * qty * price / 10_000) that can still be spent.
    pub fn max_cost_notional(&self, qty: f64, ref_price: f64) -> f64 {
        self.remaining() * qty * ref_price / 10_000.0
    }
}

// ---- ParticipationRateController ------------------------------------------

/// Controls how aggressively we participate in market volume.
/// The controller sizes each order slice as `target_rate * market_vol_in_period`.
#[derive(Debug, Clone)]
pub struct ParticipationRateController {
    /// Desired fraction of market volume to consume per window (e.g. 0.10 = 10 %)
    pub target_rate: f64,
    /// Observation window in seconds
    pub window_secs: u64,
    /// Rolling volume tracker
    pub volume_tracker: VolumeTracker,
    /// Cumulative quantity sent so far
    pub qty_sent: f64,
    /// Total quantity we need to execute
    pub total_qty: f64,
}

impl ParticipationRateController {
    pub fn new(target_rate: f64, window_secs: u64, total_qty: f64) -> Self {
        assert!(
            target_rate > 0.0 && target_rate <= 1.0,
            "target_rate must be in (0, 1]"
        );
        ParticipationRateController {
            target_rate: target_rate.min(1.0).max(0.0),
            window_secs,
            volume_tracker: VolumeTracker::new(window_secs),
            qty_sent: 0.0,
            total_qty,
        }
    }

    /// Feed a new market volume observation.
    pub fn observe_volume(&mut self, timestamp: u64, market_volume: f64) {
        self.volume_tracker.push(timestamp, market_volume);
    }

    /// Compute the recommended order size given current market volume in the
    /// window and our remaining inventory.
    ///
    /// market_vol -- total market volume in the current participation period
    /// inventory  -- remaining quantity we still need to execute
    pub fn compute_order_size(&self, market_vol: f64, inventory: f64) -> f64 {
        let raw = market_vol * self.target_rate;
        // Never exceed remaining inventory
        raw.min(inventory).max(0.0)
    }

    /// Record that a slice of `qty` was sent.
    pub fn record_sent(&mut self, qty: f64) {
        self.qty_sent += qty;
    }

    /// Remaining quantity to execute.
    pub fn remaining_qty(&self) -> f64 {
        (self.total_qty - self.qty_sent).max(0.0)
    }

    /// True when the full order has been sent.
    pub fn is_complete(&self) -> bool {
        self.qty_sent >= self.total_qty - 1e-9
    }

    /// Compute an order size using the rolling window volume.
    pub fn compute_order_size_from_window(&self, now: u64) -> f64 {
        let since = now.saturating_sub(self.window_secs);
        let window_vol = self.volume_tracker.compute_rate(since);
        self.compute_order_size(window_vol, self.remaining_qty())
    }

    /// Participation fraction actually achieved so far vs. cumulative market volume.
    pub fn actual_participation(&self) -> f64 {
        let total_mkt = self.volume_tracker.total_in_window();
        if total_mkt < 1e-9 { return 0.0; }
        self.qty_sent / total_mkt
    }
}

// ---- AdaptiveParticipationRate --------------------------------------------

/// Adapts the participation rate upward as deadline approaches.
/// urgency_score = (elapsed / total_duration)^1.5
/// adjusted_rate = base_rate * (1 + 0.5 * urgency_score)
#[derive(Debug, Clone)]
pub struct AdaptiveParticipationRate {
    /// Baseline participation rate (no urgency)
    pub base_rate: f64,
    /// Maximum allowed rate regardless of urgency
    pub max_rate: f64,
    /// Total execution duration in seconds
    pub total_duration_secs: f64,
    /// Elapsed seconds at last update
    elapsed_secs: f64,
}

impl AdaptiveParticipationRate {
    pub fn new(base_rate: f64, total_duration_secs: f64) -> Self {
        AdaptiveParticipationRate {
            base_rate,
            max_rate: (base_rate * 2.5).min(0.50), // never exceed 50%
            total_duration_secs,
            elapsed_secs: 0.0,
        }
    }

    pub fn with_max_rate(mut self, max_rate: f64) -> Self {
        self.max_rate = max_rate.min(1.0).max(self.base_rate);
        self
    }

    /// Update elapsed time and return the new adjusted rate.
    pub fn update(&mut self, elapsed_secs: f64) -> f64 {
        self.elapsed_secs = elapsed_secs.max(0.0).min(self.total_duration_secs);
        self.current_rate()
    }

    /// Current adjusted rate based on elapsed time.
    pub fn current_rate(&self) -> f64 {
        let urgency = self.urgency_score();
        let rate = self.base_rate * (1.0 + 0.5 * urgency);
        rate.min(self.max_rate).max(0.0)
    }

    /// urgency_score = (elapsed / total_duration)^1.5
    pub fn urgency_score(&self) -> f64 {
        if self.total_duration_secs <= 0.0 { return 1.0; }
        let frac = (self.elapsed_secs / self.total_duration_secs).min(1.0);
        frac.powf(1.5)
    }

    /// Fraction of time remaining in [0, 1].
    pub fn time_remaining_frac(&self) -> f64 {
        if self.total_duration_secs <= 0.0 { return 0.0; }
        1.0 - (self.elapsed_secs / self.total_duration_secs).min(1.0)
    }

    /// True when we are in the final 10 % of the execution window.
    pub fn is_near_deadline(&self) -> bool {
        self.time_remaining_frac() < 0.10
    }
}

// ---- PoVStrategy -----------------------------------------------------------

/// Percent-of-Volume strategy: given total quantity and a volume schedule,
/// compute slice sizes that maintain constant participation.
#[derive(Debug, Clone)]
pub struct PoVStrategy {
    /// Total quantity to execute
    pub total_qty: f64,
    /// Target fraction of market volume per period
    pub target_pov: f64,
    /// Whether to clamp slices to remaining inventory (prevents over-execution)
    pub clamp_to_inventory: bool,
}

/// A scheduled execution slice produced by PoV strategy.
#[derive(Debug, Clone)]
pub struct PoVSlice {
    /// Slice index (0-based)
    pub index: usize,
    /// Expected market volume in this period
    pub expected_market_vol: f64,
    /// Recommended order quantity
    pub order_qty: f64,
    /// Running total sent after this slice
    pub cumulative_sent: f64,
}

impl PoVStrategy {
    pub fn new(total_qty: f64, target_pov: f64) -> Self {
        PoVStrategy {
            total_qty,
            target_pov: target_pov.min(1.0).max(0.0),
            clamp_to_inventory: true,
        }
    }

    pub fn with_clamp(mut self, clamp: bool) -> Self {
        self.clamp_to_inventory = clamp;
        self
    }

    /// Compute the full slice schedule given a market volume forecast per period.
    ///
    /// `market_vol_schedule` -- expected market volume in each period (e.g. ADV / n_periods)
    /// Returns slices; they may not sum exactly to total_qty if market volume is insufficient.
    pub fn compute_slices(&self, market_vol_schedule: &[f64]) -> Vec<PoVSlice> {
        let mut slices = Vec::with_capacity(market_vol_schedule.len());
        let mut remaining = self.total_qty;
        let mut cumulative = 0.0;

        for (i, &mkt_vol) in market_vol_schedule.iter().enumerate() {
            if remaining <= 1e-9 { break; }

            let raw_qty = mkt_vol * self.target_pov;
            let qty = if self.clamp_to_inventory {
                raw_qty.min(remaining)
            } else {
                raw_qty
            };

            remaining -= qty;
            cumulative += qty;

            slices.push(PoVSlice {
                index: i,
                expected_market_vol: mkt_vol,
                order_qty: qty,
                cumulative_sent: cumulative,
            });
        }
        slices
    }

    /// Compute a single-slice quantity given observed market volume.
    pub fn qty_for_volume(&self, market_vol: f64, inventory: f64) -> f64 {
        let raw = market_vol * self.target_pov;
        raw.min(inventory).max(0.0)
    }

    /// Total expected execution if market volumes materialise as forecast.
    pub fn expected_total(&self, market_vol_schedule: &[f64]) -> f64 {
        self.compute_slices(market_vol_schedule)
            .iter()
            .map(|s| s.order_qty)
            .sum()
    }
}

// ---- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_volume_tracker_basic() {
        let mut t = VolumeTracker::new(60);
        t.push(100, 500.0);
        t.push(110, 300.0);
        t.push(120, 200.0);
        assert!((t.total_in_window() - 1000.0).abs() < 1e-9);
        assert_eq!(t.len(), 3);
    }

    #[test]
    fn test_volume_tracker_eviction() {
        let mut t = VolumeTracker::new(60);
        t.push(0, 1000.0);
        t.push(61, 200.0); // pushes out entry at t=0
        assert!((t.total_in_window() - 200.0).abs() < 1e-9);
    }

    #[test]
    fn test_volume_tracker_compute_rate_since() {
        let mut t = VolumeTracker::new(300);
        t.push(0, 100.0);
        t.push(100, 200.0);
        t.push(200, 150.0);
        // Rate since t=50: should include 200+150 = 350
        let rate = t.compute_rate(50);
        assert!((rate - 350.0).abs() < 1e-9);
    }

    #[test]
    fn test_slippage_budget() {
        let mut b = SlippageBudget::new(10.0);
        assert!((b.remaining() - 10.0).abs() < 1e-9);
        assert!(!b.is_exhausted());
        b.record_fill(6.0);
        assert!((b.remaining() - 4.0).abs() < 1e-9);
        b.record_fill(5.0);
        assert!(b.is_exhausted());
        assert_eq!(b.remaining(), 0.0);
    }

    #[test]
    fn test_prc_compute_order_size() {
        let prc = ParticipationRateController::new(0.10, 60, 10_000.0);
        // 10 % of 5000 volume = 500; capped by inventory 10000
        assert!((prc.compute_order_size(5_000.0, 10_000.0) - 500.0).abs() < 1e-9);
        // Inventory constraint: only 200 left
        assert!((prc.compute_order_size(5_000.0, 200.0) - 200.0).abs() < 1e-9);
    }

    #[test]
    fn test_adaptive_rate_urgency() {
        let mut apr = AdaptiveParticipationRate::new(0.10, 3600.0);
        // At start: urgency = 0, rate = base_rate
        assert!((apr.current_rate() - 0.10).abs() < 1e-9);
        // Halfway: urgency = 0.5^1.5 ~= 0.354, rate = 0.10 * 1.177 ~= 0.1177
        apr.update(1800.0);
        let r = apr.current_rate();
        assert!(r > 0.10, "rate={}", r);
        // At end: urgency = 1.0, rate = 0.10 * 1.5 = 0.15
        apr.update(3600.0);
        let r_end = apr.current_rate();
        assert!((r_end - 0.15).abs() < 1e-6, "r_end={}", r_end);
    }

    #[test]
    fn test_adaptive_rate_capped() {
        let apr = AdaptiveParticipationRate::new(0.40, 3600.0).with_max_rate(0.45);
        // Even with urgency=1 rate should not exceed max_rate
        let mut apr2 = apr;
        apr2.update(3600.0);
        assert!(apr2.current_rate() <= 0.45 + 1e-9);
    }

    #[test]
    fn test_pov_strategy_slices_sum() {
        let pov = PoVStrategy::new(10_000.0, 0.10);
        let mkt_vols: Vec<f64> = vec![50_000.0; 20]; // 20 periods, 50k vol each
        let slices = pov.compute_slices(&mkt_vols);
        let total: f64 = slices.iter().map(|s| s.order_qty).sum();
        // 10% of 50k = 5000/period; need 10k total => fills in 2 periods
        assert!((total - 10_000.0).abs() < 1e-6, "total={}", total);
    }

    #[test]
    fn test_pov_strategy_partial_fill() {
        // Market volume is insufficient to fill the full order
        let pov = PoVStrategy::new(10_000.0, 0.10);
        let mkt_vols: Vec<f64> = vec![10_000.0; 5]; // only 5k available at 10%
        let total = pov.expected_total(&mkt_vols);
        assert!((total - 5_000.0).abs() < 1e-6, "total={}", total);
    }

    #[test]
    fn test_prc_window_integration() {
        let mut prc = ParticipationRateController::new(0.10, 60, 10_000.0);
        prc.observe_volume(1000, 2000.0);
        prc.observe_volume(1020, 3000.0);
        prc.observe_volume(1040, 1500.0);
        let qty = prc.compute_order_size_from_window(1060);
        // Window vol = 6500, 10% = 650
        assert!((qty - 650.0).abs() < 1e-6, "qty={}", qty);
        prc.record_sent(qty);
        assert!((prc.remaining_qty() - (10_000.0 - qty)).abs() < 1e-6);
    }

    #[test]
    fn test_prc_completion() {
        let mut prc = ParticipationRateController::new(0.50, 60, 1000.0);
        prc.record_sent(500.0);
        assert!(!prc.is_complete());
        prc.record_sent(500.0);
        assert!(prc.is_complete());
    }

    #[test]
    fn test_slippage_budget_max_cost() {
        let b = SlippageBudget::new(5.0);
        // 5 bps * 1000 qty * 100 price / 10000 = 5.0
        let cost = b.max_cost_notional(1000.0, 100.0);
        assert!((cost - 5.0).abs() < 1e-9);
    }
}
