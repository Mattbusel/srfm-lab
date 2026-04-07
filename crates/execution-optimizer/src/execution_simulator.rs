/// execution_simulator.rs -- simulate order fills and compare execution strategies.
///
/// Provides fill simulation against a provided price path, and a multi-path
/// backtest harness that compares TWAP, VWAP, AC-optimal, and IS strategies.

use crate::optimal_execution::{
    ac_optimal_schedule, twap_schedule, vwap_schedule,
};

// ── ImpactModel ───────────────────────────────────────────────────────────────

/// Market-impact model used during simulation.
#[derive(Debug, Clone)]
pub enum ImpactModel {
    /// Linear temporary impact: cost per share = eta * (qty / time_step).
    Linear { eta: f64 },
    /// Square-root temporary impact: cost per share = eta * sqrt(qty / adv).
    SqrtModel { eta: f64 },
    /// Almgren-Chriss combined model: temporary eta + permanent gamma.
    AC { eta: f64, gamma: f64 },
}

impl ImpactModel {
    /// Temporary price impact for trading `qty` shares in one interval.
    ///
    /// Returns the impact as a fraction of price (e.g. 0.001 = 10 bps).
    pub fn temp_impact(&self, qty: f64, adv: f64) -> f64 {
        match self {
            ImpactModel::Linear { eta } => {
                // Impact proportional to participation rate.
                let participation = qty / adv.max(1.0);
                eta * participation
            }
            ImpactModel::SqrtModel { eta } => {
                // Impact proportional to sqrt of participation rate.
                let participation = qty / adv.max(1.0);
                eta * participation.sqrt()
            }
            ImpactModel::AC { eta, gamma: _ } => {
                let participation = qty / adv.max(1.0);
                eta * participation
            }
        }
    }

    /// Permanent price impact from trading `qty` shares (shifts subsequent prices).
    pub fn perm_impact(&self, qty: f64, adv: f64) -> f64 {
        match self {
            ImpactModel::AC { eta: _, gamma } => {
                let participation = qty / adv.max(1.0);
                gamma * participation
            }
            _ => 0.0,
        }
    }
}

// ── SimResult ─────────────────────────────────────────────────────────────────

/// Result of a single execution simulation.
#[derive(Debug, Clone)]
pub struct SimResult {
    /// (price, qty) pairs for each fill, in execution order.
    pub fills:           Vec<(f64, f64)>,
    /// Volume-weighted average fill price.
    pub avg_fill_price:  f64,
    /// Slippage vs. arrival price in basis points.
    pub slippage_bps:    f64,
    /// Total execution cost in basis points (impact + half-spread).
    pub total_cost_bps:  f64,
}

impl SimResult {
    /// Arrival price is the price at the first fill.
    pub fn arrival_price(&self) -> Option<f64> {
        self.fills.first().map(|(p, _)| *p)
    }

    /// Total quantity filled.
    pub fn total_qty(&self) -> f64 {
        self.fills.iter().map(|(_, q)| q).sum()
    }
}

// ── ExecutionSimulator ────────────────────────────────────────────────────────

/// Simulates execution of a schedule against a price path.
pub struct ExecutionSimulator {
    /// Average daily volume, used for impact calculations.
    pub adv: f64,
}

impl ExecutionSimulator {
    pub fn new(adv: f64) -> Self {
        ExecutionSimulator { adv }
    }

    /// Simulate fills for a given schedule and price path.
    ///
    /// `schedule`    -- per-interval trade sizes (positive = buy).
    /// `price_path`  -- mid-price at each interval (length >= schedule.len()).
    /// `spread_bps`  -- bid-ask spread in bps (half-spread is the crossing cost).
    /// `impact_model`-- impact model to apply.
    pub fn simulate_fills(
        &self,
        schedule:     &[f64],
        price_path:   &[f64],
        spread_bps:   f64,
        impact_model: &ImpactModel,
    ) -> SimResult {
        assert!(
            price_path.len() >= schedule.len(),
            "price_path must have at least as many entries as schedule"
        );

        let half_spread_frac = spread_bps / 2.0 / 10_000.0;
        let arrival_price = price_path[0];

        let mut fills: Vec<(f64, f64)> = Vec::with_capacity(schedule.len());
        let mut cum_perm_impact = 0.0f64;

        for (i, &qty) in schedule.iter().enumerate() {
            if qty == 0.0 {
                continue;
            }
            let mid = price_path[i];

            // Apply accumulated permanent impact to mid.
            let adjusted_mid = mid + mid * cum_perm_impact;

            // Temporary impact for this interval.
            let temp = impact_model.temp_impact(qty, self.adv);

            // Crossing the spread (buyer pays ask = mid + half-spread).
            let fill_price = adjusted_mid * (1.0 + half_spread_frac + temp);

            fills.push((fill_price, qty));

            // Permanent impact shifts future mid prices.
            cum_perm_impact += impact_model.perm_impact(qty, self.adv);
        }

        if fills.is_empty() {
            return SimResult {
                fills,
                avg_fill_price:  arrival_price,
                slippage_bps:    0.0,
                total_cost_bps:  0.0,
            };
        }

        // VWAP fill price.
        let total_qty: f64 = fills.iter().map(|(_, q)| q).sum();
        let vwap_fill: f64 = fills.iter().map(|(p, q)| p * q).sum::<f64>() / total_qty;

        let slippage_bps = (vwap_fill / arrival_price - 1.0) * 10_000.0;
        let total_cost_bps = slippage_bps;

        SimResult {
            fills,
            avg_fill_price:  vwap_fill,
            slippage_bps,
            total_cost_bps,
        }
    }
}

// ── ScheduleComparison ────────────────────────────────────────────────────────

/// Mean execution costs for each strategy across a set of price paths.
#[derive(Debug, Clone)]
pub struct ScheduleComparison {
    /// Mean total cost in bps for TWAP strategy.
    pub twap_cost: f64,
    /// Mean total cost in bps for VWAP strategy.
    pub vwap_cost: f64,
    /// Mean total cost in bps for Almgren-Chriss optimal strategy.
    pub ac_cost:   f64,
    /// Mean total cost in bps for Implementation Shortfall strategy.
    pub is_cost:   f64,
}

/// Backtest multiple schedule strategies across many simulated price paths.
///
/// For each price path, runs TWAP, VWAP (uniform profile), AC optimal, and
/// Implementation Shortfall (front-loaded) schedules.  Returns mean costs.
///
/// `price_paths` -- each inner Vec is one simulated price path of length >= n.
/// `qty`         -- total shares to execute.
/// `n`           -- number of execution intervals.
pub fn backtest_schedule_strategies(
    price_paths: &[Vec<f64>],
    qty:         f64,
    n:           usize,
) -> ScheduleComparison {
    assert!(!price_paths.is_empty(), "price_paths must not be empty");
    assert!(n >= 1, "n must be >= 1");
    assert!(qty > 0.0, "qty must be positive");

    // Default simulation parameters.
    let spread_bps = 5.0;
    let adv = 1_000_000.0;
    let sim = ExecutionSimulator::new(adv);
    let impact = ImpactModel::Linear { eta: 0.1 };

    // Pre-build schedules.
    let twap  = twap_schedule(qty, n);
    let uniform_profile = vec![1.0f64 / n as f64; n];
    let vwap_sched = vwap_schedule(qty, &uniform_profile);

    // AC optimal with representative parameters.
    let ac_sched = ac_optimal_schedule(
        qty, 1.0, n,
        0.02,  // sigma
        1e-5,  // eta
        1e-6,  // gamma
        1e-4,  // risk_aversion
    );

    // IS (implementation shortfall): front-loaded -- 50% in first interval.
    let is_sched = build_is_schedule(qty, n);

    let mut twap_sum = 0.0f64;
    let mut vwap_sum = 0.0f64;
    let mut ac_sum   = 0.0f64;
    let mut is_sum   = 0.0f64;
    let k = price_paths.len() as f64;

    for path in price_paths {
        twap_sum += sim.simulate_fills(&twap,       path, spread_bps, &impact).total_cost_bps;
        vwap_sum += sim.simulate_fills(&vwap_sched, path, spread_bps, &impact).total_cost_bps;
        ac_sum   += sim.simulate_fills(&ac_sched,   path, spread_bps, &impact).total_cost_bps;
        is_sum   += sim.simulate_fills(&is_sched,   path, spread_bps, &impact).total_cost_bps;
    }

    ScheduleComparison {
        twap_cost: twap_sum / k,
        vwap_cost: vwap_sum / k,
        ac_cost:   ac_sum   / k,
        is_cost:   is_sum   / k,
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Build an IS (implementation shortfall) front-loaded schedule.
/// 50% in the first interval, remainder spread uniformly.
fn build_is_schedule(qty: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![qty];
    }
    let first = qty * 0.5;
    let remainder = qty - first;
    let rest_slice = remainder / (n - 1) as f64;
    let mut s = vec![rest_slice; n];
    s[0] = first;
    s
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_path(price: f64, len: usize) -> Vec<f64> {
        vec![price; len]
    }

    fn drifting_path(start: f64, drift_per_step: f64, len: usize) -> Vec<f64> {
        (0..len).map(|i| start + i as f64 * drift_per_step).collect()
    }

    // -- simulate_fills -------------------------------------------------------

    #[test]
    fn test_simulate_fills_flat_price_positive_cost() {
        let sim = ExecutionSimulator::new(1_000_000.0);
        let schedule = twap_schedule(1000.0, 10);
        let path = flat_path(100.0, 10);
        let result = sim.simulate_fills(
            &schedule,
            &path,
            5.0,
            &ImpactModel::Linear { eta: 0.05 },
        );
        assert_eq!(result.fills.len(), 10);
        assert!(result.avg_fill_price > 100.0, "fill price must exceed mid due to spread+impact");
        assert!(result.total_cost_bps > 0.0);
    }

    #[test]
    fn test_simulate_fills_correct_total_qty() {
        let sim = ExecutionSimulator::new(1_000_000.0);
        let qty = 5000.0;
        let schedule = twap_schedule(qty, 5);
        let path = flat_path(200.0, 5);
        let result = sim.simulate_fills(&schedule, &path, 2.0, &ImpactModel::Linear { eta: 0.01 });
        let total = result.total_qty();
        assert!((total - qty).abs() < 1e-6, "total qty {} != {}", total, qty);
    }

    #[test]
    fn test_simulate_fills_sqrt_model() {
        let sim = ExecutionSimulator::new(1_000_000.0);
        let schedule = twap_schedule(1000.0, 4);
        let path = flat_path(50.0, 4);
        let result = sim.simulate_fills(
            &schedule,
            &path,
            10.0,
            &ImpactModel::SqrtModel { eta: 0.2 },
        );
        assert!(result.total_cost_bps > 0.0);
    }

    #[test]
    fn test_simulate_fills_ac_model_permanent_impact() {
        let sim = ExecutionSimulator::new(500_000.0);
        let schedule = vec![1000.0, 1000.0]; // 2 intervals
        let path = flat_path(100.0, 2);
        let result = sim.simulate_fills(
            &schedule,
            &path,
            5.0,
            &ImpactModel::AC { eta: 0.1, gamma: 0.05 },
        );
        // Second fill should be slightly more expensive due to perm impact.
        assert!(result.fills[1].0 >= result.fills[0].0);
    }

    #[test]
    fn test_simulate_fills_zero_schedule_returns_arrival_price() {
        let sim = ExecutionSimulator::new(1_000_000.0);
        let schedule = vec![0.0, 0.0, 0.0];
        let path = flat_path(75.0, 3);
        let result = sim.simulate_fills(&schedule, &path, 5.0, &ImpactModel::Linear { eta: 0.1 });
        assert!(result.fills.is_empty());
        assert_eq!(result.total_qty(), 0.0);
    }

    #[test]
    fn test_simulate_fills_rising_market_higher_cost() {
        let sim = ExecutionSimulator::new(1_000_000.0);
        let schedule_early = {
            let mut s = vec![0.0f64; 10];
            s[0] = 1000.0;
            s
        };
        let schedule_late = {
            let mut s = vec![0.0f64; 10];
            s[9] = 1000.0;
            s
        };
        let path = drifting_path(100.0, 1.0, 10); // rising 1 point/step
        let impact = ImpactModel::Linear { eta: 0.0 };
        let r_early = sim.simulate_fills(&schedule_early, &path, 5.0, &impact);
        let r_late  = sim.simulate_fills(&schedule_late,  &path, 5.0, &impact);
        assert!(
            r_early.avg_fill_price < r_late.avg_fill_price,
            "early fill {} should be cheaper than late fill {} in rising market",
            r_early.avg_fill_price,
            r_late.avg_fill_price
        );
    }

    // -- backtest_schedule_strategies -----------------------------------------

    #[test]
    fn test_backtest_returns_finite_costs() {
        let paths: Vec<Vec<f64>> = (0..10)
            .map(|i| flat_path(100.0 + i as f64 * 0.5, 20))
            .collect();
        let result = backtest_schedule_strategies(&paths, 1000.0, 20);
        assert!(result.twap_cost.is_finite());
        assert!(result.vwap_cost.is_finite());
        assert!(result.ac_cost.is_finite());
        assert!(result.is_cost.is_finite());
    }

    #[test]
    fn test_backtest_single_path() {
        let paths = vec![flat_path(50.0, 5)];
        let result = backtest_schedule_strategies(&paths, 100.0, 5);
        // All strategies on a flat path should yield similar costs.
        assert!(result.twap_cost >= 0.0);
        assert!(result.vwap_cost >= 0.0);
    }

    #[test]
    fn test_is_schedule_front_loaded() {
        let s = build_is_schedule(1000.0, 4);
        assert_eq!(s.len(), 4);
        assert!((s.iter().sum::<f64>() - 1000.0).abs() < 1e-9);
        // First interval should have 50%.
        assert!((s[0] - 500.0).abs() < 1e-9);
    }

    #[test]
    fn test_is_schedule_single_interval() {
        let s = build_is_schedule(300.0, 1);
        assert_eq!(s.len(), 1);
        assert!((s[0] - 300.0).abs() < 1e-9);
    }

    #[test]
    fn test_impact_model_linear_zero_qty() {
        let m = ImpactModel::Linear { eta: 0.5 };
        let impact = m.temp_impact(0.0, 1_000_000.0);
        assert_eq!(impact, 0.0);
    }

    #[test]
    fn test_impact_model_sqrt_greater_than_linear_for_small_qty() {
        // For participation < 1, sqrt(p) > p, so sqrt impact > linear impact.
        let qty = 50_000.0;
        let adv = 1_000_000.0; // participation = 5%
        let linear = ImpactModel::Linear { eta: 0.1 }.temp_impact(qty, adv);
        let sqrt   = ImpactModel::SqrtModel { eta: 0.1 }.temp_impact(qty, adv);
        // participation = 0.05; sqrt(0.05) ~= 0.224 > 0.05
        assert!(sqrt > linear, "sqrt impact should be larger than linear for pct < 1");
    }
}
