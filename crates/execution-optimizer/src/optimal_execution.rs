/// optimal_execution.rs -- closed-form optimal execution schedules.
///
/// Implements Almgren-Chriss (2001) optimal schedule, TWAP, VWAP, and
/// constant participation-rate schedules.  All schedules return a Vec<f64>
/// of per-interval trade sizes that sum to `qty`.

// ── OptimalExecutionEngine ────────────────────────────────────────────────────

/// Stateless engine that produces optimal execution schedules.
pub struct OptimalExecutionEngine;

impl OptimalExecutionEngine {
    pub fn new() -> Self { OptimalExecutionEngine }
}

impl Default for OptimalExecutionEngine {
    fn default() -> Self { Self::new() }
}

// ── Almgren-Chriss ────────────────────────────────────────────────────────────

/// Compute the Almgren-Chriss optimal execution schedule.
///
/// # Parameters
/// - `qty`           Total shares to execute (positive).
/// - `T`             Total execution horizon in the same time units as `sigma`.
/// - `n`             Number of intervals.
/// - `sigma`         Volatility per unit time (must match units of T).
/// - `eta`           Temporary impact coefficient (cost per unit of trade rate).
/// - `gamma`         Permanent impact coefficient (cost per unit of trade size).
/// - `risk_aversion` Lambda: risk-aversion coefficient.
///
/// # Returns
/// Vec of length `n` containing trade sizes (positive) for each interval.
/// The sizes sum to `qty`.
///
/// # Reference
/// Almgren & Chriss (2001), equations (18)-(20).
pub fn ac_optimal_schedule(
    qty:           f64,
    t_horizon:     f64,
    n:             usize,
    sigma:         f64,
    eta:           f64,
    _gamma:        f64,
    risk_aversion: f64,
) -> Vec<f64> {
    assert!(n >= 1, "n must be >= 1");
    assert!(qty > 0.0, "qty must be positive");
    assert!(eta > 0.0, "eta must be positive");
    assert!(t_horizon > 0.0, "t_horizon must be positive");

    let tau = t_horizon / n as f64;

    // Characteristic decay rate kappa = sqrt(lambda * sigma^2 / eta).
    let kappa = (risk_aversion * sigma * sigma / eta).sqrt();

    // Time points at the START of each interval: t_j = j * tau, j = 0..n-1.
    // Optimal inventory at t_j: x_j = sinh(kappa*(T - t_j)) / sinh(kappa*T) * X.
    // Trade at interval j = x_j - x_{j+1}.

    let sinh_kappa_t = (kappa * t_horizon).sinh();

    // Handle degenerate case where kappa is near zero -- use TWAP.
    if kappa.abs() < 1e-12 || sinh_kappa_t.abs() < 1e-12 {
        return twap_schedule(qty, n);
    }

    // Compute inventory trajectory x_j for j = 0..=n.
    let inventory: Vec<f64> = (0..=n)
        .map(|j| {
            let t_j = j as f64 * tau;
            let remaining = t_horizon - t_j;
            (kappa * remaining).sinh() / sinh_kappa_t * qty
        })
        .collect();

    // Trade sizes: n_j = x_j - x_{j+1} for j = 0..n-1.
    let mut trades: Vec<f64> = inventory
        .windows(2)
        .map(|w| w[0] - w[1])
        .collect();

    // Clamp to zero and renormalize to ensure exact sum.
    for t in &mut trades {
        if *t < 0.0 { *t = 0.0; }
    }
    let total: f64 = trades.iter().sum();
    if total > 0.0 {
        for t in &mut trades {
            *t = *t / total * qty;
        }
    }

    trades
}

/// Expected execution cost for an AC schedule.
///
/// Cost = sum_j [ gamma * n_j * x_j + eta * n_j^2 / tau ]
/// where x_j is remaining inventory at step j and n_j is the trade at step j.
pub fn ac_expected_cost(
    schedule: &[f64],
    sigma:    f64,
    eta:      f64,
    gamma:    f64,
) -> f64 {
    if schedule.is_empty() { return 0.0; }

    // Reconstruct inventory: x_0 = sum(schedule), x_{j+1} = x_j - n_j.
    let total: f64 = schedule.iter().sum();
    let mut inventory = total;
    let mut cost = 0.0;

    // We assume unit tau (1 per interval) for simplicity unless caller provides.
    let tau = 1.0;

    for &n_j in schedule {
        // Permanent impact on remaining inventory.
        cost += gamma * n_j * inventory;
        // Temporary impact (linear in trade rate).
        cost += eta * n_j * n_j / tau;
        inventory -= n_j;
    }

    // Add spread cost approximation: 0.5 * sigma^2 * sum(x_j^2 * tau).
    // This is the timing-risk component.
    inventory = total;
    for &n_j in schedule {
        cost += 0.5 * sigma * sigma * inventory * inventory * tau;
        inventory -= n_j;
    }

    cost
}

/// Variance of execution cost for an AC schedule (timing risk component).
///
/// Var = sigma^2 * sum_j ( x_j^2 * tau )
pub fn ac_variance(
    schedule: &[f64],
    sigma:    f64,
    _eta:     f64,
    _gamma:   f64,
) -> f64 {
    if schedule.is_empty() { return 0.0; }

    let total: f64 = schedule.iter().sum();
    let mut inventory = total;
    let tau = 1.0;
    let mut variance = 0.0;

    for &n_j in schedule {
        variance += sigma * sigma * inventory * inventory * tau;
        inventory -= n_j;
    }

    variance
}

// ── TWAP ──────────────────────────────────────────────────────────────────────

/// Uniform time-weighted schedule: equal trade size each interval.
pub fn twap_schedule(qty: f64, n: usize) -> Vec<f64> {
    assert!(n >= 1, "n must be >= 1");
    let slice = qty / n as f64;
    vec![slice; n]
}

// ── VWAP ──────────────────────────────────────────────────────────────────────

/// Volume-profile-weighted schedule.
///
/// Trade sizes are proportional to `volume_profile`, normalized to sum to `qty`.
/// `volume_profile` must be non-negative and have at least one positive entry.
pub fn vwap_schedule(qty: f64, volume_profile: &[f64]) -> Vec<f64> {
    assert!(!volume_profile.is_empty(), "volume_profile must not be empty");
    let total_vol: f64 = volume_profile.iter().sum();
    assert!(total_vol > 0.0, "volume_profile must have a positive sum");
    volume_profile
        .iter()
        .map(|&v| (v / total_vol) * qty)
        .collect()
}

// ── Participation rate ────────────────────────────────────────────────────────

/// Constant participation-rate schedule.
///
/// Spreads `qty` across `n` intervals such that each interval trades
/// `target_pct` of the expected per-interval volume (= adv / n).
/// If the implied total exceeds `qty`, sizes are scaled down to match `qty`.
///
/// # Parameters
/// - `qty`         Total shares to trade.
/// - `adv`         Average daily volume (in same units as qty).
/// - `target_pct`  Target participation rate (e.g. 0.10 = 10%).
/// - `n`           Number of intervals.
pub fn participation_rate_schedule(
    qty:        f64,
    adv:        f64,
    target_pct: f64,
    n:          usize,
) -> Vec<f64> {
    assert!(n >= 1, "n must be >= 1");
    assert!(adv > 0.0, "adv must be positive");
    assert!(target_pct > 0.0 && target_pct <= 1.0, "target_pct must be in (0, 1]");

    // Volume available in each interval.
    let interval_vol = adv / n as f64;
    // Unconstrained trade per interval.
    let unconstrained = interval_vol * target_pct;
    // Total unconstrained -- may differ from qty.
    let total_unconstrained = unconstrained * n as f64;

    if total_unconstrained <= 0.0 {
        return twap_schedule(qty, n);
    }

    // Scale to match qty exactly.
    let scale = qty / total_unconstrained;
    vec![unconstrained * scale; n]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // -- twap -----------------------------------------------------------------

    #[test]
    fn test_twap_sums_to_qty() {
        let schedule = twap_schedule(1000.0, 10);
        let total: f64 = schedule.iter().sum();
        assert!(approx_eq(total, 1000.0, EPS));
    }

    #[test]
    fn test_twap_equal_slices() {
        let schedule = twap_schedule(300.0, 3);
        for &s in &schedule {
            assert!(approx_eq(s, 100.0, EPS));
        }
    }

    #[test]
    fn test_twap_single_interval() {
        let schedule = twap_schedule(500.0, 1);
        assert_eq!(schedule.len(), 1);
        assert!(approx_eq(schedule[0], 500.0, EPS));
    }

    // -- vwap -----------------------------------------------------------------

    #[test]
    fn test_vwap_sums_to_qty() {
        let profile = vec![0.1, 0.2, 0.4, 0.2, 0.1];
        let schedule = vwap_schedule(1000.0, &profile);
        let total: f64 = schedule.iter().sum();
        assert!(approx_eq(total, 1000.0, 1e-9));
    }

    #[test]
    fn test_vwap_proportions() {
        let profile = vec![1.0, 3.0]; // 25% and 75%
        let schedule = vwap_schedule(400.0, &profile);
        assert!(approx_eq(schedule[0], 100.0, EPS));
        assert!(approx_eq(schedule[1], 300.0, EPS));
    }

    // -- participation rate ---------------------------------------------------

    #[test]
    fn test_participation_rate_sums_to_qty() {
        let schedule = participation_rate_schedule(500.0, 1_000_000.0, 0.05, 20);
        let total: f64 = schedule.iter().sum();
        assert!(approx_eq(total, 500.0, 1e-6));
    }

    #[test]
    fn test_participation_rate_equal_slices() {
        let schedule = participation_rate_schedule(200.0, 1_000_000.0, 0.10, 10);
        for &s in &schedule {
            assert!(approx_eq(s, 20.0, 1e-6));
        }
    }

    // -- ac_optimal_schedule --------------------------------------------------

    #[test]
    fn test_ac_schedule_sums_to_qty() {
        let qty = 10_000.0;
        let schedule = ac_optimal_schedule(qty, 1.0, 10, 0.02, 1e-6, 1e-7, 1e-4);
        let total: f64 = schedule.iter().sum();
        assert!(
            approx_eq(total, qty, qty * 1e-9),
            "AC schedule sum {} != qty {}",
            total,
            qty
        );
    }

    #[test]
    fn test_ac_schedule_length() {
        let schedule = ac_optimal_schedule(1000.0, 1.0, 5, 0.02, 1e-6, 1e-7, 1e-4);
        assert_eq!(schedule.len(), 5);
    }

    #[test]
    fn test_ac_schedule_front_loaded_for_high_risk_aversion() {
        // High risk aversion should front-load trades.
        let hi_ra = ac_optimal_schedule(1000.0, 1.0, 10, 0.02, 1e-6, 1e-7, 1e-2);
        let lo_ra = ac_optimal_schedule(1000.0, 1.0, 10, 0.02, 1e-6, 1e-7, 1e-6);
        // First interval should be larger for high risk aversion.
        assert!(
            hi_ra[0] >= lo_ra[0],
            "high risk aversion should front-load: hi={} lo={}",
            hi_ra[0],
            lo_ra[0]
        );
    }

    #[test]
    fn test_ac_schedule_degenerates_to_twap_at_zero_kappa() {
        // With very tiny risk aversion, schedule should be near uniform.
        let schedule = ac_optimal_schedule(1000.0, 1.0, 10, 0.001, 1e-3, 1e-4, 1e-15);
        let avg = 100.0;
        for &s in &schedule {
            assert!(
                (s - avg).abs() < avg * 0.1,
                "slice {} far from TWAP avg {}",
                s,
                avg
            );
        }
    }

    // -- cost and variance ----------------------------------------------------

    #[test]
    fn test_ac_expected_cost_positive() {
        let schedule = twap_schedule(1000.0, 10);
        let cost = ac_expected_cost(&schedule, 0.02, 1e-6, 1e-7);
        assert!(cost > 0.0, "expected cost should be positive");
    }

    #[test]
    fn test_ac_variance_positive() {
        let schedule = twap_schedule(1000.0, 10);
        let var = ac_variance(&schedule, 0.02, 1e-6, 1e-7);
        assert!(var > 0.0, "variance should be positive for non-zero schedule");
    }

    #[test]
    fn test_ac_variance_zero_for_empty() {
        let var = ac_variance(&[], 0.02, 1e-6, 1e-7);
        assert_eq!(var, 0.0);
    }
}
