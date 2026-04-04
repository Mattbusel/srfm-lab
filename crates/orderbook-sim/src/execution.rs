/// Execution algorithms: TWAP, VWAP, Implementation Shortfall, Percent-of-Volume.

use chrono::{DateTime, Duration, Utc};

/// A single slice of an execution schedule.
#[derive(Debug, Clone)]
pub struct ExecutionSlice {
    pub timestamp: DateTime<Utc>,
    pub qty: f64,
    pub limit_price: f64,
}

// ── TWAP ──────────────────────────────────────────────────────────────────────

/// Time-Weighted Average Price execution.
///
/// Splits `total_qty` into `n_slices` equal parts distributed uniformly
/// over `horizon_seconds`.
pub fn twap_schedule(
    total_qty: f64,
    n_slices: usize,
    horizon_seconds: i64,
    start_time: DateTime<Utc>,
    limit_price: f64,
) -> Vec<ExecutionSlice> {
    assert!(n_slices > 0, "n_slices must be > 0");
    let slice_qty = total_qty / n_slices as f64;
    let interval_secs = horizon_seconds / n_slices as i64;
    (0..n_slices)
        .map(|i| ExecutionSlice {
            timestamp: start_time + Duration::seconds(i as i64 * interval_secs),
            qty: slice_qty,
            limit_price,
        })
        .collect()
}

// ── VWAP ──────────────────────────────────────────────────────────────────────

/// Volume-Weighted Average Price execution.
///
/// `volume_profile` is a slice of historical volume fractions (must sum to ~1.0)
/// representing the expected volume in each time bucket.
pub fn vwap_schedule(
    total_qty: f64,
    volume_profile: &[f64],
    start_time: DateTime<Utc>,
    bucket_duration_secs: i64,
    limit_price: f64,
) -> Vec<ExecutionSlice> {
    let total_profile: f64 = volume_profile.iter().sum();
    volume_profile
        .iter()
        .enumerate()
        .map(|(i, &frac)| ExecutionSlice {
            timestamp: start_time + Duration::seconds(i as i64 * bucket_duration_secs),
            qty: total_qty * frac / total_profile,
            limit_price,
        })
        .collect()
}

// ── Implementation Shortfall ──────────────────────────────────────────────────

/// Implementation Shortfall (IS) schedule.
///
/// Minimise E[cost] + λ * Var[cost].
/// Uses the Almgren-Chriss front-loaded solution shape:
///   v_j ∝ sinh(κ(T − t_j)) / sinh(κT)
///
/// # Arguments
/// * `arrival_price` — mid price when order is received (decision price).
/// * `sigma` — per-interval volatility.
/// * `risk_aversion` — λ parameter (higher → more front-loaded).
/// * `temp_impact` — η (linear temporary impact coefficient).
pub fn is_schedule(
    total_qty: f64,
    n_slices: usize,
    start_time: DateTime<Utc>,
    interval_secs: i64,
    arrival_price: f64,
    sigma: f64,
    risk_aversion: f64,
    temp_impact: f64,
) -> Vec<ExecutionSlice> {
    assert!(n_slices > 0);
    let n = n_slices as f64;
    let kappa = (risk_aversion * sigma * sigma / temp_impact).sqrt();

    let mut weights: Vec<f64> = (0..n_slices)
        .map(|j| {
            let remaining_intervals = n - j as f64;
            (kappa * remaining_intervals).sinh()
        })
        .collect();
    let weight_sum: f64 = weights.iter().sum();
    for w in &mut weights {
        *w /= weight_sum;
    }

    weights
        .into_iter()
        .enumerate()
        .map(|(i, w)| ExecutionSlice {
            timestamp: start_time + Duration::seconds(i as i64 * interval_secs),
            qty: total_qty * w,
            limit_price: arrival_price, // limit = arrival price (aggressive IS)
        })
        .collect()
}

// ── Percent of Volume ─────────────────────────────────────────────────────────

/// Percent-of-Volume (POV) execution schedule.
///
/// Participates at a fixed `pov_rate` fraction of expected market volume
/// in each interval. The schedule will take as many intervals as needed.
///
/// # Arguments
/// * `expected_volume_per_interval` — expected total market volume per interval.
/// * `pov_rate` — participation rate, e.g. 0.10 = 10% of market volume.
pub fn pov_schedule(
    total_qty: f64,
    expected_volume_per_interval: f64,
    pov_rate: f64,
    start_time: DateTime<Utc>,
    interval_secs: i64,
    limit_price: f64,
) -> Vec<ExecutionSlice> {
    assert!(pov_rate > 0.0 && pov_rate <= 1.0);
    let per_interval = expected_volume_per_interval * pov_rate;
    let n_full = (total_qty / per_interval).floor() as usize;
    let remainder = total_qty - per_interval * n_full as f64;
    let mut slices: Vec<ExecutionSlice> = (0..n_full)
        .map(|i| ExecutionSlice {
            timestamp: start_time + Duration::seconds(i as i64 * interval_secs),
            qty: per_interval,
            limit_price,
        })
        .collect();
    if remainder > 1e-9 {
        slices.push(ExecutionSlice {
            timestamp: start_time + Duration::seconds(n_full as i64 * interval_secs),
            qty: remainder,
            limit_price,
        });
    }
    slices
}

// ── Schedule evaluation ───────────────────────────────────────────────────────

/// Compute the Implementation Shortfall given fills vs arrival price.
pub fn implementation_shortfall(
    fills: &[(f64, f64)], // (price, qty)
    arrival_price: f64,
    side_sign: f64, // +1 for buy, -1 for sell
) -> f64 {
    let total_qty: f64 = fills.iter().map(|(_, q)| q).sum();
    if total_qty == 0.0 {
        return 0.0;
    }
    let vwap: f64 = fills.iter().map(|(p, q)| p * q).sum::<f64>() / total_qty;
    side_sign * (vwap - arrival_price) / arrival_price
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn twap_sums_to_total() {
        let schedule = twap_schedule(10_000.0, 10, 3600, Utc::now(), 100.0);
        let total: f64 = schedule.iter().map(|s| s.qty).sum();
        assert!((total - 10_000.0).abs() < 1e-9);
        assert_eq!(schedule.len(), 10);
    }

    #[test]
    fn vwap_sums_to_total() {
        let profile = vec![0.1, 0.2, 0.3, 0.25, 0.15];
        let schedule = vwap_schedule(5_000.0, &profile, Utc::now(), 300, 50.0);
        let total: f64 = schedule.iter().map(|s| s.qty).sum();
        assert!((total - 5_000.0).abs() < 1e-9);
    }

    #[test]
    fn is_schedule_front_loaded() {
        let schedule = is_schedule(10_000.0, 5, Utc::now(), 60, 100.0, 0.02, 1e-6, 2.5e-6);
        // First slice should be larger than last (front-loaded).
        assert!(schedule[0].qty > schedule[4].qty);
        let total: f64 = schedule.iter().map(|s| s.qty).sum();
        assert!((total - 10_000.0).abs() < 1e-6);
    }
}
