// alpha-decay: Signal alpha decay measurement and prediction library.
// Provides IC analysis, portfolio alpha tracking, and simulation utilities.

pub mod ic_analysis;
pub mod portfolio_alpha;
pub mod simulation;

pub use ic_analysis::{
    decay_curve::{DecayCurve, DecayModel, DecayFit, HalfLifeEstimate},
    ic_series::{IcSeries, IcHorizon, IcObservation, IcStats},
    icir_tracker::{IcirTracker, SignalState, RegimeIcir},
};
pub use portfolio_alpha::{
    alpha_book::{AlphaBook, AlphaSignalEntry, CrowdingFlag},
    alpha_combination::{AlphaCombiner, CombinationMethod, CombinedAlpha},
    factor_alpha::{FactorAlphaModel, FactorExposure, AlphaDecomposition},
};
pub use simulation::{
    alpha_simulator::{AlphaSimulator, DecayScenario, SimulationResult},
    capacity::{CapacityModel, CapacityEstimate},
};

/// Number of bars in one trading day (used for bar-to-day conversion).
pub const BARS_PER_DAY: usize = 390;

/// Minimum observations required for IC estimation.
pub const MIN_IC_OBS: usize = 30;

/// Default Newey-West lag truncation (sqrt rule).
pub fn newey_west_lags(n: usize) -> usize {
    ((n as f64).sqrt().floor() as usize).max(1)
}

/// Compute Spearman rank correlation between two slices.
pub fn spearman_rank_corr(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "Slices must be same length");
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let rx = rank_data(x);
    let ry = rank_data(y);
    pearson_corr(&rx, &ry)
}

/// Compute Pearson correlation between two slices.
pub fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let cov: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - mx) * (b - my)).sum::<f64>() / n;
    let sx = (x.iter().map(|a| (a - mx).powi(2)).sum::<f64>() / n).sqrt();
    let sy = (y.iter().map(|b| (b - my).powi(2)).sum::<f64>() / n).sqrt();
    if sx < 1e-12 || sy < 1e-12 {
        0.0
    } else {
        cov / (sx * sy)
    }
}

/// Assign ranks (average ranks for ties).
pub fn rank_data(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && (x[idx[j]] - x[idx[i]]).abs() < 1e-14 {
            j += 1;
        }
        let avg_rank = (i + j - 1) as f64 / 2.0 + 1.0;
        for k in i..j {
            ranks[idx[k]] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// OLS regression: returns (beta, alpha) for y = alpha + beta*x.
pub fn ols_simple(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let sxx: f64 = x.iter().map(|v| (v - mx).powi(2)).sum();
    let sxy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| (xi - mx) * (yi - my)).sum();
    let beta = if sxx.abs() < 1e-14 { 0.0 } else { sxy / sxx };
    let alpha = my - beta * mx;
    (beta, alpha)
}

/// Newey-West HAC standard error for a time series of products e_t * x_t.
/// Returns variance estimate.
pub fn newey_west_variance(scores: &[f64], lags: usize) -> f64 {
    let n = scores.len() as f64;
    let mean = scores.iter().sum::<f64>() / n;
    let centered: Vec<f64> = scores.iter().map(|v| v - mean).collect();
    let mut sigma = centered.iter().map(|v| v * v).sum::<f64>() / n;
    for l in 1..=lags {
        let bartlett = 1.0 - l as f64 / (lags as f64 + 1.0);
        let gamma_l: f64 = centered[l..]
            .iter()
            .zip(centered[..centered.len() - l].iter())
            .map(|(a, b)| a * b)
            .sum::<f64>()
            / n;
        sigma += 2.0 * bartlett * gamma_l;
    }
    sigma.max(1e-14)
}
