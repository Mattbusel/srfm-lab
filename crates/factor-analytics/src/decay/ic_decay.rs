//! IC decay curve analysis.
//!
//! Computes Information Coefficient (IC) at lags 1..60,
//! fits an exponential decay model, and estimates half-life.

use ndarray::{Array1, Array2};
use crate::error::{FactorError, Result};
use crate::cross_section::normalize::rank_normalize;

/// IC decay analysis results.
#[derive(Debug, Clone)]
pub struct IcDecayCurve {
    /// IC at each lag (1..=max_lag)
    pub ic_by_lag: Vec<f64>,
    /// Lags (1, 2, ..., max_lag)
    pub lags: Vec<usize>,
    /// Fitted decay coefficient (lambda): IC(t) = IC_0 * exp(-lambda * t)
    pub decay_lambda: f64,
    /// Initial IC estimate (IC at lag 0, extrapolated)
    pub ic_zero: f64,
    /// Half-life in periods: ln(2) / lambda
    pub half_life: f64,
    /// R-squared of exponential fit
    pub fit_r_squared: f64,
}

/// Compute the Spearman rank IC between factor scores and forward returns.
///
/// IC = Spearman correlation between factor and next-period return.
///
/// Both series are rank-normalized before computing Pearson correlation,
/// which gives the Spearman rank correlation.
pub fn compute_ic(factor_scores: &[f64], forward_returns: &[f64]) -> f64 {
    let n = factor_scores.len().min(forward_returns.len());
    if n < 5 {
        return f64::NAN;
    }

    // Rank normalize both
    let ranked_f = rank_normalize(&factor_scores[..n]);
    let ranked_r = rank_normalize(&forward_returns[..n]);

    // Compute Pearson correlation of ranks = Spearman IC
    let pairs: Vec<(f64, f64)> = ranked_f
        .iter()
        .zip(ranked_r.iter())
        .filter_map(|(&f, &r)| {
            if f.is_finite() && r.is_finite() {
                Some((f, r))
            } else {
                None
            }
        })
        .collect();

    let m = pairs.len();
    if m < 5 {
        return f64::NAN;
    }

    let mf = pairs.iter().map(|p| p.0).sum::<f64>() / m as f64;
    let mr = pairs.iter().map(|p| p.1).sum::<f64>() / m as f64;

    let cov = pairs.iter().map(|p| (p.0 - mf) * (p.1 - mr)).sum::<f64>();
    let sf = pairs.iter().map(|p| (p.0 - mf).powi(2)).sum::<f64>().sqrt();
    let sr = pairs.iter().map(|p| (p.1 - mr).powi(2)).sum::<f64>().sqrt();

    if sf < 1e-12 || sr < 1e-12 {
        return 0.0;
    }

    cov / (sf * sr)
}

/// Compute IC at multiple lags.
///
/// # Arguments
/// * `factor_panel` -- shape (n_periods, n_assets), factor scores at each period
/// * `returns_panel` -- shape (n_periods, n_assets), returns at each period
/// * `max_lag` -- maximum lag to compute (typically 60)
///
/// Returns IC vector of length max_lag.
pub fn compute_ic_by_lag(
    factor_panel: &Array2<f64>,
    returns_panel: &Array2<f64>,
    max_lag: usize,
) -> Result<Vec<f64>> {
    let (n_periods, n_assets) = factor_panel.dim();
    if returns_panel.dim() != (n_periods, n_assets) {
        return Err(FactorError::ShapeMismatch {
            msg: format!(
                "factor_panel {:?} != returns_panel {:?}",
                factor_panel.dim(),
                returns_panel.dim()
            ),
        });
    }
    if n_periods <= max_lag {
        return Err(FactorError::InsufficientData {
            required: max_lag + 1,
            got: n_periods,
        });
    }

    let mut ic_vector = vec![f64::NAN; max_lag];

    for lag in 1..=max_lag {
        // For each period t, compute IC between factor at t and return at t+lag
        let mut ic_sum = 0.0;
        let mut ic_count = 0;

        for t in 0..n_periods - lag {
            let factor_slice: Vec<f64> = factor_panel.row(t).to_vec();
            let return_slice: Vec<f64> = returns_panel.row(t + lag).to_vec();
            let ic = compute_ic(&factor_slice, &return_slice);
            if ic.is_finite() {
                ic_sum += ic;
                ic_count += 1;
            }
        }

        ic_vector[lag - 1] = if ic_count > 0 {
            ic_sum / ic_count as f64
        } else {
            f64::NAN
        };
    }

    Ok(ic_vector)
}

/// Fit exponential decay model to IC curve.
///
/// Model: IC(t) = IC_0 * exp(-lambda * t)
/// Linearize: ln(IC(t)) = ln(IC_0) - lambda * t
/// Fit via OLS on ln(IC) ~ t.
///
/// Only uses positive IC values for fitting.
pub fn fit_exponential_decay(ic_by_lag: &[f64]) -> Result<(f64, f64, f64)> {
    // Returns (ic_zero, lambda, r_squared)
    let max_lag = ic_by_lag.len();

    // Collect valid (positive IC) points for fitting
    let mut xs: Vec<f64> = Vec::new();
    let mut ys: Vec<f64> = Vec::new();

    for (lag_idx, &ic) in ic_by_lag.iter().enumerate() {
        if ic.is_finite() && ic > 0.0 {
            xs.push((lag_idx + 1) as f64);
            ys.push(ic.ln());
        }
    }

    if xs.len() < 3 {
        return Err(FactorError::InsufficientData {
            required: 3,
            got: xs.len(),
        });
    }

    // OLS: ln(IC) = a + b*t  =>  IC_0 = exp(a), lambda = -b
    let (a, b) = crate::factors::momentum::ols_simple(&ys, &xs)?;

    let ic_zero = a.exp();
    let lambda = -b; // b should be negative for decay

    // Compute R-squared on log scale
    let y_mean = ys.iter().sum::<f64>() / ys.len() as f64;
    let ss_tot = ys.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>();
    let ss_res = xs.iter().zip(ys.iter())
        .map(|(x, y)| (y - (a + b * x)).powi(2))
        .sum::<f64>();
    let r_sq = if ss_tot > 1e-14 { 1.0 - ss_res / ss_tot } else { 0.0 };

    Ok((ic_zero, lambda, r_sq))
}

/// Compute the full IC decay curve analysis.
///
/// # Arguments
/// * `factor_panel` -- (n_periods, n_assets) factor scores
/// * `returns_panel` -- (n_periods, n_assets) returns
/// * `max_lag` -- typically 60
pub fn ic_decay_analysis(
    factor_panel: &Array2<f64>,
    returns_panel: &Array2<f64>,
    max_lag: usize,
) -> Result<IcDecayCurve> {
    let ic_by_lag = compute_ic_by_lag(factor_panel, returns_panel, max_lag)?;
    let lags: Vec<usize> = (1..=max_lag).collect();

    match fit_exponential_decay(&ic_by_lag) {
        Ok((ic_zero, lambda, r_sq)) => {
            let half_life = if lambda > 1e-10 {
                2.0_f64.ln() / lambda
            } else {
                f64::INFINITY
            };

            Ok(IcDecayCurve {
                ic_by_lag,
                lags,
                decay_lambda: lambda,
                ic_zero,
                half_life,
                fit_r_squared: r_sq,
            })
        }
        Err(_) => {
            // Return curve without decay fit
            Ok(IcDecayCurve {
                ic_by_lag,
                lags,
                decay_lambda: f64::NAN,
                ic_zero: f64::NAN,
                half_life: f64::NAN,
                fit_r_squared: f64::NAN,
            })
        }
    }
}

/// ICIR (Information Coefficient Information Ratio).
///
/// ICIR = mean(IC) / std(IC)  -- a measure of IC consistency.
pub fn compute_icir(ic_series: &[f64]) -> f64 {
    let valid: Vec<f64> = ic_series.iter().copied().filter(|v| v.is_finite()).collect();
    let n = valid.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = valid.iter().sum::<f64>() / n as f64;
    let std = (valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();
    if std < 1e-12 {
        return f64::INFINITY;
    }
    mean / std
}

/// Cumulative IC: compute the running sum of ICs to identify persistence regimes.
pub fn cumulative_ic(ic_series: &[f64]) -> Vec<f64> {
    let mut cum = 0.0;
    ic_series
        .iter()
        .map(|&ic| {
            if ic.is_finite() {
                cum += ic;
            }
            cum
        })
        .collect()
}

/// Statistical test for IC significance.
///
/// Under null hypothesis IC = 0, t-statistic = IC * sqrt(N) / sqrt(1 - IC^2).
/// Returns (t_statistic, is_significant_at_5pct).
pub fn ic_t_test(ic: f64, n_obs: usize) -> (f64, bool) {
    if !ic.is_finite() || n_obs < 3 {
        return (f64::NAN, false);
    }
    let ic_sq = ic * ic;
    let denom = ((1.0 - ic_sq.min(0.9999)) / (n_obs - 2) as f64).sqrt();
    if denom < 1e-12 {
        return (f64::INFINITY, true);
    }
    let t_stat = ic / denom;
    // Critical value at 5% two-tailed for large n ~= 1.96
    let significant = t_stat.abs() > 1.96;
    (t_stat, significant)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_panel(n_periods: usize, n_assets: usize, ic_strength: f64) -> (Array2<f64>, Array2<f64>) {
        // Create factor scores with predictive power (controlled by ic_strength)
        let factor = Array2::from_shape_fn((n_periods, n_assets), |(t, i)| {
            ((t as f64 * 0.1 + i as f64 * 0.2).sin())
        });

        // Returns = ic_strength * factor + noise
        let returns = Array2::from_shape_fn((n_periods, n_assets), |(t, i)| {
            let signal = ic_strength * ((t as f64 * 0.1 + i as f64 * 0.2).sin());
            let noise = 0.02 * ((t as f64 * 0.7 + i as f64 * 1.3).sin());
            signal + noise
        });

        (factor, returns)
    }

    #[test]
    fn test_compute_ic_positive() {
        let (factor, returns) = make_panel(100, 50, 0.5);
        let factor_t0: Vec<f64> = factor.row(0).to_vec();
        let return_t1: Vec<f64> = returns.row(1).to_vec();
        let ic = compute_ic(&factor_t0, &return_t1);
        assert!(ic.is_finite());
    }

    #[test]
    fn test_ic_by_lag() {
        let (factor, returns) = make_panel(100, 50, 0.3);
        let ic_vec = compute_ic_by_lag(&factor, &returns, 5).unwrap();
        assert_eq!(ic_vec.len(), 5);
    }

    #[test]
    fn test_fit_exponential_decay() {
        // Synthetic decaying IC: IC_t = 0.05 * exp(-0.1 * t)
        let ic_by_lag: Vec<f64> = (1..=20).map(|t| 0.05 * (-0.1 * t as f64).exp()).collect();
        let (ic0, lambda, r_sq) = fit_exponential_decay(&ic_by_lag).unwrap();
        assert!((ic0 - 0.05).abs() < 0.01);
        assert!((lambda - 0.1).abs() < 0.02);
        assert!(r_sq > 0.99);
    }

    #[test]
    fn test_icir() {
        let ics = vec![0.05, 0.04, 0.06, 0.05, 0.03, 0.07, 0.04, 0.05];
        let icir = compute_icir(&ics);
        assert!(icir > 0.0);
    }

    #[test]
    fn test_ic_t_test() {
        let (t, sig) = ic_t_test(0.05, 252);
        // IC of 0.05 with 252 obs: t = 0.05 * sqrt(250) / sqrt(1 - 0.0025)
        assert!(t > 0.0);
        // May or may not be significant depending on exact calc
        let _ = sig;
    }
}
