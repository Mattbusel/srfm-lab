//! Factor stability analysis: rank autocorrelation, factor turnover,
//! and information coefficient persistence.

use ndarray::{Array1, Array2};
use crate::error::{FactorError, Result};
use crate::cross_section::normalize::rank_normalize;
use crate::decay::ic_decay::{compute_ic, compute_icir};

/// Factor stability report.
#[derive(Debug, Clone)]
pub struct FactorStabilityReport {
    /// Rank autocorrelation at lag 1 (Spearman: ranks_t vs ranks_{t+1})
    pub rank_autocorr_lag1: f64,
    /// Rank autocorrelation at lag 5 (weekly)
    pub rank_autocorr_lag5: f64,
    /// Rank autocorrelation at lag 21 (monthly)
    pub rank_autocorr_lag21: f64,
    /// Average monthly factor turnover (fraction of portfolio that changes)
    pub avg_monthly_turnover: f64,
    /// IC persistence (autocorrelation of IC time series)
    pub ic_persistence: f64,
    /// ICIR (IC Information Ratio)
    pub icir: f64,
    /// Fraction of periods with positive IC
    pub pct_positive_ic: f64,
    /// Maximum consecutive periods with negative IC
    pub max_consecutive_neg_ic: usize,
}

/// Compute rank autocorrelation of a factor at a given lag.
///
/// Uses Spearman rank correlation between factor at t and factor at t+lag.
pub fn rank_autocorrelation(factor_panel: &Array2<f64>, lag: usize) -> f64 {
    let (n_periods, _n_assets) = factor_panel.dim();
    if n_periods <= lag {
        return f64::NAN;
    }

    let mut corr_sum = 0.0;
    let mut count = 0;

    for t in 0..(n_periods - lag) {
        let factors_t: Vec<f64> = factor_panel.row(t).to_vec();
        let factors_t_lag: Vec<f64> = factor_panel.row(t + lag).to_vec();
        let ic = compute_ic(&factors_t, &factors_t_lag);
        if ic.is_finite() {
            corr_sum += ic;
            count += 1;
        }
    }

    if count > 0 {
        corr_sum / count as f64
    } else {
        f64::NAN
    }
}

/// Compute factor turnover at a given lag.
///
/// Turnover = 1 - rank_autocorrelation (conceptually).
/// More precisely: fraction of the long quintile that changes from t to t+lag.
///
/// Uses the top-quintile membership overlap approach.
pub fn factor_turnover(factor_panel: &Array2<f64>, lag: usize) -> f64 {
    let (n_periods, n_assets) = factor_panel.dim();
    if n_periods <= lag || n_assets < 5 {
        return f64::NAN;
    }

    let quintile_size = n_assets / 5;
    if quintile_size == 0 {
        return f64::NAN;
    }

    let mut turnover_sum = 0.0;
    let mut count = 0;

    for t in 0..(n_periods - lag) {
        let factors_t: Vec<f64> = factor_panel.row(t).to_vec();
        let factors_t_lag: Vec<f64> = factor_panel.row(t + lag).to_vec();

        let top_quintile_t: std::collections::HashSet<usize> =
            top_k_indices(&factors_t, quintile_size).into_iter().collect();
        let top_quintile_lag: std::collections::HashSet<usize> =
            top_k_indices(&factors_t_lag, quintile_size).into_iter().collect();

        let overlap = top_quintile_t.intersection(&top_quintile_lag).count();
        let turnover = 1.0 - (overlap as f64 / quintile_size as f64);

        turnover_sum += turnover;
        count += 1;
    }

    if count > 0 {
        turnover_sum / count as f64
    } else {
        f64::NAN
    }
}

/// Get indices of top k finite values (highest scores).
fn top_k_indices(values: &[f64], k: usize) -> Vec<usize> {
    let mut indexed: Vec<(usize, f64)> = values
        .iter()
        .enumerate()
        .filter_map(|(i, &v)| if v.is_finite() { Some((i, v)) } else { None })
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap()); // descending
    indexed.iter().take(k).map(|&(i, _)| i).collect()
}

/// Compute IC time series.
///
/// For each period t, computes IC between factor at t and returns at t+1.
///
/// # Arguments
/// * `factor_panel` -- (n_periods, n_assets)
/// * `returns_panel` -- (n_periods, n_assets)
///
/// Returns Vec of length (n_periods - 1).
pub fn compute_ic_time_series(
    factor_panel: &Array2<f64>,
    returns_panel: &Array2<f64>,
) -> Result<Vec<f64>> {
    let (n_periods, n_assets) = factor_panel.dim();
    if returns_panel.dim() != (n_periods, n_assets) {
        return Err(FactorError::ShapeMismatch {
            msg: "factor_panel and returns_panel must have same shape".into(),
        });
    }
    if n_periods < 2 {
        return Err(FactorError::InsufficientData { required: 2, got: n_periods });
    }

    let ic_series: Vec<f64> = (0..n_periods - 1)
        .map(|t| {
            let factor_t: Vec<f64> = factor_panel.row(t).to_vec();
            let return_t1: Vec<f64> = returns_panel.row(t + 1).to_vec();
            compute_ic(&factor_t, &return_t1)
        })
        .collect();

    Ok(ic_series)
}

/// Compute IC persistence -- Spearman autocorrelation of the IC time series.
///
/// High persistence means IC regimes cluster (both good and bad).
pub fn ic_persistence(ic_series: &[f64]) -> f64 {
    let n = ic_series.len();
    if n < 4 {
        return f64::NAN;
    }

    let t: Vec<f64> = ic_series[..n - 1].to_vec();
    let t1: Vec<f64> = ic_series[1..].to_vec();

    // Spearman correlation via ranks
    let rt = rank_normalize(&t);
    let rt1 = rank_normalize(&t1);

    let pairs: Vec<(f64, f64)> = rt
        .iter()
        .zip(rt1.iter())
        .filter_map(|(&a, &b)| if a.is_finite() && b.is_finite() { Some((a, b)) } else { None })
        .collect();

    let m = pairs.len();
    if m < 3 {
        return f64::NAN;
    }

    let ma = pairs.iter().map(|p| p.0).sum::<f64>() / m as f64;
    let mb = pairs.iter().map(|p| p.1).sum::<f64>() / m as f64;
    let cov = pairs.iter().map(|p| (p.0 - ma) * (p.1 - mb)).sum::<f64>();
    let sa = pairs.iter().map(|p| (p.0 - ma).powi(2)).sum::<f64>().sqrt();
    let sb = pairs.iter().map(|p| (p.1 - mb).powi(2)).sum::<f64>().sqrt();

    if sa < 1e-12 || sb < 1e-12 {
        return 0.0;
    }

    cov / (sa * sb)
}

/// Compute the maximum consecutive run of negative ICs.
fn max_consecutive_negative(ic_series: &[f64]) -> usize {
    let mut max_run = 0usize;
    let mut current_run = 0usize;

    for &ic in ic_series {
        if ic.is_finite() && ic < 0.0 {
            current_run += 1;
            if current_run > max_run {
                max_run = current_run;
            }
        } else if ic.is_finite() {
            current_run = 0;
        }
    }
    max_run
}

/// Compute the full factor stability report.
///
/// # Arguments
/// * `factor_panel` -- (n_periods, n_assets) factor scores
/// * `returns_panel` -- (n_periods, n_assets) returns
pub fn compute_stability_report(
    factor_panel: &Array2<f64>,
    returns_panel: &Array2<f64>,
) -> Result<FactorStabilityReport> {
    let (n_periods, _) = factor_panel.dim();
    if n_periods < 25 {
        return Err(FactorError::InsufficientData { required: 25, got: n_periods });
    }

    let rank_autocorr_lag1 = rank_autocorrelation(factor_panel, 1);
    let rank_autocorr_lag5 = rank_autocorrelation(factor_panel, 5);
    let rank_autocorr_lag21 = rank_autocorrelation(factor_panel, 21.min(n_periods / 2));

    let avg_monthly_turnover = factor_turnover(factor_panel, 21.min(n_periods / 2));

    let ic_series = compute_ic_time_series(factor_panel, returns_panel)?;

    let icir_val = compute_icir(&ic_series);

    let ic_persistence_val = ic_persistence(&ic_series);

    let n_ic = ic_series.iter().filter(|v| v.is_finite()).count();
    let n_pos = ic_series.iter().filter(|v| v.is_finite() && **v > 0.0).count();
    let pct_positive = if n_ic > 0 { n_pos as f64 / n_ic as f64 } else { f64::NAN };

    let max_neg_run = max_consecutive_negative(&ic_series);

    Ok(FactorStabilityReport {
        rank_autocorr_lag1,
        rank_autocorr_lag5,
        rank_autocorr_lag21,
        avg_monthly_turnover,
        ic_persistence: ic_persistence_val,
        icir: icir_val,
        pct_positive_ic: pct_positive,
        max_consecutive_neg_ic: max_neg_run,
    })
}

/// Compute factor decay summary: at what lag does IC fall to 50% of lag-1 IC?
///
/// Searches the IC decay curve for the half-life point.
pub fn empirical_half_life(ic_by_lag: &[f64]) -> f64 {
    if ic_by_lag.is_empty() {
        return f64::NAN;
    }

    let ic_1 = ic_by_lag[0];
    if !ic_1.is_finite() || ic_1 <= 0.0 {
        return f64::NAN;
    }

    let half_ic = ic_1 / 2.0;

    // Find first lag where IC drops below half
    for (i, &ic) in ic_by_lag[1..].iter().enumerate() {
        if ic.is_finite() && ic <= half_ic {
            // Linear interpolation
            let prev = ic_by_lag[i];
            let curr = ic;
            if prev > curr {
                let frac = (prev - half_ic) / (prev - curr);
                return (i as f64) + frac;
            }
            return (i + 1) as f64;
        }
    }

    // IC never decays to half -- return length as lower bound
    ic_by_lag.len() as f64
}

/// Rolling IC statistics for regime detection.
///
/// Computes mean and std of IC over a rolling window.
pub fn rolling_ic_stats(ic_series: &[f64], window: usize) -> Vec<(f64, f64)> {
    let n = ic_series.len();
    if n < window {
        return Vec::new();
    }

    (0..=n - window)
        .map(|i| {
            let window_slice = &ic_series[i..i + window];
            let valid: Vec<f64> = window_slice.iter().copied().filter(|v| v.is_finite()).collect();
            if valid.len() < 2 {
                return (f64::NAN, f64::NAN);
            }
            let mean = valid.iter().sum::<f64>() / valid.len() as f64;
            let std = (valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / (valid.len() - 1) as f64)
                .sqrt();
            (mean, std)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_stable_factor(n_periods: usize, n_assets: usize) -> (Array2<f64>, Array2<f64>) {
        // Factor with high rank autocorrelation (slow-moving)
        let factor = Array2::from_shape_fn((n_periods, n_assets), |(t, i)| {
            // Slow drift -- high autocorrelation
            (i as f64) + 0.01 * t as f64 + 0.1 * ((i as f64 * 0.5).sin())
        });

        let returns = Array2::from_shape_fn((n_periods, n_assets), |(t, i)| {
            0.001 * (i as f64) + 0.01 * ((t as f64 + i as f64) * 0.2).sin()
        });

        (factor, returns)
    }

    #[test]
    fn test_rank_autocorrelation() {
        let (factor, _) = make_stable_factor(50, 30);
        let autocorr = rank_autocorrelation(&factor, 1);
        assert!(autocorr.is_finite());
        // Stable factor should have high autocorrelation
        assert!(autocorr > 0.8, "expected high autocorr, got {}", autocorr);
    }

    #[test]
    fn test_factor_turnover() {
        let (factor, _) = make_stable_factor(50, 30);
        let turnover = factor_turnover(&factor, 1);
        assert!(turnover.is_finite());
        assert!(turnover >= 0.0 && turnover <= 1.0);
    }

    #[test]
    fn test_ic_time_series() {
        let (factor, returns) = make_stable_factor(50, 30);
        let ic_series = compute_ic_time_series(&factor, &returns).unwrap();
        assert_eq!(ic_series.len(), 49);
    }

    #[test]
    fn test_stability_report() {
        let (factor, returns) = make_stable_factor(60, 30);
        let report = compute_stability_report(&factor, &returns).unwrap();
        assert!(report.rank_autocorr_lag1.is_finite());
        assert!(report.pct_positive_ic >= 0.0 && report.pct_positive_ic <= 1.0);
    }

    #[test]
    fn test_empirical_half_life() {
        // Synthetic: IC decays from 0.05 to near 0 by lag 10
        let ic_by_lag: Vec<f64> = (1..=20).map(|t| 0.05 * (-0.15 * t as f64).exp()).collect();
        let hl = empirical_half_life(&ic_by_lag);
        // True half-life: ln(2)/0.15 ~= 4.6
        assert!(hl > 3.0 && hl < 8.0, "half-life = {}", hl);
    }

    #[test]
    fn test_rolling_ic_stats() {
        let ic_series: Vec<f64> = (0..60).map(|i| 0.03 + 0.02 * ((i as f64) * 0.3).sin()).collect();
        let stats = rolling_ic_stats(&ic_series, 12);
        assert_eq!(stats.len(), 49);
    }
}
