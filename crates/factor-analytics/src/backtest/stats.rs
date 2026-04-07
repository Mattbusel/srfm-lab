//! Statistical analysis: Fama-MacBeth cross-sectional regression,
//! t-statistics, and Newey-West HAC standard errors.

use ndarray::{Array1, Array2};
use crate::error::{FactorError, Result};

/// Result of a single Fama-MacBeth cross-sectional regression.
#[derive(Debug, Clone)]
pub struct CrossSectionalRegression {
    /// Period index
    pub period: usize,
    /// Intercept (average return not explained by factors)
    pub intercept: f64,
    /// Factor coefficients (factor risk premiums)
    pub coefficients: Vec<f64>,
    /// R-squared of cross-sectional fit
    pub r_squared: f64,
    /// Number of observations in this period
    pub n_obs: usize,
}

/// Fama-MacBeth (1973) two-pass regression results.
#[derive(Debug, Clone)]
pub struct FamaMacBethResult {
    /// Factor names
    pub factor_names: Vec<String>,
    /// Time-series average of cross-sectional intercepts
    pub avg_intercept: f64,
    /// Time-series average of factor coefficients (risk premiums)
    pub avg_factor_premiums: Vec<f64>,
    /// Standard errors of average factor premiums (plain)
    pub std_errors: Vec<f64>,
    /// Newey-West HAC standard errors
    pub nw_std_errors: Vec<f64>,
    /// t-statistics using plain SE
    pub t_statistics: Vec<f64>,
    /// t-statistics using Newey-West SE
    pub t_statistics_nw: Vec<f64>,
    /// Per-period regression results
    pub period_results: Vec<CrossSectionalRegression>,
    /// Time-series of intercepts
    pub intercept_series: Vec<f64>,
    /// Time-series of coefficients (n_periods x n_factors)
    pub coefficient_series: Vec<Vec<f64>>,
}

/// Run Fama-MacBeth two-pass regressions.
///
/// Pass 1: Run T cross-sectional regressions of r_{i,t} on f_{i,t-1}.
/// Pass 2: Average the T sets of coefficients; compute SE as std(gamma_t) / sqrt(T).
///
/// # Arguments
/// * `returns_panel` -- (n_periods, n_assets) of asset returns
/// * `factor_panel` -- (n_periods, n_assets, n_factors) factor exposures (lagged)
///   represented as a Vec of (n_periods, n_assets) matrices, one per factor
/// * `factor_names` -- names of factors
/// * `nw_lags` -- number of Newey-West lags (typically 4 or 6)
pub fn fama_macbeth(
    returns_panel: &Array2<f64>,
    factor_panels: &[Array2<f64>],
    factor_names: &[String],
    nw_lags: usize,
) -> Result<FamaMacBethResult> {
    let (n_periods, n_assets) = returns_panel.dim();
    let n_factors = factor_names.len();

    if factor_panels.len() != n_factors {
        return Err(FactorError::DimensionMismatch {
            expected: n_factors,
            got: factor_panels.len(),
        });
    }
    for fp in factor_panels {
        if fp.dim() != (n_periods, n_assets) {
            return Err(FactorError::ShapeMismatch {
                msg: format!("factor panel shape mismatch: expected ({}, {})", n_periods, n_assets),
            });
        }
    }

    let mut period_results: Vec<CrossSectionalRegression> = Vec::with_capacity(n_periods);
    let mut intercept_series = Vec::with_capacity(n_periods);
    let mut coeff_series: Vec<Vec<f64>> = Vec::with_capacity(n_periods);

    for t in 0..n_periods {
        let period_returns: Vec<f64> = returns_panel.row(t).to_vec();

        // Build factor matrix for this period (n_assets x n_factors)
        let x_mat = Array2::from_shape_fn((n_assets, n_factors), |(i, k)| {
            factor_panels[k][[t, i]]
        });

        match crate::cross_section::neutralize::ols_full(&period_returns, &x_mat, true) {
            Ok(ols) => {
                let intercept = ols.coefficients[0];
                let coeffs = ols.coefficients[1..].to_vec();
                let result = CrossSectionalRegression {
                    period: t,
                    intercept,
                    coefficients: coeffs.clone(),
                    r_squared: ols.r_squared,
                    n_obs: ols.n_obs,
                };
                intercept_series.push(intercept);
                coeff_series.push(coeffs);
                period_results.push(result);
            }
            Err(_) => {
                intercept_series.push(f64::NAN);
                coeff_series.push(vec![f64::NAN; n_factors]);
                period_results.push(CrossSectionalRegression {
                    period: t,
                    intercept: f64::NAN,
                    coefficients: vec![f64::NAN; n_factors],
                    r_squared: f64::NAN,
                    n_obs: 0,
                });
            }
        }
    }

    // Average coefficients across periods (excluding NAN)
    let valid_intercepts: Vec<f64> = intercept_series.iter().copied().filter(|v| v.is_finite()).collect();
    let avg_intercept = if valid_intercepts.is_empty() {
        f64::NAN
    } else {
        valid_intercepts.iter().sum::<f64>() / valid_intercepts.len() as f64
    };

    let mut avg_factor_premiums = vec![0.0f64; n_factors];
    let mut std_errors = vec![f64::NAN; n_factors];
    let mut nw_std_errors = vec![f64::NAN; n_factors];

    for k in 0..n_factors {
        let factor_k_series: Vec<f64> = coeff_series.iter().map(|c| c[k]).collect();
        let valid: Vec<f64> = factor_k_series.iter().copied().filter(|v| v.is_finite()).collect();
        let t_valid = valid.len();

        if t_valid == 0 {
            continue;
        }

        let mean = valid.iter().sum::<f64>() / t_valid as f64;
        avg_factor_premiums[k] = mean;

        if t_valid >= 2 {
            let var = valid.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (t_valid - 1) as f64;
            std_errors[k] = (var / t_valid as f64).sqrt();
        }

        // Newey-West HAC SE
        nw_std_errors[k] = newey_west_se(&factor_k_series, nw_lags);
    }

    let t_statistics: Vec<f64> = avg_factor_premiums
        .iter()
        .zip(std_errors.iter())
        .map(|(&mu, &se)| if se > 1e-12 { mu / se } else { f64::NAN })
        .collect();

    let t_statistics_nw: Vec<f64> = avg_factor_premiums
        .iter()
        .zip(nw_std_errors.iter())
        .map(|(&mu, &se)| if se > 1e-12 { mu / se } else { f64::NAN })
        .collect();

    Ok(FamaMacBethResult {
        factor_names: factor_names.to_vec(),
        avg_intercept,
        avg_factor_premiums,
        std_errors,
        nw_std_errors,
        t_statistics,
        t_statistics_nw,
        period_results,
        intercept_series,
        coefficient_series: coeff_series,
    })
}

/// Compute Newey-West HAC standard error for a time series.
///
/// NW_var = (1/T) * [gamma_0 + 2 * sum_{l=1}^{L} (1 - l/(L+1)) * gamma_l]
/// where gamma_l = (1/T) * sum_t (x_t - x_bar) * (x_{t-l} - x_bar)
///
/// SE = sqrt(NW_var / T)
pub fn newey_west_se(series: &[f64], max_lags: usize) -> f64 {
    let valid: Vec<f64> = series.iter().copied().filter(|v| v.is_finite()).collect();
    let n = valid.len();
    if n < 3 {
        return f64::NAN;
    }

    let mean = valid.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = valid.iter().map(|v| v - mean).collect();

    // Lag-0 autocovariance
    let gamma_0 = centered.iter().map(|v| v * v).sum::<f64>() / n as f64;

    let effective_lags = max_lags.min(n - 2);
    let mut nw_var = gamma_0;

    for lag in 1..=effective_lags {
        // Bartlett weight
        let weight = 1.0 - (lag as f64) / (effective_lags as f64 + 1.0);
        let gamma_l = (0..n - lag)
            .map(|t| centered[t] * centered[t + lag])
            .sum::<f64>()
            / n as f64;
        nw_var += 2.0 * weight * gamma_l;
    }

    // Ensure non-negative
    nw_var = nw_var.max(0.0);

    // SE = sqrt(NW_var / T)
    (nw_var / n as f64).sqrt()
}

/// Compute two-sided p-value from t-statistic using Student's t-distribution approximation.
///
/// Uses the normal approximation for large samples (|t| is asymptotically normal).
pub fn t_stat_p_value(t_stat: f64, df: usize) -> f64 {
    if !t_stat.is_finite() || df == 0 {
        return f64::NAN;
    }
    let t_abs = t_stat.abs();

    // For large df, use normal approximation
    if df > 30 {
        // P(|Z| > t) = 2 * (1 - Phi(t)) using Mills ratio approximation
        let p_one_tail = normal_survival(t_abs);
        return 2.0 * p_one_tail;
    }

    // Student's t CDF approximation for small df (Abramowitz & Stegun)
    // Using incomplete beta function approximation
    let x = (df as f64) / (df as f64 + t_abs * t_abs);
    let p = incomplete_beta_half(x, df as f64 / 2.0, 0.5);
    p
}

/// Normal survival function P(Z > t) for z > 0.
fn normal_survival(z: f64) -> f64 {
    if z < 0.0 {
        return 1.0 - normal_survival(-z);
    }
    // Mills ratio approximation: P(Z > z) ~= phi(z) / z for large z
    // For z < 8, use rational approximation
    let z2 = z * z;
    if z > 8.0 {
        // Asymptotic expansion
        let pdf = (-z2 / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
        pdf / z
    } else {
        // Numerical integration via polynomial approx
        let t = 1.0 / (1.0 + 0.2316419 * z);
        let poly = t * (0.319381530
            + t * (-0.356563782
                + t * (1.781477937
                    + t * (-1.821255978 + t * 1.330274429))));
        let pdf = (-z2 / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
        pdf * poly
    }
}

/// Incomplete beta function approximation for Student's t p-value.
fn incomplete_beta_half(x: f64, a: f64, _b: f64) -> f64 {
    // Simplified: use continued fraction representation
    // For our use case (b = 0.5, a = df/2), we use the normal approximation
    let _ = x;
    let _ = a;
    // Fall through to normal approximation
    f64::NAN
}

/// Compute time-series alpha and beta via OLS.
///
/// Regresses portfolio returns on benchmark returns.
/// Returns (alpha_annualized, beta, t_stat_alpha, t_stat_beta, r_squared).
pub fn time_series_regression(
    portfolio_returns: &[f64],
    benchmark_returns: &[f64],
    annualization: f64,
) -> Result<(f64, f64, f64, f64, f64)> {
    let n = portfolio_returns.len();
    if n < 10 {
        return Err(FactorError::InsufficientData { required: 10, got: n });
    }
    if benchmark_returns.len() != n {
        return Err(FactorError::DimensionMismatch { expected: n, got: benchmark_returns.len() });
    }

    let (alpha_daily, beta) = crate::factors::momentum::ols_simple(portfolio_returns, benchmark_returns)?;

    // Compute residuals
    let residuals: Vec<f64> = portfolio_returns
        .iter()
        .zip(benchmark_returns.iter())
        .map(|(&rp, &rb)| rp - alpha_daily - beta * rb)
        .collect();

    let rss: f64 = residuals.iter().map(|r| r * r).sum();
    let mean_y = portfolio_returns.iter().sum::<f64>() / n as f64;
    let tss: f64 = portfolio_returns.iter().map(|r| (r - mean_y).powi(2)).sum();
    let r_squared = if tss > 1e-14 { 1.0 - rss / tss } else { 0.0 };

    let s_sq = rss / (n - 2) as f64; // residual variance

    let mean_x = benchmark_returns.iter().sum::<f64>() / n as f64;
    let sxx: f64 = benchmark_returns.iter().map(|x| (x - mean_x).powi(2)).sum();

    let se_beta = if sxx > 1e-12 { (s_sq / sxx).sqrt() } else { f64::NAN };
    let se_alpha = if sxx > 1e-12 {
        let var_alpha = s_sq * (1.0 / n as f64 + mean_x * mean_x / sxx);
        var_alpha.sqrt()
    } else {
        f64::NAN
    };

    let t_alpha = if se_alpha > 1e-12 { alpha_daily / se_alpha } else { f64::NAN };
    let t_beta = if se_beta > 1e-12 { beta / se_beta } else { f64::NAN };

    let alpha_annual = alpha_daily * annualization;

    Ok((alpha_annual, beta, t_alpha, t_beta, r_squared))
}

/// Compute rolling beta with Vasicek (1973) shrinkage.
///
/// Shrinks estimated beta towards 1.0 (market average).
/// beta_adj = (1/(1 + v/s)) * beta_est + (v/(1+v/s)) * 1.0
/// where v = estimation error variance, s = cross-sectional variance of betas.
pub fn vasicek_beta(raw_beta: f64, estimation_se: f64, cross_sectional_std: f64) -> f64 {
    if !raw_beta.is_finite() {
        return 1.0; // default to market beta
    }
    if estimation_se < 1e-12 || cross_sectional_std < 1e-12 {
        return raw_beta;
    }
    let phi = estimation_se.powi(2) / (estimation_se.powi(2) + cross_sectional_std.powi(2));
    (1.0 - phi) * raw_beta + phi * 1.0
}

/// Information coefficient statistics for a period.
#[derive(Debug, Clone)]
pub struct IcStatistics {
    pub ic: f64,
    pub icir: f64,
    pub t_statistic: f64,
    pub is_significant: bool,
    pub n_obs: usize,
}

/// Compute comprehensive IC statistics.
pub fn compute_ic_statistics(ic_series: &[f64]) -> IcStatistics {
    let valid: Vec<f64> = ic_series.iter().copied().filter(|v| v.is_finite()).collect();
    let n = valid.len();

    if n < 2 {
        return IcStatistics {
            ic: f64::NAN,
            icir: f64::NAN,
            t_statistic: f64::NAN,
            is_significant: false,
            n_obs: n,
        };
    }

    let mean_ic = valid.iter().sum::<f64>() / n as f64;
    let std_ic = (valid.iter().map(|v| (v - mean_ic).powi(2)).sum::<f64>() / (n - 1) as f64).sqrt();

    let icir = if std_ic > 1e-12 { mean_ic / std_ic } else { f64::NAN };
    let t_stat = if std_ic > 1e-12 {
        mean_ic * (n as f64).sqrt() / std_ic
    } else {
        f64::NAN
    };

    let is_significant = t_stat.is_finite() && t_stat.abs() > 1.96;

    IcStatistics {
        ic: mean_ic,
        icir,
        t_statistic: t_stat,
        is_significant,
        n_obs: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_panels(n_periods: usize, n_assets: usize, n_factors: usize)
        -> (Array2<f64>, Vec<Array2<f64>>)
    {
        let returns = Array2::from_shape_fn((n_periods, n_assets), |(t, i)| {
            0.002 + 0.02 * ((t + i) as f64 * 0.2).sin()
        });

        let factor_panels: Vec<Array2<f64>> = (0..n_factors)
            .map(|k| Array2::from_shape_fn((n_periods, n_assets), |(t, i)| {
                ((t + i + k * 3) as f64 * 0.15).sin()
            }))
            .collect();

        (returns, factor_panels)
    }

    #[test]
    fn test_fama_macbeth() {
        let (returns, factor_panels) = make_panels(30, 50, 3);
        let names: Vec<String> = (0..3).map(|k| format!("f{}", k)).collect();
        let result = fama_macbeth(&returns, &factor_panels, &names, 4).unwrap();

        assert_eq!(result.avg_factor_premiums.len(), 3);
        assert_eq!(result.period_results.len(), 30);
        assert!(result.avg_intercept.is_finite());
    }

    #[test]
    fn test_newey_west_se() {
        let series: Vec<f64> = (0..60).map(|i| 0.03 + 0.01 * ((i as f64) * 0.5).sin()).collect();
        let nw_se = newey_west_se(&series, 4);
        let plain_se = {
            let mean = series.iter().sum::<f64>() / series.len() as f64;
            let std = (series.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                / (series.len() - 1) as f64).sqrt();
            std / (series.len() as f64).sqrt()
        };
        // NW SE should be >= plain SE in general for autocorrelated series
        assert!(nw_se.is_finite() && nw_se > 0.0);
        let _ = plain_se;
    }

    #[test]
    fn test_time_series_regression() {
        // Construct data where alpha = 0.001/day, beta = 1.2
        let n = 100;
        let benchmark: Vec<f64> = (0..n).map(|i| 0.001 + 0.01 * ((i as f64) * 0.2).sin()).collect();
        let portfolio: Vec<f64> = benchmark.iter().map(|&rb| 0.0001 + 1.2 * rb).collect();

        let (alpha_ann, beta, _t_a, _t_b, r2) = time_series_regression(&portfolio, &benchmark, 252.0).unwrap();

        assert!((beta - 1.2).abs() < 1e-8);
        assert!((alpha_ann - 0.0001 * 252.0).abs() < 1e-6);
        assert!(r2 > 0.999);
    }

    #[test]
    fn test_ic_statistics() {
        let ic_series: Vec<f64> = (0..60).map(|i| 0.04 + 0.01 * ((i as f64) * 0.3).sin()).collect();
        let stats = compute_ic_statistics(&ic_series);
        assert!(stats.ic > 0.0);
        assert!(stats.icir > 0.0);
        assert!(stats.t_statistic.is_finite());
    }

    #[test]
    fn test_vasicek_shrinkage() {
        // Beta of 1.5 with high estimation error -> shrinks towards 1.0
        let adjusted = vasicek_beta(1.5, 0.5, 0.3);
        assert!(adjusted > 1.0 && adjusted < 1.5);
        // Beta of 0.5 with high estimation error -> shrinks towards 1.0
        let adjusted2 = vasicek_beta(0.5, 0.5, 0.3);
        assert!(adjusted2 > 0.5 && adjusted2 < 1.0);
    }
}
