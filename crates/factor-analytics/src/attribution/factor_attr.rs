//! Factor return decomposition.
//!
//! Regresses portfolio returns on factor exposures to compute:
//! * Factor contributions to return
//! * Residual alpha (unexplained by factors)
//! * Factor betas and t-statistics

use ndarray::{Array1, Array2};
use crate::error::{FactorError, Result};
use crate::cross_section::neutralize::{ols_full, OlsResult};

/// Factor exposure for a single asset at a point in time.
#[derive(Debug, Clone)]
pub struct FactorExposure {
    /// Asset identifier
    pub asset_id: String,
    /// Portfolio weight
    pub weight: f64,
    /// Factor exposures (one per factor)
    pub exposures: Vec<f64>,
}

/// Factor attribution result.
#[derive(Debug, Clone)]
pub struct FactorAttributionResult {
    /// Factor names
    pub factor_names: Vec<String>,
    /// Estimated factor returns (coefficients from cross-sectional regression)
    pub factor_returns: Vec<f64>,
    /// Portfolio factor exposures (weighted average of asset exposures)
    pub portfolio_exposures: Vec<f64>,
    /// Factor contributions to portfolio return: exposure * factor_return
    pub factor_contributions: Vec<f64>,
    /// Residual alpha (unexplained return)
    pub alpha: f64,
    /// R-squared of the cross-sectional regression
    pub r_squared: f64,
    /// Total explained return (sum of factor contributions)
    pub total_explained: f64,
    /// Portfolio total return
    pub total_return: f64,
}

/// Compute factor attribution via cross-sectional regression.
///
/// For each time period, regress asset returns on factor exposures to estimate
/// factor returns, then decompose portfolio return by factor.
///
/// # Arguments
/// * `asset_returns` -- return of each asset
/// * `factor_exposures` -- vec of per-asset exposures, one Vec<f64> per asset
/// * `portfolio_weights` -- weight of each asset in portfolio
/// * `factor_names` -- name of each factor
pub fn compute_factor_attribution(
    asset_returns: &[f64],
    factor_exposures: &[Vec<f64>],
    portfolio_weights: &[f64],
    factor_names: &[String],
) -> Result<FactorAttributionResult> {
    let n_assets = asset_returns.len();
    let n_factors = factor_names.len();

    if factor_exposures.len() != n_assets {
        return Err(FactorError::DimensionMismatch {
            expected: n_assets,
            got: factor_exposures.len(),
        });
    }
    if portfolio_weights.len() != n_assets {
        return Err(FactorError::DimensionMismatch {
            expected: n_assets,
            got: portfolio_weights.len(),
        });
    }

    // Validate all exposure vectors have the same length
    for (i, exp) in factor_exposures.iter().enumerate() {
        if exp.len() != n_factors {
            return Err(FactorError::DimensionMismatch {
                expected: n_factors,
                got: exp.len(),
            });
        }
    }

    // Build the factor exposure matrix (n_assets x n_factors)
    let x_matrix = Array2::from_shape_fn((n_assets, n_factors), |(i, k)| factor_exposures[i][k]);

    // Run OLS: returns = intercept + factor_exposures * factor_returns + residual
    let ols = ols_full(asset_returns, &x_matrix, true)?;

    // OLS coefficients: [intercept, factor_return_1, ..., factor_return_k]
    let intercept = ols.coefficients[0];
    let factor_returns: Vec<f64> = ols.coefficients[1..].to_vec();

    // Compute portfolio factor exposures (weighted average)
    let total_weight: f64 = portfolio_weights.iter().sum();
    let norm_weights: Vec<f64> = portfolio_weights.iter().map(|w| w / total_weight).collect();

    let mut portfolio_exposures = vec![0.0f64; n_factors];
    for (i, exp) in factor_exposures.iter().enumerate() {
        for k in 0..n_factors {
            portfolio_exposures[k] += norm_weights[i] * exp[k];
        }
    }

    // Factor contributions to portfolio return
    let factor_contributions: Vec<f64> = portfolio_exposures
        .iter()
        .zip(factor_returns.iter())
        .map(|(e, r)| e * r)
        .collect();

    let total_explained: f64 = factor_contributions.iter().sum();

    // Portfolio total return
    let total_return: f64 = asset_returns
        .iter()
        .zip(norm_weights.iter())
        .map(|(r, w)| r * w)
        .sum();

    // Alpha = intercept contribution (weighted by portfolio)
    let alpha = total_return - total_explained;

    Ok(FactorAttributionResult {
        factor_names: factor_names.to_vec(),
        factor_returns,
        portfolio_exposures,
        factor_contributions,
        alpha,
        r_squared: ols.r_squared,
        total_explained,
        total_return,
    })
}

/// Time-series factor attribution.
///
/// For each period, runs cross-sectional regression to extract factor returns,
/// then computes cumulative factor contributions.
///
/// # Arguments
/// * `returns_panel` -- (n_periods, n_assets) matrix of asset returns
/// * `exposure_panels` -- vec of (n_periods, n_assets) factor exposure matrices
/// * `weights_panel` -- (n_periods, n_assets) portfolio weights
/// * `factor_names` -- names of factors
///
/// Returns per-period factor attribution results.
pub fn time_series_factor_attribution(
    returns_panel: &Array2<f64>,
    exposure_panels: &[Array2<f64>],
    weights_panel: &Array2<f64>,
    factor_names: &[String],
) -> Result<Vec<FactorAttributionResult>> {
    let (n_periods, n_assets) = returns_panel.dim();
    let n_factors = factor_names.len();

    if exposure_panels.len() != n_factors {
        return Err(FactorError::DimensionMismatch {
            expected: n_factors,
            got: exposure_panels.len(),
        });
    }
    if weights_panel.dim() != (n_periods, n_assets) {
        return Err(FactorError::ShapeMismatch {
            msg: "weights_panel shape mismatch".into(),
        });
    }

    let mut results = Vec::with_capacity(n_periods);

    for t in 0..n_periods {
        let asset_returns: Vec<f64> = returns_panel.row(t).to_vec();
        let portfolio_weights: Vec<f64> = weights_panel.row(t).to_vec();

        // Extract per-asset factor exposures at time t
        let factor_exposures: Vec<Vec<f64>> = (0..n_assets)
            .map(|i| {
                (0..n_factors)
                    .map(|k| exposure_panels[k][[t, i]])
                    .collect()
            })
            .collect();

        match compute_factor_attribution(
            &asset_returns,
            &factor_exposures,
            &portfolio_weights,
            factor_names,
        ) {
            Ok(result) => results.push(result),
            Err(e) => {
                // Push a NAN result for this period
                results.push(FactorAttributionResult {
                    factor_names: factor_names.to_vec(),
                    factor_returns: vec![f64::NAN; n_factors],
                    portfolio_exposures: vec![f64::NAN; n_factors],
                    factor_contributions: vec![f64::NAN; n_factors],
                    alpha: f64::NAN,
                    r_squared: f64::NAN,
                    total_explained: f64::NAN,
                    total_return: f64::NAN,
                });
            }
        }
    }

    Ok(results)
}

/// Compute cumulative factor contributions across periods.
///
/// Sums factor contributions across all periods to get total attribution.
pub fn cumulative_factor_attribution(
    period_results: &[FactorAttributionResult],
) -> Result<FactorAttributionResult> {
    if period_results.is_empty() {
        return Err(FactorError::InsufficientData { required: 1, got: 0 });
    }

    let n_factors = period_results[0].factor_names.len();
    let factor_names = period_results[0].factor_names.clone();

    let mut total_factor_returns = vec![0.0f64; n_factors];
    let mut total_contributions = vec![0.0f64; n_factors];
    let mut total_alpha = 0.0f64;
    let mut total_return = 0.0f64;
    let mut r_sq_sum = 0.0f64;
    let mut valid_count = 0;

    for result in period_results {
        if result.total_return.is_finite() {
            for k in 0..n_factors {
                if result.factor_returns[k].is_finite() {
                    total_factor_returns[k] += result.factor_returns[k];
                }
                if result.factor_contributions[k].is_finite() {
                    total_contributions[k] += result.factor_contributions[k];
                }
            }
            if result.alpha.is_finite() {
                total_alpha += result.alpha;
            }
            total_return += result.total_return;
            if result.r_squared.is_finite() {
                r_sq_sum += result.r_squared;
                valid_count += 1;
            }
        }
    }

    let total_explained: f64 = total_contributions.iter().sum();
    let avg_r_sq = if valid_count > 0 { r_sq_sum / valid_count as f64 } else { f64::NAN };

    // Use zero avg exposures in cumulative (not meaningful to average exposures)
    let portfolio_exposures = vec![f64::NAN; n_factors];

    Ok(FactorAttributionResult {
        factor_names,
        factor_returns: total_factor_returns,
        portfolio_exposures,
        factor_contributions: total_contributions,
        alpha: total_alpha,
        r_squared: avg_r_sq,
        total_explained,
        total_return,
    })
}

/// Compute Barra-style factor return covariance matrix.
///
/// Uses the time series of factor returns from cross-sectional regressions.
///
/// Returns (factor_cov, specific_variance_per_asset).
pub fn compute_factor_covariance(
    period_results: &[FactorAttributionResult],
) -> Result<Array2<f64>> {
    if period_results.len() < 5 {
        return Err(FactorError::InsufficientData {
            required: 5,
            got: period_results.len(),
        });
    }

    let n_factors = period_results[0].factor_names.len();
    let n_periods = period_results.len();

    // Collect factor return time series
    let mut factor_return_matrix = Array2::<f64>::from_elem((n_periods, n_factors), f64::NAN);
    for (t, result) in period_results.iter().enumerate() {
        for k in 0..n_factors {
            factor_return_matrix[[t, k]] = result.factor_returns[k];
        }
    }

    // Compute covariance using composite module
    crate::cross_section::composite::sample_covariance(&factor_return_matrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_attribution_inputs(n: usize, n_factors: usize) -> (Vec<f64>, Vec<Vec<f64>>, Vec<f64>, Vec<String>) {
        let returns: Vec<f64> = (0..n).map(|i| 0.001 * i as f64 - 0.005).collect();
        // Use varied offsets to avoid near-singular design matrix
        let exposures: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..n_factors)
                    .map(|k| ((i as f64 * (0.1 + 0.3 * k as f64) + k as f64 * 1.7).sin()))
                    .collect()
            })
            .collect();
        let weights: Vec<f64> = vec![1.0 / n as f64; n];
        let names: Vec<String> = (0..n_factors).map(|k| format!("factor_{}", k)).collect();
        (returns, exposures, weights, names)
    }

    #[test]
    fn test_factor_attribution_basic() {
        let (returns, exposures, weights, names) = make_attribution_inputs(50, 3);
        let result = compute_factor_attribution(&returns, &exposures, &weights, &names).unwrap();

        assert_eq!(result.factor_returns.len(), 3);
        assert_eq!(result.factor_contributions.len(), 3);

        // Total return = factor contributions + alpha
        let reconstructed = result.total_explained + result.alpha;
        assert!(
            (reconstructed - result.total_return).abs() < 1e-10,
            "reconstructed={}, total_return={}",
            reconstructed,
            result.total_return
        );
    }

    #[test]
    fn test_factor_attribution_perfect_fit() {
        let n = 30;
        let n_factors = 2;
        // True factor returns: factor_0 = 0.01, factor_1 = -0.005
        let true_f = [0.01, -0.005];
        let exposures: Vec<Vec<f64>> = (0..n)
            .map(|i| vec![(i as f64 * 0.1).sin(), (i as f64 * 0.2).cos()])
            .collect();
        let returns: Vec<f64> = (0..n)
            .map(|i| exposures[i][0] * true_f[0] + exposures[i][1] * true_f[1])
            .collect();
        let weights = vec![1.0 / n as f64; n];
        let names: Vec<String> = vec!["f0".into(), "f1".into()];

        let result = compute_factor_attribution(&returns, &exposures, &weights, &names).unwrap();
        // With perfect fit, factor returns should match true values
        assert!((result.factor_returns[0] - true_f[0]).abs() < 1e-6);
        assert!((result.factor_returns[1] - true_f[1]).abs() < 1e-6);
        assert!(result.r_squared > 0.999);
    }

    #[test]
    fn test_cumulative_attribution() {
        let n_periods = 10;
        let (returns, exposures, weights, names) = make_attribution_inputs(30, 3);
        let period_results: Vec<FactorAttributionResult> = (0..n_periods)
            .map(|_| compute_factor_attribution(&returns, &exposures, &weights, &names).unwrap())
            .collect();

        let cumulative = cumulative_factor_attribution(&period_results).unwrap();
        assert_eq!(cumulative.factor_names.len(), 3);
        let sum_period_returns: f64 = period_results.iter().map(|r| r.total_return).sum();
        assert!((cumulative.total_return - sum_period_returns).abs() < 1e-10);
    }

    #[test]
    fn test_factor_covariance() {
        let (_base_returns, exposures, weights, names) = make_attribution_inputs(50, 4);
        let period_results: Vec<FactorAttributionResult> = (0..20)
            .map(|i| {
                // Vary returns each period to get non-degenerate time series
                let r: Vec<f64> = (0..50)
                    .map(|j| 0.001 * j as f64 * (1.0 + 0.1 * i as f64).recip() - 0.005)
                    .collect();
                compute_factor_attribution(&r, &exposures, &weights, &names).unwrap()
            })
            .collect();

        let cov = compute_factor_covariance(&period_results).unwrap();
        assert_eq!(cov.dim(), (4, 4));
        // Diagonal should be non-negative
        for k in 0..4 { assert!(cov[[k, k]] >= 0.0); }
    }
}
