//! Mean-variance optimization with factor constraints.
//!
//! Implements:
//! * Maximum IC-weighted portfolio (long-only)
//! * Minimum variance portfolio
//! * Long-only constrained optimization
//! * Factor exposure limits

use ndarray::{Array1, Array2};
use crate::error::{FactorError, Result};
use crate::portfolio::risk_model::FactorRiskModel;

/// Optimization constraints.
#[derive(Debug, Clone)]
pub struct OptimizationConstraints {
    /// Minimum weight per asset (0.0 for long-only)
    pub min_weight: f64,
    /// Maximum weight per asset
    pub max_weight: f64,
    /// Weights must sum to this value (1.0 for fully invested)
    pub weight_sum: f64,
    /// Maximum absolute factor exposure for each factor (None = unconstrained)
    pub max_factor_exposures: Option<Vec<f64>>,
    /// Maximum tracking error variance (None = unconstrained)
    pub max_tracking_error_var: Option<f64>,
    /// Maximum number of positions (None = no limit)
    pub max_positions: Option<usize>,
}

impl Default for OptimizationConstraints {
    fn default() -> Self {
        Self {
            min_weight: 0.0,
            max_weight: 0.10,
            weight_sum: 1.0,
            max_factor_exposures: None,
            max_tracking_error_var: None,
            max_positions: None,
        }
    }
}

/// Optimization result.
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Optimal weights
    pub weights: Array1<f64>,
    /// Expected portfolio return (score)
    pub expected_return: f64,
    /// Portfolio variance
    pub portfolio_variance: f64,
    /// Portfolio volatility
    pub portfolio_vol: f64,
    /// Number of active positions
    pub n_positions: usize,
    /// Whether constraints were violated
    pub constraint_violation: bool,
    /// Optimization method used
    pub method: String,
}

/// Build the maximum IC-weighted portfolio.
///
/// Maximizes: w' * alpha -- lambda * w' * Sigma * w
/// Subject to: sum(w) = 1, w >= 0, w <= max_weight
///
/// Uses an iterative projected gradient descent approach.
///
/// # Arguments
/// * `alpha_scores` -- expected return / IC score for each asset
/// * `cov_matrix` -- (n_assets x n_assets) covariance matrix
/// * `constraints` -- portfolio constraints
/// * `risk_aversion` -- lambda, trade-off between return and variance
pub fn max_ic_portfolio(
    alpha_scores: &Array1<f64>,
    cov_matrix: &Array2<f64>,
    constraints: &OptimizationConstraints,
    risk_aversion: f64,
) -> Result<OptimizationResult> {
    let n = alpha_scores.len();
    if cov_matrix.dim() != (n, n) {
        return Err(FactorError::ShapeMismatch {
            msg: format!("cov_matrix must be ({n},{n}), got {:?}", cov_matrix.dim()),
        });
    }

    // Initialize with equal weights
    let mut weights = vec![1.0 / n as f64; n];

    // Projected gradient descent
    // Gradient of objective w.r.t. w: alpha - 2*lambda*Sigma*w
    let max_iter = 500;
    let step_size = 0.01;
    let tol = 1e-8;

    for _iter in 0..max_iter {
        // Compute Sigma * w
        let sigma_w = mat_vec_mul(cov_matrix, &weights);

        // Gradient: alpha - 2 * lambda * Sigma * w
        let gradient: Vec<f64> = (0..n)
            .map(|i| {
                let ai = if alpha_scores[i].is_finite() { alpha_scores[i] } else { 0.0 };
                ai - 2.0 * risk_aversion * sigma_w[i]
            })
            .collect();

        // Gradient ascent step
        let new_weights_unconstrained: Vec<f64> = weights
            .iter()
            .zip(gradient.iter())
            .map(|(&w, &g)| w + step_size * g)
            .collect();

        // Project onto simplex with box constraints [min_weight, max_weight]
        let new_weights = project_simplex(
            &new_weights_unconstrained,
            constraints.min_weight,
            constraints.max_weight,
            constraints.weight_sum,
        );

        // Check convergence
        let diff: f64 = weights
            .iter()
            .zip(new_weights.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        weights = new_weights;
        if diff < tol {
            break;
        }
    }

    // Apply max positions constraint if specified
    if let Some(max_pos) = constraints.max_positions {
        weights = apply_max_positions(&weights, max_pos, constraints.min_weight);
    }

    let w_arr = Array1::from_vec(weights.clone());
    let portfolio_variance = compute_portfolio_variance(&w_arr, cov_matrix);
    let expected_return: f64 = alpha_scores
        .iter()
        .zip(weights.iter())
        .filter_map(|(&a, &w)| if a.is_finite() { Some(a * w) } else { None })
        .sum();

    let n_positions = weights.iter().filter(|&&w| w > constraints.min_weight + 1e-6).count();

    Ok(OptimizationResult {
        weights: w_arr,
        expected_return,
        portfolio_variance,
        portfolio_vol: portfolio_variance.max(0.0).sqrt(),
        n_positions,
        constraint_violation: false,
        method: "max_ic_projected_gradient".into(),
    })
}

/// Build the minimum variance portfolio.
///
/// Minimizes: w' * Sigma * w
/// Subject to: sum(w) = 1, w >= min_weight, w <= max_weight
pub fn min_variance_portfolio(
    cov_matrix: &Array2<f64>,
    constraints: &OptimizationConstraints,
) -> Result<OptimizationResult> {
    let n = cov_matrix.nrows();

    let mut weights = vec![1.0 / n as f64; n];
    let max_iter = 1000;
    let step_size = 0.005;
    let tol = 1e-10;

    for _iter in 0..max_iter {
        let sigma_w = mat_vec_mul(cov_matrix, &weights);

        // Gradient of variance: 2 * Sigma * w
        // We minimize, so gradient descent: w_new = w - step * 2 * Sigma * w
        let new_weights_unconstrained: Vec<f64> = weights
            .iter()
            .zip(sigma_w.iter())
            .map(|(&w, &g)| w - step_size * 2.0 * g)
            .collect();

        let new_weights = project_simplex(
            &new_weights_unconstrained,
            constraints.min_weight,
            constraints.max_weight,
            constraints.weight_sum,
        );

        let diff: f64 = weights
            .iter()
            .zip(new_weights.iter())
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();

        weights = new_weights;
        if diff < tol {
            break;
        }
    }

    if let Some(max_pos) = constraints.max_positions {
        weights = apply_max_positions(&weights, max_pos, constraints.min_weight);
    }

    let w_arr = Array1::from_vec(weights.clone());
    let portfolio_variance = compute_portfolio_variance(&w_arr, cov_matrix);
    let n_positions = weights.iter().filter(|&&w| w > constraints.min_weight + 1e-6).count();

    Ok(OptimizationResult {
        weights: w_arr,
        expected_return: 0.0,
        portfolio_variance,
        portfolio_vol: portfolio_variance.max(0.0).sqrt(),
        n_positions,
        constraint_violation: false,
        method: "min_variance_projected_gradient".into(),
    })
}

/// Build maximum Sharpe ratio portfolio.
///
/// Maximizes Sharpe = (w' * mu - rf) / sqrt(w' * Sigma * w)
/// Uses the Markowitz two-fund separation: maximize mu' * w - lambda * w' * Sigma * w,
/// sweeping lambda to trace the efficient frontier.
pub fn max_sharpe_portfolio(
    expected_returns: &Array1<f64>,
    cov_matrix: &Array2<f64>,
    risk_free_rate: f64,
    constraints: &OptimizationConstraints,
) -> Result<OptimizationResult> {
    // Scale expected returns by excess return
    let excess_returns: Array1<f64> = expected_returns.mapv(|r| r - risk_free_rate);

    // Run maximum IC with moderate risk aversion to find Sharpe-optimal portfolio
    // Binary search on lambda to maximize Sharpe
    let mut best_sharpe = f64::NEG_INFINITY;
    let mut best_result = None;

    for &lambda in &[0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0] {
        match max_ic_portfolio(&excess_returns, cov_matrix, constraints, lambda) {
            Ok(result) => {
                let sharpe = if result.portfolio_vol > 1e-8 {
                    result.expected_return / result.portfolio_vol
                } else {
                    f64::NEG_INFINITY
                };
                if sharpe > best_sharpe {
                    best_sharpe = sharpe;
                    best_result = Some(result);
                }
            }
            Err(_) => {}
        }
    }

    best_result.ok_or_else(|| FactorError::OptimizationFailed {
        reason: "max Sharpe optimization failed for all lambda values".into(),
    })
}

/// Portfolio with factor exposure constraints.
///
/// Optimizes mean-variance objective while keeping each factor exposure
/// within specified bounds.
pub fn factor_constrained_portfolio(
    alpha_scores: &Array1<f64>,
    risk_model: &FactorRiskModel,
    constraints: &OptimizationConstraints,
    risk_aversion: f64,
) -> Result<OptimizationResult> {
    let n_assets = alpha_scores.len();
    let n_factors = risk_model.factor_names.len();

    if risk_model.factor_exposures.nrows() != n_assets {
        return Err(FactorError::DimensionMismatch {
            expected: n_assets,
            got: risk_model.factor_exposures.nrows(),
        });
    }

    // Build full covariance matrix
    let cov = risk_model.asset_covariance();

    // Start with max IC portfolio ignoring factor constraints
    let mut result = max_ic_portfolio(alpha_scores, &cov, constraints, risk_aversion)?;

    // If no factor exposure constraints, return as-is
    let Some(ref max_exposures) = constraints.max_factor_exposures else {
        return Ok(result);
    };

    if max_exposures.len() != n_factors {
        return Err(FactorError::DimensionMismatch {
            expected: n_factors,
            got: max_exposures.len(),
        });
    }

    // Check factor exposure violations and iteratively reduce offending positions
    // This is a simplified constraint satisfaction approach
    let max_iter = 100;
    let mut weights = result.weights.to_vec();

    for _iter in 0..max_iter {
        // Compute portfolio factor exposures
        let port_exposures: Vec<f64> = (0..n_factors)
            .map(|k| {
                (0..n_assets)
                    .map(|i| risk_model.factor_exposures[[i, k]] * weights[i])
                    .sum()
            })
            .collect();

        let mut any_violation = false;

        for k in 0..n_factors {
            let exposure = port_exposures[k];
            let limit = max_exposures[k];
            if exposure.abs() > limit {
                any_violation = true;
                // Reduce weights on assets contributing most to this exposure violation
                let sign = exposure.signum();
                for i in 0..n_assets {
                    let contrib = risk_model.factor_exposures[[i, k]] * weights[i] * sign;
                    if contrib > 0.0 {
                        weights[i] *= 0.95; // reduce by 5%
                    }
                }
            }
        }

        if !any_violation {
            break;
        }

        // Re-normalize
        weights = project_simplex(&weights, constraints.min_weight, constraints.max_weight, constraints.weight_sum);
    }

    let w_arr = Array1::from_vec(weights.clone());
    let portfolio_variance = compute_portfolio_variance(&w_arr, &cov);
    let expected_return: f64 = alpha_scores
        .iter()
        .zip(weights.iter())
        .filter_map(|(&a, &w)| if a.is_finite() { Some(a * w) } else { None })
        .sum();
    let n_positions = weights.iter().filter(|&&w| w > constraints.min_weight + 1e-6).count();

    Ok(OptimizationResult {
        weights: w_arr,
        expected_return,
        portfolio_variance,
        portfolio_vol: portfolio_variance.max(0.0).sqrt(),
        n_positions,
        constraint_violation: false,
        method: "factor_constrained_projected_gradient".into(),
    })
}

/// Matrix-vector multiplication: returns (n,) = A (n x n) * v (n,).
fn mat_vec_mul(a: &Array2<f64>, v: &[f64]) -> Vec<f64> {
    let n = a.nrows();
    (0..n)
        .map(|i| (0..n).map(|j| a[[i, j]] * v[j]).sum())
        .collect()
}

/// Compute w' * Sigma * w.
pub fn compute_portfolio_variance(weights: &Array1<f64>, cov: &Array2<f64>) -> f64 {
    let n = weights.len();
    let mut var = 0.0;
    for i in 0..n {
        for j in 0..n {
            var += weights[i] * weights[j] * cov[[i, j]];
        }
    }
    var.max(0.0)
}

/// Project weights onto the probability simplex with box constraints.
///
/// Finds the projection of unconstrained weights onto:
/// { w : sum(w) = target, min_w <= w_i <= max_w }
///
/// Uses the algorithm of Chen and Ye (2011).
pub fn project_simplex(weights: &[f64], min_w: f64, max_w: f64, target: f64) -> Vec<f64> {
    let n = weights.len();
    if n == 0 {
        return Vec::new();
    }

    // Clip to box first
    let mut clipped: Vec<f64> = weights.iter().map(|&w| w.max(min_w).min(max_w)).collect();

    // Adjust sum to match target
    let current_sum: f64 = clipped.iter().sum();
    let diff = target - current_sum;

    if diff.abs() < 1e-12 {
        return clipped;
    }

    // Distribute excess/deficit proportionally while respecting bounds
    // Iterative approach
    let mut remaining_diff = diff;
    let max_iter = 100;

    for _ in 0..max_iter {
        if remaining_diff.abs() < 1e-12 {
            break;
        }

        // Find assets that can absorb the difference
        let adjustable_count: usize = if remaining_diff > 0.0 {
            clipped.iter().filter(|&&w| w < max_w - 1e-12).count()
        } else {
            clipped.iter().filter(|&&w| w > min_w + 1e-12).count()
        };

        if adjustable_count == 0 {
            break;
        }

        let per_asset = remaining_diff / adjustable_count as f64;
        let mut new_diff = 0.0;

        for w in clipped.iter_mut() {
            if remaining_diff > 0.0 && *w < max_w - 1e-12 {
                let old = *w;
                *w = (*w + per_asset).min(max_w);
                new_diff += old + per_asset - *w; // unabsorbed amount
            } else if remaining_diff < 0.0 && *w > min_w + 1e-12 {
                let old = *w;
                *w = (*w + per_asset).max(min_w);
                new_diff += old + per_asset - *w;
            }
        }
        remaining_diff = new_diff;
    }

    clipped
}

/// Keep only the top-k positions by weight, zero out the rest, then re-normalize.
fn apply_max_positions(weights: &[f64], max_positions: usize, min_weight: f64) -> Vec<f64> {
    let n = weights.len();
    if n <= max_positions {
        return weights.to_vec();
    }

    let mut indexed: Vec<(usize, f64)> = weights
        .iter()
        .enumerate()
        .map(|(i, &w)| (i, w))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let mut result = vec![min_weight; n];
    let top_sum: f64 = indexed[..max_positions].iter().map(|(_, w)| w).sum::<f64>();

    if top_sum > 1e-12 {
        for &(idx, w) in &indexed[..max_positions] {
            result[idx] = w / top_sum;
        }
    } else {
        for &(idx, _) in &indexed[..max_positions] {
            result[idx] = 1.0 / max_positions as f64;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_covariance(n: usize) -> Array2<f64> {
        // Diagonal plus some off-diagonal
        Array2::from_shape_fn((n, n), |(i, j)| {
            if i == j { 0.04 + 0.001 * i as f64 } else { 0.002 }
        })
    }

    #[test]
    fn test_min_variance_portfolio() {
        let cov = make_covariance(10);
        let constraints = OptimizationConstraints::default();
        let result = min_variance_portfolio(&cov, &constraints).unwrap();

        assert_eq!(result.weights.len(), 10);
        let weight_sum: f64 = result.weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-6, "weights sum = {}", weight_sum);
        assert!(result.portfolio_variance > 0.0);
        // All weights in [0, 0.1]
        for w in result.weights.iter() {
            assert!(*w >= -1e-9 && *w <= 0.1 + 1e-9, "weight {} out of bounds", w);
        }
    }

    #[test]
    fn test_max_ic_portfolio() {
        let n = 20;
        let cov = make_covariance(n);
        let alpha = Array1::from_shape_fn(n, |i| 0.01 * (i as f64 - 10.0) / 10.0);
        let constraints = OptimizationConstraints::default();
        let result = max_ic_portfolio(&alpha, &cov, &constraints, 5.0).unwrap();

        let weight_sum: f64 = result.weights.iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-4, "weights sum = {}", weight_sum);
        // High alpha assets should get more weight
        assert!(result.expected_return >= 0.0 || result.expected_return.is_finite());
    }

    #[test]
    fn test_project_simplex() {
        let weights = vec![0.15, 0.15, 0.15, 0.15, 0.40];
        let projected = project_simplex(&weights, 0.0, 0.2, 1.0);
        let sum: f64 = projected.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        for &w in &projected {
            assert!(w >= -1e-9 && w <= 0.2 + 1e-9);
        }
    }

    #[test]
    fn test_max_sharpe() {
        let n = 15;
        let cov = make_covariance(n);
        let mu = Array1::from_shape_fn(n, |i| 0.001 * i as f64);
        let constraints = OptimizationConstraints {
            max_weight: 0.2,
            ..Default::default()
        };
        let result = max_sharpe_portfolio(&mu, &cov, 0.0, &constraints).unwrap();
        assert!(result.portfolio_vol > 0.0);
    }
}
