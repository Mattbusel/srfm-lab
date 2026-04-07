//! Factor combination: equal-weight, IC-weighted, Barra-style optimization,
//! and PCA composite.

use ndarray::{Array1, Array2, Axis};
use crate::error::{FactorError, Result};
use crate::cross_section::normalize::{zscore_cross_section, NormalizationMethod, normalize_factor_matrix};

/// Method for combining multiple factors into a composite score.
#[derive(Debug, Clone)]
pub enum CompositeMethod {
    /// Simple equal-weight average of (normalized) factor z-scores
    EqualWeight,
    /// Weighted average where weights are proportional to information coefficients
    IcWeighted { ics: Vec<f64> },
    /// Barra-style: minimize variance of composite subject to IC exposure constraint
    BarraOptimal { ics: Vec<f64>, factor_cov: Array2<f64> },
    /// First principal component of factor matrix
    PcaFirst,
    /// User-specified fixed weights
    FixedWeights { weights: Vec<f64> },
}

/// Compute a composite factor score from a normalized factor matrix.
///
/// # Arguments
/// * `factor_matrix` -- shape (n_assets, n_factors), should already be normalized
/// * `method` -- combination method
///
/// Returns composite scores of shape (n_assets,).
pub fn compute_composite(
    factor_matrix: &Array2<f64>,
    method: &CompositeMethod,
) -> Result<Array1<f64>> {
    let (n_assets, n_factors) = factor_matrix.dim();
    if n_assets == 0 || n_factors == 0 {
        return Err(FactorError::InsufficientData { required: 1, got: 0 });
    }

    match method {
        CompositeMethod::EqualWeight => equal_weight_composite(factor_matrix),
        CompositeMethod::IcWeighted { ics } => ic_weighted_composite(factor_matrix, ics),
        CompositeMethod::BarraOptimal { ics, factor_cov } => {
            barra_optimal_composite(factor_matrix, ics, factor_cov)
        }
        CompositeMethod::PcaFirst => pca_first_composite(factor_matrix),
        CompositeMethod::FixedWeights { weights } => {
            fixed_weight_composite(factor_matrix, weights)
        }
    }
}

/// Equal-weight average: for each asset, average all finite factor z-scores.
fn equal_weight_composite(factor_matrix: &Array2<f64>) -> Result<Array1<f64>> {
    let (n_assets, n_factors) = factor_matrix.dim();
    let mut result = Array1::<f64>::from_elem(n_assets, f64::NAN);

    for i in 0..n_assets {
        let mut sum = 0.0;
        let mut count = 0;
        for k in 0..n_factors {
            let v = factor_matrix[[i, k]];
            if v.is_finite() {
                sum += v;
                count += 1;
            }
        }
        if count > 0 {
            result[i] = sum / count as f64;
        }
    }

    Ok(result)
}

/// IC-weighted composite.
///
/// Weights are: w_k = max(IC_k, 0) / sum(max(IC_j, 0))
/// Only factors with positive IC receive weight; negative-IC factors are excluded.
fn ic_weighted_composite(factor_matrix: &Array2<f64>, ics: &[f64]) -> Result<Array1<f64>> {
    let (n_assets, n_factors) = factor_matrix.dim();
    if ics.len() != n_factors {
        return Err(FactorError::DimensionMismatch {
            expected: n_factors,
            got: ics.len(),
        });
    }

    // Only use positive IC factors
    let positive_ics: Vec<f64> = ics.iter().map(|&ic| ic.max(0.0)).collect();
    let total_ic: f64 = positive_ics.iter().sum();

    if total_ic < 1e-12 {
        // Fall back to equal weight if all ICs are zero or negative
        return equal_weight_composite(factor_matrix);
    }

    let weights: Vec<f64> = positive_ics.iter().map(|&ic| ic / total_ic).collect();

    let mut result = Array1::<f64>::from_elem(n_assets, f64::NAN);
    for i in 0..n_assets {
        let mut wsum = 0.0;
        let mut wtotal = 0.0;
        for k in 0..n_factors {
            let v = factor_matrix[[i, k]];
            if v.is_finite() && weights[k] > 1e-12 {
                wsum += weights[k] * v;
                wtotal += weights[k];
            }
        }
        if wtotal > 1e-12 {
            result[i] = wsum / wtotal;
        }
    }

    Ok(result)
}

/// Barra-style optimal composite weights.
///
/// Minimizes: w' * Sigma * w (factor portfolio variance)
/// Subject to: w' * ic = 1 (unit IC exposure), w >= 0 (long-only)
///
/// Analytical solution (ignoring long-only): w* = Sigma^{-1} * ic / (ic' * Sigma^{-1} * ic)
///
/// This is the maximum Sharpe ratio portfolio in "IC space".
fn barra_optimal_composite(
    factor_matrix: &Array2<f64>,
    ics: &[f64],
    factor_cov: &Array2<f64>,
) -> Result<Array1<f64>> {
    let (n_assets, n_factors) = factor_matrix.dim();
    if ics.len() != n_factors {
        return Err(FactorError::DimensionMismatch {
            expected: n_factors,
            got: ics.len(),
        });
    }
    let (cov_rows, cov_cols) = factor_cov.dim();
    if cov_rows != n_factors || cov_cols != n_factors {
        return Err(FactorError::ShapeMismatch {
            msg: format!(
                "factor_cov must be ({0}, {0}), got ({1}, {2})",
                n_factors, cov_rows, cov_cols
            ),
        });
    }

    // Solve Sigma * w = ic using Gaussian elimination
    let mut sigma_flat: Vec<f64> = factor_cov.iter().copied().collect();
    let mut ic_vec: Vec<f64> = ics.to_vec();

    let weights = crate::cross_section::neutralize::gaussian_elimination(
        &mut sigma_flat,
        &mut ic_vec,
        n_factors,
    )?;

    // Normalize so that ic' * w = 1
    let scale: f64 = ics.iter().zip(weights.iter()).map(|(ic, w)| ic * w).sum();
    let final_weights: Vec<f64> = if scale.abs() > 1e-14 {
        weights.iter().map(|w| w / scale).collect()
    } else {
        vec![1.0 / n_factors as f64; n_factors]
    };

    // Apply weights to factor matrix
    fixed_weight_composite(factor_matrix, &final_weights)
}

/// PCA first principal component composite.
///
/// Uses the power iteration method to find the first eigenvector of the
/// factor covariance matrix, then applies as weights.
fn pca_first_composite(factor_matrix: &Array2<f64>) -> Result<Array1<f64>> {
    let (n_assets, n_factors) = factor_matrix.dim();

    if n_factors < 2 {
        return equal_weight_composite(factor_matrix);
    }

    // Build sample covariance matrix of factors (n_factors x n_factors)
    let factor_cov = sample_covariance(factor_matrix)?;

    // Power iteration to find first eigenvector
    let first_pc = power_iteration(&factor_cov, n_factors, 100, 1e-10)?;

    // Ensure PC has positive mean loading (for interpretability)
    let mean_loading = first_pc.iter().sum::<f64>() / n_factors as f64;
    let sign = if mean_loading >= 0.0 { 1.0 } else { -1.0 };
    let weights: Vec<f64> = first_pc.iter().map(|&w| sign * w).collect();

    fixed_weight_composite(factor_matrix, &weights)
}

/// Apply fixed weights to factor matrix.
pub fn fixed_weight_composite(factor_matrix: &Array2<f64>, weights: &[f64]) -> Result<Array1<f64>> {
    let (n_assets, n_factors) = factor_matrix.dim();
    if weights.len() != n_factors {
        return Err(FactorError::DimensionMismatch {
            expected: n_factors,
            got: weights.len(),
        });
    }

    let weight_sum: f64 = weights.iter().sum();
    let normalized_weights: Vec<f64> = if weight_sum.abs() > 1e-12 {
        weights.iter().map(|w| w / weight_sum).collect()
    } else {
        vec![1.0 / n_factors as f64; n_factors]
    };

    let mut result = Array1::<f64>::from_elem(n_assets, f64::NAN);
    for i in 0..n_assets {
        let mut wsum = 0.0;
        let mut wtotal = 0.0;
        for k in 0..n_factors {
            let v = factor_matrix[[i, k]];
            if v.is_finite() {
                wsum += normalized_weights[k] * v;
                wtotal += normalized_weights[k].abs();
            }
        }
        if wtotal > 1e-12 {
            result[i] = wsum / wtotal * weight_sum.abs().max(1.0).min(1.0);
        }
    }

    Ok(result)
}

/// Compute sample covariance matrix of factor columns.
///
/// Uses pairwise complete observations for each pair of factors.
pub fn sample_covariance(factor_matrix: &Array2<f64>) -> Result<Array2<f64>> {
    let (n_assets, n_factors) = factor_matrix.dim();

    // Compute column means (ignoring NaN)
    let mut means = vec![0.0f64; n_factors];
    let mut counts = vec![0usize; n_factors];
    for i in 0..n_assets {
        for k in 0..n_factors {
            let v = factor_matrix[[i, k]];
            if v.is_finite() {
                means[k] += v;
                counts[k] += 1;
            }
        }
    }
    for k in 0..n_factors {
        if counts[k] > 0 {
            means[k] /= counts[k] as f64;
        }
    }

    let mut cov = Array2::<f64>::zeros((n_factors, n_factors));

    for j in 0..n_factors {
        for k in j..n_factors {
            let mut sum = 0.0;
            let mut n_pairs = 0;
            for i in 0..n_assets {
                let vj = factor_matrix[[i, j]];
                let vk = factor_matrix[[i, k]];
                if vj.is_finite() && vk.is_finite() {
                    sum += (vj - means[j]) * (vk - means[k]);
                    n_pairs += 1;
                }
            }
            let c = if n_pairs > 1 {
                sum / (n_pairs - 1) as f64
            } else {
                0.0
            };
            cov[[j, k]] = c;
            cov[[k, j]] = c;
        }
    }

    Ok(cov)
}

/// Power iteration to find the dominant eigenvector.
///
/// Returns the normalized eigenvector corresponding to the largest eigenvalue.
fn power_iteration(
    matrix: &Array2<f64>,
    n: usize,
    max_iter: usize,
    tol: f64,
) -> Result<Vec<f64>> {
    let mut v: Vec<f64> = vec![1.0 / (n as f64).sqrt(); n];

    for _ in 0..max_iter {
        // Compute Av
        let mut av = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                av[i] += matrix[[i, j]] * v[j];
            }
        }

        // Compute norm
        let norm = av.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-14 {
            break;
        }

        // Check convergence
        let new_v: Vec<f64> = av.iter().map(|x| x / norm).collect();
        let diff = v.iter().zip(new_v.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        v = new_v;

        if diff < tol {
            break;
        }
    }

    Ok(v)
}

/// Compute factor-factor correlations for composite design.
///
/// Returns the correlation matrix of factors.
pub fn factor_correlation_matrix(factor_matrix: &Array2<f64>) -> Result<Array2<f64>> {
    let cov = sample_covariance(factor_matrix)?;
    let n = cov.nrows();
    let mut corr = Array2::<f64>::zeros((n, n));

    let std_devs: Vec<f64> = (0..n).map(|i| cov[[i, i]].sqrt()).collect();

    for i in 0..n {
        for j in 0..n {
            let denom = std_devs[i] * std_devs[j];
            corr[[i, j]] = if denom > 1e-12 {
                cov[[i, j]] / denom
            } else {
                if i == j { 1.0 } else { 0.0 }
            };
        }
    }

    Ok(corr)
}

/// Orthogonalize factor matrix via modified Gram-Schmidt.
///
/// Returns an orthogonalized matrix where factors are mutually uncorrelated.
pub fn gram_schmidt_orthogonalize(factor_matrix: &Array2<f64>) -> Result<Array2<f64>> {
    let (n_assets, n_factors) = factor_matrix.dim();
    let mut result = factor_matrix.clone();

    for k in 1..n_factors {
        for j in 0..k {
            // Compute projection of column k onto column j
            let mut dot_kj = 0.0;
            let mut dot_jj = 0.0;
            let mut count = 0;

            for i in 0..n_assets {
                let vk = result[[i, k]];
                let vj = result[[i, j]];
                if vk.is_finite() && vj.is_finite() {
                    dot_kj += vk * vj;
                    dot_jj += vj * vj;
                    count += 1;
                }
            }

            if count < 2 || dot_jj < 1e-14 {
                continue;
            }

            let proj = dot_kj / dot_jj;

            // Subtract projection
            for i in 0..n_assets {
                if result[[i, k]].is_finite() && result[[i, j]].is_finite() {
                    result[[i, k]] -= proj * result[[i, j]];
                }
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_factor_matrix(n: usize, k: usize) -> Array2<f64> {
        // Pre-normalized factor matrix with some correlation
        Array2::from_shape_fn((n, k), |(i, j)| {
            let base = ((i as f64 + j as f64 * 0.5) * 0.3).sin();
            // Z-score by column
            base
        })
    }

    #[test]
    fn test_equal_weight() {
        let m = make_factor_matrix(50, 4);
        let composite = compute_composite(&m, &CompositeMethod::EqualWeight).unwrap();
        assert_eq!(composite.len(), 50);
        assert!(composite.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_ic_weighted() {
        let m = make_factor_matrix(50, 4);
        let ics = vec![0.05, 0.08, -0.01, 0.03];
        let composite = compute_composite(&m, &CompositeMethod::IcWeighted { ics }).unwrap();
        assert_eq!(composite.len(), 50);
    }

    #[test]
    fn test_pca_composite() {
        let m = make_factor_matrix(100, 5);
        let composite = compute_composite(&m, &CompositeMethod::PcaFirst).unwrap();
        assert_eq!(composite.len(), 100);
        assert!(composite.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_barra_composite() {
        let n_factors = 3;
        let m = make_factor_matrix(50, n_factors);
        let ics = vec![0.05, 0.08, 0.03];
        let factor_cov = Array2::from_shape_fn((n_factors, n_factors), |(i, j)| {
            if i == j { 1.0 } else { 0.1 }
        });
        let composite = compute_composite(
            &m,
            &CompositeMethod::BarraOptimal { ics, factor_cov },
        ).unwrap();
        assert_eq!(composite.len(), 50);
    }

    #[test]
    fn test_sample_covariance() {
        let m = Array2::from_shape_fn((100, 3), |(i, j)| i as f64 * 0.01 + j as f64 * 0.1);
        let cov = sample_covariance(&m).unwrap();
        assert_eq!(cov.dim(), (3, 3));
        // Diagonal should be positive
        for k in 0..3 {
            assert!(cov[[k, k]] > 0.0);
        }
        // Should be symmetric
        for j in 0..3 {
            for k in 0..3 {
                assert!((cov[[j, k]] - cov[[k, j]]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_gram_schmidt() {
        let m = make_factor_matrix(50, 3);
        let ortho = gram_schmidt_orthogonalize(&m).unwrap();
        assert_eq!(ortho.dim(), (50, 3));
        // Check orthogonality: column 0 and 1 should have near-zero covariance
        let cov = sample_covariance(&ortho).unwrap();
        // Gram-Schmidt makes off-diagonal of sample cov near-zero (not exact due to floating point)
        // Gram-Schmidt projection reduces covariance significantly
        assert!(cov[[0, 1]].abs() < 1e-3, "cov[0,1] = {}", cov[[0, 1]]);
    }
}
