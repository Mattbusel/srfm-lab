//! Cross-sectional neutralization: sector neutralization and market cap
//! neutralization via OLS residualization.

use ndarray::{Array1, Array2};
use crate::error::{FactorError, Result};

/// Neutralize a factor by removing sector effects.
///
/// For each sector, demean the factor values within that sector.
/// This removes cross-sector variation from the factor, leaving only
/// within-sector variation.
///
/// # Arguments
/// * `factor` -- factor values for each asset
/// * `sector_ids` -- integer sector identifier for each asset
///
/// Returns residualized factor values.
pub fn sector_neutralize(factor: &[f64], sector_ids: &[usize]) -> Result<Vec<f64>> {
    let n = factor.len();
    if sector_ids.len() != n {
        return Err(FactorError::DimensionMismatch {
            expected: n,
            got: sector_ids.len(),
        });
    }

    // Find unique sectors
    let mut unique_sectors: Vec<usize> = sector_ids.to_vec();
    unique_sectors.sort_unstable();
    unique_sectors.dedup();

    let mut result = factor.to_vec();

    for &sector in &unique_sectors {
        // Collect indices and values for this sector
        let sector_indices: Vec<usize> = (0..n)
            .filter(|&i| sector_ids[i] == sector && factor[i].is_finite())
            .collect();

        if sector_indices.is_empty() {
            continue;
        }

        let sector_mean = sector_indices.iter().map(|&i| factor[i]).sum::<f64>()
            / sector_indices.len() as f64;

        for &i in &sector_indices {
            result[i] = factor[i] - sector_mean;
        }
    }

    Ok(result)
}

/// Neutralize a factor by removing sector AND market cap effects.
///
/// Runs OLS of factor ~ sector_dummies + log(market_cap)
/// and returns residuals. This is the standard Barra-style neutralization.
///
/// # Arguments
/// * `factor` -- factor values (n_assets,)
/// * `sector_ids` -- integer sector IDs (n_assets,)
/// * `log_market_caps` -- log market caps (n_assets,)
///
/// Returns OLS residuals (the neutralized factor).
pub fn sector_and_cap_neutralize(
    factor: &[f64],
    sector_ids: &[usize],
    log_market_caps: &[f64],
) -> Result<Vec<f64>> {
    let n = factor.len();
    if sector_ids.len() != n || log_market_caps.len() != n {
        return Err(FactorError::DimensionMismatch {
            expected: n,
            got: sector_ids.len().min(log_market_caps.len()),
        });
    }

    // Find unique sectors -- sort for determinism
    let mut unique_sectors: Vec<usize> = sector_ids.to_vec();
    unique_sectors.sort_unstable();
    unique_sectors.dedup();
    let n_sectors = unique_sectors.len();

    // Identify rows with valid data
    let valid_mask: Vec<bool> = (0..n)
        .map(|i| factor[i].is_finite() && log_market_caps[i].is_finite())
        .collect();
    let valid_idx: Vec<usize> = (0..n).filter(|&i| valid_mask[i]).collect();
    let m = valid_idx.len();

    if m < n_sectors + 2 {
        return Err(FactorError::InsufficientData {
            required: n_sectors + 2,
            got: m,
        });
    }

    // Build design matrix X: [sector dummies | log_mcap]
    // We use n_sectors dummy columns (no intercept -- dummies span the intercept)
    // This avoids multicollinearity.
    let n_cols = n_sectors + 1; // sector dummies + mcap
    let mut x_mat = vec![0.0f64; m * n_cols];
    let mut y_vec = vec![0.0f64; m];

    for (row, &orig_i) in valid_idx.iter().enumerate() {
        y_vec[row] = factor[orig_i];
        // Sector dummy
        let sector_pos = unique_sectors
            .iter()
            .position(|&s| s == sector_ids[orig_i])
            .unwrap();
        x_mat[row * n_cols + sector_pos] = 1.0;
        // Log market cap
        x_mat[row * n_cols + n_sectors] = log_market_caps[orig_i];
    }

    // Solve via normal equations: beta = (X'X)^{-1} X'y
    let coeffs = solve_ols_normal_equations(&x_mat, &y_vec, m, n_cols)?;

    // Compute residuals for all valid rows
    let mut result = vec![f64::NAN; n];
    for (row, &orig_i) in valid_idx.iter().enumerate() {
        let mut fitted = 0.0;
        let sector_pos = unique_sectors
            .iter()
            .position(|&s| s == sector_ids[orig_i])
            .unwrap();
        fitted += coeffs[sector_pos];
        fitted += coeffs[n_sectors] * log_market_caps[orig_i];
        result[orig_i] = y_vec[row] - fitted;
    }

    Ok(result)
}

/// Solve OLS via normal equations: (X'X) beta = X'y.
///
/// X is stored row-major as a flat Vec of length m * n_cols.
/// Returns coefficient vector of length n_cols.
fn solve_ols_normal_equations(
    x_flat: &[f64],
    y: &[f64],
    m: usize,
    n_cols: usize,
) -> Result<Vec<f64>> {
    // Compute X'X (n_cols x n_cols) and X'y (n_cols)
    let mut xtx = vec![0.0f64; n_cols * n_cols];
    let mut xty = vec![0.0f64; n_cols];

    for row in 0..m {
        let x_row = &x_flat[row * n_cols..(row + 1) * n_cols];
        for j in 0..n_cols {
            xty[j] += x_row[j] * y[row];
            for k in 0..n_cols {
                xtx[j * n_cols + k] += x_row[j] * x_row[k];
            }
        }
    }

    // Solve (X'X) beta = X'y via Cholesky / Gaussian elimination
    gaussian_elimination(&mut xtx, &mut xty, n_cols)
}

/// Gaussian elimination with partial pivoting.
///
/// Solves A * x = b in-place. A is n x n stored row-major in a flat Vec.
/// Returns solution vector x.
pub fn gaussian_elimination(a: &mut [f64], b: &mut [f64], n: usize) -> Result<Vec<f64>> {
    for col in 0..n {
        // Find pivot
        let pivot_row = (col..n)
            .max_by(|&i, &j| a[i * n + col].abs().partial_cmp(&a[j * n + col].abs()).unwrap())
            .unwrap();

        if a[pivot_row * n + col].abs() < 1e-14 {
            return Err(FactorError::SingularMatrix {
                operation: "OLS neutralization".into(),
            });
        }

        // Swap rows
        if pivot_row != col {
            for k in 0..n {
                a.swap(col * n + k, pivot_row * n + k);
            }
            b.swap(col, pivot_row);
        }

        let pivot = a[col * n + col];
        // Eliminate below
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            for k in col..n {
                let tmp = a[col * n + k];
                a[row * n + k] -= factor * tmp;
            }
            b[row] -= factor * b[col];
        }
    }

    // Back-substitution
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let mut sum = b[col];
        for k in (col + 1)..n {
            sum -= a[col * n + k] * x[k];
        }
        x[col] = sum / a[col * n + col];
    }

    Ok(x)
}

/// Neutralize factor against an arbitrary set of control variables.
///
/// Computes factor = X_controls * beta + residual via OLS.
/// Returns the residuals as the neutralized factor.
///
/// # Arguments
/// * `factor` -- dependent variable (n,)
/// * `controls` -- control variables as columns, shape (n, k)
/// * `add_intercept` -- whether to add a constant column to controls
pub fn ols_residualize(
    factor: &[f64],
    controls: &Array2<f64>,
    add_intercept: bool,
) -> Result<Vec<f64>> {
    let n = factor.len();
    let (n_obs, n_ctrl) = controls.dim();
    if n_obs != n {
        return Err(FactorError::DimensionMismatch {
            expected: n,
            got: n_obs,
        });
    }

    let n_cols = if add_intercept { n_ctrl + 1 } else { n_ctrl };

    let valid_mask: Vec<bool> = (0..n)
        .map(|i| {
            factor[i].is_finite()
                && controls.row(i).iter().all(|v| v.is_finite())
        })
        .collect();
    let valid_idx: Vec<usize> = (0..n).filter(|&i| valid_mask[i]).collect();
    let m = valid_idx.len();

    if m < n_cols + 1 {
        return Err(FactorError::InsufficientData {
            required: n_cols + 1,
            got: m,
        });
    }

    let mut x_flat = vec![0.0f64; m * n_cols];
    let mut y_vec = vec![0.0f64; m];

    for (row, &orig_i) in valid_idx.iter().enumerate() {
        y_vec[row] = factor[orig_i];
        let mut col_offset = 0;
        if add_intercept {
            x_flat[row * n_cols] = 1.0;
            col_offset = 1;
        }
        for k in 0..n_ctrl {
            x_flat[row * n_cols + col_offset + k] = controls[[orig_i, k]];
        }
    }

    let coeffs = solve_ols_normal_equations(&x_flat, &y_vec, m, n_cols)?;

    let mut result = vec![f64::NAN; n];
    for (row, &orig_i) in valid_idx.iter().enumerate() {
        let mut fitted = 0.0;
        let x_row = &x_flat[row * n_cols..(row + 1) * n_cols];
        for k in 0..n_cols {
            fitted += x_row[k] * coeffs[k];
        }
        result[orig_i] = y_vec[row] - fitted;
    }

    Ok(result)
}

/// Compute OLS regression and return (coefficients, fitted values, residuals, R-squared).
pub struct OlsResult {
    pub coefficients: Vec<f64>,
    pub fitted: Vec<f64>,
    pub residuals: Vec<f64>,
    pub r_squared: f64,
    pub n_obs: usize,
}

/// Full OLS regression with diagnostics.
pub fn ols_full(y: &[f64], x: &Array2<f64>, add_intercept: bool) -> Result<OlsResult> {
    let n = y.len();
    let (n_obs, n_cols_x) = x.dim();
    if n_obs != n {
        return Err(FactorError::DimensionMismatch { expected: n, got: n_obs });
    }

    let n_cols = if add_intercept { n_cols_x + 1 } else { n_cols_x };
    let valid_idx: Vec<usize> = (0..n)
        .filter(|&i| y[i].is_finite() && x.row(i).iter().all(|v| v.is_finite()))
        .collect();
    let m = valid_idx.len();

    if m < n_cols + 1 {
        return Err(FactorError::InsufficientData { required: n_cols + 1, got: m });
    }

    let mut x_flat = vec![0.0f64; m * n_cols];
    let mut y_vec = vec![0.0f64; m];

    for (row, &orig_i) in valid_idx.iter().enumerate() {
        y_vec[row] = y[orig_i];
        let mut col_offset = 0;
        if add_intercept {
            x_flat[row * n_cols] = 1.0;
            col_offset = 1;
        }
        for k in 0..n_cols_x {
            x_flat[row * n_cols + col_offset + k] = x[[orig_i, k]];
        }
    }

    let coefficients = solve_ols_normal_equations(&x_flat, &y_vec, m, n_cols)?;

    let fitted: Vec<f64> = (0..m)
        .map(|row| {
            let x_row = &x_flat[row * n_cols..(row + 1) * n_cols];
            x_row.iter().zip(coefficients.iter()).map(|(xi, ci)| xi * ci).sum()
        })
        .collect();

    let residuals: Vec<f64> = (0..m).map(|i| y_vec[i] - fitted[i]).collect();

    let y_mean = y_vec.iter().sum::<f64>() / m as f64;
    let ss_tot = y_vec.iter().map(|yi| (yi - y_mean).powi(2)).sum::<f64>();
    let ss_res = residuals.iter().map(|r| r.powi(2)).sum::<f64>();
    let r_squared = if ss_tot > 1e-14 { 1.0 - ss_res / ss_tot } else { 0.0 };

    Ok(OlsResult {
        coefficients,
        fitted,
        residuals,
        r_squared,
        n_obs: m,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sector_neutralize() {
        let factor = vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0];
        let sectors = vec![0, 0, 0, 1, 1, 1];
        let neutralized = sector_neutralize(&factor, &sectors).unwrap();
        // Within sector 0: mean = 2, so residuals are -1, 0, 1
        assert!((neutralized[0] - (-1.0)).abs() < 1e-10);
        assert!((neutralized[1] - 0.0).abs() < 1e-10);
        assert!((neutralized[2] - 1.0).abs() < 1e-10);
        // Within sector 1: mean = 11, so residuals are -1, 0, 1
        assert!((neutralized[3] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_sector_and_cap_neutralize() {
        let n = 20;
        let factor: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let sectors: Vec<usize> = (0..n).map(|i| i % 4).collect();
        let log_caps: Vec<f64> = (0..n).map(|i| (10.0 + i as f64).ln()).collect();
        let result = sector_and_cap_neutralize(&factor, &sectors, &log_caps).unwrap();
        assert_eq!(result.len(), n);
        // All residuals should be finite
        for r in &result {
            assert!(r.is_finite(), "got NAN residual");
        }
    }

    #[test]
    fn test_ols_residualize() {
        // Perfect linear relationship -- residuals should be ~0
        let n = 30;
        let x_vals: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y_vals: Vec<f64> = x_vals.iter().map(|&x| 2.0 * x + 1.0).collect();
        let x_matrix = Array2::from_shape_fn((n, 1), |(i, _)| x_vals[i]);
        let residuals = ols_residualize(&y_vals, &x_matrix, true).unwrap();
        for r in residuals.iter().filter(|v| v.is_finite()) {
            assert!(r.abs() < 1e-8, "residual too large: {}", r);
        }
    }

    #[test]
    fn test_gaussian_elimination() {
        // 2x2 system: [2 1; 5 7] * [x; y] = [11; 13] => x=7.36..., y=-3.73...
        let mut a = vec![2.0, 1.0, 5.0, 7.0];
        let mut b = vec![11.0, 13.0];
        let x = gaussian_elimination(&mut a, &mut b, 2).unwrap();
        // Solution: x = 54/9 = 6, y = -1; let's verify
        assert!((2.0 * x[0] + 1.0 * x[1] - 11.0).abs() < 1e-10);
        assert!((5.0 * x[0] + 7.0 * x[1] - 13.0).abs() < 1e-10);
    }
}
