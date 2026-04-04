/// Factor model analytics: Fama-French, factor attribution, residualisation.

use crate::covariance::{Matrix, Vector};

// ── OLS Regression ────────────────────────────────────────────────────────────

/// OLS regression of `y` on `X` (T × K matrix including an intercept column).
/// Returns (coefficients, r_squared).
pub fn ols(y: &[f64], x: &Matrix) -> (Vec<f64>, f64) {
    let t = y.len();
    let k = x.first().map_or(0, |r| r.len());
    if t < k + 1 || k == 0 {
        return (vec![0.0; k], 0.0);
    }

    // Compute X'X (K×K) and X'y (K).
    let mut xtx = vec![vec![0.0_f64; k]; k];
    let mut xty = vec![0.0_f64; k];
    for (i, row) in x.iter().enumerate() {
        for (j, &xij) in row.iter().enumerate() {
            xty[j] += xij * y[i];
            for (l, &xil) in row.iter().enumerate() {
                xtx[j][l] += xij * xil;
            }
        }
    }

    // Solve via Gaussian elimination.
    let beta = match solve_linear(&xtx, &xty) {
        Some(b) => b,
        None => return (vec![0.0; k], 0.0),
    };

    // R² = 1 - SS_res / SS_tot
    let y_mean = y.iter().sum::<f64>() / t as f64;
    let ss_tot: f64 = y.iter().map(|yi| (yi - y_mean).powi(2)).sum();
    let ss_res: f64 = (0..t)
        .map(|i| {
            let yhat: f64 = x[i].iter().zip(beta.iter()).map(|(xij, bj)| xij * bj).sum();
            (y[i] - yhat).powi(2)
        })
        .sum();
    let r2 = if ss_tot < 1e-12 { 0.0 } else { 1.0 - ss_res / ss_tot };

    (beta, r2)
}

/// Gaussian elimination solver for Ax = b.
fn solve_linear(a: &Matrix, b: &[f64]) -> Option<Vec<f64>> {
    let n = a.len();
    let mut aug: Vec<Vec<f64>> = a
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let mut r = row.clone();
            r.push(b[i]);
            r
        })
        .collect();

    for col in 0..n {
        // Partial pivot.
        let pivot_row = (col..n)
            .max_by(|&a, &b| aug[a][col].abs().partial_cmp(&aug[b][col].abs()).unwrap())?;
        aug.swap(col, pivot_row);
        let diag = aug[col][col];
        if diag.abs() < 1e-14 {
            return None;
        }
        for j in col..=n {
            aug[col][j] /= diag;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in col..=n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }
    Some((0..n).map(|i| aug[i][n]).collect())
}

// ── Factor Loading Estimation ─────────────────────────────────────────────────

/// Estimate factor loadings for a single asset via OLS.
///
/// `asset_returns` — T-vector of asset returns.
/// `factor_returns` — T × K matrix of factor returns (intercept NOT included; added internally).
///
/// Returns `(alpha, betas[K], r_squared)`.
pub fn estimate_loadings(asset_returns: &[f64], factor_returns: &Matrix) -> (f64, Vec<f64>, f64) {
    let t = asset_returns.len();
    let k = factor_returns.first().map_or(0, |r| r.len());

    // Build design matrix [1 | F].
    let x: Matrix = (0..t)
        .map(|i| {
            let mut row = vec![1.0]; // intercept
            row.extend_from_slice(&factor_returns[i]);
            row
        })
        .collect();

    let (beta, r2) = ols(asset_returns, &x);
    let alpha = beta[0];
    let betas = beta[1..].to_vec();
    (alpha, betas, r2)
}

// ── Factor Attribution ─────────────────────────────────────────────────────────

/// Result of factor-based performance attribution.
#[derive(Debug, Clone)]
pub struct FactorAttribution {
    /// PnL attributable to each factor (one value per factor, summed over all assets).
    pub factor_pnl: Vec<f64>,
    /// Names of the factors (if provided).
    pub factor_names: Vec<String>,
    /// Specific (idiosyncratic) PnL: total - factor total.
    pub specific_pnl: f64,
    /// Total portfolio PnL.
    pub total_pnl: f64,
}

/// Attribute portfolio PnL to factors.
///
/// * `weights` — N-vector of portfolio weights.
/// * `loadings` — N × K matrix of factor betas (no intercept).
/// * `factor_returns` — K-vector of factor returns for the period.
/// * `asset_returns` — N-vector of actual asset returns.
pub fn factor_attribution(
    weights: &[f64],
    loadings: &Matrix,
    factor_returns: &[f64],
    asset_returns: &[f64],
    factor_names: Vec<String>,
) -> FactorAttribution {
    let n = weights.len();
    let k = factor_returns.len();

    // Total portfolio return.
    let total_pnl: f64 = weights.iter().zip(asset_returns.iter()).map(|(w, r)| w * r).sum();

    // Factor PnL: for each factor f, sum_i w_i * beta_if * F_f.
    let mut factor_pnl = vec![0.0_f64; k];
    for (i, row) in loadings.iter().enumerate().take(n) {
        for (f, &beta) in row.iter().enumerate().take(k) {
            factor_pnl[f] += weights[i] * beta * factor_returns[f];
        }
    }

    let total_factor_pnl: f64 = factor_pnl.iter().sum();
    let specific_pnl = total_pnl - total_factor_pnl;

    let names = if factor_names.len() == k {
        factor_names
    } else {
        (0..k).map(|i| format!("Factor_{}", i + 1)).collect()
    };

    FactorAttribution {
        factor_pnl,
        factor_names: names,
        specific_pnl,
        total_pnl,
    }
}

// ── Residualisation ───────────────────────────────────────────────────────────

/// Remove factor exposure from asset returns (idiosyncratic returns).
///
/// * `asset_returns` — T-vector of asset returns.
/// * `factor_returns` — T × K matrix of factor returns.
/// Returns specific (residual) returns.
pub fn residualize(asset_returns: &[f64], factor_returns: &Matrix) -> Vec<f64> {
    let t = asset_returns.len();
    let (alpha, betas, _) = estimate_loadings(asset_returns, factor_returns);
    let k = betas.len();
    (0..t)
        .map(|i| {
            let factor_fitted: f64 = (0..k).map(|f| betas[f] * factor_returns[i][f]).sum();
            asset_returns[i] - alpha - factor_fitted
        })
        .collect()
}

// ── Fama-French 3-Factor Model ────────────────────────────────────────────────

/// Fama-French 3-factor model: MKT, SMB, HML.
#[derive(Debug, Clone)]
pub struct FF3Result {
    pub alpha: f64,
    pub beta_mkt: f64,
    pub beta_smb: f64,
    pub beta_hml: f64,
    pub r_squared: f64,
    pub specific_returns: Vec<f64>,
}

/// Fit the Fama-French 3-factor model to a single asset.
///
/// * `asset_returns` — T-vector.
/// * `mkt` — T-vector of market excess returns.
/// * `smb` — T-vector of SMB factor.
/// * `hml` — T-vector of HML factor.
pub fn fama_french_3(
    asset_returns: &[f64],
    mkt: &[f64],
    smb: &[f64],
    hml: &[f64],
) -> FF3Result {
    let t = asset_returns.len();
    let factor_matrix: Matrix = (0..t)
        .map(|i| vec![mkt[i], smb[i], hml[i]])
        .collect();
    let (alpha, betas, r2) = estimate_loadings(asset_returns, &factor_matrix);
    let specific = residualize(asset_returns, &factor_matrix);
    FF3Result {
        alpha,
        beta_mkt: betas.get(0).copied().unwrap_or(0.0),
        beta_smb: betas.get(1).copied().unwrap_or(0.0),
        beta_hml: betas.get(2).copied().unwrap_or(0.0),
        r_squared: r2,
        specific_returns: specific,
    }
}

// ── Cross-sectional factor estimation (Barra-style) ────────────────────────────

/// Estimate cross-sectional factor returns for a single period.
///
/// Given N-vector of asset returns `r` and N × K loading matrix `B`,
/// solves: r = B * f + eps via OLS.
/// Returns K-vector of factor returns.
pub fn cross_sectional_factor_returns(
    asset_returns: &[f64],
    loadings: &Matrix, // N × K
) -> Vec<f64> {
    let n = asset_returns.len();
    let k = loadings.first().map_or(0, |r| r.len());
    if k == 0 { return vec![]; }

    // Transpose loadings to K × N.
    let b_t: Matrix = (0..k)
        .map(|f| (0..n).map(|i| loadings[i][f]).collect())
        .collect();

    // B'B (K×K) and B'r (K).
    let mut btb: Matrix = vec![vec![0.0; k]; k];
    let mut btr: Vec<f64> = vec![0.0; k];
    for f1 in 0..k {
        for f2 in 0..k {
            btb[f1][f2] = b_t[f1].iter().zip(b_t[f2].iter()).map(|(a, b)| a * b).sum();
        }
        btr[f1] = b_t[f1].iter().zip(asset_returns.iter()).map(|(b, r)| b * r).sum();
    }
    solve_linear(&btb, &btr).unwrap_or_else(|| vec![0.0; k])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ols_recovers_known_coefficients() {
        // y = 2 + 3x
        let y: Vec<f64> = (0..10).map(|i| 2.0 + 3.0 * i as f64).collect();
        let x: Matrix = (0..10).map(|i| vec![1.0, i as f64]).collect();
        let (beta, r2) = ols(&y, &x);
        assert!((beta[0] - 2.0).abs() < 1e-8, "intercept={}", beta[0]);
        assert!((beta[1] - 3.0).abs() < 1e-8, "slope={}", beta[1]);
        assert!((r2 - 1.0).abs() < 1e-8, "r2={r2}");
    }

    #[test]
    fn residualize_near_zero_for_perfect_fit() {
        let factor_returns: Matrix = (0..20).map(|i| vec![i as f64 * 0.01]).collect();
        let asset_returns: Vec<f64> = (0..20).map(|i| 0.001 + 1.5 * i as f64 * 0.01).collect();
        let resid = residualize(&asset_returns, &factor_returns);
        let rms: f64 = (resid.iter().map(|r| r * r).sum::<f64>() / resid.len() as f64).sqrt();
        assert!(rms < 1e-8, "rms={rms}");
    }
}
