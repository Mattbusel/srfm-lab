/// Covariance matrix estimation methods.
///
/// All functions accept a returns matrix as a Vec<Vec<f64>> where
/// rows = time periods, columns = assets.

// ── Helpers ───────────────────────────────────────────────────────────────────

pub type Matrix = Vec<Vec<f64>>;
pub type Vector = Vec<f64>;

/// Number of rows in a matrix.
pub fn nrows(m: &Matrix) -> usize { m.len() }
/// Number of columns in a matrix (uses first row).
pub fn ncols(m: &Matrix) -> usize { m.first().map_or(0, |r| r.len()) }

fn col(m: &Matrix, j: usize) -> Vector {
    m.iter().map(|row| row[j]).collect()
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn demean(v: &[f64]) -> Vec<f64> {
    let m = mean(v);
    v.iter().map(|x| x - m).collect()
}

/// Transpose a matrix.
pub fn transpose(m: &Matrix) -> Matrix {
    let r = nrows(m);
    let c = ncols(m);
    (0..c).map(|j| (0..r).map(|i| m[i][j]).collect()).collect()
}

/// Matrix–vector product: A * v.
pub fn mat_vec_mul(a: &Matrix, v: &[f64]) -> Vector {
    a.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(x, y)| x * y).sum())
        .collect()
}

/// v^T A v
pub fn quad_form(a: &Matrix, v: &[f64]) -> f64 {
    let av = mat_vec_mul(a, v);
    av.iter().zip(v.iter()).map(|(x, y)| x * y).sum()
}

/// Matrix addition.
pub fn mat_add(a: &Matrix, b: &Matrix) -> Matrix {
    a.iter()
        .zip(b.iter())
        .map(|(ra, rb)| ra.iter().zip(rb.iter()).map(|(x, y)| x + y).collect())
        .collect()
}

/// Scalar × matrix.
pub fn mat_scale(a: &Matrix, s: f64) -> Matrix {
    a.iter()
        .map(|row| row.iter().map(|x| x * s).collect())
        .collect()
}

/// Identity matrix n×n.
pub fn identity(n: usize) -> Matrix {
    (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect()
}

/// Covariance between two demeaned series.
fn cov_pair(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    x[..n].iter().zip(y[..n].iter()).map(|(a, b)| a * b).sum::<f64>() / (n - 1) as f64
}

// ── Sample Covariance ─────────────────────────────────────────────────────────

/// Compute the sample covariance matrix from a returns matrix (T × N).
pub fn sample_covariance(returns: &Matrix) -> Matrix {
    let t = nrows(returns);
    let n = ncols(returns);
    assert!(t > 1, "Need at least 2 observations");

    let demeaned: Vec<Vec<f64>> = (0..n).map(|j| demean(&col(returns, j))).collect();

    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| cov_pair(&demeaned[i], &demeaned[j]))
                .collect()
        })
        .collect()
}

// ── Ledoit-Wolf Shrinkage ─────────────────────────────────────────────────────

/// Ledoit-Wolf analytical shrinkage estimator (Oracle approximating shrinkage).
/// Returns (shrunk_covariance, shrinkage_coefficient).
pub fn ledoit_wolf(returns: &Matrix) -> (Matrix, f64) {
    let t = nrows(returns) as f64;
    let n = ncols(returns);
    let nf = n as f64;

    let s = sample_covariance(returns);

    // Target: constant correlation model (scaled identity simplification).
    // Trace of S.
    let trace_s: f64 = (0..n).map(|i| s[i][i]).sum();
    let mu = trace_s / nf;

    // delta^2: Frobenius norm ||S - muI||^2
    let mut delta_sq = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            let target = if i == j { mu } else { 0.0 };
            delta_sq += (s[i][j] - target).powi(2);
        }
    }

    // beta^2: estimation error variance (simplified Oracle formula).
    let mut beta_sq = 0.0_f64;
    let demeaned: Vec<Vec<f64>> = (0..n).map(|j| demean(&col(returns, j))).collect();
    for i in 0..n {
        for k in 0..n {
            let mut sum4 = 0.0_f64;
            let t_int = t as usize;
            for obs in 0..t_int {
                sum4 += demeaned[i][obs] * demeaned[k][obs];
            }
            let z_ik = sum4 / t;
            beta_sq += (z_ik - s[i][k]).powi(2);
        }
    }
    beta_sq /= t;

    // Shrinkage coefficient α = min(beta^2/delta^2, 1).
    let alpha = if delta_sq == 0.0 { 0.0 } else { (beta_sq / delta_sq).min(1.0) };

    // Shrunk matrix = (1-α) * S + α * μ * I
    let shrunk = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let target = if i == j { mu } else { 0.0 };
                    (1.0 - alpha) * s[i][j] + alpha * target
                })
                .collect()
        })
        .collect();

    (shrunk, alpha)
}

// ── Exponentially Weighted Covariance ────────────────────────────────────────

/// Exponentially weighted covariance matrix.
/// `lambda` in (0, 1): closer to 1 = slower decay.
pub fn exponential_weighted(returns: &Matrix, lambda: f64) -> Matrix {
    let t = nrows(returns);
    let n = ncols(returns);
    assert!(t > 1);

    let mut cov = vec![vec![0.0_f64; n]; n];
    let mut weight_sum = 0.0_f64;

    // Most recent observation gets highest weight.
    for (rev_idx, row) in returns.iter().rev().enumerate() {
        let w = lambda.powi(rev_idx as i32);
        weight_sum += w;
        for i in 0..n {
            for j in 0..n {
                cov[i][j] += w * row[i] * row[j];
            }
        }
    }
    for i in 0..n {
        for j in 0..n {
            cov[i][j] /= weight_sum;
        }
    }

    // Demean correction (subtract weighted outer product of means).
    let mut wmeans = vec![0.0_f64; n];
    let mut w_sum2 = 0.0_f64;
    for (rev_idx, row) in returns.iter().rev().enumerate() {
        let w = lambda.powi(rev_idx as i32);
        w_sum2 += w;
        for j in 0..n {
            wmeans[j] += w * row[j];
        }
    }
    for j in 0..n {
        wmeans[j] /= w_sum2;
    }
    for i in 0..n {
        for j in 0..n {
            cov[i][j] -= wmeans[i] * wmeans[j];
        }
    }
    cov
}

// ── DCC-GARCH ─────────────────────────────────────────────────────────────────

/// Simplified DCC-GARCH time-varying correlation.
///
/// Steps:
/// 1. Fit per-asset GARCH(1,1) to get conditional volatilities.
/// 2. Standardise returns.
/// 3. Compute EW correlation of standardised residuals (DCC step).
///
/// Returns the conditional covariance matrix at the final time step.
pub fn dcc_garch_correlation(returns: &Matrix, a: f64, b: f64, dcc_a: f64, dcc_b: f64) -> Matrix {
    let t = nrows(returns);
    let n = ncols(returns);
    assert!(t > 10);

    // Step 1: per-asset GARCH(1,1) conditional variance h_t.
    let mut h: Vec<Vec<f64>> = Vec::with_capacity(n); // h[asset][t]
    let mut std_resid: Vec<Vec<f64>> = Vec::with_capacity(n);

    for j in 0..n {
        let series: Vec<f64> = (0..t).map(|i| returns[i][j]).collect();
        let var_series = garch11_variance(&series, a, b);
        let resid: Vec<f64> = series
            .iter()
            .zip(var_series.iter())
            .map(|(r, h)| r / h.sqrt().max(1e-12))
            .collect();
        h.push(var_series);
        std_resid.push(resid);
    }

    // Step 2: DCC correlation update.
    // Q_bar = unconditional correlation of std residuals.
    let std_resid_t: Vec<Vec<f64>> = (0..t)
        .map(|i| (0..n).map(|j| std_resid[j][i]).collect())
        .collect();
    let q_bar = sample_covariance(&std_resid_t);

    // Iterate DCC: Q_t = (1-a-b)*Q_bar + a*eps_{t-1}*eps_{t-1}' + b*Q_{t-1}
    let mut qt = q_bar.clone();
    for i in 1..t {
        let eps: Vec<f64> = (0..n).map(|j| std_resid[j][i - 1]).collect();
        let mut new_q = mat_scale(&q_bar, 1.0 - dcc_a - dcc_b);
        // outer product of eps.
        let outer: Matrix = (0..n)
            .map(|r| (0..n).map(|c| dcc_a * eps[r] * eps[c]).collect())
            .collect();
        let bq: Matrix = mat_scale(&qt, dcc_b);
        new_q = mat_add(&mat_add(&new_q, &outer), &bq);
        qt = new_q;
    }

    // Correlation R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2}
    let qt_diag_sqrt: Vec<f64> = (0..n).map(|i| qt[i][i].sqrt().max(1e-12)).collect();
    let r: Matrix = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| qt[i][j] / (qt_diag_sqrt[i] * qt_diag_sqrt[j]))
                .collect()
        })
        .collect();

    // Conditional covariance = D_T * R_T * D_T  (D = diag of conditional stds).
    let cond_std: Vec<f64> = (0..n).map(|j| h[j][t - 1].sqrt()).collect();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| cond_std[i] * r[i][j] * cond_std[j])
                .collect()
        })
        .collect()
}

/// GARCH(1,1) conditional variance series.
/// Returns h[t] for t=0..T.
pub fn garch11_variance(returns: &[f64], alpha: f64, beta: f64) -> Vec<f64> {
    let t = returns.len();
    let omega_hat = returns.iter().map(|r| r * r).sum::<f64>() / t as f64 * (1.0 - alpha - beta);
    let omega = omega_hat.max(1e-12);

    let mut h = vec![0.0_f64; t];
    h[0] = returns.iter().map(|r| r * r).sum::<f64>() / t as f64;
    for i in 1..t {
        h[i] = omega + alpha * returns[i - 1].powi(2) + beta * h[i - 1];
    }
    h
}

// ── Factor Model Covariance ───────────────────────────────────────────────────

/// Factor-model covariance: Σ = B * Σ_F * B' + diag(specific_var).
///
/// * `factor_cov` — K×K factor covariance matrix.
/// * `loadings` — N×K matrix of factor loadings.
/// * `specific_var` — N-vector of idiosyncratic variances.
pub fn factor_model_cov(
    factor_cov: &Matrix,
    loadings: &Matrix,
    specific_var: &Vector,
) -> Matrix {
    let n = loadings.len();
    let k = factor_cov.len();

    // BΣF: N×K matrix
    let mut b_sf: Matrix = vec![vec![0.0; k]; n];
    for i in 0..n {
        for l in 0..k {
            for m in 0..k {
                b_sf[i][l] += loadings[i][m] * factor_cov[m][l];
            }
        }
    }

    // BΣF B': N×N
    let mut cov: Matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0_f64;
            for l in 0..k {
                sum += b_sf[i][l] * loadings[j][l];
            }
            cov[i][j] = sum;
        }
    }
    // Add specific variance on diagonal.
    for i in 0..n {
        cov[i][i] += specific_var[i];
    }
    cov
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mock_returns(t: usize, n: usize, seed: u64) -> Matrix {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut rng_state = seed;
        let mut pseudo_rand = || -> f64 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            ((rng_state as f64) / u64::MAX as f64 - 0.5) * 0.02
        };
        (0..t).map(|_| (0..n).map(|_| pseudo_rand()).collect()).collect()
    }

    #[test]
    fn sample_cov_symmetric() {
        let returns = mock_returns(100, 5, 42);
        let cov = sample_covariance(&returns);
        for i in 0..5 {
            for j in 0..5 {
                assert!((cov[i][j] - cov[j][i]).abs() < 1e-12, "not symmetric");
            }
        }
    }

    #[test]
    fn lw_shrinkage_in_range() {
        let returns = mock_returns(60, 10, 99);
        let (_, alpha) = ledoit_wolf(&returns);
        assert!(alpha >= 0.0 && alpha <= 1.0, "alpha={alpha}");
    }
}
