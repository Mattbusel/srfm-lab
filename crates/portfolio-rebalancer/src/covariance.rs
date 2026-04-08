use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════════════
// SAMPLE COVARIANCE
// ═══════════════════════════════════════════════════════════════════════════

/// Compute sample mean of each column.
pub fn sample_mean(data: &[Vec<f64>]) -> Vec<f64> {
    let t = data.len();
    if t == 0 { return vec![]; }
    let n = data[0].len();
    let mut means = vec![0.0; n];
    for row in data {
        for (i, &v) in row.iter().enumerate() {
            means[i] += v;
        }
    }
    for m in means.iter_mut() {
        *m /= t as f64;
    }
    means
}

/// Sample covariance matrix (unbiased, divides by T-1).
pub fn sample_covariance(returns: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let t = returns.len();
    if t < 2 { return vec![]; }
    let n = returns[0].len();
    let means = sample_mean(returns);

    let mut cov = vec![vec![0.0; n]; n];
    for row in returns {
        for i in 0..n {
            for j in 0..n {
                cov[i][j] += (row[i] - means[i]) * (row[j] - means[j]);
            }
        }
    }

    let denom = (t - 1) as f64;
    for i in 0..n {
        for j in 0..n {
            cov[i][j] /= denom;
        }
    }
    cov
}

/// Sample correlation matrix.
pub fn sample_correlation(returns: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let cov = sample_covariance(returns);
    let n = cov.len();
    let mut corr = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let vol_i = cov[i][i].max(0.0).sqrt();
            let vol_j = cov[j][j].max(0.0).sqrt();
            corr[i][j] = if vol_i > 1e-15 && vol_j > 1e-15 {
                cov[i][j] / (vol_i * vol_j)
            } else {
                if i == j { 1.0 } else { 0.0 }
            };
        }
    }
    corr
}

// ═══════════════════════════════════════════════════════════════════════════
// LEDOIT-WOLF SHRINKAGE
// ═══════════════════════════════════════════════════════════════════════════

/// Ledoit-Wolf (2004) shrinkage estimator.
/// Shrinks sample covariance toward scaled identity: Σ_shrink = δF + (1-δ)S
/// where F = μI (scaled identity), S = sample cov, δ = optimal shrinkage intensity.
pub fn ledoit_wolf_shrinkage(returns: &[Vec<f64>]) -> (Vec<Vec<f64>>, f64) {
    let t = returns.len();
    if t < 2 {
        return (vec![], 0.0);
    }
    let n = returns[0].len();
    let means = sample_mean(returns);
    let sample_cov = sample_covariance(returns);

    // Target: scaled identity F = μI where μ = tr(S)/n
    let mu = (0..n).map(|i| sample_cov[i][i]).sum::<f64>() / n as f64;

    // Compute optimal shrinkage intensity
    // δ* = min(sum_ij(Var(s_ij)) / sum_ij(s_ij - f_ij)², 1)

    // Numerator: sum of asymptotic variances of s_ij
    let mut pi_sum = 0.0; // sum of pi_ij
    for i in 0..n {
        for j in 0..n {
            let mut pi_ij = 0.0;
            for k in 0..t {
                let x = (returns[k][i] - means[i]) * (returns[k][j] - means[j]) - sample_cov[i][j];
                pi_ij += x * x;
            }
            pi_ij /= t as f64;
            pi_sum += pi_ij;
        }
    }

    // Denominator: sum of (s_ij - f_ij)²
    let mut gamma_sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            let f_ij = if i == j { mu } else { 0.0 };
            let diff = sample_cov[i][j] - f_ij;
            gamma_sum += diff * diff;
        }
    }

    let delta = if gamma_sum > 1e-15 {
        (pi_sum / (t as f64 * gamma_sum)).min(1.0).max(0.0)
    } else {
        0.0
    };

    // Shrunk covariance
    let mut shrunk = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let f_ij = if i == j { mu } else { 0.0 };
            shrunk[i][j] = delta * f_ij + (1.0 - delta) * sample_cov[i][j];
        }
    }

    (shrunk, delta)
}

/// Ledoit-Wolf shrinkage toward constant correlation.
pub fn ledoit_wolf_constant_corr(returns: &[Vec<f64>]) -> (Vec<Vec<f64>>, f64) {
    let t = returns.len();
    if t < 2 { return (vec![], 0.0); }
    let n = returns[0].len();
    let means = sample_mean(returns);
    let sample_cov = sample_covariance(returns);

    // Compute average correlation
    let vols: Vec<f64> = (0..n).map(|i| sample_cov[i][i].max(0.0).sqrt()).collect();
    let mut avg_corr = 0.0;
    let mut count = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if vols[i] > 1e-15 && vols[j] > 1e-15 {
                avg_corr += sample_cov[i][j] / (vols[i] * vols[j]);
                count += 1;
            }
        }
    }
    avg_corr = if count > 0 { avg_corr / count as f64 } else { 0.0 };

    // Target: constant correlation matrix F_{ij} = σ_i * σ_j * ρ_bar (i≠j), σ_i² (i=j)
    let mut target = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                target[i][j] = sample_cov[i][i];
            } else {
                target[i][j] = avg_corr * vols[i] * vols[j];
            }
        }
    }

    // Shrinkage intensity (simplified)
    let mut pi_sum = 0.0;
    let mut gamma_sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            let mut pi_ij = 0.0;
            for k in 0..t {
                let x = (returns[k][i] - means[i]) * (returns[k][j] - means[j]) - sample_cov[i][j];
                pi_ij += x * x;
            }
            pi_ij /= t as f64;
            pi_sum += pi_ij;

            let diff = sample_cov[i][j] - target[i][j];
            gamma_sum += diff * diff;
        }
    }

    let delta = if gamma_sum > 1e-15 {
        (pi_sum / (t as f64 * gamma_sum)).min(1.0).max(0.0)
    } else {
        0.0
    };

    let mut shrunk = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            shrunk[i][j] = delta * target[i][j] + (1.0 - delta) * sample_cov[i][j];
        }
    }

    (shrunk, delta)
}

// ═══════════════════════════════════════════════════════════════════════════
// EXPONENTIALLY WEIGHTED COVARIANCE
// ═══════════════════════════════════════════════════════════════════════════

/// Exponentially weighted moving average (EWMA) covariance.
/// λ = decay factor (typically 0.94 for daily, 0.97 for monthly).
pub fn ewma_covariance(returns: &[Vec<f64>], lambda: f64) -> Vec<Vec<f64>> {
    let t = returns.len();
    if t < 2 { return vec![]; }
    let n = returns[0].len();

    let mut cov = vec![vec![0.0; n]; n];
    // Initialize with first observation
    for i in 0..n {
        for j in 0..n {
            cov[i][j] = returns[0][i] * returns[0][j];
        }
    }

    // Recursive EWMA update
    for k in 1..t {
        for i in 0..n {
            for j in 0..n {
                cov[i][j] = lambda * cov[i][j] + (1.0 - lambda) * returns[k][i] * returns[k][j];
            }
        }
    }

    cov
}

/// EWMA with demeaned returns.
pub fn ewma_covariance_demeaned(returns: &[Vec<f64>], lambda: f64) -> Vec<Vec<f64>> {
    let t = returns.len();
    if t < 2 { return vec![]; }
    let n = returns[0].len();
    let means = sample_mean(returns);

    let demeaned: Vec<Vec<f64>> = returns.iter().map(|row| {
        row.iter().zip(means.iter()).map(|(r, m)| r - m).collect()
    }).collect();

    ewma_covariance(&demeaned, lambda)
}

/// Adaptive EWMA: compute optimal lambda that minimizes log-likelihood.
pub fn optimal_ewma_lambda(returns: &[Vec<f64>], lambda_range: (f64, f64), n_grid: usize) -> f64 {
    let mut best_lambda = 0.94;
    let mut best_ll = f64::NEG_INFINITY;

    let dl = (lambda_range.1 - lambda_range.0) / n_grid as f64;

    for i in 0..=n_grid {
        let lambda = lambda_range.0 + i as f64 * dl;
        let cov = ewma_covariance(returns, lambda);
        if cov.is_empty() { continue; }

        // Simple log-likelihood proxy: -0.5 * (log|Σ| + r'Σ⁻¹r) averaged
        let det = matrix_determinant_2x2_approx(&cov);
        if det > 1e-15 {
            let ll = -(det.ln());
            if ll > best_ll {
                best_ll = ll;
                best_lambda = lambda;
            }
        }
    }

    best_lambda
}

fn matrix_determinant_2x2_approx(mat: &[Vec<f64>]) -> f64 {
    // Product of diagonal (approximate for PD matrices)
    let n = mat.len();
    let mut det = 1.0;
    for i in 0..n {
        det *= mat[i][i].max(1e-15);
    }
    det
}

// ═══════════════════════════════════════════════════════════════════════════
// DCC-LIKE ROLLING COVARIANCE
// ═══════════════════════════════════════════════════════════════════════════

/// Dynamic Conditional Correlation (DCC) - simplified rolling version.
/// Estimates time-varying correlations using EWMA on standardized residuals.
pub fn dcc_rolling_covariance(
    returns: &[Vec<f64>],
    vol_lambda: f64,   // EWMA lambda for volatilities (e.g., 0.94)
    corr_lambda: f64,  // EWMA lambda for correlations (e.g., 0.96)
) -> Vec<Vec<Vec<f64>>> {
    let t = returns.len();
    if t < 2 { return vec![]; }
    let n = returns[0].len();

    // Step 1: Estimate time-varying volatilities via EWMA
    let mut vol_sq = vec![vec![0.0; n]; t];
    // Initialize
    for i in 0..n {
        vol_sq[0][i] = returns[0][i] * returns[0][i];
    }
    for k in 1..t {
        for i in 0..n {
            vol_sq[k][i] = vol_lambda * vol_sq[k - 1][i]
                + (1.0 - vol_lambda) * returns[k][i] * returns[k][i];
        }
    }

    // Step 2: Standardize returns
    let mut std_returns = vec![vec![0.0; n]; t];
    for k in 0..t {
        for i in 0..n {
            let vol = vol_sq[k][i].max(1e-15).sqrt();
            std_returns[k][i] = returns[k][i] / vol;
        }
    }

    // Step 3: EWMA on standardized returns for correlation
    let mut q_mat = vec![vec![0.0; n]; n];
    // Initialize Q with identity-ish
    for i in 0..n {
        for j in 0..n {
            q_mat[i][j] = if i == j { 1.0 } else { 0.0 };
        }
    }

    let mut covariances = Vec::with_capacity(t);

    for k in 0..t {
        // Update Q
        for i in 0..n {
            for j in 0..n {
                q_mat[i][j] = corr_lambda * q_mat[i][j]
                    + (1.0 - corr_lambda) * std_returns[k][i] * std_returns[k][j];
            }
        }

        // Normalize Q to get correlation matrix R
        let mut r_mat = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let denom = (q_mat[i][i] * q_mat[j][j]).max(1e-15).sqrt();
                r_mat[i][j] = q_mat[i][j] / denom;
            }
        }

        // Convert to covariance: Σ = D * R * D where D = diag(σ)
        let mut cov = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let vi = vol_sq[k][i].max(0.0).sqrt();
                let vj = vol_sq[k][j].max(0.0).sqrt();
                cov[i][j] = vi * vj * r_mat[i][j];
            }
        }

        covariances.push(cov);
    }

    covariances
}

/// Get most recent DCC covariance estimate.
pub fn dcc_latest_covariance(returns: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let covs = dcc_rolling_covariance(returns, 0.94, 0.96);
    covs.last().cloned().unwrap_or_default()
}

// ═══════════════════════════════════════════════════════════════════════════
// FACTOR MODEL COVARIANCE
// ═══════════════════════════════════════════════════════════════════════════

/// Factor model covariance: Σ = BΣ_fB' + D
pub fn factor_model_covariance(
    factor_loadings: &[Vec<f64>],  // n x k
    factor_cov: &[Vec<f64>],       // k x k
    specific_var: &[f64],          // n diagonal
) -> Vec<Vec<f64>> {
    let n = factor_loadings.len();
    let k = if n > 0 { factor_loadings[0].len() } else { 0 };

    let mut cov = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for l in 0..k {
                for m in 0..k {
                    cov[i][j] += factor_loadings[i][l] * factor_cov[l][m] * factor_loadings[j][m];
                }
            }
            if i == j && i < specific_var.len() {
                cov[i][j] += specific_var[i];
            }
        }
    }
    cov
}

/// Estimate factor loadings via OLS regression.
pub fn estimate_factor_loadings(
    asset_returns: &[Vec<f64>],   // T x n
    factor_returns: &[Vec<f64>],  // T x k
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let t = asset_returns.len();
    if t < 2 || factor_returns.is_empty() {
        return (vec![], vec![]);
    }
    let n = asset_returns[0].len();
    let k = factor_returns[0].len();

    // B = (F'F)^{-1} F'R for each asset
    // F'F (k x k)
    let mut ftf = vec![vec![0.0; k]; k];
    for row in factor_returns {
        for i in 0..k {
            for j in 0..k {
                ftf[i][j] += row[i] * row[j];
            }
        }
    }
    let ftf_inv = matrix_inverse(&ftf);

    let mut loadings = vec![vec![0.0; k]; n];
    let mut residual_var = vec![0.0; n];

    for asset in 0..n {
        // F'r (k x 1)
        let mut ftr = vec![0.0; k];
        for row_idx in 0..t {
            for f in 0..k {
                ftr[f] += factor_returns[row_idx][f] * asset_returns[row_idx][asset];
            }
        }

        // β = (F'F)^{-1} F'r
        for i in 0..k {
            for j in 0..k {
                loadings[asset][i] += ftf_inv[i][j] * ftr[j];
            }
        }

        // Compute specific variance from residuals
        let mut res_sum_sq = 0.0;
        for row_idx in 0..t {
            let mut predicted = 0.0;
            for f in 0..k {
                predicted += loadings[asset][f] * factor_returns[row_idx][f];
            }
            let residual = asset_returns[row_idx][asset] - predicted;
            res_sum_sq += residual * residual;
        }
        residual_var[asset] = res_sum_sq / (t - k) as f64;
    }

    (loadings, residual_var)
}

// ═══════════════════════════════════════════════════════════════════════════
// RANDOM MATRIX THEORY DENOISING (MARCENKO-PASTUR)
// ═══════════════════════════════════════════════════════════════════════════

/// Marcenko-Pastur distribution bounds.
pub fn marcenko_pastur_bounds(n: usize, t: usize, sigma_sq: f64) -> (f64, f64) {
    let q = n as f64 / t as f64;
    let lambda_plus = sigma_sq * (1.0 + q.sqrt()).powi(2);
    let lambda_minus = sigma_sq * (1.0 - q.sqrt()).powi(2);
    (lambda_minus, lambda_plus)
}

/// Marcenko-Pastur PDF.
pub fn marcenko_pastur_pdf(x: f64, q: f64, sigma_sq: f64) -> f64 {
    let lambda_plus = sigma_sq * (1.0 + q.sqrt()).powi(2);
    let lambda_minus = sigma_sq * (1.0 - q.sqrt()).powi(2);
    if x < lambda_minus || x > lambda_plus {
        return 0.0;
    }
    let term = ((lambda_plus - x) * (x - lambda_minus)).max(0.0).sqrt();
    term / (2.0 * PI * q * sigma_sq * x)
}

/// Denoise covariance matrix using Marcenko-Pastur.
/// Eigenvalues below the MP upper bound are replaced with their average.
pub fn rmt_denoise(cov: &[Vec<f64>], n_observations: usize) -> Vec<Vec<f64>> {
    let n = cov.len();
    if n == 0 { return vec![]; }

    // Eigendecomposition via power iteration (simplified)
    let (eigenvalues, eigenvectors) = eigen_decomposition(cov, 100);

    // Marcenko-Pastur bound
    let avg_var = eigenvalues.iter().sum::<f64>() / n as f64;
    let (_, lambda_max) = marcenko_pastur_bounds(n, n_observations, avg_var);

    // Denoise: replace eigenvalues below threshold
    let noise_eigenvalues: Vec<f64> = eigenvalues.iter()
        .filter(|&&e| e <= lambda_max)
        .copied()
        .collect();
    let noise_avg = if noise_eigenvalues.is_empty() {
        0.0
    } else {
        noise_eigenvalues.iter().sum::<f64>() / noise_eigenvalues.len() as f64
    };

    let denoised_eigenvalues: Vec<f64> = eigenvalues.iter().map(|&e| {
        if e > lambda_max { e } else { noise_avg }
    }).collect();

    // Reconstruct: Σ_clean = V * diag(λ_clean) * V'
    reconstruct_from_eigen(&denoised_eigenvalues, &eigenvectors)
}

/// Denoised correlation matrix via target shrinkage.
pub fn rmt_denoise_targeted(cov: &[Vec<f64>], n_observations: usize) -> Vec<Vec<f64>> {
    let n = cov.len();
    let (eigenvalues, eigenvectors) = eigen_decomposition(cov, 100);

    let avg_var = eigenvalues.iter().sum::<f64>() / n as f64;
    let (_, lambda_max) = marcenko_pastur_bounds(n, n_observations, avg_var);

    // Separate signal and noise eigenvalues
    let mut signal_vals = Vec::new();
    let mut signal_vecs = Vec::new();
    let mut noise_vals = Vec::new();

    for (i, &ev) in eigenvalues.iter().enumerate() {
        if ev > lambda_max {
            signal_vals.push(ev);
            signal_vecs.push(eigenvectors[i].clone());
        } else {
            noise_vals.push(ev);
        }
    }

    // Target: preserve signal eigenvalues, shrink noise to preserve trace
    let signal_trace: f64 = signal_vals.iter().sum();
    let total_trace: f64 = eigenvalues.iter().sum();
    let noise_trace = total_trace - signal_trace;
    let n_noise = noise_vals.len();
    let noise_replacement = if n_noise > 0 { noise_trace / n_noise as f64 } else { 0.0 };

    let denoised: Vec<f64> = eigenvalues.iter().map(|&e| {
        if e > lambda_max { e } else { noise_replacement }
    }).collect();

    reconstruct_from_eigen(&denoised, &eigenvectors)
}

/// Simple eigendecomposition via Jacobi iteration.
fn eigen_decomposition(mat: &[Vec<f64>], max_iter: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = mat.len();
    let mut a = mat.to_vec();
    let mut v = vec![vec![0.0; n]; n];
    for i in 0..n { v[i][i] = 1.0; }

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-12 { break; }

        // Jacobi rotation
        let theta = if (a[q][q] - a[p][p]).abs() < 1e-15 {
            PI / 4.0
        } else {
            0.5 * (2.0 * a[p][q] / (a[q][q] - a[p][p])).atan()
        };

        let cos_t = theta.cos();
        let sin_t = theta.sin();

        // Update A = G'AG
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                new_a[i][p] = cos_t * a[i][p] - sin_t * a[i][q];
                new_a[p][i] = new_a[i][p];
                new_a[i][q] = sin_t * a[i][p] + cos_t * a[i][q];
                new_a[q][i] = new_a[i][q];
            }
        }
        new_a[p][p] = cos_t * cos_t * a[p][p] - 2.0 * sin_t * cos_t * a[p][q] + sin_t * sin_t * a[q][q];
        new_a[q][q] = sin_t * sin_t * a[p][p] + 2.0 * sin_t * cos_t * a[p][q] + cos_t * cos_t * a[q][q];
        new_a[p][q] = 0.0;
        new_a[q][p] = 0.0;
        a = new_a;

        // Update eigenvectors
        for i in 0..n {
            let vip = v[i][p];
            let viq = v[i][q];
            v[i][p] = cos_t * vip - sin_t * viq;
            v[i][q] = sin_t * vip + cos_t * viq;
        }
    }

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    // Eigenvectors as columns → transpose to rows
    let mut eigenvectors = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            eigenvectors[i][j] = v[j][i];
        }
    }

    // Sort by eigenvalue descending
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| eigenvalues[b].partial_cmp(&eigenvalues[a]).unwrap());

    let sorted_vals: Vec<f64> = indices.iter().map(|&i| eigenvalues[i]).collect();
    let sorted_vecs: Vec<Vec<f64>> = indices.iter().map(|&i| eigenvectors[i].clone()).collect();

    (sorted_vals, sorted_vecs)
}

fn reconstruct_from_eigen(eigenvalues: &[f64], eigenvectors: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = eigenvalues.len();
    let mut mat = vec![vec![0.0; n]; n];
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                mat[i][j] += eigenvalues[k] * eigenvectors[k][i] * eigenvectors[k][j];
            }
        }
    }
    mat
}

// ═══════════════════════════════════════════════════════════════════════════
// GERBER STATISTIC
// ═══════════════════════════════════════════════════════════════════════════

/// Gerber statistic: robust co-movement measure.
/// Counts concordant and discordant moves exceeding a threshold.
pub fn gerber_statistic(returns: &[Vec<f64>], threshold: f64) -> Vec<Vec<f64>> {
    let t = returns.len();
    if t < 2 { return vec![]; }
    let n = returns[0].len();

    // Compute per-asset volatilities for normalization
    let means = sample_mean(returns);
    let vols: Vec<f64> = (0..n).map(|i| {
        let var: f64 = returns.iter().map(|r| (r[i] - means[i]).powi(2)).sum::<f64>() / (t - 1) as f64;
        var.sqrt()
    }).collect();

    let mut gerber = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in 0..=i {
            let mut n_concordant = 0.0;
            let mut n_discordant = 0.0;
            let mut n_total = 0.0;

            let thresh_i = threshold * vols[i];
            let thresh_j = threshold * vols[j];

            for k in 0..t {
                let ri = returns[k][i] - means[i];
                let rj = returns[k][j] - means[j];

                let big_i = ri.abs() > thresh_i;
                let big_j = rj.abs() > thresh_j;

                if big_i && big_j {
                    n_total += 1.0;
                    if (ri > 0.0 && rj > 0.0) || (ri < 0.0 && rj < 0.0) {
                        n_concordant += 1.0;
                    } else {
                        n_discordant += 1.0;
                    }
                }
            }

            let g = if n_total > 0.0 {
                (n_concordant - n_discordant) / n_total
            } else {
                if i == j { 1.0 } else { 0.0 }
            };

            gerber[i][j] = g;
            gerber[j][i] = g;
        }
    }

    gerber
}

/// Convert Gerber statistic matrix to covariance matrix.
pub fn gerber_covariance(returns: &[Vec<f64>], threshold: f64) -> Vec<Vec<f64>> {
    let n = if returns.is_empty() { 0 } else { returns[0].len() };
    let gerber_corr = gerber_statistic(returns, threshold);
    let means = sample_mean(returns);
    let t = returns.len();
    let vols: Vec<f64> = (0..n).map(|i| {
        let var: f64 = returns.iter().map(|r| (r[i] - means[i]).powi(2)).sum::<f64>() / (t - 1) as f64;
        var.sqrt()
    }).collect();

    let mut cov = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            cov[i][j] = gerber_corr[i][j] * vols[i] * vols[j];
        }
    }
    cov
}

// ═══════════════════════════════════════════════════════════════════════════
// COVARIANCE UTILITIES
// ═══════════════════════════════════════════════════════════════════════════

/// Ensure covariance matrix is positive semi-definite via nearest PSD projection.
pub fn nearest_psd(mat: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = mat.len();
    let (eigenvalues, eigenvectors) = eigen_decomposition(mat, 100);

    // Replace negative eigenvalues with zero
    let fixed: Vec<f64> = eigenvalues.iter().map(|&e| e.max(0.0)).collect();

    reconstruct_from_eigen(&fixed, &eigenvectors)
}

/// Check if matrix is positive semi-definite (all eigenvalues >= 0).
pub fn is_psd(mat: &[Vec<f64>]) -> bool {
    let (eigenvalues, _) = eigen_decomposition(mat, 50);
    eigenvalues.iter().all(|&e| e >= -1e-10)
}

/// Covariance to correlation conversion.
pub fn cov_to_corr(cov: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = cov.len();
    let mut corr = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let vi = cov[i][i].max(0.0).sqrt();
            let vj = cov[j][j].max(0.0).sqrt();
            corr[i][j] = if vi > 1e-15 && vj > 1e-15 {
                (cov[i][j] / (vi * vj)).max(-1.0).min(1.0)
            } else {
                if i == j { 1.0 } else { 0.0 }
            };
        }
    }
    corr
}

/// Correlation to covariance conversion.
pub fn corr_to_cov(corr: &[Vec<f64>], vols: &[f64]) -> Vec<Vec<f64>> {
    let n = corr.len();
    let mut cov = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            cov[i][j] = corr[i][j] * vols[i] * vols[j];
        }
    }
    cov
}

/// Annualize a covariance matrix (multiply by factor).
pub fn annualize_covariance(cov: &[Vec<f64>], periods_per_year: f64) -> Vec<Vec<f64>> {
    let n = cov.len();
    let mut ann = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            ann[i][j] = cov[i][j] * periods_per_year;
        }
    }
    ann
}

/// Matrix inverse via Gauss-Jordan.
fn matrix_inverse(mat: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = mat.len();
    let mut aug = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n { aug[i][j] = mat[i][j]; }
        aug[i][n + i] = 1.0;
    }
    for k in 0..n {
        let mut max_idx = k;
        for i in k + 1..n { if aug[i][k].abs() > aug[max_idx][k].abs() { max_idx = i; } }
        aug.swap(k, max_idx);
        let pivot = aug[k][k];
        if pivot.abs() < 1e-15 { aug[k][k] += 1e-10; continue; }
        for j in 0..2 * n { aug[k][j] /= pivot; }
        for i in 0..n {
            if i != k {
                let factor = aug[i][k];
                for j in 0..2 * n { aug[i][j] -= factor * aug[k][j]; }
            }
        }
    }
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n { for j in 0..n { result[i][j] = aug[i][n + j]; } }
    result
}

/// Blend two covariance matrices: αΣ₁ + (1-α)Σ₂
pub fn blend_covariance(cov1: &[Vec<f64>], cov2: &[Vec<f64>], alpha: f64) -> Vec<Vec<f64>> {
    let n = cov1.len();
    let mut blended = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            blended[i][j] = alpha * cov1[i][j] + (1.0 - alpha) * cov2[i][j];
        }
    }
    blended
}

/// Condition number of a covariance matrix (ratio of max/min eigenvalue).
pub fn condition_number(cov: &[Vec<f64>]) -> f64 {
    let (eigenvalues, _) = eigen_decomposition(cov, 50);
    let max_ev = eigenvalues.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_ev = eigenvalues.iter().cloned().filter(|&e| e > 1e-15).fold(f64::INFINITY, f64::min);
    if min_ev > 1e-15 { max_ev / min_ev } else { f64::INFINITY }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_returns() -> Vec<Vec<f64>> {
        vec![
            vec![0.01, 0.02, -0.01],
            vec![-0.005, 0.01, 0.02],
            vec![0.015, -0.01, 0.005],
            vec![0.02, 0.015, -0.005],
            vec![-0.01, 0.005, 0.01],
            vec![0.008, 0.012, 0.003],
            vec![-0.003, -0.008, 0.015],
            vec![0.012, 0.018, -0.002],
            vec![0.005, -0.005, 0.008],
            vec![-0.008, 0.003, 0.012],
        ]
    }

    #[test]
    fn test_sample_covariance_symmetric() {
        let returns = sample_returns();
        let cov = sample_covariance(&returns);
        let n = cov.len();
        for i in 0..n {
            for j in 0..n {
                assert!((cov[i][j] - cov[j][i]).abs() < 1e-15, "Asymmetric at ({},{})", i, j);
            }
        }
    }

    #[test]
    fn test_ledoit_wolf_shrinkage_intensity() {
        let returns = sample_returns();
        let (shrunk, delta) = ledoit_wolf_shrinkage(&returns);
        assert!(delta >= 0.0 && delta <= 1.0, "Shrinkage intensity out of range: {}", delta);
        assert!(!shrunk.is_empty());
    }

    #[test]
    fn test_ewma_positive_definite() {
        let returns = sample_returns();
        let cov = ewma_covariance(&returns, 0.94);
        for i in 0..cov.len() {
            assert!(cov[i][i] >= 0.0, "Negative diagonal at {}: {}", i, cov[i][i]);
        }
    }

    #[test]
    fn test_gerber_diagonal() {
        let returns = sample_returns();
        let g = gerber_statistic(&returns, 0.5);
        for i in 0..g.len() {
            assert!((g[i][i] - 1.0).abs() < 1e-10, "Gerber diagonal should be 1: {}", g[i][i]);
        }
    }

    #[test]
    fn test_nearest_psd() {
        let mat = vec![
            vec![1.0, 0.9, 0.9],
            vec![0.9, 1.0, 0.9],
            vec![0.9, 0.9, 1.0],
        ];
        let psd = nearest_psd(&mat);
        assert!(is_psd(&psd), "Should be PSD after projection");
    }

    #[test]
    fn test_cov_corr_roundtrip() {
        let returns = sample_returns();
        let cov = sample_covariance(&returns);
        let corr = cov_to_corr(&cov);
        let vols: Vec<f64> = (0..cov.len()).map(|i| cov[i][i].sqrt()).collect();
        let reconstructed = corr_to_cov(&corr, &vols);
        for i in 0..cov.len() {
            for j in 0..cov.len() {
                assert!((cov[i][j] - reconstructed[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn test_marcenko_pastur_bounds() {
        let (low, high) = marcenko_pastur_bounds(100, 500, 1.0);
        assert!(low > 0.0 && low < 1.0);
        assert!(high > 1.0 && high < 3.0);
    }
}
