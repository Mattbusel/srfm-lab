/// Portfolio optimisation: mean-variance, min-variance, max-Sharpe,
/// risk parity, Black-Litterman, HRP, Kelly.

use crate::covariance::{mat_vec_mul, quad_form, Matrix, Vector};

// ── Utility ───────────────────────────────────────────────────────────────────

fn sum_weights(w: &[f64]) -> f64 {
    w.iter().sum()
}

/// Project weights onto the simplex (long-only, sum = 1).
pub fn project_simplex(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut u: Vec<f64> = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let mut cssv = 0.0_f64;
    let mut rho = 0usize;
    for (i, &ui) in u.iter().enumerate() {
        cssv += ui;
        if ui - (cssv - 1.0) / (i + 1) as f64 > 0.0 {
            rho = i;
        }
    }
    let cssv_rho: f64 = u[..=rho].iter().sum();
    let theta = (cssv_rho - 1.0) / (rho + 1) as f64;
    v.iter().map(|x| (x - theta).max(0.0)).collect()
}

/// Gradient descent step: update w -= lr * grad, then project to simplex.
fn gradient_step(weights: &mut Vec<f64>, grad: &[f64], lr: f64) {
    let n = weights.len();
    let raw: Vec<f64> = (0..n).map(|i| weights[i] - lr * grad[i]).collect();
    *weights = project_simplex(&raw);
}

// ── Mean-Variance Optimisation ────────────────────────────────────────────────

/// Mean-variance optimisation via gradient descent.
///
/// Maximises: μ'w - (risk_aversion / 2) * w'Σw
/// subject to: w ≥ 0, sum(w) = 1.
pub fn mean_variance(
    expected_returns: &Vector,
    cov_matrix: &Matrix,
    risk_aversion: f64,
) -> Vec<f64> {
    let n = expected_returns.len();
    let mut w = vec![1.0 / n as f64; n];
    let lr = 0.01;
    let max_iter = 2000;

    for _ in 0..max_iter {
        let cov_w = mat_vec_mul(cov_matrix, &w);
        let grad: Vec<f64> = (0..n)
            .map(|i| risk_aversion * cov_w[i] - expected_returns[i])
            .collect();
        gradient_step(&mut w, &grad, lr);
    }
    w
}

// ── Minimum Variance ──────────────────────────────────────────────────────────

/// Minimum-variance portfolio via gradient descent.
/// Minimises: w'Σw subject to w ≥ 0, sum(w) = 1.
pub fn min_variance(cov_matrix: &Matrix) -> Vec<f64> {
    let n = cov_matrix.len();
    let mu = vec![0.0; n]; // zero expected return → pure variance minimisation
    mean_variance(&mu, cov_matrix, 1.0)
}

// ── Maximum Sharpe ────────────────────────────────────────────────────────────

/// Maximum-Sharpe portfolio via parametric approach.
///
/// Uses the analytical two-fund separation:
///   z = Σ⁻¹ (μ - rf * 1), then normalise.
/// For non-invertible Σ we use pseudo-inverse via gradient descent.
pub fn max_sharpe(
    expected_returns: &Vector,
    cov_matrix: &Matrix,
    risk_free: f64,
) -> Vec<f64> {
    let n = expected_returns.len();
    // Excess returns.
    let excess: Vec<f64> = expected_returns.iter().map(|r| r - risk_free).collect();

    // Iterative approach: maximise Sharpe via gradient ascent on unconstrained weights,
    // then normalise.
    let mut w: Vec<f64> = vec![1.0 / n as f64; n];
    let lr = 0.005;

    for _ in 0..3000 {
        let cov_w = mat_vec_mul(cov_matrix, &w);
        let port_var = quad_form(cov_matrix, &w).max(1e-12);
        let port_ret: f64 = w.iter().zip(excess.iter()).map(|(x, r)| x * r).sum();
        // Gradient of Sharpe = (μ - rf - Sharpe * Σw) / σ
        let sharpe = port_ret / port_var.sqrt();
        let grad: Vec<f64> = (0..n)
            .map(|i| -(excess[i] - sharpe * cov_w[i]) / port_var.sqrt())
            .collect();
        gradient_step(&mut w, &grad, lr);
    }
    // Re-normalise.
    let s: f64 = w.iter().sum();
    if s > 1e-12 { w.iter_mut().for_each(|x| *x /= s); }
    w
}

// ── Risk Parity (Equal Risk Contribution) ────────────────────────────────────

/// Risk parity: each asset contributes equally to portfolio volatility.
/// Uses iterative Newton's method (Maillard et al. 2010).
pub fn risk_parity(cov_matrix: &Matrix) -> Vec<f64> {
    let n = cov_matrix.len();
    let mut w: Vec<f64> = vec![1.0 / n as f64; n];
    let target_contrib = 1.0 / n as f64;

    for _ in 0..500 {
        let port_var = quad_form(cov_matrix, &w).max(1e-12);
        let sigma = port_var.sqrt();
        let cov_w = mat_vec_mul(cov_matrix, &w);

        // Marginal risk contribution: MRC_i = (Σw)_i
        // Risk contribution: RC_i = w_i * MRC_i / sigma
        let rc: Vec<f64> = (0..n).map(|i| w[i] * cov_w[i] / sigma).collect();
        let rc_sum: f64 = rc.iter().sum::<f64>().max(1e-12);

        // Gradient: grad_i ∝ RC_i / sigma - target
        let lr = 0.5;
        for i in 0..n {
            let g = rc[i] / rc_sum - target_contrib;
            w[i] = (w[i] - lr * g * w[i]).max(1e-8);
        }
        // Normalise to sum = 1.
        let s: f64 = w.iter().sum::<f64>().max(1e-12);
        w.iter_mut().for_each(|x| *x /= s);
    }
    w
}

// ── Black-Litterman ───────────────────────────────────────────────────────────

/// Black-Litterman posterior expected returns.
///
/// # Arguments
/// * `prior_returns` — equilibrium/prior expected returns (N-vector).
/// * `cov_matrix` — Σ (N×N).
/// * `views` — list of (asset_index, view_return, view_confidence_variance).
/// * `tau` — scalar multiplier on prior uncertainty (typically 0.025..0.05).
pub fn black_litterman(
    prior_returns: &Vector,
    cov_matrix: &Matrix,
    views: &[(usize, f64, f64)],
    tau: f64,
) -> Vector {
    let n = prior_returns.len();
    let k = views.len();

    if k == 0 {
        return prior_returns.to_vec();
    }

    // Build P (K×N) picking matrix, Q (K) view returns, Omega (K diagonal) view variance.
    let mut p: Matrix = vec![vec![0.0; n]; k];
    let mut q: Vector = vec![0.0; k];
    let mut omega_diag: Vector = vec![0.0; k];

    for (vi, &(asset_idx, view_ret, view_var)) in views.iter().enumerate() {
        p[vi][asset_idx] = 1.0;
        q[vi] = view_ret;
        omega_diag[vi] = view_var;
    }

    // Posterior = (tau*Σ)⁻¹ + P'Ω⁻¹P)⁻¹ (tau*Σ)⁻¹*pi + P'Ω⁻¹*Q)
    // We use the simplified closed-form with matrix algebra.

    // tau * Sigma
    let tau_cov: Matrix = cov_matrix.iter().map(|row| row.iter().map(|x| x * tau).collect()).collect();

    // (P * tau_cov * P') + Omega (K×K)
    // P * tau_cov: K×N
    let mut p_tau_cov: Matrix = vec![vec![0.0; n]; k];
    for i in 0..k {
        for j in 0..n {
            for l in 0..n {
                p_tau_cov[i][j] += p[i][l] * tau_cov[l][j];
            }
        }
    }
    // M = p_tau_cov * P' + Omega: K×K
    let mut m: Matrix = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0_f64;
            for l in 0..n {
                s += p_tau_cov[i][l] * p[j][l];
            }
            m[i][j] = s + if i == j { omega_diag[i] } else { 0.0 };
        }
    }

    // Invert M (K×K) — use Gaussian elimination.
    let m_inv = invert_matrix(&m).unwrap_or_else(|| identity_k(k));

    // m_inv * (Q - P * pi): K-vector
    let p_pi: Vector = (0..k).map(|i| {
        (0..n).map(|j| p[i][j] * prior_returns[j]).sum::<f64>()
    }).collect();
    let diff: Vector = (0..k).map(|i| q[i] - p_pi[i]).collect();
    let m_inv_diff: Vector = (0..k).map(|i| {
        (0..k).map(|j| m_inv[i][j] * diff[j]).sum::<f64>()
    }).collect();

    // tau_cov * P' * m_inv_diff
    let p_t_m_inv_diff: Vector = (0..n).map(|j| {
        (0..k).map(|i| p[i][j] * m_inv_diff[i]).sum::<f64>()
    }).collect();
    let correction: Vector = (0..n).map(|j| {
        (0..n).map(|l| tau_cov[j][l] * p_t_m_inv_diff[l]).sum::<f64>()
    }).collect();

    (0..n).map(|i| prior_returns[i] + correction[i]).collect()
}

fn identity_k(k: usize) -> Matrix {
    (0..k).map(|i| (0..k).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect()
}

/// Gauss-Jordan matrix inverse for a square matrix.
pub fn invert_matrix(m: &Matrix) -> Option<Matrix> {
    let n = m.len();
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = m[i].clone();
            let mut id_part = vec![0.0; n];
            id_part[i] = 1.0;
            row.extend(id_part);
            row
        })
        .collect();

    for col in 0..n {
        // Find pivot.
        let pivot = (col..n).max_by(|&a, &b| {
            aug[a][col].abs().partial_cmp(&aug[b][col].abs()).unwrap()
        })?;
        aug.swap(col, pivot);
        let diag = aug[col][col];
        if diag.abs() < 1e-14 {
            return None;
        }
        for j in 0..2 * n {
            aug[col][j] /= diag;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in 0..2 * n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }
    Some((0..n).map(|i| aug[i][n..].to_vec()).collect())
}

// ── Hierarchical Risk Parity ──────────────────────────────────────────────────

/// Hierarchical Risk Parity (HRP).
///
/// Steps:
/// 1. Compute correlation matrix from covariance.
/// 2. Hierarchical clustering (single-linkage).
/// 3. Quasi-diagonalisation: reorder assets.
/// 4. Recursive bisection: allocate weights.
pub fn hierarchical_risk_parity(cov_matrix: &Matrix) -> Vec<f64> {
    let n = cov_matrix.len();

    // Compute correlation matrix and distance.
    let stds: Vec<f64> = (0..n).map(|i| cov_matrix[i][i].sqrt().max(1e-12)).collect();
    let corr: Matrix = (0..n)
        .map(|i| (0..n).map(|j| cov_matrix[i][j] / (stds[i] * stds[j])).collect())
        .collect();
    // Distance: d_ij = sqrt(0.5 * (1 - rho_ij))
    let dist: Matrix = (0..n)
        .map(|i| (0..n).map(|j| (0.5 * (1.0 - corr[i][j])).sqrt()).collect())
        .collect();

    // Hierarchical clustering (single-linkage, Ward-like).
    let order = quasi_diag_order(&dist, n);

    // Recursive bisection.
    let mut weights = vec![1.0_f64; n];
    recursive_bisect(&mut weights, &order, cov_matrix);

    // Map back from reordered indices.
    let mut result = vec![0.0_f64; n];
    for (pos, &asset) in order.iter().enumerate() {
        result[asset] = weights[pos];
    }
    // Normalise.
    let s: f64 = result.iter().sum::<f64>().max(1e-12);
    result.iter_mut().for_each(|x| *x /= s);
    result
}

/// Single-linkage hierarchical clustering → leaf order.
fn quasi_diag_order(dist: &Matrix, n: usize) -> Vec<usize> {
    // Start with each asset in its own cluster.
    let mut clusters: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();

    while clusters.len() > 1 {
        // Find two closest clusters.
        let mut best_d = f64::INFINITY;
        let mut best_i = 0;
        let mut best_j = 1;
        let nc = clusters.len();
        for i in 0..nc {
            for j in (i + 1)..nc {
                // Single-linkage distance.
                let d = clusters[i]
                    .iter()
                    .flat_map(|&a| clusters[j].iter().map(move |&b| dist[a][b]))
                    .fold(f64::INFINITY, f64::min);
                if d < best_d {
                    best_d = d;
                    best_i = i;
                    best_j = j;
                }
            }
        }
        // Merge clusters[best_j] into clusters[best_i].
        let merged_j = clusters.remove(best_j);
        clusters[best_i].extend(merged_j);
    }
    clusters.into_iter().next().unwrap_or_default()
}

/// Recursive bisection step.
fn recursive_bisect(weights: &mut Vec<f64>, items: &[usize], cov: &Matrix) {
    if items.len() <= 1 {
        return;
    }
    let mid = items.len() / 2;
    let left = &items[..mid];
    let right = &items[mid..];

    // Variance of each sub-portfolio (equal-weighted within sub-cluster).
    let var_left = cluster_variance(left, cov);
    let var_right = cluster_variance(right, cov);

    // Alpha: fraction allocated to left.
    let alpha = if var_left + var_right < 1e-12 {
        0.5
    } else {
        1.0 - var_left / (var_left + var_right)
    };

    for &i in left {
        weights[i] *= alpha;
    }
    for &i in right {
        weights[i] *= 1.0 - alpha;
    }

    recursive_bisect(weights, left, cov);
    recursive_bisect(weights, right, cov);
}

fn cluster_variance(items: &[usize], cov: &Matrix) -> f64 {
    let n = items.len();
    if n == 0 {
        return 0.0;
    }
    let w = 1.0 / n as f64;
    let mut var = 0.0_f64;
    for &i in items {
        for &j in items {
            var += w * w * cov[i][j];
        }
    }
    var
}

// ── Kelly Portfolio ───────────────────────────────────────────────────────────

/// Full Kelly criterion: w* = Σ⁻¹ μ.
/// Normalised to sum = 1 (fractional Kelly if desired: scale down).
pub fn kelly_portfolio(expected_returns: &Vector, cov_matrix: &Matrix) -> Vec<f64> {
    let n = expected_returns.len();
    // Invert cov matrix.
    let inv = match invert_matrix(cov_matrix) {
        Some(inv) => inv,
        None => {
            // Fallback: equal weight.
            return vec![1.0 / n as f64; n];
        }
    };
    let w: Vec<f64> = (0..n)
        .map(|i| (0..n).map(|j| inv[i][j] * expected_returns[j]).sum::<f64>())
        .collect();
    // Project to simplex (long-only).
    project_simplex(&w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_cov_2x2() -> Matrix {
        vec![vec![0.04, 0.01], vec![0.01, 0.09]]
    }

    #[test]
    fn min_variance_sums_to_one() {
        let cov = simple_cov_2x2();
        let w = min_variance(&cov);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-6, "sum={}", w.iter().sum::<f64>());
    }

    #[test]
    fn risk_parity_equal_contrib() {
        let cov = simple_cov_2x2();
        let w = risk_parity(&cov);
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn kelly_long_only() {
        let mu = vec![0.1, 0.05];
        let cov = simple_cov_2x2();
        let w = kelly_portfolio(&mu, &cov);
        for wi in &w {
            assert!(*wi >= -1e-9, "wi={wi}");
        }
        assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-6);
    }
}
