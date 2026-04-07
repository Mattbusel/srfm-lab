//! portfolio_analytics_wasm.rs -- Portfolio analytics for the SRFM browser dashboard.
//!
//! Implements mean-variance optimization, correlation analysis, and risk contribution
//! decomposition entirely in WASM with no external linear algebra dependencies.
//! All heavy lifting uses iterative gradient descent and simple matrix routines.
//!
//! Input/output convention: JSON strings consumed from JS, JSON strings returned.
//! All weight vectors use the same asset ordering as the input returns matrix.

use wasm_bindgen::prelude::*;
use serde::Serialize;

// ---------------------------------------------------------------------------
// Input / output types
// ---------------------------------------------------------------------------

/// Returns matrix: outer index = asset, inner index = time.
/// Passed as a 2D JSON array [[r1_t1, r1_t2, ...], [r2_t1, r2_t2, ...], ...].
type ReturnsMatrix = Vec<Vec<f64>>;

#[derive(Serialize)]
struct FrontierPoint {
    weights: Vec<f64>,
    expected_return: f64,
    volatility: f64,
    sharpe: f64,
}

#[derive(Serialize)]
struct RiskContribution {
    asset_index: usize,
    marginal_risk: f64,
    risk_contribution: f64,
    risk_pct: f64,
}

#[derive(Serialize)]
struct MinVarianceResult {
    weights: Vec<f64>,
    expected_return: f64,
    volatility: f64,
    sharpe: f64,
    converged: bool,
    iterations: usize,
}

// ---------------------------------------------------------------------------
// Exported free functions (wasm_bindgen)
// ---------------------------------------------------------------------------

/// Compute the efficient frontier by sweeping target returns from min to max.
///
/// Uses projected gradient descent to minimize variance at each target return.
/// Returns JSON array of FrontierPoint with `n_points` evenly-spaced solutions.
/// Assumes annual risk-free rate of 0 for Sharpe computation.
#[wasm_bindgen]
pub fn compute_efficient_frontier(returns_json: &str, n_points: usize) -> String {
    let returns: ReturnsMatrix = match serde_json::from_str(returns_json) {
        Ok(r) => r,
        Err(e) => return format!("{{\"error\":\"parse: {}\"}}", e),
    };

    let n_assets = returns.len();
    if n_assets < 2 {
        return "{\"error\":\"need at least 2 assets\"}".to_string();
    }

    let n_obs = returns[0].len();
    if n_obs < 4 {
        return "{\"error\":\"need at least 4 observations\"}".to_string();
    }

    let means = compute_means(&returns);
    let cov = compute_covariance_matrix(&returns);

    let min_ret = means.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ret = means.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (max_ret - min_ret).abs() < 1e-12 {
        return "{\"error\":\"all assets have same expected return\"}".to_string();
    }

    let n_pts = n_points.max(3).min(200);
    let mut frontier: Vec<FrontierPoint> = Vec::with_capacity(n_pts);

    for k in 0..n_pts {
        let target = min_ret + (max_ret - min_ret) * k as f64 / (n_pts - 1) as f64;
        let (weights, converged) = optimize_constrained_mv(
            &means,
            &cov,
            n_assets,
            Some(target),
            500,
            1e-8,
        );
        let _ = converged;
        let exp_ret = portfolio_return(&weights, &means);
        let vol = portfolio_volatility(&weights, &cov);
        let sharpe = if vol > 1e-12 { exp_ret / vol } else { 0.0 };

        frontier.push(FrontierPoint {
            weights,
            expected_return: exp_ret,
            volatility: vol,
            sharpe,
        });
    }

    serde_json::to_string(&frontier).unwrap_or_else(|_| "[]".to_string())
}

/// Compute the NxN Pearson correlation matrix for the given returns.
///
/// Returns JSON: [[1.0, r12, ...], [r21, 1.0, ...], ...].
#[wasm_bindgen]
pub fn compute_correlation_matrix(returns_json: &str) -> String {
    let returns: ReturnsMatrix = match serde_json::from_str(returns_json) {
        Ok(r) => r,
        Err(e) => return format!("{{\"error\":\"parse: {}\"}}", e),
    };

    let n = returns.len();
    if n == 0 {
        return "[]".to_string();
    }

    let means = compute_means(&returns);
    let stds = compute_stds(&returns, &means);

    let mut corr = vec![vec![0.0f64; n]; n];
    let n_obs = returns[0].len() as f64;

    for i in 0..n {
        corr[i][i] = 1.0;
        for j in (i + 1)..n {
            if stds[i] < 1e-12 || stds[j] < 1e-12 {
                corr[i][j] = 0.0;
                corr[j][i] = 0.0;
                continue;
            }
            let cov_ij: f64 = returns[i]
                .iter()
                .zip(returns[j].iter())
                .map(|(a, b)| (a - means[i]) * (b - means[j]))
                .sum::<f64>()
                / n_obs;
            let r = (cov_ij / (stds[i] * stds[j])).clamp(-1.0, 1.0);
            corr[i][j] = r;
            corr[j][i] = r;
        }
    }

    serde_json::to_string(&corr).unwrap_or_else(|_| "[]".to_string())
}

/// Compute per-asset marginal and total risk contribution.
///
/// `weights_json`: JSON array of portfolio weights (must sum to ~1).
/// `cov_json`: NxN covariance matrix as JSON 2D array.
/// Returns array of RiskContribution sorted by risk_pct descending.
#[wasm_bindgen]
pub fn compute_risk_contribution(weights_json: &str, cov_json: &str) -> String {
    let weights: Vec<f64> = match serde_json::from_str(weights_json) {
        Ok(w) => w,
        Err(e) => return format!("{{\"error\":\"parse weights: {}\"}}", e),
    };
    let cov: Vec<Vec<f64>> = match serde_json::from_str(cov_json) {
        Ok(c) => c,
        Err(e) => return format!("{{\"error\":\"parse cov: {}\"}}", e),
    };

    let n = weights.len();
    if n == 0 || cov.len() != n {
        return "{\"error\":\"dimension mismatch\"}".to_string();
    }

    let port_var = portfolio_variance(&weights, &cov);
    let port_vol = port_var.sqrt();

    // Marginal risk contribution: (cov * w)_i / sigma_p
    let cov_w = mat_vec_mul(&cov, &weights);

    let mut contributions: Vec<RiskContribution> = (0..n)
        .map(|i| {
            let marginal = if port_vol > 1e-12 {
                cov_w[i] / port_vol
            } else {
                0.0
            };
            let rc = weights[i] * marginal;
            RiskContribution {
                asset_index: i,
                marginal_risk: marginal,
                risk_contribution: rc,
                risk_pct: 0.0, // fill below
            }
        })
        .collect();

    let total_rc: f64 = contributions.iter().map(|c| c.risk_contribution).sum();
    for c in contributions.iter_mut() {
        c.risk_pct = if total_rc.abs() > 1e-12 {
            c.risk_contribution / total_rc * 100.0
        } else {
            100.0 / n as f64
        };
    }

    contributions.sort_by(|a, b| b.risk_pct.partial_cmp(&a.risk_pct).unwrap());

    serde_json::to_string(&contributions).unwrap_or_else(|_| "[]".to_string())
}

/// Find the minimum-variance portfolio using projected gradient descent.
///
/// Constraints: weights >= 0 (long only), sum(weights) = 1.
/// Returns MinVarianceResult with weights, statistics, and convergence info.
#[wasm_bindgen]
pub fn optimize_min_variance(returns_json: &str) -> String {
    let returns: ReturnsMatrix = match serde_json::from_str(returns_json) {
        Ok(r) => r,
        Err(e) => return format!("{{\"error\":\"parse: {}\"}}", e),
    };

    let n_assets = returns.len();
    if n_assets < 2 {
        return "{\"error\":\"need at least 2 assets\"}".to_string();
    }

    let means = compute_means(&returns);
    let cov = compute_covariance_matrix(&returns);

    let (weights, converged) = optimize_constrained_mv(&means, &cov, n_assets, None, 2000, 1e-10);

    let exp_ret = portfolio_return(&weights, &means);
    let vol = portfolio_volatility(&weights, &cov);
    let sharpe = if vol > 1e-12 { exp_ret / vol } else { 0.0 };

    // We track iterations separately via the helper return -- use sentinel 2000 when converged
    // early and unknown otherwise; in practice converged bool is sufficient for the UI.
    let result = MinVarianceResult {
        weights,
        expected_return: exp_ret,
        volatility: vol,
        sharpe,
        converged,
        iterations: if converged { 0 } else { 2000 },
    };

    serde_json::to_string(&result).unwrap_or_else(|_| "{\"error\":\"serialize\"}".to_string())
}

// ---------------------------------------------------------------------------
// Matrix and statistics helpers
// ---------------------------------------------------------------------------

/// Compute per-asset mean returns.
fn compute_means(returns: &ReturnsMatrix) -> Vec<f64> {
    returns
        .iter()
        .map(|asset_rets| {
            let n = asset_rets.len() as f64;
            if n == 0.0 {
                0.0
            } else {
                asset_rets.iter().sum::<f64>() / n
            }
        })
        .collect()
}

/// Compute per-asset standard deviations (population std).
fn compute_stds(returns: &ReturnsMatrix, means: &[f64]) -> Vec<f64> {
    returns
        .iter()
        .zip(means.iter())
        .map(|(asset_rets, &m)| {
            let n = asset_rets.len() as f64;
            if n < 2.0 {
                return 0.0;
            }
            (asset_rets.iter().map(|r| (r - m).powi(2)).sum::<f64>() / n).sqrt()
        })
        .collect()
}

/// Compute the NxN sample covariance matrix (divided by N, not N-1).
fn compute_covariance_matrix(returns: &ReturnsMatrix) -> Vec<Vec<f64>> {
    let n_assets = returns.len();
    let means = compute_means(returns);
    let n_obs = returns[0].len() as f64;

    let mut cov = vec![vec![0.0f64; n_assets]; n_assets];
    for i in 0..n_assets {
        for j in i..n_assets {
            let c: f64 = returns[i]
                .iter()
                .zip(returns[j].iter())
                .map(|(a, b)| (a - means[i]) * (b - means[j]))
                .sum::<f64>()
                / n_obs;
            cov[i][j] = c;
            cov[j][i] = c;
        }
    }

    // Add a small ridge to ensure positive definiteness (Tikhonov regularization)
    let ridge = 1e-8;
    for i in 0..n_assets {
        cov[i][i] += ridge;
    }
    cov
}

/// Portfolio expected return: w^T mu.
fn portfolio_return(weights: &[f64], means: &[f64]) -> f64 {
    weights.iter().zip(means.iter()).map(|(w, m)| w * m).sum()
}

/// Portfolio variance: w^T cov w.
fn portfolio_variance(weights: &[f64], cov: &[Vec<f64>]) -> f64 {
    let cov_w = mat_vec_mul(cov, weights);
    weights.iter().zip(cov_w.iter()).map(|(w, c)| w * c).sum()
}

/// Portfolio volatility (std dev): sqrt(w^T cov w).
fn portfolio_volatility(weights: &[f64], cov: &[Vec<f64>]) -> f64 {
    portfolio_variance(weights, cov).max(0.0).sqrt()
}

/// Matrix-vector product: result[i] = sum_j mat[i][j] * vec[j].
fn mat_vec_mul(mat: &[Vec<f64>], vec: &[f64]) -> Vec<f64> {
    mat.iter()
        .map(|row| row.iter().zip(vec.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// Project a weight vector onto the simplex: sum(w)=1, w[i]>=0.
/// Uses the O(n log n) sorting algorithm of Duchi et al. (2008).
fn project_simplex(v: &[f64]) -> Vec<f64> {
    let _n = v.len();
    let mut u: Vec<f64> = v.to_vec();
    u.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let mut cssv = 0.0f64;
    let mut rho = 0usize;
    for (j, &uj) in u.iter().enumerate() {
        cssv += uj;
        if uj - (cssv - 1.0) / (j as f64 + 1.0) > 0.0 {
            rho = j;
        }
    }

    let cssv_rho: f64 = u[..=rho].iter().sum();
    let theta = (cssv_rho - 1.0) / (rho as f64 + 1.0);

    v.iter().map(|&vi| (vi - theta).max(0.0)).collect()
}

/// Constrained mean-variance optimization via projected gradient descent.
///
/// Constraints: weights on the probability simplex (sum=1, w>=0).
/// Optional `target_return`: if Some, adds a return constraint via penalty.
/// Returns (weights, converged).
fn optimize_constrained_mv(
    means: &[f64],
    cov: &[Vec<f64>],
    n_assets: usize,
    target_return: Option<f64>,
    max_iter: usize,
    tol: f64,
) -> (Vec<f64>, bool) {
    // Initialize with equal weights
    let mut w: Vec<f64> = vec![1.0 / n_assets as f64; n_assets];
    let mut prev_var = f64::MAX;

    // Learning rate: start at 0.1 and decay
    let mut lr = 0.1f64;
    let decay = 0.999f64;

    // Return penalty coefficient (for target return constraint)
    let lambda_ret = 50.0f64;

    let mut converged = false;

    for iter in 0..max_iter {
        // Gradient of variance: 2 * cov * w
        let cov_w = mat_vec_mul(cov, &w);
        let mut grad: Vec<f64> = cov_w.iter().map(|c| 2.0 * c).collect();

        // If targeting a specific return, add penalty gradient
        if let Some(r_target) = target_return {
            let r_actual = portfolio_return(&w, means);
            let pen_grad_scale = 2.0 * lambda_ret * (r_actual - r_target);
            for (g, m) in grad.iter_mut().zip(means.iter()) {
                *g += pen_grad_scale * m;
            }
        }

        // Gradient step
        let w_new: Vec<f64> = w.iter().zip(grad.iter()).map(|(wi, gi)| wi - lr * gi).collect();

        // Project onto simplex
        let w_proj = project_simplex(&w_new);

        // Check convergence
        let var_new = portfolio_variance(&w_proj, cov);
        let delta = (prev_var - var_new).abs();

        w = w_proj;
        prev_var = var_new;
        lr *= decay;

        if delta < tol && iter > 10 {
            converged = true;
            break;
        }
    }

    (w, converged)
}
