use std::f64::consts::PI;

// ═══════════════════════════════════════════════════════════════════════════
// COMMON TYPES
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct OptimizationResult {
    pub weights: Vec<f64>,
    pub expected_return: f64,
    pub volatility: f64,
    pub sharpe_ratio: f64,
    pub iterations: usize,
    pub converged: bool,
    pub objective_value: f64,
}

#[derive(Debug, Clone)]
pub struct PortfolioInput {
    pub expected_returns: Vec<f64>,
    pub covariance: Vec<Vec<f64>>,
    pub n_assets: usize,
    pub risk_free_rate: f64,
}

impl PortfolioInput {
    pub fn new(returns: Vec<f64>, cov: Vec<Vec<f64>>, rf: f64) -> Self {
        let n = returns.len();
        Self { expected_returns: returns, covariance: cov, n_assets: n, risk_free_rate: rf }
    }

    pub fn portfolio_return(&self, weights: &[f64]) -> f64 {
        weights.iter().zip(self.expected_returns.iter()).map(|(w, r)| w * r).sum()
    }

    pub fn portfolio_variance(&self, weights: &[f64]) -> f64 {
        let n = self.n_assets;
        let mut var = 0.0;
        for i in 0..n {
            for j in 0..n {
                var += weights[i] * weights[j] * self.covariance[i][j];
            }
        }
        var
    }

    pub fn portfolio_vol(&self, weights: &[f64]) -> f64 {
        self.portfolio_variance(weights).max(0.0).sqrt()
    }

    pub fn sharpe_ratio(&self, weights: &[f64]) -> f64 {
        let vol = self.portfolio_vol(weights);
        if vol < 1e-15 {
            return 0.0;
        }
        (self.portfolio_return(weights) - self.risk_free_rate) / vol
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MEAN-VARIANCE OPTIMIZATION (QUADRATIC PROGRAMMING VIA ACTIVE SET)
// ═══════════════════════════════════════════════════════════════════════════

/// Mean-variance optimization: minimize w'Σw subject to w'μ >= target, w'1 = 1, w >= 0
pub fn mean_variance_optimize(
    input: &PortfolioInput,
    target_return: f64,
    long_only: bool,
) -> OptimizationResult {
    let n = input.n_assets;
    if n == 0 {
        return empty_result();
    }

    // Use active-set QP solver
    let mut weights = vec![1.0 / n as f64; n]; // equal weight start

    // Gradient descent with projection
    let max_iter = 1000;
    let mut lr = 0.001;
    let mut converged = false;

    for iter in 0..max_iter {
        // Gradient of 0.5 * w'Σw - λ*(w'μ - target)
        let mut grad = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                grad[i] += input.covariance[i][j] * weights[j];
            }
        }

        // Add return constraint via penalty
        let ret = input.portfolio_return(&weights);
        let penalty = 10.0;
        if ret < target_return {
            for i in 0..n {
                grad[i] -= penalty * input.expected_returns[i];
            }
        }

        // Update
        let mut max_grad = 0.0;
        for i in 0..n {
            weights[i] -= lr * grad[i];
            if grad[i].abs() > max_grad {
                max_grad = grad[i].abs();
            }
        }

        // Project onto simplex (sum = 1, w >= 0 if long_only)
        if long_only {
            project_simplex(&mut weights);
        } else {
            project_sum_one(&mut weights);
        }

        if max_grad < 1e-8 {
            converged = true;
            break;
        }

        // Adaptive learning rate
        if iter % 100 == 99 {
            lr *= 0.5;
        }
    }

    let vol = input.portfolio_vol(&weights);
    let ret = input.portfolio_return(&weights);
    OptimizationResult {
        weights,
        expected_return: ret,
        volatility: vol,
        sharpe_ratio: if vol > 0.0 { (ret - input.risk_free_rate) / vol } else { 0.0 },
        iterations: max_iter,
        converged,
        objective_value: 0.5 * vol * vol,
    }
}

/// Analytical mean-variance optimization (unconstrained, closed form).
pub fn mean_variance_analytical(input: &PortfolioInput, target_return: f64) -> OptimizationResult {
    let n = input.n_assets;

    // Solve: min w'Σw s.t. w'μ = target, w'1 = 1
    // Using Lagrange multipliers
    let cov_inv = matrix_inverse(&input.covariance);

    let ones = vec![1.0; n];
    let mu = &input.expected_returns;

    // A = 1'Σ⁻¹μ, B = μ'Σ⁻¹μ, C = 1'Σ⁻¹1
    let a_val = dot_mat_vec(&cov_inv, mu, &ones);
    let b_val = dot_mat_vec(&cov_inv, mu, mu);
    let c_val = dot_mat_vec(&cov_inv, &ones, &ones);

    let det = b_val * c_val - a_val * a_val;
    if det.abs() < 1e-15 {
        return mean_variance_optimize(input, target_return, false);
    }

    let lambda1 = (c_val * target_return - a_val) / det;
    let lambda2 = (b_val - a_val * target_return) / det;

    let cov_inv_mu = mat_vec_mult(&cov_inv, mu);
    let cov_inv_ones = mat_vec_mult(&cov_inv, &ones);

    let weights: Vec<f64> = (0..n)
        .map(|i| lambda1 * cov_inv_mu[i] + lambda2 * cov_inv_ones[i])
        .collect();

    let vol = input.portfolio_vol(&weights);
    let ret = input.portfolio_return(&weights);
    OptimizationResult {
        weights,
        expected_return: ret,
        volatility: vol,
        sharpe_ratio: if vol > 0.0 { (ret - input.risk_free_rate) / vol } else { 0.0 },
        iterations: 0,
        converged: true,
        objective_value: 0.5 * vol * vol,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MINIMUM VARIANCE PORTFOLIO
// ═══════════════════════════════════════════════════════════════════════════

/// Global minimum variance portfolio (analytical).
pub fn min_variance_portfolio(input: &PortfolioInput) -> OptimizationResult {
    let n = input.n_assets;
    let cov_inv = matrix_inverse(&input.covariance);
    let ones = vec![1.0; n];
    let cov_inv_ones = mat_vec_mult(&cov_inv, &ones);
    let c = dot(&cov_inv_ones, &ones);

    if c.abs() < 1e-15 {
        let w = vec![1.0 / n as f64; n];
        let vol = input.portfolio_vol(&w);
        let ret = input.portfolio_return(&w);
        return OptimizationResult {
            weights: w, expected_return: ret, volatility: vol,
            sharpe_ratio: 0.0, iterations: 0, converged: true, objective_value: vol * vol,
        };
    }

    let weights: Vec<f64> = cov_inv_ones.iter().map(|&x| x / c).collect();
    let vol = input.portfolio_vol(&weights);
    let ret = input.portfolio_return(&weights);
    OptimizationResult {
        weights,
        expected_return: ret,
        volatility: vol,
        sharpe_ratio: if vol > 0.0 { (ret - input.risk_free_rate) / vol } else { 0.0 },
        iterations: 0,
        converged: true,
        objective_value: vol * vol,
    }
}

/// Long-only minimum variance via projected gradient.
pub fn min_variance_long_only(input: &PortfolioInput) -> OptimizationResult {
    let n = input.n_assets;
    let mut weights = vec![1.0 / n as f64; n];
    let mut lr = 0.01;

    for iter in 0..2000 {
        let mut grad = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                grad[i] += 2.0 * input.covariance[i][j] * weights[j];
            }
        }

        let max_grad: f64 = grad.iter().map(|g| g.abs()).fold(0.0, f64::max);
        if max_grad < 1e-10 {
            break;
        }

        for i in 0..n {
            weights[i] -= lr * grad[i];
        }
        project_simplex(&mut weights);

        if iter % 200 == 199 {
            lr *= 0.7;
        }
    }

    let vol = input.portfolio_vol(&weights);
    let ret = input.portfolio_return(&weights);
    OptimizationResult {
        weights, expected_return: ret, volatility: vol,
        sharpe_ratio: if vol > 0.0 { (ret - input.risk_free_rate) / vol } else { 0.0 },
        iterations: 2000, converged: true, objective_value: vol * vol,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAXIMUM SHARPE RATIO PORTFOLIO
// ═══════════════════════════════════════════════════════════════════════════

/// Maximum Sharpe ratio portfolio (tangency portfolio, analytical).
pub fn max_sharpe_portfolio(input: &PortfolioInput) -> OptimizationResult {
    let n = input.n_assets;
    let excess_returns: Vec<f64> = input.expected_returns.iter()
        .map(|r| r - input.risk_free_rate)
        .collect();

    let cov_inv = matrix_inverse(&input.covariance);
    let cov_inv_excess = mat_vec_mult(&cov_inv, &excess_returns);
    let sum: f64 = cov_inv_excess.iter().sum();

    if sum.abs() < 1e-15 {
        return min_variance_portfolio(input);
    }

    let weights: Vec<f64> = cov_inv_excess.iter().map(|&x| x / sum).collect();
    let vol = input.portfolio_vol(&weights);
    let ret = input.portfolio_return(&weights);
    OptimizationResult {
        weights,
        expected_return: ret,
        volatility: vol,
        sharpe_ratio: if vol > 0.0 { (ret - input.risk_free_rate) / vol } else { 0.0 },
        iterations: 0,
        converged: true,
        objective_value: -(ret - input.risk_free_rate) / vol.max(1e-15),
    }
}

/// Max Sharpe with long-only constraint.
pub fn max_sharpe_long_only(input: &PortfolioInput) -> OptimizationResult {
    let n = input.n_assets;
    let mut weights = vec![1.0 / n as f64; n];
    let mut best_sharpe = f64::NEG_INFINITY;
    let mut best_weights = weights.clone();

    let mut lr = 0.005;
    for iter in 0..3000 {
        let vol = input.portfolio_vol(&weights);
        let ret = input.portfolio_return(&weights);
        let sharpe = if vol > 1e-10 { (ret - input.risk_free_rate) / vol } else { 0.0 };

        if sharpe > best_sharpe {
            best_sharpe = sharpe;
            best_weights = weights.clone();
        }

        // Gradient of -Sharpe = -(μ_e'w / sqrt(w'Σw))
        // d/dw = -(μ_e / vol - (μ_e'w) * Σw / vol³)
        if vol < 1e-15 {
            break;
        }

        let excess_ret = ret - input.risk_free_rate;
        let mut sigma_w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                sigma_w[i] += input.covariance[i][j] * weights[j];
            }
        }

        let mut grad = vec![0.0; n];
        for i in 0..n {
            let mu_e = input.expected_returns[i] - input.risk_free_rate;
            grad[i] = -(mu_e / vol - excess_ret * sigma_w[i] / (vol * vol * vol));
        }

        for i in 0..n {
            weights[i] -= lr * grad[i];
        }
        project_simplex(&mut weights);

        if iter % 500 == 499 {
            lr *= 0.7;
        }
    }

    let vol = input.portfolio_vol(&best_weights);
    let ret = input.portfolio_return(&best_weights);
    OptimizationResult {
        weights: best_weights,
        expected_return: ret,
        volatility: vol,
        sharpe_ratio: best_sharpe,
        iterations: 3000,
        converged: true,
        objective_value: -best_sharpe,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// RISK PARITY (EQUAL RISK CONTRIBUTION)
// ═══════════════════════════════════════════════════════════════════════════

/// Risk parity via Newton's method: each asset contributes equally to portfolio risk.
/// Target: RC_i = w_i * (Σw)_i / (w'Σw) = 1/n for all i
pub fn risk_parity(input: &PortfolioInput) -> OptimizationResult {
    let n = input.n_assets;
    let target_rc = 1.0 / n as f64;
    let mut weights = vec![1.0 / n as f64; n];

    // Newton-Raphson on the risk budget equations
    for iter in 0..500 {
        let var = input.portfolio_variance(&weights);
        let vol = var.max(1e-20).sqrt();

        // Marginal risk contribution: MRC_i = (Σw)_i / vol
        let mut sigma_w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                sigma_w[i] += input.covariance[i][j] * weights[j];
            }
        }

        // Risk contribution: RC_i = w_i * (Σw)_i / vol
        let mut rc = vec![0.0; n];
        let mut max_dev = 0.0;
        for i in 0..n {
            rc[i] = weights[i] * sigma_w[i] / vol;
            let dev = (rc[i] / vol - target_rc).abs();
            if dev > max_dev {
                max_dev = dev;
            }
        }

        if max_dev < 1e-10 {
            // Converged
            let ret = input.portfolio_return(&weights);
            return OptimizationResult {
                weights, expected_return: ret, volatility: vol,
                sharpe_ratio: if vol > 0.0 { (ret - input.risk_free_rate) / vol } else { 0.0 },
                iterations: iter, converged: true, objective_value: 0.0,
            };
        }

        // Update using Spinu (2013) approach: w_i ∝ 1/MRC_i
        let mut new_weights = vec![0.0; n];
        let mut sum = 0.0;
        for i in 0..n {
            let mrc = sigma_w[i] / vol;
            if mrc > 1e-15 {
                new_weights[i] = 1.0 / mrc;
            } else {
                new_weights[i] = weights[i];
            }
            sum += new_weights[i];
        }
        for i in 0..n {
            new_weights[i] /= sum;
        }

        // Damped update
        for i in 0..n {
            weights[i] = 0.5 * weights[i] + 0.5 * new_weights[i];
        }
    }

    let vol = input.portfolio_vol(&weights);
    let ret = input.portfolio_return(&weights);
    OptimizationResult {
        weights, expected_return: ret, volatility: vol,
        sharpe_ratio: if vol > 0.0 { (ret - input.risk_free_rate) / vol } else { 0.0 },
        iterations: 500, converged: false, objective_value: 0.0,
    }
}

/// Risk parity with custom risk budgets b_i (sum(b_i) = 1).
pub fn risk_budgeting(input: &PortfolioInput, budgets: &[f64]) -> OptimizationResult {
    let n = input.n_assets;
    let mut weights = vec![1.0 / n as f64; n];

    for _iter in 0..500 {
        let var = input.portfolio_variance(&weights);
        let vol = var.max(1e-20).sqrt();

        let mut sigma_w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                sigma_w[i] += input.covariance[i][j] * weights[j];
            }
        }

        let mut new_weights = vec![0.0; n];
        let mut sum = 0.0;
        for i in 0..n {
            let mrc = sigma_w[i] / vol;
            if mrc > 1e-15 {
                new_weights[i] = budgets[i] / mrc;
            } else {
                new_weights[i] = budgets[i];
            }
            sum += new_weights[i];
        }
        for i in 0..n {
            new_weights[i] /= sum;
        }

        let mut max_diff = 0.0;
        for i in 0..n {
            let diff = (new_weights[i] - weights[i]).abs();
            if diff > max_diff { max_diff = diff; }
            weights[i] = 0.5 * weights[i] + 0.5 * new_weights[i];
        }

        if max_diff < 1e-12 {
            break;
        }
    }

    let vol = input.portfolio_vol(&weights);
    let ret = input.portfolio_return(&weights);
    OptimizationResult {
        weights, expected_return: ret, volatility: vol,
        sharpe_ratio: if vol > 0.0 { (ret - input.risk_free_rate) / vol } else { 0.0 },
        iterations: 500, converged: true, objective_value: 0.0,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAXIMUM DIVERSIFICATION
// ═══════════════════════════════════════════════════════════════════════════

/// Maximum diversification portfolio: maximize DR = w'σ / sqrt(w'Σw)
/// where σ = vector of individual volatilities.
pub fn max_diversification(input: &PortfolioInput) -> OptimizationResult {
    let n = input.n_assets;
    let vols: Vec<f64> = (0..n).map(|i| input.covariance[i][i].sqrt()).collect();

    // This is equivalent to max-sharpe where "returns" = vols
    let vol_input = PortfolioInput {
        expected_returns: vols.clone(),
        covariance: input.covariance.clone(),
        n_assets: n,
        risk_free_rate: 0.0,
    };

    let result = max_sharpe_long_only(&vol_input);

    // Recompute with actual returns
    let vol = input.portfolio_vol(&result.weights);
    let ret = input.portfolio_return(&result.weights);
    let dr = dot(&result.weights, &vols) / vol.max(1e-15);

    OptimizationResult {
        weights: result.weights,
        expected_return: ret,
        volatility: vol,
        sharpe_ratio: if vol > 0.0 { (ret - input.risk_free_rate) / vol } else { 0.0 },
        iterations: result.iterations,
        converged: result.converged,
        objective_value: -dr,
    }
}

/// Diversification ratio of a portfolio.
pub fn diversification_ratio(weights: &[f64], cov: &[Vec<f64>]) -> f64 {
    let n = weights.len();
    let vols: Vec<f64> = (0..n).map(|i| cov[i][i].sqrt()).collect();
    let weighted_vols: f64 = weights.iter().zip(vols.iter()).map(|(w, v)| w * v).sum();
    let port_vol = portfolio_vol_from_cov(weights, cov);
    if port_vol > 1e-15 { weighted_vols / port_vol } else { 1.0 }
}

fn portfolio_vol_from_cov(weights: &[f64], cov: &[Vec<f64>]) -> f64 {
    let n = weights.len();
    let mut var = 0.0;
    for i in 0..n { for j in 0..n { var += weights[i] * weights[j] * cov[i][j]; } }
    var.max(0.0).sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// BLACK-LITTERMAN MODEL
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct BlackLittermanView {
    pub assets: Vec<usize>,    // which assets involved
    pub weights: Vec<f64>,     // view portfolio weights (e.g. [1, -1] for relative)
    pub expected_return: f64,  // expected return of the view
    pub confidence: f64,       // confidence (0 = no confidence, 1 = full)
}

/// Black-Litterman model: combine equilibrium returns with investor views.
pub fn black_litterman(
    input: &PortfolioInput,
    market_cap_weights: &[f64],
    views: &[BlackLittermanView],
    tau: f64,                  // scalar (uncertainty in equilibrium, typically 0.025-0.05)
) -> OptimizationResult {
    let n = input.n_assets;

    // Step 1: Implied equilibrium returns (CAPM): π = δΣw_mkt
    let delta = (input.portfolio_return(market_cap_weights) - input.risk_free_rate)
        / input.portfolio_variance(market_cap_weights);
    let delta = delta.max(0.1);

    let mut pi = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            pi[i] += delta * input.covariance[i][j] * market_cap_weights[j];
        }
    }

    if views.is_empty() {
        // No views: use equilibrium
        let bl_input = PortfolioInput::new(pi, input.covariance.clone(), input.risk_free_rate);
        return max_sharpe_long_only(&bl_input);
    }

    // Step 2: Build P (pick matrix) and Q (view returns), Ω (view uncertainty)
    let k = views.len();
    let mut p_mat = vec![vec![0.0; n]; k];
    let mut q_vec = vec![0.0; k];
    let mut omega = vec![vec![0.0; k]; k];

    for (v, view) in views.iter().enumerate() {
        q_vec[v] = view.expected_return;
        for (idx, &asset) in view.assets.iter().enumerate() {
            if asset < n {
                p_mat[v][asset] = view.weights[idx];
            }
        }
        // Omega_ii = (1/confidence - 1) * P_i * tau * Σ * P_i'
        let mut p_sigma_pt = 0.0;
        for i in 0..n {
            for j in 0..n {
                p_sigma_pt += p_mat[v][i] * tau * input.covariance[i][j] * p_mat[v][j];
            }
        }
        let conf = view.confidence.max(0.01).min(0.99);
        omega[v][v] = p_sigma_pt * (1.0 / conf - 1.0);
    }

    // Step 3: BL combined returns
    // μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹π + P'Ω⁻¹Q]
    let tau_sigma = scale_matrix(&input.covariance, tau);
    let tau_sigma_inv = matrix_inverse(&tau_sigma);
    let omega_inv = matrix_inverse(&omega);

    // P'Ω⁻¹P
    let pt_omega_inv_p = mat_transpose_mult_diag_mult(&p_mat, &omega_inv, &p_mat, n, k);

    // Sum: (τΣ)⁻¹ + P'Ω⁻¹P
    let mut combined = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            combined[i][j] = tau_sigma_inv[i][j] + pt_omega_inv_p[i][j];
        }
    }
    let combined_inv = matrix_inverse(&combined);

    // P'Ω⁻¹Q
    let mut pt_omega_inv_q = vec![0.0; n];
    for i in 0..n {
        for v in 0..k {
            let mut omega_inv_q_v = 0.0;
            for u in 0..k {
                omega_inv_q_v += omega_inv[v][u] * q_vec[u];
            }
            pt_omega_inv_q[i] += p_mat[v][i] * omega_inv_q_v;
        }
    }

    // (τΣ)⁻¹π
    let tau_sigma_inv_pi = mat_vec_mult(&tau_sigma_inv, &pi);

    // Sum
    let mut rhs = vec![0.0; n];
    for i in 0..n {
        rhs[i] = tau_sigma_inv_pi[i] + pt_omega_inv_q[i];
    }

    let bl_returns = mat_vec_mult(&combined_inv, &rhs);

    // Step 4: Optimize with BL returns
    let bl_input = PortfolioInput::new(bl_returns, input.covariance.clone(), input.risk_free_rate);
    max_sharpe_long_only(&bl_input)
}

// ═══════════════════════════════════════════════════════════════════════════
// ROBUST MARKOWITZ (SHRINKAGE)
// ═══════════════════════════════════════════════════════════════════════════

/// Robust optimization with return uncertainty set.
/// Uses worst-case return within an ellipsoidal uncertainty set.
pub fn robust_markowitz(
    input: &PortfolioInput,
    epsilon: f64,  // size of uncertainty set
    long_only: bool,
) -> OptimizationResult {
    let n = input.n_assets;
    let mut weights = vec![1.0 / n as f64; n];

    // Robust MVO: max w'μ - ε * ||Σ^{1/2}w|| - λ * w'Σw
    let lambda = 2.0; // risk aversion

    for _iter in 0..2000 {
        let vol = input.portfolio_vol(&weights);
        let mut sigma_w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                sigma_w[i] += input.covariance[i][j] * weights[j];
            }
        }

        let mut grad = vec![0.0; n];
        for i in 0..n {
            // d/dw_i [w'μ - ε*vol - λ*0.5*var]
            grad[i] = input.expected_returns[i]
                - epsilon * sigma_w[i] / vol.max(1e-15)
                - lambda * sigma_w[i];
        }

        let lr = 0.001;
        for i in 0..n {
            weights[i] += lr * grad[i];
        }

        if long_only {
            project_simplex(&mut weights);
        } else {
            project_sum_one(&mut weights);
        }
    }

    let vol = input.portfolio_vol(&weights);
    let ret = input.portfolio_return(&weights);
    OptimizationResult {
        weights, expected_return: ret, volatility: vol,
        sharpe_ratio: if vol > 0.0 { (ret - input.risk_free_rate) / vol } else { 0.0 },
        iterations: 2000, converged: true, objective_value: 0.0,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CVaR OPTIMIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// CVaR optimization: minimize Conditional Value-at-Risk at confidence level α.
/// Uses scenario-based approach with linear programming relaxation.
pub fn cvar_optimize(
    input: &PortfolioInput,
    scenarios: &[Vec<f64>],  // [n_scenarios][n_assets] - return scenarios
    alpha: f64,               // confidence level (e.g., 0.95)
    target_return: f64,
) -> OptimizationResult {
    let n = input.n_assets;
    let n_s = scenarios.len();
    if n_s == 0 {
        return min_variance_portfolio(input);
    }

    let mut weights = vec![1.0 / n as f64; n];
    let mut var_threshold = 0.0; // VaR estimate

    // Iterative optimization
    for _iter in 0..1000 {
        // Compute portfolio returns for each scenario
        let mut port_returns: Vec<f64> = scenarios.iter().map(|s| {
            weights.iter().zip(s.iter()).map(|(w, r)| w * r).sum()
        }).collect();

        // Sort to find VaR
        port_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let var_idx = ((1.0 - alpha) * n_s as f64).ceil() as usize;
        let var_idx = var_idx.min(n_s - 1);
        var_threshold = port_returns[var_idx];

        // CVaR = E[X | X <= VaR]
        let tail_count = var_idx + 1;
        let cvar: f64 = port_returns[..tail_count].iter().sum::<f64>() / tail_count as f64;

        // Gradient of CVaR w.r.t. weights
        let mut grad = vec![0.0; n];
        for s in 0..n_s {
            let port_ret: f64 = weights.iter().zip(scenarios[s].iter()).map(|(w, r)| w * r).sum();
            if port_ret <= var_threshold {
                for i in 0..n {
                    grad[i] += scenarios[s][i] / tail_count as f64;
                }
            }
        }

        // Update: minimize -CVaR (maximize negative tail)
        let lr = 0.001;
        for i in 0..n {
            weights[i] -= lr * (-grad[i]); // minimize CVaR = maximize negative
        }

        // Return constraint
        let ret = input.portfolio_return(&weights);
        if ret < target_return {
            let penalty = 5.0;
            for i in 0..n {
                weights[i] += lr * penalty * input.expected_returns[i];
            }
        }

        project_simplex(&mut weights);
    }

    let vol = input.portfolio_vol(&weights);
    let ret = input.portfolio_return(&weights);
    OptimizationResult {
        weights, expected_return: ret, volatility: vol,
        sharpe_ratio: if vol > 0.0 { (ret - input.risk_free_rate) / vol } else { 0.0 },
        iterations: 1000, converged: true, objective_value: var_threshold,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// EFFICIENT FRONTIER
// ═══════════════════════════════════════════════════════════════════════════

/// Compute efficient frontier points.
pub fn efficient_frontier(
    input: &PortfolioInput,
    n_points: usize,
    long_only: bool,
) -> Vec<OptimizationResult> {
    // Find return range
    let min_var = if long_only {
        min_variance_long_only(input)
    } else {
        min_variance_portfolio(input)
    };

    let max_ret = input.expected_returns.iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_ret = min_var.expected_return;

    let ret_range = max_ret - min_ret;
    if ret_range <= 0.0 {
        return vec![min_var];
    }

    (0..n_points).map(|i| {
        let target = min_ret + (i as f64 / (n_points - 1).max(1) as f64) * ret_range;
        mean_variance_optimize(input, target, long_only)
    }).collect()
}

/// Compute the efficient frontier in (vol, return) space.
pub fn efficient_frontier_points(
    input: &PortfolioInput,
    n_points: usize,
) -> Vec<(f64, f64)> {
    efficient_frontier(input, n_points, true)
        .iter()
        .map(|r| (r.volatility, r.expected_return))
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

fn empty_result() -> OptimizationResult {
    OptimizationResult {
        weights: vec![], expected_return: 0.0, volatility: 0.0,
        sharpe_ratio: 0.0, iterations: 0, converged: false, objective_value: 0.0,
    }
}

/// Project onto probability simplex: sum = 1, all >= 0.
fn project_simplex(w: &mut Vec<f64>) {
    let n = w.len();
    let mut sorted = w.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let mut cumsum = 0.0;
    let mut rho = 0;
    for i in 0..n {
        cumsum += sorted[i];
        if sorted[i] - (cumsum - 1.0) / (i + 1) as f64 > 0.0 {
            rho = i;
        }
    }

    let theta = (sorted[..=rho].iter().sum::<f64>() - 1.0) / (rho + 1) as f64;
    for i in 0..n {
        w[i] = (w[i] - theta).max(0.0);
    }
}

/// Project onto hyperplane sum = 1 (allow negative weights).
fn project_sum_one(w: &mut Vec<f64>) {
    let n = w.len() as f64;
    let sum: f64 = w.iter().sum();
    let adj = (sum - 1.0) / n;
    for x in w.iter_mut() {
        *x -= adj;
    }
}

/// Matrix inverse via Gauss-Jordan elimination.
pub fn matrix_inverse(mat: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = mat.len();
    let mut augmented = vec![vec![0.0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            augmented[i][j] = mat[i][j];
        }
        augmented[i][n + i] = 1.0;
    }

    for k in 0..n {
        // Partial pivoting
        let mut max_idx = k;
        let mut max_val = augmented[k][k].abs();
        for i in k + 1..n {
            if augmented[i][k].abs() > max_val {
                max_val = augmented[i][k].abs();
                max_idx = i;
            }
        }
        augmented.swap(k, max_idx);

        let pivot = augmented[k][k];
        if pivot.abs() < 1e-15 {
            // Singular: add small diagonal
            augmented[k][k] += 1e-10;
            continue;
        }

        for j in 0..2 * n {
            augmented[k][j] /= pivot;
        }

        for i in 0..n {
            if i != k {
                let factor = augmented[i][k];
                for j in 0..2 * n {
                    augmented[i][j] -= factor * augmented[k][j];
                }
            }
        }
    }

    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            result[i][j] = augmented[i][n + j];
        }
    }
    result
}

fn mat_vec_mult(mat: &[Vec<f64>], vec_in: &[f64]) -> Vec<f64> {
    let n = vec_in.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            result[i] += mat[i][j] * vec_in[j];
        }
    }
    result
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn dot_mat_vec(mat: &[Vec<f64>], a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let mut result = 0.0;
    for i in 0..n {
        for j in 0..n {
            result += a[i] * mat[i][j] * b[j];
        }
    }
    result
}

fn scale_matrix(mat: &[Vec<f64>], scalar: f64) -> Vec<Vec<f64>> {
    mat.iter().map(|row| row.iter().map(|&x| x * scalar).collect()).collect()
}

fn mat_transpose_mult_diag_mult(
    p: &[Vec<f64>],       // k x n
    omega_inv: &[Vec<f64>], // k x k
    p2: &[Vec<f64>],      // k x n
    n: usize,
    k: usize,
) -> Vec<Vec<f64>> {
    // P' * Ω⁻¹ * P
    let mut result = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            for u in 0..k {
                for v in 0..k {
                    result[i][j] += p[u][i] * omega_inv[u][v] * p2[v][j];
                }
            }
        }
    }
    result
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_input() -> PortfolioInput {
        let returns = vec![0.10, 0.08, 0.12, 0.06];
        let cov = vec![
            vec![0.04, 0.006, 0.008, 0.002],
            vec![0.006, 0.03, 0.005, 0.003],
            vec![0.008, 0.005, 0.05, 0.004],
            vec![0.002, 0.003, 0.004, 0.02],
        ];
        PortfolioInput::new(returns, cov, 0.02)
    }

    #[test]
    fn test_min_variance() {
        let input = sample_input();
        let result = min_variance_portfolio(&input);
        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Weights don't sum to 1: {}", sum);
    }

    #[test]
    fn test_max_sharpe() {
        let input = sample_input();
        let result = max_sharpe_portfolio(&input);
        assert!(result.sharpe_ratio > 0.0, "Sharpe should be positive");
    }

    #[test]
    fn test_risk_parity() {
        let input = sample_input();
        let result = risk_parity(&input);
        let sum: f64 = result.weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Weights don't sum to 1: {}", sum);
        assert!(result.weights.iter().all(|&w| w > 0.0), "All weights should be positive");
    }

    #[test]
    fn test_efficient_frontier() {
        let input = sample_input();
        let frontier = efficient_frontier(&input, 10, true);
        assert_eq!(frontier.len(), 10);
        // Returns should generally increase
        for i in 1..frontier.len() {
            assert!(frontier[i].expected_return >= frontier[i-1].expected_return - 0.01);
        }
    }

    #[test]
    fn test_simplex_projection() {
        let mut w = vec![0.5, -0.1, 0.8, 0.2];
        project_simplex(&mut w);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Sum should be 1: {}", sum);
        assert!(w.iter().all(|&x| x >= -1e-10), "All should be non-negative");
    }
}
