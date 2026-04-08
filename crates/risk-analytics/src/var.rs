// var.rs — Value at Risk: historical, parametric, Monte Carlo, component, marginal, EVT, backtesting

use quant_math::distributions::{norm_ppf, norm_cdf, norm_pdf, student_t_ppf, Xoshiro256PlusPlus};
use quant_math::statistics;

/// Historical VaR at given confidence level (e.g., 0.99)
pub fn historical_var(returns: &[f64], confidence: f64) -> f64 {
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
    let idx = idx.min(sorted.len() - 1);
    -sorted[idx]
}

/// Historical CVaR (Expected Shortfall)
pub fn historical_cvar(returns: &[f64], confidence: f64) -> f64 {
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cutoff = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
    let cutoff = cutoff.max(1);
    let tail = &sorted[..cutoff];
    -tail.iter().sum::<f64>() / cutoff as f64
}

/// Parametric VaR (normal distribution assumption)
pub fn parametric_var_normal(returns: &[f64], confidence: f64) -> f64 {
    let mu = statistics::mean(returns);
    let sigma = statistics::std_dev(returns);
    let z = norm_ppf(1.0 - confidence);
    -(mu + z * sigma)
}

/// Parametric CVaR (normal)
pub fn parametric_cvar_normal(returns: &[f64], confidence: f64) -> f64 {
    let mu = statistics::mean(returns);
    let sigma = statistics::std_dev(returns);
    let z = norm_ppf(1.0 - confidence);
    let phi_z = norm_pdf(z);
    -(mu - sigma * phi_z / (1.0 - confidence))
}

/// Parametric VaR (Student-t)
pub fn parametric_var_student_t(returns: &[f64], confidence: f64, df: f64) -> f64 {
    let mu = statistics::mean(returns);
    let sigma = statistics::std_dev(returns);
    let t_q = student_t_ppf(1.0 - confidence, df);
    // Scale: sigma * sqrt((df-2)/df) for student-t
    let scale = sigma * ((df - 2.0) / df).sqrt();
    -(mu + t_q * scale)
}

/// Cornish-Fisher VaR (adjusts normal VaR for skewness and kurtosis)
pub fn cornish_fisher_var(returns: &[f64], confidence: f64) -> f64 {
    let mu = statistics::mean(returns);
    let sigma = statistics::std_dev(returns);
    let s = statistics::skewness(returns);
    let k = statistics::kurtosis(returns);
    let z = norm_ppf(1.0 - confidence);

    // Cornish-Fisher expansion
    let z_cf = z + (z * z - 1.0) * s / 6.0
        + (z.powi(3) - 3.0 * z) * k / 24.0
        - (2.0 * z.powi(3) - 5.0 * z) * s * s / 36.0;

    -(mu + z_cf * sigma)
}

/// Monte Carlo VaR using normal simulation
pub fn monte_carlo_var_normal(
    returns: &[f64], confidence: f64, n_simulations: usize, horizon: usize, seed: u64,
) -> f64 {
    let mu = statistics::mean(returns);
    let sigma = statistics::std_dev(returns);
    let mut rng = Xoshiro256PlusPlus::new(seed);

    let mut simulated_losses = Vec::with_capacity(n_simulations);
    for _ in 0..n_simulations {
        let mut cumulative = 0.0;
        for _ in 0..horizon {
            cumulative += mu + sigma * rng.normal();
        }
        simulated_losses.push(-cumulative);
    }

    simulated_losses.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((1.0 - confidence) * n_simulations as f64).floor() as usize;
    simulated_losses[idx.min(n_simulations - 1)]
}

/// Monte Carlo VaR using historical bootstrap
pub fn monte_carlo_var_bootstrap(
    returns: &[f64], confidence: f64, n_simulations: usize, horizon: usize, seed: u64,
) -> f64 {
    let n = returns.len();
    let mut rng = Xoshiro256PlusPlus::new(seed);
    let mut simulated_losses = Vec::with_capacity(n_simulations);

    for _ in 0..n_simulations {
        let mut cumulative = 0.0;
        for _ in 0..horizon {
            let idx = (rng.next_u64() as usize) % n;
            cumulative += returns[idx];
        }
        simulated_losses.push(-cumulative);
    }

    simulated_losses.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((1.0 - confidence) * n_simulations as f64).floor() as usize;
    simulated_losses[idx.min(n_simulations - 1)]
}

/// Monte Carlo VaR with GARCH volatility
pub fn monte_carlo_var_garch(
    returns: &[f64], confidence: f64, n_simulations: usize, horizon: usize,
    omega: f64, alpha: f64, beta: f64, seed: u64,
) -> f64 {
    let mu = statistics::mean(returns);
    let n = returns.len();
    // Estimate last sigma^2
    let mut sigma2 = statistics::variance(returns);
    for i in 1..n {
        let r = returns[i] - mu;
        sigma2 = omega + alpha * r * r + beta * sigma2;
    }

    let mut rng = Xoshiro256PlusPlus::new(seed);
    let mut simulated_losses = Vec::with_capacity(n_simulations);

    for _ in 0..n_simulations {
        let mut s2 = sigma2;
        let mut cumulative = 0.0;
        for _ in 0..horizon {
            let z = rng.normal();
            let r = mu + s2.sqrt() * z;
            cumulative += r;
            s2 = omega + alpha * (r - mu).powi(2) + beta * s2;
        }
        simulated_losses.push(-cumulative);
    }

    simulated_losses.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((1.0 - confidence) * n_simulations as f64).floor() as usize;
    simulated_losses[idx.min(n_simulations - 1)]
}

/// Component VaR for a portfolio
/// weights: portfolio weights, cov: covariance matrix
pub fn component_var(weights: &[f64], cov: &[Vec<f64>], confidence: f64) -> Vec<f64> {
    let n = weights.len();
    let z = -norm_ppf(1.0 - confidence);

    // Portfolio variance
    let mut port_var = 0.0;
    for i in 0..n {
        for j in 0..n {
            port_var += weights[i] * weights[j] * cov[i][j];
        }
    }
    let port_sigma = port_var.sqrt();
    let portfolio_var = z * port_sigma;

    // Marginal contribution: ∂VaR/∂wᵢ = z * (Σw)ᵢ / σₚ
    let mut sigma_w = vec![0.0; n];
    for i in 0..n {
        for j in 0..n { sigma_w[i] += cov[i][j] * weights[j]; }
    }

    let mut comp_var = vec![0.0; n];
    for i in 0..n {
        comp_var[i] = weights[i] * z * sigma_w[i] / port_sigma;
    }
    comp_var
}

/// Marginal VaR: how much VaR changes per unit increase in position
pub fn marginal_var(weights: &[f64], cov: &[Vec<f64>], confidence: f64) -> Vec<f64> {
    let n = weights.len();
    let z = -norm_ppf(1.0 - confidence);
    let mut port_var = 0.0;
    for i in 0..n {
        for j in 0..n { port_var += weights[i] * weights[j] * cov[i][j]; }
    }
    let port_sigma = port_var.sqrt();
    let mut sigma_w = vec![0.0; n];
    for i in 0..n {
        for j in 0..n { sigma_w[i] += cov[i][j] * weights[j]; }
    }
    sigma_w.iter().map(|&sw| z * sw / port_sigma).collect()
}

/// Incremental VaR: VaR change from adding a new position
pub fn incremental_var(
    weights: &[f64], cov: &[Vec<f64>], confidence: f64,
    new_asset_cov: &[f64], new_weight: f64,
) -> f64 {
    let n = weights.len();
    let z = -norm_ppf(1.0 - confidence);

    // Current portfolio VaR
    let mut port_var = 0.0;
    for i in 0..n {
        for j in 0..n { port_var += weights[i] * weights[j] * cov[i][j]; }
    }
    let current_var = z * port_var.sqrt();

    // New portfolio with extra asset
    let mut new_var = port_var;
    // Add cross terms
    for i in 0..n {
        new_var += 2.0 * weights[i] * new_weight * new_asset_cov[i];
    }
    // Add own variance (last element of new_asset_cov would be own variance)
    let own_var = if new_asset_cov.len() > n { new_asset_cov[n] } else { 0.01 };
    new_var += new_weight * new_weight * own_var;

    let new_var_val = z * new_var.sqrt();
    new_var_val - current_var
}

/// Stressed VaR: multiply volatility by stress factor
pub fn stressed_var(returns: &[f64], confidence: f64, stress_factor: f64) -> f64 {
    let mu = statistics::mean(returns);
    let sigma = statistics::std_dev(returns) * stress_factor;
    let z = norm_ppf(1.0 - confidence);
    -(mu + z * sigma)
}

/// Conditional VaR with regime detection (2 regimes: low/high vol)
pub fn regime_conditional_var(returns: &[f64], confidence: f64, vol_threshold: f64) -> (f64, f64) {
    let n = returns.len();
    let window = 20.min(n / 4).max(5);
    let mut low_vol_returns = Vec::new();
    let mut high_vol_returns = Vec::new();

    for i in window..n {
        let window_data = &returns[i - window..i];
        let vol = statistics::std_dev(window_data);
        if vol < vol_threshold {
            low_vol_returns.push(returns[i]);
        } else {
            high_vol_returns.push(returns[i]);
        }
    }

    let low_var = if low_vol_returns.len() > 5 {
        historical_var(&low_vol_returns, confidence)
    } else { 0.0 };

    let high_var = if high_vol_returns.len() > 5 {
        historical_var(&high_vol_returns, confidence)
    } else { 0.0 };

    (low_var, high_var)
}

/// EVT VaR using Generalized Pareto Distribution (peaks over threshold)
pub fn evt_var_gpd(returns: &[f64], confidence: f64, threshold_quantile: f64) -> f64 {
    let n = returns.len();
    let losses: Vec<f64> = returns.iter().map(|r| -r).collect();
    let mut sorted = losses.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let threshold_idx = (threshold_quantile * n as f64).floor() as usize;
    let threshold = sorted[threshold_idx.min(n - 1)];

    let exceedances: Vec<f64> = losses.iter()
        .filter(|&&l| l > threshold)
        .map(|&l| l - threshold)
        .collect();

    let n_exceed = exceedances.len();
    if n_exceed < 5 { return historical_var(returns, confidence); }

    // Fit GPD by probability-weighted moments
    let (xi, sigma) = fit_gpd_pwm(&exceedances);

    let nu = n_exceed as f64 / n as f64;
    let p = 1.0 - confidence;

    // VaR = u + (sigma/xi) * ((p/nu)^(-xi) - 1)
    if xi.abs() < 1e-10 {
        threshold - sigma * (p / nu).ln()
    } else {
        threshold + (sigma / xi) * ((p / nu).powf(-xi) - 1.0)
    }
}

/// CVaR from GPD tail
pub fn evt_cvar_gpd(returns: &[f64], confidence: f64, threshold_quantile: f64) -> f64 {
    let var = evt_var_gpd(returns, confidence, threshold_quantile);
    let n = returns.len();
    let losses: Vec<f64> = returns.iter().map(|r| -r).collect();

    let exceedances: Vec<f64> = losses.iter()
        .filter(|&&l| l > var)
        .copied()
        .collect();

    if exceedances.is_empty() { return var; }
    exceedances.iter().sum::<f64>() / exceedances.len() as f64
}

fn fit_gpd_pwm(data: &[f64]) -> (f64, f64) {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len() as f64;
    let b0: f64 = sorted.iter().sum::<f64>() / n;
    let b1: f64 = sorted.iter().enumerate()
        .map(|(i, &x)| i as f64 / (n - 1.0).max(1.0) * x)
        .sum::<f64>() / n;
    if (2.0 * b1 - b0).abs() < 1e-15 { return (0.0, b0); }
    let xi = b0 / (2.0 * b1 - b0) - 2.0;
    let sigma = 2.0 * b0 * b1 / (2.0 * b1 - b0);
    (xi.clamp(-0.5, 2.0), sigma.max(1e-10))
}

// ============================================================
// VaR Backtesting
// ============================================================

/// Kupiec unconditional coverage test
/// Returns (test statistic, p-value, reject at 5%)
pub fn kupiec_test(returns: &[f64], var_estimates: &[f64], confidence: f64) -> (f64, f64, bool) {
    let n = returns.len().min(var_estimates.len());
    let p_expected = 1.0 - confidence;
    let mut violations = 0usize;

    for i in 0..n {
        if returns[i] < -var_estimates[i] {
            violations += 1;
        }
    }

    let p_observed = violations as f64 / n as f64;
    if p_observed < 1e-15 || p_observed > 1.0 - 1e-15 {
        return (0.0, 1.0, false);
    }

    // Likelihood ratio
    let lr = 2.0 * (violations as f64 * (p_observed / p_expected).ln()
        + (n - violations) as f64 * ((1.0 - p_observed) / (1.0 - p_expected)).ln());
    let lr = lr.max(0.0);

    // Chi-squared(1) p-value
    let p_value = 1.0 - chi2_cdf_1(lr);
    (lr, p_value, p_value < 0.05)
}

/// Christoffersen conditional coverage test (tests independence of violations)
pub fn christoffersen_test(returns: &[f64], var_estimates: &[f64], confidence: f64) -> (f64, f64, bool) {
    let n = returns.len().min(var_estimates.len());
    let mut violations: Vec<bool> = Vec::with_capacity(n);
    for i in 0..n {
        violations.push(returns[i] < -var_estimates[i]);
    }

    // Transition counts
    let mut n00 = 0u64; let mut n01 = 0u64;
    let mut n10 = 0u64; let mut n11 = 0u64;
    for i in 1..n {
        match (violations[i - 1], violations[i]) {
            (false, false) => n00 += 1,
            (false, true) => n01 += 1,
            (true, false) => n10 += 1,
            (true, true) => n11 += 1,
        }
    }

    let p01 = if n00 + n01 > 0 { n01 as f64 / (n00 + n01) as f64 } else { 0.0 };
    let p11 = if n10 + n11 > 0 { n11 as f64 / (n10 + n11) as f64 } else { 0.0 };
    let p = (n01 + n11) as f64 / (n - 1) as f64;

    if p < 1e-15 || p > 1.0 - 1e-15 || p01 < 1e-15 || p01 > 1.0 - 1e-15 {
        return (0.0, 1.0, false);
    }

    // Independence LR
    let lr_ind = 2.0 * (
        safe_xlnx(n00 as f64, 1.0 - p01) + safe_xlnx(n01 as f64, p01)
        + safe_xlnx(n10 as f64, 1.0 - p11) + safe_xlnx(n11 as f64, p11)
        - safe_xlnx((n00 + n10) as f64, 1.0 - p) - safe_xlnx((n01 + n11) as f64, p)
    );

    // Unconditional LR
    let (lr_uc, _, _) = kupiec_test(returns, var_estimates, confidence);

    let lr_cc = lr_uc + lr_ind.max(0.0);
    let p_value = 1.0 - chi2_cdf_2(lr_cc);
    (lr_cc, p_value, p_value < 0.05)
}

fn safe_xlnx(x: f64, p: f64) -> f64 {
    if x < 1e-15 || p < 1e-15 { 0.0 } else { x * p.ln() }
}

/// Traffic light backtest (Basel)
pub fn traffic_light_test(violations: usize, n_obs: usize, confidence: f64) -> &'static str {
    let p = 1.0 - confidence;
    let expected = p * n_obs as f64;
    let ratio = violations as f64 / expected;
    if ratio <= 1.5 { "Green" }
    else if ratio <= 2.0 { "Yellow" }
    else { "Red" }
}

/// Rolling VaR calculation
pub fn rolling_var(returns: &[f64], window: usize, confidence: f64) -> Vec<f64> {
    let n = returns.len();
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        if i + 1 < window {
            result.push(f64::NAN);
        } else {
            let window_returns = &returns[i + 1 - window..=i];
            result.push(historical_var(window_returns, confidence));
        }
    }
    result
}

/// Rolling CVaR
pub fn rolling_cvar(returns: &[f64], window: usize, confidence: f64) -> Vec<f64> {
    let n = returns.len();
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        if i + 1 < window {
            result.push(f64::NAN);
        } else {
            let window_returns = &returns[i + 1 - window..=i];
            result.push(historical_cvar(window_returns, confidence));
        }
    }
    result
}

/// Weighted historical simulation VaR (exponentially decaying weights)
pub fn weighted_historical_var(returns: &[f64], confidence: f64, decay: f64) -> f64 {
    let n = returns.len();
    let mut weights = Vec::with_capacity(n);
    let mut sum_w = 0.0;
    for i in 0..n {
        let w = decay.powi((n - 1 - i) as i32);
        weights.push(w);
        sum_w += w;
    }
    for w in &mut weights { *w /= sum_w; }

    // Sort returns with weights
    let mut paired: Vec<(f64, f64)> = returns.iter().copied().zip(weights).collect();
    paired.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Find quantile
    let target = 1.0 - confidence;
    let mut cum_w = 0.0;
    for &(r, w) in &paired {
        cum_w += w;
        if cum_w >= target { return -r; }
    }
    -paired[0].0
}

/// Conditional VaR using DCC-like approach (simplified)
pub fn dcc_var(returns: &[Vec<f64>], weights: &[f64], confidence: f64, decay: f64) -> Vec<f64> {
    let n = returns[0].len();
    let k = returns.len();
    let z = -norm_ppf(1.0 - confidence);

    let mut result = Vec::with_capacity(n);
    let mut variances: Vec<f64> = returns.iter().map(|r| statistics::variance(r)).collect();
    let mut covariances = vec![vec![0.0; k]; k];
    for i in 0..k {
        for j in 0..k {
            covariances[i][j] = statistics::covariance(&returns[i], &returns[j]);
        }
    }

    for t in 1..n {
        // Update variances with EWMA
        for i in 0..k {
            let r = returns[i][t];
            let m = statistics::mean(&returns[i][..=t]);
            let dev = (r - m).powi(2);
            variances[i] = decay * variances[i] + (1.0 - decay) * dev;
        }
        // Update covariances
        for i in 0..k {
            for j in i..k {
                let ri = returns[i][t] - statistics::mean(&returns[i][..=t]);
                let rj = returns[j][t] - statistics::mean(&returns[j][..=t]);
                covariances[i][j] = decay * covariances[i][j] + (1.0 - decay) * ri * rj;
                covariances[j][i] = covariances[i][j];
            }
        }

        // Portfolio variance
        let mut port_var = 0.0;
        for i in 0..k {
            for j in 0..k {
                port_var += weights[i] * weights[j] * covariances[i][j];
            }
        }
        result.push(z * port_var.max(0.0).sqrt());
    }
    result
}

/// Diversified VaR vs undiversified VaR
pub fn diversification_ratio(weights: &[f64], cov: &[Vec<f64>]) -> f64 {
    let n = weights.len();
    // Undiversified: sum of individual VaRs
    let mut undiv = 0.0;
    for i in 0..n {
        undiv += weights[i].abs() * cov[i][i].sqrt();
    }
    // Diversified: portfolio sigma
    let mut port_var = 0.0;
    for i in 0..n {
        for j in 0..n {
            port_var += weights[i] * weights[j] * cov[i][j];
        }
    }
    let div = port_var.sqrt();
    if div > 1e-15 { undiv / div } else { 1.0 }
}

/// Risk contribution (% of total VaR from each asset)
pub fn risk_contribution(weights: &[f64], cov: &[Vec<f64>]) -> Vec<f64> {
    let comp = component_var(weights, cov, 0.99);
    let total: f64 = comp.iter().sum();
    if total.abs() < 1e-15 { return vec![0.0; weights.len()]; }
    comp.iter().map(|c| c / total).collect()
}

/// Parametric VaR for a portfolio with correlated assets
pub fn portfolio_parametric_var(
    weights: &[f64], means: &[f64], cov: &[Vec<f64>], confidence: f64,
) -> f64 {
    let n = weights.len();
    let z = -norm_ppf(1.0 - confidence);
    let port_mean: f64 = weights.iter().zip(means).map(|(w, m)| w * m).sum();
    let mut port_var = 0.0;
    for i in 0..n {
        for j in 0..n {
            port_var += weights[i] * weights[j] * cov[i][j];
        }
    }
    -(port_mean + z * port_var.sqrt())
}

fn chi2_cdf_1(x: f64) -> f64 {
    // P(X <= x) for chi-squared(1) = 2*Phi(sqrt(x)) - 1
    if x <= 0.0 { return 0.0; }
    2.0 * norm_cdf(x.sqrt()) - 1.0
}

fn chi2_cdf_2(x: f64) -> f64 {
    // chi-squared(2) CDF = 1 - exp(-x/2)
    if x <= 0.0 { return 0.0; }
    1.0 - (-x / 2.0).exp()
}

/// VaR scaling (square root of time rule)
pub fn scale_var(var_1day: f64, horizon_days: usize) -> f64 {
    var_1day * (horizon_days as f64).sqrt()
}

/// Filtered historical simulation VaR (GARCH-filtered)
pub fn filtered_historical_var(
    returns: &[f64], confidence: f64,
    omega: f64, alpha: f64, beta: f64,
) -> f64 {
    let n = returns.len();
    let mu = statistics::mean(returns);
    let mut sigma2 = statistics::variance(returns);
    let mut standardized = Vec::with_capacity(n);

    for i in 0..n {
        let r = returns[i] - mu;
        if sigma2 > 1e-20 {
            standardized.push(r / sigma2.sqrt());
        }
        sigma2 = omega + alpha * r * r + beta * sigma2;
    }

    // Current vol forecast
    let current_sigma = sigma2.sqrt();
    // VaR = current_sigma * quantile of standardized residuals
    let var = historical_var(&standardized, confidence) * current_sigma;
    var
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_historical_var() {
        let returns: Vec<f64> = (-50..50).map(|i| i as f64 / 100.0).collect();
        let var = historical_var(&returns, 0.95);
        assert!(var > 0.0);
    }

    #[test]
    fn test_parametric_var() {
        let returns = vec![-0.02, 0.01, -0.01, 0.03, -0.015, 0.005, -0.02, 0.01, -0.005, 0.02];
        let var = parametric_var_normal(&returns, 0.95);
        assert!(var > 0.0);
    }

    #[test]
    fn test_kupiec() {
        let returns = vec![-0.05, 0.01, 0.02, -0.03, 0.01, -0.06, 0.02, -0.01, 0.03, 0.01];
        let vars = vec![0.04; 10];
        let (lr, _pval, _reject) = kupiec_test(&returns, &vars, 0.95);
        assert!(lr >= 0.0);
    }
}
