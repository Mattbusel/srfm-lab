/// Portfolio-level risk metrics.

use crate::covariance::{mat_vec_mul, quad_form, Matrix, Vector};

// ── Normal distribution helpers ───────────────────────────────────────────────

/// Inverse cumulative normal (probit) via rational approximation (Beasley-Springer-Moro).
pub fn inv_normal_cdf(p: f64) -> f64 {
    const A: [f64; 4] = [2.515517, 0.802853, 0.010328, 0.0];
    const B: [f64; 4] = [1.432788, 0.189269, 0.001308, 0.0];
    let p = p.clamp(1e-10, 1.0 - 1e-10);
    let sgn = if p < 0.5 { -1.0 } else { 1.0 };
    let pp = if p < 0.5 { p } else { 1.0 - p };
    let t = (-2.0 * pp.ln()).sqrt();
    let num = A[0] + t * (A[1] + t * A[2]);
    let den = 1.0 + t * (B[0] + t * (B[1] + t * B[2]));
    sgn * (t - num / den)
}

fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    // Abramowitz & Stegun approximation.
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let y = 1.0 - poly * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}

fn mean_f(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn std_f(v: &[f64]) -> f64 {
    let m = mean_f(v);
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64;
    var.sqrt()
}

// ── Parametric VaR / CVaR ────────────────────────────────────────────────────

/// Parametric VaR: z_alpha * sigma (for a portfolio with known sigma).
pub fn parametric_var(mu: f64, sigma: f64, confidence: f64) -> f64 {
    let z = -inv_normal_cdf(1.0 - confidence);
    -(mu - z * sigma) // expressed as a positive loss
}

/// Portfolio parametric VaR.
pub fn portfolio_var(weights: &[f64], cov_matrix: &Matrix, confidence: f64) -> f64 {
    let port_var = quad_form(cov_matrix, weights);
    let sigma = port_var.sqrt();
    parametric_var(0.0, sigma, confidence)
}

// ── Historical VaR / CVaR ─────────────────────────────────────────────────────

/// Historical VaR: sorted empirical quantile.
pub fn historical_var(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((1.0 - confidence) * sorted.len() as f64) as usize;
    let idx = idx.min(sorted.len() - 1);
    -sorted[idx]
}

/// Historical CVaR (Expected Shortfall).
pub fn expected_shortfall(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((1.0 - confidence) * sorted.len() as f64) as usize;
    let tail = &sorted[..idx.max(1)];
    -mean_f(tail)
}

/// Portfolio CVaR using simulated returns.
pub fn portfolio_cvar(
    weights: &[f64],
    returns_history: &Matrix, // T × N
    confidence: f64,
) -> f64 {
    let port_returns: Vec<f64> = returns_history
        .iter()
        .map(|row| row.iter().zip(weights.iter()).map(|(r, w)| r * w).sum::<f64>())
        .collect();
    expected_shortfall(&port_returns, confidence)
}

// ── Monte Carlo VaR ───────────────────────────────────────────────────────────

/// Monte Carlo VaR using Box-Muller normal samples.
pub fn monte_carlo_var(
    mu: f64,
    sigma: f64,
    confidence: f64,
    n_sims: usize,
    horizon: usize,
) -> f64 {
    // Simple pseudo-random using LCG.
    let mut state = 12345_u64;
    let mut pseudo_normal = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = state as f64 / u64::MAX as f64;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = state as f64 / u64::MAX as f64;
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    let h = horizon as f64;
    let mut final_rets: Vec<f64> = (0..n_sims)
        .map(|_| {
            // Cumulative return over horizon.
            (0..horizon).map(|_| mu + sigma * pseudo_normal()).sum::<f64>()
        })
        .collect();
    final_rets.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((1.0 - confidence) * n_sims as f64) as usize;
    -final_rets[idx.min(n_sims - 1)]
}

// ── Stressed VaR ──────────────────────────────────────────────────────────────

/// VaR computed only over a specified stress period.
pub fn stressed_var(returns: &[f64], stress_indices: &[usize], confidence: f64) -> f64 {
    let stress_returns: Vec<f64> = stress_indices
        .iter()
        .filter(|&&i| i < returns.len())
        .map(|&i| returns[i])
        .collect();
    historical_var(&stress_returns, confidence)
}

// ── Component & Marginal VaR ──────────────────────────────────────────────────

/// Marginal VaR: ∂VaR/∂w_i = z * (Σw)_i / σ_p.
pub fn marginal_var(weights: &[f64], cov_matrix: &Matrix, confidence: f64) -> Vec<f64> {
    let port_var = quad_form(cov_matrix, weights).max(1e-12);
    let sigma = port_var.sqrt();
    let z = inv_normal_cdf(confidence);
    let cov_w = mat_vec_mul(cov_matrix, weights);
    cov_w.iter().map(|x| z * x / sigma).collect()
}

/// Component VaR: w_i * MVaR_i.
pub fn component_var(weights: &[f64], cov_matrix: &Matrix, confidence: f64) -> Vec<f64> {
    let mvar = marginal_var(weights, cov_matrix, confidence);
    weights.iter().zip(mvar.iter()).map(|(w, m)| w * m).collect()
}

// ── Beta, Tracking Error, Information Ratio ────────────────────────────────────

/// Beta of portfolio relative to benchmark.
pub fn beta(portfolio_returns: &[f64], benchmark_returns: &[f64]) -> f64 {
    let n = portfolio_returns.len().min(benchmark_returns.len());
    if n < 2 {
        return 1.0;
    }
    let pm = mean_f(&portfolio_returns[..n]);
    let bm = mean_f(&benchmark_returns[..n]);
    let cov: f64 = (0..n).map(|i| (portfolio_returns[i] - pm) * (benchmark_returns[i] - bm)).sum::<f64>() / (n - 1) as f64;
    let bvar: f64 = (0..n).map(|i| (benchmark_returns[i] - bm).powi(2)).sum::<f64>() / (n - 1) as f64;
    if bvar < 1e-12 { 0.0 } else { cov / bvar }
}

/// Annualised tracking error (assuming daily returns, 252 trading days).
pub fn tracking_error(portfolio_returns: &[f64], benchmark_returns: &[f64]) -> f64 {
    let n = portfolio_returns.len().min(benchmark_returns.len());
    if n < 2 {
        return 0.0;
    }
    let diffs: Vec<f64> = (0..n).map(|i| portfolio_returns[i] - benchmark_returns[i]).collect();
    std_f(&diffs) * 252_f64.sqrt()
}

/// Information ratio = active_return / tracking_error (annualised).
pub fn information_ratio(portfolio_returns: &[f64], benchmark_returns: &[f64]) -> f64 {
    let n = portfolio_returns.len().min(benchmark_returns.len());
    if n < 2 {
        return 0.0;
    }
    let active: f64 = (0..n)
        .map(|i| portfolio_returns[i] - benchmark_returns[i])
        .sum::<f64>()
        / n as f64
        * 252.0;
    let te = tracking_error(portfolio_returns, benchmark_returns);
    if te < 1e-12 { 0.0 } else { active / te }
}

// ── Sortino Ratio ─────────────────────────────────────────────────────────────

/// Sortino ratio: (mean_return - target) / downside_deviation.
pub fn sortino_ratio(returns: &[f64], target: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mean = mean_f(returns) * 252.0;
    let downside: Vec<f64> = returns.iter().map(|r| (target - r).max(0.0).powi(2)).collect();
    let dd = (downside.iter().sum::<f64>() / returns.len() as f64).sqrt() * 252_f64.sqrt();
    if dd < 1e-12 { 0.0 } else { (mean - target * 252.0) / dd }
}

// ── Calmar Ratio ──────────────────────────────────────────────────────────────

/// Calmar ratio: annualised_return / max_drawdown.
pub fn calmar_ratio(returns: &[f64], max_drawdown: f64) -> f64 {
    if returns.is_empty() || max_drawdown.abs() < 1e-12 {
        return 0.0;
    }
    let ann_ret = mean_f(returns) * 252.0;
    ann_ret / max_drawdown.abs()
}

// ── Ulcer Index ───────────────────────────────────────────────────────────────

/// Ulcer index: RMS of drawdowns from rolling maximum.
pub fn ulcer_index(equity_curve: &[f64]) -> f64 {
    if equity_curve.is_empty() {
        return 0.0;
    }
    let mut peak = equity_curve[0];
    let mut sq_dd_sum = 0.0_f64;
    for &e in equity_curve {
        if e > peak { peak = e; }
        let dd = (e - peak) / peak.max(1e-12) * 100.0;
        sq_dd_sum += dd * dd;
    }
    (sq_dd_sum / equity_curve.len() as f64).sqrt()
}

// ── Max Drawdown ──────────────────────────────────────────────────────────────

/// Maximum drawdown from an equity curve (returns the magnitude, a positive number).
pub fn max_drawdown(equity_curve: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0_f64;
    for &e in equity_curve {
        if e > peak { peak = e; }
        let dd = (peak - e) / peak.abs().max(1e-12);
        if dd > max_dd { max_dd = dd; }
    }
    max_dd
}

/// Sharpe ratio (annualised, daily returns, 252 trading days).
pub fn sharpe_ratio(returns: &[f64], risk_free_daily: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let excess: Vec<f64> = returns.iter().map(|r| r - risk_free_daily).collect();
    let m = mean_f(&excess);
    let s = std_f(&excess);
    if s < 1e-12 { 0.0 } else { m / s * 252_f64.sqrt() }
}

// ── VaR Backtesting ───────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct VaRBacktest {
    pub n_violations: usize,
    pub violation_rate: f64,
    pub kupiec_lr_stat: f64,
    pub kupiec_p_value: f64,
    pub christoffersen_lr_stat: f64,
    pub christoffersen_p_value: f64,
}

/// Kupiec (1995) Proportion of Failures test.
pub fn kupiec_test(n_obs: usize, n_violations: usize, confidence: f64) -> (f64, f64) {
    let p = 1.0 - confidence;
    let x = n_violations as f64;
    let n = n_obs as f64;
    if x == 0.0 {
        return (0.0, 1.0);
    }
    let lr = 2.0 * (((x / n).powf(x) * (1.0 - x / n).powf(n - x))
        / (p.powf(x) * (1.0 - p).powf(n - x)))
        .ln();
    // Chi-squared(1) p-value approximation.
    let p_val = 1.0 - chi2_cdf_1(lr);
    (lr.max(0.0), p_val)
}

fn chi2_cdf_1(x: f64) -> f64 {
    // Chi^2(1) CDF = erf(sqrt(x/2)).
    if x < 0.0 { return 0.0; }
    erf((x / 2.0).sqrt())
}

/// Full VaR backtest with Kupiec and Christoffersen tests.
pub fn var_backtest(predicted_var: &[f64], actual_returns: &[f64], confidence: f64) -> VaRBacktest {
    let n = predicted_var.len().min(actual_returns.len());
    let mut violations: Vec<bool> = Vec::with_capacity(n);
    let mut n_viol = 0usize;
    for i in 0..n {
        // VaR is expressed as positive loss; violation when loss > VaR.
        let is_viol = actual_returns[i] < -predicted_var[i];
        violations.push(is_viol);
        if is_viol { n_viol += 1; }
    }
    let viol_rate = n_viol as f64 / n as f64;
    let (kupiec_lr, kupiec_p) = kupiec_test(n, n_viol, confidence);

    // Christoffersen independence test (simple 2x2 transition matrix).
    let (christoffersen_lr, christoffersen_p) = christoffersen_test(&violations, 1.0 - confidence);

    VaRBacktest {
        n_violations: n_viol,
        violation_rate: viol_rate,
        kupiec_lr_stat: kupiec_lr,
        kupiec_p_value: kupiec_p,
        christoffersen_lr_stat: christoffersen_lr,
        christoffersen_p_value: christoffersen_p,
    }
}

fn christoffersen_test(violations: &[bool], p: f64) -> (f64, f64) {
    if violations.len() < 2 {
        return (0.0, 1.0);
    }
    let mut n00 = 0.0_f64;
    let mut n01 = 0.0_f64;
    let mut n10 = 0.0_f64;
    let mut n11 = 0.0_f64;
    for i in 1..violations.len() {
        match (violations[i - 1], violations[i]) {
            (false, false) => n00 += 1.0,
            (false, true) => n01 += 1.0,
            (true, false) => n10 += 1.0,
            (true, true) => n11 += 1.0,
        }
    }
    let pi01 = if n00 + n01 > 0.0 { n01 / (n00 + n01) } else { p };
    let pi11 = if n10 + n11 > 0.0 { n11 / (n10 + n11) } else { p };
    let pi = (n01 + n11) / (n00 + n01 + n10 + n11);

    let ll_h0 = safe_binom_ll(pi, n00, n01) + safe_binom_ll(pi, n10, n11);
    let ll_h1 = safe_binom_ll(pi01, n00, n01) + safe_binom_ll(pi11, n10, n11);
    let lr = 2.0 * (ll_h1 - ll_h0);
    let p_val = 1.0 - chi2_cdf_1(lr.max(0.0));
    (lr.max(0.0), p_val)
}

fn safe_binom_ll(p: f64, n0: f64, n1: f64) -> f64 {
    let p = p.clamp(1e-10, 1.0 - 1e-10);
    n0 * (1.0 - p).ln() + n1 * p.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn historical_var_basic() {
        let returns = vec![-0.1, -0.05, 0.0, 0.05, 0.1, -0.08, 0.03, -0.02, 0.01, 0.04];
        let var = historical_var(&returns, 0.95);
        assert!(var >= 0.0, "VaR should be non-negative");
    }

    #[test]
    fn ulcer_index_positive() {
        let equity = vec![100.0, 98.0, 95.0, 97.0, 100.0, 102.0];
        let ui = ulcer_index(&equity);
        assert!(ui >= 0.0);
    }

    #[test]
    fn tracking_error_zero_identical() {
        let r = vec![0.01, -0.02, 0.005];
        let te = tracking_error(&r, &r);
        assert!(te < 1e-10);
    }
}
