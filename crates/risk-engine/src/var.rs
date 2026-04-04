/// Value at Risk implementations.

// ── Normal distribution helpers ───────────────────────────────────────────────

fn erf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let y = 1.0 - poly * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}

pub fn inv_normal_cdf(p: f64) -> f64 {
    let p = p.clamp(1e-10, 1.0 - 1e-10);
    let sgn = if p < 0.5 { -1.0 } else { 1.0 };
    let pp = if p < 0.5 { p } else { 1.0 - p };
    let t = (-2.0 * pp.ln()).sqrt();
    let num = 2.515517 + t * (0.802853 + t * 0.010328);
    let den = 1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308));
    sgn * (t - num / den)
}

fn mean_f(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

// ── Historical VaR ────────────────────────────────────────────────────────────

/// Historical simulation VaR.
/// `confidence` = 0.95 means the 5th percentile of losses.
pub fn historical_var(returns: &[f64], confidence: f64, window: usize) -> f64 {
    let slice = if window > 0 && window < returns.len() {
        &returns[returns.len() - window..]
    } else {
        returns
    };
    if slice.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = slice.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
    -sorted[idx.min(sorted.len() - 1)]
}

// ── Parametric VaR ────────────────────────────────────────────────────────────

/// Parametric (normal) VaR: z_α * σ - μ.
pub fn parametric_var(mu: f64, sigma: f64, confidence: f64) -> f64 {
    let z = -inv_normal_cdf(1.0 - confidence);
    -(mu - z * sigma)
}

// ── Monte Carlo VaR ───────────────────────────────────────────────────────────

/// Monte Carlo VaR via simple normal path simulation.
pub fn monte_carlo_var(mu: f64, sigma: f64, confidence: f64, n_sims: usize, horizon: usize) -> f64 {
    let mut state = 98765_u64;
    let mut pseudo_normal = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = (state >> 11) as f64 / (1u64 << 53) as f64;
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (state >> 11) as f64 / (1u64 << 53) as f64;
        let u1 = u1.max(1e-15);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };

    let mut final_returns: Vec<f64> = (0..n_sims)
        .map(|_| (0..horizon).map(|_| mu + sigma * pseudo_normal()).sum::<f64>())
        .collect();
    final_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((1.0 - confidence) * n_sims as f64).floor() as usize;
    -final_returns[idx.min(n_sims - 1)]
}

// ── Expected Shortfall ────────────────────────────────────────────────────────

/// CVaR/ES: average loss beyond the VaR threshold.
pub fn expected_shortfall(returns: &[f64], confidence: f64) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let cut = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
    let tail = &sorted[..cut.max(1)];
    -mean_f(tail)
}

// ── Stressed VaR ──────────────────────────────────────────────────────────────

/// Compute VaR using only the worst `stress_period_indices` observations.
pub fn stressed_var(returns: &[f64], stress_period_indices: &[usize], confidence: f64) -> f64 {
    let stress: Vec<f64> = stress_period_indices
        .iter()
        .filter(|&&i| i < returns.len())
        .map(|&i| returns[i])
        .collect();
    historical_var(&stress, confidence, 0)
}

// ── VaR Backtesting ───────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct VaRBacktest {
    pub n_obs: usize,
    pub n_violations: usize,
    pub violation_rate: f64,
    pub expected_violations: f64,
    pub kupiec_lr: f64,
    pub kupiec_p_value: f64,
    pub christoffersen_lr: f64,
    pub christoffersen_p_value: f64,
    pub joint_lr: f64,
    pub joint_p_value: f64,
}

fn chi2_1_p_value(x: f64) -> f64 {
    // P(chi2(1) > x) = 1 - erf(sqrt(x/2))
    if x < 0.0 { return 1.0; }
    1.0 - erf((x / 2.0).sqrt())
}

pub fn var_backtest(predicted_var_series: &[f64], actual_returns: &[f64], confidence: f64) -> VaRBacktest {
    let n = predicted_var_series.len().min(actual_returns.len());
    let mut violations: Vec<bool> = Vec::with_capacity(n);
    let mut n_viol = 0usize;

    for i in 0..n {
        let viol = actual_returns[i] < -predicted_var_series[i];
        violations.push(viol);
        if viol { n_viol += 1; }
    }

    let p = 1.0 - confidence;
    let expected = p * n as f64;
    let viol_rate = n_viol as f64 / n as f64;

    // Kupiec LR.
    let kupiec_lr = kupiec_lr_stat(n, n_viol, p);
    let kupiec_p = chi2_1_p_value(kupiec_lr);

    // Christoffersen independence LR.
    let christoffersen_lr = christoffersen_lr_stat(&violations, p);
    let christoffersen_p = chi2_1_p_value(christoffersen_lr);

    // Joint test.
    let joint_lr = kupiec_lr + christoffersen_lr;
    let joint_p = {
        // Chi2(2) p-value approximation.
        let x = joint_lr;
        if x < 0.0 { 1.0 } else { (-x / 2.0).exp() * (1.0 + x / 2.0) }
    };

    VaRBacktest {
        n_obs: n,
        n_violations: n_viol,
        violation_rate: viol_rate,
        expected_violations: expected,
        kupiec_lr,
        kupiec_p_value: kupiec_p,
        christoffersen_lr,
        christoffersen_p_value: christoffersen_p,
        joint_lr,
        joint_p_value: joint_p,
    }
}

fn kupiec_lr_stat(n: usize, x: usize, p: f64) -> f64 {
    if x == 0 || x == n { return 0.0; }
    let xf = x as f64;
    let nf = n as f64;
    let p_hat = xf / nf;
    let p = p.clamp(1e-10, 1.0 - 1e-10);
    let p_hat = p_hat.clamp(1e-10, 1.0 - 1e-10);
    2.0 * (xf * (p_hat / p).ln() + (nf - xf) * ((1.0 - p_hat) / (1.0 - p)).ln())
}

fn christoffersen_lr_stat(violations: &[bool], p: f64) -> f64 {
    let m = violations.len();
    if m < 2 { return 0.0; }
    let mut n00 = 0.0_f64;
    let mut n01 = 0.0_f64;
    let mut n10 = 0.0_f64;
    let mut n11 = 0.0_f64;
    for i in 1..m {
        match (violations[i - 1], violations[i]) {
            (false, false) => n00 += 1.0,
            (false, true) => n01 += 1.0,
            (true, false) => n10 += 1.0,
            (true, true) => n11 += 1.0,
        }
    }
    let pi01 = (n01 / (n00 + n01).max(1e-10)).clamp(1e-10, 1.0 - 1e-10);
    let pi11 = (n11 / (n10 + n11).max(1e-10)).clamp(1e-10, 1.0 - 1e-10);
    let pi = ((n01 + n11) / (n00 + n01 + n10 + n11).max(1e-10)).clamp(1e-10, 1.0 - 1e-10);

    let ll0 = n00 * (1.0 - pi).ln() + n01 * pi.ln()
        + n10 * (1.0 - pi).ln() + n11 * pi.ln();
    let ll1 = n00 * (1.0 - pi01).ln() + n01 * pi01.ln()
        + n10 * (1.0 - pi11).ln() + n11 * pi11.ln();
    (2.0 * (ll1 - ll0)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn historical_var_sanity() {
        let returns: Vec<f64> = (0..100).map(|i| (i as f64 - 50.0) * 0.001).collect();
        let var95 = historical_var(&returns, 0.95, 0);
        assert!(var95 > 0.0, "var={var95}");
    }

    #[test]
    fn parametric_var_normal() {
        let var = parametric_var(0.0, 0.01, 0.99);
        let z99 = 2.326_f64;
        assert!((var - z99 * 0.01).abs() < 0.001, "var={var}");
    }
}
