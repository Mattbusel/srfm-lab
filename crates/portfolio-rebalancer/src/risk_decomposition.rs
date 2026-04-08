// ═══════════════════════════════════════════════════════════════════════════
// RISK MEASURES
// ═══════════════════════════════════════════════════════════════════════════

use std::f64::consts::PI;

/// Portfolio variance: w'Σw
pub fn portfolio_variance(weights: &[f64], cov: &[Vec<f64>]) -> f64 {
    let n = weights.len();
    let mut var = 0.0;
    for i in 0..n {
        for j in 0..n {
            var += weights[i] * weights[j] * cov[i][j];
        }
    }
    var
}

/// Portfolio volatility: sqrt(w'Σw)
pub fn portfolio_volatility(weights: &[f64], cov: &[Vec<f64>]) -> f64 {
    portfolio_variance(weights, cov).max(0.0).sqrt()
}

/// Marginal risk contribution: Σw / σ_p
pub fn marginal_risk_contribution(weights: &[f64], cov: &[Vec<f64>]) -> Vec<f64> {
    let n = weights.len();
    let vol = portfolio_volatility(weights, cov);
    if vol < 1e-15 {
        return vec![0.0; n];
    }
    let sigma_w = mat_vec_mult(cov, weights);
    sigma_w.iter().map(|&sw| sw / vol).collect()
}

/// Component risk contribution: w_i * (Σw)_i / σ_p
pub fn component_risk_contribution(weights: &[f64], cov: &[Vec<f64>]) -> Vec<f64> {
    let n = weights.len();
    let vol = portfolio_volatility(weights, cov);
    if vol < 1e-15 {
        return vec![0.0; n];
    }
    let sigma_w = mat_vec_mult(cov, weights);
    (0..n).map(|i| weights[i] * sigma_w[i] / vol).collect()
}

/// Percentage risk contribution: RC_i / σ_p
pub fn pct_risk_contribution(weights: &[f64], cov: &[Vec<f64>]) -> Vec<f64> {
    let rc = component_risk_contribution(weights, cov);
    let vol = portfolio_volatility(weights, cov);
    if vol < 1e-15 {
        return vec![0.0; rc.len()];
    }
    rc.iter().map(|&r| r / vol).collect()
}

/// Verify Euler decomposition: sum of component risk = portfolio vol
pub fn euler_decomposition_check(weights: &[f64], cov: &[Vec<f64>]) -> f64 {
    let rc = component_risk_contribution(weights, cov);
    let sum: f64 = rc.iter().sum();
    let vol = portfolio_volatility(weights, cov);
    (sum - vol).abs()
}

// ═══════════════════════════════════════════════════════════════════════════
// COMPONENT VaR
// ═══════════════════════════════════════════════════════════════════════════

/// Parametric VaR (assuming normal distribution).
pub fn parametric_var(weights: &[f64], cov: &[Vec<f64>], confidence: f64, portfolio_value: f64) -> f64 {
    let vol = portfolio_volatility(weights, cov);
    let z = norm_inv(1.0 - confidence);
    -z * vol * portfolio_value
}

/// Component VaR: how much each asset contributes to total VaR.
pub fn component_var(weights: &[f64], cov: &[Vec<f64>], confidence: f64, portfolio_value: f64) -> Vec<f64> {
    let n = weights.len();
    let vol = portfolio_volatility(weights, cov);
    let z = norm_inv(1.0 - confidence);
    if vol < 1e-15 {
        return vec![0.0; n];
    }

    let sigma_w = mat_vec_mult(cov, weights);
    (0..n).map(|i| {
        -z * weights[i] * sigma_w[i] / vol * portfolio_value
    }).collect()
}

/// Marginal VaR: dVaR/dw_i
pub fn marginal_var(weights: &[f64], cov: &[Vec<f64>], confidence: f64, portfolio_value: f64) -> Vec<f64> {
    let vol = portfolio_volatility(weights, cov);
    let z = norm_inv(1.0 - confidence);
    if vol < 1e-15 {
        return vec![0.0; weights.len()];
    }

    let sigma_w = mat_vec_mult(cov, weights);
    sigma_w.iter().map(|&sw| -z * sw / vol * portfolio_value).collect()
}

/// Incremental VaR: VaR change from adding position i.
pub fn incremental_var(
    weights: &[f64], cov: &[Vec<f64>], confidence: f64,
    portfolio_value: f64, asset_idx: usize, weight_increment: f64,
) -> f64 {
    let n = weights.len();
    let var_before = parametric_var(weights, cov, confidence, portfolio_value);

    let mut new_weights = weights.to_vec();
    new_weights[asset_idx] += weight_increment;
    // Renormalize
    let sum: f64 = new_weights.iter().sum();
    if sum.abs() > 1e-15 {
        for w in new_weights.iter_mut() { *w /= sum; }
    }

    let var_after = parametric_var(&new_weights, cov, confidence, portfolio_value);
    var_after - var_before
}

// ═══════════════════════════════════════════════════════════════════════════
// CONDITIONAL VALUE-AT-RISK (CVaR / Expected Shortfall)
// ═══════════════════════════════════════════════════════════════════════════

/// Parametric CVaR (normal distribution).
pub fn parametric_cvar(weights: &[f64], cov: &[Vec<f64>], confidence: f64, portfolio_value: f64) -> f64 {
    let vol = portfolio_volatility(weights, cov);
    let alpha = 1.0 - confidence;
    let z = norm_inv(1.0 - confidence);
    let pdf_z = norm_pdf(z);
    pdf_z / alpha * vol * portfolio_value
}

/// Historical CVaR from scenario returns.
pub fn historical_cvar(portfolio_returns: &[f64], confidence: f64) -> f64 {
    let n = portfolio_returns.len();
    if n == 0 { return 0.0; }

    let mut sorted = portfolio_returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let cutoff = ((1.0 - confidence) * n as f64).ceil() as usize;
    let cutoff = cutoff.max(1).min(n);

    let sum: f64 = sorted[..cutoff].iter().sum();
    -sum / cutoff as f64
}

/// Component CVaR decomposition.
pub fn component_cvar(weights: &[f64], cov: &[Vec<f64>], confidence: f64, portfolio_value: f64) -> Vec<f64> {
    let n = weights.len();
    let vol = portfolio_volatility(weights, cov);
    let alpha = 1.0 - confidence;
    let z = norm_inv(1.0 - confidence);
    let pdf_z = norm_pdf(z);

    if vol < 1e-15 {
        return vec![0.0; n];
    }

    let sigma_w = mat_vec_mult(cov, weights);
    (0..n).map(|i| {
        pdf_z / alpha * weights[i] * sigma_w[i] / vol * portfolio_value
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// FACTOR RISK DECOMPOSITION
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct FactorModel {
    pub factor_names: Vec<String>,
    pub factor_loadings: Vec<Vec<f64>>,  // [n_assets][n_factors] = B matrix
    pub factor_covariance: Vec<Vec<f64>>, // [n_factors][n_factors] = Σ_f
    pub specific_variance: Vec<f64>,      // diagonal specific risk
}

impl FactorModel {
    pub fn new(
        names: Vec<String>,
        loadings: Vec<Vec<f64>>,
        factor_cov: Vec<Vec<f64>>,
        specific_var: Vec<f64>,
    ) -> Self {
        Self {
            factor_names: names,
            factor_loadings: loadings,
            factor_covariance: factor_cov,
            specific_variance: specific_var,
        }
    }

    /// Reconstruct full covariance: Σ = BΣ_fB' + D
    pub fn full_covariance(&self) -> Vec<Vec<f64>> {
        let n = self.factor_loadings.len();
        let k = self.factor_names.len();
        let mut cov = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                // (BΣ_fB')_{ij} = Σ_{l,m} B_{il} Σ_f_{lm} B_{jm}
                let mut systematic = 0.0;
                for l in 0..k {
                    for m in 0..k {
                        systematic += self.factor_loadings[i][l]
                            * self.factor_covariance[l][m]
                            * self.factor_loadings[j][m];
                    }
                }
                cov[i][j] = systematic;
                if i == j {
                    cov[i][j] += self.specific_variance[i];
                }
            }
        }
        cov
    }

    /// Portfolio factor exposures: B'w
    pub fn portfolio_factor_exposure(&self, weights: &[f64]) -> Vec<f64> {
        let k = self.factor_names.len();
        let n = weights.len();
        let mut exposures = vec![0.0; k];
        for f in 0..k {
            for i in 0..n {
                if i < self.factor_loadings.len() && f < self.factor_loadings[i].len() {
                    exposures[f] += weights[i] * self.factor_loadings[i][f];
                }
            }
        }
        exposures
    }

    /// Systematic (factor) risk: sqrt(w'BΣ_fB'w)
    pub fn systematic_risk(&self, weights: &[f64]) -> f64 {
        let exposures = self.portfolio_factor_exposure(weights);
        let k = exposures.len();
        let mut var = 0.0;
        for i in 0..k {
            for j in 0..k {
                var += exposures[i] * self.factor_covariance[i][j] * exposures[j];
            }
        }
        var.max(0.0).sqrt()
    }

    /// Specific (idiosyncratic) risk: sqrt(w'Dw)
    pub fn specific_risk(&self, weights: &[f64]) -> f64 {
        let n = weights.len();
        let mut var = 0.0;
        for i in 0..n {
            if i < self.specific_variance.len() {
                var += weights[i] * weights[i] * self.specific_variance[i];
            }
        }
        var.max(0.0).sqrt()
    }

    /// Total risk decomposition into systematic + specific.
    pub fn risk_decomposition(&self, weights: &[f64]) -> RiskDecomposition {
        let sys = self.systematic_risk(weights);
        let spec = self.specific_risk(weights);
        let total = (sys * sys + spec * spec).sqrt();
        let factor_exp = self.portfolio_factor_exposure(weights);

        // Factor-level risk contribution
        let k = self.factor_names.len();
        let mut factor_risk = vec![0.0; k];
        for i in 0..k {
            let mut var_i = 0.0;
            for j in 0..k {
                var_i += factor_exp[i] * self.factor_covariance[i][j] * factor_exp[j];
            }
            factor_risk[i] = if total > 1e-15 {
                var_i / total
            } else {
                0.0
            };
        }

        RiskDecomposition {
            total_risk: total,
            systematic_risk: sys,
            specific_risk: spec,
            systematic_pct: if total > 0.0 { sys * sys / (total * total) } else { 0.0 },
            specific_pct: if total > 0.0 { spec * spec / (total * total) } else { 0.0 },
            factor_exposures: factor_exp,
            factor_risk_contributions: factor_risk,
            factor_names: self.factor_names.clone(),
        }
    }

    /// Active risk decomposition (vs benchmark).
    pub fn active_risk_decomposition(
        &self, weights: &[f64], benchmark: &[f64],
    ) -> RiskDecomposition {
        let n = weights.len();
        let active: Vec<f64> = (0..n).map(|i| {
            weights[i] - benchmark.get(i).copied().unwrap_or(0.0)
        }).collect();
        self.risk_decomposition(&active)
    }
}

#[derive(Debug, Clone)]
pub struct RiskDecomposition {
    pub total_risk: f64,
    pub systematic_risk: f64,
    pub specific_risk: f64,
    pub systematic_pct: f64,
    pub specific_pct: f64,
    pub factor_exposures: Vec<f64>,
    pub factor_risk_contributions: Vec<f64>,
    pub factor_names: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════
// STRESS TESTING / CONTRIBUTION
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct StressScenario {
    pub name: String,
    pub factor_shocks: Vec<f64>,   // shock to each factor
    pub specific_shocks: Vec<f64>, // specific shocks per asset
}

#[derive(Debug, Clone)]
pub struct StressContribution {
    pub scenario_name: String,
    pub total_impact: f64,
    pub factor_impacts: Vec<f64>,
    pub specific_impacts: Vec<f64>,
    pub asset_contributions: Vec<f64>,
}

/// Compute stress test P&L and contribution decomposition.
pub fn stress_contribution(
    weights: &[f64],
    factor_model: &FactorModel,
    scenario: &StressScenario,
    portfolio_value: f64,
) -> StressContribution {
    let n = weights.len();
    let k = factor_model.factor_names.len();

    // Asset returns under stress: r = B * f_shock + specific_shock
    let mut asset_returns = vec![0.0; n];
    for i in 0..n {
        for f in 0..k {
            if i < factor_model.factor_loadings.len() && f < factor_model.factor_loadings[i].len() {
                asset_returns[i] += factor_model.factor_loadings[i][f] * scenario.factor_shocks[f];
            }
        }
        if i < scenario.specific_shocks.len() {
            asset_returns[i] += scenario.specific_shocks[i];
        }
    }

    let total_impact: f64 = (0..n).map(|i| weights[i] * asset_returns[i] * portfolio_value).sum();

    // Factor contributions
    let exposures = factor_model.portfolio_factor_exposure(weights);
    let factor_impacts: Vec<f64> = (0..k).map(|f| {
        exposures[f] * scenario.factor_shocks[f] * portfolio_value
    }).collect();

    let specific_impacts: Vec<f64> = (0..n).map(|i| {
        let spec = if i < scenario.specific_shocks.len() { scenario.specific_shocks[i] } else { 0.0 };
        weights[i] * spec * portfolio_value
    }).collect();

    let asset_contributions: Vec<f64> = (0..n).map(|i| {
        weights[i] * asset_returns[i] * portfolio_value
    }).collect();

    StressContribution {
        scenario_name: scenario.name.clone(),
        total_impact,
        factor_impacts,
        specific_impacts,
        asset_contributions,
    }
}

/// Run multiple stress scenarios.
pub fn stress_test_suite(
    weights: &[f64],
    factor_model: &FactorModel,
    scenarios: &[StressScenario],
    portfolio_value: f64,
) -> Vec<StressContribution> {
    scenarios.iter().map(|s| stress_contribution(weights, factor_model, s, portfolio_value)).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// RISK BUDGETING ANALYTICS
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct RiskBudgetAnalysis {
    pub asset_names: Vec<String>,
    pub weights: Vec<f64>,
    pub marginal_risk: Vec<f64>,
    pub component_risk: Vec<f64>,
    pub pct_contribution: Vec<f64>,
    pub risk_per_unit_weight: Vec<f64>,
    pub herfindahl_index: f64,      // concentration measure
    pub effective_n: f64,            // 1/HHI
    pub max_contributor: usize,
    pub min_contributor: usize,
}

/// Full risk budget analysis.
pub fn risk_budget_analysis(
    weights: &[f64],
    cov: &[Vec<f64>],
    asset_names: &[String],
) -> RiskBudgetAnalysis {
    let n = weights.len();
    let vol = portfolio_volatility(weights, cov);
    let mrc = marginal_risk_contribution(weights, cov);
    let crc = component_risk_contribution(weights, cov);
    let pct_rc = pct_risk_contribution(weights, cov);

    let risk_per_weight: Vec<f64> = (0..n).map(|i| {
        if weights[i].abs() > 1e-15 { crc[i] / weights[i] } else { 0.0 }
    }).collect();

    // Herfindahl index of risk contributions
    let hhi: f64 = pct_rc.iter().map(|&r| r * r).sum();
    let eff_n = if hhi > 1e-15 { 1.0 / hhi } else { n as f64 };

    let max_idx = pct_rc.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i).unwrap_or(0);
    let min_idx = pct_rc.iter().enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i).unwrap_or(0);

    RiskBudgetAnalysis {
        asset_names: asset_names.to_vec(),
        weights: weights.to_vec(),
        marginal_risk: mrc,
        component_risk: crc,
        pct_contribution: pct_rc,
        risk_per_unit_weight: risk_per_weight,
        herfindahl_index: hhi,
        effective_n: eff_n,
        max_contributor: max_idx,
        min_contributor: min_idx,
    }
}

/// Risk budget deviation from target.
pub fn risk_budget_deviation(
    weights: &[f64],
    cov: &[Vec<f64>],
    target_budgets: &[f64],
) -> f64 {
    let pct = pct_risk_contribution(weights, cov);
    pct.iter().zip(target_budgets.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ═══════════════════════════════════════════════════════════════════════════
// CORRELATION AND BETA DECOMPOSITION
// ═══════════════════════════════════════════════════════════════════════════

/// Compute beta of each asset w.r.t. the portfolio.
pub fn asset_betas(weights: &[f64], cov: &[Vec<f64>]) -> Vec<f64> {
    let n = weights.len();
    let var_p = portfolio_variance(weights, cov);
    if var_p < 1e-15 {
        return vec![0.0; n];
    }
    let sigma_w = mat_vec_mult(cov, weights);
    sigma_w.iter().map(|&sw| sw / var_p).collect()
}

/// Compute correlation of each asset with the portfolio.
pub fn asset_portfolio_correlations(weights: &[f64], cov: &[Vec<f64>]) -> Vec<f64> {
    let n = weights.len();
    let vol_p = portfolio_volatility(weights, cov);
    if vol_p < 1e-15 {
        return vec![0.0; n];
    }
    let sigma_w = mat_vec_mult(cov, weights);
    (0..n).map(|i| {
        let vol_i = cov[i][i].max(0.0).sqrt();
        if vol_i > 1e-15 { sigma_w[i] / (vol_p * vol_i) } else { 0.0 }
    }).collect()
}

/// Correlation contribution: how much each asset's correlation adds to portfolio risk.
pub fn correlation_contribution(weights: &[f64], cov: &[Vec<f64>]) -> Vec<f64> {
    let n = weights.len();
    let vol_p = portfolio_volatility(weights, cov);
    let sigma_w = mat_vec_mult(cov, weights);

    (0..n).map(|i| {
        let vol_i = cov[i][i].max(0.0).sqrt();
        if vol_p > 1e-15 && vol_i > 1e-15 {
            weights[i] * vol_i * sigma_w[i] / (vol_p * vol_i * vol_p)
        } else {
            0.0
        }
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// DRAWDOWN RISK
// ═══════════════════════════════════════════════════════════════════════════

/// Compute maximum drawdown from a return series.
pub fn max_drawdown(returns: &[f64]) -> f64 {
    let mut peak = 1.0;
    let mut max_dd = 0.0;
    let mut value = 1.0;

    for &r in returns {
        value *= 1.0 + r;
        if value > peak { peak = value; }
        let dd = (peak - value) / peak;
        if dd > max_dd { max_dd = dd; }
    }
    max_dd
}

/// Calmar ratio: annualized return / max drawdown.
pub fn calmar_ratio(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n == 0 { return 0.0; }

    let total: f64 = returns.iter().map(|r| (1.0 + r).ln()).sum();
    let ann_return = (total * 252.0 / n as f64).exp() - 1.0;
    let mdd = max_drawdown(returns);
    if mdd > 1e-15 { ann_return / mdd } else { 0.0 }
}

/// Conditional drawdown at risk (CDaR).
pub fn conditional_drawdown_at_risk(returns: &[f64], confidence: f64) -> f64 {
    let n = returns.len();
    if n < 2 { return 0.0; }

    // Compute drawdown series
    let mut drawdowns = Vec::new();
    let mut peak = 1.0;
    let mut value = 1.0;
    for &r in returns {
        value *= 1.0 + r;
        if value > peak { peak = value; }
        drawdowns.push((peak - value) / peak);
    }

    drawdowns.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let cutoff = ((1.0 - confidence) * n as f64).ceil() as usize;
    let cutoff = cutoff.max(1).min(n);
    drawdowns[..cutoff].iter().sum::<f64>() / cutoff as f64
}

// ═══════════════════════════════════════════════════════════════════════════
// TAIL RISK METRICS
// ═══════════════════════════════════════════════════════════════════════════

/// Skewness of return distribution.
pub fn return_skewness(returns: &[f64]) -> f64 {
    let n = returns.len() as f64;
    if n < 3.0 { return 0.0; }
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = var.sqrt();
    if std < 1e-15 { return 0.0; }
    let m3 = returns.iter().map(|r| ((r - mean) / std).powi(3)).sum::<f64>();
    m3 * n / ((n - 1.0) * (n - 2.0))
}

/// Excess kurtosis of return distribution.
pub fn return_kurtosis(returns: &[f64]) -> f64 {
    let n = returns.len() as f64;
    if n < 4.0 { return 0.0; }
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = var.sqrt();
    if std < 1e-15 { return 0.0; }
    let m4 = returns.iter().map(|r| ((r - mean) / std).powi(4)).sum::<f64>() / n;
    m4 - 3.0
}

/// Cornish-Fisher VaR (adjusting for skewness and kurtosis).
pub fn cornish_fisher_var(vol: f64, skew: f64, kurt: f64, confidence: f64) -> f64 {
    let z = norm_inv(1.0 - confidence);
    let z_cf = z + (z * z - 1.0) / 6.0 * skew
        + (z.powi(3) - 3.0 * z) / 24.0 * kurt
        - (2.0 * z.powi(3) - 5.0 * z) / 36.0 * skew * skew;
    -z_cf * vol
}

/// Tail ratio: ratio of 95th percentile gains to 5th percentile losses.
pub fn tail_ratio(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n < 20 { return 1.0; }
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p5_idx = (0.05 * n as f64) as usize;
    let p95_idx = (0.95 * n as f64) as usize;
    let lower = sorted[p5_idx].abs();
    let upper = sorted[p95_idx.min(n - 1)];
    if lower > 1e-15 { upper / lower } else { f64::INFINITY }
}

// ═══════════════════════════════════════════════════════════════════════════
// UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

fn mat_vec_mult(mat: &[Vec<f64>], vec_in: &[f64]) -> Vec<f64> {
    let n = vec_in.len();
    let mut result = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            if i < mat.len() && j < mat[i].len() {
                result[i] += mat[i][j] * vec_in[j];
            }
        }
    }
    result
}

fn norm_pdf(x: f64) -> f64 {
    (-0.5 * x * x).exp() / (2.0 * PI).sqrt()
}

fn norm_inv(p: f64) -> f64 {
    if p <= 0.0 { return f64::NEG_INFINITY; }
    if p >= 1.0 { return f64::INFINITY; }
    let a = [
        -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
        1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
        6.680131188771972e+01, -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
        -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00,
    ];
    let d = [
        7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
        3.754408661907416e+00,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    } else {
        let q = (-2.0*(1.0-p).ln()).sqrt();
        -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_cov() -> Vec<Vec<f64>> {
        vec![
            vec![0.04, 0.006, 0.008],
            vec![0.006, 0.03, 0.005],
            vec![0.008, 0.005, 0.05],
        ]
    }

    #[test]
    fn test_euler_decomposition() {
        let w = vec![0.4, 0.35, 0.25];
        let cov = sample_cov();
        let err = euler_decomposition_check(&w, &cov);
        assert!(err < 1e-10, "Euler decomposition error: {}", err);
    }

    #[test]
    fn test_component_var_sums() {
        let w = vec![0.4, 0.35, 0.25];
        let cov = sample_cov();
        let total_var = parametric_var(&w, &cov, 0.95, 1_000_000.0);
        let comp_var = component_var(&w, &cov, 0.95, 1_000_000.0);
        let sum: f64 = comp_var.iter().sum();
        assert!((sum - total_var).abs() < 1.0, "Component VaR sum {} vs total {}", sum, total_var);
    }

    #[test]
    fn test_risk_budget_pct_sums_to_one() {
        let w = vec![0.4, 0.35, 0.25];
        let cov = sample_cov();
        let pct = pct_risk_contribution(&w, &cov);
        let sum: f64 = pct.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "Pct RC should sum to ~1: {}", sum);
    }

    #[test]
    fn test_max_drawdown() {
        let returns = vec![0.01, 0.02, -0.05, -0.03, 0.01, 0.02];
        let mdd = max_drawdown(&returns);
        assert!(mdd > 0.0 && mdd < 1.0, "MDD should be between 0 and 1: {}", mdd);
    }

    #[test]
    fn test_factor_decomposition() {
        let fm = FactorModel::new(
            vec!["Market".into(), "Size".into()],
            vec![vec![1.0, 0.5], vec![0.8, -0.3], vec![1.2, 0.1]],
            vec![vec![0.04, 0.005], vec![0.005, 0.02]],
            vec![0.01, 0.015, 0.02],
        );
        let w = vec![0.4, 0.35, 0.25];
        let decomp = fm.risk_decomposition(&w);
        assert!(decomp.total_risk > 0.0);
        assert!((decomp.systematic_pct + decomp.specific_pct - 1.0).abs() < 0.01);
    }
}
