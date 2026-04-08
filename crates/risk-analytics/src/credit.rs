// credit.rs — Credit risk: Merton, KMV, hazard rates, copula, portfolio credit VaR, migration

use quant_math::distributions::{norm_cdf, norm_ppf, norm_pdf, Xoshiro256PlusPlus};
use quant_math::statistics;

/// Merton structural model for default probability
pub struct MertonModel {
    pub asset_value: f64,
    pub asset_volatility: f64,
    pub debt_face: f64,
    pub risk_free_rate: f64,
    pub time_to_maturity: f64,
}

impl MertonModel {
    pub fn new(asset_value: f64, asset_vol: f64, debt: f64, rate: f64, maturity: f64) -> Self {
        Self {
            asset_value: asset_value,
            asset_volatility: asset_vol,
            debt_face: debt,
            risk_free_rate: rate,
            time_to_maturity: maturity,
        }
    }

    /// Distance to default
    pub fn distance_to_default(&self) -> f64 {
        let t = self.time_to_maturity;
        let sigma = self.asset_volatility;
        let v = self.asset_value;
        let d = self.debt_face;
        let r = self.risk_free_rate;
        ((v / d).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt())
    }

    /// Probability of default (risk-neutral)
    pub fn default_probability(&self) -> f64 {
        let dd = self.distance_to_default();
        norm_cdf(-dd)
    }

    /// Physical default probability (with risk premium)
    pub fn physical_default_probability(&self, risk_premium: f64) -> f64 {
        let dd = self.distance_to_default();
        let dd_physical = dd + risk_premium * self.time_to_maturity.sqrt();
        norm_cdf(-dd_physical)
    }

    /// d1 and d2 (Black-Scholes style)
    pub fn d1_d2(&self) -> (f64, f64) {
        let t = self.time_to_maturity;
        let sigma = self.asset_volatility;
        let d1 = ((self.asset_value / self.debt_face).ln()
            + (self.risk_free_rate + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
        let d2 = d1 - sigma * t.sqrt();
        (d1, d2)
    }

    /// Equity value (call option on firm assets)
    pub fn equity_value(&self) -> f64 {
        let (d1, d2) = self.d1_d2();
        let pv_debt = self.debt_face * (-self.risk_free_rate * self.time_to_maturity).exp();
        self.asset_value * norm_cdf(d1) - pv_debt * norm_cdf(d2)
    }

    /// Debt value
    pub fn debt_value(&self) -> f64 {
        self.asset_value - self.equity_value()
    }

    /// Credit spread
    pub fn credit_spread(&self) -> f64 {
        let debt_val = self.debt_value();
        let t = self.time_to_maturity;
        if debt_val <= 0.0 || t <= 0.0 { return 0.0; }
        let yield_risky = -(debt_val / self.debt_face).ln() / t;
        yield_risky - self.risk_free_rate
    }

    /// Expected loss
    pub fn expected_loss(&self, lgd: f64) -> f64 {
        self.default_probability() * lgd * self.debt_face
    }

    /// Implied asset value from equity (iterative)
    pub fn implied_asset_value(equity: f64, equity_vol: f64, debt: f64, rate: f64, maturity: f64) -> (f64, f64) {
        let mut v = equity + debt;
        let mut sigma_v = equity_vol * equity / v;

        for _ in 0..100 {
            let model = MertonModel::new(v, sigma_v, debt, rate, maturity);
            let (d1, _d2) = model.d1_d2();
            let eq_calc = model.equity_value();
            let nd1 = norm_cdf(d1);

            // Update sigma_v: σ_E * E = N(d1) * σ_V * V
            if nd1 > 1e-15 && v > 1e-15 {
                sigma_v = equity_vol * equity / (nd1 * v);
            }
            // Update V: solve E = V*N(d1) - D*exp(-rT)*N(d2)
            v = (equity + debt * (-rate * maturity).exp() * norm_cdf(_d2)) / norm_cdf(d1).max(1e-15);

            if (eq_calc - equity).abs() < 1e-6 { break; }
        }
        (v, sigma_v)
    }
}

/// KMV-style distance-to-default
pub fn kmv_distance_to_default(
    asset_value: f64, asset_growth: f64, asset_vol: f64,
    default_point: f64, horizon: f64,
) -> f64 {
    if asset_vol * horizon.sqrt() < 1e-15 { return 0.0; }
    (asset_value - default_point + asset_growth * asset_value * horizon)
        / (asset_vol * asset_value * horizon.sqrt())
}

/// KMV default point: short-term debt + 0.5 * long-term debt
pub fn kmv_default_point(short_term_debt: f64, long_term_debt: f64) -> f64 {
    short_term_debt + 0.5 * long_term_debt
}

/// Map distance-to-default to EDF (empirical default frequency)
pub fn dd_to_edf(dd: f64) -> f64 {
    norm_cdf(-dd)
}

// ============================================================
// Hazard Rates and CDS
// ============================================================

/// Bootstrap hazard rate from CDS spread
pub fn hazard_rate_from_cds(cds_spread_bp: f64, recovery_rate: f64) -> f64 {
    let spread = cds_spread_bp / 10000.0;
    spread / (1.0 - recovery_rate)
}

/// Survival probability given constant hazard rate
pub fn survival_probability(hazard_rate: f64, t: f64) -> f64 {
    (-hazard_rate * t).exp()
}

/// Cumulative default probability
pub fn cumulative_default_probability(hazard_rate: f64, t: f64) -> f64 {
    1.0 - survival_probability(hazard_rate, t)
}

/// Bootstrap hazard rates from CDS term structure
pub fn bootstrap_hazard_rates(tenors: &[f64], spreads_bp: &[f64], recovery: f64) -> Vec<f64> {
    let n = tenors.len();
    let mut hazard_rates = Vec::with_capacity(n);
    let mut prev_surv = 1.0;
    let mut prev_t = 0.0;

    for i in 0..n {
        let spread = spreads_bp[i] / 10000.0;
        let dt = tenors[i] - prev_t;
        // Simplified: h_i = spread / (1 - R)
        let h = spread / (1.0 - recovery);
        hazard_rates.push(h);
        prev_surv *= (-h * dt).exp();
        prev_t = tenors[i];
    }
    hazard_rates
}

/// CDS mark-to-market
pub fn cds_mtm(
    notional: f64, cds_spread_bp: f64, current_spread_bp: f64,
    recovery: f64, tenor: f64, risk_free_rate: f64,
) -> f64 {
    let spread_diff = (current_spread_bp - cds_spread_bp) / 10000.0;
    let h = current_spread_bp / 10000.0 / (1.0 - recovery);
    // Risky PV01 ≈ (1 - exp(-(r+h)T)) / (r+h)
    let rh = risk_free_rate + h;
    let rpv01 = if rh.abs() > 1e-10 {
        (1.0 - (-(rh) * tenor).exp()) / rh
    } else { tenor };
    notional * spread_diff * rpv01
}

// ============================================================
// Default Correlation (Gaussian Copula)
// ============================================================

/// One-factor Gaussian copula model
pub struct GaussianCopula {
    pub n_obligors: usize,
    pub default_probs: Vec<f64>,
    pub correlations: Vec<f64>, // asset correlation for each obligor
    pub lgd: Vec<f64>,
    pub exposures: Vec<f64>,
}

impl GaussianCopula {
    pub fn new(
        default_probs: Vec<f64>, correlations: Vec<f64>,
        lgd: Vec<f64>, exposures: Vec<f64>,
    ) -> Self {
        let n = default_probs.len();
        assert_eq!(correlations.len(), n);
        assert_eq!(lgd.len(), n);
        assert_eq!(exposures.len(), n);
        Self { n_obligors: n, default_probs, correlations, lgd, exposures }
    }

    /// Default threshold for each obligor
    pub fn default_thresholds(&self) -> Vec<f64> {
        self.default_probs.iter().map(|&p| norm_ppf(p)).collect()
    }

    /// Conditional default probability given systematic factor
    pub fn conditional_default_prob(&self, obligor: usize, z: f64) -> f64 {
        let c = self.default_thresholds();
        let rho = self.correlations[obligor];
        let sqrt_rho = rho.sqrt();
        let sqrt_1_rho = (1.0 - rho).sqrt();
        norm_cdf((c[obligor] - sqrt_rho * z) / sqrt_1_rho)
    }

    /// Monte Carlo portfolio loss simulation
    pub fn simulate_losses(&self, n_simulations: usize, seed: u64) -> Vec<f64> {
        let mut rng = Xoshiro256PlusPlus::new(seed);
        let thresholds = self.default_thresholds();
        let mut losses = Vec::with_capacity(n_simulations);

        for _ in 0..n_simulations {
            let z = rng.normal(); // systematic factor
            let mut total_loss = 0.0;

            for i in 0..self.n_obligors {
                let rho = self.correlations[i];
                let sqrt_rho = rho.sqrt();
                let sqrt_1_rho = (1.0 - rho).sqrt();
                let eps = rng.normal(); // idiosyncratic
                let asset_return = sqrt_rho * z + sqrt_1_rho * eps;

                if asset_return < thresholds[i] {
                    total_loss += self.lgd[i] * self.exposures[i];
                }
            }
            losses.push(total_loss);
        }
        losses
    }

    /// Portfolio credit VaR from simulation
    pub fn credit_var(&self, confidence: f64, n_simulations: usize, seed: u64) -> f64 {
        let mut losses = self.simulate_losses(n_simulations, seed);
        losses.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (confidence * n_simulations as f64).floor() as usize;
        losses[idx.min(n_simulations - 1)]
    }

    /// Expected loss
    pub fn expected_loss(&self) -> f64 {
        (0..self.n_obligors)
            .map(|i| self.default_probs[i] * self.lgd[i] * self.exposures[i])
            .sum()
    }

    /// Unexpected loss
    pub fn unexpected_loss(&self, n_simulations: usize, seed: u64) -> f64 {
        let losses = self.simulate_losses(n_simulations, seed);
        let el = self.expected_loss();
        let var = statistics::variance(&losses);
        var.sqrt()
    }

    /// Vasicek asymptotic single risk factor formula
    pub fn vasicek_var(&self, confidence: f64) -> f64 {
        // For homogeneous portfolio
        let avg_pd = statistics::mean(&self.default_probs);
        let avg_rho = statistics::mean(&self.correlations);
        let avg_lgd = statistics::mean(&self.lgd);
        let total_exposure: f64 = self.exposures.iter().sum();

        let z_alpha = norm_ppf(confidence);
        let sqrt_rho = avg_rho.sqrt();
        let sqrt_1_rho = (1.0 - avg_rho).sqrt();

        let cond_pd = norm_cdf((norm_ppf(avg_pd) + sqrt_rho * z_alpha) / sqrt_1_rho);
        cond_pd * avg_lgd * total_exposure
    }
}

// ============================================================
// Credit Migration
// ============================================================

/// Credit rating transition matrix
pub struct TransitionMatrix {
    pub ratings: Vec<String>,
    pub matrix: Vec<Vec<f64>>, // matrix[from][to]
}

impl TransitionMatrix {
    /// Typical 1-year transition matrix (S&P style)
    pub fn typical_1y() -> Self {
        let ratings = vec!["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "D"]
            .iter().map(|s| s.to_string()).collect();
        let matrix = vec![
            vec![0.9081, 0.0833, 0.0068, 0.0006, 0.0012, 0.0000, 0.0000, 0.0000],
            vec![0.0070, 0.9065, 0.0779, 0.0064, 0.0006, 0.0014, 0.0002, 0.0000],
            vec![0.0009, 0.0227, 0.9105, 0.0552, 0.0074, 0.0026, 0.0001, 0.0006],
            vec![0.0002, 0.0033, 0.0595, 0.8693, 0.0530, 0.0117, 0.0012, 0.0018],
            vec![0.0003, 0.0014, 0.0067, 0.0773, 0.8053, 0.0884, 0.0100, 0.0106],
            vec![0.0000, 0.0011, 0.0024, 0.0043, 0.0648, 0.8346, 0.0407, 0.0521],
            vec![0.0022, 0.0000, 0.0022, 0.0130, 0.0238, 0.1124, 0.6486, 0.1978],
            vec![0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000],
        ];
        Self { ratings, matrix }
    }

    /// Multi-year transition matrix (matrix power)
    pub fn multi_year(&self, years: usize) -> Vec<Vec<f64>> {
        let n = self.matrix.len();
        let mut result = self.matrix.clone();
        for _ in 1..years {
            let prev = result.clone();
            result = vec![vec![0.0; n]; n];
            for i in 0..n {
                for j in 0..n {
                    for k in 0..n {
                        result[i][j] += prev[i][k] * self.matrix[k][j];
                    }
                }
            }
        }
        result
    }

    /// Default probability for a given rating over T years
    pub fn default_probability(&self, rating_idx: usize, years: usize) -> f64 {
        let tm = self.multi_year(years);
        let n = tm.len();
        tm[rating_idx][n - 1] // last column is default state
    }

    /// Expected rating distribution after T years
    pub fn expected_distribution(&self, rating_idx: usize, years: usize) -> Vec<f64> {
        let tm = self.multi_year(years);
        tm[rating_idx].clone()
    }

    /// Credit migration VaR for a bond
    pub fn migration_var(
        &self, rating_idx: usize, face_value: f64,
        spread_curve: &[f64], // spreads by rating
        duration: f64, years: usize, confidence: f64,
    ) -> f64 {
        let n = self.ratings.len();
        let tm = self.multi_year(years);
        let probs = &tm[rating_idx];

        // P&L for each possible migration
        let current_spread = spread_curve[rating_idx];
        let mut pnl_probs: Vec<(f64, f64)> = Vec::with_capacity(n);
        for j in 0..(n - 1) {
            let spread_change = spread_curve[j] - current_spread;
            let pnl = -duration * spread_change / 10000.0 * face_value;
            pnl_probs.push((pnl, probs[j]));
        }
        // Default
        let default_loss = -face_value * 0.6; // 60% LGD
        pnl_probs.push((default_loss, probs[n - 1]));

        // Sort by P&L
        pnl_probs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        // Find VaR quantile
        let target = 1.0 - confidence;
        let mut cum_prob = 0.0;
        for &(pnl, prob) in &pnl_probs {
            cum_prob += prob;
            if cum_prob >= target { return -pnl; }
        }
        -pnl_probs.last().unwrap().0
    }
}

// ============================================================
// Counterparty Credit Risk
// ============================================================

/// Credit Valuation Adjustment (unilateral CVA)
pub fn cva(
    expected_exposure: &[f64],  // EE at each time step
    hazard_rates: &[f64],       // hazard rate at each time step
    time_steps: &[f64],         // time points
    lgd: f64,
    discount_factors: &[f64],
) -> f64 {
    let n = expected_exposure.len();
    let mut cva_val = 0.0;
    let mut prev_surv = 1.0;

    for i in 0..n {
        let dt = if i == 0 { time_steps[0] } else { time_steps[i] - time_steps[i - 1] };
        let surv = prev_surv * (-hazard_rates[i] * dt).exp();
        let pd = prev_surv - surv;
        cva_val += lgd * expected_exposure[i] * pd * discount_factors[i];
        prev_surv = surv;
    }
    cva_val
}

/// Bilateral CVA (DVA)
pub fn bilateral_cva(
    ee_counterparty: &[f64],
    ee_own: &[f64],
    hazard_counterparty: &[f64],
    hazard_own: &[f64],
    time_steps: &[f64],
    lgd_counterparty: f64,
    lgd_own: f64,
    discount_factors: &[f64],
) -> (f64, f64) {
    let n = ee_counterparty.len();
    let mut cva_val = 0.0;
    let mut dva_val = 0.0;
    let mut surv_c = 1.0;
    let mut surv_o = 1.0;

    for i in 0..n {
        let dt = if i == 0 { time_steps[0] } else { time_steps[i] - time_steps[i - 1] };
        let new_surv_c = surv_c * (-hazard_counterparty[i] * dt).exp();
        let new_surv_o = surv_o * (-hazard_own[i] * dt).exp();
        let pd_c = surv_c - new_surv_c;
        let pd_o = surv_o - new_surv_o;

        cva_val += lgd_counterparty * ee_counterparty[i] * pd_c * surv_o * discount_factors[i];
        dva_val += lgd_own * ee_own[i] * pd_o * surv_c * discount_factors[i];

        surv_c = new_surv_c;
        surv_o = new_surv_o;
    }
    (cva_val, dva_val)
}

/// Potential Future Exposure (PFE) from simulation
pub fn pfe_from_simulations(
    simulated_exposures: &[Vec<f64>], // [time_step][simulation]
    confidence: f64,
) -> Vec<f64> {
    simulated_exposures.iter().map(|exposures| {
        let mut sorted = exposures.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let idx = (confidence * sorted.len() as f64).floor() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }).collect()
}

/// Expected exposure from simulation
pub fn expected_exposure(simulated_exposures: &[Vec<f64>]) -> Vec<f64> {
    simulated_exposures.iter().map(|exposures| {
        let positives: Vec<f64> = exposures.iter().map(|&e| e.max(0.0)).collect();
        statistics::mean(&positives)
    }).collect()
}

/// Wrong-way risk adjustment
pub fn wrong_way_risk_factor(correlation_exposure_default: f64) -> f64 {
    // Simple multiplicative adjustment
    1.0 + 2.0 * correlation_exposure_default.max(0.0)
}

/// Netting benefit ratio
pub fn netting_benefit(gross_exposure: f64, net_exposure: f64) -> f64 {
    if gross_exposure < 1e-15 { return 0.0; }
    1.0 - net_exposure / gross_exposure
}

/// Loss Given Default model (recovery rate from seniority and sector)
pub fn lgd_model(seniority: &str, sector: &str) -> f64 {
    let base_recovery: f64 = match seniority {
        "senior_secured" => 0.53,
        "senior_unsecured" => 0.37,
        "subordinated" => 0.24,
        "junior_subordinated" => 0.17,
        _ => 0.40,
    };
    let sector_adj = match sector {
        "financial" => -0.05,
        "utility" => 0.05,
        "industrial" => 0.00,
        "technology" => -0.03,
        _ => 0.00,
    };
    let recovery = (base_recovery + sector_adj).max(0.05_f64).min(0.95_f64);
    1.0 - recovery
}

/// Expected default frequency from credit score (logistic model)
pub fn edf_from_score(score: f64, intercept: f64, coefficient: f64) -> f64 {
    let logit = intercept + coefficient * score;
    1.0 / (1.0 + (-logit).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merton() {
        let model = MertonModel::new(100.0, 0.3, 60.0, 0.05, 1.0);
        let dd = model.distance_to_default();
        assert!(dd > 0.0);
        let pd = model.default_probability();
        assert!(pd > 0.0 && pd < 1.0);
    }

    #[test]
    fn test_hazard_rate() {
        let h = hazard_rate_from_cds(100.0, 0.4);
        assert!((h - 0.01 / 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_copula() {
        let pds = vec![0.01, 0.02, 0.05];
        let rhos = vec![0.2, 0.2, 0.3];
        let lgds = vec![0.6, 0.6, 0.6];
        let exps = vec![100.0, 200.0, 150.0];
        let model = GaussianCopula::new(pds, rhos, lgds, exps);
        let el = model.expected_loss();
        assert!(el > 0.0);
    }

    #[test]
    fn test_transition_matrix() {
        let tm = TransitionMatrix::typical_1y();
        let pd = tm.default_probability(3, 5); // BBB over 5 years
        assert!(pd > 0.0 && pd < 1.0);
    }
}
