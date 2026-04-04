/// Risk attribution: marginal/percent contribution, diversification.

// ── Matrix helpers ────────────────────────────────────────────────────────────

fn mat_vec(a: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(v.iter()).map(|(x, y)| x * y).sum())
        .collect()
}

fn quad_form(a: &[Vec<f64>], v: &[f64]) -> f64 {
    let av = mat_vec(a, v);
    av.iter().zip(v.iter()).map(|(x, y)| x * y).sum()
}

// ── Marginal Contribution to Risk ────────────────────────────────────────────

/// Marginal risk contribution: MCR_i = (Σw)_i / σ_p.
/// Units: per unit of weight.
pub fn marginal_contribution_to_risk(weights: &[f64], cov_matrix: &[Vec<f64>]) -> Vec<f64> {
    let port_var = quad_form(cov_matrix, weights).max(1e-12);
    let sigma_p = port_var.sqrt();
    let cov_w = mat_vec(cov_matrix, weights);
    cov_w.iter().map(|x| x / sigma_p).collect()
}

/// Total risk contribution: TRC_i = w_i * MCR_i.
pub fn total_contribution_to_risk(weights: &[f64], cov_matrix: &[Vec<f64>]) -> Vec<f64> {
    let mcr = marginal_contribution_to_risk(weights, cov_matrix);
    weights.iter().zip(mcr.iter()).map(|(w, m)| w * m).collect()
}

/// Percent contribution to risk: PCR_i = TRC_i / σ_p.
pub fn percent_contribution_to_risk(weights: &[f64], cov_matrix: &[Vec<f64>]) -> Vec<f64> {
    let port_var = quad_form(cov_matrix, weights).max(1e-12);
    let sigma_p = port_var.sqrt();
    let trc = total_contribution_to_risk(weights, cov_matrix);
    trc.iter().map(|x| x / sigma_p).collect()
}

// ── Diversification Ratio ─────────────────────────────────────────────────────

/// Diversification ratio: (weighted sum of individual vols) / portfolio vol.
/// DR > 1 indicates diversification benefit.
pub fn diversification_ratio(weights: &[f64], cov_matrix: &[Vec<f64>]) -> f64 {
    let n = weights.len();
    let individual_vols: Vec<f64> = (0..n).map(|i| cov_matrix[i][i].sqrt()).collect();
    let weighted_vol_sum: f64 = weights.iter().zip(individual_vols.iter()).map(|(w, v)| w.abs() * v).sum();
    let port_var = quad_form(cov_matrix, weights).max(1e-12);
    let port_vol = port_var.sqrt();
    weighted_vol_sum / port_vol
}

// ── Risk Parity Diagnostics ───────────────────────────────────────────────────

/// Check how close a portfolio is to risk parity.
/// Returns per-asset deviation from equal risk contribution (1/N).
pub fn risk_parity_deviation(weights: &[f64], cov_matrix: &[Vec<f64>]) -> Vec<f64> {
    let n = weights.len();
    let pcr = percent_contribution_to_risk(weights, cov_matrix);
    let target = 1.0 / n as f64;
    pcr.iter().map(|x| x - target).collect()
}

/// Sum of squared deviations from equal risk parity.
pub fn risk_parity_score(weights: &[f64], cov_matrix: &[Vec<f64>]) -> f64 {
    risk_parity_deviation(weights, cov_matrix)
        .iter()
        .map(|d| d * d)
        .sum()
}

// ── Contribution to VaR ───────────────────────────────────────────────────────

fn inv_normal(p: f64) -> f64 {
    let p = p.clamp(1e-10, 1.0 - 1e-10);
    let sgn = if p < 0.5 { -1.0 } else { 1.0 };
    let pp = if p < 0.5 { p } else { 1.0 - p };
    let t = (-2.0 * pp.ln()).sqrt();
    let num = 2.515517 + t * (0.802853 + t * 0.010328);
    let den = 1.0 + t * (1.432788 + t * (0.189269 + t * 0.001308));
    sgn * (t - num / den)
}

/// Component VaR at given confidence level.
pub fn component_var(weights: &[f64], cov_matrix: &[Vec<f64>], confidence: f64) -> Vec<f64> {
    let z = -inv_normal(1.0 - confidence);
    let port_var = quad_form(cov_matrix, weights).max(1e-12);
    let sigma_p = port_var.sqrt();
    let cov_w = mat_vec(cov_matrix, weights);
    weights
        .iter()
        .zip(cov_w.iter())
        .map(|(w, cw)| z * w * cw / sigma_p)
        .collect()
}

// ── Tail Risk Attribution ─────────────────────────────────────────────────────

/// Attribute historical CVaR to individual assets.
///
/// Uses marginal CVaR approximation: CVaR_i ≈ w_i * dCVaR/dw_i.
/// Approximated via the delta-normal method.
pub fn component_cvar(weights: &[f64], cov_matrix: &[Vec<f64>], confidence: f64) -> Vec<f64> {
    // Under normality, CVaR_i ≈ Component VaR * (phi(z) / (1-alpha)) / z
    // where phi is normal PDF, z is the VaR z-score.
    let z = -inv_normal(1.0 - confidence);
    let phi_z = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let cvar_adjustment = phi_z / (1.0 - confidence);
    let comp_var = component_var(weights, cov_matrix, confidence);
    comp_var.iter().map(|cv| cv * cvar_adjustment / z).collect()
}

// ── Correlation Attribution ───────────────────────────────────────────────────

/// Compute the contribution of each pairwise correlation to portfolio variance.
/// Returns an N×N matrix where entry (i,j) is the variance contribution from correlation_ij.
pub fn correlation_contribution_to_variance(
    weights: &[f64],
    cov_matrix: &[Vec<f64>],
) -> Vec<Vec<f64>> {
    let n = weights.len();
    let port_var = quad_form(cov_matrix, weights).max(1e-12);
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| weights[i] * cov_matrix[i][j] * weights[j] / port_var)
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pcr_sums_to_one() {
        let w = vec![0.4, 0.3, 0.3];
        let cov = vec![
            vec![0.04, 0.01, 0.005],
            vec![0.01, 0.09, 0.02],
            vec![0.005, 0.02, 0.06],
        ];
        let pcr = percent_contribution_to_risk(&w, &cov);
        let sum: f64 = pcr.iter().sum();
        assert!((sum - 1.0).abs() < 1e-8, "sum={sum}");
    }

    #[test]
    fn diversification_ratio_gt_1() {
        let w = vec![0.5, 0.5];
        let cov = vec![vec![0.04, 0.001], vec![0.001, 0.09]];
        let dr = diversification_ratio(&w, &cov);
        assert!(dr > 1.0, "dr={dr}");
    }
}
