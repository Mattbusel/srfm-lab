/// Hamilton Markov regime-switching model and regime analytics.

use std::f64;

// ── Hamilton Filter ───────────────────────────────────────────────────────────

/// Parameters of a single regime.
#[derive(Debug, Clone)]
pub struct RegimeParams {
    pub mean: f64,
    pub std: f64,
}

/// Result of the Hamilton filter.
#[derive(Debug)]
pub struct HamiltonResult {
    /// T × K matrix of filtered regime probabilities.
    pub regime_probs: Vec<Vec<f64>>,
    /// K × K transition matrix.
    pub transition_matrix: Vec<Vec<f64>>,
    /// Per-regime (mean, variance) parameters.
    pub regime_params: Vec<RegimeParams>,
    /// Log-likelihood.
    pub log_likelihood: f64,
}

fn normal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let sigma = sigma.max(1e-10);
    let z = (x - mu) / sigma;
    (-0.5 * z * z).exp() / (sigma * (2.0 * f64::consts::PI).sqrt())
}

fn mean_f(v: &[f64]) -> f64 {
    if v.is_empty() { return 0.0; }
    v.iter().sum::<f64>() / v.len() as f64
}

fn var_f(v: &[f64]) -> f64 {
    if v.len() < 2 { return 1e-6; }
    let m = mean_f(v);
    v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (v.len() - 1) as f64
}

/// Hamilton (1989) filter / EM estimation of a Markov regime-switching model.
///
/// * `returns` — T-vector of asset returns.
/// * `n_regimes` — number of regimes (typically 2).
///
/// Uses EM algorithm with forward filtering.
pub fn hamilton_filter(returns: &[f64], n_regimes: usize) -> HamiltonResult {
    assert!(n_regimes >= 2);
    let t = returns.len();
    let k = n_regimes;

    // Initialise parameters by clustering.
    let mut sorted = returns.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let bucket_size = t / k;

    let mut params: Vec<RegimeParams> = (0..k)
        .map(|i| {
            let start = i * bucket_size;
            let end = if i == k - 1 { t } else { (i + 1) * bucket_size };
            let bucket = &sorted[start..end];
            RegimeParams {
                mean: mean_f(bucket),
                std: var_f(bucket).sqrt().max(1e-4),
            }
        })
        .collect();

    // Uniform transition matrix.
    let mut trans: Vec<Vec<f64>> = vec![vec![1.0 / k as f64; k]; k];
    // Boost diagonal (persistence).
    for i in 0..k {
        trans[i][i] = 0.8;
        let off = (1.0 - 0.8) / (k - 1) as f64;
        for j in 0..k {
            if j != i { trans[i][j] = off; }
        }
    }

    let mut pi = vec![1.0 / k as f64; k]; // initial regime probabilities.
    let mut ll = f64::NEG_INFINITY;

    for _iter in 0..100 {
        // ── E-step: forward filter ──────────────────────────────────────────
        let mut probs: Vec<Vec<f64>> = Vec::with_capacity(t);
        let mut curr_pi = pi.clone();
        let mut log_lik = 0.0_f64;

        for &y in returns {
            // Predicted joint: p(s_t = j, y_1..t) = sum_i p(s_{t-1}=i) * p(s_t=j|s_{t-1}=i) * p(y|s_t=j)
            let mut joint: Vec<f64> = (0..k)
                .map(|j| {
                    let transition_sum: f64 = (0..k).map(|i| curr_pi[i] * trans[i][j]).sum();
                    transition_sum * normal_pdf(y, params[j].mean, params[j].std)
                })
                .collect();
            let f_t: f64 = joint.iter().sum::<f64>().max(1e-300);
            log_lik += f_t.ln();
            for j in 0..k { joint[j] /= f_t; }
            curr_pi = joint.clone();
            probs.push(joint);
        }

        let new_ll = log_lik;
        if (new_ll - ll).abs() < 1e-6 { ll = new_ll; break; }
        ll = new_ll;

        // ── M-step: update parameters ──────────────────────────────────────
        // Update initial probabilities.
        pi = probs[0].clone();

        // Update means and variances.
        for j in 0..k {
            let gamma_j_sum: f64 = probs.iter().map(|p| p[j]).sum::<f64>().max(1e-10);
            params[j].mean = probs
                .iter()
                .zip(returns.iter())
                .map(|(p, &y)| p[j] * y)
                .sum::<f64>()
                / gamma_j_sum;
            params[j].std = (probs
                .iter()
                .zip(returns.iter())
                .map(|(p, &y)| p[j] * (y - params[j].mean).powi(2))
                .sum::<f64>()
                / gamma_j_sum)
                .sqrt()
                .max(1e-6);
        }

        // Update transition matrix (simplified: use lag-1 covariance of probabilities).
        for i in 0..k {
            for j in 0..k {
                let numer: f64 = (1..t).map(|s| probs[s - 1][i] * probs[s][j]).sum();
                let denom: f64 = (0..t - 1).map(|s| probs[s][i]).sum::<f64>().max(1e-10);
                trans[i][j] = numer / denom;
            }
            // Normalise row.
            let row_sum: f64 = trans[i].iter().sum::<f64>().max(1e-10);
            for j in 0..k { trans[i][j] /= row_sum; }
        }
    }

    HamiltonResult {
        regime_probs: probs_from_filter(returns, &params, &trans, &pi),
        transition_matrix: trans,
        regime_params: params,
        log_likelihood: ll,
    }
}

fn probs_from_filter(
    returns: &[f64],
    params: &[RegimeParams],
    trans: &[Vec<f64>],
    pi: &[f64],
) -> Vec<Vec<f64>> {
    let k = params.len();
    let mut probs = Vec::with_capacity(returns.len());
    let mut curr = pi.to_vec();
    for &y in returns {
        let mut joint: Vec<f64> = (0..k)
            .map(|j| {
                (0..k).map(|i| curr[i] * trans[i][j]).sum::<f64>()
                    * normal_pdf(y, params[j].mean, params[j].std)
            })
            .collect();
        let s: f64 = joint.iter().sum::<f64>().max(1e-300);
        for j in 0..k { joint[j] /= s; }
        curr = joint.clone();
        probs.push(joint);
    }
    probs
}

// ── Regime Conditional Moments ────────────────────────────────────────────────

/// Compute (mean, variance) of returns conditional on being in each regime,
/// weighted by the smoothed regime probabilities.
pub fn regime_conditional_moments(
    returns: &[f64],
    regime_probs: &[Vec<f64>],
) -> Vec<(f64, f64)> {
    let k = regime_probs.first().map_or(0, |p| p.len());
    let t = returns.len().min(regime_probs.len());
    (0..k)
        .map(|j| {
            let gamma_sum: f64 = (0..t).map(|s| regime_probs[s][j]).sum::<f64>().max(1e-10);
            let cond_mean: f64 =
                (0..t).map(|s| regime_probs[s][j] * returns[s]).sum::<f64>() / gamma_sum;
            let cond_var: f64 = (0..t)
                .map(|s| regime_probs[s][j] * (returns[s] - cond_mean).powi(2))
                .sum::<f64>()
                / gamma_sum;
            (cond_mean, cond_var)
        })
        .collect()
}

// ── Expected Regime Duration ──────────────────────────────────────────────────

/// Expected duration of each regime in bars.
/// For regime i: E[duration] = 1 / (1 - p_{ii}).
pub fn expected_regime_duration(transition_matrix: &[Vec<f64>]) -> Vec<f64> {
    transition_matrix
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let p_ii = row[i].clamp(0.0, 1.0 - 1e-10);
            1.0 / (1.0 - p_ii)
        })
        .collect()
}

// ── Regime Probability Smoothing ──────────────────────────────────────────────

/// Apply exponential smoothing to regime probabilities for cleaner signals.
pub fn smooth_regime_probs(probs: &[Vec<f64>], alpha: f64) -> Vec<Vec<f64>> {
    if probs.is_empty() { return vec![]; }
    let k = probs[0].len();
    let mut smoothed = Vec::with_capacity(probs.len());
    let mut prev = probs[0].clone();
    smoothed.push(prev.clone());
    for p in probs.iter().skip(1) {
        let curr: Vec<f64> = (0..k).map(|j| alpha * p[j] + (1.0 - alpha) * prev[j]).collect();
        smoothed.push(curr.clone());
        prev = curr;
    }
    smoothed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamilton_2_regime_runs() {
        let mut returns: Vec<f64> = vec![0.01; 100];
        for r in returns[50..].iter_mut() { *r = -0.02; }
        let result = hamilton_filter(&returns, 2);
        assert_eq!(result.regime_probs.len(), 100);
        assert_eq!(result.regime_params.len(), 2);
    }

    #[test]
    fn expected_duration_gt_1() {
        let trans = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
        let durations = expected_regime_duration(&trans);
        for d in durations {
            assert!(d >= 1.0, "d={d}");
        }
    }
}
