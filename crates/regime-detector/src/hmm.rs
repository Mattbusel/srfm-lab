/// Gaussian Hidden Markov Model with Baum-Welch EM training and Viterbi decoding.

use std::f64;

// ── HMM Model ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct HMMModel {
    pub n_states: usize,
    /// Transition matrix A[i][j] = P(s_t = j | s_{t-1} = i).
    pub transition_matrix: Vec<Vec<f64>>,
    /// Emission means μ[k] for state k.
    pub emission_means: Vec<f64>,
    /// Emission standard deviations σ[k] for state k.
    pub emission_stds: Vec<f64>,
    /// Initial state probabilities π[k].
    pub initial_probs: Vec<f64>,
    /// Log-likelihood of the fitted model.
    pub log_likelihood: f64,
}

impl HMMModel {
    pub fn new_random(n_states: usize, obs_mean: f64, obs_std: f64) -> Self {
        let mut trans = vec![vec![0.0_f64; n_states]; n_states];
        for i in 0..n_states {
            let sum = n_states as f64;
            for j in 0..n_states {
                trans[i][j] = 1.0 / sum;
            }
        }

        // Spread means evenly around obs_mean.
        let means: Vec<f64> = (0..n_states)
            .map(|k| obs_mean + (k as f64 - (n_states - 1) as f64 / 2.0) * obs_std)
            .collect();
        let stds: Vec<f64> = vec![obs_std; n_states];
        let pi: Vec<f64> = vec![1.0 / n_states as f64; n_states];

        HMMModel {
            n_states,
            transition_matrix: trans,
            emission_means: means,
            emission_stds: stds,
            initial_probs: pi,
            log_likelihood: f64::NEG_INFINITY,
        }
    }
}

// ── Normal PDF ────────────────────────────────────────────────────────────────

fn normal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let sigma = sigma.max(1e-10);
    let z = (x - mu) / sigma;
    (-0.5 * z * z).exp() / (sigma * (2.0 * std::f64::consts::PI).sqrt())
}

// ── Forward Algorithm ─────────────────────────────────────────────────────────

/// Scaled forward probabilities and scaling factors.
fn forward(obs: &[f64], model: &HMMModel) -> (Vec<Vec<f64>>, Vec<f64>) {
    let t = obs.len();
    let n = model.n_states;
    let mut alpha = vec![vec![0.0_f64; n]; t];
    let mut scales = vec![0.0_f64; t];

    // t = 0.
    for k in 0..n {
        alpha[0][k] = model.initial_probs[k]
            * normal_pdf(obs[0], model.emission_means[k], model.emission_stds[k]);
    }
    scales[0] = alpha[0].iter().sum::<f64>().max(1e-300);
    for k in 0..n { alpha[0][k] /= scales[0]; }

    for i in 1..t {
        for k in 0..n {
            let sum: f64 = (0..n).map(|j| alpha[i - 1][j] * model.transition_matrix[j][k]).sum();
            alpha[i][k] = sum * normal_pdf(obs[i], model.emission_means[k], model.emission_stds[k]);
        }
        scales[i] = alpha[i].iter().sum::<f64>().max(1e-300);
        for k in 0..n { alpha[i][k] /= scales[i]; }
    }
    (alpha, scales)
}

// ── Backward Algorithm ────────────────────────────────────────────────────────

fn backward(obs: &[f64], model: &HMMModel, scales: &[f64]) -> Vec<Vec<f64>> {
    let t = obs.len();
    let n = model.n_states;
    let mut beta = vec![vec![0.0_f64; n]; t];

    // t = T-1.
    for k in 0..n { beta[t - 1][k] = 1.0 / scales[t - 1].max(1e-300); }

    for i in (0..t - 1).rev() {
        for k in 0..n {
            beta[i][k] = (0..n)
                .map(|j| {
                    model.transition_matrix[k][j]
                        * normal_pdf(obs[i + 1], model.emission_means[j], model.emission_stds[j])
                        * beta[i + 1][j]
                })
                .sum();
        }
        let s = scales[i].max(1e-300);
        for k in 0..n { beta[i][k] /= s; }
    }
    beta
}

// ── E-step: Compute Gammas and Xis ───────────────────────────────────────────

fn compute_gamma_xi(
    obs: &[f64],
    model: &HMMModel,
    alpha: &[Vec<f64>],
    beta: &[Vec<f64>],
) -> (Vec<Vec<f64>>, Vec<Vec<Vec<f64>>>) {
    let t = obs.len();
    let n = model.n_states;

    // gamma[t][k] = P(s_t = k | obs).
    let mut gamma = vec![vec![0.0_f64; n]; t];
    for i in 0..t {
        let sum: f64 = (0..n).map(|k| alpha[i][k] * beta[i][k]).sum::<f64>().max(1e-300);
        for k in 0..n {
            gamma[i][k] = alpha[i][k] * beta[i][k] / sum;
        }
    }

    // xi[t][i][j] = P(s_t = i, s_{t+1} = j | obs).
    let mut xi = vec![vec![vec![0.0_f64; n]; n]; t - 1];
    for i in 0..t - 1 {
        let mut sum = 0.0_f64;
        for k in 0..n {
            for l in 0..n {
                let v = alpha[i][k]
                    * model.transition_matrix[k][l]
                    * normal_pdf(obs[i + 1], model.emission_means[l], model.emission_stds[l])
                    * beta[i + 1][l];
                xi[i][k][l] = v;
                sum += v;
            }
        }
        let sum = sum.max(1e-300);
        for k in 0..n {
            for l in 0..n {
                xi[i][k][l] /= sum;
            }
        }
    }
    (gamma, xi)
}

// ── M-step ────────────────────────────────────────────────────────────────────

fn m_step(
    obs: &[f64],
    model: &mut HMMModel,
    gamma: &[Vec<f64>],
    xi: &[Vec<Vec<f64>>],
) {
    let t = obs.len();
    let n = model.n_states;

    // Update initial probs.
    for k in 0..n { model.initial_probs[k] = gamma[0][k].max(1e-10); }
    let pi_sum: f64 = model.initial_probs.iter().sum::<f64>().max(1e-10);
    for k in 0..n { model.initial_probs[k] /= pi_sum; }

    // Update transition matrix.
    for i in 0..n {
        let denom: f64 = (0..t - 1).map(|s| gamma[s][i]).sum::<f64>().max(1e-10);
        for j in 0..n {
            model.transition_matrix[i][j] =
                (0..t - 1).map(|s| xi[s][i][j]).sum::<f64>() / denom;
        }
        // Normalise row.
        let row_sum: f64 = model.transition_matrix[i].iter().sum::<f64>().max(1e-10);
        for j in 0..n { model.transition_matrix[i][j] /= row_sum; }
    }

    // Update emission parameters.
    for k in 0..n {
        let denom: f64 = (0..t).map(|s| gamma[s][k]).sum::<f64>().max(1e-10);
        let mean = (0..t).map(|s| gamma[s][k] * obs[s]).sum::<f64>() / denom;
        let var = (0..t)
            .map(|s| gamma[s][k] * (obs[s] - mean).powi(2))
            .sum::<f64>()
            / denom;
        model.emission_means[k] = mean;
        model.emission_stds[k] = var.sqrt().max(1e-6);
    }
}

// ── Baum-Welch Training ───────────────────────────────────────────────────────

/// Fit a Gaussian HMM via Baum-Welch EM.
pub fn fit(observations: &[f64], n_states: usize, max_iter: usize) -> HMMModel {
    assert!(!observations.is_empty() && n_states > 0);

    let mean = observations.iter().sum::<f64>() / observations.len() as f64;
    let var = observations.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / observations.len() as f64;
    let std = var.sqrt();

    let mut model = HMMModel::new_random(n_states, mean, std);

    let mut prev_ll = f64::NEG_INFINITY;

    for _iter in 0..max_iter {
        let (alpha, scales) = forward(observations, &model);
        let beta = backward(observations, &model, &scales);
        let (gamma, xi) = compute_gamma_xi(observations, &model, &alpha, &beta);

        // Log-likelihood = sum of log scales.
        let ll: f64 = scales.iter().map(|s| s.ln()).sum();
        model.log_likelihood = ll;

        if (ll - prev_ll).abs() < 1e-6 { break; }
        prev_ll = ll;

        m_step(observations, &mut model, &gamma, &xi);
    }

    // Sort states by emission mean (bull = highest mean, bear = lowest).
    let mut order: Vec<usize> = (0..n_states).collect();
    order.sort_by(|&a, &b| model.emission_means[a].partial_cmp(&model.emission_means[b]).unwrap());
    let sorted_means: Vec<f64> = order.iter().map(|&i| model.emission_means[i]).collect();
    let sorted_stds: Vec<f64> = order.iter().map(|&i| model.emission_stds[i]).collect();
    let sorted_pi: Vec<f64> = order.iter().map(|&i| model.initial_probs[i]).collect();
    let sorted_trans: Vec<Vec<f64>> = order
        .iter()
        .map(|&i| order.iter().map(|&j| model.transition_matrix[i][j]).collect())
        .collect();

    model.emission_means = sorted_means;
    model.emission_stds = sorted_stds;
    model.initial_probs = sorted_pi;
    model.transition_matrix = sorted_trans;
    model
}

// ── State Probability Prediction ─────────────────────────────────────────────

/// Compute posterior state probabilities for a new observation given history.
pub fn predict(model: &HMMModel, observations: &[f64]) -> Vec<f64> {
    if observations.is_empty() {
        return model.initial_probs.clone();
    }
    let (alpha, scales) = forward(observations, model);
    let t = observations.len();
    let sum: f64 = alpha[t - 1].iter().sum::<f64>().max(1e-300);
    alpha[t - 1].iter().map(|x| x / sum).collect()
}

// ── Viterbi Decoding ──────────────────────────────────────────────────────────

/// Most likely state sequence via Viterbi algorithm.
pub fn viterbi(model: &HMMModel, observations: &[f64]) -> Vec<usize> {
    let t = observations.len();
    let n = model.n_states;
    if t == 0 { return vec![]; }

    let mut delta = vec![vec![f64::NEG_INFINITY; n]; t];
    let mut psi = vec![vec![0usize; n]; t];

    for k in 0..n {
        let p = normal_pdf(observations[0], model.emission_means[k], model.emission_stds[k]);
        delta[0][k] = model.initial_probs[k].ln() + p.ln().max(-1e30);
    }

    for i in 1..t {
        for k in 0..n {
            let (best_prev, best_val) = (0..n)
                .map(|j| (j, delta[i - 1][j] + model.transition_matrix[j][k].ln().max(-1e30)))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap_or((0, f64::NEG_INFINITY));
            let p = normal_pdf(observations[i], model.emission_means[k], model.emission_stds[k]);
            delta[i][k] = best_val + p.ln().max(-1e30);
            psi[i][k] = best_prev;
        }
    }

    // Backtrack.
    let mut path = vec![0usize; t];
    path[t - 1] = (0..n)
        .max_by(|&a, &b| delta[t - 1][a].partial_cmp(&delta[t - 1][b]).unwrap())
        .unwrap_or(0);
    for i in (0..t - 1).rev() {
        path[i] = psi[i + 1][path[i + 1]];
    }
    path
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hmm_fit_3_states() {
        // Simulate 3-regime data.
        let mut obs: Vec<f64> = Vec::new();
        for _ in 0..100 { obs.push(0.01); }
        for _ in 0..100 { obs.push(-0.02); }
        for _ in 0..100 { obs.push(0.00); }

        let model = fit(&obs, 3, 50);
        assert_eq!(model.n_states, 3);
        // Means should be distinct.
        let range = model.emission_means.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - model.emission_means.iter().copied().fold(f64::INFINITY, f64::min);
        assert!(range > 0.001, "range={range}");
    }

    #[test]
    fn viterbi_length_matches_input() {
        let obs: Vec<f64> = (0..50).map(|i| if i < 25 { 0.01 } else { -0.01 }).collect();
        let model = fit(&obs, 2, 30);
        let path = viterbi(&model, &obs);
        assert_eq!(path.len(), obs.len());
    }
}
