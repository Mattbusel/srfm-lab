// hmm.rs
// Hidden Markov Model implementation.
// Baum-Welch EM for parameter estimation.
// Viterbi decoding.
// 2- and 3-state models.
// States: low-vol bull, high-vol bear, mean-reversion/sideways.
// Emission: Gaussian(returns) + Gamma(vol).

use serde::{Deserialize, Serialize};
use crate::{normalize_probs, log_sum_exp, gaussian_log_pdf, gamma_log_pdf};

/// Number of EM iterations default.
const DEFAULT_EM_ITER: usize = 200;
/// Convergence tolerance for log-likelihood improvement.
const EM_CONVERGENCE_TOL: f64 = 1e-6;

/// Named states for 3-state HMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HmmState {
    LowVolBull = 0,
    HighVolBear = 1,
    MeanReversion = 2,
}

impl HmmState {
    pub fn index(&self) -> usize {
        *self as usize
    }

    pub fn from_index(i: usize, n_states: usize) -> String {
        match (i, n_states) {
            (0, 2) => "Bull".to_string(),
            (1, 2) => "Bear".to_string(),
            (0, 3) => "LowVolBull".to_string(),
            (1, 3) => "HighVolBear".to_string(),
            (2, 3) => "MeanReversion".to_string(),
            _ => format!("State{}", i),
        }
    }
}

/// HMM emission parameters per state.
/// Emission: bivariate (return, vol) where return ~ N(mu, sigma^2), vol ~ Gamma(shape, rate).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmissionParams {
    /// Mean return for this state.
    pub mu: f64,
    /// Return standard deviation.
    pub sigma: f64,
    /// Gamma shape parameter for volatility.
    pub vol_shape: f64,
    /// Gamma rate parameter for volatility.
    pub vol_rate: f64,
}

impl EmissionParams {
    /// Log emission probability for (return, vol) observation.
    pub fn log_emission(&self, ret: f64, vol: f64) -> f64 {
        let lr = gaussian_log_pdf(ret, self.mu, self.sigma);
        let lv = gamma_log_pdf(vol, self.vol_shape, self.vol_rate);
        lr + lv
    }

    /// Log emission probability for return-only observation.
    pub fn log_emission_return_only(&self, ret: f64) -> f64 {
        gaussian_log_pdf(ret, self.mu, self.sigma)
    }
}

/// Complete HMM parameter set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HmmParams {
    pub n_states: usize,
    /// Initial state distribution.
    pub pi: Vec<f64>,
    /// Transition matrix: A[i][j] = P(state_t+1 = j | state_t = i).
    pub transition: Vec<Vec<f64>>,
    /// Emission parameters per state.
    pub emission: Vec<EmissionParams>,
}

impl HmmParams {
    /// Initialize parameters randomly for n_states.
    pub fn random_init(n_states: usize, returns: &[f64], vols: &[f64]) -> Self {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(12345);

        let pi = normalize_probs(&(0..n_states).map(|_| rng.gen::<f64>()).collect::<Vec<_>>());

        let transition: Vec<Vec<f64>> = (0..n_states)
            .map(|_| {
                normalize_probs(&(0..n_states).map(|_| rng.gen::<f64>()).collect::<Vec<_>>())
            })
            .collect();

        // Initialize emissions from data quantiles.
        let mean_ret = returns.iter().sum::<f64>() / returns.len() as f64;
        let std_ret = (returns
            .iter()
            .map(|r| (r - mean_ret).powi(2))
            .sum::<f64>()
            / returns.len() as f64)
            .sqrt()
            .max(1e-6);

        let mean_vol = if !vols.is_empty() {
            vols.iter().sum::<f64>() / vols.len() as f64
        } else {
            std_ret
        };

        let emission: Vec<EmissionParams> = (0..n_states)
            .map(|i| {
                let offset = (i as f64 - (n_states - 1) as f64 / 2.0) * std_ret;
                let vol_scale = 1.0 + i as f64 * 0.5;
                EmissionParams {
                    mu: mean_ret + offset,
                    sigma: std_ret * (0.5 + i as f64 * 0.3),
                    vol_shape: 2.0 / vol_scale,
                    vol_rate: 2.0 / (mean_vol * vol_scale),
                }
            })
            .collect();

        HmmParams { n_states, pi, transition, emission }
    }

    /// Stationary distribution of the transition matrix.
    pub fn stationary_distribution(&self) -> Vec<f64> {
        let n = self.n_states;
        // Power iteration.
        let mut v = vec![1.0 / n as f64; n];
        for _ in 0..1000 {
            let mut new_v = vec![0.0f64; n];
            for i in 0..n {
                for j in 0..n {
                    new_v[j] += v[i] * self.transition[i][j];
                }
            }
            let diff: f64 = new_v.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            v = new_v;
            if diff < 1e-10 {
                break;
            }
        }
        normalize_probs(&v)
    }
}

/// Result of Viterbi decoding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViterbiResult {
    /// Most likely state sequence.
    pub states: Vec<usize>,
    /// Log-probability of the state sequence.
    pub log_prob: f64,
    /// Human-readable state names.
    pub state_names: Vec<String>,
}

/// Result of Baum-Welch EM fitting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaumWelchResult {
    pub params: HmmParams,
    /// Log-likelihood at convergence.
    pub log_likelihood: f64,
    /// Number of EM iterations.
    pub n_iter: usize,
    /// Log-likelihood history.
    pub ll_history: Vec<f64>,
    pub converged: bool,
}

/// Hidden Markov Model.
pub struct HiddenMarkovModel {
    pub params: HmmParams,
}

impl HiddenMarkovModel {
    /// Create HMM with given parameters.
    pub fn new(params: HmmParams) -> Self {
        HiddenMarkovModel { params }
    }

    /// Create HMM with random initialization.
    pub fn random(n_states: usize, returns: &[f64], vols: &[f64]) -> Self {
        HiddenMarkovModel {
            params: HmmParams::random_init(n_states, returns, vols),
        }
    }

    /// Forward algorithm (scaled).
    /// Returns (alpha, scaling_factors) where alpha[t][i] = P(o_1..o_t, s_t=i | params) / c_t.
    fn forward(&self, log_emissions: &[Vec<f64>]) -> (Vec<Vec<f64>>, Vec<f64>) {
        let t = log_emissions.len();
        let n = self.params.n_states;
        let mut alpha = vec![vec![0.0f64; n]; t];
        let mut scales = vec![0.0f64; t];

        // Initialize.
        for i in 0..n {
            alpha[0][i] = self.params.pi[i] * log_emissions[0][i].exp();
        }
        let s = alpha[0].iter().sum::<f64>().max(1e-300);
        scales[0] = s;
        for i in 0..n {
            alpha[0][i] /= s;
        }

        // Recurse.
        for t_idx in 1..t {
            for j in 0..n {
                let sum: f64 = (0..n)
                    .map(|i| alpha[t_idx - 1][i] * self.params.transition[i][j])
                    .sum();
                alpha[t_idx][j] = sum * log_emissions[t_idx][j].exp();
            }
            let s = alpha[t_idx].iter().sum::<f64>().max(1e-300);
            scales[t_idx] = s;
            for j in 0..n {
                alpha[t_idx][j] /= s;
            }
        }
        (alpha, scales)
    }

    /// Backward algorithm (scaled).
    fn backward(&self, log_emissions: &[Vec<f64>], scales: &[f64]) -> Vec<Vec<f64>> {
        let t = log_emissions.len();
        let n = self.params.n_states;
        let mut beta = vec![vec![0.0f64; n]; t];

        // Initialize.
        for i in 0..n {
            beta[t - 1][i] = 1.0;
        }

        // Recurse.
        for t_idx in (0..t - 1).rev() {
            for i in 0..n {
                beta[t_idx][i] = (0..n)
                    .map(|j| {
                        self.params.transition[i][j]
                            * log_emissions[t_idx + 1][j].exp()
                            * beta[t_idx + 1][j]
                    })
                    .sum();
                if scales[t_idx + 1] > 1e-300 {
                    beta[t_idx][i] /= scales[t_idx + 1];
                }
            }
        }
        beta
    }

    /// Compute log emission matrix.
    fn compute_log_emissions(&self, returns: &[f64], vols: &[f64]) -> Vec<Vec<f64>> {
        let t = returns.len();
        let n = self.params.n_states;
        let use_vol = !vols.is_empty() && vols.len() == t;
        (0..t)
            .map(|idx| {
                (0..n)
                    .map(|i| {
                        if use_vol {
                            self.params.emission[i].log_emission(returns[idx], vols[idx])
                        } else {
                            self.params.emission[i].log_emission_return_only(returns[idx])
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Baum-Welch EM algorithm.
    pub fn fit(
        &mut self,
        returns: &[f64],
        vols: &[f64],
        max_iter: usize,
    ) -> BaumWelchResult {
        let t = returns.len();
        let n = self.params.n_states;
        let mut ll_history = Vec::new();
        let mut converged = false;
        let mut prev_ll = f64::NEG_INFINITY;

        for iter in 0..max_iter {
            let log_em = self.compute_log_emissions(returns, vols);
            let (alpha, scales) = self.forward(&log_em);
            let beta = self.backward(&log_em, &scales);

            // Log-likelihood = sum of log(scale factors).
            let ll: f64 = scales.iter().map(|s| s.ln()).sum();
            ll_history.push(ll);

            if (ll - prev_ll).abs() < EM_CONVERGENCE_TOL && iter > 5 {
                converged = true;
                break;
            }
            prev_ll = ll;

            // E-step: compute gamma and xi.
            // gamma[t][i] = P(s_t = i | O, params).
            let gamma: Vec<Vec<f64>> = (0..t)
                .map(|idx| {
                    let unnorm: Vec<f64> = (0..n).map(|i| alpha[idx][i] * beta[idx][i]).collect();
                    normalize_probs(&unnorm)
                })
                .collect();

            // xi[t][i][j] = P(s_t=i, s_{t+1}=j | O, params).
            let mut xi = vec![vec![vec![0.0f64; n]; n]; t - 1];
            for t_idx in 0..t - 1 {
                for i in 0..n {
                    for j in 0..n {
                        xi[t_idx][i][j] = alpha[t_idx][i]
                            * self.params.transition[i][j]
                            * log_em[t_idx + 1][j].exp()
                            * beta[t_idx + 1][j];
                    }
                }
                // Normalize row.
                let row_sum: f64 = xi[t_idx].iter().flat_map(|r| r.iter()).sum();
                if row_sum > 1e-300 {
                    for i in 0..n {
                        for j in 0..n {
                            xi[t_idx][i][j] /= row_sum;
                        }
                    }
                }
            }

            // M-step: update parameters.
            // Update pi.
            self.params.pi = gamma[0].clone();

            // Update transition.
            for i in 0..n {
                let denom: f64 = (0..t - 1).map(|idx| gamma[idx][i]).sum::<f64>().max(1e-300);
                for j in 0..n {
                    let numer: f64 = (0..t - 1).map(|idx| xi[idx][i][j]).sum();
                    self.params.transition[i][j] = numer / denom;
                }
                // Normalize row.
                let row = normalize_probs(&self.params.transition[i].clone());
                self.params.transition[i] = row;
            }

            // Update emission parameters.
            for i in 0..n {
                let w: Vec<f64> = gamma.iter().map(|g| g[i]).collect();
                let w_sum = w.iter().sum::<f64>().max(1e-300);

                // Update Gaussian return params.
                let mu = w.iter().zip(returns.iter()).map(|(wi, r)| wi * r).sum::<f64>() / w_sum;
                let sigma2 = w
                    .iter()
                    .zip(returns.iter())
                    .map(|(wi, r)| wi * (r - mu).powi(2))
                    .sum::<f64>()
                    / w_sum;
                self.params.emission[i].mu = mu;
                self.params.emission[i].sigma = sigma2.sqrt().max(1e-6);

                // Update Gamma vol params if vol data provided.
                if !vols.is_empty() && vols.len() == t {
                    let v_mean =
                        w.iter().zip(vols.iter()).map(|(wi, v)| wi * v).sum::<f64>() / w_sum;
                    let v_mean2 = w
                        .iter()
                        .zip(vols.iter())
                        .map(|(wi, v)| wi * v * v)
                        .sum::<f64>()
                        / w_sum;
                    let v_var = (v_mean2 - v_mean * v_mean).max(1e-14);
                    // Method of moments: shape = v_mean^2 / v_var, rate = v_mean / v_var.
                    self.params.emission[i].vol_shape = v_mean * v_mean / v_var;
                    self.params.emission[i].vol_rate = v_mean / v_var;
                }
            }
        }

        BaumWelchResult {
            params: self.params.clone(),
            log_likelihood: prev_ll,
            n_iter: ll_history.len(),
            ll_history,
            converged,
        }
    }

    /// Viterbi algorithm for most likely state sequence.
    pub fn viterbi(&self, returns: &[f64], vols: &[f64]) -> ViterbiResult {
        let t = returns.len();
        let n = self.params.n_states;
        let log_em = self.compute_log_emissions(returns, vols);

        // delta[t][i] = max log-prob path ending in state i at time t.
        let mut delta = vec![vec![f64::NEG_INFINITY; n]; t];
        let mut psi = vec![vec![0usize; n]; t];

        // Initialize.
        for i in 0..n {
            let pi_i = self.params.pi[i].max(1e-300);
            delta[0][i] = pi_i.ln() + log_em[0][i];
        }

        // Recursion.
        for t_idx in 1..t {
            for j in 0..n {
                let mut best = f64::NEG_INFINITY;
                let mut best_i = 0;
                for i in 0..n {
                    let a_ij = self.params.transition[i][j].max(1e-300);
                    let val = delta[t_idx - 1][i] + a_ij.ln();
                    if val > best {
                        best = val;
                        best_i = i;
                    }
                }
                delta[t_idx][j] = best + log_em[t_idx][j];
                psi[t_idx][j] = best_i;
            }
        }

        // Backtrack.
        let mut states = vec![0usize; t];
        let last = &delta[t - 1];
        states[t - 1] = last
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let log_prob = delta[t - 1][states[t - 1]];

        for t_idx in (0..t - 1).rev() {
            states[t_idx] = psi[t_idx + 1][states[t_idx + 1]];
        }

        let state_names: Vec<String> = states
            .iter()
            .map(|&s| HmmState::from_index(s, n))
            .collect();

        ViterbiResult { states, log_prob, state_names }
    }

    /// Filtered state probabilities (forward pass only).
    pub fn filtered_probs(&self, returns: &[f64], vols: &[f64]) -> Vec<Vec<f64>> {
        let log_em = self.compute_log_emissions(returns, vols);
        let (alpha, _) = self.forward(&log_em);
        alpha
            .iter()
            .map(|a| normalize_probs(a))
            .collect()
    }

    /// Smoothed state probabilities (forward-backward).
    pub fn smoothed_probs(&self, returns: &[f64], vols: &[f64]) -> Vec<Vec<f64>> {
        let log_em = self.compute_log_emissions(returns, vols);
        let (alpha, scales) = self.forward(&log_em);
        let beta = self.backward(&log_em, &scales);
        let t = returns.len();
        (0..t)
            .map(|idx| {
                let unnorm: Vec<f64> = (0..self.params.n_states)
                    .map(|i| alpha[idx][i] * beta[idx][i])
                    .collect();
                normalize_probs(&unnorm)
            })
            .collect()
    }

    /// BIC for model selection: lower is better.
    pub fn bic(&self, returns: &[f64], vols: &[f64]) -> f64 {
        let log_em = self.compute_log_emissions(returns, vols);
        let (_, scales) = self.forward(&log_em);
        let ll: f64 = scales.iter().map(|s| s.max(1e-300).ln()).sum();
        let n = self.params.n_states;
        // Number of free params: (n-1) for pi, n*(n-1) for transitions, 2n for Gaussian, 2n for Gamma.
        let k = (n - 1) + n * (n - 1) + 4 * n;
        (k as f64) * (returns.len() as f64).ln() - 2.0 * ll
    }

    /// Classify each bar into a regime based on smoothed probabilities.
    pub fn classify(&self, returns: &[f64], vols: &[f64]) -> Vec<usize> {
        let smoothed = self.smoothed_probs(returns, vols);
        smoothed.iter().map(|p| crate::argmax(p)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_two_regime_data(n: usize) -> (Vec<f64>, Vec<f64>) {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(2024);
        let mut returns = Vec::with_capacity(n);
        let mut vols = Vec::with_capacity(n);
        let mut state = 0usize;
        for _ in 0..n {
            // Transition.
            let r: f64 = rng.gen();
            if state == 0 && r < 0.05 {
                state = 1;
            } else if state == 1 && r < 0.10 {
                state = 0;
            }
            if state == 0 {
                let ret = 0.001 + rng.gen::<f64>() * 0.01 - 0.005;
                returns.push(ret);
                vols.push(0.008 + rng.gen::<f64>() * 0.004);
            } else {
                let ret = -0.002 + rng.gen::<f64>() * 0.03 - 0.015;
                returns.push(ret);
                vols.push(0.020 + rng.gen::<f64>() * 0.010);
            }
        }
        (returns, vols)
    }

    #[test]
    fn test_hmm_fit_2state() {
        let (returns, vols) = make_two_regime_data(500);
        let mut hmm = HiddenMarkovModel::random(2, &returns, &vols);
        let result = hmm.fit(&returns, &vols, 50);
        assert!(result.log_likelihood.is_finite(), "Log-likelihood should be finite");
        assert!(result.n_iter > 0);
    }

    #[test]
    fn test_viterbi_returns_valid_states() {
        let (returns, vols) = make_two_regime_data(200);
        let mut hmm = HiddenMarkovModel::random(2, &returns, &vols);
        hmm.fit(&returns, &vols, 30);
        let vit = hmm.viterbi(&returns, &vols);
        assert_eq!(vit.states.len(), 200);
        for &s in &vit.states {
            assert!(s < 2, "State index out of range: {}", s);
        }
    }

    #[test]
    fn test_smoothed_probs_sum_to_one() {
        let (returns, vols) = make_two_regime_data(100);
        let mut hmm = HiddenMarkovModel::random(2, &returns, &vols);
        hmm.fit(&returns, &vols, 20);
        let smoothed = hmm.smoothed_probs(&returns, &vols);
        for row in &smoothed {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Probs should sum to 1: {}", sum);
        }
    }

    #[test]
    fn test_3state_hmm() {
        let (returns, vols) = make_two_regime_data(300);
        let mut hmm = HiddenMarkovModel::random(3, &returns, &vols);
        let result = hmm.fit(&returns, &vols, 30);
        assert!(result.params.n_states == 3);
        let classified = hmm.classify(&returns, &vols);
        for &s in &classified {
            assert!(s < 3);
        }
    }
}
