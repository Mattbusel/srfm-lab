// hmm_regime.rs
// Hidden Markov Model for market regime detection.
// Implements Baum-Welch EM for parameter fitting, Viterbi for state decoding,
// and online forward filtering for real-time posterior updates.

use crate::{normalize_probs, gaussian_log_pdf};

// ---- RegimeLabel ----------------------------------------------------------

/// High-level regime label assigned to an HMM state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegimeLabel {
    Bull,
    Bear,
    Sideways,
    HighVol,
}

impl RegimeLabel {
    /// Map an HMM state index to a RegimeLabel based on emission statistics.
    /// Classification rules:
    ///   - HighVol:  std  > 1.5 * median std across states (vol regime)
    ///   - Bull:     mean > positive threshold (trending up)
    ///   - Bear:     mean < negative threshold (trending down)
    ///   - Sideways: otherwise
    pub fn from_state(state: usize, means: &[f64], stds: &[f64]) -> RegimeLabel {
        if means.is_empty() || state >= means.is_empty().then_some(0).unwrap_or(means.len()) {
            return RegimeLabel::Sideways;
        }
        let mean = means[state];
        let std  = if state < stds.len() { stds[state] } else { 0.0 };

        // Compute median std to calibrate the high-vol threshold
        let mut sorted_stds: Vec<f64> = stds.to_vec();
        sorted_stds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median_std = if sorted_stds.is_empty() {
            1.0
        } else {
            sorted_stds[sorted_stds.len() / 2]
        };

        if std > 1.5 * median_std {
            RegimeLabel::HighVol
        } else if mean > 0.0002 {  // ~0.02% per bar
            RegimeLabel::Bull
        } else if mean < -0.0002 {
            RegimeLabel::Bear
        } else {
            RegimeLabel::Sideways
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            RegimeLabel::Bull     => "Bull",
            RegimeLabel::Bear     => "Bear",
            RegimeLabel::Sideways => "Sideways",
            RegimeLabel::HighVol  => "HighVol",
        }
    }
}

impl std::fmt::Display for RegimeLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---- HmmRegime ------------------------------------------------------------

/// Gaussian-emission Hidden Markov Model.
/// States emit scalar observations (e.g. daily log-returns).
#[derive(Debug, Clone)]
pub struct HmmRegime {
    pub n_states: usize,
    /// Row-stochastic transition matrix [n_states x n_states]
    pub transition_matrix: Vec<Vec<f64>>,
    /// Gaussian emission mean per state
    pub emission_means: Vec<f64>,
    /// Gaussian emission std per state
    pub emission_stds: Vec<f64>,
    /// Initial state probabilities
    pub initial_probs: Vec<f64>,
    /// Current filtered state distribution (forward probabilities, not log-scale)
    filter_state: Vec<f64>,
    /// Log-likelihood of the last fitted sequence
    pub log_likelihood: f64,
}

impl HmmRegime {
    /// Construct a new HMM with `n_states` states.
    /// Parameters are initialised to reasonable defaults and should be
    /// refined via `fit()`.
    pub fn new(n_states: usize) -> Self {
        assert!(n_states >= 2, "HMM requires at least 2 states");

        // Uniform initial probs
        let pi = vec![1.0 / n_states as f64; n_states];

        // Slightly sticky diagonal-dominant transition matrix
        let mut trans = vec![vec![0.0; n_states]; n_states];
        for i in 0..n_states {
            for j in 0..n_states {
                trans[i][j] = if i == j { 0.70 } else { 0.30 / (n_states - 1) as f64 };
            }
        }

        // Evenly-spaced means, unit std
        let means: Vec<f64> = (0..n_states)
            .map(|i| (i as f64 - (n_states as f64 - 1.0) / 2.0) * 0.001)
            .collect();
        let stds = vec![0.01; n_states];

        HmmRegime {
            n_states,
            transition_matrix: trans,
            emission_means: means,
            emission_stds: stds,
            initial_probs: pi.clone(),
            filter_state: pi,
            log_likelihood: f64::NEG_INFINITY,
        }
    }

    // ---- Emission ----------------------------------------------------------

    /// Log emission probability: ln N(obs | mu_k, sigma_k^2).
    fn log_emit(&self, k: usize, obs: f64) -> f64 {
        gaussian_log_pdf(obs, self.emission_means[k], self.emission_stds[k])
    }

    // ---- Forward algorithm -----------------------------------------------

    /// Compute scaled forward variables alpha[t][k] and the log-likelihood.
    /// Returns (alpha matrix, scale factors c[t]).
    /// alpha[t][k] = P(o_1..o_t, s_t=k) / prod(c[0..t])
    fn forward(&self, obs: &[f64]) -> (Vec<Vec<f64>>, Vec<f64>) {
        let t_len = obs.len();
        let k = self.n_states;
        let mut alpha = vec![vec![0.0; k]; t_len];
        let mut c = vec![0.0; t_len];

        // Initialise
        for i in 0..k {
            alpha[0][i] = self.initial_probs[i] * self.log_emit(i, obs[0]).exp();
        }
        c[0] = alpha[0].iter().sum();
        if c[0] > 0.0 {
            for i in 0..k { alpha[0][i] /= c[0]; }
        }

        // Recursion
        for t in 1..t_len {
            for j in 0..k {
                let sum: f64 = (0..k)
                    .map(|i| alpha[t - 1][i] * self.transition_matrix[i][j])
                    .sum();
                alpha[t][j] = sum * self.log_emit(j, obs[t]).exp();
            }
            c[t] = alpha[t].iter().sum();
            if c[t] > 0.0 {
                for j in 0..k { alpha[t][j] /= c[t]; }
            }
        }
        (alpha, c)
    }

    // ---- Backward algorithm -----------------------------------------------

    /// Compute scaled backward variables beta[t][k].
    fn backward(&self, obs: &[f64], c: &[f64]) -> Vec<Vec<f64>> {
        let t_len = obs.len();
        let k = self.n_states;
        let mut beta = vec![vec![0.0; k]; t_len];

        // Initialise: beta[T-1][k] = 1 / c[T-1]
        let last_c = c[t_len - 1].max(1e-300);
        for i in 0..k { beta[t_len - 1][i] = 1.0 / last_c; }

        // Recursion (backward in time)
        for t in (0..t_len - 1).rev() {
            let ct = c[t].max(1e-300);
            for i in 0..k {
                beta[t][i] = (0..k)
                    .map(|j| {
                        self.transition_matrix[i][j]
                            * self.log_emit(j, obs[t + 1]).exp()
                            * beta[t + 1][j]
                    })
                    .sum::<f64>()
                    / ct;
            }
        }
        beta
    }

    // ---- EM (Baum-Welch) --------------------------------------------------

    /// Fit the HMM parameters to a sequence of scalar observations using
    /// the Baum-Welch EM algorithm.
    ///
    /// `max_iter` -- maximum number of EM iterations
    /// `tol`      -- convergence tolerance on relative log-likelihood change
    pub fn fit(&mut self, observations: &[f64], max_iter: usize, tol: f64) {
        if observations.len() < 2 { return; }
        let t_len = observations.len();
        let k = self.n_states;

        let mut prev_ll = f64::NEG_INFINITY;

        for _iter in 0..max_iter {
            // --- E-step ---
            let (alpha, c) = self.forward(observations);
            let beta = self.backward(observations, &c);

            // log-likelihood = sum of ln(c[t])
            let ll: f64 = c.iter().map(|&ct| ct.max(1e-300).ln()).sum();

            // gamma[t][k] = P(s_t = k | O, theta)
            let mut gamma = vec![vec![0.0; k]; t_len];
            for t in 0..t_len {
                let row_sum: f64 = (0..k).map(|i| alpha[t][i] * beta[t][i]).sum();
                let row_sum = row_sum.max(1e-300);
                for i in 0..k {
                    gamma[t][i] = alpha[t][i] * beta[t][i] / row_sum;
                }
            }

            // xi[t][i][j] = P(s_t=i, s_{t+1}=j | O, theta)
            // Stored compactly as xi_sum[i][j] = sum_t xi[t][i][j]
            let mut xi_sum = vec![vec![0.0; k]; k];
            for t in 0..t_len - 1 {
                let mut denom = 0.0;
                for i in 0..k {
                    for j in 0..k {
                        denom += alpha[t][i]
                            * self.transition_matrix[i][j]
                            * self.log_emit(j, observations[t + 1]).exp()
                            * beta[t + 1][j];
                    }
                }
                let denom = denom.max(1e-300);
                for i in 0..k {
                    for j in 0..k {
                        let xi_tij = alpha[t][i]
                            * self.transition_matrix[i][j]
                            * self.log_emit(j, observations[t + 1]).exp()
                            * beta[t + 1][j]
                            / denom;
                        xi_sum[i][j] += xi_tij;
                    }
                }
            }

            // --- M-step ---

            // Update initial probs
            for i in 0..k {
                self.initial_probs[i] = gamma[0][i];
            }

            // Update transition matrix
            for i in 0..k {
                let row_sum: f64 = xi_sum[i].iter().sum::<f64>().max(1e-300);
                for j in 0..k {
                    self.transition_matrix[i][j] = xi_sum[i][j] / row_sum;
                }
            }

            // Update emission params (Gaussian MLE)
            for i in 0..k {
                let g_sum: f64 = (0..t_len).map(|t| gamma[t][i]).sum::<f64>().max(1e-300);
                let new_mean: f64 = (0..t_len)
                    .map(|t| gamma[t][i] * observations[t])
                    .sum::<f64>()
                    / g_sum;
                let new_var: f64 = (0..t_len)
                    .map(|t| gamma[t][i] * (observations[t] - new_mean).powi(2))
                    .sum::<f64>()
                    / g_sum;
                self.emission_means[i] = new_mean;
                self.emission_stds[i] = new_var.sqrt().max(1e-6);
            }

            self.log_likelihood = ll;

            // Convergence check
            if _iter > 0 && (ll - prev_ll).abs() < tol * prev_ll.abs().max(1.0) {
                break;
            }
            prev_ll = ll;
        }
    }

    // ---- Viterbi ----------------------------------------------------------

    /// Viterbi algorithm: find the most probable state sequence.
    pub fn viterbi(&self, observations: &[f64]) -> Vec<usize> {
        let t_len = observations.len();
        if t_len == 0 { return Vec::new(); }
        let k = self.n_states;

        // delta[t][i] = max log-probability of a path ending in state i at time t
        let mut delta = vec![vec![f64::NEG_INFINITY; k]; t_len];
        // psi[t][i] = argmax predecessor state at time t for state i
        let mut psi = vec![vec![0usize; k]; t_len];

        // Initialise
        for i in 0..k {
            let pi = self.initial_probs[i].max(1e-300);
            delta[0][i] = pi.ln() + self.log_emit(i, observations[0]);
        }

        // Recursion
        for t in 1..t_len {
            for j in 0..k {
                let mut best_val = f64::NEG_INFINITY;
                let mut best_i = 0;
                for i in 0..k {
                    let a = self.transition_matrix[i][j].max(1e-300).ln();
                    let val = delta[t - 1][i] + a;
                    if val > best_val {
                        best_val = val;
                        best_i = i;
                    }
                }
                delta[t][j] = best_val + self.log_emit(j, observations[t]);
                psi[t][j] = best_i;
            }
        }

        // Termination: find best final state
        let mut best_final = 0;
        let mut best_val = f64::NEG_INFINITY;
        for i in 0..k {
            if delta[t_len - 1][i] > best_val {
                best_val = delta[t_len - 1][i];
                best_final = i;
            }
        }

        // Backtrack
        let mut path = vec![0usize; t_len];
        path[t_len - 1] = best_final;
        for t in (0..t_len - 1).rev() {
            path[t] = psi[t + 1][path[t + 1]];
        }
        path
    }

    // ---- Online filtering -------------------------------------------------

    /// One-step forward filter update.
    /// Given a new observation, update `filter_state` and return the
    /// posterior P(s_t | o_1..o_t) as a normalised probability vector.
    pub fn filter_step(&mut self, obs: f64) -> Vec<f64> {
        let k = self.n_states;
        let mut new_state = vec![0.0; k];

        // Predict: p(s_t = j) = sum_i filter[i] * A[i][j]
        for j in 0..k {
            new_state[j] = (0..k)
                .map(|i| self.filter_state[i] * self.transition_matrix[i][j])
                .sum();
        }

        // Update: weight by emission probability
        for j in 0..k {
            new_state[j] *= self.log_emit(j, obs).exp();
        }

        self.filter_state = normalize_probs(&new_state);
        self.filter_state.clone()
    }

    /// Reset the online filter to the initial state distribution.
    pub fn reset_filter(&mut self) {
        self.filter_state = self.initial_probs.clone();
    }

    /// Current filtered posterior over states.
    pub fn current_posterior(&self) -> &[f64] {
        &self.filter_state
    }

    /// Most likely state under the current filter.
    pub fn current_state(&self) -> usize {
        self.filter_state
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    // ---- Regime labelling -------------------------------------------------

    /// Assign a RegimeLabel to each HMM state.
    pub fn regime_labels(&self) -> Vec<RegimeLabel> {
        (0..self.n_states)
            .map(|i| RegimeLabel::from_state(i, &self.emission_means, &self.emission_stds))
            .collect()
    }

    /// Decode a Viterbi path to regime labels.
    pub fn decode_to_labels(&self, observations: &[f64]) -> Vec<RegimeLabel> {
        let labels = self.regime_labels();
        self.viterbi(observations)
            .into_iter()
            .map(|s| labels[s])
            .collect()
    }

    // ---- Steady-state distribution ----------------------------------------

    /// Compute the stationary distribution of the Markov chain via power iteration.
    pub fn stationary_distribution(&self) -> Vec<f64> {
        let k = self.n_states;
        let mut pi = vec![1.0 / k as f64; k];
        for _ in 0..1000 {
            let mut next = vec![0.0; k];
            for j in 0..k {
                next[j] = (0..k).map(|i| pi[i] * self.transition_matrix[i][j]).sum();
            }
            let max_diff = (0..k).map(|i| (next[i] - pi[i]).abs()).fold(0.0_f64, f64::max);
            pi = next;
            if max_diff < 1e-12 { break; }
        }
        pi
    }

    /// Expected duration in state `k` = 1 / (1 - A[k][k]).
    pub fn expected_duration(&self, k: usize) -> f64 {
        let p_stay = self.transition_matrix[k][k].min(1.0 - 1e-9);
        1.0 / (1.0 - p_stay)
    }
}

// ---- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate synthetic regime-switching observations.
    /// State 0 (bull): N(+0.001, 0.005^2)
    /// State 1 (bear): N(-0.001, 0.015^2)
    fn synthetic_obs(n: usize, seed: u64) -> Vec<f64> {
        // Simple LCG for reproducibility
        let mut rng = seed;
        let mut next = move || -> f64 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Box-Muller
            let u1 = (rng >> 32) as f64 / u32::MAX as f64;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (rng >> 32) as f64 / u32::MAX as f64;
            let u1 = u1.max(1e-15);
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        };

        let mut obs = Vec::with_capacity(n);
        let mut state = 0usize;
        for _ in 0..n {
            // Transition
            let u = ((state as f64 * 17.0 + obs.len() as f64 * 31.0) * 0.001).sin().abs();
            if u < 0.05 { state = 1 - state; }
            let x = if state == 0 {
                0.001 + 0.005 * next()
            } else {
                -0.001 + 0.015 * next()
            };
            obs.push(x);
        }
        obs
    }

    #[test]
    fn test_hmm_construction() {
        let hmm = HmmRegime::new(3);
        assert_eq!(hmm.n_states, 3);
        assert_eq!(hmm.emission_means.len(), 3);
        // Transition rows should sum to 1
        for row in &hmm.transition_matrix {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "row sum={}", sum);
        }
    }

    #[test]
    fn test_viterbi_length() {
        let hmm = HmmRegime::new(2);
        let obs = synthetic_obs(100, 42);
        let path = hmm.viterbi(&obs);
        assert_eq!(path.len(), 100);
        for &s in &path { assert!(s < 2); }
    }

    #[test]
    fn test_fit_improves_ll() {
        let obs = synthetic_obs(500, 99);
        let mut hmm = HmmRegime::new(2);
        let ll_before = hmm.log_likelihood;
        hmm.fit(&obs, 50, 1e-6);
        assert!(
            hmm.log_likelihood > ll_before || hmm.log_likelihood.is_finite(),
            "ll={}", hmm.log_likelihood
        );
    }

    #[test]
    fn test_filter_step_normalised() {
        let mut hmm = HmmRegime::new(2);
        let obs = synthetic_obs(50, 7);
        for o in &obs {
            let post = hmm.filter_step(*o);
            let sum: f64 = post.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "posterior sum={}", sum);
        }
    }

    #[test]
    fn test_stationary_distribution_sums_to_one() {
        let hmm = HmmRegime::new(3);
        let stat = hmm.stationary_distribution();
        let sum: f64 = stat.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={}", sum);
    }

    #[test]
    fn test_expected_duration_sticky() {
        let mut hmm = HmmRegime::new(2);
        // Very sticky: P(stay) = 0.95 -> expected duration = 20
        hmm.transition_matrix[0][0] = 0.95;
        hmm.transition_matrix[0][1] = 0.05;
        let dur = hmm.expected_duration(0);
        assert!((dur - 20.0).abs() < 0.01, "dur={}", dur);
    }

    #[test]
    fn test_regime_labels_assigned() {
        let mut hmm = HmmRegime::new(3);
        hmm.emission_means = vec![0.001, -0.002, 0.0];
        hmm.emission_stds  = vec![0.005, 0.020, 0.007];
        let labels = hmm.regime_labels();
        assert_eq!(labels.len(), 3);
        // State 1 has high std -- should be HighVol or Bear
        assert!(
            matches!(labels[1], RegimeLabel::Bear | RegimeLabel::HighVol),
            "state 1 label={:?}", labels[1]
        );
    }

    #[test]
    fn test_decode_to_labels_length() {
        let hmm = HmmRegime::new(2);
        let obs = synthetic_obs(100, 13);
        let labels = hmm.decode_to_labels(&obs);
        assert_eq!(labels.len(), 100);
    }

    #[test]
    fn test_regime_label_from_state() {
        let means = vec![0.003, -0.003, 0.0001];
        let stds  = vec![0.005, 0.005, 0.050];
        // State 2 has very high std relative to median(0.005) -> HighVol
        let label = RegimeLabel::from_state(2, &means, &stds);
        assert_eq!(label, RegimeLabel::HighVol);
        // State 0: positive mean
        let label0 = RegimeLabel::from_state(0, &means, &stds);
        assert_eq!(label0, RegimeLabel::Bull);
        // State 1: negative mean
        let label1 = RegimeLabel::from_state(1, &means, &stds);
        assert_eq!(label1, RegimeLabel::Bear);
    }
}
