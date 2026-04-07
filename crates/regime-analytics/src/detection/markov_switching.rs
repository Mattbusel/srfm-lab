// markov_switching.rs
// Markov Switching Regression: Hamilton (1989) model.
// Two regimes, EM estimation.
// Filtered and smoothed probabilities.
// Regime duration distribution.

use serde::{Deserialize, Serialize};
use crate::{normalize_probs, gaussian_log_pdf};

const MAX_EM_ITER: usize = 300;
const EM_TOL: f64 = 1e-7;

/// Parameters for the Markov Switching model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MsParams {
    /// Number of regimes (2).
    pub n_regimes: usize,
    /// Regime means.
    pub mu: Vec<f64>,
    /// Regime standard deviations.
    pub sigma: Vec<f64>,
    /// Transition matrix: p[i][j] = P(s_{t+1}=j | s_t=i).
    pub p: Vec<Vec<f64>>,
    /// Ergodic (stationary) probabilities.
    pub ergodic: Vec<f64>,
}

impl MsParams {
    /// Initialize for 2-regime model from data.
    pub fn init_2regime(y: &[f64]) -> Self {
        let n = y.len();
        let mean = y.iter().sum::<f64>() / n as f64;
        let std = (y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64)
            .sqrt()
            .max(1e-6);

        MsParams {
            n_regimes: 2,
            mu: vec![mean + 0.5 * std, mean - 0.5 * std],
            sigma: vec![std * 0.8, std * 1.5],
            p: vec![vec![0.90, 0.10], vec![0.10, 0.90]],
            ergodic: vec![0.5, 0.5],
        }
    }

    /// Compute ergodic distribution from transition matrix.
    pub fn compute_ergodic(&mut self) {
        let n = self.n_regimes;
        let mut v = vec![1.0 / n as f64; n];
        for _ in 0..1000 {
            let mut nv = vec![0.0f64; n];
            for i in 0..n {
                for j in 0..n {
                    nv[j] += v[i] * self.p[i][j];
                }
            }
            let diff: f64 = nv.iter().zip(v.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
            v = normalize_probs(&nv);
            if diff < 1e-12 {
                break;
            }
        }
        self.ergodic = v;
    }

    /// Expected duration of regime i: 1 / (1 - p[i][i]).
    pub fn expected_duration(&self, regime: usize) -> f64 {
        let p_stay = self.p[regime][regime];
        if (1.0 - p_stay).abs() < 1e-14 {
            f64::INFINITY
        } else {
            1.0 / (1.0 - p_stay)
        }
    }
}

/// Filtered probabilities from the Hamilton filter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilteredProbabilities {
    /// Filtered P(s_t = i | I_t) for each t.
    pub filtered: Vec<Vec<f64>>,
    /// Predicted P(s_t = i | I_{t-1}) for each t.
    pub predicted: Vec<Vec<f64>>,
    /// Smoothed P(s_t = i | I_T) for each t.
    pub smoothed: Vec<Vec<f64>>,
    /// Log-likelihood.
    pub log_likelihood: f64,
}

/// Markov Switching regression model.
pub struct MarkovSwitchingModel {
    pub params: MsParams,
}

impl MarkovSwitchingModel {
    pub fn new(params: MsParams) -> Self {
        MarkovSwitchingModel { params }
    }

    /// Construct default 2-regime model from data.
    pub fn from_data(y: &[f64]) -> Self {
        MarkovSwitchingModel {
            params: MsParams::init_2regime(y),
        }
    }

    /// Hamilton filter: compute filtered probabilities and log-likelihood.
    fn hamilton_filter(&self, y: &[f64]) -> FilteredProbabilities {
        let t = y.len();
        let n = self.params.n_regimes;
        let mut filtered = Vec::with_capacity(t);
        let mut predicted = Vec::with_capacity(t);
        let mut ll = 0.0f64;

        // Initialize with ergodic distribution.
        let mut pred = self.params.ergodic.clone();

        for obs in y {
            predicted.push(pred.clone());

            // Emission probabilities.
            let em: Vec<f64> = (0..n)
                .map(|i| gaussian_log_pdf(*obs, self.params.mu[i], self.params.sigma[i]).exp())
                .collect();

            // Update: f_t = pred * em.
            let unnorm: Vec<f64> = pred.iter().zip(em.iter()).map(|(p, e)| p * e).collect();
            let f_t: f64 = unnorm.iter().sum::<f64>().max(1e-300);
            ll += f_t.ln();
            let filt = unnorm.iter().map(|v| v / f_t).collect::<Vec<f64>>();
            filtered.push(filt.clone());

            // Predict next step: pred_{t+1} = P' * filt_t.
            let mut new_pred = vec![0.0f64; n];
            for j in 0..n {
                for i in 0..n {
                    new_pred[j] += filt[i] * self.params.p[i][j];
                }
            }
            pred = normalize_probs(&new_pred);
        }

        // Kim smoother (backward pass).
        let smoothed = self.kim_smoother(&filtered, &predicted);

        FilteredProbabilities {
            filtered,
            predicted,
            smoothed,
            log_likelihood: ll,
        }
    }

    /// Kim (1994) smoother for smoothed probabilities.
    fn kim_smoother(&self, filtered: &[Vec<f64>], predicted: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let t = filtered.len();
        let n = self.params.n_regimes;
        let mut smoothed = vec![vec![0.0f64; n]; t];
        smoothed[t - 1] = filtered[t - 1].clone();

        for t_idx in (0..t - 1).rev() {
            for i in 0..n {
                let mut s_i = 0.0f64;
                for j in 0..n {
                    let pred_j = predicted[t_idx + 1][j].max(1e-300);
                    s_i += self.params.p[i][j] * smoothed[t_idx + 1][j] / pred_j;
                }
                smoothed[t_idx][i] = filtered[t_idx][i] * s_i;
            }
            // Normalize.
            let norm: f64 = smoothed[t_idx].iter().sum::<f64>().max(1e-300);
            for v in smoothed[t_idx].iter_mut() {
                *v /= norm;
            }
        }
        smoothed
    }

    /// EM algorithm for parameter estimation.
    pub fn fit(&mut self, y: &[f64]) -> f64 {
        let t = y.len();
        let n = self.params.n_regimes;
        let mut prev_ll = f64::NEG_INFINITY;

        for _ in 0..MAX_EM_ITER {
            let fp = self.hamilton_filter(y);
            let ll = fp.log_likelihood;

            if (ll - prev_ll).abs() < EM_TOL {
                break;
            }
            prev_ll = ll;

            // E-step: compute joint smoothed probs P(s_t=i, s_{t+1}=j | I_T).
            // xi[t][i][j] = P(s_t=i, s_{t+1}=j | y, params).
            let mut xi_sum = vec![vec![0.0f64; n]; n];
            let mut gamma_sum = vec![0.0f64; n];
            let mut gamma_t0 = vec![0.0f64; n]; // sum over t for initial state.

            for t_idx in 0..t - 1 {
                let filt_t = &fp.filtered[t_idx];
                let pred_t1 = &fp.predicted[t_idx + 1];
                let sm_t1 = &fp.smoothed[t_idx + 1];

                for i in 0..n {
                    gamma_t0[i] += fp.smoothed[t_idx][i];
                    for j in 0..n {
                        let pred_j = pred_t1[j].max(1e-300);
                        let xi_ij = filt_t[i] * self.params.p[i][j] * sm_t1[j] / pred_j;
                        xi_sum[i][j] += xi_ij;
                    }
                }
            }
            // Add last time point to gamma.
            for i in 0..n {
                gamma_sum[i] = gamma_t0[i] + fp.smoothed[t - 1][i];
            }

            // M-step.
            // Update mu and sigma.
            for i in 0..n {
                let w_sum = gamma_sum[i].max(1e-300);
                let mu_new: f64 = fp
                    .smoothed
                    .iter()
                    .zip(y.iter())
                    .map(|(g, &yi)| g[i] * yi)
                    .sum::<f64>()
                    / w_sum;

                let sigma2_new: f64 = fp
                    .smoothed
                    .iter()
                    .zip(y.iter())
                    .map(|(g, &yi)| g[i] * (yi - mu_new).powi(2))
                    .sum::<f64>()
                    / w_sum;

                self.params.mu[i] = mu_new;
                self.params.sigma[i] = sigma2_new.sqrt().max(1e-6);
            }

            // Update transition matrix.
            for i in 0..n {
                let row_sum: f64 = xi_sum[i].iter().sum::<f64>().max(1e-300);
                for j in 0..n {
                    self.params.p[i][j] = xi_sum[i][j] / row_sum;
                }
                let row = normalize_probs(&self.params.p[i].clone());
                self.params.p[i] = row;
            }

            self.params.compute_ergodic();
        }

        prev_ll
    }

    /// Get filtered probabilities for data.
    pub fn filter(&self, y: &[f64]) -> FilteredProbabilities {
        self.hamilton_filter(y)
    }

    /// Classify each bar by most probable regime (using smoothed probs).
    pub fn classify(&self, y: &[f64]) -> Vec<usize> {
        let fp = self.hamilton_filter(y);
        fp.smoothed
            .iter()
            .map(|p| crate::argmax(p))
            .collect()
    }

    /// Compute duration distribution for each regime.
    /// Returns Vec<(regime, mean_duration, std_duration)>.
    pub fn regime_durations(&self, y: &[f64]) -> Vec<(usize, f64, f64)> {
        let states = self.classify(y);
        let n = self.params.n_regimes;
        let mut result = Vec::new();

        for regime in 0..n {
            // Extract run lengths.
            let mut runs = Vec::new();
            let mut run_len = 0usize;
            for &s in &states {
                if s == regime {
                    run_len += 1;
                } else if run_len > 0 {
                    runs.push(run_len as f64);
                    run_len = 0;
                }
            }
            if run_len > 0 {
                runs.push(run_len as f64);
            }

            if runs.is_empty() {
                result.push((regime, 0.0, 0.0));
                continue;
            }

            let mean = runs.iter().sum::<f64>() / runs.len() as f64;
            let std = (runs
                .iter()
                .map(|v| (v - mean).powi(2))
                .sum::<f64>()
                / runs.len() as f64)
                .sqrt();
            result.push((regime, mean, std));
        }
        result
    }

    /// Log-likelihood of data under the fitted model.
    pub fn log_likelihood(&self, y: &[f64]) -> f64 {
        self.hamilton_filter(y).log_likelihood
    }

    /// AIC: 2k - 2 log L.
    pub fn aic(&self, y: &[f64]) -> f64 {
        let n = self.params.n_regimes;
        // Params: n mu, n sigma, n*(n-1) off-diagonal transitions.
        let k = 2 * n + n * (n - 1);
        2.0 * k as f64 - 2.0 * self.log_likelihood(y)
    }

    /// BIC: k * ln(T) - 2 log L.
    pub fn bic(&self, y: &[f64]) -> f64 {
        let n = self.params.n_regimes;
        let k = 2 * n + n * (n - 1);
        k as f64 * (y.len() as f64).ln() - 2.0 * self.log_likelihood(y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_regime_series(n: usize) -> Vec<f64> {
        use rand::prelude::*;
        let mut rng = StdRng::seed_from_u64(99);
        let mut y = Vec::with_capacity(n);
        let mut state = 0usize;
        for _ in 0..n {
            let r: f64 = rng.gen();
            state = if state == 0 {
                if r < 0.05 { 1 } else { 0 }
            } else {
                if r < 0.08 { 0 } else { 1 }
            };
            let v = if state == 0 {
                0.001 + rng.gen::<f64>() * 0.008
            } else {
                -0.002 + rng.gen::<f64>() * 0.020 - 0.010
            };
            y.push(v);
        }
        y
    }

    #[test]
    fn test_em_increases_likelihood() {
        let y = two_regime_series(300);
        let mut ms = MarkovSwitchingModel::from_data(&y);
        let ll_init = ms.log_likelihood(&y);
        ms.fit(&y);
        let ll_final = ms.log_likelihood(&y);
        assert!(
            ll_final >= ll_init - 1.0,
            "EM should not decrease log-likelihood: {} vs {}",
            ll_final,
            ll_init
        );
    }

    #[test]
    fn test_filtered_probs_sum_to_one() {
        let y = two_regime_series(200);
        let ms = MarkovSwitchingModel::from_data(&y);
        let fp = ms.filter(&y);
        for row in &fp.filtered {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-6, "Filtered probs should sum to 1: {}", sum);
        }
    }

    #[test]
    fn test_regime_durations_positive() {
        let y = two_regime_series(500);
        let mut ms = MarkovSwitchingModel::from_data(&y);
        ms.fit(&y);
        let durations = ms.regime_durations(&y);
        for (_, mean_dur, _) in &durations {
            assert!(*mean_dur >= 0.0);
        }
    }

    #[test]
    fn test_classify_valid_states() {
        let y = two_regime_series(200);
        let mut ms = MarkovSwitchingModel::from_data(&y);
        ms.fit(&y);
        let states = ms.classify(&y);
        assert_eq!(states.len(), 200);
        for &s in &states {
            assert!(s < 2, "State out of range: {}", s);
        }
    }
}
