//! FTRL-Proximal (Follow The Regularized Leader with Proximal L1 regularisation).
//!
//! Reference: McMahan et al. (2013), "Ad Click Prediction: a View from the Trenches".
//!
//! Update rule per weight w_i:
//!   z_i += g_i - sigma_i * w_i
//!   n_i += g_i^2
//!   sigma_i = (sqrt(n_i) - sqrt(n_i - g_i^2)) / alpha
//!
//! Weight recovery:
//!   w_i = 0                                   if |z_i| <= lambda1
//!   w_i = -(z_i - sign(z_i)*lambda1) /
//!           (lambda2 + (beta + sqrt(n_i)) / alpha)   otherwise
//!
//! Parameters:
//!   alpha   : learning rate (step size)
//!   beta    : learning rate smoothing (prevents large steps early on)
//!   lambda1 : L1 regularisation coefficient (induces sparsity)
//!   lambda2 : L2 regularisation coefficient

use crate::{dot, ensure_capacity, OnlineLearner};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// FTRL state
// ---------------------------------------------------------------------------

/// Per-feature accumulators for the FTRL-Proximal algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FtrlState {
    /// z_i: running sum of adjusted gradients.
    pub z: Vec<f64>,
    /// n_i: running sum of squared gradients.
    pub n: Vec<f64>,
}

impl FtrlState {
    fn new(n_features: usize) -> Self {
        Self {
            z: vec![0.0; n_features],
            n: vec![0.0; n_features],
        }
    }

    fn ensure_capacity(&mut self, n: usize) {
        if self.z.len() < n {
            self.z.resize(n, 0.0);
            self.n.resize(n, 0.0);
        }
    }
}

// ---------------------------------------------------------------------------
// FtrlProximal model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FtrlProximal {
    // Hyperparameters
    pub alpha: f64,
    pub beta: f64,
    pub lambda1: f64,
    pub lambda2: f64,

    // Model state
    state: FtrlState,
    weights_cache: Vec<f64>,   // w_i recovered from z_i, n_i
    bias_z: f64,
    bias_n: f64,
    n_seen: u64,
}

impl FtrlProximal {
    /// Create a new FTRL-Proximal model.
    ///
    /// # Arguments
    /// * `alpha`   -- learning rate (default 0.1)
    /// * `beta`    -- smoothing parameter (default 1.0)
    /// * `lambda1` -- L1 coefficient (default 0.0)
    /// * `lambda2` -- L2 coefficient (default 0.0)
    /// * `n_features` -- initial feature dimension (auto-expands)
    pub fn new(alpha: f64, beta: f64, lambda1: f64, lambda2: f64, n_features: usize) -> Self {
        Self {
            alpha,
            beta,
            lambda1,
            lambda2,
            state: FtrlState::new(n_features),
            weights_cache: vec![0.0; n_features],
            bias_z: 0.0,
            bias_n: 0.0,
            n_seen: 0,
        }
    }

    /// Recover weight w_i from z_i and n_i accumulator.
    #[inline]
    fn recover_weight(&self, z_i: f64, n_i: f64) -> f64 {
        let abs_z = z_i.abs();
        if abs_z <= self.lambda1 {
            return 0.0;
        }
        let sign_z = if z_i > 0.0 { 1.0_f64 } else { -1.0_f64 };
        let denom = self.lambda2 + (self.beta + n_i.sqrt()) / self.alpha;
        if denom.abs() < 1e-12 {
            return 0.0;
        }
        -(z_i - sign_z * self.lambda1) / denom
    }

    /// Rebuild the weights_cache from current z/n accumulators.
    fn rebuild_weights(&mut self) {
        ensure_capacity(&mut self.weights_cache, self.state.z.len());
        for i in 0..self.state.z.len() {
            self.weights_cache[i] = self.recover_weight(self.state.z[i], self.state.n[i]);
        }
    }

    /// Gradient of squared loss: g = prediction - label (residual).
    #[inline]
    fn gradient_sq_loss(pred: f64, label: f64) -> f64 {
        pred - label
    }

    /// Update a single feature's z and n accumulator.
    #[inline]
    fn update_feature(&mut self, i: usize, g: f64) {
        let n_old = self.state.n[i];
        let n_new = n_old + g * g;
        let sigma = (n_new.sqrt() - n_old.sqrt()) / self.alpha;
        self.state.z[i] += g - sigma * self.weights_cache[i];
        self.state.n[i] = n_new;
    }

    /// Update the bias accumulator (no L1/L2 on bias).
    #[inline]
    fn update_bias(&mut self, g: f64) {
        let n_old = self.bias_n;
        let n_new = n_old + g * g;
        let sigma = (n_new.sqrt() - n_old.sqrt()) / self.alpha;
        self.bias_z += g - sigma * self.recover_weight(self.bias_z, self.bias_n);
        self.bias_n = n_new;
    }

    /// Recover the current bias from its accumulators.
    #[inline]
    fn recover_bias(&self) -> f64 {
        // Bias uses L1 = 0, L2 = 0 (no regularisation on intercept)
        let denom = (self.beta + self.bias_n.sqrt()) / self.alpha;
        if denom.abs() < 1e-12 {
            return 0.0;
        }
        -self.bias_z / denom
    }

    /// Sparsity ratio: fraction of weights that are exactly 0 (due to L1).
    pub fn sparsity(&self) -> f64 {
        let n = self.weights_cache.len();
        if n == 0 {
            return 0.0;
        }
        let zeros = self.weights_cache.iter().filter(|&&w| w == 0.0).count();
        zeros as f64 / n as f64
    }
}

impl OnlineLearner for FtrlProximal {
    fn update(&mut self, features: &[f64], label: f64) {
        let n = features.len();
        self.state.ensure_capacity(n);
        ensure_capacity(&mut self.weights_cache, n);
        self.rebuild_weights();

        let pred = dot(&self.weights_cache[..n], features) + self.recover_bias();
        let g = Self::gradient_sq_loss(pred, label);

        for i in 0..n {
            self.update_feature(i, g * features[i]);
        }
        self.update_bias(g);
        self.rebuild_weights();
        self.n_seen += 1;
    }

    fn predict(&self, features: &[f64]) -> f64 {
        let n = features.len().min(self.weights_cache.len());
        dot(&self.weights_cache[..n], &features[..n]) + self.recover_bias()
    }

    fn weights(&self) -> &[f64] {
        &self.weights_cache
    }

    fn bias(&self) -> f64 {
        self.recover_bias()
    }

    fn reset(&mut self) {
        let n = self.state.z.len();
        self.state = FtrlState::new(n);
        self.weights_cache = vec![0.0; n];
        self.bias_z = 0.0;
        self.bias_n = 0.0;
        self.n_seen = 0;
    }

    fn n_seen(&self) -> u64 {
        self.n_seen
    }

    fn algorithm_name(&self) -> &'static str {
        "ftrl_proximal"
    }
}

// ---------------------------------------------------------------------------
// Convenience builder
// ---------------------------------------------------------------------------

/// Builder for `FtrlProximal` with sensible defaults.
#[derive(Debug, Clone)]
pub struct FtrlBuilder {
    alpha: f64,
    beta: f64,
    lambda1: f64,
    lambda2: f64,
    n_features: usize,
}

impl Default for FtrlBuilder {
    fn default() -> Self {
        Self {
            alpha: 0.1,
            beta: 1.0,
            lambda1: 0.0,
            lambda2: 0.0,
            n_features: 16,
        }
    }
}

impl FtrlBuilder {
    pub fn alpha(mut self, v: f64) -> Self { self.alpha = v; self }
    pub fn beta(mut self, v: f64) -> Self { self.beta = v; self }
    pub fn lambda1(mut self, v: f64) -> Self { self.lambda1 = v; self }
    pub fn lambda2(mut self, v: f64) -> Self { self.lambda2 = v; self }
    pub fn n_features(mut self, v: usize) -> Self { self.n_features = v; self }

    pub fn build(self) -> FtrlProximal {
        FtrlProximal::new(self.alpha, self.beta, self.lambda1, self.lambda2, self.n_features)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{train_online, Sample};

    fn make_linear_samples(n: usize, noise: f64) -> Vec<Sample> {
        (0..n)
            .map(|i| {
                let x0 = i as f64 * 0.05;
                let x1 = (i as f64 * 0.1).cos();
                let label = 1.5 * x0 - 0.8 * x1 + noise * ((i % 7) as f64 - 3.0) * 0.01;
                Sample { features: vec![x0, x1], label }
            })
            .collect()
    }

    #[test]
    fn test_ftrl_learns_linear() {
        let mut model = FtrlBuilder::default().alpha(0.5).build();
        let samples = make_linear_samples(2000, 0.1);
        let (_errors, metrics) = train_online(&mut model, &samples, 0);
        // FTRL on this non-normalised problem converges to a reasonable MAE
        assert!(metrics.mean_absolute_error < 2.0, "MAE = {}", metrics.mean_absolute_error);
    }

    #[test]
    fn test_ftrl_l1_induces_sparsity() {
        // With very large L1, weights whose gradients never exceed lambda1 stay zero.
        let mut model = FtrlBuilder::default()
            .alpha(0.1)
            .lambda1(100.0)   // very large: forces most weights to zero
            .n_features(10)
            .build();

        let samples: Vec<Sample> = (0..500)
            .map(|i| {
                let mut features = vec![0.0; 10];
                features[0] = i as f64 * 0.01;
                features[1] = (i as f64 * 0.05).sin();
                for j in 2..10 {
                    features[j] = ((i * j) as f64 * 0.001).cos();
                }
                let label = 2.0 * features[0] + features[1];
                Sample { features, label }
            })
            .collect();

        train_online(&mut model, &samples, 0);
        // With very large L1, most weights remain exactly zero
        let sparsity = model.sparsity();
        assert!(sparsity > 0.0, "Expected some sparsity with large lambda1, got {}", sparsity);
    }

    #[test]
    fn test_ftrl_predict_before_training() {
        let model = FtrlBuilder::default().build();
        let pred = model.predict(&[1.0, 2.0]);
        assert_eq!(pred, 0.0);
    }

    #[test]
    fn test_ftrl_reset_clears_state() {
        let mut model = FtrlBuilder::default().alpha(0.5).build();
        let samples = make_linear_samples(100, 0.0);
        train_online(&mut model, &samples, 0);
        assert!(model.n_seen() == 100);
        model.reset();
        assert_eq!(model.n_seen(), 0);
        let pred = model.predict(&[1.0, 2.0]);
        assert_eq!(pred, 0.0);
    }

    #[test]
    fn test_ftrl_algorithm_name() {
        let model = FtrlBuilder::default().build();
        assert_eq!(model.algorithm_name(), "ftrl_proximal");
    }

    #[test]
    fn test_ftrl_weight_count_grows() {
        let mut model = FtrlProximal::new(0.1, 1.0, 0.0, 0.0, 2);
        model.update(&[1.0, 2.0, 3.0, 4.0], 5.0);
        assert!(model.weights().len() >= 4);
    }
}
