//! Mini-batch Stochastic Gradient Descent with optional momentum and Adam.
//!
//! Three update modes selectable at construction time:
//!   - `SgdMode::Vanilla`   -- plain SGD: w -= lr * grad
//!   - `SgdMode::Momentum`  -- SGD with momentum: v = beta*v - lr*grad; w += v
//!   - `SgdMode::Adam`      -- Adam optimizer (Kingma & Ba, 2015)
//!
//! Supports mini-batch updates via `update_batch`.
//! Loss function: mean squared error (MSE) for regression.

use crate::{dot, ensure_capacity, OnlineLearner};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// SGD mode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SgdMode {
    Vanilla,
    Momentum,
    Adam,
}

impl std::fmt::Display for SgdMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SgdMode::Vanilla  => write!(f, "vanilla"),
            SgdMode::Momentum => write!(f, "momentum"),
            SgdMode::Adam     => write!(f, "adam"),
        }
    }
}

// ---------------------------------------------------------------------------
// SGD model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiniBatchSgd {
    // Hyperparameters
    pub learning_rate: f64,
    pub mode: SgdMode,
    pub l2: f64,             // L2 weight decay
    pub batch_size: usize,   // mini-batch size (1 = online SGD)

    // Momentum hyperparameters
    pub beta_momentum: f64,  // momentum coefficient (default 0.9)

    // Adam hyperparameters
    pub beta1: f64,          // first moment decay (default 0.9)
    pub beta2: f64,          // second moment decay (default 0.999)
    pub epsilon_adam: f64,   // numerical stability (default 1e-8)

    // Weights
    weights: Vec<f64>,
    bias: f64,

    // Momentum state
    velocity: Vec<f64>,
    bias_velocity: f64,

    // Adam state
    m1: Vec<f64>,     // first moment
    m2: Vec<f64>,     // second moment
    bias_m1: f64,
    bias_m2: f64,
    adam_t: u64,      // time step for bias correction

    // Mini-batch accumulator
    grad_accum: Vec<f64>,
    bias_grad_accum: f64,
    batch_count: usize,

    n_seen: u64,
}

impl MiniBatchSgd {
    pub fn new(
        learning_rate: f64,
        mode: SgdMode,
        l2: f64,
        batch_size: usize,
        n_features: usize,
    ) -> Self {
        let bs = batch_size.max(1);
        Self {
            learning_rate,
            mode,
            l2,
            batch_size: bs,
            beta_momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            epsilon_adam: 1e-8,
            weights: vec![0.0; n_features],
            bias: 0.0,
            velocity: vec![0.0; n_features],
            bias_velocity: 0.0,
            m1: vec![0.0; n_features],
            m2: vec![0.0; n_features],
            bias_m1: 0.0,
            bias_m2: 0.0,
            adam_t: 0,
            grad_accum: vec![0.0; n_features],
            bias_grad_accum: 0.0,
            batch_count: 0,
            n_seen: 0,
        }
    }

    /// Ensure all state vectors have capacity for `n` features.
    fn grow(&mut self, n: usize) {
        ensure_capacity(&mut self.weights, n);
        ensure_capacity(&mut self.velocity, n);
        ensure_capacity(&mut self.m1, n);
        ensure_capacity(&mut self.m2, n);
        ensure_capacity(&mut self.grad_accum, n);
    }

    /// Accumulate gradients for the current sample into the mini-batch buffer.
    fn accumulate(&mut self, features: &[f64], grad: f64) {
        let n = features.len();
        self.grow(n);
        for (i, &x) in features.iter().enumerate() {
            if i < self.grad_accum.len() {
                self.grad_accum[i] += grad * x;
            }
        }
        self.bias_grad_accum += grad;
        self.batch_count += 1;
    }

    /// Flush the accumulated mini-batch gradient and apply the weight update.
    fn flush(&mut self) {
        if self.batch_count == 0 {
            return;
        }
        let scale = 1.0 / self.batch_count as f64;
        let n = self.grad_accum.len();

        match self.mode {
            SgdMode::Vanilla => {
                for i in 0..n {
                    let g = self.grad_accum[i] * scale + self.l2 * self.weights[i];
                    self.weights[i] -= self.learning_rate * g;
                }
                let bg = self.bias_grad_accum * scale;
                self.bias -= self.learning_rate * bg;
            }

            SgdMode::Momentum => {
                ensure_capacity(&mut self.velocity, n);
                for i in 0..n {
                    let g = self.grad_accum[i] * scale + self.l2 * self.weights[i];
                    self.velocity[i] = self.beta_momentum * self.velocity[i] - self.learning_rate * g;
                    self.weights[i] += self.velocity[i];
                }
                let bg = self.bias_grad_accum * scale;
                self.bias_velocity = self.beta_momentum * self.bias_velocity - self.learning_rate * bg;
                self.bias += self.bias_velocity;
            }

            SgdMode::Adam => {
                self.adam_t += 1;
                let t = self.adam_t as f64;
                let lr_t = self.learning_rate
                    * (1.0 - self.beta2.powf(t)).sqrt()
                    / (1.0 - self.beta1.powf(t));

                ensure_capacity(&mut self.m1, n);
                ensure_capacity(&mut self.m2, n);
                for i in 0..n {
                    let g = self.grad_accum[i] * scale + self.l2 * self.weights[i];
                    self.m1[i] = self.beta1 * self.m1[i] + (1.0 - self.beta1) * g;
                    self.m2[i] = self.beta2 * self.m2[i] + (1.0 - self.beta2) * g * g;
                    self.weights[i] -= lr_t * self.m1[i] / (self.m2[i].sqrt() + self.epsilon_adam);
                }

                let bg = self.bias_grad_accum * scale;
                self.bias_m1 = self.beta1 * self.bias_m1 + (1.0 - self.beta1) * bg;
                self.bias_m2 = self.beta2 * self.bias_m2 + (1.0 - self.beta2) * bg * bg;
                self.bias -= lr_t * self.bias_m1 / (self.bias_m2.sqrt() + self.epsilon_adam);
            }
        }

        // Reset accumulators
        for g in self.grad_accum.iter_mut() {
            *g = 0.0;
        }
        self.bias_grad_accum = 0.0;
        self.batch_count = 0;
    }

    /// Process a batch of (features, label) pairs in one call.
    pub fn update_batch(&mut self, batch: &[(&[f64], f64)]) {
        for (features, label) in batch {
            let pred = self.predict(features);
            let grad = pred - label;   // MSE gradient: residual
            self.accumulate(features, grad);
        }
        self.flush();
        self.n_seen += batch.len() as u64;
    }

    /// Return current learning rate (alias for documentation purposes).
    pub fn lr(&self) -> f64 {
        self.learning_rate
    }
}

impl OnlineLearner for MiniBatchSgd {
    fn update(&mut self, features: &[f64], label: f64) {
        let pred = self.predict(features);
        let grad = pred - label;   // MSE gradient
        self.accumulate(features, grad);
        self.n_seen += 1;

        if self.batch_count >= self.batch_size {
            self.flush();
        }
    }

    fn predict(&self, features: &[f64]) -> f64 {
        let n = features.len().min(self.weights.len());
        dot(&self.weights[..n], &features[..n]) + self.bias
    }

    fn weights(&self) -> &[f64] {
        &self.weights
    }

    fn bias(&self) -> f64 {
        self.bias
    }

    fn reset(&mut self) {
        let n = self.weights.len();
        self.weights = vec![0.0; n];
        self.bias = 0.0;
        self.velocity = vec![0.0; n];
        self.bias_velocity = 0.0;
        self.m1 = vec![0.0; n];
        self.m2 = vec![0.0; n];
        self.bias_m1 = 0.0;
        self.bias_m2 = 0.0;
        self.adam_t = 0;
        self.grad_accum = vec![0.0; n];
        self.bias_grad_accum = 0.0;
        self.batch_count = 0;
        self.n_seen = 0;
    }

    fn n_seen(&self) -> u64 {
        self.n_seen
    }

    fn algorithm_name(&self) -> &'static str {
        "mini_batch_sgd"
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct SgdBuilder {
    lr: f64,
    mode: SgdMode,
    l2: f64,
    batch_size: usize,
    n_features: usize,
}

impl Default for SgdBuilder {
    fn default() -> Self {
        Self {
            lr: 0.01,
            mode: SgdMode::Adam,
            l2: 0.0,
            batch_size: 32,
            n_features: 16,
        }
    }
}

impl SgdBuilder {
    pub fn lr(mut self, v: f64) -> Self { self.lr = v; self }
    pub fn mode(mut self, v: SgdMode) -> Self { self.mode = v; self }
    pub fn l2(mut self, v: f64) -> Self { self.l2 = v; self }
    pub fn batch_size(mut self, v: usize) -> Self { self.batch_size = v; self }
    pub fn n_features(mut self, v: usize) -> Self { self.n_features = v; self }

    pub fn build(self) -> MiniBatchSgd {
        MiniBatchSgd::new(self.lr, self.mode, self.l2, self.batch_size, self.n_features)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{train_online, Sample};

    /// Normalised samples so that SGD converges stably with standard LR values.
    fn linear_samples(n: usize) -> Vec<Sample> {
        (0..n)
            .map(|i| {
                // Keep features in [-1, 1] range to avoid divergence
                let t = i as f64 / n as f64;   // 0..1
                let x0 = t * 2.0 - 1.0;        // -1..1
                let x1 = (t * std::f64::consts::TAU).sin();
                Sample {
                    features: vec![x0, x1],
                    label: 3.0 * x0 + 1.5 * x1 - 0.5,
                }
            })
            .collect()
    }

    #[test]
    fn test_sgd_vanilla_learns() {
        let mut model = SgdBuilder::default()
            .lr(0.1)
            .mode(SgdMode::Vanilla)
            .batch_size(1)
            .build();
        let samples = linear_samples(2000);
        let (_errors, metrics) = train_online(&mut model, &samples, 0);
        assert!(metrics.mean_absolute_error < 1.0, "MAE = {}", metrics.mean_absolute_error);
    }

    #[test]
    fn test_sgd_momentum_learns() {
        let mut model = SgdBuilder::default()
            .lr(0.05)
            .mode(SgdMode::Momentum)
            .batch_size(16)
            .build();
        let samples = linear_samples(2000);
        let (_errors, metrics) = train_online(&mut model, &samples, 0);
        assert!(metrics.mean_absolute_error < 1.0, "MAE = {}", metrics.mean_absolute_error);
    }

    #[test]
    fn test_sgd_adam_learns() {
        let mut model = SgdBuilder::default()
            .lr(0.01)
            .mode(SgdMode::Adam)
            .batch_size(32)
            .build();
        let samples = linear_samples(2000);
        let (_errors, metrics) = train_online(&mut model, &samples, 0);
        assert!(metrics.mean_absolute_error < 1.0, "MAE = {}", metrics.mean_absolute_error);
    }

    #[test]
    fn test_sgd_batch_update() {
        let mut model = SgdBuilder::default().lr(0.01).batch_size(4).build();
        let samples = linear_samples(100);
        let batch: Vec<(&[f64], f64)> = samples[..4]
            .iter()
            .map(|s| (s.features.as_slice(), s.label))
            .collect();
        model.update_batch(&batch);
        assert_eq!(model.n_seen(), 4);
    }

    #[test]
    fn test_sgd_predict_before_training() {
        let model = SgdBuilder::default().build();
        assert_eq!(model.predict(&[1.0, 2.0]), 0.0);
    }

    #[test]
    fn test_sgd_reset() {
        let mut model = SgdBuilder::default().build();
        let samples = linear_samples(50);
        train_online(&mut model, &samples, 0);
        model.reset();
        assert_eq!(model.n_seen(), 0);
        assert_eq!(model.predict(&[1.0, 2.0]), 0.0);
    }

    #[test]
    fn test_sgd_algorithm_name() {
        let model = SgdBuilder::default().build();
        assert_eq!(model.algorithm_name(), "mini_batch_sgd");
    }

    #[test]
    fn test_sgd_weight_count() {
        let mut model = MiniBatchSgd::new(0.01, SgdMode::Vanilla, 0.0, 1, 3);
        model.update(&[1.0, 2.0, 3.0, 4.0], 5.0);
        assert!(model.weights().len() >= 4);
    }
}
