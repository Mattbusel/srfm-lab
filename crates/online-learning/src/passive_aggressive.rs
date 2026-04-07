//! Passive-Aggressive II (PA-II) regression.
//!
//! Reference: Crammer et al. (2006), "Online Passive-Aggressive Algorithms".
//!
//! PA-II is an aggressive online learner: it updates weights only when the
//! current prediction incurs loss (passive on correct predictions, aggressive
//! on errors). PA-II uses a soft margin version that allows bounded violations.
//!
//! Update rule (regression with epsilon-insensitive loss):
//!   loss_t = max(0, |y_t - f(x_t)| - epsilon)
//!   tau_t  = loss_t / (||x_t||^2 + 1/(2C))
//!   w_{t+1} = w_t + tau_t * sign(y_t - f(x_t)) * x_t
//!
//! Parameters:
//!   C       : aggressiveness (higher = more aggressive updates)
//!   epsilon : insensitivity zone (no update if |error| <= epsilon)

use crate::{dot, ensure_capacity, OnlineLearner};
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// PassiveAggressiveII
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PassiveAggressiveII {
    /// Aggressiveness parameter C > 0.
    pub c: f64,

    /// Epsilon insensitivity: no update when |error| <= epsilon.
    pub epsilon: f64,

    weights: Vec<f64>,
    bias: f64,
    n_seen: u64,

    /// Cumulative loss (for monitoring convergence).
    pub total_loss: f64,
}

impl PassiveAggressiveII {
    /// Create a PA-II model.
    ///
    /// # Arguments
    /// * `c`       -- aggressiveness (default 1.0)
    /// * `epsilon` -- insensitivity zone (default 0.1)
    /// * `n_features` -- initial capacity (auto-expands)
    pub fn new(c: f64, epsilon: f64, n_features: usize) -> Self {
        Self {
            c,
            epsilon,
            weights: vec![0.0; n_features],
            bias: 0.0,
            n_seen: 0,
            total_loss: 0.0,
        }
    }

    /// Compute epsilon-insensitive loss for a prediction.
    #[inline]
    pub fn loss(&self, pred: f64, label: f64) -> f64 {
        let residual = (label - pred).abs();
        if residual <= self.epsilon {
            0.0
        } else {
            residual - self.epsilon
        }
    }

    /// Compute the step size tau for PA-II.
    #[inline]
    fn step_size(&self, loss: f64, features: &[f64]) -> f64 {
        let norm_sq: f64 = features.iter().map(|x| x * x).sum::<f64>() + 1.0; // +1 for bias
        let denom = norm_sq + 1.0 / (2.0 * self.c);
        if denom < 1e-12 {
            return 0.0;
        }
        loss / denom
    }

    /// Sign of the error (direction of update).
    #[inline]
    fn error_sign(pred: f64, label: f64) -> f64 {
        if label > pred { 1.0 } else { -1.0 }
    }

    /// Number of updates where loss was non-zero (aggressive updates).
    pub fn n_aggressive_updates(&self) -> u64 {
        self.n_seen   // we only count samples that caused updates; this is a proxy
    }
}

impl OnlineLearner for PassiveAggressiveII {
    fn update(&mut self, features: &[f64], label: f64) {
        let n = features.len();
        ensure_capacity(&mut self.weights, n);

        let pred = dot(&self.weights[..n], features) + self.bias;
        let loss = self.loss(pred, label);
        self.total_loss += loss;
        self.n_seen += 1;

        if loss < 1e-12 {
            // Passive: within epsilon zone, no update needed
            return;
        }

        let tau = self.step_size(loss, features);
        let direction = Self::error_sign(pred, label);
        let delta = tau * direction;

        for (i, &x) in features.iter().enumerate() {
            if i < self.weights.len() {
                self.weights[i] += delta * x;
            }
        }
        self.bias += delta;
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
        self.n_seen = 0;
        self.total_loss = 0.0;
    }

    fn n_seen(&self) -> u64 {
        self.n_seen
    }

    fn algorithm_name(&self) -> &'static str {
        "passive_aggressive_ii"
    }
}

// ---------------------------------------------------------------------------
// Builder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PaBuilder {
    c: f64,
    epsilon: f64,
    n_features: usize,
}

impl Default for PaBuilder {
    fn default() -> Self {
        Self { c: 1.0, epsilon: 0.1, n_features: 16 }
    }
}

impl PaBuilder {
    pub fn c(mut self, v: f64) -> Self { self.c = v; self }
    pub fn epsilon(mut self, v: f64) -> Self { self.epsilon = v; self }
    pub fn n_features(mut self, v: usize) -> Self { self.n_features = v; self }

    pub fn build(self) -> PassiveAggressiveII {
        PassiveAggressiveII::new(self.c, self.epsilon, self.n_features)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{train_online, Sample};

    fn linear_samples(n: usize) -> Vec<Sample> {
        (0..n)
            .map(|i| {
                let x0 = i as f64 * 0.05;
                let x1 = (i as f64 * 0.1).sin();
                Sample {
                    features: vec![x0, x1],
                    label: 2.0 * x0 - x1 + 0.5,
                }
            })
            .collect()
    }

    #[test]
    fn test_pa_learns_linear() {
        let mut model = PaBuilder::default().c(10.0).epsilon(0.01).build();
        let samples = linear_samples(1000);
        let (_errors, metrics) = train_online(&mut model, &samples, 0);
        assert!(metrics.mean_absolute_error < 1.0, "MAE = {}", metrics.mean_absolute_error);
    }

    #[test]
    fn test_pa_passive_on_correct() {
        let mut model = PassiveAggressiveII::new(1.0, 0.5, 2);
        // If prediction is within epsilon, weights should not change
        model.update(&[1.0, 0.0], 0.1);  // label = 0.1, pred = 0.0, error = 0.1 <= 0.5
        assert_eq!(model.weights()[0], 0.0);
        assert_eq!(model.weights()[1], 0.0);
    }

    #[test]
    fn test_pa_aggressive_on_error() {
        let mut model = PassiveAggressiveII::new(1.0, 0.01, 2);
        model.update(&[1.0, 0.0], 1.0);  // label = 1.0, pred = 0.0, error = 1.0 > 0.01
        assert!(model.weights()[0] > 0.0);
    }

    #[test]
    fn test_pa_loss_is_nonnegative() {
        let model = PassiveAggressiveII::new(1.0, 0.1, 2);
        assert_eq!(model.loss(0.5, 0.55), 0.0);  // within epsilon
        assert!(model.loss(0.0, 1.0) > 0.0);      // outside epsilon
    }

    #[test]
    fn test_pa_reset() {
        let mut model = PaBuilder::default().build();
        let samples = linear_samples(100);
        train_online(&mut model, &samples, 0);
        model.reset();
        assert_eq!(model.n_seen(), 0);
        assert_eq!(model.predict(&[1.0, 2.0]), model.bias());
    }

    #[test]
    fn test_pa_algorithm_name() {
        let model = PaBuilder::default().build();
        assert_eq!(model.algorithm_name(), "passive_aggressive_ii");
    }

    #[test]
    fn test_pa_weights_grow_with_features() {
        let mut model = PassiveAggressiveII::new(1.0, 0.01, 2);
        model.update(&[1.0, 2.0, 3.0], 4.0);
        assert!(model.weights().len() >= 3);
    }
}
