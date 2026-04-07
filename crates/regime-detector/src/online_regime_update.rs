/// online_regime_update.rs
/// =======================
/// Online regime classifier using stochastic gradient descent with softmax
/// cross-entropy loss.
///
/// Feature vector: [bh_mass_norm, hurst_h, vol_percentile, ofi_zscore, kyle_lambda_zscore]
///   - 5 features, 4 output classes
///   - Regime labels: 0=TRENDING, 1=RANGING, 2=HIGH_VOL, 3=CRISIS
///
/// Learning rate schedule: alpha(t) = alpha_0 / sqrt(t)
/// Weights: 5x4 matrix (input_dim x n_classes), one bias per class.

/// Number of input features.
pub const N_FEATURES: usize = 5;
/// Number of output regime classes.
pub const N_CLASSES: usize = 4;

// ---------------------------------------------------------------------------
// Softmax helper
// ---------------------------------------------------------------------------

/// Compute numerically stable softmax over a slice of logits.
/// Returns a new array of probabilities that sum to 1.
fn softmax(logits: &[f64; N_CLASSES]) -> [f64; N_CLASSES] {
    let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut exps = [0.0_f64; N_CLASSES];
    let mut sum = 0.0_f64;
    for (i, &l) in logits.iter().enumerate() {
        exps[i] = (l - max_logit).exp();
        sum += exps[i];
    }
    for e in &mut exps {
        *e /= sum;
    }
    exps
}

// ---------------------------------------------------------------------------
// OnlineRegimeModel
// ---------------------------------------------------------------------------

/// Online linear classifier for regime detection.
///
/// Uses a single-layer softmax model:
///   logit_k = w_k . x + b_k  for each class k
///   P(k | x) = exp(logit_k) / sum_j exp(logit_j)
///
/// Updated online via SGD on the cross-entropy loss gradient:
///   dL/dw_k = (p_k - y_k) * x   for each class k
///   dL/db_k = (p_k - y_k)
///
/// where y_k = 1 if k == true_regime, else 0.
#[derive(Debug, Clone)]
pub struct OnlineRegimeModel {
    /// Weight matrix: weights[k][f] = weight of feature f for class k.
    weights: [[f64; N_FEATURES]; N_CLASSES],
    /// Bias terms per class.
    biases: [f64; N_CLASSES],
    /// Initial learning rate alpha_0.
    alpha_0: f64,
    /// Step counter (used for learning rate decay).
    step: u64,
    /// L2 regularisation coefficient (0 = no regularisation).
    l2_lambda: f64,
}

impl OnlineRegimeModel {
    /// Create a new model with small random-like fixed initial weights.
    ///
    /// Weights are initialised to a small constant to break symmetry in a
    /// deterministic way suitable for production (no PRNG dependency).
    ///
    /// `alpha_0`   : initial learning rate (typical: 0.01 to 0.1)
    /// `l2_lambda` : L2 regularisation coefficient (typical: 1e-4 to 1e-3)
    pub fn new(alpha_0: f64, l2_lambda: f64) -> Self {
        // Deterministic pseudo-random initialisation using simple arithmetic.
        // Each weight = small signed value derived from class and feature index.
        let mut weights = [[0.0_f64; N_FEATURES]; N_CLASSES];
        for k in 0..N_CLASSES {
            for f in 0..N_FEATURES {
                // Small non-zero values to break symmetry.
                let v = 0.01 * ((k as f64 * 1.7) - (f as f64 * 0.9) + 0.3);
                weights[k][f] = v;
            }
        }
        Self {
            weights,
            biases: [0.0; N_CLASSES],
            alpha_0,
            step: 0,
            l2_lambda,
        }
    }

    /// Predict the most likely regime and its probability given a feature vector.
    ///
    /// Returns `(regime_index, probability)`.
    pub fn predict(&self, features: &[f64; N_FEATURES]) -> (u8, f64) {
        let probs = self.regime_probability_distribution(features);
        let best = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &p)| (i as u8, p))
            .unwrap_or((0, 0.0));
        best
    }

    /// Compute the full softmax probability distribution over all 4 regimes.
    pub fn regime_probability_distribution(&self, features: &[f64; N_FEATURES]) -> [f64; N_CLASSES] {
        let logits = self.compute_logits(features);
        softmax(&logits)
    }

    /// Perform one online SGD update step given a feature vector and true regime.
    ///
    /// Updates weights using the cross-entropy gradient:
    ///   w_k <- w_k - alpha(t) * [(p_k - y_k) * x + lambda * w_k]
    ///   b_k <- b_k - alpha(t) * (p_k - y_k)
    pub fn update(&mut self, features: &[f64; N_FEATURES], true_regime: u8) {
        debug_assert!((true_regime as usize) < N_CLASSES, "true_regime must be in [0, 3]");
        self.step += 1;
        let alpha = self.learning_rate();

        let probs = self.regime_probability_distribution(features);

        for k in 0..N_CLASSES {
            // One-hot indicator.
            let y_k = if k == true_regime as usize { 1.0 } else { 0.0 };
            let grad_base = probs[k] - y_k;

            // Update weights.
            for f in 0..N_FEATURES {
                let grad = grad_base * features[f] + self.l2_lambda * self.weights[k][f];
                self.weights[k][f] -= alpha * grad;
            }
            // Update bias.
            self.biases[k] -= alpha * grad_base;
        }
    }

    /// Current learning rate: alpha_0 / sqrt(step).
    /// Returns alpha_0 before the first step.
    pub fn learning_rate(&self) -> f64 {
        if self.step == 0 {
            return self.alpha_0;
        }
        self.alpha_0 / (self.step as f64).sqrt()
    }

    /// Number of SGD updates performed.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Read-only access to the weight matrix.
    pub fn weights(&self) -> &[[f64; N_FEATURES]; N_CLASSES] {
        &self.weights
    }

    /// Read-only access to the bias vector.
    pub fn biases(&self) -> &[f64; N_CLASSES] {
        &self.biases
    }

    // -- Private helpers --

    fn compute_logits(&self, features: &[f64; N_FEATURES]) -> [f64; N_CLASSES] {
        let mut logits = [0.0_f64; N_CLASSES];
        for k in 0..N_CLASSES {
            let mut dot = self.biases[k];
            for f in 0..N_FEATURES {
                dot += self.weights[k][f] * features[f];
            }
            logits[k] = dot;
        }
        logits
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model() -> OnlineRegimeModel {
        OnlineRegimeModel::new(0.05, 1e-4)
    }

    fn trending_features() -> [f64; N_FEATURES] {
        // bh_mass_norm=0.8 (trending), hurst=0.75 (persistent),
        // vol_percentile=0.2 (low vol), ofi_zscore=0.3, kyle_lambda_z=0.1
        [0.8, 0.75, 0.2, 0.3, 0.1]
    }

    fn crisis_features() -> [f64; N_FEATURES] {
        // bh_mass_norm=0.05, hurst=0.35, vol_percentile=0.97, ofi_zscore=4.0, kyle_lambda_z=3.0
        [0.05, 0.35, 0.97, 4.0, 3.0]
    }

    // 1. Probability distribution sums to 1
    #[test]
    fn test_prob_dist_sums_to_one() {
        let m = make_model();
        let probs = m.regime_probability_distribution(&trending_features());
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "probabilities must sum to 1, got {sum}");
    }

    // 2. All probabilities in [0, 1]
    #[test]
    fn test_prob_dist_in_range() {
        let m = make_model();
        let probs = m.regime_probability_distribution(&crisis_features());
        for (i, &p) in probs.iter().enumerate() {
            assert!(p >= 0.0 && p <= 1.0, "prob[{i}] = {p} out of range");
        }
    }

    // 3. predict returns index in [0, 3]
    #[test]
    fn test_predict_returns_valid_regime() {
        let m = make_model();
        let (regime, prob) = m.predict(&trending_features());
        assert!(regime < N_CLASSES as u8);
        assert!(prob >= 0.0 && prob <= 1.0);
    }

    // 4. After many updates on TRENDING examples, model should increase P(TRENDING)
    #[test]
    fn test_learning_increases_target_prob() {
        let mut m = make_model();
        let feat = trending_features();
        let p_before = m.regime_probability_distribution(&feat)[0];
        for _ in 0..100 {
            m.update(&feat, 0); // true regime = 0 (TRENDING)
        }
        let p_after = m.regime_probability_distribution(&feat)[0];
        assert!(p_after > p_before, "P(TRENDING) should increase after updates: {p_before} -> {p_after}");
    }

    // 5. Learning rate decreases with step count
    #[test]
    fn test_learning_rate_decays() {
        let mut m = make_model();
        let lr_0 = m.learning_rate();
        m.update(&trending_features(), 0);
        let lr_1 = m.learning_rate();
        m.update(&trending_features(), 0);
        let lr_2 = m.learning_rate();
        assert!(lr_0 >= lr_1, "lr should be non-increasing");
        assert!(lr_1 >= lr_2, "lr should be non-increasing");
    }

    // 6. step_count increments on each update
    #[test]
    fn test_step_count() {
        let mut m = make_model();
        assert_eq!(m.step_count(), 0);
        m.update(&trending_features(), 0);
        assert_eq!(m.step_count(), 1);
        m.update(&trending_features(), 1);
        assert_eq!(m.step_count(), 2);
    }

    // 7. After consistent CRISIS training the model assigns highest prob to regime 3
    #[test]
    fn test_learn_crisis_regime() {
        let mut m = OnlineRegimeModel::new(0.1, 0.0);
        let feat = crisis_features();
        for _ in 0..200 {
            m.update(&feat, 3); // true regime = 3 (CRISIS)
        }
        let (regime, _) = m.predict(&feat);
        assert_eq!(regime, 3, "model should predict CRISIS after training, got {regime}");
    }

    // 8. softmax is invariant to constant shift in logits
    #[test]
    fn test_softmax_shift_invariance() {
        let logits = [1.0, 2.0, 3.0, 4.0];
        let shifted = [logits[0] + 10.0, logits[1] + 10.0, logits[2] + 10.0, logits[3] + 10.0];
        let p1 = softmax(&logits);
        let p2 = softmax(&shifted);
        for i in 0..N_CLASSES {
            assert!((p1[i] - p2[i]).abs() < 1e-10, "softmax not shift-invariant at index {i}");
        }
    }

    // 9. Model generalises: different features give different distributions
    #[test]
    fn test_predict_differentiates_features() {
        let mut m = OnlineRegimeModel::new(0.1, 0.0);
        for _ in 0..100 { m.update(&trending_features(), 0); }
        for _ in 0..100 { m.update(&crisis_features(), 3); }
        let (r_trend, _) = m.predict(&trending_features());
        let (r_crisis, _) = m.predict(&crisis_features());
        // After training both should be identifiable (though not guaranteed).
        // At minimum they should not both be the same if the model has learned.
        let p_trend = m.regime_probability_distribution(&trending_features());
        let p_crisis = m.regime_probability_distribution(&crisis_features());
        // P(TRENDING | trending_features) > P(TRENDING | crisis_features)
        assert!(
            p_trend[0] > p_crisis[0],
            "trending features should give higher P(TRENDING) than crisis features: {} vs {}",
            p_trend[0], p_crisis[0]
        );
        let _ = (r_trend, r_crisis); // suppress unused warnings
    }

    // 10. Learning rate at step 0 equals alpha_0
    #[test]
    fn test_learning_rate_initial() {
        let m = OnlineRegimeModel::new(0.05, 1e-4);
        assert!((m.learning_rate() - 0.05).abs() < 1e-10);
    }

    // 11. L2 regularisation pulls weights toward zero
    #[test]
    fn test_l2_regularisation_shrinks_weights() {
        // Train with high L2.
        let mut m_reg = OnlineRegimeModel::new(0.1, 1.0);
        let mut m_noreg = OnlineRegimeModel::new(0.1, 0.0);
        let feat = trending_features();
        for _ in 0..50 {
            m_reg.update(&feat, 0);
            m_noreg.update(&feat, 0);
        }
        let norm_reg: f64 = m_reg.weights().iter()
            .flat_map(|row| row.iter())
            .map(|&w| w * w)
            .sum::<f64>()
            .sqrt();
        let norm_noreg: f64 = m_noreg.weights().iter()
            .flat_map(|row| row.iter())
            .map(|&w| w * w)
            .sum::<f64>()
            .sqrt();
        assert!(norm_reg < norm_noreg, "L2 should produce smaller weights: {norm_reg} vs {norm_noreg}");
    }
}
