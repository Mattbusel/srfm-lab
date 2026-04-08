// ensemble.rs — Model ensemble, online weight update, model selection, calibration
use crate::tensor::Tensor;

/// Generic model prediction trait
pub trait Predictor {
    fn predict(&self, features: &[f64]) -> f64;
    fn predict_batch(&self, features: &[Vec<f64>]) -> Vec<f64> {
        features.iter().map(|f| self.predict(f)).collect()
    }
}

/// Boxed predictor for dynamic dispatch
pub type BoxPredictor = Box<dyn Predictor + Send + Sync>;

/// Simple function-based predictor wrapper
pub struct FnPredictor {
    pub func: fn(&[f64]) -> f64,
}

impl Predictor for FnPredictor {
    fn predict(&self, features: &[f64]) -> f64 { (self.func)(features) }
}

/// Ensemble methods
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum EnsembleMethod {
    Average,
    WeightedAverage,
    Median,
    TrimmedMean(f64), // trim fraction
    Stacking,
    Voting,           // classification majority vote
    SoftVoting,       // classification probability average
    BayesianModelAveraging,
}

/// Model ensemble with weights
#[derive(Clone, Debug)]
pub struct Ensemble {
    pub weights: Vec<f64>,
    pub method: EnsembleMethod,
    pub stacking_weights: Option<Tensor>, // for stacking: [num_models, 1] or meta-learner
    pub stacking_bias: f64,
}

impl Ensemble {
    pub fn uniform(num_models: usize) -> Self {
        let w = 1.0 / num_models as f64;
        Self {
            weights: vec![w; num_models],
            method: EnsembleMethod::Average,
            stacking_weights: None,
            stacking_bias: 0.0,
        }
    }

    pub fn weighted(weights: Vec<f64>) -> Self {
        let sum: f64 = weights.iter().sum();
        let norm: Vec<f64> = weights.iter().map(|&w| w / sum).collect();
        Self {
            weights: norm,
            method: EnsembleMethod::WeightedAverage,
            stacking_weights: None,
            stacking_bias: 0.0,
        }
    }

    pub fn stacking(meta_weights: Vec<f64>, bias: f64) -> Self {
        let n = meta_weights.len();
        Self {
            weights: vec![1.0 / n as f64; n],
            method: EnsembleMethod::Stacking,
            stacking_weights: Some(Tensor::from_vec(meta_weights, &[n])),
            stacking_bias: bias,
        }
    }

    pub fn combine(&self, predictions: &[f64]) -> f64 {
        match self.method {
            EnsembleMethod::Average => {
                predictions.iter().sum::<f64>() / predictions.len() as f64
            }
            EnsembleMethod::WeightedAverage => {
                predictions.iter().zip(self.weights.iter())
                    .map(|(&p, &w)| p * w).sum()
            }
            EnsembleMethod::Median => {
                let mut sorted = predictions.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let n = sorted.len();
                if n % 2 == 0 { (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 }
                else { sorted[n / 2] }
            }
            EnsembleMethod::TrimmedMean(frac) => {
                let mut sorted = predictions.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let trim = (sorted.len() as f64 * frac) as usize;
                let trimmed = &sorted[trim..sorted.len() - trim];
                if trimmed.is_empty() { return sorted[sorted.len() / 2]; }
                trimmed.iter().sum::<f64>() / trimmed.len() as f64
            }
            EnsembleMethod::Stacking => {
                if let Some(ref sw) = self.stacking_weights {
                    let dot: f64 = predictions.iter().zip(sw.data.iter())
                        .map(|(&p, &w)| p * w).sum();
                    dot + self.stacking_bias
                } else {
                    predictions.iter().sum::<f64>() / predictions.len() as f64
                }
            }
            EnsembleMethod::Voting => {
                // Round to class labels then majority vote
                let mut counts = std::collections::HashMap::new();
                for &p in predictions {
                    let cls = p.round() as i64;
                    *counts.entry(cls).or_insert(0) += 1;
                }
                *counts.iter().max_by_key(|(_, &v)| v).unwrap().0 as f64
            }
            EnsembleMethod::SoftVoting => {
                // Average probabilities
                predictions.iter().sum::<f64>() / predictions.len() as f64
            }
            EnsembleMethod::BayesianModelAveraging => {
                // BMA with log-likelihood weights
                predictions.iter().zip(self.weights.iter())
                    .map(|(&p, &w)| p * w).sum()
            }
        }
    }

    pub fn combine_multi(&self, predictions: &[Vec<f64>]) -> Vec<f64> {
        // predictions[model_idx][output_idx]
        if predictions.is_empty() { return vec![]; }
        let n_out = predictions[0].len();
        let n_models = predictions.len();
        let mut result = vec![0.0; n_out];
        for o in 0..n_out {
            let preds: Vec<f64> = predictions.iter().map(|p| p[o]).collect();
            result[o] = self.combine(&preds);
        }
        result
    }

    pub fn num_models(&self) -> usize { self.weights.len() }
}

/// Online weight updater for ensemble using exponential weighting
#[derive(Clone, Debug)]
pub struct OnlineWeightUpdater {
    pub weights: Vec<f64>,
    pub learning_rate: f64,
    pub losses: Vec<Vec<f64>>,  // [model][timestep]
    pub cumulative_loss: Vec<f64>,
    pub method: OnlineMethod,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OnlineMethod {
    ExponentialWeighting,
    FollowTheLeader,
    FollowTheRegularizedLeader(f64), // regularization
    Hedge,
    AdaptiveHedge,
}

impl OnlineWeightUpdater {
    pub fn new(num_models: usize, learning_rate: f64, method: OnlineMethod) -> Self {
        let w = 1.0 / num_models as f64;
        Self {
            weights: vec![w; num_models],
            learning_rate,
            losses: vec![vec![]; num_models],
            cumulative_loss: vec![0.0; num_models],
            method,
        }
    }

    pub fn update(&mut self, losses: &[f64]) {
        let n = self.weights.len();
        assert_eq!(losses.len(), n);

        for (i, &l) in losses.iter().enumerate() {
            self.losses[i].push(l);
            self.cumulative_loss[i] += l;
        }

        match self.method {
            OnlineMethod::ExponentialWeighting | OnlineMethod::Hedge => {
                let eta = self.learning_rate;
                let mut new_weights: Vec<f64> = self.weights.iter().zip(losses.iter())
                    .map(|(&w, &l)| w * (-eta * l).exp())
                    .collect();
                let sum: f64 = new_weights.iter().sum();
                if sum > 0.0 { for w in new_weights.iter_mut() { *w /= sum; } }
                self.weights = new_weights;
            }
            OnlineMethod::FollowTheLeader => {
                let best = self.cumulative_loss.iter().enumerate()
                    .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(i, _)| i).unwrap_or(0);
                self.weights = vec![0.0; n];
                self.weights[best] = 1.0;
            }
            OnlineMethod::FollowTheRegularizedLeader(lambda) => {
                let eta = self.learning_rate;
                let mut new_weights = vec![0.0; n];
                for i in 0..n {
                    new_weights[i] = (-eta * self.cumulative_loss[i]).exp();
                }
                // add regularization (entropy)
                let sum: f64 = new_weights.iter().sum();
                if sum > 0.0 { for w in new_weights.iter_mut() { *w /= sum; } }
                self.weights = new_weights;
            }
            OnlineMethod::AdaptiveHedge => {
                let t = self.losses[0].len() as f64;
                let eta = (2.0 * (n as f64).ln() / t.max(1.0)).sqrt();
                let mut new_weights: Vec<f64> = (0..n).map(|i| {
                    (-eta * self.cumulative_loss[i]).exp()
                }).collect();
                let sum: f64 = new_weights.iter().sum();
                if sum > 0.0 { for w in new_weights.iter_mut() { *w /= sum; } }
                self.weights = new_weights;
            }
        }
    }

    pub fn get_weights(&self) -> &[f64] { &self.weights }

    pub fn best_model(&self) -> usize {
        self.cumulative_loss.iter().enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i).unwrap_or(0)
    }

    pub fn regret(&self) -> f64 {
        let total: f64 = (0..self.losses[0].len()).map(|t| {
            self.weights.iter().enumerate()
                .map(|(i, &w)| w * self.losses[i][t]).sum::<f64>()
        }).sum();
        let best_cum = self.cumulative_loss.iter().cloned()
            .fold(f64::INFINITY, f64::min);
        total - best_cum
    }
}

/// Model selection criteria
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SelectionCriterion {
    AIC,
    BIC,
    AICc,
    HQIC,
    CrossValidation(usize), // k-fold
    LOOCV,
}

/// Model selection result
#[derive(Clone, Debug)]
pub struct ModelSelectionResult {
    pub scores: Vec<f64>,
    pub best_idx: usize,
    pub best_score: f64,
    pub criterion: SelectionCriterion,
}

/// Compute AIC = 2k - 2ln(L) where k=num_params, L=likelihood
pub fn aic(log_likelihood: f64, num_params: usize) -> f64 {
    2.0 * num_params as f64 - 2.0 * log_likelihood
}

/// Compute BIC = k*ln(n) - 2ln(L)
pub fn bic(log_likelihood: f64, num_params: usize, num_samples: usize) -> f64 {
    num_params as f64 * (num_samples as f64).ln() - 2.0 * log_likelihood
}

/// Corrected AIC for small samples
pub fn aicc(log_likelihood: f64, num_params: usize, num_samples: usize) -> f64 {
    let k = num_params as f64;
    let n = num_samples as f64;
    let base = aic(log_likelihood, num_params);
    if n - k - 1.0 > 0.0 {
        base + 2.0 * k * (k + 1.0) / (n - k - 1.0)
    } else {
        f64::INFINITY
    }
}

/// Hannan-Quinn information criterion
pub fn hqic(log_likelihood: f64, num_params: usize, num_samples: usize) -> f64 {
    let k = num_params as f64;
    let n = num_samples as f64;
    -2.0 * log_likelihood + 2.0 * k * (n.ln().ln())
}

/// Log-likelihood from MSE (Gaussian assumption)
pub fn log_likelihood_gaussian(mse: f64, num_samples: usize) -> f64 {
    let n = num_samples as f64;
    -n / 2.0 * (2.0 * std::f64::consts::PI * mse).ln() - n / 2.0
}

/// Select best model based on information criterion
pub fn select_model(
    mses: &[f64],
    num_params: &[usize],
    num_samples: usize,
    criterion: SelectionCriterion,
) -> ModelSelectionResult {
    let n = mses.len();
    let scores: Vec<f64> = match criterion {
        SelectionCriterion::AIC => {
            mses.iter().zip(num_params.iter()).map(|(&mse, &k)| {
                aic(log_likelihood_gaussian(mse, num_samples), k)
            }).collect()
        }
        SelectionCriterion::BIC => {
            mses.iter().zip(num_params.iter()).map(|(&mse, &k)| {
                bic(log_likelihood_gaussian(mse, num_samples), k, num_samples)
            }).collect()
        }
        SelectionCriterion::AICc => {
            mses.iter().zip(num_params.iter()).map(|(&mse, &k)| {
                aicc(log_likelihood_gaussian(mse, num_samples), k, num_samples)
            }).collect()
        }
        SelectionCriterion::HQIC => {
            mses.iter().zip(num_params.iter()).map(|(&mse, &k)| {
                hqic(log_likelihood_gaussian(mse, num_samples), k, num_samples)
            }).collect()
        }
        _ => mses.to_vec(), // for CV, use MSEs directly
    };

    let (best_idx, &best_score) = scores.iter().enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    ModelSelectionResult { scores, best_idx, best_score, criterion }
}

/// Platt scaling calibration for converting raw scores to probabilities
#[derive(Clone, Debug)]
pub struct PlattScaling {
    pub a: f64,
    pub b: f64,
}

impl PlattScaling {
    pub fn new(a: f64, b: f64) -> Self { Self { a, b } }

    /// Fit Platt scaling from raw scores and binary labels
    pub fn fit(scores: &[f64], labels: &[f64], max_iter: usize, lr: f64) -> Self {
        let n = scores.len() as f64;
        let n_pos = labels.iter().filter(|&&l| l > 0.5).count() as f64;
        let n_neg = n - n_pos;
        // Target probabilities
        let t_pos = (n_pos + 1.0) / (n_pos + 2.0);
        let t_neg = 1.0 / (n_neg + 2.0);
        let targets: Vec<f64> = labels.iter().map(|&l| if l > 0.5 { t_pos } else { t_neg }).collect();

        let mut a = 0.0f64;
        let mut b = ((n_neg + 1.0) / (n_pos + 1.0)).ln();

        for _ in 0..max_iter {
            let mut grad_a = 0.0;
            let mut grad_b = 0.0;
            let mut hess_aa = 0.0;
            let mut hess_bb = 0.0;
            let mut hess_ab = 0.0;

            for i in 0..scores.len() {
                let f = a * scores[i] + b;
                let p = 1.0 / (1.0 + (-f).exp());
                let t = targets[i];
                let d1 = p - t;
                let d2 = p * (1.0 - p);
                grad_a += d1 * scores[i];
                grad_b += d1;
                hess_aa += d2 * scores[i] * scores[i];
                hess_bb += d2;
                hess_ab += d2 * scores[i];
            }

            // Newton step
            let det = hess_aa * hess_bb - hess_ab * hess_ab;
            if det.abs() < 1e-15 {
                a -= lr * grad_a;
                b -= lr * grad_b;
            } else {
                a -= (hess_bb * grad_a - hess_ab * grad_b) / det;
                b -= (hess_aa * grad_b - hess_ab * grad_a) / det;
            }
        }

        Self { a, b }
    }

    pub fn calibrate(&self, score: f64) -> f64 {
        1.0 / (1.0 + (-(self.a * score + self.b)).exp())
    }

    pub fn calibrate_batch(&self, scores: &[f64]) -> Vec<f64> {
        scores.iter().map(|&s| self.calibrate(s)).collect()
    }
}

/// Isotonic regression calibration
#[derive(Clone, Debug)]
pub struct IsotonicCalibration {
    pub x_values: Vec<f64>,
    pub y_values: Vec<f64>,
}

impl IsotonicCalibration {
    pub fn fit(scores: &[f64], labels: &[f64]) -> Self {
        let n = scores.len();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());

        let sorted_x: Vec<f64> = indices.iter().map(|&i| scores[i]).collect();
        let sorted_y: Vec<f64> = indices.iter().map(|&i| labels[i]).collect();

        // Pool Adjacent Violators (PAVA)
        let mut y_iso = sorted_y.clone();
        let mut weights = vec![1.0; n];

        let mut i = 0;
        while i < n - 1 {
            if y_iso[i] > y_iso[i + 1] {
                // pool
                let combined_w = weights[i] + weights[i + 1];
                let combined_y = (weights[i] * y_iso[i] + weights[i + 1] * y_iso[i + 1]) / combined_w;
                y_iso[i] = combined_y;
                weights[i] = combined_w;
                y_iso.remove(i + 1);
                weights.remove(i + 1);
                // need to re-check backwards
                if i > 0 { i -= 1; }
            } else {
                i += 1;
            }
        }

        // Build lookup — we need to expand back to original grid
        let mut x_values = Vec::new();
        let mut y_values = Vec::new();
        let mut idx = 0;
        for (yi, &yv) in y_iso.iter().enumerate() {
            let w = weights[yi] as usize;
            let x_start = sorted_x[idx];
            let x_end = sorted_x[(idx + w - 1).min(n - 1)];
            x_values.push((x_start + x_end) / 2.0);
            y_values.push(yv);
            idx += w;
        }

        Self { x_values, y_values }
    }

    pub fn calibrate(&self, score: f64) -> f64 {
        if self.x_values.is_empty() { return score; }
        if score <= self.x_values[0] { return self.y_values[0]; }
        if score >= *self.x_values.last().unwrap() { return *self.y_values.last().unwrap(); }

        // binary search + interpolation
        let mut lo = 0;
        let mut hi = self.x_values.len() - 1;
        while lo < hi - 1 {
            let mid = (lo + hi) / 2;
            if self.x_values[mid] <= score { lo = mid; } else { hi = mid; }
        }
        let frac = (score - self.x_values[lo]) / (self.x_values[hi] - self.x_values[lo]);
        self.y_values[lo] + frac * (self.y_values[hi] - self.y_values[lo])
    }

    pub fn calibrate_batch(&self, scores: &[f64]) -> Vec<f64> {
        scores.iter().map(|&s| self.calibrate(s)).collect()
    }
}

/// Temperature scaling for neural network calibration
#[derive(Clone, Debug)]
pub struct TemperatureScaling {
    pub temperature: f64,
}

impl TemperatureScaling {
    pub fn new(temperature: f64) -> Self { Self { temperature } }

    pub fn fit(logits: &[Vec<f64>], labels: &[usize], max_iter: usize) -> Self {
        let mut temp = 1.0f64;
        let lr = 0.01;
        for _ in 0..max_iter {
            let mut grad = 0.0;
            for (log, &label) in logits.iter().zip(labels.iter()) {
                let scaled: Vec<f64> = log.iter().map(|&l| l / temp).collect();
                let max_s = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exp_s: Vec<f64> = scaled.iter().map(|&s| (s - max_s).exp()).collect();
                let sum: f64 = exp_s.iter().sum();
                let probs: Vec<f64> = exp_s.iter().map(|&e| e / sum).collect();
                // gradient of NLL w.r.t. temperature
                for (i, &p) in probs.iter().enumerate() {
                    let target = if i == label { 1.0 } else { 0.0 };
                    grad += (p - target) * (-log[i] / (temp * temp));
                }
            }
            temp -= lr * grad / logits.len() as f64;
            temp = temp.max(0.01).min(100.0);
        }
        Self { temperature: temp }
    }

    pub fn calibrate(&self, logits: &[f64]) -> Vec<f64> {
        let scaled: Vec<f64> = logits.iter().map(|&l| l / self.temperature).collect();
        let max_s = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_s: Vec<f64> = scaled.iter().map(|&s| (s - max_s).exp()).collect();
        let sum: f64 = exp_s.iter().sum();
        exp_s.iter().map(|&e| e / sum).collect()
    }
}

/// Expected Calibration Error
pub fn expected_calibration_error(probs: &[f64], labels: &[f64], num_bins: usize) -> f64 {
    let n = probs.len();
    let mut bin_sums = vec![0.0; num_bins];
    let mut bin_correct = vec![0.0; num_bins];
    let mut bin_counts = vec![0usize; num_bins];

    for i in 0..n {
        let bin = ((probs[i] * num_bins as f64) as usize).min(num_bins - 1);
        bin_sums[bin] += probs[i];
        bin_correct[bin] += labels[i];
        bin_counts[bin] += 1;
    }

    let mut ece = 0.0;
    for b in 0..num_bins {
        if bin_counts[b] > 0 {
            let avg_conf = bin_sums[b] / bin_counts[b] as f64;
            let avg_acc = bin_correct[b] / bin_counts[b] as f64;
            ece += (avg_conf - avg_acc).abs() * bin_counts[b] as f64 / n as f64;
        }
    }
    ece
}

/// Brier score for probability calibration
pub fn brier_score(probs: &[f64], labels: &[f64]) -> f64 {
    probs.iter().zip(labels.iter())
        .map(|(&p, &l)| (p - l) * (p - l))
        .sum::<f64>() / probs.len() as f64
}

/// Log loss
pub fn log_loss(probs: &[f64], labels: &[f64]) -> f64 {
    let eps = 1e-15;
    -probs.iter().zip(labels.iter())
        .map(|(&p, &l)| {
            let p = p.max(eps).min(1.0 - eps);
            l * p.ln() + (1.0 - l) * (1.0 - p).ln()
        })
        .sum::<f64>() / probs.len() as f64
}

/// Reliability diagram data
pub fn reliability_diagram(probs: &[f64], labels: &[f64], num_bins: usize) -> (Vec<f64>, Vec<f64>, Vec<usize>) {
    let mut bin_sums = vec![0.0; num_bins];
    let mut bin_correct = vec![0.0; num_bins];
    let mut bin_counts = vec![0usize; num_bins];

    for (&p, &l) in probs.iter().zip(labels.iter()) {
        let bin = ((p * num_bins as f64) as usize).min(num_bins - 1);
        bin_sums[bin] += p;
        bin_correct[bin] += l;
        bin_counts[bin] += 1;
    }

    let mean_pred: Vec<f64> = (0..num_bins).map(|b| {
        if bin_counts[b] > 0 { bin_sums[b] / bin_counts[b] as f64 } else { (b as f64 + 0.5) / num_bins as f64 }
    }).collect();

    let mean_true: Vec<f64> = (0..num_bins).map(|b| {
        if bin_counts[b] > 0 { bin_correct[b] / bin_counts[b] as f64 } else { 0.0 }
    }).collect();

    (mean_pred, mean_true, bin_counts)
}

/// Conformal prediction wrapper
#[derive(Clone, Debug)]
pub struct ConformalPredictor {
    pub calibration_scores: Vec<f64>, // sorted nonconformity scores
}

impl ConformalPredictor {
    pub fn new(mut scores: Vec<f64>) -> Self {
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Self { calibration_scores: scores }
    }

    pub fn prediction_interval(&self, point_pred: f64, confidence: f64) -> (f64, f64) {
        let n = self.calibration_scores.len();
        let idx = ((confidence * (n + 1) as f64).ceil() as usize).min(n) - 1;
        let q = self.calibration_scores[idx.min(n - 1)];
        (point_pred - q, point_pred + q)
    }
}

/// Diversity measures for ensemble
pub fn disagreement_measure(preds: &[Vec<f64>]) -> f64 {
    let n_models = preds.len();
    let n_samples = preds[0].len();
    if n_models < 2 { return 0.0; }
    let mut disagreement = 0.0;
    for i in 0..n_models {
        for j in (i + 1)..n_models {
            let d: f64 = preds[i].iter().zip(preds[j].iter())
                .map(|(&a, &b)| if (a.round() as i64) != (b.round() as i64) { 1.0 } else { 0.0 })
                .sum::<f64>() / n_samples as f64;
            disagreement += d;
        }
    }
    let pairs = (n_models * (n_models - 1)) as f64 / 2.0;
    disagreement / pairs
}

pub fn correlation_diversity(preds: &[Vec<f64>]) -> f64 {
    let n = preds.len();
    let m = preds[0].len();
    if n < 2 || m < 2 { return 0.0; }
    let means: Vec<f64> = preds.iter().map(|p| p.iter().sum::<f64>() / m as f64).collect();
    let mut avg_corr = 0.0;
    let mut count = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            let cov: f64 = preds[i].iter().zip(preds[j].iter())
                .map(|(&a, &b)| (a - means[i]) * (b - means[j]))
                .sum::<f64>() / m as f64;
            let std_i = (preds[i].iter().map(|&a| (a - means[i]).powi(2)).sum::<f64>() / m as f64).sqrt();
            let std_j = (preds[j].iter().map(|&b| (b - means[j]).powi(2)).sum::<f64>() / m as f64).sqrt();
            if std_i > 1e-12 && std_j > 1e-12 {
                avg_corr += cov / (std_i * std_j);
            }
            count += 1;
        }
    }
    if count > 0 { avg_corr / count as f64 } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_ensemble() {
        let ens = Ensemble::uniform(3);
        let preds = vec![1.0, 2.0, 3.0];
        assert!((ens.combine(&preds) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_ensemble() {
        let ens = Ensemble::weighted(vec![1.0, 2.0, 1.0]);
        let preds = vec![1.0, 2.0, 3.0];
        let expected = (1.0 * 0.25 + 2.0 * 0.5 + 3.0 * 0.25);
        assert!((ens.combine(&preds) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_online_weights() {
        let mut updater = OnlineWeightUpdater::new(3, 0.5, OnlineMethod::ExponentialWeighting);
        updater.update(&[0.1, 0.5, 0.3]);
        assert!(updater.weights[0] > updater.weights[1]); // lower loss -> higher weight
    }

    #[test]
    fn test_platt_scaling() {
        let scores = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0];
        let ps = PlattScaling::fit(&scores, &labels, 100, 0.01);
        let cal = ps.calibrate(2.0);
        assert!(cal > 0.5 && cal <= 1.0);
        let cal_neg = ps.calibrate(-2.0);
        assert!(cal_neg < 0.5);
    }

    #[test]
    fn test_isotonic() {
        let scores = vec![0.1, 0.2, 0.3, 0.6, 0.8];
        let labels = vec![0.0, 0.0, 1.0, 1.0, 1.0];
        let iso = IsotonicCalibration::fit(&scores, &labels);
        let cal = iso.calibrate(0.7);
        assert!(cal >= 0.0 && cal <= 1.0);
    }

    #[test]
    fn test_aic_bic() {
        let ll = log_likelihood_gaussian(1.0, 100);
        let a = aic(ll, 5);
        let b = bic(ll, 5, 100);
        assert!(a.is_finite());
        assert!(b > a); // BIC penalizes more for n>7
    }

    #[test]
    fn test_ece() {
        let probs = vec![0.9, 0.8, 0.7, 0.3, 0.2];
        let labels = vec![1.0, 1.0, 1.0, 0.0, 0.0];
        let ece = expected_calibration_error(&probs, &labels, 10);
        assert!(ece >= 0.0 && ece <= 1.0);
    }

    #[test]
    fn test_brier() {
        let probs = vec![0.9, 0.1];
        let labels = vec![1.0, 0.0];
        let bs = brier_score(&probs, &labels);
        assert!(bs < 0.02); // well calibrated
    }

    #[test]
    fn test_conformal() {
        let scores = vec![0.1, 0.2, 0.3, 0.5, 0.8, 1.0];
        let cp = ConformalPredictor::new(scores);
        let (lo, hi) = cp.prediction_interval(5.0, 0.9);
        assert!(lo < 5.0);
        assert!(hi > 5.0);
    }
}
