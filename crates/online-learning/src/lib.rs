//! online-learning crate
//!
//! Provides three online / streaming learning algorithms used by the IAE pipeline
//! for fast training on historical trade data:
//!
//!   - FTRL-Proximal  (src/ftrl.rs)   -- Follow The Regularized Leader with L1/L2
//!   - Passive-Aggressive II (src/passive_aggressive.rs) -- PA-II regression
//!   - Mini-batch SGD (src/sgd.rs)    -- with momentum and Adam variants
//!
//! All models implement the [`OnlineLearner`] trait which requires:
//!   - `update(&mut self, features: &[f64], label: f64)` -- single sample update
//!   - `predict(&self, features: &[f64]) -> f64`         -- linear prediction
//!   - `weights(&self) -> &[f64]`                        -- current weight vector
//!
//! Models are serializable to/from JSON via `ModelState` for checkpoint/restore.

pub mod ftrl;
pub mod passive_aggressive;
pub mod sgd;
pub mod hedge_algorithm;
pub mod adaptive_learning_rate;
pub mod bandit_explorer;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Shared trait
// ---------------------------------------------------------------------------

/// Common interface for all online learning models.
pub trait OnlineLearner {
    /// Process one (features, label) sample and update internal state.
    fn update(&mut self, features: &[f64], label: f64);

    /// Predict label for features using current weights.
    fn predict(&self, features: &[f64]) -> f64;

    /// Return current weight vector (read-only).
    fn weights(&self) -> &[f64];

    /// Return bias term (intercept).
    fn bias(&self) -> f64;

    /// Reset model to initial state (zero weights).
    fn reset(&mut self);

    /// Number of samples seen so far.
    fn n_seen(&self) -> u64;

    /// Name of the algorithm for logging/serialisation.
    fn algorithm_name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Data loading
// ---------------------------------------------------------------------------

/// A single labelled sample: dense feature vector + scalar label.
#[derive(Debug, Clone)]
pub struct Sample {
    pub features: Vec<f64>,
    pub label: f64,
}

/// Load samples from a CSV file.
///
/// The CSV must have a header row. The last column is treated as the label;
/// all other columns are features. Missing values are replaced with 0.0.
pub fn load_csv(path: &str) -> Result<Vec<Sample>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    let n_cols = headers.len();
    if n_cols < 2 {
        anyhow::bail!("CSV must have at least 2 columns (features + label)");
    }

    let mut samples = Vec::new();
    for result in rdr.records() {
        let record = result?;
        let mut values: Vec<f64> = record
            .iter()
            .map(|s| s.parse::<f64>().unwrap_or(0.0))
            .collect();

        if values.len() < n_cols {
            values.resize(n_cols, 0.0);
        }

        let label = values[n_cols - 1];
        let features = values[..n_cols - 1].to_vec();
        samples.push(Sample { features, label });
    }
    Ok(samples)
}

// ---------------------------------------------------------------------------
// Model serialisation
// ---------------------------------------------------------------------------

/// Serialisable snapshot of any online model. Written to JSON by the CLI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelState {
    /// Algorithm identifier: "ftrl", "passive_aggressive", "sgd"
    pub algorithm: String,

    /// Weight vector (dense, same ordering as training features).
    pub weights: Vec<f64>,

    /// Bias / intercept term.
    pub bias: f64,

    /// Number of training samples processed.
    pub n_seen: u64,

    /// Hyperparameters used during training.
    pub hyperparams: HashMap<String, f64>,

    /// Training metrics collected during fitting.
    pub metrics: TrainingMetrics,

    /// UTC timestamp of when the model was saved.
    pub saved_at: String,
}

impl ModelState {
    pub fn save_to_file(&self, path: &str) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    pub fn load_from_file(path: &str) -> Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let state: ModelState = serde_json::from_str(&json)?;
        Ok(state)
    }
}

// ---------------------------------------------------------------------------
// Training metrics
// ---------------------------------------------------------------------------

/// Metrics accumulated during an online training pass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub n_updates: u64,
    pub final_loss: f64,
    pub mean_absolute_error: f64,
    pub root_mean_squared_error: f64,
    pub loss_history: Vec<f64>,   // one entry per epoch or per N samples
}

impl TrainingMetrics {
    pub fn compute_from_errors(errors: &[f64]) -> Self {
        if errors.is_empty() {
            return Self::default();
        }
        let n = errors.len() as f64;
        let mae = errors.iter().map(|e| e.abs()).sum::<f64>() / n;
        let mse = errors.iter().map(|e| e * e).sum::<f64>() / n;
        let rmse = mse.sqrt();
        let final_loss = *errors.last().unwrap_or(&0.0);
        Self {
            n_updates: errors.len() as u64,
            final_loss: final_loss * final_loss,
            mean_absolute_error: mae,
            root_mean_squared_error: rmse,
            loss_history: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Training loop helper
// ---------------------------------------------------------------------------

/// Train any `OnlineLearner` on a slice of samples, collecting per-sample errors.
/// Returns vector of (label, prediction, error) tuples.
pub fn train_online<L: OnlineLearner>(
    learner: &mut L,
    samples: &[Sample],
    collect_every: usize,
) -> (Vec<f64>, TrainingMetrics) {
    let mut errors: Vec<f64> = Vec::with_capacity(samples.len());
    let mut loss_history: Vec<f64> = Vec::new();
    let mut running_loss = 0.0_f64;

    for (i, sample) in samples.iter().enumerate() {
        let pred = learner.predict(&sample.features);
        let err = sample.label - pred;
        errors.push(err);
        running_loss += err * err;

        learner.update(&sample.features, sample.label);

        if collect_every > 0 && (i + 1) % collect_every == 0 {
            loss_history.push(running_loss / collect_every as f64);
            running_loss = 0.0;
        }
    }

    let mut metrics = TrainingMetrics::compute_from_errors(&errors);
    metrics.loss_history = loss_history;
    metrics.n_updates = samples.len() as u64;
    (errors, metrics)
}

/// Evaluate a trained `OnlineLearner` on held-out samples without updating.
pub fn evaluate<L: OnlineLearner>(learner: &L, samples: &[Sample]) -> TrainingMetrics {
    let errors: Vec<f64> = samples
        .iter()
        .map(|s| s.label - learner.predict(&s.features))
        .collect();
    TrainingMetrics::compute_from_errors(&errors)
}

// ---------------------------------------------------------------------------
// Normalisation helper (z-score per feature)
// ---------------------------------------------------------------------------

/// Z-score normalise a dataset in-place. Returns (means, stds) per feature.
pub fn normalize_samples(samples: &mut [Sample]) -> (Vec<f64>, Vec<f64>) {
    if samples.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let n_features = samples[0].features.len();
    let n = samples.len() as f64;

    let mut means = vec![0.0_f64; n_features];
    let mut stds = vec![1.0_f64; n_features];

    for sample in samples.iter() {
        for (j, &x) in sample.features.iter().enumerate() {
            if j < n_features {
                means[j] += x / n;
            }
        }
    }

    for sample in samples.iter() {
        for (j, &x) in sample.features.iter().enumerate() {
            if j < n_features {
                stds[j] += (x - means[j]).powi(2) / n;
            }
        }
    }
    for s in stds.iter_mut() {
        *s = s.sqrt().max(1e-8);
    }

    for sample in samples.iter_mut() {
        for (j, x) in sample.features.iter_mut().enumerate() {
            if j < n_features {
                *x = (*x - means[j]) / stds[j];
            }
        }
    }

    (means, stds)
}

// ---------------------------------------------------------------------------
// Dot product helper
// ---------------------------------------------------------------------------

/// Dense dot product, zero-extending shorter slice.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    let mut sum = 0.0_f64;
    for i in 0..len {
        sum += a[i] * b[i];
    }
    sum
}

/// Ensure `weights` has at least `n` entries (zero-extended).
#[inline]
pub fn ensure_capacity(weights: &mut Vec<f64>, n: usize) {
    if weights.len() < n {
        weights.resize(n, 0.0);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_samples(n: usize) -> Vec<Sample> {
        // y = 2*x0 + 3*x1 + noise
        (0..n)
            .map(|i| {
                let x0 = i as f64 * 0.1;
                let x1 = (i as f64 * 0.07).sin();
                Sample {
                    features: vec![x0, x1],
                    label: 2.0 * x0 + 3.0 * x1,
                }
            })
            .collect()
    }

    #[test]
    fn test_dot() {
        assert!((dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_dot_unequal_lengths() {
        assert!((dot(&[1.0, 2.0], &[3.0]) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_load_csv_roundtrip() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.csv");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "x0,x1,y").unwrap();
        writeln!(f, "1.0,2.0,3.0").unwrap();
        writeln!(f, "4.0,5.0,6.0").unwrap();

        let samples = load_csv(path.to_str().unwrap()).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].features, vec![1.0, 2.0]);
        assert_eq!(samples[0].label, 3.0);
    }

    #[test]
    fn test_normalize_zero_mean() {
        let mut samples = make_samples(100);
        let (means, _stds) = normalize_samples(&mut samples);
        // After normalisation, mean of feature 0 should be ~0
        let mean0: f64 = samples.iter().map(|s| s.features[0]).sum::<f64>() / 100.0;
        assert!(mean0.abs() < 1e-6, "mean0 = {}", mean0);
        assert_eq!(means.len(), 2);
    }

    #[test]
    fn test_training_metrics() {
        let errors = vec![1.0, -1.0, 2.0, -2.0];
        let m = TrainingMetrics::compute_from_errors(&errors);
        assert!((m.mean_absolute_error - 1.5).abs() < 1e-10);
        assert!((m.root_mean_squared_error - (2.5_f64).sqrt()).abs() < 1e-8);
    }

    #[test]
    fn test_model_state_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.json");
        let state = ModelState {
            algorithm: "ftrl".to_string(),
            weights: vec![1.0, 2.0, 3.0],
            bias: 0.5,
            n_seen: 1000,
            hyperparams: [("alpha".to_string(), 0.1)].into(),
            metrics: TrainingMetrics::default(),
            saved_at: "2026-01-01T00:00:00Z".to_string(),
        };
        state.save_to_file(path.to_str().unwrap()).unwrap();
        let loaded = ModelState::load_from_file(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.algorithm, "ftrl");
        assert_eq!(loaded.weights, vec![1.0, 2.0, 3.0]);
        assert_eq!(loaded.n_seen, 1000);
    }
}
