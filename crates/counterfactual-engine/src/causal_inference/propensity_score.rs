//! Propensity score matching for causal effect estimation in trading.
//!
//! Use case: "What would have happened if we hadn't blocked this trade due to
//! the event calendar filter?"
//!
//! # Method
//!
//! 1. Fit a logistic regression model to estimate P(treatment | features).
//! 2. Match treated and control units by propensity score within a caliper.
//! 3. Compute Average Treatment Effect (ATE) and ATT from matched pairs.
//!
//! # Feature vector convention
//!
//! Each `FeatureVec` holds five named trading features:
//!   - `bh_mass`        -- Black Hole mass at trade time
//!   - `hurst_h`        -- Hurst exponent (rolling 128-bar)
//!   - `vol_percentile` -- realised volatility percentile [0, 1]
//!   - `time_of_day`    -- fractional hour in [0, 24)
//!   - `day_of_week`    -- 0 = Monday, 4 = Friday

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Feature vector
// ---------------------------------------------------------------------------

/// Five-dimensional feature vector for a single trade observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureVec {
    /// Black Hole mass (continuous, typically in [0, 10]).
    pub bh_mass: f64,
    /// Hurst exponent [0, 1].
    pub hurst_h: f64,
    /// Realised-volatility percentile [0, 1].
    pub vol_percentile: f64,
    /// Fractional hour of day [0, 24).
    pub time_of_day: f64,
    /// Day of week: 0 = Mon, 4 = Fri.
    pub day_of_week: f64,
}

impl FeatureVec {
    pub fn new(
        bh_mass: f64,
        hurst_h: f64,
        vol_percentile: f64,
        time_of_day: f64,
        day_of_week: f64,
    ) -> Self {
        Self { bh_mass, hurst_h, vol_percentile, time_of_day, day_of_week }
    }

    /// Return features as a fixed-size array in canonical order.
    pub fn as_array(&self) -> [f64; 5] {
        [
            self.bh_mass,
            self.hurst_h,
            self.vol_percentile,
            self.time_of_day / 24.0, // normalise to [0,1]
            self.day_of_week / 4.0,  // normalise to [0,1]
        ]
    }

    /// Euclidean dot product with a weight vector (length 5).
    pub fn dot(&self, weights: &[f64; 5]) -> f64 {
        let arr = self.as_array();
        arr.iter().zip(weights.iter()).map(|(x, w)| x * w).sum()
    }
}

// ---------------------------------------------------------------------------
// Logistic regression (binary outcome, L2 regularisation, gradient descent)
// ---------------------------------------------------------------------------

/// Binary logistic regression trained by mini-batch gradient descent.
///
/// Model: P(y=1 | x) = sigmoid(x . w + b)
/// Loss:  cross-entropy + (lambda/2) * ||w||^2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticRegression {
    /// Learned weights (one per feature).
    pub weights: [f64; 5],
    /// Bias term.
    pub bias: f64,
    /// L2 regularisation strength.
    pub lambda: f64,
    /// Learning rate for gradient descent.
    pub lr: f64,
    /// Number of training epochs.
    pub epochs: usize,
}

impl LogisticRegression {
    /// Create a new logistic regression model with given hyperparameters.
    pub fn new(lambda: f64, lr: f64, epochs: usize) -> Self {
        Self {
            weights: [0.0; 5],
            bias: 0.0,
            lambda,
            lr,
            epochs,
        }
    }

    /// Create with sensible defaults for propensity scoring.
    pub fn default_propensity() -> Self {
        Self::new(0.01, 0.05, 500)
    }

    fn sigmoid(z: f64) -> f64 {
        1.0 / (1.0 + (-z).exp())
    }

    /// Predict P(y=1 | x) for a single feature vector.
    pub fn predict_proba(&self, x: &FeatureVec) -> f64 {
        let arr = x.as_array();
        let z: f64 = arr.iter().zip(self.weights.iter()).map(|(xi, wi)| xi * wi).sum::<f64>()
            + self.bias;
        Self::sigmoid(z)
    }

    /// Fit the model on labelled data.
    ///
    /// `y = 1` for treated units, `y = 0` for control units.
    pub fn fit(&mut self, xs: &[FeatureVec], ys: &[f64]) {
        assert_eq!(xs.len(), ys.len(), "xs and ys must have equal length");
        let n = xs.len();
        if n == 0 {
            return;
        }

        for _epoch in 0..self.epochs {
            let mut grad_w = [0.0f64; 5];
            let mut grad_b = 0.0f64;

            for (x, &y) in xs.iter().zip(ys.iter()) {
                let arr = x.as_array();
                let z: f64 = arr.iter().zip(self.weights.iter()).map(|(xi, wi)| xi * wi).sum::<f64>()
                    + self.bias;
                let p = Self::sigmoid(z);
                let err = p - y; // d(loss)/d(z)
                for j in 0..5 {
                    grad_w[j] += err * arr[j];
                }
                grad_b += err;
            }

            // Average gradient + L2 regularisation
            for j in 0..5 {
                self.weights[j] -= self.lr * (grad_w[j] / n as f64 + self.lambda * self.weights[j]);
            }
            self.bias -= self.lr * (grad_b / n as f64);
        }
    }

    /// Compute binary cross-entropy loss on a labelled dataset.
    pub fn loss(&self, xs: &[FeatureVec], ys: &[f64]) -> f64 {
        let n = xs.len();
        if n == 0 {
            return 0.0;
        }
        let mut total = 0.0f64;
        for (x, &y) in xs.iter().zip(ys.iter()) {
            let p = self.predict_proba(x).clamp(1e-12, 1.0 - 1e-12);
            total += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
        }
        let reg: f64 = (self.lambda / 2.0) * self.weights.iter().map(|w| w * w).sum::<f64>();
        total / n as f64 + reg
    }

    /// Accuracy on a labelled dataset (threshold = 0.5).
    pub fn accuracy(&self, xs: &[FeatureVec], ys: &[f64]) -> f64 {
        let n = xs.len();
        if n == 0 {
            return 0.0;
        }
        let correct = xs.iter().zip(ys.iter())
            .filter(|(x, &y)| {
                let pred = if self.predict_proba(x) >= 0.5 { 1.0 } else { 0.0 };
                (pred - y).abs() < 1e-9
            })
            .count();
        correct as f64 / n as f64
    }
}

// ---------------------------------------------------------------------------
// PropensityScoreEstimator
// ---------------------------------------------------------------------------

/// Fits a propensity model and scores observations.
///
/// The propensity score e(x) = P(treatment = 1 | x) is used as a
/// one-dimensional summary that balances covariates for matching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropensityScoreEstimator {
    model: LogisticRegression,
    /// Set to true after calling `fit`.
    pub fitted: bool,
    /// Stored treated propensity scores after fitting.
    pub treated_scores: Vec<f64>,
    /// Stored control propensity scores after fitting.
    pub control_scores: Vec<f64>,
}

impl PropensityScoreEstimator {
    /// Create estimator with default logistic regression hyperparameters.
    pub fn new() -> Self {
        Self {
            model: LogisticRegression::default_propensity(),
            fitted: false,
            treated_scores: Vec::new(),
            control_scores: Vec::new(),
        }
    }

    /// Create estimator with explicit hyperparameters.
    pub fn with_hyperparams(lambda: f64, lr: f64, epochs: usize) -> Self {
        Self {
            model: LogisticRegression::new(lambda, lr, epochs),
            fitted: false,
            treated_scores: Vec::new(),
            control_scores: Vec::new(),
        }
    }

    /// Fit the propensity model on treated and control feature vectors.
    ///
    /// Treated units receive label y=1, control units y=0.
    pub fn fit(&mut self, treated: &[FeatureVec], control: &[FeatureVec]) {
        let mut xs: Vec<FeatureVec> = Vec::with_capacity(treated.len() + control.len());
        let mut ys: Vec<f64> = Vec::with_capacity(treated.len() + control.len());

        for fv in treated {
            xs.push(fv.clone());
            ys.push(1.0);
        }
        for fv in control {
            xs.push(fv.clone());
            ys.push(0.0);
        }

        self.model.fit(&xs, &ys);
        self.fitted = true;

        self.treated_scores = treated.iter().map(|x| self.model.predict_proba(x)).collect();
        self.control_scores = control.iter().map(|x| self.model.predict_proba(x)).collect();
    }

    /// Estimate P(treatment | x) for a new observation.
    ///
    /// Returns a value in (0, 1).
    pub fn estimate(&self, x: &FeatureVec) -> f64 {
        self.model.predict_proba(x)
    }

    /// Match treated units to control units by nearest-neighbour propensity score.
    ///
    /// Only pairs within `caliper` distance on the propensity score are kept.
    /// Returns a vec of `(treated_idx, control_idx)` pairs.
    /// Each control unit can be matched at most once (greedy 1:1 matching).
    pub fn match_units(
        &self,
        treated: &[FeatureVec],
        control: &[FeatureVec],
        caliper: f64,
    ) -> Vec<(usize, usize)> {
        let t_scores: Vec<f64> = treated.iter().map(|x| self.model.predict_proba(x)).collect();
        let c_scores: Vec<f64> = control.iter().map(|x| self.model.predict_proba(x)).collect();

        let mut matched: Vec<(usize, usize)> = Vec::new();
        let mut used_control: Vec<bool> = vec![false; control.len()];

        for (ti, &ts) in t_scores.iter().enumerate() {
            let mut best_dist = caliper;
            let mut best_ci: Option<usize> = None;

            for (ci, &cs) in c_scores.iter().enumerate() {
                if used_control[ci] {
                    continue;
                }
                let dist = (ts - cs).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_ci = Some(ci);
                }
            }

            if let Some(ci) = best_ci {
                matched.push((ti, ci));
                used_control[ci] = true;
            }
        }
        matched
    }

    /// Standardised mean difference (SMD) for a single feature dimension.
    ///
    /// SMD < 0.1 indicates good balance after matching.
    pub fn smd_after_matching(
        &self,
        treated: &[FeatureVec],
        control: &[FeatureVec],
        matched_pairs: &[(usize, usize)],
        feature_fn: impl Fn(&FeatureVec) -> f64,
    ) -> f64 {
        if matched_pairs.is_empty() {
            return f64::NAN;
        }
        let t_vals: Vec<f64> = matched_pairs.iter().map(|(ti, _)| feature_fn(&treated[*ti])).collect();
        let c_vals: Vec<f64> = matched_pairs.iter().map(|(_, ci)| feature_fn(&control[*ci])).collect();

        let t_mean = mean(&t_vals);
        let c_mean = mean(&c_vals);
        let t_var = variance(&t_vals, t_mean);
        let c_var = variance(&c_vals, c_mean);
        let pooled_sd = ((t_var + c_var) / 2.0).sqrt();

        if pooled_sd < 1e-12 {
            return 0.0;
        }
        (t_mean - c_mean).abs() / pooled_sd
    }

    /// Check covariate balance: returns a map from feature name to SMD.
    pub fn balance_report(
        &self,
        treated: &[FeatureVec],
        control: &[FeatureVec],
        matched_pairs: &[(usize, usize)],
    ) -> HashMap<String, f64> {
        let mut report = HashMap::new();
        let features: &[(&str, Box<dyn Fn(&FeatureVec) -> f64>)] = &[
            ("bh_mass",        Box::new(|f: &FeatureVec| f.bh_mass)),
            ("hurst_h",        Box::new(|f: &FeatureVec| f.hurst_h)),
            ("vol_percentile", Box::new(|f: &FeatureVec| f.vol_percentile)),
            ("time_of_day",    Box::new(|f: &FeatureVec| f.time_of_day)),
            ("day_of_week",    Box::new(|f: &FeatureVec| f.day_of_week)),
        ];
        for (name, func) in features {
            let smd = self.smd_after_matching(treated, control, matched_pairs, func.as_ref());
            report.insert(name.to_string(), smd);
        }
        report
    }
}

impl Default for PropensityScoreEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ATEEstimator
// ---------------------------------------------------------------------------

/// Estimates Average Treatment Effect (ATE) and ATT from matched pairs.
///
/// Given matched pairs `(treated_idx, control_idx)` and outcome vectors,
/// the ATE estimator computes the average pairwise difference in outcomes.
///
/// # Notation
///
/// - Y(1) = outcome under treatment
/// - Y(0) = outcome under control
/// - ATE  = E[Y(1) - Y(0)] over the full population
/// - ATT  = E[Y(1) - Y(0) | treated]
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ATEEstimator;

impl ATEEstimator {
    pub fn new() -> Self {
        Self
    }

    /// Compute ATE and its standard error from matched outcome pairs.
    ///
    /// # Arguments
    ///
    /// - `matched_pairs`      -- vec of `(treated_idx, control_idx)`
    /// - `treated_outcomes`   -- outcome for each treated observation
    /// - `control_outcomes`   -- outcome for each control observation
    ///
    /// # Returns
    ///
    /// `(ate, std_error)` where std_error is the standard error of the
    /// mean pairwise difference.
    pub fn compute_ate(
        &self,
        matched_pairs: &[(usize, usize)],
        treated_outcomes: &[f64],
        control_outcomes: &[f64],
    ) -> (f64, f64) {
        if matched_pairs.is_empty() {
            return (0.0, 0.0);
        }

        let diffs: Vec<f64> = matched_pairs
            .iter()
            .filter_map(|(ti, ci)| {
                let y1 = treated_outcomes.get(*ti)?;
                let y0 = control_outcomes.get(*ci)?;
                Some(y1 - y0)
            })
            .collect();

        if diffs.is_empty() {
            return (0.0, 0.0);
        }

        let ate = mean(&diffs);
        let se = std_error(&diffs, ate);
        (ate, se)
    }

    /// Compute ATT: average treatment effect on the treated.
    ///
    /// Uses the same matched pair structure but focuses on matched treated units.
    /// `outcomes` is indexed jointly: `outcomes[0..n_treated]` are treated,
    /// `outcomes[n_treated..]` are control.  Pass them separately for clarity.
    pub fn compute_att(
        &self,
        matched_pairs: &[(usize, usize)],
        treated_outcomes: &[f64],
        control_outcomes: &[f64],
    ) -> f64 {
        // ATT is E[Y(1) - Y(0) | T=1], estimated as the mean pairwise diff
        // over matched treated units (same computation as ATE here since each
        // treated unit appears at most once in the greedy 1:1 matching).
        let (att, _) = self.compute_ate(matched_pairs, treated_outcomes, control_outcomes);
        att
    }

    /// Compute weighted ATE where each pair is weighted by a supplied vector.
    pub fn compute_weighted_ate(
        &self,
        matched_pairs: &[(usize, usize)],
        treated_outcomes: &[f64],
        control_outcomes: &[f64],
        weights: &[f64],
    ) -> f64 {
        if matched_pairs.is_empty() {
            return 0.0;
        }
        let mut weighted_sum = 0.0f64;
        let mut weight_total = 0.0f64;
        for (idx, (ti, ci)) in matched_pairs.iter().enumerate() {
            let y1 = match treated_outcomes.get(*ti) {
                Some(&v) => v,
                None => continue,
            };
            let y0 = match control_outcomes.get(*ci) {
                Some(&v) => v,
                None => continue,
            };
            let w = weights.get(idx).copied().unwrap_or(1.0);
            weighted_sum += w * (y1 - y0);
            weight_total += w;
        }
        if weight_total < 1e-14 {
            return 0.0;
        }
        weighted_sum / weight_total
    }

    /// Bootstrap confidence interval for ATE at a given significance level.
    ///
    /// Returns `(lower, upper)` at `1 - alpha` confidence level.
    pub fn bootstrap_ci(
        &self,
        matched_pairs: &[(usize, usize)],
        treated_outcomes: &[f64],
        control_outcomes: &[f64],
        n_bootstrap: usize,
        alpha: f64,
        seed: u64,
    ) -> (f64, f64) {
        let m = matched_pairs.len();
        if m == 0 {
            return (0.0, 0.0);
        }

        let diffs: Vec<f64> = matched_pairs
            .iter()
            .filter_map(|(ti, ci)| {
                let y1 = treated_outcomes.get(*ti)?;
                let y0 = control_outcomes.get(*ci)?;
                Some(y1 - y0)
            })
            .collect();

        if diffs.is_empty() {
            return (0.0, 0.0);
        }

        let mut boot_ates: Vec<f64> = Vec::with_capacity(n_bootstrap);
        let mut state = seed;

        for _ in 0..n_bootstrap {
            let mut boot_sum = 0.0f64;
            for _ in 0..m {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let idx = (state >> 33) as usize % diffs.len();
                boot_sum += diffs[idx];
            }
            boot_ates.push(boot_sum / m as f64);
        }

        boot_ates.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lo_idx = ((alpha / 2.0) * n_bootstrap as f64) as usize;
        let hi_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;
        (
            boot_ates[lo_idx.min(boot_ates.len() - 1)],
            boot_ates[hi_idx.min(boot_ates.len() - 1)],
        )
    }

    /// Null hypothesis test: is ATE significantly different from zero?
    ///
    /// Returns the t-statistic and a flag `significant` when |t| > 1.96.
    pub fn t_test(&self, ate: f64, se: f64) -> (f64, bool) {
        if se < 1e-14 {
            return (0.0, false);
        }
        let t = ate / se;
        (t, t.abs() > 1.96)
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn variance(xs: &[f64], m: f64) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    xs.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (xs.len() - 1) as f64
}

fn std_error(xs: &[f64], m: f64) -> f64 {
    let var = variance(xs, m);
    (var / xs.len() as f64).sqrt()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_treated_features() -> Vec<FeatureVec> {
        // Treated: high bh_mass, high hurst (trend-following trades that were blocked)
        vec![
            FeatureVec::new(3.5, 0.72, 0.80, 10.0, 1.0),
            FeatureVec::new(3.8, 0.68, 0.75, 11.0, 2.0),
            FeatureVec::new(4.0, 0.70, 0.85, 14.0, 3.0),
            FeatureVec::new(3.6, 0.65, 0.78, 9.5, 1.0),
            FeatureVec::new(4.2, 0.73, 0.82, 13.0, 2.0),
            FeatureVec::new(3.9, 0.69, 0.76, 10.5, 0.0),
            FeatureVec::new(3.7, 0.71, 0.79, 12.0, 4.0),
            FeatureVec::new(4.1, 0.67, 0.81, 15.0, 3.0),
        ]
    }

    fn make_control_features() -> Vec<FeatureVec> {
        // Control: lower bh_mass, lower hurst (trades that were allowed through)
        vec![
            FeatureVec::new(1.5, 0.48, 0.40, 9.0, 1.0),
            FeatureVec::new(1.8, 0.52, 0.45, 10.0, 2.0),
            FeatureVec::new(2.0, 0.45, 0.50, 11.0, 3.0),
            FeatureVec::new(1.6, 0.50, 0.42, 8.5, 1.0),
            FeatureVec::new(2.2, 0.55, 0.48, 12.0, 0.0),
            FeatureVec::new(1.9, 0.47, 0.44, 9.5, 4.0),
            FeatureVec::new(2.1, 0.51, 0.46, 13.0, 2.0),
            FeatureVec::new(1.7, 0.49, 0.41, 14.0, 3.0),
        ]
    }

    #[test]
    fn test_feature_vec_normalisation() {
        let fv = FeatureVec::new(2.0, 0.6, 0.5, 12.0, 2.0);
        let arr = fv.as_array();
        // time_of_day should be normalised by /24
        assert!((arr[3] - 12.0 / 24.0).abs() < 1e-9);
        // day_of_week should be normalised by /4
        assert!((arr[4] - 2.0 / 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_logistic_regression_learns() {
        let treated = make_treated_features();
        let control = make_control_features();

        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for fv in &treated { xs.push(fv.clone()); ys.push(1.0); }
        for fv in &control { xs.push(fv.clone()); ys.push(0.0); }

        let mut lr = LogisticRegression::new(0.01, 0.05, 1000);
        let loss_before = lr.loss(&xs, &ys);
        lr.fit(&xs, &ys);
        let loss_after = lr.loss(&xs, &ys);

        assert!(loss_after < loss_before, "loss should decrease: {loss_before:.4} -> {loss_after:.4}");
    }

    #[test]
    fn test_logistic_regression_accuracy_above_chance() {
        let treated = make_treated_features();
        let control = make_control_features();

        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for fv in &treated { xs.push(fv.clone()); ys.push(1.0); }
        for fv in &control { xs.push(fv.clone()); ys.push(0.0); }

        let mut lr = LogisticRegression::new(0.01, 0.05, 2000);
        lr.fit(&xs, &ys);
        let acc = lr.accuracy(&xs, &ys);
        assert!(acc > 0.5, "accuracy {acc:.3} should be above chance (0.5)");
    }

    #[test]
    fn test_propensity_estimator_fit() {
        let treated = make_treated_features();
        let control = make_control_features();

        let mut est = PropensityScoreEstimator::new();
        est.fit(&treated, &control);
        assert!(est.fitted);
        assert_eq!(est.treated_scores.len(), treated.len());
        assert_eq!(est.control_scores.len(), control.len());
    }

    #[test]
    fn test_propensity_scores_in_unit_interval() {
        let treated = make_treated_features();
        let control = make_control_features();

        let mut est = PropensityScoreEstimator::new();
        est.fit(&treated, &control);

        for &s in &est.treated_scores {
            assert!(s >= 0.0 && s <= 1.0, "treated score {s} out of [0,1]");
        }
        for &s in &est.control_scores {
            assert!(s >= 0.0 && s <= 1.0, "control score {s} out of [0,1]");
        }
    }

    #[test]
    fn test_match_units_respects_caliper() {
        let treated = make_treated_features();
        let control = make_control_features();

        let mut est = PropensityScoreEstimator::new();
        est.fit(&treated, &control);

        let pairs = est.match_units(&treated, &control, 0.05);
        // Verify caliper constraint
        for (ti, ci) in &pairs {
            let ts = est.estimate(&treated[*ti]);
            let cs = est.estimate(&control[*ci]);
            assert!(
                (ts - cs).abs() <= 0.05 + 1e-9,
                "pair ({ti},{ci}) score diff {:.4} exceeds caliper 0.05",
                (ts - cs).abs()
            );
        }
    }

    #[test]
    fn test_match_units_no_duplicate_controls() {
        let treated = make_treated_features();
        let control = make_control_features();

        let mut est = PropensityScoreEstimator::new();
        est.fit(&treated, &control);

        let pairs = est.match_units(&treated, &control, 0.5); // wide caliper
        let mut seen_ci = std::collections::HashSet::new();
        for (_, ci) in &pairs {
            assert!(seen_ci.insert(*ci), "control index {ci} matched more than once");
        }
    }

    #[test]
    fn test_ate_direction_positive() {
        // Treated has higher outcomes than control -> ATE should be positive
        let matched_pairs = vec![(0, 0), (1, 1), (2, 2), (3, 3)];
        let treated_outcomes = vec![0.05, 0.07, 0.06, 0.08];
        let control_outcomes = vec![0.01, 0.02, 0.01, 0.03];

        let est = ATEEstimator::new();
        let (ate, se) = est.compute_ate(&matched_pairs, &treated_outcomes, &control_outcomes);
        assert!(ate > 0.0, "ATE should be positive when treated outcomes are higher, got {ate:.4}");
        assert!(se >= 0.0, "std error should be non-negative");
    }

    #[test]
    fn test_ate_zero_for_equal_outcomes() {
        let matched_pairs = vec![(0, 0), (1, 1)];
        let outcomes = vec![0.05, 0.07];

        let est = ATEEstimator::new();
        let (ate, _) = est.compute_ate(&matched_pairs, &outcomes, &outcomes);
        assert!(ate.abs() < 1e-9, "ATE should be 0 when outcomes are identical, got {ate}");
    }

    #[test]
    fn test_att_matches_ate_for_1_to_1_matching() {
        let matched_pairs = vec![(0, 0), (1, 1), (2, 2)];
        let treated_outcomes = vec![0.10, 0.12, 0.09];
        let control_outcomes = vec![0.05, 0.04, 0.06];

        let est = ATEEstimator::new();
        let (ate, _) = est.compute_ate(&matched_pairs, &treated_outcomes, &control_outcomes);
        let att = est.compute_att(&matched_pairs, &treated_outcomes, &control_outcomes);
        assert!((ate - att).abs() < 1e-9, "ATE and ATT should be equal for 1:1 matching");
    }

    #[test]
    fn test_t_test_significant_large_ate() {
        let est = ATEEstimator::new();
        let (t, sig) = est.t_test(0.05, 0.01); // t = 5.0
        assert!(sig, "t={t:.2} should be significant");
    }

    #[test]
    fn test_t_test_not_significant_small_ate() {
        let est = ATEEstimator::new();
        let (t, sig) = est.t_test(0.01, 0.02); // t = 0.5
        assert!(!sig, "t={t:.2} should not be significant");
    }

    #[test]
    fn test_bootstrap_ci_contains_ate() {
        let matched_pairs = vec![(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)];
        let treated_outcomes = vec![0.08, 0.09, 0.07, 0.10, 0.06];
        let control_outcomes = vec![0.02, 0.03, 0.01, 0.04, 0.02];

        let est = ATEEstimator::new();
        let (ate, _) = est.compute_ate(&matched_pairs, &treated_outcomes, &control_outcomes);
        let (lo, hi) = est.bootstrap_ci(&matched_pairs, &treated_outcomes, &control_outcomes, 500, 0.05, 42);

        assert!(lo <= ate && ate <= hi, "ATE {ate:.4} should be inside CI [{lo:.4}, {hi:.4}]");
    }

    #[test]
    fn test_balance_report_keys() {
        let treated = make_treated_features();
        let control = make_control_features();

        let mut est = PropensityScoreEstimator::new();
        est.fit(&treated, &control);
        let pairs = est.match_units(&treated, &control, 0.5);
        let report = est.balance_report(&treated, &control, &pairs);

        assert!(report.contains_key("bh_mass"));
        assert!(report.contains_key("hurst_h"));
        assert!(report.contains_key("vol_percentile"));
        assert!(report.contains_key("time_of_day"));
        assert!(report.contains_key("day_of_week"));
    }
}
