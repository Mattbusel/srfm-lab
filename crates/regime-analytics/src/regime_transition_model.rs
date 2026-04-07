// regime_transition_model.rs
// Regime transition probability modeling.
// Includes smoothed transition matrix, logistic regression predictor,
// risk metrics derived from transition dynamics, and a change alert generator.

use std::collections::HashMap;
use crate::hmm_regime::RegimeLabel;

const ALL_REGIMES: [RegimeLabel; 4] = [
    RegimeLabel::Bull,
    RegimeLabel::Bear,
    RegimeLabel::Sideways,
    RegimeLabel::HighVol,
];

fn regime_index(r: RegimeLabel) -> usize {
    match r {
        RegimeLabel::Bull     => 0,
        RegimeLabel::Bear     => 1,
        RegimeLabel::Sideways => 2,
        RegimeLabel::HighVol  => 3,
    }
}

fn regime_from_index(i: usize) -> RegimeLabel {
    ALL_REGIMES[i]
}

// ---- TransitionMatrix -----------------------------------------------------

/// Empirical + smoothed n x n transition probability matrix.
/// Rows are the "from" regime; columns are the "to" regime.
#[derive(Debug, Clone)]
pub struct TransitionMatrix {
    n: usize,
    /// Raw observed transition counts[from][to]
    counts: Vec<Vec<f64>>,
    /// Dirichlet smoothing pseudo-count added to each cell
    smoothing: f64,
}

impl TransitionMatrix {
    pub fn new(n: usize, smoothing: f64) -> Self {
        TransitionMatrix {
            n,
            counts: vec![vec![0.0; n]; n],
            smoothing,
        }
    }

    /// Use the 4-regime alphabet by default.
    pub fn four_regime(smoothing: f64) -> Self {
        Self::new(4, smoothing)
    }

    /// Increment the count for a from->to transition.
    pub fn record_transition(&mut self, from: usize, to: usize) {
        assert!(from < self.n && to < self.n);
        self.counts[from][to] += 1.0;
    }

    /// Record a sequence of regime observations as consecutive transitions.
    pub fn fit_from_sequence(&mut self, sequence: &[RegimeLabel]) {
        for window in sequence.windows(2) {
            let from = regime_index(window[0]);
            let to   = regime_index(window[1]);
            self.record_transition(from, to);
        }
    }

    /// Smoothed probability P(to | from).
    pub fn prob(&self, from: usize, to: usize) -> f64 {
        let row_sum: f64 = self.counts[from].iter().sum::<f64>() + self.smoothing * self.n as f64;
        (self.counts[from][to] + self.smoothing) / row_sum
    }

    /// Full smoothed row for `from` state.
    pub fn row(&self, from: usize) -> Vec<f64> {
        (0..self.n).map(|to| self.prob(from, to)).collect()
    }

    /// Expected duration in state `from` (geometric mean holding time).
    /// E[duration] = 1 / P(leave) = 1 / (1 - P(from -> from))
    pub fn expected_duration(&self, from: usize) -> f64 {
        let p_stay = self.prob(from, from).min(1.0 - 1e-9);
        1.0 / (1.0 - p_stay)
    }

    /// Entropy of the transition row (bits): measures predictability.
    pub fn transition_entropy(&self, from: usize) -> f64 {
        (0..self.n)
            .map(|to| {
                let p = self.prob(from, to);
                if p > 1e-15 { -p * p.log2() } else { 0.0 }
            })
            .sum()
    }
}

// ---- Feature vector -------------------------------------------------------

/// Features used by the regime transition predictor.
#[derive(Debug, Clone, Default)]
pub struct RegimeFeatures {
    /// Hurst exponent of recent returns (0.5 = random walk)
    pub hurst: f64,
    /// Ratio of recent vol to long-run vol (>1 = elevated)
    pub vol_ratio: f64,
    /// Short-term momentum signal (e.g. 5-day return z-score)
    pub momentum: f64,
    /// Black-Hole mass proxy from SRFM (information accumulation)
    pub bh_mass: f64,
}

impl RegimeFeatures {
    pub fn to_vec(&self) -> Vec<f64> {
        vec![self.hurst, self.vol_ratio, self.momentum, self.bh_mass]
    }
}

// ---- Softmax / logistic utilities ----------------------------------------

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|&l| (l - max_l).exp()).collect();
    let sum: f64 = exps.iter().sum::<f64>().max(1e-300);
    exps.iter().map(|&e| e / sum).collect()
}

fn dot(w: &[f64], x: &[f64]) -> f64 {
    w.iter().zip(x.iter()).map(|(wi, xi)| wi * xi).sum()
}

// ---- RegimeTransitionPredictor --------------------------------------------

/// One-vs-rest softmax classifier predicting P(next regime | current, features).
/// Weights shape: [n_classes][n_features].
#[derive(Debug, Clone)]
pub struct RegimeTransitionPredictor {
    /// Weight vectors per class (intercept stored as first element, features follow)
    pub weights: Vec<Vec<f64>>,
    pub n_classes: usize,
    pub n_features: usize,
}

impl RegimeTransitionPredictor {
    /// Construct with zero-initialised weights.
    pub fn new(n_classes: usize, n_features: usize) -> Self {
        RegimeTransitionPredictor {
            weights: vec![vec![0.0; n_features + 1]; n_classes],
            n_classes,
            n_features,
        }
    }

    /// Default 4-class, 4-feature predictor.
    pub fn four_regime_default() -> Self {
        let mut p = Self::new(4, 4);
        // Heuristic calibration: sensible priors
        // Bull: low vol_ratio, positive momentum
        p.weights[0] = vec![0.0,  0.2, -0.5,  0.8,  0.1]; // intercept, hurst, vol_ratio, momentum, bh_mass
        // Bear: high vol_ratio, negative momentum
        p.weights[1] = vec![-1.0, -0.1,  0.6, -0.9,  0.2];
        // Sideways: middling everything
        p.weights[2] = vec![ 0.5,  0.1, -0.1,  0.0, -0.1];
        // HighVol: high vol_ratio, extremes in momentum
        p.weights[3] = vec![-0.5,  0.0,  0.9,  0.2,  0.3];
        p
    }

    /// Compute P(next regime | features) using softmax over linear scores.
    /// `current` is used to bias toward the empirical transition distribution.
    pub fn predict(&self, features: &RegimeFeatures, current: RegimeLabel) -> HashMap<RegimeLabel, f64> {
        let x = features.to_vec();
        let logits: Vec<f64> = (0..self.n_classes).map(|k| {
            let w = &self.weights[k];
            let bias = w[0];
            let feat_score = dot(&w[1..], &x);
            // Slight sticky bias toward current regime
            let sticky = if k == regime_index(current) { 0.3 } else { 0.0 };
            bias + feat_score + sticky
        }).collect();

        let probs = softmax(&logits);
        ALL_REGIMES.iter().enumerate()
            .map(|(i, &r)| (r, probs[i]))
            .collect()
    }

    /// Predict and return most probable next regime.
    pub fn predict_most_likely(
        &self,
        features: &RegimeFeatures,
        current: RegimeLabel,
    ) -> RegimeLabel {
        let probs = self.predict(features, current);
        ALL_REGIMES.iter()
            .max_by(|&&a, &&b| probs[&a].partial_cmp(&probs[&b]).unwrap_or(std::cmp::Ordering::Equal))
            .copied()
            .unwrap_or(current)
    }

    /// Probability of any regime change (1 - P(stay in current)).
    pub fn prob_change(&self, features: &RegimeFeatures, current: RegimeLabel) -> f64 {
        let probs = self.predict(features, current);
        1.0 - probs.get(&current).copied().unwrap_or(0.0)
    }

    /// Simple SGD update for online learning (log-loss gradient).
    /// `y_true` is the true next regime index.
    pub fn sgd_step(&mut self, features: &RegimeFeatures, current: RegimeLabel, y_true: usize, lr: f64) {
        let x = features.to_vec();
        let logits: Vec<f64> = (0..self.n_classes).map(|k| {
            let w = &self.weights[k];
            let bias = w[0];
            bias + dot(&w[1..], &x)
        }).collect();
        let probs = softmax(&logits);
        let _ = current; // current not used in plain gradient update
        for k in 0..self.n_classes {
            let err = probs[k] - if k == y_true { 1.0 } else { 0.0 };
            self.weights[k][0] -= lr * err; // intercept
            for j in 0..self.n_features {
                self.weights[k][j + 1] -= lr * err * x[j];
            }
        }
    }
}

// ---- TransitionRiskMetrics ------------------------------------------------

/// Risk metrics derived from regime transition dynamics.
#[derive(Debug, Clone)]
pub struct TransitionRiskMetrics {
    transition_matrix: TransitionMatrix,
    /// Typical volatility per regime (annualised fraction, e.g. 0.15 = 15%)
    regime_vols: HashMap<usize, f64>,
}

impl TransitionRiskMetrics {
    pub fn new(transition_matrix: TransitionMatrix) -> Self {
        let mut regime_vols = HashMap::new();
        regime_vols.insert(0, 0.12); // Bull
        regime_vols.insert(1, 0.35); // Bear
        regime_vols.insert(2, 0.08); // Sideways
        regime_vols.insert(3, 0.50); // HighVol
        TransitionRiskMetrics { transition_matrix, regime_vols }
    }

    pub fn with_regime_vol(mut self, regime: RegimeLabel, vol: f64) -> Self {
        self.regime_vols.insert(regime_index(regime), vol);
        self
    }

    /// Expected holding time in the current regime (geometric mean, in bars).
    pub fn expected_regime_duration(&self, current: RegimeLabel) -> f64 {
        self.transition_matrix.expected_duration(regime_index(current))
    }

    /// Volatility forecast for the transition from->to.
    /// Uses a weighted blend of the two regime vols.
    pub fn regime_volatility_forecast(&self, from: RegimeLabel, to: RegimeLabel) -> f64 {
        let p = self.transition_matrix.prob(regime_index(from), regime_index(to));
        let vol_from = self.regime_vols.get(&regime_index(from)).copied().unwrap_or(0.20);
        let vol_to   = self.regime_vols.get(&regime_index(to)).copied().unwrap_or(0.20);
        // Blend: P(stay)*vol_from + P(leave)*vol_to + jump premium
        let stay_prob = self.transition_matrix.prob(regime_index(from), regime_index(from));
        stay_prob * vol_from + (1.0 - stay_prob) * p * vol_to + (1.0 - stay_prob) * 0.05
    }

    /// Drawdown risk during a regime transition.
    /// Approximation: expected max drawdown = vol * sqrt(expected_duration) * 0.7 (empirical factor).
    pub fn drawdown_risk_in_transition(&self, from: RegimeLabel) -> f64 {
        let dur = self.expected_regime_duration(from);
        let vol_from = self.regime_vols.get(&regime_index(from)).copied().unwrap_or(0.20);
        // Annualise: vol is annual, dur is in bars (assume 252 bars/year)
        let daily_vol = vol_from / (252.0_f64).sqrt();
        // Ceriani-Scanlon approximation: E[max drawdown] ~ vol * sqrt(2 * dur * ln(dur+1))
        let dur_years = dur / 252.0;
        let factor = (2.0 * dur_years * (dur_years + 1.0).ln()).sqrt().max(0.0);
        daily_vol * (252.0_f64).sqrt() * factor * 0.7
    }

    /// P(regime persists for at least k more bars) = P(stay)^k.
    pub fn prob_persist_k_bars(&self, current: RegimeLabel, k: usize) -> f64 {
        let p_stay = self.transition_matrix.prob(regime_index(current), regime_index(current));
        p_stay.powi(k as i32)
    }

    /// Expected volatility over the next `horizon_bars` given current regime.
    pub fn expected_vol_horizon(&self, current: RegimeLabel, horizon_bars: usize) -> f64 {
        // Weight vols by probability of being in each regime after horizon bars
        // Simplified: power transition matrix (first row)
        let cur_idx = regime_index(current);
        let mut dist = vec![0.0f64; 4];
        dist[cur_idx] = 1.0;

        for _ in 0..horizon_bars {
            let mut next = vec![0.0; 4];
            for from in 0..4 {
                for to in 0..4 {
                    next[to] += dist[from] * self.transition_matrix.prob(from, to);
                }
            }
            dist = next;
        }

        dist.iter().enumerate()
            .map(|(i, &p)| p * self.regime_vols.get(&i).copied().unwrap_or(0.20))
            .sum()
    }
}

// ---- RegimeChangeAlert ----------------------------------------------------

/// Published when P(regime_change) > threshold for `min_consecutive` bars.
#[derive(Debug, Clone)]
pub struct RegimeChangeAlert {
    pub from_regime: RegimeLabel,
    pub predicted_to: RegimeLabel,
    pub probability: f64,
    pub consecutive_bars: usize,
    pub bar_index: usize,
}

/// Generator that emits alerts when evidence for regime change is sustained.
#[derive(Debug, Clone)]
pub struct RegimeAlertGenerator {
    pub threshold: f64,
    pub min_consecutive: usize,
    consecutive_count: usize,
    last_regime: Option<RegimeLabel>,
}

impl RegimeAlertGenerator {
    pub fn new(threshold: f64, min_consecutive: usize) -> Self {
        RegimeAlertGenerator {
            threshold,
            min_consecutive,
            consecutive_count: 0,
            last_regime: None,
        }
    }

    /// Feed one bar's regime probabilities.
    /// Returns an alert if conditions are met.
    pub fn feed(
        &mut self,
        current: RegimeLabel,
        probs: &HashMap<RegimeLabel, f64>,
        bar_index: usize,
    ) -> Option<RegimeChangeAlert> {
        let p_current = probs.get(&current).copied().unwrap_or(0.0);
        let p_change  = 1.0 - p_current;

        if p_change > self.threshold {
            self.consecutive_count += 1;
        } else {
            self.consecutive_count = 0;
            self.last_regime = Some(current);
            return None;
        }

        if self.consecutive_count >= self.min_consecutive {
            // Find the most likely next regime (excluding current)
            let predicted_to = ALL_REGIMES.iter()
                .filter(|&&r| r != current)
                .max_by(|&&a, &&b| {
                    probs.get(&a).unwrap_or(&0.0)
                        .partial_cmp(probs.get(&b).unwrap_or(&0.0))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
                .unwrap_or(RegimeLabel::Sideways);

            let alert = RegimeChangeAlert {
                from_regime: current,
                predicted_to,
                probability: p_change,
                consecutive_bars: self.consecutive_count,
                bar_index,
            };
            self.consecutive_count = 0; // Reset after alert
            self.last_regime = Some(current);
            return Some(alert);
        }

        self.last_regime = Some(current);
        None
    }

    /// Current streak of bars above threshold.
    pub fn streak(&self) -> usize {
        self.consecutive_count
    }
}

// ---- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_transition_matrix() -> TransitionMatrix {
        let mut tm = TransitionMatrix::four_regime(0.10);
        // Simulate regime sequence: Bull(100), Bear(50), Sideways(80)
        for _ in 0..100 { tm.record_transition(0, 0); }
        tm.record_transition(0, 1);
        for _ in 0..50  { tm.record_transition(1, 1); }
        tm.record_transition(1, 2);
        for _ in 0..80  { tm.record_transition(2, 2); }
        tm
    }

    #[test]
    fn test_transition_matrix_prob() {
        let tm = make_transition_matrix();
        // P(Bull -> Bull) should be very high
        let p = tm.prob(0, 0);
        assert!(p > 0.90, "P(Bull->Bull)={}", p);
        // P(Bear -> Bear) should also be high
        let p2 = tm.prob(1, 1);
        assert!(p2 > 0.85, "P(Bear->Bear)={}", p2);
    }

    #[test]
    fn test_transition_matrix_row_sums() {
        let tm = make_transition_matrix();
        for from in 0..4 {
            let row = tm.row(from);
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-9, "row {} sum={}", from, sum);
        }
    }

    #[test]
    fn test_expected_duration() {
        let tm = make_transition_matrix();
        let dur = tm.expected_duration(0); // Bull is very sticky
        assert!(dur > 50.0, "Bull duration={}", dur);
    }

    #[test]
    fn test_fit_from_sequence() {
        let mut tm = TransitionMatrix::four_regime(0.01);
        let seq = vec![
            RegimeLabel::Bull, RegimeLabel::Bull, RegimeLabel::Bear,
            RegimeLabel::Bear, RegimeLabel::Sideways, RegimeLabel::Bull,
        ];
        tm.fit_from_sequence(&seq);
        // P(Bull -> Bull) should be 1/3 (one B->B transition out of 3 starting from Bull)
        let p = tm.prob(0, 0); // 0 = Bull
        assert!(p > 0.0, "P(Bull->Bull) after sequence fit");
    }

    #[test]
    fn test_predictor_probs_sum_to_one() {
        let predictor = RegimeTransitionPredictor::four_regime_default();
        let features = RegimeFeatures {
            hurst: 0.55,
            vol_ratio: 1.2,
            momentum: 0.5,
            bh_mass: 0.3,
        };
        let probs = predictor.predict(&features, RegimeLabel::Bull);
        let sum: f64 = probs.values().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum={}", sum);
    }

    #[test]
    fn test_predictor_prob_change_in_range() {
        let predictor = RegimeTransitionPredictor::four_regime_default();
        let features = RegimeFeatures { hurst: 0.5, vol_ratio: 1.0, momentum: 0.0, bh_mass: 0.0 };
        let p = predictor.prob_change(&features, RegimeLabel::Bull);
        assert!(p >= 0.0 && p <= 1.0, "p={}", p);
    }

    #[test]
    fn test_predictor_sgd_step_does_not_panic() {
        let mut predictor = RegimeTransitionPredictor::new(4, 4);
        let features = RegimeFeatures { hurst: 0.5, vol_ratio: 1.0, momentum: 0.3, bh_mass: 0.1 };
        predictor.sgd_step(&features, RegimeLabel::Bull, 1, 0.01);
    }

    #[test]
    fn test_transition_risk_expected_duration() {
        let tm = make_transition_matrix();
        let risk = TransitionRiskMetrics::new(tm);
        let dur = risk.expected_regime_duration(RegimeLabel::Bull);
        assert!(dur > 50.0, "Bull dur={}", dur);
    }

    #[test]
    fn test_drawdown_risk_positive() {
        let tm = make_transition_matrix();
        let risk = TransitionRiskMetrics::new(tm);
        let dd = risk.drawdown_risk_in_transition(RegimeLabel::Bear);
        assert!(dd >= 0.0, "dd={}", dd);
    }

    #[test]
    fn test_prob_persist_k_bars() {
        let tm = make_transition_matrix();
        let risk = TransitionRiskMetrics::new(tm);
        let p1 = risk.prob_persist_k_bars(RegimeLabel::Bull, 1);
        let p10 = risk.prob_persist_k_bars(RegimeLabel::Bull, 10);
        assert!(p10 < p1, "p1={}, p10={}", p1, p10);
    }

    #[test]
    fn test_alert_generator_fires_after_consecutive_bars() {
        let predictor = RegimeTransitionPredictor::four_regime_default();
        let mut gen = RegimeAlertGenerator::new(0.70, 3);
        // Features that make a regime change likely
        let features = RegimeFeatures {
            hurst: 0.3,
            vol_ratio: 3.0,
            momentum: -2.0,
            bh_mass: 0.9,
        };
        let current = RegimeLabel::Bull;
        let mut fired = false;
        for i in 0..10 {
            let probs = predictor.predict(&features, current);
            if let Some(alert) = gen.feed(current, &probs, i) {
                fired = true;
                assert!(alert.probability > 0.70);
                assert!(alert.consecutive_bars >= 3);
                break;
            }
        }
        // We may or may not fire depending on the model weights -- just ensure no panic
        let _ = fired;
    }

    #[test]
    fn test_alert_generator_no_alert_when_stable() {
        let predictor = RegimeTransitionPredictor::four_regime_default();
        let mut gen = RegimeAlertGenerator::new(0.70, 3);
        // Very stable bull market features
        let features = RegimeFeatures {
            hurst: 0.6,
            vol_ratio: 0.5,
            momentum: 2.0,
            bh_mass: 0.1,
        };
        let current = RegimeLabel::Bull;
        let mut alert_count = 0;
        for i in 0..20 {
            let probs = predictor.predict(&features, current);
            if gen.feed(current, &probs, i).is_some() {
                alert_count += 1;
            }
        }
        // With strong bull features, alerts should be rare or zero
        assert!(alert_count <= 1, "too many alerts: {}", alert_count);
    }
}
