// bandit_explorer.rs -- Multi-armed bandit algorithms for hyperparameter
// exploration in the SRFM online learning pipeline.
//
// Provides:
//   - UCBBandit:            UCB1 with incremental mean tracking
//   - ThompsonSamplingBandit: Beta-Bernoulli Thompson sampling
//   - ContextualBandit:     per-context UCB1 (regime-aware)
//
// Use case: dynamically select the best signal combination given the
// current BH_mass level x Hurst regime context key.

use std::collections::HashMap;
use rand::Rng;

// ---------------------------------------------------------------------------
// Helper: Beta distribution sample via Johnk's method
// ---------------------------------------------------------------------------

/// Sample from Beta(alpha, beta) using Johnk's acceptance-rejection method.
/// Returns a value in (0, 1).
fn sample_beta<R: Rng>(alpha: f64, beta: f64, rng: &mut R) -> f64 {
    // Use the gamma-ratio method: Beta(a,b) = Gamma(a) / (Gamma(a) + Gamma(b))
    // We approximate Gamma sampling via the Marsaglia-Tsang method for shape >= 1
    // and use a log-normal approximation for small shapes.
    let g1 = sample_gamma(alpha, rng);
    let g2 = sample_gamma(beta,  rng);
    let sum = g1 + g2;
    if sum <= 0.0 {
        return 0.5;
    }
    (g1 / sum).clamp(1e-10, 1.0 - 1e-10)
}

/// Sample from Gamma(shape, 1) using Marsaglia-Tsang's method (shape >= 1).
/// For shape < 1 we use the Ahrens-Dieter boost: Gamma(a) = Gamma(a+1) * U^{1/a}.
fn sample_gamma<R: Rng>(shape: f64, rng: &mut R) -> f64 {
    if shape <= 0.0 {
        return 1e-10;
    }
    if shape < 1.0 {
        let u: f64 = rng.gen::<f64>().max(1e-10);
        return sample_gamma(shape + 1.0, rng) * u.powf(1.0 / shape);
    }
    // Marsaglia-Tsang: d = shape - 1/3, c = 1 / sqrt(9*d)
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = sample_normal_scalar(rng);
        let v_raw = 1.0 + c * x;
        if v_raw <= 0.0 {
            continue;
        }
        let v = v_raw.powi(3);
        let u: f64 = rng.gen::<f64>().max(1e-10);
        // Accept/reject.
        if u < 1.0 - 0.0331 * x.powi(4)
            || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln())
        {
            return d * v;
        }
    }
}

/// Draw a single standard normal sample via Box-Muller.
fn sample_normal_scalar<R: Rng>(rng: &mut R) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-10);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ---------------------------------------------------------------------------
// UCBBandit
// ---------------------------------------------------------------------------

/// UCB1 multi-armed bandit with incremental mean tracking.
///
/// Selection rule: argmax_i [ mu_i + c * sqrt(ln(t) / n_i) ]
/// where mu_i = empirical mean reward for arm i,
///       n_i  = number of times arm i was pulled,
///       t    = total pulls so far.
///
/// Unplayed arms are always selected first (n_i = 0 => infinite UCB).
#[derive(Debug, Clone)]
pub struct UCBBandit {
    /// Pull counts per arm.
    pub n:       Vec<u64>,
    /// Accumulated mean reward per arm.
    pub rewards: Vec<f64>,
    /// Total number of pulls.
    pub t:       u64,
    /// Exploration constant (higher = more exploration).
    pub c:       f64,
}

impl UCBBandit {
    /// Create a new UCB1 bandit with `n_arms` arms and exploration constant `c`.
    /// A typical default for `c` is 2.0 (sqrt(2) is also common).
    pub fn new(n_arms: usize, c: f64) -> Self {
        assert!(n_arms >= 1, "must have at least one arm");
        UCBBandit {
            n:       vec![0; n_arms],
            rewards: vec![0.0; n_arms],
            t:       0,
            c,
        }
    }

    /// Number of arms.
    #[inline]
    pub fn n_arms(&self) -> usize {
        self.n.len()
    }

    /// Select the arm with the highest UCB index.
    /// Unplayed arms (n_i = 0) always take priority.
    pub fn select(&self) -> usize {
        // Always explore unplayed arms first.
        for (i, &count) in self.n.iter().enumerate() {
            if count == 0 {
                return i;
            }
        }

        let log_t = (self.t as f64).ln();
        self.n
            .iter()
            .zip(self.rewards.iter())
            .enumerate()
            .map(|(i, (&ni, &mu))| {
                let bonus = self.c * (log_t / ni as f64).sqrt();
                (i, mu + bonus)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Update the bandit with an observed reward for the pulled arm.
    /// Uses incremental mean: mu_i <- mu_i + (r - mu_i) / n_i.
    pub fn update(&mut self, arm: usize, reward: f64) {
        assert!(arm < self.n.len(), "arm index out of range");
        self.n[arm] += 1;
        self.t      += 1;
        // Incremental mean update.
        self.rewards[arm] += (reward - self.rewards[arm]) / self.n[arm] as f64;
    }

    /// Mean reward for arm `i`.
    pub fn mean_reward(&self, arm: usize) -> f64 {
        self.rewards[arm]
    }

    /// Best arm by empirical mean (exploitation).
    pub fn best_arm(&self) -> usize {
        self.rewards
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// ThompsonSamplingBandit
// ---------------------------------------------------------------------------

/// Beta-Bernoulli Thompson Sampling bandit.
///
/// Maintains a Beta(alpha_i, beta_i) prior over the success probability
/// of each arm.  Each round a sample theta_i ~ Beta(alpha_i, beta_i) is
/// drawn and the arm with the highest sample is selected.
///
/// Update:
///   success=true  -> alpha_i += 1
///   success=false -> beta_i  += 1
#[derive(Debug, Clone)]
pub struct ThompsonSamplingBandit {
    /// Alpha parameter of the Beta prior for each arm.
    pub alpha: Vec<f64>,
    /// Beta parameter of the Beta prior for each arm.
    pub beta:  Vec<f64>,
}

impl ThompsonSamplingBandit {
    /// Construct with uniform Beta(1, 1) priors.
    pub fn new(n_arms: usize) -> Self {
        assert!(n_arms >= 1, "must have at least one arm");
        ThompsonSamplingBandit {
            alpha: vec![1.0; n_arms],
            beta:  vec![1.0; n_arms],
        }
    }

    /// Construct with informative initial priors.
    pub fn with_priors(alpha: Vec<f64>, beta: Vec<f64>) -> Self {
        assert_eq!(alpha.len(), beta.len(), "alpha and beta must have equal length");
        ThompsonSamplingBandit { alpha, beta }
    }

    /// Number of arms.
    #[inline]
    pub fn n_arms(&self) -> usize {
        self.alpha.len()
    }

    /// Sample theta_i ~ Beta(alpha_i, beta_i) for each arm and return argmax.
    pub fn select<R: Rng>(&self, rng: &mut R) -> usize {
        self.alpha
            .iter()
            .zip(self.beta.iter())
            .enumerate()
            .map(|(i, (&a, &b))| (i, sample_beta(a, b, rng)))
            .max_by(|x, y| x.1.partial_cmp(&y.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Update the posterior for arm `arm` with a Bernoulli reward.
    pub fn update(&mut self, arm: usize, success: bool) {
        assert!(arm < self.alpha.len(), "arm index out of range");
        if success {
            self.alpha[arm] += 1.0;
        } else {
            self.beta[arm]  += 1.0;
        }
    }

    /// Posterior mean (expected success probability) for arm `i`.
    pub fn posterior_mean(&self, arm: usize) -> f64 {
        self.alpha[arm] / (self.alpha[arm] + self.beta[arm])
    }

    /// Arm with the highest posterior mean.
    pub fn best_arm(&self) -> usize {
        (0..self.alpha.len())
            .max_by(|&a, &b| {
                self.posterior_mean(a)
                    .partial_cmp(&self.posterior_mean(b))
                    .unwrap()
            })
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// ContextualBandit
// ---------------------------------------------------------------------------

/// Contextual multi-armed bandit that maintains a separate `UCBBandit` per
/// context key.
///
/// Context key format for SRFM:
///   "<bh_mass_level>_<hurst_regime>"
///   e.g. "high_trending", "low_mean_reverting", "mid_random"
///
/// Each regime gets its own independent UCB1 bandit over the same arm set.
/// New contexts are initialized lazily on first access.
pub struct ContextualBandit {
    /// Policy: context_key -> UCBBandit.
    pub policy:    HashMap<String, UCBBandit>,
    /// Number of arms in every per-context bandit.
    n_arms:        usize,
    /// Exploration constant forwarded to each new bandit.
    c:             f64,
}

impl ContextualBandit {
    /// Construct with `n_arms` arms and UCB exploration constant `c`.
    pub fn new(n_arms: usize, c: f64) -> Self {
        ContextualBandit {
            policy:  HashMap::new(),
            n_arms,
            c,
        }
    }

    /// Select an arm for the given context key.
    /// If the context has not been seen before a new UCBBandit is created.
    pub fn select(&mut self, context_key: &str) -> usize {
        let bandit = self
            .policy
            .entry(context_key.to_string())
            .or_insert_with(|| UCBBandit::new(self.n_arms, self.c));
        bandit.select()
    }

    /// Update the bandit for the given context after observing a reward.
    pub fn update(&mut self, context_key: &str, arm: usize, reward: f64) {
        let bandit = self
            .policy
            .entry(context_key.to_string())
            .or_insert_with(|| UCBBandit::new(self.n_arms, self.c));
        bandit.update(arm, reward);
    }

    /// Number of distinct contexts seen so far.
    pub fn n_contexts(&self) -> usize {
        self.policy.len()
    }

    /// Best arm for a given context (by empirical mean).
    /// Returns None if the context has not been seen.
    pub fn best_arm(&self, context_key: &str) -> Option<usize> {
        self.policy.get(context_key).map(|b| b.best_arm())
    }

    /// Total pulls across all contexts.
    pub fn total_pulls(&self) -> u64 {
        self.policy.values().map(|b| b.t).sum()
    }

    /// Return a reference to the per-context bandit if it exists.
    pub fn get_bandit(&self, context_key: &str) -> Option<&UCBBandit> {
        self.policy.get(context_key)
    }
}

// ---------------------------------------------------------------------------
// SRFM regime context keys
// ---------------------------------------------------------------------------

/// Generate a standard SRFM context key from BH mass level and Hurst regime.
///
/// BH mass levels:  "low"  (< 0.3), "mid"  (0.3..0.7), "high"  (>= 0.7)
/// Hurst regimes:   "trending"     (H > 0.55)
///                  "random_walk"  (0.45 <= H <= 0.55)
///                  "mean_reverting" (H < 0.45)
pub fn srfm_context_key(bh_mass: f64, hurst: f64) -> String {
    let bh_level = if bh_mass < 0.3 {
        "low"
    } else if bh_mass < 0.7 {
        "mid"
    } else {
        "high"
    };

    let hurst_regime = if hurst > 0.55 {
        "trending"
    } else if hurst < 0.45 {
        "mean_reverting"
    } else {
        "random_walk"
    };

    format!("{}_{}", bh_level, hurst_regime)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(123)
    }

    // -- UCBBandit tests -----------------------------------------------------

    #[test]
    fn test_ucb_selects_unplayed_first() {
        let bandit = UCBBandit::new(4, 2.0);
        // With t=0 and all n_i=0, should select arm 0 (first unplayed).
        assert_eq!(bandit.select(), 0);
    }

    #[test]
    fn test_ucb_update_increments_counts() {
        let mut bandit = UCBBandit::new(3, 2.0);
        bandit.update(0, 1.0);
        bandit.update(0, 0.5);
        assert_eq!(bandit.n[0], 2);
        assert_eq!(bandit.t, 2);
    }

    #[test]
    fn test_ucb_incremental_mean() {
        let mut bandit = UCBBandit::new(2, 1.0);
        bandit.update(0, 1.0);
        bandit.update(0, 3.0);
        // Mean should be 2.0.
        assert!((bandit.mean_reward(0) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_ucb_converges_to_best_arm() {
        let mut bandit = UCBBandit::new(3, 2.0);
        let mut rng = make_rng();
        // Arm 1 has true reward 0.9, others have 0.1.
        for _ in 0..500 {
            let arm = bandit.select();
            let reward = if arm == 1 {
                0.9 + rng.gen::<f64>() * 0.1
            } else {
                0.1 + rng.gen::<f64>() * 0.1
            };
            bandit.update(arm, reward);
        }
        assert_eq!(bandit.best_arm(), 1, "UCB should identify arm 1 as best");
    }

    #[test]
    fn test_ucb_select_after_all_played() {
        let mut bandit = UCBBandit::new(3, 2.0);
        // Play each arm once.
        bandit.update(0, 0.5);
        bandit.update(1, 0.9);
        bandit.update(2, 0.2);
        // select() should return some valid arm.
        let arm = bandit.select();
        assert!(arm < 3);
    }

    // -- ThompsonSamplingBandit tests ----------------------------------------

    #[test]
    fn test_thompson_select_in_range() {
        let bandit = ThompsonSamplingBandit::new(4);
        let mut rng = make_rng();
        let arm = bandit.select(&mut rng);
        assert!(arm < 4);
    }

    #[test]
    fn test_thompson_update_alpha_beta() {
        let mut bandit = ThompsonSamplingBandit::new(2);
        bandit.update(0, true);
        bandit.update(0, true);
        bandit.update(1, false);
        assert_eq!(bandit.alpha[0] as u64, 3); // 1 initial + 2 successes
        assert_eq!(bandit.beta[1]  as u64, 2); // 1 initial + 1 failure
    }

    #[test]
    fn test_thompson_posterior_mean() {
        let bandit = ThompsonSamplingBandit::new(1);
        // Uniform Beta(1,1) -> mean = 0.5.
        assert!((bandit.posterior_mean(0) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_thompson_best_arm_after_updates() {
        let mut bandit = ThompsonSamplingBandit::new(3);
        // Arm 2 gets many successes.
        for _ in 0..50 {
            bandit.update(2, true);
        }
        bandit.update(0, false);
        bandit.update(1, false);
        assert_eq!(bandit.best_arm(), 2);
    }

    #[test]
    fn test_thompson_sample_beta_in_unit_interval() {
        let mut rng = make_rng();
        for _ in 0..100 {
            let s = sample_beta(2.0, 5.0, &mut rng);
            assert!(s > 0.0 && s < 1.0, "Beta sample out of (0,1): {}", s);
        }
    }

    // -- ContextualBandit tests ----------------------------------------------

    #[test]
    fn test_contextual_creates_context_on_first_access() {
        let mut cb = ContextualBandit::new(3, 2.0);
        let arm = cb.select("high_trending");
        assert!(arm < 3);
        assert_eq!(cb.n_contexts(), 1);
    }

    #[test]
    fn test_contextual_separate_per_context() {
        let mut cb = ContextualBandit::new(3, 2.0);
        // Play arm 2 many times for context "A".
        for _ in 0..30 {
            cb.update("A", 2, 1.0);
            cb.update("A", 0, 0.0);
            cb.update("A", 1, 0.0);
        }
        // Context "B" is fresh and should still explore.
        let arm_b = cb.select("B");
        // Context A best arm should be 2.
        assert_eq!(cb.best_arm("A"), Some(2));
        assert!(arm_b < 3);
    }

    #[test]
    fn test_contextual_total_pulls() {
        let mut cb = ContextualBandit::new(2, 1.0);
        cb.update("ctx1", 0, 0.5);
        cb.update("ctx2", 1, 0.8);
        cb.update("ctx1", 1, 0.3);
        assert_eq!(cb.total_pulls(), 3);
    }

    #[test]
    fn test_srfm_context_key_format() {
        assert_eq!(srfm_context_key(0.8, 0.6), "high_trending");
        assert_eq!(srfm_context_key(0.5, 0.5), "mid_random_walk");
        assert_eq!(srfm_context_key(0.1, 0.3), "low_mean_reverting");
    }

    #[test]
    fn test_contextual_get_bandit_none_before_use() {
        let cb = ContextualBandit::new(3, 2.0);
        assert!(cb.get_bandit("unseen").is_none());
    }
}
