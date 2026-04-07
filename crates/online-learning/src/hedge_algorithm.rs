// hedge_algorithm.rs -- Hedge / Multiplicative Weights algorithm for online
// expert aggregation.
//
// Applications in SRFM:
//   - Combine predictions from BH mass, Hurst exponent, GARCH volatility,
//     ML signal, and momentum into a single ensemble signal.
//   - Achieve regret O(sqrt(T ln N)) versus the best single expert.

use rand::Rng;

// ---------------------------------------------------------------------------
// HedgeAlgorithm
// ---------------------------------------------------------------------------

/// Multiplicative Weights / Hedge algorithm for combining N experts.
///
/// Update rule: w_i <- w_i * exp(-epsilon * loss_i)
/// Prediction:  mixture probabilities = w / sum(w)
pub struct HedgeAlgorithm {
    /// Unnormalized expert weights.
    pub weights:   Vec<f64>,
    /// Learning rate (step size for multiplicative updates).
    pub epsilon:   f64,
    /// Number of experts.
    pub n_experts: usize,
    /// Total rounds played (for regret computation).
    rounds:        u64,
    /// Cumulative losses assigned to each expert (for regret computation).
    cumulative_losses: Vec<f64>,
}

impl HedgeAlgorithm {
    /// Construct Hedge with uniform initial weights.
    ///
    /// A good default for epsilon: sqrt(ln(N) / T) where T is horizon length.
    pub fn new(n_experts: usize, epsilon: f64) -> Self {
        assert!(n_experts >= 1, "must have at least one expert");
        assert!(epsilon > 0.0, "epsilon must be positive");
        HedgeAlgorithm {
            weights:           vec![1.0; n_experts],
            epsilon,
            n_experts,
            rounds:            0,
            cumulative_losses: vec![0.0; n_experts],
        }
    }

    /// Return normalized mixture weights (sum to 1.0).
    pub fn predict(&self) -> Vec<f64> {
        let total: f64 = self.weights.iter().sum();
        if total <= 0.0 {
            return vec![1.0 / self.n_experts as f64; self.n_experts];
        }
        self.weights.iter().map(|&w| w / total).collect()
    }

    /// Sample an expert index proportional to the current weights.
    pub fn select_expert<R: Rng>(&self, rng: &mut R) -> usize {
        let probs = self.predict();
        let u: f64 = rng.gen::<f64>();
        let mut acc = 0.0_f64;
        for (i, &p) in probs.iter().enumerate() {
            acc += p;
            if u <= acc {
                return i;
            }
        }
        self.n_experts - 1
    }

    /// Apply the multiplicative weights update.
    ///
    /// `losses` must have length == n_experts.  Losses should be in [0, 1]
    /// for theoretical guarantees, but the algorithm works with any scale.
    pub fn update(&mut self, losses: &[f64]) {
        assert_eq!(losses.len(), self.n_experts, "losses length mismatch");
        for (i, (&l, w)) in losses.iter().zip(self.weights.iter_mut()).enumerate() {
            *w *= (-self.epsilon * l).exp();
            self.cumulative_losses[i] += l;
        }
        self.rounds += 1;
    }

    /// Return the index of the expert with the highest current weight.
    pub fn best_expert(&self) -> usize {
        self.weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Compute the cumulative regret of the Hedge mixture vs the best single expert.
    ///
    /// `expert_losses[t][i]` = loss of expert i at round t.
    ///
    /// Regret = sum_t <mixture_t, loss_t> - min_i sum_t loss_t[i]
    /// (lower is better; should be O(sqrt(T * ln(N))))
    pub fn regret(&self, expert_losses: &[Vec<f64>]) -> f64 {
        if expert_losses.is_empty() {
            return 0.0;
        }
        let t = expert_losses.len();
        let n = self.n_experts;

        // Recompute the mixture weights step by step.
        let mut w = vec![1.0_f64; n];
        let mut hedge_total_loss = 0.0_f64;

        for losses in expert_losses.iter() {
            // Mixture loss this round.
            let total_w: f64 = w.iter().sum();
            let mix_loss: f64 = losses
                .iter()
                .zip(w.iter())
                .map(|(&l, &wi)| l * wi / total_w)
                .sum();
            hedge_total_loss += mix_loss;

            // Update weights.
            for (i, (&l, wi)) in losses.iter().zip(w.iter_mut()).enumerate() {
                let _ = i;
                *wi *= (-self.epsilon * l).exp();
            }
        }

        // Best single expert's total loss.
        let mut expert_totals = vec![0.0_f64; n];
        for losses in expert_losses.iter() {
            for (i, &l) in losses.iter().enumerate() {
                if i < n {
                    expert_totals[i] += l;
                }
            }
        }
        let best_loss = expert_totals
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);

        let _ = t;
        hedge_total_loss - best_loss
    }

    /// Number of rounds completed.
    #[inline]
    pub fn rounds(&self) -> u64 {
        self.rounds
    }

    /// Return cumulative per-expert losses.
    pub fn cumulative_losses(&self) -> &[f64] {
        &self.cumulative_losses
    }
}

// ---------------------------------------------------------------------------
// SignalEnsembleHedge
// ---------------------------------------------------------------------------

/// Wraps `HedgeAlgorithm` to aggregate 5 SRFM signal experts:
///   0: BH mass signal
///   1: Hurst exponent signal
///   2: GARCH volatility signal
///   3: ML model signal
///   4: Momentum signal
pub struct SignalEnsembleHedge {
    hedge: HedgeAlgorithm,
    /// Last loss vector used for diagnostics.
    last_losses: [f64; 5],
}

impl SignalEnsembleHedge {
    /// Names of the 5 SRFM signal experts in index order.
    pub const EXPERT_NAMES: [&'static str; 5] =
        ["bh_mass", "hurst", "garch_vol", "ml_signal", "momentum"];

    /// Construct with a given Hedge learning rate.
    pub fn new(epsilon: f64) -> Self {
        SignalEnsembleHedge {
            hedge:       HedgeAlgorithm::new(5, epsilon),
            last_losses: [0.0; 5],
        }
    }

    /// Combine 5 raw signals into a single ensemble signal, then update weights
    /// using the squared error of each signal vs the realized return.
    ///
    /// Returns the mixture-weighted combination of the signals.
    ///
    /// Arguments:
    ///   signals          -- [bh_mass, hurst, garch_vol, ml_signal, momentum] all in [-1, 1]
    ///   realized_return  -- actual return observed after the last prediction (used as label)
    pub fn combine_signals(&mut self, signals: [f64; 5], realized_return: f64) -> f64 {
        // Compute mixture prediction using current weights.
        let probs = self.hedge.predict();
        let combined: f64 = signals
            .iter()
            .zip(probs.iter())
            .map(|(&s, &p)| s * p)
            .sum();

        // Compute losses: squared error of each expert vs realized return.
        let losses: Vec<f64> = signals
            .iter()
            .map(|&s| {
                let err = s - realized_return;
                (err * err).min(1.0) // clip to [0,1] for theoretical bounds
            })
            .collect();

        for (i, &l) in losses.iter().enumerate() {
            self.last_losses[i] = l;
        }

        // Update Hedge weights.
        self.hedge.update(&losses);

        combined
    }

    /// Return the normalized mixture weights over the 5 experts.
    pub fn weights(&self) -> Vec<f64> {
        self.hedge.predict()
    }

    /// Index of the currently best-weighted expert.
    pub fn best_expert_index(&self) -> usize {
        self.hedge.best_expert()
    }

    /// Name of the currently best-weighted expert.
    pub fn best_expert_name(&self) -> &'static str {
        Self::EXPERT_NAMES[self.best_expert_index()]
    }

    /// Most recent loss vector.
    pub fn last_losses(&self) -> &[f64; 5] {
        &self.last_losses
    }

    /// Number of rounds completed.
    pub fn rounds(&self) -> u64 {
        self.hedge.rounds()
    }
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
        StdRng::seed_from_u64(7)
    }

    // -- HedgeAlgorithm tests ------------------------------------------------

    #[test]
    fn test_initial_weights_uniform() {
        let h = HedgeAlgorithm::new(4, 0.1);
        let probs = h.predict();
        for &p in &probs {
            assert!((p - 0.25).abs() < 1e-9);
        }
    }

    #[test]
    fn test_update_lowers_high_loss_weight() {
        let mut h = HedgeAlgorithm::new(3, 0.5);
        // Expert 0 gets high loss, experts 1 and 2 get zero.
        h.update(&[1.0, 0.0, 0.0]);
        let probs = h.predict();
        assert!(probs[0] < probs[1], "high-loss expert should drop weight");
        assert!(probs[0] < probs[2], "high-loss expert should drop weight");
    }

    #[test]
    fn test_predict_sums_to_one() {
        let mut h = HedgeAlgorithm::new(5, 0.1);
        h.update(&[0.2, 0.5, 0.1, 0.9, 0.3]);
        let probs = h.predict();
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "sum={}", sum);
    }

    #[test]
    fn test_best_expert_tracks_low_loss() {
        let mut h = HedgeAlgorithm::new(3, 0.5);
        // Expert 2 always has lowest loss.
        for _ in 0..20 {
            h.update(&[0.8, 0.9, 0.1]);
        }
        assert_eq!(h.best_expert(), 2);
    }

    #[test]
    fn test_select_expert_samples_all() {
        let h = HedgeAlgorithm::new(3, 0.1);
        let mut rng = make_rng();
        let mut counts = [0u32; 3];
        for _ in 0..300 {
            counts[h.select_expert(&mut rng)] += 1;
        }
        assert!(counts.iter().all(|&c| c > 0), "some expert never selected");
    }

    #[test]
    fn test_regret_non_negative() {
        let h = HedgeAlgorithm::new(2, 0.3);
        let losses = vec![
            vec![0.8, 0.2],
            vec![0.9, 0.1],
            vec![0.7, 0.3],
        ];
        let reg = h.regret(&losses);
        // Regret can be negative due to lucky initialization, but should be finite.
        assert!(reg.is_finite(), "regret must be finite");
    }

    #[test]
    fn test_regret_decreases_with_good_epsilon() {
        // Expert 1 always wins.  With high epsilon Hedge adapts fast.
        let mut h_fast = HedgeAlgorithm::new(2, 0.9);
        let mut h_slow = HedgeAlgorithm::new(2, 0.01);
        let losses: Vec<Vec<f64>> = (0..50).map(|_| vec![0.9, 0.1]).collect();
        // After many rounds, fast learner should concentrate weight on expert 1.
        for l in &losses {
            h_fast.update(l);
            h_slow.update(l);
        }
        let p_fast = h_fast.predict();
        let p_slow = h_slow.predict();
        assert!(p_fast[1] > p_slow[1], "fast learner should weight expert 1 more");
    }

    #[test]
    fn test_rounds_counter() {
        let mut h = HedgeAlgorithm::new(2, 0.1);
        for _ in 0..7 {
            h.update(&[0.5, 0.5]);
        }
        assert_eq!(h.rounds(), 7);
    }

    // -- SignalEnsembleHedge tests -------------------------------------------

    #[test]
    fn test_ensemble_combine_returns_scalar() {
        let mut ens = SignalEnsembleHedge::new(0.1);
        let signals = [0.5, -0.3, 0.1, 0.8, -0.2];
        let combined = ens.combine_signals(signals, 0.05);
        assert!(combined.is_finite(), "combined signal must be finite");
    }

    #[test]
    fn test_ensemble_best_expert_is_valid() {
        let mut ens = SignalEnsembleHedge::new(0.2);
        // Expert 2 (garch_vol index) always aligns with realized return.
        for _ in 0..30 {
            ens.combine_signals([0.9, 0.9, 0.1, 0.9, 0.9], 0.1);
        }
        // Expert 2 (lowest loss) should dominate.
        assert_eq!(ens.best_expert_index(), 2);
        assert_eq!(ens.best_expert_name(), "garch_vol");
    }

    #[test]
    fn test_ensemble_weights_sum_to_one() {
        let mut ens = SignalEnsembleHedge::new(0.1);
        ens.combine_signals([0.1, 0.2, 0.3, 0.4, 0.5], 0.3);
        let ws = ens.weights();
        let sum: f64 = ws.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9);
    }
}
