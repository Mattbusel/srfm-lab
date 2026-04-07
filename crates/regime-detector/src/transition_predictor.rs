/// transition_predictor.rs
/// =======================
/// Learns empirical regime transition probabilities from observed transitions
/// and provides predictions, entropy estimates, and expected durations.
///
/// Supports 4 regimes: 0=TRENDING, 1=RANGING, 2=HIGH_VOL, 3=CRISIS.

/// Number of regime states in the transition model.
pub const N_REGIMES: usize = 4;

// ---------------------------------------------------------------------------
// TransitionMatrix
// ---------------------------------------------------------------------------

/// Stores raw transition counts and derives probability estimates.
///
/// `counts[i][j]` = number of times regime `i` was followed by regime `j`.
/// `matrix[i][j]` = P(to = j | from = i), derived from counts via Laplace
///                  smoothing (alpha = 1) to avoid zero probabilities.
#[derive(Debug, Clone)]
pub struct TransitionMatrix {
    /// Raw observed transition counts.
    pub counts: [[u64; N_REGIMES]; N_REGIMES],
    /// Transition probability matrix (row-stochastic).
    pub matrix: [[f64; N_REGIMES]; N_REGIMES],
}

impl Default for TransitionMatrix {
    fn default() -> Self {
        Self::new()
    }
}

impl TransitionMatrix {
    /// Create an empty transition matrix (uniform prior via Laplace smoothing).
    pub fn new() -> Self {
        let uniform = 1.0 / N_REGIMES as f64;
        Self {
            counts: [[0; N_REGIMES]; N_REGIMES],
            matrix: [[uniform; N_REGIMES]; N_REGIMES],
        }
    }

    /// Record an observed transition from `from_regime` to `to_regime` and
    /// recompute the probability row for `from_regime`.
    pub fn observe(&mut self, from: usize, to: usize) {
        debug_assert!(from < N_REGIMES && to < N_REGIMES);
        self.counts[from][to] += 1;
        self.recompute_row(from);
    }

    /// Recompute probability row `row` from counts using Laplace (add-1) smoothing.
    fn recompute_row(&mut self, row: usize) {
        // Laplace smoothed counts: count + 1 for each cell.
        let alpha = 1.0_f64; // Laplace smoothing pseudo-count
        let total: f64 = self.counts[row].iter().sum::<u64>() as f64
            + alpha * N_REGIMES as f64;
        for j in 0..N_REGIMES {
            self.matrix[row][j] = (self.counts[row][j] as f64 + alpha) / total;
        }
    }

    /// Return P(to | from).
    pub fn prob(&self, from: usize, to: usize) -> f64 {
        self.matrix[from][to]
    }

    /// Total observations originating from `from_regime`.
    pub fn row_count(&self, from: usize) -> u64 {
        self.counts[from].iter().sum()
    }
}

// ---------------------------------------------------------------------------
// TransitionPredictor
// ---------------------------------------------------------------------------

/// Maintains a transition count matrix and provides transition-based predictions.
#[derive(Debug, Clone, Default)]
pub struct TransitionPredictor {
    pub matrix: TransitionMatrix,
    /// Most recently observed regime (for stateful stream processing).
    last_regime: Option<u8>,
}

impl TransitionPredictor {
    /// Create a new predictor with a uniform prior.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an explicit from->to transition.
    pub fn observe_transition(&mut self, from_regime: u8, to_regime: u8) {
        debug_assert!((from_regime as usize) < N_REGIMES);
        debug_assert!((to_regime as usize) < N_REGIMES);
        self.matrix.observe(from_regime as usize, to_regime as usize);
        self.last_regime = Some(to_regime);
    }

    /// Feed the next observed regime, automatically pairing it with the
    /// previous observation to form a transition.
    ///
    /// Returns the transition that was recorded, or `None` on the first call.
    pub fn observe_regime(&mut self, regime: u8) -> Option<(u8, u8)> {
        let prev = self.last_regime;
        self.last_regime = Some(regime);
        if let Some(from) = prev {
            self.observe_transition(from, regime);
            Some((from, regime))
        } else {
            None
        }
    }

    /// Empirical transition probability P(to | from).
    ///
    /// Returns a Laplace-smoothed estimate so the result is always positive.
    pub fn transition_probability(&self, from: u8, to: u8) -> f64 {
        self.matrix.prob(from as usize, to as usize)
    }

    /// Return the most likely next regime given the current one.
    ///
    /// Returns `(regime, probability)`.
    pub fn most_likely_next_regime(&self, current: u8) -> (u8, f64) {
        let row = current as usize;
        let mut best_j = 0usize;
        let mut best_p = 0.0_f64;
        for j in 0..N_REGIMES {
            if self.matrix.matrix[row][j] > best_p {
                best_p = self.matrix.matrix[row][j];
                best_j = j;
            }
        }
        (best_j as u8, best_p)
    }

    /// Shannon entropy of the transition distribution from `current_regime`.
    ///
    /// H = -sum_{j} P(j | current) * log2(P(j | current))
    ///
    /// High entropy means uncertain transitions (close to uniform).
    /// Low entropy means predictable transitions (concentrated on one regime).
    /// Maximum entropy for 4 regimes = log2(4) = 2.0 bits.
    pub fn transition_entropy(&self, current: u8) -> f64 {
        let row = current as usize;
        let mut h = 0.0_f64;
        for j in 0..N_REGIMES {
            let p = self.matrix.matrix[row][j];
            if p > 0.0 {
                h -= p * p.log2();
            }
        }
        h
    }

    /// Expected duration of `regime` in periods, assuming geometric distribution.
    ///
    /// E[duration] = 1 / (1 - P(stay | regime))
    ///
    /// where P(stay) = P(regime -> regime) = P(transition to same regime).
    ///
    /// If P(stay) >= 1 (shouldn't happen with Laplace smoothing), returns a
    /// large finite number.
    pub fn expected_regime_duration(&self, regime: u8) -> f64 {
        let p_stay = self.transition_probability(regime, regime);
        if p_stay >= 1.0 {
            return 1e10;
        }
        1.0 / (1.0 - p_stay)
    }

    /// Return the current transition probability matrix as a flat 4x4 array.
    pub fn probability_matrix(&self) -> &[[f64; N_REGIMES]; N_REGIMES] {
        &self.matrix.matrix
    }

    /// Return the raw counts matrix.
    pub fn counts_matrix(&self) -> &[[u64; N_REGIMES]; N_REGIMES] {
        &self.matrix.counts
    }

    /// Most recently observed regime.
    pub fn last_regime(&self) -> Option<u8> {
        self.last_regime
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Initial probabilities are uniform
    #[test]
    fn test_initial_uniform_prior() {
        let p = TransitionPredictor::new();
        for from in 0..N_REGIMES as u8 {
            let mut sum = 0.0;
            for to in 0..N_REGIMES as u8 {
                let prob = p.transition_probability(from, to);
                assert!(prob > 0.0 && prob <= 1.0);
                sum += prob;
            }
            assert!((sum - 1.0).abs() < 1e-10, "row {from} does not sum to 1: {sum}");
        }
    }

    // 2. observe_transition updates the row distribution
    #[test]
    fn test_observe_updates_probs() {
        let mut p = TransitionPredictor::new();
        // Repeatedly observe 0->0 (self-transition) to make P(0->0) dominant.
        for _ in 0..100 {
            p.observe_transition(0, 0);
        }
        let p00 = p.transition_probability(0, 0);
        let p01 = p.transition_probability(0, 1);
        assert!(p00 > p01, "P(0->0) should dominate after many self-transitions");
    }

    // 3. Row probabilities always sum to 1 after updates
    #[test]
    fn test_row_sums_to_one_after_updates() {
        let mut p = TransitionPredictor::new();
        p.observe_transition(1, 2);
        p.observe_transition(1, 0);
        p.observe_transition(1, 1);
        let sum: f64 = (0..N_REGIMES as u8).map(|to| p.transition_probability(1, to)).sum();
        assert!((sum - 1.0).abs() < 1e-10, "sum = {sum}");
    }

    // 4. most_likely_next_regime returns most frequent observed next regime
    #[test]
    fn test_most_likely_next_regime() {
        let mut p = TransitionPredictor::new();
        // From regime 2, mostly transition to regime 3.
        for _ in 0..50 {
            p.observe_transition(2, 3);
        }
        p.observe_transition(2, 0);
        let (next, prob) = p.most_likely_next_regime(2);
        assert_eq!(next, 3, "most likely next from 2 should be 3");
        assert!(prob > 0.5);
    }

    // 5. Transition entropy is max (log2(4) = 2) for uniform distribution
    #[test]
    fn test_entropy_max_for_uniform() {
        let p = TransitionPredictor::new();
        let h = p.transition_entropy(0);
        let max_h = (N_REGIMES as f64).log2();
        assert!((h - max_h).abs() < 1e-10, "uniform entropy should be log2(4), got {h}");
    }

    // 6. Transition entropy decreases when distribution becomes concentrated
    #[test]
    fn test_entropy_decreases_with_concentration() {
        let mut p = TransitionPredictor::new();
        let h_before = p.transition_entropy(1);
        for _ in 0..500 {
            p.observe_transition(1, 2);
        }
        let h_after = p.transition_entropy(1);
        assert!(h_after < h_before, "entropy should decrease as distribution concentrates");
    }

    // 7. expected_regime_duration >= 1.0
    #[test]
    fn test_expected_duration_at_least_one() {
        let p = TransitionPredictor::new();
        for regime in 0..N_REGIMES as u8 {
            let d = p.expected_regime_duration(regime);
            assert!(d >= 1.0, "duration must be >= 1 for regime {regime}, got {d}");
        }
    }

    // 8. expected_regime_duration increases as self-transition probability increases
    #[test]
    fn test_expected_duration_increases_with_persistence() {
        let mut p = TransitionPredictor::new();
        let d_before = p.expected_regime_duration(0);
        for _ in 0..200 {
            p.observe_transition(0, 0);
        }
        let d_after = p.expected_regime_duration(0);
        assert!(d_after > d_before, "duration should increase with self-transition observations");
    }

    // 9. observe_regime pairs consecutive observations
    #[test]
    fn test_observe_regime_pairs() {
        let mut p = TransitionPredictor::new();
        let r1 = p.observe_regime(0);
        assert!(r1.is_none(), "first call returns None");
        let r2 = p.observe_regime(1);
        assert_eq!(r2, Some((0, 1)));
        assert_eq!(p.counts_matrix()[0][1], 1);
    }

    // 10. last_regime tracks most recently observed regime
    #[test]
    fn test_last_regime_tracking() {
        let mut p = TransitionPredictor::new();
        assert_eq!(p.last_regime(), None);
        p.observe_transition(0, 2);
        assert_eq!(p.last_regime(), Some(2));
        p.observe_transition(2, 1);
        assert_eq!(p.last_regime(), Some(1));
    }

    // 11. Transition probability is in (0, 1] at all times
    #[test]
    fn test_transition_probability_in_range() {
        let mut p = TransitionPredictor::new();
        for _ in 0..10 {
            p.observe_transition(3, 3);
        }
        for from in 0..N_REGIMES as u8 {
            for to in 0..N_REGIMES as u8 {
                let prob = p.transition_probability(from, to);
                assert!(prob > 0.0 && prob <= 1.0, "P({from}->{to}) = {prob} out of range");
            }
        }
    }

    // 12. Transition entropy is non-negative
    #[test]
    fn test_entropy_non_negative() {
        let mut p = TransitionPredictor::new();
        for _ in 0..50 { p.observe_transition(0, 0); }
        for regime in 0..N_REGIMES as u8 {
            let h = p.transition_entropy(regime);
            assert!(h >= 0.0, "entropy cannot be negative for regime {regime}");
        }
    }

    // 13. TransitionMatrix row_count reflects observations
    #[test]
    fn test_row_count() {
        let mut tm = TransitionMatrix::new();
        tm.observe(1, 0);
        tm.observe(1, 2);
        tm.observe(1, 2);
        assert_eq!(tm.row_count(1), 3);
        assert_eq!(tm.row_count(0), 0);
    }

    // 14. probability_matrix rows sum to 1
    #[test]
    fn test_probability_matrix_rows_sum_to_one() {
        let mut p = TransitionPredictor::new();
        p.observe_transition(0, 1);
        p.observe_transition(0, 2);
        p.observe_transition(3, 0);
        let mat = p.probability_matrix();
        for (i, row) in mat.iter().enumerate() {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-10, "row {i} sums to {s}");
        }
    }
}
