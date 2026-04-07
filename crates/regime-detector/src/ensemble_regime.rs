/// ensemble_regime.rs
/// ==================
/// Ensemble regime detection: combines multiple regime detectors via weighted
/// majority voting with hysteresis to prevent rapid regime switching.
///
/// Hysteresis rule: a regime change is only accepted after 5 consecutive
/// agreement predictions from the ensemble.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Detector identifiers
// ---------------------------------------------------------------------------

/// Available regime detector types that can contribute votes to the ensemble.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegimeDetector {
    /// Breadth-and-height mass (e.g. BH-mass regime indicator).
    BHMass,
    /// Hurst exponent based persistence/anti-persistence classifier.
    Hurst,
    /// Hidden Markov Model state classifier.
    HMM,
    /// Hamilton Markov switching regime.
    Markov,
    /// Volatility percentile classifier.
    VolPercentile,
    /// Microstructure-based regime signal.
    Microstructure,
}

// ---------------------------------------------------------------------------
// Vote types
// ---------------------------------------------------------------------------

/// A single detector's vote on the current regime.
#[derive(Debug, Clone)]
pub struct DetectorVote {
    /// Which detector cast this vote.
    pub detector: RegimeDetector,
    /// The regime index this detector is voting for (0-3 for 4-regime model).
    pub regime: u8,
    /// Detector's confidence in its vote, in [0, 1].
    pub confidence: f64,
    /// Relative weight of this detector in the ensemble (not required to sum to 1).
    pub weight: f64,
}

// ---------------------------------------------------------------------------
// Context passed to the ensemble on each update
// ---------------------------------------------------------------------------

/// All raw signal inputs needed to compute individual detector votes.
#[derive(Debug, Clone, Default)]
pub struct RegimeContext {
    /// BH-mass normalised to [0, 1].
    pub bh_mass_norm: f64,
    /// Hurst exponent H in [0, 1].
    pub hurst_h: f64,
    /// HMM regime index (0-3) and its posterior probability.
    pub hmm_regime: u8,
    pub hmm_prob: f64,
    /// Hamilton Markov regime index (0-3) and probability.
    pub markov_regime: u8,
    pub markov_prob: f64,
    /// Volatility percentile in [0, 1].
    pub vol_percentile: f64,
    /// OFI z-score for microstructure detector.
    pub ofi_zscore: f64,
    /// VPIN estimate in [0, 1].
    pub vpin: f64,
    /// Kyle lambda normalised (current / baseline, typically in [0, 5]).
    pub kyle_lambda_norm: f64,
}

// ---------------------------------------------------------------------------
// Ensemble state snapshot
// ---------------------------------------------------------------------------

/// Snapshot of ensemble state after each update.
#[derive(Debug, Clone)]
pub struct RegimeEnsembleState {
    /// Current accepted regime index (0-3).
    pub current_regime: u8,
    /// Fraction of total ensemble weight in agreement with current regime.
    pub regime_confidence: f64,
    /// All votes cast in the most recent update.
    pub votes: Vec<DetectorVote>,
    /// Rolling history of (regime, timestamp) pairs.
    pub regime_history: VecDeque<(u8, i64)>,
}

// ---------------------------------------------------------------------------
// ensemble_vote -- weighted majority vote
// ---------------------------------------------------------------------------

/// Compute a weighted majority vote over a set of detector votes.
///
/// Returns `(winning_regime, confidence)` where:
///   - `winning_regime` is the regime with the highest total weight.
///   - `confidence`     is the fraction of total weight supporting that regime.
///
/// If `votes` is empty, returns `(0, 0.0)`.
pub fn ensemble_vote(votes: &[DetectorVote]) -> (u8, f64) {
    if votes.is_empty() {
        return (0, 0.0);
    }
    // Accumulate weighted votes per regime.
    let mut weights = [0.0_f64; 256];
    let mut total_weight = 0.0_f64;
    for v in votes {
        let effective = v.weight * v.confidence;
        weights[v.regime as usize] += effective;
        total_weight += effective;
    }
    if total_weight == 0.0 {
        return (0, 0.0);
    }
    // Find the regime with the highest accumulated weight.
    let winning_regime = weights
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u8)
        .unwrap_or(0);
    let confidence = weights[winning_regime as usize] / total_weight;
    (winning_regime, confidence)
}

// ---------------------------------------------------------------------------
// RegimeEnsemble
// ---------------------------------------------------------------------------

/// Ensemble regime detector combining multiple independent detectors.
///
/// Uses weighted majority voting and applies a hysteresis rule: the regime
/// only changes after `hysteresis_count` consecutive predictions agree on
/// the new regime.
#[derive(Debug, Clone)]
pub struct RegimeEnsemble {
    /// Current accepted (stable) regime index.
    current_regime: u8,
    /// Confidence of current regime from last stable vote.
    current_confidence: f64,
    /// Candidate regime being evaluated for a transition.
    candidate_regime: u8,
    /// Number of consecutive predictions supporting the candidate.
    candidate_streak: usize,
    /// Required streak length before a regime transition is accepted.
    hysteresis_count: usize,
    /// Rolling history of accepted (regime, timestamp) pairs.
    regime_history: VecDeque<(u8, i64)>,
    /// Maximum entries in the regime history buffer.
    history_cap: usize,
    /// Detector weights: maps RegimeDetector variant to weight.
    weights: DetectorWeights,
}

/// Per-detector weights for the ensemble.
#[derive(Debug, Clone)]
pub struct DetectorWeights {
    pub bh_mass: f64,
    pub hurst: f64,
    pub hmm: f64,
    pub markov: f64,
    pub vol_percentile: f64,
    pub microstructure: f64,
}

impl Default for DetectorWeights {
    fn default() -> Self {
        Self {
            bh_mass: 1.5,
            hurst: 1.0,
            hmm: 2.0,
            markov: 2.0,
            vol_percentile: 1.2,
            microstructure: 0.8,
        }
    }
}

impl RegimeEnsemble {
    /// Create a new ensemble with default detector weights.
    ///
    /// `hysteresis_count` : number of consecutive agreeing predictions before
    ///                      a regime change is accepted (default = 5).
    pub fn new(hysteresis_count: usize) -> Self {
        Self {
            current_regime: 0,
            current_confidence: 0.0,
            candidate_regime: 0,
            candidate_streak: 0,
            hysteresis_count,
            regime_history: VecDeque::new(),
            history_cap: 1000,
            weights: DetectorWeights::default(),
        }
    }

    /// Create with custom detector weights.
    pub fn with_weights(hysteresis_count: usize, weights: DetectorWeights) -> Self {
        let mut e = Self::new(hysteresis_count);
        e.weights = weights;
        e
    }

    /// Update the ensemble with a new context observation.
    ///
    /// `timestamp` is the event timestamp (caller-defined epoch).
    ///
    /// Returns the updated ensemble state.
    pub fn update(&mut self, context: &RegimeContext, timestamp: i64) -> RegimeEnsembleState {
        let votes = self.compute_votes(context);
        let (raw_regime, raw_confidence) = ensemble_vote(&votes);

        // Apply hysteresis.
        if raw_regime == self.candidate_regime {
            self.candidate_streak += 1;
        } else {
            self.candidate_regime = raw_regime;
            self.candidate_streak = 1;
        }

        if self.candidate_streak >= self.hysteresis_count
            && raw_regime != self.current_regime
        {
            // Regime change accepted.
            self.current_regime = raw_regime;
            self.current_confidence = raw_confidence;
            self.candidate_streak = 0;
            self.push_history(raw_regime, timestamp);
        } else if self.candidate_streak == 0 {
            // No candidate yet.
            self.current_confidence = raw_confidence;
        } else {
            // Update confidence even if regime unchanged.
            self.current_confidence = raw_confidence;
        }

        RegimeEnsembleState {
            current_regime: self.current_regime,
            regime_confidence: self.current_confidence,
            votes,
            regime_history: self.regime_history.clone(),
        }
    }

    /// Current accepted regime index.
    pub fn current_regime(&self) -> u8 {
        self.current_regime
    }

    /// Current regime confidence.
    pub fn current_confidence(&self) -> f64 {
        self.current_confidence
    }

    // -- Private helpers --

    fn push_history(&mut self, regime: u8, timestamp: i64) {
        if self.regime_history.len() >= self.history_cap {
            self.regime_history.pop_front();
        }
        self.regime_history.push_back((regime, timestamp));
    }

    /// Derive individual detector votes from the context.
    fn compute_votes(&self, ctx: &RegimeContext) -> Vec<DetectorVote> {
        vec![
            self.bh_mass_vote(ctx),
            self.hurst_vote(ctx),
            self.hmm_vote(ctx),
            self.markov_vote(ctx),
            self.vol_percentile_vote(ctx),
            self.microstructure_vote(ctx),
        ]
    }

    /// BH-mass vote: high mass -> trending (regime 0 = TRENDING).
    /// bh_mass_norm near 1.0 => trending, near 0.0 => ranging.
    fn bh_mass_vote(&self, ctx: &RegimeContext) -> DetectorVote {
        let (regime, confidence) = if ctx.bh_mass_norm > 0.6 {
            (0u8, ctx.bh_mass_norm)
        } else if ctx.bh_mass_norm < 0.3 {
            (1u8, 1.0 - ctx.bh_mass_norm)
        } else {
            // Ambiguous -- lower confidence.
            (1u8, 0.5)
        };
        DetectorVote {
            detector: RegimeDetector::BHMass,
            regime,
            confidence: confidence.clamp(0.0, 1.0),
            weight: self.weights.bh_mass,
        }
    }

    /// Hurst vote: H > 0.6 -> trending (persistent), H < 0.4 -> ranging.
    fn hurst_vote(&self, ctx: &RegimeContext) -> DetectorVote {
        let h = ctx.hurst_h;
        let (regime, confidence) = if h > 0.65 {
            (0u8, (h - 0.5) * 2.0) // scales [0.65, 1.0] to [0.3, 1.0]
        } else if h < 0.45 {
            (1u8, (0.5 - h) * 2.0) // scales [0, 0.45] to [1.0, 0.1]
        } else {
            (1u8, 0.4)
        };
        DetectorVote {
            detector: RegimeDetector::Hurst,
            regime,
            confidence: confidence.clamp(0.0, 1.0),
            weight: self.weights.hurst,
        }
    }

    /// HMM vote: directly use HMM's posterior regime and probability.
    fn hmm_vote(&self, ctx: &RegimeContext) -> DetectorVote {
        DetectorVote {
            detector: RegimeDetector::HMM,
            regime: ctx.hmm_regime,
            confidence: ctx.hmm_prob.clamp(0.0, 1.0),
            weight: self.weights.hmm,
        }
    }

    /// Markov vote: directly use Hamilton filter regime and probability.
    fn markov_vote(&self, ctx: &RegimeContext) -> DetectorVote {
        DetectorVote {
            detector: RegimeDetector::Markov,
            regime: ctx.markov_regime,
            confidence: ctx.markov_prob.clamp(0.0, 1.0),
            weight: self.weights.markov,
        }
    }

    /// Vol percentile vote: high percentile -> HIGH_VOL (regime 2) or CRISIS (3).
    fn vol_percentile_vote(&self, ctx: &RegimeContext) -> DetectorVote {
        let vp = ctx.vol_percentile;
        let (regime, confidence) = if vp > 0.95 {
            (3u8, vp)
        } else if vp > 0.75 {
            (2u8, (vp - 0.5) * 2.0)
        } else if vp < 0.4 {
            (0u8, 1.0 - vp)
        } else {
            (1u8, 0.5)
        };
        DetectorVote {
            detector: RegimeDetector::VolPercentile,
            regime,
            confidence: confidence.clamp(0.0, 1.0),
            weight: self.weights.vol_percentile,
        }
    }

    /// Microstructure vote: high VPIN or extreme OFI -> HIGH_VOL or CRISIS.
    fn microstructure_vote(&self, ctx: &RegimeContext) -> DetectorVote {
        let stress = ctx.vpin * 0.5
            + (ctx.ofi_zscore.abs() / 4.0).clamp(0.0, 0.5) * 0.3
            + (ctx.kyle_lambda_norm / 5.0).clamp(0.0, 1.0) * 0.2;
        let (regime, confidence) = if stress > 0.7 {
            (3u8, stress)
        } else if stress > 0.4 {
            (2u8, stress)
        } else {
            (1u8, 1.0 - stress)
        };
        DetectorVote {
            detector: RegimeDetector::Microstructure,
            regime,
            confidence: confidence.clamp(0.0, 1.0),
            weight: self.weights.microstructure,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn calm_context() -> RegimeContext {
        RegimeContext {
            bh_mass_norm: 0.7,
            hurst_h: 0.7,
            hmm_regime: 0,
            hmm_prob: 0.9,
            markov_regime: 0,
            markov_prob: 0.85,
            vol_percentile: 0.3,
            ofi_zscore: 0.2,
            vpin: 0.1,
            kyle_lambda_norm: 0.5,
        }
    }

    fn crisis_context() -> RegimeContext {
        RegimeContext {
            bh_mass_norm: 0.1,
            hurst_h: 0.35,
            hmm_regime: 3,
            hmm_prob: 0.95,
            markov_regime: 3,
            markov_prob: 0.92,
            vol_percentile: 0.98,
            ofi_zscore: 3.5,
            vpin: 0.85,
            kyle_lambda_norm: 4.0,
        }
    }

    // 1. ensemble_vote returns 0,0 for empty votes
    #[test]
    fn test_ensemble_vote_empty() {
        let (regime, conf) = ensemble_vote(&[]);
        assert_eq!(regime, 0);
        assert_eq!(conf, 0.0);
    }

    // 2. ensemble_vote single vote
    #[test]
    fn test_ensemble_vote_single() {
        let votes = vec![DetectorVote {
            detector: RegimeDetector::HMM,
            regime: 2,
            confidence: 0.8,
            weight: 1.0,
        }];
        let (regime, conf) = ensemble_vote(&votes);
        assert_eq!(regime, 2);
        assert!((conf - 1.0).abs() < 1e-10);
    }

    // 3. ensemble_vote majority wins
    #[test]
    fn test_ensemble_vote_majority() {
        let votes = vec![
            DetectorVote { detector: RegimeDetector::HMM, regime: 0, confidence: 1.0, weight: 3.0 },
            DetectorVote { detector: RegimeDetector::Markov, regime: 1, confidence: 1.0, weight: 1.0 },
            DetectorVote { detector: RegimeDetector::Hurst, regime: 0, confidence: 1.0, weight: 2.0 },
        ];
        let (regime, conf) = ensemble_vote(&votes);
        assert_eq!(regime, 0, "regime 0 has 5 out of 6 total weight");
        assert!((conf - 5.0 / 6.0).abs() < 1e-10);
    }

    // 4. Hysteresis: regime does NOT change before 5 consecutive predictions
    #[test]
    fn test_hysteresis_prevents_early_change() {
        let mut e = RegimeEnsemble::new(5);
        // Start at regime 0 (default).
        let ctx = crisis_context(); // will vote for regime 3
        for i in 0..4 {
            let state = e.update(&ctx, i as i64);
            assert_eq!(state.current_regime, 0, "should still be 0 after {i} updates");
        }
    }

    // 5. Hysteresis: regime changes after exactly 5 consecutive predictions
    #[test]
    fn test_hysteresis_allows_change_after_5() {
        let mut e = RegimeEnsemble::new(5);
        let ctx = crisis_context();
        let mut last_state = e.update(&ctx, 0);
        for i in 1..5 {
            last_state = e.update(&ctx, i as i64);
        }
        assert_ne!(last_state.current_regime, 0, "regime should have changed after 5 updates");
    }

    // 6. Calm context votes for regime 0 (TRENDING)
    #[test]
    fn test_calm_context_votes_trending() {
        let mut e = RegimeEnsemble::new(1);
        let state = e.update(&calm_context(), 0);
        assert_eq!(state.current_regime, 0);
    }

    // 7. Crisis context votes for regime 3 (CRISIS)
    #[test]
    fn test_crisis_context_votes_crisis() {
        let mut e = RegimeEnsemble::new(1);
        let state = e.update(&crisis_context(), 0);
        assert_eq!(state.current_regime, 3);
    }

    // 8. State votes vector has 6 entries (one per detector)
    #[test]
    fn test_state_vote_count() {
        let mut e = RegimeEnsemble::new(1);
        let state = e.update(&calm_context(), 0);
        assert_eq!(state.votes.len(), 6);
    }

    // 9. Regime history grows with each accepted change
    #[test]
    fn test_regime_history_grows() {
        let mut e = RegimeEnsemble::new(1);
        // Alternate between calm and crisis.
        for i in 0..6 {
            let ctx = if i % 2 == 0 { calm_context() } else { crisis_context() };
            e.update(&ctx, i as i64);
        }
        // At least some history entries from accepted changes.
        assert!(!e.regime_history.is_empty());
    }

    // 10. Streak resets when candidate changes
    #[test]
    fn test_streak_resets_on_candidate_change() {
        let mut e = RegimeEnsemble::new(5);
        // Build a partial streak toward crisis (3 updates).
        let crisis = crisis_context();
        e.update(&crisis, 0);
        e.update(&crisis, 1);
        e.update(&crisis, 2);
        // Now switch to calm -- streak should reset.
        let calm = calm_context();
        let state = e.update(&calm, 3);
        // After reset the current regime is still 0 (no accepted change yet).
        assert_eq!(state.current_regime, 0);
    }

    // 11. ensemble_vote confidence in [0,1]
    #[test]
    fn test_ensemble_vote_confidence_range() {
        let votes = vec![
            DetectorVote { detector: RegimeDetector::HMM, regime: 0, confidence: 0.7, weight: 2.0 },
            DetectorVote { detector: RegimeDetector::Markov, regime: 1, confidence: 0.6, weight: 2.0 },
            DetectorVote { detector: RegimeDetector::Hurst, regime: 0, confidence: 0.5, weight: 1.0 },
        ];
        let (_, conf) = ensemble_vote(&votes);
        assert!(conf >= 0.0 && conf <= 1.0);
    }

    // 12. Zero-confidence votes contribute zero weight
    #[test]
    fn test_zero_confidence_vote_ignored() {
        let votes = vec![
            DetectorVote { detector: RegimeDetector::HMM, regime: 2, confidence: 0.0, weight: 100.0 },
            DetectorVote { detector: RegimeDetector::Markov, regime: 1, confidence: 0.9, weight: 1.0 },
        ];
        let (regime, _) = ensemble_vote(&votes);
        // Regime 2 has 0 effective weight; regime 1 should win.
        assert_eq!(regime, 1);
    }
}
