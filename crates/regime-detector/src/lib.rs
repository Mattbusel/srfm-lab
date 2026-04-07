pub mod hmm;
pub mod change_point;
pub mod filters;
pub mod markov;
pub mod classification;
pub mod ensemble_regime;
pub mod online_regime_update;
pub mod transition_predictor;

pub use hmm::{HMMModel, fit as hmm_fit, predict as hmm_predict, viterbi};
pub use change_point::{
    cusum, binary_segmentation, pelt, online_change_point,
    SegmentStats, segment_statistics,
};
pub use filters::{
    kalman_filter_1d, particle_filter, hp_filter, bk_filter,
};
pub use markov::{
    HamiltonResult, RegimeParams,
    hamilton_filter, regime_conditional_moments,
    expected_regime_duration, smooth_regime_probs,
};
pub use classification::{
    Regime, VolRegime, MacroRegime, CompositeRegime,
    trend_regime, trend_regime_with_score, volatility_regime,
    macro_regime, composite_regime,
    breadth_momentum_score, rsi_regime,
    QuadrantRegime, quadrant_regime, regime_instability,
};
pub use ensemble_regime::{
    RegimeEnsemble, RegimeEnsembleState, RegimeDetector,
    DetectorVote, DetectorWeights, RegimeContext, ensemble_vote,
};
pub use online_regime_update::{OnlineRegimeModel, N_FEATURES, N_CLASSES};
pub use transition_predictor::{
    TransitionPredictor, TransitionMatrix, N_REGIMES,
};
