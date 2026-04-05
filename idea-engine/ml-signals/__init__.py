"""
SRFM ML Signals Library
=======================
Machine-learned trading signals for the SRFM crypto trading lab.

This library adds a learned signal layer on top of the Black-Hole (BH)
physics-based signals produced by the core SRFM engine.  Three base
learners (LSTM, Transformer, XGBoost) are stacked via a ridge-regression
meta-learner whose weights are further adjusted each day by rolling
information-coefficient (IC).

Architecture overview
---------------------
features/
    feature_engineer  – raw-to-model feature pipeline (no lookahead)
    label_generator   – regression, classification, triple-barrier targets
    feature_store     – SQLite cache for incremental feature updates

models/
    base              – abstract MLSignal + SignalMetrics dataclass
    lstm_signal       – 2-layer LSTM, Adam + BPTT, numpy only
    transformer_signal– encoder-only transformer, numpy only
    xgboost_signal    – GBDT with stumps, two prediction heads
    ensemble          – stacking meta-learner + dynamic IC weighting

training/
    trainer           – walk-forward retraining + early stopping
    cross_validator   – purged k-fold + CPCV
    hyperopt          – random search + successive halving

inference/
    predictor         – <1 ms live prediction with feature caching
    signal_bridge     – convert ensemble score → IAE hypothesis
    drift_detector    – PSI-based staleness detection + retrain trigger

evaluation/
    backtester        – IC, ICIR, Sharpe, drawdown vs BH baseline
    report            – text-table feature importance + rolling IC report
"""

from importlib import import_module as _imp

# Lazy public API – avoids importing heavy numpy arrays at import time.
def __getattr__(name: str):  # noqa: D401
    _map = {
        "LSTMSignal":        "models.lstm_signal",
        "TransformerSignal": "models.transformer_signal",
        "XGBoostSignal":     "models.xgboost_signal",
        "EnsembleSignal":    "models.ensemble",
        "FeatureEngineer":   "features.feature_engineer",
        "LabelGenerator":    "features.label_generator",
        "FeatureStore":      "features.feature_store",
        "MLTrainer":         "training.trainer",
        "LivePredictor":     "inference.predictor",
        "SignalBridge":      "inference.signal_bridge",
        "DriftDetector":     "inference.drift_detector",
        "Backtester":        "evaluation.backtester",
        "Report":            "evaluation.report",
    }
    if name in _map:
        module = _imp(f"idea_engine.ml_signals.{_map[name]}")
        return getattr(module, name)
    raise AttributeError(f"module 'ml_signals' has no attribute {name!r}")


__all__ = [
    "LSTMSignal",
    "TransformerSignal",
    "XGBoostSignal",
    "EnsembleSignal",
    "FeatureEngineer",
    "LabelGenerator",
    "FeatureStore",
    "MLTrainer",
    "LivePredictor",
    "SignalBridge",
    "DriftDetector",
    "Backtester",
    "Report",
]

__version__ = "0.1.0"
