"""
regime_ml — Real-time regime detection and ML tooling suite.

Modules
-------
changepoint        Changepoint detection (PELT, BinSeg, Window, BOCPD ensemble)
hurst_monitor      Hurst exponent & fractal analysis (R/S, DFA, Variogram)
hawkes_intensity   Hawkes process for trade-cluster / order-flow intensity
entropy_analyzer   Shannon, SampEn, Permutation, ApEn, Transfer entropy
online_learner     River-based online ML with concept-drift detection
feature_store      DuckDB-backed point-in-time feature store + REST API
"""

from .changepoint import ChangepointDetector, ChangepointEvent, BOCPDDetector
from .hurst_monitor import HurstMonitor, HurstResult, HurstRegime
from .hawkes_intensity import HawkesProcess, HawkesIntensityResult
from .entropy_analyzer import EntropyAnalyzer, EntropyResult
from .online_learner import OnlineLearner, OnlinePrediction
from .feature_store import FeatureStore

__all__ = [
    "ChangepointDetector",
    "ChangepointEvent",
    "BOCPDDetector",
    "HurstMonitor",
    "HurstResult",
    "HurstRegime",
    "HawkesProcess",
    "HawkesIntensityResult",
    "EntropyAnalyzer",
    "EntropyResult",
    "OnlineLearner",
    "OnlinePrediction",
    "FeatureStore",
]

__version__ = "1.0.0"
