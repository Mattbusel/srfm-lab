"""
research/signal_analytics/__init__.py
======================================
Signal Analytics & IC/ICIR Framework for SRFM-Lab.

Provides comprehensive tools for measuring signal quality, factor attribution,
alpha decay analysis, and BH-specific signal diagnostics.

BH Signal Context
-----------------
- tf_score   : timeframe alignment score (0–7 across 7 timeframes)
- mass       : conviction weight (0–2)
- ATR        : Average True Range (raw)
- ensemble_signal : composite directional score (−1 to +1)
- delta_score = tf_score × mass × ATR / vol²

Trade Schema
------------
exit_time, sym, entry_price, exit_price, dollar_pos,
pnl, hold_bars, regime

Modules
-------
ic_framework      – IC, ICIR, IC decay, regime-conditioned IC
factor_model      – Barra-style cross-sectional regression, Fama-MacBeth, PCA
alpha_decay       – Half-life estimation, optimal holding period, turnover
quantile_analysis – Quintile portfolio analysis, monotonicity, hit-rate
bh_signal_quality – BH-specific diagnostics (mass sweep, tf_score, ensemble)
portfolio_signal  – Portfolio-level aggregation, concentration, capacity
report            – HTML/console tearsheet generation
cli               – Click CLI entry-points
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "ic_framework",
    "factor_model",
    "alpha_decay",
    "quantile_analysis",
    "bh_signal_quality",
    "portfolio_signal",
    "report",
    "cli",
    "scoring",
    "regime_signals",
]

# Convenience re-exports of the most frequently used symbols
from research.signal_analytics.ic_framework import (
    ICCalculator,
    ICDecayResult,
    compute_ic,
    icir,
    rolling_ic,
)
from research.signal_analytics.factor_model import (
    FactorModel,
    FMBResult,
    AttributionResult,
)
from research.signal_analytics.alpha_decay import (
    AlphaDecayAnalyzer,
    DecayModel,
    TurnoverStats,
)
from research.signal_analytics.quantile_analysis import (
    QuantileAnalyzer,
    QuantileResult,
)
from research.signal_analytics.bh_signal_quality import (
    BHSignalAnalyzer,
    ActivationQualityReport,
)
from research.signal_analytics.portfolio_signal import (
    PortfolioSignalAnalyzer,
)
from research.signal_analytics.report import (
    SignalAnalyticsReport,
    generate_signal_report,
)
from research.signal_analytics.diagnostics import (
    SignalDiagnostics,
    PermutationTestResult,
    WalkForwardResult,
    OverfittingDiagnosis,
)
from research.signal_analytics.utils import (
    validate_trades,
    normalize_returns,
    sharpe_ratio,
    sortino_ratio,
    compute_max_drawdown,
    performance_summary,
    generate_synthetic_trades,
    generate_synthetic_panel,
    combine_signals,
    compute_forward_returns,
    ic_significance_grid,
)
from research.signal_analytics.scoring import (
    SignalScorer,
    EnsembleOptimisationResult,
    DecayAdjustedPositions,
    RiskParityWeights,
)
from research.signal_analytics.regime_signals import (
    RegimeSignalAnalyzer,
    RegimeStats,
    RegimeSignalSummary,
    RegimeTransitionMatrix,
    HurstResult,
    AdaptiveFilterResult,
    label_vol_regimes,
    label_trend_regimes,
    label_correlation_regimes,
    regime_transition_matrix,
    hurst_exponent,
)
