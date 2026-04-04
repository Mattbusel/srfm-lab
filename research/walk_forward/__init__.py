"""
research/walk_forward/__init__.py
Walk-Forward Analysis + CPCV (Combinatorial Purged Cross-Validation) Platform
for the SRFM-Lab trading research suite.

López de Prado (2018) CPCV implementation with full out-of-sample validation,
parameter stability testing, regime-conditional analysis, and anti-overfitting
diagnostics.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__  = "SRFM-Lab Research"

# Core split utilities
from .splits import (
    WFSplit,
    walk_forward_splits,
    expanding_window_splits,
    rolling_window_splits,
    CPCVSplitter,
    regime_stratified_splits,
    purge_overlap,
)

# Walk-forward engine and results
from .engine import (
    WalkForwardEngine,
    WFResult,
    CPCVResult,
    FoldResult,
)

# Optimization backends
from .optimizer import (
    ParamOptimizer,
    OptResult,
    grid_search,
    random_search,
    sobol_search,
    bayesian_opt,
    param_importance,
)

# Performance metrics
from .metrics import (
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    max_drawdown,
    drawdown_series,
    profit_factor,
    payoff_ratio,
    win_rate,
    expectancy,
    kelly_fraction,
    omega_ratio,
    tail_ratio,
    value_at_risk,
    conditional_var,
    hurst_exponent,
    deflated_sharpe_ratio,
    probability_of_backtest_overfitting,
    bootstrap_confidence_interval,
)

# Regime filtering
from .regime_filter import (
    RegimeFilter,
)

# Stability analysis
from .stability import (
    StabilityAnalyzer,
    StabilityReport,
    RobustnessResult,
)

# Reporting
from .report import (
    WFReport,
    generate_wf_report,
)

# BH adapter
from .backtest_adapter import (
    BHStrategyAdapter,
    run_bh_strategy,
)

__all__ = [
    # splits
    "WFSplit",
    "walk_forward_splits",
    "expanding_window_splits",
    "rolling_window_splits",
    "CPCVSplitter",
    "regime_stratified_splits",
    "purge_overlap",
    # engine
    "WalkForwardEngine",
    "WFResult",
    "CPCVResult",
    "FoldResult",
    # optimizer
    "ParamOptimizer",
    "OptResult",
    "grid_search",
    "random_search",
    "sobol_search",
    "bayesian_opt",
    "param_importance",
    # metrics
    "sharpe_ratio",
    "sortino_ratio",
    "calmar_ratio",
    "max_drawdown",
    "drawdown_series",
    "profit_factor",
    "payoff_ratio",
    "win_rate",
    "expectancy",
    "kelly_fraction",
    "omega_ratio",
    "tail_ratio",
    "value_at_risk",
    "conditional_var",
    "hurst_exponent",
    "deflated_sharpe_ratio",
    "probability_of_backtest_overfitting",
    "bootstrap_confidence_interval",
    # regime
    "RegimeFilter",
    # stability
    "StabilityAnalyzer",
    "StabilityReport",
    "RobustnessResult",
    # report
    "WFReport",
    "generate_wf_report",
    # adapter
    "BHStrategyAdapter",
    "run_bh_strategy",
]
