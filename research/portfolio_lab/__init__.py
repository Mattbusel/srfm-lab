"""
research/portfolio_lab/__init__.py

Portfolio construction and risk analysis lab for SRFM-Lab.
"""

from __future__ import annotations

from research.portfolio_lab.construction import (
    EqualWeightPortfolio,
    InverseVolPortfolio,
    MinVariancePortfolio,
    MaxSharpePortfolio,
    HRPPortfolio,
    BlackLittermanPortfolio,
    RiskParityPortfolio,
    KellyPortfolio,
)
from research.portfolio_lab.risk import (
    PortfolioRiskAnalyzer,
)
from research.portfolio_lab.rebalancing import (
    RebalancingAnalyzer,
    RebalResult,
)
from research.portfolio_lab.correlation import (
    rolling_correlation_matrix,
    dynamic_conditional_correlation,
    correlation_clustering,
    correlation_regime_analysis,
    diversification_ratio,
    effective_n_bets,
    plot_correlation_heatmap,
    plot_rolling_correlation,
)

__all__ = [
    "EqualWeightPortfolio",
    "InverseVolPortfolio",
    "MinVariancePortfolio",
    "MaxSharpePortfolio",
    "HRPPortfolio",
    "BlackLittermanPortfolio",
    "RiskParityPortfolio",
    "KellyPortfolio",
    "PortfolioRiskAnalyzer",
    "RebalancingAnalyzer",
    "RebalResult",
    "rolling_correlation_matrix",
    "dynamic_conditional_correlation",
    "correlation_clustering",
    "correlation_regime_analysis",
    "diversification_ratio",
    "effective_n_bets",
    "plot_correlation_heatmap",
    "plot_rolling_correlation",
]

__version__ = "0.1.0"
