# ============================================================
# portfolio_analytics/__init__.py
# Quantitative trading portfolio analytics suite
# ============================================================

from .correlation_monitor import CorrelationMonitor
from .factor_attribution import FactorModel, FactorAttributionEngine
from .tail_risk import TailRiskMonitor
from .kelly_optimizer import KellyOptimizer
from .performance_attribution import PerformanceAttributionEngine

__all__ = [
    "CorrelationMonitor",
    "FactorModel",
    "FactorAttributionEngine",
    "TailRiskMonitor",
    "KellyOptimizer",
    "PerformanceAttributionEngine",
]

__version__ = "1.0.0"
