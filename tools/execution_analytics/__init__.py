"""
execution_analytics — Execution quality, TCA, and latency monitoring suite.

Modules
-------
tca                Transaction Cost Analysis engine
slippage_tracker   Real-time slippage monitoring
market_impact      Market impact estimation (Almgren-Chriss, square-root)
fill_quality       Fill quality analytics
latency_monitor    End-to-end latency tracking
"""

from .tca import TCAEngine
from .slippage_tracker import SlippageTracker
from .market_impact import AlmgrenChrissModel, SquareRootImpact, LinearProgramExecution
from .fill_quality import FillQualityAnalyzer
from .latency_monitor import LatencyMonitor

__all__ = [
    "TCAEngine",
    "SlippageTracker",
    "AlmgrenChrissModel",
    "SquareRootImpact",
    "LinearProgramExecution",
    "FillQualityAnalyzer",
    "LatencyMonitor",
]
