"""
live-feedback
=============
Closes the loop between live trading results and the Idea Automation Engine.

Reads live_trades.db in real-time, scores recently-adopted hypotheses against
live performance, and feeds outcomes back to the Bayesian hypothesis scorer.

Submodules
----------
live_monitor       — LiveFeedbackMonitor: main polling loop and Bayesian updates
attribution        — TradeAttributor: maps every trade to a hypothesis
drift_detector     — DriftDetector: statistical drift tests (PH, CUSUM, KS)
performance_tracker — PerformanceTracker: streaming Sharpe / DD / Calmar / etc.
"""

from __future__ import annotations

from .attribution import TradeAttributor
from .drift_detector import DriftDetector, DriftReport
from .live_monitor import LiveFeedbackMonitor, LiveScore
from .performance_tracker import PerformanceTracker

__all__ = [
    "LiveFeedbackMonitor",
    "LiveScore",
    "TradeAttributor",
    "DriftDetector",
    "DriftReport",
    "PerformanceTracker",
]
