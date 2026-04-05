"""
data-quality — Data Quality Monitor for the Idea Automation Engine
==================================================================

Monitors incoming market data and simulation inputs for quality issues:
gaps, outliers, stale feeds, lookahead bias, and statistical biases.

Submodules:
    checker          — DataQualityChecker: OHLCV validation, gap/outlier/stale detection
    lookahead_guard  — LookaheadGuard: static-analysis + runtime signal timing validation
    feed_monitor     — FeedMonitor: live feed health, staleness, cross-exchange deviation
    bias_detector    — BiasDetector: survivorship, selection, overfitting, data-snooping
"""

from .checker import DataQualityChecker, QualityReport, QualityIssue
from .lookahead_guard import LookaheadGuard, LookaheadViolation
from .feed_monitor import FeedMonitor, FeedHealth, PriceDeviation
from .bias_detector import BiasDetector, BiasReport

__all__ = [
    "DataQualityChecker",
    "QualityReport",
    "QualityIssue",
    "LookaheadGuard",
    "LookaheadViolation",
    "FeedMonitor",
    "FeedHealth",
    "PriceDeviation",
    "BiasDetector",
    "BiasReport",
]
