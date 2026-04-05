"""Champion sub-package: champion tracking, performance monitoring, degradation detection."""

from .champion_tracker import ChampionTracker, ChampionTenure
from .performance_monitor import PerformanceMonitor, PerformanceAlert
from .degradation_detector import DegradationDetector, DegradationSignal

__all__ = [
    "ChampionTracker", "ChampionTenure",
    "PerformanceMonitor", "PerformanceAlert",
    "DegradationDetector", "DegradationSignal",
]
