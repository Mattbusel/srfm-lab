"""
bridge/__init__.py

Live Trading Bridge — connects the autonomous loop to the live trader in
real-time. Allows parameter updates, signal injection, and live monitoring
without stopping the trader.
"""

from .live_param_bridge import LiveParamBridge
from .signal_injector import SignalInjector
from .trade_monitor import TradeMonitor
from .performance_tracker import PerformanceTracker
from .heartbeat import HeartbeatServer

__all__ = [
    "LiveParamBridge",
    "SignalInjector",
    "TradeMonitor",
    "PerformanceTracker",
    "HeartbeatServer",
]
