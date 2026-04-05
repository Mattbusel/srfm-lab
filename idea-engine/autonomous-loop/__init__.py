"""
idea-engine/autonomous-loop/__init__.py

Autonomous Research Loop — the self-improving layer of the IAE.

This package runs continuously, observing live performance, generating
hypotheses, debating them, backtesting winners, and automatically applying
validated improvements to the live strategy.
"""

from .orchestrator import AutonomousOrchestrator
from .signal_collector import SignalCollector, SystemSignal
from .pattern_miner import PatternMiner
from .hypothesis_queue import HypothesisQueue, HypothesisStage
from .backtest_bridge import BacktestBridge, BacktestResult
from .parameter_applier import ParameterApplier
from .loop_monitor import LoopMonitor

__all__ = [
    "AutonomousOrchestrator",
    "SignalCollector",
    "SystemSignal",
    "PatternMiner",
    "HypothesisQueue",
    "HypothesisStage",
    "BacktestBridge",
    "BacktestResult",
    "ParameterApplier",
    "LoopMonitor",
]
