"""
backtester — Backtester Abstraction Layer for the Idea Automation Engine
========================================================================

A unified abstraction over crypto_backtest_mc.py that allows the IAE to run
backtests programmatically with any hypothesis parameter delta, capture results
consistently, and queue backtests efficiently.

Submodules:
    runner         — BacktestRunner: subprocess execution, parallel batches, DB storage
    param_manager  — ParamManager: baseline params, delta application, validation
    result_parser  — ResultParser: stdout/CSV parsing, BacktestMetrics computation
    queue_manager  — BacktestQueue: priority queue, worker loop, job lifecycle
"""

from .runner import BacktestRunner, BacktestResult
from .param_manager import ParamManager, BASELINE_PARAMS
from .result_parser import ResultParser, BacktestMetrics
from .queue_manager import BacktestQueue, BacktestJob, Priority

__all__ = [
    "BacktestRunner",
    "BacktestResult",
    "ParamManager",
    "BASELINE_PARAMS",
    "ResultParser",
    "BacktestMetrics",
    "BacktestQueue",
    "BacktestJob",
    "Priority",
]
