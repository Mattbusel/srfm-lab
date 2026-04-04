"""
live_monitor — Live Trading Monitor and Diagnostics
====================================================

Continuous monitoring, diagnostics, and alerting for the srfm-lab
Alpaca live trading system.

Modules
-------
monitor     : LiveTraderMonitor — polls live_trades.db, P&L, health checks
diagnostics : LiveDiagnostics — signal quality, regime, ensemble, concentration
dashboard   : Rich terminal dashboard
alerts      : Configurable alert system
cli         : Click CLI entry-points
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "monitor",
    "diagnostics",
    "dashboard",
    "alerts",
    "cli",
]
