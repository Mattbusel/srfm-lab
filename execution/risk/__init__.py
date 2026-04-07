"""
execution/risk/__init__.py
==========================
Real-time risk aggregation for the SRFM Lab trading system.

Submodules
----------
live_var            -- VaR and CVaR via parametric, historical, and Monte Carlo methods
attribution         -- P&L attribution by signal factor
limits              -- Risk limit enforcement and position sizing guard
correlation_monitor -- Correlation matrix monitoring and concentration risk
risk_api            -- FastAPI service exposing risk metrics on port 8791

Typical usage::

    from execution.risk.live_var import VaRMonitor
    from execution.risk.limits import LimitChecker, DrawdownGuard
    from execution.risk.attribution import PnLAttributor
    from execution.risk.correlation_monitor import CorrelationMonitor
"""

from __future__ import annotations

__all__ = [
    "live_var",
    "attribution",
    "limits",
    "correlation_monitor",
    "risk_api",
]

RISK_MODULE_VERSION = "1.0.0"
