"""
execution_research — Transaction Cost Analysis and Execution Quality Research
=============================================================================

Provides TCA, order splitting, market impact modelling, and latency analysis
for the srfm-lab live Alpaca trading system.

Modules
-------
tca             : Transaction Cost Analysis engine and reports
order_splitting : Alpaca $200K notional limit fix + optimal scheduling
market_impact   : Impact model calibration (linear, sqrt, Almgren-Chriss)
latency         : Order-to-fill latency diagnostics
cli             : Click CLI entry-points
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = [
    "tca",
    "order_splitting",
    "market_impact",
    "latency",
    "cli",
]
