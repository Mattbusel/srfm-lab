"""
macro-factor/__init__.py
─────────────────────────
Public API for the Macro Factor Engine.

Crypto doesn't trade in isolation.  DXY, rates, gold, VIX, and equity momentum
all exert powerful forces on the crypto market.  This module:

  1. Fetches daily factor data from yfinance / FRED.
  2. Classifies the current macro regime: RISK_ON | RISK_NEUTRAL | RISK_OFF | CRISIS.
  3. Converts the regime to position-size multipliers for IAE hypotheses.
  4. Persists results and emits hypotheses for the hypothesis queue.

Typical usage
─────────────
    from macro_factor import MacroFactorPipeline

    pipe = MacroFactorPipeline(db_path="idea_engine.db")
    report = pipe.run()
    print(report.regime)              # "RISK_ON"
    print(report.position_multiplier) # 1.2
    print(report.hypotheses)          # list[Hypothesis]
"""

from __future__ import annotations

from .pipeline import MacroFactorPipeline, MacroReport
from .regime_classifier import RegimeClassifier, MacroRegime
from .signal_adapter import SignalAdapter
from .factor_store import FactorStore

__all__ = [
    "MacroFactorPipeline",
    "MacroReport",
    "RegimeClassifier",
    "MacroRegime",
    "SignalAdapter",
    "FactorStore",
]
