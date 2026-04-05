"""
onchain/__init__.py
───────────────────
Public API for the On-Chain Analytics Engine.

On-chain metrics are leading indicators — price action historically lags
on-chain accumulation/distribution behaviour by days to weeks.  This module
aggregates seven complementary on-chain signals into a single composite score
and converts extreme readings into IAE hypotheses.

Typical usage
─────────────
    from onchain import OnChainEngine

    engine = OnChainEngine(db_path="idea_engine.db")
    result = engine.run("BTC-USD")
    print(result.composite_score)   # -1 (bearish) … +1 (bullish)
    print(result.hypotheses)        # list[Hypothesis] ready for IAE queue
"""

from __future__ import annotations

from .composite_signal import OnChainEngine, OnChainResult
from .data_store import OnChainDataStore
from .hypothesis_generator import OnChainHypothesisGenerator

__all__ = [
    "OnChainEngine",
    "OnChainResult",
    "OnChainDataStore",
    "OnChainHypothesisGenerator",
]
