"""
walk-forward/__init__.py
────────────────────────
Public API for the Walk-Forward Analysis (WFA) subsystem.

The WFA subsystem validates every hypothesis on out-of-sample data before
it can be adopted into the live strategy.  It enforces a strict no-lookahead
discipline by splitting historical data into anchored IS/OOS folds and scoring
each fold independently.

Typical usage
-------------
    from walk_forward import WalkForwardEngine, FoldManager, WFAMetrics

    engine = WalkForwardEngine(db_path="idea_engine.db")
    verdict = engine.run_wfa(hypothesis_id=42, n_folds=8,
                              in_sample_months=12, out_sample_months=3)
    print(verdict)          # 'ADOPT' | 'REJECT' | 'RETEST'
"""

from __future__ import annotations

from .engine import WalkForwardEngine, WFAVerdict
from .fold_manager import (
    AnchoredFold,
    FoldManager,
    FoldMode,
    RollingFold,
)
from .metrics import (
    WFAMetrics,
    overfitting_score,
    minimum_track_record_length,
    oos_sharpe_vs_is_sharpe,
)

__all__ = [
    # engine
    "WalkForwardEngine",
    "WFAVerdict",
    # fold_manager
    "AnchoredFold",
    "FoldManager",
    "FoldMode",
    "RollingFold",
    # metrics
    "WFAMetrics",
    "overfitting_score",
    "minimum_track_record_length",
    "oos_sharpe_vs_is_sharpe",
]
