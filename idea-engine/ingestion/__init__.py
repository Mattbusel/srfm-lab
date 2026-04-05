"""
ingestion/__init__.py
──────────────────────
Public API for the idea-engine ingestion layer.
"""

from .config import DB_PATH, SRFM_ROOT, ENGINE_ROOT
from .pipeline import IngestionPipeline, PipelineResult
from .types import (
    BacktestResult,
    FoldMetrics,
    LiveTradeData,
    MinedPattern,
    PatternStatus,
    PatternType,
    WalkForwardResult,
)

__all__ = [
    # Pipeline
    "IngestionPipeline",
    "PipelineResult",
    # Types
    "MinedPattern",
    "PatternType",
    "PatternStatus",
    "BacktestResult",
    "LiveTradeData",
    "WalkForwardResult",
    "FoldMetrics",
    # Paths
    "DB_PATH",
    "SRFM_ROOT",
    "ENGINE_ROOT",
]
