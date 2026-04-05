"""
idea-engine/feature-store/__init__.py
========================================
Public API for the SRFM Feature Store.

The feature store provides persistent caching of pre-computed signal values
across all instruments and time periods, so that hypothesis testing and
strategy backtesting do not recompute features on each call.

Exports
-------
FeatureStore     : core SQLite-backed cache
FeaturePipeline  : orchestrates bulk signal computation + IC tracking
ICTracker        : IC (information coefficient) persistence and analysis
compute_ic       : standalone IC computation (Spearman / Pearson)
rolling_ic       : rolling-window IC time series
ic_decay         : IC as a function of forward-return lag
icir             : IC information ratio
t_stat           : t-statistic for IC > 0

Usage
-----
    from feature_store import FeatureStore, FeaturePipeline, ICTracker
    from signal_library import RSI, MACD, GARCHVolForecast

    pipeline = FeaturePipeline("idea_engine.db")
    pipeline.add_signals([RSI(), MACD(), GARCHVolForecast()])
    pipeline.run({"BTC": btc_df, "ETH": eth_df})

    store  = pipeline.store
    series = store.get("BTC", "rsi", start_ts="2024-01-01")

    top = pipeline.top_signals(n=10)
"""

from .store import FeatureStore
from .pipeline import FeaturePipeline, PipelineResult
from .ic_tracker import (
    ICTracker,
    compute_ic,
    rolling_ic,
    ic_decay,
    icir,
    t_stat,
)

__all__ = [
    "FeatureStore",
    "FeaturePipeline",
    "PipelineResult",
    "ICTracker",
    "compute_ic",
    "rolling_ic",
    "ic_decay",
    "icir",
    "t_stat",
]
