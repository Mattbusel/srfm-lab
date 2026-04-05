"""
ingestion/loaders/__init__.py
─────────────────────────────
Public API for all ingestion loaders.
"""

from .backtest_loader import load_backtest
from .equity_curve_loader import (
    from_csv,
    from_json,
    from_series,
    from_sqlite,
    load_equity_curve,
)
from .trade_loader import load_live_trades
from .walk_forward_loader import load_walk_forward

__all__ = [
    "load_backtest",
    "load_live_trades",
    "load_walk_forward",
    "load_equity_curve",
    "from_sqlite",
    "from_csv",
    "from_json",
    "from_series",
]
