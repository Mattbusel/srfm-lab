"""
idea-engine/signal-library/base.py
====================================
Base classes and registry for all trading signals in the SRFM Idea Engine.

Every concrete signal:
  - Inherits from ``Signal``
  - Implements ``compute(df)`` returning a ``pd.Series`` indexed like df
  - Declares name, category, lookback, signal_type
  - Can optionally override ``validate(df)``

The ``SignalRegistry`` is a singleton-style registry used by the feature store
and hypothesis generator to look up and enumerate signals.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_OHLCV_COLS: List[str] = ["Open", "High", "Low", "Close", "Volume"]

CATEGORIES: List[str] = [
    "momentum",
    "mean_reversion",
    "volatility",
    "volume",
    "cross_asset",
    "macro",
    "microstructure",
    "composite",
]

SIGNAL_TYPES: List[str] = [
    "continuous",   # unbounded or bounded real value
    "binary",       # 0 or 1
    "categorical",  # discrete labels
]


# ---------------------------------------------------------------------------
# SignalResult dataclass
# ---------------------------------------------------------------------------

@dataclass
class SignalResult:
    """
    Structured output from a signal computation.

    Attributes
    ----------
    values : pd.Series
        The signal time series, indexed identically to the input DataFrame.
    signal_name : str
        Canonical name of the signal (matches Signal.name).
    category : str
        One of CATEGORIES.
    metadata : dict
        Any extra information the signal wants to expose (e.g. intermediate
        series, parameter settings, warnings).
    """
    values:      pd.Series
    signal_name: str
    category:    str
    metadata:    Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.values, pd.Series):
            raise TypeError("SignalResult.values must be a pd.Series")
        if self.category not in CATEGORIES:
            raise ValueError(
                f"Unknown category '{self.category}'. "
                f"Must be one of {CATEGORIES}."
            )


# ---------------------------------------------------------------------------
# Abstract base Signal
# ---------------------------------------------------------------------------

class Signal(ABC):
    """
    Abstract base class for all trading signals.

    Subclasses MUST set class-level attributes:
        name        : str   — unique identifier used in registry lookups
        category    : str   — one of CATEGORIES
        lookback    : int   — minimum bars required for a meaningful value
        signal_type : str   — one of SIGNAL_TYPES

    And MUST implement:
        compute(df) -> pd.Series
    """

    # ------------------------------------------------------------------
    # Class-level declarations (overridden in each subclass)
    # ------------------------------------------------------------------
    name:        str = ""
    category:    str = ""
    lookback:    int = 1
    signal_type: str = "continuous"

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Validate that concrete (non-abstract) subclasses declare required attrs
        if not getattr(cls, "__abstractmethods__", None):
            for attr in ("name", "category", "lookback", "signal_type"):
                val = getattr(cls, attr, None)
                if not val and val != 0:
                    logger.warning(
                        "Signal subclass %s has empty '%s'. "
                        "It will not be auto-registered.",
                        cls.__name__,
                        attr,
                    )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute the signal over the entire DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV DataFrame with at minimum the columns this signal needs.
            Columns are expected to be titled with standard capitalisation:
            Open, High, Low, Close, Volume.

        Returns
        -------
        pd.Series
            Signal values indexed exactly like df. NaN is acceptable for
            warmup periods (first ``lookback`` rows).
        """
        ...  # pragma: no cover

    def compute_result(self, df: pd.DataFrame) -> SignalResult:
        """
        Convenience wrapper: validate, compute, and return a SignalResult.
        """
        self.validate(df)
        values = self.compute(df)
        return SignalResult(
            values=values,
            signal_name=self.name,
            category=self.category,
            metadata={"lookback": self.lookback, "signal_type": self.signal_type},
        )

    def validate(self, df: pd.DataFrame) -> None:
        """
        Validate that df contains the columns this signal needs.

        Default implementation checks for Close column only (the minimum
        requirement). Override in subclasses that need High/Low/Volume.

        Raises
        ------
        ValueError
            If required columns are missing or df is empty.
        """
        if df is None or df.empty:
            raise ValueError(f"Signal '{self.name}': received empty DataFrame.")
        if "Close" not in df.columns:
            raise ValueError(
                f"Signal '{self.name}': DataFrame must have a 'Close' column."
            )

    def validate_ohlcv(self, df: pd.DataFrame) -> None:
        """Strict OHLCV validation — call from subclasses that need all columns."""
        if df is None or df.empty:
            raise ValueError(f"Signal '{self.name}': received empty DataFrame.")
        missing = [c for c in REQUIRED_OHLCV_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Signal '{self.name}': missing columns {missing}. "
                f"Required: {REQUIRED_OHLCV_COLS}"
            )

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        """Exponential moving average."""
        return series.ewm(span=span, min_periods=1, adjust=False).mean()

    @staticmethod
    def _sma(series: pd.Series, window: int) -> pd.Series:
        """Simple moving average."""
        return series.rolling(window, min_periods=1).mean()

    @staticmethod
    def _true_range(df: pd.DataFrame) -> pd.Series:
        """Wilder's True Range."""
        prev_close = df["Close"].shift(1)
        tr = pd.concat(
            [
                df["High"] - df["Low"],
                (df["High"] - prev_close).abs(),
                (df["Low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return tr

    @staticmethod
    def _returns(series: pd.Series, periods: int = 1) -> pd.Series:
        """Log returns over N periods."""
        return np.log(series / series.shift(periods))

    @staticmethod
    def _zscore(series: pd.Series, window: int) -> pd.Series:
        """Rolling z-score."""
        mu  = series.rolling(window, min_periods=max(window // 2, 2)).mean()
        std = series.rolling(window, min_periods=max(window // 2, 2)).std()
        return (series - mu) / std.replace(0.0, np.nan)

    @staticmethod
    def _percentile_rank(series: pd.Series, window: int) -> pd.Series:
        """
        Rolling percentile rank (0-100) of the current value in the window.
        """
        def _rank(arr: np.ndarray) -> float:
            if len(arr) == 0:
                return np.nan
            return float(np.sum(arr[:-1] < arr[-1]) / (len(arr) - 1) * 100) \
                if len(arr) > 1 else 50.0

        return series.rolling(window, min_periods=2).apply(_rank, raw=True)

    def __repr__(self) -> str:
        return (
            f"<Signal name={self.name!r} "
            f"category={self.category!r} "
            f"lookback={self.lookback} "
            f"signal_type={self.signal_type!r}>"
        )


# ---------------------------------------------------------------------------
# Signal Registry
# ---------------------------------------------------------------------------

class SignalRegistry:
    """
    A global registry that maps signal names to Signal classes.

    Usage
    -----
        registry = SignalRegistry()
        registry.register(MySignal)
        cls  = registry.get("my_signal")
        inst = cls()
        sig  = inst.compute(df)

    The module-level ``REGISTRY`` singleton is populated automatically by
    signal-library/__init__.py.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, Type[Signal]] = {}

    # ------------------------------------------------------------------

    def register(self, signal_cls: Type[Signal]) -> Type[Signal]:
        """
        Register a Signal subclass.

        Can be used as a class decorator::

            @REGISTRY.register
            class MySignal(Signal):
                name = "my_signal"
                ...

        Returns the class unchanged so it can still be used normally.
        """
        if not (isinstance(signal_cls, type) and issubclass(signal_cls, Signal)):
            raise TypeError(f"{signal_cls} is not a Signal subclass.")
        key = signal_cls.name
        if not key:
            raise ValueError(
                f"Cannot register {signal_cls.__name__}: name is empty."
            )
        if key in self._registry and self._registry[key] is not signal_cls:
            logger.warning(
                "SignalRegistry: overwriting existing registration for '%s'.", key
            )
        self._registry[key] = signal_cls
        return signal_cls

    def get(self, name: str) -> Type[Signal]:
        """Return the Signal class for *name*, raising KeyError if absent."""
        try:
            return self._registry[name]
        except KeyError:
            raise KeyError(
                f"No signal named '{name}'. "
                f"Available: {sorted(self._registry)}"
            ) from None

    def instantiate(self, name: str, **kwargs: Any) -> Signal:
        """Convenience: get + instantiate with kwargs."""
        return self.get(name)(**kwargs)

    def list_all(self) -> List[str]:
        """Return sorted list of all registered signal names."""
        return sorted(self._registry)

    def list_by_category(self, category: str) -> List[str]:
        """Return sorted list of signal names in a given category."""
        return sorted(
            name
            for name, cls in self._registry.items()
            if cls.category == category
        )

    def categories(self) -> List[str]:
        """Return all categories that have at least one registered signal."""
        return sorted({cls.category for cls in self._registry.values()})

    def all_classes(self) -> Dict[str, Type[Signal]]:
        """Return a shallow copy of the internal registry dict."""
        return dict(self._registry)

    def summary(self) -> pd.DataFrame:
        """
        Return a DataFrame summarising all registered signals.

        Columns: name, category, lookback, signal_type
        """
        rows = []
        for name, cls in sorted(self._registry.items()):
            rows.append(
                {
                    "name":        name,
                    "category":    cls.category,
                    "lookback":    cls.lookback,
                    "signal_type": cls.signal_type,
                    "class":       cls.__name__,
                }
            )
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"<SignalRegistry n={len(self._registry)} signals>"


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

REGISTRY = SignalRegistry()
