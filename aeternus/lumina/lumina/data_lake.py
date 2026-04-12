"""
lumina/data_lake.py

Large-scale financial data pipeline for Lumina.

Covers:
  - Streaming Parquet/Arrow reading via PyArrow and pandas
  - Multi-asset tick data loader with alignment
  - Market calendar awareness (NYSE, NASDAQ, LSE, etc.)
  - Corporate action adjustment (splits, dividends)
  - Survivorship bias handling via universe snapshot files
  - Data quality checks (stale prices, outliers, gaps)
  - Efficient batching with prefetching via background threads
  - Memory-mapped dataset for huge corpora
  - Integration with Chronos LOB (Limit Order Book) output format
  - Feature engineering pipeline (returns, vol, microstructure)
  - Time-series normalization (z-score, rank, robust)
"""

from __future__ import annotations

import bisect
import collections
import concurrent.futures
import contextlib
import datetime
import functools
import hashlib
import io
import json
import logging
import math
import mmap
import os
import pathlib
import pickle
import queue
import random
import struct
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Generator, Iterable, Iterator,
    List, Optional, Set, Sequence, Tuple, Union,
)

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, IterableDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.dataset as pad
    _ARROW_AVAILABLE = True
except ImportError:
    _ARROW_AVAILABLE = False
    logger.warning("PyArrow not available; Parquet streaming disabled.")

try:
    import pandas_market_calendars as mcal
    _CALENDAR_AVAILABLE = True
except ImportError:
    _CALENDAR_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OHLCV_COLS = ["open", "high", "low", "close", "volume"]
TICK_COLS = ["timestamp", "price", "size", "side"]
LOB_COLS = [
    "bid_p1", "bid_s1", "bid_p2", "bid_s2", "bid_p3", "bid_s3",
    "ask_p1", "ask_s1", "ask_p2", "ask_s2", "ask_p3", "ask_s3",
    "mid_price", "spread", "imbalance",
]
MICROSTRUCTURE_COLS = [
    "bid_ask_spread", "depth_imbalance", "trade_flow_imbalance",
    "realized_vol_1m", "realized_vol_5m", "price_impact",
    "vwap_deviation", "turnover_rate",
]

SECONDS_PER_DAY = 86_400
MILLISECONDS_PER_DAY = 86_400_000
TRADING_DAYS_PER_YEAR = 252

# ---------------------------------------------------------------------------
# Data quality enums
# ---------------------------------------------------------------------------

class DataQualityFlag:
    CLEAN = 0
    STALE_PRICE = 1 << 0
    OUTLIER_PRICE = 1 << 1
    GAP_DETECTED = 1 << 2
    ZERO_VOLUME = 1 << 3
    NEGATIVE_PRICE = 1 << 4
    CROSSED_MARKET = 1 << 5
    MISSING_DATA = 1 << 6
    CORPORATE_ACTION = 1 << 7


# ---------------------------------------------------------------------------
# Market calendar
# ---------------------------------------------------------------------------

class MarketCalendar:
    """
    Wraps pandas_market_calendars (or falls back to simple NYSE approximation).
    Provides trading day enumeration and session time lookup.
    """

    _KNOWN_EXCHANGES = {
        "NYSE": {"open": datetime.time(9, 30), "close": datetime.time(16, 0)},
        "NASDAQ": {"open": datetime.time(9, 30), "close": datetime.time(16, 0)},
        "LSE": {"open": datetime.time(8, 0), "close": datetime.time(16, 30)},
        "TSE": {"open": datetime.time(9, 0), "close": datetime.time(15, 30)},
        "HKEX": {"open": datetime.time(9, 30), "close": datetime.time(16, 0)},
        "EUREX": {"open": datetime.time(8, 0), "close": datetime.time(22, 0)},
        "CME": {"open": datetime.time(17, 0), "close": datetime.time(16, 0)},  # next day
        "CRYPTO": {"open": datetime.time(0, 0), "close": datetime.time(23, 59)},
    }

    def __init__(self, exchange: str = "NYSE"):
        self.exchange = exchange.upper()
        self._holidays: Set[datetime.date] = set()
        self._schedule: Optional[pd.DataFrame] = None
        self._load_calendar()

    def _load_calendar(self) -> None:
        if _CALENDAR_AVAILABLE:
            try:
                cal = mcal.get_calendar(self.exchange)
                start = "2000-01-01"
                end = "2030-12-31"
                schedule = cal.schedule(start_date=start, end_date=end)
                self._schedule = schedule
                return
            except Exception as e:
                logger.warning(f"Could not load mcal calendar for {self.exchange}: {e}")
        # Fallback: US federal holidays approximation
        self._holidays = self._generate_us_holidays(2000, 2030)

    def _generate_us_holidays(
        self, start_year: int, end_year: int
    ) -> Set[datetime.date]:
        """Approximate US market holidays."""
        holidays: Set[datetime.date] = set()
        for year in range(start_year, end_year + 1):
            # New Year's Day
            holidays.add(datetime.date(year, 1, 1))
            # MLK Day (3rd Monday in January)
            holidays.add(self._nth_weekday(year, 1, 0, 3))
            # Presidents' Day (3rd Monday in February)
            holidays.add(self._nth_weekday(year, 2, 0, 3))
            # Memorial Day (last Monday in May)
            holidays.add(self._last_weekday(year, 5, 0))
            # Independence Day
            holidays.add(datetime.date(year, 7, 4))
            # Labor Day (1st Monday in September)
            holidays.add(self._nth_weekday(year, 9, 0, 1))
            # Thanksgiving (4th Thursday in November)
            holidays.add(self._nth_weekday(year, 11, 3, 4))
            # Christmas
            holidays.add(datetime.date(year, 12, 25))
        return holidays

    @staticmethod
    def _nth_weekday(year: int, month: int, weekday: int, n: int) -> datetime.date:
        first = datetime.date(year, month, 1)
        delta = (weekday - first.weekday()) % 7
        return first + datetime.timedelta(days=delta + 7 * (n - 1))

    @staticmethod
    def _last_weekday(year: int, month: int, weekday: int) -> datetime.date:
        if month == 12:
            last = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
        else:
            last = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
        delta = (last.weekday() - weekday) % 7
        return last - datetime.timedelta(days=delta)

    def is_trading_day(self, date: datetime.date) -> bool:
        if self._schedule is not None:
            date_str = str(date)
            return date_str in self._schedule.index.astype(str)
        # Fallback
        if date.weekday() >= 5:  # weekend
            return False
        return date not in self._holidays

    def trading_days(
        self,
        start: Union[str, datetime.date],
        end: Union[str, datetime.date],
    ) -> List[datetime.date]:
        if isinstance(start, str):
            start = datetime.date.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.date.fromisoformat(end)
        days = []
        current = start
        while current <= end:
            if self.is_trading_day(current):
                days.append(current)
            current += datetime.timedelta(days=1)
        return days

    def session_times(self, date: Optional[datetime.date] = None) -> Dict[str, datetime.time]:
        if self._schedule is not None and date is not None:
            date_str = str(date)
            if date_str in self._schedule.index.astype(str):
                row = self._schedule.loc[self._schedule.index.astype(str) == date_str].iloc[0]
                return {
                    "open": row["market_open"].time() if hasattr(row["market_open"], "time") else row["market_open"],
                    "close": row["market_close"].time() if hasattr(row["market_close"], "time") else row["market_close"],
                }
        fallback = self._KNOWN_EXCHANGES.get(self.exchange, {"open": datetime.time(9, 30), "close": datetime.time(16, 0)})
        return fallback

    def num_trading_days(self, start: str, end: str) -> int:
        return len(self.trading_days(start, end))


# ---------------------------------------------------------------------------
# Corporate action adjustment
# ---------------------------------------------------------------------------

@dataclass
class CorporateAction:
    """Represents a single corporate action event."""
    symbol: str
    date: datetime.date
    action_type: str          # "split", "dividend", "rights", "merger"
    factor: float             # Split factor (e.g., 2.0 for 2-for-1) or dividend amount
    currency: str = "USD"

    def __post_init__(self):
        if self.action_type == "split" and self.factor <= 0:
            raise ValueError(f"Split factor must be positive, got {self.factor}")


class CorporateActionAdjuster:
    """
    Adjusts historical price/volume data for corporate actions (backward-adjusted).

    Backward adjustment ensures that the most recent data is in "current" units
    and historical data is scaled to be comparable.
    """

    def __init__(self, actions: Optional[List[CorporateAction]] = None):
        self._actions: Dict[str, List[CorporateAction]] = collections.defaultdict(list)
        if actions:
            for action in actions:
                self.add_action(action)

    def add_action(self, action: CorporateAction) -> None:
        self._actions[action.symbol].append(action)
        # Keep sorted by date ascending
        self._actions[action.symbol].sort(key=lambda a: a.date)

    def load_from_csv(self, path: Union[str, pathlib.Path]) -> None:
        """Load corporate actions from CSV with columns: symbol, date, action_type, factor."""
        df = pd.read_csv(path, parse_dates=["date"])
        for _, row in df.iterrows():
            action = CorporateAction(
                symbol=row["symbol"],
                date=row["date"].date(),
                action_type=row["action_type"],
                factor=float(row["factor"]),
            )
            self.add_action(action)

    def compute_adjustment_factor(
        self,
        symbol: str,
        as_of_date: datetime.date,
        target_date: datetime.date,
    ) -> float:
        """
        Compute cumulative price adjustment factor from target_date up to as_of_date.
        Multiplying historical prices by this factor gives backward-adjusted prices.
        """
        actions = self._actions.get(symbol, [])
        factor = 1.0
        for action in actions:
            if target_date <= action.date <= as_of_date:
                if action.action_type == "split":
                    factor /= action.factor
                elif action.action_type == "dividend":
                    # Cash dividend: adjust by (price - dividend) / price
                    # We approximate: factor *= (1 - dividend/price)
                    # In practice, price is needed; we store the raw factor
                    factor *= action.factor
        return factor

    def adjust_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        price_cols: Optional[List[str]] = None,
        volume_col: Optional[str] = "volume",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        Apply backward adjustment to a DataFrame of OHLCV data.
        The latest price is kept as-is; historical prices are scaled.
        """
        df = df.copy()
        if price_cols is None:
            price_cols = [c for c in OHLCV_COLS if c in df.columns and c != "volume"]

        actions = self._actions.get(symbol, [])
        if not actions:
            return df

        df[date_col] = pd.to_datetime(df[date_col]).dt.date
        df = df.sort_values(date_col).reset_index(drop=True)

        # Cumulative adjustment factor per row (backward)
        factors = np.ones(len(df))
        for i, row_date in enumerate(df[date_col]):
            for action in reversed(actions):
                if action.date > row_date:
                    if action.action_type == "split":
                        factors[i] /= action.factor
                    elif action.action_type == "dividend":
                        factors[i] *= action.factor

        for col in price_cols:
            if col in df.columns:
                df[col] = df[col] * factors

        if volume_col in df.columns:
            # Volume adjusts inversely to price for splits
            vol_factors = np.ones(len(df))
            for i, row_date in enumerate(df[date_col]):
                for action in reversed(actions):
                    if action.date > row_date and action.action_type == "split":
                        vol_factors[i] *= action.factor
            df[volume_col] = df[volume_col] * vol_factors

        return df


# ---------------------------------------------------------------------------
# Survivorship bias handler
# ---------------------------------------------------------------------------

class SurvivorshipBiasHandler:
    """
    Manages universe snapshots to avoid survivorship bias in backtests.

    At each historical date, we know which symbols were in the universe
    (including those that later went bankrupt or were delisted).
    """

    def __init__(self, snapshot_path: Optional[Union[str, pathlib.Path]] = None):
        # Dict: date -> set of symbols
        self._snapshots: Dict[datetime.date, Set[str]] = {}
        if snapshot_path:
            self.load_snapshots(snapshot_path)

    def load_snapshots(self, path: Union[str, pathlib.Path]) -> None:
        """
        Load universe snapshots from a JSON file.
        Format: {"2020-01-01": ["AAPL", "MSFT", ...], ...}
        """
        path = pathlib.Path(path)
        if not path.exists():
            logger.warning(f"Snapshot file not found: {path}")
            return
        with open(path) as f:
            data = json.load(f)
        for date_str, symbols in data.items():
            date = datetime.date.fromisoformat(date_str)
            self._snapshots[date] = set(symbols)
        logger.info(f"Loaded {len(self._snapshots)} universe snapshots from {path}")

    def add_snapshot(self, date: datetime.date, symbols: Iterable[str]) -> None:
        self._snapshots[date] = set(symbols)

    def get_universe(self, as_of_date: datetime.date) -> Set[str]:
        """Return the universe as of a given date (latest snapshot on or before date)."""
        if not self._snapshots:
            logger.warning("No universe snapshots loaded; returning empty universe.")
            return set()
        dates = sorted(self._snapshots.keys())
        idx = bisect.bisect_right(dates, as_of_date) - 1
        if idx < 0:
            return set()
        return self._snapshots[dates[idx]]

    def filter_dataframe(
        self,
        df: pd.DataFrame,
        symbol_col: str = "symbol",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """Filter DataFrame to include only symbols present in universe at each date."""
        if not self._snapshots:
            return df
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col]).dt.date

        def is_in_universe(row) -> bool:
            universe = self.get_universe(row[date_col])
            return row[symbol_col] in universe

        mask = df.apply(is_in_universe, axis=1)
        return df[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Data quality checker
# ---------------------------------------------------------------------------

class DataQualityChecker:
    """
    Comprehensive data quality checks for financial time series.
    Returns a DataFrame with quality flags per row.
    """

    def __init__(
        self,
        max_price_change_pct: float = 0.5,    # 50% single-period change flagged
        max_stale_periods: int = 5,             # 5 consecutive unchanged prices
        min_volume: float = 0.0,
        outlier_sigma: float = 6.0,             # Z-score threshold for outliers
        max_spread_pct: float = 0.1,            # 10% bid-ask spread
    ):
        self.max_price_change_pct = max_price_change_pct
        self.max_stale_periods = max_stale_periods
        self.min_volume = min_volume
        self.outlier_sigma = outlier_sigma
        self.max_spread_pct = max_spread_pct

    def check(
        self,
        df: pd.DataFrame,
        price_col: str = "close",
        volume_col: Optional[str] = "volume",
        bid_col: Optional[str] = None,
        ask_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Run all quality checks. Returns df with added 'quality_flag' column.
        """
        df = df.copy()
        flags = np.zeros(len(df), dtype=np.int32)

        prices = df[price_col].values.astype(float)

        # Negative prices
        flags[prices < 0] |= DataQualityFlag.NEGATIVE_PRICE

        # Missing data
        nan_mask = np.isnan(prices)
        flags[nan_mask] |= DataQualityFlag.MISSING_DATA

        # Outlier detection via rolling z-score
        if len(prices) > 30:
            returns = np.diff(prices, prepend=prices[0]) / (np.abs(prices) + 1e-10)
            roll_mean = pd.Series(returns).rolling(30, min_periods=10).mean().values
            roll_std = pd.Series(returns).rolling(30, min_periods=10).std().values
            with np.errstate(invalid="ignore", divide="ignore"):
                z_scores = np.abs((returns - roll_mean) / (roll_std + 1e-10))
            outlier_mask = z_scores > self.outlier_sigma
            flags[outlier_mask] |= DataQualityFlag.OUTLIER_PRICE

        # Large price jumps
        returns_abs = np.abs(np.diff(prices, prepend=prices[0]) / (np.abs(prices) + 1e-10))
        jump_mask = returns_abs > self.max_price_change_pct
        flags[jump_mask] |= DataQualityFlag.OUTLIER_PRICE

        # Stale prices (consecutive unchanged values)
        stale_count = 0
        for i in range(1, len(prices)):
            if prices[i] == prices[i - 1]:
                stale_count += 1
                if stale_count >= self.max_stale_periods:
                    flags[i] |= DataQualityFlag.STALE_PRICE
            else:
                stale_count = 0

        # Zero volume
        if volume_col is not None and volume_col in df.columns:
            volumes = df[volume_col].values.astype(float)
            zero_vol_mask = volumes <= self.min_volume
            flags[zero_vol_mask] |= DataQualityFlag.ZERO_VOLUME

        # Crossed market (bid > ask)
        if bid_col is not None and ask_col is not None:
            if bid_col in df.columns and ask_col in df.columns:
                bids = df[bid_col].values.astype(float)
                asks = df[ask_col].values.astype(float)
                crossed = bids > asks
                flags[crossed] |= DataQualityFlag.CROSSED_MARKET
                # Wide spreads
                mid = (bids + asks) / 2.0
                spread_pct = (asks - bids) / (mid + 1e-10)
                wide_spread = spread_pct > self.max_spread_pct
                flags[wide_spread] |= DataQualityFlag.CROSSED_MARKET

        # Gap detection (time gaps)
        if "timestamp" in df.columns or "datetime" in df.columns:
            time_col = "timestamp" if "timestamp" in df.columns else "datetime"
            times = pd.to_datetime(df[time_col]).values.astype(np.int64)
            if len(times) > 1:
                diffs = np.diff(times, prepend=times[0])
                median_diff = np.median(diffs[diffs > 0])
                if median_diff > 0:
                    gap_mask = diffs > 3 * median_diff
                    flags[gap_mask] |= DataQualityFlag.GAP_DETECTED

        df["quality_flag"] = flags
        df["is_clean"] = flags == DataQualityFlag.CLEAN
        return df

    def flag_summary(self, df: pd.DataFrame) -> Dict[str, int]:
        """Summarize quality flags."""
        if "quality_flag" not in df.columns:
            df = self.check(df)
        flags = df["quality_flag"].values
        summary = {
            "total": len(flags),
            "clean": int((flags == 0).sum()),
            "stale_price": int((flags & DataQualityFlag.STALE_PRICE).astype(bool).sum()),
            "outlier_price": int((flags & DataQualityFlag.OUTLIER_PRICE).astype(bool).sum()),
            "gap_detected": int((flags & DataQualityFlag.GAP_DETECTED).astype(bool).sum()),
            "zero_volume": int((flags & DataQualityFlag.ZERO_VOLUME).astype(bool).sum()),
            "negative_price": int((flags & DataQualityFlag.NEGATIVE_PRICE).astype(bool).sum()),
            "crossed_market": int((flags & DataQualityFlag.CROSSED_MARKET).astype(bool).sum()),
            "missing_data": int((flags & DataQualityFlag.MISSING_DATA).astype(bool).sum()),
        }
        return summary


# ---------------------------------------------------------------------------
# Parquet streaming reader
# ---------------------------------------------------------------------------

class ParquetStreamReader:
    """
    Streaming reader for large Parquet datasets.
    Reads row groups sequentially without loading the full file into memory.
    Supports predicate pushdown and column projection.
    """

    def __init__(
        self,
        path: Union[str, pathlib.Path],
        columns: Optional[List[str]] = None,
        filters: Optional[List[Tuple]] = None,
        batch_size: int = 100_000,
    ):
        if not _ARROW_AVAILABLE:
            raise ImportError("PyArrow required for ParquetStreamReader.")
        self.path = pathlib.Path(path)
        self.columns = columns
        self.filters = filters
        self.batch_size = batch_size
        self._pf: Optional[pq.ParquetFile] = None

    def __enter__(self):
        self._pf = pq.ParquetFile(self.path)
        return self

    def __exit__(self, *args):
        if self._pf is not None:
            self._pf.close()

    def __iter__(self) -> Iterator[pd.DataFrame]:
        if self._pf is None:
            with pq.ParquetFile(self.path) as pf:
                yield from self._iterate(pf)
        else:
            yield from self._iterate(self._pf)

    def _iterate(self, pf) -> Iterator[pd.DataFrame]:
        for batch in pf.iter_batches(
            batch_size=self.batch_size,
            columns=self.columns,
            # filters=self.filters,  # row-group level
        ):
            df = batch.to_pandas()
            if self.filters:
                df = self._apply_filters(df)
            yield df

    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.filters:
            return df
        mask = pd.Series([True] * len(df))
        for col, op, val in self.filters:
            if col not in df.columns:
                continue
            if op == "==":
                mask &= df[col] == val
            elif op == "!=":
                mask &= df[col] != val
            elif op == ">":
                mask &= df[col] > val
            elif op == ">=":
                mask &= df[col] >= val
            elif op == "<":
                mask &= df[col] < val
            elif op == "<=":
                mask &= df[col] <= val
            elif op == "in":
                mask &= df[col].isin(val)
        return df[mask].reset_index(drop=True)

    def metadata(self) -> Dict[str, Any]:
        if not _ARROW_AVAILABLE:
            return {}
        with pq.ParquetFile(self.path) as pf:
            meta = pf.metadata
            return {
                "num_rows": meta.num_rows,
                "num_row_groups": meta.num_row_groups,
                "num_columns": meta.num_columns,
                "serialized_size": meta.serialized_size,
            }


class ArrowDatasetReader:
    """
    Reads a directory of Parquet/Arrow files as a unified dataset.
    Supports partitioned datasets (e.g., partitioned by date/symbol).
    """

    def __init__(
        self,
        root: Union[str, pathlib.Path],
        format: str = "parquet",
        columns: Optional[List[str]] = None,
        filters: Optional[Any] = None,
        batch_size: int = 100_000,
    ):
        if not _ARROW_AVAILABLE:
            raise ImportError("PyArrow required.")
        self.root = pathlib.Path(root)
        self.format = format
        self.columns = columns
        self.filters = filters
        self.batch_size = batch_size

    def scan(self) -> Iterator[pd.DataFrame]:
        """Yield DataFrames from all files in the dataset."""
        dataset = pad.dataset(self.root, format=self.format)
        scanner = dataset.scanner(
            columns=self.columns,
            filter=self.filters,
            batch_size=self.batch_size,
        )
        for batch in scanner.to_batches():
            yield batch.to_pandas()

    def to_dataframe(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Read entire dataset into a single DataFrame."""
        dfs = []
        total = 0
        for df in self.scan():
            dfs.append(df)
            total += len(df)
            if max_rows is not None and total >= max_rows:
                break
        if not dfs:
            return pd.DataFrame()
        result = pd.concat(dfs, ignore_index=True)
        if max_rows is not None:
            result = result.iloc[:max_rows]
        return result


# ---------------------------------------------------------------------------
# Multi-asset tick data loader
# ---------------------------------------------------------------------------

@dataclass
class TickDataConfig:
    """Configuration for tick data loading."""
    data_root: str = "data/ticks"
    symbols: List[str] = field(default_factory=list)
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"
    exchange: str = "NYSE"
    frequency: str = "1min"       # "tick", "1min", "5min", "1h", "1d"
    include_lob: bool = False      # Include limit order book snapshots
    include_microstructure: bool = True
    adjust_corporate_actions: bool = True
    remove_survivorship_bias: bool = True
    quality_filter: bool = True
    min_quality_pct: float = 0.95  # Require 95% clean data per symbol
    cache_dir: Optional[str] = None


class MultiAssetTickLoader:
    """
    Loads and aligns tick/bar data across multiple assets.

    Handles:
      - Multiple file formats (Parquet, CSV, HDF5)
      - Temporal alignment across assets (forward-fill missing bars)
      - Corporate action adjustment
      - Quality filtering
      - Survivorship bias
      - Feature engineering
    """

    def __init__(
        self,
        config: TickDataConfig,
        adjuster: Optional[CorporateActionAdjuster] = None,
        survivorship: Optional[SurvivorshipBiasHandler] = None,
        quality_checker: Optional[DataQualityChecker] = None,
        calendar: Optional[MarketCalendar] = None,
    ):
        self.config = config
        self.adjuster = adjuster or CorporateActionAdjuster()
        self.survivorship = survivorship or SurvivorshipBiasHandler()
        self.quality_checker = quality_checker or DataQualityChecker()
        self.calendar = calendar or MarketCalendar(config.exchange)
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_symbol(
        self,
        symbol: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load data for a single symbol."""
        start = start or self.config.start_date
        end = end or self.config.end_date

        cache_key = f"{symbol}_{start}_{end}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        df = self._load_raw(symbol, start, end)
        if df is None or len(df) == 0:
            logger.warning(f"No data found for {symbol} [{start}, {end}]")
            return pd.DataFrame()

        # Corporate action adjustment
        if self.config.adjust_corporate_actions:
            df = self.adjuster.adjust_dataframe(df, symbol)

        # Quality check
        if self.config.quality_filter:
            df = self.quality_checker.check(df)
            clean_pct = df["is_clean"].mean()
            if clean_pct < self.config.min_quality_pct:
                logger.warning(
                    f"{symbol}: only {clean_pct:.1%} clean data; threshold {self.config.min_quality_pct:.1%}"
                )

        # Feature engineering
        df = self._add_features(df)

        if self.config.cache_dir:
            self._save_to_cache(cache_key, df)

        self._cache[cache_key] = df
        return df

    def _load_raw(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> Optional[pd.DataFrame]:
        """Load raw data from disk. Tries Parquet first, then CSV."""
        root = pathlib.Path(self.config.data_root)
        # Try various file patterns
        patterns = [
            root / symbol / f"{symbol}_{self.config.frequency}.parquet",
            root / f"{symbol}.parquet",
            root / symbol / f"{symbol}_{self.config.frequency}.csv",
            root / f"{symbol}.csv",
        ]
        for path in patterns:
            if path.exists():
                try:
                    if path.suffix == ".parquet" and _ARROW_AVAILABLE:
                        df = pq.read_table(
                            path,
                            filters=[("date", ">=", start), ("date", "<=", end)]
                            if "date" in pq.read_schema(path).names else None,
                        ).to_pandas()
                    elif path.suffix == ".csv":
                        df = pd.read_csv(path, parse_dates=True, index_col=0)
                    else:
                        continue

                    # Filter by date
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                        df = df[(df["date"] >= start) & (df["date"] <= end)]
                    elif "timestamp" in df.columns:
                        df["timestamp"] = pd.to_datetime(df["timestamp"])
                        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]

                    df["symbol"] = symbol
                    return df
                except Exception as e:
                    logger.debug(f"Failed to load {path}: {e}")
                    continue
        return None

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add standard financial features."""
        df = df.copy()
        if "close" not in df.columns:
            return df

        prices = df["close"].values.astype(float)

        # Log returns
        log_ret = np.diff(np.log(prices + 1e-10), prepend=np.nan)
        df["log_return"] = log_ret

        # Realized volatility (rolling)
        for window in [5, 10, 20, 60]:
            df[f"rvol_{window}"] = pd.Series(log_ret).rolling(window).std().values * math.sqrt(TRADING_DAYS_PER_YEAR)

        # Momentum features
        for lag in [1, 5, 10, 20]:
            df[f"mom_{lag}"] = pd.Series(prices).pct_change(lag).values

        # Volume features
        if "volume" in df.columns:
            vol = df["volume"].values.astype(float)
            df["volume_zscore"] = (vol - np.nanmean(vol)) / (np.nanstd(vol) + 1e-10)
            df["volume_ma_20"] = pd.Series(vol).rolling(20).mean().values
            df["volume_ratio"] = vol / (df["volume_ma_20"].values + 1e-10)

        # OHLC features
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            o, h, l, c = (df[col].values.astype(float) for col in ["open", "high", "low", "close"])
            df["bar_range"] = (h - l) / (c + 1e-10)
            df["bar_return"] = (c - o) / (o + 1e-10)
            df["upper_shadow"] = (h - np.maximum(o, c)) / (c + 1e-10)
            df["lower_shadow"] = (np.minimum(o, c) - l) / (c + 1e-10)

        return df

    def _save_to_cache(self, key: str, df: pd.DataFrame) -> None:
        if not self.config.cache_dir:
            return
        cache_dir = pathlib.Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        safe_key = hashlib.md5(key.encode()).hexdigest()
        if _ARROW_AVAILABLE:
            pq.write_table(pa.Table.from_pandas(df), cache_dir / f"{safe_key}.parquet")
        else:
            df.to_pickle(cache_dir / f"{safe_key}.pkl")

    def load_panel(self) -> Dict[str, pd.DataFrame]:
        """Load all symbols and return dict of symbol -> DataFrame."""
        panel = {}
        for symbol in self.config.symbols:
            df = self.load_symbol(symbol)
            if len(df) > 0:
                panel[symbol] = df
        return panel

    def to_aligned_panel(
        self,
        freq: str = "1D",
        fill_method: str = "ffill",
    ) -> pd.DataFrame:
        """
        Load all symbols and align them on a common time index.
        Returns a MultiIndex DataFrame or stacked DataFrame.
        """
        panel = self.load_panel()
        if not panel:
            return pd.DataFrame()

        aligned = {}
        for symbol, df in panel.items():
            if "date" in df.columns:
                df = df.set_index("date")
            elif "timestamp" in df.columns:
                df = df.set_index("timestamp")
            df.index = pd.DatetimeIndex(df.index)
            df = df.resample(freq).last()
            aligned[symbol] = df

        # Common index
        all_indices = [df.index for df in aligned.values()]
        common_idx = all_indices[0]
        for idx in all_indices[1:]:
            common_idx = common_idx.union(idx)

        result_dfs = []
        for symbol, df in aligned.items():
            df = df.reindex(common_idx)
            if fill_method == "ffill":
                df = df.fillna(method="ffill")
            df["symbol"] = symbol
            result_dfs.append(df)

        return pd.concat(result_dfs).sort_index()


# ---------------------------------------------------------------------------
# LOB (Limit Order Book) data integration
# ---------------------------------------------------------------------------

@dataclass
class LOBSnapshot:
    """A single limit order book snapshot."""
    timestamp: int           # Nanoseconds since epoch
    symbol: str
    bid_prices: np.ndarray   # Shape (depth,)
    bid_sizes: np.ndarray    # Shape (depth,)
    ask_prices: np.ndarray   # Shape (depth,)
    ask_sizes: np.ndarray    # Shape (depth,)
    trade_price: Optional[float] = None
    trade_size: Optional[float] = None

    @property
    def mid_price(self) -> float:
        if len(self.bid_prices) > 0 and len(self.ask_prices) > 0:
            return (self.bid_prices[0] + self.ask_prices[0]) / 2.0
        return float("nan")

    @property
    def spread(self) -> float:
        if len(self.bid_prices) > 0 and len(self.ask_prices) > 0:
            return self.ask_prices[0] - self.bid_prices[0]
        return float("nan")

    @property
    def depth_imbalance(self) -> float:
        total_bid = self.bid_sizes.sum()
        total_ask = self.ask_sizes.sum()
        denom = total_bid + total_ask
        if denom == 0:
            return 0.0
        return (total_bid - total_ask) / denom

    def to_feature_vector(self, normalize: bool = True) -> np.ndarray:
        """Convert LOB snapshot to a flat feature vector."""
        depth = min(len(self.bid_prices), len(self.ask_prices))
        features = []
        for i in range(depth):
            features.extend([
                self.bid_prices[i],
                self.bid_sizes[i],
                self.ask_prices[i],
                self.ask_sizes[i],
            ])
        features.extend([self.mid_price, self.spread, self.depth_imbalance])
        arr = np.array(features, dtype=np.float32)
        if normalize and self.mid_price > 0:
            # Normalize prices by mid price
            arr[:depth * 4:4] /= self.mid_price
            arr[1:depth * 4:4] = np.log1p(arr[1:depth * 4:4])
            arr[3:depth * 4:4] = np.log1p(arr[3:depth * 4:4])
        return arr


class LOBDataLoader:
    """
    Loads and processes Limit Order Book data from Chronos-format files.

    Chronos LOB format (custom binary):
      Header: 8 bytes magic + 4 bytes version + 4 bytes depth + 8 bytes n_rows
      Rows: timestamp(8) + n_bid_levels*(price(8)+size(8)) + n_ask_levels*(price(8)+size(8))
    """

    MAGIC = b"LOBCHRON"
    VERSION = 1

    def __init__(
        self,
        depth: int = 10,
        normalize: bool = True,
    ):
        self.depth = depth
        self.normalize = normalize
        self._row_size = 8 + (depth * 2 * 8) * 2  # timestamp + bids + asks

    def write_binary(
        self,
        path: Union[str, pathlib.Path],
        snapshots: List[LOBSnapshot],
    ) -> None:
        """Write LOB snapshots to custom binary format."""
        with open(path, "wb") as f:
            # Header
            f.write(self.MAGIC)
            f.write(struct.pack(">I", self.VERSION))
            f.write(struct.pack(">I", self.depth))
            f.write(struct.pack(">Q", len(snapshots)))
            # Rows
            for snap in snapshots:
                f.write(struct.pack(">Q", snap.timestamp))
                for i in range(self.depth):
                    bp = snap.bid_prices[i] if i < len(snap.bid_prices) else 0.0
                    bs = snap.bid_sizes[i] if i < len(snap.bid_sizes) else 0.0
                    f.write(struct.pack(">dd", bp, bs))
                for i in range(self.depth):
                    ap = snap.ask_prices[i] if i < len(snap.ask_prices) else 0.0
                    as_ = snap.ask_sizes[i] if i < len(snap.ask_sizes) else 0.0
                    f.write(struct.pack(">dd", ap, as_))

    def read_binary(self, path: Union[str, pathlib.Path]) -> Iterator[LOBSnapshot]:
        """Stream LOB snapshots from binary file."""
        with open(path, "rb") as f:
            magic = f.read(8)
            if magic != self.MAGIC:
                raise ValueError(f"Invalid magic bytes: {magic!r}")
            version = struct.unpack(">I", f.read(4))[0]
            depth = struct.unpack(">I", f.read(4))[0]
            n_rows = struct.unpack(">Q", f.read(8))[0]
            symbol = pathlib.Path(path).stem

            for _ in range(n_rows):
                ts = struct.unpack(">Q", f.read(8))[0]
                bid_prices = np.empty(depth)
                bid_sizes = np.empty(depth)
                ask_prices = np.empty(depth)
                ask_sizes = np.empty(depth)
                for i in range(depth):
                    bp, bs = struct.unpack(">dd", f.read(16))
                    bid_prices[i] = bp
                    bid_sizes[i] = bs
                for i in range(depth):
                    ap, as_ = struct.unpack(">dd", f.read(16))
                    ask_prices[i] = ap
                    ask_sizes[i] = as_
                yield LOBSnapshot(
                    timestamp=ts,
                    symbol=symbol,
                    bid_prices=bid_prices,
                    bid_sizes=bid_sizes,
                    ask_prices=ask_prices,
                    ask_sizes=ask_sizes,
                )

    def to_tensor(self, snapshots: List[LOBSnapshot]) -> Tensor:
        """Convert list of LOB snapshots to a tensor of shape (T, feature_dim)."""
        features = [s.to_feature_vector(normalize=self.normalize) for s in snapshots]
        return torch.tensor(np.stack(features), dtype=torch.float32)

    def load_from_parquet(self, path: Union[str, pathlib.Path]) -> pd.DataFrame:
        """Load LOB data from Parquet format (alternative to binary)."""
        if not _ARROW_AVAILABLE:
            return pd.read_parquet(path)
        return pq.read_table(path).to_pandas()


# ---------------------------------------------------------------------------
# Memory-mapped dataset
# ---------------------------------------------------------------------------

class MemoryMappedFinancialDataset(Dataset):
    """
    Memory-mapped dataset for huge financial time-series corpora.
    Data is stored as a flat binary file (float32 numpy array).
    The index file maps (sequence_id -> byte_offset, length).

    File format:
      data.bin: flat float32 array of shape (total_timesteps, feature_dim)
      index.npy: int64 array of shape (n_sequences, 2) = [(offset, length), ...]
    """

    def __init__(
        self,
        data_path: Union[str, pathlib.Path],
        index_path: Union[str, pathlib.Path],
        seq_len: int,
        feature_dim: int,
        stride: int = 1,
        dtype: np.dtype = np.float32,
        transform: Optional[Callable] = None,
    ):
        self.data_path = pathlib.Path(data_path)
        self.index_path = pathlib.Path(index_path)
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.stride = stride
        self.dtype = dtype
        self.transform = transform

        self._mmap: Optional[np.memmap] = None
        self._index: Optional[np.ndarray] = None
        self._load()

    def _load(self) -> None:
        self._mmap = np.memmap(
            self.data_path,
            dtype=self.dtype,
            mode="r",
        )
        total_samples = len(self._mmap) // self.feature_dim
        self._mmap = self._mmap.reshape(total_samples, self.feature_dim)

        if self.index_path.exists():
            self._index = np.load(self.index_path)
        else:
            # Auto-generate index: sliding window
            n_windows = (total_samples - self.seq_len) // self.stride + 1
            self._index = np.array(
                [[i * self.stride, self.seq_len] for i in range(n_windows)],
                dtype=np.int64,
            )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        offset, length = self._index[idx]
        data = self._mmap[offset: offset + length].copy()
        tensor = torch.from_numpy(data)

        if self.transform is not None:
            tensor = self.transform(tensor)

        # Last timestep is the target (predict next value)
        x = tensor[:-1]
        y = tensor[1:, 0]  # Predict first feature (e.g., close price return)

        return {"input_ids": x, "labels": y}

    @classmethod
    def create(
        cls,
        dataframes: List[pd.DataFrame],
        data_path: Union[str, pathlib.Path],
        index_path: Union[str, pathlib.Path],
        feature_cols: List[str],
        seq_len: int,
        stride: int = 1,
        dtype: np.dtype = np.float32,
    ) -> "MemoryMappedFinancialDataset":
        """Create memory-mapped dataset from list of DataFrames."""
        data_path = pathlib.Path(data_path)
        index_path = pathlib.Path(index_path)

        all_data = []
        seq_boundaries = []
        offset = 0

        for df in dataframes:
            arr = df[feature_cols].values.astype(dtype)
            n = len(arr)
            # Generate windows
            for start in range(0, n - seq_len, stride):
                seq_boundaries.append([offset + start, seq_len])
            all_data.append(arr)
            offset += n

        flat_data = np.concatenate(all_data, axis=0)
        feature_dim = flat_data.shape[1]

        mm = np.memmap(data_path, dtype=dtype, mode="w+", shape=flat_data.shape)
        mm[:] = flat_data
        mm.flush()
        del mm

        index_arr = np.array(seq_boundaries, dtype=np.int64)
        np.save(index_path, index_arr)

        return cls(data_path, index_path, seq_len, feature_dim, stride, dtype)


# ---------------------------------------------------------------------------
# Streaming iterable dataset with prefetching
# ---------------------------------------------------------------------------

class StreamingFinancialDataset(IterableDataset):
    """
    Iterable dataset that streams data from multiple Parquet files.
    Uses a background thread to prefetch batches.

    Suitable for datasets too large to fit in RAM.
    """

    def __init__(
        self,
        file_paths: List[Union[str, pathlib.Path]],
        seq_len: int,
        feature_cols: List[str],
        label_col: str = "log_return",
        batch_size: int = 256,
        prefetch_batches: int = 8,
        shuffle_files: bool = True,
        transform: Optional[Callable] = None,
        quality_checker: Optional[DataQualityChecker] = None,
    ):
        self.file_paths = [pathlib.Path(p) for p in file_paths]
        self.seq_len = seq_len
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.batch_size = batch_size
        self.prefetch_batches = prefetch_batches
        self.shuffle_files = shuffle_files
        self.transform = transform
        self.quality_checker = quality_checker or DataQualityChecker()

    def _file_generator(self) -> Iterator[pd.DataFrame]:
        paths = list(self.file_paths)
        if self.shuffle_files:
            random.shuffle(paths)
        for path in paths:
            try:
                if _ARROW_AVAILABLE and path.suffix == ".parquet":
                    df = pq.read_table(path).to_pandas()
                elif path.suffix == ".csv":
                    df = pd.read_csv(path)
                else:
                    continue
                # Quality filter
                df = self.quality_checker.check(df)
                df = df[df["is_clean"]].reset_index(drop=True)
                yield df
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

    def _window_generator(self, df: pd.DataFrame) -> Iterator[Dict[str, Tensor]]:
        """Generate sliding window sequences from a DataFrame."""
        if len(df) <= self.seq_len:
            return
        available_cols = [c for c in self.feature_cols if c in df.columns]
        features = df[available_cols].values.astype(np.float32)
        labels = df[self.label_col].values.astype(np.float32) if self.label_col in df.columns else None

        for i in range(len(df) - self.seq_len):
            x = features[i: i + self.seq_len]
            x_tensor = torch.from_numpy(x)
            if self.transform is not None:
                x_tensor = self.transform(x_tensor)
            item: Dict[str, Tensor] = {"input_ids": x_tensor}
            if labels is not None:
                item["labels"] = torch.tensor(labels[i + self.seq_len], dtype=torch.float32)
            yield item

    def __iter__(self) -> Iterator[Dict[str, Tensor]]:
        for df in self._file_generator():
            yield from self._window_generator(df)


class PrefetchingDataLoader:
    """
    Wrapper around DataLoader that prefetches batches to GPU asynchronously.
    Reduces CPU-GPU transfer latency.
    """

    def __init__(
        self,
        loader: DataLoader,
        device: torch.device,
        n_prefetch: int = 2,
    ):
        self.loader = loader
        self.device = device
        self.n_prefetch = n_prefetch

    def __iter__(self) -> Iterator[Dict[str, Tensor]]:
        q: queue.Queue = queue.Queue(maxsize=self.n_prefetch)
        _sentinel = object()

        def producer():
            for batch in self.loader:
                q.put(batch)
            q.put(_sentinel)

        t = threading.Thread(target=producer, daemon=True)
        t.start()

        while True:
            item = q.get()
            if item is _sentinel:
                break
            # Move to device
            if isinstance(item, dict):
                yield {
                    k: v.to(self.device, non_blocking=True) if isinstance(v, Tensor) else v
                    for k, v in item.items()
                }
            else:
                yield item.to(self.device, non_blocking=True)

    def __len__(self) -> int:
        return len(self.loader)


# ---------------------------------------------------------------------------
# Normalization utilities
# ---------------------------------------------------------------------------

class OnlineNormalizer:
    """
    Online/incremental normalizer using Welford's algorithm.
    Computes running mean and variance without storing all data.
    """

    def __init__(self, n_features: int):
        self.n_features = n_features
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2 = np.zeros(n_features, dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Update statistics with a new batch of data (shape: [T, F])."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        for row in x:
            self.n += 1
            delta = row - self.mean
            self.mean += delta / self.n
            delta2 = row - self.mean
            self.M2 += delta * delta2

    @property
    def var(self) -> np.ndarray:
        if self.n < 2:
            return np.ones(self.n_features)
        return self.M2 / (self.n - 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def state_dict(self) -> Dict[str, Any]:
        return {"n": self.n, "mean": self.mean.tolist(), "M2": self.M2.tolist()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.n = state["n"]
        self.mean = np.array(state["mean"])
        self.M2 = np.array(state["M2"])


class RobustNormalizer:
    """
    Robust normalization using median and IQR.
    Less sensitive to outliers than z-score normalization.
    """

    def __init__(self, quantile_range: Tuple[float, float] = (25.0, 75.0)):
        self.quantile_range = quantile_range
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "RobustNormalizer":
        q_lo, q_hi = self.quantile_range
        self.center_ = np.nanmedian(X, axis=0)
        q_lo_val = np.nanpercentile(X, q_lo, axis=0)
        q_hi_val = np.nanpercentile(X, q_hi, axis=0)
        self.scale_ = q_hi_val - q_lo_val
        self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.center_ is None:
            raise RuntimeError("Call fit() first.")
        return (X - self.center_) / self.scale_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return X * self.scale_ + self.center_


class CrossSectionalNormalizer:
    """
    Cross-sectional (rank) normalization for multi-asset panels.
    Normalizes each feature across assets at each timestep.
    """

    def __init__(self, method: str = "rank"):
        assert method in ("rank", "zscore", "minmax")
        self.method = method

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        X: shape (T, N, F) where T=time, N=assets, F=features
        Returns normalized array of same shape.
        """
        if X.ndim == 2:
            X = X[:, np.newaxis, :]  # (T, 1, F)
            squeeze = True
        else:
            squeeze = False

        T, N, F = X.shape
        result = np.zeros_like(X)

        for t in range(T):
            for f in range(F):
                vals = X[t, :, f]
                valid = ~np.isnan(vals)
                if valid.sum() < 2:
                    result[t, :, f] = vals
                    continue
                if self.method == "rank":
                    ranks = pd.Series(vals[valid]).rank(pct=True).values
                    result[t, valid, f] = 2 * ranks - 1  # [-1, 1]
                    result[t, ~valid, f] = 0.0
                elif self.method == "zscore":
                    m = np.nanmean(vals)
                    s = np.nanstd(vals)
                    result[t, :, f] = (vals - m) / (s + 1e-10)
                elif self.method == "minmax":
                    mn = np.nanmin(vals)
                    mx = np.nanmax(vals)
                    result[t, :, f] = (vals - mn) / (mx - mn + 1e-10)

        if squeeze:
            result = result[:, 0, :]
        return result


# ---------------------------------------------------------------------------
# PyTorch Datasets
# ---------------------------------------------------------------------------

class OHLCVDataset(Dataset):
    """
    PyTorch Dataset for OHLCV bar data.
    Returns sequences of length seq_len with target being next-period return.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        seq_len: int = 64,
        feature_cols: Optional[List[str]] = None,
        label_col: str = "log_return",
        stride: int = 1,
        normalize: bool = True,
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.label_col = label_col

        if feature_cols is None:
            feature_cols = [c for c in df.columns
                           if c not in ["date", "timestamp", "symbol", "quality_flag", "is_clean"]]
        self.feature_cols = [c for c in feature_cols if c in df.columns]

        features = df[self.feature_cols].values.astype(np.float32)
        labels = df[self.label_col].values.astype(np.float32) if label_col in df.columns else None

        # Handle NaN
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if normalize:
            norm = RobustNormalizer()
            features = norm.fit_transform(features)

        # Build index
        n = len(features)
        indices = list(range(0, n - seq_len, stride))
        self.features = features
        self.labels = labels
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        start = self.indices[idx]
        x = torch.from_numpy(self.features[start: start + self.seq_len])
        item: Dict[str, Tensor] = {"input_ids": x}
        if self.labels is not None:
            item["labels"] = torch.tensor(
                self.labels[start + self.seq_len - 1], dtype=torch.float32
            )
        return item


class MultiAssetDataset(Dataset):
    """
    Dataset for multi-asset time series, supporting cross-asset attention.
    Returns a tensor of shape (seq_len, n_assets, feature_dim).
    """

    def __init__(
        self,
        panel: Dict[str, pd.DataFrame],
        seq_len: int = 64,
        feature_cols: Optional[List[str]] = None,
        label_col: str = "log_return",
        stride: int = 1,
        normalizer: Optional[CrossSectionalNormalizer] = None,
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.label_col = label_col
        self.symbols = sorted(panel.keys())
        n_assets = len(self.symbols)

        # Align on common date index
        first_df = next(iter(panel.values()))
        date_col = "date" if "date" in first_df.columns else "timestamp"
        if feature_cols is None:
            feature_cols = [c for c in first_df.columns
                           if c not in [date_col, "symbol", "quality_flag", "is_clean"]]
        feature_cols = [c for c in feature_cols if c in first_df.columns]
        self.feature_cols = feature_cols
        n_feat = len(feature_cols)

        # Build common timeline
        all_dates = sorted(set().union(*[
            set(df[date_col].values) for df in panel.values()
        ]))
        n_dates = len(all_dates)
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        # Fill tensor: (n_dates, n_assets, n_features)
        data = np.zeros((n_dates, n_assets, n_feat), dtype=np.float32)
        label_arr = np.zeros((n_dates, n_assets), dtype=np.float32)

        for j, symbol in enumerate(self.symbols):
            df = panel[symbol]
            for _, row in df.iterrows():
                t_idx = date_to_idx.get(row[date_col])
                if t_idx is None:
                    continue
                for k, col in enumerate(feature_cols):
                    val = row.get(col, 0.0)
                    data[t_idx, j, k] = val if not np.isnan(val) else 0.0
                if label_col in df.columns:
                    label_arr[t_idx, j] = row.get(label_col, 0.0)

        if normalizer is not None:
            data = normalizer.transform(data)

        self.data = data
        self.labels = label_arr
        self.n_dates = n_dates
        self.indices = list(range(0, n_dates - seq_len, stride))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        start = self.indices[idx]
        x = torch.from_numpy(self.data[start: start + self.seq_len])
        y = torch.from_numpy(self.labels[start + self.seq_len - 1])
        return {"input_ids": x, "labels": y}


# ---------------------------------------------------------------------------
# Data pipeline builder
# ---------------------------------------------------------------------------

class FinancialDataPipeline:
    """
    High-level pipeline that chains all data loading, quality, and
    normalization steps into a unified interface.

    Usage:
        pipeline = FinancialDataPipeline(config)
        train_loader, val_loader = pipeline.build_loaders()
    """

    def __init__(
        self,
        config: TickDataConfig,
        seq_len: int = 64,
        batch_size: int = 256,
        val_split: float = 0.1,
        num_workers: int = 4,
        seed: int = 42,
    ):
        self.config = config
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.seed = seed

        self.calendar = MarketCalendar(config.exchange)
        self.adjuster = CorporateActionAdjuster()
        self.survivorship = SurvivorshipBiasHandler()
        self.quality_checker = DataQualityChecker()
        self.normalizer = CrossSectionalNormalizer(method="rank")
        self.tick_loader = MultiAssetTickLoader(
            config,
            self.adjuster,
            self.survivorship,
            self.quality_checker,
            self.calendar,
        )
        self.online_norm = OnlineNormalizer(n_features=64)  # will resize on first data

    def build_loaders(
        self,
    ) -> Tuple[DataLoader, DataLoader]:
        """Build train and validation DataLoaders."""
        panel = self.tick_loader.load_panel()

        if not panel:
            raise RuntimeError("No data loaded. Check config.data_root and config.symbols.")

        # Create dataset
        dataset = MultiAssetDataset(
            panel,
            seq_len=self.seq_len,
            normalizer=self.normalizer,
        )

        # Train/val split
        n = len(dataset)
        n_val = int(n * self.val_split)
        n_train = n - n_val
        # Time-series split (no shuffle)
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n))

        from torch.utils.data import Subset
        train_ds = Subset(dataset, train_indices)
        val_ds = Subset(dataset, val_indices)

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        logger.info(
            f"Built data loaders: {n_train} train, {n_val} val sequences, "
            f"batch_size={self.batch_size}"
        )
        return train_loader, val_loader

    def build_streaming_loader(
        self,
        file_paths: List[str],
        feature_cols: List[str],
    ) -> DataLoader:
        """Build a streaming DataLoader for huge datasets."""
        dataset = StreamingFinancialDataset(
            file_paths=file_paths,
            seq_len=self.seq_len,
            feature_cols=feature_cols,
            quality_checker=self.quality_checker,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def build_memory_mapped_loader(
        self,
        data_path: str,
        index_path: str,
        feature_dim: int,
    ) -> DataLoader:
        """Build a DataLoader backed by memory-mapped data."""
        dataset = MemoryMappedFinancialDataset(
            data_path=data_path,
            index_path=index_path,
            seq_len=self.seq_len,
            feature_dim=feature_dim,
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )


# ---------------------------------------------------------------------------
# Feature engineering utilities
# ---------------------------------------------------------------------------

class FinancialFeatureEngineer:
    """
    Comprehensive financial feature engineering.
    Computes technical indicators, microstructure features, and macro features.
    """

    @staticmethod
    def add_technical_indicators(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        df = df.copy()
        p = df[price_col].values.astype(float)
        n = len(p)

        # Moving averages
        for window in [5, 10, 20, 50, 200]:
            df[f"sma_{window}"] = pd.Series(p).rolling(window).mean().values
            df[f"ema_{window}"] = pd.Series(p).ewm(span=window, adjust=False).mean().values

        # RSI
        returns = np.diff(p, prepend=p[0])
        gains = np.where(returns > 0, returns, 0.0)
        losses = np.where(returns < 0, -returns, 0.0)
        for window in [14, 28]:
            avg_gain = pd.Series(gains).rolling(window).mean().values
            avg_loss = pd.Series(losses).rolling(window).mean().values
            rs = avg_gain / (avg_loss + 1e-10)
            df[f"rsi_{window}"] = 100 - 100 / (1 + rs)

        # Bollinger Bands
        for window in [20]:
            ma = pd.Series(p).rolling(window).mean().values
            std = pd.Series(p).rolling(window).std().values
            df[f"bb_upper_{window}"] = ma + 2 * std
            df[f"bb_lower_{window}"] = ma - 2 * std
            df[f"bb_pct_{window}"] = (p - ma) / (2 * std + 1e-10)

        # ATR (Average True Range)
        if all(c in df.columns for c in ["high", "low"]):
            h = df["high"].values.astype(float)
            l = df["low"].values.astype(float)
            tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(p, 1)), np.abs(l - np.roll(p, 1))))
            for window in [14]:
                df[f"atr_{window}"] = pd.Series(tr).ewm(span=window, adjust=False).mean().values

        # MACD
        ema12 = pd.Series(p).ewm(span=12, adjust=False).mean().values
        ema26 = pd.Series(p).ewm(span=26, adjust=False).mean().values
        macd = ema12 - ema26
        signal = pd.Series(macd).ewm(span=9, adjust=False).mean().values
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = macd - signal

        return df

    @staticmethod
    def add_microstructure_features(
        df: pd.DataFrame,
        price_col: str = "close",
        volume_col: str = "volume",
    ) -> pd.DataFrame:
        df = df.copy()
        p = df[price_col].values.astype(float)

        if volume_col in df.columns:
            v = df[volume_col].values.astype(float)
            # VWAP
            pv = p * v
            cum_pv = np.cumsum(pv)
            cum_v = np.cumsum(v)
            df["vwap"] = cum_pv / (cum_v + 1e-10)
            df["vwap_dev"] = (p - df["vwap"].values) / (p + 1e-10)

            # Dollar volume
            df["dollar_volume"] = p * v
            df["dollar_volume_ma20"] = pd.Series(p * v).rolling(20).mean().values

            # Amihud illiquidity (|return| / dollar volume)
            returns_abs = np.abs(np.diff(np.log(p + 1e-10), prepend=0))
            df["amihud"] = returns_abs / (p * v + 1e-10)
            df["amihud_ma20"] = pd.Series(df["amihud"].values).rolling(20).mean().values

            # Kyle's lambda (price impact)
            signed_ret = np.diff(np.log(p + 1e-10), prepend=0)
            signed_vol = np.sign(signed_ret) * v
            df["kyle_lambda"] = signed_ret / (signed_vol + 1e-10)

        return df

    @staticmethod
    def add_regime_features(df: pd.DataFrame, price_col: str = "close") -> pd.DataFrame:
        """Add market regime indicators (trend, volatility regimes)."""
        df = df.copy()
        p = df[price_col].values.astype(float)
        log_ret = np.diff(np.log(p + 1e-10), prepend=0)

        # Volatility regime via rolling realized vol
        rvol = pd.Series(log_ret).rolling(20).std().values * math.sqrt(252)
        rvol_ma = pd.Series(rvol).rolling(60).mean().values

        df["rvol_20"] = rvol
        df["rvol_regime"] = (rvol > rvol_ma).astype(float)  # 1 = high vol, 0 = low vol

        # Trend regime via Hurst exponent approximation
        for window in [30, 60]:
            series = pd.Series(log_ret)
            rs_list = []
            for i in range(0, len(series) - window, window):
                chunk = series.iloc[i: i + window].values
                m = np.mean(chunk)
                dev = np.cumsum(chunk - m)
                r = dev.max() - dev.min()
                s = np.std(chunk, ddof=1)
                if s > 0:
                    rs_list.append(r / s)
            if rs_list:
                avg_rs = np.mean(rs_list)
                hurst = math.log(avg_rs) / math.log(window / 2) if window > 2 else 0.5
                df[f"hurst_{window}"] = hurst
            else:
                df[f"hurst_{window}"] = 0.5

        return df


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Config
    "TickDataConfig",
    # Data quality
    "DataQualityFlag",
    "DataQualityChecker",
    # Calendar
    "MarketCalendar",
    # Corporate actions
    "CorporateAction",
    "CorporateActionAdjuster",
    # Survivorship bias
    "SurvivorshipBiasHandler",
    # File readers
    "ParquetStreamReader",
    "ArrowDatasetReader",
    # Tick data
    "MultiAssetTickLoader",
    # LOB
    "LOBSnapshot",
    "LOBDataLoader",
    # Datasets
    "MemoryMappedFinancialDataset",
    "StreamingFinancialDataset",
    "OHLCVDataset",
    "MultiAssetDataset",
    # Normalizers
    "OnlineNormalizer",
    "RobustNormalizer",
    "CrossSectionalNormalizer",
    # Pipeline
    "FinancialDataPipeline",
    "PrefetchingDataLoader",
    # Feature engineering
    "FinancialFeatureEngineer",
]


# =============================================================================
# SECTION: Advanced Data Lake Management
# =============================================================================

import os
import json
import hashlib
import datetime
from typing import Optional, List, Dict, Tuple, Any, Iterator
import torch
import numpy as np


class DataCatalog:
    """Catalog for tracking all datasets in the financial data lake.

    Maintains a registry of dataset metadata: schema, provenance,
    quality stats, versioning, and access patterns.
    """

    def __init__(self, catalog_path: str = None):
        self.catalog_path = catalog_path or os.path.join(os.getcwd(), "catalog.json")
        self._entries: Dict[str, dict] = {}
        self._load()

    def _load(self):
        """Load catalog from disk if it exists."""
        if os.path.exists(self.catalog_path):
            try:
                with open(self.catalog_path, "r") as f:
                    self._entries = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._entries = {}

    def _save(self):
        """Persist catalog to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(self.catalog_path)), exist_ok=True)
        with open(self.catalog_path, "w") as f:
            json.dump(self._entries, f, indent=2, default=str)

    def register_dataset(
        self,
        name: str,
        path: str,
        schema: dict,
        description: str = "",
        tags: List[str] = None,
        owner: str = "lumina",
    ) -> dict:
        """Register a new dataset in the catalog."""
        entry = {
            "name": name,
            "path": path,
            "schema": schema,
            "description": description,
            "tags": tags or [],
            "owner": owner,
            "created_at": datetime.datetime.utcnow().isoformat(),
            "updated_at": datetime.datetime.utcnow().isoformat(),
            "version": 1,
            "quality_score": None,
            "n_records": None,
            "size_bytes": None,
        }
        self._entries[name] = entry
        self._save()
        return entry

    def get(self, name: str) -> Optional[dict]:
        """Get dataset entry by name."""
        return self._entries.get(name)

    def search(self, tags: List[str] = None, owner: str = None) -> List[dict]:
        """Search catalog by tags and/or owner."""
        results = list(self._entries.values())
        if tags:
            results = [e for e in results if any(t in e.get("tags", []) for t in tags)]
        if owner:
            results = [e for e in results if e.get("owner") == owner]
        return results

    def update_quality_stats(self, name: str, stats: dict):
        """Update quality statistics for a dataset."""
        if name in self._entries:
            self._entries[name]["quality_score"] = stats.get("quality_score")
            self._entries[name]["n_records"] = stats.get("n_records")
            self._entries[name]["size_bytes"] = stats.get("size_bytes")
            self._entries[name]["updated_at"] = datetime.datetime.utcnow().isoformat()
            self._save()

    def increment_version(self, name: str):
        """Increment dataset version after an update."""
        if name in self._entries:
            self._entries[name]["version"] += 1
            self._entries[name]["updated_at"] = datetime.datetime.utcnow().isoformat()
            self._save()

    def list_all(self) -> List[str]:
        """List all registered dataset names."""
        return list(self._entries.keys())

    def delete(self, name: str):
        """Remove a dataset from the catalog."""
        self._entries.pop(name, None)
        self._save()


class DataLineageTracker:
    """Track data transformation lineage for reproducibility and audit.

    Records:
    - Source datasets
    - Transformation operations applied
    - Output datasets
    - Timestamps and user attribution
    """

    def __init__(self):
        self._graph: Dict[str, dict] = {}

    def record_transform(
        self,
        output_name: str,
        input_names: List[str],
        operation: str,
        params: dict = None,
        user: str = "system",
    ):
        """Record a data transformation event."""
        node = {
            "output": output_name,
            "inputs": input_names,
            "operation": operation,
            "params": params or {},
            "user": user,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        }
        self._graph[output_name] = node

    def get_lineage(self, dataset_name: str, depth: int = 10) -> List[dict]:
        """Get full lineage chain for a dataset up to given depth."""
        lineage = []
        current = dataset_name

        for _ in range(depth):
            if current not in self._graph:
                break
            node = self._graph[current]
            lineage.append(node)
            if not node["inputs"]:
                break
            current = node["inputs"][0]  # Follow first input

        return lineage

    def get_descendants(self, dataset_name: str) -> List[str]:
        """Get all datasets derived from the given dataset."""
        descendants = []
        for name, node in self._graph.items():
            if dataset_name in node["inputs"]:
                descendants.append(name)
                descendants.extend(self.get_descendants(name))
        return list(set(descendants))

    def export_dot(self) -> str:
        """Export lineage graph as DOT format for visualization."""
        lines = ["digraph DataLineage {"]
        for name, node in self._graph.items():
            for inp in node["inputs"]:
                lines.append(f'  "{inp}" -> "{name}" [label="{node["operation"]}"];')
        lines.append("}")
        return "\n".join(lines)


class PartitionedDataStore:
    """Partitioned storage for time-series financial data.

    Organizes data by (symbol, date, frequency) with efficient
    range queries and lazy loading.

    Directory structure:
    root/
      {symbol}/
        {frequency}/
          {year}/
            {month}/
              data.pt
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        os.makedirs(root_dir, exist_ok=True)
        self._cache = {}
        self._cache_maxsize = 256

    def _partition_path(
        self,
        symbol: str,
        frequency: str,
        year: int,
        month: int,
    ) -> str:
        return os.path.join(
            self.root_dir, symbol, frequency,
            str(year), f"{month:02d}"
        )

    def write(
        self,
        symbol: str,
        frequency: str,
        year: int,
        month: int,
        data: torch.Tensor,
        metadata: dict = None,
    ):
        """Write a data partition to disk."""
        partition_dir = self._partition_path(symbol, frequency, year, month)
        os.makedirs(partition_dir, exist_ok=True)
        path = os.path.join(partition_dir, "data.pt")
        payload = {"data": data}
        if metadata:
            payload["metadata"] = metadata
        torch.save(payload, path)

    def read(
        self,
        symbol: str,
        frequency: str,
        year: int,
        month: int,
    ) -> Optional[Tuple[torch.Tensor, dict]]:
        """Read a data partition from disk."""
        path = os.path.join(
            self._partition_path(symbol, frequency, year, month), "data.pt"
        )

        if path in self._cache:
            return self._cache[path]

        if not os.path.exists(path):
            return None

        payload = torch.load(path, map_location="cpu")
        data = payload["data"]
        meta = payload.get("metadata", {})

        # LRU-style cache management
        if len(self._cache) >= self._cache_maxsize:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        self._cache[path] = (data, meta)

        return data, meta

    def read_range(
        self,
        symbol: str,
        frequency: str,
        start_year: int,
        start_month: int,
        end_year: int,
        end_month: int,
    ) -> List[Tuple[torch.Tensor, dict]]:
        """Read all partitions within a date range."""
        results = []
        y, m = start_year, start_month

        while (y, m) <= (end_year, end_month):
            result = self.read(symbol, frequency, y, m)
            if result is not None:
                results.append(result)

            m += 1
            if m > 12:
                m = 1
                y += 1

        return results

    def list_partitions(self, symbol: str, frequency: str) -> List[Tuple[int, int]]:
        """List available (year, month) partitions for a symbol."""
        base = os.path.join(self.root_dir, symbol, frequency)
        if not os.path.exists(base):
            return []

        partitions = []
        for year_dir in sorted(os.listdir(base)):
            try:
                year = int(year_dir)
                month_dir = os.path.join(base, year_dir)
                for month_str in sorted(os.listdir(month_dir)):
                    try:
                        month = int(month_str)
                        data_file = os.path.join(month_dir, month_str, "data.pt")
                        if os.path.exists(data_file):
                            partitions.append((year, month))
                    except (ValueError, NotADirectoryError):
                        pass
            except (ValueError, NotADirectoryError):
                pass

        return partitions


class DataVersionControl:
    """Version control for financial datasets.

    Tracks dataset snapshots, supports branching, and provides
    rollback capabilities for reproducible experiments.
    """

    def __init__(self, repo_dir: str):
        self.repo_dir = repo_dir
        self.versions_dir = os.path.join(repo_dir, ".versions")
        os.makedirs(self.versions_dir, exist_ok=True)
        self._history: List[dict] = self._load_history()

    def _history_path(self) -> str:
        return os.path.join(self.versions_dir, "history.json")

    def _load_history(self) -> List[dict]:
        if os.path.exists(self._history_path()):
            try:
                with open(self._history_path(), "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                pass
        return []

    def _save_history(self):
        with open(self._history_path(), "w") as f:
            json.dump(self._history, f, indent=2, default=str)

    def _compute_hash(self, data: torch.Tensor) -> str:
        """Compute content hash for a data tensor."""
        return hashlib.sha256(data.numpy().tobytes()).hexdigest()[:16]

    def commit(
        self,
        data: torch.Tensor,
        message: str,
        branch: str = "main",
        author: str = "lumina",
    ) -> str:
        """Commit a new version of dataset data."""
        content_hash = self._compute_hash(data)
        commit_id = f"{branch}_{len(self._history):06d}_{content_hash}"

        # Save snapshot
        snapshot_path = os.path.join(self.versions_dir, f"{commit_id}.pt")
        torch.save(data, snapshot_path)

        # Record commit
        commit = {
            "id": commit_id,
            "branch": branch,
            "message": message,
            "author": author,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "hash": content_hash,
            "snapshot_path": snapshot_path,
        }
        self._history.append(commit)
        self._save_history()

        return commit_id

    def checkout(self, commit_id: str) -> Optional[torch.Tensor]:
        """Load data from a specific commit."""
        for commit in self._history:
            if commit["id"] == commit_id:
                if os.path.exists(commit["snapshot_path"]):
                    return torch.load(commit["snapshot_path"], map_location="cpu")
        return None

    def log(self, branch: str = None, n: int = 10) -> List[dict]:
        """Show recent commit history."""
        commits = self._history
        if branch:
            commits = [c for c in commits if c["branch"] == branch]
        return commits[-n:]

    def diff(self, commit_id_a: str, commit_id_b: str) -> dict:
        """Compute statistics about differences between two commits."""
        a = self.checkout(commit_id_a)
        b = self.checkout(commit_id_b)

        if a is None or b is None:
            return {"error": "One or both commits not found"}

        return {
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
            "mean_diff": (b.float() - a.float()).mean().item() if a.shape == b.shape else None,
            "max_diff": (b.float() - a.float()).abs().max().item() if a.shape == b.shape else None,
        }


# =============================================================================
# SECTION: Real-Time Data Ingestion
# =============================================================================

class StreamingDataBuffer:
    """Thread-safe buffer for real-time streaming data.

    Accumulates incoming data points and provides batch-level access
    for online learning scenarios.
    """

    def __init__(
        self,
        feature_dim: int,
        max_size: int = 10000,
        dtype: torch.dtype = torch.float32,
    ):
        self.feature_dim = feature_dim
        self.max_size = max_size
        self.dtype = dtype
        self._buffer = torch.zeros(max_size, feature_dim, dtype=dtype)
        self._labels = torch.zeros(max_size, dtype=dtype)
        self._timestamps = []
        self._head = 0
        self._size = 0

        import threading
        self._lock = threading.Lock()

    def push(self, features: torch.Tensor, label: float = 0.0, timestamp=None):
        """Add one data point to the buffer (circular)."""
        with self._lock:
            idx = self._head % self.max_size
            self._buffer[idx] = features.detach()
            self._labels[idx] = label
            if len(self._timestamps) < self.max_size:
                self._timestamps.append(timestamp or datetime.datetime.utcnow())
            else:
                self._timestamps[idx] = timestamp or datetime.datetime.utcnow()
            self._head += 1
            self._size = min(self._size + 1, self.max_size)

    def get_batch(self, batch_size: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Get the most recent batch_size data points."""
        with self._lock:
            if self._size < batch_size:
                return None
            if self._head <= self.max_size:
                features = self._buffer[:self._size][-batch_size:]
                labels = self._labels[:self._size][-batch_size:]
            else:
                start = (self._head - batch_size) % self.max_size
                end = self._head % self.max_size
                if end > start:
                    features = self._buffer[start:end]
                    labels = self._labels[start:end]
                else:
                    features = torch.cat([self._buffer[start:], self._buffer[:end]])
                    labels = torch.cat([self._labels[start:], self._labels[:end]])
            return features.clone(), labels.clone()

    def current_size(self) -> int:
        with self._lock:
            return self._size

    def clear(self):
        with self._lock:
            self._head = 0
            self._size = 0


class TickDataProcessor:
    """Process raw tick data (trade-by-trade) into OHLCV bars.

    Supports:
    - Time-based bars (1min, 5min, 1h)
    - Volume bars (equal volume per bar)
    - Dollar bars (equal dollar volume per bar)
    - Tick bars (equal number of trades per bar)
    - Information-ratio bars (Advances Marcos Lopez de Prado)
    """

    def __init__(
        self,
        bar_type: str = "time",
        bar_size: float = 60.0,
    ):
        self.bar_type = bar_type
        self.bar_size = bar_size
        self._pending_trades = []
        self._current_bar_accum = 0.0
        self._bars = []

    def process_tick(self, price: float, volume: float, timestamp: float) -> Optional[dict]:
        """Process one tick and return a completed bar if threshold met."""
        self._pending_trades.append({"price": price, "volume": volume, "ts": timestamp})

        if self.bar_type == "volume":
            self._current_bar_accum += volume
        elif self.bar_type == "dollar":
            self._current_bar_accum += price * volume
        elif self.bar_type == "tick":
            self._current_bar_accum += 1
        elif self.bar_type == "time":
            if len(self._pending_trades) > 1:
                span = timestamp - self._pending_trades[0]["ts"]
                if span >= self.bar_size:
                    return self._finalize_bar()
            return None

        if self._current_bar_accum >= self.bar_size:
            return self._finalize_bar()

        return None

    def _finalize_bar(self) -> dict:
        """Compute OHLCV from pending trades."""
        trades = self._pending_trades
        prices = [t["price"] for t in trades]
        volumes = [t["volume"] for t in trades]

        bar = {
            "open": prices[0],
            "high": max(prices),
            "low": min(prices),
            "close": prices[-1],
            "volume": sum(volumes),
            "vwap": sum(p * v for p, v in zip(prices, volumes)) / max(sum(volumes), 1e-10),
            "n_trades": len(trades),
            "timestamp": trades[0]["ts"],
        }

        self._pending_trades = []
        self._current_bar_accum = 0.0
        self._bars.append(bar)
        return bar

    def get_bars_as_tensor(self) -> Optional[torch.Tensor]:
        """Convert completed bars to tensor format."""
        if not self._bars:
            return None
        keys = ["open", "high", "low", "close", "volume", "vwap", "n_trades"]
        rows = [[b[k] for k in keys] for b in self._bars]
        return torch.tensor(rows, dtype=torch.float32)


class CorpActionAdjuster:
    """Adjust price/volume data for corporate actions.

    Handles:
    - Stock splits and reverse splits
    - Dividends (cash and stock)
    - Spinoffs and mergers
    - Rights offerings

    Applies backward-adjusted prices using cumulative adjustment factors.
    """

    def __init__(self):
        self._actions: Dict[str, List[dict]] = {}

    def record_split(self, symbol: str, ex_date: str, ratio: float):
        """Record a stock split (ratio = new_shares / old_shares)."""
        self._actions.setdefault(symbol, []).append({
            "type": "split",
            "ex_date": ex_date,
            "factor": 1.0 / ratio,
            "volume_factor": ratio,
        })
        self._actions[symbol].sort(key=lambda x: x["ex_date"])

    def record_dividend(self, symbol: str, ex_date: str, amount: float, price_on_ex: float):
        """Record a cash dividend adjustment factor."""
        factor = 1.0 - amount / max(price_on_ex, 1e-10)
        self._actions.setdefault(symbol, []).append({
            "type": "dividend",
            "ex_date": ex_date,
            "factor": factor,
            "volume_factor": 1.0,
        })
        self._actions[symbol].sort(key=lambda x: x["ex_date"])

    def get_adjustment_factor(
        self,
        symbol: str,
        price_date: str,
        reference_date: str,
    ) -> Tuple[float, float]:
        """Compute cumulative price and volume adjustment factors.

        Returns (price_factor, volume_factor) to apply to prices/volumes
        at price_date to make them comparable to reference_date.
        """
        actions = self._actions.get(symbol, [])
        price_factor = 1.0
        volume_factor = 1.0

        for action in actions:
            if price_date <= action["ex_date"] <= reference_date:
                price_factor *= action["factor"]
                volume_factor *= action["volume_factor"]

        return price_factor, volume_factor

    def adjust_series(
        self,
        symbol: str,
        prices: np.ndarray,
        volumes: np.ndarray,
        dates: List[str],
        reference_date: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply adjustment factors to entire price/volume series."""
        adj_prices = prices.copy().astype(float)
        adj_volumes = volumes.copy().astype(float)

        for i, date in enumerate(dates):
            pf, vf = self.get_adjustment_factor(symbol, date, reference_date)
            adj_prices[i] *= pf
            adj_volumes[i] *= vf

        return adj_prices, adj_volumes


# =============================================================================
# SECTION: Feature Store
# =============================================================================

class FeatureStore:
    """Centralized feature store for financial ML features.

    Provides:
    - Feature materialization and caching
    - Point-in-time correct feature retrieval
    - Feature drift monitoring
    - Online/offline serving with consistent logic
    """

    def __init__(self, store_dir: str):
        self.store_dir = store_dir
        os.makedirs(store_dir, exist_ok=True)
        self._feature_defs: Dict[str, dict] = {}
        self._feature_cache: Dict[str, torch.Tensor] = {}

    def register_feature(
        self,
        name: str,
        compute_fn,
        inputs: List[str],
        dtype: torch.dtype = torch.float32,
        description: str = "",
    ):
        """Register a feature definition."""
        self._feature_defs[name] = {
            "name": name,
            "compute_fn": compute_fn,
            "inputs": inputs,
            "dtype": dtype,
            "description": description,
        }

    def materialize(self, feature_names: List[str], data: dict) -> Dict[str, torch.Tensor]:
        """Compute and cache feature values for given data."""
        results = {}

        for fname in feature_names:
            if fname in self._feature_cache:
                results[fname] = self._feature_cache[fname]
                continue

            if fname not in self._feature_defs:
                raise KeyError(f"Feature '{fname}' not registered")

            defn = self._feature_defs[fname]
            input_data = {k: data[k] for k in defn["inputs"] if k in data}
            result = defn["compute_fn"](**input_data)

            if not isinstance(result, torch.Tensor):
                result = torch.tensor(result, dtype=defn["dtype"])

            self._feature_cache[fname] = result
            results[fname] = result

        return results

    def save_feature(self, name: str, values: torch.Tensor, timestamp: str = None):
        """Save materialized feature values to disk."""
        ts = timestamp or datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.store_dir, f"{name}_{ts}.pt")
        torch.save({"feature": name, "values": values, "timestamp": ts}, path)

    def load_feature(self, name: str, timestamp: str = None) -> Optional[torch.Tensor]:
        """Load feature values from disk."""
        import glob

        if timestamp:
            path = os.path.join(self.store_dir, f"{name}_{timestamp}.pt")
        else:
            # Load most recent
            paths = sorted(glob.glob(os.path.join(self.store_dir, f"{name}_*.pt")))
            if not paths:
                return None
            path = paths[-1]

        if not os.path.exists(path):
            return None

        payload = torch.load(path, map_location="cpu")
        return payload["values"]

    def monitor_drift(
        self,
        feature_name: str,
        reference: torch.Tensor,
        current: torch.Tensor,
        method: str = "psi",
    ) -> dict:
        """Monitor feature distribution drift.

        Methods:
        - psi: Population Stability Index
        - ks: Kolmogorov-Smirnov test statistic
        - wasserstein: 1-Wasserstein distance
        """
        ref = reference.float().numpy()
        cur = current.float().numpy()

        if method == "psi":
            n_bins = 10
            ref_hist, bin_edges = np.histogram(ref, bins=n_bins)
            cur_hist, _ = np.histogram(cur, bins=bin_edges)

            ref_pct = ref_hist / (ref_hist.sum() + 1e-10)
            cur_pct = cur_hist / (cur_hist.sum() + 1e-10)

            # Avoid log(0)
            ref_pct = np.clip(ref_pct, 1e-6, None)
            cur_pct = np.clip(cur_pct, 1e-6, None)

            psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
            return {
                "method": "psi",
                "value": float(psi),
                "alert": psi > 0.2,
                "feature": feature_name,
            }

        elif method == "ks":
            from scipy import stats
            ks_stat, p_value = stats.ks_2samp(ref, cur)
            return {
                "method": "ks",
                "value": float(ks_stat),
                "p_value": float(p_value),
                "alert": p_value < 0.05,
                "feature": feature_name,
            }

        else:
            # Wasserstein distance
            sorted_ref = np.sort(ref)
            sorted_cur = np.sort(cur)
            n = min(len(sorted_ref), len(sorted_cur))
            dist = float(np.abs(sorted_ref[:n] - sorted_cur[:n]).mean())
            return {
                "method": "wasserstein",
                "value": dist,
                "feature": feature_name,
            }


# =============================================================================
# SECTION: Data Pipeline Orchestration
# =============================================================================

class DataPipelineStep:
    """A single step in a data pipeline DAG."""

    def __init__(
        self,
        name: str,
        fn,
        inputs: List[str],
        outputs: List[str],
        retries: int = 3,
        timeout_s: int = 300,
    ):
        self.name = name
        self.fn = fn
        self.inputs = inputs
        self.outputs = outputs
        self.retries = retries
        self.timeout_s = timeout_s
        self._status = "pending"
        self._error = None
        self._result = None

    def execute(self, context: dict) -> dict:
        """Execute this step with retry logic."""
        for attempt in range(self.retries):
            try:
                input_data = {k: context[k] for k in self.inputs if k in context}
                result = self.fn(**input_data)
                self._status = "completed"
                self._result = result
                if isinstance(result, dict):
                    return result
                elif len(self.outputs) == 1:
                    return {self.outputs[0]: result}
                return {}
            except Exception as e:
                self._error = str(e)
                if attempt == self.retries - 1:
                    self._status = "failed"
                    raise

        return {}

    @property
    def status(self) -> str:
        return self._status


class DataPipelineDAG:
    """DAG-based data pipeline for financial data processing.

    Steps execute in topological order. Failed steps are retried.
    Results are stored in a shared context dictionary.
    """

    def __init__(self, name: str):
        self.name = name
        self.steps: Dict[str, DataPipelineStep] = {}
        self._context: dict = {}

    def add_step(self, step: DataPipelineStep):
        """Add a step to the pipeline."""
        self.steps[step.name] = step

    def _topological_sort(self) -> List[str]:
        """Return steps in dependency order."""
        visited = set()
        order = []

        def dfs(name):
            if name in visited:
                return
            visited.add(name)
            step = self.steps[name]
            for inp in step.inputs:
                # Find steps that produce this input
                for other_name, other_step in self.steps.items():
                    if inp in other_step.outputs:
                        dfs(other_name)
            order.append(name)

        for name in self.steps:
            dfs(name)

        return order

    def run(self, initial_context: dict = None) -> dict:
        """Execute all pipeline steps in topological order."""
        self._context = dict(initial_context or {})
        order = self._topological_sort()

        results = {
            "pipeline": self.name,
            "steps": {},
            "final_context_keys": [],
        }

        for step_name in order:
            step = self.steps[step_name]
            try:
                step_results = step.execute(self._context)
                self._context.update(step_results)
                results["steps"][step_name] = {"status": "completed"}
            except Exception as e:
                results["steps"][step_name] = {"status": "failed", "error": str(e)}
                break

        results["final_context_keys"] = list(self._context.keys())
        return results

    def get_output(self, key: str):
        """Get a specific output from the pipeline context."""
        return self._context.get(key)


# =============================================================================
# SECTION: Temporal Join and Asof Merge
# =============================================================================

class TemporalJoiner:
    """Join datasets on timestamps using asof (last-value-carry-forward) semantics.

    Essential for point-in-time correct feature assembly in financial ML.
    """

    @staticmethod
    def asof_join(
        left_timestamps: List[float],
        left_values: np.ndarray,
        right_timestamps: List[float],
        right_values: np.ndarray,
        tolerance: float = None,
    ) -> np.ndarray:
        """For each left timestamp, find the most recent right value.

        Args:
            left_timestamps: sorted list of timestamps for query points
            left_values: [N_left, D] values for left dataset (unused, returned)
            right_timestamps: sorted list of timestamps for right dataset
            right_values: [N_right, D_right] values for right dataset
            tolerance: maximum allowable staleness (in same units as timestamps)

        Returns:
            [N_left, D_right] right values aligned to left timestamps
        """
        right_ts = np.array(right_timestamps)
        result = np.full((len(left_timestamps), right_values.shape[1]), np.nan)

        for i, ts in enumerate(left_timestamps):
            # Find rightmost right_timestamp <= ts
            idx = np.searchsorted(right_ts, ts, side="right") - 1
            if idx >= 0:
                if tolerance is None or (ts - right_ts[idx]) <= tolerance:
                    result[i] = right_values[idx]

        return result

    @staticmethod
    def point_in_time_correct_join(
        event_dates: List[str],
        feature_release_dates: List[str],
        feature_values: np.ndarray,
    ) -> np.ndarray:
        """Enforce point-in-time correctness for fundamental data.

        Only uses features that were available (released) before each event date.
        """
        result = np.full((len(event_dates), feature_values.shape[1]), np.nan)

        for i, event_date in enumerate(event_dates):
            # Find features released before event date
            valid_mask = [d < event_date for d in feature_release_dates]
            valid_indices = [j for j, v in enumerate(valid_mask) if v]

            if valid_indices:
                # Use most recently released features
                latest_idx = max(valid_indices, key=lambda j: feature_release_dates[j])
                result[i] = feature_values[latest_idx]

        return result


# =============================================================================
# SECTION: Dataset Quality Monitor
# =============================================================================

class OnlineDataQualityMonitor:
    """Monitor data quality in real-time streaming scenarios.

    Tracks:
    - Missing value rates
    - Out-of-range values
    - Distribution shifts via running statistics
    - Duplicate detection
    - Temporal gap anomalies
    """

    def __init__(
        self,
        n_features: int,
        expected_ranges: Dict[int, Tuple[float, float]] = None,
        max_gap_seconds: float = 300.0,
    ):
        self.n_features = n_features
        self.expected_ranges = expected_ranges or {}
        self.max_gap_seconds = max_gap_seconds

        # Running statistics (Welford's online algorithm)
        self._n = 0
        self._mean = np.zeros(n_features)
        self._M2 = np.zeros(n_features)
        self._missing_count = np.zeros(n_features)
        self._range_violation_count = np.zeros(n_features)
        self._last_timestamp = None
        self._gap_violations = 0
        self._duplicate_hashes = set()
        self._total_records = 0

    def update(
        self,
        x: np.ndarray,
        timestamp: float = None,
    ) -> dict:
        """Process one data record and return quality metrics."""
        assert len(x) == self.n_features
        self._total_records += 1

        # Hash for duplicate detection
        row_hash = hashlib.md5(x.tobytes()).hexdigest()
        is_duplicate = row_hash in self._duplicate_hashes
        self._duplicate_hashes.add(row_hash)

        # Temporal gap check
        gap_violation = False
        if timestamp is not None and self._last_timestamp is not None:
            gap = timestamp - self._last_timestamp
            if gap > self.max_gap_seconds:
                self._gap_violations += 1
                gap_violation = True
        self._last_timestamp = timestamp

        # Feature-wise checks
        missing = np.isnan(x)
        self._missing_count += missing.astype(float)

        range_violations = np.zeros(self.n_features, dtype=bool)
        for feat_idx, (lo, hi) in self.expected_ranges.items():
            if not missing[feat_idx]:
                if x[feat_idx] < lo or x[feat_idx] > hi:
                    range_violations[feat_idx] = True
                    self._range_violation_count[feat_idx] += 1

        # Update running statistics (Welford)
        valid_mask = ~missing
        self._n += 1
        if valid_mask.any():
            delta = np.where(valid_mask, x - self._mean, 0)
            self._mean += delta / self._n
            delta2 = np.where(valid_mask, x - self._mean, 0)
            self._M2 += delta * delta2

        return {
            "missing": missing.tolist(),
            "range_violations": range_violations.tolist(),
            "is_duplicate": is_duplicate,
            "gap_violation": gap_violation,
            "record_id": self._total_records,
        }

    def summary(self) -> dict:
        """Return quality summary statistics."""
        variance = self._M2 / max(self._n - 1, 1)
        return {
            "total_records": self._total_records,
            "missing_rate": (self._missing_count / max(self._total_records, 1)).tolist(),
            "range_violation_rate": (self._range_violation_count / max(self._total_records, 1)).tolist(),
            "gap_violations": self._gap_violations,
            "duplicate_count": len(self._duplicate_hashes),
            "running_mean": self._mean.tolist(),
            "running_std": np.sqrt(variance).tolist(),
        }
