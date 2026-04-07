"""
data_handler.py -- Historical data management for LARSA backtesting.

Handles CSV/Parquet loading, multi-timeframe bar streaming, synthetic
data generation (GBM, OU, regime-switching), and multi-frequency alignment.
"""

from __future__ import annotations

import logging
import os
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from .engine import EventType, MarketEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# BarBuffer: rolling OHLCV window per symbol/timeframe
# ---------------------------------------------------------------------------

class BarBuffer:
    """
    Maintains a fixed-length rolling window of bars for a single symbol/timeframe.
    Supports O(1) append and O(n) numpy extraction for indicator calculations.
    """

    OHLCV_COLS = ["open", "high", "low", "close", "volume"]

    def __init__(self, symbol: str, timeframe: str, maxlen: int = 500):
        self.symbol = symbol
        self.timeframe = timeframe
        self.maxlen = maxlen
        self._opens: deque = deque(maxlen=maxlen)
        self._highs: deque = deque(maxlen=maxlen)
        self._lows: deque = deque(maxlen=maxlen)
        self._closes: deque = deque(maxlen=maxlen)
        self._volumes: deque = deque(maxlen=maxlen)
        self._timestamps: deque = deque(maxlen=maxlen)

    def push(self, event: MarketEvent) -> None:
        self._timestamps.append(event.timestamp)
        self._opens.append(event.open)
        self._highs.append(event.high)
        self._lows.append(event.low)
        self._closes.append(event.close)
        self._volumes.append(event.volume)

    def push_row(self, ts: pd.Timestamp, o: float, h: float, l: float, c: float, v: float) -> None:
        self._timestamps.append(ts)
        self._opens.append(o)
        self._highs.append(h)
        self._lows.append(l)
        self._closes.append(c)
        self._volumes.append(v)

    @property
    def size(self) -> int:
        return len(self._closes)

    def is_ready(self, min_bars: int = 1) -> bool:
        return self.size >= min_bars

    def closes(self, n: Optional[int] = None) -> np.ndarray:
        arr = np.array(self._closes)
        return arr[-n:] if n and len(arr) >= n else arr

    def opens(self, n: Optional[int] = None) -> np.ndarray:
        arr = np.array(self._opens)
        return arr[-n:] if n and len(arr) >= n else arr

    def highs(self, n: Optional[int] = None) -> np.ndarray:
        arr = np.array(self._highs)
        return arr[-n:] if n and len(arr) >= n else arr

    def lows(self, n: Optional[int] = None) -> np.ndarray:
        arr = np.array(self._lows)
        return arr[-n:] if n and len(arr) >= n else arr

    def volumes(self, n: Optional[int] = None) -> np.ndarray:
        arr = np.array(self._volumes)
        return arr[-n:] if n and len(arr) >= n else arr

    def timestamps(self, n: Optional[int] = None) -> List[pd.Timestamp]:
        lst = list(self._timestamps)
        return lst[-n:] if n else lst

    def typical_prices(self, n: Optional[int] = None) -> np.ndarray:
        h = self.highs(n)
        lo = self.lows(n)
        c = self.closes(n)
        return (h + lo + c) / 3.0

    def returns(self, n: Optional[int] = None) -> np.ndarray:
        c = self.closes(n)
        if len(c) < 2:
            return np.array([])
        return np.diff(np.log(c + 1e-12))

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "open": list(self._opens),
                "high": list(self._highs),
                "low": list(self._lows),
                "close": list(self._closes),
                "volume": list(self._volumes),
            },
            index=pd.DatetimeIndex(list(self._timestamps), name="timestamp"),
        )

    def last_close(self) -> float:
        return self._closes[-1] if self._closes else 0.0

    def last_bar(self) -> Optional[Dict[str, Any]]:
        if not self._closes:
            return None
        return {
            "timestamp": self._timestamps[-1],
            "open": self._opens[-1],
            "high": self._highs[-1],
            "low": self._lows[-1],
            "close": self._closes[-1],
            "volume": self._volumes[-1],
        }


# ---------------------------------------------------------------------------
# Historical Data Handler
# ---------------------------------------------------------------------------

class HistoricalDataHandler:
    """
    Streams historical OHLCV data chronologically as MarketEvents.
    Supports multiple symbols and multiple timeframes simultaneously.
    Data can be loaded from CSV, Parquet, or injected as DataFrames.

    Usage:
        handler = HistoricalDataHandler(symbols=["BTC/USDT"], timeframes=["15m"])
        handler.load({"BTC/USDT": df})
        for event in handler.stream():
            ...
    """

    REQUIRED_COLS = {"open", "high", "low", "close", "volume"}
    COL_ALIASES = {
        "Open": "open", "High": "high", "Low": "low", "Close": "close",
        "Volume": "volume", "OPEN": "open", "HIGH": "high", "LOW": "low",
        "CLOSE": "close", "VOLUME": "volume",
    }

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        buffer_size: int = 500,
    ):
        self.symbols = symbols or []
        self.timeframes = timeframes or ["15m"]
        self.buffer_size = buffer_size

        # Keyed by (symbol, timeframe)
        self._data: Dict[Tuple[str, str], pd.DataFrame] = {}
        self._iterators: Dict[Tuple[str, str], Iterator] = {}
        self._buffers: Dict[Tuple[str, str], BarBuffer] = {}
        self._exhausted: set = set()
        self._next_bars: Dict[Tuple[str, str], Optional[pd.Series]] = {}
        self._event_cache: List[MarketEvent] = []

    def _normalize_df(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Standardize column names and ensure DatetimeIndex."""
        df = df.copy()
        df.rename(columns=self.COL_ALIASES, inplace=True)
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame for {symbol} missing columns: {missing}")
        if not isinstance(df.index, pd.DatetimeIndex):
            if "timestamp" in df.columns:
                df = df.set_index("timestamp")
            elif "date" in df.columns:
                df = df.set_index("date")
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = df[~df.index.duplicated(keep="first")]
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").ffill().fillna(0.0)
        return df

    def load(
        self,
        data: Dict[str, Any],
        timeframe: str = "15m",
    ) -> None:
        """
        Load data from a dict of {symbol: source} where source can be
        a DataFrame, a CSV path, or a Parquet path.
        """
        for symbol, source in data.items():
            if symbol not in self.symbols:
                self.symbols.append(symbol)
            df = self._load_source(source, symbol)
            self.add_data(symbol, timeframe, df)

    def load_multi_timeframe(
        self, data: Dict[Tuple[str, str], Any]
    ) -> None:
        """Load {(symbol, timeframe): source} dict."""
        for (symbol, tf), source in data.items():
            if symbol not in self.symbols:
                self.symbols.append(symbol)
            if tf not in self.timeframes:
                self.timeframes.append(tf)
            df = self._load_source(source, symbol)
            self.add_data(symbol, tf, df)

    def _load_source(self, source: Any, symbol: str) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return self._normalize_df(source, symbol)
        if isinstance(source, str):
            if source.endswith(".parquet"):
                df = pd.read_parquet(source)
            elif source.endswith(".csv"):
                df = pd.read_csv(source, index_col=0, parse_dates=True)
            else:
                raise ValueError(f"Unknown file format for {source}")
            return self._normalize_df(df, symbol)
        raise TypeError(f"Cannot load data from {type(source)}")

    def load_from_csv(self, path: str, symbol: str, timeframe: str = "15m") -> None:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = self._normalize_df(df, symbol)
        self.add_data(symbol, timeframe, df)

    def load_from_parquet(self, path: str, symbol: str, timeframe: str = "15m") -> None:
        df = pd.read_parquet(path)
        df = self._normalize_df(df, symbol)
        self.add_data(symbol, timeframe, df)

    def add_data(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        key = (symbol, timeframe)
        self._data[key] = df
        self._buffers[key] = BarBuffer(symbol, timeframe, self.buffer_size)
        # Initialize iterator
        self._iterators[key] = iter(df.iterrows())
        self._next_bars[key] = None
        self._advance_iterator(key)

    def _advance_iterator(self, key: Tuple[str, str]) -> None:
        """Pull next row from iterator and cache it."""
        if key in self._exhausted:
            self._next_bars[key] = None
            return
        try:
            ts, row = next(self._iterators[key])
            self._next_bars[key] = (ts, row)
        except StopIteration:
            self._exhausted.add(key)
            self._next_bars[key] = None

    def get_next_bars(self) -> List[MarketEvent]:
        """
        Return the next chronological batch of MarketEvents (one per symbol
        that has the earliest timestamp). Called once per event loop iteration.
        """
        if not self._next_bars:
            return []

        # Find the earliest timestamp across all active streams
        pending = {k: v for k, v in self._next_bars.items() if v is not None}
        if not pending:
            return []

        earliest_ts = min(v[0] for v in pending.values())

        events = []
        for key, (ts, row) in list(pending.items()):
            if ts == earliest_ts:
                symbol, timeframe = key
                evt = MarketEvent(
                    event_type=EventType.MARKET,
                    timestamp=pd.Timestamp(ts),
                    symbol=symbol,
                    timeframe=timeframe,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                    vwap=float(row.get("vwap", row["close"])),
                )
                self._buffers[key].push(evt)
                events.append(evt)
                self._advance_iterator(key)

        return events

    def on_market_event(self, event: MarketEvent) -> None:
        """Passthrough handler -- buffers are updated during get_next_bars."""
        pass

    def stream(self) -> Generator[MarketEvent, None, None]:
        """Generator that yields all events in chronological order."""
        while True:
            events = self.get_next_bars()
            if not events:
                break
            for evt in events:
                yield evt

    def get_buffer(self, symbol: str, timeframe: str = "15m") -> Optional[BarBuffer]:
        return self._buffers.get((symbol, timeframe))

    def reset(self) -> None:
        """Reset iterators to replay the data."""
        self._exhausted.clear()
        self._next_bars.clear()
        for key in self._data:
            self._buffers[key] = BarBuffer(key[0], key[1], self.buffer_size)
            self._iterators[key] = iter(self._data[key].iterrows())
            self._advance_iterator(key)

    @property
    def is_exhausted(self) -> bool:
        return all(v is None for v in self._next_bars.values())

    def date_range(self, symbol: str, timeframe: str = "15m") -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        key = (symbol, timeframe)
        if key not in self._data:
            return None
        df = self._data[key]
        return df.index[0], df.index[-1]

    def num_bars(self, symbol: str, timeframe: str = "15m") -> int:
        key = (symbol, timeframe)
        return len(self._data.get(key, []))


# ---------------------------------------------------------------------------
# Synthetic Data Generator
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """
    Generates synthetic OHLCV price paths for testing.

    Supported processes:
      - GBM: geometric Brownian motion
      - OU: Ornstein-Uhlenbeck (mean-reverting)
      - Regime-switching: alternates between trending and mean-reverting regimes
      - Jump-diffusion: GBM with Poisson jumps
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def gbm(
        self,
        n_bars: int = 1000,
        s0: float = 50_000.0,
        mu: float = 0.0003,     # per-bar drift (15m bar ~ 0.03% avg drift)
        sigma: float = 0.005,   # per-bar vol
        start: str = "2023-01-01",
        freq: str = "15min",
        symbol: str = "SYN_GBM",
    ) -> pd.DataFrame:
        """Geometric Brownian Motion path."""
        dt = 1.0
        returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * self.rng.standard_normal(n_bars)
        prices = s0 * np.exp(np.cumsum(returns))
        prices = np.insert(prices, 0, s0)[:-1]
        return self._prices_to_ohlcv(prices, start, freq, symbol)

    def ou_process(
        self,
        n_bars: int = 1000,
        s0: float = 50_000.0,
        theta: float = 0.05,    # mean-reversion speed
        mu: float = 50_000.0,   # long-run mean
        sigma: float = 500.0,   # vol (absolute)
        start: str = "2023-01-01",
        freq: str = "15min",
        symbol: str = "SYN_OU",
    ) -> pd.DataFrame:
        """Ornstein-Uhlenbeck mean-reverting process."""
        prices = np.zeros(n_bars)
        prices[0] = s0
        for i in range(1, n_bars):
            dX = theta * (mu - prices[i - 1]) + sigma * self.rng.standard_normal()
            prices[i] = max(prices[i - 1] + dX, 1.0)
        return self._prices_to_ohlcv(prices, start, freq, symbol)

    def regime_switching(
        self,
        n_bars: int = 2000,
        s0: float = 50_000.0,
        regime_params: Optional[Dict[str, Dict]] = None,
        transition_matrix: Optional[np.ndarray] = None,
        start: str = "2023-01-01",
        freq: str = "15min",
        symbol: str = "SYN_REGIME",
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Two-state regime-switching model.
        Returns (ohlcv_df, regime_labels) where labels: 0=trending, 1=mean-reverting.
        """
        if regime_params is None:
            regime_params = {
                "trending": {"mu": 0.0005, "sigma": 0.008},
                "mean_rev": {"mu": -0.0001, "sigma": 0.004},
            }
        if transition_matrix is None:
            # 95% stay, 5% transition
            transition_matrix = np.array([[0.95, 0.05], [0.05, 0.95]])

        regimes = np.zeros(n_bars, dtype=int)
        prices = np.zeros(n_bars)
        prices[0] = s0
        current_regime = 0

        p_trending = regime_params["trending"]
        p_meanrev = regime_params["mean_rev"]

        for i in range(1, n_bars):
            regimes[i] = current_regime
            if current_regime == 0:
                r = p_trending["mu"] - 0.5 * p_trending["sigma"]**2
                r += p_trending["sigma"] * self.rng.standard_normal()
                prices[i] = prices[i - 1] * np.exp(r)
            else:
                r = p_meanrev["mu"] - 0.5 * p_meanrev["sigma"]**2
                r += p_meanrev["sigma"] * self.rng.standard_normal()
                prices[i] = prices[i - 1] * np.exp(r)

            # Regime transition
            if self.rng.random() < transition_matrix[current_regime, 1 - current_regime]:
                current_regime = 1 - current_regime

        df = self._prices_to_ohlcv(prices, start, freq, symbol)
        return df, regimes

    def jump_diffusion(
        self,
        n_bars: int = 1000,
        s0: float = 50_000.0,
        mu: float = 0.0003,
        sigma: float = 0.005,
        jump_intensity: float = 0.02,   # avg jumps per bar
        jump_mean: float = -0.01,
        jump_std: float = 0.03,
        start: str = "2023-01-01",
        freq: str = "15min",
        symbol: str = "SYN_JUMP",
    ) -> pd.DataFrame:
        """Merton jump-diffusion model."""
        diffusion = (mu - 0.5 * sigma**2) + sigma * self.rng.standard_normal(n_bars)
        n_jumps = self.rng.poisson(jump_intensity, n_bars)
        jump_sizes = np.array([
            np.sum(self.rng.normal(jump_mean, jump_std, k)) if k > 0 else 0.0
            for k in n_jumps
        ])
        returns = diffusion + jump_sizes
        prices = s0 * np.exp(np.cumsum(returns))
        prices = np.insert(prices, 0, s0)[:-1]
        return self._prices_to_ohlcv(prices, start, freq, symbol)

    def correlated_assets(
        self,
        symbols: List[str],
        n_bars: int = 1000,
        s0_list: Optional[List[float]] = None,
        mu_list: Optional[List[float]] = None,
        sigma_list: Optional[List[float]] = None,
        correlation_matrix: Optional[np.ndarray] = None,
        start: str = "2023-01-01",
        freq: str = "15min",
    ) -> Dict[str, pd.DataFrame]:
        """Generate correlated multi-asset GBM paths."""
        n = len(symbols)
        if s0_list is None:
            s0_list = [50_000.0] * n
        if mu_list is None:
            mu_list = [0.0003] * n
        if sigma_list is None:
            sigma_list = [0.005] * n
        if correlation_matrix is None:
            # Default: 0.7 pairwise correlation
            corr = np.full((n, n), 0.7)
            np.fill_diagonal(corr, 1.0)
            correlation_matrix = corr

        sigma_arr = np.array(sigma_list)
        cov = np.outer(sigma_arr, sigma_arr) * correlation_matrix
        L = np.linalg.cholesky(cov)
        z = self.rng.standard_normal((n, n_bars))
        corr_z = L @ z  # (n, n_bars)

        result = {}
        for i, sym in enumerate(symbols):
            mu_i = mu_list[i]
            sig_i = sigma_list[i]
            s0_i = s0_list[i]
            ret = (mu_i - 0.5 * sig_i**2) + corr_z[i]
            prices = s0_i * np.exp(np.cumsum(ret))
            prices = np.insert(prices, 0, s0_i)[:-1]
            result[sym] = self._prices_to_ohlcv(prices, start, freq, sym)

        return result

    def _prices_to_ohlcv(
        self,
        prices: np.ndarray,
        start: str,
        freq: str,
        symbol: str,
    ) -> pd.DataFrame:
        """Convert a series of close prices to a realistic OHLCV DataFrame."""
        n = len(prices)
        # Simulate intrabar volatility for high/low
        intrabar_noise = np.abs(self.rng.normal(0, 0.001, n))
        highs = prices * (1 + intrabar_noise)
        lows = prices * (1 - intrabar_noise)
        # Open: previous close with small gap
        opens = np.roll(prices, 1)
        opens[0] = prices[0]
        opens = opens * (1 + self.rng.normal(0, 0.0002, n))
        # Volume: lognormal
        volumes = np.exp(self.rng.normal(10, 1, n))  # median ~22k units

        idx = pd.date_range(start=start, periods=n, freq=freq)
        return pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
            },
            index=idx,
        )


# ---------------------------------------------------------------------------
# DataAligner: aligns multiple timeframes to a common timeline
# ---------------------------------------------------------------------------

class DataAligner:
    """
    Aligns OHLCV data from multiple timeframes to the base (lowest) timeframe.

    For a 15m base timeframe, it adds lagged 1h and 4h bar values as additional
    columns on each 15m bar, using the last completed higher-tf bar only
    (no lookahead).
    """

    TF_MINUTES = {
        "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
        "1h": 60, "2h": 120, "4h": 240, "1d": 1440,
    }

    def __init__(self, base_timeframe: str = "15m"):
        self.base_tf = base_timeframe
        self.base_minutes = self.TF_MINUTES.get(base_timeframe, 15)

    def align(
        self,
        base_df: pd.DataFrame,
        higher_dfs: Dict[str, pd.DataFrame],
        method: str = "ffill",
    ) -> pd.DataFrame:
        """
        Merge higher-timeframe data into the base DataFrame.

        For each higher TF bar at time T, the value is available at T + bar_duration
        (i.e., when the bar closes). This prevents lookahead bias.

        Returns an enriched base DataFrame with extra columns like
        close_1h, close_4h, volume_1h, etc.
        """
        result = base_df.copy()

        for tf, df in higher_dfs.items():
            tf_mins = self.TF_MINUTES.get(tf, 60)
            if tf_mins <= self.base_minutes:
                logger.warning("Skipping TF %s which is not higher than base %s", tf, self.base_tf)
                continue

            # Shift timestamps: bar opens at T, closes at T + duration
            # So data available starting at T + duration
            offset = pd.Timedelta(minutes=tf_mins)
            df_shifted = df.copy()
            df_shifted.index = df_shifted.index + offset

            # Rename columns
            col_map = {col: f"{col}_{tf}" for col in df_shifted.columns}
            df_shifted = df_shifted.rename(columns=col_map)

            # Merge and forward-fill
            result = result.join(df_shifted, how="left")
            tf_cols = list(col_map.values())
            result[tf_cols] = result[tf_cols].ffill() if method == "ffill" else result[tf_cols].bfill()

        return result

    def resample_to_higher(
        self,
        base_df: pd.DataFrame,
        target_tf: str,
    ) -> pd.DataFrame:
        """
        Resample a base-timeframe DataFrame to a higher timeframe.
        Useful for constructing higher-tf bars from 15m data.
        """
        tf_mins = self.TF_MINUTES.get(target_tf)
        if tf_mins is None:
            raise ValueError(f"Unknown timeframe: {target_tf}")

        rule = f"{tf_mins}min"
        resampled = base_df.resample(rule).agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        ).dropna()
        return resampled

    def build_multi_tf_dataset(
        self,
        base_df: pd.DataFrame,
        timeframes: List[str],
    ) -> pd.DataFrame:
        """
        Build a full multi-timeframe dataset from a single base DataFrame
        by resampling up and then re-aligning down.
        """
        higher_dfs = {}
        for tf in timeframes:
            tf_mins = self.TF_MINUTES.get(tf, 0)
            if tf_mins > self.base_minutes:
                higher_dfs[tf] = self.resample_to_higher(base_df, tf)

        if not higher_dfs:
            return base_df

        return self.align(base_df, higher_dfs)


# ---------------------------------------------------------------------------
# Multi-symbol data container
# ---------------------------------------------------------------------------

class MultiAssetData:
    """
    Container for aligned multi-symbol, multi-timeframe data.

    Provides a unified interface for strategy access to bar data:
      data["BTC/USDT"]["15m"].closes(50) -- last 50 closes on the 15m TF
    """

    def __init__(self):
        self._buffers: Dict[str, Dict[str, BarBuffer]] = defaultdict(dict)
        self._current_ts: Optional[pd.Timestamp] = None

    def get_buffer(self, symbol: str, timeframe: str = "15m") -> Optional[BarBuffer]:
        return self._buffers.get(symbol, {}).get(timeframe)

    def update(self, event: MarketEvent) -> None:
        sym = event.symbol
        tf = event.timeframe
        if tf not in self._buffers[sym]:
            self._buffers[sym][tf] = BarBuffer(sym, tf)
        self._buffers[sym][tf].push(event)
        self._current_ts = event.timestamp

    def __getitem__(self, symbol: str) -> Dict[str, BarBuffer]:
        return self._buffers[symbol]

    def symbols(self) -> List[str]:
        return list(self._buffers.keys())

    def timeframes(self, symbol: str) -> List[str]:
        return list(self._buffers.get(symbol, {}).keys())

    def all_ready(self, min_bars: int, timeframe: str = "15m") -> bool:
        """True if all symbols have at least min_bars in the given timeframe."""
        for sym in self._buffers:
            buf = self._buffers[sym].get(timeframe)
            if buf is None or not buf.is_ready(min_bars):
                return False
        return True
