"""
aftp_pipeline.py — Automated Feature-to-Tensor Pipeline (AFTP) for AETERNUS.

High-speed ETL that transforms raw market data into pre-allocated tensor buffers
conforming to UTR schemas.

Modes:
  - Streaming: process one tick at a time, maintain rolling window state
  - Batch: process full historical window for backtest

Inputs:
  - Raw OHLCV parquet / CSV files
  - Chronos shm-bus channel (mmap shared-memory ring buffer)

Outputs:
  - UTR-schema tensors in pre-allocated shared memory buffers

Features computed:
  - Returns, log-returns
  - Rolling volatility (std-dev based)
  - Realized volatility (Parkinson, Garman-Klass)
  - Bid-ask features (spread, mid, imbalance)
  - VWAP deviation
  - Technical indicators: RSI, MACD, Bollinger Bands
  - Order imbalance
"""

from __future__ import annotations

import ctypes
import logging
import math
import mmap
import multiprocessing as mp
import os
import queue
import struct
import time
import threading
import warnings
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Callable, Deque, Dict, Generator, Iterator, List,
    Optional, Sequence, Tuple, Union
)

import numpy as np

from .unified_tensor_registry import (
    TensorEnvelope,
    UnifiedTensorRegistry,
    allocate_chronos_buffer,
    make_chronos_envelope,
    CHRONOS_OUTPUT_SCHEMA,
    ValidationResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default rolling window for most features
DEFAULT_WINDOW = 20

# OHLCV column indices in raw array
COL_OPEN   = 0
COL_HIGH   = 1
COL_LOW    = 2
COL_CLOSE  = 3
COL_VOLUME = 4
COL_BID    = 5
COL_ASK    = 6

# ChronosOutput feature indices
FEAT_BID       = 0
FEAT_ASK       = 1
FEAT_MID       = 2
FEAT_SPREAD    = 3
FEAT_VOLUME    = 4
FEAT_IMBALANCE = 5

# Shared memory magic header
SHM_MAGIC   = b"AFTP"
SHM_VERSION = 1


# ---------------------------------------------------------------------------
# Tick data structures
# ---------------------------------------------------------------------------

@dataclass
class RawTick:
    """Single market data tick for one asset."""
    asset_idx: int
    timestamp_ns: int
    open:   float
    high:   float
    low:    float
    close:  float
    volume: float
    bid:    float
    ask:    float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def imbalance(self) -> float:
        """Simple price-based imbalance proxy: (close - open) / (high - low + 1e-8)."""
        denom = self.high - self.low + 1e-8
        return (self.close - self.open) / denom


@dataclass
class OHLCVFrame:
    """
    A batch of OHLCV rows for multiple assets.

    Shape conventions:
      data: (n_assets, n_bars, 7) — columns [open, high, low, close, vol, bid, ask]
      timestamps: (n_bars,) int64 ns
    """
    data: np.ndarray        # (n_assets, n_bars, 7) float32
    timestamps: np.ndarray  # (n_bars,) int64
    asset_ids: List[str]
    source: str = "unknown"


# ---------------------------------------------------------------------------
# Rolling window state
# ---------------------------------------------------------------------------

class RollingState:
    """
    Maintains a fixed-length circular buffer of scalar values per asset.
    Supports O(1) push and O(window) statistics.
    """

    def __init__(self, n_assets: int, window: int) -> None:
        self.n_assets = n_assets
        self.window = window
        # (n_assets, window) ring buffer
        self._buf: np.ndarray = np.zeros((n_assets, window), dtype=np.float64)
        self._pos: np.ndarray = np.zeros(n_assets, dtype=np.int64)
        self._count: np.ndarray = np.zeros(n_assets, dtype=np.int64)

    def push(self, values: np.ndarray) -> None:
        """Push a (n_assets,) array of new values."""
        for i in range(self.n_assets):
            idx = int(self._pos[i]) % self.window
            self._buf[i, idx] = values[i]
            self._pos[i] += 1
            if self._count[i] < self.window:
                self._count[i] += 1

    def mean(self) -> np.ndarray:
        result = np.zeros(self.n_assets, dtype=np.float64)
        for i in range(self.n_assets):
            c = int(self._count[i])
            if c == 0:
                continue
            result[i] = np.mean(self._buf[i, :c]) if c < self.window else np.mean(self._buf[i])
        return result

    def std(self, ddof: int = 1) -> np.ndarray:
        result = np.zeros(self.n_assets, dtype=np.float64)
        for i in range(self.n_assets):
            c = int(self._count[i])
            if c <= ddof:
                continue
            vals = self._buf[i, :c] if c < self.window else self._buf[i]
            result[i] = np.std(vals, ddof=ddof)
        return result

    def last(self, n: int = 1) -> np.ndarray:
        """Return the last n values for each asset; shape (n_assets, n)."""
        result = np.zeros((self.n_assets, n), dtype=np.float64)
        for i in range(self.n_assets):
            c = int(self._count[i])
            if c == 0:
                continue
            buf = self._buf[i]
            pos = int(self._pos[i])
            indices = [(pos - 1 - k) % self.window for k in range(min(n, c))]
            result[i, :len(indices)] = buf[indices[::-1]]
        return result

    def full(self) -> bool:
        """True when all assets have at least window observations."""
        return bool(np.all(self._count >= self.window))

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._pos[:] = 0
        self._count[:] = 0


# ---------------------------------------------------------------------------
# Feature extractors
# ---------------------------------------------------------------------------

class FeatureExtractor(ABC):
    """Base class for all feature extractors."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    @abstractmethod
    def update(self, tick: np.ndarray) -> np.ndarray:
        """
        Update internal state with new tick(s) and return feature values.
        tick: (n_assets, 7) array for one timestep.
        Returns (n_assets, output_dim) float32.
        """

    def reset(self) -> None:
        pass


class ReturnsExtractor(FeatureExtractor):
    """Simple and log returns from close prices."""

    def __init__(self, n_assets: int) -> None:
        self._n = n_assets
        self._prev_close: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "returns"

    @property
    def output_dim(self) -> int:
        return 2  # [simple_ret, log_ret]

    def update(self, tick: np.ndarray) -> np.ndarray:
        close = tick[:, COL_CLOSE].astype(np.float64)
        if self._prev_close is None:
            self._prev_close = close.copy()
            return np.zeros((self._n, 2), dtype=np.float32)
        simple = (close - self._prev_close) / (self._prev_close + 1e-12)
        log_ret = np.log((close + 1e-12) / (self._prev_close + 1e-12))
        self._prev_close = close.copy()
        out = np.stack([simple, log_ret], axis=-1).astype(np.float32)
        return out

    def reset(self) -> None:
        self._prev_close = None


class RollingVolatilityExtractor(FeatureExtractor):
    """Rolling standard deviation of log returns."""

    def __init__(self, n_assets: int, window: int = DEFAULT_WINDOW) -> None:
        self._n = n_assets
        self._window = window
        self._log_ret_state = RollingState(n_assets, window)
        self._prev_close: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return f"rolling_vol_{self._window}"

    @property
    def output_dim(self) -> int:
        return 1

    def update(self, tick: np.ndarray) -> np.ndarray:
        close = tick[:, COL_CLOSE].astype(np.float64)
        if self._prev_close is None:
            self._prev_close = close.copy()
            return np.zeros((self._n, 1), dtype=np.float32)
        log_ret = np.log((close + 1e-12) / (self._prev_close + 1e-12))
        self._log_ret_state.push(log_ret)
        self._prev_close = close.copy()
        vol = self._log_ret_state.std(ddof=1)
        return vol.reshape(self._n, 1).astype(np.float32)

    def reset(self) -> None:
        self._prev_close = None
        self._log_ret_state.reset()


class ParkinsonVolatilityExtractor(FeatureExtractor):
    """
    Parkinson realized volatility estimator.
    RV_park = sqrt( 1/(4*ln2) * E[(ln(H/L))^2] ) over rolling window.
    """

    def __init__(self, n_assets: int, window: int = DEFAULT_WINDOW) -> None:
        self._n = n_assets
        self._window = window
        self._hl_sq_state = RollingState(n_assets, window)
        self._factor = 1.0 / (4.0 * math.log(2.0))

    @property
    def name(self) -> str:
        return f"parkinson_vol_{self._window}"

    @property
    def output_dim(self) -> int:
        return 1

    def update(self, tick: np.ndarray) -> np.ndarray:
        high  = tick[:, COL_HIGH].astype(np.float64)
        low   = tick[:, COL_LOW].astype(np.float64)
        hl_sq = (np.log((high + 1e-12) / (low + 1e-12))) ** 2
        self._hl_sq_state.push(hl_sq)
        mean_sq = self._hl_sq_state.mean()
        rv = np.sqrt(np.maximum(self._factor * mean_sq, 0.0))
        return rv.reshape(self._n, 1).astype(np.float32)

    def reset(self) -> None:
        self._hl_sq_state.reset()


class GarmanKlassVolatilityExtractor(FeatureExtractor):
    """
    Garman-Klass volatility estimator.
    GK = 0.5*(ln(H/L))^2 - (2*ln2 - 1)*(ln(C/O))^2
    Rolling mean over window.
    """

    def __init__(self, n_assets: int, window: int = DEFAULT_WINDOW) -> None:
        self._n = n_assets
        self._window = window
        self._gk_state = RollingState(n_assets, window)
        self._c = 2.0 * math.log(2.0) - 1.0

    @property
    def name(self) -> str:
        return f"gk_vol_{self._window}"

    @property
    def output_dim(self) -> int:
        return 1

    def update(self, tick: np.ndarray) -> np.ndarray:
        open_p  = tick[:, COL_OPEN].astype(np.float64)
        high    = tick[:, COL_HIGH].astype(np.float64)
        low     = tick[:, COL_LOW].astype(np.float64)
        close   = tick[:, COL_CLOSE].astype(np.float64)
        hl_term = 0.5 * (np.log((high + 1e-12) / (low + 1e-12))) ** 2
        co_term = self._c * (np.log((close + 1e-12) / (open_p + 1e-12))) ** 2
        gk_val  = hl_term - co_term
        self._gk_state.push(gk_val)
        rv = np.sqrt(np.maximum(self._gk_state.mean(), 0.0))
        return rv.reshape(self._n, 1).astype(np.float32)

    def reset(self) -> None:
        self._gk_state.reset()


class BidAskFeatureExtractor(FeatureExtractor):
    """
    Bid-ask features: spread, mid, spread-to-mid ratio, rolling avg spread.
    """

    def __init__(self, n_assets: int, window: int = DEFAULT_WINDOW) -> None:
        self._n = n_assets
        self._spread_state = RollingState(n_assets, window)

    @property
    def name(self) -> str:
        return "bid_ask"

    @property
    def output_dim(self) -> int:
        return 4  # [spread, mid, spread/mid, rolling_avg_spread]

    def update(self, tick: np.ndarray) -> np.ndarray:
        bid = tick[:, COL_BID].astype(np.float64)
        ask = tick[:, COL_ASK].astype(np.float64)
        spread = ask - bid
        mid    = (bid + ask) / 2.0
        ratio  = spread / (mid + 1e-12)
        self._spread_state.push(spread)
        avg_spread = self._spread_state.mean()
        out = np.stack([spread, mid, ratio, avg_spread], axis=-1).astype(np.float32)
        return out

    def reset(self) -> None:
        self._spread_state.reset()


class OrderImbalanceExtractor(FeatureExtractor):
    """
    Order imbalance: (close - open) / (high - low + eps).
    Rolling mean over window.
    """

    def __init__(self, n_assets: int, window: int = DEFAULT_WINDOW) -> None:
        self._n = n_assets
        self._imb_state = RollingState(n_assets, window)

    @property
    def name(self) -> str:
        return "order_imbalance"

    @property
    def output_dim(self) -> int:
        return 2  # [current_imbalance, rolling_mean_imbalance]

    def update(self, tick: np.ndarray) -> np.ndarray:
        open_p = tick[:, COL_OPEN].astype(np.float64)
        high   = tick[:, COL_HIGH].astype(np.float64)
        low    = tick[:, COL_LOW].astype(np.float64)
        close  = tick[:, COL_CLOSE].astype(np.float64)
        imb = (close - open_p) / (high - low + 1e-8)
        self._imb_state.push(imb)
        rolling_imb = self._imb_state.mean()
        out = np.stack([imb, rolling_imb], axis=-1).astype(np.float32)
        return out

    def reset(self) -> None:
        self._imb_state.reset()


class VWAPDeviationExtractor(FeatureExtractor):
    """
    Rolling VWAP and deviation of close from VWAP.
    VWAP = sum(close * volume) / sum(volume) over window.
    """

    def __init__(self, n_assets: int, window: int = DEFAULT_WINDOW) -> None:
        self._n = n_assets
        self._pv_state = RollingState(n_assets, window)   # price * volume
        self._vol_state = RollingState(n_assets, window)  # volume

    @property
    def name(self) -> str:
        return "vwap_deviation"

    @property
    def output_dim(self) -> int:
        return 2  # [vwap, close_deviation]

    def update(self, tick: np.ndarray) -> np.ndarray:
        close  = tick[:, COL_CLOSE].astype(np.float64)
        volume = tick[:, COL_VOLUME].astype(np.float64)
        pv = close * volume
        self._pv_state.push(pv)
        self._vol_state.push(volume)
        sum_pv  = self._pv_state.mean() * self._pv_state._count.clip(1)
        sum_vol = self._vol_state.mean() * self._vol_state._count.clip(1)
        vwap = sum_pv / (sum_vol + 1e-12)
        deviation = (close - vwap) / (vwap + 1e-12)
        out = np.stack([vwap, deviation], axis=-1).astype(np.float32)
        return out

    def reset(self) -> None:
        self._pv_state.reset()
        self._vol_state.reset()


class RSIExtractor(FeatureExtractor):
    """RSI (Relative Strength Index) with Wilder smoothing."""

    def __init__(self, n_assets: int, period: int = 14) -> None:
        self._n = n_assets
        self._period = period
        self._avg_gain = np.zeros(n_assets, dtype=np.float64)
        self._avg_loss = np.zeros(n_assets, dtype=np.float64)
        self._prev_close: Optional[np.ndarray] = None
        self._tick_count = 0

    @property
    def name(self) -> str:
        return f"rsi_{self._period}"

    @property
    def output_dim(self) -> int:
        return 1

    def update(self, tick: np.ndarray) -> np.ndarray:
        close = tick[:, COL_CLOSE].astype(np.float64)
        if self._prev_close is None:
            self._prev_close = close.copy()
            return np.full((self._n, 1), 50.0, dtype=np.float32)
        diff = close - self._prev_close
        gain = np.maximum(diff, 0.0)
        loss = np.maximum(-diff, 0.0)
        self._tick_count += 1
        alpha = 1.0 / self._period
        if self._tick_count == 1:
            self._avg_gain = gain
            self._avg_loss = loss
        else:
            self._avg_gain = (1.0 - alpha) * self._avg_gain + alpha * gain
            self._avg_loss = (1.0 - alpha) * self._avg_loss + alpha * loss
        rs = self._avg_gain / (self._avg_loss + 1e-12)
        rsi = 100.0 - 100.0 / (1.0 + rs)
        self._prev_close = close.copy()
        return rsi.reshape(self._n, 1).astype(np.float32)

    def reset(self) -> None:
        self._avg_gain[:] = 0.0
        self._avg_loss[:] = 0.0
        self._prev_close = None
        self._tick_count = 0


class MACDExtractor(FeatureExtractor):
    """
    MACD = EMA(fast) - EMA(slow), plus signal line EMA(MACD, signal_period).
    Returns [macd_line, signal_line, histogram].
    """

    def __init__(
        self,
        n_assets: int,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> None:
        self._n = n_assets
        self._fast = fast
        self._slow = slow
        self._signal = signal
        self._ema_fast = np.zeros(n_assets, dtype=np.float64)
        self._ema_slow = np.zeros(n_assets, dtype=np.float64)
        self._ema_sig  = np.zeros(n_assets, dtype=np.float64)
        self._initialized = False
        self._tick_count = 0

    @property
    def name(self) -> str:
        return f"macd_{self._fast}_{self._slow}_{self._signal}"

    @property
    def output_dim(self) -> int:
        return 3

    def update(self, tick: np.ndarray) -> np.ndarray:
        close = tick[:, COL_CLOSE].astype(np.float64)
        if not self._initialized:
            self._ema_fast = close.copy()
            self._ema_slow = close.copy()
            self._ema_sig  = np.zeros(self._n)
            self._initialized = True
            return np.zeros((self._n, 3), dtype=np.float32)
        a_fast = 2.0 / (self._fast + 1.0)
        a_slow = 2.0 / (self._slow + 1.0)
        a_sig  = 2.0 / (self._signal + 1.0)
        self._ema_fast += a_fast * (close - self._ema_fast)
        self._ema_slow += a_slow * (close - self._ema_slow)
        macd = self._ema_fast - self._ema_slow
        self._ema_sig += a_sig * (macd - self._ema_sig)
        hist = macd - self._ema_sig
        out = np.stack([macd, self._ema_sig, hist], axis=-1).astype(np.float32)
        return out

    def reset(self) -> None:
        self._ema_fast[:] = 0.0
        self._ema_slow[:] = 0.0
        self._ema_sig[:] = 0.0
        self._initialized = False


class BollingerBandsExtractor(FeatureExtractor):
    """
    Bollinger Bands: upper, middle (SMA), lower, %B, bandwidth.
    """

    def __init__(
        self,
        n_assets: int,
        window: int = 20,
        n_std: float = 2.0,
    ) -> None:
        self._n = n_assets
        self._window = window
        self._n_std = n_std
        self._close_state = RollingState(n_assets, window)

    @property
    def name(self) -> str:
        return f"bbands_{self._window}"

    @property
    def output_dim(self) -> int:
        return 5  # [upper, mid, lower, pct_b, bandwidth]

    def update(self, tick: np.ndarray) -> np.ndarray:
        close = tick[:, COL_CLOSE].astype(np.float64)
        self._close_state.push(close)
        mid = self._close_state.mean()
        std = self._close_state.std(ddof=0)
        upper = mid + self._n_std * std
        lower = mid - self._n_std * std
        bandwidth = (upper - lower) / (mid + 1e-12)
        pct_b = (close - lower) / (upper - lower + 1e-12)
        out = np.stack([upper, mid, lower, pct_b, bandwidth], axis=-1).astype(np.float32)
        return out

    def reset(self) -> None:
        self._close_state.reset()


# ---------------------------------------------------------------------------
# Feature group registry
# ---------------------------------------------------------------------------

class FeatureGroupConfig:
    """Configuration for a named group of extractors."""

    def __init__(
        self,
        name: str,
        extractors: List[FeatureExtractor],
        enabled: bool = True,
    ) -> None:
        self.name = name
        self.extractors = extractors
        self.enabled = enabled

    @property
    def total_output_dim(self) -> int:
        return sum(e.output_dim for e in self.extractors)


def build_default_feature_groups(
    n_assets: int,
    window: int = DEFAULT_WINDOW,
) -> List[FeatureGroupConfig]:
    """Create the default set of feature groups for AFTP."""
    return [
        FeatureGroupConfig("returns", [
            ReturnsExtractor(n_assets),
        ]),
        FeatureGroupConfig("rolling_vol", [
            RollingVolatilityExtractor(n_assets, window),
        ]),
        FeatureGroupConfig("realized_vol", [
            ParkinsonVolatilityExtractor(n_assets, window),
            GarmanKlassVolatilityExtractor(n_assets, window),
        ]),
        FeatureGroupConfig("bid_ask", [
            BidAskFeatureExtractor(n_assets, window),
        ]),
        FeatureGroupConfig("imbalance", [
            OrderImbalanceExtractor(n_assets, window),
        ]),
        FeatureGroupConfig("vwap", [
            VWAPDeviationExtractor(n_assets, window),
        ]),
        FeatureGroupConfig("technical", [
            RSIExtractor(n_assets),
            MACDExtractor(n_assets),
            BollingerBandsExtractor(n_assets, window),
        ]),
    ]


# ---------------------------------------------------------------------------
# Throughput profiler
# ---------------------------------------------------------------------------

@dataclass
class ProfilerReport:
    total_ticks: int
    elapsed_sec: float
    ticks_per_sec: float
    group_latency_us: Dict[str, float]   # group name -> mean latency us
    total_latency_us: float


class ThroughputProfiler:
    """Lightweight profiler that tracks per-group latency and throughput."""

    def __init__(self) -> None:
        self._start_time: Optional[float] = None
        self._total_ticks: int = 0
        self._group_times: Dict[str, List[float]] = {}

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def record_group(self, group_name: str, elapsed_ns: int) -> None:
        lst = self._group_times.setdefault(group_name, [])
        lst.append(elapsed_ns / 1_000.0)  # to us

    def tick_done(self) -> None:
        self._total_ticks += 1

    def report(self) -> ProfilerReport:
        elapsed = (time.perf_counter() - (self._start_time or time.perf_counter()))
        tps = self._total_ticks / (elapsed + 1e-12)
        group_lat = {
            name: float(np.mean(times)) if times else 0.0
            for name, times in self._group_times.items()
        }
        total_lat = sum(group_lat.values())
        return ProfilerReport(
            total_ticks=self._total_ticks,
            elapsed_sec=elapsed,
            ticks_per_sec=tps,
            group_latency_us=group_lat,
            total_latency_us=total_lat,
        )

    def reset(self) -> None:
        self._start_time = None
        self._total_ticks = 0
        self._group_times.clear()


# ---------------------------------------------------------------------------
# Shared memory buffer management
# ---------------------------------------------------------------------------

@dataclass
class ShmBufferSpec:
    name: str
    shape: Tuple[int, ...]
    dtype: np.dtype
    capacity_ticks: int


class ShmRingBuffer:
    """
    Fixed-size shared-memory ring buffer for a single tensor type.
    Layout: [header: 64 bytes][data: capacity * frame_bytes]
    """

    HEADER_SIZE = 64
    HEADER_FMT  = "<4sHHIQ"  # magic(4) ver(2) dtype_idx(2) frame_bytes(4) write_ptr(8)

    _DTYPE_IDX = {
        np.dtype("float32"): 0,
        np.dtype("float64"): 1,
        np.dtype("int32"):   2,
        np.dtype("int64"):   3,
    }

    def __init__(
        self,
        spec: ShmBufferSpec,
        create: bool = True,
    ) -> None:
        self._spec = spec
        self._frame_shape = spec.shape
        self._dtype = spec.dtype
        frame_bytes = int(np.prod(spec.shape)) * spec.dtype.itemsize
        self._frame_bytes = frame_bytes
        total_bytes = self.HEADER_SIZE + spec.capacity_ticks * frame_bytes

        if create:
            self._mm = mmap.mmap(-1, total_bytes)
            self._write_header()
        else:
            # Attach to existing (not implemented here — would use shared-memory name)
            raise NotImplementedError("Attach mode requires OS-level shared memory API.")

        self._capacity = spec.capacity_ticks
        self._write_ptr: int = 0
        self._lock = threading.Lock()

    def _write_header(self) -> None:
        dtype_idx = self._DTYPE_IDX.get(self._dtype, 0)
        header = struct.pack(
            self.HEADER_FMT,
            SHM_MAGIC,
            SHM_VERSION,
            dtype_idx,
            self._frame_bytes,
            0,  # write_ptr
        )
        self._mm.seek(0)
        self._mm.write(header)

    def write(self, frame: np.ndarray) -> None:
        """Write one frame (shape matching spec.shape) to the ring buffer."""
        if frame.shape != self._frame_shape:
            raise ValueError(f"Frame shape {frame.shape} != spec shape {self._frame_shape}")
        frame_bytes = frame.astype(self._dtype).tobytes()
        with self._lock:
            slot = self._write_ptr % self._capacity
            offset = self.HEADER_SIZE + slot * self._frame_bytes
            self._mm.seek(offset)
            self._mm.write(frame_bytes)
            self._write_ptr += 1
            # Update header write_ptr
            self._mm.seek(self.HEADER_SIZE - 8)
            self._mm.write(struct.pack("<Q", self._write_ptr))

    def read_latest(self) -> Optional[np.ndarray]:
        """Return the most recently written frame, or None if empty."""
        with self._lock:
            if self._write_ptr == 0:
                return None
            slot = (self._write_ptr - 1) % self._capacity
            offset = self.HEADER_SIZE + slot * self._frame_bytes
            self._mm.seek(offset)
            raw = self._mm.read(self._frame_bytes)
        arr = np.frombuffer(raw, dtype=self._dtype).reshape(self._frame_shape).copy()
        return arr

    def read_window(self, n: int) -> np.ndarray:
        """Return the last n frames as (n, *spec.shape)."""
        with self._lock:
            available = min(n, self._write_ptr, self._capacity)
            frames = []
            for k in range(available - 1, -1, -1):
                slot = (self._write_ptr - 1 - k) % self._capacity
                offset = self.HEADER_SIZE + slot * self._frame_bytes
                self._mm.seek(offset)
                raw = self._mm.read(self._frame_bytes)
                frames.append(np.frombuffer(raw, dtype=self._dtype).reshape(self._frame_shape))
        if not frames:
            return np.zeros((0,) + self._frame_shape, dtype=self._dtype)
        return np.stack(frames, axis=0).copy()

    def close(self) -> None:
        self._mm.close()


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

def _feature_worker_fn(
    task_queue: "mp.Queue[Optional[Tuple[int, np.ndarray]]]",
    result_queue: "mp.Queue[Tuple[int, Dict[str, np.ndarray]]]",
    extractor_configs: List[Dict[str, Any]],
    n_assets: int,
    window: int,
) -> None:
    """
    Worker process: receives tick arrays, runs assigned feature extractors,
    returns results keyed by group name.
    """
    # Rebuild extractors in worker process
    groups = _build_groups_from_configs(extractor_configs, n_assets, window)
    while True:
        item = task_queue.get()
        if item is None:
            break
        tick_id, tick_array = item
        results: Dict[str, np.ndarray] = {}
        for group in groups:
            if not group.enabled:
                continue
            group_feats = []
            for extractor in group.extractors:
                feat = extractor.update(tick_array)
                group_feats.append(feat)
            results[group.name] = np.concatenate(group_feats, axis=-1)
        result_queue.put((tick_id, results))


def _build_groups_from_configs(
    configs: List[Dict[str, Any]],
    n_assets: int,
    window: int,
) -> List[FeatureGroupConfig]:
    """Reconstruct feature groups from a list of serializable config dicts."""
    return build_default_feature_groups(n_assets, window)


# ---------------------------------------------------------------------------
# AFTP core
# ---------------------------------------------------------------------------

class AFTPMode(Enum):
    STREAMING = auto()
    BATCH = auto()


@dataclass
class AFTPConfig:
    n_assets: int
    t_ticks: int              # rolling window size for ChronosOutput buffer
    window: int = DEFAULT_WINDOW
    mode: AFTPMode = AFTPMode.STREAMING
    n_workers: int = 1        # number of parallel feature workers (>1 = multiprocessing)
    enable_profiler: bool = True
    registry: Optional[UnifiedTensorRegistry] = None
    shm_buffer_capacity: int = 1024  # ring buffer capacity in ticks
    validate_output: bool = True


class AutomatedFeatureToPipeline:
    """
    Automated Feature-to-Tensor Pipeline (AFTP).

    Transforms raw OHLCV market data (from file or streaming shm-bus)
    into UTR-compliant ChronosOutput tensors in pre-allocated buffers.

    Usage (streaming)
    -----------------
    >>> cfg = AFTPConfig(n_assets=10, t_ticks=64)
    >>> aftp = AutomatedFeatureToPipeline(cfg)
    >>> aftp.start()
    >>> for tick_array in my_tick_source:
    ...     envelope = aftp.push_tick(tick_array)
    >>> report = aftp.profiler_report()
    >>> aftp.stop()

    Usage (batch)
    -------------
    >>> frames = aftp.process_batch(ohlcv_frame)
    """

    def __init__(self, config: AFTPConfig) -> None:
        self._cfg = config
        self._registry = config.registry or UnifiedTensorRegistry.global_registry()
        self._tick_id: int = 0
        self._profiler = ThroughputProfiler() if config.enable_profiler else None

        # Feature groups (single-process extractors for streaming)
        self._feature_groups = build_default_feature_groups(config.n_assets, config.window)

        # Pre-allocated ChronosOutput buffer (n_assets, t_ticks, 6)
        self._chronos_buf: np.ndarray = allocate_chronos_buffer(
            config.n_assets, config.t_ticks
        )
        # Rolling write position for time dimension
        self._buf_pos: int = 0
        self._buf_filled: bool = False

        # Shm ring buffer for downstream consumers
        self._shm_buf: Optional[ShmRingBuffer] = None
        self._running: bool = False

        # Multi-process workers
        self._workers: List[mp.Process] = []
        self._task_queue: Optional[mp.Queue] = None  # type: ignore[type-arg]
        self._result_queue: Optional[mp.Queue] = None  # type: ignore[type-arg]
        self._pending: Dict[int, None] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Initialize profiler and optionally launch worker processes."""
        if self._profiler:
            self._profiler.start()
        self._running = True

        if self._cfg.n_workers > 1:
            self._task_queue = mp.Queue(maxsize=256)
            self._result_queue = mp.Queue(maxsize=256)
            for _ in range(self._cfg.n_workers):
                p = mp.Process(
                    target=_feature_worker_fn,
                    args=(
                        self._task_queue,
                        self._result_queue,
                        [],           # extractor_configs (simplified)
                        self._cfg.n_assets,
                        self._cfg.window,
                    ),
                    daemon=True,
                )
                p.start()
                self._workers.append(p)
        logger.info(
            "AFTP started: n_assets=%d, t_ticks=%d, workers=%d, mode=%s",
            self._cfg.n_assets, self._cfg.t_ticks,
            self._cfg.n_workers, self._cfg.mode.name,
        )

    def stop(self) -> None:
        """Shut down worker processes and finalize profiler."""
        self._running = False
        if self._task_queue is not None:
            for _ in self._workers:
                self._task_queue.put(None)
            for p in self._workers:
                p.join(timeout=5.0)
        self._workers.clear()
        if self._shm_buf:
            self._shm_buf.close()
        logger.info("AFTP stopped after %d ticks.", self._tick_id)

    # ------------------------------------------------------------------ #
    # Streaming mode
    # ------------------------------------------------------------------ #

    def push_tick(self, tick_array: np.ndarray) -> TensorEnvelope:
        """
        Push a single tick into the pipeline.

        Parameters
        ----------
        tick_array:
            (n_assets, 7) array: [open, high, low, close, volume, bid, ask]

        Returns
        -------
        TensorEnvelope wrapping the current ChronosOutput window.
        """
        if tick_array.shape != (self._cfg.n_assets, 7):
            raise ValueError(
                f"tick_array shape {tick_array.shape} != ({self._cfg.n_assets}, 7)"
            )
        t0 = time.perf_counter_ns()

        # Extract bid-ask columns to fill ChronosOutput format
        bid     = tick_array[:, COL_BID].astype(np.float32)
        ask     = tick_array[:, COL_ASK].astype(np.float32)
        mid     = ((bid + ask) / 2.0).astype(np.float32)
        spread  = (ask - bid).astype(np.float32)
        volume  = tick_array[:, COL_VOLUME].astype(np.float32)

        # Compute imbalance
        open_p = tick_array[:, COL_OPEN].astype(np.float64)
        high   = tick_array[:, COL_HIGH].astype(np.float64)
        low    = tick_array[:, COL_LOW].astype(np.float64)
        close  = tick_array[:, COL_CLOSE].astype(np.float64)
        imbalance = ((close - open_p) / (high - low + 1e-8)).astype(np.float32)

        frame = np.stack([bid, ask, mid, spread, volume, imbalance], axis=-1)  # (N, 6)

        # Write into rolling buffer
        slot = self._buf_pos % self._cfg.t_ticks
        self._chronos_buf[:, slot, :] = frame
        self._buf_pos += 1
        if self._buf_pos >= self._cfg.t_ticks:
            self._buf_filled = True

        # Run feature groups for profiling
        if self._profiler and self._feature_groups:
            for group in self._feature_groups:
                if not group.enabled:
                    continue
                g_t0 = time.perf_counter_ns()
                for ext in group.extractors:
                    ext.update(tick_array)
                self._profiler.record_group(group.name, time.perf_counter_ns() - g_t0)
            self._profiler.tick_done()

        self._tick_id += 1

        # Write to shm ring buffer if available
        if self._shm_buf is not None:
            self._shm_buf.write(frame)

        env = TensorEnvelope(
            schema_name="ChronosOutput",
            data=self._chronos_buf.copy(),
            producer="AFTP",
            tick_id=self._tick_id,
        )

        if self._cfg.validate_output:
            # Only validate structure, skip value check for speed
            self._registry.validate("ChronosOutput", env.data, check_values=False)

        return env

    # ------------------------------------------------------------------ #
    # Batch mode
    # ------------------------------------------------------------------ #

    def process_batch(
        self,
        frame: OHLCVFrame,
        *,
        validate: bool = True,
    ) -> TensorEnvelope:
        """
        Process a full OHLCVFrame batch.

        Parameters
        ----------
        frame:
            OHLCVFrame with data shape (n_assets, n_bars, 7).

        Returns
        -------
        TensorEnvelope with ChronosOutput data of shape (n_assets, n_bars, 6).
        """
        N, T, C = frame.data.shape
        if C != 7:
            raise ValueError(f"OHLCVFrame data must have 7 columns, got {C}")
        if N != self._cfg.n_assets:
            raise ValueError(
                f"n_assets mismatch: config={self._cfg.n_assets}, frame={N}"
            )

        bid      = frame.data[:, :, COL_BID].astype(np.float32)
        ask      = frame.data[:, :, COL_ASK].astype(np.float32)
        mid      = ((bid + ask) / 2.0)
        spread   = (ask - bid)
        volume   = frame.data[:, :, COL_VOLUME].astype(np.float32)
        open_p   = frame.data[:, :, COL_OPEN].astype(np.float64)
        high_p   = frame.data[:, :, COL_HIGH].astype(np.float64)
        low_p    = frame.data[:, :, COL_LOW].astype(np.float64)
        close_p  = frame.data[:, :, COL_CLOSE].astype(np.float64)
        imbalance = ((close_p - open_p) / (high_p - low_p + 1e-8)).astype(np.float32)

        # Stack to (N, T, 6)
        out = np.stack([bid, ask, mid, spread, volume, imbalance], axis=-1)

        if validate:
            # Shape check: UTR schema is symbolic, any N and T are OK
            self._registry.validate("ChronosOutput", out, check_values=False)

        self._tick_id += T
        env = TensorEnvelope(
            schema_name="ChronosOutput",
            data=out,
            producer="AFTP_batch",
            tick_id=self._tick_id,
            metadata={"source": frame.source, "n_bars": T},
        )
        return env

    # ------------------------------------------------------------------ #
    # File ingestion
    # ------------------------------------------------------------------ #

    def ingest_csv(
        self,
        path: Union[str, Path],
        asset_idx: int = 0,
        *,
        n_assets_total: Optional[int] = None,
    ) -> OHLCVFrame:
        """
        Load an OHLCV CSV file (columns: timestamp, open, high, low, close, volume).
        Adds dummy bid/ask columns derived from close +/- spread_proxy.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for CSV ingestion.")

        df = pd.read_csv(path)
        required_cols = {"open", "high", "low", "close", "volume"}
        lower_cols = {c.lower(): c for c in df.columns}
        mapped = {k: lower_cols[k] for k in required_cols if k in lower_cols}
        if len(mapped) < len(required_cols):
            missing = required_cols - set(mapped.keys())
            raise ValueError(f"CSV missing columns: {missing}")

        n_bars = len(df)
        n_a = n_assets_total or self._cfg.n_assets

        data = np.zeros((n_a, n_bars, 7), dtype=np.float32)
        o = df[mapped["open"]].values.astype(np.float32)
        h = df[mapped["high"]].values.astype(np.float32)
        lo = df[mapped["low"]].values.astype(np.float32)
        c = df[mapped["close"]].values.astype(np.float32)
        v = df[mapped["volume"]].values.astype(np.float32)

        # Proxy bid/ask from 0.05% of close
        spread_proxy = c * 0.0005
        bid = c - spread_proxy / 2
        ask = c + spread_proxy / 2

        data[asset_idx, :, 0] = o
        data[asset_idx, :, 1] = h
        data[asset_idx, :, 2] = lo
        data[asset_idx, :, 3] = c
        data[asset_idx, :, 4] = v
        data[asset_idx, :, 5] = bid
        data[asset_idx, :, 6] = ask

        timestamps = np.arange(n_bars, dtype=np.int64)
        if "timestamp" in lower_cols:
            try:
                ts_vals = pd.to_datetime(df[lower_cols["timestamp"]]).astype(np.int64).values
                timestamps = ts_vals
            except Exception:
                pass

        return OHLCVFrame(
            data=data,
            timestamps=timestamps,
            asset_ids=[str(path)] + [f"pad_{i}" for i in range(n_a - 1)],
            source=str(path),
        )

    def ingest_parquet(
        self,
        path: Union[str, Path],
        asset_idx: int = 0,
    ) -> OHLCVFrame:
        """Load an OHLCV Parquet file. Same column conventions as ingest_csv."""
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required for parquet ingestion.")
        df = pd.read_parquet(path)
        # Delegate to CSV path via temp CSV (simple approach)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f, index=False)
            tmp_path = f.name
        result = self.ingest_csv(tmp_path, asset_idx=asset_idx)
        os.unlink(tmp_path)
        return result

    # ------------------------------------------------------------------ #
    # Profiler
    # ------------------------------------------------------------------ #

    def profiler_report(self) -> Optional[ProfilerReport]:
        if self._profiler is None:
            return None
        return self._profiler.report()

    def reset_profiler(self) -> None:
        if self._profiler:
            self._profiler.reset()
            self._profiler.start()

    # ------------------------------------------------------------------ #
    # State accessors
    # ------------------------------------------------------------------ #

    @property
    def current_buffer(self) -> np.ndarray:
        """Return the current pre-allocated ChronosOutput buffer (no copy)."""
        return self._chronos_buf

    @property
    def tick_id(self) -> int:
        return self._tick_id

    @property
    def buffer_filled(self) -> bool:
        return self._buf_filled

    def reset_state(self) -> None:
        """Reset rolling state and buffer."""
        self._tick_id = 0
        self._buf_pos = 0
        self._buf_filled = False
        self._chronos_buf[:] = 0.0
        for group in self._feature_groups:
            for ext in group.extractors:
                ext.reset()
        if self._profiler:
            self._profiler.reset()
            self._profiler.start()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------

def create_aftp(
    n_assets: int,
    t_ticks: int = 64,
    window: int = DEFAULT_WINDOW,
    mode: AFTPMode = AFTPMode.STREAMING,
    n_workers: int = 1,
    enable_profiler: bool = True,
    validate_output: bool = True,
    registry: Optional[UnifiedTensorRegistry] = None,
) -> AutomatedFeatureToPipeline:
    """Convenience factory for AFTP with common defaults."""
    cfg = AFTPConfig(
        n_assets=n_assets,
        t_ticks=t_ticks,
        window=window,
        mode=mode,
        n_workers=n_workers,
        enable_profiler=enable_profiler,
        validate_output=validate_output,
        registry=registry,
    )
    return AutomatedFeatureToPipeline(cfg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "DEFAULT_WINDOW",
    "COL_OPEN", "COL_HIGH", "COL_LOW", "COL_CLOSE", "COL_VOLUME", "COL_BID", "COL_ASK",
    "FEAT_BID", "FEAT_ASK", "FEAT_MID", "FEAT_SPREAD", "FEAT_VOLUME", "FEAT_IMBALANCE",
    # Data types
    "RawTick",
    "OHLCVFrame",
    # Rolling state
    "RollingState",
    # Feature extractors
    "FeatureExtractor",
    "ReturnsExtractor",
    "RollingVolatilityExtractor",
    "ParkinsonVolatilityExtractor",
    "GarmanKlassVolatilityExtractor",
    "BidAskFeatureExtractor",
    "OrderImbalanceExtractor",
    "VWAPDeviationExtractor",
    "RSIExtractor",
    "MACDExtractor",
    "BollingerBandsExtractor",
    # Feature group
    "FeatureGroupConfig",
    "build_default_feature_groups",
    # Profiler
    "ProfilerReport",
    "ThroughputProfiler",
    # Shared memory
    "ShmBufferSpec",
    "ShmRingBuffer",
    # Pipeline
    "AFTPMode",
    "AFTPConfig",
    "AutomatedFeatureToPipeline",
    "create_aftp",
]
