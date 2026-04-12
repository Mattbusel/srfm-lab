"""
AETERNUS Real-Time Execution Layer (RTEL)
data_pipeline.py — Data ingestion, normalization, and distribution pipeline

Handles:
- Multi-source data ingestion (simulated and real)
- Real-time normalization and validation
- Feature streaming to downstream consumers
- Data quality monitoring and anomaly detection
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RawTick:
    asset_id:   int
    timestamp:  float   # unix seconds
    bid:        float
    ask:        float
    bid_size:   float
    ask_size:   float
    last_price: float
    volume:     float

    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)

    def spread(self) -> float:
        return self.ask - self.bid

    def is_valid(self) -> bool:
        return (self.bid > 0 and self.ask > 0 and
                self.ask >= self.bid and
                self.bid_size >= 0 and self.ask_size >= 0)


@dataclass
class OHLCV:
    asset_id:  int
    timestamp: float
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float
    n_ticks:   int = 0

    def returns(self) -> float:
        if self.open > _EPS:
            return (self.close - self.open) / self.open
        return 0.0

    def high_low_range(self) -> float:
        return self.high - self.low

    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3.0


@dataclass
class NormalizedFeatures:
    asset_id:   int
    timestamp:  float
    features:   np.ndarray
    feature_names: List[str] = field(default_factory=list)
    quality_score: float = 1.0


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------

class DataValidator:
    """Validates incoming tick data for quality and anomalies."""

    def __init__(self,
                 max_spread_bps:   float = 100.0,
                 max_price_jump_pct: float = 5.0,
                 min_size:         float = 0.0):
        self.max_spread_bps     = max_spread_bps
        self.max_price_jump_pct = max_price_jump_pct
        self.min_size           = min_size
        self._last_prices: Dict[int, float] = {}
        self._n_rejected = 0
        self._n_accepted = 0

    def validate(self, tick: RawTick) -> Tuple[bool, str]:
        """Returns (is_valid, reason). reason is empty string if valid."""
        if not tick.is_valid():
            self._n_rejected += 1
            return False, "invalid_basic"

        mid = tick.mid()
        if mid <= 0:
            self._n_rejected += 1
            return False, "non_positive_price"

        # Spread check
        spread_bps = tick.spread() / mid * 1e4
        if spread_bps > self.max_spread_bps:
            self._n_rejected += 1
            return False, f"spread_too_wide_{spread_bps:.1f}bps"

        # Price jump check
        prev = self._last_prices.get(tick.asset_id)
        if prev is not None and prev > _EPS:
            jump_pct = abs(mid - prev) / prev * 100.0
            if jump_pct > self.max_price_jump_pct:
                self._n_rejected += 1
                return False, f"price_jump_{jump_pct:.1f}pct"

        self._last_prices[tick.asset_id] = mid
        self._n_accepted += 1
        return True, ""

    @property
    def rejection_rate(self) -> float:
        total = self._n_accepted + self._n_rejected
        return self._n_rejected / total if total > 0 else 0.0

    def reset(self) -> None:
        self._last_prices.clear()
        self._n_rejected = 0
        self._n_accepted = 0


# ---------------------------------------------------------------------------
# Bar aggregator
# ---------------------------------------------------------------------------

class BarAggregator:
    """Aggregates ticks into OHLCV bars."""

    def __init__(self, bar_duration_s: float = 1.0):
        self.bar_duration = bar_duration_s
        self._current_bars: Dict[int, dict] = {}
        self._completed_bars: deque = deque(maxlen=10000)

    def update(self, tick: RawTick) -> Optional[OHLCV]:
        """Update with tick; returns completed bar if bar closed."""
        aid = tick.asset_id
        mid = tick.mid()
        t   = tick.timestamp

        if aid not in self._current_bars:
            self._current_bars[aid] = {
                "open": mid, "high": mid, "low": mid, "close": mid,
                "volume": tick.volume, "n_ticks": 1,
                "bar_start": t - (t % self.bar_duration)
            }
            return None

        bar = self._current_bars[aid]
        bar_end = bar["bar_start"] + self.bar_duration

        if t >= bar_end:
            # Complete current bar
            completed = OHLCV(
                asset_id=aid,
                timestamp=bar["bar_start"],
                open=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
                volume=bar["volume"],
                n_ticks=bar["n_ticks"],
            )
            self._completed_bars.append(completed)
            # Start new bar
            self._current_bars[aid] = {
                "open": mid, "high": mid, "low": mid, "close": mid,
                "volume": tick.volume, "n_ticks": 1,
                "bar_start": t - (t % self.bar_duration)
            }
            return completed
        else:
            bar["high"]   = max(bar["high"], mid)
            bar["low"]    = min(bar["low"], mid)
            bar["close"]  = mid
            bar["volume"] += tick.volume
            bar["n_ticks"] += 1
            return None

    def recent_bars(self, asset_id: int, n: int) -> List[OHLCV]:
        return [b for b in list(self._completed_bars)[-n*2:]
                if b.asset_id == asset_id][-n:]


# ---------------------------------------------------------------------------
# Online feature normalizer (per-asset Welford)
# ---------------------------------------------------------------------------

class OnlineNormalizer:
    """Welford's algorithm for streaming feature normalization."""

    def __init__(self, n_features: int, clip_z: float = 4.0):
        self.n      = n_features
        self.clip_z = clip_z
        self._mean  = np.zeros(n_features)
        self._M2    = np.zeros(n_features)
        self._count = 0

    def update(self, x: np.ndarray) -> None:
        self._count += 1
        delta  = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._M2 += delta * delta2

    @property
    def variance(self) -> np.ndarray:
        if self._count < 2:
            return np.ones(self.n)
        return self._M2 / (self._count - 1)

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.variance, _EPS))

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self._count < 2:
            return np.zeros_like(x)
        z = (x - self._mean) / self.std
        return np.clip(z, -self.clip_z, self.clip_z)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        return z * self.std + self._mean

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.update(x)
        return self.transform(x)

    def reset(self) -> None:
        self._mean  = np.zeros(self.n)
        self._M2    = np.zeros(self.n)
        self._count = 0


# ---------------------------------------------------------------------------
# Anomaly detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Detects anomalies in feature streams using rolling statistics.
    Uses z-score and inter-quartile range.
    """

    def __init__(self, window: int = 100, z_threshold: float = 4.0):
        self.window      = window
        self.z_threshold = z_threshold
        self._history:   Dict[str, deque] = {}

    def update(self, feature_name: str, value: float) -> bool:
        """Returns True if anomaly detected."""
        if feature_name not in self._history:
            self._history[feature_name] = deque(maxlen=self.window)
        hist = self._history[feature_name]
        hist.append(value)
        if len(hist) < 10:
            return False

        vals = np.array(hist)
        mean = vals.mean()
        std  = vals.std()
        if std < _EPS:
            return False

        z = abs(value - mean) / std
        return z > self.z_threshold

    def check_batch(self, features: Dict[str, float]) -> Dict[str, bool]:
        return {k: self.update(k, v) for k, v in features.items()}

    def anomaly_count(self) -> Dict[str, int]:
        result = {}
        for name, hist in self._history.items():
            vals = np.array(hist)
            if len(vals) < 10:
                result[name] = 0
                continue
            mean = vals.mean()
            std  = vals.std()
            if std < _EPS:
                result[name] = 0
                continue
            z_scores = np.abs((vals - mean) / std)
            result[name] = int((z_scores > self.z_threshold).sum())
        return result


# ---------------------------------------------------------------------------
# Data pipeline handlers
# ---------------------------------------------------------------------------

Handler = Callable[[NormalizedFeatures], None]


class DataPipelineStage:
    """A processing stage in the data pipeline."""

    def __init__(self, name: str):
        self.name      = name
        self._latency  = deque(maxlen=1000)
        self._n_processed = 0
        self._n_errors    = 0

    def process(self, data):
        t0 = time.perf_counter()
        try:
            result = self._process(data)
            self._n_processed += 1
            return result
        except Exception as e:
            self._n_errors += 1
            logger.error("Pipeline stage %s error: %s", self.name, e)
            return None
        finally:
            self._latency.append((time.perf_counter() - t0) * 1e6)  # microseconds

    def _process(self, data):
        raise NotImplementedError

    @property
    def mean_latency_us(self) -> float:
        if not self._latency:
            return 0.0
        return float(np.mean(list(self._latency)))

    @property
    def p99_latency_us(self) -> float:
        if not self._latency:
            return 0.0
        return float(np.percentile(list(self._latency), 99))

    @property
    def error_rate(self) -> float:
        total = self._n_processed + self._n_errors
        return self._n_errors / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Feature extractor stage
# ---------------------------------------------------------------------------

class LOBFeatureStage(DataPipelineStage):
    """Extracts LOB features from raw ticks."""

    def __init__(self):
        super().__init__("lob_feature_extractor")
        self._lob_history: Dict[int, deque] = {}

    def _process(self, tick: RawTick) -> Optional[np.ndarray]:
        if not isinstance(tick, RawTick):
            return None
        aid = tick.asset_id
        mid = tick.mid()

        if aid not in self._lob_history:
            self._lob_history[aid] = deque(maxlen=20)
        hist = self._lob_history[aid]
        hist.append(mid)

        spread = tick.spread()
        imbal  = ((tick.bid_size - tick.ask_size) /
                  (tick.bid_size + tick.ask_size + _EPS))

        # Returns
        ret_1  = (mid / hist[-2] - 1.0) if len(hist) >= 2 else 0.0
        ret_5  = (mid / hist[-6] - 1.0) if len(hist) >= 6 else 0.0

        # Vol
        if len(hist) >= 5:
            prices = list(hist)[-5:]
            rets   = [math.log(prices[i+1]/prices[i]+_EPS) for i in range(len(prices)-1)
                      if prices[i] > _EPS]
            vol_5  = float(np.std(rets)) if rets else 0.0
        else:
            vol_5 = 0.0

        features = np.array([
            imbal,
            spread / (mid + _EPS),
            tick.bid_size / (tick.bid_size + tick.ask_size + _EPS),
            ret_1,
            ret_5,
            vol_5,
            tick.volume,
            mid,
        ], dtype=np.float32)

        return features


# ---------------------------------------------------------------------------
# Normalization stage
# ---------------------------------------------------------------------------

class NormalizationStage(DataPipelineStage):
    """Normalizes features per-asset using online Welford stats."""

    def __init__(self, n_features: int):
        super().__init__("normalizer")
        self._normalizers: Dict[int, OnlineNormalizer] = {}
        self.n_features = n_features

    def _process(self, item: Tuple[int, np.ndarray]) -> Optional[np.ndarray]:
        asset_id, features = item
        if asset_id not in self._normalizers:
            self._normalizers[asset_id] = OnlineNormalizer(self.n_features)
        return self._normalizers[asset_id].fit_transform(features)


# ---------------------------------------------------------------------------
# Main DataPipeline
# ---------------------------------------------------------------------------

class DataPipeline:
    """
    Orchestrates the full data processing pipeline:
    Tick → Validate → Aggregate → Feature Extract → Normalize → Distribute
    """

    def __init__(self, n_assets: int, bar_duration_s: float = 1.0,
                 n_lob_features: int = 8):
        self.n_assets      = n_assets
        self.n_lob_features = n_lob_features

        self.validator     = DataValidator()
        self.bar_agg       = BarAggregator(bar_duration_s)
        self.lob_stage     = LOBFeatureStage()
        self.norm_stage    = NormalizationStage(n_lob_features)
        self.anomaly_det   = AnomalyDetector()

        self._handlers:    List[Handler] = []
        self._n_ticks      = 0
        self._n_rejected   = 0
        self._n_features   = 0

        # Per-asset feature history for sequence models
        self._feature_seq: Dict[int, deque] = {}

    def add_handler(self, handler: Handler) -> None:
        self._handlers.append(handler)

    def process_tick(self, tick: RawTick) -> bool:
        """Process a single tick through the pipeline. Returns True if successful."""
        self._n_ticks += 1

        # Validate
        is_valid, reason = self.validator.validate(tick)
        if not is_valid:
            self._n_rejected += 1
            return False

        # Extract LOB features
        features = self.lob_stage.process(tick)
        if features is None:
            return False

        # Normalize
        normalized = self.norm_stage.process((tick.asset_id, features))
        if normalized is None:
            return False

        # Anomaly detection
        feat_dict = {f"feat_{i}": float(normalized[i])
                     for i in range(len(normalized))}
        anomalies = self.anomaly_det.check_batch(feat_dict)
        quality   = 1.0 - sum(anomalies.values()) / max(1, len(anomalies))

        # Update sequence history
        if tick.asset_id not in self._feature_seq:
            self._feature_seq[tick.asset_id] = deque(maxlen=64)
        self._feature_seq[tick.asset_id].append(normalized)

        # Build output
        output = NormalizedFeatures(
            asset_id      = tick.asset_id,
            timestamp     = tick.timestamp,
            features      = normalized,
            feature_names = [f"feat_{i}" for i in range(len(normalized))],
            quality_score = quality,
        )

        # Dispatch to handlers
        for handler in self._handlers:
            try:
                handler(output)
            except Exception as e:
                logger.warning("Handler error: %s", e)

        self._n_features += 1
        return True

    def process_batch(self, ticks: List[RawTick]) -> int:
        """Process a batch of ticks. Returns number successfully processed."""
        return sum(1 for t in ticks if self.process_tick(t))

    def get_feature_sequence(self, asset_id: int, seq_len: int) -> Optional[np.ndarray]:
        """Get recent feature sequence as [seq_len × n_features] array."""
        if asset_id not in self._feature_seq:
            return None
        hist = list(self._feature_seq[asset_id])
        n = len(hist)
        if n == 0:
            return None
        result = np.zeros((seq_len, self.n_lob_features), dtype=np.float32)
        fill_n = min(n, seq_len)
        for i in range(fill_n):
            result[seq_len - fill_n + i] = hist[n - fill_n + i]
        return result

    def stats(self) -> dict:
        return {
            "n_ticks":        self._n_ticks,
            "n_rejected":     self._n_rejected,
            "n_features":     self._n_features,
            "rejection_rate": self.validator.rejection_rate,
            "lob_latency_us": self.lob_stage.mean_latency_us,
            "norm_latency_us": self.norm_stage.mean_latency_us,
        }


# ---------------------------------------------------------------------------
# Synthetic data source
# ---------------------------------------------------------------------------

class SyntheticDataSource:
    """
    Generates synthetic tick data with GBM prices and
    realistic LOB dynamics for testing.
    """

    class Asset:
        def __init__(self, asset_id: int, initial_price: float,
                     mu: float = 0.0, sigma: float = 0.01,
                     spread_bps: float = 5.0, tick_rate: float = 10.0):
            self.asset_id      = asset_id
            self.price         = initial_price
            self.mu            = mu
            self.sigma         = sigma
            self.spread_bps    = spread_bps
            self.tick_rate     = tick_rate
            self._vol          = sigma  # current vol (GARCH-like)
            self._rng          = np.random.default_rng(asset_id + 42)
            self._t            = 0.0

        def next_tick(self, dt: float = 1.0/252.0/6.5/3600.0) -> RawTick:
            """Generate next tick. dt = time step in years."""
            # GARCH-like vol
            z = self._rng.standard_normal()
            self._vol = 0.9 * self._vol + 0.1 * abs(self.sigma * z)
            self._vol = max(self._vol, self.sigma * 0.1)

            # GBM step
            drift = (self.mu - 0.5 * self._vol**2) * dt
            shock = self._vol * math.sqrt(dt) * z
            self.price *= math.exp(drift + shock)

            spread_half = self.price * self.spread_bps / 2e4
            bid_size = float(abs(self._rng.normal(1000.0, 200.0)))
            ask_size = float(abs(self._rng.normal(1000.0, 200.0)))
            volume   = float(abs(self._rng.normal(500.0, 100.0)))

            self._t += dt
            return RawTick(
                asset_id   = self.asset_id,
                timestamp  = self._t * 252.0 * 6.5 * 3600.0,  # approx unix seconds
                bid        = self.price - spread_half,
                ask        = self.price + spread_half,
                bid_size   = bid_size,
                ask_size   = ask_size,
                last_price = self.price,
                volume     = volume,
            )

    def __init__(self, n_assets: int,
                 initial_price: float = 100.0,
                 sigma: float = 0.01):
        self.assets = [
            self.Asset(
                i,
                initial_price * (1.0 + 0.1 * i),
                mu    = 0.05 / 252 / 6.5 / 3600,
                sigma = sigma * (1.0 + 0.2 * (i % 3)),
            )
            for i in range(n_assets)
        ]

    def next_ticks(self) -> List[RawTick]:
        return [a.next_tick() for a in self.assets]

    def generate(self, n_steps: int) -> List[List[RawTick]]:
        return [self.next_ticks() for _ in range(n_steps)]


# ---------------------------------------------------------------------------
# Metrics reporter for the data pipeline
# ---------------------------------------------------------------------------

class PipelineMetricsReporter:
    def __init__(self, pipeline: DataPipeline):
        self.pipeline = pipeline

    def prometheus_metrics(self) -> str:
        s = self.pipeline.stats()
        return (
            f"rtel_pipeline_ticks_total {s['n_ticks']}\n"
            f"rtel_pipeline_rejected_total {s['n_rejected']}\n"
            f"rtel_pipeline_features_total {s['n_features']}\n"
            f"rtel_pipeline_rejection_rate {s['rejection_rate']:.6f}\n"
            f"rtel_pipeline_lob_latency_us {s['lob_latency_us']:.2f}\n"
            f"rtel_pipeline_norm_latency_us {s['norm_latency_us']:.2f}\n"
        )
