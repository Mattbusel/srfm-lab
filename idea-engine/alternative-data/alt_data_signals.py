"""
alt_data_signals.py
===================
Alternative data signal processor for the idea-engine.

Converts raw alternative data sources into normalised trading signals:
satellite imagery proxies, web traffic, app downloads, credit card
transactions, job postings, patent filings, supply chain data, weather
anomalies, and geolocation foot traffic.
"""

from __future__ import annotations

import logging
import math
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core structures
# ---------------------------------------------------------------------------

class SignalQuality(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    STALE = "stale"


@dataclass
class AltDataPoint:
    timestamp: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    source: str = ""
    raw_value: float = 0.0
    normalized_value: float = 0.0
    z_score: float = 0.0
    quality: SignalQuality = SignalQuality.MEDIUM
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositeSignal:
    timestamp: datetime = field(default_factory=datetime.utcnow)
    symbol: str = ""
    value: float = 0.0
    components: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    n_sources: int = 0


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class SignalNormalizer:
    """Rolling z-score normalisation with decay."""

    def __init__(self, window: int = 60, min_periods: int = 10):
        self.window = window
        self.min_periods = min_periods
        self._history: Dict[str, Deque[float]] = {}

    def update(self, key: str, value: float) -> float:
        if key not in self._history:
            self._history[key] = deque(maxlen=self.window)
        self._history[key].append(value)
        buf = self._history[key]
        if len(buf) < self.min_periods:
            return 0.0
        arr = np.array(list(buf))
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1))
        if sigma < 1e-15:
            return 0.0
        return (value - mu) / sigma

    def percentile_rank(self, key: str, value: float) -> float:
        buf = self._history.get(key)
        if not buf or len(buf) < self.min_periods:
            return 0.5
        arr = np.array(list(buf))
        return float(np.sum(arr <= value)) / len(arr)

    def reset(self, key: Optional[str] = None) -> None:
        if key:
            self._history.pop(key, None)
        else:
            self._history.clear()


# ---------------------------------------------------------------------------
# Base signal class
# ---------------------------------------------------------------------------

class AltDataSignal(ABC):
    """Base class for all alternative data signals."""

    def __init__(self, name: str, weight: float = 1.0, lookback: int = 60):
        self.name = name
        self.weight = weight
        self.lookback = lookback
        self.normalizer = SignalNormalizer(window=lookback)
        self._raw_history: Deque[Tuple[datetime, float]] = deque(maxlen=lookback * 2)

    @abstractmethod
    def process_raw(self, raw_data: Dict[str, Any]) -> float:
        """Convert raw data to a single numeric value."""
        ...

    def ingest(self, raw_data: Dict[str, Any], timestamp: Optional[datetime] = None) -> AltDataPoint:
        ts = timestamp or datetime.utcnow()
        raw_val = self.process_raw(raw_data)
        self._raw_history.append((ts, raw_val))
        z = self.normalizer.update(self.name, raw_val)
        quality = self._assess_quality(raw_data, ts)
        return AltDataPoint(
            timestamp=ts, symbol=raw_data.get("symbol", ""),
            source=self.name, raw_value=raw_val,
            normalized_value=z, z_score=z, quality=quality,
        )

    def _assess_quality(self, raw_data: Dict[str, Any], ts: datetime) -> SignalQuality:
        if not self._raw_history:
            return SignalQuality.LOW
        last_ts = self._raw_history[-1][0]
        staleness = (ts - last_ts).total_seconds()
        if staleness > 86400 * 3:
            return SignalQuality.STALE
        n = len(self._raw_history)
        if n >= self.lookback:
            return SignalQuality.HIGH
        if n >= self.lookback // 2:
            return SignalQuality.MEDIUM
        return SignalQuality.LOW

    def recent_values(self, n: int = 20) -> np.ndarray:
        vals = [v for _, v in list(self._raw_history)[-n:]]
        return np.array(vals) if vals else np.array([])

    def trend(self, n: int = 10) -> float:
        vals = self.recent_values(n)
        if len(vals) < 3:
            return 0.0
        x = np.arange(len(vals), dtype=float)
        x -= np.mean(x)
        vals_m = vals - np.mean(vals)
        denom = np.sum(x ** 2)
        return float(np.sum(x * vals_m) / denom) if denom > 1e-15 else 0.0


# ---------------------------------------------------------------------------
# Satellite data proxy
# ---------------------------------------------------------------------------

class SatelliteDataSignal(AltDataSignal):
    """Parking lot occupancy as a proxy for retail sales."""

    def __init__(self, weight: float = 1.0, lookback: int = 60):
        super().__init__("satellite_parking_lot", weight, lookback)
        self._seasonal_avg: Dict[int, float] = {}

    def process_raw(self, raw_data: Dict[str, Any]) -> float:
        occupancy = raw_data.get("occupancy_pct", 0.0)
        capacity = raw_data.get("total_spots", 1000)
        cars = raw_data.get("car_count", occupancy * capacity / 100)
        normalized = cars / max(capacity, 1) * 100
        # Seasonal adjustment
        month = raw_data.get("month", datetime.utcnow().month)
        seasonal = self._seasonal_avg.get(month, normalized)
        return normalized - seasonal if seasonal > 0 else normalized

    def set_seasonal_baselines(self, baselines: Dict[int, float]) -> None:
        self._seasonal_avg = dict(baselines)

    def yoy_change(self) -> float:
        vals = self.recent_values(52)
        if len(vals) < 52:
            return 0.0
        current = float(np.mean(vals[-4:]))
        prior = float(np.mean(vals[:4]))
        return (current - prior) / abs(prior) if abs(prior) > 1e-10 else 0.0


# ---------------------------------------------------------------------------
# Web traffic signal
# ---------------------------------------------------------------------------

class WebTrafficSignal(AltDataSignal):
    """Company web visits as revenue proxy."""

    def __init__(self, weight: float = 1.0, lookback: int = 60):
        super().__init__("web_traffic", weight, lookback)

    def process_raw(self, raw_data: Dict[str, Any]) -> float:
        visits = raw_data.get("unique_visitors", 0)
        page_views = raw_data.get("page_views", 0)
        avg_duration = raw_data.get("avg_session_duration_sec", 0)
        bounce_rate = raw_data.get("bounce_rate", 0.5)
        engagement = visits * (1 - bounce_rate) * (avg_duration / 60.0)
        return engagement + page_views * 0.1

    def growth_rate(self, periods: int = 4) -> float:
        vals = self.recent_values(periods * 2)
        if len(vals) < periods * 2:
            return 0.0
        recent = float(np.mean(vals[-periods:]))
        prior = float(np.mean(vals[:periods]))
        return (recent - prior) / abs(prior) if abs(prior) > 1e-10 else 0.0


# ---------------------------------------------------------------------------
# App download signal
# ---------------------------------------------------------------------------

class AppDownloadSignal(AltDataSignal):
    """Mobile app rankings as growth proxy."""

    def __init__(self, weight: float = 1.0, lookback: int = 60):
        super().__init__("app_downloads", weight, lookback)

    def process_raw(self, raw_data: Dict[str, Any]) -> float:
        downloads = raw_data.get("daily_downloads", 0)
        rank = raw_data.get("store_rank", 500)
        ratings = raw_data.get("avg_rating", 3.0)
        reviews = raw_data.get("new_reviews", 0)
        # Rank signal is inverse (lower rank = better)
        rank_signal = max(0, 1000 - rank) / 1000.0
        return downloads * 0.5 + rank_signal * 300 + ratings * 20 + reviews * 0.3

    def rank_momentum(self, periods: int = 7) -> float:
        vals = self.recent_values(periods)
        if len(vals) < 3:
            return 0.0
        return self.trend(periods)


# ---------------------------------------------------------------------------
# Credit card transaction signal
# ---------------------------------------------------------------------------

class CreditCardSignal(AltDataSignal):
    """Consumer spending trend from aggregated transaction data."""

    def __init__(self, weight: float = 1.2, lookback: int = 60):
        super().__init__("credit_card_txn", weight, lookback)

    def process_raw(self, raw_data: Dict[str, Any]) -> float:
        total_spend = raw_data.get("total_spend", 0.0)
        txn_count = raw_data.get("transaction_count", 0)
        avg_ticket = raw_data.get("avg_ticket_size", 0.0)
        online_pct = raw_data.get("online_pct", 0.5)
        # Composite: total spend weighted by online shift
        return total_spend * (1 + online_pct * 0.2) + txn_count * avg_ticket * 0.01

    def spending_acceleration(self, short: int = 4, long: int = 12) -> float:
        vals = self.recent_values(long)
        if len(vals) < long:
            return 0.0
        short_avg = float(np.mean(vals[-short:]))
        long_avg = float(np.mean(vals))
        return (short_avg - long_avg) / abs(long_avg) if abs(long_avg) > 1e-10 else 0.0


# ---------------------------------------------------------------------------
# Job posting signal
# ---------------------------------------------------------------------------

class JobPostingSignal(AltDataSignal):
    """Hiring momentum as growth indicator."""

    def __init__(self, weight: float = 0.8, lookback: int = 90):
        super().__init__("job_postings", weight, lookback)

    def process_raw(self, raw_data: Dict[str, Any]) -> float:
        new_postings = raw_data.get("new_postings", 0)
        removed = raw_data.get("removed_postings", 0)
        seniority_mix = raw_data.get("senior_pct", 0.3)
        engineering_pct = raw_data.get("engineering_pct", 0.2)
        net = new_postings - removed
        # Weight engineering and senior roles higher
        quality_adj = 1 + engineering_pct * 0.5 + seniority_mix * 0.3
        return net * quality_adj

    def hiring_momentum(self, periods: int = 4) -> float:
        return self.trend(periods)


# ---------------------------------------------------------------------------
# Patent filing signal
# ---------------------------------------------------------------------------

class PatentFilingSignal(AltDataSignal):
    """Innovation intensity from patent filings."""

    def __init__(self, weight: float = 0.6, lookback: int = 120):
        super().__init__("patent_filings", weight, lookback)

    def process_raw(self, raw_data: Dict[str, Any]) -> float:
        filings = raw_data.get("new_filings", 0)
        grants = raw_data.get("grants", 0)
        citations = raw_data.get("forward_citations", 0)
        categories = raw_data.get("n_categories", 1)
        # Weighted composite
        return filings * 1.0 + grants * 2.0 + citations * 0.5 + categories * 0.3

    def innovation_score(self) -> float:
        vals = self.recent_values(12)
        if len(vals) < 3:
            return 0.0
        return float(np.mean(vals)) * self.weight


# ---------------------------------------------------------------------------
# Supply chain signal
# ---------------------------------------------------------------------------

class SupplyChainSignal(AltDataSignal):
    """Shipping and logistics data as economic activity proxy."""

    def __init__(self, weight: float = 1.0, lookback: int = 60):
        super().__init__("supply_chain", weight, lookback)

    def process_raw(self, raw_data: Dict[str, Any]) -> float:
        shipping_volume = raw_data.get("shipping_volume", 0)
        port_congestion = raw_data.get("port_congestion_days", 0)
        freight_rate = raw_data.get("freight_rate_index", 100)
        container_throughput = raw_data.get("container_throughput", 0)
        inventory_ratio = raw_data.get("inventory_to_sales", 1.0)
        # High shipping + low congestion = positive
        activity = shipping_volume * 0.4 + container_throughput * 0.3
        friction = port_congestion * 10 + max(0, freight_rate - 100) * 0.5
        inv_adj = max(0, 1.5 - inventory_ratio) * 50  # low inventory = demand
        return activity - friction + inv_adj

    def bottleneck_score(self) -> float:
        """Higher = more supply chain stress."""
        vals = self.recent_values(10)
        if len(vals) < 3:
            return 0.0
        # Negative trend = worsening
        return -self.trend(10)


# ---------------------------------------------------------------------------
# Weather impact signal
# ---------------------------------------------------------------------------

class WeatherImpactSignal(AltDataSignal):
    """Temperature/precipitation anomaly on commodity demand."""

    def __init__(self, weight: float = 0.7, lookback: int = 90):
        super().__init__("weather_impact", weight, lookback)
        self._climate_normals: Dict[int, Dict[str, float]] = {}

    def set_climate_normals(self, normals: Dict[int, Dict[str, float]]) -> None:
        """normals: {month: {temp_c: ..., precip_mm: ...}}"""
        self._climate_normals = normals

    def process_raw(self, raw_data: Dict[str, Any]) -> float:
        temp_c = raw_data.get("temperature_c", 20.0)
        precip_mm = raw_data.get("precipitation_mm", 0.0)
        month = raw_data.get("month", datetime.utcnow().month)
        normal = self._climate_normals.get(month, {"temp_c": 20.0, "precip_mm": 50.0})
        temp_anomaly = temp_c - normal.get("temp_c", 20.0)
        precip_anomaly = precip_mm - normal.get("precip_mm", 50.0)
        commodity = raw_data.get("commodity", "general")
        # Temperature extremes increase energy demand
        energy_impact = abs(temp_anomaly) * 0.5 if abs(temp_anomaly) > 5 else 0
        # Precipitation anomalies affect agriculture
        ag_impact = -precip_anomaly * 0.1 if commodity in ("corn", "wheat", "soy") else 0
        return temp_anomaly * 0.3 + energy_impact + ag_impact

    def extreme_weather_flag(self, threshold_std: float = 2.0) -> bool:
        vals = self.recent_values(30)
        if len(vals) < 10:
            return False
        z = abs(float(vals[-1]) - float(np.mean(vals))) / max(float(np.std(vals)), 1e-10)
        return z > threshold_std


# ---------------------------------------------------------------------------
# Geolocation foot traffic signal
# ---------------------------------------------------------------------------

class GeolocationSignal(AltDataSignal):
    """Foot traffic to retail locations."""

    def __init__(self, weight: float = 1.0, lookback: int = 60):
        super().__init__("geolocation_traffic", weight, lookback)

    def process_raw(self, raw_data: Dict[str, Any]) -> float:
        foot_traffic = raw_data.get("daily_visits", 0)
        dwell_time_min = raw_data.get("avg_dwell_time_min", 0)
        unique_devices = raw_data.get("unique_devices", 0)
        repeat_rate = raw_data.get("repeat_visit_rate", 0.3)
        # Engaged traffic: longer dwell + repeat visitors
        engagement = foot_traffic * (dwell_time_min / 30.0) * (1 + repeat_rate)
        device_coverage = unique_devices / max(foot_traffic, 1)
        return engagement * device_coverage * 100

    def wow_change(self, day_window: int = 7) -> float:
        vals = self.recent_values(day_window * 2)
        if len(vals) < day_window * 2:
            return 0.0
        recent = float(np.mean(vals[-day_window:]))
        prior = float(np.mean(vals[:day_window]))
        return (recent - prior) / abs(prior) if abs(prior) > 1e-10 else 0.0


# ---------------------------------------------------------------------------
# Alt data signal engine (master aggregator)
# ---------------------------------------------------------------------------

class AltDataSignalEngine:
    """Normalize, combine, and produce composite signals from all alt data."""

    def __init__(self) -> None:
        self.signals: Dict[str, AltDataSignal] = {}
        self._composite_history: Dict[str, Deque[CompositeSignal]] = defaultdict(lambda: deque(maxlen=500))
        self._weight_override: Dict[str, float] = {}

    def register(self, signal: AltDataSignal) -> None:
        self.signals[signal.name] = signal
        logger.info("Registered alt data signal: %s (weight=%.2f)", signal.name, signal.weight)

    def register_defaults(self) -> None:
        """Register all built-in signal types."""
        self.register(SatelliteDataSignal())
        self.register(WebTrafficSignal())
        self.register(AppDownloadSignal())
        self.register(CreditCardSignal())
        self.register(JobPostingSignal())
        self.register(PatentFilingSignal())
        self.register(SupplyChainSignal())
        self.register(WeatherImpactSignal())
        self.register(GeolocationSignal())

    def set_weight(self, name: str, weight: float) -> None:
        self._weight_override[name] = weight
        if name in self.signals:
            self.signals[name].weight = weight

    def ingest(
        self, source_name: str, raw_data: Dict[str, Any], timestamp: Optional[datetime] = None
    ) -> Optional[AltDataPoint]:
        sig = self.signals.get(source_name)
        if not sig:
            logger.warning("Unknown signal source: %s", source_name)
            return None
        return sig.ingest(raw_data, timestamp)

    def ingest_batch(
        self, data: List[Tuple[str, Dict[str, Any]]], timestamp: Optional[datetime] = None
    ) -> List[AltDataPoint]:
        results: List[AltDataPoint] = []
        for source, raw in data:
            pt = self.ingest(source, raw, timestamp)
            if pt:
                results.append(pt)
        return results

    def compute_composite(
        self,
        symbol: str,
        data_points: Optional[List[AltDataPoint]] = None,
        timestamp: Optional[datetime] = None,
    ) -> CompositeSignal:
        """Weighted sum of z-scores from all available signals for a symbol."""
        ts = timestamp or datetime.utcnow()
        components: Dict[str, float] = {}
        total_weight = 0.0
        weighted_sum = 0.0

        for name, sig in self.signals.items():
            vals = sig.recent_values(1)
            if len(vals) == 0:
                continue
            z = sig.normalizer.update(name, float(vals[-1]))
            w = self._weight_override.get(name, sig.weight)
            components[name] = z
            weighted_sum += z * w
            total_weight += abs(w)

        value = weighted_sum / total_weight if total_weight > 1e-15 else 0.0
        n_sources = len(components)
        confidence = min(n_sources / max(len(self.signals), 1), 1.0)

        comp = CompositeSignal(
            timestamp=ts, symbol=symbol, value=value,
            components=components, confidence=confidence, n_sources=n_sources,
        )
        self._composite_history[symbol].append(comp)
        return comp

    def composite_trend(self, symbol: str, n: int = 10) -> float:
        history = list(self._composite_history.get(symbol, []))
        if len(history) < 3:
            return 0.0
        vals = np.array([c.value for c in history[-n:]])
        x = np.arange(len(vals), dtype=float)
        x -= np.mean(x)
        v = vals - np.mean(vals)
        d = np.sum(x ** 2)
        return float(np.sum(x * v) / d) if d > 1e-15 else 0.0

    def signal_health(self) -> Dict[str, Dict[str, Any]]:
        """Report on each signal's data quality and recency."""
        report: Dict[str, Dict[str, Any]] = {}
        for name, sig in self.signals.items():
            vals = sig.recent_values(sig.lookback)
            last_ts = sig._raw_history[-1][0] if sig._raw_history else None
            report[name] = {
                "n_observations": len(vals),
                "coverage_pct": len(vals) / sig.lookback * 100,
                "last_timestamp": last_ts.isoformat() if last_ts else None,
                "current_trend": sig.trend(),
                "weight": self._weight_override.get(name, sig.weight),
            }
        return report

    def correlation_matrix(self) -> Tuple[List[str], np.ndarray]:
        """Compute pairwise correlation of signal z-scores."""
        names = sorted(self.signals.keys())
        n = len(names)
        mat = np.eye(n)
        histories: Dict[str, np.ndarray] = {}
        for name in names:
            histories[name] = self.signals[name].recent_values(60)
        min_len = min((len(v) for v in histories.values()), default=0)
        if min_len < 5:
            return names, mat
        for i in range(n):
            for j in range(i + 1, n):
                a = histories[names[i]][:min_len]
                b = histories[names[j]][:min_len]
                am = a - np.mean(a)
                bm = b - np.mean(b)
                d = np.sqrt(np.sum(am ** 2) * np.sum(bm ** 2))
                c = float(np.sum(am * bm) / d) if d > 1e-15 else 0.0
                mat[i, j] = c
                mat[j, i] = c
        return names, mat

    def top_signals(self, symbol: str, n: int = 3) -> List[Tuple[str, float]]:
        comp = self.compute_composite(symbol)
        ranked = sorted(comp.components.items(), key=lambda x: abs(x[1]), reverse=True)
        return ranked[:n]

    def summary(self) -> Dict[str, Any]:
        return {
            "n_signals": len(self.signals),
            "registered": list(self.signals.keys()),
            "health": self.signal_health(),
        }
