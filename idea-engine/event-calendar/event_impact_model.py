"""
event_impact_model.py
=====================
Event calendar and impact modeling for the idea-engine.

Models pre-event behaviour, post-event surprise impact, event clustering,
seasonal patterns, FOMC drift, earnings straddles, event-conditional signal
blending, and forward-looking calendar construction.
"""

from __future__ import annotations

import logging
import math
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums & data structures
# ---------------------------------------------------------------------------

class EventType(Enum):
    EARNINGS = "earnings"
    FOMC = "fomc"
    CPI = "cpi"
    NFP = "nfp"
    GDP = "gdp"
    PMI = "pmi"
    OPTIONS_EXPIRY = "options_expiry"
    INDEX_REBALANCE = "index_rebalance"
    ECB = "ecb"
    BOJ = "boj"
    RETAIL_SALES = "retail_sales"
    HOUSING = "housing"
    ISM = "ism"
    CUSTOM = "custom"


class ImpactMagnitude(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class MarketEvent:
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    event_type: EventType = EventType.CUSTOM
    symbol: str = ""
    event_datetime: datetime = field(default_factory=datetime.utcnow)
    expected_impact: ImpactMagnitude = ImpactMagnitude.MEDIUM
    actual_impact: Optional[float] = None
    consensus: Optional[float] = None
    actual_value: Optional[float] = None
    surprise: Optional[float] = None
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.actual_value is not None and self.consensus is not None:
            self.surprise = self.actual_value - self.consensus

    @property
    def is_upcoming(self) -> bool:
        return self.event_datetime > datetime.utcnow()

    @property
    def surprise_magnitude(self) -> float:
        return abs(self.surprise) if self.surprise is not None else 0.0


@dataclass
class SeasonalPattern:
    name: str = ""
    day_of_week_effects: Dict[int, float] = field(default_factory=dict)   # 0=Mon..6=Sun
    month_effects: Dict[int, float] = field(default_factory=dict)         # 1-12
    turn_of_month: float = 0.0    # effect around month boundaries
    quarter_end: float = 0.0
    holiday_effect: float = 0.0


@dataclass
class EventImpactResult:
    event: MarketEvent = field(default_factory=MarketEvent)
    pre_event_vol_ratio: float = 1.0
    post_event_return: float = 0.0
    surprise_impact: float = 0.0
    model_predicted_impact: float = 0.0
    signal_adjustment: float = 0.0


# ---------------------------------------------------------------------------
# Impact sensitivity by event type
# ---------------------------------------------------------------------------

DEFAULT_SENSITIVITIES: Dict[EventType, float] = {
    EventType.FOMC: 2.5,
    EventType.CPI: 2.0,
    EventType.NFP: 1.8,
    EventType.GDP: 1.5,
    EventType.PMI: 1.2,
    EventType.EARNINGS: 3.0,
    EventType.OPTIONS_EXPIRY: 0.8,
    EventType.INDEX_REBALANCE: 0.6,
    EventType.ECB: 2.0,
    EventType.BOJ: 1.5,
    EventType.RETAIL_SALES: 1.0,
    EventType.HOUSING: 0.7,
    EventType.ISM: 1.1,
    EventType.CUSTOM: 1.0,
}


# ---------------------------------------------------------------------------
# Pre-event model
# ---------------------------------------------------------------------------

class PreEventModel:
    """Model typical vol compression and positioning before events."""

    def __init__(self, lookback_days: int = 5, vol_compression_factor: float = 0.85):
        self.lookback_days = lookback_days
        self.vol_compression_factor = vol_compression_factor

    def vol_ratio(
        self,
        returns: np.ndarray,
        event_indices: List[int],
        window_before: int = 5,
    ) -> Dict[str, float]:
        """Compute avg vol ratio (pre-event / normal) across events."""
        if len(returns) < window_before * 2 or not event_indices:
            return {"avg_vol_ratio": 1.0, "n_events": 0}
        normal_vol = float(np.std(returns))
        if normal_vol < 1e-15:
            return {"avg_vol_ratio": 1.0, "n_events": 0}
        ratios: List[float] = []
        for idx in event_indices:
            start = max(0, idx - window_before)
            if start >= idx:
                continue
            pre_vol = float(np.std(returns[start:idx]))
            ratios.append(pre_vol / normal_vol)
        avg = float(np.mean(ratios)) if ratios else 1.0
        return {"avg_vol_ratio": avg, "n_events": len(ratios)}

    def position_adjustment(self, event_type: EventType) -> float:
        """Suggest position size multiplier before an event."""
        sensitivity = DEFAULT_SENSITIVITIES.get(event_type, 1.0)
        # Higher sensitivity -> reduce position more
        return max(0.2, 1.0 - 0.15 * sensitivity)

    def expected_vol_expansion(self, event_type: EventType) -> float:
        sensitivity = DEFAULT_SENSITIVITIES.get(event_type, 1.0)
        return 1.0 + 0.3 * sensitivity


# ---------------------------------------------------------------------------
# Post-event surprise impact model
# ---------------------------------------------------------------------------

class PostEventImpactModel:
    """Surprise-based impact: (actual - consensus) * sensitivity."""

    def __init__(self, sensitivities: Optional[Dict[EventType, float]] = None):
        self.sensitivities = sensitivities or dict(DEFAULT_SENSITIVITIES)

    def compute_impact(self, event: MarketEvent) -> float:
        if event.surprise is None:
            return 0.0
        sens = self.sensitivities.get(event.event_type, 1.0)
        return event.surprise * sens

    def calibrate(
        self,
        events: List[MarketEvent],
        realized_returns: List[float],
    ) -> Dict[EventType, float]:
        """Calibrate sensitivity per event type from historical data."""
        by_type: Dict[EventType, List[Tuple[float, float]]] = defaultdict(list)
        for evt, ret in zip(events, realized_returns):
            if evt.surprise is not None:
                by_type[evt.event_type].append((evt.surprise, ret))
        calibrated: Dict[EventType, float] = {}
        for etype, pairs in by_type.items():
            if len(pairs) < 5:
                calibrated[etype] = self.sensitivities.get(etype, 1.0)
                continue
            surprises = np.array([p[0] for p in pairs])
            returns = np.array([p[1] for p in pairs])
            ss = np.sum(surprises ** 2)
            if ss < 1e-15:
                calibrated[etype] = 1.0
            else:
                calibrated[etype] = float(np.sum(surprises * returns) / ss)
        self.sensitivities.update(calibrated)
        return calibrated

    def decay_curve(self, event: MarketEvent, horizon_bars: int = 20) -> np.ndarray:
        """Model how surprise impact decays over time (exponential decay)."""
        impact = self.compute_impact(event)
        decay_rate = 0.15
        return np.array([impact * math.exp(-decay_rate * t) for t in range(horizon_bars)])


# ---------------------------------------------------------------------------
# Event clustering
# ---------------------------------------------------------------------------

class EventClusterAnalyzer:
    """When multiple events coincide, compound the impact."""

    def __init__(self, cluster_window_hours: int = 24):
        self.cluster_window_hours = cluster_window_hours

    def find_clusters(self, events: List[MarketEvent]) -> List[List[MarketEvent]]:
        if not events:
            return []
        sorted_events = sorted(events, key=lambda e: e.event_datetime)
        clusters: List[List[MarketEvent]] = [[sorted_events[0]]]
        for evt in sorted_events[1:]:
            last = clusters[-1][-1]
            delta = (evt.event_datetime - last.event_datetime).total_seconds() / 3600
            if delta <= self.cluster_window_hours:
                clusters[-1].append(evt)
            else:
                clusters.append([evt])
        return clusters

    def compound_impact(self, cluster: List[MarketEvent], impact_model: PostEventImpactModel) -> float:
        """Compound impact: sum of individual impacts * cluster multiplier."""
        if not cluster:
            return 0.0
        individual = [impact_model.compute_impact(e) for e in cluster]
        base = sum(individual)
        # Interaction effect: slight amplification when events cluster
        multiplier = 1.0 + 0.1 * (len(cluster) - 1)
        return base * multiplier

    def cluster_vol_multiplier(self, cluster: List[MarketEvent]) -> float:
        """Expected vol multiplier from a cluster of events."""
        if not cluster:
            return 1.0
        sensitivities = [DEFAULT_SENSITIVITIES.get(e.event_type, 1.0) for e in cluster]
        # Root-sum-of-squares for vol
        return math.sqrt(sum(s ** 2 for s in sensitivities))


# ---------------------------------------------------------------------------
# Seasonal patterns
# ---------------------------------------------------------------------------

class SeasonalPatternModel:
    """Day-of-week, month-of-year, turn-of-month effects."""

    def __init__(self) -> None:
        self.patterns: Dict[str, SeasonalPattern] = {}

    def fit(self, dates: List[datetime], returns: np.ndarray, name: str = "default") -> SeasonalPattern:
        n = min(len(dates), len(returns))
        dates, returns_arr = dates[:n], returns[:n]
        # Day of week
        dow_returns: Dict[int, List[float]] = defaultdict(list)
        month_returns: Dict[int, List[float]] = defaultdict(list)
        tom_returns: List[float] = []
        other_returns: List[float] = []
        for i, dt in enumerate(dates):
            dow_returns[dt.weekday()].append(float(returns_arr[i]))
            month_returns[dt.month].append(float(returns_arr[i]))
            if dt.day <= 3 or dt.day >= 28:
                tom_returns.append(float(returns_arr[i]))
            else:
                other_returns.append(float(returns_arr[i]))

        dow_fx: Dict[int, float] = {}
        for d, vals in dow_returns.items():
            dow_fx[d] = float(np.mean(vals)) if vals else 0.0

        month_fx: Dict[int, float] = {}
        for m, vals in month_returns.items():
            month_fx[m] = float(np.mean(vals)) if vals else 0.0

        tom_effect = float(np.mean(tom_returns)) - float(np.mean(other_returns)) if tom_returns and other_returns else 0.0

        pattern = SeasonalPattern(
            name=name,
            day_of_week_effects=dow_fx,
            month_effects=month_fx,
            turn_of_month=tom_effect,
        )
        self.patterns[name] = pattern
        return pattern

    def predict(self, dt: datetime, name: str = "default") -> float:
        p = self.patterns.get(name)
        if not p:
            return 0.0
        effect = 0.0
        effect += p.day_of_week_effects.get(dt.weekday(), 0.0)
        effect += p.month_effects.get(dt.month, 0.0)
        if dt.day <= 3 or dt.day >= 28:
            effect += p.turn_of_month
        return effect

    def best_day(self, name: str = "default") -> Optional[int]:
        p = self.patterns.get(name)
        if not p or not p.day_of_week_effects:
            return None
        return max(p.day_of_week_effects, key=p.day_of_week_effects.get)  # type: ignore

    def worst_day(self, name: str = "default") -> Optional[int]:
        p = self.patterns.get(name)
        if not p or not p.day_of_week_effects:
            return None
        return min(p.day_of_week_effects, key=p.day_of_week_effects.get)  # type: ignore


# ---------------------------------------------------------------------------
# FOMC drift model
# ---------------------------------------------------------------------------

class FOMCDriftModel:
    """Pre-FOMC announcement drift anomaly."""

    def __init__(self, drift_window_days: int = 3, historical_drift_bps: float = 30.0):
        self.drift_window_days = drift_window_days
        self.historical_drift_bps = historical_drift_bps

    def is_in_drift_window(self, current_dt: datetime, fomc_dt: datetime) -> bool:
        delta = (fomc_dt - current_dt).total_seconds() / 86400
        return 0 < delta <= self.drift_window_days

    def expected_drift(self, days_to_fomc: float) -> float:
        """Expected drift in bps, linearly interpolated."""
        if days_to_fomc <= 0 or days_to_fomc > self.drift_window_days:
            return 0.0
        fraction = 1.0 - days_to_fomc / self.drift_window_days
        return self.historical_drift_bps * fraction / 10_000.0

    def calibrate(
        self,
        fomc_dates: List[datetime],
        daily_dates: List[datetime],
        daily_returns: np.ndarray,
    ) -> float:
        """Calibrate historical drift from data."""
        pre_returns: List[float] = []
        date_set = {d.date(): i for i, d in enumerate(daily_dates)}
        for fdt in fomc_dates:
            for offset in range(1, self.drift_window_days + 1):
                check = (fdt - timedelta(days=offset)).date()
                if check in date_set:
                    pre_returns.append(float(daily_returns[date_set[check]]))
        if pre_returns:
            self.historical_drift_bps = float(np.mean(pre_returns)) * 10_000
        return self.historical_drift_bps

    def signal(self, current_dt: datetime, upcoming_fomc: List[datetime]) -> float:
        """Return signal strength (0-1) based on proximity to next FOMC."""
        for fdt in sorted(upcoming_fomc):
            if self.is_in_drift_window(current_dt, fdt):
                days = (fdt - current_dt).total_seconds() / 86400
                return max(0.0, 1.0 - days / self.drift_window_days)
        return 0.0


# ---------------------------------------------------------------------------
# Earnings straddle tracker
# ---------------------------------------------------------------------------

class EarningsStraddleTracker:
    """Track implied move vs realized move around earnings."""

    def __init__(self) -> None:
        self._records: List[Dict[str, Any]] = []

    def record(
        self,
        symbol: str,
        earnings_date: datetime,
        implied_move_pct: float,
        realized_move_pct: float,
    ) -> None:
        self._records.append({
            "symbol": symbol,
            "date": earnings_date,
            "implied": implied_move_pct,
            "realized": realized_move_pct,
            "ratio": realized_move_pct / implied_move_pct if abs(implied_move_pct) > 1e-10 else 0,
        })

    def avg_ratio(self, symbol: Optional[str] = None) -> float:
        subset = self._records
        if symbol:
            subset = [r for r in subset if r["symbol"] == symbol]
        if not subset:
            return 1.0
        return float(np.mean([r["ratio"] for r in subset]))

    def straddle_edge(self, symbol: str) -> float:
        """If avg ratio < 1, selling straddles has edge; > 1 buying has edge."""
        return self.avg_ratio(symbol) - 1.0

    def by_symbol(self) -> Dict[str, float]:
        symbols = set(r["symbol"] for r in self._records)
        return {s: self.avg_ratio(s) for s in symbols}

    def implied_overstatement(self) -> float:
        """Fraction of times implied > realized."""
        if not self._records:
            return 0.0
        over = sum(1 for r in self._records if r["implied"] > abs(r["realized"]))
        return over / len(self._records)

    def recent_records(self, n: int = 20) -> List[Dict[str, Any]]:
        return self._records[-n:]


# ---------------------------------------------------------------------------
# Event-conditional signal blending
# ---------------------------------------------------------------------------

class EventConditionalBlender:
    """Adjust signal weights around events."""

    def __init__(
        self,
        base_weights: Optional[Dict[str, float]] = None,
        event_dampening: float = 0.5,
        event_window_hours: int = 4,
    ):
        self.base_weights = base_weights or {}
        self.event_dampening = event_dampening
        self.event_window_hours = event_window_hours

    def set_base_weights(self, weights: Dict[str, float]) -> None:
        self.base_weights = dict(weights)

    def blend(
        self,
        signals: Dict[str, float],
        current_time: datetime,
        upcoming_events: List[MarketEvent],
    ) -> Dict[str, float]:
        """Produce blended signal values, dampened if near an event."""
        near_event = False
        max_sensitivity = 0.0
        for evt in upcoming_events:
            hours_until = (evt.event_datetime - current_time).total_seconds() / 3600
            if 0 < hours_until <= self.event_window_hours:
                near_event = True
                sens = DEFAULT_SENSITIVITIES.get(evt.event_type, 1.0)
                max_sensitivity = max(max_sensitivity, sens)

        result: Dict[str, float] = {}
        for sig_name, sig_val in signals.items():
            w = self.base_weights.get(sig_name, 1.0)
            if near_event:
                dampening = self.event_dampening * (max_sensitivity / 3.0)
                w *= max(0.1, 1.0 - dampening)
            result[sig_name] = sig_val * w
        return result

    def composite(
        self,
        signals: Dict[str, float],
        current_time: datetime,
        upcoming_events: List[MarketEvent],
    ) -> float:
        blended = self.blend(signals, current_time, upcoming_events)
        return sum(blended.values())


# ---------------------------------------------------------------------------
# Calendar construction
# ---------------------------------------------------------------------------

class EventCalendar:
    """Build and query a forward-looking event calendar."""

    def __init__(self) -> None:
        self.events: List[MarketEvent] = []
        self._by_type: Dict[EventType, List[MarketEvent]] = defaultdict(list)
        self._by_symbol: Dict[str, List[MarketEvent]] = defaultdict(list)

    def add_event(self, event: MarketEvent) -> None:
        self.events.append(event)
        self._by_type[event.event_type].append(event)
        if event.symbol:
            self._by_symbol[event.symbol].append(event)

    def add_recurring(
        self,
        event_type: EventType,
        start: datetime,
        interval_days: int,
        count: int,
        description: str = "",
    ) -> List[MarketEvent]:
        created: List[MarketEvent] = []
        for i in range(count):
            dt = start + timedelta(days=interval_days * i)
            evt = MarketEvent(event_type=event_type, event_datetime=dt, description=description)
            self.add_event(evt)
            created.append(evt)
        return created

    def upcoming(self, hours_ahead: float = 168.0) -> List[MarketEvent]:
        now = datetime.utcnow()
        cutoff = now + timedelta(hours=hours_ahead)
        return sorted(
            [e for e in self.events if now <= e.event_datetime <= cutoff],
            key=lambda e: e.event_datetime,
        )

    def by_type(self, event_type: EventType) -> List[MarketEvent]:
        return sorted(self._by_type.get(event_type, []), key=lambda e: e.event_datetime)

    def by_symbol(self, symbol: str) -> List[MarketEvent]:
        return sorted(self._by_symbol.get(symbol, []), key=lambda e: e.event_datetime)

    def next_event(self, event_type: Optional[EventType] = None) -> Optional[MarketEvent]:
        now = datetime.utcnow()
        candidates = self.events if event_type is None else self._by_type.get(event_type, [])
        future = [e for e in candidates if e.event_datetime > now]
        return min(future, key=lambda e: e.event_datetime) if future else None

    def events_on_date(self, dt: datetime) -> List[MarketEvent]:
        return [e for e in self.events if e.event_datetime.date() == dt.date()]

    def density(self, dt: datetime, window_hours: int = 24) -> int:
        start = dt - timedelta(hours=window_hours / 2)
        end = dt + timedelta(hours=window_hours / 2)
        return sum(1 for e in self.events if start <= e.event_datetime <= end)

    def busy_days(self, threshold: int = 3) -> List[datetime]:
        from collections import Counter
        day_counts: Counter[str] = Counter()
        for e in self.events:
            day_counts[e.event_datetime.strftime("%Y-%m-%d")] += 1
        return [
            datetime.strptime(d, "%Y-%m-%d")
            for d, c in day_counts.items()
            if c >= threshold
        ]


# ---------------------------------------------------------------------------
# Integrated event impact engine
# ---------------------------------------------------------------------------

class EventImpactEngine:
    """Orchestrate all event-related models."""

    def __init__(self) -> None:
        self.calendar = EventCalendar()
        self.pre_event = PreEventModel()
        self.post_event = PostEventImpactModel()
        self.cluster_analyzer = EventClusterAnalyzer()
        self.seasonal = SeasonalPatternModel()
        self.fomc_drift = FOMCDriftModel()
        self.straddle_tracker = EarningsStraddleTracker()
        self.blender = EventConditionalBlender()

    def assess_current(
        self,
        current_time: datetime,
        signals: Dict[str, float],
        hours_ahead: float = 48.0,
    ) -> Dict[str, Any]:
        upcoming = self.calendar.upcoming(hours_ahead)
        clusters = self.cluster_analyzer.find_clusters(upcoming)
        fomc_dates = [e.event_datetime for e in self.calendar.by_type(EventType.FOMC)]
        fomc_signal = self.fomc_drift.signal(current_time, fomc_dates)
        blended = self.blender.blend(signals, current_time, upcoming)
        composite = sum(blended.values())
        # Position adjustment
        pos_adj = 1.0
        for evt in upcoming:
            hours_until = (evt.event_datetime - current_time).total_seconds() / 3600
            if 0 < hours_until <= 4:
                pos_adj = min(pos_adj, self.pre_event.position_adjustment(evt.event_type))
        cluster_vol = 1.0
        for cl in clusters:
            cluster_vol = max(cluster_vol, self.cluster_analyzer.cluster_vol_multiplier(cl))
        seasonal_adj = self.seasonal.predict(current_time)
        return {
            "n_upcoming": len(upcoming),
            "n_clusters": len(clusters),
            "fomc_drift_signal": fomc_signal,
            "composite_signal": composite,
            "position_adjustment": pos_adj,
            "cluster_vol_multiplier": cluster_vol,
            "seasonal_adjustment": seasonal_adj,
            "blended_signals": blended,
        }

    def process_event_outcome(self, event: MarketEvent, realized_return: float) -> EventImpactResult:
        predicted = self.post_event.compute_impact(event)
        return EventImpactResult(
            event=event,
            post_event_return=realized_return,
            surprise_impact=event.surprise if event.surprise else 0.0,
            model_predicted_impact=predicted,
            signal_adjustment=0.0,
        )
