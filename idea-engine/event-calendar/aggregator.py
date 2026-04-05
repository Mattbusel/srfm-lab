"""
EventAggregator: combines all event sources into a unified calendar.

Responsibilities:
  - Pull from CryptoCalendar, UnlockTracker, EconomicCalendar, ExchangeListings
  - Deduplicate same event appearing in multiple sources
  - Compute a composite EventRisk score per event
  - Expose query methods: get_events_next_7_days, get_events_today,
    is_high_risk_window
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Union

from .sources.cryptocalendar_scraper import CryptoCalendarScraper, CalendarEvent
from .sources.unlock_tracker import UnlockTracker, UnlockEvent
from .sources.economic_calendar import EconomicCalendar, MacroEvent
from .sources.exchange_listings import ExchangeListingMonitor, ListingEvent

logger = logging.getLogger(__name__)

AnyEvent = Union[CalendarEvent, UnlockEvent, MacroEvent, ListingEvent]

RISK_WEIGHTS: Dict[str, float] = {
    "HIGH": 1.0,
    "MEDIUM": 0.5,
    "LOW": 0.2,
}


@dataclass
class AggregatedEvent:
    """Normalised event record from any source with composite risk score."""

    event_id: str
    event_name: str
    symbol: str               # "MACRO" for macro events not tied to a coin
    event_date: datetime
    category: str
    impact: str               # HIGH / MEDIUM / LOW
    event_risk_score: float   # 0.0 – 1.0
    sources: List[str] = field(default_factory=list)
    description: str = ""
    raw_events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["event_date"] = self.event_date.isoformat()
        return d


class EventCalendar:
    """Sorted list of AggregatedEvents with query helpers."""

    def __init__(self, events: List[AggregatedEvent]) -> None:
        self._events = sorted(events, key=lambda e: e.event_date)

    def __len__(self) -> int:
        return len(self._events)

    def all(self) -> List[AggregatedEvent]:
        return list(self._events)

    def for_symbol(self, symbol: str) -> List[AggregatedEvent]:
        sym = symbol.upper()
        return [e for e in self._events
                if e.symbol.upper() == sym or e.symbol == "MACRO"]

    def in_window(self, start: datetime, end: datetime) -> List[AggregatedEvent]:
        return [e for e in self._events if start <= e.event_date <= end]

    def high_risk(self) -> List[AggregatedEvent]:
        return [e for e in self._events if e.impact == "HIGH"]


class EventAggregator:
    """
    Aggregates events from all sources.

    Usage::

        agg  = EventAggregator()
        cal  = agg.build_calendar()
        evts = agg.get_events_next_7_days("SOL")
        risk = agg.is_high_risk_window("BTC", datetime.now(timezone.utc))
    """

    def __init__(
        self,
        coinmarketcal_api_key: str = "",
        fred_api_key: str = "",
    ) -> None:
        self._crypto_scraper = CryptoCalendarScraper(
            coinmarketcal_api_key=coinmarketcal_api_key
        )
        self._unlock_tracker = UnlockTracker()
        self._econ_calendar = EconomicCalendar(fred_api_key=fred_api_key)
        self._listing_monitor = ExchangeListingMonitor()

    # ── public API ──────────────────────────────────────────────────────────────

    def build_calendar(self, days_ahead: int = 30) -> EventCalendar:
        """Fetch all sources and return a unified EventCalendar."""
        raw: List[AggregatedEvent] = []

        # --- crypto calendar events ---
        try:
            for ev in self._crypto_scraper.events_next_n_days(days_ahead):
                raw.append(self._from_calendar_event(ev))
        except Exception as exc:
            logger.warning("CryptoCalendar scrape error: %s", exc)

        # --- token unlocks ---
        try:
            for ev in self._unlock_tracker.fetch_upcoming(days=days_ahead):
                raw.append(self._from_unlock_event(ev))
        except Exception as exc:
            logger.warning("Unlock tracker error: %s", exc)

        # --- macro events ---
        try:
            for ev in self._econ_calendar.fetch_upcoming(days=days_ahead):
                raw.append(self._from_macro_event(ev))
        except Exception as exc:
            logger.warning("Economic calendar error: %s", exc)

        # --- exchange listings ---
        try:
            for ev in self._listing_monitor.fetch_all():
                raw.append(self._from_listing_event(ev))
        except Exception as exc:
            logger.warning("Listing monitor error: %s", exc)

        deduped = self._deduplicate(raw)
        return EventCalendar(deduped)

    def get_events_next_7_days(self, symbol: str) -> List[AggregatedEvent]:
        cal = self.build_calendar(days_ahead=7)
        return cal.for_symbol(symbol)

    def get_events_today(self) -> List[AggregatedEvent]:
        now = datetime.now(timezone.utc)
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(hours=24)
        return self.build_calendar(days_ahead=1).in_window(start, end)

    def is_high_risk_window(self, symbol: str, ts: datetime) -> bool:
        """
        Return True if *ts* is within 24h of any HIGH-impact event for *symbol*
        or any macro HIGH-impact event.
        """
        window_start = ts - timedelta(hours=2)
        window_end = ts + timedelta(hours=24)
        cal = self.build_calendar(days_ahead=2)
        relevant = cal.for_symbol(symbol)
        return any(
            e.impact == "HIGH" and window_start <= e.event_date <= window_end
            for e in relevant
        )

    def compute_event_risk_score(self, symbol: str, ts: datetime) -> float:
        """
        Composite risk score 0-1 for a given symbol at time *ts*.

        Accumulates contributions from all events within ±3 days, decayed
        by time distance.
        """
        window_start = ts - timedelta(days=3)
        window_end = ts + timedelta(days=3)
        cal = self.build_calendar(days_ahead=6)
        relevant = [
            e for e in cal.for_symbol(symbol)
            if window_start <= e.event_date <= window_end
        ]
        score = 0.0
        for ev in relevant:
            time_dist_days = abs((ev.event_date - ts).total_seconds()) / 86400
            decay = max(0.0, 1.0 - time_dist_days / 3.0)
            score += ev.event_risk_score * decay
        return min(score, 1.0)

    # ── normalisation helpers ────────────────────────────────────────────────────

    @staticmethod
    def _from_calendar_event(ev: CalendarEvent) -> AggregatedEvent:
        score = RISK_WEIGHTS.get(ev.impact_estimate, 0.2) * ev.confidence
        return AggregatedEvent(
            event_id=ev.event_id,
            event_name=ev.event_name,
            symbol=ev.symbol,
            event_date=ev.date,
            category=ev.category,
            impact=ev.impact_estimate,
            event_risk_score=min(score, 1.0),
            sources=[ev.source],
            description=ev.description,
        )

    @staticmethod
    def _from_unlock_event(ev: UnlockEvent) -> AggregatedEvent:
        impact = "HIGH" if ev.is_high_impact else "MEDIUM"
        score = min(abs(ev.estimated_sell_pressure) / 10.0, 1.0)
        return AggregatedEvent(
            event_id=ev.event_id,
            event_name=f"{ev.symbol} Token Unlock ({ev.unlock_type})",
            symbol=ev.symbol,
            event_date=ev.unlock_date,
            category="unlock",
            impact=impact,
            event_risk_score=score,
            sources=[ev.source],
            description=(
                f"{ev.unlock_pct_of_circ:.2f}% of circulating supply; "
                f"est. {ev.estimated_sell_pressure:.1f}% price impact"
            ),
        )

    @staticmethod
    def _from_macro_event(ev: MacroEvent) -> AggregatedEvent:
        score = RISK_WEIGHTS.get(ev.impact, 0.2) * (ev.expected_btc_move_pct / 5.0)
        return AggregatedEvent(
            event_id=ev.event_id,
            event_name=ev.event_name,
            symbol="MACRO",
            event_date=ev.scheduled_datetime,
            category=ev.event_type,
            impact=ev.impact,
            event_risk_score=min(score, 1.0),
            sources=[ev.source],
            description=ev.description,
        )

    @staticmethod
    def _from_listing_event(ev: ListingEvent) -> AggregatedEvent:
        score = min(abs(ev.expected_price_change_pct) / 40.0, 1.0)
        return AggregatedEvent(
            event_id=ev.event_id,
            event_name=ev.title,
            symbol=ev.symbol,
            event_date=ev.detected_at,
            category=f"exchange_{ev.event_subtype}",
            impact=ev.impact,
            event_risk_score=score,
            sources=[ev.exchange],
            description=f"{ev.exchange} {ev.event_subtype}: expected {ev.expected_price_change_pct:+.0f}%",
        )

    @staticmethod
    def _deduplicate(events: List[AggregatedEvent]) -> List[AggregatedEvent]:
        seen: Dict[str, AggregatedEvent] = {}
        for ev in events:
            key = (
                f"{ev.symbol.upper()}_"
                f"{ev.event_date.date()}_"
                f"{ev.event_name[:25].lower().replace(' ', '_')}"
            )
            if key not in seen:
                seen[key] = ev
            else:
                # Merge sources and keep higher risk score
                existing = seen[key]
                existing.sources = list(set(existing.sources + ev.sources))
                existing.event_risk_score = max(existing.event_risk_score, ev.event_risk_score)
                if ev.impact == "HIGH":
                    existing.impact = "HIGH"
        return list(seen.values())
