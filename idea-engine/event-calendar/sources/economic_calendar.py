"""
Macro economic calendar for crypto traders.

Fetches scheduled macro events (FOMC, CPI, NFP, GDP) from FRED or a
ForexFactory-style API.  Provides a buffer mechanism: signals to reduce new
trade entries 2h before and 1h after major macro events.

Historical correlation baseline:
  - FOMC surprise (rate unexpected) -> ±5% BTC within 24h
  - CPI above / below expectation  -> ±3% BTC within 6h
  - NFP surprise                   -> ±2% BTC within 4h
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

FRED_RELEASES_URL = (
    "https://api.stlouisfed.org/fred/releases/dates"
    "?realtime_start={start}&realtime_end={end}&api_key={key}&file_type=json"
)

CACHE_TTL = 4 * 3600


# Impact in expected BTC % move (absolute value) and direction "surprise"
MACRO_EVENTS: Dict[str, Dict[str, Any]] = {
    "FOMC":   {"impact_pct": 5.0,  "buffer_before_h": 2, "buffer_after_h": 1, "impact": "HIGH"},
    "CPI":    {"impact_pct": 3.0,  "buffer_before_h": 2, "buffer_after_h": 1, "impact": "HIGH"},
    "NFP":    {"impact_pct": 2.0,  "buffer_before_h": 1, "buffer_after_h": 1, "impact": "MEDIUM"},
    "GDP":    {"impact_pct": 1.5,  "buffer_before_h": 1, "buffer_after_h": 1, "impact": "MEDIUM"},
    "PPI":    {"impact_pct": 1.0,  "buffer_before_h": 1, "buffer_after_h": 0, "impact": "LOW"},
    "RETAIL": {"impact_pct": 1.0,  "buffer_before_h": 1, "buffer_after_h": 0, "impact": "LOW"},
}


@dataclass
class MacroEvent:
    """A scheduled macro economic announcement."""

    event_id: str
    event_name: str
    event_type: str          # FOMC | CPI | NFP | GDP | ...
    scheduled_datetime: datetime
    impact: str              # HIGH / MEDIUM / LOW
    expected_btc_move_pct: float
    buffer_before: timedelta
    buffer_after: timedelta
    source: str
    description: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def buffer_start(self) -> datetime:
        return self.scheduled_datetime - self.buffer_before

    @property
    def buffer_end(self) -> datetime:
        return self.scheduled_datetime + self.buffer_after

    def is_in_buffer(self, ts: datetime) -> bool:
        return self.buffer_start <= ts <= self.buffer_end

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["scheduled_datetime"] = self.scheduled_datetime.isoformat()
        d["buffer_before"] = self.buffer_before.total_seconds()
        d["buffer_after"] = self.buffer_after.total_seconds()
        return d


class EconomicCalendar:
    """
    Provides upcoming macro events and buffer windows.

    Usage::

        cal   = EconomicCalendar(fred_api_key="YOUR_KEY")
        events = cal.fetch_upcoming(days=30)
        if cal.is_macro_buffer_active():
            reduce_entries()
    """

    def __init__(
        self,
        fred_api_key: str = "",
        cache_ttl: int = CACHE_TTL,
    ) -> None:
        self._fred_key = fred_api_key
        self._cache_ttl = cache_ttl
        self._cache: Dict[str, Tuple[float, Any]] = {}

    # ── public API ──────────────────────────────────────────────────────────────

    def fetch_upcoming(self, days: int = 30) -> List[MacroEvent]:
        """Return macro events in the next *days* days, sorted by date."""
        key = f"macro_{days}"
        cached = self._get_cache(key)
        if cached is not None:
            return cached

        events = self._fetch_fred(days) if self._fred_key else []
        if not events:
            logger.info("FRED unavailable or no key — using scheduled calendar model")
            events = self._generate_scheduled_events(days)

        events.sort(key=lambda e: e.scheduled_datetime)
        self._set_cache(key, events)
        return events

    def get_events_today(self) -> List[MacroEvent]:
        now = datetime.now(timezone.utc)
        return [e for e in self.fetch_upcoming(days=1)
                if e.scheduled_datetime.date() == now.date()]

    def is_macro_buffer_active(self, ts: Optional[datetime] = None) -> bool:
        """Return True if *ts* (default: now) falls inside any macro buffer window."""
        ts = ts or datetime.now(timezone.utc)
        return any(e.is_in_buffer(ts) for e in self.fetch_upcoming(days=3))

    def active_buffer_events(self, ts: Optional[datetime] = None) -> List[MacroEvent]:
        """Return list of events currently in their buffer window."""
        ts = ts or datetime.now(timezone.utc)
        return [e for e in self.fetch_upcoming(days=3) if e.is_in_buffer(ts)]

    def entry_size_multiplier(self, ts: Optional[datetime] = None) -> float:
        """
        Return position-size multiplier.

        Returns 0.5 if a HIGH-impact buffer is active, 0.75 for MEDIUM,
        otherwise 1.0.
        """
        active = self.active_buffer_events(ts)
        if not active:
            return 1.0
        impacts = {e.impact for e in active}
        if "HIGH" in impacts:
            return 0.5
        if "MEDIUM" in impacts:
            return 0.75
        return 0.9

    # ── FRED fetch ───────────────────────────────────────────────────────────────

    def _fetch_fred(self, days: int) -> List[MacroEvent]:
        events: List[MacroEvent] = []
        today = datetime.now(timezone.utc)
        end = today + timedelta(days=days)
        url = FRED_RELEASES_URL.format(
            start=today.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            key=self._fred_key,
        )
        try:
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            for item in data.get("release_dates", []):
                ev = self._parse_fred_item(item)
                if ev:
                    events.append(ev)
        except (urllib.error.URLError, json.JSONDecodeError) as exc:
            logger.warning("FRED fetch failed: %s", exc)
        return events

    def _parse_fred_item(self, item: Dict[str, Any]) -> Optional[MacroEvent]:
        name: str = item.get("release_name", "")
        date_str: str = item.get("date", "")
        event_type = self._match_event_type(name)
        if not event_type:
            return None
        meta = MACRO_EVENTS[event_type]
        try:
            dt = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
        except ValueError:
            return None
        return self._make_event(event_type, name, dt, meta, source="fred", raw=item)

    # ── scheduled calendar model ─────────────────────────────────────────────────

    def _generate_scheduled_events(self, days: int) -> List[MacroEvent]:
        """
        Return a plausible forward calendar based on typical US release cadence.

        FOMC: 8x/year (roughly every 6 weeks)
        CPI:  monthly on ~13th
        NFP:  first Friday of month
        GDP:  quarterly (Jan/Apr/Jul/Oct ~30th)
        """
        now = datetime.now(timezone.utc)
        events: List[MacroEvent] = []

        for offset_days in range(0, days + 1):
            d = now + timedelta(days=offset_days)
            events.extend(self._check_day(d))

        return events

    def _check_day(self, d: datetime) -> List[MacroEvent]:
        events: List[MacroEvent] = []
        dom = d.day
        dow = d.weekday()  # 0=Mon, 4=Fri
        month = d.month

        # CPI: ~13th of every month at 08:30 ET (13:30 UTC)
        if dom == 13:
            events.append(self._make_event(
                "CPI", f"CPI Release {d.strftime('%b %Y')}",
                d.replace(hour=13, minute=30, second=0, microsecond=0, tzinfo=timezone.utc),
                MACRO_EVENTS["CPI"], source="scheduled",
            ))

        # NFP: first Friday of month at 08:30 ET
        if dow == 4 and dom <= 7:
            events.append(self._make_event(
                "NFP", f"Non-Farm Payrolls {d.strftime('%b %Y')}",
                d.replace(hour=13, minute=30, second=0, microsecond=0, tzinfo=timezone.utc),
                MACRO_EVENTS["NFP"], source="scheduled",
            ))

        # GDP: last week of Jan, Apr, Jul, Oct (dom >= 25)
        if month in (1, 4, 7, 10) and 25 <= dom <= 31 and dow == 2:  # Wednesday
            events.append(self._make_event(
                "GDP", f"GDP Advance {d.strftime('%b %Y')}",
                d.replace(hour=13, minute=30, second=0, microsecond=0, tzinfo=timezone.utc),
                MACRO_EVENTS["GDP"], source="scheduled",
            ))

        # FOMC: roughly every 6 weeks; approximate with specific months
        # Jan/Mar/May/Jun/Jul/Sep/Nov/Dec — meeting ends Wednesday
        if month in (1, 3, 5, 6, 7, 9, 11, 12) and dom in range(25, 32) and dow == 2:
            events.append(self._make_event(
                "FOMC", f"FOMC Decision {d.strftime('%b %Y')}",
                d.replace(hour=18, minute=0, second=0, microsecond=0, tzinfo=timezone.utc),
                MACRO_EVENTS["FOMC"], source="scheduled",
            ))

        return events

    # ── helpers ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _match_event_type(name: str) -> Optional[str]:
        name_upper = name.upper()
        for key in MACRO_EVENTS:
            if key in name_upper:
                return key
        return None

    @staticmethod
    def _make_event(
        event_type: str,
        name: str,
        dt: datetime,
        meta: Dict[str, Any],
        source: str,
        raw: Optional[Dict[str, Any]] = None,
    ) -> MacroEvent:
        return MacroEvent(
            event_id=f"macro_{event_type}_{dt.strftime('%Y%m%d%H%M')}",
            event_name=name,
            event_type=event_type,
            scheduled_datetime=dt,
            impact=meta["impact"],
            expected_btc_move_pct=meta["impact_pct"],
            buffer_before=timedelta(hours=meta["buffer_before_h"]),
            buffer_after=timedelta(hours=meta["buffer_after_h"]),
            source=source,
            raw=raw or {},
        )

    def _get_cache(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry and (time.time() - entry[0]) < self._cache_ttl:
            return entry[1]
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        self._cache[key] = (time.time(), value)
