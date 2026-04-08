"""
Real-Time Event Calendar Feed (T1-3)
Replaces synthetic event generation with real data from free APIs.

Sources:
  - CME FedWatch (FOMC dates via scraping or free API)
  - CoinGecko token unlock schedule
  - General economic calendar

Falls back to hardcoded calendar if network is unavailable.
Persists to config/event_calendar.json and refreshes daily.
"""
import json
import logging
import time
import urllib.request
import urllib.parse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger(__name__)

@dataclass
class CalendarEvent:
    timestamp: datetime
    event_type: str         # "fomc", "token_unlock", "earnings", "economic"
    instrument: str         # symbol or "ALL"
    description: str
    sizing_multiplier: float = 0.5  # position size during event window
    window_hours_before: int = 2
    window_hours_after: int = 1

class EventCalendarFeed:
    """
    Fetches and caches real economic events. Falls back to static calendar.

    Usage:
        feed = EventCalendarFeed()
        feed.refresh()  # call once at startup and daily
        multiplier = feed.get_sizing_multiplier("BTC", datetime.now(UTC))
    """

    # Hardcoded 2025-2026 FOMC dates as fallback
    FOMC_FALLBACK_DATES = [
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
    ]

    def __init__(self, cache_path: str = None):
        self._cache_path = Path(cache_path or "config/event_calendar_live.json")
        self._events: list[CalendarEvent] = []
        self._last_refresh: float = 0.0
        self._load_fallback()

    def refresh(self, force: bool = False) -> None:
        """Refresh calendar from APIs. Call at startup and daily."""
        now = time.time()
        if not force and now - self._last_refresh < 86400:  # 24 hours
            return
        self._last_refresh = now

        events = list(self._load_fallback_events())  # always include hardcoded

        # Try fetching real FOMC dates
        try:
            fomc = self._fetch_fomc_dates()
            if fomc:
                events = [e for e in events if e.event_type != "fomc"] + fomc
                log.info("EventCalendar: fetched %d FOMC dates", len(fomc))
        except Exception as e:
            log.debug("EventCalendar: FOMC fetch failed: %s", e)

        # Try CoinGecko token unlocks
        try:
            unlocks = self._fetch_coingecko_events()
            events += unlocks
            log.info("EventCalendar: fetched %d CoinGecko events", len(unlocks))
        except Exception as e:
            log.debug("EventCalendar: CoinGecko fetch failed: %s", e)

        self._events = sorted(events, key=lambda e: e.timestamp)
        self._persist()
        log.info("EventCalendar: %d events loaded", len(self._events))

    def get_sizing_multiplier(self, instrument: str, ts: datetime) -> float:
        """
        Returns position sizing multiplier for instrument at timestamp.
        1.0 = normal, 0.5 = reduce (event window), 0.0 = skip entirely.
        """
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)

        min_mult = 1.0
        for event in self._events:
            etv = event.timestamp
            if etv.tzinfo is None:
                etv = etv.replace(tzinfo=timezone.utc)
            window_start = etv - timedelta(hours=event.window_hours_before)
            window_end   = etv + timedelta(hours=event.window_hours_after)

            if not (window_start <= ts <= window_end):
                continue

            # Check if event applies to this instrument
            if event.instrument not in ("ALL", instrument, "CRYPTO", "EQUITY"):
                continue
            if event.instrument == "CRYPTO" and instrument not in (
                "BTC","ETH","XRP","AVAX","LINK","AAVE","LTC","BCH","MKR","YFI"
            ):
                continue
            if event.instrument == "EQUITY" and instrument not in (
                "SPY","QQQ","IWM","GLD","TLT","SLV","USO","NVDA","AAPL","TSLA","MSFT"
            ):
                continue

            min_mult = min(min_mult, event.sizing_multiplier)

        return min_mult

    def get_upcoming_events(self, hours: int = 48) -> list[CalendarEvent]:
        """Return events within the next N hours."""
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours)
        return [e for e in self._events if now <= e.timestamp <= cutoff]

    def _load_fallback_events(self) -> list[CalendarEvent]:
        events = []
        for date_str in self.FOMC_FALLBACK_DATES:
            try:
                ts = datetime.fromisoformat(date_str + "T14:00:00+00:00")
                events.append(CalendarEvent(
                    timestamp=ts,
                    event_type="fomc",
                    instrument="ALL",
                    description="FOMC Meeting (fallback)",
                    sizing_multiplier=0.5,
                    window_hours_before=2,
                    window_hours_after=1,
                ))
            except Exception:
                pass
        return events

    def _load_fallback(self):
        """Load from local cache or hardcoded fallback."""
        if self._cache_path.exists():
            try:
                with open(self._cache_path) as f:
                    data = json.load(f)
                self._events = []
                for item in data.get("events", []):
                    try:
                        ts = datetime.fromisoformat(item["timestamp"])
                        self._events.append(CalendarEvent(
                            timestamp=ts,
                            event_type=item.get("event_type", "unknown"),
                            instrument=item.get("instrument", "ALL"),
                            description=item.get("description", ""),
                            sizing_multiplier=item.get("sizing_multiplier", 0.5),
                            window_hours_before=item.get("window_hours_before", 2),
                            window_hours_after=item.get("window_hours_after", 1),
                        ))
                    except Exception:
                        pass
                log.info("EventCalendar: loaded %d events from cache", len(self._events))
                return
            except Exception as e:
                log.warning("EventCalendar: cache load failed: %s", e)

        self._events = list(self._load_fallback_events())

    def _fetch_fomc_dates(self) -> list[CalendarEvent]:
        """Fetch FOMC dates from Federal Reserve website."""
        # FR publishes meeting dates in plain HTML — parse year pages
        events = []
        for year in [datetime.now().year, datetime.now().year + 1]:
            url = f"https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            try:
                with urllib.request.urlopen(req, timeout=8) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
                # Look for date patterns like "January 28-29" in Fed HTML
                import re
                months = {
                    "January":1,"February":2,"March":3,"April":4,"May":5,"June":6,
                    "July":7,"August":8,"September":9,"October":10,"November":11,"December":12
                }
                pattern = r'(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:-\d{1,2})?'
                for m in re.finditer(pattern, html):
                    month_name, day = m.group(1), int(m.group(2))
                    month_num = months[month_name]
                    try:
                        ts = datetime(year, month_num, day, 14, 0, 0, tzinfo=timezone.utc)
                        if ts > datetime.now(timezone.utc):
                            events.append(CalendarEvent(
                                timestamp=ts,
                                event_type="fomc",
                                instrument="ALL",
                                description=f"FOMC Meeting {month_name} {year}",
                                sizing_multiplier=0.5,
                            ))
                    except ValueError:
                        pass
                break  # one page is enough
            except Exception:
                pass

        # Deduplicate
        seen = set()
        unique = []
        for e in events:
            key = e.timestamp.date()
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    def _fetch_coingecko_events(self) -> list[CalendarEvent]:
        """Fetch upcoming token unlock events from CoinGecko (no API key required)."""
        events = []
        # CoinGecko events API (free tier)
        url = "https://api.coingecko.com/api/v3/events?upcoming_events_only=true&page=1&per_page=50"
        req = urllib.request.Request(url, headers={"User-Agent": "srfm-lab/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read())

            sym_map = {
                "bitcoin": "BTC", "ethereum": "ETH", "ripple": "XRP",
                "avalanche-2": "AVAX", "chainlink": "LINK", "aave": "AAVE",
                "litecoin": "LTC", "bitcoin-cash": "BCH", "maker": "MKR",
            }

            for item in data.get("data", []):
                coin_id = item.get("coin_id", "")
                sym = sym_map.get(coin_id, "")
                if not sym:
                    continue
                event_type = item.get("type", "").lower()
                if "unlock" not in event_type and "release" not in event_type:
                    continue
                ts_str = item.get("start_date", "")
                if not ts_str:
                    continue
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    events.append(CalendarEvent(
                        timestamp=ts,
                        event_type="token_unlock",
                        instrument=sym,
                        description=item.get("description", f"{sym} token unlock"),
                        sizing_multiplier=0.6,
                        window_hours_before=4,
                        window_hours_after=2,
                    ))
                except Exception:
                    pass
        except Exception as e:
            log.debug("CoinGecko events fetch failed: %s", e)

        return events

    def _persist(self):
        """Save events to cache file."""
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = {"events": [
                {
                    "timestamp": e.timestamp.isoformat(),
                    "event_type": e.event_type,
                    "instrument": e.instrument,
                    "description": e.description,
                    "sizing_multiplier": e.sizing_multiplier,
                    "window_hours_before": e.window_hours_before,
                    "window_hours_after": e.window_hours_after,
                }
                for e in self._events
            ], "refreshed_at": datetime.now(timezone.utc).isoformat()}
            with open(self._cache_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log.warning("EventCalendar: persist failed: %s", e)
