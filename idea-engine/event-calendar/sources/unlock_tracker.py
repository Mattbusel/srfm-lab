"""
Token unlock schedule tracker.

Fetches or simulates upcoming token unlock events for a watchlist.
Classifies cliff vs linear unlocks, estimates sell pressure, and flags
events that could create near-term bearish price action.

Historical baseline: each 1% of circulating supply unlocked correlates
with approximately -2% price impact within 7 days for liquid tokens.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

UNLOCKS_API = "https://token.unlocks.app/api/upcoming"
CACHE_TTL = 4 * 3600

# Watchlist with approximate circulating supply (millions of tokens) for
# sell-pressure modelling.  These figures are periodically stale and are used
# only for simulation when the API is unavailable.
WATCHLIST: Dict[str, Dict[str, Any]] = {
    "SOL":  {"circ_supply_M": 430,  "total_supply_M": 588},
    "AVAX": {"circ_supply_M": 395,  "total_supply_M": 720},
    "ARB":  {"circ_supply_M": 1_270, "total_supply_M": 10_000},
    "OP":   {"circ_supply_M": 1_050, "total_supply_M": 4_294},
    "APT":  {"circ_supply_M": 360,  "total_supply_M": 1_100},
}

PRICE_IMPACT_PER_PCT = -0.02  # -2% price per 1% supply unlock


@dataclass
class UnlockEvent:
    """A single token unlock event."""

    event_id: str
    symbol: str
    unlock_date: datetime
    unlock_type: str          # cliff_unlock | linear_unlock
    unlock_tokens: float      # absolute number of tokens
    circ_supply: float        # circulating supply at unlock time
    unlock_pct_of_circ: float
    estimated_sell_pressure: float   # expected % price change (negative = bearish)
    vesting_cliff: bool
    source: str
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_high_impact(self) -> bool:
        return self.unlock_pct_of_circ >= 2.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["unlock_date"] = self.unlock_date.isoformat()
        return d


class UnlockTracker:
    """
    Tracks token unlock schedules for the configured watchlist.

    Attempts live data from token.unlocks.app; falls back to a deterministic
    simulation model when the API is unreachable.

    Usage::

        tracker = UnlockTracker()
        events  = tracker.fetch_upcoming(days=30)
        big     = tracker.get_high_impact_events(days=7)
    """

    def __init__(
        self,
        watchlist: Optional[Dict[str, Dict[str, Any]]] = None,
        cache_ttl: int = CACHE_TTL,
    ) -> None:
        self._watchlist = watchlist or WATCHLIST
        self._cache_ttl = cache_ttl
        self._cache: Dict[str, tuple[float, Any]] = {}
        self._timeout = 10

    # ── public API ──────────────────────────────────────────────────────────────

    def fetch_upcoming(self, days: int = 30) -> List[UnlockEvent]:
        """Return all unlock events within the next *days* days."""
        cache_key = f"upcoming_{days}"
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        events = self._fetch_live(days)
        if not events:
            logger.info("Live unlock API unavailable — using simulation model")
            events = self._simulate_unlocks(days)

        self._set_cache(cache_key, events)
        return events

    def get_high_impact_events(self, days: int = 7) -> List[UnlockEvent]:
        """Return events with unlock_pct_of_circ >= 2% within *days* days."""
        return [e for e in self.fetch_upcoming(days) if e.is_high_impact]

    def sell_pressure_estimate(self, symbol: str, days: int = 7) -> float:
        """
        Return aggregate expected price impact (%) from unlocks in next *days*.
        """
        events = [
            e for e in self.fetch_upcoming(days)
            if e.symbol.upper() == symbol.upper()
        ]
        return sum(e.estimated_sell_pressure for e in events)

    # ── live fetch ──────────────────────────────────────────────────────────────

    def _fetch_live(self, days: int) -> List[UnlockEvent]:
        events: List[UnlockEvent] = []
        try:
            url = f"{UNLOCKS_API}?days={days}"
            req = urllib.request.Request(url)
            req.add_header("Accept", "application/json")
            req.add_header("User-Agent", "SRFM-IdeaEngine/1.0")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode())
            for item in data.get("data", []):
                ev = self._parse_live_item(item)
                if ev and ev.symbol.upper() in self._watchlist:
                    events.append(ev)
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
            logger.warning("Unlock API fetch failed: %s", exc)
        return events

    def _parse_live_item(self, item: Dict[str, Any]) -> Optional[UnlockEvent]:
        try:
            symbol = item["symbol"].upper()
            ts = item.get("date", item.get("timestamp", 0))
            date = datetime.fromtimestamp(float(ts), tz=timezone.utc)
            unlock_tokens = float(item.get("amount", 0))
            circ = float(item.get("circulating_supply", 1))
            pct = (unlock_tokens / circ * 100) if circ else 0.0
            return UnlockEvent(
                event_id=f"live_{symbol}_{int(ts)}",
                symbol=symbol,
                unlock_date=date,
                unlock_type="cliff_unlock" if item.get("is_cliff", False) else "linear_unlock",
                unlock_tokens=unlock_tokens,
                circ_supply=circ,
                unlock_pct_of_circ=pct,
                estimated_sell_pressure=pct * PRICE_IMPACT_PER_PCT,
                vesting_cliff=bool(item.get("is_cliff", False)),
                source="token_unlocks_app",
                raw=item,
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("Could not parse unlock item: %s", exc)
            return None

    # ── simulation model ────────────────────────────────────────────────────────

    def _simulate_unlocks(self, days: int) -> List[UnlockEvent]:
        """
        Deterministic simulation of monthly unlock schedules.

        Each token in the watchlist has a fixed monthly unlock percentage
        derived from its vesting schedule.  Events are placed at the start
        of each calendar month within the window.
        """
        import math
        now = datetime.now(timezone.utc)
        events: List[UnlockEvent] = []

        # Monthly unlock % assumptions (cliff + linear blend)
        monthly_pcts: Dict[str, tuple[float, str]] = {
            "SOL":  (0.5,  "linear_unlock"),
            "AVAX": (0.3,  "linear_unlock"),
            "ARB":  (3.0,  "cliff_unlock"),   # large cliff schedule
            "OP":   (2.5,  "cliff_unlock"),
            "APT":  (1.8,  "linear_unlock"),
        }

        for symbol, meta in self._watchlist.items():
            pct, unlock_type = monthly_pcts.get(symbol, (0.5, "linear_unlock"))
            circ = meta["circ_supply_M"] * 1_000_000

            # Generate one unlock event per month within window
            months_ahead = math.ceil(days / 30)
            for m in range(1, months_ahead + 1):
                # First day of the month, m months from now
                future_month = (now.month + m - 1) % 12 + 1
                future_year = now.year + (now.month + m - 1) // 12
                unlock_date = now.replace(
                    year=future_year, month=future_month, day=1,
                    hour=0, minute=0, second=0, microsecond=0,
                )
                if (unlock_date - now).total_seconds() > days * 86400:
                    continue
                tokens = circ * pct / 100
                events.append(UnlockEvent(
                    event_id=f"sim_{symbol}_{future_year}{future_month:02d}",
                    symbol=symbol,
                    unlock_date=unlock_date,
                    unlock_type=unlock_type,
                    unlock_tokens=tokens,
                    circ_supply=circ,
                    unlock_pct_of_circ=pct,
                    estimated_sell_pressure=pct * PRICE_IMPACT_PER_PCT,
                    vesting_cliff=(unlock_type == "cliff_unlock"),
                    source="simulation",
                ))

        events.sort(key=lambda e: e.unlock_date)
        return events

    # ── cache helpers ────────────────────────────────────────────────────────────

    def _get_cache(self, key: str) -> Optional[Any]:
        entry = self._cache.get(key)
        if entry and (time.time() - entry[0]) < self._cache_ttl:
            return entry[1]
        return None

    def _set_cache(self, key: str, value: Any) -> None:
        self._cache[key] = (time.time(), value)
