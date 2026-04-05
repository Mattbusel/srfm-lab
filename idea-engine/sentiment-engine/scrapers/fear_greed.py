"""
sentiment_engine/scrapers/fear_greed.py
=======================================
Client for the Alternative.me Fear & Greed Index API.

Financial rationale
-------------------
The Crypto Fear & Greed Index (0 = Extreme Fear, 100 = Extreme Greed) is a
contrarian indicator with documented mean-reverting properties over 7-30 day
horizons.  Readings below 20 historically coincide with cyclical lows within
+/- 3 weeks; readings above 80 often precede corrections of 10-25% within 2
weeks (Chen et al., 2022, "Sentiment and Cryptocurrency Returns").

We use it in two ways:
  1. Absolute value — threshold filter on generated hypotheses
     (don't generate bullish signals if fear < 20 or greed > 85)
  2. Trend direction — 7-day change in the index is a momentum signal
     independent of the absolute level.

Live endpoint: https://api.alternative.me/fng/
Docs: https://alternative.me/crypto/fear-and-greed-index/#api
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

FNG_API_URL   = "https://api.alternative.me/fng/"
FETCH_TIMEOUT = 10  # seconds
CACHE_TTL_S   = 900  # 15 minutes — index updates daily, but we cache short-term


# ---------------------------------------------------------------------------
# Sentiment labels (as defined by Alternative.me)
# ---------------------------------------------------------------------------

def _label_for_value(value: int) -> str:
    if value <= 24:
        return "Extreme Fear"
    if value <= 44:
        return "Fear"
    if value <= 55:
        return "Neutral"
    if value <= 74:
        return "Greed"
    return "Extreme Greed"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FearGreedReading:
    """
    A single Fear & Greed index data point.

    Attributes
    ----------
    value          : Index value 0-100
    label          : Human-readable classification
    timestamp      : When this reading was recorded (UTC)
    time_until_update : Seconds until the next daily update (may be None)
    """
    value:             int
    label:             str
    timestamp:         datetime
    time_until_update: Optional[int] = None

    @property
    def is_extreme_fear(self)  -> bool: return self.value <= 20
    @property
    def is_fear(self)           -> bool: return self.value <= 44
    @property
    def is_neutral(self)        -> bool: return 45 <= self.value <= 55
    @property
    def is_greed(self)          -> bool: return self.value >= 56
    @property
    def is_extreme_greed(self) -> bool: return self.value >= 80

    @classmethod
    def from_api_entry(cls, entry: dict) -> "FearGreedReading":
        """Parse a single entry from the Alternative.me /fng/ response."""
        raw_ts = entry.get("timestamp", 0)
        try:
            ts = datetime.fromtimestamp(int(raw_ts), tz=timezone.utc)
        except (TypeError, ValueError):
            ts = datetime.now(timezone.utc)

        value = int(entry.get("value", 50))
        label = entry.get("value_classification", _label_for_value(value))

        time_until: Optional[int] = None
        if "time_until_update" in entry:
            try:
                time_until = int(entry["time_until_update"])
            except (TypeError, ValueError):
                pass

        return cls(
            value=value,
            label=label,
            timestamp=ts,
            time_until_update=time_until,
        )


@dataclass
class FearGreedHistory:
    """
    Container for multiple readings with derived trend statistics.

    Attributes
    ----------
    readings   : List of FearGreedReading sorted newest-first
    current    : Most recent reading (readings[0])
    delta_7d   : Change in value over the last 7 days (positive = improving sentiment)
    trend      : 'improving' | 'deteriorating' | 'stable'
    """
    readings: list[FearGreedReading]

    @property
    def current(self) -> FearGreedReading:
        return self.readings[0]

    @property
    def delta_7d(self) -> Optional[float]:
        """Value change from 7 days ago to today."""
        if len(self.readings) < 8:
            return None
        return float(self.readings[0].value - self.readings[7].value)

    @property
    def trend(self) -> str:
        d = self.delta_7d
        if d is None:
            return "stable"
        if d > 5:
            return "improving"
        if d < -5:
            return "deteriorating"
        return "stable"

    def to_summary_dict(self) -> dict:
        return {
            "current_value":  self.current.value,
            "current_label":  self.current.label,
            "delta_7d":       self.delta_7d,
            "trend":          self.trend,
            "timestamp":      self.current.timestamp.isoformat(),
            "readings_count": len(self.readings),
        }


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class FearGreedClient:
    """
    Fetches current and historical Fear & Greed Index data from Alternative.me.

    Results are cached in memory for CACHE_TTL_S seconds to avoid hammering
    the free-tier API.

    Parameters
    ----------
    session       : Optional pre-configured requests.Session
    cache_ttl     : Override cache TTL (seconds)
    """

    def __init__(
        self,
        session:   Optional[requests.Session] = None,
        cache_ttl: int = CACHE_TTL_S,
    ) -> None:
        self._session    = session or requests.Session()
        self._session.headers["User-Agent"] = "sentiment-engine/0.1 (IAE research)"
        self._cache_ttl  = cache_ttl
        self._cache:      Optional[FearGreedHistory] = None
        self._cache_ts:   float = 0.0

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_current(self) -> FearGreedReading:
        """Return the most recent Fear & Greed reading."""
        return self.get_history(days=1).current

    def get_history(self, days: int = 30) -> FearGreedHistory:
        """
        Return up to *days* of historical readings, newest-first.

        Results are served from an in-memory cache if fresh.
        """
        now = time.monotonic()
        if self._cache is not None and (now - self._cache_ts) < self._cache_ttl:
            logger.debug("FearGreedClient: cache hit (age %.0fs)", now - self._cache_ts)
            return self._trim_history(self._cache, days)

        history = self._fetch_history(days=max(days, 30))
        self._cache    = history
        self._cache_ts = now
        return self._trim_history(history, days)

    def invalidate_cache(self) -> None:
        """Force next call to hit the API."""
        self._cache    = None
        self._cache_ts = 0.0

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _fetch_history(self, days: int) -> FearGreedHistory:
        """Fetch fresh data from the Alternative.me endpoint."""
        params = {"limit": days, "format": "json"}
        try:
            resp = self._session.get(FNG_API_URL, params=params, timeout=FETCH_TIMEOUT)
            resp.raise_for_status()
            body = resp.json()
        except requests.RequestException as exc:
            logger.error("FearGreedClient: API request failed: %s", exc)
            return self._fallback_history()
        except ValueError as exc:
            logger.error("FearGreedClient: JSON decode error: %s", exc)
            return self._fallback_history()

        raw_data = body.get("data", [])
        if not raw_data:
            logger.warning("FearGreedClient: empty data from API.")
            return self._fallback_history()

        readings = [FearGreedReading.from_api_entry(e) for e in raw_data]
        # API returns newest-first; ensure that order
        readings.sort(key=lambda r: r.timestamp, reverse=True)

        logger.info(
            "FearGreedClient: fetched %d readings; current value=%d (%s).",
            len(readings), readings[0].value, readings[0].label,
        )
        return FearGreedHistory(readings=readings)

    @staticmethod
    def _trim_history(history: FearGreedHistory, days: int) -> FearGreedHistory:
        """Return a FearGreedHistory with at most *days* readings."""
        return FearGreedHistory(readings=history.readings[:days])

    @staticmethod
    def _fallback_history() -> FearGreedHistory:
        """
        Return a neutral (50) placeholder when the API is unavailable.
        This prevents the pipeline from crashing; the low-confidence
        placeholder will be down-weighted by the aggregator.
        """
        placeholder = FearGreedReading(
            value=50,
            label="Neutral",
            timestamp=datetime.now(timezone.utc),
            time_until_update=None,
        )
        return FearGreedHistory(readings=[placeholder])
