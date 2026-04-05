"""
alternative_data/google_trends.py
===================================
Google Trends search volume signals using pytrends.

Financial rationale
-------------------
Google search volume is a publicly observable proxy for retail investor
attention.  Key findings from the literature:

1. "crypto buy" search spikes → demand signal, leads price by 1-7 days
   (Da, Engelberg & Gao, 2011 — applied to crypto by Kristoufek 2013)
2. "crypto crash" searches spike DURING or AFTER crashes, not before —
   this is a capitulation/sentiment indicator (confirming, not leading)
3. BTC/ETH search volume rising while price is flat = accumulation phase;
   price tends to follow search within 2-4 weeks
4. Search volume acceleration (2nd derivative > 0) is more predictive
   than level: a term accelerating is capturing emerging retail attention

We compute:
  - trend_value       : Latest weekly index value (0-100, relative)
  - delta_1w          : 1-week change (first derivative)
  - acceleration      : Change in delta (second derivative, sign matters most)
  - signal_type       : 'bullish_demand' | 'capitulation' | 'neutral'

pytrends API: https://github.com/GeneralMills/pytrends
Requires: pip install pytrends
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Keywords tracked
# ---------------------------------------------------------------------------

CRYPTO_KEYWORDS: list[str] = [
    "bitcoin",
    "ethereum",
    "crypto crash",
    "crypto buy",
    "blockchain",
]

# Interest categories (pytrends category 0 = All, 7 = Finance)
PYTRENDS_CATEGORY: int = 7

# Geo: worldwide; set to 'US' for US-focused strategy
GEO: str = ""

# Timeframe: last 90 days of weekly data gives enough for 2nd derivative
TIMEFRAME: str = "today 3-m"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class TrendSignal:
    """
    Computed signal for a single keyword.

    Attributes
    ----------
    keyword       : The search term
    latest_value  : Most recent weekly index (0-100)
    delta_1w      : Week-over-week change (first derivative)
    acceleration  : Change in delta (second derivative — most predictive)
    signal_type   : 'bullish_demand' | 'rising_awareness' | 'capitulation' |
                    'decelerating' | 'neutral'
    timestamp     : When this was computed (UTC)
    """
    keyword:       str
    latest_value:  float
    delta_1w:      float
    acceleration:  float
    signal_type:   str
    timestamp:     str

    @property
    def is_bullish(self) -> bool:
        return self.signal_type == "bullish_demand"

    @property
    def is_capitulation(self) -> bool:
        return self.signal_type == "capitulation"


def _classify_signal(keyword: str, latest: float, delta: float, accel: float) -> str:
    """Classify a trend into a signal type based on keyword semantics and derivatives."""
    if "buy" in keyword.lower():
        if delta > 5 and accel > 0:
            return "bullish_demand"
        if delta > 0:
            return "rising_awareness"
        return "neutral"
    elif "crash" in keyword.lower():
        if latest > 50 and delta > 10:
            return "capitulation"
        return "neutral"
    else:  # coin name searches
        if accel > 2 and delta > 3:
            return "rising_awareness"
        if accel < -3:
            return "decelerating"
        return "neutral"


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

class GoogleTrendsFetcher:
    """
    Fetches Google Trends data for crypto keywords and computes
    trend acceleration signals.

    Parameters
    ----------
    keywords     : List of search terms to track
    timeframe    : pytrends timeframe string
    geo          : Country code or '' for worldwide
    request_delay_s : Seconds to wait between pytrends requests (rate limiting)
    """

    def __init__(
        self,
        keywords:        list[str]  = None,
        timeframe:       str        = TIMEFRAME,
        geo:             str        = GEO,
        request_delay_s: float      = 1.5,
    ) -> None:
        self.keywords        = keywords or CRYPTO_KEYWORDS
        self.timeframe       = timeframe
        self.geo             = geo
        self.request_delay_s = request_delay_s
        self._pytrends       = None
        self._init_pytrends()

    def _init_pytrends(self) -> None:
        try:
            from pytrends.request import TrendReq  # type: ignore
            self._pytrends = TrendReq(
                hl="en-US",
                tz=0,
                timeout=(10, 30),
                retries=2,
                backoff_factor=1.5,
            )
        except ImportError:
            logger.warning("pytrends not installed — using simulated trend data.")
            self._pytrends = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fetch_all(self) -> list[TrendSignal]:
        """
        Fetch trend data for all configured keywords.

        Returns
        -------
        List of TrendSignal objects, one per keyword.
        """
        signals: list[TrendSignal] = []
        ts_now = datetime.now(timezone.utc).isoformat()

        # pytrends can only handle 5 keywords at once
        batch_size = 5
        for i in range(0, len(self.keywords), batch_size):
            batch = self.keywords[i : i + batch_size]
            try:
                batch_signals = (
                    self._fetch_live_batch(batch, ts_now)
                    if self._pytrends
                    else self._fetch_mock_batch(batch, ts_now)
                )
                signals.extend(batch_signals)
                if self._pytrends and i + batch_size < len(self.keywords):
                    time.sleep(self.request_delay_s)
            except Exception as exc:
                logger.error("GoogleTrends batch %s failed: %s", batch, exc)
                signals.extend(self._fetch_mock_batch(batch, ts_now))

        logger.info("GoogleTrendsFetcher: %d keyword signals computed.", len(signals))
        return signals

    # ------------------------------------------------------------------ #
    # Internal — live                                                      #
    # ------------------------------------------------------------------ #

    def _fetch_live_batch(self, keywords: list[str], ts_now: str) -> list[TrendSignal]:
        """Fetch a batch of up to 5 keywords from the live API."""
        self._pytrends.build_payload(
            kw_list=keywords,
            cat=PYTRENDS_CATEGORY,
            timeframe=self.timeframe,
            geo=self.geo,
        )
        df = self._pytrends.interest_over_time()
        if df is None or df.empty:
            return self._fetch_mock_batch(keywords, ts_now)

        signals = []
        for kw in keywords:
            if kw not in df.columns:
                continue
            series = df[kw].dropna().tolist()
            sig = self._compute_signal(kw, series, ts_now)
            signals.append(sig)
        return signals

    # ------------------------------------------------------------------ #
    # Internal — mock / fallback                                          #
    # ------------------------------------------------------------------ #

    def _fetch_mock_batch(self, keywords: list[str], ts_now: str) -> list[TrendSignal]:
        """
        Generate realistic-looking mock trend data using deterministic
        pseudo-random values seeded by the keyword name.
        """
        import hashlib
        import math

        signals = []
        for kw in keywords:
            seed = int(hashlib.md5(kw.encode()).hexdigest()[:8], 16)
            rng  = seed % 100

            # Simulate 13 weeks of data
            base = 30 + (rng % 40)
            series: list[float] = []
            for j in range(13):
                noise = math.sin(j * 1.3 + seed * 0.01) * 8
                val   = max(0.0, min(100.0, base + noise + (j * 0.5)))
                series.append(val)

            # Inject trend based on keyword
            if "buy" in kw.lower():
                series[-1] = series[-1] * 1.15  # slight uptick on buy queries
            elif "crash" in kw.lower():
                series[-2] = series[-2] * 1.4   # crash searches spike recently

            sig = self._compute_signal(kw, series, ts_now)
            signals.append(sig)
        return signals

    @staticmethod
    def _compute_signal(keyword: str, series: list[float], ts_now: str) -> TrendSignal:
        """
        Compute first and second derivatives from a weekly index series.

        The series is expected newest-last.  We use the last 3 values for
        a robust but responsive second derivative.
        """
        if len(series) < 3:
            return TrendSignal(
                keyword=keyword, latest_value=0, delta_1w=0,
                acceleration=0, signal_type="neutral", timestamp=ts_now,
            )

        latest  = float(series[-1])
        prev    = float(series[-2])
        prev2   = float(series[-3])

        delta_1w     = latest - prev          # 1-week change
        delta_prev   = prev   - prev2         # prior 1-week change
        acceleration = delta_1w - delta_prev  # 2nd derivative

        signal_type = _classify_signal(keyword, latest, delta_1w, acceleration)

        return TrendSignal(
            keyword=keyword,
            latest_value=round(latest, 2),
            delta_1w=round(delta_1w, 2),
            acceleration=round(acceleration, 2),
            signal_type=signal_type,
            timestamp=ts_now,
        )
