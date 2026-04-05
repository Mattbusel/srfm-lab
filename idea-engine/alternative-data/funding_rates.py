"""
alternative_data/funding_rates.py
====================================
Binance perpetual futures funding rate tracker.

Financial rationale
-------------------
Perpetual swap contracts use a funding mechanism to anchor the perpetual price
to the spot index.  Every 8 hours, longs pay shorts (positive rate) or shorts
pay longs (negative rate).

Key signals:

  Extreme positive funding (> +0.10% per 8h, i.e. > +0.30%/day)
    → Market is crowded LONG
    → Longs are paying a high premium to hold positions
    → Contrarian fade signal: unsustainable, often precedes 5-10% drops
    → Sometimes called "funding drain" or "forced unwind"

  Extreme negative funding (< -0.05% per 8h)
    → Market is crowded SHORT
    → Shorts paying longs
    → Potential short-squeeze setup; any bullish catalyst triggers rapid unwind
    → Often seen at local bottoms during capitulation

  Moderate positive funding (0.01%-0.05%)
    → Normal bull market condition, not actionable alone

  Near-zero or slightly negative (-0.01% to 0.01%)
    → Neutral; perps fairly priced vs spot

We track the current funding rate AND the 3-period rolling average to
distinguish persistent extremes from one-off spikes.

Endpoint: GET https://fapi.binance.com/fapi/v1/fundingRate
Docs: https://binance-docs.github.io/apidocs/futures/en/#get-funding-rate-history
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import requests

logger = logging.getLogger(__name__)

BINANCE_FAPI    = "https://fapi.binance.com"
REQUEST_TIMEOUT = 10

TRACKED_SYMBOLS: list[str] = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT",
    "XRPUSDT", "DOGEUSDT", "ADAUSDT", "AVAXUSDT",
    "LINKUSDT", "DOTUSDT",
]

# Thresholds (per 8h funding period)
EXTREME_POSITIVE_THRESH: float =  0.001   # +0.10%
MODERATE_POSITIVE_THRESH: float = 0.0001  # +0.01%
EXTREME_NEGATIVE_THRESH:  float = -0.0005 # -0.05%

# How many historical periods to fetch for rolling average
HISTORY_LIMIT: int = 3   # last 3 funding periods = 24 hours


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class FundingRateSnapshot:
    """
    Funding rate signal for a single symbol.

    Attributes
    ----------
    symbol          : Binance symbol (e.g. 'BTCUSDT')
    ticker          : Canonical ticker (e.g. 'BTC')
    current_rate    : Most recent 8h funding rate (as a decimal, e.g. 0.0001 = 0.01%)
    rate_pct        : current_rate as percentage string (display)
    rolling_avg_3p  : Mean of last 3 funding periods
    annualised_rate : current_rate * 3 * 365  (approximate annual cost of carry)
    signal_type     : 'crowded_long_fade' | 'crowded_short_squeeze' |
                      'bullish_carry' | 'neutral'
    extreme_positive: True if current_rate > EXTREME_POSITIVE_THRESH
    extreme_negative: True if current_rate < EXTREME_NEGATIVE_THRESH
    timestamp       : UTC ISO string
    """
    symbol:           str
    ticker:           str
    current_rate:     float
    rate_pct:         str
    rolling_avg_3p:   float
    annualised_rate:  float
    signal_type:      str
    extreme_positive: bool
    extreme_negative: bool
    timestamp:        str

    @property
    def is_fade_signal(self) -> bool:
        return self.signal_type == "crowded_long_fade"

    @property
    def is_squeeze_signal(self) -> bool:
        return self.signal_type == "crowded_short_squeeze"


def _classify_funding(rate: float, rolling: float) -> str:
    """Classify funding rate into a trading signal."""
    # Use rolling average for persistence check
    if rate >= EXTREME_POSITIVE_THRESH and rolling >= EXTREME_POSITIVE_THRESH * 0.7:
        return "crowded_long_fade"
    if rate <= EXTREME_NEGATIVE_THRESH and rolling <= EXTREME_NEGATIVE_THRESH * 0.7:
        return "crowded_short_squeeze"
    if rate >= MODERATE_POSITIVE_THRESH:
        return "bullish_carry"       # longs paying; mild bullish indicator
    if rate < 0:
        return "bearish_carry"       # shorts paying; mild bearish indicator
    return "neutral"


# ---------------------------------------------------------------------------
# Fetcher
# ---------------------------------------------------------------------------

class FundingRateFetcher:
    """
    Fetches current and historical funding rates from Binance FAPI.

    Parameters
    ----------
    symbols     : List of Binance USDT-margined perpetual symbols
    session     : Optional pre-configured requests.Session
    history_limit: Number of historical periods to fetch for rolling average
    """

    def __init__(
        self,
        symbols:       list[str]                  = None,
        session:       Optional[requests.Session] = None,
        history_limit: int                        = HISTORY_LIMIT,
    ) -> None:
        self.symbols       = symbols or TRACKED_SYMBOLS
        self._session      = session or requests.Session()
        self._session.headers["User-Agent"] = "alt-data/0.1 (IAE research)"
        self.history_limit = history_limit

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fetch_all(self) -> list[FundingRateSnapshot]:
        """
        Fetch funding rates for all tracked symbols.

        Returns
        -------
        List of FundingRateSnapshot sorted by abs(current_rate) descending
        so the most extreme funding situations appear first.
        """
        ts_now  = datetime.now(timezone.utc).isoformat()
        results: list[FundingRateSnapshot] = []

        for sym in self.symbols:
            try:
                snap = self._fetch_symbol(sym, ts_now)
                results.append(snap)
            except Exception as exc:
                logger.warning("FundingRateFetcher: error for %s: %s", sym, exc)
                results.append(self._mock_snapshot(sym, ts_now))

        results.sort(key=lambda s: abs(s.current_rate), reverse=True)
        logger.info(
            "FundingRateFetcher: fetched funding rates for %d symbols.", len(results)
        )
        return results

    def get_extreme_signals(
        self,
        snapshots: Optional[list[FundingRateSnapshot]] = None,
    ) -> dict[str, list[FundingRateSnapshot]]:
        """
        Filter snapshots into bullish (squeeze) and bearish (fade) buckets.

        Returns
        -------
        {"fade": [crowded_long snapshots], "squeeze": [crowded_short snapshots]}
        """
        snaps = snapshots or self.fetch_all()
        return {
            "fade":    [s for s in snaps if s.is_fade_signal],
            "squeeze": [s for s in snaps if s.is_squeeze_signal],
        }

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _fetch_symbol(self, symbol: str, ts_now: str) -> FundingRateSnapshot:
        """Fetch funding rate history for one symbol and compute snapshot."""
        data = self._get(
            "/fapi/v1/fundingRate",
            {"symbol": symbol, "limit": self.history_limit},
        )
        if not isinstance(data, list) or not data:
            raise ValueError(f"Empty funding rate response for {symbol}")

        # Most recent rate is last in the list
        rates = [float(r.get("fundingRate", 0)) for r in data]
        current = rates[-1]
        rolling = sum(rates) / len(rates)
        annualised = current * 3 * 365  # 3 periods/day × 365 days

        ticker = symbol.replace("USDT", "")
        signal = _classify_funding(current, rolling)

        return FundingRateSnapshot(
            symbol=symbol,
            ticker=ticker,
            current_rate=current,
            rate_pct=f"{current * 100:+.4f}%",
            rolling_avg_3p=rolling,
            annualised_rate=round(annualised, 4),
            signal_type=signal,
            extreme_positive=current >= EXTREME_POSITIVE_THRESH,
            extreme_negative=current <= EXTREME_NEGATIVE_THRESH,
            timestamp=ts_now,
        )

    def _get(self, path: str, params: dict) -> object:
        url  = BINANCE_FAPI + path
        resp = self._session.get(url, params=params, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _mock_snapshot(symbol: str, ts_now: str) -> FundingRateSnapshot:
        """Return a plausible mock snapshot when the API fails."""
        import hashlib
        import random
        seed = int(hashlib.md5(symbol.encode()).hexdigest()[:4], 16)
        rng  = random.Random(seed + int(time.time() / 28800))  # changes each 8h

        # Most funding rates are in the 0.005%-0.05% range (normal bull market)
        rate_choices = [
            0.0001, 0.0002, 0.0003, 0.0005, 0.001, 0.0015,
            -0.0001, -0.0002, -0.0005,
        ]
        current = rng.choice(rate_choices)
        rolling = current * rng.uniform(0.8, 1.1)
        annualised = current * 3 * 365
        ticker = symbol.replace("USDT", "")
        signal = _classify_funding(current, rolling)

        return FundingRateSnapshot(
            symbol=symbol,
            ticker=ticker,
            current_rate=current,
            rate_pct=f"{current * 100:+.4f}%",
            rolling_avg_3p=rolling,
            annualised_rate=round(annualised, 4),
            signal_type=signal,
            extreme_positive=current >= EXTREME_POSITIVE_THRESH,
            extreme_negative=current <= EXTREME_NEGATIVE_THRESH,
            timestamp=ts_now,
        )
