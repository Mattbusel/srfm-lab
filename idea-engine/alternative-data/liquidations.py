"""
alternative_data/liquidations.py
===================================
Binance liquidation feed simulation and cascade detection.

Financial rationale
-------------------
Liquidations in crypto perpetual markets are forced position closures
triggered by margin insufficient to cover losses.  They have distinctive
market-impact properties:

1. Longs liquidated (price falling):
   - Exchange sells the underlying to recover margin
   - This creates additional sell pressure, which can trigger more liquidations
   - "Long liquidation cascade": self-reinforcing downward spiral
   - Strategy: these events mark SHORT-TERM bottoms within 30-120 min
     (the selling exhausts itself), but can be violent during transit

2. Shorts liquidated (price rising):
   - Exchange buys the underlying
   - Creates self-reinforcing buying — "short squeeze"
   - Cascades can produce 5-30% moves in < 2 hours in crypto

Spike detection:
  We compute a rolling z-score of 5-minute liquidation volume.
  z-score > 2.5  → cascade event (significant statistical outlier)
  z-score > 4.0  → extreme cascade (rare, high-impact event)

The Binance public WebSocket streams real-time liquidations; the REST
snapshot below provides the last N liquidations.  We simulate here
with realistic models.

Real-time WebSocket (for future production upgrade):
  wss://fstream.binance.com/ws/!forceOrder@arr
"""

from __future__ import annotations

import logging
import math
import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

TRACKED_SYMBOLS: list[str] = [
    "BTC", "ETH", "SOL", "BNB", "XRP",
    "DOGE", "ADA", "AVAX", "LINK", "DOT",
]

# Simulation parameters per symbol: (typical_5min_liq_usd, hourly_events)
# Based on approximate historical Binance liquidation data
_SYM_PARAMS: dict[str, tuple[float, int]] = {
    "BTC":  (2_000_000, 8),   # $2M typical 5-min liquidations, ~8 events/hour
    "ETH":  (1_000_000, 10),
    "SOL":  (400_000,   12),
    "BNB":  (200_000,   6),
    "XRP":  (150_000,   8),
    "DOGE": (100_000,   6),
    "ADA":  (80_000,    5),
    "AVAX": (90_000,    6),
    "LINK": (60_000,    5),
    "DOT":  (70_000,    5),
}

DEFAULT_PARAMS = (100_000, 6)

HISTORY_WINDOW: int = 12   # number of 5-min buckets to keep (= 1 hour)
CASCADE_Z_THRESH:     float = 2.5
EXTREME_Z_THRESH:     float = 4.0


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class LiquidationEvent:
    """
    A single liquidation event.

    Attributes
    ----------
    symbol    : Canonical ticker
    side      : 'long' (position was long, forced to sell) | 'short'
    usd_value : USD value of the liquidated position
    timestamp : Event timestamp (UTC ISO)
    """
    symbol:    str
    side:      str     # 'long' | 'short'
    usd_value: float
    timestamp: str


@dataclass
class LiquidationSnapshot:
    """
    Rolling liquidation volume snapshot for a symbol.

    Attributes
    ----------
    symbol              : Canonical ticker
    long_liq_1h_usd     : Total long liquidations in last hour (USD)
    short_liq_1h_usd    : Total short liquidations in last hour (USD)
    total_liq_1h_usd    : Total liquidations (both sides) in last hour (USD)
    dominant_side       : 'long' | 'short' | 'balanced'
    z_score             : Statistical z-score of current 5-min volume
    is_cascade          : True if z_score > CASCADE_Z_THRESH
    is_extreme_cascade  : True if z_score > EXTREME_Z_THRESH
    signal_type         : 'long_cascade_bottom' | 'short_squeeze_cascade' |
                          'elevated_liq' | 'normal'
    events              : List of individual LiquidationEvent objects
    timestamp           : UTC ISO string
    """
    symbol:             str
    long_liq_1h_usd:    float
    short_liq_1h_usd:   float
    total_liq_1h_usd:   float
    dominant_side:      str
    z_score:            float
    is_cascade:         bool
    is_extreme_cascade: bool
    signal_type:        str
    events:             list[LiquidationEvent] = field(default_factory=list)
    timestamp:          str                     = ""

    @property
    def is_volatility_event(self) -> bool:
        return self.is_cascade or self.is_extreme_cascade


def _classify_liq_signal(
    dominant_side: str,
    z_score:       float,
    long_usd:      float,
    short_usd:     float,
) -> str:
    if z_score >= CASCADE_Z_THRESH:
        if dominant_side == "long":
            return "long_cascade_bottom"
        if dominant_side == "short":
            return "short_squeeze_cascade"
        return "elevated_liq"
    return "normal"


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

class LiquidationSimulator:
    """
    Simulates Binance liquidation feed data with realistic cascade detection.

    Maintains a rolling 1-hour history of 5-minute volume buckets per symbol
    to compute z-scores for cascade detection.

    Parameters
    ----------
    symbols   : List of symbols to simulate
    seed_salt : Seed modifier for simulation variation
    """

    def __init__(
        self,
        symbols:   list[str] = None,
        seed_salt: int       = 0,
    ) -> None:
        self.symbols   = symbols or TRACKED_SYMBOLS
        self.seed_salt = seed_salt
        # Rolling history of 5-min bucket totals: {symbol: deque of floats}
        self._history: dict[str, deque] = {
            sym: deque(maxlen=HISTORY_WINDOW) for sym in self.symbols
        }

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def fetch_all(self) -> list[LiquidationSnapshot]:
        """
        Simulate current liquidation snapshot for all tracked symbols.

        Returns
        -------
        List sorted by total_liq_1h_usd descending.
        """
        ts_now  = datetime.now(timezone.utc).isoformat()
        results: list[LiquidationSnapshot] = []

        for sym in self.symbols:
            snap = self._simulate_symbol(sym, ts_now)
            results.append(snap)

        results.sort(key=lambda s: s.total_liq_1h_usd, reverse=True)
        logger.info(
            "LiquidationSimulator: simulated liquidations for %d symbols.", len(results)
        )
        return results

    def get_cascades(self) -> list[LiquidationSnapshot]:
        """Return only snapshots currently experiencing a cascade."""
        return [s for s in self.fetch_all() if s.is_cascade]

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _simulate_symbol(self, symbol: str, ts_now: str) -> LiquidationSnapshot:
        """Simulate one 5-minute liquidation bucket and update rolling history."""
        base_vol, hourly_events = _SYM_PARAMS.get(symbol, DEFAULT_PARAMS)

        # Seed: consistent within a 5-minute window
        bucket = int(time.time() / 300)
        rng    = random.Random(hash(symbol) ^ (bucket * 6271) ^ self.seed_salt)

        # Occasionally inject a cascade event (5% of buckets)
        inject_cascade = rng.random() < 0.05
        cascade_multiplier = rng.uniform(4, 8) if inject_cascade else 1.0

        # Simulate events in this bucket
        n_events  = max(1, int(rng.gauss(hourly_events / 12, hourly_events / 24)))
        events:   list[LiquidationEvent] = []
        long_vol  = 0.0
        short_vol = 0.0

        for _ in range(n_events):
            side     = "long" if rng.random() < 0.55 else "short"  # slight long bias in bull market
            usd_val  = max(100.0, rng.expovariate(1 / base_vol) * cascade_multiplier)
            events.append(LiquidationEvent(
                symbol=symbol, side=side, usd_value=usd_val, timestamp=ts_now
            ))
            if side == "long":
                long_vol += usd_val
            else:
                short_vol += usd_val

        current_bucket_total = long_vol + short_vol

        # Update rolling history
        hist = self._history[symbol]
        hist.append(current_bucket_total)

        # Ensure at least 6 data points for meaningful stats
        while len(hist) < 6:
            hist.appendleft(base_vol * rng.uniform(0.5, 1.5))

        # Z-score of current bucket vs rolling history
        vals   = list(hist)
        mean   = sum(vals[:-1]) / len(vals[:-1])
        std    = math.sqrt(sum((x - mean) ** 2 for x in vals[:-1]) / len(vals[:-1])) or 1.0
        z_score = (current_bucket_total - mean) / std

        # 1-hour totals: scale from last-bucket to full hour
        scale_factor = 12  # 12 × 5-min buckets per hour
        # Use the rolling history sum for a better 1-hour estimate
        total_1h      = sum(vals) * (HISTORY_WINDOW / len(vals))
        long_1h       = total_1h * (long_vol / current_bucket_total if current_bucket_total > 0 else 0.55)
        short_1h      = total_1h - long_1h

        dominant_side = "long" if long_vol > short_vol * 1.2 else (
            "short" if short_vol > long_vol * 1.2 else "balanced"
        )
        is_cascade         = z_score > CASCADE_Z_THRESH
        is_extreme_cascade = z_score > EXTREME_Z_THRESH
        signal_type        = _classify_liq_signal(dominant_side, z_score, long_1h, short_1h)

        return LiquidationSnapshot(
            symbol=symbol,
            long_liq_1h_usd=round(long_1h, 0),
            short_liq_1h_usd=round(short_1h, 0),
            total_liq_1h_usd=round(total_1h, 0),
            dominant_side=dominant_side,
            z_score=round(z_score, 3),
            is_cascade=is_cascade,
            is_extreme_cascade=is_extreme_cascade,
            signal_type=signal_type,
            events=events[:20],  # cap to avoid bloat
            timestamp=ts_now,
        )
