"""
microstructure/models/amihud.py

Amihud (2002) illiquidity ratio.

Definition
----------
ILLIQ_t = |R_t| / VOLD_t

where R_t is the daily return and VOLD_t is the dollar trading volume.

Interpretation
--------------
Higher ILLIQ = less liquid.  A given dollar of order flow moves the price
more.  This is a measure of price impact, not spread — but the two are
correlated because both rise in thin markets.

IAE application
---------------
When a symbol's current ILLIQ exceeds 2× its rolling 30-day mean, the
market is considered "thinly traded" and new entries should be avoided.
The effective bid-ask spread is wider, expected fill quality is worse, and
adverse selection risk is elevated.

Calibration notes
-----------------
In crypto, dollar volume should be denominated in USDT/USDC and already
accounts for the 24h/7d nature of the market.  No market-hours adjustment
is needed unlike for equity markets.

Reference: Amihud, Y. (2002). Illiquidity and stock returns: cross-section
and time-series effects. Journal of Financial Markets, 5(1), 31–56.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class AmihudReading:
    """Single-bar Amihud illiquidity reading."""
    symbol: str
    timestamp: str           # ISO 8601
    illiq: float             # current |R| / Volume
    rolling_mean: float      # 30-day rolling mean of ILLIQ
    rolling_std: float       # 30-day rolling std (for z-score)
    z_score: float           # (illiq - rolling_mean) / rolling_std
    is_thin: bool            # True if illiq > 2× rolling_mean

    @property
    def thinness_ratio(self) -> float:
        """illiq / rolling_mean — ratio of current to historical baseline."""
        if self.rolling_mean < 1e-20:
            return 1.0
        return self.illiq / self.rolling_mean


class AmihudCalculator:
    """
    Computes the Amihud illiquidity ratio for a single symbol.

    Designed for use with OHLCV bar data (daily or lower timeframes).
    For intraday use, aggregate to daily first — mixing bar frequencies
    distorts the ratio because volume units are not comparable.

    Parameters
    ----------
    window       : Rolling window for historical mean/std (default 30 days).
    thin_multiple: Illiquidity ratio threshold relative to mean (default 2×).
    min_bars     : Minimum bars required before emitting a reading (default 10).
    """

    def __init__(
        self,
        window: int = 30,
        thin_multiple: float = 2.0,
        min_bars: int = 10,
    ) -> None:
        self.window = window
        self.thin_multiple = thin_multiple
        self.min_bars = min_bars

    def compute(
        self,
        symbol: str,
        closes: Sequence[float],
        volumes: Sequence[float],    # dollar volume preferred; coin volume acceptable
        timestamps: Sequence[str],
    ) -> list[AmihudReading]:
        """
        Compute Amihud ILLIQ for each bar and the rolling statistics.

        Parameters
        ----------
        symbol     : Ticker symbol for labelling.
        closes     : Sequence of close prices (len N).
        volumes    : Sequence of trading volumes (len N, same unit throughout).
        timestamps : ISO 8601 timestamps (len N).

        Returns
        -------
        List of AmihudReading, one per bar (starting from bar 1 where return is defined).
        Readings before min_bars history is available use a partial window.
        """
        if len(closes) < 2:
            return []

        n = min(len(closes), len(volumes), len(timestamps))
        # Daily returns
        returns = [
            (closes[i] - closes[i - 1]) / closes[i - 1]
            if closes[i - 1] != 0 else 0.0
            for i in range(1, n)
        ]
        # ILLIQ per bar
        raw_illiq = [
            abs(returns[i]) / max(volumes[i + 1], 1e-12)
            for i in range(len(returns))
        ]

        readings: list[AmihudReading] = []
        for idx, illiq in enumerate(raw_illiq):
            bar_idx = idx + 1    # offset due to return calculation
            # Rolling window slice — use at least min_bars data
            window_start = max(0, idx - self.window + 1)
            window_slice = raw_illiq[window_start: idx + 1]

            if len(window_slice) < self.min_bars:
                # Not enough history — skip until we have min_bars
                continue

            roll_mean = sum(window_slice) / len(window_slice)
            roll_std = self._std(window_slice, roll_mean)
            z_score = (illiq - roll_mean) / max(roll_std, 1e-20)
            is_thin = illiq > self.thin_multiple * roll_mean

            readings.append(
                AmihudReading(
                    symbol=symbol,
                    timestamp=timestamps[bar_idx] if bar_idx < len(timestamps) else "",
                    illiq=illiq,
                    rolling_mean=roll_mean,
                    rolling_std=roll_std,
                    z_score=z_score,
                    is_thin=is_thin,
                )
            )

        return readings

    def latest(
        self,
        symbol: str,
        closes: Sequence[float],
        volumes: Sequence[float],
        timestamps: Sequence[str],
    ) -> AmihudReading | None:
        """Return only the most recent reading (efficient for live use)."""
        readings = self.compute(symbol, closes, volumes, timestamps)
        return readings[-1] if readings else None

    def thinness_alert(
        self,
        symbol: str,
        closes: Sequence[float],
        volumes: Sequence[float],
        timestamps: Sequence[str],
    ) -> bool:
        """
        Quick check: is the market currently too thin for new entries?
        Returns True if the latest Amihud reading flags thinly-traded conditions.
        """
        reading = self.latest(symbol, closes, volumes, timestamps)
        if reading is None:
            return False   # insufficient data — don't block by default
        return reading.is_thin

    # ------------------------------------------------------------------
    # Statistical helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _std(values: list[float], mean: float) -> float:
        """Population standard deviation of a list."""
        if len(values) < 2:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(variance)
