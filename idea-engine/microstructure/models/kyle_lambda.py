"""
microstructure/models/kyle_lambda.py

Kyle's Lambda — price impact coefficient.

Theory (Kyle, 1985)
--------------------
In Kyle's sequential trade model, market makers set prices such that:
    ΔP = λ · Q + ε

where ΔP is the price change, Q is the signed order flow (positive = buy),
and λ (lambda) is the price impact coefficient.

Higher λ → price moves more per unit of order flow → market is less liquid →
position sizing should be reduced to avoid self-moving the price.

Empirical estimation from OHLCV
---------------------------------
Without tick data or L2 order book, we proxy signed volume as:
    signed_vol_t = volume_t × sign(close_t - open_t)

This is sometimes called the "bar-signed volume" convention. It's an
approximation: intrabar flow direction is unknown, so we use bar direction.

Kyle's lambda is then estimated as the OLS slope of:
    ΔP_t ~ λ · ΔV_t

Equivalently: λ = Cov(ΔP, ΔV) / Var(ΔV)

Units: ΔP is in price units, ΔV is in volume units.
For cross-asset comparability, use dollar volume so ΔV is in USD.

IAE application
---------------
High lambda means position sizes should shrink — the IAE should reduce
recommended_size_multiplier in MicrostructureSignal when λ is elevated
relative to its historical distribution.

Reference: Kyle, A.S. (1985). Continuous auctions and insider trading.
Econometrica, 53(6), 1315–1335.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class KyleLambdaReading:
    """Kyle's lambda estimate over a rolling window."""
    symbol: str
    timestamp: str
    lambda_val: float          # estimated price impact coefficient
    rolling_mean: float        # historical mean lambda (for percentile context)
    rolling_std: float
    percentile: float          # 0-1, where current lambda sits in its history
    size_multiplier: float     # recommended position size multiplier (1.0=normal)


class KyleLambdaCalculator:
    """
    Estimates Kyle's lambda from OHLCV bar data.

    Parameters
    ----------
    window       : Rolling window for lambda estimation and history (default 30 bars).
    min_bars     : Minimum bars required to estimate lambda (default 20).
    high_impact_percentile : Lambda percentile above which to reduce sizing.
    """

    def __init__(
        self,
        window: int = 30,
        min_bars: int = 20,
        high_impact_percentile: float = 0.75,
    ) -> None:
        self.window = window
        self.min_bars = min_bars
        self.high_impact_percentile = high_impact_percentile

    def compute(
        self,
        symbol: str,
        opens: Sequence[float],
        closes: Sequence[float],
        volumes: Sequence[float],   # dollar volume preferred
        timestamps: Sequence[str],
    ) -> list[KyleLambdaReading]:
        """
        Compute Kyle's lambda for each bar with sufficient history.

        For each bar, a rolling window of (closes, volumes) is used to
        estimate the Cov(ΔP, ΔV) / Var(ΔV) OLS slope.
        """
        n = min(len(opens), len(closes), len(volumes), len(timestamps))
        if n < self.min_bars + 1:
            return []

        # Compute bar-level price changes and signed volumes
        dp = [closes[i] - closes[i - 1] for i in range(1, n)]
        dv = [
            volumes[i] * (1.0 if closes[i] >= opens[i] else -1.0)
            for i in range(1, n)
        ]
        signed_dv = [dv[i] for i in range(len(dv))]

        readings: list[KyleLambdaReading] = []
        lambda_history: list[float] = []

        for idx in range(self.min_bars - 1, len(dp)):
            window_start = max(0, idx - self.window + 1)
            dp_win = dp[window_start: idx + 1]
            dv_win = signed_dv[window_start: idx + 1]

            lam = self._estimate_lambda(dp_win, dv_win)
            if lam is None:
                continue

            lambda_history.append(lam)

            hist_mean = sum(lambda_history) / len(lambda_history)
            hist_std = self._std(lambda_history, hist_mean)
            pct = self._percentile_rank(lambda_history, lam)
            size_mult = self._size_multiplier(pct)

            bar_ts_idx = idx + 1   # timestamps offset by 1 due to diff
            ts = timestamps[bar_ts_idx] if bar_ts_idx < len(timestamps) else ""

            readings.append(
                KyleLambdaReading(
                    symbol=symbol,
                    timestamp=ts,
                    lambda_val=lam,
                    rolling_mean=hist_mean,
                    rolling_std=hist_std,
                    percentile=pct,
                    size_multiplier=size_mult,
                )
            )

        return readings

    def latest(
        self,
        symbol: str,
        opens: Sequence[float],
        closes: Sequence[float],
        volumes: Sequence[float],
        timestamps: Sequence[str],
    ) -> KyleLambdaReading | None:
        readings = self.compute(symbol, opens, closes, volumes, timestamps)
        return readings[-1] if readings else None

    # ------------------------------------------------------------------
    # Estimation helpers
    # ------------------------------------------------------------------

    def _estimate_lambda(
        self,
        dp: list[float],
        dv: list[float],
    ) -> float | None:
        """
        OLS estimate of λ in:  ΔP_t = λ · ΔV_t + ε_t

        λ = Σ(dv_i · dp_i) / Σ(dv_i²)

        Returns None if Var(ΔV) is zero (no variance in flow).
        """
        n = len(dp)
        if n < 3:
            return None
        sum_dv2 = sum(v ** 2 for v in dv)
        if sum_dv2 < 1e-20:
            return None
        sum_dp_dv = sum(dp[i] * dv[i] for i in range(n))
        return sum_dp_dv / sum_dv2

    def _size_multiplier(self, percentile: float) -> float:
        """
        Map lambda percentile to a recommended position size multiplier.

        Logic:
            0.00 – 0.50 (low impact)     → 1.00 (full size)
            0.50 – 0.75 (moderate impact)→ 0.75 (reduce 25%)
            0.75 – 0.90 (high impact)    → 0.50 (reduce 50%)
            0.90 – 1.00 (extreme impact) → 0.25 (reduce 75%)
        """
        if percentile < 0.50:
            return 1.00
        elif percentile < 0.75:
            return 0.75
        elif percentile < 0.90:
            return 0.50
        else:
            return 0.25

    @staticmethod
    def _std(values: list[float], mean: float) -> float:
        if len(values) < 2:
            return 0.0
        return math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))

    @staticmethod
    def _percentile_rank(sorted_values: list[float], target: float) -> float:
        """Fraction of values <= target (empirical CDF at target)."""
        if not sorted_values:
            return 0.5
        count_le = sum(1 for v in sorted_values if v <= target)
        return count_le / len(sorted_values)
