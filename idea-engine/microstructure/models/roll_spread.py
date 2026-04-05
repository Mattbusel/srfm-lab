"""
microstructure/models/roll_spread.py

Roll (1984) effective spread estimator from OHLCV data.

Theory
------
Roll showed that in a frictionless market with bid-ask spread 2s, the
first-order serial covariance of transaction price changes is:
    Cov(ΔP_t, ΔP_{t-1}) = -s²

Therefore the effective half-spread is:
    s = sqrt(-Cov(ΔP_t, ΔP_{t-1}))
    effective_spread = 2s = 2 * sqrt(max(0, -Cov(ΔP_t, ΔP_{t-1})))

The max(0, ...) clamp is needed because in practice the covariance can be
positive (momentum periods), in which case Roll's estimator is undefined.
We return 0 in that case to indicate the estimator cannot function — NOT
that the spread is zero.

Why this is useful without L2 data
------------------------------------
Crypto L2 order book data is expensive.  Roll's estimator gives a rough
but useful signal from the free OHLCV data alone.  For liquidity monitoring,
the relative change in Roll spread (is it widening?) matters more than the
absolute level.

IAE application
---------------
Rolling 20-bar Roll spread is tracked per symbol.  When the estimated spread
is 3× its 30-day baseline, we flag poor execution conditions and reduce size.
The intraday_patterns module uses hourly Roll spread to characterise which
trading hours have structurally wide spreads.

Reference: Roll, R. (1984). A simple implicit measure of the effective
bid-ask spread in an efficient market. Journal of Finance, 39(4), 1127–1139.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass
class RollSpreadReading:
    """Roll effective spread estimate at one point in time."""
    symbol: str
    timestamp: str
    effective_spread: float      # 2 * sqrt(-cov) in price units
    cov_estimate: float          # raw covariance (negative = spread exists)
    rolling_baseline: float      # 30-day rolling mean spread
    spread_ratio: float          # effective_spread / rolling_baseline
    estimator_valid: bool        # False when cov >= 0 (momentum period)
    wide_spread_alert: bool      # True if spread_ratio >= 3.0


class RollSpreadCalculator:
    """
    Estimates bid-ask spread via Roll's serial covariance method.

    Parameters
    ----------
    estimation_window : Bars used for the serial covariance estimate.
    baseline_window   : Bars used for the historical baseline mean.
    wide_alert_ratio  : Flag as wide if spread / baseline > this.
    min_bars          : Minimum bars required.
    """

    def __init__(
        self,
        estimation_window: int = 20,
        baseline_window: int = 60,
        wide_alert_ratio: float = 3.0,
        min_bars: int = 22,
    ) -> None:
        self.estimation_window = estimation_window
        self.baseline_window = baseline_window
        self.wide_alert_ratio = wide_alert_ratio
        self.min_bars = min_bars

    def compute(
        self,
        symbol: str,
        closes: Sequence[float],
        timestamps: Sequence[str],
    ) -> list[RollSpreadReading]:
        """
        Compute rolling Roll spread for each bar with sufficient history.

        Parameters
        ----------
        symbol     : Ticker.
        closes     : Close prices (N bars).
        timestamps : ISO 8601 timestamps (N bars).
        """
        n = len(closes)
        if n < self.min_bars + 1:
            return []

        # Price changes
        dP = [closes[i] - closes[i - 1] for i in range(1, n)]
        ts_for_dp = list(timestamps[1:n])

        readings: list[RollSpreadReading] = []
        spread_history: list[float] = []

        for idx in range(self.estimation_window - 1, len(dP)):
            # Estimation window for serial covariance
            win_start = max(0, idx - self.estimation_window + 1)
            dP_win = dP[win_start: idx + 1]

            cov = self._serial_cov(dP_win)
            valid = cov < 0
            effective_spread = 2.0 * math.sqrt(-cov) if valid else 0.0

            if valid:
                spread_history.append(effective_spread)

            # Baseline from longer history
            baseline_start = max(0, len(spread_history) - self.baseline_window)
            baseline_slice = spread_history[baseline_start:]
            baseline = (
                sum(baseline_slice) / len(baseline_slice)
                if baseline_slice else effective_spread
            )

            spread_ratio = (
                effective_spread / baseline if baseline > 1e-12 else 1.0
            )
            wide_alert = valid and spread_ratio >= self.wide_alert_ratio

            readings.append(
                RollSpreadReading(
                    symbol=symbol,
                    timestamp=ts_for_dp[idx] if idx < len(ts_for_dp) else "",
                    effective_spread=effective_spread,
                    cov_estimate=cov,
                    rolling_baseline=baseline,
                    spread_ratio=spread_ratio,
                    estimator_valid=valid,
                    wide_spread_alert=wide_alert,
                )
            )

        return readings

    def latest(
        self,
        symbol: str,
        closes: Sequence[float],
        timestamps: Sequence[str],
    ) -> RollSpreadReading | None:
        readings = self.compute(symbol, closes, timestamps)
        return readings[-1] if readings else None

    def hourly_spreads(
        self,
        symbol: str,
        closes_by_hour: dict[int, list[float]],
    ) -> dict[int, float]:
        """
        Compute mean Roll spread per hour-of-day.

        Parameters
        ----------
        closes_by_hour : Dict mapping UTC hour (0-23) to list of close prices
                         for all bars at that hour across the history window.

        Returns
        -------
        Dict[hour, mean_effective_spread] for hours with sufficient data.
        """
        result: dict[int, float] = {}
        for hour, prices in closes_by_hour.items():
            if len(prices) < self.min_bars:
                continue
            dP = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
            spreads: list[float] = []
            for i in range(self.estimation_window - 1, len(dP)):
                win = dP[max(0, i - self.estimation_window + 1): i + 1]
                cov = self._serial_cov(win)
                if cov < 0:
                    spreads.append(2.0 * math.sqrt(-cov))
            if spreads:
                result[hour] = sum(spreads) / len(spreads)
        return result

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _serial_cov(dP: list[float]) -> float:
        """
        First-order serial covariance of price changes.
        Cov(ΔP_t, ΔP_{t-1}) over a window.
        """
        n = len(dP)
        if n < 2:
            return 0.0
        mean = sum(dP) / n
        # Pairs: (dP[t], dP[t-1]) for t = 1..n-1
        pairs = [(dP[t] - mean) * (dP[t - 1] - mean) for t in range(1, n)]
        return sum(pairs) / len(pairs)
