"""
microstructure/models/adverse_selection.py

Adverse selection component of the bid-ask spread.

Theory
------
The bid-ask spread compensates market makers for three costs:
1. Order-processing costs (fixed administrative costs)
2. Inventory holding costs (bearing price risk)
3. Adverse selection costs (trading against informed traders)

The adverse selection component is the most dangerous for a directional
trader: if the spread is wide because informed traders are active, entering
a position means you are likely on the wrong side of an informed trade.

PIN (Probability of Informed Trading)
---------------------------------------
Easley et al. (1996) showed that PIN can be estimated from the arrival
rates of buy-initiated and sell-initiated trades.  With only OHLCV data
we proxy order flow imbalance using the tick rule:
    - Up-tick: close > open → buyer-initiated proxy
    - Down-tick: close < open → seller-initiated proxy

Full PIN estimation requires MLE on a mixture model with parameters:
    μ: informed trader arrival rate
    ε: uninformed trader arrival rate (both sides)
    α: probability of information event occurring
    δ: probability that the event is bad news

PIN = αμ / (αμ + 2ε)

Simplified EHO (Easley-Hvidkjaer-O'Hara) proxy
------------------------------------------------
Without tick-level data we use the VBA (Volume-imbalance Based Approximation):
    imbalance = (buy_vol - sell_vol) / total_vol
    PIN_proxy ≈ |imbalance| × volume_skewness_adjustment

This is not true PIN but captures the same phenomenon: when buy and sell
volumes are highly unequal in a period, informed traders are likely active.

IAE application
---------------
High adverse selection (PIN > 0.3) → someone knows something →
adverse_selection_risk = 'high' → avoid new entries / reduce size.

Reference: Easley, D., Kiefer, N., O'Hara, M., & Paperman, J. (1996).
Liquidity, information, and infrequently traded stocks.
Journal of Finance, 51(4), 1405–1436.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


class AdverseSelectionRisk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class AdverseSelectionReading:
    """Adverse selection estimate at one bar."""
    symbol: str
    timestamp: str
    pin_proxy: float                      # 0-1, probability of informed trading proxy
    volume_imbalance: float               # (buy - sell) / total
    rolling_mean_pin: float               # 30-bar rolling mean PIN proxy
    risk_level: AdverseSelectionRisk
    size_adjustment: float                # 1.0 (normal) → 0.0 (avoid entry)


class AdverseSelectionCalculator:
    """
    Estimates the adverse selection component of the spread.

    Computes a PIN proxy from OHLCV bar data using volume imbalance
    between buyer-initiated and seller-initiated bars.

    Parameters
    ----------
    estimation_window : Bars for rolling PIN proxy (default 20).
    pin_high_threshold : PIN proxy above this = HIGH risk (default 0.35).
    pin_medium_threshold: PIN proxy above this = MEDIUM risk (default 0.20).
    """

    def __init__(
        self,
        estimation_window: int = 20,
        pin_high_threshold: float = 0.35,
        pin_medium_threshold: float = 0.20,
        min_bars: int = 10,
    ) -> None:
        self.estimation_window = estimation_window
        self.pin_high = pin_high_threshold
        self.pin_medium = pin_medium_threshold
        self.min_bars = min_bars

    def compute(
        self,
        symbol: str,
        opens: Sequence[float],
        closes: Sequence[float],
        volumes: Sequence[float],
        timestamps: Sequence[str],
    ) -> list[AdverseSelectionReading]:
        """
        Compute rolling adverse selection estimates.

        Each bar is classified as buyer- or seller-initiated by comparing
        close to open.  Volume imbalance in the estimation window proxies
        for informed trading activity.
        """
        n = min(len(opens), len(closes), len(volumes), len(timestamps))
        if n < self.min_bars:
            return []

        # Classify each bar
        buy_vols: list[float] = []
        sell_vols: list[float] = []
        for i in range(n):
            if closes[i] > opens[i]:
                buy_vols.append(float(volumes[i]))
                sell_vols.append(0.0)
            elif closes[i] < opens[i]:
                buy_vols.append(0.0)
                sell_vols.append(float(volumes[i]))
            else:
                # Doji: split evenly
                buy_vols.append(float(volumes[i]) / 2.0)
                sell_vols.append(float(volumes[i]) / 2.0)

        readings: list[AdverseSelectionReading] = []
        pin_history: list[float] = []

        for idx in range(self.min_bars - 1, n):
            win_start = max(0, idx - self.estimation_window + 1)
            buy_w = buy_vols[win_start: idx + 1]
            sell_w = sell_vols[win_start: idx + 1]

            total_buy = sum(buy_w)
            total_sell = sum(sell_w)
            total_vol = total_buy + total_sell

            if total_vol < 1e-12:
                continue

            imbalance = (total_buy - total_sell) / total_vol
            pin_proxy = self._compute_pin_proxy(buy_w, sell_w)
            pin_history.append(pin_proxy)

            hist_start = max(0, len(pin_history) - 30)
            rolling_mean = sum(pin_history[hist_start:]) / len(pin_history[hist_start:])

            risk = self._classify_risk(pin_proxy)
            size_adj = self._size_adjustment(pin_proxy)

            readings.append(
                AdverseSelectionReading(
                    symbol=symbol,
                    timestamp=timestamps[idx],
                    pin_proxy=round(pin_proxy, 4),
                    volume_imbalance=round(imbalance, 4),
                    rolling_mean_pin=round(rolling_mean, 4),
                    risk_level=risk,
                    size_adjustment=round(size_adj, 3),
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
    ) -> AdverseSelectionReading | None:
        readings = self.compute(symbol, opens, closes, volumes, timestamps)
        return readings[-1] if readings else None

    # ------------------------------------------------------------------
    # PIN proxy computation
    # ------------------------------------------------------------------

    def _compute_pin_proxy(
        self,
        buy_vols: list[float],
        sell_vols: list[float],
    ) -> float:
        """
        Simplified PIN proxy using volume-imbalance-based approximation.

        The intuition: a high proportion of one-directional volume
        (consistently buy or consistently sell) suggests informed trader
        pressure.  We measure this as the average absolute imbalance
        across sub-windows within the estimation window.

        Returns a value in [0, 1].
        """
        n = len(buy_vols)
        if n == 0:
            return 0.0

        # Sub-window imbalances (using 5-bar sub-windows)
        sub_size = 5
        sub_imbalances: list[float] = []
        for i in range(0, n - sub_size + 1, sub_size):
            sb = sum(buy_vols[i: i + sub_size])
            ss = sum(sell_vols[i: i + sub_size])
            total = sb + ss
            if total > 1e-12:
                sub_imbalances.append(abs(sb - ss) / total)

        if not sub_imbalances:
            total_buy = sum(buy_vols)
            total_sell = sum(sell_vols)
            total = total_buy + total_sell
            if total < 1e-12:
                return 0.0
            return abs(total_buy - total_sell) / total

        return sum(sub_imbalances) / len(sub_imbalances)

    def _classify_risk(self, pin: float) -> AdverseSelectionRisk:
        if pin >= self.pin_high:
            return AdverseSelectionRisk.HIGH
        elif pin >= self.pin_medium:
            return AdverseSelectionRisk.MEDIUM
        return AdverseSelectionRisk.LOW

    def _size_adjustment(self, pin: float) -> float:
        """
        Linear interpolation from 1.0 (PIN=0) to 0.0 (PIN=0.5+).
        Above PIN=0.5, fully avoid.
        """
        if pin >= 0.5:
            return 0.0
        return max(0.0, 1.0 - (pin / 0.5))
