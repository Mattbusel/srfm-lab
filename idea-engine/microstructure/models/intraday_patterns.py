"""
microstructure/models/intraday_patterns.py

Intraday microstructure pattern analysis.

Purpose
-------
Build an hourly microstructure profile for each symbol over the past 30 days.
Detect which UTC hours have structurally poor microstructure conditions
(wide spreads, low depth, low volume, high adverse selection).

This VALIDATES the IAE's entry-hour blocking by showing WHY those hours
are bad in microstructure terms.  If hour 1 UTC has 3× wider Roll spread
than the daily average, the hour block is not just empirically observed —
it has a structural microstructure explanation.

Profile dimensions per hour
----------------------------
- avg_volume_ratio   : avg_volume_this_hour / daily_avg_volume
- avg_roll_spread    : mean Roll spread estimate
- spread_ratio       : avg_roll_spread / daily_avg_roll_spread
- avg_price_range    : (high - low) / close — intrabar volatility proxy
- trade_frequency    : number of bars with volume above median (activity proxy)
- microstructure_score: 0-1 composite (1 = healthy, 0 = avoid)

Hours with microstructure_score < 0.4 are flagged as structural disadvantage
windows and reported back to hypothesis_generator.py for new hypothesis seeds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class HourlyProfile:
    """Microstructure profile for one UTC hour across a symbol's history."""
    hour: int                    # 0-23 UTC
    symbol: str
    avg_volume_ratio: float      # relative to daily average
    avg_roll_spread: float       # absolute spread estimate
    spread_ratio: float          # vs daily average spread
    avg_price_range: float       # (H-L)/C as a fraction
    bar_count: int               # number of observations
    microstructure_score: float  # 0-1, 1 = healthy
    is_structural_disadvantage: bool   # True if score < 0.4


@dataclass
class IntradayMicrostructureProfile:
    """Full 24-hour profile for one symbol."""
    symbol: str
    computed_at: str
    profiles: dict[int, HourlyProfile] = field(default_factory=dict)
    bad_hours: list[int] = field(default_factory=list)
    best_hours: list[int] = field(default_factory=list)

    def get_hour(self, hour: int) -> HourlyProfile | None:
        return self.profiles.get(hour)

    def should_block_entry(self, utc_hour: int) -> bool:
        return utc_hour in self.bad_hours


class IntradayPatternAnalyzer:
    """
    Builds hourly microstructure profiles from historical bar data.

    Parameters
    ----------
    history_days        : Number of days of history to use (default 30).
    bars_per_hour       : Expected bars per hour (1 for hourly bars).
    disadvantage_threshold : Score below this = structural disadvantage.
    best_threshold      : Score above this = ideal entry window.
    """

    def __init__(
        self,
        history_days: int = 30,
        bars_per_hour: int = 1,
        disadvantage_threshold: float = 0.40,
        best_threshold: float = 0.70,
    ) -> None:
        self.history_days = history_days
        self.bars_per_hour = bars_per_hour
        self.disadvantage_threshold = disadvantage_threshold
        self.best_threshold = best_threshold

    def build_profile(
        self,
        symbol: str,
        opens: Sequence[float],
        highs: Sequence[float],
        lows: Sequence[float],
        closes: Sequence[float],
        volumes: Sequence[float],
        utc_hours: Sequence[int],    # hour-of-day for each bar
        computed_at: str = "",
    ) -> IntradayMicrostructureProfile:
        """
        Build the full 24-hour microstructure profile.

        Parameters
        ----------
        All sequences are bar-aligned (same length).
        utc_hours : UTC hour of day (0-23) for each bar.
        """
        n = min(len(opens), len(highs), len(lows), len(closes),
                len(volumes), len(utc_hours))

        # Group bars by hour
        hour_data: dict[int, dict[str, list]] = {
            h: {"opens": [], "highs": [], "lows": [], "closes": [],
                "volumes": [], "ranges": []}
            for h in range(24)
        }

        for i in range(n):
            h = int(utc_hours[i]) % 24
            c = closes[i]
            hour_data[h]["opens"].append(opens[i])
            hour_data[h]["highs"].append(highs[i])
            hour_data[h]["lows"].append(lows[i])
            hour_data[h]["closes"].append(closes[i])
            hour_data[h]["volumes"].append(volumes[i])
            hour_data[h]["ranges"].append(
                (highs[i] - lows[i]) / c if c > 1e-12 else 0.0
            )

        # Compute daily averages for normalisation
        all_volumes = list(volumes[:n])
        daily_avg_vol = sum(all_volumes) / len(all_volumes) if all_volumes else 1.0

        # Compute daily average Roll spread across all hours
        all_closes = list(closes[:n])
        dP_all = [all_closes[i] - all_closes[i - 1] for i in range(1, len(all_closes))]
        daily_roll_spread = self._roll_spread_from_dp(dP_all, window=20)

        profiles: dict[int, HourlyProfile] = {}

        for hour in range(24):
            hd = hour_data[hour]
            if len(hd["closes"]) < 5:
                continue

            vol_list = hd["volumes"]
            avg_vol = sum(vol_list) / len(vol_list)
            vol_ratio = avg_vol / (daily_avg_vol / 24.0) if daily_avg_vol > 1e-12 else 1.0

            range_list = hd["ranges"]
            avg_range = sum(range_list) / len(range_list)

            # Roll spread for this hour's close sequence
            hour_dp = [
                hd["closes"][i] - hd["closes"][i - 1]
                for i in range(1, len(hd["closes"]))
            ]
            hour_roll = self._roll_spread_from_dp(hour_dp, window=10)
            spread_ratio = hour_roll / daily_roll_spread if daily_roll_spread > 1e-12 else 1.0

            # Composite microstructure score
            score = self._compute_score(vol_ratio, spread_ratio, avg_range)

            profiles[hour] = HourlyProfile(
                hour=hour,
                symbol=symbol,
                avg_volume_ratio=round(vol_ratio, 4),
                avg_roll_spread=round(hour_roll, 6),
                spread_ratio=round(spread_ratio, 4),
                avg_price_range=round(avg_range, 6),
                bar_count=len(hd["closes"]),
                microstructure_score=round(score, 4),
                is_structural_disadvantage=score < self.disadvantage_threshold,
            )

        bad_hours = sorted(
            h for h, p in profiles.items()
            if p.is_structural_disadvantage
        )
        best_hours = sorted(
            h for h, p in profiles.items()
            if p.microstructure_score >= self.best_threshold
        )

        return IntradayMicrostructureProfile(
            symbol=symbol,
            computed_at=computed_at,
            profiles=profiles,
            bad_hours=bad_hours,
            best_hours=best_hours,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _roll_spread_from_dp(self, dP: list[float], window: int = 20) -> float:
        """
        Mean Roll spread estimate over a list of price changes.
        Returns 0 if not enough data or covariance is non-negative.
        """
        if len(dP) < 4:
            return 0.0
        spreads: list[float] = []
        for i in range(window - 1, len(dP)):
            win = dP[max(0, i - window + 1): i + 1]
            cov = self._serial_cov(win)
            if cov < 0:
                spreads.append(2.0 * math.sqrt(-cov))
        return sum(spreads) / len(spreads) if spreads else 0.0

    @staticmethod
    def _serial_cov(dP: list[float]) -> float:
        n = len(dP)
        if n < 2:
            return 0.0
        mean = sum(dP) / n
        return sum((dP[t] - mean) * (dP[t - 1] - mean) for t in range(1, n)) / (n - 1)

    def _compute_score(
        self,
        vol_ratio: float,
        spread_ratio: float,
        avg_range: float,
    ) -> float:
        """
        Composite microstructure health score for one hour.

        Components
        ----------
        Volume score  : Higher volume = better liquidity. Penalise low vol.
                        score_v = min(1.0, vol_ratio / 1.5)
        Spread score  : Lower spread = better execution.
                        score_s = max(0.0, 1.0 - (spread_ratio - 1.0) / 3.0)
        Range score   : Moderate range is good (active but not chaotic).
                        score_r = 1.0 if range is near average, penalise extremes.

        Composite = 0.4 * score_v + 0.4 * score_s + 0.2 * score_r
        """
        score_v = min(1.0, vol_ratio / 1.5)
        score_s = max(0.0, 1.0 - max(0.0, spread_ratio - 1.0) / 3.0)
        # Range score: penalise very low range (inactive) and very high (chaotic)
        if avg_range < 1e-6:
            score_r = 0.0
        else:
            log_range = math.log(avg_range + 1e-10)
            # Typical crypto hourly range ~0.002-0.005; log around -6 to -5
            # Score peaks near log=-5.5, declines toward extremes
            score_r = max(0.0, 1.0 - abs(log_range + 5.5) / 3.0)

        return 0.40 * score_v + 0.40 * score_s + 0.20 * score_r
