"""
intraday_patterns.py — Intraday seasonality analytics.

Covers:
  - Volume by time of day (U-shaped pattern)
  - Volatility by time of day
  - Bid-ask spread intraday seasonality
  - Day-of-week effects (volume, volatility, returns)
  - MOC/MOO (Market-on-Close / Market-on-Open) imbalance patterns
  - Liquidity scoring by time slot
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta, timezone
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np

from trade_classification import Trade, SyntheticTradeGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MARKET_OPEN  = time(9, 30)
MARKET_CLOSE = time(16, 0)
MOO_END      = time(9, 45)     # first 15 min
MOC_START    = time(15, 45)    # last 15 min

SLOT_MINUTES = 30              # 30-minute time slots by default
TRADING_HOURS_PER_DAY = 6.5
SLOTS_PER_DAY = int(TRADING_HOURS_PER_DAY * 60 / SLOT_MINUTES)   # = 13

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TimeSlot(Enum):
    MOO_RUSH   = "moo_rush"     # 9:30-9:45 — opening rush
    MORNING    = "morning"      # 9:45-11:30
    MIDDAY     = "midday"       # 11:30-14:00 — lunch doldrums
    AFTERNOON  = "afternoon"    # 14:00-15:30
    MOC_RUSH   = "moc_rush"     # 15:30-16:00 — closing rush


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class IntradaySlot:
    """Statistics for a 30-minute intraday slot."""
    slot_label: str        # e.g. "09:30"
    slot_start: time
    slot_end: time
    avg_volume: float
    volume_share: float    # fraction of daily volume
    avg_volatility: float  # annualized vol from returns in slot
    avg_spread_bps: float
    avg_trade_size: float
    n_trades_avg: float
    time_slot: TimeSlot
    liquidity_score: float  # 0-1 (higher = more liquid)

    @property
    def is_moo(self) -> bool:
        return self.time_slot == TimeSlot.MOO_RUSH

    @property
    def is_moc(self) -> bool:
        return self.time_slot == TimeSlot.MOC_RUSH


@dataclass
class DayOfWeekPattern:
    day_name: str
    avg_volume: float
    volume_vs_mean: float    # relative to weekly mean
    avg_daily_vol: float     # realized daily volatility
    avg_return: float        # average daily return
    avg_spread_bps: float
    n_observations: int


@dataclass
class MOCMOOImbalance:
    """Market-on-Open / Market-on-Close imbalance statistics."""
    date: str
    moo_buy_imbalance: float     # net buy imbalance at open
    moo_volume_pct: float        # MOO volume as % of daily
    moc_buy_imbalance: float
    moc_volume_pct: float
    moo_price_impact_bps: float  # estimated impact from imbalance
    moc_price_impact_bps: float


@dataclass
class IntradaySeasonality:
    ticker: str
    computation_date: datetime
    slots: List[IntradaySlot]
    dow_patterns: List[DayOfWeekPattern]
    moc_moo_summary: Dict
    peak_volume_slot: IntradaySlot
    lowest_spread_slot: IntradaySlot
    worst_spread_slot: IntradaySlot
    optimal_trading_window: Tuple[time, time]


# ---------------------------------------------------------------------------
# Time slot classifier
# ---------------------------------------------------------------------------

class TimeSlotClassifier:
    """Assigns trades to time-of-day slots."""

    def classify_time(self, t: time) -> TimeSlot:
        if t <= MOO_END:
            return TimeSlot.MOO_RUSH
        elif t <= time(11, 30):
            return TimeSlot.MORNING
        elif t <= time(14, 0):
            return TimeSlot.MIDDAY
        elif t <= time(15, 30):
            return TimeSlot.AFTERNOON
        else:
            return TimeSlot.MOC_RUSH

    def slot_label(self, t: time) -> str:
        """Round time down to nearest slot_minutes interval."""
        total_minutes = t.hour * 60 + t.minute
        slot_start_min = (total_minutes // SLOT_MINUTES) * SLOT_MINUTES
        h = slot_start_min // 60
        m = slot_start_min % 60
        return f"{h:02d}:{m:02d}"

    def all_slot_labels(self) -> List[str]:
        """Generate all 30-min slot labels during trading hours."""
        labels = []
        t = 9 * 60 + 30  # 09:30
        end = 16 * 60      # 16:00
        while t < end:
            h, m = divmod(t, 60)
            labels.append(f"{h:02d}:{m:02d}")
            t += SLOT_MINUTES
        return labels


# ---------------------------------------------------------------------------
# Intraday volume pattern
# ---------------------------------------------------------------------------

class IntradayVolumeAnalyzer:
    """Computes average volume by time slot from historical trade data."""

    def __init__(self):
        self.classifier = TimeSlotClassifier()

    def build_profile(self, all_days_trades: Dict[str, List[Trade]]) -> Dict[str, float]:
        """
        Build average volume profile from multiple trading days.
        all_days_trades: {date_str: [trades]}
        """
        slot_volumes: Dict[str, List[float]] = defaultdict(list)

        for date_str, trades in all_days_trades.items():
            day_vol = defaultdict(float)
            for trade in trades:
                label = self.classifier.slot_label(trade.timestamp.time())
                day_vol[label] += trade.size

            for label, vol in day_vol.items():
                slot_volumes[label].append(vol)

        avg_vol = {label: float(np.mean(vols)) for label, vols in slot_volumes.items()}

        # Normalize to daily volume share
        total = sum(avg_vol.values())
        if total > 0:
            vol_share = {label: v / total for label, v in avg_vol.items()}
        else:
            vol_share = {label: 1 / len(avg_vol) for label in avg_vol}

        return vol_share

    def synthetic_volume_profile(self) -> Dict[str, float]:
        """
        Return a realistic synthetic U-shaped intraday volume profile.
        Based on empirical research on NYSE/NASDAQ volume patterns.
        """
        labels = TimeSlotClassifier().all_slot_labels()
        n = len(labels)

        # U-shaped volume: high at open and close, low midday
        profile = {}
        for i, label in enumerate(labels):
            x = i / (n - 1)  # 0 to 1
            # U-shape: high at 0 and 1, minimum around 0.4
            vol = 0.35 * math.exp(-8 * (x - 0.0)**2) + 0.25 * math.exp(-8 * (x - 1.0)**2) + 0.08
            # Add slight afternoon ramp-up
            if x > 0.7:
                vol += 0.1 * (x - 0.7)
            profile[label] = vol

        total = sum(profile.values())
        return {k: v / total for k, v in profile.items()}


# ---------------------------------------------------------------------------
# Intraday volatility pattern
# ---------------------------------------------------------------------------

class IntradayVolatilityAnalyzer:
    """Computes intraday volatility pattern by time slot."""

    def __init__(self):
        self.classifier = TimeSlotClassifier()

    def synthetic_vol_profile(self) -> Dict[str, float]:
        """
        Synthetic volatility profile: highest at open/close, lowest midday.
        Returns annualized volatility (fraction) by slot.
        """
        labels = TimeSlotClassifier().all_slot_labels()
        n = len(labels)
        profile = {}

        for i, label in enumerate(labels):
            x = i / (n - 1)
            # Similar U-shape to volume but slightly different parameters
            vol = (
                0.40 * math.exp(-10 * x**2) +           # open spike
                0.30 * math.exp(-10 * (x - 1.0)**2) +   # close spike
                0.12                                       # baseline
            )
            profile[label] = vol

        # Annualize: scale so peak is ~0.35 (35% annualized at open)
        max_v = max(profile.values())
        return {k: v / max_v * 0.35 for k, v in profile.items()}

    def vol_of_vol(self, vol_profile: Dict[str, float]) -> float:
        """Intraday volatility of volatility."""
        vals = list(vol_profile.values())
        return float(np.std(vals) / np.mean(vals)) if vals else 0.0


# ---------------------------------------------------------------------------
# Intraday spread pattern
# ---------------------------------------------------------------------------

class IntradaySpreadAnalyzer:
    """Computes bid-ask spread intraday seasonality."""

    def synthetic_spread_profile(self) -> Dict[str, float]:
        """
        Synthetic spread profile: widest at open, narrows, widens slightly at close.
        Returns spread in bps.
        """
        labels = TimeSlotClassifier().all_slot_labels()
        n = len(labels)
        profile = {}

        for i, label in enumerate(labels):
            x = i / (n - 1)
            # Spread: high at open, decays, slight uptick at close
            spread_bps = (
                15 * math.exp(-8 * x) +    # open-wide spread
                1.5 * math.exp(-5 * (x - 1.0)**2) +  # close bump
                2.5                          # baseline
            )
            profile[label] = spread_bps

        return profile


# ---------------------------------------------------------------------------
# Day-of-week effect analyzer
# ---------------------------------------------------------------------------

class DayOfWeekAnalyzer:
    """Analyzes day-of-week effects on volume, volatility, and returns."""

    def synthetic_dow_patterns(self) -> List[DayOfWeekPattern]:
        """Return synthetic DOW patterns based on empirical research."""
        # Stylized facts:
        # - Monday: slightly lower volume, often negative return (Monday effect)
        # - Friday: higher volume (position closing), mixed returns
        # - Tuesday-Thursday: highest volume and most efficient prices
        patterns = [
            DayOfWeekPattern("Monday",    avg_volume=0.88, volume_vs_mean=-0.12, avg_daily_vol=0.022, avg_return=-0.0003, avg_spread_bps=8.5, n_observations=52),
            DayOfWeekPattern("Tuesday",   avg_volume=1.05, volume_vs_mean=+0.05, avg_daily_vol=0.019, avg_return=+0.0004, avg_spread_bps=7.8, n_observations=52),
            DayOfWeekPattern("Wednesday", avg_volume=1.08, volume_vs_mean=+0.08, avg_daily_vol=0.018, avg_return=+0.0003, avg_spread_bps=7.6, n_observations=52),
            DayOfWeekPattern("Thursday",  avg_volume=1.05, volume_vs_mean=+0.05, avg_daily_vol=0.020, avg_return=+0.0002, avg_spread_bps=7.9, n_observations=52),
            DayOfWeekPattern("Friday",    avg_volume=0.94, volume_vs_mean=-0.06, avg_daily_vol=0.021, avg_return=+0.0005, avg_spread_bps=8.2, n_observations=52),
        ]
        return patterns

    def best_trading_day(self, patterns: List[DayOfWeekPattern]) -> str:
        """Return the day with lowest spread and highest volume."""
        if not patterns:
            return "Wednesday"
        return min(patterns, key=lambda p: p.avg_spread_bps).day_name


# ---------------------------------------------------------------------------
# MOC/MOO imbalance analyzer
# ---------------------------------------------------------------------------

class MOCMOOAnalyzer:
    """
    Analyzes Market-on-Open and Market-on-Close imbalance patterns.
    Large MOO/MOC imbalances create predictable short-term price pressure.
    """

    def __init__(self):
        self.classifier = TimeSlotClassifier()

    def extract_moo_trades(self, trades: List[Trade]) -> List[Trade]:
        return [t for t in trades if t.timestamp.time() <= MOO_END]

    def extract_moc_trades(self, trades: List[Trade]) -> List[Trade]:
        return [t for t in trades if t.timestamp.time() >= MOC_START]

    def compute_imbalance(self, trades: List[Trade]) -> float:
        """Net directional imbalance: buy_vol - sell_vol / total."""
        buy_vol  = sum(t.size for t in trades if t.price >= (t.bid + t.ask) / 2)
        sell_vol = sum(t.size for t in trades if t.price < (t.bid + t.ask) / 2)
        total    = buy_vol + sell_vol
        return (buy_vol - sell_vol) / total if total > 0 else 0.0

    def estimate_impact(self, imbalance: float, avg_spread_bps: float) -> float:
        """Estimate price impact from imbalance in bps."""
        return abs(imbalance) * avg_spread_bps * 2.5   # empirical multiplier

    def synthetic_moc_moo_stats(self) -> Dict:
        """Typical MOC/MOO statistics for reference."""
        return {
            "moo_avg_volume_pct": 8.5,      # MOO = 8.5% of daily volume
            "moc_avg_volume_pct": 12.0,     # MOC = 12% of daily volume
            "moo_avg_imbalance": 0.15,      # typically 15% net buy imbalance at open
            "moc_avg_imbalance": 0.22,      # 22% at close (more institutional)
            "moo_impact_bps": 3.2,
            "moc_impact_bps": 5.1,
            "high_imbalance_threshold": 0.35,  # unusual if >35%
        }


# ---------------------------------------------------------------------------
# Liquidity scorer
# ---------------------------------------------------------------------------

class IntradayLiquidityScorer:
    """Scores time slots by their liquidity conditions."""

    def score(
        self,
        vol_share: float,    # volume share of this slot
        spread_bps: float,   # bid-ask spread
        volatility: float,   # realized vol
    ) -> float:
        """
        Higher score = more liquid = better for execution.
        Liquidity score: volume share / (spread * volatility)
        """
        spread_cost = spread_bps / 10000
        if spread_cost == 0 or volatility == 0:
            return 0.5

        score = vol_share / (spread_cost * volatility)
        return min(1.0, score / 10.0)  # normalize


# ---------------------------------------------------------------------------
# Intraday pattern builder
# ---------------------------------------------------------------------------

class IntradayPatternAnalyzer:
    """Builds complete intraday seasonality profile."""

    def __init__(self):
        self.vol_analyzer = IntradayVolumeAnalyzer()
        self.vola_analyzer = IntradayVolatilityAnalyzer()
        self.spread_analyzer = IntradaySpreadAnalyzer()
        self.dow_analyzer = DayOfWeekAnalyzer()
        self.moc_moo = MOCMOOAnalyzer()
        self.liq_scorer = IntradayLiquidityScorer()
        self.classifier = TimeSlotClassifier()

    def build(self, ticker: str = "SPY") -> IntradaySeasonality:
        vol_profile    = self.vol_analyzer.synthetic_volume_profile()
        vola_profile   = self.vola_analyzer.synthetic_vol_profile()
        spread_profile = self.spread_analyzer.synthetic_spread_profile()
        dow_patterns   = self.dow_analyzer.synthetic_dow_patterns()
        moc_moo_stats  = self.moc_moo.synthetic_moc_moo_stats()

        slots = []
        labels = self.classifier.all_slot_labels()

        for label in labels:
            slot_time = datetime.strptime(label, "%H:%M").time()
            slot_end_dt = datetime.combine(datetime.today(), slot_time) + timedelta(minutes=SLOT_MINUTES)
            slot_end = slot_end_dt.time()

            v_share = vol_profile.get(label, 1 / SLOTS_PER_DAY)
            spread  = spread_profile.get(label, 5.0)
            vola    = vola_profile.get(label, 0.20)
            liq_s   = self.liq_scorer.score(v_share, spread, vola)

            # Estimated absolute volume (assume daily vol = 5M shares)
            est_daily_vol = 5_000_000
            avg_vol = v_share * est_daily_vol

            slot = IntradaySlot(
                slot_label=label,
                slot_start=slot_time,
                slot_end=slot_end,
                avg_volume=avg_vol,
                volume_share=v_share,
                avg_volatility=vola,
                avg_spread_bps=spread,
                avg_trade_size=500.0,
                n_trades_avg=avg_vol / 500.0,
                time_slot=self.classifier.classify_time(slot_time),
                liquidity_score=liq_s,
            )
            slots.append(slot)

        peak_vol = max(slots, key=lambda s: s.volume_share)
        best_spread = min(slots, key=lambda s: s.avg_spread_bps)
        worst_spread = max(slots, key=lambda s: s.avg_spread_bps)

        # Optimal trading window: high volume, low spread
        scores = [(s, s.volume_share / s.avg_spread_bps) for s in slots]
        sorted_slots = sorted(scores, key=lambda x: x[1], reverse=True)
        best_start = sorted_slots[0][0].slot_start
        best_end   = sorted_slots[0][0].slot_end

        return IntradaySeasonality(
            ticker=ticker,
            computation_date=datetime.now(timezone.utc),
            slots=slots,
            dow_patterns=dow_patterns,
            moc_moo_summary=moc_moo_stats,
            peak_volume_slot=peak_vol,
            lowest_spread_slot=best_spread,
            worst_spread_slot=worst_spread,
            optimal_trading_window=(best_start, best_end),
        )

    def format_report(self, seasonality: IntradaySeasonality) -> str:
        lines = [
            f"=== Intraday Seasonality: {seasonality.ticker} ===",
            f"Computed: {seasonality.computation_date.strftime('%Y-%m-%d %H:%M')} UTC",
            "",
            "Intraday Volume Profile:",
            f"{'Slot':>6} {'Vol%':>7} {'Spread':>8} {'Vol(ann)':>10} {'Liq':>5} {'Type':>12}",
            "-" * 55,
        ]
        for slot in seasonality.slots:
            lines.append(
                f"{slot.slot_label:>6} {slot.volume_share:>7.1%} "
                f"{slot.avg_spread_bps:>8.1f} bps {slot.avg_volatility:>9.1%} "
                f"{slot.liquidity_score:>5.2f} {slot.time_slot.value:>12}"
            )

        lines.append("")
        lines.append(f"Peak Volume:    {seasonality.peak_volume_slot.slot_label}")
        lines.append(f"Lowest Spread:  {seasonality.lowest_spread_slot.slot_label} ({seasonality.lowest_spread_slot.avg_spread_bps:.1f} bps)")
        lines.append(f"Worst Spread:   {seasonality.worst_spread_slot.slot_label} ({seasonality.worst_spread_slot.avg_spread_bps:.1f} bps)")

        ow = seasonality.optimal_trading_window
        lines.append(f"Optimal Window: {ow[0].strftime('%H:%M')}-{ow[1].strftime('%H:%M')}")

        lines.append("")
        lines.append("Day-of-Week Pattern:")
        lines.append(f"{'Day':>12} {'Vol Rel':>8} {'Avg Return':>12} {'Spread bps':>11} {'Daily Vol':>10}")
        for dow in seasonality.dow_patterns:
            lines.append(
                f"{dow.day_name:>12} {dow.volume_vs_mean:>+8.1%} "
                f"{dow.avg_return:>+12.4%} {dow.avg_spread_bps:>11.1f} {dow.avg_daily_vol:>10.1%}"
            )

        moc_moo = seasonality.moc_moo_summary
        lines.append("")
        lines.append("MOO/MOC Statistics:")
        lines.append(f"  MOO volume share: {moc_moo['moo_avg_volume_pct']:.1f}%")
        lines.append(f"  MOC volume share: {moc_moo['moc_avg_volume_pct']:.1f}%")
        lines.append(f"  MOC avg impact:   {moc_moo['moc_impact_bps']:.1f} bps")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Intraday patterns CLI")
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--action", choices=["report", "slots", "dow", "moc_moo"], default="report")
    args = parser.parse_args()

    analyzer = IntradayPatternAnalyzer()

    if args.action == "report":
        seasonality = analyzer.build(args.ticker)
        print(analyzer.format_report(seasonality))
    elif args.action == "slots":
        seasonality = analyzer.build(args.ticker)
        for slot in seasonality.slots:
            bar = "█" * int(slot.volume_share * 100)
            print(f"  {slot.slot_label}: {bar:<15} {slot.volume_share:.1%} vol, {slot.avg_spread_bps:.1f} bps")
    elif args.action == "dow":
        patterns = analyzer.dow_analyzer.synthetic_dow_patterns()
        for d in patterns:
            print(f"{d.day_name:12s}: vol={d.volume_vs_mean:+.1%}, ret={d.avg_return:+.4%}, spread={d.avg_spread_bps:.1f} bps")
    elif args.action == "moc_moo":
        stats = analyzer.moc_moo.synthetic_moc_moo_stats()
        for k, v in stats.items():
            print(f"  {k}: {v}")
