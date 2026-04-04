"""
network_metrics.py — Blockchain network health analytics.

Covers:
  - Active addresses (daily, 7d MA, 30d MA)
  - Transaction count and volume
  - NVT ratio (Network Value to Transactions) — simple and signal variants
  - MVRV (Market Value to Realized Value) ratio
  - SOPR (Spent Output Profit Ratio) — entity-adjusted
  - Composite network health score
  - Cycle-phase detector (NVT + MVRV combined)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from data_fetchers import GlassnodeClient, GlassnodeMetric, OnChainDataProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class NetworkSnapshot:
    timestamp: int
    active_addresses: float
    tx_count: float
    tx_volume_usd: float
    market_cap_usd: float
    realized_cap_usd: float
    nvt: float
    nvt_signal: float
    mvrv: float
    sopr: float
    exchange_inflow: float
    exchange_outflow: float

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.timestamp, tz=timezone.utc)

    @property
    def net_exchange_flow(self) -> float:
        return self.exchange_outflow - self.exchange_inflow


@dataclass
class NVTAnalysis:
    current_nvt: float
    nvt_signal: float          # 90-day MA of volume denominator
    historical_mean: float
    historical_std: float
    z_score: float
    percentile: float          # 0-100
    signal: str                # "overvalued", "fair", "undervalued"
    threshold_high: float      # NVT above this = overvalued
    threshold_low: float       # NVT below this = undervalued
    series: List[Tuple[datetime, float, float]] = field(default_factory=list)  # (dt, nvt, nvt_signal)


@dataclass
class MVRVAnalysis:
    current_mvrv: float
    historical_mean: float
    historical_std: float
    z_score: float
    percentile: float
    signal: str                # "top_zone", "overvalued", "fair", "undervalued", "bottom_zone"
    unrealized_profit_pct: float   # (market_cap - realized_cap) / realized_cap
    series: List[Tuple[datetime, float]] = field(default_factory=list)


@dataclass
class SOPRAnalysis:
    current_sopr: float
    smoothed_sopr: float       # 7-day MA
    historical_mean: float
    z_score: float
    signal: str                # "euphoria", "profit_taking", "neutral", "capitulation"
    above_one_streak: int      # consecutive days SOPR > 1
    below_one_streak: int
    series: List[Tuple[datetime, float, float]] = field(default_factory=list)  # (dt, sopr, smoothed)


@dataclass
class NetworkHealthScore:
    timestamp: datetime
    overall_score: float       # 0-100
    activity_score: float      # based on active addresses, tx count
    valuation_score: float     # based on NVT, MVRV (inverted — low NVT = healthy)
    sentiment_score: float     # based on SOPR, exchange flows
    trend_score: float         # based on momentum of metrics
    cycle_phase: str           # "accumulation", "early_bull", "mid_bull", "late_bull", "distribution", "bear"
    breakdown: Dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# NVT ratio calculator
# ---------------------------------------------------------------------------

class NVTCalculator:
    """
    Network Value to Transactions ratio.
    NVT = Market Cap / Daily TX Volume (USD)
    NVT Signal = Market Cap / 90-day MA of TX Volume
    """

    NVT_HIGH_THRESHOLD  = 65    # historically overvalued
    NVT_LOW_THRESHOLD   = 27    # historically undervalued
    NVT_SIGNAL_WINDOW   = 90    # days for MA

    def __init__(self, glassnode: GlassnodeClient):
        self.glassnode = glassnode

    def compute(self, asset: str = "BTC", days: int = 365) -> NVTAnalysis:
        end = int(time.time())
        start = end - days * 86400

        mcap_series = self.glassnode.market_cap(asset, start=start, end=end)
        vol_series  = self.glassnode.transaction_volume(asset, start=start, end=end)

        # Align by timestamp
        mcap_map = {m.timestamp: m.value for m in mcap_series}
        vol_map  = {m.timestamp: m.value for m in vol_series}
        common_ts = sorted(set(mcap_map) & set(vol_map))

        if not common_ts:
            return NVTAnalysis(
                current_nvt=0, nvt_signal=0, historical_mean=0,
                historical_std=0, z_score=0, percentile=50,
                signal="unknown", threshold_high=self.NVT_HIGH_THRESHOLD,
                threshold_low=self.NVT_LOW_THRESHOLD,
            )

        nvts = []
        for ts in common_ts:
            vol = vol_map[ts]
            mc = mcap_map[ts]
            nvt = mc / vol if vol > 0 else float("nan")
            nvts.append((ts, nvt))

        nvt_vals = [v for _, v in nvts if not math.isnan(v)]
        nvt_arr = np.array(nvt_vals)

        # NVT Signal: use rolling 90-day MA of volume in denominator
        vol_vals = [vol_map[ts] for ts in common_ts]
        vol_ma = self._rolling_mean(vol_vals, self.NVT_SIGNAL_WINDOW)
        nvt_signal_vals = []
        for i, ts in enumerate(common_ts):
            mc = mcap_map[ts]
            vm = vol_ma[i]
            nvt_signal_vals.append(mc / vm if vm > 0 else float("nan"))

        current_nvt = nvt_vals[-1] if nvt_vals else 0.0
        current_nvt_signal = next((v for v in reversed(nvt_signal_vals) if not math.isnan(v)), 0.0)

        mean_nvt = float(np.nanmean(nvt_arr))
        std_nvt = float(np.nanstd(nvt_arr))
        z_score = (current_nvt - mean_nvt) / std_nvt if std_nvt > 0 else 0.0

        below = sum(1 for v in nvt_vals if v < current_nvt)
        percentile = 100.0 * below / len(nvt_vals)

        if current_nvt_signal > self.NVT_HIGH_THRESHOLD:
            signal = "overvalued"
        elif current_nvt_signal < self.NVT_LOW_THRESHOLD:
            signal = "undervalued"
        else:
            signal = "fair"

        series = []
        for (ts, nvt), nvt_s in zip(nvts, nvt_signal_vals):
            if not math.isnan(nvt) and not math.isnan(nvt_s):
                series.append((datetime.fromtimestamp(ts, tz=timezone.utc), nvt, nvt_s))

        return NVTAnalysis(
            current_nvt=current_nvt,
            nvt_signal=current_nvt_signal,
            historical_mean=mean_nvt,
            historical_std=std_nvt,
            z_score=z_score,
            percentile=percentile,
            signal=signal,
            threshold_high=self.NVT_HIGH_THRESHOLD,
            threshold_low=self.NVT_LOW_THRESHOLD,
            series=series,
        )

    @staticmethod
    def _rolling_mean(values: List[float], window: int) -> List[float]:
        result = []
        for i in range(len(values)):
            lo = max(0, i - window + 1)
            segment = values[lo : i + 1]
            valid = [v for v in segment if not math.isnan(v)]
            result.append(float(np.mean(valid)) if valid else float("nan"))
        return result


# ---------------------------------------------------------------------------
# MVRV calculator
# ---------------------------------------------------------------------------

class MVRVCalculator:
    """
    Market Value to Realized Value.
    MVRV = Market Cap / Realized Cap
    MVRV > 3.7 historically marks cycle tops.
    MVRV < 1.0 historically marks cycle bottoms.
    """

    MVRV_TOP_ZONE    = 3.7
    MVRV_HIGH        = 2.4
    MVRV_LOW         = 1.0
    MVRV_BOTTOM_ZONE = 0.7

    def __init__(self, glassnode: GlassnodeClient):
        self.glassnode = glassnode

    def compute(self, asset: str = "BTC", days: int = 1460) -> MVRVAnalysis:
        end = int(time.time())
        start = end - days * 86400

        mcap_series = self.glassnode.market_cap(asset, start=start, end=end)
        rcap_series = self.glassnode.realized_cap(asset, start=start, end=end)

        mcap_map = {m.timestamp: m.value for m in mcap_series}
        rcap_map = {m.timestamp: m.value for m in rcap_series}
        common_ts = sorted(set(mcap_map) & set(rcap_map))

        if not common_ts:
            return MVRVAnalysis(
                current_mvrv=1.0, historical_mean=1.5, historical_std=0.8,
                z_score=0.0, percentile=50.0, signal="fair",
                unrealized_profit_pct=0.0,
            )

        mvrv_vals = []
        series = []
        for ts in common_ts:
            mc = mcap_map[ts]
            rc = rcap_map[ts]
            mvrv = mc / rc if rc > 0 else float("nan")
            mvrv_vals.append(mvrv)
            if not math.isnan(mvrv):
                series.append((datetime.fromtimestamp(ts, tz=timezone.utc), mvrv))

        valid_vals = [v for v in mvrv_vals if not math.isnan(v)]
        current_mvrv = valid_vals[-1] if valid_vals else 1.0
        mean_mvrv = float(np.mean(valid_vals))
        std_mvrv = float(np.std(valid_vals))
        z_score = (current_mvrv - mean_mvrv) / std_mvrv if std_mvrv > 0 else 0.0

        below = sum(1 for v in valid_vals if v < current_mvrv)
        percentile = 100.0 * below / len(valid_vals)

        if current_mvrv >= self.MVRV_TOP_ZONE:
            signal = "top_zone"
        elif current_mvrv >= self.MVRV_HIGH:
            signal = "overvalued"
        elif current_mvrv <= self.MVRV_BOTTOM_ZONE:
            signal = "bottom_zone"
        elif current_mvrv <= self.MVRV_LOW:
            signal = "undervalued"
        else:
            signal = "fair"

        # Unrealized profit = (MC - RC) / RC
        last_ts = common_ts[-1]
        mc_last = mcap_map[last_ts]
        rc_last = rcap_map[last_ts]
        unrealized_profit = (mc_last - rc_last) / rc_last if rc_last > 0 else 0.0

        return MVRVAnalysis(
            current_mvrv=current_mvrv,
            historical_mean=mean_mvrv,
            historical_std=std_mvrv,
            z_score=z_score,
            percentile=percentile,
            signal=signal,
            unrealized_profit_pct=unrealized_profit,
            series=series,
        )


# ---------------------------------------------------------------------------
# SOPR calculator
# ---------------------------------------------------------------------------

class SOPRCalculator:
    """
    Spent Output Profit Ratio.
    SOPR > 1: coins moved at a profit.
    SOPR < 1: coins moved at a loss (capitulation signal).
    Entity-adjusted SOPR (aSOPR) excludes short-term holders.
    """

    def __init__(self, glassnode: GlassnodeClient):
        self.glassnode = glassnode

    def compute(self, asset: str = "BTC", days: int = 365, smooth_window: int = 7) -> SOPRAnalysis:
        end = int(time.time())
        start = end - days * 86400

        raw = self.glassnode.sopr(asset, start=start, end=end)
        if not raw:
            return SOPRAnalysis(
                current_sopr=1.0, smoothed_sopr=1.0, historical_mean=1.0,
                z_score=0.0, signal="neutral", above_one_streak=0, below_one_streak=0,
            )

        vals = [m.value for m in raw]
        smooth = self._ema(vals, smooth_window)

        current_sopr = vals[-1]
        current_smooth = smooth[-1]

        arr = np.array([v for v in vals if not math.isnan(v)])
        mean_sopr = float(np.mean(arr))
        std_sopr  = float(np.std(arr))
        z_score   = (current_sopr - mean_sopr) / std_sopr if std_sopr > 0 else 0.0

        if current_smooth > 1.02:
            signal = "euphoria" if current_smooth > 1.05 else "profit_taking"
        elif current_smooth < 0.97:
            signal = "capitulation"
        else:
            signal = "neutral"

        # Count consecutive streak above/below 1
        above_streak = 0
        for v in reversed(vals):
            if v > 1.0:
                above_streak += 1
            else:
                break

        below_streak = 0
        for v in reversed(vals):
            if v < 1.0:
                below_streak += 1
            else:
                break

        series = [
            (m.dt, m.value, smooth[i])
            for i, m in enumerate(raw)
        ]

        return SOPRAnalysis(
            current_sopr=current_sopr,
            smoothed_sopr=current_smooth,
            historical_mean=mean_sopr,
            z_score=z_score,
            signal=signal,
            above_one_streak=above_streak,
            below_one_streak=below_streak,
            series=series,
        )

    @staticmethod
    def _ema(values: List[float], window: int) -> List[float]:
        alpha = 2.0 / (window + 1)
        result = []
        ema_val = values[0] if values else 1.0
        for v in values:
            if math.isnan(v):
                result.append(ema_val)
            else:
                ema_val = alpha * v + (1 - alpha) * ema_val
                result.append(ema_val)
        return result


# ---------------------------------------------------------------------------
# Active address analyzer
# ---------------------------------------------------------------------------

class ActiveAddressAnalyzer:
    """Analyzes active address trends as a proxy for network adoption."""

    def __init__(self, glassnode: GlassnodeClient):
        self.glassnode = glassnode

    def get_series(self, asset: str = "BTC", days: int = 365) -> List[Tuple[datetime, float]]:
        end = int(time.time())
        start = end - days * 86400
        raw = self.glassnode.active_addresses(asset, start=start, end=end)
        return [(m.dt, m.value) for m in raw]

    def momentum(self, asset: str = "BTC", short: int = 7, long: int = 30) -> float:
        """Returns z-score of short-term vs long-term active address count."""
        series = self.get_series(asset, long + 7)
        if len(series) < long:
            return 0.0
        vals = [v for _, v in series]
        short_mean = float(np.mean(vals[-short:]))
        long_mean = float(np.mean(vals[-long:]))
        long_std = float(np.std(vals[-long:]))
        return (short_mean - long_mean) / long_std if long_std > 0 else 0.0

    def growth_rate(self, asset: str = "BTC", days: int = 90) -> float:
        series = self.get_series(asset, days)
        if len(series) < 2:
            return 0.0
        vals = [v for _, v in series]
        return (vals[-1] - vals[0]) / vals[0] if vals[0] > 0 else 0.0

    def summary(self, asset: str = "BTC") -> Dict:
        series = self.get_series(asset, 90)
        if not series:
            return {}
        vals = [v for _, v in series]
        return {
            "current": vals[-1],
            "mean_30d": float(np.mean(vals[-30:])),
            "mean_90d": float(np.mean(vals)),
            "max_90d": float(np.max(vals)),
            "min_90d": float(np.min(vals)),
            "momentum_7d": self.momentum(asset, 7, 30),
            "growth_90d": self.growth_rate(asset, 90),
        }


# ---------------------------------------------------------------------------
# Transaction volume analyzer
# ---------------------------------------------------------------------------

class TransactionVolumeAnalyzer:
    """Analyzes on-chain transaction count and volume."""

    def __init__(self, glassnode: GlassnodeClient):
        self.glassnode = glassnode

    def get_count_series(self, asset: str = "BTC", days: int = 90) -> List[Tuple[datetime, float]]:
        end = int(time.time())
        start = end - days * 86400
        raw = self.glassnode.transaction_count(asset, start=start, end=end)
        return [(m.dt, m.value) for m in raw]

    def get_volume_series(self, asset: str = "BTC", days: int = 90) -> List[Tuple[datetime, float]]:
        end = int(time.time())
        start = end - days * 86400
        raw = self.glassnode.transaction_volume(asset, start=start, end=end)
        return [(m.dt, m.value) for m in raw]

    def volume_zscore(self, asset: str = "BTC", days: int = 60) -> float:
        series = self.get_volume_series(asset, days)
        if not series:
            return 0.0
        vals = [v for _, v in series]
        current = vals[-1]
        mean = float(np.mean(vals[:-1]))
        std = float(np.std(vals[:-1]))
        return (current - mean) / std if std > 0 else 0.0

    def summary(self, asset: str = "BTC") -> Dict:
        count_series = self.get_count_series(asset, 30)
        vol_series = self.get_volume_series(asset, 30)

        counts = [v for _, v in count_series]
        vols = [v for _, v in vol_series]

        return {
            "current_tx_count": counts[-1] if counts else 0,
            "mean_tx_count_30d": float(np.mean(counts)) if counts else 0,
            "current_volume": vols[-1] if vols else 0,
            "mean_volume_30d": float(np.mean(vols)) if vols else 0,
            "volume_zscore": self.volume_zscore(asset, 60),
        }


# ---------------------------------------------------------------------------
# Cycle phase detector
# ---------------------------------------------------------------------------

class CyclePhaseDetector:
    """
    Classifies the current market cycle phase based on NVT + MVRV + SOPR.
    Phases: accumulation, early_bull, mid_bull, late_bull, distribution, bear
    """

    def detect(
        self,
        nvt: NVTAnalysis,
        mvrv: MVRVAnalysis,
        sopr: SOPRAnalysis,
    ) -> str:
        """Returns cycle phase string."""
        mvrv_v = mvrv.current_mvrv
        sopr_v = sopr.smoothed_sopr
        nvt_sig = nvt.nvt_signal

        # Late bull / distribution: high MVRV + high NVT + SOPR euphoria
        if mvrv_v >= 3.0 and nvt_sig >= 55 and sopr_v >= 1.03:
            return "late_bull"

        # Distribution: very high MVRV
        if mvrv_v >= 3.7:
            return "distribution"

        # Mid bull: MVRV 2-3.7, NVT fair, SOPR profit taking
        if 2.0 <= mvrv_v < 3.7 and sopr_v > 1.0:
            return "mid_bull"

        # Early bull: MVRV 1.0-2.0, NVT fair/undervalued, SOPR recovering
        if 1.0 <= mvrv_v < 2.0 and sopr_v >= 1.0:
            return "early_bull"

        # Bear: MVRV < 1.0 and SOPR < 1 (selling at loss)
        if mvrv_v < 1.0 and sopr_v < 0.99:
            return "bear"

        # Accumulation: MVRV < 1.2, SOPR around 1, low NVT
        if mvrv_v < 1.2 and nvt_sig <= 35:
            return "accumulation"

        return "transition"

    def phase_to_bias(self, phase: str) -> str:
        mapping = {
            "accumulation": "strong_buy",
            "early_bull": "buy",
            "mid_bull": "hold",
            "late_bull": "reduce",
            "distribution": "sell",
            "bear": "strong_avoid",
            "transition": "neutral",
        }
        return mapping.get(phase, "neutral")


# ---------------------------------------------------------------------------
# Network health scorer
# ---------------------------------------------------------------------------

class NetworkHealthScorer:
    """
    Combines all metrics into a composite 0-100 health score.
    """

    def __init__(self, provider: OnChainDataProvider):
        self.provider = provider
        self.nvt_calc  = NVTCalculator(provider.glassnode)
        self.mvrv_calc = MVRVCalculator(provider.glassnode)
        self.sopr_calc = SOPRCalculator(provider.glassnode)
        self.aa_analyzer = ActiveAddressAnalyzer(provider.glassnode)
        self.vol_analyzer = TransactionVolumeAnalyzer(provider.glassnode)
        self.cycle_detector = CyclePhaseDetector()

    def compute(self, asset: str = "BTC") -> NetworkHealthScore:
        nvt  = self.nvt_calc.compute(asset, 365)
        mvrv = self.mvrv_calc.compute(asset, 365)
        sopr = self.sopr_calc.compute(asset, 90)
        aa   = self.aa_analyzer.summary(asset)
        vol  = self.vol_analyzer.summary(asset)

        # Activity score (0-100): higher active addresses and volume = healthier
        aa_z = aa.get("momentum_7d", 0.0)
        vol_z = vol.get("volume_zscore", 0.0)
        activity_score = 50.0 + 10 * aa_z + 5 * vol_z
        activity_score = max(0.0, min(100.0, activity_score))

        # Valuation score (0-100): inverted NVT and MVRV z-scores
        # Lower NVT z-score = undervalued = healthier
        nvt_component  = 50.0 - 10 * nvt.z_score
        mvrv_component = 50.0 - 10 * (mvrv.z_score - 0.5)   # slight bullish bias
        valuation_score = max(0.0, min(100.0, (nvt_component + mvrv_component) / 2))

        # Sentiment score (0-100): SOPR around 1 = healthy, high = euphoria (sell), low = fear (buy)
        sopr_deviation = sopr.current_sopr - 1.0
        if sopr_deviation > 0:
            sentiment_score = 50.0 + min(50.0, sopr_deviation * 500)   # buying pressure
        else:
            sentiment_score = 50.0 + max(-50.0, sopr_deviation * 500)  # selling pressure
        sentiment_score = max(0.0, min(100.0, sentiment_score))

        # Trend score: combination of 7d momentum signals
        trend_z = aa_z * 0.4 + vol_z * 0.3 + sopr.z_score * 0.3
        trend_score = max(0.0, min(100.0, 50.0 + 10 * trend_z))

        overall = (
            0.25 * activity_score +
            0.30 * valuation_score +
            0.25 * sentiment_score +
            0.20 * trend_score
        )

        phase = self.cycle_detector.detect(nvt, mvrv, sopr)

        return NetworkHealthScore(
            timestamp=datetime.now(timezone.utc),
            overall_score=round(overall, 1),
            activity_score=round(activity_score, 1),
            valuation_score=round(valuation_score, 1),
            sentiment_score=round(sentiment_score, 1),
            trend_score=round(trend_score, 1),
            cycle_phase=phase,
            breakdown={
                "nvt_signal": nvt.nvt_signal,
                "nvt_z_score": nvt.z_score,
                "nvt_interpretation": nvt.signal,
                "mvrv": mvrv.current_mvrv,
                "mvrv_z_score": mvrv.z_score,
                "mvrv_interpretation": mvrv.signal,
                "sopr": sopr.current_sopr,
                "sopr_smoothed": sopr.smoothed_sopr,
                "sopr_interpretation": sopr.signal,
                "active_address_momentum": aa.get("momentum_7d", 0),
                "volume_z_score": vol.get("volume_zscore", 0),
                "cycle_bias": self.cycle_detector.phase_to_bias(phase),
            },
        )


# ---------------------------------------------------------------------------
# Full report builder
# ---------------------------------------------------------------------------

class NetworkMetricsReport:
    """Generates comprehensive network metrics reports."""

    def __init__(self, provider: OnChainDataProvider = None):
        self.provider = provider or OnChainDataProvider()
        self.scorer = NetworkHealthScorer(self.provider)
        self.nvt_calc  = NVTCalculator(self.provider.glassnode)
        self.mvrv_calc = MVRVCalculator(self.provider.glassnode)
        self.sopr_calc = SOPRCalculator(self.provider.glassnode)

    def full_report(self, asset: str = "BTC") -> Dict:
        score = self.scorer.compute(asset)
        nvt   = self.nvt_calc.compute(asset, 365)
        mvrv  = self.mvrv_calc.compute(asset, 730)
        sopr  = self.sopr_calc.compute(asset, 90)

        return {
            "asset": asset,
            "timestamp": score.timestamp.isoformat(),
            "health_score": score.overall_score,
            "cycle_phase": score.cycle_phase,
            "cycle_bias": score.breakdown["cycle_bias"],
            "sub_scores": {
                "activity": score.activity_score,
                "valuation": score.valuation_score,
                "sentiment": score.sentiment_score,
                "trend": score.trend_score,
            },
            "nvt": {
                "current": nvt.current_nvt,
                "signal": nvt.nvt_signal,
                "z_score": nvt.z_score,
                "interpretation": nvt.signal,
                "percentile": nvt.percentile,
            },
            "mvrv": {
                "current": mvrv.current_mvrv,
                "z_score": mvrv.z_score,
                "interpretation": mvrv.signal,
                "unrealized_profit_pct": mvrv.unrealized_profit_pct,
                "percentile": mvrv.percentile,
            },
            "sopr": {
                "current": sopr.current_sopr,
                "smoothed_7d": sopr.smoothed_sopr,
                "interpretation": sopr.signal,
                "above_1_streak": sopr.above_one_streak,
                "below_1_streak": sopr.below_one_streak,
            },
        }

    def format_report(self, asset: str = "BTC") -> str:
        r = self.full_report(asset)
        lines = [
            f"=== Network Metrics Report: {r['asset']} ===",
            f"Timestamp: {r['timestamp']}",
            f"Overall Health Score: {r['health_score']:.1f}/100",
            f"Cycle Phase: {r['cycle_phase'].upper()} → Bias: {r['cycle_bias'].upper()}",
            "",
            "Sub-scores:",
            f"  Activity:   {r['sub_scores']['activity']:.1f}",
            f"  Valuation:  {r['sub_scores']['valuation']:.1f}",
            f"  Sentiment:  {r['sub_scores']['sentiment']:.1f}",
            f"  Trend:      {r['sub_scores']['trend']:.1f}",
            "",
            "NVT Ratio:",
            f"  Current NVT:        {r['nvt']['current']:.1f}",
            f"  NVT Signal (90d MA):{r['nvt']['signal']:.1f}",
            f"  Z-Score:            {r['nvt']['z_score']:+.2f}",
            f"  Interpretation:     {r['nvt']['interpretation'].upper()}",
            f"  Historical %ile:    {r['nvt']['percentile']:.0f}th",
            "",
            "MVRV Ratio:",
            f"  Current MVRV:       {r['mvrv']['current']:.3f}",
            f"  Z-Score:            {r['mvrv']['z_score']:+.2f}",
            f"  Interpretation:     {r['mvrv']['interpretation'].upper()}",
            f"  Unrealized P/L:     {r['mvrv']['unrealized_profit_pct']*100:+.1f}%",
            f"  Historical %ile:    {r['mvrv']['percentile']:.0f}th",
            "",
            "SOPR:",
            f"  Current SOPR:       {r['sopr']['current']:.4f}",
            f"  7d Smoothed:        {r['sopr']['smoothed_7d']:.4f}",
            f"  Interpretation:     {r['sopr']['interpretation'].upper()}",
            f"  Streak above 1:     {r['sopr']['above_1_streak']} days",
            f"  Streak below 1:     {r['sopr']['below_1_streak']} days",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Network metrics CLI")
    parser.add_argument("--asset", default="BTC")
    parser.add_argument("--format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    import json as _json

    reporter = NetworkMetricsReport()
    if args.format == "text":
        print(reporter.format_report(args.asset))
    else:
        report = reporter.full_report(args.asset)
        print(_json.dumps(report, indent=2))
