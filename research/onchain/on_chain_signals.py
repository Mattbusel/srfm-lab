"""
on_chain_signals.py — Composite on-chain alpha signal builder.

Combines:
  - Whale exchange flows (inflow/outflow)
  - Funding rate regime
  - NVT ratio (valuation signal)
  - MVRV ratio (cycle position)
  - SOPR (profit-taking / capitulation)
  - Active address momentum
  - Liquidation bias (liq ratio)

Each component is z-scored and decay-weighted to produce:
  - Individual component signals
  - Composite alpha score (-1 to +1)
  - Signal breakdown and confidence estimate
  - Historical signal backtest summary
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from data_fetchers import GlassnodeClient, GlassnodeMetric, OnChainDataProvider
from funding_rates import (
    CrossExchangeFundingAggregator,
    FundingRegimeClassifier,
    PERIODS_PER_YEAR,
)
from network_metrics import (
    MVRVCalculator,
    NVTCalculator,
    SOPRCalculator,
    ActiveAddressAnalyzer,
    CyclePhaseDetector,
)
from whale_tracker import ExchangeFlowAnalyzer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal component model
# ---------------------------------------------------------------------------

@dataclass
class SignalComponent:
    name: str
    raw_value: float          # the raw metric value
    z_score: float            # standardized score
    normalized_signal: float  # mapped to [-1, +1]
    weight: float             # contribution weight
    weighted_contribution: float
    interpretation: str       # human-readable
    is_bullish: bool
    confidence: float         # 0-1


@dataclass
class CompositeSignal:
    timestamp: datetime
    asset: str
    composite_score: float    # weighted average of normalized signals, [-1, +1]
    composite_z: float        # z-score of composite
    direction: str            # "strong_bull", "bull", "neutral", "bear", "strong_bear"
    confidence: float         # 0-1
    components: List[SignalComponent]
    cycle_phase: str
    cycle_bias: str
    signal_agreement: float   # fraction of components pointing same direction
    breakdown: Dict[str, float]

    @property
    def is_actionable(self) -> bool:
        return abs(self.composite_score) > 0.3 and self.confidence > 0.5

    @property
    def signal_summary(self) -> str:
        direction_emoji = {
            "strong_bull": "🚀 STRONG BULL",
            "bull":        "📈 BULL",
            "neutral":     "➡️  NEUTRAL",
            "bear":        "📉 BEAR",
            "strong_bear": "💀 STRONG BEAR",
        }
        label = direction_emoji.get(self.direction, self.direction.upper())
        return f"{label} | Score: {self.composite_score:+.3f} | Confidence: {self.confidence:.0%}"


# ---------------------------------------------------------------------------
# Exponential decay weighting
# ---------------------------------------------------------------------------

class DecayWeighter:
    """
    Applies time-decay to historical signal values.
    More recent observations get higher weight.
    Decay: w_t = exp(-lambda * (T - t))
    """

    def __init__(self, half_life_days: float = 14.0):
        self.half_life_days = half_life_days
        self.decay_rate = math.log(2) / half_life_days

    def weights(self, n: int) -> np.ndarray:
        """Returns array of decay weights, oldest first."""
        ages = np.arange(n - 1, -1, -1, dtype=float)   # n-1, n-2, ..., 0
        w = np.exp(-self.decay_rate * ages)
        return w / w.sum()

    def weighted_mean(self, values: List[float]) -> float:
        if not values:
            return 0.0
        n = len(values)
        w = self.weights(n)
        return float(np.dot(w, values))

    def weighted_zscore(self, values: List[float]) -> float:
        """Z-score of latest value relative to decay-weighted distribution."""
        if len(values) < 5:
            return 0.0
        arr = np.array(values)
        w = self.weights(len(arr))
        mean = float(np.dot(w, arr))
        var  = float(np.dot(w, (arr - mean) ** 2))
        std  = math.sqrt(var)
        return (arr[-1] - mean) / std if std > 0 else 0.0


# ---------------------------------------------------------------------------
# Individual signal extractors
# ---------------------------------------------------------------------------

def _z_to_signal(z: float, cap: float = 2.5) -> float:
    """Convert z-score to [-1, +1] via tanh normalization."""
    return math.tanh(z / cap)


class WhaleFlowSignalExtractor:
    """Extracts signal from exchange inflow/outflow data."""

    def __init__(self, glassnode: GlassnodeClient, decay: DecayWeighter):
        self.gl = glassnode
        self.decay = decay

    def extract(self, asset: str = "BTC", days: int = 60) -> SignalComponent:
        end = int(time.time())
        start = end - days * 86400

        inflow  = self.gl.exchange_inflow(asset, start=start, end=end)
        outflow = self.gl.exchange_outflow(asset, start=start, end=end)

        ts_map: Dict[int, Dict] = {}
        for m in inflow:
            ts_map.setdefault(m.timestamp, {})["in"] = m.value
        for m in outflow:
            ts_map.setdefault(m.timestamp, {})["out"] = m.value

        net_flows = [
            d.get("out", 0) - d.get("in", 0)
            for d in sorted(ts_map.values(), key=lambda x: x.get("in", 0))
        ]

        if not net_flows:
            return SignalComponent("whale_flow", 0, 0, 0, 0.20, 0, "no data", False, 0)

        z = self.decay.weighted_zscore(net_flows)
        sig = _z_to_signal(z)  # positive net outflow → positive signal (bullish)

        return SignalComponent(
            name="whale_exchange_flow",
            raw_value=net_flows[-1],
            z_score=z,
            normalized_signal=sig,
            weight=0.20,
            weighted_contribution=sig * 0.20,
            interpretation="coins leaving exchanges" if sig > 0 else "coins entering exchanges",
            is_bullish=sig > 0,
            confidence=min(1.0, abs(z) / 2.0),
        )


class FundingRateSignalExtractor:
    """Extracts signal from perpetual funding rates."""

    def __init__(self, agg: CrossExchangeFundingAggregator, decay: DecayWeighter):
        self.agg = agg
        self.decay = decay
        self.classifier = FundingRegimeClassifier()

    def extract(self, asset: str = "BTC") -> SignalComponent:
        try:
            snapshots = self.agg.get_all_current(asset)
            if not snapshots:
                return SignalComponent("funding_rate", 0, 0, 0, 0.20, 0, "no data", False, 0)

            # Average funding rate across exchanges
            avg_rate = float(np.mean([s.funding_rate for s in snapshots.values()]))
        except Exception:
            avg_rate = 0.0003  # fallback: slightly positive

        # Funding rate signal is contrarian:
        # High positive funding → longs are paying → crowded → bearish
        # Negative funding → shorts are paying → oversold → bullish
        z = avg_rate / 0.001  # normalize: 0.1% per 8h = 1 std

        # Invert: high funding = negative signal (crowded longs)
        sig = _z_to_signal(-z)

        regime = self.classifier.classify(avg_rate)
        interp = self.classifier.regime_signal(regime)

        return SignalComponent(
            name="funding_rate",
            raw_value=avg_rate,
            z_score=z,
            normalized_signal=sig,
            weight=0.20,
            weighted_contribution=sig * 0.20,
            interpretation=interp,
            is_bullish=sig > 0,
            confidence=min(1.0, abs(z) / 2.0),
        )


class NVTSignalExtractor:
    """Extracts signal from NVT ratio (undervalued = bullish)."""

    def __init__(self, glassnode: GlassnodeClient, decay: DecayWeighter):
        self.calc = NVTCalculator(glassnode)
        self.decay = decay

    def extract(self, asset: str = "BTC") -> SignalComponent:
        nvt = self.calc.compute(asset, 365)

        # Low NVT = undervalued = bullish (invert z-score)
        z = -nvt.z_score
        sig = _z_to_signal(z)

        mapping = {"overvalued": "bearish NVT", "fair": "neutral NVT", "undervalued": "bullish NVT"}
        interp = mapping.get(nvt.signal, nvt.signal)

        return SignalComponent(
            name="nvt_ratio",
            raw_value=nvt.nvt_signal,
            z_score=nvt.z_score,
            normalized_signal=sig,
            weight=0.15,
            weighted_contribution=sig * 0.15,
            interpretation=interp,
            is_bullish=sig > 0,
            confidence=min(1.0, abs(nvt.z_score) / 2.0),
        )


class MVRVSignalExtractor:
    """Extracts signal from MVRV ratio."""

    def __init__(self, glassnode: GlassnodeClient, decay: DecayWeighter):
        self.calc = MVRVCalculator(glassnode)
        self.decay = decay

    def extract(self, asset: str = "BTC") -> SignalComponent:
        mvrv = self.calc.compute(asset, 730)

        # MVRV interpretation:
        # Low MVRV (<1.0) = below realized value = bullish
        # High MVRV (>3.7) = well above cost basis = bearish (top zone)
        z = -mvrv.z_score
        sig = _z_to_signal(z * 0.7)   # slightly damped

        mapping = {
            "bottom_zone":  "deep value: strong bull signal",
            "undervalued":  "MVRV below historical mean: bullish",
            "fair":         "MVRV at fair value",
            "overvalued":   "MVRV elevated: cautious",
            "top_zone":     "MVRV extreme: distribution risk",
        }
        interp = mapping.get(mvrv.signal, mvrv.signal)

        return SignalComponent(
            name="mvrv_ratio",
            raw_value=mvrv.current_mvrv,
            z_score=mvrv.z_score,
            normalized_signal=sig,
            weight=0.20,
            weighted_contribution=sig * 0.20,
            interpretation=interp,
            is_bullish=sig > 0,
            confidence=min(1.0, abs(mvrv.z_score) / 2.0),
        )


class SOPRSignalExtractor:
    """Extracts signal from SOPR (capitulation/euphoria)."""

    def __init__(self, glassnode: GlassnodeClient, decay: DecayWeighter):
        self.calc = SOPRCalculator(glassnode)
        self.decay = decay

    def extract(self, asset: str = "BTC") -> SignalComponent:
        sopr = self.calc.compute(asset, 90)

        # SOPR signal:
        # SOPR < 1 (selling at loss) → capitulation → buy
        # SOPR >> 1 (euphoria) → distribution → sell
        deviation = sopr.smoothed_sopr - 1.0
        z = deviation / 0.02   # normalize: 2% above 1 = 1 std

        sig = _z_to_signal(-z)  # invert: selling at loss = bullish signal

        mapping = {
            "capitulation":   "SOPR below 1: capitulation → bullish",
            "neutral":        "SOPR near 1: balanced",
            "profit_taking":  "SOPR above 1: taking profits → mild caution",
            "euphoria":       "SOPR elevated: market euphoria → bearish",
        }
        interp = mapping.get(sopr.signal, sopr.signal)

        return SignalComponent(
            name="sopr",
            raw_value=sopr.current_sopr,
            z_score=sopr.z_score,
            normalized_signal=sig,
            weight=0.15,
            weighted_contribution=sig * 0.15,
            interpretation=interp,
            is_bullish=sig > 0,
            confidence=min(1.0, abs(sopr.z_score) / 2.0),
        )


class ActiveAddressSignalExtractor:
    """Extracts signal from active address momentum."""

    def __init__(self, glassnode: GlassnodeClient, decay: DecayWeighter):
        self.analyzer = ActiveAddressAnalyzer(glassnode)
        self.decay = decay

    def extract(self, asset: str = "BTC") -> SignalComponent:
        summary = self.analyzer.summary(asset)
        momentum = summary.get("momentum_7d", 0.0)
        growth = summary.get("growth_90d", 0.0)

        # Combined: short-term momentum + medium-term growth
        combined_z = momentum * 0.6 + growth * 3 * 0.4   # scale growth similarly
        sig = _z_to_signal(combined_z)

        if momentum > 0.5:
            interp = "strong address growth: network expansion"
        elif momentum > 0:
            interp = "mild address growth"
        elif momentum < -0.5:
            interp = "declining addresses: network contraction"
        else:
            interp = "flat address activity"

        return SignalComponent(
            name="active_addresses",
            raw_value=summary.get("current", 0),
            z_score=combined_z,
            normalized_signal=sig,
            weight=0.10,
            weighted_contribution=sig * 0.10,
            interpretation=interp,
            is_bullish=sig > 0,
            confidence=min(1.0, abs(combined_z) / 2.0),
        )


# ---------------------------------------------------------------------------
# Composite signal builder
# ---------------------------------------------------------------------------

class CompositeSignalBuilder:
    """
    Combines all component signals into a single alpha signal.
    Weights are configurable and default to research-based values.
    """

    DEFAULT_WEIGHTS = {
        "whale_exchange_flow": 0.20,
        "funding_rate":        0.20,
        "nvt_ratio":           0.15,
        "mvrv_ratio":          0.20,
        "sopr":                0.15,
        "active_addresses":    0.10,
    }

    DIRECTION_THRESHOLDS = {
        "strong_bull":  0.40,
        "bull":         0.15,
        "bear":        -0.15,
        "strong_bear": -0.40,
    }

    def __init__(
        self,
        provider: OnChainDataProvider,
        weights: Dict[str, float] = None,
        decay_half_life: float = 14.0,
    ):
        self.provider = provider
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.decay = DecayWeighter(decay_half_life)

        agg = CrossExchangeFundingAggregator()
        gl = provider.glassnode

        self.extractors: Dict[str, object] = {
            "whale_exchange_flow": WhaleFlowSignalExtractor(gl, self.decay),
            "funding_rate":        FundingRateSignalExtractor(agg, self.decay),
            "nvt_ratio":           NVTSignalExtractor(gl, self.decay),
            "mvrv_ratio":          MVRVSignalExtractor(gl, self.decay),
            "sopr":                SOPRSignalExtractor(gl, self.decay),
            "active_addresses":    ActiveAddressSignalExtractor(gl, self.decay),
        }

        self.cycle_detector = CyclePhaseDetector()
        self._nvt_calc  = NVTCalculator(gl)
        self._mvrv_calc = MVRVCalculator(gl)
        self._sopr_calc = SOPRCalculator(gl)

    def build(self, asset: str = "BTC") -> CompositeSignal:
        components = []
        errors = []

        for name, extractor in self.extractors.items():
            try:
                comp = extractor.extract(asset)
                # Apply configured weight (overrides extractor default)
                configured_weight = self.weights.get(name, comp.weight)
                comp.weight = configured_weight
                comp.weighted_contribution = comp.normalized_signal * configured_weight
                components.append(comp)
            except Exception as exc:
                logger.warning("Signal extraction failed for %s: %s", name, exc)
                errors.append(name)

        if not components:
            return CompositeSignal(
                timestamp=datetime.now(timezone.utc),
                asset=asset,
                composite_score=0.0,
                composite_z=0.0,
                direction="neutral",
                confidence=0.0,
                components=[],
                cycle_phase="unknown",
                cycle_bias="neutral",
                signal_agreement=0.5,
                breakdown={},
            )

        # Normalize weights to sum to 1
        total_w = sum(c.weight for c in components)
        if total_w > 0:
            for c in components:
                c.weight /= total_w
                c.weighted_contribution = c.normalized_signal * c.weight

        composite = sum(c.weighted_contribution for c in components)

        # Composite z-score: weighted average of component z-scores
        composite_z = sum(c.z_score * c.weight for c in components)

        # Direction classification
        if composite >= self.DIRECTION_THRESHOLDS["strong_bull"]:
            direction = "strong_bull"
        elif composite >= self.DIRECTION_THRESHOLDS["bull"]:
            direction = "bull"
        elif composite <= self.DIRECTION_THRESHOLDS["strong_bear"]:
            direction = "strong_bear"
        elif composite <= self.DIRECTION_THRESHOLDS["bear"]:
            direction = "bear"
        else:
            direction = "neutral"

        # Agreement: fraction of components pointing same direction as composite
        bull_count = sum(1 for c in components if c.is_bullish)
        agreement = bull_count / len(components) if composite > 0 else (1 - bull_count / len(components))

        # Confidence: weighted average of component confidences × agreement factor
        avg_confidence = float(np.mean([c.confidence for c in components]))
        confidence = avg_confidence * (0.5 + 0.5 * agreement)

        # Cycle phase
        try:
            nvt  = self._nvt_calc.compute(asset, 180)
            mvrv = self._mvrv_calc.compute(asset, 365)
            sopr = self._sopr_calc.compute(asset, 60)
            phase = self.cycle_detector.detect(nvt, mvrv, sopr)
            cycle_bias = self.cycle_detector.phase_to_bias(phase)
        except Exception:
            phase = "unknown"
            cycle_bias = "neutral"

        breakdown = {c.name: c.normalized_signal for c in components}
        breakdown["composite"] = composite
        breakdown["agreement"] = agreement

        return CompositeSignal(
            timestamp=datetime.now(timezone.utc),
            asset=asset,
            composite_score=round(composite, 4),
            composite_z=round(composite_z, 3),
            direction=direction,
            confidence=round(confidence, 3),
            components=components,
            cycle_phase=phase,
            cycle_bias=cycle_bias,
            signal_agreement=round(agreement, 3),
            breakdown=breakdown,
        )

    def format_signal(self, signal: CompositeSignal) -> str:
        lines = [
            f"=== On-Chain Composite Signal: {signal.asset} ===",
            f"Timestamp: {signal.timestamp.strftime('%Y-%m-%d %H:%M')} UTC",
            f"Signal: {signal.signal_summary}",
            f"Cycle Phase: {signal.cycle_phase.upper()} → {signal.cycle_bias.upper()}",
            f"Component Agreement: {signal.signal_agreement:.0%}",
            "",
            f"{'Component':>25} {'Raw':>12} {'Z-Score':>8} {'Signal':>8} {'Weight':>7} {'Contrib':>8}",
            "-" * 75,
        ]
        for c in signal.components:
            arrow = "▲" if c.is_bullish else "▼"
            lines.append(
                f"{c.name:>25} {c.raw_value:>12.4f} {c.z_score:>8.2f} "
                f"{c.normalized_signal:>8.3f} {c.weight:>7.1%} {c.weighted_contribution:>8.4f} {arrow}"
            )
        lines.append("-" * 75)
        lines.append(f"{'COMPOSITE':>25} {'':>12} {signal.composite_z:>8.2f} "
                    f"{signal.composite_score:>8.3f} {'1.000':>7} {signal.composite_score:>8.4f}")
        lines.append("")
        lines.append("Component Interpretations:")
        for c in signal.components:
            lines.append(f"  {c.name}: {c.interpretation}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Signal history tracker
# ---------------------------------------------------------------------------

class SignalHistoryTracker:
    """Tracks signal history for backtest and regime analysis."""

    def __init__(self, max_history: int = 252):
        self.max_history = max_history
        self._history: List[CompositeSignal] = []

    def record(self, signal: CompositeSignal) -> None:
        self._history.append(signal)
        if len(self._history) > self.max_history:
            self._history.pop(0)

    def get_history(self) -> List[CompositeSignal]:
        return list(self._history)

    def average_score(self, lookback: int = 7) -> float:
        recent = self._history[-lookback:]
        if not recent:
            return 0.0
        return float(np.mean([s.composite_score for s in recent]))

    def signal_changes(self) -> List[Dict]:
        """Detect direction changes in the signal."""
        changes = []
        for i in range(1, len(self._history)):
            prev = self._history[i - 1]
            curr = self._history[i]
            if prev.direction != curr.direction:
                changes.append({
                    "dt": curr.timestamp.isoformat(),
                    "from": prev.direction,
                    "to": curr.direction,
                    "score": curr.composite_score,
                })
        return changes

    def score_series(self) -> List[Tuple[datetime, float]]:
        return [(s.timestamp, s.composite_score) for s in self._history]

    def backtest_summary(self, price_returns: List[float] = None) -> Dict:
        """
        If price_returns (daily returns) are provided, compute signal-return correlation.
        """
        scores = [s.composite_score for s in self._history]
        if not scores:
            return {}

        summary = {
            "count": len(scores),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "pct_bull": float(np.mean([1 for s in scores if s > 0.15])),
            "pct_bear": float(np.mean([1 for s in scores if s < -0.15])),
            "pct_neutral": float(np.mean([1 for s in scores if -0.15 <= s <= 0.15])),
        }

        if price_returns and len(price_returns) == len(scores):
            ret_arr = np.array(price_returns)
            score_arr = np.array(scores[:-1])  # lag 1
            ret_arr = ret_arr[1:]
            if len(score_arr) == len(ret_arr) and np.std(score_arr) > 0 and np.std(ret_arr) > 0:
                corr = float(np.corrcoef(score_arr, ret_arr)[0, 1])
                summary["signal_return_correlation"] = corr
                summary["information_coefficient"] = corr

        return summary


# ---------------------------------------------------------------------------
# Multi-asset scanner
# ---------------------------------------------------------------------------

class MultiAssetSignalScanner:
    """Scans composite signals across multiple crypto assets."""

    SUPPORTED_ASSETS = ["BTC", "ETH"]   # add more as Glassnode coverage allows

    def __init__(self, provider: OnChainDataProvider = None, weights: Dict[str, float] = None):
        self.provider = provider or OnChainDataProvider()
        self.weights = weights
        self._builder = CompositeSignalBuilder(self.provider, weights)

    def scan_all(self) -> List[CompositeSignal]:
        signals = []
        for asset in self.SUPPORTED_ASSETS:
            try:
                sig = self._builder.build(asset)
                signals.append(sig)
            except Exception as exc:
                logger.error("Signal build failed for %s: %s", asset, exc)
        return sorted(signals, key=lambda s: s.composite_score, reverse=True)

    def format_all(self, signals: List[CompositeSignal] = None) -> str:
        if signals is None:
            signals = self.scan_all()
        lines = [
            "=== Multi-Asset On-Chain Signal Scanner ===",
            f"Scan time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC",
            "",
            f"{'Asset':>6} {'Score':>7} {'Direction':>12} {'Conf':>6} {'Phase':>15} {'Bias':>12}",
            "-" * 65,
        ]
        for s in signals:
            lines.append(
                f"{s.asset:>6} {s.composite_score:>+7.3f} {s.direction:>12} "
                f"{s.confidence:>6.0%} {s.cycle_phase:>15} {s.cycle_bias:>12}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main OnChainSignals facade
# ---------------------------------------------------------------------------

class OnChainSignals:
    """
    Top-level API for on-chain alpha signal generation.
    """

    def __init__(self, provider: OnChainDataProvider = None, weights: Dict[str, float] = None):
        self.provider = provider or OnChainDataProvider()
        self.builder = CompositeSignalBuilder(self.provider, weights)
        self.scanner = MultiAssetSignalScanner(self.provider, weights)
        self.history = SignalHistoryTracker()

    def get_signal(self, asset: str = "BTC") -> CompositeSignal:
        signal = self.builder.build(asset)
        self.history.record(signal)
        return signal

    def format_signal(self, asset: str = "BTC") -> str:
        signal = self.get_signal(asset)
        return self.builder.format_signal(signal)

    def scan_all_assets(self) -> str:
        signals = self.scanner.scan_all()
        return self.scanner.format_all(signals)

    def get_component_breakdown(self, asset: str = "BTC") -> Dict[str, Dict]:
        signal = self.get_signal(asset)
        return {
            c.name: {
                "raw_value": c.raw_value,
                "z_score": c.z_score,
                "signal": c.normalized_signal,
                "interpretation": c.interpretation,
                "is_bullish": c.is_bullish,
                "confidence": c.confidence,
                "weight": c.weight,
            }
            for c in signal.components
        }

    def signal_history_summary(self) -> Dict:
        return self.history.backtest_summary()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="On-chain signals CLI")
    parser.add_argument("--action", choices=["signal", "scan", "breakdown"], default="signal")
    parser.add_argument("--asset", default="BTC")
    args = parser.parse_args()

    signals = OnChainSignals()

    if args.action == "signal":
        print(signals.format_signal(args.asset))
    elif args.action == "scan":
        print(signals.scan_all_assets())
    elif args.action == "breakdown":
        import json as _json
        breakdown = signals.get_component_breakdown(args.asset)
        print(_json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, bool)} for k, v in breakdown.items()}, indent=2))
