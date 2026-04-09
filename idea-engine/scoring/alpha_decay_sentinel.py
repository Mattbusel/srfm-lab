"""
Alpha Decay Sentinel: detect hypothesis decay BEFORE it happens.

Monitors leading indicators of alpha decay:
  1. IC trend: if information coefficient is declining, alpha is eroding
  2. Regime misalignment: if the hypothesis's preferred regime is ending
  3. Crowding signal: if many similar hypotheses are active (alpha dilution)
  4. Volatility compression: if the vol regime that drives the signal is fading
  5. Correlation breakdown: if the cross-asset relationships the signal depends on are shifting

The sentinel outputs a "decay probability" for each active hypothesis,
allowing the portfolio allocator to reduce exposure BEFORE the actual
drawdown materializes.
"""

from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DecayIndicator:
    """Single leading indicator of alpha decay."""
    name: str
    value: float            # current value (0=healthy, 1=fully decayed)
    trend: float            # rate of change (positive = worsening)
    weight: float = 1.0     # importance in composite score
    threshold: float = 0.5  # above this = warning


@dataclass
class HypothesisHealth:
    """Health assessment for a single active hypothesis."""
    hypothesis_id: str
    hypothesis_name: str
    decay_probability: float     # 0-1, composite across all indicators
    days_until_estimated_decay: float  # extrapolated from trend
    indicators: List[DecayIndicator] = field(default_factory=list)
    recommendation: str = "hold"  # hold / reduce / exit / watch
    urgency: str = "low"         # low / medium / high / critical


class ICTrendMonitor:
    """Track rolling Information Coefficient and detect downtrends."""

    def __init__(self, window_fast: int = 21, window_slow: int = 63):
        self.fast = window_fast
        self.slow = window_slow

    def compute(self, ic_history: List[float]) -> DecayIndicator:
        n = len(ic_history)
        if n < self.fast:
            return DecayIndicator("ic_trend", 0.0, 0.0, weight=2.0)

        ic_fast = sum(ic_history[-self.fast:]) / self.fast
        ic_slow = sum(ic_history[-min(n, self.slow):]) / min(n, self.slow)

        # Trend: slope of IC over fast window
        if n >= self.fast:
            t = list(range(self.fast))
            ics = ic_history[-self.fast:]
            mean_t = sum(t) / len(t)
            mean_ic = sum(ics) / len(ics)
            cov = sum((ti - mean_t) * (ici - mean_ic) for ti, ici in zip(t, ics)) / len(t)
            var_t = sum((ti - mean_t)**2 for ti in t) / len(t)
            slope = cov / max(var_t, 1e-10)
        else:
            slope = 0.0

        # Decay signal: IC below half of slow average, or trending down
        if ic_slow > 0.01:
            ratio = ic_fast / ic_slow
            decay_value = max(0, 1 - ratio)
        else:
            decay_value = 0.5 if ic_fast < 0.01 else 0.0

        # Boost decay if slope is negative
        if slope < -0.001:
            decay_value = min(1.0, decay_value + abs(slope) * 100)

        return DecayIndicator(
            name="ic_trend",
            value=min(1.0, decay_value),
            trend=float(-slope * 100),  # positive trend = worsening
            weight=2.0,
            threshold=0.4,
        )


class RegimeMisalignmentMonitor:
    """Detect when the current regime is shifting away from the hypothesis's preferred regime."""

    def compute(self, preferred_regime: str, current_regime: str,
                regime_transition_prob: float = 0.0) -> DecayIndicator:
        """
        preferred_regime: the regime this hypothesis works best in
        current_regime: what the regime detector currently says
        regime_transition_prob: probability of leaving current regime (from Markov model)
        """
        is_aligned = (preferred_regime == current_regime)

        if is_aligned:
            # Currently aligned, but check if regime is about to end
            decay_value = regime_transition_prob * 0.5
            trend = regime_transition_prob
        else:
            # Misaligned: high decay signal
            decay_value = 0.7 + 0.3 * (1 - regime_transition_prob)
            trend = 0.5

        return DecayIndicator(
            name="regime_misalignment",
            value=min(1.0, decay_value),
            trend=trend,
            weight=1.5,
            threshold=0.3,
        )


class CrowdingMonitor:
    """Detect alpha dilution from too many similar active hypotheses."""

    def compute(self, hypothesis_id: str, active_hypotheses: List[Dict]) -> DecayIndicator:
        """
        Check how many active hypotheses share the same template_type and regime.
        More crowding = faster alpha decay.
        """
        if not active_hypotheses:
            return DecayIndicator("crowding", 0.0, 0.0, weight=1.0)

        # Find this hypothesis
        this_hyp = None
        for h in active_hypotheses:
            if h.get("id") == hypothesis_id:
                this_hyp = h
                break

        if this_hyp is None:
            return DecayIndicator("crowding", 0.0, 0.0, weight=1.0)

        # Count similar hypotheses
        template = this_hyp.get("template_type", "")
        regime = this_hyp.get("regime", "")
        direction = this_hyp.get("direction", 0)

        similar_count = sum(
            1 for h in active_hypotheses
            if h.get("id") != hypothesis_id
            and h.get("template_type") == template
            and h.get("direction") == direction
        )

        # More than 3 similar = crowding concern
        if similar_count <= 1:
            decay_value = 0.0
        elif similar_count <= 3:
            decay_value = similar_count * 0.15
        else:
            decay_value = min(1.0, 0.5 + (similar_count - 3) * 0.1)

        return DecayIndicator(
            name="crowding",
            value=decay_value,
            trend=0.0,  # would need historical crowding data for trend
            weight=1.0,
            threshold=0.4,
        )


class VolatilityCompressionMonitor:
    """Detect when vol is compressing, which kills vol-dependent signals."""

    def compute(self, vol_history: List[float], signal_type: str) -> DecayIndicator:
        """
        vol_history: recent realized vol values
        signal_type: type of signal (momentum signals need vol; mean_rev needs vol too)
        """
        if len(vol_history) < 21:
            return DecayIndicator("vol_compression", 0.0, 0.0, weight=1.0)

        vol_fast = sum(vol_history[-5:]) / 5
        vol_slow = sum(vol_history[-21:]) / 21

        # Vol compression ratio
        if vol_slow > 0.01:
            compression = 1 - (vol_fast / vol_slow)
        else:
            compression = 0.0

        # Momentum strategies need volatility. If vol is compressing, alpha decays.
        vol_dependent = signal_type in ("momentum", "breakout", "volatility_arb", "trend_following")
        weight = 1.5 if vol_dependent else 0.5

        # Compression > 0 means vol is falling
        decay_value = max(0, compression) if vol_dependent else max(0, -compression) * 0.5

        return DecayIndicator(
            name="vol_compression",
            value=min(1.0, decay_value),
            trend=float(compression),
            weight=weight,
            threshold=0.3,
        )


class AlphaDecaySentinel:
    """
    Master sentinel: combines all leading indicators into a composite
    decay probability for each active hypothesis.
    """

    def __init__(self):
        self.ic_monitor = ICTrendMonitor()
        self.regime_monitor = RegimeMisalignmentMonitor()
        self.crowding_monitor = CrowdingMonitor()
        self.vol_monitor = VolatilityCompressionMonitor()

    def assess(
        self,
        hypothesis_id: str,
        hypothesis_name: str,
        template_type: str,
        preferred_regime: str,
        current_regime: str,
        regime_transition_prob: float,
        ic_history: List[float],
        vol_history: List[float],
        active_hypotheses: List[Dict],
    ) -> HypothesisHealth:
        """
        Assess health of a single hypothesis using all leading indicators.
        Returns HypothesisHealth with decay_probability and recommendation.
        """
        indicators = []

        # 1. IC trend
        indicators.append(self.ic_monitor.compute(ic_history))

        # 2. Regime alignment
        indicators.append(self.regime_monitor.compute(
            preferred_regime, current_regime, regime_transition_prob
        ))

        # 3. Crowding
        indicators.append(self.crowding_monitor.compute(hypothesis_id, active_hypotheses))

        # 4. Volatility compression
        indicators.append(self.vol_monitor.compute(vol_history, template_type))

        # Composite decay probability (weighted average)
        total_weight = sum(ind.weight for ind in indicators)
        if total_weight > 0:
            decay_prob = sum(ind.value * ind.weight for ind in indicators) / total_weight
        else:
            decay_prob = 0.0

        decay_prob = min(1.0, max(0.0, decay_prob))

        # Estimate days until decay (from trend)
        active_trends = [ind.trend for ind in indicators if ind.trend > 0]
        if active_trends and decay_prob < 0.8:
            avg_trend = sum(active_trends) / len(active_trends)
            remaining = (0.8 - decay_prob) / max(avg_trend, 0.001)
            days_est = max(1, remaining * 5)  # rough: each trend unit ~ 5 days
        else:
            days_est = float("inf") if decay_prob < 0.3 else 1.0

        # Recommendation
        if decay_prob > 0.8:
            recommendation = "exit"
            urgency = "critical"
        elif decay_prob > 0.6:
            recommendation = "reduce"
            urgency = "high"
        elif decay_prob > 0.4:
            recommendation = "watch"
            urgency = "medium"
        else:
            recommendation = "hold"
            urgency = "low"

        return HypothesisHealth(
            hypothesis_id=hypothesis_id,
            hypothesis_name=hypothesis_name,
            decay_probability=decay_prob,
            days_until_estimated_decay=days_est,
            indicators=indicators,
            recommendation=recommendation,
            urgency=urgency,
        )

    def scan_all(
        self,
        hypotheses: List[Dict],
        current_regime: str,
        regime_transition_prob: float,
        vol_history: List[float],
    ) -> List[HypothesisHealth]:
        """Scan all active hypotheses and return sorted by decay probability."""
        results = []
        for hyp in hypotheses:
            health = self.assess(
                hypothesis_id=hyp.get("id", ""),
                hypothesis_name=hyp.get("name", ""),
                template_type=hyp.get("template_type", "unknown"),
                preferred_regime=hyp.get("regime", "unknown"),
                current_regime=current_regime,
                regime_transition_prob=regime_transition_prob,
                ic_history=hyp.get("ic_history", []),
                vol_history=vol_history,
                active_hypotheses=hypotheses,
            )
            results.append(health)

        results.sort(key=lambda h: h.decay_probability, reverse=True)
        return results

    def alert_summary(self, results: List[HypothesisHealth]) -> Dict:
        """Generate alert summary from scan results."""
        critical = [h for h in results if h.urgency == "critical"]
        high = [h for h in results if h.urgency == "high"]
        medium = [h for h in results if h.urgency == "medium"]

        return {
            "total_scanned": len(results),
            "critical_count": len(critical),
            "high_count": len(high),
            "medium_count": len(medium),
            "healthy_count": len(results) - len(critical) - len(high) - len(medium),
            "critical_hypotheses": [
                {"id": h.hypothesis_id, "name": h.hypothesis_name,
                 "decay_prob": h.decay_probability, "action": h.recommendation}
                for h in critical
            ],
            "high_hypotheses": [
                {"id": h.hypothesis_id, "name": h.hypothesis_name,
                 "decay_prob": h.decay_probability, "days_est": h.days_until_estimated_decay}
                for h in high
            ],
        }
