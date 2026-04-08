"""
Regime expert debate agent — evaluates hypotheses through the lens of market regime.

Specializes in:
  - Regime identification and classification
  - Regime-conditional signal validity
  - Regime transition risk
  - Historical regime performance attribution
  - Regime persistence estimation
  - Regime-adaptive position sizing recommendations
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


REGIMES = ["trending_bull", "trending_bear", "mean_reverting_low_vol",
           "mean_reverting_high_vol", "chaotic", "crisis", "recovery"]

REGIME_SIGNAL_AFFINITY = {
    # Which signal types work in which regimes
    "momentum": {
        "trending_bull": 0.85, "trending_bear": 0.80,
        "mean_reverting_low_vol": 0.20, "mean_reverting_high_vol": 0.15,
        "chaotic": 0.30, "crisis": 0.25, "recovery": 0.60,
    },
    "mean_reversion": {
        "trending_bull": 0.25, "trending_bear": 0.20,
        "mean_reverting_low_vol": 0.90, "mean_reverting_high_vol": 0.70,
        "chaotic": 0.35, "crisis": 0.10, "recovery": 0.55,
    },
    "market_making": {
        "trending_bull": 0.60, "trending_bear": 0.50,
        "mean_reverting_low_vol": 0.85, "mean_reverting_high_vol": 0.40,
        "chaotic": 0.20, "crisis": 0.05, "recovery": 0.65,
    },
    "liquidation_cascade": {
        "trending_bull": 0.30, "trending_bear": 0.70,
        "mean_reverting_low_vol": 0.20, "mean_reverting_high_vol": 0.55,
        "chaotic": 0.80, "crisis": 0.90, "recovery": 0.35,
    },
    "on_chain": {
        "trending_bull": 0.75, "trending_bear": 0.65,
        "mean_reverting_low_vol": 0.60, "mean_reverting_high_vol": 0.55,
        "chaotic": 0.40, "crisis": 0.35, "recovery": 0.70,
    },
}

REGIME_RISK_MULTIPLIERS = {
    "trending_bull": 1.0,
    "trending_bear": 1.2,
    "mean_reverting_low_vol": 0.8,
    "mean_reverting_high_vol": 1.4,
    "chaotic": 2.0,
    "crisis": 3.0,
    "recovery": 1.1,
}


@dataclass
class RegimeEvaluation:
    regime: str
    regime_confidence: float
    signal_affinity: float
    regime_risk_multiplier: float
    transition_risk: float
    recommended_size_scalar: float
    regime_duration_estimate: int          # periods
    historical_regime_sharpe: Optional[float]
    warnings: list[str]
    verdict: str                           # FAVORABLE, NEUTRAL, UNFAVORABLE, VETO
    reasoning: str


class RegimeExpertAgent:
    """
    Evaluates hypothesis from regime perspective.
    Updates regime beliefs via rolling feature analysis.
    """

    def __init__(
        self,
        vol_window: int = 20,
        trend_window: int = 60,
        corr_window: int = 30,
    ):
        self.vol_window = vol_window
        self.trend_window = trend_window
        self.corr_window = corr_window
        self._regime_history: list[str] = []
        self._performance_by_regime: dict[str, list[float]] = {r: [] for r in REGIMES}

    def classify_regime(
        self,
        returns: np.ndarray,
        volume: Optional[np.ndarray] = None,
        cross_asset_returns: Optional[np.ndarray] = None,
    ) -> tuple[str, float]:
        """
        Classify current market regime from return features.
        Returns (regime_name, confidence).
        """
        n = len(returns)
        if n < self.vol_window:
            return "mean_reverting_low_vol", 0.3

        # Feature extraction
        recent = returns[-self.vol_window:]
        baseline = returns[-min(self.trend_window, n):]

        vol_recent = float(recent.std() * math.sqrt(252))
        vol_baseline = float(baseline.std() * math.sqrt(252))
        vol_ratio = vol_recent / max(vol_baseline, 1e-6)

        # Trend strength (Hurst-like via R/S over short window)
        trend_5d = float(returns[-min(5, n):].sum())
        trend_20d = float(returns[-min(20, n):].sum())
        trend_60d = float(returns[-min(60, n):].sum())

        # Autocorrelation
        if n >= 5:
            acf1 = float(np.corrcoef(returns[1:], returns[:-1])[0, 1])
        else:
            acf1 = 0.0

        # Cross-asset correlation spike (crisis signal)
        corr_spike = 0.0
        if cross_asset_returns is not None and len(cross_asset_returns.shape) == 2:
            corr = np.corrcoef(cross_asset_returns.T)
            off_diag = corr[np.triu_indices_from(corr, k=1)]
            corr_spike = float(off_diag.mean())

        # Regime scoring
        scores = {r: 0.0 for r in REGIMES}

        # Crisis: extreme vol + high cross-asset correlation
        if vol_ratio > 2.5 or (corr_spike > 0.7 and vol_ratio > 1.5):
            scores["crisis"] += 3.0
        # Chaotic: high vol, no trend
        elif vol_ratio > 1.8 and abs(trend_20d) < vol_recent * 0.5:
            scores["chaotic"] += 2.5
        # Trending: strong directional move
        elif abs(trend_20d) > vol_recent * 1.0:
            if trend_20d > 0:
                scores["trending_bull"] += 2.0 + abs(trend_60d) * 5
            else:
                scores["trending_bear"] += 2.0 + abs(trend_60d) * 5
        # Mean reverting: negative autocorrelation + moderate vol
        elif acf1 < -0.1 and vol_ratio < 1.3:
            if vol_recent < 0.20:
                scores["mean_reverting_low_vol"] += 2.5
            else:
                scores["mean_reverting_high_vol"] += 2.0
        # Recovery: recent strong move after low point
        elif trend_5d > 0.03 and trend_60d < -0.1:
            scores["recovery"] += 2.0
        else:
            scores["mean_reverting_low_vol"] += 1.0

        best_regime = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = float(scores[best_regime] / max(total_score, 1e-6))

        self._regime_history.append(best_regime)
        return best_regime, min(confidence, 0.95)

    def regime_transition_probability(self, regime: str, window: int = 20) -> dict:
        """
        Estimate probability of transitioning to each other regime.
        Based on historical regime sequence in self._regime_history.
        """
        history = self._regime_history
        if len(history) < 2:
            return {r: 1.0 / len(REGIMES) for r in REGIMES}

        transitions = {r: 0 for r in REGIMES}
        total = 0
        for i in range(len(history) - 1):
            if history[i] == regime:
                transitions[history[i + 1]] += 1
                total += 1

        if total == 0:
            return {r: 1.0 / len(REGIMES) for r in REGIMES}

        return {r: transitions[r] / total for r in REGIMES}

    def regime_persistence(self, regime: str) -> float:
        """
        Estimate average duration of a regime in periods.
        Based on observed regime runs.
        """
        history = self._regime_history
        if not history:
            return 20.0

        runs = []
        current_run = 0
        for r in history:
            if r == regime:
                current_run += 1
            elif current_run > 0:
                runs.append(current_run)
                current_run = 0
        if current_run > 0:
            runs.append(current_run)

        return float(sum(runs) / len(runs)) if runs else 20.0

    def update_regime_performance(self, regime: str, realized_sharpe: float) -> None:
        """Record realized performance during a regime for calibration."""
        if regime in self._performance_by_regime:
            self._performance_by_regime[regime].append(realized_sharpe)

    def historical_regime_sharpe(self, regime: str, edge_type: str) -> Optional[float]:
        """Historical Sharpe for this edge type in this regime."""
        # Use affinity as proxy if no history
        affinity = REGIME_SIGNAL_AFFINITY.get(edge_type, {}).get(regime, 0.5)
        perf = self._performance_by_regime.get(regime, [])
        if perf:
            return float(sum(perf) / len(perf))
        # Proxy from affinity: affinity=1 → Sharpe~2, affinity=0 → Sharpe~-0.5
        return float(affinity * 2.5 - 0.5)

    def evaluate(
        self,
        hypothesis: dict,
        returns: Optional[np.ndarray] = None,
        volume: Optional[np.ndarray] = None,
        cross_asset_returns: Optional[np.ndarray] = None,
    ) -> RegimeEvaluation:
        """
        Full regime evaluation of a hypothesis.
        """
        # Classify current regime
        if returns is not None and len(returns) >= self.vol_window:
            regime, regime_conf = self.classify_regime(returns, volume, cross_asset_returns)
        else:
            regime = hypothesis.get("regime_at_creation", "mean_reverting_low_vol")
            regime_conf = 0.4

        edge = hypothesis.get("edge", "momentum")
        tags = hypothesis.get("tags", [])

        # Signal affinity for this edge in current regime
        edge_affinities = REGIME_SIGNAL_AFFINITY.get(edge, {})
        affinity = float(edge_affinities.get(regime, 0.5))

        # Check tags for more specific affinity
        for tag in tags:
            if tag in REGIME_SIGNAL_AFFINITY:
                affinity = max(affinity, REGIME_SIGNAL_AFFINITY[tag].get(regime, affinity))

        risk_mult = REGIME_RISK_MULTIPLIERS.get(regime, 1.0)
        hist_sharpe = self.historical_regime_sharpe(regime, edge)

        # Transition risk: probability of regime changing adversely
        transition_probs = self.regime_transition_probability(regime)
        adverse_regimes = ["crisis", "chaotic"]
        transition_risk = float(sum(transition_probs.get(r, 0) for r in adverse_regimes))

        # Regime persistence
        expected_duration = self.regime_persistence(regime)

        # Position size scalar
        base_size = 1.0
        affinity_adj = (affinity - 0.5) * 2   # -1 to +1
        size_scalar = float(max(0.1, base_size + affinity_adj * 0.6 - transition_risk * 0.3))
        size_scalar /= risk_mult

        # Build warnings
        warnings = []
        if transition_risk > 0.25:
            warnings.append(f"High regime transition risk to adverse regime: {transition_risk:.1%}")
        if affinity < 0.35:
            warnings.append(f"Edge type '{edge}' has low affinity for {regime}: {affinity:.2f}")
        if regime == "crisis":
            warnings.append("CRISIS REGIME: All non-crisis strategies impaired, reduce size severely")
        if regime_conf < 0.5:
            warnings.append(f"Regime classification confidence low: {regime_conf:.2f}")
        if expected_duration < 5:
            warnings.append(f"Short expected regime duration: ~{expected_duration:.0f} periods")

        # Verdict
        if regime == "crisis" and edge not in ["liquidation_cascade"]:
            verdict = "VETO"
        elif affinity >= 0.75 and regime_conf >= 0.6:
            verdict = "FAVORABLE"
        elif affinity >= 0.45:
            verdict = "NEUTRAL"
        else:
            verdict = "UNFAVORABLE"

        # Reasoning
        reasoning_parts = [
            f"Current regime: {regime} (confidence: {regime_conf:.2f})",
            f"Signal affinity: {affinity:.2f}",
            f"Risk multiplier: {risk_mult:.1f}x",
            f"Expected regime duration: ~{expected_duration:.0f} periods",
            f"Transition risk to crisis/chaotic: {transition_risk:.1%}",
            f"Historical Sharpe in {regime}: {hist_sharpe:.2f}" if hist_sharpe else "",
            f"Recommended size scalar: {size_scalar:.2f}x",
        ]
        if warnings:
            reasoning_parts.append("WARNINGS: " + "; ".join(warnings))

        return RegimeEvaluation(
            regime=regime,
            regime_confidence=regime_conf,
            signal_affinity=affinity,
            regime_risk_multiplier=risk_mult,
            transition_risk=transition_risk,
            recommended_size_scalar=size_scalar,
            regime_duration_estimate=int(expected_duration),
            historical_regime_sharpe=hist_sharpe,
            warnings=warnings,
            verdict=verdict,
            reasoning="\n".join(r for r in reasoning_parts if r),
        )
