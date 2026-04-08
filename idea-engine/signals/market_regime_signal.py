"""
Market regime detection signal — unified multi-method regime classifier.

Combines:
  - HMM-based regime detection (bull/bear/sideways)
  - Volatility regime (low/normal/high/crisis)
  - Trend regime (strong up/weak up/flat/weak down/strong down)
  - Correlation regime (normal/crisis/decorrelated)
  - Liquidity regime (abundant/normal/stressed)
  - Macro regime proxy (risk-on/off/transition)

Outputs a comprehensive RegimeSnapshot used to route all downstream signals.
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RegimeSnapshot:
    """Complete regime state at a point in time."""
    # Core regime
    primary_regime: str          # bull/bear/sideways/crisis
    regime_confidence: float     # 0-1

    # Sub-regimes
    vol_regime: str              # low/normal/high/crisis
    trend_regime: str            # strong_up/weak_up/flat/weak_down/strong_down
    correlation_regime: str      # normal/crisis/decorrelated
    liquidity_regime: str        # abundant/normal/stressed

    # Quantitative signals
    vol_level: float             # realized annualized vol
    trend_score: float           # -1 to +1
    momentum_score: float        # -1 to +1
    mean_reversion_score: float  # -1 to +1

    # Regime probability vector (bull, bear, sideways)
    regime_probs: np.ndarray = field(default_factory=lambda: np.array([1/3, 1/3, 1/3]))

    # Transition flags
    regime_change: bool = False
    days_in_regime: int = 0
    regime_stability: float = 0.5   # 0=unstable, 1=stable


def _rolling_vol(returns: np.ndarray, window: int, ann_factor: float = 252.0) -> float:
    n = min(len(returns), window)
    return float(returns[-n:].std() * math.sqrt(ann_factor)) if n >= 2 else 0.0


def _trend_score(prices: np.ndarray, window: int) -> float:
    """Normalized trend score via regression slope."""
    n = min(len(prices), window)
    if n < 3:
        return 0.0
    p = prices[-n:]
    t = np.arange(n)
    slope = float(np.polyfit(t, p, 1)[0])
    vol = float(p.std())
    return float(math.tanh(slope / max(vol, 1e-10) * n * 0.1))


def _momentum_score(prices: np.ndarray, lookback: int) -> float:
    """Rate of change momentum."""
    n = min(len(prices), lookback + 1)
    if n < 2:
        return 0.0
    roc = (prices[-1] - prices[-n]) / max(abs(prices[-n]), 1e-10)
    return float(math.tanh(roc * 5))


def _mean_reversion_score(prices: np.ndarray, window: int) -> float:
    """Z-score based mean reversion."""
    n = min(len(prices), window)
    if n < 5:
        return 0.0
    sub = prices[-n:]
    z = float((prices[-1] - sub.mean()) / max(sub.std(), 1e-10))
    return float(-math.tanh(z))


def classify_vol_regime(vol: float) -> str:
    if vol < 0.10:
        return "low"
    elif vol < 0.20:
        return "normal"
    elif vol < 0.40:
        return "high"
    else:
        return "crisis"


def classify_trend_regime(trend: float) -> str:
    if trend > 0.5:
        return "strong_up"
    elif trend > 0.15:
        return "weak_up"
    elif trend > -0.15:
        return "flat"
    elif trend > -0.5:
        return "weak_down"
    else:
        return "strong_down"


def classify_primary_regime(
    trend: float,
    vol_regime: str,
    momentum: float,
) -> tuple[str, float]:
    """Classify primary regime with confidence."""
    if vol_regime == "crisis":
        return "crisis", 0.85

    if trend > 0.3 and momentum > 0.2:
        confidence = min(trend + momentum, 1.0) * 0.7
        return "bull", confidence

    if trend < -0.3 and momentum < -0.2:
        confidence = min(abs(trend) + abs(momentum), 1.0) * 0.7
        return "bear", confidence

    confidence = max(0.3, 1 - abs(trend) - abs(momentum))
    return "sideways", confidence


class MarketRegimeSignal:
    """
    Multi-method market regime classifier.
    Uses HMM-inspired probabilities + rule-based classification.
    """

    def __init__(
        self,
        trend_window: int = 50,
        vol_window: int = 21,
        momentum_window: int = 20,
        mr_window: int = 30,
        smoothing_alpha: float = 0.2,
    ):
        self.trend_window = trend_window
        self.vol_window = vol_window
        self.momentum_window = momentum_window
        self.mr_window = mr_window
        self.alpha = smoothing_alpha

        # State
        self._prev_regime = "sideways"
        self._days_in_regime = 0
        self._regime_probs = np.array([1/3, 1/3, 1/3])  # bull, bear, sideways
        self._ema_vol = None

    def compute(self, prices: np.ndarray) -> RegimeSnapshot:
        """Compute regime snapshot from price series."""
        T = len(prices)
        if T < max(self.trend_window, self.vol_window) + 5:
            return RegimeSnapshot(
                primary_regime="insufficient_data",
                regime_confidence=0.0,
                vol_regime="normal",
                trend_regime="flat",
                correlation_regime="normal",
                liquidity_regime="normal",
                vol_level=0.0,
                trend_score=0.0,
                momentum_score=0.0,
                mean_reversion_score=0.0,
            )

        returns = np.diff(np.log(prices + 1e-10))

        # Core measurements
        vol = _rolling_vol(returns, self.vol_window)
        if self._ema_vol is None:
            self._ema_vol = vol
        else:
            self._ema_vol = self.alpha * vol + (1 - self.alpha) * self._ema_vol

        trend = _trend_score(prices, self.trend_window)
        momentum = _momentum_score(prices, self.momentum_window)
        mr_score = _mean_reversion_score(prices, self.mr_window)

        # Regime classifications
        vol_regime = classify_vol_regime(vol)
        trend_regime = classify_trend_regime(trend)
        primary, confidence = classify_primary_regime(trend, vol_regime, momentum)

        # Smooth regime probabilities
        raw_probs = self._scores_to_probs(trend, momentum, vol_regime)
        self._regime_probs = (
            (1 - self.alpha) * self._regime_probs + self.alpha * raw_probs
        )
        self._regime_probs /= self._regime_probs.sum() + 1e-10

        # Detect regime change
        regime_change = primary != self._prev_regime
        if regime_change:
            self._days_in_regime = 0
        else:
            self._days_in_regime += 1
        self._prev_regime = primary

        # Regime stability
        max_prob = float(self._regime_probs.max())
        stability = float(max_prob * (1 - max(abs(self._regime_probs[0] - self._regime_probs[1]), 0) * 0.5))

        # Correlation regime (proxy from autocorrelation)
        if T >= 21:
            acf1 = float(np.corrcoef(returns[-21:][1:], returns[-21:][:-1])[0, 1])
            if acf1 > 0.3:
                corr_regime = "trending"
            elif acf1 < -0.3:
                corr_regime = "mean_reverting"
            elif vol_regime == "crisis":
                corr_regime = "crisis"
            else:
                corr_regime = "normal"
        else:
            corr_regime = "normal"

        # Liquidity regime (proxy from vol and autocorrelation)
        if vol_regime == "crisis":
            liq_regime = "stressed"
        elif vol_regime == "high":
            liq_regime = "stressed"
        elif vol_regime == "low":
            liq_regime = "abundant"
        else:
            liq_regime = "normal"

        return RegimeSnapshot(
            primary_regime=primary,
            regime_confidence=confidence,
            vol_regime=vol_regime,
            trend_regime=trend_regime,
            correlation_regime=corr_regime,
            liquidity_regime=liq_regime,
            vol_level=vol,
            trend_score=trend,
            momentum_score=momentum,
            mean_reversion_score=mr_score,
            regime_probs=self._regime_probs.copy(),
            regime_change=regime_change,
            days_in_regime=self._days_in_regime,
            regime_stability=stability,
        )

    def _scores_to_probs(
        self,
        trend: float,
        momentum: float,
        vol_regime: str,
    ) -> np.ndarray:
        """Convert trend/momentum/vol to regime probabilities [bull, bear, sideways]."""
        # Bull: positive trend + momentum
        bull_score = max(trend, 0) * 0.5 + max(momentum, 0) * 0.5
        # Bear: negative trend + momentum
        bear_score = max(-trend, 0) * 0.5 + max(-momentum, 0) * 0.5
        # Sideways: neither
        side_score = max(1 - abs(trend) - abs(momentum), 0.1)

        if vol_regime == "crisis":
            bear_score *= 1.5

        probs = np.array([bull_score, bear_score, side_score])
        probs = np.maximum(probs, 0.05)
        probs /= probs.sum()
        return probs

    def rolling_regimes(
        self,
        prices: np.ndarray,
        window: int = 200,
        step: int = 5,
    ) -> list[RegimeSnapshot]:
        """Compute rolling regime snapshots for the full price series."""
        T = len(prices)
        snapshots = []
        # Reset state
        self._prev_regime = "sideways"
        self._days_in_regime = 0
        self._regime_probs = np.array([1/3, 1/3, 1/3])
        self._ema_vol = None

        for t in range(window, T + 1, step):
            snap = self.compute(prices[max(0, t - window): t])
            snapshots.append(snap)

        return snapshots


def regime_to_signal_weights(snapshot: RegimeSnapshot) -> dict:
    """
    Given a regime snapshot, return recommended signal weights.
    This drives the IdeaPipeline's template selection.
    """
    weights = {
        "trend": 0.25,
        "momentum": 0.25,
        "mean_reversion": 0.25,
        "volatility": 0.15,
        "microstructure": 0.10,
    }

    regime = snapshot.primary_regime
    vol = snapshot.vol_regime

    if regime == "bull":
        weights["trend"] = 0.40
        weights["momentum"] = 0.35
        weights["mean_reversion"] = 0.10
        weights["volatility"] = 0.10
        weights["microstructure"] = 0.05

    elif regime == "bear":
        weights["trend"] = 0.35
        weights["momentum"] = 0.30
        weights["mean_reversion"] = 0.15
        weights["volatility"] = 0.10
        weights["microstructure"] = 0.10

    elif regime == "sideways":
        weights["trend"] = 0.10
        weights["momentum"] = 0.10
        weights["mean_reversion"] = 0.45
        weights["volatility"] = 0.20
        weights["microstructure"] = 0.15

    elif regime == "crisis":
        weights["trend"] = 0.20
        weights["momentum"] = 0.10
        weights["mean_reversion"] = 0.10
        weights["volatility"] = 0.35
        weights["microstructure"] = 0.25

    # Additional vol regime adjustments
    if vol == "crisis":
        for k in weights:
            weights[k] *= 0.5 if k not in ("volatility", "microstructure") else 1.5

    # Normalize
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def regime_risk_budget(snapshot: RegimeSnapshot) -> dict:
    """
    Recommended risk budget (max position size) per regime.
    """
    base_risk = 0.20  # 20% of portfolio per position

    multipliers = {
        ("bull", "low"): 1.5,
        ("bull", "normal"): 1.2,
        ("bull", "high"): 0.8,
        ("bull", "crisis"): 0.4,
        ("bear", "low"): 1.0,
        ("bear", "normal"): 0.9,
        ("bear", "high"): 0.6,
        ("bear", "crisis"): 0.3,
        ("sideways", "low"): 1.0,
        ("sideways", "normal"): 0.8,
        ("sideways", "high"): 0.5,
        ("sideways", "crisis"): 0.2,
        ("crisis", "crisis"): 0.15,
    }

    key = (snapshot.primary_regime, snapshot.vol_regime)
    mult = multipliers.get(key, 0.8)

    # Further scale by regime confidence
    mult *= snapshot.regime_confidence

    return {
        "max_single_position": float(base_risk * mult),
        "max_gross_exposure": float(base_risk * mult * 5),
        "risk_multiplier": float(mult),
        "regime": snapshot.primary_regime,
        "vol_regime": snapshot.vol_regime,
    }
