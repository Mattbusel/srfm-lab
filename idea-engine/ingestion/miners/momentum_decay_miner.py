"""
Momentum decay miner — detects when momentum signal is exhausting.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MomentumDecayMinerConfig:
    lookback_fast: int = 5
    lookback_slow: int = 20
    lookback_baseline: int = 60
    min_momentum: float = 0.03
    jerk_threshold: float = 0.5
    volume_divergence_threshold: float = 0.3


def mine_momentum_decay(
    prices: np.ndarray,
    volumes: np.ndarray,
    config: Optional[MomentumDecayMinerConfig] = None,
) -> list[dict]:
    """
    Mine signals indicating momentum exhaustion / reversal setup.
    """
    if config is None:
        config = MomentumDecayMinerConfig()

    T = min(len(prices), len(volumes))
    if T < config.lookback_baseline + 5:
        return []

    findings = []
    returns = np.diff(np.log(prices[:T]))

    # 1. Price jerk reversal
    jerk = _compute_jerk(prices[:T])
    current_jerk = float(jerk[-1]) if len(jerk) > 0 else 0.0
    fast_mom = float(np.sum(returns[-config.lookback_fast:]))
    slow_mom = float(np.sum(returns[-config.lookback_slow:]))

    if abs(fast_mom) > config.min_momentum and abs(current_jerk) > config.jerk_threshold:
        # Price reversing direction (jerk opposes momentum)
        if current_jerk * fast_mom < 0:
            findings.append({
                "type": "momentum_jerk_reversal",
                "score": float(min(abs(current_jerk), 1.0)),
                "momentum": fast_mom,
                "jerk": current_jerk,
                "action": "fade_momentum",
                "template": "momentum_exhaustion",
                "confidence": float(min(abs(current_jerk) * abs(fast_mom) * 10, 0.85)),
                "description": f"Price jerk reversal: momentum={fast_mom:.3f}, jerk={current_jerk:.3f}",
            })

    # 2. Volume-price divergence
    price_move_5d = float(np.sum(returns[-5:]))
    vol_trend = _linear_trend(volumes[T - 10: T])
    if abs(price_move_5d) > config.min_momentum:
        # Price moving up but volume declining (or vice versa)
        if price_move_5d > 0 and vol_trend < -config.volume_divergence_threshold:
            findings.append({
                "type": "vol_price_divergence",
                "score": float(min(abs(vol_trend), 1.0)),
                "price_move": price_move_5d,
                "volume_trend": vol_trend,
                "action": "fade_uptrend",
                "template": "momentum_exhaustion",
                "confidence": 0.6,
                "description": f"Bearish vol-price divergence: price +{price_move_5d:.2%}, vol trend {vol_trend:.2f}",
            })
        elif price_move_5d < 0 and vol_trend < -config.volume_divergence_threshold:
            findings.append({
                "type": "vol_price_divergence",
                "score": float(min(abs(vol_trend), 1.0)),
                "price_move": price_move_5d,
                "volume_trend": vol_trend,
                "action": "fade_downtrend",
                "template": "momentum_exhaustion",
                "confidence": 0.55,
                "description": f"Bearish vol-price divergence (low conviction selling): price {price_move_5d:.2%}, vol declining",
            })

    # 3. Momentum crossover reversal (fast < slow, was fast > slow)
    if len(returns) >= config.lookback_slow + 2:
        prev_fast = float(np.sum(returns[-config.lookback_fast - 1: -1]))
        if prev_fast > 0 and fast_mom < 0 and slow_mom > 0:
            findings.append({
                "type": "momentum_crossover_bearish",
                "score": abs(fast_mom - prev_fast),
                "action": "fade_momentum",
                "template": "momentum_exhaustion",
                "confidence": 0.50,
                "description": "Fast momentum crossed negative while slow still positive",
            })
        elif prev_fast < 0 and fast_mom > 0 and slow_mom < 0:
            findings.append({
                "type": "momentum_crossover_bullish",
                "score": abs(fast_mom - prev_fast),
                "action": "enter_long_reversal",
                "template": "momentum_exhaustion",
                "confidence": 0.50,
                "description": "Fast momentum crossed positive while slow still negative",
            })

    # 4. Higher timeframe divergence
    if len(returns) >= config.lookback_baseline:
        baseline_mom = float(np.sum(returns[-config.lookback_baseline:]))
        if abs(slow_mom) > config.min_momentum and baseline_mom * slow_mom < 0:
            findings.append({
                "type": "htf_divergence",
                "score": float(min(abs(slow_mom), 1.0)),
                "slow_mom": slow_mom,
                "baseline_mom": baseline_mom,
                "action": "fade_short_term_momentum",
                "template": "cross_timeframe_divergence",
                "confidence": 0.55,
                "description": f"HTF momentum divergence: short={slow_mom:.3f}, baseline={baseline_mom:.3f}",
            })

    return sorted(findings, key=lambda x: x["confidence"], reverse=True)


def _compute_jerk(prices: np.ndarray, smooth: int = 3) -> np.ndarray:
    """Price jerk = rate of change of acceleration = 3rd derivative."""
    if len(prices) < smooth + 3:
        return np.zeros(max(len(prices) - 3, 1))
    v = np.diff(prices)
    a = np.diff(v)
    j = np.diff(a)
    return j


def _linear_trend(x: np.ndarray) -> float:
    """Normalized linear trend slope."""
    n = len(x)
    if n < 2:
        return 0.0
    t = np.arange(n)
    slope = np.polyfit(t, x, 1)[0]
    return float(slope / (x.mean() + 1e-10))
