"""
Liquidation cascade miner — discovers liquidation-driven opportunities.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LiquidationMinerConfig:
    cascade_risk_threshold: float = 0.5
    squeeze_threshold: float = 0.4
    depletion_threshold: float = 0.25
    reflexivity_threshold: float = 0.2
    window: int = 20


def mine_liquidation_signals(
    prices: np.ndarray,
    open_interest: np.ndarray,
    volumes: np.ndarray,
    funding_rate: Optional[np.ndarray] = None,
    config: Optional[LiquidationMinerConfig] = None,
) -> list[dict]:
    """
    Mine liquidation-driven hypothesis candidates from market data.
    """
    from lib.signals.liquidation_cascade import (
        cascade_risk_index,
        squeeze_signal,
        oi_depletion_signal,
        reflexivity_detector,
    )

    if config is None:
        config = LiquidationMinerConfig()

    findings = []
    T = min(len(prices), len(open_interest), len(volumes))

    # 1. Cascade Risk Index
    risk = cascade_risk_index(prices[:T], volumes[:T], open_interest[:T], funding_rate, config.window)
    current_risk = float(risk[-1])

    if current_risk > config.cascade_risk_threshold:
        direction = "down" if (funding_rate is not None and float(funding_rate[-1]) > 0) else "down"
        findings.append({
            "type": "cascade_risk",
            "score": current_risk,
            "action": "prepare_short" if direction == "down" else "prepare_long",
            "template": "cascade_front_run",
            "confidence": float(current_risk),
            "description": f"Cascade risk index {current_risk:.2f} above threshold {config.cascade_risk_threshold}",
        })

    # 2. Squeeze Detection
    sq = squeeze_signal(prices[:T], open_interest[:T], funding_rate, volumes[:T], config.window)
    if sq["short_squeeze_alert"]:
        findings.append({
            "type": "short_squeeze",
            "score": sq["current_short_squeeze"],
            "action": "enter_long",
            "template": "short_squeeze_setup",
            "confidence": float(sq["current_short_squeeze"]),
            "description": f"Short squeeze conditions detected: score={sq['current_short_squeeze']:.2f}",
        })
    if sq["long_squeeze_alert"]:
        findings.append({
            "type": "long_squeeze",
            "score": sq["current_long_squeeze"],
            "action": "enter_short",
            "template": "long_squeeze_setup",
            "confidence": float(sq["current_long_squeeze"]),
            "description": f"Long squeeze conditions detected: score={sq['current_long_squeeze']:.2f}",
        })

    # 3. OI Depletion (post-cascade recovery)
    depl = oi_depletion_signal(open_interest[:T], prices[:T], config.window)
    if depl["post_cascade_recovery_signal"]:
        findings.append({
            "type": "post_cascade_recovery",
            "score": depl["recent_max_depletion"],
            "action": "enter_long",
            "template": "post_cascade_recovery",
            "confidence": 0.6,
            "description": "Post-cascade recovery conditions: OI depleted, vol declining",
        })

    # 4. Reflexivity
    liq_proxy = np.abs(np.diff(open_interest[:T]))
    if len(liq_proxy) > config.window:
        ref = reflexivity_detector(prices[:T], liq_proxy)
        if ref["is_reflexive"]:
            findings.append({
                "type": "reflexivity_loop",
                "score": ref["reflexivity_score"],
                "action": "trend_follow",
                "template": "reflexivity_amplifier",
                "confidence": float(min(ref["reflexivity_score"] * 2, 0.9)),
                "description": f"Reflexivity loop detected: score={ref['reflexivity_score']:.2f}",
            })

    return sorted(findings, key=lambda x: x["score"], reverse=True)
