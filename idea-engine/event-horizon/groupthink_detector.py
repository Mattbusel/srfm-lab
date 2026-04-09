"""
Groupthink Detector: prevents the autonomous system from becoming too consensus-driven.

When all signals agree, all debate agents agree, and the system has high conviction,
it's often WRONG -- because markets punish consensus. This module:

1. Measures internal consensus level across all subsystems
2. Detects when consensus is dangerously high (groupthink threshold)
3. Automatically injects contrarian noise to break the echo chamber
4. Forces the Devil's Advocate agent to argue harder
5. Activates the Red Queen engine to stress-test the current view

The Warren Buffett module: "Be fearful when others are greedy,
be greedy when others are fearful" -- applied to the system's own agents.
"""

from __future__ import annotations
import math
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class GroupthinkAlert:
    """Alert that the system is becoming too consensus-driven."""
    alert_level: str              # "mild" / "moderate" / "severe" / "critical"
    consensus_score: float        # 0-1 how aligned everything is
    duration_bars: int            # how long consensus has persisted
    affected_subsystems: List[str]
    recommended_action: str
    noise_injection_amount: float  # how much randomness to inject


class GroupthinkDetector:
    """
    Detects and disrupts internal consensus to prevent systematic bias.

    Monitors agreement levels across:
    - Debate agents (are they all voting the same way?)
    - Signal sources (are all signals pointing the same direction?)
    - Timeframes (does every timeframe agree?)
    - Regimes (do all regime detectors agree?)

    When consensus exceeds threshold, injects disruption:
    - Random noise into signal weights
    - Forces Devil's Advocate to argue harder
    - Increases Red Queen adversarial pressure
    - Reduces position sizes (uncertainty premium)
    """

    def __init__(self, mild_threshold: float = 0.7, severe_threshold: float = 0.9,
                  max_consensus_bars: int = 20):
        self.mild_threshold = mild_threshold
        self.severe_threshold = severe_threshold
        self.max_consensus_bars = max_consensus_bars
        self.rng = random.Random(42)

        self._consensus_history: deque = deque(maxlen=200)
        self._consecutive_consensus: int = 0
        self._disruptions_applied: int = 0

    def check(
        self,
        debate_votes: List[float],          # per-agent conviction (-1 to +1)
        signal_values: List[float],          # per-signal values (-1 to +1)
        timeframe_signals: List[float],      # per-timeframe values
        regime_agreements: List[bool],       # per-detector agreement with majority
    ) -> Optional[GroupthinkAlert]:
        """
        Check for groupthink across all subsystems.
        Returns an alert if consensus is dangerously high, else None.
        """
        # Debate consensus
        if debate_votes:
            debate_agreement = float(np.mean([1 if v > 0 else 0 for v in debate_votes if abs(v) > 0.1]))
            debate_consensus = max(debate_agreement, 1 - debate_agreement)
        else:
            debate_consensus = 0.5

        # Signal consensus
        if signal_values:
            signal_agreement = float(np.mean([1 if v > 0 else 0 for v in signal_values if abs(v) > 0.1]))
            signal_consensus = max(signal_agreement, 1 - signal_agreement)
        else:
            signal_consensus = 0.5

        # Timeframe consensus
        if timeframe_signals:
            tf_agreement = float(np.mean([1 if v > 0 else 0 for v in timeframe_signals if abs(v) > 0.1]))
            tf_consensus = max(tf_agreement, 1 - tf_agreement)
        else:
            tf_consensus = 0.5

        # Regime consensus
        if regime_agreements:
            regime_consensus = float(np.mean(regime_agreements))
        else:
            regime_consensus = 0.5

        # Overall consensus: weighted average
        overall = (
            debate_consensus * 0.30 +
            signal_consensus * 0.30 +
            tf_consensus * 0.20 +
            regime_consensus * 0.20
        )

        self._consensus_history.append(overall)

        # Track consecutive high consensus
        if overall > self.mild_threshold:
            self._consecutive_consensus += 1
        else:
            self._consecutive_consensus = 0

        # Determine alert level
        affected = []
        if debate_consensus > self.mild_threshold:
            affected.append("debate")
        if signal_consensus > self.mild_threshold:
            affected.append("signals")
        if tf_consensus > self.mild_threshold:
            affected.append("timeframes")
        if regime_consensus > self.mild_threshold:
            affected.append("regimes")

        if overall < self.mild_threshold:
            return None

        if overall > self.severe_threshold or self._consecutive_consensus > self.max_consensus_bars:
            level = "critical" if self._consecutive_consensus > self.max_consensus_bars * 1.5 else "severe"
            noise = 0.3
            action = "Inject significant noise + force Devil's Advocate + reduce positions 50%"
        elif overall > 0.8:
            level = "moderate"
            noise = 0.15
            action = "Inject moderate noise + increase Red Queen pressure"
        else:
            level = "mild"
            noise = 0.05
            action = "Mild noise injection + log warning"

        return GroupthinkAlert(
            alert_level=level,
            consensus_score=overall,
            duration_bars=self._consecutive_consensus,
            affected_subsystems=affected,
            recommended_action=action,
            noise_injection_amount=noise,
        )

    def inject_noise(self, signal_weights: Dict[str, float],
                      noise_amount: float) -> Dict[str, float]:
        """
        Inject random noise into signal weights to break echo chamber.
        Returns modified weights.
        """
        self._disruptions_applied += 1
        noisy = {}
        for name, weight in signal_weights.items():
            noise = self.rng.gauss(0, noise_amount)
            noisy[name] = float(np.clip(weight + noise, -1, 1))
        return noisy

    def get_contrarian_boost(self, current_direction: float,
                              consensus_score: float) -> float:
        """
        How much should the Devil's Advocate be boosted?
        Higher consensus -> more contrarian pressure.
        """
        if consensus_score < self.mild_threshold:
            return 0.0
        # Scale: at mild_threshold boost = 0.5x, at 1.0 boost = 3.0x
        boost = 0.5 + (consensus_score - self.mild_threshold) / (1 - self.mild_threshold) * 2.5
        return float(boost)

    def get_position_multiplier(self, consensus_score: float) -> float:
        """
        Reduce position sizes when consensus is too high.
        The uncertainty premium: high internal agreement = higher chance of being wrong.
        """
        if consensus_score < self.mild_threshold:
            return 1.0
        # Scale down: 1.0 at threshold, 0.5 at maximum consensus
        return float(max(0.5, 1.0 - (consensus_score - self.mild_threshold) * 1.5))

    def get_status(self) -> Dict:
        hist = list(self._consensus_history)
        return {
            "current_consensus": float(hist[-1]) if hist else 0.5,
            "consecutive_high_bars": self._consecutive_consensus,
            "disruptions_applied": self._disruptions_applied,
            "avg_consensus_50bar": float(np.mean(hist[-50:])) if len(hist) >= 50 else 0.5,
            "max_consensus_observed": float(max(hist)) if hist else 0.5,
        }
