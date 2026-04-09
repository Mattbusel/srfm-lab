"""
Unified Alpha Signal: the single most important signal in the system.

Combines ALL available alpha sources into one master signal using
dynamic, regime-adaptive weighting. This is the signal that the
Portfolio Brain uses as its primary input.

Sources (with adaptive weights):
  - BH Physics (mass, Hawking temp, geodesic deviation)
  - Fractal timeframe coherence
  - Information surprise (entropy-based pre-move detection)
  - Liquidity black hole (cascade prevention)
  - Whale activity (smart money tracking)
  - Market consciousness (emergent beliefs)
  - Correlation regime (herding/dispersion transitions)
  - Cross-exchange flow (smart money between venues)
  - Regime transition prediction (next regime positioning)
  - Fear/greed contrarian (fade the system's own emotions)
  - Market memory (gravitational levels)

The weights adapt in real-time based on:
  - Rolling IC per source (best predictors get more weight)
  - Regime-conditional performance (some sources work better in some regimes)
  - Dream fragility (fragile sources get reduced weight)
  - Groupthink dampening (if all sources agree, reduce confidence)
"""

from __future__ import annotations
import math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class AlphaSourceInput:
    """Input from a single alpha source."""
    name: str
    value: float              # -1 to +1 signal value
    confidence: float         # 0-1 how confident this source is
    rolling_ic: float         # recent information coefficient
    dream_fragility: float    # 0-1 how fragile in dream scenarios
    regime_affinity: float    # 0-1 how well this source works in current regime


@dataclass
class UnifiedSignalOutput:
    """Output of the unified alpha signal."""
    signal: float             # -1 to +1 final composite
    confidence: float         # 0-1 overall confidence
    n_sources_contributing: int
    dominant_source: str      # which source is driving
    source_agreement: float   # 0-1 how much sources agree
    groupthink_dampened: bool
    regime_adapted: bool
    narrative: str

    # Full source breakdown
    source_weights: Dict[str, float] = field(default_factory=dict)
    source_signals: Dict[str, float] = field(default_factory=dict)
    source_contributions: Dict[str, float] = field(default_factory=dict)


class UnifiedAlphaSignal:
    """
    The master signal: combines all alpha sources with adaptive weighting.
    """

    # Base weights (before adaptation)
    BASE_WEIGHTS = {
        "bh_physics": 0.15,
        "fractal": 0.12,
        "info_surprise": 0.10,
        "liquidity_blackhole": 0.08,
        "whale_activity": 0.10,
        "consciousness": 0.08,
        "correlation_regime": 0.08,
        "exchange_flow": 0.07,
        "regime_transition": 0.07,
        "fear_greed_contrarian": 0.05,
        "market_memory": 0.05,
        "swarm_consensus": 0.05,
    }

    def __init__(self, ic_lookback: int = 63):
        self.ic_lookback = ic_lookback
        self._ic_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=ic_lookback))
        self._return_history = deque(maxlen=ic_lookback)

    def record_return(self, actual_return: float) -> None:
        """Record the actual market return (for IC computation)."""
        self._return_history.append(actual_return)

    def compute(
        self,
        sources: List[AlphaSourceInput],
        current_regime: str = "unknown",
        groupthink_score: float = 0.0,  # from GroupthinkDetector
        fear_greed_multiplier: float = 1.0,  # from FearGreedOscillator
    ) -> UnifiedSignalOutput:
        """
        Compute the unified alpha signal from all sources.
        """
        if not sources:
            return UnifiedSignalOutput(0.0, 0.0, 0, "", 0.0, False, False, "No sources")

        # Step 1: Compute adaptive weights
        weights = {}
        for source in sources:
            base_w = self.BASE_WEIGHTS.get(source.name, 0.05)

            # IC-based adaptation: boost sources with high recent IC
            ic_boost = 1.0 + max(0, source.rolling_ic * 10)

            # Regime adaptation: boost sources that work in this regime
            regime_boost = 0.5 + source.regime_affinity

            # Dream fragility: reduce fragile sources
            fragility_penalty = max(0.2, 1 - source.dream_fragility)

            # Confidence scaling
            conf_scale = 0.3 + 0.7 * source.confidence

            weights[source.name] = base_w * ic_boost * regime_boost * fragility_penalty * conf_scale

        # Normalize weights
        total_w = sum(weights.values())
        if total_w > 0:
            weights = {k: v / total_w for k, v in weights.items()}

        # Step 2: Compute weighted signal
        signal = 0.0
        contributions = {}
        for source in sources:
            w = weights.get(source.name, 0)
            contrib = source.value * w
            signal += contrib
            contributions[source.name] = contrib

            # Update IC tracking
            self._ic_history[source.name].append(source.value)

        # Step 3: Groupthink dampening
        groupthink_dampened = False
        if groupthink_score > 0.7:
            signal *= (1 - groupthink_score * 0.5)
            groupthink_dampened = True

        # Step 4: Fear/greed contrarian
        signal *= fear_greed_multiplier

        signal = float(np.clip(signal, -1, 1))

        # Step 5: Compute confidence
        source_values = [s.value for s in sources if abs(s.value) > 0.05]
        if source_values:
            # Agreement: fraction of sources that agree on direction
            signs = [1 if v > 0 else -1 for v in source_values]
            majority = 1 if sum(signs) > 0 else -1
            agreement = sum(1 for s in signs if s == majority) / len(signs)

            # Confidence: agreement * average confidence * (1 - groupthink)
            avg_conf = float(np.mean([s.confidence for s in sources]))
            confidence = agreement * avg_conf * (1 - groupthink_score * 0.3)
        else:
            agreement = 0.0
            confidence = 0.0

        # Dominant source
        dominant = max(contributions, key=lambda k: abs(contributions[k])) if contributions else ""

        # Narrative
        top_3 = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
        parts = [f"{'Long' if signal > 0 else 'Short' if signal < 0 else 'Flat'} (signal={signal:+.3f})"]
        for name, contrib in top_3:
            if abs(contrib) > 0.01:
                parts.append(f"{name}: {contrib:+.3f}")
        if groupthink_dampened:
            parts.append("(groupthink dampened)")
        narrative = " | ".join(parts)

        return UnifiedSignalOutput(
            signal=signal,
            confidence=float(confidence),
            n_sources_contributing=len([s for s in sources if abs(s.value) > 0.05]),
            dominant_source=dominant,
            source_agreement=float(agreement),
            groupthink_dampened=groupthink_dampened,
            regime_adapted=True,
            narrative=narrative,
            source_weights=weights,
            source_signals={s.name: s.value for s in sources},
            source_contributions=contributions,
        )
