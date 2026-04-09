"""
Internal Fear/Greed Oscillator: meta-signal from the system's own state.

Most fear/greed indicators use external data (VIX, put/call ratio, etc).
This one is derived PURELY from the internal state of the autonomous system:
  - How aggressive are the current positions? (greed)
  - How many signals are bullish vs bearish? (sentiment)
  - How high is the consciousness consensus? (conviction)
  - How much are the dream scenarios worrying the system? (fear)
  - How fast is the system evolving? (urgency/panic)
  - How many Guardian alerts are active? (stress)

When the system itself is too greedy, it should be contrarian.
When the system itself is too fearful, it should be buying.

This is the system's emotional self-awareness.
"""

from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class FearGreedState:
    """The system's internal emotional state."""
    fear_greed_index: float        # -100 (extreme fear) to +100 (extreme greed)
    label: str                     # "extreme_fear" / "fear" / "neutral" / "greed" / "extreme_greed"
    components: Dict[str, float] = field(default_factory=dict)
    contrarian_signal: float = 0.0  # signal derived from being contrarian to own emotions
    trend: float = 0.0             # is fear/greed increasing or decreasing?
    historical_percentile: float = 0.5


class InternalFearGreedOscillator:
    """
    Measures the autonomous system's own emotional state and generates
    a contrarian signal from it.

    When the system is too confident (greed), reduce position sizes.
    When the system is too scared (fear), increase position sizes.

    This prevents the system from amplifying its own biases.
    """

    def __init__(self, history_window: int = 200):
        self.window = history_window
        self._history: deque = deque(maxlen=history_window)

        # Component trackers
        self._position_aggression: deque = deque(maxlen=50)
        self._signal_sentiment: deque = deque(maxlen=50)
        self._consciousness_conviction: deque = deque(maxlen=50)
        self._dream_worry: deque = deque(maxlen=50)
        self._evolution_speed: deque = deque(maxlen=50)
        self._guardian_stress: deque = deque(maxlen=50)

    def update(
        self,
        position_utilization: float,        # 0-1 how much of max position is used
        n_bullish_signals: int,
        n_bearish_signals: int,
        consciousness_activation: float,    # -1 to +1
        dream_fragility_avg: float,         # 0-1 average fragility across signals
        evolution_mutation_rate: float,      # current mutation rate from RMEA
        n_guardian_alerts: int,
    ) -> FearGreedState:
        """
        Compute the internal fear/greed index from system state.
        """
        # 1. Position Aggression (0-100 greed scale)
        aggression = position_utilization * 100
        self._position_aggression.append(aggression)

        # 2. Signal Sentiment (-100 to +100)
        total_signals = max(n_bullish_signals + n_bearish_signals, 1)
        sentiment = ((n_bullish_signals - n_bearish_signals) / total_signals) * 100
        self._signal_sentiment.append(sentiment)

        # 3. Consciousness Conviction (0-100 greed scale)
        conviction = abs(consciousness_activation) * 100
        self._consciousness_conviction.append(conviction)

        # 4. Dream Worry (0-100 fear scale)
        worry = dream_fragility_avg * 100
        self._dream_worry.append(worry)

        # 5. Evolution Speed (high = panicking, trying to adapt fast)
        evo_urgency = min(100, evolution_mutation_rate * 500)
        self._evolution_speed.append(evo_urgency)

        # 6. Guardian Stress (more alerts = more fear)
        stress = min(100, n_guardian_alerts * 25)
        self._guardian_stress.append(stress)

        # Composite Fear/Greed Index
        # Greed components (positive = greedy)
        greed_score = (
            aggression * 0.25 +
            max(sentiment, 0) * 0.20 +
            conviction * 0.15
        )

        # Fear components (positive = fearful)
        fear_score = (
            worry * 0.20 +
            evo_urgency * 0.10 +
            stress * 0.10 +
            max(-sentiment, 0) * 0.15
        )

        # Net: positive = greed, negative = fear
        fg_index = greed_score - fear_score
        fg_index = max(-100, min(100, fg_index))

        self._history.append(fg_index)

        # Label
        if fg_index > 75:
            label = "extreme_greed"
        elif fg_index > 25:
            label = "greed"
        elif fg_index > -25:
            label = "neutral"
        elif fg_index > -75:
            label = "fear"
        else:
            label = "extreme_fear"

        # Trend
        if len(self._history) >= 10:
            recent = list(self._history)[-10:]
            trend = float(np.polyfit(range(len(recent)), recent, 1)[0])
        else:
            trend = 0.0

        # Historical percentile
        if len(self._history) >= 20:
            hist = np.array(list(self._history))
            percentile = float(np.mean(hist <= fg_index))
        else:
            percentile = 0.5

        # Contrarian signal: fade the system's own emotions
        # Extreme greed -> bearish contrarian
        # Extreme fear -> bullish contrarian
        if abs(fg_index) > 50:
            contrarian = -np.sign(fg_index) * min(abs(fg_index) / 100, 1.0) * 0.5
        else:
            contrarian = 0.0

        components = {
            "position_aggression": float(aggression),
            "signal_sentiment": float(sentiment),
            "consciousness_conviction": float(conviction),
            "dream_worry": float(worry),
            "evolution_urgency": float(evo_urgency),
            "guardian_stress": float(stress),
            "greed_composite": float(greed_score),
            "fear_composite": float(fear_score),
        }

        return FearGreedState(
            fear_greed_index=float(fg_index),
            label=label,
            components=components,
            contrarian_signal=float(contrarian),
            trend=float(trend),
            historical_percentile=float(percentile),
        )

    def get_position_size_multiplier(self, fg_state: FearGreedState) -> float:
        """
        Convert fear/greed into a position size multiplier.

        Extreme greed -> reduce size to 0.5x
        Neutral -> 1.0x
        Extreme fear -> increase size to 1.5x (be greedy when others are fearful)
        """
        fg = fg_state.fear_greed_index
        if fg > 50:
            # Greed: reduce (Warren Buffett: be fearful when others are greedy)
            return max(0.5, 1.0 - (fg - 50) / 100)
        elif fg < -50:
            # Fear: increase (be greedy when others are fearful)
            return min(1.5, 1.0 + abs(fg + 50) / 100)
        else:
            return 1.0

    def get_history(self) -> Dict:
        """Get fear/greed history for charting."""
        hist = list(self._history)
        return {
            "values": hist,
            "current": hist[-1] if hist else 0,
            "mean": float(np.mean(hist)) if hist else 0,
            "std": float(np.std(hist)) if hist else 0,
            "min": float(np.min(hist)) if hist else 0,
            "max": float(np.max(hist)) if hist else 0,
        }
