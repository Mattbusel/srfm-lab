"""
Regime Transition Predictor: predict the NEXT regime before it arrives.

Most regime detectors tell you what regime you're IN. This module
predicts which regime is COMING by detecting early warning signs:

1. Correlation structure changes (topology shift precedes price moves)
2. Volatility term structure inversion (front vol > back vol = stress coming)
3. Options skew steepening (put demand increasing = fear building)
4. Breadth divergence (leaders turning while laggards still rising)
5. Entropy acceleration (information rate spiking before regime change)
6. Markov transition probability from the regime ensemble

The signal: position for the NEXT regime, not the current one.
This gives you a head start over regime-following strategies.
"""

from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class TransitionForecast:
    """Prediction of an upcoming regime transition."""
    current_regime: str
    predicted_next_regime: str
    transition_probability: float   # 0-1 probability of transition
    estimated_bars_until: int       # how many bars until transition
    confidence: float
    early_warning_signals: List[str]
    recommended_positioning: str    # e.g., "reduce_equity_increase_vol"


@dataclass
class EarlyWarning:
    """A single early warning indicator."""
    name: str
    value: float
    threshold: float
    triggered: bool
    description: str


class RegimeTransitionPredictor:
    """
    Predict regime transitions before they happen.
    """

    def __init__(self, lookback: int = 63):
        self.lookback = lookback
        self._correlation_history = deque(maxlen=lookback)
        self._vol_term_history = deque(maxlen=lookback)
        self._skew_history = deque(maxlen=lookback)
        self._breadth_history = deque(maxlen=lookback)
        self._entropy_history = deque(maxlen=lookback)

        # Transition matrix (learned from data)
        self._regime_sequence = deque(maxlen=500)
        self._transition_counts: Dict[tuple, int] = {}

    def update(
        self,
        current_regime: str,
        avg_correlation: float,
        front_vol: float,           # near-term implied vol
        back_vol: float,            # longer-term implied vol
        put_call_skew: float,       # put IV - call IV at 25 delta
        breadth_pct: float,         # % of assets above their 20-day MA
        return_entropy: float,      # from information surprise detector
    ) -> TransitionForecast:
        """Update with current market data and predict the next regime."""
        self._correlation_history.append(avg_correlation)
        self._vol_term_history.append(front_vol - back_vol)
        self._skew_history.append(put_call_skew)
        self._breadth_history.append(breadth_pct)
        self._entropy_history.append(return_entropy)

        # Update transition matrix
        self._regime_sequence.append(current_regime)
        if len(self._regime_sequence) >= 2:
            prev = self._regime_sequence[-2]
            curr = current_regime
            key = (prev, curr)
            self._transition_counts[key] = self._transition_counts.get(key, 0) + 1

        # Compute early warning indicators
        warnings = self._compute_warnings()

        # Predict next regime
        n_triggered = sum(1 for w in warnings if w.triggered)

        # Markov prediction
        markov_probs = self._markov_predict(current_regime)

        # Determine most likely next regime
        if n_triggered >= 3:
            # Multiple warnings = transition likely
            if any(w.name == "vol_term_inversion" and w.triggered for w in warnings):
                predicted = "high_volatility"
            elif any(w.name == "breadth_divergence" and w.triggered for w in warnings):
                predicted = "trending_down"
            elif any(w.name == "entropy_spike" and w.triggered for w in warnings):
                predicted = "regime_change"
            else:
                predicted = max(markov_probs, key=markov_probs.get) if markov_probs else current_regime
        else:
            predicted = max(markov_probs, key=markov_probs.get) if markov_probs else current_regime

        # Transition probability
        trans_prob = markov_probs.get(predicted, 0) if predicted != current_regime else 0
        trans_prob = max(trans_prob, n_triggered / 6)  # boost by warning count

        # Estimated bars until transition (from historical data)
        avg_regime_duration = self._avg_regime_duration(current_regime)
        current_duration = self._current_regime_duration()
        bars_until = max(0, int(avg_regime_duration - current_duration))

        # Recommended positioning
        positioning = self._recommend_positioning(current_regime, predicted, trans_prob)

        warning_names = [w.name for w in warnings if w.triggered]

        return TransitionForecast(
            current_regime=current_regime,
            predicted_next_regime=predicted,
            transition_probability=float(min(1.0, trans_prob)),
            estimated_bars_until=bars_until,
            confidence=float(min(1.0, n_triggered / 4)),
            early_warning_signals=warning_names,
            recommended_positioning=positioning,
        )

    def _compute_warnings(self) -> List[EarlyWarning]:
        """Compute all early warning indicators."""
        warnings = []

        # 1. Correlation acceleration
        if len(self._correlation_history) >= 10:
            corrs = np.array(list(self._correlation_history)[-10:])
            accel = float(np.polyfit(range(len(corrs)), corrs, 1)[0])
            warnings.append(EarlyWarning(
                name="correlation_acceleration",
                value=accel,
                threshold=0.01,
                triggered=accel > 0.01,
                description=f"Correlation rising at {accel:.4f}/bar (herding building)",
            ))

        # 2. Vol term structure inversion
        if len(self._vol_term_history) >= 5:
            recent = np.array(list(self._vol_term_history)[-5:])
            avg_inversion = float(recent.mean())
            warnings.append(EarlyWarning(
                name="vol_term_inversion",
                value=avg_inversion,
                threshold=0.0,
                triggered=avg_inversion > 0,  # front > back = inverted
                description=f"Vol term structure {'inverted' if avg_inversion > 0 else 'normal'} ({avg_inversion:.4f})",
            ))

        # 3. Skew steepening
        if len(self._skew_history) >= 10:
            skews = np.array(list(self._skew_history))
            skew_trend = float(np.polyfit(range(len(skews[-10:])), skews[-10:], 1)[0])
            warnings.append(EarlyWarning(
                name="skew_steepening",
                value=skew_trend,
                threshold=0.005,
                triggered=skew_trend > 0.005,
                description=f"Put skew steepening at {skew_trend:.4f}/bar (fear building)",
            ))

        # 4. Breadth divergence
        if len(self._breadth_history) >= 10:
            breadths = np.array(list(self._breadth_history)[-10:])
            breadth_trend = float(np.polyfit(range(len(breadths)), breadths, 1)[0])
            warnings.append(EarlyWarning(
                name="breadth_divergence",
                value=breadth_trend,
                threshold=-0.02,
                triggered=breadth_trend < -0.02,
                description=f"Breadth declining at {breadth_trend:.4f}/bar (leaders turning)",
            ))

        # 5. Entropy spike
        if len(self._entropy_history) >= 10:
            entropies = np.array(list(self._entropy_history)[-10:])
            entropy_accel = float(np.polyfit(range(len(entropies)), entropies, 1)[0])
            warnings.append(EarlyWarning(
                name="entropy_spike",
                value=entropy_accel,
                threshold=0.01,
                triggered=entropy_accel > 0.01,
                description=f"Information rate accelerating ({entropy_accel:.4f}/bar)",
            ))

        # 6. Regime duration exceeding average
        avg_dur = self._avg_regime_duration(list(self._regime_sequence)[-1] if self._regime_sequence else "")
        current_dur = self._current_regime_duration()
        warnings.append(EarlyWarning(
            name="regime_overstay",
            value=float(current_dur),
            threshold=float(avg_dur),
            triggered=current_dur > avg_dur * 1.5,
            description=f"Current regime has lasted {current_dur} bars (avg: {avg_dur:.0f})",
        ))

        return warnings

    def _markov_predict(self, current_regime: str) -> Dict[str, float]:
        """Predict next regime from transition matrix."""
        transitions_from = {k: v for k, v in self._transition_counts.items() if k[0] == current_regime}
        total = sum(transitions_from.values())
        if total == 0:
            return {}
        return {k[1]: v / total for k, v in transitions_from.items()}

    def _avg_regime_duration(self, regime: str) -> float:
        """Average duration of a regime in bars."""
        if not self._regime_sequence:
            return 50.0
        durations = []
        count = 0
        for r in self._regime_sequence:
            if r == regime:
                count += 1
            elif count > 0:
                durations.append(count)
                count = 0
        if count > 0:
            durations.append(count)
        return float(np.mean(durations)) if durations else 50.0

    def _current_regime_duration(self) -> int:
        """How long has the current regime lasted?"""
        if not self._regime_sequence:
            return 0
        current = self._regime_sequence[-1]
        count = 0
        for r in reversed(self._regime_sequence):
            if r == current:
                count += 1
            else:
                break
        return count

    def _recommend_positioning(self, current: str, predicted: str, prob: float) -> str:
        """Recommend portfolio adjustments for the predicted transition."""
        if prob < 0.3:
            return "maintain_current_positioning"

        transitions = {
            ("trending_up", "high_volatility"): "reduce_equity_buy_vol_protection",
            ("trending_up", "mean_reverting"): "take_profits_increase_reversion_allocation",
            ("trending_up", "trending_down"): "hedge_with_puts_reduce_long_exposure",
            ("high_volatility", "trending_down"): "maximize_hedges_go_defensive",
            ("high_volatility", "mean_reverting"): "sell_vol_buy_dips",
            ("mean_reverting", "trending_up"): "increase_momentum_allocation",
            ("mean_reverting", "high_volatility"): "buy_straddles_reduce_size",
            ("trending_down", "high_volatility"): "flatten_wait_for_clarity",
            ("trending_down", "mean_reverting"): "start_accumulating_reduce_shorts",
        }

        key = (current, predicted)
        return transitions.get(key, f"prepare_for_{predicted}")
