"""
Correlation Regime Signal: trade the STRUCTURE of correlations, not just prices.

Most signals look at individual asset prices. This signal looks at how
assets RELATE to each other and trades changes in those relationships.

Key insight: when correlations break down, diversification fails and
risk management fails. When correlations spike, everything moves together
and you need to position for the trend. When correlations are normal,
mean reversion strategies work best.

Three correlation regimes:
  1. DISPERSED (low corr < 0.2): assets are independent, mean reversion works
  2. NORMAL (0.2-0.5): standard diversification, momentum moderately works
  3. HERDING (>0.5): everything moves together, momentum is dominant

The alpha: trade the TRANSITIONS between correlation regimes.
Detecting when correlations are about to spike (before a crash) or
about to collapse (before a recovery) is worth real money.
"""

from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class CorrelationRegimeState:
    """Current state of the correlation regime."""
    regime: str                   # "dispersed" / "normal" / "herding"
    avg_correlation: float
    correlation_velocity: float   # rate of change of average correlation
    absorption_ratio: float       # top eigenvalue / total variance
    dispersion: float             # cross-sectional return dispersion
    herding_signal: float         # how much herding is occurring
    transition_signal: str        # "stable" / "converging" / "diverging"
    trading_signal: float         # -1 to +1


class CorrelationRegimeSignal:
    """
    Trade correlation regime transitions.
    """

    def __init__(self, n_assets: int, window_fast: int = 21, window_slow: int = 63):
        self.n_assets = n_assets
        self.fast = window_fast
        self.slow = window_slow
        self._returns_buffer: List[deque] = [deque(maxlen=window_slow * 2) for _ in range(n_assets)]
        self._corr_history = deque(maxlen=200)

    def update(self, returns: np.ndarray) -> CorrelationRegimeState:
        """
        Update with new bar returns for all assets.
        returns: (n_assets,) array of returns for this bar.
        """
        for i in range(min(self.n_assets, len(returns))):
            self._returns_buffer[i].append(returns[i])

        # Check we have enough data
        min_len = min(len(buf) for buf in self._returns_buffer)
        if min_len < self.fast + 5:
            return CorrelationRegimeState("normal", 0.3, 0.0, 0.3, 0.02, 0.0, "stable", 0.0)

        # Build return matrix
        R_fast = np.array([list(buf)[-self.fast:] for buf in self._returns_buffer])
        R_slow = np.array([list(buf)[-min(min_len, self.slow):] for buf in self._returns_buffer])

        # Correlation matrix (fast window)
        corr_fast = np.corrcoef(R_fast)
        upper_fast = corr_fast[np.triu_indices(self.n_assets, k=1)]
        avg_corr_fast = float(upper_fast.mean()) if len(upper_fast) > 0 else 0.3

        # Correlation matrix (slow window)
        corr_slow = np.corrcoef(R_slow)
        upper_slow = corr_slow[np.triu_indices(self.n_assets, k=1)]
        avg_corr_slow = float(upper_slow.mean()) if len(upper_slow) > 0 else 0.3

        self._corr_history.append(avg_corr_fast)

        # Correlation velocity
        if len(self._corr_history) >= 5:
            recent = list(self._corr_history)[-5:]
            velocity = float(np.polyfit(range(len(recent)), recent, 1)[0])
        else:
            velocity = 0.0

        # Absorption ratio (eigenvalue concentration)
        try:
            eigvals = np.linalg.eigvalsh(corr_fast + np.eye(self.n_assets) * 0.01)
            eigvals = eigvals[eigvals > 0]
            absorption = float(eigvals[-1] / eigvals.sum()) if len(eigvals) > 0 else 0.3
        except:
            absorption = 0.3

        # Cross-sectional dispersion
        cross_returns = R_fast[:, -1]  # latest bar
        dispersion = float(np.std(cross_returns))

        # Herding score: absorption ratio * average correlation
        herding = absorption * avg_corr_fast

        # Regime classification
        if avg_corr_fast > 0.5 or absorption > 0.6:
            regime = "herding"
        elif avg_corr_fast < 0.2:
            regime = "dispersed"
        else:
            regime = "normal"

        # Transition signal
        if velocity > 0.005:
            transition = "converging"  # correlations rising -> heading toward herding
        elif velocity < -0.005:
            transition = "diverging"   # correlations falling -> heading toward dispersion
        else:
            transition = "stable"

        # Trading signal
        signal = 0.0

        # In herding: momentum works, go with the trend
        if regime == "herding":
            avg_return = float(np.mean(cross_returns))
            signal = float(np.tanh(avg_return * 50))

        # In dispersed: mean reversion works, fade extremes
        elif regime == "dispersed":
            avg_return = float(np.mean(cross_returns))
            signal = float(-np.tanh(avg_return * 30))  # contrarian

        # Transition bonus: converging toward herding = get long momentum early
        if transition == "converging" and regime != "herding":
            signal += 0.2 * float(np.sign(np.mean(cross_returns)))

        # Transition bonus: diverging from herding = mean reversion about to work
        if transition == "diverging" and regime == "herding":
            signal -= 0.2 * float(np.sign(np.mean(cross_returns)))

        signal = float(np.clip(signal, -1, 1))

        return CorrelationRegimeState(
            regime=regime,
            avg_correlation=avg_corr_fast,
            correlation_velocity=velocity,
            absorption_ratio=absorption,
            dispersion=dispersion,
            herding_signal=herding,
            transition_signal=transition,
            trading_signal=signal,
        )
