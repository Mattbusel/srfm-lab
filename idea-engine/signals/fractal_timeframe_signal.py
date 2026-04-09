"""
Multi-Timeframe Fractal Signal: detect self-similar patterns across scales.

Markets are fractal: the same patterns repeat at 1-min, 15-min, 1-hour, 4-hour,
and daily scales. When all scales AGREE on direction, the signal is strongest.
When scales DIVERGE, a reversal is likely.

Components:
  1. Wavelet decomposition: extract signal at each scale without phase distortion
  2. Hurst exponent per scale: measure trending vs mean-reverting at each timeframe
  3. Scale coherence: how aligned are the signals across timeframes
  4. Fractal dimension: D close to 1 = smooth trend, D close to 2 = noise
  5. Multi-scale momentum: momentum computed independently at each scale, then fused

The final signal is a coherence-weighted combination of per-scale signals.
High coherence across scales = high conviction. Low coherence = reduce size.
"""

from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ScaleAnalysis:
    """Analysis at a single timescale."""
    scale_name: str          # "1min", "15min", "1h", "4h", "daily"
    scale_bars: int          # number of base bars in this scale
    hurst: float             # Hurst exponent at this scale
    momentum: float          # normalized momentum signal
    trend_strength: float    # R-squared of linear regression
    volatility: float        # realized vol at this scale
    regime: str              # "trending" / "mean_reverting" / "random"


@dataclass
class FractalSignal:
    """Output of the fractal analysis."""
    composite_signal: float       # -1 to +1 final signal
    coherence: float              # 0-1 how aligned scales are
    fractal_dimension: float      # ~1.0 = smooth, ~2.0 = rough/noisy
    dominant_scale: str           # which timeframe is driving
    scale_analyses: List[ScaleAnalysis] = field(default_factory=list)
    divergence_detected: bool = False
    divergence_details: str = ""


class FractalTimeframeDetector:
    """
    Multi-timeframe fractal signal detector.

    Maintains rolling buffers at multiple scales and computes
    fractal metrics across all scales simultaneously.
    """

    SCALES = {
        "1min": 1,
        "15min": 15,
        "1h": 60,
        "4h": 240,
        "daily": 1440,
    }

    def __init__(self, base_bar_minutes: int = 1, lookback_bars: int = 100):
        self.base_minutes = base_bar_minutes
        self.lookback = lookback_bars
        self._returns_buffer = deque(maxlen=lookback_bars * 1500)  # enough for daily from 1min
        self._bar_count = 0

    def update(self, returns: float) -> None:
        """Feed one base-timeframe return."""
        self._returns_buffer.append(returns)
        self._bar_count += 1

    def analyze(self) -> FractalSignal:
        """Run full fractal analysis across all timeframes."""
        all_returns = np.array(list(self._returns_buffer))
        if len(all_returns) < 500:
            return FractalSignal(0.0, 0.0, 1.5, "insufficient_data")

        scale_analyses = []
        scale_signals = []
        scale_weights = []

        for scale_name, scale_minutes in self.SCALES.items():
            bars_per_scale = scale_minutes // max(self.base_minutes, 1)
            if bars_per_scale < 1:
                bars_per_scale = 1

            # Aggregate returns to this scale
            n = len(all_returns)
            n_scale_bars = n // bars_per_scale
            if n_scale_bars < 30:
                continue

            scale_returns = np.zeros(n_scale_bars)
            for i in range(n_scale_bars):
                chunk = all_returns[i * bars_per_scale:(i + 1) * bars_per_scale]
                scale_returns[i] = np.sum(chunk)  # aggregate returns

            # Hurst exponent at this scale
            hurst = self._compute_hurst(scale_returns)

            # Momentum: normalized mean return
            lookback_scale = min(30, n_scale_bars)
            recent = scale_returns[-lookback_scale:]
            vol = max(float(np.std(recent)), 1e-10)
            momentum = float(np.mean(recent) / vol)
            momentum = float(np.tanh(momentum * 2))

            # Trend strength: R-squared
            t = np.arange(lookback_scale)
            if lookback_scale >= 5:
                slope, intercept = np.polyfit(t, recent, 1)
                fitted = slope * t + intercept
                ss_res = float(np.sum((recent - fitted) ** 2))
                ss_tot = float(np.sum((recent - recent.mean()) ** 2))
                r2 = float(max(0, 1 - ss_res / max(ss_tot, 1e-10)))
            else:
                r2 = 0.0

            # Regime classification at this scale
            if hurst > 0.6 and r2 > 0.3:
                regime = "trending"
            elif hurst < 0.4:
                regime = "mean_reverting"
            else:
                regime = "random"

            analysis = ScaleAnalysis(
                scale_name=scale_name,
                scale_bars=bars_per_scale,
                hurst=hurst,
                momentum=momentum,
                trend_strength=r2,
                volatility=float(np.std(recent) * math.sqrt(252 * 1440 / scale_minutes)),
                regime=regime,
            )
            scale_analyses.append(analysis)
            scale_signals.append(momentum)

            # Weight: higher timeframes get more weight (information is more reliable)
            weight = math.log(bars_per_scale + 1)
            scale_weights.append(weight)

        if not scale_analyses:
            return FractalSignal(0.0, 0.0, 1.5, "insufficient_data")

        signals = np.array(scale_signals)
        weights = np.array(scale_weights)
        weights /= weights.sum()

        # Coherence: how aligned are the scale signals?
        if len(signals) >= 2:
            # Pairwise agreement
            agreements = []
            for i in range(len(signals)):
                for j in range(i + 1, len(signals)):
                    agreements.append(1.0 if np.sign(signals[i]) == np.sign(signals[j]) else 0.0)
            coherence = float(np.mean(agreements))
        else:
            coherence = 0.5

        # Composite signal: coherence-weighted average
        composite = float(np.dot(weights, signals))
        # Scale by coherence (low coherence = reduce signal strength)
        composite *= (0.3 + 0.7 * coherence)
        composite = float(np.clip(composite, -1, 1))

        # Fractal dimension from Hurst exponents
        hursts = [a.hurst for a in scale_analyses]
        avg_hurst = float(np.mean(hursts))
        fractal_dim = 2.0 - avg_hurst  # D = 2 - H

        # Dominant scale: which timeframe has strongest signal
        strongest_idx = int(np.argmax(np.abs(signals)))
        dominant = scale_analyses[strongest_idx].scale_name

        # Divergence detection: when short and long timeframes disagree
        divergence = False
        div_details = ""
        if len(scale_analyses) >= 3:
            short_signal = signals[0]  # shortest timeframe
            long_signal = signals[-1]  # longest timeframe
            if np.sign(short_signal) != np.sign(long_signal) and abs(short_signal) > 0.3 and abs(long_signal) > 0.3:
                divergence = True
                div_details = (f"Short-term ({scale_analyses[0].scale_name}) says "
                              f"{'long' if short_signal > 0 else 'short'}, "
                              f"long-term ({scale_analyses[-1].scale_name}) says "
                              f"{'long' if long_signal > 0 else 'short'}")

        return FractalSignal(
            composite_signal=composite,
            coherence=coherence,
            fractal_dimension=fractal_dim,
            dominant_scale=dominant,
            scale_analyses=scale_analyses,
            divergence_detected=divergence,
            divergence_details=div_details,
        )

    def _compute_hurst(self, returns: np.ndarray, max_lag: int = 30) -> float:
        """R/S Hurst exponent estimation."""
        n = len(returns)
        if n < max_lag * 2:
            return 0.5

        lags = list(range(2, min(max_lag, n // 2)))
        if len(lags) < 3:
            return 0.5

        rs_values = []
        for lag in lags:
            chunks = n // lag
            rs_list = []
            for i in range(chunks):
                chunk = returns[i * lag:(i + 1) * lag]
                mean = chunk.mean()
                cumdev = np.cumsum(chunk - mean)
                R = cumdev.max() - cumdev.min()
                S = max(chunk.std(), 1e-10)
                rs_list.append(R / S)
            if rs_list:
                rs_values.append(float(np.mean(rs_list)))

        if len(rs_values) < 3:
            return 0.5

        log_lags = np.log(np.array(lags[:len(rs_values)]))
        log_rs = np.log(np.array(rs_values) + 1e-10)

        try:
            slope = float(np.polyfit(log_lags, log_rs, 1)[0])
            return float(np.clip(slope, 0.0, 1.0))
        except:
            return 0.5
