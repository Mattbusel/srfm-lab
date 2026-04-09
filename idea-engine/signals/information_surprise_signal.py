"""
Information Surprise Signal: detect when the market is about to move
using information theory before traditional indicators catch it.

Measures the SURPRISE content of each new bar using:
  1. Shannon entropy of return distribution (how unpredictable is the market?)
  2. Kolmogorov complexity approximation (how compressible are recent returns?)
  3. Permutation entropy (how ordered vs random is the price sequence?)
  4. Surprise score: how unexpected was THIS bar given recent history?
  5. Information rate: bits per bar (is the market generating more information?)

Key insight: markets generate MORE information before big moves.
The information rate INCREASES before a breakout or crash, because
the market is "deciding" -- new information is being priced in.
Low entropy = ordered/predictable = trend continuation.
High entropy spike = new information = move coming.
"""

from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from itertools import permutations

import numpy as np


@dataclass
class InformationState:
    """Current information-theoretic state of the market."""
    shannon_entropy: float        # bits of uncertainty
    permutation_entropy: float    # ordinal pattern complexity (0=ordered, 1=random)
    compressibility: float        # 0=incompressible(random), 1=highly compressible(ordered)
    surprise_score: float         # how unexpected was the last bar (0=expected, 5+=very surprising)
    information_rate: float       # bits generated per bar (rolling)
    entropy_trend: float          # is entropy increasing or decreasing?
    regime: str                   # "ordered" / "normal" / "high_information" / "chaos"
    signal: float                 # -1 to +1 trading signal


class ShannonEntropyEstimator:
    """Estimate Shannon entropy from a stream of returns."""

    def __init__(self, n_bins: int = 20, window: int = 63):
        self.n_bins = n_bins
        self.window = window
        self._history = deque(maxlen=window)

    def update(self, value: float) -> float:
        self._history.append(value)
        if len(self._history) < 20:
            return 0.0
        return self._estimate(np.array(list(self._history)))

    def _estimate(self, data: np.ndarray) -> float:
        """Estimate Shannon entropy H = -sum(p * log2(p))."""
        if data.std() < 1e-10:
            return 0.0

        hist, _ = np.histogram(data, bins=self.n_bins, density=True)
        hist = hist[hist > 0]
        if len(hist) == 0:
            return 0.0

        # Normalize to proper probability distribution
        probs = hist / hist.sum()
        entropy = -float(np.sum(probs * np.log2(probs + 1e-15)))

        # Normalize by max entropy (uniform distribution)
        max_entropy = math.log2(self.n_bins)
        return entropy / max(max_entropy, 1e-10)


class PermutationEntropyEstimator:
    """
    Bandt-Pompe permutation entropy: measures the complexity of ordinal patterns.

    Takes a time series and extracts the ordinal patterns of successive values.
    Uniform distribution of patterns = high entropy (random).
    Concentrated patterns = low entropy (ordered/predictable).
    """

    def __init__(self, order: int = 4, delay: int = 1, window: int = 100):
        self.order = order
        self.delay = delay
        self.window = window
        self._history = deque(maxlen=window)

    def update(self, value: float) -> float:
        self._history.append(value)
        if len(self._history) < self.order * self.delay + 10:
            return 0.5
        return self._estimate(np.array(list(self._history)))

    def _estimate(self, data: np.ndarray) -> float:
        """Compute permutation entropy."""
        n = len(data)
        m = self.order
        tau = self.delay

        # Extract ordinal patterns
        pattern_counts: Dict[tuple, int] = {}
        total = 0

        for i in range(n - (m - 1) * tau):
            # Extract m values at delay tau
            values = [data[i + j * tau] for j in range(m)]
            # Convert to ordinal pattern (rank)
            ranks = tuple(sorted(range(m), key=lambda k: values[k]))
            pattern_counts[ranks] = pattern_counts.get(ranks, 0) + 1
            total += 1

        if total == 0:
            return 0.5

        # Shannon entropy of pattern distribution
        entropy = 0.0
        for count in pattern_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize by maximum (log2(m!))
        max_entropy = math.log2(math.factorial(m))
        return float(entropy / max(max_entropy, 1e-10))


class CompressibilityEstimator:
    """
    Estimate Kolmogorov complexity via compressibility.

    If a sequence is random, it's incompressible (high K-complexity).
    If a sequence has patterns, it's compressible (low K-complexity).

    We approximate by:
    1. Discretize the return series into symbols
    2. Run-length encode it
    3. Compressibility = 1 - (compressed_length / original_length)
    """

    def __init__(self, n_symbols: int = 5, window: int = 100):
        self.n_symbols = n_symbols
        self.window = window
        self._history = deque(maxlen=window)

    def update(self, value: float) -> float:
        self._history.append(value)
        if len(self._history) < 30:
            return 0.5
        return self._estimate(np.array(list(self._history)))

    def _estimate(self, data: np.ndarray) -> float:
        """Estimate compressibility via discretization + RLE."""
        # Discretize into symbols
        percentiles = np.percentile(data, np.linspace(0, 100, self.n_symbols + 1))
        symbols = np.digitize(data, percentiles[1:-1])

        # Run-length encoding
        n = len(symbols)
        compressed_length = 0
        i = 0
        while i < n:
            j = i
            while j < n and symbols[j] == symbols[i]:
                j += 1
            compressed_length += 1  # one entry per run
            i = j

        compressibility = 1.0 - (compressed_length / max(n, 1))
        return float(max(0, min(1, compressibility)))


class SurpriseEstimator:
    """
    Measure how surprising each new observation is.

    Surprise = -log2(P(x)) where P(x) is estimated from recent history.
    High surprise = the market did something unexpected.
    """

    def __init__(self, window: int = 63, n_bins: int = 30):
        self.window = window
        self.n_bins = n_bins
        self._history = deque(maxlen=window)

    def update(self, value: float) -> float:
        """Update with new value. Returns surprise in bits."""
        if len(self._history) < 20:
            self._history.append(value)
            return 0.0

        data = np.array(list(self._history))
        mu = float(data.mean())
        sigma = max(float(data.std()), 1e-10)

        # P(x) from Gaussian fit
        z = (value - mu) / sigma
        log_p = -0.5 * z ** 2 - 0.5 * math.log(2 * math.pi) - math.log(sigma)
        surprise = max(0, -log_p / math.log(2))  # convert to bits

        self._history.append(value)
        return float(surprise)


class InformationSurpriseDetector:
    """
    Master information-theoretic signal detector.

    Combines Shannon entropy, permutation entropy, compressibility,
    and surprise into a single signal that detects when the market
    is about to move.
    """

    def __init__(self, window: int = 63):
        self.shannon = ShannonEntropyEstimator(window=window)
        self.permutation = PermutationEntropyEstimator(window=window)
        self.compressibility = CompressibilityEstimator(window=window)
        self.surprise = SurpriseEstimator(window=window)

        self._entropy_history = deque(maxlen=window)
        self._info_rate_history = deque(maxlen=window)
        self._prev_return = 0.0

    def update(self, market_return: float) -> InformationState:
        """Process one return and compute information state."""
        h_shannon = self.shannon.update(market_return)
        h_perm = self.permutation.update(market_return)
        compress = self.compressibility.update(market_return)
        surprise = self.surprise.update(market_return)

        self._entropy_history.append(h_shannon)

        # Information rate: bits per bar (Shannon entropy * bar frequency)
        info_rate = h_shannon
        self._info_rate_history.append(info_rate)

        # Entropy trend
        if len(self._entropy_history) >= 10:
            recent = list(self._entropy_history)
            early = np.mean(recent[-20:-10]) if len(recent) >= 20 else np.mean(recent[:len(recent) // 2])
            late = np.mean(recent[-10:])
            entropy_trend = float(late - early)
        else:
            entropy_trend = 0.0

        # Regime classification
        if h_perm < 0.3:
            regime = "ordered"         # highly predictable
        elif h_perm > 0.85 and surprise > 3.0:
            regime = "chaos"           # extremely random + surprising
        elif entropy_trend > 0.05 or surprise > 2.5:
            regime = "high_information"  # market is "deciding", move coming
        else:
            regime = "normal"

        # Trading signal
        signal = self._compute_signal(h_shannon, h_perm, compress, surprise,
                                        entropy_trend, market_return)

        self._prev_return = market_return

        return InformationState(
            shannon_entropy=h_shannon,
            permutation_entropy=h_perm,
            compressibility=compress,
            surprise_score=surprise,
            information_rate=info_rate,
            entropy_trend=entropy_trend,
            regime=regime,
            signal=signal,
        )

    def _compute_signal(
        self,
        shannon: float,
        perm: float,
        compress: float,
        surprise: float,
        entropy_trend: float,
        last_return: float,
    ) -> float:
        """
        Convert information metrics into a trading signal.

        Key insight: LOW entropy = ordered = trend is strong, follow it.
        HIGH entropy SPIKE = new information = reversal or breakout.
        """
        signal = 0.0

        # 1. Low permutation entropy -> strong trend, follow momentum
        if perm < 0.4:
            signal += 0.5 * np.sign(last_return)

        # 2. High compressibility -> pattern detected, bet on continuation
        if compress > 0.5:
            signal += 0.3 * np.sign(last_return)

        # 3. Entropy increasing + surprise spike -> reversal signal
        if entropy_trend > 0.05 and surprise > 2.5:
            signal -= 0.4 * np.sign(last_return)  # fade the move

        # 4. Very high surprise -> something abnormal, reduce exposure
        if surprise > 4.0:
            signal *= 0.3  # scale down: too uncertain

        return float(np.clip(signal, -1, 1))
