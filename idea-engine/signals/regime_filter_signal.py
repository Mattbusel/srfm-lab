"""
regime_filter_signal.py — Regime-filtered signal engine.

Combines raw signals with a regime classifier to produce regime-gated,
confidence-weighted output. Includes hysteresis, per-regime Sharpe history,
signal blending, and transition detection.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Deque, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Regime definitions
# ---------------------------------------------------------------------------

class Regime(Enum):
    TRENDING_UP   = auto()
    TRENDING_DOWN = auto()
    MEAN_REVERTING = auto()
    HIGH_VOL      = auto()
    LOW_VOL       = auto()
    CRISIS        = auto()
    UNKNOWN       = auto()


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RawSignal:
    name: str
    value: float                  # raw signal in [-1, 1] (or unbounded)
    confidence: float             # in [0, 1]
    metadata: Dict = field(default_factory=dict)


@dataclass
class FilteredSignal:
    """Output of the RegimeFilterSignal engine."""
    signal: float                       # final blended signal
    confidence: float                   # overall confidence
    active_regime: Regime
    regime_confidence: float            # how confident we are in regime
    raw_signals_by_regime: Dict[str, float]   # {signal_name: value}
    active_signal_names: List[str]      # signals that were switched on
    inactive_signal_names: List[str]    # signals that were switched off
    regime_transition: bool             # True if regime just changed
    previous_regime: Optional[Regime]
    sharpe_estimate: float              # expected Sharpe in this regime
    blend_weights: Dict[str, float]     # {signal_name: weight used}


@dataclass
class RegimeHistory:
    """Sliding window of (regime, signal_value, return) triples."""
    max_len: int = 500
    regimes: Deque[Regime] = field(default_factory=deque)
    signals: Deque[float] = field(default_factory=deque)
    returns: Deque[float] = field(default_factory=deque)

    def push(self, regime: Regime, signal: float, ret: float) -> None:
        self.regimes.append(regime)
        self.signals.append(signal)
        self.returns.append(ret)
        if len(self.regimes) > self.max_len:
            self.regimes.popleft()
            self.signals.popleft()
            self.returns.popleft()


# ---------------------------------------------------------------------------
# Default signal activation maps by regime
# ---------------------------------------------------------------------------

# {Regime: set of signal names that are ACTIVE in that regime}
DEFAULT_ACTIVATION: Dict[Regime, List[str]] = {
    Regime.TRENDING_UP:    ["momentum", "breakout", "trend_following"],
    Regime.TRENDING_DOWN:  ["momentum", "short_momentum", "trend_following"],
    Regime.MEAN_REVERTING: ["mean_reversion", "rsi_fade", "zscore_revert"],
    Regime.HIGH_VOL:       ["vol_breakout", "gamma_scalp", "crisis_alpha"],
    Regime.LOW_VOL:        ["carry", "mean_reversion", "rsi_fade"],
    Regime.CRISIS:         ["crisis_alpha", "defensive", "short_momentum"],
    Regime.UNKNOWN:        [],
}

# Default blend weights per regime (relative; will be normalised)
DEFAULT_BLEND_WEIGHTS: Dict[Regime, Dict[str, float]] = {
    Regime.TRENDING_UP:    {"momentum": 0.5, "breakout": 0.3, "trend_following": 0.2},
    Regime.TRENDING_DOWN:  {"momentum": 0.4, "short_momentum": 0.4, "trend_following": 0.2},
    Regime.MEAN_REVERTING: {"mean_reversion": 0.5, "rsi_fade": 0.3, "zscore_revert": 0.2},
    Regime.HIGH_VOL:       {"vol_breakout": 0.5, "gamma_scalp": 0.3, "crisis_alpha": 0.2},
    Regime.LOW_VOL:        {"carry": 0.4, "mean_reversion": 0.4, "rsi_fade": 0.2},
    Regime.CRISIS:         {"crisis_alpha": 0.5, "defensive": 0.3, "short_momentum": 0.2},
    Regime.UNKNOWN:        {},
}


# ---------------------------------------------------------------------------
# Regime Detector
# ---------------------------------------------------------------------------

class RegimeDetector:
    """
    Classifies the current market regime from recent return/vol data.

    Uses rolling statistics:
      - trending: Hurst-like or autocorrelation signal
      - volatility level: high vs low relative to history
      - crisis: extreme drawdown or vol spike
    """

    def __init__(self,
                 vol_window: int = 20,
                 trend_window: int = 30,
                 crisis_vol_multiplier: float = 2.5,
                 high_vol_percentile: float = 70.0,
                 trend_ac_threshold: float = 0.15,
                 history_len: int = 252):
        self.vol_window = vol_window
        self.trend_window = trend_window
        self.crisis_vol_mult = crisis_vol_multiplier
        self.high_vol_pct = high_vol_percentile
        self.trend_ac_thresh = trend_ac_threshold
        self._return_history: Deque[float] = deque(maxlen=history_len)
        self._vol_history: Deque[float] = deque(maxlen=history_len)

    def update(self, ret: float) -> None:
        self._return_history.append(ret)

    def _rolling_vol(self, window: int) -> float:
        if len(self._return_history) < 2:
            return 0.0
        data = list(self._return_history)[-window:]
        return float(np.std(data, ddof=1)) if len(data) > 1 else 0.0

    def _autocorrelation_lag1(self) -> float:
        data = list(self._return_history)[-self.trend_window:]
        if len(data) < 4:
            return 0.0
        arr = np.array(data)
        mean = arr.mean()
        c0 = np.sum((arr - mean) ** 2)
        c1 = np.sum((arr[:-1] - mean) * (arr[1:] - mean))
        return float(c1 / c0) if c0 > 0 else 0.0

    def _vol_percentile(self) -> float:
        """Current vol's percentile rank in historical vol distribution."""
        if len(self._vol_history) < 10:
            return 50.0
        v_hist = list(self._vol_history)
        cur_vol = self._rolling_vol(self.vol_window)
        return float(np.mean(np.array(v_hist) <= cur_vol) * 100)

    def detect(self) -> Tuple[Regime, float]:
        """
        Returns (Regime, confidence ∈ [0, 1]).
        """
        if len(self._return_history) < self.vol_window:
            return Regime.UNKNOWN, 0.0

        cur_vol = self._rolling_vol(self.vol_window)
        long_vol = self._rolling_vol(min(len(self._return_history), 252))
        ac = self._autocorrelation_lag1()

        # Store rolling vol for percentile tracking
        self._vol_history.append(cur_vol)
        vol_ratio = cur_vol / (long_vol + 1e-10)

        # Crisis: vol spike
        if vol_ratio > self.crisis_vol_mult:
            confidence = min(1.0, (vol_ratio - self.crisis_vol_mult) / self.crisis_vol_mult)
            return Regime.CRISIS, 0.5 + 0.5 * confidence

        # High vol regime
        vol_pct = self._vol_percentile()
        if vol_pct >= self.high_vol_pct:
            conf = (vol_pct - self.high_vol_pct) / (100.0 - self.high_vol_pct + 1e-9)
            return Regime.HIGH_VOL, 0.5 + 0.5 * conf

        if vol_pct <= 100.0 - self.high_vol_pct:
            conf = (100.0 - self.high_vol_pct - vol_pct) / (100.0 - self.high_vol_pct + 1e-9)
            return Regime.LOW_VOL, 0.5 + 0.5 * conf

        # Trending vs mean-reverting based on lag-1 autocorrelation
        if ac > self.trend_ac_thresh:
            # Trending: check direction
            data = list(self._return_history)[-self.trend_window:]
            net_return = float(np.sum(data))
            conf = min(1.0, abs(ac) / 0.5)
            if net_return >= 0:
                return Regime.TRENDING_UP, 0.4 + 0.4 * conf
            else:
                return Regime.TRENDING_DOWN, 0.4 + 0.4 * conf
        elif ac < -self.trend_ac_thresh:
            conf = min(1.0, abs(ac) / 0.5)
            return Regime.MEAN_REVERTING, 0.4 + 0.4 * conf

        # Borderline: low confidence trending
        data = list(self._return_history)[-self.trend_window:]
        net_return = float(np.sum(data))
        if net_return >= 0:
            return Regime.TRENDING_UP, 0.3
        else:
            return Regime.MEAN_REVERTING, 0.3


# ---------------------------------------------------------------------------
# Hysteresis Filter
# ---------------------------------------------------------------------------

class HysteresisFilter:
    """
    Prevent rapid regime flipping: a new regime must be confirmed over
    `confirm_bars` consecutive observations before being accepted.
    """

    def __init__(self, confirm_bars: int = 5, min_confidence: float = 0.45):
        self.confirm_bars = confirm_bars
        self.min_confidence = min_confidence
        self._current: Regime = Regime.UNKNOWN
        self._candidate: Optional[Regime] = None
        self._candidate_count: int = 0

    def update(self, proposed: Regime, confidence: float) -> Tuple[Regime, bool]:
        """
        Returns (accepted_regime, is_transition).
        """
        if confidence < self.min_confidence:
            proposed = Regime.UNKNOWN

        if proposed == self._current:
            self._candidate = None
            self._candidate_count = 0
            return self._current, False

        if proposed == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = proposed
            self._candidate_count = 1

        if self._candidate_count >= self.confirm_bars:
            prev = self._current
            self._current = proposed
            self._candidate = None
            self._candidate_count = 0
            return self._current, (prev != self._current)

        return self._current, False

    @property
    def current_regime(self) -> Regime:
        return self._current


# ---------------------------------------------------------------------------
# Per-Regime Sharpe Estimator
# ---------------------------------------------------------------------------

class RegimeSharpeEstimator:
    """
    Maintains per-regime Sharpe estimate from historical (signal, return) pairs.
    Uses exponentially decaying weights for recency bias.
    """

    def __init__(self, min_obs: int = 20, decay: float = 0.97):
        self.min_obs = min_obs
        self.decay = decay
        self._data: Dict[Regime, List[Tuple[float, float]]] = {r: [] for r in Regime}

    def record(self, regime: Regime, signal: float, ret: float) -> None:
        self._data[regime].append((signal, ret))
        # Cap at 1000 observations per regime
        if len(self._data[regime]) > 1000:
            self._data[regime] = self._data[regime][-1000:]

    def estimate_sharpe(self, regime: Regime) -> float:
        """
        Return expected annualised Sharpe (rough) for the given regime.
        Returns NaN if insufficient history.
        """
        history = self._data[regime]
        if len(history) < self.min_obs:
            return float('nan')
        n = len(history)
        weights = np.array([self.decay ** (n - 1 - i) for i in range(n)])
        weights /= weights.sum()
        signals = np.array([h[0] for h in history])
        rets = np.array([h[1] for h in history])
        # Signal-weighted returns
        pnl = signals * rets
        w_mean = float(np.sum(weights * pnl))
        w_var = float(np.sum(weights * (pnl - w_mean) ** 2))
        w_std = math.sqrt(w_var) if w_var > 0 else 1e-10
        sharpe = w_mean / w_std * math.sqrt(252)
        return sharpe


# ---------------------------------------------------------------------------
# Main: RegimeFilterSignal
# ---------------------------------------------------------------------------

class RegimeFilterSignal:
    """
    Combines a set of raw signals with a regime classifier to produce
    a regime-filtered, confidence-weighted signal output.

    Usage
    -----
    engine = RegimeFilterSignal(signal_names=["momentum", "mean_reversion", ...])
    for bar in data:
        raw_signals = [RawSignal("momentum", 0.4, 0.8), ...]
        engine.update_returns(bar_return)
        result: FilteredSignal = engine.process(raw_signals)
    """

    def __init__(self,
                 signal_names: Optional[List[str]] = None,
                 activation_map: Optional[Dict[Regime, List[str]]] = None,
                 blend_weights: Optional[Dict[Regime, Dict[str, float]]] = None,
                 hysteresis_bars: int = 5,
                 hysteresis_min_conf: float = 0.45,
                 regime_detector_kwargs: Optional[Dict] = None,
                 sharpe_decay: float = 0.97):
        self.signal_names: List[str] = signal_names or []
        self.activation_map = activation_map or DEFAULT_ACTIVATION
        self.blend_weights = blend_weights or DEFAULT_BLEND_WEIGHTS
        det_kwargs = regime_detector_kwargs or {}
        self._detector = RegimeDetector(**det_kwargs)
        self._hysteresis = HysteresisFilter(confirm_bars=hysteresis_bars,
                                             min_confidence=hysteresis_min_conf)
        self._sharpe_est = RegimeSharpeEstimator(decay=sharpe_decay)
        self._history = RegimeHistory()
        self._prev_regime: Optional[Regime] = None
        self._last_signal: float = 0.0

    def update_returns(self, ret: float) -> None:
        """Feed a new bar's return so regime detection stays current."""
        self._detector.update(ret)
        # Record in Sharpe estimator using last signal
        regime = self._hysteresis.current_regime
        self._sharpe_est.record(regime, self._last_signal, ret)

    def process(self, raw_signals: List[RawSignal]) -> FilteredSignal:
        """
        Process a list of raw signals through the regime filter.

        Returns a FilteredSignal with blended output, active regime,
        and per-signal information.
        """
        # Detect regime
        proposed_regime, regime_conf = self._detector.detect()
        active_regime, is_transition = self._hysteresis.update(proposed_regime, regime_conf)

        # Determine active signals for this regime
        active_names = set(self.activation_map.get(active_regime, []))
        # Retrieve blend weights for this regime
        raw_weights = dict(self.blend_weights.get(active_regime, {}))

        # Build signal lookup
        signal_lookup: Dict[str, RawSignal] = {s.name: s for s in raw_signals}
        raw_by_name: Dict[str, float] = {s.name: s.value for s in raw_signals}

        active_sigs: List[str] = []
        inactive_sigs: List[str] = []
        for s in raw_signals:
            if s.name in active_names:
                active_sigs.append(s.name)
            else:
                inactive_sigs.append(s.name)

        # Blend: weighted sum of active signals
        total_weight = 0.0
        blended = 0.0
        blend_used: Dict[str, float] = {}
        confidence_acc = 0.0

        for name in active_sigs:
            if name not in signal_lookup:
                continue
            sig = signal_lookup[name]
            w = raw_weights.get(name, 1.0 / max(len(active_sigs), 1))
            # Scale weight by signal's own confidence
            eff_weight = w * sig.confidence
            blended += eff_weight * sig.value
            total_weight += eff_weight
            blend_used[name] = eff_weight
            confidence_acc += sig.confidence

        if total_weight > 0:
            blended /= total_weight
            overall_conf = (confidence_acc / len(active_sigs)) * regime_conf
        else:
            blended = 0.0
            overall_conf = 0.0
            blend_used = {}

        # Normalise blend weights for reporting
        if total_weight > 0:
            blend_used = {k: v / total_weight for k, v in blend_used.items()}

        # Estimate Sharpe for this regime
        sharpe = self._sharpe_est.estimate_sharpe(active_regime)
        if math.isnan(sharpe):
            sharpe = 0.0

        # Modulate signal by Sharpe sign: suppress if Sharpe is negative
        if sharpe < -0.5:
            blended *= 0.0   # disable entirely
            overall_conf *= 0.1
        elif sharpe < 0.0:
            blended *= 0.3

        # Store
        self._last_signal = blended
        self._history.push(active_regime, blended, 0.0)  # return filled by update_returns

        return FilteredSignal(
            signal=float(np.clip(blended, -1.0, 1.0)),
            confidence=float(np.clip(overall_conf, 0.0, 1.0)),
            active_regime=active_regime,
            regime_confidence=float(regime_conf),
            raw_signals_by_regime=raw_by_name,
            active_signal_names=active_sigs,
            inactive_signal_names=inactive_sigs,
            regime_transition=is_transition,
            previous_regime=self._prev_regime if is_transition else None,
            sharpe_estimate=sharpe,
            blend_weights=blend_used,
        )

    def set_activation(self, regime: Regime, signal_names: List[str]) -> None:
        """Override which signals are active for a given regime."""
        self.activation_map[regime] = signal_names

    def set_blend_weights(self, regime: Regime, weights: Dict[str, float]) -> None:
        """Override blend weights for a given regime."""
        self.blend_weights[regime] = weights

    def regime_sharpe_summary(self) -> Dict[str, float]:
        """Return estimated Sharpe for each regime with sufficient history."""
        return {r.name: self._sharpe_est.estimate_sharpe(r)
                for r in Regime if not math.isnan(self._sharpe_est.estimate_sharpe(r))}

    def current_regime(self) -> Regime:
        return self._hysteresis.current_regime


# ---------------------------------------------------------------------------
# Convenience: build from config dict
# ---------------------------------------------------------------------------

def build_regime_filter(config: Dict) -> RegimeFilterSignal:
    """
    Build a RegimeFilterSignal from a config dictionary.

    Example config:
    {
        "signal_names": ["momentum", "mean_reversion"],
        "hysteresis_bars": 5,
        "regime_detector": {"vol_window": 20, "trend_window": 30}
    }
    """
    return RegimeFilterSignal(
        signal_names=config.get("signal_names"),
        hysteresis_bars=config.get("hysteresis_bars", 5),
        hysteresis_min_conf=config.get("hysteresis_min_conf", 0.45),
        regime_detector_kwargs=config.get("regime_detector", {}),
        sharpe_decay=config.get("sharpe_decay", 0.97),
    )
