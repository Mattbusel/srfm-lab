"""
Alpha decay signal — monitors and responds to signal alpha decay.

Implements:
  - Rolling IC (Information Coefficient) tracking
  - IC half-life estimation
  - Alpha decay rate classification (fast/moderate/slow)
  - Decay-adjusted position sizing
  - Signal regime transition: live → decaying → dead
  - Decay alert generation
  - Autocorrelation of signal errors as decay indicator
  - Multi-signal decay portfolio: blend signals by inverse decay rate
"""

from __future__ import annotations
import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ICMeasurement:
    """Single IC measurement."""
    timestamp: int
    ic: float
    ic_ir: float          # IC / std(IC)
    signal: float
    realized_return: float


@dataclass
class DecayStatus:
    """Current alpha decay status."""
    signal_name: str
    current_ic: float
    rolling_ic_21d: float
    rolling_ic_63d: float
    ic_trend: float             # positive = improving, negative = decaying
    half_life_est: float        # estimated IC half-life in days
    decay_rate: str             # fast/moderate/slow/stable/improving
    size_multiplier: float      # how much to scale position
    is_decaying: bool
    decay_warning: bool
    measurements: list[ICMeasurement] = field(default_factory=list)


class AlphaDecaySignal:
    """
    Monitors signal alpha decay in real-time.
    Uses rolling IC and IC trend to adapt sizing.
    """

    def __init__(
        self,
        signal_name: str,
        ic_window_fast: int = 21,
        ic_window_slow: int = 63,
        min_ic_for_trading: float = 0.02,
        decay_warning_threshold: float = -0.3,  # IC trend below this = warning
    ):
        self.signal_name = signal_name
        self.fast_window = ic_window_fast
        self.slow_window = ic_window_slow
        self.min_ic = min_ic_for_trading
        self.decay_threshold = decay_warning_threshold

        self._signals: list[float] = []
        self._returns: list[float] = []
        self._ics: list[float] = []

    def record(self, signal: float, realized_return: float) -> None:
        """Record a signal prediction and its realized return."""
        self._signals.append(signal)
        self._returns.append(realized_return)

        n = len(self._signals)
        if n >= 5:
            sigs = np.array(self._signals[-min(n, self.slow_window):])
            rets = np.array(self._returns[-min(n, self.slow_window):])
            ic = float(np.corrcoef(sigs, rets)[0, 1])
            self._ics.append(ic)
        else:
            self._ics.append(0.0)

    def compute_decay_status(self) -> DecayStatus:
        n = len(self._ics)
        if n < 5:
            return DecayStatus(
                signal_name=self.signal_name,
                current_ic=0.0,
                rolling_ic_21d=0.0,
                rolling_ic_63d=0.0,
                ic_trend=0.0,
                half_life_est=float("inf"),
                decay_rate="insufficient_data",
                size_multiplier=0.5,
                is_decaying=False,
                decay_warning=False,
            )

        ics = np.array(self._ics)

        # Current IC
        current_ic = float(ics[-1])

        # Rolling ICs
        fast = min(n, self.fast_window)
        slow = min(n, self.slow_window)
        ic_21 = float(ics[-fast:].mean())
        ic_63 = float(ics[-slow:].mean())

        # IC trend (slope of IC over fast window)
        if fast >= 5:
            t = np.arange(fast)
            ic_trend_slope = float(np.polyfit(t, ics[-fast:], 1)[0])
        else:
            ic_trend_slope = 0.0

        # Half-life: fit exponential decay to |IC| series
        half_life = _estimate_half_life(ics)

        # Decay rate classification
        if ic_trend_slope > 0.001:
            decay_rate = "improving"
        elif ic_trend_slope > -0.001:
            decay_rate = "stable"
        elif half_life < 10:
            decay_rate = "fast"
        elif half_life < 30:
            decay_rate = "moderate"
        else:
            decay_rate = "slow"

        # Size multiplier
        if ic_21 < self.min_ic:
            size_mult = 0.0
        elif decay_rate == "fast":
            size_mult = max(ic_21 / max(abs(ic_63), self.min_ic), 0.1)
        elif decay_rate == "improving":
            size_mult = 1.2
        else:
            size_mult = float(np.clip(ic_21 / 0.05, 0.2, 1.5))

        is_decaying = ic_trend_slope < self.decay_threshold and decay_rate in ("fast", "moderate")
        decay_warning = ic_21 < self.min_ic * 0.5 or decay_rate == "fast"

        return DecayStatus(
            signal_name=self.signal_name,
            current_ic=current_ic,
            rolling_ic_21d=ic_21,
            rolling_ic_63d=ic_63,
            ic_trend=ic_trend_slope,
            half_life_est=half_life,
            decay_rate=decay_rate,
            size_multiplier=float(np.clip(size_mult, 0, 2)),
            is_decaying=is_decaying,
            decay_warning=decay_warning,
        )

    def rolling_ic_series(self, window: int = 21) -> np.ndarray:
        """Return rolling IC time series."""
        n = len(self._signals)
        if n < window + 1:
            return np.zeros(n)
        ic_series = np.zeros(n)
        for t in range(window, n):
            sigs = np.array(self._signals[t - window: t])
            rets = np.array(self._returns[t - window: t])
            if sigs.std() > 1e-10 and rets.std() > 1e-10:
                ic_series[t] = float(np.corrcoef(sigs, rets)[0, 1])
        return ic_series


def _estimate_half_life(ics: np.ndarray) -> float:
    """Estimate half-life of IC from its autocorrelation."""
    if len(ics) < 5:
        return float("inf")
    abs_ic = np.abs(ics)
    if abs_ic.std() < 1e-10:
        return float("inf")
    # OU-process fit: kappa from lag-1 autocorrelation
    if len(ics) >= 3:
        acf1 = float(np.corrcoef(abs_ic[1:], abs_ic[:-1])[0, 1])
        acf1 = np.clip(acf1, -0.999, 0.999)
        if acf1 > 0:
            kappa = -math.log(acf1)
            half_life = math.log(2) / max(kappa, 1e-6)
        else:
            half_life = 1.0
    else:
        half_life = float("inf")
    return float(min(half_life, 1e6))


# ── Multi-Signal Decay Portfolio ──────────────────────────────────────────────

class MultiSignalDecayPortfolio:
    """
    Blend multiple signals weighted by their inverse decay rate.
    Signals that are decaying get reduced weight.
    """

    def __init__(self, signal_names: list[str]):
        self.trackers = {
            name: AlphaDecaySignal(name) for name in signal_names
        }
        self._signal_names = signal_names

    def record_all(
        self,
        signals: dict[str, float],
        realized_return: float,
    ) -> None:
        for name, tracker in self.trackers.items():
            if name in signals:
                tracker.record(signals[name], realized_return)

    def blend_signals(self, current_signals: dict[str, float]) -> dict:
        """
        Compute decay-adjusted blended signal.
        Weights = IC_21d (positive only), normalized.
        """
        statuses = {
            name: tracker.compute_decay_status()
            for name, tracker in self.trackers.items()
        }

        weights = {}
        for name, status in statuses.items():
            if status.is_decaying or status.decay_warning:
                w = max(status.rolling_ic_21d, 0) * 0.3
            else:
                w = max(status.rolling_ic_21d, 0)
            weights[name] = w

        total_w = sum(weights.values()) + 1e-10
        weights = {k: v / total_w for k, v in weights.items()}

        blended = sum(
            weights.get(name, 0) * current_signals.get(name, 0)
            for name in self._signal_names
        )

        return {
            "blended_signal": float(blended),
            "weights": weights,
            "statuses": {name: {
                "ic_21d": s.rolling_ic_21d,
                "decay_rate": s.decay_rate,
                "size_mult": s.size_multiplier,
            } for name, s in statuses.items()},
            "n_active_signals": sum(1 for s in statuses.values() if not s.decay_warning),
        }

    def decay_report(self) -> dict:
        """Summary of decay status across all signals."""
        statuses = {
            name: tracker.compute_decay_status()
            for name, tracker in self.trackers.items()
        }
        n_decaying = sum(1 for s in statuses.values() if s.is_decaying)
        avg_ic = float(np.mean([s.rolling_ic_21d for s in statuses.values()]))
        avg_half_life = float(np.mean([
            s.half_life_est for s in statuses.values()
            if s.half_life_est < 1e5
        ] or [30.0]))

        return {
            "n_signals": len(statuses),
            "n_decaying": n_decaying,
            "n_healthy": len(statuses) - n_decaying,
            "avg_rolling_ic_21d": avg_ic,
            "avg_half_life_days": avg_half_life,
            "signal_statuses": {
                name: s.decay_rate for name, s in statuses.items()
            },
            "warnings": [
                name for name, s in statuses.items() if s.decay_warning
            ],
        }
