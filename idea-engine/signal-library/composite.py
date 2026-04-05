"""
idea-engine/signal-library/composite.py
=========================================
Composite signal combiners for the SRFM Idea Engine Signal Library.

These classes take multiple Signal instances and produce a single combined
output, enabling ensemble methods, regime-conditional switching, and
adaptive weight learning.

Classes
-------
SignalEnsemble          — weighted average of N signals
SignalVoter             — majority vote across binary signals
RegimeConditionalSignal — uses signal A in BULL, signal B in BEAR
SignalStack             — stacks multiple signals into a feature matrix
AdaptiveSignalWeighter  — adjusts weights based on rolling signal IC
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .base import Signal, SignalResult, CATEGORIES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_signals(signals: List[Signal]) -> None:
    for sig in signals:
        if not isinstance(sig, Signal):
            raise TypeError(f"Expected Signal instance, got {type(sig)}")


def _compute_all(
    signals: List[Signal],
    df: pd.DataFrame,
) -> Dict[str, pd.Series]:
    """Compute all signals, returning a dict name → Series."""
    results: Dict[str, pd.Series] = {}
    for sig in signals:
        try:
            results[sig.name] = sig.compute(df)
        except Exception as exc:
            logger.warning(
                "CompositeSignal: error computing signal '%s': %s — filling NaN.",
                sig.name, exc
            )
            results[sig.name] = pd.Series(np.nan, index=df.index)
    return results


# ---------------------------------------------------------------------------
# 1. SignalEnsemble
# ---------------------------------------------------------------------------

class SignalEnsemble(Signal):
    """
    Weighted average ensemble of N signals.

    Each sub-signal is standardised (z-scored) before averaging so that
    signals with larger absolute values don't dominate the ensemble.

    Parameters
    ----------
    signals : list[Signal]
        Sub-signals to combine.
    weights : list[float] | None
        Per-signal weights. If None, equal weights are used.
        Weights do not need to sum to 1 — they are normalised automatically.
    standardise : bool
        If True (default), each signal is z-scored before weighting.
    std_window : int
        Window for the rolling z-score standardisation.
    """

    name:        str = "signal_ensemble"
    category:    str = "composite"
    lookback:    int = 1
    signal_type: str = "continuous"

    def __init__(
        self,
        signals:     List[Signal],
        weights:     Optional[List[float]] = None,
        standardise: bool                  = True,
        std_window:  int                   = 60,
    ) -> None:
        _validate_signals(signals)
        self.signals     = signals
        self.standardise = standardise
        self.std_window  = std_window
        self.lookback    = max(sig.lookback for sig in signals)

        if weights is None:
            self._weights = np.ones(len(signals)) / len(signals)
        else:
            if len(weights) != len(signals):
                raise ValueError("len(weights) must match len(signals).")
            w = np.array(weights, dtype=float)
            self._weights = w / w.sum()

    def compute(self, df: pd.DataFrame) -> pd.Series:
        raw_signals = _compute_all(self.signals, df)
        combined    = pd.Series(0.0, index=df.index)

        for i, sig in enumerate(self.signals):
            values = raw_signals[sig.name]
            if self.standardise:
                mu  = values.rolling(self.std_window, min_periods=2).mean()
                std = values.rolling(self.std_window, min_periods=2).std()
                values = (values - mu) / std.replace(0.0, np.nan)
            combined = combined + self._weights[i] * values.fillna(0.0)

        combined.name = self.name
        return combined

    def compute_result(self, df: pd.DataFrame) -> SignalResult:
        raw_signals = _compute_all(self.signals, df)
        combined    = self.compute(df)
        return SignalResult(
            values=combined,
            signal_name=self.name,
            category=self.category,
            metadata={
                "weights":       dict(zip([s.name for s in self.signals], self._weights.tolist())),
                "n_signals":     len(self.signals),
                "standardise":   self.standardise,
                "sub_signals":   raw_signals,
                "lookback":      self.lookback,
                "signal_type":   self.signal_type,
            },
        )


# ---------------------------------------------------------------------------
# 2. SignalVoter
# ---------------------------------------------------------------------------

class SignalVoter(Signal):
    """
    Majority vote across binary or continuous signals.

    Each sub-signal is thresholded:
        value > threshold  → vote +1 (bullish)
        value < -threshold → vote -1 (bearish)
        otherwise          → vote  0 (neutral)

    Final signal = sum of votes / n_signals, giving a value in [-1, +1].
    If ``require_majority=True``, only returns ±1 when abs(vote_sum) >= majority_thresh.

    Parameters
    ----------
    signals : list[Signal]
    threshold : float
        Absolute signal value required to cast a directional vote.
    majority_thresh : int
        Minimum absolute vote count to output a non-zero signal.
    require_majority : bool
    """

    name:        str = "signal_voter"
    category:    str = "composite"
    lookback:    int = 1
    signal_type: str = "continuous"

    def __init__(
        self,
        signals:          List[Signal],
        threshold:        float = 0.0,
        majority_thresh:  int   = 1,
        require_majority: bool  = False,
    ) -> None:
        _validate_signals(signals)
        self.signals          = signals
        self.threshold        = threshold
        self.majority_thresh  = majority_thresh
        self.require_majority = require_majority
        self.lookback         = max(sig.lookback for sig in signals)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        raw_signals  = _compute_all(self.signals, df)
        vote_total   = pd.Series(0.0, index=df.index)

        for sig in self.signals:
            values = raw_signals[sig.name].fillna(0.0)
            votes  = np.where(values >  self.threshold,  1.0,
                     np.where(values < -self.threshold, -1.0, 0.0))
            vote_total = vote_total + pd.Series(votes, index=df.index)

        n = len(self.signals)
        normalised = vote_total / n

        if self.require_majority:
            normalised = normalised.where(vote_total.abs() >= self.majority_thresh, 0.0)

        normalised.name = self.name
        return normalised


# ---------------------------------------------------------------------------
# 3. RegimeConditionalSignal
# ---------------------------------------------------------------------------

class RegimeConditionalSignal(Signal):
    """
    Regime-conditional signal switcher.

    Uses signal_a in a BULL regime and signal_b in a BEAR regime,
    as determined by a regime indicator.

    Regime indicator: any continuous signal whose value > 0 → BULL, ≤ 0 → BEAR.
    Common choices: EMAMomentum, ADX with sign, or a dedicated regime oracle.

    Parameters
    ----------
    signal_a : Signal
        Signal used in BULL regime (regime_signal > regime_threshold).
    signal_b : Signal
        Signal used in BEAR regime.
    regime_signal : Signal
        The signal whose value determines current regime.
    regime_threshold : float
        Threshold for regime_signal to classify BULL vs BEAR.
    blend : bool
        If True, blend between signal_a and signal_b using regime_signal
        as a soft weight rather than a hard switch.
    """

    name:        str = "regime_conditional"
    category:    str = "composite"
    lookback:    int = 1
    signal_type: str = "continuous"

    def __init__(
        self,
        signal_a:          Signal,
        signal_b:          Signal,
        regime_signal:     Signal,
        regime_threshold:  float = 0.0,
        blend:             bool  = False,
    ) -> None:
        for s in (signal_a, signal_b, regime_signal):
            if not isinstance(s, Signal):
                raise TypeError(f"Expected Signal, got {type(s)}")
        self.signal_a          = signal_a
        self.signal_b          = signal_b
        self.regime_signal     = regime_signal
        self.regime_threshold  = regime_threshold
        self.blend             = blend
        self.lookback          = max(signal_a.lookback, signal_b.lookback,
                                     regime_signal.lookback)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        try:
            val_a   = self.signal_a.compute(df)
        except Exception as exc:
            logger.warning("RegimeConditional: signal_a error: %s", exc)
            val_a = pd.Series(np.nan, index=df.index)

        try:
            val_b   = self.signal_b.compute(df)
        except Exception as exc:
            logger.warning("RegimeConditional: signal_b error: %s", exc)
            val_b = pd.Series(np.nan, index=df.index)

        try:
            regime  = self.regime_signal.compute(df)
        except Exception as exc:
            logger.warning("RegimeConditional: regime_signal error: %s", exc)
            regime = pd.Series(0.0, index=df.index)

        if self.blend:
            # Soft blend: weight = sigmoid of regime_signal around threshold
            z      = regime - self.regime_threshold
            weight = 1.0 / (1.0 + np.exp(-z * 5.0))   # sigmoid, steepness=5
            result = weight * val_a.fillna(0.0) + (1.0 - weight) * val_b.fillna(0.0)
        else:
            # Hard switch
            bull   = (regime > self.regime_threshold)
            result = val_a.where(bull, val_b)

        result.name = self.name
        return result

    def compute_result(self, df: pd.DataFrame) -> SignalResult:
        values  = self.compute(df)
        regime  = self.regime_signal.compute(df)
        return SignalResult(
            values=values,
            signal_name=self.name,
            category=self.category,
            metadata={
                "signal_a":         self.signal_a.name,
                "signal_b":         self.signal_b.name,
                "regime_signal":    self.regime_signal.name,
                "regime_threshold": self.regime_threshold,
                "blend":            self.blend,
                "bull_fraction":    float((regime > self.regime_threshold).mean()),
                "lookback":         self.lookback,
                "signal_type":      self.signal_type,
            },
        )


# ---------------------------------------------------------------------------
# 4. SignalStack
# ---------------------------------------------------------------------------

class SignalStack(Signal):
    """
    Signal stack: computes all sub-signals and stacks them into a feature matrix.

    The ``compute`` method returns the first sub-signal as a scalar Series
    (for API compatibility), but ``compute_matrix`` returns the full
    DataFrame with one column per signal.

    Useful for passing features to an ML model or for correlation analysis.

    Parameters
    ----------
    signals : list[Signal]
    align : bool
        If True, ensure all signals are aligned to the same index.
    fillna : float | None
        If not None, fill NaN values with this value (e.g. 0.0).
    """

    name:        str = "signal_stack"
    category:    str = "composite"
    lookback:    int = 1
    signal_type: str = "continuous"

    def __init__(
        self,
        signals:  List[Signal],
        align:    bool          = True,
        fillna:   Optional[float] = None,
    ) -> None:
        _validate_signals(signals)
        self.signals  = signals
        self.align    = align
        self._fillna  = fillna
        self.lookback = max(sig.lookback for sig in signals)

    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Returns the first signal's values (compatibility method)."""
        if not self.signals:
            return pd.Series(np.nan, index=df.index, name=self.name)
        try:
            result = self.signals[0].compute(df)
        except Exception:
            result = pd.Series(np.nan, index=df.index)
        result.name = self.name
        return result

    def compute_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all signals and return a feature matrix (rows=bars, cols=signals).
        """
        raw = _compute_all(self.signals, df)
        frame = pd.DataFrame(raw, index=df.index)
        if self._fillna is not None:
            frame = frame.fillna(self._fillna)
        return frame

    def compute_result(self, df: pd.DataFrame) -> SignalResult:
        matrix = self.compute_matrix(df)
        # Primary value = first signal
        primary = matrix.iloc[:, 0] if not matrix.empty else pd.Series(np.nan, index=df.index)
        primary.name = self.name
        return SignalResult(
            values=primary,
            signal_name=self.name,
            category=self.category,
            metadata={
                "feature_matrix":  matrix,
                "signal_names":    list(matrix.columns),
                "n_signals":       len(self.signals),
                "lookback":        self.lookback,
                "signal_type":     self.signal_type,
            },
        )

    @property
    def signal_names(self) -> List[str]:
        return [sig.name for sig in self.signals]


# ---------------------------------------------------------------------------
# 5. AdaptiveSignalWeighter
# ---------------------------------------------------------------------------

class AdaptiveSignalWeighter(Signal):
    """
    Adaptive Signal Weighter: adjusts signal weights based on recent
    Information Coefficient (IC) with forward returns.

    IC is computed as the Spearman correlation between signal values and
    the subsequent N-bar return. Signals with higher recent IC get more weight.

    Weights are updated every ``rebalance_every`` bars.
    Negative-IC signals can be inverted (``allow_inversion=True``) or zeroed.

    Parameters
    ----------
    signals : list[Signal]
    ic_window : int
        Rolling window (bars) over which to compute IC.
    forward_bars : int
        Forward return horizon (bars) for IC computation.
    rebalance_every : int
        How often (bars) to re-solve signal weights.
    allow_inversion : bool
        If True, signals with consistently negative IC are inverted.
    min_weight : float
        Minimum weight per signal (prevents full exclusion).
    """

    name:        str = "adaptive_weighted"
    category:    str = "composite"
    lookback:    int = 120
    signal_type: str = "continuous"

    def __init__(
        self,
        signals:          List[Signal],
        ic_window:        int   = 60,
        forward_bars:     int   = 5,
        rebalance_every:  int   = 20,
        allow_inversion:  bool  = True,
        min_weight:       float = 0.0,
        std_window:       int   = 60,
    ) -> None:
        _validate_signals(signals)
        self.signals          = signals
        self.ic_window        = ic_window
        self.forward_bars     = forward_bars
        self.rebalance_every  = rebalance_every
        self.allow_inversion  = allow_inversion
        self.min_weight       = min_weight
        self.std_window       = std_window
        self.lookback         = max(
            max(sig.lookback for sig in signals),
            ic_window + forward_bars
        )

    def _compute_ic(
        self,
        signal_vals: np.ndarray,
        fwd_returns: np.ndarray,
    ) -> float:
        """Spearman IC over a window."""
        valid = ~(np.isnan(signal_vals) | np.isnan(fwd_returns))
        if valid.sum() < 10:
            return 0.0
        try:
            ic, _ = sp_stats.spearmanr(signal_vals[valid], fwd_returns[valid])
            return float(ic) if not np.isnan(ic) else 0.0
        except Exception:
            return 0.0

    def _solve_weights(
        self,
        ic_values: np.ndarray,
        allow_inversion: bool,
        min_weight: float,
    ) -> np.ndarray:
        """
        Convert IC values to positive weights.

        Strategy:
          - If allow_inversion: abs(IC) is the raw weight; sign is stored
            to flip the signal before weighting.
          - Zero or near-zero IC signals get min_weight.
          - Normalise to sum = 1.
        """
        weights = np.abs(ic_values)
        weights = np.maximum(weights, min_weight)
        total   = weights.sum()
        if total == 0:
            return np.ones(len(ic_values)) / len(ic_values)
        return weights / total

    def compute(self, df: pd.DataFrame) -> pd.Series:
        n = len(df)
        if n < self.lookback:
            return pd.Series(np.nan, index=df.index, name=self.name)

        # Compute all sub-signals
        raw = _compute_all(self.signals, df)

        # Forward returns
        close   = df["Close"]
        fwd_ret = np.log(close.shift(-self.forward_bars) / close).values

        # Standardise each signal
        std_signals: Dict[str, np.ndarray] = {}
        for sig in self.signals:
            vals = raw[sig.name].values.copy()
            mu   = pd.Series(vals).rolling(self.std_window, min_periods=2).mean().values
            std  = pd.Series(vals).rolling(self.std_window, min_periods=2).std().values
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                z = (vals - mu) / np.where(std > 0, std, np.nan)
            std_signals[sig.name] = np.nan_to_num(z, nan=0.0)

        n_sigs   = len(self.signals)
        result   = np.zeros(n, dtype=float)
        weights  = np.ones(n_sigs) / n_sigs  # equal weights initially
        ic_signs = np.ones(n_sigs)           # inversion factors (+1 or -1)

        for i in range(self.ic_window, n):
            # Recompute weights every rebalance_every bars
            if (i - self.ic_window) % self.rebalance_every == 0:
                ic_vals = np.zeros(n_sigs)
                for j, sig in enumerate(self.signals):
                    sv = std_signals[sig.name][i - self.ic_window: i]
                    fr = fwd_ret[i - self.ic_window: i]
                    ic = self._compute_ic(sv, fr)
                    ic_vals[j] = ic

                if self.allow_inversion:
                    ic_signs = np.where(ic_vals < 0, -1.0, 1.0)
                else:
                    ic_signs = np.ones(n_sigs)

                weights = self._solve_weights(
                    ic_vals, self.allow_inversion, self.min_weight
                )

            # Compute weighted signal at bar i
            bar_val = 0.0
            for j, sig in enumerate(self.signals):
                bar_val += weights[j] * ic_signs[j] * std_signals[sig.name][i]

            result[i] = bar_val

        # Set warmup to NaN
        result[: self.ic_window] = np.nan

        output = pd.Series(result, index=df.index, name=self.name)
        return output

    def compute_result(self, df: pd.DataFrame) -> SignalResult:
        values = self.compute(df)
        return SignalResult(
            values=values,
            signal_name=self.name,
            category=self.category,
            metadata={
                "n_signals":       len(self.signals),
                "ic_window":       self.ic_window,
                "forward_bars":    self.forward_bars,
                "rebalance_every": self.rebalance_every,
                "allow_inversion": self.allow_inversion,
                "signal_names":    [s.name for s in self.signals],
                "lookback":        self.lookback,
                "signal_type":     self.signal_type,
            },
        )
