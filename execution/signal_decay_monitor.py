"""
execution/signal_decay_monitor.py -- Signal quality monitoring with auto-retirement.

Tracks per-signal rolling IC and ICIR, fits exponential decay curves,
and manages signal lifecycle (active -> probation -> retired -> restored).

Classes
-------
  SignalRecord      : dataclass holding per-signal history
  DecayModel        : OLS-based exponential decay / half-life estimator
  SignalDecayMonitor: main manager with probation / retirement / restoration logic
"""

from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SignalStatus(str, Enum):
    ACTIVE = "active"
    PROBATION = "probation"
    RETIRED = "retired"


# ---------------------------------------------------------------------------
# SignalRecord
# ---------------------------------------------------------------------------

@dataclass
class SignalRecord:
    """
    Persistent record for one signal's quality history.

    Fields
    ------
    signal_name    : str
    ic_history     : deque of daily IC values
    icir_history   : deque of rolling ICIR snapshots
    status         : SignalStatus
    last_updated   : unix timestamp of last IC update
    probation_days : consecutive days with ICIR < 0.25
    retired_days   : consecutive days with ICIR < 0.20
    recovery_days  : consecutive days post-retirement with ICIR > 0.35
    """
    signal_name: str
    ic_history: Deque[float] = field(default_factory=lambda: deque(maxlen=90))
    icir_history: Deque[float] = field(default_factory=lambda: deque(maxlen=90))
    status: SignalStatus = SignalStatus.ACTIVE
    last_updated: float = field(default_factory=time.time)
    probation_days: int = 0
    retired_days: int = 0
    recovery_days: int = 0

    def icir(self, window: int = 30) -> float:
        """Rolling ICIR over the last `window` days."""
        hist = list(self.ic_history)[-window:]
        if len(hist) < 5:
            return 0.0
        arr = np.array(hist)
        std = arr.std()
        if std < 1e-9:
            return 0.0
        return float(arr.mean() / std)

    def mean_ic(self, window: int = 30) -> float:
        hist = list(self.ic_history)[-window:]
        if not hist:
            return 0.0
        return float(np.mean(hist))


# ---------------------------------------------------------------------------
# DecayModel
# ---------------------------------------------------------------------------

class DecayModel:
    """
    Fits an exponential decay model on IC(lag) data via OLS on:
        log|IC(lag)| = alpha + beta * lag

    Half-life = -log(2) / beta  (number of lags until IC halves).

    Usage
    -----
    model = DecayModel()
    model.fit(lags, ic_values)
    hl = model.half_life
    """

    def __init__(self) -> None:
        self.alpha: float = 0.0       # intercept
        self.beta: float = -0.01      # slope (should be negative for decay)
        self.half_life: float = float("inf")
        self.r_squared: float = 0.0
        self.fitted: bool = False

    def fit(self, lags: List[int], ic_values: List[float]) -> bool:
        """
        Fit the decay model.

        Parameters
        ----------
        lags      : list of lag integers (e.g. [1,2,...,20])
        ic_values : IC at each lag (Spearman correlation)

        Returns
        -------
        bool -- True if fit succeeded, False if insufficient data
        """
        if len(lags) < 5 or len(ic_values) < 5:
            return False

        x = np.array(lags, dtype=float)
        y_raw = np.array(ic_values, dtype=float)

        # Filter out zero or near-zero ICs (log undefined)
        mask = np.abs(y_raw) > 1e-6
        x = x[mask]
        y_raw = y_raw[mask]

        if len(x) < 4:
            return False

        y = np.log(np.abs(y_raw))

        try:
            slope, intercept, r, _, _ = scipy_stats.linregress(x, y)
        except Exception as exc:
            logger.debug("DecayModel.fit failed: %s", exc)
            return False

        self.alpha = float(intercept)
        self.beta = float(slope)
        self.r_squared = float(r ** 2)
        self.fitted = True

        if slope < -1e-9:
            self.half_life = float(-math.log(2) / slope)
        else:
            # No decay detected -- set very long half-life
            self.half_life = 1000.0

        return True

    def predict_ic(self, lag: int) -> float:
        """Predict IC magnitude at a given lag."""
        return math.exp(self.alpha + self.beta * lag)

    def __repr__(self) -> str:
        return (
            f"DecayModel(alpha={self.alpha:.4f}, beta={self.beta:.4f}, "
            f"half_life={self.half_life:.1f}, R2={self.r_squared:.3f})"
        )


# ---------------------------------------------------------------------------
# SignalDecayMonitor
# ---------------------------------------------------------------------------

class SignalDecayMonitor:
    """
    Monitors signal quality over time and manages signal lifecycle.

    Lifecycle transitions
    ---------------------
    ACTIVE:
      - If ICIR < 0.25 for 14+ consecutive days -> PROBATION
    PROBATION:
      - If ICIR recovers >= 0.25 -> ACTIVE (probation_days reset)
      - If ICIR < 0.20 for 30+ consecutive days total -> RETIRED
    RETIRED:
      - If ICIR > 0.35 for 5+ consecutive days -> restored to PROBATION

    Parameters
    ----------
    ic_window          : int -- window for ICIR computation (default 30)
    probation_icir     : float -- ICIR threshold triggering probation (0.25)
    probation_days     : int  -- consecutive days below threshold for probation (14)
    retirement_icir    : float -- ICIR threshold triggering retirement (0.20)
    retirement_days    : int  -- consecutive days below threshold for retirement (30)
    restoration_icir   : float -- ICIR threshold for restoration from retirement (0.35)
    restoration_days   : int  -- consecutive days above threshold for restoration (5)
    decay_lag_max      : int  -- number of lags used in decay curve fitting (20)
    """

    PROBATION_ICIR = 0.25
    PROBATION_DAYS = 14
    RETIREMENT_ICIR = 0.20
    RETIREMENT_DAYS = 30
    RESTORATION_ICIR = 0.35
    RESTORATION_DAYS = 5
    DECAY_LAG_MAX = 20
    IC_WINDOW = 30

    def __init__(
        self,
        ic_window: int = IC_WINDOW,
        probation_icir: float = PROBATION_ICIR,
        probation_days: int = PROBATION_DAYS,
        retirement_icir: float = RETIREMENT_ICIR,
        retirement_days: int = RETIREMENT_DAYS,
        restoration_icir: float = RESTORATION_ICIR,
        restoration_days: int = RESTORATION_DAYS,
        decay_lag_max: int = DECAY_LAG_MAX,
    ) -> None:
        self._ic_window = ic_window
        self._probation_icir = probation_icir
        self._probation_days = probation_days
        self._retirement_icir = retirement_icir
        self._retirement_days = retirement_days
        self._restoration_icir = restoration_icir
        self._restoration_days = restoration_days
        self._decay_lag_max = decay_lag_max

        self._records: Dict[str, SignalRecord] = {}
        self._decay_models: Dict[str, DecayModel] = {}

        # Per-signal lag-IC buffer for decay fitting:
        # signal_name -> list of (lag, ic) tuples
        self._lag_ic_buffer: Dict[str, List[Tuple[int, float]]] = {}

    # ------------------------------------------------------------------
    # Signal registration
    # ------------------------------------------------------------------

    def register_signal(self, name: str) -> None:
        """Register a new signal for monitoring."""
        if name not in self._records:
            self._records[name] = SignalRecord(signal_name=name)
            self._decay_models[name] = DecayModel()
            self._lag_ic_buffer[name] = []
            logger.info("Registered signal '%s' for decay monitoring.", name)

    # ------------------------------------------------------------------
    # IC update (call daily)
    # ------------------------------------------------------------------

    def update_ic(
        self,
        signal_name: str,
        signal_history: List[float],
        forward_returns: List[float],
        lag: int = 1,
    ) -> Optional[float]:
        """
        Compute Spearman IC between signal_history and forward_returns,
        record it, and update the signal's lifecycle state.

        Parameters
        ----------
        signal_name     : str
        signal_history  : list of signal values (aligned with forward_returns)
        forward_returns : list of realized returns one period ahead
        lag             : int -- the prediction horizon used (for decay fitting)

        Returns
        -------
        float IC or None if insufficient data
        """
        if signal_name not in self._records:
            self.register_signal(signal_name)

        if len(signal_history) < 5 or len(forward_returns) < 5:
            return None

        n = min(len(signal_history), len(forward_returns))
        sig_arr = np.array(signal_history[-n:])
        ret_arr = np.array(forward_returns[-n:])

        ic_val, _ = scipy_stats.spearmanr(sig_arr, ret_arr)
        if math.isnan(ic_val):
            ic_val = 0.0

        rec = self._records[signal_name]
        rec.ic_history.append(float(ic_val))
        rec.last_updated = time.time()

        # Compute rolling ICIR and store
        icir_val = rec.icir(window=self._ic_window)
        rec.icir_history.append(icir_val)

        # Lag-IC buffer for decay curve
        buf = self._lag_ic_buffer[signal_name]
        buf.append((lag, float(ic_val)))
        if len(buf) > self._decay_lag_max * 3:
            self._lag_ic_buffer[signal_name] = buf[-self._decay_lag_max * 3:]

        # Fit decay model periodically
        if len(rec.ic_history) >= 10:
            self._fit_decay(signal_name)

        # Update lifecycle
        self._update_lifecycle(signal_name, icir_val)

        return float(ic_val)

    def update_ic_bulk(
        self,
        signal_name: str,
        ic_series: List[float],
    ) -> None:
        """
        Bulk-load pre-computed IC values (e.g. from historical backtest).
        Lifecycle transitions are applied for each entry.

        Parameters
        ----------
        signal_name : str
        ic_series   : list of IC floats in chronological order
        """
        if signal_name not in self._records:
            self.register_signal(signal_name)

        for ic_val in ic_series:
            rec = self._records[signal_name]
            rec.ic_history.append(float(ic_val))
            icir_val = rec.icir(window=self._ic_window)
            rec.icir_history.append(icir_val)
            self._update_lifecycle(signal_name, icir_val)

    # ------------------------------------------------------------------
    # Lifecycle management
    # ------------------------------------------------------------------

    def _update_lifecycle(self, signal_name: str, icir_val: float) -> None:
        rec = self._records[signal_name]
        prev_status = rec.status

        if rec.status == SignalStatus.ACTIVE:
            if icir_val < self._probation_icir:
                rec.probation_days += 1
            else:
                rec.probation_days = 0

            if rec.probation_days >= self._probation_days:
                rec.status = SignalStatus.PROBATION
                logger.warning(
                    "Signal '%s' moved to PROBATION (ICIR=%.3f, days=%d).",
                    signal_name, icir_val, rec.probation_days,
                )

        elif rec.status == SignalStatus.PROBATION:
            if icir_val < self._retirement_icir:
                rec.retired_days += 1
                rec.probation_days += 1
            else:
                # Still in probation but ICIR improving
                rec.retired_days = 0
                if icir_val >= self._probation_icir:
                    rec.probation_days = 0
                    rec.status = SignalStatus.ACTIVE
                    logger.info("Signal '%s' restored to ACTIVE from probation.", signal_name)
                else:
                    rec.probation_days += 1

            if rec.retired_days >= self._retirement_days:
                rec.status = SignalStatus.RETIRED
                logger.warning(
                    "Signal '%s' RETIRED (ICIR=%.3f, days=%d).",
                    signal_name, icir_val, rec.retired_days,
                )

        elif rec.status == SignalStatus.RETIRED:
            if icir_val > self._restoration_icir:
                rec.recovery_days += 1
            else:
                rec.recovery_days = 0

            if rec.recovery_days >= self._restoration_days:
                rec.status = SignalStatus.PROBATION
                rec.retired_days = 0
                rec.recovery_days = 0
                logger.info(
                    "Signal '%s' restored to PROBATION after recovery (ICIR=%.3f).",
                    signal_name, icir_val,
                )

        if rec.status != prev_status:
            logger.info(
                "Signal '%s' status: %s -> %s", signal_name, prev_status.value, rec.status.value
            )

    # ------------------------------------------------------------------
    # Decay model fitting
    # ------------------------------------------------------------------

    def _fit_decay(self, signal_name: str) -> None:
        """
        Fit the exponential decay model using the lag-IC buffer.
        Groups ICs by lag and uses the median IC per lag.
        """
        buf = self._lag_ic_buffer.get(signal_name, [])
        if not buf:
            return

        # Group by lag
        lag_groups: Dict[int, List[float]] = {}
        for lag, ic in buf:
            lag_groups.setdefault(lag, []).append(ic)

        lags = sorted(lag_groups.keys())
        if len(lags) < 4:
            return

        ic_medians = [float(np.median(lag_groups[l])) for l in lags]
        model = self._decay_models[signal_name]
        model.fit(lags, ic_medians)

    def fit_decay_from_lag_profile(
        self, signal_name: str, lags: List[int], ic_values: List[float]
    ) -> DecayModel:
        """
        External API: fit decay model directly from a lag profile.

        Parameters
        ----------
        signal_name : str
        lags        : list of lag integers
        ic_values   : IC at each lag

        Returns
        -------
        DecayModel (also stored internally)
        """
        if signal_name not in self._records:
            self.register_signal(signal_name)
        model = self._decay_models[signal_name]
        model.fit(lags, ic_values)
        return model

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_active_signals(self) -> List[str]:
        """Returns signal names that are not retired."""
        return [
            name
            for name, rec in self._records.items()
            if rec.status != SignalStatus.RETIRED
        ]

    def get_signal_status(self, signal_name: str) -> Optional[SignalStatus]:
        rec = self._records.get(signal_name)
        return rec.status if rec else None

    def is_active(self, signal_name: str) -> bool:
        return self.get_signal_status(signal_name) not in (
            SignalStatus.RETIRED, None
        )

    def get_signal_ic(self, signal_name: str, window: int = 30) -> float:
        rec = self._records.get(signal_name)
        return rec.mean_ic(window) if rec else 0.0

    def get_signal_icir(self, signal_name: str, window: int = 30) -> float:
        rec = self._records.get(signal_name)
        return rec.icir(window) if rec else 0.0

    def get_decay_model(self, signal_name: str) -> Optional[DecayModel]:
        return self._decay_models.get(signal_name)

    def get_report(self) -> Dict[str, Dict]:
        """
        Returns full status per signal as a nested dict.

        Structure
        ---------
        {
          signal_name: {
            "status": str,
            "mean_ic_30": float,
            "icir_30": float,
            "probation_days": int,
            "retired_days": int,
            "recovery_days": int,
            "ic_history_len": int,
            "last_updated": float,
            "decay_half_life": float or None,
            "decay_r2": float or None,
          }
        }
        """
        report = {}
        for name, rec in self._records.items():
            decay = self._decay_models.get(name)
            report[name] = {
                "status": rec.status.value,
                "mean_ic_30": round(rec.mean_ic(30), 5),
                "icir_30": round(rec.icir(30), 5),
                "probation_days": rec.probation_days,
                "retired_days": rec.retired_days,
                "recovery_days": rec.recovery_days,
                "ic_history_len": len(rec.ic_history),
                "last_updated": rec.last_updated,
                "decay_half_life": (
                    round(decay.half_life, 2) if decay and decay.fitted else None
                ),
                "decay_r2": (
                    round(decay.r_squared, 4) if decay and decay.fitted else None
                ),
            }
        return report

    def force_retire(self, signal_name: str) -> None:
        """Manually retire a signal regardless of ICIR."""
        if signal_name in self._records:
            self._records[signal_name].status = SignalStatus.RETIRED
            logger.info("Signal '%s' manually retired.", signal_name)

    def force_activate(self, signal_name: str) -> None:
        """Manually restore a signal to active status."""
        if signal_name in self._records:
            rec = self._records[signal_name]
            rec.status = SignalStatus.ACTIVE
            rec.probation_days = 0
            rec.retired_days = 0
            rec.recovery_days = 0
            logger.info("Signal '%s' manually activated.", signal_name)

    def remove_signal(self, signal_name: str) -> None:
        """Completely remove a signal from monitoring."""
        self._records.pop(signal_name, None)
        self._decay_models.pop(signal_name, None)
        self._lag_ic_buffer.pop(signal_name, None)

    @property
    def all_signals(self) -> List[str]:
        return list(self._records.keys())

    @property
    def retired_signals(self) -> List[str]:
        return [n for n, r in self._records.items() if r.status == SignalStatus.RETIRED]

    @property
    def probation_signals(self) -> List[str]:
        return [n for n, r in self._records.items() if r.status == SignalStatus.PROBATION]


__all__ = [
    "SignalStatus",
    "SignalRecord",
    "DecayModel",
    "SignalDecayMonitor",
]
