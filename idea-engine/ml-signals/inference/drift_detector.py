"""
inference/drift_detector.py
============================
Feature drift detection using Population Stability Index (PSI).

Financial rationale
-------------------
A machine-learned signal is trained on a historical distribution of
feature values.  When that distribution shifts in production (e.g.
because volatility doubles after a crisis event), the model's output is
no longer calibrated to the new environment.  The signal becomes "stale"
and may actually be harmful if traded.

Population Stability Index (PSI)
---------------------------------
PSI is borrowed from credit risk modelling.  For each feature, we
compare a reference distribution (training data) to a current
distribution (recent live data) using:

    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)

PSI interpretation (industry convention):
    PSI < 0.10 → No significant shift     (green)
    PSI < 0.20 → Moderate shift           (yellow) – monitor
    PSI >= 0.20 → Significant shift       (red)    – retrain required

If PSI >= 0.20 for any *key feature*, DriftDetector:
1. Marks the model as stale.
2. Emits a retrain trigger event (a dict with metadata).
3. Optionally calls a user-supplied ``on_drift`` callback.

The detector tracks a rolling window of recent feature values and
compares them to a reference window established at training time.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Set

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PSI_MODERATE  = 0.10
PSI_CRITICAL  = 0.20
N_BINS        = 10
MIN_SAMPLES   = 50    # minimum samples needed for reliable PSI


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DriftEvent:
    """Emitted when significant feature drift is detected."""
    timestamp:     float
    instrument:    str
    features:      Dict[str, float]   # feature → PSI value
    max_psi:       float
    drifted_feats: List[str]          # features with PSI >= critical threshold
    severity:      str                # 'MODERATE' or 'CRITICAL'
    retrain_flag:  bool
    meta:          Dict = field(default_factory=dict)

    def __str__(self) -> str:
        drifted = ", ".join(f"{f}={v:.3f}" for f, v in self.features.items()
                            if v >= PSI_CRITICAL)
        return (
            f"DriftEvent [{self.severity}] instrument={self.instrument} "
            f"max_psi={self.max_psi:.3f} drifted=[{drifted}] "
            f"retrain={self.retrain_flag}"
        )


# ---------------------------------------------------------------------------
# PSI computation
# ---------------------------------------------------------------------------

def _compute_psi(
    reference: np.ndarray,
    current:   np.ndarray,
    n_bins:    int = N_BINS,
    eps:       float = 1e-6,
) -> float:
    """Compute PSI between reference and current distributions.

    Uses reference quantiles as bin edges to ensure meaningful
    comparison (avoids empty bins in reference distribution).

    Parameters
    ----------
    reference : np.ndarray  (training period values)
    current   : np.ndarray  (recent live values)
    n_bins    : int
    eps       : float  smoothing to avoid log(0)

    Returns
    -------
    float  PSI value
    """
    if len(reference) < MIN_SAMPLES or len(current) < MIN_SAMPLES:
        return 0.0

    # Remove NaN
    reference = reference[~np.isnan(reference)]
    current   = current[~np.isnan(current)]

    if len(reference) < 5 or len(current) < 5:
        return 0.0

    # Use reference quantiles as bin edges
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(reference, quantiles)
    # Ensure unique edges
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = (ref_counts / len(reference)) + eps
    cur_pct = (cur_counts / len(current)) + eps

    # Normalise
    ref_pct /= ref_pct.sum()
    cur_pct /= cur_pct.sum()

    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return max(0.0, psi)


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------

class DriftDetector:
    """Monitor feature distributions and trigger model retraining on drift.

    Parameters
    ----------
    reference_window : int
        Number of bars used as the training-time reference distribution.
    monitor_window : int
        Number of recent bars used as the current distribution.
    check_interval : int
        Minimum number of new bars between drift checks.
    psi_critical : float
        PSI threshold above which drift is considered critical (retrain).
    psi_moderate : float
        PSI threshold above which drift is considered moderate (monitor).
    key_features : list[str], optional
        Features to monitor.  If None, all features are monitored.
    on_drift : callable, optional
        Callback called with a DriftEvent when critical drift is detected.
    """

    def __init__(
        self,
        reference_window: int   = 252,
        monitor_window:   int   = 30,
        check_interval:   int   = 10,
        psi_critical:     float = PSI_CRITICAL,
        psi_moderate:     float = PSI_MODERATE,
        key_features:     Optional[List[str]] = None,
        on_drift:         Optional[Callable[[DriftEvent], None]] = None,
    ) -> None:
        self.reference_window = reference_window
        self.monitor_window   = monitor_window
        self.check_interval   = check_interval
        self.psi_critical     = psi_critical
        self.psi_moderate     = psi_moderate
        self.key_features     = key_features
        self.on_drift         = on_drift

        self._reference_df: Optional[pd.DataFrame] = None
        self._live_buffer:  Dict[str, pd.DataFrame] = {}
        self._bar_counts:   Dict[str, int] = {}
        self._stale_flags:  Dict[str, bool] = {}
        self._drift_history: List[DriftEvent] = []
        self._last_psi:      Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Reference setting
    # ------------------------------------------------------------------

    def set_reference(self, df: pd.DataFrame) -> None:
        """Set the reference (training) distribution.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame from the training period.
            Uses the last ``reference_window`` rows.
        """
        self._reference_df = df.tail(self.reference_window).copy()

    # ------------------------------------------------------------------
    # Incremental update
    # ------------------------------------------------------------------

    def update(self, instrument: str, new_bar: pd.DataFrame) -> Optional[DriftEvent]:
        """Append new feature data and check for drift.

        Parameters
        ----------
        instrument : str
        new_bar : pd.DataFrame  single or multiple new rows

        Returns
        -------
        DriftEvent if critical drift detected, else None.
        """
        # Accumulate live buffer
        if instrument not in self._live_buffer:
            self._live_buffer[instrument] = new_bar.tail(self.monitor_window * 2).copy()
            self._bar_counts[instrument]  = 0
        else:
            combined = pd.concat([self._live_buffer[instrument], new_bar])
            self._live_buffer[instrument] = combined.tail(self.monitor_window * 2).copy()

        self._bar_counts[instrument] = self._bar_counts.get(instrument, 0) + len(new_bar)

        # Only check at intervals
        if self._bar_counts[instrument] % self.check_interval != 0:
            return None

        return self._check_drift(instrument)

    def check_now(self, instrument: str, current_df: pd.DataFrame) -> Optional[DriftEvent]:
        """Force a drift check using ``current_df`` as the current distribution."""
        if instrument not in self._live_buffer:
            self._live_buffer[instrument] = current_df.copy()
        else:
            self._live_buffer[instrument] = current_df.copy()
        return self._check_drift(instrument)

    # ------------------------------------------------------------------
    # PSI computation
    # ------------------------------------------------------------------

    def _check_drift(self, instrument: str) -> Optional[DriftEvent]:
        if self._reference_df is None:
            return None

        ref_df = self._reference_df
        cur_df = self._live_buffer[instrument].tail(self.monitor_window)

        if len(cur_df) < MIN_SAMPLES // 2:
            return None

        # Determine features to monitor
        common_cols = [c for c in ref_df.columns if c in cur_df.columns]
        features_to_check = (
            [f for f in (self.key_features or common_cols) if f in common_cols]
        )

        psi_values: Dict[str, float] = {}
        for feat in features_to_check:
            ref_vals = ref_df[feat].values.astype(float)
            cur_vals = cur_df[feat].values.astype(float)
            psi      = _compute_psi(ref_vals, cur_vals)
            psi_values[feat] = psi

        self._last_psi[instrument] = psi_values

        if not psi_values:
            return None

        max_psi = max(psi_values.values())
        drifted_feats = [f for f, p in psi_values.items() if p >= self.psi_critical]

        # Determine severity
        if max_psi >= self.psi_critical:
            severity = "CRITICAL"
            retrain  = True
            self._stale_flags[instrument] = True
        elif max_psi >= self.psi_moderate:
            severity = "MODERATE"
            retrain  = False
        else:
            return None

        event = DriftEvent(
            timestamp     = time.time(),
            instrument    = instrument,
            features      = psi_values,
            max_psi       = max_psi,
            drifted_feats = drifted_feats,
            severity      = severity,
            retrain_flag  = retrain,
        )
        self._drift_history.append(event)

        if retrain and self.on_drift is not None:
            try:
                self.on_drift(event)
            except Exception:
                pass

        return event

    # ------------------------------------------------------------------
    # Status queries
    # ------------------------------------------------------------------

    def is_stale(self, instrument: str) -> bool:
        """Return True if the model for ``instrument`` should be retrained."""
        return self._stale_flags.get(instrument, False)

    def clear_stale(self, instrument: str) -> None:
        """Clear the stale flag after a successful retrain."""
        self._stale_flags[instrument] = False

    def last_psi(self, instrument: str) -> Dict[str, float]:
        """Return the most recent PSI values for ``instrument``."""
        return dict(self._last_psi.get(instrument, {}))

    def psi_summary(self, instrument: str) -> str:
        """Return a human-readable PSI summary."""
        psi = self.last_psi(instrument)
        if not psi:
            return f"No PSI data for {instrument}"
        lines = [f"PSI summary for {instrument}:"]
        for feat, val in sorted(psi.items(), key=lambda x: -x[1]):
            flag = ("🔴" if val >= self.psi_critical else
                    "🟡" if val >= self.psi_moderate else "🟢")
            lines.append(f"  {flag} {feat:<30} PSI={val:.4f}")
        return "\n".join(lines)

    @property
    def drift_history(self) -> List[DriftEvent]:
        return list(self._drift_history)

    def drift_history_df(self) -> "pd.DataFrame":
        import pandas as pd
        if not self._drift_history:
            return pd.DataFrame()
        rows = []
        for e in self._drift_history:
            row = {
                "timestamp":    pd.Timestamp(e.timestamp, unit="s", tz="UTC"),
                "instrument":   e.instrument,
                "max_psi":      e.max_psi,
                "severity":     e.severity,
                "retrain_flag": e.retrain_flag,
                "n_drifted":    len(e.drifted_feats),
            }
            rows.append(row)
        return pd.DataFrame(rows)
