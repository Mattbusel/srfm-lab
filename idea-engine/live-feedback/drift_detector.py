"""
live-feedback/drift_detector.py
================================
Statistical drift detection for live vs backtest behaviour.

Implements three complementary change-point / distribution-shift tests:

* **Page-Hinkley**  — sequential, low-latency, detects mean shifts in a stream.
* **CUSUM**         — cumulative sum control chart, detects sustained shifts.
* **Kolmogorov-Smirnov** — distribution test, detects any shape change between
  backtest and live return distributions.

Additionally provides higher-level monitors that apply the above tests to
signals, features, and regime distributions, firing alerts to
``narrative_alerts`` when drift is detected.

All methods are pure Python / NumPy — no external ML dependencies.
"""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DriftReport:
    """
    Full diagnostic output from a drift check on a single metric or signal.

    Attributes
    ----------
    name           : identifier of the signal / feature being tested
    drift_detected : whether any test flagged drift
    ph_drift       : Page-Hinkley test result
    cusum_drift    : CUSUM test result
    ks_drift       : KS test result (if backtest values supplied)
    change_point   : estimated index of the change point (CUSUM), or None
    cusum_stat     : final CUSUM statistic value
    ph_stat        : final Page-Hinkley cumulative stat
    ks_statistic   : KS D-statistic
    ks_p_value     : KS p-value
    alert_fired    : whether a narrative alert was inserted
    checked_at     : UTC ISO-8601 timestamp
    """

    name: str
    drift_detected: bool
    ph_drift: bool
    cusum_drift: bool
    ks_drift: bool
    change_point: int | None
    cusum_stat: float
    ph_stat: float
    ks_statistic: float
    ks_p_value: float
    alert_fired: bool = False
    checked_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# DriftDetector
# ---------------------------------------------------------------------------


class DriftDetector:
    """
    Detects when live trading behaviour diverges from backtest expectations.

    Parameters
    ----------
    iae_conn : sqlite3.Connection
        Open connection to ``idea_engine.db`` for writing alerts.
    """

    def __init__(self, iae_conn: sqlite3.Connection) -> None:
        self._conn = iae_conn

    # ------------------------------------------------------------------
    # Core statistical tests
    # ------------------------------------------------------------------

    def page_hinkley_test(
        self,
        observations: list[float] | np.ndarray,
        threshold: float = 50.0,
        delta: float = 0.005,
    ) -> bool:
        """
        Page-Hinkley sequential change-point test.

        Detects a persistent upward shift in the mean of ``observations``.
        Run on the *negated* sequence to test for downward shifts.

        The test statistic ``M_t`` is defined as::

            x̄_t  = (1/t) Σ x_i
            U_t  = Σ (x_i − x̄_t − δ)        (cumulative sum)
            m_t  = min U_s  for s ∈ [1, t]
            M_t  = U_t − m_t

        Drift is detected when ``M_t > threshold``.

        Parameters
        ----------
        observations : sequence of floats
        threshold    : detection threshold (higher → less sensitive)
        delta        : minimum change magnitude to detect

        Returns
        -------
        bool — True if drift detected at any point in the sequence.
        """
        obs = np.asarray(observations, dtype=float)
        if len(obs) < 2:
            return False

        cumsum = 0.0
        min_cumsum = 0.0
        running_mean = 0.0

        for i, x in enumerate(obs, start=1):
            running_mean += (x - running_mean) / i
            cumsum += x - running_mean - delta
            if cumsum < min_cumsum:
                min_cumsum = cumsum
            ph_stat = cumsum - min_cumsum
            if ph_stat > threshold:
                return True

        return False

    def cusum_test(
        self,
        observations: list[float] | np.ndarray,
        k: float = 0.5,
        h: float = 5.0,
    ) -> tuple[bool, int | None]:
        """
        Two-sided CUSUM (cumulative sum) control chart.

        Detects sustained upward or downward shifts in the mean, estimated
        using a reference value of ``μ ± k·σ``.

        The positive and negative statistics are::

            C⁺_t = max(0,  C⁺_{t-1} + x_t − μ − k·σ)
            C⁻_t = max(0,  C⁻_{t-1} − x_t + μ − k·σ)

        Drift is declared when either statistic exceeds ``h·σ``.

        Parameters
        ----------
        observations : sequence of floats
        k            : allowance parameter (slack, in σ units)
        h            : decision threshold (in σ units)

        Returns
        -------
        (drift_detected: bool, change_point: int | None)
            ``change_point`` is the index at which drift was first detected.
        """
        obs = np.asarray(observations, dtype=float)
        if len(obs) < 2:
            return False, None

        mu = float(np.mean(obs))
        sigma = float(np.std(obs, ddof=1))
        if sigma == 0:
            return False, None

        c_pos = 0.0
        c_neg = 0.0
        boundary = h * sigma
        slack = k * sigma

        for i, x in enumerate(obs):
            c_pos = max(0.0, c_pos + (x - mu) - slack)
            c_neg = max(0.0, c_neg - (x - mu) - slack)
            if c_pos > boundary or c_neg > boundary:
                return True, i

        return False, None

    def kolmogorov_smirnov_drift(
        self,
        backtest_returns: list[float] | np.ndarray,
        live_returns: list[float] | np.ndarray,
        alpha: float = 0.05,
    ) -> bool:
        """
        Two-sample Kolmogorov-Smirnov test for distribution shift.

        Tests whether ``backtest_returns`` and ``live_returns`` are drawn
        from the same distribution.  Does not assume normality.

        This is a pure-Python implementation of the two-sample KS statistic
        that avoids any scipy dependency::

            D = max |F_n(x) − G_m(x)|

        The critical value at significance ``α`` uses the approximation::

            c(α) = sqrt(−ln(α/2) / 2)
            D_crit = c(α) · sqrt((n + m) / (n · m))

        Parameters
        ----------
        backtest_returns : sequence of floats
        live_returns     : sequence of floats
        alpha            : significance level (default 0.05)

        Returns
        -------
        bool — True if the null hypothesis (same distribution) is rejected.
        """
        bt = np.sort(np.asarray(backtest_returns, dtype=float))
        live = np.sort(np.asarray(live_returns, dtype=float))
        n, m = len(bt), len(live)
        if n < 2 or m < 2:
            return False

        # Compute the KS D-statistic via sorted merge
        combined = np.sort(np.concatenate([bt, live]))
        cdf_bt = np.searchsorted(bt, combined, side="right") / n
        cdf_live = np.searchsorted(live, combined, side="right") / m
        ks_stat = float(np.max(np.abs(cdf_bt - cdf_live)))

        # Critical value (Smirnov approximation)
        c_alpha = math.sqrt(-math.log(alpha / 2) / 2)
        d_crit = c_alpha * math.sqrt((n + m) / (n * m))

        return ks_stat > d_crit

    # ------------------------------------------------------------------
    # Higher-level monitors
    # ------------------------------------------------------------------

    def monitor_signal_drift(
        self,
        signal_name: str,
        backtest_values: list[float] | np.ndarray,
        live_values: list[float] | np.ndarray,
        ph_threshold: float = 50.0,
        ph_delta: float = 0.005,
        cusum_k: float = 0.5,
        cusum_h: float = 5.0,
        ks_alpha: float = 0.05,
    ) -> DriftReport:
        """
        Run all three drift tests on a named signal and return a
        comprehensive ``DriftReport``.

        Parameters
        ----------
        signal_name      : human-readable name for the signal
        backtest_values  : historical (in-sample) signal values
        live_values      : recent live signal values
        ph_threshold     : Page-Hinkley detection threshold
        ph_delta         : Page-Hinkley minimum change magnitude
        cusum_k          : CUSUM allowance parameter
        cusum_h          : CUSUM decision threshold
        ks_alpha         : KS test significance level

        Returns
        -------
        DriftReport
        """
        live_arr = np.asarray(live_values, dtype=float)
        bt_arr = np.asarray(backtest_values, dtype=float)

        # Page-Hinkley (upward and downward)
        ph_up = self.page_hinkley_test(live_arr, threshold=ph_threshold, delta=ph_delta)
        ph_down = self.page_hinkley_test(-live_arr, threshold=ph_threshold, delta=ph_delta)
        ph_drift = ph_up or ph_down

        # CUSUM
        cusum_drift, change_point = self.cusum_test(live_arr, k=cusum_k, h=cusum_h)

        # KS (only if backtest values supplied)
        ks_drift = False
        ks_stat = 0.0
        if len(bt_arr) >= 2 and len(live_arr) >= 2:
            ks_drift = self.kolmogorov_smirnov_drift(bt_arr, live_arr, alpha=ks_alpha)
            # Re-compute the raw statistic for the report
            ks_stat = self._ks_statistic(bt_arr, live_arr)

        # Compute summary stats for the report
        cusum_stat_val = self._cusum_final_stat(live_arr, cusum_k)
        ph_stat_val = self._ph_final_stat(live_arr, ph_delta)

        any_drift = ph_drift or cusum_drift or ks_drift

        # Approximate p-value for KS (Smirnov formula)
        n, m = len(bt_arr), len(live_arr)
        ks_p = 1.0
        if n >= 2 and m >= 2 and ks_stat > 0:
            en = math.sqrt(n * m / (n + m))
            ks_p = max(0.0, min(1.0, 2 * math.exp(-2 * (en * ks_stat) ** 2)))

        report = DriftReport(
            name=signal_name,
            drift_detected=any_drift,
            ph_drift=ph_drift,
            cusum_drift=cusum_drift,
            ks_drift=ks_drift,
            change_point=change_point,
            cusum_stat=cusum_stat_val,
            ph_stat=ph_stat_val,
            ks_statistic=ks_stat,
            ks_p_value=ks_p,
        )

        if any_drift:
            self._fire_drift_alert(report)

        return report

    def feature_drift_scan(
        self,
        feature_store: dict[str, tuple[list[float], list[float]]],
        lookback_days: int = 30,
    ) -> dict[str, DriftReport]:
        """
        Scan all features in ``feature_store`` for drift and return a report
        for each.

        Parameters
        ----------
        feature_store : dict
            Mapping ``{feature_name: (backtest_values, live_values)}``.
            Each value is a tuple of two float lists.
        lookback_days : int
            Informational — used in alert payloads only; filtering is the
            caller's responsibility before passing values in.

        Returns
        -------
        dict  {feature_name: DriftReport}
        """
        reports: dict[str, DriftReport] = {}
        for name, (bt_vals, live_vals) in feature_store.items():
            try:
                report = self.monitor_signal_drift(
                    signal_name=name,
                    backtest_values=bt_vals,
                    live_values=live_vals,
                )
                reports[name] = report
            except Exception:
                logger.exception("Drift scan failed for feature '%s'.", name)
        logger.info(
            "Feature drift scan complete: %d/%d features show drift.",
            sum(r.drift_detected for r in reports.values()),
            len(reports),
        )
        return reports

    def regime_drift(
        self,
        backtest_regime_dist: dict[str, float],
        live_regime_dist: dict[str, float],
        tv_distance_threshold: float = 0.20,
    ) -> bool:
        """
        Detect whether the live regime distribution differs materially from
        the backtest regime distribution using Total-Variation distance.

        TV distance = (1/2) Σ |p_i − q_i|

        Drift is declared when TV distance > ``tv_distance_threshold``.

        Parameters
        ----------
        backtest_regime_dist : dict  {regime: probability}
        live_regime_dist     : dict  {regime: probability}
        tv_distance_threshold : float  (default 0.20)

        Returns
        -------
        bool — True if regime drift detected.
        """
        all_regimes = set(backtest_regime_dist.keys()) | set(live_regime_dist.keys())
        if not all_regimes:
            return False

        tv = 0.0
        for regime in all_regimes:
            p = backtest_regime_dist.get(regime, 0.0)
            q = live_regime_dist.get(regime, 0.0)
            tv += abs(p - q)
        tv /= 2.0

        drift_detected = tv > tv_distance_threshold

        if drift_detected:
            payload = json.dumps(
                {
                    "drift_type": "regime_distribution",
                    "tv_distance": round(tv, 4),
                    "threshold": tv_distance_threshold,
                    "backtest_dist": backtest_regime_dist,
                    "live_dist": live_regime_dist,
                    "detected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            )
            self._insert_alert("regime_drift_detected", payload, severity="warning")

        return drift_detected

    # ------------------------------------------------------------------
    # Alert helpers
    # ------------------------------------------------------------------

    def _fire_drift_alert(self, report: DriftReport) -> None:
        """
        Insert a drift alert into ``narrative_alerts`` (if the table exists)
        and into ``event_log`` as a fallback.
        """
        payload = json.dumps(report.to_dict())
        alert_type = f"signal_drift_detected:{report.name}"

        # Try narrative_alerts first
        try:
            self._conn.execute(
                """
                INSERT INTO narrative_alerts
                    (alert_type, payload_json, severity, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    alert_type,
                    payload,
                    "warning",
                    report.checked_at,
                ),
            )
            self._conn.commit()
            report.alert_fired = True
            return
        except sqlite3.OperationalError:
            pass

        # Fall back to event_log
        self._insert_alert(alert_type, payload, severity="warning")
        report.alert_fired = True

    def _insert_alert(
        self,
        event_type: str,
        payload: str,
        severity: str = "warning",
    ) -> None:
        """Write a generic alert to ``event_log``."""
        try:
            self._conn.execute(
                "INSERT INTO event_log (event_type, payload_json, severity) VALUES (?, ?, ?)",
                (event_type, payload, severity),
            )
            self._conn.commit()
        except Exception:
            logger.warning("Could not write alert '%s' to event_log.", event_type)

    # ------------------------------------------------------------------
    # Internal stat helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ks_statistic(
        bt: np.ndarray,
        live: np.ndarray,
    ) -> float:
        """Compute the two-sample KS D-statistic."""
        bt_s = np.sort(bt)
        live_s = np.sort(live)
        n, m = len(bt_s), len(live_s)
        combined = np.sort(np.concatenate([bt_s, live_s]))
        cdf_bt = np.searchsorted(bt_s, combined, side="right") / n
        cdf_live = np.searchsorted(live_s, combined, side="right") / m
        return float(np.max(np.abs(cdf_bt - cdf_live)))

    @staticmethod
    def _cusum_final_stat(obs: np.ndarray, k: float) -> float:
        """Return the maximum CUSUM statistic across both tails."""
        if len(obs) < 2:
            return 0.0
        mu = float(np.mean(obs))
        sigma = float(np.std(obs, ddof=1)) or 1.0
        slack = k * sigma
        c_pos = c_neg = 0.0
        max_stat = 0.0
        for x in obs:
            c_pos = max(0.0, c_pos + (x - mu) - slack)
            c_neg = max(0.0, c_neg - (x - mu) - slack)
            max_stat = max(max_stat, c_pos, c_neg)
        return max_stat

    @staticmethod
    def _ph_final_stat(obs: np.ndarray, delta: float) -> float:
        """Return the final Page-Hinkley statistic value."""
        if len(obs) < 2:
            return 0.0
        cumsum = 0.0
        min_cumsum = 0.0
        running_mean = 0.0
        max_ph = 0.0
        for i, x in enumerate(obs, start=1):
            running_mean += (x - running_mean) / i
            cumsum += x - running_mean - delta
            if cumsum < min_cumsum:
                min_cumsum = cumsum
            max_ph = max(max_ph, cumsum - min_cumsum)
        return max_ph
