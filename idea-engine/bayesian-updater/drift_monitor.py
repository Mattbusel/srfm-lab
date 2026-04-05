"""
drift_monitor.py
================
Monitor for regime change and distribution shift in live trading P&L.

Two complementary detection algorithms are implemented:

1. **Kolmogorov-Smirnov (KS) test**
   Compares the empirical CDF of recent trade P&L against the historical
   baseline CDF.  The two-sample KS statistic is sensitive to both
   location and shape differences, making it effective for detecting
   regime changes that shift the full return distribution.

   Trigger: p-value < ks_alpha (default 0.01).

2. **CUSUM (Cumulative Sum) control chart**
   Tracks the cumulative deviation of the observed win rate from the
   historical baseline win rate.  The CUSUM statistic rises when the
   process is systematically above or below target and resets when it
   returns.

   Two-sided CUSUM::

       C_hi[t] = max(0, C_hi[t-1] + (x[t] - mu_0 - k))
       C_lo[t] = max(0, C_lo[t-1] + (mu_0 - k - x[t]))

   where k = 0.5 * shift_to_detect and the control limit H is set to
   give a target in-control ARL (default ARL_0 = 500 observations).

   Trigger: C_hi > H  or  C_lo > H.

When either detector fires, a REGIME_CHANGE alert is emitted and a
full parameter re-estimation from scratch is recommended.

Usage::

    monitor = DriftMonitor(historical_pnl=baseline_array)
    alert = monitor.check(recent_pnl=new_array, recent_win_flags=win_arr)
    if alert:
        print(alert.message)
        updater._computer.reset_to_priors()
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Alert container
# ---------------------------------------------------------------------------

@dataclass
class DriftAlert:
    """
    Emitted when a regime change is detected.

    Attributes
    ----------
    alert_type        : "KS_TEST" or "CUSUM" or "BOTH".
    message           : human-readable description.
    ks_stat           : KS test statistic (if applicable).
    ks_pvalue         : KS test p-value (if applicable).
    cusum_hi          : current high-side CUSUM value.
    cusum_lo          : current low-side CUSUM value.
    cusum_limit       : control limit H.
    recommendation    : "RESET_POSTERIORS" or "INCREASE_JITTER".
    severity          : "WARNING" or "CRITICAL".
    """

    alert_type:      str
    message:         str
    ks_stat:         Optional[float] = None
    ks_pvalue:       Optional[float] = None
    cusum_hi:        float = 0.0
    cusum_lo:        float = 0.0
    cusum_limit:     float = 0.0
    recommendation:  str  = "RESET_POSTERIORS"
    severity:        str  = "WARNING"

    def to_dict(self) -> dict:
        return {
            "alert_type":     self.alert_type,
            "message":        self.message,
            "ks_stat":        self.ks_stat,
            "ks_pvalue":      self.ks_pvalue,
            "cusum_hi":       self.cusum_hi,
            "cusum_lo":       self.cusum_lo,
            "cusum_limit":    self.cusum_limit,
            "recommendation": self.recommendation,
            "severity":       self.severity,
        }


# ---------------------------------------------------------------------------
# CUSUM state
# ---------------------------------------------------------------------------

@dataclass
class CUSUMState:
    """
    Mutable CUSUM control chart state.

    Parameters
    ----------
    baseline_win_rate : expected win rate under the in-control process.
    shift_to_detect   : minimum win-rate shift (in fraction) to detect.
    arl_0             : target in-control average run length.
    """

    baseline_win_rate: float = 0.55
    shift_to_detect:   float = 0.10   # detect a 10 pp shift in win rate
    arl_0:             float = 500.0  # tolerate ~500 obs between false alarms

    C_hi: float = field(default=0.0, init=False)
    C_lo: float = field(default=0.0, init=False)
    n_obs: int  = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.C_hi = 0.0
        self.C_lo = 0.0
        self.n_obs = 0
        # k = half the shift in standard-Normal units mapped back to win-rate
        self.k = self.shift_to_detect / 2.0
        # H calibrated approximately from ARL tables for Normal CUSUM
        # h ≈ 4-5 for common ARL targets; we use ln(ARL_0) as a rough estimate
        self.H = math.log(self.arl_0)

    def update_one(self, win: float) -> None:
        """
        Update the CUSUM with a single binary observation (1=win, 0=loss).

        Parameters
        ----------
        win : 1.0 if the trade won, 0.0 if it lost.
        """
        mu0 = self.baseline_win_rate
        self.C_hi = max(0.0, self.C_hi + (win - mu0 - self.k))
        self.C_lo = max(0.0, self.C_lo + (mu0 - self.k - win))
        self.n_obs += 1

    def update_batch(self, wins: np.ndarray) -> None:
        """Update CUSUM with an array of binary observations."""
        for w in wins:
            self.update_one(float(w))

    def is_triggered(self) -> bool:
        """Return True if either side of the CUSUM has exceeded H."""
        return self.C_hi > self.H or self.C_lo > self.H

    def reset(self) -> None:
        """Reset CUSUM counters (not the baseline)."""
        self.C_hi = 0.0
        self.C_lo = 0.0
        self.n_obs = 0

    def direction(self) -> str:
        """Which side triggered: 'up', 'down', or 'none'."""
        if self.C_hi > self.H:
            return "up"
        if self.C_lo > self.H:
            return "down"
        return "none"


# ---------------------------------------------------------------------------
# DriftMonitor
# ---------------------------------------------------------------------------

class DriftMonitor:
    """
    Monitors live trade P&L for regime changes using KS test and CUSUM.

    Parameters
    ----------
    historical_pnl    : baseline P&L distribution (numpy array of fractional
                        P&L values from historical / backtest period).
    historical_wins   : binary win indicators for historical trades (same length
                        as historical_pnl if provided).
    ks_alpha          : KS test significance level (default 0.01).
    min_recent_trades : minimum recent trades required before running KS test.
    shift_to_detect   : CUSUM shift parameter (win-rate fraction).
    arl_0             : CUSUM target in-control ARL.
    """

    def __init__(
        self,
        historical_pnl: Optional[np.ndarray] = None,
        historical_wins: Optional[np.ndarray] = None,
        ks_alpha: float = 0.01,
        min_recent_trades: int = 30,
        shift_to_detect: float = 0.10,
        arl_0: float = 500.0,
    ):
        self.historical_pnl     = (
            np.asarray(historical_pnl, dtype=float)
            if historical_pnl is not None
            else np.array([])
        )
        self.ks_alpha           = ks_alpha
        self.min_recent_trades  = min_recent_trades

        # Derive baseline win rate from historical data
        baseline_win_rate = 0.55
        if historical_wins is not None and len(historical_wins) > 0:
            baseline_win_rate = float(np.mean(historical_wins > 0))
        elif len(self.historical_pnl) > 0:
            baseline_win_rate = float(np.mean(self.historical_pnl > 0))

        self.cusum = CUSUMState(
            baseline_win_rate=baseline_win_rate,
            shift_to_detect=shift_to_detect,
            arl_0=arl_0,
        )

        # Running buffer of recent P&L for KS comparisons
        self._recent_pnl_buffer: List[float] = []
        self._alert_history:     List[DriftAlert] = []

    # ------------------------------------------------------------------
    # Main check method
    # ------------------------------------------------------------------

    def check(
        self,
        recent_pnl: np.ndarray,
        recent_win_flags: Optional[np.ndarray] = None,
    ) -> Optional[DriftAlert]:
        """
        Run all drift detectors against a batch of recent trades.

        Parameters
        ----------
        recent_pnl       : array of fractional P&L for recent trades.
        recent_win_flags : binary win indicators (1=win).  If None,
                           derived from recent_pnl > 0.

        Returns
        -------
        DriftAlert if a regime change is detected, else None.
        """
        if len(recent_pnl) == 0:
            return None

        recent_pnl = np.asarray(recent_pnl, dtype=float)
        if recent_win_flags is None:
            recent_win_flags = (recent_pnl > 0).astype(float)
        else:
            recent_win_flags = np.asarray(recent_win_flags, dtype=float)

        # Accumulate buffer for KS test
        self._recent_pnl_buffer.extend(recent_pnl.tolist())
        # Keep buffer at most 500 recent trades
        if len(self._recent_pnl_buffer) > 500:
            self._recent_pnl_buffer = self._recent_pnl_buffer[-500:]

        # Update CUSUM
        self.cusum.update_batch(recent_win_flags)

        ks_fired   = False
        cusum_fired = False
        ks_stat    = None
        ks_pvalue  = None

        # KS test (only if we have enough data)
        if (
            len(self._recent_pnl_buffer) >= self.min_recent_trades
            and len(self.historical_pnl) >= self.min_recent_trades
        ):
            ks_stat, ks_pvalue = self._run_ks_test(
                np.array(self._recent_pnl_buffer)
            )
            if ks_pvalue < self.ks_alpha:
                ks_fired = True
                logger.warning(
                    "KS drift detected: stat=%.4f, p=%.4f (threshold=%.4f)",
                    ks_stat, ks_pvalue, self.ks_alpha,
                )

        # CUSUM test
        if self.cusum.is_triggered():
            cusum_fired = True
            logger.warning(
                "CUSUM drift detected: C_hi=%.3f, C_lo=%.3f (H=%.3f), dir=%s",
                self.cusum.C_hi, self.cusum.C_lo, self.cusum.H,
                self.cusum.direction(),
            )

        if not ks_fired and not cusum_fired:
            return None

        # Build alert
        alert_type = (
            "BOTH" if ks_fired and cusum_fired
            else ("KS_TEST" if ks_fired else "CUSUM")
        )
        severity = "CRITICAL" if (ks_fired and cusum_fired) else "WARNING"
        message  = self._build_message(
            ks_fired, cusum_fired, ks_stat, ks_pvalue
        )

        alert = DriftAlert(
            alert_type=alert_type,
            message=message,
            ks_stat=ks_stat,
            ks_pvalue=ks_pvalue,
            cusum_hi=self.cusum.C_hi,
            cusum_lo=self.cusum.C_lo,
            cusum_limit=self.cusum.H,
            recommendation="RESET_POSTERIORS",
            severity=severity,
        )
        self._alert_history.append(alert)

        # Reset CUSUM after firing to avoid continuous alarming
        self.cusum.reset()

        return alert

    # ------------------------------------------------------------------
    # KS test
    # ------------------------------------------------------------------

    def _run_ks_test(
        self, recent: np.ndarray
    ) -> Tuple[float, float]:
        """
        Two-sample KS test between *recent* P&L and historical baseline.

        Returns
        -------
        (ks_statistic, p_value)
        """
        ks_result = stats.ks_2samp(self.historical_pnl, recent)
        return float(ks_result.statistic), float(ks_result.pvalue)

    # ------------------------------------------------------------------
    # Message builder
    # ------------------------------------------------------------------

    def _build_message(
        self,
        ks_fired: bool,
        cusum_fired: bool,
        ks_stat: Optional[float],
        ks_pvalue: Optional[float],
    ) -> str:
        parts = []
        if ks_fired:
            parts.append(
                f"KS test detects P&L distribution shift "
                f"(stat={ks_stat:.4f}, p={ks_pvalue:.4f} < {self.ks_alpha}). "
                "The recent return distribution is statistically different from "
                "the historical baseline."
            )
        if cusum_fired:
            direction_msg = {
                "up":   "WIN RATE IS RISING -- possible regime improvement.",
                "down": "WIN RATE IS FALLING -- possible regime deterioration.",
            }.get(self.cusum.direction(), "Win rate has shifted.")
            parts.append(
                f"CUSUM control chart exceeded limit "
                f"(C_hi={self.cusum.C_hi:.3f}, C_lo={self.cusum.C_lo:.3f}, "
                f"H={self.cusum.H:.3f}). {direction_msg}"
            )
        parts.append(
            "Recommendation: reset Bayesian posteriors to priors and "
            "re-estimate from scratch using only post-regime-change data."
        )
        return " | ".join(parts)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def alert_history(self) -> List[DriftAlert]:
        """Return the full history of emitted alerts."""
        return list(self._alert_history)

    def update_historical_baseline(self, new_historical: np.ndarray) -> None:
        """
        Replace the historical baseline distribution.

        Call this after a confirmed regime change to set the new "normal".

        Parameters
        ----------
        new_historical : array of P&L values to use as the new baseline.
        """
        self.historical_pnl = np.asarray(new_historical, dtype=float)
        new_win_rate = float(np.mean(self.historical_pnl > 0))
        self.cusum.baseline_win_rate = new_win_rate
        self.cusum.reset()
        self._recent_pnl_buffer.clear()
        logger.info(
            "DriftMonitor baseline updated: n=%d, win_rate=%.3f",
            len(self.historical_pnl), new_win_rate,
        )

    def cusum_status(self) -> dict:
        """Return current CUSUM state as a dict."""
        return {
            "C_hi":              self.cusum.C_hi,
            "C_lo":              self.cusum.C_lo,
            "H":                 self.cusum.H,
            "n_obs":             self.cusum.n_obs,
            "baseline_win_rate": self.cusum.baseline_win_rate,
            "triggered":         self.cusum.is_triggered(),
            "direction":         self.cusum.direction(),
        }
