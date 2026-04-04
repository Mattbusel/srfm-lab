"""
research/reconciliation/drift.py
==================================
Signal-drift and regime-drift detection for the live-vs-backtest
reconciliation pipeline.

The live BH (Black-Hole physics) engine and the backtest engine may diverge
over time due to:
  * Parameter recalibration
  * Data distribution shift in features fed to D3QN/DDQN/TD3QN ensembles
  * Changes in market micro-structure that affect ATR, mass, tf_score
  * Regime classification disagreements

This module provides statistical machinery to detect, quantify, and
visualise these drifts.

Classes
-------
SignalDriftDetector
    Central analysis class.

Dataclasses
-----------
LjungBoxResult          – Q-statistic, p-values, lags
ParameterStabilityResult – Chow-test breakpoints, rolling parameter series
RegimeDriftPeriod        – Detected period where live and BT regimes diverge
ActivationOverlapResult  – Rolling Jaccard similarity between activation sets
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

log = logging.getLogger(__name__)


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class LjungBoxResult:
    """Result of a Ljung-Box portmanteau test for residual autocorrelation."""
    lags: np.ndarray
    q_stats: np.ndarray
    p_values: np.ndarray
    reject_at_05: np.ndarray     # boolean mask: p < 0.05
    max_significant_lag: int     # highest lag with p < 0.05, or 0 if none
    autocorrelation_present: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "lags": self.lags.tolist(),
            "q_stats": self.q_stats.tolist(),
            "p_values": self.p_values.tolist(),
            "max_significant_lag": self.max_significant_lag,
            "autocorrelation_present": self.autocorrelation_present,
        }


@dataclass
class ParameterStabilityResult:
    """
    Result of the rolling parameter stability test with Chow-test breakpoints.
    """
    window: int
    rolling_win_rate: pd.Series
    rolling_avg_win: pd.Series
    rolling_avg_loss: pd.Series
    rolling_profit_factor: pd.Series
    breakpoints: list[int]          # indices where Chow test rejects H0
    breakpoint_times: list[Any]     # timestamps of breakpoints if available
    chow_f_stats: list[float]
    chow_p_values: list[float]
    is_stable: bool                 # True if no significant breakpoints found

    def to_dict(self) -> dict[str, Any]:
        return {
            "window": self.window,
            "n_breakpoints": len(self.breakpoints),
            "breakpoints": self.breakpoints,
            "breakpoint_times": [str(t) for t in self.breakpoint_times],
            "chow_p_values": self.chow_p_values,
            "is_stable": self.is_stable,
        }


@dataclass
class RegimeDriftPeriod:
    """A contiguous period where live and backtest regime labels disagree."""
    start_idx: int
    end_idx: int
    start_time: Optional[pd.Timestamp]
    end_time: Optional[pd.Timestamp]
    live_regime: str
    bt_regime: str
    duration_bars: int
    magnitude: float   # fraction of the window that disagrees

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "start_time": str(self.start_time),
            "end_time": str(self.end_time),
            "live_regime": self.live_regime,
            "bt_regime": self.bt_regime,
            "duration_bars": self.duration_bars,
            "magnitude": self.magnitude,
        }


@dataclass
class ActivationOverlapResult:
    """Rolling Jaccard similarity between live and backtest BH activations."""
    window: int
    overlap_series: pd.Series    # rolling Jaccard index
    mean_overlap: float
    min_overlap: float
    max_overlap: float
    low_overlap_periods: list[tuple[Any, Any]]  # (start, end) of overlap < threshold


# ── Helper functions ──────────────────────────────────────────────────────────


def _safe_series(x: Any, name: str = "signal") -> pd.Series:
    """Convert various inputs to a clean float pd.Series."""
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce")
    if isinstance(x, (list, np.ndarray)):
        return pd.to_numeric(pd.Series(x, name=name), errors="coerce")
    if isinstance(x, pd.DataFrame):
        if name in x.columns:
            return pd.to_numeric(x[name], errors="coerce")
        return pd.to_numeric(x.iloc[:, 0], errors="coerce")
    raise TypeError(f"Cannot convert {type(x)} to Series")


def _jaccard(set_a: set, set_b: set) -> float:
    """Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def _encode_regime(r: str) -> int:
    mapping = {"BULL": 1, "BEAR": -1, "SIDEWAYS": 0, "HIGH_VOL": 2, "UNKNOWN": -99}
    return mapping.get(str(r).upper(), -99)


# ── SignalDriftDetector ───────────────────────────────────────────────────────


class SignalDriftDetector:
    """
    Detect and quantify drift between live trading signals and backtest
    signals for the same instruments and time periods.

    Parameters
    ----------
    activation_threshold : float
        Minimum BH activation strength to count an asset as "active"
        (default 0.5).
    regime_window : int
        Window size (in bars) for computing rolling regime disagreement
        (default 20).
    chow_significance : float
        Significance level for Chow breakpoint tests (default 0.05).
    min_overlap_threshold : float
        Jaccard overlap below this value triggers a "low overlap" flag
        (default 0.3).
    """

    def __init__(
        self,
        activation_threshold: float = 0.5,
        regime_window: int = 20,
        chow_significance: float = 0.05,
        min_overlap_threshold: float = 0.3,
    ) -> None:
        self.activation_threshold = activation_threshold
        self.regime_window = regime_window
        self.chow_significance = chow_significance
        self.min_overlap_threshold = min_overlap_threshold

    # ── Activation overlap ────────────────────────────────────────────────

    def compute_activation_overlap(
        self,
        live_acts: pd.DataFrame,
        bt_acts: pd.DataFrame,
        window: int = 20,
    ) -> ActivationOverlapResult:
        """
        Compute rolling Jaccard similarity between live and backtest BH
        activations.

        Parameters
        ----------
        live_acts : pd.DataFrame
            Boolean or numeric activation matrix (rows = bars, cols = symbols).
            Each cell should be the BH activation score for that bar×symbol.
        bt_acts : pd.DataFrame
            Same structure as live_acts, aligned to the same bar index.
        window : int
            Rolling window size in bars.

        Returns
        -------
        ActivationOverlapResult
        """
        # Align indices
        common_idx = live_acts.index.intersection(bt_acts.index)
        if len(common_idx) == 0:
            empty = pd.Series(dtype=float, name="jaccard")
            return ActivationOverlapResult(
                window=window,
                overlap_series=empty,
                mean_overlap=float("nan"),
                min_overlap=float("nan"),
                max_overlap=float("nan"),
                low_overlap_periods=[],
            )

        L = live_acts.loc[common_idx]
        B = bt_acts.loc[common_idx]

        # Binarize
        L_bool = L.ge(self.activation_threshold)
        B_bool = B.ge(self.activation_threshold)

        overlap_values: list[float] = []
        for i in range(len(common_idx)):
            start = max(0, i - window + 1)
            l_window = L_bool.iloc[start : i + 1]
            b_window = B_bool.iloc[start : i + 1]

            # For each bar in the window, collect active symbols
            l_active: set[str] = set()
            b_active: set[str] = set()
            for _, row in l_window.iterrows():
                l_active |= set(row[row].index.tolist())
            for _, row in b_window.iterrows():
                b_active |= set(row[row].index.tolist())

            overlap_values.append(_jaccard(l_active, b_active))

        overlap_series = pd.Series(overlap_values, index=common_idx, name="jaccard_overlap")

        # Find low-overlap periods
        low_mask = overlap_series < self.min_overlap_threshold
        low_periods: list[tuple[Any, Any]] = []
        in_period = False
        period_start = None
        for idx_val, is_low in zip(overlap_series.index, low_mask):
            if is_low and not in_period:
                in_period = True
                period_start = idx_val
            elif not is_low and in_period:
                in_period = False
                low_periods.append((period_start, idx_val))
        if in_period and period_start is not None:
            low_periods.append((period_start, overlap_series.index[-1]))

        return ActivationOverlapResult(
            window=window,
            overlap_series=overlap_series,
            mean_overlap=float(overlap_series.mean()),
            min_overlap=float(overlap_series.min()),
            max_overlap=float(overlap_series.max()),
            low_overlap_periods=low_periods,
        )

    def compute_activation_overlap_from_trades(
        self,
        live_trades: pd.DataFrame,
        bt_trades: pd.DataFrame,
        window: int = 20,
        time_col: str = "exit_time",
        sym_col: str = "sym",
        score_col: str = "delta_score",
    ) -> ActivationOverlapResult:
        """
        Convenience method: build activation matrices from trade DataFrames
        and then compute rolling Jaccard overlap.

        This pivots trade data into a bar×symbol activation matrix, using
        delta_score (or a proxy) as the activation value.
        """
        def _build_matrix(df: pd.DataFrame) -> pd.DataFrame:
            if df.empty:
                return pd.DataFrame()
            tcol = time_col if time_col in df.columns else "exit_time"
            scol = sym_col if sym_col in df.columns else "sym"
            vcol = score_col if score_col in df.columns else "pnl"

            subset = df[[tcol, scol, vcol]].dropna(subset=[tcol, scol])
            subset = subset.rename(columns={tcol: "ts", scol: "symbol", vcol: "value"})
            subset["ts"] = pd.to_datetime(subset["ts"], utc=True, errors="coerce")
            subset["value"] = pd.to_numeric(subset["value"], errors="coerce").fillna(0).abs()

            pivot = subset.pivot_table(
                index="ts", columns="symbol", values="value", aggfunc="mean"
            ).fillna(0)
            return pivot

        live_matrix = _build_matrix(live_trades)
        bt_matrix = _build_matrix(bt_trades)

        if live_matrix.empty or bt_matrix.empty:
            empty = pd.Series(dtype=float, name="jaccard")
            return ActivationOverlapResult(
                window=window,
                overlap_series=empty,
                mean_overlap=float("nan"),
                min_overlap=float("nan"),
                max_overlap=float("nan"),
                low_overlap_periods=[],
            )

        # Align columns
        all_syms = list(set(live_matrix.columns) | set(bt_matrix.columns))
        live_matrix = live_matrix.reindex(columns=all_syms, fill_value=0)
        bt_matrix = bt_matrix.reindex(columns=all_syms, fill_value=0)

        return self.compute_activation_overlap(live_matrix, bt_matrix, window=window)

    # ── Regime drift ──────────────────────────────────────────────────────

    def detect_regime_drift(
        self,
        live_regimes: pd.Series,
        bt_regimes: pd.Series,
    ) -> list[RegimeDriftPeriod]:
        """
        Identify contiguous periods where live and backtest regime
        classifications disagree.

        Parameters
        ----------
        live_regimes : pd.Series
            Time-indexed series of regime labels from the live system.
        bt_regimes : pd.Series
            Time-indexed series of regime labels from the backtest.

        Returns
        -------
        list[RegimeDriftPeriod]
            Sorted by start_idx.
        """
        # Align
        common = live_regimes.index.intersection(bt_regimes.index)
        if len(common) == 0:
            return []

        L = live_regimes.loc[common]
        B = bt_regimes.loc[common]

        disagree = L.values != B.values
        drift_periods: list[RegimeDriftPeriod] = []

        in_drift = False
        drift_start = 0
        drift_live: list[str] = []
        drift_bt: list[str] = []

        for i, (is_diff, l_reg, b_reg) in enumerate(zip(disagree, L.values, B.values)):
            if is_diff and not in_drift:
                in_drift = True
                drift_start = i
                drift_live = [str(l_reg)]
                drift_bt = [str(b_reg)]
            elif is_diff and in_drift:
                drift_live.append(str(l_reg))
                drift_bt.append(str(b_reg))
            elif not is_diff and in_drift:
                in_drift = False
                duration = i - drift_start
                # Most common regime in each sequence
                live_mode = max(set(drift_live), key=drift_live.count)
                bt_mode = max(set(drift_bt), key=drift_bt.count)
                magnitude = len(drift_live) / max(len(common), 1)

                try:
                    start_time = common[drift_start]
                    end_time = common[i - 1]
                except IndexError:
                    start_time = None
                    end_time = None

                drift_periods.append(RegimeDriftPeriod(
                    start_idx=drift_start,
                    end_idx=i - 1,
                    start_time=start_time,
                    end_time=end_time,
                    live_regime=live_mode,
                    bt_regime=bt_mode,
                    duration_bars=duration,
                    magnitude=magnitude,
                ))

        # Close any open drift at end of series
        if in_drift:
            duration = len(common) - drift_start
            live_mode = max(set(drift_live), key=drift_live.count)
            bt_mode = max(set(drift_bt), key=drift_bt.count)
            magnitude = len(drift_live) / max(len(common), 1)
            try:
                start_time = common[drift_start]
                end_time = common[-1]
            except IndexError:
                start_time = None
                end_time = None
            drift_periods.append(RegimeDriftPeriod(
                start_idx=drift_start,
                end_idx=len(common) - 1,
                start_time=start_time,
                end_time=end_time,
                live_regime=live_mode,
                bt_regime=bt_mode,
                duration_bars=duration,
                magnitude=magnitude,
            ))

        return sorted(drift_periods, key=lambda d: d.start_idx)

    def regime_disagreement_rate(
        self,
        live_regimes: pd.Series,
        bt_regimes: pd.Series,
    ) -> float:
        """
        Compute the fraction of bars where live and backtest disagree on
        regime classification.
        """
        common = live_regimes.index.intersection(bt_regimes.index)
        if len(common) == 0:
            return float("nan")
        disagree = (live_regimes.loc[common].values != bt_regimes.loc[common].values)
        return float(disagree.mean())

    def regime_confusion_matrix(
        self,
        live_regimes: pd.Series,
        bt_regimes: pd.Series,
    ) -> pd.DataFrame:
        """
        Compute a regime confusion matrix (live rows vs backtest columns).

        Returns
        -------
        pd.DataFrame
            Each cell [l, b] = number of bars where live=l and bt=b.
        """
        common = live_regimes.index.intersection(bt_regimes.index)
        if len(common) == 0:
            return pd.DataFrame()

        L = live_regimes.loc[common].astype(str)
        B = bt_regimes.loc[common].astype(str)
        labels = sorted(set(L.unique()) | set(B.unique()))

        mat = pd.DataFrame(0, index=labels, columns=labels)
        for l_val, b_val in zip(L.values, B.values):
            mat.loc[l_val, b_val] += 1
        mat.index.name = "live"
        mat.columns.name = "backtest"
        return mat

    # ── Signal autocorrelation ────────────────────────────────────────────

    def compute_signal_autocorrelation(
        self,
        signals: Any,
        max_lag: int = 20,
    ) -> np.ndarray:
        """
        Compute the autocorrelation function (ACF) of a signal series.

        Parameters
        ----------
        signals : array-like or pd.Series
            The signal time series (e.g. delta_score, tf_score, ensemble).
        max_lag : int
            Maximum lag to compute.

        Returns
        -------
        np.ndarray of shape (max_lag + 1,)
            ACF values at lags 0, 1, ..., max_lag.
        """
        s = _safe_series(signals).dropna().values.astype(float)
        n = len(s)
        if n < max_lag + 2:
            warnings.warn(
                f"Signal too short ({n} obs) for ACF with max_lag={max_lag}; "
                "padding with NaN.",
                stacklevel=2,
            )
            acf = np.full(max_lag + 1, np.nan)
            available = min(n - 1, max_lag)
            for lag in range(available + 1):
                if lag == 0:
                    acf[0] = 1.0
                    continue
                x = s[:n - lag] - s.mean()
                y = s[lag:] - s.mean()
                denom = np.sum((s - s.mean()) ** 2)
                if denom == 0:
                    acf[lag] = np.nan
                else:
                    acf[lag] = np.sum(x * y) / denom
            return acf

        mean = s.mean()
        var = np.sum((s - mean) ** 2)
        acf = np.ones(max_lag + 1)
        for lag in range(1, max_lag + 1):
            cov = np.sum((s[:n - lag] - mean) * (s[lag:] - mean))
            acf[lag] = cov / var if var != 0 else np.nan
        return acf

    def compute_pacf(
        self,
        signals: Any,
        max_lag: int = 20,
    ) -> np.ndarray:
        """
        Compute the partial autocorrelation function (PACF) via
        Yule-Walker equations.

        Returns
        -------
        np.ndarray of shape (max_lag + 1,)
        """
        s = _safe_series(signals).dropna().values.astype(float)
        n = len(s)
        if n < 4:
            return np.full(max_lag + 1, np.nan)

        acf = self.compute_signal_autocorrelation(s, max_lag=max_lag)

        # Durbin-Levinson algorithm for PACF
        pacf = np.zeros(max_lag + 1)
        pacf[0] = 1.0
        if max_lag < 1 or np.isnan(acf[1]):
            return pacf

        pacf[1] = acf[1]
        phi = np.array([acf[1]])

        for k in range(2, max_lag + 1):
            if np.any(np.isnan(acf[:k + 1])):
                pacf[k] = np.nan
                continue
            # phi_{k,k}
            num = acf[k] - np.dot(phi, acf[1:k][::-1])
            denom = 1.0 - np.dot(phi, acf[1:k])
            if abs(denom) < 1e-12:
                pacf[k] = np.nan
                break
            phi_kk = num / denom
            pacf[k] = phi_kk
            # Update phi vector
            phi_new = phi - phi_kk * phi[::-1]
            phi = np.append(phi_new, phi_kk)

        return pacf

    # ── Ljung-Box test ────────────────────────────────────────────────────

    def ljung_box_test(
        self,
        residuals: Any,
        lags: int = 20,
    ) -> LjungBoxResult:
        """
        Perform the Ljung-Box portmanteau test for residual autocorrelation.

        H0: The residuals are independently distributed (no autocorrelation).
        H1: Autocorrelation exists.

        Parameters
        ----------
        residuals : array-like
            Model residuals or raw signal values.
        lags : int
            Number of lags to test.

        Returns
        -------
        LjungBoxResult
        """
        r = _safe_series(residuals).dropna().values.astype(float)
        n = len(r)

        if n < lags + 2:
            lags = max(1, n - 2)

        acf_vals = self.compute_signal_autocorrelation(r, max_lag=lags)
        lag_array = np.arange(1, lags + 1)

        q_stats = np.zeros(lags)
        for k in range(1, lags + 1):
            # Q_LB = n(n+2) * sum_{j=1}^{k} rho_j^2 / (n-j)
            q_stats[k - 1] = n * (n + 2) * np.sum(
                [acf_vals[j] ** 2 / (n - j) for j in range(1, k + 1) if n > j]
            )

        p_values = np.array([
            1.0 - stats.chi2.cdf(q, df=lag)
            for q, lag in zip(q_stats, lag_array)
        ])

        reject = p_values < 0.05
        significant_lags = lag_array[reject]
        max_sig = int(significant_lags.max()) if len(significant_lags) > 0 else 0

        return LjungBoxResult(
            lags=lag_array,
            q_stats=q_stats,
            p_values=p_values,
            reject_at_05=reject,
            max_significant_lag=max_sig,
            autocorrelation_present=bool(reject.any()),
        )

    # ── Parameter stability ───────────────────────────────────────────────

    def parameter_stability_test(
        self,
        trades: pd.DataFrame,
        window: int = 500,
        pnl_col: str = "pnl",
        time_col: Optional[str] = None,
        min_segment: int = 30,
    ) -> ParameterStabilityResult:
        """
        Test whether win_rate, avg_win, avg_loss are stable over time using
        rolling statistics and Chow-test structural-break detection.

        Parameters
        ----------
        trades : pd.DataFrame
            Trade DataFrame sorted by time.
        window : int
            Rolling window in number of trades.
        pnl_col : str
            Column containing per-trade PnL.
        time_col : str | None
            Column for timestamps (used to annotate breakpoints).
        min_segment : int
            Minimum segment size for Chow test.

        Returns
        -------
        ParameterStabilityResult
        """
        if pnl_col not in trades.columns:
            # Try common aliases
            for alias in ("live_pnl", "bt_pnl", "return_pct"):
                if alias in trades.columns:
                    pnl_col = alias
                    break

        pnl = pd.to_numeric(trades.get(pnl_col, pd.Series(dtype=float)), errors="coerce").reset_index(drop=True)
        n = len(pnl)

        if n < window // 2:
            empty = pd.Series(dtype=float)
            return ParameterStabilityResult(
                window=window, rolling_win_rate=empty,
                rolling_avg_win=empty, rolling_avg_loss=empty,
                rolling_profit_factor=empty,
                breakpoints=[], breakpoint_times=[],
                chow_f_stats=[], chow_p_values=[],
                is_stable=True,
            )

        win_rate_vals: list[float] = []
        avg_win_vals: list[float] = []
        avg_loss_vals: list[float] = []
        pf_vals: list[float] = []

        for i in range(n):
            start = max(0, i - window + 1)
            w = pnl.iloc[start: i + 1].dropna()
            if len(w) == 0:
                win_rate_vals.append(np.nan)
                avg_win_vals.append(np.nan)
                avg_loss_vals.append(np.nan)
                pf_vals.append(np.nan)
                continue

            wins = w[w > 0]
            losses = w[w <= 0]
            win_rate_vals.append(float((w > 0).mean()))
            avg_win_vals.append(float(wins.mean()) if len(wins) > 0 else np.nan)
            avg_loss_vals.append(float(losses.mean()) if len(losses) > 0 else np.nan)
            gross_profit = wins.sum() if len(wins) > 0 else 0.0
            gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0
            pf_vals.append(gross_profit / gross_loss if gross_loss > 0 else float("inf"))

        rolling_win_rate = pd.Series(win_rate_vals, name="rolling_win_rate")
        rolling_avg_win = pd.Series(avg_win_vals, name="rolling_avg_win")
        rolling_avg_loss = pd.Series(avg_loss_vals, name="rolling_avg_loss")
        rolling_pf = pd.Series(pf_vals, name="rolling_profit_factor")

        # Chow test breakpoint detection
        breakpoints: list[int] = []
        chow_f_stats: list[float] = []
        chow_p_values: list[float] = []

        pnl_clean = pnl.dropna().values
        n_clean = len(pnl_clean)
        candidate_breaks = range(min_segment, n_clean - min_segment, max(1, n_clean // 20))

        for bp in candidate_breaks:
            f_stat, p_val = self._chow_test(pnl_clean, bp)
            if p_val < self.chow_significance:
                breakpoints.append(bp)
                chow_f_stats.append(f_stat)
                chow_p_values.append(p_val)

        # Suppress adjacent breakpoints (keep the most significant in each cluster)
        breakpoints, chow_f_stats, chow_p_values = self._deduplicate_breakpoints(
            breakpoints, chow_f_stats, chow_p_values, min_gap=min_segment
        )

        # Map breakpoints back to timestamps
        breakpoint_times: list[Any] = []
        if time_col and time_col in trades.columns:
            times = trades[time_col].reset_index(drop=True)
            for bp in breakpoints:
                if bp < len(times):
                    breakpoint_times.append(times.iloc[bp])
                else:
                    breakpoint_times.append(None)
        else:
            breakpoint_times = [None] * len(breakpoints)

        return ParameterStabilityResult(
            window=window,
            rolling_win_rate=rolling_win_rate,
            rolling_avg_win=rolling_avg_win,
            rolling_avg_loss=rolling_avg_loss,
            rolling_profit_factor=rolling_pf,
            breakpoints=breakpoints,
            breakpoint_times=breakpoint_times,
            chow_f_stats=chow_f_stats,
            chow_p_values=chow_p_values,
            is_stable=len(breakpoints) == 0,
        )

    def _chow_test(
        self,
        data: np.ndarray,
        breakpoint: int,
    ) -> tuple[float, float]:
        """
        Chow test for structural break at the given index.

        Models: y ~ const (mean-only model for PnL series).
        RSS_full  = RSS(data[:bp]) + RSS(data[bp:])
        RSS_restricted = RSS(data)
        F = [(RSS_R - RSS_F) / k] / [RSS_F / (n - 2k)]
          where k = 1 (intercept only).
        """
        n = len(data)
        if breakpoint < 2 or breakpoint > n - 2:
            return 0.0, 1.0

        seg1 = data[:breakpoint]
        seg2 = data[breakpoint:]

        def _rss(arr: np.ndarray) -> float:
            return float(np.sum((arr - arr.mean()) ** 2))

        rss1 = _rss(seg1)
        rss2 = _rss(seg2)
        rss_all = _rss(data)

        rss_full = rss1 + rss2
        k = 1  # intercept only

        if rss_full < 1e-15:
            return 0.0, 1.0

        f_stat = ((rss_all - rss_full) / k) / (rss_full / (n - 2 * k))
        p_val = 1.0 - stats.f.cdf(f_stat, dfn=k, dfd=n - 2 * k)
        return float(f_stat), float(p_val)

    def _deduplicate_breakpoints(
        self,
        breakpoints: list[int],
        f_stats: list[float],
        p_values: list[float],
        min_gap: int = 30,
    ) -> tuple[list[int], list[float], list[float]]:
        """Keep only the most significant breakpoint within each cluster."""
        if not breakpoints:
            return [], [], []

        combined = sorted(zip(breakpoints, f_stats, p_values), key=lambda x: x[0])
        result_bps: list[int] = []
        result_fs: list[float] = []
        result_ps: list[float] = []

        cluster: list[tuple[int, float, float]] = [combined[0]]
        for bp, f, p in combined[1:]:
            if bp - cluster[-1][0] < min_gap:
                cluster.append((bp, f, p))
            else:
                # Keep the one with highest F-stat
                best = max(cluster, key=lambda x: x[1])
                result_bps.append(best[0])
                result_fs.append(best[1])
                result_ps.append(best[2])
                cluster = [(bp, f, p)]

        if cluster:
            best = max(cluster, key=lambda x: x[1])
            result_bps.append(best[0])
            result_fs.append(best[1])
            result_ps.append(best[2])

        return result_bps, result_fs, result_ps

    # ── Live vs BT comparison helpers ────────────────────────────────────

    def compare_signal_distributions(
        self,
        live_trades: pd.DataFrame,
        bt_trades: pd.DataFrame,
        signal_col: str = "delta_score",
    ) -> dict[str, Any]:
        """
        Compare the distribution of a signal column between live and backtest
        using the two-sample Kolmogorov-Smirnov test and descriptive stats.

        Returns
        -------
        dict with keys: ks_statistic, ks_p_value, live_mean, bt_mean,
            live_std, bt_std, distribution_shift, diverged (bool)
        """
        live_col = f"live_{signal_col}" if f"live_{signal_col}" in live_trades.columns else signal_col
        bt_col = f"bt_{signal_col}" if f"bt_{signal_col}" in bt_trades.columns else signal_col

        live_vals = pd.to_numeric(live_trades.get(live_col, pd.Series(dtype=float)), errors="coerce").dropna()
        bt_vals = pd.to_numeric(bt_trades.get(bt_col, pd.Series(dtype=float)), errors="coerce").dropna()

        if len(live_vals) < 5 or len(bt_vals) < 5:
            return {
                "ks_statistic": np.nan, "ks_p_value": np.nan,
                "live_mean": float(live_vals.mean()) if len(live_vals) > 0 else np.nan,
                "bt_mean": float(bt_vals.mean()) if len(bt_vals) > 0 else np.nan,
                "live_std": float(live_vals.std()) if len(live_vals) > 0 else np.nan,
                "bt_std": float(bt_vals.std()) if len(bt_vals) > 0 else np.nan,
                "distribution_shift": np.nan,
                "diverged": False,
            }

        ks_stat, ks_p = stats.ks_2samp(live_vals.values, bt_vals.values)
        shift = (live_vals.mean() - bt_vals.mean()) / max(bt_vals.std(), 1e-8)

        return {
            "ks_statistic": float(ks_stat),
            "ks_p_value": float(ks_p),
            "live_mean": float(live_vals.mean()),
            "bt_mean": float(bt_vals.mean()),
            "live_std": float(live_vals.std()),
            "bt_std": float(bt_vals.std()),
            "distribution_shift": float(shift),
            "diverged": bool(ks_p < 0.05 and abs(shift) > 0.5),
        }

    def compute_ensemble_drift(
        self,
        live_trades: pd.DataFrame,
        bt_trades: pd.DataFrame,
        time_col: str = "exit_time",
        window: int = 50,
    ) -> pd.DataFrame:
        """
        Compute rolling mean-shift of ensemble_signal between live and
        backtest over time.

        Returns
        -------
        pd.DataFrame
            Columns: period, live_mean_signal, bt_mean_signal, drift
        """
        def _rolling_mean(df: pd.DataFrame, col: str, w: int) -> pd.Series:
            prefix_col = f"live_{col}" if f"live_{col}" in df.columns else col
            bt_prefix = f"bt_{col}" if f"bt_{col}" in df.columns else col
            # Try live prefix first, then raw
            for candidate in (prefix_col, bt_prefix, col):
                if candidate in df.columns:
                    return pd.to_numeric(df[candidate], errors="coerce").rolling(w, min_periods=1).mean()
            return pd.Series(np.nan, index=df.index)

        live_signal = _rolling_mean(live_trades, "ensemble_signal", window)
        bt_signal = _rolling_mean(bt_trades, "ensemble_signal", window)

        # Align lengths
        min_len = min(len(live_signal), len(bt_signal))
        if min_len == 0:
            return pd.DataFrame()

        out = pd.DataFrame({
            "live_mean_signal": live_signal.values[:min_len],
            "bt_mean_signal": bt_signal.values[:min_len],
        })
        out["drift"] = out["live_mean_signal"] - out["bt_mean_signal"]
        return out

    # ── Plotting ──────────────────────────────────────────────────────────

    def plot_activation_overlap(
        self,
        overlap_result: ActivationOverlapResult,
        save_path: str | Path,
        title: str = "BH Activation Overlap (Live vs Backtest)",
        dpi: int = 150,
    ) -> Path:
        """
        Plot rolling Jaccard overlap between live and backtest BH activations.

        Parameters
        ----------
        overlap_result : ActivationOverlapResult
            Output of compute_activation_overlap.
        save_path : str | Path
            Output file path.
        title : str
            Chart title.
        dpi : int
            Image resolution.

        Returns
        -------
        Path
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        overlap = overlap_result.overlap_series

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        ax = axes[0]
        if len(overlap) > 0:
            x = range(len(overlap))
            ax.plot(x, overlap.values, color="steelblue", linewidth=1.2, label="Jaccard Overlap")
            ax.axhline(overlap_result.mean_overlap, color="orange", linestyle="--",
                       linewidth=1.2, label=f"Mean={overlap_result.mean_overlap:.2f}")
            ax.axhline(self.min_overlap_threshold, color="red", linestyle=":",
                       linewidth=1.2, label=f"Threshold={self.min_overlap_threshold:.2f}")

            # Shade low-overlap periods
            for start, end in overlap_result.low_overlap_periods:
                if start in overlap.index and end in overlap.index:
                    s_pos = overlap.index.get_loc(start)
                    e_pos = overlap.index.get_loc(end)
                    ax.axvspan(s_pos, e_pos, alpha=0.2, color="red")

            ax.fill_between(x, overlap.values, self.min_overlap_threshold,
                            where=overlap.values < self.min_overlap_threshold,
                            alpha=0.3, color="red", label="Below threshold")
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Jaccard Similarity")
            ax.set_title(title)
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.text(0.5, 0.5, "No overlap data", ha="center", va="center")
            ax.set_title(title)

        # Rolling average with smoothing
        ax2 = axes[1]
        if len(overlap) > 5:
            smooth = overlap.rolling(10, min_periods=1).mean()
            ax2.plot(range(len(smooth)), smooth.values, color="green",
                     linewidth=1.5, label="10-bar Smoothed Overlap")
            ax2.fill_between(range(len(smooth)), smooth.values, 0, alpha=0.15, color="green")
            ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="50% overlap")
            ax2.set_ylabel("Smoothed Jaccard")
            ax2.set_xlabel("Bar Index")
            ax2.legend(loc="upper right", fontsize=8)

        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        log.info("Activation overlap plot saved to %s", save_path)
        return save_path

    def plot_regime_drift(
        self,
        live_regimes: pd.Series,
        bt_regimes: pd.Series,
        drift_periods: list[RegimeDriftPeriod],
        save_path: str | Path,
        dpi: int = 150,
    ) -> Path:
        """
        Visualise live vs backtest regime classification with drift periods
        highlighted.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        common = live_regimes.index.intersection(bt_regimes.index)
        L = live_regimes.loc[common]
        B = bt_regimes.loc[common]

        regime_to_int = {"BULL": 0, "BEAR": 1, "SIDEWAYS": 2, "HIGH_VOL": 3, "UNKNOWN": 4}
        colors = ["green", "red", "gray", "orange", "black"]
        cmap = ListedColormap(colors)

        l_num = np.array([regime_to_int.get(str(r), 4) for r in L.values])
        b_num = np.array([regime_to_int.get(str(r), 4) for r in B.values])
        disagree = (l_num != b_num).astype(float)

        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

        x = range(len(common))
        axes[0].scatter(x, l_num, c=l_num, cmap=cmap, vmin=0, vmax=4, s=8, alpha=0.7)
        axes[0].set_ylabel("Live Regime")
        axes[0].set_yticks(list(regime_to_int.values()))
        axes[0].set_yticklabels(list(regime_to_int.keys()), fontsize=7)
        axes[0].set_title("Live vs Backtest Regime Classification")

        axes[1].scatter(x, b_num, c=b_num, cmap=cmap, vmin=0, vmax=4, s=8, alpha=0.7)
        axes[1].set_ylabel("BT Regime")
        axes[1].set_yticks(list(regime_to_int.values()))
        axes[1].set_yticklabels(list(regime_to_int.keys()), fontsize=7)

        axes[2].fill_between(x, disagree, alpha=0.6, color="red", label="Disagreement")
        axes[2].set_ylabel("Disagreement")
        axes[2].set_xlabel("Bar Index")
        axes[2].set_ylim(-0.05, 1.2)
        total_dis = float(disagree.mean()) * 100
        axes[2].set_title(f"Regime Disagreement (Overall: {total_dis:.1f}%)")

        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        log.info("Regime drift plot saved to %s", save_path)
        return save_path

    def plot_acf_pacf(
        self,
        signals: Any,
        save_path: str | Path,
        max_lag: int = 20,
        title: str = "ACF / PACF",
        dpi: int = 150,
    ) -> Path:
        """
        Plot autocorrelation and partial autocorrelation functions.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        s = _safe_series(signals).dropna()
        n = len(s)
        acf = self.compute_signal_autocorrelation(s, max_lag=max_lag)
        pacf = self.compute_pacf(s, max_lag=max_lag)

        # Confidence bands: ±1.96/sqrt(n)
        conf = 1.96 / np.sqrt(max(n, 1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        lags = np.arange(max_lag + 1)
        ax1.bar(lags, acf, color="steelblue", alpha=0.8)
        ax1.axhline(conf, color="red", linestyle="--", linewidth=0.8)
        ax1.axhline(-conf, color="red", linestyle="--", linewidth=0.8)
        ax1.axhline(0, color="black", linewidth=0.5)
        ax1.set_title("Autocorrelation Function (ACF)")
        ax1.set_xlabel("Lag")
        ax1.set_ylabel("ACF")

        ax2.bar(lags, pacf, color="darkorange", alpha=0.8)
        ax2.axhline(conf, color="red", linestyle="--", linewidth=0.8)
        ax2.axhline(-conf, color="red", linestyle="--", linewidth=0.8)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.set_title("Partial ACF (PACF)")
        ax2.set_xlabel("Lag")
        ax2.set_ylabel("PACF")

        fig.suptitle(title, fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def parameter_stability_summary(
        self,
        result: ParameterStabilityResult,
    ) -> dict[str, Any]:
        """
        Convert a ParameterStabilityResult to a human-readable summary dict.
        """
        return {
            "is_stable": result.is_stable,
            "n_breakpoints": len(result.breakpoints),
            "breakpoints": result.breakpoints,
            "breakpoint_times": [str(t) for t in result.breakpoint_times],
            "final_win_rate": float(result.rolling_win_rate.dropna().iloc[-1])
            if len(result.rolling_win_rate.dropna()) > 0 else np.nan,
            "final_profit_factor": float(result.rolling_profit_factor.dropna().iloc[-1])
            if len(result.rolling_profit_factor.dropna()) > 0 else np.nan,
            "win_rate_range": [
                float(result.rolling_win_rate.dropna().min()),
                float(result.rolling_win_rate.dropna().max()),
            ] if len(result.rolling_win_rate.dropna()) > 0 else [np.nan, np.nan],
        }
