"""
research/reconciliation/leakage.py
=====================================
Data-leakage and overfitting detection for the live-vs-backtest
reconciliation pipeline.

Three sources of spurious backtest outperformance are targeted:

1. **Lookahead bias** – backtest uses data not available at decision time
   (e.g., uses the close of the current bar to decide the current bar's entry).
   Detected by comparing live and backtest returns in the *same* calendar
   period: if backtest significantly exceeds live, lookahead is suspected.

2. **Training/test contamination** – the strategy parameters were fitted on
   data that overlaps with the evaluation period.  The *Combinatorial Purged
   Cross-Validation* (CPCV) embargo technique is used to purge overlapping
   observations.

3. **Multiple-testing / overfitting** – the strategy was selected from many
   parameter combinations, inflating apparent Sharpe.  Addressed via the
   Bailey-Borwein-Plouffe deflated Sharpe ratio (DSR).

Classes
-------
DataLeakageAuditor

Dataclasses
-----------
LeakageReport
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger(__name__)

# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class LeakageReport:
    """
    Full data-leakage audit report.
    """
    # Lookahead bias
    leakage_score: float          # in [0, 1]; > 0.5 flags likely lookahead
    lookahead_suspected: bool
    live_sharpe_in_period: float
    bt_sharpe_in_period: float
    sharpe_ratio_excess: float    # bt_sharpe / live_sharpe - 1

    # Purged-observations analysis
    n_purged: int                 # trades removed by embargo
    n_remaining: int
    purge_fraction: float

    # Overfitting scores
    in_sample_sharpe: float
    oos_sharpe: float
    deflated_sharpe: float        # DSR (Bailey et al.)
    overfitting_probability: float  # probability strategy is overfit

    # Variance Inflation Factors (if factor data available)
    vif_series: pd.Series
    high_vif_factors: list[str]   # factors with VIF > 5

    # Autocorrelation-adjusted performance
    ac_adjusted_sharpe: float     # Newey-West adjusted
    autocorrelation_score: float  # how much AC inflates apparent Sharpe

    metadata: dict[str, Any] = field(default_factory=dict)

    def summary_dict(self) -> dict[str, Any]:
        return {
            "leakage_score": self.leakage_score,
            "lookahead_suspected": self.lookahead_suspected,
            "live_sharpe_in_period": self.live_sharpe_in_period,
            "bt_sharpe_in_period": self.bt_sharpe_in_period,
            "sharpe_excess": self.sharpe_ratio_excess,
            "n_purged": self.n_purged,
            "purge_fraction": self.purge_fraction,
            "deflated_sharpe": self.deflated_sharpe,
            "overfitting_probability": self.overfitting_probability,
            "ac_adjusted_sharpe": self.ac_adjusted_sharpe,
            "autocorrelation_score": self.autocorrelation_score,
            "high_vif_factors": self.high_vif_factors,
        }


# ── DataLeakageAuditor ────────────────────────────────────────────────────────


class DataLeakageAuditor:
    """
    Detect and quantify data leakage and overfitting in backtest results.

    Parameters
    ----------
    embargo_bars : int
        Number of bars to embargo after each training observation to prevent
        label leakage from overlapping samples (default 5).
    leakage_threshold : float
        Leakage score above which lookahead bias is flagged (default 0.5).
    n_trials_equivalent : int
        Equivalent number of strategy trials (for DSR computation).
        Set to the approximate number of parameter combinations tested
        (default 100).
    """

    def __init__(
        self,
        embargo_bars: int = 5,
        leakage_threshold: float = 0.5,
        n_trials_equivalent: int = 100,
    ) -> None:
        self.embargo_bars = embargo_bars
        self.leakage_threshold = leakage_threshold
        self.n_trials_equivalent = n_trials_equivalent

    # ── internal utilities ────────────────────────────────────────────────

    def _compute_sharpe(
        self,
        returns: pd.Series,
        annual_factor: float = 252.0,
    ) -> float:
        r = pd.to_numeric(returns, errors="coerce").dropna()
        if len(r) < 2:
            return float("nan")
        std = float(r.std())
        if std < 1e-10:
            return float("nan")
        return float(r.mean() / std * np.sqrt(annual_factor))

    def _get_pnl_col(self, df: pd.DataFrame) -> str:
        for c in ("pnl", "live_pnl", "bt_pnl", "return_pct"):
            if c in df.columns:
                return c
        raise ValueError("No PnL column found in DataFrame.")

    def _get_time_col(self, df: pd.DataFrame) -> Optional[str]:
        for c in ("exit_time", "live_exit_time", "bt_exit_time", "ts"):
            if c in df.columns:
                return c
        return None

    # ── Purge test ────────────────────────────────────────────────────────

    def purge_test(
        self,
        trades: pd.DataFrame,
        embargo_bars: Optional[int] = None,
        time_col: Optional[str] = None,
        hold_col: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Remove trades whose label (exit return) overlaps with any prior
        training observation within an embargo window.

        In the context of walk-forward backtesting, trades whose holding
        period overlaps with the test-set period of the previous fold must
        be purged to avoid forward leakage.

        This implementation uses a simple time-based embargo: after each
        trade closes, the next ``embargo_bars`` worth of trades starting
        within the same bar-window are flagged and removed.

        Parameters
        ----------
        trades : pd.DataFrame
            Sorted trade DataFrame (by entry or exit time).
        embargo_bars : int | None
            Override the instance's default embargo_bars.
        time_col : str | None
            Column with exit (or entry) timestamps.
        hold_col : str | None
            Column with hold duration in bars.

        Returns
        -------
        pd.DataFrame
            Purged trade DataFrame with an added boolean column
            ``purged`` indicating which rows were removed.
        """
        emb = embargo_bars if embargo_bars is not None else self.embargo_bars
        if time_col is None:
            time_col = self._get_time_col(trades)
        if hold_col is None:
            for c in ("hold_bars", "live_hold_bars", "bt_hold_bars"):
                if c in trades.columns:
                    hold_col = c
                    break

        out = trades.copy()
        out["purged"] = False

        if time_col is None or time_col not in out.columns:
            log.warning("No time column available for purge test; skipping.")
            return out

        out["_ts"] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
        out = out.sort_values("_ts").reset_index(drop=True)

        if hold_col and hold_col in out.columns:
            holds = pd.to_numeric(out[hold_col], errors="coerce").fillna(1).astype(int)
        else:
            holds = pd.Series(1, index=out.index)

        purge_until: pd.Timestamp = pd.Timestamp.min.tz_localize("UTC")
        for i in range(len(out)):
            ts = out.at[i, "_ts"]
            if pd.isna(ts):
                continue
            if ts <= purge_until:
                out.at[i, "purged"] = True
            else:
                # Set purge window based on hold duration
                hold = int(holds.iloc[i])
                purge_until = ts + pd.Timedelta(hours=hold + emb)

        out.drop(columns=["_ts"], inplace=True)
        n_purged = int(out["purged"].sum())
        log.info("Purge test: %d / %d trades purged (%.1f%%)",
                 n_purged, len(out), n_purged / max(len(out), 1) * 100)
        return out

    # ── Future data contamination ─────────────────────────────────────────

    def check_future_data_contamination(
        self,
        bt_trades: pd.DataFrame,
        live_trades: pd.DataFrame,
        overlap_window_hours: float = 24.0,
    ) -> float:
        """
        Check whether backtest fills occur suspiciously close (in time) to
        live fills for the same symbol—a sign that live execution prices
        contaminated the backtest.

        The contamination score is the fraction of backtest trades that fall
        within ``overlap_window_hours`` of a live trade for the same symbol.

        Parameters
        ----------
        bt_trades, live_trades : pd.DataFrame
        overlap_window_hours : float
            Time window for calling two trades "suspicious" (default 24h).

        Returns
        -------
        float
            Contamination fraction in [0, 1].  Values > 0.1 warrant scrutiny.
        """
        if bt_trades.empty or live_trades.empty:
            return 0.0

        bt_t = self._get_time_col(bt_trades)
        live_t = self._get_time_col(live_trades)
        bt_s = "sym" if "sym" in bt_trades.columns else "live_sym" if "live_sym" in bt_trades.columns else None
        live_s = "sym" if "sym" in live_trades.columns else "live_sym" if "live_sym" in live_trades.columns else None

        if not (bt_t and live_t and bt_s and live_s):
            return 0.0

        bt_df = bt_trades[[bt_s, bt_t]].copy()
        bt_df.columns = ["sym", "ts"]
        bt_df["ts"] = pd.to_datetime(bt_df["ts"], utc=True, errors="coerce")

        live_df = live_trades[[live_s, live_t]].copy()
        live_df.columns = ["sym", "ts"]
        live_df["ts"] = pd.to_datetime(live_df["ts"], utc=True, errors="coerce")

        tol = pd.Timedelta(hours=overlap_window_hours)
        contaminated = 0
        total = 0

        for sym, bt_grp in bt_df.groupby("sym"):
            live_grp = live_df[live_df["sym"] == sym]
            if live_grp.empty:
                continue
            for bt_ts in bt_grp["ts"].dropna():
                total += 1
                diffs = (live_grp["ts"] - bt_ts).abs()
                if diffs.min() <= tol:
                    contaminated += 1

        if total == 0:
            return 0.0
        return contaminated / total

    # ── Overfitting score (Deflated Sharpe) ────────────────────────────────

    def compute_overfitting_score(
        self,
        in_sample_sharpe: float,
        oos_sharpe: float,
        n_obs_is: Optional[int] = None,
        n_obs_oos: Optional[int] = None,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> float:
        """
        Compute the Deflated Sharpe Ratio (DSR) following Bailey et al. (2014).

        DSR adjusts the observed Sharpe for:
          * Finite-sample bias
          * Non-normality of returns (skew, excess kurtosis)
          * Multiple testing across strategy trials

        Formula (simplified):
          SR*  = E[max SR over n_trials] ≈ sqrt(2) * erfinv((n_trials-1)/(n_trials))
          DSR  = Phi( (SR - SR*) * sqrt(T) / sqrt(1 - gamma_3 SR + (gamma_4-1)/4 SR^2) )
        where gamma_3, gamma_4 are skew and excess kurtosis.

        Parameters
        ----------
        in_sample_sharpe : float
            Annualised Sharpe of the strategy on training data.
        oos_sharpe : float
            Annualised Sharpe on out-of-sample data.
        n_obs_is : int | None
            Number of observations in IS period.
        n_obs_oos : int | None
            Number of observations in OOS period.
        skewness : float
            Return distribution skewness.
        kurtosis : float
            Return distribution kurtosis (full, not excess).

        Returns
        -------
        float
            DSR in [0, 1].  Values < 0.05 imply the strategy is likely overfit.
        """
        if np.isnan(in_sample_sharpe) or np.isnan(oos_sharpe):
            return float("nan")

        n_trials = self.n_trials_equivalent
        n_obs = n_obs_oos or 252  # default 1 year of daily obs

        # Expected max Sharpe under multiple testing
        # Approximation from Bailey et al. Eq.(2)
        if n_trials > 1:
            expected_max_sr = (
                (1 - np.euler_gamma) * stats.norm.ppf(1 - 1 / n_trials)
                + np.euler_gamma * stats.norm.ppf(1 - 1 / (n_trials * np.e))
            )
        else:
            expected_max_sr = 0.0

        # Variance of the SR estimator (adjusted for non-normality)
        excess_kurt = kurtosis - 3.0
        sr_var = (1 - skewness * oos_sharpe + (excess_kurt / 4) * oos_sharpe ** 2) / n_obs
        sr_std = max(math.sqrt(max(sr_var, 0)), 1e-8)

        # DSR = Phi((SR - SR*) / sigma_SR)
        z = (oos_sharpe - expected_max_sr) / sr_std
        dsr = float(stats.norm.cdf(z))

        return dsr

    def overfitting_probability(
        self,
        is_sharpe: float,
        oos_sharpe: float,
        n_obs_is: int = 252,
        n_obs_oos: int = 252,
    ) -> float:
        """
        Estimate the probability that the strategy is overfit, defined as
        P(OOS Sharpe < 0 | IS Sharpe > threshold).

        Uses a Bayesian-style calculation based on the sampling distribution
        of the Sharpe ratio.

        Returns
        -------
        float
            Probability in [0, 1].
        """
        if np.isnan(is_sharpe) or np.isnan(oos_sharpe):
            return float("nan")

        # Standard error of SR estimator
        se_is = math.sqrt((1 + 0.5 * is_sharpe ** 2) / max(n_obs_is, 1))
        se_oos = math.sqrt((1 + 0.5 * oos_sharpe ** 2) / max(n_obs_oos, 1))

        # P(true SR ≤ 0 | observed OOS SR)
        p_overfit = float(stats.norm.cdf(0, loc=oos_sharpe, scale=se_oos))
        return p_overfit

    # ── Variance Inflation Factor ─────────────────────────────────────────

    def variance_inflation_factor(
        self,
        X: pd.DataFrame,
        max_condition: float = 1000.0,
    ) -> pd.Series:
        """
        Compute Variance Inflation Factors for each column in X.

        VIF_j = 1 / (1 - R²_j) where R²_j is the R² from regressing
        column j on all other columns.

        VIF > 5 indicates moderate multicollinearity;
        VIF > 10 indicates severe multicollinearity.

        Parameters
        ----------
        X : pd.DataFrame
            Factor / feature matrix (numeric, no NaN).
        max_condition : float
            If the condition number of X'X exceeds this, return NaN
            (matrix is near-singular).

        Returns
        -------
        pd.Series
            VIF for each column, indexed by column name.
        """
        if X.empty or X.shape[1] < 2:
            return pd.Series(dtype=float)

        X_clean = X.select_dtypes(include=[np.number]).dropna()
        if X_clean.shape[0] < X_clean.shape[1] + 2:
            warnings.warn(
                "Too few observations relative to features for VIF computation.",
                stacklevel=2,
            )
            return pd.Series(np.nan, index=X_clean.columns)

        # Check condition number
        try:
            cond = np.linalg.cond(X_clean.values)
        except np.linalg.LinAlgError:
            cond = float("inf")

        if cond > max_condition:
            warnings.warn(
                f"Near-singular feature matrix (condition={cond:.1f}); "
                "VIF may be unreliable.",
                stacklevel=2,
            )

        vif_values: dict[str, float] = {}
        for col in X_clean.columns:
            y = X_clean[col].values
            others = X_clean.drop(columns=[col]).values
            try:
                # OLS: y ~ others
                coef, residuals, rank, sv = np.linalg.lstsq(
                    np.column_stack([np.ones(len(others)), others]),
                    y,
                    rcond=None,
                )
                y_hat = np.column_stack([np.ones(len(others)), others]) @ coef
                ss_res = float(np.sum((y - y_hat) ** 2))
                ss_tot = float(np.sum((y - y.mean()) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                r2 = min(r2, 1.0 - 1e-10)  # prevent division by zero
                vif = 1.0 / (1.0 - r2)
            except (np.linalg.LinAlgError, ZeroDivisionError):
                vif = float("nan")
            vif_values[col] = vif

        return pd.Series(vif_values, name="VIF")

    # ── Newey-West autocorrelation adjustment ─────────────────────────────

    def autocorrelation_score(
        self,
        returns: pd.Series | np.ndarray,
        lags: int = 20,
    ) -> float:
        """
        Compute an autocorrelation score reflecting how much positive
        serial correlation in returns artificially inflates the Sharpe ratio.

        Uses the Newey-West (1987) variance estimator to obtain the
        autocorrelation-adjusted standard deviation of returns, then
        returns the ratio (naive_sharpe / nw_sharpe - 1) as the score.

        Score interpretation
        --------------------
        ~0.0  → no autocorrelation inflation
        >0.2  → significant inflation (strategy exploits autocorrelation)
        >0.5  → severe inflation; reported Sharpe is misleading

        Parameters
        ----------
        returns : array-like
            Return series (per-trade PnL or daily returns).
        lags : int
            Number of lags for Newey-West correction.

        Returns
        -------
        float
            Autocorrelation inflation score.
        """
        r = pd.to_numeric(pd.Series(returns), errors="coerce").dropna().values
        n = len(r)

        if n < lags + 2:
            return float("nan")

        # Naive variance
        r_demeaned = r - r.mean()
        naive_var = float(np.sum(r_demeaned ** 2) / n)

        # Newey-West variance: V_NW = V_0 + 2 * sum_{j=1}^{lags} w_j * gamma_j
        # where w_j = 1 - j / (lags + 1) (Bartlett kernel)
        nw_var = naive_var
        for j in range(1, lags + 1):
            weight = 1.0 - j / (lags + 1)
            gamma_j = float(np.sum(r_demeaned[:n - j] * r_demeaned[j:]) / n)
            nw_var += 2 * weight * gamma_j

        nw_var = max(nw_var, 1e-15)

        naive_sr = float(r.mean() / math.sqrt(naive_var)) if naive_var > 0 else 0.0
        nw_sr = float(r.mean() / math.sqrt(nw_var)) if nw_var > 0 else 0.0

        if abs(nw_sr) < 1e-10:
            return 0.0

        inflation = naive_sr / nw_sr - 1.0
        return float(inflation)

    def newey_west_sharpe(
        self,
        returns: pd.Series | np.ndarray,
        lags: int = 20,
        annual_factor: float = 252.0,
    ) -> float:
        """
        Compute the Newey-West autocorrelation-adjusted annualised Sharpe.

        Parameters
        ----------
        returns : array-like
        lags : int
        annual_factor : float

        Returns
        -------
        float
        """
        r = pd.to_numeric(pd.Series(returns), errors="coerce").dropna().values
        n = len(r)

        if n < lags + 2:
            return float("nan")

        r_demeaned = r - r.mean()
        nw_var = float(np.sum(r_demeaned ** 2) / n)
        for j in range(1, lags + 1):
            weight = 1.0 - j / (lags + 1)
            gamma_j = float(np.sum(r_demeaned[:n - j] * r_demeaned[j:]) / n)
            nw_var += 2 * weight * gamma_j

        nw_var = max(nw_var, 1e-15)
        return float(r.mean() / math.sqrt(nw_var) * math.sqrt(annual_factor))

    # ── Lookahead bias detection ──────────────────────────────────────────

    def detect_lookahead_bias(
        self,
        live_trades: pd.DataFrame,
        bt_trades: pd.DataFrame,
        overlap_threshold_days: float = 30.0,
        annual_factor: float = 252.0,
    ) -> dict[str, Any]:
        """
        Detect potential lookahead bias by comparing live and backtest
        performance in their overlapping calendar period.

        A significant outperformance of backtest over live in the same period
        is suspicious.

        Parameters
        ----------
        live_trades, bt_trades : pd.DataFrame
        overlap_threshold_days : float
            Minimum overlap in days to perform the comparison (default 30).

        Returns
        -------
        dict with live_sharpe, bt_sharpe, sharpe_excess, suspected,
            overlap_days, test_result
        """
        live_t = self._get_time_col(live_trades)
        bt_t = self._get_time_col(bt_trades)

        if not (live_t and bt_t):
            return {"suspected": False, "reason": "No time column"}

        live_times = pd.to_datetime(live_trades[live_t], utc=True, errors="coerce").dropna()
        bt_times = pd.to_datetime(bt_trades[bt_t], utc=True, errors="coerce").dropna()

        if len(live_times) == 0 or len(bt_times) == 0:
            return {"suspected": False, "reason": "No timestamp data"}

        # Find overlapping period
        overlap_start = max(live_times.min(), bt_times.min())
        overlap_end = min(live_times.max(), bt_times.max())

        if overlap_start >= overlap_end:
            return {"suspected": False, "reason": "No overlapping period"}

        overlap_days = (overlap_end - overlap_start).total_seconds() / 86400
        if overlap_days < overlap_threshold_days:
            return {
                "suspected": False,
                "reason": f"Overlap too short ({overlap_days:.0f} days)",
            }

        # Filter to overlapping period
        live_pnl_col = self._get_pnl_col(live_trades)
        bt_pnl_col = self._get_pnl_col(bt_trades)

        live_in = live_trades[
            (pd.to_datetime(live_trades[live_t], utc=True, errors="coerce") >= overlap_start)
            & (pd.to_datetime(live_trades[live_t], utc=True, errors="coerce") <= overlap_end)
        ]
        bt_in = bt_trades[
            (pd.to_datetime(bt_trades[bt_t], utc=True, errors="coerce") >= overlap_start)
            & (pd.to_datetime(bt_trades[bt_t], utc=True, errors="coerce") <= overlap_end)
        ]

        live_sr = self._compute_sharpe(live_in[live_pnl_col], annual_factor)
        bt_sr = self._compute_sharpe(bt_in[bt_pnl_col], annual_factor)

        if np.isnan(live_sr) or np.isnan(bt_sr):
            return {
                "suspected": False,
                "reason": "Insufficient data for Sharpe computation",
            }

        excess = (bt_sr - live_sr) / (abs(live_sr) + 1e-6)
        suspected = bool(bt_sr > live_sr * 1.5 and bt_sr > 0.5)

        # t-test on the PnL distributions
        live_p = pd.to_numeric(live_in[live_pnl_col], errors="coerce").dropna()
        bt_p = pd.to_numeric(bt_in[bt_pnl_col], errors="coerce").dropna()
        if len(live_p) >= 3 and len(bt_p) >= 3:
            t_stat, p_val = stats.ttest_ind(bt_p.values, live_p.values, equal_var=False)
        else:
            t_stat, p_val = float("nan"), float("nan")

        return {
            "suspected": suspected,
            "live_sharpe": live_sr,
            "bt_sharpe": bt_sr,
            "sharpe_excess": excess,
            "overlap_days": overlap_days,
            "t_stat": t_stat,
            "p_value": p_val,
            "n_live": len(live_in),
            "n_bt": len(bt_in),
        }

    # ── Full audit ────────────────────────────────────────────────────────

    def audit(
        self,
        live_trades: pd.DataFrame,
        bt_trades: pd.DataFrame,
        factor_matrix: Optional[pd.DataFrame] = None,
        in_sample_sharpe: Optional[float] = None,
        oos_sharpe: Optional[float] = None,
    ) -> LeakageReport:
        """
        Run the complete leakage and overfitting audit.

        Parameters
        ----------
        live_trades : pd.DataFrame
        bt_trades : pd.DataFrame
        factor_matrix : pd.DataFrame | None
            Feature/factor matrix for VIF analysis.
        in_sample_sharpe : float | None
            Pre-computed IS Sharpe.  If None, computed from bt_trades.
        oos_sharpe : float | None
            Pre-computed OOS (live) Sharpe.  If None, computed from live_trades.

        Returns
        -------
        LeakageReport
        """
        # Lookahead bias
        lookahead = self.detect_lookahead_bias(live_trades, bt_trades)
        leakage_score_val = float(lookahead.get("sharpe_excess", 0.0) or 0.0)
        leakage_score_val = float(np.clip(leakage_score_val, 0.0, 1.0))
        lookahead_suspected = bool(lookahead.get("suspected", False))

        live_sharpe_in_period = float(lookahead.get("live_sharpe", float("nan")))
        bt_sharpe_in_period = float(lookahead.get("bt_sharpe", float("nan")))
        sharpe_excess = float(lookahead.get("sharpe_excess", float("nan")))

        # Contamination score contributes to leakage
        contamination = self.check_future_data_contamination(bt_trades, live_trades)
        leakage_score_val = float(np.clip(
            leakage_score_val * 0.5 + contamination * 0.5, 0.0, 1.0
        ))

        # Purge analysis
        try:
            purged_df = self.purge_test(bt_trades)
            n_purged = int(purged_df["purged"].sum()) if "purged" in purged_df.columns else 0
            n_remaining = len(purged_df) - n_purged
        except Exception as exc:
            log.warning("Purge test failed: %s", exc)
            n_purged = 0
            n_remaining = len(bt_trades)

        purge_fraction = n_purged / max(len(bt_trades), 1)

        # Sharpe computation
        try:
            live_pnl_col = self._get_pnl_col(live_trades)
            live_pnl = pd.to_numeric(live_trades[live_pnl_col], errors="coerce").dropna()
        except Exception:
            live_pnl = pd.Series(dtype=float)

        try:
            bt_pnl_col = self._get_pnl_col(bt_trades)
            bt_pnl = pd.to_numeric(bt_trades[bt_pnl_col], errors="coerce").dropna()
        except Exception:
            bt_pnl = pd.Series(dtype=float)

        is_sr = in_sample_sharpe if in_sample_sharpe is not None else self._compute_sharpe(bt_pnl)
        oos_sr = oos_sharpe if oos_sharpe is not None else self._compute_sharpe(live_pnl)

        # Deflated Sharpe
        n_oos = max(len(live_pnl), 30)
        n_is = max(len(bt_pnl), 30)
        skew = float(stats.skew(live_pnl.values)) if len(live_pnl) > 3 else 0.0
        kurt = float(stats.kurtosis(live_pnl.values, fisher=False)) if len(live_pnl) > 3 else 3.0

        dsr = self.compute_overfitting_score(
            in_sample_sharpe=is_sr if not np.isnan(is_sr) else 0.0,
            oos_sharpe=oos_sr if not np.isnan(oos_sr) else 0.0,
            n_obs_is=n_is,
            n_obs_oos=n_oos,
            skewness=skew,
            kurtosis=kurt,
        )

        overfit_prob = self.overfitting_probability(
            is_sharpe=is_sr if not np.isnan(is_sr) else 0.0,
            oos_sharpe=oos_sr if not np.isnan(oos_sr) else 0.0,
            n_obs_is=n_is,
            n_obs_oos=n_oos,
        )

        # VIF
        vif_series = pd.Series(dtype=float)
        high_vif: list[str] = []
        if factor_matrix is not None and not factor_matrix.empty:
            vif_series = self.variance_inflation_factor(factor_matrix)
            high_vif = [str(k) for k, v in vif_series.items() if not np.isnan(v) and v > 5]

        # Autocorrelation adjustment
        pnl_for_ac = live_pnl if len(live_pnl) > 0 else bt_pnl
        ac_score = self.autocorrelation_score(pnl_for_ac, lags=20)
        nw_sr = self.newey_west_sharpe(pnl_for_ac, lags=20)

        return LeakageReport(
            leakage_score=leakage_score_val,
            lookahead_suspected=lookahead_suspected,
            live_sharpe_in_period=live_sharpe_in_period,
            bt_sharpe_in_period=bt_sharpe_in_period,
            sharpe_ratio_excess=sharpe_excess,
            n_purged=n_purged,
            n_remaining=n_remaining,
            purge_fraction=purge_fraction,
            in_sample_sharpe=is_sr if not np.isnan(is_sr) else 0.0,
            oos_sharpe=oos_sr if not np.isnan(oos_sr) else 0.0,
            deflated_sharpe=dsr if not np.isnan(dsr) else 0.0,
            overfitting_probability=overfit_prob if not np.isnan(overfit_prob) else 0.5,
            vif_series=vif_series,
            high_vif_factors=high_vif,
            ac_adjusted_sharpe=nw_sr if not np.isnan(nw_sr) else 0.0,
            autocorrelation_score=ac_score if not np.isnan(ac_score) else 0.0,
            metadata={
                "contamination_score": contamination,
                "lookahead_details": lookahead,
                "embargo_bars": self.embargo_bars,
                "n_trials_equivalent": self.n_trials_equivalent,
            },
        )

    # ── Plotting ──────────────────────────────────────────────────────────

    def plot_leakage_summary(
        self,
        report: LeakageReport,
        save_path: str | Path,
        dpi: int = 150,
    ) -> Path:
        """
        Summary plot of leakage and overfitting metrics.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # 1. Gauge-style leakage score
        ax = axes[0, 0]
        score = report.leakage_score
        color = "#4CAF50" if score < 0.3 else "#FF9800" if score < 0.6 else "#F44336"
        ax.barh(["Leakage Score"], [score], color=color, alpha=0.85)
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_title(f"Leakage Score: {score:.3f} {'⚠ SUSPECTED' if report.lookahead_suspected else '✓ OK'}")
        ax.text(score + 0.02, 0, f"{score:.3f}", va="center", fontsize=10)

        # 2. Sharpe comparison
        ax = axes[0, 1]
        labels = ["Live (OOS)", "Backtest (IS)"]
        vals = [report.oos_sharpe, report.in_sample_sharpe]
        colors = ["steelblue", "orange"]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85)
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Sharpe Comparison (DSR={report.deflated_sharpe:.3f})")
        ax.set_ylabel("Annualised Sharpe")

        # 3. VIF chart
        ax = axes[1, 0]
        if len(report.vif_series) > 0:
            vif_clean = report.vif_series.dropna().sort_values(ascending=False).head(15)
            vif_colors = ["#F44336" if v > 10 else "#FF9800" if v > 5 else "#4CAF50"
                          for v in vif_clean.values]
            bars = ax.barh(list(vif_clean.index), vif_clean.values, color=vif_colors, alpha=0.85)
            ax.axvline(5, color="orange", linestyle="--", linewidth=1, label="VIF=5")
            ax.axvline(10, color="red", linestyle="--", linewidth=1, label="VIF=10")
            ax.set_title("Variance Inflation Factors")
            ax.set_xlabel("VIF")
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "No VIF data", ha="center", va="center")
            ax.set_title("VIF")

        # 4. Overfitting probability gauge
        ax = axes[1, 1]
        op = report.overfitting_probability
        op_color = "#4CAF50" if op < 0.2 else "#FF9800" if op < 0.5 else "#F44336"
        ax.barh(["Overfit Prob"], [op], color=op_color, alpha=0.85)
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1)
        ax.set_xlim(0, 1)
        label = "⚠ HIGH RISK" if op > 0.5 else "✓ OK"
        ax.set_title(f"Overfitting Probability: {op:.1%} {label}")
        ax.text(op + 0.02, 0, f"{op:.1%}", va="center", fontsize=10)

        fig.suptitle("Data Leakage & Overfitting Audit", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        log.info("Leakage summary plot saved to %s", save_path)
        return save_path
