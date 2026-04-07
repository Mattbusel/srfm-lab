"""
regime_backtest.py -- Regime-conditioned performance analysis for LARSA.

Splits backtest P&L by BH regime (BH_ACTIVE vs INACTIVE) and Hurst regime
(trending vs mean-reverting). Computes per-regime statistics and transition
matrices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

BARS_PER_YEAR = 26 * 252  # 15-min bars


# ---------------------------------------------------------------------------
# Regime labels
# ---------------------------------------------------------------------------

BH_REGIMES = ["BH_ACTIVE", "BH_INACTIVE", "TRANSITIONING"]
HURST_REGIMES = ["TRENDING", "RANDOM_WALK", "MEAN_REVERTING"]


# ---------------------------------------------------------------------------
# Conditional Statistics helper
# ---------------------------------------------------------------------------

class ConditionalStats:
    """
    Computes performance statistics conditioned on a regime label series.

    Given a returns series and a regime label series (aligned by index),
    computes Sharpe, Sortino, hit rate, avg hold, and drawdown per regime.
    """

    def __init__(
        self,
        returns: pd.Series,
        regime_labels: pd.Series,
    ):
        self.returns = returns.dropna()
        self.regime_labels = regime_labels.reindex(self.returns.index).ffill()

    def per_regime_stats(self) -> pd.DataFrame:
        """Return a DataFrame with one row per regime."""
        regimes = self.regime_labels.unique()
        rows = []
        for regime in regimes:
            mask = self.regime_labels == regime
            r = self.returns[mask]
            if len(r) < 5:
                continue
            rows.append(self._stats_for_slice(r, str(regime)))
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("regime")

    def _stats_for_slice(self, r: pd.Series, regime: str) -> Dict[str, Any]:
        n = len(r)
        mean_r = float(r.mean())
        std_r = float(r.std())
        sharpe = mean_r / std_r * np.sqrt(BARS_PER_YEAR) if std_r > 1e-12 else 0.0

        downside = r[r < 0]
        downside_std = float(downside.std()) if len(downside) > 1 else 1e-12
        sortino = mean_r / downside_std * np.sqrt(BARS_PER_YEAR) if downside_std > 1e-12 else 0.0

        hit_rate = float((r > 0).mean())
        cumr = (1 + r).cumprod()
        peak = cumr.cummax()
        dd = (cumr - peak) / peak
        max_dd = float(dd.min())

        return {
            "regime": regime,
            "n_bars": n,
            "mean_return": mean_r,
            "annualized_return": mean_r * BARS_PER_YEAR,
            "annualized_vol": std_r * np.sqrt(BARS_PER_YEAR),
            "sharpe": sharpe,
            "sortino": sortino,
            "hit_rate": hit_rate,
            "max_drawdown": max_dd,
            "fraction_of_time": n / max(len(self.returns), 1),
        }

    def regime_contribution(self) -> pd.DataFrame:
        """Fraction of total P&L contributed by each regime."""
        total_pnl = self.returns.sum()
        rows = []
        for regime in self.regime_labels.unique():
            mask = self.regime_labels == regime
            regime_pnl = float(self.returns[mask].sum())
            rows.append({
                "regime": str(regime),
                "pnl": regime_pnl,
                "pnl_pct": regime_pnl / (total_pnl + 1e-12),
            })
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("regime")


# ---------------------------------------------------------------------------
# Regime Transition Matrix
# ---------------------------------------------------------------------------

class RegimeTransitionMatrix:
    """
    Computes the empirical transition matrix between regimes.

    P[i,j] = probability of transitioning from regime i to regime j
    in the next bar.
    """

    def __init__(self, regime_series: pd.Series):
        self.series = regime_series.dropna()
        self._regimes = sorted(self.series.unique().tolist())
        self._matrix: Optional[pd.DataFrame] = None

    def compute(self) -> pd.DataFrame:
        """Compute and return the normalized transition matrix."""
        regimes = self._regimes
        n = len(regimes)
        counts = pd.DataFrame(0, index=regimes, columns=regimes, dtype=float)

        s = self.series.values
        for i in range(len(s) - 1):
            from_r = s[i]
            to_r = s[i + 1]
            if from_r in counts.index and to_r in counts.columns:
                counts.loc[from_r, to_r] += 1

        # Normalize rows
        row_sums = counts.sum(axis=1)
        matrix = counts.div(row_sums + 1e-12, axis=0)
        self._matrix = matrix
        return matrix

    @property
    def matrix(self) -> pd.DataFrame:
        if self._matrix is None:
            self.compute()
        return self._matrix

    def stationary_distribution(self) -> pd.Series:
        """
        Compute the stationary distribution via the left eigenvector
        of the transition matrix corresponding to eigenvalue 1.
        """
        M = self.matrix.values
        n = len(M)
        # Solve (M^T - I) pi = 0 with constraint sum(pi) = 1
        A = (M.T - np.eye(n))
        A = np.vstack([A, np.ones(n)])
        b = np.zeros(n + 1)
        b[-1] = 1.0

        try:
            pi, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            pi = np.abs(pi)
            pi = pi / (pi.sum() + 1e-12)
        except np.linalg.LinAlgError:
            pi = np.ones(n) / n

        return pd.Series(pi, index=self.matrix.index, name="stationary")

    def mean_sojourn_time(self) -> pd.Series:
        """
        Expected time (bars) spent in each regime before transitioning.
        For a Markov chain: E[T_i] = 1 / (1 - P[i,i])
        """
        diag = pd.Series(
            {regime: self.matrix.loc[regime, regime] for regime in self._regimes}
        )
        sojourn = 1.0 / (1 - diag + 1e-12)
        return sojourn.rename("mean_sojourn_bars")

    def transition_summary(self) -> Dict[str, Any]:
        M = self.matrix
        stationary = self.stationary_distribution()
        sojourn = self.mean_sojourn_time()
        return {
            "transition_matrix": M,
            "stationary_distribution": stationary,
            "mean_sojourn_bars": sojourn,
        }


# ---------------------------------------------------------------------------
# BH Regime Analyzer
# ---------------------------------------------------------------------------

class RegimeAnalyzer:
    """
    Splits backtest P&L and metrics by BH regime (BH_ACTIVE vs BH_INACTIVE).

    Input: equity curve and a time-aligned regime label series.
    Output: per-regime performance metrics, transition matrix, and regime timeline.
    """

    def __init__(
        self,
        equity: pd.Series,
        bh_regime: Optional[pd.Series] = None,
        hurst_regime: Optional[pd.Series] = None,
        trade_log: Optional[pd.DataFrame] = None,
    ):
        self.equity = equity.dropna()
        self.returns = self.equity.pct_change().dropna()
        self.bh_regime = bh_regime
        self.hurst_regime = hurst_regime
        self.trade_log = trade_log if trade_log is not None else pd.DataFrame()

    def bh_conditioned_stats(self) -> Optional[pd.DataFrame]:
        """Per-regime performance stats conditioned on BH regime."""
        if self.bh_regime is None:
            logger.warning("No BH regime series provided")
            return None
        cs = ConditionalStats(self.returns, self.bh_regime)
        return cs.per_regime_stats()

    def hurst_conditioned_stats(self) -> Optional[pd.DataFrame]:
        """Per-regime performance stats conditioned on Hurst regime."""
        if self.hurst_regime is None:
            logger.warning("No Hurst regime series provided")
            return None
        cs = ConditionalStats(self.returns, self.hurst_regime)
        return cs.per_regime_stats()

    def bh_transition_matrix(self) -> Optional[pd.DataFrame]:
        if self.bh_regime is None:
            return None
        tm = RegimeTransitionMatrix(self.bh_regime)
        return tm.compute()

    def hurst_transition_matrix(self) -> Optional[pd.DataFrame]:
        if self.hurst_regime is None:
            return None
        tm = RegimeTransitionMatrix(self.hurst_regime)
        return tm.compute()

    def regime_timeline(self) -> pd.DataFrame:
        """Return a DataFrame showing regime labels over time."""
        cols = {}
        if self.bh_regime is not None:
            cols["bh_regime"] = self.bh_regime.reindex(self.returns.index).ffill()
        if self.hurst_regime is not None:
            cols["hurst_regime"] = self.hurst_regime.reindex(self.returns.index).ffill()
        cols["return"] = self.returns
        cols["equity"] = self.equity.reindex(self.returns.index)
        return pd.DataFrame(cols)

    def regime_duration_stats(self, regime_series: pd.Series) -> pd.DataFrame:
        """
        Compute run-length statistics for each regime.
        Returns mean, std, min, max duration per regime in bars.
        """
        runs = []
        current = None
        run_len = 0
        for label in regime_series:
            if label != current:
                if current is not None:
                    runs.append({"regime": current, "duration": run_len})
                current = label
                run_len = 1
            else:
                run_len += 1
        if current is not None:
            runs.append({"regime": current, "duration": run_len})

        if not runs:
            return pd.DataFrame()

        df = pd.DataFrame(runs)
        return df.groupby("regime")["duration"].agg(
            mean="mean", std="std", min="min", max="max", count="count"
        )

    def full_report(self) -> Dict[str, Any]:
        """Compile all regime analysis results."""
        report: Dict[str, Any] = {}

        bh_stats = self.bh_conditioned_stats()
        if bh_stats is not None:
            report["bh_conditioned_performance"] = bh_stats

        hurst_stats = self.hurst_conditioned_stats()
        if hurst_stats is not None:
            report["hurst_conditioned_performance"] = hurst_stats

        bh_tm = self.bh_transition_matrix()
        if bh_tm is not None:
            report["bh_transition_matrix"] = bh_tm
            tm = RegimeTransitionMatrix(self.bh_regime)
            report["bh_stationary_dist"] = tm.stationary_distribution()
            report["bh_sojourn_times"] = tm.mean_sojourn_time()

        hurst_tm = self.hurst_transition_matrix()
        if hurst_tm is not None:
            report["hurst_transition_matrix"] = hurst_tm

        if self.bh_regime is not None:
            report["bh_duration_stats"] = self.regime_duration_stats(
                self.bh_regime.reindex(self.returns.index).ffill().dropna()
            )
        if self.hurst_regime is not None:
            report["hurst_duration_stats"] = self.regime_duration_stats(
                self.hurst_regime.reindex(self.returns.index).ffill().dropna()
            )

        return report


# ---------------------------------------------------------------------------
# Hurst-Conditioned Performance Analyzer
# ---------------------------------------------------------------------------

class HurstConditioned:
    """
    Detailed performance analysis conditioned on the Hurst regime.

    Computes:
      - Per-regime Sharpe, Sortino, win rate
      - Regime-conditional trade statistics (from trade log)
      - Alpha/beta vs a benchmark (e.g., buy-and-hold) per regime
    """

    def __init__(
        self,
        returns: pd.Series,
        hurst_series: pd.Series,
        hurst_regime_series: pd.Series,
        trade_log: Optional[pd.DataFrame] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ):
        self.returns = returns.dropna()
        self.hurst = hurst_series.reindex(self.returns.index).ffill()
        self.regime = hurst_regime_series.reindex(self.returns.index).ffill()
        self.trade_log = trade_log
        self.benchmark = benchmark_returns

    def regime_performance(self) -> pd.DataFrame:
        cs = ConditionalStats(self.returns, self.regime)
        return cs.per_regime_stats()

    def regime_alpha_beta(self) -> Optional[pd.DataFrame]:
        """Compute alpha and beta vs benchmark per Hurst regime."""
        if self.benchmark is None:
            return None

        bm = self.benchmark.reindex(self.returns.index).fillna(0)
        rows = []
        for r_name in self.regime.unique():
            mask = self.regime == r_name
            ret_r = self.returns[mask]
            bm_r = bm[mask]
            if len(ret_r) < 10:
                continue
            # OLS: return = alpha + beta * benchmark + epsilon
            X = np.vstack([np.ones(len(bm_r)), bm_r.values]).T
            try:
                coeffs, _, _, _ = np.linalg.lstsq(X, ret_r.values, rcond=None)
                alpha, beta = coeffs[0], coeffs[1]
            except np.linalg.LinAlgError:
                alpha, beta = 0.0, 0.0

            rows.append({
                "regime": str(r_name),
                "alpha": float(alpha * BARS_PER_YEAR),
                "beta": float(beta),
                "n_bars": len(ret_r),
            })

        if not rows:
            return None
        return pd.DataFrame(rows).set_index("regime")

    def hurst_vs_performance(self, n_buckets: int = 5) -> pd.DataFrame:
        """
        Bin the Hurst series into quantiles and compute performance per bucket.
        Reveals the relationship between Hurst value and strategy edge.
        """
        valid = pd.DataFrame({"return": self.returns, "hurst": self.hurst}).dropna()
        if len(valid) < n_buckets * 10:
            return pd.DataFrame()

        valid["hurst_bucket"] = pd.qcut(valid["hurst"], n_buckets, labels=False, duplicates="drop")
        rows = []
        for bucket in sorted(valid["hurst_bucket"].unique()):
            subset = valid[valid["hurst_bucket"] == bucket]
            r = subset["return"]
            h = subset["hurst"]
            std = float(r.std())
            sharpe = float(r.mean() / std * np.sqrt(BARS_PER_YEAR)) if std > 1e-12 else 0.0
            rows.append({
                "hurst_bucket": bucket,
                "hurst_mean": float(h.mean()),
                "hurst_min": float(h.min()),
                "hurst_max": float(h.max()),
                "n_bars": len(r),
                "sharpe": sharpe,
                "hit_rate": float((r > 0).mean()),
                "avg_return": float(r.mean()),
            })

        return pd.DataFrame(rows).set_index("hurst_bucket")

    def conditional_trade_stats(self) -> Optional[pd.DataFrame]:
        """Trade statistics split by Hurst regime at entry."""
        if self.trade_log is None or self.trade_log.empty:
            return None
        if "hurst_entry" not in self.trade_log.columns:
            return None

        def classify_hurst(h: float) -> str:
            if h > 0.6:
                return "TRENDING"
            elif h < 0.4:
                return "MEAN_REVERTING"
            else:
                return "RANDOM_WALK"

        df = self.trade_log.copy()
        df["hurst_regime_at_entry"] = df["hurst_entry"].apply(classify_hurst)

        return df.groupby("hurst_regime_at_entry").agg(
            n_trades=("net_pnl", "count"),
            total_pnl=("net_pnl", "sum"),
            avg_pnl=("net_pnl", "mean"),
            win_rate=("net_pnl", lambda x: (x > 0).mean()),
            avg_hold_bars=("hold_bars", "mean") if "hold_bars" in df.columns else ("net_pnl", "count"),
        )
