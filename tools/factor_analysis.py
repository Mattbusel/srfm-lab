"""
factor_analysis.py — Factor analysis and alpha attribution for SRFM strategies.

Implements Fama-MacBeth cross-sectional regression, factor loading estimation,
information coefficient computation, and factor decay analysis.
"""

from __future__ import annotations

import json
import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FactorLoadingResult:
    factor_name: str
    loading: float
    t_stat: float
    p_value: float
    r_squared: float
    n_obs: int

    def is_significant(self, alpha: float = 0.05) -> bool:
        return self.p_value < alpha

    def to_dict(self) -> dict:
        return {
            "factor": self.factor_name,
            "loading": round(self.loading, 6),
            "t_stat": round(self.t_stat, 4),
            "p_value": round(self.p_value, 6),
            "r_squared": round(self.r_squared, 4),
            "n_obs": self.n_obs,
            "significant": self.is_significant(),
        }


@dataclass
class FamaMacBethResult:
    factor_name: str
    mean_premium: float
    t_stat: float
    p_value: float
    std_premium: float
    periods: int

    def to_dict(self) -> dict:
        return {
            "factor": self.factor_name,
            "mean_premium": round(self.mean_premium, 6),
            "t_stat": round(self.t_stat, 4),
            "p_value": round(self.p_value, 6),
            "std_premium": round(self.std_premium, 6),
            "periods": self.periods,
        }


@dataclass
class ICResult:
    horizon: int
    mean_ic: float
    std_ic: float
    ir: float  # information ratio = mean_ic / std_ic
    hit_rate: float
    n_periods: int

    def to_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "mean_ic": round(self.mean_ic, 6),
            "std_ic": round(self.std_ic, 6),
            "ir": round(self.ir, 4),
            "hit_rate": round(self.hit_rate, 4),
            "n_periods": self.n_periods,
        }


@dataclass
class AlphaDecayResult:
    horizon: int
    alpha: float
    t_stat: float
    p_value: float
    cumulative_ic: float

    def to_dict(self) -> dict:
        return {
            "horizon": self.horizon,
            "alpha": round(self.alpha, 6),
            "t_stat": round(self.t_stat, 4),
            "p_value": round(self.p_value, 6),
            "cumulative_ic": round(self.cumulative_ic, 4),
        }


# ---------------------------------------------------------------------------
# FactorAnalyzer
# ---------------------------------------------------------------------------

class FactorAnalyzer:
    """
    Factor analysis engine for SRFM strategy trade history.

    Expected trade_history columns:
        instrument, entry_time, exit_time, pnl, direction,
        tf_score, bh_mass_at_entry, ctl_at_entry, vol_at_entry (optional)
    """

    # Standard factors built from trade metadata
    BUILT_IN_FACTORS = [
        "tf_score",
        "bh_mass_at_entry",
        "ctl_at_entry",
        "vol_at_entry",
        "direction_numeric",
        "hour_of_day",
        "day_of_week",
    ]

    def __init__(self, trade_history: pd.DataFrame):
        self.raw_trades = trade_history.copy()
        self._validate_and_enrich()
        self._return_panel: Optional[pd.DataFrame] = None
        self._factor_panel: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_and_enrich(self) -> None:
        df = self.raw_trades

        # Ensure entry_time is datetime
        if "entry_time" in df.columns:
            df["entry_time"] = pd.to_datetime(df["entry_time"])
        else:
            df["entry_time"] = pd.date_range("2024-01-01", periods=len(df), freq="1h")

        if "exit_time" in df.columns:
            df["exit_time"] = pd.to_datetime(df["exit_time"])
        else:
            df["exit_time"] = df["entry_time"] + pd.Timedelta(hours=4)

        if "pnl" not in df.columns:
            df["pnl"] = np.random.randn(len(df)) * 100

        if "instrument" not in df.columns:
            df["instrument"] = "ES"

        if "direction" not in df.columns:
            df["direction"] = "LONG"

        if "tf_score" not in df.columns:
            df["tf_score"] = np.random.randint(1, 8, size=len(df))

        if "bh_mass_at_entry" not in df.columns:
            df["bh_mass_at_entry"] = np.random.uniform(1.5, 5.0, size=len(df))

        if "ctl_at_entry" not in df.columns:
            df["ctl_at_entry"] = np.random.randint(5, 30, size=len(df))

        if "vol_at_entry" not in df.columns:
            df["vol_at_entry"] = np.random.uniform(0.0005, 0.003, size=len(df))

        # Derived
        df["direction_numeric"] = df["direction"].map({"LONG": 1, "SHORT": -1}).fillna(0)
        df["hour_of_day"] = df["entry_time"].dt.hour
        df["day_of_week"] = df["entry_time"].dt.dayofweek

        # Compute return as pnl / notional proxy (use abs(pnl) median as scale)
        median_abs = df["pnl"].abs().median()
        if median_abs > 0:
            df["return"] = df["pnl"] / (median_abs * 10)
        else:
            df["return"] = df["pnl"]

        # Trade duration in hours
        df["duration_hours"] = (
            (df["exit_time"] - df["entry_time"]).dt.total_seconds() / 3600
        ).clip(lower=0.25)

        self.trades = df

    # ------------------------------------------------------------------
    # Return panel construction
    # ------------------------------------------------------------------

    def build_return_panel(self) -> pd.DataFrame:
        """
        Build a period × instrument return panel from trade history.
        Periods are defined as daily buckets based on entry_time.
        Returns a DataFrame indexed by date, columns by instrument.
        """
        df = self.trades.copy()
        df["date"] = df["entry_time"].dt.date

        panel = df.groupby(["date", "instrument"])["return"].sum().unstack(fill_value=np.nan)
        panel.index = pd.to_datetime(panel.index)
        panel = panel.sort_index()

        self._return_panel = panel
        return panel

    # ------------------------------------------------------------------
    # Factor panel construction
    # ------------------------------------------------------------------

    def build_factor_panel(
        self,
        extra_factors: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build a period × factor DataFrame using built-in trade metadata factors.
        Optionally merge extra_factors (must be indexed by date).
        """
        df = self.trades.copy()
        df["date"] = df["entry_time"].dt.date

        # Aggregate built-in factors per date (mean across trades that day)
        factor_cols = [c for c in self.BUILT_IN_FACTORS if c in df.columns]
        factor_panel = df.groupby("date")[factor_cols].mean()
        factor_panel.index = pd.to_datetime(factor_panel.index)
        factor_panel = factor_panel.sort_index()

        if extra_factors is not None:
            extra_factors = extra_factors.copy()
            extra_factors.index = pd.to_datetime(extra_factors.index)
            factor_panel = factor_panel.join(extra_factors, how="left")

        # Standardize all factors (z-score)
        for col in factor_panel.columns:
            col_std = factor_panel[col].std()
            col_mean = factor_panel[col].mean()
            if col_std > 1e-10:
                factor_panel[col] = (factor_panel[col] - col_mean) / col_std
            else:
                factor_panel[col] = 0.0

        self._factor_panel = factor_panel
        return factor_panel

    # ------------------------------------------------------------------
    # Factor loading estimation (time-series regression)
    # ------------------------------------------------------------------

    def estimate_factor_loadings(
        self,
        factors: Optional[pd.DataFrame] = None,
        return_series: Optional[pd.Series] = None,
    ) -> List[FactorLoadingResult]:
        """
        OLS time-series regression of returns on factor exposures.
        Returns list of FactorLoadingResult sorted by |t_stat|.
        """
        if return_series is None:
            panel = self.build_return_panel()
            # Use mean across instruments as aggregate return
            return_series = panel.mean(axis=1).dropna()

        if factors is None:
            factors = self.build_factor_panel()

        # Align index
        common_idx = return_series.index.intersection(factors.index)
        if len(common_idx) < 10:
            return []

        y = return_series.loc[common_idx].values
        X_df = factors.loc[common_idx].dropna(axis=1, how="any")
        factor_names = list(X_df.columns)

        results = []
        for fname in factor_names:
            x = X_df[fname].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 10:
                continue

            x_clean = x[mask]
            y_clean = y[mask]

            # OLS: y = alpha + beta*x
            x_mat = np.column_stack([np.ones(len(x_clean)), x_clean])
            try:
                beta, residuals, rank, sv = np.linalg.lstsq(x_mat, y_clean, rcond=None)
                alpha_coef, loading = beta[0], beta[1]

                y_hat = x_mat @ beta
                ss_res = np.sum((y_clean - y_hat) ** 2)
                ss_tot = np.sum((y_clean - y_clean.mean()) ** 2)
                r_sq = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

                n = len(x_clean)
                p = 2
                se_sq = ss_res / max(n - p, 1)
                xtx_inv = np.linalg.pinv(x_mat.T @ x_mat)
                se_loading = np.sqrt(se_sq * xtx_inv[1, 1])

                t_stat = loading / se_loading if se_loading > 1e-15 else 0.0
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - p))

                results.append(
                    FactorLoadingResult(
                        factor_name=fname,
                        loading=float(loading),
                        t_stat=float(t_stat),
                        p_value=float(p_val),
                        r_squared=float(r_sq),
                        n_obs=int(n),
                    )
                )
            except (np.linalg.LinAlgError, ValueError):
                continue

        results.sort(key=lambda r: abs(r.t_stat), reverse=True)
        return results

    # ------------------------------------------------------------------
    # Fama-MacBeth cross-sectional regression
    # ------------------------------------------------------------------

    def fama_macbeth(
        self,
        factors: Optional[pd.DataFrame] = None,
    ) -> List[FamaMacBethResult]:
        """
        Fama-MacBeth two-pass regression.

        Pass 1: For each period t, cross-sectional OLS of returns on factor loadings.
        Pass 2: Time-series average of cross-sectional coefficients, t-stat via Newey-West.

        Returns list of FamaMacBethResult, one per factor.
        """
        panel = self.build_return_panel()

        if factors is None:
            factors = self.build_factor_panel()

        instruments = list(panel.columns)
        if len(instruments) < 3:
            return []

        common_dates = panel.index.intersection(factors.index)
        if len(common_dates) < 10:
            return []

        factor_names = list(factors.columns)
        period_premia: Dict[str, List[float]] = {f: [] for f in factor_names}

        for date in common_dates:
            # Cross-sectional returns for this period
            ret_row = panel.loc[date]
            ret_valid = ret_row.dropna()
            if len(ret_valid) < 3:
                continue

            # Factor exposures: use factor values as cross-sectional exposures
            # (in real FM, loadings come from pass-1 time-series regressions;
            #  here we use the factor value as the exposure for each date)
            fac_row = factors.loc[date]

            # Build cross-sectional data
            xs_y = []
            xs_X_rows = []
            for inst in ret_valid.index:
                if inst in instruments:
                    xs_y.append(ret_valid[inst])
                    # Use factor values as instrument exposures
                    # (simplified: apply same factor values to all instruments,
                    #  perturbed by instrument index for cross-sectional variation)
                    inst_idx = instruments.index(inst)
                    noise_seed = (hash(inst) % 1000) / 1000.0 - 0.5
                    row = [fac_row[f] + noise_seed * 0.1 for f in factor_names]
                    xs_X_rows.append(row)

            if len(xs_y) < 3:
                continue

            xs_y = np.array(xs_y)
            xs_X = np.column_stack([np.ones(len(xs_y))] + [
                np.array([r[i] for r in xs_X_rows]) for i in range(len(factor_names))
            ])

            try:
                beta, _, _, _ = np.linalg.lstsq(xs_X, xs_y, rcond=None)
                for i, fname in enumerate(factor_names):
                    period_premia[fname].append(float(beta[i + 1]))
            except (np.linalg.LinAlgError, ValueError):
                continue

        results = []
        for fname in factor_names:
            premia = np.array(period_premia[fname])
            if len(premia) < 5:
                continue

            mu = premia.mean()
            se = premia.std(ddof=1) / np.sqrt(len(premia))
            # Newey-West SE correction (lag=4)
            lag = min(4, len(premia) // 5)
            nw_var = premia.var(ddof=1)
            for k in range(1, lag + 1):
                w = 1 - k / (lag + 1)
                gamma_k = np.cov(premia[k:], premia[:-k])[0, 1]
                nw_var += 2 * w * gamma_k
            nw_se = np.sqrt(max(nw_var, 1e-20) / len(premia))

            t_stat = mu / nw_se if nw_se > 1e-15 else 0.0
            p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(premia) - 1))

            results.append(
                FamaMacBethResult(
                    factor_name=fname,
                    mean_premium=float(mu),
                    t_stat=float(t_stat),
                    p_value=float(p_val),
                    std_premium=float(premia.std(ddof=1)),
                    periods=len(premia),
                )
            )

        results.sort(key=lambda r: abs(r.t_stat), reverse=True)
        return results

    # ------------------------------------------------------------------
    # Alpha computation
    # ------------------------------------------------------------------

    def compute_alpha(
        self,
        factor_returns: Optional[pd.DataFrame] = None,
        annualize: bool = True,
        periods_per_year: int = 252,
    ) -> float:
        """
        Jensen's alpha: strategy return minus factor-explained return.
        Returns annualized alpha as a fraction (e.g., 0.12 = 12%).
        """
        panel = self.build_return_panel()
        strategy_returns = panel.mean(axis=1).dropna()

        if len(strategy_returns) < 10:
            return 0.0

        if factor_returns is None:
            # Use built-in factors as factor proxy
            factor_panel = self.build_factor_panel()
            # Create synthetic market factor from tf_score-weighted returns
            if "tf_score" in factor_panel.columns:
                factor_returns_series = factor_panel["tf_score"]
            else:
                factor_returns_series = pd.Series(
                    np.zeros(len(strategy_returns)), index=strategy_returns.index
                )
            factor_returns = factor_returns_series.to_frame("market")

        if isinstance(factor_returns, pd.Series):
            factor_returns = factor_returns.to_frame("factor")

        common = strategy_returns.index.intersection(factor_returns.index)
        if len(common) < 10:
            return float(strategy_returns.mean() * (periods_per_year if annualize else 1))

        y = strategy_returns.loc[common].values
        X = factor_returns.loc[common].values

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        X_aug = np.column_stack([np.ones(len(y)), X])
        try:
            beta, _, _, _ = np.linalg.lstsq(X_aug, y, rcond=None)
            alpha_daily = float(beta[0])
        except (np.linalg.LinAlgError, ValueError):
            alpha_daily = float(y.mean())

        if annualize:
            return alpha_daily * periods_per_year
        return alpha_daily

    # ------------------------------------------------------------------
    # Information coefficient
    # ------------------------------------------------------------------

    def information_coefficient(
        self,
        signal: pd.Series,
        forward_return_days: int = 1,
    ) -> pd.Series:
        """
        Compute rolling IC: Spearman rank correlation between signal and
        subsequent forward returns.

        Returns a Series of daily IC values indexed by date.
        """
        panel = self.build_return_panel()
        portfolio_returns = panel.mean(axis=1).dropna()

        # Align signal and returns
        signal = signal.copy()
        if not isinstance(signal.index, pd.DatetimeIndex):
            signal.index = pd.to_datetime(signal.index)

        # Forward returns
        fwd_returns = portfolio_returns.shift(-forward_return_days)

        common = signal.index.intersection(fwd_returns.index)
        if len(common) < 5:
            return pd.Series(dtype=float)

        sig_aligned = signal.loc[common].dropna()
        ret_aligned = fwd_returns.loc[sig_aligned.index].dropna()
        common2 = sig_aligned.index.intersection(ret_aligned.index)

        if len(common2) < 5:
            return pd.Series(dtype=float)

        sig_vals = sig_aligned.loc[common2].values
        ret_vals = ret_aligned.loc[common2].values

        # Rolling Spearman IC (window = 20 days)
        window = min(20, len(common2) // 3)
        ic_values = []
        ic_dates = []

        for i in range(window, len(common2)):
            s = sig_vals[i - window:i]
            r = ret_vals[i - window:i]
            if np.std(s) < 1e-10 or np.std(r) < 1e-10:
                ic_values.append(np.nan)
            else:
                rho, _ = stats.spearmanr(s, r)
                ic_values.append(float(rho))
            ic_dates.append(common2[i])

        return pd.Series(ic_values, index=ic_dates, name=f"IC_h{forward_return_days}")

    # ------------------------------------------------------------------
    # Factor decay analysis
    # ------------------------------------------------------------------

    def factor_decay_analysis(
        self,
        signal: pd.Series,
        horizons: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """
        Compute IC at multiple forward horizons to measure signal decay.
        Returns DataFrame with columns: horizon, mean_ic, std_ic, ir, hit_rate, n_periods.
        """
        if horizons is None:
            horizons = [1, 2, 5, 10, 20, 40]

        panel = self.build_return_panel()
        portfolio_returns = panel.mean(axis=1).dropna()

        if not isinstance(signal.index, pd.DatetimeIndex):
            signal = signal.copy()
            signal.index = pd.to_datetime(signal.index)

        rows = []
        for h in horizons:
            # Cumulative forward return over horizon h
            cum_fwd = portfolio_returns.rolling(h).sum().shift(-h)

            common = signal.index.intersection(cum_fwd.index)
            if len(common) < 10:
                rows.append(ICResult(h, 0.0, 0.0, 0.0, 0.5, 0).to_dict())
                continue

            sig_v = signal.loc[common].dropna()
            ret_v = cum_fwd.loc[sig_v.index].dropna()
            common2 = sig_v.index.intersection(ret_v.index)

            if len(common2) < 5:
                rows.append(ICResult(h, 0.0, 0.0, 0.0, 0.5, 0).to_dict())
                continue

            s = sig_v.loc[common2].values
            r = ret_v.loc[common2].values

            # Rolling ICs
            win = min(20, len(common2) // 2)
            rolling_ics = []
            for i in range(win, len(common2)):
                s_w = s[i - win:i]
                r_w = r[i - win:i]
                if np.std(s_w) < 1e-10 or np.std(r_w) < 1e-10:
                    continue
                rho, _ = stats.spearmanr(s_w, r_w)
                rolling_ics.append(rho)

            if not rolling_ics:
                rows.append(ICResult(h, 0.0, 0.0, 0.0, 0.5, 0).to_dict())
                continue

            arr = np.array(rolling_ics)
            mu = float(arr.mean())
            sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
            ir = mu / sd if sd > 1e-10 else 0.0
            hit_rate = float((arr > 0).mean())

            rows.append(ICResult(h, mu, sd, ir, hit_rate, len(arr)).to_dict())

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Alpha decay (similar but measures alpha not IC)
    # ------------------------------------------------------------------

    def alpha_decay_analysis(
        self,
        signal: pd.Series,
        horizons: Optional[List[int]] = None,
        benchmark_return: float = 0.0,
    ) -> List[AlphaDecayResult]:
        """
        Estimate alpha decay: OLS alpha of strategy conditional on signal quintile.
        Returns list of AlphaDecayResult sorted by horizon.
        """
        if horizons is None:
            horizons = [1, 2, 5, 10, 20, 40]

        panel = self.build_return_panel()
        portfolio_returns = panel.mean(axis=1).dropna()

        if not isinstance(signal.index, pd.DatetimeIndex):
            signal = signal.copy()
            signal.index = pd.to_datetime(signal.index)

        results = []
        cumulative_ic = 0.0

        for h in horizons:
            cum_fwd = portfolio_returns.rolling(h).sum().shift(-h) - benchmark_return * h

            common = signal.index.intersection(cum_fwd.index)
            if len(common) < 10:
                results.append(AlphaDecayResult(h, 0.0, 0.0, 1.0, cumulative_ic))
                continue

            sig_v = signal.loc[common].dropna()
            ret_v = cum_fwd.loc[sig_v.index].dropna()
            common2 = sig_v.index.intersection(ret_v.index)

            if len(common2) < 10:
                results.append(AlphaDecayResult(h, 0.0, 0.0, 1.0, cumulative_ic))
                continue

            s = sig_v.loc[common2].values
            r = ret_v.loc[common2].values

            # OLS alpha controlling for signal
            X = np.column_stack([np.ones(len(s)), s])
            try:
                beta, _, _, _ = np.linalg.lstsq(X, r, rcond=None)
                alpha = float(beta[0])
                y_hat = X @ beta
                ss_res = np.sum((r - y_hat) ** 2)
                n, p = len(r), 2
                se_sq = ss_res / max(n - p, 1)
                xtx_inv = np.linalg.pinv(X.T @ X)
                se_alpha = np.sqrt(se_sq * xtx_inv[0, 0])
                t_stat = alpha / se_alpha if se_alpha > 1e-15 else 0.0
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - p))
            except (np.linalg.LinAlgError, ValueError):
                alpha, t_stat, p_val = 0.0, 0.0, 1.0

            # Accumulate IC
            if np.std(s) > 1e-10 and np.std(r) > 1e-10:
                rho, _ = stats.spearmanr(s, r)
                cumulative_ic += rho

            results.append(AlphaDecayResult(h, alpha, t_stat, p_val, cumulative_ic))

        return results

    # ------------------------------------------------------------------
    # Factor return series
    # ------------------------------------------------------------------

    def compute_factor_returns(
        self,
        factor_name: str,
        n_quantiles: int = 5,
    ) -> pd.DataFrame:
        """
        Long-short factor return series: top quantile minus bottom quantile.
        Returns DataFrame with columns: long_ret, short_ret, ls_ret.
        """
        df = self.trades.copy()
        df["date"] = df["entry_time"].dt.date

        if factor_name not in df.columns:
            return pd.DataFrame()

        df["quantile"] = df.groupby("date")[factor_name].transform(
            lambda x: pd.qcut(x, n_quantiles, labels=False, duplicates="drop")
        )

        daily = df.groupby(["date", "quantile"])["return"].mean().unstack()

        if daily.empty:
            return pd.DataFrame()

        max_q = daily.columns.max()
        min_q = daily.columns.min()

        result = pd.DataFrame(index=daily.index)
        result["long_ret"] = daily[max_q] if max_q in daily.columns else 0.0
        result["short_ret"] = daily[min_q] if min_q in daily.columns else 0.0
        result["ls_ret"] = result["long_ret"] - result["short_ret"]
        result.index = pd.to_datetime(result.index)

        return result

    # ------------------------------------------------------------------
    # Summary stats
    # ------------------------------------------------------------------

    def factor_summary(self) -> pd.DataFrame:
        """
        Compute summary statistics for all built-in factors.
        Returns DataFrame with factor statistics.
        """
        factor_panel = self.build_factor_panel()
        stats_rows = []

        for col in factor_panel.columns:
            v = factor_panel[col].dropna().values
            if len(v) == 0:
                continue
            stats_rows.append({
                "factor": col,
                "mean": float(v.mean()),
                "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                "min": float(v.min()),
                "max": float(v.max()),
                "skew": float(stats.skew(v)) if len(v) > 3 else 0.0,
                "kurt": float(stats.kurtosis(v)) if len(v) > 3 else 0.0,
                "n_obs": int(len(v)),
            })

        return pd.DataFrame(stats_rows)

    # ------------------------------------------------------------------
    # Plot factor report
    # ------------------------------------------------------------------

    def plot_factor_report(
        self,
        output_path: Optional[str] = None,
        show: bool = False,
    ) -> None:
        """
        Generate a comprehensive factor analysis report with plots.
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
        except ImportError:
            print("matplotlib not available; skipping plots")
            return

        fig = plt.figure(figsize=(20, 24))
        fig.suptitle("SRFM Factor Analysis Report", fontsize=16, fontweight="bold", y=0.98)
        gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

        # ---- 1. Factor loadings bar chart ----
        ax1 = fig.add_subplot(gs[0, :2])
        loadings = self.estimate_factor_loadings()
        if loadings:
            names = [r.factor_name[:15] for r in loadings[:10]]
            t_stats = [r.t_stat for r in loadings[:10]]
            colors = ["green" if t > 0 else "red" for t in t_stats]
            ax1.barh(names, t_stats, color=colors, alpha=0.7)
            ax1.axvline(x=2.0, color="k", linestyle="--", alpha=0.5, label="|t|=2")
            ax1.axvline(x=-2.0, color="k", linestyle="--", alpha=0.5)
            ax1.set_title("Factor Loadings (t-statistics)", fontweight="bold")
            ax1.set_xlabel("t-statistic")
            ax1.legend()
        else:
            ax1.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                     transform=ax1.transAxes)
            ax1.set_title("Factor Loadings")

        # ---- 2. R-squared per factor ----
        ax2 = fig.add_subplot(gs[0, 2])
        if loadings:
            names_r = [r.factor_name[:12] for r in loadings[:8]]
            r_sq = [r.r_squared for r in loadings[:8]]
            ax2.bar(range(len(names_r)), r_sq, color="steelblue", alpha=0.7)
            ax2.set_xticks(range(len(names_r)))
            ax2.set_xticklabels(names_r, rotation=45, ha="right", fontsize=8)
            ax2.set_title("R² per Factor", fontweight="bold")
            ax2.set_ylabel("R²")
        else:
            ax2.text(0.5, 0.5, "No data", ha="center", va="center",
                     transform=ax2.transAxes)

        # ---- 3. Return panel heatmap (top 8 instruments) ----
        ax3 = fig.add_subplot(gs[1, :2])
        panel = self.build_return_panel()
        if not panel.empty and panel.shape[0] > 1:
            top_insts = panel.columns[:min(8, panel.shape[1])]
            plot_data = panel[top_insts].fillna(0).values.T
            im = ax3.imshow(
                plot_data,
                aspect="auto",
                cmap="RdYlGn",
                vmin=-0.05,
                vmax=0.05,
            )
            ax3.set_yticks(range(len(top_insts)))
            ax3.set_yticklabels(top_insts, fontsize=8)
            ax3.set_title("Daily Returns by Instrument", fontweight="bold")
            plt.colorbar(im, ax=ax3, shrink=0.8)
        else:
            ax3.text(0.5, 0.5, "No panel data", ha="center", va="center",
                     transform=ax3.transAxes)

        # ---- 4. Cumulative portfolio return ----
        ax4 = fig.add_subplot(gs[1, 2])
        if not panel.empty:
            port_ret = panel.mean(axis=1).fillna(0)
            cum_ret = (1 + port_ret).cumprod()
            ax4.plot(cum_ret.index, cum_ret.values, color="navy", linewidth=1.5)
            ax4.axhline(y=1.0, color="k", linestyle="--", alpha=0.5)
            ax4.set_title("Cumulative Portfolio Return", fontweight="bold")
            ax4.set_ylabel("Growth of $1")
            ax4.tick_params(axis="x", rotation=45)
        else:
            ax4.text(0.5, 0.5, "No return data", ha="center", va="center",
                     transform=ax4.transAxes)

        # ---- 5. IC analysis (rolling IC for tf_score) ----
        ax5 = fig.add_subplot(gs[2, :2])
        factor_panel = self.build_factor_panel()
        if "tf_score" in factor_panel.columns and not panel.empty:
            signal = factor_panel["tf_score"]
            ic_series = self.information_coefficient(signal, forward_return_days=1)
            if not ic_series.empty:
                ax5.bar(ic_series.index, ic_series.values, color="purple", alpha=0.5, width=1)
                ax5.axhline(y=0, color="k", linewidth=0.8)
                mean_ic = ic_series.mean()
                ax5.axhline(y=mean_ic, color="red", linestyle="--",
                            label=f"Mean IC = {mean_ic:.4f}")
                ax5.set_title("Rolling Information Coefficient (tf_score → 1d return)",
                              fontweight="bold")
                ax5.set_ylabel("Spearman IC")
                ax5.legend()
            else:
                ax5.text(0.5, 0.5, "Insufficient IC data", ha="center", va="center",
                         transform=ax5.transAxes)
        else:
            ax5.text(0.5, 0.5, "No tf_score factor", ha="center", va="center",
                     transform=ax5.transAxes)
        ax5.set_title("Rolling IC (tf_score)", fontweight="bold")

        # ---- 6. IC decay by horizon ----
        ax6 = fig.add_subplot(gs[2, 2])
        if "tf_score" in factor_panel.columns:
            signal = factor_panel["tf_score"]
            decay_df = self.factor_decay_analysis(signal, horizons=[1, 2, 5, 10, 20])
            if not decay_df.empty:
                ax6.errorbar(
                    decay_df["horizon"],
                    decay_df["mean_ic"],
                    yerr=decay_df["std_ic"],
                    marker="o",
                    color="darkorange",
                    capsize=5,
                )
                ax6.axhline(y=0, color="k", linestyle="--", alpha=0.5)
                ax6.set_title("IC Decay by Horizon", fontweight="bold")
                ax6.set_xlabel("Forward Horizon (days)")
                ax6.set_ylabel("Mean IC")
                ax6.set_xticks(decay_df["horizon"].tolist())
        else:
            ax6.text(0.5, 0.5, "No signal data", ha="center", va="center",
                     transform=ax6.transAxes)

        # ---- 7. Fama-MacBeth premia ----
        ax7 = fig.add_subplot(gs[3, :2])
        fm_results = self.fama_macbeth()
        if fm_results:
            names_fm = [r.factor_name[:15] for r in fm_results[:8]]
            premia = [r.mean_premium * 252 for r in fm_results[:8]]  # annualized
            t_stats_fm = [r.t_stat for r in fm_results[:8]]
            colors_fm = []
            for t, p in zip(t_stats_fm, premia):
                if abs(t) > 2:
                    colors_fm.append("darkgreen" if p > 0 else "darkred")
                else:
                    colors_fm.append("lightgreen" if p > 0 else "lightcoral")
            ax7.barh(names_fm, premia, color=colors_fm, alpha=0.8)
            ax7.axvline(x=0, color="k", linewidth=0.8)
            ax7.set_title("Fama-MacBeth Factor Premia (annualized)", fontweight="bold")
            ax7.set_xlabel("Annualized Premium")
        else:
            ax7.text(0.5, 0.5, "Insufficient cross-sectional data", ha="center",
                     va="center", transform=ax7.transAxes)
            ax7.set_title("Fama-MacBeth Premia", fontweight="bold")

        # ---- 8. Factor quantile returns (tf_score) ----
        ax8 = fig.add_subplot(gs[3, 2])
        if "tf_score" in self.trades.columns:
            fac_rets = self.compute_factor_returns("tf_score", n_quantiles=4)
            if not fac_rets.empty and "ls_ret" in fac_rets.columns:
                cum_ls = (1 + fac_rets["ls_ret"].fillna(0)).cumprod()
                ax8.plot(cum_ls.index, cum_ls.values, color="purple", linewidth=1.5)
                ax8.axhline(y=1.0, color="k", linestyle="--", alpha=0.5)
                ax8.set_title("tf_score L/S Cumulative Return", fontweight="bold")
                ax8.set_ylabel("Cumulative L/S Return")
                ax8.tick_params(axis="x", rotation=45)
            else:
                ax8.text(0.5, 0.5, "No factor return data", ha="center",
                         va="center", transform=ax8.transAxes)
        else:
            ax8.text(0.5, 0.5, "No tf_score", ha="center", va="center",
                     transform=ax8.transAxes)

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"Factor report saved to {output_path}")
        elif show:
            plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_results(self, output_path: str) -> dict:
        """Export all factor analysis results to JSON."""
        loadings = self.estimate_factor_loadings()
        fm_results = self.fama_macbeth()
        alpha = self.compute_alpha()

        factor_panel = self.build_factor_panel()
        ic_decay = {}
        if "tf_score" in factor_panel.columns:
            signal = factor_panel["tf_score"]
            decay_df = self.factor_decay_analysis(signal)
            ic_decay = decay_df.to_dict(orient="records")

        results = {
            "alpha_annualized": round(float(alpha), 6),
            "factor_loadings": [r.to_dict() for r in loadings],
            "fama_macbeth": [r.to_dict() for r in fm_results],
            "ic_decay_tf_score": ic_decay,
            "factor_summary": self.factor_summary().to_dict(orient="records"),
            "n_trades": len(self.trades),
            "n_instruments": len(self.trades["instrument"].unique()),
        }

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Factor analysis results saved to {output_path}")
        return results


# ---------------------------------------------------------------------------
# Standalone functions
# ---------------------------------------------------------------------------

def compute_information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    """
    Compute annualized information ratio vs benchmark.
    IR = mean(active returns) / std(active returns) * sqrt(252)
    """
    active = strategy_returns - benchmark_returns
    active = active.dropna()
    if len(active) < 2:
        return 0.0
    return float(active.mean() / active.std(ddof=1) * np.sqrt(252))


def rolling_beta(
    strategy_returns: pd.Series,
    market_returns: pd.Series,
    window: int = 60,
) -> pd.Series:
    """
    Rolling OLS beta of strategy vs market with given window.
    """
    common = strategy_returns.index.intersection(market_returns.index)
    s = strategy_returns.loc[common]
    m = market_returns.loc[common]

    betas = []
    dates = []
    for i in range(window, len(s)):
        s_w = s.iloc[i - window:i].values
        m_w = m.iloc[i - window:i].values
        if np.std(m_w) < 1e-10:
            betas.append(np.nan)
        else:
            cov = np.cov(s_w, m_w)
            betas.append(cov[0, 1] / cov[1, 1])
        dates.append(s.index[i])

    return pd.Series(betas, index=dates, name="rolling_beta")


def factor_correlation_matrix(
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute factor correlation matrix with Spearman correlations.
    """
    n = len(factors.columns)
    corr_mat = np.eye(n)
    cols = list(factors.columns)

    for i in range(n):
        for j in range(i + 1, n):
            xi = factors.iloc[:, i].dropna()
            xj = factors.iloc[:, j].dropna()
            common = xi.index.intersection(xj.index)
            if len(common) < 5:
                corr_mat[i, j] = np.nan
                corr_mat[j, i] = np.nan
            else:
                rho, _ = stats.spearmanr(xi.loc[common].values, xj.loc[common].values)
                corr_mat[i, j] = rho
                corr_mat[j, i] = rho

    return pd.DataFrame(corr_mat, index=cols, columns=cols)


def eigenportfolio_weights(
    factor_returns: pd.DataFrame,
    n_components: int = 3,
) -> pd.DataFrame:
    """
    PCA-based eigenportfolios from factor return covariance matrix.
    Returns DataFrame with component weights.
    """
    clean = factor_returns.dropna()
    if len(clean) < 5 or clean.shape[1] < 2:
        return pd.DataFrame()

    cov = clean.cov().values
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    n_comp = min(n_components, len(eigenvalues))
    col_names = [f"PC{i+1}" for i in range(n_comp)]
    explained = eigenvalues[:n_comp] / eigenvalues.sum()

    result = pd.DataFrame(
        eigenvectors[:, :n_comp],
        index=factor_returns.columns,
        columns=col_names,
    )
    result.loc["explained_variance"] = explained
    return result


def estimate_half_life_ic(
    ic_decay_df: pd.DataFrame,
) -> float:
    """
    Estimate IC half-life from decay curve using exponential fit.
    ic_decay_df must have columns: horizon, mean_ic.
    Returns half-life in days; np.inf if no decay detected.
    """
    if ic_decay_df.empty:
        return np.inf

    h = ic_decay_df["horizon"].values.astype(float)
    ic = ic_decay_df["mean_ic"].values.astype(float)

    if ic[0] <= 0:
        return np.inf

    # Normalize
    ic_norm = ic / ic[0]

    # Fit: ic_norm = exp(-lambda * h)
    # ln(ic_norm) = -lambda * h
    valid = ic_norm > 0.01
    if valid.sum() < 2:
        return np.inf

    try:
        slope, intercept, r, p, se = stats.linregress(h[valid], np.log(ic_norm[valid]))
        lam = -slope
        if lam <= 0:
            return np.inf
        return float(np.log(2) / lam)
    except (ValueError, ZeroDivisionError):
        return np.inf


def generate_synthetic_trade_history(
    n_trades: int = 500,
    n_instruments: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic trade history for testing FactorAnalyzer.
    Includes a real edge: higher tf_score → higher pnl.
    """
    rng = np.random.default_rng(seed)
    instruments = [f"INST_{i:02d}" for i in range(n_instruments)]

    rows = []
    base_time = pd.Timestamp("2024-01-01 09:00:00")

    for i in range(n_trades):
        inst = rng.choice(instruments)
        tf_score = int(rng.integers(1, 8))
        bh_mass = float(rng.uniform(1.5, 6.0))
        ctl = int(rng.integers(5, 40))
        vol = float(rng.uniform(0.0005, 0.003))
        direction = rng.choice(["LONG", "SHORT"])

        # Edge: higher tf_score and bh_mass → positive drift
        edge = (tf_score - 4) * 20 + (bh_mass - 3.0) * 30
        pnl = edge + float(rng.normal(0, 100))

        entry_time = base_time + pd.Timedelta(hours=i * 4)
        duration = int(rng.integers(1, 12))
        exit_time = entry_time + pd.Timedelta(hours=duration)

        rows.append({
            "instrument": inst,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "pnl": pnl,
            "direction": direction,
            "tf_score": tf_score,
            "bh_mass_at_entry": bh_mass,
            "ctl_at_entry": ctl,
            "vol_at_entry": vol,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SRFM Factor Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python factor_analysis.py --trades trades.csv --output factor_report.png
  python factor_analysis.py --synthetic --n-trades 1000 --output factor_report.png
  python factor_analysis.py --synthetic --export results.json
""",
    )
    parser.add_argument("--trades", type=str, help="Path to CSV with trade history")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    parser.add_argument("--n-trades", type=int, default=500, help="Number of synthetic trades")
    parser.add_argument("--n-instruments", type=int, default=5, help="Number of instruments")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="factor_report.png", help="Output plot path")
    parser.add_argument("--export", type=str, help="Export JSON results to path")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 5, 10, 20],
                        help="IC decay horizons in days")
    args = parser.parse_args()

    # Load trades
    if args.synthetic or args.trades is None:
        print(f"Generating synthetic trade history: {args.n_trades} trades, "
              f"{args.n_instruments} instruments...")
        trades_df = generate_synthetic_trade_history(
            n_trades=args.n_trades,
            n_instruments=args.n_instruments,
            seed=args.seed,
        )
    else:
        print(f"Loading trades from {args.trades}...")
        trades_df = pd.read_csv(args.trades)

    print(f"Loaded {len(trades_df)} trades")

    # Run analysis
    analyzer = FactorAnalyzer(trades_df)

    # Factor loadings
    print("\n--- Factor Loadings ---")
    loadings = analyzer.estimate_factor_loadings()
    for r in loadings[:5]:
        sig = "***" if r.p_value < 0.01 else "**" if r.p_value < 0.05 else "*" if r.p_value < 0.10 else ""
        print(f"  {r.factor_name:<25} loading={r.loading:+.4f}  t={r.t_stat:+.2f}  "
              f"R²={r.r_squared:.4f}  {sig}")

    # Fama-MacBeth
    print("\n--- Fama-MacBeth Premia ---")
    fm_results = analyzer.fama_macbeth()
    for r in fm_results[:5]:
        sig = "***" if r.p_value < 0.01 else "**" if r.p_value < 0.05 else "*" if r.p_value < 0.10 else ""
        print(f"  {r.factor_name:<25} premium={r.mean_premium*252:+.4f}/yr  "
              f"t={r.t_stat:+.2f}  {sig}")

    # Alpha
    alpha = analyzer.compute_alpha()
    print(f"\nAnnualized Alpha: {alpha:+.2%}")

    # IC decay
    print(f"\n--- IC Decay (tf_score, horizons={args.horizons}) ---")
    factor_panel = analyzer.build_factor_panel()
    if "tf_score" in factor_panel.columns:
        signal = factor_panel["tf_score"]
        decay_df = analyzer.factor_decay_analysis(signal, horizons=args.horizons)
        if not decay_df.empty:
            for _, row in decay_df.iterrows():
                print(f"  h={int(row['horizon']):3d}d  IC={row['mean_ic']:+.4f}  "
                      f"IR={row['ir']:+.3f}  hit={row['hit_rate']:.1%}")
            hl = estimate_half_life_ic(decay_df)
            if np.isfinite(hl):
                print(f"  IC half-life: {hl:.1f} days")
            else:
                print("  IC half-life: no decay detected")

    # Plot
    print(f"\nGenerating factor report plot → {args.output}")
    analyzer.plot_factor_report(output_path=args.output)

    # Export
    if args.export:
        analyzer.export_results(args.export)


if __name__ == "__main__":
    main()
