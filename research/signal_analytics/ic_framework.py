"""
research/signal_analytics/ic_framework.py
==========================================
Information Coefficient (IC) analysis framework.

Provides a full suite of IC calculation methods:
  - Cross-sectional IC (Pearson / Spearman / Kendall)
  - Rolling IC over time
  - IC decay across forward-return horizons
  - ICIR (risk-adjusted IC)
  - Regime-conditioned IC
  - IC t-test and autocorrelation diagnostics
  - Matplotlib visualisations (decay, rolling, distribution)

All public functions are pure (no global state) and accept pandas
objects so they compose naturally with the rest of the research stack.

Usage example
-------------
>>> from research.signal_analytics.ic_framework import ICCalculator
>>> calc = ICCalculator()
>>> ic_val = calc.compute_ic(signal_series, fwd_returns_series)
>>> decay  = calc.ic_decay(signal_df, returns_df, max_horizon=20)
>>> calc.plot_ic_decay(decay, save_path="results/ic_decay.png")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from statsmodels.graphics.tsaplots import plot_acf


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ICDecayResult:
    """Container for IC-decay analysis results."""
    horizons: List[int]
    ic_values: List[float]
    ic_stderr: List[float]
    half_life: float          # bars until IC halves
    decay_rate: float         # λ in IC(h)=IC(0)·exp(-λ·h)
    ic_at_zero: float         # extrapolated IC(0)
    r_squared: float          # goodness-of-fit of exponential
    n_obs: int

    @property
    def peak_ic(self) -> float:
        if not self.ic_values:
            return float("nan")
        return max(self.ic_values, key=abs)

    @property
    def peak_horizon(self) -> int:
        if not self.ic_values:
            return 0
        return self.horizons[np.argmax(np.abs(self.ic_values))]

    def to_series(self) -> pd.Series:
        return pd.Series(self.ic_values, index=self.horizons, name="IC")


@dataclass
class ICByQuantileResult:
    """IC decomposed by signal-strength quantile bucket."""
    quantile_labels: List[str]
    ic_values: List[float]
    spread: float   # Q_top - Q_bottom
    n_obs_per_q: List[int]


# ---------------------------------------------------------------------------
# Standalone convenience function (wraps ICCalculator)
# ---------------------------------------------------------------------------

def compute_ic(
    signal: pd.Series,
    forward_returns: pd.Series,
    method: Literal["pearson", "spearman", "kendall"] = "spearman",
) -> float:
    """Compute Information Coefficient between *signal* and *forward_returns*.

    Parameters
    ----------
    signal          : pd.Series — signal values (any scale)
    forward_returns : pd.Series — realised forward returns, same index
    method          : correlation family to use

    Returns
    -------
    float — IC value in [−1, +1]
    """
    return ICCalculator().compute_ic(signal, forward_returns, method=method)


def icir(ic_series: pd.Series) -> float:
    """IC Information Ratio = mean(IC) / std(IC)."""
    return ICCalculator().icir(ic_series)


def rolling_ic(
    signal_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    window: int = 60,
    method: Literal["pearson", "spearman", "kendall"] = "spearman",
) -> pd.Series:
    """Rolling IC computed in a sliding window."""
    return ICCalculator().rolling_ic(signal_df, returns_df, window=window, method=method)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class ICCalculator:
    """Information Coefficient analysis toolkit.

    Methods mirror the module-level standalone functions but are grouped
    for convenient instantiation and state sharing (e.g. cached aligned data).
    """

    # ------------------------------------------------------------------ #
    # Alignment helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _align(
        signal: pd.Series,
        forward_returns: pd.Series,
        dropna: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Align two series on index and drop NaN rows."""
        df = pd.concat({"sig": signal, "ret": forward_returns}, axis=1)
        if dropna:
            df = df.dropna()
        return df["sig"].values, df["ret"].values

    # ------------------------------------------------------------------ #
    # Core IC
    # ------------------------------------------------------------------ #

    def compute_ic(
        self,
        signal: pd.Series,
        forward_returns: pd.Series,
        method: Literal["pearson", "spearman", "kendall"] = "spearman",
    ) -> float:
        """Single cross-sectional IC.

        Parameters
        ----------
        signal          : signal values for one period
        forward_returns : corresponding realised returns
        method          : 'pearson' | 'spearman' | 'kendall'

        Returns
        -------
        float
        """
        sig, ret = self._align(signal, forward_returns)
        if len(sig) < 3:
            return float("nan")

        method = method.lower()  # type: ignore[assignment]
        if method == "pearson":
            r, _ = stats.pearsonr(sig, ret)
        elif method == "spearman":
            r, _ = stats.spearmanr(sig, ret)
        elif method == "kendall":
            r, _ = stats.kendalltau(sig, ret)
        else:
            raise ValueError(f"Unknown method '{method}'. Choose pearson/spearman/kendall.")
        return float(r)

    def rank_ic(
        self,
        signal: pd.Series,
        forward_returns: pd.Series,
    ) -> float:
        """Spearman rank IC — alias for compute_ic(method='spearman')."""
        return self.compute_ic(signal, forward_returns, method="spearman")

    # ------------------------------------------------------------------ #
    # Rolling IC
    # ------------------------------------------------------------------ #

    def rolling_ic(
        self,
        signal_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        window: int = 60,
        method: Literal["pearson", "spearman", "kendall"] = "spearman",
    ) -> pd.Series:
        """Compute rolling IC across a *time × assets* panel.

        Parameters
        ----------
        signal_df   : DataFrame[time × assets] — signal values
        returns_df  : DataFrame[time × assets] — forward returns
        window      : rolling look-back in periods
        method      : correlation method

        Returns
        -------
        pd.Series indexed by time with rolling IC values
        """
        # Align both frames to the same columns / index
        cols = signal_df.columns.intersection(returns_df.columns)
        idx = signal_df.index.intersection(returns_df.index)
        sig = signal_df.loc[idx, cols]
        ret = returns_df.loc[idx, cols]

        ic_values: list[float] = []
        ic_index: list = []

        for i in range(window - 1, len(idx)):
            window_sig = sig.iloc[i - window + 1 : i + 1].values.flatten()
            window_ret = ret.iloc[i - window + 1 : i + 1].values.flatten()

            mask = ~(np.isnan(window_sig) | np.isnan(window_ret))
            ws, wr = window_sig[mask], window_ret[mask]

            if len(ws) < 3:
                ic_values.append(float("nan"))
            else:
                if method == "pearson":
                    r, _ = stats.pearsonr(ws, wr)
                elif method == "spearman":
                    r, _ = stats.spearmanr(ws, wr)
                else:
                    r, _ = stats.kendalltau(ws, wr)
                ic_values.append(float(r))
            ic_index.append(idx[i])

        return pd.Series(ic_values, index=ic_index, name="rolling_ic")

    # ------------------------------------------------------------------ #
    # IC Decay
    # ------------------------------------------------------------------ #

    def ic_decay(
        self,
        signal_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        max_horizon: int = 20,
        method: Literal["pearson", "spearman", "kendall"] = "spearman",
    ) -> ICDecayResult:
        """Compute IC at each forward-return horizon 1 .. max_horizon.

        Parameters
        ----------
        signal_df    : DataFrame[time × assets] — signals at time t
        returns_df   : DataFrame[time × assets] — returns starting at t
                       (caller should shift so that returns_df.shift(-h) is
                        the h-bar forward return; OR pass a multi-horizon
                        returns_df with columns named as ints)
        max_horizon  : furthest horizon to test
        method       : IC correlation method

        Returns
        -------
        ICDecayResult with fitted exponential decay curve
        """
        cols = signal_df.columns.intersection(returns_df.columns)
        idx = signal_df.index.intersection(returns_df.index)
        sig_panel = signal_df.loc[idx, cols]
        ret_panel = returns_df.loc[idx, cols]

        horizons: list[int] = []
        ic_vals: list[float] = []
        ic_errs: list[float] = []

        for h in range(1, max_horizon + 1):
            # Shift returns backward by h to get h-bar forward returns
            fwd_ret = ret_panel.shift(-h)
            cross_ics: list[float] = []

            for t in range(len(idx)):
                s_t = sig_panel.iloc[t]
                r_t = fwd_ret.iloc[t] if t < len(fwd_ret) else None
                if r_t is None:
                    continue
                mask = ~(s_t.isna() | r_t.isna())
                sv, rv = s_t[mask].values, r_t[mask].values
                if len(sv) < 3:
                    continue
                if method == "pearson":
                    r_corr, _ = stats.pearsonr(sv, rv)
                elif method == "spearman":
                    r_corr, _ = stats.spearmanr(sv, rv)
                else:
                    r_corr, _ = stats.kendalltau(sv, rv)
                cross_ics.append(float(r_corr))

            if cross_ics:
                horizons.append(h)
                ic_vals.append(float(np.nanmean(cross_ics)))
                ic_errs.append(float(np.nanstd(cross_ics) / np.sqrt(len(cross_ics))))
            else:
                horizons.append(h)
                ic_vals.append(float("nan"))
                ic_errs.append(float("nan"))

        # Fit exponential decay:  IC(h) = IC0 * exp(-lambda * h)
        decay_rate, half_life, ic0, r2 = self._fit_exponential_decay(horizons, ic_vals)

        return ICDecayResult(
            horizons=horizons,
            ic_values=ic_vals,
            ic_stderr=ic_errs,
            half_life=half_life,
            decay_rate=decay_rate,
            ic_at_zero=ic0,
            r_squared=r2,
            n_obs=len(idx),
        )

    def _fit_exponential_decay(
        self,
        horizons: list[int],
        ic_vals: list[float],
    ) -> Tuple[float, float, float, float]:
        """Fit IC(h) = IC0 * exp(-λ * h).

        Returns (λ, half_life, IC0, R²).
        """
        h_arr = np.array(horizons, dtype=float)
        ic_arr = np.array(ic_vals, dtype=float)
        mask = ~np.isnan(ic_arr)
        if mask.sum() < 3:
            return float("nan"), float("nan"), float("nan"), float("nan")

        h_clean = h_arr[mask]
        ic_clean = ic_arr[mask]

        def _exp_model(h: np.ndarray, ic0: float, lam: float) -> np.ndarray:
            return ic0 * np.exp(-lam * h)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(
                    _exp_model,
                    h_clean,
                    ic_clean,
                    p0=[ic_clean[0], 0.1],
                    maxfev=5000,
                )
            ic0_fit, lam_fit = popt
            ic_pred = _exp_model(h_clean, ic0_fit, lam_fit)
            ss_res = np.sum((ic_clean - ic_pred) ** 2)
            ss_tot = np.sum((ic_clean - ic_clean.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            half_life = np.log(2) / lam_fit if lam_fit > 0 else float("inf")
            return float(lam_fit), float(half_life), float(ic0_fit), float(r2)
        except Exception:
            return float("nan"), float("nan"), float("nan"), float("nan")

    # ------------------------------------------------------------------ #
    # ICIR
    # ------------------------------------------------------------------ #

    def icir(self, ic_series: pd.Series) -> float:
        """IC Information Ratio = mean(IC) / std(IC).

        Annualised by √252 is optional (not done here — caller decides).
        """
        clean = ic_series.dropna()
        if len(clean) < 2:
            return float("nan")
        std = clean.std(ddof=1)
        if std == 0:
            return float("nan")
        return float(clean.mean() / std)

    # ------------------------------------------------------------------ #
    # IC by regime
    # ------------------------------------------------------------------ #

    def ic_by_regime(
        self,
        signal_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        regime_series: pd.Series,
        method: Literal["pearson", "spearman", "kendall"] = "spearman",
    ) -> Dict[str, float]:
        """Compute per-regime average IC.

        Parameters
        ----------
        signal_df     : time × assets signals
        returns_df    : time × assets 1-bar forward returns
        regime_series : pd.Series[time] — regime label at each bar
        method        : correlation method

        Returns
        -------
        Dict mapping regime label → mean IC
        """
        cols = signal_df.columns.intersection(returns_df.columns)
        idx = signal_df.index.intersection(returns_df.index).intersection(regime_series.index)
        sig = signal_df.loc[idx, cols]
        ret = returns_df.loc[idx, cols]
        reg = regime_series.loc[idx]

        results: dict[str, list[float]] = {}
        for label in reg.unique():
            mask_t = reg == label
            regime_ics: list[float] = []
            for t in sig.index[mask_t]:
                s_t = sig.loc[t]
                r_t = ret.loc[t]
                both = ~(s_t.isna() | r_t.isna())
                sv, rv = s_t[both].values, r_t[both].values
                if len(sv) < 3:
                    continue
                if method == "pearson":
                    r_corr, _ = stats.pearsonr(sv, rv)
                elif method == "spearman":
                    r_corr, _ = stats.spearmanr(sv, rv)
                else:
                    r_corr, _ = stats.kendalltau(sv, rv)
                regime_ics.append(float(r_corr))
            results[str(label)] = regime_ics

        return {k: float(np.nanmean(v)) if v else float("nan") for k, v in results.items()}

    # ------------------------------------------------------------------ #
    # IC by quantile
    # ------------------------------------------------------------------ #

    def ic_by_quantile(
        self,
        signal: pd.Series,
        forward_returns: pd.Series,
        n_quantiles: int = 5,
    ) -> ICByQuantileResult:
        """Compute IC within each signal-strength quantile bucket.

        Splits *signal* into *n_quantiles* equal-frequency buckets and
        computes the IC within each bucket (measuring whether rank
        ordering still holds within a narrower signal range).

        Parameters
        ----------
        signal          : cross-sectional signal values
        forward_returns : corresponding forward returns
        n_quantiles     : number of equal-frequency buckets

        Returns
        -------
        ICByQuantileResult
        """
        df = pd.concat({"sig": signal, "ret": forward_returns}, axis=1).dropna()
        if len(df) < n_quantiles * 3:
            empty = [""] * n_quantiles
            return ICByQuantileResult(empty, [float("nan")] * n_quantiles, float("nan"), [0] * n_quantiles)

        df["q"] = pd.qcut(df["sig"], q=n_quantiles, labels=False, duplicates="drop")
        labels: list[str] = []
        ic_vals: list[float] = []
        n_obs: list[int] = []

        for q in sorted(df["q"].dropna().unique()):
            bucket = df[df["q"] == q]
            if len(bucket) < 3:
                ic_vals.append(float("nan"))
            else:
                r, _ = stats.spearmanr(bucket["sig"], bucket["ret"])
                ic_vals.append(float(r))
            labels.append(f"Q{int(q)+1}")
            n_obs.append(len(bucket))

        spread = (ic_vals[-1] - ic_vals[0]) if len(ic_vals) >= 2 else float("nan")
        return ICByQuantileResult(labels, ic_vals, spread, n_obs)

    # ------------------------------------------------------------------ #
    # IC stability
    # ------------------------------------------------------------------ #

    def ic_stability_score(self, ic_series: pd.Series) -> float:
        """Fraction of periods where IC > 0 (directional hit-rate of the signal).

        Returns
        -------
        float in [0, 1]
        """
        clean = ic_series.dropna()
        if len(clean) == 0:
            return float("nan")
        return float((clean > 0).mean())

    # ------------------------------------------------------------------ #
    # t-test
    # ------------------------------------------------------------------ #

    def ic_ttest(self, ic_series: pd.Series) -> Tuple[float, float]:
        """One-sample t-test: is mean IC significantly different from 0?

        Returns
        -------
        (t_stat, p_value) — two-tailed
        """
        clean = ic_series.dropna()
        if len(clean) < 2:
            return float("nan"), float("nan")
        t_stat, p_val = stats.ttest_1samp(clean, popmean=0.0)
        return float(t_stat), float(p_val)

    # ------------------------------------------------------------------ #
    # IC autocorrelation
    # ------------------------------------------------------------------ #

    def ic_autocorrelation(
        self,
        ic_series: pd.Series,
        max_lag: int = 20,
    ) -> pd.Series:
        """Autocorrelation Function (ACF) of the IC time-series.

        Returns
        -------
        pd.Series[lag → ACF value] for lags 1 .. max_lag
        """
        clean = ic_series.dropna().values
        n = len(clean)
        if n < max_lag + 2:
            return pd.Series({lag: float("nan") for lag in range(1, max_lag + 1)}, name="IC_ACF")

        mean_ic = clean.mean()
        var_ic = np.var(clean, ddof=1)
        acf_vals: dict[int, float] = {}
        for lag in range(1, max_lag + 1):
            cov = np.mean((clean[: n - lag] - mean_ic) * (clean[lag:] - mean_ic))
            acf_vals[lag] = cov / var_ic if var_ic > 0 else float("nan")
        return pd.Series(acf_vals, name="IC_ACF")

    # ------------------------------------------------------------------ #
    # Visualisations
    # ------------------------------------------------------------------ #

    def plot_ic_decay(
        self,
        decay_result: ICDecayResult,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """Bar-and-error plot of IC vs horizon with exponential fit overlay.

        Parameters
        ----------
        decay_result : ICDecayResult from ic_decay()
        save_path    : if given, save figure to this path

        Returns
        -------
        matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        horizons = np.array(decay_result.horizons)
        ic_vals = np.array(decay_result.ic_values)
        ic_errs = np.array(decay_result.ic_stderr)

        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in ic_vals]
        ax.bar(horizons, ic_vals, color=colors, alpha=0.7, label="IC per horizon")
        ax.errorbar(
            horizons, ic_vals, yerr=ic_errs,
            fmt="none", color="black", capsize=3, linewidth=0.8,
        )

        # Overlay fitted decay curve
        if not np.isnan(decay_result.decay_rate) and not np.isnan(decay_result.ic_at_zero):
            h_fine = np.linspace(1, horizons[-1], 200)
            ic_fit = decay_result.ic_at_zero * np.exp(-decay_result.decay_rate * h_fine)
            ax.plot(h_fine, ic_fit, "k--", linewidth=1.5, label=f"Exp fit (λ={decay_result.decay_rate:.3f})")

        ax.axhline(0, color="black", linewidth=0.8, linestyle="-")
        ax.set_xlabel("Forward-return horizon (bars)")
        ax.set_ylabel("Information Coefficient")
        ax.set_title(
            f"IC Decay  |  Peak IC={decay_result.peak_ic:.4f} @ h={decay_result.peak_horizon}"
            f"  |  Half-life={decay_result.half_life:.1f} bars  |  R²={decay_result.r_squared:.3f}"
        )
        ax.legend()
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_rolling_ic(
        self,
        ic_series: pd.Series,
        save_path: Optional[str | Path] = None,
        title: str = "Rolling IC",
    ) -> plt.Figure:
        """Time-series plot of rolling IC with ±1σ shading and ICIR annotation.

        Parameters
        ----------
        ic_series : pd.Series of rolling IC values
        save_path : optional save path
        title     : figure title
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        clean = ic_series.dropna()

        ax.plot(clean.index, clean.values, color="#3498db", linewidth=1.0, label="Rolling IC")
        ax.axhline(0, color="black", linewidth=0.8)

        mu = clean.mean()
        sigma = clean.std(ddof=1)
        ax.axhline(mu, color="#e67e22", linewidth=1.2, linestyle="--", label=f"Mean IC={mu:.4f}")
        ax.fill_between(clean.index, mu - sigma, mu + sigma, alpha=0.15, color="#e67e22", label="±1σ")

        icir_val = self.icir(clean)
        t_stat, p_val = self.ic_ttest(clean)
        ax.set_title(
            f"{title}  |  ICIR={icir_val:.3f}  |  t={t_stat:.2f}  |  p={p_val:.4f}"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("IC")
        ax.legend()
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_ic_distribution(
        self,
        ic_series: pd.Series,
        save_path: Optional[str | Path] = None,
        title: str = "IC Distribution",
    ) -> plt.Figure:
        """Histogram of IC values with normal distribution overlay.

        Parameters
        ----------
        ic_series : pd.Series of IC values
        save_path : optional save path
        title     : figure title
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        clean = ic_series.dropna().values

        ax.hist(clean, bins=40, density=True, color="#3498db", alpha=0.65, edgecolor="white", label="IC histogram")

        mu, sigma = clean.mean(), clean.std(ddof=1)
        x = np.linspace(clean.min(), clean.max(), 300)
        normal_pdf = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, normal_pdf, "r-", linewidth=2, label=f"N({mu:.4f}, {sigma:.4f}²)")

        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.axvline(mu, color="#e67e22", linewidth=1.5, linestyle="--", label=f"Mean={mu:.4f}")

        icir_val = self.icir(ic_series)
        pct_pos = self.ic_stability_score(ic_series)
        t_stat, p_val = self.ic_ttest(ic_series)

        textstr = "\n".join([
            f"ICIR  = {icir_val:.3f}",
            f"% pos = {pct_pos:.1%}",
            f"t     = {t_stat:.2f}",
            f"p     = {p_val:.4f}",
        ])
        ax.text(
            0.97, 0.97, textstr,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            horizontalalignment="right",
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "wheat", "alpha": 0.8},
        )

        ax.set_xlabel("IC Value")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend()
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    # ------------------------------------------------------------------ #
    # Batch panel IC (time-series of cross-sectional ICs)
    # ------------------------------------------------------------------ #

    def panel_ic_series(
        self,
        signal_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        method: Literal["pearson", "spearman", "kendall"] = "spearman",
    ) -> pd.Series:
        """Compute cross-sectional IC for every row in the panel.

        Each row of signal_df / returns_df is one time period; the IC is
        computed across all assets (columns) for that period.

        Parameters
        ----------
        signal_df   : DataFrame[time × assets]
        returns_df  : DataFrame[time × assets]
        method      : correlation method

        Returns
        -------
        pd.Series[time → IC]
        """
        cols = signal_df.columns.intersection(returns_df.columns)
        idx = signal_df.index.intersection(returns_df.index)
        sig = signal_df.loc[idx, cols]
        ret = returns_df.loc[idx, cols]

        ic_list: list[float] = []
        for t in idx:
            s_t = sig.loc[t].dropna()
            r_t = ret.loc[t].dropna()
            common = s_t.index.intersection(r_t.index)
            sv, rv = s_t.loc[common].values, r_t.loc[common].values
            if len(sv) < 3:
                ic_list.append(float("nan"))
                continue
            if method == "pearson":
                r_corr, _ = stats.pearsonr(sv, rv)
            elif method == "spearman":
                r_corr, _ = stats.spearmanr(sv, rv)
            else:
                r_corr, _ = stats.kendalltau(sv, rv)
            ic_list.append(float(r_corr))

        return pd.Series(ic_list, index=idx, name="IC")

    # ------------------------------------------------------------------ #
    # Comprehensive IC summary
    # ------------------------------------------------------------------ #

    def ic_summary(
        self,
        signal_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        method: Literal["pearson", "spearman", "kendall"] = "spearman",
    ) -> Dict[str, float]:
        """Produce a comprehensive IC summary dict.

        Keys: mean_ic, std_ic, icir, pct_positive, t_stat, p_value,
              skew, kurtosis, min_ic, max_ic, median_ic
        """
        ic_series = self.panel_ic_series(signal_df, returns_df, method=method)
        clean = ic_series.dropna()

        t_stat, p_val = self.ic_ttest(ic_series)
        return {
            "mean_ic": float(clean.mean()),
            "std_ic": float(clean.std(ddof=1)),
            "icir": self.icir(ic_series),
            "pct_positive": self.ic_stability_score(ic_series),
            "t_stat": t_stat,
            "p_value": p_val,
            "skew": float(stats.skew(clean)),
            "kurtosis": float(stats.kurtosis(clean)),
            "min_ic": float(clean.min()),
            "max_ic": float(clean.max()),
            "median_ic": float(clean.median()),
            "n_periods": len(clean),
        }

    # ------------------------------------------------------------------ #
    # Newey-West adjusted IC t-stat
    # ------------------------------------------------------------------ #

    def ic_ttest_newey_west(
        self,
        ic_series: pd.Series,
        n_lags: int = 5,
    ) -> Tuple[float, float]:
        """t-test with Newey-West HAC standard error (for autocorrelated IC).

        Parameters
        ----------
        ic_series : IC time-series
        n_lags    : Newey-West lag truncation (default 5 bars)

        Returns
        -------
        (t_stat, p_value) — two-tailed
        """
        import statsmodels.api as sm

        clean = ic_series.dropna().values
        n = len(clean)
        if n < n_lags + 2:
            return self.ic_ttest(ic_series)

        # OLS of IC on constant, then NW standard error
        X = np.ones((n, 1))
        model = sm.OLS(clean, X).fit()
        nw_result = model.get_robustcov_results(cov_type="HAC", maxlags=n_lags)
        t_stat = float(nw_result.tvalues[0])
        p_val = float(nw_result.pvalues[0])
        return t_stat, p_val

    # ------------------------------------------------------------------ #
    # IC at specific horizons via flat trade DataFrame
    # ------------------------------------------------------------------ #

    @staticmethod
    def ic_from_trades(
        trades: pd.DataFrame,
        signal_col: str,
        return_col: str = "pnl",
        method: Literal["pearson", "spearman", "kendall"] = "spearman",
        dollar_pos_col: str = "dollar_pos",
        normalise_by_pos: bool = True,
    ) -> float:
        """Compute IC from a flat trades DataFrame.

        Parameters
        ----------
        trades          : DataFrame with columns including signal and return
        signal_col      : name of the signal column
        return_col      : name of the return/pnl column
        method          : correlation method
        dollar_pos_col  : column for position size (used if normalise_by_pos)
        normalise_by_pos: if True, convert pnl → return = pnl / |dollar_pos|

        Returns
        -------
        float IC
        """
        df = trades[[signal_col, return_col]].copy()
        if normalise_by_pos and dollar_pos_col in trades.columns:
            pos = trades[dollar_pos_col].abs()
            pos = pos.replace(0, np.nan)
            df[return_col] = trades[return_col] / pos
        df = df.dropna()
        if len(df) < 3:
            return float("nan")

        sig = df[signal_col].values
        ret = df[return_col].values
        if method == "pearson":
            r, _ = stats.pearsonr(sig, ret)
        elif method == "spearman":
            r, _ = stats.spearmanr(sig, ret)
        else:
            r, _ = stats.kendalltau(sig, ret)
        return float(r)

    # ------------------------------------------------------------------ #
    # Forward-IC grid (decay) from flat trades
    # ------------------------------------------------------------------ #

    def ic_decay_from_trades(
        self,
        trades: pd.DataFrame,
        signal_col: str,
        return_col: str = "pnl",
        hold_col: str = "hold_bars",
        max_horizon: int = 20,
        dollar_pos_col: str = "dollar_pos",
        normalise_by_pos: bool = True,
        method: Literal["pearson", "spearman", "kendall"] = "spearman",
    ) -> ICDecayResult:
        """Approximate IC decay from a trades table by slicing hold_bars.

        Trades held ≥ h bars contribute to the h-bar IC estimate.

        Parameters
        ----------
        trades          : flat trade records
        signal_col      : signal column name
        return_col      : pnl / return column
        hold_col        : column recording hold duration in bars
        max_horizon     : furthest horizon
        dollar_pos_col  : position-size column for return normalisation
        normalise_by_pos: normalise pnl → return
        method          : correlation method

        Returns
        -------
        ICDecayResult
        """
        df = trades.copy()
        if normalise_by_pos and dollar_pos_col in df.columns:
            pos = df[dollar_pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[return_col] / pos
        else:
            df["_ret"] = df[return_col]

        horizons, ic_vals, ic_errs = [], [], []
        for h in range(1, max_horizon + 1):
            subset = df[df[hold_col] >= h][[signal_col, "_ret"]].dropna()
            if len(subset) < 5:
                horizons.append(h)
                ic_vals.append(float("nan"))
                ic_errs.append(float("nan"))
                continue
            sig = subset[signal_col].values
            ret = subset["_ret"].values
            if method == "pearson":
                r, _ = stats.pearsonr(sig, ret)
            elif method == "spearman":
                r, _ = stats.spearmanr(sig, ret)
            else:
                r, _ = stats.kendalltau(sig, ret)
            horizons.append(h)
            ic_vals.append(float(r))
            ic_errs.append(float(np.std(ret) / np.sqrt(len(ret))))

        lam, hl, ic0, r2 = self._fit_exponential_decay(horizons, ic_vals)
        return ICDecayResult(
            horizons=horizons,
            ic_values=ic_vals,
            ic_stderr=ic_errs,
            half_life=hl,
            decay_rate=lam,
            ic_at_zero=ic0,
            r_squared=r2,
            n_obs=len(df),
        )

    # ------------------------------------------------------------------ #
    # Forward-return IC convenience wrapper
    # ------------------------------------------------------------------ #

    def forward_return_ic_grid(
        self,
        signal_series: pd.Series,
        price_series: pd.Series,
        horizons: List[int] | None = None,
        method: Literal["pearson", "spearman", "kendall"] = "spearman",
    ) -> Dict[int, float]:
        """Compute IC between signal_series and h-bar forward returns of price_series.

        Parameters
        ----------
        signal_series : pd.Series indexed by time
        price_series  : pd.Series of prices indexed by time
        horizons      : list of horizons (default [1,2,3,5,10,15,20])
        method        : correlation method

        Returns
        -------
        Dict[horizon → IC]
        """
        if horizons is None:
            horizons = [1, 2, 3, 5, 10, 15, 20]
        log_ret = np.log(price_series / price_series.shift(1))
        result: dict[int, float] = {}
        for h in horizons:
            fwd = log_ret.rolling(h).sum().shift(-h)
            ic = self.compute_ic(signal_series, fwd, method=method)
            result[h] = ic
        return result

    # ------------------------------------------------------------------ #
    # IC by symbol (cross-instrument)
    # ------------------------------------------------------------------ #

    def ic_by_symbol(
        self,
        trades: pd.DataFrame,
        signal_col: str,
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
        min_obs: int = 10,
    ) -> pd.Series:
        """Compute IC separately for each instrument (sym) in the trades table.

        Parameters
        ----------
        trades        : flat trade records with 'sym' column
        signal_col    : signal column
        return_col    : P&L column
        dollar_pos_col: position size column (for normalisation)
        min_obs       : minimum trades per symbol to compute IC

        Returns
        -------
        pd.Series indexed by sym with IC values
        """
        if "sym" not in trades.columns:
            raise ValueError("trades must have 'sym' column")

        df = trades.copy()
        if dollar_pos_col in df.columns:
            pos = df[dollar_pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[return_col] / pos
        else:
            df["_ret"] = df[return_col]

        result: dict[str, float] = {}
        for sym in df["sym"].unique():
            sub = df[df["sym"] == sym][[signal_col, "_ret"]].dropna()
            if len(sub) < min_obs:
                result[sym] = float("nan")
                continue
            r, _ = stats.spearmanr(sub[signal_col], sub["_ret"])
            result[sym] = float(r)

        return pd.Series(result, name="IC_by_sym").sort_values(ascending=False)

    # ------------------------------------------------------------------ #
    # IC by time bucket
    # ------------------------------------------------------------------ #

    def ic_by_time_period(
        self,
        trades: pd.DataFrame,
        signal_col: str,
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
        freq: str = "M",
    ) -> pd.Series:
        """Compute monthly (or other frequency) IC from flat trades.

        Parameters
        ----------
        trades        : flat trade records
        signal_col    : signal column
        return_col    : P&L column
        dollar_pos_col: position column for normalisation
        freq          : pandas frequency string ('M', 'Q', 'W', etc.)

        Returns
        -------
        pd.Series[period -> IC]
        """
        if "exit_time" not in trades.columns:
            raise ValueError("trades must have 'exit_time' column for time-period IC")

        df = trades.copy()
        df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
        if dollar_pos_col in df.columns:
            pos = df[dollar_pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[return_col] / pos
        else:
            df["_ret"] = df[return_col]

        df = df.set_index("exit_time").sort_index()
        sub = df[[signal_col, "_ret"]].dropna()
        grouped = sub.resample(freq)

        ic_vals: dict = {}
        for period, group in grouped:
            if len(group) < 3:
                ic_vals[period] = float("nan")
                continue
            r, _ = stats.spearmanr(group[signal_col], group["_ret"])
            ic_vals[period] = float(r)

        return pd.Series(ic_vals, name="IC").dropna()

    # ------------------------------------------------------------------ #
    # Conditional IC (above/below median signal)
    # ------------------------------------------------------------------ #

    def conditional_ic(
        self,
        signal: pd.Series,
        forward_returns: pd.Series,
        condition: pd.Series,
    ) -> Dict[str, float]:
        """IC conditioned on an external boolean series.

        Parameters
        ----------
        signal          : signal values
        forward_returns : forward returns
        condition       : boolean Series — True/False condition mask

        Returns
        -------
        Dict with keys 'ic_true', 'ic_false', 'ic_all'
        """
        df = pd.concat({"sig": signal, "ret": forward_returns, "cond": condition}, axis=1).dropna()
        ic_all = self.compute_ic(df["sig"], df["ret"])

        true_df = df[df["cond"] == True]
        false_df = df[df["cond"] == False]

        ic_true = (
            self.compute_ic(true_df["sig"], true_df["ret"])
            if len(true_df) >= 3
            else float("nan")
        )
        ic_false = (
            self.compute_ic(false_df["sig"], false_df["ret"])
            if len(false_df) >= 3
            else float("nan")
        )
        return {"ic_true": ic_true, "ic_false": ic_false, "ic_all": ic_all}

    # ------------------------------------------------------------------ #
    # IC Sharpe (annualised ICIR)
    # ------------------------------------------------------------------ #

    def ic_sharpe(
        self,
        ic_series: pd.Series,
        bars_per_year: int = 252,
    ) -> float:
        """Annualised IC Sharpe ratio = ICIR * sqrt(bars_per_year / n_per_year_in_series).

        For daily IC series: ic_sharpe = ICIR * sqrt(252).
        For monthly IC series: ic_sharpe = ICIR * sqrt(12).

        Parameters
        ----------
        ic_series    : time-series of IC values
        bars_per_year: number of IC observations per year

        Returns
        -------
        float annualised IC Sharpe
        """
        raw_icir = self.icir(ic_series)
        if np.isnan(raw_icir):
            return float("nan")
        return float(raw_icir * np.sqrt(bars_per_year))

    # ------------------------------------------------------------------ #
    # Bootstrap IC confidence interval
    # ------------------------------------------------------------------ #

    def ic_bootstrap_ci(
        self,
        signal: pd.Series,
        forward_returns: pd.Series,
        n_bootstrap: int = 1000,
        confidence: float = 0.95,
        method: str = "spearman",
        seed: int = 42,
    ) -> Tuple[float, float, float]:
        """Bootstrap confidence interval for IC.

        Parameters
        ----------
        signal          : signal values
        forward_returns : forward returns
        n_bootstrap     : number of bootstrap resamples
        confidence      : confidence level (e.g. 0.95 for 95% CI)
        method          : correlation method
        seed            : random seed

        Returns
        -------
        (ic_point, ci_lower, ci_upper)
        """
        rng = np.random.default_rng(seed)
        df = pd.concat({"sig": signal, "ret": forward_returns}, axis=1).dropna()
        n = len(df)
        if n < 10:
            ic_pt = self.compute_ic(signal, forward_returns, method=method)
            return ic_pt, float("nan"), float("nan")

        ic_pt = self.compute_ic(df["sig"], df["ret"], method=method)
        boot_ics: list[float] = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            boot = df.iloc[idx]
            r, _ = (stats.spearmanr if method == "spearman" else stats.pearsonr)(
                boot["sig"], boot["ret"]
            )
            boot_ics.append(float(r))

        alpha = (1 - confidence) / 2
        ci_lo = float(np.quantile(boot_ics, alpha))
        ci_hi = float(np.quantile(boot_ics, 1 - alpha))
        return ic_pt, ci_lo, ci_hi

    # ------------------------------------------------------------------ #
    # IC vector for multiple signals at once
    # ------------------------------------------------------------------ #

    def multi_signal_ic(
        self,
        trades: pd.DataFrame,
        signal_cols: List[str],
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
        method: str = "spearman",
    ) -> pd.Series:
        """Compute IC for multiple signal columns simultaneously.

        Parameters
        ----------
        trades      : trade DataFrame
        signal_cols : list of signal column names to evaluate
        return_col  : P&L column
        dollar_pos_col: position column for normalisation
        method      : correlation method

        Returns
        -------
        pd.Series indexed by signal_col with IC values
        """
        df = trades.copy()
        if dollar_pos_col in df.columns:
            pos = df[dollar_pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[return_col] / pos
        else:
            df["_ret"] = df[return_col]

        ics: dict[str, float] = {}
        for col in signal_cols:
            if col not in df.columns:
                ics[col] = float("nan")
                continue
            sub = df[[col, "_ret"]].dropna()
            if len(sub) < 3:
                ics[col] = float("nan")
                continue
            ics[col] = self.compute_ic(sub[col], sub["_ret"], method=method)
        return pd.Series(ics, name="IC")

    # ------------------------------------------------------------------ #
    # IC correlation matrix (signal-to-signal)
    # ------------------------------------------------------------------ #

    def signal_correlation_matrix(
        self,
        trades: pd.DataFrame,
        signal_cols: List[str],
        method: str = "spearman",
    ) -> pd.DataFrame:
        """Pearson or Spearman correlation matrix of signal columns.

        Useful for detecting redundant signals.

        Parameters
        ----------
        trades      : trade DataFrame
        signal_cols : list of signal column names
        method      : 'spearman' or 'pearson'

        Returns
        -------
        pd.DataFrame — symmetric correlation matrix (n_signals x n_signals)
        """
        available = [c for c in signal_cols if c in trades.columns]
        sub = trades[available].dropna()
        if method == "spearman":
            return sub.rank().corr()
        return sub.corr()

    # ------------------------------------------------------------------ #
    # Time-stratified IC (training / test split)
    # ------------------------------------------------------------------ #

    def train_test_ic(
        self,
        trades: pd.DataFrame,
        signal_col: str,
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
        train_frac: float = 0.7,
    ) -> Dict[str, float]:
        """Compute IC on training and test subsets.

        Parameters
        ----------
        trades       : trade records (sorted by time if exit_time present)
        signal_col   : signal column
        return_col   : P&L column
        dollar_pos_col: position column
        train_frac   : fraction of data for training (chronological split)

        Returns
        -------
        Dict with keys: train_ic, test_ic, train_n, test_n
        """
        df = trades.copy()
        if "exit_time" in df.columns:
            df = df.sort_values("exit_time")

        if dollar_pos_col in df.columns:
            pos = df[dollar_pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[return_col] / pos
        else:
            df["_ret"] = df[return_col]

        n = len(df)
        split = int(n * train_frac)
        train = df.iloc[:split]
        test = df.iloc[split:]

        def _ic(subset: pd.DataFrame) -> float:
            sub = subset[[signal_col, "_ret"]].dropna()
            if len(sub) < 3:
                return float("nan")
            r, _ = stats.spearmanr(sub[signal_col], sub["_ret"])
            return float(r)

        return {
            "train_ic": _ic(train),
            "test_ic": _ic(test),
            "train_n": len(train),
            "test_n": len(test),
            "ic_degradation": (
                _ic(train) - _ic(test)
                if not np.isnan(_ic(train)) and not np.isnan(_ic(test))
                else float("nan")
            ),
        }

    # ------------------------------------------------------------------ #
    # IC heatmap by year/month
    # ------------------------------------------------------------------ #

    def ic_heatmap_data(
        self,
        trades: pd.DataFrame,
        signal_col: str,
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
    ) -> pd.DataFrame:
        """Produce a year x month matrix of IC values for heatmap display.

        Parameters
        ----------
        trades        : trade records with exit_time column
        signal_col    : signal column
        return_col    : P&L column
        dollar_pos_col: position column

        Returns
        -------
        pd.DataFrame[year x month] of IC values
        """
        if "exit_time" not in trades.columns:
            raise ValueError("trades must have 'exit_time' column")

        df = trades.copy()
        df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
        if dollar_pos_col in df.columns:
            pos = df[dollar_pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[return_col] / pos
        else:
            df["_ret"] = df[return_col]

        df["year"] = df["exit_time"].dt.year
        df["month"] = df["exit_time"].dt.month

        records: list[dict] = []
        for (year, month), group in df.groupby(["year", "month"]):
            sub = group[[signal_col, "_ret"]].dropna()
            if len(sub) < 3:
                ic = float("nan")
            else:
                r, _ = stats.spearmanr(sub[signal_col], sub["_ret"])
                ic = float(r)
            records.append({"year": year, "month": month, "ic": ic})

        result = pd.DataFrame(records)
        if result.empty:
            return pd.DataFrame()
        return result.pivot(index="year", columns="month", values="ic")

    def plot_ic_heatmap(
        self,
        trades: pd.DataFrame,
        signal_col: str,
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """Monthly IC heatmap (years x months).

        Parameters
        ----------
        trades        : trade records
        signal_col    : signal column
        save_path     : optional save path
        """
        import seaborn as sns

        ic_matrix = self.ic_heatmap_data(trades, signal_col, return_col, dollar_pos_col)
        if ic_matrix.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            return fig

        month_labels = {
            1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
            7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
        }
        ic_matrix.columns = [month_labels.get(c, str(c)) for c in ic_matrix.columns]

        fig, ax = plt.subplots(figsize=(14, max(4, len(ic_matrix) * 0.6)))
        sns.heatmap(
            ic_matrix, ax=ax, center=0, cmap="RdYlGn",
            annot=True, fmt=".3f", linewidths=0.5,
            cbar_kws={"label": "Spearman IC"},
        )
        ax.set_title(f"IC Heatmap: {signal_col}")
        ax.set_xlabel("Month")
        ax.set_ylabel("Year")
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    # ------------------------------------------------------------------ #
    # Comprehensive IC report from trades
    # ------------------------------------------------------------------ #

    def ic_report_from_trades(
        self,
        trades: pd.DataFrame,
        signal_cols: Optional[List[str]] = None,
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
        method: str = "spearman",
    ) -> pd.DataFrame:
        """Generate a comprehensive IC report DataFrame from a trades table.

        Computes IC, t-stat, p-value, and bootstrap CI for each signal column.

        Parameters
        ----------
        trades      : trade records
        signal_cols : list of signal columns (auto-detected if None)
        return_col  : P&L column
        dollar_pos_col: position column
        method      : correlation method

        Returns
        -------
        pd.DataFrame[signal_col x (ic, t_stat, p_value, ci_lower, ci_upper,
                                    n_obs, significant_5pct)]
        """
        if signal_cols is None:
            # Auto-detect numeric columns that look like signals
            exclude = {return_col, dollar_pos_col, "hold_bars", "entry_price",
                       "exit_price", "exit_time", "sym", "regime"}
            signal_cols = [c for c in trades.select_dtypes(include=[np.number]).columns
                           if c not in exclude]

        df = trades.copy()
        if dollar_pos_col in df.columns:
            pos = df[dollar_pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[return_col] / pos
        else:
            df["_ret"] = df[return_col]

        records: list[dict] = []
        for col in signal_cols:
            if col not in df.columns:
                continue
            sub = df[[col, "_ret"]].dropna()
            n = len(sub)
            if n < 3:
                records.append({
                    "signal": col, "ic": float("nan"), "t_stat": float("nan"),
                    "p_value": float("nan"), "ci_lower": float("nan"),
                    "ci_upper": float("nan"), "n_obs": n, "significant_5pct": False,
                })
                continue

            r, p = stats.spearmanr(sub[col], sub["_ret"])
            ic = float(r)
            t = ic * np.sqrt(n - 2) / max(np.sqrt(max(1 - ic**2, 1e-10)), 1e-10)

            # Bootstrap CI (fast: 200 resamples)
            _, ci_lo, ci_hi = self.ic_bootstrap_ci(
                sub[col], sub["_ret"], n_bootstrap=200, method=method
            )

            records.append({
                "signal": col,
                "ic": ic,
                "t_stat": float(t),
                "p_value": float(p),
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "n_obs": n,
                "significant_5pct": abs(t) > 1.96,
            })

        result = pd.DataFrame(records).set_index("signal")
        return result.sort_values("ic", key=lambda s: s.abs(), ascending=False)

    def ic_pairwise_comparison(
        self,
        signal_a: "pd.Series",
        signal_b: "pd.Series",
        forward_returns: "pd.Series",
        method: str = "spearman",
        n_bootstrap: int = 500,
        seed: int = 42,
    ) -> dict:
        """
        Compare two signals using Diebold-Mariano style bootstrap test of
        equal IC. Returns IC for each signal, the difference, and a
        bootstrap p-value for H0: IC_a == IC_b.

        Parameters
        ----------
        signal_a, signal_b : pd.Series
            Two signal time series.
        forward_returns : pd.Series
            Forward return series.
        method : str
            Correlation method: 'spearman', 'pearson', or 'kendall'.
        n_bootstrap : int
            Bootstrap resamples.
        seed : int
            RNG seed.

        Returns
        -------
        dict with keys: ic_a, ic_b, ic_diff, p_value, significant.
        """
        from scipy import stats as _stats

        df = pd.DataFrame({
            "a": signal_a, "b": signal_b, "ret": forward_returns
        }).dropna()

        if len(df) < 10:
            raise ValueError("Insufficient data for pairwise IC comparison.")

        a = df["a"].values.astype(float)
        b = df["b"].values.astype(float)
        r = df["ret"].values.astype(float)

        if method == "spearman":
            ic_a, _ = _stats.spearmanr(a, r)
            ic_b, _ = _stats.spearmanr(b, r)
        elif method == "kendall":
            ic_a, _ = _stats.kendalltau(a, r)
            ic_b, _ = _stats.kendalltau(b, r)
        else:
            ic_a, _ = _stats.pearsonr(a, r)
            ic_b, _ = _stats.pearsonr(b, r)

        observed_diff = float(ic_a) - float(ic_b)
        rng = np.random.default_rng(seed)
        boot_diffs: list[float] = []

        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(df), size=len(df))
            a_b = a[idx]
            b_b = b[idx]
            r_b = r[idx]
            if method == "spearman":
                ba, _ = _stats.spearmanr(a_b, r_b)
                bb, _ = _stats.spearmanr(b_b, r_b)
            elif method == "kendall":
                ba, _ = _stats.kendalltau(a_b, r_b)
                bb, _ = _stats.kendalltau(b_b, r_b)
            else:
                ba, _ = _stats.pearsonr(a_b, r_b)
                bb, _ = _stats.pearsonr(b_b, r_b)
            if np.isfinite(ba) and np.isfinite(bb):
                boot_diffs.append(float(ba) - float(bb))

        # Two-sided bootstrap p-value
        centered = np.array(boot_diffs) - np.mean(boot_diffs)
        p_val = float(np.mean(np.abs(centered) >= abs(observed_diff)))

        return {
            "ic_a": float(ic_a),
            "ic_b": float(ic_b),
            "ic_diff": observed_diff,
            "p_value": p_val,
            "significant": p_val < 0.05,
            "n_obs": len(df),
            "n_bootstrap": n_bootstrap,
        }
