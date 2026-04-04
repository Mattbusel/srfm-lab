"""
research/signal_analytics/factor_model.py
==========================================
Barra-style factor attribution model.

Supports:
  - Cross-sectional regression (OLS / WLS) of returns on factor exposures
  - Fama-MacBeth time-series of cross-sectional regressions with Newey-West
    corrected standard errors
  - Factor-level IC diagnostics
  - Systematic vs idiosyncratic return attribution
  - PCA factor extraction from return panels
  - BH-specific factors: mass, tf_score

Factor set
----------
  Momentum   : 20d, 60d, 120d log-return
  Volatility : realized 20d volatility
  Size       : dollar position (log)
  BH-specific: mass (0–2), tf_score (0–7)

Usage example
-------------
>>> fm = FactorModel()
>>> F = fm.build_factor_matrix(trades, price_history)
>>> fmb = fm.fama_macbeth(returns_panel, factors_panel)
>>> fm.plot_attribution_waterfall(attr, save_path="results/attribution.png")
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CrossSectionalRegressionResult:
    """Output of a single cross-sectional regression."""
    factor_names: List[str]
    factor_returns: pd.Series          # one value per factor (γ)
    t_stats: pd.Series
    p_values: pd.Series
    r_squared: float
    adj_r_squared: float
    n_obs: int
    residuals: pd.Series


@dataclass
class FMBResult:
    """Fama-MacBeth regression results."""
    factor_names: List[str]
    mean_factor_returns: pd.Series     # time-average of cross-sectional γ
    fmb_t_stats: pd.Series            # Newey-West corrected t-stats
    fmb_p_values: pd.Series
    factor_return_series: pd.DataFrame  # T × K time-series of γ_t
    mean_r_squared: float
    n_periods: int
    n_lags: int                        # NW lags used


@dataclass
class AttributionResult:
    """Factor attribution for a set of trades."""
    total_return: float
    systematic_return: float
    idiosyncratic_return: float
    factor_contributions: pd.Series    # factor → $ contribution
    factor_names: List[str]
    r_squared: float
    trades_count: int


@dataclass
class PCAFactorResult:
    """PCA factor extraction result."""
    n_components: int
    explained_variance_ratio: np.ndarray
    cumulative_variance: np.ndarray
    factor_loadings: pd.DataFrame      # assets × components
    factor_returns: pd.DataFrame       # time × components
    eigenvalues: np.ndarray


# ---------------------------------------------------------------------------
# FactorModel
# ---------------------------------------------------------------------------

class FactorModel:
    """Barra-style cross-sectional factor model.

    Parameters
    ----------
    factor_names : optional list of custom factor names to use
    winsorise     : clip extreme returns to ±winsorise stddevs before regression
    """

    DEFAULT_FACTORS = [
        "mom_20d",
        "mom_60d",
        "mom_120d",
        "vol_20d",
        "log_size",
        "mass",
        "tf_score",
    ]

    def __init__(
        self,
        factor_names: Optional[List[str]] = None,
        winsorise: float = 5.0,
    ) -> None:
        self.factor_names = factor_names or self.DEFAULT_FACTORS
        self.winsorise = winsorise

    # ------------------------------------------------------------------ #
    # Factor matrix construction
    # ------------------------------------------------------------------ #

    def build_factor_matrix(
        self,
        trades: pd.DataFrame,
        price_history: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Build factor exposure matrix from trades and optional price history.

        Parameters
        ----------
        trades        : trade records with columns: sym, entry_price,
                        exit_price, dollar_pos, pnl, hold_bars,
                        mass (optional), tf_score (optional),
                        ensemble_signal (optional), ATR (optional)
        price_history : wide DataFrame[time × sym] of close prices
                        (used for momentum and vol factors)

        Returns
        -------
        pd.DataFrame — n_trades × n_factors, standardised
        """
        factors: dict[str, pd.Series] = {}
        n = len(trades)
        idx = trades.index

        # BH-specific factors (directly available in trades)
        if "mass" in trades.columns:
            factors["mass"] = trades["mass"].fillna(0.0)
        else:
            factors["mass"] = pd.Series(np.zeros(n), index=idx)

        if "tf_score" in trades.columns:
            factors["tf_score"] = trades["tf_score"].fillna(0.0)
        else:
            factors["tf_score"] = pd.Series(np.zeros(n), index=idx)

        if "ensemble_signal" in trades.columns:
            factors["ensemble_signal"] = trades["ensemble_signal"].fillna(0.0)

        # Size factor
        if "dollar_pos" in trades.columns:
            pos = trades["dollar_pos"].abs().replace(0, np.nan)
            factors["log_size"] = np.log(pos).fillna(0.0)

        # ATR-based vol factor
        if "ATR" in trades.columns:
            factors["atr"] = trades["ATR"].fillna(0.0)

        # Momentum and vol factors from price history
        if price_history is not None and "sym" in trades.columns:
            mom_20, mom_60, mom_120, vol_20 = self._compute_price_factors(
                trades, price_history
            )
            factors["mom_20d"] = mom_20
            factors["mom_60d"] = mom_60
            factors["mom_120d"] = mom_120
            factors["vol_20d"] = vol_20

        factor_df = pd.DataFrame(factors, index=idx)

        # Standardise each factor (z-score cross-sectionally)
        for col in factor_df.columns:
            mu = factor_df[col].mean()
            sigma = factor_df[col].std(ddof=1)
            if sigma > 0:
                factor_df[col] = (factor_df[col] - mu) / sigma

        return factor_df

    def _compute_price_factors(
        self,
        trades: pd.DataFrame,
        price_history: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Compute momentum and volatility factors at trade entry time."""
        log_prices = np.log(price_history)
        log_ret = log_prices.diff()

        mom_20_vals, mom_60_vals, mom_120_vals, vol_20_vals = [], [], [], []

        for _, row in trades.iterrows():
            sym = row.get("sym", None)
            entry = row.get("exit_time", None) or row.name

            if sym is None or sym not in log_ret.columns:
                mom_20_vals.append(0.0)
                mom_60_vals.append(0.0)
                mom_120_vals.append(0.0)
                vol_20_vals.append(0.0)
                continue

            try:
                loc = log_prices.index.get_loc(entry, method="ffill")
            except (KeyError, TypeError):
                loc = -1

            if loc < 120:
                mom_20_vals.append(0.0)
                mom_60_vals.append(0.0)
                mom_120_vals.append(0.0)
                vol_20_vals.append(0.0)
                continue

            price_col = log_prices[sym]
            ret_col = log_ret[sym]

            mom_20_vals.append(float(price_col.iloc[loc] - price_col.iloc[max(0, loc - 20)]))
            mom_60_vals.append(float(price_col.iloc[loc] - price_col.iloc[max(0, loc - 60)]))
            mom_120_vals.append(float(price_col.iloc[loc] - price_col.iloc[max(0, loc - 120)]))
            vol_20_vals.append(float(ret_col.iloc[max(0, loc - 20): loc + 1].std(ddof=1)))

        idx = trades.index
        return (
            pd.Series(mom_20_vals, index=idx, name="mom_20d"),
            pd.Series(mom_60_vals, index=idx, name="mom_60d"),
            pd.Series(mom_120_vals, index=idx, name="mom_120d"),
            pd.Series(vol_20_vals, index=idx, name="vol_20d"),
        )

    # ------------------------------------------------------------------ #
    # Cross-sectional regression
    # ------------------------------------------------------------------ #

    def cross_sectional_regression(
        self,
        returns: pd.Series,
        factors: pd.DataFrame,
        weights: Optional[pd.Series] = None,
    ) -> CrossSectionalRegressionResult:
        """OLS/WLS cross-sectional regression: r_i = Σ γ_k f_{i,k} + ε_i.

        Parameters
        ----------
        returns : realised returns for each asset/trade (n,)
        factors : factor exposure matrix (n × K)
        weights : optional regression weights (e.g. sqrt(market_cap))

        Returns
        -------
        CrossSectionalRegressionResult
        """
        df = pd.concat({"ret": returns}, axis=1).join(factors).dropna()
        if len(df) < factors.shape[1] + 2:
            raise ValueError(f"Insufficient observations: {len(df)} < {factors.shape[1] + 2}")

        y = df["ret"].values
        X = df[factors.columns].values
        factor_names = list(factors.columns)

        # Winsorise returns
        if self.winsorise > 0:
            mu, sig = y.mean(), y.std(ddof=1)
            y = np.clip(y, mu - self.winsorise * sig, mu + self.winsorise * sig)

        # Add intercept
        X_int = np.column_stack([np.ones(len(y)), X])
        k_names = ["intercept"] + factor_names

        if weights is not None:
            w = df.index.map(weights).fillna(1.0).values
            w_sqrt = np.sqrt(np.abs(w))
            y_w = y * w_sqrt
            X_w = X_int * w_sqrt[:, None]
        else:
            y_w, X_w = y, X_int

        # Least squares
        beta, residuals_ss, rank, _ = np.linalg.lstsq(X_w, y_w, rcond=None)

        y_hat = X_int @ beta
        resid = y - y_hat
        n, k = len(y), len(beta)

        ss_res = np.sum(resid**2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1) if n > k + 1 else float("nan")

        # Standard errors via (X'X)^{-1} * s²
        try:
            cov = np.linalg.inv(X_int.T @ X_int) * (ss_res / max(n - k, 1))
            se = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            se = np.full(len(beta), float("nan"))

        t_stats = beta / se
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n - k))

        # Exclude intercept from factor returns
        return CrossSectionalRegressionResult(
            factor_names=factor_names,
            factor_returns=pd.Series(beta[1:], index=factor_names, name="factor_returns"),
            t_stats=pd.Series(t_stats[1:], index=factor_names, name="t_stats"),
            p_values=pd.Series(p_values[1:], index=factor_names, name="p_values"),
            r_squared=float(r2),
            adj_r_squared=float(adj_r2),
            n_obs=n,
            residuals=pd.Series(resid, index=df.index, name="residuals"),
        )

    # ------------------------------------------------------------------ #
    # Fama-MacBeth
    # ------------------------------------------------------------------ #

    def fama_macbeth(
        self,
        returns_panel: pd.DataFrame,
        factors_panel: pd.DataFrame,
        n_lags: int = 5,
    ) -> FMBResult:
        """Fama-MacBeth two-pass regression with Newey-West corrected t-stats.

        Pass 1: For each time t, run cross-sectional OLS of r_t on F_t → get γ_t.
        Pass 2: Time-series mean of γ_t; Newey-West HAC standard error.

        Parameters
        ----------
        returns_panel : DataFrame[time × assets] — realised returns
        factors_panel : DataFrame indexed as MultiIndex (time, factor) × assets
                        OR a 3-D structure flattened as (time × assets, factors)
                        In the simple case, pass a dict of DataFrames keyed by
                        factor name, each [time × assets].
        n_lags        : Newey-West lag truncation

        Notes
        -----
        This implementation accepts *factors_panel* as a DataFrame[time × (asset*factor)]
        with MultiIndex columns (asset, factor), which is the output of
        build_factor_panel() below.  Alternatively if factors_panel is a plain
        DataFrame[time × assets] it is treated as a single-factor regression.

        Returns
        -------
        FMBResult
        """
        # Determine factor structure
        if isinstance(factors_panel.columns, pd.MultiIndex):
            factor_names = list(factors_panel.columns.get_level_values(1).unique())
            assets = list(factors_panel.columns.get_level_values(0).unique())
        else:
            # Single factor or flat (assets as columns)
            factor_names = ["factor"]
            assets = list(factors_panel.columns)

        common_idx = returns_panel.index.intersection(factors_panel.index)

        gamma_t_records: list[dict[str, float]] = []
        r2_list: list[float] = []

        for t in common_idx:
            r_t = returns_panel.loc[t]

            if isinstance(factors_panel.columns, pd.MultiIndex):
                f_t = factors_panel.loc[t].unstack(level=1)  # assets × factors
            else:
                f_t_vals = factors_panel.loc[t]
                f_t = pd.DataFrame({factor_names[0]: f_t_vals})

            # Align
            common_assets = r_t.dropna().index.intersection(f_t.dropna().index)
            if len(common_assets) < len(factor_names) + 2:
                continue

            r_vec = r_t.loc[common_assets]
            F_mat = f_t.loc[common_assets, factor_names]

            try:
                res = self.cross_sectional_regression(r_vec, F_mat)
                gamma_t_records.append(res.factor_returns.to_dict())
                r2_list.append(res.r_squared)
            except Exception:
                continue

        if not gamma_t_records:
            raise ValueError("No valid cross-sectional regressions completed.")

        gamma_df = pd.DataFrame(gamma_t_records, index=common_idx[: len(gamma_t_records)])
        mean_gamma = gamma_df.mean()

        # Newey-West t-stats
        nw_t_stats: dict[str, float] = {}
        nw_p_values: dict[str, float] = {}

        for col in gamma_df.columns:
            g_series = gamma_df[col].dropna()
            nw_se = self._newey_west_se(g_series.values, n_lags=n_lags)
            t = float(g_series.mean() / nw_se) if nw_se > 0 else float("nan")
            p = float(2 * (1 - stats.t.cdf(abs(t), df=len(g_series) - 1)))
            nw_t_stats[col] = t
            nw_p_values[col] = p

        return FMBResult(
            factor_names=factor_names,
            mean_factor_returns=mean_gamma,
            fmb_t_stats=pd.Series(nw_t_stats, name="t_stat"),
            fmb_p_values=pd.Series(nw_p_values, name="p_value"),
            factor_return_series=gamma_df,
            mean_r_squared=float(np.nanmean(r2_list)),
            n_periods=len(gamma_t_records),
            n_lags=n_lags,
        )

    @staticmethod
    def _newey_west_se(gamma_series: np.ndarray, n_lags: int = 5) -> float:
        """Compute Newey-West HAC standard error of a time-series mean."""
        n = len(gamma_series)
        if n < 2:
            return float("nan")
        demeaned = gamma_series - gamma_series.mean()
        # Variance
        s0 = np.dot(demeaned, demeaned) / n
        s_sum = s0
        for lag in range(1, n_lags + 1):
            w = 1 - lag / (n_lags + 1)  # Bartlett kernel
            cov_lag = np.dot(demeaned[lag:], demeaned[: n - lag]) / n
            s_sum += 2 * w * cov_lag
        variance_mean = max(s_sum / n, 0.0)
        return float(np.sqrt(variance_mean))

    # ------------------------------------------------------------------ #
    # Factor IC
    # ------------------------------------------------------------------ #

    def factor_ic(
        self,
        factor_values: pd.DataFrame,
        forward_returns: pd.Series,
        method: str = "spearman",
    ) -> Dict[str, float]:
        """Compute IC between each factor and forward returns.

        Parameters
        ----------
        factor_values   : DataFrame[obs × factors]
        forward_returns : pd.Series of forward returns for same obs
        method          : 'spearman' | 'pearson' | 'kendall'

        Returns
        -------
        Dict[factor_name → IC]
        """
        result: dict[str, float] = {}
        for col in factor_values.columns:
            df = pd.concat({"f": factor_values[col], "r": forward_returns}, axis=1).dropna()
            if len(df) < 3:
                result[col] = float("nan")
                continue
            f_val = df["f"].values
            r_val = df["r"].values
            if method == "pearson":
                r, _ = stats.pearsonr(f_val, r_val)
            elif method == "spearman":
                r, _ = stats.spearmanr(f_val, r_val)
            else:
                r, _ = stats.kendalltau(f_val, r_val)
            result[col] = float(r)
        return result

    # ------------------------------------------------------------------ #
    # Factor attribution
    # ------------------------------------------------------------------ #

    def factor_attribution(
        self,
        trades: pd.DataFrame,
        price_history: Optional[pd.DataFrame] = None,
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
    ) -> AttributionResult:
        """Decompose trade returns into factor-explained and idiosyncratic.

        Parameters
        ----------
        trades       : trade DataFrame
        price_history: optional price panel for momentum/vol factors
        return_col   : column containing realised P&L
        dollar_pos_col: column for position size

        Returns
        -------
        AttributionResult
        """
        # Compute returns (normalise by position if available)
        df = trades.copy()
        if dollar_pos_col in df.columns:
            pos = df[dollar_pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[return_col] / pos
        else:
            df["_ret"] = df[return_col]

        factor_df = self.build_factor_matrix(df, price_history)
        returns = df["_ret"].loc[factor_df.index].dropna()
        factors_aligned = factor_df.loc[returns.index]

        try:
            reg_result = self.cross_sectional_regression(returns, factors_aligned)
        except Exception as e:
            raise RuntimeError(f"Factor attribution regression failed: {e}") from e

        # Factor contributions: β_k × F_{i,k} summed, then averaged
        factor_contrib_matrix = factors_aligned * reg_result.factor_returns
        factor_contributions = factor_contrib_matrix.mean()

        sys_returns = (factors_aligned * reg_result.factor_returns).sum(axis=1)
        idio_returns = reg_result.residuals

        total_ret = float(returns.mean())
        sys_ret = float(sys_returns.mean())
        idio_ret = float(idio_returns.mean())

        return AttributionResult(
            total_return=total_ret,
            systematic_return=sys_ret,
            idiosyncratic_return=idio_ret,
            factor_contributions=factor_contributions,
            factor_names=list(factors_aligned.columns),
            r_squared=reg_result.r_squared,
            trades_count=len(returns),
        )

    # ------------------------------------------------------------------ #
    # Residual returns
    # ------------------------------------------------------------------ #

    def residual_returns(
        self,
        returns: pd.Series,
        factor_loadings: pd.DataFrame,
        factor_returns: pd.Series,
    ) -> pd.Series:
        """Compute idiosyncratic (residual) returns.

        r_idio = r - F @ γ

        Parameters
        ----------
        returns        : pd.Series of realised returns
        factor_loadings: DataFrame[obs × factors]
        factor_returns : pd.Series[factor → γ]

        Returns
        -------
        pd.Series of residual returns
        """
        common_factors = factor_loadings.columns.intersection(factor_returns.index)
        F = factor_loadings[common_factors]
        gamma = factor_returns[common_factors]
        systematic = (F * gamma).sum(axis=1)
        return (returns - systematic).rename("idiosyncratic_return")

    # ------------------------------------------------------------------ #
    # PCA factors
    # ------------------------------------------------------------------ #

    def pca_factors(
        self,
        returns_panel: pd.DataFrame,
        n_components: int = 5,
        standardise: bool = True,
    ) -> PCAFactorResult:
        """Extract latent factors via PCA on the returns panel.

        Parameters
        ----------
        returns_panel : DataFrame[time × assets]
        n_components  : number of principal components to retain
        standardise   : z-score each asset's returns before PCA

        Returns
        -------
        PCAFactorResult
        """
        df = returns_panel.dropna(axis=1, thresh=int(0.8 * len(returns_panel)))
        df = df.fillna(df.mean())

        X = df.values  # T × N
        if standardise:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

        n_comp = min(n_components, X.shape[1], X.shape[0] - 1)
        pca = PCA(n_components=n_comp)
        factor_returns_arr = pca.fit_transform(X)  # T × K

        factor_cols = [f"PC{i+1}" for i in range(n_comp)]
        factor_returns_df = pd.DataFrame(
            factor_returns_arr, index=df.index, columns=factor_cols
        )
        # Loadings: N × K
        loadings_df = pd.DataFrame(
            pca.components_.T, index=df.columns, columns=factor_cols
        )
        cum_var = np.cumsum(pca.explained_variance_ratio_)

        return PCAFactorResult(
            n_components=n_comp,
            explained_variance_ratio=pca.explained_variance_ratio_,
            cumulative_variance=cum_var,
            factor_loadings=loadings_df,
            factor_returns=factor_returns_df,
            eigenvalues=pca.explained_variance_,
        )

    # ------------------------------------------------------------------ #
    # Build factor panel (MultiIndex columns helper)
    # ------------------------------------------------------------------ #

    def build_factor_panel(
        self,
        factor_dicts: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Combine per-factor DataFrames (each time × assets) into MultiIndex panel.

        Parameters
        ----------
        factor_dicts : {factor_name: DataFrame[time × assets]}

        Returns
        -------
        DataFrame with MultiIndex columns (asset, factor)
        """
        frames = []
        for fname, df in factor_dicts.items():
            df_copy = df.copy()
            df_copy.columns = pd.MultiIndex.from_tuples(
                [(col, fname) for col in df_copy.columns],
                names=["asset", "factor"],
            )
            frames.append(df_copy)
        return pd.concat(frames, axis=1)

    # ------------------------------------------------------------------ #
    # Visualisations
    # ------------------------------------------------------------------ #

    def plot_factor_ic_bar(
        self,
        factor_ics: Dict[str, float],
        save_path: Optional[str | Path] = None,
        title: str = "Factor IC",
    ) -> plt.Figure:
        """Horizontal bar chart of per-factor IC values.

        Parameters
        ----------
        factor_ics : dict of factor_name → IC
        save_path  : optional save path
        title      : figure title
        """
        fig, ax = plt.subplots(figsize=(8, max(4, len(factor_ics) * 0.5)))
        names = list(factor_ics.keys())
        values = [factor_ics[n] for n in names]
        colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in values]

        y_pos = np.arange(len(names))
        ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor="white")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=10)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("IC (Spearman)")
        ax.set_title(title)
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_attribution_waterfall(
        self,
        attribution: AttributionResult,
        save_path: Optional[str | Path] = None,
        title: str = "Return Attribution",
    ) -> plt.Figure:
        """Waterfall chart of factor contributions to total return.

        Parameters
        ----------
        attribution : AttributionResult from factor_attribution()
        save_path   : optional save path
        title       : figure title
        """
        fig, ax = plt.subplots(figsize=(10, 5))

        # Build waterfall
        contrib = attribution.factor_contributions.copy()
        contrib["idiosyncratic"] = attribution.idiosyncratic_return
        contrib["TOTAL"] = attribution.total_return

        names = list(contrib.index)
        values = contrib.values.tolist()

        # Running total for bar offsets
        running = 0.0
        bottoms: list[float] = []
        colors: list[str] = []
        for i, v in enumerate(values):
            if names[i] == "TOTAL":
                bottoms.append(0.0)
                colors.append("#2c3e50")
            else:
                bottoms.append(running)
                running += v
                colors.append("#2ecc71" if v >= 0 else "#e74c3c")

        x = np.arange(len(names))
        bars = ax.bar(x, values, bottom=bottoms, color=colors, alpha=0.8, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.8)

        # Add value labels
        for bar, v, b in zip(bars, values, bottoms):
            label_y = b + v / 2
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{v:.4f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
            )

        ax.set_ylabel("Return Contribution")
        ax.set_title(f"{title}  |  R²={attribution.r_squared:.3f}  |  N={attribution.trades_count}")
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_pca_scree(
        self,
        pca_result: PCAFactorResult,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """Scree plot of PCA explained variance."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        components = np.arange(1, pca_result.n_components + 1)
        ax1.bar(components, pca_result.explained_variance_ratio * 100, color="#3498db", alpha=0.7)
        ax1.set_xlabel("Component")
        ax1.set_ylabel("Explained Variance (%)")
        ax1.set_title("PCA Scree Plot")
        ax1.set_xticks(components)

        ax2.plot(components, pca_result.cumulative_variance * 100, "o-", color="#e74c3c", linewidth=2)
        ax2.set_xlabel("Component")
        ax2.set_ylabel("Cumulative Variance (%)")
        ax2.set_title("Cumulative Explained Variance")
        ax2.set_xticks(components)
        ax2.axhline(80, color="gray", linestyle="--", label="80% threshold")
        ax2.legend()

        fig.tight_layout()
        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_factor_returns_heatmap(
        self,
        fmb_result: FMBResult,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """Heatmap of factor return time-series from FMB."""
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(12, max(4, len(fmb_result.factor_names) * 0.8)))
        data = fmb_result.factor_return_series.T  # factors × time
        sns.heatmap(
            data,
            ax=ax,
            center=0,
            cmap="RdYlGn",
            linewidths=0.1,
            cbar_kws={"label": "Factor Return γ"},
        )
        ax.set_title("Fama-MacBeth Factor Returns Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Factor")
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    # ------------------------------------------------------------------ #
    # FMB summary table
    # ------------------------------------------------------------------ #

    def fmb_summary_table(self, fmb_result: FMBResult) -> pd.DataFrame:
        """Return a summary DataFrame of FMB results.

        Columns: mean_gamma, t_stat, p_value, significant
        """
        rows = []
        for fname in fmb_result.factor_names:
            mg = fmb_result.mean_factor_returns.get(fname, float("nan"))
            t = fmb_result.fmb_t_stats.get(fname, float("nan"))
            p = fmb_result.fmb_p_values.get(fname, float("nan"))
            rows.append({
                "factor": fname,
                "mean_gamma": mg,
                "t_stat": t,
                "p_value": p,
                "significant_5pct": abs(t) > 1.96 if not np.isnan(t) else False,
                "significant_1pct": abs(t) > 2.576 if not np.isnan(t) else False,
            })
        return pd.DataFrame(rows).set_index("factor")

    # ------------------------------------------------------------------ #
    # Rolling factor attribution
    # ------------------------------------------------------------------ #

    def rolling_factor_attribution(
        self,
        returns_panel: pd.DataFrame,
        factors_panel: pd.DataFrame,
        window: int = 60,
    ) -> pd.DataFrame:
        """Rolling cross-sectional R² over time.

        Parameters
        ----------
        returns_panel : DataFrame[time × assets]
        factors_panel : DataFrame[time × assets] (single factor) or MultiIndex
        window        : rolling window size

        Returns
        -------
        pd.DataFrame with columns: r_squared, n_obs per time step
        """
        common_idx = returns_panel.index.intersection(factors_panel.index)
        records: list[dict] = []

        for i in range(window - 1, len(common_idx)):
            t_slice = common_idx[i - window + 1 : i + 1]
            r_sub = returns_panel.loc[t_slice]
            f_sub = factors_panel.loc[t_slice]

            # Flatten to obs-level
            r_flat = r_sub.stack().dropna()
            if isinstance(f_sub.columns, pd.MultiIndex):
                f_flat = f_sub.stack(level=0).dropna()
            else:
                f_flat = f_sub.stack().to_frame(name="factor").dropna()

            try:
                r_aligned = r_flat.loc[r_flat.index.intersection(f_flat.index)]
                f_aligned = f_flat.loc[r_aligned.index]
                if len(r_aligned) < 5:
                    records.append({"time": common_idx[i], "r_squared": float("nan"), "n_obs": 0})
                    continue
                res = self.cross_sectional_regression(r_aligned, f_aligned)
                records.append({
                    "time": common_idx[i],
                    "r_squared": res.r_squared,
                    "n_obs": res.n_obs,
                })
            except Exception:
                records.append({"time": common_idx[i], "r_squared": float("nan"), "n_obs": 0})

        return pd.DataFrame(records).set_index("time")

    # ------------------------------------------------------------------ #
    # Factor correlation matrix
    # ------------------------------------------------------------------ #

    def factor_correlation_matrix(
        self,
        factor_df: pd.DataFrame,
        method: str = "pearson",
    ) -> pd.DataFrame:
        """Correlation matrix of factor exposures.

        High inter-factor correlations indicate collinearity issues and
        potentially misleading attribution.

        Parameters
        ----------
        factor_df : DataFrame[obs x factors] from build_factor_matrix()
        method    : 'pearson' or 'spearman'

        Returns
        -------
        pd.DataFrame — symmetric correlation matrix
        """
        if method == "spearman":
            return factor_df.rank().corr()
        return factor_df.corr()

    # ------------------------------------------------------------------ #
    # VIF (Variance Inflation Factor)
    # ------------------------------------------------------------------ #

    def variance_inflation_factors(
        self,
        factor_df: pd.DataFrame,
    ) -> pd.Series:
        """Compute Variance Inflation Factor for each factor.

        VIF > 10 suggests severe collinearity; VIF > 5 is moderate.

        Parameters
        ----------
        factor_df : factor exposure matrix (obs x factors)

        Returns
        -------
        pd.Series[factor -> VIF]
        """
        df = factor_df.dropna()
        vifs: dict[str, float] = {}
        cols = df.columns.tolist()

        for i, col in enumerate(cols):
            other_cols = [c for c in cols if c != col]
            if not other_cols:
                vifs[col] = float("nan")
                continue

            X = df[other_cols].values
            y = df[col].values

            # Add intercept
            X_int = np.column_stack([np.ones(len(y)), X])
            beta, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
            y_hat = X_int @ beta
            ss_res = np.sum((y - y_hat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vifs[col] = 1 / (1 - r2) if r2 < 1 else float("inf")

        return pd.Series(vifs, name="VIF")

    # ------------------------------------------------------------------ #
    # Factor premium time-series
    # ------------------------------------------------------------------ #

    def factor_premium_timeseries(
        self,
        fmb_result: FMBResult,
    ) -> pd.DataFrame:
        """Cumulative factor premium returns over time.

        Parameters
        ----------
        fmb_result : FMBResult from fama_macbeth()

        Returns
        -------
        pd.DataFrame — cumulative factor returns (time x factors)
        """
        gamma_df = fmb_result.factor_return_series
        cum_gamma = gamma_df.cumsum()
        return cum_gamma

    # ------------------------------------------------------------------ #
    # Rolling factor IC (by factor)
    # ------------------------------------------------------------------ #

    def rolling_factor_ic(
        self,
        factor_df: pd.DataFrame,
        returns: pd.Series,
        window: int = 60,
        method: str = "spearman",
    ) -> pd.DataFrame:
        """Rolling IC for each factor over a time-indexed trades series.

        Parameters
        ----------
        factor_df : DataFrame[obs x factors] — factor exposures indexed by time/trade
        returns   : pd.Series of forward returns indexed by time/trade
        window    : rolling window (number of observations)
        method    : correlation method

        Returns
        -------
        pd.DataFrame[time x factors] of rolling IC values
        """
        aligned = factor_df.join(returns.rename("_ret"), how="inner").dropna()
        n = len(aligned)
        idx = aligned.index

        rolling_records: list[dict] = []
        for i in range(window - 1, n):
            window_df = aligned.iloc[i - window + 1 : i + 1]
            row: dict = {"idx": idx[i]}
            for col in factor_df.columns:
                sv = window_df[col].values
                rv = window_df["_ret"].values
                mask = ~(np.isnan(sv) | np.isnan(rv))
                if mask.sum() < 3:
                    row[col] = float("nan")
                    continue
                if method == "spearman":
                    r, _ = stats.spearmanr(sv[mask], rv[mask])
                else:
                    r, _ = stats.pearsonr(sv[mask], rv[mask])
                row[col] = float(r)
            rolling_records.append(row)

        if not rolling_records:
            return pd.DataFrame()
        result = pd.DataFrame(rolling_records).set_index("idx")
        return result

    # ------------------------------------------------------------------ #
    # Factor exposure stability
    # ------------------------------------------------------------------ #

    def factor_exposure_stability(
        self,
        trades: pd.DataFrame,
        price_history: Optional[pd.DataFrame] = None,
        n_periods: int = 4,
    ) -> pd.DataFrame:
        """Measure stability of factor loadings across time periods.

        Parameters
        ----------
        trades       : trade records
        price_history: optional price panel
        n_periods    : number of time sub-periods

        Returns
        -------
        pd.DataFrame with mean and std of factor loadings per period
        """
        df = trades.copy()
        if "exit_time" in df.columns:
            df = df.sort_values("exit_time")

        period_size = max(1, len(df) // n_periods)
        records: list[dict] = []

        for i in range(n_periods):
            sub = df.iloc[i * period_size : (i + 1) * period_size]
            try:
                factors = self.build_factor_matrix(sub, price_history)
                row = {"period": i + 1}
                row.update(factors.mean().to_dict())
                records.append(row)
            except Exception:
                records.append({"period": i + 1})

        return pd.DataFrame(records).set_index("period")

    # ------------------------------------------------------------------ #
    # Idiosyncratic return Sharpe
    # ------------------------------------------------------------------ #

    def idiosyncratic_sharpe(
        self,
        trades: pd.DataFrame,
        price_history: Optional[pd.DataFrame] = None,
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
        bars_per_year: int = 252,
    ) -> float:
        """Sharpe ratio of idiosyncratic (factor-residual) returns.

        High idio Sharpe suggests genuine alpha beyond factor exposures.

        Parameters
        ----------
        trades        : trade records
        price_history : optional price panel
        return_col    : P&L column
        dollar_pos_col: position column
        bars_per_year : annualisation factor

        Returns
        -------
        float annualised Sharpe of idiosyncratic returns
        """
        attr = self.factor_attribution(
            trades, price_history=price_history,
            return_col=return_col, dollar_pos_col=dollar_pos_col,
        )
        idio = attr.idiosyncratic_return
        # Approximate: we only have the mean idio return from attribution
        # For a proper Sharpe we need the distribution — use total return std
        df = trades.copy()
        if dollar_pos_col in df.columns:
            pos = df[dollar_pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[return_col] / pos
        else:
            df["_ret"] = df[return_col]

        r_arr = df["_ret"].dropna().values
        if len(r_arr) < 2:
            return float("nan")

        std_total = float(np.std(r_arr, ddof=1))
        if std_total == 0:
            return float("nan")
        return float(idio / std_total * np.sqrt(bars_per_year))

    # ------------------------------------------------------------------ #
    # Factor model diagnostic summary
    # ------------------------------------------------------------------ #

    def model_diagnostic_summary(
        self,
        trades: pd.DataFrame,
        price_history: Optional[pd.DataFrame] = None,
        return_col: str = "pnl",
        dollar_pos_col: str = "dollar_pos",
    ) -> pd.DataFrame:
        """Produce a comprehensive factor model diagnostic summary table.

        Returns a DataFrame with columns:
          factor, ic, vif, mean_exposure, std_exposure

        Parameters
        ----------
        trades       : trade records
        price_history: optional price panel

        Returns
        -------
        pd.DataFrame indexed by factor name
        """
        df = trades.copy()
        if dollar_pos_col in df.columns:
            pos = df[dollar_pos_col].abs().replace(0, np.nan)
            df["_ret"] = df[return_col] / pos
        else:
            df["_ret"] = df[return_col]

        factor_df = self.build_factor_matrix(df, price_history)
        ret = df["_ret"].loc[factor_df.index].dropna()
        factor_aligned = factor_df.loc[ret.index]

        # Factor ICs
        fics = self.factor_ic(factor_aligned, ret)

        # VIFs
        vifs = self.variance_inflation_factors(factor_aligned)

        # Exposure stats
        mean_exp = factor_aligned.mean()
        std_exp = factor_aligned.std(ddof=1)

        records: list[dict] = []
        for col in factor_aligned.columns:
            records.append({
                "factor": col,
                "ic": fics.get(col, float("nan")),
                "vif": vifs.get(col, float("nan")),
                "mean_exposure": float(mean_exp.get(col, float("nan"))),
                "std_exposure": float(std_exp.get(col, float("nan"))),
            })

        return pd.DataFrame(records).set_index("factor")

    # ------------------------------------------------------------------ #
    # Orthogonalise factors (Gram-Schmidt)
    # ------------------------------------------------------------------ #

    def orthogonalise_factors(
        self,
        factor_df: pd.DataFrame,
        order: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Orthogonalise factor exposures using Gram-Schmidt.

        Each factor is regressed on all previous factors and the residual
        is used as the "purified" factor.

        Parameters
        ----------
        factor_df : factor exposure matrix (obs x factors)
        order     : ordering of factors (priority order); defaults to column order

        Returns
        -------
        pd.DataFrame of orthogonalised factor exposures
        """
        df = factor_df.dropna().copy()
        if order is None:
            order = list(df.columns)

        ortho = pd.DataFrame(index=df.index)
        for i, col in enumerate(order):
            if col not in df.columns:
                continue
            y = df[col].values
            if i == 0:
                ortho[col] = y
                continue
            # Regress on all previous orthogonal factors
            prev_cols = [c for c in order[:i] if c in ortho.columns]
            X = ortho[prev_cols].values
            X_int = np.column_stack([np.ones(len(y)), X])
            beta, _, _, _ = np.linalg.lstsq(X_int, y, rcond=None)
            residual = y - X_int @ beta
            ortho[col] = residual

        return ortho

    # ------------------------------------------------------------------ #
    # Factor return predictability
    # ------------------------------------------------------------------ #

    def factor_return_autocorrelation(
        self,
        fmb_result: FMBResult,
        max_lag: int = 10,
    ) -> pd.DataFrame:
        """ACF of each factor's time-series of cross-sectional returns.

        Persistent factor returns suggest potential for factor timing.

        Parameters
        ----------
        fmb_result : FMBResult from fama_macbeth()
        max_lag    : maximum lag to compute

        Returns
        -------
        pd.DataFrame[lag x factor] of autocorrelation values
        """
        gamma_df = fmb_result.factor_return_series.dropna()
        result: dict[str, list[float]] = {}

        for col in gamma_df.columns:
            g = gamma_df[col].values
            n = len(g)
            mean_g = g.mean()
            var_g = np.var(g, ddof=1)
            acfs: list[float] = []
            for lag in range(1, min(max_lag + 1, n - 1)):
                cov = np.mean((g[:n - lag] - mean_g) * (g[lag:] - mean_g))
                acfs.append(cov / var_g if var_g > 0 else float("nan"))
            result[col] = acfs

        max_len = max(len(v) for v in result.values()) if result else 0
        # Pad with NaN if lengths differ
        for k in result:
            while len(result[k]) < max_len:
                result[k].append(float("nan"))

        return pd.DataFrame(result, index=range(1, max_len + 1))

    # ------------------------------------------------------------------ #
    # Plot factor loading heatmap
    # ------------------------------------------------------------------ #

    def plot_factor_loading_heatmap(
        self,
        factor_df: pd.DataFrame,
        n_top: int = 20,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """Heatmap of factor loadings for top-N most active assets.

        Parameters
        ----------
        factor_df : factor exposure matrix (obs x factors)
        n_top     : number of rows to show (sorted by total absolute loading)
        save_path : optional save path
        """
        import seaborn as sns

        # If obs are individual trades, we just show the factor distributions
        # as a histogram — fall back to this if no asset index
        data = factor_df.head(n_top).T

        fig, ax = plt.subplots(figsize=(max(8, n_top * 0.4), len(factor_df.columns) * 0.6 + 2))
        sns.heatmap(
            data, ax=ax, center=0, cmap="RdYlGn",
            linewidths=0.2,
            cbar_kws={"label": "Factor Exposure (z-score)"},
        )
        ax.set_title(f"Factor Exposures (first {n_top} observations)")
        ax.set_ylabel("Factor")
        ax.set_xlabel("Observation")
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    # ------------------------------------------------------------------ #
    # Cross-sectional R² time-series plot
    # ------------------------------------------------------------------ #

    def plot_rolling_r_squared(
        self,
        rolling_attr_df: pd.DataFrame,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """Plot rolling cross-sectional R² over time.

        Parameters
        ----------
        rolling_attr_df : output of rolling_factor_attribution()
        save_path       : optional save path
        """
        fig, ax = plt.subplots(figsize=(12, 5))
        r2 = rolling_attr_df["r_squared"].dropna()
        ax.plot(r2.index, r2.values, color="#3498db", linewidth=1.0)
        ax.fill_between(r2.index, 0, r2.values, alpha=0.2, color="#3498db")
        ax.axhline(r2.mean(), color="#e67e22", linestyle="--", linewidth=1.2,
                   label=f"Mean R^2={r2.mean():.3f}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Cross-sectional R^2")
        ax.set_title("Rolling Factor Model R^2")
        ax.legend()
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    # ------------------------------------------------------------------
    # Extended diagnostics – Barra-style risk decomposition
    # ------------------------------------------------------------------

    def factor_risk_decomposition(
        self,
        panel: pd.DataFrame,
        factor_cols: list[str] | None = None,
        return_col: str = "fwd_ret",
    ) -> pd.DataFrame:
        """
        Decompose total return variance into systematic (factor) and
        idiosyncratic components for each asset using a Barra-style
        cross-sectional OLS.

        For each asset i with T observations:
            R_i = sum_k (beta_ik * F_k) + eps_i
        Systematic variance = Var(sum_k beta_ik * F_k)
        Idiosyncratic variance = Var(eps_i)

        Parameters
        ----------
        panel : pd.DataFrame
            Long-format panel with columns [sym/asset, date, return_col,
            factor_cols...].
        factor_cols : list of str, optional
            Factor columns to include. Defaults to DEFAULT_FACTORS
            intersected with panel columns.
        return_col : str
            Forward-return column name.

        Returns
        -------
        pd.DataFrame indexed by asset with columns:
            total_var, systematic_var, idiosyncratic_var,
            systematic_pct, idiosyncratic_pct, beta_<factor>.
        """
        fcols = factor_cols or [f for f in self.DEFAULT_FACTORS if f in panel.columns]
        asset_col = "sym" if "sym" in panel.columns else panel.columns[0]

        rows: list[dict] = []
        for asset, grp in panel.groupby(asset_col):
            grp = grp.dropna(subset=[return_col] + fcols)
            if len(grp) < len(fcols) + 2:
                continue
            X = grp[fcols].values.astype(float)
            y = grp[return_col].values.astype(float)

            X_c = np.column_stack([np.ones(len(X)), X])
            try:
                coef, _, _, _ = np.linalg.lstsq(X_c, y, rcond=None)
            except np.linalg.LinAlgError:
                continue

            fitted = X_c @ coef
            residuals = y - fitted

            total_var = float(np.var(y, ddof=1)) if len(y) > 1 else np.nan
            sys_var = float(np.var(fitted, ddof=1)) if len(fitted) > 1 else np.nan
            idio_var = float(np.var(residuals, ddof=1)) if len(residuals) > 1 else np.nan

            row: dict = {
                asset_col: asset,
                "total_var": total_var,
                "systematic_var": sys_var,
                "idiosyncratic_var": idio_var,
                "systematic_pct": sys_var / total_var if total_var > 0 else np.nan,
                "idiosyncratic_pct": idio_var / total_var if total_var > 0 else np.nan,
            }
            for k, fname in enumerate(fcols):
                row[f"beta_{fname}"] = float(coef[k + 1])
            rows.append(row)

        return pd.DataFrame(rows).set_index(asset_col)

    def cross_sectional_r_squared_timeseries(
        self,
        panel: pd.DataFrame,
        factor_cols: list[str] | None = None,
        return_col: str = "fwd_ret",
        date_col: str = "date",
    ) -> pd.Series:
        """
        Compute the cross-sectional R^2 at each time slice and return a
        time series.

        Parameters
        ----------
        panel : pd.DataFrame
            Long-format panel with date, return and factor columns.
        factor_cols : list of str, optional
            Factor columns.
        return_col : str
            Return column name.
        date_col : str
            Date column name.

        Returns
        -------
        pd.Series indexed by date of cross-sectional R^2 values.
        """
        fcols = factor_cols or [f for f in self.DEFAULT_FACTORS if f in panel.columns]
        rows: list[tuple] = []
        for date, grp in panel.groupby(date_col):
            sub = grp.dropna(subset=[return_col] + fcols)
            if len(sub) < len(fcols) + 2:
                continue
            X = sub[fcols].values.astype(float)
            y = sub[return_col].values.astype(float)
            X_c = np.column_stack([np.ones(len(X)), X])
            try:
                coef, _, _, _ = np.linalg.lstsq(X_c, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            fitted = X_c @ coef
            ss_res = np.sum((y - fitted) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            rows.append((date, float(r2)))

        if not rows:
            return pd.Series(dtype=float, name="r_squared")
        dates, vals = zip(*rows)
        return pd.Series(vals, index=list(dates), name="r_squared")

    def factor_beta_stability(
        self,
        panel: pd.DataFrame,
        factor_cols: list[str] | None = None,
        return_col: str = "fwd_ret",
        date_col: str = "date",
        window: int = 12,
    ) -> pd.DataFrame:
        """
        Compute rolling cross-sectional OLS factor betas and their stability
        statistics: mean, std, autocorrelation, and CV = std/abs(mean).

        Parameters
        ----------
        panel : pd.DataFrame
            Long-format panel.
        factor_cols : list of str, optional
            Factor columns.
        return_col : str
            Return column.
        date_col : str
            Date column.
        window : int
            Rolling window in time periods.

        Returns
        -------
        pd.DataFrame indexed by factor_name with columns:
            mean_beta, std_beta, min_beta, max_beta, cv, autocorr_lag1,
            pct_positive, t_stat, n_periods.
        """
        fcols = factor_cols or [f for f in self.DEFAULT_FACTORS if f in panel.columns]
        beta_records: dict[str, list[float]] = {f: [] for f in fcols}
        dates_list: list = []

        for date, grp in panel.groupby(date_col):
            sub = grp.dropna(subset=[return_col] + fcols)
            if len(sub) < len(fcols) + 2:
                continue
            X = sub[fcols].values.astype(float)
            y = sub[return_col].values.astype(float)
            X_c = np.column_stack([np.ones(len(X)), X])
            try:
                coef, _, _, _ = np.linalg.lstsq(X_c, y, rcond=None)
            except np.linalg.LinAlgError:
                continue
            dates_list.append(date)
            for k, fname in enumerate(fcols):
                beta_records[fname].append(float(coef[k + 1]))

        if not dates_list:
            return pd.DataFrame()

        beta_df = pd.DataFrame(beta_records, index=dates_list)
        stats_rows: list[dict] = []
        for fname in fcols:
            arr = beta_df[fname].dropna().values
            if len(arr) < 4:
                continue
            mean_b = float(np.mean(arr))
            std_b = float(np.std(arr, ddof=1))
            from scipy.stats import pearsonr
            autocorr = float(pearsonr(arr[:-1], arr[1:])[0]) if len(arr) > 2 else np.nan
            n = len(arr)
            t_stat = mean_b / (std_b / np.sqrt(n)) if std_b > 0 else np.nan
            pct_pos = float((arr > 0).mean())
            stats_rows.append({
                "factor": fname,
                "mean_beta": mean_b,
                "std_beta": std_b,
                "min_beta": float(arr.min()),
                "max_beta": float(arr.max()),
                "cv": std_b / abs(mean_b) if mean_b != 0 else np.nan,
                "autocorr_lag1": autocorr,
                "t_stat": t_stat,
                "pct_positive": pct_pos,
                "n_periods": n,
            })

        return pd.DataFrame(stats_rows).set_index("factor") if stats_rows else pd.DataFrame()

    def marginal_factor_contribution(
        self,
        panel: pd.DataFrame,
        factor_cols: list[str] | None = None,
        return_col: str = "fwd_ret",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        Compute the incremental R^2 of each factor via leave-one-out
        cross-sectional regressions, averaged over time.

        Parameters
        ----------
        panel : pd.DataFrame
            Long-format panel.
        factor_cols : list of str, optional
        return_col : str
        date_col : str

        Returns
        -------
        pd.DataFrame indexed by factor with columns:
            full_r2_mean, loo_r2_mean, marginal_r2, marginal_r2_pct.
        """
        fcols = factor_cols or [f for f in self.DEFAULT_FACTORS if f in panel.columns]
        if not fcols:
            return pd.DataFrame()

        def _r2(X_: np.ndarray, y_: np.ndarray) -> float:
            X_c = np.column_stack([np.ones(len(X_)), X_])
            try:
                coef, _, _, _ = np.linalg.lstsq(X_c, y_, rcond=None)
            except np.linalg.LinAlgError:
                return np.nan
            fitted = X_c @ coef
            ss_res = np.sum((y_ - fitted) ** 2)
            ss_tot = np.sum((y_ - y_.mean()) ** 2)
            return float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan

        full_r2s: list[float] = []
        loo_r2s: dict[str, list[float]] = {f: [] for f in fcols}

        for date, grp in panel.groupby(date_col):
            sub = grp.dropna(subset=[return_col] + fcols)
            if len(sub) < len(fcols) + 3:
                continue
            X_full = sub[fcols].values.astype(float)
            y = sub[return_col].values.astype(float)
            fr2 = _r2(X_full, y)
            full_r2s.append(fr2)

            for fname in fcols:
                loo_cols = [c for c in fcols if c != fname]
                if loo_cols:
                    X_loo = sub[loo_cols].values.astype(float)
                    loo_r2s[fname].append(_r2(X_loo, y))
                else:
                    loo_r2s[fname].append(0.0)

        full_mean = float(np.nanmean(full_r2s)) if full_r2s else np.nan
        rows: list[dict] = []
        for fname in fcols:
            loo_mean = float(np.nanmean(loo_r2s[fname])) if loo_r2s[fname] else np.nan
            marginal = full_mean - loo_mean if np.isfinite(full_mean) and np.isfinite(loo_mean) else np.nan
            marginal_pct = marginal / full_mean if full_mean and full_mean > 0 else np.nan
            rows.append({
                "factor": fname,
                "full_r2_mean": full_mean,
                "loo_r2_mean": loo_mean,
                "marginal_r2": marginal,
                "marginal_r2_pct": marginal_pct,
            })

        return pd.DataFrame(rows).set_index("factor") if rows else pd.DataFrame()

    def information_ratio_by_factor(
        self,
        panel: pd.DataFrame,
        factor_cols: list[str] | None = None,
        return_col: str = "fwd_ret",
        date_col: str = "date",
    ) -> pd.DataFrame:
        """
        Compute IC and ICIR for each factor against the forward return using
        cross-sectional Spearman correlation at each time slice.

        IC_t = Spearman corr(factor_k, return) at date t.
        ICIR = mean(IC_t) / std(IC_t)

        Parameters
        ----------
        panel : pd.DataFrame
        factor_cols : list of str, optional
        return_col : str
        date_col : str

        Returns
        -------
        pd.DataFrame indexed by factor with columns:
            ic_mean, ic_std, icir, ic_positive_pct, ic_t_stat, ic_p_value, n_periods.
        """
        from scipy.stats import spearmanr, t as t_dist

        fcols = factor_cols or [f for f in self.DEFAULT_FACTORS if f in panel.columns]
        factor_ics: dict[str, list[float]] = {f: [] for f in fcols}

        for date, grp in panel.groupby(date_col):
            sub = grp.dropna(subset=[return_col] + fcols)
            if len(sub) < 5:
                continue
            y = sub[return_col].values.astype(float)
            for fname in fcols:
                x = sub[fname].values.astype(float)
                valid = np.isfinite(x) & np.isfinite(y)
                if valid.sum() < 5:
                    continue
                rho, _ = spearmanr(x[valid], y[valid])
                factor_ics[fname].append(float(rho))

        rows: list[dict] = []
        for fname in fcols:
            arr = np.array(factor_ics[fname])
            if len(arr) < 3:
                continue
            ic_mean = float(np.mean(arr))
            ic_std = float(np.std(arr, ddof=1))
            icir = ic_mean / ic_std if ic_std > 0 else np.nan
            n = len(arr)
            t_stat = ic_mean / (ic_std / np.sqrt(n)) if ic_std > 0 else np.nan
            pval = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 1)) if np.isfinite(t_stat) else np.nan
            rows.append({
                "factor": fname,
                "ic_mean": ic_mean,
                "ic_std": ic_std,
                "icir": icir,
                "ic_positive_pct": float((arr > 0).mean()),
                "ic_t_stat": t_stat,
                "ic_p_value": pval,
                "n_periods": n,
            })

        return pd.DataFrame(rows).set_index("factor") if rows else pd.DataFrame()

    def plot_factor_beta_stability(
        self,
        stability_df: pd.DataFrame,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """
        Grouped bar chart of factor mean beta +- std.

        Parameters
        ----------
        stability_df : output of factor_beta_stability()
        save_path    : optional path to save the figure

        Returns
        -------
        matplotlib.figure.Figure
        """
        df = stability_df.reset_index()
        if "factor" not in df.columns:
            df = df.rename_axis("factor").reset_index()

        factors = df["factor"].tolist()
        means = df["mean_beta"].values
        stds = df["std_beta"].values if "std_beta" in df.columns else np.zeros(len(means))

        fig, ax = plt.subplots(figsize=(max(7, len(factors) * 1.2), 5))
        colors = ["#2ecc71" if m > 0 else "#e74c3c" for m in means]
        ax.bar(factors, means, yerr=stds, color=colors, alpha=0.75,
               edgecolor="black", capsize=5)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title("Factor Beta Stability (Mean +- Std)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Factor")
        ax.set_ylabel("Cross-sectional Beta")
        ax.set_xticks(range(len(factors)))
        ax.set_xticklabels(factors, rotation=30, ha="right")
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_marginal_r2_bar(
        self,
        marginal_df: pd.DataFrame,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """
        Horizontal bar chart of each factor's marginal R^2 contribution.

        Parameters
        ----------
        marginal_df : output of marginal_factor_contribution()
        save_path   : optional save path

        Returns
        -------
        matplotlib.figure.Figure
        """
        df = marginal_df.reset_index()
        if "factor" not in df.columns:
            df = df.rename_axis("factor").reset_index()

        df = df.sort_values("marginal_r2", ascending=True)
        factors = df["factor"].tolist()
        vals = df["marginal_r2"].values * 100

        fig, ax = plt.subplots(figsize=(8, max(4, len(factors) * 0.5)))
        colors = ["#3498db" if v >= 0 else "#e74c3c" for v in vals]
        ax.barh(factors, vals, color=colors, alpha=0.75, edgecolor="black")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title("Marginal R^2 Contribution per Factor (%)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Incremental R^2 (%)")
        ax.set_ylabel("Factor")
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig

    def plot_factor_icir_bar(
        self,
        ir_df: pd.DataFrame,
        save_path: Optional[str | Path] = None,
    ) -> plt.Figure:
        """
        Side-by-side bar charts: ICIR per factor and IC mean +- std.

        Parameters
        ----------
        ir_df     : output of information_ratio_by_factor()
        save_path : optional save path

        Returns
        -------
        matplotlib.figure.Figure
        """
        df = ir_df.reset_index()
        if "factor" not in df.columns:
            df = df.rename_axis("factor").reset_index()

        df = df.sort_values("icir", ascending=False)
        factors = df["factor"].tolist()
        icirs = df["icir"].values
        ic_means = df["ic_mean"].values if "ic_mean" in df.columns else np.zeros(len(icirs))
        ic_stds = df["ic_std"].values if "ic_std" in df.columns else np.zeros(len(icirs))

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax1 = axes[0]
        colors1 = ["#27ae60" if v > 0 else "#c0392b" for v in icirs]
        ax1.bar(factors, icirs, color=colors1, alpha=0.75, edgecolor="black")
        ax1.axhline(0, color="black", linewidth=0.8)
        ax1.axhline(0.5, color="green", linewidth=0.8, linestyle="--", alpha=0.6, label="ICIR=0.5")
        ax1.axhline(-0.5, color="red", linewidth=0.8, linestyle="--", alpha=0.6)
        ax1.set_title("Factor ICIR", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Factor")
        ax1.set_ylabel("ICIR")
        ax1.set_xticks(range(len(factors)))
        ax1.set_xticklabels(factors, rotation=30, ha="right")
        ax1.legend(fontsize=8)

        ax2 = axes[1]
        colors2 = ["#27ae60" if v > 0 else "#c0392b" for v in ic_means]
        ax2.bar(factors, ic_means, yerr=ic_stds, color=colors2, alpha=0.75,
                edgecolor="black", capsize=4)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_title("Factor IC Mean +- Std", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Factor")
        ax2.set_ylabel("Spearman IC")
        ax2.set_xticks(range(len(factors)))
        ax2.set_xticklabels(factors, rotation=30, ha="right")

        fig.suptitle("Factor Information Ratio Summary", fontsize=14, fontweight="bold")
        fig.tight_layout()

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150)
        return fig
