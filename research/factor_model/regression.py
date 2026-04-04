"""
Factor model regression utilities.

Implements:
- Time-series regression (OLS with Newey-West HAC standard errors)
- Fama-MacBeth cross-sectional regression
- Rolling factor regression
- Factor exposure decomposition (returns attribution)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _newey_west_se(residuals: np.ndarray, X: np.ndarray, lags: int = 4) -> np.ndarray:
    """
    Newey-West HAC standard errors.

    Parameters
    ----------
    residuals : (T,) array
    X : (T, k) design matrix
    lags : int
        Number of lags for HAC kernel.

    Returns
    -------
    np.ndarray
        (k,) standard errors.
    """
    T, k = X.shape
    xe = X * residuals[:, None]  # (T, k)
    # Sandwich estimator: (X'X)^{-1} S (X'X)^{-1}
    XtX = X.T @ X / T
    try:
        XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(k))
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    # S matrix (HAC)
    S = xe.T @ xe / T
    for lag in range(1, lags + 1):
        weight = 1 - lag / (lags + 1)
        gamma = xe[lag:].T @ xe[:-lag] / T
        S += weight * (gamma + gamma.T)

    cov = XtX_inv @ S @ XtX_inv / T
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    return se


@dataclass
class RegressionResult:
    """Time-series factor regression output."""
    alpha: float
    betas: Dict[str, float]
    alpha_tstat: float
    beta_tstats: Dict[str, float]
    alpha_se: float
    beta_ses: Dict[str, float]
    r_squared: float
    adj_r_squared: float
    residuals: pd.Series
    fitted: pd.Series
    n_obs: int
    information_ratio: float  # alpha / tracking_error
    tracking_error: float


@dataclass
class FamaMacBethResult:
    """Fama-MacBeth regression output."""
    factor_premia: Dict[str, float]
    factor_premia_se: Dict[str, float]
    factor_premia_tstat: Dict[str, float]
    r_squared_mean: float
    n_cross_sections: int
    gamma_series: pd.DataFrame  # time series of cross-section coefficients
    intercept_series: pd.Series


# ---------------------------------------------------------------------------
# Time-Series Regression
# ---------------------------------------------------------------------------

class TimeSeriesRegression:
    """
    Time-series OLS regression of asset returns on factor returns.

    r_it = alpha_i + sum_k beta_ik * F_kt + eps_it

    Parameters
    ----------
    use_newey_west : bool
        If True, use Newey-West HAC standard errors.
    nw_lags : int
        Number of lags for Newey-West.
    annualize : bool
        If True, annualize alpha (multiply by freq).
    freq : int
        Number of periods per year (252 for daily, 12 for monthly).
    """

    def __init__(
        self,
        use_newey_west: bool = True,
        nw_lags: int = 4,
        annualize: bool = True,
        freq: int = 252,
    ) -> None:
        self.use_newey_west = use_newey_west
        self.nw_lags = nw_lags
        self.annualize = annualize
        self.freq = freq

    def fit(
        self,
        asset_returns: pd.Series,
        factor_returns: pd.DataFrame,
        include_intercept: bool = True,
    ) -> RegressionResult:
        """
        Fit time-series factor regression.

        Parameters
        ----------
        asset_returns : pd.Series
            Asset or portfolio return series.
        factor_returns : pd.DataFrame
            Factor return matrix (dates x factors).
        include_intercept : bool
            Whether to include alpha (intercept).

        Returns
        -------
        RegressionResult
        """
        combined = pd.concat([asset_returns, factor_returns], axis=1).dropna()
        y = combined.iloc[:, 0].values
        F = combined.iloc[:, 1:].values
        factor_names = list(factor_returns.columns)
        T = len(y)

        if include_intercept:
            X = np.column_stack([np.ones(T), F])
        else:
            X = F

        # OLS
        try:
            coeffs = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            coeffs = np.zeros(X.shape[1])

        fitted = X @ coeffs
        residuals = y - fitted
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        k = X.shape[1]
        adj_r2 = 1 - (1 - r2) * (T - 1) / max(T - k, 1)

        # Standard errors
        if self.use_newey_west:
            se = _newey_west_se(residuals, X, self.nw_lags)
        else:
            sigma2 = ss_res / max(T - k, 1)
            try:
                cov = sigma2 * np.linalg.inv(X.T @ X)
            except np.linalg.LinAlgError:
                cov = sigma2 * np.linalg.pinv(X.T @ X)
            se = np.sqrt(np.maximum(np.diag(cov), 0))

        t_stats = coeffs / (se + 1e-12)

        if include_intercept:
            alpha_raw = float(coeffs[0])
            alpha_se_val = float(se[0])
            alpha_t = float(t_stats[0])
            betas = {n: float(c) for n, c in zip(factor_names, coeffs[1:])}
            beta_ses = {n: float(s) for n, s in zip(factor_names, se[1:])}
            beta_ts = {n: float(t) for n, t in zip(factor_names, t_stats[1:])}
        else:
            alpha_raw = 0.0
            alpha_se_val = np.nan
            alpha_t = np.nan
            betas = {n: float(c) for n, c in zip(factor_names, coeffs)}
            beta_ses = {n: float(s) for n, s in zip(factor_names, se)}
            beta_ts = {n: float(t) for n, t in zip(factor_names, t_stats)}

        alpha = alpha_raw * self.freq if self.annualize else alpha_raw

        te = float(residuals.std() * np.sqrt(self.freq))
        ir = alpha / (te + 1e-12)

        return RegressionResult(
            alpha=round(alpha, 6),
            betas={k: round(v, 6) for k, v in betas.items()},
            alpha_tstat=round(alpha_t, 4),
            beta_tstats={k: round(v, 4) for k, v in beta_ts.items()},
            alpha_se=round(alpha_se_val, 6),
            beta_ses={k: round(v, 6) for k, v in beta_ses.items()},
            r_squared=round(r2, 6),
            adj_r_squared=round(adj_r2, 6),
            residuals=pd.Series(residuals, index=combined.index),
            fitted=pd.Series(fitted, index=combined.index),
            n_obs=T,
            information_ratio=round(ir, 4),
            tracking_error=round(te, 6),
        )

    def fit_universe(
        self,
        universe_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit regression for each asset in universe.

        Returns
        -------
        pd.DataFrame
            Alpha and beta estimates for each asset.
        """
        rows = []
        for col in universe_returns.columns:
            result = self.fit(universe_returns[col], factor_returns)
            row = {"ticker": col, "alpha": result.alpha, "r_squared": result.r_squared,
                   "ir": result.information_ratio, "tracking_error": result.tracking_error,
                   "alpha_tstat": result.alpha_tstat, "n_obs": result.n_obs}
            row.update({f"beta_{k}": v for k, v in result.betas.items()})
            row.update({f"tstat_{k}": v for k, v in result.beta_tstats.items()})
            rows.append(row)
        return pd.DataFrame(rows).set_index("ticker")


# ---------------------------------------------------------------------------
# Fama-MacBeth Cross-Sectional Regression
# ---------------------------------------------------------------------------

class FamaMacBeth:
    """
    Fama-MacBeth (1973) two-pass cross-sectional regression.

    Pass 1: Time-series OLS to estimate factor exposures (betas) for each asset.
    Pass 2: At each date, cross-sectional regression of returns on betas.
    Report time-series mean and standard error of cross-section coefficients.

    Parameters
    ----------
    first_pass_window : int
        Rolling window (periods) for estimating betas in Pass 1.
    newey_west_lags : int
        HAC lags for standard errors of factor premia.
    min_assets : int
        Minimum assets for a valid cross-sectional regression.
    """

    def __init__(
        self,
        first_pass_window: int = 60,
        newey_west_lags: int = 4,
        min_assets: int = 10,
    ) -> None:
        self.first_pass_window = first_pass_window
        self.newey_west_lags = newey_west_lags
        self.min_assets = min_assets

    def first_pass(
        self,
        universe_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Rolling time-series regression to get time-varying betas.

        Returns
        -------
        pd.DataFrame
            Multi-level columns: (factor, ticker) with rolling beta values.
        """
        factor_names = list(factor_returns.columns)
        # Store betas: index=dates, columns=MultiIndex(factor, ticker)
        beta_frames = {f: pd.DataFrame(np.nan, index=universe_returns.index,
                                        columns=universe_returns.columns)
                       for f in factor_names}

        for i in range(self.first_pass_window, len(universe_returns.index)):
            dt = universe_returns.index[i]
            window_rets = universe_returns.iloc[i - self.first_pass_window:i]
            window_factors = factor_returns.reindex(window_rets.index)

            for col in universe_returns.columns:
                y = window_rets[col].values
                F = window_factors.values
                mask = ~(np.isnan(y) | np.any(np.isnan(F), axis=1))
                if mask.sum() < self.first_pass_window // 2:
                    continue
                y_m = y[mask]
                F_m = F[mask]
                X = np.column_stack([np.ones(len(y_m)), F_m])
                try:
                    coeffs = np.linalg.lstsq(X, y_m, rcond=None)[0]
                except Exception:
                    continue
                for fi, fname in enumerate(factor_names):
                    beta_frames[fname].loc[dt, col] = coeffs[fi + 1]

        return beta_frames

    def second_pass(
        self,
        universe_returns: pd.DataFrame,
        beta_frames: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Cross-sectional regression at each date.

        Returns
        -------
        pd.DataFrame
            gamma_series: (dates x [intercept] + factors) coefficient estimates.
        """
        factor_names = list(beta_frames.keys())
        gamma_list = []
        gamma_index = []

        for i, dt in enumerate(universe_returns.index):
            if dt not in beta_frames[factor_names[0]].index:
                continue
            rets = universe_returns.loc[dt].dropna()
            if len(rets) < self.min_assets:
                continue

            # Build beta matrix at date dt
            beta_rows = {}
            for fname in factor_names:
                betas = beta_frames[fname].loc[dt].reindex(rets.index)
                beta_rows[fname] = betas

            beta_df = pd.DataFrame(beta_rows).dropna()
            common = beta_df.index.intersection(rets.index)
            if len(common) < self.min_assets:
                continue

            y = rets.loc[common].values
            X = np.column_stack([np.ones(len(common)), beta_df.loc[common].values])

            try:
                gamma = np.linalg.lstsq(X, y, rcond=None)[0]
            except Exception:
                continue

            gamma_list.append(gamma)
            gamma_index.append(dt)

        cols = ["intercept"] + factor_names
        gamma_df = pd.DataFrame(gamma_list, index=gamma_index, columns=cols)
        return gamma_df

    def fit(
        self,
        universe_returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
    ) -> FamaMacBethResult:
        """
        Full Fama-MacBeth estimation.

        Returns
        -------
        FamaMacBethResult
        """
        beta_frames = self.first_pass(universe_returns, factor_returns)
        gamma_df = self.second_pass(universe_returns, beta_frames)

        factor_names = list(factor_returns.columns)
        premia = {}
        premia_se = {}
        premia_t = {}

        for col in gamma_df.columns:
            series = gamma_df[col].dropna()
            if len(series) == 0:
                continue
            mu = series.mean()
            # Newey-West SE
            T = len(series)
            x = series.values
            resid = x - mu
            se_val = np.std(resid, ddof=1) / np.sqrt(T)
            # Apply NW correction
            nw_correction = 1.0
            for lag in range(1, self.newey_west_lags + 1):
                w = 1 - lag / (self.newey_west_lags + 1)
                autocov = np.mean(resid[lag:] * resid[:-lag])
                nw_correction += 2 * w * autocov / np.var(resid, ddof=1)
            se_nw = se_val * np.sqrt(max(nw_correction, 0.01))
            t_stat = mu / (se_nw + 1e-12)
            premia[col] = round(float(mu * 252), 4)  # annualized
            premia_se[col] = round(float(se_nw * np.sqrt(252)), 4)
            premia_t[col] = round(float(t_stat), 4)

        r2_mean = float(gamma_df.apply(
            lambda row: np.nan, axis=1
        ).mean()) if len(gamma_df) > 0 else np.nan

        return FamaMacBethResult(
            factor_premia=premia,
            factor_premia_se=premia_se,
            factor_premia_tstat=premia_t,
            r_squared_mean=r2_mean,
            n_cross_sections=len(gamma_df),
            gamma_series=gamma_df,
            intercept_series=gamma_df["intercept"] if "intercept" in gamma_df.columns
            else pd.Series(dtype=float),
        )

    def summary(self, result: FamaMacBethResult) -> pd.DataFrame:
        """Pretty-print summary of Fama-MacBeth results."""
        rows = []
        for factor in result.factor_premia:
            rows.append({
                "factor": factor,
                "premium_annual": result.factor_premia.get(factor, np.nan),
                "se_annual": result.factor_premia_se.get(factor, np.nan),
                "t_stat": result.factor_premia_tstat.get(factor, np.nan),
            })
        return pd.DataFrame(rows).set_index("factor")


# ---------------------------------------------------------------------------
# Rolling Factor Regression
# ---------------------------------------------------------------------------

class RollingRegression:
    """
    Rolling window time-series factor regression.

    Tracks how factor exposures change over time.

    Parameters
    ----------
    window : int
        Rolling window length.
    step : int
        Step size between regression windows.
    min_periods : int
        Minimum periods to compute regression.
    """

    def __init__(
        self,
        window: int = 252,
        step: int = 21,
        min_periods: int = 60,
    ) -> None:
        self.window = window
        self.step = step
        self.min_periods = min_periods

    def fit(
        self,
        asset_returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Rolling regression returning time-series of alpha and betas.

        Returns
        -------
        pd.DataFrame
            Columns: alpha, alpha_tstat, beta_F1, beta_F2, ..., r_squared.
        """
        combined = pd.concat([asset_returns, factor_returns], axis=1).dropna()
        y_all = combined.iloc[:, 0].values
        F_all = combined.iloc[:, 1:].values
        factor_names = list(factor_returns.columns)
        idx = combined.index

        records = []
        for i in range(self.window, len(idx) + 1, self.step):
            start = max(0, i - self.window)
            y = y_all[start:i]
            F = F_all[start:i]

            if len(y) < self.min_periods:
                continue

            mask = ~(np.isnan(y) | np.any(np.isnan(F), axis=1))
            y_m = y[mask]
            F_m = F[mask]
            X = np.column_stack([np.ones(len(y_m)), F_m])

            if len(y_m) < self.min_periods:
                continue

            try:
                coeffs = np.linalg.lstsq(X, y_m, rcond=None)[0]
            except Exception:
                continue

            fitted = X @ coeffs
            resid = y_m - fitted
            ss_res = np.sum(resid ** 2)
            ss_tot = np.sum((y_m - y_m.mean()) ** 2)
            r2 = 1 - ss_res / (ss_tot + 1e-12)

            sigma2 = ss_res / max(len(y_m) - X.shape[1], 1)
            try:
                cov = sigma2 * np.linalg.inv(X.T @ X)
            except Exception:
                cov = sigma2 * np.linalg.pinv(X.T @ X)
            se = np.sqrt(np.maximum(np.diag(cov), 0))
            tstats = coeffs / (se + 1e-12)

            rec = {
                "date": idx[min(i - 1, len(idx) - 1)],
                "alpha": float(coeffs[0] * 252),
                "alpha_tstat": float(tstats[0]),
                "r_squared": round(r2, 6),
                "n_obs": int(mask.sum()),
            }
            for fi, fname in enumerate(factor_names):
                rec[f"beta_{fname}"] = float(coeffs[fi + 1])
                rec[f"tstat_{fname}"] = float(tstats[fi + 1])

            records.append(rec)

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records).set_index("date")
        return df

    def plot_rolling_betas(
        self, rolling_result: pd.DataFrame, factor_names: List[str]
    ) -> Dict[str, pd.Series]:
        """Extract rolling beta series as a dict of pd.Series."""
        result = {}
        for fname in factor_names:
            col = f"beta_{fname}"
            if col in rolling_result.columns:
                result[fname] = rolling_result[col]
        return result


# ---------------------------------------------------------------------------
# Factor Exposure Decomposition (Return Attribution)
# ---------------------------------------------------------------------------

class FactorAttribution:
    """
    Decompose portfolio returns into factor contributions.

    Attribution formula:
      r_t = alpha_t + sum_k beta_k * F_kt + epsilon_t

    The contribution of factor k at time t is: beta_k * F_kt

    Parameters
    ----------
    regression_window : int
        Rolling window for beta estimation.
    factor_names : list of str
        Names of factors.
    """

    def __init__(
        self,
        regression_window: int = 252,
        factor_names: Optional[List[str]] = None,
    ) -> None:
        self.regression_window = regression_window
        self.factor_names = factor_names or []

    def decompose(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        rolling: bool = True,
    ) -> pd.DataFrame:
        """
        Decompose portfolio returns into factor contributions.

        Parameters
        ----------
        portfolio_returns : pd.Series
        factor_returns : pd.DataFrame
        rolling : bool
            If True, use rolling betas. If False, full-period betas.

        Returns
        -------
        pd.DataFrame
            Columns: alpha_contribution, factor_k_contribution, ..., residual.
        """
        if rolling:
            roller = RollingRegression(window=self.regression_window, step=1,
                                       min_periods=self.regression_window // 2)
            rolling_betas = roller.fit(portfolio_returns, factor_returns)
            # Build contribution frame
            combined = pd.concat([portfolio_returns, factor_returns], axis=1).dropna()
            y = combined.iloc[:, 0]
            F_df = combined.iloc[:, 1:]
            factor_cols = list(F_df.columns)

            contrib_df = pd.DataFrame(index=combined.index)
            contrib_df["actual_return"] = y

            for fname in factor_cols:
                beta_series = rolling_betas.get(f"beta_{fname}", pd.Series(dtype=float))
                beta_aligned = beta_series.reindex(combined.index, method="ffill")
                contrib_df[f"contrib_{fname}"] = (beta_aligned * F_df[fname]).values

            alpha_series = rolling_betas["alpha"].reindex(combined.index, method="ffill") / 252 \
                if "alpha" in rolling_betas.columns else pd.Series(0.0, index=combined.index)
            contrib_df["alpha_contribution"] = alpha_series.values

            factor_sum = contrib_df[[c for c in contrib_df.columns if c.startswith("contrib_")]].sum(axis=1)
            contrib_df["residual"] = y - factor_sum - contrib_df["alpha_contribution"]

        else:
            ts_reg = TimeSeriesRegression(annualize=False)
            result = ts_reg.fit(portfolio_returns, factor_returns)

            combined = pd.concat([portfolio_returns, factor_returns], axis=1).dropna()
            y = combined.iloc[:, 0]
            F_df = combined.iloc[:, 1:]

            contrib_df = pd.DataFrame(index=combined.index)
            contrib_df["actual_return"] = y
            contrib_df["alpha_contribution"] = result.alpha / 252  # daily alpha

            for fname, beta in result.betas.items():
                if fname in F_df.columns:
                    contrib_df[f"contrib_{fname}"] = beta * F_df[fname]

            factor_sum = contrib_df[[c for c in contrib_df.columns if c.startswith("contrib_")]].sum(axis=1)
            contrib_df["residual"] = y - factor_sum - result.alpha / 252

        return contrib_df

    def attribution_summary(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        rolling: bool = False,
    ) -> pd.DataFrame:
        """
        Summarize annualized contribution of each source.

        Returns
        -------
        pd.DataFrame
            Annualized mean contribution and fraction of total return.
        """
        decomp = self.decompose(portfolio_returns, factor_returns, rolling=rolling)
        total_return = float(decomp["actual_return"].sum() * 252)

        rows = []
        for col in decomp.columns:
            if col == "actual_return":
                continue
            ann_contrib = float(decomp[col].sum() * 252)
            pct_total = ann_contrib / (total_return + 1e-12)
            rows.append({
                "source": col,
                "annualized_contribution": round(ann_contrib, 4),
                "pct_of_total_return": round(pct_total, 4),
            })

        return pd.DataFrame(rows).set_index("source")

    def conditional_alpha(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        condition_series: pd.Series,
        condition_values: Optional[List] = None,
    ) -> pd.DataFrame:
        """
        Compute alpha conditional on different states of condition_series.

        Parameters
        ----------
        portfolio_returns : pd.Series
        factor_returns : pd.DataFrame
        condition_series : pd.Series
            Categorical or discretized series (e.g., VIX regime).
        condition_values : list or None
            Specific values to condition on.  If None, uses unique values.

        Returns
        -------
        pd.DataFrame
            Alpha and Sharpe for each conditional state.
        """
        ts_reg = TimeSeriesRegression()
        if condition_values is None:
            condition_values = sorted(condition_series.dropna().unique().tolist())

        rows = []
        for val in condition_values:
            mask = condition_series == val
            port_subset = portfolio_returns[mask]
            factor_subset = factor_returns[mask]
            combined = pd.concat([port_subset, factor_subset], axis=1).dropna()
            if len(combined) < 20:
                continue
            result = ts_reg.fit(combined.iloc[:, 0], combined.iloc[:, 1:])
            rows.append({
                "condition": val,
                "alpha_annual": result.alpha,
                "alpha_tstat": result.alpha_tstat,
                "sharpe_residual": round(
                    result.residuals.mean() / (result.residuals.std() + 1e-12) * 252 ** 0.5, 4
                ),
                "n_obs": result.n_obs,
                "r_squared": result.r_squared,
            })

        return pd.DataFrame(rows).set_index("condition")
