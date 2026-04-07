"""
research/portfolio_lab/factor_exposure_analyzer.py

Factor exposure analysis and return decomposition for SRFM-Lab.
Supports OLS-based betas, rolling exposures, drift detection, and
per-factor contribution time series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STANDARD_FACTORS = [
    "Market",
    "Size",
    "Value",
    "Momentum",
    "Low_Vol",
    "Quality",
    "Carry",
]


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FactorExposures:
    """OLS factor model results for a single asset / portfolio."""

    betas: Dict[str, float]  # OLS beta per factor
    t_stats: Dict[str, float]  # t-statistic per factor
    r_squared: float
    alpha: float  # intercept (annualised if caller scales data)
    alpha_t_stat: float
    residual_vol: float  # annualised residual standard deviation
    factor_names: List[str] = field(default_factory=list)
    n_obs: int = 0

    def significant_factors(self, threshold: float = 2.0) -> List[str]:
        """Return factors with |t-stat| >= threshold."""
        return [f for f, t in self.t_stats.items() if abs(t) >= threshold]

    def as_series(self) -> pd.Series:
        s: Dict[str, float] = {"alpha": self.alpha, "alpha_t": self.alpha_t_stat}
        for f in self.factor_names:
            s[f"beta_{f}"] = self.betas[f]
            s[f"t_{f}"] = self.t_stats[f]
        s["r_squared"] = self.r_squared
        s["residual_vol"] = self.residual_vol
        return pd.Series(s)


@dataclass
class DriftReport:
    """Rolling factor exposure drift statistics."""

    factor_names: List[str]
    # rolling betas: index = date, columns = factor names
    rolling_betas: pd.DataFrame
    # mean absolute change in beta per factor
    mean_abs_drift: Dict[str, float]
    # dates where beta change > 2 std deviations
    structural_breaks: Dict[str, pd.DatetimeIndex]
    total_drift_score: float  # sum of mean abs drifts


@dataclass
class ExplainedReturn:
    """Return decomposition into systematic + idiosyncratic."""

    total_return: float
    systematic_return: float  # alpha + sum(beta_i * factor_mean_i)
    idiosyncratic_return: float  # residual
    factor_contributions: Dict[str, float]  # beta_i * factor_mean_i
    r_squared: float


# ---------------------------------------------------------------------------
# Synthetic factor generation
# ---------------------------------------------------------------------------


def generate_synthetic_factors(
    n_obs: int = 252,
    seed: int = 42,
    annualise_scale: int = 1,
) -> pd.DataFrame:
    """
    Generate synthetic daily factor returns for testing.

    Returns a DataFrame with columns matching STANDARD_FACTORS.
    All returns are in daily frequency unless annualise_scale is provided.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-02", periods=n_obs, freq="B")

    # approximate daily stats for each factor
    factor_params: Dict[str, Tuple[float, float]] = {
        "Market":   (0.0003, 0.010),   # (daily_mean, daily_vol)
        "Size":     (0.0001, 0.005),
        "Value":    (0.0001, 0.005),
        "Momentum": (0.0002, 0.007),
        "Low_Vol":  (-0.0001, 0.004),
        "Quality":  (0.0001, 0.003),
        "Carry":    (0.0001, 0.003),
    }

    data: Dict[str, np.ndarray] = {}
    for fname, (mu, sigma) in factor_params.items():
        data[fname] = rng.normal(mu * annualise_scale, sigma * np.sqrt(annualise_scale), n_obs)

    return pd.DataFrame(data, index=idx)


# ---------------------------------------------------------------------------
# OLS helper
# ---------------------------------------------------------------------------


def _ols_fit(
    y: np.ndarray,
    X: np.ndarray,
    factor_names: List[str],
    trading_days: int = 252,
) -> FactorExposures:
    """
    Run OLS of y on X (with intercept prepended).
    y: (T,)  X: (T, k)
    Returns FactorExposures with annualised alpha and residual_vol.
    """
    T, k = X.shape
    # add intercept column
    Xc = np.column_stack([np.ones(T), X])

    # normal equations (fast for small k)
    try:
        coef, residuals, rank, sv = np.linalg.lstsq(Xc, y, rcond=None)
    except np.linalg.LinAlgError:
        # fallback: return zeros
        betas = {f: 0.0 for f in factor_names}
        t_stats = {f: 0.0 for f in factor_names}
        return FactorExposures(
            betas=betas,
            t_stats=t_stats,
            r_squared=0.0,
            alpha=0.0,
            alpha_t_stat=0.0,
            residual_vol=float(np.std(y)),
            factor_names=factor_names,
            n_obs=T,
        )

    y_hat = Xc @ coef
    resid = y - y_hat
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 0.0

    # standard errors
    dof = T - k - 1
    if dof <= 0:
        dof = 1
    sigma2 = ss_res / dof
    try:
        cov_coef = sigma2 * np.linalg.inv(Xc.T @ Xc)
        se = np.sqrt(np.maximum(np.diag(cov_coef), 0.0))
    except np.linalg.LinAlgError:
        se = np.ones(k + 1)

    t_vals = coef / (se + 1e-15)

    alpha_daily = float(coef[0])
    alpha_ann = alpha_daily * trading_days
    alpha_t = float(t_vals[0])
    residual_vol = float(np.std(resid) * np.sqrt(trading_days))

    betas: Dict[str, float] = {}
    t_stats: Dict[str, float] = {}
    for i, fname in enumerate(factor_names):
        betas[fname] = float(coef[i + 1])
        t_stats[fname] = float(t_vals[i + 1])

    return FactorExposures(
        betas=betas,
        t_stats=t_stats,
        r_squared=float(r2),
        alpha=alpha_ann,
        alpha_t_stat=alpha_t,
        residual_vol=residual_vol,
        factor_names=factor_names,
        n_obs=T,
    )


# ---------------------------------------------------------------------------
# Main analyser
# ---------------------------------------------------------------------------


class FactorExposureAnalyzer:
    """
    Compute and track factor exposures for a return series.

    Parameters
    ----------
    trading_days : number of trading days per year (used to annualise alpha)
    """

    def __init__(self, trading_days: int = 252) -> None:
        self._trading_days = trading_days

    # ------------------------------------------------------------------
    # Core exposure computation
    # ------------------------------------------------------------------

    def compute_exposures(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame,
    ) -> FactorExposures:
        """
        Compute factor exposures (betas) via OLS.

        Parameters
        ----------
        returns : asset / portfolio daily returns
        factor_returns : DataFrame of factor returns (same frequency)
        """
        aligned = pd.concat([returns, factor_returns], axis=1).dropna()
        if len(aligned) < max(20, factor_returns.shape[1] + 2):
            raise ValueError(
                f"Insufficient observations ({len(aligned)}) for OLS with "
                f"{factor_returns.shape[1]} factors"
            )
        y = aligned.iloc[:, 0].values
        X = aligned.iloc[:, 1:].values
        factor_names = list(factor_returns.columns)
        return _ols_fit(y, X, factor_names, self._trading_days)

    # ------------------------------------------------------------------
    # Rolling exposures
    # ------------------------------------------------------------------

    def rolling_exposures(
        self,
        returns: pd.Series,
        factors: pd.DataFrame,
        window: int = 63,
    ) -> pd.DataFrame:
        """
        Compute rolling OLS factor betas.

        Returns a DataFrame indexed by date with columns for each factor
        beta, plus alpha, r_squared, and residual_vol.
        """
        aligned = pd.concat([returns, factors], axis=1).dropna()
        factor_names = list(factors.columns)
        n = len(aligned)

        records: List[Dict] = []
        dates: List[pd.Timestamp] = []

        for end in range(window, n + 1):
            window_data = aligned.iloc[end - window : end]
            y = window_data.iloc[:, 0].values
            X = window_data.iloc[:, 1:].values
            exp = _ols_fit(y, X, factor_names, self._trading_days)
            row: Dict[str, float] = {"alpha": exp.alpha, "r_squared": exp.r_squared}
            for f in factor_names:
                row[f"beta_{f}"] = exp.betas[f]
            records.append(row)
            dates.append(aligned.index[end - 1])

        if not records:
            return pd.DataFrame()

        df_out = pd.DataFrame(records, index=dates)
        return df_out

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------

    def exposure_drift(
        self,
        returns: pd.Series,
        factors: pd.DataFrame,
        window: int = 63,
    ) -> DriftReport:
        """
        Analyse how factor exposures drift over time.

        Detects structural breaks where rolling beta changes by more than
        2 standard deviations from the rolling change distribution.
        """
        roll = self.rolling_exposures(returns, factors, window)
        if roll.empty:
            raise ValueError("Not enough data to compute rolling exposures")

        factor_names = list(factors.columns)
        beta_cols = [f"beta_{f}" for f in factor_names]

        beta_df = roll[beta_cols].copy()
        beta_df.columns = factor_names

        diff = beta_df.diff().dropna()
        mean_abs_drift: Dict[str, float] = {}
        structural_breaks: Dict[str, pd.DatetimeIndex] = {}

        for f in factor_names:
            d = diff[f]
            mean_abs_drift[f] = float(d.abs().mean())
            mu_d = d.mean()
            sigma_d = d.std()
            if sigma_d > 1e-12:
                breaks = d[np.abs(d - mu_d) > 2 * sigma_d].index
            else:
                breaks = pd.DatetimeIndex([])
            structural_breaks[f] = breaks

        total_drift_score = float(sum(mean_abs_drift.values()))

        return DriftReport(
            factor_names=factor_names,
            rolling_betas=beta_df,
            mean_abs_drift=mean_abs_drift,
            structural_breaks=structural_breaks,
            total_drift_score=total_drift_score,
        )

    # ------------------------------------------------------------------
    # Return explanation
    # ------------------------------------------------------------------

    def explain_return(
        self,
        returns: pd.Series,
        factors: pd.DataFrame,
    ) -> ExplainedReturn:
        """
        Decompose total return into factor and idiosyncratic components.

        Uses full-sample OLS betas and mean factor returns to attribute
        the period average return.
        """
        exp = self.compute_exposures(returns, factors)
        aligned = pd.concat([returns, factors], axis=1).dropna()
        factor_means = aligned.iloc[:, 1:].mean().values * self._trading_days

        factor_contributions: Dict[str, float] = {}
        systematic = exp.alpha
        for i, f in enumerate(exp.factor_names):
            contrib = exp.betas[f] * float(factor_means[i])
            factor_contributions[f] = contrib
            systematic += contrib

        total_ret = float(returns.mean() * self._trading_days)
        idiosyncratic = total_ret - systematic

        return ExplainedReturn(
            total_return=total_ret,
            systematic_return=systematic,
            idiosyncratic_return=idiosyncratic,
            factor_contributions=factor_contributions,
            r_squared=exp.r_squared,
        )


# ---------------------------------------------------------------------------
# Return decomposer
# ---------------------------------------------------------------------------


class ReturnDecomposer:
    """
    Decompose realised returns into per-factor contribution time series.

    For each date t:
        contribution_i(t) = beta_i * factor_return_i(t)
        specific_return(t) = asset_return(t) - alpha_daily - sum_i(contribution_i(t))
    """

    def __init__(self, trading_days: int = 252) -> None:
        self._trading_days = trading_days

    def decompose(
        self,
        returns: pd.Series,
        factors: pd.DataFrame,
    ) -> Dict[str, pd.Series]:
        """
        Decompose returns into per-factor contribution and specific return.

        Returns a dict of pd.Series (same index as aligned input), with keys:
        - one entry per factor name
        - "specific" for the residual
        - "alpha_daily" for the constant contribution
        """
        aligned = pd.concat([returns, factors], axis=1).dropna()
        asset_ret = aligned.iloc[:, 0]
        factor_ret = aligned.iloc[:, 1:]
        factor_names = list(factor_ret.columns)

        analyzer = FactorExposureAnalyzer(self._trading_days)
        exp = analyzer.compute_exposures(returns, factors)

        # daily alpha contribution
        alpha_daily = exp.alpha / self._trading_days

        contributions: Dict[str, pd.Series] = {}
        contributions["alpha_daily"] = pd.Series(
            np.full(len(aligned), alpha_daily), index=aligned.index
        )

        total_systematic = pd.Series(
            np.full(len(aligned), alpha_daily), index=aligned.index
        )

        for f in factor_names:
            contrib = exp.betas[f] * factor_ret[f]
            contributions[f] = contrib
            total_systematic = total_systematic + contrib

        contributions["specific"] = asset_ret - total_systematic

        return contributions

    def cumulative_decomposition(
        self,
        returns: pd.Series,
        factors: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Return cumulative compounded contribution of each component.

        Compounds daily contributions: prod(1 + r_t) - 1
        """
        daily = self.decompose(returns, factors)
        cum: Dict[str, pd.Series] = {}
        for k, s in daily.items():
            cum[k] = (1.0 + s).cumprod() - 1.0
        return pd.DataFrame(cum)
